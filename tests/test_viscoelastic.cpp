// Finite viscoelasticity in the solid formulations: the quadrature-point
// history behind the kernels (step length from SetLoadFactor, the update in
// AcceptStep, the accepted state between steps, a rejected step, ResetHistory)
// checked against the material point through a homogeneous relaxation test in
// both formulations; the assembled Jacobian with a history and through the
// update against finite differences; the rigid-sphere penalty contact
// (resultant against its closed form, tangent, equilibrium with the support
// reaction); the dynamics decorator advancing the wrapped problem's history;
// and the input schema (branches, the time block, contact entries).
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "base/coefficients.hpp"
#include "base/mesh_input.hpp"
#include "base/probes.hpp"
#include "materials/materials.hpp"
#include "mfem.hpp"
#include "physics/dynamic_solid_problem.hpp"
#include "physics/solid_problem.hpp"
#include "solvers/direct_solver.hpp"
#include "solvers/quasi_static.hpp"
#include "test_util.hpp"

using cmf::tensor;

namespace
{

using Mat3 = tensor<double, 3, 3>;
using V = cmf::Viscoelastic<cmf::IsoNeoHookean>;

const double kMu = 1.0, kKappa = 100.0;
const std::vector<cmf::MaxwellBranchConfig> kBranches = {{0.5, 0.2}, {0.25, 2.0}};

// Unit cube of n^3 hexahedra of the given order, iso_neo_hookean with two
// Maxwell branches, no YAML loads; boundary attributes of MFEM's Cartesian
// box: bottom z=0 1, front y=0 2, right x=1 3, back y=1 4, left x=0 5, top 6.
cmf::AppConfig BaseConfig(const std::string &formulation, int n, int order)
{
  cmf::AppConfig cfg;
  cfg.formulation = formulation;
  cfg.mesh.cartesian = true;
  cfg.mesh.box.dim = 3;
  cfg.mesh.box.element = "hex";
  cfg.mesh.box.nx = cfg.mesh.box.ny = cfg.mesh.box.nz = n;
  cfg.mesh.order = order;
  cfg.material.model = "iso_neo_hookean";
  cfg.material.mu = kMu;
  cfg.material.kappa = kKappa;
  cfg.material.branches = kBranches;
  cfg.solver.newton.rtol = 1e-11;
  cfg.solver.newton.atol = 1e-14;
  cfg.solver.newton.max_it = 30;
  cfg.solver.newton.print_level = 0;
  // Sparse LU: the displacement formulation at kappa = 100 mu stalls the AMG;
  // the tangent predictor spreads a Dirichlet increment through the body
  // before Newton starts (the penalty form struggles without it).
  cfg.solver.linear.type = "direct";
  cfg.solver.predictor = "tangent";
  return cfg;
}

V MaterialOf(const cmf::AppConfig &cfg)
{
  return std::get<V>(cmf::MakeMixedMaterial(cfg.material));
}

std::vector<double> InitialHistory(const V &m)
{
  std::vector<double> h(std::size_t(m.HistorySize()));
  m.InitialHistory(h.data());
  return h;
}

double MaxAbs(const Mat3 &A)
{
  double v = 0.0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) { v = std::max(v, std::abs(A(i, j))); }
  return v;
}

const cmf::Reaction &Named(const std::vector<cmf::Reaction> &r, const std::string &name)
{
  for (const cmf::Reaction &x : r) { if (x.name == name) { return x; } }
  MFEM_ABORT("no reaction named " << name);
  return r[0];
}

// Uniaxial tension of the cube held at the stretch 1.15 from the first step on:
// rollers on x = 0, y = 0, z = 0, u_y = 0.15 on y = 1 (the loaded face; its
// lateral faces are free, so the reaction per unit area is P_22 exactly). The
// state is homogeneous, F = diag(1 + u_x, 1.15, 1 + u_z) with the lateral
// stretch of the solution, so the material point evaluated along the sequence
// of F with the manual history update predicts every reaction; the history,
// dt and acceptance of the physics must reproduce it.
void RelaxationTest(const std::string &formulation)
{
  std::printf("relaxation under a held stretch, %s formulation\n", formulation.c_str());
  cmf::AppConfig cfg = BaseConfig(formulation, 2, 2);
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
  cmf::SolidProblem &physics = *problem;
  CHECK(physics.HasHistory());
  mfem::Vector zero(3), pull(3);
  zero = 0.0;
  pull = 0.0;
  pull(1) = 0.15;
  mfem::VectorConstantCoefficient zero_coef(zero), pull_coef(pull);
  cmf::BCOptions roller_x, roller_y, roller_z, loaded;
  roller_x.components = {0};
  roller_y.components = {1};
  roller_z.components = {2};
  roller_x.schedule = roller_y.schedule = roller_z.schedule = cmf::Schedule::Constant();
  loaded.components = {1};
  loaded.schedule = cmf::Schedule::Constant();
  loaded.name = "loaded";
  physics.AddDirichlet({5}, zero_coef, roller_x);
  physics.AddDirichlet({2}, zero_coef, roller_y);
  physics.AddDirichlet({1}, zero_coef, roller_z);
  physics.AddDirichlet({4}, pull_coef, loaded);
  physics.Finalize();
  physics.SetPhysicalTime(true);
  std::unique_ptr<mfem::Solver> linear = physics.MakeLinearSolver(cfg.solver.linear);
  mfem::Vector x(physics.Height());
  x = 0.0;

  const V m = MaterialOf(cfg);
  std::vector<double> h = InitialHistory(m), hn(h);
  const double dt = 0.1;
  std::vector<double> times;
  for (int n = 1; n <= 6; n++) { times.push_back(dt * n); }
  double worst = 0.0, first_reaction = 0.0, last_reaction = 0.0;
  Mat3 F_last;
  const cmf::LoadStepCallback on_step = [&](const cmf::LoadStepReport &step, const mfem::Vector &xs)
  {
    physics.UpdateFields(xs);
    const std::vector<double> u = cmf::ProbeVector(physics.Displacement(), {1.0, 1.0, 1.0});
    Mat3 F = cmf::I<3>();
    for (int i = 0; i < 3; i++) { F(i, i) += u[std::size_t(i)]; }
    const Mat3 P = m.PK1(F, h.data(), dt);             // the step's stress from the accepted history
    const cmf::Reaction &rx = Named(physics.Reactions(xs), "loaded");   // at the accepted state: dt = 0
    const double rel = std::abs(rx.force[1] - P(1, 1)) / std::abs(P(1, 1));
    worst = std::max(worst, rel);
    std::printf("  step %d t = %.2f: lateral stretch %.6f, P22 %.10e, reaction %.10e, rel %.1e\n",
                step.step, step.load_factor, F(0, 0), P(1, 1), rx.force[1], rel);
    if (step.step == 1) { first_reaction = rx.force[1]; }
    last_reaction = rx.force[1];
    F_last = F;
    m.Update(F, h.data(), dt, hn.data());
    h = hn;
  };
  cmf::QuasiStaticReport report = cmf::SolveInTime(physics, *linear, cfg.solver, times, 0.0, x, on_step);
  CHECK(report.converged);
  CHECK(int(report.steps.size()) == 6);
  CHECK_MSG(worst <= 1e-9, formulation + ": reactions vs the material point along the history, rel " + std::to_string(worst));
  // The stress relaxes: the first reaction is the largest.
  CHECK(last_reaction < 0.9 * first_reaction);
  // A rejected step: the step length follows SetLoadFactor and the history stays accepted.
  physics.SetLoadFactor(0.9);
  {
    const Mat3 P = m.PK1(F_last, h.data(), 0.3);
    const cmf::Reaction &rx = Named(physics.Reactions(x), "loaded");
    CHECK_CLOSE(rx.force[1], P(1, 1), 1e-9 * std::abs(P(1, 1)));
  }
  physics.SetLoadFactor(0.7);
  {
    const Mat3 P = m.PK1(F_last, h.data(), 0.1);
    const cmf::Reaction &rx = Named(physics.Reactions(x), "loaded");
    CHECK_CLOSE(rx.force[1], P(1, 1), 1e-9 * std::abs(P(1, 1)));
  }
  // ResetHistory: the first step again from the rest state.
  physics.ResetHistory(0.0);
  x = 0.0;
  physics.SetLoadFactor(dt);
  physics.ApplyDirichlet(x);
  cmf::DampedNewtonSolve(physics, *linear, x, cfg.solver.newton, physics.Comm());
  physics.AcceptStep(x);
  const cmf::Reaction &again = Named(physics.Reactions(x), "loaded");
  CHECK_CLOSE(again.force[1], first_reaction, 1e-9 * std::abs(first_reaction));
}

// The assembled Jacobian against central differences of the residual, at a
// state with a history and a step under way (the tangent includes dCv/dF).
void JacobianTest(const std::string &formulation)
{
  std::printf("Jacobian with a history, %s formulation\n", formulation.c_str());
  cmf::AppConfig cfg = BaseConfig(formulation, 2, 2);
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
  cmf::SolidProblem &physics = *problem;
  mfem::Vector value(3);
  value = 0.0;
  mfem::DenseMatrix H(3);
  H(0, 0) = 0.03; H(0, 1) = -0.08; H(0, 2) = 0.02;
  H(1, 0) = -0.08; H(1, 1) = 0.01; H(1, 2) = 0.07;
  H(2, 0) = 0.02; H(2, 1) = 0.07; H(2, 2) = -0.04;
  cmf::AffineVectorCoefficient affine(value, H);
  cmf::BCOptions all;
  all.schedule = cmf::Schedule::Constant();
  physics.AddDirichlet({1, 2, 3, 4, 5, 6}, affine, all);
  physics.Finalize();
  physics.SetPhysicalTime(true);
  std::unique_ptr<mfem::Solver> linear = physics.MakeLinearSolver(cfg.solver.linear);
  mfem::Vector x(physics.Height());
  x = 0.0;
  // Two accepted steps build a history, then a third step is under way.
  cmf::QuasiStaticReport report = cmf::SolveInTime(physics, *linear, cfg.solver, {0.1, 0.2}, 0.0, x);
  CHECK(report.converged);
  physics.SetLoadFactor(0.35);
  // A perturbed state (free dofs only; the essential rows are zeroed by Mult).
  std::mt19937 rng(3u);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  const mfem::Array<int> &ess = physics.EssentialTrueDofs();
  mfem::Array<int> is_ess(x.Size());
  is_ess = 0;
  for (int i = 0; i < ess.Size(); i++) { is_ess[ess[i]] = 1; }
  mfem::Vector v(x.Size());
  for (int i = 0; i < x.Size(); i++)
  {
    if (!is_ess[i]) { x(i) += 0.01 * unit(rng); }
    v(i) = is_ess[i] ? 0.0 : unit(rng);
  }
  mfem::Operator &J = physics.GetGradient(x);
  mfem::Vector Jv(x.Size()), rp(x.Size()), rm(x.Size()), xp(x), xm(x);
  J.Mult(v, Jv);
  const double eps = 1e-6;
  xp.Add(eps, v);
  xm.Add(-eps, v);
  physics.Mult(xp, rp);
  physics.Mult(xm, rm);
  rp -= rm;
  rp /= 2.0 * eps;
  rp -= Jv;
  const double err = std::sqrt(mfem::InnerProduct(physics.Comm(), rp, rp)) /
                     std::sqrt(mfem::InnerProduct(physics.Comm(), Jv, Jv));
  std::printf("  |J v - FD| / |J v| = %.2e\n", err);
  CHECK_MSG(err <= 1e-6, formulation + ": Jacobian vs finite differences with a history");
}

// Rigid-sphere penalty contact: the resultant on the undeformed top face of
// the unit cube, wholly inside a sphere of radius 10 whose centre sits d
// above the face on its axis (g = r^2 - (r - d)^2 - rho^2 > 0 everywhere),
// against the closed form F_z = -2 k (r - d) [A (r^2 - (r - d)^2) - int rho^2 dA]
// with int rho^2 dA = 1/6, M_x = F_z / 2, M_y = -F_z / 2; the tangent against
// finite differences; and, after a solve with the sphere pushed in, the
// contact resultant against the reaction of the supported bottom face.
void ContactTest(const std::string &formulation)
{
  std::printf("rigid-sphere contact, %s formulation\n", formulation.c_str());
  cmf::AppConfig cfg = BaseConfig(formulation, 2, 2);
  cfg.material.branches.clear(); // hyperelastic here
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
  cmf::SolidProblem &physics = *problem;
  const double r = 10.0, d = 0.1, k = 3.0;
  mfem::Vector zero(3), c(3);
  zero = 0.0;
  c(0) = 0.5; c(1) = 0.5; c(2) = 1.0 + r - d;
  mfem::VectorConstantCoefficient zero_coef(zero), center(c);
  cmf::BCOptions roller_x, roller_y, bottom, contact;
  roller_x.components = {0};
  roller_y.components = {1};
  bottom.components = {2};
  bottom.name = "bottom";
  contact.name = "sphere";
  physics.AddDirichlet({5}, zero_coef, roller_x);
  physics.AddDirichlet({2}, zero_coef, roller_y);
  physics.AddDirichlet({1}, zero_coef, bottom);
  physics.AddRigidSphereContact({6}, center, r, k, contact);
  physics.Finalize();
  physics.SetLoadFactor(1.0);
  mfem::Vector x(physics.Height());
  x = 0.0;
  {
    const cmf::Reaction &rx = Named(physics.Reactions(x), "sphere");
    const double Fz = -2.0 * k * (r - d) * ((r * r - (r - d) * (r - d)) - 1.0 / 6.0);
    std::printf("  undeformed face: F = (%.3e, %.3e, %.6e) closed form %.6e; M = (%.6e, %.6e, %.3e)\n",
                rx.force[0], rx.force[1], rx.force[2], Fz, rx.moment[0], rx.moment[1], rx.moment[2]);
    CHECK_CLOSE(rx.force[2], Fz, 1e-12 * std::abs(Fz));
    CHECK(std::abs(rx.force[0]) <= 1e-12 * std::abs(Fz) && std::abs(rx.force[1]) <= 1e-12 * std::abs(Fz));
    CHECK_CLOSE(rx.moment[0], 0.5 * Fz, 1e-12 * std::abs(Fz));
    CHECK_CLOSE(rx.moment[1], -0.5 * Fz, 1e-12 * std::abs(Fz));
    CHECK(std::abs(rx.moment[2]) <= 1e-12 * std::abs(Fz));
  }
  // Tangent of the contact term (the face stays inside the sphere).
  {
    std::mt19937 rng(11u);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    const mfem::Array<int> &ess = physics.EssentialTrueDofs();
    mfem::Array<int> is_ess(x.Size());
    is_ess = 0;
    for (int i = 0; i < ess.Size(); i++) { is_ess[ess[i]] = 1; }
    mfem::Vector v(x.Size()), xs(x);
    for (int i = 0; i < x.Size(); i++)
    {
      if (!is_ess[i]) { xs(i) += 0.01 * unit(rng); }
      v(i) = is_ess[i] ? 0.0 : unit(rng);
    }
    mfem::Operator &J = physics.GetGradient(xs);
    mfem::Vector Jv(x.Size()), rp(x.Size()), rm(x.Size()), xp(xs), xm(xs);
    J.Mult(v, Jv);
    const double eps = 1e-6;
    xp.Add(eps, v);
    xm.Add(-eps, v);
    physics.Mult(xp, rp);
    physics.Mult(xm, rm);
    rp -= rm;
    rp /= 2.0 * eps;
    rp -= Jv;
    const double err = std::sqrt(mfem::InnerProduct(physics.Comm(), rp, rp)) /
                       std::sqrt(mfem::InnerProduct(physics.Comm(), Jv, Jv));
    std::printf("  |J v - FD| / |J v| = %.2e with the contact term\n", err);
    CHECK_MSG(err <= 1e-6, formulation + ": Jacobian vs finite differences with the contact term");
  }
  // Solve: the sphere presses the cube onto its supported bottom face.
  std::unique_ptr<mfem::Solver> linear = physics.MakeLinearSolver(cfg.solver.linear);
  x = 0.0;
  cfg.solver.load_steps = 8;
  cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(physics, *linear, cfg.solver, x);
  CHECK(report.converged);
  physics.UpdateFields(x);
  const std::vector<double> u = cmf::ProbeVector(physics.Displacement(), {0.5, 0.5, 1.0});
  const std::vector<cmf::Reaction> reactions = physics.Reactions(x);
  const cmf::Reaction &sphere = Named(reactions, "sphere");
  const cmf::Reaction &support = Named(reactions, "bottom");
  std::printf("  pressed: top centre u_z %.4e, contact F_z %.6e, support reaction %.6e\n",
              u[2], sphere.force[2], support.force[2]);
  CHECK(u[2] < -1e-3);
  CHECK(sphere.force[2] < 0.0);
  CHECK_CLOSE(sphere.force[2] + support.force[2], 0.0, 1e-9 * std::abs(sphere.force[2]));
}

// The dynamics decorator advances the wrapped problem's history: a single
// trilinear element (every node on the boundary) under an affine motion
// prescribed on all faces, three steps with inertia, against the same steps of
// the quasi-static problem in time; both static residuals at the accepted
// state must agree.
void DynamicHistoryTest()
{
  std::printf("history through the dynamics decorator\n");
  cmf::AppConfig cfg = BaseConfig("displacement", 1, 1);
  cfg.material.rho0 = 2.0;
  cfg.dynamics.enabled = true;
  cfg.dynamics.t_final = 0.3;
  cfg.dynamics.breakpoints = {0.1, 0.2, 0.3};
  mfem::Vector value(3);
  value = 0.0;
  mfem::DenseMatrix H(3);
  H(0, 0) = 0.03; H(0, 1) = -0.08; H(0, 2) = 0.02;
  H(1, 0) = -0.08; H(1, 1) = 0.01; H(1, 2) = 0.07;
  H(2, 0) = 0.02; H(2, 1) = 0.07; H(2, 2) = -0.04;
  // u = t H X: the strain grows with the time, so each step's history differs.
  mfem::VectorFunctionCoefficient affine(3, [&H](const mfem::Vector &X, double t, mfem::Vector &u)
  {
    u.SetSize(3);
    H.Mult(X, u);
    u *= t;
  });
  cmf::BCOptions all;
  all.schedule = cmf::Schedule::Constant();
  all.time_dependent = true;

  std::unique_ptr<mfem::ParMesh> mesh_d = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem_d = cmf::MakeSolidProblem(*mesh_d, cfg);
  problem_d->AddDirichlet({1, 2, 3, 4, 5, 6}, affine, all);
  problem_d->Finalize();
  cmf::DynamicSolidProblem dynamic(*problem_d, cfg.dynamics);
  std::unique_ptr<mfem::Solver> linear_d = dynamic.MakeLinearSolver(cfg.solver.linear);
  mfem::Vector xd(problem_d->Height());
  xd = 0.0;
  dynamic.Initialize(xd);
  cmf::QuasiStaticReport rd = cmf::SolveDynamic(dynamic, *linear_d, cfg.solver, cfg.dynamics.breakpoints, 0.0, xd);
  CHECK(rd.converged);

  cmf::AppConfig cfg_s = cfg;
  cfg_s.dynamics = cmf::DynamicsConfig();
  std::unique_ptr<mfem::ParMesh> mesh_s = cmf::BuildParMesh(MPI_COMM_WORLD, cfg_s.mesh);
  std::unique_ptr<cmf::SolidProblem> problem_s = cmf::MakeSolidProblem(*mesh_s, cfg_s);
  problem_s->AddDirichlet({1, 2, 3, 4, 5, 6}, affine, all);
  problem_s->Finalize();
  problem_s->SetPhysicalTime(true);
  std::unique_ptr<mfem::Solver> linear_s = problem_s->MakeLinearSolver(cfg_s.solver.linear);
  mfem::Vector xs(problem_s->Height());
  xs = 0.0;
  cmf::QuasiStaticReport rs = cmf::SolveInTime(*problem_s, *linear_s, cfg_s.solver, cfg.dynamics.breakpoints, 0.0, xs);
  CHECK(rs.converged);

  mfem::Vector r_d, r_s, r_rest;
  problem_d->FullResidual(xd, r_d);
  problem_s->FullResidual(xs, r_s);
  // The same state (every dof prescribed) and the same history: the same static residual;
  // a rest history (Cv = I) gives another one.
  mfem::Vector diff(r_d);
  diff -= r_s;
  mfem::Vector dx(xd);
  dx -= xs;
  problem_s->ResetHistory(0.3);
  problem_s->FullResidual(xs, r_rest);
  r_rest -= r_s;
  std::printf("  |x_dyn - x_static| = %.2e, |R_dyn - R_static| / |R| = %.2e, |R(rest history) - R| / |R| = %.2e\n",
              dx.Normlinf(), diff.Norml2() / r_s.Norml2(), r_rest.Norml2() / r_s.Norml2());
  CHECK(dx.Normlinf() <= 1e-12);
  CHECK_MSG(diff.Norml2() <= 1e-10 * r_s.Norml2(), "the decorator advances the wrapped history");
  CHECK_MSG(r_rest.Norml2() >= 1e-3 * r_s.Norml2(), "the history changed the stress");
}

void ConfigTest()
{
  std::printf("input schema\n");
  const std::string head = R"(
mesh: { file: apps/mesh/cube.msh, order: 2 }
material: { model: arruda_boyce, mu: 15.36, N: 34.2225, kappa: 15360.0,
            branches: [ { G: 26.06, tau: 0.6074 }, { G: 26.53, tau: 6.56 } ] }
bcs:
  dirichlet: [ { attr: [back], expression: ["0", "1", "0"], schedule: { type: ramp, from: 0.0, to: 5.0 } } ]
  contact: [ { attr: [top], name: ball, radius: 2.0, penalty: 10.0, center: ["0", "0", "3 - t"] } ]
)";
  {
    const cmf::AppConfig cfg = cmf::ParseConfig(YAML::Load(head + "time: { t_final: 10.0, dt: 0.5 }\n"));
    CHECK(cfg.time.enabled && !cfg.dynamics.enabled);
    CHECK(int(cfg.time.breakpoints.size()) == 20);
    CHECK_CLOSE(cfg.time.breakpoints.back(), 10.0, 1e-15);
    CHECK(cfg.material.branches.size() == 2);
    CHECK_CLOSE(cfg.material.branches[1].tau, 6.56, 1e-15);
    CHECK(cfg.bcs.contact.size() == 1 && cfg.bcs.contact[0].type == "rigid_sphere");
    CHECK(cfg.bcs.contact[0].name == "ball" && cfg.bcs.contact[0].center.size() == 3);
    CHECK(cfg.bcs.contact[0].schedule.kind == cmf::Schedule::Kind::Constant);
    // the ramp of the entry is in physical time
    CHECK_CLOSE(cfg.bcs.dirichlet[0].schedule.Eval(2.5), 0.5, 1e-15);
    CHECK_CLOSE(cfg.solver.substep.min_dt, 5e-4, 1e-15);
  }
  {
    const cmf::AppConfig cfg = cmf::ParseConfig(YAML::Load(head + "time: { t_final: 460.0, steps: [ { to: 60.0, n: 24 }, { to: 460.0, n: 10 } ] }\n"));
    CHECK(int(cfg.time.breakpoints.size()) == 34);
    CHECK_CLOSE(cfg.time.breakpoints[23], 60.0, 1e-12);
  }
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head)), cmf::ConfigError, "time block");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head + "time: { t_final: 1.0, dt: 0.5 }\ndynamics: { t_final: 1.0, dt: 0.5 }\n")),
               cmf::ConfigError, "one, not both");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head + "time: { t_final: 10.0, dt: 0.5 }\nsolver: { load_steps: 4 }\n")),
               cmf::ConfigError, "load_steps");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head + "time: { t_final: 1.0 }\n")), cmf::ConfigError, "exactly one");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(R"(
mesh: { file: apps/mesh/cube.msh, order: 2 }
material: { model: neo_hookean, E: 1.0, nu: 0.3, branches: [ { G: 1.0, tau: 1.0 } ] }
time: { t_final: 1.0, dt: 0.5 }
)")), cmf::ConfigError, "not used by model");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(R"(
mesh: { file: apps/mesh/cube.msh, order: 2 }
material: { model: iso_neo_hookean, mu: 1.0, kappa: 10.0, branches: [ { G: 1.0, tau: -1.0 } ] }
time: { t_final: 1.0, dt: 0.5 }
)")), cmf::ConfigError, "tau");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(R"(
mesh: { file: apps/mesh/cube.msh, order: 2 }
material: { model: iso_neo_hookean, mu: 1.0, kappa: 10.0 }
bcs: { contact: [ { attr: [top], radius: 2.0, penalty: 10.0 } ] }
)")), cmf::ConfigError, "center");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(R"(
mesh: { file: apps/mesh/cube.msh, order: 2 }
material: { model: iso_neo_hookean, mu: 1.0, kappa: 10.0 }
bcs: { contact: [ { attr: [top], type: plane, radius: 2.0, penalty: 10.0, center: ["0", "0", "3"] } ] }
)")), cmf::ConfigError, "rigid_sphere");
  // A region with its own branches; the mixed material of the input.
  {
    const cmf::AppConfig cfg = cmf::ParseConfig(YAML::Load(R"(
mesh: { file: apps/mesh/cube.msh, order: 2 }
material: { model: iso_neo_hookean, mu: 1.0, kappa: 10.0, branches: [ { G: 1.0, tau: 1.0 } ],
            regions: [ { attr: [domain], mu: 2.0, branches: [ { G: 2.0, tau: 2.0 }, { G: 3.0, tau: 3.0 } ] } ] }
time: { t_final: 1.0, dt: 0.5 }
)"));
    CHECK(cfg.material.regions.size() == 1 && cfg.material.regions[0].branches.size() == 2);
    CHECK(cmf::MaterialName(cmf::MakeMixedMaterial(cfg.material.regions[0])) == "iso_neo_hookean with 2 Maxwell branches");
  }
}

} // namespace

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  const cmf::PetscSession petsc; // solver.linear.type: direct
  ConfigTest();
  RelaxationTest("displacement");
  RelaxationTest("mixed");
  JacobianTest("displacement");
  JacobianTest("mixed");
  ContactTest("displacement");
  ContactTest("mixed");
  DynamicHistoryTest();
  return cmf_test::Report("test_viscoelastic");
}

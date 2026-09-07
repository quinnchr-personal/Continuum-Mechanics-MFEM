// Boundary conditions and load scheduling (plan doc/bc_loading_plan.md):
// L1 schedules and step control, L2 expression data, L3 component-wise
// Dirichlet data, L4 pressure tractions (dead and follower).
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <random>

#include "base/coefficients.hpp"
#include "base/fields.hpp"
#include "base/mesh_input.hpp"
#include "base/probes.hpp"
#include "kernels/total_lagrangian.hpp"
#include "materials/materials.hpp"
#include "mfem.hpp"
#include "physics/mixed_solid_mechanics_tl.hpp"
#include "physics/solid_mechanics_tl.hpp"
#include "physics/solid_problem.hpp"
#include "solvers/quasi_static.hpp"
#include "test_util.hpp"

using cmf::tensor;

namespace
{

// Cook's membrane, compressible neo-Hookean, displacement formulation, with
// the YAML loads removed so each test installs its own.
cmf::AppConfig CookConfig()
{
  cmf::AppConfig cfg = cmf::LoadConfig("apps/input/cook.yaml");
  cfg.output.paraview.clear();
  cfg.output.fields = {"displacement"};
  cfg.output.probes.clear();
  cfg.solver.newton.print_level = 0;
  cfg.solver.newton.rtol = 1e-10;
  cfg.solver.newton.atol = 1e-12;
  cfg.solver.linear.rtol = 1e-13;
  cfg.bcs.traction.clear();
  cfg.body_force = cmf::BodyForceConfig();
  return cfg;
}

struct Run
{
  cmf::QuasiStaticReport report;
  mfem::Vector u;
  double u_norm = 0.0;
  std::vector<double> corner;
};

Run SolveCook(const cmf::AppConfig &cfg, mfem::ParMesh &mesh,
              const std::function<void(cmf::SolidProblem &)> &install)
{
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(mesh, cfg);
  install(*problem);
  problem->Finalize();
  std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
  Run r;
  r.u.SetSize(problem->Height());
  r.u = 0.0;
  r.report = cmf::SolveQuasiStatic(*problem, *linear, cfg.solver, r.u);
  problem->UpdateFields(r.u);
  r.u_norm = std::sqrt(mfem::InnerProduct(MPI_COMM_WORLD, r.u, r.u));
  r.corner = cmf::ProbeVector(problem->Displacement(), {48.0, 60.0});
  return r;
}

// Staged loading: a traction ramped over [0, 0.5] and held, a second one
// ramped over [0.5, 1]; the final state equals the single-stage solve with
// both (path independence of hyperelasticity), and the intermediate state
// at t = 0.5 equals the first traction alone.
void StagedLoadingTest()
{
  cmf::AppConfig cfg = CookConfig();
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  mfem::Vector t1(2), t2(2), tsum(2);
  t1(0) = 0.0; t1(1) = 2.0;
  t2(0) = 0.0; t2(1) = 1.75;
  tsum = t1; tsum += t2;
  mfem::VectorConstantCoefficient c1(t1), c2(t2), csum(tsum);

  cfg.solver.load_steps = 1;
  const Run single = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p) { p.AddTraction({2}, csum); });
  const Run first = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p) { p.AddTraction({2}, c1); });

  cfg.solver.load_steps = 4;
  std::vector<mfem::Vector> states;
  cmf::BCOptions early, late;
  early.schedule = cmf::Schedule::Ramp(0.0, 0.5);
  late.schedule = cmf::Schedule::Ramp(0.5, 1.0);
  Run staged;
  {
    std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
    problem->AddTraction({2}, c1, early);
    problem->AddTraction({2}, c2, late);
    problem->Finalize();
    std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
    staged.u.SetSize(problem->Height());
    staged.u = 0.0;
    staged.report = cmf::SolveQuasiStatic(*problem, *linear, cfg.solver, staged.u,
      [&](const cmf::LoadStepReport &, const mfem::Vector &x) { states.push_back(x); });
    // Scheduled external load: the held first traction plus half of the second at t = 0.75.
    problem->SetLoadFactor(0.75);
    mfem::Vector want(problem->Loads().ExternalLoad().Size());
    {
      std::unique_ptr<cmf::SolidProblem> ref = cmf::MakeSolidProblem(*mesh, cfg);
      mfem::Vector t75(2);
      t75 = t1; t75.Add(0.5, t2);
      mfem::VectorConstantCoefficient c75(t75);
      ref->AddTraction({2}, c75);
      ref->Finalize();
      ref->SetLoadFactor(1.0);
      want = ref->Loads().ExternalLoad();
    }
    want -= problem->Loads().ExternalLoad();
    CHECK_MSG(want.Normlinf() <= 1e-14, "scheduled load at t = 0.75 is T1 + T2/2");
  }
  CHECK_MSG(staged.report.converged && staged.report.steps.size() == 4, "staged run converged in 4 steps");
  CHECK(states.size() == 4);
  auto diff = [](const mfem::Vector &a, const mfem::Vector &b)
  {
    mfem::Vector d(a);
    d -= b;
    return d.Normlinf() / std::max(1e-300, a.Normlinf());
  };
  const double d_final = diff(single.u, staged.u);
  const double d_half = states.size() >= 2 ? diff(first.u, states[1]) : 1.0;
  std::printf("  staged loading: |u_single - u_staged| / |u| = %.3e, at t = 0.5 vs first alone %.3e\n",
              d_final, d_half);
  CHECK_MSG(d_final <= 1e-9, "staged final state matches the single-stage solve");
  CHECK_MSG(d_half <= 1e-9, "state at t = 0.5 matches the first traction alone");
}

// Load-unload through a table schedule: the state returns to zero.
void LoadUnloadTest()
{
  cmf::AppConfig cfg = CookConfig();
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  mfem::Vector t(2);
  t(0) = 0.0; t(1) = 3.75;
  mfem::VectorConstantCoefficient c(t);
  cmf::BCOptions opt;
  opt.schedule = cmf::Schedule::Table({0.0, 0.5, 1.0}, {0.0, 1.0, 0.0});
  cfg.solver.load_steps = 4;
  double u_peak = 0.0;
  Run r;
  {
    std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
    problem->AddTraction({2}, c, opt);
    problem->Finalize();
    std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
    r.u.SetSize(problem->Height());
    r.u = 0.0;
    r.report = cmf::SolveQuasiStatic(*problem, *linear, cfg.solver, r.u,
      [&](const cmf::LoadStepReport &s, const mfem::Vector &x)
      {
        if (s.step == 2) { u_peak = x.Normlinf(); }
      });
  }
  std::printf("  load-unload: |u| at peak %.4e, at the end %.3e\n", u_peak, r.u.Normlinf());
  CHECK_MSG(r.report.converged, "load-unload converged");
  CHECK_MSG(u_peak > 1.0, "the membrane deformed at the peak");
  CHECK_MSG(r.u.Normlinf() <= 1e-9 * u_peak, "the state returns to zero after unloading");
}

// Bisection: with a Newton iteration cap the single planned step fails and
// the stepper recovers by halving; without substepping it reports failure.
void BisectionTest()
{
  cmf::AppConfig cfg = CookConfig();
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  mfem::Vector t(2);
  t(0) = 0.0; t(1) = 3.75;
  mfem::VectorConstantCoefficient c(t);
  cfg.solver.load_steps = 1;
  const Run reference = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p) { p.AddTraction({2}, c); });
  const int full_its = reference.report.steps.back().newton.iterations;
  cfg.solver.newton.max_it = std::max(2, full_its - 2);
  const Run failed = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p) { p.AddTraction({2}, c); });
  CHECK_MSG(!failed.report.converged, "capped Newton fails on the single step");
  cfg.solver.substep.on_failure = true;
  cfg.solver.substep.max_bisections = 4;
  cfg.solver.substep.min_dt = 1e-3;
  const Run bisected = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p) { p.AddTraction({2}, c); });
  std::printf("  bisection: full-step newton its %d, cap %d; recovered in %zu increments with %d bisections\n",
              full_its, cfg.solver.newton.max_it, bisected.report.steps.size(), bisected.report.bisections);
  CHECK_MSG(bisected.report.converged, "bisected run converged");
  CHECK_MSG(bisected.report.bisections >= 1, "at least one bisection happened");
  CHECK_MSG(bisected.report.steps.size() >= 2, "more than one accepted increment");
  if (!bisected.report.steps.empty())
  {
    CHECK_CLOSE(bisected.report.steps.back().load_factor, 1.0, 1e-14);
    CHECK_MSG(bisected.report.steps.front().attempts >= 2, "the first accepted increment records the failed attempts");
  }
  mfem::Vector d(reference.u);
  d -= bisected.u;
  const double rel = d.Normlinf() / reference.u.Normlinf();
  std::printf("  bisection: |u_ref - u_bisected| / |u| = %.3e\n", rel);
  CHECK_MSG(rel <= 1e-9, "bisected path reaches the same state");
}

// ---------------------------------------------------------------- L2: f(x, y, z, t)

// A programmatic time-dependent coefficient with the constant schedule
// reproduces the ramp of a constant coefficient: same external load and
// same prescribed displacement at t = 0.7.
void TimeDependentCoefficientTest()
{
  cmf::AppConfig cfg = CookConfig();
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  mfem::Vector t0(2), u0(2);
  t0(0) = 0.3; t0(1) = 2.0;
  u0(0) = 0.1; u0(1) = -0.2;
  mfem::VectorConstantCoefficient traction(t0), ubar(u0);
  mfem::VectorFunctionCoefficient traction_t(2, [&](const mfem::Vector &, double t, mfem::Vector &v)
  {
    v = t0; v *= t;
  });
  mfem::VectorFunctionCoefficient ubar_t(2, [&](const mfem::Vector &, double t, mfem::Vector &v)
  {
    v = u0; v *= t;
  });
  cmf::BCOptions constant;
  constant.schedule = cmf::Schedule::Constant();
  constant.time_dependent = true;

  std::unique_ptr<cmf::SolidProblem> ramp = cmf::MakeSolidProblem(*mesh, cfg);
  ramp->AddTraction({2}, traction);
  ramp->AddDirichlet({3}, ubar);
  ramp->Finalize();
  ramp->SetLoadFactor(0.7);
  std::unique_ptr<cmf::SolidProblem> timed = cmf::MakeSolidProblem(*mesh, cfg);
  timed->AddTraction({2}, traction_t, constant);
  timed->AddDirichlet({3}, ubar_t, constant);
  timed->Finalize();
  timed->SetLoadFactor(0.7);

  mfem::Vector dL(ramp->Loads().ExternalLoad());
  dL -= timed->Loads().ExternalLoad();
  mfem::Vector xa(ramp->Height()), xb(timed->Height());
  xa = 0.0; xb = 0.0;
  ramp->ApplyDirichlet(xa);
  timed->ApplyDirichlet(xb);
  xa -= xb;
  std::printf("  time-dependent coefficient at t = 0.7: |dL| = %.3e, |du_bar| = %.3e\n",
              dL.Normlinf(), xa.Normlinf());
  CHECK_MSG(dL.Normlinf() <= 1e-14 * ramp->Loads().ExternalLoad().Normlinf(),
            "t-dependent traction with constant schedule = ramp of the constant traction");
  CHECK_MSG(xa.Normlinf() <= 1e-15, "t-dependent Dirichlet data with constant schedule = ramp");
  // The load is reassembled when t changes.
  timed->SetLoadFactor(0.35);
  mfem::Vector half(timed->Loads().ExternalLoad());
  half *= 2.0;
  half -= ramp->Loads().ExternalLoad();
  CHECK_MSG(half.Normlinf() <= 1e-14 * ramp->Loads().ExternalLoad().Normlinf(),
            "time-dependent load reassembled at t = 0.35");
}

// The manufactured problem of tests/input/mms_expression.yaml, from string
// to residual: expression Dirichlet data and body force, p = 2 convergence.
void ExpressionMMSTest()
{
  cmf::AppConfig cfg = cmf::LoadConfig("tests/input/mms_expression.yaml");
  mfem::VectorFunctionCoefficient exact(2, [](const mfem::Vector &X, mfem::Vector &u)
  {
    u.SetSize(2);
    u(0) = 0.05 * std::sin(M_PI * X(0)) * std::sin(M_PI * X(1));
    u(1) = 0.05 * X(0) * X(0) * X(1) * (1.0 - X(1));
  });
  std::vector<double> errors;
  for (int refine = 1; refine <= 3; refine++)
  {
    cfg.mesh.serial_refine = refine;
    std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
    problem->Finalize();
    std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
    mfem::Vector u(problem->Height());
    u = 0.0;
    const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(*problem, *linear, cfg.solver, u);
    CHECK_MSG(report.converged, "expression MMS converged at refine " + std::to_string(refine));
    problem->UpdateFields(u);
    errors.push_back(problem->Displacement().ComputeL2Error(exact));
    std::printf("  expression MMS refine %d: L2 error %.3e, newton its %d\n", refine,
                errors.back(), report.steps.back().newton.iterations);
  }
  for (std::size_t i = 1; i < errors.size(); i++)
  {
    const double rate = std::log(errors[i - 1] / errors[i]) / std::log(2.0);
    std::printf("  expression MMS rate %zu: %.3f\n", i, rate);
    CHECK_MSG(rate >= 2.8, "p = 2 MMS rate " + std::to_string(rate) + " >= 2.8");
  }
}

// A traction expression linear in t with 4 steps equals value + ramp; one
// that does not mention t is assembled once.
void ExpressionReassemblyTest()
{
  cmf::AppConfig cfg = CookConfig();
  cfg.solver.load_steps = 4;
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  cmf::BoundaryCondition ramp_bc, expr_bc, fixed_bc;
  ramp_bc.value = {0.0, 3.75};
  expr_bc.expression = {"0", "3.75*t*(1 + 0.2*sin(pi*y/16))"};
  expr_bc.schedule = cmf::Schedule::Constant();
  fixed_bc.expression = {"0", "3.75*(1 + 0.2*sin(pi*y/16))"};
  std::unique_ptr<mfem::VectorCoefficient> c_ramp = cmf::MakeBCCoefficient(ramp_bc, 2, "ramp");
  std::unique_ptr<mfem::VectorCoefficient> c_expr = cmf::MakeBCCoefficient(expr_bc, 2, "expr");
  std::unique_ptr<mfem::VectorCoefficient> c_fixed = cmf::MakeBCCoefficient(fixed_bc, 2, "fixed");
  CHECK(cmf::OptionsOf(expr_bc).time_dependent);
  CHECK(!cmf::OptionsOf(fixed_bc).time_dependent);

  const Run a = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p) { p.AddTraction({2}, *c_fixed); });
  const Run b = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p)
  {
    p.AddTraction({2}, *c_expr, cmf::OptionsOf(expr_bc));
  });
  mfem::Vector d(a.u);
  d -= b.u;
  const double rel = d.Normlinf() / a.u.Normlinf();
  std::printf("  expression traction linear in t vs ramp of the fixed expression: %.3e\n", rel);
  CHECK_MSG(a.report.converged && b.report.converged, "expression traction runs converged");
  CHECK_MSG(rel <= 1e-12, "expression in t reproduces the ramp");
  // The spatial variation did something: differs from the uniform traction.
  const Run u = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p) { p.AddTraction({2}, *c_ramp); });
  d = a.u;
  d -= u.u;
  CHECK_MSG(d.Normlinf() / a.u.Normlinf() > 1e-3, "the y-dependent traction differs from the uniform one");
}

// ---------------------------------------------------------- L3: component-wise Dirichlet

// The uniaxial cube as a symmetry model (rollers on three planes, the
// stretch in x only on X = 1, the other faces free) reproduces the affine
// field and the closed-form pressure of the full-cube input.
void SymmetryCubeTest()
{
  cmf::AppConfig cfg = cmf::LoadConfig("apps/input/homogeneous/symmetry_uniaxial_neo_hookean.yaml");
  cfg.output.paraview.clear();
  cfg.output.fields = {"displacement", "pressure"};
  cfg.solver.newton.print_level = 0;
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
  problem->Finalize();
  // Only the listed components are essential: one per node on each of the
  // four constrained faces (25 Q2 nodes per face of the 2 x 2 x 2 cube; the
  // faces constrain different components, so shared edges count twice).
  const int ess_local = problem->EssentialTrueDofs().Size();
  int ess = 0;
  MPI_Allreduce(&ess_local, &ess, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  std::printf("  symmetry cube: %d essential true dofs of %lld\n", ess,
              static_cast<long long>(problem->GlobalTrueVSize()));
  CHECK_MSG(ess == 4 * 25, "one component per node on each of the four constrained faces");
  std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
  mfem::Vector x(problem->Height());
  x = 0.0;
  const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(*problem, *linear, cfg.solver, x);
  CHECK_MSG(report.converged, "symmetry cube converged");
  problem->UpdateFields(x);
  const double lam = 2.0;
  mfem::Vector zero(3);
  zero = 0.0;
  mfem::DenseMatrix G(3);
  G = 0.0;
  G(0, 0) = lam - 1.0;
  G(1, 1) = G(2, 2) = 1.0 / std::sqrt(lam) - 1.0;
  cmf::AffineVectorCoefficient exact(zero, G);
  mfem::ParGridFunction err(&problem->DisplacementSpace());
  err.ProjectCoefficient(exact);
  err -= problem->Displacement();
  cmf::FieldRegistry fields;
  problem->RegisterFields(fields);
  const std::vector<double> p_corner = cmf::ProbeVector(fields.Get("pressure"), {1.0, 1.0, 1.0});
  const std::vector<double> p_center = cmf::ProbeVector(fields.Get("pressure"), {0.5, 0.5, 0.5});
  const double sigma11 = lam * lam - 1.0 / lam; // mu = 1
  std::printf("  symmetry cube: max nodal |u - u_exact| = %.3e, p = %.10f (exact %.10f)\n",
              err.Normlinf(), p_corner[0], sigma11 / 3.0);
  CHECK_MSG(err.Normlinf() <= 1e-10, "affine field reproduced with rollers");
  CHECK_CLOSE(p_corner[0], sigma11 / 3.0, 1e-9);
  CHECK_CLOSE(p_center[0], sigma11 / 3.0, 1e-9);
}

// Rollers on all six faces (each its normal component) under a body force:
// no rigid mode survives, the linear solves converge and the state is
// nontrivial; the normal displacement vanishes on every face.
void RollerRankTest()
{
  cmf::AppConfig cfg = cmf::LoadConfig("apps/input/homogeneous/symmetry_uniaxial_neo_hookean.yaml");
  cfg.output.paraview.clear();
  cfg.output.fields = {"displacement"};
  cfg.solver.newton.print_level = 0;
  cfg.formulation = "displacement";
  cfg.material = cmf::MaterialConfig();
  cfg.material.model = "neo_hookean";
  cfg.material.E = 10.0;
  cfg.material.nu = 0.3;
  cfg.bcs = cmf::BCConfig();
  cfg.body_force = cmf::BodyForceConfig();
  cfg.solver.load_steps = 2;
  cfg.solver.linear = cmf::LinearSolverConfig();
  cfg.solver.linear.rtol = 1e-13;
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
  mfem::Vector zero(3), bvec(3);
  zero = 0.0;
  bvec(0) = 0.4; bvec(1) = -0.7; bvec(2) = -1.0;
  mfem::VectorConstantCoefficient zero_coef(zero), body(bvec);
  // box.geo: bottom (z) 1, front (y) 2, right (x) 3, back (y) 4, left (x) 5, top (z) 6.
  const int normal_of[7] = {-1, 2, 1, 0, 1, 0, 2};
  for (int attr = 1; attr <= 6; attr++)
  {
    cmf::BCOptions opt;
    opt.components = {normal_of[attr]};
    problem->AddDirichlet({attr}, zero_coef, opt);
  }
  problem->SetBodyForce(body);
  problem->Finalize();
  std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
  mfem::Vector x(problem->Height());
  x = 0.0;
  const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(*problem, *linear, cfg.solver, x);
  int linear_failures = 0;
  for (const cmf::LoadStepReport &s : report.steps) { linear_failures += s.newton.linear_solve_failures; }
  std::printf("  rollers on all faces: converged %d, |u|_inf = %.4e, linear failures %d\n",
              int(report.converged), x.Normlinf(), linear_failures);
  CHECK_MSG(report.converged, "all-roller box converged");
  CHECK_MSG(linear_failures == 0, "no linear solve failure (no rigid mode left)");
  CHECK_MSG(x.Normlinf() > 1e-3, "the body force deforms the box");
  // Normal displacements vanish on the faces, tangential ones do not.
  problem->UpdateFields(x);
  const std::vector<double> u_face = cmf::ProbeVector(problem->Displacement(), {1.0, 0.5, 0.5});
  CHECK_MSG(std::abs(u_face[0]) <= 1e-14, "u_x = 0 on X = 1");
  CHECK_MSG(std::abs(u_face[1]) + std::abs(u_face[2]) > 1e-4, "tangential slip on X = 1");
}

// The plane-strain MMS of tests/input/mms_expression.yaml with x prescribed
// on bottom and top only; the y component there is natural with the exact
// traction P N supplied. Same p = 2 rate as with the full Dirichlet data.
void ComponentMMSTest()
{
  cmf::AppConfig cfg = cmf::LoadConfig("tests/input/mms_expression.yaml");
  cfg.bcs.dirichlet.clear(); // the body force expression stays
  const double alpha = 0.05;
  mfem::VectorFunctionCoefficient exact(2, [alpha](const mfem::Vector &X, mfem::Vector &u)
  {
    u.SetSize(2);
    u(0) = alpha * std::sin(M_PI * X(0)) * std::sin(M_PI * X(1));
    u(1) = alpha * X(0) * X(0) * X(1) * (1.0 - X(1));
  });
  const cmf::StVenantKirchhoff svk = std::get<cmf::StVenantKirchhoff>(cmf::MakeMaterial(cfg.material));
  auto traction = [&](double ny)
  {
    return mfem::VectorFunctionCoefficient(2, [alpha, svk, ny](const mfem::Vector &X, mfem::Vector &T)
    {
      tensor<double, 2, 2> H;
      H(0, 0) = alpha * M_PI * std::cos(M_PI * X(0)) * std::sin(M_PI * X(1));
      H(0, 1) = alpha * M_PI * std::sin(M_PI * X(0)) * std::cos(M_PI * X(1));
      H(1, 0) = alpha * 2.0 * X(0) * X(1) * (1.0 - X(1));
      H(1, 1) = alpha * X(0) * X(0) * (1.0 - 2.0 * X(1));
      const tensor<double, 2, 2> P = cmf::QPointStress<cmf::StVenantKirchhoff, 2>(svk, H);
      T.SetSize(2);
      T(0) = P(0, 1) * ny;
      T(1) = P(1, 1) * ny;
    });
  };
  mfem::VectorFunctionCoefficient T_bottom = traction(-1.0), T_top = traction(1.0);
  std::vector<double> errors;
  for (int refine = 1; refine <= 3; refine++)
  {
    cfg.mesh.serial_refine = refine;
    std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
    cmf::BCOptions x_only;
    x_only.components = {0};
    problem->AddDirichlet({2, 4}, exact);          // right, left: both components
    problem->AddDirichlet({1, 3}, exact, x_only);  // bottom, top: x only
    problem->AddTraction({1}, T_bottom);
    problem->AddTraction({3}, T_top);
    problem->Finalize();
    std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
    mfem::Vector u(problem->Height());
    u = 0.0;
    const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(*problem, *linear, cfg.solver, u);
    CHECK_MSG(report.converged, "component MMS converged at refine " + std::to_string(refine));
    problem->UpdateFields(u);
    errors.push_back(problem->Displacement().ComputeL2Error(exact));
    std::printf("  component MMS refine %d: L2 error %.3e\n", refine, errors.back());
  }
  for (std::size_t i = 1; i < errors.size(); i++)
  {
    const double rate = std::log(errors[i - 1] / errors[i]) / std::log(2.0);
    std::printf("  component MMS rate %zu: %.3f\n", i, rate);
    CHECK_MSG(rate >= 2.8, "p = 2 component MMS rate " + std::to_string(rate) + " >= 2.8");
  }
}

// ------------------------------------------------- L4: pressure tractions, dead and follower

cmf::AppConfig BoxConfig(int dim, int n, int order, double perturb)
{
  cmf::AppConfig cfg;
  cfg.mesh.cartesian = true;
  cfg.mesh.box.dim = dim;
  cfg.mesh.box.nx = cfg.mesh.box.ny = n;
  cfg.mesh.box.nz = dim == 3 ? n : 1;
  cfg.mesh.box.element = dim == 3 ? "hex" : "quad";
  cfg.mesh.order = order;
  cfg.mesh.perturb = perturb;
  cfg.material.model = "neo_hookean";
  cfg.material.E = 250.0;
  cfg.material.nu = 0.3;
  cfg.solver.newton.print_level = 0;
  return cfg;
}

// Assembled tangent vs central differences of the residual along a random
// direction, with a spatially varying follower pressure on the face X = L
// (attribute 2 in 2D, 3 in 3D) at a random nonzero state.
void FollowerTangentTest(cmf::SolidProblem &physics, const std::string &label, unsigned seed,
                         double amplitude)
{
  physics.Finalize();
  physics.SetLoadFactor(0.8);
  const int n = physics.Height();
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  mfem::Vector x(n), v(n), Jv(n), rp(n), rm(n), xp(n), xm(n);
  for (int i = 0; i < n; i++) { x(i) = amplitude * unit(rng); v(i) = unit(rng); }
  physics.ApplyDirichlet(x);
  for (int i = 0; i < physics.EssentialTrueDofs().Size(); i++) { v(physics.EssentialTrueDofs()[i]) = 0.0; }
  mfem::Operator &J = physics.GetGradient(x);
  J.Mult(v, Jv);
  const double eps = 1e-6;
  xp = x; xp.Add(eps, v);
  xm = x; xm.Add(-eps, v);
  physics.Mult(xp, rp);
  physics.Mult(xm, rm);
  rp -= rm;
  rp /= 2.0 * eps;
  rp -= Jv;
  const double rel = rp.Normlinf() / Jv.Normlinf();
  std::printf("  follower tangent %s: |J v - FD| / |J v| = %.3e\n", label.c_str(), rel);
  CHECK_MSG(rel <= 1e-6, label + " follower tangent vs FD relative error " + std::to_string(rel));
}

void FollowerTangentTests()
{
  cmf::ExpressionCoefficient p2("0.7 + 0.3*y + 0.1*t");
  cmf::ExpressionCoefficient p3("0.7 + 0.3*y - 0.2*z");
  cmf::BCOptions timed;
  timed.time_dependent = true;
  {
    cmf::AppConfig cfg = BoxConfig(2, 3, 2, 0.2);
    std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    cmf::SolidMechanicsTL physics(*mesh, cfg, cmf::MakeMaterial(cfg.material));
    mfem::Vector zero(2);
    zero = 0.0;
    mfem::VectorConstantCoefficient clamp(zero);
    physics.AddDirichlet({4}, clamp);
    physics.AddPressure({2}, p2, true, timed);
    FollowerTangentTest(physics, "2D quad p=2", 3u, 0.05);
    // The tangent is not symmetric, so cg_amg is refused by name.
    cmf::LinearSolverConfig lc;
    lc.type = "cg_amg";
    CHECK_THROWS(physics.MakeLinearSolver(lc), cmf::ConfigError, "follower_pressure");
  }
  {
    cmf::AppConfig cfg = BoxConfig(3, 2, 2, 0.15);
    std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    cmf::SolidMechanicsTL physics(*mesh, cfg, cmf::MakeMaterial(cfg.material));
    mfem::Vector zero(3);
    zero = 0.0;
    mfem::VectorConstantCoefficient clamp(zero);
    physics.AddDirichlet({5}, clamp);
    physics.AddPressure({3}, p3, true);
    FollowerTangentTest(physics, "3D hex p=2", 5u, 0.04);
  }
  {
    cmf::AppConfig cfg = BoxConfig(2, 3, 2, 0.2);
    cfg.material = cmf::MaterialConfig();
    cfg.material.model = "iso_neo_hookean";
    cfg.material.mu = 40.0;
    cfg.material.incompressible = true;
    std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    cmf::MixedSolidMechanicsTL physics(*mesh, cfg, cmf::MakeMixedMaterial(cfg.material));
    mfem::Vector zero(2);
    zero = 0.0;
    mfem::VectorConstantCoefficient clamp(zero);
    physics.AddDirichlet({4}, clamp);
    physics.AddPressure({2}, p2, true, timed);
    FollowerTangentTest(physics, "mixed 2D Q2-Q1", 7u, 0.03);
  }
}

// At F = I the follower pressure equals the dead pressure exactly (normal
// orientation and area scaling of the face kernel); under a small load the
// two solutions agree to O(p^2); on Cook's membrane the dead pressure on the
// right edge is the uniform traction -p N, N = (1, 0).
void FollowerVsDeadTest()
{
  for (int dim = 2; dim <= 3; dim++)
  {
    cmf::AppConfig cfg = BoxConfig(dim, 2, 2, 0.15);
    std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    cmf::ExpressionCoefficient p(dim == 2 ? "1.3 + 0.5*y" : "1.3 + 0.5*y - 0.4*z");
    const std::vector<int> face = {dim == 2 ? 2 : 3, dim == 2 ? 3 : 6}; // X = L and the top
    cmf::SolidMechanicsTL dead(*mesh, cfg, cmf::MakeMaterial(cfg.material));
    cmf::SolidMechanicsTL follower(*mesh, cfg, cmf::MakeMaterial(cfg.material));
    dead.AddPressure(face, p, false);
    follower.AddPressure(face, p, true);
    dead.Finalize();
    follower.Finalize();
    dead.SetLoadFactor(1.0);
    follower.SetLoadFactor(1.0);
    mfem::Vector u(dead.Height()), rd(dead.Height()), rf(dead.Height());
    u = 0.0;
    dead.Mult(u, rd);
    follower.Mult(u, rf);
    rf -= rd;
    std::printf("  follower vs dead pressure at F = I (%dD): |R_f - R_d| / |R_d| = %.3e\n", dim,
                rf.Normlinf() / rd.Normlinf());
    CHECK_MSG(rf.Normlinf() <= 1e-13 * rd.Normlinf(), "follower load at F = I equals the dead pressure");
  }
  cmf::AppConfig cfg = CookConfig();
  cfg.solver.load_steps = 1;
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  mfem::ConstantCoefficient p_small(1e-3), p_dead(-2.0);
  mfem::Vector t(2);
  t(0) = 2.0; t(1) = 0.0;
  mfem::VectorConstantCoefficient traction(t);
  const Run a = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p) { p.AddPressure({2}, p_small, false); });
  const Run b = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p) { p.AddPressure({2}, p_small, true); });
  mfem::Vector d(a.u);
  d -= b.u;
  const double rel = d.Normlinf() / a.u.Normlinf();
  std::printf("  follower vs dead pressure p = 1e-3 on Cook: relative difference %.3e\n", rel);
  CHECK_MSG(a.report.converged && b.report.converged, "small-pressure runs converged");
  CHECK_MSG(rel <= 5e-3 && rel > 1e-8, "dead and follower agree to O(p) relative");
  // Dead pressure -2 on the right edge (N = (1, 0)) is the traction (2, 0).
  const Run c = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p) { p.AddPressure({2}, p_dead, false); });
  const Run e = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p) { p.AddTraction({2}, traction); });
  d = c.u;
  d -= e.u;
  CHECK_MSG(d.Normlinf() <= 1e-12 * c.u.Normlinf(), "dead pressure -p N equals the vector traction");
}

// Thick-walled incompressible cylinder under internal follower pressure
// (apps/input/cylinder_inflation.yaml) vs the closed-form inflation.
void CylinderInflationTest()
{
  const double mu = 1.0, A = 1.0, B = 2.0, P = 0.3;
  auto pressure_of = [&](double a)
  {
    const double b = std::sqrt(B * B - A * A + a * a);
    return mu * (std::log(B / A) - std::log(b / a) + 0.5 * (A * A - a * a) * (1.0 / (b * b) - 1.0 / (a * a)));
  };
  double lo = A, hi = 3.0 * A;
  for (int i = 0; i < 200; i++)
  {
    const double m = 0.5 * (lo + hi);
    (pressure_of(m) < P ? lo : hi) = m;
  }
  const double a_exact = 0.5 * (lo + hi);
  const double b_exact = std::sqrt(B * B - A * A + a_exact * a_exact);

  cmf::AppConfig cfg = cmf::LoadConfig("apps/input/cylinder_inflation.yaml");
  cfg.output.paraview.clear();
  cfg.output.fields = {"displacement"};
  cfg.solver.newton.print_level = 0;
  CHECK(cfg.bcs.traction.size() == 1 && cfg.bcs.traction[0].type == "follower_pressure");
  std::vector<double> errors;
  for (int refine = 0; refine <= 1; refine++)
  {
    cfg.mesh.serial_refine = refine;
    std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
    problem->Finalize();
    std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
    mfem::Vector x(problem->Height());
    x = 0.0;
    const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(*problem, *linear, cfg.solver, x);
    CHECK_MSG(report.converged, "cylinder inflation converged at refine " + std::to_string(refine));
    problem->UpdateFields(x);
    const std::vector<double> ui = cmf::ProbeVector(problem->Displacement(), {A, 0.0});
    const std::vector<double> uo = cmf::ProbeVector(problem->Displacement(), {B, 0.0});
    const std::vector<double> ut = cmf::ProbeVector(problem->Displacement(), {0.0, A});
    const double a = A + ui[0], b = B + uo[0];
    errors.push_back(std::abs(a - a_exact) / (a_exact - A));
    std::printf("  cylinder inflation refine %d: a = %.8f (exact %.8f), b = %.8f (exact %.8f), "
                "newton its (last step) %d\n", refine, a, a_exact, b, b_exact,
                report.steps.back().newton.iterations);
    CHECK_MSG(std::abs(a - a_exact) <= 2e-3 * (a_exact - A), "inner radius within 0.2% of the inflation");
    CHECK_MSG(std::abs(b - b_exact) <= 2e-3 * (b_exact - B), "outer radius within 0.2% of the inflation");
    CHECK_MSG(std::abs(ut[1] - ui[0]) <= 1e-6 * ui[0], "axisymmetric: same radial displacement at (0, A)");
    CHECK_MSG(std::abs(ui[1]) <= 1e-12 && std::abs(ut[0]) <= 1e-12, "rollers hold the symmetry lines");
    CHECK_MSG(report.steps.back().newton.iterations <= 6, "quadratic Newton with the follower tangent");
  }
  CHECK_MSG(errors.size() == 2 && errors[1] < errors[0], "the inflation error decreases under refinement");
}

} // namespace

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  StagedLoadingTest();
  LoadUnloadTest();
  BisectionTest();
  TimeDependentCoefficientTest();
  ExpressionMMSTest();
  ExpressionReassemblyTest();
  SymmetryCubeTest();
  RollerRankTest();
  ComponentMMSTest();
  FollowerTangentTests();
  FollowerVsDeadTest();
  CylinderInflationTest();
  return cmf_test::Report("test_loading");
}

// Axisymmetric kinematics (plane: axisymmetric; x = r, y = z, hoop stretch
// 1 + u_r / r, weight 2 pi r) in both formulations: the patch test of a
// uniform dilatation on a ring and on a disc (F including the hoop stretch,
// zero interior residual, the axial reaction as the stress times the area of
// the solid of revolution), Rivlin's inflated cylinder as a meridian strip and
// the Green-Zerna sphere as a quarter annulus against their closed forms (the
// verification inputs), the assembled Jacobian with a follower pressure and a
// rigid-sphere contact against finite differences, and the mass of a solid of
// revolution through the dynamics decorator.
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

// The (r, z) rectangle [A, B] x [0, H] as n x n Q2 quadrilaterals (Q2-Q1 in the
// mixed formulation) through the corner map of the unit square; boundary
// attributes of MFEM's box: bottom (z = 0) 1, right (r = B) 2, top (z = H) 3,
// left (r = A) 4.
cmf::AppConfig RectangleConfig(const std::string &formulation, double A, double B, double H, int n)
{
  cmf::AppConfig cfg;
  cfg.formulation = formulation;
  cfg.plane = "axisymmetric";
  cfg.mesh.cartesian = true;
  cfg.mesh.box.dim = 2;
  cfg.mesh.box.element = "quad";
  cfg.mesh.box.nx = cfg.mesh.box.ny = n;
  cfg.mesh.corners = {{A, 0.0}, {B, 0.0}, {B, H}, {A, H}};
  cfg.mesh.order = 2;
  cfg.material.model = "iso_neo_hookean";
  cfg.material.mu = 1.0;
  cfg.material.kappa = 10.0;
  cfg.solver.newton.rtol = 1e-11;
  cfg.solver.newton.atol = 1e-14;
  cfg.solver.newton.max_it = 30;
  cfg.solver.newton.print_level = 0;
  cfg.solver.linear.type = "direct";
  cfg.solver.predictor = "tangent";
  cfg.output.fields = {"displacement", "deformation_gradient"};
  return cfg;
}

const cmf::Reaction &Named(const std::vector<cmf::Reaction> &r, const std::string &name)
{
  for (const cmf::Reaction &x : r) { if (x.name == name) { return x; } }
  MFEM_ABORT("no reaction named " << name);
  return r[0];
}

double GlobalNorm(MPI_Comm comm, const mfem::Vector &v)
{
  return std::sqrt(mfem::InnerProduct(comm, v, v));
}

// u = eps X on every face: F = (1 + eps) I including the hoop stretch, a
// uniform stress, no residual on the free dofs, and the axial reaction of the
// top face P_zz pi (B^2 - A^2).
void PatchTest(const std::string &formulation, const std::string &model, double A, double B)
{
  std::printf("patch test, %s formulation, %s, ring [%g, %g]\n", formulation.c_str(), model.c_str(), A, B);
  const double H = 0.7, eps = 0.02;
  cmf::AppConfig cfg = RectangleConfig(formulation, A, B, H, 3);
  cfg.material.model = model;
  if (model == "linear_elastic") { cfg.solver.predictor = "none"; cfg.solver.newton.rtol = 1e-8; }
  cfg.output.quadrature_at = {"nodes", "elements"};
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
  cmf::SolidProblem &physics = *problem;
  mfem::Vector value(2);
  value = 0.0;
  mfem::DenseMatrix G(2);
  G = 0.0;
  G(0, 0) = G(1, 1) = eps;
  cmf::AffineVectorCoefficient affine(value, G);
  cmf::BCOptions opt;
  opt.name = "top";
  physics.AddDirichlet({3}, affine, opt);
  cmf::BCOptions rest;
  rest.name = "rest";
  physics.AddDirichlet({1, 2, 4}, affine, rest);
  physics.Finalize();
  std::unique_ptr<mfem::Solver> linear = physics.MakeLinearSolver(cfg.solver.linear);
  mfem::Vector x(physics.Height());
  x = 0.0;
  cfg.solver.load_steps = 1;
  const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(physics, *linear, cfg.solver, x);
  CHECK(report.converged);
  physics.UpdateFields(x);
  Mat3 F = cmf::I<3>();
  for (int i = 0; i < 3; i++) { F(i, i) = 1.0 + eps; }
  // The stress of the material at this F (the mixed constraint is exact for a uniform state).
  cmf::MaterialConfig mc = cfg.material;
  const cmf::Material material = cmf::MakeMaterial(mc);
  const Mat3 P = std::visit([&](const auto &m) -> Mat3
  {
    using M = std::decay_t<decltype(m)>;
    if constexpr (cmf::has_history<M>::value) { return Mat3(); }
    else { return m.PK1(F); }
  }, material);
  const std::vector<cmf::Reaction> reactions = physics.Reactions(x);
  const cmf::Reaction &top = Named(reactions, "top");
  const double expected = P(1, 1) * M_PI * (B * B - A * A);
  std::printf("  top reaction %.12e, P_zz pi (B^2 - A^2) = %.12e, rel %.2e; F_33 at the centre: ",
              top.force[1], expected, std::abs(top.force[1] - expected) / std::abs(expected));
  CHECK_CLOSE(top.force[1], expected, 1e-10 * std::abs(expected));
  // (The radial component of this entry is not zero: its corner nodes carry
  // the radial tractions P_rr of the lateral faces they share.)
  // The residual vanishes on the free dofs (Mult zeroes the essential rows).
  mfem::Vector r(x.Size());
  physics.Mult(x, r);
  mfem::Vector full;
  physics.FullResidual(x, full);
  const double free_norm = GlobalNorm(physics.Comm(), r), full_norm = GlobalNorm(physics.Comm(), full);
  CHECK_MSG(free_norm <= 1e-12 * full_norm, "no residual on the free dofs");
  // The deformation gradient field carries the hoop stretch.
  cmf::FieldRegistry fields;
  physics.RegisterFields(fields);
  physics.UpdateFields(x);
  const std::vector<double> Fnode = cmf::ProbeVector(fields.Get("deformation_gradient"), {0.5 * (A + B), 0.5 * H});
  const std::vector<double> Felem = cmf::ProbeVector(fields.Get("deformation_gradient_elem"), {0.5 * (A + B), 0.5 * H});
  std::printf("%.12f (nodes), %.12f (elements)\n", Fnode[8], Felem[8]);
  CHECK_CLOSE(Fnode[8], 1.0 + eps, 1e-10);
  CHECK_CLOSE(Felem[8], 1.0 + eps, 1e-10);
  CHECK_CLOSE(Fnode[0], 1.0 + eps, 1e-10);
  CHECK(std::abs(Fnode[1]) <= 1e-10);
}

// Rivlin's cylinder: the verification input, a against the closed form.
void CylinderTest()
{
  std::printf("Rivlin's cylinder as an axisymmetric strip\n");
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
  cmf::AppConfig cfg = cmf::LoadConfig("apps/input/finite_elasticity/verification/rivlin_cylinder_axisymmetric.yaml");
  cfg.output.paraview.clear();
  cfg.solver.newton.print_level = 0;
  CHECK(cfg.plane == "axisymmetric");
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
  problem->Finalize();
  std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
  mfem::Vector x(problem->Height());
  x = 0.0;
  const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(*problem, *linear, cfg.solver, x);
  CHECK(report.converged);
  problem->UpdateFields(x);
  const std::vector<double> ui = cmf::ProbeVector(problem->Displacement(), {A, 0.25});
  const std::vector<double> uo = cmf::ProbeVector(problem->Displacement(), {B, 0.25});
  const double a = A + ui[0], b = B + uo[0];
  std::printf("  a = %.8f (exact %.8f, rel %.2e), b = %.8f (exact %.8f, rel %.2e), newton its %d\n",
              a, a_exact, std::abs(a - a_exact) / (a_exact - A), b, b_exact, std::abs(b - b_exact) / (b_exact - B),
              report.steps.back().newton.iterations);
  CHECK_MSG(std::abs(a - a_exact) <= 1e-4 * (a_exact - A), "inner radius within 1e-4 of Rivlin");
  CHECK_MSG(std::abs(b - b_exact) <= 1e-4 * (b_exact - B), "outer radius within 1e-4 of Rivlin");
  CHECK(std::abs(ui[1]) <= 1e-12);
  CHECK(report.steps.back().newton.iterations <= 6);
  // The axial reaction of the rollers is the total force of the tube: zero net
  // (both faces) and, on each, the integral of P_zz 2 pi r.
  const std::vector<cmf::Reaction> reactions = problem->Reactions(x);
  const cmf::Reaction &bottom = Named(reactions, "bottom"), &top = Named(reactions, "top");
  std::printf("  axial reactions: bottom %.6e, top %.6e\n", bottom.force[1], top.force[1]);
  CHECK_CLOSE(bottom.force[1] + top.force[1], 0.0, 1e-10 * std::abs(top.force[1]) + 1e-14);
}

// Green-Zerna's sphere: the verification input, a against the closed form.
void SphereTest()
{
  std::printf("Green-Zerna's sphere as a quarter annulus\n");
  const double mu = 1.0, A = 1.0, B = 2.0;
  auto pressure_of = [&](double a)
  {
    const double b = std::cbrt(B * B * B - A * A * A + a * a * a);
    const double la = a / A, lb = b / B;
    return 0.5 * mu * ((4.0 / lb + 1.0 / std::pow(lb, 4)) - (4.0 / la + 1.0 / std::pow(la, 4)));
  };
  cmf::AppConfig cfg = cmf::LoadConfig("apps/input/finite_elasticity/verification/green_zerna_sphere_axisymmetric.yaml");
  cfg.output.paraview.clear();
  cfg.solver.newton.print_level = 0;
  const double P = cmf::Expression::Parse(cfg.bcs.traction.at(0).expression.at(0)).Eval(0, 0, 0, 1);
  double lo = A, hi = 1.8 * A;
  for (int i = 0; i < 200; i++)
  {
    const double m = 0.5 * (lo + hi);
    (pressure_of(m) < P ? lo : hi) = m;
  }
  const double a_exact = 0.5 * (lo + hi);
  const double b_exact = std::cbrt(B * B * B - A * A * A + a_exact * a_exact * a_exact);
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
  problem->Finalize();
  std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
  mfem::Vector x(problem->Height());
  x = 0.0;
  const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(*problem, *linear, cfg.solver, x);
  CHECK(report.converged);
  problem->UpdateFields(x);
  const std::vector<double> ui = cmf::ProbeVector(problem->Displacement(), {A, 0.0});
  const std::vector<double> uo = cmf::ProbeVector(problem->Displacement(), {B, 0.0});
  const std::vector<double> up = cmf::ProbeVector(problem->Displacement(), {0.0, A});
  const double a = A + ui[0], b = B + uo[0], a_pole = A + up[1];
  std::printf("  a = %.6f (exact %.6f, rel %.2e), b = %.6f (exact %.6f, rel %.2e), a at the pole %.6f\n",
              a, a_exact, std::abs(a - a_exact) / (a_exact - A), b, b_exact, std::abs(b - b_exact) / (b_exact - B), a_pole);
  CHECK_MSG(std::abs(a - a_exact) <= 2e-3 * (a_exact - A), "inner radius within 0.2% of Green-Zerna");
  CHECK_MSG(std::abs(b - b_exact) <= 2e-3 * (b_exact - B), "outer radius within 0.2% of Green-Zerna");
  CHECK_MSG(std::abs(a_pole - a) <= 1e-3 * (a - A), "spherically symmetric response");
}

// The assembled Jacobian against central differences of the residual, with a
// follower pressure on the top of a disc and a rigid sphere on its axis
// pressing the top, in both formulations.
void JacobianTest(const std::string &formulation)
{
  std::printf("Jacobian with a follower pressure and a contact, %s formulation\n", formulation.c_str());
  cmf::AppConfig cfg = RectangleConfig(formulation, 0.0, 1.0, 0.5, 3);
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
  cmf::SolidProblem &physics = *problem;
  mfem::Vector zero(2), c(2);
  zero = 0.0;
  c(0) = 0.0;
  c(1) = 0.5 + 2.0 - 0.02;
  mfem::VectorConstantCoefficient zero_coef(zero), center(c);
  mfem::ConstantCoefficient pressure(0.1);
  cmf::BCOptions axis, bottom, contact;
  axis.components = {0};
  bottom.components = {1};
  bottom.name = "bottom";
  contact.name = "sphere";
  physics.AddDirichlet({4}, zero_coef, axis);
  physics.AddDirichlet({1}, zero_coef, bottom);
  physics.AddPressure({3}, pressure, true);
  physics.AddRigidSphereContact({3}, center, 2.0, 3.0, contact);
  physics.Finalize();
  physics.SetLoadFactor(1.0);
  std::mt19937 rng(7u);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  const mfem::Array<int> &ess = physics.EssentialTrueDofs();
  mfem::Vector x(physics.Height()), v(x.Size());
  mfem::Array<int> is_ess(x.Size());
  is_ess = 0;
  for (int i = 0; i < ess.Size(); i++) { is_ess[ess[i]] = 1; }
  for (int i = 0; i < x.Size(); i++)
  {
    x(i) = is_ess[i] ? 0.0 : 0.02 * unit(rng);
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
  const double err = GlobalNorm(physics.Comm(), rp) / GlobalNorm(physics.Comm(), Jv);
  std::printf("  |J v - FD| / |J v| = %.2e\n", err);
  CHECK_MSG(err <= 1e-6, formulation + ": axisymmetric Jacobian vs finite differences");
  // The contact acts: its axial resultant on the undeformed top is negative
  // (the r component of an axisymmetric resultant is the sum of the
  // ring-integrated radial nodal forces, not a force of the solid).
  x = 0.0;
  const cmf::Reaction &rx = Named(physics.Reactions(x), "sphere");
  std::printf("  contact resultant at rest: (%.3e, %.6e)\n", rx.force[0], rx.force[1]);
  CHECK(rx.force[1] < 0.0);
}

// The mass of the solid of revolution: 1 . M . 1 = dim rho pi (B^2 - A^2) H.
void MassTest()
{
  std::printf("mass of a solid of revolution\n");
  const double A = 0.5, B = 1.5, H = 0.4, rho = 2.5;
  cmf::AppConfig cfg = RectangleConfig("displacement", A, B, H, 3);
  cfg.material.rho0 = rho;
  cfg.dynamics.enabled = true;
  cfg.dynamics.t_final = 1.0;
  cfg.dynamics.breakpoints = {1.0};
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
  mfem::Vector zero(2);
  zero = 0.0;
  mfem::VectorConstantCoefficient zero_coef(zero);
  cmf::BCOptions axis;
  axis.components = {1};
  problem->AddDirichlet({1}, zero_coef, axis);
  problem->Finalize();
  cmf::DynamicSolidProblem dynamic(*problem, cfg.dynamics);
  mfem::Vector x(problem->Height());
  x = 0.0;
  dynamic.Initialize(x);
  mfem::HypreParMatrix &M = dynamic.Mass();
  mfem::Vector ones(M.Height()), Mones(M.Height());
  ones = 1.0;
  M.Mult(ones, Mones);
  const double total = mfem::InnerProduct(problem->Comm(), ones, Mones);
  const double expected = 2.0 * rho * M_PI * (B * B - A * A) * H;
  std::printf("  1.M.1 = %.12e, dim rho pi (B^2 - A^2) H = %.12e\n", total, expected);
  CHECK_CLOSE(total, expected, 1e-12 * expected);
}

void ConfigTest()
{
  std::printf("input schema\n");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(R"(
mesh: { file: apps/mesh/cube.msh, order: 2 }
plane: cylindrical
material: { model: iso_neo_hookean, mu: 1.0, kappa: 10.0 }
)")), cmf::ConfigError, "axisymmetric");
  // A 3D mesh refuses the option at construction.
  cmf::AppConfig cfg = cmf::ParseConfig(YAML::Load(R"(
mesh: { file: apps/mesh/cube.msh, order: 2 }
plane: axisymmetric
material: { model: iso_neo_hookean, mu: 1.0, kappa: 10.0 }
)"));
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  CHECK_THROWS(cmf::MakeSolidProblem(*mesh, cfg), cmf::ConfigError, "2D mesh");
}

} // namespace

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  const cmf::PetscSession petsc; // solver.linear.type: direct
  ConfigTest();
  PatchTest("displacement", "iso_neo_hookean", 1.0, 2.0);
  PatchTest("mixed", "iso_neo_hookean", 1.0, 2.0);
  PatchTest("displacement", "iso_neo_hookean", 0.0, 1.0);
  PatchTest("mixed", "linear_elastic", 0.0, 1.0);
  CylinderTest();
  SphereTest();
  JacobianTest("displacement");
  JacobianTest("mixed");
  MassTest();
  return cmf_test::Report("test_axisymmetric");
}

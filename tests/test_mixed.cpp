// Mixed u-p formulation gates: patch test (isochoric affine field with a
// constant pressure), assembled block Jacobian vs finite differences, MMS
// convergence for near-incompressible (finite kappa) and fully incompressible
// (kappa = inf, volume-preserving manufactured motion) cases, and agreement
// between the mixed and displacement formulations at finite kappa. The same
// gates for small-strain linear elasticity (linear_elastic), whose mixed form
// is Herrmann's linear, symmetric saddle-point problem: the volume measure is
// 1 + tr(eps) with gradient I, every solve takes one or two Newton steps, and
// the block Jacobian does not depend on the state.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "base/mesh_input.hpp"
#include "kernels/mixed_total_lagrangian.hpp"
#include "materials/materials.hpp"
#include "mfem.hpp"
#include "physics/mixed_solid_mechanics_tl.hpp"
#include "physics/solid_mechanics_tl.hpp"
#include "solvers/quasi_static.hpp"
#include "test_util.hpp"

using cmf::tensor;

namespace
{

const double kMu = 80.0, kRho0 = 2.0;

cmf::AppConfig BaseConfig(int nx, int order, double perturb, const std::string &model,
                          double nu_or_half, bool incompressible)
{
  cmf::AppConfig cfg;
  cfg.formulation = "mixed";
  cfg.mesh.cartesian = true;
  cfg.mesh.box.nx = nx;
  cfg.mesh.box.ny = nx;
  cfg.mesh.order = order;
  cfg.mesh.perturb = perturb;
  cfg.material.model = model;
  if (model == "mooney_rivlin") { cfg.material.c1 = 0.3 * kMu; cfg.material.c2 = 0.2 * kMu; }
  else { cfg.material.mu = kMu; }
  if (incompressible) { cfg.material.incompressible = true; }
  else { cfg.material.nu = nu_or_half; }
  cfg.material.rho0 = kRho0;
  cfg.solver.newton.rtol = 1e-11;
  cfg.solver.newton.atol = 1e-14;
  cfg.solver.newton.max_it = 30;
  cfg.solver.newton.print_level = 0;
  cfg.solver.linear.rtol = 1e-13;
  cfg.solver.linear.max_it = 400;
  cfg.solver.linear.krylov_dim = 100;
  cfg.solver.linear.inner_rtol = 1e-4;
  cfg.solver.linear.inner_max_it = 100;
  if (std::getenv("CMF_LINEAR_PRINT")) { cfg.solver.linear.print_level = 1; }
  if (std::getenv("CMF_NEWTON_PRINT")) { cfg.solver.newton.print_level = 1; }
  return cfg;
}

// Manufactured displacement u(X), its gradient, and pressure p(X).
struct Manufactured
{
  std::function<void(const mfem::Vector &, mfem::Vector &)> u;
  std::function<tensor<double, 2, 2>(const mfem::Vector &)> grad;
  std::function<double(const mfem::Vector &)> p;
};

// Affine u = A X with a constant pressure p0. For finite kappa the pressure
// must satisfy the constraint, p0 = kappa (det(I + A) - 1); for an
// incompressible material det(I + A) = 1 and p0 is free.
Manufactured Affine(const tensor<double, 2, 2> &A, double p0)
{
  Manufactured m;
  m.u = [A](const mfem::Vector &X, mfem::Vector &u)
  {
    u.SetSize(2);
    for (int i = 0; i < 2; i++) { u(i) = A(i, 0) * X(0) + A(i, 1) * X(1); }
  };
  m.grad = [A](const mfem::Vector &) { return A; };
  m.p = [p0](const mfem::Vector &) { return p0; };
  return m;
}

// Volume-preserving composition of two shears:
//   x = X + f(Y),  y = Y + g(x),  f = a sin(pi Y), g = b sin(pi x)   (J = 1)
// with a smooth pressure p = c sin(pi X) cos(pi Y).
Manufactured Isochoric(double a, double b, double c)
{
  Manufactured m;
  m.u = [a, b](const mfem::Vector &X, mfem::Vector &u)
  {
    const double x = X(0) + a * std::sin(M_PI * X(1));
    const double y = X(1) + b * std::sin(M_PI * x);
    u.SetSize(2);
    u(0) = x - X(0);
    u(1) = y - X(1);
  };
  m.grad = [a, b](const mfem::Vector &X)
  {
    const double x = X(0) + a * std::sin(M_PI * X(1));
    const double dxdX = 1.0, dxdY = a * M_PI * std::cos(M_PI * X(1));
    const double gp = b * M_PI * std::cos(M_PI * x);
    tensor<double, 2, 2> H;
    H(0, 0) = dxdX - 1.0;
    H(0, 1) = dxdY;
    H(1, 0) = gp * dxdX;
    H(1, 1) = gp * dxdY;
    return H;
  };
  m.p = [c](const mfem::Vector &X) { return c * std::sin(M_PI * X(0)) * std::cos(M_PI * X(1)); };
  return m;
}

// Smooth compressible field u = alpha (sin(pi X) sin(pi Y), X^2 Y (1 - Y));
// for finite kappa the pressure is p = kappa (J - 1).
Manufactured Smooth(double alpha, double kappa)
{
  Manufactured m;
  m.u = [alpha](const mfem::Vector &X, mfem::Vector &u)
  {
    u.SetSize(2);
    u(0) = alpha * std::sin(M_PI * X(0)) * std::sin(M_PI * X(1));
    u(1) = alpha * X(0) * X(0) * X(1) * (1.0 - X(1));
  };
  m.grad = [alpha](const mfem::Vector &X)
  {
    tensor<double, 2, 2> H;
    H(0, 0) = alpha * M_PI * std::cos(M_PI * X(0)) * std::sin(M_PI * X(1));
    H(0, 1) = alpha * M_PI * std::sin(M_PI * X(0)) * std::cos(M_PI * X(1));
    H(1, 0) = alpha * 2.0 * X(0) * X(1) * (1.0 - X(1));
    H(1, 1) = alpha * X(0) * X(0) * (1.0 - 2.0 * X(1));
    return H;
  };
  m.p = [m, kappa](const mfem::Vector &X)
  {
    const tensor<double, 2, 2> F = cmf::I<2>() + m.grad(X);
    return kappa * (det(F) - 1.0);
  };
  return m;
}

// Total first Piola-Kirchhoff stress of the manufactured state at X.
template <typename Material>
tensor<double, 2, 2> ManufacturedStress(const Manufactured &m, const Material &mat,
                                        const mfem::Vector &X)
{
  return cmf::QPointMixedStress<Material, 2>(mat, m.grad(X), m.p(X));
}

// Body force b = -(1/rho0) Div P by central differences in X (step 1e-5).
template <typename Material>
class ManufacturedBodyForce : public mfem::VectorCoefficient
{
public:
  ManufacturedBodyForce(const Manufactured &m, const Material &mat, double rho0)
    : mfem::VectorCoefficient(2), m_(m), mat_(mat), rho0_(rho0) {}
  void Eval(mfem::Vector &b, mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip) override
  {
    mfem::Vector X(2);
    T.Transform(ip, X);
    const double h = 1e-5;
    b.SetSize(2);
    b = 0.0;
    for (int j = 0; j < 2; j++)
    {
      mfem::Vector Xp(X), Xm(X);
      Xp(j) += h;
      Xm(j) -= h;
      const tensor<double, 2, 2> Pp = ManufacturedStress(m_, mat_, Xp);
      const tensor<double, 2, 2> Pm = ManufacturedStress(m_, mat_, Xm);
      for (int i = 0; i < 2; i++) { b(i) += (Pp(i, j) - Pm(i, j)) / (2.0 * h); }
    }
    b *= -1.0 / rho0_;
  }
private:
  Manufactured m_;
  Material mat_;
  double rho0_;
};

// Nominal traction T = P N on the boundary (outward reference normal).
template <typename Material>
class ManufacturedTraction : public mfem::VectorCoefficient
{
public:
  ManufacturedTraction(const Manufactured &m, const Material &mat)
    : mfem::VectorCoefficient(2), m_(m), mat_(mat) {}
  void Eval(mfem::Vector &t, mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip) override
  {
    mfem::Vector X(2), N(2);
    T.SetIntPoint(&ip);
    T.Transform(ip, X);
    mfem::CalcOrtho(T.Jacobian(), N);
    N /= N.Norml2();
    const tensor<double, 2, 2> P = ManufacturedStress(m_, mat_, X);
    t.SetSize(2);
    for (int i = 0; i < 2; i++) { t(i) = P(i, 0) * N(0) + P(i, 1) * N(1); }
  }
private:
  Manufactured m_;
  Material mat_;
};

struct SolveResult
{
  double u_l2 = 0.0;
  double p_l2 = 0.0;
  double max_nodal_u = 0.0;
  double max_nodal_p = 0.0;
  cmf::NewtonReport newton;
  int load_step = 0;
  int ndofs = 0;
};

// Dirichlet u on attributes 1, 3, 4; manufactured traction on the right edge
// (attr 2) so the pressure is uniquely determined.
template <typename Material>
SolveResult SolveManufactured(const cmf::AppConfig &cfg, const Manufactured &m,
                              const Material &material, bool with_body_force)
{
  std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  cmf::MixedSolidMechanicsTL physics(*pmesh, cfg, cmf::MixedMaterial(material));
  mfem::VectorFunctionCoefficient exact_u(2, m.u);
  mfem::FunctionCoefficient exact_p(m.p);
  physics.AddDirichlet({1, 3, 4}, exact_u);
  ManufacturedTraction<Material> traction(m, material);
  physics.AddTraction({2}, traction);
  ManufacturedBodyForce<Material> body(m, material, cfg.material.rho0);
  if (with_body_force) { physics.SetBodyForce(body); }
  physics.Finalize();

  std::unique_ptr<mfem::Solver> linear = physics.MakeLinearSolver(cfg.solver.linear);
  mfem::Vector x(physics.Height());
  x = 0.0;
  cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(physics, *linear, cfg.solver, x);

  SolveResult r;
  r.newton = report.steps.back().newton;
  r.load_step = report.steps.back().step;
  r.ndofs = int(physics.GlobalTrueVSize());
  physics.UpdateFields(x);
  r.u_l2 = physics.Displacement().ComputeL2Error(exact_u);
  r.p_l2 = physics.Pressure().ComputeL2Error(exact_p);
  mfem::ParGridFunction iu(&physics.DisplacementSpace());
  iu.ProjectCoefficient(exact_u);
  iu -= physics.Displacement();
  r.max_nodal_u = iu.Normlinf();
  mfem::ParGridFunction ip(&physics.PressureSpace());
  ip.ProjectCoefficient(exact_p);
  ip -= physics.Pressure();
  r.max_nodal_p = ip.Normlinf();
  return r;
}

template <typename Material>
void PatchTest(const std::string &model, const Material &material, bool incompressible)
{
  const std::string label = model + cmf::VolumetricLawSuffixOf(material);
  cmf::AppConfig cfg = BaseConfig(4, 2, 0.15, model, 0.45, incompressible);
  cfg.solver.load_steps = 3;
  cfg.solver.newton.rtol = 1e-13;
  tensor<double, 2, 2> A;
  double p0 = 0.0;
  if (incompressible)
  {
    A(0, 1) = 0.3; // simple shear, det(I + A) = 1, pressure is free
    p0 = 12.5;
  }
  else
  {
    A(0, 0) = 0.08; A(0, 1) = 0.25;
    A(1, 0) = -0.05; A(1, 1) = -0.06;
    p0 = material.VolumetricPressure(det(cmf::I<2>() + A)); // U'(J) of the material's law
  }
  const Manufactured m = Affine(A, p0);
  SolveResult r = SolveManufactured(cfg, m, material, false);
  std::printf("  patch %s %s (p0 %.4f): max nodal error u %.3e p %.3e, last-step newton its %d, |R|:",
              label.c_str(), incompressible ? "incompressible" : "nu=0.45", p0,
              r.max_nodal_u, r.max_nodal_p, r.newton.iterations);
  for (const cmf::NewtonIteration &it : r.newton.history)
  {
    std::printf(" %.1e%s", it.residual, it.iteration > 0 && it.alpha < 1.0 ? "*" : "");
  }
  std::printf("  (* = damped step)\n");
  CHECK_MSG(r.newton.converged, label + " patch converged");
  CHECK_MSG(r.max_nodal_u <= 1e-12, label + " patch displacement reproduced (" +
            std::to_string(r.max_nodal_u) + ")");
  CHECK_MSG(r.max_nodal_p <= 1e-10 * std::max(1.0, std::abs(p0)),
            label + " patch pressure reproduced (" + std::to_string(r.max_nodal_p) + ")");
}

template <typename Material>
void JacobianTest(const std::string &model, const Material &material, bool incompressible)
{
  cmf::AppConfig cfg = BaseConfig(3, 2, 0.2, model, 0.45, incompressible);
  std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  cmf::MixedSolidMechanicsTL physics(*pmesh, cfg, cmf::MixedMaterial(material));
  const std::string label = model + cmf::VolumetricLawSuffixOf(material);
  const Manufactured m = Isochoric(0.05, 0.04, 3.0);
  mfem::VectorFunctionCoefficient exact_u(2, m.u);
  physics.AddDirichlet({4}, exact_u);
  mfem::Vector t(2);
  t(0) = 1.0; t(1) = 2.0;
  mfem::VectorConstantCoefficient traction(t);
  physics.AddTraction({2}, traction);
  physics.Finalize();
  physics.SetLoadFactor(0.6);
  const int n = physics.Height();
  std::mt19937 rng(11u);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  mfem::Vector x(n), v(n), Jv(n), rp(n), rm(n), xp(n), xm(n);
  for (int i = 0; i < n; i++) { x(i) = 0.03 * unit(rng); v(i) = unit(rng); }
  for (int i = physics.BlockOffsets()[1]; i < n; i++) { x(i) = 5.0 * unit(rng); }
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
  // The pressure rows alone (K_pu, K_pp), which carry the volumetric law.
  const int n_u = physics.BlockOffsets()[1];
  mfem::Vector rp_p(rp.GetData() + n_u, n - n_u), Jv_p(Jv.GetData() + n_u, n - n_u);
  const double rel_p = rp_p.Normlinf() / Jv_p.Normlinf();
  std::printf("  jacobian %s %s: |J v - FD| / |J v| = %.3e (pressure rows %.3e)\n", label.c_str(),
              incompressible ? "incompressible" : "nu=0.45", rel, rel_p);
  CHECK_MSG(rel <= 1e-6, label + " block Jacobian vs FD relative error " + std::to_string(rel));
  CHECK_MSG(rel_p <= 1e-6, label + " pressure rows vs FD relative error " + std::to_string(rel_p));
}

double Seconds()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <typename Material>
void ConvergenceTest(const std::string &label, const Manufactured &m,
                     const Material &material, bool incompressible, double nu,
                     double u_gate, double p_gate, int levels)
{
  const std::string model = cmf::ModelName<Material>();
  const bool linear = cmf::is_small_strain<Material>::value;
  std::cout << "mms " << label << std::endl;
  std::vector<double> eu, ep;
  cmf::NewtonReport finest;
  const int base = 4;
  for (int level = 0; level < levels; level++)
  {
    cmf::AppConfig cfg = BaseConfig(base, 2, 0.15, model, nu, incompressible);
    cfg.mesh.serial_refine = level;
    cfg.solver.load_steps = linear ? 1 : 4; // the zero interior guess inverts first-layer elements at full load
    SolveResult r = SolveManufactured(cfg, m, material, true);
    CHECK_MSG(r.newton.converged, label + " level " + std::to_string(level) + " converged");
    if (linear)
    {
      CHECK_MSG(r.newton.iterations <= 2, label + ": a linear problem, at most two Newton steps, got " +
                std::to_string(r.newton.iterations));
    }
    eu.push_back(r.u_l2);
    ep.push_back(r.p_l2);
    std::printf("  %s nx=%3d dofs %6d: L2 error u %.4e p %.4e, newton its %d (load step %d/%d)%s%s\n",
                label.c_str(), base << level, r.ndofs, r.u_l2, r.p_l2, r.newton.iterations,
                r.load_step, cfg.solver.load_steps, r.newton.failure.empty() ? "" : ": ",
                r.newton.failure.c_str());
    if (!r.newton.converged)
    {
      std::printf("    history:");
      for (const cmf::NewtonIteration &it : r.newton.history)
      {
        std::printf(" %.2e(a=%.3g)", it.residual, it.alpha);
      }
      std::printf("\n");
    }
    finest = r.newton;
  }
  for (std::size_t k = 0; k + 1 < eu.size(); k++)
  {
    const double ru = std::log2(eu[k] / eu[k + 1]);
    const double rp = std::log2(ep[k] / ep[k + 1]);
    std::printf("  %s rate %zu: u %.3f, p %.3f\n", label.c_str(), k + 1, ru, rp);
    CHECK_MSG(ru >= u_gate, label + " u rate " + std::to_string(ru) + " >= " + std::to_string(u_gate));
    CHECK_MSG(rp >= p_gate, label + " p rate " + std::to_string(rp) + " >= " + std::to_string(p_gate));
  }
  std::printf("  %s newton history (finest):", label.c_str());
  for (const cmf::NewtonIteration &it : finest.history) { std::printf(" %.2e", it.residual); }
  std::printf("\n");
}

// At finite kappa the mixed and displacement formulations solve the same
// continuous problem: their |u|_L2 must converge to each other under
// refinement (a clamped/loaded square, nu = 0.45).
void FormulationAgreementTest(int finest_nx, const std::string &model = "iso_neo_hookean")
{
  auto solve = [&](const std::string &formulation, int nx)
  {
    cmf::AppConfig cfg = BaseConfig(nx, 2, 0.0, model, 0.45, false);
    cmf::BoundaryCondition clamp, load;
    clamp.attr = {4};
    clamp.expression = {"0", "0"};
    load.attr = {2};
    load.expression = {"0", "8.0"};
    cfg.bcs.dirichlet.push_back(clamp);
    cfg.bcs.traction.push_back(load);
    cfg.solver.load_steps = 2;
    cfg.formulation = formulation;
    // The elasticity AMG options stall at nu = 0.45 and fall back to these anyway.
    if (model == "linear_elastic") { cfg.solver.linear.amg = "systems"; }
    std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    std::unique_ptr<cmf::SolidProblem> physics = cmf::MakeSolidProblem(*pmesh, cfg);
    physics->Finalize();
    std::unique_ptr<mfem::Solver> linear = physics->MakeLinearSolver(cfg.solver.linear);
    mfem::Vector x(physics->Height());
    x = 0.0;
    cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(*physics, *linear, cfg.solver, x);
    CHECK_MSG(report.converged, formulation + " formulation converged");
    physics->UpdateFields(x);
    mfem::Vector zero(2);
    zero = 0.0;
    mfem::VectorConstantCoefficient zc(zero);
    return physics->Displacement().ComputeL2Error(zc);
  };
  std::vector<double> rel;
  for (int nx = 8; nx <= finest_nx; nx *= 2)
  {
    const double mixed = solve("mixed", nx);
    const double disp = solve("displacement", nx);
    rel.push_back(std::abs(mixed - disp) / disp);
    std::printf("  formulation agreement%s (nu = 0.45, %dx%d p=2): |u|_L2 mixed %.8e displacement %.8e rel %.2e\n",
                model == "iso_neo_hookean" ? "" : (" " + model).c_str(), nx, nx, mixed, disp, rel.back());
  }
  for (std::size_t k = 0; k + 1 < rel.size(); k++)
  {
    CHECK_MSG(rel[k + 1] < rel[k], "mixed vs displacement difference shrinks under refinement");
  }
  CHECK_MSG(rel.back() <= (finest_nx >= 32 ? 2e-3 : 5e-3),
            "mixed vs displacement |u|_L2 agree on the finest mesh (" + std::to_string(rel.back()) + ")");
}

// ------------------------------------------ small strain (linear_elastic)

// Divergence-free u = a (d psi/dY, -d psi/dX) of the stream function
// psi = sin(pi X) sin(pi Y) / pi, with a smooth pressure c sin(pi X) cos(pi Y).
Manufactured StreamFunction(double a, double c)
{
  Manufactured m;
  m.u = [a](const mfem::Vector &X, mfem::Vector &u)
  {
    u.SetSize(2);
    u(0) = a * std::sin(M_PI * X(0)) * std::cos(M_PI * X(1));
    u(1) = -a * std::cos(M_PI * X(0)) * std::sin(M_PI * X(1));
  };
  m.grad = [a](const mfem::Vector &X)
  {
    const double cc = a * M_PI * std::cos(M_PI * X(0)) * std::cos(M_PI * X(1));
    const double ss = a * M_PI * std::sin(M_PI * X(0)) * std::sin(M_PI * X(1));
    tensor<double, 2, 2> H;
    H(0, 0) = cc; H(0, 1) = -ss;
    H(1, 0) = ss; H(1, 1) = -cc;
    return H;
  };
  m.p = [c](const mfem::Vector &X) { return c * std::sin(M_PI * X(0)) * std::cos(M_PI * X(1)); };
  return m;
}

// The smooth compressible field with the small-strain pressure p = kappa div u.
Manufactured SmoothLinear(double alpha, double kappa)
{
  Manufactured m = Smooth(alpha, kappa);
  const auto grad = m.grad;
  m.p = [grad, kappa](const mfem::Vector &X)
  {
    const tensor<double, 2, 2> H = grad(X);
    return kappa * (H(0, 0) + H(1, 1));
  };
  return m;
}

void LinearElasticTests(int levels, int finest_nx)
{
  const double kappa45 = 2.0 * kMu * 1.45 / (3.0 * 0.1);
  const double inf = std::numeric_limits<double>::infinity();
  const cmf::LinearElastic le45(kMu, kappa45), le_inc(kMu, inf);

  // The kernel at a point: with p = kappa tr(eps) the mixed stress is the
  // displacement formulation's; the volume measure is 1 + tr(H), gradient I.
  {
    tensor<double, 3, 3> F = cmf::I<3>();
    const double G[3][3] = {{0.05, -0.03, 0.02}, {0.01, -0.02, 0.04}, {-0.015, 0.025, 0.03}};
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) { F(i, j) += G[i][j]; }
    const double p = kappa45 * (G[0][0] + G[1][1] + G[2][2]);
    const tensor<double, 3, 3> Pm = cmf::MixedPK1(le45, F, p), Pd = le45.PK1(F);
    double diff = 0.0;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) { diff = std::max(diff, std::abs(Pm(i, j) - Pd(i, j))); }
    CHECK_MSG(diff <= 1e-13 * kappa45, "linear_elastic: mixed stress at p = kappa tr(eps) equals PK1");
    tensor<double, 2, 2> H;
    H(0, 0) = 0.05; H(0, 1) = -0.03; H(1, 0) = 0.01; H(1, 1) = -0.02;
    double J = 0.0;
    const tensor<double, 2, 2> Gv = cmf::QPointVolumeGradient<cmf::LinearElastic, 2>(H, J);
    CHECK_CLOSE(J, 1.03, 1e-15);
    CHECK(Gv(0, 0) == 1.0 && Gv(1, 1) == 1.0 && Gv(0, 1) == 0.0 && Gv(1, 0) == 0.0);
    double Jf = 0.0;
    cmf::QPointVolumeGradient<cmf::IsoNeoHookean, 2>(H, Jf);
    CHECK_CLOSE(Jf, 1.05 * 0.98 + 0.03 * 0.01, 1e-15); // det(I + H), unchanged for finite strain
  }

  std::cout << "linear_elastic patch tests" << std::endl;
  for (const bool incompressible : {false, true})
  {
    // Amplitudes of order 0.3: nothing is small, and one load step suffices.
    cmf::AppConfig cfg = BaseConfig(4, 2, 0.15, "linear_elastic", 0.45, incompressible);
    tensor<double, 2, 2> A;
    double p0 = 0.0;
    if (incompressible)
    {
      A(0, 0) = 0.2; A(0, 1) = 0.3; A(1, 0) = -0.1; A(1, 1) = -0.2; // tr A = 0, the pressure is free
      p0 = 12.5;
    }
    else
    {
      A(0, 0) = 0.08; A(0, 1) = 0.25; A(1, 0) = -0.05; A(1, 1) = -0.06;
      p0 = kappa45 * (A(0, 0) + A(1, 1));
    }
    const SolveResult r = SolveManufactured(cfg, Affine(A, p0), incompressible ? le_inc : le45, false);
    std::printf("  patch linear_elastic %s (p0 %.4f): max nodal error u %.3e p %.3e, newton its %d, |R|:",
                incompressible ? "incompressible" : "nu=0.45", p0, r.max_nodal_u, r.max_nodal_p,
                r.newton.iterations);
    for (const cmf::NewtonIteration &it : r.newton.history) { std::printf(" %.1e", it.residual); }
    std::printf("\n");
    const std::string label = std::string("linear_elastic ") + (incompressible ? "incompressible" : "nu=0.45");
    CHECK_MSG(r.newton.converged && r.newton.iterations <= 2, label + " patch: at most two Newton steps");
    CHECK_MSG(r.max_nodal_u <= 1e-12, label + " patch displacement reproduced (" + std::to_string(r.max_nodal_u) + ")");
    CHECK_MSG(r.max_nodal_p <= 1e-10 * std::max(1.0, std::abs(p0)),
              label + " patch pressure reproduced (" + std::to_string(r.max_nodal_p) + ")");
  }

  // The block Jacobian is symmetric and does not depend on the state.
  std::cout << "linear_elastic jacobian" << std::endl;
  for (const bool incompressible : {false, true})
  {
    cmf::AppConfig cfg = BaseConfig(3, 2, 0.2, "linear_elastic", 0.45, incompressible);
    std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    cmf::MixedSolidMechanicsTL physics(*pmesh, cfg, cmf::MixedMaterial(incompressible ? le_inc : le45));
    const Manufactured m = StreamFunction(0.05, 3.0);
    mfem::VectorFunctionCoefficient exact_u(2, m.u);
    physics.AddDirichlet({4}, exact_u);
    physics.Finalize();
    physics.SetLoadFactor(1.0);
    CHECK_MSG(physics.IsLinear(), "linear_elastic mixed problem declares itself linear");
    const int n = physics.Height();
    std::mt19937 rng(17u);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    mfem::Vector x1(n), x2(n), v(n), w(n), J1v(n), J2v(n), J1w(n);
    for (int i = 0; i < n; i++) { x1(i) = unit(rng); x2(i) = 7.0 * unit(rng); v(i) = unit(rng); w(i) = unit(rng); }
    for (int i = 0; i < physics.EssentialTrueDofs().Size(); i++)
    {
      v(physics.EssentialTrueDofs()[i]) = 0.0;
      w(physics.EssentialTrueDofs()[i]) = 0.0;
    }
    physics.GetGradient(x1).Mult(v, J1v);
    physics.GetGradient(x1).Mult(w, J1w);
    physics.GetGradient(x2).Mult(v, J2v);
    mfem::Vector d(J1v);
    d -= J2v;
    const double state = d.Norml2() / J1v.Norml2();
    const double symmetry = std::abs((w * J1v) - (v * J1w)) / (w.Norml2() * J1v.Norml2());
    std::printf("  jacobian linear_elastic %s: state dependence %.2e, asymmetry %.2e\n",
                incompressible ? "incompressible" : "nu=0.45", state, symmetry);
    CHECK_MSG(state <= 1e-13, "linear_elastic mixed Jacobian is independent of the state");
    CHECK_MSG(symmetry <= 1e-13, "linear_elastic mixed Jacobian is symmetric");
    mfem::ConstantCoefficient pr(1.0);
    CHECK_THROWS(physics.AddPressure({2}, pr, true), cmf::ConfigError,
                 "follower_pressure is not used by model 'linear_elastic'");
  }

  // Amplitude 0.2: |Grad u| ~ 0.6, far beyond what the finite-strain MMS can take.
  ConvergenceTest("linear_elastic nu=0.45", SmoothLinear(0.2, kappa45), le45, false, 0.45, 2.9, 1.9, levels);
  ConvergenceTest("linear_elastic incompressible", StreamFunction(0.1, 6.0), le_inc, true, 0.5, 2.9, 1.9, levels);
  FormulationAgreementTest(finest_nx, "linear_elastic");

  // Options that do not exist at small strain, through the factory.
  {
    cmf::AppConfig cfg = BaseConfig(2, 2, 0.0, "linear_elastic", 0.45, true);
    std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    std::unique_ptr<cmf::SolidProblem> ok = cmf::MakeSolidProblem(*pmesh, cfg);
    CHECK_MSG(ok->Description().find("linear_elastic (small strain)") != std::string::npos &&
              ok->Description().find("incompressible") != std::string::npos, ok->Description());
    cfg.solver.predictor = "tangent";
    CHECK_THROWS(cmf::MakeSolidProblem(*pmesh, cfg), cmf::ConfigError,
                 "'solver.predictor': tangent is not used by model 'linear_elastic'");
    cfg.solver.predictor = "none";
    cfg.formulation = "displacement";
    CHECK_THROWS(cmf::MakeSolidProblem(*pmesh, cfg), cmf::ConfigError,
                 "'linear_elastic' is incompressible; use formulation: mixed");
  }
}

} // namespace

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  if (mfem::Mpi::WorldSize() != 1)
  {
    if (mfem::Mpi::Root()) { std::cout << "test_mixed is a serial test" << std::endl; }
    return 1;
  }
  // Fast subset for make check (MMS up to 16x16, agreement up to 16x16);
  // --full adds the 32x32 levels (make test).
  bool full = false;
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&full, "-f", "--full", "-no-f", "--no-full", "Run the 32x32 levels as well.");
  args.Parse();
  if (!args.Good()) { args.PrintUsage(std::cout); return 1; }
  const int levels = full ? 4 : 3;
  const int finest_nx = full ? 32 : 16;

  const double kappa45 = 2.0 * kMu * 1.45 / (3.0 * 0.1);
  const cmf::IsoNeoHookean nh45(kMu, kappa45);
  const cmf::IsoNeoHookean nh_inc(kMu, std::numeric_limits<double>::infinity());
  const cmf::MooneyRivlin mr45(0.3 * kMu, 0.2 * kMu, kappa45);
  const cmf::MooneyRivlin mr_inc(0.3 * kMu, 0.2 * kMu, std::numeric_limits<double>::infinity());
  // Non-quadratic volumetric laws: the constraint u'(J) = p / kappa and the
  // non-symmetric K_pu = u''(J) K_up^T of the general form.
  cmf::IsoNeoHookean nh45_log(kMu, kappa45), nh45_st(kMu, kappa45);
  nh45_log.law = cmf::VolumetricLaw::Logarithmic;
  nh45_st.law = cmf::VolumetricLaw::SimoTaylor;
  cmf::MooneyRivlin mr45_jlj(0.3 * kMu, 0.2 * kMu, kappa45);
  mr45_jlj.law = cmf::VolumetricLaw::JLogJ;

  double t0 = Seconds();
  std::cout << "patch tests" << std::endl;
  PatchTest("iso_neo_hookean", nh45, false);
  PatchTest("iso_neo_hookean", nh_inc, true);
  PatchTest("mooney_rivlin", mr45, false);
  PatchTest("mooney_rivlin", mr_inc, true);
  PatchTest("iso_neo_hookean", nh45_log, false);
  PatchTest("iso_neo_hookean", nh45_st, false);
  PatchTest("mooney_rivlin", mr45_jlj, false);
  std::printf("  [%.1f s]\n", Seconds() - t0);
  t0 = Seconds();
  std::cout << "jacobian consistency" << std::endl;
  JacobianTest("iso_neo_hookean", nh45, false);
  JacobianTest("iso_neo_hookean", nh_inc, true);
  JacobianTest("mooney_rivlin", mr_inc, true);
  JacobianTest("iso_neo_hookean", nh45_log, false);
  JacobianTest("mooney_rivlin", mr45_jlj, false);
  std::printf("  [%.1f s]\n", Seconds() - t0);
  t0 = Seconds();
  // Amplitude 0.02 (max |Grad u| ~ 0.06): at kappa/mu ~ 10 the pressure
  // kappa (J - 1) already reaches ~0.8 mu, which is what makes the load
  // increments strongly nonlinear for Newton.
  ConvergenceTest("near-incompressible nu=0.45", Smooth(0.02, kappa45), nh45, false, 0.45, 2.9, 1.9, levels);
  std::printf("  [%.1f s]\n", Seconds() - t0);
  t0 = Seconds();
  ConvergenceTest("incompressible", Isochoric(0.04, 0.03, 6.0), nh_inc, true, 0.5, 2.9, 1.9, levels);
  std::printf("  [%.1f s]\n", Seconds() - t0);
  t0 = Seconds();
  FormulationAgreementTest(finest_nx);
  std::printf("  [%.1f s]\n", Seconds() - t0);
  t0 = Seconds();
  LinearElasticTests(levels, finest_nx);
  std::printf("  [%.1f s]\n", Seconds() - t0);
  return cmf_test::Report("test_mixed");
}

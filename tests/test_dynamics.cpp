// DY1 gate: the inertial term and implicit time stepping in the displacement
// formulation. Mass matrix with the density by region, the free-fall patch
// test, temporal order without spatial error, energy conservation and
// numerical dissipation, the energy and momentum balances with loads and a
// moving support, the nonlinear path (manufactured solution, Newton order,
// self-convergence), one Jacobian and one solver setup per run of a linear
// problem, the stepper in physical time, the path from a YAML input, and the
// mixed u-p formulation (DY4): orders of u and p, the pressure mode, p_0.
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "base/fields.hpp"
#include "base/mesh_input.hpp"
#include "base/probes.hpp"
#include "kernels/total_lagrangian.hpp"
#include "materials/materials.hpp"
#include "mfem.hpp"
#include "physics/dynamic_solid_problem.hpp"
#include "physics/mixed_solid_mechanics_tl.hpp"
#include "physics/solid_mechanics_tl.hpp"
#include "physics/solid_problem.hpp"
#include "solvers/linear_solver.hpp"
#include "solvers/quasi_static.hpp"
#include "solvers/time_integration.hpp"
#include "test_util.hpp"

using cmf::tensor;

namespace
{

const double kE = 250.0, kNu = 0.3;
const double kRho = 2.5, kRhoRegion = 7.5; // two densities: the mass and the body force must agree on them

cmf::AppConfig BaseConfig(int dim, const std::string &element, int n, int order, double perturb,
                          const std::string &model)
{
  cmf::AppConfig cfg;
  cfg.mesh.cartesian = true;
  cfg.mesh.box.dim = dim;
  cfg.mesh.box.element = element;
  cfg.mesh.box.nx = cfg.mesh.box.ny = cfg.mesh.box.nz = n;
  cfg.mesh.order = order;
  cfg.mesh.perturb = perturb;
  cfg.material.model = model;
  cfg.material.E = kE;
  cfg.material.nu = kNu;
  cfg.material.rho0 = kRho;
  cfg.solver.newton.rtol = 1e-10;
  cfg.solver.newton.atol = 1e-14;
  cfg.solver.newton.max_it = 25;
  cfg.solver.newton.print_level = 0;
  cfg.solver.linear.type = "gmres_amg";
  cfg.solver.linear.amg = "systems";
  cfg.solver.linear.rtol = 1e-14;
  cfg.solver.linear.max_it = 1000;
  return cfg;
}

// Element attribute 2 where the centroid has x > 1/2 (attribute 1 elsewhere),
// and a material region with its own density on it.
std::unique_ptr<mfem::ParMesh> BuildMesh(cmf::AppConfig &cfg, bool two_regions)
{
  std::unique_ptr<mfem::Mesh> serial = cmf::BuildSerialMesh(cfg.mesh);
  if (two_regions)
  {
    mfem::Vector c;
    for (int e = 0; e < serial->GetNE(); e++)
    {
      serial->GetElementCenter(e, c);
      serial->SetAttribute(e, c(0) > 0.5 ? 2 : 1);
    }
    serial->SetAttributes();
    cmf::MaterialConfig region = cfg.material;
    region.attr = {2};
    region.rho0 = kRhoRegion;
    cfg.material.regions = {region};
  }
  return std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, *serial);
}

cmf::DynamicsConfig Scheme(const std::string &scheme, double parameter = 0.0)
{
  cmf::DynamicsConfig d;
  d.enabled = true;
  d.scheme = scheme;
  if (scheme == "hht") { d.alpha = parameter; }
  if (scheme == "generalized_alpha") { d.rho_inf = parameter; }
  return d;
}

cmf::BCOptions Constant(bool time_dependent = false)
{
  cmf::BCOptions opt;
  opt.schedule = cmf::Schedule::Constant();
  opt.time_dependent = time_dependent;
  return opt;
}

mfem::Vector TrueDofs(mfem::ParFiniteElementSpace &fes, mfem::VectorCoefficient &c, double t)
{
  c.SetTime(t);
  mfem::ParGridFunction g(&fes);
  g.ProjectCoefficient(c);
  mfem::Vector tv;
  g.GetTrueDofs(tv);
  return tv;
}

double MaxDiff(const mfem::Vector &a, const mfem::Vector &b)
{
  double d = 0.0;
  for (int i = 0; i < a.Size(); i++) { d = std::max(d, std::abs(a(i) - b(i))); }
  return d;
}

// ---------------------------------------------------------------------------
// 1. The mass matrix.
void MassTest()
{
  for (const std::string element : {"quad", "tri", "hex", "tet"})
    for (int order = 1; order <= 2; order++)
    {
      const int dim = (element == "quad" || element == "tri") ? 2 : 3;
      cmf::AppConfig cfg = BaseConfig(dim, element, 3, order, 0.15, "linear_elastic");
      std::unique_ptr<mfem::ParMesh> mesh = BuildMesh(cfg, true);
      // sum_r rho_r V_r, the volumes by a rule that is exact for the
      // Jacobian determinant of the perturbed (bi- and trilinear) elements.
      double mass = 0.0;
      for (int e = 0; e < mesh->GetNE(); e++)
      {
        mfem::ElementTransformation &T = *mesh->GetElementTransformation(e);
        const mfem::IntegrationRule &ir = mfem::IntRules.Get(mesh->GetElementBaseGeometry(e), 6);
        double volume = 0.0;
        for (int q = 0; q < ir.GetNPoints(); q++)
        {
          T.SetIntPoint(&ir.IntPoint(q));
          volume += ir.IntPoint(q).weight * T.Weight();
        }
        mass += (mesh->GetAttribute(e) == 2 ? kRhoRegion : kRho) * volume;
      }
      std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
      mfem::Vector zero(dim);
      zero = 0.0;
      mfem::VectorConstantCoefficient fixed(zero);
      problem->AddDirichlet({1}, fixed);
      cmf::DynamicSolidProblem dyn(*problem, Scheme("newmark"));
      mfem::Vector x(problem->Height());
      x = 0.0;
      dyn.Initialize(x);

      const int n = x.Size();
      mfem::Vector e0(n), e1(n), Me(n);
      e0 = 0.0;
      e1 = 0.0;
      for (int i = 0; i < n; i++)
      {
        if (i % dim == 0) { e0(i) = 1.0; }
        if (i % dim == 1) { e1(i) = 1.0; }
      }
      dyn.Mass().Mult(e0, Me);
      const double m00 = Me * e0, m01 = Me * e1;
      std::mt19937 rng(11u);
      std::uniform_real_distribution<double> unit(-1.0, 1.0);
      mfem::Vector v(n), w(n), Mv(n), Mw(n);
      for (int i = 0; i < n; i++) { v(i) = unit(rng); w(i) = unit(rng); }
      dyn.Mass().Mult(v, Mv);
      dyn.Mass().Mult(w, Mw);
      const double asym = std::abs(w * Mv - v * Mw) / (w.Norml2() * Mv.Norml2());

      // The eliminated copy: zero essential rows, and on the free rows the
      // action of M on a vector that vanishes on the essential dofs.
      const mfem::Array<int> &ess = problem->EssentialTrueDofs();
      mfem::Vector v_free(v), Mev(n), Mvf(n);
      for (int i = 0; i < ess.Size(); i++) { v_free(ess[i]) = 0.0; }
      dyn.EliminatedMass().Mult(v, Mev);
      dyn.Mass().Mult(v_free, Mvf);
      double ess_rows = 0.0;
      for (int i = 0; i < ess.Size(); i++)
      {
        ess_rows = std::max(ess_rows, std::abs(Mev(ess[i])));
        Mvf(ess[i]) = 0.0;
      }
      const double elim = MaxDiff(Mev, Mvf) / Mvf.Normlinf();

      std::printf("  mass %s p=%d: 1.M.1 = %.15e, sum rho V = %.15e, cross %.1e, asymmetry %.1e, "
                  "eliminated rows %.1e, eliminated action %.1e\n", element.c_str(), order, m00,
                  mass, std::abs(m01), asym, ess_rows, elim);
      const std::string tag = element + " p=" + std::to_string(order);
      CHECK_MSG(std::abs(m00 - mass) <= 1e-13 * mass, tag + ": total mass with two densities");
      CHECK_MSG(std::abs(m01) <= 1e-13 * mass, tag + ": no mass between the components");
      CHECK_MSG(asym <= 1e-13, tag + ": symmetric mass matrix");
      CHECK_MSG(ess_rows == 0.0 && ess.Size() > 0, tag + ": eliminated mass has zero essential rows");
      CHECK_MSG(elim <= 1e-13, tag + ": eliminated mass acts as M on the free dofs");
    }
}

// ---------------------------------------------------------------------------
// 2. Free fall: no supports, a body force g, two densities, any material. The
// trapezoidal rule is exact for a constant acceleration, so u = g t^2 / 2 at
// every node on any mesh, and the mass and the body force must agree on rho.
void FreeFallTest()
{
  struct Variant { std::string element, model; int dim, order; };
  const std::vector<Variant> variants = {{"quad", "linear_elastic", 2, 2},
                                         {"quad", "neo_hookean", 2, 2},
                                         {"tet", "neo_hookean", 3, 1}};
  for (const Variant &var : variants)
  {
    cmf::AppConfig cfg = BaseConfig(var.dim, var.element, 3, var.order, 0.15, var.model);
    std::unique_ptr<mfem::ParMesh> mesh = BuildMesh(cfg, true);
    std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
    mfem::Vector g(var.dim);
    g = 0.0;
    g(0) = 0.3;
    g(var.dim - 1) = -9.81;
    mfem::VectorConstantCoefficient gravity(g);
    problem->SetBodyForce(gravity, Constant());
    cmf::DynamicSolidProblem dyn(*problem, Scheme("newmark"));
    mfem::Vector x(problem->Height());
    x = 0.0;
    dyn.Initialize(x);

    const int n = x.Size();
    double a0_err = 0.0;
    for (int i = 0; i < n; i++)
    {
      a0_err = std::max(a0_err, std::abs(dyn.Acceleration()(i) - g(i % var.dim)));
    }
    std::unique_ptr<mfem::Solver> linear = dyn.MakeLinearSolver(cfg.solver.linear);
    double worst_u = 0.0, worst_a = 0.0;
    const cmf::QuasiStaticReport report = cmf::SolveDynamic(
      dyn, *linear, cfg.solver, cmf::UniformTimeSteps(0.5, 50), 0.0, x,
      [&](const cmf::LoadStepReport &s, const mfem::Vector &u)
      {
        const double t = s.load_factor;
        for (int i = 0; i < n; i++)
        {
          const double exact = 0.5 * g(i % var.dim) * t * t;
          worst_u = std::max(worst_u, std::abs(u(i) - exact) / (0.5 * 9.81 * t * t));
          worst_a = std::max(worst_a, std::abs(dyn.Acceleration()(i) - g(i % var.dim)) / 9.81);
        }
      });
    const double strain_energy = problem->InternalEnergy(x);
    const double kinetic = dyn.KineticEnergy();
    std::printf("  free fall %s %s p=%d: |a_0 - g| %.1e, max |u - g t^2/2| / (g t^2/2) %.1e, "
                "max |a - g| / g %.1e, strain energy / kinetic %.1e\n", var.element.c_str(),
                var.model.c_str(), var.order, a0_err, worst_u, worst_a, strain_energy / kinetic);
    const std::string tag = "free fall " + var.element + " " + var.model;
    CHECK_MSG(report.converged && report.steps.size() == 50, tag + ": 50 steps converged");
    CHECK_MSG(a0_err <= 1e-11, tag + ": the initial acceleration is g (step load on at t = 0)");
    CHECK_MSG(worst_u <= 1e-11, tag + ": u = g t^2 / 2 at every node");
    CHECK_MSG(worst_a <= 1e-8, tag + ": a = g at every node");
    CHECK_MSG(std::abs(strain_energy) <= 1e-12 * kinetic, tag + ": no strain");
  }
}

// ---------------------------------------------------------------------------
// 3. Temporal order without spatial error: u = sin(w t) U(X) with U quadratic,
// which the p = 2 space on an affine mesh holds exactly, so the semi-discrete
// system has this solution and the error is that of the time integrator. The
// error at a fixed time carries the free vibrations that the truncation error
// excites, each with the phase of its numerical frequency, so the ratios are
// only clean once dt resolves every mode of the mesh: 2 x 2 elements, whose
// highest frequency is about 150 rad/s, and steps from 2e-3 down.
// U vanishes on the left side: U = (0.02 x^2 + 0.01 x y, 0.015 x y - 0.01 x^2).
struct QuadraticField
{
  double w = 5.0;
  double lambda, mu, rho;

  static void U(const mfem::Vector &X, mfem::Vector &u)
  {
    const double x = X(0), y = X(1);
    u.SetSize(2);
    u(0) = 0.02 * x * x + 0.01 * x * y;
    u(1) = 0.015 * x * y - 0.01 * x * x;
  }
  // sigma(U) = lambda tr(eps) I + 2 mu eps, eps = sym(grad U).
  void Stress(const mfem::Vector &X, double s[2][2]) const
  {
    const double x = X(0), y = X(1);
    const double H[2][2] = {{0.04 * x + 0.01 * y, 0.01 * x}, {0.015 * y - 0.02 * x, 0.015 * x}};
    const double tr = H[0][0] + H[1][1];
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
      {
        s[i][j] = mu * (H[i][j] + H[j][i]) + (i == j ? lambda * tr : 0.0);
      }
  }
  // Per unit mass: b = u_tt - Div sigma / rho, with Div sigma(U) = (lambda + mu)
  // grad(div U) + mu lap(U) = (lambda + mu) (0.055, 0.01) + mu (0.04, -0.02).
  void BodyForce(const mfem::Vector &X, double t, mfem::Vector &b) const
  {
    mfem::Vector u;
    U(X, u);
    const double div_sigma[2] = {(lambda + mu) * 0.055 + mu * 0.04, (lambda + mu) * 0.01 - mu * 0.02};
    b.SetSize(2);
    for (int i = 0; i < 2; i++) { b(i) = std::sin(w * t) * (-w * w * u(i) - div_sigma[i] / rho); }
  }
  // Nominal traction on the sides of the unit square other than the left one.
  void Traction(const mfem::Vector &X, double t, mfem::Vector &T) const
  {
    double s[2][2];
    Stress(X, s);
    double N[2] = {0.0, 0.0};
    if (X(1) < 1e-12) { N[1] = -1.0; }
    else if (X(1) > 1.0 - 1e-12) { N[1] = 1.0; }
    else { N[0] = 1.0; }
    T.SetSize(2);
    for (int i = 0; i < 2; i++) { T(i) = std::sin(w * t) * (s[i][0] * N[0] + s[i][1] * N[1]); }
  }
};

void TemporalOrderTest()
{
  const cmf::LameParameters lame = cmf::LameFromYoungPoisson(kE, kNu);
  QuadraticField f;
  f.lambda = lame.lambda;
  f.mu = lame.mu;
  f.rho = kRho;
  const double t_final = 0.8;
  mfem::VectorFunctionCoefficient exact_u(
    2, [&f](const mfem::Vector &X, double t, mfem::Vector &u)
    { QuadraticField::U(X, u); u *= std::sin(f.w * t); });
  mfem::VectorFunctionCoefficient exact_v(
    2, [&f](const mfem::Vector &X, double t, mfem::Vector &u)
    { QuadraticField::U(X, u); u *= f.w * std::cos(f.w * t); });
  mfem::VectorFunctionCoefficient body(
    2, [&f](const mfem::Vector &X, double t, mfem::Vector &b) { f.BodyForce(X, t, b); });
  mfem::VectorFunctionCoefficient traction(
    2, [&f](const mfem::Vector &X, double t, mfem::Vector &T) { f.Traction(X, t, T); });

  struct SchemeCase { std::string name; double parameter; };
  const std::vector<SchemeCase> schemes = {{"newmark", 0.0}, {"hht", 0.1}, {"generalized_alpha", 0.8}};
  for (int variant = 0; variant < 2; variant++)
    for (const SchemeCase &sc : schemes)
    {
      std::vector<double> err_u, err_v;
      for (int level = 0; level < 4; level++)
      {
        const int steps = 400 << level;
        cmf::AppConfig cfg = BaseConfig(2, "quad", 2, 2, 0.0, "linear_elastic");
        cfg.solver.linear.type = "cg_amg";
        std::unique_ptr<mfem::ParMesh> mesh = BuildMesh(cfg, false);
        std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
        if (variant == 0)
        {
          // U = 0 on the left; exact tractions on the other sides.
          problem->AddDirichlet({4}, exact_u, Constant());
          problem->AddTraction({1, 2, 3}, traction, Constant(true));
        }
        else { problem->AddDirichlet({1, 2, 3, 4}, exact_u, Constant()); } // prescribed motion
        problem->SetBodyForce(body, Constant(true));
        cmf::DynamicSolidProblem dyn(*problem, Scheme(sc.name, sc.parameter));
        dyn.SetInitialVelocity(exact_v);
        mfem::Vector x(problem->Height());
        x = 0.0;
        dyn.Initialize(x);
        std::unique_ptr<mfem::Solver> linear = dyn.MakeLinearSolver(cfg.solver.linear);
        const cmf::QuasiStaticReport report = cmf::SolveDynamic(
          dyn, *linear, cfg.solver, cmf::UniformTimeSteps(t_final, steps), 0.0, x);
        CHECK_MSG(report.converged, sc.name + ": every time step converged");
        err_u.push_back(MaxDiff(x, TrueDofs(problem->DisplacementSpace(), exact_u, t_final)));
        err_v.push_back(MaxDiff(dyn.Velocity(),
                                TrueDofs(problem->DisplacementSpace(), exact_v, t_final)));
      }
      std::printf("  temporal order, %s, %s:", variant == 0 ? "tractions" : "prescribed motion",
                  sc.name.c_str());
      for (std::size_t k = 0; k + 1 < err_u.size(); k++)
      {
        const double ru = std::log2(err_u[k] / err_u[k + 1]), rv = std::log2(err_v[k] / err_v[k + 1]);
        std::printf(" u %.3f v %.3f;", ru, rv);
        const std::string tag = sc.name + " variant " + std::to_string(variant);
        const double band = k + 2 == err_u.size() ? 0.02 : 0.1; // the finest pair is the asymptotic one
        CHECK_MSG(std::abs(ru - 2.0) <= band, tag + ": order 2 in dt for u, got " + std::to_string(ru));
        CHECK_MSG(std::abs(rv - 2.0) <= band, tag + ": order 2 in dt for v, got " + std::to_string(rv));
      }
      std::printf(" errors u %.3e -> %.3e\n", err_u.front(), err_u.back());
    }
}

// ---------------------------------------------------------------------------
// A block clamped on its left side: the common problem of tests 4, 5, 6, 8, 9.
struct ClampedBlock
{
  cmf::AppConfig cfg;
  std::unique_ptr<mfem::ParMesh> mesh;
  std::unique_ptr<cmf::SolidProblem> problem;
  std::unique_ptr<mfem::VectorCoefficient> fixed, initial;

  ClampedBlock(const std::string &model, int n, int order)
  {
    cfg = BaseConfig(2, "quad", n, order, 0.1, model);
    mesh = BuildMesh(cfg, false);
    problem = cmf::MakeSolidProblem(*mesh, cfg);
    mfem::Vector zero(2);
    zero = 0.0;
    fixed = std::make_unique<mfem::VectorConstantCoefficient>(zero);
    initial = std::make_unique<mfem::VectorFunctionCoefficient>(
      2, [](const mfem::Vector &X, mfem::Vector &u) { QuadraticField::U(X, u); });
  }
};

// 4. Free vibration of a linear problem: the trapezoidal rule conserves the
// energy exactly; rho_inf < 1 removes it, at infinite frequency by rho_inf^2
// per step (a defective root, hence a slope and not a ratio).
void EnergyTest()
{
  auto run = [](const cmf::DynamicsConfig &scheme, double dt, int steps, double atol,
                std::vector<double> &energy)
  {
    ClampedBlock b("linear_elastic", 4, 2);
    b.cfg.solver.linear.type = "cg_amg";
    b.cfg.solver.newton.atol = atol;
    b.problem->AddDirichlet({4}, *b.fixed);
    cmf::DynamicSolidProblem dyn(*b.problem, scheme);
    dyn.SetInitialDisplacement(*b.initial);
    mfem::Vector x(b.problem->Height());
    x = 0.0;
    dyn.Initialize(x);
    energy.assign(1, dyn.KineticEnergy() + b.problem->InternalEnergy(x));
    std::unique_ptr<mfem::Solver> linear = dyn.MakeLinearSolver(b.cfg.solver.linear);
    const cmf::QuasiStaticReport report = cmf::SolveDynamic(
      dyn, *linear, b.cfg.solver, cmf::UniformTimeSteps(dt * steps, steps), 0.0, x,
      [&](const cmf::LoadStepReport &, const mfem::Vector &u)
      { energy.push_back(dyn.KineticEnergy() + b.problem->InternalEnergy(u)); });
    return report.converged;
  };

  std::vector<double> e;
  CHECK_MSG(run(Scheme("newmark"), 0.002, 500, 1e-14, e), "trapezoidal rule: 500 steps converged");
  double drift = 0.0;
  for (double v : e) { drift = std::max(drift, std::abs(v - e[0]) / e[0]); }
  std::printf("  energy, trapezoidal rule, 500 steps: E_0 = %.6e, max |E_n - E_0| / E_0 = %.2e\n", e[0], drift);
  CHECK_MSG(drift <= 1e-10, "the trapezoidal rule conserves the energy of a linear problem");

  // The scheme dissipates in its own norm, not step by step in the physical
  // energy, which may rise between two steps but not above its initial value.
  CHECK_MSG(run(Scheme("generalized_alpha", 0.8), 0.002, 500, 1e-14, e), "rho_inf 0.8: 500 steps converged");
  double highest = 0.0, largest_rise = 0.0;
  for (std::size_t k = 0; k + 1 < e.size(); k++)
  {
    highest = std::max(highest, e[k + 1] / e[0]);
    largest_rise = std::max(largest_rise, (e[k + 1] - e[k]) / e[0]);
  }
  std::printf("  energy, generalized-alpha rho_inf 0.8, 500 steps: E_end / E_0 = %.6f, max E_n / E_0 = %.12f, "
              "largest rise between two steps %.2e E_0\n", e.back() / e[0], highest, largest_rise);
  CHECK_MSG(highest <= 1.0 + 1e-12, "rho_inf 0.8: the energy never exceeds its initial value");
  CHECK_MSG(e.back() < e[0] && e.back() > 0.5 * e[0], "rho_inf 0.8: little dissipation of a resolved motion");

  // Every mode in the high-frequency limit: dt = 100 against periods below one.
  CHECK_MSG(run(Scheme("generalized_alpha", 0.0), 100.0, 10, 0.0, e), "rho_inf 0: steps converged");
  std::printf("  energy, rho_inf 0, w dt >> 1: E_10 / E_0 = %.2e\n", e[10] / e[0]);
  CHECK_MSG(e[10] <= 1e-6 * e[0], "rho_inf 0 annihilates the high-frequency response");
  CHECK_MSG(run(Scheme("generalized_alpha", 0.5), 100.0, 60, 0.0, e), "rho_inf 0.5: steps converged");
  const double slope = std::log(e[60] / e[20]) / 40.0;
  std::printf("  energy, rho_inf 0.5, w dt >> 1: slope of ln E = %.4f, 2 ln rho_inf = %.4f\n", slope,
              2.0 * std::log(0.5));
  CHECK_MSG(std::abs(slope / (2.0 * std::log(0.5)) - 1.0) <= 0.1, "the energy decays as rho_inf^(2n)");
  CHECK_MSG(run(Scheme("generalized_alpha", 1.0), 100.0, 60, 0.0, e), "rho_inf 1: steps converged");
  std::printf("  energy, rho_inf 1, w dt >> 1: E_60 / E_0 = %.12f\n", e[60] / e[0]);
  CHECK_MSG(std::abs(e[60] / e[0] - 1.0) <= 1e-8, "rho_inf 1 has no dissipation");

  // The parameter map itself.
  const cmf::TimeIntegration ga = cmf::MakeTimeIntegration(Scheme("generalized_alpha", 0.6));
  CHECK_CLOSE(ga.SpectralRadiusAtInfinity(), 0.6, 1e-14);
  CHECK_CLOSE(ga.gamma, 0.5 - ga.alpha_m + ga.alpha_f, 1e-15);
  const cmf::TimeIntegration hht = cmf::MakeTimeIntegration(Scheme("hht", 0.2));
  CHECK_CLOSE(hht.SpectralRadiusAtInfinity(), 0.8 / 1.2, 1e-14);
  CHECK_CLOSE(cmf::MakeTimeIntegration(Scheme("newmark")).SpectralRadiusAtInfinity(), 1.0, 1e-14);
  CHECK(!cmf::MakeTimeIntegration(Scheme("newmark")).Dissipative() && ga.Dissipative());
  CHECK_THROWS(cmf::MakeTimeIntegration(Scheme("generalized_alpha", 1.5)), cmf::ConfigError, "rho_inf");
  CHECK_THROWS(cmf::MakeTimeIntegration(Scheme("hht", 0.5)), cmf::ConfigError, "alpha");
  CHECK_THROWS(cmf::MakeTimeIntegration(Scheme("leapfrog")), cmf::ConfigError, "unknown scheme");
  cmf::DynamicsConfig bad = Scheme("newmark");
  bad.beta = 0.0;
  CHECK_THROWS(cmf::MakeTimeIntegration(bad), cmf::ConfigError, "beta");
}

// 5, 6. A step traction and a moving support: with the trapezoidal rule on a
// linear problem, kinetic + internal - external work is constant, and the
// reactions plus the external force equal the resultant of M a.
void BalanceTest()
{
  ClampedBlock b("linear_elastic", 4, 2);
  b.cfg.solver.linear.type = "cg_amg";
  mfem::VectorFunctionCoefficient support(
    2, [](const mfem::Vector &, double t, mfem::Vector &u)
    { u.SetSize(2); u(0) = 0.01 * (1.0 - std::cos(5.0 * t)); u(1) = 0.004 * std::sin(3.0 * t) * std::sin(3.0 * t); });
  mfem::Vector T(2);
  T(0) = 0.2;
  T(1) = 0.5;
  mfem::VectorConstantCoefficient traction(T);
  cmf::BCOptions named = Constant();
  named.name = "support";
  b.problem->AddDirichlet({4}, support, named);
  b.problem->AddTraction({2}, traction, Constant());
  cmf::DynamicSolidProblem dyn(*b.problem, Scheme("newmark"));
  mfem::Vector x(b.problem->Height());
  x = 0.0;
  dyn.Initialize(x);
  const double e0 = dyn.KineticEnergy() + b.problem->InternalEnergy(x);
  std::unique_ptr<mfem::Solver> linear = dyn.MakeLinearSolver(b.cfg.solver.linear);
  double worst_energy = 0.0, worst_momentum = 0.0, peak = 0.0, force_scale = 0.0;
  const cmf::QuasiStaticReport report = cmf::SolveDynamic(
    dyn, *linear, b.cfg.solver, cmf::UniformTimeSteps(0.4, 200), 0.0, x,
    [&](const cmf::LoadStepReport &, const mfem::Vector &u)
    {
      const double kinetic = dyn.KineticEnergy(), internal = b.problem->InternalEnergy(u);
      peak = std::max(peak, kinetic + internal);
      worst_energy = std::max(worst_energy, std::abs(kinetic + internal - dyn.ExternalWork() - e0));
      const std::vector<cmf::Reaction> rx = dyn.Reactions();
      const std::vector<double> inertia = dyn.InertialForce();
      const mfem::Vector &f = b.problem->Loads().ExternalLoad();
      for (int c = 0; c < 2; c++)
      {
        double external = 0.0;
        for (int i = c; i < f.Size(); i += 2) { external += f(i); }
        force_scale = std::max(force_scale, std::abs(rx[0].force[c]));
        worst_momentum = std::max(worst_momentum, std::abs(rx[0].force[c] + external - inertia[std::size_t(c)]));
      }
    });
  std::printf("  balances, 200 steps: |kinetic + internal - work - E_0| / peak energy = %.2e, "
              "|reaction + external - 1.(M a)| / |reaction| = %.2e, work %.4e, peak energy %.4e\n",
              worst_energy / peak, worst_momentum / force_scale, dyn.ExternalWork(), peak);
  CHECK_MSG(report.converged, "balances: every step converged");
  CHECK_MSG(dyn.ExternalWork() > 0.1 * peak, "the loads and the support do work");
  CHECK_MSG(worst_energy <= 1e-10 * peak, "energy balance with a step traction and a moving support");
  CHECK_MSG(worst_momentum <= 1e-10 * force_scale, "global momentum balance with inertia");
}

// ---------------------------------------------------------------------------
// 7. The nonlinear path.
struct Manufactured
{
  double alpha, w;
  void u(const mfem::Vector &X, double t, mfem::Vector &v) const
  {
    v.SetSize(2);
    v(0) = alpha * std::sin(M_PI * X(0)) * std::sin(M_PI * X(1));
    v(1) = alpha * X(0) * X(0) * X(1) * (1.0 - X(1));
    v *= std::sin(w * t);
  }
  tensor<double, 2, 2> grad(const mfem::Vector &X, double t) const
  {
    tensor<double, 2, 2> H;
    H(0, 0) = alpha * M_PI * std::cos(M_PI * X(0)) * std::sin(M_PI * X(1));
    H(0, 1) = alpha * M_PI * std::sin(M_PI * X(0)) * std::cos(M_PI * X(1));
    H(1, 0) = alpha * 2.0 * X(0) * X(1) * (1.0 - X(1));
    H(1, 1) = alpha * X(0) * X(0) * (1.0 - 2.0 * X(1));
    return std::sin(w * t) * H;
  }
};

// b = u_tt - Div P / rho per unit mass, Div P by central differences in X
// (step 1e-5) of P(X) = PK1(I + Grad u(X, t)).
template <typename Material>
class ManufacturedBodyForce : public mfem::VectorCoefficient
{
public:
  ManufacturedBodyForce(const Manufactured &m, const Material &mat, double rho)
    : mfem::VectorCoefficient(2), m_(m), mat_(mat), rho_(rho) {}

  void Eval(mfem::Vector &b, mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip) override
  {
    mfem::Vector X(2), u;
    T.Transform(ip, X);
    const double h = 1e-5, t = GetTime();
    b.SetSize(2);
    b = 0.0;
    for (int j = 0; j < 2; j++)
    {
      mfem::Vector Xp(X), Xm(X);
      Xp(j) += h;
      Xm(j) -= h;
      const tensor<double, 2, 2> Pp = cmf::QPointStress<Material, 2>(mat_, m_.grad(Xp, t));
      const tensor<double, 2, 2> Pm = cmf::QPointStress<Material, 2>(mat_, m_.grad(Xm, t));
      for (int i = 0; i < 2; i++) { b(i) += (Pp(i, j) - Pm(i, j)) / (2.0 * h); }
    }
    b *= -1.0 / rho_;
    m_.u(X, t, u);
    b.Add(-m_.w * m_.w, u); // u_tt = -w^2 u
  }

private:
  Manufactured m_;
  Material mat_;
  double rho_;
};

void NonlinearTest()
{
  const cmf::LameParameters lame = cmf::LameFromYoungPoisson(kE, kNu);
  const cmf::NeoHookean material{lame.mu, lame.lambda};

  // Manufactured solution in space and time. At a fixed time the temporal
  // error of a nonlinear problem is not a clean power of dt on a mesh whose
  // modes the steps do not resolve, so the order in dt is taken from the
  // self-convergence below; here (h, dt) -> (h/2, dt/2) with dt small enough
  // that the spatial error leads: the dynamic solution converges at the
  // spatial rate p + 1 = 3, and halving dt alone changes nothing.
  const Manufactured m{0.03, 5.0};
  const double t_final = 0.4;
  mfem::VectorFunctionCoefficient exact_u(
    2, [&m](const mfem::Vector &X, double t, mfem::Vector &u) { m.u(X, t, u); });
  mfem::VectorFunctionCoefficient exact_v(
    2, [&m](const mfem::Vector &X, double t, mfem::Vector &u)
    { m.u(X, 0.5 * M_PI / m.w, u); u *= m.w * std::cos(m.w * t); });
  auto mms = [&](int nx, int steps)
  {
    cmf::AppConfig cfg = BaseConfig(2, "quad", 4, 2, 0.1, "neo_hookean");
    cfg.mesh.serial_refine = nx == 4 ? 0 : 1;
    std::unique_ptr<mfem::ParMesh> mesh = BuildMesh(cfg, false);
    cmf::SolidMechanicsTL problem(*mesh, cfg, cmf::Material(material));
    ManufacturedBodyForce<cmf::NeoHookean> body(m, material, kRho);
    problem.AddDirichlet({1, 2, 3, 4}, exact_u, Constant());
    problem.SetBodyForce(body, Constant(true));
    cmf::DynamicSolidProblem dyn(problem, Scheme("generalized_alpha", 0.8));
    dyn.SetInitialVelocity(exact_v);
    mfem::Vector x(problem.Height());
    x = 0.0;
    dyn.Initialize(x);
    std::unique_ptr<mfem::Solver> linear = dyn.MakeLinearSolver(cfg.solver.linear);
    const cmf::QuasiStaticReport report = cmf::SolveDynamic(
      dyn, *linear, cfg.solver, cmf::UniformTimeSteps(t_final, steps), 0.0, x);
    CHECK_MSG(report.converged, "nonlinear MMS: every time step converged");
    problem.UpdateFields(x);
    exact_u.SetTime(t_final);
    return problem.Displacement().ComputeL2Error(exact_u);
  };
  const double e_coarse = mms(4, 16), e_fine = mms(8, 32), e_fine_half_dt = mms(8, 64);
  const double rate = std::log2(e_coarse / e_fine);
  std::printf("  neo-Hookean space-time MMS: L2 error %.4e (nx 4, 16 steps) -> %.4e (nx 8, 32 steps), rate %.3f; "
              "nx 8 with 64 steps %.4e\n", e_coarse, e_fine, rate, e_fine_half_dt);
  CHECK_MSG(rate >= 2.8, "nonlinear MMS: spatial rate 3 under joint refinement, got " + std::to_string(rate));
  CHECK_MSG(std::abs(e_fine_half_dt / e_fine - 1.0) <= 0.1, "nonlinear MMS: the temporal error is below the spatial one");

  // Large-amplitude free vibration: Newton order of a coarse step, and
  // self-convergence in dt of the whole displacement field.
  std::vector<mfem::Vector> finals;
  cmf::NewtonReport coarse_step;
  double tip = 0.0;
  for (int level = 0; level < 5; level++)
  {
    // 2 x 2 elements: every mode is resolved from dt = 0.004 down, so the
    // differences between the levels are clean powers of dt.
    ClampedBlock b("neo_hookean", 2, 2);
    b.problem->AddDirichlet({4}, *b.fixed);
    mfem::VectorFunctionCoefficient swing(
      2, [](const mfem::Vector &X, mfem::Vector &v) { v.SetSize(2); v(0) = 0.0; v(1) = 6.0 * X(0); });
    cmf::DynamicSolidProblem dyn(*b.problem, Scheme("generalized_alpha", 0.8));
    dyn.SetInitialVelocity(swing);
    mfem::Vector x(b.problem->Height());
    x = 0.0;
    dyn.Initialize(x);
    std::unique_ptr<mfem::Solver> linear = dyn.MakeLinearSolver(b.cfg.solver.linear);
    b.cfg.solver.newton.rtol = 1e-12;
    const int steps = level == 0 ? 8 : 20 << level; // dt = 0.02, then 0.004 ... 0.0005
    const cmf::QuasiStaticReport report = cmf::SolveDynamic(
      dyn, *linear, b.cfg.solver, cmf::UniformTimeSteps(0.16, steps), 0.0, x);
    CHECK_MSG(report.converged, "free vibration: every time step converged");
    if (level == 0) { coarse_step = report.steps[3].newton; } // Newton has work to do at dt = 0.02
    else { finals.push_back(x); }
    tip = std::max(tip, x.Normlinf());
  }
  std::vector<double> res;
  for (const cmf::NewtonIteration &it : coarse_step.history)
  {
    if (it.residual > 1e-11 * coarse_step.initial_residual) { res.push_back(it.residual); }
  }
  std::printf("  neo-Hookean free vibration, max |u| = %.3f; Newton residuals of a coarse step:", tip);
  for (const cmf::NewtonIteration &it : coarse_step.history) { std::printf(" %.2e", it.residual); }
  std::printf("\n");
  CHECK_MSG(tip >= 0.1, "the vibration is of large amplitude");
  CHECK_MSG(res.size() >= 3, "at least three residuals above the floor");
  if (res.size() >= 3)
  {
    const std::size_t n = res.size();
    const double q = std::log(res[n - 1] / res[n - 2]) / std::log(res[n - 2] / res[n - 3]);
    std::printf("  Newton contraction order estimate: %.3f\n", q);
    CHECK_MSG(q >= 1.8, "K + c_M M is the consistent Jacobian: order " + std::to_string(q) + " >= 1.8");
  }
  for (std::size_t k = 0; k + 2 < finals.size(); k++)
  {
    const double ratio = MaxDiff(finals[k], finals[k + 1]) / MaxDiff(finals[k + 1], finals[k + 2]);
    std::printf("  self-convergence in dt, levels %zu-%zu: ratio %.3f\n", k, k + 2, ratio);
    CHECK_MSG(ratio >= 3.6 && ratio <= 4.4, "self-convergence ratio " + std::to_string(ratio) + " near 4");
  }
}

// ---------------------------------------------------------------------------
// 8. A linear problem forms K + c_M M, and sets its solver up, once per dt.
// A step that fails once (the residual is made non-finite) is bisected; the
// history is that of the last accepted step, so the run equals the one with
// the two half steps planned.
class FailsOnce : public cmf::DynamicSolidProblem
{
public:
  FailsOnce(cmf::SolidProblem &problem, const cmf::DynamicsConfig &cfg, double at)
    : cmf::DynamicSolidProblem(problem, cfg), at_(at) {}
  void Mult(const mfem::Vector &x, mfem::Vector &y) const override
  {
    cmf::DynamicSolidProblem::Mult(x, y);
    if (armed_ && LoadFactor() == at_)
    {
      y = std::numeric_limits<double>::quiet_NaN();
      armed_ = false;
    }
  }

private:
  double at_;
  mutable bool armed_ = true;
};

void ReuseTest()
{
  auto run = [](const std::vector<double> &times, bool reuse, double fail_at, mfem::Vector &x,
                int &inner, int &outer, int &setups, int &bisections, double &t_end)
  {
    ClampedBlock b("linear_elastic", 4, 2);
    b.cfg.solver.linear.type = "cg_amg";
    b.cfg.solver.substep.on_failure = true;
    b.cfg.solver.substep.min_dt = 1e-12;
    b.problem->AddDirichlet({4}, *b.fixed);
    auto *tl = dynamic_cast<cmf::SolidMechanicsTL *>(b.problem.get());
    tl->ReuseConstantGradient(reuse);
    FailsOnce dyn(*b.problem, Scheme("generalized_alpha", 0.9), fail_at);
    dyn.ReuseConstantOperator(reuse);
    dyn.SetInitialDisplacement(*b.initial);
    x.SetSize(b.problem->Height());
    x = 0.0;
    dyn.Initialize(x);
    std::unique_ptr<mfem::Solver> linear = dyn.MakeLinearSolver(b.cfg.solver.linear);
    const cmf::QuasiStaticReport report = cmf::SolveDynamic(dyn, *linear, b.cfg.solver, times, 0.0, x);
    inner = tl->GradientAssemblies();
    outer = dyn.OperatorAssemblies();
    setups = dynamic_cast<cmf::LinearSolver &>(*linear).Setups();
    bisections = report.bisections;
    t_end = dyn.Time();
    return report.converged;
  };

  const double never = -1.0;
  mfem::Vector x_reuse, x_fresh, x_bisected, x_planned;
  int inner = 0, outer = 0, setups = 0, bisections = 0;
  double t_end = 0.0;
  CHECK(run(cmf::UniformTimeSteps(0.4, 200), true, never, x_reuse, inner, outer, setups, bisections, t_end));
  std::printf("  reuse on, 200 steps: K assembled %d, K + c_M M formed %d, solver setups %d\n", inner, outer, setups);
  CHECK_MSG(inner == 1 && outer == 1 && setups == 1, "one Jacobian and one solver setup for the run");
  CHECK(run(cmf::UniformTimeSteps(0.4, 200), false, never, x_fresh, inner, outer, setups, bisections, t_end));
  std::printf("  reuse off: K assembled %d, K + c_M M formed %d, solver setups %d; max difference %.1e\n",
              inner, outer, setups, MaxDiff(x_reuse, x_fresh));
  CHECK_MSG(inner >= 200 && outer >= 200, "without the reuse every Newton step assembles");
  CHECK_MSG(MaxDiff(x_reuse, x_fresh) == 0.0, "the reuse does not change a digit");

  // Ten steps of 0.004, the fifth failing once: 0.002 + 0.002 instead.
  const std::vector<double> ten = cmf::UniformTimeSteps(0.04, 10);
  CHECK(run(ten, true, ten[4], x_bisected, inner, outer, setups, bisections, t_end));
  std::printf("  bisected step: %d bisection, K + c_M M formed %d, solver setups %d, end time %.17g\n",
              bisections, outer, setups, t_end);
  CHECK_MSG(bisections == 1, "the failed step is halved once");
  CHECK_MSG(inner == 1 && outer == 3 && setups == 3, "K + c_M M is formed for dt, dt/2 and dt again");
  CHECK_MSG(t_end == 0.04, "the end time is the last target exactly");
  std::vector<double> planned(ten);
  planned.insert(planned.begin() + 4, 0.5 * (ten[3] + ten[4]));
  CHECK(run(planned, true, never, x_planned, inner, outer, setups, bisections, t_end));
  std::printf("  bisected against planned half steps: max difference %.1e\n", MaxDiff(x_bisected, x_planned));
  CHECK_MSG(MaxDiff(x_bisected, x_planned) <= 1e-13 * x_planned.Normlinf(),
            "a rejected step leaves the history untouched");
}

// Without the external work the step does not evaluate the full static
// residual: S_n comes from Newton's last residual and the reactions are formed
// on demand. Nothing may change.
void LazyBalanceTest()
{
  for (const std::string model : {"linear_elastic", "neo_hookean"})
  {
    mfem::Vector x_ref;
    std::vector<cmf::Reaction> rx_ref;
    for (int pass = 0; pass < 2; pass++)
    {
      ClampedBlock b(model, 4, 2);
      mfem::VectorFunctionCoefficient support(
        2, [](const mfem::Vector &, double t, mfem::Vector &u)
        { u.SetSize(2); u(0) = 0.02 * (1.0 - std::cos(5.0 * t)); u(1) = 0.0; });
      b.problem->AddDirichlet({4}, support, Constant());
      cmf::DynamicSolidProblem dyn(*b.problem, Scheme("generalized_alpha", 0.7));
      dyn.TrackExternalWork(pass == 0);
      dyn.SetInitialDisplacement(*b.initial);
      mfem::Vector x(b.problem->Height());
      x = 0.0;
      dyn.Initialize(x);
      std::unique_ptr<mfem::Solver> linear = dyn.MakeLinearSolver(b.cfg.solver.linear);
      const cmf::QuasiStaticReport report = cmf::SolveDynamic(
        dyn, *linear, b.cfg.solver, cmf::UniformTimeSteps(0.2, 40), 0.0, x);
      CHECK_MSG(report.converged, model + ": converged");
      if (pass == 0) { x_ref = x; rx_ref = dyn.Reactions(); continue; }
      const std::vector<cmf::Reaction> rx = dyn.Reactions();
      const double dr = std::abs(rx[0].force[0] - rx_ref[0].force[0]) + std::abs(rx[0].force[1] - rx_ref[0].force[1]);
      std::printf("  %s, work not tracked: max difference of the state %.1e, of the reaction %.1e (force %.4e)\n",
                  model.c_str(), MaxDiff(x, x_ref), dr, rx_ref[0].force[0]);
      CHECK_MSG(MaxDiff(x, x_ref) == 0.0, model + ": the same state to the last digit");
      CHECK_MSG(dr == 0.0, model + ": the same reaction to the last digit");
    }
  }
}

// 9. The stepper in physical time, far from the unit interval.
void StepperTest()
{
  for (const double t_final : {1e-3, 1e3})
  {
    ClampedBlock b("linear_elastic", 2, 1);
    b.cfg.solver.linear.type = "cg_amg";
    b.cfg.solver.substep.on_failure = true;
    b.cfg.solver.substep.min_dt = 1e-12 * t_final;
    b.problem->AddDirichlet({4}, *b.fixed);
    const std::vector<double> times = cmf::UniformTimeSteps(t_final, 1000);
    FailsOnce dyn(*b.problem, Scheme("generalized_alpha", 0.9), times[617]);
    dyn.SetInitialDisplacement(*b.initial);
    mfem::Vector x(b.problem->Height());
    x = 0.0;
    dyn.Initialize(x);
    std::unique_ptr<mfem::Solver> linear = dyn.MakeLinearSolver(b.cfg.solver.linear);
    double smallest = t_final;
    const cmf::QuasiStaticReport report = cmf::SolveDynamic(
      dyn, *linear, b.cfg.solver, times, 0.0, x,
      [&](const cmf::LoadStepReport &s, const mfem::Vector &)
      { smallest = std::min(smallest, s.load_factor - s.t_begin); });
    std::printf("  stepper, t_final %g: %zu accepted steps (%d bisection), smallest step %.3g of the planned, "
                "end time - t_final = %.1e\n", t_final, report.steps.size(), report.bisections,
                smallest / (t_final / 1000.0), dyn.Time() - t_final);
    CHECK_MSG(report.converged, "stepper: converged");
    CHECK_MSG(report.steps.size() == 1001 && dyn.Steps() == 1001, "1000 planned steps, one of them halved");
    CHECK_MSG(dyn.Time() == t_final, "the end time is t_final exactly");
    CHECK_MSG(smallest >= 0.49 * t_final / 1000.0, "no degenerate step");
  }
}

// ---------------------------------------------------------------------------
// 11. Mixed u-p formulation (DY4): M on the displacement block, the constraint
// row unweighted, a differential-algebraic system. Small strain (Herrmann), so
// that u = sin(w t) U with U quadratic and p = sin(w t) P with P linear are
// held exactly by the Taylor-Hood pair and the errors are those of the time
// integrator. Nearly incompressible: U of QuadraticField, P = kappa div U.
// Incompressible: U = (0.02 x^2, -0.04 x y), divergence free and zero on the
// left side, P = 0.3 x - 0.2 y + 0.1.
struct IncompressibleField
{
  double w = 5.0, mu, rho;
  static void U(const mfem::Vector &X, mfem::Vector &u)
  {
    u.SetSize(2);
    u(0) = 0.02 * X(0) * X(0);
    u(1) = -0.04 * X(0) * X(1);
  }
  static double P(const mfem::Vector &X) { return 0.3 * X(0) - 0.2 * X(1) + 0.1; }
  // sigma = 2 mu eps(U) + P I (tr eps = 0); Div sigma = mu lap(U) + grad P = (0.04 mu + 0.3, -0.2).
  void BodyForce(const mfem::Vector &X, double t, mfem::Vector &b) const
  {
    mfem::Vector u;
    U(X, u);
    const double div_sigma[2] = {0.04 * mu + 0.3, -0.2};
    b.SetSize(2);
    for (int i = 0; i < 2; i++) { b(i) = std::sin(w * t) * (-w * w * u(i) - div_sigma[i] / rho); }
  }
  void Traction(const mfem::Vector &X, double t, mfem::Vector &T) const
  {
    const double x = X(0), y = X(1);
    const double H[2][2] = {{0.04 * x, 0.0}, {-0.04 * y, -0.04 * x}};
    double N[2] = {0.0, 0.0};
    if (y < 1e-12) { N[1] = -1.0; }
    else if (y > 1.0 - 1e-12) { N[1] = 1.0; }
    else { N[0] = 1.0; }
    T.SetSize(2);
    for (int i = 0; i < 2; i++)
    {
      T(i) = P(X) * N[i];
      for (int j = 0; j < 2; j++) { T(i) += mu * (H[i][j] + H[j][i]) * N[j]; }
      T(i) *= std::sin(w * t);
    }
  }
};

mfem::Vector PressureTrueDofs(cmf::MixedSolidMechanicsTL &mixed, mfem::Coefficient &c, double t)
{
  c.SetTime(t);
  mfem::ParGridFunction g(&mixed.PressureSpace());
  g.ProjectCoefficient(c);
  mfem::Vector tv;
  g.GetTrueDofs(tv);
  return tv;
}

void MixedTest()
{
  const double t_final = 0.8;
  for (int incompressible = 0; incompressible < 2; incompressible++)
  {
    const double nu = 0.4999;
    const cmf::LameParameters lame = cmf::LameFromYoungPoisson(kE, nu);
    QuadraticField f;
    f.lambda = lame.lambda;
    f.mu = lame.mu;
    f.rho = kRho;
    const double kappa = lame.lambda + 2.0 * lame.mu / 3.0;
    IncompressibleField g;
    g.mu = lame.mu;
    g.rho = kRho;
    const double w = f.w;
    mfem::VectorFunctionCoefficient exact_u(2, [&](const mfem::Vector &X, double t, mfem::Vector &u)
    {
      if (incompressible) { IncompressibleField::U(X, u); }
      else { QuadraticField::U(X, u); }
      u *= std::sin(w * t);
    });
    mfem::VectorFunctionCoefficient exact_v(2, [&](const mfem::Vector &X, double t, mfem::Vector &u)
    {
      if (incompressible) { IncompressibleField::U(X, u); }
      else { QuadraticField::U(X, u); }
      u *= w * std::cos(w * t);
    });
    mfem::FunctionCoefficient exact_p([&](const mfem::Vector &X, double t)
    {
      const double P = incompressible ? IncompressibleField::P(X) : kappa * (0.055 * X(0) + 0.01 * X(1));
      return std::sin(w * t) * P;
    });
    mfem::VectorFunctionCoefficient body(2, [&](const mfem::Vector &X, double t, mfem::Vector &b)
    { if (incompressible) { g.BodyForce(X, t, b); } else { f.BodyForce(X, t, b); } });
    mfem::VectorFunctionCoefficient traction(2, [&](const mfem::Vector &X, double t, mfem::Vector &T)
    { if (incompressible) { g.Traction(X, t, T); } else { f.Traction(X, t, T); } });

    std::vector<double> err_u, err_p;
    for (int level = 0; level < 3; level++)
    {
      cmf::AppConfig cfg = BaseConfig(2, "quad", 2, 2, 0.0, "linear_elastic");
      cfg.formulation = "mixed";
      cfg.material.nu = nu;
      if (incompressible)
      {
        cfg.material.E = std::numeric_limits<double>::quiet_NaN();
        cfg.material.nu = std::numeric_limits<double>::quiet_NaN();
        cfg.material.mu = lame.mu;
        cfg.material.incompressible = true;
      }
      // The pressure balances M a, and a = (u - u*) / (beta dt^2) carries the
      // error of the solve multiplied by c_M = O(1 / dt^2): with the usual
      // tolerances that noise (4e-5 here) hides the temporal error of p from
      // 400 steps on, so the solves are tight and the steps not too small.
      cfg.solver.linear.rtol = 1e-15;
      cfg.solver.newton.rtol = 1e-13;
      cfg.solver.linear.inner_rtol = 1e-4;
      std::unique_ptr<mfem::ParMesh> mesh = BuildMesh(cfg, false);
      std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
      auto &mixed = dynamic_cast<cmf::MixedSolidMechanicsTL &>(*problem);
      problem->AddDirichlet({4}, exact_u, Constant());
      problem->AddTraction({1, 2, 3}, traction, Constant(true));
      problem->SetBodyForce(body, Constant(true));
      cmf::DynamicSolidProblem dyn(*problem, Scheme("generalized_alpha", 0.8));
      dyn.SetInitialVelocity(exact_v);
      mfem::Vector x(problem->Height());
      x = 0.0;
      dyn.Initialize(x);
      std::unique_ptr<mfem::Solver> linear = dyn.MakeLinearSolver(cfg.solver.linear);
      const cmf::QuasiStaticReport report = cmf::SolveDynamic(
        dyn, *linear, cfg.solver, cmf::UniformTimeSteps(t_final, (incompressible ? 100 : 50) << level), 0.0, x);
      CHECK_MSG(report.converged, "mixed: every time step converged");
      const int n_u = problem->DisplacementSpace().GetTrueVSize();
      const mfem::Vector x_u(x.GetData(), n_u), x_p(x.GetData() + n_u, x.Size() - n_u);
      err_u.push_back(MaxDiff(x_u, TrueDofs(problem->DisplacementSpace(), exact_u, t_final)));
      err_p.push_back(MaxDiff(x_p, PressureTrueDofs(mixed, exact_p, t_final)));
    }
    std::printf("  mixed, %s, temporal order:", incompressible ? "incompressible" : "nu = 0.4999");
    for (std::size_t k = 0; k + 1 < err_u.size(); k++)
    {
      const double ru = std::log2(err_u[k] / err_u[k + 1]), rp = std::log2(err_p[k] / err_p[k + 1]);
      std::printf(" u %.3f p %.3f;", ru, rp);
      const std::string tag = incompressible ? "mixed incompressible" : "mixed nu 0.4999";
      CHECK_MSG(std::abs(ru - 2.0) <= 0.1, tag + ": order 2 in dt for u, got " + std::to_string(ru));
      CHECK_MSG(rp >= 1.8, tag + ": order 2 in dt for p, got " + std::to_string(rp));
    }
    std::printf(" errors u %.3e -> %.3e, p %.3e -> %.3e\n", err_u.front(), err_u.back(), err_p.front(),
                err_p.back());
  }

  // The pressure mode: two incompressible runs that differ in p_0 alone. The
  // pressure force is interpolated with the rest of S_u, so the difference
  // returns with the factor -af / (1 - af) = -rho_inf at every step.
  const double rho_inf = 0.6;
  std::vector<mfem::Vector> pressures[2];
  for (int pass = 0; pass < 2; pass++)
  {
    cmf::AppConfig cfg = BaseConfig(2, "quad", 4, 2, 0.1, "linear_elastic");
    cfg.formulation = "mixed";
    cfg.material.E = cfg.material.nu = std::numeric_limits<double>::quiet_NaN();
    cfg.material.mu = 96.0;
    cfg.material.incompressible = true;
    cfg.solver.linear.rtol = 1e-13;
    std::unique_ptr<mfem::ParMesh> mesh = BuildMesh(cfg, false);
    std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
    mfem::Vector zero(2);
    zero = 0.0;
    mfem::VectorConstantCoefficient fixed(zero);
    mfem::VectorFunctionCoefficient swing(
      2, [](const mfem::Vector &X, mfem::Vector &v) { v.SetSize(2); v(0) = 0.0; v(1) = 0.2 * X(0); });
    problem->AddDirichlet({4}, fixed);
    cmf::DynamicSolidProblem dyn(*problem, Scheme("generalized_alpha", rho_inf));
    dyn.SetInitialVelocity(swing);
    mfem::Vector x(problem->Height());
    x = 0.0;
    const int n_u = problem->DisplacementSpace().GetTrueVSize();
    if (pass == 1) { for (int i = n_u; i < x.Size(); i++) { x(i) = 0.5 * std::sin(1.0 + 3.0 * i); } }
    // The same initial acceleration in both, so that the runs differ in p_0 only.
    mfem::Vector a0(n_u);
    a0 = 0.0;
    dyn.SetInitialAcceleration(a0);
    dyn.Initialize(x);
    std::unique_ptr<mfem::Solver> linear = dyn.MakeLinearSolver(cfg.solver.linear);
    cmf::SolveDynamic(dyn, *linear, cfg.solver, cmf::UniformTimeSteps(0.08, 8), 0.0, x,
                      [&](const cmf::LoadStepReport &, const mfem::Vector &y)
                      {
                        // a copy: pushing the view itself would move it, and leave an alias of y
                        const mfem::Vector view(y.GetData() + n_u, y.Size() - n_u);
                        mfem::Vector copy(view);
                        pressures[pass].push_back(copy);
                      });
  }
  std::printf("  mixed, pressure mode with rho_inf = %.1f: |p' - p| of successive steps in the ratio", rho_inf);
  double previous = 0.0;
  for (std::size_t k = 0; k < pressures[0].size(); k++)
  {
    mfem::Vector d(pressures[1][k]);
    d -= pressures[0][k];
    const double norm = d.Norml2();
    if (k > 0)
    {
      std::printf(" %.4f", norm / previous);
      CHECK_MSG(std::abs(norm / previous - rho_inf) <= 0.05 * rho_inf, "an error in the pressure returns with the factor rho_inf");
    }
    previous = norm;
  }
  std::printf("\n");

  // Finite kappa and an initial displacement: p_0 is the pressure of u_0.
  {
    cmf::AppConfig cfg = BaseConfig(2, "quad", 2, 2, 0.0, "linear_elastic");
    cfg.formulation = "mixed";
    cfg.material.nu = 0.45;
    std::unique_ptr<mfem::ParMesh> mesh = BuildMesh(cfg, false);
    std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
    auto &mixed = dynamic_cast<cmf::MixedSolidMechanicsTL &>(*problem);
    const cmf::LameParameters lame = cmf::LameFromYoungPoisson(kE, 0.45);
    const double kappa = lame.lambda + 2.0 * lame.mu / 3.0;
    mfem::VectorFunctionCoefficient u0(2, [](const mfem::Vector &X, mfem::Vector &u) { QuadraticField::U(X, u); });
    mfem::FunctionCoefficient p0([kappa](const mfem::Vector &X) { return kappa * (0.055 * X(0) + 0.01 * X(1)); });
    cmf::DynamicSolidProblem dyn(*problem, Scheme("generalized_alpha", 0.8));
    dyn.SetInitialDisplacement(u0);
    mfem::Vector x(problem->Height());
    x = 0.0;
    dyn.Initialize(x);
    const int n_u = problem->DisplacementSpace().GetTrueVSize();
    const mfem::Vector x_p(x.GetData() + n_u, x.Size() - n_u);
    const mfem::Vector exact = PressureTrueDofs(mixed, p0, 0.0);
    std::printf("  mixed, nu = 0.45, initial displacement: max |p_0 - kappa div u_0| = %.2e of %.3f\n",
                MaxDiff(x_p, exact), exact.Normlinf());
    CHECK_MSG(MaxDiff(x_p, exact) <= 1e-10 * exact.Normlinf(), "the initial pressure is that of the initial displacement");
  }

  // The header warns when the scheme has no dissipation.
  cmf::AppConfig cfg;
  cfg.formulation = "mixed";
  cfg.dynamics = Scheme("newmark");
  cfg.dynamics.t_final = 1.0;
  cfg.dynamics.breakpoints = cmf::UniformTimeSteps(1.0, 10);
  bool warned = false;
  for (const std::string &line : cmf::DescribeDynamics(cfg)) { warned = warned || line.find("carries no inertia") != std::string::npos; }
  CHECK_MSG(warned, "mixed formulation with the trapezoidal rule: the header warns");
  cfg.dynamics = Scheme("generalized_alpha", 0.8);
  cfg.dynamics.t_final = 1.0;
  cfg.dynamics.breakpoints = cmf::UniformTimeSteps(1.0, 10);
  warned = false;
  for (const std::string &line : cmf::DescribeDynamics(cfg)) { warned = warned || line.find("warning") != std::string::npos; }
  CHECK_MSG(!warned, "mixed formulation with rho_inf = 0.8: no warning");
}

// 10. From a YAML input to the library objects (DY2): the dynamics block, the
// initial state from expressions, the nodal fields velocity and acceleration,
// the header of the run.
void YamlTest()
{
  cmf::AppConfig cfg = cmf::LoadConfig("apps/input/dynamics/bar_free_vibration.yaml");
  CHECK(cfg.dynamics.enabled && cfg.dynamics.breakpoints.size() == 2000 && cfg.output.energy);
  cfg.output.paraview.clear();
  cfg.dynamics.t_final = 0.4;
  cfg.dynamics.breakpoints = cmf::UniformTimeSteps(0.4, 20);
  const std::vector<std::string> header = cmf::DescribeDynamics(cfg);
  CHECK(header.size() == 2 && header[0].find("trapezoidal rule") != std::string::npos &&
        header[0].find("20 time steps of 0.02") != std::string::npos);
  CHECK(header.size() == 2 && header[1] == "  dirichlet wall: constant in time (on from t = 0)");

  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
  problem->Finalize();
  std::unique_ptr<cmf::DynamicSolidProblem> dyn = cmf::MakeDynamicSolidProblem(*problem, cfg);
  cmf::FieldRegistry fields;
  dyn->RegisterFields(fields);
  CHECK(fields.Has("displacement") && fields.Has("velocity") && fields.Has("acceleration"));
  mfem::Vector x(problem->Height());
  x = 0.0;
  const double change = dyn->Initialize(x);
  CHECK_MSG(change <= 1e-15, "the initial displacement agrees with the clamp");
  const double e0 = dyn->KineticEnergy() + problem->InternalEnergy(x);
  std::unique_ptr<mfem::Solver> linear = dyn->MakeLinearSolver(cfg.solver.linear);
  const cmf::QuasiStaticReport report =
    cmf::SolveDynamic(*dyn, *linear, cfg.solver, cfg.dynamics.breakpoints, 0.0, x);
  CHECK_MSG(report.converged && dyn->Time() == 0.4, "the input runs to its end time");
  dyn->UpdateFields(x);
  // The trapezoidal rule keeps the mode and its amplitude and lengthens the
  // period: u_n = A cos(w~ t_n) with tan(w~ dt / 2) = w dt / 2.
  const double A = 0.01, t = 0.4, dt = 0.02;
  const double w = (2.0 / dt) * std::atan(0.5 * (0.5 * M_PI) * dt);
  const std::vector<double> u = cmf::ProbeVector(fields.Get("displacement"), {10.0, 0.5, 0.5});
  const std::vector<double> v = cmf::ProbeVector(fields.Get("velocity"), {10.0, 0.5, 0.5});
  const std::vector<double> a = cmf::ProbeVector(fields.Get("acceleration"), {10.0, 0.5, 0.5});
  const double e1 = dyn->KineticEnergy() + problem->InternalEnergy(x);
  std::printf("  bar input, 20 steps: tip u %.9e (discrete dispersion %.9e), v %.9e (%.9e), a %.9e (%.9e), energy drift %.1e\n",
              u[0], A * std::cos(w * t), v[0], -A * w * std::sin(w * t), a[0], -A * w * w * std::cos(w * t),
              std::abs(e1 - e0) / e0);
  CHECK_CLOSE(u[0], A * std::cos(w * t), 1e-7 * A);
  CHECK_CLOSE(v[0], -A * w * std::sin(w * t), 1e-3 * A * w);
  CHECK_CLOSE(a[0], -A * w * w * std::cos(w * t), 1e-3 * A * w * w);
  CHECK_MSG(std::abs(e1 - e0) <= 1e-12 * e0, "energy of the bar input is conserved");

  cfg.dynamics.initial_velocity = {"0", "0"};
  CHECK_THROWS(cmf::MakeDynamicSolidProblem(*problem, cfg), cmf::ConfigError, "dynamics.initial.velocity");
}

} // namespace

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  if (mfem::Mpi::WorldSize() != 1)
  {
    if (mfem::Mpi::Root()) { std::cout << "test_dynamics is a serial test" << std::endl; }
    return 1;
  }
  std::cout << "mass matrix" << std::endl;
  MassTest();
  std::cout << "free fall" << std::endl;
  FreeFallTest();
  std::cout << "temporal order without spatial error" << std::endl;
  TemporalOrderTest();
  std::cout << "energy and numerical dissipation" << std::endl;
  EnergyTest();
  std::cout << "energy and momentum balances" << std::endl;
  BalanceTest();
  std::cout << "nonlinear path" << std::endl;
  NonlinearTest();
  std::cout << "one Jacobian per run, bisection" << std::endl;
  ReuseTest();
  LazyBalanceTest();
  std::cout << "stepper in physical time" << std::endl;
  StepperTest();
  std::cout << "from a YAML input" << std::endl;
  YamlTest();
  std::cout << "mixed u-p formulation" << std::endl;
  MixedTest();
  return cmf_test::Report("test_dynamics");
}

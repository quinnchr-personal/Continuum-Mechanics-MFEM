// Small-strain linear elasticity (model linear_elastic) in the displacement
// formulation. The total Lagrangian kernel with the flux sigma(sym(F - I))
// must be the classical linear elasticity operator:
//   1. oracle: the assembled Jacobian equals MFEM's ElasticityIntegrator
//      (same quadrature rule) on perturbed quad / tri / hex / tet meshes;
//   2. it is the tangent at u = 0 of the hyperelastic models with the same
//      small-strain moduli;
//   3. the solve is linear: one Newton iteration from any start and at any
//      load amplitude, scaling, superposition, Clapeyron's theorem;
//   4. patch test at a finite amplitude on the four element types;
//   5. manufactured solution with the analytic body force
//      -(lambda + mu) grad(div u) - mu lap(u): L2 rates p + 1;
//   6. st_venant_kirchhoff approaches it linearly in the load;
//   7. reactions balance the loads with reference moment arms; follower
//      pressures and the tangent predictor are configuration errors;
//   8. every output quantity of homogeneous Hooke states at 5 percent strain,
//      in every presentation: sigma = P (no push-forward), J = 1 + tr(eps),
//      strain = eps, the thickness stretch 1 + eps_33 under plane stress;
//   9. a Newton tolerance below the round-off floor of the residual: a linear
//      problem is accepted at the floor, a nonlinear one still fails.
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "base/fields.hpp"
#include "base/mesh_input.hpp"
#include "base/probes.hpp"
#include "materials/materials.hpp"
#include "mfem.hpp"
#include "physics/solid_mechanics_tl.hpp"
#include "physics/solid_problem.hpp"
#include "solvers/linear_solver.hpp"
#include "solvers/quasi_static.hpp"
#include "test_util.hpp"

namespace
{

const double kE = 250.0, kNu = 0.3, kRho0 = 2.5;

struct ElementCase
{
  int dim;
  const char *element;
  int n;          // elements per direction of the base box
  double perturb; // interior vertex jitter, fraction of h
};

// Perturbed meshes: the element maps are not affine (quad, hex) and the
// simplices are not congruent, so the geometric factors are exercised.
const std::vector<ElementCase> kElementCases = {
  {2, "quad", 4, 0.2}, {2, "tri", 4, 0.2}, {3, "hex", 3, 0.15}, {3, "tet", 2, 0.1}};

cmf::AppConfig BoxConfig(const ElementCase &ec, int order, const std::string &model = "linear_elastic")
{
  cmf::AppConfig cfg;
  cfg.mesh.cartesian = true;
  cfg.mesh.box.dim = ec.dim;
  cfg.mesh.box.element = ec.element;
  cfg.mesh.box.nx = cfg.mesh.box.ny = cfg.mesh.box.nz = ec.n;
  cfg.mesh.order = order;
  cfg.mesh.perturb = ec.perturb;
  cfg.material.model = model;
  cfg.material.E = kE;
  cfg.material.nu = kNu;
  cfg.material.rho0 = kRho0;
  cfg.solver.load_steps = 1;
  cfg.solver.newton.rtol = 1e-10;
  cfg.solver.newton.atol = 0.0; // convergence by the relative criterion only
  cfg.solver.newton.max_it = 10;
  cfg.solver.newton.print_level = 0;
  cfg.solver.linear.type = "cg_amg";
  cfg.solver.linear.rtol = 1e-14;
  cfg.solver.linear.max_it = 2000;
  return cfg;
}

// Boundary attributes of MFEM's Cartesian boxes: 2D 1 bottom, 2 right, 3 top,
// 4 left; 3D 1 bottom (z = 0), 2 front (y = 0), 3 right, 4 back, 5 left, 6 top.
int LeftAttr(int dim) { return dim == 2 ? 4 : 5; }
int RightAttr(int dim) { return dim == 2 ? 2 : 3; }
std::vector<int> AllAttrs(int dim)
{
  return dim == 2 ? std::vector<int>{1, 2, 3, 4} : std::vector<int>{1, 2, 3, 4, 5, 6};
}

void FillRandom(mfem::Vector &v, std::mt19937 &rng, double amplitude)
{
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  for (int i = 0; i < v.Size(); i++) { v(i) = amplitude * unit(rng); }
}

double RelDiff(const mfem::Vector &a, const mfem::Vector &b)
{
  mfem::Vector d(a);
  d -= b;
  return d.Norml2() / b.Norml2();
}

// 1. and 2.: K v of the assembled Jacobian against MFEM's elasticity operator
// with the kernel's quadrature rule, and against the hyperelastic Jacobians at
// u = 0. The state at which the linear Jacobian is taken is random: the
// tangent does not depend on it.
void OperatorTests()
{
  const cmf::LameParameters lame = cmf::LameFromYoungPoisson(kE, kNu);
  std::mt19937 rng(11u);
  for (const ElementCase &ec : kElementCases)
  {
    for (int order = 1; order <= 2; order++)
    {
      const cmf::AppConfig cfg = BoxConfig(ec, order);
      std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
      cmf::SolidMechanicsTL physics(*pmesh, cfg, cmf::MakeMaterial(cfg.material));
      mfem::Vector zero(ec.dim);
      zero = 0.0;
      mfem::VectorConstantCoefficient zero_coef(zero);
      physics.AddDirichlet({LeftAttr(ec.dim)}, zero_coef);
      physics.Finalize();
      physics.SetLoadFactor(1.0);

      const int n = physics.FESpace().GetTrueVSize();
      mfem::Vector u(n), v(n), Kv(n), Av(n);
      FillRandom(u, rng, 0.3);
      FillRandom(v, rng, 1.0);
      physics.ApplyDirichlet(u);
      const mfem::Array<int> &ess = physics.EssentialTrueDofs();
      for (int i = 0; i < ess.Size(); i++) { v(ess[i]) = 0.0; }
      physics.GetGradient(u).Mult(v, Kv);

      mfem::ConstantCoefficient lambda_c(lame.lambda), mu_c(lame.mu);
      mfem::ParBilinearForm a(&physics.FESpace());
      auto *integ = new mfem::ElasticityIntegrator(lambda_c, mu_c);
      integ->SetIntRule(&mfem::IntRules.Get(pmesh->GetElementBaseGeometry(0), 2 * order + 3));
      a.AddDomainIntegrator(integ);
      a.Assemble();
      mfem::HypreParMatrix A;
      a.FormSystemMatrix(ess, A);
      A.Mult(v, Av);
      const double oracle = RelDiff(Kv, Av);
      std::printf("  %-4s p=%d (%5d dofs): |K v - K_mfem v| / |K_mfem v| = %.2e", ec.element, order, n, oracle);
      CHECK_MSG(oracle <= 1e-12, std::string(ec.element) + " p=" + std::to_string(order) +
                ": Jacobian vs mfem::ElasticityIntegrator " + std::to_string(oracle));

      // The hyperelastic models with the same small-strain moduli, at u = 0.
      for (const std::string model : {"neo_hookean", "st_venant_kirchhoff", "iso_neo_hookean"})
      {
        const cmf::AppConfig hcfg = BoxConfig(ec, order, model);
        cmf::SolidMechanicsTL hyper(*pmesh, hcfg, cmf::MakeMaterial(hcfg.material));
        hyper.AddDirichlet({LeftAttr(ec.dim)}, zero_coef);
        hyper.Finalize();
        hyper.SetLoadFactor(1.0);
        mfem::Vector u0(n), Hv(n);
        u0 = 0.0;
        hyper.GetGradient(u0).Mult(v, Hv);
        const double rel = RelDiff(Hv, Kv);
        std::printf(", %s %.1e", model.c_str(), rel);
        CHECK_MSG(rel <= 1e-12, std::string(ec.element) + " p=" + std::to_string(order) + ": " + model +
                  " Jacobian at u = 0 vs linear_elastic " + std::to_string(rel));
        // Control: away from u = 0 the hyperelastic tangent does differ, by the
        // order of the displacement gradient (so the comparison above can fail).
        FillRandom(u0, rng, 1e-3);
        hyper.ApplyDirichlet(u0);
        hyper.GetGradient(u0).Mult(v, Hv);
        const double away = RelDiff(Hv, Kv);
        CHECK_MSG(away >= 1e-5 && away <= 1e-1, std::string(ec.element) + " p=" + std::to_string(order) + ": " +
                  model + " Jacobian at u != 0 differs from linear_elastic by " + std::to_string(away));
      }
      std::printf("\n");
    }
  }
}

// A problem with every kind of data: nonzero Dirichlet values on the left
// face, a traction on the right face and a body force, each with a factor.
struct LoadFactors
{
  double dirichlet = 1.0, traction = 1.0, body = 1.0;
};

struct Solution
{
  mfem::Vector u;
  cmf::NewtonReport newton;
  double internal_energy = 0.0;
  double external_work = 0.0; // f_ext . u
};

Solution SolveLoaded(const ElementCase &ec, int order, const LoadFactors &f, const mfem::Vector *start = nullptr)
{
  const cmf::AppConfig cfg = BoxConfig(ec, order);
  std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  cmf::SolidMechanicsTL physics(*pmesh, cfg, cmf::MakeMaterial(cfg.material));
  const int dim = ec.dim;
  mfem::VectorFunctionCoefficient left(dim, [dim, f](const mfem::Vector &X, mfem::Vector &g)
  {
    g.SetSize(dim);
    g = 0.0;
    g(0) = f.dirichlet * 2e-3 * X(1) * (1.0 - X(1));
    g(1) = f.dirichlet * -1e-3 * std::sin(M_PI * X(1));
  });
  mfem::VectorFunctionCoefficient traction(dim, [dim, f](const mfem::Vector &X, mfem::Vector &t)
  {
    t.SetSize(dim);
    t = 0.0;
    t(0) = f.traction * 0.8 * X(1);
    t(1) = f.traction * -0.5;
    if (dim == 3) { t(2) = f.traction * 0.3 * X(2); }
  });
  mfem::VectorFunctionCoefficient body(dim, [dim, f](const mfem::Vector &X, mfem::Vector &b)
  {
    b.SetSize(dim);
    b = 0.0;
    b(0) = f.body * 0.2 * X(0);
    b(dim - 1) = f.body * -0.4;
  });
  physics.AddDirichlet({LeftAttr(dim)}, left);
  physics.AddTraction({RightAttr(dim)}, traction);
  physics.SetBodyForce(body);
  physics.Finalize();

  cmf::LinearSolver linear(cfg.solver.linear, physics.FESpace());
  Solution s;
  s.u.SetSize(physics.FESpace().GetTrueVSize());
  if (start) { s.u = *start; }
  else { s.u = 0.0; }
  const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(physics, linear, cfg.solver, s.u);
  s.newton = report.steps.back().newton;
  s.internal_energy = physics.InternalEnergy(s.u);
  s.external_work = mfem::InnerProduct(MPI_COMM_WORLD, physics.ExternalLoad(), s.u);
  return s;
}

void LinearityTests()
{
  const std::vector<std::pair<ElementCase, int>> cases = {{kElementCases[0], 2}, {kElementCases[3], 1}};
  std::mt19937 rng(23u);
  for (const auto &c : cases)
  {
    const ElementCase &ec = c.first;
    const int order = c.second;
    const std::string tag = std::string(ec.element) + " p=" + std::to_string(order);
    const Solution all = SolveLoaded(ec, order, LoadFactors());
    CHECK_MSG(all.newton.converged, tag + ": converged");
    CHECK_MSG(all.newton.iterations == 1, tag + ": one Newton iteration from u = 0, got " +
              std::to_string(all.newton.iterations));
    CHECK_MSG(all.newton.residual <= 1e-10 * all.newton.initial_residual, tag + ": |R| <= 1e-10 |R0|");

    mfem::Vector start(all.u.Size());
    FillRandom(start, rng, 0.5);
    const Solution restarted = SolveLoaded(ec, order, LoadFactors(), &start);
    CHECK_MSG(restarted.newton.converged && restarted.newton.iterations == 1,
              tag + ": one Newton iteration from a random start");
    const double restart_diff = RelDiff(restarted.u, all.u);
    CHECK_MSG(restart_diff <= 1e-10, tag + ": the start does not matter, rel " + std::to_string(restart_diff));

    // A load a thousand times larger (strains of order one to ten, where a
    // hyperelastic model needs load stepping or fails): still one iteration,
    // and the solution scales.
    const Solution big = SolveLoaded(ec, order, LoadFactors{1e3, 1e3, 1e3});
    CHECK_MSG(big.newton.converged && big.newton.iterations == 1, tag + ": one Newton iteration at 1000x the load");
    mfem::Vector scaled(all.u);
    scaled *= 1e3;
    const double scale_diff = RelDiff(big.u, scaled);
    CHECK_MSG(scale_diff <= 1e-10, tag + ": u(1000 f) = 1000 u(f), rel " + std::to_string(scale_diff));

    const Solution only_d = SolveLoaded(ec, order, LoadFactors{1.0, 0.0, 0.0});
    const Solution only_t = SolveLoaded(ec, order, LoadFactors{0.0, 1.0, 0.0});
    const Solution only_b = SolveLoaded(ec, order, LoadFactors{0.0, 0.0, 1.0});
    mfem::Vector sum(only_d.u);
    sum += only_t.u;
    sum += only_b.u;
    const double super_diff = RelDiff(sum, all.u);
    CHECK_MSG(super_diff <= 1e-10, tag + ": superposition, rel " + std::to_string(super_diff));

    // Clapeyron (homogeneous Dirichlet data): 2 W_int = f_ext . u.
    const Solution dead = SolveLoaded(ec, order, LoadFactors{0.0, 1.0, 1.0});
    const double clapeyron = std::abs(2.0 * dead.internal_energy - dead.external_work) / dead.external_work;
    CHECK_MSG(dead.external_work > 0.0, tag + ": positive external work");
    CHECK_MSG(clapeyron <= 1e-10, tag + ": Clapeyron 2 W = f.u, rel " + std::to_string(clapeyron));
    std::printf("  %-8s: |R|/|R0| %.1e, restart %.1e, scaling %.1e, superposition %.1e, Clapeyron %.1e\n",
                tag.c_str(), all.newton.residual / all.newton.initial_residual, restart_diff, scale_diff,
                super_diff, clapeyron);
  }
}

// 4. Affine displacement on the whole boundary at a finite amplitude: the
// interior is reproduced in one iteration (the nonlinear materials need an
// amplitude below 1e-5 for that).
void PatchTests()
{
  for (const ElementCase &ec : kElementCases)
  {
    for (int order = 1; order <= 2; order++)
    {
      const cmf::AppConfig cfg = BoxConfig(ec, order);
      std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
      cmf::SolidMechanicsTL physics(*pmesh, cfg, cmf::MakeMaterial(cfg.material));
      const int dim = ec.dim;
      const double A[3][3] = {{0.10, -0.06, 0.04}, {0.075, -0.125, 0.02}, {-0.03, 0.05, 0.08}};
      const double c[3] = {0.025, -0.02, 0.01};
      mfem::VectorFunctionCoefficient affine(dim, [dim, A, c](const mfem::Vector &X, mfem::Vector &u)
      {
        u.SetSize(dim);
        for (int i = 0; i < dim; i++)
        {
          u(i) = c[i];
          for (int j = 0; j < dim; j++) { u(i) += A[i][j] * X(j); }
        }
      });
      physics.AddDirichlet(AllAttrs(dim), affine);
      physics.Finalize();
      cmf::LinearSolver linear(cfg.solver.linear, physics.FESpace());
      mfem::Vector u(physics.FESpace().GetTrueVSize());
      u = 0.0;
      const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(physics, linear, cfg.solver, u);
      const cmf::NewtonReport &newton = report.steps.back().newton;
      physics.UpdateFields(u);
      mfem::ParGridFunction exact(&physics.FESpace());
      exact.ProjectCoefficient(affine);
      const double u_max = exact.Normlinf();
      exact -= physics.Displacement();
      const double err = exact.Normlinf() / u_max;
      std::printf("  patch %-4s p=%d: max nodal error / max|u| = %.2e, newton its %d\n", ec.element, order,
                  err, newton.iterations);
      const std::string tag = std::string("patch ") + ec.element + " p=" + std::to_string(order);
      CHECK_MSG(newton.converged && newton.iterations == 1, tag + ": one Newton iteration");
      CHECK_MSG(err <= 1e-12, tag + ": relative nodal error " + std::to_string(err));
    }
  }
}

// 5. u = alpha (sin(pi X) sin(pi Y), X^2 Y (1 - Y)) in plane strain, with the
// body force of the linear operator written out, so the check does not pass
// through the material's own stress function:
//   rho0 b = -(lambda + mu) grad(div u) - mu lap(u).
void ManufacturedSolutionTest()
{
  const cmf::LameParameters lame = cmf::LameFromYoungPoisson(kE, kNu);
  const double alpha = 0.1, lambda = lame.lambda, mu = lame.mu;
  mfem::VectorFunctionCoefficient exact(2, [alpha](const mfem::Vector &X, mfem::Vector &u)
  {
    u.SetSize(2);
    u(0) = alpha * std::sin(M_PI * X(0)) * std::sin(M_PI * X(1));
    u(1) = alpha * X(0) * X(0) * X(1) * (1.0 - X(1));
  });
  mfem::VectorFunctionCoefficient body(2, [alpha, lambda, mu](const mfem::Vector &X, mfem::Vector &b)
  {
    const double x = X(0), y = X(1), pi = M_PI;
    const double grad_div[2] = {-pi * pi * std::sin(pi * x) * std::sin(pi * y) + 2.0 * x * (1.0 - 2.0 * y),
                                pi * pi * std::cos(pi * x) * std::cos(pi * y) - 2.0 * x * x};
    const double lap[2] = {-2.0 * pi * pi * std::sin(pi * x) * std::sin(pi * y),
                           2.0 * y * (1.0 - y) - 2.0 * x * x};
    b.SetSize(2);
    for (int i = 0; i < 2; i++) { b(i) = -alpha * ((lambda + mu) * grad_div[i] + mu * lap[i]) / kRho0; }
  });
  for (int order = 1; order <= 2; order++)
  {
    std::vector<double> errors;
    const int base = order == 1 ? 8 : 4;
    for (int level = 0; level < 4; level++)
    {
      ElementCase ec{2, "quad", base, 0.15};
      cmf::AppConfig cfg = BoxConfig(ec, order);
      cfg.mesh.serial_refine = level;
      std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
      cmf::SolidMechanicsTL physics(*pmesh, cfg, cmf::MakeMaterial(cfg.material));
      physics.AddDirichlet(AllAttrs(2), exact);
      physics.SetBodyForce(body);
      physics.Finalize();
      cmf::LinearSolver linear(cfg.solver.linear, physics.FESpace());
      mfem::Vector u(physics.FESpace().GetTrueVSize());
      u = 0.0;
      const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(physics, linear, cfg.solver, u);
      physics.UpdateFields(u);
      const double err = physics.Displacement().ComputeL2Error(exact);
      errors.push_back(err);
      const cmf::NewtonReport &newton = report.steps.back().newton;
      CHECK_MSG(newton.converged && newton.iterations == 1, "MMS p=" + std::to_string(order) + " level " +
                std::to_string(level) + ": one Newton iteration");
      std::printf("  mms p=%d nx=%3d dofs %6d: L2 error %.4e\n", order, base << level,
                  physics.FESpace().GetTrueVSize(), err);
    }
    for (std::size_t k = 0; k + 1 < errors.size(); k++)
    {
      const double rate = std::log2(errors[k] / errors[k + 1]);
      std::printf("  mms p=%d rate %zu: %.3f\n", order, k + 1, rate);
      CHECK_MSG(rate >= order + 0.95, "MMS p=" + std::to_string(order) + " rate " + std::to_string(rate) +
                " >= " + std::to_string(order + 0.95));
    }
  }
}

// 6. The geometrically nonlinear model approaches the linear one linearly in
// the load: halving the traction halves the relative difference.
void FiniteStrainLimitTest()
{
  const ElementCase ec = kElementCases[0];
  std::vector<double> diffs;
  for (const double load : {1.0, 0.5, 0.25})
  {
    mfem::Vector u[2];
    int k = 0;
    for (const std::string model : {"linear_elastic", "st_venant_kirchhoff"})
    {
      cmf::AppConfig cfg = BoxConfig(ec, 2, model);
      cfg.solver.newton.max_it = 25;
      std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
      cmf::SolidMechanicsTL physics(*pmesh, cfg, cmf::MakeMaterial(cfg.material));
      mfem::Vector zero(2), t(2);
      zero = 0.0;
      t(0) = 0.0; t(1) = load;
      mfem::VectorConstantCoefficient zero_coef(zero), traction(t);
      physics.AddDirichlet({LeftAttr(2)}, zero_coef);
      physics.AddTraction({RightAttr(2)}, traction);
      physics.Finalize();
      cmf::LinearSolver linear(cfg.solver.linear, physics.FESpace());
      u[k].SetSize(physics.FESpace().GetTrueVSize());
      u[k] = 0.0;
      const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(physics, linear, cfg.solver, u[k]);
      CHECK_MSG(report.converged, model + " converged at load " + std::to_string(load));
      k++;
    }
    diffs.push_back(RelDiff(u[1], u[0]));
  }
  std::printf("  |u_svk - u_lin| / |u_lin| at loads 1, 1/2, 1/4: %.3e %.3e %.3e\n", diffs[0], diffs[1], diffs[2]);
  for (std::size_t k = 0; k + 1 < diffs.size(); k++)
  {
    const double ratio = diffs[k] / diffs[k + 1];
    CHECK_MSG(ratio >= 1.8 && ratio <= 2.2, "st_venant_kirchhoff -> linear_elastic ratio " + std::to_string(ratio));
  }
}

// 7. Reactions of the clamped face against the resultants of the loads, with
// the moment taken about the origin at the reference positions; and the
// loads and solver options that do not exist at small strain.
void ReactionAndConfigTests()
{
  const ElementCase ec = kElementCases[0];
  cmf::AppConfig cfg = BoxConfig(ec, 2);
  std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  {
    cmf::SolidMechanicsTL physics(*pmesh, cfg, cmf::MakeMaterial(cfg.material));
    mfem::Vector zero(2), t(2);
    zero = 0.0;
    t(0) = 3.0; t(1) = -2.0;
    mfem::VectorConstantCoefficient zero_coef(zero), traction(t);
    physics.AddDirichlet({LeftAttr(2)}, zero_coef);
    physics.AddTraction({RightAttr(2)}, traction);
    physics.Finalize();
    cmf::LinearSolver linear(cfg.solver.linear, physics.FESpace());
    mfem::Vector u(physics.FESpace().GetTrueVSize());
    u = 0.0;
    cmf::SolveQuasiStatic(physics, linear, cfg.solver, u);
    const std::vector<cmf::Reaction> rx = physics.Reactions(u);
    CHECK(rx.size() == 1);
    // Traction T on X = 1, Y in [0, 1]: force T, moment int (X T_y - Y T_x) dY = T_y - T_x / 2.
    const double moment = t(1) - 0.5 * t(0);
    std::printf("  reaction: force (%.10f, %.10f), moment %.10f (loads: %.1f, %.1f, %.1f); max |u| %.2e\n",
                rx[0].force[0], rx[0].force[1], rx[0].moment[2], t(0), t(1), moment, u.Normlinf());
    CHECK_CLOSE(rx[0].force[0], -t(0), 1e-9);
    CHECK_CLOSE(rx[0].force[1], -t(1), 1e-9);
    CHECK_CLOSE(rx[0].moment[2], -moment, 1e-9);
    CHECK_MSG(u.Normlinf() > 1e-3, "displacements large enough for current arms to break the balance");

    mfem::ConstantCoefficient p(1.0);
    CHECK_THROWS(physics.AddPressure({3}, p, true), cmf::ConfigError,
                 "follower_pressure is not used by model 'linear_elastic'");
    physics.AddPressure({3}, p, false); // the dead pressure is the same load
  }
  {
    cmf::AppConfig bad = cfg;
    cmf::BoundaryCondition bc;
    bc.attr = {2};
    bc.expression = {"1.0"};
    bc.type = "follower_pressure";
    bad.bcs.traction.push_back(bc);
    CHECK_THROWS(cmf::MakeSolidProblem(*pmesh, bad), cmf::ConfigError, "bcs.traction[0]: traction type follower_pressure");
    bad = cfg;
    bad.solver.predictor = "tangent";
    CHECK_THROWS(cmf::MakeSolidProblem(*pmesh, bad), cmf::ConfigError,
                 "'solver.predictor': tangent is not used by model 'linear_elastic'");
    bad.material.model = "st_venant_kirchhoff";
    std::unique_ptr<cmf::SolidProblem> ok = cmf::MakeSolidProblem(*pmesh, bad);
    CHECK(ok->Description().find("small strain") == std::string::npos);
    std::unique_ptr<cmf::SolidProblem> lin = cmf::MakeSolidProblem(*pmesh, cfg);
    CHECK_MSG(lin->Description().find("linear_elastic (small strain)") != std::string::npos, lin->Description());
  }
}

// 9. The round-off floor of the residual. A slender beam in bending puts the
// floor of |R| / |R0| (eps |K| |u| / |f|) near 1e-10; with a Newton tolerance
// below it, a linear problem is accepted at the floor after a second step
// that cannot reduce the residual (SolidMechanicsTL::IsLinear ->
// NewtonConfig::linear_problem), with the same solution; a nonlinear model
// under the same tolerance still reports the failed line search.
void ResidualFloorTest()
{
  auto solve = [](const std::string &model, double rtol, double load, mfem::Vector &u)
  {
    ElementCase ec{2, "quad", 4, 0.0};
    cmf::AppConfig cfg = BoxConfig(ec, 2, model);
    cfg.mesh.box.nx = 80;
    cfg.mesh.box.sx = 20.0;
    cfg.solver.newton.rtol = rtol;
    std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    cmf::SolidMechanicsTL physics(*pmesh, cfg, cmf::MakeMaterial(cfg.material));
    mfem::Vector zero(2), t(2);
    zero = 0.0;
    t(0) = 0.0; t(1) = load;
    mfem::VectorConstantCoefficient zero_coef(zero), traction(t);
    physics.AddDirichlet({LeftAttr(2)}, zero_coef);
    physics.AddTraction({RightAttr(2)}, traction);
    physics.Finalize();
    cmf::LinearSolver linear(cfg.solver.linear, physics.FESpace());
    u.SetSize(physics.FESpace().GetTrueVSize());
    u = 0.0;
    return cmf::SolveQuasiStatic(physics, linear, cfg.solver, u);
  };
  mfem::Vector u_ref, u_floor, u_svk;
  const cmf::QuasiStaticReport ref = solve("linear_elastic", 1e-8, 1e-3, u_ref);
  const cmf::NewtonReport &rn = ref.steps.back().newton;
  CHECK_MSG(ref.converged && rn.iterations == 1 && !rn.at_floor, "floor test: rtol 1e-8 is met by the first step");
  const cmf::QuasiStaticReport low = solve("linear_elastic", 1e-15, 1e-3, u_floor);
  const cmf::NewtonReport &ln = low.steps.back().newton;
  std::printf("  slender beam: first step |R|/|R0| = %.2e; rtol 1e-15: converged %d, at the floor %d, its %d, "
              "|R|/|R0| = %.2e, solution rel diff %.1e\n", rn.residual / rn.initial_residual, int(low.converged),
              int(ln.at_floor), ln.iterations, ln.residual / ln.initial_residual, RelDiff(u_floor, u_ref));
  CHECK_MSG(rn.residual / rn.initial_residual > 1e-13, "floor test: the floor of this problem lies above 1e-13");
  CHECK_MSG(low.converged && ln.at_floor, "floor test: a linear problem is accepted at the round-off floor");
  CHECK_MSG(ln.iterations <= 3, "floor test: at most three steps, got " + std::to_string(ln.iterations));
  CHECK_MSG(ln.residual > 1e-15 * ln.initial_residual, "floor test: the tolerance was indeed out of reach");
  CHECK_MSG(RelDiff(u_floor, u_ref) <= 1e-9, "floor test: the accepted state is the solution");
  // The nonlinear model in its linear range, same tolerance: unchanged behaviour.
  const cmf::QuasiStaticReport svk = solve("st_venant_kirchhoff", 1e-15, 1e-6, u_svk);
  const cmf::NewtonReport &sn = svk.steps.back().newton;
  std::printf("  st_venant_kirchhoff, rtol 1e-15: converged %d (%s)\n", int(svk.converged), sn.failure.c_str());
  CHECK_MSG(!svk.converged && !sn.at_floor, "floor test: a nonlinear problem is not accepted at the floor");
}

// 8. Output quantities of homogeneous Hooke states at a strain of 5 percent,
// where the finite-strain measures (J^{-1} P F^T, det F) would be off by
// percents. A state is an affine displacement u = H X; the expected values
// come from (E, nu) and H alone: eps = sym(H), sigma = lambda tr(eps) I +
// 2 mu eps, F = I + H, J = 1 + tr(eps), W = sigma:eps / 2.
using Mat3 = cmf::tensor<double, 3, 3>;

struct HookeState
{
  std::string label;
  int dim = 3;
  bool plane_stress = false;
  Mat3 H;                          // full 3x3 gradient, out-of-plane strain included
  bool whole_boundary = true;      // else uniaxial: u_x on the end faces, rollers on X_i = 0
};

std::vector<std::pair<std::string, std::vector<double>>> ExpectedQuantities(const HookeState &s)
{
  const cmf::LameParameters lame = cmf::LameFromYoungPoisson(kE, kNu);
  const Mat3 eps = cmf::sym(s.H);
  const Mat3 sigma = (lame.lambda * cmf::tr(eps)) * cmf::I<3>() + (2.0 * lame.mu) * eps;
  const Mat3 F = cmf::I<3>() + s.H;
  auto voigt = [](const Mat3 &A)
  { return std::vector<double>{A(0, 0), A(1, 1), A(2, 2), A(0, 1), A(1, 2), A(0, 2)}; };
  auto rows = [](const Mat3 &A)
  {
    std::vector<double> v;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) { v.push_back(A(i, j)); }
    return v;
  };
  const Mat3 dev = cmf::dev(sigma);
  std::vector<std::pair<std::string, std::vector<double>>> x;
  x.push_back({"cauchy_stress", voigt(sigma)});
  x.push_back({"pk1_stress", rows(sigma)});
  x.push_back({"deformation_gradient", rows(F)});
  x.push_back({"strain", voigt(eps)});
  x.push_back({"jacobian", {1.0 + cmf::tr(eps)}});
  x.push_back({"vonmises", {std::sqrt(1.5 * cmf::ddot(dev, dev))}});
  x.push_back({"energy_density", {0.5 * cmf::ddot(sigma, eps)}});
  if (s.plane_stress) { x.push_back({"thickness_stretch", {F(2, 2)}}); }
  return x;
}

void HomogeneousOutputTest(const HookeState &s, const std::string &projection)
{
  const ElementCase ec = s.dim == 2 ? kElementCases[0] : kElementCases[2];
  cmf::AppConfig cfg = BoxConfig(ec, 2);
  cfg.plane = s.plane_stress ? "stress" : "strain";
  cfg.output.fields = {"displacement", "cauchy_stress", "pk1_stress", "deformation_gradient", "strain",
                       "jacobian", "vonmises", "energy_density"};
  if (s.plane_stress) { cfg.output.fields.push_back("thickness_stretch"); }
  cfg.output.quadrature_at = {"nodes", "elements", "quadrature_points"};
  cfg.output.nodal_projection = projection;
  std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  cmf::SolidMechanicsTL physics(*pmesh, cfg, cmf::MakeMaterial(cfg.material, s.plane_stress));
  const int dim = s.dim;
  const Mat3 H = s.H;
  mfem::VectorFunctionCoefficient affine(dim, [dim, H](const mfem::Vector &X, mfem::Vector &u)
  {
    u.SetSize(dim);
    for (int i = 0; i < dim; i++)
    {
      u(i) = 0.0;
      for (int j = 0; j < dim; j++) { u(i) += H(i, j) * X(j); }
    }
  });
  if (s.whole_boundary) { physics.AddDirichlet(AllAttrs(dim), affine); }
  else
  {
    // u_x on the end faces; u_y = 0 on Y = 0 and u_z = 0 on Z = 0 (the affine
    // field vanishes there); the other faces are traction free.
    cmf::BCOptions ox, oy, oz;
    ox.components = {0};
    oy.components = {1};
    oz.components = {2};
    physics.AddDirichlet({LeftAttr(dim)}, affine, ox);
    physics.AddDirichlet({RightAttr(dim)}, affine, ox);
    physics.AddDirichlet({dim == 2 ? 1 : 2}, affine, oy);
    if (dim == 3) { physics.AddDirichlet({1}, affine, oz); }
  }
  physics.Finalize();
  cmf::LinearSolver linear(cfg.solver.linear, physics.FESpace());
  mfem::Vector u(physics.FESpace().GetTrueVSize());
  u = 0.0;
  const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(physics, linear, cfg.solver, u);
  physics.UpdateFields(u);

  mfem::ParGridFunction exact(&physics.FESpace());
  exact.ProjectCoefficient(affine);
  exact -= physics.Displacement();
  const double u_err = exact.Normlinf();

  cmf::FieldRegistry fields;
  physics.RegisterFields(fields);
  const std::vector<double> center(dim, 0.5);
  double nodal = 0.0, elem = 0.0, qp = 0.0;
  for (const auto &kv : ExpectedQuantities(s))
  {
    const std::vector<double> &want = kv.second;
    const std::vector<double> gn = cmf::ProbeVector(fields.Get(kv.first), center);
    const std::vector<double> ge = cmf::ProbeVector(fields.Get(kv.first + "_elem"), center);
    const mfem::QuadratureFunction &qf = fields.GetQ(kv.first + "_qp");
    CHECK_MSG(gn.size() == want.size() && ge.size() == want.size(), s.label + ": components of " + kv.first);
    for (std::size_t c = 0; c < want.size(); c++)
    {
      nodal = std::max(nodal, std::abs(gn[c] - want[c]));
      elem = std::max(elem, std::abs(ge[c] - want[c]));
    }
    for (int i = 0; i < qf.Size(); i++) { qp = std::max(qp, std::abs(qf(i) - want[i % want.size()])); }
  }
  const double scale = kE * 0.05; // the stress level
  const cmf::NewtonReport &newton = report.steps.back().newton;
  std::printf("  %-28s %-9s: errors u %.1e, quantities nodal %.1e elem %.1e qp %.1e (stress scale %.1f), its %d\n",
              s.label.c_str(), projection.c_str(), u_err, nodal, elem, qp, scale, newton.iterations);
  CHECK_MSG(newton.converged && newton.iterations == 1, s.label + ": one Newton iteration");
  CHECK_MSG(u_err <= 1e-12, s.label + " " + projection + ": displacement " + std::to_string(u_err));
  CHECK_MSG(nodal <= 1e-10 * scale, s.label + " " + projection + ": nodal presentations " + std::to_string(nodal));
  CHECK_MSG(elem <= 1e-10 * scale, s.label + " " + projection + ": element presentations " + std::to_string(elem));
  CHECK_MSG(qp <= 1e-10 * scale, s.label + " " + projection + ": quadrature-point values " + std::to_string(qp));

  if (!s.whole_boundary)
  {
    // The end faces carry -/+ sigma_xx A with the reference area A = 1 (a
    // current area would differ by 2 nu eps = 3 percent); the rollers carry
    // no load.
    const std::vector<cmf::Reaction> rx = physics.Reactions(u);
    const double sxx = ExpectedQuantities(s)[0].second[0];
    CHECK_MSG(std::abs(sxx - kE * 0.05) <= 1e-12 * scale, s.label + ": sigma_xx = E eps");
    CHECK_MSG(std::abs(rx[0].force[0] + sxx) <= 1e-9 * scale, s.label + ": left face reaction -sigma_xx A");
    CHECK_MSG(std::abs(rx[1].force[0] - sxx) <= 1e-9 * scale, s.label + ": right face reaction sigma_xx A");
    for (std::size_t k = 2; k < rx.size(); k++)
    {
      for (int c = 0; c < dim; c++)
      {
        CHECK_MSG(std::abs(rx[k].force[c]) <= 1e-9 * scale, s.label + ": rollers carry no load");
      }
    }
  }
}

void HomogeneousOutputTests()
{
  const double e = 0.05, nu = kNu;
  std::vector<HookeState> states;
  {
    HookeState s;
    s.label = "3D uniaxial stress";
    s.H(0, 0) = e; s.H(1, 1) = -nu * e; s.H(2, 2) = -nu * e;
    s.whole_boundary = false;
    states.push_back(s);
  }
  {
    HookeState s;
    s.label = "3D general (with rotation)";
    const double G[3][3] = {{0.05, -0.03, 0.02}, {0.01, -0.02, 0.04}, {-0.015, 0.025, 0.03}};
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) { s.H(i, j) = G[i][j]; }
    states.push_back(s);
  }
  {
    HookeState s;
    s.label = "plane strain general";
    s.dim = 2;
    s.H(0, 0) = 0.05; s.H(0, 1) = -0.03; s.H(1, 0) = 0.01; s.H(1, 1) = -0.02;
    states.push_back(s);
  }
  {
    HookeState s;
    s.label = "plane stress uniaxial";
    s.dim = 2;
    s.plane_stress = true;
    s.H(0, 0) = e; s.H(1, 1) = -nu * e; s.H(2, 2) = -nu * e;
    s.whole_boundary = false;
    states.push_back(s);
  }
  {
    HookeState s;
    s.label = "plane stress general";
    s.dim = 2;
    s.plane_stress = true;
    s.H(0, 0) = 0.05; s.H(0, 1) = -0.03; s.H(1, 0) = 0.01; s.H(1, 1) = -0.02;
    s.H(2, 2) = -nu / (1.0 - nu) * (s.H(0, 0) + s.H(1, 1)); // sigma_33 = 0
    states.push_back(s);
  }
  for (const HookeState &s : states)
  {
    if (s.plane_stress)
    {
      const double s33 = ExpectedQuantities(s)[0].second[2];
      CHECK_MSG(std::abs(s33) <= 1e-13 * kE, s.label + ": the expected state has sigma_33 = 0");
    }
    HomogeneousOutputTest(s, "averaged");
    HomogeneousOutputTest(s, "projected");
  }
}

} // namespace

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  if (mfem::Mpi::WorldSize() != 1)
  {
    if (mfem::Mpi::Root()) { std::cout << "test_linear_elasticity is a serial test" << std::endl; }
    return 1;
  }
  std::cout << "operator: mfem::ElasticityIntegrator and the hyperelastic tangents at u = 0" << std::endl;
  OperatorTests();
  std::cout << "linearity of the solve" << std::endl;
  LinearityTests();
  std::cout << "patch tests" << std::endl;
  PatchTests();
  std::cout << "manufactured solution" << std::endl;
  ManufacturedSolutionTest();
  std::cout << "finite-strain limit" << std::endl;
  FiniteStrainLimitTest();
  std::cout << "reactions and configuration errors" << std::endl;
  ReactionAndConfigTests();
  std::cout << "output quantities of homogeneous states" << std::endl;
  HomogeneousOutputTests();
  std::cout << "round-off floor of the residual" << std::endl;
  ResidualFloorTest();
  return cmf_test::Report("test_linear_elasticity");
}

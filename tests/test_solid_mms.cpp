// S3 gate: patch test on a perturbed quad mesh, MMS convergence for p = 1, 2,
// Newton quadratic contraction, and an assembled-Jacobian consistency check.
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "base/mesh_input.hpp"
#include "kernels/total_lagrangian.hpp"
#include "materials/materials.hpp"
#include "mfem.hpp"
#include "physics/solid_mechanics_tl.hpp"
#include "solvers/linear_solver.hpp"
#include "solvers/quasi_static.hpp"
#include "test_util.hpp"

using cmf::tensor;

namespace
{

const double kE = 250.0, kNu = 0.3, kRho0 = 1.0;

cmf::AppConfig BaseConfig(int nx, int order, double perturb, const std::string &model)
{
  cmf::AppConfig cfg;
  cfg.mesh.cartesian = true;
  cfg.mesh.box.nx = nx;
  cfg.mesh.box.ny = nx;
  cfg.mesh.order = order;
  cfg.mesh.perturb = perturb;
  cfg.material.model = model;
  cfg.material.E = kE;
  cfg.material.nu = kNu;
  cfg.material.rho0 = kRho0;
  cfg.solver.newton.rtol = 1e-10;
  cfg.solver.newton.atol = 1e-14;
  cfg.solver.newton.max_it = 25;
  cfg.solver.newton.print_level = 0;
  cfg.solver.linear.rtol = 1e-13;
  cfg.solver.linear.max_it = 1000;
  return cfg;
}

// Manufactured displacement given as u(X) and its analytic gradient.
struct Manufactured
{
  std::function<void(const mfem::Vector &, mfem::Vector &)> u;
  std::function<tensor<double, 2, 2>(const mfem::Vector &)> grad;
};

// Affine: u = A X + c.
Manufactured Affine(const tensor<double, 2, 2> &A, const tensor<double, 2> &c)
{
  Manufactured m;
  m.u = [A, c](const mfem::Vector &X, mfem::Vector &u)
  {
    u.SetSize(2);
    for (int i = 0; i < 2; i++)
    {
      u(i) = c(i) + A(i, 0) * X(0) + A(i, 1) * X(1);
    }
  };
  m.grad = [A](const mfem::Vector &) { return A; };
  return m;
}

// Smooth: u = alpha (sin(pi X) sin(pi Y), X^2 Y (1 - Y)).
Manufactured Smooth(double alpha)
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
  return m;
}

// b = -(1/rho0) Div P(X), with Div P by central differences in X (step 1e-5)
// of the closed-form composition P(X) = PK1(I + Grad u(X)).
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
      const tensor<double, 2, 2> Pp = cmf::QPointStress<Material, 2>(mat_, m_.grad(Xp));
      const tensor<double, 2, 2> Pm = cmf::QPointStress<Material, 2>(mat_, m_.grad(Xm));
      for (int i = 0; i < 2; i++) { b(i) += (Pp(i, j) - Pm(i, j)) / (2.0 * h); }
    }
    b *= -1.0 / rho0_;
  }

private:
  Manufactured m_;
  Material mat_;
  double rho0_;
};

struct SolveResult
{
  double l2_error = 0.0;
  double max_nodal_error = 0.0;
  cmf::NewtonReport newton;
  int ndofs = 0;
};

// Solve the manufactured problem: exact u on the whole boundary, body force
// from the manufactured stress divergence (zero for affine u).
template <typename Material>
SolveResult SolveManufactured(const cmf::AppConfig &cfg, const Manufactured &m,
                              const Material &material, bool with_body_force)
{
  std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  cmf::SolidMechanicsTL physics(*pmesh, cfg, cmf::Material(material));
  mfem::VectorFunctionCoefficient exact(2, m.u);
  physics.AddDirichlet({1, 2, 3, 4}, exact);
  ManufacturedBodyForce<Material> body(m, material, cfg.material.rho0);
  if (with_body_force) { physics.SetBodyForce(body); }
  physics.Finalize();

  cmf::LinearSolver linear(cfg.solver.linear, physics.FESpace());
  mfem::Vector u(physics.FESpace().GetTrueVSize());
  u = 0.0;
  cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(physics, linear, cfg.solver, u);

  SolveResult r;
  r.newton = report.steps.back().newton;
  r.ndofs = physics.FESpace().GetTrueVSize();
  physics.UpdateFields(u);
  r.l2_error = physics.Displacement().ComputeL2Error(exact);
  mfem::ParGridFunction interp(&physics.FESpace());
  interp.ProjectCoefficient(exact);
  interp -= physics.Displacement();
  r.max_nodal_error = interp.Normlinf();
  return r;
}

template <typename Material>
void PatchTest(const std::string &model, const Material &material)
{
  for (int order = 1; order <= 2; order++)
  {
    // Linear regime: the first Newton step is the linear-elastic solution and
    // leaves an O(|A|) relative residual; the second step removes it to
    // O(|A|^3) relative, which must fall below rtol = 1e-10.
    tensor<double, 2, 2> A;
    A(0, 0) = 3.0e-5; A(0, 1) = -1.75e-5;
    A(1, 0) = 2.25e-5; A(1, 1) = -3.75e-5;
    tensor<double, 2> c;
    c(0) = 0.75e-5; c(1) = -0.5e-5;
    cmf::AppConfig cfg = BaseConfig(5, order, 0.3, model);
    SolveResult r = SolveManufactured(cfg, Affine(A, c), material, false);
    std::printf("  patch %s p=%d: max nodal error %.3e, L2 error %.3e, newton its %d, |R|:",
                model.c_str(), order, r.max_nodal_error, r.l2_error, r.newton.iterations);
    for (const cmf::NewtonIteration &it : r.newton.history) { std::printf(" %.2e", it.residual); }
    std::printf("\n");
    CHECK_MSG(r.newton.converged, model + " p=" + std::to_string(order) + " patch converged");
    CHECK_MSG(r.newton.iterations <= 2, model + " p=" + std::to_string(order) +
              " patch Newton iterations " + std::to_string(r.newton.iterations));
    CHECK_MSG(r.max_nodal_error <= 1e-12, model + " p=" + std::to_string(order) +
              " patch max nodal error " + std::to_string(r.max_nodal_error));
    CHECK_MSG(r.l2_error <= 1e-12, model + " p=" + std::to_string(order) +
              " patch L2 error " + std::to_string(r.l2_error));

    // Finite strain: the affine field is still reproduced exactly (P is
    // constant, so Div P = 0 holds for every material).
    tensor<double, 2, 2> B;
    B(0, 0) = 0.15; B(0, 1) = -0.08;
    B(1, 0) = 0.12; B(1, 1) = -0.10;
    cfg.solver.newton.rtol = 1e-12;
    cfg.solver.load_steps = 4; // the zero interior guess would invert boundary elements at full load
    SolveResult rf = SolveManufactured(cfg, Affine(B, c), material, false);
    std::printf("  patch %s p=%d finite strain (4 load steps): max nodal error %.3e, last-step newton its %d\n",
                model.c_str(), order, rf.max_nodal_error, rf.newton.iterations);
    CHECK_MSG(rf.newton.converged, model + " finite-strain patch converged");
    CHECK_MSG(rf.max_nodal_error <= 1e-12, model + " p=" + std::to_string(order) +
              " finite-strain patch max nodal error " + std::to_string(rf.max_nodal_error));
  }
}

void ConvergenceTest()
{
  const cmf::NeoHookean material{cmf::LameFromYoungPoisson(kE, kNu).mu,
                                 cmf::LameFromYoungPoisson(kE, kNu).lambda};
  const double alpha = 0.03; // max |Grad u| ~ alpha pi ~ 0.094
  const Manufactured m = Smooth(alpha);
  cmf::NewtonReport finest;
  for (int order = 1; order <= 2; order++)
  {
    std::vector<double> errors;
    const int base = order == 1 ? 8 : 4;
    for (int level = 0; level < 4; level++)
    {
      const int nx = base << level;
      cmf::AppConfig cfg = BaseConfig(nx, order, 0.0, "neo_hookean");
      cfg.solver.newton.rtol = 1e-12;
      SolveResult r = SolveManufactured(cfg, m, material, true);
      CHECK_MSG(r.newton.converged, "MMS p=" + std::to_string(order) + " nx=" +
                std::to_string(nx) + " converged");
      errors.push_back(r.l2_error);
      std::printf("  mms p=%d nx=%3d dofs %6d: L2 error %.4e, newton its %d\n",
                  order, nx, r.ndofs, r.l2_error, r.newton.iterations);
      if (order == 2 && level == 3) { finest = r.newton; }
    }
    for (std::size_t k = 0; k + 1 < errors.size(); k++)
    {
      const double rate = std::log2(errors[k] / errors[k + 1]);
      std::printf("  mms p=%d rate %zu: %.3f\n", order, k + 1, rate);
      CHECK_MSG(rate >= order + 0.9, "MMS p=" + std::to_string(order) + " rate " +
                std::to_string(rate) + " >= " + std::to_string(order + 0.9));
    }
  }

  // Newton quality on the finest p = 2 mesh: the last two steps above the
  // round-off floor contract quadratically (order estimate >= 1.5).
  std::vector<double> res;
  for (const cmf::NewtonIteration &it : finest.history)
  {
    if (it.residual > 1e-12 * finest.initial_residual) { res.push_back(it.residual); }
  }
  std::printf("  newton history (finest p=2):");
  for (const cmf::NewtonIteration &it : finest.history)
  {
    std::printf(" %.2e", it.residual);
  }
  std::printf("\n");
  CHECK_MSG(res.size() >= 3, "at least three residuals above the floor");
  if (res.size() >= 3)
  {
    const std::size_t n = res.size();
    const double q = std::log(res[n - 1] / res[n - 2]) / std::log(res[n - 2] / res[n - 3]);
    std::printf("  newton contraction order estimate: %.3f\n", q);
    CHECK_MSG(q >= 1.5, "quadratic contraction estimate " + std::to_string(q) + " >= 1.5");
  }
  bool full_steps = true;
  for (const cmf::NewtonIteration &it : finest.history)
  {
    if (it.iteration > 0 && it.alpha != 1.0) { full_steps = false; }
  }
  CHECK_MSG(full_steps, "no line-search damping needed on the finest MMS mesh");
}

// Assembled Jacobian vs central differences of the residual along a random
// direction, on a small perturbed mesh with a nonzero state.
void JacobianConsistencyTest()
{
  for (const std::string model : {"neo_hookean", "st_venant_kirchhoff"})
  {
    cmf::AppConfig cfg = BaseConfig(3, 2, 0.2, model);
    std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    cmf::MaterialConfig mc;
    mc.model = model; mc.E = kE; mc.nu = kNu;
    cmf::SolidMechanicsTL physics(*pmesh, cfg, cmf::MakeMaterial(mc));
    const Manufactured m = Smooth(0.05);
    mfem::VectorFunctionCoefficient exact(2, m.u);
    physics.AddDirichlet({4}, exact);
    mfem::Vector t(2);
    t(0) = 0.0; t(1) = 2.0;
    mfem::VectorConstantCoefficient traction(t);
    physics.AddTraction({2}, traction);
    physics.Finalize();
    physics.SetLoadFactor(0.7);

    const int n = physics.FESpace().GetTrueVSize();
    std::mt19937 rng(7u);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    mfem::Vector u(n), v(n), Jv(n), rp(n), rm(n), up(n), um(n);
    for (int i = 0; i < n; i++) { u(i) = 0.05 * unit(rng); v(i) = unit(rng); }
    physics.ApplyDirichlet(u);
    for (int i = 0; i < physics.EssentialTrueDofs().Size(); i++)
    {
      v(physics.EssentialTrueDofs()[i]) = 0.0;
    }
    mfem::Operator &J = physics.GetGradient(u);
    J.Mult(v, Jv);
    const double eps = 1e-6;
    up = u; up.Add(eps, v);
    um = u; um.Add(-eps, v);
    physics.Mult(up, rp);
    physics.Mult(um, rm);
    rp -= rm;
    rp /= 2.0 * eps;
    rp -= Jv;
    const double rel = rp.Normlinf() / Jv.Normlinf();
    std::printf("  jacobian %s: |J v - FD| / |J v| = %.3e\n", model.c_str(), rel);
    CHECK_MSG(rel <= 1e-6, model + " Jacobian vs FD relative error " + std::to_string(rel));
    // Residual at essential dofs is zero and the Jacobian has identity rows there.
    mfem::Vector r(n);
    physics.Mult(u, r);
    double ess_res = 0.0;
    for (int i = 0; i < physics.EssentialTrueDofs().Size(); i++)
    {
      ess_res = std::max(ess_res, std::abs(r(physics.EssentialTrueDofs()[i])));
    }
    CHECK_MSG(ess_res == 0.0, "residual vanishes at essential dofs");
  }
}

} // namespace

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  const cmf::LameParameters lame = cmf::LameFromYoungPoisson(kE, kNu);
  std::cout << "patch tests" << std::endl;
  PatchTest("neo_hookean", cmf::NeoHookean{lame.mu, lame.lambda});
  PatchTest("st_venant_kirchhoff", cmf::StVenantKirchhoff{lame.mu, lame.lambda});
  std::cout << "jacobian consistency" << std::endl;
  JacobianConsistencyTest();
  std::cout << "mms convergence" << std::endl;
  ConvergenceTest();
  return cmf_test::Report("test_solid_mms");
}

// ST1 and ST2 gates: the scalar CG kernel (kernels/scalar_flux.hpp) and the
// scalar transport physics (physics/scalar_transport.hpp).
// Kernel: the assembled matrix of constant laws against MFEM's stock
// integrators (both convection forms); the Jacobian of nonlinear laws against
// finite differences; the scalar patch test; the Newton contraction on a
// nonlinear conductivity; the two convection forms for a constant velocity;
// erf and erfc in the expressions.
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "base/coefficients.hpp"
#include "base/expression.hpp"
#include "base/mesh_input.hpp"
#include "kernels/scalar_flux.hpp"
#include "materials/scalar_transport_model.hpp"
#include "mfem.hpp"
#include "solvers/linear_solver.hpp"
#include "solvers/newton.hpp"
#include "test_util.hpp"

using cmf::tensor;

namespace
{

using Model = cmf::ScalarTransportModel;
using Kernel = cmf::ScalarFluxIntegrator<Model>;

struct Box
{
  std::unique_ptr<mfem::ParMesh> mesh;
  std::unique_ptr<mfem::H1_FECollection> fec;
  std::unique_ptr<mfem::ParFiniteElementSpace> fes;
  int dim = 2;
  int order = 1;
};

// A unit box of n^dim elements of the given type and order, optionally jittered.
Box MakeBox(int dim, const std::string &element, int n, int order, double perturb = 0.0)
{
  cmf::MeshConfig mc;
  mc.cartesian = true;
  mc.box.dim = dim;
  mc.box.element = element;
  mc.box.nx = mc.box.ny = mc.box.nz = n;
  mc.perturb = perturb;
  mc.order = order;
  Box b;
  b.mesh = cmf::BuildParMesh(MPI_COMM_WORLD, mc);
  b.fec = std::make_unique<mfem::H1_FECollection>(order, dim);
  b.fes = std::make_unique<mfem::ParFiniteElementSpace>(b.mesh.get(), b.fec.get());
  b.dim = dim;
  b.order = order;
  return b;
}

double GlobalNorm(const mfem::Vector &v)
{
  return std::sqrt(mfem::InnerProduct(MPI_COMM_WORLD, v, v));
}

double RelDiff(const mfem::Vector &a, const mfem::Vector &b)
{
  mfem::Vector d(a);
  d -= b;
  return GlobalNorm(d) / std::max(GlobalNorm(b), 1e-300);
}

void FillRandom(mfem::Vector &v, std::mt19937 &rng, double amplitude, double offset = 0.0)
{
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (int i = 0; i < v.Size(); i++) { v(i) = offset + amplitude * u(rng); }
}

// The boundary attribute of the whole boundary of a box: all of them.
mfem::Array<int> AllBoundary(const mfem::ParMesh &mesh)
{
  mfem::Array<int> marker(mesh.bdr_attributes.Max());
  marker = 1;
  return marker;
}

// The unknown of the wrapped form with the essential rows zeroed, for Newton.
class FormOperator : public mfem::Operator
{
public:
  explicit FormOperator(mfem::ParNonlinearForm &nlf) : mfem::Operator(nlf.ParFESpace()->GetTrueVSize()), nlf_(nlf) {}
  void Mult(const mfem::Vector &x, mfem::Vector &y) const override { nlf_.Mult(x, y); }
  mfem::Operator &GetGradient(const mfem::Vector &x) const override { return nlf_.GetGradient(x); }

private:
  mfem::ParNonlinearForm &nlf_;
};

// (a) Constant laws: the kernel's Jacobian equals the stock integrators, its
// residual is affine with that matrix, and the rate term carries the accepted
// state through the mass matrix.
void StockIntegratorTest()
{
  std::printf("kernel against the stock integrators\n");
  const double c = 2.5, kappa = 0.7, s = 1.3, dt = 0.1;
  std::mt19937 rng(11u);
  struct Case { int dim; const char *element; };
  for (const Case ec : {Case{2, "quad"}, Case{2, "tri"}, Case{3, "hex"}})
    for (int order = 1; order <= 2; order++)
      for (const bool conservative : {false, true})
      {
        Box b = MakeBox(ec.dim, ec.element, 3, order, 0.15);
        mfem::Vector bvec(ec.dim);
        bvec(0) = 1.0;
        bvec(1) = -2.0;
        if (ec.dim == 3) { bvec(2) = 0.5; }
        mfem::VectorConstantCoefficient beta(bvec);
        Model model;
        model.capacity = {c, 0.0, 0.0};
        model.conductivity = {kappa, 0.0, 0.0};
        model.reaction = s;
        model.convection = conservative ? cmf::ConvectionForm::Conservative : cmf::ConvectionForm::NonConservative;
        mfem::ParGridFunction u_old(b.fes.get());
        FillRandom(u_old, rng, 1.0);
        mfem::ParNonlinearForm nlf(b.fes.get());
        auto *kernel = new Kernel(model);
        kernel->SetVelocity(&beta);
        kernel->SetOldState(&u_old);
        kernel->SetDt(dt);
        nlf.AddDomainIntegrator(kernel);
        const int n = b.fes->GetTrueVSize();
        mfem::Vector x(n), v(n), Kv(n), Av(n), Rx(n), R0(n), zero(n), Mu(n);
        FillRandom(x, rng, 1.0);
        FillRandom(v, rng, 1.0);
        zero = 0.0;
        nlf.GetGradient(x).Mult(v, Kv);
        nlf.Mult(x, Rx);
        nlf.Mult(zero, R0);

        const mfem::Geometry::Type geom = b.mesh->GetElementBaseGeometry(0);
        const mfem::IntegrationRule &ir = mfem::IntRules.Get(geom, 2 * order + 3);
        mfem::ConstantCoefficient c_dt(c / dt), kappa_c(kappa), s_c(s);
        mfem::ParBilinearForm a(b.fes.get());
        std::vector<mfem::BilinearFormIntegrator *> integs = {
          new mfem::MassIntegrator(c_dt), new mfem::DiffusionIntegrator(kappa_c), new mfem::MassIntegrator(s_c)};
        if (conservative) { integs.push_back(new mfem::ConservativeConvectionIntegrator(beta)); }
        else { integs.push_back(new mfem::ConvectionIntegrator(beta)); }
        for (mfem::BilinearFormIntegrator *integ : integs)
        {
          integ->SetIntRule(&ir);
          a.AddDomainIntegrator(integ);
        }
        a.Assemble();
        a.Finalize();
        std::unique_ptr<mfem::HypreParMatrix> A(a.ParallelAssemble());
        A->Mult(v, Av);
        const double jac = RelDiff(Kv, Av);
        // R(x) - R(0) = A x.
        mfem::Vector Ax(n), dR(Rx);
        A->Mult(x, Ax);
        dR -= R0;
        const double affine = RelDiff(dR, Ax);
        // R(0) = -(c/dt) M u_n.
        mfem::ParBilinearForm m(b.fes.get());
        auto *mass = new mfem::MassIntegrator(c_dt);
        mass->SetIntRule(&ir);
        m.AddDomainIntegrator(mass);
        m.Assemble();
        m.Finalize();
        std::unique_ptr<mfem::HypreParMatrix> M(m.ParallelAssemble());
        mfem::Vector u_old_true(n);
        u_old.GetTrueDofs(u_old_true);
        M->Mult(u_old_true, Mu);
        Mu *= -1.0;
        const double rate = RelDiff(R0, Mu);
        std::printf("  %-4s p=%d %-16s (%5d dofs): |K v - A v|/|A v| = %.1e, affine %.1e, rate term %.1e\n",
                    ec.element, order, conservative ? "conservative" : "non-conservative", n, jac, affine, rate);
        const std::string what = std::string(ec.element) + " p=" + std::to_string(order) +
                                 (conservative ? " conservative" : " non-conservative");
        CHECK_MSG(jac <= 1e-13, what + ": Jacobian vs stock integrators " + std::to_string(jac));
        CHECK_MSG(affine <= 1e-12, what + ": residual affine " + std::to_string(affine));
        CHECK_MSG(rate <= 1e-12, what + ": rate term " + std::to_string(rate));
      }
}

// (b) Nonlinear laws, a velocity and a source of x, a reaction: the assembled
// Jacobian against central differences of the residual, with essential dofs
// on one face.
void JacobianTest()
{
  std::printf("Jacobian of nonlinear laws against finite differences\n");
  std::mt19937 rng(3u);
  struct Case { int dim; const char *element; };
  for (const Case ec : {Case{2, "quad"}, Case{2, "tri"}, Case{3, "hex"}})
    for (const bool conservative : {false, true})
    {
      Box b = MakeBox(ec.dim, ec.element, 3, 2, 0.1);
      Model model;
      model.capacity = {2.0, 0.5, 0.3};
      model.conductivity = {1.0, 0.4, 0.2};
      model.reaction = 0.7;
      model.convection = conservative ? cmf::ConvectionForm::Conservative : cmf::ConvectionForm::NonConservative;
      std::vector<std::string> bexpr = {"1 + x", "-2*y"};
      if (ec.dim == 3) { bexpr.push_back("0.5*z"); }
      cmf::ExpressionVectorCoefficient beta(bexpr);
      cmf::ExpressionCoefficient source("sin(x)*cos(y) + 0.3");
      mfem::ParGridFunction u_old(b.fes.get());
      FillRandom(u_old, rng, 0.5, 0.3);
      mfem::ParNonlinearForm nlf(b.fes.get());
      auto *kernel = new Kernel(model);
      kernel->SetVelocity(&beta);
      kernel->SetSource(&source);
      kernel->SetOldState(&u_old);
      kernel->SetDt(0.2);
      nlf.AddDomainIntegrator(kernel);
      mfem::Array<int> marker(b.mesh->bdr_attributes.Max()), ess;
      marker = 0;
      marker[0] = 1;
      b.fes->GetEssentialTrueDofs(marker, ess);
      nlf.SetEssentialTrueDofs(ess);
      const int n = b.fes->GetTrueVSize();
      mfem::Vector x(n), v(n), Jv(n), rp(n), rm(n), xp(n), xm(n);
      FillRandom(x, rng, 0.5, 0.3);
      mfem::Operator &J = nlf.GetGradient(x);
      const double eps = 1e-6;
      double worst = 0.0;
      for (int trial = 0; trial < 3; trial++)
      {
        FillRandom(v, rng, 1.0);
        for (int i = 0; i < ess.Size(); i++) { v(ess[i]) = 0.0; }
        J.Mult(v, Jv);
        xp = x;
        xp.Add(eps, v);
        xm = x;
        xm.Add(-eps, v);
        nlf.Mult(xp, rp);
        nlf.Mult(xm, rm);
        rp -= rm;
        rp /= 2.0 * eps;
        worst = std::max(worst, RelDiff(rp, Jv));
      }
      std::printf("  %-4s %-16s: |J v - dR/dv| / |J v| = %.2e\n", ec.element,
                  conservative ? "conservative" : "non-conservative", worst);
      CHECK_MSG(worst <= 1e-6, std::string(ec.element) + ": Jacobian vs finite differences " + std::to_string(worst));
    }
}

// (c) The scalar patch test: a linear field with the source that makes it a
// solution of the steady operator is reproduced to round-off on a jittered
// mesh, in both convection forms.
void PatchTest()
{
  std::printf("patch test\n");
  const double a0 = 0.3, kappa = 0.8, s = 0.9;
  struct Case { int dim; const char *element; };
  for (const Case ec : {Case{2, "quad"}, Case{2, "tri"}, Case{3, "hex"}})
    for (int order = 1; order <= 2; order++)
      for (const bool conservative : {false, true})
      {
        Box b = MakeBox(ec.dim, ec.element, 3, order, 0.2);
        const int dim = ec.dim;
        mfem::Vector bvec(dim), grad(dim);
        bvec(0) = 1.0; bvec(1) = -2.0;
        grad(0) = 1.5; grad(1) = -0.7;
        if (dim == 3) { bvec(2) = 0.5; grad(2) = 0.4; }
        auto linear = [&](const mfem::Vector &X) { return a0 + (grad * X); };
        mfem::FunctionCoefficient exact(linear);
        mfem::FunctionCoefficient source([&](const mfem::Vector &X) { return (bvec * grad) + s * linear(X); });
        mfem::VectorConstantCoefficient beta(bvec);
        Model model;
        model.conductivity = {kappa, 0.0, 0.0};
        model.reaction = s;
        model.convection = conservative ? cmf::ConvectionForm::Conservative : cmf::ConvectionForm::NonConservative;
        mfem::ParNonlinearForm nlf(b.fes.get());
        auto *kernel = new Kernel(model);
        kernel->SetVelocity(&beta);
        kernel->SetSource(&source);
        nlf.AddDomainIntegrator(kernel);
        mfem::Array<int> marker = AllBoundary(*b.mesh), ess;
        b.fes->GetEssentialTrueDofs(marker, ess);
        nlf.SetEssentialTrueDofs(ess);
        mfem::ParGridFunction u(b.fes.get());
        u.ProjectCoefficient(exact);
        mfem::Vector x(b.fes->GetTrueVSize()), r(x.Size());
        u.GetTrueDofs(x);
        nlf.Mult(x, r);
        const double res = GlobalNorm(r), scale = GlobalNorm(x);
        std::printf("  %-4s p=%d %-16s: |R(u_exact)| / |u| = %.1e\n", ec.element, order,
                    conservative ? "conservative" : "non-conservative", res / scale);
        CHECK_MSG(res <= 1e-12 * scale, std::string(ec.element) + " p=" + std::to_string(order) + ": patch residual " +
                  std::to_string(res / scale));
      }
}

// The scalar linear solver of the framework (gmres_amg, scalar AMG) with a tight tolerance.
std::unique_ptr<cmf::LinearSolver> TightSolver(mfem::ParFiniteElementSpace &fes, const std::string &type = "gmres_amg")
{
  cmf::LinearSolverConfig lc;
  lc.type = type;
  lc.rtol = 1e-14;
  lc.atol = 0.0;
  lc.max_it = 2000;
  return cmf::MakeLinearSolver(lc, fes);
}

// (d) A strongly nonlinear conductivity, steady: Newton from a poor start
// contracts quadratically once in its basin.
void NewtonContractionTest()
{
  std::printf("Newton contraction on a nonlinear conductivity\n");
  Box b = MakeBox(2, "quad", 4, 2, 0.1);
  Model model;
  model.conductivity = {1.0, 3.0, 0.0}; // kappa = 1 + 3 u
  mfem::ParNonlinearForm nlf(b.fes.get());
  nlf.AddDomainIntegrator(new Kernel(model));
  // u = 0 on the left, u = 1 on the right; natural elsewhere.
  mfem::Array<int> marker(b.mesh->bdr_attributes.Max()), ess;
  marker = 0;
  marker[1] = marker[3] = 1;
  b.fes->GetEssentialTrueDofs(marker, ess);
  nlf.SetEssentialTrueDofs(ess);
  mfem::ParGridFunction g(b.fes.get());
  mfem::FunctionCoefficient right([](const mfem::Vector &X) { return X(0) > 0.5 ? 1.0 : 0.0; });
  g = 0.0;
  mfem::Array<int> right_marker(marker.Size());
  right_marker = 0;
  right_marker[1] = 1;
  g.ProjectBdrCoefficient(right, right_marker);
  mfem::Vector x(b.fes->GetTrueVSize());
  g.GetTrueDofs(x);
  FormOperator op(nlf);
  std::unique_ptr<cmf::LinearSolver> linear = TightSolver(*b.fes);
  cmf::NewtonConfig cfg;
  cfg.rtol = 1e-14;
  cfg.atol = 0.0;
  cfg.max_it = 30;
  cfg.print_level = 0;
  const cmf::NewtonReport report = cmf::DampedNewtonSolve(op, *linear, x, cfg, MPI_COMM_WORLD);
  std::printf("  newton history:");
  for (const cmf::NewtonIteration &it : report.history) { std::printf(" %.2e", it.residual); }
  std::printf("\n");
  std::vector<double> res;
  for (const cmf::NewtonIteration &it : report.history)
  {
    if (it.residual > 1e-11 * report.initial_residual) { res.push_back(it.residual); }
  }
  CHECK_MSG(res.size() >= 3, "at least three residuals above the floor");
  if (res.size() >= 3)
  {
    const std::size_t n = res.size();
    const double q = std::log(res[n - 1] / res[n - 2]) / std::log(res[n - 2] / res[n - 3]);
    std::printf("  contraction order estimate: %.3f\n", q);
    CHECK_MSG(q >= 1.8, "quadratic contraction estimate " + std::to_string(q) + " >= 1.8");
  }
  CHECK_MSG(report.converged, "Newton converged");
  // The 1D solution: (1 + 3u) u' = const, i.e. u + 3u^2/2 linear in x.
  mfem::ParGridFunction u(b.fes.get());
  u.SetFromTrueDofs(x);
  mfem::FunctionCoefficient exact([](const mfem::Vector &X)
  {
    const double phi = X(0) * (1.0 + 1.5); // phi(1) = 1 + 3/2
    return (-1.0 + std::sqrt(1.0 + 6.0 * phi)) / 3.0;
  });
  const double err = u.ComputeL2Error(exact);
  std::printf("  L2 error against the 1D closed form: %.2e (4 x 4 p = 2, jittered)\n", err);
  CHECK_MSG(err <= 4e-3, "closed form of the nonlinear bar " + std::to_string(err));
}

// (e) The two convection forms coincide for a constant velocity with
// Dirichlet data all around (the boundary term of the integration by parts
// vanishes on the test functions).
void ConvectionFormsTest()
{
  std::printf("conservative and non-conservative convection for a constant velocity\n");
  Box b = MakeBox(2, "tri", 4, 3, 0.1);
  mfem::Vector bvec(2);
  bvec(0) = 1.0;
  bvec(1) = -2.0;
  mfem::VectorConstantCoefficient beta(bvec);
  cmf::ExpressionCoefficient source("sin(3*pi*x)*sin(3*pi*y)");
  mfem::FunctionCoefficient data([](const mfem::Vector &X) { return 0.2 + X(0) * X(1); });
  std::vector<mfem::Vector> solutions;
  for (const bool conservative : {false, true})
  {
    Model model;
    model.conductivity = {0.1, 0.0, 0.0};
    model.reaction = 1.0;
    model.convection = conservative ? cmf::ConvectionForm::Conservative : cmf::ConvectionForm::NonConservative;
    mfem::ParNonlinearForm nlf(b.fes.get());
    auto *kernel = new Kernel(model);
    kernel->SetVelocity(&beta);
    kernel->SetSource(&source);
    nlf.AddDomainIntegrator(kernel);
    mfem::Array<int> marker = AllBoundary(*b.mesh), ess;
    b.fes->GetEssentialTrueDofs(marker, ess);
    nlf.SetEssentialTrueDofs(ess);
    mfem::ParGridFunction g(b.fes.get());
    g = 0.0;
    g.ProjectBdrCoefficient(data, marker);
    mfem::Vector x(b.fes->GetTrueVSize());
    g.GetTrueDofs(x);
    FormOperator op(nlf);
    std::unique_ptr<cmf::LinearSolver> linear = TightSolver(*b.fes);
    cmf::NewtonConfig cfg;
    cfg.rtol = 1e-13;
    cfg.atol = 0.0;
    cfg.max_it = 5;
    cfg.print_level = 0;
    cfg.linear_problem = true;
    const cmf::NewtonReport report = cmf::DampedNewtonSolve(op, *linear, x, cfg, MPI_COMM_WORLD);
    CHECK_MSG(report.converged, "linear solve converged");
    solutions.push_back(x);
  }
  const double diff = RelDiff(solutions[0], solutions[1]);
  std::printf("  |u_cons - u_noncons| / |u| = %.2e\n", diff);
  CHECK_MSG(diff <= 1e-11, "the two convection forms agree " + std::to_string(diff));
}

void ExpressionTest()
{
  std::printf("erf and erfc in expressions\n");
  const double x = 0.37, t = 1.9;
  CHECK_CLOSE(cmf::Expression::Parse("erf(x) + erfc(t)").Eval(x, 0.0, 0.0, t), std::erf(x) + std::erfc(t), 1e-15);
  CHECK_CLOSE(cmf::Expression::Parse("erfc(-x)").Eval(x, 0.0, 0.0, 0.0), std::erfc(-x), 1e-15);
  // The select of if(): a NaN in the untaken branch is discarded.
  CHECK_CLOSE(cmf::Expression::Parse("if(t <= 0, 0, erfc(x/(2*sqrt(t))))").Eval(x, 0.0, 0.0, 0.0), 0.0, 0.0);
  CHECK_CLOSE(cmf::Expression::Parse("if(t <= 0, 0, erfc(x/(2*sqrt(t))))").Eval(x, 0.0, 0.0, t),
              std::erfc(x / (2.0 * std::sqrt(t))), 1e-15);
  CHECK_THROWS(cmf::Expression::Parse("erf(x, y)"), cmf::ConfigError, "expected ')' closing 'erf'");
}

} // namespace

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  ExpressionTest();
  StockIntegratorTest();
  JacobianTest();
  PatchTest();
  NewtonContractionTest();
  ConvectionFormsTest();
  return cmf_test::Report("test_scalar_transport");
}

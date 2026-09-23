// ST1 and ST2 gates: the scalar CG kernel (kernels/scalar_flux.hpp) and the
// scalar transport physics (physics/scalar_transport.hpp).
// Kernel: the assembled matrix of constant laws against MFEM's stock
// integrators (both convection forms); the Jacobian of nonlinear laws against
// finite differences; the scalar patch test; the Newton contraction on a
// nonlinear conductivity; the two convection forms for a constant velocity;
// erf and erfc in the expressions.
// Module: the patch test through the schema with the flows of two faces; the
// exactness of implicit Euler on a spatially constant state and the flow
// balance of a prescribed flux; manufactured solutions of the steady operator
// (L2 and H1 rates at k = 1, 2, 3, one Newton iteration, one assembly and one
// solver setup per run); first order in time and the reuse of the Jacobian;
// the nonlinear diffusion of the Kirchhoff case against its series solution
// and against the transformed linear solve; the initial condition, the point
// pin, a scheduled flux, the errors and the fields; the schema errors.
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "base/coefficients.hpp"
#include "base/expression.hpp"
#include "base/fields.hpp"
#include "base/mesh_input.hpp"
#include "base/probes.hpp"
#include "base/scalar_config.hpp"
#include "kernels/scalar_flux.hpp"
#include "materials/scalar_transport_model.hpp"
#include "mfem.hpp"
#include "physics/scalar_transport.hpp"
#include "solvers/linear_solver.hpp"
#include "solvers/newton.hpp"
#include "solvers/quasi_static.hpp"
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

double GlobalMax(double v)
{
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
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
        // The accepted state through its true dofs, consistent across ranks.
        mfem::ParGridFunction u_old(b.fes.get());
        mfem::Vector u_old_true(b.fes->GetTrueVSize());
        FillRandom(u_old_true, rng, 1.0);
        u_old.SetFromTrueDofs(u_old_true);
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
      mfem::Vector u_old_true(b.fes->GetTrueVSize());
      FillRandom(u_old_true, rng, 0.5, 0.3);
      u_old.SetFromTrueDofs(u_old_true);
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


// ----------------------------------------------------------------- module (ST2)

// A cartesian box config in place of the mesh file of a YAML string.
void UseBox(cmf::ScalarAppConfig &cfg, int dim, const std::string &element, int n, int order, double perturb = 0.0,
            double size = 1.0)
{
  cfg.mesh.file.clear();
  cfg.mesh.cartesian = true;
  cfg.mesh.box.dim = dim;
  cfg.mesh.box.element = element;
  cfg.mesh.box.nx = cfg.mesh.box.ny = cfg.mesh.box.nz = n;
  cfg.mesh.box.sx = cfg.mesh.box.sy = cfg.mesh.box.sz = size;
  cfg.mesh.perturb = perturb;
  cfg.mesh.order = order;
}

cmf::ScalarAppConfig ParseText(const std::string &text)
{
  return cmf::ParseScalarConfig(YAML::Load(text));
}

const cmf::Flow &Named(const std::vector<cmf::Flow> &flows, const std::string &name)
{
  for (const cmf::Flow &f : flows) { if (f.name == name) { return f; } }
  MFEM_ABORT("no flow named " << name);
  return flows[0];
}

cmf::SolverConfig TightSolverConfig(const std::string &type = "gmres_amg")
{
  cmf::SolverConfig sc;
  sc.newton.rtol = 1e-11;
  sc.newton.atol = 1e-15;
  sc.newton.max_it = 20;
  sc.newton.print_level = 0;
  sc.linear.type = type;
  sc.linear.rtol = 1e-14;
  sc.linear.max_it = 2000;
  return sc;
}

// (a) The patch test through the schema: a linear field with its source,
// Dirichlet on every face, the interior exact; the flows of the two faces
// normal to the gradient are kappa b . n |face| (the sign of Flows).
void ModulePatchTest()
{
  std::printf("module: patch test from a YAML input, flows\n");
  const std::string text = R"yaml(
physics: scalar_transport
mesh: { file: unused, order: 2 }
transport:
  conductivity: 0.8
  velocity: ["1", "-2"]
  reaction: 0.9
  source: "1.5 + 0.9*(0.3 + 1.5*x)"
bcs:
  dirichlet:
    - { attr: [4], name: left, expression: "0.3 + 1.5*x" }
    - { attr: [2], name: right, expression: "0.3 + 1.5*x" }
    - { attr: [1, 3], name: sides, expression: "0.3 + 1.5*x" }
solver:
  newton: { rtol: 1e-12, atol: 1e-15, print_level: 0 }
  linear: { type: gmres_amg, rtol: 1e-14 }
output:
  exact: "0.3 + 1.5*x"
  flows: true
)yaml";
  for (const std::string element : {"quad", "tri"})
  {
    cmf::ScalarAppConfig cfg = ParseText(text);
    UseBox(cfg, 2, element, 4, 2, 0.2);
    std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    cmf::ScalarTransport problem(*mesh, cfg);
    problem.Finalize();
    std::unique_ptr<mfem::Solver> linear = problem.MakeLinearSolver(cfg.solver.linear);
    mfem::Vector x(problem.Height());
    problem.InitialState(x);
    const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(problem, *linear, cfg.solver, x);
    CHECK_MSG(report.converged, element + ": converged");
    CHECK_MSG(report.steps.back().newton.iterations == 1, element + ": one Newton iteration");
    const cmf::ScalarErrors e = problem.Errors(x, 1.0);
    const std::vector<cmf::Flow> flows = problem.Flows(x);
    std::printf("  %-4s: l2 error %.1e, nodal %.1e; flow left %.12f right %.12f sides %.1e\n", element.c_str(),
                e.l2, e.linf_nodal, Named(flows, "left").value, Named(flows, "right").value,
                Named(flows, "sides").value);
    CHECK_MSG(e.linf_nodal <= 1e-12, element + ": interior exact " + std::to_string(e.linf_nodal));
    CHECK_CLOSE(Named(flows, "left").value, -0.8 * 1.5, 1e-11);   // n = -e_x
    CHECK_CLOSE(Named(flows, "right").value, 0.8 * 1.5, 1e-11);   // n = +e_x
    CHECK_CLOSE(Named(flows, "sides").value, 0.0, 1e-11);
  }
}

// (b) Implicit Euler is exact on a spatially constant state with a constant
// source (u = u0 + f t / c); the flow through a Dirichlet face balances a
// prescribed flux on the opposite face.
void TransientExactnessTest()
{
  std::printf("module: transient exactness, flux balance\n");
  {
    cmf::ScalarTransportModel model;
    model.capacity = {2.0, 0.0, 0.0};
    model.conductivity = {1.0, 0.0, 0.0};
    Box b = MakeBox(2, "quad", 3, 2, 0.1);
    cmf::ScalarTransport problem(*b.mesh, 2, model, true);
    mfem::ConstantCoefficient f(3.0), u0(0.5);
    problem.SetSource(f);
    problem.SetInitialCondition(u0);
    problem.Finalize();
    problem.SetPhysicalTime(true);
    cmf::SolverConfig sc = TightSolverConfig("cg_amg");
    std::unique_ptr<mfem::Solver> linear = problem.MakeLinearSolver(sc.linear);
    mfem::Vector x(problem.Height());
    problem.InitialState(x);
    const std::vector<double> times = {0.1, 0.25, 0.5};
    double worst = 0.0;
    const cmf::QuasiStaticReport report = cmf::SolveInTime(problem, *linear, sc, times, 0.0, x,
      [&](const cmf::LoadStepReport &step, const mfem::Vector &xs)
      {
        const double want = 0.5 + 3.0 * step.load_factor / 2.0;
        for (int i = 0; i < xs.Size(); i++) { worst = std::max(worst, std::abs(xs(i) - want)); }
      });
    worst = GlobalMax(worst);
    std::printf("  constant state: largest deviation from u0 + f t / c over three steps %.1e\n", worst);
    CHECK_MSG(report.converged, "constant state converged");
    CHECK_MSG(worst <= 1e-13, "implicit Euler exact on the constant state " + std::to_string(worst));
  }
  {
    cmf::ScalarTransportModel model;
    model.conductivity = {2.0, 0.0, 0.0};
    Box b = MakeBox(2, "tri", 4, 2, 0.1);
    cmf::ScalarTransport problem(*b.mesh, 2, model, false);
    mfem::ConstantCoefficient g(3.0), zero(0.0);
    cmf::BCOptions opt;
    opt.name = "right";
    problem.AddFlux({4}, g);            // inward flux on x = 0
    problem.AddDirichlet({2}, zero, opt); // u = 0 on x = 1
    problem.Finalize();
    cmf::SolverConfig sc = TightSolverConfig("cg_amg");
    std::unique_ptr<mfem::Solver> linear = problem.MakeLinearSolver(sc.linear);
    mfem::Vector x(problem.Height());
    problem.InitialState(x);
    const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(problem, *linear, sc, x);
    CHECK_MSG(report.converged, "flux bar converged");
    const std::vector<cmf::Flow> flows = problem.Flows(x);
    problem.UpdateFields(x);
    const double u_left = cmf::ProbeVector(problem.Unknown(), {0.0, 0.5})[0];
    std::printf("  flux bar: flow through the Dirichlet face %.12f (want -3), u(0) = %.12f (want 1.5)\n",
                Named(flows, "right").value, u_left);
    CHECK_CLOSE(Named(flows, "right").value, -3.0, 1e-11);
    CHECK_CLOSE(u_left, 1.5, 1e-11);
  }
}

// The steady convection-diffusion-reaction field of case 2 and its data.
struct SteadyMMS
{
  double kappa = 0.1, s = 1.0, cx = 1.0, cy = -2.0;
  int n = 3, m = 3;
  double U(const mfem::Vector &X) const { return std::sin(n * M_PI * X(0)) * std::sin(m * M_PI * X(1)); }
  void Grad(const mfem::Vector &X, mfem::Vector &g) const
  {
    g.SetSize(2);
    g(0) = n * M_PI * std::cos(n * M_PI * X(0)) * std::sin(m * M_PI * X(1));
    g(1) = m * M_PI * std::sin(n * M_PI * X(0)) * std::cos(m * M_PI * X(1));
  }
  double F(const mfem::Vector &X) const
  {
    mfem::Vector g;
    Grad(X, g);
    return kappa * (n * n + m * m) * M_PI * M_PI * U(X) + cx * g(0) + cy * g(1) + s * U(X);
  }
};

// (c) Manufactured solutions of the steady operator: L2 and H1 rates at
// k = 1, 2, 3, one Newton iteration, one assembly and one solver setup.
void SteadyRatesTest()
{
  std::printf("module: manufactured solutions of the steady operator\n");
  const SteadyMMS mms;
  mfem::FunctionCoefficient exact([&](const mfem::Vector &X) { return mms.U(X); });
  mfem::VectorFunctionCoefficient exact_grad(2, [&](const mfem::Vector &X, mfem::Vector &g) { mms.Grad(X, g); });
  mfem::FunctionCoefficient source([&](const mfem::Vector &X) { return mms.F(X); });
  mfem::Vector bvec(2);
  bvec(0) = mms.cx;
  bvec(1) = mms.cy;
  mfem::VectorConstantCoefficient beta(bvec);
  for (int order = 1; order <= 3; order++)
  {
    std::vector<double> l2, h1;
    for (const int n : {8, 16, 32})
    {
      cmf::ScalarTransportModel model;
      model.conductivity = {mms.kappa, 0.0, 0.0};
      model.reaction = mms.s;
      Box b = MakeBox(2, "quad", n, order, 0.0);
      cmf::ScalarTransport problem(*b.mesh, order, model, false);
      problem.SetVelocity(beta, false);
      problem.SetSource(source);
      problem.AddDirichlet({1, 2, 3, 4}, exact);
      problem.Finalize();
      cmf::SolverConfig sc = TightSolverConfig();
      std::unique_ptr<mfem::Solver> linear = problem.MakeLinearSolver(sc.linear);
      mfem::Vector x(problem.Height());
      problem.InitialState(x);
      const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(problem, *linear, sc, x);
      CHECK_MSG(report.converged, "MMS k=" + std::to_string(order) + " n=" + std::to_string(n) + " converged");
      CHECK_MSG(report.steps.back().newton.iterations == 1, "MMS: one Newton iteration");
      CHECK_MSG(problem.GradientAssemblies() == 1, "MMS: one Jacobian assembly, got " +
                std::to_string(problem.GradientAssemblies()));
      const auto *ls = dynamic_cast<const cmf::LinearSolver *>(linear.get());
      CHECK_MSG(ls && ls->Setups() == 1, "MMS: one solver setup");
      problem.UpdateFields(x);
      std::vector<const mfem::IntegrationRule *> irs(mfem::Geometry::NumGeom, nullptr);
      for (int g = 0; g < mfem::Geometry::NumGeom; g++) { irs[g] = &mfem::IntRules.Get(g, 2 * order + 3); }
      l2.push_back(problem.Unknown().ComputeL2Error(exact, irs.data()));
      h1.push_back(problem.Unknown().ComputeGradError(&exact_grad, irs.data()));
      std::printf("  k=%d n=%2d: L2 %.3e H1 %.3e\n", order, n, l2.back(), h1.back());
    }
    for (std::size_t i = 1; i < l2.size(); i++)
    {
      const double rl2 = std::log(l2[i - 1] / l2[i]) / std::log(2.0), rh1 = std::log(h1[i - 1] / h1[i]) / std::log(2.0);
      std::printf("  k=%d rates: L2 %.2f H1 %.2f\n", order, rl2, rh1);
      if (i + 1 == l2.size())
      {
        CHECK_MSG(rl2 >= order + 0.9, "k=" + std::to_string(order) + " L2 rate " + std::to_string(rl2));
        CHECK_MSG(rh1 >= order - 0.1, "k=" + std::to_string(order) + " H1 rate " + std::to_string(rh1));
      }
    }
  }
}

// The transient manufactured solution of case 5: u = sin t cos q.
struct TransientMMS
{
  double alpha = 0.1;
  double Q(const mfem::Vector &X) const { return 2.0 * std::pow(X(0) - 0.5, 2) + 2.0 * std::pow(X(1) - 0.5, 2); }
  double U(const mfem::Vector &X, double t) const { return std::sin(t) * std::cos(Q(X)); }
  double F(const mfem::Vector &X, double t) const
  {
    const double q = Q(X), r2 = std::pow(X(0) - 0.5, 2) + std::pow(X(1) - 0.5, 2);
    const double lap = std::sin(t) * (-16.0 * r2 * std::cos(q) - 8.0 * std::sin(q));
    return std::cos(t) * std::cos(q) - alpha * lap;
  }
};

// One transient run of case 5 on a fixed mesh; returns the L2 error at t_final.
struct TransientRun
{
  double error = 0.0;
  int assemblies = 0;
  int steps = 0;
};

TransientRun RunTransientMMS(int n, int order, const std::vector<double> &times, bool velocity_in_time = false)
{
  const TransientMMS mms;
  mfem::FunctionCoefficient exact([&](const mfem::Vector &X, double t) { return mms.U(X, t); });
  mfem::FunctionCoefficient source([&](const mfem::Vector &X, double t) { return mms.F(X, t); });
  cmf::ScalarTransportModel model;
  model.conductivity = {mms.alpha, 0.0, 0.0};
  Box b = MakeBox(2, "quad", n, order, 0.0);
  cmf::ScalarTransport problem(*b.mesh, order, model, true);
  // A velocity that mentions t but vanishes: the operator is reassembled every step and unchanged.
  mfem::VectorFunctionCoefficient beta(2, [](const mfem::Vector &, double, mfem::Vector &v) { v.SetSize(2); v = 0.0; });
  if (velocity_in_time) { problem.SetVelocity(beta, true); }
  problem.SetSource(source);
  problem.SetInitialCondition(exact);
  problem.SetExact(exact);
  cmf::BCOptions constant;   // the data is the expression at t (the YAML default under time)
  constant.schedule = cmf::Schedule::Constant();
  problem.AddDirichlet({1, 2, 3, 4}, exact, constant);
  problem.Finalize();
  problem.SetPhysicalTime(true);
  cmf::SolverConfig sc = TightSolverConfig(velocity_in_time ? "gmres_amg" : "cg_amg");
  std::unique_ptr<mfem::Solver> linear = problem.MakeLinearSolver(sc.linear);
  mfem::Vector x(problem.Height());
  problem.InitialState(x);
  const cmf::QuasiStaticReport report = cmf::SolveInTime(problem, *linear, sc, times, 0.0, x);
  CHECK_MSG(report.converged, "transient MMS converged");
  for (const cmf::LoadStepReport &s : report.steps)
  {
    CHECK_MSG(s.newton.iterations == 1, "transient MMS: one Newton iteration per step");
  }
  TransientRun r;
  r.error = problem.Errors(x, times.back()).l2;
  r.assemblies = problem.GradientAssemblies();
  r.steps = int(report.steps.size());
  return r;
}

// (d) First order in time on a fixed fine mesh, and the reuse of the Jacobian.
void TemporalOrderTest()
{
  std::printf("module: temporal order and the reuse of the Jacobian\n");
  std::vector<double> errors;
  const double t_final = 0.4;
  for (const int steps : {20, 40, 80, 160})
  {
    const TransientRun r = RunTransientMMS(16, 3, cmf::UniformTimeSteps(t_final, steps));
    errors.push_back(r.error);
    std::printf("  dt = %.4f: L2 error at t = %.1f %.3e, Jacobian assemblies %d\n", t_final / steps, t_final, r.error,
                r.assemblies);
    CHECK_MSG(r.assemblies == 1, "one assembly for the run at constant dt, got " + std::to_string(r.assemblies));
  }
  for (std::size_t i = 1; i < errors.size(); i++)
  {
    const double rate = std::log(errors[i - 1] / errors[i]) / std::log(2.0);
    std::printf("  rate in dt: %.3f\n", rate);
    CHECK_MSG(rate >= 0.95, "first order in dt: rate " + std::to_string(rate));
  }
  // Two segments of different dt: two assemblies.
  std::vector<double> times = cmf::UniformTimeSteps(0.2, 10);
  for (double t : cmf::UniformTimeSteps(0.2, 5)) { times.push_back(0.2 + t); }
  const TransientRun two = RunTransientMMS(4, 2, times);
  std::printf("  two dt segments: %d assemblies over %d steps\n", two.assemblies, two.steps);
  CHECK_MSG(two.assemblies == 2, "one assembly per dt segment, got " + std::to_string(two.assemblies));
  // A velocity expression of t: one assembly per step.
  const TransientRun each = RunTransientMMS(4, 2, cmf::UniformTimeSteps(0.2, 5), true);
  std::printf("  velocity of t: %d assemblies over %d steps\n", each.assemblies, each.steps);
  CHECK_MSG(each.assemblies == each.steps, "one assembly per step with a velocity of t");
}

// The nonlinear diffusion of case 4 (myapps nonlinear_convection_diffusion_1D,
// nonlinear_heat.m): the Kirchhoff-transformed series solution.
struct KirchhoffCase
{
  double a0 = 10.0, a1 = 0.09, m0 = 4.0e6, m1 = 3.6e4, u_ref = 300.0;
  double kappa1 = 10.0, kappa2 = 100.0, T0 = 300.0, T1 = 300.0, T2 = 1300.0, qbar = 7.5e5, L = 0.01;
  int terms = 1000;
  double alpha() const { return a0 / m0; }
  double Exact(double x, double t) const
  {
    const double decay = M_PI * M_PI * alpha() * t / (L * L);
    double S1 = 0.0;
    for (int n = 1; n <= terms; n++)
    {
      S1 += std::exp(-double(n) * n * decay) * std::cos(n * M_PI * x / L) / (double(n) * n);
    }
    const double f = alpha() * t / (L * L) + 1.0 / 3.0 - x / L + 0.5 * x * x / (L * L) - 2.0 / (M_PI * M_PI) * S1;
    const double theta0 = (T0 - T1) + (kappa2 - kappa1) / (T2 - T1) / (2.0 * kappa1) * (T0 - T1) * (T0 - T1);
    const double theta = f * qbar * L / kappa1 + theta0;
    const double gamma = 2.0 * (kappa2 - kappa1) / ((T2 - T1) * kappa1);
    const double sq = std::sqrt(std::max(1e-14, 1.0 + gamma * theta));
    return T1 + (T2 - T1) * (kappa1 / (kappa2 - kappa1)) * (-1.0 + sq);
  }
  // The inverse Kirchhoff transform: u from phi = a0 (u - u_ref) + a1 (u - u_ref)^2 / 2.
  double FromPhi(double phi) const { return u_ref + (-a0 + std::sqrt(a0 * a0 + 2.0 * a1 * phi)) / a1; }
};

struct KirchhoffRun
{
  double error = 0.0;   // L2 error against the series at t = 1
  int max_newton = 0;
  std::unique_ptr<mfem::ParMesh> mesh;
  std::unique_ptr<cmf::ScalarTransport> problem;
  mfem::Vector x;
};

KirchhoffRun RunKirchhoff(const KirchhoffCase &kc, int n, int order, int steps, bool linear_phi)
{
  cmf::ScalarTransportModel model;
  if (linear_phi)
  {
    // (m0 / a0) phi_t = lap phi with the flux datum of u: grad phi . n = a du/dn = g.
    model.capacity = {kc.m0 / kc.a0, 0.0, 0.0};
    model.conductivity = {1.0, 0.0, 0.0};
  }
  else
  {
    model.capacity = {kc.m0, kc.m1, kc.u_ref};
    model.conductivity = {kc.a0, kc.a1, kc.u_ref};
  }
  cmf::MeshConfig mc;
  mc.cartesian = true;
  mc.box.dim = 2;
  mc.box.element = "quad";
  mc.box.nx = mc.box.ny = n;
  mc.box.sx = mc.box.sy = kc.L;
  mc.order = order;
  KirchhoffRun r;
  r.mesh = cmf::BuildParMesh(MPI_COMM_WORLD, mc);
  r.problem = std::make_unique<cmf::ScalarTransport>(*r.mesh, order, model, true);
  static mfem::ConstantCoefficient flux(7.5e5), initial(300.0), zero(0.0);
  cmf::BCOptions constant;
  constant.schedule = cmf::Schedule::Constant();
  r.problem->AddFlux({4}, flux, constant);
  r.problem->SetInitialCondition(linear_phi ? zero : initial);
  r.problem->Finalize();
  r.problem->SetPhysicalTime(true);
  cmf::SolverConfig sc = TightSolverConfig("cg_amg");
  sc.newton.rtol = 1e-10;
  sc.newton.atol = 1e-9 * kc.m0;   // the round-off floor of rows of size m0 |Omega| / N
  std::unique_ptr<mfem::Solver> linear = r.problem->MakeLinearSolver(sc.linear);
  r.x.SetSize(r.problem->Height());
  r.problem->InitialState(r.x);
  const cmf::QuasiStaticReport report = cmf::SolveInTime(*r.problem, *linear, sc, cmf::UniformTimeSteps(1.0, steps),
                                                         0.0, r.x);
  CHECK_MSG(report.converged, "Kirchhoff run converged");
  for (const cmf::LoadStepReport &s : report.steps) { r.max_newton = std::max(r.max_newton, s.newton.iterations); }
  mfem::FunctionCoefficient exact([&](const mfem::Vector &X) { return kc.Exact(X(0), 1.0); });
  r.problem->UpdateFields(r.x);
  if (!linear_phi)
  {
    std::vector<const mfem::IntegrationRule *> irs(mfem::Geometry::NumGeom, nullptr);
    for (int g = 0; g < mfem::Geometry::NumGeom; g++) { irs[g] = &mfem::IntRules.Get(g, 2 * order + 3); }
    r.error = r.problem->Unknown().ComputeL2Error(exact, irs.data());
  }
  return r;
}

// (e) The nonlinear laws of case 4 against the series: first order in dt,
// few Newton iterations, and the inverse transform of the linear solve.
void KirchhoffTest()
{
  std::printf("module: nonlinear diffusion of the Kirchhoff case\n");
  const KirchhoffCase kc;
  std::vector<double> errors;
  for (const int steps : {10, 20, 40})
  {
    const KirchhoffRun r = RunKirchhoff(kc, 4, 3, steps, false);
    errors.push_back(r.error);
    std::printf("  dt = %.4f: L2 error at t = 1 %.3e (relative %.2e), Newton iterations per step <= %d\n",
                1.0 / steps, r.error, r.error / (300.0 * kc.L), r.max_newton);
    CHECK_MSG(r.max_newton <= 4, "at most four Newton iterations per step, got " + std::to_string(r.max_newton));
  }
  for (std::size_t i = 1; i < errors.size(); i++)
  {
    const double rate = std::log(errors[i - 1] / errors[i]) / std::log(2.0);
    std::printf("  rate in dt: %.3f\n", rate);
    CHECK_MSG(rate >= 0.9, "first order in dt on the Kirchhoff case: rate " + std::to_string(rate));
  }
  // The transformed linear solve at dt = 0.05 against the nonlinear one.
  const KirchhoffRun nonlinear = RunKirchhoff(kc, 4, 3, 20, false);
  KirchhoffRun phi = RunKirchhoff(kc, 4, 3, 20, true);
  mfem::ParGridFunction &phi_gf = phi.problem->Unknown();
  mfem::ParGridFunction transformed(phi_gf);
  for (int i = 0; i < transformed.Size(); i++) { transformed(i) = kc.FromPhi(phi_gf(i)); }
  mfem::GridFunctionCoefficient transformed_coef(&transformed);
  const double diff = nonlinear.problem->Unknown().ComputeL2Error(transformed_coef);
  mfem::FunctionCoefficient exact([&](const mfem::Vector &X) { return kc.Exact(X(0), 1.0); });
  const double err_phi = transformed.ComputeL2Error(exact);
  std::printf("  nonlinear vs transformed linear solve: L2 difference %.3e; errors against the series %.3e (nonlinear) "
              "%.3e (transformed)\n", diff, nonlinear.error, err_phi);
  CHECK_MSG(diff <= std::max(nonlinear.error, err_phi), "the two discretisations agree to their error");
}

// (f) The initial condition, the point pin, a scheduled flux, the errors and
// the registered fields.
void FeaturesTest()
{
  std::printf("module: initial condition, pin, scheduled flux, errors and fields\n");
  {
    // Pure Neumann steady problem made well-posed by a pin: the constant.
    const std::string text = R"yaml(
physics: scalar_transport
mesh: { file: unused, order: 2 }
transport: { conductivity: 1.0 }
bcs:
  dirichlet: [ { point: [0.0, 0.0], name: pin, expression: "2" } ]
solver:
  newton: { rtol: 1e-12, atol: 1e-15, print_level: 0 }
  linear: { type: cg_amg, rtol: 1e-14 }
)yaml";
    cmf::ScalarAppConfig cfg = ParseText(text);
    UseBox(cfg, 2, "quad", 3, 2, 0.1);
    std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    cmf::ScalarTransport problem(*mesh, cfg);
    problem.Finalize();
    std::unique_ptr<mfem::Solver> linear = problem.MakeLinearSolver(cfg.solver.linear);
    mfem::Vector x(problem.Height());
    problem.InitialState(x);
    const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(problem, *linear, cfg.solver, x);
    CHECK_MSG(report.converged, "pinned problem converged");
    double worst = 0.0;
    for (int i = 0; i < x.Size(); i++) { worst = std::max(worst, std::abs(x(i) - 2.0)); }
    worst = GlobalMax(worst);
    std::printf("  pin: largest deviation from the pinned constant %.1e\n", worst);
    CHECK_MSG(worst <= 1e-11, "the pin fixes the constant " + std::to_string(worst));
    const std::vector<cmf::Flow> flows = problem.Flows(x);
    CHECK_CLOSE(Named(flows, "pin").value, 0.0, 1e-11);
  }
  {
    // A ramped flux over two load steps: the flow follows the schedule.
    const std::string text = R"yaml(
physics: scalar_transport
mesh: { file: unused, order: 1 }
transport: { conductivity: 2.0 }
bcs:
  dirichlet: [ { attr: [2], name: right, expression: "0" } ]
  flux: [ { attr: [4], expression: "3", schedule: { type: ramp, from: 0.0, to: 1.0 } } ]
solver:
  load_steps: 2
  newton: { rtol: 1e-12, atol: 1e-15, print_level: 0 }
  linear: { type: cg_amg, rtol: 1e-14 }
)yaml";
    cmf::ScalarAppConfig cfg = ParseText(text);
    UseBox(cfg, 2, "quad", 3, 1);
    std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    cmf::ScalarTransport problem(*mesh, cfg);
    problem.Finalize();
    std::unique_ptr<mfem::Solver> linear = problem.MakeLinearSolver(cfg.solver.linear);
    mfem::Vector x(problem.Height());
    problem.InitialState(x);
    std::vector<double> flows;
    cmf::SolveQuasiStatic(problem, *linear, cfg.solver, x, [&](const cmf::LoadStepReport &, const mfem::Vector &xs)
    {
      flows.push_back(Named(problem.Flows(xs), "right").value);
    });
    std::printf("  scheduled flux: flows after the two steps %.12f %.12f (want -1.5, -3)\n", flows.at(0), flows.at(1));
    CHECK_CLOSE(flows.at(0), -1.5, 1e-11);
    CHECK_CLOSE(flows.at(1), -3.0, 1e-11);
  }
  {
    // The initial condition and the exact-solution machinery.
    const std::string text = R"yaml(
physics: scalar_transport
mesh: { file: unused, order: 2 }
transport: { unknown: c, conductivity: 1.0 }
initial: "sin(pi*x)*sin(pi*y)"
time: { t_final: 1.0, dt: 0.5 }
output:
  fields: [c, c_exact, c_error, flux]
  exact: "sin(pi*x)*sin(pi*y)"
)yaml";
    cmf::ScalarAppConfig cfg = ParseText(text);
    UseBox(cfg, 2, "quad", 4, 2);
    std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    cmf::ScalarTransport problem(*mesh, cfg);
    problem.Finalize();
    mfem::Vector x(problem.Height());
    problem.InitialState(x);
    const cmf::ScalarErrors e = problem.Errors(x, 0.0);
    std::printf("  interpolant of the exact field: l2 %.2e (interpolation error), nodal %.1e\n", e.l2, e.linf_nodal);
    CHECK_MSG(e.l2 > 1e-6 && e.l2 < 1e-2, "the L2 error of the interpolant is the interpolation error");
    CHECK_MSG(e.linf_nodal <= 1e-14, "the nodal error of the interpolant vanishes");
    CHECK_MSG(e.rel_l2 > 0.0 && e.rel_l2 < 1e-2, "relative error");
    cmf::FieldRegistry fields;
    problem.RegisterFields(fields);
    problem.UpdateFields(x);
    for (const char *name : {"c", "c_exact", "c_error", "flux"})
    {
      CHECK_MSG(fields.Has(name), std::string("field ") + name + " registered");
    }
    const std::vector<double> fl = cmf::ProbeVector(fields.Get("flux"), {0.5, 0.5});
    CHECK_MSG(fl.size() == 2 && std::abs(fl[0]) < 1e-2 && std::abs(fl[1]) < 1e-2,
              "the flux at the centre of sin sin vanishes");
    CHECK_MSG(problem.UnknownName() == "c", "unknown name");
  }
}

// (g) The schema errors.
void SchemaTest()
{
  std::printf("module: schema errors\n");
  const std::string ok = R"yaml(
physics: scalar_transport
mesh: { file: unused, order: 2 }
transport: { conductivity: 1.0, velocity: ["1", "0"] }
time: { t_final: 1.0, dt: 0.5 }
bcs:
  dirichlet: [ { attr: [1], expression: "0" } ]
)yaml";
  auto with = [&](const std::string &find, const std::string &replace)
  {
    std::string text = ok;
    const std::size_t at = text.find(find);
    MFEM_VERIFY(at != std::string::npos, "needle not found");
    return text.replace(at, find.size(), replace);
  };
  ParseText(ok);
  CHECK_THROWS(ParseText(with("physics: scalar_transport\n", "")), cmf::ConfigError, "missing key 'physics'");
  CHECK_THROWS(ParseText(with("scalar_transport", "solid")), cmf::ConfigError, "unknown value 'solid'");
  CHECK_THROWS(ParseText(ok + "formulation: mixed\n"), cmf::ConfigError, "'formulation' belongs to the solid");
  CHECK_THROWS(ParseText(with("conductivity: 1.0", "conductivity: -1.0")), cmf::ConfigError, "must be positive");
  CHECK_THROWS(ParseText(with("time: { t_final: 1.0, dt: 0.5 }\n", "initial: \"1\"\n")), cmf::ConfigError,
               "'initial' needs a time block");
  CHECK_THROWS(ParseText(ok + "solver: { linear: { type: cg_amg } }\n"), cmf::ConfigError, "cg_amg needs a symmetric");
  CHECK_THROWS(ParseText(ok + "solver: { linear: { amg: systems } }\n"), cmf::ConfigError, "takes scalar");
  CHECK_THROWS(ParseText(ok + "output: { fields: [u_exact] }\n"), cmf::ConfigError, "needs output.exact");
  CHECK_THROWS(ParseText(ok + "output: { fields: [pressure] }\n"), cmf::ConfigError, "unknown field 'pressure'");
  CHECK_THROWS(ParseText(ok + "output: { reactions: true }\n"), cmf::ConfigError, "belongs to the solid schema");
  CHECK_THROWS(ParseText(with("expression: \"0\"", "expression: \"2*u\"")), cmf::ConfigError, "unknown identifier 'u'");
  CHECK_THROWS(ParseText(with("{ attr: [1], expression: \"0\" }", "{ attr: [1], point: [0, 0], expression: \"0\" }")),
               cmf::ConfigError, "give one, not both");
  CHECK_THROWS(ParseText(with("dirichlet:", "traction:")), cmf::ConfigError, "'bcs.traction' belongs to the solid");
  CHECK_THROWS(ParseText(with("velocity: [\"1\", \"0\"]", "velocity: \"1\"")), cmf::ConfigError, "list of strings");
  CHECK_THROWS(ParseText(ok + "unknown_key: 1\n"), cmf::ConfigError, "unknown key 'unknown_key'");
  // Errors that need the mesh: the velocity components, a point off the nodes.
  {
    cmf::ScalarAppConfig cfg = ParseText(with("velocity: [\"1\", \"0\"]", "velocity: [\"1\", \"0\", \"0\"]"));
    UseBox(cfg, 2, "quad", 2, 1);
    std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    CHECK_THROWS(cmf::ScalarTransport(*mesh, cfg), cmf::ConfigError, "3 components, mesh dimension is 2");
  }
  {
    cmf::ScalarAppConfig cfg = ParseText(with("{ attr: [1], expression: \"0\" }", "{ point: [0.3, 0.3], expression: \"0\" }"));
    UseBox(cfg, 2, "quad", 2, 1);
    std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    cmf::ScalarTransport problem(*mesh, cfg);
    CHECK_THROWS(problem.Finalize(), cmf::ConfigError, "is not a node of the mesh");
  }
  // The solid executable rejects a physics key.
  CHECK_THROWS(cmf::ParseConfig(YAML::Load("physics: scalar_transport\nmesh: { file: x }\n")), cmf::ConfigError,
               "belongs to the executable of that physics");
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
  ModulePatchTest();
  TransientExactnessTest();
  SteadyRatesTest();
  TemporalOrderTest();
  KirchhoffTest();
  FeaturesTest();
  SchemaTest();
  return cmf_test::Report("test_scalar_transport");
}

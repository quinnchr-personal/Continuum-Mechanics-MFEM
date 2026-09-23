// Scalar CG kernel with the flux/source contract of the framework: the second
// physics of the seam, on a scalar H1 unknown u. Strong form on the fixed
// domain, per unit volume,
//   c(u) du/dt - div F(u, grad u) + S(u, grad u) = 0,
//   F = kappa(u) grad u          S = beta . grad u + s u - f   (non-conservative convection)
//   F = kappa(u) grad u - beta u S = s u - f                   (conservative convection),
// so that F is the negative of the physical flux (the convention of the
// thermo kernel's flux = -Q) and F . n = 0 is the natural boundary condition.
// Weak form, with the rate term by implicit Euler over the step of length dt
// from the accepted state u_n (dt = 0: the steady problem),
//   R(u; q) = int q c(u) (u - u_n) / dt dV + int grad q . F dV + int q S dV,
// the boundary terms (prescribed inward fluxes) being linear forms of the
// physics module (physics/scalar_conditions.hpp). The point contract is one
// templated function, ScalarDensities<T>, returning the scalar density r =
// c (u - u_n)/dt + S and the vector F; the tangent comes from 1 + dim dual
// seeds (u and the components of grad u), each a rank-one update of the
// element matrix, the loop of the thermo kernel. The velocity and the source
// are coefficients evaluated per quadrature point and passed in as values;
// the accepted state is read from a grid function on the same space at the
// element's dofs. The quadrature rule is of order `quadrature_order` when
// given, 2p + 3 (the framework's rule) otherwise; the same rule integrates
// the source.
#pragma once

#include "base/dual.hpp"
#include "base/tensor.hpp"
#include "mfem.hpp"

namespace cmf
{

template <typename T>
struct ScalarPointDensities
{
  T r;               // c (u - u_n) / dt + S(u, grad u): tested by q
  tensor<T, 3> flux; // F(u, grad u): tested by grad q
};

// The densities at a point; beta and source are values there, u_old the
// accepted state, dt the step (0 drops the rate term). Components beyond the
// space dimension are zero.
template <typename Model, typename T>
inline ScalarPointDensities<T> ScalarDensities(const Model &m, const T &u, const tensor<T, 3> &grad_u,
                                               double u_old, double dt, const tensor<double, 3> &beta,
                                               double source)
{
  ScalarPointDensities<T> d;
  const T kappa = m.Conductivity(u);
  T S = m.reaction * u - source;
  for (int j = 0; j < 3; j++)
  {
    d.flux(j) = kappa * grad_u(j);
    if (m.Conservative()) { d.flux(j) = d.flux(j) - beta(j) * u; }
    else { S = S + beta(j) * grad_u(j); }
  }
  d.r = S;
  if (dt > 0.0) { d.r = d.r + m.Capacity(u) * (u - u_old) / dt; }
  return d;
}

template <typename Model>
class ScalarFluxIntegrator : public mfem::NonlinearFormIntegrator
{
public:
  explicit ScalarFluxIntegrator(const Model &model, int quadrature_order = 0)
    : model_(model), quadrature_order_(quadrature_order) {}

  // beta(x, t), one component per space dimension; may be null (none).
  void SetVelocity(mfem::VectorCoefficient *beta) { beta_ = beta; }
  // f(x, t); may be null (none).
  void SetSource(mfem::Coefficient *f) { source_ = f; }
  // The accepted state u_n on the space of the unknown, and the step length
  // dt of the rate term; without a state, or at dt = 0, the problem is steady.
  void SetOldState(const mfem::GridFunction *u_old) { u_old_ = u_old; }
  void SetDt(double dt) { dt_ = dt; }
  double Dt() const { return dt_; }
  const Model &GetModel() const { return model_; }
  int QuadratureOrder(const mfem::FiniteElement &el) const
  {
    return quadrature_order_ > 0 ? quadrature_order_ : 2 * el.GetOrder() + 3;
  }
  const mfem::IntegrationRule &Rule(const mfem::FiniteElement &el) const
  {
    return mfem::IntRules.Get(el.GetGeomType(), QuadratureOrder(el));
  }

  void AssembleElementVector(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
                             const mfem::Vector &elfun, mfem::Vector &elvect) override
  {
    Prepare(el, Tr);
    elvect.SetSize(dof_);
    elvect = 0.0;
    const mfem::IntegrationRule &ir = Rule(el);
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const Point pt = PointAt(el, Tr, ir.IntPoint(q), elfun);
      const ScalarPointDensities<double> d =
        ScalarDensities(model_, pt.u, pt.grad_u, pt.u_old, dt_, pt.beta, pt.source);
      for (int a = 0; a < dof_; a++)
      {
        double s = d.r * shape_(a);
        for (int j = 0; j < dim_; j++) { s += d.flux(j) * DS_(a, j); }
        elvect(a) += pt.w * s;
      }
    }
  }

  void AssembleElementGrad(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
                           const mfem::Vector &elfun, mfem::DenseMatrix &elmat) override
  {
    Prepare(el, Tr);
    elmat.SetSize(dof_);
    elmat = 0.0;
    row_.SetSize(dof_);
    const mfem::IntegrationRule &ir = Rule(el);
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const Point pt = PointAt(el, Tr, ir.IntPoint(q), elfun);
      dual ud(pt.u, 0.0);
      tensor<dual, 3> gd;
      for (int j = 0; j < 3; j++) { gd(j) = dual(pt.grad_u(j), 0.0); }
      // Seeds: u, then the dim components of grad u.
      for (int s = 0; s <= dim_; s++)
      {
        if (s == 0) { ud.d = 1.0; } else { gd(s - 1).d = 1.0; }
        const ScalarPointDensities<dual> d =
          ScalarDensities(model_, ud, gd, pt.u_old, dt_, pt.beta, pt.source);
        if (s == 0) { ud.d = 0.0; } else { gd(s - 1).d = 0.0; }
        for (int a = 0; a < dof_; a++)
        {
          double v = d.r.d * shape_(a);
          for (int j = 0; j < dim_; j++) { v += d.flux(j).d * DS_(a, j); }
          row_(a) = pt.w * v;
        }
        for (int b = 0; b < dof_; b++)
        {
          const double cv = s == 0 ? shape_(b) : DS_(b, s - 1);
          if (cv == 0.0) { continue; }
          for (int a = 0; a < dof_; a++) { elmat(a, b) += row_(a) * cv; }
        }
      }
    }
  }

private:
  struct Point
  {
    double u = 0.0, u_old = 0.0, w = 0.0, source = 0.0;
    tensor<double, 3> grad_u;
    tensor<double, 3> beta;
  };

  void Prepare(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr)
  {
    dof_ = el.GetDof();
    dim_ = el.GetDim();
    shape_.SetSize(dof_);
    dshape_.SetSize(dof_, dim_);
    DS_.SetSize(dof_, dim_);
    Jrt_.SetSize(dim_);
    rate_ = u_old_ != nullptr && dt_ > 0.0;
    if (rate_)
    {
      u_old_->FESpace()->GetElementVDofs(Tr.ElementNo, vdofs_);
      u_old_->GetSubVector(vdofs_, old_);
    }
  }

  Point PointAt(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
                const mfem::IntegrationPoint &ip, const mfem::Vector &elfun)
  {
    Point pt;
    Tr.SetIntPoint(&ip);
    el.CalcShape(ip, shape_);
    el.CalcDShape(ip, dshape_);
    mfem::CalcInverse(Tr.Jacobian(), Jrt_);
    mfem::Mult(dshape_, Jrt_, DS_);
    pt.u = shape_ * elfun;
    for (int a = 0; a < dof_; a++)
      for (int j = 0; j < dim_; j++) { pt.grad_u(j) += DS_(a, j) * elfun(a); }
    pt.w = ip.weight * Tr.Weight();
    if (rate_) { pt.u_old = shape_ * old_; }
    if (beta_)
    {
      beta_->Eval(bvec_, Tr, ip);
      for (int j = 0; j < dim_; j++) { pt.beta(j) = bvec_(j); }
    }
    if (source_) { pt.source = source_->Eval(Tr, ip); }
    return pt;
  }

  Model model_;
  int quadrature_order_ = 0;
  mfem::VectorCoefficient *beta_ = nullptr;
  mfem::Coefficient *source_ = nullptr;
  const mfem::GridFunction *u_old_ = nullptr;
  double dt_ = 0.0;
  bool rate_ = false;
  int dof_ = 0, dim_ = 0;
  mfem::Vector shape_, old_, row_, bvec_;
  mfem::DenseMatrix dshape_, DS_, Jrt_;
  mfem::Array<int> vdofs_;
};

} // namespace cmf

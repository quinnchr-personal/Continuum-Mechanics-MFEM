// Coupled displacement-pressure-temperature total Lagrangian kernel for the
// thermoelastic materials (materials/thermoelastic.hpp) on Taylor-Hood style
// spaces (u of order k; p and theta of order k - 1): the mixed kernel of
// mixed_total_lagrangian.hpp with a third block, the heat equation of Anand's
// finite thermoelasticity in temperature form, implicit Euler over the step
// of the time block.
//
//   R_u(u, p, theta).w  = int [s(theta) P_iso(F) + p J F^-T] : Grad w dV
//   R_p(u, p, theta).q  = int q [u'(J / J_theta) / J_theta - p / kappa] dV
//   R_t(u, theta).q     = int q [c_v (theta - theta_n) - 1/2 theta M : (C - C_n)] dV
//                         + dt int (k J C^-1 Grad theta) . Grad q dV - dt [heat flux entries]
// with M = F^-1 dP/dtheta the thermal tangent of the displacement-form stress
// (its constitutive pressure), Q = -k J C^-1 Grad theta the referential heat
// flux, and C_n, theta_n the accepted state, held at the quadrature points
// in a HistoryField of seven doubles (C in the order xx yy zz xy yz xz, then
// theta). The nine tangent blocks come from dual seeds of one templated point
// evaluation over the entries of F (the in-plane ones, plus the hoop stretch
// of an axisymmetric point), p, theta and Grad theta; each seed is a rank-one
// update of the element matrices with the B-rows of the dofs.
#pragma once

#include <cmath>
#include <memory>
#include <type_traits>
#include <vector>

#include "base/dual.hpp"
#include "base/tensor.hpp"
#include "kernels/history_field.hpp"
#include "kernels/total_lagrangian.hpp"
#include "mfem.hpp"

namespace cmf
{

// The densities of the three residuals at a point, templated on the scalar
// type so that one dual seed gives their derivatives.
template <typename T>
struct ThermoPointDensities
{
  tensor<T, 3, 3> P;    // first Piola-Kirchhoff stress with the field pressure
  T c;                  // the constraint u'(J_m)/J_theta - p/kappa
  T h;                  // c_v (theta - theta_n) - 1/2 theta M : (C - C_n)
  tensor<T, 3> flux;    // -Q = k J C^-1 Grad theta
};

template <typename Material, typename T>
inline ThermoPointDensities<T> ThermoDensities(const Material &m, const tensor<T, 3, 3> &F, const T &p,
                                               const T &theta, const tensor<T, 3> &grad_theta,
                                               const tensor<double, 3, 3> &C_old, double theta_old,
                                               double inv_kappa)
{
  ThermoPointDensities<T> d;
  const T J = det(F);
  d.P = m.PK1Iso(F, theta) + (p * J) * transpose(inv(F));
  d.c = m.NormalizedVolumetricPressure(J, theta) - inv_kappa * p;
  const tensor<T, 3, 3> C = transpose(F) * F;
  const tensor<T, 3, 3> M = m.ThermalTangent(F, theta);
  d.h = m.thermal.c_v * (theta - theta_old) - 0.5 * theta * ddot(M, C - C_old);
  d.flux = (-1.0) * m.HeatFlux(F, grad_theta);
  return d;
}

// P with the field pressure at a point (outputs).
template <typename Material, typename T>
inline tensor<T, 3, 3> ThermoMixedPK1(const Material &m, const tensor<T, 3, 3> &F, double p, double theta)
{
  const T J = det(F);
  return m.PK1Iso(F, theta) + (p * J) * transpose(inv(F));
}

template <typename Material>
class ThermoMixedTotalLagrangianIntegrator : public mfem::BlockNonlinearFormIntegrator
{
public:
  static constexpr int kHistory = 7; // C_n (6), theta_n

  explicit ThermoMixedTotalLagrangianIntegrator(const std::vector<Material> &by_attribute)
    : materials_(by_attribute) {}

  void SetHistory(const HistoryField *history) { history_ = history; }
  void SetAxisymmetric(bool on) { axisymmetric_ = on; }

  // The mechanical free energy s(theta) Psi_iso(F) + kappa u(J / J_theta) of
  // the displacement form (the field pressure is not used).
  mfem::real_t GetElementEnergy(const mfem::Array<const mfem::FiniteElement *> &el,
                                mfem::ElementTransformation &Tr,
                                const mfem::Array<const mfem::Vector *> &elfun) override
  {
    return Tr.GetDimension() == 2 ? Energy<2>(el, Tr, elfun) : Energy<3>(el, Tr, elfun);
  }

  void AssembleElementVector(const mfem::Array<const mfem::FiniteElement *> &el,
                             mfem::ElementTransformation &Tr,
                             const mfem::Array<const mfem::Vector *> &elfun,
                             const mfem::Array<mfem::Vector *> &elvec) override
  {
    if (Tr.GetDimension() == 2) { Residual<2>(el, Tr, elfun, elvec); }
    else { Residual<3>(el, Tr, elfun, elvec); }
  }

  void AssembleElementGrad(const mfem::Array<const mfem::FiniteElement *> &el,
                           mfem::ElementTransformation &Tr,
                           const mfem::Array<const mfem::Vector *> &elfun,
                           const mfem::Array2D<mfem::DenseMatrix *> &elmats) override
  {
    if (Tr.GetDimension() == 2) { Tangent<2>(el, Tr, elfun, elmats); }
    else { Tangent<3>(el, Tr, elfun, elmats); }
  }

  const Material &MaterialOf(const mfem::ElementTransformation &Tr) const
  {
    return materials_.size() == 1 ? materials_[0] : materials_[std::size_t(Tr.Attribute)];
  }

  // Unpacks the accepted C and theta of a point.
  static void UnpackHistory(const double *h, tensor<double, 3, 3> &C, double &theta)
  {
    C(0, 0) = h[0]; C(1, 1) = h[1]; C(2, 2) = h[2];
    C(0, 1) = C(1, 0) = h[3];
    C(1, 2) = C(2, 1) = h[4];
    C(0, 2) = C(2, 0) = h[5];
    theta = h[6];
  }
  static void PackHistory(const tensor<double, 3, 3> &C, double theta, double *h)
  {
    h[0] = C(0, 0); h[1] = C(1, 1); h[2] = C(2, 2);
    h[3] = 0.5 * (C(0, 1) + C(1, 0));
    h[4] = 0.5 * (C(1, 2) + C(2, 1));
    h[5] = 0.5 * (C(0, 2) + C(2, 0));
    h[6] = theta;
  }

private:
  static double InvKappa(const Material &m) { return m.Incompressible() ? 0.0 : 1.0 / m.kappa; }

  const mfem::IntegrationRule &Rule(const mfem::FiniteElement &el) const
  {
    return mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3);
  }

  // The state of a point: F (completed), p, theta, Grad theta (3-vector),
  // the B data of the dofs, the weight and the history.
  template <int dim>
  struct Point
  {
    tensor<double, 3, 3> F;
    double p = 0.0, theta = 0.0;
    tensor<double, 3> grad_theta;
    double w = 0.0;
    AxisymmetricPoint axi;
    bool hoop = false;
    tensor<double, 3, 3> C_old;
    double theta_old = 0.0;
  };

  template <int dim>
  void Prepare(const mfem::Array<const mfem::FiniteElement *> &el,
               const mfem::Array<const mfem::Vector *> &elfun)
  {
    MFEM_VERIFY(el.Size() == 3, "thermo kernel needs (displacement, pressure, temperature) spaces");
    MFEM_VERIFY(history_, "ThermoMixedTotalLagrangianIntegrator: SetHistory first");
    const int dof_u = el[0]->GetDof(), dof_p = el[1]->GetDof(), dof_t = el[2]->GetDof();
    DSh_.SetSize(dof_u, dim);
    DS_.SetSize(dof_u, dim);
    DSth_.SetSize(dof_t, dim);
    DSt_.SetSize(dof_t, dim);
    Jrt_.SetSize(dim);
    Hmat_.SetSize(dim);
    Sh_.SetSize(dof_p);
    St_.SetSize(dof_t);
    shape_.SetSize(dof_u);
    PMatI_.UseExternalData(elfun[0]->GetData(), dof_u, dim);
  }

  template <int dim>
  Point<dim> PointSetup(const mfem::Array<const mfem::FiniteElement *> &el,
                        mfem::ElementTransformation &Tr, const mfem::IntegrationPoint &ip, int q,
                        const mfem::Array<const mfem::Vector *> &elfun)
  {
    Point<dim> pt;
    Tr.SetIntPoint(&ip);
    mfem::CalcInverse(Tr.Jacobian(), Jrt_);
    el[0]->CalcDShape(ip, DSh_);
    mfem::Mult(DSh_, Jrt_, DS_);
    mfem::MultAtB(PMatI_, DS_, Hmat_);
    tensor<double, dim, dim> H;
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++) { H(i, j) = Hmat_(i, j); }
    pt.F = DeformationGradient<dim>(H);
    pt.w = ip.weight * Tr.Weight();
    if constexpr (dim == 2)
    {
      if (axisymmetric_)
      {
        el[0]->CalcShape(ip, shape_);
        Tr.Transform(ip, X_);
        double u_r = 0.0;
        for (int a = 0; a < el[0]->GetDof(); a++) { u_r += shape_(a) * PMatI_(a, 0); }
        pt.axi = AxisymmetricAt(X_(0), u_r, H);
        pt.F(2, 2) = pt.axi.F33;
        pt.w *= pt.axi.weight;
        pt.hoop = true;
      }
    }
    el[1]->CalcShape(ip, Sh_);
    pt.p = Sh_ * (*elfun[1]);
    el[2]->CalcShape(ip, St_);
    pt.theta = St_ * (*elfun[2]);
    el[2]->CalcDShape(ip, DSth_);
    mfem::Mult(DSth_, Jrt_, DSt_);
    for (int j = 0; j < 3; j++) { pt.grad_theta(j) = 0.0; }
    for (int c = 0; c < el[2]->GetDof(); c++)
      for (int j = 0; j < dim; j++) { pt.grad_theta(j) += DSt_(c, j) * (*elfun[2])(c); }
    UnpackHistory(history_->Old(Tr.ElementNo, q), pt.C_old, pt.theta_old);
    return pt;
  }

  // The entries of F a displacement dof reaches: (i, j) in-plane, and (2, 2)
  // for the hoop stretch.
  template <int dim>
  int Entries(int (*idx)[2], bool hoop) const
  {
    int n = 0;
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++) { idx[n][0] = i; idx[n][1] = j; n++; }
    if (hoop) { idx[n][0] = 2; idx[n][1] = 2; n++; }
    return n;
  }

  // B(a i, m): dF_m of the displacement dof (a, i).
  template <int dim>
  double Brow(int a, int i, int m, const int (*idx)[2], const Point<dim> &pt) const
  {
    if (pt.hoop && idx[m][0] == 2) { return i == 0 ? pt.axi.Hoop(a, shape_, DS_) : 0.0; }
    return i == idx[m][0] ? DS_(a, idx[m][1]) : 0.0;
  }

  template <int dim>
  double Energy(const mfem::Array<const mfem::FiniteElement *> &el, mfem::ElementTransformation &Tr,
                const mfem::Array<const mfem::Vector *> &elfun)
  {
    Prepare<dim>(el, elfun);
    const Material &m = MaterialOf(Tr);
    const mfem::IntegrationRule &ir = Rule(*el[0]);
    double energy = 0.0;
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const Point<dim> pt = PointSetup<dim>(el, Tr, ir.IntPoint(q), q, elfun);
      energy += pt.w * m.Energy(pt.F, pt.theta);
    }
    return energy;
  }

  template <int dim>
  void Residual(const mfem::Array<const mfem::FiniteElement *> &el, mfem::ElementTransformation &Tr,
                const mfem::Array<const mfem::Vector *> &elfun, const mfem::Array<mfem::Vector *> &elvec)
  {
    Prepare<dim>(el, elfun);
    const int dof_u = el[0]->GetDof(), dof_p = el[1]->GetDof(), dof_t = el[2]->GetDof();
    elvec[0]->SetSize(dof_u * dim);
    elvec[1]->SetSize(dof_p);
    elvec[2]->SetSize(dof_t);
    *elvec[0] = 0.0;
    *elvec[1] = 0.0;
    *elvec[2] = 0.0;
    mfem::DenseMatrix PMatO(elvec[0]->GetData(), dof_u, dim);
    const Material &m = MaterialOf(Tr);
    const double inv_kappa = InvKappa(m);
    const double dt = history_->Dt();
    const mfem::IntegrationRule &ir = Rule(*el[0]);
    int idx[10][2];
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const Point<dim> pt = PointSetup<dim>(el, Tr, ir.IntPoint(q), q, elfun);
      const int n = Entries<dim>(idx, pt.hoop);
      const ThermoPointDensities<double> d =
        ThermoDensities(m, pt.F, pt.p, pt.theta, pt.grad_theta, pt.C_old, pt.theta_old, inv_kappa);
      for (int a = 0; a < dof_u; a++)
        for (int i = 0; i < dim; i++)
        {
          double s = 0.0;
          for (int mm = 0; mm < n; mm++) { s += d.P(idx[mm][0], idx[mm][1]) * Brow<dim>(a, i, mm, idx, pt); }
          PMatO(a, i) += pt.w * s;
        }
      for (int b = 0; b < dof_p; b++) { (*elvec[1])(b) += pt.w * d.c * Sh_(b); }
      for (int c = 0; c < dof_t; c++)
      {
        double s = d.h * St_(c);
        for (int j = 0; j < dim; j++) { s += dt * d.flux(j) * DSt_(c, j); }
        (*elvec[2])(c) += pt.w * s;
      }
    }
  }

  template <int dim>
  void Tangent(const mfem::Array<const mfem::FiniteElement *> &el, mfem::ElementTransformation &Tr,
               const mfem::Array<const mfem::Vector *> &elfun,
               const mfem::Array2D<mfem::DenseMatrix *> &elmats)
  {
    Prepare<dim>(el, elfun);
    const int dof_u = el[0]->GetDof(), dof_p = el[1]->GetDof(), dof_t = el[2]->GetDof();
    const int n_u = dof_u * dim;
    const int sizes[3] = {n_u, dof_p, dof_t};
    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++)
      {
        elmats(r, c)->SetSize(sizes[r], sizes[c]);
        *elmats(r, c) = 0.0;
      }
    const Material &m = MaterialOf(Tr);
    const double inv_kappa = InvKappa(m);
    const double dt = history_->Dt();
    const mfem::IntegrationRule &ir = Rule(*el[0]);
    int idx[10][2];
    std::vector<double> row_u(n_u), row_p(dof_p), row_t(dof_t), col(n_u + dof_p + dof_t);
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const Point<dim> pt = PointSetup<dim>(el, Tr, ir.IntPoint(q), q, elfun);
      const int n = Entries<dim>(idx, pt.hoop);
      // Seeds: the n entries of F, p, theta, the dim components of Grad theta.
      const int n_seeds = n + 2 + dim;
      tensor<dual, 3, 3> Fd;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) { Fd(i, j) = dual(pt.F(i, j), 0.0); }
      dual pd(pt.p, 0.0), td(pt.theta, 0.0);
      tensor<dual, 3> gd;
      for (int j = 0; j < 3; j++) { gd(j) = dual(pt.grad_theta(j), 0.0); }
      for (int s = 0; s < n_seeds; s++)
      {
        // Seed.
        if (s < n) { Fd(idx[s][0], idx[s][1]).d = 1.0; }
        else if (s == n) { pd.d = 1.0; }
        else if (s == n + 1) { td.d = 1.0; }
        else { gd(s - n - 2).d = 1.0; }
        const ThermoPointDensities<dual> d =
          ThermoDensities(m, Fd, pd, td, gd, pt.C_old, pt.theta_old, inv_kappa);
        if (s < n) { Fd(idx[s][0], idx[s][1]).d = 0.0; }
        else if (s == n) { pd.d = 0.0; }
        else if (s == n + 1) { td.d = 0.0; }
        else { gd(s - n - 2).d = 0.0; }
        // The derivative of every residual row with respect to the seed.
        for (int a = 0; a < dof_u; a++)
          for (int i = 0; i < dim; i++)
          {
            double v = 0.0;
            for (int mm = 0; mm < n; mm++) { v += d.P(idx[mm][0], idx[mm][1]).d * Brow<dim>(a, i, mm, idx, pt); }
            row_u[std::size_t(a + i * dof_u)] = pt.w * v;
          }
        for (int b = 0; b < dof_p; b++) { row_p[std::size_t(b)] = pt.w * d.c.d * Sh_(b); }
        for (int c = 0; c < dof_t; c++)
        {
          double v = d.h.d * St_(c);
          for (int j = 0; j < dim; j++) { v += dt * d.flux(j).d * DSt_(c, j); }
          row_t[std::size_t(c)] = pt.w * v;
        }
        // The dependence of the seed on the dofs, and the rank-one update.
        if (s < n)
        {
          for (int b = 0; b < dof_u; b++)
            for (int k = 0; k < dim; k++)
            {
              const double cv = Brow<dim>(b, k, s, idx, pt);
              if (cv == 0.0) { continue; }
              const int cj = b + k * dof_u;
              for (int r = 0; r < n_u; r++) { (*elmats(0, 0))(r, cj) += row_u[std::size_t(r)] * cv; }
              for (int r = 0; r < dof_p; r++) { (*elmats(1, 0))(r, cj) += row_p[std::size_t(r)] * cv; }
              for (int r = 0; r < dof_t; r++) { (*elmats(2, 0))(r, cj) += row_t[std::size_t(r)] * cv; }
            }
        }
        else if (s == n)
        {
          for (int e = 0; e < dof_p; e++)
          {
            const double cv = Sh_(e);
            for (int r = 0; r < n_u; r++) { (*elmats(0, 1))(r, e) += row_u[std::size_t(r)] * cv; }
            for (int r = 0; r < dof_p; r++) { (*elmats(1, 1))(r, e) += row_p[std::size_t(r)] * cv; }
            for (int r = 0; r < dof_t; r++) { (*elmats(2, 1))(r, e) += row_t[std::size_t(r)] * cv; }
          }
        }
        else
        {
          const int j = s - n - 2; // -1 for the temperature value itself
          for (int f = 0; f < dof_t; f++)
          {
            const double cv = j < 0 ? St_(f) : DSt_(f, j);
            for (int r = 0; r < n_u; r++) { (*elmats(0, 2))(r, f) += row_u[std::size_t(r)] * cv; }
            for (int r = 0; r < dof_p; r++) { (*elmats(1, 2))(r, f) += row_p[std::size_t(r)] * cv; }
            for (int r = 0; r < dof_t; r++) { (*elmats(2, 2))(r, f) += row_t[std::size_t(r)] * cv; }
          }
        }
      }
    }
  }

  std::vector<Material> materials_;
  const HistoryField *history_ = nullptr;
  bool axisymmetric_ = false;
  mfem::DenseMatrix DSh_, DS_, DSth_, DSt_, Jrt_, Hmat_, PMatI_;
  mfem::Vector Sh_, St_, shape_, X_;
};

// A boundary integrator of the mixed u-p form (follower pressure, contact)
// in the u-p-theta form: the wrapped integrator fills the displacement and
// pressure entries and the temperature entries are sized to zero, since the
// block form keeps one set of element vectors and matrices across
// integrators (a block left untouched would carry the previous element's).
class ThreeBlockFaceAdapter : public mfem::BlockNonlinearFormIntegrator
{
public:
  explicit ThreeBlockFaceAdapter(mfem::BlockNonlinearFormIntegrator *inner) : inner_(inner) {}

  void AssembleFaceVector(const mfem::Array<const mfem::FiniteElement *> &el1,
                          const mfem::Array<const mfem::FiniteElement *> &el2,
                          mfem::FaceElementTransformations &Tr,
                          const mfem::Array<const mfem::Vector *> &elfun,
                          const mfem::Array<mfem::Vector *> &elvect) override
  {
    inner_->AssembleFaceVector(el1, el2, Tr, elfun, elvect);
    elvect[2]->SetSize(0);
  }

  void AssembleFaceGrad(const mfem::Array<const mfem::FiniteElement *> &el1,
                        const mfem::Array<const mfem::FiniteElement *> &el2,
                        mfem::FaceElementTransformations &Tr,
                        const mfem::Array<const mfem::Vector *> &elfun,
                        const mfem::Array2D<mfem::DenseMatrix *> &elmats) override
  {
    inner_->AssembleFaceGrad(el1, el2, Tr, elfun, elmats);
    for (int b = 0; b < 3; b++)
    {
      elmats(b, 2)->SetSize(0);
      elmats(2, b)->SetSize(0);
    }
  }

private:
  std::unique_ptr<mfem::BlockNonlinearFormIntegrator> inner_;
};

// Inward heat flux h on a boundary, per unit current area (through the areal
// Jacobian |cof F N|, with its tangent in the displacement) or per unit
// reference area: R_t(q) -= dt s(t) int h |cof F N| q dA_R on the
// temperature block. `scale` is the entry's schedule value, `history` gives dt.
class HeatFluxIntegrator : public mfem::BlockNonlinearFormIntegrator
{
public:
  HeatFluxIntegrator(mfem::Coefficient &flux, const double *scale, bool current_area,
                     const HistoryField *history, bool axisymmetric = false)
    : flux_(flux), scale_(scale), current_area_(current_area), history_(history),
      axisymmetric_(axisymmetric) {}

  void AssembleFaceVector(const mfem::Array<const mfem::FiniteElement *> &el1,
                          const mfem::Array<const mfem::FiniteElement *> &,
                          mfem::FaceElementTransformations &Tr,
                          const mfem::Array<const mfem::Vector *> &elfun,
                          const mfem::Array<mfem::Vector *> &elvect) override
  {
    if (Tr.GetSpaceDim() == 2) { Face<2>(el1, Tr, elfun, &elvect, nullptr); }
    else { Face<3>(el1, Tr, elfun, &elvect, nullptr); }
  }

  void AssembleFaceGrad(const mfem::Array<const mfem::FiniteElement *> &el1,
                        const mfem::Array<const mfem::FiniteElement *> &,
                        mfem::FaceElementTransformations &Tr,
                        const mfem::Array<const mfem::Vector *> &elfun,
                        const mfem::Array2D<mfem::DenseMatrix *> &elmats) override
  {
    if (Tr.GetSpaceDim() == 2) { Face<2>(el1, Tr, elfun, nullptr, &elmats); }
    else { Face<3>(el1, Tr, elfun, nullptr, &elmats); }
  }

private:
  template <int dim>
  void Face(const mfem::Array<const mfem::FiniteElement *> &el, mfem::FaceElementTransformations &Tr,
            const mfem::Array<const mfem::Vector *> &elfun, const mfem::Array<mfem::Vector *> *elvect,
            const mfem::Array2D<mfem::DenseMatrix *> *elmats)
  {
    const mfem::FiniteElement &eu = *el[0], &et = *el[2];
    const int dof_u = eu.GetDof(), dof_p = el[1]->GetDof(), dof_t = et.GetDof();
    if (elvect)
    {
      (*elvect)[0]->SetSize(0);
      (*elvect)[1]->SetSize(0);
      (*elvect)[2]->SetSize(dof_t);
      *(*elvect)[2] = 0.0;
    }
    if (elmats)
    {
      const int sizes[3] = {dof_u * dim, dof_p, dof_t};
      for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
        {
          if (r == 2 && c == 0) { (*elmats)(r, c)->SetSize(dof_t, dof_u * dim); *(*elmats)(r, c) = 0.0; }
          else { (*elmats)(r, c)->SetSize(0); }
        }
      (void)sizes;
    }
    const double scale = *scale_ * history_->Dt();
    if (scale == 0.0) { return; }
    mfem::DenseMatrix PMatI(const_cast<double *>(elfun[0]->GetData()), dof_u, dim);
    mfem::DenseMatrix DSh(dof_u, dim), DS(dof_u, dim), Jrt(dim), Hmat(dim);
    mfem::Vector shape_u(dof_u), shape_t(dof_t), nor(dim), xf(dim), xc(dim);
    const mfem::IntegrationRule &ir = mfem::IntRules.Get(Tr.GetGeometryType(), 2 * eu.GetOrder() + 3);
    mfem::ElementTransformation &T1 = Tr.GetElement1Transformation();
    double orientation = 1.0;
    {
      const mfem::IntegrationPoint &fc = mfem::Geometries.GetCenter(Tr.GetGeometryType());
      Tr.SetAllIntPoints(&fc);
      mfem::CalcOrtho(Tr.Jacobian(), nor);
      Tr.Transform(fc, xf);
      T1.Transform(mfem::Geometries.GetCenter(eu.GetGeomType()), xc);
      double d = 0.0;
      for (int i = 0; i < dim; i++) { d += nor(i) * (xf(i) - xc(i)); }
      if (d < 0.0) { orientation = -1.0; }
    }
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
      Tr.SetAllIntPoints(&ip);
      const mfem::IntegrationPoint &eip = Tr.GetElement1IntPoint();
      mfem::CalcOrtho(Tr.Jacobian(), nor);
      et.CalcShape(eip, shape_t);
      double w = scale * ip.weight * flux_.Eval(Tr, ip);
      double r = 0.0;
      if (axisymmetric_)
      {
        Tr.Transform(ip, xf);
        r = xf(0);
        w *= 2.0 * M_PI * r;
      }
      tensor<double, 3> N;
      for (int i = 0; i < 3; i++) { N(i) = i < dim ? orientation * nor(i) : 0.0; }
      if (!current_area_)
      {
        const double area = std::sqrt(dot(N, N));
        if (elvect)
        {
          for (int c = 0; c < dof_t; c++) { (*(*elvect)[2])(c) -= w * area * shape_t(c); }
        }
        continue;
      }
      eu.CalcShape(eip, shape_u);
      eu.CalcDShape(eip, DSh);
      mfem::CalcInverse(T1.Jacobian(), Jrt);
      mfem::Mult(DSh, Jrt, DS);
      mfem::MultAtB(PMatI, DS, Hmat);
      tensor<double, dim, dim> H;
      for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++) { H(i, j) = Hmat(i, j); }
      tensor<double, 3, 3> F = DeformationGradient<dim>(H);
      if (axisymmetric_)
      {
        double u_r = 0.0;
        for (int a = 0; a < dof_u; a++) { u_r += shape_u(a) * PMatI(a, 0); }
        F(2, 2) = r > 0.0 ? 1.0 + u_r / r : 1.0 + H(0, 0);
      }
      if (elvect)
      {
        const tensor<double, 3> v = (det(F) * transpose(inv(F))) * N;
        const double area = std::sqrt(dot(v, v));
        for (int c = 0; c < dof_t; c++) { (*(*elvect)[2])(c) -= w * area * shape_t(c); }
      }
      if (elmats)
      {
        tensor<dual, 3, 3> Fd;
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) { Fd(i, j) = dual(F(i, j), 0.0); }
        tensor<dual, 3> Nd;
        for (int i = 0; i < 3; i++) { Nd(i) = dual(N(i), 0.0); }
        for (int b = 0; b < dof_u; b++)
          for (int k = 0; k < dim; k++)
          {
            for (int j = 0; j < dim; j++) { Fd(k, j).d = DS(b, j); }
            if (axisymmetric_ && k == 0) { Fd(2, 2).d = r > 0.0 ? shape_u(b) / r : DS(b, 0); }
            const tensor<dual, 3> v = (det(Fd) * transpose(inv(Fd))) * Nd;
            const dual area = sqrt(dot(v, v));
            for (int j = 0; j < dim; j++) { Fd(k, j).d = 0.0; }
            Fd(2, 2).d = 0.0;
            for (int c = 0; c < dof_t; c++) { (*(*elmats)(2, 0))(c, b + k * dof_u) -= w * area.d * shape_t(c); }
          }
      }
    }
  }

  mfem::Coefficient &flux_;
  const double *scale_;
  bool current_area_;
  const HistoryField *history_;
  bool axisymmetric_;
};

} // namespace cmf

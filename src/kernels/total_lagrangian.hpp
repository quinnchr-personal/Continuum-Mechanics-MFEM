// Total Lagrangian CG kernel: the quadrature-point contract (free functions on
// plain tensors) and the NonlinearFormIntegrator that assembles it.
//
//   R(u).w = int P(F) : Grad w dV,   F = I + Grad u
//   K(u)   = int Grad w : A : Grad du dV,  A = dP/dF (dual numbers)
//
// The qpoint bodies are free functions so they can be lifted into the general
// F(u, grad u) . S(u) . Fhat contract when the second physics arrives.
#pragma once

#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

#include "base/tensor.hpp"
#include "kernels/history_bound.hpp"
#include "materials/kinematics.hpp"
#include "materials/material_tangent.hpp"
#include "mfem.hpp"

namespace cmf
{

// The material contract is PK1<T>(F); Energy<T>(F) is optional and only
// used by GetElementEnergy (ParNonlinearForm::GetEnergy).
template <typename M, typename = void>
struct has_energy : std::false_type {};
template <typename M>
struct has_energy<M, std::void_t<decltype(std::declval<const M &>().Energy(
  std::declval<const tensor<double, 3, 3> &>()))>> : std::true_type {};

// F = I + H, embedding a 2x2 displacement gradient as plane strain (F33 = 1).
template <int dim>
inline tensor<double, 3, 3> DeformationGradient(const tensor<double, dim, dim> &H)
{
  tensor<double, 3, 3> F = I<3>();
  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++) { F(i, j) += H(i, j); }
  return F;
}

// Materials that determine part of F themselves (the plane-stress adapter:
// F33 = thickness stretch) expose Complete(F); the outputs (Cauchy stress,
// J) use the completed F. Everything else keeps the padded F.
template <typename M, typename = void>
struct has_complete : std::false_type {};
template <typename M>
struct has_complete<M, std::void_t<decltype(std::declval<const M &>().Complete(
  std::declval<const tensor<double, 3, 3> &>()))>> : std::true_type {};

template <typename Material>
inline tensor<double, 3, 3> CompleteF(const Material &material, const tensor<double, 3, 3> &F)
{
  if constexpr (has_complete<Material>::value) { return material.Complete(F); }
  else { return F; }
}

// In-plane block of P(F(H)).
template <typename Material, int dim>
inline tensor<double, dim, dim> QPointStress(const Material &material,
                                             const tensor<double, dim, dim> &H)
{
  const tensor<double, 3, 3> P = material.PK1(DeformationGradient<dim>(H));
  tensor<double, dim, dim> Pd;
  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++) { Pd(i, j) = P(i, j); }
  return Pd;
}

// A_ijkl = dP_ij/dF_kl for the in-plane indices (dim x dim seeds).
template <typename Material, int dim>
inline tensor<double, 3, 3, 3, 3> QPointTangent(const Material &material,
                                                const tensor<double, dim, dim> &H)
{
  return MaterialTangent(material, DeformationGradient<dim>(H), dim);
}

// Axisymmetric kinematics on a 2D mesh with x = r and y = z: the hoop stretch
// F_33 = 1 + u_r / r (F_rr on the axis, its limit), the weight 2 pi r on every
// integral (so that forces, energies and masses are those of the whole solid
// of revolution), and the hoop term of the test-function gradient, w_r / r in
// the (3, 3) slot. A dof (a, i) of the element enters the in-plane entries
// F_ij through DS(a, j) and, for i = r, F_33 through N_a / r (DS(a, r) on
// the axis): its B-row over the five entries 00, 01, 10, 11, 33.
struct AxisymmetricPoint
{
  double r = 0.0;
  double weight = 1.0;   // 2 pi r
  double F33 = 1.0;
  bool on_axis = false;
  // dF_33 for the r-dof a: N_a / r, or DS(a, r) on the axis.
  double Hoop(int a, const mfem::Vector &shape, const mfem::DenseMatrix &DS) const
  {
    return on_axis ? DS(a, 0) : shape(a) / r;
  }
};

// The point from the reference position X, the displacement u and the
// gradient H at a quadrature point.
inline AxisymmetricPoint AxisymmetricAt(double r, double u_r, const tensor<double, 2, 2> &H)
{
  AxisymmetricPoint a;
  a.r = r;
  a.on_axis = !(r > 0.0);
  a.F33 = a.on_axis ? 1.0 + H(0, 0) : 1.0 + u_r / r;
  a.weight = 2.0 * M_PI * r;
  return a;
}

// The B-row of dof (a, i) over the entries 00, 01, 10, 11, 33.
inline void AxisymmetricRow(int a, int i, const mfem::Vector &shape, const mfem::DenseMatrix &DS,
                            const AxisymmetricPoint &pt, double *B)
{
  for (int m = 0; m < 4; m++) { B[m] = (i == m / 2) ? DS(a, m % 2) : 0.0; }
  B[4] = i == 0 ? pt.Hoop(a, shape, DS) : 0.0;
}

// A_ijkl seeded on the in-plane entries and on (3, 3): the tangent of an
// axisymmetric point.
template <typename Material>
inline tensor<double, 3, 3, 3, 3> AxisymmetricTangent(const Material &material,
                                                      const tensor<double, 3, 3> &F)
{
  tensor<double, 3, 3, 3, 3> A = MaterialTangent(material, F, 2);
  tensor<dual, 3, 3> Fd;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) { Fd(i, j) = dual(F(i, j), 0.0); }
  Fd(2, 2).d = 1.0;
  const tensor<dual, 3, 3> P = material.PK1(Fd);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) { A(i, j, 2, 2) = P(i, j).d; }
  return A;
}

// The five entries of a 3x3 tensor in the order 00, 01, 10, 11, 33.
inline void Pack5(const tensor<double, 3, 3> &A, double *v)
{
  v[0] = A(0, 0); v[1] = A(0, 1); v[2] = A(1, 0); v[3] = A(1, 1); v[4] = A(2, 2);
}

// Cauchy stress at a point, full 3x3 (plane strain keeps sigma_33, plane
// stress uses the thickness stretch): sigma = J^{-1} P F^T, or P itself for a
// small-strain material (materials/kinematics.hpp); and its von Mises
// equivalent.
template <typename Material, int dim>
inline tensor<double, 3, 3> QPointCauchyStress(const Material &material,
                                               const tensor<double, dim, dim> &H)
{
  const tensor<double, 3, 3> F = CompleteF(material, DeformationGradient<dim>(H));
  return CauchyStress(material, F, material.PK1(F));
}

inline double VonMises(const tensor<double, 3, 3> &sigma)
{
  const tensor<double, 3, 3> s = dev(sym(sigma));
  return std::sqrt(1.5 * ddot(s, s));
}

// Consumes any material with the PK1<T>(F) signature (template, not virtual).
// One material for every element, or a table indexed by element attribute
// (size max attribute + 1; entry 0 unused). A history-dependent material
// (kernels/history_bound.hpp) is evaluated through the HistoryField given
// to SetHistory, whose rule must be this integrator's (order 2p + 3).
template <typename Material>
class TotalLagrangianIntegrator : public mfem::NonlinearFormIntegrator
{
public:
  explicit TotalLagrangianIntegrator(const Material &material)
    : materials_(1, material) {}
  explicit TotalLagrangianIntegrator(const std::vector<Material> &by_attribute)
    : materials_(by_attribute) {}

  void SetHistory(const HistoryField *history) { history_ = history; }
  const HistoryField *History() const { return history_; }
  // Axisymmetric kinematics (2D mesh, x = r, y = z; see AxisymmetricPoint).
  void SetAxisymmetric(bool on) { axisymmetric_ = on; }
  bool Axisymmetric() const { return axisymmetric_; }

  mfem::real_t GetElementEnergy(const mfem::FiniteElement &el,
                                mfem::ElementTransformation &Tr,
                                const mfem::Vector &elfun) override
  {
    return Tr.GetDimension() == 2 ? Energy<2>(el, Tr, elfun)
                                  : Energy<3>(el, Tr, elfun);
  }

  void AssembleElementVector(const mfem::FiniteElement &el,
                             mfem::ElementTransformation &Tr,
                             const mfem::Vector &elfun,
                             mfem::Vector &elvect) override
  {
    if (Tr.GetDimension() == 2) { Residual<2>(el, Tr, elfun, elvect); }
    else { Residual<3>(el, Tr, elfun, elvect); }
  }

  void AssembleElementGrad(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &Tr,
                           const mfem::Vector &elfun,
                           mfem::DenseMatrix &elmat) override
  {
    if (Tr.GetDimension() == 2) { Tangent<2>(el, Tr, elfun, elmat); }
    else { Tangent<3>(el, Tr, elfun, elmat); }
  }

  const Material &GetMaterial() const { return materials_[0]; }
  const Material &MaterialOf(const mfem::ElementTransformation &Tr) const
  {
    return materials_.size() == 1 ? materials_[0] : materials_[std::size_t(Tr.Attribute)];
  }

private:
  const mfem::IntegrationRule &Rule(const mfem::FiniteElement &el,
                                    mfem::ElementTransformation &Tr) const
  {
    const mfem::IntegrationRule *ir = GetIntegrationRule(el, Tr);
    if (ir) { return *ir; }
    return mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3);
  }

  // Physical-space shape gradients DS (dof x dim) and H = Grad u at ip.
  template <int dim>
  void PointSetup(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
                  const mfem::IntegrationPoint &ip, tensor<double, dim, dim> &H)
  {
    Tr.SetIntPoint(&ip);
    mfem::CalcInverse(Tr.Jacobian(), Jrt_);
    el.CalcDShape(ip, DSh_);
    mfem::Mult(DSh_, Jrt_, DS_);
    mfem::MultAtB(PMatI_, DS_, Hmat_);
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++) { H(i, j) = Hmat_(i, j); }
  }

  template <int dim>
  void Prepare(const mfem::FiniteElement &el, const mfem::Vector &elfun)
  {
    const int dof = el.GetDof();
    DSh_.SetSize(dof, dim);
    DS_.SetSize(dof, dim);
    Jrt_.SetSize(dim);
    Hmat_.SetSize(dim);
    PMatI_.UseExternalData(elfun.GetData(), dof, dim);
    if (axisymmetric_) { shape_.SetSize(dof); }
  }

  // The axisymmetric data of the point set up by PointSetup<2>: r, u_r, F_33
  // and the weight; the shape functions for the hoop rows.
  AxisymmetricPoint AxiSetup(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
                             const mfem::IntegrationPoint &ip, const tensor<double, 2, 2> &H)
  {
    el.CalcShape(ip, shape_);
    Tr.Transform(ip, X_);
    double u_r = 0.0;
    for (int a = 0; a < el.GetDof(); a++) { u_r += shape_(a) * PMatI_(a, 0); }
    return AxisymmetricAt(X_(0), u_r, H);
  }

  template <int dim>
  double Energy(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
                const mfem::Vector &elfun)
  {
    if constexpr (!has_energy<bound_t<Material>>::value)
    {
      MFEM_ABORT("TotalLagrangianIntegrator: this material has no Energy(F)");
      return 0.0;
    }
    else
    {
      Prepare<dim>(el, elfun);
      const Material &material = MaterialOf(Tr);
      CheckHistory();
      const mfem::IntegrationRule &ir = Rule(el, Tr);
      double energy = 0.0;
      tensor<double, dim, dim> H;
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
        const mfem::IntegrationPoint &ip = ir.IntPoint(q);
        PointSetup<dim>(el, Tr, ip, H);
        const auto &mat = AtPoint(material, history_, Tr.ElementNo, q);
        tensor<double, 3, 3> F = DeformationGradient<dim>(H);
        double w = ip.weight * Tr.Weight();
        if constexpr (dim == 2)
        {
          if (axisymmetric_)
          {
            const AxisymmetricPoint pt = AxiSetup(el, Tr, ip, H);
            F(2, 2) = pt.F33;
            w *= pt.weight;
          }
        }
        energy += w * mat.Energy(F);
      }
      return energy;
    }
  }

  template <int dim>
  void Residual(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
                const mfem::Vector &elfun, mfem::Vector &elvect)
  {
    const int dof = el.GetDof();
    Prepare<dim>(el, elfun);
    elvect.SetSize(dof * dim);
    elvect = 0.0;
    mfem::DenseMatrix PMatO(elvect.GetData(), dof, dim);
    const Material &material = MaterialOf(Tr);
    CheckHistory();
    const mfem::IntegrationRule &ir = Rule(el, Tr);
    tensor<double, dim, dim> H;
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
      PointSetup<dim>(el, Tr, ip, H);
      const auto &mat = AtPoint(material, history_, Tr.ElementNo, q);
      if constexpr (dim == 2)
      {
        if (axisymmetric_)
        {
          // r(a, i) += w [P_ij DS(a, j) + delta_ir P_33 N_a / r]
          const AxisymmetricPoint pt = AxiSetup(el, Tr, ip, H);
          tensor<double, 3, 3> F = DeformationGradient<2>(H);
          F(2, 2) = pt.F33;
          const tensor<double, 3, 3> P = mat.PK1(F);
          const double w = ip.weight * Tr.Weight() * pt.weight;
          for (int a = 0; a < dof; a++)
            for (int i = 0; i < 2; i++)
            {
              double s = 0.0;
              for (int j = 0; j < 2; j++) { s += P(i, j) * DS_(a, j); }
              if (i == 0) { s += P(2, 2) * pt.Hoop(a, shape_, DS_); }
              PMatO(a, i) += w * s;
            }
          continue;
        }
      }
      const tensor<double, dim, dim> P = QPointStress<bound_t<Material>, dim>(mat, H);
      const double w = ip.weight * Tr.Weight();
      // r(a, i) += w P_ij DS(a, j)
      for (int a = 0; a < dof; a++)
        for (int i = 0; i < dim; i++)
        {
          double s = 0.0;
          for (int j = 0; j < dim; j++) { s += P(i, j) * DS_(a, j); }
          PMatO(a, i) += w * s;
        }
    }
  }

  template <int dim>
  void Tangent(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
               const mfem::Vector &elfun, mfem::DenseMatrix &elmat)
  {
    const int dof = el.GetDof();
    Prepare<dim>(el, elfun);
    elmat.SetSize(dof * dim);
    elmat = 0.0;
    const Material &material = MaterialOf(Tr);
    CheckHistory();
    const mfem::IntegrationRule &ir = Rule(el, Tr);
    tensor<double, dim, dim> H;
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
      PointSetup<dim>(el, Tr, ip, H);
      const auto &mat = AtPoint(material, history_, Tr.ElementNo, q);
      if constexpr (dim == 2)
      {
        if (axisymmetric_)
        {
          const AxisymmetricPoint pt = AxiSetup(el, Tr, ip, H);
          tensor<double, 3, 3> F = DeformationGradient<2>(H);
          F(2, 2) = pt.F33;
          AxisymmetricTangent5(mat, F, pt, el, Tr, ip, elmat);
          continue;
        }
      }
      const tensor<double, 3, 3, 3, 3> A = QPointTangent<bound_t<Material>, dim>(mat, H);
      const double w = ip.weight * Tr.Weight();
      // K(a i, b k) += w DS(a, j) A_ijkl DS(b, l)
      for (int a = 0; a < dof; a++)
        for (int i = 0; i < dim; i++)
          for (int k = 0; k < dim; k++)
          {
            double t[3] = {0.0, 0.0, 0.0};
            for (int l = 0; l < dim; l++)
              for (int j = 0; j < dim; j++) { t[l] += DS_(a, j) * A(i, j, k, l); }
            for (int b = 0; b < dof; b++)
            {
              double s = 0.0;
              for (int l = 0; l < dim; l++) { s += t[l] * DS_(b, l); }
              elmat(a + i * dof, b + k * dof) += w * s;
            }
          }
    }
  }

  // K(a i, b k) += w B(a i, m) A_mn B(b k, n) over the five entries of an
  // axisymmetric point (AxisymmetricRow), with the tangent A seeded on them.
  template <typename Bound>
  void AxisymmetricTangent5(const Bound &mat, const tensor<double, 3, 3> &F,
                            const AxisymmetricPoint &pt, const mfem::FiniteElement &el,
                            mfem::ElementTransformation &Tr, const mfem::IntegrationPoint &ip,
                            mfem::DenseMatrix &elmat)
  {
    const int dof = el.GetDof();
    const tensor<double, 3, 3, 3, 3> A = AxisymmetricTangent(mat, F);
    const double w = ip.weight * Tr.Weight() * pt.weight;
    static const int idx[5][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 2}};
    double A5[5][5];
    for (int m = 0; m < 5; m++)
      for (int n = 0; n < 5; n++) { A5[m][n] = A(idx[m][0], idx[m][1], idx[n][0], idx[n][1]); }
    double Ba[5], Bb[5], t[5];
    for (int a = 0; a < dof; a++)
      for (int i = 0; i < 2; i++)
      {
        AxisymmetricRow(a, i, shape_, DS_, pt, Ba);
        for (int n = 0; n < 5; n++)
        {
          t[n] = 0.0;
          for (int m = 0; m < 5; m++) { t[n] += Ba[m] * A5[m][n]; }
        }
        for (int b = 0; b < dof; b++)
          for (int k = 0; k < 2; k++)
          {
            AxisymmetricRow(b, k, shape_, DS_, pt, Bb);
            double s = 0.0;
            for (int n = 0; n < 5; n++) { s += t[n] * Bb[n]; }
            elmat(a + i * dof, b + k * dof) += w * s;
          }
      }
  }

  void CheckHistory() const
  {
    if constexpr (has_history<Material>::value)
    {
      MFEM_VERIFY(history_, "TotalLagrangianIntegrator: a history-dependent material needs SetHistory");
    }
  }

  std::vector<Material> materials_;
  const HistoryField *history_ = nullptr;
  bool axisymmetric_ = false;
  mfem::DenseMatrix DSh_, DS_, Jrt_, Hmat_, PMatI_;
  mfem::Vector shape_, X_;
};

} // namespace cmf

// Total Lagrangian CG kernel: the quadrature-point contract (free functions on
// plain tensors) and the NonlinearFormIntegrator that assembles it.
//
//   R(u).w = int P(F) : Grad w dV,   F = I + Grad u
//   K(u)   = int Grad w : A : Grad du dV,  A = dP/dF (dual numbers)
//
// The qpoint bodies are free functions so they can be lifted into the general
// F(u, grad u) . S(u) . Fhat contract when the second physics arrives.
#pragma once

#include "base/tensor.hpp"
#include "materials/material_tangent.hpp"
#include "mfem.hpp"

namespace cmf
{

// F = I + H, embedding a 2x2 displacement gradient as plane strain (F33 = 1).
template <int dim>
inline tensor<double, 3, 3> DeformationGradient(const tensor<double, dim, dim> &H)
{
  tensor<double, 3, 3> F = I<3>();
  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++) { F(i, j) += H(i, j); }
  return F;
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

// Cauchy stress sigma = J^{-1} P F^T at a point, full 3x3 (plane strain keeps
// sigma_33), and its von Mises equivalent.
template <typename Material, int dim>
inline tensor<double, 3, 3> QPointCauchyStress(const Material &material,
                                               const tensor<double, dim, dim> &H)
{
  const tensor<double, 3, 3> F = DeformationGradient<dim>(H);
  const tensor<double, 3, 3> P = material.PK1(F);
  return (1.0 / det(F)) * (P * transpose(F));
}

inline double VonMises(const tensor<double, 3, 3> &sigma)
{
  const tensor<double, 3, 3> s = dev(sym(sigma));
  return std::sqrt(1.5 * ddot(s, s));
}

// Consumes any material with the PK1<T>(F) signature (template, not virtual).
template <typename Material>
class TotalLagrangianIntegrator : public mfem::NonlinearFormIntegrator
{
public:
  explicit TotalLagrangianIntegrator(const Material &material)
    : material_(material) {}

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

  const Material &GetMaterial() const { return material_; }

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
  }

  template <int dim>
  double Energy(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
                const mfem::Vector &elfun)
  {
    Prepare<dim>(el, elfun);
    const mfem::IntegrationRule &ir = Rule(el, Tr);
    double energy = 0.0;
    tensor<double, dim, dim> H;
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
      PointSetup<dim>(el, Tr, ip, H);
      energy += ip.weight * Tr.Weight() *
                material_.Energy(DeformationGradient<dim>(H));
    }
    return energy;
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
    const mfem::IntegrationRule &ir = Rule(el, Tr);
    tensor<double, dim, dim> H;
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
      PointSetup<dim>(el, Tr, ip, H);
      const tensor<double, dim, dim> P = QPointStress<Material, dim>(material_, H);
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
    const mfem::IntegrationRule &ir = Rule(el, Tr);
    tensor<double, dim, dim> H;
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
      PointSetup<dim>(el, Tr, ip, H);
      const tensor<double, 3, 3, 3, 3> A = QPointTangent<Material, dim>(material_, H);
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

  Material material_;
  mfem::DenseMatrix DSh_, DS_, Jrt_, Hmat_, PMatI_;
};

} // namespace cmf

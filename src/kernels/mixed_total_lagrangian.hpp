// Mixed displacement-pressure total Lagrangian CG kernel (Taylor-Hood style
// spaces) for decoupled hyperelastic materials with P = P_iso(F) + p J F^{-T}.
//
//   R_u(u, p).w = int [P_iso(F) + p J F^{-T}] : Grad w dV
//   R_p(u, p).q = int q (u'(J) - p / kappa) dV      (kappa = inf: q u'(J), i.e. J = 1)
// with U(J) = kappa u(J) the material's volumetric law (materials/volumetric.hpp;
// u' = J - 1 for the default quadratic law, which recovers J - 1 - p / kappa).
//
// Tangent blocks: K_uu = int Grad w : A(F, p) : Grad du with A = dP/dF at
// fixed p (dual seeding), K_up = int (J F^{-T}) : Grad w  q,
// K_pu = int q u''(J) (J F^{-T}) : Grad du (= K_up^T for the quadratic law),
// K_pp = -1/kappa int q dp.
#pragma once

#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

#include "base/dual.hpp"
#include "base/tensor.hpp"
#include "kernels/total_lagrangian.hpp"
#include "mfem.hpp"

namespace cmf
{

// P(F, p) = P_iso(F) + p J F^{-T}; templated so duals give dP/dF at fixed p.
template <typename Material, typename T>
inline tensor<T, 3, 3> MixedPK1(const Material &material, const tensor<T, 3, 3> &F,
                                double p)
{
  const T J = det(F);
  return material.PK1Iso(F) + (p * J) * transpose(inv(F));
}

template <typename Material, int dim>
inline tensor<double, dim, dim> QPointMixedStress(const Material &material,
                                                  const tensor<double, dim, dim> &H,
                                                  double p)
{
  const tensor<double, 3, 3> P = MixedPK1(material, DeformationGradient<dim>(H), p);
  tensor<double, dim, dim> Pd;
  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++) { Pd(i, j) = P(i, j); }
  return Pd;
}

// A_ijkl = dP_ij/dF_kl at fixed p, in-plane seeds only.
template <typename Material, int dim>
inline tensor<double, 3, 3, 3, 3> QPointMixedTangent(const Material &material,
                                                     const tensor<double, dim, dim> &H,
                                                     double p)
{
  const tensor<double, 3, 3> F = DeformationGradient<dim>(H);
  tensor<double, 3, 3, 3, 3> A;
  tensor<dual, 3, 3> Fd;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) { Fd(i, j) = dual(F(i, j), 0.0); }
  for (int k = 0; k < dim; k++)
    for (int l = 0; l < dim; l++)
    {
      Fd(k, l).d = 1.0;
      const tensor<dual, 3, 3> P = MixedPK1(material, Fd, p);
      Fd(k, l).d = 0.0;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) { A(i, j, k, l) = P(i, j).d; }
    }
  return A;
}

// dJ/dF = J F^{-T}, in-plane block (plane strain: J = det of the 2x2 block).
template <int dim>
inline tensor<double, dim, dim> QPointVolumeGradient(const tensor<double, dim, dim> &H,
                                                     double &J)
{
  const tensor<double, dim, dim> F = I<dim>() + H;
  J = det(F);
  return J * transpose(inv(F));
}

// Cauchy stress of the mixed formulation, sigma = J^{-1} P F^T.
template <typename Material, int dim>
inline tensor<double, 3, 3> QPointMixedCauchyStress(const Material &material,
                                                    const tensor<double, dim, dim> &H,
                                                    double p)
{
  const tensor<double, 3, 3> F = DeformationGradient<dim>(H);
  const tensor<double, 3, 3> P = MixedPK1(material, F, p);
  return (1.0 / det(F)) * (P * transpose(F));
}

// Block integrator over (displacement, pressure) spaces.
template <typename Material>
class MixedTotalLagrangianIntegrator : public mfem::BlockNonlinearFormIntegrator
{
public:
  explicit MixedTotalLagrangianIntegrator(const Material &material)
    : materials_(1, material) {}
  // One material per element attribute (size max attribute + 1, entry 0 unused).
  explicit MixedTotalLagrangianIntegrator(const std::vector<Material> &by_attribute)
    : materials_(by_attribute) {}

  // Mixed functional int Psi_iso(F) + p (J - 1) - kappa u*(p / kappa) dV with
  // u* the Legendre transform of the volumetric law (p^2 / (2 kappa) for the
  // quadratic law, whose perturbed Lagrangian this is); kappa = inf drops the
  // last term.
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

  const Material &GetMaterial() const { return materials_[0]; }
  const Material &MaterialOf(const mfem::ElementTransformation &Tr) const
  {
    return materials_.size() == 1 ? materials_[0] : materials_[std::size_t(Tr.Attribute)];
  }

private:
  static double InvKappa(const Material &m) { return m.Incompressible() ? 0.0 : 1.0 / m.kappa; }

  const mfem::IntegrationRule &Rule(const mfem::FiniteElement &el) const
  {
    return mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3);
  }

  template <int dim>
  void Prepare(const mfem::Array<const mfem::FiniteElement *> &el,
               const mfem::Array<const mfem::Vector *> &elfun)
  {
    MFEM_VERIFY(el.Size() == 2, "mixed integrator needs (displacement, pressure) spaces");
    const int dof_u = el[0]->GetDof();
    const int dof_p = el[1]->GetDof();
    DSh_.SetSize(dof_u, dim);
    DS_.SetSize(dof_u, dim);
    Jrt_.SetSize(dim);
    Hmat_.SetSize(dim);
    Sh_.SetSize(dof_p);
    PMatI_.UseExternalData(elfun[0]->GetData(), dof_u, dim);
  }

  template <int dim>
  void PointSetup(const mfem::Array<const mfem::FiniteElement *> &el,
                  mfem::ElementTransformation &Tr, const mfem::IntegrationPoint &ip,
                  const mfem::Vector &p_dofs, tensor<double, dim, dim> &H, double &p)
  {
    Tr.SetIntPoint(&ip);
    mfem::CalcInverse(Tr.Jacobian(), Jrt_);
    el[0]->CalcDShape(ip, DSh_);
    mfem::Mult(DSh_, Jrt_, DS_);
    mfem::MultAtB(PMatI_, DS_, Hmat_);
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++) { H(i, j) = Hmat_(i, j); }
    el[1]->CalcShape(ip, Sh_);
    p = Sh_ * p_dofs;
  }

  template <int dim>
  double Energy(const mfem::Array<const mfem::FiniteElement *> &el,
                mfem::ElementTransformation &Tr,
                const mfem::Array<const mfem::Vector *> &elfun)
  {
    Prepare<dim>(el, elfun);
    const Material &material = MaterialOf(Tr);
    const double inv_kappa = InvKappa(material);
    const mfem::IntegrationRule &ir = Rule(*el[0]);
    tensor<double, dim, dim> H;
    double p = 0.0, energy = 0.0;
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
      PointSetup<dim>(el, Tr, ip, *elfun[1], H, p);
      const double w = ip.weight * Tr.Weight();
      const tensor<double, 3, 3> F = DeformationGradient<dim>(H);
      const double J = det(F);
      energy += w * (material.EnergyIso(F) + p * (J - 1.0) -
                     (inv_kappa > 0.0 ? material.ComplementaryVolumetricEnergy(p) : 0.0));
    }
    return energy;
  }

  template <int dim>
  void Residual(const mfem::Array<const mfem::FiniteElement *> &el,
                mfem::ElementTransformation &Tr,
                const mfem::Array<const mfem::Vector *> &elfun,
                const mfem::Array<mfem::Vector *> &elvec)
  {
    Prepare<dim>(el, elfun);
    const int dof_u = el[0]->GetDof();
    const int dof_p = el[1]->GetDof();
    elvec[0]->SetSize(dof_u * dim);
    elvec[1]->SetSize(dof_p);
    *elvec[0] = 0.0;
    *elvec[1] = 0.0;
    mfem::DenseMatrix PMatO(elvec[0]->GetData(), dof_u, dim);
    const Material &material = MaterialOf(Tr);
    const double inv_kappa = InvKappa(material);
    const mfem::IntegrationRule &ir = Rule(*el[0]);
    tensor<double, dim, dim> H;
    double p = 0.0;
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
      PointSetup<dim>(el, Tr, ip, *elfun[1], H, p);
      const double w = ip.weight * Tr.Weight();
      const tensor<double, dim, dim> P = QPointMixedStress<Material, dim>(material, H, p);
      for (int a = 0; a < dof_u; a++)
        for (int i = 0; i < dim; i++)
        {
          double s = 0.0;
          for (int j = 0; j < dim; j++) { s += P(i, j) * DS_(a, j); }
          PMatO(a, i) += w * s;
        }
      double J = 1.0;
      QPointVolumeGradient<dim>(H, J);
      const double constraint = material.NormalizedVolumetricPressure(J) - inv_kappa * p;
      for (int b = 0; b < dof_p; b++) { (*elvec[1])(b) += w * constraint * Sh_(b); }
    }
  }

  template <int dim>
  void Tangent(const mfem::Array<const mfem::FiniteElement *> &el,
               mfem::ElementTransformation &Tr,
               const mfem::Array<const mfem::Vector *> &elfun,
               const mfem::Array2D<mfem::DenseMatrix *> &elmats)
  {
    Prepare<dim>(el, elfun);
    const int dof_u = el[0]->GetDof();
    const int dof_p = el[1]->GetDof();
    mfem::DenseMatrix &Kuu = *elmats(0, 0);
    mfem::DenseMatrix &Kup = *elmats(0, 1);
    mfem::DenseMatrix &Kpu = *elmats(1, 0);
    mfem::DenseMatrix &Kpp = *elmats(1, 1);
    Kuu.SetSize(dof_u * dim, dof_u * dim);
    Kup.SetSize(dof_u * dim, dof_p);
    Kpu.SetSize(dof_p, dof_u * dim);
    Kpp.SetSize(dof_p, dof_p);
    Kuu = 0.0;
    Kup = 0.0;
    Kpu = 0.0;
    Kpp = 0.0;
    const Material &material = MaterialOf(Tr);
    const double inv_kappa = InvKappa(material);
    const mfem::IntegrationRule &ir = Rule(*el[0]);
    tensor<double, dim, dim> H;
    double p = 0.0;
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
      PointSetup<dim>(el, Tr, ip, *elfun[1], H, p);
      const double w = ip.weight * Tr.Weight();
      const tensor<double, 3, 3, 3, 3> A =
        QPointMixedTangent<Material, dim>(material, H, p);
      for (int a = 0; a < dof_u; a++)
        for (int i = 0; i < dim; i++)
          for (int k = 0; k < dim; k++)
          {
            double t[3] = {0.0, 0.0, 0.0};
            for (int l = 0; l < dim; l++)
              for (int j = 0; j < dim; j++) { t[l] += DS_(a, j) * A(i, j, k, l); }
            for (int b = 0; b < dof_u; b++)
            {
              double s = 0.0;
              for (int l = 0; l < dim; l++) { s += t[l] * DS_(b, l); }
              Kuu(a + i * dof_u, b + k * dof_u) += w * s;
            }
          }
      double J = 1.0;
      const tensor<double, dim, dim> G = QPointVolumeGradient<dim>(H, J);
      // Kup(a i, b) = w (J F^{-T})_ij DS(a, j) N_b ; Kpu = u''(J) Kup^T
      const double upp = material.NormalizedVolumetricModulus(J);
      for (int a = 0; a < dof_u; a++)
        for (int i = 0; i < dim; i++)
        {
          double s = 0.0;
          for (int j = 0; j < dim; j++) { s += G(i, j) * DS_(a, j); }
          for (int b = 0; b < dof_p; b++)
          {
            const double v = w * s * Sh_(b);
            Kup(a + i * dof_u, b) += v;
            Kpu(b, a + i * dof_u) += upp * v;
          }
        }
      if (inv_kappa > 0.0)
      {
        for (int a = 0; a < dof_p; a++)
          for (int b = 0; b < dof_p; b++)
          {
            Kpp(a, b) -= w * inv_kappa * Sh_(a) * Sh_(b);
          }
      }
    }
  }

  std::vector<Material> materials_;
  mfem::DenseMatrix DSh_, DS_, Jrt_, Hmat_, PMatI_;
  mfem::Vector Sh_;
};

} // namespace cmf

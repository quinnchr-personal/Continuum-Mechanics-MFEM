// Mixed displacement-pressure total Lagrangian CG kernel (Taylor-Hood style
// spaces) for decoupled hyperelastic materials with P = P_iso(F) + p J F^{-T}.
//
//   R_u(u, p).w = int [P_iso(F) + p J F^{-T}] : Grad w dV
//   R_p(u, p).q = int q (J - 1 - p / kappa) dV      (kappa = inf: q (J - 1))
//
// Tangent blocks: K_uu = int Grad w : A(F, p) : Grad du with A = dP/dF at
// fixed p (dual seeding), K_up = int (J F^{-T}) : Grad w  q,
// K_pu = K_up^T, K_pp = -1/kappa int q dp.
#pragma once

#include <cmath>
#include <type_traits>
#include <utility>

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
    : material_(material),
      inv_kappa_(material.Incompressible() ? 0.0 : 1.0 / material.kappa) {}

  // Mixed functional int Psi_iso(F) + p (J - 1) - p^2 / (2 kappa) dV.
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

  const Material &GetMaterial() const { return material_; }

private:
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
      energy += w * (material_.EnergyIso(F) + p * (J - 1.0) - 0.5 * inv_kappa_ * p * p);
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
    const mfem::IntegrationRule &ir = Rule(*el[0]);
    tensor<double, dim, dim> H;
    double p = 0.0;
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
      PointSetup<dim>(el, Tr, ip, *elfun[1], H, p);
      const double w = ip.weight * Tr.Weight();
      const tensor<double, dim, dim> P = QPointMixedStress<Material, dim>(material_, H, p);
      for (int a = 0; a < dof_u; a++)
        for (int i = 0; i < dim; i++)
        {
          double s = 0.0;
          for (int j = 0; j < dim; j++) { s += P(i, j) * DS_(a, j); }
          PMatO(a, i) += w * s;
        }
      double J = 1.0;
      QPointVolumeGradient<dim>(H, J);
      const double constraint = J - 1.0 - inv_kappa_ * p;
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
    const mfem::IntegrationRule &ir = Rule(*el[0]);
    tensor<double, dim, dim> H;
    double p = 0.0;
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
      PointSetup<dim>(el, Tr, ip, *elfun[1], H, p);
      const double w = ip.weight * Tr.Weight();
      const tensor<double, 3, 3, 3, 3> A =
        QPointMixedTangent<Material, dim>(material_, H, p);
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
      // Kup(a i, b) = w (J F^{-T})_ij DS(a, j) N_b ; Kpu = Kup^T
      for (int a = 0; a < dof_u; a++)
        for (int i = 0; i < dim; i++)
        {
          double s = 0.0;
          for (int j = 0; j < dim; j++) { s += G(i, j) * DS_(a, j); }
          for (int b = 0; b < dof_p; b++)
          {
            const double v = w * s * Sh_(b);
            Kup(a + i * dof_u, b) += v;
            Kpu(b, a + i * dof_u) += v;
          }
        }
      if (inv_kappa_ > 0.0)
      {
        for (int a = 0; a < dof_p; a++)
          for (int b = 0; b < dof_p; b++)
          {
            Kpp(a, b) -= w * inv_kappa_ * Sh_(a) * Sh_(b);
          }
      }
    }
  }

  Material material_;
  double inv_kappa_;
  mfem::DenseMatrix DSh_, DS_, Jrt_, Hmat_, PMatI_;
  mfem::Vector Sh_;
};

} // namespace cmf

// Penalty contact of a boundary of the reference mesh with a rigid sphere
// (an indenter; a circle in 2D) of radius r and centre c(t), the contact of
// the reference's FEniCSx codes (finite viscoelasticity, FV12). Per unit
// reference area the penalty energy
//   E = k/2 <r^2 - |x - c|^2>_+^2,   x = X + u the current position,
// gives the traction on the body
//   t = -dE/du = 2 k <r^2 - |x - c|^2>_+ (x - c),
// which pushes whatever lies inside the sphere out of it, and the boundary
// term of the nonlinear form (R = internal - external)
//   R_ext(u).w = -int_Gamma t . w dA_R,
// with the tangent dt/du = 2 k [g I - 2 (x - c) (x) (x - c)] for g > 0
// (symmetric; the kink at g = 0 is that of the penalty). Face quadrature of
// order 2p + 3. `scale` is the schedule value of the entry, owned by the
// LoadSet of the physics; k = scale * penalty.
#pragma once

#include "mfem.hpp"

namespace cmf
{

namespace contact_detail
{

template <int dim>
void FaceContribution(const mfem::FiniteElement &el, mfem::FaceElementTransformations &Tr,
                      const mfem::Vector &elfun, mfem::VectorCoefficient &center, double radius,
                      double k, mfem::Vector *elvect, mfem::DenseMatrix *elmat)
{
  const int dof = el.GetDof();
  mfem::DenseMatrix PMatI(const_cast<double *>(elfun.GetData()), dof, dim);
  mfem::Vector shape(dof), nor(dim), X(dim), c(dim);
  if (elvect) { elvect->SetSize(dof * dim); *elvect = 0.0; }
  if (elmat) { elmat->SetSize(dof * dim); *elmat = 0.0; }
  if (k == 0.0) { return; }

  const mfem::IntegrationRule &ir =
    mfem::IntRules.Get(Tr.GetGeometryType(), 2 * el.GetOrder() + 3);
  for (int q = 0; q < ir.GetNPoints(); q++)
  {
    const mfem::IntegrationPoint &ip = ir.IntPoint(q);
    Tr.SetAllIntPoints(&ip);
    const mfem::IntegrationPoint &eip = Tr.GetElement1IntPoint();
    mfem::CalcOrtho(Tr.Jacobian(), nor); // reference normal times the area element
    el.CalcShape(eip, shape);
    Tr.Transform(ip, X);                 // reference position of the face point
    center.Eval(c, Tr, ip);
    double d[3] = {0.0, 0.0, 0.0}, g = radius * radius;
    for (int i = 0; i < dim; i++)
    {
      double u = 0.0;
      for (int a = 0; a < dof; a++) { u += shape(a) * PMatI(a, i); }
      d[i] = X(i) + u - c(i);
      g -= d[i] * d[i];
    }
    if (g <= 0.0) { continue; }
    const double w = ip.weight * nor.Norml2();
    if (elvect)
    {
      for (int a = 0; a < dof; a++)
        for (int i = 0; i < dim; i++) { (*elvect)(a + i * dof) -= w * shape(a) * 2.0 * k * g * d[i]; }
    }
    if (elmat)
    {
      for (int a = 0; a < dof; a++)
        for (int b = 0; b < dof; b++)
        {
          const double s = w * shape(a) * shape(b) * 2.0 * k;
          for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
            {
              const double dt = (i == j ? g : 0.0) - 2.0 * d[i] * d[j];
              (*elmat)(a + i * dof, b + j * dof) -= s * dt;
            }
        }
    }
  }
}

} // namespace contact_detail

// Displacement formulation: boundary face integrator of a ParNonlinearForm.
class RigidSphereContactIntegrator : public mfem::NonlinearFormIntegrator
{
public:
  RigidSphereContactIntegrator(mfem::VectorCoefficient &center, double radius, double penalty,
                               const double *scale)
    : center_(center), radius_(radius), penalty_(penalty), scale_(scale) {}

  void AssembleFaceVector(const mfem::FiniteElement &el1, const mfem::FiniteElement &,
                          mfem::FaceElementTransformations &Tr, const mfem::Vector &elfun,
                          mfem::Vector &elvect) override
  {
    if (Tr.GetSpaceDim() == 2)
    {
      contact_detail::FaceContribution<2>(el1, Tr, elfun, center_, radius_, *scale_ * penalty_, &elvect, nullptr);
    }
    else
    {
      contact_detail::FaceContribution<3>(el1, Tr, elfun, center_, radius_, *scale_ * penalty_, &elvect, nullptr);
    }
  }

  void AssembleFaceGrad(const mfem::FiniteElement &el1, const mfem::FiniteElement &,
                        mfem::FaceElementTransformations &Tr, const mfem::Vector &elfun,
                        mfem::DenseMatrix &elmat) override
  {
    if (Tr.GetSpaceDim() == 2)
    {
      contact_detail::FaceContribution<2>(el1, Tr, elfun, center_, radius_, *scale_ * penalty_, nullptr, &elmat);
    }
    else
    {
      contact_detail::FaceContribution<3>(el1, Tr, elfun, center_, radius_, *scale_ * penalty_, nullptr, &elmat);
    }
  }

private:
  mfem::VectorCoefficient &center_;
  double radius_, penalty_;
  const double *scale_;
};

// Mixed u-p formulation: the same term on the displacement block.
class BlockRigidSphereContactIntegrator : public mfem::BlockNonlinearFormIntegrator
{
public:
  BlockRigidSphereContactIntegrator(mfem::VectorCoefficient &center, double radius, double penalty,
                                    const double *scale)
    : center_(center), radius_(radius), penalty_(penalty), scale_(scale) {}

  void AssembleFaceVector(const mfem::Array<const mfem::FiniteElement *> &el1,
                          const mfem::Array<const mfem::FiniteElement *> &,
                          mfem::FaceElementTransformations &Tr,
                          const mfem::Array<const mfem::Vector *> &elfun,
                          const mfem::Array<mfem::Vector *> &elvect) override
  {
    if (Tr.GetSpaceDim() == 2)
    {
      contact_detail::FaceContribution<2>(*el1[0], Tr, *elfun[0], center_, radius_, *scale_ * penalty_, elvect[0], nullptr);
    }
    else
    {
      contact_detail::FaceContribution<3>(*el1[0], Tr, *elfun[0], center_, radius_, *scale_ * penalty_, elvect[0], nullptr);
    }
    elvect[1]->SetSize(0);
  }

  void AssembleFaceGrad(const mfem::Array<const mfem::FiniteElement *> &el1,
                        const mfem::Array<const mfem::FiniteElement *> &,
                        mfem::FaceElementTransformations &Tr,
                        const mfem::Array<const mfem::Vector *> &elfun,
                        const mfem::Array2D<mfem::DenseMatrix *> &elmats) override
  {
    if (Tr.GetSpaceDim() == 2)
    {
      contact_detail::FaceContribution<2>(*el1[0], Tr, *elfun[0], center_, radius_, *scale_ * penalty_, nullptr, elmats(0, 0));
    }
    else
    {
      contact_detail::FaceContribution<3>(*el1[0], Tr, *elfun[0], center_, radius_, *scale_ * penalty_, nullptr, elmats(0, 0));
    }
    elmats(0, 1)->SetSize(0);
    elmats(1, 0)->SetSize(0);
    elmats(1, 1)->SetSize(0);
  }

private:
  mfem::VectorCoefficient &center_;
  double radius_, penalty_;
  const double *scale_;
};

} // namespace cmf

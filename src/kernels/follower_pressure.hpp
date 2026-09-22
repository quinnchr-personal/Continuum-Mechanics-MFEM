// Follower pressure on a boundary of the reference mesh: a pressure p per
// unit current area acts along the current normal, which in the reference
// configuration is the traction T = -p J F^{-T} N (Nanson). It depends on the
// displacement, so it is a boundary term of the nonlinear form,
//   R_ext(u).w = int_Gamma s p (cof F . N) . w dA_R,   cof F = J F^{-T},
// (the sign convention R = internal - external of the solid physics) with
// tangent by dual numbers over the element displacement dofs. `scale` is the
// schedule value s(t), owned by the LoadSet of the physics.
#pragma once

#include <cmath>

#include "base/dual.hpp"
#include "base/tensor.hpp"
#include "kernels/total_lagrangian.hpp"
#include "mfem.hpp"

namespace cmf
{

namespace follower_detail
{

// Face residual and (optionally) tangent for one boundary face of el (the
// adjacent volume element). elfun: dof x dim, column-major (MFEM element
// ordering). Face quadrature of order 2 p + 3 on the face geometry.
// axisymmetric (dim = 2, x = r): F_33 = 1 + u_r / r at the face point and
// the weight 2 pi r (kernels/total_lagrangian.hpp, AxisymmetricPoint).
template <int dim>
void FaceContribution(const mfem::FiniteElement &el, mfem::FaceElementTransformations &Tr,
                      const mfem::Vector &elfun, mfem::Coefficient &p, double scale,
                      mfem::Vector *elvect, mfem::DenseMatrix *elmat, bool axisymmetric = false)
{
  const int dof = el.GetDof();
  mfem::DenseMatrix PMatI(const_cast<double *>(elfun.GetData()), dof, dim);
  mfem::DenseMatrix DSh(dof, dim), DS(dof, dim), Jrt(dim), Hmat(dim);
  mfem::Vector shape(dof), nor(dim), xf(dim), xc(dim);
  if (elvect) { elvect->SetSize(dof * dim); *elvect = 0.0; }
  if (elmat) { elmat->SetSize(dof * dim); *elmat = 0.0; }
  if (scale == 0.0) { return; }

  const mfem::IntegrationRule &ir =
    mfem::IntRules.Get(Tr.GetGeometryType(), 2 * el.GetOrder() + 3);
  mfem::ElementTransformation &T1 = Tr.GetElement1Transformation();
  // Outward orientation of the face normal, checked once against the vector
  // from the element centre to the face centre.
  double orientation = 1.0;
  {
    const mfem::IntegrationPoint &fc = mfem::Geometries.GetCenter(Tr.GetGeometryType());
    Tr.SetAllIntPoints(&fc);
    mfem::CalcOrtho(Tr.Jacobian(), nor);
    Tr.Transform(fc, xf);
    T1.Transform(mfem::Geometries.GetCenter(el.GetGeomType()), xc);
    double d = 0.0;
    for (int i = 0; i < dim; i++) { d += nor(i) * (xf(i) - xc(i)); }
    if (d < 0.0) { orientation = -1.0; }
  }

  for (int q = 0; q < ir.GetNPoints(); q++)
  {
    const mfem::IntegrationPoint &ip = ir.IntPoint(q);
    Tr.SetAllIntPoints(&ip);
    const mfem::IntegrationPoint &eip = Tr.GetElement1IntPoint();
    mfem::CalcOrtho(Tr.Jacobian(), nor); // reference normal times the area element
    el.CalcShape(eip, shape);
    el.CalcDShape(eip, DSh);
    mfem::CalcInverse(T1.Jacobian(), Jrt);
    mfem::Mult(DSh, Jrt, DS);
    mfem::MultAtB(PMatI, DS, Hmat);
    tensor<double, dim, dim> H;
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++) { H(i, j) = Hmat(i, j); }
    tensor<double, 3, 3> F = DeformationGradient<dim>(H);
    tensor<double, 3> N;
    for (int i = 0; i < 3; i++) { N(i) = i < dim ? orientation * nor(i) : 0.0; }
    double w = scale * ip.weight * p.Eval(Tr, ip);
    double r = 0.0, hoop_seed = 0.0; // dF_33 of the r-dof: N_b / r (DS(b, r) on the axis)
    if (axisymmetric)
    {
      Tr.Transform(ip, xf);
      r = xf(0);
      double u_r = 0.0;
      for (int a = 0; a < dof; a++) { u_r += shape(a) * PMatI(a, 0); }
      F(2, 2) = r > 0.0 ? 1.0 + u_r / r : 1.0 + H(0, 0);
      w *= 2.0 * M_PI * r;
    }

    if (elvect)
    {
      const tensor<double, 3> v = (det(F) * transpose(inv(F))) * N;
      for (int a = 0; a < dof; a++)
        for (int i = 0; i < dim; i++) { (*elvect)(a + i * dof) += w * shape(a) * v(i); }
    }
    if (elmat)
    {
      tensor<dual, 3, 3> Fd;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) { Fd(i, j) = dual(F(i, j), 0.0); }
      tensor<dual, 3> Nd;
      for (int i = 0; i < 3; i++) { Nd(i) = dual(N(i), 0.0); }
      for (int b = 0; b < dof; b++)
        for (int k = 0; k < dim; k++)
        {
          // Direction: du_k = phi_b, so dF_kj = DS(b, j) (and dF_33 for the r-dof).
          for (int j = 0; j < dim; j++) { Fd(k, j).d = DS(b, j); }
          if (axisymmetric && k == 0)
          {
            hoop_seed = r > 0.0 ? shape(b) / r : DS(b, 0);
            Fd(2, 2).d = hoop_seed;
          }
          const tensor<dual, 3> v = (det(Fd) * transpose(inv(Fd))) * Nd;
          for (int j = 0; j < dim; j++) { Fd(k, j).d = 0.0; }
          Fd(2, 2).d = 0.0;
          for (int a = 0; a < dof; a++)
            for (int i = 0; i < dim; i++)
            {
              (*elmat)(a + i * dof, b + k * dof) += w * shape(a) * v(i).d;
            }
        }
    }
  }
}

} // namespace follower_detail

// Displacement formulation: boundary face integrator of a ParNonlinearForm.
class FollowerPressureIntegrator : public mfem::NonlinearFormIntegrator
{
public:
  FollowerPressureIntegrator(mfem::Coefficient &p, const double *scale, bool axisymmetric = false)
    : p_(p), scale_(scale), axisymmetric_(axisymmetric) {}

  void AssembleFaceVector(const mfem::FiniteElement &el1, const mfem::FiniteElement &,
                          mfem::FaceElementTransformations &Tr, const mfem::Vector &elfun,
                          mfem::Vector &elvect) override
  {
    if (Tr.GetSpaceDim() == 2)
    {
      follower_detail::FaceContribution<2>(el1, Tr, elfun, p_, *scale_, &elvect, nullptr, axisymmetric_);
    }
    else
    {
      follower_detail::FaceContribution<3>(el1, Tr, elfun, p_, *scale_, &elvect, nullptr);
    }
  }

  void AssembleFaceGrad(const mfem::FiniteElement &el1, const mfem::FiniteElement &,
                        mfem::FaceElementTransformations &Tr, const mfem::Vector &elfun,
                        mfem::DenseMatrix &elmat) override
  {
    if (Tr.GetSpaceDim() == 2)
    {
      follower_detail::FaceContribution<2>(el1, Tr, elfun, p_, *scale_, nullptr, &elmat, axisymmetric_);
    }
    else
    {
      follower_detail::FaceContribution<3>(el1, Tr, elfun, p_, *scale_, nullptr, &elmat);
    }
  }

private:
  mfem::Coefficient &p_;
  const double *scale_;
  bool axisymmetric_;
};

// Mixed u-p formulation: the same term on the displacement block of a
// ParBlockNonlinearForm (pressure block untouched).
class BlockFollowerPressureIntegrator : public mfem::BlockNonlinearFormIntegrator
{
public:
  BlockFollowerPressureIntegrator(mfem::Coefficient &p, const double *scale, bool axisymmetric = false)
    : p_(p), scale_(scale), axisymmetric_(axisymmetric) {}

  void AssembleFaceVector(const mfem::Array<const mfem::FiniteElement *> &el1,
                          const mfem::Array<const mfem::FiniteElement *> &,
                          mfem::FaceElementTransformations &Tr,
                          const mfem::Array<const mfem::Vector *> &elfun,
                          const mfem::Array<mfem::Vector *> &elvect) override
  {
    if (Tr.GetSpaceDim() == 2)
    {
      follower_detail::FaceContribution<2>(*el1[0], Tr, *elfun[0], p_, *scale_, elvect[0], nullptr, axisymmetric_);
    }
    else
    {
      follower_detail::FaceContribution<3>(*el1[0], Tr, *elfun[0], p_, *scale_, elvect[0], nullptr);
    }
    elvect[1]->SetSize(0); // no pressure-block contribution
  }

  void AssembleFaceGrad(const mfem::Array<const mfem::FiniteElement *> &el1,
                        const mfem::Array<const mfem::FiniteElement *> &,
                        mfem::FaceElementTransformations &Tr,
                        const mfem::Array<const mfem::Vector *> &elfun,
                        const mfem::Array2D<mfem::DenseMatrix *> &elmats) override
  {
    if (Tr.GetSpaceDim() == 2)
    {
      follower_detail::FaceContribution<2>(*el1[0], Tr, *elfun[0], p_, *scale_, nullptr, elmats(0, 0), axisymmetric_);
    }
    else
    {
      follower_detail::FaceContribution<3>(*el1[0], Tr, *elfun[0], p_, *scale_, nullptr, elmats(0, 0));
    }
    // Blocks of height 0 are skipped by the block form.
    elmats(0, 1)->SetSize(0);
    elmats(1, 0)->SetSize(0);
    elmats(1, 1)->SetSize(0);
  }

private:
  mfem::Coefficient &p_;
  const double *scale_;
  bool axisymmetric_;
};

} // namespace cmf

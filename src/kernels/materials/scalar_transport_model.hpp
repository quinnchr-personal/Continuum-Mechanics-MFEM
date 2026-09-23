// Laws of the scalar transport physics (kernels/scalar_flux.hpp,
// physics/scalar_transport.hpp): the capacity c(u) of the rate term, the
// isotropic conductivity kappa(u) of the diffusive flux and the reaction
// coefficient s of the term s u, each a value type templated on the scalar
// type so that one dual seed of the kernel gives the tangent. The laws are
// affine in the unknown,
//   a(u) = value + slope (u - reference),
// which covers constant coefficients (slope 0) and the property laws of the
// nonlinear diffusion case of myapps/convection_diffusion; a problem whose
// capacity and conductivity are constant is linear in the unknown
// (ScalarTransportModel::Linear), whatever the velocity, source and data do
// with x and t. The velocity beta(x, t) and the source f(x, t) are
// coefficients of the kernel, not laws: they never enter the tangent.
#pragma once

#include "base/dual.hpp"

namespace cmf
{

struct AffineLaw
{
  double value = 1.0;
  double slope = 0.0;
  double reference = 0.0;

  template <typename T> T operator()(const T &u) const
  {
    return value + slope * (u - reference);
  }
  bool Constant() const { return slope == 0.0; }
};

// Convection in the weak form: beta . grad u as a source (non-conservative,
// the form of MFEM's ConvectionIntegrator and of the myapps drivers) or
// -beta u inside the flux (conservative, -div(beta u) in the strong form);
// the two coincide for a divergence-free velocity.
enum class ConvectionForm { NonConservative, Conservative };

struct ScalarTransportModel
{
  AffineLaw capacity;
  AffineLaw conductivity;
  double reaction = 0.0;
  ConvectionForm convection = ConvectionForm::NonConservative;

  template <typename T> T Capacity(const T &u) const { return capacity(u); }
  template <typename T> T Conductivity(const T &u) const { return conductivity(u); }
  bool Conservative() const { return convection == ConvectionForm::Conservative; }
  // The residual is affine in the unknown.
  bool Linear() const { return capacity.Constant() && conductivity.Constant(); }
};

} // namespace cmf

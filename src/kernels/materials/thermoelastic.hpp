// Finite thermoelasticity of elastomers: a decoupled hyperelastic model (the
// base class) with the temperature as a second state variable, the model of
// the finite thermoelasticity chapter of Anand's coupled theories and of its
// FEniCSx codes. Per unit reference volume
//   psi(F, theta) = s(theta) Psi_iso(Fbar) + kappa u(J / J_theta) - c_v [theta ln(theta/theta0) - (theta - theta0)],
//   s(theta) = theta / theta0 (entropic: the shear modulus is proportional to
//              the temperature, G = G0 theta/theta0 zeta) or 1,
//   J_theta  = exp(3 alpha (theta - theta0)), the stress-free thermal volume ratio,
// so that the isochoric stress is s(theta) P_iso of the base, the constitutive
// pressure U'(J, theta) = kappa u'(J_m) / J_theta with J_m = J / J_theta (for
// the logarithmic law exactly the reference's K (ln J - 3 alpha (theta -
// theta0)) / J), the mixed constraint is u'(J_m) / J_theta - p / kappa = 0
// with the tangent u''(J_m) / J_theta^2, and the heat capacity c_v (per unit
// reference volume) is constant. The thermal stress derivative
//   dP/dtheta = s'(theta) P_iso^base(F) + dU'/dtheta J F^-T,
//   dU'/dtheta = -3 alpha kappa [u''(J_m) J_m + u'(J_m)] / J_theta
//   (for the logarithmic law -3 alpha kappa / J, the reference's fac2),
// is an explicit function of (F, theta), so the thermal tangent
// M = F^-1 dP/dtheta of the heat equation and every derivative the kernel
// needs come from one templated evaluation. Fourier's law with the spatial
// conductivity k: the referential flux is Q = -k J C^-1 Grad theta
// (heat_flux.hpp, kernels/thermo_mixed_total_lagrangian.hpp).
#pragma once

#include <cmath>
#include <limits>

#include "base/dual.hpp"
#include "base/tensor.hpp"
#include "materials/volumetric.hpp"

namespace cmf
{

struct ThermalParameters
{
  double theta0 = 298.0;   // reference (initial) temperature
  double alpha = 0.0;      // linear coefficient of thermal expansion
  double c_v = 1.0;        // heat capacity per unit reference volume
  double k = 1.0;          // spatial thermal conductivity (isotropic)
  bool entropic = true;    // shear modulus proportional to theta
};

template <typename Base>
struct Thermoelastic : Base
{
  using Equilibrium = Base;
  ThermalParameters thermal;

  Thermoelastic() = default;
  Thermoelastic(const Base &base, const ThermalParameters &thermal_) : Base(base), thermal(thermal_) {}

  // s(theta) and s'(theta).
  template <typename T>
  T ModulusScale(const T &theta) const { return thermal.entropic ? theta / thermal.theta0 : T(1.0); }
  double ModulusScaleDerivative() const { return thermal.entropic ? 1.0 / thermal.theta0 : 0.0; }
  // J_theta(theta).
  template <typename T>
  T ThermalVolumeRatio(const T &theta) const { return exp(3.0 * thermal.alpha * (theta - thermal.theta0)); }

  template <typename TF, typename TT>
  tensor<product_t<TT, TF>, 3, 3> PK1Iso(const tensor<TF, 3, 3> &F, const TT &theta) const
  {
    return ModulusScale(theta) * Base::PK1Iso(F);
  }
  template <typename TF, typename TT>
  product_t<TT, TF> EnergyIso(const tensor<TF, 3, 3> &F, const TT &theta) const
  {
    return ModulusScale(theta) * Base::EnergyIso(F);
  }

  // u'(J_m) / J_theta: the constitutive pressure per unit bulk modulus, and
  // its derivatives in J and theta.
  template <typename TJ, typename TT>
  product_t<TT, TJ> NormalizedVolumetricPressure(const TJ &J, const TT &theta) const
  {
    const TT Jt = ThermalVolumeRatio(theta);
    return cmf::NormalizedVolumetricPressure(Base::law, J / Jt) / Jt;
  }
  template <typename TJ, typename TT>
  product_t<TT, TJ> NormalizedVolumetricModulus(const TJ &J, const TT &theta) const
  {
    const TT Jt = ThermalVolumeRatio(theta);
    return cmf::NormalizedVolumetricModulus(Base::law, J / Jt) / (Jt * Jt);
  }
  // d(u'(J_m)/J_theta)/dtheta = -3 alpha [u''(J_m) J_m + u'(J_m)] / J_theta.
  template <typename TJ, typename TT>
  product_t<TT, TJ> NormalizedVolumetricPressureThermalDerivative(const TJ &J, const TT &theta) const
  {
    const TT Jt = ThermalVolumeRatio(theta);
    const product_t<TT, TJ> Jm = J / Jt;
    return (-3.0 * thermal.alpha) *
           (cmf::NormalizedVolumetricModulus(Base::law, Jm) * Jm + cmf::NormalizedVolumetricPressure(Base::law, Jm)) / Jt;
  }
  template <typename TJ, typename TT>
  product_t<TT, TJ> VolumetricPressure(const TJ &J, const TT &theta) const
  {
    return Base::kappa * NormalizedVolumetricPressure(J, theta);
  }
  template <typename TJ, typename TT>
  product_t<TT, TJ> VolumetricEnergy(const TJ &J, const TT &theta) const
  {
    return Base::kappa * cmf::NormalizedVolumetricEnergy(Base::law, J / ThermalVolumeRatio(theta));
  }

  // The displacement-form stress with the constitutive pressure (finite kappa).
  template <typename TF, typename TT>
  tensor<product_t<TT, TF>, 3, 3> PK1(const tensor<TF, 3, 3> &F, const TT &theta) const
  {
    const TF J = det(F);
    return PK1Iso(F, theta) + (VolumetricPressure(J, theta) * J) * transpose(inv(F));
  }
  // dP/dtheta of the displacement-form stress (independent of the field pressure).
  template <typename TF, typename TT>
  tensor<product_t<TT, TF>, 3, 3> ThermalStressDerivative(const tensor<TF, 3, 3> &F, const TT &theta) const
  {
    const TF J = det(F);
    tensor<product_t<TT, TF>, 3, 3> D = ModulusScaleDerivative() * Base::PK1Iso(F);
    if (Base::Incompressible()) { return D; }
    return D + (Base::kappa * NormalizedVolumetricPressureThermalDerivative(J, theta) * J) * transpose(inv(F));
  }
  // The thermal tangent of the heat equation, M = dS/dtheta = F^-1 dP/dtheta.
  template <typename TF, typename TT>
  tensor<product_t<TT, TF>, 3, 3> ThermalTangent(const tensor<TF, 3, 3> &F, const TT &theta) const
  {
    return inv(F) * ThermalStressDerivative(F, theta);
  }
  // Mechanical part of the free energy (the displacement form; the thermal
  // term -c_v [theta ln(theta/theta0) - (theta - theta0)] is left out of
  // the energy diagnostic).
  template <typename TF, typename TT>
  product_t<TT, TF> Energy(const tensor<TF, 3, 3> &F, const TT &theta) const
  {
    return EnergyIso(F, theta) + VolumetricEnergy(det(F), theta);
  }
  // Referential heat flux Q = -k J C^-1 Grad theta.
  template <typename TF, typename TG>
  tensor<product_t<TG, TF>, 3> HeatFlux(const tensor<TF, 3, 3> &F, const tensor<TG, 3> &grad_theta) const
  {
    const tensor<TF, 3, 3> Cinv = inv(transpose(F) * F);
    return (-thermal.k * det(F)) * (Cinv * grad_theta);
  }
};

template <typename M> struct is_thermoelastic : std::false_type {};
template <typename B> struct is_thermoelastic<Thermoelastic<B>> : std::true_type {};

} // namespace cmf

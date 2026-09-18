// Decoupled Gent material (limited chain extensibility).
//   Psi_iso = -mu Jm/2 ln(1 - (I1bar - 3)/Jm),   I1bar - 3 < Jm
//   U(J)    = kappa/2 (J - 1)^2                  (kappa = inf: incompressible;
//             other laws selectable, see volumetric.hpp)
//   P_iso   = dPsi_iso/dI1bar * dI1bar/dF        (see isochoric.hpp)
// Small-strain shear modulus mu; Jm -> inf recovers the neo-Hookean model.
// Beyond the locking stretch the energy is NaN, which makes the Newton line
// search reject the step.
#pragma once

#include <cmath>
#include <limits>

#include "base/dual.hpp"
#include "base/tensor.hpp"
#include "materials/volumetric.hpp"
#include "materials/isochoric.hpp"

namespace cmf
{

struct Gent
{
  double mu = 1.0;
  double Jm = 10.0;
  double kappa = std::numeric_limits<double>::infinity();
  VolumetricLaw law = VolumetricLaw::Quadratic;

  Gent() = default;
  Gent(double mu_, double Jm_, double kappa_) : mu(mu_), Jm(Jm_), kappa(kappa_) {}

  bool Incompressible() const { return !std::isfinite(kappa); }
  double ShearModulus() const { return mu; }

  // dPsi_iso/dI1bar = mu/2 Jm / (Jm - (I1bar - 3)).
  template <typename T>
  T DPsiDI1(const T &I1bar) const
  {
    return (0.5 * mu * Jm) / (Jm - (I1bar - 3.0));
  }

  template <typename T>
  tensor<T, 3, 3> PK1Iso(const tensor<T, 3, 3> &F) const
  {
    return DPsiDI1(FirstModifiedInvariant(F)) * FirstModifiedInvariantGradient(F);
  }

  template <typename T>
  T EnergyIso(const tensor<T, 3, 3> &F) const
  {
    const T x = FirstModifiedInvariant(F) - 3.0;
    return (-0.5 * mu * Jm) * log(1.0 - x / Jm);
  }

  // Volumetric law U(J) = kappa u(J) (materials/volumetric.hpp): U'(J) and
  // U(J) for the displacement formulation, u'(J) and u''(J) for the mixed
  // constraint u'(J) - p / kappa = 0 and its tangent.
  template <typename T>
  T VolumetricPressure(const T &J) const { return kappa * cmf::NormalizedVolumetricPressure(law, J); }

  template <typename T>
  T VolumetricEnergy(const T &J) const { return kappa * cmf::NormalizedVolumetricEnergy(law, J); }

  template <typename T>
  T NormalizedVolumetricPressure(const T &J) const { return cmf::NormalizedVolumetricPressure(law, J); }

  template <typename T>
  T NormalizedVolumetricModulus(const T &J) const { return cmf::NormalizedVolumetricModulus(law, J); }

  // kappa u*(p / kappa), the volumetric energy as a function of the pressure
  // (p^2 / (2 kappa) for the quadratic law); finite kappa only.
  double ComplementaryVolumetricEnergy(double p) const
  {
    return kappa * cmf::NormalizedComplementaryEnergy(law, p / kappa);
  }

  template <typename T>
  tensor<T, 3, 3> PK1(const tensor<T, 3, 3> &F) const
  {
    const T J = det(F);
    return PK1Iso(F) + (VolumetricPressure(J) * J) * transpose(inv(F));
  }

  template <typename T>
  T Energy(const tensor<T, 3, 3> &F) const
  {
    return EnergyIso(F) + VolumetricEnergy(det(F));
  }
};

} // namespace cmf

// Decoupled Yeoh material, cubic in the first modified invariant.
//   Psi_iso = c10 (I1bar - 3) + c20 (I1bar - 3)^2 + c30 (I1bar - 3)^3
//   U(J)    = kappa/2 (J - 1)^2         (kappa = inf: incompressible;
//             other laws selectable, see volumetric.hpp)
//   P_iso   = dPsi_iso/dI1bar * dI1bar/dF   (see isochoric.hpp)
// Small-strain shear modulus mu = 2 c10.
#pragma once

#include <cmath>
#include <limits>

#include "base/dual.hpp"
#include "base/tensor.hpp"
#include "materials/volumetric.hpp"
#include "materials/isochoric.hpp"

namespace cmf
{

struct Yeoh
{
  double c10 = 0.5;
  double c20 = 0.0;
  double c30 = 0.0;
  double kappa = std::numeric_limits<double>::infinity();
  VolumetricLaw law = VolumetricLaw::Quadratic;

  Yeoh() = default;
  Yeoh(double c10_, double c20_, double c30_, double kappa_)
    : c10(c10_), c20(c20_), c30(c30_), kappa(kappa_) {}

  bool Incompressible() const { return !std::isfinite(kappa); }
  double ShearModulus() const { return 2.0 * c10; }

  // dPsi_iso/dI1bar.
  template <typename T>
  T DPsiDI1(const T &I1bar) const
  {
    const T x = I1bar - 3.0;
    return c10 + (2.0 * c20) * x + (3.0 * c30) * (x * x);
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
    return c10 * x + c20 * (x * x) + c30 * (x * x * x);
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

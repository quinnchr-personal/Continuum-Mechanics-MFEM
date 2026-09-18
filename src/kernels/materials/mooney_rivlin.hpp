// Decoupled Mooney-Rivlin material.
//   Psi_iso = c1 (I1bar - 3) + c2 (I2bar - 3),
//   I1bar = J^{-2/3} I1,  I2bar = J^{-4/3} I2,  I2 = (I1^2 - C:C)/2,  C = F^T F
//   U(J) = kappa/2 (J - 1)^2 (kappa = inf: incompressible;
//             other laws selectable, see volumetric.hpp)
//   P_iso = c1 J^{-2/3} (2F - 2/3 I1 F^{-T})
//         + c2 J^{-4/3} (2 I1 F - 2 F C - 4/3 I2 F^{-T})
// Small-strain shear modulus mu = 2 (c1 + c2).
#pragma once

#include <cmath>
#include <limits>

#include "base/dual.hpp"
#include "base/tensor.hpp"
#include "materials/volumetric.hpp"

namespace cmf
{

struct MooneyRivlin
{
  double c1 = 0.5;
  double c2 = 0.0;
  double kappa = std::numeric_limits<double>::infinity();
  VolumetricLaw law = VolumetricLaw::Quadratic;

  MooneyRivlin() = default;
  MooneyRivlin(double c1_, double c2_, double kappa_) : c1(c1_), c2(c2_), kappa(kappa_) {}

  bool Incompressible() const { return !std::isfinite(kappa); }
  double ShearModulus() const { return 2.0 * (c1 + c2); }

  template <typename T>
  tensor<T, 3, 3> PK1Iso(const tensor<T, 3, 3> &F) const
  {
    const T J = det(F);
    const tensor<T, 3, 3> C = transpose(F) * F;
    const T I1 = tr(C);
    const T I2 = 0.5 * (I1 * I1 - ddot(C, C));
    const T Jm23 = pow(J, -2.0 / 3.0);
    const T Jm43 = Jm23 * Jm23;
    const tensor<T, 3, 3> FinvT = transpose(inv(F));
    return (c1 * Jm23) * (2.0 * F - (2.0 / 3.0 * I1) * FinvT)
           + (c2 * Jm43) * ((2.0 * I1) * F - 2.0 * (F * C) - (4.0 / 3.0 * I2) * FinvT);
  }

  template <typename T>
  T EnergyIso(const tensor<T, 3, 3> &F) const
  {
    const T J = det(F);
    const tensor<T, 3, 3> C = transpose(F) * F;
    const T I1 = tr(C);
    const T I2 = 0.5 * (I1 * I1 - ddot(C, C));
    const T Jm23 = pow(J, -2.0 / 3.0);
    return c1 * (Jm23 * I1 - 3.0) + c2 * (Jm23 * Jm23 * I2 - 3.0);
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

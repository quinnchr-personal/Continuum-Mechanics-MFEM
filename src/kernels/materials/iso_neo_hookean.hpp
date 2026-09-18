// Decoupled (isochoric-volumetric) neo-Hookean material for near- and fully
// incompressible problems.
//   Psi_iso(F) = mu/2 (J^{-2/3} I1 - 3),   I1 = tr(F^T F)
//   U(J)       = kappa/2 (J - 1)^2         (kappa = inf: incompressible;
//             other laws selectable, see volumetric.hpp)
//   P_iso      = mu J^{-2/3} (F - I1/3 F^{-T})
// Displacement formulation (finite kappa): P = P_iso + U'(J) J F^{-T}.
// Mixed formulation: P = P_iso + p J F^{-T} with p an independent field.
#pragma once

#include <cmath>
#include <limits>

#include "base/dual.hpp"
#include "base/tensor.hpp"
#include "materials/volumetric.hpp"

namespace cmf
{

struct IsoNeoHookean
{
  double mu = 1.0;
  double kappa = std::numeric_limits<double>::infinity();
  VolumetricLaw law = VolumetricLaw::Quadratic;

  IsoNeoHookean() = default;
  IsoNeoHookean(double mu_, double kappa_) : mu(mu_), kappa(kappa_) {}

  bool Incompressible() const { return !std::isfinite(kappa); }
  double ShearModulus() const { return mu; }

  template <typename T>
  tensor<T, 3, 3> PK1Iso(const tensor<T, 3, 3> &F) const
  {
    const T J = det(F);
    const T I1 = ddot(F, F);
    const T Jm23 = pow(J, -2.0 / 3.0);
    const tensor<T, 3, 3> FinvT = transpose(inv(F));
    return (mu * Jm23) * (F - (I1 / 3.0) * FinvT);
  }

  template <typename T>
  T EnergyIso(const tensor<T, 3, 3> &F) const
  {
    const T J = det(F);
    return 0.5 * mu * (pow(J, -2.0 / 3.0) * ddot(F, F) - 3.0);
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

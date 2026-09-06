// Decoupled (isochoric-volumetric) neo-Hookean material for near- and fully
// incompressible problems.
//   Psi_iso(F) = mu/2 (J^{-2/3} I1 - 3),   I1 = tr(F^T F)
//   U(J)       = kappa/2 (J - 1)^2         (kappa = inf: incompressible)
//   P_iso      = mu J^{-2/3} (F - I1/3 F^{-T})
// Displacement formulation (finite kappa): P = P_iso + U'(J) J F^{-T}.
// Mixed formulation: P = P_iso + p J F^{-T} with p an independent field.
#pragma once

#include <cmath>
#include <limits>

#include "base/dual.hpp"
#include "base/tensor.hpp"

namespace cmf
{

struct IsoNeoHookean
{
  double mu = 1.0;
  double kappa = std::numeric_limits<double>::infinity();

  IsoNeoHookean() = default;
  IsoNeoHookean(double mu_, double kappa_) : mu(mu_), kappa(kappa_) {}

  bool Incompressible() const { return !std::isfinite(kappa); }

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

  // dU/dJ, the volumetric (penalty) pressure; finite kappa only.
  template <typename T>
  T VolumetricPressure(const T &J) const { return kappa * (J - 1.0); }

  template <typename T>
  T VolumetricEnergy(const T &J) const { return 0.5 * kappa * (J - 1.0) * (J - 1.0); }

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

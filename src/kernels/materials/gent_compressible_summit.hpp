// Compressible Gent hyperelasticity in the form of SUMMIT's gent-compressible
// material (summit::GentCompressibleHyperelastic): a coupled model, the Gent
// chain term on the full I1 = tr C (not the isochoric I1bar of gent.hpp) with
// a quartic volumetric penalty,
//   W(F) = -mu/2 (Jm ln(1 - (I1 - 3)/Jm) + 2 ln J) + kappa/2 A^4,
//   A    = (J^2 - 1)/2 - ln J,                       I1 - 3 < Jm
//   P(F) = mu Jm/(Jm - I1 + 3) F + (2 kappa A^3 (J^2 - 1) - mu) F^{-T}.
// Jm -> inf gives mu/2 (I1 - 3) - mu ln J + kappa/2 A^4. A = (J - 1)^2 + O((J - 1)^3),
// so the penalty is O((J - 1)^8) and kappa does not enter the small-strain
// response: the linearised moduli are mu and lambda = 2 mu / Jm (from the Gent
// term), nu = 1 / (Jm + 2), whatever kappa is; kappa stiffens the volume change
// at finite strain only. (SUMMIT reports E and nu from kappa and mu as if
// kappa were the small-strain bulk modulus; it is not.) Displacement
// formulation only. Beyond the locking stretch the energy is NaN, which makes
// the Newton line search reject the step. Stateless value type; PK1 is
// templated so dual numbers give the tangent.
#pragma once

#include "base/dual.hpp"
#include "base/tensor.hpp"

namespace cmf
{

struct GentCompressibleSummit
{
  double mu = 1.0;
  double kappa = 1.0;
  double Jm = 10.0;

  // Small-strain Lame modulus, 2 mu / Jm (kappa does not contribute).
  double SmallStrainLambda() const { return 2.0 * mu / Jm; }

  template <typename T>
  tensor<T, 3, 3> PK1(const tensor<T, 3, 3> &F) const
  {
    const tensor<T, 3, 3> FinvT = transpose(inv(F));
    const T J = det(F);
    const T I1 = ddot(F, F); // tr(F^T F)
    const T A = 0.5 * (J * J - 1.0) - log(J);
    const T chain = mu * Jm / (Jm - I1 + 3.0);
    return chain * F + (2.0 * kappa * A * A * A * (J * J - 1.0) - mu) * FinvT;
  }

  template <typename T>
  T Energy(const tensor<T, 3, 3> &F) const
  {
    const T J = det(F);
    const T lnJ = log(J);
    const T I1 = ddot(F, F);
    const T A = 0.5 * (J * J - 1.0) - lnJ;
    const T A2 = A * A;
    return -0.5 * mu * (Jm * log(1.0 - (I1 - 3.0) / Jm) + 2.0 * lnJ) + 0.5 * kappa * A2 * A2;
  }
};

} // namespace cmf

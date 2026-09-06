// Shared pieces of the isochoric-volumetric split for models whose isochoric
// energy is a function of the first modified invariant alone,
//   Psi_iso = f(I1bar),   I1bar = J^{-2/3} tr(F^T F),
//   P_iso   = f'(I1bar) dI1bar/dF = 2 f'(I1bar) J^{-2/3} (F - I1/3 F^{-T}).
// Templated on the scalar type so dual numbers give the tangent.
#pragma once

#include "base/dual.hpp"
#include "base/tensor.hpp"

namespace cmf
{

template <typename T>
inline T FirstModifiedInvariant(const tensor<T, 3, 3> &F)
{
  return pow(det(F), -2.0 / 3.0) * ddot(F, F);
}

// dI1bar/dF = J^{-2/3} (2 F - 2/3 I1 F^{-T}).
template <typename T>
inline tensor<T, 3, 3> FirstModifiedInvariantGradient(const tensor<T, 3, 3> &F)
{
  const T J = det(F);
  const T I1 = ddot(F, F);
  const T Jm23 = pow(J, -2.0 / 3.0);
  return Jm23 * (2.0 * F - (2.0 / 3.0 * I1) * transpose(inv(F)));
}

} // namespace cmf

// The kinematics a material is written for, as a compile-time trait.
//   finite strain (default): PK1(F) is the first Piola-Kirchhoff stress of the
//     deformation gradient, sigma = J^{-1} P F^T and the volume ratio is
//     J = det F;
//   small strain (the material declares `static constexpr bool small_strain =
//     true`): PK1(F) reads eps = sym(F - I) and returns the symmetric stress of
//     the geometrically linear theory, so sigma = P and the volume ratio is
//     1 + tr(eps). Reference and current configuration coincide there: a
//     follower load is the dead load, and moment arms are reference positions.
// The flux contract of the kernels is the same for both; everything outside
// the flux that depends on the kinematics goes through the helpers below.
#pragma once

#include <type_traits>

#include "base/tensor.hpp"

namespace cmf
{

template <typename M, typename = void>
struct is_small_strain : std::false_type {};
template <typename M>
struct is_small_strain<M, std::enable_if_t<M::small_strain>> : std::true_type {};

// Cauchy stress from F and P = PK1(F) of this material.
template <typename Material>
inline tensor<double, 3, 3> CauchyStress(const Material &, const tensor<double, 3, 3> &F,
                                         const tensor<double, 3, 3> &P)
{
  if constexpr (is_small_strain<Material>::value) { return P; }
  else { return (1.0 / det(F)) * (P * transpose(F)); }
}

// Current volume per unit reference volume.
template <typename Material>
inline double VolumeRatio(const Material &, const tensor<double, 3, 3> &F)
{
  if constexpr (is_small_strain<Material>::value)
  {
    return 1.0 + ((F(0, 0) - 1.0) + (F(1, 1) - 1.0) + (F(2, 2) - 1.0));
  }
  else { return det(F); }
}

// The strain of the material's kinematics: the infinitesimal strain
// eps = sym(F - I), or the Green-Lagrange strain E = (F^T F - I) / 2, which
// tends to it.
template <typename Material>
inline tensor<double, 3, 3> Strain(const Material &, const tensor<double, 3, 3> &F)
{
  if constexpr (is_small_strain<Material>::value) { return sym(F - I<3>()); }
  else { return 0.5 * (transpose(F) * F - I<3>()); }
}

} // namespace cmf

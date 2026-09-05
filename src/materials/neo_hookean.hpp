// Compressible neo-Hookean hyperelasticity (Simo-Hughes / Bonet-Wood form).
//   W(F) = mu/2 (tr C - 3) - mu ln J + lambda/2 (ln J)^2
//   P(F) = mu (F - F^{-T}) + lambda ln(J) F^{-T}
// Stateless value type; PK1 is templated so dual numbers give the tangent.
#pragma once

#include "base/dual.hpp"
#include "base/tensor.hpp"

namespace cmf
{

struct NeoHookean
{
  double mu = 1.0;
  double lambda = 1.0;

  template <typename T>
  tensor<T, 3, 3> PK1(const tensor<T, 3, 3> &F) const
  {
    const tensor<T, 3, 3> FinvT = transpose(inv(F));
    const T J = det(F);
    return mu * (F - FinvT) + (lambda * log(J)) * FinvT;
  }

  template <typename T>
  T Energy(const tensor<T, 3, 3> &F) const
  {
    const T J = det(F);
    const T lnJ = log(J);
    const T I1 = ddot(F, F); // tr(F^T F)
    return 0.5 * mu * (I1 - 3.0) - mu * lnJ + 0.5 * lambda * lnJ * lnJ;
  }
};

} // namespace cmf

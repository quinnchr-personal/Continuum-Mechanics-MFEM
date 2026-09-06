// Saint Venant-Kirchhoff hyperelasticity.
//   E = (F^T F - I)/2,  S = lambda tr(E) I + 2 mu E,  P = F S
//   W = lambda/2 (tr E)^2 + mu E:E
#pragma once

#include "base/dual.hpp"
#include "base/tensor.hpp"

namespace cmf
{

struct StVenantKirchhoff
{
  double mu = 1.0;
  double lambda = 1.0;

  template <typename T>
  tensor<T, 3, 3> PK1(const tensor<T, 3, 3> &F) const
  {
    const tensor<T, 3, 3> E = 0.5 * (transpose(F) * F - I<3>());
    const tensor<T, 3, 3> S = (lambda * tr(E)) * I<3>() + (2.0 * mu) * E;
    return F * S;
  }

  template <typename T>
  T Energy(const tensor<T, 3, 3> &F) const
  {
    const tensor<T, 3, 3> E = 0.5 * (transpose(F) * F - I<3>());
    const T trE = tr(E);
    return 0.5 * lambda * trE * trE + mu * ddot(E, E);
  }
};

} // namespace cmf

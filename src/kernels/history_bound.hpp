// A history-dependent material (one with HistorySize(), whose stresses take
// the accepted history block and the step length) bound to the history of
// one quadrature point and to the length of the step under way, so that it
// meets the stateless contract the kernels consume: PK1(F), Energy(F),
// PK1Iso(F), EnergyIso(F). Everything that does not depend on the history
// (the volumetric law, kappa, Incompressible(), ShearModulus()) is read
// from the material itself by the kernels. AtPoint gives a kernel either
// the bound view or the material itself, whichever the material needs.
#pragma once

#include <type_traits>
#include <utility>

#include "base/tensor.hpp"
#include "kernels/history_field.hpp"

namespace cmf
{

template <typename M, typename = void>
struct has_history : std::false_type {};
template <typename M>
struct has_history<M, std::void_t<decltype(std::declval<const M &>().HistorySize())>> : std::true_type {};

template <typename M>
struct HistoryBound
{
  const M &material;
  const double *history;
  double dt;

  template <typename T>
  tensor<T, 3, 3> PK1(const tensor<T, 3, 3> &F) const { return material.PK1(F, history, dt); }
  template <typename T>
  T Energy(const tensor<T, 3, 3> &F) const { return material.Energy(F, history, dt); }
  template <typename T>
  tensor<T, 3, 3> PK1Iso(const tensor<T, 3, 3> &F) const { return material.PK1Iso(F, history, dt); }
  template <typename T>
  T EnergyIso(const tensor<T, 3, 3> &F) const { return material.EnergyIso(F, history, dt); }
};

// The material as a kernel evaluates it at point q of element elem: bound to
// its history there (history must then be set), or the material itself.
template <typename M>
inline decltype(auto) AtPoint(const M &material, const HistoryField *history, int elem, int q)
{
  if constexpr (has_history<M>::value)
  {
    return HistoryBound<M>{material, history->Old(elem, q), history->Dt()};
  }
  else { return (material); }
}

// The type AtPoint returns for M, decayed.
template <typename M>
using bound_t = std::conditional_t<has_history<M>::value, HistoryBound<M>, M>;

} // namespace cmf

// Plane-stress adapter for the 2D displacement formulation. The kernels pad
// the in-plane deformation gradient with F33 = 1 (plane strain); this adapter
// replaces F33 by the thickness stretch lambda3 at which the out-of-plane
// stress vanishes, P33 = 0 (equivalently sigma33 = 0), and returns P at that
// state. Its (3,3) entry is zero, so the kernels' in-plane block is the
// plane-stress response.
//   incompressible base (kappa = inf): lambda3 = 1 / det F2D and the mean
//     stress p = -sigma_iso,33 replaces the Lagrange multiplier,
//     P = P_iso + p F^{-T}; no pressure unknown is needed (small strain:
//     eps_33 = -(eps_11 + eps_22), p = -P_iso,33, P = P_iso + p I);
//   compressible base: lambda3 solves P33(lambda3) = 0 by a scalar Newton
//     iteration. With dual numbers the converged root is refined by one Newton
//     step in dual arithmetic, which carries the implicit derivative
//     d lambda3 / dF, so MaterialTangent yields the consistent tangent.
// Energy(F) is the base energy at the completed F; since P33 = 0 there, its
// in-plane derivative is the in-plane P (envelope theorem).
#pragma once

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#include "base/dual.hpp"
#include "base/tensor.hpp"
#include "materials/kinematics.hpp"

namespace cmf
{

template <typename M, typename = void>
struct has_incompressible : std::false_type {};
template <typename M>
struct has_incompressible<M, std::void_t<decltype(std::declval<const M &>().Incompressible())>>
  : std::true_type {};

template <typename Material>
struct PlaneStress
{
  using Base = Material;
  // The adapter has the kinematics of its base (materials/kinematics.hpp).
  static constexpr bool small_strain = is_small_strain<Material>::value;

  Material material;

  PlaneStress() = default;
  explicit PlaneStress(const Material &m) : material(m) {}

  bool Incompressible() const
  {
    if constexpr (has_incompressible<Material>::value) { return material.Incompressible(); }
    else { return false; }
  }

  // The thickness stretch lambda3 = F33 of the plane-stress state of the
  // in-plane block of F.
  template <typename T>
  T Thickness(const tensor<T, 3, 3> &F) const
  {
    if (Incompressible())
    {
      if constexpr (small_strain) { return 1.0 - ((F(0, 0) - 1.0) + (F(1, 1) - 1.0)); }
      else { return 1.0 / (F(0, 0) * F(1, 1) - F(0, 1) * F(1, 0)); }
    }
    tensor<double, 3, 3> Fv;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) { Fv(i, j) = value(F(i, j)); }
    double dr = 0.0;
    const double l3 = SolveThickness(InPlane(Fv), dr);
    if constexpr (std::is_same_v<T, double>) { return l3; }
    else
    {
      // One Newton step in dual arithmetic at the root: the value is kept, the
      // derivative becomes -(dP33/dF : dF) / (dP33/dF33).
      tensor<T, 3, 3> Fc = InPlane(F);
      Fc(2, 2) = T(l3);
      const T r = material.PK1(Fc)(2, 2);
      return T(l3) - (r - value(r)) / dr;
    }
  }

  // F with the out-of-plane shears removed and F33 = lambda3.
  template <typename T>
  tensor<T, 3, 3> Complete(const tensor<T, 3, 3> &F) const
  {
    tensor<T, 3, 3> Fc = InPlane(F);
    Fc(2, 2) = Thickness(F);
    return Fc;
  }

  template <typename T>
  tensor<T, 3, 3> PK1(const tensor<T, 3, 3> &F) const
  {
    const tensor<T, 3, 3> Fc = Complete(F);
    if constexpr (has_incompressible<Material>::value)
    {
      if (material.Incompressible())
      {
        // J = 1: P = P_iso + p F^{-T} with p = -sigma_iso,33 = -P_iso,33 lambda3
        // (small strain, tr(eps) = 0: P = P_iso + p I with p = -P_iso,33).
        const tensor<T, 3, 3> Piso = material.PK1Iso(Fc);
        if constexpr (small_strain) { return Piso + (-Piso(2, 2)) * I<3>(); }
        else
        {
          const T p = -Piso(2, 2) * Fc(2, 2);
          return Piso + p * transpose(inv(Fc));
        }
      }
    }
    return material.PK1(Fc);
  }

  template <typename T>
  T Energy(const tensor<T, 3, 3> &F) const
  {
    const tensor<T, 3, 3> Fc = Complete(F);
    if constexpr (has_incompressible<Material>::value)
    {
      if (material.Incompressible()) { return material.EnergyIso(Fc); }
    }
    return material.Energy(Fc);
  }

private:
  template <typename T>
  static tensor<T, 3, 3> InPlane(const tensor<T, 3, 3> &F)
  {
    tensor<T, 3, 3> Fc;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) { Fc(i, j) = F(i, j); }
    Fc(2, 2) = T(1.0);
    return Fc;
  }

  // Newton on r(l3) = P33(F2D, l3) = 0 from l3 = 1, steps limited to
  // [-l3/2, l3] to keep the stretch positive; returns l3 and dr/dl3 at the
  // root. A non-finite state (inverted in-plane block during a line search)
  // returns NaN, which the damped Newton solver rejects like any other
  // non-finite residual.
  double SolveThickness(const tensor<double, 3, 3> &F, double &dr) const
  {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    tensor<dual, 3, 3> Fd;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) { Fd(i, j) = dual(F(i, j), 0.0); }
    double l3 = 1.0;
    for (int it = 0; it < 100; it++)
    {
      Fd(2, 2) = dual(l3, 1.0);
      const dual r = material.PK1(Fd)(2, 2);
      dr = r.d;
      if (!std::isfinite(r.v) || !std::isfinite(dr) || dr == 0.0) { dr = nan; return nan; }
      double step = -r.v / dr;
      if (step < -0.5 * l3) { step = -0.5 * l3; }
      if (step > l3) { step = l3; }
      l3 += step;
      if (std::abs(step) <= 1e-14 * l3)
      {
        Fd(2, 2) = dual(l3, 1.0);
        dr = material.PK1(Fd)(2, 2).d;
        return l3;
      }
    }
    dr = nan;
    return nan;
  }
};

} // namespace cmf

// Isotropic linear elasticity at small strain (the geometrically linear theory).
//   eps   = sym(F - I) = sym(Grad u)
//   sigma = 2 mu dev(eps) + kappa tr(eps) I
//         = lambda tr(eps) I + 2 mu eps,        lambda = kappa - 2 mu / 3
//   W     = mu dev(eps):dev(eps) + kappa/2 tr(eps)^2
// The kernels' flux contract is PK1(F). At small strain the stress measures
// coincide, so PK1 returns sigma; it is symmetric, hence sigma : Grad w =
// sigma : eps(w), and the total Lagrangian kernel assembles the classical
// B^T C B stiffness. The tangent dP/dF = C is constant, so the first Newton
// step is the exact linear solve. The model is not objective under finite
// rotations by construction; it is invariant under infinitesimal ones,
// PK1(I + W) = 0 for skew W.
// Forming F = I + H and subtracting I again leaves an absolute error of
// 1e-16 in eps, a relative floor of 1e-16 / |Grad u| in the stress: loads are
// chosen with |Grad u| >= 1e-4 and results scaled, never the other way round.
// Mixed u-p formulation (Herrmann), kappa possibly infinite: the volume
// measure of the small-strain kinematics is theta = 1 + tr(eps) with
// d theta / dF = I (mixed_total_lagrangian.hpp), so
//   sigma = 2 mu dev(eps) + p I,   tr(eps) - p / kappa = 0   (weakly),
// i.e. PK1Iso = 2 mu dev(eps) and the quadratic volumetric law u(theta) =
// (theta - 1)^2 / 2 by construction (there is no law to choose).
#pragma once

#include <cmath>

#include "base/dual.hpp"
#include "base/tensor.hpp"

namespace cmf
{

struct LinearElastic
{
  static constexpr bool small_strain = true; // materials/kinematics.hpp

  double mu = 1.0;
  double kappa = 1.0;

  LinearElastic() = default;
  LinearElastic(double mu_, double kappa_) : mu(mu_), kappa(kappa_) {}

  static LinearElastic FromYoungPoisson(double E, double nu)
  {
    return LinearElastic(E / (2.0 * (1.0 + nu)), E / (3.0 * (1.0 - 2.0 * nu)));
  }

  double Lambda() const { return kappa - 2.0 * mu / 3.0; }
  bool Incompressible() const { return !std::isfinite(kappa); }
  double ShearModulus() const { return mu; }

  template <typename T>
  static tensor<T, 3, 3> Strain(const tensor<T, 3, 3> &F)
  {
    return sym(F - I<3>());
  }

  template <typename T>
  tensor<T, 3, 3> PK1(const tensor<T, 3, 3> &F) const
  {
    const tensor<T, 3, 3> eps = Strain(F);
    return (2.0 * mu) * dev(eps) + (kappa * tr(eps)) * I<3>();
  }

  template <typename T>
  T Energy(const tensor<T, 3, 3> &F) const
  {
    const tensor<T, 3, 3> eps = Strain(F);
    const tensor<T, 3, 3> e = dev(eps);
    const T theta = tr(eps);
    return mu * ddot(e, e) + 0.5 * kappa * theta * theta;
  }

  // The deviatoric response, the P_iso of the mixed formulation.
  template <typename T>
  tensor<T, 3, 3> PK1Iso(const tensor<T, 3, 3> &F) const
  {
    return (2.0 * mu) * dev(Strain(F));
  }

  template <typename T>
  T EnergyIso(const tensor<T, 3, 3> &F) const
  {
    const tensor<T, 3, 3> e = dev(Strain(F));
    return mu * ddot(e, e);
  }

  // Volumetric response in terms of the volume ratio theta = 1 + tr(eps):
  // u'(theta) = theta - 1, u'' = 1, and kappa u*(p / kappa) = p^2 / (2 kappa).
  template <typename T>
  T NormalizedVolumetricPressure(const T &theta) const { return theta - 1.0; }

  template <typename T>
  T NormalizedVolumetricModulus(const T &) const { return T(1.0); }

  double ComplementaryVolumetricEnergy(double p) const { return 0.5 * p * p / kappa; }
};

} // namespace cmf

// Volumetric strain energy of the decoupled (isochoric-volumetric) materials,
//   U(J) = kappa u(J),
// selectable per material (YAML key material.volumetric). Every law satisfies
// u(1) = 0, u'(1) = 0, u''(1) = 1, so kappa is the small-strain bulk modulus
// for all of them and the meaning of the kappa / nu input does not change;
// the laws differ at finite strain, chiefly in strong compression and in the
// pressure of nearly incompressible problems.
//
//   quadratic    u = (J - 1)^2 / 2            u' = J - 1           u'' = 1
//   simo_taylor  u = (J^2 - 1 - 2 ln J) / 4   u' = (J - 1/J) / 2   u'' = (1 + J^-2) / 2
//   logarithmic  u = (ln J)^2 / 2             u' = ln J / J        u'' = (1 - ln J) / J^2
//                (u' peaks at J = e: the pressure is not monotone beyond it, so
//                 Newton iterates must stay near J = 1; see solver.predictor)
//   j_log_j      u = J ln J - J + 1           u' = ln J            u'' = 1 / J
//
// The displacement formulation uses the pressure U'(J) = kappa u'(J) in
// P = P_iso + U'(J) J F^{-T}. The mixed formulation uses the normalized
// derivatives: its constraint is u'(J) - p / kappa = 0 (weakly), which is
// J = 1 for kappa = inf whatever the law, and its tangent needs u''(J).
// Only the quadratic law has u'' = 1 and a symmetric u-p tangent; the others
// make K_pu = u''(J) K_up^T, which the FGMRES saddle-point solver accepts.
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

#include "base/dual.hpp"

namespace cmf
{

enum class VolumetricLaw { Quadratic, SimoTaylor, Logarithmic, JLogJ };

inline const char *VolumetricLawName(VolumetricLaw law)
{
  switch (law)
  {
    case VolumetricLaw::Quadratic: return "quadratic";
    case VolumetricLaw::SimoTaylor: return "simo_taylor";
    case VolumetricLaw::Logarithmic: return "logarithmic";
    case VolumetricLaw::JLogJ: return "j_log_j";
  }
  return "quadratic";
}

inline VolumetricLaw ParseVolumetricLaw(const std::string &name)
{
  if (name == "quadratic") { return VolumetricLaw::Quadratic; }
  if (name == "simo_taylor") { return VolumetricLaw::SimoTaylor; }
  if (name == "logarithmic") { return VolumetricLaw::Logarithmic; }
  if (name == "j_log_j") { return VolumetricLaw::JLogJ; }
  throw std::invalid_argument("unknown volumetric law '" + name +
                              "' (expected quadratic, simo_taylor, logarithmic, or j_log_j)");
}

// u(J)
template <typename T>
inline T NormalizedVolumetricEnergy(VolumetricLaw law, const T &J)
{
  switch (law)
  {
    case VolumetricLaw::Quadratic: return 0.5 * (J - 1.0) * (J - 1.0);
    case VolumetricLaw::SimoTaylor: return 0.25 * (J * J - 1.0 - 2.0 * log(J));
    case VolumetricLaw::Logarithmic: { const T l = log(J); return 0.5 * l * l; }
    case VolumetricLaw::JLogJ: return J * log(J) - J + 1.0;
  }
  return T(0.0);
}

// u'(J), the pressure per unit bulk modulus.
template <typename T>
inline T NormalizedVolumetricPressure(VolumetricLaw law, const T &J)
{
  switch (law)
  {
    case VolumetricLaw::Quadratic: return J - 1.0;
    case VolumetricLaw::SimoTaylor: return 0.5 * (J - 1.0 / J);
    case VolumetricLaw::Logarithmic: return log(J) / J;
    case VolumetricLaw::JLogJ: return log(J);
  }
  return T(0.0);
}

// u''(J), the tangent bulk modulus per unit kappa.
template <typename T>
inline T NormalizedVolumetricModulus(VolumetricLaw law, const T &J)
{
  switch (law)
  {
    case VolumetricLaw::Quadratic: return T(1.0);
    case VolumetricLaw::SimoTaylor: return 0.5 * (1.0 + 1.0 / (J * J));
    case VolumetricLaw::Logarithmic: return (1.0 - log(J)) / (J * J);
    case VolumetricLaw::JLogJ: return 1.0 / J;
  }
  return T(1.0);
}

// The Legendre transform u*(pi) = sup_J [pi (J - 1) - u(J)] of the normalized
// law, pi = p / kappa: the volumetric energy as a function of the pressure
// (pi^2 / 2 for the quadratic law). It enters the energy diagnostic of the
// mixed formulation, Psi_iso + p (J - 1) - kappa u*(p / kappa), which is the
// perturbed Lagrangian for the quadratic law and stays insensitive to the
// pointwise constraint violation of the discrete solution for the others.
// The logarithmic law has no closed-form inverse of u'; its supremum is
// found by Newton on ln J / J = pi, which exists for pi < 1/e (the law's
// pressure cap); larger pi is clamped to J = e.
inline double NormalizedComplementaryEnergy(VolumetricLaw law, double pi)
{
  auto transform = [&](double J)
  { return pi * (J - 1.0) - NormalizedVolumetricEnergy(law, J); };
  switch (law)
  {
    case VolumetricLaw::Quadratic: return 0.5 * pi * pi;
    case VolumetricLaw::SimoTaylor: return transform(pi + std::sqrt(pi * pi + 1.0));
    case VolumetricLaw::JLogJ: return std::exp(pi) - pi - 1.0;
    case VolumetricLaw::Logarithmic:
    {
      const double e = std::exp(1.0);
      if (pi >= 1.0 / e) { return transform(e); }
      double J = 1.0;
      for (int it = 0; it < 50; it++)
      {
        const double r = std::log(J) / J - pi;
        const double dr = (1.0 - std::log(J)) / (J * J);
        double step = -r / dr;
        if (J + step <= 0.0) { step = -0.5 * J; }
        if (J + step >= e) { step = 0.5 * (e - J); }
        J += step;
        if (std::abs(step) <= 1e-15 * J) { break; }
      }
      return transform(J);
    }
  }
  return 0.5 * pi * pi;
}

} // namespace cmf

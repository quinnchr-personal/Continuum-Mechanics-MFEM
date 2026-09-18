// One-step time integration of M a + S(u, t) = 0 with the displacement as the
// unknown: the generalized-alpha family with interpolated forces,
//   u* = u_n + dt v_n + dt^2 (1/2 - beta) a_n,      v* = v_n + dt (1 - gamma) a_n,
//   a(u) = (u - u*) / (beta dt^2),                  v(u) = v* + gamma dt a(u),
//   M [(1 - am) a(u) + am a_n] + (1 - af) S(u, t_{n+1}) + af S(u_n, t_n) = 0.
// Divided by 1 - af the static operator enters with unit weight,
//   G(u) = S(u, t_{n+1}) + c_M M (u - u*) + h_n,    dG/du = K(u) + c_M M,
//   c_M = (1 - am) / ((1 - af) beta dt^2),          h_n = (am M a_n + af S_n) / (1 - af),
// which is what physics/dynamic_solid_problem.hpp assembles. Parameter sets:
//   newmark            am = af = 0; beta, gamma given (1/4, 1/2: the trapezoidal rule)
//   hht                am = 0, af = alpha in [0, 1/3], beta = (1 + alpha)^2/4, gamma = 1/2 + alpha
//   generalized_alpha  am = (2r - 1)/(r + 1), af = r/(r + 1), beta = (1 - am + af)^2/4,
//                      gamma = 1/2 - am + af,  r = rho_inf in [0, 1]
// (Newmark 1959; Hilber, Hughes, Taylor 1977; Chung, Hulbert 1993). All are
// second order with gamma = 1/2 - am + af; newmark with gamma > 1/2 is first order.
#pragma once

#include <string>

#include "base/config.hpp"

namespace cmf
{

struct TimeIntegration
{
  std::string scheme = "newmark";
  double alpha_m = 0.0;
  double alpha_f = 0.0;
  double beta = 0.25;
  double gamma = 0.5;

  // c_M of the step equation for the increment dt.
  double MassFactor(double dt) const
  {
    return (1.0 - alpha_m) / ((1.0 - alpha_f) * beta * dt * dt);
  }
  // Spectral radius of the amplification matrix at infinite frequency: 1
  // means no numerical dissipation, 0 asymptotic annihilation.
  double SpectralRadiusAtInfinity() const;
  bool Dissipative() const { return SpectralRadiusAtInfinity() < 1.0 - 1e-12; }
  // Linear problems: stable for every dt (newmark: 2 beta >= gamma >= 1/2).
  bool UnconditionallyStable() const;
  // "generalized-alpha (rho_inf 0.8)" etc., for the run header.
  std::string Description() const;
};

// The parameters of cfg.scheme; throws ConfigError naming the key (under
// `path`) for an unknown scheme or a parameter outside its range.
TimeIntegration MakeTimeIntegration(const DynamicsConfig &cfg,
                                    const std::string &path = "dynamics");

} // namespace cmf

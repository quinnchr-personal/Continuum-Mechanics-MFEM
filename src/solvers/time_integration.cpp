#include "solvers/time_integration.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cmf
{

double TimeIntegration::SpectralRadiusAtInfinity() const
{
  if (scheme == "newmark")
  {
    // Roots of lambda^2 - (2 - (gamma + 1/2)/beta) lambda + 1 - (gamma - 1/2)/beta.
    const double tr = 2.0 - (gamma + 0.5) / beta;
    const double det = 1.0 - (gamma - 0.5) / beta;
    const double disc = tr * tr - 4.0 * det;
    if (disc < 0.0) { return std::sqrt(det); }
    return std::max(std::abs(0.5 * (tr + std::sqrt(disc))), std::abs(0.5 * (tr - std::sqrt(disc))));
  }
  // With beta = (1 - am + af)^2/4 the principal roots meet in a double real
  // root; the third root belongs to the acceleration.
  const double principal = std::abs((alpha_f - alpha_m - 1.0) / (alpha_f - alpha_m + 1.0));
  const double spurious = std::abs(alpha_f / (alpha_f - 1.0));
  return std::max(principal, spurious);
}

bool TimeIntegration::UnconditionallyStable() const
{
  if (scheme == "newmark") { return gamma >= 0.5 && 2.0 * beta >= gamma; }
  return alpha_m <= alpha_f && alpha_f <= 0.5 &&
         beta >= 0.25 + 0.5 * (alpha_f - alpha_m) - 1e-14;
}

std::string TimeIntegration::Description() const
{
  char buf[160];
  if (scheme == "newmark")
  {
    std::snprintf(buf, sizeof(buf), "Newmark (beta %g, gamma %g%s)", beta, gamma,
                  beta == 0.25 && gamma == 0.5 ? ": trapezoidal rule" : "");
  }
  else if (scheme == "hht")
  {
    std::snprintf(buf, sizeof(buf), "HHT-alpha (alpha %g, rho_inf %.4g)", alpha_f,
                  SpectralRadiusAtInfinity());
  }
  else
  {
    std::snprintf(buf, sizeof(buf), "generalized-alpha (rho_inf %g)", SpectralRadiusAtInfinity());
  }
  return buf;
}

TimeIntegration MakeTimeIntegration(const DynamicsConfig &cfg, const std::string &path)
{
  ValidateDynamicsScheme(cfg, path);
  TimeIntegration ti;
  ti.scheme = cfg.scheme;
  if (cfg.scheme == "newmark")
  {
    ti.beta = cfg.beta;
    ti.gamma = cfg.gamma;
  }
  else if (cfg.scheme == "hht")
  {
    ti.alpha_f = cfg.alpha;
    ti.beta = 0.25 * (1.0 + cfg.alpha) * (1.0 + cfg.alpha);
    ti.gamma = 0.5 + cfg.alpha;
  }
  else
  {
    const double r = cfg.rho_inf;
    ti.alpha_m = (2.0 * r - 1.0) / (r + 1.0);
    ti.alpha_f = r / (r + 1.0);
    ti.beta = 0.25 * (1.0 - ti.alpha_m + ti.alpha_f) * (1.0 - ti.alpha_m + ti.alpha_f);
    ti.gamma = 0.5 - ti.alpha_m + ti.alpha_f;
  }
  return ti;
}

} // namespace cmf

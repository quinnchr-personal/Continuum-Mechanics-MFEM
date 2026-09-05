// Material library entry point: the variant of available hyperelastic models,
// the YAML factory, and the (E, nu) -> (mu, lambda) conversion.
#pragma once

#include <string>
#include <variant>

#include "base/config.hpp"
#include "materials/material_tangent.hpp"
#include "materials/neo_hookean.hpp"
#include "materials/st_venant_kirchhoff.hpp"

namespace cmf
{

using Material = std::variant<NeoHookean, StVenantKirchhoff>;

struct LameParameters
{
  double mu;
  double lambda;
};

inline LameParameters LameFromYoungPoisson(double E, double nu)
{
  return {E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))};
}

inline Material MakeMaterial(const MaterialConfig &cfg)
{
  const LameParameters lame = LameFromYoungPoisson(cfg.E, cfg.nu);
  if (cfg.model == "neo_hookean") { return NeoHookean{lame.mu, lame.lambda}; }
  if (cfg.model == "st_venant_kirchhoff")
  {
    return StVenantKirchhoff{lame.mu, lame.lambda};
  }
  throw ConfigError("material.model: unknown model '" + cfg.model + "'");
}

inline std::string MaterialName(const Material &m)
{
  return std::visit([](const auto &mat) -> std::string
  {
    using M = std::decay_t<decltype(mat)>;
    if constexpr (std::is_same_v<M, NeoHookean>) { return "neo_hookean"; }
    else { return "st_venant_kirchhoff"; }
  }, m);
}

} // namespace cmf

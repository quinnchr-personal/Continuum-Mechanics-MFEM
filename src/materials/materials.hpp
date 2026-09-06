// Material library entry point: the variant of available hyperelastic models,
// the YAML factory, and the (E, nu) -> (mu, lambda) conversion.
#pragma once

#include <string>
#include <variant>

#include <cmath>
#include <limits>

#include "base/config.hpp"
#include "materials/iso_neo_hookean.hpp"
#include "materials/material_tangent.hpp"
#include "materials/mooney_rivlin.hpp"
#include "materials/neo_hookean.hpp"
#include "materials/st_venant_kirchhoff.hpp"

namespace cmf
{

// Every model usable in the displacement formulation (needs PK1<T>(F)).
using Material = std::variant<NeoHookean, StVenantKirchhoff, IsoNeoHookean, MooneyRivlin>;
// Decoupled models usable in the mixed u-p formulation (PK1Iso<T>(F),
// VolumetricPressure<T>(J), kappa possibly infinite).
using MixedMaterial = std::variant<IsoNeoHookean, MooneyRivlin>;

struct LameParameters
{
  double mu;
  double lambda;
};

inline LameParameters LameFromYoungPoisson(double E, double nu)
{
  return {E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))};
}

// Resolved small-strain moduli of a MaterialConfig (kappa = inf when
// incompressible). Validates the per-model key combinations.
struct ResolvedModuli
{
  double mu = 0.0;
  double kappa = std::numeric_limits<double>::infinity();
  double lambda = 0.0; // kappa - 2 mu / 3 (inf when incompressible)
  bool incompressible = false;
};

ResolvedModuli ResolveModuli(const MaterialConfig &cfg);

Material MakeMaterial(const MaterialConfig &cfg);
MixedMaterial MakeMixedMaterial(const MaterialConfig &cfg);
bool IsDecoupledModel(const std::string &model);

inline std::string MaterialName(const Material &m)
{
  return std::visit([](const auto &mat) -> std::string
  {
    using M = std::decay_t<decltype(mat)>;
    if constexpr (std::is_same_v<M, NeoHookean>) { return "neo_hookean"; }
    else if constexpr (std::is_same_v<M, StVenantKirchhoff>) { return "st_venant_kirchhoff"; }
    else if constexpr (std::is_same_v<M, IsoNeoHookean>) { return "iso_neo_hookean"; }
    else { return "mooney_rivlin"; }
  }, m);
}

inline std::string MaterialName(const MixedMaterial &m)
{
  return std::visit([](const auto &mat) -> std::string
  {
    using M = std::decay_t<decltype(mat)>;
    if constexpr (std::is_same_v<M, IsoNeoHookean>) { return "iso_neo_hookean"; }
    else { return "mooney_rivlin"; }
  }, m);
}

} // namespace cmf

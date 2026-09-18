// Material library entry point: the variant of available hyperelastic models,
// the YAML factory, and the (E, nu) -> (mu, lambda) conversion.
#pragma once

#include <string>
#include <type_traits>
#include <variant>

#include <cmath>
#include <limits>

#include "base/config.hpp"
#include "materials/arruda_boyce.hpp"
#include "materials/gent.hpp"
#include "materials/gent_compressible_summit.hpp"
#include "materials/iso_neo_hookean.hpp"
#include "materials/material_tangent.hpp"
#include "materials/mooney_rivlin.hpp"
#include "materials/neo_hookean.hpp"
#include "materials/ogden.hpp"
#include "materials/plane_stress.hpp"
#include "materials/st_venant_kirchhoff.hpp"
#include "materials/volumetric.hpp"
#include "materials/yeoh.hpp"

namespace cmf
{

// Every model usable in the displacement formulation (needs PK1<T>(F)),
// each also wrapped by the plane-stress adapter (2D, plane: stress).
using Material = std::variant<NeoHookean, StVenantKirchhoff, GentCompressibleSummit, IsoNeoHookean,
                              MooneyRivlin, Yeoh, Gent, ArrudaBoyce, Ogden,
                              PlaneStress<NeoHookean>, PlaneStress<StVenantKirchhoff>,
                              PlaneStress<GentCompressibleSummit>,
                              PlaneStress<IsoNeoHookean>, PlaneStress<MooneyRivlin>,
                              PlaneStress<Yeoh>, PlaneStress<Gent>, PlaneStress<ArrudaBoyce>,
                              PlaneStress<Ogden>>;

template <typename M> struct is_plane_stress : std::false_type {};
template <typename B> struct is_plane_stress<PlaneStress<B>> : std::true_type {};
// Decoupled models usable in the mixed u-p formulation (PK1Iso<T>(F),
// VolumetricPressure<T>(J), ShearModulus(), kappa possibly infinite).
using MixedMaterial = std::variant<IsoNeoHookean, MooneyRivlin, Yeoh, Gent, ArrudaBoyce, Ogden>;

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

// plane_stress wraps the model in the PlaneStress adapter (2D displacement
// formulation), which also admits incompressible models.
Material MakeMaterial(const MaterialConfig &cfg, bool plane_stress = false);
MixedMaterial MakeMixedMaterial(const MaterialConfig &cfg);
bool IsDecoupledModel(const std::string &model);

// Materials by element attribute: a table of size 1 is one material for
// every element; otherwise index = attribute (entry 0 unused, filled with
// the base) and every entry holds the same variant alternative.
template <typename V>
inline const V &MaterialAt(const std::vector<V> &table, int attribute)
{
  return table.size() == 1 ? table[0] : table[std::size_t(attribute)];
}

// The concrete materials of a table, all of type M (the alternative of
// table[0]); throws if a region holds another alternative.
template <typename M, typename V>
inline std::vector<M> UnpackMaterials(const std::vector<V> &table)
{
  std::vector<M> out;
  for (const V &v : table)
  {
    if (!std::holds_alternative<M>(v))
    {
      throw ConfigError("material.regions: every region must use the base model");
    }
    out.push_back(std::get<M>(v));
  }
  return out;
}

// The YAML name of a material type (the plane-stress adapter reports its base).
template <typename M>
constexpr const char *ModelName()
{
  if constexpr (is_plane_stress<M>::value) { return ModelName<typename M::Base>(); }
  else if constexpr (std::is_same_v<M, NeoHookean>) { return "neo_hookean"; }
  else if constexpr (std::is_same_v<M, StVenantKirchhoff>) { return "st_venant_kirchhoff"; }
  else if constexpr (std::is_same_v<M, GentCompressibleSummit>) { return "gent_compressible_summit"; }
  else if constexpr (std::is_same_v<M, IsoNeoHookean>) { return "iso_neo_hookean"; }
  else if constexpr (std::is_same_v<M, MooneyRivlin>) { return "mooney_rivlin"; }
  else if constexpr (std::is_same_v<M, Yeoh>) { return "yeoh"; }
  else if constexpr (std::is_same_v<M, Gent>) { return "gent"; }
  else if constexpr (std::is_same_v<M, ArrudaBoyce>) { return "arruda_boyce"; }
  else { return "ogden"; }
}

inline std::string MaterialName(const Material &m)
{
  return std::visit([](const auto &mat) -> std::string
  {
    using M = std::decay_t<decltype(mat)>;
    std::string name = ModelName<M>();
    if constexpr (is_plane_stress<M>::value) { name += " (plane stress)"; }
    return name;
  }, m);
}

inline std::string MaterialName(const MixedMaterial &m)
{
  return std::visit([](const auto &mat) -> std::string
  { return ModelName<std::decay_t<decltype(mat)>>(); }, m);
}

// ", <law> volumetric law" for a decoupled material with a finite bulk modulus
// and a non-default law; "" otherwise (the plane-stress adapter reports its base).
template <typename M, typename = void>
struct has_volumetric_law : std::false_type {};
template <typename M>
struct has_volumetric_law<M, std::void_t<decltype(std::declval<const M &>().law)>> : std::true_type {};

template <typename M>
inline std::string VolumetricLawSuffixOf(const M &mat)
{
  if constexpr (is_plane_stress<M>::value) { return VolumetricLawSuffixOf(mat.material); }
  else if constexpr (has_volumetric_law<M>::value)
  {
    if (mat.Incompressible() || mat.law == VolumetricLaw::Quadratic) { return ""; }
    return std::string(", ") + VolumetricLawName(mat.law) + " volumetric law";
  }
  else { return ""; }
}

inline std::string VolumetricLawSuffix(const Material &m)
{
  return std::visit([](const auto &mat) { return VolumetricLawSuffixOf(mat); }, m);
}

inline std::string VolumetricLawSuffix(const MixedMaterial &m)
{
  return std::visit([](const auto &mat) { return VolumetricLawSuffixOf(mat); }, m);
}

} // namespace cmf

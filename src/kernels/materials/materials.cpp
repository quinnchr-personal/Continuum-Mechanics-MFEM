#include "materials/materials.hpp"

#include <string>

namespace cmf
{

namespace
{

bool Set(double v) { return !std::isnan(v); }
double Or(double v, double fallback) { return std::isnan(v) ? fallback : v; }

MixedMaterial MakeDecoupledBase(const MaterialConfig &cfg, const ResolvedModuli &m)
{
  if (cfg.model == "iso_neo_hookean") { return IsoNeoHookean(m.mu, m.kappa); }
  if (cfg.model == "mooney_rivlin") { return MooneyRivlin(cfg.c1, cfg.c2, m.kappa); }
  if (cfg.model == "yeoh") { return Yeoh(cfg.c10, Or(cfg.c20, 0.0), Or(cfg.c30, 0.0), m.kappa); }
  if (cfg.model == "gent") { return Gent(cfg.mu, cfg.Jm, m.kappa); }
  if (cfg.model == "arruda_boyce") { return ArrudaBoyce(cfg.mu, cfg.N, m.kappa); }
  return Ogden(cfg.mu_r, cfg.alpha_r, m.kappa);
}

// The decoupled material with its volumetric law (material.volumetric).
MixedMaterial MakeDecoupled(const MaterialConfig &cfg, const ResolvedModuli &m)
{
  MixedMaterial material = MakeDecoupledBase(cfg, m);
  const VolumetricLaw law = ParseVolumetricLaw(cfg.volumetric);
  std::visit([law](auto &mat) { mat.law = law; }, material);
  return material;
}

} // namespace

bool IsDecoupledModel(const std::string &model)
{
  return model == "iso_neo_hookean" || model == "mooney_rivlin" || model == "yeoh" ||
         model == "gent" || model == "arruda_boyce" || model == "ogden";
}

ResolvedModuli ResolveModuli(const MaterialConfig &cfg)
{
  ValidateMaterialConfig(cfg);
  ResolvedModuli r;
  const std::string &model = cfg.model;
  if (model == "neo_hookean" || model == "st_venant_kirchhoff")
  {
    const LameParameters lame = LameFromYoungPoisson(cfg.E, cfg.nu);
    r.mu = lame.mu;
    r.lambda = lame.lambda;
    r.kappa = lame.lambda + 2.0 * lame.mu / 3.0;
    return r;
  }
  const double inf = std::numeric_limits<double>::infinity();
  if (model == "mooney_rivlin") { r.mu = 2.0 * (cfg.c1 + cfg.c2); }
  else if (model == "yeoh") { r.mu = 2.0 * cfg.c10; }
  else if (model == "arruda_boyce") { r.mu = ArrudaBoyce(cfg.mu, cfg.N, inf).ShearModulus(); }
  else if (model == "ogden") { r.mu = Ogden(cfg.mu_r, cfg.alpha_r, inf).ShearModulus(); }
  else if (Set(cfg.mu)) { r.mu = cfg.mu; }
  else { r.mu = cfg.E / (2.0 * (1.0 + cfg.nu)); }
  if (!(r.mu > 0.0)) { throw ConfigError("material: shear modulus must be positive"); }

  if (cfg.incompressible || (Set(cfg.nu) && cfg.nu == 0.5))
  {
    r.incompressible = true;
    r.kappa = std::numeric_limits<double>::infinity();
    r.lambda = std::numeric_limits<double>::infinity();
    return r;
  }
  if (Set(cfg.kappa)) { r.kappa = cfg.kappa; }
  else { r.kappa = 2.0 * r.mu * (1.0 + cfg.nu) / (3.0 * (1.0 - 2.0 * cfg.nu)); }
  if (!(r.kappa > 0.0)) { throw ConfigError("material: bulk modulus must be positive"); }
  r.lambda = r.kappa - 2.0 * r.mu / 3.0;
  return r;
}

Material MakeMaterial(const MaterialConfig &cfg, bool plane_stress)
{
  const ResolvedModuli m = ResolveModuli(cfg);
  Material base;
  if (cfg.model == "neo_hookean") { base = NeoHookean{m.mu, m.lambda}; }
  else if (cfg.model == "st_venant_kirchhoff") { base = StVenantKirchhoff{m.mu, m.lambda}; }
  else
  {
    if (m.incompressible && !plane_stress)
    {
      throw ConfigError("material: model '" + cfg.model + "' is incompressible; "
                        "use formulation: mixed (or plane: stress in 2D)");
    }
    base = std::visit([](const auto &mat) -> Material { return mat; }, MakeDecoupled(cfg, m));
  }
  if (!plane_stress) { return base; }
  return std::visit([](const auto &mat) -> Material
  {
    using M = std::decay_t<decltype(mat)>;
    if constexpr (is_plane_stress<M>::value) { return mat; }
    else { return PlaneStress<M>(mat); }
  }, base);
}

MixedMaterial MakeMixedMaterial(const MaterialConfig &cfg)
{
  if (!IsDecoupledModel(cfg.model))
  {
    throw ConfigError("material.model: '" + cfg.model + "' has no isochoric-"
                      "volumetric split; formulation: mixed needs one of iso_neo_hookean, "
                      "mooney_rivlin, yeoh, gent, arruda_boyce, ogden");
  }
  return MakeDecoupled(cfg, ResolveModuli(cfg));
}

} // namespace cmf

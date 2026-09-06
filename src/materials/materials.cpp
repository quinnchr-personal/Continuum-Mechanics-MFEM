#include "materials/materials.hpp"

#include <string>

namespace cmf
{

namespace
{

bool Set(double v) { return !std::isnan(v); }

} // namespace

bool IsDecoupledModel(const std::string &model)
{
  return model == "iso_neo_hookean" || model == "mooney_rivlin";
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
  if (model == "mooney_rivlin") { r.mu = 2.0 * (cfg.c1 + cfg.c2); }
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

Material MakeMaterial(const MaterialConfig &cfg)
{
  const ResolvedModuli m = ResolveModuli(cfg);
  if (cfg.model == "neo_hookean") { return NeoHookean{m.mu, m.lambda}; }
  if (cfg.model == "st_venant_kirchhoff") { return StVenantKirchhoff{m.mu, m.lambda}; }
  if (m.incompressible)
  {
    throw ConfigError("material: model '" + cfg.model + "' is incompressible; "
                      "use formulation: mixed");
  }
  if (cfg.model == "iso_neo_hookean") { return IsoNeoHookean(m.mu, m.kappa); }
  return MooneyRivlin(cfg.c1, cfg.c2, m.kappa);
}

MixedMaterial MakeMixedMaterial(const MaterialConfig &cfg)
{
  if (!IsDecoupledModel(cfg.model))
  {
    throw ConfigError("material.model: '" + cfg.model + "' has no isochoric-"
                      "volumetric split; formulation: mixed needs iso_neo_hookean "
                      "or mooney_rivlin");
  }
  const ResolvedModuli m = ResolveModuli(cfg);
  if (cfg.model == "iso_neo_hookean") { return IsoNeoHookean(m.mu, m.kappa); }
  return MooneyRivlin(cfg.c1, cfg.c2, m.kappa);
}

} // namespace cmf

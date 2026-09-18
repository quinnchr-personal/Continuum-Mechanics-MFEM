#include "physics/solid_problem.hpp"

#include "base/coefficients.hpp"
#include "base/mesh_input.hpp"
#include "physics/mixed_solid_mechanics_tl.hpp"
#include "physics/solid_mechanics_tl.hpp"

namespace cmf
{

namespace
{

template <typename V, typename Make>
std::vector<V> TableByAttribute(const MaterialConfig &cfg, mfem::Mesh &mesh, const Make &make)
{
  std::vector<V> table(1, make(cfg));
  if (cfg.regions.empty()) { return table; }
  const int max_attr = mesh.attributes.Size() ? mesh.attributes.Max() : 0;
  table.assign(std::size_t(max_attr) + 1, table[0]);
  std::vector<int> owner(std::size_t(max_attr) + 1, -1);
  for (std::size_t i = 0; i < cfg.regions.size(); i++)
  {
    const MaterialConfig &region = cfg.regions[i];
    const std::string what = "material.regions[" + std::to_string(i) + "]";
    const V material = make(region);
    for (int a : ResolveElementAttributes(mesh, region.attr, region.attr_names, what))
    {
      if (owner[std::size_t(a)] >= 0)
      {
        throw ConfigError(what + ": element attribute " + std::to_string(a) +
                          " is already covered by material.regions[" +
                          std::to_string(owner[std::size_t(a)]) + "]");
      }
      owner[std::size_t(a)] = int(i);
      table[std::size_t(a)] = material;
    }
  }
  return table;
}

} // namespace

std::vector<Material> MakeMaterialTable(const MaterialConfig &cfg, mfem::Mesh &mesh,
                                        bool plane_stress)
{
  return TableByAttribute<Material>(cfg, mesh, [plane_stress](const MaterialConfig &c)
  { return MakeMaterial(c, plane_stress); });
}

std::vector<MixedMaterial> MakeMixedMaterialTable(const MaterialConfig &cfg, mfem::Mesh &mesh)
{
  return TableByAttribute<MixedMaterial>(cfg, mesh, [](const MaterialConfig &c)
  { return MakeMixedMaterial(c); });
}

mfem::Vector MakeDensityTable(const MaterialConfig &cfg, mfem::Mesh &mesh)
{
  const int max_attr = mesh.attributes.Size() ? mesh.attributes.Max() : 1;
  mfem::Vector table(max_attr);
  table = cfg.rho0;
  for (std::size_t i = 0; i < cfg.regions.size(); i++)
  {
    const MaterialConfig &region = cfg.regions[i];
    const std::string what = "material.regions[" + std::to_string(i) + "]";
    for (int a : ResolveElementAttributes(mesh, region.attr, region.attr_names, what))
    {
      table(a - 1) = region.rho0;
    }
  }
  return table;
}

std::unique_ptr<SolidProblem> MakeSolidProblem(mfem::ParMesh &mesh, const AppConfig &cfg)
{
  const bool mixed = cfg.formulation == "mixed";
  std::vector<MixedMaterial> mixed_materials;
  std::vector<Material> materials;
  if (mixed) { mixed_materials = MakeMixedMaterialTable(cfg.material, mesh); }
  else { materials = MakeMaterialTable(cfg.material, mesh, cfg.plane == "stress"); }
  const bool small_strain = mixed ? IsSmallStrain(mixed_materials[0]) : IsSmallStrain(materials[0]);
  if (small_strain && cfg.solver.predictor == "tangent")
  {
    // The first Newton step of a linear problem is the exact solve. After an
    // exact predictor Newton would start at the round-off floor, where
    // rtol |R0| cannot be reached and the line search fails.
    throw ConfigError("key 'solver.predictor': tangent is not used by model '" + cfg.material.model +
                      "' (small strain: the first Newton step is already the exact linear solve)");
  }
  if (mixed) { return std::make_unique<MixedSolidMechanicsTL>(mesh, cfg, mixed_materials); }
  return std::make_unique<SolidMechanicsTL>(mesh, cfg, materials);
}

BCOptions OptionsOf(const BoundaryCondition &bc)
{
  BCOptions opt;
  opt.components = bc.components;
  opt.schedule = bc.schedule;
  opt.time_dependent = ExpressionsUseTime(bc.expression);
  opt.name = bc.name;
  return opt;
}

void InstallYamlLoads(SolidProblem &problem, mfem::Mesh &mesh, const AppConfig &cfg, int dim,
                      std::vector<std::unique_ptr<mfem::VectorCoefficient>> &owned_vectors,
                      std::vector<std::unique_ptr<mfem::Coefficient>> &owned_scalars)
{
  for (std::size_t i = 0; i < cfg.bcs.dirichlet.size(); i++)
  {
    const BoundaryCondition &bc = cfg.bcs.dirichlet[i];
    const std::string what = "bcs.dirichlet[" + std::to_string(i) + "]";
    owned_vectors.push_back(MakeBCCoefficient(bc, dim, what));
    problem.AddDirichlet(ResolveBoundaryAttributes(mesh, bc, what), *owned_vectors.back(),
                         OptionsOf(bc));
  }
  for (std::size_t i = 0; i < cfg.bcs.traction.size(); i++)
  {
    const BoundaryCondition &bc = cfg.bcs.traction[i];
    const std::string what = "bcs.traction[" + std::to_string(i) + "]";
    const std::vector<int> attrs = ResolveBoundaryAttributes(mesh, bc, what);
    if (bc.IsPressure())
    {
      owned_scalars.push_back(MakeScalarBCCoefficient(bc, what));
      try
      {
        problem.AddPressure(attrs, *owned_scalars.back(), bc.type == "follower_pressure",
                            OptionsOf(bc));
      }
      catch (const ConfigError &e) { throw ConfigError(what + ": " + e.what()); }
    }
    else
    {
      owned_vectors.push_back(MakeBCCoefficient(bc, dim, what));
      problem.AddTraction(attrs, *owned_vectors.back(), OptionsOf(bc));
    }
  }
  if (!cfg.body_force.Empty())
  {
    const BodyForceConfig &bf = cfg.body_force;
    if (int(bf.expression.size()) != dim)
    {
      throw ConfigError("body_force.expression has " + std::to_string(bf.expression.size()) +
                        " entries, expected " + std::to_string(dim));
    }
    owned_vectors.push_back(std::make_unique<ExpressionVectorCoefficient>(bf.expression));
    BCOptions opt;
    opt.schedule = bf.schedule;
    opt.time_dependent = ExpressionsUseTime(bf.expression);
    problem.SetBodyForce(*owned_vectors.back(), opt);
  }
}

} // namespace cmf

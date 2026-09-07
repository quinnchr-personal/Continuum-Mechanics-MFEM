#include "physics/solid_problem.hpp"

#include "base/coefficients.hpp"
#include "base/mesh_input.hpp"
#include "physics/mixed_solid_mechanics_tl.hpp"
#include "physics/solid_mechanics_tl.hpp"

namespace cmf
{

std::unique_ptr<SolidProblem> MakeSolidProblem(mfem::ParMesh &mesh, const AppConfig &cfg)
{
  if (cfg.formulation == "mixed")
  {
    return std::make_unique<MixedSolidMechanicsTL>(mesh, cfg, MakeMixedMaterial(cfg.material));
  }
  return std::make_unique<SolidMechanicsTL>(mesh, cfg, MakeMaterial(cfg.material, cfg.plane == "stress"));
}

BCOptions OptionsOf(const BoundaryCondition &bc)
{
  BCOptions opt;
  opt.components = bc.components;
  opt.schedule = bc.schedule;
  opt.time_dependent = ExpressionsUseTime(bc.expression);
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
      problem.AddPressure(attrs, *owned_scalars.back(), bc.type == "follower_pressure",
                          OptionsOf(bc));
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

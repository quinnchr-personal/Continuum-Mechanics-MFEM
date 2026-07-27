#include "physics/physics_factory.hpp"

#include "physics/perfect_gas_model.hpp"

#include <stdexcept>
#include <string>

namespace hycfd
{
namespace
{

template <typename T>
T Required(const YAML::Node &node, const std::string &key)
{
   if (!node || !node[key])
   {
      throw std::runtime_error(
         "missing required physics key: " + key);
   }
   return node[key].as<T>();
}

template <typename T>
T Optional(const YAML::Node &node, const std::string &key, T fallback)
{
   if (!node || !node[key]) { return fallback; }
   return node[key].as<T>();
}

} // namespace

std::unique_ptr<PhysicsModel> MakePhysics(const YAML::Node &physics)
{
   if (!physics)
   {
      throw std::runtime_error("missing required YAML block: physics");
   }
   const std::string model =
      Optional<std::string>(physics, "model", "perfect_gas");
   if (model != "perfect_gas")
   {
      throw std::runtime_error(
         "unknown physics model '" + model +
         "'; supported: perfect_gas");
   }
   PerfectGasParams params;
   params.gamma = Optional<double>(physics, "gamma", 1.4);
   params.reynolds = Required<double>(physics, "reynolds");
   params.prandtl = Optional<double>(physics, "prandtl", 0.71);
   params.mach = Required<double>(physics, "mach");
   params.T_inf_K = Required<double>(physics, "T_inf_K");
   params.Twall_K = Required<double>(physics, "Twall_K");
   params.tau = Optional<double>(physics, "tau", 1.0);
   const std::string regularization =
      Optional<std::string>(physics, "regularization", "none");
   if (regularization == "floors")
   {
      params.regularized = true;
   }
   else if (regularization != "none")
   {
      throw std::runtime_error(
         "physics.regularization must be floors or none");
   }
   return std::make_unique<PerfectGasModel>(params);
}

} // namespace hycfd

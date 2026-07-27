#pragma once

#include "physics/physics_model.hpp"

#include "yaml-cpp/yaml.h"

#include <memory>

namespace hycfd
{

// Builds a PhysicsModel from a YAML physics block. The block's `model` key
// selects the model (default and currently only option: perfect_gas); the
// remaining keys are model parameters.
std::unique_ptr<PhysicsModel> MakePhysics(const YAML::Node &physics);

} // namespace hycfd

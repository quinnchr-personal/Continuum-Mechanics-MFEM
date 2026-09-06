// YAML input schema for the framework: flat config structs plus a loader that
// validates types and names the offending key in every error.
#pragma once

#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "yaml-cpp/yaml.h"

namespace cmf
{

class ConfigError : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

struct CartesianMeshConfig
{
  int dim = 2;
  int nx = 1, ny = 1, nz = 1;
  double sx = 1.0, sy = 1.0, sz = 1.0;
  std::string element = "quad"; // quad | tri | hex | tet
};

struct MeshConfig
{
  std::string file;              // mesh file; empty -> cartesian box
  bool cartesian = false;
  CartesianMeshConfig box;
  // Optional bilinear image of the 2D box: corners at (0,0), (sx,0), (sx,sy),
  // (0,sy) are mapped to corners[0..3].
  std::vector<std::array<double, 2>> corners;
  double perturb = 0.0;          // interior vertex jitter of the base mesh, fraction of h
  int serial_refine = 0;
  int parallel_refine = 0;
  int order = 1;
};

// Material parameters; unset numeric keys are NaN. Which keys a model needs
// is validated in materials.cpp (ResolveModuli):
//   neo_hookean, st_venant_kirchhoff: E, nu
//   iso_neo_hookean: mu or (E, nu); bulk from kappa | nu | incompressible
//   mooney_rivlin: c1, c2; bulk from kappa | nu | incompressible
struct MaterialConfig
{
  std::string model = "neo_hookean";
  double E = std::numeric_limits<double>::quiet_NaN();
  double nu = std::numeric_limits<double>::quiet_NaN();
  double mu = std::numeric_limits<double>::quiet_NaN();
  double kappa = std::numeric_limits<double>::quiet_NaN();
  double c1 = std::numeric_limits<double>::quiet_NaN();
  double c2 = std::numeric_limits<double>::quiet_NaN();
  bool incompressible = false;
  double rho0 = 1.0;
};

struct BoundaryCondition
{
  std::vector<int> attr;
  std::vector<double> value;
};

struct BCConfig
{
  std::vector<BoundaryCondition> dirichlet;
  std::vector<BoundaryCondition> traction;
};

struct NewtonConfig
{
  double rtol = 1e-10;
  double atol = 1e-12;
  int max_it = 25;
  double armijo_c = 1e-4;
  int max_halvings = 8;
  int print_level = 1;
};

struct LinearSolverConfig
{
  std::string type = "gmres_amg"; // gmres_amg | cg_amg
  std::string amg = "elasticity"; // elasticity | systems
  double rtol = 1e-12;
  double atol = 0.0;
  int max_it = 500;
  int krylov_dim = 50;
  int print_level = 0;
  // Mixed (u-p) formulation only: inner displacement-block solve inside the
  // block preconditioner (GMRES + AMG to inner_rtol, at most inner_max_it)
  // and the augmented Lagrangian parameter gamma = augmentation * mu
  // (0 disables the augmentation).
  double inner_rtol = 1e-3;
  int inner_max_it = 50;
  double augmentation = 1.0;
};

struct SolverConfig
{
  int load_steps = 1;
  NewtonConfig newton;
  LinearSolverConfig linear;
};

struct ProbeConfig
{
  std::string name;
  std::vector<double> point;
};

struct OutputConfig
{
  std::string paraview;              // collection path; empty -> no output
  std::vector<std::string> fields;   // displacement | pressure | vonmises | jacobian
  bool high_order = true;
  std::vector<ProbeConfig> probes;   // displacement printed at these points
};

struct AppConfig
{
  std::string formulation = "displacement"; // displacement | mixed (u-p)
  MeshConfig mesh;
  MaterialConfig material;
  BCConfig bcs;
  std::vector<double> body_force;
  SolverConfig solver;
  OutputConfig output;
};

// Parse the schema above. Unknown keys, missing required keys, and type
// mismatches throw ConfigError with the full key path (e.g. 'material.E').
AppConfig ParseConfig(const YAML::Node &root);
AppConfig LoadConfig(const std::string &path);

// Per-model key requirements of a MaterialConfig (which keys must be given,
// which are unused, which bulk-modulus specification). Called by
// ParseMaterialConfig; errors name the key path under `path`.
void ValidateMaterialConfig(const MaterialConfig &cfg, const std::string &path = "material");

// Individual section parsers, exposed so tests and other physics can reuse them.
MeshConfig ParseMeshConfig(const YAML::Node &node, const std::string &path);
MaterialConfig ParseMaterialConfig(const YAML::Node &node, const std::string &path);
BCConfig ParseBCConfig(const YAML::Node &node, const std::string &path);
SolverConfig ParseSolverConfig(const YAML::Node &node, const std::string &path);
OutputConfig ParseOutputConfig(const YAML::Node &node, const std::string &path);

} // namespace cmf

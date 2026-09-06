// YAML input schema for the framework: flat config structs plus a loader that
// validates types and names the offending key in every error.
#pragma once

#include <array>
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

struct MaterialConfig
{
  std::string model = "neo_hookean"; // neo_hookean | st_venant_kirchhoff
  double E = 1.0;
  double nu = 0.3;
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
  std::vector<std::string> fields;   // displacement | vonmises | jacobian
  bool high_order = true;
  std::vector<ProbeConfig> probes;   // displacement printed at these points
};

struct AppConfig
{
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

// Individual section parsers, exposed so tests and other physics can reuse them.
MeshConfig ParseMeshConfig(const YAML::Node &node, const std::string &path);
MaterialConfig ParseMaterialConfig(const YAML::Node &node, const std::string &path);
BCConfig ParseBCConfig(const YAML::Node &node, const std::string &path);
SolverConfig ParseSolverConfig(const YAML::Node &node, const std::string &path);
OutputConfig ParseOutputConfig(const YAML::Node &node, const std::string &path);

} // namespace cmf

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

// The YAML schema reads meshes from files (Gmsh .msh, or any format MFEM
// reads): `file` is required and the physical groups become the element and
// boundary attributes, with their names kept for the boundary conditions.
// The Cartesian box and the corner map are not part of the YAML schema; they
// remain here for the tests, which build their meshes programmatically.
struct MeshConfig
{
  std::string file;              // mesh file (required in YAML)
  bool cartesian = false;        // programmatic only: box instead of file
  CartesianMeshConfig box;
  // Programmatic only: bilinear image of the 2D box, corners at (0,0), (sx,0),
  // (sx,sy), (0,sy) mapped to corners[0..3].
  std::vector<std::array<double, 2>> corners;
  double perturb = 0.0;          // interior vertex jitter of the base mesh, fraction of h
  int serial_refine = 0;
  int parallel_refine = 0;
  int order = 1;
};

// Material parameters; unset numeric keys are NaN. Which keys a model needs
// is validated by ValidateMaterialConfig:
//   neo_hookean, st_venant_kirchhoff: E, nu
//   iso_neo_hookean: mu or (E, nu)
//   mooney_rivlin: c1, c2                (mu = 2 (c1 + c2))
//   yeoh: c10, [c20, c30]                (mu = 2 c10)
//   gent: mu, Jm
//   arruda_boyce: mu, N                  (small-strain modulus mu (1 + 3/(5N) + ...))
//   ogden: mu_r, alpha_r lists           (mu = 1/2 sum mu_r alpha_r)
// The decoupled models (all but the first two) take the bulk modulus from
// exactly one of kappa | nu | incompressible.
struct MaterialConfig
{
  std::string model = "neo_hookean";
  double E = std::numeric_limits<double>::quiet_NaN();
  double nu = std::numeric_limits<double>::quiet_NaN();
  double mu = std::numeric_limits<double>::quiet_NaN();
  double kappa = std::numeric_limits<double>::quiet_NaN();
  double c1 = std::numeric_limits<double>::quiet_NaN();
  double c2 = std::numeric_limits<double>::quiet_NaN();
  double c10 = std::numeric_limits<double>::quiet_NaN();
  double c20 = std::numeric_limits<double>::quiet_NaN();
  double c30 = std::numeric_limits<double>::quiet_NaN();
  double Jm = std::numeric_limits<double>::quiet_NaN();
  double N = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> mu_r;
  std::vector<double> alpha_r;
  bool incompressible = false;
  double rho0 = 1.0;
};

// Boundary attributes by number (attr) and/or by physical-group name
// (attr_names, resolved against the mesh when the physics is built); value,
// plus an optional gradient: the data is value + gradient X in the reference
// coordinates X (affine, for homogeneous deformation tests).
struct BoundaryCondition
{
  std::vector<int> attr;
  std::vector<std::string> attr_names;
  std::vector<double> value;
  std::vector<std::vector<double>> gradient;
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
  // Nodal unknowns: displacement | pressure (mixed). Quadrature-point
  // quantities: cauchy_stress | pk1_stress | deformation_gradient | jacobian
  // | vonmises | energy_density | thickness_stretch (plane stress).
  std::vector<std::string> fields;
  // Presentations of the quadrature-point quantities (physics/quadrature_fields.hpp):
  // "nodes" (continuous H1 field <name>), "elements" (element average
  // <name>_elem), "quadrature_points" (raw point cloud <name>_qp).
  std::vector<std::string> quadrature_at{"nodes"};
  // How the nodal presentation is derived from the quadrature data:
  // "averaged" (element-wise projection, mean at shared nodes) or
  // "projected" (global L2 projection).
  std::string nodal_projection = "averaged";
  bool high_order = true;
  std::vector<ProbeConfig> probes;   // every registered field printed at these points
};

struct AppConfig
{
  std::string formulation = "displacement"; // displacement | mixed (u-p)
  // 2D kinematics: strain (F33 = 1) | stress (F33 = thickness stretch with
  // sigma33 = 0, displacement formulation only; incompressible models need no
  // pressure unknown there).
  std::string plane = "strain";
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

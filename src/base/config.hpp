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
// exactly one of kappa | nu | incompressible, and optionally a volumetric law.
// `regions` override parameters by element attribute (numbers and/or
// physical-volume names in attr/attr_names): the same model with other
// values, every key not given inherited from the base (a region that gives
// any bulk key drops the base's bulk keys first). The base applies to every
// attribute no region names.
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
  // Volumetric law U(J) = kappa u(J) of the decoupled models: quadratic
  // (default, (J - 1)^2 / 2) | simo_taylor | logarithmic | j_log_j
  // (kernels/materials/volumetric.hpp). Not a key of the coupled models.
  std::string volumetric = "quadratic";
  double rho0 = 1.0;
  std::vector<int> attr;                 // regions only
  std::vector<std::string> attr_names;   // regions only
  std::vector<MaterialConfig> regions;   // base only; parameters already merged
};

// Scalar load schedule s(t) of the pseudo-time t in [0, 1] (piecewise
// linear). ramp: 0 at t <= from, 1 at t >= to, linear between (the default,
// from 0 to 1, is the proportional load path). constant: 1 for t > 0.
// table: linear interpolation of (t, s) pairs, clamped outside.
struct Schedule
{
  enum class Kind { Ramp, Constant, Table };
  Kind kind = Kind::Ramp;
  double from = 0.0;
  double to = 1.0;
  std::vector<double> t;
  std::vector<double> s;

  double Eval(double time) const;
  static Schedule Ramp(double from = 0.0, double to = 1.0);
  static Schedule Constant();
  static Schedule Table(const std::vector<double> &t, const std::vector<double> &s);
};

// Boundary attributes by number (attr) and/or by physical-group name
// (attr_names, resolved against the mesh when the physics is built). The
// data is `expression`: one string f(x, y, z, t) per component in the
// reference coordinates and the pseudo-time (base/expression.hpp; a single
// string for the scalar pressure types). Dirichlet entries may restrict the
// prescribed components (empty = all). Every entry carries a schedule; its
// data is schedule(t) * f. The default schedule is the ramp s = t unless the
// expression mentions t, in which case it is constant (t enters through the
// function only). Traction types: vector (nominal traction per reference
// area), pressure (dead, T = -p N), follower_pressure (T = -p J F^{-T} N per
// current area).
struct BoundaryCondition
{
  std::string name;              // optional label (reactions output); default "dirichlet[i]"
  std::vector<int> attr;
  std::vector<std::string> attr_names;
  std::vector<std::string> expression;
  std::vector<int> components;   // Dirichlet only; 0-based, empty = all
  Schedule schedule;
  std::string type = "vector";   // traction only: vector | pressure | follower_pressure
  bool IsPressure() const { return type != "vector"; }
};

struct BCConfig
{
  std::vector<BoundaryCondition> dirichlet;
  std::vector<BoundaryCondition> traction;
};

// Body force per unit mass (rho0 b enters the weak form): an expression per
// component, with a schedule (same default rule as the boundary entries).
struct BodyForceConfig
{
  std::vector<std::string> expression;
  Schedule schedule;
  bool Empty() const { return expression.empty(); }
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

// Recovery from a load step whose Newton solve fails: restore the last
// converged state, halve the increment (at most max_bisections times, never
// below min_dt) and retry; after a converged reduced step the stepper aims at
// the planned breakpoint again (no growth beyond the planned grid).
struct SubstepConfig
{
  bool on_failure = false;
  int max_bisections = 4;
  double min_dt = 1e-4;
};

// The pseudo-time path: `breakpoints` are the targets t_1 < ... < t_n = 1 of
// the load steps (from `load_steps` equal increments or the `steps`
// segments); empty means load_steps equal increments (programmatic use).
struct SolverConfig
{
  int load_steps = 1;
  std::vector<double> breakpoints;
  // Initial guess of each increment. none: the last converged state with the
  // new Dirichlet data written on the boundary (the interior lags, so the
  // first Newton linearisation is about a state whose boundary layer of
  // elements carries the whole increment). tangent: one linear solve about
  // the last converged state with the Dirichlet increment imposed on the
  // update, x = x_n + d - J(x_n)^{-1} (R(x_n) + J(x_n) d), which spreads the
  // increment through the body before Newton starts.
  std::string predictor = "none";
  SubstepConfig substep;
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
  bool probe_every_step = false;     // also after every load step, prefixed by "step k t = ..."
  // Resultant force and moment (about the origin, current positions) that
  // each Dirichlet entry exerts on the body, after every step and at the end.
  bool reactions = false;
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
  BodyForceConfig body_force;
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
BodyForceConfig ParseBodyForceConfig(const YAML::Node &node, const std::string &path);
Schedule ParseSchedule(const YAML::Node &node, const std::string &path);
SolverConfig ParseSolverConfig(const YAML::Node &node, const std::string &path);
OutputConfig ParseOutputConfig(const YAML::Node &node, const std::string &path);

} // namespace cmf

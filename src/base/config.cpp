#include "base/config.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>

namespace cmf
{

namespace
{

template <typename T> const char *TypeName();
template <> const char *TypeName<int>() { return "an integer"; }
template <> const char *TypeName<double>() { return "a number"; }
template <> const char *TypeName<bool>() { return "a boolean"; }
template <> const char *TypeName<std::string>() { return "a string"; }
template <> const char *TypeName<std::vector<double>>() { return "a list of numbers"; }
template <> const char *TypeName<std::vector<std::string>>() { return "a list of strings"; }

std::string Describe(const YAML::Node &node)
{
  if (node.IsScalar()) { return "'" + node.Scalar() + "'"; }
  if (node.IsSequence()) { return "a list"; }
  if (node.IsMap()) { return "a map"; }
  return "null";
}

// Reads one YAML map while tracking the key path for error messages and the
// keys consumed, so unknown keys are reported instead of silently ignored.
class NodeReader
{
public:
  NodeReader(const YAML::Node &node, const std::string &path)
    : node_(node), path_(path)
  {
    if (!node_.IsDefined() || node_.IsNull())
    {
      throw ConfigError("missing section '" + path_ + "'");
    }
    if (!node_.IsMap())
    {
      throw ConfigError("'" + path_ + "' must be a map, got " +
                        Describe(node_));
    }
  }

  std::string Path(const std::string &key) const
  {
    return path_.empty() ? key : path_ + "." + key;
  }

  bool Has(const std::string &key) const
  {
    return node_[key].IsDefined() && !node_[key].IsNull();
  }

  YAML::Node Raw(const std::string &key)
  {
    used_.insert(key);
    return node_[key];
  }

  template <typename T> T Require(const std::string &key)
  {
    if (!Has(key)) { throw ConfigError("missing key '" + Path(key) + "'"); }
    return Convert<T>(key);
  }

  template <typename T> T Optional(const std::string &key, const T &fallback)
  {
    if (!Has(key)) { used_.insert(key); return fallback; }
    return Convert<T>(key);
  }

  void Finish() const
  {
    for (const auto &kv : node_)
    {
      const std::string key = kv.first.as<std::string>();
      if (!used_.count(key))
      {
        throw ConfigError("unknown key '" + Path(key) + "'");
      }
    }
  }

private:
  template <typename T> T Convert(const std::string &key)
  {
    used_.insert(key);
    try
    {
      return node_[key].as<T>();
    }
    catch (const YAML::Exception &)
    {
      throw ConfigError("key '" + Path(key) + "' expected " + TypeName<T>() +
                        ", got " + Describe(node_[key]));
    }
  }

  YAML::Node node_;
  std::string path_;
  std::set<std::string> used_;
};

std::vector<BoundaryCondition> ParseBCList(const YAML::Node &node,
                                           const std::string &path)
{
  std::vector<BoundaryCondition> list;
  if (!node.IsDefined() || node.IsNull()) { return list; }
  if (!node.IsSequence())
  {
    throw ConfigError("'" + path + "' must be a list of {attr, value} maps");
  }
  for (std::size_t i = 0; i < node.size(); i++)
  {
    const std::string item_path = path + "[" + std::to_string(i) + "]";
    NodeReader item(node[i], item_path);
    BoundaryCondition bc;
    // attr: a list of attribute numbers and/or physical-group names.
    if (!item.Has("attr")) { throw ConfigError("missing key '" + item_path + ".attr'"); }
    YAML::Node attr = item.Raw("attr");
    if (!attr.IsSequence() || attr.size() == 0)
    {
      throw ConfigError("key '" + item_path + ".attr' must be a non-empty list of "
                        "boundary attribute numbers or physical-group names");
    }
    for (std::size_t k = 0; k < attr.size(); k++)
    {
      if (!attr[k].IsScalar())
      {
        throw ConfigError("key '" + item_path + ".attr[" + std::to_string(k) +
                          "]' must be an attribute number or a physical-group name");
      }
      try { bc.attr.push_back(attr[k].as<int>()); }
      catch (const YAML::Exception &) { bc.attr_names.push_back(attr[k].as<std::string>()); }
    }
    bc.value = item.Require<std::vector<double>>("value");
    if (item.Has("gradient"))
    {
      const std::string gpath = item_path + ".gradient";
      YAML::Node g = item.Raw("gradient");
      if (!g.IsSequence())
      {
        throw ConfigError("key '" + gpath + "' must be a list of rows (data = value + gradient X)");
      }
      for (std::size_t r = 0; r < g.size(); r++)
      {
        std::vector<double> row;
        try { row = g[r].as<std::vector<double>>(); }
        catch (const YAML::Exception &) { row.clear(); }
        if (row.empty())
        {
          throw ConfigError("key '" + gpath + "[" + std::to_string(r) +
                            "]' must be a list of numbers");
        }
        bc.gradient.push_back(row);
      }
    }
    else { item.Optional<int>("gradient", 0); }
    for (int a : bc.attr)
    {
      if (a < 1)
      {
        throw ConfigError("key '" + item_path + ".attr' has attribute " +
                          std::to_string(a) + "; attributes start at 1");
      }
    }
    item.Finish();
    list.push_back(bc);
  }
  return list;
}

void CheckPositive(double v, const std::string &path)
{
  if (!(v > 0.0))
  {
    throw ConfigError("key '" + path + "' must be positive, got " +
                      std::to_string(v));
  }
}

void CheckNonNegative(int v, const std::string &path)
{
  if (v < 0)
  {
    throw ConfigError("key '" + path + "' must be non-negative, got " +
                      std::to_string(v));
  }
}

} // namespace

MeshConfig ParseMeshConfig(const YAML::Node &node, const std::string &path)
{
  NodeReader r(node, path);
  MeshConfig cfg;
  for (const char *key : {"cartesian", "corners"})
  {
    if (node[key].IsDefined())
    {
      throw ConfigError("key '" + r.Path(key) + "' is not supported: meshes are read from "
                        "Gmsh files (mesh.file, physical groups -> attributes); see "
                        "apps/mesh/*.geo and 'make meshes'");
    }
  }
  cfg.file = r.Require<std::string>("file");
  cfg.serial_refine = r.Optional<int>("serial_refine", 0);
  cfg.parallel_refine = r.Optional<int>("parallel_refine", 0);
  cfg.order = r.Optional<int>("order", 1);
  cfg.perturb = r.Optional<double>("perturb", 0.0);
  CheckNonNegative(cfg.serial_refine, r.Path("serial_refine"));
  CheckNonNegative(cfg.parallel_refine, r.Path("parallel_refine"));
  if (cfg.order < 1)
  {
    throw ConfigError("key '" + r.Path("order") + "' must be >= 1");
  }
  // Interior offsets of at most perturb*h per coordinate keep every corner
  // Jacobian positive only for perturb < 0.25.
  if (cfg.perturb < 0.0 || cfg.perturb >= 0.25)
  {
    throw ConfigError("key '" + r.Path("perturb") + "' must lie in [0, 0.25)");
  }

  r.Finish();
  return cfg;
}

void ValidateMaterialConfig(const MaterialConfig &cfg, const std::string &path)
{
  auto set = [](double v) { return !std::isnan(v); };
  auto key = [&](const std::string &k) { return "'" + path + "." + k + "'"; };
  const std::string &model = cfg.model;

  struct Key { const char *name; bool set; };
  const Key keys[] = {
    {"E", set(cfg.E)}, {"nu", set(cfg.nu)}, {"mu", set(cfg.mu)}, {"kappa", set(cfg.kappa)},
    {"c1", set(cfg.c1)}, {"c2", set(cfg.c2)}, {"c10", set(cfg.c10)}, {"c20", set(cfg.c20)},
    {"c30", set(cfg.c30)}, {"Jm", set(cfg.Jm)}, {"N", set(cfg.N)},
    {"mu_r", !cfg.mu_r.empty()}, {"alpha_r", !cfg.alpha_r.empty()},
    {"incompressible", cfg.incompressible}};
  auto is_set = [&](const std::string &k)
  {
    for (const Key &e : keys) { if (k == e.name) { return e.set; } }
    return false;
  };

  // The keys each model reads (any other key that is set is an error), the
  // ones it requires, and the phrase used in the missing-key message.
  std::vector<std::string> allowed, required;
  std::string needs;
  const bool coupled = model == "neo_hookean" || model == "st_venant_kirchhoff";
  if (coupled) { allowed = {"E", "nu"}; required = {"E", "nu"}; needs = "E and nu"; }
  else if (model == "iso_neo_hookean")
  {
    allowed = {"mu", "E", "nu", "kappa", "incompressible"};
    needs = "mu, or E and nu";
  }
  else if (model == "mooney_rivlin")
  {
    allowed = {"c1", "c2", "nu", "kappa", "incompressible"};
    required = {"c1", "c2"};
    needs = "c1 and c2";
  }
  else if (model == "yeoh")
  {
    allowed = {"c10", "c20", "c30", "nu", "kappa", "incompressible"};
    required = {"c10"};
    needs = "c10 (and optionally c20, c30)";
  }
  else if (model == "gent")
  {
    allowed = {"mu", "Jm", "nu", "kappa", "incompressible"};
    required = {"mu", "Jm"};
    needs = "mu and Jm";
  }
  else if (model == "arruda_boyce")
  {
    allowed = {"mu", "N", "nu", "kappa", "incompressible"};
    required = {"mu", "N"};
    needs = "mu and N";
  }
  else if (model == "ogden")
  {
    allowed = {"mu_r", "alpha_r", "nu", "kappa", "incompressible"};
    required = {"mu_r", "alpha_r"};
    needs = "mu_r and alpha_r";
  }
  else
  {
    throw ConfigError("key " + key("model") + ": unknown model '" + model + "'");
  }
  for (const std::string &k : required)
  {
    if (!is_set(k))
    {
      throw ConfigError("missing key " + key(k) + " (model '" + model + "' needs " + needs + ")");
    }
  }
  for (const Key &e : keys)
  {
    if (e.set && std::find(allowed.begin(), allowed.end(), e.name) == allowed.end())
    {
      throw ConfigError("key " + key(e.name) + " is not used by model '" + model + "'");
    }
  }

  if (coupled)
  {
    if (!(cfg.nu < 0.5))
    {
      throw ConfigError("key " + key("nu") + " must be < 0.5 for model '" + model +
                        "' (use a decoupled model with formulation: mixed)");
    }
    return;
  }
  if (model == "iso_neo_hookean")
  {
    if (set(cfg.mu) && set(cfg.E))
    {
      throw ConfigError("keys " + key("mu") + " and " + key("E") + ": give one, not both");
    }
    if (!set(cfg.mu) && !(set(cfg.E) && set(cfg.nu)))
    {
      throw ConfigError("missing key " + key("mu") + " (model 'iso_neo_hookean' needs mu, or E and nu)");
    }
  }
  if (model == "mooney_rivlin" && !(cfg.c1 + cfg.c2 > 0.0))
  {
    throw ConfigError("keys " + key("c1") + " + " + key("c2") + " must be positive (shear modulus 2 (c1 + c2))");
  }
  if (model == "yeoh" && !(cfg.c10 > 0.0))
  {
    throw ConfigError("key " + key("c10") + " must be positive (shear modulus 2 c10)");
  }
  if (model == "gent" && !(cfg.Jm > 0.0))
  {
    throw ConfigError("key " + key("Jm") + " must be positive (limiting value of I1 - 3)");
  }
  if (model == "arruda_boyce" && !(cfg.N > 0.0))
  {
    throw ConfigError("key " + key("N") + " must be positive (links per chain)");
  }
  if (model == "ogden")
  {
    if (cfg.mu_r.size() != cfg.alpha_r.size() || cfg.mu_r.size() > 6)
    {
      throw ConfigError("keys " + key("mu_r") + " and " + key("alpha_r") +
                        " must have the same length (1 to 6 terms)");
    }
    for (std::size_t r = 0; r < cfg.mu_r.size(); r++)
    {
      if (!(cfg.mu_r[r] * cfg.alpha_r[r] > 0.0))
      {
        throw ConfigError("keys " + key("mu_r") + "[" + std::to_string(r) + "] * " +
                          key("alpha_r") + "[" + std::to_string(r) + "] must be positive");
      }
    }
  }
  const int ways = int(set(cfg.kappa)) + int(set(cfg.nu)) + int(cfg.incompressible);
  if (ways != 1)
  {
    throw ConfigError("section '" + path + "': model '" + model + "' needs exactly one of " +
                      key("kappa") + ", " + key("nu") + " (0.5 = incompressible), or " +
                      key("incompressible") + ": true");
  }
}

MaterialConfig ParseMaterialConfig(const YAML::Node &node,
                                   const std::string &path)
{
  NodeReader r(node, path);
  MaterialConfig cfg;
  const double unset = std::numeric_limits<double>::quiet_NaN();
  cfg.model = r.Require<std::string>("model");
  cfg.E = r.Optional<double>("E", unset);
  cfg.nu = r.Optional<double>("nu", unset);
  cfg.mu = r.Optional<double>("mu", unset);
  cfg.kappa = r.Optional<double>("kappa", unset);
  cfg.c1 = r.Optional<double>("c1", unset);
  cfg.c2 = r.Optional<double>("c2", unset);
  cfg.c10 = r.Optional<double>("c10", unset);
  cfg.c20 = r.Optional<double>("c20", unset);
  cfg.c30 = r.Optional<double>("c30", unset);
  cfg.Jm = r.Optional<double>("Jm", unset);
  cfg.N = r.Optional<double>("N", unset);
  cfg.mu_r = r.Optional<std::vector<double>>("mu_r", {});
  cfg.alpha_r = r.Optional<std::vector<double>>("alpha_r", {});
  cfg.incompressible = r.Optional<bool>("incompressible", false);
  cfg.rho0 = r.Optional<double>("rho0", 1.0);
  static const char *models[] = {"neo_hookean", "st_venant_kirchhoff", "iso_neo_hookean",
                                 "mooney_rivlin", "yeoh", "gent", "arruda_boyce", "ogden"};
  bool known = false;
  for (const char *m : models) { known = known || cfg.model == m; }
  if (!known)
  {
    throw ConfigError("key '" + r.Path("model") + "': unknown model '" + cfg.model +
                      "' (expected neo_hookean, st_venant_kirchhoff, iso_neo_hookean, "
                      "mooney_rivlin, yeoh, gent, arruda_boyce, or ogden)");
  }
  if (!std::isnan(cfg.E)) { CheckPositive(cfg.E, r.Path("E")); }
  if (!std::isnan(cfg.mu)) { CheckPositive(cfg.mu, r.Path("mu")); }
  if (!std::isnan(cfg.kappa)) { CheckPositive(cfg.kappa, r.Path("kappa")); }
  if (!std::isnan(cfg.c10)) { CheckPositive(cfg.c10, r.Path("c10")); }
  if (!std::isnan(cfg.Jm)) { CheckPositive(cfg.Jm, r.Path("Jm")); }
  if (!std::isnan(cfg.N)) { CheckPositive(cfg.N, r.Path("N")); }
  CheckPositive(cfg.rho0, r.Path("rho0"));
  if (!std::isnan(cfg.nu) && !(cfg.nu > -1.0 && cfg.nu <= 0.5))
  {
    throw ConfigError("key '" + r.Path("nu") + "' must lie in (-1, 0.5], got " +
                      std::to_string(cfg.nu));
  }
  ValidateMaterialConfig(cfg, path);
  r.Finish();
  return cfg;
}

BCConfig ParseBCConfig(const YAML::Node &node, const std::string &path)
{
  BCConfig cfg;
  if (!node.IsDefined() || node.IsNull()) { return cfg; }
  NodeReader r(node, path);
  cfg.dirichlet = ParseBCList(r.Raw("dirichlet"), r.Path("dirichlet"));
  cfg.traction = ParseBCList(r.Raw("traction"), r.Path("traction"));
  r.Finish();
  return cfg;
}

SolverConfig ParseSolverConfig(const YAML::Node &node, const std::string &path)
{
  SolverConfig cfg;
  if (!node.IsDefined() || node.IsNull()) { return cfg; }
  NodeReader r(node, path);
  cfg.load_steps = r.Optional<int>("load_steps", 1);
  if (cfg.load_steps < 1)
  {
    throw ConfigError("key '" + r.Path("load_steps") + "' must be >= 1");
  }
  if (r.Has("newton"))
  {
    NodeReader n(r.Raw("newton"), r.Path("newton"));
    NewtonConfig &nc = cfg.newton;
    nc.rtol = n.Optional<double>("rtol", nc.rtol);
    nc.atol = n.Optional<double>("atol", nc.atol);
    nc.max_it = n.Optional<int>("max_it", nc.max_it);
    nc.armijo_c = n.Optional<double>("armijo_c", nc.armijo_c);
    nc.max_halvings = n.Optional<int>("max_halvings", nc.max_halvings);
    nc.print_level = n.Optional<int>("print_level", nc.print_level);
    if (!(nc.rtol >= 0.0) || !(nc.atol >= 0.0) || (nc.rtol == 0.0 && nc.atol == 0.0))
    {
      throw ConfigError("keys '" + n.Path("rtol") + "'/'atol' must be >= 0 "
                        "and not both zero");
    }
    if (nc.max_it < 1)
    {
      throw ConfigError("key '" + n.Path("max_it") + "' must be >= 1");
    }
    if (!(nc.armijo_c > 0.0 && nc.armijo_c < 1.0))
    {
      throw ConfigError("key '" + n.Path("armijo_c") + "' must lie in (0, 1)");
    }
    CheckNonNegative(nc.max_halvings, n.Path("max_halvings"));
    n.Finish();
  }
  else { r.Optional<int>("newton", 0); }
  if (r.Has("linear"))
  {
    NodeReader l(r.Raw("linear"), r.Path("linear"));
    LinearSolverConfig &lc = cfg.linear;
    lc.type = l.Optional<std::string>("type", lc.type);
    lc.amg = l.Optional<std::string>("amg", lc.amg);
    lc.rtol = l.Optional<double>("rtol", lc.rtol);
    lc.atol = l.Optional<double>("atol", lc.atol);
    lc.max_it = l.Optional<int>("max_it", lc.max_it);
    lc.krylov_dim = l.Optional<int>("krylov_dim", lc.krylov_dim);
    lc.print_level = l.Optional<int>("print_level", lc.print_level);
    lc.inner_rtol = l.Optional<double>("inner_rtol", lc.inner_rtol);
    lc.inner_max_it = l.Optional<int>("inner_max_it", lc.inner_max_it);
    lc.augmentation = l.Optional<double>("augmentation", lc.augmentation);
    if (!(lc.inner_rtol > 0.0 && lc.inner_rtol < 1.0) || lc.inner_max_it < 1)
    {
      throw ConfigError("keys '" + l.Path("inner_rtol") + "'/'inner_max_it' must be in (0,1) / >= 1");
    }
    if (!(lc.augmentation >= 0.0))
    {
      throw ConfigError("key '" + l.Path("augmentation") + "' must be >= 0");
    }
    if (lc.type != "gmres_amg" && lc.type != "cg_amg")
    {
      throw ConfigError("key '" + l.Path("type") + "': unknown type '" +
                        lc.type + "' (expected gmres_amg or cg_amg)");
    }
    if (lc.amg != "elasticity" && lc.amg != "systems")
    {
      throw ConfigError("key '" + l.Path("amg") + "': unknown option '" +
                        lc.amg + "' (expected elasticity or systems)");
    }
    if (!(lc.rtol >= 0.0) || !(lc.atol >= 0.0))
    {
      throw ConfigError("keys '" + l.Path("rtol") + "'/'atol' must be >= 0");
    }
    if (lc.max_it < 1 || lc.krylov_dim < 1)
    {
      throw ConfigError("keys '" + l.Path("max_it") + "'/'krylov_dim' must be >= 1");
    }
    l.Finish();
  }
  else { r.Optional<int>("linear", 0); }
  r.Finish();
  return cfg;
}

OutputConfig ParseOutputConfig(const YAML::Node &node, const std::string &path)
{
  OutputConfig cfg;
  if (!node.IsDefined() || node.IsNull()) { return cfg; }
  NodeReader r(node, path);
  cfg.paraview = r.Optional<std::string>("paraview", "");
  cfg.fields = r.Optional<std::vector<std::string>>(
    "fields", std::vector<std::string>{"displacement"});
  cfg.high_order = r.Optional<bool>("high_order", true);
  for (const std::string &f : cfg.fields)
  {
    if (f != "displacement" && f != "pressure" && f != "cauchy_stress" && f != "pk1_stress" &&
        f != "deformation_gradient" && f != "jacobian" && f != "vonmises" &&
        f != "energy_density" && f != "thickness_stretch")
    {
      throw ConfigError("key '" + r.Path("fields") + "': unknown field '" + f +
                        "' (expected displacement, pressure, cauchy_stress, pk1_stress, "
                        "deformation_gradient, jacobian, vonmises, energy_density, or "
                        "thickness_stretch)");
    }
  }
  cfg.quadrature_at = r.Optional<std::vector<std::string>>("quadrature_at", cfg.quadrature_at);
  if (cfg.quadrature_at.empty())
  {
    throw ConfigError("key '" + r.Path("quadrature_at") +
                      "' must list nodes, elements, and/or quadrature_points");
  }
  for (const std::string &where : cfg.quadrature_at)
  {
    if (where != "nodes" && where != "elements" && where != "quadrature_points")
    {
      throw ConfigError("key '" + r.Path("quadrature_at") + "': unknown presentation '" + where +
                        "' (expected nodes, elements, or quadrature_points)");
    }
  }
  cfg.nodal_projection = r.Optional<std::string>("nodal_projection", cfg.nodal_projection);
  if (cfg.nodal_projection != "averaged" && cfg.nodal_projection != "projected")
  {
    throw ConfigError("key '" + r.Path("nodal_projection") + "': unknown value '" +
                      cfg.nodal_projection + "' (expected averaged or projected)");
  }
  if (r.Has("probes"))
  {
    YAML::Node probes = r.Raw("probes");
    const std::string ppath = r.Path("probes");
    if (!probes.IsSequence())
    {
      throw ConfigError("'" + ppath + "' must be a list of {name, point} maps");
    }
    for (std::size_t i = 0; i < probes.size(); i++)
    {
      NodeReader item(probes[i], ppath + "[" + std::to_string(i) + "]");
      ProbeConfig probe;
      probe.name = item.Require<std::string>("name");
      probe.point = item.Require<std::vector<double>>("point");
      if (probe.point.size() < 1 || probe.point.size() > 3)
      {
        throw ConfigError("key '" + item.Path("point") + "' must have 1 to 3 coordinates");
      }
      item.Finish();
      cfg.probes.push_back(probe);
    }
  }
  else { r.Optional<int>("probes", 0); }
  r.Finish();
  return cfg;
}

AppConfig ParseConfig(const YAML::Node &root)
{
  if (!root.IsDefined() || root.IsNull() || !root.IsMap())
  {
    throw ConfigError("input must be a YAML map with a 'mesh' section");
  }
  NodeReader r(root, "");
  AppConfig cfg;
  cfg.formulation = r.Optional<std::string>("formulation", "displacement");
  if (cfg.formulation != "displacement" && cfg.formulation != "mixed")
  {
    throw ConfigError("key 'formulation': unknown value '" + cfg.formulation +
                      "' (expected displacement or mixed)");
  }
  cfg.plane = r.Optional<std::string>("plane", "strain");
  if (cfg.plane != "strain" && cfg.plane != "stress")
  {
    throw ConfigError("key 'plane': unknown value '" + cfg.plane +
                      "' (expected strain or stress)");
  }
  if (cfg.plane == "stress" && cfg.formulation == "mixed")
  {
    throw ConfigError("key 'plane': stress needs formulation: displacement (the "
                      "thickness stretch and the pressure are eliminated pointwise)");
  }
  cfg.mesh = ParseMeshConfig(r.Raw("mesh"), "mesh");
  cfg.material = ParseMaterialConfig(r.Raw("material"), "material");
  cfg.bcs = ParseBCConfig(r.Raw("bcs"), "bcs");
  cfg.body_force = r.Optional<std::vector<double>>("body_force", {});
  cfg.solver = ParseSolverConfig(r.Raw("solver"), "solver");
  cfg.output = ParseOutputConfig(r.Raw("output"), "output");
  r.Finish();
  return cfg;
}

AppConfig LoadConfig(const std::string &path)
{
  if (path.empty()) { throw ConfigError("no input file given (use -i)"); }
  std::ifstream in(path);
  if (!in) { throw ConfigError("cannot open input file '" + path + "'"); }
  YAML::Node root;
  try
  {
    root = YAML::Load(in);
  }
  catch (const YAML::Exception &e)
  {
    throw ConfigError("YAML syntax error in '" + path + "': " + e.what());
  }
  try
  {
    return ParseConfig(root);
  }
  catch (const ConfigError &e)
  {
    throw ConfigError(std::string(e.what()) + " (in '" + path + "')");
  }
}

} // namespace cmf

#include "base/config.hpp"

#include "base/expression.hpp"

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

// Ramp s = t, unless the data already depends on t (then constant).
Schedule DefaultSchedule(const std::vector<std::string> &expression)
{
  for (const std::string &e : expression)
  {
    if (Expression::Parse(e).UsesTime()) { return Schedule::Constant(); }
  }
  return Schedule::Ramp();
}

std::vector<BoundaryCondition> ParseBCList(const YAML::Node &node,
                                           const std::string &path, bool dirichlet)
{
  std::vector<BoundaryCondition> list;
  if (!node.IsDefined() || node.IsNull()) { return list; }
  if (!node.IsSequence())
  {
    throw ConfigError("'" + path + "' must be a list of {attr, expression} maps");
  }
  for (std::size_t i = 0; i < node.size(); i++)
  {
    const std::string item_path = path + "[" + std::to_string(i) + "]";
    NodeReader item(node[i], item_path);
    BoundaryCondition bc;
    bc.name = item.Optional<std::string>("name", (dirichlet ? "dirichlet[" : "traction[") +
                                                   std::to_string(i) + "]");
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
    if (!dirichlet)
    {
      bc.type = item.Optional<std::string>("type", "vector");
      if (bc.type != "vector" && bc.type != "pressure" && bc.type != "follower_pressure")
      {
        throw ConfigError("key '" + item_path + ".type': unknown type '" + bc.type +
                          "' (expected vector, pressure, or follower_pressure)");
      }
    }
    // Data: expression strings, one per component (one string for a pressure).
    if (!item.Has("expression"))
    {
      throw ConfigError("missing key '" + item_path + ".expression'");
    }
    {
      YAML::Node ex = item.Raw("expression");
      if (ex.IsScalar())
      {
        if (!bc.IsPressure())
        {
          throw ConfigError("key '" + item_path + ".expression' must be a list of one string per "
                            "component (a single string only for the pressure types)");
        }
        bc.expression = {ex.Scalar()};
      }
      else
      {
        try { bc.expression = ex.as<std::vector<std::string>>(); }
        catch (const YAML::Exception &) { bc.expression.clear(); }
        if (bc.expression.empty() || (bc.IsPressure() && bc.expression.size() != 1))
        {
          throw ConfigError("key '" + item_path + ".expression' must be " +
                            (bc.IsPressure() ? "a string" : "a list of strings, one per component"));
        }
      }
      for (std::size_t k = 0; k < bc.expression.size(); k++)
      {
        try { Expression::Parse(bc.expression[k]); }
        catch (const ConfigError &e)
        {
          throw ConfigError("key '" + item_path + ".expression[" + std::to_string(k) + "]': " + e.what());
        }
      }
    }
    if (item.Has("components"))
    {
      const std::string cpath = item_path + ".components";
      if (!dirichlet)
      {
        throw ConfigError("key '" + cpath + "' applies to Dirichlet entries only");
      }
      YAML::Node comps = item.Raw("components");
      if (!comps.IsSequence() || comps.size() == 0)
      {
        throw ConfigError("key '" + cpath + "' must be a non-empty list of x, y, z or 0, 1, 2");
      }
      for (std::size_t k = 0; k < comps.size(); k++)
      {
        int c = -1;
        if (comps[k].IsScalar())
        {
          const std::string name = comps[k].Scalar();
          if (name == "x" || name == "0") { c = 0; }
          else if (name == "y" || name == "1") { c = 1; }
          else if (name == "z" || name == "2") { c = 2; }
        }
        if (c < 0)
        {
          throw ConfigError("key '" + cpath + "[" + std::to_string(k) +
                            "]' must be x, y, z or 0, 1, 2, got " + Describe(comps[k]));
        }
        if (std::find(bc.components.begin(), bc.components.end(), c) != bc.components.end())
        {
          throw ConfigError("key '" + cpath + "' lists component " + std::to_string(c) + " twice");
        }
        bc.components.push_back(c);
      }
    }
    else { item.Optional<int>("components", 0); }
    if (item.Has("schedule")) { bc.schedule = ParseSchedule(item.Raw("schedule"), item_path + ".schedule"); }
    else
    {
      item.Optional<int>("schedule", 0);
      bc.schedule = DefaultSchedule(bc.expression);
    }
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

double Schedule::Eval(double time) const
{
  switch (kind)
  {
    case Kind::Ramp:
      if (time <= from) { return 0.0; }
      if (time >= to) { return 1.0; }
      return (time - from) / (to - from);
    case Kind::Constant:
      return time > 0.0 ? 1.0 : 0.0;
    case Kind::Table:
      if (time <= t.front()) { return s.front(); }
      if (time >= t.back()) { return s.back(); }
      for (std::size_t i = 1; i < t.size(); i++)
      {
        if (time <= t[i])
        {
          const double w = (time - t[i - 1]) / (t[i] - t[i - 1]);
          return s[i - 1] + w * (s[i] - s[i - 1]);
        }
      }
      return s.back();
  }
  return 0.0;
}

Schedule Schedule::Ramp(double from, double to)
{
  Schedule sch;
  sch.kind = Kind::Ramp;
  sch.from = from;
  sch.to = to;
  return sch;
}

Schedule Schedule::Constant()
{
  Schedule sch;
  sch.kind = Kind::Constant;
  return sch;
}

Schedule Schedule::Table(const std::vector<double> &t, const std::vector<double> &s)
{
  Schedule sch;
  sch.kind = Kind::Table;
  sch.t = t;
  sch.s = s;
  return sch;
}

// schedule: { type: ramp, from: 0.0, to: 1.0 } | { type: constant }
//         | { type: table, t: [..], s: [..] }
Schedule ParseSchedule(const YAML::Node &node, const std::string &path)
{
  NodeReader r(node, path);
  Schedule sch;
  const std::string type = r.Require<std::string>("type");
  if (type == "ramp")
  {
    sch = Schedule::Ramp(r.Optional<double>("from", 0.0), r.Optional<double>("to", 1.0));
    if (!(sch.from >= 0.0 && sch.to <= 1.0 && sch.from < sch.to))
    {
      throw ConfigError("keys '" + r.Path("from") + "'/'to' must satisfy 0 <= from < to <= 1");
    }
  }
  else if (type == "constant") { sch = Schedule::Constant(); }
  else if (type == "table")
  {
    sch = Schedule::Table(r.Require<std::vector<double>>("t"), r.Require<std::vector<double>>("s"));
    if (sch.t.size() < 2 || sch.t.size() != sch.s.size())
    {
      throw ConfigError("keys '" + r.Path("t") + "'/'s' must be lists of equal length >= 2");
    }
    for (std::size_t i = 0; i < sch.t.size(); i++)
    {
      if (sch.t[i] < 0.0 || sch.t[i] > 1.0 || (i > 0 && !(sch.t[i] > sch.t[i - 1])))
      {
        throw ConfigError("key '" + r.Path("t") + "' must increase strictly within [0, 1]");
      }
    }
  }
  else
  {
    throw ConfigError("key '" + r.Path("type") + "': unknown schedule '" + type +
                      "' (expected ramp, constant, or table)");
  }
  r.Finish();
  return sch;
}

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
    {"incompressible", cfg.incompressible}, {"volumetric", cfg.volumetric != "quadratic"}};
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
    allowed = {"mu", "E", "nu", "kappa", "incompressible", "volumetric"};
    needs = "mu, or E and nu";
  }
  else if (model == "mooney_rivlin")
  {
    allowed = {"c1", "c2", "nu", "kappa", "incompressible", "volumetric"};
    required = {"c1", "c2"};
    needs = "c1 and c2";
  }
  else if (model == "yeoh")
  {
    allowed = {"c10", "c20", "c30", "nu", "kappa", "incompressible", "volumetric"};
    required = {"c10"};
    needs = "c10 (and optionally c20, c30)";
  }
  else if (model == "gent")
  {
    allowed = {"mu", "Jm", "nu", "kappa", "incompressible", "volumetric"};
    required = {"mu", "Jm"};
    needs = "mu and Jm";
  }
  else if (model == "arruda_boyce")
  {
    allowed = {"mu", "N", "nu", "kappa", "incompressible", "volumetric"};
    required = {"mu", "N"};
    needs = "mu and N";
  }
  else if (model == "ogden")
  {
    allowed = {"mu_r", "alpha_r", "nu", "kappa", "incompressible", "volumetric"};
    required = {"mu_r", "alpha_r"};
    needs = "mu_r and alpha_r";
  }
  else
  {
    throw ConfigError("key " + key("model") + ": unknown model '" + model + "'");
  }
  {
    static const char *laws[] = {"quadratic", "simo_taylor", "logarithmic", "j_log_j"};
    bool known = false;
    for (const char *l : laws) { known = known || cfg.volumetric == l; }
    if (!known)
    {
      throw ConfigError("key " + key("volumetric") + ": unknown volumetric law '" + cfg.volumetric +
                        "' (expected quadratic, simo_taylor, logarithmic, or j_log_j)");
    }
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

namespace
{

// The parameter keys of a material map, read over `base` (a region inherits
// what it does not give; the base starts from an empty config).
MaterialConfig ReadMaterialKeys(NodeReader &r, const std::string &path, const MaterialConfig &base,
                                bool region)
{
  MaterialConfig cfg = base;
  cfg.regions.clear();
  cfg.attr.clear();
  cfg.attr_names.clear();
  const double unset = std::numeric_limits<double>::quiet_NaN();
  if (region)
  {
    const std::string model = r.Optional<std::string>("model", base.model);
    if (model != base.model)
    {
      throw ConfigError("key '" + r.Path("model") + "': a region must use the base model '" +
                        base.model + "' (got '" + model + "')");
    }
    // A region choosing its own bulk-modulus specification replaces the base's.
    if (r.Has("kappa") || r.Has("nu") || r.Has("incompressible"))
    {
      cfg.kappa = unset;
      cfg.nu = unset;
      cfg.incompressible = false;
    }
  }
  else { cfg.model = r.Require<std::string>("model"); }
  cfg.E = r.Optional<double>("E", cfg.E);
  cfg.nu = r.Optional<double>("nu", cfg.nu);
  cfg.mu = r.Optional<double>("mu", cfg.mu);
  cfg.kappa = r.Optional<double>("kappa", cfg.kappa);
  cfg.c1 = r.Optional<double>("c1", cfg.c1);
  cfg.c2 = r.Optional<double>("c2", cfg.c2);
  cfg.c10 = r.Optional<double>("c10", cfg.c10);
  cfg.c20 = r.Optional<double>("c20", cfg.c20);
  cfg.c30 = r.Optional<double>("c30", cfg.c30);
  cfg.Jm = r.Optional<double>("Jm", cfg.Jm);
  cfg.N = r.Optional<double>("N", cfg.N);
  cfg.mu_r = r.Optional<std::vector<double>>("mu_r", cfg.mu_r);
  cfg.alpha_r = r.Optional<std::vector<double>>("alpha_r", cfg.alpha_r);
  cfg.incompressible = r.Optional<bool>("incompressible", cfg.incompressible);
  cfg.volumetric = r.Optional<std::string>("volumetric", cfg.volumetric);
  cfg.rho0 = r.Optional<double>("rho0", cfg.rho0);
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
  return cfg;
}

} // namespace

MaterialConfig ParseMaterialConfig(const YAML::Node &node,
                                   const std::string &path)
{
  NodeReader r(node, path);
  MaterialConfig cfg = ReadMaterialKeys(r, path, MaterialConfig(), false);
  if (r.Has("regions"))
  {
    YAML::Node regions = r.Raw("regions");
    const std::string rpath = r.Path("regions");
    if (!regions.IsSequence() || regions.size() == 0)
    {
      throw ConfigError("'" + rpath + "' must be a non-empty list of {attr, <parameters>} maps");
    }
    for (std::size_t i = 0; i < regions.size(); i++)
    {
      const std::string item_path = rpath + "[" + std::to_string(i) + "]";
      NodeReader item(regions[i], item_path);
      if (!item.Has("attr")) { throw ConfigError("missing key '" + item_path + ".attr'"); }
      YAML::Node attr = item.Raw("attr");
      if (!attr.IsSequence() || attr.size() == 0)
      {
        throw ConfigError("key '" + item_path + ".attr' must be a non-empty list of "
                          "element attribute numbers or physical-volume names");
      }
      MaterialConfig region = ReadMaterialKeys(item, item_path, cfg, true);
      for (std::size_t k = 0; k < attr.size(); k++)
      {
        if (!attr[k].IsScalar())
        {
          throw ConfigError("key '" + item_path + ".attr[" + std::to_string(k) +
                            "]' must be an attribute number or a physical-volume name");
        }
        try { region.attr.push_back(attr[k].as<int>()); }
        catch (const YAML::Exception &) { region.attr_names.push_back(attr[k].as<std::string>()); }
      }
      if (region.incompressible != cfg.incompressible ||
          (!std::isnan(region.nu) && (region.nu == 0.5) != (!std::isnan(cfg.nu) && cfg.nu == 0.5)))
      {
        throw ConfigError("'" + item_path + "': every region must be incompressible or none");
      }
      item.Finish();
      cfg.regions.push_back(region);
    }
  }
  else { r.Optional<int>("regions", 0); }
  r.Finish();
  return cfg;
}

BCConfig ParseBCConfig(const YAML::Node &node, const std::string &path)
{
  BCConfig cfg;
  if (!node.IsDefined() || node.IsNull()) { return cfg; }
  NodeReader r(node, path);
  cfg.dirichlet = ParseBCList(r.Raw("dirichlet"), r.Path("dirichlet"), true);
  cfg.traction = ParseBCList(r.Raw("traction"), r.Path("traction"), false);
  r.Finish();
  return cfg;
}

// body_force: { expression: ["bx", "by", "bz"], schedule: {..} }.
BodyForceConfig ParseBodyForceConfig(const YAML::Node &node, const std::string &path)
{
  BodyForceConfig cfg;
  if (!node.IsDefined() || node.IsNull()) { return cfg; }
  if (!node.IsMap())
  {
    throw ConfigError("'" + path + "' must be a map { expression: [..], schedule: {..} }");
  }
  NodeReader r(node, path);
  cfg.expression = r.Require<std::vector<std::string>>("expression");
  for (std::size_t k = 0; k < cfg.expression.size(); k++)
  {
    try { Expression::Parse(cfg.expression[k]); }
    catch (const ConfigError &e)
    {
      throw ConfigError("key '" + path + ".expression[" + std::to_string(k) + "]': " + e.what());
    }
  }
  if (r.Has("schedule")) { cfg.schedule = ParseSchedule(r.Raw("schedule"), r.Path("schedule")); }
  else
  {
    r.Optional<int>("schedule", 0);
    cfg.schedule = DefaultSchedule(cfg.expression);
  }
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
  if (r.Has("steps"))
  {
    // steps: [ { to: t_1, n: n_1 }, ... ], t_k increasing, last t = 1.
    if (r.Has("load_steps"))
    {
      throw ConfigError("keys '" + r.Path("load_steps") + "' and '" + r.Path("steps") +
                        "': give one, not both");
    }
    YAML::Node steps = r.Raw("steps");
    const std::string spath = r.Path("steps");
    if (!steps.IsSequence() || steps.size() == 0)
    {
      throw ConfigError("'" + spath + "' must be a non-empty list of {to, n} maps");
    }
    double t_prev = 0.0;
    for (std::size_t i = 0; i < steps.size(); i++)
    {
      NodeReader seg(steps[i], spath + "[" + std::to_string(i) + "]");
      const double to = seg.Require<double>("to");
      const int n = seg.Require<int>("n");
      if (!(to > t_prev) || to > 1.0 + 1e-12)
      {
        throw ConfigError("key '" + seg.Path("to") + "' must increase and end at 1");
      }
      if (n < 1) { throw ConfigError("key '" + seg.Path("n") + "' must be >= 1"); }
      for (int k = 1; k <= n; k++)
      {
        cfg.breakpoints.push_back(t_prev + (to - t_prev) * double(k) / double(n));
      }
      t_prev = to;
      seg.Finish();
    }
    if (std::abs(t_prev - 1.0) > 1e-12)
    {
      throw ConfigError("'" + spath + "': the last segment must end at to: 1.0");
    }
    cfg.breakpoints.back() = 1.0;
    cfg.load_steps = int(cfg.breakpoints.size());
  }
  if (r.Has("substep"))
  {
    NodeReader ss(r.Raw("substep"), r.Path("substep"));
    SubstepConfig &sc = cfg.substep;
    sc.on_failure = ss.Optional<bool>("on_failure", true);
    sc.max_bisections = ss.Optional<int>("max_bisections", sc.max_bisections);
    sc.min_dt = ss.Optional<double>("min_dt", sc.min_dt);
    if (sc.max_bisections < 1)
    {
      throw ConfigError("key '" + ss.Path("max_bisections") + "' must be >= 1");
    }
    if (!(sc.min_dt > 0.0 && sc.min_dt <= 1.0))
    {
      throw ConfigError("key '" + ss.Path("min_dt") + "' must lie in (0, 1]");
    }
    ss.Finish();
  }
  else { r.Optional<int>("substep", 0); }
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
  cfg.probe_every_step = r.Optional<bool>("probe_every_step", false);
  cfg.reactions = r.Optional<bool>("reactions", false);
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
  cfg.body_force = ParseBodyForceConfig(r.Raw("body_force"), "body_force");
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

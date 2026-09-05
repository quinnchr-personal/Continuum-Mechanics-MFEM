#include "base/config.hpp"

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
template <> const char *TypeName<std::vector<int>>() { return "a list of integers"; }
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
    bc.attr = item.Require<std::vector<int>>("attr");
    bc.value = item.Require<std::vector<double>>("value");
    if (bc.attr.empty())
    {
      throw ConfigError("key '" + item_path + ".attr' must not be empty");
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

MeshConfig ParseMeshConfig(const YAML::Node &node, const std::string &path)
{
  NodeReader r(node, path);
  MeshConfig cfg;
  cfg.file = r.Optional<std::string>("file", "");
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

  if (r.Has("cartesian"))
  {
    cfg.cartesian = true;
    NodeReader c(r.Raw("cartesian"), r.Path("cartesian"));
    CartesianMeshConfig &box = cfg.box;
    box.nx = c.Require<int>("nx");
    box.ny = c.Require<int>("ny");
    box.sx = c.Optional<double>("sx", 1.0);
    box.sy = c.Optional<double>("sy", 1.0);
    const bool three_d = c.Has("nz");
    box.dim = three_d ? 3 : 2;
    box.nz = c.Optional<int>("nz", 1);
    box.sz = c.Optional<double>("sz", 1.0);
    box.element = c.Optional<std::string>("element", three_d ? "hex" : "quad");
    if (box.nx < 1 || box.ny < 1 || box.nz < 1)
    {
      throw ConfigError("keys '" + c.Path("nx") + "', 'ny', 'nz' must be >= 1");
    }
    CheckPositive(box.sx, c.Path("sx"));
    CheckPositive(box.sy, c.Path("sy"));
    CheckPositive(box.sz, c.Path("sz"));
    const bool ok2 = !three_d && (box.element == "quad" || box.element == "tri");
    const bool ok3 = three_d && (box.element == "hex" || box.element == "tet");
    if (!ok2 && !ok3)
    {
      throw ConfigError("key '" + c.Path("element") + "': unknown element '" +
                        box.element + "' (expected quad/tri in 2D, hex/tet in 3D)");
    }
    c.Finish();
  }
  if (cfg.file.empty() && !cfg.cartesian)
  {
    throw ConfigError("section '" + path +
                      "' needs either 'file' or 'cartesian'");
  }
  if (!cfg.file.empty() && cfg.cartesian)
  {
    throw ConfigError("section '" + path +
                      "' has both 'file' and 'cartesian'; give one");
  }

  if (r.Has("corners"))
  {
    const std::string cpath = r.Path("corners");
    YAML::Node corners = r.Raw("corners");
    if (!corners.IsSequence() || corners.size() != 4)
    {
      throw ConfigError("key '" + cpath + "' must be a list of 4 [x, y] points");
    }
    if (!cfg.cartesian || cfg.box.dim != 2)
    {
      throw ConfigError("key '" + cpath + "' requires a 2D cartesian mesh");
    }
    for (std::size_t i = 0; i < 4; i++)
    {
      std::vector<double> p;
      try { p = corners[i].as<std::vector<double>>(); }
      catch (const YAML::Exception &) { p.clear(); }
      if (p.size() != 2)
      {
        throw ConfigError("key '" + cpath + "[" + std::to_string(i) +
                          "]' must be an [x, y] point");
      }
      cfg.corners.push_back({p[0], p[1]});
    }
  }
  r.Finish();
  return cfg;
}

MaterialConfig ParseMaterialConfig(const YAML::Node &node,
                                   const std::string &path)
{
  NodeReader r(node, path);
  MaterialConfig cfg;
  cfg.model = r.Require<std::string>("model");
  cfg.E = r.Require<double>("E");
  cfg.nu = r.Require<double>("nu");
  cfg.rho0 = r.Optional<double>("rho0", 1.0);
  if (cfg.model != "neo_hookean" && cfg.model != "st_venant_kirchhoff")
  {
    throw ConfigError("key '" + r.Path("model") + "': unknown model '" +
                      cfg.model +
                      "' (expected neo_hookean or st_venant_kirchhoff)");
  }
  CheckPositive(cfg.E, r.Path("E"));
  CheckPositive(cfg.rho0, r.Path("rho0"));
  if (!(cfg.nu > -1.0 && cfg.nu < 0.5))
  {
    throw ConfigError("key '" + r.Path("nu") + "' must lie in (-1, 0.5), got " +
                      std::to_string(cfg.nu));
  }
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
    if (f != "displacement" && f != "vonmises" && f != "jacobian")
    {
      throw ConfigError("key '" + r.Path("fields") + "': unknown field '" + f +
                        "' (expected displacement, vonmises, or jacobian)");
    }
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

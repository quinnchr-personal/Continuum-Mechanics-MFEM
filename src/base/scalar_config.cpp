#include "base/scalar_config.hpp"

#include <fstream>

#include "base/expression.hpp"
#include "base/node_reader.hpp"

namespace cmf
{

namespace
{

void CheckExpression(const std::string &text, const std::string &path)
{
  try { Expression::Parse(text); }
  catch (const ConfigError &e) { throw ConfigError("key '" + path + "': " + e.what()); }
}

// A number, or a map { value, slope, reference }.
LawConfig ParseLaw(NodeReader &r, const std::string &key, const LawConfig &fallback)
{
  LawConfig law = fallback;
  if (!r.Has(key)) { r.Consume(key); return law; }
  YAML::Node node = r.Raw(key);
  if (node.IsScalar())
  {
    try { law.value = node.as<double>(); }
    catch (const YAML::Exception &)
    {
      throw ConfigError("key '" + r.Path(key) + "' expected a number or a map {value, slope, reference}, got " +
                        DescribeNode(node));
    }
    law.slope = 0.0;
    return law;
  }
  NodeReader l(node, r.Path(key));
  law.value = l.Require<double>("value");
  law.slope = l.Optional<double>("slope", 0.0);
  law.reference = l.Optional<double>("reference", 0.0);
  l.Finish();
  return law;
}

std::vector<ScalarCondition> ParseConditionList(const YAML::Node &node, const std::string &path,
                                                bool dirichlet, double t_final)
{
  std::vector<ScalarCondition> list;
  if (!node.IsDefined() || node.IsNull()) { return list; }
  if (!node.IsSequence())
  {
    throw ConfigError("'" + path + "' must be a list of {attr, expression} maps");
  }
  for (std::size_t i = 0; i < node.size(); i++)
  {
    const std::string item_path = path + "[" + std::to_string(i) + "]";
    NodeReader item(node[i], item_path);
    ScalarCondition c;
    c.name = item.Optional<std::string>("name", (dirichlet ? "dirichlet[" : "flux[") + std::to_string(i) + "]");
    if (item.Has("point"))
    {
      if (!dirichlet)
      {
        throw ConfigError("key '" + item_path + ".point' applies to Dirichlet entries only");
      }
      if (item.Has("attr"))
      {
        throw ConfigError("keys '" + item_path + ".attr' and '" + item_path + ".point': give one, not both");
      }
      c.point = item.Require<std::vector<double>>("point");
      if (c.point.size() < 2 || c.point.size() > 3)
      {
        throw ConfigError("key '" + item_path + ".point' must have 2 or 3 coordinates");
      }
    }
    else { ReadAttributeList(item, item_path, c.attr, c.attr_names); }
    c.expression = item.Require<std::string>("expression");
    CheckExpression(c.expression, item_path + ".expression");
    if (item.Has("schedule"))
    {
      c.schedule = ParseSchedule(item.Raw("schedule"), item_path + ".schedule", t_final);
    }
    else
    {
      item.Consume("schedule");
      c.schedule = DefaultSchedule({c.expression}, t_final);
    }
    item.Finish();
    list.push_back(c);
  }
  return list;
}

TransportConfig ParseTransportConfig(const YAML::Node &node, const std::string &path)
{
  NodeReader r(node, path);
  TransportConfig cfg;
  cfg.unknown = r.Optional<std::string>("unknown", cfg.unknown);
  if (cfg.unknown.empty() || cfg.unknown.find_first_of(" \t/") != std::string::npos)
  {
    throw ConfigError("key '" + r.Path("unknown") + "' must be a name without spaces or slashes");
  }
  cfg.capacity = ParseLaw(r, "capacity", cfg.capacity);
  cfg.conductivity = ParseLaw(r, "conductivity", cfg.conductivity);
  if (!(cfg.conductivity.value > 0.0))
  {
    throw ConfigError("key '" + r.Path("conductivity") + "' must be positive at the reference");
  }
  if (!(cfg.capacity.value > 0.0))
  {
    throw ConfigError("key '" + r.Path("capacity") + "' must be positive at the reference");
  }
  if (r.Has("velocity"))
  {
    try { cfg.velocity = r.Raw("velocity").as<std::vector<std::string>>(); }
    catch (const YAML::Exception &)
    {
      throw ConfigError("key '" + r.Path("velocity") + "' must be a list of strings, one per space dimension");
    }
    if (cfg.velocity.empty() || cfg.velocity.size() > 3)
    {
      throw ConfigError("key '" + r.Path("velocity") + "' must have 1 to 3 components");
    }
    for (std::size_t k = 0; k < cfg.velocity.size(); k++)
    {
      CheckExpression(cfg.velocity[k], r.Path("velocity") + "[" + std::to_string(k) + "]");
    }
  }
  else { r.Consume("velocity"); }
  cfg.convection = r.Optional<std::string>("convection", cfg.convection);
  if (cfg.convection != "nonconservative" && cfg.convection != "conservative")
  {
    throw ConfigError("key '" + r.Path("convection") + "': expected nonconservative or conservative, got '" +
                      cfg.convection + "'");
  }
  cfg.reaction = r.Optional<double>("reaction", 0.0);
  if (r.Has("source"))
  {
    cfg.source = r.Require<std::string>("source");
    CheckExpression(cfg.source, r.Path("source"));
  }
  else { r.Consume("source"); }
  cfg.quadrature_order = r.Optional<int>("quadrature_order", 0);
  if (cfg.quadrature_order < 0)
  {
    throw ConfigError("key '" + r.Path("quadrature_order") + "' must be >= 0 (0: the default 2 k + 3)");
  }
  r.Finish();
  return cfg;
}

// The output section: the solid's keys through ParseOutputConfig (with the
// scalar-only keys and the field list removed from a copy of the node), the
// field list validated against the scalar names, exact and flows.
void ParseScalarOutput(const YAML::Node &node, const std::string &path, ScalarAppConfig &cfg)
{
  if (!node.IsDefined() || node.IsNull()) { cfg.output.fields = {cfg.transport.unknown}; return; }
  if (!node.IsMap()) { throw ConfigError("'" + path + "' must be a map, got " + DescribeNode(node)); }
  for (const char *key : {"reactions", "energy"})
  {
    if (node[key].IsDefined())
    {
      throw ConfigError("key '" + path + "." + key + "' belongs to the solid schema (the flow through a Dirichlet "
                        "entry is output.flows)");
    }
  }
  YAML::Node copy = YAML::Clone(node);
  std::vector<std::string> fields;
  if (copy["fields"].IsDefined())
  {
    try { fields = copy["fields"].as<std::vector<std::string>>(); }
    catch (const YAML::Exception &)
    {
      throw ConfigError("key '" + path + ".fields' expected a list of strings, got " + DescribeNode(copy["fields"]));
    }
    copy.remove("fields");
  }
  else { fields = {cfg.transport.unknown}; }
  if (copy["exact"].IsDefined())
  {
    try { cfg.exact = copy["exact"].as<std::string>(); }
    catch (const YAML::Exception &)
    {
      throw ConfigError("key '" + path + ".exact' expected a string, got " + DescribeNode(copy["exact"]));
    }
    CheckExpression(cfg.exact, path + ".exact");
    copy.remove("exact");
  }
  if (copy["flows"].IsDefined())
  {
    try { cfg.flows = copy["flows"].as<bool>(); }
    catch (const YAML::Exception &)
    {
      throw ConfigError("key '" + path + ".flows' expected a boolean, got " + DescribeNode(copy["flows"]));
    }
    copy.remove("flows");
  }
  cfg.output = ParseOutputConfig(copy, path, false);
  const std::string &u = cfg.transport.unknown;
  for (const std::string &f : fields)
  {
    if (f == u || f == "flux") { continue; }
    if (f == u + "_exact" || f == u + "_error")
    {
      if (cfg.exact.empty())
      {
        throw ConfigError("key '" + path + ".fields': field '" + f + "' needs " + path + ".exact");
      }
      continue;
    }
    throw ConfigError("key '" + path + ".fields': unknown field '" + f + "' (expected " + u + ", " + u +
                      "_exact, " + u + "_error, or flux)");
  }
  cfg.output.fields = fields;
}

} // namespace

ScalarAppConfig ParseScalarConfig(const YAML::Node &root)
{
  if (!root.IsDefined() || root.IsNull() || !root.IsMap())
  {
    throw ConfigError("input must be a YAML map with 'physics', 'mesh' and 'transport' sections");
  }
  NodeReader r(root, "");
  ScalarAppConfig cfg;
  const std::string physics = r.Optional<std::string>("physics", "");
  if (physics != "scalar_transport")
  {
    throw ConfigError(physics.empty()
                        ? std::string("missing key 'physics' (this executable needs physics: scalar_transport; "
                                      "solid mechanics inputs belong to build/apps/solid_mechanics)")
                        : "key 'physics': unknown value '" + physics + "' (expected scalar_transport)");
  }
  for (const char *key : {"formulation", "plane", "material", "body_force", "dynamics"})
  {
    if (r.Has(key))
    {
      throw ConfigError(std::string("key '") + key + "' belongs to the solid mechanics schema; "
                        "the scalar transport input has transport, initial, bcs.dirichlet and bcs.flux");
    }
  }
  cfg.mesh = ParseMeshConfig(r.Raw("mesh"), "mesh");
  cfg.transport = ParseTransportConfig(r.Raw("transport"), "transport");
  cfg.time = ParseTimeConfig(r.Raw("time"), "time");
  const double t_final = cfg.time.enabled ? cfg.time.t_final : 0.0;
  if (r.Has("initial"))
  {
    cfg.initial = r.Require<std::string>("initial");
    CheckExpression(cfg.initial, "initial");
    if (!cfg.time.enabled)
    {
      throw ConfigError("key 'initial' needs a time block (a steady problem has no initial state)");
    }
  }
  else { r.Consume("initial"); }
  if (r.Has("bcs"))
  {
    NodeReader b(r.Raw("bcs"), "bcs");
    cfg.bcs.dirichlet = ParseConditionList(b.Raw("dirichlet"), "bcs.dirichlet", true, t_final);
    cfg.bcs.flux = ParseConditionList(b.Raw("flux"), "bcs.flux", false, t_final);
    for (const char *key : {"traction", "contact", "temperature", "heat_flux"})
    {
      if (b.Has(key))
      {
        throw ConfigError(std::string("key 'bcs.") + key + "' belongs to the solid mechanics schema "
                          "(the scalar conditions are bcs.dirichlet and bcs.flux)");
      }
    }
    b.Finish();
  }
  else { r.Consume("bcs"); }
  cfg.solver = ParseSolverConfig(r.Raw("solver"), "solver", DynamicsConfig(), cfg.time);
  if (cfg.solver.linear.amg == "elasticity") { cfg.solver.linear.amg = "scalar"; } // the vector default
  if (cfg.solver.linear.amg != "scalar")
  {
    throw ConfigError("key 'solver.linear.amg': a scalar unknown takes scalar (got '" + cfg.solver.linear.amg + "')");
  }
  if (cfg.solver.linear.type == "cg_amg" && !cfg.transport.velocity.empty())
  {
    throw ConfigError("key 'solver.linear.type': cg_amg needs a symmetric operator, but a velocity makes "
                      "it non-symmetric; use gmres_amg or direct");
  }
  ParseScalarOutput(r.Raw("output"), "output", cfg);
  r.Finish();
  return cfg;
}

ScalarAppConfig LoadScalarConfig(const std::string &path)
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
    return ParseScalarConfig(root);
  }
  catch (const ConfigError &e)
  {
    throw ConfigError(std::string(e.what()) + " (in '" + path + "')");
  }
}

} // namespace cmf

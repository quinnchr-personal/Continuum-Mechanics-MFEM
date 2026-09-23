// Reader of one YAML map for the section parsers of the schemas
// (base/config.cpp, base/scalar_config.cpp): tracks the key path for error
// messages and the keys consumed, so that unknown keys are reported instead
// of silently ignored. Internal to the parsers; not part of the schema.
#pragma once

#include <set>
#include <string>
#include <vector>

#include "base/config.hpp"
#include "yaml-cpp/yaml.h"

namespace cmf
{

template <typename T> inline const char *TypeName();
template <> inline const char *TypeName<int>() { return "an integer"; }
template <> inline const char *TypeName<double>() { return "a number"; }
template <> inline const char *TypeName<bool>() { return "a boolean"; }
template <> inline const char *TypeName<std::string>() { return "a string"; }
template <> inline const char *TypeName<std::vector<double>>() { return "a list of numbers"; }
template <> inline const char *TypeName<std::vector<std::string>>() { return "a list of strings"; }

inline std::string DescribeNode(const YAML::Node &node)
{
  if (node.IsScalar()) { return "'" + node.Scalar() + "'"; }
  if (node.IsSequence()) { return "a list"; }
  if (node.IsMap()) { return "a map"; }
  return "null";
}

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
      throw ConfigError("'" + path_ + "' must be a map, got " + DescribeNode(node_));
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

  // Marks a key consumed without reading it (a key another parser handles).
  void Consume(const std::string &key) { used_.insert(key); }

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
                        ", got " + DescribeNode(node_[key]));
    }
  }

  YAML::Node node_;
  std::string path_;
  std::set<std::string> used_;
};

// attr: a non-empty list of boundary attribute numbers and/or physical-group
// names of the entry at item_path, split into the two lists.
inline void ReadAttributeList(NodeReader &item, const std::string &item_path,
                              std::vector<int> &attr, std::vector<std::string> &attr_names)
{
  if (!item.Has("attr")) { throw ConfigError("missing key '" + item_path + ".attr'"); }
  YAML::Node node = item.Raw("attr");
  if (!node.IsSequence() || node.size() == 0)
  {
    throw ConfigError("key '" + item_path + ".attr' must be a non-empty list of "
                      "boundary attribute numbers or physical-group names");
  }
  for (std::size_t k = 0; k < node.size(); k++)
  {
    if (!node[k].IsScalar())
    {
      throw ConfigError("key '" + item_path + ".attr[" + std::to_string(k) +
                        "]' must be an attribute number or a physical-group name");
    }
    try { attr.push_back(node[k].as<int>()); }
    catch (const YAML::Exception &) { attr_names.push_back(node[k].as<std::string>()); }
  }
  for (int a : attr)
  {
    if (a < 1)
    {
      throw ConfigError("key '" + item_path + ".attr' has attribute " + std::to_string(a) +
                        "; attributes start at 1");
    }
  }
}

} // namespace cmf

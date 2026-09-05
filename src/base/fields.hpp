// Named registry of grid functions: physics modules publish fields here and
// the output layer looks them up by the names given in the YAML input.
#pragma once

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mfem.hpp"

namespace cmf
{

class FieldRegistry
{
public:
  mfem::ParGridFunction &Add(const std::string &name,
                             std::unique_ptr<mfem::ParGridFunction> gf)
  {
    mfem::ParGridFunction &ref = *gf;
    owned_[name] = std::move(gf);
    fields_[name] = &ref;
    return ref;
  }

  void AddExternal(const std::string &name, mfem::ParGridFunction &gf)
  {
    fields_[name] = &gf;
  }

  bool Has(const std::string &name) const { return fields_.count(name) > 0; }

  mfem::ParGridFunction &Get(const std::string &name) const
  {
    auto it = fields_.find(name);
    if (it == fields_.end())
    {
      throw std::runtime_error("unknown field '" + name + "'");
    }
    return *it->second;
  }

  std::vector<std::string> Names() const
  {
    std::vector<std::string> names;
    for (const auto &kv : fields_) { names.push_back(kv.first); }
    return names;
  }

private:
  std::map<std::string, mfem::ParGridFunction *> fields_;
  std::map<std::string, std::unique_ptr<mfem::ParGridFunction>> owned_;
};

} // namespace cmf

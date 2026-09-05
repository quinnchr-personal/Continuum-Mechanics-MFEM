#include "base/output.hpp"

namespace cmf
{

namespace
{

std::string BaseName(const std::string &path)
{
  const auto slash = path.find_last_of('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::string DirName(const std::string &path)
{
  const auto slash = path.find_last_of('/');
  return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

} // namespace

ParaViewWriter::ParaViewWriter(const std::string &path, mfem::ParMesh &mesh,
                               int order, bool high_order)
  : dc_(BaseName(path), &mesh)
{
  dc_.SetPrefixPath(DirName(path));
  dc_.SetLevelsOfDetail(order);
  dc_.SetDataFormat(mfem::VTKFormat::BINARY);
  dc_.SetHighOrderOutput(high_order);
}

void ParaViewWriter::Register(const std::string &name, mfem::ParGridFunction &gf)
{
  dc_.RegisterField(name, &gf);
}

void ParaViewWriter::RegisterAll(const OutputConfig &cfg,
                                 const FieldRegistry &fields)
{
  for (const std::string &name : cfg.fields)
  {
    if (!fields.Has(name))
    {
      throw ConfigError("output.fields: field '" + name +
                        "' is not provided by this physics");
    }
    Register(name, fields.Get(name));
  }
}

void ParaViewWriter::Save(int cycle, double time)
{
  dc_.SetCycle(cycle);
  dc_.SetTime(time);
  dc_.Save();
}

} // namespace cmf

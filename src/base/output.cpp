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

void ParaViewWriter::RegisterQField(const std::string &name, mfem::QuadratureFunction &qf)
{
  dc_.RegisterQField(name, &qf);
}

void ParaViewWriter::RegisterAll(const OutputConfig &cfg,
                                 const FieldRegistry &fields)
{
  for (const std::string &name : cfg.fields)
  {
    // Nodal unknowns are registered under their name; quadrature quantities
    // under whichever presentations the physics created.
    bool found = false;
    for (const char *suffix : {"", "_elem"})
    {
      const std::string presented = name + suffix;
      if (fields.Has(presented))
      {
        Register(presented, fields.Get(presented));
        found = true;
      }
    }
    if (fields.HasQ(name + "_qp"))
    {
      RegisterQField(name + "_qp", fields.GetQ(name + "_qp"));
      found = true;
    }
    if (!found)
    {
      throw ConfigError("output.fields: field '" + name +
                        "' is not provided by this physics");
    }
  }
}

void ParaViewWriter::Save(int cycle, double time)
{
  dc_.SetCycle(cycle);
  dc_.SetTime(time);
  dc_.Save();
  if (dc_.Error() != mfem::DataCollection::No_Error)
  {
    dc_.ResetError();
    throw ConfigError("output.paraview: failed to write '" + dc_.GetPrefixPath() +
                      dc_.GetCollectionName() + "'");
  }
}

} // namespace cmf

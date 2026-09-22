#include "base/output.hpp"

#include <iomanip>

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

ReactionWriter::ReactionWriter(const std::string &collection_path, std::vector<std::string> names,
                               bool root)
  : path_(collection_path + "/reactions.csv"), names_(std::move(names)), root_(root)
{
}

void ReactionWriter::Append(int step, double t, const std::vector<std::array<double, 6>> &values)
{
  if (!root_) { return; }
  MFEM_VERIFY(values.size() == names_.size(), "ReactionWriter: one value set per named entry");
  if (!out_.is_open())
  {
    out_.open(path_);
    if (!out_) { throw ConfigError("output.reactions: cannot write '" + path_ + "'"); }
    out_ << "step,t";
    for (const std::string &n : names_)
    {
      for (const char *c : {"fx", "fy", "fz", "mx", "my", "mz"}) { out_ << "," << n << "_" << c; }
    }
    out_ << "\n";
  }
  out_ << step << "," << std::setprecision(17) << t;
  for (const auto &v : values)
  {
    for (double x : v) { out_ << "," << x; }
  }
  out_ << "\n" << std::flush;
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

// ParaView output of registered fields, and the CSV of the reactions beside it.
#pragma once

#include <array>
#include <fstream>
#include <string>
#include <vector>

#include "base/config.hpp"
#include "base/fields.hpp"
#include "mfem.hpp"

namespace cmf
{

class ParaViewWriter
{
public:
  // path is "<prefix>/<collection>"; order sets the level of detail.
  ParaViewWriter(const std::string &path, mfem::ParMesh &mesh, int order,
                 bool high_order = true);

  void Register(const std::string &name, mfem::ParGridFunction &gf);
  // Quadrature-point field: written per cycle as a point-cloud VTU per rank
  // (<name><rank>.vtu next to the mesh pieces) and listed in the same .pvd
  // as a part named `name`.
  void RegisterQField(const std::string &name, mfem::QuadratureFunction &qf);
  // Register every name in cfg.fields from the registry (throws on unknown):
  // the nodal unknown, or the presentations <name>, <name>_elem, <name>_qp
  // of a quadrature quantity.
  void RegisterAll(const OutputConfig &cfg, const FieldRegistry &fields);
  void Save(int cycle = 0, double time = 0.0);

private:
  mfem::ParaViewDataCollection dc_;
};

// The reactions of the Dirichlet entries per accepted step as a CSV next to
// the ParaView collection, <collection dir>/reactions.csv: the columns are
// step, t, then <name>_fx, _fy, _fz, _mx, _my, _mz for every entry (fz = 0 in
// 2D), one row per step including the initial state, so the rows line up with
// the cycles of the .pvd when output.every is 1. Only the root rank writes;
// every row is flushed, so a run that stops leaves the rows it has. The file is
// opened at the first row, after the collection directory exists.
class ReactionWriter
{
public:
  ReactionWriter(const std::string &collection_path, std::vector<std::string> names, bool root);
  // One row; `values` holds fx, fy, fz, mx, my, mz per entry, in the order of the names.
  void Append(int step, double t, const std::vector<std::array<double, 6>> &values);
  const std::string &Path() const { return path_; }

private:
  std::string path_;
  std::vector<std::string> names_;
  bool root_;
  std::ofstream out_;
};

// A CSV of named columns next to a ParaView collection (<collection dir>/<file>):
// the row of every accepted step, flushed as written; only the root rank
// writes. The file is opened at the first row, after the collection
// directory exists. The error histories and the flows of the scalar
// transport executable use it.
class CsvWriter
{
public:
  CsvWriter(const std::string &collection_path, const std::string &file, std::vector<std::string> columns,
            bool root);
  // One row: step, t, then one value per named column.
  void Append(int step, double t, const std::vector<double> &values);
  const std::string &Path() const { return path_; }

private:
  std::string path_;
  std::vector<std::string> columns_;
  bool root_;
  std::ofstream out_;
};

} // namespace cmf

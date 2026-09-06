// ParaView output of registered fields.
#pragma once

#include <string>

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

} // namespace cmf

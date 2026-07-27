#pragma once

#include "mfem.hpp"

#include <memory>
#include <string>

namespace hycfd
{

// Loads a 2D mesh from an MFEM or Gmsh file (format detected by MFEM's
// mesh reader). curved_order > 0 raises/projects the geometry to that
// order; 0 keeps the file's geometry.
std::unique_ptr<mfem::Mesh> LoadMeshFile(const std::string &path,
                                         int curved_order = 0);

} // namespace hycfd

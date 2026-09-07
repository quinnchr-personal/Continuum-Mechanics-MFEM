// Mesh construction from MeshConfig: file or Cartesian box, optional bilinear
// corner map and interior-vertex jitter, serial/parallel uniform refinement.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "base/config.hpp"
#include "mfem.hpp"

namespace cmf
{

// Serial mesh: load or generate, map corners, jitter, then serial_refine.
// Built in place and returned by pointer: MFEM's move construction and
// assignment swap everything except the attribute sets, which would drop
// the physical-group names of a Gmsh file.
std::unique_ptr<mfem::Mesh> BuildSerialMesh(const MeshConfig &cfg);

// Partition BuildSerialMesh() over comm and apply parallel_refine.
std::unique_ptr<mfem::ParMesh> BuildParMesh(MPI_Comm comm, const MeshConfig &cfg);

// Move every interior vertex by a deterministic pseudo-random offset of at
// most amplitude*h in each coordinate (h = smallest element size). Straight-
// sided meshes only.
void PerturbInteriorVertices(mfem::Mesh &mesh, double amplitude,
                             unsigned seed = 12345u);

// Boundary attributes of a boundary condition: numbers are checked against
// the mesh, physical-group names (Gmsh $PhysicalNames, kept by MFEM as
// boundary attribute sets) are resolved to their numbers. Errors list what the
// mesh provides. `what` names the YAML entry.
std::vector<int> ResolveBoundaryAttributes(mfem::Mesh &mesh, const BoundaryCondition &bc,
                                           const std::string &what);

// Element attributes of a material region (numbers checked, physical-volume
// names resolved through the element attribute sets), like the boundary case.
std::vector<int> ResolveElementAttributes(mfem::Mesh &mesh, const std::vector<int> &attr,
                                          const std::vector<std::string> &attr_names,
                                          const std::string &what);

// "1 (bottom), 2 (right), ..." for the boundary or the element attributes.
std::string DescribeAttributes(mfem::Mesh &mesh, bool boundary);

} // namespace cmf

// Mesh construction from MeshConfig: file or Cartesian box, optional bilinear
// corner map and interior-vertex jitter, serial/parallel uniform refinement.
#pragma once

#include <memory>

#include "base/config.hpp"
#include "mfem.hpp"

namespace cmf
{

// Serial mesh: load or generate, map corners, jitter, then serial_refine.
mfem::Mesh BuildSerialMesh(const MeshConfig &cfg);

// Partition BuildSerialMesh() over comm and apply parallel_refine.
std::unique_ptr<mfem::ParMesh> BuildParMesh(MPI_Comm comm, const MeshConfig &cfg);

// Move every interior vertex by a deterministic pseudo-random offset of at
// most amplitude*h in each coordinate (h = smallest element size). Straight-
// sided meshes only.
void PerturbInteriorVertices(mfem::Mesh &mesh, double amplitude,
                             unsigned seed = 12345u);

} // namespace cmf

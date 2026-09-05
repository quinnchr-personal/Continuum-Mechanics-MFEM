#include "base/probes.hpp"

#include <stdexcept>
#include <string>

namespace cmf
{

std::vector<double> ProbeVector(const mfem::ParGridFunction &gf,
                                const std::vector<double> &point)
{
  mfem::ParFiniteElementSpace &fes = *const_cast<mfem::ParGridFunction &>(gf).ParFESpace();
  mfem::ParMesh &mesh = *fes.GetParMesh();
  const int sdim = mesh.SpaceDimension();
  if (int(point.size()) != sdim)
  {
    throw std::runtime_error("ProbeVector: point has " + std::to_string(point.size()) +
                             " coordinates, mesh has " + std::to_string(sdim));
  }
  mfem::DenseMatrix pm(sdim, 1);
  for (int i = 0; i < sdim; i++) { pm(i, 0) = point[i]; }
  mfem::Array<int> elem;
  mfem::Array<mfem::IntegrationPoint> ips;
  mesh.FindPoints(pm, elem, ips, false);

  const int vdim = fes.GetVDim();
  std::vector<double> local(vdim, 0.0), global(vdim, 0.0);
  int count = 0, total = 0;
  if (elem[0] >= 0)
  {
    mfem::Vector v;
    gf.GetVectorValue(elem[0], ips[0], v);
    for (int i = 0; i < vdim; i++) { local[i] = v(i); }
    count = 1;
  }
  MPI_Allreduce(local.data(), global.data(), vdim, MPI_DOUBLE, MPI_SUM, fes.GetComm());
  MPI_Allreduce(&count, &total, 1, MPI_INT, MPI_SUM, fes.GetComm());
  if (total == 0)
  {
    std::string p;
    for (double x : point) { p += (p.empty() ? "" : ", ") + std::to_string(x); }
    throw std::runtime_error("ProbeVector: point (" + p + ") is not inside the mesh");
  }
  for (double &x : global) { x /= total; }
  return global;
}

} // namespace cmf

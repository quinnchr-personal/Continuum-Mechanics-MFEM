#include "base/mesh_input.hpp"

#include <algorithm>
#include <random>

namespace cmf
{

namespace
{

mfem::Element::Type ElementType(const CartesianMeshConfig &box)
{
  if (box.element == "quad") { return mfem::Element::QUADRILATERAL; }
  if (box.element == "tri") { return mfem::Element::TRIANGLE; }
  if (box.element == "hex") { return mfem::Element::HEXAHEDRON; }
  if (box.element == "tet") { return mfem::Element::TETRAHEDRON; }
  throw ConfigError("mesh.cartesian.element: unknown element '" + box.element + "'");
}

} // namespace

void PerturbInteriorVertices(mfem::Mesh &mesh, double amplitude, unsigned seed)
{
  if (amplitude <= 0.0) { return; }
  if (mesh.GetNodes())
  {
    throw ConfigError("mesh.perturb: only straight-sided meshes can be perturbed");
  }
  const int dim = mesh.Dimension();
  const int nv = mesh.GetNV();
  // Topological boundary: vertices of faces with a single adjacent element.
  // (Boundary elements can be incomplete for file meshes with partial
  // physical groups, so they are not used here.)
  mfem::Array<bool> on_boundary(nv);
  on_boundary = false;
  mfem::Array<int> verts;
  for (int f = 0; f < mesh.GetNumFaces(); f++)
  {
    int e1 = -1, e2 = -1;
    mesh.GetFaceElements(f, &e1, &e2);
    if (e2 >= 0) { continue; }
    mesh.GetFaceVertices(f, verts);
    for (int v : verts) { on_boundary[v] = true; }
  }
  double h = mesh.GetElementSize(0, 1);
  for (int e = 1; e < mesh.GetNE(); e++)
  {
    h = std::min(h, mesh.GetElementSize(e, 1));
  }
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  mfem::Vector displacement(nv * dim);
  displacement = 0.0;
  // MoveVertices reads component-major: d(i*nv + v) moves vertex v in
  // direction i (the GetVertices/SetVertices layout).
  for (int v = 0; v < nv; v++)
  {
    for (int i = 0; i < dim; i++)
    {
      const double r = unit(rng);
      if (!on_boundary[v]) { displacement(i * nv + v) = amplitude * h * r; }
    }
  }
  mesh.MoveVertices(displacement);
  const int inverted = mesh.CheckElementOrientation(false);
  if (inverted > 0)
  {
    throw ConfigError("mesh.perturb: " + std::to_string(inverted) +
                      " elements inverted; reduce the amplitude");
  }
}

mfem::Mesh BuildSerialMesh(const MeshConfig &cfg)
{
  mfem::Mesh mesh;
  if (!cfg.file.empty())
  {
    mesh = mfem::Mesh::LoadFromFile(cfg.file, 1, 1);
    if (mesh.GetNE() == 0)
    {
      throw ConfigError("mesh.file: '" + cfg.file + "' contains no elements");
    }
  }
  else
  {
    const CartesianMeshConfig &box = cfg.box;
    const mfem::Element::Type type = ElementType(box);
    if (box.dim == 2)
    {
      mesh = mfem::Mesh::MakeCartesian2D(box.nx, box.ny, type, true,
                                         box.sx, box.sy, false);
    }
    else
    {
      mesh = mfem::Mesh::MakeCartesian3D(box.nx, box.ny, box.nz, type,
                                         box.sx, box.sy, box.sz, false);
    }
  }

  if (!cfg.corners.empty())
  {
    const double sx = cfg.box.sx, sy = cfg.box.sy;
    const auto c = cfg.corners;
    mesh.Transform([sx, sy, c](const mfem::Vector &x, mfem::Vector &y)
    {
      const double xi = x(0) / sx, eta = x(1) / sy;
      y.SetSize(2);
      for (int i = 0; i < 2; i++)
      {
        y(i) = (1.0 - xi) * (1.0 - eta) * c[0][i] + xi * (1.0 - eta) * c[1][i]
               + xi * eta * c[2][i] + (1.0 - xi) * eta * c[3][i];
      }
    });
    if (mesh.CheckElementOrientation(false) > 0)
    {
      throw ConfigError("mesh.corners: the corner quadrilateral must be listed "
                        "counter-clockwise and be non-degenerate");
    }
  }

  // Jitter the base mesh, then refine uniformly: refinement levels share
  // one distortion pattern, as in a distorted-mesh convergence study.
  if (cfg.perturb > 0.0) { PerturbInteriorVertices(mesh, cfg.perturb); }
  for (int l = 0; l < cfg.serial_refine; l++) { mesh.UniformRefinement(); }
  return mesh;
}

std::unique_ptr<mfem::ParMesh> BuildParMesh(MPI_Comm comm, const MeshConfig &cfg)
{
  mfem::Mesh serial = BuildSerialMesh(cfg);
  auto pmesh = std::make_unique<mfem::ParMesh>(comm, serial);
  serial.Clear();
  for (int l = 0; l < cfg.parallel_refine; l++) { pmesh->UniformRefinement(); }
  return pmesh;
}

} // namespace cmf

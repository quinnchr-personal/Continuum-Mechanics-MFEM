#include "base/mesh_input.hpp"

#include <algorithm>
#include <map>
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

std::unique_ptr<mfem::Mesh> BuildSerialMesh(const MeshConfig &cfg)
{
  std::unique_ptr<mfem::Mesh> mesh_ptr;
  if (!cfg.file.empty())
  {
    // Constructed in place (generate edges, fix orientation) so that the
    // attribute sets read from $PhysicalNames survive: MFEM's move
    // operations swap everything except those sets.
    mesh_ptr = std::make_unique<mfem::Mesh>(cfg.file.c_str(), 1, 1);
    if (mesh_ptr->GetNE() == 0)
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
      mesh_ptr = std::make_unique<mfem::Mesh>(
        mfem::Mesh::MakeCartesian2D(box.nx, box.ny, type, true, box.sx, box.sy, false));
    }
    else
    {
      mesh_ptr = std::make_unique<mfem::Mesh>(
        mfem::Mesh::MakeCartesian3D(box.nx, box.ny, box.nz, type, box.sx, box.sy, box.sz, false));
    }
  }
  mfem::Mesh &mesh = *mesh_ptr;

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
  return mesh_ptr;
}

namespace
{

std::map<int, std::vector<std::string>> NamesByAttribute(mfem::AttributeSets &sets)
{
  std::map<int, std::vector<std::string>> names;
  for (const std::string &name : sets.GetAttributeSetNames())
  {
    const mfem::Array<int> &attrs = sets.GetAttributeSet(name);
    for (int i = 0; i < attrs.Size(); i++) { names[attrs[i]].push_back(name); }
  }
  return names;
}

} // namespace

std::string DescribeAttributes(mfem::Mesh &mesh, bool boundary)
{
  const mfem::Array<int> &attrs = boundary ? mesh.bdr_attributes : mesh.attributes;
  const auto names = NamesByAttribute(boundary ? mesh.bdr_attribute_sets : mesh.attribute_sets);
  std::string s;
  for (int i = 0; i < attrs.Size(); i++)
  {
    s += (i ? ", " : "") + std::to_string(attrs[i]);
    const auto it = names.find(attrs[i]);
    if (it != names.end())
    {
      s += " (";
      for (std::size_t k = 0; k < it->second.size(); k++) { s += (k ? ", " : "") + it->second[k]; }
      s += ")";
    }
  }
  return s.empty() ? "none" : s;
}

std::vector<int> ResolveBoundaryAttributes(mfem::Mesh &mesh, const BoundaryCondition &bc,
                                           const std::string &what)
{
  std::vector<int> out;
  for (int a : bc.attr)
  {
    if (mesh.bdr_attributes.Find(a) < 0)
    {
      throw ConfigError(what + ": boundary attribute " + std::to_string(a) +
                        " is not in the mesh (boundary attributes: " +
                        DescribeAttributes(mesh, true) + ")");
    }
    out.push_back(a);
  }
  for (const std::string &name : bc.attr_names)
  {
    if (!mesh.bdr_attribute_sets.AttributeSetExists(name))
    {
      throw ConfigError(what + ": the mesh has no boundary physical group named '" + name +
                        "' (boundary attributes: " + DescribeAttributes(mesh, true) + ")");
    }
    const mfem::Array<int> &attrs = mesh.bdr_attribute_sets.GetAttributeSet(name);
    for (int i = 0; i < attrs.Size(); i++) { out.push_back(attrs[i]); }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::vector<int> ResolveElementAttributes(mfem::Mesh &mesh, const std::vector<int> &attr,
                                          const std::vector<std::string> &attr_names,
                                          const std::string &what)
{
  std::vector<int> out;
  for (int a : attr)
  {
    if (mesh.attributes.Find(a) < 0)
    {
      throw ConfigError(what + ": element attribute " + std::to_string(a) +
                        " is not in the mesh (element attributes: " +
                        DescribeAttributes(mesh, false) + ")");
    }
    out.push_back(a);
  }
  for (const std::string &name : attr_names)
  {
    if (!mesh.attribute_sets.AttributeSetExists(name))
    {
      throw ConfigError(what + ": the mesh has no physical volume named '" + name +
                        "' (element attributes: " + DescribeAttributes(mesh, false) + ")");
    }
    const mfem::Array<int> &attrs = mesh.attribute_sets.GetAttributeSet(name);
    for (int i = 0; i < attrs.Size(); i++) { out.push_back(attrs[i]); }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::unique_ptr<mfem::ParMesh> BuildParMesh(MPI_Comm comm, const MeshConfig &cfg)
{
  std::unique_ptr<mfem::Mesh> serial = BuildSerialMesh(cfg);
  // The ParMesh constructor copies the attribute sets (physical names).
  auto pmesh = std::make_unique<mfem::ParMesh>(comm, *serial);
  serial.reset();
  for (int l = 0; l < cfg.parallel_refine; l++) { pmesh->UniformRefinement(); }
  return pmesh;
}

} // namespace cmf

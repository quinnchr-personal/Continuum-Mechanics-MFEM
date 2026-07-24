#include "exasim_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace hdg_ns
{
namespace
{

constexpr std::array<int, 4> kExasimCornerNodes{{0, 4, 24, 20}};

int ClassifyBoundary(double x, double y)
{
   if (std::hypot(x, y) < 1.0 + 1.0e-6) { return 1; }
   if (x > -1.0e-7) { return 2; }
   return 3;
}

struct EdgeRecord
{
   int count = 0;
   int first = -1;
   int second = -1;
};

using EdgeKey = std::pair<int, int>;

EdgeKey SortedEdge(int first, int second)
{
   return std::minmax(first, second);
}

std::map<EdgeKey, EdgeRecord> EnumerateEdges(const ExasimGrid &grid)
{
   std::map<EdgeKey, EdgeRecord> edges;
   for (int element = 0; element < grid.ne; ++element)
   {
      for (int local_edge = 0; local_edge < 4; ++local_edge)
      {
         const int first = grid.Vertex(local_edge, element);
         const int second = grid.Vertex((local_edge + 1) % 4, element);
         EdgeRecord &record = edges[SortedEdge(first, second)];
         ++record.count;
         if (record.count == 1)
         {
            record.first = first;
            record.second = second;
         }
      }
   }
   return edges;
}

double CoordinateDistanceSquared(const ExasimArray &xdg, int element,
                                 int exasim_corner, const double vertex[2])
{
   const int node = kExasimCornerNodes[exasim_corner];
   const double dx = xdg(node, 0, element) - vertex[0];
   const double dy = xdg(node, 1, element) - vertex[1];
   return dx * dx + dy * dy;
}

bool IsDihedralPermutation(const std::array<int, 4> &mapping)
{
   std::set<int> unique(mapping.begin(), mapping.end());
   if (unique.size() != 4 || *unique.begin() != 0 ||
       *unique.rbegin() != 3)
   {
      return false;
   }
   const int step = (mapping[1] - mapping[0] + 4) % 4;
   if (step != 1 && step != 3) { return false; }
   for (int corner = 1; corner < 4; ++corner)
   {
      if (mapping[corner] != (mapping[0] + step * corner) % 4)
      {
         return false;
      }
   }
   return true;
}

ElementOrientation DetermineOrientation(const mfem::Mesh &mesh,
                                        const ExasimArray &xdg, int element)
{
   mfem::Array<int> vertices;
   mesh.GetElementVertices(element, vertices);
   if (vertices.Size() != 4)
   {
      throw std::runtime_error("Exasim converter expected quadrilateral");
   }
   ElementOrientation orientation;
   std::set<int> used;
   for (int mfem_corner = 0; mfem_corner < 4; ++mfem_corner)
   {
      const double *vertex = mesh.GetVertex(vertices[mfem_corner]);
      int best_corner = -1;
      double best_distance = std::numeric_limits<double>::infinity();
      for (int exasim_corner = 0; exasim_corner < 4; ++exasim_corner)
      {
         const double distance =
            CoordinateDistanceSquared(xdg, element, exasim_corner, vertex);
         if (distance < best_distance)
         {
            best_distance = distance;
            best_corner = exasim_corner;
         }
      }
      const double coordinate_scale =
         std::max({1.0, std::abs(vertex[0]), std::abs(vertex[1])});
      if (std::sqrt(best_distance) > 1.0e-13 * coordinate_scale ||
          used.count(best_corner) != 0)
      {
         std::ostringstream message;
         message << "element " << element
                 << ": MFEM/Exasim corner match failed, distance "
                 << std::sqrt(best_distance);
         throw std::runtime_error(message.str());
      }
      orientation.mfem_corner_to_exasim[mfem_corner] = best_corner;
      used.insert(best_corner);
   }
   if (!IsDihedralPermutation(orientation.mfem_corner_to_exasim))
   {
      throw std::runtime_error(
         "MFEM-to-Exasim local corner map is not a dihedral orientation");
   }
   return orientation;
}

void InstallExactGeometry(ExasimMesh &converted)
{
   mfem::Mesh &mesh = *converted.mesh;
   mesh.SetCurvature(kOrder, false, 2, mfem::Ordering::byNODES);
   mfem::GridFunction *nodes = mesh.GetNodes();
   mfem::FiniteElementSpace *nodes_fes = nodes->FESpace();
   std::vector<char> assigned(static_cast<std::size_t>(nodes->Size()), 0);
   mfem::Array<int> dofs;

   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      const mfem::FiniteElement *finite_element = nodes_fes->GetFE(element);
      const mfem::IntegrationRule &target_nodes = finite_element->GetNodes();
      if (target_nodes.GetNPoints() != kNodes2D)
      {
         throw std::runtime_error("MFEM H1 Q4 geometry does not have 25 nodes");
      }
      nodes_fes->GetElementDofs(element, dofs);
      if (dofs.Size() != kNodes2D)
      {
         throw std::runtime_error("MFEM H1 Q4 geometry has wrong dof count");
      }
      for (int local_dof = 0; local_dof < kNodes2D; ++local_dof)
      {
         if (dofs[local_dof] < 0)
         {
            throw std::runtime_error(
               "signed H1 geometry dof is unsupported by converter");
         }
         const mfem::IntegrationPoint &point =
            target_nodes.IntPoint(local_dof);
         const auto exasim_point =
            converted.orientations[element].MfemToExasim(point.x, point.y);
         for (int component = 0; component < 2; ++component)
         {
            const double value = EvaluateTensorQ4(
               converted.xdg, component, element,
               exasim_point[0], exasim_point[1]);
            const int vdof =
               nodes_fes->DofToVDof(dofs[local_dof], component);
            if (!assigned[static_cast<std::size_t>(vdof)])
            {
               (*nodes)[vdof] = value;
               assigned[static_cast<std::size_t>(vdof)] = 1;
            }
            else
            {
               const double scale =
                  std::max({1.0, std::abs(value), std::abs((*nodes)[vdof])});
               if (std::abs((*nodes)[vdof] - value) > 1.0e-13 * scale)
               {
                  std::ostringstream message;
                  message << "discontinuous xdg geometry at shared MFEM dof "
                          << vdof << ": values " << (*nodes)[vdof]
                          << " and " << value;
                  throw std::runtime_error(message.str());
               }
            }
         }
      }
   }
   if (std::find(assigned.begin(), assigned.end(), 0) != assigned.end())
   {
      throw std::runtime_error("not every MFEM geometry dof was assigned");
   }
   mesh.NodesUpdated();
}

std::array<double, 2> BoundaryMidpoint(mfem::Mesh &mesh,
                                      int boundary_element)
{
   mfem::ElementTransformation *transformation =
      mesh.GetBdrElementTransformation(boundary_element);
   mfem::IntegrationPoint midpoint;
   midpoint.x = 0.5;
   mfem::Vector physical(2);
   transformation->Transform(midpoint, physical);
   return {{physical[0], physical[1]}};
}

} // namespace

ExasimMesh BuildExasimMesh(const std::string &exasim_directory)
{
   ExasimMesh converted;
   converted.grid = ReadExasimGrid(exasim_directory + "/grid.bin");
   converted.xdg = ReadExasimArray(exasim_directory + "/xdg.bin");
   if (converted.grid.nd != 2 || converted.grid.nve != 4 ||
       converted.grid.np != 704 || converted.grid.ne != 651)
   {
      throw std::runtime_error(
         "grid.bin is not the specified 2D 704-vertex/651-quad mesh");
   }
   if (converted.xdg.nnode != kNodes2D ||
       converted.xdg.ncomp != 2 ||
       converted.xdg.nelem != converted.grid.ne)
   {
      throw std::runtime_error("xdg.bin does not have layout [25,2,651]");
   }

   const auto edges = EnumerateEdges(converted.grid);
   int boundary_count = 0;
   int interior_count = 0;
   for (const auto &entry : edges)
   {
      if (entry.second.count == 1) { ++boundary_count; }
      else if (entry.second.count == 2) { ++interior_count; }
      else
      {
         throw std::runtime_error("non-manifold edge in Exasim grid");
      }
   }
   if (static_cast<int>(edges.size()) != 1354 ||
       boundary_count != 104 || interior_count != 1250 ||
       4 * converted.grid.ne != 2 * interior_count + boundary_count)
   {
      throw std::runtime_error("Exasim grid edge/Euler sanity check failed");
   }

   converted.mesh = std::make_unique<mfem::Mesh>(
      2, converted.grid.np, converted.grid.ne, boundary_count, 2);
   for (int vertex = 0; vertex < converted.grid.np; ++vertex)
   {
      const double coordinate[2] =
      {
         converted.grid.Point(0, vertex),
         converted.grid.Point(1, vertex)
      };
      converted.mesh->AddVertex(coordinate);
   }
   for (int element = 0; element < converted.grid.ne; ++element)
   {
      int vertices[4];
      for (int local = 0; local < 4; ++local)
      {
         vertices[local] = converted.grid.Vertex(local, element);
      }
      converted.mesh->AddQuad(vertices, 1);
   }
   for (const auto &entry : edges)
   {
      const EdgeRecord &edge = entry.second;
      if (edge.count != 1) { continue; }
      const double x =
         0.5 * (converted.grid.Point(0, edge.first) +
                converted.grid.Point(0, edge.second));
      const double y =
         0.5 * (converted.grid.Point(1, edge.first) +
                converted.grid.Point(1, edge.second));
      converted.mesh->AddBdrSegment(
         edge.first, edge.second, ClassifyBoundary(x, y));
   }
   converted.mesh->FinalizeQuadMesh(1);

   converted.orientations.resize(converted.grid.ne);
   for (int element = 0; element < converted.grid.ne; ++element)
   {
      converted.orientations[element] =
         DetermineOrientation(*converted.mesh, converted.xdg, element);
      if (element > 0 &&
          !(converted.orientations[element] == converted.orientations[0]))
      {
         throw std::runtime_error(
            "element-local MFEM/Exasim orientation is not uniform");
      }
   }

   InstallExactGeometry(converted);
   return converted;
}

std::unique_ptr<mfem::Mesh> BuildAnalyticMesh(int nr, int nc, int order)
{
   if (nr <= 0 || nc <= 0 || order <= 0)
   {
      throw std::invalid_argument(
         "BuildAnalyticMesh requires positive nr, nc, and order");
   }
   auto mesh = std::make_unique<mfem::Mesh>(
      mfem::Mesh::MakeCartesian2D(
         nr, nc, mfem::Element::QUADRILATERAL, true,
         1.0, 1.0, false));
   mesh->SetCurvature(order, false, 2, mfem::Ordering::byNODES);
   mesh->Transform([](const mfem::Vector &parameter, mfem::Vector &physical)
   {
      constexpr double pi = 3.141592653589793238462643383279502884;
      const double theta = 1.5 * pi - pi * parameter[1];
      const double outer_radius = 4.7 + 1.7 * std::cos(theta);
      const double stretched =
         (1.0 - std::exp(-5.0 * parameter[0])) /
         (1.0 - std::exp(-5.0));
      const double radius =
         outer_radius + stretched * (1.0 - outer_radius);
      physical[0] = radius * std::cos(theta);
      physical[1] = radius * std::sin(theta);
   });
   for (int boundary = 0; boundary < mesh->GetNBE(); ++boundary)
   {
      const auto midpoint = BoundaryMidpoint(*mesh, boundary);
      mesh->SetBdrAttribute(
         boundary, ClassifyBoundary(midpoint[0], midpoint[1]));
   }
   mesh->SetAttributes(false, true);
   return mesh;
}

std::array<int, 3> CountBoundaryAttributes(const mfem::Mesh &mesh)
{
   std::array<int, 3> counts{{0, 0, 0}};
   for (int boundary = 0; boundary < mesh.GetNBE(); ++boundary)
   {
      const int attribute = mesh.GetBdrAttribute(boundary);
      if (attribute < 1 || attribute > 3)
      {
         throw std::runtime_error("boundary attribute outside 1,2,3");
      }
      ++counts[attribute - 1];
   }
   return counts;
}

double GeometryReproductionError(const ExasimMesh &converted)
{
   const mfem::Mesh &mesh = *converted.mesh;
   double maximum_error = 0.0;
   mfem::Vector physical(2);
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      mfem::ElementTransformation *transformation =
         converted.mesh->GetElementTransformation(element);
      for (int j = 0; j < kNodes1D; ++j)
      {
         for (int i = 0; i < kNodes1D; ++i)
         {
            const int node = i + kNodes1D * j;
            const auto mfem_point =
               converted.orientations[element].ExasimToMfem(
                  ExasimNodes1D()[i], ExasimNodes1D()[j]);
            mfem::IntegrationPoint point;
            point.Set2(mfem_point[0], mfem_point[1]);
            transformation->Transform(point, physical);
            for (int component = 0; component < 2; ++component)
            {
               const double expected =
                  converted.xdg(node, component, element);
               maximum_error =
                  std::max(
                     maximum_error,
                     std::abs(physical[component] - expected) /
                     std::max(1.0, std::abs(expected)));
            }
         }
      }
   }
   return maximum_error;
}

std::array<std::array<double, 2>, kNodes2D>
FiniteElementNodes(const mfem::FiniteElement &finite_element)
{
   const mfem::IntegrationRule &nodes = finite_element.GetNodes();
   if (nodes.GetNPoints() != kNodes2D)
   {
      throw std::runtime_error("finite element does not have 25 Q4 nodes");
   }
   std::array<std::array<double, 2>, kNodes2D> points{};
   for (int node = 0; node < kNodes2D; ++node)
   {
      points[node][0] = nodes.IntPoint(node).x;
      points[node][1] = nodes.IntPoint(node).y;
   }
   return points;
}

} // namespace hdg_ns

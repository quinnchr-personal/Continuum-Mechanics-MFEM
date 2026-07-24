#pragma once

#include "exasim_io.hpp"

#include "mfem.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace hdg_ns
{

struct ExasimMesh
{
   std::unique_ptr<mfem::Mesh> mesh;
   ExasimGrid grid;
   ExasimArray xdg;
   std::vector<ElementOrientation> orientations;
};

ExasimMesh BuildExasimMesh(const std::string &exasim_directory);
std::unique_ptr<mfem::Mesh> BuildAnalyticMesh(int nr, int nc, int order);

std::array<int, 3> CountBoundaryAttributes(const mfem::Mesh &mesh);
double GeometryReproductionError(const ExasimMesh &converted);

std::array<std::array<double, 2>, kNodes2D>
FiniteElementNodes(const mfem::FiniteElement &finite_element);

} // namespace hdg_ns

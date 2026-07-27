#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace hycfd
{

constexpr int kOrder = 4;
constexpr int kNodes1D = kOrder + 1;
constexpr int kNodes2D = kNodes1D * kNodes1D;

struct ExasimArray
{
   int nnode = 0;
   int ncomp = 0;
   int nelem = 0;
   std::vector<double> data;

   double &operator()(int node, int component, int element);
   double operator()(int node, int component, int element) const;
};

struct ExasimGrid
{
   int nd = 0;
   int np = 0;
   int nve = 0;
   int ne = 0;
   std::vector<double> points;
   std::vector<int> elements;

   double Point(int component, int vertex) const;
   int Vertex(int local_vertex, int element) const;
};

ExasimArray ReadExasimArray(const std::string &path);
ExasimGrid ReadExasimGrid(const std::string &path);
std::vector<double> ReadExasimDoubles(const std::string &path);

const std::array<double, kNodes1D> &ExasimNodes1D();
std::array<double, kNodes1D> Lagrange1D(double coordinate);
double EvaluateTensorQ4(const ExasimArray &array, int component, int element,
                        double xi, double eta);

struct ElementOrientation
{
   // Exasim corner number at each MFEM corner. Both use the standard corner
   // order (0,0), (1,0), (1,1), (0,1).
   std::array<int, 4> mfem_corner_to_exasim{{0, 1, 2, 3}};

   std::array<double, 2> MfemToExasim(double xi, double eta) const;
   std::array<double, 2> ExasimToMfem(double xi, double eta) const;
   bool operator==(const ElementOrientation &other) const;
};

struct TensorBasisTransform
{
   // Row-major matrices. forward maps Exasim CGL nodal values to values at
   // target_points; inverse maps target values back to Exasim nodes.
   std::array<double, kNodes2D * kNodes2D> forward{};
   std::array<double, kNodes2D * kNodes2D> inverse{};

   void ToTarget(const double source[kNodes2D],
                 double target[kNodes2D]) const;
   void ToExasim(const double target[kNodes2D],
                 double source[kNodes2D]) const;
};

TensorBasisTransform BuildTensorBasisTransform(
   const std::array<std::array<double, 2>, kNodes2D> &target_points,
   const ElementOrientation &orientation);

} // namespace hycfd

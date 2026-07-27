#include "io/exasim_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace hycfd
{
namespace
{

int CheckedInteger(double value, const std::string &path,
                   const std::string &label)
{
   if (!std::isfinite(value) || value < 0.0 ||
       std::abs(value - std::round(value)) >
       8.0 * std::numeric_limits<double>::epsilon() *
       std::max(1.0, std::abs(value)) ||
       value > static_cast<double>(std::numeric_limits<int>::max()))
   {
      std::ostringstream message;
      message << path << ": invalid integer-valued " << label
              << " header entry " << value;
      throw std::runtime_error(message.str());
   }
   return static_cast<int>(std::llround(value));
}

std::vector<double> ReadDoubles(const std::string &path)
{
   std::ifstream input(path, std::ios::binary | std::ios::ate);
   if (!input)
   {
      throw std::runtime_error("cannot open Exasim binary: " + path);
   }
   const std::streamoff byte_count = input.tellg();
   if (byte_count < 0 ||
       byte_count % static_cast<std::streamoff>(sizeof(double)) != 0)
   {
      throw std::runtime_error(
         path + ": byte count is not a multiple of sizeof(double)");
   }
   input.seekg(0);
   std::vector<double> values(
      static_cast<std::size_t>(byte_count / sizeof(double)));
   if (!values.empty())
   {
      input.read(reinterpret_cast<char *>(values.data()), byte_count);
   }
   if (!input)
   {
      throw std::runtime_error("short read from Exasim binary: " + path);
   }
   return values;
}

constexpr std::array<std::array<double, 2>, 4> kCornerCoordinates =
{{
   {{0.0, 0.0}},
   {{1.0, 0.0}},
   {{1.0, 1.0}},
   {{0.0, 1.0}}
}};

std::array<double, 4> BilinearWeights(double xi, double eta)
{
   return
   {
      (1.0 - xi) * (1.0 - eta),
      xi * (1.0 - eta),
      xi * eta,
      (1.0 - xi) * eta
   };
}

std::array<double, kNodes2D * kNodes2D> InvertMatrix(
   const std::array<double, kNodes2D * kNodes2D> &matrix)
{
   constexpr int n = kNodes2D;
   std::array<double, n * 2 * n> augmented{};
   for (int row = 0; row < n; ++row)
   {
      for (int column = 0; column < n; ++column)
      {
         augmented[row * (2 * n) + column] =
            matrix[row * n + column];
         augmented[row * (2 * n) + n + column] =
            row == column ? 1.0 : 0.0;
      }
   }

   for (int column = 0; column < n; ++column)
   {
      int pivot_row = column;
      double pivot_magnitude =
         std::abs(augmented[pivot_row * (2 * n) + column]);
      for (int candidate = column + 1; candidate < n; ++candidate)
      {
         const double magnitude =
            std::abs(augmented[candidate * (2 * n) + column]);
         if (magnitude > pivot_magnitude)
         {
            pivot_magnitude = magnitude;
            pivot_row = candidate;
         }
      }
      if (pivot_magnitude <= 100.0 * std::numeric_limits<double>::epsilon())
      {
         throw std::runtime_error("singular Q4 nodal change-of-basis matrix");
      }
      if (pivot_row != column)
      {
         for (int entry = 0; entry < 2 * n; ++entry)
         {
            std::swap(augmented[column * (2 * n) + entry],
                      augmented[pivot_row * (2 * n) + entry]);
         }
      }

      const double pivot = augmented[column * (2 * n) + column];
      for (int entry = 0; entry < 2 * n; ++entry)
      {
         augmented[column * (2 * n) + entry] /= pivot;
      }
      for (int row = 0; row < n; ++row)
      {
         if (row == column) { continue; }
         const double multiplier = augmented[row * (2 * n) + column];
         for (int entry = 0; entry < 2 * n; ++entry)
         {
            augmented[row * (2 * n) + entry] -=
               multiplier * augmented[column * (2 * n) + entry];
         }
      }
   }

   std::array<double, n * n> inverse{};
   for (int row = 0; row < n; ++row)
   {
      for (int column = 0; column < n; ++column)
      {
         inverse[row * n + column] =
            augmented[row * (2 * n) + n + column];
      }
   }
   return inverse;
}

void Multiply(const std::array<double, kNodes2D * kNodes2D> &matrix,
              const double input[kNodes2D], double output[kNodes2D])
{
   for (int row = 0; row < kNodes2D; ++row)
   {
      double sum = 0.0;
      for (int column = 0; column < kNodes2D; ++column)
      {
         sum += matrix[row * kNodes2D + column] * input[column];
      }
      output[row] = sum;
   }
}

} // namespace

double &ExasimArray::operator()(int node, int component, int element)
{
   return data.at(static_cast<std::size_t>(node) +
                  static_cast<std::size_t>(nnode) *
                  (static_cast<std::size_t>(component) +
                   static_cast<std::size_t>(ncomp) *
                   static_cast<std::size_t>(element)));
}

double ExasimArray::operator()(int node, int component, int element) const
{
   return data.at(static_cast<std::size_t>(node) +
                  static_cast<std::size_t>(nnode) *
                  (static_cast<std::size_t>(component) +
                   static_cast<std::size_t>(ncomp) *
                   static_cast<std::size_t>(element)));
}

double ExasimGrid::Point(int component, int vertex) const
{
   return points.at(static_cast<std::size_t>(component) +
                    static_cast<std::size_t>(nd) *
                    static_cast<std::size_t>(vertex));
}

int ExasimGrid::Vertex(int local_vertex, int element) const
{
   return elements.at(static_cast<std::size_t>(local_vertex) +
                      static_cast<std::size_t>(nve) *
                      static_cast<std::size_t>(element));
}

ExasimArray ReadExasimArray(const std::string &path)
{
   const std::vector<double> raw = ReadDoubles(path);
   if (raw.size() < 3)
   {
      throw std::runtime_error(path + ": missing 3-double array header");
   }
   ExasimArray array;
   array.nnode = CheckedInteger(raw[0], path, "nnode");
   array.ncomp = CheckedInteger(raw[1], path, "ncomp");
   array.nelem = CheckedInteger(raw[2], path, "nelem");
   const std::size_t expected =
      3 + static_cast<std::size_t>(array.nnode) *
          static_cast<std::size_t>(array.ncomp) *
          static_cast<std::size_t>(array.nelem);
   if (raw.size() != expected)
   {
      std::ostringstream message;
      message << path << ": header implies " << expected
              << " doubles, file contains " << raw.size();
      throw std::runtime_error(message.str());
   }
   array.data.assign(raw.begin() + 3, raw.end());
   return array;
}

std::vector<double> ReadExasimDoubles(const std::string &path)
{
   return ReadDoubles(path);
}

ExasimGrid ReadExasimGrid(const std::string &path)
{
   const std::vector<double> raw = ReadDoubles(path);
   if (raw.size() < 4)
   {
      throw std::runtime_error(path + ": missing 4-double grid header");
   }
   ExasimGrid grid;
   grid.nd = CheckedInteger(raw[0], path, "nd");
   grid.np = CheckedInteger(raw[1], path, "np");
   grid.nve = CheckedInteger(raw[2], path, "nve");
   grid.ne = CheckedInteger(raw[3], path, "ne");
   const std::size_t coordinate_count =
      static_cast<std::size_t>(grid.nd) *
      static_cast<std::size_t>(grid.np);
   const std::size_t connectivity_count =
      static_cast<std::size_t>(grid.nve) *
      static_cast<std::size_t>(grid.ne);
   const std::size_t expected = 4 + coordinate_count + connectivity_count;
   if (raw.size() != expected)
   {
      std::ostringstream message;
      message << path << ": header implies " << expected
              << " doubles, file contains " << raw.size();
      throw std::runtime_error(message.str());
   }
   grid.points.assign(raw.begin() + 4,
                      raw.begin() + 4 + coordinate_count);
   grid.elements.resize(connectivity_count);
   for (std::size_t i = 0; i < connectivity_count; ++i)
   {
      const int one_based =
         CheckedInteger(raw[4 + coordinate_count + i], path, "connectivity");
      if (one_based < 1 || one_based > grid.np)
      {
         throw std::runtime_error(path + ": connectivity index out of range");
      }
      grid.elements[i] = one_based - 1;
   }
   return grid;
}

const std::array<double, kNodes1D> &ExasimNodes1D()
{
   static const std::array<double, kNodes1D> nodes =
   {{
      0.0,
      (3.0 - std::sqrt(5.0)) / 4.0,
      0.5,
      (1.0 + std::sqrt(5.0)) / 4.0,
      1.0
   }};
   return nodes;
}

std::array<double, kNodes1D> Lagrange1D(double coordinate)
{
   const auto &nodes = ExasimNodes1D();
   std::array<double, kNodes1D> values{};
   for (int i = 0; i < kNodes1D; ++i)
   {
      values[i] = 1.0;
      for (int j = 0; j < kNodes1D; ++j)
      {
         if (i == j) { continue; }
         values[i] *=
            (coordinate - nodes[j]) / (nodes[i] - nodes[j]);
      }
   }
   return values;
}

double EvaluateTensorQ4(const ExasimArray &array, int component, int element,
                        double xi, double eta)
{
   if (array.nnode != kNodes2D)
   {
      throw std::runtime_error(
         "EvaluateTensorQ4 requires 25 Exasim nodes per element");
   }
   const auto shape_x = Lagrange1D(xi);
   const auto shape_y = Lagrange1D(eta);
   double value = 0.0;
   for (int j = 0; j < kNodes1D; ++j)
   {
      for (int i = 0; i < kNodes1D; ++i)
      {
         value += shape_x[i] * shape_y[j] *
                  array(i + kNodes1D * j, component, element);
      }
   }
   return value;
}

std::array<double, 2> ElementOrientation::MfemToExasim(
   double xi, double eta) const
{
   const auto weights = BilinearWeights(xi, eta);
   std::array<double, 2> mapped{{0.0, 0.0}};
   for (int corner = 0; corner < 4; ++corner)
   {
      const int target = mfem_corner_to_exasim[corner];
      mapped[0] += weights[corner] * kCornerCoordinates[target][0];
      mapped[1] += weights[corner] * kCornerCoordinates[target][1];
   }
   return mapped;
}

std::array<double, 2> ElementOrientation::ExasimToMfem(
   double xi, double eta) const
{
   std::array<int, 4> inverse{};
   for (int mfem_corner = 0; mfem_corner < 4; ++mfem_corner)
   {
      const int exasim_corner = mfem_corner_to_exasim[mfem_corner];
      if (exasim_corner < 0 || exasim_corner >= 4)
      {
         throw std::runtime_error("invalid element corner orientation");
      }
      inverse[exasim_corner] = mfem_corner;
   }
   const auto weights = BilinearWeights(xi, eta);
   std::array<double, 2> mapped{{0.0, 0.0}};
   for (int exasim_corner = 0; exasim_corner < 4; ++exasim_corner)
   {
      const int target = inverse[exasim_corner];
      mapped[0] += weights[exasim_corner] * kCornerCoordinates[target][0];
      mapped[1] += weights[exasim_corner] * kCornerCoordinates[target][1];
   }
   return mapped;
}

bool ElementOrientation::operator==(const ElementOrientation &other) const
{
   return mfem_corner_to_exasim == other.mfem_corner_to_exasim;
}

void TensorBasisTransform::ToTarget(
   const double source[kNodes2D], double target[kNodes2D]) const
{
   Multiply(forward, source, target);
}

void TensorBasisTransform::ToExasim(
   const double target[kNodes2D], double source[kNodes2D]) const
{
   Multiply(inverse, target, source);
}

TensorBasisTransform BuildTensorBasisTransform(
   const std::array<std::array<double, 2>, kNodes2D> &target_points,
   const ElementOrientation &orientation)
{
   TensorBasisTransform transform;
   for (int target = 0; target < kNodes2D; ++target)
   {
      const auto exasim_point = orientation.MfemToExasim(
         target_points[target][0], target_points[target][1]);
      const auto shape_x = Lagrange1D(exasim_point[0]);
      const auto shape_y = Lagrange1D(exasim_point[1]);
      for (int j = 0; j < kNodes1D; ++j)
      {
         for (int i = 0; i < kNodes1D; ++i)
         {
            const int source = i + kNodes1D * j;
            transform.forward[target * kNodes2D + source] =
               shape_x[i] * shape_y[j];
         }
      }
   }
   transform.inverse = InvertMatrix(transform.forward);
   return transform;
}

} // namespace hycfd

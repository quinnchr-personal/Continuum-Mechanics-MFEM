#include "post/surface_post.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace hycfd
{
namespace
{

double RelativeDifference(double computed, double reference)
{
   return std::abs(computed - reference) /
          std::max(1.0e-14, std::abs(reference));
}

int LocatePoint(mfem::Mesh &mesh, const mfem::Vector &physical,
                mfem::IntegrationPoint &point)
{
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      mfem::ElementTransformation *transformation =
         mesh.GetElementTransformation(element);
      const int result =
         transformation->TransformBack(physical, point, 1.0e-12);
      if (result == mfem::InverseElementTransformation::Inside &&
          point.x >= -1.0e-10 && point.x <= 1.0 + 1.0e-10 &&
          point.y >= -1.0e-10 && point.y <= 1.0 + 1.0e-10)
      {
         return element;
      }
   }
   return -1;
}

double StagnationOuterRadius(mfem::Mesh &mesh)
{
   double maximum_radius = 0.0;
   mfem::Vector physical(2);
   for (int boundary = 0; boundary < mesh.GetNBE(); ++boundary)
   {
      if (mesh.GetBdrAttribute(boundary) != 3) { continue; }
      mfem::ElementTransformation *transformation =
         mesh.GetBdrElementTransformation(boundary);
      mfem::IntegrationPoint left, right, point;
      left.x = 0.0;
      right.x = 1.0;
      transformation->Transform(left, physical);
      const double left_y = physical[1];
      transformation->Transform(right, physical);
      const double right_y = physical[1];
      if (left_y * right_y > 0.0) { continue; }
      for (int iteration = 0; iteration < 80; ++iteration)
      {
         point.x = 0.5 * (left.x + right.x);
         transformation->Transform(point, physical);
         if ((physical[1] >= 0.0) == (left_y >= 0.0))
         {
            left.x = point.x;
         }
         else
         {
            right.x = point.x;
         }
      }
      point.x = 0.5 * (left.x + right.x);
      transformation->Transform(point, physical);
      maximum_radius = std::max(maximum_radius, -physical[0]);
   }
   const mfem::ParMesh *par_mesh =
      dynamic_cast<const mfem::ParMesh *>(&mesh);
   if (par_mesh && par_mesh->GetNRanks() > 1)
   {
      MPI_Allreduce(MPI_IN_PLACE, &maximum_radius, 1, MPI_DOUBLE,
                    MPI_MAX, par_mesh->GetComm());
   }
   if (!(maximum_radius > 1.0))
   {
      throw std::runtime_error(
         "could not identify the stagnation-line outer radius");
   }
   return maximum_radius;
}

} // namespace

std::vector<WallSample> ComputeWallSamples(
   mfem::Mesh &mesh, const HDGOperator &op,
   const HDGState &state, const PerfectGasParams &params)
{
   std::vector<WallSample> samples;
   mfem::Vector physical(2), weighted_normal(2);
   for (int boundary = 0; boundary < mesh.GetNBE(); ++boundary)
   {
      if (mesh.GetBdrAttribute(boundary) != 1) { continue; }
      const int face = mesh.GetBdrElementFaceIndex(boundary);
      mfem::FaceElementTransformations *transformation =
         mesh.GetFaceElementTransformations(face, 31);
      if (!transformation || transformation->Elem1No < 0 ||
          transformation->Elem2No >= 0)
      {
         throw std::runtime_error(
            "wall postprocessor expected a boundary face");
      }
      for (int qpoint = 0; qpoint < op.FaceRule().GetNPoints(); ++qpoint)
      {
         const mfem::IntegrationPoint &face_point =
            op.FaceRule().IntPoint(qpoint);
         transformation->SetAllIntPoints(&face_point);
         double uq[12], uhat[4], conductive[2];
         op.EvaluateElementState(
            state, transformation->Elem1No,
            transformation->GetElement1IntPoint(), uq);
         op.EvaluateTraceState(state, face, face_point, uhat);
         NSHeatFlux(uhat, uq, params, conductive);
         mfem::CalcOrtho(transformation->Jacobian(), weighted_normal);
         const double surface_jacobian = weighted_normal.Norml2();
         if (!(surface_jacobian > 0.0))
         {
            throw std::runtime_error("degenerate wall face");
         }
         weighted_normal /= surface_jacobian;
         transformation->Face->Transform(face_point, physical);
         double theta = std::atan2(physical[1], physical[0]);
         if (theta < 0.0) { theta += 2.0 * M_PI; }
         const double pinf =
            1.0 / (params.gamma * params.mach * params.mach);
         samples.push_back(
            {theta, physical[0], physical[1],
             2.0 * (Pressure(uhat, params) - pinf),
             conductive[0] * weighted_normal[0] +
             conductive[1] * weighted_normal[1] +
             params.tau * (uq[3] - uhat[3])});
      }
   }
   // Wall faces are distributed across ranks: gather everything so every
   // rank sees the full, identically ordered sample set.
   const mfem::ParMesh *par_mesh =
      dynamic_cast<const mfem::ParMesh *>(&mesh);
   if (par_mesh && par_mesh->GetNRanks() > 1)
   {
      MPI_Comm comm = par_mesh->GetComm();
      const int ranks = par_mesh->GetNRanks();
      std::vector<double> packed;
      packed.reserve(samples.size() * 5);
      for (const WallSample &sample : samples)
      {
         packed.push_back(sample.theta);
         packed.push_back(sample.x);
         packed.push_back(sample.y);
         packed.push_back(sample.cp);
         packed.push_back(sample.heat_flux);
      }
      const int local_count = static_cast<int>(packed.size());
      std::vector<int> counts(static_cast<std::size_t>(ranks), 0);
      MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT,
                    comm);
      std::vector<int> offsets(static_cast<std::size_t>(ranks) + 1, 0);
      for (int rank = 0; rank < ranks; ++rank)
      {
         offsets[static_cast<std::size_t>(rank) + 1] =
            offsets[static_cast<std::size_t>(rank)] +
            counts[static_cast<std::size_t>(rank)];
      }
      std::vector<double> all(static_cast<std::size_t>(
         offsets[static_cast<std::size_t>(ranks)]));
      MPI_Allgatherv(packed.data(), local_count, MPI_DOUBLE, all.data(),
                     counts.data(), offsets.data(), MPI_DOUBLE, comm);
      samples.clear();
      for (std::size_t i = 0; i + 4 < all.size(); i += 5)
      {
         samples.push_back(
            {all[i], all[i + 1], all[i + 2], all[i + 3], all[i + 4]});
      }
   }
   std::sort(samples.begin(), samples.end(),
             [](const WallSample &first, const WallSample &second)
             {
                return first.theta < second.theta;
             });
   if (samples.empty())
   {
      throw std::runtime_error(
         "wall postprocessor found no boundary faces with attribute 1");
   }
   return samples;
}

void WriteWallCSV(const std::string &path,
                  const std::vector<WallSample> &samples)
{
   std::ofstream output(path);
   if (!output)
   {
      throw std::runtime_error("cannot open wall CSV: " + path);
   }
   output << "theta,x,y,Cp,q_w\n" << std::setprecision(16);
   for (const WallSample &sample : samples)
   {
      output << sample.theta << ',' << sample.x << ',' << sample.y << ','
             << sample.cp << ',' << sample.heat_flux << '\n';
   }
   if (!output)
   {
      throw std::runtime_error("failed while writing wall CSV: " + path);
   }
}

std::vector<WallSample> ReadWallCSV(const std::string &path)
{
   std::ifstream input(path);
   if (!input)
   {
      throw std::runtime_error("cannot open wall CSV: " + path);
   }
   std::string line;
   if (!std::getline(input, line) || line != "theta,x,y,Cp,q_w")
   {
      throw std::runtime_error("wall CSV has an unexpected header: " + path);
   }
   std::vector<WallSample> samples;
   while (std::getline(input, line))
   {
      if (line.empty()) { continue; }
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream values(line);
      WallSample sample;
      if (!(values >> sample.theta >> sample.x >> sample.y >>
            sample.cp >> sample.heat_flux))
      {
         throw std::runtime_error("malformed wall CSV row: " + path);
      }
      std::string trailing;
      if (values >> trailing)
      {
         throw std::runtime_error(
            "wall CSV row has extra columns: " + path);
      }
      samples.push_back(sample);
   }
   if (samples.empty())
   {
      throw std::runtime_error("wall CSV contains no samples: " + path);
   }
   return samples;
}

ShockStandoff ComputeShockStandoff(
   mfem::Mesh &mesh, const HDGOperator &op,
   const HDGState &state, int sample_count)
{
   if (sample_count < 3)
   {
      throw std::runtime_error(
         "shock standoff requires at least three samples");
   }
   mfem::L2_FECollection scalar_collection(op.Order(), 2);
   mfem::FiniteElementSpace scalar_space(
      &mesh, &scalar_collection);
   mfem::GridFunction conservative(
      const_cast<mfem::FiniteElementSpace *>(&op.VolumeSpace()));
   op.FillConservativeGridFunction(state, conservative);
   mfem::GridFunction density(&scalar_space);
   if (conservative.Size() != 4 * density.Size())
   {
      throw std::runtime_error(
         "unexpected conservative GridFunction layout");
   }
   for (int i = 0; i < density.Size(); ++i)
   {
      density[i] = conservative[4 * i];
   }

   const double outer_radius = StagnationOuterRadius(mesh);
   const double spacing =
      (outer_radius - 1.0) / static_cast<double>(sample_count - 1);
   std::vector<double> values(static_cast<std::size_t>(sample_count));
   std::vector<int> elements(static_cast<std::size_t>(sample_count), -1);
   mfem::Vector physical(2);
   physical[1] = 0.0;
   constexpr double kMissing = -1.0e300;
   for (int sample = 0; sample < sample_count; ++sample)
   {
      const double radius = 1.0 + spacing * sample;
      physical[0] = -radius;
      mfem::IntegrationPoint point;
      const int element = LocatePoint(mesh, physical, point);
      elements[static_cast<std::size_t>(sample)] = element;
      values[static_cast<std::size_t>(sample)] =
         element >= 0 ? density.GetValue(element, point) : kMissing;
   }
   const mfem::ParMesh *par_mesh =
      dynamic_cast<const mfem::ParMesh *>(&mesh);
   if (par_mesh && par_mesh->GetNRanks() > 1)
   {
      MPI_Allreduce(MPI_IN_PLACE, values.data(), sample_count,
                    MPI_DOUBLE, MPI_MAX, par_mesh->GetComm());
   }
   for (int sample = 0; sample < sample_count; ++sample)
   {
      if (values[static_cast<std::size_t>(sample)] <= kMissing / 2.0)
      {
         throw std::runtime_error(
            "failed to locate a stagnation-line sample");
      }
   }

   int maximum_index = 1;
   double maximum_derivative = 0.0;
   for (int sample = 1; sample + 1 < sample_count; ++sample)
   {
      const double derivative =
         std::abs(values[static_cast<std::size_t>(sample + 1)] -
                  values[static_cast<std::size_t>(sample - 1)]) /
         (2.0 * spacing);
      if (derivative > maximum_derivative)
      {
         maximum_derivative = derivative;
         maximum_index = sample;
      }
   }
   const double radius = 1.0 + spacing * maximum_index;
   // Radial cell width from locally owned contiguous samples around the
   // maximum; the owning rank's value wins in the reduction.
   double radial_cell_width = 0.0;
   if (elements[static_cast<std::size_t>(maximum_index)] >= 0)
   {
      int first = maximum_index;
      int last = maximum_index;
      while (first > 0 &&
             elements[static_cast<std::size_t>(first - 1)] ==
             elements[static_cast<std::size_t>(maximum_index)])
      {
         --first;
      }
      while (last + 1 < sample_count &&
             elements[static_cast<std::size_t>(last + 1)] ==
             elements[static_cast<std::size_t>(maximum_index)])
      {
         ++last;
      }
      radial_cell_width =
         spacing * static_cast<double>(last - first + 1);
   }
   if (par_mesh && par_mesh->GetNRanks() > 1)
   {
      MPI_Allreduce(MPI_IN_PLACE, &radial_cell_width, 1, MPI_DOUBLE,
                    MPI_MAX, par_mesh->GetComm());
   }
   return {radius - 1.0, radius, radial_cell_width, maximum_index};
}

M3Comparison CompareWallAndShock(
   const std::vector<WallSample> &computed,
   const std::vector<WallSample> &reference,
   const ShockStandoff &computed_shock,
   const ShockStandoff &reference_shock,
   bool require_matching_coordinates)
{
   if (computed.size() != reference.size())
   {
      throw std::runtime_error("wall sample counts differ");
   }
   M3Comparison comparison;
   for (std::size_t i = 0; i < computed.size(); ++i)
   {
      const double coordinate_difference =
         std::hypot(computed[i].x - reference[i].x,
                    computed[i].y - reference[i].y);
      comparison.wall_coordinate_maximum_difference =
         std::max(comparison.wall_coordinate_maximum_difference,
                  coordinate_difference);
      if (require_matching_coordinates &&
          (RelativeDifference(computed[i].x, reference[i].x) > 1.0e-13 ||
           RelativeDifference(computed[i].y, reference[i].y) > 1.0e-13))
      {
         throw std::runtime_error(
            "wall sample coordinates do not match");
      }
      comparison.cp_max_relative_difference =
         std::max(comparison.cp_max_relative_difference,
                  RelativeDifference(computed[i].cp, reference[i].cp));
      comparison.heat_flux_max_relative_difference =
         std::max(
            comparison.heat_flux_max_relative_difference,
            RelativeDifference(computed[i].heat_flux,
                               reference[i].heat_flux));
   }
   comparison.shock_standoff_difference =
      std::abs(computed_shock.distance - reference_shock.distance);
   return comparison;
}

void WriteM3ComparisonReport(
   const std::string &path,
   const std::array<double, 4> &field_relative_l2,
   const M3Comparison &comparison,
   const ShockStandoff &computed_shock,
   const ShockStandoff &reference_shock)
{
   std::ofstream output(path);
   if (!output)
   {
      throw std::runtime_error(
         "cannot open M3 comparison report: " + path);
   }
   output << std::setprecision(16)
          << "# M3 Mach 8 comparison report\n\n"
          << "| Quantity | Result | Acceptance |\n"
          << "|---|---:|---:|\n"
          << "| rho relative L2 | " << field_relative_l2[0]
          << " | <= 1e-5 |\n"
          << "| rhou relative L2 | " << field_relative_l2[1]
          << " | <= 1e-5 |\n"
          << "| rhov relative L2 | " << field_relative_l2[2]
          << " | <= 1e-5 |\n"
          << "| rhoE relative L2 | " << field_relative_l2[3]
          << " | <= 1e-5 |\n"
          << "| wall Cp max relative difference | "
          << comparison.cp_max_relative_difference
          << " | <= 1e-4 |\n"
          << "| wall Fint heat-flux max relative difference | "
          << comparison.heat_flux_max_relative_difference
          << " | <= 1e-3 |\n"
          << "| HDG shock standoff | " << computed_shock.distance
          << " | — |\n"
          << "| Exasim shock standoff | " << reference_shock.distance
          << " | — |\n"
          << "| shock-standoff absolute difference | "
          << comparison.shock_standoff_difference
          << " | <= one radial cell ("
          << reference_shock.radial_cell_width << ") |\n";
   if (!output)
   {
      throw std::runtime_error(
         "failed while writing M3 comparison report: " + path);
   }
}

} // namespace hycfd

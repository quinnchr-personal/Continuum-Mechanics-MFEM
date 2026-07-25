#include "wall_post.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hdg_ns
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
   if (!(maximum_radius > 1.0))
   {
      throw std::runtime_error(
         "could not identify the stagnation-line outer radius");
   }
   return maximum_radius;
}

} // namespace

std::vector<WallSample> ComputeWallSamples(
   mfem::Mesh &mesh, const HDGNavierStokesOperator &op,
   const HDGState &state, const NSParams &params)
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
            1.0 / (params.mu[0] * params.mu[3] * params.mu[3]);
         samples.push_back(
            {theta, physical[0], physical[1],
             2.0 * (Pressure(uhat, params) - pinf),
             conductive[0] * weighted_normal[0] +
             conductive[1] * weighted_normal[1] +
             params.tau * (uq[3] - uhat[3])});
      }
   }
   std::sort(samples.begin(), samples.end(),
             [](const WallSample &first, const WallSample &second)
             {
                return first.theta < second.theta;
             });
   if (samples.size() != 105)
   {
      throw std::runtime_error(
         "wall postprocessor expected 21 faces x 5 points");
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

ShockStandoff ComputeShockStandoff(
   mfem::Mesh &mesh, const HDGNavierStokesOperator &op,
   const HDGState &state, int sample_count)
{
   if (sample_count < 3)
   {
      throw std::runtime_error(
         "shock standoff requires at least three samples");
   }
   mfem::L2_FECollection scalar_collection(kOrder, 2);
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
   for (int sample = 0; sample < sample_count; ++sample)
   {
      const double radius = 1.0 + spacing * sample;
      physical[0] = -radius;
      mfem::IntegrationPoint point;
      const int element = LocatePoint(mesh, physical, point);
      if (element < 0)
      {
         throw std::runtime_error(
            "failed to locate a stagnation-line sample");
      }
      elements[static_cast<std::size_t>(sample)] = element;
      values[static_cast<std::size_t>(sample)] =
         density.GetValue(element, point);
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
   const double radial_cell_width =
      spacing * static_cast<double>(last - first + 1);
   return {radius - 1.0, radius, radial_cell_width, maximum_index};
}

M3Comparison CompareWallAndShock(
   const std::vector<WallSample> &computed,
   const std::vector<WallSample> &reference,
   const ShockStandoff &computed_shock,
   const ShockStandoff &reference_shock)
{
   if (computed.size() != reference.size())
   {
      throw std::runtime_error("wall sample counts differ");
   }
   M3Comparison comparison;
   for (std::size_t i = 0; i < computed.size(); ++i)
   {
      if (RelativeDifference(computed[i].x, reference[i].x) > 1.0e-13 ||
          RelativeDifference(computed[i].y, reference[i].y) > 1.0e-13)
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

} // namespace hdg_ns

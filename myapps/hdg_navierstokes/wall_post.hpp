#pragma once

#include "hdg_ns_operator.hpp"

#include <array>
#include <string>
#include <vector>

namespace hdg_ns
{

struct WallSample
{
   double theta = 0.0;
   double x = 0.0;
   double y = 0.0;
   double cp = 0.0;
   double heat_flux = 0.0;
};

struct ShockStandoff
{
   double distance = 0.0;
   double radius = 0.0;
   double radial_cell_width = 0.0;
   int sample_index = -1;
};

struct M3Comparison
{
   double cp_max_relative_difference = 0.0;
   double heat_flux_max_relative_difference = 0.0;
   double shock_standoff_difference = 0.0;
};

std::vector<WallSample> ComputeWallSamples(
   mfem::Mesh &mesh, const HDGNavierStokesOperator &op,
   const HDGState &state, const NSParams &params);

void WriteWallCSV(const std::string &path,
                  const std::vector<WallSample> &samples);

M3Comparison CompareWallAndShock(
   const std::vector<WallSample> &computed,
   const std::vector<WallSample> &reference,
   const ShockStandoff &computed_shock,
   const ShockStandoff &reference_shock);

ShockStandoff ComputeShockStandoff(
   mfem::Mesh &mesh, const HDGNavierStokesOperator &op,
   const HDGState &state, int sample_count = 1000);

void WriteM3ComparisonReport(
   const std::string &path,
   const std::array<double, 4> &field_relative_l2,
   const M3Comparison &comparison,
   const ShockStandoff &computed_shock,
   const ShockStandoff &reference_shock);

} // namespace hdg_ns

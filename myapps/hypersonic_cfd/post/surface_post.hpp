#pragma once

#include "discretization/hdg_operator.hpp"
#include "physics/perfect_gas.hpp"

#include <array>
#include <string>
#include <vector>

namespace hycfd
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
   double wall_coordinate_maximum_difference = 0.0;
};

std::vector<WallSample> ComputeWallSamples(
   mfem::Mesh &mesh, const HDGOperator &op,
   const HDGState &state, const PerfectGasParams &params);

void WriteWallCSV(const std::string &path,
                  const std::vector<WallSample> &samples);
std::vector<WallSample> ReadWallCSV(const std::string &path);

M3Comparison CompareWallAndShock(
   const std::vector<WallSample> &computed,
   const std::vector<WallSample> &reference,
   const ShockStandoff &computed_shock,
   const ShockStandoff &reference_shock,
   bool require_matching_coordinates = true);

ShockStandoff ComputeShockStandoff(
   mfem::Mesh &mesh, const HDGOperator &op,
   const HDGState &state, int sample_count = 1000);

// Physical bow-shock position for validation against correlations:
// scanning the stagnation line (x = -r, y = 0) from the outer boundary
// inward, returns the standoff distance r - 1 where the density first
// crosses `threshold` (linearly interpolated between samples). Unlike
// ComputeShockStandoff's maximum-gradient fingerprint, this is immune to
// the near-wall thermal-layer gradient. Threshold should sit mid-jump,
// e.g. 0.5*(1 + rho_post_normal_shock).
double StagnationDensityCrossing(
   mfem::Mesh &mesh, const HDGOperator &op,
   const HDGState &state, double threshold, int sample_count = 600);

void WriteM3ComparisonReport(
   const std::string &path,
   const std::array<double, 4> &field_relative_l2,
   const M3Comparison &comparison,
   const ShockStandoff &computed_shock,
   const ShockStandoff &reference_shock);

} // namespace hycfd

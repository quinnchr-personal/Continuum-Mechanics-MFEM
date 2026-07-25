#pragma once

#include "exasim_mesh.hpp"
#include "hdg_ns_operator.hpp"

#include <array>
#include <string>
#include <vector>

namespace hdg_ns
{

struct ExasimReferenceData
{
   ExasimArray udg;
   std::vector<std::array<double, 20>> trace_face_values;
   double maximum_xdg_difference = 0.0;
   double maximum_face_geometry_difference = 0.0;
   double maximum_shared_trace_difference = 0.0;
   int local_face_count = 0;
   int duplicate_face_count = 0;
};

ExasimReferenceData ReadExasimReferenceData(
   const std::string &run_directory, const ExasimMesh &mesh);

void LoadExasimReferenceState(const ExasimReferenceData &reference,
                              HDGNavierStokesOperator &op,
                              bool load_gradient,
                              HDGState &state);

} // namespace hdg_ns

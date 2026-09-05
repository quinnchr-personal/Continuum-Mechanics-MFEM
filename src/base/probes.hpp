// Point probes: evaluate a (vector) grid function at a physical point,
// collective over the mesh communicator.
#pragma once

#include <vector>

#include "mfem.hpp"

namespace cmf
{

// Returns the vdim values of gf at point (size = space dimension). Throws if
// no rank owns an element containing the point. When several ranks find it
// (shared faces) the values are averaged, which is exact for a continuous
// field.
std::vector<double> ProbeVector(const mfem::ParGridFunction &gf,
                                const std::vector<double> &point);

} // namespace cmf

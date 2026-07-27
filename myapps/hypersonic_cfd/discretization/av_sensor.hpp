#pragma once

#include "discretization/hdg_operator.hpp"
#include "physics/physics_model.hpp"

#include "mfem.hpp"

namespace hycfd
{

struct DilatationSensorOptions
{
   // AV amplitude: eps = lambda * h_K * (|u|+c) * s(ratio).
   double lambda = 0.2;
   // Ramp argument: ratio = -div(u) * h_K / ((order+1) * kappa * c).
   double kappa = 0.25;
};

// Dilatation-based artificial viscosity: an elementwise sensor
//   eps_K = lambda * h_K * max_qp[(|u|+c) * s(-div(u) h_K/((p+1) kappa c))]
// with a compact-support smoothstep ramp s (identically zero below
// ratio=0.2, one above 0.8), vertex-max smoothed onto a continuous H1
// order-1 field. Resolved smooth flows produce exactly zero AV.
//
// The field is frozen data for the Newton solves that follow (lagged
// sensor fixed point); it is never differentiated with respect to the
// state.
void ComputeDilatationAV(const HDGOperator &op, const HDGState &state,
                         const PhysicsModel &physics,
                         const DilatationSensorOptions &options,
                         mfem::ParGridFunction &av);

} // namespace hycfd

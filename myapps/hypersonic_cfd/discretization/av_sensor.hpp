#pragma once

#include "discretization/hdg_operator.hpp"
#include "physics/physics_model.hpp"

#include "mfem.hpp"

#include <functional>

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
// The field is frozen data for the Newton solves that follow; it is never
// differentiated with respect to the state.
void ComputeDilatationAV(const HDGOperator &op, const HDGState &state,
                         const PhysicsModel &physics,
                         const DilatationSensorOptions &options,
                         mfem::ParGridFunction &av);

struct SensorAVSchedule
{
   DilatationSensorOptions sensor;
   // EMA weight on the fresh sensor field in (0, 1]; smaller is more
   // damped.
   double relax = 0.5;
   // Optional bootstrap floor: cold starts need AV where the shock will
   // form before the sensor can see it. The profile decays to zero
   // linearly over bootstrap_decay_iterations Newton iterations.
   std::function<double(const mfem::Vector &)> bootstrap_profile;
   int bootstrap_decay_iterations = 25;
   // Optional (0 disables): once the Newton residual falls below this
   // threshold — after the bootstrap floor has fully decayed — the field
   // freezes permanently so Newton can close quadratically instead of
   // chasing a sensor<->shock fixed point that hovers near, but never
   // below, the solve tolerance (the observed Mach 8 failure mode).
   double freeze_residual = 0.0;
   bool verbose = true;
};

// Adaptive frozen-per-iteration AV: bind Refresh as the NewtonPrepare
// hook. At the top of every Newton iteration it re-evaluates the
// dilatation sensor from the current iterate, under-relaxes it against
// the previous field (EMA), floors it with the decaying bootstrap
// profile, and installs the result as the operator's frozen AV tables so
// each linearization stays self-consistent.
class SensorAVController
{
public:
   SensorAVController(mfem::ParMesh &mesh, HDGOperator &op,
                      const PhysicsModel &physics,
                      const SensorAVSchedule &schedule);

   // residual is the previous iteration's accepted Newton residual
   // (infinity on the first call); it drives the freeze decision.
   void Refresh(int iteration, const HDGState &state, double residual);

private:
   HDGOperator &op_;
   const PhysicsModel &physics_;
   SensorAVSchedule schedule_;
   mfem::H1_FECollection fec_;
   mfem::ParFiniteElementSpace fes_;
   mfem::ParGridFunction av_, sensor_, bootstrap_;
   bool have_previous_ = false;
   bool frozen_ = false;
};

} // namespace hycfd

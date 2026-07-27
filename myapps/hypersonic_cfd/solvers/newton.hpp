#pragma once

#include "discretization/hdg_operator.hpp"
#include "solvers/trace_linear_solver.hpp"

#include "mfem.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace hycfd
{

struct NewtonConfig
{
   int max_iterations = 30;
   double tolerance = 1.0e-6;
   bool pseudo_transient = false;
   double initial_pseudo_time_step = 1.0e-3;

   // Pure Newton phase (pseudo-time step off): Armijo backtracking,
   // accept alpha when |R(x + alpha d)| <= (1 - armijo_c1*alpha) |R(x)|,
   // halving from 1 down to alpha_min; failure aborts the solve.
   // PTC phase: the damped step is accepted unconditionally when finite
   // (Kelley-Keyes semantics; the pseudo-time term damps only the volume
   // block, so shrinking dtau does not improve the direction) and the SER
   // controller cuts dtau on residual growth. Non-finite trials backtrack
   // alpha; if every alpha is non-finite the step is rejected and dtau is
   // multiplied by ptc_reject_factor.
   double armijo_c1 = 1.0e-4;
   double alpha_min = 0.015625;
   double ptc_reject_factor = 0.25;
   // Reject line-search trials whose minimum density or pressure is not
   // positive, exactly like non-finite residuals. With floors active a
   // globally-decreasing step can park the state in floor-land, where
   // the Jacobian is degenerate and Newton stalls; this keeps every
   // accepted iterate physical. Off by default.
   bool require_admissible = false;
   // SER: after an accepted step, dtau *= old_residual/new_residual
   // (clamped); PTC turns off once the residual drops below this threshold.
   double ptc_off_residual = 1.0e-2;
   double minimum_pseudo_time_step = 1.0e-12;
   double maximum_pseudo_time_step = 1.0e12;
};

struct NewtonIteration
{
   int iteration = 0;
   double residual = 0.0;
   double alpha = 0.0;
   double pseudo_time_step = 0.0;
};

struct NewtonReport
{
   bool converged = false;
   int iterations = 0;
   double residual = 0.0;
   double assembly_seconds = 0.0;
   double linear_solve_seconds = 0.0;
   double total_seconds = 0.0;
   std::string failure;
   std::vector<NewtonIteration> history;
};

using NewtonOutput =
   std::function<void(int iteration, const HDGState &state,
                      double residual)>;

// Called at the top of every iteration, before the residual/Jacobian
// assembly. Lets the caller refresh frozen operator data (e.g. a sensor
// artificial-viscosity field) from the current iterate; whatever it
// installs stays frozen through this iteration's Jacobian and line
// search, keeping each linearization self-consistent. `residual` is the
// previous iteration's accepted residual (infinity at iteration 0), so
// the caller can stop adapting once the solve is nearly converged.
using NewtonPrepare =
   std::function<void(int iteration, const HDGState &state,
                      double residual)>;

// Damped Newton with Armijo backtracking and an optional pseudo-transient
// continuation (backward-Euler mass term, SER time-step growth). Step
// rejection on line-search failure shrinks the pseudo-time step instead of
// accepting a residual increase.
inline NewtonReport DampedNewtonSolve(
   HDGOperator &op, HDGState &state,
   const NewtonConfig &config,
   const NewtonOutput &output = NewtonOutput(),
   const NewtonPrepare &prepare = NewtonPrepare())
{
   if (config.max_iterations <= 0 || !(config.tolerance > 0.0))
   {
      throw std::runtime_error("invalid Newton iteration controls");
   }
   if (config.pseudo_transient &&
       !(config.initial_pseudo_time_step > 0.0))
   {
      throw std::runtime_error("invalid initial pseudo-time step");
   }
   if (!(config.armijo_c1 > 0.0) || !(config.armijo_c1 < 1.0) ||
       !(config.alpha_min > 0.0) || !(config.alpha_min <= 1.0))
   {
      throw std::runtime_error("invalid Armijo line-search controls");
   }

   NewtonReport report;
   const auto solve_start = std::chrono::steady_clock::now();
   const auto seconds_since = [](
      const std::chrono::steady_clock::time_point &start)
   {
      return std::chrono::duration<double>(
         std::chrono::steady_clock::now() - start).count();
   };
   double pseudo_time_step =
      config.pseudo_transient ? config.initial_pseudo_time_step : 0.0;
   double last_residual = std::numeric_limits<double>::infinity();
   for (int iteration = 0; iteration <= config.max_iterations; ++iteration)
   {
      if (prepare) { prepare(iteration, state, last_residual); }
      const double inverse_step =
         pseudo_time_step > 0.0 ? 1.0 / pseudo_time_step : 0.0;
      const auto assembly_start = std::chrono::steady_clock::now();
      const HDGResidualNorms old_norms =
         op.Assemble(state, true, inverse_step);
      report.assembly_seconds += seconds_since(assembly_start);
      const double old_residual = old_norms.Total();
      last_residual = old_residual;
      if (!std::isfinite(old_residual))
      {
         throw std::runtime_error("Newton residual is NaN or infinite");
      }
      if (output) { output(iteration, state, old_residual); }
      if (iteration == 0)
      {
         report.history.push_back(
            {0, old_residual, 0.0, pseudo_time_step});
      }
      report.residual = old_residual;
      report.iterations = iteration;
      if (old_residual < config.tolerance)
      {
         report.converged = true;
         report.total_seconds = seconds_since(solve_start);
         return report;
      }
      if (iteration == config.max_iterations) { break; }

      mfem::Vector true_increment, trace_increment;
      const auto linear_start = std::chrono::steady_clock::now();
      try
      {
         SolveCondensedPetscDirect(
            op.CondensedParMatrix(), op.CondensedTrueRHS(),
            true_increment);
      }
      catch (const std::exception &error)
      {
         report.linear_solve_seconds += seconds_since(linear_start);
         report.failure = error.what();
         report.total_seconds = seconds_since(solve_start);
         return report;
      }
      report.linear_solve_seconds += seconds_since(linear_start);
      op.ExpandTraceIncrement(true_increment, trace_increment);
      mfem::Vector volume_increment;
      op.RecoverIncrement(trace_increment, volume_increment);

      const HDGState base = state;
      const bool ptc_active = pseudo_time_step > 0.0;
      double alpha = 1.0;
      double new_residual = std::numeric_limits<double>::infinity();
      bool accepted = false;
      while (alpha >= config.alpha_min)
      {
         state = base;
         state.u.Add(alpha, volume_increment);
         state.uhat.Add(alpha, trace_increment);
         op.RecomputeGradient(state);
         const auto residual_start = std::chrono::steady_clock::now();
         const HDGResidualNorms new_norms = op.Assemble(state, false);
         report.assembly_seconds += seconds_since(residual_start);
         new_residual = new_norms.Total();
         const bool admissible =
            !config.require_admissible ||
            (op.MinimumDensity(state) > 0.0 &&
             op.MinimumPressure(state) > 0.0);
         if (std::isfinite(new_residual) && admissible)
         {
            if (ptc_active)
            {
               // Monotone PTC: accept plain decrease (the exactly-solved
               // damped step reduces the residual by only O(dtau), so the
               // Armijo sufficient-decrease test is too strict here);
               // rejection at every alpha shrinks dtau below.
               if (new_residual < old_residual)
               {
                  accepted = true;
                  break;
               }
            }
            else if (new_residual <=
                     (1.0 - config.armijo_c1 * alpha) * old_residual)
            {
               accepted = true;
               break;
            }
         }
         alpha *= 0.5;
      }

      if (!accepted)
      {
         state = base;
         op.RecomputeGradient(state);
         if (pseudo_time_step > 0.0)
         {
            pseudo_time_step *= config.ptc_reject_factor;
            report.history.push_back(
               {iteration + 1, old_residual, 0.0, pseudo_time_step});
            if (pseudo_time_step < config.minimum_pseudo_time_step)
            {
               report.failure =
                  "PTC pseudo-time step underflowed after repeated"
                  " line-search rejections";
               report.total_seconds = seconds_since(solve_start);
               return report;
            }
            continue;
         }
         report.failure =
            "Armijo line search failed to find a descent step";
         report.total_seconds = seconds_since(solve_start);
         return report;
      }

      if (pseudo_time_step > 0.0)
      {
         if (new_residual < config.ptc_off_residual)
         {
            pseudo_time_step = 0.0;
         }
         else if (new_residual > 0.0)
         {
            pseudo_time_step *= old_residual / new_residual;
            pseudo_time_step =
               std::min(std::max(pseudo_time_step,
                                 config.minimum_pseudo_time_step),
                        config.maximum_pseudo_time_step);
         }
      }
      report.history.push_back(
         {iteration + 1, new_residual, alpha, pseudo_time_step});
   }
   report.total_seconds = seconds_since(solve_start);
   return report;
}

} // namespace hycfd

#pragma once

#include "hdg_ns_operator.hpp"

#include "mfem.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace hdg_ns
{

struct NewtonConfig
{
   int max_iterations = 30;
   double tolerance = 1.0e-6;
   bool pseudo_transient = false;
   double initial_pseudo_time_step = 1.0e-3;
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
   std::vector<NewtonIteration> history;
};

using NewtonOutput =
   std::function<void(int iteration, const HDGState &state,
                      double residual)>;

inline void SolveCondensedPetscDirect(
   const mfem::SparseMatrix &matrix, const mfem::Vector &right_hand_side,
   mfem::Vector &solution)
{
   if (mfem::Mpi::WorldSize() != 1)
   {
      throw std::runtime_error(
         "M2 serial condensed PETSc solve supports np=1 only");
   }
   mfem::PetscParMatrix petsc_matrix(
      &matrix, mfem::Operator::PETSC_MATAIJ);
   mfem::PetscLinearSolver solver(petsc_matrix, "", false);
   solver.SetPrintLevel(0);
   solution.SetSize(right_hand_side.Size());
   solution = 0.0;
   solver.Mult(right_hand_side, solution);
   if (!solver.GetConverged())
   {
      throw std::runtime_error(
         "PETSc direct trace solve did not converge: iterations=" +
         std::to_string(solver.GetNumIterations()) +
         " residual=" + std::to_string(solver.GetFinalNorm()));
   }
}

inline NewtonReport DampedNewtonSolve(
   HDGNavierStokesOperator &op, HDGState &state,
   const NewtonConfig &config,
   const NewtonOutput &output = NewtonOutput())
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

   NewtonReport report;
   double pseudo_time_step =
      config.pseudo_transient ? config.initial_pseudo_time_step : 0.0;
   for (int iteration = 0; iteration <= config.max_iterations; ++iteration)
   {
      const double inverse_step =
         pseudo_time_step > 0.0 ? 1.0 / pseudo_time_step : 0.0;
      const HDGResidualNorms old_norms =
         op.Assemble(state, true, inverse_step);
      const double old_residual = old_norms.Total();
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
         return report;
      }
      if (iteration == config.max_iterations) { break; }

      mfem::Vector trace_increment;
      SolveCondensedPetscDirect(
         op.CondensedMatrix(), op.CondensedRHS(), trace_increment);
      mfem::Vector volume_increment;
      op.RecoverIncrement(trace_increment, volume_increment);

      const HDGState base = state;
      double alpha = 1.0;
      double new_residual = std::numeric_limits<double>::infinity();
      while (true)
      {
         state = base;
         state.u.Add(alpha, volume_increment);
         state.uhat.Add(alpha, trace_increment);
         op.RecomputeGradient(state);
         const HDGResidualNorms new_norms = op.Assemble(state, false);
         new_residual = new_norms.Total();

         if (!std::isfinite(new_residual))
         {
            throw std::runtime_error(
               "Newton trial residual is NaN or infinite");
         }
         if (new_residual > old_residual && new_residual > 1.0e6)
         {
            throw std::runtime_error(
               "Newton trial residual increased above 1e6");
         }
         // Exact Exasim semantics: test alpha before halving. This accepts
         // alpha=0.0625 even when the residual still increases.
         if (new_residual > old_residual && alpha > 0.1)
         {
            alpha /= 2.0;
            continue;
         }
         break;
      }

      if (pseudo_time_step > 0.0)
      {
         if (new_residual < 1.0e-2)
         {
            pseudo_time_step = 0.0;
         }
         else if (new_residual > 0.0)
         {
            pseudo_time_step *= old_residual / new_residual;
            pseudo_time_step =
               std::min(std::max(pseudo_time_step, 1.0e-12), 1.0e12);
         }
      }
      report.history.push_back(
         {iteration + 1, new_residual, alpha, pseudo_time_step});
   }
   return report;
}

} // namespace hdg_ns

#ifndef NEWTON_PETSC_SOLVER_HPP
#define NEWTON_PETSC_SOLVER_HPP

#include "mfem.hpp"

#ifndef MFEM_USE_PETSC
#error "PetscNewtonSolver requires MFEM built with PETSc."
#endif
#include <petscksp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace newton_utils
{

using steady_clock_t = std::chrono::steady_clock;

inline double ElapsedSec(const steady_clock_t::time_point &t0,
                         const steady_clock_t::time_point &t1)
{
   return std::chrono::duration<double>(t1 - t0).count();
}

struct NewtonConfig
{
   double abs_tol = 1.0e-10;
   double rel_tol = 1.0e-8;
   int max_iter = 20;
   // Rebuild Jacobian every N Newton iterations (1 = every iteration).
   int jacobian_rebuild_freq = 1;
};

struct PetscLinearConfig
{
   std::string ksp_prefix = "newton_ls_";
   int ksp_print_level = 0;
};

struct NewtonIterationInfo
{
   int iter = 0;
   double residual_norm = 0.0;
   double residual_norm0 = 1.0;
   double relative_residual = 0.0;
   double update_norm = 0.0;
   double update_norm0 = 1.0;
   double relative_update = 0.0;
   bool converged = false;
};

struct NewtonTiming
{
   double solve_sec = 0.0;
   double residual_eval_sec = 0.0;
   double jacobian_sec = 0.0;
   double linear_sec = 0.0;
   double update_sec = 0.0;
};

struct NewtonSolveResult
{
   bool converged = false;
   int iterations = 0;
   double final_residual = std::numeric_limits<double>::infinity();
   double initial_residual = 1.0;
   double final_relative_residual = std::numeric_limits<double>::infinity();
   double final_update_norm = 0.0;
   double initial_update_norm = 1.0;
   double final_relative_update = 0.0;
   NewtonTiming timing;
};

inline double GlobalNormL2(const mfem::Vector &v, MPI_Comm comm)
{
   return std::sqrt(mfem::InnerProduct(comm, v, v));
}

inline const char *ResolvePetscOptionsFile(const std::string &path,
                                           const int myid,
                                           std::string &resolved_path_storage)
{
   resolved_path_storage.clear();
   if (path.empty())
   {
      return nullptr;
   }

   std::ifstream in(path);
   if (in.good())
   {
      resolved_path_storage = path;
      return resolved_path_storage.c_str();
   }

   if (myid == 0)
   {
      std::cerr << "PETSc options file not found: " << path
                << ". Proceeding without options file." << std::endl;
   }
   return nullptr;
}

class PetscNewtonSolver
{
public:
   PetscNewtonSolver(MPI_Comm comm,
                     NewtonConfig newton_cfg,
                     PetscLinearConfig linear_cfg)
      : comm_(comm),
        newton_cfg_(std::move(newton_cfg)),
        linear_cfg_(std::move(linear_cfg))
   {
   }

   template <typename NonlinearOperatorType,
             typename EnforceBCFunc,
             typename IterationLoggerFunc>
   NewtonSolveResult Solve(NonlinearOperatorType &residual_operator,
                           mfem::Vector &x,
                           EnforceBCFunc enforce_bc,
                           IterationLoggerFunc log_iteration,
                           const int step = -1) const
   {
      auto pre_residual_hook = [](const int, mfem::Vector &) {};
      return SolveImpl(residual_operator,
                       x,
                       enforce_bc,
                       log_iteration,
                       pre_residual_hook,
                       step);
   }

   template <typename NonlinearOperatorType,
             typename EnforceBCFunc,
             typename IterationLoggerFunc,
             typename PreResidualHookFunc>
   NewtonSolveResult Solve(NonlinearOperatorType &residual_operator,
                           mfem::Vector &x,
                           EnforceBCFunc enforce_bc,
                           IterationLoggerFunc log_iteration,
                           PreResidualHookFunc pre_residual_hook,
                           const int step = -1) const
   {
      return SolveImpl(residual_operator,
                       x,
                       enforce_bc,
                       log_iteration,
                       pre_residual_hook,
                       step);
   }

private:
   template <typename NonlinearOperatorType,
             typename EnforceBCFunc,
             typename IterationLoggerFunc,
             typename PreResidualHookFunc>
   NewtonSolveResult SolveImpl(NonlinearOperatorType &residual_operator,
                               mfem::Vector &x,
                               EnforceBCFunc enforce_bc,
                               IterationLoggerFunc log_iteration,
                               PreResidualHookFunc pre_residual_hook,
                               const int step) const
   {
      const auto solve_t0 = steady_clock_t::now();
      NewtonSolveResult result;
      double r0 = 1.0;
      double du0 = 1.0;
      mfem::Operator *jacobian = nullptr;
      const int jacobian_rebuild_freq = std::max(1, newton_cfg_.jacobian_rebuild_freq);

      for (int iter = 0; iter < newton_cfg_.max_iter; ++iter)
      {
         pre_residual_hook(iter, x);

         const auto residual_t0 = steady_clock_t::now();
         mfem::Vector residual(x.Size());
         residual = 0.0;
         residual_operator.Mult(x, residual);
         result.timing.residual_eval_sec +=
            ElapsedSec(residual_t0, steady_clock_t::now());

         const double res_norm = GlobalNormL2(residual, comm_);
         if (iter == 0)
         {
            r0 = std::max(1.0, res_norm);
            result.initial_residual = r0;
         }
         const double rel_res = (r0 > 0.0) ? (res_norm / r0) : res_norm;
         result.final_residual = res_norm;
         result.final_relative_residual = rel_res;

         if (res_norm < newton_cfg_.abs_tol || rel_res < newton_cfg_.rel_tol)
         {
            result.converged = true;
            result.iterations = iter;

            NewtonIterationInfo info;
            info.iter = iter;
            info.residual_norm = res_norm;
            info.residual_norm0 = r0;
            info.relative_residual = rel_res;
            info.update_norm = 0.0;
            info.update_norm0 = du0;
            info.relative_update = 0.0;
            info.converged = true;
            log_iteration(info);
            break;
         }

         mfem::Vector rhs(residual);
         rhs *= -1.0;
         mfem::Vector dx(x.Size());
         dx = 0.0;

         if (iter == 0 || (iter % jacobian_rebuild_freq) == 0 || jacobian == nullptr)
         {
            const auto jac_t0 = steady_clock_t::now();
            jacobian = &residual_operator.GetGradient(x);
            result.timing.jacobian_sec += ElapsedSec(jac_t0, steady_clock_t::now());
         }

         const auto lin_t0 = steady_clock_t::now();
         SolveLinearSystem(*jacobian, rhs, dx, step, iter);
         result.timing.linear_sec += ElapsedSec(lin_t0, steady_clock_t::now());

         const auto update_t0 = steady_clock_t::now();
         const double update_norm = GlobalNormL2(dx, comm_);
         if (iter == 0)
         {
            du0 = std::max(1.0, update_norm);
            result.initial_update_norm = du0;
         }
         const double rel_update = (du0 > 0.0) ? (update_norm / du0) : update_norm;
         x += dx;
         enforce_bc(x);
         result.timing.update_sec += ElapsedSec(update_t0, steady_clock_t::now());
         result.final_update_norm = update_norm;
         result.final_relative_update = rel_update;

         NewtonIterationInfo info;
         info.iter = iter;
         info.residual_norm = res_norm;
         info.residual_norm0 = r0;
         info.relative_residual = rel_res;
         info.update_norm = update_norm;
         info.update_norm0 = du0;
         info.relative_update = rel_update;
         info.converged = false;
         log_iteration(info);
      }

      if (!result.converged)
      {
         result.iterations = newton_cfg_.max_iter;
      }
      result.timing.solve_sec = ElapsedSec(solve_t0, steady_clock_t::now());

      return result;
   }
   void SolveLinearSystem(mfem::Operator &jacobian,
                          const mfem::Vector &rhs,
                          mfem::Vector &dx,
                          const int step,
                          const int iter) const
   {
      if (mfem::PetscParMatrix *J_petsc =
             dynamic_cast<mfem::PetscParMatrix *>(&jacobian))
      {
         SolveWithPetscMatrix(*J_petsc, rhs, dx, step, iter);
         return;
      }

      if (mfem::HypreParMatrix *J_hypre =
             dynamic_cast<mfem::HypreParMatrix *>(&jacobian))
      {
         mfem::PetscParMatrix Jpetsc(J_hypre, mfem::Operator::PETSC_MATAIJ);
         SolveWithPetscMatrix(Jpetsc, rhs, dx, step, iter);
         return;
      }

      if (mfem::BlockOperator *J_block =
             dynamic_cast<mfem::BlockOperator *>(&jacobian))
      {
         const int n_row = J_block->NumRowBlocks();
         const int n_col = J_block->NumColBlocks();
         mfem::Array2D<const mfem::HypreParMatrix *> hypre_blocks(n_row, n_col);
         mfem::Array2D<mfem::real_t> block_coeff(n_row, n_col);
         bool all_hypre_blocks = true;

         for (int i = 0; i < n_row; ++i)
         {
            for (int j = 0; j < n_col; ++j)
            {
               if (J_block->IsZeroBlock(i, j))
               {
                  hypre_blocks(i, j) = nullptr;
                  block_coeff(i, j) = 0.0;
                  continue;
               }

               const mfem::Operator &blk = J_block->GetBlock(i, j);
               const mfem::HypreParMatrix *blk_hypre =
                  dynamic_cast<const mfem::HypreParMatrix *>(&blk);
               if (!blk_hypre)
               {
                  all_hypre_blocks = false;
                  break;
               }

               hypre_blocks(i, j) = blk_hypre;
               block_coeff(i, j) = J_block->GetBlockCoef(i, j);
            }
            if (!all_hypre_blocks) { break; }
         }

         if (all_hypre_blocks)
         {
            std::unique_ptr<mfem::HypreParMatrix> J_hypre_merged(
               mfem::HypreParMatrixFromBlocks(hypre_blocks, &block_coeff));
            mfem::PetscParMatrix Jpetsc(J_hypre_merged.get(),
                                        mfem::Operator::PETSC_MATAIJ);
            SolveWithPetscMatrix(Jpetsc, rhs, dx, step, iter);
            return;
         }

         mfem::PetscParMatrix Jpetsc(comm_, J_block, mfem::Operator::PETSC_MATAIJ);
         SolveWithPetscMatrix(Jpetsc, rhs, dx, step, iter);
         return;
      }

      throw std::runtime_error(
         "Unsupported Jacobian operator type in PETSc Newton linear solve.");
   }

   void SolveWithPetscMatrix(const mfem::PetscParMatrix &A,
                             const mfem::Vector &rhs,
                             mfem::Vector &dx,
                             const int step,
                             const int iter) const
   {
      mfem::PetscLinearSolver linear_solver(A, linear_cfg_.ksp_prefix, false);
      linear_solver.SetPrintLevel(linear_cfg_.ksp_print_level);
      linear_solver.Mult(rhs, dx);

      if (!linear_solver.GetConverged())
      {
         throw std::runtime_error(
            "PETSc linear solve did not converge. step=" +
            std::to_string(step) +
            ", iter=" + std::to_string(iter) +
            ", iterations=" + std::to_string(linear_solver.GetNumIterations()) +
            ", residual=" + std::to_string(linear_solver.GetFinalNorm()));
      }
   }

   MPI_Comm comm_;
   NewtonConfig newton_cfg_;
   PetscLinearConfig linear_cfg_;
};

} // namespace newton_utils

#endif

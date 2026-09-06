// Damped Newton with Armijo backtracking on a plain mfem::Operator.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "base/config.hpp"
#include "mfem.hpp"

namespace cmf
{

struct NewtonIteration
{
  int iteration = 0;
  double residual = 0.0;
  double alpha = 0.0;
};

struct NewtonReport
{
  bool converged = false;
  int iterations = 0;
  int linear_solve_failures = 0; // Krylov solves that hit max_it (if known)
  double initial_residual = 0.0;
  double residual = 0.0;
  std::string failure;
  std::vector<NewtonIteration> history;
};

using NewtonMonitor = std::function<void(const NewtonIteration &)>;

// Solves R(x) = 0 with op.Mult = R and op.GetGradient = dR/dx. The linear
// solver receives each Jacobian through SetOperator. The residual norm is the
// global l2 norm over comm. Converged when ||R|| <= atol or
// ||R|| <= rtol ||R_0||; the step is halved while
// ||R(x - alpha dx)|| > (1 - c alpha) ||R(x)||, at most max_halvings times.
NewtonReport DampedNewtonSolve(mfem::Operator &op, mfem::Solver &linear_solver,
                               mfem::Vector &x, const NewtonConfig &cfg,
                               MPI_Comm comm,
                               const NewtonMonitor &monitor = NewtonMonitor());

} // namespace cmf

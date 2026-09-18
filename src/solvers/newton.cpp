#include "solvers/newton.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

#include "solvers/linear_solver.hpp"
#include "solvers/saddle_point_solver.hpp"

namespace cmf
{

namespace
{

double GlobalNorm(MPI_Comm comm, const mfem::Vector &v)
{
  return std::sqrt(mfem::InnerProduct(comm, v, v));
}

// Convergence of the last linear solve when the solver can report it.
bool LinearSolveConverged(const mfem::Solver &solver)
{
  if (const auto *ours = dynamic_cast<const LinearSolver *>(&solver))
  {
    return ours->Converged();
  }
  if (const auto *saddle = dynamic_cast<const SaddlePointSolver *>(&solver))
  {
    return saddle->Converged();
  }
  if (const auto *it = dynamic_cast<const mfem::IterativeSolver *>(&solver))
  {
    return it->GetConverged();
  }
  return true;
}

} // namespace

NewtonReport DampedNewtonSolve(mfem::Operator &op, mfem::Solver &linear_solver,
                               mfem::Vector &x, const NewtonConfig &cfg,
                               MPI_Comm comm, const NewtonMonitor &monitor)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  const bool verbose = cfg.print_level > 0 && rank == 0;

  NewtonReport report;
  const int n = x.Size();
  mfem::Vector r(n), dx(n), x_trial(n), r_trial(n);

  op.Mult(x, r);
  double rn = GlobalNorm(comm, r);
  report.initial_residual = rn;
  report.residual = rn;
  report.history.push_back({0, rn, 0.0});
  if (monitor) { monitor(report.history.back()); }
  if (verbose) { std::printf("newton it %2d: |R| = %.6e\n", 0, rn); }

  for (int it = 0; it <= cfg.max_it; it++)
  {
    if (!std::isfinite(rn))
    {
      report.failure = "residual is not finite";
      break;
    }
    if (rn <= cfg.atol || rn <= cfg.rtol * report.initial_residual)
    {
      report.converged = true;
      break;
    }
    if (it == cfg.max_it)
    {
      report.failure = "maximum Newton iterations reached";
      break;
    }

    mfem::Operator &J = op.GetGradient(x);
    linear_solver.SetOperator(J);
    dx = 0.0;
    linear_solver.Mult(r, dx);
    if (!LinearSolveConverged(linear_solver))
    {
      report.linear_solve_failures++;
      if (verbose)
      {
        std::printf("newton it %2d: warning, linear solve did not converge; "
                    "continuing with the inexact step\n", it + 1);
      }
    }

    double alpha = 1.0;
    bool accepted = false;
    double rn_trial = std::numeric_limits<double>::infinity();
    for (int h = 0; h <= cfg.max_halvings; h++)
    {
      x_trial = x;
      x_trial.Add(-alpha, dx);
      op.Mult(x_trial, r_trial);
      rn_trial = GlobalNorm(comm, r_trial);
      if (std::isfinite(rn_trial) && rn_trial <= (1.0 - cfg.armijo_c * alpha) * rn)
      {
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    // Linear problem, second or later step, every linear solve converged: a
    // step that cannot reduce the residual (or, below, reduces it by less
    // than ten) has met the round-off floor of the residual evaluation.
    const bool linear_floor = cfg.linear_problem && report.iterations >= 1 &&
                              report.linear_solve_failures == 0;
    if (!accepted && linear_floor)
    {
      report.converged = true;
      report.at_floor = true;
      break;
    }
    if (!accepted)
    {
      char buf[160];
      std::snprintf(buf, sizeof(buf),
                    "line search failed to reduce the residual (|R|/|R0| = %.2e; "
                    "the tolerance may lie below the round-off floor)",
                    rn / report.initial_residual);
      report.failure = buf;
      break;
    }
    const double rn_before = rn;
    x = x_trial;
    r = r_trial;
    rn = rn_trial;
    report.iterations = it + 1;
    report.residual = rn;
    report.history.push_back({it + 1, rn, alpha});
    if (monitor) { monitor(report.history.back()); }
    if (verbose)
    {
      std::printf("newton it %2d: |R| = %.6e  alpha = %.4f\n", it + 1, rn, alpha);
    }
    if (linear_floor && rn > 0.1 * rn_before && rn > cfg.atol && rn > cfg.rtol * report.initial_residual)
    {
      report.converged = true;
      report.at_floor = true;
      break;
    }
  }
  if (verbose && report.at_floor)
  {
    std::printf("newton: linear problem, the residual is at its round-off floor "
                "(|R|/|R0| = %.2e); accepted\n", rn / report.initial_residual);
  }
  if (verbose && !report.failure.empty())
  {
    std::printf("newton: %s\n", report.failure.c_str());
  }
  return report;
}

} // namespace cmf

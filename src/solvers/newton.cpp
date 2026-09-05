#include "solvers/newton.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

namespace cmf
{

namespace
{

double GlobalNorm(MPI_Comm comm, const mfem::Vector &v)
{
  return std::sqrt(mfem::InnerProduct(comm, v, v));
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
    if (!accepted)
    {
      report.failure = "line search failed to reduce the residual";
      break;
    }
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
  }
  if (verbose && !report.failure.empty())
  {
    std::printf("newton: %s\n", report.failure.c_str());
  }
  return report;
}

} // namespace cmf

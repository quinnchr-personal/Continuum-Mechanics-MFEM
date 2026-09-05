#include "solvers/quasi_static.hpp"

#include <cstdio>

namespace cmf
{

QuasiStaticReport SolveQuasiStatic(QuasiStaticProblem &problem,
                                   mfem::Solver &linear_solver,
                                   const SolverConfig &cfg, mfem::Vector &x,
                                   const LoadStepCallback &on_step)
{
  int rank = 0;
  MPI_Comm_rank(problem.Comm(), &rank);
  QuasiStaticReport report;
  report.converged = true;
  for (int step = 1; step <= cfg.load_steps; step++)
  {
    const double lambda = double(step) / double(cfg.load_steps);
    if (rank == 0 && cfg.newton.print_level > 0)
    {
      std::printf("load step %d/%d: lambda = %.6f\n", step, cfg.load_steps, lambda);
    }
    problem.SetLoadFactor(lambda);
    problem.ApplyDirichlet(x);
    LoadStepReport s;
    s.step = step;
    s.load_factor = lambda;
    s.newton = DampedNewtonSolve(problem, linear_solver, x, cfg.newton,
                                 problem.Comm());
    report.steps.push_back(s);
    if (on_step) { on_step(s, x); }
    if (!s.newton.converged)
    {
      report.converged = false;
      break;
    }
  }
  return report;
}

} // namespace cmf

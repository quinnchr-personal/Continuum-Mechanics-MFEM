#include "solvers/quasi_static.hpp"

#include <algorithm>
#include <cstdio>

namespace cmf
{

std::vector<double> LoadStepBreakpoints(const SolverConfig &cfg)
{
  if (!cfg.breakpoints.empty()) { return cfg.breakpoints; }
  std::vector<double> b;
  for (int step = 1; step <= cfg.load_steps; step++)
  {
    b.push_back(double(step) / double(cfg.load_steps));
  }
  return b;
}

QuasiStaticReport SolveQuasiStatic(QuasiStaticProblem &problem,
                                   mfem::Solver &linear_solver,
                                   const SolverConfig &cfg, mfem::Vector &x,
                                   const LoadStepCallback &on_step)
{
  int rank = 0;
  MPI_Comm_rank(problem.Comm(), &rank);
  const bool verbose = rank == 0 && cfg.newton.print_level > 0;
  const std::vector<double> targets = LoadStepBreakpoints(cfg);
  QuasiStaticReport report;
  report.converged = true;
  mfem::Vector x_last(x);
  double t = 0.0;
  int accepted = 0;
  for (std::size_t k = 0; k < targets.size(); k++)
  {
    const double target = targets[k];
    double dt = target - t;
    int attempts = 0;
    int bisections = 0;
    while (t < target - 1e-14)
    {
      const double t_try = std::min(t + dt, target);
      if (verbose)
      {
        std::printf("load step %d/%zu: t = %.6f -> %.6f\n", accepted + 1, targets.size(),
                    t, t_try);
      }
      problem.SetLoadFactor(t_try);
      problem.ApplyDirichlet(x);
      attempts++;
      LoadStepReport s;
      s.step = accepted + 1;
      s.load_factor = t_try;
      s.t_begin = t;
      s.attempts = attempts;
      s.newton = DampedNewtonSolve(problem, linear_solver, x, cfg.newton, problem.Comm());
      if (s.newton.converged)
      {
        accepted++;
        t = t_try;
        x_last = x;
        report.steps.push_back(s);
        if (on_step) { on_step(s, x); }
        attempts = 0;
        dt = target - t; // back to the planned breakpoint
        continue;
      }
      const bool can_bisect = cfg.substep.on_failure &&
                              bisections < cfg.substep.max_bisections &&
                              0.5 * dt >= cfg.substep.min_dt;
      if (!can_bisect)
      {
        report.steps.push_back(s);
        report.converged = false;
        return report;
      }
      x = x_last;
      dt *= 0.5;
      bisections++;
      report.bisections++;
      if (verbose)
      {
        std::printf("load step %d: Newton failed (%s); bisecting to dt = %.6f\n",
                    accepted + 1, s.newton.failure.c_str(), dt);
      }
    }
  }
  return report;
}

} // namespace cmf

#include "solvers/quasi_static.hpp"

#include <algorithm>
#include <cmath>
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

namespace
{

double GlobalMaxAbs(MPI_Comm comm, const mfem::Vector &v)
{
  double local = v.Size() ? v.Normlinf() : 0.0, global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm);
  return global;
}

// Tangent predictor: with x the last converged state (old Dirichlet values)
// and the loads of the problem already at the new pseudo-time, d = g - x is
// the Dirichlet increment (g = x with the new data applied; nonzero on the
// essential dofs only). One linear solve about x,
//   J(x) s = R(x) + J(x) d,      x <- x + d - s,
// imposes the increment on the update instead of on the state: the Jacobian
// is that of the converged, undistorted configuration. J(x) d is formed by a
// forward difference of the residual, since the assembled Jacobian has its
// essential columns eliminated. Returns false (x = g, the plain start) when
// there is no increment, the linear solve fails, or the result is not finite.
bool TangentPredictor(QuasiStaticProblem &problem, mfem::Solver &linear_solver, mfem::Vector &x)
{
  const MPI_Comm comm = problem.Comm();
  mfem::Vector g(x);
  problem.ApplyDirichlet(g);
  mfem::Vector d(g);
  d -= x;
  const double dmax = GlobalMaxAbs(comm, d);
  if (dmax == 0.0) { x = g; return false; }
  const double eps = 1e-6 * std::max(dmax, GlobalMaxAbs(comm, x)) / dmax;
  const int n = x.Size();
  mfem::Vector r0(n), r1(n), xe(x), s(n);
  problem.Mult(x, r0);
  xe.Add(eps, d);
  problem.Mult(xe, r1);
  r1 -= r0;
  r1 /= eps;       // J d on the free rows (essential rows are zeroed by Mult)
  r1 += r0;        // R + J d
  mfem::Operator &J = problem.GetGradient(x);
  linear_solver.SetOperator(J);
  s = 0.0;
  linear_solver.Mult(r1, s);
  mfem::Vector trial(g);
  trial -= s;
  problem.ApplyDirichlet(trial); // essential entries exactly the prescribed ones
  problem.Mult(trial, r1);
  const double rn = std::sqrt(mfem::InnerProduct(comm, r1, r1));
  if (!std::isfinite(rn)) { x = g; return false; }
  x = trial;
  return true;
}

} // namespace

QuasiStaticReport SolveQuasiStatic(QuasiStaticProblem &problem,
                                   mfem::Solver &linear_solver,
                                   const SolverConfig &cfg, mfem::Vector &x,
                                   const LoadStepCallback &on_step)
{
  int rank = 0;
  MPI_Comm_rank(problem.Comm(), &rank);
  const bool verbose = rank == 0 && cfg.newton.print_level > 0;
  const std::vector<double> targets = LoadStepBreakpoints(cfg);
  NewtonConfig newton = cfg.newton;
  newton.linear_problem = problem.IsLinear();
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
      if (cfg.predictor == "tangent") { TangentPredictor(problem, linear_solver, x); }
      else { problem.ApplyDirichlet(x); }
      attempts++;
      LoadStepReport s;
      s.step = accepted + 1;
      s.load_factor = t_try;
      s.t_begin = t;
      s.attempts = attempts;
      s.newton = DampedNewtonSolve(problem, linear_solver, x, newton, problem.Comm());
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

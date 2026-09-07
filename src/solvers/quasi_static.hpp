// Load stepping in the pseudo-time t in (0, 1]: the problem evaluates its
// load schedules at t, each increment is solved by damped Newton warm-started
// from the previous one, and a failed increment is bisected when the solver
// config allows it (SubstepConfig).
#pragma once

#include <functional>
#include <vector>

#include "base/config.hpp"
#include "mfem.hpp"
#include "solvers/newton.hpp"

namespace cmf
{

// What the load stepper needs from a physics module, on top of mfem::Operator.
class QuasiStaticProblem : public mfem::Operator
{
public:
  using mfem::Operator::Operator;
  // Set the pseudo-time t of every load (tractions, body force, prescribed
  // displacements follow their schedules s_i(t)).
  virtual void SetLoadFactor(double t) = 0;
  virtual double LoadFactor() const = 0;
  // Overwrite the essential true dofs of x with the scaled Dirichlet data.
  virtual void ApplyDirichlet(mfem::Vector &x) const = 0;
  virtual MPI_Comm Comm() const = 0;
};

// One accepted increment (or the final failed one): t_begin -> load_factor,
// after `attempts` Newton solves (1 unless bisected).
struct LoadStepReport
{
  int step = 0;
  double load_factor = 0.0; // t at the end of the increment
  double t_begin = 0.0;
  int attempts = 1;
  NewtonReport newton;
};

struct QuasiStaticReport
{
  bool converged = false;
  std::vector<LoadStepReport> steps;
  int bisections = 0; // total number of halved increments
};

// The breakpoints of cfg (cfg.breakpoints, or load_steps equal increments).
std::vector<double> LoadStepBreakpoints(const SolverConfig &cfg);

using LoadStepCallback =
  std::function<void(const LoadStepReport &, const mfem::Vector &x)>;

QuasiStaticReport SolveQuasiStatic(QuasiStaticProblem &problem,
                                   mfem::Solver &linear_solver,
                                   const SolverConfig &cfg, mfem::Vector &x,
                                   const LoadStepCallback &on_step = LoadStepCallback());

} // namespace cmf

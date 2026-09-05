// Load stepping: all loads scaled by lambda in (0, 1] over n increments, each
// solved by damped Newton warm-started from the previous increment.
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
  // Scale every load (tractions, body force, prescribed displacements).
  virtual void SetLoadFactor(double lambda) = 0;
  virtual double LoadFactor() const = 0;
  // Overwrite the essential true dofs of x with the scaled Dirichlet data.
  virtual void ApplyDirichlet(mfem::Vector &x) const = 0;
  virtual MPI_Comm Comm() const = 0;
};

struct LoadStepReport
{
  int step = 0;
  double load_factor = 0.0;
  NewtonReport newton;
};

struct QuasiStaticReport
{
  bool converged = false;
  std::vector<LoadStepReport> steps;
};

using LoadStepCallback =
  std::function<void(const LoadStepReport &, const mfem::Vector &x)>;

QuasiStaticReport SolveQuasiStatic(QuasiStaticProblem &problem,
                                   mfem::Solver &linear_solver,
                                   const SolverConfig &cfg, mfem::Vector &x,
                                   const LoadStepCallback &on_step = LoadStepCallback());

} // namespace cmf

// YAML-selected linear solver: Krylov (GMRES or CG) preconditioned by
// BoomerAMG with elasticity (rigid-body-mode) or plain systems options for a
// vector unknown, or BoomerAMG's defaults for a scalar one (amg: scalar, the
// only mode of a space with one component; chosen automatically there when
// the config carries the vector default).
// MFEM's Krylov tolerances are relative to the left-preconditioned residual
// ||M^{-1} r||, which with AMG behaves like an error norm; the true residual
// reduction is typically 10-100x weaker than rtol (measured in S4).
//
// Operator stamp: a counter shared between a problem and the linear solver it
// made (SolidProblem::MakeLinearSolver). The problem increments it whenever
// the matrix it returns from GetGradient has been (re)assembled. A solver
// handed the same operator object with an unchanged stamp keeps its setup. A
// nonlinear problem assembles, and stamps, on every call, so nothing changes
// for it; a linear problem assembles its constant Jacobian once, and the AMG
// hierarchy is then built once for the whole load path. The pointer alone
// would not do: after new boundary conditions the reassembled matrix can sit
// at the address of the old one. Without a stamp every SetOperator rebuilds.
#pragma once

#include <memory>

#include "base/config.hpp"
#include "mfem.hpp"

namespace cmf
{

using OperatorStamp = std::shared_ptr<const long>;

class LinearSolver : public mfem::Solver
{
public:
  LinearSolver(const LinearSolverConfig &cfg, mfem::ParFiniteElementSpace &fes);

  // op must be a HypreParMatrix; the AMG hierarchy is rebuilt for it, unless
  // it is the operator of the last call with an unchanged stamp.
  void SetOperator(const mfem::Operator &op) override;
  void SetOperatorStamp(OperatorStamp stamp) { stamp_ = std::move(stamp); }
  // Krylov solve. If the elasticity AMG options do not converge, the solve
  // is repeated once with the plain systems options, which then stay in
  // effect for this solver (the plan's sanctioned fallback).
  void Mult(const mfem::Vector &b, mfem::Vector &x) const override;

  bool Converged() const { return krylov_->GetConverged(); }
  int Iterations() const { return krylov_->GetNumIterations(); }
  double FinalNorm() const { return krylov_->GetFinalNorm(); }
  const LinearSolverConfig &Config() const { return cfg_; }

  // AMG options currently in effect ("elasticity" or "systems").
  const std::string &ActiveAMG() const { return amg_mode_; }
  // Number of AMG hierarchies built so far (the systems fallback included).
  int Setups() const { return setups_; }

private:
  void BuildPreconditioner(const std::string &mode) const;

  LinearSolverConfig cfg_;
  mfem::ParFiniteElementSpace &fes_;
  const mfem::HypreParMatrix *A_ = nullptr;
  mutable std::string amg_mode_;
  mutable std::unique_ptr<mfem::HypreBoomerAMG> amg_;
  std::unique_ptr<mfem::IterativeSolver> krylov_;
  OperatorStamp stamp_;
  long seen_stamp_ = -1;
  mutable int setups_ = 0;
};

std::unique_ptr<LinearSolver> MakeLinearSolver(const LinearSolverConfig &cfg,
                                               mfem::ParFiniteElementSpace &fes);

} // namespace cmf

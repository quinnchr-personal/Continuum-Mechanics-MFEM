// YAML-selected linear solver: Krylov (GMRES or CG) preconditioned by
// BoomerAMG with elasticity (rigid-body-mode) or plain systems options.
#pragma once

#include <memory>

#include "base/config.hpp"
#include "mfem.hpp"

namespace cmf
{

class LinearSolver : public mfem::Solver
{
public:
  LinearSolver(const LinearSolverConfig &cfg, mfem::ParFiniteElementSpace &fes);

  // op must be a HypreParMatrix; the AMG hierarchy is rebuilt for it.
  void SetOperator(const mfem::Operator &op) override;
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

private:
  void BuildPreconditioner(const std::string &mode) const;

  LinearSolverConfig cfg_;
  mfem::ParFiniteElementSpace &fes_;
  const mfem::HypreParMatrix *A_ = nullptr;
  mutable std::string amg_mode_;
  mutable std::unique_ptr<mfem::HypreBoomerAMG> amg_;
  std::unique_ptr<mfem::IterativeSolver> krylov_;
};

std::unique_ptr<LinearSolver> MakeLinearSolver(const LinearSolverConfig &cfg,
                                               mfem::ParFiniteElementSpace &fes);

} // namespace cmf

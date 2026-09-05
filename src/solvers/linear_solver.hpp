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
  void Mult(const mfem::Vector &b, mfem::Vector &x) const override;

  bool Converged() const { return krylov_->GetConverged(); }
  int Iterations() const { return krylov_->GetNumIterations(); }
  double FinalNorm() const { return krylov_->GetFinalNorm(); }
  const LinearSolverConfig &Config() const { return cfg_; }

private:
  LinearSolverConfig cfg_;
  mfem::ParFiniteElementSpace &fes_;
  std::unique_ptr<mfem::HypreBoomerAMG> amg_;
  std::unique_ptr<mfem::IterativeSolver> krylov_;
};

std::unique_ptr<LinearSolver> MakeLinearSolver(const LinearSolverConfig &cfg,
                                               mfem::ParFiniteElementSpace &fes);

} // namespace cmf

// Linear solver for the mixed u-p Jacobian J = [[K, B], [B^T, -C]]
// (C = M_p / kappa, zero when incompressible).
//
// The system is first put in the algebraically equivalent augmented
// Lagrangian form (gamma = augmentation * mu, W = diag(M_p)^{-1}):
//   [[K + gamma B W B^T,  B - gamma B W C], [B^T, -C]] [u; p]
//     = [b_u + gamma B W b_p; b_p],
// whose displacement block stays well conditioned when the pressure is large
// (the raw K carries the indefinite p-weighted geometric stiffness). It is
// solved by FGMRES with the block upper-triangular preconditioner
//   p = -S~^{-1} r_p,   u = K~^{-1} (r_u - B~ p),
// S~ = M_p (kappa + mu) / (kappa (mu + gamma)) by CG + Jacobi, and K~^{-1}
// by an inner GMRES + BoomerAMG (systems options) to inner_rtol.
#pragma once

#include <memory>

#include "base/config.hpp"
#include "mfem.hpp"
#include "solvers/linear_solver.hpp"

namespace cmf
{

class SaddlePointSolver : public mfem::Solver
{
public:
  SaddlePointSolver(const LinearSolverConfig &cfg, mfem::ParFiniteElementSpace &fes_u,
                    const mfem::Array<int> &offsets, mfem::HypreParMatrix &pressure_mass,
                    double mu, double kappa);

  // op must be a 2x2 BlockOperator whose blocks are HypreParMatrix.
  void SetOperator(const mfem::Operator &op) override;
  void Mult(const mfem::Vector &b, mfem::Vector &x) const override;

  bool Converged() const { return outer_->GetConverged(); }
  int Iterations() const { return outer_->GetNumIterations(); }
  double Gamma() const { return gamma_; }

private:
  class BlockPreconditioner : public mfem::Solver
  {
  public:
    BlockPreconditioner(const SaddlePointSolver &owner) : owner_(owner) {}
    void SetOperator(const mfem::Operator &) override {}
    void Mult(const mfem::Vector &r, mfem::Vector &z) const override;
  private:
    const SaddlePointSolver &owner_;
  };

  LinearSolverConfig cfg_;
  MPI_Comm comm_;
  mfem::Array<int> offsets_;
  double mu_;
  double kappa_; // inf when incompressible
  double gamma_;
  mfem::HypreParMatrix &pressure_mass_;
  mfem::Vector mass_diag_;

  // Augmented operator and its blocks (rebuilt in SetOperator).
  const mfem::BlockOperator *jacobian_ = nullptr;
  std::unique_ptr<mfem::HypreParMatrix> BtW_;    // W B^T
  std::unique_ptr<mfem::HypreParMatrix> K_aug_;  // K + gamma B W B^T
  std::unique_ptr<mfem::HypreParMatrix> B_aug_;  // B - gamma B W C (or null: use B)
  std::unique_ptr<mfem::BlockOperator> A_aug_;

  std::unique_ptr<LinearSolver> stiffness_;
  std::unique_ptr<mfem::HypreParMatrix> scaled_mass_;
  std::unique_ptr<mfem::HypreSmoother> mass_prec_;
  std::unique_ptr<mfem::CGSolver> mass_solver_;
  std::unique_ptr<BlockPreconditioner> prec_;
  std::unique_ptr<mfem::FGMRESSolver> outer_;
  mutable int inner_iterations_ = 0;
  mutable int inner_failures_ = 0;
};

} // namespace cmf

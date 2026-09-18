// Linear solver for the mixed u-p Jacobian J = [[K, B], [Bt, -C]]
// (C = M_p / kappa, zero when incompressible; Bt = B^T for the quadratic
// volumetric law, Bt = u''(J)-weighted B^T otherwise, so J need not be symmetric).
//
// The system is first put in the algebraically equivalent augmented
// Lagrangian form (gamma = augmentation * mu, W = diag(M_p)^{-1}), adding
// gamma B W times the second block row to the first:
//   [[K + gamma B W Bt,  B - gamma B W C], [Bt, -C]] [u; p]
//     = [b_u + gamma B W b_p; b_p],
// whose displacement block stays well conditioned when the pressure is large
// (the raw K carries the indefinite p-weighted geometric stiffness). It is
// solved by FGMRES with the block upper-triangular preconditioner
//   p = -S~^{-1} r_p,   u = K~^{-1} (r_u - B~ p),
// S~ = M_p (kappa + mu) / (kappa (mu + gamma)) by CG + Jacobi, and K~^{-1}
// by an inner GMRES + BoomerAMG (systems options) to inner_rtol.
//
// Dynamic analysis (SetInertia): the displacement block is K + c_M M, and for
// small time steps the mass dominates it. The Schur complement of a
// mass-dominated block is a pressure Laplacian, Bt (c_M M)^{-1} B, which a
// scaled pressure mass does not represent: measured on the quarter annulus,
// the outer iterations rise from 18 (quasi-static) to 43 and, one refinement
// on, to 81 at dt = 0.002. Following Cahouet and Chabard the two limits are
// added,
//   S~^{-1} = [M_p (kappa + mu) / (kappa (mu + gamma))]^{-1} + [Bt D^{-1} B / c_M + C]^{-1},
// with D the lumped mass (diagonal of M scaled to the total mass), formed from
// the blocks of the Jacobian, so that it carries their boundary conditions
// (prescribed displacements: Neumann; free surfaces: Dirichlet), and solved by
// CG + BoomerAMG. Without SetInertia nothing of this exists.
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

  // op must be a 2x2 BlockOperator whose blocks are HypreParMatrix. The
  // augmented blocks and the AMG hierarchy of the displacement block are
  // rebuilt for it, unless it is the operator of the last call with an
  // unchanged stamp (linear_solver.hpp: a linear problem).
  void SetOperator(const mfem::Operator &op) override;
  void SetOperatorStamp(OperatorStamp stamp) { stamp_ = std::move(stamp); }
  // The displacement block of the operators to come is K + c_M M: lumped_mass
  // is the diagonal D of the displacement true dofs, *mass_factor the current
  // c_M (read at every setup; the owner changes it with the time step).
  void SetInertia(const mfem::Vector &lumped_mass, std::shared_ptr<const double> mass_factor)
  {
    lumped_mass_ = lumped_mass;
    mass_factor_ = std::move(mass_factor);
  }
  // Number of setups so far (augmented blocks + AMG hierarchy).
  int Setups() const { return setups_; }
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
  std::unique_ptr<mfem::HypreParMatrix> BW_;     // B W
  std::unique_ptr<mfem::HypreParMatrix> K_aug_;  // K + gamma B W Bt
  std::unique_ptr<mfem::HypreParMatrix> B_aug_;  // B - gamma B W C (or null: use B)
  std::unique_ptr<mfem::BlockOperator> A_aug_;

  // Inertial part of the Schur complement approximation (dynamic analysis).
  mfem::Vector lumped_mass_;
  std::shared_ptr<const double> mass_factor_;
  std::unique_ptr<mfem::HypreParMatrix> inertial_schur_; // Bt D^{-1} B / c_M + C
  std::unique_ptr<mfem::HypreBoomerAMG> inertial_amg_;
  std::unique_ptr<mfem::CGSolver> inertial_solver_;

  std::unique_ptr<LinearSolver> stiffness_;
  std::unique_ptr<mfem::HypreParMatrix> scaled_mass_;
  std::unique_ptr<mfem::HypreSmoother> mass_prec_;
  std::unique_ptr<mfem::CGSolver> mass_solver_;
  std::unique_ptr<BlockPreconditioner> prec_;
  std::unique_ptr<mfem::FGMRESSolver> outer_;
  mutable int inner_iterations_ = 0;
  mutable int inner_failures_ = 0;
  OperatorStamp stamp_;
  long seen_stamp_ = -1;
  int setups_ = 0;
};

} // namespace cmf

#include "solvers/saddle_point_solver.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace cmf
{

SaddlePointSolver::SaddlePointSolver(const LinearSolverConfig &cfg,
                                     mfem::ParFiniteElementSpace &fes_u,
                                     const mfem::Array<int> &offsets,
                                     mfem::HypreParMatrix &pressure_mass,
                                     double mu, double kappa)
  : mfem::Solver(offsets[2]), cfg_(cfg), comm_(fes_u.GetComm()), mu_(mu),
    kappa_(kappa), gamma_(cfg.augmentation * mu), pressure_mass_(pressure_mass)
{
  offsets.Copy(offsets_);
  pressure_mass_.GetDiag(mass_diag_);

  // Inner displacement-block solve on the augmented block.
  LinearSolverConfig inner = cfg;
  inner.type = "gmres_amg";
  inner.amg = "systems";
  inner.rtol = cfg.inner_rtol;
  inner.atol = 0.0;
  inner.max_it = cfg.inner_max_it;
  inner.print_level = 0;
  stiffness_ = std::make_unique<LinearSolver>(inner, fes_u);

  // Schur complement approximation.
  const double inv_kappa = std::isfinite(kappa_) ? 1.0 / kappa_ : 0.0;
  const double schur_scale = (1.0 + mu_ * inv_kappa) / (mu_ + gamma_);
  scaled_mass_ = std::make_unique<mfem::HypreParMatrix>(pressure_mass_);
  *scaled_mass_ *= schur_scale;
  mass_prec_ = std::make_unique<mfem::HypreSmoother>(*scaled_mass_,
                                                     mfem::HypreSmoother::Jacobi);
  mass_solver_ = std::make_unique<mfem::CGSolver>(comm_);
  mass_solver_->SetRelTol(1e-8);
  mass_solver_->SetAbsTol(0.0);
  mass_solver_->SetMaxIter(100);
  mass_solver_->SetPrintLevel(0);
  mass_solver_->SetPreconditioner(*mass_prec_);
  mass_solver_->SetOperator(*scaled_mass_);
  mass_solver_->iterative_mode = false;

  prec_ = std::make_unique<BlockPreconditioner>(*this);
  outer_ = std::make_unique<mfem::FGMRESSolver>(comm_);
  outer_->SetKDim(cfg.krylov_dim);
  outer_->SetRelTol(cfg.rtol);
  outer_->SetAbsTol(cfg.atol);
  outer_->SetMaxIter(cfg.max_it);
  outer_->SetPrintLevel(cfg.print_level > 1 ? 1 : 0);
  outer_->SetPreconditioner(*prec_);
  outer_->iterative_mode = false;
}

void SaddlePointSolver::SetOperator(const mfem::Operator &op)
{
  const auto *J = dynamic_cast<const mfem::BlockOperator *>(&op);
  if (!J || J->NumRowBlocks() != 2 || J->NumColBlocks() != 2)
  {
    throw std::runtime_error("SaddlePointSolver::SetOperator expects a 2x2 BlockOperator");
  }
  auto &Jm = *const_cast<mfem::BlockOperator *>(J);
  const auto *K = dynamic_cast<const mfem::HypreParMatrix *>(&Jm.GetBlock(0, 0));
  const auto *B = dynamic_cast<const mfem::HypreParMatrix *>(&Jm.GetBlock(0, 1));
  const auto *Bt = dynamic_cast<const mfem::HypreParMatrix *>(&Jm.GetBlock(1, 0));
  const auto *Cneg = dynamic_cast<const mfem::HypreParMatrix *>(&Jm.GetBlock(1, 1));
  if (!K || !B || !Bt || !Cneg)
  {
    throw std::runtime_error("SaddlePointSolver: Jacobian blocks must be HypreParMatrix");
  }
  jacobian_ = J;
  height = width = J->Height();

  if (gamma_ > 0.0)
  {
    // W B^T with W = diag(M_p)^{-1}.
    BtW_ = std::make_unique<mfem::HypreParMatrix>(*Bt);
    BtW_->InvScaleRows(mass_diag_);
    std::unique_ptr<mfem::HypreParMatrix> BWBt(mfem::ParMult(B, BtW_.get()));
    K_aug_.reset(mfem::Add(1.0, *K, gamma_, *BWBt));
    if (std::isfinite(kappa_))
    {
      // B~ = B - gamma B W C = B + gamma (W B^T)^T (-C)
      std::unique_ptr<mfem::HypreParMatrix> BW(BtW_->Transpose());
      std::unique_ptr<mfem::HypreParMatrix> BWCneg(mfem::ParMult(BW.get(), Cneg));
      B_aug_.reset(mfem::Add(1.0, *B, gamma_, *BWCneg));
    }
    else
    {
      B_aug_.reset();
    }
  }
  else
  {
    BtW_.reset();
    K_aug_ = std::make_unique<mfem::HypreParMatrix>(*K);
    B_aug_.reset();
  }

  A_aug_ = std::make_unique<mfem::BlockOperator>(offsets_);
  A_aug_->SetBlock(0, 0, K_aug_.get());
  A_aug_->SetBlock(0, 1, B_aug_ ? B_aug_.get() : const_cast<mfem::HypreParMatrix *>(B));
  A_aug_->SetBlock(1, 0, const_cast<mfem::HypreParMatrix *>(Bt));
  A_aug_->SetBlock(1, 1, const_cast<mfem::HypreParMatrix *>(Cneg));

  stiffness_->SetOperator(*K_aug_);
  outer_->SetOperator(*A_aug_);
}

void SaddlePointSolver::BlockPreconditioner::Mult(const mfem::Vector &r,
                                                  mfem::Vector &z) const
{
  const mfem::Array<int> &off = owner_.offsets_;
  const int n_u = off[1] - off[0], n_p = off[2] - off[1];
  mfem::Vector r_u(const_cast<mfem::Vector &>(r).GetData(), n_u);
  mfem::Vector r_p(const_cast<mfem::Vector &>(r).GetData() + off[1], n_p);
  mfem::Vector z_u(z.GetData(), n_u);
  mfem::Vector z_p(z.GetData() + off[1], n_p);

  // p = -S~^{-1} r_p
  z_p = 0.0;
  owner_.mass_solver_->Mult(r_p, z_p);
  z_p *= -1.0;

  // u = K~^{-1} (r_u - B~ p)
  mfem::Vector rhs(n_u);
  owner_.A_aug_->GetBlock(0, 1).Mult(z_p, rhs);
  rhs *= -1.0;
  rhs += r_u;
  z_u = 0.0;
  owner_.stiffness_->Mult(rhs, z_u);
  owner_.inner_iterations_ += owner_.stiffness_->Iterations();
  if (!owner_.stiffness_->Converged()) { owner_.inner_failures_++; }
}

void SaddlePointSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const
{
  if (!jacobian_) { throw std::runtime_error("SaddlePointSolver::Mult before SetOperator"); }
  const int n_u = offsets_[1], n_p = offsets_[2] - offsets_[1];
  mfem::Vector b_aug(b);
  if (BtW_)
  {
    // b~_u = b_u + gamma B W b_p
    mfem::Vector b_p(const_cast<mfem::Vector &>(b).GetData() + n_u, n_p);
    mfem::Vector b_u(b_aug.GetData(), n_u);
    mfem::Vector t(n_u);
    BtW_->MultTranspose(b_p, t);
    b_u.Add(gamma_, t);
  }
  inner_iterations_ = 0;
  inner_failures_ = 0;
  outer_->Mult(b_aug, x);
  if (cfg_.print_level > 0)
  {
    int rank = 0;
    MPI_Comm_rank(comm_, &rank);
    if (rank == 0)
    {
      std::printf("saddle-point solve: %d fgmres its (%s), inner amg-gmres its %d (%d unconverged), "
                  "gamma %.3g\n", outer_->GetNumIterations(),
                  outer_->GetConverged() ? "converged" : "NOT converged", inner_iterations_,
                  inner_failures_, gamma_);
    }
  }
}

} // namespace cmf

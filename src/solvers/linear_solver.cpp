#include "solvers/linear_solver.hpp"

#include <cstdio>
#include <stdexcept>

namespace cmf
{

LinearSolver::LinearSolver(const LinearSolverConfig &cfg,
                           mfem::ParFiniteElementSpace &fes)
  : mfem::Solver(fes.GetTrueVSize()), cfg_(cfg), fes_(fes)
{
  MPI_Comm comm = fes_.GetComm();
  if (cfg_.type == "gmres_amg")
  {
    auto gmres = std::make_unique<mfem::GMRESSolver>(comm);
    gmres->SetKDim(cfg_.krylov_dim);
    krylov_ = std::move(gmres);
  }
  else if (cfg_.type == "cg_amg")
  {
    krylov_ = std::make_unique<mfem::CGSolver>(comm);
  }
  else
  {
    throw ConfigError("solver.linear.type: unknown type '" + cfg_.type + "'");
  }
  krylov_->SetRelTol(cfg_.rtol);
  krylov_->SetAbsTol(cfg_.atol);
  krylov_->SetMaxIter(cfg_.max_it);
  krylov_->SetPrintLevel(cfg_.print_level);
  krylov_->iterative_mode = false;
  amg_mode_ = cfg_.amg;
}

void LinearSolver::BuildPreconditioner(const std::string &mode) const
{
  amg_ = std::make_unique<mfem::HypreBoomerAMG>(*A_);
  const int dim = fes_.GetParMesh()->Dimension();
  if (mode == "elasticity")
  {
    amg_->SetElasticityOptions(&fes_);
  }
  else
  {
    amg_->SetSystemsOptions(dim, fes_.GetOrdering() == mfem::Ordering::byNODES);
  }
  amg_->SetPrintLevel(0);
  amg_mode_ = mode;
  krylov_->SetPreconditioner(*amg_);
}

void LinearSolver::SetOperator(const mfem::Operator &op)
{
  const auto *A = dynamic_cast<const mfem::HypreParMatrix *>(&op);
  if (!A)
  {
    throw std::runtime_error("LinearSolver::SetOperator expects a HypreParMatrix");
  }
  height = A->Height();
  width = A->Width();
  A_ = A;
  BuildPreconditioner(amg_mode_);
  krylov_->SetOperator(*A);
}

void LinearSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const
{
  if (!amg_) { throw std::runtime_error("LinearSolver::Mult before SetOperator"); }
  krylov_->Mult(b, x);
  if (!krylov_->GetConverged() && amg_mode_ == "elasticity")
  {
    int rank = 0;
    MPI_Comm_rank(fes_.GetComm(), &rank);
    if (rank == 0)
    {
      std::printf("linear solver: elasticity AMG did not converge in %d iterations; "
                  "falling back to systems AMG options\n", krylov_->GetNumIterations());
    }
    BuildPreconditioner("systems");
    krylov_->SetOperator(*A_);
    x = 0.0;
    krylov_->Mult(b, x);
  }
}

std::unique_ptr<LinearSolver> MakeLinearSolver(const LinearSolverConfig &cfg,
                                               mfem::ParFiniteElementSpace &fes)
{
  return std::make_unique<LinearSolver>(cfg, fes);
}

} // namespace cmf

#include "solvers/linear_solver.hpp"

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
  amg_ = std::make_unique<mfem::HypreBoomerAMG>(*A);
  const int dim = fes_.GetParMesh()->Dimension();
  if (cfg_.amg == "elasticity")
  {
    amg_->SetElasticityOptions(&fes_);
  }
  else
  {
    amg_->SetSystemsOptions(dim, fes_.GetOrdering() == mfem::Ordering::byNODES);
  }
  amg_->SetPrintLevel(0);
  krylov_->SetPreconditioner(*amg_);
  krylov_->SetOperator(*A);
}

void LinearSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const
{
  if (!amg_) { throw std::runtime_error("LinearSolver::Mult before SetOperator"); }
  krylov_->Mult(b, x);
}

std::unique_ptr<LinearSolver> MakeLinearSolver(const LinearSolverConfig &cfg,
                                               mfem::ParFiniteElementSpace &fes)
{
  return std::make_unique<LinearSolver>(cfg, fes);
}

} // namespace cmf

#include "solvers/direct_solver.hpp"
#include <cstdio>

#include <stdexcept>

#ifdef MFEM_USE_PETSC
#include <dlfcn.h>
#include <petscsys.h>
#endif

namespace cmf
{

#ifndef MFEM_USE_PETSC

PetscSession::~PetscSession() = default;
DirectSolver::DirectSolver(MPI_Comm comm, const LinearSolverConfig &cfg) : comm_(comm), cfg_(cfg)
{
  throw ConfigError("solver.linear.type: direct needs an MFEM built with MFEM_USE_PETSC "
                    "(and a PETSc with MUMPS)");
}
DirectSolver::~DirectSolver() = default;
void DirectSolver::SetOperator(const mfem::Operator &) {}
void DirectSolver::Mult(const mfem::Vector &, mfem::Vector &) const {}
bool DirectSolver::Converged() const { return false; }

#else

namespace
{

const char *kPrefix = "cmf_direct_";

void EnsurePetsc()
{
  PetscBool initialized = PETSC_FALSE;
  PetscInitialized(&initialized);
  if (initialized) { return; }
  mfem::MFEMInitializePetsc(nullptr, nullptr, nullptr, nullptr);
  const std::string p = std::string("-") + kPrefix;
  PetscOptionsSetValue(nullptr, (p + "ksp_type").c_str(), "preonly");
  PetscOptionsSetValue(nullptr, (p + "pc_type").c_str(), "lu");
  PetscOptionsSetValue(nullptr, (p + "pc_factor_mat_solver_type").c_str(), "mumps");
  // More room for the fill-in of the pivoting on the saddle-point matrix.
  PetscOptionsSetValue(nullptr, (p + "mat_mumps_icntl_14").c_str(), "80");
  // One thread per MPI rank in the factorization: the parallelism is MPI's, and on
  // the small matrices this solver is meant for a threaded BLAS only spins (measured:
  // 15 times the CPU time and a longer wall time with OpenBLAS's default).
  PetscOptionsSetValue(nullptr, (p + "mat_mumps_icntl_16").c_str(), "1");
  using SetThreads = void (*)(int);
  for (const char *name : {"openblas_set_num_threads", "omp_set_num_threads"})
  {
    if (auto set = reinterpret_cast<SetThreads>(dlsym(RTLD_DEFAULT, name))) { set(1); }
  }
}

} // namespace

PetscSession::~PetscSession()
{
  PetscBool initialized = PETSC_FALSE, finalized = PETSC_FALSE;
  PetscInitialized(&initialized);
  PetscFinalized(&finalized);
  if (initialized && !finalized) { mfem::MFEMFinalizePetsc(); }
}

DirectSolver::DirectSolver(MPI_Comm comm, const LinearSolverConfig &cfg)
  : comm_(comm), cfg_(cfg)
{
  EnsurePetsc();
  // wrap = false: the HypreParMatrix is converted to PETSc's AIJ format, which the
  // factorization needs.
  petsc_ = std::make_unique<mfem::PetscLinearSolver>(comm_, kPrefix, false);
  petsc_->iterative_mode = false;
}

DirectSolver::~DirectSolver() = default;

void DirectSolver::SetOperator(const mfem::Operator &op)
{
  // The same, unchanged operator (a linear problem): keep the factorization.
  if (stamp_ && op_ == &op && *stamp_ == seen_stamp_) { return; }
  if (stamp_) { seen_stamp_ = *stamp_; }
  op_ = &op;
  height = width = op.Height();

  const mfem::HypreParMatrix *A = dynamic_cast<const mfem::HypreParMatrix *>(&op);
  if (!A)
  {
    const auto *J = dynamic_cast<const mfem::BlockOperator *>(&op);
    if (!J) { throw std::runtime_error("DirectSolver::SetOperator expects a HypreParMatrix or a BlockOperator"); }
    auto &Jm = *const_cast<mfem::BlockOperator *>(J);
    mfem::Array2D<const mfem::HypreParMatrix *> blocks(J->NumRowBlocks(), J->NumColBlocks());
    for (int i = 0; i < J->NumRowBlocks(); i++)
      for (int j = 0; j < J->NumColBlocks(); j++)
      {
        blocks(i, j) = Jm.IsZeroBlock(i, j) ? nullptr
                       : dynamic_cast<const mfem::HypreParMatrix *>(&Jm.GetBlock(i, j));
        if (!Jm.IsZeroBlock(i, j) && !blocks(i, j))
        {
          throw std::runtime_error("DirectSolver: Jacobian blocks must be HypreParMatrix");
        }
      }
    merged_.reset(mfem::HypreParMatrixFromBlocks(blocks));
    A = merged_.get();
  }
  petsc_->SetOperator(*A);
  setups_++;
}

void DirectSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const
{
  if (!op_) { throw std::runtime_error("DirectSolver::Mult before SetOperator"); }
  petsc_->Mult(b, x);
  if (cfg_.print_level > 0)
  {
    int rank = 0;
    MPI_Comm_rank(comm_, &rank);
    if (rank == 0) { std::printf("direct solve (mumps): %d unknowns, factorization %d\n", height, setups_); }
  }
}

bool DirectSolver::Converged() const { return petsc_->GetConverged() != 0; }

#endif // MFEM_USE_PETSC

} // namespace cmf

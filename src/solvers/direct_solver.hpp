// Sparse direct linear solver (solver.linear.type: direct): LU factorization
// by MUMPS through PETSc (mfem::PetscLinearSolver, -ksp_type preonly
// -pc_type lu), for problems small enough to factor, where it is much
// cheaper than the Krylov-AMG solvers. The operator is a HypreParMatrix
// (displacement formulation) or the 2x2 BlockOperator of HypreParMatrix
// blocks of the mixed formulation, which is merged into one matrix; the zero
// (p, p) block of an incompressible material is handled by the pivoting of
// the factorization. The matrix is refactored on every SetOperator, unless it
// is the operator of the last call with an unchanged stamp (a linear problem,
// see solvers/linear_solver.hpp).
//
// Without MFEM_USE_PETSC the class still compiles and its constructor throws
// a ConfigError. PETSc is initialized on first use; an executable that may use the solver
// holds a PetscSession in main, after mfem::Mpi::Init, so that PETSc is
// finalized before MPI.
#pragma once

#include <memory>

#include "base/config.hpp"
#include "mfem.hpp"
#include "solvers/linear_solver.hpp"

namespace cmf
{

// Finalizes PETSc at the end of its scope if it has been initialized.
struct PetscSession
{
  PetscSession() = default;
  PetscSession(const PetscSession &) = delete;
  PetscSession &operator=(const PetscSession &) = delete;
  ~PetscSession();
};

class DirectSolver : public mfem::Solver
{
public:
  DirectSolver(MPI_Comm comm, const LinearSolverConfig &cfg);
  ~DirectSolver() override;

  void SetOperator(const mfem::Operator &op) override;
  void SetOperatorStamp(OperatorStamp stamp) { stamp_ = std::move(stamp); }
  void Mult(const mfem::Vector &b, mfem::Vector &x) const override;

  bool Converged() const;
  // Number of factorizations so far.
  int Setups() const { return setups_; }

private:
  MPI_Comm comm_;
  LinearSolverConfig cfg_;
  const mfem::Operator *op_ = nullptr;
  std::unique_ptr<mfem::HypreParMatrix> merged_;   // mixed formulation only
#ifdef MFEM_USE_PETSC
  std::unique_ptr<mfem::PetscLinearSolver> petsc_;
#endif
  OperatorStamp stamp_;
  long seen_stamp_ = -1;
  int setups_ = 0;
};

} // namespace cmf

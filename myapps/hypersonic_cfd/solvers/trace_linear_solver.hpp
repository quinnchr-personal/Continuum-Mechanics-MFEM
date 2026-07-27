#pragma once

#include "mfem.hpp"

#include <stdexcept>
#include <string>

namespace hycfd
{

// Direct solve of the condensed true-dof trace system with PETSc (MUMPS
// via the options file input/petsc.opts; swap the options file for GMRES +
// ASM/ILU or other PETSc-supported solvers). Works serial and parallel —
// the matrix is the PtAP-condensed HypreParMatrix.
inline void SolveCondensedPetscDirect(
   const mfem::HypreParMatrix &matrix,
   const mfem::Vector &right_hand_side,
   mfem::Vector &solution)
{
   mfem::PetscParMatrix petsc_matrix(
      &matrix, mfem::Operator::PETSC_MATAIJ);
   mfem::PetscLinearSolver solver(petsc_matrix, "", false);
   solver.SetPrintLevel(0);
   solution.SetSize(right_hand_side.Size());
   solution = 0.0;
   solver.Mult(right_hand_side, solution);
   if (!solver.GetConverged())
   {
      throw std::runtime_error(
         "PETSc trace solve did not converge: iterations=" +
         std::to_string(solver.GetNumIterations()) +
         " residual=" + std::to_string(solver.GetFinalNorm()));
   }
}

} // namespace hycfd

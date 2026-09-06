// Common interface of the solid mechanics formulations, so the app and the
// tests can drive either the displacement or the mixed u-p problem.
#pragma once

#include <memory>
#include <string>

#include "base/config.hpp"
#include "base/fields.hpp"
#include "mfem.hpp"
#include "solvers/quasi_static.hpp"

namespace cmf
{

class SolidProblem : public QuasiStaticProblem
{
public:
  using QuasiStaticProblem::QuasiStaticProblem;
  ~SolidProblem() override = default;

  // Programmatic boundary conditions (coefficients are not owned).
  virtual void AddDirichlet(const std::vector<int> &attrs, mfem::VectorCoefficient &u_bar) = 0;
  virtual void AddTraction(const std::vector<int> &attrs, mfem::VectorCoefficient &T_bar) = 0;
  virtual void SetBodyForce(mfem::VectorCoefficient &b) = 0;
  virtual void ClearBoundaryConditions() = 0;
  virtual void Finalize() = 0;

  virtual mfem::ParFiniteElementSpace &DisplacementSpace() = 0;
  virtual const mfem::Array<int> &EssentialTrueDofs() const = 0;
  virtual HYPRE_BigInt GlobalTrueVSize() const = 0; // collective
  virtual std::string Description() const = 0;

  virtual double InternalEnergy(const mfem::Vector &x) const = 0;
  virtual void UpdateFields(const mfem::Vector &x) = 0;
  virtual void RegisterFields(FieldRegistry &registry) = 0;
  virtual mfem::ParGridFunction &Displacement() = 0;

  // The linear solver matching this problem's Jacobian structure.
  virtual std::unique_ptr<mfem::Solver> MakeLinearSolver(const LinearSolverConfig &cfg) = 0;
};

// Builds the problem selected by cfg.formulation with the YAML boundary
// conditions installed (Finalize() still has to be called).
std::unique_ptr<SolidProblem> MakeSolidProblem(mfem::ParMesh &mesh, const AppConfig &cfg);

} // namespace cmf

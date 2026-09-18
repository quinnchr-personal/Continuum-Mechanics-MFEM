// Common interface of the solid mechanics formulations, so the app and the
// tests can drive either the displacement or the mixed u-p problem.
#pragma once

#include <memory>
#include <string>

#include "base/config.hpp"
#include "base/fields.hpp"
#include "materials/materials.hpp"
#include "mfem.hpp"
#include "physics/loads.hpp"
#include "solvers/linear_solver.hpp"
#include "solvers/quasi_static.hpp"

namespace cmf
{

class SolidProblem : public QuasiStaticProblem
{
public:
  using QuasiStaticProblem::QuasiStaticProblem;
  ~SolidProblem() override = default;

  // Programmatic boundary conditions (coefficients are not owned; see
  // physics/loads.hpp for the options: components, schedule, time dependence).
  virtual void AddDirichlet(const std::vector<int> &attrs, mfem::VectorCoefficient &u_bar,
                            const BCOptions &opt = BCOptions()) = 0;
  virtual void AddTraction(const std::vector<int> &attrs, mfem::VectorCoefficient &T_bar,
                           const BCOptions &opt = BCOptions()) = 0;
  // Normal pressure p: dead (T = -p N per reference area) or follower
  // (T = -p J F^{-T} N, per current area).
  virtual void AddPressure(const std::vector<int> &attrs, mfem::Coefficient &p, bool follower,
                           const BCOptions &opt = BCOptions()) = 0;
  virtual void SetBodyForce(mfem::VectorCoefficient &b, const BCOptions &opt = BCOptions()) = 0;
  virtual void ClearBoundaryConditions() = 0;
  virtual void Finalize() = 0;
  virtual const LoadSet &Loads() const = 0;
  // The static residual at x with its essential rows kept (Mult zeroes them):
  // internal minus external force on every dof of the unknown.
  virtual void FullResidual(const mfem::Vector &x, mfem::Vector &r) const = 0;
  // Reactions of the Dirichlet entries (see loads.hpp) from a full residual r
  // of the unknown at the state x; a dynamic analysis adds the inertial force
  // to r (physics/dynamic_solid_problem.hpp).
  virtual std::vector<Reaction> ReactionsFrom(const mfem::Vector &r,
                                              const mfem::Vector &x) const = 0;
  // Reactions of the static balance at x.
  std::vector<Reaction> Reactions(const mfem::Vector &x) const
  {
    mfem::Vector r;
    FullResidual(x, r);
    return ReactionsFrom(r, x);
  }
  // Dynamic analysis: the loads follow the physical time (LoadSet::SetPhysicalTime).
  virtual void SetPhysicalTime(bool on) = 0;
  // Reference density rho_R by element attribute (material.rho0 and the
  // regions'): the body force and the mass matrix share it.
  virtual mfem::Coefficient &ReferenceDensity() = 0;
  // The stamp of the operator GetGradient returns (solvers/linear_solver.hpp).
  virtual OperatorStamp GradientStamp() const = 0;

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

// Materials by element attribute from cfg.material and its regions (see
// materials.hpp for the table convention): the base everywhere, each region
// on its attributes (numbers checked against the mesh, physical-volume names
// resolved through the element attribute sets).
std::vector<Material> MakeMaterialTable(const MaterialConfig &cfg, mfem::Mesh &mesh,
                                        bool plane_stress);
std::vector<MixedMaterial> MakeMixedMaterialTable(const MaterialConfig &cfg, mfem::Mesh &mesh);
// Reference density by element attribute (entry a - 1 for attribute a, the
// layout of mfem::PWConstCoefficient): cfg.rho0, and a region's rho0 on its
// attributes.
mfem::Vector MakeDensityTable(const MaterialConfig &cfg, mfem::Mesh &mesh);

// Installs cfg.bcs and cfg.body_force into problem: attributes resolved
// against mesh, coefficients built by base/coefficients.hpp and kept alive in
// the given containers. Used by both formulations' constructors.
void InstallYamlLoads(SolidProblem &problem, mfem::Mesh &mesh, const AppConfig &cfg, int dim,
                      std::vector<std::unique_ptr<mfem::VectorCoefficient>> &owned_vectors,
                      std::vector<std::unique_ptr<mfem::Coefficient>> &owned_scalars);

// Options of a YAML entry: components, schedule, time dependence.
BCOptions OptionsOf(const BoundaryCondition &bc);

} // namespace cmf

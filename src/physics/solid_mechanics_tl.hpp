// Quasi-static total Lagrangian solid mechanics on the reference mesh.
//   R(u).w = int P(F) : Grad w dV - lambda [ int rho0 b.w dV + int T.w dA ]
// Owns the vector H1 space, the essential dofs, the ParNonlinearForm, and the
// load factor. Mult = residual with essential rows zeroed; GetGradient = the
// assembled HypreParMatrix with eliminated essential rows/columns.
#pragma once

#include <memory>
#include <vector>

#include "base/config.hpp"
#include "base/fields.hpp"
#include "materials/materials.hpp"
#include "mfem.hpp"
#include "physics/solid_problem.hpp"
#include "physics/quadrature_fields.hpp"
#include "solvers/quasi_static.hpp"

namespace cmf
{

class SolidMechanicsTL : public SolidProblem
{
public:
  // Parses the full input schema from root (mesh.order, material.rho0, bcs,
  // body_force) and installs the YAML boundary conditions.
  SolidMechanicsTL(mfem::ParMesh &mesh, const YAML::Node &root,
                   const Material &material);
  SolidMechanicsTL(mfem::ParMesh &mesh, const AppConfig &cfg,
                   const Material &material);
  ~SolidMechanicsTL() override = default;

  // Programmatic boundary conditions and loads. Coefficients are not owned
  // and must outlive this object. Each call invalidates Finalize().
  void AddDirichlet(const std::vector<int> &attrs, mfem::VectorCoefficient &u_bar) override;
  void AddTraction(const std::vector<int> &attrs, mfem::VectorCoefficient &T_bar) override;
  void SetBodyForce(mfem::VectorCoefficient &b) override; // per unit mass
  void ClearBoundaryConditions() override;
  // Builds the essential dof list and assembles the external load vector;
  // called lazily by SetLoadFactor/ApplyDirichlet, and required before Mult.
  void Finalize() override;

  // mfem::Operator on true dofs.
  void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
  mfem::Operator &GetGradient(const mfem::Vector &x) const override;

  // QuasiStaticProblem. The load factor scales the tractions, the body
  // force, and the prescribed (Dirichlet) displacements together, so that
  // lambda = 1 is the problem of the weak form and lambda < 1 a proportional
  // path to it.
  void SetLoadFactor(double lambda) override;
  double LoadFactor() const override { return load_factor_; }
  void ApplyDirichlet(mfem::Vector &x) const override;
  MPI_Comm Comm() const override { return fes_.GetComm(); }

  mfem::ParMesh &Mesh() { return mesh_; }
  mfem::ParFiniteElementSpace &FESpace() { return fes_; }
  mfem::ParFiniteElementSpace &DisplacementSpace() override { return fes_; }
  const mfem::Array<int> &EssentialTrueDofs() const override { return ess_tdof_list_; }
  HYPRE_BigInt GlobalTrueVSize() const override;
  std::string Description() const override;
  const mfem::Vector &ExternalLoad() const { return load_true_; }
  const Material &GetMaterial() const { return material_; }
  double Rho0() const { return rho0_; }
  int Order() const { return order_; }

  // Stored energy int W(F) dV at x.
  double InternalEnergy(const mfem::Vector &x) const override;

  // Post-processing: refresh the displacement and the quadrature-point
  // quantities of output.fields with the presentations of
  // output.quadrature_at (see quadrature_fields.hpp), from x.
  void UpdateFields(const mfem::Vector &x) override;
  void RegisterFields(FieldRegistry &registry) override;
  mfem::ParGridFunction &Displacement() override { return *displacement_; }

  // GMRES/CG + BoomerAMG on the assembled Jacobian.
  std::unique_ptr<mfem::Solver> MakeLinearSolver(const LinearSolverConfig &cfg) override;

private:
  struct BCEntry
  {
    mfem::Array<int> marker;
    mfem::VectorCoefficient *coef;
  };

  void Build(const AppConfig &cfg);
  mfem::Array<int> Marker(const std::vector<int> &attrs) const;
  void CheckVectorSize(const std::vector<double> &v, const std::string &what) const;
  void CheckCoefficient(mfem::VectorCoefficient &c, const std::string &what) const;
  void EnsureFields();

  mfem::ParMesh &mesh_;
  int dim_;
  int order_;
  double rho0_;
  Material material_;
  mfem::H1_FECollection fec_;
  mfem::ParFiniteElementSpace fes_;
  mfem::ParNonlinearForm nlf_;

  std::vector<std::unique_ptr<mfem::VectorCoefficient>> owned_coefs_;
  std::vector<BCEntry> dirichlet_;
  std::vector<BCEntry> traction_;
  mfem::VectorCoefficient *body_force_ = nullptr;
  std::unique_ptr<mfem::VectorCoefficient> rho0_body_force_;

  bool finalized_ = false;
  double load_factor_ = 1.0;
  mfem::Array<int> ess_tdof_list_;
  mfem::Vector load_true_;

  std::unique_ptr<mfem::ParGridFunction> displacement_;
  bool plane_stress_ = false;
  OutputConfig output_cfg_;
  std::unique_ptr<QuadratureFields> qfields_;
};

} // namespace cmf

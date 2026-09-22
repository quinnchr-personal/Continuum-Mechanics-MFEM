// Quasi-static total Lagrangian solid mechanics on the reference mesh.
//   R(u).w = int P(F) : Grad w dV + [follower pressures](u, t)
//            - sum_i s_i(t) [ int rho0 b_i.w dV + int T_i.w dA ]
// Owns the vector H1 space, the ParNonlinearForm and the loads (LoadSet:
// essential dofs, schedules s_i of the pseudo-time t). Mult = residual with
// essential rows zeroed; GetGradient = the assembled HypreParMatrix with
// eliminated essential rows/columns.
#pragma once

#include <deque>
#include <memory>
#include <vector>

#include "base/config.hpp"
#include "base/fields.hpp"
#include "kernels/history_field.hpp"
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
  // Materials by element attribute (materials.hpp: size 1, or index =
  // attribute with entry 0 unused; all the same model).
  SolidMechanicsTL(mfem::ParMesh &mesh, const AppConfig &cfg,
                   const std::vector<Material> &materials);
  ~SolidMechanicsTL() override = default;

  // Programmatic boundary conditions and loads. Coefficients are not owned
  // and must outlive this object. Each call invalidates Finalize().
  void AddDirichlet(const std::vector<int> &attrs, mfem::VectorCoefficient &u_bar,
                    const BCOptions &opt = BCOptions()) override;
  void AddTraction(const std::vector<int> &attrs, mfem::VectorCoefficient &T_bar,
                   const BCOptions &opt = BCOptions()) override;
  void AddPressure(const std::vector<int> &attrs, mfem::Coefficient &p, bool follower,
                   const BCOptions &opt = BCOptions()) override;
  void AddRigidSphereContact(const std::vector<int> &attrs, mfem::VectorCoefficient &center,
                             double radius, double penalty,
                             const BCOptions &opt = BCOptions()) override;
  void SetBodyForce(mfem::VectorCoefficient &b,
                    const BCOptions &opt = BCOptions()) override; // per unit mass
  void ClearBoundaryConditions() override;
  // Builds the essential dof list and assembles the external load vectors;
  // called lazily by SetLoadFactor/ApplyDirichlet, and required before Mult.
  void Finalize() override;
  const LoadSet &Loads() const override { return loads_; }
  void FullResidual(const mfem::Vector &x, mfem::Vector &r) const override;
  std::vector<Reaction> ReactionsFrom(const mfem::Vector &r, const mfem::Vector &x) const override;
  void SetPhysicalTime(bool on) override { loads_.SetPhysicalTime(on); }
  // rho_R, times 2 pi r in an axisymmetric problem (the mass of the solid of revolution).
  mfem::Coefficient &ReferenceDensity() override
  {
    if (axisymmetric_) { return *density_axi_; }
    return density_;
  }
  bool Axisymmetric() const { return axisymmetric_; }
  OperatorStamp GradientStamp() const override { return gradient_stamp_; }
  // History of a material with Maxwell branches (SolidProblem): the
  // quadrature-point field, the time of the last accepted step.
  bool HasHistory() const override { return history_ != nullptr; }
  void ResetHistory(double t) override;
  void AcceptStep(const mfem::Vector &x) override;
  const HistoryField *History() const { return history_.get(); }
  double AcceptedTime() const { return t_accepted_; }

  // mfem::Operator on true dofs.
  void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
  mfem::Operator &GetGradient(const mfem::Vector &x) const override;

  // QuasiStaticProblem. The argument is the pseudo-time t in [0, 1] (or the
  // physical time); every load and prescribed displacement follows its own
  // schedule s_i(t) (the default ramp s = t makes t < 1 a proportional path
  // to the t = 1 problem), and a history-dependent material takes
  // t - AcceptedTime() as the length of the step under way.
  void SetLoadFactor(double t) override;
  double LoadFactor() const override { return loads_.Time(); }
  void ApplyDirichlet(mfem::Vector &x) const override;
  MPI_Comm Comm() const override { return fes_.GetComm(); }
  // A small-strain material with dead loads (follower loads are rejected for it).
  bool IsLinear() const override { return IsSmallStrain(materials_[0]) && !loads_.HasFollowerPressure(); }
  // The Jacobian of a linear problem depends neither on the state nor on the
  // pseudo-time (the essential dofs are fixed by Finalize), so GetGradient
  // assembles it once and returns that matrix until the boundary conditions
  // change; the solver of MakeLinearSolver then keeps its AMG hierarchy
  // (solvers/linear_solver.hpp, operator stamp). On by default; off
  // reassembles on every call, like a nonlinear problem.
  void ReuseConstantGradient(bool on) { reuse_gradient_ = on; gradient_ = nullptr; }
  // Number of Jacobian assemblies so far.
  int GradientAssemblies() const { return int(*gradient_stamp_); }

  mfem::ParMesh &Mesh() { return mesh_; }
  mfem::ParFiniteElementSpace &FESpace() { return fes_; }
  mfem::ParFiniteElementSpace &DisplacementSpace() override { return fes_; }
  const mfem::Array<int> &EssentialTrueDofs() const override { return loads_.EssentialTrueDofs(); }
  HYPRE_BigInt GlobalTrueVSize() const override;
  std::string Description() const override;
  // Scheduled dead load at the current pseudo-time.
  const mfem::Vector &ExternalLoad() const { return loads_.ExternalLoad(); }
  const Material &GetMaterial() const { return materials_[0]; }
  const std::vector<Material> &Materials() const { return materials_; }
  double Rho0() const { return rho0_; } // of the base material (regions: ReferenceDensity)
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
  void Build(const AppConfig &cfg);
  // (Re)creates the nonlinear form with the material integrator; follower
  // pressures and contacts add boundary integrators that can only be
  // removed this way.
  void ResetForm();
  void EnsureFields();
  // The quadrature-point history of a material with one (Cv = I everywhere).
  void InitializeHistory();
  // Advances the history to the end of the step under way at the state x.
  void UpdateHistory(const mfem::Vector &x);

  mfem::ParMesh &mesh_;
  int dim_;
  int order_;
  double rho0_;
  mfem::Vector density_table_;
  mfem::PWConstCoefficient density_;
  std::vector<Material> materials_;
  mfem::H1_FECollection fec_;
  mfem::ParFiniteElementSpace fes_;
  std::unique_ptr<mfem::ParNonlinearForm> nlf_;
  // Domain integrator only, for InternalEnergy (MFEM's GetEnergy has no
  // boundary face terms; the follower pressure is not conservative anyway).
  std::unique_ptr<mfem::ParNonlinearForm> energy_form_;

  std::vector<std::unique_ptr<mfem::VectorCoefficient>> owned_coefs_;
  std::vector<std::unique_ptr<mfem::Coefficient>> owned_scalars_;
  LoadSet loads_;
  std::deque<mfem::Array<int>> follower_markers_; // referenced by the form's integrators (stable)
  std::deque<mfem::Array<int>> contact_markers_;
  // One form per contact entry holding its integrator alone: Mult gives the
  // nodal contact forces, whose resultant the reactions report.
  std::vector<std::unique_ptr<mfem::ParNonlinearForm>> contact_forms_;
  bool finalized_ = false;

  std::unique_ptr<HistoryField> history_;
  double t_accepted_ = 0.0;

  // The assembled Jacobian of a linear problem (owned by nlf_) and the stamp
  // shared with the linear solver, incremented at every assembly.
  bool reuse_gradient_ = true;
  mutable mfem::Operator *gradient_ = nullptr;
  std::shared_ptr<long> gradient_stamp_ = std::make_shared<long>(0);

  std::unique_ptr<mfem::ParGridFunction> displacement_;
  bool plane_stress_ = false;
  bool axisymmetric_ = false;
  mfem::FunctionCoefficient r2pi_;
  std::unique_ptr<mfem::ProductCoefficient> density_axi_;
  OutputConfig output_cfg_;
  std::unique_ptr<QuadratureFields> qfields_;
  // F at a quadrature point from the displacement gradient, completed with
  // the hoop stretch 1 + u_r / r of an axisymmetric problem.
  tensor<double, 3, 3> GradientToF(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip,
                                   const mfem::DenseMatrix &grad) const;
};

} // namespace cmf

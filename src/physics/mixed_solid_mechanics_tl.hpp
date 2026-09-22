// Mixed displacement-pressure total Lagrangian solid mechanics (Taylor-Hood:
// displacement H1 order p, pressure H1 order p - 1) for near- and fully
// incompressible decoupled materials.
//   R_u(u, p).w = int [P_iso(F) + p J F^{-T}] : Grad w dV + [follower pressures]
//                 - sum_i s_i(t) [ext. loads_i]
//   R_p(u, p).q = int q (u'(J) - p/kappa) dV       (kappa = inf: q u'(J) <=> J = 1;
//                 U = kappa u(J) the volumetric law, u' = J - 1 by default)
// The unknown is the block true-dof vector [u; p]; Mult/GetGradient act on it.
// Loads and essential dofs live in a LoadSet on the displacement space.
#pragma once

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "base/config.hpp"
#include "base/fields.hpp"
#include "kernels/history_field.hpp"
#include "materials/materials.hpp"
#include "mfem.hpp"
#include "physics/solid_problem.hpp"
#include "physics/quadrature_fields.hpp"

namespace cmf
{

class MixedSolidMechanicsTL : public SolidProblem
{
public:
  MixedSolidMechanicsTL(mfem::ParMesh &mesh, const AppConfig &cfg,
                        const MixedMaterial &material);
  // Materials by element attribute (materials.hpp; all the same model and
  // all incompressible or none; the solver scales with the base entry).
  MixedSolidMechanicsTL(mfem::ParMesh &mesh, const AppConfig &cfg,
                        const std::vector<MixedMaterial> &materials);
  ~MixedSolidMechanicsTL() override = default;

  void AddDirichlet(const std::vector<int> &attrs, mfem::VectorCoefficient &u_bar,
                    const BCOptions &opt = BCOptions()) override;
  void AddTraction(const std::vector<int> &attrs, mfem::VectorCoefficient &T_bar,
                   const BCOptions &opt = BCOptions()) override;
  void AddPressure(const std::vector<int> &attrs, mfem::Coefficient &p, bool follower,
                   const BCOptions &opt = BCOptions()) override;
  void AddRigidSphereContact(const std::vector<int> &attrs, mfem::VectorCoefficient &center,
                             double radius, double penalty,
                             const BCOptions &opt = BCOptions()) override;
  void SetBodyForce(mfem::VectorCoefficient &b, const BCOptions &opt = BCOptions()) override;
  void ClearBoundaryConditions() override;
  void Finalize() override;
  const LoadSet &Loads() const override { return loads_; }
  void FullResidual(const mfem::Vector &x, mfem::Vector &r) const override;
  std::vector<Reaction> ReactionsFrom(const mfem::Vector &r, const mfem::Vector &x) const override;
  void SetPhysicalTime(bool on) override { loads_.SetPhysicalTime(on); }
  mfem::Coefficient &ReferenceDensity() override { return density_; }
  OperatorStamp GradientStamp() const override { return gradient_stamp_; }
  // History of a material with Maxwell branches (SolidProblem, as in SolidMechanicsTL).
  bool HasHistory() const override { return history_ != nullptr; }
  void ResetHistory(double t) override;
  void AcceptStep(const mfem::Vector &x) override;
  const HistoryField *History() const { return history_.get(); }
  double AcceptedTime() const { return t_accepted_; }

  void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
  mfem::Operator &GetGradient(const mfem::Vector &x) const override;

  void SetLoadFactor(double t) override;
  double LoadFactor() const override { return loads_.Time(); }
  void ApplyDirichlet(mfem::Vector &x) const override;
  // A small-strain material with dead loads: Herrmann's linear saddle-point problem.
  bool IsLinear() const override { return IsSmallStrain(materials_[0]) && !loads_.HasFollowerPressure(); }
  // As in SolidMechanicsTL: the block Jacobian of a linear problem is assembled
  // once, and the saddle-point solver keeps its augmented blocks and hierarchy.
  void ReuseConstantGradient(bool on) { reuse_gradient_ = on; gradient_ = nullptr; }
  int GradientAssemblies() const { return int(*gradient_stamp_); }
  MPI_Comm Comm() const override { return fes_u_.GetComm(); }

  mfem::ParMesh &Mesh() { return mesh_; }
  mfem::ParFiniteElementSpace &DisplacementSpace() override { return fes_u_; }
  mfem::ParFiniteElementSpace &PressureSpace() { return fes_p_; }
  const mfem::Array<int> &BlockOffsets() const { return offsets_; }
  const mfem::Array<int> &EssentialTrueDofs() const override { return loads_.EssentialTrueDofs(); }
  HYPRE_BigInt GlobalTrueVSize() const override;
  std::string Description() const override;
  const MixedMaterial &GetMaterial() const { return materials_[0]; }
  const std::vector<MixedMaterial> &Materials() const { return materials_; }
  double ShearModulus() const { return mu_; }
  double BulkModulus() const { return kappa_; } // inf when incompressible
  bool Incompressible() const { return incompressible_; }
  mfem::HypreParMatrix &PressureMass() { return *pressure_mass_; }

  double InternalEnergy(const mfem::Vector &x) const override;
  void UpdateFields(const mfem::Vector &x) override;
  void RegisterFields(FieldRegistry &registry) override;
  mfem::ParGridFunction &Displacement() override { return *displacement_; }
  mfem::ParGridFunction &Pressure() { return *pressure_; }

  // FGMRES with a block upper-triangular preconditioner (see saddle_point_solver).
  std::unique_ptr<mfem::Solver> MakeLinearSolver(const LinearSolverConfig &cfg) override;

private:
  // ParBlockNonlinearForm marks essential dofs by attribute (all
  // components); this exposes the true-dof lists for component-wise data.
  class BlockForm : public mfem::ParBlockNonlinearForm
  {
  public:
    using mfem::ParBlockNonlinearForm::ParBlockNonlinearForm;
    void SetEssentialTrueDofs(int block, const mfem::Array<int> &list)
    {
      list.Copy(*ess_tdofs[block]);
    }
  };

  void Build(const AppConfig &cfg);
  void ResetForm();
  void EnsureFields();
  void InitializeHistory();
  void UpdateHistory(const mfem::Vector &x);

  mfem::ParMesh &mesh_;
  int dim_;
  int order_;
  double rho0_;
  mfem::Vector density_table_;
  mfem::PWConstCoefficient density_;
  std::vector<MixedMaterial> materials_;
  double mu_;
  double kappa_;
  bool incompressible_;
  mfem::H1_FECollection fec_u_;
  mfem::ParFiniteElementSpace fes_u_;
  mfem::H1_FECollection fec_p_;
  mfem::ParFiniteElementSpace fes_p_;
  mfem::Array<mfem::ParFiniteElementSpace *> spaces_;
  mfem::Array<int> offsets_;
  std::unique_ptr<BlockForm> nlf_;
  std::unique_ptr<mfem::ParBlockNonlinearForm> energy_form_; // domain integrator only

  std::vector<std::unique_ptr<mfem::VectorCoefficient>> owned_coefs_;
  std::vector<std::unique_ptr<mfem::Coefficient>> owned_scalars_;
  LoadSet loads_;
  std::deque<mfem::Array<int>> follower_markers_;
  std::deque<mfem::Array<int>> contact_markers_;
  std::vector<std::unique_ptr<mfem::ParBlockNonlinearForm>> contact_forms_; // one per entry, its integrator alone
  mfem::Array<int> ess_p_empty_;

  bool finalized_ = false;
  std::unique_ptr<HistoryField> history_;
  double t_accepted_ = 0.0;
  std::unique_ptr<mfem::HypreParMatrix> pressure_mass_;

  // The assembled block Jacobian of a linear problem (owned by nlf_) and the
  // stamp shared with the saddle-point solver, incremented at every assembly.
  bool reuse_gradient_ = true;
  mutable mfem::Operator *gradient_ = nullptr;
  std::shared_ptr<long> gradient_stamp_ = std::make_shared<long>(0);

  std::unique_ptr<mfem::ParGridFunction> displacement_;
  std::unique_ptr<mfem::ParGridFunction> pressure_;
  OutputConfig output_cfg_;
  std::unique_ptr<QuadratureFields> qfields_;
};

} // namespace cmf

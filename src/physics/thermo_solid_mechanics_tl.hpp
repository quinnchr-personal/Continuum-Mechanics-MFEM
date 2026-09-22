// Coupled displacement-pressure-temperature total Lagrangian solid mechanics
// (the mixed u-p formulation of mixed_solid_mechanics_tl.hpp with the
// temperature as a third unknown; materials/thermoelastic.hpp,
// kernels/thermo_mixed_total_lagrangian.hpp): Taylor-Hood displacement of
// order p, pressure and temperature of order p - 1, quasi-static in physical
// time (the `time` block) with the heat equation integrated by the implicit
// Euler method over the step of the stepper. The unknown is the block true-dof
// vector [u; p; theta]; the accepted C and theta of every quadrature point are
// the history of the problem (a HistoryField of 7 doubles), advanced in
// AcceptStep. Mechanical loads and essential displacement dofs live in the
// LoadSet; the thermal entries (prescribed temperatures, inward heat fluxes)
// are kept here on the temperature space.
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
#include "physics/quadrature_fields.hpp"
#include "physics/solid_problem.hpp"

namespace cmf
{

class ThermoSolidMechanicsTL : public SolidProblem
{
public:
  ThermoSolidMechanicsTL(mfem::ParMesh &mesh, const AppConfig &cfg, const ThermoMaterial &material);
  // Materials by element attribute (all the same model, one reference
  // temperature: the base's).
  ThermoSolidMechanicsTL(mfem::ParMesh &mesh, const AppConfig &cfg,
                         const std::vector<ThermoMaterial> &materials);
  ~ThermoSolidMechanicsTL() override = default;

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
  // theta = schedule(t) * theta_bar on the faces of attrs (opt.components unused).
  void AddTemperature(const std::vector<int> &attrs, mfem::Coefficient &theta_bar,
                      const BCOptions &opt = BCOptions()) override;
  // Inward heat flux schedule(t) * h per unit current area (through |cof F N|)
  // or per unit reference area.
  void AddHeatFlux(const std::vector<int> &attrs, mfem::Coefficient &h, bool current_area,
                   const BCOptions &opt = BCOptions()) override;
  void ClearBoundaryConditions() override;
  void Finalize() override;
  const LoadSet &Loads() const override { return loads_; }
  void FullResidual(const mfem::Vector &x, mfem::Vector &r) const override;
  std::vector<Reaction> ReactionsFrom(const mfem::Vector &r, const mfem::Vector &x) const override;
  void SetPhysicalTime(bool on) override { loads_.SetPhysicalTime(on); physical_time_ = on; }
  mfem::Coefficient &ReferenceDensity() override
  {
    if (axisymmetric_) { return *density_axi_; }
    return density_;
  }
  bool Axisymmetric() const { return axisymmetric_; }
  OperatorStamp GradientStamp() const override { return gradient_stamp_; }
  // The accepted C and theta of the quadrature points.
  bool HasHistory() const override { return true; }
  void ResetHistory(double t) override;
  void AcceptStep(const mfem::Vector &x) override;
  const HistoryField *History() const { return history_.get(); }
  double AcceptedTime() const { return t_accepted_; }
  // u = 0, p = 0, theta = theta0.
  void InitialState(mfem::Vector &x) const override;

  void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
  mfem::Operator &GetGradient(const mfem::Vector &x) const override;

  void SetLoadFactor(double t) override;
  double LoadFactor() const override { return loads_.Time(); }
  void ApplyDirichlet(mfem::Vector &x) const override;
  bool IsLinear() const override { return false; }
  MPI_Comm Comm() const override { return fes_u_.GetComm(); }

  mfem::ParMesh &Mesh() { return mesh_; }
  mfem::ParFiniteElementSpace &DisplacementSpace() override { return fes_u_; }
  mfem::ParFiniteElementSpace &PressureSpace() { return fes_p_; }
  mfem::ParFiniteElementSpace &TemperatureSpace() { return fes_t_; }
  const mfem::Array<int> &BlockOffsets() const { return offsets_; }
  // The essential displacement dofs (the temperature's are EssentialTemperatureDofs).
  const mfem::Array<int> &EssentialTrueDofs() const override { return loads_.EssentialTrueDofs(); }
  const mfem::Array<int> &EssentialTemperatureDofs() const { return ess_t_; }
  HYPRE_BigInt GlobalTrueVSize() const override;
  std::string Description() const override;
  const ThermoMaterial &GetMaterial() const { return materials_[0]; }
  const std::vector<ThermoMaterial> &Materials() const { return materials_; }
  double ShearModulus() const { return mu_; }
  double BulkModulus() const { return kappa_; } // inf when incompressible
  bool Incompressible() const { return incompressible_; }
  double ReferenceTemperature() const { return theta0_; }

  // The mechanical free energy s(theta) Psi_iso + kappa u(J / J_theta) of the state.
  double InternalEnergy(const mfem::Vector &x) const override;
  void UpdateFields(const mfem::Vector &x) override;
  void RegisterFields(FieldRegistry &registry) override;
  mfem::ParGridFunction &Displacement() override { return *displacement_; }
  mfem::ParGridFunction &Pressure() { return *pressure_; }
  mfem::ParGridFunction &Temperature() { return *temperature_; }

  // Sparse LU (solver.linear.type: direct) only: the 3 x 3 block system has
  // no iterative solver here.
  std::unique_ptr<mfem::Solver> MakeLinearSolver(const LinearSolverConfig &cfg) override;

private:
  class BlockForm : public mfem::ParBlockNonlinearForm
  {
  public:
    using mfem::ParBlockNonlinearForm::ParBlockNonlinearForm;
    void SetEssentialTrueDofs(int block, const mfem::Array<int> &list)
    {
      list.Copy(*ess_tdofs[block]);
    }
  };
  struct TemperatureEntry
  {
    mfem::Array<int> marker;
    mfem::Coefficient *coef;
    BCOptions opt;
    mfem::Array<int> tdofs;
  };
  struct FluxEntry
  {
    mfem::Array<int> marker;
    mfem::Coefficient *coef;
    bool current_area;
    BCOptions opt;
    std::unique_ptr<double> scale;
  };

  void Build(const AppConfig &cfg);
  void ResetForm();
  void EnsureFields();
  void InitializeHistory();
  void UpdateHistory(const mfem::Vector &x);
  void SetEssential();
  tensor<double, 3, 3> GradientToF(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip,
                                   const mfem::DenseMatrix &grad) const;

  mfem::ParMesh &mesh_;
  int dim_;
  int order_;
  double rho0_;
  mfem::Vector density_table_;
  mfem::PWConstCoefficient density_;
  std::vector<ThermoMaterial> materials_;
  double mu_;
  double kappa_;
  bool incompressible_;
  double theta0_;
  mfem::H1_FECollection fec_u_;
  mfem::ParFiniteElementSpace fes_u_;
  mfem::H1_FECollection fec_p_;
  mfem::ParFiniteElementSpace fes_p_;
  mfem::ParFiniteElementSpace fes_t_; // the pressure's collection
  mfem::Array<mfem::ParFiniteElementSpace *> spaces_;
  mfem::Array<int> offsets_;
  std::unique_ptr<BlockForm> nlf_;
  std::unique_ptr<mfem::ParBlockNonlinearForm> energy_form_; // domain integrator only

  std::vector<std::unique_ptr<mfem::VectorCoefficient>> owned_coefs_;
  std::vector<std::unique_ptr<mfem::Coefficient>> owned_scalars_;
  LoadSet loads_;
  std::deque<mfem::Array<int>> follower_markers_;
  std::deque<mfem::Array<int>> contact_markers_;
  std::vector<std::unique_ptr<mfem::ParBlockNonlinearForm>> contact_forms_;
  std::vector<TemperatureEntry> temperatures_;
  std::deque<FluxEntry> fluxes_;
  mfem::Array<int> ess_p_empty_;
  mfem::Array<int> ess_t_;
  bool physical_time_ = false;

  bool finalized_ = false;
  std::unique_ptr<HistoryField> history_;
  double t_accepted_ = 0.0;

  mutable mfem::Operator *gradient_ = nullptr;
  std::shared_ptr<long> gradient_stamp_ = std::make_shared<long>(0);

  std::unique_ptr<mfem::ParGridFunction> displacement_;
  std::unique_ptr<mfem::ParGridFunction> pressure_;
  std::unique_ptr<mfem::ParGridFunction> temperature_;
  bool axisymmetric_ = false;
  mfem::FunctionCoefficient r2pi_;
  std::unique_ptr<mfem::ProductCoefficient> density_axi_;
  OutputConfig output_cfg_;
  std::unique_ptr<QuadratureFields> qfields_;
};

} // namespace cmf

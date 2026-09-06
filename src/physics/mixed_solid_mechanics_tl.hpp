// Mixed displacement-pressure total Lagrangian solid mechanics (Taylor-Hood:
// displacement H1 order p, pressure H1 order p - 1) for near- and fully
// incompressible decoupled materials.
//   R_u(u, p).w = int [P_iso(F) + p J F^{-T}] : Grad w dV - lambda [ext. loads]
//   R_p(u, p).q = int q (J - 1 - p/kappa) dV       (kappa = inf: q (J - 1))
// The unknown is the block true-dof vector [u; p]; Mult/GetGradient act on it.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "base/config.hpp"
#include "base/fields.hpp"
#include "materials/materials.hpp"
#include "mfem.hpp"
#include "physics/solid_problem.hpp"

namespace cmf
{

class MixedSolidMechanicsTL : public SolidProblem
{
public:
  MixedSolidMechanicsTL(mfem::ParMesh &mesh, const AppConfig &cfg,
                        const MixedMaterial &material);
  ~MixedSolidMechanicsTL() override = default;

  void AddDirichlet(const std::vector<int> &attrs, mfem::VectorCoefficient &u_bar) override;
  void AddTraction(const std::vector<int> &attrs, mfem::VectorCoefficient &T_bar) override;
  void SetBodyForce(mfem::VectorCoefficient &b) override;
  void ClearBoundaryConditions() override;
  void Finalize() override;

  void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
  mfem::Operator &GetGradient(const mfem::Vector &x) const override;

  void SetLoadFactor(double lambda) override;
  double LoadFactor() const override { return load_factor_; }
  void ApplyDirichlet(mfem::Vector &x) const override;
  MPI_Comm Comm() const override { return fes_u_.GetComm(); }

  mfem::ParMesh &Mesh() { return mesh_; }
  mfem::ParFiniteElementSpace &DisplacementSpace() override { return fes_u_; }
  mfem::ParFiniteElementSpace &PressureSpace() { return fes_p_; }
  const mfem::Array<int> &BlockOffsets() const { return offsets_; }
  const mfem::Array<int> &EssentialTrueDofs() const override { return ess_tdof_list_; }
  HYPRE_BigInt GlobalTrueVSize() const override;
  std::string Description() const override;
  const MixedMaterial &GetMaterial() const { return material_; }
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
  MixedMaterial material_;
  double mu_;
  double kappa_;
  bool incompressible_;
  mfem::H1_FECollection fec_u_;
  mfem::ParFiniteElementSpace fes_u_;
  mfem::H1_FECollection fec_p_;
  mfem::ParFiniteElementSpace fes_p_;
  mfem::Array<mfem::ParFiniteElementSpace *> spaces_;
  mfem::Array<int> offsets_;
  std::unique_ptr<mfem::ParBlockNonlinearForm> nlf_;

  std::vector<std::unique_ptr<mfem::VectorCoefficient>> owned_coefs_;
  std::vector<BCEntry> dirichlet_;
  std::vector<BCEntry> traction_;
  mfem::VectorCoefficient *body_force_ = nullptr;
  std::unique_ptr<mfem::VectorCoefficient> rho0_body_force_;

  bool finalized_ = false;
  double load_factor_ = 1.0;
  mfem::Array<int> ess_u_marker_;
  mfem::Array<int> ess_p_marker_;
  mfem::Array<int> ess_tdof_list_;
  mfem::Vector load_true_; // displacement block
  std::unique_ptr<mfem::HypreParMatrix> pressure_mass_;

  std::unique_ptr<mfem::ParGridFunction> displacement_;
  std::unique_ptr<mfem::ParGridFunction> pressure_;
  std::unique_ptr<mfem::L2_FECollection> l2_fec_;
  std::unique_ptr<mfem::ParFiniteElementSpace> l2_fes_;
  std::unique_ptr<mfem::ParGridFunction> vonmises_;
  std::unique_ptr<mfem::ParGridFunction> jacobian_;
};

} // namespace cmf

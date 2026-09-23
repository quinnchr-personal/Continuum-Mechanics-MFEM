// Scalar transport: the first-order flux/source physics of the framework on a
// scalar H1 space of order k (the unknown u: a concentration, a temperature,
// a potential), the second physics of the seam next to the solid.
//   c(u) du/dt - div F(u, grad u) + S(u, grad u) = 0,
//   F = kappa(u) grad u [- beta u],   S = [beta . grad u +] s u - f,
// with the affine laws of materials/scalar_transport_model.hpp, the kernel
// kernels/scalar_flux.hpp (implicit Euler over the step of the stepper from
// the accepted state u_n, held here as a grid function; dt = 0 is the steady
// problem of the pseudo-time loop), the boundary conditions of
// physics/scalar_conditions.hpp (Dirichlet entries with flows, inward flux
// entries), and the errors against an exact expression as an output. The
// residual is affine in the unknown when both laws are constant (IsLinear):
// Newton then accepts it at its round-off floor after one solve, and the
// Jacobian is assembled once and reused with its stamp until dt changes, the
// boundary conditions change or the velocity depends on t.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "base/config.hpp"
#include "base/fields.hpp"
#include "base/scalar_config.hpp"
#include "materials/scalar_transport_model.hpp"
#include "mfem.hpp"
#include "physics/quadrature_fields.hpp"
#include "physics/scalar_conditions.hpp"
#include "solvers/linear_solver.hpp"
#include "solvers/quasi_static.hpp"

namespace cmf
{

// L2 error, its ratio to the L2 norm of the exact solution (0 when that is
// below 1e-14) and the largest nodal error against the interpolant.
struct ScalarErrors
{
  double l2 = 0.0;
  double rel_l2 = 0.0;
  double linf_nodal = 0.0;
};

template <typename Model> class ScalarFluxIntegrator;

class ScalarTransport : public QuasiStaticProblem
{
public:
  // From the YAML schema: the laws, the coefficients, the conditions and the
  // exact solution installed; Finalize() still has to be called.
  ScalarTransport(mfem::ParMesh &mesh, const ScalarAppConfig &cfg);
  // Programmatic: the laws from the model, everything else through the API.
  ScalarTransport(mfem::ParMesh &mesh, int order, const ScalarTransportModel &model,
                  bool transient, int quadrature_order = 0);
  ~ScalarTransport() override;

  // Coefficients are not owned and must outlive this object; each call
  // invalidates Finalize().
  void AddDirichlet(const std::vector<int> &attrs, mfem::Coefficient &g,
                    const BCOptions &opt = BCOptions());
  void AddFlux(const std::vector<int> &attrs, mfem::Coefficient &g,
               const BCOptions &opt = BCOptions());
  void SetVelocity(mfem::VectorCoefficient &beta, bool uses_time);
  void SetSource(mfem::Coefficient &f);
  // u(x) at t = 0 (InitialState); default zero.
  void SetInitialCondition(mfem::Coefficient &u0);
  // u_ex(x, t) for Errors and the fields <u>_exact, <u>_error.
  void SetExact(mfem::Coefficient &u_ex);
  void ClearBoundaryConditions();
  void Finalize();
  const ScalarConditions &Conditions() const { return conditions_; }

  // mfem::Operator on true dofs.
  void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
  mfem::Operator &GetGradient(const mfem::Vector &x) const override;

  // QuasiStaticProblem: t is the pseudo-time (steady) or the physical time
  // (transient; dt = t - AcceptedTime() is the step under way).
  void SetLoadFactor(double t) override;
  double LoadFactor() const override { return conditions_.Time(); }
  void ApplyDirichlet(mfem::Vector &x) const override;
  MPI_Comm Comm() const override { return fes_.GetComm(); }
  bool IsLinear() const override { return model_.Linear(); }
  void AcceptStep(const mfem::Vector &x) override;

  void SetPhysicalTime(bool on) { conditions_.SetPhysicalTime(on); physical_time_ = on; }
  bool Transient() const { return transient_; }
  double Dt() const { return dt_; }
  double AcceptedTime() const { return t_accepted_; }
  // The initial state: the initial condition projected (zero without one);
  // it becomes the accepted state at t = 0.
  void InitialState(mfem::Vector &x);
  // Makes x the accepted state at time t (the start of an analysis).
  void ResetHistory(double t, const mfem::Vector &x);

  // The residual with its essential rows kept.
  void FullResidual(const mfem::Vector &x, mfem::Vector &r) const;
  // The flow into the domain through every Dirichlet entry at the state x.
  std::vector<Flow> Flows(const mfem::Vector &x) const;
  bool HasExact() const { return exact_ != nullptr; }
  // The errors of x against the exact solution at time t (the rule of order 2k + 3).
  ScalarErrors Errors(const mfem::Vector &x, double t);
  // The L2 norm of x.
  double NormL2(const mfem::Vector &x);

  mfem::ParMesh &Mesh() { return mesh_; }
  mfem::ParFiniteElementSpace &Space() { return fes_; }
  const mfem::Array<int> &EssentialTrueDofs() const { return conditions_.EssentialTrueDofs(); }
  HYPRE_BigInt GlobalTrueVSize() const; // collective
  std::string Description() const;
  const ScalarTransportModel &Model() const { return model_; }
  int Order() const { return order_; }
  const std::string &UnknownName() const { return unknown_name_; }
  OperatorStamp GradientStamp() const { return gradient_stamp_; }
  // Number of Jacobian assemblies so far.
  int GradientAssemblies() const { return int(*gradient_stamp_); }
  // Off: the Jacobian of a linear problem is reassembled at every call.
  void ReuseConstantGradient(bool on) { reuse_gradient_ = on; gradient_ = nullptr; }

  // The nodal unknown (and, with an exact solution, its interpolant and the
  // nodal error), the quadrature quantity flux with its presentations.
  void UpdateFields(const mfem::Vector &x);
  void RegisterFields(FieldRegistry &registry);
  mfem::ParGridFunction &Unknown() { return *unknown_; }
  // GMRES (or CG without a velocity) + scalar BoomerAMG, or the direct solver.
  std::unique_ptr<mfem::Solver> MakeLinearSolver(const LinearSolverConfig &cfg);

private:
  using Kernel = ScalarFluxIntegrator<ScalarTransportModel>;
  void Build(const ScalarAppConfig &cfg);
  void ResetForm();
  void EnsureFields();
  const mfem::IntegrationRule *ErrorRule(int geom) const;

  mfem::ParMesh &mesh_;
  int dim_;
  int order_;
  ScalarTransportModel model_;
  int quadrature_order_ = 0;
  bool transient_ = false;
  bool physical_time_ = false;
  std::string unknown_name_ = "u";
  mfem::H1_FECollection fec_;
  mfem::ParFiniteElementSpace fes_;
  std::unique_ptr<mfem::ParNonlinearForm> nlf_;
  Kernel *kernel_ = nullptr; // owned by nlf_
  ScalarConditions conditions_;
  mfem::VectorCoefficient *velocity_ = nullptr;
  bool velocity_uses_time_ = false;
  mfem::Coefficient *source_ = nullptr;
  mfem::Coefficient *initial_ = nullptr;
  mfem::Coefficient *exact_ = nullptr;
  std::vector<std::unique_ptr<mfem::VectorCoefficient>> owned_vectors_;
  std::vector<std::unique_ptr<mfem::Coefficient>> owned_scalars_;
  bool finalized_ = false;

  std::unique_ptr<mfem::ParGridFunction> u_old_; // the accepted state
  double t_accepted_ = 0.0;
  double dt_ = 0.0;

  bool reuse_gradient_ = true;
  mutable mfem::Operator *gradient_ = nullptr;
  mutable double gradient_dt_ = -1.0;
  std::shared_ptr<long> gradient_stamp_ = std::make_shared<long>(0);

  std::unique_ptr<mfem::ParGridFunction> unknown_;
  std::unique_ptr<mfem::ParGridFunction> exact_gf_;
  std::unique_ptr<mfem::ParGridFunction> error_gf_;
  OutputConfig output_cfg_;
  std::unique_ptr<QuadratureFields> qfields_;
  mutable std::vector<const mfem::IntegrationRule *> error_rules_;
};

// The physics of cfg on mesh, its YAML conditions installed.
std::unique_ptr<ScalarTransport> MakeScalarTransport(mfem::ParMesh &mesh, const ScalarAppConfig &cfg);

// The header of a transient run: the time steps and the time dependence of
// every entry, as DescribeTimeStepping of the solid.
std::vector<std::string> DescribeScalarTimeStepping(const ScalarAppConfig &cfg);

} // namespace cmf

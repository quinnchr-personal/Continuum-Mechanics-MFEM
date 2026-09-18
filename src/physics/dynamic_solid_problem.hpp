// Inertia for the solid formulations: the step equation of an implicit time
// integrator as a decorator over a SolidProblem.
//   int rho_R u_tt . w dV + R(u; w) = 0,     u(0) = u_0,  u_t(0) = v_0
// In total Lagrangian form the inertial term is integrated over the fixed
// reference mesh with the reference density, so its matrix M is constant and
// assembled once (consistent mass, rho_R by element attribute; per unit
// reference thickness in 2D). With the displacement as the unknown of the
// generalized-alpha family (solvers/time_integration.hpp) a time step solves
//   G(u) = S(u, t_{n+1}) + c_M M (u - u*) + h_n = 0,    dG/du = K(u) + c_M M,
// where S and K are the static residual and Jacobian of the wrapped problem:
// the dynamic term is a linear spring c_M M plus a known history load. This
// class is that equation behind the stepper's interface, so Newton, the line
// search, the linear solvers, Dirichlet data and bisection are those of the
// quasi-static analysis (SolveDynamic in solvers/quasi_static.hpp):
// SetLoadFactor(t) begins the step t_n -> t, AcceptStep(x) advances the
// history (u_n, v_n, a_n, S_n). A rejected step leaves the history untouched.
//
// Mixed u-p formulation: the unknown is [u; p], M acts on the displacement
// block, the constraint row is that of the static problem at t_{n+1} and h_n
// has no pressure block: a differential-algebraic system. Because the
// pressure force is interpolated with the rest of S_u, an error in p_n
// returns with the factor -af / (1 - af) = -rho_inf: use a dissipative scheme.
//
// Prescribed displacements are imposed on u_{n+1}; v and a on those dofs
// follow from the update formulas. M is kept twice: whole (residual, energies,
// reactions, and the coupling of a prescribed boundary acceleration into the
// free rows) and with essential rows and columns zeroed (the Jacobian).
#pragma once

#include <memory>
#include <vector>

#include "base/config.hpp"
#include "base/fields.hpp"
#include "mfem.hpp"
#include "physics/solid_problem.hpp"
#include "solvers/quasi_static.hpp"
#include "solvers/time_integration.hpp"

namespace cmf
{

class DynamicSolidProblem : public QuasiStaticProblem
{
public:
  // The boundary conditions of problem must not change after Initialize().
  DynamicSolidProblem(SolidProblem &problem, const DynamicsConfig &cfg);
  ~DynamicSolidProblem() override = default;

  // Initial state; coefficients are evaluated in Initialize and not kept.
  // Without an initial displacement the displacement block of the vector
  // given to Initialize is u_0.
  void SetInitialDisplacement(mfem::VectorCoefficient &u0) { u0_ = &u0; }
  void SetInitialVelocity(mfem::VectorCoefficient &v0) { v0_ = &v0; }
  // Keeps a coefficient alive for the life of this object (MakeDynamicSolidProblem).
  mfem::VectorCoefficient &Own(std::unique_ptr<mfem::VectorCoefficient> c)
  {
    owned_.push_back(std::move(c));
    return *owned_.back();
  }
  // a_0 on every displacement true dof instead of the consistent one (a
  // manufactured solution whose boundary accelerates at t = 0).
  void SetInitialAcceleration(const mfem::Vector &a0) { a0_given_ = a0; }

  // Switches the loads to physical time, assembles M, writes u_0 and the
  // Dirichlet data of t0 into x (the whole unknown) and forms v_0, S_0 and
  // a_0: M a_0 = -S(u_0, t0) on the free dofs with the loads at their right
  // limit, zero on the essential dofs. Returns the largest change the
  // Dirichlet data made to u_0 on the essential dofs (global; an initial
  // displacement that contradicts the boundary data shows here).
  double Initialize(mfem::Vector &x, double t0 = 0.0);

  // mfem::Operator / QuasiStaticProblem on the unknown of the wrapped problem.
  void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
  mfem::Operator &GetGradient(const mfem::Vector &x) const override;
  void SetLoadFactor(double t) override; // begins the step from Time() to t
  double LoadFactor() const override { return t_; }
  void ApplyDirichlet(mfem::Vector &x) const override { problem_.ApplyDirichlet(x); }
  MPI_Comm Comm() const override { return problem_.Comm(); }
  bool IsLinear() const override { return problem_.IsLinear(); }
  void AcceptStep(const mfem::Vector &x) override;

  // The solver of the wrapped problem, following this operator's stamp: for a
  // linear problem K + c_M M is formed, and the solver set up, once per dt.
  std::unique_ptr<mfem::Solver> MakeLinearSolver(const LinearSolverConfig &cfg);
  void ReuseConstantOperator(bool on) { reuse_ = on; sum_.reset(); }
  // Number of times K + c_M M has been formed.
  int OperatorAssemblies() const { return int(*stamp_); }

  // The accepted state (displacement true dofs).
  double Time() const { return t_n_; }
  int Steps() const { return steps_; }
  const mfem::Vector &Displacement() const { return u_n_; }
  const mfem::Vector &Velocity() const { return v_n_; }
  const mfem::Vector &Acceleration() const { return a_n_; }
  const TimeIntegration &Scheme() const { return ti_; }
  SolidProblem &Static() { return problem_; }
  // The consistent mass matrix of the displacement space, whole, and with
  // the essential rows and columns zeroed (the part that joins the Jacobian).
  mfem::HypreParMatrix &Mass() { return *M_; }
  mfem::HypreParMatrix &EliminatedMass() { return *M_e_; }

  // Energies of the accepted state: v.Mv/2; the work of the dead loads and of
  // the supports (reactions through prescribed motion) accumulated by the
  // trapezoidal rule, for which kinetic + internal - external work is
  // constant when the problem is linear and the scheme the trapezoidal rule.
  // Follower pressures are not dead loads and are not counted.
  double KineticEnergy() const;
  double ExternalWork() const;
  // The external work costs one evaluation of the full static residual per
  // accepted step (it needs the support forces of every step). Without it
  // that evaluation is left to the first call of Reactions() for a state, and
  // the S_n a force-interpolating scheme needs is taken from Newton's last
  // residual. On by default.
  void TrackExternalWork(bool on) { track_work_ = on; }
  // Resultant of M a_n per component (the rate of linear momentum), global.
  std::vector<double> InertialForce() const;
  // Reactions of the full balance S_full + M a at the accepted state: a
  // support force includes the inertia it carries.
  std::vector<Reaction> Reactions() const;

  // Nodal fields velocity and acceleration next to those of the wrapped
  // problem; UpdateFields(x) refreshes all of them from the accepted state.
  void RegisterFields(FieldRegistry &registry);
  void UpdateFields(const mfem::Vector &x);

private:
  void AssembleMass(bool eliminated);
  void ZeroEssentialRows(mfem::Vector &y) const;
  // S_full(x) + M a on the displacement block.
  void FullBalance(const mfem::Vector &x, const mfem::Vector &a, mfem::Vector &balance) const;

  SolidProblem &problem_;
  TimeIntegration ti_;
  mfem::ParFiniteElementSpace &fes_;
  int n_u_;

  std::unique_ptr<mfem::HypreParMatrix> M_;   // whole
  std::unique_ptr<mfem::HypreParMatrix> M_e_; // essential rows and columns zeroed

  std::vector<std::unique_ptr<mfem::VectorCoefficient>> owned_;
  mfem::VectorCoefficient *u0_ = nullptr;
  mfem::VectorCoefficient *v0_ = nullptr;
  mfem::Vector a0_given_;
  bool initialized_ = false;

  // History of the accepted state and the step under way.
  double t_n_ = 0.0, t_ = 0.0, dt_ = 0.0, c_M_ = 0.0;
  // c_M again, shared with the saddle-point solver of the mixed formulation
  // (the inertial part of its Schur complement approximation).
  std::shared_ptr<double> mass_factor_ = std::make_shared<double>(0.0);
  bool stepping_ = false;
  int steps_ = 0;
  mfem::Vector u_n_, v_n_, a_n_;   // displacement true dofs
  mfem::Vector x_n_;               // the accepted unknown (with the pressure block, if any)
  mfem::Vector S_n_;               // static residual of the unknown, essential rows zero
  // S_full + M a of the unknown (reactions, support work); formed on demand
  // when the step did not need it.
  mutable mfem::Vector balance_n_;
  mutable bool balance_valid_ = false;
  bool track_work_ = true;
  mfem::Vector f_ext_n_;           // dead load at t_n
  mfem::Vector u_pred_, v_pred_, h_n_;
  double external_work_ = 0.0;
  mutable mfem::Vector w_, Mw_;
  // The static residual of the last Mult and its argument: Newton's last
  // evaluation is at the state it returns, so S_{n+1} is usually already here.
  mutable mfem::Vector x_last_, S_last_;

  // K + c_M M (displacement block) and, for the mixed unknown, the block
  // operator around it; the stamp is shared with the linear solver.
  bool reuse_ = true;
  mutable std::unique_ptr<mfem::HypreParMatrix> sum_;
  mutable std::unique_ptr<mfem::BlockOperator> block_;
  mutable mfem::Array<int> block_offsets_;
  mutable const mfem::Operator *seen_operator_ = nullptr;
  mutable long seen_stamp_ = -1;
  mutable double seen_c_M_ = -1.0;
  std::shared_ptr<long> stamp_ = std::make_shared<long>(0);

  std::unique_ptr<mfem::ParGridFunction> velocity_;
  std::unique_ptr<mfem::ParGridFunction> acceleration_;
};

// The decorator for cfg.dynamics with the initial state of its expressions
// (evaluated in Initialize; the coefficients are owned by the returned object).
std::unique_ptr<DynamicSolidProblem> MakeDynamicSolidProblem(SolidProblem &problem,
                                                             const AppConfig &cfg);

// The header of a dynamic run, one line per entry: the scheme and the time
// steps, the resolved time dependence of every Dirichlet entry, traction and
// the body force (the default schedule differs from the quasi-static one, so
// nothing about it is left implicit), and warnings (a conditionally stable
// Newmark pair).
std::vector<std::string> DescribeDynamics(const AppConfig &cfg);

} // namespace cmf

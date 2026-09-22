#include "physics/dynamic_solid_problem.hpp"

#include <algorithm>
#include <cmath>

#include "base/coefficients.hpp"
#include "solvers/direct_solver.hpp"
#include "solvers/linear_solver.hpp"
#include "solvers/saddle_point_solver.hpp"

namespace cmf
{

namespace
{

// The displacement block of the unknown (it leads the block vector of the
// mixed formulation and is the whole unknown otherwise), as a view. Name the
// view before copying from it: assigning the temporary itself would bind to
// mfem::Vector's move assignment and leave the target an alias of x.
mfem::Vector Head(const mfem::Vector &x, int n)
{
  return mfem::Vector(const_cast<double *>(x.GetData()), n);
}

void ProjectTrueDofs(mfem::ParFiniteElementSpace &fes, mfem::VectorCoefficient &c, double t,
                     mfem::Vector &tv)
{
  if (c.GetVDim() != fes.GetVDim())
  {
    throw ConfigError("initial state: coefficient has " + std::to_string(c.GetVDim()) +
                      " components, expected " + std::to_string(fes.GetVDim()));
  }
  mfem::ParGridFunction g(&fes);
  c.SetTime(t);
  g.ProjectCoefficient(c);
  mfem::Vector projected;
  g.GetTrueDofs(projected);
  tv = projected;
}

} // namespace

DynamicSolidProblem::DynamicSolidProblem(SolidProblem &problem, const DynamicsConfig &cfg)
  : QuasiStaticProblem(problem.Height()), problem_(problem), ti_(MakeTimeIntegration(cfg)),
    fes_(problem.DisplacementSpace()), n_u_(fes_.GetTrueVSize())
{
  w_.SetSize(n_u_);
  Mw_.SetSize(n_u_);
}

// M once (it depends on nothing but the mesh and the density); the copy with
// the essential rows and columns zeroed whenever Initialize is called.
void DynamicSolidProblem::AssembleMass(bool eliminated)
{
  if (!M_)
  {
    mfem::ParBilinearForm mass(&fes_);
    mass.AddDomainIntegrator(new mfem::VectorMassIntegrator(problem_.ReferenceDensity()));
    mass.Assemble();
    mass.Finalize();
    M_.reset(mass.ParallelAssemble());
  }
  if (eliminated)
  {
    M_e_ = std::make_unique<mfem::HypreParMatrix>(*M_);
    M_e_->EliminateBC(problem_.EssentialTrueDofs(), mfem::Operator::DIAG_ZERO);
    sum_.reset();
  }
}

void DynamicSolidProblem::ZeroEssentialRows(mfem::Vector &y) const
{
  const mfem::Array<int> &ess = problem_.EssentialTrueDofs();
  for (int i = 0; i < ess.Size(); i++) { y(ess[i]) = 0.0; }
}

void DynamicSolidProblem::FullBalance(const mfem::Vector &x, const mfem::Vector &a,
                                      mfem::Vector &balance) const
{
  problem_.FullResidual(x, balance);
  M_->Mult(a, Mw_);
  mfem::Vector balance_u = Head(balance, n_u_);
  balance_u += Mw_;
}

double DynamicSolidProblem::ExternalWork() const
{
  MFEM_VERIFY(track_work_, "DynamicSolidProblem: the external work is not tracked (TrackExternalWork)");
  return external_work_;
}

double DynamicSolidProblem::Initialize(mfem::Vector &x, double t0)
{
  MFEM_VERIFY(x.Size() == Height(), "DynamicSolidProblem: the unknown has the wrong size");
  problem_.SetPhysicalTime(true);
  t_n_ = t_ = t0;
  problem_.ResetHistory(t0);   // a rate-dependent material starts from its rest state at t0
  problem_.SetLoadFactor(t0); // finalizes the boundary conditions if needed
  AssembleMass(true);

  mfem::Vector x_u = Head(x, n_u_);
  if (u0_) { ProjectTrueDofs(fes_, *u0_, t0, x_u); }
  const mfem::Vector before(x_u);
  problem_.ApplyDirichlet(x);
  double change = 0.0, global_change = 0.0;
  const mfem::Array<int> &ess = problem_.EssentialTrueDofs();
  for (int i = 0; i < ess.Size(); i++)
  {
    change = std::max(change, std::abs(x_u(ess[i]) - before(ess[i])));
  }
  MPI_Allreduce(&change, &global_change, 1, MPI_DOUBLE, MPI_MAX, Comm());

  u_n_ = x_u;
  v_n_.SetSize(n_u_);
  v_n_ = 0.0;
  if (v0_) { ProjectTrueDofs(fes_, *v0_, t0, v_n_); }

  mfem::Vector r;
  problem_.FullResidual(x, r);
  if (Height() > n_u_)
  {
    // Mixed formulation at finite kappa: the pressure that belongs to u_0. The
    // constraint row is affine in p, R_p(u_0, p + dp) = R_p(u_0, p) + K_pp dp,
    // and K_pp = -M_p / kappa is definite. An incompressible material has
    // K_pp = 0 and leaves p_0 as given (zero): its initial pressure would
    // follow from the constraint on the acceleration, which is not enforced.
    auto *J = dynamic_cast<mfem::BlockOperator *>(&problem_.GetGradient(x));
    MFEM_VERIFY(J, "DynamicSolidProblem: a block unknown needs a block Jacobian");
    auto *Kpp = J->IsZeroBlock(1, 1) ? nullptr : dynamic_cast<mfem::HypreParMatrix *>(&J->GetBlock(1, 1));
    mfem::Vector diag;
    if (Kpp) { Kpp->GetDiag(diag); }
    double largest = diag.Size() ? diag.Normlinf() : 0.0, global_largest = 0.0;
    MPI_Allreduce(&largest, &global_largest, 1, MPI_DOUBLE, MPI_MAX, Comm());
    const int n_p = Height() - n_u_;
    mfem::Vector r_p(r.GetData() + n_u_, n_p);
    const double rp_norm = std::sqrt(mfem::InnerProduct(Comm(), r_p, r_p));
    if (global_largest > 0.0 && rp_norm > 0.0)
    {
      mfem::HypreParMatrix C(*Kpp);
      C *= -1.0;
      mfem::HypreSmoother jacobi(C, mfem::HypreSmoother::Jacobi);
      mfem::CGSolver cg(Comm());
      cg.SetRelTol(1e-14);
      cg.SetAbsTol(0.0);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(-1);
      cg.SetPreconditioner(jacobi);
      cg.SetOperator(C);
      cg.iterative_mode = false;
      mfem::Vector dp(n_p), x_p(x.GetData() + n_u_, n_p);
      cg.Mult(r_p, dp);
      x_p += dp;
      problem_.FullResidual(x, r);
    }
  }
  S_n_ = r;
  ZeroEssentialRows(S_n_);

  a_n_.SetSize(n_u_);
  a_n_ = 0.0;
  if (a0_given_.Size())
  {
    MFEM_VERIFY(a0_given_.Size() == n_u_, "DynamicSolidProblem: initial acceleration has the wrong size");
    a_n_ = a0_given_;
  }
  else
  {
    // M a_0 = -S(u_0, t0) on the free dofs; zero on the essential ones.
    mfem::HypreParMatrix M_free(*M_);
    M_free.EliminateBC(ess, mfem::Operator::DIAG_ONE);
    const mfem::Vector S_u = Head(S_n_, n_u_);
    mfem::Vector rhs(S_u);
    rhs *= -1.0;
    mfem::HypreSmoother jacobi(M_free, mfem::HypreSmoother::Jacobi);
    mfem::CGSolver cg(Comm());
    cg.SetRelTol(1e-15);
    cg.SetAbsTol(0.0);
    cg.SetMaxIter(2000);
    cg.SetPrintLevel(0);
    cg.SetPreconditioner(jacobi);
    cg.SetOperator(M_free);
    cg.iterative_mode = false;
    cg.Mult(rhs, a_n_);
  }

  balance_n_ = r;
  M_->Mult(a_n_, Mw_);
  mfem::Vector balance_u = Head(balance_n_, n_u_);
  balance_u += Mw_;
  balance_valid_ = true;
  x_n_ = x;
  f_ext_n_ = problem_.Loads().ExternalLoad();
  external_work_ = 0.0;
  steps_ = 0;
  initialized_ = true;
  return global_change;
}

void DynamicSolidProblem::SetLoadFactor(double t)
{
  MFEM_VERIFY(initialized_, "DynamicSolidProblem: call Initialize() first");
  MFEM_VERIFY(t > t_n_, "DynamicSolidProblem: a time step must advance the time");
  t_ = t;
  // Equal steps t_final k / n differ in the last digits of t_{k+1} - t_k: the
  // increment of the previous step is kept when the new one agrees with it to
  // 1e-12, so that c_M, and with it K + c_M M of a linear problem, stays the same.
  const double dt = t - t_n_;
  if (!(std::abs(dt - dt_) <= 1e-12 * dt_)) { dt_ = dt; }
  stepping_ = true;
  c_M_ = ti_.MassFactor(dt_);
  *mass_factor_ = c_M_;
  const double am = ti_.alpha_m, af = ti_.alpha_f;

  u_pred_ = u_n_;
  u_pred_.Add(dt_, v_n_);
  u_pred_.Add(dt_ * dt_ * (0.5 - ti_.beta), a_n_);
  v_pred_ = v_n_;
  v_pred_.Add(dt_ * (1.0 - ti_.gamma), a_n_);

  // h_n = (am M a_n + af S_n) / (1 - af) on the displacement block.
  h_n_.SetSize(Height());
  h_n_ = 0.0;
  mfem::Vector h_u = Head(h_n_, n_u_);
  if (am != 0.0)
  {
    M_->Mult(a_n_, Mw_);
    h_u.Add(am / (1.0 - af), Mw_);
  }
  if (af != 0.0) { h_u.Add(af / (1.0 - af), Head(S_n_, n_u_)); }
  ZeroEssentialRows(h_n_);

  problem_.SetLoadFactor(t);
}

void DynamicSolidProblem::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
  MFEM_VERIFY(stepping_, "DynamicSolidProblem: no time step under way (SetLoadFactor)");
  problem_.Mult(x, y);
  x_last_ = x;
  S_last_ = y;
  const mfem::Vector x_u = Head(x, n_u_);
  w_ = x_u;
  w_ -= u_pred_;
  M_->Mult(w_, Mw_);
  mfem::Vector y_u = Head(y, n_u_);
  y_u.Add(c_M_, Mw_);
  y += h_n_;
  ZeroEssentialRows(y);
}

mfem::Operator &DynamicSolidProblem::GetGradient(const mfem::Vector &x) const
{
  MFEM_VERIFY(stepping_, "DynamicSolidProblem: no time step under way (SetLoadFactor)");
  mfem::Operator &K = problem_.GetGradient(x);
  const long inner_stamp = *problem_.GradientStamp();
  const bool unchanged = reuse_ && sum_ && &K == seen_operator_ && inner_stamp == seen_stamp_ &&
                         c_M_ == seen_c_M_;
  if (!unchanged)
  {
    seen_operator_ = &K;
    seen_stamp_ = inner_stamp;
    seen_c_M_ = c_M_;
    ++*stamp_;
    if (auto *Kh = dynamic_cast<mfem::HypreParMatrix *>(&K))
    {
      sum_.reset(mfem::Add(1.0, *Kh, c_M_, *M_e_));
      block_.reset();
    }
    else
    {
      auto *Kb = dynamic_cast<mfem::BlockOperator *>(&K);
      MFEM_VERIFY(Kb && Kb->NumRowBlocks() == 2 && Kb->NumColBlocks() == 2,
                  "DynamicSolidProblem: the Jacobian must be a HypreParMatrix or a 2x2 BlockOperator");
      auto *Kuu = dynamic_cast<mfem::HypreParMatrix *>(&Kb->GetBlock(0, 0));
      MFEM_VERIFY(Kuu, "DynamicSolidProblem: the displacement block must be a HypreParMatrix");
      sum_.reset(mfem::Add(1.0, *Kuu, c_M_, *M_e_));
      Kb->RowOffsets().Copy(block_offsets_);
      block_ = std::make_unique<mfem::BlockOperator>(block_offsets_);
      block_->SetBlock(0, 0, sum_.get());
      for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
          if ((i == 0 && j == 0) || Kb->IsZeroBlock(i, j)) { continue; }
          block_->SetBlock(i, j, &Kb->GetBlock(i, j));
        }
    }
  }
  if (block_) { return *block_; }
  return *sum_;
}

void DynamicSolidProblem::AcceptStep(const mfem::Vector &x)
{
  MFEM_VERIFY(stepping_, "DynamicSolidProblem: no time step under way (SetLoadFactor)");
  const mfem::Vector x_u = Head(x, n_u_);
  mfem::Vector a_new(x_u), v_new(v_pred_), du(x_u);
  a_new -= u_pred_;
  a_new /= ti_.beta * dt_ * dt_;
  v_new.Add(ti_.gamma * dt_, a_new);
  du -= u_n_;

  // The full balance S_full + M a of the new state: its essential rows are the
  // support forces; with those rows zeroed the static part is S_{n+1}, which
  // the next step interpolates when alpha_f is not zero. It costs a residual
  // evaluation and is formed here only for the external work.
  if (track_work_)
  {
    mfem::Vector balance;
    problem_.FullResidual(x, balance);
    S_n_ = balance;
    ZeroEssentialRows(S_n_);
    M_->Mult(a_new, Mw_);
    mfem::Vector balance_u = Head(balance, n_u_);
    balance_u += Mw_;

    // External work over the step by the trapezoidal rule: dead loads on every
    // dof, support forces through the prescribed motion.
    MFEM_VERIFY(balance_valid_, "DynamicSolidProblem: TrackExternalWork must be set before Initialize");
    const mfem::Vector &f_new = problem_.Loads().ExternalLoad();
    const double work = 0.5 * (mfem::InnerProduct(Comm(), f_ext_n_, du) +
                               mfem::InnerProduct(Comm(), f_new, du));
    const mfem::Array<int> &ess = problem_.EssentialTrueDofs();
    double support = 0.0, global_support = 0.0;
    for (int k = 0; k < ess.Size(); k++)
    {
      const int i = ess[k];
      support += 0.5 * (balance_n_(i) + balance(i)) * du(i);
    }
    MPI_Allreduce(&support, &global_support, 1, MPI_DOUBLE, MPI_SUM, Comm());
    external_work_ += work + global_support;
    f_ext_n_ = f_new;
    balance_n_ = balance;
    balance_valid_ = true;
  }
  else
  {
    // Only S_{n+1} is needed, and only when the scheme interpolates the
    // forces: it is the static part of Newton's last residual when that was
    // evaluated at the accepted state (the same on every rank).
    if (ti_.alpha_f != 0.0)
    {
      int same = x_last_.Size() == x.Size() ? 1 : 0, all_same = 0;
      for (int i = 0; same && i < x.Size(); i++) { same = x_last_(i) == x(i) ? 1 : 0; }
      MPI_Allreduce(&same, &all_same, 1, MPI_INT, MPI_MIN, Comm());
      if (all_same) { S_n_ = S_last_; }
      else { problem_.Mult(x, S_n_); }
    }
    balance_valid_ = false;
  }

  u_n_ = x_u;
  v_n_ = v_new;
  a_n_ = a_new;
  x_n_ = x;
  t_n_ = t_;
  stepping_ = false;
  steps_++;
  // The wrapped problem's history (a rate-dependent material) advances last:
  // everything above evaluated the static residual of the step at its end.
  problem_.AcceptStep(x);
}

double DynamicSolidProblem::KineticEnergy() const
{
  MFEM_VERIFY(initialized_, "DynamicSolidProblem: call Initialize() first");
  M_->Mult(v_n_, Mw_);
  return 0.5 * mfem::InnerProduct(Comm(), v_n_, Mw_);
}

std::vector<double> DynamicSolidProblem::InertialForce() const
{
  MFEM_VERIFY(initialized_, "DynamicSolidProblem: call Initialize() first");
  // True dofs are ordered by vdim (all components of a node consecutive).
  const int dim = fes_.GetVDim();
  M_->Mult(a_n_, Mw_);
  std::vector<double> local(std::size_t(dim), 0.0), global(std::size_t(dim), 0.0);
  for (int i = 0; i < n_u_; i++) { local[std::size_t(i % dim)] += Mw_(i); }
  MPI_Allreduce(local.data(), global.data(), dim, MPI_DOUBLE, MPI_SUM, Comm());
  return global;
}

std::vector<Reaction> DynamicSolidProblem::Reactions() const
{
  MFEM_VERIFY(initialized_, "DynamicSolidProblem: call Initialize() first");
  if (!balance_valid_)
  {
    FullBalance(x_n_, a_n_, balance_n_);
    balance_valid_ = true;
  }
  return problem_.ReactionsFrom(balance_n_, u_n_);
}

void DynamicSolidProblem::RegisterFields(FieldRegistry &registry)
{
  problem_.RegisterFields(registry);
  if (!velocity_)
  {
    velocity_ = std::make_unique<mfem::ParGridFunction>(&fes_);
    acceleration_ = std::make_unique<mfem::ParGridFunction>(&fes_);
    *velocity_ = 0.0;
    *acceleration_ = 0.0;
  }
  registry.AddExternal("velocity", *velocity_);
  registry.AddExternal("acceleration", *acceleration_);
}

void DynamicSolidProblem::UpdateFields(const mfem::Vector &x)
{
  problem_.UpdateFields(x);
  if (velocity_ && initialized_)
  {
    velocity_->SetFromTrueDofs(v_n_);
    acceleration_->SetFromTrueDofs(a_n_);
  }
}

std::unique_ptr<mfem::Solver> DynamicSolidProblem::MakeLinearSolver(const LinearSolverConfig &cfg)
{
  std::unique_ptr<mfem::Solver> solver = problem_.MakeLinearSolver(cfg);
  if (auto *ours = dynamic_cast<LinearSolver *>(solver.get())) { ours->SetOperatorStamp(stamp_); }
  else if (auto *direct = dynamic_cast<DirectSolver *>(solver.get())) { direct->SetOperatorStamp(stamp_); }
  else if (auto *saddle = dynamic_cast<SaddlePointSolver *>(solver.get()))
  {
    saddle->SetOperatorStamp(stamp_);
    // Lumped mass: the diagonal of M scaled to the total mass (positive on
    // every element type and order, which row sums are not).
    AssembleMass(false);
    mfem::Vector lumped(n_u_), ones(n_u_);
    M_->GetDiag(lumped);
    ones = 1.0;
    M_->Mult(ones, Mw_);
    const double total = mfem::InnerProduct(Comm(), ones, Mw_), trace = mfem::InnerProduct(Comm(), ones, lumped);
    lumped *= total / trace;
    saddle->SetInertia(lumped, mass_factor_);
  }
  return solver;
}

std::unique_ptr<DynamicSolidProblem> MakeDynamicSolidProblem(SolidProblem &problem,
                                                             const AppConfig &cfg)
{
  auto dynamic = std::make_unique<DynamicSolidProblem>(problem, cfg.dynamics);
  const int dim = problem.DisplacementSpace().GetVDim();
  auto check = [dim](const std::vector<std::string> &e, const std::string &key)
  {
    if (int(e.size()) != dim)
    {
      throw ConfigError("key 'dynamics.initial." + key + "' has " + std::to_string(e.size()) +
                        " components, mesh dimension is " + std::to_string(dim));
    }
  };
  if (!cfg.dynamics.initial_displacement.empty())
  {
    check(cfg.dynamics.initial_displacement, "displacement");
    dynamic->SetInitialDisplacement(dynamic->Own(
      std::make_unique<ExpressionVectorCoefficient>(cfg.dynamics.initial_displacement)));
  }
  if (!cfg.dynamics.initial_velocity.empty())
  {
    check(cfg.dynamics.initial_velocity, "velocity");
    dynamic->SetInitialVelocity(dynamic->Own(
      std::make_unique<ExpressionVectorCoefficient>(cfg.dynamics.initial_velocity)));
  }
  return dynamic;
}

namespace
{

// "<n> time steps of <dt>" or "... from <smallest> to <largest>".
std::string DescribeSteps(double t_final, const std::vector<double> &breakpoints)
{
  double smallest = t_final, largest = 0.0, t_prev = 0.0;
  for (double t : breakpoints)
  {
    smallest = std::min(smallest, t - t_prev);
    largest = std::max(largest, t - t_prev);
    t_prev = t;
  }
  char buf[128];
  if (largest - smallest <= 1e-9 * largest)
  {
    std::snprintf(buf, sizeof(buf), "%zu time steps of %g", breakpoints.size(), largest);
  }
  else
  {
    std::snprintf(buf, sizeof(buf), "%zu time steps from %g to %g", breakpoints.size(), smallest,
                  largest);
  }
  return buf;
}

// The time dependence of every load entry, one line each.
void DescribeEntries(const AppConfig &cfg, std::vector<std::string> &lines)
{
  for (const BoundaryCondition &bc : cfg.bcs.dirichlet)
  {
    lines.push_back("  dirichlet " + bc.name + ": " + DescribeTimeDependence(bc.schedule, bc.expression));
  }
  for (const BoundaryCondition &bc : cfg.bcs.traction)
  {
    lines.push_back("  traction " + bc.name + " (" + bc.type + "): " +
                    DescribeTimeDependence(bc.schedule, bc.expression));
  }
  for (const ContactCondition &c : cfg.bcs.contact)
  {
    lines.push_back("  contact " + c.name + " (" + c.type + "): centre " +
                    DescribeTimeDependence(Schedule::Constant(), c.center) + ", penalty " +
                    DescribeTimeDependence(c.schedule, {}));
  }
  if (!cfg.body_force.Empty())
  {
    lines.push_back("  body force: " +
                    DescribeTimeDependence(cfg.body_force.schedule, cfg.body_force.expression));
  }
}

} // namespace

std::vector<std::string> DescribeTimeStepping(const AppConfig &cfg)
{
  std::vector<std::string> lines;
  char buf[256];
  std::snprintf(buf, sizeof(buf), "time: quasi-static in physical time, t_final %g, %s",
                cfg.time.t_final, DescribeSteps(cfg.time.t_final, cfg.time.breakpoints).c_str());
  lines.push_back(buf);
  DescribeEntries(cfg, lines);
  return lines;
}

std::vector<std::string> DescribeDynamics(const AppConfig &cfg)
{
  const DynamicsConfig &d = cfg.dynamics;
  const TimeIntegration ti = MakeTimeIntegration(d);
  std::vector<std::string> lines;
  char buf[256];
  std::snprintf(buf, sizeof(buf), "dynamics: %s, t_final %g, %s", ti.Description().c_str(),
                d.t_final, DescribeSteps(d.t_final, d.breakpoints).c_str());
  lines.push_back(buf);
  DescribeEntries(cfg, lines);
  if (cfg.formulation == "mixed" && !ti.Dissipative())
  {
    lines.push_back("  warning: the pressure of the mixed formulation carries no inertia, and an error in it "
                    "returns at every step with the spectral radius at infinity, here 1: use "
                    "generalized_alpha with rho_inf < 1 (or hht with alpha > 0)");
  }
  if (!ti.UnconditionallyStable())
  {
    lines.push_back("  warning: this parameter pair is only conditionally stable "
                    "(unconditional stability needs 2 beta >= gamma >= 1/2)");
  }
  return lines;
}

} // namespace cmf

#include "physics/dynamic_solid_problem.hpp"

#include <algorithm>
#include <cmath>

#include "base/coefficients.hpp"
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

void DynamicSolidProblem::AssembleMass()
{
  mfem::ParBilinearForm mass(&fes_);
  mass.AddDomainIntegrator(new mfem::VectorMassIntegrator(problem_.ReferenceDensity()));
  mass.Assemble();
  mass.Finalize();
  M_.reset(mass.ParallelAssemble());
  M_e_ = std::make_unique<mfem::HypreParMatrix>(*M_);
  M_e_->EliminateBC(problem_.EssentialTrueDofs(), mfem::Operator::DIAG_ZERO);
  sum_.reset();
}

void DynamicSolidProblem::ZeroEssentialRows(mfem::Vector &y) const
{
  const mfem::Array<int> &ess = problem_.EssentialTrueDofs();
  for (int i = 0; i < ess.Size(); i++) { y(ess[i]) = 0.0; }
}

double DynamicSolidProblem::Initialize(mfem::Vector &x, double t0)
{
  MFEM_VERIFY(x.Size() == Height(), "DynamicSolidProblem: the unknown has the wrong size");
  problem_.SetPhysicalTime(true);
  t_n_ = t_ = t0;
  problem_.SetLoadFactor(t0); // finalizes the boundary conditions if needed
  AssembleMass();

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
  // support forces; with those rows zeroed the static part is S_{n+1}.
  mfem::Vector balance;
  problem_.FullResidual(x, balance);
  S_n_ = balance;
  ZeroEssentialRows(S_n_);
  M_->Mult(a_new, Mw_);
  mfem::Vector balance_u = Head(balance, n_u_);
  balance_u += Mw_;

  // External work over the step by the trapezoidal rule: dead loads on every
  // dof, support forces through the prescribed motion.
  const mfem::Vector &f_new = problem_.Loads().ExternalLoad();
  double work = 0.5 * (mfem::InnerProduct(Comm(), f_ext_n_, du) +
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

  u_n_ = x_u;
  v_n_ = v_new;
  a_n_ = a_new;
  balance_n_ = balance;
  f_ext_n_ = f_new;
  t_n_ = t_;
  stepping_ = false;
  steps_++;
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
  else if (auto *saddle = dynamic_cast<SaddlePointSolver *>(solver.get()))
  {
    saddle->SetOperatorStamp(stamp_);
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

} // namespace cmf

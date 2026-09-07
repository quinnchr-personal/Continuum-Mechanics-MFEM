// Boundary conditions and load scheduling (plan doc/bc_loading_plan.md):
// L1 schedules and step control, L2 expression data, L3 component-wise
// Dirichlet data, L4 pressure tractions (dead and follower).
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "base/coefficients.hpp"
#include "base/mesh_input.hpp"
#include "base/probes.hpp"
#include "mfem.hpp"
#include "physics/solid_problem.hpp"
#include "solvers/quasi_static.hpp"
#include "test_util.hpp"

namespace
{

// Cook's membrane, compressible neo-Hookean, displacement formulation, with
// the YAML loads removed so each test installs its own.
cmf::AppConfig CookConfig()
{
  cmf::AppConfig cfg = cmf::LoadConfig("apps/input/cook.yaml");
  cfg.output.paraview.clear();
  cfg.output.fields = {"displacement"};
  cfg.output.probes.clear();
  cfg.solver.newton.print_level = 0;
  cfg.solver.newton.rtol = 1e-10;
  cfg.solver.newton.atol = 1e-12;
  cfg.solver.linear.rtol = 1e-13;
  cfg.bcs.traction.clear();
  cfg.body_force = cmf::BodyForceConfig();
  return cfg;
}

struct Run
{
  cmf::QuasiStaticReport report;
  mfem::Vector u;
  double u_norm = 0.0;
  std::vector<double> corner;
};

Run SolveCook(const cmf::AppConfig &cfg, mfem::ParMesh &mesh,
              const std::function<void(cmf::SolidProblem &)> &install)
{
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(mesh, cfg);
  install(*problem);
  problem->Finalize();
  std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
  Run r;
  r.u.SetSize(problem->Height());
  r.u = 0.0;
  r.report = cmf::SolveQuasiStatic(*problem, *linear, cfg.solver, r.u);
  problem->UpdateFields(r.u);
  r.u_norm = std::sqrt(mfem::InnerProduct(MPI_COMM_WORLD, r.u, r.u));
  r.corner = cmf::ProbeVector(problem->Displacement(), {48.0, 60.0});
  return r;
}

// Staged loading: a traction ramped over [0, 0.5] and held, a second one
// ramped over [0.5, 1]; the final state equals the single-stage solve with
// both (path independence of hyperelasticity), and the intermediate state
// at t = 0.5 equals the first traction alone.
void StagedLoadingTest()
{
  cmf::AppConfig cfg = CookConfig();
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  mfem::Vector t1(2), t2(2), tsum(2);
  t1(0) = 0.0; t1(1) = 2.0;
  t2(0) = 0.0; t2(1) = 1.75;
  tsum = t1; tsum += t2;
  mfem::VectorConstantCoefficient c1(t1), c2(t2), csum(tsum);

  cfg.solver.load_steps = 1;
  const Run single = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p) { p.AddTraction({2}, csum); });
  const Run first = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p) { p.AddTraction({2}, c1); });

  cfg.solver.load_steps = 4;
  std::vector<mfem::Vector> states;
  cmf::BCOptions early, late;
  early.schedule = cmf::Schedule::Ramp(0.0, 0.5);
  late.schedule = cmf::Schedule::Ramp(0.5, 1.0);
  Run staged;
  {
    std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
    problem->AddTraction({2}, c1, early);
    problem->AddTraction({2}, c2, late);
    problem->Finalize();
    std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
    staged.u.SetSize(problem->Height());
    staged.u = 0.0;
    staged.report = cmf::SolveQuasiStatic(*problem, *linear, cfg.solver, staged.u,
      [&](const cmf::LoadStepReport &, const mfem::Vector &x) { states.push_back(x); });
    // Scheduled external load: the held first traction plus half of the second at t = 0.75.
    problem->SetLoadFactor(0.75);
    mfem::Vector want(problem->Loads().ExternalLoad().Size());
    {
      std::unique_ptr<cmf::SolidProblem> ref = cmf::MakeSolidProblem(*mesh, cfg);
      mfem::Vector t75(2);
      t75 = t1; t75.Add(0.5, t2);
      mfem::VectorConstantCoefficient c75(t75);
      ref->AddTraction({2}, c75);
      ref->Finalize();
      ref->SetLoadFactor(1.0);
      want = ref->Loads().ExternalLoad();
    }
    want -= problem->Loads().ExternalLoad();
    CHECK_MSG(want.Normlinf() <= 1e-14, "scheduled load at t = 0.75 is T1 + T2/2");
  }
  CHECK_MSG(staged.report.converged && staged.report.steps.size() == 4, "staged run converged in 4 steps");
  CHECK(states.size() == 4);
  auto diff = [](const mfem::Vector &a, const mfem::Vector &b)
  {
    mfem::Vector d(a);
    d -= b;
    return d.Normlinf() / std::max(1e-300, a.Normlinf());
  };
  const double d_final = diff(single.u, staged.u);
  const double d_half = states.size() >= 2 ? diff(first.u, states[1]) : 1.0;
  std::printf("  staged loading: |u_single - u_staged| / |u| = %.3e, at t = 0.5 vs first alone %.3e\n",
              d_final, d_half);
  CHECK_MSG(d_final <= 1e-9, "staged final state matches the single-stage solve");
  CHECK_MSG(d_half <= 1e-9, "state at t = 0.5 matches the first traction alone");
}

// Load-unload through a table schedule: the state returns to zero.
void LoadUnloadTest()
{
  cmf::AppConfig cfg = CookConfig();
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  mfem::Vector t(2);
  t(0) = 0.0; t(1) = 3.75;
  mfem::VectorConstantCoefficient c(t);
  cmf::BCOptions opt;
  opt.schedule = cmf::Schedule::Table({0.0, 0.5, 1.0}, {0.0, 1.0, 0.0});
  cfg.solver.load_steps = 4;
  double u_peak = 0.0;
  Run r;
  {
    std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
    problem->AddTraction({2}, c, opt);
    problem->Finalize();
    std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
    r.u.SetSize(problem->Height());
    r.u = 0.0;
    r.report = cmf::SolveQuasiStatic(*problem, *linear, cfg.solver, r.u,
      [&](const cmf::LoadStepReport &s, const mfem::Vector &x)
      {
        if (s.step == 2) { u_peak = x.Normlinf(); }
      });
  }
  std::printf("  load-unload: |u| at peak %.4e, at the end %.3e\n", u_peak, r.u.Normlinf());
  CHECK_MSG(r.report.converged, "load-unload converged");
  CHECK_MSG(u_peak > 1.0, "the membrane deformed at the peak");
  CHECK_MSG(r.u.Normlinf() <= 1e-9 * u_peak, "the state returns to zero after unloading");
}

// Bisection: with a Newton iteration cap the single planned step fails and
// the stepper recovers by halving; without substepping it reports failure.
void BisectionTest()
{
  cmf::AppConfig cfg = CookConfig();
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  mfem::Vector t(2);
  t(0) = 0.0; t(1) = 3.75;
  mfem::VectorConstantCoefficient c(t);
  cfg.solver.load_steps = 1;
  const Run reference = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p) { p.AddTraction({2}, c); });
  const int full_its = reference.report.steps.back().newton.iterations;
  cfg.solver.newton.max_it = std::max(2, full_its - 2);
  const Run failed = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p) { p.AddTraction({2}, c); });
  CHECK_MSG(!failed.report.converged, "capped Newton fails on the single step");
  cfg.solver.substep.on_failure = true;
  cfg.solver.substep.max_bisections = 4;
  cfg.solver.substep.min_dt = 1e-3;
  const Run bisected = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p) { p.AddTraction({2}, c); });
  std::printf("  bisection: full-step newton its %d, cap %d; recovered in %zu increments with %d bisections\n",
              full_its, cfg.solver.newton.max_it, bisected.report.steps.size(), bisected.report.bisections);
  CHECK_MSG(bisected.report.converged, "bisected run converged");
  CHECK_MSG(bisected.report.bisections >= 1, "at least one bisection happened");
  CHECK_MSG(bisected.report.steps.size() >= 2, "more than one accepted increment");
  if (!bisected.report.steps.empty())
  {
    CHECK_CLOSE(bisected.report.steps.back().load_factor, 1.0, 1e-14);
    CHECK_MSG(bisected.report.steps.front().attempts >= 2, "the first accepted increment records the failed attempts");
  }
  mfem::Vector d(reference.u);
  d -= bisected.u;
  const double rel = d.Normlinf() / reference.u.Normlinf();
  std::printf("  bisection: |u_ref - u_bisected| / |u| = %.3e\n", rel);
  CHECK_MSG(rel <= 1e-9, "bisected path reaches the same state");
}

// ---------------------------------------------------------------- L2: f(x, y, z, t)

// A programmatic time-dependent coefficient with the constant schedule
// reproduces the ramp of a constant coefficient: same external load and
// same prescribed displacement at t = 0.7.
void TimeDependentCoefficientTest()
{
  cmf::AppConfig cfg = CookConfig();
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  mfem::Vector t0(2), u0(2);
  t0(0) = 0.3; t0(1) = 2.0;
  u0(0) = 0.1; u0(1) = -0.2;
  mfem::VectorConstantCoefficient traction(t0), ubar(u0);
  mfem::VectorFunctionCoefficient traction_t(2, [&](const mfem::Vector &, double t, mfem::Vector &v)
  {
    v = t0; v *= t;
  });
  mfem::VectorFunctionCoefficient ubar_t(2, [&](const mfem::Vector &, double t, mfem::Vector &v)
  {
    v = u0; v *= t;
  });
  cmf::BCOptions constant;
  constant.schedule = cmf::Schedule::Constant();
  constant.time_dependent = true;

  std::unique_ptr<cmf::SolidProblem> ramp = cmf::MakeSolidProblem(*mesh, cfg);
  ramp->AddTraction({2}, traction);
  ramp->AddDirichlet({3}, ubar);
  ramp->Finalize();
  ramp->SetLoadFactor(0.7);
  std::unique_ptr<cmf::SolidProblem> timed = cmf::MakeSolidProblem(*mesh, cfg);
  timed->AddTraction({2}, traction_t, constant);
  timed->AddDirichlet({3}, ubar_t, constant);
  timed->Finalize();
  timed->SetLoadFactor(0.7);

  mfem::Vector dL(ramp->Loads().ExternalLoad());
  dL -= timed->Loads().ExternalLoad();
  mfem::Vector xa(ramp->Height()), xb(timed->Height());
  xa = 0.0; xb = 0.0;
  ramp->ApplyDirichlet(xa);
  timed->ApplyDirichlet(xb);
  xa -= xb;
  std::printf("  time-dependent coefficient at t = 0.7: |dL| = %.3e, |du_bar| = %.3e\n",
              dL.Normlinf(), xa.Normlinf());
  CHECK_MSG(dL.Normlinf() <= 1e-14 * ramp->Loads().ExternalLoad().Normlinf(),
            "t-dependent traction with constant schedule = ramp of the constant traction");
  CHECK_MSG(xa.Normlinf() <= 1e-15, "t-dependent Dirichlet data with constant schedule = ramp");
  // The load is reassembled when t changes.
  timed->SetLoadFactor(0.35);
  mfem::Vector half(timed->Loads().ExternalLoad());
  half *= 2.0;
  half -= ramp->Loads().ExternalLoad();
  CHECK_MSG(half.Normlinf() <= 1e-14 * ramp->Loads().ExternalLoad().Normlinf(),
            "time-dependent load reassembled at t = 0.35");
}

// The manufactured problem of tests/input/mms_expression.yaml, from string
// to residual: expression Dirichlet data and body force, p = 2 convergence.
void ExpressionMMSTest()
{
  cmf::AppConfig cfg = cmf::LoadConfig("tests/input/mms_expression.yaml");
  mfem::VectorFunctionCoefficient exact(2, [](const mfem::Vector &X, mfem::Vector &u)
  {
    u.SetSize(2);
    u(0) = 0.05 * std::sin(M_PI * X(0)) * std::sin(M_PI * X(1));
    u(1) = 0.05 * X(0) * X(0) * X(1) * (1.0 - X(1));
  });
  std::vector<double> errors;
  for (int refine = 1; refine <= 3; refine++)
  {
    cfg.mesh.serial_refine = refine;
    std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
    problem->Finalize();
    std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
    mfem::Vector u(problem->Height());
    u = 0.0;
    const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(*problem, *linear, cfg.solver, u);
    CHECK_MSG(report.converged, "expression MMS converged at refine " + std::to_string(refine));
    problem->UpdateFields(u);
    errors.push_back(problem->Displacement().ComputeL2Error(exact));
    std::printf("  expression MMS refine %d: L2 error %.3e, newton its %d\n", refine,
                errors.back(), report.steps.back().newton.iterations);
  }
  for (std::size_t i = 1; i < errors.size(); i++)
  {
    const double rate = std::log(errors[i - 1] / errors[i]) / std::log(2.0);
    std::printf("  expression MMS rate %zu: %.3f\n", i, rate);
    CHECK_MSG(rate >= 2.8, "p = 2 MMS rate " + std::to_string(rate) + " >= 2.8");
  }
}

// A traction expression linear in t with 4 steps equals value + ramp; one
// that does not mention t is assembled once.
void ExpressionReassemblyTest()
{
  cmf::AppConfig cfg = CookConfig();
  cfg.solver.load_steps = 4;
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  cmf::BoundaryCondition ramp_bc, expr_bc, fixed_bc;
  ramp_bc.value = {0.0, 3.75};
  expr_bc.expression = {"0", "3.75*t*(1 + 0.2*sin(pi*y/16))"};
  expr_bc.schedule = cmf::Schedule::Constant();
  fixed_bc.expression = {"0", "3.75*(1 + 0.2*sin(pi*y/16))"};
  std::unique_ptr<mfem::VectorCoefficient> c_ramp = cmf::MakeBCCoefficient(ramp_bc, 2, "ramp");
  std::unique_ptr<mfem::VectorCoefficient> c_expr = cmf::MakeBCCoefficient(expr_bc, 2, "expr");
  std::unique_ptr<mfem::VectorCoefficient> c_fixed = cmf::MakeBCCoefficient(fixed_bc, 2, "fixed");
  CHECK(cmf::OptionsOf(expr_bc).time_dependent);
  CHECK(!cmf::OptionsOf(fixed_bc).time_dependent);

  const Run a = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p) { p.AddTraction({2}, *c_fixed); });
  const Run b = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p)
  {
    p.AddTraction({2}, *c_expr, cmf::OptionsOf(expr_bc));
  });
  mfem::Vector d(a.u);
  d -= b.u;
  const double rel = d.Normlinf() / a.u.Normlinf();
  std::printf("  expression traction linear in t vs ramp of the fixed expression: %.3e\n", rel);
  CHECK_MSG(a.report.converged && b.report.converged, "expression traction runs converged");
  CHECK_MSG(rel <= 1e-12, "expression in t reproduces the ramp");
  // The spatial variation did something: differs from the uniform traction.
  const Run u = SolveCook(cfg, *mesh, [&](cmf::SolidProblem &p) { p.AddTraction({2}, *c_ramp); });
  d = a.u;
  d -= u.u;
  CHECK_MSG(d.Normlinf() / a.u.Normlinf() > 1e-3, "the y-dependent traction differs from the uniform one");
}

} // namespace

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  StagedLoadingTest();
  LoadUnloadTest();
  BisectionTest();
  TimeDependentCoefficientTest();
  ExpressionMMSTest();
  ExpressionReassemblyTest();
  return cmf_test::Report("test_loading");
}

// Verification of the scalar transport physics driven by the inputs of
// apps/input/scalar_transport, the convection-diffusion cases of the
// myapps/convection_diffusion drivers:
//   (a) the cross-check against the drivers' error histories
//       (apps/input/scalar_transport/reference/myapps_*.csv): on the same
//       triangulation, order, steps and quadrature the discrete problems
//       coincide, so the errors agree to the drivers' solver tolerance;
//       cases 2, 3 and 5 with the drivers' source rule of order 2p, case 4
//       with its rule of order 2p + 2 and the series initial condition;
//   (b) the framework's own checks: first order in dt of the Peclet case
//       and of the transient manufactured solution, second order in h of the
//       latter with dt proportional to h^2, the L2 and H1 rates of the steady
//       square case under uniform refinement, the rates of the disk case on
//       the curved meshes and on the refined polygon (whose boundary data
//       is the exact solution, so the polygon has no geometric error),
//       first order in dt of the Kirchhoff case against its series with few
//       Newton iterations per step.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "base/mesh_input.hpp"
#include "base/scalar_config.hpp"
#include "mfem.hpp"
#include "physics/scalar_transport.hpp"
#include "solvers/quasi_static.hpp"
#include "test_util.hpp"

namespace
{

const std::string kDir = "apps/input/scalar_transport/";
const std::string kRef = kDir + "reference/";

// Rows of a CSV by column name.
using Table = std::vector<std::map<std::string, double>>;

Table ReadCsv(const std::string &path)
{
  std::ifstream in(path);
  if (!in) { MFEM_ABORT("cannot read " << path); }
  std::string line;
  std::getline(in, line);
  std::vector<std::string> names;
  {
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) { names.push_back(cell); }
  }
  Table table;
  while (std::getline(in, line))
  {
    if (line.empty()) { continue; }
    std::stringstream ss(line);
    std::string cell;
    std::map<std::string, double> row;
    for (std::size_t i = 0; std::getline(ss, cell, ','); i++)
    {
      if (i < names.size()) { row[names[i]] = std::stod(cell); }
    }
    table.push_back(row);
  }
  return table;
}

struct StepErrors
{
  int step = 0;
  double t = 0.0;
  int newton = 0;
  cmf::ScalarErrors e;
};

struct Overrides
{
  int quadrature_order = -1;
  int serial_refine = -1;
  int time_steps = 0;             // equal steps to t_final in place of the input's
  std::string mesh_file;
  int order = 0;
  mfem::Coefficient *initial = nullptr;
  mfem::Coefficient *exact = nullptr;
  double newton_rtol = 0.0;
};

struct Solved
{
  std::unique_ptr<mfem::ParMesh> mesh;
  std::unique_ptr<cmf::ScalarTransport> problem;
  mfem::Vector x;
  cmf::QuasiStaticReport report;
  std::vector<StepErrors> history;  // per accepted step (transient), or the final state (steady)
  double t_final = 0.0;
  const cmf::ScalarErrors &Final() const { return history.back().e; }
};

std::unique_ptr<Solved> Solve(cmf::ScalarAppConfig cfg, const Overrides &o = Overrides())
{
  cfg.output.paraview.clear();
  cfg.solver.newton.print_level = 0;
  if (o.quadrature_order >= 0) { cfg.transport.quadrature_order = o.quadrature_order; }
  if (o.serial_refine >= 0) { cfg.mesh.serial_refine = o.serial_refine; }
  if (!o.mesh_file.empty()) { cfg.mesh.file = o.mesh_file; }
  if (o.order > 0) { cfg.mesh.order = o.order; }
  if (o.time_steps > 0) { cfg.time.breakpoints = cmf::UniformTimeSteps(cfg.time.t_final, o.time_steps); }
  if (o.newton_rtol > 0.0) { cfg.solver.newton.rtol = o.newton_rtol; }
  auto s = std::make_unique<Solved>();
  s->mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  s->problem = std::make_unique<cmf::ScalarTransport>(*s->mesh, cfg);
  cmf::ScalarTransport &problem = *s->problem;
  if (o.initial) { problem.SetInitialCondition(*o.initial); }
  if (o.exact) { problem.SetExact(*o.exact); }
  problem.Finalize();
  problem.SetPhysicalTime(cfg.time.enabled);
  std::unique_ptr<mfem::Solver> linear = problem.MakeLinearSolver(cfg.solver.linear);
  s->x.SetSize(problem.Height());
  problem.InitialState(s->x);
  const cmf::LoadStepCallback on_step = [&](const cmf::LoadStepReport &step, const mfem::Vector &x)
  {
    if (!cfg.time.enabled) { return; }
    StepErrors se;
    se.step = step.step;
    se.t = step.load_factor;
    se.newton = step.newton.iterations;
    if (problem.HasExact()) { se.e = problem.Errors(x, step.load_factor); }
    s->history.push_back(se);
  };
  s->report = cfg.time.enabled
                ? cmf::SolveInTime(problem, *linear, cfg.solver, cfg.time.breakpoints, 0.0, s->x, on_step)
                : cmf::SolveQuasiStatic(problem, *linear, cfg.solver, s->x, on_step);
  s->t_final = cfg.time.enabled ? cfg.time.t_final : 1.0;
  if (!cfg.time.enabled)
  {
    StepErrors se;
    se.step = 1;
    se.t = 1.0;
    se.newton = s->report.steps.back().newton.iterations;
    if (problem.HasExact()) { se.e = problem.Errors(s->x, 1.0); }
    s->history.push_back(se);
  }
  return s;
}

double Rel(double got, double want) { return std::abs(got - want) / std::max(std::abs(want), 1e-300); }

double Rate(double coarse, double fine) { return std::log(coarse / fine) / std::log(2.0); }

// The largest relative deviation of the framework's per-step errors from the
// reference rows (abs and, when named, rel), over the steps > 0.
double WorstDeviation(const Solved &s, const Table &ref, const std::string &abs_col, const std::string &rel_col)
{
  std::map<int, const std::map<std::string, double> *> by_step;
  for (const auto &row : ref)
  {
    const int step = row.count("step") ? int(row.at("step")) : 1;
    by_step[step] = &row;
  }
  double worst = 0.0;
  int compared = 0;
  for (const StepErrors &se : s.history)
  {
    const auto it = by_step.find(se.step);
    if (it == by_step.end() || se.step == 0) { continue; }
    const double want = it->second->at(abs_col);
    if (want < 1e-14) { continue; }
    worst = std::max(worst, Rel(se.e.l2, want));
    if (!rel_col.empty()) { worst = std::max(worst, Rel(se.e.rel_l2, it->second->at(rel_col))); }
    compared++;
  }
  CHECK_MSG(compared > 0, "reference rows compared");
  return worst;
}

// ------------------------------------------------------------- the Kirchhoff series

struct KirchhoffSeries
{
  double a0 = 10.0, m0 = 4.0e6, kappa1 = 10.0, kappa2 = 100.0, T0 = 300.0, T1 = 300.0, T2 = 1300.0,
         qbar = 7.5e5, L = 0.01;
  int terms = 1000;
  double Exact(double x, double t) const
  {
    const double alpha = a0 / m0;
    const double decay = M_PI * M_PI * alpha * t / (L * L);
    double S1 = 0.0;
    for (int n = 1; n <= terms; n++)
    {
      S1 += std::exp(-double(n) * n * decay) * std::cos(n * M_PI * x / L) / (double(n) * n);
    }
    const double f = alpha * t / (L * L) + 1.0 / 3.0 - x / L + 0.5 * x * x / (L * L) - 2.0 / (M_PI * M_PI) * S1;
    const double theta0 = (T0 - T1) + (kappa2 - kappa1) / (T2 - T1) / (2.0 * kappa1) * (T0 - T1) * (T0 - T1);
    const double theta = f * qbar * L / kappa1 + theta0;
    const double gamma = 2.0 * (kappa2 - kappa1) / ((T2 - T1) * kappa1);
    const double sq = std::sqrt(std::max(1e-14, 1.0 + gamma * theta));
    return T1 + (T2 - T1) * (kappa1 / (kappa2 - kappa1)) * (-1.0 + sq);
  }
};

// ------------------------------------------------------------- (a) the cross-checks

void PecletCrossCheck(std::map<std::string, std::unique_ptr<Solved>> &runs)
{
  std::printf("case 1: transient convection-diffusion against the driver's error history\n");
  const Table ref = ReadCsv(kRef + "myapps_convection_diffusion_peclet.csv");
  const char *names[3] = {"1", "10", "100"};
  for (int k = 0; k < 3; k++)
  {
    const std::string pe = names[k];
    cmf::ScalarAppConfig cfg = cmf::LoadScalarConfig(kDir + "convection_diffusion_peclet_" + pe + ".yaml");
    std::unique_ptr<Solved> s = Solve(cfg);
    CHECK_MSG(s->report.converged, "Pe = " + pe + " converged");
    const std::string suffix = "_pe" + std::to_string(k + 1);
    const double worst = WorstDeviation(*s, ref, "abs_l2" + suffix, "rel_l2" + suffix);
    std::printf("  Pe = %-3s: L2 error at t = 1 %.6e (driver %.6e), relative error %.3e; worst deviation over the "
                "steps %.2e\n", pe.c_str(), s->Final().l2, ref.back().at("abs_l2" + suffix), s->Final().rel_l2, worst);
    CHECK_MSG(worst <= 1e-5, "Pe = " + pe + ": error history agrees with the driver's to 1e-5, got " + std::to_string(worst));
    runs["peclet_" + pe] = std::move(s);
  }
}

void SteadyCrossCheck(const std::string &input, const std::string &reference, const std::string &what)
{
  std::printf("%s against the driver's error\n", what.c_str());
  const Table ref = ReadCsv(kRef + reference);
  cmf::ScalarAppConfig cfg = cmf::LoadScalarConfig(kDir + input);
  Overrides o;
  o.quadrature_order = 2 * cfg.mesh.order; // the driver's DomainLFIntegrator rule
  std::unique_ptr<Solved> s = Solve(cfg, o);
  CHECK_MSG(s->report.converged, what + " converged");
  CHECK_MSG(s->report.steps.back().newton.iterations == 1, what + ": one Newton iteration");
  const double worst = WorstDeviation(*s, ref, "abs_l2", "rel_l2");
  std::printf("  L2 error %.6e (driver %.6e), relative %.6e (driver %.6e); deviation %.2e\n", s->Final().l2,
              ref[0].at("abs_l2"), s->Final().rel_l2, ref[0].at("rel_l2"), worst);
  CHECK_MSG(worst <= 1e-5, what + ": agrees with the driver to 1e-5, got " + std::to_string(worst));
  // With the framework's rule the difference is the quadrature of the source (reported).
  std::unique_ptr<Solved> own = Solve(cfg);
  std::printf("  with the framework's rule 2p + 3: L2 error %.6e (differs by %.2e)\n", own->Final().l2,
              Rel(own->Final().l2, s->Final().l2));
}

void KirchhoffCrossCheck()
{
  std::printf("case 4: nonlinear diffusion against the driver's error history\n");
  const Table ref = ReadCsv(kRef + "myapps_nonlinear_diffusion_kirchhoff.csv");
  const KirchhoffSeries ks;
  mfem::FunctionCoefficient exact([&](const mfem::Vector &X, double t) { return ks.Exact(X(0), t); });
  cmf::ScalarAppConfig cfg = cmf::LoadScalarConfig(kDir + "nonlinear_diffusion_kirchhoff.yaml");
  Overrides o;
  o.quadrature_order = 2 * cfg.mesh.order + 2; // the driver's hand-written integrators
  o.initial = &exact;                          // the series at t = 0, as the driver projects it
  o.exact = &exact;
  o.newton_rtol = 1e-12;
  std::unique_ptr<Solved> s = Solve(cfg, o);
  CHECK_MSG(s->report.converged, "Kirchhoff converged");
  const double worst = WorstDeviation(*s, ref, "abs_l2", "rel_l2");
  std::printf("  per step: t, L2 error (driver), Newton iterations (driver)\n");
  for (const StepErrors &se : s->history)
  {
    const auto &row = ref[std::size_t(se.step)];
    std::printf("    %.1f  %.6e (%.6e)  %d (%d)\n", se.t, se.e.l2, row.at("abs_l2"), se.newton, int(row.at("newton_iters")));
  }
  std::printf("  worst deviation of the error history %.2e\n", worst);
  CHECK_MSG(worst <= 1e-5, "Kirchhoff: error history agrees with the driver's to 1e-5, got " + std::to_string(worst));
  // With the input's own initial condition (300) the early history differs by the truncation of the series.
  std::unique_ptr<Solved> own = Solve(cfg, [&]{ Overrides p; p.exact = &exact; return p; }());
  std::printf("  with initial 300 and the rule 2p + 3: L2 error at t = 1 %.6e (deviation %.2e), at the first step %.2e\n",
              own->Final().l2, Rel(own->Final().l2, s->Final().l2), Rel(own->history[0].e.l2, s->history[0].e.l2));
}

void TransientMMSCrossCheck(std::map<std::string, std::unique_ptr<Solved>> &runs)
{
  std::printf("case 5: transient diffusion against the driver's error history\n");
  const Table ref = ReadCsv(kRef + "myapps_transient_diffusion_mms.csv");
  cmf::ScalarAppConfig cfg = cmf::LoadScalarConfig(kDir + "transient_diffusion_mms.yaml");
  Overrides o;
  o.quadrature_order = 2 * cfg.mesh.order;
  std::unique_ptr<Solved> s = Solve(cfg, o);
  CHECK_MSG(s->report.converged, "transient MMS converged");
  const double worst = WorstDeviation(*s, ref, "l2_error", "");
  double worst_linf = 0.0;
  for (const StepErrors &se : s->history)
  {
    if (se.step == 0) { continue; }
    worst_linf = std::max(worst_linf, Rel(se.e.linf_nodal, ref[std::size_t(se.step)].at("linf_error")));
  }
  std::printf("  L2 error at t = 2 %.6e (driver %.6e), nodal Linf %.6e (driver %.6e); worst deviations %.2e (L2) %.2e (Linf)\n",
              s->Final().l2, ref.back().at("l2_error"), s->Final().linf_nodal, ref.back().at("linf_error"), worst, worst_linf);
  CHECK_MSG(worst <= 1e-5, "transient MMS: L2 history agrees with the driver's to 1e-5, got " + std::to_string(worst));
  CHECK_MSG(worst_linf <= 1e-5, "transient MMS: Linf history agrees with the driver's to 1e-5, got " + std::to_string(worst_linf));
  runs["transient_mms"] = std::move(s);
}

// ------------------------------------------------------------- (b) the framework's own checks

void PecletOrderTest(const std::map<std::string, std::unique_ptr<Solved>> &runs)
{
  std::printf("case 1: first order in dt at Pe = 1 (the temporal error dominates at p = 3)\n");
  cmf::ScalarAppConfig cfg = cmf::LoadScalarConfig(kDir + "convection_diffusion_peclet_1.yaml");
  std::vector<double> errors = {runs.at("peclet_1")->Final().l2};
  std::vector<int> steps = {1000, 500, 250};
  for (std::size_t i = 1; i < steps.size(); i++)
  {
    Overrides o;
    o.time_steps = steps[i];
    std::unique_ptr<Solved> s = Solve(cfg, o);
    CHECK_MSG(s->report.converged, "Pe = 1 converged");
    errors.push_back(s->Final().l2);
  }
  for (std::size_t i = 0; i < steps.size(); i++)
  {
    std::printf("  %4d steps: L2 error at t = 1 %.4e%s\n", steps[i], errors[i],
                i ? (", rate " + std::to_string(Rate(errors[i], errors[i - 1]))).c_str() : "");
    if (i) { CHECK_MSG(Rate(errors[i], errors[i - 1]) >= 0.9, "Pe = 1: first order in dt"); }
  }
  for (const char *pe : {"10", "100"})
  {
    const Solved &s = *runs.at(std::string("peclet_") + pe);
    std::printf("  Pe = %s: relative L2 error at t = 1 %.3e (the front is under-resolved on this mesh at Pe = 100)\n",
                pe, s.Final().rel_l2);
  }
}

void SteadySquareRatesTest()
{
  std::printf("case 2: L2 and H1 rates under uniform refinement of the triangulation (p = 3)\n");
  cmf::ScalarAppConfig cfg = cmf::LoadScalarConfig(kDir + "steady_cdr_square_mms.yaml");
  mfem::VectorFunctionCoefficient grad(2, [](const mfem::Vector &X, mfem::Vector &g)
  {
    g.SetSize(2);
    g(0) = 3.0 * M_PI * std::cos(3.0 * M_PI * X(0)) * std::sin(3.0 * M_PI * X(1));
    g(1) = 3.0 * M_PI * std::sin(3.0 * M_PI * X(0)) * std::cos(3.0 * M_PI * X(1));
  });
  std::vector<double> l2, h1;
  for (int refine = 0; refine <= 2; refine++)
  {
    Overrides o;
    o.serial_refine = refine;
    std::unique_ptr<Solved> s = Solve(cfg, o);
    CHECK_MSG(s->report.converged, "square refine " + std::to_string(refine) + " converged");
    l2.push_back(s->Final().l2);
    std::vector<const mfem::IntegrationRule *> irs(mfem::Geometry::NumGeom, nullptr);
    for (int g = 0; g < mfem::Geometry::NumGeom; g++) { irs[g] = &mfem::IntRules.Get(g, 2 * cfg.mesh.order + 3); }
    h1.push_back(s->problem->Unknown().ComputeGradError(&grad, irs.data()));
    std::printf("  refine %d: L2 %.4e H1 %.4e%s\n", refine, l2.back(), h1.back(),
                refine ? (", rates " + std::to_string(Rate(l2[refine - 1], l2[refine])) + " / " +
                          std::to_string(Rate(h1[refine - 1], h1[refine]))).c_str() : "");
  }
  CHECK_MSG(Rate(l2[1], l2[2]) >= 3.9, "square: L2 rate 4 at p = 3");
  CHECK_MSG(Rate(h1[1], h1[2]) >= 2.9, "square: H1 rate 3 at p = 3");
}

void DiskRatesTest()
{
  std::printf("case 3: rates on the curved disks (generated at lc = 0.1, 0.05, 0.025) and on the refined polygon\n");
  cmf::ScalarAppConfig cfg = cmf::LoadScalarConfig(kDir + "steady_cdr_disk_mms_curved.yaml");
  for (int order = 2; order <= 3; order++)
  {
    std::vector<double> l2;
    for (int level = 1; level <= 3; level++)
    {
      Overrides o;
      o.mesh_file = "apps/mesh/disk_p3_" + std::to_string(level) + ".msh";
      o.order = order;
      std::unique_ptr<Solved> s = Solve(cfg, o);
      CHECK_MSG(s->report.converged, "disk converged");
      l2.push_back(s->Final().l2);
      std::printf("  k=%d lc=%.4f: L2 %.4e%s\n", order, 0.1 / (1 << (level - 1)), l2.back(),
                  level > 1 ? (", rate " + std::to_string(Rate(l2[level - 2], l2[level - 1]))).c_str() : "");
    }
    CHECK_MSG(Rate(l2[1], l2[2]) >= order + 0.9, "disk: L2 rate " + std::to_string(order + 1) + " at k = " + std::to_string(order));
  }
  cmf::ScalarAppConfig straight = cmf::LoadScalarConfig(kDir + "steady_cdr_disk_mms.yaml");
  std::vector<double> plateau;
  for (int refine = 0; refine <= 1; refine++)
  {
    Overrides o;
    o.serial_refine = refine;
    std::unique_ptr<Solved> s = Solve(straight, o);
    plateau.push_back(s->Final().l2);
  }
  // The driver's boundary data is the exact solution projected on the
  // polygon, so the polygon problem has U as its exact solution and no
  // geometric error: rate 4 under uniform refinement as on the curved disks.
  std::printf("  straight polygon refined once: L2 %.4e -> %.4e, rate %.2f (the data is U on the polygon, whose exact "
              "solution is U)\n", plateau[0], plateau[1], Rate(plateau[0], plateau[1]));
  CHECK_MSG(Rate(plateau[0], plateau[1]) >= 3.9, "disk polygon: L2 rate 4 at p = 3 under refinement");
}

void TransientMMSOrderTest(const std::map<std::string, std::unique_ptr<Solved>> &runs)
{
  std::printf("case 5: first order in dt at the fixed mesh (k = 2: the spatial error of the p = 1 mesh, 2e-4, would "
              "contaminate the finest step), second order in h with dt ~ h^2 (p = 1)\n");
  cmf::ScalarAppConfig cfg = cmf::LoadScalarConfig(kDir + "transient_diffusion_mms.yaml");
  std::vector<double> errors;
  for (const int steps : {200, 400, 800})
  {
    Overrides o;
    o.time_steps = steps;
    o.order = 2;
    std::unique_ptr<Solved> s = Solve(cfg, o);
    errors.push_back(s->Final().l2);
  }
  std::printf("  k = 2, dt 0.01, 0.005, 0.0025: L2 at t = 2 %.4e %.4e %.4e, rates %.3f %.3f (the input's run at k = 1: %.4e)\n",
              errors[0], errors[1], errors[2], Rate(errors[0], errors[1]), Rate(errors[1], errors[2]),
              runs.at("transient_mms")->Final().l2);
  CHECK_MSG(Rate(errors[1], errors[2]) >= 0.95, "transient MMS: first order in dt");
  std::vector<double> spatial;
  const int refines[3] = {0, 1, 2}, steps[3] = {50, 200, 800};
  for (int i = 0; i < 3; i++)
  {
    Overrides o;
    o.serial_refine = refines[i];
    o.time_steps = steps[i];
    std::unique_ptr<Solved> s = Solve(cfg, o);
    spatial.push_back(s->Final().l2);
  }
  std::printf("  refine 0, 1, 2 with 50, 200, 800 steps: L2 at t = 2 %.4e %.4e %.4e, rates %.3f %.3f\n", spatial[0],
              spatial[1], spatial[2], Rate(spatial[0], spatial[1]), Rate(spatial[1], spatial[2]));
  CHECK_MSG(Rate(spatial[1], spatial[2]) >= 1.9, "transient MMS: second order in h with dt ~ h^2");
}

void KirchhoffOrderTest()
{
  std::printf("case 4: first order in dt against the series on the input's mesh\n");
  const KirchhoffSeries ks;
  mfem::FunctionCoefficient exact([&](const mfem::Vector &X, double t) { return ks.Exact(X(0), t); });
  cmf::ScalarAppConfig cfg = cmf::LoadScalarConfig(kDir + "nonlinear_diffusion_kirchhoff.yaml");
  std::vector<double> errors;
  for (const int steps : {10, 20, 40})
  {
    Overrides o;
    o.time_steps = steps;
    o.exact = &exact;
    std::unique_ptr<Solved> s = Solve(cfg, o);
    CHECK_MSG(s->report.converged, "Kirchhoff converged");
    int max_newton = 0;
    for (const StepErrors &se : s->history) { max_newton = std::max(max_newton, se.newton); }
    errors.push_back(s->Final().l2);
    std::printf("  dt = %.4f: L2 error at t = 1 %.4e (relative %.3e), Newton iterations per step <= %d%s\n", 1.0 / steps,
                errors.back(), s->Final().rel_l2, max_newton,
                errors.size() > 1 ? (", rate " + std::to_string(Rate(errors[errors.size() - 2], errors.back()))).c_str() : "");
    CHECK_MSG(max_newton <= 4, "Kirchhoff: at most four Newton iterations per step");
    if (errors.size() > 1) { CHECK_MSG(Rate(errors[errors.size() - 2], errors.back()) >= 0.9, "Kirchhoff: first order in dt"); }
  }
}

} // namespace

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  std::map<std::string, std::unique_ptr<Solved>> runs;
  PecletCrossCheck(runs);
  SteadyCrossCheck("steady_cdr_square_mms.yaml", "myapps_steady_cdr_square.csv", "case 2: steady square");
  SteadyCrossCheck("steady_cdr_disk_mms.yaml", "myapps_steady_cdr_disk.csv", "case 3: steady disk");
  KirchhoffCrossCheck();
  TransientMMSCrossCheck(runs);
  PecletOrderTest(runs);
  SteadySquareRatesTest();
  DiskRatesTest();
  TransientMMSOrderTest(runs);
  KirchhoffOrderTest();
  return cmf_test::Report("test_scalar_verification");
}

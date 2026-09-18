// DY3 gate: the dynamic inputs of apps/input/dynamics against their
// references (doc/verification_manual.tex, "Dynamic cases").
//   bar_free_vibration           first axial mode; period elongation of the trapezoidal rule
//   bar_step_load                d'Alembert's wave: peak, mean, front arrival, wall reaction
//   cantilever_vibration         first bending frequency against Euler-Bernoulli
//   mms_dynamic_2d, mms_dynamic_3d   order p + 1 in h at a small dt, order 2 in dt (self-convergence)
//   neo_hookean_block_vibration  self-convergence in dt, energy of the two schemes
// With --write FILE / --check FILE only short runs of the two bar inputs are
// made and their norms written or compared to 1e-12 (np 2 and 4 against serial).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "base/config.hpp"
#include "base/mesh_input.hpp"
#include "base/probes.hpp"
#include "mfem.hpp"
#include "physics/dynamic_solid_problem.hpp"
#include "physics/solid_problem.hpp"
#include "solvers/quasi_static.hpp"
#include "test_util.hpp"

namespace
{

const std::string kDir = "apps/input/dynamics/";

bool Root() { return mfem::Mpi::Root(); }

struct DynamicRun
{
  cmf::AppConfig cfg;
  std::unique_ptr<mfem::ParMesh> mesh;
  std::unique_ptr<cmf::SolidProblem> problem;
  std::unique_ptr<cmf::DynamicSolidProblem> dyn;
  mfem::Vector x;
  bool converged = false;
  double initial_energy = 0.0;

  double Energy() { return dyn->KineticEnergy() + problem->InternalEnergy(x); }
  std::vector<double> Probe(const std::vector<double> &point)
  {
    problem->UpdateFields(x);
    return cmf::ProbeVector(problem->Displacement(), point);
  }
};

using StepFn = std::function<void(double t, DynamicRun &run)>;

// Runs cfg through the library as the app does (no files written); on_step is
// called after every accepted step with run.x the new state.
std::unique_ptr<DynamicRun> Run(const cmf::AppConfig &config, const StepFn &on_step = StepFn(),
                                bool track_work = false)
{
  auto run = std::make_unique<DynamicRun>();
  run->cfg = config;
  cmf::AppConfig &cfg = run->cfg;
  cfg.output.paraview.clear();
  cfg.output.fields = {"displacement"};
  cfg.solver.newton.print_level = 0;
  run->mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  run->problem = cmf::MakeSolidProblem(*run->mesh, cfg);
  run->problem->Finalize();
  run->dyn = cmf::MakeDynamicSolidProblem(*run->problem, cfg);
  run->dyn->TrackExternalWork(track_work);
  run->x.SetSize(run->problem->Height());
  run->x = 0.0;
  run->dyn->Initialize(run->x);
  run->initial_energy = run->Energy();
  std::unique_ptr<mfem::Solver> linear = run->dyn->MakeLinearSolver(cfg.solver.linear);
  DynamicRun *r = run.get();
  const cmf::QuasiStaticReport report = cmf::SolveDynamic(
    *run->dyn, *linear, cfg.solver, cfg.dynamics.breakpoints, 0.0, run->x,
    [&on_step, r](const cmf::LoadStepReport &s, const mfem::Vector &)
    { if (on_step) { on_step(s.load_factor, *r); } });
  run->converged = report.converged;
  return run;
}

void SetSteps(cmf::AppConfig &cfg, double t_final, int n)
{
  cfg.dynamics.t_final = t_final;
  cfg.dynamics.breakpoints = cmf::UniformTimeSteps(t_final, n);
}

// Times at which the sampled y(t) crosses zero upwards (linear interpolation).
std::vector<double> UpwardCrossings(const std::vector<double> &t, const std::vector<double> &y,
                                    double from)
{
  std::vector<double> c;
  for (std::size_t k = 0; k + 1 < t.size(); k++)
  {
    if (t[k] >= from && y[k] < 0.0 && y[k + 1] >= 0.0)
    {
      c.push_back(t[k] + (t[k + 1] - t[k]) * (-y[k]) / (y[k + 1] - y[k]));
    }
  }
  return c;
}

// ---------------------------------------------------------------------------
void BarFreeVibration()
{
  const double A = 0.01, L = 10.0, c = 10.0, w = 0.5 * M_PI * c / L, period = 2.0 * M_PI / w;
  const std::vector<double> tip = {10.0, 0.5, 0.5};

  // The input as it stands: five periods, the tip against A cos(w t).
  cmf::AppConfig cfg = cmf::LoadConfig(kDir + "bar_free_vibration.yaml");
  double worst = 0.0;
  std::unique_ptr<DynamicRun> run = Run(cfg, [&](double t, DynamicRun &r)
  {
    if (r.dyn->Steps() % 10 != 0) { return; }
    worst = std::max(worst, std::abs(r.Probe(tip)[0] - A * std::cos(w * t)));
  });
  const double drift = std::abs(run->Energy() - run->initial_energy) / run->initial_energy;
  if (Root())
  {
    std::printf("  bar, first mode, 2000 steps over five periods: max |u_tip - A cos(w t)| = %.2e A, "
                "energy drift %.1e\n", worst / A, drift);
  }
  CHECK_MSG(run->converged, "bar free vibration: converged");
  CHECK_MSG(worst <= 1e-3 * A, "bar free vibration: tip history to 1e-3 A over five periods");
  CHECK_MSG(drift <= 1e-10, "bar free vibration: the trapezoidal rule conserves the energy");

  // A coarse step: the period grows by (w dt)^2 / 12 (exactly: tan(w~ dt / 2) = w dt / 2).
  const double dt = 0.2;
  SetSteps(cfg, 22.0, 110);
  std::vector<double> ts, us;
  run = Run(cfg, [&](double t, DynamicRun &r) { ts.push_back(t); us.push_back(r.Probe(tip)[0]); });
  const std::vector<double> crossings = UpwardCrossings(ts, us, 0.0);
  CHECK_MSG(crossings.size() >= 5, "bar free vibration: five upward zero crossings");
  if (crossings.size() >= 5)
  {
    const double measured = (crossings[4] - crossings[0]) / 4.0 / period - 1.0;
    const double leading = w * dt * w * dt / 12.0;
    const double exact = 0.5 * w * dt / std::atan(0.5 * w * dt) - 1.0;
    if (Root())
    {
      std::printf("  bar, dt = 0.2 (20 steps per period): period elongation %.5e, (w dt)^2/12 = %.5e, "
                  "discrete dispersion %.5e\n", measured, leading, exact);
    }
    CHECK_MSG(std::abs(measured / leading - 1.0) <= 0.1, "bar free vibration: period elongation (w dt)^2 / 12");
    CHECK_MSG(std::abs(measured / exact - 1.0) <= 0.02, "bar free vibration: the scheme's dispersion relation");
  }
}

// ---------------------------------------------------------------------------
void BarStepLoad()
{
  const double p = 0.25, L = 10.0, E = 250.0, c = 10.0, period = 4.0 * L / c;
  const double u_static = p * L / E;
  struct Result { double peak, mean, arrival, overshoot, reaction_mean, ringing; };
  auto measure = [&](const std::string &scheme, double rho_inf)
  {
    cmf::AppConfig cfg = cmf::LoadConfig(kDir + "bar_step_load.yaml");
    cfg.dynamics.scheme = scheme;
    cfg.dynamics.rho_inf = rho_inf;
    SetSteps(cfg, period, 800); // one period at the input's step
    std::vector<double> ts{0.0}, tip{0.0}, mid{0.0}, wall{0.0};
    std::unique_ptr<DynamicRun> run = Run(cfg, [&](double t, DynamicRun &r)
    {
      ts.push_back(t);
      tip.push_back(r.Probe({10.0, 0.5, 0.5})[0]);
      mid.push_back(r.Probe({5.0, 0.5, 0.5})[0]);
      wall.push_back(r.dyn->Reactions()[0].force[0]);
    });
    CHECK_MSG(run->converged, scheme + ": converged");
    Result res{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int late = 0;
    for (std::size_t k = 0; k + 1 < ts.size(); k++)
    {
      res.peak = std::max(res.peak, tip[k + 1]);
      res.mean += 0.5 * (tip[k] + tip[k + 1]) * (ts[k + 1] - ts[k]) / period;
      res.reaction_mean += 0.5 * (wall[k] + wall[k + 1]) * (ts[k + 1] - ts[k]) / period;
      const double over = -wall[k + 1] / (2.0 * p) - 1.0; // the wall carries 2 p A for 1 < t < 3
      res.overshoot = std::max(res.overshoot, over);
      // What is left of the ringing behind the front late in that plateau.
      if (ts[k + 1] > 2.0 && ts[k + 1] < 2.9) { res.ringing += over * over; late++; }
    }
    res.ringing = std::sqrt(res.ringing / late);
    // The front at mid-span: u = (p / (rho c)) (t - t_a) behind it; t_a from a
    // line through the samples between 20 and 80 percent of the first rise.
    const double rise = p * (L / 2.0) / E; // u_mid when the front reaches the wall
    double st = 0.0, su = 0.0, stt = 0.0, stu = 0.0;
    int n = 0;
    for (std::size_t k = 0; k < ts.size() && ts[k] < 1.0; k++)
    {
      if (mid[k] < 0.2 * rise || mid[k] > 0.8 * rise) { continue; }
      st += ts[k]; su += mid[k]; stt += ts[k] * ts[k]; stu += ts[k] * mid[k]; n++;
    }
    const double slope = (n * stu - st * su) / (n * stt - st * st);
    res.arrival = (st - su / slope) / n;
    return res;
  };
  const Result ga = measure("generalized_alpha", 0.8), tr = measure("newmark", 1.0);
  if (Root())
  {
    std::printf("  step load, generalized-alpha 0.8: peak %.4f, mean %.4f of the static tip deflection; front at "
                "mid-span t = %.4f (exact 0.5); wall reaction mean %.4f p A, overshoot %.1f %%, rms ringing for "
                "2 < t < 2.9 %.2f %%\n", ga.peak / u_static, ga.mean / u_static, ga.arrival,
                -ga.reaction_mean / p, 100.0 * ga.overshoot, 100.0 * ga.ringing);
    std::printf("  step load, trapezoidal rule:      peak %.4f, mean %.4f; front t = %.4f; wall reaction mean "
                "%.4f p A, overshoot %.1f %%, rms ringing %.2f %%\n", tr.peak / u_static, tr.mean / u_static,
                tr.arrival, -tr.reaction_mean / p, 100.0 * tr.overshoot, 100.0 * tr.ringing);
  }
  for (const Result &r : {ga, tr})
  {
    CHECK_MSG(std::abs(r.peak / u_static - 2.0) <= 0.04, "step load: the tip peaks at twice the static deflection");
    CHECK_MSG(std::abs(r.mean / u_static - 1.0) <= 0.02, "step load: the tip oscillates about the static deflection");
    CHECK_MSG(std::abs(r.arrival - 0.5) <= 0.01, "step load: the front reaches mid-span at (L/2)/c");
    CHECK_MSG(std::abs(-r.reaction_mean / p - 1.0) <= 0.02, "step load: the wall carries the load on average");
  }
  // The first overshoot behind a front is made by the mesh and is the same
  // for both schemes; what the dissipation removes is the ringing that follows.
  CHECK_MSG(ga.overshoot <= 0.35 && tr.overshoot <= 0.35, "step load: overshoot of the wall reaction");
  CHECK_MSG(ga.ringing <= 0.5 * tr.ringing, "step load: numerical dissipation removes the ringing behind the front");
}

// ---------------------------------------------------------------------------
void CantileverFrequency()
{
  const double E = 250.0, I = 1.0 / 12.0, rhoA = 2.5, L = 10.0;
  const double w_eb = 1.875104 * 1.875104 * std::sqrt(E * I / (rhoA * L * L * L * L));
  cmf::AppConfig cfg = cmf::LoadConfig(kDir + "cantilever_vibration.yaml");
  std::vector<double> ts, uz;
  std::unique_ptr<DynamicRun> run = Run(cfg, [&](double t, DynamicRun &r)
  { ts.push_back(t); uz.push_back(r.Probe({10.0, 0.5, 0.5})[2]); });
  CHECK_MSG(run->converged, "cantilever: converged");
  const std::vector<double> crossings = UpwardCrossings(ts, uz, 30.0); // free vibration after the pulse
  CHECK_MSG(crossings.size() >= 3, "cantilever: three upward zero crossings after the pulse");
  if (crossings.size() >= 3)
  {
    const double measured = 2.0 * M_PI * double(crossings.size() - 1) / (crossings.back() - crossings.front());
    double amplitude = 0.0;
    for (std::size_t k = 0; k < ts.size(); k++) { if (ts[k] > 30.0) { amplitude = std::max(amplitude, std::abs(uz[k])); } }
    if (Root())
    {
      std::printf("  cantilever: w_1 = %.6f from %zu periods, Euler-Bernoulli %.6f (%.2f %%), amplitude %.4f\n",
                  measured, crossings.size() - 1, w_eb, 100.0 * (measured / w_eb - 1.0), amplitude);
    }
    CHECK_MSG(std::abs(measured / w_eb - 1.0) <= 0.02, "cantilever: first frequency within 2 percent of Euler-Bernoulli");
    CHECK_MSG(measured < w_eb, "cantilever: shear deformation and rotary inertia lower the frequency");
  }
}

// ---------------------------------------------------------------------------
double L2Error(DynamicRun &run, mfem::VectorCoefficient &exact, double t)
{
  run.problem->UpdateFields(run.x);
  exact.SetTime(t);
  return run.problem->Displacement().ComputeL2Error(exact);
}

void Manufactured()
{
  // 2D: sin(5 t) 0.05 (sin(pi x) sin(pi y), x^2 y (1 - y)).
  mfem::VectorFunctionCoefficient u2(2, [](const mfem::Vector &X, double t, mfem::Vector &u)
  {
    u.SetSize(2);
    u(0) = 0.05 * std::sin(M_PI * X(0)) * std::sin(M_PI * X(1));
    u(1) = 0.05 * X(0) * X(0) * X(1) * (1.0 - X(1));
    u *= std::sin(5.0 * t);
  });
  cmf::AppConfig cfg = cmf::LoadConfig(kDir + "mms_dynamic_2d.yaml");
  std::vector<double> err;
  for (int level = 1; level <= 3; level++)
  {
    cfg.mesh.serial_refine = level;
    std::unique_ptr<DynamicRun> run = Run(cfg);
    CHECK_MSG(run->converged, "mms 2d: converged");
    err.push_back(L2Error(*run, u2, cfg.dynamics.t_final));
  }
  for (std::size_t k = 0; k + 1 < err.size(); k++)
  {
    const double rate = std::log2(err[k] / err[k + 1]);
    if (Root()) { std::printf("  mms 2d, dt = 1e-3, refine %zu -> %zu: L2 error %.4e -> %.4e, rate %.3f\n", k + 1, k + 2, err[k], err[k + 1], rate); }
    CHECK_MSG(rate >= 2.9, "mms 2d: order 3 in h, got " + std::to_string(rate));
  }
  // Order in dt by self-convergence (the spatial error cancels), on the
  // unrefined mesh and with steps that resolve its every mode: only then are
  // the differences at a fixed time clean powers of dt (the free vibrations
  // that the truncation error excites carry the phase of their numerical frequency).
  cfg.mesh.serial_refine = 0;
  std::vector<mfem::Vector> finals;
  for (int steps : {800, 1600, 3200, 6400})
  {
    SetSteps(cfg, 0.4, steps);
    finals.push_back(Run(cfg)->x);
  }
  for (std::size_t k = 0; k + 2 < finals.size(); k++)
  {
    mfem::Vector d1(finals[k]), d2(finals[k + 1]);
    d1 -= finals[k + 1];
    d2 -= finals[k + 2];
    const double ratio = d1.Normlinf() / d2.Normlinf();
    if (Root()) { std::printf("  mms 2d, refine 0, steps %d -> %d -> %d: self-convergence ratio %.3f\n", 800 << k, 1600 << k, 3200 << k, ratio); }
    CHECK_MSG(ratio >= 3.8 && ratio <= 4.2, "mms 2d: order 2 in dt, ratio " + std::to_string(ratio));
  }

  // 3D: sin(20 t) 0.05 (sin sin sin, x^2 y (1 - y) z, x z (1 - z) cos(pi y / 2)).
  mfem::VectorFunctionCoefficient u3(3, [](const mfem::Vector &X, double t, mfem::Vector &u)
  {
    u.SetSize(3);
    u(0) = 0.05 * std::sin(M_PI * X(0)) * std::sin(M_PI * X(1)) * std::sin(M_PI * X(2));
    u(1) = 0.05 * X(0) * X(0) * X(1) * (1.0 - X(1)) * X(2);
    u(2) = 0.05 * X(0) * X(2) * (1.0 - X(2)) * std::cos(0.5 * M_PI * X(1));
    u *= std::sin(20.0 * t);
  });
  cmf::AppConfig cfg3 = cmf::LoadConfig(kDir + "mms_dynamic_3d.yaml");
  err.clear();
  for (int level = 0; level <= 2; level++)
  {
    cfg3.mesh.serial_refine = level;
    std::unique_ptr<DynamicRun> run = Run(cfg3);
    CHECK_MSG(run->converged, "mms 3d: converged");
    err.push_back(L2Error(*run, u3, cfg3.dynamics.t_final));
  }
  for (std::size_t k = 0; k + 1 < err.size(); k++)
  {
    const double rate = std::log2(err[k] / err[k + 1]);
    if (Root()) { std::printf("  mms 3d, dt = 5e-4, refine %zu -> %zu: L2 error %.4e -> %.4e, rate %.3f\n", k, k + 1, err[k], err[k + 1], rate); }
    CHECK_MSG(rate >= 2.8, "mms 3d: order 3 in h, got " + std::to_string(rate));
  }
}

// ---------------------------------------------------------------------------
void NeoHookeanBlock()
{
  const std::vector<double> corner = {1.0, 1.0};
  cmf::AppConfig cfg = cmf::LoadConfig(kDir + "neo_hookean_block_vibration.yaml");
  const double t_final = 0.16;

  // generalized-alpha, rho_inf = 0.8, at the input's step: the energy.
  {
    SetSteps(cfg, t_final, 160);
    double highest = 0.0;
    std::unique_ptr<DynamicRun> run = Run(cfg, [&](double, DynamicRun &r)
    { highest = std::max(highest, r.Energy() / r.initial_energy); });
    const double end = run->Energy() / run->initial_energy;
    if (Root()) { std::printf("  neo-Hookean block, rho_inf 0.8, dt = 1e-3: max E / E_0 = %.12f, E_end / E_0 = %.6f\n", highest, end); }
    CHECK_MSG(run->converged, "neo-Hookean block: converged");
    CHECK_MSG(highest <= 1.0 + 1e-10, "neo-Hookean block: the energy never exceeds its initial value");
    CHECK_MSG(end < 1.0, "neo-Hookean block: the energy ends below its initial value");
  }
  // Self-convergence of the corner, with steps that resolve the mesh (from
  // dt = 5e-4 down; one level coarser the ratio is 3.65).
  std::vector<std::vector<double>> ends;
  double travel = 0.0;
  for (int steps : {320, 640, 1280, 2560})
  {
    SetSteps(cfg, t_final, steps);
    std::unique_ptr<DynamicRun> run = Run(cfg);
    CHECK_MSG(run->converged, "neo-Hookean block: converged");
    ends.push_back(run->Probe(corner));
    travel = std::max(travel, std::abs(ends.back()[1]));
  }
  CHECK_MSG(travel >= 0.3, "neo-Hookean block: the motion is of large amplitude");
  for (std::size_t k = 0; k + 2 < ends.size(); k++)
  {
    const double d1 = std::hypot(ends[k][0] - ends[k + 1][0], ends[k][1] - ends[k + 1][1]);
    const double d2 = std::hypot(ends[k + 1][0] - ends[k + 2][0], ends[k + 1][1] - ends[k + 2][1]);
    if (Root()) { std::printf("  neo-Hookean block, corner at t = 0.16, steps %d -> %d -> %d: ratio %.3f (corner u_y %.6f)\n", 320 << k, 640 << k, 1280 << k, d1 / d2, ends[k + 2][1]); }
    CHECK_MSG(d1 / d2 >= 3.8 && d1 / d2 <= 4.2, "neo-Hookean block: order 2 in dt, ratio " + std::to_string(d1 / d2));
  }

  // Trapezoidal rule: the energy error of a nonlinear problem falls as dt^2.
  cfg.dynamics.scheme = "newmark";
  std::vector<double> drift;
  for (int steps : {160, 320, 640})
  {
    SetSteps(cfg, t_final, steps);
    double worst = 0.0;
    std::unique_ptr<DynamicRun> run = Run(cfg, [&](double, DynamicRun &r)
    { worst = std::max(worst, std::abs(r.Energy() / r.initial_energy - 1.0)); });
    CHECK_MSG(run->converged, "neo-Hookean block, trapezoidal rule: converged");
    drift.push_back(worst);
  }
  for (std::size_t k = 0; k + 1 < drift.size(); k++)
  {
    if (Root()) { std::printf("  neo-Hookean block, trapezoidal rule, steps %d -> %d: max |E / E_0 - 1| %.3e -> %.3e, ratio %.3f\n", 160 << k, 320 << k, drift[k], drift[k + 1], drift[k] / drift[k + 1]); }
    CHECK_MSG(drift[k] / drift[k + 1] >= 3.5 && drift[k] / drift[k + 1] <= 4.5, "neo-Hookean block: energy error of order dt^2");
  }
}

// ---------------------------------------------------------------------------
// Parallel consistency: 130 steps of the two bar inputs (not 100: at t = 1 the
// free vibration passes through zero, and the tip displacement, the strain
// energy and the wall reaction are then small differences of large numbers).
std::vector<double> ParallelNorms()
{
  std::vector<double> norms;
  for (const std::string input : {"bar_free_vibration.yaml", "bar_step_load.yaml"})
  {
    cmf::AppConfig cfg = cmf::LoadConfig(kDir + input);
    cfg.solver.linear.rtol = 1e-14;
    SetSteps(cfg, 130 * (cfg.dynamics.breakpoints[0]), 130);
    std::unique_ptr<DynamicRun> run = Run(cfg, StepFn(), true);
    CHECK_MSG(run->converged, input + ": converged");
    const std::vector<double> tip = run->Probe({10.0, 0.5, 0.5});
    norms.push_back(tip[0]);
    norms.push_back(run->dyn->KineticEnergy());
    norms.push_back(run->problem->InternalEnergy(run->x));
    norms.push_back(run->dyn->ExternalWork());
    norms.push_back(run->dyn->Reactions()[0].force[0]);
  }
  return norms;
}

} // namespace

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  const char *write_path = "";
  const char *check_path = "";
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&write_path, "-w", "--write", "Write the norms of the parallel consistency runs to this file.");
  args.AddOption(&check_path, "-c", "--check", "Compare the norms of those runs with this file.");
  args.Parse();
  if (!args.Good())
  {
    if (Root()) { args.PrintUsage(std::cout); }
    return 1;
  }
  const std::string write(write_path), check(check_path);
  if (!write.empty() || !check.empty())
  {
    const std::vector<double> norms = ParallelNorms();
    if (!write.empty() && Root())
    {
      std::ofstream out(write);
      out.precision(17);
      for (double v : norms) { out << std::scientific << v << "\n"; }
    }
    if (!check.empty())
    {
      std::ifstream in(check);
      for (std::size_t k = 0; k < norms.size(); k++)
      {
        double ref = 0.0;
        CHECK_MSG(bool(in >> ref), "read reference file " + check);
        // Quantities that vanish identically (no external work in free vibration) compare absolutely.
        const double diff = std::abs(norms[k] - ref), scale = std::max(std::abs(ref), 1e-30);
        if (Root()) { std::printf("  np %d, norm %zu: %.15e, reference %.15e, relative difference %.2e\n", mfem::Mpi::WorldSize(), k, norms[k], ref, diff / scale); }
        CHECK_MSG(diff <= 1e-12 * scale || diff <= 1e-25, "parallel run matches the serial reference to 1e-12");
      }
    }
  }
  else
  {
    if (mfem::Mpi::WorldSize() != 1)
    {
      if (Root()) { std::cout << "test_dynamic_verification: the reference checks are serial (use --write / --check in parallel)" << std::endl; }
      return 1;
    }
    std::cout << "bar, free vibration" << std::endl;
    BarFreeVibration();
    std::cout << "bar, step load" << std::endl;
    BarStepLoad();
    std::cout << "cantilever" << std::endl;
    CantileverFrequency();
    std::cout << "manufactured solutions" << std::endl;
    Manufactured();
    std::cout << "neo-Hookean block" << std::endl;
    NeoHookeanBlock();
  }
  int code = cmf_test::Report(Root() ? "test_dynamic_verification" : "test_dynamic_verification (rank)");
  int global = 0;
  MPI_Allreduce(&code, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return global;
}

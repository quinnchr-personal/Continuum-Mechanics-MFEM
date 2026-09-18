// Verification of small-strain linear elasticity against closed-form
// solutions, driven by the inputs of apps/input/linear_elasticity: Lame's
// thick-walled cylinder and sphere, Kirsch's stress concentration, the
// Euler-Bernoulli cantilever, manufactured solutions in 2D and 3D, a
// two-material cube (Voigt-Reuss bounds), and Cook's membrane in its original
// small-strain form with frozen regression values; in the mixed u-p
// formulation the incompressible Lame cylinder (uniform pressure unknown),
// the locking record at nu = 0.4999 and the incompressible membrane. Every
// displacement solve is one Newton iteration, every mixed one at most two. Where a finite-strain input poses the same problem in its linear
// limit (Kirsch, the cantilever), the two runs are compared.
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "base/coefficients.hpp"
#include "base/expression.hpp"
#include "base/fields.hpp"
#include "base/mesh_input.hpp"
#include "base/probes.hpp"
#include "materials/materials.hpp"
#include "mfem.hpp"
#include "physics/solid_problem.hpp"
#include "solvers/quasi_static.hpp"
#include "test_util.hpp"

namespace
{

const std::string kDir = "apps/input/linear_elasticity/";
const std::string kFiniteDir = "apps/input/finite_elasticity/";

// Frozen regression oracle: Cook's membrane (E = 1, nu = 1/3, unit load, plane
// stress), vertical displacement of the top-right corner (48, 60) and of the
// mid-point of the free edge (48, 52) on the 64x64 p = 2 mesh (serial_refine
// 4). Recorded 2026-09-18, asserted within 1e-8 relative thereafter.
const double kCookLinearCornerFrozen = 2.516395365489e+01;
const double kCookLinearMidFrozen = 2.396503970351e+01;
const int kCookLinearFinestRefine = 4;

struct Solved
{
  std::unique_ptr<mfem::ParMesh> mesh;
  std::unique_ptr<cmf::SolidProblem> problem;
  cmf::FieldRegistry fields;
  mfem::Vector x;
  cmf::QuasiStaticReport report;

  std::vector<double> Probe(const std::string &field, const std::vector<double> &point) const
  {
    return cmf::ProbeVector(fields.Get(field), point);
  }
  int NewtonIterations() const { return report.steps.back().newton.iterations; }
};

// Load, solve and register the fields of an input (quiet Newton).
std::unique_ptr<Solved> Solve(cmf::AppConfig cfg)
{
  cfg.output.paraview.clear();
  cfg.solver.newton.print_level = 0;
  auto s = std::make_unique<Solved>();
  s->mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  s->problem = cmf::MakeSolidProblem(*s->mesh, cfg);
  s->problem->Finalize();
  std::unique_ptr<mfem::Solver> linear = s->problem->MakeLinearSolver(cfg.solver.linear);
  s->x.SetSize(s->problem->Height());
  s->x = 0.0;
  s->report = cmf::SolveQuasiStatic(*s->problem, *linear, cfg.solver, s->x);
  s->problem->UpdateFields(s->x);
  s->problem->RegisterFields(s->fields);
  return s;
}

double Rel(double got, double want) { return std::abs(got - want) / std::max(std::abs(want), 1e-300); }

// A linear problem: converged, in exactly one Newton iteration.
void CheckLinearSolve(const Solved &s, const std::string &what)
{
  CHECK_MSG(s.report.converged, what + " converged");
  CHECK_MSG(s.NewtonIterations() == 1, what + ": one Newton iteration, got " + std::to_string(s.NewtonIterations()));
}

// Radial, tangential (mean of the directions normal to n) and the largest
// shear component of a packed symmetric tensor (xx yy zz xy yz xz) with
// respect to the unit vector n.
struct Polar
{
  double rr = 0.0, tt = 0.0, zz = 0.0, shear = 0.0;
};

Polar PolarComponents2D(const std::vector<double> &c, double x, double y)
{
  const double r = std::hypot(x, y), n0 = x / r, n1 = y / r;
  Polar p;
  p.rr = n0 * n0 * c[0] + n1 * n1 * c[1] + 2.0 * n0 * n1 * c[3];
  p.tt = n1 * n1 * c[0] + n0 * n0 * c[1] - 2.0 * n0 * n1 * c[3];
  p.zz = c[2];
  p.shear = std::abs(n0 * n1 * (c[1] - c[0]) + (n0 * n0 - n1 * n1) * c[3]);
  return p;
}

Polar PolarComponents3D(const std::vector<double> &c, const std::vector<double> &X)
{
  const double r = std::sqrt(X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);
  const double n[3] = {X[0] / r, X[1] / r, X[2] / r};
  const double S[3][3] = {{c[0], c[3], c[5]}, {c[3], c[1], c[4]}, {c[5], c[4], c[2]}};
  double t[3] = {0.0, 0.0, 0.0};
  Polar p;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) { t[i] += S[i][j] * n[j]; }
  for (int i = 0; i < 3; i++) { p.rr += n[i] * t[i]; }
  p.tt = 0.5 * (c[0] + c[1] + c[2] - p.rr); // the two hoop stresses are equal
  double shear2 = 0.0;
  for (int i = 0; i < 3; i++) { shear2 += (t[i] - p.rr * n[i]) * (t[i] - p.rr * n[i]); }
  p.shear = std::sqrt(shear2);
  return p;
}

// ------------------------------------------------------------ Lame's cylinder

void LameCylinderTest()
{
  cmf::AppConfig cfg = cmf::LoadConfig(kDir + "verification/lame_cylinder.yaml");
  const double E = cfg.material.E, nu = cfg.material.nu, a = 1.0, b = 2.0;
  const double p = cmf::Expression::Parse(cfg.bcs.traction.at(0).expression.at(0)).Eval(0, 0, 0, 1);
  const double A = p * a * a / (b * b - a * a);
  auto u_r = [&](double r) { return (1.0 + nu) / E * A * ((1.0 - 2.0 * nu) * r + b * b / r); };
  // Displacements at the input's refinement, stresses and strains over a
  // refinement sequence one level beyond it. The second-order arcs of the
  // coarse mesh are kept under refinement, so the geometry error (about 1e-6)
  // bounds the displacement agreement; the pointwise stresses of the projected
  // field converge with h^2.
  const int input_refine = cfg.mesh.serial_refine, finest = input_refine + 1;
  std::vector<double> stress_err, strain_err;
  for (int refine = 0; refine <= finest; refine++)
  {
    cfg.mesh.serial_refine = refine;
    std::unique_ptr<Solved> s = Solve(cfg);
    CheckLinearSolve(*s, "Lame cylinder refine " + std::to_string(refine));
    const double ua = s->Probe("displacement", {a, 0.0})[0], ub = s->Probe("displacement", {b, 0.0})[0];
    double se = 0.0, ee = 0.0, shear = 0.0, ezz = 0.0, ue = 0.0;
    for (const cmf::ProbeConfig &probe : cfg.output.probes)
    {
      if (probe.name.rfind("wall_", 0) != 0) { continue; }
      const double x = probe.point[0], y = probe.point[1], r = std::hypot(x, y);
      const std::vector<double> u = s->Probe("displacement", probe.point);
      const Polar sig = PolarComponents2D(s->Probe("cauchy_stress", probe.point), x, y);
      const Polar eps = PolarComponents2D(s->Probe("strain", probe.point), x, y);
      const double srr = A * (1.0 - b * b / (r * r)), stt = A * (1.0 + b * b / (r * r)), szz = 2.0 * nu * A;
      const double err = (1.0 + nu) / E * A * ((1.0 - 2.0 * nu) - b * b / (r * r)); // du_r/dr
      se = std::max({se, std::abs(sig.rr - srr), std::abs(sig.tt - stt), std::abs(sig.zz - szz)});
      ee = std::max({ee, std::abs(eps.rr - err), std::abs(eps.tt - u_r(r) / r)});
      shear = std::max(shear, sig.shear);
      ezz = std::max(ezz, std::abs(eps.zz));
      ue = std::max(ue, Rel(std::hypot(u[0], u[1]), u_r(r)));
      if (refine == input_refine)
      {
        std::printf("  Lame cylinder r = %.2f: sigma_rr %.5f (%.5f), sigma_tt %.5f (%.5f), sigma_zz %.5f (%.5f)\n",
                    r, sig.rr, srr, sig.tt, stt, sig.zz, szz);
      }
    }
    stress_err.push_back(se / p);
    strain_err.push_back(ee / (u_r(a) / a));
    std::printf("  Lame cylinder refine %d: u_r(a) rel %.2e, u_r(b) rel %.2e, wall u_r rel %.2e; wall stress "
                "error / p %.3e, strain error / eps_tt(a) %.3e, shear %.1e\n", refine, Rel(ua, u_r(a)),
                Rel(ub, u_r(b)), ue, stress_err.back(), strain_err.back(), shear / p);
    CHECK_MSG(shear <= 1e-8 * p, "Lame cylinder: no r-theta shear on the diagonal, refine " + std::to_string(refine));
    CHECK_MSG(ezz <= 1e-14, "Lame cylinder: plane strain, refine " + std::to_string(refine));
    if (refine == input_refine)
    {
      CHECK_MSG(Rel(ua, u_r(a)) <= 1e-5, "Lame cylinder u_r(a)");
      CHECK_MSG(Rel(ub, u_r(b)) <= 1e-5, "Lame cylinder u_r(b)");
      CHECK_MSG(Rel(s->Probe("displacement", {0.0, a})[1], u_r(a)) <= 1e-5, "Lame cylinder u_r(a) on the y axis");
      CHECK_MSG(ue <= 1e-5, "Lame cylinder u_r in the wall");
      CHECK_MSG(stress_err.back() <= 5e-3, "Lame cylinder wall stresses within 0.5% of p");
    }
  }
  for (std::size_t k = 0; k + 1 < stress_err.size(); k++)
  {
    const double rs = stress_err[k] / stress_err[k + 1], re = strain_err[k] / strain_err[k + 1];
    std::printf("  Lame cylinder error ratios %zu: stress %.2f, strain %.2f\n", k + 1, rs, re);
    CHECK_MSG(rs >= 3.0 && re >= 3.0, "Lame cylinder: pointwise stress and strain converge with h^2");
  }
  CHECK_MSG(stress_err.back() <= 1.5e-3, "Lame cylinder wall stresses within 0.15% of p on the finest mesh");
}

// ------------------------------------- Lame's cylinder, incompressible (mixed)

void LameCylinderIncompressibleTest()
{
  cmf::AppConfig cfg = cmf::LoadConfig(kDir + "verification/lame_cylinder_incompressible.yaml");
  const double E = cfg.material.E, a = 1.0, b = 2.0;
  const double P = cmf::Expression::Parse(cfg.bcs.traction.at(0).expression.at(0)).Eval(0, 0, 0, 1);
  const double A = P * a * a / (b * b - a * a);
  auto u_r = [&](double r, double nu) { return (1.0 + nu) / E * A * ((1.0 - 2.0 * nu) * r + b * b / r); };
  std::unique_ptr<Solved> s = Solve(cfg);
  CHECK_MSG(s->report.converged && s->NewtonIterations() <= 2, "incompressible Lame cylinder: at most two Newton steps");
  const double ua = s->Probe("displacement", {a, 0.0})[0], ub = s->Probe("displacement", {b, 0.0})[0];
  std::printf("  incompressible Lame cylinder: u_r(a) %.10e (exact %.10e, rel %.2e), u_r(b) rel %.2e; "
              "pressure at a, b: %.6f, %.6f (exact %.6f)\n", ua, u_r(a, 0.5), Rel(ua, u_r(a, 0.5)),
              Rel(ub, u_r(b, 0.5)), s->Probe("pressure", {a, 0.0})[0], s->Probe("pressure", {b, 0.0})[0], A);
  CHECK_MSG(Rel(ua, u_r(a, 0.5)) <= 1e-5, "incompressible Lame cylinder u_r(a)");
  CHECK_MSG(Rel(ub, u_r(b, 0.5)) <= 1e-5, "incompressible Lame cylinder u_r(b)");
  // The mean stress of the incompressible field is uniform: the pressure
  // unknown is the constant A (least accurate in the corner elements).
  CHECK_MSG(Rel(s->Probe("pressure", {a, 0.0})[0], A) <= 5e-4, "incompressible Lame cylinder pressure at r = a");
  CHECK_MSG(Rel(s->Probe("pressure", {b, 0.0})[0], A) <= 5e-4, "incompressible Lame cylinder pressure at r = b");
  for (const cmf::ProbeConfig &probe : cfg.output.probes)
  {
    if (probe.name.rfind("wall_", 0) != 0) { continue; }
    const double x = probe.point[0], y = probe.point[1], r = std::hypot(x, y);
    const Polar sig = PolarComponents2D(s->Probe("cauchy_stress", probe.point), x, y);
    const Polar eps = PolarComponents2D(s->Probe("strain", probe.point), x, y);
    const double pr = s->Probe("pressure", probe.point)[0];
    const double srr = A * (1.0 - b * b / (r * r)), stt = A * (1.0 + b * b / (r * r));
    std::printf("  incompressible Lame cylinder r = %.2f: p %.7f (%.7f), sigma_rr %.5f (%.5f), sigma_tt %.5f "
                "(%.5f), sigma_zz %.5f (%.5f), tr(eps) %.1e\n", r, pr, A, sig.rr, srr, sig.tt, stt, sig.zz, A,
                eps.rr + eps.tt + eps.zz);
    CHECK_MSG(Rel(pr, A) <= 1e-5, "incompressible Lame cylinder: uniform pressure A at " + probe.name);
    CHECK_MSG(std::abs(sig.rr - srr) <= 5e-3 * P, "incompressible Lame cylinder sigma_rr at " + probe.name);
    CHECK_MSG(std::abs(sig.tt - stt) <= 5e-3 * P, "incompressible Lame cylinder sigma_tt at " + probe.name);
    CHECK_MSG(std::abs(sig.zz - A) <= 5e-3 * P, "incompressible Lame cylinder sigma_zz = A at " + probe.name);
    // The constraint holds weakly (against the Q1 pressures); pointwise, tr(eps)
    // of the projected strain carries the h^2 error of the strains themselves.
    CHECK_MSG(std::abs(eps.rr + eps.tt + eps.zz) <= 5e-3 * u_r(a, 0.5) / a,
              "incompressible Lame cylinder: tr(eps) small against eps_tt(a) at " + probe.name);
  }

  // Volumetric locking at nu = 0.4999 on the same mesh: the displacement
  // formulation with p = 1 and p = 2 against the mixed one.
  const double nu = 0.4999;
  cmf::AppConfig dcfg = cmf::LoadConfig(kDir + "verification/lame_cylinder.yaml");
  dcfg.output.fields = {"displacement"};
  dcfg.material.nu = nu;
  dcfg.solver.linear.amg = "systems";
  std::vector<double> err;
  for (int order = 1; order <= 2; order++)
  {
    dcfg.mesh.order = order;
    std::unique_ptr<Solved> d = Solve(dcfg);
    CHECK_MSG(d->report.converged, "locking record: displacement formulation converged");
    err.push_back(Rel(d->Probe("displacement", {a, 0.0})[0], u_r(a, nu)));
  }
  cmf::AppConfig mcfg = cfg;
  mcfg.output.fields = {"displacement"};
  mcfg.material.nu = nu;
  std::unique_ptr<Solved> m = Solve(mcfg);
  CHECK_MSG(m->report.converged, "locking record: mixed formulation converged");
  const double mixed_err = Rel(m->Probe("displacement", {a, 0.0})[0], u_r(a, nu));
  std::printf("  locking at nu = 0.4999, error of u_r(a): displacement p=1 %.2e, p=2 %.2e, mixed Q2-Q1 %.2e\n",
              err[0], err[1], mixed_err);
  CHECK_MSG(mixed_err <= 1e-5, "mixed formulation at nu = 0.4999: u_r(a)");
  CHECK_MSG(err[0] >= 100.0 * mixed_err, "the p = 1 displacement formulation locks at nu = 0.4999");
}

// -------------------------------------------------------------- Lame's sphere

void LameSphereTest()
{
  cmf::AppConfig cfg = cmf::LoadConfig(kDir + "verification/lame_sphere.yaml");
  const double E = cfg.material.E, nu = cfg.material.nu, a = 10.0, b = 11.0;
  const double p = cmf::Expression::Parse(cfg.bcs.traction.at(0).expression.at(0)).Eval(0, 0, 0, 1);
  const double A = p * a * a * a / (b * b * b - a * a * a);
  auto u_r = [&](double r) { return A / E * ((1.0 - 2.0 * nu) * r + (1.0 + nu) * b * b * b / (2.0 * r * r)); };
  std::unique_ptr<Solved> s = Solve(cfg);
  CheckLinearSolve(*s, "Lame sphere");
  const double ua = s->Probe("displacement", {a, 0.0, 0.0})[0], ub = s->Probe("displacement", {b, 0.0, 0.0})[0];
  const double uz = s->Probe("displacement", {0.0, 0.0, a})[2];
  std::printf("  Lame sphere: u_r(a) %.8e (exact %.8e, rel %.2e), u_r(b) %.8e (exact %.8e, rel %.2e), "
              "u_r(a) on the z axis rel %.2e\n", ua, u_r(a), Rel(ua, u_r(a)), ub, u_r(b), Rel(ub, u_r(b)),
              Rel(uz, u_r(a)));
  CHECK_MSG(Rel(ua, u_r(a)) <= 3e-4, "Lame sphere u_r(a)");
  CHECK_MSG(Rel(ub, u_r(b)) <= 3e-4, "Lame sphere u_r(b)");
  CHECK_MSG(Rel(uz, u_r(a)) <= 3e-4, "Lame sphere u_r(a) on the z axis");
  for (const cmf::ProbeConfig &probe : cfg.output.probes)
  {
    if (probe.name.rfind("wall_", 0) != 0) { continue; }
    const std::vector<double> &X = probe.point;
    const double r = std::sqrt(X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);
    const Polar sig = PolarComponents3D(s->Probe("cauchy_stress", X), X);
    const double srr = A * (1.0 - b * b * b / (r * r * r)), stt = A * (1.0 + b * b * b / (2.0 * r * r * r));
    std::printf("  Lame sphere r = %.2f: sigma_rr %.5f (%.5f), sigma_tt %.5f (%.5f), shear %.1e\n", r, sig.rr,
                srr, sig.tt, stt, sig.shear);
    // The hoop stress (about 5 p) dominates in this thin shell.
    CHECK_MSG(std::abs(sig.rr - srr) <= 0.01 * stt, "Lame sphere sigma_rr at " + probe.name);
    CHECK_MSG(std::abs(sig.tt - stt) <= 0.01 * stt, "Lame sphere sigma_tt at " + probe.name);
    CHECK_MSG(sig.shear <= 0.01 * stt, "Lame sphere: no shear on the radial direction at " + probe.name);
  }
}

// -------------------------------------------------------------- Kirsch's plate

void KirschTest()
{
  cmf::AppConfig cfg = cmf::LoadConfig(kDir + "verification/kirsch_plate_with_hole.yaml");
  cfg.output.fields = {"displacement", "cauchy_stress"};
  const double s0 = cmf::Expression::Parse(cfg.bcs.traction.at(0).expression.at(0)).Eval(0, 0, 0, 1);
  const double a = 1.0;
  std::unique_ptr<Solved> s = Solve(cfg);
  CheckLinearSolve(*s, "Kirsch plate");
  // The same plate with the compressible neo-Hookean model at a strain of 1e-4.
  cmf::AppConfig fcfg = cmf::LoadConfig(kFiniteDir + "verification/kirsch_plate_with_hole.yaml");
  fcfg.output.fields = {"displacement", "cauchy_stress"};
  const double f0 = cmf::Expression::Parse(fcfg.bcs.traction.at(0).expression.at(0)).Eval(0, 0, 0, 1);
  std::unique_ptr<Solved> f = Solve(fcfg);
  CHECK_MSG(f->report.converged, "Kirsch plate (neo-Hookean) converged");
  for (const cmf::ProbeConfig &probe : cfg.output.probes)
  {
    const std::vector<double> sig = s->Probe("cauchy_stress", probe.point); // xx yy zz xy yz xz
    const std::vector<double> fsig = f->Probe("cauchy_stress", probe.point);
    const double q = a * a / std::pow(std::max(probe.point[0], probe.point[1]), 2), q2 = q * q;
    double sxx = 0.0, syy = 0.0;
    // Probes sit a hair off the symmetry lines; the formula of the nearer axis applies.
    if (probe.point[1] > probe.point[0]) { sxx = s0 * (1.0 + 0.5 * q + 1.5 * q2); syy = s0 * 1.5 * (q - q2); }
    else { sxx = s0 * (1.0 - 2.5 * q + 1.5 * q2); syy = s0 * 0.5 * (q - 3.0 * q2); }
    const double vs_finite = std::max(std::abs(sig[0] / s0 - fsig[0] / f0), std::abs(sig[1] / s0 - fsig[1] / f0));
    std::printf("  Kirsch %s: sigma_xx/s0 %.4f (Kirsch %.4f), sigma_yy/s0 %.4f (%.4f), sigma_xy/s0 %.1e, "
                "sigma_zz/s0 %.1e; vs neo-Hookean at strain 1e-4: %.1e\n", probe.name.c_str(), sig[0] / s0,
                sxx / s0, sig[1] / s0, syy / s0, sig[3] / s0, sig[2] / s0, vs_finite);
    CHECK_MSG(std::abs(sig[0] - sxx) <= 0.02 * 3.0 * s0, "Kirsch sigma_xx at " + probe.name);
    CHECK_MSG(std::abs(sig[1] - syy) <= 0.02 * 3.0 * s0, "Kirsch sigma_yy at " + probe.name);
    CHECK_MSG(std::abs(sig[3]) <= 0.01 * 3.0 * s0, "no shear stress on the symmetry axes at " + probe.name);
    CHECK_MSG(std::abs(sig[2]) <= 1e-10 * s0, "plane stress: sigma_zz = 0 at " + probe.name);
    CHECK_MSG(vs_finite <= 5e-4, "Kirsch: the neo-Hookean run is the linear limit at " + probe.name);
  }
}

// ----------------------------------------------------------------- cantilever

void CantileverTest()
{
  cmf::AppConfig cfg = cmf::LoadConfig(kDir + "verification/euler_bernoulli_cantilever3d.yaml");
  cfg.output.fields = {"displacement"};
  const double P = -cmf::Expression::Parse(cfg.bcs.traction.at(0).expression.at(2)).Eval(0, 0, 0, 1);
  const double E = cfg.material.E, L = 10.0, I = 1.0 / 12.0;
  const double eb = P * L * L * L / (3.0 * E * I);
  std::unique_ptr<Solved> s = Solve(cfg);
  CheckLinearSolve(*s, "cantilever");
  const double tip = -s->Probe("displacement", {10.0, 0.5, 0.5})[2];
  // The finite-strain input: the same beam under a load small enough to be linear.
  cmf::AppConfig fcfg = cmf::LoadConfig(kFiniteDir + "verification/euler_bernoulli_cantilever3d.yaml");
  fcfg.output.fields = {"displacement"};
  const double fP = -cmf::Expression::Parse(fcfg.bcs.traction.at(0).expression.at(2)).Eval(0, 0, 0, 1);
  std::unique_ptr<Solved> f = Solve(fcfg);
  CHECK_MSG(f->report.converged, "cantilever (neo-Hookean) converged");
  const double ftip = -f->Probe("displacement", {10.0, 0.5, 0.5})[2];
  std::printf("  cantilever: tip %.10e (Euler-Bernoulli %.4e, rel %.2e); compliance tip/P %.8e, "
              "neo-Hookean at P = %.0e: %.8e (rel %.1e)\n", tip, eb, Rel(tip, eb), tip / P, fP, ftip / fP,
              Rel(tip / P, ftip / fP));
  CHECK_MSG(Rel(tip, eb) <= 2e-3, "cantilever tip within 0.2% of Euler-Bernoulli");
  // The transverse deflection of a symmetric beam is odd in the load, so the
  // nonlinear correction is second order in |Grad u| = 5e-5.
  CHECK_MSG(Rel(tip / P, ftip / fP) <= 1e-7, "cantilever compliance equals the small-load neo-Hookean one");
  const std::vector<cmf::Reaction> rx = s->problem->Reactions(s->x);
  CHECK_MSG(std::abs(rx.at(0).force[2] - P) <= 1e-9 * P, "cantilever: the clamp carries the load");
  CHECK_MSG(std::abs(rx.at(0).moment[1] + P * L) <= 1e-9 * P * L, "cantilever: clamp moment P L (reference arms)");
}

// --------------------------------------------------- manufactured solutions

void ManufacturedTest(const std::string &input, const std::vector<std::string> &exact_u, int refine_from,
                      int refine_to, double min_rate)
{
  cmf::AppConfig cfg = cmf::LoadConfig(kDir + "verification/manufactured_solutions/" + input);
  cfg.output.fields = {"displacement"};
  cmf::ExpressionVectorCoefficient exact(exact_u);
  std::vector<double> errors;
  for (int refine = refine_from; refine <= refine_to; refine++)
  {
    cfg.mesh.serial_refine = refine;
    std::unique_ptr<Solved> s = Solve(cfg);
    CheckLinearSolve(*s, input + " refine " + std::to_string(refine));
    errors.push_back(s->problem->Displacement().ComputeL2Error(exact));
    std::printf("  %s refine %d: %lld dofs, L2 error %.3e\n", input.c_str(), refine,
                static_cast<long long>(s->problem->GlobalTrueVSize()), errors.back());
  }
  for (std::size_t i = 1; i < errors.size(); i++)
  {
    const double rate = std::log(errors[i - 1] / errors[i]) / std::log(2.0);
    std::printf("  %s rate: %.3f\n", input.c_str(), rate);
    CHECK_MSG(rate >= min_rate, input + " rate " + std::to_string(rate) + " >= " + std::to_string(min_rate));
  }
}

void ManufacturedTests()
{
  const std::vector<std::string> u2 = {"0.05*sin(pi*x)*sin(pi*y)", "0.05*x^2*y*(1 - y)"};
  const std::vector<std::string> u3 = {
    "0.05*sin(pi*x)*sin(pi*y)*sin(pi*z)", "0.05*x^2*y*(1 - y)*z", "0.05*x*z*(1 - z)*cos(pi*y/2)"};
  ManufacturedTest("mms_2d_plane_strain.yaml", u2, 1, 4, 2.9);
  ManufacturedTest("mms_3d_hex.yaml", u3, 1, 3, 2.9);
  ManufacturedTest("mms_3d_tet.yaml", u3, 1, 2, 2.8);
}

// ------------------------------------------------- two materials by regions

void InclusionTest()
{
  cmf::AppConfig cfg = cmf::LoadConfig(kDir + "verification/spherical_inclusion.yaml");
  cfg.output.fields = {"displacement"};
  const double L = 10.0, a = 5.0, delta = 0.1, eps = delta / L, V = L * L * L;
  const double f = M_PI * a * a * a / (6.0 * V);
  auto M_of = [](const cmf::MaterialConfig &m)
  { return m.E * (1.0 - m.nu) / ((1.0 + m.nu) * (1.0 - 2.0 * m.nu)); };
  const double Mm = M_of(cfg.material), Mi = M_of(cfg.material.regions.at(0));
  const double voigt = f * Mi + (1.0 - f) * Mm, reuss = 1.0 / (f / Mi + (1.0 - f) / Mm);
  std::unique_ptr<Solved> s = Solve(cfg);
  CheckLinearSolve(*s, "inclusion");
  double Fz = 0.0;
  for (const cmf::Reaction &rx : s->problem->Reactions(s->x))
  {
    if (rx.name == "loaded") { Fz = rx.force[2]; }
  }
  const double M_reaction = Fz / (L * L * eps);
  const double M_energy = 2.0 * s->problem->InternalEnergy(s->x) / (V * eps * eps);
  std::printf("  inclusion (f = %.4f): M from the reaction %.6f, from the energy %.6f; Reuss %.3f, Voigt %.3f, "
              "matrix %.3f\n", f, M_reaction, M_energy, reuss, voigt, Mm);
  CHECK_MSG(Rel(M_reaction, M_energy) <= 1e-9, "inclusion: F delta = 2 W");
  CHECK_MSG(M_energy > reuss && M_energy < voigt, "inclusion: Reuss < M < Voigt");

  // Regions that repeat the base's parameters are the homogeneous problem:
  // uniaxial strain, exact in the space.
  cfg.material.regions.at(0).E = cfg.material.E;
  std::unique_ptr<Solved> h = Solve(cfg);
  CheckLinearSolve(*h, "inclusion (homogeneous)");
  const double M_homogeneous = 2.0 * h->problem->InternalEnergy(h->x) / (V * eps * eps);
  const std::vector<double> u = h->Probe("displacement", {3.0, 4.0, 5.0});
  std::printf("  inclusion with the base's parameters: M %.10f (exact %.10f), u(3, 4, 5) = (%.1e, %.1e, %.10f)\n",
              M_homogeneous, Mm, u[0], u[1], u[2]);
  CHECK_MSG(Rel(M_homogeneous, Mm) <= 1e-10, "homogeneous regions: M = lambda + 2 mu");
  CHECK_MSG(std::abs(u[2] - eps * 5.0) <= 1e-11 && std::abs(u[0]) + std::abs(u[1]) <= 1e-11,
            "homogeneous regions: uniaxial strain field");
}

// ------------------------------------------------------------ Cook's membrane

void CookLinearTest()
{
  cmf::AppConfig cfg = cmf::LoadConfig(kDir + "cooks_membrane/cook_linear.yaml");
  cfg.output.fields = {"displacement"};
  std::vector<double> corner, mid;
  for (int refine = 0; refine <= kCookLinearFinestRefine; refine++)
  {
    cfg.mesh.serial_refine = refine;
    std::unique_ptr<Solved> s = Solve(cfg);
    CheckLinearSolve(*s, "linear Cook refine " + std::to_string(refine));
    corner.push_back(s->Probe("displacement", {48.0, 60.0})[1]);
    mid.push_back(s->Probe("displacement", {48.0, 52.0})[1]);
    std::printf("  linear cook %2dx%-2d p=2: uy corner (48, 60) %.10f, mid-edge (48, 52) %.10f\n", 4 << refine,
                4 << refine, corner.back(), mid.back());
    if (refine == kCookLinearFinestRefine)
    {
      const std::vector<cmf::Reaction> rx = s->problem->Reactions(s->x);
      // Unit load at X = 48: force -1, moment -48 about the origin with
      // reference arms (the corner moves by u_x = -19 in these units).
      CHECK_MSG(std::abs(rx.at(0).force[1] + 1.0) <= 1e-9, "linear Cook: clamp reaction -1");
      CHECK_MSG(std::abs(rx.at(0).moment[2] + 48.0) <= 1e-7, "linear Cook: clamp moment -48 (reference arms)");
    }
  }
  // The clamped top-left corner (re-entrant for the stress field) limits the
  // point-value rate of uniform refinement, as for the finite-strain membrane.
  for (std::size_t k = 0; k + 2 < corner.size(); k++)
  {
    const double rc = std::abs(corner[k + 1] - corner[k]) / std::abs(corner[k + 2] - corner[k + 1]);
    const double rm = std::abs(mid[k + 1] - mid[k]) / std::abs(mid[k + 2] - mid[k + 1]);
    std::printf("  linear cook successive-difference ratios %zu: corner %.3f, mid-edge %.3f\n", k + 1, rc, rm);
    CHECK_MSG(rc >= 2.0 && rm >= 2.0, "linear Cook successive-difference ratios >= 2");
  }
  const double c = corner.back(), m = mid.back();
  std::printf("  linear cook finest: corner %.12e (frozen %.12e, rel %.2e), mid-edge %.12e (frozen %.12e, "
              "rel %.2e; literature 23.96)\n", c, kCookLinearCornerFrozen, Rel(c, kCookLinearCornerFrozen), m,
              kCookLinearMidFrozen, Rel(m, kCookLinearMidFrozen));
  CHECK_MSG(Rel(m, 23.96) <= 1e-2, "linear Cook mid-edge deflection within 1% of the literature's 23.96");
  CHECK_MSG(Rel(c, kCookLinearCornerFrozen) <= 1e-8, "linear Cook corner: frozen regression value within 1e-8");
  CHECK_MSG(Rel(m, kCookLinearMidFrozen) <= 1e-8, "linear Cook mid-edge: frozen regression value within 1e-8");
}

// ------------------------------------- Cook's membrane, incompressible (mixed)

// Frozen regression oracle: the incompressible plane-strain membrane (E = 1,
// unit load) in the mixed formulation, vertical displacement of the corner
// (48, 60) and of the mid-point of the free edge on the 32x32 Q2-Q1 mesh
// (serial_refine 3). Recorded 2026-09-18, asserted within 1e-8 relative.
const double kCookLinearIncCornerFrozen = 1.941762830954e+01;
const double kCookLinearIncMidFrozen = 1.849738397162e+01;

void CookLinearIncompressibleTest()
{
  cmf::AppConfig cfg = cmf::LoadConfig(kDir + "cooks_membrane/cook_linear_incompressible.yaml");
  cfg.output.fields = {"displacement", "pressure"};
  const int finest = cfg.mesh.serial_refine;
  std::vector<double> corner, mid;
  for (int refine = 0; refine <= finest; refine++)
  {
    cfg.mesh.serial_refine = refine;
    std::unique_ptr<Solved> s = Solve(cfg);
    CHECK_MSG(s->report.converged && s->NewtonIterations() <= 2, "incompressible linear Cook: at most two Newton steps");
    corner.push_back(s->Probe("displacement", {48.0, 60.0})[1]);
    mid.push_back(s->Probe("displacement", {48.0, 52.0})[1]);
    std::printf("  incompressible linear cook %2dx%-2d Q2-Q1: uy corner %.10f, mid-edge %.10f\n", 4 << refine,
                4 << refine, corner.back(), mid.back());
  }
  for (std::size_t k = 0; k + 2 < corner.size(); k++)
  {
    const double rc = std::abs(corner[k + 1] - corner[k]) / std::abs(corner[k + 2] - corner[k + 1]);
    std::printf("  incompressible linear cook successive-difference ratio %zu: corner %.3f\n", k + 1, rc);
    CHECK_MSG(rc >= 2.0, "incompressible linear Cook successive-difference ratio >= 2");
  }
  // The displacement formulation cannot reach nu = 1/2; at nu = 0.4999 with
  // p = 2 on the same mesh it must be close (it locks only mildly there).
  cmf::AppConfig dcfg = cfg;
  dcfg.formulation = "displacement";
  dcfg.material.nu = 0.4999;
  dcfg.output.fields = {"displacement"};
  dcfg.solver.linear.type = "cg_amg";
  dcfg.solver.linear.amg = "systems";
  dcfg.solver.linear.max_it = 2000;
  std::unique_ptr<Solved> d = Solve(dcfg);
  CHECK_MSG(d->report.converged, "linear Cook at nu = 0.4999 (displacement formulation) converged");
  const double dc = d->Probe("displacement", {48.0, 60.0})[1];
  std::printf("  incompressible linear cook finest: corner %.12e (frozen %.12e, rel %.2e), mid-edge %.12e (rel %.2e); "
              "displacement formulation at nu = 0.4999: %.6f (rel %.1e)\n", corner.back(), kCookLinearIncCornerFrozen,
              Rel(corner.back(), kCookLinearIncCornerFrozen), mid.back(), Rel(mid.back(), kCookLinearIncMidFrozen), dc,
              Rel(dc, corner.back()));
  CHECK_MSG(Rel(dc, corner.back()) <= 2e-2, "displacement formulation at nu = 0.4999 approaches the incompressible value");
  CHECK_MSG(Rel(corner.back(), kCookLinearIncCornerFrozen) <= 1e-8, "incompressible linear Cook corner: frozen value");
  CHECK_MSG(Rel(mid.back(), kCookLinearIncMidFrozen) <= 1e-8, "incompressible linear Cook mid-edge: frozen value");
}

} // namespace

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  LameCylinderTest();
  LameCylinderIncompressibleTest();
  LameSphereTest();
  KirschTest();
  CantileverTest();
  ManufacturedTests();
  InclusionTest();
  CookLinearTest();
  CookLinearIncompressibleTest();
  return cmf_test::Report("test_linear_verification");
}

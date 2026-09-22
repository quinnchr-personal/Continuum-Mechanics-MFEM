// Finite thermoelasticity (materials/thermoelastic.hpp, the coupled
// u-p-theta formulation of physics/thermo_solid_mechanics_tl.hpp): the
// material point (stress from the energy, the thermal stress derivative and
// the thermal tangent M against differences, objectivity); free thermal
// expansion to the stress-free volume ratio exp(3 alpha dtheta) for two
// volumetric laws; the adiabatic homogeneous stretch against the material
// point's integration of the heat equation; transient conduction against the
// series solution; the heat-flux entry per current area on a stretched face;
// the assembled 3 x 3 block Jacobian against finite differences in plane
// strain and axisymmetry, with a flux entry and a pin active; point
// constraints and their reactions; and the input schema.
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "base/coefficients.hpp"
#include "base/mesh_input.hpp"
#include "base/probes.hpp"
#include "kernels/thermo_mixed_total_lagrangian.hpp"
#include "materials/materials.hpp"
#include "mfem.hpp"
#include "physics/solid_problem.hpp"
#include "physics/thermo_solid_mechanics_tl.hpp"
#include "solvers/direct_solver.hpp"
#include "solvers/quasi_static.hpp"
#include "test_util.hpp"

using cmf::tensor;

namespace
{

using Mat3 = tensor<double, 3, 3>;
using Vec3 = tensor<double, 3>;
using TNH = cmf::Thermoelastic<cmf::IsoNeoHookean>;

const double kMu = 1.0, kKappa = 50.0, kTheta0 = 300.0, kAlpha = 1e-3, kCv = 1.0, kK = 1.0;

// A box of n^dim elements of order 2, iso_neo_hookean with a thermal block,
// quasi-static in physical time with the direct solver. Boundary attributes:
// 2D box bottom 1, right 2, top 3, left 4; 3D box bottom z=0 1, front y=0 2,
// right x=1 3, back y=1 4, left x=0 5, top z=1 6.
cmf::AppConfig BaseConfig(int dim, int n, const std::string &plane = "strain",
                          const std::string &law = "quadratic")
{
  cmf::AppConfig cfg;
  cfg.formulation = "mixed";
  cfg.plane = plane;
  cfg.mesh.cartesian = true;
  cfg.mesh.box.dim = dim;
  cfg.mesh.box.element = dim == 2 ? "quad" : "hex";
  cfg.mesh.box.nx = cfg.mesh.box.ny = cfg.mesh.box.nz = n;
  cfg.mesh.order = 2;
  cfg.material.model = "iso_neo_hookean";
  cfg.material.mu = kMu;
  cfg.material.kappa = kKappa;
  cfg.material.volumetric = law;
  cfg.material.thermal.set = true;
  cfg.material.thermal.theta0 = kTheta0;
  cfg.material.thermal.alpha = kAlpha;
  cfg.material.thermal.c_v = kCv;
  cfg.material.thermal.k = kK;
  cfg.material.thermal.entropic = true;
  cfg.time.enabled = true;
  cfg.time.t_final = 1.0;
  cfg.solver.linear.type = "direct";
  cfg.solver.newton.rtol = 1e-11;
  cfg.solver.newton.atol = 1e-11;
  cfg.solver.newton.max_it = 30;
  cfg.solver.newton.print_level = 0;
  return cfg;
}

TNH MaterialOf(const cmf::AppConfig &cfg)
{
  return std::get<TNH>(cmf::MakeThermoMaterial(cfg.material));
}

double MaxAbs(const Mat3 &A)
{
  double v = 0.0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) { v = std::max(v, std::abs(A(i, j))); }
  return v;
}

double GlobalMax(double v)
{
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}

Mat3 RandomF(std::mt19937 &rng, double amplitude)
{
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Mat3 F = cmf::I<3>();
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) { F(i, j) += amplitude * u(rng); }
  return F;
}

// A rotation about a random axis (Rodrigues).
Mat3 RandomRotation(std::mt19937 &rng)
{
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Vec3 n{u(rng), u(rng), u(rng)};
  n = (1.0 / std::sqrt(cmf::dot(n, n))) * n;
  const double phi = 1.1;
  Mat3 K{}, Q{};
  K(0, 1) = -n(2); K(0, 2) = n(1); K(1, 0) = n(2); K(1, 2) = -n(0); K(2, 0) = -n(1); K(2, 1) = n(0);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    {
      Q(i, j) = std::cos(phi) * (i == j ? 1.0 : 0.0) + std::sin(phi) * K(i, j) + (1.0 - std::cos(phi)) * n(i) * n(j);
    }
  return Q;
}

const cmf::Reaction &Named(const std::vector<cmf::Reaction> &r, const std::string &name)
{
  for (const cmf::Reaction &x : r) { if (x.name == name) { return x; } }
  MFEM_ABORT("no reaction named " << name);
  return r[0];
}

// The largest deviation of a quantity of (F, p, theta) at the quadrature
// points of every element from the value the functor returns as its error.
template <typename Fn>
double MaxPointError(cmf::ThermoSolidMechanicsTL &thermo, const mfem::Vector &x, const Fn &error)
{
  thermo.UpdateFields(x);
  mfem::ParMesh &mesh = thermo.Mesh();
  const int dim = mesh.Dimension();
  double worst = 0.0;
  mfem::DenseMatrix grad;
  mfem::Vector X, u;
  for (int e = 0; e < mesh.GetNE(); e++)
  {
    mfem::ElementTransformation &T = *mesh.GetElementTransformation(e);
    const mfem::IntegrationRule &ir = thermo.History()->Rule(e);
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
      T.SetIntPoint(&ip);
      thermo.Displacement().GetVectorGradient(T, grad);
      Mat3 F = cmf::DeformationGradientAt(grad, dim);
      if (thermo.Axisymmetric())
      {
        T.Transform(ip, X);
        thermo.Displacement().GetVectorValue(T, ip, u);
        F(2, 2) = X(0) > 0.0 ? 1.0 + u(0) / X(0) : 1.0 + grad(0, 0);
      }
      const double p = thermo.Pressure().GetValue(T, ip);
      const double theta = thermo.Temperature().GetValue(T, ip);
      worst = std::max(worst, error(F, p, theta));
    }
  }
  return GlobalMax(worst);
}

// (b) The material point.
void MaterialPointTest(const std::string &law)
{
  std::printf("material point, %s volumetric law\n", law.c_str());
  cmf::AppConfig cfg = BaseConfig(3, 1, "strain", law);
  const TNH m = MaterialOf(cfg);
  CHECK(m.thermal.entropic);
  CHECK_CLOSE(m.thermal.theta0, kTheta0, 0.0);
  std::mt19937 rng(7u);
  const Mat3 F = RandomF(rng, 0.25);
  const double theta = kTheta0 + 23.0;
  // The stress is the derivative of the energy in F.
  const Mat3 P = m.PK1(F, theta);
  {
    const double h = 1e-6;
    double worst = 0.0;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
      {
        Mat3 Fp = F, Fm = F;
        Fp(i, j) += h;
        Fm(i, j) -= h;
        const double fd = (m.Energy(Fp, theta) - m.Energy(Fm, theta)) / (2.0 * h);
        worst = std::max(worst, std::abs(fd - P(i, j)));
      }
    CHECK_MSG(worst < 1e-7 * (1.0 + MaxAbs(P)), "P = dpsi/dF: " + std::to_string(worst));
  }
  // The entropic scaling of the isochoric stress.
  {
    const cmf::IsoNeoHookean &base = m;
    const Mat3 diff = m.PK1Iso(F, theta) - (theta / kTheta0) * base.PK1Iso(F);
    CHECK(MaxAbs(diff) < 1e-13);
  }
  // dP/dtheta, M = F^-1 dP/dtheta and the constitutive pressure's thermal
  // derivative against central differences in theta.
  {
    const double h = 1e-3;
    const Mat3 fd = (1.0 / (2.0 * h)) * (m.PK1(F, theta + h) - m.PK1(F, theta - h));
    const Mat3 D = m.ThermalStressDerivative(F, theta);
    CHECK_MSG(MaxAbs(fd - D) < 1e-8 * (1.0 + MaxAbs(D)), "dP/dtheta: " + std::to_string(MaxAbs(fd - D)));
    const Mat3 M = m.ThermalTangent(F, theta);
    const Mat3 Mfd = cmf::inv(F) * fd;
    CHECK(MaxAbs(Mfd - M) < 1e-8 * (1.0 + MaxAbs(M)));
    const double J = cmf::det(F);
    const double dfd = (m.NormalizedVolumetricPressure(J, theta + h) -
                        m.NormalizedVolumetricPressure(J, theta - h)) / (2.0 * h);
    CHECK_CLOSE(m.NormalizedVolumetricPressureThermalDerivative(J, theta), dfd, 1e-9);
    const double hJ = 1e-6;
    const double dJ = (m.NormalizedVolumetricPressure(J + hJ, theta) -
                       m.NormalizedVolumetricPressure(J - hJ, theta)) / (2.0 * hJ);
    CHECK_CLOSE(m.NormalizedVolumetricModulus(J, theta), dJ, 1e-7);
    // The stress-free volume ratio: u'(J / J_theta) = 0 at J = J_theta.
    CHECK_CLOSE(m.NormalizedVolumetricPressure(m.ThermalVolumeRatio(theta), theta), 0.0, 1e-14);
    CHECK_CLOSE(m.ThermalVolumeRatio(theta), std::exp(3.0 * kAlpha * (theta - kTheta0)), 1e-15);
  }
  // Objectivity: P(QF) = Q P(F), the referential flux and the energy are
  // unchanged by a rotation of the current configuration.
  {
    const Mat3 Q = RandomRotation(rng);
    const Mat3 QF = Q * F;
    CHECK(MaxAbs(m.PK1(QF, theta) - Q * P) < 1e-12);
    CHECK_CLOSE(m.Energy(QF, theta), m.Energy(F, theta), 1e-13);
    const Vec3 g{0.3, -0.2, 0.5};
    const Vec3 d = m.HeatFlux(QF, g) - m.HeatFlux(F, g);
    CHECK(std::sqrt(cmf::dot(d, d)) < 1e-12);
    // Fourier's law: Q = -k J C^-1 Grad theta.
    const Mat3 C = cmf::transpose(F) * F;
    const Vec3 want = (-kK * cmf::det(F)) * (cmf::inv(C) * g);
    const Vec3 dd = m.HeatFlux(F, g) - want;
    CHECK(std::sqrt(cmf::dot(dd, dd)) < 1e-13);
  }
  // The isothermal material at theta0 is the base.
  {
    const cmf::IsoNeoHookean &base = m;
    CHECK(MaxAbs(m.PK1(F, kTheta0) - base.PK1(F)) < 1e-13);
  }
}

// (a) Free thermal expansion: rollers on three faces, theta0 + dtheta on
// every face, held until the interior has relaxed (five steps of 1000 time
// units, decay 1e-4 per step): J = exp(3 alpha dtheta) everywhere, no
// stress, no pressure, no support reaction.
void ExpansionTest(const std::string &law)
{
  std::printf("free thermal expansion, %s volumetric law\n", law.c_str());
  cmf::AppConfig cfg = BaseConfig(3, 2, "strain", law);
  cfg.time.t_final = 5000.0;
  cfg.solver.newton.atol = 1e-9; // the conduction rows carry dt k Grad theta . Grad q ~ 1e5: round-off ~1e-11
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
  auto *thermo = dynamic_cast<cmf::ThermoSolidMechanicsTL *>(problem.get());
  CHECK(thermo != nullptr);
  CHECK(problem->Description().find("u-p-theta") != std::string::npos);
  CHECK(problem->HasHistory());
  mfem::Vector zero(3);
  zero = 0.0;
  mfem::VectorConstantCoefficient zero_coef(zero);
  const double dtheta = 40.0;
  mfem::ConstantCoefficient hot(kTheta0 + dtheta);
  cmf::BCOptions roller_x, roller_y, roller_z, heat;
  roller_x.components = {0};
  roller_y.components = {1};
  roller_z.components = {2};
  roller_x.schedule = roller_y.schedule = roller_z.schedule = cmf::Schedule::Constant();
  roller_x.name = "roller_x";
  heat.schedule = cmf::Schedule::Constant();
  problem->AddDirichlet({5}, zero_coef, roller_x);
  problem->AddDirichlet({2}, zero_coef, roller_y);
  problem->AddDirichlet({1}, zero_coef, roller_z);
  problem->AddTemperature({1, 2, 3, 4, 5, 6}, hot, heat);
  problem->Finalize();
  problem->SetPhysicalTime(true);
  std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
  mfem::Vector x(problem->Height());
  problem->InitialState(x);
  const std::vector<double> times = {1000.0, 2000.0, 3000.0, 4000.0, 5000.0};
  const cmf::QuasiStaticReport report = cmf::SolveInTime(*problem, *linear, cfg.solver, times, 0.0, x);
  CHECK(report.converged);
  CHECK(int(report.steps.size()) == 5);
  const double J_theta = std::exp(3.0 * kAlpha * dtheta);
  const TNH m = MaterialOf(cfg);
  const double eJ = MaxPointError(*thermo, x, [&](const Mat3 &F, double, double) { return std::abs(cmf::det(F) - J_theta); });
  CHECK_MSG(eJ < 1e-10, "J = exp(3 alpha dtheta): " + std::to_string(eJ));
  const double eT = MaxPointError(*thermo, x, [&](const Mat3 &, double, double theta) { return std::abs(theta - kTheta0 - dtheta); });
  CHECK_MSG(eT < 1e-9, "uniform temperature: " + std::to_string(eT));
  const double ep = MaxPointError(*thermo, x, [&](const Mat3 &, double p, double) { return std::abs(p); });
  CHECK_MSG(ep < 1e-9 * kKappa, "zero pressure: " + std::to_string(ep));
  const double eP = MaxPointError(*thermo, x, [&](const Mat3 &F, double p, double theta)
  { return MaxAbs(cmf::ThermoMixedPK1(m, F, p, theta)); });
  CHECK_MSG(eP < 1e-9, "stress-free: " + std::to_string(eP));
  const std::vector<double> u = cmf::ProbeVector(problem->Displacement(), {1.0, 1.0, 1.0});
  for (int d = 0; d < 3; d++) { CHECK_CLOSE(u[std::size_t(d)], std::cbrt(J_theta) - 1.0, 1e-10); }
  const std::vector<cmf::Reaction> reactions = problem->Reactions(x);
  CHECK(reactions.size() == 3);
  for (const cmf::Reaction &r : reactions)
  {
    for (int d = 0; d < 3; d++) { CHECK_MSG(std::abs(r.force[d]) < 1e-8, r.name + " reaction"); }
  }
  // The energy diagnostic is the stored energy of the stress-free state: zero.
  CHECK_MSG(std::abs(problem->InternalEnergy(x)) < 1e-12, "energy " + std::to_string(problem->InternalEnergy(x)));
}

// (c) Adiabatic homogeneous stretch: rollers, u_y ramped on y = 1, no thermal
// entries (insulated). The state is homogeneous, so the material point with
// the solution's F integrates the same heat equation
//   c_v (theta - theta_n) = 1/2 theta M(F, theta) : (C - C_n)
// step by step; the pressure is the constitutive one.
void AdiabaticStretchTest()
{
  std::printf("adiabatic homogeneous stretch\n");
  cmf::AppConfig cfg = BaseConfig(3, 2);
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
  auto &thermo = dynamic_cast<cmf::ThermoSolidMechanicsTL &>(*problem);
  mfem::Vector zero(3), pull(3);
  zero = 0.0;
  pull = 0.0;
  pull(1) = 0.5;
  mfem::VectorConstantCoefficient zero_coef(zero), pull_coef(pull);
  cmf::BCOptions roller_x, roller_y, roller_z, loaded;
  roller_x.components = {0};
  roller_y.components = {1};
  roller_z.components = {2};
  roller_x.schedule = roller_y.schedule = roller_z.schedule = cmf::Schedule::Constant();
  loaded.components = {1};
  loaded.schedule = cmf::Schedule::Ramp(0.0, 1.0);
  loaded.name = "loaded";
  problem->AddDirichlet({5}, zero_coef, roller_x);
  problem->AddDirichlet({2}, zero_coef, roller_y);
  problem->AddDirichlet({1}, zero_coef, roller_z);
  problem->AddDirichlet({4}, pull_coef, loaded);
  problem->Finalize();
  problem->SetPhysicalTime(true);
  std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
  mfem::Vector x(problem->Height());
  problem->InitialState(x);
  const TNH m = MaterialOf(cfg);
  const std::vector<double> times = {0.25, 0.5, 0.75, 1.0};
  double theta_n = kTheta0;
  Mat3 C_n = cmf::I<3>();
  double worst_theta = 0.0, worst_p = 0.0, worst_P = 0.0, worst_uniform = 0.0;
  const cmf::LoadStepCallback on_step = [&](const cmf::LoadStepReport &step, const mfem::Vector &xs)
  {
    thermo.UpdateFields(xs);
    const std::vector<double> ux = cmf::ProbeVector(thermo.Displacement(), {1.0, 0.0, 0.0});
    const std::vector<double> uz = cmf::ProbeVector(thermo.Displacement(), {0.0, 0.0, 1.0});
    const std::vector<double> th = cmf::ProbeVector(thermo.Temperature(), {0.5, 0.5, 0.5});
    const std::vector<double> pr = cmf::ProbeVector(thermo.Pressure(), {0.5, 0.5, 0.5});
    Mat3 F{};
    F(0, 0) = 1.0 + ux[0];
    F(1, 1) = 1.0 + 0.5 * step.load_factor;
    F(2, 2) = 1.0 + uz[2];
    const Mat3 C = cmf::transpose(F) * F;
    // The material point's temperature: a scalar Newton iteration.
    double theta = theta_n;
    for (int it = 0; it < 50; it++)
    {
      const cmf::dual td(theta, 1.0);
      tensor<cmf::dual, 3, 3> Fd, dC;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
          Fd(i, j) = cmf::dual(F(i, j), 0.0);
          dC(i, j) = cmf::dual(C(i, j) - C_n(i, j), 0.0);
        }
      const cmf::dual r = kCv * (td - theta_n) - 0.5 * td * cmf::ddot(m.ThermalTangent(Fd, td), dC);
      theta -= r.v / r.d;
      if (std::abs(r.v) < 1e-14 * kCv * kTheta0) { break; }
    }
    worst_theta = std::max(worst_theta, std::abs(th[0] - theta));
    worst_p = std::max(worst_p, std::abs(pr[0] - m.VolumetricPressure(cmf::det(F), theta)));
    const Mat3 P = cmf::ThermoMixedPK1(m, F, pr[0], th[0]);
    worst_P = std::max(worst_P, std::max(std::abs(P(0, 0)), std::abs(P(2, 2))));
    // Uniformity of the fields at the quadrature points.
    worst_uniform = std::max(worst_uniform, MaxPointError(thermo, xs, [&](const Mat3 &Fq, double pq, double tq)
    { return std::max({MaxAbs(Fq - F), std::abs(pq - pr[0]), std::abs(tq - th[0])}); }));
    theta_n = theta;
    C_n = C;
  };
  const cmf::QuasiStaticReport report = cmf::SolveInTime(*problem, *linear, cfg.solver, times, 0.0, x, on_step);
  CHECK(report.converged);
  CHECK_MSG(worst_theta < 1e-9, "temperature vs material point: " + std::to_string(worst_theta));
  CHECK_MSG(worst_p < 1e-8, "constitutive pressure: " + std::to_string(worst_p));
  CHECK_MSG(worst_P < 1e-8, "free lateral faces: " + std::to_string(worst_P));
  CHECK_MSG(worst_uniform < 1e-8, "homogeneous state: " + std::to_string(worst_uniform));
  // Stretching an entropic elastomer adiabatically heats it (Gough-Joule).
  CHECK_MSG(theta_n > kTheta0 + 0.01, "Gough-Joule heating: " + std::to_string(theta_n - kTheta0));
  CHECK_CLOSE(thermo.AcceptedTime(), 1.0, 1e-12);
}

// (d) Transient conduction: alpha = 0 and no entropic scaling decouple the
// heat equation; a rollered strip 0 < x < 1 at theta0, insulated but for
// theta0 + 1 on x = 1 from t = 0+. The insulated end against the series
// solution; the error of the implicit Euler method halves with the step.
double ConductionError(int steps)
{
  cmf::AppConfig cfg = BaseConfig(2, 16);
  cfg.mesh.box.ny = 1;
  cfg.material.thermal.alpha = 0.0;
  cfg.material.thermal.entropic = false;
  cfg.time.t_final = 0.5;
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
  auto &thermo = dynamic_cast<cmf::ThermoSolidMechanicsTL &>(*problem);
  mfem::Vector zero(2);
  zero = 0.0;
  mfem::VectorConstantCoefficient zero_coef(zero);
  mfem::ConstantCoefficient hot(kTheta0 + 1.0);
  cmf::BCOptions roller_x, roller_y, heat;
  roller_x.components = {0};
  roller_y.components = {1};
  roller_x.schedule = roller_y.schedule = heat.schedule = cmf::Schedule::Constant();
  problem->AddDirichlet({2, 4}, zero_coef, roller_x);
  problem->AddDirichlet({1, 3}, zero_coef, roller_y);
  problem->AddTemperature({2}, hot, heat);
  problem->Finalize();
  problem->SetPhysicalTime(true);
  std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
  mfem::Vector x(problem->Height());
  problem->InitialState(x);
  std::vector<double> times;
  for (int n = 1; n <= steps; n++) { times.push_back(0.5 * n / steps); }
  const cmf::QuasiStaticReport report = cmf::SolveInTime(*problem, *linear, cfg.solver, times, 0.0, x);
  CHECK(report.converged);
  thermo.UpdateFields(x);
  // The mechanics stayed at rest.
  const mfem::Array<int> &off = thermo.BlockOffsets();
  double rest = 0.0;
  for (int i = 0; i < off[2]; i++) { rest = std::max(rest, std::abs(x(i))); }
  CHECK(GlobalMax(rest) < 1e-12);
  double exact = 1.0;
  for (int n = 0; n < 60; n++)
  {
    const double lam = (2 * n + 1) * M_PI / 2.0;
    exact -= 4.0 * (n % 2 ? -1.0 : 1.0) / ((2 * n + 1) * M_PI) * std::exp(-lam * lam * 0.5);
  }
  const std::vector<double> th = cmf::ProbeVector(thermo.Temperature(), {0.0, 0.5});
  return th[0] - kTheta0 - exact;
}

void ConductionTest()
{
  std::printf("transient conduction against the series solution\n");
  const double e1 = ConductionError(100), e2 = ConductionError(200);
  std::printf("  errors at the insulated end: %.3e (100 steps), %.3e (200 steps)\n", e1, e2);
  CHECK_MSG(std::abs(e2) < 4e-3, "conduction error: " + std::to_string(e2));
  CHECK_MSG(std::abs(e2) < 0.6 * std::abs(e1), "first order in the step");
}

// (h) The heat-flux entry: on a homogeneously stretched cube with the
// history at the current state, the temperature residual sums to
// -dt h A_current (per current area: |cof F N| A_R) or -dt h A_R.
void FluxResidualTest(bool current_area)
{
  std::printf("heat flux per %s area\n", current_area ? "current" : "reference");
  cmf::AppConfig cfg = BaseConfig(3, 1);
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
  auto &thermo = dynamic_cast<cmf::ThermoSolidMechanicsTL &>(*problem);
  mfem::ConstantCoefficient h(3.0);
  cmf::BCOptions flux;
  flux.schedule = cmf::Schedule::Constant();
  problem->AddHeatFlux({6}, h, current_area, flux);
  problem->Finalize();
  problem->SetPhysicalTime(true);
  // u = (0.2 x, -0.1 y, 0.1 z): F = diag(1.2, 0.9, 1.1), exactly in Q2.
  mfem::Vector x(problem->Height());
  problem->InitialState(x);
  mfem::VectorFunctionCoefficient stretch(3, [](const mfem::Vector &X, mfem::Vector &u)
  { u(0) = 0.2 * X(0); u(1) = -0.1 * X(1); u(2) = 0.1 * X(2); });
  mfem::ParGridFunction g(&thermo.DisplacementSpace());
  g.ProjectCoefficient(stretch);
  mfem::Vector gt(thermo.BlockOffsets()[1]);
  g.GetTrueDofs(gt);
  for (int i = 0; i < gt.Size(); i++) { x(i) = gt(i); }
  problem->SetLoadFactor(0.0);
  problem->AcceptStep(x);
  problem->SetLoadFactor(0.5); // dt = 0.5
  mfem::Vector r;
  problem->FullResidual(x, r);
  const mfem::Array<int> &off = thermo.BlockOffsets();
  double local = 0.0;
  for (int i = off[2]; i < off[3]; i++) { local += r(i); }
  double sum = 0.0;
  MPI_Allreduce(&local, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  const double area = current_area ? 1.2 * 0.9 : 1.0;
  CHECK_CLOSE(sum, -0.5 * 3.0 * area, 1e-12);
  // The displacement residual is that of the stretched state (no load on it).
  double ru = 0.0;
  for (int i = 0; i < off[1]; i++) { ru = std::max(ru, std::abs(r(i))); }
  CHECK(GlobalMax(ru) > 1e-3);
}

// (e) The assembled Jacobian against central differences of the residual,
// with a follower pressure, a prescribed temperature, a heat flux per
// current area and a pin active, in a step of length 0.4 after an accepted
// random state.
void JacobianTest(const std::string &plane, bool faces = true)
{
  std::printf("block Jacobian against finite differences, plane %s%s\n", plane.c_str(), faces ? "" : ", no face terms");
  cmf::AppConfig cfg = BaseConfig(2, 2, plane);
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
  auto &thermo = dynamic_cast<cmf::ThermoSolidMechanicsTL &>(*problem);
  mfem::Vector zero(2);
  zero = 0.0;
  mfem::VectorConstantCoefficient zero_coef(zero);
  mfem::ConstantCoefficient pressure(0.3), hot(kTheta0 + 10.0), h(2.0);
  cmf::BCOptions fixed, pin, follower, heat, flux;
  fixed.schedule = pin.schedule = follower.schedule = heat.schedule = flux.schedule = cmf::Schedule::Constant();
  pin.point = {0.0, 1.0};
  pin.components = {0};
  problem->AddDirichlet({1}, zero_coef, fixed);
  problem->AddDirichlet({}, zero_coef, pin);
  if (faces) { problem->AddPressure({2}, pressure, true, follower); }
  problem->AddTemperature({1}, hot, heat);
  if (faces) { problem->AddHeatFlux({3}, h, true, flux); }
  problem->Finalize();
  problem->SetPhysicalTime(true);
  const mfem::Array<int> &off = thermo.BlockOffsets();
  std::mt19937 rng(5u);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  auto perturb = [&](mfem::Vector &x, double au, double ap, double at)
  {
    for (int i = 0; i < off[1]; i++) { x(i) += au * u(rng); }
    for (int i = off[1]; i < off[2]; i++) { x(i) += ap * u(rng); }
    for (int i = off[2]; i < off[3]; i++) { x(i) += at * u(rng); }
  };
  mfem::Vector x(problem->Height());
  problem->InitialState(x);
  perturb(x, 0.03, 0.2, 5.0);
  problem->SetLoadFactor(0.3);
  problem->AcceptStep(x);
  problem->SetLoadFactor(0.7);
  perturb(x, 0.02, 0.1, 3.0);
  problem->ApplyDirichlet(x);
  mfem::Operator &J = problem->GetGradient(x);
  const int n = x.Size();
  mfem::Vector v(n), Jv(n), rp(n), rm(n), xp(n), xm(n);
  const mfem::Array<int> &ess_u = thermo.EssentialTrueDofs();
  const mfem::Array<int> &ess_t = thermo.EssentialTemperatureDofs();
  int n_ess_t = ess_t.Size(), n_ess_t_global = 0;
  MPI_Allreduce(&n_ess_t, &n_ess_t_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  CHECK(n_ess_t_global > 0);
  const double eps = 1e-6;
  double worst = 0.0;
  for (int trial = 0; trial < 3; trial++)
  {
    for (int i = 0; i < n; i++) { v(i) = u(rng); }
    for (int i = off[2]; i < off[3]; i++) { v(i) *= 20.0; }
    for (int i = 0; i < ess_u.Size(); i++) { v(ess_u[i]) = 0.0; }
    for (int i = 0; i < ess_t.Size(); i++) { v(off[2] + ess_t[i]) = 0.0; }
    J.Mult(v, Jv);
    xp = x;
    xp.Add(eps, v);
    xm = x;
    xm.Add(-eps, v);
    problem->Mult(xp, rp);
    problem->Mult(xm, rm);
    rp -= rm;
    rp /= 2.0 * eps;
    rp -= Jv;
    const double err = std::sqrt(mfem::InnerProduct(MPI_COMM_WORLD, rp, rp));
    const double ref = std::sqrt(mfem::InnerProduct(MPI_COMM_WORLD, Jv, Jv));
    worst = std::max(worst, err / ref);
  }
  CHECK_MSG(worst < 1e-6, "Jacobian vs finite differences: " + std::to_string(worst));
  // The direct solver on the merged 3 x 3 block system.
  std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
  mfem::Vector r(n), dx(n), chk(n);
  problem->Mult(x, r);
  linear->SetOperator(J);
  dx = 0.0;
  linear->Mult(r, dx);
  J.Mult(dx, chk);
  chk -= r;
  const double res = std::sqrt(mfem::InnerProduct(MPI_COMM_WORLD, chk, chk));
  const double rn = std::sqrt(mfem::InnerProduct(MPI_COMM_WORLD, r, r));
  CHECK_MSG(res < 1e-10 * rn, "direct solve of the block system: " + std::to_string(res / rn));
}

// (f) Point constraints: a plane-strain block on a pin at (0, 0) and a
// vertical support at (1, 0), a dead traction on the top face (two
// increments: the point supports make the response strongly nonlinear): the
// reactions balance the load, and a point off the nodes is refused.
void PinTest(const std::string &formulation)
{
  std::printf("point constraints, %s formulation\n", formulation.c_str());
  cmf::AppConfig cfg = BaseConfig(2, 4);
  if (formulation != "thermo")
  {
    cfg.formulation = formulation;
    cfg.material.thermal.set = false;
    cfg.time.enabled = false;
    cfg.solver.load_steps = 2;
    cfg.solver.linear.type = "direct";
  }
  std::unique_ptr<mfem::ParMesh> mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*mesh, cfg);
  mfem::Vector zero(2), load(2);
  zero = 0.0;
  load = 0.0;
  load(1) = -0.02;
  mfem::VectorConstantCoefficient zero_coef(zero), load_coef(load);
  cmf::BCOptions pin_a, pin_b, traction;
  pin_a.point = {0.0, 0.0};
  pin_a.name = "pin_a";
  pin_b.point = {1.0, 0.0};
  pin_b.components = {1};
  pin_b.name = "pin_b";
  problem->AddDirichlet({}, zero_coef, pin_a);
  problem->AddDirichlet({}, zero_coef, pin_b);
  problem->AddTraction({3}, load_coef, traction);
  problem->Finalize();
  CHECK(problem->EssentialTrueDofs().Size() >= 0);
  int ess = problem->EssentialTrueDofs().Size(), ess_global = 0;
  MPI_Allreduce(&ess, &ess_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  CHECK(ess_global == 3);
  std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
  mfem::Vector x(problem->Height());
  problem->InitialState(x);
  cmf::QuasiStaticReport report;
  if (formulation == "thermo")
  {
    problem->SetPhysicalTime(true);
    report = cmf::SolveInTime(*problem, *linear, cfg.solver, {0.5, 1.0}, 0.0, x);
  }
  else { report = cmf::SolveQuasiStatic(*problem, *linear, cfg.solver, x); }
  CHECK(report.converged);
  problem->UpdateFields(x);
  const std::vector<cmf::Reaction> reactions = problem->Reactions(x);
  const cmf::Reaction &a = Named(reactions, "pin_a"), &b = Named(reactions, "pin_b");
  CHECK_CLOSE(a.force[1] + b.force[1], 0.02, 1e-10);
  CHECK_CLOSE(a.force[0], 0.0, 1e-10);
  CHECK(b.force[0] == 0.0);
  CHECK(std::abs(a.force[1]) > 0.002 && std::abs(b.force[1]) > 0.002);
  // The pinned nodes did not move.
  const std::vector<double> ua = cmf::ProbeVector(problem->Displacement(), {0.0, 0.0});
  const std::vector<double> ub = cmf::ProbeVector(problem->Displacement(), {1.0, 0.0});
  CHECK(std::abs(ua[0]) < 1e-14 && std::abs(ua[1]) < 1e-14 && std::abs(ub[1]) < 1e-14);
  CHECK(std::abs(ub[0]) > 1e-6);
  // A point that is not a node.
  cmf::BCOptions off_node;
  off_node.point = {0.123, 0.456};
  problem->AddDirichlet({}, zero_coef, off_node);
  CHECK_THROWS(problem->Finalize(), cmf::ConfigError, "not a node");
}

// (g) The schema.
void ConfigTest()
{
  std::printf("schema\n");
  const std::string mesh = "mesh: { file: apps/mesh/cube.msh, order: 2 }\n";
  const std::string material = "material: { model: iso_neo_hookean, mu: 1.0, kappa: 10.0, thermal: { theta0: 300, alpha: 1e-4, c_v: 2.0, k: 0.5 } }\n";
  const std::string good = mesh + "formulation: mixed\n" + material +
    "time: { t_final: 1.0, dt: 0.5 }\nsolver: { linear: { type: direct } }\n"
    "bcs: { temperature: [ { attr: [top], expression: \"300 + 10*t\" } ], heat_flux: [ { attr: [bottom], expression: \"5\", per_unit: reference_area } ],\n"
    "       dirichlet: [ { point: [0, 0, 0], expression: [\"0\", \"0\", \"0\"] } ] }\n"
    "output: { fields: [displacement, pressure, temperature] }\n";
  {
    const cmf::AppConfig cfg = cmf::ParseConfig(YAML::Load(good));
    CHECK(cfg.material.thermal.set);
    CHECK_CLOSE(cfg.material.thermal.c_v, 2.0, 0.0);
    CHECK(cfg.material.thermal.entropic);
    CHECK(cfg.bcs.temperature.size() == 1 && cfg.bcs.heat_flux.size() == 1);
    CHECK(cfg.bcs.temperature[0].schedule.kind == cmf::Schedule::Kind::Constant);
    CHECK(!cfg.bcs.heat_flux[0].current_area);
    CHECK(cfg.bcs.dirichlet[0].IsPoint() && cfg.bcs.dirichlet[0].point.size() == 3);
    const cmf::ThermoMaterial m = cmf::MakeThermoMaterial(cfg.material);
    CHECK(cmf::MaterialName(m) == "iso_neo_hookean (entropic, thermoelastic)");
  }
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(mesh + "formulation: mixed\n" + material + "solver: { linear: { type: direct } }\n")),
               cmf::ConfigError, "physical time");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(mesh + material + "time: { t_final: 1.0, dt: 0.5 }\nsolver: { linear: { type: direct } }\n")),
               cmf::ConfigError, "formulation: mixed");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(mesh + "formulation: mixed\n" + material + "time: { t_final: 1.0, dt: 0.5 }\n")),
               cmf::ConfigError, "use direct");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(mesh + "formulation: mixed\nplane: stress\n" + material + "time: { t_final: 1.0, dt: 0.5 }\nsolver: { linear: { type: direct } }\n")),
               cmf::ConfigError, "formulation: displacement");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(mesh + "formulation: mixed\nmaterial: { model: iso_neo_hookean, mu: 1.0, kappa: 10.0 }\ntime: { t_final: 1.0, dt: 0.5 }\n"
                                           "bcs: { temperature: [ { attr: [top], expression: \"300\" } ] }\n")),
               cmf::ConfigError, "thermoelastic material");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(mesh + "formulation: mixed\nmaterial: { model: iso_neo_hookean, mu: 1.0, kappa: 10.0, branches: [ { G: 1.0, tau: 1.0 } ], thermal: { theta0: 300, alpha: 1e-4, c_v: 2.0, k: 0.5 } }\n"
                                           "time: { t_final: 1.0, dt: 0.5 }\nsolver: { linear: { type: direct } }\n")),
               cmf::ConfigError, "thermoelastic material");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(mesh + "formulation: mixed\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3, thermal: { theta0: 300, alpha: 1e-4, c_v: 2.0, k: 0.5 } }\n"
                                           "time: { t_final: 1.0, dt: 0.5 }\nsolver: { linear: { type: direct } }\n")),
               cmf::ConfigError, "isochoric-volumetric");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(mesh + "formulation: mixed\nmaterial: { model: iso_neo_hookean, mu: 1.0, kappa: 10.0, thermal: { theta0: 300, alpha: 1e-4, k: 0.5 } }\n"
                                           "time: { t_final: 1.0, dt: 0.5 }\nsolver: { linear: { type: direct } }\n")),
               cmf::ConfigError, "c_v");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(mesh + "formulation: mixed\n" + material + "time: { t_final: 1.0, dt: 0.5 }\nsolver: { linear: { type: direct } }\n"
                                           "bcs: { heat_flux: [ { attr: [top], expression: \"5\", per_unit: deformed } ] }\n")),
               cmf::ConfigError, "per_unit");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(mesh + "formulation: mixed\n" + material + "time: { t_final: 1.0, dt: 0.5 }\nsolver: { linear: { type: direct } }\n"
                                           "bcs: { dirichlet: [ { attr: [top], point: [0, 0, 0], expression: [\"0\", \"0\", \"0\"] } ] }\n")),
               cmf::ConfigError, "one, not both");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(mesh + "formulation: mixed\n" + material + "time: { t_final: 1.0, dt: 0.5 }\nsolver: { linear: { type: direct } }\n"
                                           "bcs: { traction: [ { point: [0, 0, 0], expression: [\"0\", \"0\", \"1\"] } ] }\n")),
               cmf::ConfigError, "Dirichlet entries only");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(mesh + "formulation: mixed\nmaterial: { model: iso_neo_hookean, mu: 1.0, kappa: 10.0, thermal: { theta0: 300, alpha: 1e-4, c_v: 2.0, k: 0.5 },\n"
                                           "            regions: [ { attr: [domain], thermal: { theta0: 310 } } ] }\ntime: { t_final: 1.0, dt: 0.5 }\nsolver: { linear: { type: direct } }\n")),
               cmf::ConfigError, "reference temperature");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(mesh + "formulation: mixed\n" + material + "dynamics: { t_final: 1.0, dt: 0.5 }\nsolver: { linear: { type: direct } }\n")),
               cmf::ConfigError, "not supported");
  // A region with its own expansion coefficient shares the base's theta0.
  {
    const cmf::AppConfig cfg = cmf::ParseConfig(YAML::Load(mesh + "formulation: mixed\nmaterial: { model: iso_neo_hookean, mu: 1.0, kappa: 10.0, thermal: { theta0: 300, alpha: 1e-4, c_v: 2.0, k: 0.5 },\n"
                                                           "            regions: [ { attr: [domain], thermal: { alpha: 0.0 } } ] }\ntime: { t_final: 1.0, dt: 0.5 }\nsolver: { linear: { type: direct } }\n"));
    CHECK(cfg.material.regions.size() == 1);
    CHECK_CLOSE(cfg.material.regions[0].thermal.theta0, 300.0, 0.0);
    CHECK_CLOSE(cfg.material.regions[0].thermal.alpha, 0.0, 0.0);
    CHECK_CLOSE(cfg.material.regions[0].thermal.c_v, 2.0, 0.0);
  }
}

} // namespace

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  const cmf::PetscSession petsc; // solver.linear.type: direct
  ConfigTest();
  MaterialPointTest("quadratic");
  MaterialPointTest("logarithmic");
  ExpansionTest("quadratic");
  ExpansionTest("logarithmic");
  AdiabaticStretchTest();
  ConductionTest();
  FluxResidualTest(true);
  FluxResidualTest(false);
  JacobianTest("strain", false);
  JacobianTest("strain");
  JacobianTest("axisymmetric");
  PinTest("mixed");
  PinTest("thermo");
  PinTest("displacement");
  return cmf_test::Report("test_thermoelastic");
}

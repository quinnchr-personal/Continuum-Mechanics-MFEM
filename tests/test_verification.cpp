// Verification against reference solutions, driven by the inputs of
// apps/input/finite_elasticity/verification: Rivlin's torsion of a
// neo-Hookean cylinder, the Green-Zerna inflation of a thick sphere,
// three-dimensional manufactured solutions on hexahedra and tetrahedra and
// in the mixed formulation, compressible uniaxial tension (lateral stretch
// from the material's own PK1), Kirsch's stress concentration at small
// strain, and the Euler buckling strain of an imperfect column (Southwell).
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "base/coefficients.hpp"
#include "base/fields.hpp"
#include "base/mesh_input.hpp"
#include "base/probes.hpp"
#include "materials/materials.hpp"
#include "mfem.hpp"
#include "physics/solid_problem.hpp"
#include "solvers/quasi_static.hpp"
#include "test_util.hpp"

using cmf::tensor;

namespace
{

const char *kDir = "apps/input/finite_elasticity/verification/";

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
};

// Load, solve and register the fields of an input (quiet Newton).
std::unique_ptr<Solved> Solve(cmf::AppConfig cfg, const cmf::LoadStepCallback &on_step = {})
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
  s->report = cmf::SolveQuasiStatic(*s->problem, *linear, cfg.solver, s->x, on_step);
  s->problem->UpdateFields(s->x);
  s->problem->RegisterFields(s->fields);
  return s;
}

cmf::AppConfig Load(const std::string &name)
{
  return cmf::LoadConfig(kDir + name);
}

double Rel(double got, double want) { return std::abs(got - want) / std::max(std::abs(want), 1e-300); }

// ------------------------------------------------------------- Rivlin torsion

void RivlinTorsionTest()
{
  cmf::AppConfig cfg = Load("rivlin_torsion.yaml");
  cfg.output.fields = {"displacement", "pressure", "vonmises", "cauchy_stress"};
  std::unique_ptr<Solved> s = Solve(cfg);
  CHECK_MSG(s->report.converged, "torsion converged");
  const double mu = 1.0, a = 12.7, L = 25.4, theta = 1.0, psi = theta / L;
  // Exact rotation field for the displacement error.
  cmf::ExpressionVectorCoefficient exact(std::vector<std::string>{
    "x*cos(z/25.4) - y*sin(z/25.4) - x", "x*sin(z/25.4) + y*cos(z/25.4) - y", "0"});
  const double u_err = s->problem->Displacement().ComputeL2Error(exact);
  mfem::Vector zero(3);
  zero = 0.0;
  mfem::VectorConstantCoefficient zero_coef(zero);
  const double u_norm = s->problem->Displacement().ComputeL2Error(zero_coef);
  std::printf("  torsion: |u - u_exact|_L2 / |u|_L2 = %.3e\n", u_err / u_norm);
  CHECK_MSG(u_err / u_norm <= 2e-3, "torsion displacement matches the universal solution");
  // Stresses on the mid-plane, probed at the reference points (r, 0, L/2).
  // The Cauchy stress lives in the deformed configuration, where the
  // material point has turned by phi = psi z about the axis, so its
  // cylindrical components are taken in the rotated frame.
  for (const double r : {0.5 * a, 0.9 * a})
  {
    const std::vector<double> c = s->Probe("cauchy_stress", {r, 0.0, 0.5 * L}); // xx yy zz xy yz xz
    const std::vector<double> p = s->Probe("pressure", {r, 0.0, 0.5 * L});
    const std::vector<double> vm = s->Probe("vonmises", {r, 0.0, 0.5 * L});
    const double phi = psi * 0.5 * L;
    const double er[3] = {std::cos(phi), std::sin(phi), 0.0}, et[3] = {-std::sin(phi), std::cos(phi), 0.0},
                 ez[3] = {0.0, 0.0, 1.0};
    const double S[3][3] = {{c[0], c[3], c[5]}, {c[3], c[1], c[4]}, {c[5], c[4], c[2]}};
    auto comp = [&](const double *u, const double *v)
    {
      double sum = 0.0;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) { sum += u[i] * S[i][j] * v[j]; }
      return sum;
    };
    // sig: rr, thetatheta, zz, rtheta, thetaz, rz
    const std::vector<double> sig = {comp(er, er), comp(et, et), comp(ez, ez),
                                     comp(er, et), comp(et, ez), comp(er, ez)};
    const double g = psi * r;
    const double srr = -0.5 * mu * psi * psi * (a * a - r * r);
    const double stt = srr + mu * g * g;
    const double szz = srr;
    const double stz = mu * g;
    const double p_exact = srr + mu * g * g / 3.0;
    const double vm_exact = mu * std::sqrt(g * g * g * g + 3.0 * g * g);
    std::printf("  torsion r = %.2f: sigma_thetaz %.5f (exact %.5f), sigma_rr %.5f (%.5f), "
                "sigma_thetatheta %.5f (%.5f), sigma_zz %.5f (%.5f), p %.5f (%.5f), vm %.5f (%.5f)\n",
                r, sig[4], stz, sig[0], srr, sig[1], stt, sig[2], szz, p[0], p_exact, vm[0], vm_exact);
    const double scale = mu * g; // the dominant stress
    CHECK_MSG(std::abs(sig[4] - stz) <= 0.02 * scale, "torsion shear stress");
    CHECK_MSG(std::abs(sig[0] - srr) <= 0.03 * scale, "torsion radial stress");
    CHECK_MSG(std::abs(sig[1] - stt) <= 0.03 * scale, "torsion hoop stress");
    CHECK_MSG(std::abs(sig[2] - szz) <= 0.03 * scale, "torsion axial stress");
    CHECK_MSG(std::abs(p[0] - p_exact) <= 0.03 * scale, "torsion pressure");
    CHECK_MSG(std::abs(vm[0] - vm_exact) <= 0.02 * vm_exact, "torsion von Mises");
    CHECK_MSG(std::abs(sig[3]) + std::abs(sig[5]) <= 0.02 * scale, "torsion: no r-theta / r-z shear");
  }
}

// ------------------------------------------------------ Green-Zerna sphere

void SphereInflationTest()
{
  const double mu = 1.0, A = 10.0, B = 11.0;
  auto pressure_of = [&](double a)
  {
    const double b = std::cbrt(B * B * B - A * A * A + a * a * a);
    const double la = a / A, lb = b / B;
    return 0.5 * mu * ((4.0 / lb + 1.0 / std::pow(lb, 4)) - (4.0 / la + 1.0 / std::pow(la, 4)));
  };
  cmf::AppConfig cfg = Load("green_zerna_sphere_inflation.yaml");
  cfg.output.fields = {"displacement"};
  const double P = cmf::Expression::Parse(cfg.bcs.traction.at(0).expression.at(0)).Eval(0, 0, 0, 1);
  double lo = A, hi = 1.35 * A; // below the pressure maximum of this shell
  for (int i = 0; i < 200; i++)
  {
    const double m = 0.5 * (lo + hi);
    (pressure_of(m) < P ? lo : hi) = m;
  }
  const double a_exact = 0.5 * (lo + hi);
  const double b_exact = std::cbrt(B * B * B - A * A * A + a_exact * a_exact * a_exact);
  std::unique_ptr<Solved> s = Solve(cfg);
  CHECK_MSG(s->report.converged, "sphere inflation converged");
  const double a = A + s->Probe("displacement", {A, 0.0, 0.0})[0];
  const double b = B + s->Probe("displacement", {B, 0.0, 0.0})[0];
  const double a_z = A + s->Probe("displacement", {0.0, 0.0, A})[2];
  std::printf("  sphere inflation: a = %.6f (exact %.6f, rel %.2e), b = %.6f (exact %.6f, rel %.2e), "
              "a on the z axis %.6f\n", a, a_exact, Rel(a - A, a_exact - A), b, b_exact,
              Rel(b - B, b_exact - B), a_z);
  CHECK_MSG(Rel(a - A, a_exact - A) <= 5e-3, "inner radius within 0.5% of Green-Zerna");
  CHECK_MSG(Rel(b - B, b_exact - B) <= 5e-3, "outer radius within 0.5% of Green-Zerna");
  CHECK_MSG(std::abs(a_z - a) <= 1e-3 * (a - A), "spherically symmetric response");
}

// --------------------------------------------- manufactured solutions in 3D

void ManufacturedTest(const std::string &input, const std::vector<std::string> &exact_u,
                      int refine_from, int refine_to, double min_rate)
{
  cmf::AppConfig cfg = Load("manufactured_solutions/" + input);
  cfg.output.fields = {"displacement"};
  cmf::ExpressionVectorCoefficient exact(exact_u);
  std::vector<double> errors;
  for (int refine = refine_from; refine <= refine_to; refine++)
  {
    cfg.mesh.serial_refine = refine;
    std::unique_ptr<Solved> s = Solve(cfg);
    CHECK_MSG(s->report.converged, input + " converged at refine " + std::to_string(refine));
    errors.push_back(s->problem->Displacement().ComputeL2Error(exact));
    std::printf("  %s refine %d: %lld dofs, L2 error %.3e, newton its %d\n", input.c_str(), refine,
                static_cast<long long>(s->problem->GlobalTrueVSize()), errors.back(),
                s->report.steps.back().newton.iterations);
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
  const std::vector<std::string> u3 = {
    "0.05*sin(pi*x)*sin(pi*y)*sin(pi*z)", "0.05*x^2*y*(1 - y)*z", "0.05*x*z*(1 - z)*cos(pi*y/2)"};
  const std::vector<std::string> u_iso = {"0.05*sin(pi*y)*sin(pi*z)", "0.05*z*(1 - z)", "0"};
  ManufacturedTest("mms_3d_hex.yaml", u3, 1, 2, 2.8);
  ManufacturedTest("mms_3d_tet.yaml", u3, 1, 2, 2.8);
  ManufacturedTest("mms_3d_mixed.yaml", u_iso, 1, 2, 2.8);
}

// --------------------------------------------- compressible uniaxial tension

void CompressibleUniaxialTest(const std::string &name)
{
  cmf::AppConfig cfg = Load("homogeneous_deformations/" + name);
  cfg.output.fields = {"displacement", "pk1_stress", "cauchy_stress", "jacobian"};
  const double lam1 = 1.5;
  // The penalty form of the decoupled models is the mixed one's P at p = kappa (J - 1).
  const cmf::Material material = cmf::MakeMaterial(cfg.material);
  auto P_of = [&](double lam2)
  {
    tensor<double, 3, 3> F = cmf::I<3>();
    F(0, 0) = lam1;
    F(1, 1) = F(2, 2) = lam2;
    return std::visit([&](const auto &mat) { return mat.PK1(F); }, material);
  };
  double lo = 0.3, hi = 1.5; // P_22 increases with lambda_2
  for (int i = 0; i < 200; i++)
  {
    const double m = 0.5 * (lo + hi);
    (P_of(m)(1, 1) < 0.0 ? lo : hi) = m;
  }
  const double lam2 = 0.5 * (lo + hi);
  const tensor<double, 3, 3> P = P_of(lam2);
  const double J = lam1 * lam2 * lam2;
  const double sigma11 = P(0, 0) * lam1 / J;
  std::unique_ptr<Solved> s = Solve(cfg);
  CHECK_MSG(s->report.converged, name + " converged");
  for (const std::vector<double> pt : {std::vector<double>{1.0, 1.0, 1.0}, std::vector<double>{0.5, 0.5, 0.5}})
  {
    const std::vector<double> u = s->Probe("displacement", pt);
    const std::vector<double> Pk = s->Probe("pk1_stress", pt);   // row-major
    const std::vector<double> sig = s->Probe("cauchy_stress", pt); // xx yy zz xy yz xz
    const std::vector<double> Jp = s->Probe("jacobian", pt);
    std::printf("  %s at (%.1f, %.1f, %.1f): lambda_2 %.10f (exact %.10f), P_11 %.6f (%.6f), "
                "sigma_11 %.6f (%.6f), J %.8f (%.8f)\n", name.c_str(), pt[0], pt[1], pt[2],
                1.0 + u[1] / pt[1], lam2, Pk[0], P(0, 0), sig[0], sigma11, Jp[0], J);
    CHECK_MSG(std::abs(u[0] - (lam1 - 1.0) * pt[0]) <= 1e-10, name + " axial displacement");
    CHECK_MSG(std::abs(u[1] - (lam2 - 1.0) * pt[1]) <= 1e-8, name + " lateral displacement y");
    CHECK_MSG(std::abs(u[2] - (lam2 - 1.0) * pt[2]) <= 1e-8, name + " lateral displacement z");
    CHECK_MSG(Rel(Pk[0], P(0, 0)) <= 1e-7, name + " P_11");
    CHECK_MSG(std::abs(Pk[4]) + std::abs(Pk[8]) <= 1e-7 * P(0, 0), name + " lateral P vanishes");
    CHECK_MSG(Rel(sig[0], sigma11) <= 1e-7, name + " sigma_11");
    CHECK_MSG(Rel(Jp[0], J) <= 1e-8, name + " J");
  }
}

// ---------------------------------------------------------- Kirsch's plate

void KirschTest()
{
  cmf::AppConfig cfg = Load("kirsch_plate_with_hole.yaml");
  cfg.output.fields = {"displacement", "cauchy_stress"};
  const double s0 = cmf::Expression::Parse(cfg.bcs.traction.at(0).expression.at(0)).Eval(0, 0, 0, 1);
  const double a = 1.0;
  std::unique_ptr<Solved> s = Solve(cfg);
  CHECK_MSG(s->report.converged, "Kirsch plate converged");
  // Along the y axis (theta = pi/2): sigma_xx = sigma_thetatheta, sigma_yy = sigma_rr.
  auto kirsch_y = [&](double y, double &sxx, double &syy)
  {
    const double q = a * a / (y * y), q2 = q * q;
    sxx = s0 * (1.0 + 0.5 * q + 1.5 * q2);
    syy = s0 * 1.5 * (q - q2);
  };
  // Along the x axis (theta = 0): sigma_xx = sigma_rr, sigma_yy = sigma_thetatheta.
  auto kirsch_x = [&](double x, double &sxx, double &syy)
  {
    const double q = a * a / (x * x), q2 = q * q;
    sxx = s0 * (1.0 - 2.5 * q + 1.5 * q2);
    syy = s0 * 0.5 * (q - 3.0 * q2);
  };
  for (const cmf::ProbeConfig &probe : cfg.output.probes)
  {
    const std::vector<double> sig = s->Probe("cauchy_stress", probe.point); // xx yy zz xy yz xz
    double sxx = 0.0, syy = 0.0;
    // Probes sit a hair off the symmetry lines (points on curved boundary
    // edges are not found); the axis formula of the nearer axis applies.
    if (probe.point[1] > probe.point[0]) { kirsch_y(probe.point[1], sxx, syy); }
    else { kirsch_x(probe.point[0], sxx, syy); }
    std::printf("  Kirsch %s (%.3f, %.3f): sigma_xx/s0 %.4f (Kirsch %.4f), sigma_yy/s0 %.4f (%.4f), "
                "sigma_xy/s0 %.1e\n", probe.name.c_str(), probe.point[0], probe.point[1],
                sig[0] / s0, sxx / s0, sig[1] / s0, syy / s0, sig[3] / s0);
    // The finite plate (W = 10 a) and the mesh add about a percent.
    CHECK_MSG(std::abs(sig[0] - sxx) <= 0.02 * 3.0 * s0, "Kirsch sigma_xx at " + probe.name);
    CHECK_MSG(std::abs(sig[1] - syy) <= 0.02 * 3.0 * s0, "Kirsch sigma_yy at " + probe.name);
    CHECK_MSG(std::abs(sig[3]) <= 0.01 * 3.0 * s0, "no shear stress on the symmetry axes at " + probe.name);
  }
}

// ------------------------------------------------------ Euler column (Southwell)

void EulerBucklingTest()
{
  cmf::AppConfig cfg = Load("euler_column_buckling.yaml");
  cfg.output.fields = {"displacement"};
  cfg.output.probe_every_step = false;
  const double L = 20.0, w = 1.0, I = w * w * w * w / 12.0, A = w * w;
  const double eps_cr = 4.0 * M_PI * M_PI * I / (A * L * L); // clamped-clamped Euler strain
  const double eps_max = 0.12 / L;
  std::vector<double> inv_eps, inv_u;
  std::unique_ptr<Solved> holder;
  cmf::SolidProblem *problem = nullptr;
  {
    cfg.output.paraview.clear();
    cfg.solver.newton.print_level = 0;
    holder = std::make_unique<Solved>();
    holder->mesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    holder->problem = cmf::MakeSolidProblem(*holder->mesh, cfg);
    problem = holder->problem.get();
    problem->Finalize();
    std::unique_ptr<mfem::Solver> linear = problem->MakeLinearSolver(cfg.solver.linear);
    holder->x.SetSize(problem->Height());
    holder->x = 0.0;
    holder->report = cmf::SolveQuasiStatic(*problem, *linear, cfg.solver, holder->x,
      [&](const cmf::LoadStepReport &step, const mfem::Vector &x)
      {
        problem->UpdateFields(x);
        const std::vector<double> u = cmf::ProbeVector(problem->Displacement(), {0.5, 0.5, 0.5 * L});
        const double eps = eps_max * step.load_factor;
        inv_eps.push_back(1.0 / eps);
        inv_u.push_back(1.0 / std::abs(u[0]));
        std::printf("  Euler column: eps/eps_cr %.3f, mid-height deflection %.5f (u_y %.5f)\n",
                    eps / eps_cr, u[0], u[1]);
      });
  }
  CHECK_MSG(holder->report.converged, "Euler column converged");
  // Southwell: 1/u = (eps_cr/delta_0) (1/eps) - 1/delta_0, so eps_cr = -slope/intercept.
  const std::size_t n = inv_eps.size();
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (std::size_t i = 0; i < n; i++)
  {
    sx += inv_eps[i]; sy += inv_u[i]; sxx += inv_eps[i] * inv_eps[i]; sxy += inv_eps[i] * inv_u[i];
  }
  const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  const double intercept = (sy - slope * sx) / n;
  const double eps_fit = -slope / intercept;
  const double delta0 = -1.0 / intercept;
  std::printf("  Euler column: Southwell eps_cr = %.5e (Euler %.5e, rel %.3f), delta_0 = %.5f (imperfection 0.005)\n",
              eps_fit, eps_cr, Rel(eps_fit, eps_cr), delta0);
  CHECK_MSG(Rel(eps_fit, eps_cr) <= 0.05, "Southwell critical strain within 5% of Euler");
  CHECK_MSG(Rel(delta0, 0.005) <= 0.15, "Southwell imperfection amplitude near the geometric one");
}

} // namespace

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  CompressibleUniaxialTest("compressible_uniaxial_neo_hookean.yaml");
  CompressibleUniaxialTest("compressible_uniaxial_st_venant_kirchhoff.yaml");
  CompressibleUniaxialTest("compressible_uniaxial_iso_neo_hookean_mixed.yaml");
  KirschTest();
  EulerBucklingTest();
  ManufacturedTests();
  SphereInflationTest();
  RivlinTorsionTest();
  return cmf_test::Report("test_verification");
}

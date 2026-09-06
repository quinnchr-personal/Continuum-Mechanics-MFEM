// S2 gate: stress-free reference state, AD tangent vs finite differences,
// objectivity under random rotations, and the small-strain (linear
// elasticity) limit for NeoHookean and StVenantKirchhoff.
#include <random>
#include <string>

#include "materials/materials.hpp"
#include "test_util.hpp"

using cmf::dual;
using cmf::tensor;

namespace
{

using Mat3 = tensor<double, 3, 3>;
using Tan4 = tensor<double, 3, 3, 3, 3>;

double MaxAbs(const Mat3 &A)
{
  double m = 0.0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) { m = std::max(m, std::abs(A(i, j))); }
  return m;
}

double MaxAbs(const Tan4 &A)
{
  double m = 0.0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++)
        for (int l = 0; l < 3; l++) { m = std::max(m, std::abs(A(i, j, k, l))); }
  return m;
}

Tan4 Subtract(const Tan4 &A, const Tan4 &B)
{
  Tan4 C;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++)
        for (int l = 0; l < 3; l++) { C(i, j, k, l) = A(i, j, k, l) - B(i, j, k, l); }
  return C;
}

// Random F = I + 0.4 R with det F > 0.3, deterministic sequence.
Mat3 RandomF(std::mt19937 &rng)
{
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  for (;;)
  {
    Mat3 F = cmf::I<3>();
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) { F(i, j) += 0.4 * unit(rng); }
    if (det(F) > 0.3) { return F; }
  }
}

// Random proper rotation from a random axis and angle (Rodrigues).
Mat3 RandomRotation(std::mt19937 &rng)
{
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  tensor<double, 3> n;
  n(0) = unit(rng); n(1) = unit(rng); n(2) = unit(rng);
  n = n / std::sqrt(cmf::dot(n, n));
  const double theta = 3.0 * unit(rng);
  Mat3 K;
  K(0, 1) = -n(2); K(0, 2) = n(1);
  K(1, 0) = n(2);  K(1, 2) = -n(0);
  K(2, 0) = -n(1); K(2, 1) = n(0);
  return cmf::I<3>() + std::sin(theta) * K + (1.0 - std::cos(theta)) * (K * K);
}

template <typename Material>
Tan4 FiniteDifferenceTangent(const Material &m, const Mat3 &F, double h)
{
  Tan4 A;
  for (int k = 0; k < 3; k++)
    for (int l = 0; l < 3; l++)
    {
      Mat3 Fp = F, Fm = F;
      Fp(k, l) += h;
      Fm(k, l) -= h;
      const Mat3 Pp = m.PK1(Fp), Pm = m.PK1(Fm);
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
          A(i, j, k, l) = (Pp(i, j) - Pm(i, j)) / (2.0 * h);
        }
    }
  return A;
}

template <typename Material>
void TestMaterial(const Material &m, const std::string &name, double E, double nu)
{
  std::cout << "material " << name << " (mu " << m.mu << ", lambda " << m.lambda << ")" << std::endl;
  const double scale = m.mu + m.lambda;

  // 1. Stress-free reference configuration, to machine precision.
  const Mat3 P0 = m.PK1(cmf::I<3>());
  CHECK_MSG(MaxAbs(P0) <= 1e-15 * scale, name + ": P(I) = " + std::to_string(MaxAbs(P0)));

  std::mt19937 rng(20240905u);
  double worst_tangent = 0.0, worst_objectivity = 0.0, worst_energy = 0.0;
  for (int trial = 0; trial < 5; trial++)
  {
    const Mat3 F = RandomF(rng);

    // 2. AD tangent vs central finite differences of PK1 (step 1e-6).
    const Tan4 A_ad = cmf::MaterialTangent(m, F);
    const Tan4 A_fd = FiniteDifferenceTangent(m, F, 1e-6);
    const double rel = MaxAbs(Subtract(A_ad, A_fd)) / MaxAbs(A_ad);
    worst_tangent = std::max(worst_tangent, rel);
    CHECK_MSG(rel <= 1e-6, name + " trial " + std::to_string(trial) +
              ": tangent AD vs FD relative error " + std::to_string(rel));

    // 2b. The plane-strain seeding (dim = 2) matches the in-plane block.
    const Tan4 A_2d = cmf::MaterialTangent(m, F, 2);
    double block_err = 0.0;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 2; k++)
          for (int l = 0; l < 2; l++)
          {
            block_err = std::max(block_err, std::abs(A_2d(i, j, k, l) - A_ad(i, j, k, l)));
          }
    CHECK_MSG(block_err == 0.0, name + ": dim=2 seeding differs from dim=3 block");

    // 2c. Tangent is the second derivative of the stored energy: P = dW/dF.
    tensor<dual, 3, 3> Fd;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) { Fd(i, j) = dual(F(i, j), 0.0); }
    const Mat3 P = m.PK1(F);
    double energy_err = 0.0;
    for (int k = 0; k < 3; k++)
      for (int l = 0; l < 3; l++)
      {
        Fd(k, l).d = 1.0;
        const dual W = m.Energy(Fd);
        Fd(k, l).d = 0.0;
        energy_err = std::max(energy_err, std::abs(W.d - P(k, l)));
      }
    worst_energy = std::max(worst_energy, energy_err / scale);
    CHECK_MSG(energy_err <= 1e-12 * scale, name + ": dW/dF vs PK1 error " +
              std::to_string(energy_err));

    // 3. Objectivity: P(QF) = Q P(F).
    const Mat3 Q = RandomRotation(rng);
    CHECK_MSG(MaxAbs(cmf::transpose(Q) * Q - cmf::I<3>()) <= 1e-14, "Q is orthogonal");
    const Mat3 lhs = m.PK1(Q * F);
    const Mat3 rhs = Q * P;
    const double obj = MaxAbs(lhs - rhs) / MaxAbs(P);
    worst_objectivity = std::max(worst_objectivity, obj);
    CHECK_MSG(obj <= 1e-12, name + " trial " + std::to_string(trial) +
              ": objectivity relative error " + std::to_string(obj));
  }

  std::printf("  P(I) max %.2e, tangent AD-vs-FD rel %.2e, dW/dF rel %.2e, objectivity rel %.2e\n",
              MaxAbs(P0), worst_tangent, worst_energy, worst_objectivity);

  // 4. Small-strain limit: F = I + eps H, P ~ lambda tr(e) I + 2 mu e, e = sym(H) eps.
  {
    Mat3 H;
    H(0, 0) = 0.3; H(0, 1) = -0.8; H(0, 2) = 0.2;
    H(1, 0) = 0.5; H(1, 1) = 0.1; H(1, 2) = 0.7;
    H(2, 0) = -0.4; H(2, 1) = 0.6; H(2, 2) = -0.2;
    const double eps = 1e-7;
    const Mat3 F = cmf::I<3>() + eps * H;
    const Mat3 P = m.PK1(F);
    const Mat3 e = eps * cmf::sym(H);
    const Mat3 P_lin = (m.lambda * cmf::tr(e)) * cmf::I<3>() + (2.0 * m.mu) * e;
    const double rel = MaxAbs(P - P_lin) / MaxAbs(P_lin);
    std::printf("  small-strain limit rel %.2e (eps %.0e)\n", rel, eps);
    CHECK_MSG(rel <= 1e-6, name + ": small-strain limit relative error " + std::to_string(rel));
    // Uniaxial strain: P_11 must equal the P-wave modulus E(1-nu)/((1+nu)(1-2nu)) eps
    // computed from E and nu directly, so a wrong Lame conversion cannot hide.
    Mat3 Fu = cmf::I<3>();
    Fu(0, 0) += eps;
    const Mat3 Pu = m.PK1(Fu);
    const double M = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
    CHECK_MSG(std::abs(Pu(0, 0) - M * eps) <= 1e-6 * M * eps,
              name + ": uniaxial P11 vs P-wave modulus");
    CHECK_MSG(std::abs(Pu(1, 1) - m.lambda * eps) <= 1e-6 * m.lambda * eps,
              name + ": uniaxial P22 vs lambda");
    // Simple shear: P_12 = mu gamma.
    Mat3 Fs = cmf::I<3>();
    Fs(0, 1) += eps;
    const Mat3 Ps = m.PK1(Fs);
    CHECK_MSG(std::abs(Ps(0, 1) - m.mu * eps) <= 1e-6 * m.mu * eps,
              name + ": simple shear P12 vs mu");
  }
}

// Decoupled (isochoric + volumetric) models: P = dW/dF, P_iso = dW_iso/dF,
// isochoric stress is deviatoric (tr(P_iso F^T) = 0), objectivity, AD
// tangent vs FD, and the small-strain limit with lambda = kappa - 2 mu / 3.
template <typename Material>
void TestDecoupled(const Material &m, const std::string &name, double mu, double kappa)
{
  std::cout << "material " << name << " (mu " << mu << ", kappa " << kappa << ")" << std::endl;
  const double scale = mu + kappa;
  const Mat3 P0 = m.PK1(cmf::I<3>());
  CHECK_MSG(MaxAbs(P0) <= 1e-15 * scale, name + ": P(I) = " + std::to_string(MaxAbs(P0)));

  std::mt19937 rng(77u);
  double worst_tangent = 0.0, worst_dev = 0.0, worst_energy = 0.0, worst_obj = 0.0;
  for (int trial = 0; trial < 5; trial++)
  {
    const Mat3 F = RandomF(rng);
    const Mat3 P = m.PK1(F);
    const Mat3 Piso = m.PK1Iso(F);

    // Isochoric stress is deviatoric in the Kirchhoff sense.
    const double dev = std::abs(cmf::tr(Piso * cmf::transpose(F))) / MaxAbs(Piso);
    worst_dev = std::max(worst_dev, dev);
    CHECK_MSG(dev <= 1e-12, name + ": tr(P_iso F^T) relative " + std::to_string(dev));

    // P_iso = dW_iso/dF and P = dW/dF via duals.
    tensor<dual, 3, 3> Fd;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) { Fd(i, j) = dual(F(i, j), 0.0); }
    double err = 0.0;
    for (int k = 0; k < 3; k++)
      for (int l = 0; l < 3; l++)
      {
        Fd(k, l).d = 1.0;
        const dual Wiso = m.EnergyIso(Fd);
        const dual W = m.Energy(Fd);
        Fd(k, l).d = 0.0;
        err = std::max(err, std::abs(Wiso.d - Piso(k, l)) / mu);
        err = std::max(err, std::abs(W.d - P(k, l)) / scale);
      }
    worst_energy = std::max(worst_energy, err);
    CHECK_MSG(err <= 1e-12, name + ": dW/dF vs PK1 relative error " + std::to_string(err));

    const Tan4 A_ad = cmf::MaterialTangent(m, F);
    const Tan4 A_fd = FiniteDifferenceTangent(m, F, 1e-6);
    const double rel = MaxAbs(Subtract(A_ad, A_fd)) / MaxAbs(A_ad);
    worst_tangent = std::max(worst_tangent, rel);
    CHECK_MSG(rel <= 1e-6, name + ": tangent AD vs FD relative error " + std::to_string(rel));

    const Mat3 Q = RandomRotation(rng);
    const double obj = MaxAbs(m.PK1(Q * F) - Q * P) / MaxAbs(P);
    const double obj_iso = MaxAbs(m.PK1Iso(Q * F) - Q * Piso) / MaxAbs(Piso);
    worst_obj = std::max(worst_obj, std::max(obj, obj_iso));
    CHECK_MSG(obj <= 1e-12 && obj_iso <= 1e-12, name + ": objectivity");
  }
  std::printf("  P(I) max %.2e, tr(P_iso F^T) rel %.2e, dW/dF rel %.2e, tangent rel %.2e, objectivity rel %.2e\n",
              MaxAbs(P0), worst_dev, worst_energy, worst_tangent, worst_obj);

  // Small-strain limit: lambda = kappa - 2 mu / 3.
  {
    Mat3 H;
    H(0, 0) = 0.3; H(0, 1) = -0.8; H(0, 2) = 0.2;
    H(1, 0) = 0.5; H(1, 1) = 0.1; H(1, 2) = 0.7;
    H(2, 0) = -0.4; H(2, 1) = 0.6; H(2, 2) = -0.2;
    const double eps = 1e-7;
    const double lambda = kappa - 2.0 * mu / 3.0;
    const Mat3 P = m.PK1(cmf::I<3>() + eps * H);
    const Mat3 e = eps * cmf::sym(H);
    const Mat3 P_lin = (lambda * cmf::tr(e)) * cmf::I<3>() + (2.0 * mu) * e;
    const double rel = MaxAbs(P - P_lin) / MaxAbs(P_lin);
    std::printf("  small-strain limit rel %.2e (eps %.0e)\n", rel, eps);
    CHECK_MSG(rel <= 1e-6, name + ": small-strain limit relative error " + std::to_string(rel));
    // Pure dilatation: P_11 = kappa tr(e) + 2 mu (e_11 - tr(e)/3) with e = eps I.
    const Mat3 Pv = m.PK1((1.0 + eps) * cmf::I<3>());
    CHECK_MSG(std::abs(Pv(0, 0) - 3.0 * kappa * eps) <= 1e-5 * 3.0 * kappa * eps,
              name + ": dilatation P11 vs 3 kappa eps");
  }
}

void TestModuliResolution()
{
  cmf::MaterialConfig c;
  c.model = "iso_neo_hookean"; c.E = 250.0; c.nu = 0.3;
  cmf::ResolvedModuli r = cmf::ResolveModuli(c);
  CHECK_CLOSE(r.mu, 250.0 / 2.6, 1e-12);
  CHECK_CLOSE(r.kappa, 250.0 / (3.0 * 0.4), 1e-12);
  CHECK_CLOSE(r.lambda, cmf::LameFromYoungPoisson(250.0, 0.3).lambda, 1e-10);
  c = cmf::MaterialConfig();
  c.model = "iso_neo_hookean"; c.mu = 80.0; c.nu = 0.5;
  r = cmf::ResolveModuli(c);
  CHECK(r.incompressible && std::isinf(r.kappa) && r.mu == 80.0);
  c = cmf::MaterialConfig();
  c.model = "mooney_rivlin"; c.c1 = 30.0; c.c2 = 10.0; c.kappa = 1000.0;
  r = cmf::ResolveModuli(c);
  CHECK_CLOSE(r.mu, 80.0, 0.0);
  CHECK_CLOSE(r.kappa, 1000.0, 0.0);
  c = cmf::MaterialConfig();
  c.model = "mooney_rivlin"; c.c1 = 30.0; c.c2 = 10.0; c.incompressible = true;
  CHECK(cmf::ResolveModuli(c).incompressible);
  CHECK_THROWS(cmf::MakeMaterial(c), cmf::ConfigError, "formulation: mixed");
  CHECK(cmf::MaterialName(cmf::MakeMixedMaterial(c)) == "mooney_rivlin");
  c = cmf::MaterialConfig();
  c.model = "iso_neo_hookean"; c.mu = 80.0;
  CHECK_THROWS(cmf::ResolveModuli(c), cmf::ConfigError, "needs exactly one of");
  c.nu = 0.3; c.kappa = 100.0;
  CHECK_THROWS(cmf::ResolveModuli(c), cmf::ConfigError, "needs exactly one of");
  c = cmf::MaterialConfig();
  c.model = "neo_hookean"; c.E = 1.0; c.nu = 0.3; c.kappa = 5.0;
  CHECK_THROWS(cmf::ResolveModuli(c), cmf::ConfigError, "'material.kappa' is not used");
  c = cmf::MaterialConfig();
  c.model = "neo_hookean"; c.E = 1.0;
  CHECK_THROWS(cmf::ResolveModuli(c), cmf::ConfigError, "missing key 'material.nu'");
  c = cmf::MaterialConfig();
  c.model = "neo_hookean"; c.E = 1.0; c.nu = 0.3;
  CHECK_THROWS(cmf::MakeMixedMaterial(c), cmf::ConfigError, "formulation: mixed needs");
}

} // namespace

int main()
{
  const double E = 250.0, nu = 0.3;
  const cmf::LameParameters lame = cmf::LameFromYoungPoisson(E, nu);
  CHECK_CLOSE(lame.mu, 250.0 / 2.6, 1e-12);
  CHECK_CLOSE(lame.lambda, 250.0 * 0.3 / (1.3 * 0.4), 1e-12);

  cmf::MaterialConfig cfg;
  cfg.E = E; cfg.nu = nu;
  cfg.model = "neo_hookean";
  const cmf::Material nh = cmf::MakeMaterial(cfg);
  CHECK(cmf::MaterialName(nh) == "neo_hookean");
  cfg.model = "st_venant_kirchhoff";
  const cmf::Material svk = cmf::MakeMaterial(cfg);
  CHECK(cmf::MaterialName(svk) == "st_venant_kirchhoff");

  TestMaterial(std::get<cmf::NeoHookean>(nh), "neo_hookean", E, nu);
  TestMaterial(std::get<cmf::StVenantKirchhoff>(svk), "st_venant_kirchhoff", E, nu);

  // Decoupled models at nu = 0.45 (kappa/mu ~ 9.7).
  const double mu = lame.mu, kappa = 2.0 * mu * 1.45 / (3.0 * 0.1);
  TestDecoupled(cmf::IsoNeoHookean(mu, kappa), "iso_neo_hookean", mu, kappa);
  TestDecoupled(cmf::MooneyRivlin(0.3 * mu, 0.2 * mu, kappa), "mooney_rivlin", mu, kappa);
  // Mooney-Rivlin with c2 = 0 is the isochoric neo-Hookean model.
  {
    const cmf::MooneyRivlin mr(0.5 * mu, 0.0, kappa);
    const cmf::IsoNeoHookean nhi(mu, kappa);
    std::mt19937 rng(5u);
    const Mat3 F = RandomF(rng);
    CHECK_MSG(MaxAbs(mr.PK1(F) - nhi.PK1(F)) <= 1e-12 * MaxAbs(nhi.PK1(F)),
              "mooney_rivlin(c2 = 0) equals iso_neo_hookean");
  }
  TestModuliResolution();
  return cmf_test::Report("test_materials");
}

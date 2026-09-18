// S2 gate: stress-free reference state, AD tangent vs finite differences,
// objectivity under random rotations, and the small-strain (linear
// elasticity) limit for every material; Ogden reductions and its analytic
// tangent at coincident principal stretches; the small-strain model
// linear_elastic, which is that limit at any amplitude.
#include <algorithm>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

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

// What TestMaterial expects of a model. A small-strain model is not objective
// under finite rotations, and it equals its own small-strain limit at any
// amplitude: there the limit check runs at a finite strain, above the
// 1e-16 / |Grad u| cancellation floor of eps = sym(F - I), to round-off.
struct MaterialChecks
{
  bool objective = true;
  double limit_eps = 1e-7;
  double limit_tol = 1e-6;
};

template <typename Material>
void TestMaterial(const Material &m, const std::string &name, double E, double nu,
                  const MaterialChecks &checks = MaterialChecks())
{
  // The small-strain Lame moduli follow from E and nu, not from the material's
  // members, so models parameterised otherwise run the same checks.
  const cmf::LameParameters lame = cmf::LameFromYoungPoisson(E, nu);
  std::cout << "material " << name << " (mu " << lame.mu << ", lambda " << lame.lambda << ")" << std::endl;
  const double scale = lame.mu + lame.lambda;

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
    if (!checks.objective) { continue; }
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
    const double eps = checks.limit_eps, tol = checks.limit_tol;
    const Mat3 F = cmf::I<3>() + eps * H;
    const Mat3 P = m.PK1(F);
    const Mat3 e = eps * cmf::sym(H);
    const Mat3 P_lin = (lame.lambda * cmf::tr(e)) * cmf::I<3>() + (2.0 * lame.mu) * e;
    const double rel = MaxAbs(P - P_lin) / MaxAbs(P_lin);
    std::printf("  small-strain limit rel %.2e (eps %.0e)\n", rel, eps);
    CHECK_MSG(rel <= tol, name + ": small-strain limit relative error " + std::to_string(rel));
    // Uniaxial strain: P_11 must equal the P-wave modulus E(1-nu)/((1+nu)(1-2nu)) eps
    // computed from E and nu directly, so a wrong Lame conversion cannot hide.
    Mat3 Fu = cmf::I<3>();
    Fu(0, 0) += eps;
    const Mat3 Pu = m.PK1(Fu);
    const double M = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
    CHECK_MSG(std::abs(Pu(0, 0) - M * eps) <= tol * M * eps,
              name + ": uniaxial P11 vs P-wave modulus");
    CHECK_MSG(std::abs(Pu(1, 1) - lame.lambda * eps) <= tol * lame.lambda * eps,
              name + ": uniaxial P22 vs lambda");
    // Simple shear: P_12 = mu gamma.
    Mat3 Fs = cmf::I<3>();
    Fs(0, 1) += eps;
    const Mat3 Ps = m.PK1(Fs);
    CHECK_MSG(std::abs(Ps(0, 1) - lame.mu * eps) <= tol * lame.mu * eps,
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

// Volumetric laws (materials/volumetric.hpp): every law has u(1) = u'(1) = 0
// and u''(1) = 1, u' is the derivative of u and u'' of u' (duals vs central
// differences), and the decoupled models pass TestDecoupled with each law
// (tangent, dW/dF = P, small-strain limit with the same kappa).
void TestVolumetricLaws(double mu, double kappa)
{
  const cmf::VolumetricLaw laws[] = {cmf::VolumetricLaw::Quadratic, cmf::VolumetricLaw::SimoTaylor,
                                     cmf::VolumetricLaw::Logarithmic, cmf::VolumetricLaw::JLogJ};
  for (cmf::VolumetricLaw law : laws)
  {
    const std::string name = cmf::VolumetricLawName(law);
    CHECK(cmf::ParseVolumetricLaw(name) == law);
    CHECK_MSG(std::abs(cmf::NormalizedVolumetricEnergy(law, 1.0)) <= 1e-15, name + ": u(1) = 0");
    CHECK_MSG(std::abs(cmf::NormalizedVolumetricPressure(law, 1.0)) <= 1e-15, name + ": u'(1) = 0");
    CHECK_MSG(std::abs(cmf::NormalizedVolumetricModulus(law, 1.0) - 1.0) <= 1e-15, name + ": u''(1) = 1");
    double worst = 0.0;
    for (double J : {0.5, 0.8, 0.97, 1.03, 1.4, 2.5})
    {
      const double h = 1e-6;
      const double up_fd = (cmf::NormalizedVolumetricEnergy(law, J + h) -
                            cmf::NormalizedVolumetricEnergy(law, J - h)) / (2.0 * h);
      const double upp_fd = (cmf::NormalizedVolumetricPressure(law, J + h) -
                             cmf::NormalizedVolumetricPressure(law, J - h)) / (2.0 * h);
      const dual u = cmf::NormalizedVolumetricEnergy(law, dual(J, 1.0));
      const dual up = cmf::NormalizedVolumetricPressure(law, dual(J, 1.0));
      worst = std::max({worst, std::abs(up_fd - up.v), std::abs(u.d - up.v),
                        std::abs(upp_fd - cmf::NormalizedVolumetricModulus(law, J)),
                        std::abs(up.d - cmf::NormalizedVolumetricModulus(law, J))});
    }
    std::printf("  volumetric law %s: derivative consistency %.2e\n", name.c_str(), worst);
    CHECK_MSG(worst <= 1e-8, name + ": u' = du/dJ and u'' = du'/dJ");
    // Legendre transform: u*(pi) = pi (J - 1) - u(J) at u'(J) = pi, and
    // u*(pi) + u(J) = pi (J - 1) (Fenchel equality) for J in the invertible range.
    for (double J : {0.7, 0.95, 1.1, 1.8})
    {
      const double pi = cmf::NormalizedVolumetricPressure(law, J);
      const double ustar = cmf::NormalizedComplementaryEnergy(law, pi);
      CHECK_MSG(std::abs(ustar + cmf::NormalizedVolumetricEnergy(law, J) - pi * (J - 1.0)) <= 1e-12,
                name + ": Legendre transform at J = " + std::to_string(J));
    }
    CHECK_MSG(std::abs(cmf::NormalizedComplementaryEnergy(law, 0.0)) <= 1e-15, name + ": u*(0) = 0");
    if (law == cmf::VolumetricLaw::Quadratic) { continue; }
    cmf::IsoNeoHookean nh(mu, kappa);
    nh.law = law;
    TestDecoupled(nh, "iso_neo_hookean/" + name, mu, kappa);
    cmf::Ogden og({0.63 * mu, 0.0012 * mu, -0.01 * mu}, {1.3, 5.0, -2.0}, kappa);
    og.law = law;
    TestDecoupled(og, "ogden/" + name, og.ShearModulus(), kappa);
  }
  // The factory applies the law; the coupled models refuse the key.
  cmf::MaterialConfig c;
  c.model = "iso_neo_hookean"; c.mu = mu; c.kappa = kappa; c.volumetric = "logarithmic";
  const cmf::MixedMaterial m = cmf::MakeMixedMaterial(c);
  CHECK(std::get<cmf::IsoNeoHookean>(m).law == cmf::VolumetricLaw::Logarithmic);
  CHECK(cmf::VolumetricLawSuffix(m) == ", logarithmic volumetric law");
  CHECK(cmf::VolumetricLawSuffix(cmf::MakeMaterial(c)) == ", logarithmic volumetric law");
  c.volumetric = "quadratic";
  CHECK(cmf::VolumetricLawSuffix(cmf::MakeMixedMaterial(c)).empty());
  c.volumetric = "cubic";
  CHECK_THROWS(cmf::MakeMixedMaterial(c), cmf::ConfigError, "unknown volumetric law 'cubic'");
  c.model = "neo_hookean"; c.E = 1.0; c.nu = 0.3; c.mu = c.kappa = std::numeric_limits<double>::quiet_NaN();
  c.volumetric = "logarithmic";
  CHECK_THROWS(cmf::MakeMaterial(c), cmf::ConfigError, "'material.volumetric' is not used");
}

// Ogden reduces to the isochoric neo-Hookean model for (mu, alpha) = (mu, 2)
// and to Mooney-Rivlin for (2 c1, 2) + (-2 c2, -2); its analytic tangent must
// match finite differences also at coincident principal stretches, where
// the spectral formula takes its limit branch.
void TestOgdenSpecial(double mu, double kappa)
{
  std::cout << "ogden special cases" << std::endl;
  std::mt19937 rng(9u);
  const cmf::IsoNeoHookean nh(mu, kappa);
  const cmf::Ogden og_nh({mu}, {2.0}, kappa);
  const cmf::MooneyRivlin mr(0.3 * mu, 0.2 * mu, kappa);
  const cmf::Ogden og_mr({0.6 * mu, -0.4 * mu}, {2.0, -2.0}, kappa);
  double worst = 0.0;
  for (int trial = 0; trial < 3; trial++)
  {
    const Mat3 F = RandomF(rng);
    const double e1 = MaxAbs(og_nh.PK1(F) - nh.PK1(F)) / MaxAbs(nh.PK1(F));
    const double e2 = std::abs(og_nh.Energy(F) - nh.Energy(F)) / (mu + kappa);
    const double e3 = MaxAbs(og_mr.PK1(F) - mr.PK1(F)) / MaxAbs(mr.PK1(F));
    const double e4 = MaxAbs(Subtract(cmf::MaterialTangent(og_nh, F), cmf::MaterialTangent(nh, F))) /
                      MaxAbs(cmf::MaterialTangent(nh, F));
    const double e5 = MaxAbs(Subtract(cmf::MaterialTangent(og_mr, F), cmf::MaterialTangent(mr, F))) /
                      MaxAbs(cmf::MaterialTangent(mr, F));
    worst = std::max({worst, e1, e2, e3, e4, e5});
    CHECK_MSG(e1 <= 1e-12 && e2 <= 1e-12, "ogden(mu, 2) equals iso_neo_hookean");
    CHECK_MSG(e3 <= 1e-12, "ogden(2 c1, 2; -2 c2, -2) equals mooney_rivlin");
    CHECK_MSG(e4 <= 1e-10 && e5 <= 1e-10, "ogden tangent equals the dual-number tangents of the reductions");
  }
  std::printf("  reductions to iso_neo_hookean / mooney_rivlin: worst rel %.2e\n", worst);

  const cmf::Ogden og({0.63 * mu, 0.0012 * mu, -0.01 * mu}, {1.3, 5.0, -2.0}, kappa);
  std::vector<std::pair<std::string, Mat3>> cases;
  Mat3 uni, dil, ps;
  uni(0, 0) = 1.3; uni(1, 1) = uni(2, 2) = 1.0 / std::sqrt(1.3);
  dil(0, 0) = dil(1, 1) = dil(2, 2) = 1.1;
  ps(0, 0) = 1.25; ps(1, 1) = 0.8; ps(2, 2) = 1.0;
  cases.push_back({"uniaxial (two equal stretches)", uni});
  cases.push_back({"dilatation (three equal stretches)", dil});
  cases.push_back({"plane strain (distinct stretches)", ps});
  cases.push_back({"identity", cmf::I<3>()});
  for (const auto &kv : cases)
  {
    for (int rotated = 0; rotated < 2; rotated++)
    {
      const Mat3 F = rotated ? RandomRotation(rng) * kv.second * RandomRotation(rng) : kv.second;
      const Tan4 A_ad = cmf::MaterialTangent(og, F);
      const Tan4 A_fd = FiniteDifferenceTangent(og, F, 1e-6);
      const double rel = MaxAbs(Subtract(A_ad, A_fd)) / MaxAbs(A_ad);
      std::printf("  tangent vs FD, %s%s: rel %.2e\n", kv.first.c_str(), rotated ? ", rotated" : "", rel);
      CHECK_MSG(rel <= 1e-6, "ogden tangent at " + kv.first + " vs FD (" + std::to_string(rel) + ")");
    }
  }
}

// Random in-plane F = I + a R (2x2 block, F33 = 1) with det of the block > 0.3.
Mat3 RandomInPlaneF(std::mt19937 &rng, double amplitude)
{
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  for (;;)
  {
    Mat3 F = cmf::I<3>();
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) { F(i, j) += amplitude * unit(rng); }
    if (F(0, 0) * F(1, 1) - F(0, 1) * F(1, 0) > 0.3) { return F; }
  }
}

// Plane-stress adapter: P33 = 0 and no out-of-plane shear stress, J = 1 for
// incompressible bases, in-plane dual tangent vs finite differences (the
// compressible branch differentiates a converged Newton root), dW/dF = P in
// the plane (envelope theorem), and objectivity under in-plane rotations.
template <typename Base>
void TestPlaneStressAdapter(const Base &base, const std::string &name, double scale,
                            double amplitude)
{
  const cmf::PlaneStress<Base> m(base);
  std::cout << "plane stress " << name << (m.Incompressible() ? " (incompressible)" : "") << std::endl;
  std::mt19937 rng(31u);
  double worst_p3 = 0.0, worst_tan = 0.0, worst_energy = 0.0, worst_obj = 0.0, worst_J = 0.0;
  for (int trial = 0; trial < 4; trial++)
  {
    const Mat3 F = RandomInPlaneF(rng, amplitude);
    const Mat3 Fc = m.Complete(F);
    const Mat3 P = m.PK1(F);
    const double p3 = std::max({std::abs(P(2, 2)), std::abs(P(0, 2)), std::abs(P(1, 2)),
                                std::abs(P(2, 0)), std::abs(P(2, 1))}) / scale;
    worst_p3 = std::max(worst_p3, p3);
    CHECK_MSG(p3 <= 1e-12, name + " plane stress: out-of-plane P relative " + std::to_string(p3));
    CHECK_MSG(Fc(2, 2) > 0.0, name + " plane stress: positive thickness stretch");
    if (m.Incompressible())
    {
      worst_J = std::max(worst_J, std::abs(det(Fc) - 1.0));
      CHECK_MSG(std::abs(det(Fc) - 1.0) <= 1e-13, name + " plane stress: J = 1");
    }

    const Tan4 A_ad = cmf::MaterialTangent(m, F, 2);
    const Tan4 A_fd = FiniteDifferenceTangent(m, F, 1e-6);
    double err = 0.0, ref = 0.0;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
          for (int l = 0; l < 2; l++)
          {
            err = std::max(err, std::abs(A_ad(i, j, k, l) - A_fd(i, j, k, l)));
            ref = std::max(ref, std::abs(A_ad(i, j, k, l)));
          }
    worst_tan = std::max(worst_tan, err / ref);
    CHECK_MSG(err / ref <= 1e-6, name + " plane stress: tangent vs FD relative " + std::to_string(err / ref));

    tensor<dual, 3, 3> Fd;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) { Fd(i, j) = dual(F(i, j), 0.0); }
    double energy_err = 0.0;
    for (int k = 0; k < 2; k++)
      for (int l = 0; l < 2; l++)
      {
        Fd(k, l).d = 1.0;
        const dual W = m.Energy(Fd);
        Fd(k, l).d = 0.0;
        energy_err = std::max(energy_err, std::abs(W.d - P(k, l)) / scale);
      }
    worst_energy = std::max(worst_energy, energy_err);
    CHECK_MSG(energy_err <= 1e-10, name + " plane stress: dW/dF vs P relative " + std::to_string(energy_err));

    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    const double theta = 3.0 * unit(rng);
    Mat3 Q = cmf::I<3>();
    Q(0, 0) = std::cos(theta); Q(0, 1) = -std::sin(theta);
    Q(1, 0) = std::sin(theta); Q(1, 1) = std::cos(theta);
    const double obj = MaxAbs(m.PK1(Q * F) - Q * P) / MaxAbs(P);
    worst_obj = std::max(worst_obj, obj);
    CHECK_MSG(obj <= 1e-10, name + " plane stress: objectivity " + std::to_string(obj));
  }
  std::printf("  out-of-plane P rel %.1e, J-1 %.1e, tangent vs FD rel %.1e, dW/dF rel %.1e, objectivity rel %.1e\n",
              worst_p3, worst_J, worst_tan, worst_energy, worst_obj);
}

// Small-strain plane stress: with F = I + eps e1 (x) e1 the in-plane stresses
// are E/(1 - nu^2) eps and E nu/(1 - nu^2) eps and the thickness strain is
// -nu/(1 - nu) eps; incompressible bases have nu = 1/2, E = 3 mu.
template <typename Base>
void TestPlaneStressSmallStrain(const Base &base, const std::string &name, double E, double nu)
{
  const cmf::PlaneStress<Base> m(base);
  const double eps = 1e-7;
  Mat3 F = cmf::I<3>();
  F(0, 0) += eps;
  const Mat3 P = m.PK1(F);
  const double s11 = E / (1.0 - nu * nu) * eps, s22 = nu * s11;
  const double e33 = -nu / (1.0 - nu) * eps;
  CHECK_MSG(std::abs(P(0, 0) - s11) <= 1e-6 * s11, name + " plane stress: P11 vs E/(1-nu^2)");
  CHECK_MSG(std::abs(P(1, 1) - s22) <= 1e-6 * s11, name + " plane stress: P22 vs E nu/(1-nu^2)");
  CHECK_MSG(std::abs(m.Complete(F)(2, 2) - 1.0 - e33) <= 1e-6 * std::abs(e33),
            name + " plane stress: thickness strain vs -nu/(1-nu) eps");
  std::printf("  small strain %s: P11/(E eps/(1-nu^2)) - 1 = %.1e, thickness strain error %.1e\n",
              name.c_str(), P(0, 0) / s11 - 1.0, std::abs(m.Complete(F)(2, 2) - 1.0 - e33) / std::abs(e33));
}

void TestPlaneStress(double E, double nu, double mu, double kappa)
{
  const cmf::LameParameters lame = cmf::LameFromYoungPoisson(E, nu);
  const double inf = std::numeric_limits<double>::infinity();
  TestPlaneStressAdapter(cmf::NeoHookean{lame.mu, lame.lambda}, "neo_hookean", E, 0.4);
  TestPlaneStressAdapter(cmf::StVenantKirchhoff{lame.mu, lame.lambda}, "st_venant_kirchhoff", E, 0.15);
  TestPlaneStressAdapter(cmf::IsoNeoHookean(mu, kappa), "iso_neo_hookean kappa", mu + kappa, 0.4);
  TestPlaneStressAdapter(cmf::IsoNeoHookean(mu, inf), "iso_neo_hookean", mu, 0.4);
  TestPlaneStressAdapter(cmf::MooneyRivlin(0.3 * mu, 0.2 * mu, inf), "mooney_rivlin", mu, 0.4);
  TestPlaneStressAdapter(cmf::Yeoh(0.5 * mu, -0.05 * mu, 0.01 * mu, inf), "yeoh", mu, 0.4);
  TestPlaneStressAdapter(cmf::Gent(mu, 20.0, inf), "gent", mu, 0.4);
  TestPlaneStressAdapter(cmf::ArrudaBoyce(mu, 5.0, inf), "arruda_boyce", mu, 0.4);
  TestPlaneStressAdapter(cmf::Ogden({0.63 * mu, 0.0012 * mu, -0.01 * mu}, {1.3, 5.0, -2.0}, inf), "ogden", mu, 0.4);
  TestPlaneStressAdapter(cmf::Ogden({0.63 * mu, 0.0012 * mu, -0.01 * mu}, {1.3, 5.0, -2.0}, kappa), "ogden kappa", mu + kappa, 0.4);
  TestPlaneStressSmallStrain(cmf::NeoHookean{lame.mu, lame.lambda}, "neo_hookean", E, nu);
  TestPlaneStressSmallStrain(cmf::IsoNeoHookean(mu, kappa), "iso_neo_hookean kappa", 9.0 * kappa * mu / (3.0 * kappa + mu), (3.0 * kappa - 2.0 * mu) / (2.0 * (3.0 * kappa + mu)));
  TestPlaneStressSmallStrain(cmf::IsoNeoHookean(mu, inf), "iso_neo_hookean", 3.0 * mu, 0.5);
  // Uniaxial tension of an incompressible neo-Hookean sheet: P11 = mu (lambda - lambda^-2), P22 = 0.
  {
    const cmf::PlaneStress<cmf::IsoNeoHookean> m(cmf::IsoNeoHookean(mu, inf));
    const double l = 1.7;
    Mat3 F = cmf::I<3>();
    F(0, 0) = l;
    F(1, 1) = 1.0 / std::sqrt(l);
    const Mat3 P = m.PK1(F);
    CHECK_MSG(std::abs(P(0, 0) - mu * (l - 1.0 / (l * l))) <= 1e-12 * mu, "sheet uniaxial P11 = mu (lambda - lambda^-2)");
    CHECK_MSG(std::abs(P(1, 1)) <= 1e-12 * mu, "sheet uniaxial P22 = 0");
    CHECK_MSG(std::abs(m.Complete(F)(2, 2) - 1.0 / std::sqrt(l)) <= 1e-14, "sheet uniaxial thickness lambda^-1/2");
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
  CHECK(cmf::MaterialName(cmf::MakeMaterial(c, true)) == "mooney_rivlin (plane stress)");
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

// gent_compressible_summit against a transcription of SUMMIT's
// GentCompressibleHyperelastic::Constitutive (row-major F and F^-1, its
// coefficients coef_0..coef_3 of the Lagrangian moduli): stress, stored
// energy and the AD tangent; then the generic checks with the small-strain
// moduli mu and lambda = 2 mu / Jm (kappa does not enter them).
void TestGentCompressibleSummit()
{
  const double mu = 0.5, kappa = 2000.0, Jm = 60.0;
  const cmf::GentCompressibleSummit m{mu, kappa, Jm};
  std::mt19937 rng(20260918u);
  double worst_P = 0.0, worst_W = 0.0, worst_A = 0.0;
  for (int trial = 0; trial < 5; trial++)
  {
    const Mat3 F = RandomF(rng);
    const Mat3 Finv = cmf::inv(F);
    const double detF = cmf::det(F);
    const double Jsq_minus_1 = detF * detF - 1.0, logJ = std::log(detF), trC = cmf::ddot(F, F);
    const double A = 0.5 * Jsq_minus_1 - logJ, C = Jm - trC + 3.0, B = Jm / C;
    Mat3 P_ref;
    for (int i = 0; i < 3; i++)
      for (int J = 0; J < 3; J++)
      {
        P_ref(i, J) = mu * B * F(i, J) + (2.0 * kappa * A * A * A * Jsq_minus_1 - mu) * Finv(J, i);
      }
    const double W_ref = -0.5 * mu * (Jm * std::log(1.0 - (trC - 3.0) / Jm) + 2.0 * logJ) +
                         0.5 * kappa * A * A * A * A;
    const double coef_0 = mu * B, coef_1 = 2.0 * mu * Jm / (C * C);
    const double coef_2 = mu - 2.0 * kappa * A * A * A * Jsq_minus_1;
    const double coef_3 = 2.0 * kappa * (3.0 * A * A * Jsq_minus_1 * Jsq_minus_1 + 2.0 * detF * detF * A * A * A);
    const Mat3 P = m.PK1(F);
    const Tan4 T = cmf::MaterialTangent(m, F);
    worst_P = std::max(worst_P, MaxAbs(P - P_ref) / MaxAbs(P_ref));
    worst_W = std::max(worst_W, std::abs(m.Energy(F) - W_ref) / std::abs(W_ref));
    double tmax = 0.0, terr = 0.0;
    for (int i = 0; i < 3; i++)
      for (int J = 0; J < 3; J++)
        for (int k = 0; k < 3; k++)
          for (int L = 0; L < 3; L++)
          {
            const double ref = coef_1 * F(k, L) * F(i, J) + coef_2 * Finv(J, k) * Finv(L, i) +
                               coef_3 * Finv(J, i) * Finv(L, k) + (i == k && J == L ? coef_0 : 0.0);
            tmax = std::max(tmax, std::abs(ref));
            terr = std::max(terr, std::abs(T(i, J, k, L) - ref));
          }
    worst_A = std::max(worst_A, terr / tmax);
  }
  std::printf("gent_compressible_summit vs SUMMIT's expressions: P rel %.2e, W rel %.2e, tangent rel %.2e\n",
              worst_P, worst_W, worst_A);
  CHECK_MSG(worst_P <= 1e-13, "gent_compressible_summit: PK1 matches SUMMIT's expression");
  CHECK_MSG(worst_W <= 1e-13, "gent_compressible_summit: energy matches SUMMIT's expression");
  CHECK_MSG(worst_A <= 1e-12, "gent_compressible_summit: AD tangent matches SUMMIT's moduli");

  const double lambda = m.SmallStrainLambda();
  CHECK_CLOSE(lambda, 2.0 * mu / Jm, 1e-15);
  TestMaterial(m, "gent_compressible_summit", mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu),
               lambda / (2.0 * (lambda + mu)));

  cmf::MaterialConfig cfg;
  cfg.model = "gent_compressible_summit";
  cfg.mu = mu; cfg.kappa = kappa; cfg.Jm = Jm;
  const cmf::Material made = cmf::MakeMaterial(cfg);
  CHECK(cmf::MaterialName(made) == "gent_compressible_summit");
  CHECK(std::get<cmf::GentCompressibleSummit>(made).kappa == kappa);
  const cmf::ResolvedModuli r = cmf::ResolveModuli(cfg);
  CHECK_CLOSE(r.lambda, 2.0 * mu / Jm, 1e-15);
  CHECK_THROWS(cmf::MakeMixedMaterial(cfg), cmf::ConfigError, "has no isochoric-volumetric split");
  cmf::MaterialConfig bad = cfg;
  bad.nu = 0.3;
  CHECK_THROWS(cmf::MakeMaterial(bad), cmf::ConfigError, "is not used by model 'gent_compressible_summit'");
  bad = cfg;
  bad.kappa = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(cmf::MakeMaterial(bad), cmf::ConfigError, "needs mu, kappa and Jm");
}

// linear_elastic (small strain): the generic checks without objectivity and
// with the limit check at a finite strain; invariance under infinitesimal
// rotations, PK1(I + W) = 0 for skew W, in its place; the AD tangent is the
// constant isotropic C with minor and major symmetries; W = sigma:eps / 2; the
// plane-stress adapter gives the plane-stress moduli at a finite strain, not
// only in the limit; the YAML factory and its key rules.
void TestLinearElastic(double E, double nu)
{
  const cmf::LameParameters lame = cmf::LameFromYoungPoisson(E, nu);
  const cmf::LinearElastic m = cmf::LinearElastic::FromYoungPoisson(E, nu);
  const double scale = lame.lambda + 2.0 * lame.mu;
  CHECK_CLOSE(m.mu, lame.mu, 1e-13 * scale);
  CHECK_CLOSE(m.Lambda(), lame.lambda, 1e-13 * scale);
  CHECK(cmf::is_small_strain<cmf::LinearElastic>::value);
  CHECK(cmf::is_small_strain<cmf::PlaneStress<cmf::LinearElastic>>::value);
  CHECK(!cmf::is_small_strain<cmf::NeoHookean>::value);
  CHECK(!cmf::is_small_strain<cmf::PlaneStress<cmf::NeoHookean>>::value);

  MaterialChecks checks;
  checks.objective = false;
  checks.limit_eps = 0.05;
  checks.limit_tol = 1e-13;
  TestMaterial(m, "linear_elastic", E, nu, checks);

  std::mt19937 rng(20260918u);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  // Infinitesimal rotation: F = I + W, W skew, is stress free.
  {
    Mat3 W;
    W(0, 1) = 0.3 * unit(rng); W(0, 2) = 0.3 * unit(rng); W(1, 2) = 0.3 * unit(rng);
    W(1, 0) = -W(0, 1); W(2, 0) = -W(0, 2); W(2, 1) = -W(1, 2);
    const Mat3 F = cmf::I<3>() + W;
    CHECK_MSG(MaxAbs(m.PK1(F)) <= 1e-15 * scale, "linear_elastic: PK1(I + W) = 0 for skew W");
    CHECK_MSG(std::abs(m.Energy(F)) <= 1e-15 * scale, "linear_elastic: W(I + W) = 0 for skew W");
  }
  // Constant tangent equal to lambda d_ij d_kl + mu (d_ik d_jl + d_il d_jk).
  double worst_C = 0.0, worst_energy = 0.0;
  for (int trial = 0; trial < 2; trial++)
  {
    const Mat3 F = RandomF(rng);
    const Tan4 A = cmf::MaterialTangent(m, F);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++)
          for (int l = 0; l < 3; l++)
          {
            const double C = lame.lambda * (i == j) * (k == l) +
                             lame.mu * ((i == k) * (j == l) + (i == l) * (j == k));
            worst_C = std::max(worst_C, std::abs(A(i, j, k, l) - C) / scale);
          }
    const Mat3 P = m.PK1(F);
    CHECK_MSG(MaxAbs(P - cmf::transpose(P)) == 0.0, "linear_elastic: the stress is symmetric");
    const double W = m.Energy(F), half = 0.5 * cmf::ddot(P, cmf::LinearElastic::Strain(F));
    worst_energy = std::max(worst_energy, std::abs(W - half) / std::abs(half));
  }
  std::printf("  tangent vs closed-form C rel %.2e, W vs sigma:eps/2 rel %.2e\n", worst_C, worst_energy);
  CHECK_MSG(worst_C <= 1e-14, "linear_elastic: AD tangent is the constant isotropic C");
  CHECK_MSG(worst_energy <= 1e-14, "linear_elastic: W = sigma:eps / 2");

  // Plane stress at a finite in-plane strain: sigma = E/(1-nu^2) [(1-nu) eps +
  // nu tr(eps) I], thickness strain -nu/(1-nu) tr(eps), and the in-plane
  // tangent lambda* d d + mu (dd + dd) with lambda* = 2 lambda mu/(lambda + 2 mu).
  {
    const cmf::PlaneStress<cmf::LinearElastic> ps(m);
    Mat3 F = cmf::I<3>();
    F(0, 0) += 0.05; F(1, 1) += -0.02; F(0, 1) += 0.06; F(1, 0) += -0.01;
    const double e11 = 0.05, e22 = -0.02, e12 = 0.025;
    const double Eps = E / (1.0 - nu * nu);
    const Mat3 P = ps.PK1(F);
    CHECK_CLOSE(P(0, 0), Eps * (e11 + nu * e22), 1e-13 * scale);
    CHECK_CLOSE(P(1, 1), Eps * (e22 + nu * e11), 1e-13 * scale);
    CHECK_CLOSE(P(0, 1), 2.0 * lame.mu * e12, 1e-13 * scale);
    CHECK_CLOSE(P(1, 0), 2.0 * lame.mu * e12, 1e-13 * scale);
    CHECK_CLOSE(P(2, 2), 0.0, 1e-13 * scale);
    CHECK_CLOSE(ps.Complete(F)(2, 2) - 1.0, -nu / (1.0 - nu) * (e11 + e22), 1e-14);
    const double lambda_ps = 2.0 * lame.lambda * lame.mu / (lame.lambda + 2.0 * lame.mu);
    const Tan4 A = cmf::MaterialTangent(ps, F, 2);
    double worst = 0.0;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
          for (int l = 0; l < 2; l++)
          {
            const double C = lambda_ps * (i == j) * (k == l) +
                             lame.mu * ((i == k) * (j == l) + (i == l) * (j == k));
            worst = std::max(worst, std::abs(A(i, j, k, l) - C) / scale);
          }
    std::printf("  plane stress at strain 0.05: in-plane tangent vs closed form rel %.2e\n", worst);
    CHECK_MSG(worst <= 1e-12, "linear_elastic plane stress: tangent is the plane-stress C");
    TestPlaneStressSmallStrain(m, "linear_elastic", E, nu);
  }

  // YAML factory: mu or (E, nu), the bulk modulus from one of kappa | nu.
  cmf::MaterialConfig cfg;
  cfg.model = "linear_elastic"; cfg.E = E; cfg.nu = nu;
  const cmf::Material made = cmf::MakeMaterial(cfg);
  CHECK(cmf::MaterialName(made) == "linear_elastic");
  CHECK(cmf::ModelNameOf(made) == "linear_elastic");
  CHECK(cmf::IsSmallStrain(made));
  CHECK_CLOSE(std::get<cmf::LinearElastic>(made).mu, lame.mu, 1e-13 * scale);
  CHECK_CLOSE(std::get<cmf::LinearElastic>(made).kappa, lame.lambda + 2.0 * lame.mu / 3.0, 1e-13 * scale);
  const cmf::Material made_ps = cmf::MakeMaterial(cfg, true);
  CHECK(cmf::MaterialName(made_ps) == "linear_elastic (plane stress)");
  CHECK(cmf::ModelNameOf(made_ps) == "linear_elastic");
  CHECK(cmf::IsSmallStrain(made_ps));
  cfg.model = "neo_hookean";
  CHECK(!cmf::IsSmallStrain(cmf::MakeMaterial(cfg)));
  CHECK(!cmf::IsSmallStrain(cmf::MakeMaterial(cfg, true)));
  cfg = cmf::MaterialConfig();
  cfg.model = "linear_elastic"; cfg.mu = 80.0; cfg.kappa = 200.0;
  CHECK(std::get<cmf::LinearElastic>(cmf::MakeMaterial(cfg)).kappa == 200.0);
  cfg.kappa = std::numeric_limits<double>::quiet_NaN();
  cfg.nu = 0.25; // kappa = 2 mu (1 + nu) / (3 (1 - 2 nu))
  CHECK_CLOSE(std::get<cmf::LinearElastic>(cmf::MakeMaterial(cfg)).kappa, 2.0 * 80.0 * 1.25 / 1.5, 1e-12);
  cmf::MaterialConfig bad = cfg;
  bad.volumetric = "logarithmic";
  CHECK_THROWS(cmf::MakeMaterial(bad), cmf::ConfigError, "'material.volumetric' is not used by model 'linear_elastic'");
  bad = cfg;
  bad.E = 200.0;
  CHECK_THROWS(cmf::MakeMaterial(bad), cmf::ConfigError, "give one, not both");
  bad = cfg;
  bad.mu = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(cmf::MakeMaterial(bad), cmf::ConfigError, "needs mu, or E and nu");
  bad = cfg;
  bad.Jm = 10.0;
  CHECK_THROWS(cmf::MakeMaterial(bad), cmf::ConfigError, "'material.Jm' is not used by model 'linear_elastic'");
  bad = cfg;
  bad.kappa = 100.0;
  CHECK_THROWS(cmf::MakeMaterial(bad), cmf::ConfigError, "needs exactly one of");
  // nu = 0.5: the mixed formulation, or plane stress in 2D, as for the rubber models.
  bad = cfg;
  bad.nu = 0.5;
  CHECK_THROWS(cmf::MakeMaterial(bad), cmf::ConfigError, "'linear_elastic' is incompressible; use formulation: mixed");
  CHECK(cmf::MaterialName(cmf::MakeMaterial(bad, true)) == "linear_elastic (plane stress)");
  CHECK(std::get<cmf::LinearElastic>(cmf::MakeMixedMaterial(bad)).Incompressible());
  const cmf::MixedMaterial mixed = cmf::MakeMixedMaterial(cfg);
  CHECK(cmf::MaterialName(mixed) == "linear_elastic" && cmf::IsSmallStrain(mixed));
  CHECK(std::get<cmf::LinearElastic>(mixed).ShearModulus() == 80.0);
  cfg.model = "iso_neo_hookean";
  CHECK(!cmf::IsSmallStrain(cmf::MakeMixedMaterial(cfg)));

  // The split of the mixed (Herrmann) formulation: P = P_iso + kappa tr(eps) I
  // with a deviatoric P_iso, W = W_iso + kappa/2 tr(eps)^2, and the volumetric
  // law u'(theta) = theta - 1 in the volume ratio theta = 1 + tr(eps).
  {
    const Mat3 F = RandomF(rng);
    const Mat3 eps = cmf::LinearElastic::Strain(F);
    const Mat3 Piso = m.PK1Iso(F);
    const Mat3 P = Piso + (m.kappa * cmf::tr(eps)) * cmf::I<3>();
    CHECK_MSG(std::abs(cmf::tr(Piso)) <= 1e-14 * scale, "linear_elastic: P_iso is deviatoric");
    CHECK_MSG(MaxAbs(P - m.PK1(F)) <= 1e-14 * scale, "linear_elastic: P = P_iso + kappa tr(eps) I");
    const double theta = 1.0 + cmf::tr(eps);
    CHECK_CLOSE(m.EnergyIso(F) + m.kappa * 0.5 * (theta - 1.0) * (theta - 1.0), m.Energy(F), 1e-14 * scale);
    CHECK_CLOSE(m.NormalizedVolumetricPressure(theta), cmf::tr(eps), 1e-15);
    CHECK_CLOSE(m.NormalizedVolumetricModulus(theta), 1.0, 0.0);
    CHECK_CLOSE(m.ComplementaryVolumetricEnergy(3.0), 4.5 / m.kappa, 1e-16);
  }

  // Incompressible plane stress (nu = 1/2, E = 3 mu): eps_33 = -(eps_11 + eps_22),
  // sigma = 2 mu eps + 2 mu (eps_11 + eps_22) I in the plane, tangent with
  // lambda* = 2 mu; no pressure unknown.
  {
    const double mu = m.mu;
    const cmf::PlaneStress<cmf::LinearElastic> ps(cmf::LinearElastic(mu, std::numeric_limits<double>::infinity()));
    CHECK(ps.Incompressible() && cmf::is_small_strain<cmf::PlaneStress<cmf::LinearElastic>>::value);
    Mat3 F = cmf::I<3>();
    F(0, 0) += 0.05; F(1, 1) += -0.02; F(0, 1) += 0.06; F(1, 0) += -0.01;
    const double e11 = 0.05, e22 = -0.02, e12 = 0.025;
    const Mat3 P = ps.PK1(F);
    CHECK_CLOSE(P(0, 0), 2.0 * mu * (2.0 * e11 + e22), 1e-13 * scale);
    CHECK_CLOSE(P(1, 1), 2.0 * mu * (2.0 * e22 + e11), 1e-13 * scale);
    CHECK_CLOSE(P(0, 1), 2.0 * mu * e12, 1e-13 * scale);
    CHECK_CLOSE(P(1, 0), 2.0 * mu * e12, 1e-13 * scale);
    CHECK_CLOSE(P(2, 2), 0.0, 1e-13 * scale);
    CHECK_CLOSE(ps.Complete(F)(2, 2) - 1.0, -(e11 + e22), 1e-15);
    CHECK_CLOSE(ps.Energy(F), 0.5 * (P(0, 0) * e11 + P(1, 1) * e22 + 2.0 * P(0, 1) * e12), 1e-13 * scale);
    const Tan4 A = cmf::MaterialTangent(ps, F, 2);
    double worst = 0.0;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
          for (int l = 0; l < 2; l++)
          {
            const double C = 2.0 * mu * (i == j) * (k == l) + mu * ((i == k) * (j == l) + (i == l) * (j == k));
            worst = std::max(worst, std::abs(A(i, j, k, l) - C) / scale);
          }
    std::printf("  incompressible plane stress at strain 0.05: tangent vs closed form rel %.2e\n", worst);
    CHECK_MSG(worst <= 1e-12, "linear_elastic incompressible plane stress: tangent with lambda* = 2 mu");
    TestPlaneStressSmallStrain(cmf::LinearElastic(mu, std::numeric_limits<double>::infinity()),
                               "linear_elastic incompressible", 3.0 * mu, 0.5);
  }
}

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
  TestDecoupled(cmf::Yeoh(0.5 * mu, -0.05 * mu, 0.01 * mu, kappa), "yeoh", mu, kappa);
  TestDecoupled(cmf::Gent(mu, 20.0, kappa), "gent", mu, kappa);
  {
    const cmf::ArrudaBoyce ab(mu, 5.0, kappa);
    TestDecoupled(ab, "arruda_boyce", ab.ShearModulus(), kappa);
    const cmf::Ogden og({0.63 * mu, 0.0012 * mu, -0.01 * mu}, {1.3, 5.0, -2.0}, kappa);
    TestDecoupled(og, "ogden", og.ShearModulus(), kappa);
  }
  TestOgdenSpecial(mu, kappa);
  TestVolumetricLaws(mu, kappa);
  TestGentCompressibleSummit();
  TestLinearElastic(E, nu);
  TestPlaneStress(E, nu, mu, kappa);
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

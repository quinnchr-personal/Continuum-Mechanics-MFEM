// S2 gate: stress-free reference state, AD tangent vs finite differences,
// objectivity under random rotations, and the small-strain (linear
// elasticity) limit for every material; Ogden reductions and its analytic
// tangent at coincident principal stretches.
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

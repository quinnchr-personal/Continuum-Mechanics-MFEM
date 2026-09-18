// Decoupled Ogden material in the isochoric principal stretches lb_a = J^{-1/3} lambda_a:
//   Psi_iso = sum_r mu_r/alpha_r (lb_1^alpha_r + lb_2^alpha_r + lb_3^alpha_r - 3)
//   U(J)    = kappa/2 (J - 1)^2         (kappa = inf: incompressible;
//             other laws selectable, see volumetric.hpp)
// Small-strain shear modulus mu = 1/2 sum_r mu_r alpha_r (each mu_r alpha_r > 0).
//
// Stresses are evaluated spectrally. With C = F^T F = sum_a c_a N_a (x) N_a,
// beta_a = sum_r mu_r lb_a^alpha_r and tau_a = beta_a - (beta_1 + beta_2 + beta_3)/3
// (the isochoric principal Kirchhoff stresses),
//   S_iso = sum_a (tau_a / c_a) N_a (x) N_a,   P_iso = F S_iso.
// Eigenvectors are not differentiable at coincident eigenvalues, so instead
// of running dual numbers through the eigen-solver the dual overload of
// PK1Iso forms the directional derivative analytically from the spectral
// representation (Ogden 1984; Simo & Taylor 1991), with the limit
// (s_a - s_b)/(c_a - c_b) -> ds_a/dc_a - ds_a/dc_b at c_a = c_b.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "base/dual.hpp"
#include "base/tensor.hpp"
#include "materials/volumetric.hpp"
#include "materials/spectral.hpp"

namespace cmf
{

struct Ogden
{
  static constexpr int kMaxTerms = 6;

  int terms = 1;
  std::array<double, kMaxTerms> mu{};
  std::array<double, kMaxTerms> alpha{};
  double kappa = std::numeric_limits<double>::infinity();
  VolumetricLaw law = VolumetricLaw::Quadratic;

  Ogden()
  {
    mu[0] = 1.0;
    alpha[0] = 2.0;
  }
  Ogden(const std::vector<double> &mu_r, const std::vector<double> &alpha_r, double kappa_)
    : kappa(kappa_)
  {
    if (mu_r.empty() || mu_r.size() != alpha_r.size() || int(mu_r.size()) > kMaxTerms)
    {
      throw std::invalid_argument("Ogden: mu_r and alpha_r need the same length, 1 to 6 terms");
    }
    terms = int(mu_r.size());
    for (int r = 0; r < terms; r++)
    {
      mu[r] = mu_r[r];
      alpha[r] = alpha_r[r];
    }
  }

  bool Incompressible() const { return !std::isfinite(kappa); }
  double ShearModulus() const
  {
    double s = 0.0;
    for (int r = 0; r < terms; r++) { s += mu[r] * alpha[r]; }
    return 0.5 * s;
  }

  tensor<double, 3, 3> PK1Iso(const tensor<double, 3, 3> &F) const
  {
    const Spectral sp = Decompose(F);
    double beta[3], dbeta[3], s[3];
    Principal(sp, beta, dbeta, s);
    return F * Assemble(sp.N, s);
  }

  // Value and directional derivative along the dual parts of F.
  tensor<dual, 3, 3> PK1Iso(const tensor<dual, 3, 3> &Fd) const
  {
    tensor<double, 3, 3> F, dF;
    Split(Fd, F, dF);
    const Spectral sp = Decompose(F);
    double beta[3], dbeta[3], s[3];
    Principal(sp, beta, dbeta, s);
    const tensor<double, 3, 3> S = Assemble(sp.N, s);
    const tensor<double, 3, 3> P = F * S;

    // dC = dF^T F + F^T dF in the eigenbasis of C.
    const tensor<double, 3, 3> dC = transpose(dF) * F + transpose(F) * dF;
    const tensor<double, 3, 3> dCh = transpose(sp.N) * dC * sp.N;

    // ds_a/dc_b, from dbeta_a/dc_b = dbeta_a (delta_ab / (2 c_a) - 1 / (6 c_b)).
    const double sum_dbeta = dbeta[0] + dbeta[1] + dbeta[2];
    double ds[3][3];
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++)
      {
        const double dtau =
          dbeta[a] * ((a == b ? 0.5 / sp.c[a] : 0.0) - 1.0 / (6.0 * sp.c[b]))
          - (0.5 * dbeta[b] / sp.c[b] - sum_dbeta / (6.0 * sp.c[b])) / 3.0;
        ds[a][b] = dtau / sp.c[a] - (a == b ? s[a] / sp.c[a] : 0.0);
      }
    const double cmax = std::max({sp.c[0], sp.c[1], sp.c[2]});
    tensor<double, 3, 3> dSh;
    for (int a = 0; a < 3; a++)
    {
      for (int b = 0; b < 3; b++) { dSh(a, a) += ds[a][b] * dCh(b, b); }
      for (int b = a + 1; b < 3; b++)
      {
        const double gamma = std::abs(sp.c[a] - sp.c[b]) > 1e-8 * cmax
                               ? (s[a] - s[b]) / (sp.c[a] - sp.c[b])
                               : ds[a][a] - ds[a][b];
        dSh(a, b) = dSh(b, a) = gamma * dCh(a, b);
      }
    }
    const tensor<double, 3, 3> dS = sp.N * dSh * transpose(sp.N);
    const tensor<double, 3, 3> dP = dF * S + F * dS;
    tensor<dual, 3, 3> out;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) { out(i, j) = dual(P(i, j), dP(i, j)); }
    return out;
  }

  double EnergyIso(const tensor<double, 3, 3> &F) const
  {
    const Spectral sp = Decompose(F);
    double w = 0.0;
    for (int r = 0; r < terms; r++)
    {
      w += mu[r] / alpha[r] *
           (std::pow(sp.lb[0], alpha[r]) + std::pow(sp.lb[1], alpha[r]) +
            std::pow(sp.lb[2], alpha[r]) - 3.0);
    }
    return w;
  }

  // dPsi_iso along the dual parts of F is P_iso : dF.
  dual EnergyIso(const tensor<dual, 3, 3> &Fd) const
  {
    tensor<double, 3, 3> F, dF;
    Split(Fd, F, dF);
    return dual(EnergyIso(F), ddot(PK1Iso(F), dF));
  }

  // Volumetric law U(J) = kappa u(J) (materials/volumetric.hpp): U'(J) and
  // U(J) for the displacement formulation, u'(J) and u''(J) for the mixed
  // constraint u'(J) - p / kappa = 0 and its tangent.
  template <typename T>
  T VolumetricPressure(const T &J) const { return kappa * cmf::NormalizedVolumetricPressure(law, J); }

  template <typename T>
  T VolumetricEnergy(const T &J) const { return kappa * cmf::NormalizedVolumetricEnergy(law, J); }

  template <typename T>
  T NormalizedVolumetricPressure(const T &J) const { return cmf::NormalizedVolumetricPressure(law, J); }

  template <typename T>
  T NormalizedVolumetricModulus(const T &J) const { return cmf::NormalizedVolumetricModulus(law, J); }

  // kappa u*(p / kappa), the volumetric energy as a function of the pressure
  // (p^2 / (2 kappa) for the quadratic law); finite kappa only.
  double ComplementaryVolumetricEnergy(double p) const
  {
    return kappa * cmf::NormalizedComplementaryEnergy(law, p / kappa);
  }

  template <typename T>
  tensor<T, 3, 3> PK1(const tensor<T, 3, 3> &F) const
  {
    const T J = det(F);
    return PK1Iso(F) + (VolumetricPressure(J) * J) * transpose(inv(F));
  }

  template <typename T>
  T Energy(const tensor<T, 3, 3> &F) const
  {
    return EnergyIso(F) + VolumetricEnergy(det(F));
  }

private:
  struct Spectral
  {
    double c[3];            // eigenvalues of C (squared principal stretches)
    double lb[3];           // isochoric principal stretches
    tensor<double, 3, 3> N; // eigenvectors of C as columns
  };

  Spectral Decompose(const tensor<double, 3, 3> &F) const
  {
    const SymmetricEigen3 e = EigenSymmetric3(transpose(F) * F);
    Spectral sp;
    sp.N = e.vectors;
    double J = 1.0;
    for (int a = 0; a < 3; a++)
    {
      sp.c[a] = e.values(a);
      J *= std::sqrt(sp.c[a]);
    }
    const double Jm13 = std::pow(J, -1.0 / 3.0);
    for (int a = 0; a < 3; a++) { sp.lb[a] = Jm13 * std::sqrt(sp.c[a]); }
    return sp;
  }

  // beta_a = sum_r mu_r lb_a^alpha_r, dbeta_a = sum_r mu_r alpha_r lb_a^alpha_r,
  // s_a = tau_a / c_a with tau_a = beta_a - mean(beta).
  void Principal(const Spectral &sp, double beta[3], double dbeta[3], double s[3]) const
  {
    for (int a = 0; a < 3; a++)
    {
      beta[a] = dbeta[a] = 0.0;
      for (int r = 0; r < terms; r++)
      {
        const double t = mu[r] * std::pow(sp.lb[a], alpha[r]);
        beta[a] += t;
        dbeta[a] += alpha[r] * t;
      }
    }
    const double mean = (beta[0] + beta[1] + beta[2]) / 3.0;
    for (int a = 0; a < 3; a++) { s[a] = (beta[a] - mean) / sp.c[a]; }
  }

  static tensor<double, 3, 3> Assemble(const tensor<double, 3, 3> &N, const double s[3])
  {
    tensor<double, 3, 3> S;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        for (int a = 0; a < 3; a++) { S(i, j) += s[a] * N(i, a) * N(j, a); }
    return S;
  }

  static void Split(const tensor<dual, 3, 3> &Fd, tensor<double, 3, 3> &F,
                    tensor<double, 3, 3> &dF)
  {
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
      {
        F(i, j) = Fd(i, j).v;
        dF(i, j) = Fd(i, j).d;
      }
  }
};

} // namespace cmf

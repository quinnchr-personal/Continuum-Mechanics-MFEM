// Finite viscoelasticity: an equilibrium branch, any decoupled model of the
// library (the base class), in parallel with Maxwell branches, each a
// neo-Hookean spring of modulus G_i in series with a dashpot of relaxation
// time tau_i on a multiplicative split F = F_e F_v of its own; the model of
// the finite viscoelasticity chapter of Anand's coupled theories and of its
// FEniCSx codes (VHB 4910 after Wang, Chester and Anand 2016; NBR). The
// internal variable of branch i is the viscous right Cauchy-Green tensor
// Cv_i = F_v^T F_v (symmetric, det 1, I at rest): six doubles of the history
// block of a quadrature point, xx yy zz xy yz xz, branch after branch.
//   Psi   = Psi_iso(Cbar) [base] + sum_i G_i/2 (tr(Cbar Cv_i^-1) - 3) + U(J)
//   P_iso = P_iso[base] + sum_i G_i J^-2/3 [F Cv_i^-1 - 1/3 (C : Cv_i^-1) F^-T]
//   (the derivative of Psi at fixed Cv_i; sigma_neq_i = G_i/J dev(Fbar Cv_i^-1 Fbar^T))
//   dCv_i/dt = 1/tau_i [Cbar - 1/3 (Cbar : Cv_i^-1) Cv_i]
// The evolution equation is integrated over a step of length dt by the
// implicit Euler step
//   Cv_i^{n+1} = (Cv_i^n + dt/tau_i Cbar^{n+1}) / det(Cv_i^n + dt/tau_i Cbar^{n+1})^{1/3},
// the unimodular projection being the exact solution of the implicit
// equation (its trace term only rescales Cv to det 1); this is the update of
// the reference codes. The stress of a step is a function of F at its end
// and of the accepted Cv_i^n, so with dual numbers the tangent includes
// dCv^{n+1}/dF; dt = 0 leaves Cv at its accepted value and gives the stress
// of the accepted state. In the small-strain limit each branch is a Maxwell
// element: the instantaneous shear modulus is mu + sum_i G_i, the relaxed one
// mu of the base, and under a held strain the branch stress decays as
// (1 + dt/tau_i)^-n per step (exp(-t/tau_i) as dt -> 0).
// Kernels consume the material through HistoryBound (kernels/history_bound.hpp),
// which fixes h and dt: every stress and energy here takes the accepted
// history block h and dt; Update writes the history at the end of the step.
#pragma once

#include <cmath>
#include <vector>

#include "base/dual.hpp"
#include "base/tensor.hpp"

namespace cmf
{

struct MaxwellBranch
{
  double G = 0.0;   // non-equilibrium shear modulus
  double tau = 1.0; // relaxation time
};

template <typename Base>
struct Viscoelastic : Base
{
  using Equilibrium = Base;
  static constexpr int kPerBranch = 6;

  std::vector<MaxwellBranch> branches;

  Viscoelastic() = default;
  Viscoelastic(const Base &base, std::vector<MaxwellBranch> branches_)
    : Base(base), branches(std::move(branches_)) {}

  int HistorySize() const { return kPerBranch * int(branches.size()); }
  // Cv_i = I for every branch.
  void InitialHistory(double *h) const
  {
    for (std::size_t i = 0; i < branches.size(); i++)
    {
      double *b = h + kPerBranch * i;
      b[0] = b[1] = b[2] = 1.0;
      b[3] = b[4] = b[5] = 0.0;
    }
  }
  // The instantaneous small-strain shear modulus (the base's is the relaxed one).
  double InstantaneousShearModulus() const
  {
    double g = Base::ShearModulus();
    for (const MaxwellBranch &b : branches) { g += b.G; }
    return g;
  }

  static tensor<double, 3, 3> Unpack(const double *b)
  {
    tensor<double, 3, 3> A;
    A(0, 0) = b[0]; A(1, 1) = b[1]; A(2, 2) = b[2];
    A(0, 1) = A(1, 0) = b[3];
    A(1, 2) = A(2, 1) = b[4];
    A(0, 2) = A(2, 0) = b[5];
    return A;
  }
  static void Pack(const tensor<double, 3, 3> &A, double *b)
  {
    b[0] = A(0, 0); b[1] = A(1, 1); b[2] = A(2, 2);
    b[3] = 0.5 * (A(0, 1) + A(1, 0));
    b[4] = 0.5 * (A(1, 2) + A(2, 1));
    b[5] = 0.5 * (A(0, 2) + A(2, 0));
  }

  // Cv_i at the end of a step of length dt from its accepted value in h and
  // Cbar at the end of the step.
  template <typename T>
  tensor<T, 3, 3> ViscousStretch(std::size_t i, const tensor<T, 3, 3> &Cbar, const double *h,
                                 double dt) const
  {
    const tensor<T, 3, 3> A = Unpack(h + kPerBranch * i) + (dt / branches[i].tau) * Cbar;
    return pow(det(A), -1.0 / 3.0) * A;
  }

  template <typename T>
  tensor<T, 3, 3> PK1Iso(const tensor<T, 3, 3> &F, const double *h, double dt) const
  {
    tensor<T, 3, 3> P = Base::PK1Iso(F);
    if (branches.empty()) { return P; }
    const T J = det(F);
    const T Jm23 = pow(J, -2.0 / 3.0);
    const tensor<T, 3, 3> C = transpose(F) * F;
    const tensor<T, 3, 3> Cbar = Jm23 * C;
    const tensor<T, 3, 3> FinvT = transpose(inv(F));
    for (std::size_t i = 0; i < branches.size(); i++)
    {
      const tensor<T, 3, 3> Cvinv = inv(ViscousStretch(i, Cbar, h, dt));
      P += (branches[i].G * Jm23) * (F * Cvinv - (ddot(C, Cvinv) / 3.0) * FinvT);
    }
    return P;
  }

  template <typename T>
  T EnergyIso(const tensor<T, 3, 3> &F, const double *h, double dt) const
  {
    T W = Base::EnergyIso(F);
    if (branches.empty()) { return W; }
    const tensor<T, 3, 3> Cbar = pow(det(F), -2.0 / 3.0) * (transpose(F) * F);
    for (std::size_t i = 0; i < branches.size(); i++)
    {
      const tensor<T, 3, 3> Cvinv = inv(ViscousStretch(i, Cbar, h, dt));
      W += (0.5 * branches[i].G) * (ddot(Cbar, Cvinv) - 3.0);
    }
    return W;
  }

  template <typename T>
  tensor<T, 3, 3> PK1(const tensor<T, 3, 3> &F, const double *h, double dt) const
  {
    const T J = det(F);
    return PK1Iso(F, h, dt) + (Base::VolumetricPressure(J) * J) * transpose(inv(F));
  }

  template <typename T>
  T Energy(const tensor<T, 3, 3> &F, const double *h, double dt) const
  {
    return EnergyIso(F, h, dt) + Base::VolumetricEnergy(det(F));
  }

  // The history at the end of the step of length dt that ends at F, from the
  // accepted one h_old, into h_new (HistorySize() doubles each).
  void Update(const tensor<double, 3, 3> &F, const double *h_old, double dt, double *h_new) const
  {
    const tensor<double, 3, 3> Cbar = std::pow(det(F), -2.0 / 3.0) * (transpose(F) * F);
    for (std::size_t i = 0; i < branches.size(); i++)
    {
      Pack(ViscousStretch(i, Cbar, h_old, dt), h_new + kPerBranch * i);
    }
  }

  // The equilibrium branch alone (the base model).
  template <typename T>
  tensor<T, 3, 3> EquilibriumPK1Iso(const tensor<T, 3, 3> &F) const { return Base::PK1Iso(F); }
};

template <typename M> struct is_viscoelastic : std::false_type {};
template <typename B> struct is_viscoelastic<Viscoelastic<B>> : std::true_type {};

} // namespace cmf

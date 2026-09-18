// Decoupled Arruda-Boyce (eight-chain) material, five-term series form.
//   Psi_iso = mu sum_{i=1..5} C_i / N^{i-1} (I1bar^i - 3^i)
//   C = {1/2, 1/20, 11/1050, 19/7000, 519/673750}
//   U(J)    = kappa/2 (J - 1)^2         (kappa = inf: incompressible;
//             other laws selectable, see volumetric.hpp)
//   P_iso   = dPsi_iso/dI1bar * dI1bar/dF   (see isochoric.hpp)
// mu is the chain-density parameter (n k T) and N the number of links per
// chain; the small-strain shear modulus is
//   mu0 = mu (1 + 3/(5N) + 99/(175N^2) + 513/(875N^3) + 42039/(67375N^4)).
#pragma once

#include <cmath>
#include <limits>

#include "base/dual.hpp"
#include "base/tensor.hpp"
#include "materials/volumetric.hpp"
#include "materials/isochoric.hpp"

namespace cmf
{

struct ArrudaBoyce
{
  static constexpr double kC[5] = {0.5, 1.0 / 20.0, 11.0 / 1050.0, 19.0 / 7000.0,
                                   519.0 / 673750.0};

  double mu = 1.0;
  double N = 8.0;
  double kappa = std::numeric_limits<double>::infinity();
  VolumetricLaw law = VolumetricLaw::Quadratic;

  ArrudaBoyce() = default;
  ArrudaBoyce(double mu_, double N_, double kappa_) : mu(mu_), N(N_), kappa(kappa_) {}

  bool Incompressible() const { return !std::isfinite(kappa); }
  double ShearModulus() const
  {
    const double x = 1.0 / N;
    return mu * (1.0 + x * (3.0 / 5.0 + x * (99.0 / 175.0 + x * (513.0 / 875.0 + x * 42039.0 / 67375.0))));
  }

  // dPsi_iso/dI1bar = mu sum_i i C_i I1bar^{i-1} / N^{i-1}.
  template <typename T>
  T DPsiDI1(const T &I1bar) const
  {
    T s = 0.0;
    T Ipow = 1.0;
    double Npow = 1.0;
    for (int i = 1; i <= 5; i++)
    {
      s += (double(i) * kC[i - 1] / Npow) * Ipow;
      Ipow = Ipow * I1bar;
      Npow *= N;
    }
    return mu * s;
  }

  template <typename T>
  tensor<T, 3, 3> PK1Iso(const tensor<T, 3, 3> &F) const
  {
    return DPsiDI1(FirstModifiedInvariant(F)) * FirstModifiedInvariantGradient(F);
  }

  template <typename T>
  T EnergyIso(const tensor<T, 3, 3> &F) const
  {
    const T I1bar = FirstModifiedInvariant(F);
    T s = 0.0;
    T Ipow = I1bar;
    double three = 3.0, Npow = 1.0;
    for (int i = 1; i <= 5; i++)
    {
      s += (kC[i - 1] / Npow) * (Ipow - three);
      Ipow = Ipow * I1bar;
      three *= 3.0;
      Npow *= N;
    }
    return mu * s;
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
};

} // namespace cmf

#include "ns_physics.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace exasim_reference
{
template <typename T> struct ModelDefaults {};
namespace Kokkos
{
using std::pow;
using std::sqrt;
}
#define KOKKOS_INLINE_FUNCTION inline
#include "my_model.hpp"
#undef KOKKOS_INLINE_FUNCTION
} // namespace exasim_reference

namespace
{

using hdg_ns::NSParams;

void Require(bool condition, const std::string &message)
{
   if (!condition) { throw std::runtime_error(message); }
}

double ScaledError(const double *actual, const double *expected, int n)
{
   double error = 0.0;
   for (int i = 0; i < n; ++i)
   {
      error = std::max(
         error, std::abs(actual[i] - expected[i]) /
                std::max(1.0, std::abs(expected[i])));
   }
   return error;
}

std::array<double, 4> EntropyVariables(const double u[4],
                                      const NSParams &params)
{
   const double gam = params.mu[0];
   const double p = hdg_ns::Pressure(u, params);
   const double rho = u[0];
   const double vel2 = (u[1] * u[1] + u[2] * u[2]) / (rho * rho);
   const double entropy = std::log(p) - gam * std::log(rho);
   return
   {
      (gam - entropy) / (gam - 1.0) - rho * vel2 / (2.0 * p),
      u[1] / p,
      u[2] / p,
      -rho / p
   };
}

void FillAdmissibleState(std::mt19937_64 &rng, double uq[12],
                         double uhat[4])
{
   std::uniform_real_distribution<double> rho_dist(0.25, 4.0);
   std::uniform_real_distribution<double> velocity_dist(-2.0, 2.0);
   std::uniform_real_distribution<double> pressure_dist(0.08, 3.0);
   std::uniform_real_distribution<double> gradient_dist(-1.5, 1.5);
   std::uniform_real_distribution<double> trace_perturb(-0.05, 0.05);
   constexpr double gam1 = 0.4;

   const double rho = rho_dist(rng);
   const double ux = velocity_dist(rng);
   const double uy = velocity_dist(rng);
   const double p = pressure_dist(rng);
   uq[0] = rho;
   uq[1] = rho * ux;
   uq[2] = rho * uy;
   uq[3] = p / gam1 + 0.5 * rho * (ux * ux + uy * uy);
   for (int i = 4; i < 12; ++i) { uq[i] = gradient_dist(rng); }
   for (int i = 0; i < 4; ++i)
   {
      uhat[i] = uq[i] * (1.0 + trace_perturb(rng));
   }
}

void ReferenceFlux(const double uq[12], double av, const NSParams &params,
                   double f[8], double jac[96])
{
   const double x[2] = {0.0, 0.0};
   const double v[1] = {av};
   const double w[1] = {0.0};
   const double uinf[1] = {0.0};
   exasim_reference::PdeModel::flux(
      f, x, uq, v, w, params.mu, uinf, 0.0);
   exasim_reference::PdeModel::flux_jac_uq(
      jac, x, uq, v, w, params.mu, uinf, 0.0);
}

void ReferenceBoundary(int ib, const double uq[12], const double uhat[4],
                       const double normal[2], const NSParams &params,
                       double fb[4], double jac_uq[48], double jac_uh[16])
{
   const double x[2] = {0.0, 0.0};
   const double v[1] = {0.0};
   const double w[1] = {0.0};
   const double tau[1] = {params.tau};
   const double uinf[1] = {0.0};
   exasim_reference::PdeModel::fbou_hdg(
      fb, ib, x, uq, v, w, uhat, normal, tau, params.mu, uinf, 0.0);
   exasim_reference::PdeModel::fbou_hdg_jac_uq(
      jac_uq, ib, x, uq, v, w, uhat, normal, tau, params.mu, uinf, 0.0);
   exasim_reference::PdeModel::fbou_hdg_jac_uh(
      jac_uh, ib, x, uq, v, w, uhat, normal, tau, params.mu, uinf, 0.0);
}

void CheckReferenceParity()
{
   NSParams params;
   std::mt19937_64 rng(0x4d31484447ULL);
   std::uniform_real_distribution<double> av_dist(0.0, 0.08);
   std::uniform_real_distribution<double> normal_dist(-1.0, 1.0);
   double max_flux_value_error = 0.0;
   double max_flux_jac_error = 0.0;
   double max_fb_value_error = 0.0;
   double max_fb_uq_jac_error = 0.0;
   double max_fb_uh_jac_error = 0.0;

   for (int sample = 0; sample < 1000; ++sample)
   {
      double uq[12], uhat[4];
      FillAdmissibleState(rng, uq, uhat);
      const double av = av_dist(rng);
      double normal[2] = {normal_dist(rng), normal_dist(rng)};
      const double normal_length = std::hypot(normal[0], normal[1]);
      normal[0] /= normal_length;
      normal[1] /= normal_length;

      double actual_flux[8], actual_flux_jac[96];
      double reference_flux[8], reference_flux_jac[96];
      hdg_ns::NSFlux(uq, av, params, actual_flux, actual_flux_jac);
      ReferenceFlux(uq, av, params, reference_flux, reference_flux_jac);
      max_flux_value_error =
         std::max(max_flux_value_error,
                  ScaledError(actual_flux, reference_flux, 8));
      max_flux_jac_error =
         std::max(max_flux_jac_error,
                  ScaledError(actual_flux_jac, reference_flux_jac, 96));

      double physical_uq[12], physical_flux[8], physical_jac[96];
      double reference_physical_jac[96];
      std::copy(uq, uq + 12, physical_uq);
      std::copy(reference_flux_jac, reference_flux_jac + 96,
                reference_physical_jac);
      for (int variable = 4; variable < 12; ++variable)
      {
         physical_uq[variable] *= -1.0;
         for (int output = 0; output < 8; ++output)
         {
            reference_physical_jac[output + 8 * variable] *= -1.0;
         }
      }
      hdg_ns::FluxPhysGrad(
         physical_uq, av, params, physical_flux, physical_jac);
      max_flux_value_error =
         std::max(max_flux_value_error,
                  ScaledError(physical_flux, reference_flux, 8));
      max_flux_jac_error =
         std::max(max_flux_jac_error,
                  ScaledError(physical_jac, reference_physical_jac, 96));

      for (int ib = 1; ib <= 3; ++ib)
      {
         double actual_fb[4], actual_uq_jac[48], actual_uh_jac[16];
         double reference_fb[4], reference_uq_jac[48], reference_uh_jac[16];
         hdg_ns::NSFbouHdg(ib, uq, uhat, normal, params, actual_fb,
                           actual_uq_jac, actual_uh_jac);
         ReferenceBoundary(ib, uq, uhat, normal, params, reference_fb,
                           reference_uq_jac, reference_uh_jac);
         max_fb_value_error =
            std::max(max_fb_value_error,
                     ScaledError(actual_fb, reference_fb, 4));
         max_fb_uq_jac_error =
            std::max(max_fb_uq_jac_error,
                     ScaledError(actual_uq_jac, reference_uq_jac, 48));
         max_fb_uh_jac_error =
            std::max(max_fb_uh_jac_error,
                     ScaledError(actual_uh_jac, reference_uh_jac, 16));
      }
   }

   constexpr double tolerance = 1.0e-13;
   Require(max_flux_value_error <= tolerance,
           "flux values do not match generated model");
   Require(max_flux_jac_error <= tolerance,
           "flux Jacobian does not match generated model");
   Require(max_fb_value_error <= tolerance,
           "fb values do not match generated model");
   Require(max_fb_uq_jac_error <= tolerance,
           "fb uq Jacobian does not match generated model");
   Require(max_fb_uh_jac_error <= tolerance,
           "fb uhat Jacobian does not match generated model");

   std::cout << "PASS reference parity (1000 states):"
             << " flux=" << max_flux_value_error
             << " flux_jac=" << max_flux_jac_error
             << " fb=" << max_fb_value_error
             << " fb_jac_uq=" << max_fb_uq_jac_error
             << " fb_jac_uh=" << max_fb_uh_jac_error << '\n';
}

void CheckFiniteDifferences()
{
   NSParams params;
   std::mt19937_64 rng(0xfd4d31484447ULL);
   std::uniform_real_distribution<double> av_dist(0.0, 0.08);
   double max_flux_fd_error = 0.0;
   double max_fb_uq_fd_error = 0.0;
   double max_fb_uh_fd_error = 0.0;
   const double normal[2] = {0.6, -0.8};

   for (int sample = 0; sample < 120; ++sample)
   {
      double uq[12], uhat[4];
      FillAdmissibleState(rng, uq, uhat);
      const double av = av_dist(rng);
      double base_flux[8], analytic[96];
      hdg_ns::NSFlux(uq, av, params, base_flux, analytic);

      for (int variable = 0; variable < 12; ++variable)
      {
         const double h = 1.0e-6 * std::max(1.0, std::abs(uq[variable]));
         double plus_state[12], minus_state[12];
         std::copy(uq, uq + 12, plus_state);
         std::copy(uq, uq + 12, minus_state);
         plus_state[variable] += h;
         minus_state[variable] -= h;
         double plus_flux[8], minus_flux[8];
         hdg_ns::NSFlux(plus_state, av, params, plus_flux);
         hdg_ns::NSFlux(minus_state, av, params, minus_flux);
         for (int output = 0; output < 8; ++output)
         {
            const double fd = (plus_flux[output] - minus_flux[output]) /
                              (2.0 * h);
            const double exact = analytic[output + 8 * variable];
            const double scale = std::max({1.0, std::abs(fd),
                                           std::abs(exact)});
            max_flux_fd_error =
               std::max(max_flux_fd_error, std::abs(fd - exact) / scale);
         }
      }

      for (int ib = 1; ib <= 3; ++ib)
      {
         double fb[4], analytic_uq[48], analytic_uh[16];
         hdg_ns::NSFbouHdg(ib, uq, uhat, normal, params, fb,
                           analytic_uq, analytic_uh);
         for (int variable = 0; variable < 12; ++variable)
         {
            const double h =
               1.0e-6 * std::max(1.0, std::abs(uq[variable]));
            double plus_state[12], minus_state[12];
            std::copy(uq, uq + 12, plus_state);
            std::copy(uq, uq + 12, minus_state);
            plus_state[variable] += h;
            minus_state[variable] -= h;
            double plus_fb[4], minus_fb[4];
            hdg_ns::NSFbouHdg(ib, plus_state, uhat, normal, params, plus_fb);
            hdg_ns::NSFbouHdg(ib, minus_state, uhat, normal, params, minus_fb);
            for (int output = 0; output < 4; ++output)
            {
               const double fd = (plus_fb[output] - minus_fb[output]) /
                                 (2.0 * h);
               const double exact = analytic_uq[output + 4 * variable];
               max_fb_uq_fd_error =
                  std::max(max_fb_uq_fd_error, std::abs(fd - exact));
            }
         }
         for (int variable = 0; variable < 4; ++variable)
         {
            const double h =
               1.0e-6 * std::max(1.0, std::abs(uhat[variable]));
            double plus_uh[4], minus_uh[4];
            std::copy(uhat, uhat + 4, plus_uh);
            std::copy(uhat, uhat + 4, minus_uh);
            plus_uh[variable] += h;
            minus_uh[variable] -= h;
            double plus_fb[4], minus_fb[4];
            hdg_ns::NSFbouHdg(ib, uq, plus_uh, normal, params, plus_fb);
            hdg_ns::NSFbouHdg(ib, uq, minus_uh, normal, params, minus_fb);
            for (int output = 0; output < 4; ++output)
            {
               const double fd = (plus_fb[output] - minus_fb[output]) /
                                 (2.0 * h);
               const double exact = analytic_uh[output + 4 * variable];
               max_fb_uh_fd_error =
                  std::max(max_fb_uh_fd_error, std::abs(fd - exact));
            }
         }
      }
   }

   Require(max_flux_fd_error <= 2.0e-7,
           "flux Jacobian is not central-FD consistent");
   Require(max_fb_uq_fd_error <= 2.0e-9,
           "fb uq Jacobian is not central-FD consistent");
   Require(max_fb_uh_fd_error <= 2.0e-9,
           "fb uhat Jacobian is not central-FD consistent");
   std::cout << "PASS central finite differences:"
             << " flux_jac=" << max_flux_fd_error
             << " fb_jac_uq=" << max_fb_uq_fd_error
             << " fb_jac_uh=" << max_fb_uh_fd_error << '\n';
}

void CheckDissipativity()
{
   NSParams params;
   std::mt19937_64 rng(0xd1551a471ULL);
   std::uniform_real_distribution<double> rho_dist(0.7, 2.0);
   std::uniform_real_distribution<double> velocity_dist(-0.7, 0.7);
   std::uniform_real_distribution<double> pressure_dist(0.5, 2.0);
   std::uniform_real_distribution<double> gradient_dist(-0.2, 0.2);
   double maximum_entropy_production = -std::numeric_limits<double>::infinity();

   for (int sample = 0; sample < 500; ++sample)
   {
      double state[4];
      state[0] = rho_dist(rng);
      const double vx = velocity_dist(rng);
      const double vy = velocity_dist(rng);
      state[1] = state[0] * vx;
      state[2] = state[0] * vy;
      state[3] = pressure_dist(rng) / 0.4 +
                 0.5 * state[0] * (vx * vx + vy * vy);

      double physical_gradient[2][4];
      double uq[12] = {};
      std::copy(state, state + 4, uq);
      for (int direction = 0; direction < 2; ++direction)
      {
         for (int component = 0; component < 4; ++component)
         {
            physical_gradient[direction][component] = gradient_dist(rng);
            uq[4 + 4 * direction + component] =
               -physical_gradient[direction][component];
         }
      }

      double zero_gradient_uq[12] = {};
      std::copy(state, state + 4, zero_gradient_uq);
      double flux[8], inviscid_flux[8];
      hdg_ns::NSFlux(uq, 0.025, params, flux);
      hdg_ns::NSFlux(zero_gradient_uq, 0.0, params, inviscid_flux);

      double contraction = 0.0;
      for (int direction = 0; direction < 2; ++direction)
      {
         double norm = 0.0;
         for (double value : physical_gradient[direction])
         {
            norm = std::max(norm, std::abs(value));
         }
         const double h = 1.0e-6 / std::max(1.0, norm);
         double plus_state[4], minus_state[4];
         for (int component = 0; component < 4; ++component)
         {
            plus_state[component] =
               state[component] +
               h * physical_gradient[direction][component];
            minus_state[component] =
               state[component] -
               h * physical_gradient[direction][component];
         }
         Require(hdg_ns::Pressure(plus_state, params) > 0.0 &&
                 hdg_ns::Pressure(minus_state, params) > 0.0,
                 "entropy-variable perturbation left admissible set");
         const auto plus_entropy = EntropyVariables(plus_state, params);
         const auto minus_entropy = EntropyVariables(minus_state, params);
         for (int component = 0; component < 4; ++component)
         {
            const double entropy_gradient =
               (plus_entropy[component] - minus_entropy[component]) /
               (2.0 * h);
            const int flux_component = 4 * direction + component;
            contraction += entropy_gradient *
                           (flux[flux_component] -
                            inviscid_flux[flux_component]);
         }
      }
      maximum_entropy_production =
         std::max(maximum_entropy_production, contraction);
   }

   Require(maximum_entropy_production <= 1.0e-11,
           "viscous plus AV flux has positive entropy production; q sign is wrong");
   std::cout << "PASS dissipativity (q = -grad(u)):"
             << " max_entropy_production="
             << maximum_entropy_production << '\n';
}

void CheckFreestreamIdentities()
{
   NSParams params;
   const double gam = params.mu[0];
   const double minf = params.mu[3];
   const double pinf = 1.0 / (gam * minf * minf);
   const double exact_state[4] =
   {
      1.0, 1.0, 0.0,
      0.5 + 1.0 / (gam * (gam - 1.0) * minf * minf)
   };
   const double p = hdg_ns::Pressure(exact_state, params);
   const double viscosity = hdg_ns::SutherlandMu(p, exact_state[0], params);
   const double p_error = std::abs(p - pinf);
   const double mu_error = std::abs(viscosity - 1.0 / params.mu[1]);
   Require(p_error <= 8.0 * std::numeric_limits<double>::epsilon(),
           "full-precision freestream pressure identity failed");
   Require(mu_error <=
           8.0 * std::numeric_limits<double>::epsilon() / params.mu[1],
           "full-precision freestream viscosity identity failed");
   Require(std::abs(params.TisoW() - 0.06550101502128686) <= 1.0e-16,
           "rounded-deck wall temperature split was changed");

   const double rounded_state[4] =
      {params.mu[4], params.mu[5], params.mu[6], params.mu[7]};
   const double rounded_pressure_error =
      std::abs(hdg_ns::Pressure(rounded_state, params) - pinf) / pinf;
   Require(rounded_pressure_error > 1.0e-5,
           "rounded deck state was silently made full precision");
   std::cout << "PASS freestream identities:"
             << " pressure_abs_error=" << p_error
             << " viscosity_abs_error=" << mu_error
             << " TisoW=" << params.TisoW()
             << " rounded_pressure_rel_mismatch="
             << rounded_pressure_error << '\n';
}

} // namespace

int main()
{
   try
   {
      std::cout << std::setprecision(17);
      CheckReferenceParity();
      CheckFiniteDifferences();
      CheckDissipativity();
      CheckFreestreamIdentities();
      std::cout << "ALL test_physics M1 GATES PASSED\n";
      return EXIT_SUCCESS;
   }
   catch (const std::exception &error)
   {
      std::cerr << "FAIL test_physics: " << error.what() << '\n';
      return EXIT_FAILURE;
   }
}

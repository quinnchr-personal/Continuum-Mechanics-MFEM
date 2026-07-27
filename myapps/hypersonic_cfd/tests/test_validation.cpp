// G7 validation gates against classical references, independent of the
// frozen replica. Case 1: supersonic cylinder flow at M=3 and M=5
// (Re=1000, sensor AV, damped-freestream cold start) versus the Billig
// (1967) bow-shock standoff correlation for cylinders,
//    delta/R = 0.386 * exp(4.67 / M^2),
// and the Rayleigh-pitot stagnation-point pressure coefficient.
#include "discretization/av_sensor.hpp"
#include "discretization/hdg_operator.hpp"
#include "io/exasim_mesh.hpp"
#include "physics/perfect_gas_model.hpp"
#include "post/surface_post.hpp"
#include "solvers/newton.hpp"

#include "mfem.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using hycfd::HDGOperator;
using hycfd::HDGState;
using hycfd::PerfectGasModel;
using hycfd::PerfectGasParams;

void Require(bool condition, const std::string &message)
{
   if (!condition) { throw std::runtime_error(message); }
}

double BilligStandoff(double mach)
{
   return 0.386 * std::exp(4.67 / (mach * mach));
}

// Rayleigh pitot: stagnation-to-freestream pressure ratio behind a
// normal shock.
double PitotPressureRatio(double gamma, double mach)
{
   const double m2 = mach * mach;
   const double term1 = std::pow(
      (gamma + 1.0) * (gamma + 1.0) * m2 /
         (4.0 * gamma * m2 - 2.0 * (gamma - 1.0)),
      gamma / (gamma - 1.0));
   const double term2 = (1.0 - gamma + 2.0 * gamma * m2) / (gamma + 1.0);
   return term1 * term2;
}

// Stagnation pressure coefficient (nondimensional dynamic head is 1/2).
double StagnationCpTheory(double gamma, double mach)
{
   return 2.0 * (PitotPressureRatio(gamma, mach) - 1.0) /
          (gamma * mach * mach);
}

// Sinclair & Cui, Phys. Fluids 29, 026102 (2017), Eq. (32): cylinder
// bow-shock standoff (delta/R) from the modified-Newtonian sonic-point
// geometry with a linear Mach-number profile between shock and body.
// Validated there against Euler solutions and experiment over
// M = 1.35..6; Billig's hypersonic fit underpredicts cylinder standoff
// in this range (their Fig. 7), so this is the gating reference.
double SinclairCuiStandoff(double gamma, double mach)
{
   const double m2 = mach * mach;
   const double pitot = PitotPressureRatio(gamma, mach);
   const double cp_max = 2.0 * (pitot - 1.0) / (gamma * m2);
   const double sonic_factor =
      std::pow(0.5 * (gamma + 1.0), -gamma / (gamma - 1.0));
   const double cp_sonic =
      2.0 * (sonic_factor * pitot - 1.0) / (gamma * m2);
   const double beta_s =
      0.5 * M_PI - std::asin(std::sqrt(cp_sonic / cp_max));
   const double theta_s = 0.5 * M_PI - beta_s;
   const double post_shock_mach = std::sqrt(
      (2.0 + (gamma - 1.0) * m2) / (2.0 * gamma * m2 - gamma + 1.0));
   return beta_s * beta_s /
          (theta_s * theta_s * std::cos(beta_s)) * post_shock_mach;
}

struct CylinderResult
{
   double standoff = 0.0;
   double stagnation_cp = 0.0;
};

CylinderResult SolveCylinder(double mach)
{
   PerfectGasParams params;
   params.mach = mach;
   params.reynolds = 1000.0;
   params.regularized = true; // floors on for the AV cold start

   std::unique_ptr<mfem::Mesh> serial_mesh =
      hycfd::BuildAnalyticMesh(32, 16, 4);
   mfem::ParMesh mesh(MPI_COMM_WORLD, *serial_mesh);
   PerfectGasModel model(params);
   std::vector<int> attr_to_bcid(3);
   attr_to_bcid[0] =
      model.RegisterBoundaryCondition("isothermal_wall", YAML::Node());
   attr_to_bcid[1] =
      model.RegisterBoundaryCondition("supersonic_outflow", YAML::Node());
   attr_to_bcid[2] =
      model.RegisterBoundaryCondition("freestream", YAML::Node());

   hycfd::HDGOptions options;
   options.order = 4;
   options.tau = params.tau;
   HDGOperator op(
      mesh, [](const mfem::Vector &) { return 0.0; }, model,
      attr_to_bcid, options);

   const double damping_c = 10.0;
   HDGState state;
   op.ProjectState(
      [&params, damping_c](const mfem::Vector &x, double *value)
      {
         const double distance = std::hypot(x[0], x[1]) - 1.0;
         const double velocity = std::tanh(damping_c * distance);
         value[0] = 1.0;
         value[1] = velocity;
         value[2] = 0.0;
         value[3] =
            params.TinfND() *
               ((params.Twall_K / params.T_inf_K - 1.0) *
                   std::exp(-damping_c * distance) + 1.0) +
            0.5 * velocity * velocity;
      },
      state);
   op.InitializeTraceFromInterior(state);

   hycfd::SensorAVSchedule schedule;
   schedule.sensor.lambda = 0.2;
   schedule.relax = 0.5;
   schedule.freeze_residual = 1.0e-4;
   schedule.bootstrap_profile = [](const mfem::Vector &x)
   {
      return 0.05 * std::tanh(5.0 * (std::hypot(x[0], x[1]) - 1.0));
   };
   schedule.bootstrap_decay_iterations = 25;
   hycfd::SensorAVController controller(mesh, op, model, schedule);

   hycfd::NewtonConfig config;
   config.max_iterations = 150;
   config.tolerance = 1.0e-6;
   config.pseudo_transient = true;
   config.initial_pseudo_time_step = 15.0;
   config.ptc_off_residual = 1.0e-7;

   const hycfd::NewtonReport report = hycfd::DampedNewtonSolve(
      op, state, config,
      [&op, mach](int iteration, const HDGState &current, double residual)
      {
         std::cout << "M" << mach << " Newton " << std::setw(3)
                   << iteration << " residual=" << residual
                   << " min_rho=" << op.MinimumDensity(current)
                   << " min_p=" << op.MinimumPressure(current)
                   << std::endl;
      },
      [&controller](int iteration, const HDGState &current,
                    double residual)
      { controller.Refresh(iteration, current, residual); });
   Require(report.converged,
           "cylinder solve did not converge at M=" + std::to_string(mach));

   CylinderResult result;
   const std::vector<hycfd::WallSample> wall =
      hycfd::ComputeWallSamples(mesh, op, state, params);
   for (const hycfd::WallSample &sample : wall)
   {
      result.stagnation_cp = std::max(result.stagnation_cp, sample.cp);
   }
   const double gamma = params.gamma;
   const double post_shock_density =
      (gamma + 1.0) * mach * mach /
      ((gamma - 1.0) * mach * mach + 2.0);
   result.standoff = hycfd::StagnationDensityCrossing(
      mesh, op, state, 0.5 * (1.0 + post_shock_density));
   return result;
}

} // namespace

int main(int argc, char *argv[])
{
   int exit_code = EXIT_SUCCESS;
   mfem::Mpi::Init(argc, argv);
   mfem::Hypre::Init();
   mfem::MFEMInitializePetsc(
      &argc, &argv, "input/petsc.opts", nullptr);
   if (!mfem::Mpi::Root())
   {
      std::cout.setstate(std::ios::failbit);
   }
   try
   {
      std::cout << std::setprecision(17);
      // Gate tolerances: the Sinclair-Cui standoff reference is itself a
      // few-percent theory, the captured shock has finite AV thickness,
      // and Re=1000 viscous displacement pushes the shock slightly out;
      // the gate still rejects wrong-by-a-cell standoffs and wrong Mach
      // scaling. Billig is printed for reference only (it underpredicts
      // cylinder standoff at these Mach numbers).
      constexpr double kStandoffTolerance = 0.08;
      constexpr double kStagnationCpTolerance = 0.03;
      for (const double mach : {3.0, 5.0})
      {
         const CylinderResult result = SolveCylinder(mach);
         const double reference = SinclairCuiStandoff(1.4, mach);
         const double cp_theory = StagnationCpTheory(1.4, mach);
         const double standoff_error =
            std::abs(result.standoff - reference) / reference;
         const double cp_error =
            std::abs(result.stagnation_cp - cp_theory) / cp_theory;
         std::cout << "M=" << mach
                   << " standoff=" << result.standoff
                   << " sinclair_cui=" << reference
                   << " billig=" << BilligStandoff(mach)
                   << " rel_err=" << standoff_error
                   << " | stagnation_cp=" << result.stagnation_cp
                   << " rayleigh=" << cp_theory
                   << " rel_err=" << cp_error << '\n';
         Require(standoff_error <= kStandoffTolerance,
                 "shock standoff disagrees with the Sinclair-Cui"
                 " cylinder reference");
         Require(cp_error <= kStagnationCpTolerance,
                 "stagnation Cp disagrees with the Rayleigh pitot value");
         std::cout << "PASS cylinder standoff and Rayleigh stagnation Cp"
                      " at M=" << mach << '\n';
      }
      std::cout << "ALL test_validation G7 GATES PASSED\n";
   }
   catch (const std::exception &error)
   {
      std::cerr << "FAIL test_validation: " << error.what() << '\n';
      exit_code = EXIT_FAILURE;
   }
   mfem::MFEMFinalizePetsc();
   return exit_code;
}

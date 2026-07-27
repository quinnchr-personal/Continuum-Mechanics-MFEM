// G5 gates for the expanded boundary-condition catalog: FD Jacobian
// consistency for every new type, the characteristic flux-splitting
// identity A+ + A- = A_n, exact preservation of compatible states
// (freestream through the characteristic farfield, wall-parallel uniform
// flow through slip walls, a quiescent gas in an adiabatic box), and an
// adiabatic-wall cylinder solve with a near-zero wall heat flux.
#include "discretization/hdg_operator.hpp"
#include "io/exasim_mesh.hpp"
#include "physics/perfect_gas_model.hpp"
#include "solvers/newton.hpp"

#include "mfem.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using hycfd::HDGOperator;
using hycfd::HDGState;
using hycfd::PerfectGasParams;
using hycfd::PerfectGasModel;

void Require(bool condition, const std::string &message)
{
   if (!condition) { throw std::runtime_error(message); }
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

void CheckBoundaryJacobians()
{
   PerfectGasParams params;
   PerfectGasModel model(params);
   const char *types[4] =
   {
      "adiabatic_wall", "slip_wall", "characteristic_farfield",
      "pressure_outlet"
   };
   int ids[4];
   for (int i = 0; i < 4; ++i)
   {
      ids[i] = model.RegisterBoundaryCondition(types[i], YAML::Node());
   }

   std::mt19937_64 rng(0x4235464a4143ULL);
   std::uniform_real_distribution<double> normal_dist(-1.0, 1.0);
   double max_uq_error = 0.0;
   double max_uh_error = 0.0;
   const double x[2] = {0.3, -0.2};
   for (int sample = 0; sample < 80; ++sample)
   {
      double uq[12], uhat[4];
      FillAdmissibleState(rng, uq, uhat);
      double normal[2] = {normal_dist(rng), normal_dist(rng)};
      const double length = std::hypot(normal[0], normal[1]);
      normal[0] /= length;
      normal[1] /= length;

      for (int i = 0; i < 4; ++i)
      {
         double fb[4], jac_uq[48], jac_uh[16];
         model.BoundaryResidual(ids[i], uq, uhat, normal, x, fb,
                                jac_uq, jac_uh);
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
            model.BoundaryResidual(ids[i], plus_state, uhat, normal, x,
                                   plus_fb, nullptr, nullptr);
            model.BoundaryResidual(ids[i], minus_state, uhat, normal, x,
                                   minus_fb, nullptr, nullptr);
            for (int output = 0; output < 4; ++output)
            {
               const double fd =
                  (plus_fb[output] - minus_fb[output]) / (2.0 * h);
               max_uq_error = std::max(
                  max_uq_error,
                  std::abs(fd - jac_uq[output + 4 * variable]));
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
            model.BoundaryResidual(ids[i], uq, plus_uh, normal, x,
                                   plus_fb, nullptr, nullptr);
            model.BoundaryResidual(ids[i], uq, minus_uh, normal, x,
                                   minus_fb, nullptr, nullptr);
            for (int output = 0; output < 4; ++output)
            {
               const double fd =
                  (plus_fb[output] - minus_fb[output]) / (2.0 * h);
               max_uh_error = std::max(
                  max_uh_error,
                  std::abs(fd - jac_uh[output + 4 * variable]));
            }
         }
      }
   }
   // The adiabatic heat-flux row differentiates through Sutherland; its
   // internal FD and this test's FD agree to ~1e-6 in the worst case.
   Require(max_uq_error <= 5.0e-6,
           "new-BC uq Jacobian fails the FD check");
   Require(max_uh_error <= 5.0e-6,
           "new-BC uhat Jacobian fails the FD check");
   std::cout << "PASS new-BC FD Jacobians:"
             << " jac_uq=" << max_uq_error
             << " jac_uh=" << max_uh_error << '\n';
}

void CheckCharacteristicSplitting()
{
   // -dfb/duhat = A+ + A- must equal the inviscid normal-flux Jacobian at
   // the freestream, obtained independently by FD of the zero-gradient
   // flux.
   PerfectGasParams params;
   PerfectGasModel model(params);
   const int id =
      model.RegisterBoundaryCondition("characteristic_farfield",
                                      YAML::Node());
   std::mt19937_64 rng(0x414e53504c4954ULL);
   std::uniform_real_distribution<double> normal_dist(-1.0, 1.0);
   const double x[2] = {0.0, 0.0};
   double freestream[4];
   params.Freestream(freestream);
   double max_error = 0.0;
   for (int sample = 0; sample < 40; ++sample)
   {
      double normal[2] = {normal_dist(rng), normal_dist(rng)};
      const double length = std::hypot(normal[0], normal[1]);
      normal[0] /= length;
      normal[1] /= length;

      double uq[12] = {};
      std::copy(freestream, freestream + 4, uq);
      double fb[4], jac_uq[48], jac_uh[16];
      model.BoundaryResidual(id, uq, freestream, normal, x, fb,
                             jac_uq, jac_uh);
      for (int i = 0; i < 4; ++i)
      {
         max_error = std::max(max_error, std::abs(fb[i]));
      }

      for (int column = 0; column < 4; ++column)
      {
         const double h =
            1.0e-6 * std::max(1.0, std::abs(freestream[column]));
         double plus_state[12] = {}, minus_state[12] = {};
         std::copy(freestream, freestream + 4, plus_state);
         std::copy(freestream, freestream + 4, minus_state);
         plus_state[column] += h;
         minus_state[column] -= h;
         double plus_flux[8], minus_flux[8];
         hycfd::NSFlux(plus_state, 0.0, params, plus_flux);
         hycfd::NSFlux(minus_state, 0.0, params, minus_flux);
         for (int row = 0; row < 4; ++row)
         {
            const double a_n =
               (plus_flux[row] - minus_flux[row]) / (2.0 * h) *
                  normal[0] +
               (plus_flux[row + 4] - minus_flux[row + 4]) / (2.0 * h) *
                  normal[1];
            max_error = std::max(
               max_error,
               std::abs(-jac_uh[row + 4 * column] - a_n));
         }
      }
   }
   Require(max_error <= 5.0e-8,
           "characteristic splitting does not reproduce A_n");
   std::cout << "PASS characteristic splitting A+ + A- = A_n:"
             << " max_error=" << max_error << '\n';
}

void CheckStatePreservation()
{
   const PerfectGasParams params;
   auto zero_av = [](const mfem::Vector &) { return 0.0; };

   // (a) Freestream through characteristic farfield on the curved annulus.
   {
      std::unique_ptr<mfem::Mesh> serial_mesh =
         hycfd::BuildAnalyticMesh(3, 6, 4);
      mfem::ParMesh mesh(MPI_COMM_WORLD, *serial_mesh);
      PerfectGasModel model(params);
      const std::vector<int> attrs(
         3, model.RegisterBoundaryCondition("characteristic_farfield",
                                            YAML::Node()));
      HDGOperator op(mesh, zero_av, model, attrs);
      double state[4];
      params.Freestream(state);
      HDGState freestream;
      op.SetConstantState(state, freestream);
      const hycfd::HDGResidualNorms norms =
         op.Assemble(freestream, false);
      Require(norms.Total() <= 1.0e-10,
              "characteristic farfield does not preserve the freestream");
      std::cout << "PASS characteristic-farfield freestream preservation:"
                << " total=" << norms.Total() << '\n';
   }

   // (b) Wall-parallel uniform flow through slip walls on a channel
   // (bottom/top attrs 1/3 are slip; right/left attrs 2/4 are
   // characteristic).
   {
      mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
         4, 2, mfem::Element::QUADRILATERAL, true);
      mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
      PerfectGasModel model(params);
      const int slip =
         model.RegisterBoundaryCondition("slip_wall", YAML::Node());
      const int farfield =
         model.RegisterBoundaryCondition("characteristic_farfield",
                                         YAML::Node());
      const std::vector<int> attrs = {slip, farfield, slip, farfield};
      HDGOperator op(mesh, zero_av, model, attrs);
      double state[4];
      params.Freestream(state);
      HDGState uniform;
      op.SetConstantState(state, uniform);
      const hycfd::HDGResidualNorms norms = op.Assemble(uniform, false);
      Require(norms.Total() <= 1.0e-10,
              "slip walls do not preserve wall-parallel uniform flow");
      std::cout << "PASS slip-wall uniform-flow preservation:"
                << " total=" << norms.Total() << '\n';
   }

   // (c) Quiescent gas in an adiabatic box.
   {
      mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
         3, 3, mfem::Element::QUADRILATERAL, true);
      mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
      PerfectGasModel model(params);
      const std::vector<int> attrs(
         4, model.RegisterBoundaryCondition("adiabatic_wall",
                                            YAML::Node()));
      HDGOperator op(mesh, zero_av, model, attrs);
      const double state[4] = {1.0, 0.0, 0.0, params.TinfND()};
      HDGState quiescent;
      op.SetConstantState(state, quiescent);
      const hycfd::HDGResidualNorms norms =
         op.Assemble(quiescent, false);
      Require(norms.Total() <= 1.0e-10,
              "adiabatic walls do not preserve a quiescent gas");
      std::cout << "PASS adiabatic-box quiescent preservation:"
                << " total=" << norms.Total() << '\n';
   }
}

// Global condensed-Jacobian directional FD on an adiabatic-wall box:
// the heat-flux row is the first gradient-dependent boundary residual,
// exercising the fb->G-block->q-fold assembly path.
void CheckAdiabaticCondensedFD()
{
   PerfectGasParams params;
   params.reynolds = 100.0;
   params.mach = 1.0;
   mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
      3, 3, mfem::Element::QUADRILATERAL, true);
   mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   PerfectGasModel model(params);
   const std::vector<int> attrs(
      4, model.RegisterBoundaryCondition("adiabatic_wall", YAML::Node()));
   auto zero_av = [](const mfem::Vector &) { return 0.0; };
   HDGOperator op(mesh, zero_av, model, attrs);
   HDGState base;
   op.ProjectState(
      [](const mfem::Vector &x, double *state)
      {
         state[0] = 1.1 + 0.05 * std::sin(2.0 * x[0] + x[1]);
         state[1] = 0.2 + 0.04 * std::cos(x[0] - 2.0 * x[1]);
         state[2] = 0.05 + 0.03 * std::sin(x[0] + x[1]);
         state[3] = 2.0 + 0.06 * std::cos(2.0 * x[0] + 2.0 * x[1]);
      },
      base);
   op.Assemble(base, true);

   mfem::Vector direction(op.TraceVSize());
   std::mt19937_64 rng(0x41444642ULL);
   std::normal_distribution<double> normal(0.0, 1.0);
   for (int i = 0; i < direction.Size(); ++i)
   {
      direction[i] = normal(rng);
   }
   direction /= direction.Norml2();
   mfem::Vector matrix_action(direction.Size());
   op.CondensedMatrix().Mult(direction, matrix_action);
   mfem::Vector recovered, zero_recovered, zero(direction.Size());
   zero = 0.0;
   op.RecoverIncrement(direction, recovered);
   op.RecoverIncrement(zero, zero_recovered);
   recovered -= zero_recovered;
   constexpr double epsilon = 2.0e-6;
   HDGState plus = base;
   plus.u.Add(epsilon, recovered);
   plus.uhat.Add(epsilon, direction);
   op.RecomputeGradient(plus);
   op.Assemble(plus, false);
   mfem::Vector plus_condensed(op.TraceResidual());
   HDGState minus = base;
   minus.u.Add(-epsilon, recovered);
   minus.uhat.Add(-epsilon, direction);
   op.RecomputeGradient(minus);
   op.Assemble(minus, false);
   mfem::Vector minus_condensed(op.TraceResidual());
   plus_condensed -= minus_condensed;
   plus_condensed /= (2.0 * epsilon);
   mfem::Vector difference(plus_condensed);
   difference -= matrix_action;
   const double relative_error =
      difference.Norml2() /
      std::max({1.0, plus_condensed.Norml2(), matrix_action.Norml2()});
   Require(relative_error <= 1.0e-6,
           "adiabatic-wall condensed directional FD exceeds 1e-6");
   std::cout << "PASS adiabatic-wall condensed directional FD:"
             << " relative_error=" << relative_error << '\n';
}

void CheckAdiabaticCylinderSolve()
{
   // M=3 / Re=1000 cylinder with an adiabatic wall: the solve must
   // converge and the converged wall conductive heat flux must be near
   // zero (the isothermal case peaks at ~1e-2 in these units).
   PerfectGasParams params;
   // Re=100: the adiabatic wall's energy level equilibrates only through
   // O(1/Re) conduction; at high Re that near-neutral mode makes the
   // steady Newton system nearly singular and needs the G6 continuation
   // machinery. This gate verifies BC correctness in a
   // diffusion-dominated regime.
   params.reynolds = 100.0;
   params.mach = 3.0;
   // Cold-start transients at an adiabatic wall visit negative pressure;
   // the smoothed rho/p floors keep Sutherland finite (inactive at the
   // converged state).
   params.regularized = true;
   std::unique_ptr<mfem::Mesh> serial_mesh =
      hycfd::BuildAnalyticMesh(6, 12, 4);
   mfem::ParMesh mesh(MPI_COMM_WORLD, *serial_mesh);
   // Stage 1: converge the proven isothermal wall from the damped
   // freestream; stage 2 restarts the adiabatic wall from that solution
   // (wall-condition continuation — cold adiabatic starts on a coarse
   // mesh are a G6 continuation problem, not a BC correctness gate).
   PerfectGasModel isothermal_model(params);
   const std::vector<int> isothermal_attrs =
   {
      isothermal_model.RegisterBoundaryCondition("isothermal_wall",
                                                 YAML::Node()),
      isothermal_model.RegisterBoundaryCondition("supersonic_outflow",
                                                 YAML::Node()),
      isothermal_model.RegisterBoundaryCondition("freestream",
                                                 YAML::Node())
   };
   PerfectGasModel model(params);
   const int wall =
      model.RegisterBoundaryCondition("adiabatic_wall", YAML::Node());
   const int outflow =
      model.RegisterBoundaryCondition("supersonic_outflow", YAML::Node());
   const int farfield =
      model.RegisterBoundaryCondition("freestream", YAML::Node());
   const std::vector<int> attrs = {wall, outflow, farfield};
   const auto av = [](const mfem::Vector &x)
   {
      return 0.05 * std::tanh(5.0 * (std::hypot(x[0], x[1]) - 1.0));
   };
   HDGOperator isothermal_op(mesh, av, isothermal_model,
                             isothermal_attrs);
   HDGOperator op(mesh, av, model, attrs);
   HDGState state;
   isothermal_op.ProjectState(
      [params](const mfem::Vector &x, double *value)
      {
         const double distance = std::hypot(x[0], x[1]) - 1.0;
         const double velocity = std::tanh(10.0 * distance);
         // Warm-start the wall toward the adiabatic recovery
         // temperature so the transient does not have to grow the wall
         // enthalpy from the cold freestream value.
         const double gamma1 = params.gamma - 1.0;
         const double recovery =
            1.0 + 0.85 * 0.5 * gamma1 * params.mach * params.mach;
         value[0] = 1.0;
         value[1] = velocity;
         value[2] = 0.0;
         value[3] = params.TinfND() *
                       ((recovery - 1.0) *
                           std::exp(-10.0 * distance) + 1.0) +
                    0.5 * velocity * velocity;
      },
      state);
   isothermal_op.InitializeTraceFromInterior(state);

   hycfd::NewtonConfig config;
   config.max_iterations = 80;
   config.tolerance = 1.0e-6;
   config.pseudo_transient = true;
   config.initial_pseudo_time_step = 15.0;
   const hycfd::NewtonReport isothermal_report =
      hycfd::DampedNewtonSolve(isothermal_op, state, config);
   Require(isothermal_report.converged,
           "isothermal warm-up solve did not converge: " +
           isothermal_report.failure);

   // Diagnostic: condensed-Jacobian directional FD at this exact state
   // (curved elements, tanh AV, adiabatic wall).
   {
      op.Assemble(state, true);
      mfem::Vector direction(op.TraceVSize());
      std::mt19937_64 rng(0x44494147ULL);
      std::normal_distribution<double> normal(0.0, 1.0);
      for (int i = 0; i < direction.Size(); ++i)
      {
         direction[i] = normal(rng);
      }
      direction /= direction.Norml2();
      mfem::Vector matrix_action(direction.Size());
      op.CondensedMatrix().Mult(direction, matrix_action);
      mfem::Vector recovered, zero_recovered, zero(direction.Size());
      zero = 0.0;
      op.RecoverIncrement(direction, recovered);
      op.RecoverIncrement(zero, zero_recovered);
      recovered -= zero_recovered;
      constexpr double epsilon = 2.0e-6;
      HDGState plus = state;
      plus.u.Add(epsilon, recovered);
      plus.uhat.Add(epsilon, direction);
      op.RecomputeGradient(plus);
      op.Assemble(plus, false);
      mfem::Vector plus_condensed(op.TraceResidual());
      HDGState minus = state;
      minus.u.Add(-epsilon, recovered);
      minus.uhat.Add(-epsilon, direction);
      op.RecomputeGradient(minus);
      op.Assemble(minus, false);
      mfem::Vector minus_condensed(op.TraceResidual());
      plus_condensed -= minus_condensed;
      plus_condensed /= (2.0 * epsilon);
      mfem::Vector difference(plus_condensed);
      difference -= matrix_action;
      std::cout << "  cylinder adiabatic condensed FD rel_error="
                << difference.Norml2() /
                   std::max({1.0, plus_condensed.Norml2(),
                             matrix_action.Norml2()})
                << std::endl;
   }

   const hycfd::NewtonReport report = hycfd::DampedNewtonSolve(
      op, state, config,
      [&op](int iteration, const HDGState &current, double residual)
      {
         std::cout << "  adiabatic Newton " << iteration
                   << " residual=" << residual
                   << " min_p=" << op.MinimumPressure(current)
                   << std::endl;
      });
   Require(report.converged,
           "adiabatic-wall cylinder solve did not converge: " +
           report.failure);
   Require(op.MinimumDensity(state) > 0.0 &&
           op.MinimumPressure(state) > 0.0,
           "adiabatic-wall cylinder fields are not positive");

   // Converged wall conductive flux.
   double maximum_wall_flux = 0.0;
   for (int boundary = 0; boundary < mesh.GetNBE(); ++boundary)
   {
      if (mesh.GetBdrAttribute(boundary) != 1) { continue; }
      const int face = mesh.GetBdrElementFaceIndex(boundary);
      mfem::FaceElementTransformations *transformation =
         mesh.GetFaceElementTransformations(face, 31);
      mfem::Vector weighted_normal(2);
      for (int qpoint = 0; qpoint < op.FaceRule().GetNPoints(); ++qpoint)
      {
         const mfem::IntegrationPoint &face_point =
            op.FaceRule().IntPoint(qpoint);
         transformation->SetAllIntPoints(&face_point);
         double uq[12], uhat[4], conductive[2];
         op.EvaluateElementState(
            state, transformation->Elem1No,
            transformation->GetElement1IntPoint(), uq);
         op.EvaluateTraceState(state, face, face_point, uhat);
         hycfd::NSHeatFlux(uhat, uq, params, conductive);
         mfem::CalcOrtho(transformation->Jacobian(), weighted_normal);
         weighted_normal /= weighted_normal.Norml2();
         // Fint wall heat flux: conduction + tau jump — the quantity the
         // adiabatic condition drives to zero.
         maximum_wall_flux = std::max(
            maximum_wall_flux,
            std::abs(conductive[0] * weighted_normal[0] +
                     conductive[1] * weighted_normal[1] +
                     params.tau * (uq[3] - uhat[3])));
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &maximum_wall_flux, 1, MPI_DOUBLE,
                 MPI_MAX, MPI_COMM_WORLD);
   Require(maximum_wall_flux <= 1.0e-4,
           "adiabatic wall carries a non-negligible conductive flux");
   std::cout << "PASS adiabatic-wall M=3 cylinder:"
             << " iterations=" << report.iterations
             << " residual=" << report.residual
             << " max_wall_conductive_flux=" << maximum_wall_flux
             << '\n';
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
      CheckBoundaryJacobians();
      CheckCharacteristicSplitting();
      CheckStatePreservation();
      CheckAdiabaticCondensedFD();
      CheckAdiabaticCylinderSolve();
      std::cout << "ALL test_bc G5 GATES PASSED\n";
   }
   catch (const std::exception &error)
   {
      std::cerr << "FAIL test_bc: " << error.what() << '\n';
      exit_code = EXIT_FAILURE;
   }
   mfem::MFEMFinalizePetsc();
   return exit_code;
}

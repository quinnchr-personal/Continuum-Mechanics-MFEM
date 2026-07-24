#include "exasim_mesh.hpp"
#include "hdg_newton.hpp"
#include "hdg_ns_operator.hpp"

#include "mfem.hpp"

#include <algorithm>
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

using hdg_ns::HDGNavierStokesOperator;
using hdg_ns::HDGState;
using hdg_ns::NSParams;

void Require(bool condition, const std::string &message)
{
   if (!condition) { throw std::runtime_error(message); }
}

std::array<double, 2> AnalyticCoordinates(const mfem::Vector &x)
{
   constexpr double pi = 3.141592653589793238462643383279502884;
   double theta = std::atan2(x[1], x[0]);
   if (theta < 0.5 * pi) { theta += 2.0 * pi; }
   const double p2 = (1.5 * pi - theta) / pi;
   const double outer_radius = 4.7 + 1.7 * std::cos(theta);
   const double radius = std::hypot(x[0], x[1]);
   const double stretched =
      (radius - outer_radius) / (1.0 - outer_radius);
   const double p1 =
      -std::log(1.0 - stretched * (1.0 - std::exp(-5.0))) / 5.0;
   return {{p1, p2}};
}

void ExactMMS(const mfem::Vector &x, double state[4])
{
   const auto parameter = AnalyticCoordinates(x);
   const double p1 = parameter[0];
   const double p2 = parameter[1];
   state[0] = 1.0 + 0.020 * std::pow(p1, 5);
   state[1] = 0.2 + 0.015 * std::pow(p2, 5);
   state[2] = 0.05 + 0.010 * std::pow(p1, 3) * std::pow(p2, 2);
   state[3] = 2.6 + 0.030 * std::pow(p1, 2) * std::pow(p2, 3);
}

void ExactMMSUQ(const mfem::Vector &x, double uq[12])
{
   ExactMMS(x, uq);
   constexpr double h = 2.0e-4;
   for (int direction = 0; direction < 2; ++direction)
   {
      mfem::Vector xm2(x), xm1(x), xp1(x), xp2(x);
      xm2[direction] -= 2.0 * h;
      xm1[direction] -= h;
      xp1[direction] += h;
      xp2[direction] += 2.0 * h;
      double um2[4], um1[4], up1[4], up2[4];
      ExactMMS(xm2, um2);
      ExactMMS(xm1, um1);
      ExactMMS(xp1, up1);
      ExactMMS(xp2, up2);
      for (int component = 0; component < 4; ++component)
      {
         const double derivative =
            (um2[component] - 8.0 * um1[component] +
             8.0 * up1[component] - up2[component]) / (12.0 * h);
         uq[4 + 4 * direction + component] = -derivative;
      }
   }
}

HDGNavierStokesOperator::StateFunction ManufacturedSource(
   const NSParams &params)
{
   return [params](const mfem::Vector &x, double source[4])
   {
      // Fourth-order centered differentiation of the analytic physical flux.
      // The source is test-only; h was selected so its error is far below the
      // Q4 discretization error on all three gate meshes.
      constexpr double h = 1.0e-4;
      auto flux_at = [&params](double px, double py, double flux[8])
      {
         mfem::Vector point(2);
         point[0] = px;
         point[1] = py;
         double uq[12];
         ExactMMSUQ(point, uq);
         hdg_ns::NSFlux(uq, 0.0, params, flux);
      };
      double xm2[8], xm1[8], xp1[8], xp2[8];
      double ym2[8], ym1[8], yp1[8], yp2[8];
      flux_at(x[0] - 2.0 * h, x[1], xm2);
      flux_at(x[0] - h, x[1], xm1);
      flux_at(x[0] + h, x[1], xp1);
      flux_at(x[0] + 2.0 * h, x[1], xp2);
      flux_at(x[0], x[1] - 2.0 * h, ym2);
      flux_at(x[0], x[1] - h, ym1);
      flux_at(x[0], x[1] + h, yp1);
      flux_at(x[0], x[1] + 2.0 * h, yp2);
      for (int component = 0; component < 4; ++component)
      {
         const double dx =
            (xm2[component] - 8.0 * xm1[component] +
             8.0 * xp1[component] - xp2[component]) / (12.0 * h);
         const double dy =
            (ym2[4 + component] - 8.0 * ym1[4 + component] +
             8.0 * yp1[4 + component] - yp2[4 + component]) /
            (12.0 * h);
         source[component] = dx + dy;
      }
   };
}

NSParams BenignParams()
{
   NSParams params;
   params.mu[1] = 100.0;
   params.mu[3] = 1.0;
   params.mu[4] = 1.0;
   params.mu[5] = 0.2;
   params.mu[6] = 0.05;
   params.mu[7] = 2.6;
   return params;
}

void CheckOrientationAndFreestream()
{
   std::unique_ptr<mfem::Mesh> mesh =
      hdg_ns::BuildAnalyticMesh(3, 6, 4);
   auto zero_av = [](const mfem::Vector &) { return 0.0; };
   HDGNavierStokesOperator orientation_op(*mesh, zero_av);
   const double orientation_error = orientation_op.TraceOrientationError();
   Require(orientation_error <= 2.0e-13,
           "trace orientation is inconsistent on an interior face");
   std::cout << "PASS M2(a) trace orientation on all interior faces:"
             << " faces=" << mesh->GetNumFaces() - mesh->GetNBE()
             << " max_error=" << orientation_error << '\n';

   HDGNavierStokesOperator freestream_op(
      *mesh, zero_av, NSParams(), {{1, 1, 1}});
   HDGState freestream = freestream_op.NewState();
   const NSParams defaults;
   const double state[4] =
   {
      defaults.mu[4], defaults.mu[5], defaults.mu[6], defaults.mu[7]
   };
   freestream_op.SetConstantState(state, freestream);
   const hdg_ns::HDGResidualNorms norms =
      freestream_op.Assemble(freestream, false);
   Require(norms.Total() <= 1.0e-10,
           "freestream residual exceeds 1e-10");
   std::cout << "PASS M2(d) freestream preservation with ib=1 on attrs 1/2/3:"
             << " |Ru|=" << norms.volume
             << " |Rh|=" << norms.trace
             << " total=" << norms.Total() << '\n';
}

void CheckCondensedDirectionalFD()
{
   std::unique_ptr<mfem::Mesh> mesh =
      hdg_ns::BuildAnalyticMesh(2, 4, 4);
   auto zero_av = [](const mfem::Vector &) { return 0.0; };
   const NSParams params = BenignParams();
   HDGNavierStokesOperator op(*mesh, zero_av, params);
   op.SetManufacturedSource(ManufacturedSource(params));
   op.SetDirichletStateOverride(ExactMMS);
   HDGState base;
   op.ProjectState(ExactMMS, base);
   op.Assemble(base, true);

   mfem::Vector direction(op.TraceVSize());
   std::mt19937_64 rng(0x4832434644ULL);
   std::normal_distribution<double> normal(0.0, 1.0);
   for (int i = 0; i < direction.Size(); ++i) { direction[i] = normal(rng); }
   direction /= direction.Norml2();

   mfem::Vector matrix_action(direction.Size());
   op.CondensedMatrix().Mult(direction, matrix_action);
   mfem::Vector recovered, zero_recovered, zero(direction.Size());
   zero = 0.0;
   op.RecoverIncrement(direction, recovered);
   op.RecoverIncrement(zero, zero_recovered);
   recovered -= zero_recovered; // exactly -A^{-1} F direction

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
           "global condensed residual directional FD exceeds 1e-6");
   std::cout << "PASS M2(b) global condensed directional FD:"
             << " relative_error=" << relative_error
             << " |FD|=" << plus_condensed.Norml2()
             << " |Hc*v|=" << matrix_action.Norml2() << '\n';
}

void CheckMMSConvergence()
{
   const NSParams params = BenignParams();
   const std::array<int, 3> nr{{2, 4, 8}};
   const std::array<int, 3> nc{{4, 8, 16}};
   std::array<double, 3> errors{};
   std::array<double, 3> projection_errors{};
   std::array<int, 3> elements{};
   std::array<int, 3> iterations{};
   std::array<double, 3> residuals{};
   for (int level = 0; level < 3; ++level)
   {
      std::unique_ptr<mfem::Mesh> mesh =
         hdg_ns::BuildAnalyticMesh(nr[level], nc[level], 4);
      auto zero_av = [](const mfem::Vector &) { return 0.0; };
      HDGNavierStokesOperator op(*mesh, zero_av, params);
      op.SetManufacturedSource(ManufacturedSource(params));
      op.SetDirichletStateOverride(ExactMMS);
      HDGState solution;
      op.ProjectState(ExactMMS, solution);
      projection_errors[level] = op.L2Error(solution, ExactMMS);

      hdg_ns::NewtonConfig config;
      config.max_iterations = 12;
      config.tolerance = 1.0e-10;
      const hdg_ns::NewtonReport report =
         hdg_ns::DampedNewtonSolve(op, solution, config);
      Require(report.converged,
              "MMS Newton solve did not reach 1e-10");
      errors[level] = op.L2Error(solution, ExactMMS);
      elements[level] = mesh->GetNE();
      iterations[level] = report.iterations;
      residuals[level] = report.residual;
   }

   const double order_01 =
      std::log(errors[0] / errors[1]) / std::log(2.0);
   const double order_12 =
      std::log(errors[1] / errors[2]) / std::log(2.0);
   std::cout << "M2(c) MMS convergence table\n"
             << std::setw(8) << "level"
             << std::setw(12) << "elements"
             << std::setw(24) << "L2(u)"
             << std::setw(24) << "order"
             << std::setw(24) << "projection"
             << std::setw(10) << "Newton"
             << std::setw(24) << "residual" << '\n'
             << std::scientific << std::setprecision(16);
   for (int level = 0; level < 3; ++level)
   {
      std::cout << std::setw(8) << level
                << std::setw(12) << elements[level]
                << std::setw(24) << errors[level];
      if (level == 0) { std::cout << std::setw(24) << "-"; }
      else
      {
         const double order = level == 1 ? order_01 : order_12;
         std::cout << std::setw(24) << order;
      }
      std::cout << std::setw(24) << projection_errors[level]
                << std::setw(10) << iterations[level]
                << std::setw(24) << residuals[level] << '\n';
   }
   Require(order_01 >= 4.8 && order_12 >= 4.8,
           "MMS L2 convergence order is below 4.8");
   std::cout << "PASS M2(c) MMS L2 orders:"
             << " p01=" << order_01 << " p12=" << order_12 << '\n';
}

} // namespace

int main(int argc, char *argv[])
{
   int exit_code = EXIT_SUCCESS;
   mfem::Mpi::Init(argc, argv);
   mfem::Hypre::Init();
   mfem::MFEMInitializePetsc(
      &argc, &argv, "Input/petsc.opts", nullptr);
   try
   {
      if (mfem::Mpi::WorldSize() != 1)
      {
         throw std::runtime_error("M2 acceptance tests require np=1");
      }
      std::cout << std::setprecision(17);
      CheckOrientationAndFreestream();
      CheckCondensedDirectionalFD();
      CheckMMSConvergence();
      std::cout << "ALL test_hdg_mms M2(a-d) GATES PASSED\n";
   }
   catch (const std::exception &error)
   {
      std::cerr << "FAIL test_hdg_mms: " << error.what() << '\n';
      exit_code = EXIT_FAILURE;
   }
   mfem::MFEMFinalizePetsc();
   return exit_code;
}

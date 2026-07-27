// G1 HDG-core gates: trace orientation, freestream preservation, global
// condensed-Jacobian directional FD, and MMS L2 convergence at p=4 on the
// analytic half-annulus meshes.
#include "discretization/hdg_operator.hpp"
#include "io/exasim_mesh.hpp"
#include "io/mesh_input.hpp"
#include "physics/perfect_gas_model.hpp"
#include "solvers/newton.hpp"

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

using hycfd::HDGOperator;
using hycfd::HDGState;
using hycfd::PerfectGasParams;

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

// State plus PHYSICAL gradient (q = +grad(u)) by fourth-order central FD.
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
         uq[4 + 4 * direction + component] = derivative;
      }
   }
}

HDGOperator::StateFunction ManufacturedSource(
   const PerfectGasParams &params)
{
   return [params](const mfem::Vector &x, double source[4])
   {
      // Fourth-order centered differentiation of the analytic physical flux.
      constexpr double h = 1.0e-4;
      auto flux_at = [&params](double px, double py, double flux[8])
      {
         mfem::Vector point(2);
         point[0] = px;
         point[1] = py;
         double uq[12];
         ExactMMSUQ(point, uq);
         hycfd::NSFlux(uq, 0.0, params, flux);
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

PerfectGasParams BenignParams()
{
   PerfectGasParams params;
   params.reynolds = 100.0;
   params.mach = 1.0;
   return params;
}

void CheckOrientationAndFreestream()
{
   std::unique_ptr<mfem::Mesh> serial_mesh =
      hycfd::BuildAnalyticMesh(3, 6, 4);
   mfem::ParMesh mesh(MPI_COMM_WORLD, *serial_mesh);
   auto zero_av = [](const mfem::Vector &) { return 0.0; };
   const PerfectGasParams defaults;
   hycfd::PerfectGasModel model(defaults);
   const std::vector<int> all_freestream(
      3, model.RegisterBoundaryCondition("freestream", YAML::Node()));
   HDGOperator orientation_op(mesh, zero_av, model, all_freestream);
   const double orientation_error = orientation_op.TraceOrientationError();
   Require(orientation_error <= 2.0e-13,
           "trace orientation is inconsistent on an interior face");
   std::cout << "PASS trace orientation on all interior faces:"
             << " faces=" << mesh.GetNumFaces() - mesh.GetNBE()
             << " max_error=" << orientation_error << '\n';

   HDGOperator freestream_op(
      mesh, zero_av, model, all_freestream);
   HDGState freestream = freestream_op.NewState();
   double state[4];
   defaults.Freestream(state);
   freestream_op.SetConstantState(state, freestream);
   const hycfd::HDGResidualNorms norms =
      freestream_op.Assemble(freestream, false);
   Require(norms.Total() <= 1.0e-10,
           "freestream residual exceeds 1e-10");
   std::cout << "PASS freestream preservation with ib=1 on attrs 1/2/3:"
             << " |Ru|=" << norms.volume
             << " |Rh|=" << norms.trace
             << " total=" << norms.Total() << '\n';
}

void CheckCondensedDirectionalFD()
{
   std::unique_ptr<mfem::Mesh> serial_mesh =
      hycfd::BuildAnalyticMesh(2, 4, 4);
   mfem::ParMesh mesh(MPI_COMM_WORLD, *serial_mesh);
   auto zero_av = [](const mfem::Vector &) { return 0.0; };
   const PerfectGasParams params = BenignParams();
   hycfd::PerfectGasModel model(params);
   const std::vector<int> all_freestream(
      3, model.RegisterBoundaryCondition("freestream", YAML::Node()));
   HDGOperator op(mesh, zero_av, model, all_freestream);
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
   std::cout << "PASS global condensed directional FD:"
             << " relative_error=" << relative_error
             << " |FD|=" << plus_condensed.Norml2()
             << " |Hc*v|=" << matrix_action.Norml2() << '\n';
}

void CheckMMSConvergence()
{
   const PerfectGasParams params = BenignParams();
   const std::array<int, 3> nr{{2, 4, 8}};
   const std::array<int, 3> nc{{4, 8, 16}};
   std::array<double, 3> errors{};
   std::array<double, 3> projection_errors{};
   std::array<int, 3> elements{};
   std::array<int, 3> iterations{};
   std::array<double, 3> residuals{};
   for (int level = 0; level < 3; ++level)
   {
      std::unique_ptr<mfem::Mesh> serial_mesh =
         hycfd::BuildAnalyticMesh(nr[level], nc[level], 4);
      mfem::ParMesh mesh(MPI_COMM_WORLD, *serial_mesh);
      auto zero_av = [](const mfem::Vector &) { return 0.0; };
      hycfd::PerfectGasModel model(params);
      const std::vector<int> all_freestream(
         3, model.RegisterBoundaryCondition("freestream", YAML::Node()));
      HDGOperator op(mesh, zero_av, model, all_freestream);
      op.SetManufacturedSource(ManufacturedSource(params));
      op.SetDirichletStateOverride(ExactMMS);
      HDGState solution;
      op.ProjectState(ExactMMS, solution);
      projection_errors[level] = op.L2Error(solution, ExactMMS);

      hycfd::NewtonConfig config;
      config.max_iterations = 12;
      config.tolerance = 1.0e-10;
      const hycfd::NewtonReport report =
         hycfd::DampedNewtonSolve(op, solution, config);
      Require(report.converged,
              "MMS Newton solve did not reach 1e-10");
      errors[level] = op.L2Error(solution, ExactMMS);
      elements[level] = mesh.GetNE();
      iterations[level] = report.iterations;
      residuals[level] = report.residual;
   }

   const double order_01 =
      std::log(errors[0] / errors[1]) / std::log(2.0);
   const double order_12 =
      std::log(errors[1] / errors[2]) / std::log(2.0);
   std::cout << "MMS convergence table\n"
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
   std::cout << "PASS MMS L2 orders:"
             << " p01=" << order_01 << " p12=" << order_12 << '\n';
}

// ---------------------------------------------------------------------------
// G3 gates: manufactured solution on the unit square for runtime order and
// triangle/quadrilateral/mixed meshes.

void ExactSquareMMS(const mfem::Vector &x, double *state)
{
   // Incommensurate frequencies and phase shifts: a symmetric
   // sin(pi x)sin(pi y) field on uniform Cartesian grids produces
   // superconvergent cancellations at even orders that corrupt the
   // measured rates.
   constexpr double pi = 3.141592653589793238462643383279502884;
   const double sx = std::sin(1.1 * pi * x[0] + 0.3);
   const double sy = std::sin(0.9 * pi * x[1] - 0.2);
   const double cx = std::cos(1.2 * pi * x[0] + 0.1);
   const double cy = std::cos(0.8 * pi * x[1] + 0.4);
   state[0] = 1.2 + 0.1 * sx * sy;
   state[1] = 0.25 + 0.08 * cx * sy;
   state[2] = 0.05 + 0.06 * sx * cy;
   state[3] = 2.4 + 0.12 * cx * cy;
}

// State plus PHYSICAL gradient (q = +grad(u)) by fourth-order central FD.
void ExactSquareMMSUQ(const mfem::Vector &x, double uq[12])
{
   ExactSquareMMS(x, uq);
   constexpr double h = 2.0e-4;
   for (int direction = 0; direction < 2; ++direction)
   {
      mfem::Vector xm2(x), xm1(x), xp1(x), xp2(x);
      xm2[direction] -= 2.0 * h;
      xm1[direction] -= h;
      xp1[direction] += h;
      xp2[direction] += 2.0 * h;
      double um2[4], um1[4], up1[4], up2[4];
      ExactSquareMMS(xm2, um2);
      ExactSquareMMS(xm1, um1);
      ExactSquareMMS(xp1, up1);
      ExactSquareMMS(xp2, up2);
      for (int component = 0; component < 4; ++component)
      {
         uq[4 + 4 * direction + component] =
            (um2[component] - 8.0 * um1[component] +
             8.0 * up1[component] - up2[component]) / (12.0 * h);
      }
   }
}

HDGOperator::StateFunction ManufacturedSquareSource(
   const PerfectGasParams &params)
{
   return [params](const mfem::Vector &x, double *source)
   {
      constexpr double h = 1.0e-4;
      auto flux_at = [&params](double px, double py, double flux[8])
      {
         mfem::Vector point(2);
         point[0] = px;
         point[1] = py;
         double uq[12];
         ExactSquareMMSUQ(point, uq);
         hycfd::NSFlux(uq, 0.0, params, flux);
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

// Unit square split into one quadrilateral (left half) and two triangles
// (right half); boundary attributes 1..4.
mfem::Mesh BuildMixedUnitSquareMesh()
{
   mfem::Mesh mesh(2, 6, 3, 6, 2);
   mesh.AddVertex(0.0, 0.0);
   mesh.AddVertex(0.5, 0.0);
   mesh.AddVertex(1.0, 0.0);
   mesh.AddVertex(0.0, 1.0);
   mesh.AddVertex(0.5, 1.0);
   mesh.AddVertex(1.0, 1.0);
   const int quad[4] = {0, 1, 4, 3};
   const int tri1[3] = {1, 2, 5};
   const int tri2[3] = {1, 5, 4};
   mesh.AddQuad(quad, 1);
   mesh.AddTriangle(tri1, 1);
   mesh.AddTriangle(tri2, 1);
   mesh.AddBdrSegment(0, 1, 1);
   mesh.AddBdrSegment(1, 2, 1);
   mesh.AddBdrSegment(2, 5, 2);
   mesh.AddBdrSegment(5, 4, 3);
   mesh.AddBdrSegment(4, 3, 3);
   mesh.AddBdrSegment(3, 0, 4);
   mesh.FinalizeMesh();
   return mesh;
}

void CheckSquareMMSOrders()
{
   const PerfectGasParams params = BenignParams();
   for (const mfem::Element::Type type :
        {mfem::Element::QUADRILATERAL, mfem::Element::TRIANGLE})
   {
      for (int order = 1; order <= 4; ++order)
      {
         // Coarse levels are pre-asymptotic; gate on the finest pair.
         const int levels = 4;
         std::array<double, 4> errors{};
         for (int level = 0; level < levels; ++level)
         {
            const int n = 2 << level;
            mfem::Mesh serial_mesh =
               mfem::Mesh::MakeCartesian2D(n, n, type, true);
            mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
            hycfd::PerfectGasModel model(params);
            const std::vector<int> attr_map(
               4, model.RegisterBoundaryCondition("freestream",
                                                  YAML::Node()));
            hycfd::HDGOptions options;
            options.order = order;
            options.quadrature_increment = 2;
            auto zero_av = [](const mfem::Vector &) { return 0.0; };
            HDGOperator op(mesh, zero_av, model, attr_map, options);
            op.SetManufacturedSource(ManufacturedSquareSource(params));
            op.SetDirichletStateOverride(ExactSquareMMS);
            HDGState solution;
            op.ProjectState(ExactSquareMMS, solution);
            hycfd::NewtonConfig config;
            config.max_iterations = 15;
            config.tolerance = 1.0e-10;
            const hycfd::NewtonReport report =
               hycfd::DampedNewtonSolve(op, solution, config);
            Require(report.converged,
                    "square MMS Newton solve did not converge");
            errors[level] = op.L2Error(solution, ExactSquareMMS);
         }
         std::cout << "square MMS "
                   << (type == mfem::Element::QUADRILATERAL ?
                       "quad" : "tri")
                   << " p=" << order << " L2=";
         for (int level = 0; level < levels; ++level)
         {
            std::cout << errors[level] << ' ';
         }
         std::cout << "orders=";
         double final_order = 0.0;
         for (int level = 1; level < levels; ++level)
         {
            final_order =
               std::log(errors[level - 1] / errors[level]) /
               std::log(2.0);
            std::cout << final_order << ' ';
         }
         std::cout << std::endl;
         // Measured asymptotics on affine meshes: odd p converge at
         // ~p+1.1, even p settle at ~p+0.7 (parity-dependent HDG
         // superconvergence; the curved annulus gate below still holds
         // p=4 to >= 4.8). Gate at p+0.5: any assembly defect drops the
         // rate to p or below.
         Require(final_order >= order + 0.5,
                 "square MMS convergence order below p+0.5");
      }
   }
   std::cout << "PASS square MMS orders p=1..4 on quad and tri meshes"
             << std::endl;
}

void CheckMixedMeshAndFileRoundtrip()
{
   const PerfectGasParams params = BenignParams();

   // (a) Mixed tri/quad mesh: freestream preservation and condensed FD.
   mfem::Mesh mixed_serial = BuildMixedUnitSquareMesh();
   mfem::ParMesh mixed(MPI_COMM_WORLD, mixed_serial);
   hycfd::PerfectGasModel model(params);
   const std::vector<int> attr_map(
      4, model.RegisterBoundaryCondition("freestream", YAML::Node()));
   hycfd::HDGOptions options;
   options.order = 3;
   auto zero_av = [](const mfem::Vector &) { return 0.0; };
   HDGOperator op(mixed, zero_av, model, attr_map, options);
   double state[4];
   params.Freestream(state);
   HDGState freestream;
   op.SetConstantState(state, freestream);
   const hycfd::HDGResidualNorms norms = op.Assemble(freestream, false);
   Require(norms.Total() <= 1.0e-10,
           "mixed-mesh freestream residual exceeds 1e-10");

   op.SetManufacturedSource(ManufacturedSquareSource(params));
   op.SetDirichletStateOverride(ExactSquareMMS);
   HDGState base;
   op.ProjectState(ExactSquareMMS, base);
   op.Assemble(base, true);
   mfem::Vector direction(op.TraceVSize());
   std::mt19937_64 rng(0x4d495845ULL);
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
           "mixed-mesh condensed directional FD exceeds 1e-6");

   // (b) Mesh file round trip: save the curved analytic mesh, reload it
   // through the file reader, and require freestream preservation.
   std::unique_ptr<mfem::Mesh> annulus =
      hycfd::BuildAnalyticMesh(3, 6, 4);
   const std::string path = "mesh_roundtrip_check.mesh";
   annulus->Save(path.c_str());
   std::unique_ptr<mfem::Mesh> reloaded = hycfd::LoadMeshFile(path);
   Require(reloaded->GetNE() == annulus->GetNE(),
           "mesh file round trip changed the element count");
   mfem::ParMesh reloaded_par(MPI_COMM_WORLD, *reloaded);
   hycfd::PerfectGasModel file_model(params);
   const std::vector<int> file_attrs(
      3, file_model.RegisterBoundaryCondition("freestream",
                                              YAML::Node()));
   HDGOperator file_op(reloaded_par, zero_av, file_model, file_attrs);
   HDGState file_freestream;
   file_op.SetConstantState(state, file_freestream);
   const hycfd::HDGResidualNorms file_norms =
      file_op.Assemble(file_freestream, false);
   Require(file_norms.Total() <= 1.0e-10,
           "file-mesh freestream residual exceeds 1e-10");
   std::cout << "PASS mixed tri/quad mesh (freestream "
             << norms.Total() << ", jac FD " << relative_error
             << ") and mesh-file round trip (freestream "
             << file_norms.Total() << ")" << std::endl;
}

// Delegates everything to PerfectGasModel except Flux, which falls back to
// the base-class finite-difference Jacobian: proves models without analytic
// Jacobians can converge through the same operator.
class FDOnlyPerfectGas final : public hycfd::PhysicsModel
{
public:
   explicit FDOnlyPerfectGas(const PerfectGasParams &params)
      : inner_(params) {}
   int NumComponents() const override { return inner_.NumComponents(); }
   int Dim() const override { return inner_.Dim(); }
   void FluxValue(const double *uq, double av,
                  double *flux) const override
   {
      inner_.FluxValue(uq, av, flux);
   }
   int RegisterBoundaryCondition(const std::string &type,
                                 const YAML::Node &bc_params) override
   {
      return inner_.RegisterBoundaryCondition(type, bc_params);
   }
   int NumBoundaryConditions() const override
   {
      return inner_.NumBoundaryConditions();
   }
   void BoundaryResidual(int bc_id, const double *uq, const double *uhat,
                         const double *normal, const double *x,
                         double *fb, double *dfbduq,
                         double *dfbduh) const override
   {
      inner_.BoundaryResidual(bc_id, uq, uhat, normal, x, fb,
                              dfbduq, dfbduh);
   }
   double MaxWaveSpeed(const double *u) const override
   {
      return inner_.MaxWaveSpeed(u);
   }
   bool IsAdmissible(const double *u) const override
   {
      return inner_.IsAdmissible(u);
   }
   void FreestreamState(double *u) const override
   {
      inner_.FreestreamState(u);
   }
   double Pressure(const double *u) const override
   {
      return inner_.Pressure(u);
   }
   double Temperature(const double *u) const override
   {
      return inner_.Temperature(u);
   }
   std::vector<std::string> OutputNames() const override
   {
      return inner_.OutputNames();
   }
   void Outputs(const double *u, double *values) const override
   {
      inner_.Outputs(u, values);
   }

private:
   hycfd::PerfectGasModel inner_;
};

void CheckFDJacobianFallback()
{
   const PerfectGasParams params = BenignParams();
   hycfd::PerfectGasModel analytic(params);
   FDOnlyPerfectGas fd(params);

   // (a) The FD fallback matches the analytic Jacobian pointwise.
   std::mt19937_64 rng(0x46444a4143ULL);
   std::uniform_real_distribution<double> rho_dist(0.5, 2.0);
   std::uniform_real_distribution<double> velocity_dist(-1.0, 1.0);
   std::uniform_real_distribution<double> pressure_dist(0.3, 2.0);
   std::uniform_real_distribution<double> gradient_dist(-0.5, 0.5);
   double max_error = 0.0;
   for (int sample = 0; sample < 50; ++sample)
   {
      double uq[12];
      uq[0] = rho_dist(rng);
      const double vx = velocity_dist(rng);
      const double vy = velocity_dist(rng);
      uq[1] = uq[0] * vx;
      uq[2] = uq[0] * vy;
      uq[3] = pressure_dist(rng) / 0.4 +
              0.5 * uq[0] * (vx * vx + vy * vy);
      for (int i = 4; i < 12; ++i) { uq[i] = gradient_dist(rng); }
      double flux_a[8], jac_a[96], flux_f[8], jac_f[96];
      analytic.Flux(uq, 0.02, flux_a, jac_a);
      fd.Flux(uq, 0.02, flux_f, jac_f);
      for (int i = 0; i < 8; ++i)
      {
         max_error = std::max(max_error,
                              std::abs(flux_a[i] - flux_f[i]));
      }
      for (int i = 0; i < 96; ++i)
      {
         max_error = std::max(
            max_error,
            std::abs(jac_a[i] - jac_f[i]) /
            std::max(1.0, std::abs(jac_a[i])));
      }
   }
   Require(max_error <= 2.0e-6,
           "FD Jacobian fallback disagrees with analytic Jacobian");

   // (b) The FD-only model converges the MMS problem through the operator.
   std::unique_ptr<mfem::Mesh> serial_mesh =
      hycfd::BuildAnalyticMesh(2, 4, 4);
   mfem::ParMesh mesh(MPI_COMM_WORLD, *serial_mesh);
   auto zero_av = [](const mfem::Vector &) { return 0.0; };
   FDOnlyPerfectGas fd_model(params);
   const std::vector<int> all_freestream(
      3, fd_model.RegisterBoundaryCondition("freestream", YAML::Node()));
   HDGOperator op(mesh, zero_av, fd_model, all_freestream);
   op.SetManufacturedSource(ManufacturedSource(params));
   op.SetDirichletStateOverride(ExactMMS);
   HDGState solution;
   op.ProjectState(ExactMMS, solution);
   hycfd::NewtonConfig config;
   config.max_iterations = 20;
   config.tolerance = 1.0e-9;
   const hycfd::NewtonReport report =
      hycfd::DampedNewtonSolve(op, solution, config);
   Require(report.converged,
           "FD-only physics model did not converge the MMS solve");
   const double error = op.L2Error(solution, ExactMMS);
   Require(error <= 5.0e-4,
           "FD-only model MMS error is out of range");
   std::cout << "PASS FD Jacobian fallback:"
             << " max_jac_error=" << max_error
             << " mms_iterations=" << report.iterations
             << " mms_L2=" << error << '\n';
}

} // namespace

int main(int argc, char *argv[])
{
   int exit_code = EXIT_SUCCESS;
   mfem::Mpi::Init(argc, argv);
   mfem::Hypre::Init();
   mfem::MFEMInitializePetsc(
      &argc, &argv, "input/petsc.opts", nullptr);
   try
   {
      if (mfem::Mpi::WorldSize() != 1)
      {
         throw std::runtime_error("MMS acceptance tests require np=1");
      }
      std::cout << std::setprecision(17);
      CheckOrientationAndFreestream();
      CheckCondensedDirectionalFD();
      CheckFDJacobianFallback();
      CheckSquareMMSOrders();
      CheckMixedMeshAndFileRoundtrip();
      CheckMMSConvergence();
      std::cout << "ALL test_mms G1 GATES PASSED\n";
   }
   catch (const std::exception &error)
   {
      std::cerr << "FAIL test_mms: " << error.what() << '\n';
      exit_code = EXIT_FAILURE;
   }
   mfem::MFEMFinalizePetsc();
   return exit_code;
}

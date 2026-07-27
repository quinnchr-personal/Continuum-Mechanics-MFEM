// G4 gate: np-consistency fixture. Run once with --write to record
// reference scalars from a single-rank run, then with --check under
// mpirun -np {2,4}: assembled residual norms, the first Newton increment,
// and the converged MMS solution must match the serial values.
#include "discretization/hdg_operator.hpp"
#include "physics/perfect_gas_model.hpp"
#include "solvers/newton.hpp"

#include "mfem.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
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

void ExactSquareMMS(const mfem::Vector &x, double *state)
{
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

struct FixtureValues
{
   double volume_residual = 0.0;
   double trace_residual = 0.0;
   double true_increment_norm = 0.0;
   double volume_increment_norm = 0.0;
   double newton_iterations = 0.0;
   double l2_error = 0.0;
};

FixtureValues RunFixture()
{
   mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
      6, 6, mfem::Element::QUADRILATERAL, true);
   mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   MPI_Comm comm = MPI_COMM_WORLD;

   PerfectGasParams params;
   params.reynolds = 100.0;
   params.mach = 1.0;
   hycfd::PerfectGasModel model(params);
   const std::vector<int> attr_map(
      4, model.RegisterBoundaryCondition("freestream", YAML::Node()));
   hycfd::HDGOptions options;
   options.order = 3;
   auto zero_av = [](const mfem::Vector &) { return 0.0; };
   HDGOperator op(mesh, zero_av, model, attr_map, options);
   op.SetManufacturedSource(ManufacturedSquareSource(params));
   op.SetDirichletStateOverride(ExactSquareMMS);

   HDGState state;
   op.ProjectState(ExactSquareMMS, state);

   FixtureValues values;
   const hycfd::HDGResidualNorms norms = op.Assemble(state, true);
   values.volume_residual = norms.volume;
   values.trace_residual = norms.trace;

   mfem::Vector true_increment;
   hycfd::SolveCondensedPetscDirect(
      op.CondensedParMatrix(), op.CondensedTrueRHS(), true_increment);
   values.true_increment_norm = std::sqrt(
      mfem::InnerProduct(comm, true_increment, true_increment));
   mfem::Vector trace_increment, volume_increment;
   op.ExpandTraceIncrement(true_increment, trace_increment);
   op.RecoverIncrement(trace_increment, volume_increment);
   double volume_sumsq = volume_increment * volume_increment;
   MPI_Allreduce(MPI_IN_PLACE, &volume_sumsq, 1, MPI_DOUBLE, MPI_SUM,
                 comm);
   values.volume_increment_norm = std::sqrt(volume_sumsq);

   hycfd::NewtonConfig config;
   config.max_iterations = 20;
   config.tolerance = 1.0e-9;
   const hycfd::NewtonReport report =
      hycfd::DampedNewtonSolve(op, state, config);
   Require(report.converged, "fixture Newton solve did not converge");
   values.newton_iterations = static_cast<double>(report.iterations);
   values.l2_error = op.L2Error(state, ExactSquareMMS);
   return values;
}

void CompareValue(const std::string &name, double actual,
                  double reference, double relative_tolerance)
{
   const double difference =
      std::abs(actual - reference) /
      std::max(1.0e-30, std::abs(reference));
   std::cout << "  " << name << ": np=" << actual
             << " reference=" << reference
             << " rel_diff=" << difference << '\n';
   Require(difference <= relative_tolerance,
           name + " differs from the serial reference");
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
      if (argc != 3 ||
          (std::string(argv[1]) != "--write" &&
           std::string(argv[1]) != "--check"))
      {
         throw std::runtime_error(
            "usage: test_parallel --write|--check <reference-file>");
      }
      const bool write = std::string(argv[1]) == "--write";
      const std::string path = argv[2];
      std::cout << std::setprecision(17);

      const FixtureValues values = RunFixture();

      if (write)
      {
         if (mfem::Mpi::Root())
         {
            std::ofstream output(path);
            output << std::setprecision(17)
                   << values.volume_residual << '\n'
                   << values.trace_residual << '\n'
                   << values.true_increment_norm << '\n'
                   << values.volume_increment_norm << '\n'
                   << values.newton_iterations << '\n'
                   << values.l2_error << '\n';
            if (!output)
            {
               throw std::runtime_error(
                  "failed to write parallel reference file");
            }
         }
         std::cout << "WROTE parallel fixture reference (np="
                   << mfem::Mpi::WorldSize() << ") to " << path << '\n';
      }
      else
      {
         std::ifstream input(path);
         FixtureValues reference;
         if (!(input >> reference.volume_residual >>
               reference.trace_residual >>
               reference.true_increment_norm >>
               reference.volume_increment_norm >>
               reference.newton_iterations >> reference.l2_error))
         {
            throw std::runtime_error(
               "cannot read parallel reference file: " + path);
         }
         std::cout << "np=" << mfem::Mpi::WorldSize()
                   << " consistency vs " << path << ":\n";
         CompareValue("volume_residual", values.volume_residual,
                      reference.volume_residual, 1.0e-11);
         CompareValue("trace_residual", values.trace_residual,
                      reference.trace_residual, 1.0e-11);
         CompareValue("true_increment_norm",
                      values.true_increment_norm,
                      reference.true_increment_norm, 1.0e-9);
         CompareValue("volume_increment_norm",
                      values.volume_increment_norm,
                      reference.volume_increment_norm, 1.0e-9);
         Require(values.newton_iterations == reference.newton_iterations,
                 "Newton iteration count differs from serial");
         CompareValue("l2_error", values.l2_error, reference.l2_error,
                      1.0e-4);
         std::cout << "PASS np=" << mfem::Mpi::WorldSize()
                   << " matches the serial reference\n";
      }
   }
   catch (const std::exception &error)
   {
      std::cerr << "FAIL test_parallel (rank "
                << mfem::Mpi::WorldRank() << "): " << error.what()
                << '\n';
      exit_code = EXIT_FAILURE;
   }
   mfem::MFEMFinalizePetsc();
   return exit_code;
}

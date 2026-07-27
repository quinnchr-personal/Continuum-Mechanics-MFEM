// G6 gates for the dilatation-sensor artificial viscosity: exactly zero
// on a resolved smooth flow (compact-support ramp), active under strong
// compression, and consistent field tabulation into the operator.
#include "discretization/av_sensor.hpp"
#include "discretization/hdg_operator.hpp"
#include "io/exasim_mesh.hpp"
#include "physics/perfect_gas_model.hpp"

#include "mfem.hpp"

#include <cmath>
#include <cstdlib>
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
using hycfd::PerfectGasModel;

void Require(bool condition, const std::string &message)
{
   if (!condition) { throw std::runtime_error(message); }
}

double MaxField(const mfem::ParGridFunction &field)
{
   double maximum = 0.0;
   for (int i = 0; i < field.Size(); ++i)
   {
      maximum = std::max(maximum, std::abs(field[i]));
   }
   MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MPI_DOUBLE, MPI_MAX,
                 MPI_COMM_WORLD);
   return maximum;
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
      PerfectGasParams params;
      params.reynolds = 100.0;
      params.mach = 2.0;
      mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
         4, 4, mfem::Element::QUADRILATERAL, true);
      mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
      PerfectGasModel model(params);
      const std::vector<int> attrs(
         4, model.RegisterBoundaryCondition("freestream", YAML::Node()));
      auto zero_av = [](const mfem::Vector &) { return 0.0; };
      HDGOperator op(mesh, zero_av, model, attrs);

      mfem::H1_FECollection sensor_fec(1, 2);
      mfem::ParFiniteElementSpace sensor_fes(&mesh, &sensor_fec);
      mfem::ParGridFunction av_field(&sensor_fes);
      hycfd::DilatationSensorOptions options;
      options.lambda = 0.2;

      // (a) Smooth, resolved flow: mild trig divergence must stay below
      // the ramp's compact support — AV identically zero.
      HDGState smooth;
      op.ProjectState(
         [](const mfem::Vector &x, double *state)
         {
            state[0] = 1.2 + 0.05 * std::sin(x[0] + x[1]);
            state[1] = 0.3 + 0.02 * std::cos(x[0]);
            state[2] = 0.05 + 0.02 * std::sin(x[1]);
            state[3] = 2.2 + 0.03 * std::cos(x[0] - x[1]);
         },
         smooth);
      hycfd::ComputeDilatationAV(op, smooth, model, options, av_field);
      const double smooth_max = MaxField(av_field);
      Require(smooth_max == 0.0,
              "sensor AV is nonzero on a resolved smooth flow");
      std::cout << "PASS sensor inactive on smooth flow: max_av="
                << smooth_max << '\n';

      // (b) Strong compression (div u ~ -3): the sensor must switch on at
      // the lambda*h*(|u|+c) scale.
      HDGState compressed;
      op.ProjectState(
         [](const mfem::Vector &x, double *state)
         {
            const double u = 3.0 * (0.5 - x[0]);
            state[0] = 1.0;
            state[1] = u;
            state[2] = 0.0;
            state[3] = 2.0 + 0.5 * u * u;
         },
         compressed);
      hycfd::ComputeDilatationAV(op, compressed, model, options,
                                 av_field);
      const double compressed_max = MaxField(av_field);
      Require(compressed_max > 1.0e-2 * options.lambda,
              "sensor AV did not activate under strong compression");
      std::cout << "PASS sensor active under compression: max_av="
                << compressed_max << '\n';

      // (c) Field tabulation: the operator's frozen AV tables must
      // reproduce the field's maximum (vertex interpolation bounds it).
      op.SetArtificialViscosityField(av_field);
      const double tabulated_max = op.MaximumAbsAV();
      Require(tabulated_max > 0.5 * compressed_max &&
              tabulated_max <= compressed_max * (1.0 + 1.0e-12),
              "tabulated AV is inconsistent with the sensor field");
      std::cout << "PASS AV field tabulation: tabulated_max="
                << tabulated_max << " field_max=" << compressed_max
                << '\n';

      std::cout << "ALL test_av G6 GATES PASSED\n";
   }
   catch (const std::exception &error)
   {
      std::cerr << "FAIL test_av: " << error.what() << '\n';
      exit_code = EXIT_FAILURE;
   }
   mfem::MFEMFinalizePetsc();
   return exit_code;
}

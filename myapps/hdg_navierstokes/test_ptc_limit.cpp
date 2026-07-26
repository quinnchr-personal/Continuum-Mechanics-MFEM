#include "exasim_mesh.hpp"
#include "hdg_newton.hpp"
#include "hdg_ns_operator.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace
{

void NewtonStep(hdg_ns::HDGNavierStokesOperator &op,
                const hdg_ns::HDGState &base,
                double inverse_pseudo_time_step,
                hdg_ns::HDGState &updated)
{
   op.Assemble(base, true, inverse_pseudo_time_step);
   mfem::Vector trace_increment;
   hdg_ns::SolveCondensedPetscDirect(
      op.CondensedMatrix(), op.CondensedRHS(), trace_increment);
   mfem::Vector volume_increment;
   op.RecoverIncrement(trace_increment, volume_increment);
   updated = base;
   updated.u += volume_increment;
   updated.uhat += trace_increment;
   op.RecomputeGradient(updated);
}

double RelativeDifference(const mfem::Vector &first,
                          const mfem::Vector &second)
{
   mfem::Vector difference(first);
   difference -= second;
   return difference.Norml2() /
          std::max({1.0, first.Norml2(), second.Norml2()});
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
         throw std::runtime_error("PTC limit test requires np=1");
      }
      std::unique_ptr<mfem::Mesh> mesh =
         hdg_ns::BuildAnalyticMesh(3, 2, hdg_ns::kOrder);
      const auto av = [](const mfem::Vector &x)
      {
         return 0.06 *
                std::tanh(30.0 * (std::hypot(x[0], x[1]) - 1.0));
      };
      hdg_ns::NSParams params;
      hdg_ns::HDGNavierStokesOperator op(*mesh, av, params);
      hdg_ns::HDGState base;
      op.ProjectState(
         [params](const mfem::Vector &x, double value[4])
         {
            const double distance = std::hypot(x[0], x[1]) - 1.0;
            const double velocity = std::tanh(10.0 * distance);
            value[0] = 1.0;
            value[1] = velocity;
            value[2] = 0.0;
            value[3] =
               params.TinfFlux() *
                  ((params.mu[10] / params.mu[9] - 1.0) *
                      std::exp(-10.0 * distance) + 1.0) +
               0.5 * velocity * velocity;
         },
         base);
      op.InitializeTraceFromInterior(base);

      hdg_ns::HDGState newton, ptc;
      NewtonStep(op, base, 0.0, newton);
      constexpr double huge_dtau = 1.0e30;
      NewtonStep(op, base, 1.0 / huge_dtau, ptc);

      const double u_error = RelativeDifference(newton.u, ptc.u);
      const double q_error = RelativeDifference(newton.q, ptc.q);
      const double trace_error =
         RelativeDifference(newton.uhat, ptc.uhat);
      const double maximum_error =
         std::max({u_error, q_error, trace_error});
      std::cout << std::setprecision(17)
                << "PTC huge-dtau limit: dtau=" << huge_dtau
                << " u_relative=" << u_error
                << " q_relative=" << q_error
                << " uhat_relative=" << trace_error << '\n';
      if (maximum_error > 5.0e-15)
      {
         throw std::runtime_error(
            "huge-dtau PTC step does not reproduce Newton to roundoff");
      }
      std::cout << "PASS M4 PTC huge-dtau limit reproduces pure Newton"
                << " to roundoff\n";
   }
   catch (const std::exception &error)
   {
      std::cerr << "FAIL test_ptc_limit: " << error.what() << '\n';
      exit_code = EXIT_FAILURE;
   }
   mfem::MFEMFinalizePetsc();
   return exit_code;
}

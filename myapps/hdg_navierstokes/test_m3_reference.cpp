#include "exasim_reference.hpp"

#include "mfem.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char *argv[])
{
   mfem::Mpi::Init(argc, argv);
   mfem::Hypre::Init();
   const std::string reference_directory =
      argc > 1 ? argv[1] : EXASIM_RUN_DIR;
   try
   {
      if (mfem::Mpi::WorldSize() != 1)
      {
         throw std::runtime_error("M3 reference test requires np=1");
      }
      hdg_ns::ExasimMesh converted =
         hdg_ns::BuildExasimMesh(reference_directory);
      const hdg_ns::ExasimArray vdg =
         hdg_ns::ReadExasimArray(reference_directory + "/vdg.bin");
      const hdg_ns::ExasimReferenceData reference =
         hdg_ns::ReadExasimReferenceData(
            reference_directory, converted);
      std::cout << std::setprecision(17)
                << "reference map: xdg_byte_match=yes"
                << " max_xdg_diff=" << reference.maximum_xdg_difference
                << " local_faces=" << reference.local_face_count
                << " duplicate_faces=" << reference.duplicate_face_count
                << " max_face_geometry_diff="
                << reference.maximum_face_geometry_difference
                << " max_shared_trace_diff="
                << reference.maximum_shared_trace_difference << '\n';

      hdg_ns::HDGNavierStokesOperator op(
         *converted.mesh, vdg, converted.orientations);
      hdg_ns::HDGState state;
      hdg_ns::LoadExasimReferenceState(reference, op, true, state);
      hdg_ns::HDGState recomputed(state);
      op.RecomputeGradient(recomputed);
      mfem::Vector q_difference(recomputed.q);
      q_difference -= state.q;
      const double q_relative_difference =
         q_difference.Norml2() /
         std::max(1.0, state.q.Norml2());
      std::cout << "converged pair q-fold check:"
                << " stored_norm=" << state.q.Norml2()
                << " recomputed_norm=" << recomputed.q.Norml2()
                << " relative_difference=" << q_relative_difference
                << '\n';
      const hdg_ns::HDGResidualNorms residual =
         op.Assemble(state, false);
      std::cout << "M3(a) converged-reference residual:"
                << " ||Ru||=" << residual.volume
                << " ||Rh||=" << residual.trace
                << " total=" << residual.Total() << '\n';
      if (!(residual.Total() <= 1.0e-5))
      {
         throw std::runtime_error(
            "M3(a) residual exceeds 1e-5");
      }
      std::cout << "PASS M3(a) residual cross-check\n";
      return EXIT_SUCCESS;
   }
   catch (const std::exception &error)
   {
      std::cerr << "FAIL test_m3_reference: "
                << error.what() << '\n';
      return EXIT_FAILURE;
   }
}

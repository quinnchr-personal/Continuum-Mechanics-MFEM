#include "exasim_io.hpp"
#include "exasim_mesh.hpp"
#include "hdg_ns_operator.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

void Require(bool condition, const std::string &message)
{
   if (!condition) { throw std::runtime_error(message); }
}

double RoundTripError(const hdg_ns::ExasimArray &array,
                      const hdg_ns::TensorBasisTransform &transform)
{
   double maximum_error = 0.0;
   double source[hdg_ns::kNodes2D];
   double target[hdg_ns::kNodes2D];
   double recovered[hdg_ns::kNodes2D];
   for (int element = 0; element < array.nelem; ++element)
   {
      for (int component = 0; component < array.ncomp; ++component)
      {
         for (int node = 0; node < array.nnode; ++node)
         {
            source[node] = array(node, component, element);
         }
         transform.ToTarget(source, target);
         transform.ToExasim(target, recovered);
         for (int node = 0; node < array.nnode; ++node)
         {
            maximum_error =
               std::max(
                  maximum_error,
                  std::abs(recovered[node] - source[node]) /
                  std::max(1.0, std::abs(source[node])));
         }
      }
   }
   return maximum_error;
}

double CheckBoundaryAttribution(mfem::Mesh &mesh)
{
   double maximum_mismatch = 0.0;
   for (int boundary = 0; boundary < mesh.GetNBE(); ++boundary)
   {
      mfem::ElementTransformation *transformation =
         mesh.GetBdrElementTransformation(boundary);
      mfem::IntegrationPoint midpoint;
      midpoint.x = 0.5;
      mfem::Vector physical(2);
      transformation->Transform(midpoint, physical);
      int expected = 3;
      if (std::hypot(physical[0], physical[1]) < 1.0 + 1.0e-6)
      {
         expected = 1;
      }
      else if (physical[0] > -1.0e-7)
      {
         expected = 2;
      }
      maximum_mismatch =
         std::max(maximum_mismatch,
                  std::abs(static_cast<double>(
                     mesh.GetBdrAttribute(boundary) - expected)));
   }
   return maximum_mismatch;
}

double CheckQuadrature(const hdg_ns::HDGNavierStokesOperator &op)
{
   constexpr std::array<double, 5> expected =
   {{
      0.0469100770306680036011865608503035,
      0.2307653449471584544818427896498956,
      0.5,
      0.7692346550528415455181572103501044,
      0.9530899229693319963988134391496965
   }};
   const mfem::IntegrationRule &face_rule = op.FaceRule();
   double maximum_error = 0.0;
   for (int q = 0; q < 5; ++q)
   {
      maximum_error =
         std::max(maximum_error,
                  std::abs(face_rule.IntPoint(q).x - expected[q]));
   }

   const mfem::IntegrationRule &volume_rule = op.VolumeRule();
   Require(volume_rule.GetNPoints() == 25,
           "volume quadrature is not 5x5");
   for (int q = 0; q < 25; ++q)
   {
      const double x = volume_rule.IntPoint(q).x;
      const double y = volume_rule.IntPoint(q).y;
      double x_error = std::numeric_limits<double>::infinity();
      double y_error = std::numeric_limits<double>::infinity();
      for (double point : expected)
      {
         x_error = std::min(x_error, std::abs(x - point));
         y_error = std::min(y_error, std::abs(y - point));
      }
      maximum_error = std::max({maximum_error, x_error, y_error});
   }
   return maximum_error;
}

double CheckGradientElimination(
   const hdg_ns::HDGNavierStokesOperator &op, int element_count)
{
   mfem::Vector volume_constant(25);
   mfem::Vector trace_constant(20);
   mfem::Vector c_value(25);
   mfem::Vector e_value(25);
   volume_constant = 1.0;
   trace_constant = 1.0;
   double maximum_error = 0.0;
   for (int element = 0; element < element_count; ++element)
   {
      for (int direction = 0; direction < 2; ++direction)
      {
         const mfem::DenseMatrix &c = op.C(element, direction);
         const mfem::DenseMatrix &e = op.E(element, direction);
         Require(c.Height() == 25 && c.Width() == 25,
                 "C block does not have shape 25x25");
         Require(e.Height() == 25 && e.Width() == 20,
                 "E block does not have shape 25x20");
         c.Mult(volume_constant, c_value);
         e.Mult(trace_constant, e_value);
         for (int row = 0; row < 25; ++row)
         {
            Require(std::isfinite(c_value[row]) &&
                    std::isfinite(e_value[row]),
                    "C/E block contains a non-finite value");
            maximum_error =
               std::max(maximum_error,
                        std::abs(c_value[row] - e_value[row]));
         }
      }
   }
   return maximum_error;
}

} // namespace

int main(int argc, char *argv[])
{
   try
   {
      const std::string reference_directory =
         argc > 1 ? argv[1] : EXASIM_RUN_DIR;
      std::cout << std::setprecision(17);

      hdg_ns::ExasimMesh converted =
         hdg_ns::BuildExasimMesh(reference_directory);
      const auto boundary_counts =
         hdg_ns::CountBoundaryAttributes(*converted.mesh);
      const double boundary_mismatch =
         CheckBoundaryAttribution(*converted.mesh);
      Require(converted.mesh->GetNE() == 651,
              "converted mesh does not have 651 elements");
      Require(converted.mesh->GetNV() == 704,
              "converted mesh does not have 704 vertices");
      Require(converted.mesh->GetNumFaces() == 1354,
              "converted mesh does not have 1354 edges/faces");
      Require(boundary_counts == std::array<int, 3>{{21, 62, 21}},
              "boundary counts are not wall/outflow/freestream = 21/62/21");
      Require(boundary_mismatch == 0.0,
              "one or more boundary attributes fail the specified predicate");
      std::cout << "PASS topology/boundaries:"
                << " elements=" << converted.mesh->GetNE()
                << " vertices=" << converted.mesh->GetNV()
                << " faces=" << converted.mesh->GetNumFaces()
                << " attributes=" << boundary_counts[0] << '/'
                << boundary_counts[1] << '/' << boundary_counts[2]
                << " predicate_mismatch=" << boundary_mismatch << '\n';

      const double geometry_error =
         hdg_ns::GeometryReproductionError(converted);
      Require(geometry_error <= 1.0e-13,
              "MFEM Q4 geometry does not reproduce xdg.bin to 1e-13");
      std::cout << "PASS exact xdg geometry:"
                << " relative_max_error=" << geometry_error
                << " orientation="
                << converted.orientations[0].mfem_corner_to_exasim[0]
                << converted.orientations[0].mfem_corner_to_exasim[1]
                << converted.orientations[0].mfem_corner_to_exasim[2]
                << converted.orientations[0].mfem_corner_to_exasim[3]
                << '\n';

      const hdg_ns::ExasimArray udg =
         hdg_ns::ReadExasimArray(reference_directory + "/udg.bin");
      const hdg_ns::ExasimArray vdg =
         hdg_ns::ReadExasimArray(reference_directory + "/vdg.bin");
      Require(udg.nnode == 25 && udg.ncomp == 12 && udg.nelem == 651,
              "udg.bin does not have layout [25,12,651]");
      Require(vdg.nnode == 25 && vdg.ncomp == 2 && vdg.nelem == 651,
              "vdg.bin does not have layout [25,2,651]");
      double vdg_component_one_max = 0.0;
      for (int element = 0; element < vdg.nelem; ++element)
      {
         for (int node = 0; node < vdg.nnode; ++node)
         {
            vdg_component_one_max =
               std::max(vdg_component_one_max,
                        std::abs(vdg(node, 1, element)));
         }
      }
      Require(vdg_component_one_max < 1.0e-14,
              "vdg component 1 is not zero");

      mfem::L2_FECollection l2_collection(
         hdg_ns::kOrder, 2, mfem::BasisType::GaussLegendre);
      mfem::FiniteElementSpace l2_space(
         converted.mesh.get(), &l2_collection);
      const auto target_points =
         hdg_ns::FiniteElementNodes(*l2_space.GetFE(0));
      const hdg_ns::TensorBasisTransform basis_transform =
         hdg_ns::BuildTensorBasisTransform(
            target_points, converted.orientations[0]);
      const double udg_round_trip =
         RoundTripError(udg, basis_transform);
      const double vdg_round_trip =
         RoundTripError(vdg, basis_transform);
      Require(udg_round_trip <= 1.0e-13,
              "udg change-of-basis round trip exceeds 1e-13");
      Require(vdg_round_trip <= 1.0e-13,
              "vdg change-of-basis round trip exceeds 1e-13");
      std::cout << "PASS udg/vdg change-of-basis round trip:"
                << " udg=" << udg_round_trip
                << " vdg=" << vdg_round_trip
                << " vdg_comp1_max=" << vdg_component_one_max << '\n';

      hdg_ns::HDGNavierStokesOperator op(
         *converted.mesh, vdg, converted.orientations);
      Require(op.VolumeQuadraturePoints() == 25 &&
              op.FaceQuadraturePoints() == 5,
              "operator constructor quadrature sizes are wrong");
      Require(op.VolumeDofsPerElement() == 25 &&
              op.TraceDofsPerFace() == 5,
              "operator constructor Q4 dof sizes are wrong");
      Require(op.TraceVSize() == 27080,
              "trace space size is not 27,080");
      Require(op.MinimumDetJ() > 0.0,
              "operator constructor found non-positive detJ");
      Require(op.MaximumAbsAV() <= 0.025000000000001 &&
              op.MaximumAbsAV() >= 0.024999,
              "operator constructor AV table has unexpected range");

      const double quadrature_error = CheckQuadrature(op);
      Require(quadrature_error <= 5.0e-15,
              "MFEM degree-8 rule does not match decoded Exasim 5-point GL");
      const double constant_gradient_error =
         CheckGradientElimination(op, converted.mesh->GetNE());
      Require(constant_gradient_error <= 2.0e-10,
              "q=C*u-E*uhat does not annihilate a constant field");
      std::cout << "PASS operator constructor:"
                << " trace_vsize=" << op.TraceVSize()
                << " min_detJ=" << op.MinimumDetJ()
                << " max_abs_av=" << op.MaximumAbsAV()
                << " C/E_constant_error=" << constant_gradient_error
                << '\n';
      std::cout << "PASS decoded 5-point Gauss-Legendre quadrature:"
                << " max_point_error=" << quadrature_error << '\n';

      std::unique_ptr<mfem::Mesh> analytic =
         hdg_ns::BuildAnalyticMesh(3, 4, 4);
      const auto analytic_counts =
         hdg_ns::CountBoundaryAttributes(*analytic);
      Require(analytic_counts == std::array<int, 3>{{4, 6, 4}},
              "analytic mesh boundary attribution failed");
      std::cout << "PASS analytic mesh generator:"
                << " elements=" << analytic->GetNE()
                << " attributes=" << analytic_counts[0] << '/'
                << analytic_counts[1] << '/' << analytic_counts[2] << '\n';

      std::cout << "ALL mesh_convert_check M1 GATES PASSED\n";
      return EXIT_SUCCESS;
   }
   catch (const std::exception &error)
   {
      std::cerr << "FAIL mesh_convert_check: " << error.what() << '\n';
      return EXIT_FAILURE;
   }
}

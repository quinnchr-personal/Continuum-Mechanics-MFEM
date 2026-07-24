#include "hdg_ns_operator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace hdg_ns
{
namespace
{

std::size_t VolumeIndex(int element, int quadrature_point,
                        int points_per_element)
{
   return static_cast<std::size_t>(quadrature_point) +
          static_cast<std::size_t>(points_per_element) *
          static_cast<std::size_t>(element);
}

std::size_t FaceIndex(int element, int local_face, int quadrature_point,
                      int points_per_face)
{
   return static_cast<std::size_t>(quadrature_point) +
          static_cast<std::size_t>(points_per_face) *
          (static_cast<std::size_t>(local_face) +
           4 * static_cast<std::size_t>(element));
}

} // namespace

HDGNavierStokesOperator::HDGNavierStokesOperator(
   mfem::Mesh &mesh, const ExasimArray &vdg,
   const std::vector<ElementOrientation> &orientations)
   : mesh_(mesh),
     vdg_(vdg),
     orientations_(orientations),
     volume_fec_(kOrder, 2),
     volume_fes_(&mesh_, &volume_fec_, 4, mfem::Ordering::byVDIM),
     trace_fec_(kOrder, 2),
     trace_fes_(&mesh_, &trace_fec_, 4, mfem::Ordering::byVDIM)
{
   ValidateInputs();
   volume_rule_ = &mfem::IntRules.Get(mfem::Geometry::SQUARE, 8);
   face_rule_ = &mfem::IntRules.Get(mfem::Geometry::SEGMENT, 8);
   volume_fe_ = volume_fes_.GetFE(0);
   trace_fe_ =
      trace_fec_.FiniteElementForGeometry(mfem::Geometry::SEGMENT);
   if (!volume_fe_ || !trace_fe_)
   {
      throw std::runtime_error("MFEM did not provide Q4 volume/trace elements");
   }
   if (volume_rule_->GetNPoints() != 25 ||
       face_rule_->GetNPoints() != 5 ||
       volume_fe_->GetDof() != 25 ||
       trace_fe_->GetDof() != 5)
   {
      throw std::runtime_error("MFEM Q4 quadrature/finite-element size mismatch");
   }
   if (trace_fes_.GetVSize() != 4 * 5 * mesh_.GetNumFaces())
   {
      throw std::runtime_error(
         "DG_Interface_FECollection trace size does not equal 4*5*nfaces");
   }
   mfem::Array<int> face_vdofs;
   for (int face = 0; face < mesh_.GetNumFaces(); ++face)
   {
      trace_fes_.GetFaceVDofs(face, face_vdofs);
      if (face_vdofs.Size() != 20)
      {
         throw std::runtime_error(
            "DG_Interface_FECollection face does not have 20 vector dofs");
      }
   }

   BuildReferenceTables();
   BuildGeometryAndAVTables();
   BuildGradientElimination();
}

int HDGNavierStokesOperator::VolumeQuadraturePoints() const
{
   return volume_rule_->GetNPoints();
}

int HDGNavierStokesOperator::FaceQuadraturePoints() const
{
   return face_rule_->GetNPoints();
}

int HDGNavierStokesOperator::TraceVSize() const
{
   return trace_fes_.GetVSize();
}

int HDGNavierStokesOperator::VolumeDofsPerElement() const
{
   return volume_fe_->GetDof();
}

int HDGNavierStokesOperator::TraceDofsPerFace() const
{
   return trace_fe_->GetDof();
}

const mfem::IntegrationRule &
HDGNavierStokesOperator::VolumeRule() const
{
   return *volume_rule_;
}

const mfem::IntegrationRule &
HDGNavierStokesOperator::FaceRule() const
{
   return *face_rule_;
}

const mfem::DenseMatrix &
HDGNavierStokesOperator::C(int element, int direction) const
{
   if (direction < 0 || direction > 1)
   {
      throw std::out_of_range("C direction must be 0 or 1");
   }
   return c_[direction].at(static_cast<std::size_t>(element));
}

const mfem::DenseMatrix &
HDGNavierStokesOperator::E(int element, int direction) const
{
   if (direction < 0 || direction > 1)
   {
      throw std::out_of_range("E direction must be 0 or 1");
   }
   return e_[direction].at(static_cast<std::size_t>(element));
}

double HDGNavierStokesOperator::MinimumDetJ() const
{
   return *std::min_element(volume_det_j_.begin(), volume_det_j_.end());
}

double HDGNavierStokesOperator::MaximumAbsAV() const
{
   double maximum = 0.0;
   for (double value : volume_av_)
   {
      maximum = std::max(maximum, std::abs(value));
   }
   for (double value : face_av_)
   {
      maximum = std::max(maximum, std::abs(value));
   }
   return maximum;
}

void HDGNavierStokesOperator::ValidateInputs() const
{
   if (mesh_.Dimension() != 2 || mesh_.SpaceDimension() != 2 ||
       mesh_.GetNE() <= 0)
   {
      throw std::runtime_error("HDG M1 constructor requires a nonempty 2D mesh");
   }
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      if (mesh_.GetElementGeometry(element) != mfem::Geometry::SQUARE)
      {
         throw std::runtime_error(
            "HDG M1 constructor requires an all-quadrilateral mesh");
      }
   }
   if (vdg_.nnode != 25 || vdg_.ncomp != 2 ||
       vdg_.nelem != mesh_.GetNE())
   {
      throw std::runtime_error("vdg.bin does not have layout [25,2,ne]");
   }
   if (orientations_.size() != static_cast<std::size_t>(mesh_.GetNE()))
   {
      throw std::runtime_error("missing element orientation maps");
   }
   double second_component_maximum = 0.0;
   for (int element = 0; element < vdg_.nelem; ++element)
   {
      for (int node = 0; node < vdg_.nnode; ++node)
      {
         second_component_maximum =
            std::max(second_component_maximum,
                     std::abs(vdg_(node, 1, element)));
      }
   }
   if (second_component_maximum >= 1.0e-14)
   {
      std::ostringstream message;
      message << "vdg component 1 is not zero: max="
              << second_component_maximum;
      throw std::runtime_error(message.str());
   }
}

void HDGNavierStokesOperator::BuildReferenceTables()
{
   const int nq = volume_rule_->GetNPoints();
   const int ndof = volume_fe_->GetDof();
   const int nqf = face_rule_->GetNPoints();
   const int nfdof = trace_fe_->GetDof();
   volume_shape_.resize(static_cast<std::size_t>(nq * ndof));
   volume_reference_dshape_.resize(
      static_cast<std::size_t>(nq * ndof * 2));
   trace_shape_.resize(static_cast<std::size_t>(nqf * nfdof));

   mfem::Vector shape(ndof);
   mfem::DenseMatrix dshape(ndof, 2);
   for (int q = 0; q < nq; ++q)
   {
      const mfem::IntegrationPoint &point = volume_rule_->IntPoint(q);
      volume_fe_->CalcShape(point, shape);
      volume_fe_->CalcDShape(point, dshape);
      for (int dof = 0; dof < ndof; ++dof)
      {
         volume_shape_[static_cast<std::size_t>(dof + ndof * q)] =
            shape[dof];
         for (int direction = 0; direction < 2; ++direction)
         {
            volume_reference_dshape_[
               static_cast<std::size_t>(
                  direction + 2 * (dof + ndof * q))] =
               dshape(dof, direction);
         }
      }
   }

   mfem::Vector face_shape(nfdof);
   for (int q = 0; q < nqf; ++q)
   {
      trace_fe_->CalcShape(face_rule_->IntPoint(q), face_shape);
      for (int dof = 0; dof < nfdof; ++dof)
      {
         trace_shape_[static_cast<std::size_t>(dof + nfdof * q)] =
            face_shape[dof];
      }
   }
}

void HDGNavierStokesOperator::BuildGeometryAndAVTables()
{
   const int ne = mesh_.GetNE();
   const int nq = volume_rule_->GetNPoints();
   const int nqf = face_rule_->GetNPoints();
   const int ndof = volume_fe_->GetDof();
   volume_det_j_.resize(static_cast<std::size_t>(ne * nq));
   volume_adj_j_.resize(static_cast<std::size_t>(ne * nq * 4));
   volume_av_.resize(static_cast<std::size_t>(ne * nq));
   face_weighted_normals_.resize(
      static_cast<std::size_t>(ne * 4 * nqf * 2));
   face_element_shape_.resize(
      static_cast<std::size_t>(ne * 4 * nqf * ndof));
   face_av_.resize(static_cast<std::size_t>(ne * 4 * nqf));

   mfem::Vector element_shape(ndof);
   mfem::Vector weighted_normal(2);
   mfem::Array<int> faces, orientations;
   for (int element = 0; element < ne; ++element)
   {
      mfem::ElementTransformation *transformation =
         mesh_.GetElementTransformation(element);
      for (int q = 0; q < nq; ++q)
      {
         const mfem::IntegrationPoint &point = volume_rule_->IntPoint(q);
         transformation->SetIntPoint(&point);
         const std::size_t index = VolumeIndex(element, q, nq);
         volume_det_j_[index] = transformation->Weight();
         if (!(volume_det_j_[index] > 0.0) ||
             !std::isfinite(volume_det_j_[index]))
         {
            throw std::runtime_error(
               "non-positive/non-finite Q4 geometry Jacobian");
         }
         const mfem::DenseMatrix &adjugate =
            transformation->AdjugateJacobian();
         for (int column = 0; column < 2; ++column)
         {
            for (int row = 0; row < 2; ++row)
            {
               volume_adj_j_[
                  4 * index + static_cast<std::size_t>(row + 2 * column)] =
                  adjugate(row, column);
            }
         }
         const auto exasim_point =
            orientations_[element].MfemToExasim(point.x, point.y);
         volume_av_[index] = EvaluateTensorQ4(
            vdg_, 0, element, exasim_point[0], exasim_point[1]);
      }

      mesh_.GetElementEdges(element, faces, orientations);
      if (faces.Size() != 4)
      {
         throw std::runtime_error("quadrilateral did not report four faces");
      }
      for (int local_face = 0; local_face < 4; ++local_face)
      {
         mfem::FaceElementTransformations *face_transformations =
            mesh_.GetFaceElementTransformations(faces[local_face], 31);
         if (!face_transformations)
         {
            throw std::runtime_error(
               "MFEM did not return full face-element transformations");
         }
         const bool side_one =
            face_transformations->Elem1No == element;
         const bool side_two =
            face_transformations->Elem2No == element;
         if (side_one == side_two)
         {
            throw std::runtime_error(
               "element is not exactly one side of its face");
         }

         for (int q = 0; q < nqf; ++q)
         {
            const mfem::IntegrationPoint &face_point =
               face_rule_->IntPoint(q);
            face_transformations->SetAllIntPoints(&face_point);
            const mfem::IntegrationPoint &element_point =
               side_one ?
               face_transformations->GetElement1IntPoint() :
               face_transformations->GetElement2IntPoint();
            volume_fe_->CalcShape(element_point, element_shape);
            mfem::CalcOrtho(
               face_transformations->Jacobian(), weighted_normal);
            if (side_two) { weighted_normal *= -1.0; }

            const std::size_t face_index =
               FaceIndex(element, local_face, q, nqf);
            for (int direction = 0; direction < 2; ++direction)
            {
               face_weighted_normals_[
                  2 * face_index + static_cast<std::size_t>(direction)] =
                  weighted_normal[direction];
            }
            for (int dof = 0; dof < ndof; ++dof)
            {
               face_element_shape_[
                  static_cast<std::size_t>(dof) +
                  static_cast<std::size_t>(ndof) * face_index] =
                  element_shape[dof];
            }
            const auto exasim_point =
               orientations_[element].MfemToExasim(
                  element_point.x, element_point.y);
            face_av_[face_index] = EvaluateTensorQ4(
               vdg_, 0, element, exasim_point[0], exasim_point[1]);
         }
      }
   }
}

void HDGNavierStokesOperator::BuildGradientElimination()
{
   const int ne = mesh_.GetNE();
   const int nq = volume_rule_->GetNPoints();
   const int nqf = face_rule_->GetNPoints();
   constexpr int ndof = 25;
   constexpr int nfdof = 5;
   constexpr int element_trace_dofs = 4 * nfdof;
   for (int direction = 0; direction < 2; ++direction)
   {
      c_[direction].reserve(static_cast<std::size_t>(ne));
      e_[direction].reserve(static_cast<std::size_t>(ne));
   }

   mfem::Vector shape(ndof);
   mfem::Vector face_shape(nfdof);
   mfem::DenseMatrix physical_dshape(ndof, 2);
   mfem::Array<int> faces, face_orientations;

   for (int element = 0; element < ne; ++element)
   {
      mfem::DenseMatrix mass(ndof);
      mfem::DenseMatrix derivative[2] =
      {
         mfem::DenseMatrix(ndof),
         mfem::DenseMatrix(ndof)
      };
      mfem::DenseMatrix boundary[2] =
      {
         mfem::DenseMatrix(ndof, element_trace_dofs),
         mfem::DenseMatrix(ndof, element_trace_dofs)
      };
      mass = 0.0;
      derivative[0] = 0.0;
      derivative[1] = 0.0;
      boundary[0] = 0.0;
      boundary[1] = 0.0;

      mfem::ElementTransformation *transformation =
         mesh_.GetElementTransformation(element);
      for (int q = 0; q < nq; ++q)
      {
         const mfem::IntegrationPoint &point = volume_rule_->IntPoint(q);
         transformation->SetIntPoint(&point);
         volume_fe_->CalcShape(point, shape);
         volume_fe_->CalcPhysDShape(*transformation, physical_dshape);
         const double weight = point.weight * transformation->Weight();
         for (int test = 0; test < ndof; ++test)
         {
            for (int trial = 0; trial < ndof; ++trial)
            {
               mass(test, trial) +=
                  weight * shape[test] * shape[trial];
               for (int direction = 0; direction < 2; ++direction)
               {
                  derivative[direction](test, trial) +=
                     weight * physical_dshape(test, direction) *
                     shape[trial];
               }
            }
         }
      }

      mesh_.GetElementEdges(element, faces, face_orientations);
      for (int local_face = 0; local_face < 4; ++local_face)
      {
         mfem::FaceElementTransformations *face_transformations =
            mesh_.GetFaceElementTransformations(faces[local_face], 31);
         const bool side_one =
            face_transformations->Elem1No == element;
         for (int q = 0; q < nqf; ++q)
         {
            const mfem::IntegrationPoint &face_point =
               face_rule_->IntPoint(q);
            face_transformations->SetAllIntPoints(&face_point);
            const mfem::IntegrationPoint &element_point =
               side_one ?
               face_transformations->GetElement1IntPoint() :
               face_transformations->GetElement2IntPoint();
            volume_fe_->CalcShape(element_point, shape);
            trace_fe_->CalcShape(face_point, face_shape);
            const std::size_t face_index =
               FaceIndex(element, local_face, q, nqf);
            for (int test = 0; test < ndof; ++test)
            {
               for (int trace_dof = 0; trace_dof < nfdof; ++trace_dof)
               {
                  const int column = local_face * nfdof + trace_dof;
                  for (int direction = 0; direction < 2; ++direction)
                  {
                     boundary[direction](test, column) +=
                        face_point.weight * shape[test] *
                        face_shape[trace_dof] *
                        face_weighted_normals_[
                           2 * face_index +
                           static_cast<std::size_t>(direction)];
                  }
               }
            }
         }
      }

      mfem::DenseMatrixInverse mass_inverse(mass);
      for (int direction = 0; direction < 2; ++direction)
      {
         c_[direction].emplace_back(ndof);
         e_[direction].emplace_back(ndof, element_trace_dofs);
         mass_inverse.Mult(derivative[direction], c_[direction].back());
         mass_inverse.Mult(boundary[direction], e_[direction].back());
      }
   }
}

} // namespace hdg_ns

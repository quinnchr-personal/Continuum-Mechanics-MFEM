#include "hdg_ns_operator.hpp"
#include "exasim_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

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

int UIndex(int element, int component, int dof)
{
   return dof + 25 * (component + 4 * element);
}

int QIndex(int element, int direction, int component, int dof)
{
   return dof + 25 * (component + 4 * (direction + 2 * element));
}

int LocalUIndex(int component, int dof)
{
   return dof + 25 * component;
}

int LocalQIndex(int direction, int component, int dof)
{
   return dof + 25 * (component + 4 * direction);
}

int LocalTraceIndex(int local_face, int component, int dof)
{
   return component + 4 * (dof + 5 * local_face);
}

// MFEM's byVDIM global numbering is component-fast, but DofsToVDofs expands
// returned dof lists as [all scalar dofs for component 0, then component 1,
// ...]. Keep that API-list layout distinct from LocalTraceIndex.
int FaceVDofListIndex(int component, int dof)
{
   return dof + 5 * component;
}

double VectorNorm(const mfem::Vector &vector)
{
   return vector.Norml2();
}

} // namespace

HDGNavierStokesOperator::HDGNavierStokesOperator(
   mfem::Mesh &mesh, const ExasimArray &vdg,
   const std::vector<ElementOrientation> &orientations,
   const NSParams &params,
   const std::array<int, 3> &boundary_conditions)
   : mesh_(mesh),
     vdg_(&vdg),
     orientations_(orientations),
     params_(params),
     boundary_conditions_(boundary_conditions),
     volume_fec_(kOrder, 2),
     volume_fes_(&mesh_, &volume_fec_, 4, mfem::Ordering::byVDIM),
     trace_fec_(kOrder, 2),
     trace_fes_(&mesh_, &trace_fec_, 4, mfem::Ordering::byVDIM)
{
   Initialize();
}

HDGNavierStokesOperator::HDGNavierStokesOperator(
   mfem::Mesh &mesh,
   const std::vector<ElementOrientation> &orientations,
   ScalarFunction artificial_viscosity,
   const NSParams &params,
   const std::array<int, 3> &boundary_conditions)
   : mesh_(mesh),
     orientations_(orientations),
     artificial_viscosity_(std::move(artificial_viscosity)),
     params_(params),
     boundary_conditions_(boundary_conditions),
     volume_fec_(kOrder, 2),
     volume_fes_(&mesh_, &volume_fec_, 4, mfem::Ordering::byVDIM),
     trace_fec_(kOrder, 2),
     trace_fes_(&mesh_, &trace_fec_, 4, mfem::Ordering::byVDIM)
{
   Initialize();
}

HDGNavierStokesOperator::HDGNavierStokesOperator(
   mfem::Mesh &mesh, ScalarFunction artificial_viscosity,
   const NSParams &params,
   const std::array<int, 3> &boundary_conditions)
   : mesh_(mesh),
     artificial_viscosity_(std::move(artificial_viscosity)),
     params_(params),
     boundary_conditions_(boundary_conditions),
     volume_fec_(kOrder, 2),
     volume_fes_(&mesh_, &volume_fec_, 4, mfem::Ordering::byVDIM),
     trace_fec_(kOrder, 2),
     trace_fes_(&mesh_, &trace_fec_, 4, mfem::Ordering::byVDIM)
{
   orientations_.resize(static_cast<std::size_t>(mesh_.GetNE()));
   for (ElementOrientation &orientation : orientations_)
   {
      orientation.mfem_corner_to_exasim = {{0, 1, 2, 3}};
   }
   Initialize();
}

void HDGNavierStokesOperator::Initialize()
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
      for (int i = 0; i < face_vdofs.Size(); ++i)
      {
         if (face_vdofs[i] < 0)
         {
            throw std::runtime_error(
               "trace face vdof unexpectedly has an orientation sign");
         }
      }
   }

   BuildReferenceTables();
   BuildGeometryAndAVTables();
   BuildGradientElimination();
   BuildBoundaryFaceMap();

   volume_residual_.SetSize(mesh_.GetNE() * kElementUnknowns);
   trace_residual_.SetSize(trace_fes_.GetVSize());
   condensed_residual_.SetSize(trace_fes_.GetVSize());
   condensed_rhs_.SetSize(trace_fes_.GetVSize());
   inverse_a_f_.reserve(static_cast<std::size_t>(mesh_.GetNE()));
   inverse_a_ru_.reserve(static_cast<std::size_t>(mesh_.GetNE()));
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

int HDGNavierStokesOperator::Elements() const
{
   return mesh_.GetNE();
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

const mfem::FiniteElementSpace &
HDGNavierStokesOperator::VolumeSpace() const
{
   return volume_fes_;
}

const mfem::FiniteElementSpace &
HDGNavierStokesOperator::TraceSpace() const
{
   return trace_fes_;
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

void HDGNavierStokesOperator::SetArtificialViscosity(
   ScalarFunction artificial_viscosity)
{
   if (!artificial_viscosity)
   {
      throw std::runtime_error(
         "cannot retabulate an empty artificial-viscosity function");
   }
   vdg_ = nullptr;
   artificial_viscosity_ = std::move(artificial_viscosity);
   RetabulateArtificialViscosity();
}

void HDGNavierStokesOperator::ValidateInputs() const
{
   if (mesh_.Dimension() != 2 || mesh_.SpaceDimension() != 2 ||
       mesh_.GetNE() <= 0)
   {
      throw std::runtime_error("HDG operator requires a nonempty 2D mesh");
   }
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      if (mesh_.GetElementGeometry(element) != mfem::Geometry::SQUARE)
      {
         throw std::runtime_error(
            "HDG operator requires an all-quadrilateral mesh");
      }
   }
   if (orientations_.size() != static_cast<std::size_t>(mesh_.GetNE()))
   {
      throw std::runtime_error("missing element orientation maps");
   }
   if (vdg_)
   {
      if (vdg_->nnode != 25 || vdg_->ncomp != 2 ||
          vdg_->nelem != mesh_.GetNE())
      {
         throw std::runtime_error("vdg.bin does not have layout [25,2,ne]");
      }
      double second_component_maximum = 0.0;
      for (int element = 0; element < vdg_->nelem; ++element)
      {
         for (int node = 0; node < vdg_->nnode; ++node)
         {
            second_component_maximum =
               std::max(second_component_maximum,
                        std::abs((*vdg_)(node, 1, element)));
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
   else if (!artificial_viscosity_)
   {
      throw std::runtime_error(
         "analytic HDG operator requires an artificial-viscosity function");
   }
   for (int ib : boundary_conditions_)
   {
      if (ib < 1 || ib > 3)
      {
         throw std::runtime_error("boundary condition type is not in {1,2,3}");
      }
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
   volume_physical_dshape_.resize(
      static_cast<std::size_t>(ne * nq * ndof * 2));
   volume_coordinates_.resize(static_cast<std::size_t>(ne * nq * 2));
   volume_av_.resize(static_cast<std::size_t>(ne * nq));
   face_weighted_normals_.resize(
      static_cast<std::size_t>(ne * 4 * nqf * 2));
   face_coordinates_.resize(
      static_cast<std::size_t>(ne * 4 * nqf * 2));
   face_element_shape_.resize(
      static_cast<std::size_t>(ne * 4 * nqf * ndof));
   face_av_.resize(static_cast<std::size_t>(ne * 4 * nqf));

   mfem::Vector element_shape(ndof);
   mfem::Vector weighted_normal(2);
   mfem::Vector physical(2);
   mfem::DenseMatrix physical_dshape(ndof, 2);
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
         volume_fe_->CalcPhysDShape(*transformation, physical_dshape);
         for (int dof = 0; dof < ndof; ++dof)
         {
            for (int direction = 0; direction < 2; ++direction)
            {
               volume_physical_dshape_[
                  static_cast<std::size_t>(direction) +
                  2 * (static_cast<std::size_t>(dof) +
                       static_cast<std::size_t>(ndof) * index)] =
                  physical_dshape(dof, direction);
            }
         }
         transformation->Transform(point, physical);
         volume_coordinates_[2 * index] = physical[0];
         volume_coordinates_[2 * index + 1] = physical[1];
         if (vdg_)
         {
            const auto exasim_point =
               orientations_[element].MfemToExasim(point.x, point.y);
            volume_av_[index] = EvaluateTensorQ4(
               *vdg_, 0, element, exasim_point[0], exasim_point[1]);
         }
         else
         {
            volume_av_[index] = artificial_viscosity_(physical);
         }
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
            // Mesh::GetFaceElementTransformations updates MFEM's reusable
            // transformation storage, so the element transformation pointer
            // obtained before that call is no longer safe here.  The face
            // transformation is the authoritative physical map at this point.
            face_transformations->Face->Transform(face_point, physical);
            face_coordinates_[2 * face_index] = physical[0];
            face_coordinates_[2 * face_index + 1] = physical[1];
            if (vdg_)
            {
               const auto exasim_point =
                  orientations_[element].MfemToExasim(
                     element_point.x, element_point.y);
               face_av_[face_index] = EvaluateTensorQ4(
                  *vdg_, 0, element, exasim_point[0], exasim_point[1]);
            }
            else
            {
               face_av_[face_index] = artificial_viscosity_(physical);
            }
         }
      }
   }
}

void HDGNavierStokesOperator::RetabulateArtificialViscosity()
{
   if (vdg_ || !artificial_viscosity_)
   {
      throw std::runtime_error(
         "analytic AV retabulation requires an analytic function");
   }
   mfem::Vector physical(2);
   for (std::size_t index = 0; index < volume_av_.size(); ++index)
   {
      physical[0] = volume_coordinates_[2 * index];
      physical[1] = volume_coordinates_[2 * index + 1];
      volume_av_[index] = artificial_viscosity_(physical);
   }
   for (std::size_t index = 0; index < face_av_.size(); ++index)
   {
      physical[0] = face_coordinates_[2 * index];
      physical[1] = face_coordinates_[2 * index + 1];
      face_av_[index] = artificial_viscosity_(physical);
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
   mass_.reserve(static_cast<std::size_t>(ne));
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

      mass_.push_back(mass);
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

void HDGNavierStokesOperator::BuildBoundaryFaceMap()
{
   face_boundary_attribute_.assign(
      static_cast<std::size_t>(mesh_.GetNumFaces()), 0);
   for (int boundary = 0; boundary < mesh_.GetNBE(); ++boundary)
   {
      const int face = mesh_.GetBdrElementFaceIndex(boundary);
      if (face < 0 || face >= mesh_.GetNumFaces())
      {
         throw std::runtime_error("invalid boundary face index");
      }
      face_boundary_attribute_[static_cast<std::size_t>(face)] =
         mesh_.GetBdrAttribute(boundary);
   }
}

HDGState HDGNavierStokesOperator::NewState() const
{
   HDGState solution;
   solution.u.SetSize(mesh_.GetNE() * kElementUnknowns);
   solution.q.SetSize(mesh_.GetNE() * 2 * kElementUnknowns);
   solution.uhat.SetSize(trace_fes_.GetVSize());
   solution.u = 0.0;
   solution.q = 0.0;
   solution.uhat = 0.0;
   return solution;
}

void HDGNavierStokesOperator::SetConstantState(
   const double state[4], HDGState &solution) const
{
   solution = NewState();
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      for (int component = 0; component < 4; ++component)
      {
         for (int dof = 0; dof < 25; ++dof)
         {
            solution.u[UIndex(element, component, dof)] = state[component];
         }
      }
   }
   for (int face = 0; face < mesh_.GetNumFaces(); ++face)
   {
      mfem::Array<int> vdofs;
      trace_fes_.GetFaceVDofs(face, vdofs);
      for (int component = 0; component < 4; ++component)
      {
         for (int dof = 0; dof < 5; ++dof)
         {
            solution.uhat[vdofs[FaceVDofListIndex(component, dof)]] =
               state[component];
         }
      }
   }
   RecomputeGradient(solution);
}

void HDGNavierStokesOperator::ProjectState(
   const StateFunction &function, HDGState &solution) const
{
   if (!function) { throw std::runtime_error("empty state projection function"); }
   solution = NewState();
   const mfem::IntegrationRule &nodes = volume_fe_->GetNodes();
   mfem::Vector physical(2);
   double value[4];
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      mfem::ElementTransformation *transformation =
         mesh_.GetElementTransformation(element);
      for (int dof = 0; dof < 25; ++dof)
      {
         transformation->Transform(nodes.IntPoint(dof), physical);
         function(physical, value);
         for (int component = 0; component < 4; ++component)
         {
            solution.u[UIndex(element, component, dof)] = value[component];
         }
      }
   }

   const mfem::IntegrationRule &face_nodes = trace_fe_->GetNodes();
   for (int face = 0; face < mesh_.GetNumFaces(); ++face)
   {
      mfem::FaceElementTransformations *transformation =
         mesh_.GetFaceElementTransformations(face, 31);
      mfem::Array<int> vdofs;
      trace_fes_.GetFaceVDofs(face, vdofs);
      for (int dof = 0; dof < 5; ++dof)
      {
         transformation->Face->Transform(face_nodes.IntPoint(dof), physical);
         function(physical, value);
         for (int component = 0; component < 4; ++component)
         {
            solution.uhat[vdofs[FaceVDofListIndex(component, dof)]] =
               value[component];
         }
      }
   }
   RecomputeGradient(solution);
}

void HDGNavierStokesOperator::InitializeTraceFromInterior(
   HDGState &solution) const
{
   if (solution.u.Size() != mesh_.GetNE() * kElementUnknowns)
   {
      throw std::runtime_error("interior state has the wrong size");
   }
   solution.uhat.SetSize(trace_fes_.GetVSize());
   solution.uhat = 0.0;
   const mfem::IntegrationRule &face_nodes = trace_fe_->GetNodes();
   mfem::Vector shape(25);
   for (int face = 0; face < mesh_.GetNumFaces(); ++face)
   {
      mfem::FaceElementTransformations *transformation =
         mesh_.GetFaceElementTransformations(face, 31);
      const bool use_first =
         transformation->Elem2No < 0 ||
         transformation->Elem1No < transformation->Elem2No;
      const int element = use_first ?
         transformation->Elem1No : transformation->Elem2No;
      mfem::Array<int> vdofs;
      trace_fes_.GetFaceVDofs(face, vdofs);
      for (int dof = 0; dof < 5; ++dof)
      {
         transformation->SetAllIntPoints(&face_nodes.IntPoint(dof));
         volume_fe_->CalcShape(
            use_first ? transformation->GetElement1IntPoint() :
                        transformation->GetElement2IntPoint(),
            shape);
         for (int component = 0; component < 4; ++component)
         {
            double value = 0.0;
            for (int trial = 0; trial < 25; ++trial)
            {
               value += shape[trial] *
                        solution.u[UIndex(element, component, trial)];
            }
            solution.uhat[vdofs[FaceVDofListIndex(component, dof)]] = value;
         }
      }
   }
   RecomputeGradient(solution);
}

void HDGNavierStokesOperator::LoadExasimVolumeState(
   const ExasimArray &udg, bool load_gradient, HDGState &solution) const
{
   const int required_components = load_gradient ? 12 : 4;
   if (udg.nnode != 25 || udg.ncomp < required_components ||
       udg.nelem != mesh_.GetNE())
   {
      throw std::runtime_error(
         "Exasim volume state has the wrong [25,ncomp,ne] layout");
   }
   solution = NewState();
   const auto target_points = FiniteElementNodes(*volume_fe_);
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      const TensorBasisTransform transform =
         BuildTensorBasisTransform(target_points, orientations_[element]);
      for (int component = 0; component < required_components; ++component)
      {
         double source[25], target[25];
         for (int node = 0; node < 25; ++node)
         {
            source[node] = udg(node, component, element);
         }
         transform.ToTarget(source, target);
         if (component < 4)
         {
            for (int dof = 0; dof < 25; ++dof)
            {
               solution.u[UIndex(element, component, dof)] = target[dof];
            }
         }
         else
         {
            const int direction = (component - 4) / 4;
            const int q_component = (component - 4) % 4;
            for (int dof = 0; dof < 25; ++dof)
            {
               solution.q[
                  QIndex(element, direction, q_component, dof)] =
                     target[dof];
            }
         }
      }
   }
}

void HDGNavierStokesOperator::SetTraceFaceState(
   int face, const std::array<double, 20> &values,
   HDGState &solution) const
{
   if (face < 0 || face >= mesh_.GetNumFaces() ||
       solution.uhat.Size() != trace_fes_.GetVSize())
   {
      throw std::runtime_error("invalid trace face state assignment");
   }
   mfem::Array<int> vdofs;
   trace_fes_.GetFaceVDofs(face, vdofs);
   for (int component = 0; component < 4; ++component)
   {
      for (int dof = 0; dof < 5; ++dof)
      {
         solution.uhat[vdofs[FaceVDofListIndex(component, dof)]] =
            values[component + 4 * dof];
      }
   }
}

void HDGNavierStokesOperator::EvaluateElementState(
   const HDGState &solution, int element,
   const mfem::IntegrationPoint &point, double uq[12]) const
{
   if (element < 0 || element >= mesh_.GetNE() ||
       solution.u.Size() != mesh_.GetNE() * kElementUnknowns ||
       solution.q.Size() != mesh_.GetNE() * 2 * kElementUnknowns)
   {
      throw std::runtime_error("invalid element state evaluation");
   }
   std::fill(uq, uq + 12, 0.0);
   mfem::Vector shape(25);
   volume_fe_->CalcShape(point, shape);
   for (int component = 0; component < 4; ++component)
   {
      for (int dof = 0; dof < 25; ++dof)
      {
         uq[component] += shape[dof] *
            solution.u[UIndex(element, component, dof)];
         for (int direction = 0; direction < 2; ++direction)
         {
            uq[4 + 4 * direction + component] += shape[dof] *
               solution.q[QIndex(
                  element, direction, component, dof)];
         }
      }
   }
}

void HDGNavierStokesOperator::EvaluateTraceState(
   const HDGState &solution, int face,
   const mfem::IntegrationPoint &point, double uhat[4]) const
{
   if (face < 0 || face >= mesh_.GetNumFaces() ||
       solution.uhat.Size() != trace_fes_.GetVSize())
   {
      throw std::runtime_error("invalid trace state evaluation");
   }
   std::fill(uhat, uhat + 4, 0.0);
   mfem::Vector shape(5);
   trace_fe_->CalcShape(point, shape);
   mfem::Array<int> vdofs;
   trace_fes_.GetFaceVDofs(face, vdofs);
   for (int component = 0; component < 4; ++component)
   {
      for (int dof = 0; dof < 5; ++dof)
      {
         uhat[component] += shape[dof] *
            solution.uhat[
               vdofs[FaceVDofListIndex(component, dof)]];
      }
   }
}

void HDGNavierStokesOperator::GetElementTraceVDofs(
   int element, mfem::Array<int> &vdofs) const
{
   vdofs.SetSize(kElementTraceUnknowns);
   mfem::Array<int> faces, orientations, face_vdofs;
   mesh_.GetElementEdges(element, faces, orientations);
   for (int local_face = 0; local_face < 4; ++local_face)
   {
      trace_fes_.GetFaceVDofs(faces[local_face], face_vdofs);
      for (int component = 0; component < 4; ++component)
      {
         for (int dof = 0; dof < 5; ++dof)
         {
            vdofs[LocalTraceIndex(local_face, component, dof)] =
               face_vdofs[FaceVDofListIndex(component, dof)];
         }
      }
   }
}

void HDGNavierStokesOperator::GatherElementTrace(
   const HDGState &solution, int element, mfem::Vector &local_trace) const
{
   mfem::Array<int> vdofs;
   GetElementTraceVDofs(element, vdofs);
   local_trace.SetSize(kElementTraceUnknowns);
   solution.uhat.GetSubVector(vdofs, local_trace);
}

void HDGNavierStokesOperator::RecomputeGradient(
   HDGState &solution) const
{
   if (solution.u.Size() != mesh_.GetNE() * kElementUnknowns ||
       solution.uhat.Size() != trace_fes_.GetVSize())
   {
      throw std::runtime_error("cannot recompute q from an incorrectly sized state");
   }
   solution.q.SetSize(mesh_.GetNE() * 2 * kElementUnknowns);
   mfem::Vector local_trace;
   mfem::Vector scalar_u(25), scalar_trace(20), result(25), boundary(25);
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      GatherElementTrace(solution, element, local_trace);
      for (int component = 0; component < 4; ++component)
      {
         for (int dof = 0; dof < 25; ++dof)
         {
            scalar_u[dof] = solution.u[UIndex(element, component, dof)];
         }
         for (int local_face = 0; local_face < 4; ++local_face)
         {
            for (int dof = 0; dof < 5; ++dof)
            {
               scalar_trace[local_face * 5 + dof] =
                  local_trace[LocalTraceIndex(local_face, component, dof)];
            }
         }
         for (int direction = 0; direction < 2; ++direction)
         {
            c_[direction][element].Mult(scalar_u, result);
            e_[direction][element].Mult(scalar_trace, boundary);
            result -= boundary;
            for (int dof = 0; dof < 25; ++dof)
            {
               solution.q[QIndex(element, direction, component, dof)] =
                  result[dof];
            }
         }
      }
   }
}

void HDGNavierStokesOperator::SetManufacturedSource(StateFunction source)
{
   manufactured_source_ = std::move(source);
}

void HDGNavierStokesOperator::ClearManufacturedSource()
{
   manufactured_source_ = StateFunction();
}

void HDGNavierStokesOperator::SetDirichletStateOverride(StateFunction state)
{
   dirichlet_state_override_ = std::move(state);
}

void HDGNavierStokesOperator::ClearDirichletStateOverride()
{
   dirichlet_state_override_ = StateFunction();
}

void HDGNavierStokesOperator::AssembleElement(
   const HDGState &solution, int element, bool build_jacobian,
   mfem::Vector &ru, mfem::Vector &rh,
   mfem::DenseMatrix *a, mfem::DenseMatrix *b0,
   mfem::DenseMatrix *b1, mfem::DenseMatrix *f,
   mfem::DenseMatrix *k, mfem::DenseMatrix *g0,
   mfem::DenseMatrix *g1, mfem::DenseMatrix *h) const
{
   ru.SetSize(kElementUnknowns);
   rh.SetSize(kElementTraceUnknowns);
   ru = 0.0;
   rh = 0.0;
   if (build_jacobian)
   {
      *a = 0.0;
      *b0 = 0.0;
      *b1 = 0.0;
      *f = 0.0;
      *k = 0.0;
      *g0 = 0.0;
      *g1 = 0.0;
      *h = 0.0;
   }

   mfem::Vector local_trace;
   GatherElementTrace(solution, element, local_trace);
   const int nq = volume_rule_->GetNPoints();
   for (int qpoint = 0; qpoint < nq; ++qpoint)
   {
      const std::size_t index = VolumeIndex(element, qpoint, nq);
      double uq[12] = {};
      for (int component = 0; component < 4; ++component)
      {
         for (int dof = 0; dof < 25; ++dof)
         {
            const double shape =
               volume_shape_[static_cast<std::size_t>(dof + 25 * qpoint)];
            uq[component] += shape *
               solution.u[UIndex(element, component, dof)];
            for (int direction = 0; direction < 2; ++direction)
            {
               uq[4 + 4 * direction + component] += shape *
                  solution.q[QIndex(element, direction, component, dof)];
            }
         }
      }
      double flux[8], flux_jacobian[96];
      NSFlux(uq, volume_av_[index], params_, flux,
             build_jacobian ? flux_jacobian : nullptr);
      const double weight =
         volume_rule_->IntPoint(qpoint).weight * volume_det_j_[index];
      for (int output = 0; output < 4; ++output)
      {
         for (int test = 0; test < 25; ++test)
         {
            const int row = LocalUIndex(output, test);
            for (int direction = 0; direction < 2; ++direction)
            {
               const double gradient =
                  volume_physical_dshape_[
                     static_cast<std::size_t>(direction) +
                     2 * (static_cast<std::size_t>(test) + 25 * index)];
               ru[row] += weight * gradient *
                          flux[output + 4 * direction];
               if (build_jacobian)
               {
                  for (int input = 0; input < 4; ++input)
                  {
                     for (int trial = 0; trial < 25; ++trial)
                     {
                        const double trial_shape =
                           volume_shape_[static_cast<std::size_t>(
                              trial + 25 * qpoint)];
                        (*a)(row, LocalUIndex(input, trial)) +=
                           weight * gradient * trial_shape *
                           flux_jacobian[
                              output + 4 * direction + 8 * input];
                        (*b0)(row, LocalUIndex(input, trial)) +=
                           weight * gradient * trial_shape *
                           flux_jacobian[
                              output + 4 * direction +
                              8 * (4 + input)];
                        (*b1)(row, LocalUIndex(input, trial)) +=
                           weight * gradient * trial_shape *
                           flux_jacobian[
                              output + 4 * direction +
                              8 * (8 + input)];
                     }
                  }
               }
            }
         }
      }

      if (manufactured_source_)
      {
         mfem::Vector physical(
            const_cast<double *>(&volume_coordinates_[2 * index]), 2);
         double source[4];
         manufactured_source_(physical, source);
         for (int output = 0; output < 4; ++output)
         {
            for (int test = 0; test < 25; ++test)
            {
               ru[LocalUIndex(output, test)] +=
                  weight *
                  volume_shape_[static_cast<std::size_t>(
                     test + 25 * qpoint)] *
                  source[output];
            }
         }
      }
   }
   mfem::Array<int> faces, face_orientations;
   mesh_.GetElementEdges(element, faces, face_orientations);
   const int nqf = face_rule_->GetNPoints();
   for (int local_face = 0; local_face < 4; ++local_face)
   {
      const int face = faces[local_face];
      const int boundary_attribute =
         face_boundary_attribute_[static_cast<std::size_t>(face)];
      for (int qpoint = 0; qpoint < nqf; ++qpoint)
      {
         const std::size_t index =
            FaceIndex(element, local_face, qpoint, nqf);
         const double nx_weighted = face_weighted_normals_[2 * index];
         const double ny_weighted = face_weighted_normals_[2 * index + 1];
         const double surface_jacobian =
            std::hypot(nx_weighted, ny_weighted);
         const double normal[2] =
         {
            nx_weighted / surface_jacobian,
            ny_weighted / surface_jacobian
         };
         const double weight =
            face_rule_->IntPoint(qpoint).weight * surface_jacobian;
         double uq[12] = {};
         double uhat[4] = {};
         for (int component = 0; component < 4; ++component)
         {
            for (int dof = 0; dof < 25; ++dof)
            {
               const double shape =
                  face_element_shape_[static_cast<std::size_t>(dof) +
                                      25 * index];
               uq[component] += shape *
                  solution.u[UIndex(element, component, dof)];
               for (int direction = 0; direction < 2; ++direction)
               {
                  uq[4 + 4 * direction + component] += shape *
                     solution.q[QIndex(element, direction, component, dof)];
               }
            }
            for (int dof = 0; dof < 5; ++dof)
            {
               uhat[component] +=
                  trace_shape_[static_cast<std::size_t>(dof + 5 * qpoint)] *
                  local_trace[LocalTraceIndex(
                     local_face, component, dof)];
            }
         }

         double trace_uq[12];
         std::copy(uq, uq + 12, trace_uq);
         std::copy(uhat, uhat + 4, trace_uq);
         double flux[8], flux_jacobian[96];
         NSFlux(trace_uq, face_av_[index], params_, flux,
                build_jacobian ? flux_jacobian : nullptr);
         double fhat[4];
         double normal_flux[4];
         for (int output = 0; output < 4; ++output)
         {
            normal_flux[output] =
               flux[output] * normal[0] +
               flux[output + 4] * normal[1];
            fhat[output] =
               normal_flux[output] +
               params_.tau * (uq[output] - uhat[output]);
            for (int test = 0; test < 25; ++test)
            {
               const int row = LocalUIndex(output, test);
               const double test_shape =
                  face_element_shape_[static_cast<std::size_t>(test) +
                                      25 * index];
               ru[row] -= weight * test_shape * fhat[output];
               if (build_jacobian)
               {
                  for (int trial = 0; trial < 25; ++trial)
                  {
                     const double trial_shape =
                        face_element_shape_[static_cast<std::size_t>(trial) +
                                            25 * index];
                     (*a)(row, LocalUIndex(output, trial)) -=
                        weight * test_shape * params_.tau * trial_shape;
                     for (int input = 0; input < 4; ++input)
                     {
                        const double dflux_q0 =
                           normal[0] * flux_jacobian[
                              output + 8 * (4 + input)] +
                           normal[1] * flux_jacobian[
                              output + 4 + 8 * (4 + input)];
                        const double dflux_q1 =
                           normal[0] * flux_jacobian[
                              output + 8 * (8 + input)] +
                           normal[1] * flux_jacobian[
                              output + 4 + 8 * (8 + input)];
                        (*b0)(row, LocalUIndex(input, trial)) -=
                           weight * test_shape * dflux_q0 * trial_shape;
                        (*b1)(row, LocalUIndex(input, trial)) -=
                           weight * test_shape * dflux_q1 * trial_shape;
                     }
                  }
                  for (int input = 0; input < 4; ++input)
                  {
                     const double dflux_u =
                        normal[0] *
                           flux_jacobian[output + 8 * input] +
                        normal[1] *
                           flux_jacobian[output + 4 + 8 * input] -
                        (output == input ? params_.tau : 0.0);
                     for (int trial = 0; trial < 5; ++trial)
                     {
                        (*f)(row, LocalTraceIndex(
                           local_face, input, trial)) -=
                           weight * test_shape * dflux_u *
                           trace_shape_[static_cast<std::size_t>(
                              trial + 5 * qpoint)];
                     }
                  }
               }
            }
         }

         if (boundary_attribute == 0)
         {
            for (int output = 0; output < 4; ++output)
            {
               for (int test = 0; test < 5; ++test)
               {
                  const int row =
                     LocalTraceIndex(local_face, output, test);
                  const double test_shape =
                     trace_shape_[static_cast<std::size_t>(
                        test + 5 * qpoint)];
                  rh[row] += weight * test_shape * fhat[output];
                  if (build_jacobian)
                  {
                     for (int trial = 0; trial < 25; ++trial)
                     {
                        const double trial_shape =
                           face_element_shape_[
                              static_cast<std::size_t>(trial) + 25 * index];
                        (*k)(row, LocalUIndex(output, trial)) +=
                           weight * test_shape * params_.tau * trial_shape;
                        for (int input = 0; input < 4; ++input)
                        {
                           const double dflux_q0 =
                              normal[0] * flux_jacobian[
                                 output + 8 * (4 + input)] +
                              normal[1] * flux_jacobian[
                                 output + 4 + 8 * (4 + input)];
                           const double dflux_q1 =
                              normal[0] * flux_jacobian[
                                 output + 8 * (8 + input)] +
                              normal[1] * flux_jacobian[
                                 output + 4 + 8 * (8 + input)];
                           (*g0)(row, LocalUIndex(input, trial)) +=
                              weight * test_shape * dflux_q0 * trial_shape;
                           (*g1)(row, LocalUIndex(input, trial)) +=
                              weight * test_shape * dflux_q1 * trial_shape;
                        }
                     }
                     for (int input = 0; input < 4; ++input)
                     {
                        const double dflux_u =
                           normal[0] *
                              flux_jacobian[output + 8 * input] +
                           normal[1] *
                              flux_jacobian[output + 4 + 8 * input] -
                           (output == input ? params_.tau : 0.0);
                        for (int trial = 0; trial < 5; ++trial)
                        {
                           (*h)(row, LocalTraceIndex(
                              local_face, input, trial)) +=
                              weight * test_shape * dflux_u *
                              trace_shape_[static_cast<std::size_t>(
                                 trial + 5 * qpoint)];
                        }
                     }
                  }
               }
            }
         }
         else
         {
            double fb[4], fb_uq[48], fb_uh[16];
            if (dirichlet_state_override_)
            {
               mfem::Vector physical(
                  const_cast<double *>(&face_coordinates_[2 * index]), 2);
               double prescribed[4];
               dirichlet_state_override_(physical, prescribed);
               for (int output = 0; output < 4; ++output)
               {
                  fb[output] = prescribed[output] - uhat[output];
               }
               if (build_jacobian)
               {
                  std::fill(fb_uq, fb_uq + 48, 0.0);
                  std::fill(fb_uh, fb_uh + 16, 0.0);
                  for (int component = 0; component < 4; ++component)
                  {
                     fb_uh[component + 4 * component] = -1.0;
                  }
               }
            }
            else
            {
               if (boundary_attribute < 1 || boundary_attribute > 3)
               {
                  throw std::runtime_error(
                     "boundary attribute is outside the YAML map");
               }
               const int ib =
                  boundary_conditions_[boundary_attribute - 1];
               NSFbouHdg(ib, uq, uhat, normal, params_, fb,
                         build_jacobian ? fb_uq : nullptr,
                         build_jacobian ? fb_uh : nullptr);
            }
            for (int output = 0; output < 4; ++output)
            {
               for (int test = 0; test < 5; ++test)
               {
                  const int row =
                     LocalTraceIndex(local_face, output, test);
                  const double test_shape =
                     trace_shape_[static_cast<std::size_t>(
                        test + 5 * qpoint)];
                  rh[row] += weight * test_shape * fb[output];
                  if (build_jacobian)
                  {
                     for (int input = 0; input < 4; ++input)
                     {
                        for (int trial = 0; trial < 25; ++trial)
                        {
                           const double trial_shape =
                              face_element_shape_[
                                 static_cast<std::size_t>(trial) +
                                 25 * index];
                           (*k)(row, LocalUIndex(input, trial)) +=
                              weight * test_shape *
                              fb_uq[output + 4 * input] * trial_shape;
                           (*g0)(row, LocalUIndex(input, trial)) +=
                              weight * test_shape *
                              fb_uq[output + 4 * (4 + input)] *
                              trial_shape;
                           (*g1)(row, LocalUIndex(input, trial)) +=
                              weight * test_shape *
                              fb_uq[output + 4 * (8 + input)] *
                              trial_shape;
                        }
                        for (int trial = 0; trial < 5; ++trial)
                        {
                           (*h)(row, LocalTraceIndex(
                              local_face, input, trial)) +=
                              weight * test_shape *
                              fb_uh[output + 4 * input] *
                              trace_shape_[static_cast<std::size_t>(
                                 trial + 5 * qpoint)];
                        }
                     }
                  }
               }
            }
         }
      }
   }
}

HDGResidualNorms HDGNavierStokesOperator::Assemble(
   const HDGState &solution, bool build_jacobian,
   double pseudo_time_inverse_step)
{
   if (solution.u.Size() != mesh_.GetNE() * kElementUnknowns ||
       solution.q.Size() != mesh_.GetNE() * 2 * kElementUnknowns ||
       solution.uhat.Size() != trace_fes_.GetVSize())
   {
      throw std::runtime_error("HDG state has an incorrect vector size");
   }
   volume_residual_ = 0.0;
   trace_residual_ = 0.0;
   condensed_residual_ = 0.0;
   condensed_rhs_ = 0.0;
   recovery_is_current_ = false;
   if (build_jacobian)
   {
      inverse_a_f_.clear();
      inverse_a_ru_.clear();
      if (!condensed_matrix_)
      {
         condensed_matrix_ = std::make_unique<mfem::SparseMatrix>(
            trace_fes_.GetVSize(), trace_fes_.GetVSize());
      }
      else
      {
         *condensed_matrix_ = 0.0;
      }
   }

   mfem::Vector ru(kElementUnknowns), rh(kElementTraceUnknowns);
   mfem::DenseMatrix a(kElementUnknowns), b0(kElementUnknowns);
   mfem::DenseMatrix b1(kElementUnknowns);
   mfem::DenseMatrix f(kElementUnknowns, kElementTraceUnknowns);
   mfem::DenseMatrix k(kElementTraceUnknowns, kElementUnknowns);
   mfem::DenseMatrix g0(kElementTraceUnknowns, kElementUnknowns);
   mfem::DenseMatrix g1(kElementTraceUnknowns, kElementUnknowns);
   mfem::DenseMatrix h(kElementTraceUnknowns);
   mfem::Array<int> trace_vdofs;
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      AssembleElement(solution, element, build_jacobian, ru, rh,
                      &a, &b0, &b1, &f, &k, &g0, &g1, &h);
      for (int i = 0; i < kElementUnknowns; ++i)
      {
         volume_residual_[
            element * kElementUnknowns + i] = ru[i];
      }
      GetElementTraceVDofs(element, trace_vdofs);
      trace_residual_.AddElementVector(trace_vdofs, rh);

      if (!build_jacobian) { continue; }

      // q fold: A += B_d C_d, F -= B_d E_d and the same for K/H.
      for (int direction = 0; direction < 2; ++direction)
      {
         const mfem::DenseMatrix &bd = direction == 0 ? b0 : b1;
         const mfem::DenseMatrix &gd = direction == 0 ? g0 : g1;
         const mfem::DenseMatrix &cd = c_[direction][element];
         const mfem::DenseMatrix &ed = e_[direction][element];
         for (int out = 0; out < 4; ++out)
         {
            for (int input = 0; input < 4; ++input)
            {
               for (int i = 0; i < 25; ++i)
               {
                  const int row_u = LocalUIndex(out, i);
                  for (int j = 0; j < 25; ++j)
                  {
                     const int col_u = LocalUIndex(input, j);
                     double bac = 0.0;
                     for (int m = 0; m < 25; ++m)
                     {
                        bac += bd(row_u, LocalUIndex(input, m)) *
                               cd(m, j);
                     }
                     a(row_u, col_u) += bac;
                  }
                  for (int local_face = 0; local_face < 4; ++local_face)
                  {
                     for (int j = 0; j < 5; ++j)
                     {
                        double bae = 0.0;
                        for (int m = 0; m < 25; ++m)
                        {
                           bae += bd(row_u, LocalUIndex(input, m)) *
                                  ed(m, local_face * 5 + j);
                        }
                        f(row_u, LocalTraceIndex(
                           local_face, input, j)) -= bae;
                     }
                  }
               }
               for (int local_face_row = 0;
                    local_face_row < 4; ++local_face_row)
               {
                  for (int i = 0; i < 5; ++i)
                  {
                     const int row_h =
                        LocalTraceIndex(local_face_row, out, i);
                     for (int j = 0; j < 25; ++j)
                     {
                        double gac = 0.0;
                        for (int m = 0; m < 25; ++m)
                        {
                           gac += gd(row_h, LocalUIndex(input, m)) *
                                  cd(m, j);
                        }
                        k(row_h, LocalUIndex(input, j)) += gac;
                     }
                     for (int local_face_col = 0;
                          local_face_col < 4; ++local_face_col)
                     {
                        for (int j = 0; j < 5; ++j)
                        {
                           double gae = 0.0;
                           for (int m = 0; m < 25; ++m)
                           {
                              gae += gd(row_h, LocalUIndex(input, m)) *
                                     ed(m, local_face_col * 5 + j);
                           }
                           h(row_h, LocalTraceIndex(
                              local_face_col, input, j)) -= gae;
                        }
                     }
                  }
               }
            }
         }
      }

      if (pseudo_time_inverse_step > 0.0)
      {
         for (int component = 0; component < 4; ++component)
         {
            for (int i = 0; i < 25; ++i)
            {
               for (int j = 0; j < 25; ++j)
               {
                  a(LocalUIndex(component, i),
                    LocalUIndex(component, j)) +=
                     pseudo_time_inverse_step * mass_[element](i, j);
               }
            }
         }
      }

      mfem::DenseMatrixInverse a_inverse(a);
      mfem::DenseMatrix inverse_a_f(kElementUnknowns,
                                    kElementTraceUnknowns);
      mfem::Vector inverse_a_ru(kElementUnknowns);
      a_inverse.Mult(f, inverse_a_f);
      a_inverse.Mult(ru, inverse_a_ru);

      mfem::DenseMatrix hc(h);
      for (int i = 0; i < kElementTraceUnknowns; ++i)
      {
         for (int j = 0; j < kElementTraceUnknowns; ++j)
         {
            double product = 0.0;
            for (int m = 0; m < kElementUnknowns; ++m)
            {
               product += k(i, m) * inverse_a_f(m, j);
            }
            hc(i, j) -= product;
         }
      }
      mfem::Vector local_condensed_residual(rh);
      for (int i = 0; i < kElementTraceUnknowns; ++i)
      {
         double product = 0.0;
         for (int m = 0; m < kElementUnknowns; ++m)
         {
            product += k(i, m) * inverse_a_ru[m];
         }
         local_condensed_residual[i] -= product;
      }
      condensed_residual_.AddElementVector(
         trace_vdofs, local_condensed_residual);
      local_condensed_residual *= -1.0;
      condensed_rhs_.AddElementVector(
         trace_vdofs, local_condensed_residual);

      // skip_zeros=0 is load-bearing: the first pass builds the complete graph.
      condensed_matrix_->AddSubMatrix(
         trace_vdofs, trace_vdofs, hc, 0);
      inverse_a_f_.push_back(std::move(inverse_a_f));
      inverse_a_ru_.push_back(std::move(inverse_a_ru));
   }

   if (build_jacobian && !condensed_matrix_->Finalized())
   {
      // Preserve the exact-zero graph entries inserted above so subsequent
      // Newton assemblies can reuse the finalized sparsity pattern.
      condensed_matrix_->Finalize(0);
   }
   if (build_jacobian)
   {
      recovery_is_current_ = true;
   }
   return {VectorNorm(volume_residual_), VectorNorm(trace_residual_)};
}

const mfem::Vector &HDGNavierStokesOperator::VolumeResidual() const
{
   return volume_residual_;
}

const mfem::Vector &HDGNavierStokesOperator::TraceResidual() const
{
   return trace_residual_;
}

const mfem::Vector &HDGNavierStokesOperator::CondensedResidual() const
{
   return condensed_residual_;
}

const mfem::Vector &HDGNavierStokesOperator::CondensedRHS() const
{
   return condensed_rhs_;
}

const mfem::SparseMatrix &HDGNavierStokesOperator::CondensedMatrix() const
{
   if (!condensed_matrix_ || !recovery_is_current_)
   {
      throw std::runtime_error("condensed matrix requested before full assembly");
   }
   return *condensed_matrix_;
}

void HDGNavierStokesOperator::RecoverIncrement(
   const mfem::Vector &trace_increment,
   mfem::Vector &volume_increment) const
{
   if (!recovery_is_current_ ||
       inverse_a_f_.size() != static_cast<std::size_t>(mesh_.GetNE()))
   {
      throw std::runtime_error("local recovery requested before full assembly");
   }
   if (trace_increment.Size() != trace_fes_.GetVSize())
   {
      throw std::runtime_error("trace increment has the wrong size");
   }
   volume_increment.SetSize(mesh_.GetNE() * kElementUnknowns);
   volume_increment = 0.0;
   mfem::Array<int> trace_vdofs;
   mfem::Vector local_trace(kElementTraceUnknowns);
   mfem::Vector product(kElementUnknowns);
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      GetElementTraceVDofs(element, trace_vdofs);
      trace_increment.GetSubVector(trace_vdofs, local_trace);
      inverse_a_f_[element].Mult(local_trace, product);
      for (int i = 0; i < kElementUnknowns; ++i)
      {
         volume_increment[element * kElementUnknowns + i] =
            -inverse_a_ru_[element][i] - product[i];
      }
   }
}

double HDGNavierStokesOperator::L2Error(
   const HDGState &solution, const StateFunction &exact) const
{
   const mfem::IntegrationRule &rule =
      mfem::IntRules.Get(mfem::Geometry::SQUARE, 12);
   mfem::Vector shape(25), physical(2);
   double error_squared = 0.0;
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      mfem::ElementTransformation *transformation =
         mesh_.GetElementTransformation(element);
      for (int qpoint = 0; qpoint < rule.GetNPoints(); ++qpoint)
      {
         const mfem::IntegrationPoint &point = rule.IntPoint(qpoint);
         transformation->SetIntPoint(&point);
         volume_fe_->CalcShape(point, shape);
         transformation->Transform(point, physical);
         double expected[4], actual[4] = {};
         exact(physical, expected);
         for (int component = 0; component < 4; ++component)
         {
            for (int dof = 0; dof < 25; ++dof)
            {
               actual[component] += shape[dof] *
                  solution.u[UIndex(element, component, dof)];
            }
            const double difference = actual[component] - expected[component];
            error_squared += point.weight * transformation->Weight() *
                             difference * difference;
         }
      }
   }
   return std::sqrt(error_squared);
}

std::array<double, 4>
HDGNavierStokesOperator::ComponentRelativeL2(
   const HDGState &solution, const HDGState &reference) const
{
   const int expected_size = mesh_.GetNE() * kElementUnknowns;
   if (solution.u.Size() != expected_size ||
       reference.u.Size() != expected_size)
   {
      throw std::runtime_error(
         "relative L2 comparison received an incorrectly sized state");
   }
   std::array<double, 4> error_squared{};
   std::array<double, 4> reference_squared{};
   const int nq = volume_rule_->GetNPoints();
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      for (int qpoint = 0; qpoint < nq; ++qpoint)
      {
         const std::size_t index = VolumeIndex(element, qpoint, nq);
         const double weight =
            volume_rule_->IntPoint(qpoint).weight * volume_det_j_[index];
         for (int component = 0; component < 4; ++component)
         {
            double value = 0.0;
            double expected = 0.0;
            for (int dof = 0; dof < 25; ++dof)
            {
               const double shape =
                  volume_shape_[static_cast<std::size_t>(
                     dof + 25 * qpoint)];
               value += shape *
                  solution.u[UIndex(element, component, dof)];
               expected += shape *
                  reference.u[UIndex(element, component, dof)];
            }
            const double difference = value - expected;
            error_squared[component] +=
               weight * difference * difference;
            reference_squared[component] +=
               weight * expected * expected;
         }
      }
   }
   std::array<double, 4> relative{};
   for (int component = 0; component < 4; ++component)
   {
      if (!(reference_squared[component] > 0.0))
      {
         throw std::runtime_error(
            "reference field has zero L2 norm");
      }
      relative[component] =
         std::sqrt(error_squared[component] /
                   reference_squared[component]);
   }
   return relative;
}

double HDGNavierStokesOperator::MinimumDensity(
   const HDGState &solution) const
{
   double minimum = std::numeric_limits<double>::infinity();
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      for (int qpoint = 0; qpoint < 25; ++qpoint)
      {
         double density = 0.0;
         for (int dof = 0; dof < 25; ++dof)
         {
            density += volume_shape_[static_cast<std::size_t>(
                          dof + 25 * qpoint)] *
                       solution.u[UIndex(element, 0, dof)];
         }
         minimum = std::min(minimum, density);
      }
   }
   return minimum;
}

double HDGNavierStokesOperator::MinimumPressure(
   const HDGState &solution) const
{
   double minimum = std::numeric_limits<double>::infinity();
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      for (int qpoint = 0; qpoint < 25; ++qpoint)
      {
         double state[4] = {};
         for (int component = 0; component < 4; ++component)
         {
            for (int dof = 0; dof < 25; ++dof)
            {
               state[component] += volume_shape_[static_cast<std::size_t>(
                                      dof + 25 * qpoint)] *
                  solution.u[UIndex(element, component, dof)];
            }
         }
         minimum = std::min(minimum, Pressure(state, params_));
      }
   }
   return minimum;
}

double HDGNavierStokesOperator::YSymmetryError(
   const HDGState &solution) const
{
   // The analytic and reference meshes are ordered in mirrored circumferential
   // columns. Compare matching physical quadrature points by nearest reflection.
   struct Sample
   {
      double x, y;
      std::array<double, 4> u;
   };
   std::vector<Sample> samples;
   samples.reserve(static_cast<std::size_t>(mesh_.GetNE() * 25));
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      for (int qpoint = 0; qpoint < 25; ++qpoint)
      {
         const std::size_t index = VolumeIndex(element, qpoint, 25);
         Sample sample;
         sample.x = volume_coordinates_[2 * index];
         sample.y = volume_coordinates_[2 * index + 1];
         sample.u = {{0.0, 0.0, 0.0, 0.0}};
         for (int component = 0; component < 4; ++component)
         {
            for (int dof = 0; dof < 25; ++dof)
            {
               sample.u[component] +=
                  volume_shape_[static_cast<std::size_t>(
                     dof + 25 * qpoint)] *
                  solution.u[UIndex(element, component, dof)];
            }
         }
         samples.push_back(sample);
      }
   }
   struct PairHash
   {
      std::size_t operator()(
         const std::pair<long long, long long> &key) const
      {
         const std::size_t first = std::hash<long long>()(key.first);
         const std::size_t second = std::hash<long long>()(key.second);
         return first ^ (second + 0x9e3779b97f4a7c15ULL +
                         (first << 6) + (first >> 2));
      }
   };
   constexpr double coordinate_scale = 1.0e10;
   std::unordered_multimap<std::pair<long long, long long>,
                           const Sample *, PairHash> lookup;
   lookup.reserve(samples.size());
   for (const Sample &sample : samples)
   {
      lookup.emplace(
         std::make_pair(
            std::llround(sample.x * coordinate_scale),
            std::llround(sample.y * coordinate_scale)),
         &sample);
   }
   double maximum = 0.0;
   int maximum_component = -1;
   const Sample *maximum_sample = nullptr;
   const Sample *maximum_match = nullptr;
   for (const Sample &sample : samples)
   {
      const Sample *match = nullptr;
      double best = std::numeric_limits<double>::infinity();
      const long long key_x = std::llround(sample.x * coordinate_scale);
      const long long key_y = std::llround(-sample.y * coordinate_scale);
      for (long long dx = -1; dx <= 1; ++dx)
      {
         for (long long dy = -1; dy <= 1; ++dy)
         {
            const auto range =
               lookup.equal_range(std::make_pair(key_x + dx, key_y + dy));
            for (auto candidate = range.first;
                 candidate != range.second; ++candidate)
            {
               const double distance =
                  std::hypot(candidate->second->x - sample.x,
                             candidate->second->y + sample.y);
               if (distance < best)
               {
                  best = distance;
                  match = candidate->second;
               }
            }
         }
      }
      if (!match || best > 1.0e-9)
      {
         throw std::runtime_error(
            "mesh quadrature is not reflection-paired for symmetry check");
      }
      for (int component = 0; component < 4; ++component)
      {
         const double sign = component == 2 ? -1.0 : 1.0;
         const double error =
            std::abs(sample.u[component] - sign * match->u[component]) /
            std::max({1.0, std::abs(sample.u[component]),
                      std::abs(match->u[component])});
         if (error > maximum)
         {
            maximum = error;
            maximum_component = component;
            maximum_sample = &sample;
            maximum_match = match;
         }
      }
   }
   if (maximum > 1.0e-6 && maximum_sample && maximum_match)
   {
      mfem::out << "symmetry maximum detail: component="
                << maximum_component
                << " x=" << maximum_sample->x
                << " y=" << maximum_sample->y
                << " value=" << maximum_sample->u[maximum_component]
                << " reflected_value="
                << maximum_match->u[maximum_component] << '\n';
   }
   return maximum;
}

double HDGNavierStokesOperator::TraceOrientationError() const
{
   StateFunction linear = [](const mfem::Vector &x, double state[4])
   {
      state[0] = 1.0 + 0.2 * x[0] - 0.3 * x[1];
      state[1] = -0.4 + 0.7 * x[0] + 0.1 * x[1];
      state[2] = 0.5 - 0.2 * x[0] + 0.9 * x[1];
      state[3] = 2.0 + 0.3 * x[0] + 0.4 * x[1];
   };
   HDGState projected;
   ProjectState(linear, projected);
   mfem::Vector shape(25), trace_shape(5);
   mfem::Array<int> face_vdofs;
   double maximum = 0.0;
   for (int face = 0; face < mesh_.GetNumFaces(); ++face)
   {
      mfem::FaceElementTransformations *transformation =
         mesh_.GetFaceElementTransformations(face, 31);
      if (transformation->Elem2No < 0) { continue; }
      trace_fes_.GetFaceVDofs(face, face_vdofs);
      for (int qpoint = 0; qpoint < 5; ++qpoint)
      {
         const mfem::IntegrationPoint &point = face_rule_->IntPoint(qpoint);
         transformation->SetAllIntPoints(&point);
         trace_fe_->CalcShape(point, trace_shape);
         for (int side = 0; side < 2; ++side)
         {
            const int element =
               side == 0 ? transformation->Elem1No : transformation->Elem2No;
            const mfem::IntegrationPoint &element_point =
               side == 0 ? transformation->GetElement1IntPoint() :
                           transformation->GetElement2IntPoint();
            volume_fe_->CalcShape(element_point, shape);
            for (int component = 0; component < 4; ++component)
            {
               double volume_value = 0.0;
               double trace_value = 0.0;
               for (int dof = 0; dof < 25; ++dof)
               {
                  volume_value += shape[dof] *
                     projected.u[UIndex(element, component, dof)];
               }
               for (int dof = 0; dof < 5; ++dof)
               {
                  trace_value += trace_shape[dof] *
                     projected.uhat[
                        face_vdofs[FaceVDofListIndex(component, dof)]];
               }
               maximum = std::max(maximum,
                                  std::abs(volume_value - trace_value));
            }
         }
      }
   }
   return maximum;
}

void HDGNavierStokesOperator::FillConservativeGridFunction(
   const HDGState &solution, mfem::GridFunction &field) const
{
   if (field.FESpace() != &volume_fes_)
   {
      throw std::runtime_error("conservative GridFunction uses the wrong space");
   }
   mfem::Array<int> vdofs;
   mfem::Vector local(kElementUnknowns);
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      volume_fes_.GetElementVDofs(element, vdofs);
      for (int component = 0; component < 4; ++component)
      {
         for (int dof = 0; dof < 25; ++dof)
         {
            local[dof + 25 * component] =
               solution.u[UIndex(element, component, dof)];
         }
      }
      field.SetSubVector(vdofs, local);
   }
}

void HDGNavierStokesOperator::FillPrimitiveGridFunction(
   const HDGState &solution, mfem::GridFunction &field) const
{
   if (field.FESpace() != &volume_fes_)
   {
      throw std::runtime_error("primitive GridFunction uses the wrong space");
   }
   mfem::Array<int> vdofs;
   mfem::Vector local(kElementUnknowns);
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      for (int dof = 0; dof < 25; ++dof)
      {
         double state[4], primitive[4];
         for (int component = 0; component < 4; ++component)
         {
            state[component] =
               solution.u[UIndex(element, component, dof)];
         }
         NSVisScalars(state, primitive);
         for (int component = 0; component < 4; ++component)
         {
            local[dof + 25 * component] = primitive[component];
         }
      }
      volume_fes_.GetElementVDofs(element, vdofs);
      field.SetSubVector(vdofs, local);
   }
}

void HDGNavierStokesOperator::FillArtificialViscosityGridFunction(
   mfem::GridFunction &field) const
{
   mfem::FiniteElementSpace *space = field.FESpace();
   if (!space || space->GetVDim() != 1 ||
       space->GetNE() != mesh_.GetNE() ||
       space->GetFE(0)->GetDof() != 25)
   {
      throw std::runtime_error(
         "artificial-viscosity GridFunction uses the wrong space");
   }
   const auto target_points = FiniteElementNodes(*space->GetFE(0));
   mfem::Array<int> dofs;
   mfem::Vector local(25), physical(2);
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      if (vdg_)
      {
         const TensorBasisTransform transform =
            BuildTensorBasisTransform(target_points,
                                      orientations_[element]);
         double source[25], target[25];
         for (int node = 0; node < 25; ++node)
         {
            source[node] = (*vdg_)(node, 0, element);
         }
         transform.ToTarget(source, target);
         for (int dof = 0; dof < 25; ++dof) { local[dof] = target[dof]; }
      }
      else
      {
         mfem::ElementTransformation *transformation =
            mesh_.GetElementTransformation(element);
         const mfem::IntegrationRule &nodes = space->GetFE(element)->GetNodes();
         for (int dof = 0; dof < 25; ++dof)
         {
            transformation->Transform(nodes.IntPoint(dof), physical);
            local[dof] = artificial_viscosity_(physical);
         }
      }
      space->GetElementDofs(element, dofs);
      field.SetSubVector(dofs, local);
   }
}

} // namespace hdg_ns

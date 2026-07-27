#include "discretization/hdg_operator.hpp"
#include "io/exasim_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace hycfd
{
namespace
{

int GeometryFaceCount(mfem::Geometry::Type geometry)
{
   switch (geometry)
   {
      case mfem::Geometry::TRIANGLE: return 3;
      case mfem::Geometry::SQUARE: return 4;
      default:
         throw std::runtime_error(
            "HDG operator supports triangle and quadrilateral"
            " elements only");
   }
}

} // namespace

HDGOperator::HDGOperator(
   mfem::ParMesh &mesh, const ExasimArray &vdg,
   const std::vector<ElementOrientation> &orientations,
   const PhysicsModel &physics,
   const std::vector<int> &attr_to_bcid,
   const HDGOptions &options,
   const std::vector<int> &serial_element_ids)
   : mesh_(mesh),
     vdg_(&vdg),
     global_element_(serial_element_ids),
     orientations_(orientations),
     physics_(physics),
     attr_to_bcid_(attr_to_bcid),
     options_(options),
     ncu_(physics.NumComponents()),
     volume_fec_(options.order, 2),
     volume_fes_(&mesh_, &volume_fec_, ncu_, mfem::Ordering::byVDIM),
     trace_fec_(options.order, 2),
     trace_fes_(&mesh_, &trace_fec_, ncu_, mfem::Ordering::byVDIM)
{
   Initialize();
}

HDGOperator::HDGOperator(
   mfem::ParMesh &mesh,
   const std::vector<ElementOrientation> &orientations,
   ScalarFunction artificial_viscosity,
   const PhysicsModel &physics,
   const std::vector<int> &attr_to_bcid,
   const HDGOptions &options)
   : mesh_(mesh),
     orientations_(orientations),
     artificial_viscosity_(std::move(artificial_viscosity)),
     physics_(physics),
     attr_to_bcid_(attr_to_bcid),
     options_(options),
     ncu_(physics.NumComponents()),
     volume_fec_(options.order, 2),
     volume_fes_(&mesh_, &volume_fec_, ncu_, mfem::Ordering::byVDIM),
     trace_fec_(options.order, 2),
     trace_fes_(&mesh_, &trace_fec_, ncu_, mfem::Ordering::byVDIM)
{
   Initialize();
}

HDGOperator::HDGOperator(
   mfem::ParMesh &mesh, ScalarFunction artificial_viscosity,
   const PhysicsModel &physics,
   const std::vector<int> &attr_to_bcid,
   const HDGOptions &options)
   : mesh_(mesh),
     artificial_viscosity_(std::move(artificial_viscosity)),
     physics_(physics),
     attr_to_bcid_(attr_to_bcid),
     options_(options),
     ncu_(physics.NumComponents()),
     volume_fec_(options.order, 2),
     volume_fes_(&mesh_, &volume_fec_, ncu_, mfem::Ordering::byVDIM),
     trace_fec_(options.order, 2),
     trace_fes_(&mesh_, &trace_fec_, ncu_, mfem::Ordering::byVDIM)
{
   orientations_.resize(static_cast<std::size_t>(mesh_.GetGlobalNE()));
   for (ElementOrientation &orientation : orientations_)
   {
      orientation.mfem_corner_to_exasim = {{0, 1, 2, 3}};
   }
   Initialize();
}

void HDGOperator::Initialize()
{
   ValidateInputs();
   nfdof_ = options_.order + 1;
   const int face_rule_order =
      2 * options_.order + options_.quadrature_increment;
   face_rule_ = &mfem::IntRules.Get(mfem::Geometry::SEGMENT,
                                    face_rule_order);
   trace_fe_ =
      trace_fec_.FiniteElementForGeometry(mfem::Geometry::SEGMENT);
   if (!trace_fe_ || trace_fe_->GetDof() != nfdof_)
   {
      throw std::runtime_error(
         "trace element does not have order+1 dofs per face");
   }
   if (trace_fes_.GetVSize() != ncu_ * nfdof_ * mesh_.GetNumFaces())
   {
      throw std::runtime_error(
         "DG_Interface trace size does not equal ncu*nfdof*nfaces");
   }
   mfem::Array<int> face_vdofs;
   for (int face = 0; face < mesh_.GetNumFaces(); ++face)
   {
      trace_fes_.GetFaceVDofs(face, face_vdofs);
      if (face_vdofs.Size() != ncu_ * nfdof_)
      {
         throw std::runtime_error(
            "trace face does not have ncu*nfdof vector dofs");
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

   volume_residual_.SetSize(total_u_size_);
   trace_residual_.SetSize(trace_fes_.GetVSize());
   condensed_residual_.SetSize(trace_fes_.GetVSize());
   condensed_rhs_.SetSize(trace_fes_.GetVSize());
   inverse_a_f_.reserve(static_cast<std::size_t>(mesh_.GetNE()));
   inverse_a_ru_.reserve(static_cast<std::size_t>(mesh_.GetNE()));
}

void HDGOperator::ValidateInputs() const
{
   if (mesh_.Dimension() != 2 || mesh_.SpaceDimension() != 2 ||
       mesh_.GetNE() <= 0)
   {
      throw std::runtime_error("HDG operator requires a nonempty 2D mesh");
   }
   if (options_.order < 1)
   {
      throw std::runtime_error("HDG operator requires order >= 1");
   }
   if (physics_.Dim() != 2)
   {
      throw std::runtime_error("physics model must be two-dimensional");
   }
   if (ncu_ < 1 || ncu_ > kMaxComponents ||
       physics_.NumStateEntries() > kMaxStateEntries ||
       physics_.NumFluxEntries() > kMaxFluxEntries)
   {
      throw std::runtime_error(
         "physics component count exceeds operator scratch capacity");
   }
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      GeometryFaceCount(mesh_.GetElementGeometry(element));
   }
   if (orientations_.size() !=
          static_cast<std::size_t>(mesh_.GetGlobalNE()) &&
       !orientations_.empty())
   {
      throw std::runtime_error("missing element orientation maps");
   }
   if (vdg_)
   {
      if (options_.order != 4)
      {
         throw std::runtime_error(
            "Exasim vdg artificial viscosity requires order 4");
      }
      for (int element = 0; element < mesh_.GetNE(); ++element)
      {
         if (mesh_.GetElementGeometry(element) != mfem::Geometry::SQUARE)
         {
            throw std::runtime_error(
               "Exasim vdg artificial viscosity requires an"
               " all-quadrilateral mesh");
         }
      }
      if (orientations_.size() !=
          static_cast<std::size_t>(mesh_.GetGlobalNE()))
      {
         throw std::runtime_error(
            "Exasim vdg artificial viscosity requires orientation maps");
      }
      if (vdg_->nnode != 25 || vdg_->ncomp != 2 ||
          vdg_->nelem != mesh_.GetGlobalNE())
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
   for (int boundary = 0; boundary < mesh_.GetNBE(); ++boundary)
   {
      const int attribute = mesh_.GetBdrAttribute(boundary);
      const int index = attribute - 1;
      if (index < 0 ||
          index >= static_cast<int>(attr_to_bcid_.size()) ||
          attr_to_bcid_[static_cast<std::size_t>(index)] < 0 ||
          attr_to_bcid_[static_cast<std::size_t>(index)] >=
             physics_.NumBoundaryConditions())
      {
         throw std::runtime_error(
            "boundary attribute " + std::to_string(attribute) +
            " has no registered boundary condition");
      }
   }
}

void HDGOperator::BuildReferenceTables()
{
   const int ne = mesh_.GetNE();
   const int volume_rule_order =
      2 * options_.order + options_.quadrature_increment;

   element_tables_.resize(static_cast<std::size_t>(ne), nullptr);
   elem_u_offset_.assign(static_cast<std::size_t>(ne) + 1, 0);
   if (global_element_.empty())
   {
      // Identity mapping: the local mesh IS the serial mesh. Data indexed
      // in the serial ordering (vdg/udg) is only valid on one rank then.
      if (vdg_ && mesh_.GetNRanks() > 1)
      {
         throw std::runtime_error(
            "Exasim data at np>1 requires serial_element_ids");
      }
      global_element_.resize(static_cast<std::size_t>(ne));
      for (int element = 0; element < ne; ++element)
      {
         global_element_[static_cast<std::size_t>(element)] = element;
      }
   }
   else if (global_element_.size() != static_cast<std::size_t>(ne))
   {
      throw std::runtime_error(
         "serial_element_ids does not match the local element count");
   }
   for (int element = 0; element < ne; ++element)
   {
      const mfem::Geometry::Type geometry =
         mesh_.GetElementGeometry(element);
      auto inserted = geometry_tables_.emplace(geometry, GeometryTables());
      GeometryTables &tables = inserted.first->second;
      if (inserted.second)
      {
         tables.fe = volume_fes_.GetFE(element);
         tables.rule = &mfem::IntRules.Get(geometry, volume_rule_order);
         tables.ndof = tables.fe->GetDof();
         tables.nq = tables.rule->GetNPoints();
         tables.nfaces = GeometryFaceCount(geometry);
         tables.shape.resize(
            static_cast<std::size_t>(tables.nq * tables.ndof));
         mfem::Vector shape(tables.ndof);
         for (int q = 0; q < tables.nq; ++q)
         {
            tables.fe->CalcShape(tables.rule->IntPoint(q), shape);
            for (int dof = 0; dof < tables.ndof; ++dof)
            {
               tables.shape[static_cast<std::size_t>(
                  dof + tables.ndof * q)] = shape[dof];
            }
         }
      }
      element_tables_[static_cast<std::size_t>(element)] = &tables;
      elem_u_offset_[static_cast<std::size_t>(element) + 1] =
         elem_u_offset_[static_cast<std::size_t>(element)] +
         ncu_ * tables.ndof;
   }
   total_u_size_ = elem_u_offset_[static_cast<std::size_t>(ne)];

   const int nqf = face_rule_->GetNPoints();
   trace_shape_.resize(static_cast<std::size_t>(nqf * nfdof_));
   mfem::Vector face_shape(nfdof_);
   for (int q = 0; q < nqf; ++q)
   {
      trace_fe_->CalcShape(face_rule_->IntPoint(q), face_shape);
      for (int dof = 0; dof < nfdof_; ++dof)
      {
         trace_shape_[static_cast<std::size_t>(dof + nfdof_ * q)] =
            face_shape[dof];
      }
   }
}

void HDGOperator::BuildGeometryAndAVTables()
{
   const int ne = mesh_.GetNE();
   const int nqf = face_rule_->GetNPoints();
   element_geometry_.assign(static_cast<std::size_t>(ne),
                            ElementGeometry());

   mfem::Vector weighted_normal(2);
   mfem::Vector physical(2);
   mfem::Array<int> faces, orientations;
   for (int element = 0; element < ne; ++element)
   {
      const GeometryTables &tables =
         *element_tables_[static_cast<std::size_t>(element)];
      ElementGeometry &eg =
         element_geometry_[static_cast<std::size_t>(element)];
      const int ndof = tables.ndof;
      const int nq = tables.nq;
      eg.det_j.resize(static_cast<std::size_t>(nq));
      eg.coords.resize(static_cast<std::size_t>(2 * nq));
      eg.phys_dshape.resize(static_cast<std::size_t>(2 * ndof * nq));
      eg.av.resize(static_cast<std::size_t>(nq));

      mfem::DenseMatrix physical_dshape(ndof, 2);
      mfem::Vector element_shape(ndof);
      mfem::ElementTransformation *transformation =
         mesh_.GetElementTransformation(element);
      for (int q = 0; q < nq; ++q)
      {
         const mfem::IntegrationPoint &point = tables.rule->IntPoint(q);
         transformation->SetIntPoint(&point);
         eg.det_j[static_cast<std::size_t>(q)] = transformation->Weight();
         if (!(eg.det_j[static_cast<std::size_t>(q)] > 0.0) ||
             !std::isfinite(eg.det_j[static_cast<std::size_t>(q)]))
         {
            throw std::runtime_error(
               "non-positive/non-finite geometry Jacobian");
         }
         tables.fe->CalcPhysDShape(*transformation, physical_dshape);
         for (int dof = 0; dof < ndof; ++dof)
         {
            for (int direction = 0; direction < 2; ++direction)
            {
               eg.phys_dshape[static_cast<std::size_t>(
                  direction + 2 * (dof + ndof * q))] =
                  physical_dshape(dof, direction);
            }
         }
         transformation->Transform(point, physical);
         eg.coords[static_cast<std::size_t>(2 * q)] = physical[0];
         eg.coords[static_cast<std::size_t>(2 * q + 1)] = physical[1];
         if (vdg_)
         {
            const int global =
               global_element_[static_cast<std::size_t>(element)];
            const auto exasim_point =
               orientations_[static_cast<std::size_t>(global)]
                  .MfemToExasim(point.x, point.y);
            eg.av[static_cast<std::size_t>(q)] = EvaluateTensorQ4(
               *vdg_, 0, global, exasim_point[0], exasim_point[1]);
         }
         else
         {
            eg.av[static_cast<std::size_t>(q)] =
               artificial_viscosity_(physical);
         }
      }

      mesh_.GetElementEdges(element, faces, orientations);
      const int nfaces = tables.nfaces;
      if (faces.Size() != nfaces)
      {
         throw std::runtime_error(
            "element face count does not match its geometry");
      }
      eg.face_normals.resize(static_cast<std::size_t>(2 * nqf * nfaces));
      eg.face_coords.resize(static_cast<std::size_t>(2 * nqf * nfaces));
      eg.face_element_ips.resize(
         static_cast<std::size_t>(2 * nqf * nfaces));
      eg.face_av.resize(static_cast<std::size_t>(nqf * nfaces));
      eg.face_shape.resize(
         static_cast<std::size_t>(ndof * nqf * nfaces));
      for (int local_face = 0; local_face < nfaces; ++local_face)
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
            tables.fe->CalcShape(element_point, element_shape);
            mfem::CalcOrtho(
               face_transformations->Jacobian(), weighted_normal);
            if (side_two) { weighted_normal *= -1.0; }

            const int slot = q + nqf * local_face;
            eg.face_element_ips[static_cast<std::size_t>(2 * slot)] =
               element_point.x;
            eg.face_element_ips[static_cast<std::size_t>(2 * slot + 1)] =
               element_point.y;
            for (int direction = 0; direction < 2; ++direction)
            {
               eg.face_normals[static_cast<std::size_t>(
                  direction + 2 * slot)] = weighted_normal[direction];
            }
            for (int dof = 0; dof < ndof; ++dof)
            {
               eg.face_shape[static_cast<std::size_t>(
                  dof + ndof * slot)] = element_shape[dof];
            }
            // GetFaceElementTransformations reuses MFEM's transformation
            // storage, so the element transformation pointer from before
            // that call is no longer safe here.
            face_transformations->Face->Transform(face_point, physical);
            eg.face_coords[static_cast<std::size_t>(2 * slot)] =
               physical[0];
            eg.face_coords[static_cast<std::size_t>(2 * slot + 1)] =
               physical[1];
            if (vdg_)
            {
               const int global =
                  global_element_[static_cast<std::size_t>(element)];
               const auto exasim_point =
                  orientations_[static_cast<std::size_t>(global)]
                     .MfemToExasim(element_point.x, element_point.y);
               eg.face_av[static_cast<std::size_t>(slot)] =
                  EvaluateTensorQ4(*vdg_, 0, global,
                                   exasim_point[0], exasim_point[1]);
            }
            else
            {
               eg.face_av[static_cast<std::size_t>(slot)] =
                  artificial_viscosity_(physical);
            }
         }
      }
   }
}

void HDGOperator::RetabulateArtificialViscosity()
{
   if (vdg_ || !artificial_viscosity_)
   {
      throw std::runtime_error(
         "analytic AV retabulation requires an analytic function");
   }
   mfem::Vector physical(2);
   for (ElementGeometry &eg : element_geometry_)
   {
      for (std::size_t q = 0; q < eg.av.size(); ++q)
      {
         physical[0] = eg.coords[2 * q];
         physical[1] = eg.coords[2 * q + 1];
         eg.av[q] = artificial_viscosity_(physical);
      }
      for (std::size_t slot = 0; slot < eg.face_av.size(); ++slot)
      {
         physical[0] = eg.face_coords[2 * slot];
         physical[1] = eg.face_coords[2 * slot + 1];
         eg.face_av[slot] = artificial_viscosity_(physical);
      }
   }
}

void HDGOperator::BuildGradientElimination()
{
   const int ne = mesh_.GetNE();
   const int nqf = face_rule_->GetNPoints();
   mass_.reserve(static_cast<std::size_t>(ne));
   for (int direction = 0; direction < 2; ++direction)
   {
      c_[direction].reserve(static_cast<std::size_t>(ne));
      e_[direction].reserve(static_cast<std::size_t>(ne));
   }

   mfem::Array<int> faces, face_orientations;

   for (int element = 0; element < ne; ++element)
   {
      const GeometryTables &tables =
         *element_tables_[static_cast<std::size_t>(element)];
      const ElementGeometry &eg =
         element_geometry_[static_cast<std::size_t>(element)];
      const int ndof = tables.ndof;
      const int nfaces = tables.nfaces;
      const int element_trace_dofs = nfaces * nfdof_;

      mfem::Vector shape(ndof);
      mfem::Vector face_shape(nfdof_);
      mfem::DenseMatrix physical_dshape(ndof, 2);
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
      for (int q = 0; q < tables.nq; ++q)
      {
         const mfem::IntegrationPoint &point = tables.rule->IntPoint(q);
         transformation->SetIntPoint(&point);
         tables.fe->CalcShape(point, shape);
         tables.fe->CalcPhysDShape(*transformation, physical_dshape);
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
      for (int local_face = 0; local_face < nfaces; ++local_face)
      {
         for (int q = 0; q < nqf; ++q)
         {
            const mfem::IntegrationPoint &face_point =
               face_rule_->IntPoint(q);
            trace_fe_->CalcShape(face_point, face_shape);
            const int slot = q + nqf * local_face;
            for (int test = 0; test < ndof; ++test)
            {
               const double element_shape =
                  eg.face_shape[static_cast<std::size_t>(
                     test + ndof * slot)];
               for (int trace_dof = 0; trace_dof < nfdof_; ++trace_dof)
               {
                  const int column = local_face * nfdof_ + trace_dof;
                  for (int direction = 0; direction < 2; ++direction)
                  {
                     boundary[direction](test, column) +=
                        face_point.weight * element_shape *
                        face_shape[trace_dof] *
                        eg.face_normals[static_cast<std::size_t>(
                           direction + 2 * slot)];
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
         // q = +grad(u): negate the Exasim-convention elimination matrices
         // so q = C u - E uhat yields the physical gradient.
         c_[direction].back() *= -1.0;
         e_[direction].back() *= -1.0;
      }
   }
}

void HDGOperator::BuildBoundaryFaceMap()
{
   face_bc_id_.assign(
      static_cast<std::size_t>(mesh_.GetNumFaces()), -1);
   for (int boundary = 0; boundary < mesh_.GetNBE(); ++boundary)
   {
      const int face = mesh_.GetBdrElementFaceIndex(boundary);
      if (face < 0 || face >= mesh_.GetNumFaces())
      {
         throw std::runtime_error("invalid boundary face index");
      }
      face_bc_id_[static_cast<std::size_t>(face)] =
         attr_to_bcid_[
            static_cast<std::size_t>(
               mesh_.GetBdrAttribute(boundary) - 1)];
   }
}

int HDGOperator::TraceVSize() const
{
   return trace_fes_.GetVSize();
}

int HDGOperator::TraceTrueVSize() const
{
   return trace_fes_.GetTrueVSize();
}

int HDGOperator::Elements() const
{
   return mesh_.GetNE();
}

int HDGOperator::ElementDofs(int element) const
{
   return element_tables_[static_cast<std::size_t>(element)]->ndof;
}

const mfem::IntegrationRule &
HDGOperator::VolumeRule(int element) const
{
   return *element_tables_[static_cast<std::size_t>(element)]->rule;
}

const mfem::IntegrationRule &HDGOperator::FaceRule() const
{
   return *face_rule_;
}

const mfem::FiniteElementSpace &HDGOperator::VolumeSpace() const
{
   return volume_fes_;
}

const mfem::FiniteElementSpace &HDGOperator::TraceSpace() const
{
   return trace_fes_;
}

const mfem::DenseMatrix &
HDGOperator::C(int element, int direction) const
{
   if (direction < 0 || direction > 1)
   {
      throw std::out_of_range("C direction must be 0 or 1");
   }
   return c_[direction].at(static_cast<std::size_t>(element));
}

const mfem::DenseMatrix &
HDGOperator::E(int element, int direction) const
{
   if (direction < 0 || direction > 1)
   {
      throw std::out_of_range("E direction must be 0 or 1");
   }
   return e_[direction].at(static_cast<std::size_t>(element));
}

double HDGOperator::MinimumDetJ() const
{
   double minimum = std::numeric_limits<double>::infinity();
   for (const ElementGeometry &eg : element_geometry_)
   {
      for (double value : eg.det_j)
      {
         minimum = std::min(minimum, value);
      }
   }
   return minimum;
}

double HDGOperator::MaximumAbsAV() const
{
   double maximum = 0.0;
   for (const ElementGeometry &eg : element_geometry_)
   {
      for (double value : eg.av)
      {
         maximum = std::max(maximum, std::abs(value));
      }
      for (double value : eg.face_av)
      {
         maximum = std::max(maximum, std::abs(value));
      }
   }
   return maximum;
}

void HDGOperator::SetArtificialViscosityField(
   const mfem::GridFunction &av)
{
   if (!av.FESpace() || av.FESpace()->GetMesh()->GetNE() != mesh_.GetNE())
   {
      throw std::runtime_error(
         "artificial-viscosity field lives on a different mesh");
   }
   vdg_ = nullptr;
   mfem::IntegrationPoint ip;
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      const GeometryTables &tables =
         *element_tables_[static_cast<std::size_t>(element)];
      ElementGeometry &eg =
         element_geometry_[static_cast<std::size_t>(element)];
      for (int q = 0; q < tables.nq; ++q)
      {
         eg.av[static_cast<std::size_t>(q)] =
            av.GetValue(element, tables.rule->IntPoint(q));
      }
      for (std::size_t slot = 0; slot < eg.face_av.size(); ++slot)
      {
         ip.Set2(eg.face_element_ips[2 * slot],
                 eg.face_element_ips[2 * slot + 1]);
         eg.face_av[slot] = av.GetValue(element, ip);
      }
   }
}

void HDGOperator::SetArtificialViscosity(
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

HDGState HDGOperator::NewState() const
{
   HDGState solution;
   solution.u.SetSize(total_u_size_);
   solution.q.SetSize(2 * total_u_size_);
   solution.uhat.SetSize(trace_fes_.GetVSize());
   solution.u = 0.0;
   solution.q = 0.0;
   solution.uhat = 0.0;
   return solution;
}

void HDGOperator::SetConstantState(
   const double *state, HDGState &solution) const
{
   solution = NewState();
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      const int ndof = ElementDofs(element);
      for (int component = 0; component < ncu_; ++component)
      {
         for (int dof = 0; dof < ndof; ++dof)
         {
            solution.u[UIdx(element, component, dof)] = state[component];
         }
      }
   }
   for (int face = 0; face < mesh_.GetNumFaces(); ++face)
   {
      mfem::Array<int> vdofs;
      trace_fes_.GetFaceVDofs(face, vdofs);
      for (int component = 0; component < ncu_; ++component)
      {
         for (int dof = 0; dof < nfdof_; ++dof)
         {
            solution.uhat[vdofs[FaceVDofListIndex(component, dof)]] =
               state[component];
         }
      }
   }
   RecomputeGradient(solution);
}

void HDGOperator::ProjectState(
   const StateFunction &function, HDGState &solution) const
{
   if (!function)
   {
      throw std::runtime_error("empty state projection function");
   }
   solution = NewState();
   mfem::Vector physical(2);
   std::array<double, kMaxComponents> value;
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      const GeometryTables &tables =
         *element_tables_[static_cast<std::size_t>(element)];
      const mfem::IntegrationRule &nodes = tables.fe->GetNodes();
      mfem::ElementTransformation *transformation =
         mesh_.GetElementTransformation(element);
      for (int dof = 0; dof < tables.ndof; ++dof)
      {
         transformation->Transform(nodes.IntPoint(dof), physical);
         function(physical, value.data());
         for (int component = 0; component < ncu_; ++component)
         {
            solution.u[UIdx(element, component, dof)] = value[component];
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
      for (int dof = 0; dof < nfdof_; ++dof)
      {
         transformation->Face->Transform(
            face_nodes.IntPoint(dof), physical);
         function(physical, value.data());
         for (int component = 0; component < ncu_; ++component)
         {
            solution.uhat[vdofs[FaceVDofListIndex(component, dof)]] =
               value[component];
         }
      }
   }
   RecomputeGradient(solution);
}

void HDGOperator::InitializeTraceFromInterior(
   HDGState &solution) const
{
   if (solution.u.Size() != total_u_size_)
   {
      throw std::runtime_error("interior state has the wrong size");
   }
   solution.uhat.SetSize(trace_fes_.GetVSize());
   solution.uhat = 0.0;
   const mfem::IntegrationRule &face_nodes = trace_fe_->GetNodes();
   for (int face = 0; face < mesh_.GetNumFaces(); ++face)
   {
      mfem::FaceElementTransformations *transformation =
         mesh_.GetFaceElementTransformations(face, 31);
      const bool use_first =
         transformation->Elem2No < 0 ||
         transformation->Elem1No < transformation->Elem2No;
      const int element = use_first ?
         transformation->Elem1No : transformation->Elem2No;
      const int ndof = ElementDofs(element);
      mfem::Vector shape(ndof);
      mfem::Array<int> vdofs;
      trace_fes_.GetFaceVDofs(face, vdofs);
      for (int dof = 0; dof < nfdof_; ++dof)
      {
         transformation->SetAllIntPoints(&face_nodes.IntPoint(dof));
         element_tables_[static_cast<std::size_t>(element)]->fe->CalcShape(
            use_first ? transformation->GetElement1IntPoint() :
                        transformation->GetElement2IntPoint(),
            shape);
         for (int component = 0; component < ncu_; ++component)
         {
            double value = 0.0;
            for (int trial = 0; trial < ndof; ++trial)
            {
               value += shape[trial] *
                        solution.u[UIdx(element, component, trial)];
            }
            solution.uhat[vdofs[FaceVDofListIndex(component, dof)]] =
               value;
         }
      }
   }
   // Shared processor-boundary faces were initialized from each rank's own
   // element side; average the sides through the prolongation so every
   // rank holds the same trace values (exact identity at np=1).
   if (mesh_.GetNRanks() > 1)
   {
      const mfem::HypreParMatrix *prolongation =
         trace_fes_.Dof_TrueDof_Matrix();
      mfem::Vector ones(trace_fes_.GetVSize());
      ones = 1.0;
      mfem::Vector multiplicity(trace_fes_.GetTrueVSize());
      prolongation->MultTranspose(ones, multiplicity);
      mfem::Vector true_values(trace_fes_.GetTrueVSize());
      prolongation->MultTranspose(solution.uhat, true_values);
      for (int i = 0; i < true_values.Size(); ++i)
      {
         true_values[i] /= multiplicity[i];
      }
      prolongation->Mult(true_values, solution.uhat);
   }
   RecomputeGradient(solution);
}

void HDGOperator::LoadExasimVolumeState(
   const ExasimArray &udg, bool load_gradient, HDGState &solution) const
{
   if (options_.order != 4 ||
       geometry_tables_.size() != 1 ||
       geometry_tables_.begin()->first != mfem::Geometry::SQUARE)
   {
      throw std::runtime_error(
         "Exasim volume states require an all-quadrilateral order-4 run");
   }
   const int required_components = load_gradient ? 3 * ncu_ : ncu_;
   if (udg.nnode != 25 || udg.ncomp < required_components ||
       udg.nelem != mesh_.GetGlobalNE())
   {
      throw std::runtime_error(
         "Exasim volume state has the wrong [25,ncomp,ne] layout");
   }
   solution = NewState();
   const auto target_points =
      FiniteElementNodes(*geometry_tables_.begin()->second.fe);
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      const int global =
         global_element_[static_cast<std::size_t>(element)];
      const TensorBasisTransform transform =
         BuildTensorBasisTransform(
            target_points,
            orientations_[static_cast<std::size_t>(global)]);
      for (int component = 0; component < required_components;
           ++component)
      {
         double source[25], target[25];
         for (int node = 0; node < 25; ++node)
         {
            source[node] = udg(node, component, global);
         }
         transform.ToTarget(source, target);
         if (component < ncu_)
         {
            for (int dof = 0; dof < 25; ++dof)
            {
               solution.u[UIdx(element, component, dof)] = target[dof];
            }
         }
         else
         {
            const int direction = (component - ncu_) / ncu_;
            const int q_component = (component - ncu_) % ncu_;
            for (int dof = 0; dof < 25; ++dof)
            {
               // Exasim udg stores q = -grad(u); this solver stores
               // +grad(u).
               solution.q[
                  QIdx(element, direction, q_component, dof)] =
                     -target[dof];
            }
         }
      }
   }
}

void HDGOperator::EvaluateElementState(
   const HDGState &solution, int element,
   const mfem::IntegrationPoint &point, double *uq) const
{
   if (element < 0 || element >= mesh_.GetNE() ||
       solution.u.Size() != total_u_size_ ||
       solution.q.Size() != 2 * total_u_size_)
   {
      throw std::runtime_error("invalid element state evaluation");
   }
   const int nstate = ncu_ * 3;
   std::fill(uq, uq + nstate, 0.0);
   const GeometryTables &tables =
      *element_tables_[static_cast<std::size_t>(element)];
   mfem::Vector shape(tables.ndof);
   tables.fe->CalcShape(point, shape);
   for (int component = 0; component < ncu_; ++component)
   {
      for (int dof = 0; dof < tables.ndof; ++dof)
      {
         uq[component] += shape[dof] *
            solution.u[UIdx(element, component, dof)];
         for (int direction = 0; direction < 2; ++direction)
         {
            uq[ncu_ + ncu_ * direction + component] += shape[dof] *
               solution.q[QIdx(element, direction, component, dof)];
         }
      }
   }
}

void HDGOperator::EvaluateTraceState(
   const HDGState &solution, int face,
   const mfem::IntegrationPoint &point, double *uhat) const
{
   if (face < 0 || face >= mesh_.GetNumFaces() ||
       solution.uhat.Size() != trace_fes_.GetVSize())
   {
      throw std::runtime_error("invalid trace state evaluation");
   }
   std::fill(uhat, uhat + ncu_, 0.0);
   mfem::Vector shape(nfdof_);
   trace_fe_->CalcShape(point, shape);
   mfem::Array<int> vdofs;
   trace_fes_.GetFaceVDofs(face, vdofs);
   for (int component = 0; component < ncu_; ++component)
   {
      for (int dof = 0; dof < nfdof_; ++dof)
      {
         uhat[component] += shape[dof] *
            solution.uhat[
               vdofs[FaceVDofListIndex(component, dof)]];
      }
   }
}

void HDGOperator::GetElementTraceVDofs(
   int element, mfem::Array<int> &vdofs) const
{
   const int nfaces =
      element_tables_[static_cast<std::size_t>(element)]->nfaces;
   vdofs.SetSize(ncu_ * nfdof_ * nfaces);
   mfem::Array<int> faces, orientations, face_vdofs;
   mesh_.GetElementEdges(element, faces, orientations);
   for (int local_face = 0; local_face < nfaces; ++local_face)
   {
      trace_fes_.GetFaceVDofs(faces[local_face], face_vdofs);
      for (int component = 0; component < ncu_; ++component)
      {
         for (int dof = 0; dof < nfdof_; ++dof)
         {
            vdofs[LocalTraceIndex(local_face, component, dof)] =
               face_vdofs[FaceVDofListIndex(component, dof)];
         }
      }
   }
}

void HDGOperator::GatherElementTrace(
   const HDGState &solution, int element,
   mfem::Vector &local_trace) const
{
   mfem::Array<int> vdofs;
   GetElementTraceVDofs(element, vdofs);
   local_trace.SetSize(vdofs.Size());
   solution.uhat.GetSubVector(vdofs, local_trace);
}

void HDGOperator::RecomputeGradient(HDGState &solution) const
{
   if (solution.u.Size() != total_u_size_ ||
       solution.uhat.Size() != trace_fes_.GetVSize())
   {
      throw std::runtime_error(
         "cannot recompute q from an incorrectly sized state");
   }
   solution.q.SetSize(2 * total_u_size_);
   mfem::Vector local_trace;
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      const GeometryTables &tables =
         *element_tables_[static_cast<std::size_t>(element)];
      const int ndof = tables.ndof;
      const int nfaces = tables.nfaces;
      mfem::Vector scalar_u(ndof), scalar_trace(nfaces * nfdof_);
      mfem::Vector result(ndof), boundary(ndof);
      GatherElementTrace(solution, element, local_trace);
      for (int component = 0; component < ncu_; ++component)
      {
         for (int dof = 0; dof < ndof; ++dof)
         {
            scalar_u[dof] = solution.u[UIdx(element, component, dof)];
         }
         for (int local_face = 0; local_face < nfaces; ++local_face)
         {
            for (int dof = 0; dof < nfdof_; ++dof)
            {
               scalar_trace[local_face * nfdof_ + dof] =
                  local_trace[LocalTraceIndex(local_face, component,
                                              dof)];
            }
         }
         for (int direction = 0; direction < 2; ++direction)
         {
            c_[direction][element].Mult(scalar_u, result);
            e_[direction][element].Mult(scalar_trace, boundary);
            result -= boundary;
            for (int dof = 0; dof < ndof; ++dof)
            {
               solution.q[QIdx(element, direction, component, dof)] =
                  result[dof];
            }
         }
      }
   }
}

void HDGOperator::SetManufacturedSource(StateFunction source)
{
   manufactured_source_ = std::move(source);
}

void HDGOperator::ClearManufacturedSource()
{
   manufactured_source_ = StateFunction();
}

void HDGOperator::SetDirichletStateOverride(StateFunction state)
{
   dirichlet_state_override_ = std::move(state);
}

void HDGOperator::ClearDirichletStateOverride()
{
   dirichlet_state_override_ = StateFunction();
}

void HDGOperator::AssembleElement(
   const HDGState &solution, int element, bool build_jacobian,
   mfem::Vector &ru, mfem::Vector &rh,
   mfem::DenseMatrix *a, mfem::DenseMatrix *b0,
   mfem::DenseMatrix *b1, mfem::DenseMatrix *f,
   mfem::DenseMatrix *k, mfem::DenseMatrix *g0,
   mfem::DenseMatrix *g1, mfem::DenseMatrix *h) const
{
   const GeometryTables &tables =
      *element_tables_[static_cast<std::size_t>(element)];
   const ElementGeometry &eg =
      element_geometry_[static_cast<std::size_t>(element)];
   const int ndof = tables.ndof;
   const int nfaces = tables.nfaces;
   const int nu = ncu_ * ndof;
   const int nt = ncu_ * nfdof_ * nfaces;
   const int nflux = physics_.NumFluxEntries();
   const int nstate = physics_.NumStateEntries();
   const double tau = options_.tau;

   ru.SetSize(nu);
   rh.SetSize(nt);
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
   double uq[kMaxStateEntries];
   double uhat[kMaxComponents];
   double trace_uq[kMaxStateEntries];
   double flux[kMaxFluxEntries];
   double flux_jacobian[kMaxFluxEntries * kMaxStateEntries];
   double fhat[kMaxComponents];
   double fb[kMaxComponents];
   double fb_uq[kMaxComponents * kMaxStateEntries];
   double fb_uh[kMaxComponents * kMaxComponents];

   const int nq = tables.nq;
   for (int qpoint = 0; qpoint < nq; ++qpoint)
   {
      std::fill(uq, uq + nstate, 0.0);
      for (int component = 0; component < ncu_; ++component)
      {
         for (int dof = 0; dof < ndof; ++dof)
         {
            const double shape =
               tables.shape[static_cast<std::size_t>(
                  dof + ndof * qpoint)];
            uq[component] += shape *
               solution.u[UIdx(element, component, dof)];
            for (int direction = 0; direction < 2; ++direction)
            {
               uq[ncu_ + ncu_ * direction + component] += shape *
                  solution.q[QIdx(element, direction, component, dof)];
            }
         }
      }
      physics_.Flux(uq, eg.av[static_cast<std::size_t>(qpoint)], flux,
                    build_jacobian ? flux_jacobian : nullptr);
      const double weight =
         tables.rule->IntPoint(qpoint).weight *
         eg.det_j[static_cast<std::size_t>(qpoint)];
      for (int output = 0; output < ncu_; ++output)
      {
         for (int test = 0; test < ndof; ++test)
         {
            const int row = LocalUIndex(ndof, output, test);
            for (int direction = 0; direction < 2; ++direction)
            {
               const double gradient =
                  eg.phys_dshape[static_cast<std::size_t>(
                     direction + 2 * (test + ndof * qpoint))];
               ru[row] += weight * gradient *
                          flux[output + ncu_ * direction];
               if (build_jacobian)
               {
                  const int flux_entry = output + ncu_ * direction;
                  for (int input = 0; input < ncu_; ++input)
                  {
                     for (int trial = 0; trial < ndof; ++trial)
                     {
                        const double trial_shape =
                           tables.shape[static_cast<std::size_t>(
                              trial + ndof * qpoint)];
                        (*a)(row, LocalUIndex(ndof, input, trial)) +=
                           weight * gradient * trial_shape *
                           flux_jacobian[flux_entry + nflux * input];
                        (*b0)(row, LocalUIndex(ndof, input, trial)) +=
                           weight * gradient * trial_shape *
                           flux_jacobian[
                              flux_entry + nflux * (ncu_ + input)];
                        (*b1)(row, LocalUIndex(ndof, input, trial)) +=
                           weight * gradient * trial_shape *
                           flux_jacobian[
                              flux_entry + nflux * (2 * ncu_ + input)];
                     }
                  }
               }
            }
         }
      }

      if (manufactured_source_)
      {
         mfem::Vector physical(
            const_cast<double *>(
               &eg.coords[static_cast<std::size_t>(2 * qpoint)]), 2);
         double source[kMaxComponents];
         manufactured_source_(physical, source);
         for (int output = 0; output < ncu_; ++output)
         {
            for (int test = 0; test < ndof; ++test)
            {
               ru[LocalUIndex(ndof, output, test)] +=
                  weight *
                  tables.shape[static_cast<std::size_t>(
                     test + ndof * qpoint)] *
                  source[output];
            }
         }
      }
   }

   mfem::Array<int> faces, face_orientations;
   mesh_.GetElementEdges(element, faces, face_orientations);
   const int nqf = face_rule_->GetNPoints();
   for (int local_face = 0; local_face < nfaces; ++local_face)
   {
      const int face = faces[local_face];
      const int face_bc_id =
         face_bc_id_[static_cast<std::size_t>(face)];
      for (int qpoint = 0; qpoint < nqf; ++qpoint)
      {
         const int slot = qpoint + nqf * local_face;
         const double nx_weighted =
            eg.face_normals[static_cast<std::size_t>(2 * slot)];
         const double ny_weighted =
            eg.face_normals[static_cast<std::size_t>(2 * slot + 1)];
         const double surface_jacobian =
            std::hypot(nx_weighted, ny_weighted);
         const double normal[2] =
         {
            nx_weighted / surface_jacobian,
            ny_weighted / surface_jacobian
         };
         const double weight =
            face_rule_->IntPoint(qpoint).weight * surface_jacobian;
         std::fill(uq, uq + nstate, 0.0);
         std::fill(uhat, uhat + ncu_, 0.0);
         for (int component = 0; component < ncu_; ++component)
         {
            for (int dof = 0; dof < ndof; ++dof)
            {
               const double shape =
                  eg.face_shape[static_cast<std::size_t>(
                     dof + ndof * slot)];
               uq[component] += shape *
                  solution.u[UIdx(element, component, dof)];
               for (int direction = 0; direction < 2; ++direction)
               {
                  uq[ncu_ + ncu_ * direction + component] += shape *
                     solution.q[QIdx(element, direction, component,
                                     dof)];
               }
            }
            for (int dof = 0; dof < nfdof_; ++dof)
            {
               uhat[component] +=
                  trace_shape_[static_cast<std::size_t>(
                     dof + nfdof_ * qpoint)] *
                  local_trace[LocalTraceIndex(
                     local_face, component, dof)];
            }
         }

         std::copy(uq, uq + nstate, trace_uq);
         std::copy(uhat, uhat + ncu_, trace_uq);
         physics_.Flux(trace_uq,
                       eg.face_av[static_cast<std::size_t>(slot)], flux,
                       build_jacobian ? flux_jacobian : nullptr);
         for (int output = 0; output < ncu_; ++output)
         {
            const double normal_flux =
               flux[output] * normal[0] +
               flux[output + ncu_] * normal[1];
            fhat[output] =
               normal_flux + tau * (uq[output] - uhat[output]);
            for (int test = 0; test < ndof; ++test)
            {
               const int row = LocalUIndex(ndof, output, test);
               const double test_shape =
                  eg.face_shape[static_cast<std::size_t>(
                     test + ndof * slot)];
               ru[row] -= weight * test_shape * fhat[output];
               if (build_jacobian)
               {
                  for (int trial = 0; trial < ndof; ++trial)
                  {
                     const double trial_shape =
                        eg.face_shape[static_cast<std::size_t>(
                           trial + ndof * slot)];
                     (*a)(row, LocalUIndex(ndof, output, trial)) -=
                        weight * test_shape * tau * trial_shape;
                     for (int input = 0; input < ncu_; ++input)
                     {
                        const double dflux_q0 =
                           normal[0] * flux_jacobian[
                              output + nflux * (ncu_ + input)] +
                           normal[1] * flux_jacobian[
                              output + ncu_ +
                              nflux * (ncu_ + input)];
                        const double dflux_q1 =
                           normal[0] * flux_jacobian[
                              output + nflux * (2 * ncu_ + input)] +
                           normal[1] * flux_jacobian[
                              output + ncu_ +
                              nflux * (2 * ncu_ + input)];
                        (*b0)(row, LocalUIndex(ndof, input, trial)) -=
                           weight * test_shape * dflux_q0 * trial_shape;
                        (*b1)(row, LocalUIndex(ndof, input, trial)) -=
                           weight * test_shape * dflux_q1 * trial_shape;
                     }
                  }
                  for (int input = 0; input < ncu_; ++input)
                  {
                     const double dflux_u =
                        normal[0] *
                           flux_jacobian[output + nflux * input] +
                        normal[1] *
                           flux_jacobian[output + ncu_ +
                                         nflux * input] -
                        (output == input ? tau : 0.0);
                     for (int trial = 0; trial < nfdof_; ++trial)
                     {
                        (*f)(row, LocalTraceIndex(
                           local_face, input, trial)) -=
                           weight * test_shape * dflux_u *
                           trace_shape_[static_cast<std::size_t>(
                              trial + nfdof_ * qpoint)];
                     }
                  }
               }
            }
         }

         if (face_bc_id < 0)
         {
            for (int output = 0; output < ncu_; ++output)
            {
               for (int test = 0; test < nfdof_; ++test)
               {
                  const int row =
                     LocalTraceIndex(local_face, output, test);
                  const double test_shape =
                     trace_shape_[static_cast<std::size_t>(
                        test + nfdof_ * qpoint)];
                  rh[row] += weight * test_shape * fhat[output];
                  if (build_jacobian)
                  {
                     for (int trial = 0; trial < ndof; ++trial)
                     {
                        const double trial_shape =
                           eg.face_shape[static_cast<std::size_t>(
                              trial + ndof * slot)];
                        (*k)(row, LocalUIndex(ndof, output, trial)) +=
                           weight * test_shape * tau * trial_shape;
                        for (int input = 0; input < ncu_; ++input)
                        {
                           const double dflux_q0 =
                              normal[0] * flux_jacobian[
                                 output + nflux * (ncu_ + input)] +
                              normal[1] * flux_jacobian[
                                 output + ncu_ +
                                 nflux * (ncu_ + input)];
                           const double dflux_q1 =
                              normal[0] * flux_jacobian[
                                 output +
                                 nflux * (2 * ncu_ + input)] +
                              normal[1] * flux_jacobian[
                                 output + ncu_ +
                                 nflux * (2 * ncu_ + input)];
                           (*g0)(row,
                                 LocalUIndex(ndof, input, trial)) +=
                              weight * test_shape * dflux_q0 *
                              trial_shape;
                           (*g1)(row,
                                 LocalUIndex(ndof, input, trial)) +=
                              weight * test_shape * dflux_q1 *
                              trial_shape;
                        }
                     }
                     for (int input = 0; input < ncu_; ++input)
                     {
                        const double dflux_u =
                           normal[0] *
                              flux_jacobian[output + nflux * input] +
                           normal[1] *
                              flux_jacobian[output + ncu_ +
                                            nflux * input] -
                           (output == input ? tau : 0.0);
                        for (int trial = 0; trial < nfdof_; ++trial)
                        {
                           (*h)(row, LocalTraceIndex(
                              local_face, input, trial)) +=
                              weight * test_shape * dflux_u *
                              trace_shape_[static_cast<std::size_t>(
                                 trial + nfdof_ * qpoint)];
                        }
                     }
                  }
               }
            }
         }
         else
         {
            if (dirichlet_state_override_)
            {
               mfem::Vector physical(
                  const_cast<double *>(
                     &eg.face_coords[static_cast<std::size_t>(
                        2 * slot)]), 2);
               double prescribed[kMaxComponents];
               dirichlet_state_override_(physical, prescribed);
               for (int output = 0; output < ncu_; ++output)
               {
                  fb[output] = prescribed[output] - uhat[output];
               }
               if (build_jacobian)
               {
                  std::fill(fb_uq, fb_uq + ncu_ * nstate, 0.0);
                  std::fill(fb_uh, fb_uh + ncu_ * ncu_, 0.0);
                  for (int component = 0; component < ncu_; ++component)
                  {
                     fb_uh[component + ncu_ * component] = -1.0;
                  }
               }
            }
            else
            {
               physics_.BoundaryResidual(
                  face_bc_id, uq, uhat, normal,
                  &eg.face_coords[static_cast<std::size_t>(2 * slot)],
                  fb,
                  build_jacobian ? fb_uq : nullptr,
                  build_jacobian ? fb_uh : nullptr);
            }
            for (int output = 0; output < ncu_; ++output)
            {
               for (int test = 0; test < nfdof_; ++test)
               {
                  const int row =
                     LocalTraceIndex(local_face, output, test);
                  const double test_shape =
                     trace_shape_[static_cast<std::size_t>(
                        test + nfdof_ * qpoint)];
                  rh[row] += weight * test_shape * fb[output];
                  if (build_jacobian)
                  {
                     for (int input = 0; input < ncu_; ++input)
                     {
                        for (int trial = 0; trial < ndof; ++trial)
                        {
                           const double trial_shape =
                              eg.face_shape[static_cast<std::size_t>(
                                 trial + ndof * slot)];
                           (*k)(row,
                                LocalUIndex(ndof, input, trial)) +=
                              weight * test_shape *
                              fb_uq[output + ncu_ * input] *
                              trial_shape;
                           (*g0)(row,
                                 LocalUIndex(ndof, input, trial)) +=
                              weight * test_shape *
                              fb_uq[output + ncu_ * (ncu_ + input)] *
                              trial_shape;
                           (*g1)(row,
                                 LocalUIndex(ndof, input, trial)) +=
                              weight * test_shape *
                              fb_uq[output +
                                    ncu_ * (2 * ncu_ + input)] *
                              trial_shape;
                        }
                        for (int trial = 0; trial < nfdof_; ++trial)
                        {
                           (*h)(row, LocalTraceIndex(
                              local_face, input, trial)) +=
                              weight * test_shape *
                              fb_uh[output + ncu_ * input] *
                              trace_shape_[static_cast<std::size_t>(
                                 trial + nfdof_ * qpoint)];
                        }
                     }
                  }
               }
            }
         }
      }
   }
}

HDGResidualNorms HDGOperator::Assemble(
   const HDGState &solution, bool build_jacobian,
   double pseudo_time_inverse_step)
{
   if (solution.u.Size() != total_u_size_ ||
       solution.q.Size() != 2 * total_u_size_ ||
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

   mfem::Vector ru, rh;
   mfem::DenseMatrix a, b0, b1, f, k, g0, g1, h;
   mfem::Array<int> trace_vdofs;
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      const GeometryTables &tables =
         *element_tables_[static_cast<std::size_t>(element)];
      const int ndof = tables.ndof;
      const int nfaces = tables.nfaces;
      const int nu = ncu_ * ndof;
      const int nt = ncu_ * nfdof_ * nfaces;
      if (build_jacobian)
      {
         a.SetSize(nu, nu);
         b0.SetSize(nu, nu);
         b1.SetSize(nu, nu);
         f.SetSize(nu, nt);
         k.SetSize(nt, nu);
         g0.SetSize(nt, nu);
         g1.SetSize(nt, nu);
         h.SetSize(nt, nt);
      }
      AssembleElement(solution, element, build_jacobian, ru, rh,
                      &a, &b0, &b1, &f, &k, &g0, &g1, &h);
      for (int i = 0; i < nu; ++i)
      {
         volume_residual_[elem_u_offset_[element] + i] = ru[i];
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
         for (int out = 0; out < ncu_; ++out)
         {
            for (int input = 0; input < ncu_; ++input)
            {
               for (int i = 0; i < ndof; ++i)
               {
                  const int row_u = LocalUIndex(ndof, out, i);
                  for (int j = 0; j < ndof; ++j)
                  {
                     const int col_u = LocalUIndex(ndof, input, j);
                     double bac = 0.0;
                     for (int m = 0; m < ndof; ++m)
                     {
                        bac += bd(row_u, LocalUIndex(ndof, input, m)) *
                               cd(m, j);
                     }
                     a(row_u, col_u) += bac;
                  }
                  for (int local_face = 0; local_face < nfaces;
                       ++local_face)
                  {
                     for (int j = 0; j < nfdof_; ++j)
                     {
                        double bae = 0.0;
                        for (int m = 0; m < ndof; ++m)
                        {
                           bae += bd(row_u,
                                     LocalUIndex(ndof, input, m)) *
                                  ed(m, local_face * nfdof_ + j);
                        }
                        f(row_u, LocalTraceIndex(
                           local_face, input, j)) -= bae;
                     }
                  }
               }
               for (int local_face_row = 0;
                    local_face_row < nfaces; ++local_face_row)
               {
                  for (int i = 0; i < nfdof_; ++i)
                  {
                     const int row_h =
                        LocalTraceIndex(local_face_row, out, i);
                     for (int j = 0; j < ndof; ++j)
                     {
                        double gac = 0.0;
                        for (int m = 0; m < ndof; ++m)
                        {
                           gac += gd(row_h,
                                     LocalUIndex(ndof, input, m)) *
                                  cd(m, j);
                        }
                        k(row_h, LocalUIndex(ndof, input, j)) += gac;
                     }
                     for (int local_face_col = 0;
                          local_face_col < nfaces; ++local_face_col)
                     {
                        for (int j = 0; j < nfdof_; ++j)
                        {
                           double gae = 0.0;
                           for (int m = 0; m < ndof; ++m)
                           {
                              gae += gd(row_h,
                                        LocalUIndex(ndof, input, m)) *
                                     ed(m, local_face_col * nfdof_ + j);
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
         for (int component = 0; component < ncu_; ++component)
         {
            for (int i = 0; i < ndof; ++i)
            {
               for (int j = 0; j < ndof; ++j)
               {
                  a(LocalUIndex(ndof, component, i),
                    LocalUIndex(ndof, component, j)) +=
                     pseudo_time_inverse_step * mass_[element](i, j);
               }
            }
         }
      }

      mfem::DenseMatrixInverse a_inverse(a);
      mfem::DenseMatrix inverse_a_f(nu, nt);
      mfem::Vector inverse_a_ru(nu);
      a_inverse.Mult(f, inverse_a_f);
      a_inverse.Mult(ru, inverse_a_ru);

      mfem::DenseMatrix hc(h);
      for (int i = 0; i < nt; ++i)
      {
         for (int j = 0; j < nt; ++j)
         {
            double product = 0.0;
            for (int m = 0; m < nu; ++m)
            {
               product += k(i, m) * inverse_a_f(m, j);
            }
            hc(i, j) -= product;
         }
      }
      mfem::Vector local_condensed_residual(rh);
      for (int i = 0; i < nt; ++i)
      {
         double product = 0.0;
         for (int m = 0; m < nu; ++m)
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

      // skip_zeros=0 is load-bearing: the first pass builds the complete
      // graph.
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

   // True-dof reduction: P^T sums the per-side contributions on shared
   // faces, which is exactly the interior-face flux-continuity stitching.
   MPI_Comm comm = trace_fes_.GetComm();
   const mfem::HypreParMatrix *prolongation =
      trace_fes_.Dof_TrueDof_Matrix();
   trace_true_residual_.SetSize(trace_fes_.GetTrueVSize());
   prolongation->MultTranspose(trace_residual_, trace_true_residual_);
   double volume_sumsq = volume_residual_ * volume_residual_;
   MPI_Allreduce(MPI_IN_PLACE, &volume_sumsq, 1, MPI_DOUBLE, MPI_SUM,
                 comm);
   const double trace_sumsq = mfem::InnerProduct(
      comm, trace_true_residual_, trace_true_residual_);

   if (build_jacobian)
   {
      condensed_true_rhs_.SetSize(trace_fes_.GetTrueVSize());
      prolongation->MultTranspose(condensed_rhs_, condensed_true_rhs_);
      mfem::OperatorHandle block_diagonal(mfem::Operator::Hypre_ParCSR);
      block_diagonal.MakeSquareBlockDiag(
         comm, trace_fes_.GlobalVSize(), trace_fes_.GetDofOffsets(),
         condensed_matrix_.get());
      mfem::OperatorHandle prolongation_handle(
         mfem::Operator::Hypre_ParCSR);
      prolongation_handle.ConvertFrom(
         const_cast<mfem::HypreParMatrix *>(prolongation));
      condensed_true_.Clear();
      condensed_true_.SetType(mfem::Operator::Hypre_ParCSR);
      condensed_true_.MakePtAP(block_diagonal, prolongation_handle);
      recovery_is_current_ = true;
   }
   return {std::sqrt(volume_sumsq), std::sqrt(trace_sumsq)};
}

const mfem::Vector &HDGOperator::VolumeResidual() const
{
   return volume_residual_;
}

const mfem::Vector &HDGOperator::TraceResidual() const
{
   return trace_residual_;
}

const mfem::Vector &HDGOperator::CondensedResidual() const
{
   return condensed_residual_;
}

const mfem::Vector &HDGOperator::CondensedRHS() const
{
   return condensed_rhs_;
}

const mfem::SparseMatrix &HDGOperator::CondensedMatrix() const
{
   if (!condensed_matrix_ || !recovery_is_current_)
   {
      throw std::runtime_error(
         "condensed matrix requested before full assembly");
   }
   return *condensed_matrix_;
}

const mfem::HypreParMatrix &HDGOperator::CondensedParMatrix() const
{
   if (!recovery_is_current_ || !condensed_true_.Ptr())
   {
      throw std::runtime_error(
         "condensed parallel matrix requested before full assembly");
   }
   return *condensed_true_.As<mfem::HypreParMatrix>();
}

const mfem::Vector &HDGOperator::CondensedTrueRHS() const
{
   return condensed_true_rhs_;
}

void HDGOperator::ExpandTraceIncrement(
   const mfem::Vector &true_increment,
   mfem::Vector &local_increment) const
{
   if (true_increment.Size() != trace_fes_.GetTrueVSize())
   {
      throw std::runtime_error(
         "true trace increment has the wrong size");
   }
   local_increment.SetSize(trace_fes_.GetVSize());
   trace_fes_.Dof_TrueDof_Matrix()->Mult(true_increment,
                                         local_increment);
}

void HDGOperator::RecoverIncrement(
   const mfem::Vector &trace_increment,
   mfem::Vector &volume_increment) const
{
   if (!recovery_is_current_ ||
       inverse_a_f_.size() != static_cast<std::size_t>(mesh_.GetNE()))
   {
      throw std::runtime_error(
         "local recovery requested before full assembly");
   }
   if (trace_increment.Size() != trace_fes_.GetVSize())
   {
      throw std::runtime_error("trace increment has the wrong size");
   }
   volume_increment.SetSize(total_u_size_);
   volume_increment = 0.0;
   mfem::Array<int> trace_vdofs;
   mfem::Vector local_trace, product;
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      const int nu = ncu_ * ElementDofs(element);
      GetElementTraceVDofs(element, trace_vdofs);
      local_trace.SetSize(trace_vdofs.Size());
      trace_increment.GetSubVector(trace_vdofs, local_trace);
      product.SetSize(nu);
      inverse_a_f_[element].Mult(local_trace, product);
      for (int i = 0; i < nu; ++i)
      {
         volume_increment[elem_u_offset_[element] + i] =
            -inverse_a_ru_[element][i] - product[i];
      }
   }
}

double HDGOperator::L2Error(
   const HDGState &solution, const StateFunction &exact) const
{
   mfem::Vector physical(2);
   double error_squared = 0.0;
   std::array<double, kMaxComponents> expected;
   std::array<double, kMaxComponents> actual;
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      const GeometryTables &tables =
         *element_tables_[static_cast<std::size_t>(element)];
      const mfem::IntegrationRule &rule = mfem::IntRules.Get(
         mesh_.GetElementGeometry(element), 2 * options_.order + 4);
      mfem::Vector shape(tables.ndof);
      mfem::ElementTransformation *transformation =
         mesh_.GetElementTransformation(element);
      for (int qpoint = 0; qpoint < rule.GetNPoints(); ++qpoint)
      {
         const mfem::IntegrationPoint &point = rule.IntPoint(qpoint);
         transformation->SetIntPoint(&point);
         tables.fe->CalcShape(point, shape);
         transformation->Transform(point, physical);
         exact(physical, expected.data());
         std::fill(actual.begin(), actual.begin() + ncu_, 0.0);
         for (int component = 0; component < ncu_; ++component)
         {
            for (int dof = 0; dof < tables.ndof; ++dof)
            {
               actual[component] += shape[dof] *
                  solution.u[UIdx(element, component, dof)];
            }
            const double difference =
               actual[component] - expected[component];
            error_squared += point.weight * transformation->Weight() *
                             difference * difference;
         }
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &error_squared, 1, MPI_DOUBLE, MPI_SUM,
                 trace_fes_.GetComm());
   return std::sqrt(error_squared);
}

double HDGOperator::MinimumDensity(
   const HDGState &solution) const
{
   double minimum = std::numeric_limits<double>::infinity();
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      const GeometryTables &tables =
         *element_tables_[static_cast<std::size_t>(element)];
      for (int qpoint = 0; qpoint < tables.nq; ++qpoint)
      {
         double density = 0.0;
         for (int dof = 0; dof < tables.ndof; ++dof)
         {
            density += tables.shape[static_cast<std::size_t>(
                          dof + tables.ndof * qpoint)] *
                       solution.u[UIdx(element, 0, dof)];
         }
         minimum = std::min(minimum, density);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MPI_DOUBLE, MPI_MIN,
                 trace_fes_.GetComm());
   return minimum;
}

double HDGOperator::MinimumPressure(
   const HDGState &solution) const
{
   double minimum = std::numeric_limits<double>::infinity();
   std::array<double, kMaxComponents> state;
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      const GeometryTables &tables =
         *element_tables_[static_cast<std::size_t>(element)];
      for (int qpoint = 0; qpoint < tables.nq; ++qpoint)
      {
         std::fill(state.begin(), state.begin() + ncu_, 0.0);
         for (int component = 0; component < ncu_; ++component)
         {
            for (int dof = 0; dof < tables.ndof; ++dof)
            {
               state[component] +=
                  tables.shape[static_cast<std::size_t>(
                     dof + tables.ndof * qpoint)] *
                  solution.u[UIdx(element, component, dof)];
            }
         }
         minimum = std::min(minimum, physics_.Pressure(state.data()));
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MPI_DOUBLE, MPI_MIN,
                 trace_fes_.GetComm());
   return minimum;
}

double HDGOperator::YSymmetryError(
   const HDGState &solution) const
{
   if (ncu_ != 4)
   {
      throw std::runtime_error(
         "y-symmetry diagnostic requires a 4-component model");
   }
   // The analytic and reference meshes are ordered in mirrored
   // circumferential columns. Compare matching physical quadrature points
   // by nearest reflection.
   struct Sample
   {
      double x, y;
      std::array<double, 4> u;
   };
   std::vector<Sample> samples;
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      const GeometryTables &tables =
         *element_tables_[static_cast<std::size_t>(element)];
      const ElementGeometry &eg =
         element_geometry_[static_cast<std::size_t>(element)];
      for (int qpoint = 0; qpoint < tables.nq; ++qpoint)
      {
         Sample sample;
         sample.x = eg.coords[static_cast<std::size_t>(2 * qpoint)];
         sample.y = eg.coords[static_cast<std::size_t>(2 * qpoint + 1)];
         sample.u = {{0.0, 0.0, 0.0, 0.0}};
         for (int component = 0; component < 4; ++component)
         {
            for (int dof = 0; dof < tables.ndof; ++dof)
            {
               sample.u[component] +=
                  tables.shape[static_cast<std::size_t>(
                     dof + tables.ndof * qpoint)] *
                  solution.u[UIdx(element, component, dof)];
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
         maximum = std::max(maximum, error);
      }
   }
   return maximum;
}

double HDGOperator::TraceOrientationError() const
{
   StateFunction linear = [this](const mfem::Vector &x, double *state)
   {
      for (int c = 0; c < ncu_; ++c)
      {
         state[c] = 1.0 + 0.1 * c +
                    (0.2 + 0.07 * c) * x[0] -
                    (0.3 - 0.05 * c) * x[1];
      }
   };
   HDGState projected;
   ProjectState(linear, projected);
   mfem::Vector trace_shape(nfdof_);
   mfem::Array<int> face_vdofs;
   double maximum = 0.0;
   for (int face = 0; face < mesh_.GetNumFaces(); ++face)
   {
      mfem::FaceElementTransformations *transformation =
         mesh_.GetFaceElementTransformations(face, 31);
      if (transformation->Elem2No < 0) { continue; }
      trace_fes_.GetFaceVDofs(face, face_vdofs);
      for (int qpoint = 0; qpoint < face_rule_->GetNPoints(); ++qpoint)
      {
         const mfem::IntegrationPoint &point =
            face_rule_->IntPoint(qpoint);
         transformation->SetAllIntPoints(&point);
         trace_fe_->CalcShape(point, trace_shape);
         for (int side = 0; side < 2; ++side)
         {
            const int element =
               side == 0 ? transformation->Elem1No :
                           transformation->Elem2No;
            const GeometryTables &tables =
               *element_tables_[static_cast<std::size_t>(element)];
            mfem::Vector shape(tables.ndof);
            const mfem::IntegrationPoint &element_point =
               side == 0 ? transformation->GetElement1IntPoint() :
                           transformation->GetElement2IntPoint();
            tables.fe->CalcShape(element_point, shape);
            for (int component = 0; component < ncu_; ++component)
            {
               double volume_value = 0.0;
               double trace_value = 0.0;
               for (int dof = 0; dof < tables.ndof; ++dof)
               {
                  volume_value += shape[dof] *
                     projected.u[UIdx(element, component, dof)];
               }
               for (int dof = 0; dof < nfdof_; ++dof)
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

void HDGOperator::FillConservativeGridFunction(
   const HDGState &solution, mfem::GridFunction &field) const
{
   if (field.FESpace() != &volume_fes_)
   {
      throw std::runtime_error(
         "conservative GridFunction uses the wrong space");
   }
   mfem::Array<int> vdofs;
   mfem::Vector local;
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      const int ndof = ElementDofs(element);
      local.SetSize(ncu_ * ndof);
      volume_fes_.GetElementVDofs(element, vdofs);
      for (int component = 0; component < ncu_; ++component)
      {
         for (int dof = 0; dof < ndof; ++dof)
         {
            local[dof + ndof * component] =
               solution.u[UIdx(element, component, dof)];
         }
      }
      field.SetSubVector(vdofs, local);
   }
}

void HDGOperator::FillPrimitiveGridFunction(
   const HDGState &solution, mfem::GridFunction &field) const
{
   if (field.FESpace() != &volume_fes_)
   {
      throw std::runtime_error(
         "primitive GridFunction uses the wrong space");
   }
   if (static_cast<int>(physics_.OutputNames().size()) != ncu_)
   {
      throw std::runtime_error(
         "primitive output count must match the component count");
   }
   mfem::Array<int> vdofs;
   mfem::Vector local;
   std::array<double, kMaxComponents> state;
   std::array<double, kMaxComponents> primitive;
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      const int ndof = ElementDofs(element);
      local.SetSize(ncu_ * ndof);
      for (int dof = 0; dof < ndof; ++dof)
      {
         for (int component = 0; component < ncu_; ++component)
         {
            state[component] =
               solution.u[UIdx(element, component, dof)];
         }
         physics_.Outputs(state.data(), primitive.data());
         for (int component = 0; component < ncu_; ++component)
         {
            local[dof + ndof * component] = primitive[component];
         }
      }
      volume_fes_.GetElementVDofs(element, vdofs);
      field.SetSubVector(vdofs, local);
   }
}

void HDGOperator::FillArtificialViscosityGridFunction(
   mfem::GridFunction &field) const
{
   mfem::FiniteElementSpace *space = field.FESpace();
   if (!space || space->GetVDim() != 1 ||
       space->GetNE() != mesh_.GetNE())
   {
      throw std::runtime_error(
         "artificial-viscosity GridFunction uses the wrong space");
   }
   mfem::Array<int> dofs;
   mfem::Vector local, physical(2);
   for (int element = 0; element < mesh_.GetNE(); ++element)
   {
      const int ndof = space->GetFE(element)->GetDof();
      local.SetSize(ndof);
      if (vdg_)
      {
         if (ndof != 25)
         {
            throw std::runtime_error(
               "vdg artificial-viscosity output requires 25 dofs");
         }
         const int global =
            global_element_[static_cast<std::size_t>(element)];
         const auto target_points =
            FiniteElementNodes(*space->GetFE(element));
         const TensorBasisTransform transform =
            BuildTensorBasisTransform(
               target_points,
               orientations_[static_cast<std::size_t>(global)]);
         double source[25], target[25];
         for (int node = 0; node < 25; ++node)
         {
            source[node] = (*vdg_)(node, 0, global);
         }
         transform.ToTarget(source, target);
         for (int dof = 0; dof < 25; ++dof) { local[dof] = target[dof]; }
      }
      else
      {
         mfem::ElementTransformation *transformation =
            mesh_.GetElementTransformation(element);
         const mfem::IntegrationRule &nodes =
            space->GetFE(element)->GetNodes();
         for (int dof = 0; dof < ndof; ++dof)
         {
            transformation->Transform(nodes.IntPoint(dof), physical);
            local[dof] = artificial_viscosity_(physical);
         }
      }
      space->GetElementDofs(element, dofs);
      field.SetSubVector(dofs, local);
   }
}

} // namespace hycfd

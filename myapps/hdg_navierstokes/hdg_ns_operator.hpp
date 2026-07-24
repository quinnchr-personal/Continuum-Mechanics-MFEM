#pragma once

#include "exasim_io.hpp"

#include "mfem.hpp"

#include <array>
#include <vector>

namespace hdg_ns
{

// M1 portion of the HDG operator: immutable reference/geometry/AV tables and
// the geometry-only q = C u - E uhat relation. Residual/Jacobian assembly,
// condensation, and recovery intentionally begin in M2.
class HDGNavierStokesOperator
{
public:
   HDGNavierStokesOperator(
      mfem::Mesh &mesh, const ExasimArray &vdg,
      const std::vector<ElementOrientation> &orientations);

   int VolumeQuadraturePoints() const;
   int FaceQuadraturePoints() const;
   int TraceVSize() const;
   int VolumeDofsPerElement() const;
   int TraceDofsPerFace() const;

   const mfem::IntegrationRule &VolumeRule() const;
   const mfem::IntegrationRule &FaceRule() const;

   const mfem::DenseMatrix &C(int element, int direction) const;
   const mfem::DenseMatrix &E(int element, int direction) const;

   double MinimumDetJ() const;
   double MaximumAbsAV() const;

private:
   mfem::Mesh &mesh_;
   const ExasimArray &vdg_;
   const std::vector<ElementOrientation> &orientations_;

   mfem::L2_FECollection volume_fec_;
   mfem::FiniteElementSpace volume_fes_;
   mfem::DG_Interface_FECollection trace_fec_;
   mfem::FiniteElementSpace trace_fes_;

   const mfem::IntegrationRule *volume_rule_ = nullptr;
   const mfem::IntegrationRule *face_rule_ = nullptr;
   const mfem::FiniteElement *volume_fe_ = nullptr;
   const mfem::FiniteElement *trace_fe_ = nullptr;

   std::vector<double> volume_shape_;
   std::vector<double> volume_reference_dshape_;
   std::vector<double> trace_shape_;

   std::vector<double> volume_det_j_;
   std::vector<double> volume_adj_j_;
   std::vector<double> face_weighted_normals_;
   std::vector<double> face_element_shape_;
   std::vector<double> volume_av_;
   std::vector<double> face_av_;

   std::vector<mfem::DenseMatrix> c_[2];
   std::vector<mfem::DenseMatrix> e_[2];

   void ValidateInputs() const;
   void BuildReferenceTables();
   void BuildGeometryAndAVTables();
   void BuildGradientElimination();
};

} // namespace hdg_ns

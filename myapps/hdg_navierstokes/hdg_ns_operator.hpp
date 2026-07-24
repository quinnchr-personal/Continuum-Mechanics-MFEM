#pragma once

#include "exasim_io.hpp"
#include "ns_physics.hpp"

#include "mfem.hpp"

#include <array>
#include <functional>
#include <memory>
#include <vector>

namespace hdg_ns
{

struct HDGState
{
   mfem::Vector u;
   mfem::Vector q;
   mfem::Vector uhat;
};

struct HDGResidualNorms
{
   double volume = 0.0;
   double trace = 0.0;

   double Total() const { return volume + trace; }
};

class HDGNavierStokesOperator
{
public:
   using ScalarFunction =
      std::function<double(const mfem::Vector &)>;
   using StateFunction =
      std::function<void(const mfem::Vector &, double state[4])>;

   // Reference-data constructor: AV is evaluated from vdg component zero.
   HDGNavierStokesOperator(
      mfem::Mesh &mesh, const ExasimArray &vdg,
      const std::vector<ElementOrientation> &orientations,
      const NSParams &params = NSParams(),
      const std::array<int, 3> &boundary_conditions = {{3, 2, 1}});

   // Analytic/frozen-AV constructor used by M2 MMS and sanity problems.
   HDGNavierStokesOperator(
      mfem::Mesh &mesh,
      const std::vector<ElementOrientation> &orientations,
      ScalarFunction artificial_viscosity,
      const NSParams &params = NSParams(),
      const std::array<int, 3> &boundary_conditions = {{3, 2, 1}});

   HDGNavierStokesOperator(
      mfem::Mesh &mesh, ScalarFunction artificial_viscosity,
      const NSParams &params = NSParams(),
      const std::array<int, 3> &boundary_conditions = {{3, 2, 1}});

   int VolumeQuadraturePoints() const;
   int FaceQuadraturePoints() const;
   int TraceVSize() const;
   int VolumeDofsPerElement() const;
   int TraceDofsPerFace() const;
   int Elements() const;

   const mfem::IntegrationRule &VolumeRule() const;
   const mfem::IntegrationRule &FaceRule() const;
   const mfem::FiniteElementSpace &VolumeSpace() const;
   const mfem::FiniteElementSpace &TraceSpace() const;

   const mfem::DenseMatrix &C(int element, int direction) const;
   const mfem::DenseMatrix &E(int element, int direction) const;

   double MinimumDetJ() const;
   double MaximumAbsAV() const;

   HDGState NewState() const;
   void SetConstantState(const double state[4], HDGState &solution) const;
   void ProjectState(const StateFunction &function, HDGState &solution) const;
   void InitializeTraceFromInterior(HDGState &solution) const;
   void RecomputeGradient(HDGState &solution) const;

   // The source is a test-only manufactured-source hook. Physical runs leave
   // it empty, which is exactly source = 0.
   void SetManufacturedSource(StateFunction source);
   void ClearManufacturedSource();
   // Test-only MMS boundary hook: fb = prescribed_state(x) - uhat.
   void SetDirichletStateOverride(StateFunction state);
   void ClearDirichletStateOverride();

   // Full assembly builds all clean J*delta=-R blocks, folds q, condenses,
   // and scatters. Residual-only skips every Jacobian/condensation operation.
   HDGResidualNorms Assemble(
      const HDGState &solution, bool build_jacobian,
      double pseudo_time_inverse_step = 0.0);

   const mfem::Vector &VolumeResidual() const;
   const mfem::Vector &TraceResidual() const;
   const mfem::Vector &CondensedResidual() const;
   const mfem::Vector &CondensedRHS() const;
   const mfem::SparseMatrix &CondensedMatrix() const;

   void RecoverIncrement(const mfem::Vector &trace_increment,
                         mfem::Vector &volume_increment) const;

   double L2Error(const HDGState &solution,
                  const StateFunction &exact) const;
   double MinimumDensity(const HDGState &solution) const;
   double MinimumPressure(const HDGState &solution) const;
   double YSymmetryError(const HDGState &solution) const;
   double TraceOrientationError() const;

   void FillConservativeGridFunction(
      const HDGState &solution, mfem::GridFunction &field) const;
   void FillPrimitiveGridFunction(
      const HDGState &solution, mfem::GridFunction &field) const;

private:
   static constexpr int kComponents = 4;
   static constexpr int kElementDofs = 25;
   static constexpr int kElementUnknowns = 100;
   static constexpr int kFaceDofs = 5;
   static constexpr int kFaceUnknowns = 20;
   static constexpr int kElementTraceUnknowns = 80;

   mfem::Mesh &mesh_;
   const ExasimArray *vdg_ = nullptr;
   std::vector<ElementOrientation> orientations_;
   ScalarFunction artificial_viscosity_;
   NSParams params_;
   std::array<int, 3> boundary_conditions_;
   StateFunction manufactured_source_;
   StateFunction dirichlet_state_override_;

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
   std::vector<double> volume_physical_dshape_;
   std::vector<double> trace_shape_;

   std::vector<double> volume_det_j_;
   std::vector<double> volume_adj_j_;
   std::vector<double> volume_coordinates_;
   std::vector<double> face_weighted_normals_;
   std::vector<double> face_coordinates_;
   std::vector<double> face_element_shape_;
   std::vector<double> volume_av_;
   std::vector<double> face_av_;
   std::vector<int> face_boundary_attribute_;

   std::vector<mfem::DenseMatrix> mass_;
   std::vector<mfem::DenseMatrix> c_[2];
   std::vector<mfem::DenseMatrix> e_[2];

   mfem::Vector volume_residual_;
   mfem::Vector trace_residual_;
   mfem::Vector condensed_residual_;
   mfem::Vector condensed_rhs_;
   std::unique_ptr<mfem::SparseMatrix> condensed_matrix_;
   std::vector<mfem::DenseMatrix> inverse_a_f_;
   std::vector<mfem::Vector> inverse_a_ru_;
   bool recovery_is_current_ = false;

   void Initialize();
   void ValidateInputs() const;
   void BuildReferenceTables();
   void BuildGeometryAndAVTables();
   void BuildGradientElimination();
   void BuildBoundaryFaceMap();

   void GetElementTraceVDofs(int element, mfem::Array<int> &vdofs) const;
   void GatherElementTrace(const HDGState &solution, int element,
                           mfem::Vector &local_trace) const;
   void AssembleElement(
      const HDGState &solution, int element, bool build_jacobian,
      mfem::Vector &ru, mfem::Vector &rh,
      mfem::DenseMatrix *a, mfem::DenseMatrix *b0,
      mfem::DenseMatrix *b1, mfem::DenseMatrix *f,
      mfem::DenseMatrix *k, mfem::DenseMatrix *g0,
      mfem::DenseMatrix *g1, mfem::DenseMatrix *h) const;
};

} // namespace hdg_ns

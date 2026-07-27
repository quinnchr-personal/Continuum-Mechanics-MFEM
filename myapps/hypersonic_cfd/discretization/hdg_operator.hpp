#pragma once

#include "io/exasim_io.hpp"
#include "physics/physics_model.hpp"

#include "mfem.hpp"

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace hycfd
{

struct HDGOptions
{
   int order = 4;
   double tau = 1.0;
   // Volume/face quadrature order = 2*order + quadrature_increment.
   int quadrature_increment = 1;
};

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

// HDG operator for a first-order system with auxiliary gradient
// q = +grad(u): element residual/Jacobian assembly, exact per-element
// gradient elimination q = C u - E uhat, and static condensation to the
// trace unknowns. Runtime polynomial order; triangle and quadrilateral
// elements (mixed meshes supported); physics through the PhysicsModel
// interface with a runtime component count.
class HDGOperator
{
public:
   using ScalarFunction =
      std::function<double(const mfem::Vector &)>;
   // Writes NumComponents() values at the given physical point.
   using StateFunction =
      std::function<void(const mfem::Vector &, double *)>;

   // attr_to_bcid maps boundary attribute a (1-based) to the physics
   // model's registered bc_id via attr_to_bcid[a - 1].
   //
   // Reference-data constructor: AV is evaluated from vdg component zero
   // (requires an all-quadrilateral order-4 Exasim mesh).
   // serial_element_ids maps each local element to its index in the
   // original serial (Exasim-ordered) mesh; required at np>1 because the
   // vdg/udg arrays are indexed in that ordering. Empty means identity
   // (valid only on single-rank runs).
   HDGOperator(
      mfem::ParMesh &mesh, const ExasimArray &vdg,
      const std::vector<ElementOrientation> &orientations,
      const PhysicsModel &physics,
      const std::vector<int> &attr_to_bcid,
      const HDGOptions &options = HDGOptions(),
      const std::vector<int> &serial_element_ids = std::vector<int>());

   // Analytic/frozen-AV constructor used by MMS and sanity problems.
   HDGOperator(
      mfem::ParMesh &mesh,
      const std::vector<ElementOrientation> &orientations,
      ScalarFunction artificial_viscosity,
      const PhysicsModel &physics,
      const std::vector<int> &attr_to_bcid,
      const HDGOptions &options = HDGOptions());

   HDGOperator(
      mfem::ParMesh &mesh, ScalarFunction artificial_viscosity,
      const PhysicsModel &physics,
      const std::vector<int> &attr_to_bcid,
      const HDGOptions &options = HDGOptions());

   int Order() const { return options_.order; }
   int Components() const { return ncu_; }
   int TraceVSize() const;
   int TraceTrueVSize() const;
   int TraceDofsPerFace() const { return nfdof_; }
   int Elements() const;
   int ElementDofs(int element) const;
   const mfem::IntegrationRule &VolumeRule(int element) const;
   const mfem::IntegrationRule &FaceRule() const;
   const mfem::FiniteElementSpace &VolumeSpace() const;
   const mfem::FiniteElementSpace &TraceSpace() const;

   const mfem::DenseMatrix &C(int element, int direction) const;
   const mfem::DenseMatrix &E(int element, int direction) const;

   double MinimumDetJ() const;
   double MaximumAbsAV() const;
   void SetArtificialViscosity(ScalarFunction artificial_viscosity);
   // Retabulates the frozen AV data from a scalar field on this mesh
   // (e.g. a vertex-max-smoothed sensor field).
   void SetArtificialViscosityField(const mfem::GridFunction &av);

   HDGState NewState() const;
   void SetConstantState(const double *state, HDGState &solution) const;
   void ProjectState(const StateFunction &function,
                     HDGState &solution) const;
   void InitializeTraceFromInterior(HDGState &solution) const;
   void RecomputeGradient(HDGState &solution) const;
   void LoadExasimVolumeState(const ExasimArray &udg,
                              bool load_gradient,
                              HDGState &solution) const;
   // uq must hold NumComponents()*(1+dim) values.
   void EvaluateElementState(const HDGState &solution, int element,
                             const mfem::IntegrationPoint &point,
                             double *uq) const;
   void EvaluateTraceState(const HDGState &solution, int face,
                           const mfem::IntegrationPoint &point,
                           double *uhat) const;

   // The source is a test-only manufactured-source hook. Physical runs
   // leave it empty, which is exactly source = 0.
   void SetManufacturedSource(StateFunction source);
   void ClearManufacturedSource();
   // Test-only MMS boundary hook: fb = prescribed_state(x) - uhat.
   void SetDirichletStateOverride(StateFunction state);
   void ClearDirichletStateOverride();

   // Full assembly builds all J*delta=-R blocks, folds q, condenses, and
   // scatters. Residual-only skips every Jacobian/condensation operation.
   HDGResidualNorms Assemble(
      const HDGState &solution, bool build_jacobian,
      double pseudo_time_inverse_step = 0.0);

   const mfem::Vector &VolumeResidual() const;
   const mfem::Vector &TraceResidual() const;
   const mfem::Vector &CondensedResidual() const;
   const mfem::Vector &CondensedRHS() const;
   const mfem::SparseMatrix &CondensedMatrix() const;
   // True-dof condensed system (PtAP across ranks; identity map at np=1).
   const mfem::HypreParMatrix &CondensedParMatrix() const;
   const mfem::Vector &CondensedTrueRHS() const;
   // ldof = P * tdof: expands a true-dof trace increment to the local
   // (shared-duplicated) trace layout used by HDGState::uhat.
   void ExpandTraceIncrement(const mfem::Vector &true_increment,
                             mfem::Vector &local_increment) const;

   // trace_increment is in the local (ldof) trace layout.
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
   void FillArtificialViscosityGridFunction(
      mfem::GridFunction &field) const;

   // Pointwise scratch capacities (states up to 8 components in 3D).
   static constexpr int kMaxComponents = 8;
   static constexpr int kMaxStateEntries = 32;
   static constexpr int kMaxFluxEntries = 24;

private:
   // Reference tables shared by all elements of one geometry.
   struct GeometryTables
   {
      const mfem::FiniteElement *fe = nullptr;
      const mfem::IntegrationRule *rule = nullptr;
      int ndof = 0;
      int nq = 0;
      int nfaces = 0;
      std::vector<double> shape; // [dof + ndof*q]
   };

   // Per-element geometry/AV tables at quadrature points.
   struct ElementGeometry
   {
      std::vector<double> det_j;        // [nq]
      std::vector<double> coords;       // [2*nq]
      std::vector<double> phys_dshape;  // [dir + 2*(dof + ndof*q)]
      std::vector<double> av;           // [nq]
      std::vector<double> face_normals; // [dir + 2*(q + nqf*face)]
      std::vector<double> face_coords;  // [dir + 2*(q + nqf*face)]
      std::vector<double> face_av;      // [q + nqf*face]
      std::vector<double> face_shape;   // [dof + ndof*(q + nqf*face)]
      // Element-side reference coordinates of the face quadrature points,
      // [dir + 2*(q + nqf*face)] — needed to evaluate fields at face qps.
      std::vector<double> face_element_ips;
   };

   mfem::ParMesh &mesh_;
   const ExasimArray *vdg_ = nullptr;
   std::vector<ElementOrientation> orientations_;
   ScalarFunction artificial_viscosity_;
   const PhysicsModel &physics_;
   std::vector<int> attr_to_bcid_;
   HDGOptions options_;
   StateFunction manufactured_source_;
   StateFunction dirichlet_state_override_;

   int ncu_ = 0;
   int nfdof_ = 0;
   int total_u_size_ = 0;

   mfem::L2_FECollection volume_fec_;
   mfem::ParFiniteElementSpace volume_fes_;
   mfem::DG_Interface_FECollection trace_fec_;
   mfem::ParFiniteElementSpace trace_fes_;

   const mfem::IntegrationRule *face_rule_ = nullptr;
   const mfem::FiniteElement *trace_fe_ = nullptr;
   std::vector<double> trace_shape_; // [dof + nfdof*q]

   std::map<mfem::Geometry::Type, GeometryTables> geometry_tables_;
   std::vector<const GeometryTables *> element_tables_;
   std::vector<int> elem_u_offset_; // [ne+1], prefix sums of ncu*ndof
   std::vector<int> global_element_; // local element -> serial Exasim id
   std::vector<ElementGeometry> element_geometry_;
   std::vector<int> face_bc_id_;

   std::vector<mfem::DenseMatrix> mass_;
   std::vector<mfem::DenseMatrix> c_[2];
   std::vector<mfem::DenseMatrix> e_[2];

   mfem::Vector volume_residual_;
   mfem::Vector trace_residual_;
   mfem::Vector condensed_residual_;
   mfem::Vector condensed_rhs_;
   std::unique_ptr<mfem::SparseMatrix> condensed_matrix_;
   mfem::OperatorHandle condensed_true_;
   mfem::Vector condensed_true_rhs_;
   mfem::Vector trace_true_residual_;
   std::vector<mfem::DenseMatrix> inverse_a_f_;
   std::vector<mfem::Vector> inverse_a_ru_;
   bool recovery_is_current_ = false;

   void Initialize();
   void ValidateInputs() const;
   void BuildReferenceTables();
   void BuildGeometryAndAVTables();
   void RetabulateArtificialViscosity();
   void BuildGradientElimination();
   void BuildBoundaryFaceMap();

   int UIdx(int element, int component, int dof) const
   {
      return elem_u_offset_[element] +
             element_tables_[element]->ndof * component + dof;
   }
   int QIdx(int element, int direction, int component, int dof) const
   {
      const int ndof = element_tables_[element]->ndof;
      return 2 * elem_u_offset_[element] +
             ndof * (component + ncu_ * direction) + dof;
   }
   int LocalUIndex(int ndof, int component, int dof) const
   {
      return dof + ndof * component;
   }
   int LocalTraceIndex(int local_face, int component, int dof) const
   {
      return component + ncu_ * (dof + nfdof_ * local_face);
   }
   // MFEM's byVDIM GetFaceVDofs lists [all scalar dofs for component 0,
   // then component 1, ...]. Keep that API-list layout distinct from
   // LocalTraceIndex.
   int FaceVDofListIndex(int component, int dof) const
   {
      return dof + nfdof_ * component;
   }

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

} // namespace hycfd

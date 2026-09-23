// Quadrature-point quantities of the solid formulations and their
// presentations. Every derived quantity is computed once per load step at the
// quadrature points of the kernels' rule (order 2p + 3) into a
// QuadratureFunction, the source of truth, and then presented where
// output.quadrature_at asks:
//   quadrature_points: the raw values, written as a point cloud (<name>_qp);
//   elements: the quadrature-weighted element average, an order-0 L2 field
//             (<name>_elem), flat per element in ParaView and probeable;
//   nodes: a continuous H1 field of the mesh order (<name>) derived from the
//          quadrature data alone, by output.nodal_projection:
//            averaged  - element-wise L2 projection onto each element's
//                        polynomial space, then the arithmetic mean at shared
//                        nodes (the classical extrapolate-and-average);
//            projected - the global L2 projection (consistent mass matrix,
//                        solved by CG to 1e-14).
// The formulation supplies F, P and the energy density at a point through a
// QPointEvaluator, and the measures that depend on the material's kinematics
// (Cauchy stress, volume ratio, strain; materials/kinematics.hpp) through
// CompleteState; derived scalars (von Mises, thickness stretch) are formed
// at the quadrature points and then presented, never from presented data.
// Quantities (components): cauchy_stress (6, VTK order xx yy zz xy yz xz),
// pk1_stress (9, row-major), deformation_gradient (9, row-major), strain (6,
// VTK order), jacobian (1), vonmises (1), energy_density (1),
// thickness_stretch (1, F33 under plane stress). In 2D, F carries F33 (1 for
// plane strain, the thickness stretch for plane stress), so the out-of-plane
// terms are included.
//                       finite strain          small strain (linear_elastic)
//   cauchy_stress       J^{-1} P F^T           P (= pk1_stress, symmetric)
//   jacobian            det F                  1 + tr(eps)
//   strain              (F^T F - I) / 2        eps = sym(F - I)
//   deformation_gradient, thickness_stretch:   I + Grad u, 1 + eps_33
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "base/config.hpp"
#include "base/fields.hpp"
#include "base/tensor.hpp"
#include "kernels/total_lagrangian.hpp"
#include "materials/kinematics.hpp"
#include "mfem.hpp"

namespace cmf
{

struct QPointState
{
  tensor<double, 3, 3> F;
  tensor<double, 3, 3> P;
  double energy = 0.0; // stored energy per unit reference volume
  // Set by CompleteState from F and P.
  tensor<double, 3, 3> sigma;  // Cauchy stress
  tensor<double, 3, 3> strain;
  double J = 1.0;              // volume ratio
};

// The entries of s that depend on the kinematics of the material, from s.F
// and s.P; every evaluator ends with this call.
template <typename Material>
inline void CompleteState(const Material &material, QPointState &s)
{
  s.sigma = CauchyStress(material, s.F, s.P);
  s.strain = Strain(material, s.F);
  s.J = VolumeRatio(material, s.F);
}

// Sets the integration point on T and fills the state there; q is the index
// of the point in the element's rule (the slot of a history-dependent
// material's HistoryField).
using QPointEvaluator = std::function<void(mfem::ElementTransformation &,
                                           const mfem::IntegrationPoint &, int q, QPointState &)>;

struct QuantityInfo
{
  const char *name;
  int components;
};

// All quadrature quantities, in output order.
const std::vector<QuantityInfo> &Quantities();
int QuantityComponents(const std::string &name); // 0 for an unknown name
void PackQuantity(const std::string &name, const QPointState &s, double *out);

// F from the displacement gradient at the current point (plane strain in 2D;
// the plane-stress adapter completes F33 through CompleteF).
tensor<double, 3, 3> DeformationGradientAt(const mfem::DenseMatrix &grad, int dim);

// Fills the packed components of the named quantity at the current point
// (another physics' quantities: the flux of the scalar transport).
using QValueEvaluator = std::function<void(mfem::ElementTransformation &, const mfem::IntegrationPoint &,
                                           int q, const std::string &name, double *out)>;

class QuadratureFields
{
public:
  // Creates the quantities named in out.fields, restricted to those in
  // `available`, with the presentations listed in out.quadrature_at.
  QuadratureFields(mfem::ParMesh &mesh, mfem::FiniteElementCollection &h1_fec, int order,
                   const OutputConfig &out, const std::vector<std::string> &available);
  // The same for a list of quantities of another physics (all available).
  QuadratureFields(mfem::ParMesh &mesh, mfem::FiniteElementCollection &h1_fec, int order,
                   const OutputConfig &out, const std::vector<QuantityInfo> &quantities);
  bool Empty() const { return fields_.empty(); }
  // Evaluates the state at every quadrature point and refreshes all presentations.
  void Update(const QPointEvaluator &eval);
  // The same with the values of every quantity given by name (the second constructor).
  void UpdateValues(const QValueEvaluator &eval);
  void Register(FieldRegistry &registry);

private:
  struct Field
  {
    std::string name;
    int nc = 1;
    std::unique_ptr<mfem::QuadratureFunction> qf;      // source of truth
    std::unique_ptr<mfem::ParFiniteElementSpace> h1;   // nodes
    std::unique_ptr<mfem::ParGridFunction> nodal;
    std::unique_ptr<mfem::ParFiniteElementSpace> l2;   // elements (order 0)
    std::unique_ptr<mfem::ParGridFunction> elem;
  };

  void Build(int order, const OutputConfig &out, const std::vector<QuantityInfo> &quantities);
  void Fill(const QPointEvaluator &eval);
  void FillValues(const QValueEvaluator &eval);
  void Present();
  void ElementAverage(Field &f);
  void ProjectAveraged(Field &f);
  void ProjectConsistent(Field &f);

  mfem::ParMesh &mesh_;
  mfem::FiniteElementCollection &h1_fec_;
  bool nodes_ = false, elements_ = false, qpoints_ = false, consistent_ = false;
  std::unique_ptr<mfem::QuadratureSpace> qspace_;
  std::unique_ptr<mfem::L2_FECollection> l2_fec_;
  std::unique_ptr<mfem::ParFiniteElementSpace> scalar_h1_;
  std::unique_ptr<mfem::HypreParMatrix> mass_;
  std::unique_ptr<mfem::HypreSmoother> mass_prec_;
  std::unique_ptr<mfem::CGSolver> mass_solver_;
  std::vector<Field> fields_;
};

} // namespace cmf

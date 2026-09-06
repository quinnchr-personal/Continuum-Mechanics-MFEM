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
// QPointEvaluator; derived scalars (J, von Mises, thickness stretch) are formed
// at the quadrature points and then presented, never from presented data.
// Quantities (components): cauchy_stress (6, VTK order xx yy zz xy yz xz),
// pk1_stress (9, row-major), deformation_gradient (9, row-major), jacobian
// (1), vonmises (1), energy_density (1), thickness_stretch (1, F33 under
// plane stress). In 2D, F carries F33 (1 for plane strain, the thickness
// stretch for plane stress), so the out-of-plane terms are included.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "base/config.hpp"
#include "base/fields.hpp"
#include "base/tensor.hpp"
#include "kernels/total_lagrangian.hpp"
#include "mfem.hpp"

namespace cmf
{

struct QPointState
{
  tensor<double, 3, 3> F;
  tensor<double, 3, 3> P;
  double energy = 0.0; // stored energy per unit reference volume
};

// Sets the integration point on T and fills the state there.
using QPointEvaluator = std::function<void(mfem::ElementTransformation &,
                                           const mfem::IntegrationPoint &, QPointState &)>;

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

class QuadratureFields
{
public:
  // Creates the quantities named in out.fields, restricted to those in
  // `available`, with the presentations listed in out.quadrature_at.
  QuadratureFields(mfem::ParMesh &mesh, mfem::FiniteElementCollection &h1_fec, int order,
                   const OutputConfig &out, const std::vector<std::string> &available);
  bool Empty() const { return fields_.empty(); }
  // Evaluates the state at every quadrature point and refreshes all presentations.
  void Update(const QPointEvaluator &eval);
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

  void Fill(const QPointEvaluator &eval);
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

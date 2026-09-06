#include "physics/mixed_solid_mechanics_tl.hpp"

#include <cmath>

#include "kernels/mixed_total_lagrangian.hpp"
#include "solvers/saddle_point_solver.hpp"

namespace cmf
{

namespace
{

template <typename Material>
class MixedStressCoefficient : public mfem::Coefficient
{
public:
  enum Kind { VON_MISES, JACOBIAN };
  MixedStressCoefficient(const mfem::ParGridFunction &u, const mfem::ParGridFunction &p,
                         const Material &m, Kind kind)
    : u_(u), p_(p), material_(m), kind_(kind) {}

  mfem::real_t Eval(mfem::ElementTransformation &T,
                    const mfem::IntegrationPoint &ip) override
  {
    T.SetIntPoint(&ip);
    u_.GetVectorGradient(T, grad_);
    const double p = p_.GetValue(T, ip);
    if (T.GetDimension() == 2) { return Value<2>(p); }
    return Value<3>(p);
  }

private:
  template <int dim>
  double Value(double p) const
  {
    tensor<double, dim, dim> H;
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++) { H(i, j) = grad_(i, j); }
    if (kind_ == JACOBIAN) { return det(DeformationGradient<dim>(H)); }
    return VonMises(QPointMixedCauchyStress<Material, dim>(material_, H, p));
  }

  const mfem::ParGridFunction &u_;
  const mfem::ParGridFunction &p_;
  Material material_;
  Kind kind_;
  mfem::DenseMatrix grad_;
};

mfem::Vector ToVector(const std::vector<double> &v)
{
  mfem::Vector out(int(v.size()));
  for (std::size_t i = 0; i < v.size(); i++) { out(int(i)) = v[i]; }
  return out;
}

} // namespace

MixedSolidMechanicsTL::MixedSolidMechanicsTL(mfem::ParMesh &mesh, const AppConfig &cfg,
                                             const MixedMaterial &material)
  : SolidProblem(0),
    mesh_(mesh), dim_(mesh.Dimension()), order_(cfg.mesh.order),
    rho0_(cfg.material.rho0), material_(material),
    fec_u_(cfg.mesh.order, mesh.Dimension()),
    fes_u_(&mesh, &fec_u_, mesh.Dimension(), mfem::Ordering::byVDIM),
    fec_p_(cfg.mesh.order > 1 ? cfg.mesh.order - 1 : 1, mesh.Dimension()),
    fes_p_(&mesh, &fec_p_)
{
  if (cfg.mesh.order < 2)
  {
    throw ConfigError("formulation: mixed needs mesh.order >= 2 (Taylor-Hood pair)");
  }
  std::visit([this](const auto &mat)
  {
    using M = std::decay_t<decltype(mat)>;
    if constexpr (std::is_same_v<M, MooneyRivlin>) { mu_ = mat.ShearModulus(); }
    else { mu_ = mat.mu; }
    kappa_ = mat.kappa;
    incompressible_ = mat.Incompressible();
  }, material_);
  spaces_.SetSize(2);
  spaces_[0] = &fes_u_;
  spaces_[1] = &fes_p_;
  offsets_.SetSize(3);
  offsets_[0] = 0;
  offsets_[1] = fes_u_.GetTrueVSize();
  offsets_[2] = offsets_[1] + fes_p_.GetTrueVSize();
  height = width = offsets_[2];
  nlf_ = std::make_unique<mfem::ParBlockNonlinearForm>(spaces_);
  Build(cfg);
}

void MixedSolidMechanicsTL::Build(const AppConfig &cfg)
{
  std::visit([this](const auto &mat)
  {
    using M = std::decay_t<decltype(mat)>;
    nlf_->AddDomainIntegrator(new MixedTotalLagrangianIntegrator<M>(mat));
  }, material_);

  for (const BoundaryCondition &bc : cfg.bcs.dirichlet)
  {
    CheckVectorSize(bc.value, "bcs.dirichlet[].value");
    const mfem::Vector v = ToVector(bc.value);
    owned_coefs_.push_back(std::make_unique<mfem::VectorConstantCoefficient>(v));
    AddDirichlet(bc.attr, *owned_coefs_.back());
  }
  for (const BoundaryCondition &bc : cfg.bcs.traction)
  {
    CheckVectorSize(bc.value, "bcs.traction[].value");
    const mfem::Vector v = ToVector(bc.value);
    owned_coefs_.push_back(std::make_unique<mfem::VectorConstantCoefficient>(v));
    AddTraction(bc.attr, *owned_coefs_.back());
  }
  if (!cfg.body_force.empty())
  {
    CheckVectorSize(cfg.body_force, "body_force");
    const mfem::Vector v = ToVector(cfg.body_force);
    owned_coefs_.push_back(std::make_unique<mfem::VectorConstantCoefficient>(v));
    SetBodyForce(*owned_coefs_.back());
  }
}

void MixedSolidMechanicsTL::CheckVectorSize(const std::vector<double> &v,
                                            const std::string &what) const
{
  if (int(v.size()) != dim_)
  {
    throw ConfigError(what + " has " + std::to_string(v.size()) +
                      " entries, expected " + std::to_string(dim_));
  }
}

void MixedSolidMechanicsTL::CheckCoefficient(mfem::VectorCoefficient &c,
                                             const std::string &what) const
{
  if (c.GetVDim() != dim_)
  {
    throw ConfigError(what + ": coefficient has " + std::to_string(c.GetVDim()) +
                      " components, expected " + std::to_string(dim_));
  }
}

mfem::Array<int> MixedSolidMechanicsTL::Marker(const std::vector<int> &attrs) const
{
  const int max_attr = mesh_.bdr_attributes.Size() ? mesh_.bdr_attributes.Max() : 0;
  mfem::Array<int> marker(max_attr);
  marker = 0;
  for (int a : attrs)
  {
    if (a < 1 || a > max_attr)
    {
      throw ConfigError("boundary attribute " + std::to_string(a) +
                        " is not in the mesh (max " + std::to_string(max_attr) + ")");
    }
    marker[a - 1] = 1;
  }
  return marker;
}

void MixedSolidMechanicsTL::AddDirichlet(const std::vector<int> &attrs,
                                         mfem::VectorCoefficient &u_bar)
{
  CheckCoefficient(u_bar, "AddDirichlet");
  dirichlet_.push_back({Marker(attrs), &u_bar});
  finalized_ = false;
}

void MixedSolidMechanicsTL::AddTraction(const std::vector<int> &attrs,
                                        mfem::VectorCoefficient &T_bar)
{
  CheckCoefficient(T_bar, "AddTraction");
  traction_.push_back({Marker(attrs), &T_bar});
  finalized_ = false;
}

void MixedSolidMechanicsTL::SetBodyForce(mfem::VectorCoefficient &b)
{
  CheckCoefficient(b, "SetBodyForce");
  body_force_ = &b;
  finalized_ = false;
}

void MixedSolidMechanicsTL::ClearBoundaryConditions()
{
  dirichlet_.clear();
  traction_.clear();
  body_force_ = nullptr;
  finalized_ = false;
}

void MixedSolidMechanicsTL::Finalize()
{
  const int max_attr = mesh_.bdr_attributes.Size() ? mesh_.bdr_attributes.Max() : 0;
  ess_u_marker_.SetSize(max_attr);
  ess_u_marker_ = 0;
  for (const BCEntry &bc : dirichlet_)
  {
    for (int i = 0; i < max_attr; i++) { ess_u_marker_[i] |= bc.marker[i]; }
  }
  ess_p_marker_.SetSize(max_attr);
  ess_p_marker_ = 0; // the pressure carries no essential conditions
  fes_u_.GetEssentialTrueDofs(ess_u_marker_, ess_tdof_list_);
  mfem::Array<mfem::Array<int> *> ess_bdr(2);
  ess_bdr[0] = &ess_u_marker_;
  ess_bdr[1] = &ess_p_marker_;
  mfem::Array<mfem::Vector *> rhs(2);
  rhs = nullptr;
  nlf_->SetEssentialBC(ess_bdr, rhs);

  // Dead loads on the displacement block (follower loads: see the
  // displacement formulation's TODO seam).
  mfem::ParLinearForm load(&fes_u_);
  if (body_force_)
  {
    rho0_body_force_ = std::make_unique<mfem::ScalarVectorProductCoefficient>(
      rho0_, *body_force_);
    load.AddDomainIntegrator(new mfem::VectorDomainLFIntegrator(*rho0_body_force_));
  }
  for (BCEntry &bc : traction_)
  {
    load.AddBoundaryIntegrator(new mfem::VectorBoundaryLFIntegrator(*bc.coef),
                               bc.marker);
  }
  load.Assemble();
  load_true_.SetSize(fes_u_.GetTrueVSize());
  load.ParallelAssemble(load_true_);

  // Pressure mass matrix for the Schur complement approximation.
  mfem::ParBilinearForm mass(&fes_p_);
  mfem::ConstantCoefficient one(1.0);
  mass.AddDomainIntegrator(new mfem::MassIntegrator(one));
  mass.Assemble();
  mass.Finalize();
  pressure_mass_.reset(mass.ParallelAssemble());
  finalized_ = true;
}

void MixedSolidMechanicsTL::SetLoadFactor(double lambda)
{
  if (!finalized_) { Finalize(); }
  load_factor_ = lambda;
}

void MixedSolidMechanicsTL::ApplyDirichlet(mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "MixedSolidMechanicsTL: call Finalize() first");
  mfem::ParGridFunction g(const_cast<mfem::ParFiniteElementSpace *>(&fes_u_));
  g = 0.0;
  for (const BCEntry &bc : dirichlet_)
  {
    g.ProjectBdrCoefficient(*bc.coef, bc.marker);
  }
  mfem::Vector g_true(fes_u_.GetTrueVSize());
  g.GetTrueDofs(g_true);
  for (int i = 0; i < ess_tdof_list_.Size(); i++)
  {
    x(ess_tdof_list_[i]) = load_factor_ * g_true(ess_tdof_list_[i]);
  }
}

void MixedSolidMechanicsTL::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
  MFEM_VERIFY(finalized_, "MixedSolidMechanicsTL: call Finalize() first");
  nlf_->Mult(x, y);
  const int n_u = offsets_[1];
  for (int i = 0; i < n_u; i++) { y(i) -= load_factor_ * load_true_(i); }
  for (int i = 0; i < ess_tdof_list_.Size(); i++) { y(ess_tdof_list_[i]) = 0.0; }
}

mfem::Operator &MixedSolidMechanicsTL::GetGradient(const mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "MixedSolidMechanicsTL: call Finalize() first");
  return nlf_->GetGradient(x);
}

double MixedSolidMechanicsTL::InternalEnergy(const mfem::Vector &x) const
{
  return nlf_->GetEnergy(x);
}

HYPRE_BigInt MixedSolidMechanicsTL::GlobalTrueVSize() const
{
  auto &fu = const_cast<mfem::ParFiniteElementSpace &>(fes_u_);
  auto &fp = const_cast<mfem::ParFiniteElementSpace &>(fes_p_);
  return fu.GlobalTrueVSize() + fp.GlobalTrueVSize();
}

std::string MixedSolidMechanicsTL::Description() const
{
  return "mixed u-p formulation, " + MaterialName(material_) +
         (incompressible_ ? " (incompressible)" : " (kappa " + std::to_string(kappa_) + ")");
}

void MixedSolidMechanicsTL::EnsureFields()
{
  if (displacement_) { return; }
  displacement_ = std::make_unique<mfem::ParGridFunction>(&fes_u_);
  pressure_ = std::make_unique<mfem::ParGridFunction>(&fes_p_);
  *displacement_ = 0.0;
  *pressure_ = 0.0;
  l2_fec_ = std::make_unique<mfem::L2_FECollection>(order_, dim_);
  l2_fes_ = std::make_unique<mfem::ParFiniteElementSpace>(&mesh_, l2_fec_.get());
  vonmises_ = std::make_unique<mfem::ParGridFunction>(l2_fes_.get());
  jacobian_ = std::make_unique<mfem::ParGridFunction>(l2_fes_.get());
  *vonmises_ = 0.0;
  *jacobian_ = 1.0;
}

void MixedSolidMechanicsTL::UpdateFields(const mfem::Vector &x)
{
  EnsureFields();
  mfem::Vector xu(const_cast<mfem::Vector &>(x).GetData(), offsets_[1]);
  mfem::Vector xp(const_cast<mfem::Vector &>(x).GetData() + offsets_[1],
                  offsets_[2] - offsets_[1]);
  displacement_->SetFromTrueDofs(xu);
  pressure_->SetFromTrueDofs(xp);
  std::visit([this](const auto &mat)
  {
    using M = std::decay_t<decltype(mat)>;
    MixedStressCoefficient<M> vm(*displacement_, *pressure_, mat,
                                 MixedStressCoefficient<M>::VON_MISES);
    MixedStressCoefficient<M> jac(*displacement_, *pressure_, mat,
                                  MixedStressCoefficient<M>::JACOBIAN);
    vonmises_->ProjectCoefficient(vm);
    jacobian_->ProjectCoefficient(jac);
  }, material_);
}

void MixedSolidMechanicsTL::RegisterFields(FieldRegistry &registry)
{
  EnsureFields();
  registry.AddExternal("displacement", *displacement_);
  registry.AddExternal("pressure", *pressure_);
  registry.AddExternal("vonmises", *vonmises_);
  registry.AddExternal("jacobian", *jacobian_);
}

std::unique_ptr<mfem::Solver>
MixedSolidMechanicsTL::MakeLinearSolver(const LinearSolverConfig &cfg)
{
  if (!finalized_) { Finalize(); }
  return std::make_unique<SaddlePointSolver>(cfg, fes_u_, offsets_, *pressure_mass_,
                                             mu_, kappa_);
}

} // namespace cmf

#include "physics/solid_mechanics_tl.hpp"

#include <cmath>

#include "kernels/total_lagrangian.hpp"

namespace cmf
{

namespace
{

// Scalar coefficient of the displacement field: von Mises Cauchy stress or
// J = det F, evaluated from Grad u at the point. Instantiated once per
// material type at setup (no per-point dispatch on the variant).
template <typename Material>
class StressCoefficient : public mfem::Coefficient
{
public:
  enum Kind { VON_MISES, JACOBIAN };
  StressCoefficient(const mfem::ParGridFunction &u, const Material &m, Kind kind)
    : u_(u), material_(m), kind_(kind) {}

  mfem::real_t Eval(mfem::ElementTransformation &T,
                    const mfem::IntegrationPoint &ip) override
  {
    T.SetIntPoint(&ip);
    u_.GetVectorGradient(T, grad_);
    if (T.GetDimension() == 2) { return Value<2>(); }
    return Value<3>();
  }

private:
  template <int dim>
  double Value() const
  {
    tensor<double, dim, dim> H;
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++) { H(i, j) = grad_(i, j); }
    if (kind_ == JACOBIAN) { return det(DeformationGradient<dim>(H)); }
    return VonMises(QPointCauchyStress<Material, dim>(material_, H));
  }

  const mfem::ParGridFunction &u_;
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

SolidMechanicsTL::SolidMechanicsTL(mfem::ParMesh &mesh, const YAML::Node &root,
                                   const Material &material)
  : SolidMechanicsTL(mesh, ParseConfig(root), material) {}

SolidMechanicsTL::SolidMechanicsTL(mfem::ParMesh &mesh, const AppConfig &cfg,
                                   const Material &material)
  : QuasiStaticProblem(0),
    mesh_(mesh), dim_(mesh.Dimension()), order_(cfg.mesh.order),
    rho0_(cfg.material.rho0), material_(material),
    fec_(cfg.mesh.order, mesh.Dimension()),
    fes_(&mesh, &fec_, mesh.Dimension(), mfem::Ordering::byVDIM),
    nlf_(&fes_)
{
  height = width = fes_.GetTrueVSize();
  Build(cfg);
}

void SolidMechanicsTL::Build(const AppConfig &cfg)
{
  // One integrator instantiation per material type, chosen once here.
  std::visit([this](const auto &mat)
  {
    using M = std::decay_t<decltype(mat)>;
    nlf_.AddDomainIntegrator(new TotalLagrangianIntegrator<M>(mat));
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

void SolidMechanicsTL::CheckVectorSize(const std::vector<double> &v,
                                       const std::string &what) const
{
  if (int(v.size()) != dim_)
  {
    throw ConfigError(what + " has " + std::to_string(v.size()) +
                      " entries, expected " + std::to_string(dim_));
  }
}

mfem::Array<int> SolidMechanicsTL::Marker(const std::vector<int> &attrs) const
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

void SolidMechanicsTL::AddDirichlet(const std::vector<int> &attrs,
                                    mfem::VectorCoefficient &u_bar)
{
  dirichlet_.push_back({Marker(attrs), &u_bar});
  finalized_ = false;
}

void SolidMechanicsTL::AddTraction(const std::vector<int> &attrs,
                                   mfem::VectorCoefficient &T_bar)
{
  traction_.push_back({Marker(attrs), &T_bar});
  finalized_ = false;
}

void SolidMechanicsTL::SetBodyForce(mfem::VectorCoefficient &b)
{
  body_force_ = &b;
  finalized_ = false;
}

void SolidMechanicsTL::ClearBoundaryConditions()
{
  dirichlet_.clear();
  traction_.clear();
  body_force_ = nullptr;
  finalized_ = false;
}

void SolidMechanicsTL::Finalize()
{
  // Essential true dofs: union of all Dirichlet markers, all components.
  mfem::Array<int> ess_bdr(mesh_.bdr_attributes.Size() ? mesh_.bdr_attributes.Max() : 0);
  ess_bdr = 0;
  for (const BCEntry &bc : dirichlet_)
  {
    for (int i = 0; i < ess_bdr.Size(); i++) { ess_bdr[i] |= bc.marker[i]; }
  }
  fes_.GetEssentialTrueDofs(ess_bdr, ess_tdof_list_);
  nlf_.SetEssentialTrueDofs(ess_tdof_list_);

  // Dead loads: rho0 b in the volume, nominal traction on the boundary.
  mfem::ParLinearForm load(&fes_);
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
  load_true_.SetSize(fes_.GetTrueVSize());
  load.ParallelAssemble(load_true_);
  finalized_ = true;
}

void SolidMechanicsTL::SetLoadFactor(double lambda)
{
  if (!finalized_) { Finalize(); }
  load_factor_ = lambda;
}

void SolidMechanicsTL::ApplyDirichlet(mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "SolidMechanicsTL: call Finalize() first");
  mfem::ParGridFunction g(const_cast<mfem::ParFiniteElementSpace *>(&fes_));
  g = 0.0;
  for (const BCEntry &bc : dirichlet_)
  {
    g.ProjectBdrCoefficient(*bc.coef, const_cast<mfem::Array<int> &>(bc.marker));
  }
  mfem::Vector g_true(fes_.GetTrueVSize());
  g.GetTrueDofs(g_true);
  for (int i = 0; i < ess_tdof_list_.Size(); i++)
  {
    x(ess_tdof_list_[i]) = load_factor_ * g_true(ess_tdof_list_[i]);
  }
}

void SolidMechanicsTL::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
  MFEM_VERIFY(finalized_, "SolidMechanicsTL: call Finalize() first");
  nlf_.Mult(x, y);
  y.Add(-load_factor_, load_true_);
  for (int i = 0; i < ess_tdof_list_.Size(); i++) { y(ess_tdof_list_[i]) = 0.0; }
}

mfem::Operator &SolidMechanicsTL::GetGradient(const mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "SolidMechanicsTL: call Finalize() first");
  return nlf_.GetGradient(x);
}

double SolidMechanicsTL::InternalEnergy(const mfem::Vector &x) const
{
  return const_cast<mfem::ParNonlinearForm &>(nlf_).GetEnergy(x);
}

void SolidMechanicsTL::EnsureFields()
{
  if (displacement_) { return; }
  displacement_ = std::make_unique<mfem::ParGridFunction>(&fes_);
  *displacement_ = 0.0;
  l2_fec_ = std::make_unique<mfem::L2_FECollection>(order_, dim_);
  l2_fes_ = std::make_unique<mfem::ParFiniteElementSpace>(&mesh_, l2_fec_.get());
  vonmises_ = std::make_unique<mfem::ParGridFunction>(l2_fes_.get());
  jacobian_ = std::make_unique<mfem::ParGridFunction>(l2_fes_.get());
  *vonmises_ = 0.0;
  *jacobian_ = 1.0;
}

void SolidMechanicsTL::UpdateFields(const mfem::Vector &x)
{
  EnsureFields();
  displacement_->SetFromTrueDofs(x);
  std::visit([this](const auto &mat)
  {
    using M = std::decay_t<decltype(mat)>;
    StressCoefficient<M> vm(*displacement_, mat, StressCoefficient<M>::VON_MISES);
    StressCoefficient<M> jac(*displacement_, mat, StressCoefficient<M>::JACOBIAN);
    vonmises_->ProjectCoefficient(vm);
    jacobian_->ProjectCoefficient(jac);
  }, material_);
}

void SolidMechanicsTL::RegisterFields(FieldRegistry &registry)
{
  EnsureFields();
  registry.AddExternal("displacement", *displacement_);
  registry.AddExternal("vonmises", *vonmises_);
  registry.AddExternal("jacobian", *jacobian_);
}

} // namespace cmf

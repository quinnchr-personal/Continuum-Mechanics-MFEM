#include "physics/solid_mechanics_tl.hpp"

#include "base/coefficients.hpp"
#include "base/mesh_input.hpp"

#include <cmath>

#include "kernels/total_lagrangian.hpp"
#include "physics/mixed_solid_mechanics_tl.hpp"
#include "solvers/linear_solver.hpp"

namespace cmf
{

namespace
{

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
  : SolidProblem(0),
    mesh_(mesh), dim_(mesh.Dimension()), order_(cfg.mesh.order),
    rho0_(cfg.material.rho0), material_(material),
    fec_(cfg.mesh.order, mesh.Dimension()),
    fes_(&mesh, &fec_, mesh.Dimension(), mfem::Ordering::byVDIM),
    nlf_(&fes_)
{
  height = width = fes_.GetTrueVSize();
  output_cfg_ = cfg.output;
  plane_stress_ = cfg.plane == "stress";
  if (plane_stress_ && dim_ != 2)
  {
    throw ConfigError("plane: stress needs a 2D mesh (got dimension " +
                      std::to_string(dim_) + ")");
  }
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

  for (std::size_t i = 0; i < cfg.bcs.dirichlet.size(); i++)
  {
    const BoundaryCondition &bc = cfg.bcs.dirichlet[i];
    const std::string what = "bcs.dirichlet[" + std::to_string(i) + "]";
    owned_coefs_.push_back(MakeBCCoefficient(bc, dim_, what));
    AddDirichlet(ResolveBoundaryAttributes(mesh_, bc, what), *owned_coefs_.back());
  }
  for (std::size_t i = 0; i < cfg.bcs.traction.size(); i++)
  {
    const BoundaryCondition &bc = cfg.bcs.traction[i];
    const std::string what = "bcs.traction[" + std::to_string(i) + "]";
    owned_coefs_.push_back(MakeBCCoefficient(bc, dim_, what));
    AddTraction(ResolveBoundaryAttributes(mesh_, bc, what), *owned_coefs_.back());
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

void SolidMechanicsTL::CheckCoefficient(mfem::VectorCoefficient &c,
                                        const std::string &what) const
{
  if (c.GetVDim() != dim_)
  {
    throw ConfigError(what + ": coefficient has " + std::to_string(c.GetVDim()) +
                      " components, expected " + std::to_string(dim_));
  }
}

void SolidMechanicsTL::AddDirichlet(const std::vector<int> &attrs,
                                    mfem::VectorCoefficient &u_bar)
{
  CheckCoefficient(u_bar, "AddDirichlet");
  dirichlet_.push_back({Marker(attrs), &u_bar});
  finalized_ = false;
}

void SolidMechanicsTL::AddTraction(const std::vector<int> &attrs,
                                   mfem::VectorCoefficient &T_bar)
{
  CheckCoefficient(T_bar, "AddTraction");
  traction_.push_back({Marker(attrs), &T_bar});
  finalized_ = false;
}

void SolidMechanicsTL::SetBodyForce(mfem::VectorCoefficient &b)
{
  CheckCoefficient(b, "SetBodyForce");
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
  // TODO(follower loads): a pressure per unit current area becomes
  // T = -p J F^{-T} N, which depends on u; it would leave this dead-load
  // linear form and enter the ParNonlinearForm as a boundary integrator with
  // its own tangent (plan section 3.5, out of scope here).
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
  // ParGridFunction takes a non-const space pointer; nothing is modified.
  mfem::ParGridFunction g(const_cast<mfem::ParFiniteElementSpace *>(&fes_));
  g = 0.0;
  for (const BCEntry &bc : dirichlet_)
  {
    g.ProjectBdrCoefficient(*bc.coef, bc.marker);
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
  return nlf_.GetEnergy(x);
}

HYPRE_BigInt SolidMechanicsTL::GlobalTrueVSize() const
{
  return const_cast<mfem::ParFiniteElementSpace &>(fes_).GlobalTrueVSize();
}

std::string SolidMechanicsTL::Description() const
{
  return "displacement formulation, " + MaterialName(material_);
}

std::unique_ptr<mfem::Solver>
SolidMechanicsTL::MakeLinearSolver(const LinearSolverConfig &cfg)
{
  return cmf::MakeLinearSolver(cfg, fes_);
}

std::unique_ptr<SolidProblem> MakeSolidProblem(mfem::ParMesh &mesh, const AppConfig &cfg)
{
  if (cfg.formulation == "mixed")
  {
    return std::make_unique<MixedSolidMechanicsTL>(mesh, cfg, MakeMixedMaterial(cfg.material));
  }
  return std::make_unique<SolidMechanicsTL>(mesh, cfg, MakeMaterial(cfg.material, cfg.plane == "stress"));
}

void SolidMechanicsTL::EnsureFields()
{
  if (displacement_) { return; }
  displacement_ = std::make_unique<mfem::ParGridFunction>(&fes_);
  *displacement_ = 0.0;
  std::vector<std::string> available;
  for (const QuantityInfo &q : Quantities())
  {
    if (std::string(q.name) != "thickness_stretch" || plane_stress_) { available.push_back(q.name); }
  }
  qfields_ = std::make_unique<QuadratureFields>(mesh_, fec_, order_, output_cfg_, available);
}

void SolidMechanicsTL::UpdateFields(const mfem::Vector &x)
{
  EnsureFields();
  displacement_->SetFromTrueDofs(x);
  if (qfields_->Empty()) { return; }
  std::visit([this](const auto &mat)
  {
    using M = std::decay_t<decltype(mat)>;
    mfem::DenseMatrix grad;
    qfields_->Update([&](mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip,
                         QPointState &s)
    {
      T.SetIntPoint(&ip);
      displacement_->GetVectorGradient(T, grad);
      s.F = CompleteF(mat, DeformationGradientAt(grad, T.GetDimension()));
      s.P = mat.PK1(s.F);
      if constexpr (has_energy<M>::value) { s.energy = mat.Energy(s.F); }
      else { s.energy = 0.0; }
    });
  }, material_);
}

void SolidMechanicsTL::RegisterFields(FieldRegistry &registry)
{
  EnsureFields();
  registry.AddExternal("displacement", *displacement_);
  qfields_->Register(registry);
}

} // namespace cmf

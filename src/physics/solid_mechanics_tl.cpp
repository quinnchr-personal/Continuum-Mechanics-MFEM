#include "physics/solid_mechanics_tl.hpp"

#include "base/coefficients.hpp"
#include "base/mesh_input.hpp"

#include <cmath>

#include "kernels/follower_pressure.hpp"
#include "kernels/total_lagrangian.hpp"
#include "solvers/linear_solver.hpp"

namespace cmf
{

SolidMechanicsTL::SolidMechanicsTL(mfem::ParMesh &mesh, const YAML::Node &root,
                                   const Material &material)
  : SolidMechanicsTL(mesh, ParseConfig(root), material) {}

SolidMechanicsTL::SolidMechanicsTL(mfem::ParMesh &mesh, const AppConfig &cfg,
                                   const Material &material)
  : SolidMechanicsTL(mesh, cfg, std::vector<Material>(1, material)) {}

SolidMechanicsTL::SolidMechanicsTL(mfem::ParMesh &mesh, const AppConfig &cfg,
                                   const std::vector<Material> &materials)
  : SolidProblem(0),
    mesh_(mesh), dim_(mesh.Dimension()), order_(cfg.mesh.order),
    rho0_(cfg.material.rho0), materials_(materials),
    fec_(cfg.mesh.order, mesh.Dimension()),
    fes_(&mesh, &fec_, mesh.Dimension(), mfem::Ordering::byVDIM),
    loads_(fes_)
{
  height = width = fes_.GetTrueVSize();
  output_cfg_ = cfg.output;
  plane_stress_ = cfg.plane == "stress";
  if (plane_stress_ && dim_ != 2)
  {
    throw ConfigError("plane: stress needs a 2D mesh (got dimension " +
                      std::to_string(dim_) + ")");
  }
  ResetForm();
  Build(cfg);
}

void SolidMechanicsTL::ResetForm()
{
  nlf_ = std::make_unique<mfem::ParNonlinearForm>(&fes_);
  energy_form_ = std::make_unique<mfem::ParNonlinearForm>(&fes_);
  // One integrator instantiation per material type, chosen once here.
  MFEM_VERIFY(!materials_.empty(), "SolidMechanicsTL: no material");
  std::visit([this](const auto &mat)
  {
    using M = std::decay_t<decltype(mat)>;
    const std::vector<M> table = UnpackMaterials<M>(materials_);
    nlf_->AddDomainIntegrator(new TotalLagrangianIntegrator<M>(table));
    energy_form_->AddDomainIntegrator(new TotalLagrangianIntegrator<M>(table));
  }, materials_[0]);
  follower_markers_.clear();
  finalized_ = false;
}

void SolidMechanicsTL::Build(const AppConfig &cfg)
{
  InstallYamlLoads(*this, mesh_, cfg, dim_, owned_coefs_, owned_scalars_);
}

void SolidMechanicsTL::AddDirichlet(const std::vector<int> &attrs,
                                    mfem::VectorCoefficient &u_bar, const BCOptions &opt)
{
  loads_.AddDirichlet(attrs, u_bar, opt);
  finalized_ = false;
}

void SolidMechanicsTL::AddTraction(const std::vector<int> &attrs,
                                   mfem::VectorCoefficient &T_bar, const BCOptions &opt)
{
  loads_.AddTraction(attrs, T_bar, opt);
  finalized_ = false;
}

void SolidMechanicsTL::AddPressure(const std::vector<int> &attrs, mfem::Coefficient &p,
                                   bool follower, const BCOptions &opt)
{
  if (!follower)
  {
    loads_.AddPressure(attrs, p, opt);
  }
  else
  {
    if (IsSmallStrain(materials_[0]))
    {
      throw ConfigError("traction type follower_pressure is not used by model '" +
                        ModelNameOf(materials_[0]) + "' (small strain: the reference and the "
                        "current configuration coincide; use type: pressure)");
    }
    const double *scale = loads_.AddFollowerPressure(attrs, p, opt);
    follower_markers_.push_back(loads_.Marker(attrs));
    nlf_->AddBdrFaceIntegrator(new FollowerPressureIntegrator(p, scale),
                               follower_markers_.back());
  }
  finalized_ = false;
}

void SolidMechanicsTL::SetBodyForce(mfem::VectorCoefficient &b, const BCOptions &opt)
{
  loads_.SetBodyForce(b, rho0_, opt);
  finalized_ = false;
}

void SolidMechanicsTL::ClearBoundaryConditions()
{
  const bool had_followers = loads_.HasFollowerPressure();
  loads_.Clear();
  if (had_followers) { ResetForm(); }
  finalized_ = false;
}

void SolidMechanicsTL::Finalize()
{
  loads_.Finalize();
  nlf_->SetEssentialTrueDofs(loads_.EssentialTrueDofs());
  finalized_ = true;
}

void SolidMechanicsTL::SetLoadFactor(double t)
{
  if (!finalized_) { Finalize(); }
  loads_.SetTime(t);
}

void SolidMechanicsTL::ApplyDirichlet(mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "SolidMechanicsTL: call Finalize() first");
  loads_.ApplyDirichlet(x);
}

void SolidMechanicsTL::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
  MFEM_VERIFY(finalized_, "SolidMechanicsTL: call Finalize() first");
  nlf_->Mult(x, y);
  y -= loads_.ExternalLoad();
  const mfem::Array<int> &ess = loads_.EssentialTrueDofs();
  for (int i = 0; i < ess.Size(); i++) { y(ess[i]) = 0.0; }
}

std::vector<Reaction> SolidMechanicsTL::Reactions(const mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "SolidMechanicsTL: call Finalize() first");
  // The full residual: the form zeroes its essential rows in Mult, so they
  // are lifted for this evaluation and restored afterwards.
  mfem::Array<int> none;
  nlf_->SetEssentialTrueDofs(none);
  mfem::Vector r(x.Size());
  nlf_->Mult(x, r);
  nlf_->SetEssentialTrueDofs(loads_.EssentialTrueDofs());
  r -= loads_.ExternalLoad();
  if (IsSmallStrain(materials_[0]))
  {
    // Equilibrium holds on the reference configuration: reference moment arms.
    mfem::Vector zero(x.Size());
    zero = 0.0;
    return loads_.Reactions(r, zero);
  }
  return loads_.Reactions(r, x);
}

mfem::Operator &SolidMechanicsTL::GetGradient(const mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "SolidMechanicsTL: call Finalize() first");
  return nlf_->GetGradient(x);
}

double SolidMechanicsTL::InternalEnergy(const mfem::Vector &x) const
{
  return energy_form_->GetEnergy(x);
}

HYPRE_BigInt SolidMechanicsTL::GlobalTrueVSize() const
{
  return const_cast<mfem::ParFiniteElementSpace &>(fes_).GlobalTrueVSize();
}

std::string SolidMechanicsTL::Description() const
{
  return "displacement formulation, " + MaterialName(materials_[0]) +
         (IsSmallStrain(materials_[0]) ? " (small strain)" : "") +
         VolumetricLawSuffix(materials_[0]) + (materials_.size() > 1 ? " (regions)" : "");
}

std::unique_ptr<mfem::Solver>
SolidMechanicsTL::MakeLinearSolver(const LinearSolverConfig &cfg)
{
  if (cfg.type == "cg_amg" && loads_.HasFollowerPressure())
  {
    throw ConfigError("solver.linear.type: cg_amg needs a symmetric tangent, but a "
                      "follower_pressure entry makes it non-symmetric; use gmres_amg");
  }
  return cmf::MakeLinearSolver(cfg, fes_);
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
  std::visit([this](const auto &first)
  {
    using M = std::decay_t<decltype(first)>;
    const std::vector<M> table = UnpackMaterials<M>(materials_);
    mfem::DenseMatrix grad;
    qfields_->Update([&](mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip,
                         QPointState &s)
    {
      const M &mat = MaterialAt(table, T.Attribute);
      T.SetIntPoint(&ip);
      displacement_->GetVectorGradient(T, grad);
      s.F = CompleteF(mat, DeformationGradientAt(grad, T.GetDimension()));
      s.P = mat.PK1(s.F);
      if constexpr (has_energy<M>::value) { s.energy = mat.Energy(s.F); }
      else { s.energy = 0.0; }
    });
  }, materials_[0]);
}

void SolidMechanicsTL::RegisterFields(FieldRegistry &registry)
{
  EnsureFields();
  registry.AddExternal("displacement", *displacement_);
  qfields_->Register(registry);
}

} // namespace cmf

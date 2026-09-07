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
  : SolidProblem(0),
    mesh_(mesh), dim_(mesh.Dimension()), order_(cfg.mesh.order),
    rho0_(cfg.material.rho0), material_(material),
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
  // One integrator instantiation per material type, chosen once here.
  std::visit([this](const auto &mat)
  {
    using M = std::decay_t<decltype(mat)>;
    nlf_->AddDomainIntegrator(new TotalLagrangianIntegrator<M>(mat));
  }, material_);
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

mfem::Operator &SolidMechanicsTL::GetGradient(const mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "SolidMechanicsTL: call Finalize() first");
  return nlf_->GetGradient(x);
}

double SolidMechanicsTL::InternalEnergy(const mfem::Vector &x) const
{
  return nlf_->GetEnergy(x);
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

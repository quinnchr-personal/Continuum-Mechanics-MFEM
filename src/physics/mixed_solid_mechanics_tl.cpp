#include "physics/mixed_solid_mechanics_tl.hpp"

#include <cmath>

#include "kernels/follower_pressure.hpp"
#include "kernels/mixed_total_lagrangian.hpp"
#include "solvers/saddle_point_solver.hpp"

namespace cmf
{

MixedSolidMechanicsTL::MixedSolidMechanicsTL(mfem::ParMesh &mesh, const AppConfig &cfg,
                                             const MixedMaterial &material)
  : MixedSolidMechanicsTL(mesh, cfg, std::vector<MixedMaterial>(1, material)) {}

MixedSolidMechanicsTL::MixedSolidMechanicsTL(mfem::ParMesh &mesh, const AppConfig &cfg,
                                             const std::vector<MixedMaterial> &materials)
  : SolidProblem(0),
    mesh_(mesh), dim_(mesh.Dimension()), order_(cfg.mesh.order),
    rho0_(cfg.material.rho0), materials_(materials),
    fec_u_(cfg.mesh.order, mesh.Dimension()),
    fes_u_(&mesh, &fec_u_, mesh.Dimension(), mfem::Ordering::byVDIM),
    fec_p_(cfg.mesh.order > 1 ? cfg.mesh.order - 1 : 1, mesh.Dimension()),
    fes_p_(&mesh, &fec_p_),
    loads_(fes_u_)
{
  if (cfg.mesh.order < 2)
  {
    throw ConfigError("formulation: mixed needs mesh.order >= 2 (Taylor-Hood pair)");
  }
  MFEM_VERIFY(!materials_.empty(), "MixedSolidMechanicsTL: no material");
  std::visit([this](const auto &mat)
  {
    mu_ = mat.ShearModulus();
    kappa_ = mat.kappa;
    incompressible_ = mat.Incompressible();
  }, materials_[0]);
  for (const MixedMaterial &m : materials_)
  {
    const bool inc = std::visit([](const auto &mat) { return mat.Incompressible(); }, m);
    if (inc != incompressible_)
    {
      throw ConfigError("material.regions: every region must be incompressible or none");
    }
  }
  spaces_.SetSize(2);
  spaces_[0] = &fes_u_;
  spaces_[1] = &fes_p_;
  offsets_.SetSize(3);
  offsets_[0] = 0;
  offsets_[1] = fes_u_.GetTrueVSize();
  offsets_[2] = offsets_[1] + fes_p_.GetTrueVSize();
  height = width = offsets_[2];
  output_cfg_ = cfg.output;
  ResetForm();
  Build(cfg);
}

void MixedSolidMechanicsTL::ResetForm()
{
  nlf_ = std::make_unique<BlockForm>(spaces_);
  energy_form_ = std::make_unique<mfem::ParBlockNonlinearForm>(spaces_);
  std::visit([this](const auto &mat)
  {
    using M = std::decay_t<decltype(mat)>;
    const std::vector<M> table = UnpackMaterials<M>(materials_);
    nlf_->AddDomainIntegrator(new MixedTotalLagrangianIntegrator<M>(table));
    energy_form_->AddDomainIntegrator(new MixedTotalLagrangianIntegrator<M>(table));
  }, materials_[0]);
  follower_markers_.clear();
  finalized_ = false;
}

void MixedSolidMechanicsTL::Build(const AppConfig &cfg)
{
  InstallYamlLoads(*this, mesh_, cfg, dim_, owned_coefs_, owned_scalars_);
}

void MixedSolidMechanicsTL::AddDirichlet(const std::vector<int> &attrs,
                                         mfem::VectorCoefficient &u_bar, const BCOptions &opt)
{
  loads_.AddDirichlet(attrs, u_bar, opt);
  finalized_ = false;
}

void MixedSolidMechanicsTL::AddTraction(const std::vector<int> &attrs,
                                        mfem::VectorCoefficient &T_bar, const BCOptions &opt)
{
  loads_.AddTraction(attrs, T_bar, opt);
  finalized_ = false;
}

void MixedSolidMechanicsTL::AddPressure(const std::vector<int> &attrs, mfem::Coefficient &p,
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
    nlf_->AddBdrFaceIntegrator(new BlockFollowerPressureIntegrator(p, scale),
                               follower_markers_.back());
  }
  finalized_ = false;
}

void MixedSolidMechanicsTL::SetBodyForce(mfem::VectorCoefficient &b, const BCOptions &opt)
{
  loads_.SetBodyForce(b, rho0_, opt);
  finalized_ = false;
}

void MixedSolidMechanicsTL::ClearBoundaryConditions()
{
  const bool had_followers = loads_.HasFollowerPressure();
  loads_.Clear();
  if (had_followers) { ResetForm(); }
  finalized_ = false;
}

void MixedSolidMechanicsTL::Finalize()
{
  loads_.Finalize();
  // The pressure carries no essential conditions.
  ess_p_empty_.SetSize(0);
  nlf_->SetEssentialTrueDofs(0, loads_.EssentialTrueDofs());
  nlf_->SetEssentialTrueDofs(1, ess_p_empty_);

  // Pressure mass matrix for the Schur complement approximation.
  mfem::ParBilinearForm mass(&fes_p_);
  mfem::ConstantCoefficient one(1.0);
  mass.AddDomainIntegrator(new mfem::MassIntegrator(one));
  mass.Assemble();
  mass.Finalize();
  pressure_mass_.reset(mass.ParallelAssemble());
  finalized_ = true;
}

void MixedSolidMechanicsTL::SetLoadFactor(double t)
{
  if (!finalized_) { Finalize(); }
  loads_.SetTime(t);
}

void MixedSolidMechanicsTL::ApplyDirichlet(mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "MixedSolidMechanicsTL: call Finalize() first");
  loads_.ApplyDirichlet(x); // the displacement block leads the block vector
}

void MixedSolidMechanicsTL::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
  MFEM_VERIFY(finalized_, "MixedSolidMechanicsTL: call Finalize() first");
  nlf_->Mult(x, y);
  const int n_u = offsets_[1];
  const mfem::Vector &L = loads_.ExternalLoad();
  for (int i = 0; i < n_u; i++) { y(i) -= L(i); }
  const mfem::Array<int> &ess = loads_.EssentialTrueDofs();
  for (int i = 0; i < ess.Size(); i++) { y(ess[i]) = 0.0; }
}

mfem::Operator &MixedSolidMechanicsTL::GetGradient(const mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "MixedSolidMechanicsTL: call Finalize() first");
  return nlf_->GetGradient(x);
}

double MixedSolidMechanicsTL::InternalEnergy(const mfem::Vector &x) const
{
  return energy_form_->GetEnergy(x);
}

HYPRE_BigInt MixedSolidMechanicsTL::GlobalTrueVSize() const
{
  auto &fu = const_cast<mfem::ParFiniteElementSpace &>(fes_u_);
  auto &fp = const_cast<mfem::ParFiniteElementSpace &>(fes_p_);
  return fu.GlobalTrueVSize() + fp.GlobalTrueVSize();
}

std::string MixedSolidMechanicsTL::Description() const
{
  return "mixed u-p formulation, " + MaterialName(materials_[0]) +
         (materials_.size() > 1 ? " (regions)" : "") +
         (incompressible_ ? " (incompressible)" : " (kappa " + std::to_string(kappa_) + ")");
}

void MixedSolidMechanicsTL::EnsureFields()
{
  if (displacement_) { return; }
  displacement_ = std::make_unique<mfem::ParGridFunction>(&fes_u_);
  pressure_ = std::make_unique<mfem::ParGridFunction>(&fes_p_);
  *displacement_ = 0.0;
  *pressure_ = 0.0;
  std::vector<std::string> available;
  for (const QuantityInfo &q : Quantities())
  {
    if (std::string(q.name) != "thickness_stretch") { available.push_back(q.name); }
  }
  qfields_ = std::make_unique<QuadratureFields>(mesh_, fec_u_, order_, output_cfg_, available);
}

void MixedSolidMechanicsTL::UpdateFields(const mfem::Vector &x)
{
  EnsureFields();
  mfem::Vector xu(const_cast<mfem::Vector &>(x).GetData(), offsets_[1]);
  mfem::Vector xp(const_cast<mfem::Vector &>(x).GetData() + offsets_[1],
                  offsets_[2] - offsets_[1]);
  displacement_->SetFromTrueDofs(xu);
  pressure_->SetFromTrueDofs(xp);
  if (qfields_->Empty()) { return; }
  std::visit([&](const auto &first)
  {
    using M = std::decay_t<decltype(first)>;
    const std::vector<M> table = UnpackMaterials<M>(materials_);
    mfem::DenseMatrix grad;
    qfields_->Update([&](mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip,
                         QPointState &s)
    {
      const M &mat = MaterialAt(table, T.Attribute);
      const double inv_kappa = mat.Incompressible() ? 0.0 : 1.0 / mat.kappa;
      T.SetIntPoint(&ip);
      displacement_->GetVectorGradient(T, grad);
      const double p = pressure_->GetValue(T, ip);
      s.F = DeformationGradientAt(grad, T.GetDimension());
      s.P = MixedPK1(mat, s.F, p);
      // The mixed functional's integrand, consistent with InternalEnergy.
      s.energy = mat.EnergyIso(s.F) + p * (det(s.F) - 1.0) - 0.5 * inv_kappa * p * p;
    });
  }, materials_[0]);
}

void MixedSolidMechanicsTL::RegisterFields(FieldRegistry &registry)
{
  EnsureFields();
  registry.AddExternal("displacement", *displacement_);
  registry.AddExternal("pressure", *pressure_);
  qfields_->Register(registry);
}

std::unique_ptr<mfem::Solver>
MixedSolidMechanicsTL::MakeLinearSolver(const LinearSolverConfig &cfg)
{
  if (!finalized_) { Finalize(); }
  return std::make_unique<SaddlePointSolver>(cfg, fes_u_, offsets_, *pressure_mass_,
                                             mu_, kappa_);
}

} // namespace cmf

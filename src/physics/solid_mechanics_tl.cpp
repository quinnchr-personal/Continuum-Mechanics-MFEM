#include "physics/solid_mechanics_tl.hpp"

#include "base/coefficients.hpp"
#include "base/mesh_input.hpp"

#include <cmath>

#include "kernels/follower_pressure.hpp"
#include "kernels/rigid_sphere_contact.hpp"
#include "kernels/total_lagrangian.hpp"
#include "solvers/direct_solver.hpp"
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
    rho0_(cfg.material.rho0), density_table_(MakeDensityTable(cfg.material, mesh)),
    density_(density_table_), materials_(materials),
    fec_(cfg.mesh.order, mesh.Dimension()),
    fes_(&mesh, &fec_, mesh.Dimension(), mfem::Ordering::byVDIM),
    loads_(fes_),
    r2pi_([](const mfem::Vector &x) { return 2.0 * M_PI * x(0); })
{
  height = width = fes_.GetTrueVSize();
  output_cfg_ = cfg.output;
  plane_stress_ = cfg.plane == "stress";
  axisymmetric_ = cfg.plane == "axisymmetric";
  if ((plane_stress_ || axisymmetric_) && dim_ != 2)
  {
    throw ConfigError("plane: " + cfg.plane + " needs a 2D mesh (got dimension " +
                      std::to_string(dim_) + ")");
  }
  loads_.SetAxisymmetric(axisymmetric_);
  density_axi_ = std::make_unique<mfem::ProductCoefficient>(r2pi_, density_);
  InitializeHistory();
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
    auto *integ = new TotalLagrangianIntegrator<M>(table);
    integ->SetHistory(history_.get());
    integ->SetAxisymmetric(axisymmetric_);
    nlf_->AddDomainIntegrator(integ);
    auto *energy_integ = new TotalLagrangianIntegrator<M>(table);
    energy_integ->SetHistory(history_.get());
    energy_integ->SetAxisymmetric(axisymmetric_);
    energy_form_->AddDomainIntegrator(energy_integ);
  }, materials_[0]);
  follower_markers_.clear();
  contact_markers_.clear();
  contact_forms_.clear();
  finalized_ = false;
  gradient_ = nullptr;
}

void SolidMechanicsTL::InitializeHistory()
{
  const int size = HistorySizeOf(materials_);
  if (size == 0) { history_.reset(); return; }
  if (!history_) { history_ = std::make_unique<HistoryField>(mesh_, 2 * order_ + 3, size); }
  std::vector<double> init(std::size_t(size), 0.0);
  std::visit([&](const auto &first)
  {
    using M = std::decay_t<decltype(first)>;
    if constexpr (has_history<M>::value)
    {
      const std::vector<M> table = UnpackMaterials<M>(materials_);
      for (int e = 0; e < mesh_.GetNE(); e++)
      {
        std::fill(init.begin(), init.end(), 0.0);
        MaterialAt(table, mesh_.GetAttribute(e)).InitialHistory(init.data());
        history_->Fill(e, init.data());
      }
    }
  }, materials_[0]);
  history_->SetDt(0.0);
}

void SolidMechanicsTL::ResetHistory(double t)
{
  InitializeHistory();
  t_accepted_ = t;
}

void SolidMechanicsTL::UpdateHistory(const mfem::Vector &x)
{
  EnsureFields();
  displacement_->SetFromTrueDofs(x);
  std::visit([&](const auto &first)
  {
    using M = std::decay_t<decltype(first)>;
    if constexpr (has_history<M>::value)
    {
      const std::vector<M> table = UnpackMaterials<M>(materials_);
      mfem::DenseMatrix grad;
      for (int e = 0; e < mesh_.GetNE(); e++)
      {
        mfem::ElementTransformation &T = *mesh_.GetElementTransformation(e);
        const M &mat = MaterialAt(table, T.Attribute);
        const mfem::IntegrationRule &ir = history_->Rule(e);
        for (int q = 0; q < ir.GetNPoints(); q++)
        {
          const mfem::IntegrationPoint &ip = ir.IntPoint(q);
          T.SetIntPoint(&ip);
          displacement_->GetVectorGradient(T, grad);
          const tensor<double, 3, 3> F = GradientToF(T, ip, grad);
          mat.Update(F, history_->Old(e, q), history_->Dt(), history_->New(e, q));
        }
      }
    }
  }, materials_[0]);
  history_->Commit();
  history_->SetDt(0.0);
}

void SolidMechanicsTL::AcceptStep(const mfem::Vector &x)
{
  if (history_) { UpdateHistory(x); }
  t_accepted_ = loads_.Time();
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
    nlf_->AddBdrFaceIntegrator(new FollowerPressureIntegrator(p, scale, axisymmetric_),
                               follower_markers_.back());
  }
  finalized_ = false;
}

void SolidMechanicsTL::AddRigidSphereContact(const std::vector<int> &attrs,
                                             mfem::VectorCoefficient &center, double radius,
                                             double penalty, const BCOptions &opt)
{
  const double *scale = loads_.AddRigidSphereContact(attrs, center, radius, penalty, opt);
  contact_markers_.push_back(loads_.Marker(attrs));
  nlf_->AddBdrFaceIntegrator(new RigidSphereContactIntegrator(center, radius, penalty, scale, axisymmetric_),
                             contact_markers_.back());
  auto form = std::make_unique<mfem::ParNonlinearForm>(&fes_);
  form->AddBdrFaceIntegrator(new RigidSphereContactIntegrator(center, radius, penalty, scale, axisymmetric_),
                             contact_markers_.back());
  contact_forms_.push_back(std::move(form));
  finalized_ = false;
}

void SolidMechanicsTL::SetBodyForce(mfem::VectorCoefficient &b, const BCOptions &opt)
{
  loads_.SetBodyForce(b, density_, opt);
  finalized_ = false;
}

void SolidMechanicsTL::ClearBoundaryConditions()
{
  const bool had_boundary_terms = loads_.HasFollowerPressure() || loads_.NumContacts() > 0;
  loads_.Clear();
  if (had_boundary_terms) { ResetForm(); }
  finalized_ = false;
}

void SolidMechanicsTL::Finalize()
{
  loads_.Finalize();
  nlf_->SetEssentialTrueDofs(loads_.EssentialTrueDofs());
  gradient_ = nullptr; // the essential rows and columns may have changed
  finalized_ = true;
}

void SolidMechanicsTL::SetLoadFactor(double t)
{
  if (!finalized_) { Finalize(); }
  loads_.SetTime(t);
  if (history_) { history_->SetDt(t - t_accepted_); }
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

void SolidMechanicsTL::FullResidual(const mfem::Vector &x, mfem::Vector &r) const
{
  MFEM_VERIFY(finalized_, "SolidMechanicsTL: call Finalize() first");
  // The form zeroes its essential rows in Mult, so they are lifted for this
  // evaluation and restored afterwards.
  mfem::Array<int> none;
  nlf_->SetEssentialTrueDofs(none);
  r.SetSize(x.Size());
  nlf_->Mult(x, r);
  nlf_->SetEssentialTrueDofs(loads_.EssentialTrueDofs());
  r -= loads_.ExternalLoad();
}

std::vector<Reaction> SolidMechanicsTL::ReactionsFrom(const mfem::Vector &r,
                                                      const mfem::Vector &x) const
{
  std::vector<Reaction> out;
  if (IsSmallStrain(materials_[0]))
  {
    // Equilibrium holds on the reference configuration: reference moment arms.
    mfem::Vector zero(x.Size());
    zero = 0.0;
    out = loads_.Reactions(r, zero);
  }
  else { out = loads_.Reactions(r, x); }
  // The resultant of every contact entry: its form gives -t on the dofs.
  mfem::Vector f(x.Size());
  for (std::size_t i = 0; i < contact_forms_.size(); i++)
  {
    contact_forms_[i]->Mult(x, f);
    f *= -1.0;
    out.push_back(loads_.Resultant(f, x, loads_.ContactName(i)));
  }
  return out;
}

mfem::Operator &SolidMechanicsTL::GetGradient(const mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "SolidMechanicsTL: call Finalize() first");
  if (gradient_ && reuse_gradient_ && IsLinear()) { return *gradient_; }
  ++*gradient_stamp_;
  gradient_ = &nlf_->GetGradient(x);
  return *gradient_;
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
         VolumetricLawSuffix(materials_[0]) + (materials_.size() > 1 ? " (regions)" : "") +
         (axisymmetric_ ? " (axisymmetric)" : "");
}

tensor<double, 3, 3> SolidMechanicsTL::GradientToF(mfem::ElementTransformation &T,
                                                   const mfem::IntegrationPoint &ip,
                                                   const mfem::DenseMatrix &grad) const
{
  tensor<double, 3, 3> F = DeformationGradientAt(grad, dim_);
  if (axisymmetric_)
  {
    mfem::Vector X, u;
    T.Transform(ip, X);
    displacement_->GetVectorValue(T, ip, u);
    F(2, 2) = X(0) > 0.0 ? 1.0 + u(0) / X(0) : 1.0 + grad(0, 0);
  }
  return F;
}

std::unique_ptr<mfem::Solver>
SolidMechanicsTL::MakeLinearSolver(const LinearSolverConfig &cfg)
{
  if (cfg.type == "cg_amg" && loads_.HasFollowerPressure())
  {
    throw ConfigError("solver.linear.type: cg_amg needs a symmetric tangent, but a "
                      "follower_pressure entry makes it non-symmetric; use gmres_amg");
  }
  if (cfg.type == "direct")
  {
    auto direct = std::make_unique<DirectSolver>(fes_.GetComm(), cfg);
    direct->SetOperatorStamp(gradient_stamp_);
    return direct;
  }
  std::unique_ptr<LinearSolver> solver = cmf::MakeLinearSolver(cfg, fes_);
  solver->SetOperatorStamp(gradient_stamp_);
  return solver;
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
    qfields_->Update([&](mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip, int q,
                         QPointState &s)
    {
      const M &mat = MaterialAt(table, T.Attribute);
      const auto &bound = AtPoint(mat, history_.get(), T.ElementNo, q);
      T.SetIntPoint(&ip);
      displacement_->GetVectorGradient(T, grad);
      s.F = CompleteF(mat, GradientToF(T, ip, grad));
      s.P = bound.PK1(s.F);
      if constexpr (has_energy<bound_t<M>>::value) { s.energy = bound.Energy(s.F); }
      else { s.energy = 0.0; }
      CompleteState(mat, s);
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

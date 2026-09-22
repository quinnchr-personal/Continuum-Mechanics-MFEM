#include "physics/mixed_solid_mechanics_tl.hpp"

#include <cmath>

#include "kernels/follower_pressure.hpp"
#include "kernels/mixed_total_lagrangian.hpp"
#include "kernels/rigid_sphere_contact.hpp"
#include "solvers/direct_solver.hpp"
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
    rho0_(cfg.material.rho0), density_table_(MakeDensityTable(cfg.material, mesh)),
    density_(density_table_), materials_(materials),
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
  InitializeHistory();
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
    auto *integ = new MixedTotalLagrangianIntegrator<M>(table);
    integ->SetHistory(history_.get());
    nlf_->AddDomainIntegrator(integ);
    auto *energy_integ = new MixedTotalLagrangianIntegrator<M>(table);
    energy_integ->SetHistory(history_.get());
    energy_form_->AddDomainIntegrator(energy_integ);
  }, materials_[0]);
  follower_markers_.clear();
  contact_markers_.clear();
  contact_forms_.clear();
  finalized_ = false;
  gradient_ = nullptr;
}

void MixedSolidMechanicsTL::InitializeHistory()
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

void MixedSolidMechanicsTL::ResetHistory(double t)
{
  InitializeHistory();
  t_accepted_ = t;
}

void MixedSolidMechanicsTL::UpdateHistory(const mfem::Vector &x)
{
  EnsureFields();
  mfem::Vector xu(const_cast<mfem::Vector &>(x).GetData(), offsets_[1]);
  displacement_->SetFromTrueDofs(xu);
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
          const tensor<double, 3, 3> F = DeformationGradientAt(grad, dim_);
          mat.Update(F, history_->Old(e, q), history_->Dt(), history_->New(e, q));
        }
      }
    }
  }, materials_[0]);
  history_->Commit();
  history_->SetDt(0.0);
}

void MixedSolidMechanicsTL::AcceptStep(const mfem::Vector &x)
{
  if (history_) { UpdateHistory(x); }
  t_accepted_ = loads_.Time();
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
    if (IsSmallStrain(materials_[0]))
    {
      throw ConfigError("traction type follower_pressure is not used by model '" +
                        ModelNameOf(materials_[0]) + "' (small strain: the reference and the "
                        "current configuration coincide; use type: pressure)");
    }
    const double *scale = loads_.AddFollowerPressure(attrs, p, opt);
    follower_markers_.push_back(loads_.Marker(attrs));
    nlf_->AddBdrFaceIntegrator(new BlockFollowerPressureIntegrator(p, scale),
                               follower_markers_.back());
  }
  finalized_ = false;
}

void MixedSolidMechanicsTL::AddRigidSphereContact(const std::vector<int> &attrs,
                                                  mfem::VectorCoefficient &center, double radius,
                                                  double penalty, const BCOptions &opt)
{
  const double *scale = loads_.AddRigidSphereContact(attrs, center, radius, penalty, opt);
  contact_markers_.push_back(loads_.Marker(attrs));
  nlf_->AddBdrFaceIntegrator(new BlockRigidSphereContactIntegrator(center, radius, penalty, scale),
                             contact_markers_.back());
  auto form = std::make_unique<mfem::ParBlockNonlinearForm>(spaces_);
  form->AddBdrFaceIntegrator(new BlockRigidSphereContactIntegrator(center, radius, penalty, scale),
                             contact_markers_.back());
  contact_forms_.push_back(std::move(form));
  finalized_ = false;
}

void MixedSolidMechanicsTL::SetBodyForce(mfem::VectorCoefficient &b, const BCOptions &opt)
{
  loads_.SetBodyForce(b, density_, opt);
  finalized_ = false;
}

void MixedSolidMechanicsTL::ClearBoundaryConditions()
{
  const bool had_boundary_terms = loads_.HasFollowerPressure() || loads_.NumContacts() > 0;
  loads_.Clear();
  if (had_boundary_terms) { ResetForm(); }
  finalized_ = false;
}

void MixedSolidMechanicsTL::Finalize()
{
  loads_.Finalize();
  // The pressure carries no essential conditions.
  ess_p_empty_.SetSize(0);
  nlf_->SetEssentialTrueDofs(0, loads_.EssentialTrueDofs());
  nlf_->SetEssentialTrueDofs(1, ess_p_empty_);
  gradient_ = nullptr; // the essential rows and columns may have changed

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
  if (history_) { history_->SetDt(t - t_accepted_); }
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

void MixedSolidMechanicsTL::FullResidual(const mfem::Vector &x, mfem::Vector &r) const
{
  MFEM_VERIFY(finalized_, "MixedSolidMechanicsTL: call Finalize() first");
  mfem::Array<int> none;
  nlf_->SetEssentialTrueDofs(0, none);
  r.SetSize(x.Size());
  nlf_->Mult(x, r);
  nlf_->SetEssentialTrueDofs(0, loads_.EssentialTrueDofs());
  mfem::Vector r_u(r.GetData(), offsets_[1]);
  r_u -= loads_.ExternalLoad();
}

// Only the displacement blocks of r and x are read (x may be that block alone).
std::vector<Reaction> MixedSolidMechanicsTL::ReactionsFrom(const mfem::Vector &r,
                                                           const mfem::Vector &x) const
{
  const int n_u = offsets_[1];
  mfem::Vector r_u(const_cast<double *>(r.GetData()), n_u),
               x_u(const_cast<double *>(x.GetData()), n_u);
  std::vector<Reaction> out;
  if (IsSmallStrain(materials_[0]))
  {
    // Equilibrium holds on the reference configuration: reference moment arms.
    mfem::Vector zero(n_u);
    zero = 0.0;
    out = loads_.Reactions(r_u, zero);
  }
  else { out = loads_.Reactions(r_u, x_u); }
  // The resultant of every contact entry (its form gives -t on the
  // displacement dofs); x may be the displacement block alone.
  if (!contact_forms_.empty())
  {
    mfem::Vector xb(offsets_[2]), f(offsets_[2]);
    xb = 0.0;
    for (int i = 0; i < n_u; i++) { xb(i) = x(i); }
    if (x.Size() == offsets_[2]) { xb = x; }
    for (std::size_t i = 0; i < contact_forms_.size(); i++)
    {
      contact_forms_[i]->Mult(xb, f);
      mfem::Vector f_u(f.GetData(), n_u);
      f_u *= -1.0;
      out.push_back(loads_.Resultant(f_u, x_u, loads_.ContactName(i)));
    }
  }
  return out;
}

mfem::Operator &MixedSolidMechanicsTL::GetGradient(const mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "MixedSolidMechanicsTL: call Finalize() first");
  if (gradient_ && reuse_gradient_ && IsLinear()) { return *gradient_; }
  ++*gradient_stamp_;
  gradient_ = &nlf_->GetGradient(x);
  return *gradient_;
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
         (IsSmallStrain(materials_[0]) ? " (small strain)" : "") +
         (materials_.size() > 1 ? " (regions)" : "") +
         (incompressible_ ? " (incompressible)" : " (kappa " + std::to_string(kappa_) +
                            VolumetricLawSuffix(materials_[0]) + ")");
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
    qfields_->Update([&](mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip, int q,
                         QPointState &s)
    {
      const M &mat = MaterialAt(table, T.Attribute);
      const auto &bound = AtPoint(mat, history_.get(), T.ElementNo, q);
      const double inv_kappa = mat.Incompressible() ? 0.0 : 1.0 / mat.kappa;
      T.SetIntPoint(&ip);
      displacement_->GetVectorGradient(T, grad);
      const double p = pressure_->GetValue(T, ip);
      s.F = DeformationGradientAt(grad, T.GetDimension());
      s.P = MixedPK1(bound, s.F, p);
      // The mixed functional's integrand, consistent with InternalEnergy.
      const double J = VolumeRatio(mat, s.F);
      s.energy = bound.EnergyIso(s.F) + p * (J - 1.0) -
                 (inv_kappa > 0.0 ? mat.ComplementaryVolumetricEnergy(p) : 0.0);
      CompleteState(mat, s);
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
  if (cfg.type == "direct")
  {
    auto direct = std::make_unique<DirectSolver>(fes_u_.GetComm(), cfg);
    direct->SetOperatorStamp(gradient_stamp_);
    return direct;
  }
  auto solver = std::make_unique<SaddlePointSolver>(cfg, fes_u_, offsets_, *pressure_mass_,
                                                    mu_, kappa_);
  solver->SetOperatorStamp(gradient_stamp_);
  return solver;
}

} // namespace cmf

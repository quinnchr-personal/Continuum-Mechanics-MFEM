#include "physics/thermo_solid_mechanics_tl.hpp"

#include <cmath>

#include "kernels/follower_pressure.hpp"
#include "kernels/rigid_sphere_contact.hpp"
#include "kernels/thermo_mixed_total_lagrangian.hpp"
#include "solvers/direct_solver.hpp"

namespace cmf
{

ThermoSolidMechanicsTL::ThermoSolidMechanicsTL(mfem::ParMesh &mesh, const AppConfig &cfg,
                                               const ThermoMaterial &material)
  : ThermoSolidMechanicsTL(mesh, cfg, std::vector<ThermoMaterial>(1, material)) {}

ThermoSolidMechanicsTL::ThermoSolidMechanicsTL(mfem::ParMesh &mesh, const AppConfig &cfg,
                                               const std::vector<ThermoMaterial> &materials)
  : SolidProblem(0),
    mesh_(mesh), dim_(mesh.Dimension()), order_(cfg.mesh.order),
    rho0_(cfg.material.rho0), density_table_(MakeDensityTable(cfg.material, mesh)),
    density_(density_table_), materials_(materials),
    fec_u_(cfg.mesh.order, mesh.Dimension()),
    fes_u_(&mesh, &fec_u_, mesh.Dimension(), mfem::Ordering::byVDIM),
    fec_p_(cfg.mesh.order > 1 ? cfg.mesh.order - 1 : 1, mesh.Dimension()),
    fes_p_(&mesh, &fec_p_),
    fes_t_(&mesh, &fec_p_),
    loads_(fes_u_),
    r2pi_([](const mfem::Vector &x) { return 2.0 * M_PI * x(0); })
{
  if (cfg.mesh.order < 2)
  {
    throw ConfigError("the coupled u-p-theta formulation needs mesh.order >= 2 (Taylor-Hood)");
  }
  if (cfg.plane == "stress")
  {
    throw ConfigError("material.thermal: plane stress is not available with a temperature field");
  }
  axisymmetric_ = cfg.plane == "axisymmetric";
  if (axisymmetric_ && dim_ != 2)
  {
    throw ConfigError("plane: axisymmetric needs a 2D mesh (got dimension " +
                      std::to_string(dim_) + ")");
  }
  loads_.SetAxisymmetric(axisymmetric_);
  density_axi_ = std::make_unique<mfem::ProductCoefficient>(r2pi_, density_);
  MFEM_VERIFY(!materials_.empty(), "ThermoSolidMechanicsTL: no material");
  std::visit([this](const auto &mat)
  {
    mu_ = mat.ShearModulus();
    kappa_ = mat.kappa;
    incompressible_ = mat.Incompressible();
    theta0_ = mat.thermal.theta0;
  }, materials_[0]);
  for (const ThermoMaterial &m : materials_)
  {
    std::visit([&](const auto &mat)
    {
      if (mat.Incompressible() != incompressible_)
      {
        throw ConfigError("material.regions: every region must be incompressible or none");
      }
      if (mat.thermal.theta0 != theta0_)
      {
        throw ConfigError("material.regions: every region shares the base's reference temperature theta0");
      }
    }, m);
  }
  spaces_.SetSize(3);
  spaces_[0] = &fes_u_;
  spaces_[1] = &fes_p_;
  spaces_[2] = &fes_t_;
  offsets_.SetSize(4);
  offsets_[0] = 0;
  offsets_[1] = fes_u_.GetTrueVSize();
  offsets_[2] = offsets_[1] + fes_p_.GetTrueVSize();
  offsets_[3] = offsets_[2] + fes_t_.GetTrueVSize();
  height = width = offsets_[3];
  output_cfg_ = cfg.output;
  InitializeHistory();
  ResetForm();
  Build(cfg);
}

void ThermoSolidMechanicsTL::ResetForm()
{
  nlf_ = std::make_unique<BlockForm>(spaces_);
  energy_form_ = std::make_unique<mfem::ParBlockNonlinearForm>(spaces_);
  std::visit([this](const auto &mat)
  {
    using M = std::decay_t<decltype(mat)>;
    const std::vector<M> table = UnpackMaterials<M>(materials_);
    auto *integ = new ThermoMixedTotalLagrangianIntegrator<M>(table);
    integ->SetHistory(history_.get());
    integ->SetAxisymmetric(axisymmetric_);
    nlf_->AddDomainIntegrator(integ);
    auto *energy_integ = new ThermoMixedTotalLagrangianIntegrator<M>(table);
    energy_integ->SetHistory(history_.get());
    energy_integ->SetAxisymmetric(axisymmetric_);
    energy_form_->AddDomainIntegrator(energy_integ);
  }, materials_[0]);
  follower_markers_.clear();
  contact_markers_.clear();
  contact_forms_.clear();
  fluxes_.clear();
  finalized_ = false;
  gradient_ = nullptr;
}

// C = I and theta = theta0 at every point: the state of InitialState.
void ThermoSolidMechanicsTL::InitializeHistory()
{
  constexpr int size = 7;
  if (!history_) { history_ = std::make_unique<HistoryField>(mesh_, 2 * order_ + 3, size); }
  double init[size];
  tensor<double, 3, 3> C = I<3>();
  std::visit([&](const auto &mat)
  {
    using M = std::decay_t<decltype(mat)>;
    ThermoMixedTotalLagrangianIntegrator<M>::PackHistory(C, theta0_, init);
  }, materials_[0]);
  for (int e = 0; e < mesh_.GetNE(); e++) { history_->Fill(e, init); }
  history_->SetDt(0.0);
}

void ThermoSolidMechanicsTL::ResetHistory(double t)
{
  InitializeHistory();
  t_accepted_ = t;
}

void ThermoSolidMechanicsTL::UpdateHistory(const mfem::Vector &x)
{
  EnsureFields();
  mfem::Vector xu(const_cast<mfem::Vector &>(x).GetData(), offsets_[1]);
  mfem::Vector xt(const_cast<mfem::Vector &>(x).GetData() + offsets_[2], offsets_[3] - offsets_[2]);
  displacement_->SetFromTrueDofs(xu);
  temperature_->SetFromTrueDofs(xt);
  std::visit([&](const auto &first)
  {
    using M = std::decay_t<decltype(first)>;
    mfem::DenseMatrix grad;
    for (int e = 0; e < mesh_.GetNE(); e++)
    {
      mfem::ElementTransformation &T = *mesh_.GetElementTransformation(e);
      const mfem::IntegrationRule &ir = history_->Rule(e);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
        const mfem::IntegrationPoint &ip = ir.IntPoint(q);
        T.SetIntPoint(&ip);
        displacement_->GetVectorGradient(T, grad);
        const tensor<double, 3, 3> F = GradientToF(T, ip, grad);
        const double theta = temperature_->GetValue(T, ip);
        ThermoMixedTotalLagrangianIntegrator<M>::PackHistory(transpose(F) * F, theta, history_->New(e, q));
      }
    }
  }, materials_[0]);
  history_->Commit();
  history_->SetDt(0.0);
}

void ThermoSolidMechanicsTL::AcceptStep(const mfem::Vector &x)
{
  UpdateHistory(x);
  t_accepted_ = loads_.Time();
}

void ThermoSolidMechanicsTL::InitialState(mfem::Vector &x) const
{
  MFEM_VERIFY(x.Size() == offsets_[3], "ThermoSolidMechanicsTL::InitialState: wrong size");
  x = 0.0;
  for (int i = offsets_[2]; i < offsets_[3]; i++) { x(i) = theta0_; }
}

void ThermoSolidMechanicsTL::Build(const AppConfig &cfg)
{
  InstallYamlLoads(*this, mesh_, cfg, dim_, owned_coefs_, owned_scalars_);
}

void ThermoSolidMechanicsTL::AddDirichlet(const std::vector<int> &attrs,
                                          mfem::VectorCoefficient &u_bar, const BCOptions &opt)
{
  loads_.AddDirichlet(attrs, u_bar, opt);
  finalized_ = false;
}

void ThermoSolidMechanicsTL::AddTraction(const std::vector<int> &attrs,
                                         mfem::VectorCoefficient &T_bar, const BCOptions &opt)
{
  loads_.AddTraction(attrs, T_bar, opt);
  finalized_ = false;
}

void ThermoSolidMechanicsTL::AddPressure(const std::vector<int> &attrs, mfem::Coefficient &p,
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
    nlf_->AddBdrFaceIntegrator(new ThreeBlockFaceAdapter(new BlockFollowerPressureIntegrator(p, scale, axisymmetric_)),
                               follower_markers_.back());
  }
  finalized_ = false;
}

void ThermoSolidMechanicsTL::AddRigidSphereContact(const std::vector<int> &attrs,
                                                   mfem::VectorCoefficient &center, double radius,
                                                   double penalty, const BCOptions &opt)
{
  const double *scale = loads_.AddRigidSphereContact(attrs, center, radius, penalty, opt);
  contact_markers_.push_back(loads_.Marker(attrs));
  nlf_->AddBdrFaceIntegrator(new ThreeBlockFaceAdapter(new BlockRigidSphereContactIntegrator(center, radius, penalty, scale, axisymmetric_)),
                             contact_markers_.back());
  auto form = std::make_unique<mfem::ParBlockNonlinearForm>(spaces_);
  form->AddBdrFaceIntegrator(new ThreeBlockFaceAdapter(new BlockRigidSphereContactIntegrator(center, radius, penalty, scale, axisymmetric_)),
                             contact_markers_.back());
  contact_forms_.push_back(std::move(form));
  finalized_ = false;
}

void ThermoSolidMechanicsTL::SetBodyForce(mfem::VectorCoefficient &b, const BCOptions &opt)
{
  loads_.SetBodyForce(b, density_, opt);
  finalized_ = false;
}

void ThermoSolidMechanicsTL::AddTemperature(const std::vector<int> &attrs, mfem::Coefficient &theta_bar,
                                            const BCOptions &opt)
{
  TemperatureEntry e;
  e.marker = loads_.Marker(attrs);
  e.coef = &theta_bar;
  e.opt = opt;
  if (e.opt.name.empty()) { e.opt.name = "temperature[" + std::to_string(temperatures_.size()) + "]"; }
  temperatures_.push_back(std::move(e));
  finalized_ = false;
}

void ThermoSolidMechanicsTL::AddHeatFlux(const std::vector<int> &attrs, mfem::Coefficient &h,
                                         bool current_area, const BCOptions &opt)
{
  FluxEntry e;
  e.marker = loads_.Marker(attrs);
  e.coef = &h;
  e.current_area = current_area;
  e.opt = opt;
  if (e.opt.name.empty()) { e.opt.name = "heat_flux[" + std::to_string(fluxes_.size()) + "]"; }
  e.scale = std::make_unique<double>(0.0);
  fluxes_.push_back(std::move(e));
  FluxEntry &back = fluxes_.back();
  nlf_->AddBdrFaceIntegrator(new HeatFluxIntegrator(h, back.scale.get(), current_area, history_.get(), axisymmetric_),
                             back.marker);
  finalized_ = false;
}

void ThermoSolidMechanicsTL::ClearBoundaryConditions()
{
  const bool had_boundary_terms = loads_.HasFollowerPressure() || loads_.NumContacts() > 0 || !fluxes_.empty();
  loads_.Clear();
  temperatures_.clear();
  if (had_boundary_terms) { ResetForm(); }
  finalized_ = false;
}

void ThermoSolidMechanicsTL::SetEssential()
{
  ess_p_empty_.SetSize(0);
  nlf_->SetEssentialTrueDofs(0, loads_.EssentialTrueDofs());
  nlf_->SetEssentialTrueDofs(1, ess_p_empty_);
  nlf_->SetEssentialTrueDofs(2, ess_t_);
}

void ThermoSolidMechanicsTL::Finalize()
{
  loads_.Finalize();
  ess_t_.SetSize(0);
  for (TemperatureEntry &e : temperatures_)
  {
    fes_t_.GetEssentialTrueDofs(e.marker, e.tdofs);
    ess_t_.Append(e.tdofs);
  }
  ess_t_.Sort();
  ess_t_.Unique();
  SetEssential();
  gradient_ = nullptr;
  finalized_ = true;
  SetLoadFactor(loads_.Time());
}

void ThermoSolidMechanicsTL::SetLoadFactor(double t)
{
  if (!finalized_) { Finalize(); }
  loads_.SetTime(t);
  for (TemperatureEntry &e : temperatures_) { e.coef->SetTime(t); }
  for (FluxEntry &e : fluxes_)
  {
    e.coef->SetTime(t);
    *e.scale = e.opt.schedule.Eval(t, physical_time_);
  }
  history_->SetDt(t - t_accepted_);
}

void ThermoSolidMechanicsTL::ApplyDirichlet(mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "ThermoSolidMechanicsTL: call Finalize() first");
  loads_.ApplyDirichlet(x); // the displacement block leads the block vector
  auto &fes_t = const_cast<mfem::ParFiniteElementSpace &>(fes_t_);
  mfem::ParGridFunction g(&fes_t);
  mfem::Vector g_true(fes_t.GetTrueVSize());
  for (const TemperatureEntry &e : temperatures_)
  {
    const double s = e.opt.schedule.Eval(loads_.Time(), physical_time_);
    g = 0.0;
    mfem::Array<int> marker(e.marker);
    g.ProjectBdrCoefficient(*e.coef, marker);
    g.GetTrueDofs(g_true);
    for (int i = 0; i < e.tdofs.Size(); i++) { x(offsets_[2] + e.tdofs[i]) = s * g_true(e.tdofs[i]); }
  }
}

void ThermoSolidMechanicsTL::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
  MFEM_VERIFY(finalized_, "ThermoSolidMechanicsTL: call Finalize() first");
  nlf_->Mult(x, y);
  const int n_u = offsets_[1];
  const mfem::Vector &L = loads_.ExternalLoad();
  for (int i = 0; i < n_u; i++) { y(i) -= L(i); }
  const mfem::Array<int> &ess = loads_.EssentialTrueDofs();
  for (int i = 0; i < ess.Size(); i++) { y(ess[i]) = 0.0; }
  for (int i = 0; i < ess_t_.Size(); i++) { y(offsets_[2] + ess_t_[i]) = 0.0; }
}

void ThermoSolidMechanicsTL::FullResidual(const mfem::Vector &x, mfem::Vector &r) const
{
  MFEM_VERIFY(finalized_, "ThermoSolidMechanicsTL: call Finalize() first");
  mfem::Array<int> none;
  nlf_->SetEssentialTrueDofs(0, none);
  nlf_->SetEssentialTrueDofs(2, none);
  r.SetSize(x.Size());
  nlf_->Mult(x, r);
  const_cast<ThermoSolidMechanicsTL *>(this)->SetEssential();
  mfem::Vector r_u(r.GetData(), offsets_[1]);
  r_u -= loads_.ExternalLoad();
}

std::vector<Reaction> ThermoSolidMechanicsTL::ReactionsFrom(const mfem::Vector &r,
                                                            const mfem::Vector &x) const
{
  const int n_u = offsets_[1];
  mfem::Vector r_u(const_cast<double *>(r.GetData()), n_u),
               x_u(const_cast<double *>(x.GetData()), n_u);
  std::vector<Reaction> out = loads_.Reactions(r_u, x_u);
  if (!contact_forms_.empty())
  {
    mfem::Vector xb(offsets_[3]), f(offsets_[3]);
    xb = 0.0;
    for (int i = 0; i < n_u; i++) { xb(i) = x(i); }
    if (x.Size() == offsets_[3]) { xb = x; }
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

mfem::Operator &ThermoSolidMechanicsTL::GetGradient(const mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "ThermoSolidMechanicsTL: call Finalize() first");
  ++*gradient_stamp_;
  gradient_ = &nlf_->GetGradient(x);
  return *gradient_;
}

double ThermoSolidMechanicsTL::InternalEnergy(const mfem::Vector &x) const
{
  return energy_form_->GetEnergy(x);
}

HYPRE_BigInt ThermoSolidMechanicsTL::GlobalTrueVSize() const
{
  auto &fu = const_cast<mfem::ParFiniteElementSpace &>(fes_u_);
  auto &fp = const_cast<mfem::ParFiniteElementSpace &>(fes_p_);
  auto &ft = const_cast<mfem::ParFiniteElementSpace &>(fes_t_);
  return fu.GlobalTrueVSize() + fp.GlobalTrueVSize() + ft.GlobalTrueVSize();
}

std::string ThermoSolidMechanicsTL::Description() const
{
  const std::string law = std::visit([](const auto &mat) { return VolumetricLawSuffixOf(mat); }, materials_[0]);
  return "coupled u-p-theta formulation, " + MaterialName(materials_[0]) +
         (materials_.size() > 1 ? " (regions)" : "") +
         (incompressible_ ? " (incompressible)" : " (kappa " + std::to_string(kappa_) + law + ")") +
         (axisymmetric_ ? " (axisymmetric)" : "");
}

tensor<double, 3, 3> ThermoSolidMechanicsTL::GradientToF(mfem::ElementTransformation &T,
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

void ThermoSolidMechanicsTL::EnsureFields()
{
  if (displacement_) { return; }
  displacement_ = std::make_unique<mfem::ParGridFunction>(&fes_u_);
  pressure_ = std::make_unique<mfem::ParGridFunction>(&fes_p_);
  temperature_ = std::make_unique<mfem::ParGridFunction>(&fes_t_);
  *displacement_ = 0.0;
  *pressure_ = 0.0;
  *temperature_ = theta0_;
  std::vector<std::string> available;
  for (const QuantityInfo &q : Quantities())
  {
    if (std::string(q.name) != "thickness_stretch") { available.push_back(q.name); }
  }
  qfields_ = std::make_unique<QuadratureFields>(mesh_, fec_u_, order_, output_cfg_, available);
}

void ThermoSolidMechanicsTL::UpdateFields(const mfem::Vector &x)
{
  EnsureFields();
  mfem::Vector xu(const_cast<mfem::Vector &>(x).GetData(), offsets_[1]);
  mfem::Vector xp(const_cast<mfem::Vector &>(x).GetData() + offsets_[1], offsets_[2] - offsets_[1]);
  mfem::Vector xt(const_cast<mfem::Vector &>(x).GetData() + offsets_[2], offsets_[3] - offsets_[2]);
  displacement_->SetFromTrueDofs(xu);
  pressure_->SetFromTrueDofs(xp);
  temperature_->SetFromTrueDofs(xt);
  if (qfields_->Empty()) { return; }
  std::visit([&](const auto &first)
  {
    using M = std::decay_t<decltype(first)>;
    const std::vector<M> table = UnpackMaterials<M>(materials_);
    mfem::DenseMatrix grad;
    qfields_->Update([&](mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip, int,
                         QPointState &s)
    {
      const M &mat = MaterialAt(table, T.Attribute);
      T.SetIntPoint(&ip);
      displacement_->GetVectorGradient(T, grad);
      const double p = pressure_->GetValue(T, ip);
      const double theta = temperature_->GetValue(T, ip);
      s.F = GradientToF(T, ip, grad);
      s.P = ThermoMixedPK1(mat, s.F, p, theta);
      s.energy = mat.Energy(s.F, theta);
      CompleteState(mat, s);
    });
  }, materials_[0]);
}

void ThermoSolidMechanicsTL::RegisterFields(FieldRegistry &registry)
{
  EnsureFields();
  registry.AddExternal("displacement", *displacement_);
  registry.AddExternal("pressure", *pressure_);
  registry.AddExternal("temperature", *temperature_);
  qfields_->Register(registry);
}

std::unique_ptr<mfem::Solver>
ThermoSolidMechanicsTL::MakeLinearSolver(const LinearSolverConfig &cfg)
{
  if (!finalized_) { Finalize(); }
  if (cfg.type != "direct")
  {
    throw ConfigError("solver.linear.type: the coupled u-p-theta formulation has no iterative solver; use direct");
  }
  auto direct = std::make_unique<DirectSolver>(fes_u_.GetComm(), cfg);
  direct->SetOperatorStamp(gradient_stamp_);
  return direct;
}

} // namespace cmf

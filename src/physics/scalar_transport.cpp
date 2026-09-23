#include "physics/scalar_transport.hpp"

#include <cmath>
#include <cstdio>

#include "base/coefficients.hpp"
#include "base/mesh_input.hpp"
#include "kernels/scalar_flux.hpp"
#include "solvers/direct_solver.hpp"

namespace cmf
{

namespace
{

ScalarTransportModel ModelOf(const TransportConfig &cfg)
{
  ScalarTransportModel m;
  m.capacity = {cfg.capacity.value, cfg.capacity.slope, cfg.capacity.reference};
  m.conductivity = {cfg.conductivity.value, cfg.conductivity.slope, cfg.conductivity.reference};
  m.reaction = cfg.reaction;
  m.convection = cfg.convection == "conservative" ? ConvectionForm::Conservative
                                                  : ConvectionForm::NonConservative;
  return m;
}

BCOptions OptionsOf(const ScalarCondition &c)
{
  BCOptions opt;
  opt.schedule = c.schedule;
  opt.time_dependent = ExpressionsUseTime({c.expression});
  opt.name = c.name;
  opt.point = c.point;
  return opt;
}

std::string LawText(const AffineLaw &law)
{
  char buf[96];
  if (law.Constant()) { std::snprintf(buf, sizeof(buf), "%g", law.value); }
  else { std::snprintf(buf, sizeof(buf), "%g + %g (u - %g)", law.value, law.slope, law.reference); }
  return buf;
}

} // namespace

ScalarTransport::ScalarTransport(mfem::ParMesh &mesh, int order, const ScalarTransportModel &model,
                                 bool transient, int quadrature_order)
  : QuasiStaticProblem(0),
    mesh_(mesh), dim_(mesh.Dimension()), order_(order), model_(model),
    quadrature_order_(quadrature_order), transient_(transient),
    fec_(order, mesh.Dimension()), fes_(&mesh, &fec_), conditions_(fes_)
{
  height = width = fes_.GetTrueVSize();
  u_old_ = std::make_unique<mfem::ParGridFunction>(&fes_);
  *u_old_ = 0.0;
  ResetForm();
}

ScalarTransport::ScalarTransport(mfem::ParMesh &mesh, const ScalarAppConfig &cfg)
  : ScalarTransport(mesh, cfg.mesh.order, ModelOf(cfg.transport), cfg.time.enabled,
                    cfg.transport.quadrature_order)
{
  unknown_name_ = cfg.transport.unknown;
  output_cfg_ = cfg.output;
  Build(cfg);
}

ScalarTransport::~ScalarTransport() = default;

void ScalarTransport::ResetForm()
{
  nlf_ = std::make_unique<mfem::ParNonlinearForm>(&fes_);
  kernel_ = new Kernel(model_, quadrature_order_);
  kernel_->SetOldState(u_old_.get());
  kernel_->SetDt(dt_);
  kernel_->SetVelocity(velocity_);
  kernel_->SetSource(source_);
  nlf_->AddDomainIntegrator(kernel_);
  finalized_ = false;
  gradient_ = nullptr;
}

void ScalarTransport::Build(const ScalarAppConfig &cfg)
{
  const TransportConfig &tc = cfg.transport;
  if (!tc.velocity.empty())
  {
    if (int(tc.velocity.size()) != dim_)
    {
      throw ConfigError("transport.velocity has " + std::to_string(tc.velocity.size()) +
                        " components, mesh dimension is " + std::to_string(dim_));
    }
    owned_vectors_.push_back(std::make_unique<ExpressionVectorCoefficient>(tc.velocity));
    SetVelocity(*owned_vectors_.back(), ExpressionsUseTime(tc.velocity));
  }
  if (!tc.source.empty())
  {
    owned_scalars_.push_back(std::make_unique<ExpressionCoefficient>(tc.source));
    SetSource(*owned_scalars_.back());
  }
  if (!cfg.initial.empty())
  {
    owned_scalars_.push_back(std::make_unique<ExpressionCoefficient>(cfg.initial));
    SetInitialCondition(*owned_scalars_.back());
  }
  if (!cfg.exact.empty())
  {
    owned_scalars_.push_back(std::make_unique<ExpressionCoefficient>(cfg.exact));
    SetExact(*owned_scalars_.back());
  }
  for (std::size_t i = 0; i < cfg.bcs.dirichlet.size(); i++)
  {
    const ScalarCondition &c = cfg.bcs.dirichlet[i];
    const std::string what = "bcs.dirichlet[" + std::to_string(i) + "]";
    if (c.IsPoint() && int(c.point.size()) != dim_)
    {
      throw ConfigError(what + ".point has " + std::to_string(c.point.size()) +
                        " coordinates, mesh dimension is " + std::to_string(dim_));
    }
    owned_scalars_.push_back(std::make_unique<ExpressionCoefficient>(c.expression));
    const std::vector<int> attrs = c.IsPoint() ? std::vector<int>()
                                               : ResolveBoundaryAttributes(mesh_, c.attr, c.attr_names, what);
    AddDirichlet(attrs, *owned_scalars_.back(), OptionsOf(c));
  }
  for (std::size_t i = 0; i < cfg.bcs.flux.size(); i++)
  {
    const ScalarCondition &c = cfg.bcs.flux[i];
    const std::string what = "bcs.flux[" + std::to_string(i) + "]";
    owned_scalars_.push_back(std::make_unique<ExpressionCoefficient>(c.expression));
    AddFlux(ResolveBoundaryAttributes(mesh_, c.attr, c.attr_names, what), *owned_scalars_.back(), OptionsOf(c));
  }
}

void ScalarTransport::AddDirichlet(const std::vector<int> &attrs, mfem::Coefficient &g, const BCOptions &opt)
{
  conditions_.AddDirichlet(attrs, g, opt);
  finalized_ = false;
}

void ScalarTransport::AddFlux(const std::vector<int> &attrs, mfem::Coefficient &g, const BCOptions &opt)
{
  conditions_.AddFlux(attrs, g, opt);
  finalized_ = false;
}

void ScalarTransport::SetVelocity(mfem::VectorCoefficient &beta, bool uses_time)
{
  if (beta.GetVDim() != dim_)
  {
    throw ConfigError("SetVelocity: the velocity has " + std::to_string(beta.GetVDim()) +
                      " components, expected " + std::to_string(dim_));
  }
  velocity_ = &beta;
  velocity_uses_time_ = uses_time;
  kernel_->SetVelocity(velocity_);
  gradient_ = nullptr;
}

void ScalarTransport::SetSource(mfem::Coefficient &f)
{
  source_ = &f;
  kernel_->SetSource(source_);
}

void ScalarTransport::SetInitialCondition(mfem::Coefficient &u0) { initial_ = &u0; }

void ScalarTransport::SetExact(mfem::Coefficient &u_ex) { exact_ = &u_ex; }

void ScalarTransport::ClearBoundaryConditions()
{
  conditions_.Clear();
  finalized_ = false;
}

void ScalarTransport::Finalize()
{
  conditions_.Finalize();
  nlf_->SetEssentialTrueDofs(conditions_.EssentialTrueDofs());
  gradient_ = nullptr; // the essential rows and columns may have changed
  finalized_ = true;
}

void ScalarTransport::SetLoadFactor(double t)
{
  if (!finalized_) { Finalize(); }
  conditions_.SetTime(t);
  if (velocity_) { velocity_->SetTime(t); }
  if (source_) { source_->SetTime(t); }
  dt_ = transient_ ? t - t_accepted_ : 0.0;
  // Equal steps t_final k / n differ in their last bits: a step within 1e-12
  // of the one the Jacobian was assembled for is taken as that one, so that
  // a linear problem keeps its operator and its solver setup.
  if (gradient_ && gradient_dt_ > 0.0 && std::abs(dt_ - gradient_dt_) <= 1e-12 * gradient_dt_) { dt_ = gradient_dt_; }
  kernel_->SetDt(dt_);
}

void ScalarTransport::ApplyDirichlet(mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "ScalarTransport: call Finalize() first");
  conditions_.ApplyDirichlet(x);
}

void ScalarTransport::AcceptStep(const mfem::Vector &x)
{
  if (transient_)
  {
    u_old_->SetFromTrueDofs(x);
    t_accepted_ = conditions_.Time();
    dt_ = 0.0;
    kernel_->SetDt(0.0);
  }
}

void ScalarTransport::InitialState(mfem::Vector &x)
{
  MFEM_VERIFY(x.Size() == fes_.GetTrueVSize(), "ScalarTransport::InitialState: wrong size");
  if (initial_)
  {
    // The coefficient may also serve as boundary data or exact solution,
    // whose time the conditions set: the initial state is its value at t = 0.
    initial_->SetTime(0.0);
    mfem::ParGridFunction g(&fes_);
    g.ProjectCoefficient(*initial_);
    g.GetTrueDofs(x);
  }
  else { x = 0.0; }
  ResetHistory(0.0, x);
}

void ScalarTransport::ResetHistory(double t, const mfem::Vector &x)
{
  u_old_->SetFromTrueDofs(x);
  t_accepted_ = t;
  dt_ = 0.0;
  kernel_->SetDt(0.0);
}

void ScalarTransport::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
  MFEM_VERIFY(finalized_, "ScalarTransport: call Finalize() first");
  nlf_->Mult(x, y);
  y -= conditions_.ExternalLoad();
  const mfem::Array<int> &ess = conditions_.EssentialTrueDofs();
  for (int i = 0; i < ess.Size(); i++) { y(ess[i]) = 0.0; }
}

void ScalarTransport::FullResidual(const mfem::Vector &x, mfem::Vector &r) const
{
  MFEM_VERIFY(finalized_, "ScalarTransport: call Finalize() first");
  mfem::Array<int> none;
  nlf_->SetEssentialTrueDofs(none);
  r.SetSize(x.Size());
  nlf_->Mult(x, r);
  nlf_->SetEssentialTrueDofs(conditions_.EssentialTrueDofs());
  r -= conditions_.ExternalLoad();
}

std::vector<Flow> ScalarTransport::Flows(const mfem::Vector &x) const
{
  mfem::Vector r;
  FullResidual(x, r);
  return conditions_.Flows(r);
}

mfem::Operator &ScalarTransport::GetGradient(const mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "ScalarTransport: call Finalize() first");
  if (gradient_ && reuse_gradient_ && IsLinear() && gradient_dt_ == dt_ && !velocity_uses_time_)
  {
    return *gradient_;
  }
  ++*gradient_stamp_;
  gradient_ = &nlf_->GetGradient(x);
  gradient_dt_ = dt_;
  return *gradient_;
}

const mfem::IntegrationRule *ScalarTransport::ErrorRule(int geom) const
{
  return &mfem::IntRules.Get(geom, 2 * order_ + 3);
}

ScalarErrors ScalarTransport::Errors(const mfem::Vector &x, double t)
{
  MFEM_VERIFY(exact_, "ScalarTransport::Errors: no exact solution (SetExact / output.exact)");
  EnsureFields();
  unknown_->SetFromTrueDofs(x);
  exact_->SetTime(t);
  if (error_rules_.empty())
  {
    error_rules_.resize(mfem::Geometry::NumGeom, nullptr);
    for (int g = 0; g < mfem::Geometry::NumGeom; g++) { error_rules_[g] = ErrorRule(g); }
  }
  ScalarErrors e;
  e.l2 = unknown_->ComputeL2Error(*exact_, error_rules_.data());
  const double norm = mfem::ComputeGlobalLpNorm(2.0, *exact_, mesh_, error_rules_.data());
  e.rel_l2 = norm > 1e-14 ? e.l2 / norm : 0.0;
  exact_gf_->ProjectCoefficient(*exact_);
  *error_gf_ = *unknown_;
  *error_gf_ -= *exact_gf_;
  double local = error_gf_->Size() ? error_gf_->Normlinf() : 0.0;
  MPI_Allreduce(&local, &e.linf_nodal, 1, MPI_DOUBLE, MPI_MAX, fes_.GetComm());
  return e;
}

double ScalarTransport::NormL2(const mfem::Vector &x)
{
  EnsureFields();
  unknown_->SetFromTrueDofs(x);
  mfem::ConstantCoefficient zero(0.0);
  return unknown_->ComputeL2Error(zero);
}

HYPRE_BigInt ScalarTransport::GlobalTrueVSize() const
{
  return const_cast<mfem::ParFiniteElementSpace &>(fes_).GlobalTrueVSize();
}

std::string ScalarTransport::Description() const
{
  std::string s = "scalar transport, unknown " + unknown_name_ + ", conductivity " + LawText(model_.conductivity);
  if (transient_) { s += ", capacity " + LawText(model_.capacity); }
  else { s += ", steady"; }
  if (velocity_)
  {
    s += std::string(", ") + (model_.Conservative() ? "conservative" : "non-conservative") + " convection";
  }
  if (model_.reaction != 0.0)
  {
    char buf[48];
    std::snprintf(buf, sizeof(buf), ", reaction %g", model_.reaction);
    s += buf;
  }
  if (source_) { s += ", source"; }
  s += IsLinear() ? " (linear)" : " (nonlinear)";
  return s;
}

std::unique_ptr<mfem::Solver> ScalarTransport::MakeLinearSolver(const LinearSolverConfig &cfg)
{
  if (!finalized_) { Finalize(); }
  if (cfg.type == "cg_amg" && velocity_)
  {
    throw ConfigError("solver.linear.type: cg_amg needs a symmetric operator, but a velocity makes it "
                      "non-symmetric; use gmres_amg or direct");
  }
  if (cfg.type == "direct")
  {
    auto direct = std::make_unique<DirectSolver>(fes_.GetComm(), cfg);
    direct->SetOperatorStamp(gradient_stamp_);
    return direct;
  }
  LinearSolverConfig lc = cfg;
  if (lc.amg == "elasticity") { lc.amg = "scalar"; }
  std::unique_ptr<LinearSolver> solver = cmf::MakeLinearSolver(lc, fes_);
  solver->SetOperatorStamp(gradient_stamp_);
  return solver;
}

void ScalarTransport::EnsureFields()
{
  if (unknown_) { return; }
  unknown_ = std::make_unique<mfem::ParGridFunction>(&fes_);
  *unknown_ = 0.0;
  exact_gf_ = std::make_unique<mfem::ParGridFunction>(&fes_);
  *exact_gf_ = 0.0;
  error_gf_ = std::make_unique<mfem::ParGridFunction>(&fes_);
  *error_gf_ = 0.0;
  const std::vector<QuantityInfo> quantities = {{"flux", dim_}};
  qfields_ = std::make_unique<QuadratureFields>(mesh_, fec_, order_, output_cfg_, quantities);
}

void ScalarTransport::UpdateFields(const mfem::Vector &x)
{
  EnsureFields();
  unknown_->SetFromTrueDofs(x);
  if (exact_)
  {
    exact_->SetTime(conditions_.Time());
    exact_gf_->ProjectCoefficient(*exact_);
    *error_gf_ = *unknown_;
    *error_gf_ -= *exact_gf_;
  }
  if (qfields_->Empty()) { return; }
  mfem::Vector grad, bvec;
  qfields_->UpdateValues([&](mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip, int,
                             const std::string &, double *out)
  {
    T.SetIntPoint(&ip);
    const double u = unknown_->GetValue(T, ip);
    unknown_->GetGradient(T, grad);
    const double kappa = model_.Conductivity(u);
    for (int j = 0; j < dim_; j++) { out[j] = kappa * grad(j); }
    if (velocity_ && model_.Conservative())
    {
      velocity_->Eval(bvec, T, ip);
      for (int j = 0; j < dim_; j++) { out[j] -= bvec(j) * u; }
    }
  });
}

void ScalarTransport::RegisterFields(FieldRegistry &registry)
{
  EnsureFields();
  registry.AddExternal(unknown_name_, *unknown_);
  if (exact_)
  {
    registry.AddExternal(unknown_name_ + "_exact", *exact_gf_);
    registry.AddExternal(unknown_name_ + "_error", *error_gf_);
  }
  qfields_->Register(registry);
}

std::unique_ptr<ScalarTransport> MakeScalarTransport(mfem::ParMesh &mesh, const ScalarAppConfig &cfg)
{
  return std::make_unique<ScalarTransport>(mesh, cfg);
}

std::vector<std::string> DescribeScalarTimeStepping(const ScalarAppConfig &cfg)
{
  std::vector<std::string> lines;
  if (!cfg.time.enabled) { return lines; }
  char buf[256];
  const std::vector<double> &bp = cfg.time.breakpoints;
  double dt_min = cfg.time.t_final, dt_max = 0.0, t_prev = 0.0;
  for (double t : bp)
  {
    dt_min = std::min(dt_min, t - t_prev);
    dt_max = std::max(dt_max, t - t_prev);
    t_prev = t;
  }
  std::snprintf(buf, sizeof(buf), "  transient: implicit Euler, t_final %g, %zu steps (dt %g%s)", cfg.time.t_final,
                bp.size(), dt_max, dt_min < dt_max * (1.0 - 1e-12) ? (", smallest " + std::to_string(dt_min)).c_str() : "");
  lines.push_back(buf);
  for (const ScalarCondition &c : cfg.bcs.dirichlet)
  {
    lines.push_back("  dirichlet " + c.name + ": " + DescribeTimeDependence(c.schedule, {c.expression}));
  }
  for (const ScalarCondition &c : cfg.bcs.flux)
  {
    lines.push_back("  flux " + c.name + ": " + DescribeTimeDependence(c.schedule, {c.expression}));
  }
  return lines;
}

} // namespace cmf

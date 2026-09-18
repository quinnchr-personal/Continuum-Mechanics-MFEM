#include "physics/quadrature_fields.hpp"

#include <algorithm>

namespace cmf
{

const std::vector<QuantityInfo> &Quantities()
{
  static const std::vector<QuantityInfo> q = {
    {"cauchy_stress", 6}, {"pk1_stress", 9}, {"deformation_gradient", 9}, {"strain", 6},
    {"jacobian", 1}, {"vonmises", 1}, {"energy_density", 1}, {"thickness_stretch", 1}};
  return q;
}

int QuantityComponents(const std::string &name)
{
  for (const QuantityInfo &q : Quantities())
  {
    if (name == q.name) { return q.components; }
  }
  return 0;
}

void PackQuantity(const std::string &name, const QPointState &s, double *out)
{
  if (name == "pk1_stress" || name == "deformation_gradient")
  {
    const tensor<double, 3, 3> &A = name == "pk1_stress" ? s.P : s.F;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) { out[3 * i + j] = A(i, j); }
    return;
  }
  if (name == "jacobian") { out[0] = s.J; return; }
  if (name == "thickness_stretch") { out[0] = s.F(2, 2); return; }
  if (name == "energy_density") { out[0] = s.energy; return; }
  if (name == "vonmises") { out[0] = VonMises(s.sigma); return; }
  // cauchy_stress, strain: symmetric, VTK order
  const tensor<double, 3, 3> &A = name == "strain" ? s.strain : s.sigma;
  out[0] = A(0, 0);
  out[1] = A(1, 1);
  out[2] = A(2, 2);
  out[3] = 0.5 * (A(0, 1) + A(1, 0));
  out[4] = 0.5 * (A(1, 2) + A(2, 1));
  out[5] = 0.5 * (A(0, 2) + A(2, 0));
}

tensor<double, 3, 3> DeformationGradientAt(const mfem::DenseMatrix &grad, int dim)
{
  if (dim == 2)
  {
    tensor<double, 2, 2> H;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) { H(i, j) = grad(i, j); }
    return DeformationGradient<2>(H);
  }
  tensor<double, 3, 3> H;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) { H(i, j) = grad(i, j); }
  return DeformationGradient<3>(H);
}

namespace
{

bool Contains(const std::vector<std::string> &list, const std::string &s)
{
  return std::find(list.begin(), list.end(), s) != list.end();
}

} // namespace

QuadratureFields::QuadratureFields(mfem::ParMesh &mesh, mfem::FiniteElementCollection &h1_fec,
                                   int order, const OutputConfig &out,
                                   const std::vector<std::string> &available)
  : mesh_(mesh), h1_fec_(h1_fec)
{
  nodes_ = Contains(out.quadrature_at, "nodes");
  elements_ = Contains(out.quadrature_at, "elements");
  qpoints_ = Contains(out.quadrature_at, "quadrature_points");
  consistent_ = out.nodal_projection == "projected";
  for (const QuantityInfo &q : Quantities())
  {
    if (!Contains(out.fields, q.name) || !Contains(available, q.name)) { continue; }
    // The kernels integrate with IntRules.Get(geom, 2p + 3); the same rule here.
    if (!qspace_) { qspace_ = std::make_unique<mfem::QuadratureSpace>(&mesh_, 2 * order + 3); }
    Field f;
    f.name = q.name;
    f.nc = q.components;
    f.qf = std::make_unique<mfem::QuadratureFunction>(qspace_.get(), f.nc);
    *f.qf = 0.0;
    if (nodes_)
    {
      f.h1 = std::make_unique<mfem::ParFiniteElementSpace>(&mesh_, &h1_fec_, f.nc,
                                                            mfem::Ordering::byVDIM);
      f.nodal = std::make_unique<mfem::ParGridFunction>(f.h1.get());
      *f.nodal = 0.0;
    }
    if (elements_)
    {
      if (!l2_fec_) { l2_fec_ = std::make_unique<mfem::L2_FECollection>(0, mesh_.Dimension()); }
      f.l2 = std::make_unique<mfem::ParFiniteElementSpace>(&mesh_, l2_fec_.get(), f.nc,
                                                            mfem::Ordering::byVDIM);
      f.elem = std::make_unique<mfem::ParGridFunction>(f.l2.get());
      *f.elem = 0.0;
    }
    fields_.push_back(std::move(f));
  }
  if (nodes_ && consistent_ && !fields_.empty())
  {
    scalar_h1_ = std::make_unique<mfem::ParFiniteElementSpace>(&mesh_, &h1_fec_);
    mfem::ParBilinearForm m(scalar_h1_.get());
    m.AddDomainIntegrator(new mfem::MassIntegrator);
    m.Assemble();
    m.Finalize();
    mass_.reset(m.ParallelAssemble());
    mass_prec_ = std::make_unique<mfem::HypreSmoother>(*mass_, mfem::HypreSmoother::Jacobi);
    mass_solver_ = std::make_unique<mfem::CGSolver>(mesh_.GetComm());
    mass_solver_->SetOperator(*mass_);
    mass_solver_->SetPreconditioner(*mass_prec_);
    mass_solver_->SetRelTol(1e-14);
    mass_solver_->SetAbsTol(0.0);
    mass_solver_->SetMaxIter(1000);
    mass_solver_->SetPrintLevel(0);
  }
}

void QuadratureFields::Fill(const QPointEvaluator &eval)
{
  QPointState s;
  std::vector<mfem::DenseMatrix> views(fields_.size());
  double packed[9];
  for (int e = 0; e < mesh_.GetNE(); e++)
  {
    mfem::ElementTransformation &T = *mesh_.GetElementTransformation(e);
    const mfem::IntegrationRule &ir = qspace_->GetElementIntRule(e);
    for (std::size_t k = 0; k < fields_.size(); k++) { fields_[k].qf->GetValues(e, views[k]); }
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      eval(T, ir.IntPoint(q), s);
      for (std::size_t k = 0; k < fields_.size(); k++)
      {
        PackQuantity(fields_[k].name, s, packed);
        for (int c = 0; c < fields_[k].nc; c++) { views[k](c, q) = packed[c]; }
      }
    }
  }
}

// Quadrature-weighted volume average per element.
void QuadratureFields::ElementAverage(Field &f)
{
  mfem::DenseMatrix V;
  mfem::Array<int> vdofs;
  std::vector<double> sum(f.nc);
  for (int e = 0; e < mesh_.GetNE(); e++)
  {
    mfem::ElementTransformation &T = *mesh_.GetElementTransformation(e);
    const mfem::IntegrationRule &ir = qspace_->GetElementIntRule(e);
    f.qf->GetValues(e, V);
    std::fill(sum.begin(), sum.end(), 0.0);
    double vol = 0.0;
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
      T.SetIntPoint(&ip);
      const double w = ip.weight * T.Weight();
      vol += w;
      for (int c = 0; c < f.nc; c++) { sum[c] += w * V(c, q); }
    }
    f.l2->GetElementVDofs(e, vdofs); // one dof per component
    for (int c = 0; c < f.nc; c++) { (*f.elem)(vdofs[c]) = sum[c] / vol; }
  }
}

// Element-wise L2 projection onto the element's polynomial space, then the
// arithmetic mean of the element values at shared nodes.
void QuadratureFields::ProjectAveraged(Field &f)
{
  mfem::ParFiniteElementSpace &fes = *f.h1;
  mfem::ParGridFunction &g = *f.nodal;
  g = 0.0;
  mfem::Array<int> counts(fes.GetVSize());
  counts = 0;
  mfem::DenseMatrix V, M, B, A;
  mfem::Vector shape;
  mfem::Array<int> vdofs;
  for (int e = 0; e < mesh_.GetNE(); e++)
  {
    const mfem::FiniteElement &fe = *fes.GetFE(e);
    const int nd = fe.GetDof();
    mfem::ElementTransformation &T = *mesh_.GetElementTransformation(e);
    const mfem::IntegrationRule &ir = qspace_->GetElementIntRule(e);
    f.qf->GetValues(e, V);
    M.SetSize(nd);
    M = 0.0;
    B.SetSize(nd, f.nc);
    B = 0.0;
    shape.SetSize(nd);
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const mfem::IntegrationPoint &ip = ir.IntPoint(q);
      T.SetIntPoint(&ip);
      const double w = ip.weight * T.Weight();
      fe.CalcShape(ip, shape);
      for (int i = 0; i < nd; i++)
      {
        for (int j = 0; j < nd; j++) { M(i, j) += w * shape(i) * shape(j); }
        for (int c = 0; c < f.nc; c++) { B(i, c) += w * shape(i) * V(c, q); }
      }
    }
    mfem::DenseMatrixInverse Minv(M);
    A.SetSize(nd, f.nc);
    Minv.Mult(B, A);
    fes.GetElementVDofs(e, vdofs); // vdofs[i + c nd]
    for (int c = 0; c < f.nc; c++)
      for (int i = 0; i < nd; i++)
      {
        const int k = vdofs[i + c * nd];
        g(k) += A(i, c);
        counts[k] += 1;
      }
  }
  mfem::GroupCommunicator &gcomm = fes.GroupComm();
  gcomm.Reduce<int>(counts, mfem::GroupCommunicator::Sum);
  gcomm.Bcast(counts);
  gcomm.Reduce<mfem::real_t>(g.HostReadWrite(), mfem::GroupCommunicator::Sum);
  gcomm.Bcast<mfem::real_t>(g.HostReadWrite());
  for (int k = 0; k < g.Size(); k++)
  {
    if (counts[k] > 0) { g(k) /= counts[k]; }
  }
}

// Global L2 projection: M a = int N f dV with the quadrature data as f,
// one scalar solve per component.
void QuadratureFields::ProjectConsistent(Field &f)
{
  mfem::ParFiniteElementSpace &sfes = *scalar_h1_;
  mfem::ParFiniteElementSpace &vfes = *f.h1;
  const int nl = sfes.GetVSize();
  mfem::Vector bl(nl), bt(sfes.GetTrueVSize()), at(sfes.GetTrueVSize());
  mfem::ParGridFunction tmp(&sfes);
  mfem::DenseMatrix V;
  mfem::Vector shape;
  mfem::Array<int> dofs;
  for (int c = 0; c < f.nc; c++)
  {
    bl = 0.0;
    for (int e = 0; e < mesh_.GetNE(); e++)
    {
      const mfem::FiniteElement &fe = *sfes.GetFE(e);
      const int nd = fe.GetDof();
      mfem::ElementTransformation &T = *mesh_.GetElementTransformation(e);
      const mfem::IntegrationRule &ir = qspace_->GetElementIntRule(e);
      f.qf->GetValues(e, V);
      sfes.GetElementDofs(e, dofs);
      shape.SetSize(nd);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
        const mfem::IntegrationPoint &ip = ir.IntPoint(q);
        T.SetIntPoint(&ip);
        const double w = ip.weight * T.Weight();
        fe.CalcShape(ip, shape);
        for (int i = 0; i < nd; i++) { bl(dofs[i]) += w * shape(i) * V(c, q); }
      }
    }
    sfes.GetProlongationMatrix()->MultTranspose(bl, bt);
    at = 0.0;
    mass_solver_->Mult(bt, at);
    tmp.SetFromTrueDofs(at);
    for (int i = 0; i < nl; i++) { (*f.nodal)(vfes.DofToVDof(i, c)) = tmp(i); }
  }
}

void QuadratureFields::Update(const QPointEvaluator &eval)
{
  if (fields_.empty()) { return; }
  Fill(eval);
  for (Field &f : fields_)
  {
    if (f.elem) { ElementAverage(f); }
    if (f.nodal)
    {
      if (consistent_) { ProjectConsistent(f); }
      else { ProjectAveraged(f); }
    }
  }
}

void QuadratureFields::Register(FieldRegistry &registry)
{
  for (Field &f : fields_)
  {
    if (f.nodal) { registry.AddExternal(f.name, *f.nodal); }
    if (f.elem) { registry.AddExternal(f.name + "_elem", *f.elem); }
    if (qpoints_) { registry.AddExternalQ(f.name + "_qp", *f.qf); }
  }
}

} // namespace cmf

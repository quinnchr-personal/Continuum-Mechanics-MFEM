#include "physics/scalar_conditions.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace cmf
{

ScalarConditions::ScalarConditions(mfem::ParFiniteElementSpace &fes)
  : fes_(fes), dim_(fes.GetParMesh()->Dimension())
{
  MFEM_VERIFY(fes.GetVDim() == 1, "ScalarConditions: the space must have one component");
}

mfem::Array<int> ScalarConditions::Marker(const std::vector<int> &attrs) const
{
  const mfem::Mesh &mesh = *fes_.GetMesh();
  const int max_attr = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
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

void ScalarConditions::AddDirichlet(const std::vector<int> &attrs, mfem::Coefficient &g,
                                    const BCOptions &opt)
{
  if (!opt.components.empty())
  {
    throw ConfigError("AddDirichlet: a scalar unknown has no components");
  }
  if (!opt.point.empty() && int(opt.point.size()) != dim_)
  {
    throw ConfigError("AddDirichlet: point has " + std::to_string(opt.point.size()) +
                      " coordinates, expected " + std::to_string(dim_));
  }
  DirichletEntry e;
  e.marker = opt.point.empty() ? Marker(attrs) : Marker({});
  e.coef = &g;
  e.opt = opt;
  if (e.opt.name.empty()) { e.opt.name = "dirichlet[" + std::to_string(dirichlet_.size()) + "]"; }
  dirichlet_.push_back(std::move(e));
  finalized_ = false;
}

void ScalarConditions::AddFlux(const std::vector<int> &attrs, mfem::Coefficient &g,
                               const BCOptions &opt)
{
  FluxEntry e;
  e.marker = Marker(attrs);
  e.coef = &g;
  e.opt = opt;
  if (e.opt.name.empty()) { e.opt.name = "flux[" + std::to_string(flux_.size()) + "]"; }
  flux_.push_back(std::move(e));
  finalized_ = false;
}

void ScalarConditions::Clear()
{
  dirichlet_.clear();
  flux_.clear();
  finalized_ = false;
}

void ScalarConditions::Assemble(FluxEntry &e) const
{
  mfem::ParLinearForm load(&fes_);
  load.AddBoundaryIntegrator(new mfem::BoundaryLFIntegrator(*e.coef), e.marker);
  load.Assemble();
  e.L.SetSize(fes_.GetTrueVSize());
  load.ParallelAssemble(e.L);
}

void ScalarConditions::WarnOverlaps() const
{
  int rank = 0;
  MPI_Comm_rank(fes_.GetComm(), &rank);
  if (rank != 0) { return; }
  for (std::size_t i = 0; i < dirichlet_.size(); i++)
    for (std::size_t j = i + 1; j < dirichlet_.size(); j++)
      for (int a = 0; a < ess_marker_.Size(); a++)
      {
        if (dirichlet_[i].marker[a] && dirichlet_[j].marker[a])
        {
          std::printf("warning: Dirichlet entries %zu and %zu both prescribe the unknown on boundary "
                      "attribute %d; the later entry wins where they overlap\n", i, j, a + 1);
        }
      }
}

void ScalarConditions::Finalize()
{
  const mfem::Mesh &mesh = *fes_.GetMesh();
  const int max_attr = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
  ess_marker_.SetSize(max_attr);
  ess_marker_ = 0;
  ess_tdof_list_.SetSize(0);
  for (DirichletEntry &e : dirichlet_)
  {
    for (int i = 0; i < max_attr; i++) { ess_marker_[i] |= e.marker[i]; }
    e.tdofs.SetSize(0);
    if (e.IsPoint()) { ResolvePoint(e); }
    else { fes_.GetEssentialTrueDofs(e.marker, e.tdofs); }
    ess_tdof_list_.Append(e.tdofs);
  }
  ess_tdof_list_.Sort();
  ess_tdof_list_.Unique();
  WarnOverlaps();
  for (FluxEntry &e : flux_)
  {
    if (!e.opt.time_dependent) { Assemble(e); }
  }
  external_.SetSize(fes_.GetTrueVSize());
  external_ = 0.0;
  finalized_ = true;
  SetTime(time_);
}

// The node of the space nearest to the point (MPI_MINLOC over the distance);
// the point must be a node (within 1e-8 of the mesh diameter).
void ScalarConditions::ResolvePoint(DirichletEntry &e) const
{
  EnsureCoords();
  const int n_nodes = coords_true_.Size() / dim_;
  int rank = 0;
  MPI_Comm_rank(fes_.GetComm(), &rank);
  struct { double d; int rank; } local = {std::numeric_limits<double>::infinity(), rank}, global;
  int best = -1;
  double lo[3] = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                  std::numeric_limits<double>::infinity()};
  double hi[3] = {-lo[0], -lo[1], -lo[2]};
  for (int n = 0; n < n_nodes; n++)
  {
    double d2 = 0.0;
    for (int c = 0; c < dim_; c++)
    {
      const double x = coords_true_(n * dim_ + c);
      d2 += (x - e.opt.point[std::size_t(c)]) * (x - e.opt.point[std::size_t(c)]);
      lo[c] = std::min(lo[c], x);
      hi[c] = std::max(hi[c], x);
    }
    if (d2 < local.d) { local.d = d2; best = n; }
  }
  local.d = std::sqrt(local.d);
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MINLOC, fes_.GetComm());
  double glo[3], ghi[3];
  MPI_Allreduce(lo, glo, 3, MPI_DOUBLE, MPI_MIN, fes_.GetComm());
  MPI_Allreduce(hi, ghi, 3, MPI_DOUBLE, MPI_MAX, fes_.GetComm());
  double diam2 = 0.0;
  for (int c = 0; c < dim_; c++) { diam2 += (ghi[c] - glo[c]) * (ghi[c] - glo[c]); }
  if (global.d > 1e-8 * std::sqrt(diam2))
  {
    std::string pt;
    for (int c = 0; c < dim_; c++) { pt += (c ? ", " : "") + std::to_string(e.opt.point[std::size_t(c)]); }
    throw ConfigError(e.opt.name + ": the point (" + pt + ") is not a node of the mesh (nearest node at distance " +
                      std::to_string(global.d) + ")");
  }
  if (global.rank != rank || best < 0) { return; }
  e.tdofs.Append(best);
}

void ScalarConditions::SetTime(double t)
{
  if (!finalized_) { Finalize(); }
  time_ = t;
  for (DirichletEntry &e : dirichlet_) { e.coef->SetTime(t); }
  external_ = 0.0;
  for (FluxEntry &e : flux_)
  {
    e.coef->SetTime(t);
    if (e.opt.time_dependent) { Assemble(e); }
    external_.Add(e.opt.schedule.Eval(t, physical_time_), e.L);
  }
}

void ScalarConditions::ApplyDirichlet(mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "ScalarConditions: call Finalize() first");
  mfem::ParGridFunction g(&fes_);
  mfem::Vector g_true(fes_.GetTrueVSize());
  for (const DirichletEntry &e : dirichlet_)
  {
    const double s = e.opt.schedule.Eval(time_, physical_time_);
    g = 0.0;
    if (e.IsPoint()) { g.ProjectCoefficient(*e.coef); }
    else
    {
      mfem::Array<int> marker(e.marker);
      g.ProjectBdrCoefficient(*e.coef, marker);
    }
    g.GetTrueDofs(g_true);
    for (int i = 0; i < e.tdofs.Size(); i++) { x(e.tdofs[i]) = s * g_true(e.tdofs[i]); }
  }
}

void ScalarConditions::EnsureCoords() const
{
  if (coords_true_.Size() != 0) { return; }
  // The node coordinates as a vector field on a vector space over the same
  // collection, ordered by vdim: true dof n * dim + c is node n's coordinate c,
  // node n being the scalar true dof n.
  mfem::ParFiniteElementSpace vfes(fes_.GetParMesh(), fes_.FEColl(), dim_, mfem::Ordering::byVDIM);
  mfem::VectorFunctionCoefficient X(dim_, [](const mfem::Vector &p, mfem::Vector &v) { v = p; });
  mfem::ParGridFunction g(&vfes);
  g.ProjectCoefficient(X);
  coords_true_.SetSize(vfes.GetTrueVSize());
  g.GetTrueDofs(coords_true_);
}

std::vector<Flow> ScalarConditions::Flows(const mfem::Vector &r) const
{
  MFEM_VERIFY(finalized_, "ScalarConditions: call Finalize() first");
  std::vector<Flow> out;
  for (const DirichletEntry &e : dirichlet_)
  {
    double local = 0.0, global = 0.0;
    for (int k = 0; k < e.tdofs.Size(); k++) { local += r(e.tdofs[k]); }
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, fes_.GetComm());
    Flow f;
    f.name = e.opt.name;
    f.value = global;
    out.push_back(f);
  }
  return out;
}

} // namespace cmf

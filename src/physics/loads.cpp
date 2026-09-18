#include "physics/loads.hpp"

#include <algorithm>
#include <cstdio>

namespace cmf
{

LoadSet::LoadSet(mfem::ParFiniteElementSpace &fes)
  : fes_(fes), dim_(fes.GetVDim()) {}

mfem::Array<int> LoadSet::Marker(const std::vector<int> &attrs) const
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

void LoadSet::CheckVector(mfem::VectorCoefficient &c, const std::string &what) const
{
  if (c.GetVDim() != dim_)
  {
    throw ConfigError(what + ": coefficient has " + std::to_string(c.GetVDim()) +
                      " components, expected " + std::to_string(dim_));
  }
}

void LoadSet::CheckComponents(const std::vector<int> &components,
                              const std::string &what) const
{
  for (int c : components)
  {
    if (c < 0 || c >= dim_)
    {
      throw ConfigError(what + ": component " + std::to_string(c) +
                        " is out of range for dimension " + std::to_string(dim_));
    }
  }
}

void LoadSet::AddDirichlet(const std::vector<int> &attrs, mfem::VectorCoefficient &u_bar,
                           const BCOptions &opt)
{
  CheckVector(u_bar, "AddDirichlet");
  CheckComponents(opt.components, "AddDirichlet");
  DirichletEntry e;
  e.marker = Marker(attrs);
  e.coef = &u_bar;
  e.opt = opt;
  if (e.opt.name.empty()) { e.opt.name = "dirichlet[" + std::to_string(dirichlet_.size()) + "]"; }
  dirichlet_.push_back(std::move(e));
  finalized_ = false;
}

void LoadSet::AddTraction(const std::vector<int> &attrs, mfem::VectorCoefficient &T_bar,
                          const BCOptions &opt)
{
  CheckVector(T_bar, "AddTraction");
  LoadEntry e;
  e.marker = Marker(attrs);
  e.vcoef = &T_bar;
  e.opt = opt;
  loads_.push_back(std::move(e));
  finalized_ = false;
}

void LoadSet::AddPressure(const std::vector<int> &attrs, mfem::Coefficient &p,
                          const BCOptions &opt)
{
  LoadEntry e;
  e.marker = Marker(attrs);
  e.scoef = &p;
  e.opt = opt;
  // T = -p N: the flux integrator integrates f (w . N) with f = -p.
  e.owned_scalar = std::make_unique<mfem::ProductCoefficient>(-1.0, p);
  loads_.push_back(std::move(e));
  finalized_ = false;
}

const double *LoadSet::AddFollowerPressure(const std::vector<int> &attrs, mfem::Coefficient &p,
                                           const BCOptions &opt)
{
  FollowerEntry e;
  e.marker = Marker(attrs);
  e.coef = &p;
  e.opt = opt;
  e.scale = std::make_unique<double>(0.0);
  follower_.push_back(std::move(e));
  finalized_ = false;
  return follower_.back().scale.get();
}

void LoadSet::SetBodyForce(mfem::VectorCoefficient &b, mfem::Coefficient &rho, const BCOptions &opt)
{
  CheckVector(b, "SetBodyForce");
  // One body force at a time (the YAML has one key); replace an earlier one.
  for (std::size_t i = 0; i < loads_.size(); i++)
  {
    if (loads_[i].body) { loads_.erase(loads_.begin() + long(i)); break; }
  }
  LoadEntry e;
  e.vcoef = &b;
  e.body = true;
  e.opt = opt;
  e.owned_vector = std::make_unique<mfem::ScalarVectorProductCoefficient>(rho, b);
  loads_.push_back(std::move(e));
  finalized_ = false;
}

void LoadSet::Clear()
{
  dirichlet_.clear();
  loads_.clear();
  follower_.clear();
  finalized_ = false;
}

void LoadSet::Assemble(LoadEntry &e) const
{
  mfem::ParLinearForm load(&fes_);
  if (e.body)
  {
    load.AddDomainIntegrator(new mfem::VectorDomainLFIntegrator(*e.owned_vector));
  }
  else if (e.scoef)
  {
    load.AddBoundaryIntegrator(new mfem::VectorBoundaryFluxLFIntegrator(*e.owned_scalar),
                               e.marker);
  }
  else
  {
    load.AddBoundaryIntegrator(new mfem::VectorBoundaryLFIntegrator(*e.vcoef), e.marker);
  }
  load.Assemble();
  e.L.SetSize(fes_.GetTrueVSize());
  load.ParallelAssemble(e.L);
}

void LoadSet::WarnOverlaps() const
{
  int rank = 0;
  MPI_Comm_rank(fes_.GetComm(), &rank);
  if (rank != 0) { return; }
  auto has = [](const DirichletEntry &e, int c)
  {
    return e.opt.components.empty() ||
           std::find(e.opt.components.begin(), e.opt.components.end(), c) !=
             e.opt.components.end();
  };
  for (std::size_t i = 0; i < dirichlet_.size(); i++)
    for (std::size_t j = i + 1; j < dirichlet_.size(); j++)
      for (int a = 0; a < ess_marker_.Size(); a++)
      {
        if (!dirichlet_[i].marker[a] || !dirichlet_[j].marker[a]) { continue; }
        for (int c = 0; c < dim_; c++)
        {
          if (has(dirichlet_[i], c) && has(dirichlet_[j], c))
          {
            std::printf("warning: Dirichlet entries %zu and %zu both prescribe component %d "
                        "on boundary attribute %d; the later entry wins where they overlap\n",
                        i, j, c, a + 1);
          }
        }
      }
}

void LoadSet::Finalize()
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
    if (e.opt.components.empty())
    {
      fes_.GetEssentialTrueDofs(e.marker, e.tdofs);
    }
    else
    {
      for (int c : e.opt.components)
      {
        mfem::Array<int> list;
        fes_.GetEssentialTrueDofs(e.marker, list, c);
        e.tdofs.Append(list);
      }
    }
    ess_tdof_list_.Append(e.tdofs);
  }
  ess_tdof_list_.Sort();
  ess_tdof_list_.Unique();
  WarnOverlaps();

  for (LoadEntry &e : loads_)
  {
    if (!e.opt.time_dependent) { Assemble(e); }
  }
  external_.SetSize(fes_.GetTrueVSize());
  external_ = 0.0;
  finalized_ = true;
  SetTime(time_);
}

void LoadSet::SetTime(double t)
{
  if (!finalized_) { Finalize(); }
  time_ = t;
  for (DirichletEntry &e : dirichlet_) { e.coef->SetTime(t); }
  for (FollowerEntry &e : follower_)
  {
    e.coef->SetTime(t);
    *e.scale = e.opt.schedule.Eval(t, physical_time_);
  }
  external_ = 0.0;
  for (LoadEntry &e : loads_)
  {
    if (e.vcoef) { e.vcoef->SetTime(t); }
    if (e.scoef) { e.scoef->SetTime(t); }
    if (e.owned_scalar) { e.owned_scalar->SetTime(t); }
    if (e.owned_vector) { e.owned_vector->SetTime(t); }
    if (e.opt.time_dependent) { Assemble(e); }
    external_.Add(e.opt.schedule.Eval(t, physical_time_), e.L);
  }
}

void LoadSet::ApplyDirichlet(mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "LoadSet: call Finalize() first");
  // ParGridFunction takes a non-const space pointer; nothing is modified.
  mfem::ParGridFunction g(&fes_);
  mfem::Vector g_true(fes_.GetTrueVSize());
  for (const DirichletEntry &e : dirichlet_)
  {
    const double s = e.opt.schedule.Eval(time_, physical_time_);
    g = 0.0;
    mfem::Array<int> marker(e.marker); // MFEM takes a non-const marker
    g.ProjectBdrCoefficient(*e.coef, marker);
    g.GetTrueDofs(g_true);
    for (int i = 0; i < e.tdofs.Size(); i++) { x(e.tdofs[i]) = s * g_true(e.tdofs[i]); }
  }
}

std::vector<Reaction> LoadSet::Reactions(const mfem::Vector &r, const mfem::Vector &x) const
{
  MFEM_VERIFY(finalized_, "LoadSet: call Finalize() first");
  if (coords_true_.Size() == 0)
  {
    // The reference coordinates as a vector field on the displacement space.
    mfem::VectorFunctionCoefficient X(dim_, [](const mfem::Vector &p, mfem::Vector &v) { v = p; });
    mfem::ParGridFunction g(&fes_);
    g.ProjectCoefficient(X);
    coords_true_.SetSize(fes_.GetTrueVSize());
    g.GetTrueDofs(coords_true_);
  }
  // True dofs are ordered by vdim (all components of a node consecutive).
  std::vector<Reaction> out;
  for (const DirichletEntry &e : dirichlet_)
  {
    double local[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < e.tdofs.Size(); k++)
    {
      const int i = e.tdofs[k];
      const int node = i / dim_, c = i % dim_;
      const double f = r(i);
      double pos[3] = {0.0, 0.0, 0.0}, force[3] = {0.0, 0.0, 0.0};
      for (int d = 0; d < dim_; d++) { pos[d] = coords_true_(node * dim_ + d) + x(node * dim_ + d); }
      force[c] = f;
      local[c] += f;
      local[3] += pos[1] * force[2] - pos[2] * force[1];
      local[4] += pos[2] * force[0] - pos[0] * force[2];
      local[5] += pos[0] * force[1] - pos[1] * force[0];
    }
    double global[6];
    MPI_Allreduce(local, global, 6, MPI_DOUBLE, MPI_SUM, fes_.GetComm());
    Reaction rx;
    rx.name = e.opt.name;
    for (int d = 0; d < 3; d++) { rx.force[d] = global[d]; rx.moment[d] = global[3 + d]; }
    out.push_back(rx);
  }
  return out;
}

} // namespace cmf

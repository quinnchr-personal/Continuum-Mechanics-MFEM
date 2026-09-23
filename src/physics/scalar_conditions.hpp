// Boundary conditions of a scalar unknown (physics/scalar_transport.hpp):
// Dirichlet entries (faces by attribute, or the node nearest to a point),
// each with a coefficient g(x, t) and a schedule, and flux entries, the
// prescribed inward flux g = F . n per unit area of the faces (positive into
// the domain; the natural condition is F . n = 0), assembled with MFEM's
// BoundaryLFIntegrator into one true vector per entry and summed with the
// schedules into the external vector the physics subtracts from its residual.
// The flow through a Dirichlet entry is the sum of the full residual over its
// dofs: for the exact solution that residual is the boundary integral of
// N_j F . n, so the sum is the flux entering the domain through the entry's
// faces (the scalar analogue of the reactions of physics/loads.hpp). The
// options of an entry are those of BCOptions (loads.hpp): schedule, time
// dependence, name, point; components are not used.
#pragma once

#include <string>
#include <vector>

#include "base/config.hpp"
#include "mfem.hpp"
#include "physics/loads.hpp"

namespace cmf
{

struct Flow
{
  std::string name;
  double value = 0.0; // into the domain
};

class ScalarConditions
{
public:
  explicit ScalarConditions(mfem::ParFiniteElementSpace &fes);

  // Coefficients are not owned and must outlive this object. Every call
  // invalidates Finalize().
  void AddDirichlet(const std::vector<int> &attrs, mfem::Coefficient &g,
                    const BCOptions &opt = BCOptions());
  void AddFlux(const std::vector<int> &attrs, mfem::Coefficient &g,
               const BCOptions &opt = BCOptions());
  void Clear();

  // Essential true dofs and their marker union, the vectors of the flux
  // entries that do not depend on t, overlap warnings.
  void Finalize();
  bool Finalized() const { return finalized_; }

  // Evaluates the schedules, calls SetTime on every coefficient, reassembles
  // the time-dependent flux vectors and caches the scheduled sum.
  void SetTime(double t);
  double Time() const { return time_; }
  // Physical time: the schedules take their right limit at t = 0.
  void SetPhysicalTime(bool on) { physical_time_ = on; }
  bool PhysicalTime() const { return physical_time_; }

  // Overwrite the essential true dofs of x with schedule(t) * projected data.
  void ApplyDirichlet(mfem::Vector &x) const;
  // The flow into the domain through every Dirichlet entry from the full
  // residual r (essential rows kept); collective.
  std::vector<Flow> Flows(const mfem::Vector &r) const;
  // The scheduled external vector sum_i s_i(t) L_i (true dofs).
  const mfem::Vector &ExternalLoad() const { return external_; }

  const mfem::Array<int> &EssentialTrueDofs() const { return ess_tdof_list_; }
  const mfem::Array<int> &EssentialMarker() const { return ess_marker_; }
  std::size_t NumDirichlet() const { return dirichlet_.size(); }
  const std::string &DirichletName(std::size_t i) const { return dirichlet_[i].opt.name; }
  std::size_t NumFlux() const { return flux_.size(); }

  // Attribute marker of the mesh (size max attribute) with attrs set.
  mfem::Array<int> Marker(const std::vector<int> &attrs) const;

private:
  struct DirichletEntry
  {
    mfem::Array<int> marker;
    mfem::Coefficient *coef;
    BCOptions opt;
    mfem::Array<int> tdofs;
    bool IsPoint() const { return !opt.point.empty(); }
  };
  struct FluxEntry
  {
    mfem::Array<int> marker;
    mfem::Coefficient *coef;
    BCOptions opt;
    mfem::Vector L;
  };

  void Assemble(FluxEntry &e) const;
  void ResolvePoint(DirichletEntry &e) const;
  void WarnOverlaps() const;
  void EnsureCoords() const;

  mfem::ParFiniteElementSpace &fes_;
  int dim_;
  std::vector<DirichletEntry> dirichlet_;
  std::vector<FluxEntry> flux_;
  bool finalized_ = false;
  bool physical_time_ = false;
  double time_ = 1.0;
  mfem::Array<int> ess_marker_;
  mfem::Array<int> ess_tdof_list_;
  mfem::Vector external_;
  mutable mfem::Vector coords_true_; // node coordinates by true dof (n * dim + c), lazily
};

} // namespace cmf

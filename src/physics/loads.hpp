// Boundary conditions and external loads of a displacement space, shared by
// the solid formulations: Dirichlet entries (all or some components, each
// with a schedule), dead loads (vector tractions, normal pressures, body
// force; one assembled true vector per entry) and follower pressures (a
// nonlinear boundary term the owning physics installs in its form; this
// class only tracks their scale). Everything is a function of the
// pseudo-time t through the entry schedules and the coefficients' SetTime.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "base/config.hpp"
#include "mfem.hpp"

namespace cmf
{

// Per-entry options of the programmatic boundary condition API.
struct BCOptions
{
  std::vector<int> components;  // Dirichlet: prescribed components, empty = all
  Schedule schedule;            // default: ramp over [0, 1]
  // Reassemble the dead-load vector every time the pseudo-time changes
  // (coefficients that depend on t). Dirichlet data is always re-projected.
  bool time_dependent = false;
};

class LoadSet
{
public:
  explicit LoadSet(mfem::ParFiniteElementSpace &fes);

  // Coefficients are not owned and must outlive this object. Every call
  // invalidates Finalize().
  void AddDirichlet(const std::vector<int> &attrs, mfem::VectorCoefficient &u_bar,
                    const BCOptions &opt = BCOptions());
  // Nominal traction per unit reference area (dead load).
  void AddTraction(const std::vector<int> &attrs, mfem::VectorCoefficient &T_bar,
                   const BCOptions &opt = BCOptions());
  // Dead normal pressure: T = -p N on the reference outward normal.
  void AddPressure(const std::vector<int> &attrs, mfem::Coefficient &p,
                   const BCOptions &opt = BCOptions());
  // Follower pressure bookkeeping: the returned scale pointer is stable for
  // the life of this object (or until Clear) and holds schedule(t); the
  // physics gives it to the boundary integrator it installs.
  const double *AddFollowerPressure(const std::vector<int> &attrs, mfem::Coefficient &p,
                                    const BCOptions &opt = BCOptions());
  // Body force per unit mass; rho0 b enters the weak form.
  void SetBodyForce(mfem::VectorCoefficient &b, double rho0,
                    const BCOptions &opt = BCOptions());
  void Clear();
  bool HasFollowerPressure() const { return !follower_.empty(); }

  // Essential true dofs and their marker union, dead loads of the entries
  // that do not depend on t, overlap warnings.
  void Finalize();
  bool Finalized() const { return finalized_; }

  // Pseudo-time: evaluates the schedules, calls SetTime on every coefficient,
  // reassembles the time-dependent dead loads and caches the scheduled sum.
  void SetTime(double t);
  double Time() const { return time_; }

  // Overwrite the essential true dofs of x with schedule(t) * projected data.
  void ApplyDirichlet(mfem::Vector &x) const;
  // The scheduled external load sum_i s_i(t) L_i at the current t (true dofs).
  const mfem::Vector &ExternalLoad() const { return external_; }

  const mfem::Array<int> &EssentialTrueDofs() const { return ess_tdof_list_; }
  // Attribute marker: 1 where any Dirichlet entry prescribes any component.
  const mfem::Array<int> &EssentialMarker() const { return ess_marker_; }

  // Attribute marker of the mesh (size max attribute) with attrs set.
  mfem::Array<int> Marker(const std::vector<int> &attrs) const;
  int Dim() const { return dim_; }

private:
  struct DirichletEntry
  {
    mfem::Array<int> marker;
    mfem::VectorCoefficient *coef;
    BCOptions opt;
    mfem::Array<int> tdofs; // essential true dofs of this entry's components
  };
  struct LoadEntry
  {
    mfem::Array<int> marker;
    mfem::VectorCoefficient *vcoef = nullptr; // traction or body force
    mfem::Coefficient *scoef = nullptr;       // pressure
    bool body = false;
    double rho0 = 1.0;
    BCOptions opt;
    std::unique_ptr<mfem::Coefficient> owned_scalar;        // -p for the flux integrator
    std::unique_ptr<mfem::VectorCoefficient> owned_vector;  // rho0 b
    mfem::Vector L;                                         // assembled true vector
  };
  struct FollowerEntry
  {
    mfem::Array<int> marker;
    mfem::Coefficient *coef;
    BCOptions opt;
    std::unique_ptr<double> scale;
  };

  void CheckVector(mfem::VectorCoefficient &c, const std::string &what) const;
  void CheckComponents(const std::vector<int> &components, const std::string &what) const;
  void Assemble(LoadEntry &e) const;
  void WarnOverlaps() const;

  mfem::ParFiniteElementSpace &fes_;
  int dim_;
  std::vector<DirichletEntry> dirichlet_;
  std::vector<LoadEntry> loads_;
  std::vector<FollowerEntry> follower_;
  bool finalized_ = false;
  double time_ = 1.0;
  mfem::Array<int> ess_marker_;
  mfem::Array<int> ess_tdof_list_;
  mfem::Vector external_;
};

} // namespace cmf

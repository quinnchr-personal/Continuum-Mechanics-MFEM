#pragma once

#include "yaml-cpp/yaml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace hycfd
{

// Abstract compressible-flow physics consumed by the HDG operator.
//
// Array conventions (all raw double* with runtime strides so future models
// with more components — e.g. two-temperature thermochemical nonequilibrium —
// slot in without operator changes):
//   ncu   = NumComponents(), dim = Dim(), nflux = ncu*dim,
//   nstate = ncu*(1+dim).
//   uq[nstate]: conservative state u in slots [0,ncu), then the PHYSICAL
//     gradient q = +grad(u) with slot ncu + ncu*d + c = d(u_c)/dx_d.
//   flux[nflux]: flux[c + ncu*d] = F_{c,d}(u, q) (inviscid minus viscous
//     minus artificial-viscosity contribution).
//   dfduq[nflux*nstate]: dfduq[out + nflux*in], in over all uq slots.
//   fb[ncu], dfbduq[ncu*nstate], dfbduh[ncu*ncu]: boundary residual that
//     replaces the trace rows of a boundary face, and its Jacobians.
class PhysicsModel
{
public:
   virtual ~PhysicsModel() = default;

   virtual int NumComponents() const = 0;
   virtual int Dim() const = 0;
   int NumFluxEntries() const { return NumComponents() * Dim(); }
   int NumStateEntries() const
   {
      return NumComponents() * (1 + Dim());
   }

   virtual void FluxValue(const double *uq, double av,
                          double *flux) const = 0;
   // Flux with optional Jacobian (dfduq may be null). The default
   // implementation evaluates the Jacobian by central finite differences of
   // FluxValue — the fallback future models inherit for free; models with
   // analytic Jacobians override it.
   virtual void Flux(const double *uq, double av, double *flux,
                     double *dfduq) const;

   // Registers a boundary condition of the given type (with optional
   // model-specific parameters from the YAML node) and returns a dense
   // bc_id for BoundaryResidual. The registry is per-model so future models
   // can add their own types without operator changes. Unknown types throw
   // listing the supported set.
   virtual int RegisterBoundaryCondition(const std::string &type,
                                         const YAML::Node &params) = 0;
   virtual int NumBoundaryConditions() const = 0;
   // Jacobian pointers may be null (residual-only assembly).
   virtual void BoundaryResidual(int bc_id, const double *uq,
                                 const double *uhat,
                                 const double *unit_normal,
                                 const double *x, double *fb,
                                 double *dfbduq, double *dfbduh) const = 0;

   virtual double MaxWaveSpeed(const double *u) const = 0;
   virtual bool IsAdmissible(const double *u) const = 0;
   virtual void FreestreamState(double *u) const = 0;
   virtual double Pressure(const double *u) const = 0;
   // Nondimensional temperature T/T_inf (freestream value 1).
   virtual double Temperature(const double *u) const = 0;

   // Named pointwise output fields (e.g. rho, u, v, p) for visualization.
   virtual std::vector<std::string> OutputNames() const = 0;
   virtual void Outputs(const double *u, double *values) const = 0;
};

inline void PhysicsModel::Flux(const double *uq, double av, double *flux,
                               double *dfduq) const
{
   FluxValue(uq, av, flux);
   if (!dfduq) { return; }
   constexpr int kMaxState = 64;
   constexpr int kMaxFlux = 32;
   const int n_in = NumStateEntries();
   const int n_out = NumFluxEntries();
   if (n_in > kMaxState || n_out > kMaxFlux)
   {
      throw std::runtime_error(
         "FD flux Jacobian fallback: state exceeds stack buffers");
   }
   std::array<double, kMaxState> perturbed;
   std::array<double, kMaxFlux> plus, minus;
   std::copy(uq, uq + n_in, perturbed.begin());
   for (int in = 0; in < n_in; ++in)
   {
      const double h = 1.0e-6 * std::max(1.0, std::abs(uq[in]));
      perturbed[in] = uq[in] + h;
      FluxValue(perturbed.data(), av, plus.data());
      perturbed[in] = uq[in] - h;
      FluxValue(perturbed.data(), av, minus.data());
      perturbed[in] = uq[in];
      for (int out = 0; out < n_out; ++out)
      {
         dfduq[out + n_out * in] =
            (plus[out] - minus[out]) / (2.0 * h);
      }
   }
}

} // namespace hycfd

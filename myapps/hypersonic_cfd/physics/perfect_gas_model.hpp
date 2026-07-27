#pragma once

#include "physics/perfect_gas.hpp"
#include "physics/physics_model.hpp"

#include <string>
#include <vector>

namespace hycfd
{

// Calorically perfect gas with Sutherland viscosity: wraps the verified
// analytic-Jacobian kernels in physics/perfect_gas.hpp behind the
// PhysicsModel interface (4 components, 2D).
//
// Supported boundary-condition types:
//   freestream             — supersonic farfield Dirichlet on the trace
//   supersonic_outflow     — extrapolation (u - uhat)
//   isothermal_wall        — no-slip isothermal wall; optional Twall_K
//   adiabatic_wall         — no-slip wall, zero normal heat flux
//   slip_wall              — symmetry/inviscid wall: n.uhat = 0,
//                            tangential and energy extrapolation
//   characteristic_farfield— A_n^+(u-uhat) + A_n^-(uinf-uhat), flux split
//                            frozen at the freestream (exact Jacobians)
//   pressure_outlet        — extrapolation with the trace energy pinned to
//                            a target static pressure p (default p_inf)
class PerfectGasModel final : public PhysicsModel
{
public:
   explicit PerfectGasModel(const PerfectGasParams &params)
      : params_(params) {}

   int NumComponents() const override { return 4; }
   int Dim() const override { return 2; }

   void FluxValue(const double *uq, double av,
                  double *flux) const override
   {
      NSFlux(uq, av, params_, flux, nullptr);
   }
   void Flux(const double *uq, double av, double *flux,
             double *dfduq) const override
   {
      NSFlux(uq, av, params_, flux, dfduq);
   }

   int RegisterBoundaryCondition(const std::string &type,
                                 const YAML::Node &params) override;
   int NumBoundaryConditions() const override
   {
      return static_cast<int>(boundary_conditions_.size());
   }
   void BoundaryResidual(int bc_id, const double *uq, const double *uhat,
                         const double *unit_normal, const double *x,
                         double *fb, double *dfbduq,
                         double *dfbduh) const override;

   double MaxWaveSpeed(const double *u) const override;
   bool IsAdmissible(const double *u) const override
   {
      return u[0] > 0.0 && hycfd::Pressure(u, params_) > 0.0;
   }
   void FreestreamState(double *u) const override
   {
      params_.Freestream(u);
   }
   double Pressure(const double *u) const override
   {
      return hycfd::Pressure(u, params_);
   }
   double Temperature(const double *u) const override
   {
      // T/T_inf = gamma M^2 p / rho (internal-energy temperature units).
      return params_.gamma * params_.mach * params_.mach *
             hycfd::Pressure(u, params_) / u[0];
   }

   std::vector<std::string> OutputNames() const override
   {
      return {"rho", "u", "v", "p"};
   }
   void Outputs(const double *u, double *values) const override
   {
      NSVisScalars(u, params_, values);
   }

   const PerfectGasParams &Params() const { return params_; }

private:
   struct BoundarySpec
   {
      int ib = 0;
      // Per-BC parameter copy so e.g. an isothermal wall can override
      // Twall_K without touching the flux-path parameters.
      PerfectGasParams params;
      double outlet_pressure = 0.0;
   };

   void AdiabaticWallResidual(const double *uq, const double *uhat,
                              const double *normal, double *fb,
                              double *dfbduq, double *dfbduh) const;
   void SlipWallResidual(const double *uq, const double *uhat,
                         const double *normal, double *fb,
                         double *dfbduq, double *dfbduh) const;
   void CharacteristicResidual(const double *uq, const double *uhat,
                               const double *normal, double *fb,
                               double *dfbduq, double *dfbduh) const;
   void PressureOutletResidual(const BoundarySpec &spec, const double *uq,
                               const double *uhat, double *fb,
                               double *dfbduq, double *dfbduh) const;

   PerfectGasParams params_;
   std::vector<BoundarySpec> boundary_conditions_;
};

} // namespace hycfd

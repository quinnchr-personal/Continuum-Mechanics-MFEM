#include "physics/perfect_gas_model.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hycfd
{

int PerfectGasModel::RegisterBoundaryCondition(const std::string &type,
                                               const YAML::Node &params)
{
   BoundarySpec spec;
   spec.params = params_;
   if (type == "freestream")
   {
      spec.ib = 1;
   }
   else if (type == "supersonic_outflow")
   {
      spec.ib = 2;
   }
   else if (type == "isothermal_wall")
   {
      spec.ib = 3;
      if (params && params["Twall_K"])
      {
         spec.params.Twall_K = params["Twall_K"].as<double>();
      }
   }
   else if (type == "adiabatic_wall")
   {
      spec.ib = 4;
   }
   else if (type == "slip_wall")
   {
      spec.ib = 5;
   }
   else if (type == "characteristic_farfield")
   {
      spec.ib = 6;
   }
   else if (type == "pressure_outlet")
   {
      spec.ib = 7;
      spec.outlet_pressure =
         1.0 / (spec.params.gamma * spec.params.mach * spec.params.mach);
      if (params && params["p"])
      {
         spec.outlet_pressure = params["p"].as<double>();
      }
   }
   else
   {
      throw std::runtime_error(
         "perfect_gas does not support boundary condition type '" + type +
         "'; supported: freestream, supersonic_outflow, isothermal_wall,"
         " adiabatic_wall, slip_wall, characteristic_farfield,"
         " pressure_outlet");
   }
   boundary_conditions_.push_back(spec);
   return static_cast<int>(boundary_conditions_.size()) - 1;
}

void PerfectGasModel::BoundaryResidual(
   int bc_id, const double *uq, const double *uhat,
   const double *unit_normal, const double * /*x*/, double *fb,
   double *dfbduq, double *dfbduh) const
{
   if (bc_id < 0 ||
       bc_id >= static_cast<int>(boundary_conditions_.size()))
   {
      throw std::out_of_range("unregistered boundary-condition id");
   }
   const BoundarySpec &spec =
      boundary_conditions_[static_cast<std::size_t>(bc_id)];
   switch (spec.ib)
   {
      case 1:
      case 2:
      case 3:
         NSFbouHdg(spec.ib, uq, uhat, unit_normal, spec.params, fb,
                   dfbduq, dfbduh);
         break;
      case 4:
         AdiabaticWallResidual(uq, uhat, unit_normal, fb,
                               dfbduq, dfbduh);
         break;
      case 5:
         SlipWallResidual(uq, uhat, unit_normal, fb, dfbduq, dfbduh);
         break;
      case 6:
         CharacteristicResidual(uq, uhat, unit_normal, fb,
                                dfbduq, dfbduh);
         break;
      case 7:
         PressureOutletResidual(spec, uq, uhat, fb, dfbduq, dfbduh);
         break;
      default:
         throw std::runtime_error("unhandled boundary-condition id");
   }
}

void PerfectGasModel::AdiabaticWallResidual(
   const double *uq, const double *uhat, const double *normal,
   double *fb, double *dfbduq, double *dfbduh) const
{
   // Density extrapolation and no-slip momentum; the adiabatic condition
   // is imposed on the NUMERICAL energy flux (the standard HDG Neumann
   // treatment, well-posed through the tau stabilization):
   //   fhat_E . n = F_E(uhat, q) . n + tau (u_E - uhat_E) = 0.
   // With uhat momentum pinned to zero the convective and viscous-work
   // parts vanish at the solution, leaving conduction + tau jump = 0 —
   // exactly the Fint wall heat flux the postprocessor reports.
   fb[0] = uq[0] - uhat[0];
   fb[1] = -uhat[1];
   fb[2] = -uhat[2];
   double trace_uq[12], flux[8], flux_jacobian[96];
   std::copy(uq, uq + 12, trace_uq);
   std::copy(uhat, uhat + 4, trace_uq);
   NSFlux(trace_uq, 0.0, params_, flux,
          (dfbduq || dfbduh) ? flux_jacobian : nullptr);
   const double nx = normal[0];
   const double ny = normal[1];
   const double tau = params_.tau;
   fb[3] = flux[3] * nx + flux[7] * ny + tau * (uq[3] - uhat[3]);
   if (dfbduq)
   {
      std::fill(dfbduq, dfbduq + 48, 0.0);
      dfbduq[0 + 4 * 0] = 1.0;
      dfbduq[3 + 4 * 3] = tau;
      for (int input = 4; input < 12; ++input)
      {
         dfbduq[3 + 4 * input] =
            nx * flux_jacobian[3 + 8 * input] +
            ny * flux_jacobian[7 + 8 * input];
      }
   }
   if (dfbduh)
   {
      std::fill(dfbduh, dfbduh + 16, 0.0);
      dfbduh[0 + 4 * 0] = -1.0;
      dfbduh[1 + 4 * 1] = -1.0;
      dfbduh[2 + 4 * 2] = -1.0;
      for (int input = 0; input < 4; ++input)
      {
         dfbduh[3 + 4 * input] =
            nx * flux_jacobian[3 + 8 * input] +
            ny * flux_jacobian[7 + 8 * input] -
            (input == 3 ? tau : 0.0);
      }
   }
}

void PerfectGasModel::SlipWallResidual(
   const double *uq, const double *uhat, const double *normal,
   double *fb, double *dfbduq, double *dfbduh) const
{
   const double nx = normal[0];
   const double ny = normal[1];
   // Density and energy extrapolation; momentum rows enforce n.uhat = 0
   // and tangential extrapolation.
   fb[0] = uq[0] - uhat[0];
   fb[1] = -(nx * uhat[1] + ny * uhat[2]);
   fb[2] = (-ny * uq[1] + nx * uq[2]) - (-ny * uhat[1] + nx * uhat[2]);
   fb[3] = uq[3] - uhat[3];
   if (dfbduq)
   {
      std::fill(dfbduq, dfbduq + 48, 0.0);
      dfbduq[0 + 4 * 0] = 1.0;
      dfbduq[2 + 4 * 1] = -ny;
      dfbduq[2 + 4 * 2] = nx;
      dfbduq[3 + 4 * 3] = 1.0;
   }
   if (dfbduh)
   {
      std::fill(dfbduh, dfbduh + 16, 0.0);
      dfbduh[0 + 4 * 0] = -1.0;
      dfbduh[1 + 4 * 1] = -nx;
      dfbduh[1 + 4 * 2] = -ny;
      dfbduh[2 + 4 * 1] = ny;
      dfbduh[2 + 4 * 2] = -nx;
      dfbduh[3 + 4 * 3] = -1.0;
   }
}

void PerfectGasModel::CharacteristicResidual(
   const double *uq, const double *uhat, const double *normal,
   double *fb, double *dfbduq, double *dfbduh) const
{
   // Inviscid flux-Jacobian splitting frozen at the freestream state:
   //   fb = A_n^+ (u - uhat) + A_n^- (uinf - uhat),
   // which reduces to supersonic inflow/outflow in the limits and admits
   // exact, state-independent Jacobians.
   double uinf[4];
   params_.Freestream(uinf);
   const double rho = uinf[0];
   const double u = uinf[1] / rho;
   const double v = uinf[2] / rho;
   const double p = 1.0 / (params_.gamma * params_.mach * params_.mach);
   const double c = std::sqrt(params_.gamma * p / rho);
   const double ke = 0.5 * (u * u + v * v);
   const double enthalpy = (uinf[3] + p) / rho;
   const double nx = normal[0];
   const double ny = normal[1];
   const double vn = u * nx + v * ny;
   const double vt = -u * ny + v * nx;

   const double lambda[4] = {vn - c, vn, vn, vn + c};
   mfem::DenseMatrix right(4, 4);
   right(0, 0) = 1.0;
   right(1, 0) = u - c * nx;
   right(2, 0) = v - c * ny;
   right(3, 0) = enthalpy - c * vn;
   right(0, 1) = 1.0;
   right(1, 1) = u;
   right(2, 1) = v;
   right(3, 1) = ke;
   right(0, 2) = 0.0;
   right(1, 2) = -ny;
   right(2, 2) = nx;
   right(3, 2) = vt;
   right(0, 3) = 1.0;
   right(1, 3) = u + c * nx;
   right(2, 3) = v + c * ny;
   right(3, 3) = enthalpy + c * vn;
   mfem::DenseMatrix left(right);
   left.Invert();

   mfem::DenseMatrix a_plus(4, 4), a_minus(4, 4);
   for (int row = 0; row < 4; ++row)
   {
      for (int column = 0; column < 4; ++column)
      {
         double plus = 0.0, minus = 0.0;
         for (int wave = 0; wave < 4; ++wave)
         {
            const double positive =
               0.5 * (lambda[wave] + std::abs(lambda[wave]));
            const double negative =
               0.5 * (lambda[wave] - std::abs(lambda[wave]));
            plus += right(row, wave) * positive * left(wave, column);
            minus += right(row, wave) * negative * left(wave, column);
         }
         a_plus(row, column) = plus;
         a_minus(row, column) = minus;
      }
   }

   for (int row = 0; row < 4; ++row)
   {
      fb[row] = 0.0;
      for (int column = 0; column < 4; ++column)
      {
         fb[row] += a_plus(row, column) * (uq[column] - uhat[column]) +
                    a_minus(row, column) * (uinf[column] - uhat[column]);
      }
   }
   if (dfbduq)
   {
      std::fill(dfbduq, dfbduq + 48, 0.0);
      for (int row = 0; row < 4; ++row)
      {
         for (int column = 0; column < 4; ++column)
         {
            dfbduq[row + 4 * column] = a_plus(row, column);
         }
      }
   }
   if (dfbduh)
   {
      for (int row = 0; row < 4; ++row)
      {
         for (int column = 0; column < 4; ++column)
         {
            dfbduh[row + 4 * column] =
               -(a_plus(row, column) + a_minus(row, column));
         }
      }
   }
}

void PerfectGasModel::PressureOutletResidual(
   const BoundarySpec &spec, const double *uq, const double *uhat,
   double *fb, double *dfbduq, double *dfbduh) const
{
   // Mass/momentum extrapolation; the trace energy is pinned to the
   // target static pressure evaluated with the trace density/momentum.
   const double gam1 = spec.params.gamma - 1.0;
   const double inv_rho = 1.0 / uhat[0];
   const double momentum_sq =
      uhat[1] * uhat[1] + uhat[2] * uhat[2];
   fb[0] = uq[0] - uhat[0];
   fb[1] = uq[1] - uhat[1];
   fb[2] = uq[2] - uhat[2];
   fb[3] = spec.outlet_pressure / gam1 +
           0.5 * momentum_sq * inv_rho - uhat[3];
   if (dfbduq)
   {
      std::fill(dfbduq, dfbduq + 48, 0.0);
      dfbduq[0 + 4 * 0] = 1.0;
      dfbduq[1 + 4 * 1] = 1.0;
      dfbduq[2 + 4 * 2] = 1.0;
   }
   if (dfbduh)
   {
      std::fill(dfbduh, dfbduh + 16, 0.0);
      dfbduh[0 + 4 * 0] = -1.0;
      dfbduh[1 + 4 * 1] = -1.0;
      dfbduh[2 + 4 * 2] = -1.0;
      dfbduh[3 + 4 * 0] = -0.5 * momentum_sq * inv_rho * inv_rho;
      dfbduh[3 + 4 * 1] = uhat[1] * inv_rho;
      dfbduh[3 + 4 * 2] = uhat[2] * inv_rho;
      dfbduh[3 + 4 * 3] = -1.0;
   }
}

double PerfectGasModel::MaxWaveSpeed(const double *u) const
{
   const double inv_rho = 1.0 / u[0];
   const double speed =
      std::hypot(u[1] * inv_rho, u[2] * inv_rho);
   const double pressure = hycfd::Pressure(u, params_);
   const double sound =
      std::sqrt(std::max(0.0, params_.gamma * pressure * inv_rho));
   return speed + sound;
}

} // namespace hycfd

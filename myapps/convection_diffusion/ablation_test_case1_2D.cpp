#include "mfem.hpp"
#include "newton_petsc_solver.hpp"
#include "tacot_material.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace mfem;

namespace
{

struct DriverParams
{
   string mesh_file = "Mesh/ablation_strip.msh";
   string material_file = "Input/material_tacot_case1.yaml";

   int order = 1;
   int serial_ref_levels = 0;
   int par_ref_levels = 0;

   double dt = 1.0e-2;
   double t_final = 120.0;

   double newton_abs_tol = 1.0e-8;
   double newton_rel_tol = 1.0e-6;
   int newton_max_iter = 20;
   int newton_print_level = 1;

   string petsc_options_file = "Input/petsc_ablation.opts";
   string ksp_prefix = "ablation_ls_";
   int petsc_ksp_print_level = 0;

   int output_every = 10;
   string output_path = "ParaView/ablation_case1";
   string collection_name = "ablation_test_case1_2D";
   string probes_csv = "temperature_probes.csv";
   string mass_csv = "mass_metrics.csv";
   string newton_csv = "newton_history_ablation_case1_2D.csv";
   string timing_step_csv = "driver_timing_per_step.csv";
   string timing_summary_csv = "driver_timing_summary.csv";
   bool save_paraview = true;

   int bdr_attr_top = 1;
   int bdr_attr_bottom = 2;
   int bdr_attr_sides = 3;

   double top_pressure = 101325.0;

   // (time, temperature)
   vector<pair<double, double>> top_temperature_schedule = {
      {0.0, 300.0},
      {0.1, 1644.0},
      {60.0, 1644.0},
      {60.1, 300.0},
      {120.0, 300.0}
   };

   double gravity_x = 0.0;
   double gravity_y = 0.0;

   double probe_x = 0.005;
   vector<double> probe_y = {0.05, 0.049, 0.048, 0.046, 0.042, 0.038, 0.034, 0.026};

   string fiat_temperature_file =
      "/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_1.0/data/ref/FIAT/T";
   string fiat_front_file =
      "/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_1.0/data/ref/FIAT/pyrolysisFront";

   // Acceptance tolerances for compare_ablation_case1.py
   double tol_temp_rmse = 150.0;
   double tol_temp_max_abs = 300.0;
   double tol_mdot_peak_rel = 0.5;
   double tol_mdot_peak_time = 10.0;
   double tol_front98_rmse = 0.01;
   double tol_front2_rmse = 0.01;
};

struct Bounds
{
   double xmin = 0.0;
   double xmax = 0.0;
   double ymin = 0.0;
   double ymax = 0.0;
};

using steady_clock_t = std::chrono::steady_clock;

double ElapsedSec(const steady_clock_t::time_point &t0,
                  const steady_clock_t::time_point &t1)
{
   return std::chrono::duration<double>(t1 - t0).count();
}

double EvaluateSchedule(const vector<pair<double, double>> &schedule, const double t)
{
   if (schedule.empty())
   {
      throw runtime_error("Temperature schedule is empty.");
   }

   if (t <= schedule.front().first)
   {
      return schedule.front().second;
   }
   if (t >= schedule.back().first)
   {
      return schedule.back().second;
   }

   for (int i = 0; i < static_cast<int>(schedule.size()) - 1; ++i)
   {
      const double t0 = schedule[i].first;
      const double t1 = schedule[i + 1].first;
      if (t >= t0 && t <= t1)
      {
         const double y0 = schedule[i].second;
         const double y1 = schedule[i + 1].second;
         if (std::abs(t1 - t0) < 1.0e-14)
         {
            return y1;
         }
         const double w = (t - t0) / (t1 - t0);
         return (1.0 - w) * y0 + w * y1;
      }
   }

   return schedule.back().second;
}

void LoadParams(const string &path, DriverParams &p)
{
   if (!filesystem::exists(path))
   {
      throw runtime_error("Input YAML file not found: " + path);
   }

   YAML::Node n = YAML::LoadFile(path);

   if (n["mesh_file"]) { p.mesh_file = n["mesh_file"].as<string>(); }
   if (n["material_file"]) { p.material_file = n["material_file"].as<string>(); }

   if (n["order"]) { p.order = n["order"].as<int>(); }
   if (n["serial_ref_levels"]) { p.serial_ref_levels = n["serial_ref_levels"].as<int>(); }
   if (n["par_ref_levels"]) { p.par_ref_levels = n["par_ref_levels"].as<int>(); }

   if (n["dt"]) { p.dt = n["dt"].as<double>(); }
   if (n["t_final"]) { p.t_final = n["t_final"].as<double>(); }

   if (n["newton_abs_tol"]) { p.newton_abs_tol = n["newton_abs_tol"].as<double>(); }
   if (n["newton_rel_tol"]) { p.newton_rel_tol = n["newton_rel_tol"].as<double>(); }
   if (n["newton_max_iter"]) { p.newton_max_iter = n["newton_max_iter"].as<int>(); }
   if (n["newton_print_level"]) { p.newton_print_level = n["newton_print_level"].as<int>(); }

   if (n["petsc_options_file"]) { p.petsc_options_file = n["petsc_options_file"].as<string>(); }
   if (n["ksp_prefix"]) { p.ksp_prefix = n["ksp_prefix"].as<string>(); }
   if (n["petsc_ksp_print_level"]) { p.petsc_ksp_print_level = n["petsc_ksp_print_level"].as<int>(); }

   if (n["output_every"]) { p.output_every = n["output_every"].as<int>(); }
   if (n["output_path"]) { p.output_path = n["output_path"].as<string>(); }
   if (n["collection_name"]) { p.collection_name = n["collection_name"].as<string>(); }
   if (n["probes_csv"]) { p.probes_csv = n["probes_csv"].as<string>(); }
   if (n["mass_csv"]) { p.mass_csv = n["mass_csv"].as<string>(); }
   if (n["newton_csv"]) { p.newton_csv = n["newton_csv"].as<string>(); }
   if (n["timing_step_csv"]) { p.timing_step_csv = n["timing_step_csv"].as<string>(); }
   if (n["timing_summary_csv"]) { p.timing_summary_csv = n["timing_summary_csv"].as<string>(); }
   if (n["save_paraview"]) { p.save_paraview = n["save_paraview"].as<bool>(); }

   if (n["bdr_attr_top"]) { p.bdr_attr_top = n["bdr_attr_top"].as<int>(); }
   if (n["bdr_attr_bottom"]) { p.bdr_attr_bottom = n["bdr_attr_bottom"].as<int>(); }
   if (n["bdr_attr_sides"]) { p.bdr_attr_sides = n["bdr_attr_sides"].as<int>(); }

   if (n["top_pressure"]) { p.top_pressure = n["top_pressure"].as<double>(); }

   if (n["gravity_x"]) { p.gravity_x = n["gravity_x"].as<double>(); }
   if (n["gravity_y"]) { p.gravity_y = n["gravity_y"].as<double>(); }

   if (n["probe_x"]) { p.probe_x = n["probe_x"].as<double>(); }
   if (n["probe_y"])
   {
      p.probe_y.clear();
      for (const YAML::Node &v : n["probe_y"])
      {
         p.probe_y.push_back(v.as<double>());
      }
   }

   if (n["top_temperature_schedule"])
   {
      p.top_temperature_schedule.clear();
      for (const YAML::Node &kv : n["top_temperature_schedule"])
      {
         if (!kv.IsSequence() || kv.size() != 2)
         {
            throw runtime_error("top_temperature_schedule entries must be [time, temperature].");
         }
         p.top_temperature_schedule.emplace_back(kv[0].as<double>(), kv[1].as<double>());
      }
      sort(p.top_temperature_schedule.begin(), p.top_temperature_schedule.end(),
           [](const auto &a, const auto &b) { return a.first < b.first; });
   }

   if (n["fiat_temperature_file"]) { p.fiat_temperature_file = n["fiat_temperature_file"].as<string>(); }
   if (n["fiat_front_file"]) { p.fiat_front_file = n["fiat_front_file"].as<string>(); }

   if (n["acceptance"])
   {
      YAML::Node a = n["acceptance"];
      if (a["temperature_rmse_max"]) { p.tol_temp_rmse = a["temperature_rmse_max"].as<double>(); }
      if (a["temperature_max_abs_max"]) { p.tol_temp_max_abs = a["temperature_max_abs_max"].as<double>(); }
      if (a["m_dot_g_peak_rel_error_max"]) { p.tol_mdot_peak_rel = a["m_dot_g_peak_rel_error_max"].as<double>(); }
      if (a["m_dot_g_peak_time_error_max"]) { p.tol_mdot_peak_time = a["m_dot_g_peak_time_error_max"].as<double>(); }
      if (a["front98_rmse_max"]) { p.tol_front98_rmse = a["front98_rmse_max"].as<double>(); }
      if (a["front2_rmse_max"]) { p.tol_front2_rmse = a["front2_rmse_max"].as<double>(); }
   }

   if (p.dt <= 0.0) { throw runtime_error("dt must be > 0."); }
   if (p.t_final < 0.0) { throw runtime_error("t_final must be >= 0."); }
   if (p.order < 1) { throw runtime_error("order must be >= 1."); }
   if (p.newton_max_iter < 1) { throw runtime_error("newton_max_iter must be >= 1."); }
   if (p.top_temperature_schedule.empty()) { throw runtime_error("top_temperature_schedule cannot be empty."); }
}

Bounds GetGlobalBounds(const ParMesh &pmesh)
{
   double local_min[2] = {numeric_limits<double>::infinity(),
                          numeric_limits<double>::infinity()};
   double local_max[2] = {-numeric_limits<double>::infinity(),
                          -numeric_limits<double>::infinity()};

   for (int i = 0; i < pmesh.GetNV(); ++i)
   {
      const double *v = pmesh.GetVertex(i);
      local_min[0] = min(local_min[0], v[0]);
      local_min[1] = min(local_min[1], v[1]);
      local_max[0] = max(local_max[0], v[0]);
      local_max[1] = max(local_max[1], v[1]);
   }

   Bounds b;
   MPI_Allreduce(local_min, &b.xmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(local_max, &b.xmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(local_min + 1, &b.ymin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(local_max + 1, &b.ymax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return b;
}

class ReactionStateManager
{
public:
   void Initialize(const ParFiniteElementSpace &fes,
                   const int quad_order,
                   const TACOTMaterial &material)
   {
      const int ne = fes.GetNE();
      const int nr = material.NumReactions();
      states_.assign(ne, {});
      tau_elem_.assign(ne, 1.0);
      rho_elem_.assign(ne, material.InitialSolidDensity());
      pi_elem_.assign(ne, 0.0);
      mdot_elem_.assign(ne, 0.0);
      extent_elem_.assign(nr, vector<double>(ne, 0.0));
      degree_char_elem_.assign(ne, 0.0);
      char_density_fraction_elem_.assign(ne, 0.0);

      for (int e = 0; e < ne; ++e)
      {
         const FiniteElement *fe = fes.GetFE(e);
         const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), quad_order);
         states_[e].resize(ir.GetNPoints(), material.CreateInitialState());
      }
   }

   const TACOTMaterial::InternalState &GetState(const int elem, const int qp) const
   {
      return states_.at(elem).at(qp);
   }

   void SetState(const int elem, const int qp, TACOTMaterial::InternalState s)
   {
      states_.at(elem).at(qp) = std::move(s);
   }

   int NumQPoints(const int elem) const
   {
      return static_cast<int>(states_.at(elem).size());
   }

   int NumElements() const
   {
      return static_cast<int>(states_.size());
   }

   void SetElementDiagnostics(const int elem,
                              const double tau,
                              const double rho,
                              const double pi,
                              const double mdot)
   {
      tau_elem_.at(elem) = tau;
      rho_elem_.at(elem) = rho;
      pi_elem_.at(elem) = pi;
      mdot_elem_.at(elem) = mdot;
   }

   const vector<double> &TauElement() const { return tau_elem_; }
   const vector<double> &RhoElement() const { return rho_elem_; }
   const vector<double> &PiElement() const { return pi_elem_; }
   const vector<double> &MdotElement() const { return mdot_elem_; }
   const vector<double> &ExtentElement(const int reaction_id) const
   {
      return extent_elem_.at(reaction_id);
   }
   const vector<double> &DegreeCharElement() const { return degree_char_elem_; }
   const vector<double> &CharDensityFractionElement() const
   {
      return char_density_fraction_elem_;
   }
   int NumReactions() const
   {
      return static_cast<int>(extent_elem_.size());
   }
   void SetElementInternalAverages(const int elem,
                                   const vector<double> &extent_avg,
                                   const double degree_char,
                                   const double char_density_fraction)
   {
      if (extent_avg.size() != extent_elem_.size())
      {
         throw runtime_error("Extent size mismatch in SetElementInternalAverages.");
      }
      for (int r = 0; r < static_cast<int>(extent_elem_.size()); ++r)
      {
         extent_elem_[r].at(elem) = extent_avg[r];
      }
      degree_char_elem_.at(elem) = degree_char;
      char_density_fraction_elem_.at(elem) = char_density_fraction;
   }

private:
   vector<vector<TACOTMaterial::InternalState>> states_;
   vector<double> tau_elem_;
   vector<double> rho_elem_;
   vector<double> pi_elem_;
   vector<double> mdot_elem_;
   vector<vector<double>> extent_elem_;
   vector<double> degree_char_elem_;
   vector<double> char_density_fraction_elem_;
};

class AblationTPIntegrator : public BlockNonlinearFormIntegrator
{
public:
   AblationTPIntegrator(const TACOTMaterial &material,
                        const ReactionStateManager &state_manager,
                        ParGridFunction &T_old,
                        ParGridFunction &p_old,
                        const int quad_order,
                        const Vector &gravity)
      : material_(material),
        state_manager_(state_manager),
        T_old_coeff_(&T_old),
        p_old_coeff_(&p_old),
        quad_order_(quad_order),
        gravity_(gravity)
   {
      dt_ = 1.0;
   }

   void SetTimeStep(const double dt) { dt_ = dt; }

   void AssembleElementVector(const Array<const FiniteElement *> &el,
                              ElementTransformation &Tr,
                              const Array<const Vector *> &elfun,
                              const Array<Vector *> &elvec) override
   {
      if (el.Size() != 2)
      {
         MFEM_ABORT("AblationTPIntegrator expects exactly 2 blocks (T,p).");
      }

      ComputeElementResidual(*el[0], *el[1], Tr, *elfun[0], *elfun[1], *elvec[0], *elvec[1]);
   }

   void AssembleElementGrad(const Array<const FiniteElement *> &el,
                            ElementTransformation &Tr,
                            const Array<const Vector *> &elfun,
                            const Array2D<DenseMatrix *> &elmats) override
   {
      const int dof_T = el[0]->GetDof();
      const int dof_p = el[1]->GetDof();

      elmats(0, 0)->SetSize(dof_T, dof_T);
      elmats(0, 1)->SetSize(dof_T, dof_p);
      elmats(1, 0)->SetSize(dof_p, dof_T);
      elmats(1, 1)->SetSize(dof_p, dof_p);

      *elmats(0, 0) = 0.0;
      *elmats(0, 1) = 0.0;
      *elmats(1, 0) = 0.0;
      *elmats(1, 1) = 0.0;

      Vector rT0, rp0;
      ComputeElementResidual(*el[0], *el[1], Tr, *elfun[0], *elfun[1], rT0, rp0);

      const double fd_eps = 1.0e-7;

      Vector eT(*elfun[0]);
      Vector ep(*elfun[1]);
      Vector rT_pert, rp_pert;

      for (int j = 0; j < dof_T; ++j)
      {
         const double h = fd_eps * max(1.0, std::abs(eT[j]));
         eT = *elfun[0];
         eT[j] += h;

         ComputeElementResidual(*el[0], *el[1], Tr, eT, *elfun[1], rT_pert, rp_pert);

         for (int i = 0; i < dof_T; ++i)
         {
            (*elmats(0, 0))(i, j) = (rT_pert[i] - rT0[i]) / h;
         }
         for (int i = 0; i < dof_p; ++i)
         {
            (*elmats(1, 0))(i, j) = (rp_pert[i] - rp0[i]) / h;
         }
      }

      for (int j = 0; j < dof_p; ++j)
      {
         const double h = fd_eps * max(1.0, std::abs(ep[j]));
         ep = *elfun[1];
         ep[j] += h;

         ComputeElementResidual(*el[0], *el[1], Tr, *elfun[0], ep, rT_pert, rp_pert);

         for (int i = 0; i < dof_T; ++i)
         {
            (*elmats(0, 1))(i, j) = (rT_pert[i] - rT0[i]) / h;
         }
         for (int i = 0; i < dof_p; ++i)
         {
            (*elmats(1, 1))(i, j) = (rp_pert[i] - rp0[i]) / h;
         }
      }
   }

private:
   void ComputeElementResidual(const FiniteElement &fe_T,
                               const FiniteElement &fe_p,
                               ElementTransformation &Tr,
                               const Vector &elT,
                               const Vector &elp,
                               Vector &rT,
                               Vector &rp) const
   {
      const int dof_T = fe_T.GetDof();
      const int dof_p = fe_p.GetDof();
      const int dim = fe_T.GetDim();

      rT.SetSize(dof_T);
      rp.SetSize(dof_p);
      rT = 0.0;
      rp = 0.0;

      shape_T_.SetSize(dof_T);
      shape_p_.SetSize(dof_p);
      dshape_T_.SetSize(dof_T, dim);
      dshape_p_.SetSize(dof_p, dim);
      gradT_.SetSize(dim);
      gradp_.SetSize(dim);

      const IntegrationRule &ir = IntRules.Get(fe_T.GetGeomType(), quad_order_);
      MFEM_VERIFY(Tr.ElementNo >= 0, "Invalid element number while assembling residual.");
      MFEM_VERIFY(ir.GetNPoints() <= state_manager_.NumQPoints(Tr.ElementNo),
                  "Reaction state manager quadrature size mismatch.");

      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);

         fe_T.CalcPhysShape(Tr, shape_T_);
         fe_p.CalcPhysShape(Tr, shape_p_);
         fe_T.CalcPhysDShape(Tr, dshape_T_);
         fe_p.CalcPhysDShape(Tr, dshape_p_);

         const double T = shape_T_ * elT;
         const double p = shape_p_ * elp;

         gradT_ = 0.0;
         gradp_ = 0.0;
         for (int j = 0; j < dof_T; ++j)
         {
            for (int d = 0; d < dim; ++d)
            {
               gradT_[d] += elT[j] * dshape_T_(j, d);
            }
         }
         for (int j = 0; j < dof_p; ++j)
         {
            for (int d = 0; d < dim; ++d)
            {
               gradp_[d] += elp[j] * dshape_p_(j, d);
            }
         }

         const double T_old = T_old_coeff_.Eval(Tr, ip);
         const double p_old = p_old_coeff_.Eval(Tr, ip);

         const TACOTMaterial::InternalState &old_state =
            state_manager_.GetState(Tr.ElementNo, q);

         TACOTMaterial::InternalState new_state =
            material_.SolveReactionExtents(T, dt_, old_state);
         TACOTMaterial::SolidProperties solid =
            material_.EvaluateSolid(T, p, new_state);
         TACOTMaterial::GasProperties gas =
            material_.EvaluateGas(T, p, new_state);

         TACOTMaterial::InternalState old_eval_state = old_state;
         old_eval_state.extent_old = old_eval_state.extent;
         old_eval_state.dt = dt_;
         TACOTMaterial::SolidProperties solid_old =
            material_.EvaluateSolid(T_old, p_old, old_eval_state);
         TACOTMaterial::GasProperties gas_old =
            material_.EvaluateGas(T_old, p_old, old_eval_state);

         const double mu = max(gas.mu, 1.0e-12);
         const double darcy = solid.K / mu;
         const double rho_darcy = gas.rho * darcy;
         const double rho2_darcy = gas.rho * rho_darcy;
         const double h_rho_darcy = gas.h * rho_darcy;
         const double h_rho2_darcy = gas.h * rho2_darcy;

         const double storage_p =
            (solid.eps_g * gas.rho - solid_old.eps_g * gas_old.rho) / dt_;
         const double source_p = solid.pi_total;

         const double solid_storage = solid.rho_s * solid.cp * (T - T_old) / dt_;
         const double gas_storage =
            (solid.eps_g * (gas.rho * gas.h - p) -
             solid_old.eps_g * (gas_old.rho * gas_old.h - p_old)) / dt_;

         const double w = ip.weight * Tr.Weight();

         for (int i = 0; i < dof_p; ++i)
         {
            double grad_dot = 0.0;
            double g_dot = 0.0;
            for (int d = 0; d < dim; ++d)
            {
               grad_dot += dshape_p_(i, d) * gradp_[d];
               g_dot += gravity_[d] * dshape_p_(i, d);
            }

            rp[i] += w * (shape_p_[i] * (storage_p - source_p)
                          + rho_darcy * grad_dot
                          - rho2_darcy * g_dot);
         }

         for (int i = 0; i < dof_T; ++i)
         {
            double gradT_dot = 0.0;
            double gradp_dot = 0.0;
            double g_dot = 0.0;
            for (int d = 0; d < dim; ++d)
            {
               gradT_dot += dshape_T_(i, d) * gradT_[d];
               gradp_dot += dshape_T_(i, d) * gradp_[d];
               g_dot += gravity_[d] * dshape_T_(i, d);
            }

            rT[i] += w * (shape_T_[i] * (solid_storage + gas_storage - solid.pyrolysis_heat_sink)
                          + solid.k * gradT_dot
                          + h_rho_darcy * gradp_dot
                          - h_rho2_darcy * g_dot);
         }
      }
   }

   const TACOTMaterial &material_;
   const ReactionStateManager &state_manager_;

   mutable GridFunctionCoefficient T_old_coeff_;
   mutable GridFunctionCoefficient p_old_coeff_;

   int quad_order_ = 2;
   double dt_ = 1.0;
   Vector gravity_;

   mutable Vector shape_T_;
   mutable Vector shape_p_;
   mutable DenseMatrix dshape_T_;
   mutable DenseMatrix dshape_p_;
   mutable Vector gradT_;
   mutable Vector gradp_;
};

void ApplyElementScalar(const ParFiniteElementSpace &fes,
                        const vector<double> &elem_vals,
                        ParGridFunction &gf)
{
   gf = 0.0;
   Array<int> vdofs;
   for (int e = 0; e < fes.GetNE(); ++e)
   {
      fes.GetElementVDofs(e, vdofs);
      for (int j = 0; j < vdofs.Size(); ++j)
      {
         const int dof = vdofs[j];
         gf(dof) = elem_vals[e];
      }
   }
}

double ComputeTopBoundaryMassFlux(ParMesh &pmesh,
                                  const ParFiniteElementSpace &fes_T,
                                  const ParFiniteElementSpace &fes_p,
                                  const ParGridFunction &T,
                                  const ParGridFunction &p,
                                  const TACOTMaterial &material,
                                  const ReactionStateManager &state_manager,
                                  const Vector &gravity,
                                  const int top_bdr_attr)
{
   Array<int> dofs_T, dofs_p;
   Vector elT, elp, shape_T, shape_p, gradp, normal;
   DenseMatrix dshape_p;

   const int dim = pmesh.Dimension();
   double local_flux_int = 0.0;
   double local_area = 0.0;

   for (int be = 0; be < pmesh.GetNBE(); ++be)
   {
      if (pmesh.GetBdrAttribute(be) != top_bdr_attr)
      {
         continue;
      }

      FaceElementTransformations *FT = pmesh.GetBdrFaceTransformations(be);
      if (!FT || FT->Elem1No < 0 || FT->Elem1No >= fes_T.GetNE())
      {
         continue;
      }

      const int elem = FT->Elem1No;
      const FiniteElement *fe_T = fes_T.GetFE(elem);
      const FiniteElement *fe_p = fes_p.GetFE(elem);

      fes_T.GetElementDofs(elem, dofs_T);
      fes_p.GetElementDofs(elem, dofs_p);
      T.GetSubVector(dofs_T, elT);
      p.GetSubVector(dofs_p, elp);

      shape_T.SetSize(fe_T->GetDof());
      shape_p.SetSize(fe_p->GetDof());
      dshape_p.SetSize(fe_p->GetDof(), dim);
      gradp.SetSize(dim);
      normal.SetSize(dim);

      const int face_int_order = max(2, 2 * max(fe_T->GetOrder(), fe_p->GetOrder()) + 2);
      const IntegrationRule &ir_face = IntRules.Get(FT->GetGeometryType(), face_int_order);

      for (int q = 0; q < ir_face.GetNPoints(); ++q)
      {
         const IntegrationPoint &fip = ir_face.IntPoint(q);
         IntegrationPoint eip;
         FT->Loc1.Transform(fip, eip);

         FT->Elem1->SetIntPoint(&eip);
         fe_T->CalcPhysShape(*FT->Elem1, shape_T);
         fe_p->CalcPhysShape(*FT->Elem1, shape_p);
         fe_p->CalcPhysDShape(*FT->Elem1, dshape_p);

         const double Tq = shape_T * elT;
         const double pq = shape_p * elp;

         gradp = 0.0;
         for (int j = 0; j < fe_p->GetDof(); ++j)
         {
            for (int d = 0; d < dim; ++d)
            {
               gradp[d] += elp[j] * dshape_p(j, d);
            }
         }

         // Use a representative local reaction state from the adjacent element.
         const TACOTMaterial::InternalState &state = state_manager.GetState(elem, 0);
         const TACOTMaterial::SolidProperties solid = material.EvaluateSolid(Tq, pq, state);
         const TACOTMaterial::GasProperties gas = material.EvaluateGas(Tq, pq, state);

         const double mu = max(gas.mu, 1.0e-12);
         const double rho_darcy = gas.rho * solid.K / mu;
         const double rho2_darcy = gas.rho * rho_darcy;

         Vector mflux(dim);
         for (int d = 0; d < dim; ++d)
         {
            mflux[d] = -rho_darcy * gradp[d] + rho2_darcy * gravity[d];
         }

         FT->Face->SetIntPoint(&fip);
         if (dim == 1)
         {
            normal[0] = 1.0;
         }
         else
         {
            CalcOrtho(FT->Face->Jacobian(), normal);
         }

         local_flux_int += fip.weight * (mflux * normal);
         local_area += fip.weight * normal.Norml2();
      }
   }

   double global_flux_int = 0.0;
   double global_area = 0.0;
   MPI_Allreduce(&local_flux_int, &global_flux_int, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
   MPI_Allreduce(&local_area, &global_area, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());

   if (global_area <= 0.0)
   {
      return numeric_limits<double>::quiet_NaN();
   }
   return global_flux_int / global_area;
}

double SampleFieldAtPoint(ParMesh &pmesh, const ParGridFunction &gf,
                          const double x, const double y)
{
   DenseMatrix pt(2, 1);
   pt(0, 0) = x;
   pt(1, 0) = y;

   Array<int> elem_ids;
   Array<IntegrationPoint> ips;
   pmesh.FindPoints(pt, elem_ids, ips);

   double local_val = 0.0;
   int local_found = 0;
   if (elem_ids.Size() > 0 && elem_ids[0] >= 0)
   {
      local_val = gf.GetValue(elem_ids[0], ips[0]);
      local_found = 1;
   }

   double global_sum = 0.0;
   int global_count = 0;
   MPI_Allreduce(&local_val, &global_sum, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
   MPI_Allreduce(&local_found, &global_count, 1, MPI_INT, MPI_SUM, pmesh.GetComm());

   if (global_count == 0)
   {
      return numeric_limits<double>::quiet_NaN();
   }
   return global_sum / static_cast<double>(global_count);
}

double ComputeFrontDepth(ParMesh &pmesh,
                         const ParGridFunction &tau_gf,
                         const double x,
                         const double y_top,
                         const double y_bottom,
                         const double threshold)
{
   const int ns = 250;
   const double eps = 1.0e-9;

   const double y0 = y_top - eps;
   const double y1 = y_bottom + eps;
   const double dy = (y0 - y1) / static_cast<double>(ns);

   double yp = y0;
   double vp = SampleFieldAtPoint(pmesh, tau_gf, x, yp);

   for (int k = 1; k <= ns; ++k)
   {
      const double yc = y0 - k * dy;
      const double vc = SampleFieldAtPoint(pmesh, tau_gf, x, yc);

      if (std::isfinite(vp) && std::isfinite(vc) && vp > threshold && vc <= threshold)
      {
         const double denom = (vp - vc);
         double frac = 0.0;
         if (std::abs(denom) > 1.0e-14)
         {
            frac = (vp - threshold) / denom;
            frac = std::max(0.0, std::min(1.0, frac));
         }
         const double y_cross = yp - frac * (yp - yc);
         return std::max(0.0, y_top - y_cross);
      }

      yp = yc;
      vp = vc;
   }

   return 0.0;
}

void AdvanceInternalStates(const TACOTMaterial &material,
                           ReactionStateManager &state_manager,
                           const ParFiniteElementSpace &fes_T,
                           const ParFiniteElementSpace &fes_p,
                           const ParGridFunction &T,
                           const ParGridFunction &p,
                           const int quad_order,
                           const double dt)
{
   Array<int> dofs_T, dofs_p;
   Vector elT, elp, shape_T, shape_p;
   const int nr = state_manager.NumReactions();
   const double rho_v = material.InitialSolidDensity();
   const double rho_c = material.CharSolidDensity();
   const double rho_den = rho_v - rho_c;

   for (int e = 0; e < fes_T.GetNE(); ++e)
   {
      const FiniteElement *fe_T = fes_T.GetFE(e);
      const FiniteElement *fe_p = fes_p.GetFE(e);
      ElementTransformation *Tr = fes_T.GetElementTransformation(e);

      fes_T.GetElementDofs(e, dofs_T);
      fes_p.GetElementDofs(e, dofs_p);
      T.GetSubVector(dofs_T, elT);
      p.GetSubVector(dofs_p, elp);

      const IntegrationRule &ir = IntRules.Get(fe_T->GetGeomType(), quad_order);
      MFEM_VERIFY(ir.GetNPoints() == state_manager.NumQPoints(e),
                  "State manager quadrature mismatch during state advance.");

      shape_T.SetSize(fe_T->GetDof());
      shape_p.SetSize(fe_p->GetDof());

      double tau_acc = 0.0;
      double rho_acc = 0.0;
      double pi_acc = 0.0;
      double mdot_acc = 0.0;
      vector<double> extent_acc(nr, 0.0);

      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr->SetIntPoint(&ip);

         fe_T->CalcPhysShape(*Tr, shape_T);
         fe_p->CalcPhysShape(*Tr, shape_p);

         const double Tq = shape_T * elT;
         const double pq = shape_p * elp;

         const TACOTMaterial::InternalState old_state = state_manager.GetState(e, q);
         TACOTMaterial::InternalState new_state =
            material.SolveReactionExtents(Tq, dt, old_state);
         TACOTMaterial::SolidProperties solid = material.EvaluateSolid(Tq, pq, new_state);

         tau_acc += solid.tau;
         rho_acc += solid.rho_s;
         pi_acc += solid.pi_total;
         mdot_acc += solid.m_dot_g;
         for (int r = 0; r < nr; ++r)
         {
            if (r < static_cast<int>(new_state.extent.size()))
            {
               extent_acc[r] += new_state.extent[r];
            }
         }

         TACOTMaterial::InternalState committed;
         committed.extent = new_state.extent;
         committed.extent_old = new_state.extent;
         committed.dt = 0.0;
         state_manager.SetState(e, q, committed);
      }

      const double inv_nq = 1.0 / static_cast<double>(ir.GetNPoints());
      state_manager.SetElementDiagnostics(e,
                                          tau_acc * inv_nq,
                                          rho_acc * inv_nq,
                                          pi_acc * inv_nq,
                                          mdot_acc * inv_nq);

      vector<double> extent_avg(nr, 0.0);
      for (int r = 0; r < nr; ++r)
      {
         extent_avg[r] = extent_acc[r] * inv_nq;
      }

      const double tau_avg = tau_acc * inv_nq;
      const double rho_avg = rho_acc * inv_nq;
      const double degree_char = std::max(0.0, std::min(1.0, 1.0 - tau_avg));

      double char_density_fraction = 0.0;
      if (std::abs(rho_den) > 1.0e-14)
      {
         char_density_fraction = (rho_v - rho_avg) / rho_den;
         char_density_fraction = std::max(0.0, std::min(1.0, char_density_fraction));
      }
      state_manager.SetElementInternalAverages(e, extent_avg, degree_char, char_density_fraction);
   }
}

void InitializeDiagnostics(const TACOTMaterial &material,
                           ReactionStateManager &state_manager)
{
   vector<double> zero_extents(state_manager.NumReactions(), 0.0);
   for (int e = 0; e < state_manager.NumElements(); ++e)
   {
      state_manager.SetElementDiagnostics(e,
                                          1.0,
                                          material.InitialSolidDensity(),
                                          0.0,
                                          0.0);
      state_manager.SetElementInternalAverages(e, zero_extents, 0.0, 0.0);
   }
}

void PrintConfig(const DriverParams &p)
{
   cout << "Loaded configuration:" << endl;
   cout << "  mesh_file: " << p.mesh_file << endl;
   cout << "  material_file: " << p.material_file << endl;
   cout << "  order: " << p.order << endl;
   cout << "  dt: " << p.dt << endl;
   cout << "  t_final: " << p.t_final << endl;
   cout << "  newton_abs_tol: " << p.newton_abs_tol << endl;
   cout << "  newton_rel_tol: " << p.newton_rel_tol << endl;
   cout << "  newton_max_iter: " << p.newton_max_iter << endl;
   cout << "  petsc_options_file: " << p.petsc_options_file << endl;
   cout << "  ksp_prefix: " << p.ksp_prefix << endl;
   cout << "  petsc_ksp_print_level: " << p.petsc_ksp_print_level << endl;
   cout << "  output_every: " << p.output_every << endl;
   cout << "  output_path: " << p.output_path << endl;
   cout << "  collection_name: " << p.collection_name << endl;
   cout << "  timing_step_csv: " << p.timing_step_csv << endl;
   cout << "  timing_summary_csv: " << p.timing_summary_csv << endl;
   cout << "  save_paraview: " << (p.save_paraview ? "true" : "false") << endl;
   cout << "  bdr_attr_top: " << p.bdr_attr_top << endl;
   cout << "  top_pressure: " << p.top_pressure << endl;
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const int myid = Mpi::WorldRank();

   string input_file = "Input/input_ablation_case1.yaml";
   OptionsParser args(argc, argv);
   args.AddOption(&input_file, "-i", "--input", "YAML input file.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   DriverParams params;
   try
   {
      LoadParams(input_file, params);
   }
   catch (const exception &e)
   {
      if (myid == 0) { cerr << e.what() << endl; }
      return 2;
   }

   if (myid == 0) { PrintConfig(params); }

   std::string petsc_options_path_storage;
   const char *petsc_file_to_use =
      newton_utils::ResolvePetscOptionsFile(params.petsc_options_file,
                                            myid,
                                            petsc_options_path_storage);
   MFEMInitializePetsc(&argc, &argv, petsc_file_to_use, NULL);

   int exit_code = 0;
   try
   {
      const auto run_t0 = steady_clock_t::now();
      const auto setup_t0 = run_t0;

      Device device("cpu");
      if (myid == 0) { device.Print(); }

      TACOTMaterial material;
      material.LoadFromYaml(params.material_file);

      unique_ptr<Mesh> mesh = make_unique<Mesh>(params.mesh_file.c_str(), 1, 1);
      if (mesh->Dimension() != 2)
      {
         throw runtime_error("The mesh must be 2D.");
      }
      for (int l = 0; l < params.serial_ref_levels; ++l)
      {
         mesh->UniformRefinement();
      }

      unique_ptr<ParMesh> pmesh = make_unique<ParMesh>(MPI_COMM_WORLD, *mesh);
      mesh.reset();
      for (int l = 0; l < params.par_ref_levels; ++l)
      {
         pmesh->UniformRefinement();
      }

      if (pmesh->bdr_attributes.Size() == 0)
      {
         throw runtime_error("Mesh must have boundary attributes.");
      }

      const Bounds bounds = GetGlobalBounds(*pmesh);
      const double xmid = 0.5 * (bounds.xmin + bounds.xmax);

      H1_FECollection fec(params.order, 2);
      ParFiniteElementSpace fes_T(pmesh.get(), &fec);
      ParFiniteElementSpace fes_p(pmesh.get(), &fec);

      L2_FECollection l2_fec(0, 2);
      ParFiniteElementSpace fes_diag(pmesh.get(), &l2_fec);

      Array<ParFiniteElementSpace *> spaces(2);
      spaces[0] = &fes_T;
      spaces[1] = &fes_p;

      Array<int> block_true_offsets(3);
      block_true_offsets[0] = 0;
      block_true_offsets[1] = fes_T.TrueVSize();
      block_true_offsets[2] = fes_p.TrueVSize();
      block_true_offsets.PartialSum();

      if (myid == 0)
      {
         cout << "Global true dofs (T): " << fes_T.GlobalTrueVSize() << endl;
         cout << "Global true dofs (p): " << fes_p.GlobalTrueVSize() << endl;
         cout << "Global true dofs (T+p): " << (fes_T.GlobalTrueVSize() + fes_p.GlobalTrueVSize()) << endl;
      }

      Array<int> ess_bdr_T(pmesh->bdr_attributes.Max());
      Array<int> ess_bdr_p(pmesh->bdr_attributes.Max());
      ess_bdr_T = 0;
      ess_bdr_p = 0;
      ess_bdr_T[params.bdr_attr_top - 1] = 1;
      ess_bdr_p[params.bdr_attr_top - 1] = 1;

      Array<int> ess_tdof_T, ess_tdof_p;
      fes_T.GetEssentialTrueDofs(ess_bdr_T, ess_tdof_T);
      fes_p.GetEssentialTrueDofs(ess_bdr_p, ess_tdof_p);

      ParGridFunction T(&fes_T), p(&fes_p), T_old(&fes_T), p_old(&fes_p);
      T = 300.0;
      p = params.top_pressure;
      T_old = T;
      p_old = p;

      // Apply initial essential values.
      {
         Vector Ttrue, ptrue;
         T.GetTrueDofs(Ttrue);
         p.GetTrueDofs(ptrue);
         const double Tbc0 = EvaluateSchedule(params.top_temperature_schedule, 0.0);
         Ttrue.SetSubVector(ess_tdof_T, Tbc0);
         ptrue.SetSubVector(ess_tdof_p, params.top_pressure);
         T.SetFromTrueDofs(Ttrue);
         p.SetFromTrueDofs(ptrue);
         T_old = T;
         p_old = p;
      }

      const int quad_order = max(2, 2 * params.order + 2);

      ReactionStateManager state_manager;
      state_manager.Initialize(fes_T, quad_order, material);
      InitializeDiagnostics(material, state_manager);

      Vector gravity(2);
      gravity[0] = params.gravity_x;
      gravity[1] = params.gravity_y;

      auto *tp_integrator =
         new AblationTPIntegrator(material, state_manager, T_old, p_old, quad_order, gravity);

      ParBlockNonlinearForm block_form(spaces);
      block_form.SetGradientType(Operator::Hypre_ParCSR);
      block_form.AddDomainIntegrator(tp_integrator);

      Array<Array<int> *> ess_bdr(2);
      ess_bdr[0] = &ess_bdr_T;
      ess_bdr[1] = &ess_bdr_p;
      Array<Vector *> rhs_null(2);
      rhs_null = NULL;
      block_form.SetEssentialBC(ess_bdr, rhs_null);

      newton_utils::NewtonConfig newton_cfg;
      newton_cfg.abs_tol = params.newton_abs_tol;
      newton_cfg.rel_tol = params.newton_rel_tol;
      newton_cfg.max_iter = params.newton_max_iter;

      newton_utils::PetscLinearConfig linear_cfg;
      linear_cfg.ksp_prefix = params.ksp_prefix;
      linear_cfg.ksp_print_level = params.petsc_ksp_print_level;

      newton_utils::PetscNewtonSolver newton_solver(MPI_COMM_WORLD,
                                                    newton_cfg,
                                                    linear_cfg);

      Vector x(block_true_offsets.Last());
      BlockVector xb(x, block_true_offsets);
      T.GetTrueDofs(xb.GetBlock(0));
      p.GetTrueDofs(xb.GetBlock(1));

      ParGridFunction tau_gf(&fes_diag), rho_s_gf(&fes_diag), pi_total_gf(&fes_diag), mdot_g_gf(&fes_diag);
      ParGridFunction degree_char_gf(&fes_diag), char_density_fraction_gf(&fes_diag);
      vector<unique_ptr<ParGridFunction>> extent_gf;
      vector<string> extent_field_names;
      extent_gf.reserve(state_manager.NumReactions());
      extent_field_names.reserve(state_manager.NumReactions());
      for (int r = 0; r < state_manager.NumReactions(); ++r)
      {
         extent_gf.emplace_back(make_unique<ParGridFunction>(&fes_diag));
         extent_field_names.push_back("X" + to_string(r + 1));
      }
      ApplyElementScalar(fes_diag, state_manager.TauElement(), tau_gf);
      ApplyElementScalar(fes_diag, state_manager.RhoElement(), rho_s_gf);
      ApplyElementScalar(fes_diag, state_manager.PiElement(), pi_total_gf);
      ApplyElementScalar(fes_diag, state_manager.MdotElement(), mdot_g_gf);
      ApplyElementScalar(fes_diag, state_manager.DegreeCharElement(), degree_char_gf);
      ApplyElementScalar(fes_diag, state_manager.CharDensityFractionElement(), char_density_fraction_gf);
      for (int r = 0; r < state_manager.NumReactions(); ++r)
      {
         ApplyElementScalar(fes_diag, state_manager.ExtentElement(r), *extent_gf[r]);
      }

      std::error_code ec;
      filesystem::create_directories(params.output_path, ec);
      if (ec)
      {
         throw runtime_error("Failed to create output path: " + params.output_path +
                             " (" + ec.message() + ")");
      }

      ofstream probes_csv;
      ofstream mass_csv;
      ofstream newton_csv;
      ofstream timing_step_csv;
      if (myid == 0)
      {
         probes_csv.open(filesystem::path(params.output_path) / params.probes_csv);
         mass_csv.open(filesystem::path(params.output_path) / params.mass_csv);
         newton_csv.open(filesystem::path(params.output_path) / params.newton_csv);
         timing_step_csv.open(filesystem::path(params.output_path) / params.timing_step_csv);

         if (!probes_csv || !mass_csv || !newton_csv || !timing_step_csv)
         {
            throw runtime_error("Failed to open one or more CSV output files.");
         }

         probes_csv << "time,wall,TC1,TC2,TC3,TC4,TC5,TC6,TC7\n";
         mass_csv << "time,m_dot_g_surf,m_dot_c,front_98_virgin,front_2_char,recession\n";
         newton_csv << "step,time,iter,residual,residual0,rel_residual,"
                    << "update_norm,update0,rel_update,converged\n";
         timing_step_csv << "step,bc_sec,newton_sec,newton_residual_eval_sec,"
                         << "newton_jacobian_sec,newton_linear_sec,"
                         << "newton_update_sec,state_advance_sec,"
                         << "output_sec,step_total_sec\n";

         probes_csv << setprecision(16);
         mass_csv << setprecision(16);
         newton_csv << setprecision(16);
         timing_step_csv << setprecision(16);
      }

      ParaViewDataCollection paraview_dc(params.collection_name.c_str(), pmesh.get());
      if (params.save_paraview)
      {
         paraview_dc.SetPrefixPath(params.output_path.c_str());
         paraview_dc.SetLevelsOfDetail(params.order);
         paraview_dc.SetDataFormat(VTKFormat::BINARY);
         paraview_dc.SetHighOrderOutput(true);
         paraview_dc.RegisterField("temperature", &T);
         paraview_dc.RegisterField("pressure", &p);
         paraview_dc.RegisterField("tau", &tau_gf);
         paraview_dc.RegisterField("rho_s", &rho_s_gf);
         paraview_dc.RegisterField("pi_total", &pi_total_gf);
         paraview_dc.RegisterField("m_dot_g", &mdot_g_gf);
         paraview_dc.RegisterField("degree_char", &degree_char_gf);
         paraview_dc.RegisterField("char_density_fraction", &char_density_fraction_gf);
         for (int r = 0; r < state_manager.NumReactions(); ++r)
         {
            paraview_dc.RegisterField(extent_field_names[r].c_str(), extent_gf[r].get());
         }
      }

      auto write_outputs = [&](const int step, const double time)
      {
         ApplyElementScalar(fes_diag, state_manager.TauElement(), tau_gf);
         ApplyElementScalar(fes_diag, state_manager.RhoElement(), rho_s_gf);
         ApplyElementScalar(fes_diag, state_manager.PiElement(), pi_total_gf);
         ApplyElementScalar(fes_diag, state_manager.MdotElement(), mdot_g_gf);
         ApplyElementScalar(fes_diag, state_manager.DegreeCharElement(), degree_char_gf);
         ApplyElementScalar(fes_diag, state_manager.CharDensityFractionElement(), char_density_fraction_gf);
         for (int r = 0; r < state_manager.NumReactions(); ++r)
         {
            ApplyElementScalar(fes_diag, state_manager.ExtentElement(r), *extent_gf[r]);
         }

         const double wallT = EvaluateSchedule(params.top_temperature_schedule, time);
         vector<double> probe_vals;
         probe_vals.reserve(params.probe_y.size() - 1);
         for (int i = 1; i < static_cast<int>(params.probe_y.size()); ++i)
         {
            const double val = SampleFieldAtPoint(*pmesh, T, params.probe_x, params.probe_y[i]);
            probe_vals.push_back(val);
         }

         const double mdot_surf = ComputeTopBoundaryMassFlux(*pmesh,
                                                             fes_T,
                                                             fes_p,
                                                             T,
                                                             p,
                                                             material,
                                                             state_manager,
                                                             gravity,
                                                             params.bdr_attr_top);
         const double front98 = ComputeFrontDepth(*pmesh, tau_gf, xmid,
                                                  bounds.ymax, bounds.ymin, 0.98);
         const double front2 = ComputeFrontDepth(*pmesh, tau_gf, xmid,
                                                 bounds.ymax, bounds.ymin, 0.02);

         if (myid == 0)
         {
            probes_csv << time << "," << wallT;
            for (double v : probe_vals)
            {
               probes_csv << "," << v;
            }
            probes_csv << "\n";

            mass_csv << time << "," << mdot_surf << ","
                     << 0.0 << ","        // m_dot_c placeholder
                     << front98 << ","
                     << front2 << ","
                     << 0.0 << "\n";      // recession placeholder

            probes_csv.flush();
            mass_csv.flush();
         }

         if (params.save_paraview && (step % params.output_every == 0))
         {
            paraview_dc.SetCycle(step);
            paraview_dc.SetTime(time);
            paraview_dc.Save();
         }
      };

      write_outputs(0, 0.0);

      MPI_Barrier(MPI_COMM_WORLD);
      const double setup_time_local = ElapsedSec(setup_t0, steady_clock_t::now());
      double setup_time_global = 0.0;
      MPI_Allreduce(&setup_time_local, &setup_time_global, 1, MPI_DOUBLE,
                    MPI_MAX, MPI_COMM_WORLD);

      const int nsteps = static_cast<int>(ceil(params.t_final / params.dt - 1.0e-12));
      if (myid == 0)
      {
         cout << "Time steps: " << nsteps
              << ", nominal final time: " << (nsteps * params.dt) << endl;
      }

      double timing_sum_bc = 0.0;
      double timing_sum_newton = 0.0;
      double timing_sum_newton_res = 0.0;
      double timing_sum_newton_jac = 0.0;
      double timing_sum_newton_lin = 0.0;
      double timing_sum_newton_upd = 0.0;
      double timing_sum_state = 0.0;
      double timing_sum_output = 0.0;
      double timing_sum_step = 0.0;

      double time = 0.0;
      for (int step = 1; step <= nsteps; ++step)
      {
         const auto step_t0 = steady_clock_t::now();
         const double t_next = min(params.t_final, time + params.dt);
         const double dt_step = t_next - time;
         time = t_next;

         T_old = T;
         p_old = p;
         tp_integrator->SetTimeStep(dt_step);

         // Build the initial Newton iterate from previous solution with updated BCs.
         const auto bc_t0 = steady_clock_t::now();
         T.GetTrueDofs(xb.GetBlock(0));
         p.GetTrueDofs(xb.GetBlock(1));

         const double Tbc = EvaluateSchedule(params.top_temperature_schedule, time);
         xb.GetBlock(0).SetSubVector(ess_tdof_T, Tbc);
         xb.GetBlock(1).SetSubVector(ess_tdof_p, params.top_pressure);
         const double step_bc_sec = ElapsedSec(bc_t0, steady_clock_t::now());

         auto enforce_bc = [&](Vector &x_true)
         {
            BlockVector x_true_b(x_true, block_true_offsets);
            x_true_b.GetBlock(0).SetSubVector(ess_tdof_T, Tbc);
            x_true_b.GetBlock(1).SetSubVector(ess_tdof_p, params.top_pressure);
         };
         auto log_iteration = [&](const newton_utils::NewtonIterationInfo &it)
         {
            if (myid == 0)
            {
               newton_csv << step << "," << time << "," << it.iter << ","
                          << it.residual_norm << "," << it.residual_norm0 << ","
                          << it.relative_residual << ","
                          << it.update_norm << "," << it.update_norm0 << ","
                          << it.relative_update << ","
                          << (it.converged ? 1 : 0) << "\n";

               if (params.newton_print_level > 0 && !it.converged)
               {
                  cout << "NR iteration " << it.iter << ":\n"
                       << "|R|/|R0|= " << it.relative_residual << "\n"
                       << "|R|= " << it.residual_norm << "\n"
                       << "|du|/|du0|= " << it.relative_update << "\n"
                       << "|du|= " << it.update_norm << endl;
               }
            }
         };

         const auto newton_t0 = steady_clock_t::now();
         const newton_utils::NewtonSolveResult newton_result =
            newton_solver.Solve(block_form, x, enforce_bc, log_iteration, step);
         const double step_newton_sec = ElapsedSec(newton_t0, steady_clock_t::now());
         if (myid == 0)
         {
            newton_csv.flush();
         }

         if (!newton_result.converged)
         {
            throw runtime_error("Newton did not converge at step " + to_string(step) +
                                ", final residual=" + to_string(newton_result.final_residual) +
                                ", final relative residual=" +
                                to_string(newton_result.final_relative_residual));
         }

         T.SetFromTrueDofs(xb.GetBlock(0));
         p.SetFromTrueDofs(xb.GetBlock(1));

         const auto state_t0 = steady_clock_t::now();
         AdvanceInternalStates(material, state_manager,
                               fes_T, fes_p,
                               T, p,
                               quad_order,
                               dt_step);
         const double step_state_sec = ElapsedSec(state_t0, steady_clock_t::now());

         const auto output_t0 = steady_clock_t::now();
         write_outputs(step, time);
         const double step_output_sec = ElapsedSec(output_t0, steady_clock_t::now());
         const double step_total_sec = ElapsedSec(step_t0, steady_clock_t::now());

         double step_local[9] = {step_bc_sec,
                                 step_newton_sec,
                                 newton_result.timing.residual_eval_sec,
                                 newton_result.timing.jacobian_sec,
                                 newton_result.timing.linear_sec,
                                 newton_result.timing.update_sec,
                                 step_state_sec,
                                 step_output_sec,
                                 step_total_sec};
         double step_global[9] = {0.0};
         MPI_Allreduce(step_local, step_global, 9, MPI_DOUBLE, MPI_MAX,
                       MPI_COMM_WORLD);

         if (myid == 0)
         {
            timing_sum_bc += step_global[0];
            timing_sum_newton += step_global[1];
            timing_sum_newton_res += step_global[2];
            timing_sum_newton_jac += step_global[3];
            timing_sum_newton_lin += step_global[4];
            timing_sum_newton_upd += step_global[5];
            timing_sum_state += step_global[6];
            timing_sum_output += step_global[7];
            timing_sum_step += step_global[8];

            timing_step_csv << step << "," << step_global[0] << ","
                            << step_global[1] << "," << step_global[2] << ","
                            << step_global[3] << "," << step_global[4] << ","
                            << step_global[5] << "," << step_global[6] << ","
                            << step_global[7] << "," << step_global[8] << "\n";
            timing_step_csv.flush();
         }
      }

      const double run_time_local = ElapsedSec(run_t0, steady_clock_t::now());
      double run_time_global = 0.0;
      MPI_Allreduce(&run_time_local, &run_time_global, 1, MPI_DOUBLE,
                    MPI_MAX, MPI_COMM_WORLD);

      if (myid == 0)
      {
         ofstream tol_csv(filesystem::path(params.output_path) / "fiat_error_tolerances.csv");
         tol_csv << "signal,tolerance\n";
         tol_csv << "temperature_rmse_max," << params.tol_temp_rmse << "\n";
         tol_csv << "temperature_max_abs_max," << params.tol_temp_max_abs << "\n";
         tol_csv << "m_dot_g_peak_rel_error_max," << params.tol_mdot_peak_rel << "\n";
         tol_csv << "m_dot_g_peak_time_error_max," << params.tol_mdot_peak_time << "\n";
         tol_csv << "front98_rmse_max," << params.tol_front98_rmse << "\n";
         tol_csv << "front2_rmse_max," << params.tol_front2_rmse << "\n";

         ofstream timing_summary_csv(filesystem::path(params.output_path) /
                                     params.timing_summary_csv);
         if (!timing_summary_csv)
         {
            throw runtime_error("Failed to open timing summary CSV.");
         }
         timing_summary_csv << "metric,seconds\n";
         timing_summary_csv << setprecision(16);
         timing_summary_csv << "setup_time_maxrank," << setup_time_global << "\n";
         timing_summary_csv << "run_time_maxrank," << run_time_global << "\n";
         timing_summary_csv << "sum_step_time_maxrank," << timing_sum_step << "\n";
         timing_summary_csv << "sum_bc_time_maxrank," << timing_sum_bc << "\n";
         timing_summary_csv << "sum_newton_time_maxrank," << timing_sum_newton << "\n";
         timing_summary_csv << "sum_newton_residual_eval_time_maxrank,"
                            << timing_sum_newton_res << "\n";
         timing_summary_csv << "sum_newton_jacobian_time_maxrank,"
                            << timing_sum_newton_jac << "\n";
         timing_summary_csv << "sum_newton_linear_time_maxrank,"
                            << timing_sum_newton_lin << "\n";
         timing_summary_csv << "sum_newton_update_time_maxrank,"
                            << timing_sum_newton_upd << "\n";
         timing_summary_csv << "sum_state_advance_time_maxrank,"
                            << timing_sum_state << "\n";
         timing_summary_csv << "sum_output_time_maxrank," << timing_sum_output
                            << "\n";
         timing_summary_csv << "avg_step_time_maxrank,"
                            << (timing_sum_step / static_cast<double>(max(1, nsteps)))
                            << "\n";

         cout << "Timing summary (max over ranks):" << endl
              << "  setup: " << setup_time_global << " s\n"
              << "  run total: " << run_time_global << " s\n"
              << "  step total sum: " << timing_sum_step << " s\n"
              << "  step avg: "
              << (timing_sum_step / static_cast<double>(max(1, nsteps)))
              << " s\n"
              << "  bc: " << timing_sum_bc << " s\n"
              << "  newton: " << timing_sum_newton << " s\n"
              << "    residual eval: " << timing_sum_newton_res << " s\n"
              << "    jacobian: " << timing_sum_newton_jac << " s\n"
              << "    linear solve: " << timing_sum_newton_lin << " s\n"
              << "    update: " << timing_sum_newton_upd << " s\n"
              << "  state advance: " << timing_sum_state << " s\n"
              << "  output: " << timing_sum_output << " s" << endl;
      }
   }
   catch (const exception &e)
   {
      if (myid == 0)
      {
         cerr << "Error: " << e.what() << endl;
      }
      exit_code = 3;
   }

   MFEMFinalizePetsc();
   return exit_code;
}

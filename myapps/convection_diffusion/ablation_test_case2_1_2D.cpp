#include "mfem.hpp"
#include "bprime_table.hpp"
#include "newton_petsc_solver.hpp"
#include "surface_bc_schedule.hpp"
#include "tacot_material.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <tuple>
#include <utility>
#include <vector>

using namespace std;
using namespace mfem;

namespace
{

enum class PatoCompatMode
{
   Off = 0,
   CoolingExact = 1
};

PatoCompatMode ParsePatoCompatMode(std::string mode)
{
   transform(mode.begin(), mode.end(), mode.begin(),
             [](unsigned char c) { return static_cast<char>(tolower(c)); });
   if (mode == "off")
   {
      return PatoCompatMode::Off;
   }
   if (mode == "cooling_exact")
   {
      return PatoCompatMode::CoolingExact;
   }
   throw runtime_error(
      "pato_compat_mode must be either \"off\" or \"cooling_exact\".");
}

const char *PatoCompatModeName(const PatoCompatMode mode)
{
   switch (mode)
   {
      case PatoCompatMode::Off:
         return "off";
      case PatoCompatMode::CoolingExact:
         return "cooling_exact";
   }
   return "off";
}

struct DriverParams
{
   string mesh_file = "Mesh/ablation_strip.msh";
   string material_file = "Input/material_tacot_case2_1.yaml";

   int order = 1;
   int serial_ref_levels = 0;
   int par_ref_levels = 0;

   double dt = 1.0e-2;
   double t_final = 120.0;

   double newton_abs_tol = 1.0e-8;
   double newton_rel_tol = 1.0e-6;
   int newton_max_iter = 20;
   int newton_jacobian_rebuild_freq = 1;
   int newton_print_level = 1;
   // Consistency check validates both domain and surface Jacobians.
   bool jacobian_check = false;
   double jacobian_check_abs_tol = 1.0e-6;
   double jacobian_check_rel_tol = 1.0e-4;

   string petsc_options_file = "Input/petsc_ablation_case2_1.opts";
   string ksp_prefix = "ablation21_ls_";
   int petsc_ksp_print_level = 0;

   int output_every = 10;
   string output_path = "ParaView/ablation_case2_1";
   string collection_name = "ablation_test_case2_1_2D";
   string probes_csv = "temperature_probes.csv";
   string mass_csv = "mass_metrics.csv";
   string boundary_csv = "boundary_diagnostics.csv";
   string newton_csv = "newton_history_ablation_case2_1_2D.csv";
   string timing_step_csv = "driver_timing_per_step.csv";
   string timing_summary_csv = "driver_timing_summary.csv";
   bool save_paraview = true;
   string restart_read_file = "";
   string restart_write_file = "";
   int restart_write_every = 0;
   double restart_write_at_time = std::numeric_limits<double>::quiet_NaN();

   int bdr_attr_top = 1;
   int bdr_attr_bottom = 2;
   int bdr_attr_sides = 3;

   string bprime_table_file =
      "/home/quinnchr/Downloads/pato-3.1/data/Environments/Tables/TACOT-Earth-1atm";
   string boundary_conditions_file = "Input/boundary_conditions_ablation_case2_1.dat";
   // Top thermal BC mode: "surface_energy_balance" or "temperature_dirichlet".
   string top_thermal_bc = "surface_energy_balance";
   // Used only when top_thermal_bc == "temperature_dirichlet".
   double top_temperature_value = 300.0;
   // Optional time-temperature table for top_thermal_bc=temperature_dirichlet.
   // Columns: time(s) temperature(K).
   string top_temperature_file = "";

   // Surface energy-balance model (Test Case 2.1).
   double lambda = 0.5;
   double q_rad = 0.0;
   double T_background = 300.0;
   double T_edge = 300.0;
   double hconv = 0.0;
   double emissivity = std::numeric_limits<double>::quiet_NaN();
   double absorptivity = std::numeric_limits<double>::quiet_NaN();
   double stefan_boltzmann = 5.670374419e-8;
   bool strict_case2_1 = true;
   PatoCompatMode pato_compat_mode = PatoCompatMode::Off;

   double gravity_x = 0.0;
   double gravity_y = 0.0;

   double probe_x = 0.005;
   vector<double> probe_y = {0.05, 0.049, 0.048, 0.046, 0.042, 0.038, 0.034, 0.026, 0.01};

   string amaryllis_energy_file =
      "/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_2.x/data/ref/Amaryllis/Amaryllis_Energy_TestCase_2.1.txt";
   string amaryllis_mass_file =
      "/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_2.x/data/ref/Amaryllis/Amaryllis_Mass_TestCase_2.1.txt";

   // Acceptance tolerances for compare_ablation_case2_1.py
   double tol_temp_rmse = 250.0;
   double tol_temp_max_abs = 500.0;
   double tol_mdot_rmse = 0.02;
   double tol_mdot_max_abs = 0.06;
   double tol_mdot_peak_rel = 0.5;
   double tol_mdot_peak_time = 10.0;
   double tol_front98_max_abs = 0.01;
   double tol_front98_rmse = 0.01;
   double tol_front2_max_abs = 0.01;
   double tol_front2_rmse = 0.01;
   double tol_mdot_c_max_abs = 1.0e-8;
   double tol_recession_max_abs = 1.0e-8;
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

string NormalizeTopThermalBC(string mode)
{
   transform(mode.begin(), mode.end(), mode.begin(),
             [](unsigned char c) { return static_cast<char>(tolower(c)); });

   if (mode == "seb")
   {
      return "surface_energy_balance";
   }
   if (mode == "temp_dirichlet" || mode == "temperature")
   {
      return "temperature_dirichlet";
   }
   return mode;
}

class TopTemperatureSchedule
{
public:
   void LoadFromFile(const string &path)
   {
      times_.clear();
      values_.clear();

      ifstream in(path);
      if (!in)
      {
         throw runtime_error("Failed to open top temperature schedule: " + path);
      }

      string line;
      int line_no = 0;
      while (std::getline(in, line))
      {
         ++line_no;
         const size_t comment_pos = line.find("//");
         if (comment_pos != string::npos)
         {
            line = line.substr(0, comment_pos);
         }

         std::istringstream iss(line);
         double t = 0.0;
         double v = 0.0;
         if (!(iss >> t >> v))
         {
            continue;
         }
         if (t < 0.0)
         {
            throw runtime_error("Negative time in top temperature schedule at line " +
                                to_string(line_no) + ": " + path);
         }
         if (!times_.empty() && t < times_.back())
         {
            throw runtime_error("Top temperature schedule times must be nondecreasing at line " +
                                to_string(line_no) + ": " + path);
         }
         times_.push_back(t);
         values_.push_back(v);
      }

      if (times_.empty())
      {
         throw runtime_error("Top temperature schedule is empty: " + path);
      }
   }

   bool Empty() const { return times_.empty(); }

   double Eval(const double time) const
   {
      if (times_.empty())
      {
         throw runtime_error("Top temperature schedule Eval() called before LoadFromFile().");
      }
      if (time <= times_.front())
      {
         return values_.front();
      }
      if (time >= times_.back())
      {
         return values_.back();
      }

      auto it = std::upper_bound(times_.begin(), times_.end(), time);
      const int i1 = static_cast<int>(it - times_.begin());
      const int i0 = i1 - 1;
      const double t0 = times_[i0];
      const double t1 = times_[i1];
      const double v0 = values_[i0];
      const double v1 = values_[i1];
      if (std::abs(t1 - t0) < 1.0e-14)
      {
         return v1;
      }
      const double alpha = (time - t0) / (t1 - t0);
      return (1.0 - alpha) * v0 + alpha * v1;
   }

private:
   vector<double> times_;
   vector<double> values_;
};

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
   if (n["newton_jacobian_rebuild_freq"])
   {
      p.newton_jacobian_rebuild_freq = n["newton_jacobian_rebuild_freq"].as<int>();
   }
   if (n["newton_print_level"]) { p.newton_print_level = n["newton_print_level"].as<int>(); }
   if (n["jacobian_check"]) { p.jacobian_check = n["jacobian_check"].as<bool>(); }
   if (n["jacobian_check_abs_tol"])
   {
      p.jacobian_check_abs_tol = n["jacobian_check_abs_tol"].as<double>();
   }
   if (n["jacobian_check_rel_tol"])
   {
      p.jacobian_check_rel_tol = n["jacobian_check_rel_tol"].as<double>();
   }

   if (n["petsc_options_file"]) { p.petsc_options_file = n["petsc_options_file"].as<string>(); }
   if (n["ksp_prefix"]) { p.ksp_prefix = n["ksp_prefix"].as<string>(); }
   if (n["petsc_ksp_print_level"]) { p.petsc_ksp_print_level = n["petsc_ksp_print_level"].as<int>(); }

   if (n["output_every"]) { p.output_every = n["output_every"].as<int>(); }
   if (n["output_path"]) { p.output_path = n["output_path"].as<string>(); }
   if (n["collection_name"]) { p.collection_name = n["collection_name"].as<string>(); }
   if (n["probes_csv"]) { p.probes_csv = n["probes_csv"].as<string>(); }
   if (n["mass_csv"]) { p.mass_csv = n["mass_csv"].as<string>(); }
   if (n["boundary_csv"]) { p.boundary_csv = n["boundary_csv"].as<string>(); }
   if (n["newton_csv"]) { p.newton_csv = n["newton_csv"].as<string>(); }
   if (n["timing_step_csv"]) { p.timing_step_csv = n["timing_step_csv"].as<string>(); }
   if (n["timing_summary_csv"]) { p.timing_summary_csv = n["timing_summary_csv"].as<string>(); }
   if (n["save_paraview"]) { p.save_paraview = n["save_paraview"].as<bool>(); }
   if (n["restart_read_file"]) { p.restart_read_file = n["restart_read_file"].as<string>(); }
   if (n["restart_write_file"]) { p.restart_write_file = n["restart_write_file"].as<string>(); }
   if (n["restart_write_every"]) { p.restart_write_every = n["restart_write_every"].as<int>(); }
   if (n["restart_write_at_time"])
   {
      p.restart_write_at_time = n["restart_write_at_time"].as<double>();
   }

   if (n["bdr_attr_top"]) { p.bdr_attr_top = n["bdr_attr_top"].as<int>(); }
   if (n["bdr_attr_bottom"]) { p.bdr_attr_bottom = n["bdr_attr_bottom"].as<int>(); }
   if (n["bdr_attr_sides"]) { p.bdr_attr_sides = n["bdr_attr_sides"].as<int>(); }

   if (n["bprime_table_file"]) { p.bprime_table_file = n["bprime_table_file"].as<string>(); }
   if (n["boundary_conditions_file"]) { p.boundary_conditions_file = n["boundary_conditions_file"].as<string>(); }
   if (n["top_thermal_bc"]) { p.top_thermal_bc = n["top_thermal_bc"].as<string>(); }
   if (n["top_temperature_value"]) { p.top_temperature_value = n["top_temperature_value"].as<double>(); }
   if (n["top_temperature_file"]) { p.top_temperature_file = n["top_temperature_file"].as<string>(); }
   if (n["lambda"]) { p.lambda = n["lambda"].as<double>(); }
   if (n["q_rad"]) { p.q_rad = n["q_rad"].as<double>(); }
   if (n["T_background"]) { p.T_background = n["T_background"].as<double>(); }
   if (n["T_edge"]) { p.T_edge = n["T_edge"].as<double>(); }
   if (n["hconv"]) { p.hconv = n["hconv"].as<double>(); }
   if (n["emissivity"]) { p.emissivity = n["emissivity"].as<double>(); }
   if (n["absorptivity"]) { p.absorptivity = n["absorptivity"].as<double>(); }
   if (n["stefan_boltzmann"]) { p.stefan_boltzmann = n["stefan_boltzmann"].as<double>(); }
   if (n["strict_case2_1"]) { p.strict_case2_1 = n["strict_case2_1"].as<bool>(); }
   if (n["pato_compat_mode"])
   {
      p.pato_compat_mode = ParsePatoCompatMode(n["pato_compat_mode"].as<string>());
   }

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

   if (n["amaryllis_energy_file"]) { p.amaryllis_energy_file = n["amaryllis_energy_file"].as<string>(); }
   if (n["amaryllis_mass_file"]) { p.amaryllis_mass_file = n["amaryllis_mass_file"].as<string>(); }

   if (n["acceptance"])
   {
      YAML::Node a = n["acceptance"];
      if (a["temperature_rmse_max"]) { p.tol_temp_rmse = a["temperature_rmse_max"].as<double>(); }
      if (a["temperature_max_abs_max"]) { p.tol_temp_max_abs = a["temperature_max_abs_max"].as<double>(); }
      if (a["m_dot_g_rmse_max"]) { p.tol_mdot_rmse = a["m_dot_g_rmse_max"].as<double>(); }
      if (a["m_dot_g_max_abs_max"]) { p.tol_mdot_max_abs = a["m_dot_g_max_abs_max"].as<double>(); }
      if (a["m_dot_g_peak_rel_error_max"]) { p.tol_mdot_peak_rel = a["m_dot_g_peak_rel_error_max"].as<double>(); }
      if (a["m_dot_g_peak_time_error_max"]) { p.tol_mdot_peak_time = a["m_dot_g_peak_time_error_max"].as<double>(); }
      if (a["front98_max_abs_max"]) { p.tol_front98_max_abs = a["front98_max_abs_max"].as<double>(); }
      if (a["front98_rmse_max"]) { p.tol_front98_rmse = a["front98_rmse_max"].as<double>(); }
      if (a["front2_max_abs_max"]) { p.tol_front2_max_abs = a["front2_max_abs_max"].as<double>(); }
      if (a["front2_rmse_max"]) { p.tol_front2_rmse = a["front2_rmse_max"].as<double>(); }
      if (a["m_dot_c_max_abs_max"]) { p.tol_mdot_c_max_abs = a["m_dot_c_max_abs_max"].as<double>(); }
      if (a["recession_max_abs_max"]) { p.tol_recession_max_abs = a["recession_max_abs_max"].as<double>(); }
   }

   if (p.dt <= 0.0) { throw runtime_error("dt must be > 0."); }
   if (p.t_final < 0.0) { throw runtime_error("t_final must be >= 0."); }
   if (p.order < 1) { throw runtime_error("order must be >= 1."); }
   if (p.newton_max_iter < 1) { throw runtime_error("newton_max_iter must be >= 1."); }
   if (p.newton_jacobian_rebuild_freq < 1)
   {
      throw runtime_error("newton_jacobian_rebuild_freq must be >= 1.");
   }
   if (p.jacobian_check_abs_tol < 0.0)
   {
      throw runtime_error("jacobian_check_abs_tol must be >= 0.");
   }
   if (p.jacobian_check_rel_tol < 0.0)
   {
      throw runtime_error("jacobian_check_rel_tol must be >= 0.");
   }
   if (p.restart_write_every < 0)
   {
      throw runtime_error("restart_write_every must be >= 0.");
   }
   if (std::isfinite(p.restart_write_at_time) && p.restart_write_at_time < 0.0)
   {
      throw runtime_error("restart_write_at_time must be >= 0 when provided.");
   }
   if (p.restart_write_file.empty() &&
       (p.restart_write_every > 0 ||
        std::isfinite(p.restart_write_at_time)))
   {
      throw runtime_error(
         "restart_write_file must be set when restart write triggers are enabled.");
   }
   if (p.lambda < 0.0) { throw runtime_error("lambda must be >= 0."); }
   if (p.probe_y.size() < 2) { throw runtime_error("probe_y must contain wall and at least one in-depth probe."); }

   p.top_thermal_bc = NormalizeTopThermalBC(p.top_thermal_bc);
   if (p.top_thermal_bc != "surface_energy_balance" &&
       p.top_thermal_bc != "temperature_dirichlet")
   {
      throw runtime_error(
         "top_thermal_bc must be either \"surface_energy_balance\" "
         "or \"temperature_dirichlet\".");
   }
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

struct SurfaceFluxModelParams
{
   double lambda = 0.5;
   double q_rad = 0.0;
   double T_background = 300.0;
   double T_edge = 300.0;
   double hconv = 0.0;
   double emissivity = 1.0;
   double absorptivity = 1.0;
   bool use_emissivity_override = false;
   bool use_absorptivity_override = false;
   double stefan_boltzmann = 5.670374419e-8;
   bool strict_case2_1 = true;
   PatoCompatMode pato_compat_mode = PatoCompatMode::Off;
};

struct SurfaceFluxTerms
{
   double BprimeG = 0.0;
   double BprimeC = 0.0;
   double h_w = 0.0;
   double emissivity = 1.0;
   double absorptivity = 1.0;
   double reflectivity = 0.0;
   double blowing_correction = 1.0;
   double q_conv = 0.0;
   // Boundary term contribution after accounting for h_g*m_dot transport
   // already present in the domain energy operator.
   double q_adv_pyro = 0.0;
   double q_adv_char = 0.0;
   double q_rad_emit = 0.0;
   double q_rad_abs = 0.0;
   double q_surf = 0.0;
};

enum class SurfaceFluxBranch
{
   ChemistryBlowing = 0,
   ChemistryNoBlowing = 1,
   CoolingStandard = 2,
   CoolingExact = 3
};

const char *SurfaceFluxBranchName(const SurfaceFluxBranch branch)
{
   switch (branch)
   {
      case SurfaceFluxBranch::ChemistryBlowing:
         return "chemistry_blowing";
      case SurfaceFluxBranch::ChemistryNoBlowing:
         return "chemistry_no_blowing";
      case SurfaceFluxBranch::CoolingStandard:
         return "cooling_standard";
      case SurfaceFluxBranch::CoolingExact:
         return "cooling_exact";
   }
   return "unknown";
}

int SurfaceFluxBranchIndex(const SurfaceFluxBranch branch)
{
   return static_cast<int>(branch);
}

SurfaceFluxBranch ClassifySurfaceFluxBranch(
   const SurfaceBCSchedule::BoundaryState &bc_state,
   const SurfaceFluxModelParams &model)
{
   const bool cooling_exact =
      (model.pato_compat_mode == PatoCompatMode::CoolingExact &&
       bc_state.chemistryOn == 0);
   if (cooling_exact)
   {
      return SurfaceFluxBranch::CoolingExact;
   }

   const bool chemistry_on = (bc_state.chemistryOn != 0);
   const double rhoeUeCH = std::max(0.0, bc_state.rhoeUeCH);
   if (chemistry_on)
   {
      if (rhoeUeCH > 1.0e-14)
      {
         return SurfaceFluxBranch::ChemistryBlowing;
      }
      return SurfaceFluxBranch::ChemistryNoBlowing;
   }
   return SurfaceFluxBranch::CoolingStandard;
}

struct BlowingCorrectionValue
{
   double value = 1.0;
   double dvalue_dBg = 0.0;
};

struct SurfaceBlowingState
{
   double BprimeG = 0.0;
   double dBprimeG_dmdot = 0.0;
   double blowing = 1.0;
   double dblowing_dmdot = 0.0;
   bool nonsmooth = false;
};

struct SurfaceFluxLinearization
{
   SurfaceFluxTerms terms;
   SurfaceFluxBranch branch = SurfaceFluxBranch::CoolingStandard;
   double dq_dm_dot = 0.0;
   double dq_dh_g = 0.0;
   double dq_dT_w = 0.0;
   double dq_dT_eval = 0.0;
   double dq_demissivity = 0.0;
   double dq_dabsorptivity = 0.0;
   bool nonsmooth = false;
};

struct SurfaceBoundaryDiagnostics
{
   double m_dot_g_surf = 0.0;
   double BprimeG_surf = 0.0;
   double BprimeC_surf = 0.0;
   double h_w_surf = 0.0;
   double emissivity_surf = 0.0;
   double absorptivity_surf = 0.0;
   double reflectivity_surf = 0.0;
   double blowing_correction_surf = 0.0;
   double q_conv_surf = 0.0;
   double q_adv_pyro_surf = 0.0;
   double q_rad_emit_surf = 0.0;
   double q_rad_abs_surf = 0.0;
   double q_surf = 0.0;
};

BlowingCorrectionValue ComputeBlowingCorrectionWithDerivative(const double BprimeG,
                                                              const double lambda)
{
   BlowingCorrectionValue out;

   const double lam = std::max(lambda, 0.0);
   const double Bg_pos = std::max(BprimeG, 0.0);
   const double arg = 2.0 * lam * Bg_pos;
   if (arg < 1.0e-10)
   {
      const double arg2 = arg * arg;
      out.value = 1.0 - 0.5 * arg + (1.0 / 3.0) * arg2;
      if (Bg_pos > 0.0)
      {
         const double df_darg = -0.5 + (2.0 / 3.0) * arg;
         out.dvalue_dBg = df_darg * (2.0 * lam);
      }
      return out;
   }

   out.value = std::log1p(arg) / arg;
   if (Bg_pos > 0.0)
   {
      const double df_darg = (arg / (1.0 + arg) - std::log1p(arg)) / (arg * arg);
      out.dvalue_dBg = df_darg * (2.0 * lam);
   }
   return out;
}

double ComputeBlowingCorrection(const double BprimeG, const double lambda)
{
   return ComputeBlowingCorrectionWithDerivative(BprimeG, lambda).value;
}

SurfaceBlowingState SolveSurfaceBlowingState(const double m_dot_g_w,
                                             const double rhoeUeCH,
                                             const double lambda,
                                             const bool enable_blowing)
{
   SurfaceBlowingState out;
   if (!enable_blowing)
   {
      return out;
   }

   for (int it = 0; it < 3; ++it)
   {
      const double blowing_eff = std::max(out.blowing, 1.0e-12);
      const bool blowing_clamped = (out.blowing <= 1.0e-12);
      const double dbeff_dm = blowing_clamped ? 0.0 : out.dblowing_dmdot;
      if (blowing_clamped) { out.nonsmooth = true; }

      const double denom = rhoeUeCH * blowing_eff;
      const double raw_Bg = m_dot_g_w / denom;
      if (raw_Bg > 0.0)
      {
         out.BprimeG = raw_Bg;
         out.dBprimeG_dmdot =
            (1.0 / denom) - (m_dot_g_w / (denom * blowing_eff)) * dbeff_dm;
      }
      else
      {
         out.BprimeG = 0.0;
         out.dBprimeG_dmdot = 0.0;
         out.nonsmooth = true;
      }

      const BlowingCorrectionValue corr =
         ComputeBlowingCorrectionWithDerivative(out.BprimeG, lambda);
      out.blowing = corr.value;
      out.dblowing_dmdot = corr.dvalue_dBg * out.dBprimeG_dmdot;
   }

   return out;
}

SurfaceFluxLinearization EvaluateSurfaceFluxTermsLinearized(
   const double m_dot_g_w,
   const double h_g,
   const double T_w,
   const double T_eval,
   const double emissivity,
   const double absorptivity,
   const double reflectivity,
   const SurfaceBCSchedule::BoundaryState &bc_state,
   const BPrimeTable &bprime_table,
   const SurfaceFluxModelParams &model)
{
   SurfaceFluxLinearization out;
   out.branch = ClassifySurfaceFluxBranch(bc_state, model);
   if (out.branch == SurfaceFluxBranch::ChemistryBlowing)
   {
      // This branch combines fixed-point blowing updates with piecewise table
      // interpolation, so exact directional derivatives are generally non-smooth.
      out.nonsmooth = true;
   }
   out.terms.emissivity = emissivity;
   out.terms.absorptivity = absorptivity;
   out.terms.reflectivity = reflectivity;

   const double hconv_eff = bc_state.has_hconv ? bc_state.hconv : model.hconv;
   const double Tedge_eff = bc_state.has_Tedge ? bc_state.Tedge : model.T_edge;
   const double sigma = model.stefan_boltzmann;
   const double T_bg4 = std::pow(model.T_background, 4.0);

   if (out.branch == SurfaceFluxBranch::CoolingExact)
   {
      out.terms.BprimeG = 0.0;
      out.terms.BprimeC = 0.0;
      out.terms.h_w = 0.0;
      out.terms.blowing_correction = 1.0;

      out.terms.q_conv = hconv_eff * (Tedge_eff - T_eval);
      out.terms.q_adv_pyro = -m_dot_g_w * h_g;
      out.terms.q_adv_char = 0.0;
      out.terms.q_rad_emit =
         -out.terms.emissivity * sigma * (std::pow(T_eval, 4.0) - T_bg4);
      out.terms.q_rad_abs = out.terms.absorptivity * model.q_rad;
      out.terms.q_surf = out.terms.q_conv + out.terms.q_adv_pyro +
                         out.terms.q_adv_char + out.terms.q_rad_emit +
                         out.terms.q_rad_abs;

      out.dq_dm_dot = -h_g;
      out.dq_dh_g = -m_dot_g_w;
      out.dq_dT_w = 0.0;
      out.dq_dT_eval = -hconv_eff - out.terms.emissivity * sigma * 4.0 *
                                        std::pow(T_eval, 3.0);
      out.dq_demissivity = -sigma * (std::pow(T_eval, 4.0) - T_bg4);
      out.dq_dabsorptivity = model.q_rad;
      return out;
   }

   const bool chemistry_on = (bc_state.chemistryOn != 0);
   const double rhoeUeCH = std::max(0.0, bc_state.rhoeUeCH);
   const bool blowing_active = (out.branch == SurfaceFluxBranch::ChemistryBlowing);
   const SurfaceBlowingState blowing =
      SolveSurfaceBlowingState(m_dot_g_w, rhoeUeCH, model.lambda, blowing_active);

   const BPrimeTable::LookupDerivatives lookup =
      bprime_table.LookupWithDerivatives(bc_state.p_w, blowing.BprimeG, T_w);
   out.nonsmooth = out.nonsmooth || blowing.nonsmooth ||
                   lookup.clamped_bg || lookup.clamped_t || lookup.nonsmooth_bg;
   const double h_w = chemistry_on ? lookup.hw : 0.0;
   const double dh_w_dT = chemistry_on ? lookup.dhw_dT : 0.0;
   const double dh_w_dmdot =
      chemistry_on ? (lookup.dhw_dbg * blowing.dBprimeG_dmdot) : 0.0;

   out.terms.BprimeG = blowing.BprimeG;
   out.terms.BprimeC = (chemistry_on && !model.strict_case2_1) ? lookup.bc : 0.0;
   out.terms.h_w = h_w;
   out.terms.blowing_correction = blowing.blowing;

   out.terms.q_rad_emit =
      -out.terms.emissivity * sigma * (std::pow(T_w, 4.0) - T_bg4);
   out.terms.q_rad_abs = out.terms.absorptivity * model.q_rad;
   out.terms.q_adv_pyro = -m_dot_g_w * h_w;
   out.terms.q_adv_char = 0.0;

   if (chemistry_on)
   {
      out.terms.q_conv = rhoeUeCH * blowing.blowing * (bc_state.h_r - h_w);
   }
   else
   {
      out.terms.q_conv = hconv_eff * (Tedge_eff - T_w);
   }

   out.terms.q_surf = out.terms.q_conv + out.terms.q_adv_pyro + out.terms.q_adv_char +
                      out.terms.q_rad_emit + out.terms.q_rad_abs;

   const double dq_rad_dT =
      -out.terms.emissivity * sigma * 4.0 * std::pow(T_w, 3.0);
   const double dq_adv_dmdot = -h_w - m_dot_g_w * dh_w_dmdot;
   const double dq_adv_dT = -m_dot_g_w * dh_w_dT;
   double dq_conv_dmdot = 0.0;
   double dq_conv_dT = 0.0;
   if (chemistry_on)
   {
      dq_conv_dmdot = rhoeUeCH *
                      (blowing.dblowing_dmdot * (bc_state.h_r - h_w) -
                       blowing.blowing * dh_w_dmdot);
      dq_conv_dT = -rhoeUeCH * blowing.blowing * dh_w_dT;
   }
   else
   {
      dq_conv_dmdot = 0.0;
      dq_conv_dT = -hconv_eff;
   }

   out.dq_dm_dot = dq_conv_dmdot + dq_adv_dmdot;
   out.dq_dh_g = 0.0;
   out.dq_dT_w = dq_conv_dT + dq_adv_dT + dq_rad_dT;
   out.dq_dT_eval = 0.0;
   out.dq_demissivity = -sigma * (std::pow(T_w, 4.0) - T_bg4);
   out.dq_dabsorptivity = model.q_rad;

   return out;
}

SurfaceFluxTerms EvaluateSurfaceFluxTerms(const double m_dot_g_w,
                                          const double h_g,
                                          const double T_w,
                                          const double T_eval,
                                          const TACOTMaterial::SolidProperties &solid,
                                          const SurfaceBCSchedule::BoundaryState &bc_state,
                                          const BPrimeTable &bprime_table,
                                          const SurfaceFluxModelParams &model)
{
   const double emissivity =
      model.use_emissivity_override ? model.emissivity : solid.emissivity;
   const double absorptivity =
      model.use_absorptivity_override ? model.absorptivity : solid.absorptivity;
   const SurfaceFluxLinearization flux = EvaluateSurfaceFluxTermsLinearized(
      m_dot_g_w,
      h_g,
      T_w,
      T_eval,
      emissivity,
      absorptivity,
      solid.reflectivity,
      bc_state,
      bprime_table,
      model);
   return flux.terms;
}

struct JacobianCheckOptions
{
   bool enable = false;
   double abs_tol = 1.0e-6;
   double rel_tol = 1.0e-4;
};

double DenseMaxAbs(const DenseMatrix &A)
{
   double v = 0.0;
   for (int i = 0; i < A.Height(); ++i)
   {
      for (int j = 0; j < A.Width(); ++j)
      {
         v = std::max(v, std::abs(A(i, j)));
      }
   }
   return v;
}

double DenseMaxAbsDiff(const DenseMatrix &A, const DenseMatrix &B)
{
   MFEM_VERIFY(A.Height() == B.Height() && A.Width() == B.Width(),
               "DenseMaxAbsDiff size mismatch.");
   double v = 0.0;
   for (int i = 0; i < A.Height(); ++i)
   {
      for (int j = 0; j < A.Width(); ++j)
      {
         v = std::max(v, std::abs(A(i, j) - B(i, j)));
      }
   }
   return v;
}

tuple<double, int, int> DenseMaxAbsDiffWithIndex(const DenseMatrix &A,
                                                 const DenseMatrix &B)
{
   MFEM_VERIFY(A.Height() == B.Height() && A.Width() == B.Width(),
               "DenseMaxAbsDiffWithIndex size mismatch.");
   double v = 0.0;
   int imax = -1;
   int jmax = -1;
   for (int i = 0; i < A.Height(); ++i)
   {
      for (int j = 0; j < A.Width(); ++j)
      {
         const double d = std::abs(A(i, j) - B(i, j));
         if (d > v)
         {
            v = d;
            imax = i;
            jmax = j;
         }
      }
   }
   return {v, imax, jmax};
}

constexpr std::uint64_t kRestartMagic = 0x41424C32525A5441ull;
constexpr std::uint32_t kRestartVersion = 1u;
constexpr double kRestartTimeTol = 1.0e-12;

template <typename T>
void WriteBinaryPod(std::ostream &os, const T &value)
{
   static_assert(std::is_trivially_copyable<T>::value,
                 "WriteBinaryPod requires trivially copyable type.");
   os.write(reinterpret_cast<const char *>(&value), sizeof(T));
   if (!os)
   {
      throw runtime_error("Failed while writing restart data.");
   }
}

template <typename T>
T ReadBinaryPod(std::istream &is)
{
   static_assert(std::is_trivially_copyable<T>::value,
                 "ReadBinaryPod requires trivially copyable type.");
   T value{};
   is.read(reinterpret_cast<char *>(&value), sizeof(T));
   if (!is)
   {
      throw runtime_error("Failed while reading restart data.");
   }
   return value;
}

template <typename T>
void WriteBinaryVector(std::ostream &os, const vector<T> &values)
{
   static_assert(std::is_trivially_copyable<T>::value,
                 "WriteBinaryVector requires trivially copyable type.");
   const std::int64_t n = static_cast<std::int64_t>(values.size());
   WriteBinaryPod(os, n);
   if (n > 0)
   {
      os.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(n * sizeof(T)));
      if (!os)
      {
         throw runtime_error("Failed while writing restart vector payload.");
      }
   }
}

template <typename T>
vector<T> ReadBinaryVector(std::istream &is)
{
   static_assert(std::is_trivially_copyable<T>::value,
                 "ReadBinaryVector requires trivially copyable type.");
   const std::int64_t n = ReadBinaryPod<std::int64_t>(is);
   if (n < 0)
   {
      throw runtime_error("Corrupt restart data: negative vector size.");
   }
   vector<T> out(static_cast<std::size_t>(n));
   if (n > 0)
   {
      is.read(reinterpret_cast<char *>(out.data()),
              static_cast<std::streamsize>(n * sizeof(T)));
      if (!is)
      {
         throw runtime_error("Failed while reading restart vector payload.");
      }
   }
   return out;
}

void WriteMFEMVector(std::ostream &os, const Vector &v)
{
   const std::int64_t n = static_cast<std::int64_t>(v.Size());
   WriteBinaryPod(os, n);
   for (int i = 0; i < v.Size(); ++i)
   {
      WriteBinaryPod(os, v(i));
   }
}

void ReadMFEMVector(std::istream &is, Vector &v)
{
   const std::int64_t n = ReadBinaryPod<std::int64_t>(is);
   if (n < 0)
   {
      throw runtime_error("Corrupt restart data: negative MFEM vector size.");
   }
   v.SetSize(static_cast<int>(n));
   for (int i = 0; i < v.Size(); ++i)
   {
      v(i) = ReadBinaryPod<double>(is);
   }
}

string RestartPathForRank(const string &base_path,
                          const int rank,
                          const int world_size)
{
   if (base_path.empty())
   {
      return base_path;
   }
   if (world_size <= 1)
   {
      return base_path;
   }

   std::ostringstream oss;
   oss << base_path << ".rank" << std::setw(6) << std::setfill('0') << rank;
   return oss.str();
}

void EnsureParentDirectoryExists(const string &path)
{
   std::error_code ec;
   const filesystem::path p(path);
   if (!p.has_parent_path())
   {
      return;
   }
   filesystem::create_directories(p.parent_path(), ec);
   if (ec)
   {
      throw runtime_error("Failed to create restart directory: " +
                          p.parent_path().string() + " (" + ec.message() + ")");
   }
}

void VerifyJacobianBlockClose(const DenseMatrix &analytic,
                              const DenseMatrix &reference_fd,
                              const JacobianCheckOptions &opts,
                              const string &integrator_name,
                              const string &block_name,
                              const int entity_id)
{
   const double max_ref = DenseMaxAbs(reference_fd);
   const auto diff_info = DenseMaxAbsDiffWithIndex(analytic, reference_fd);
   const double max_diff = std::get<0>(diff_info);
   const int imax = std::get<1>(diff_info);
   const int jmax = std::get<2>(diff_info);
   const double threshold = opts.abs_tol + opts.rel_tol * std::max(1.0, max_ref);
   if (max_diff > threshold)
   {
      std::ostringstream oss;
      oss << integrator_name
          << " Jacobian consistency check failed at entity " << entity_id
          << ", block " << block_name
          << ": max_diff=" << max_diff
          << ", max_ref=" << max_ref
          << ", threshold=" << threshold;
      if (imax >= 0 && jmax >= 0)
      {
         oss << ", argmax=(" << imax << "," << jmax << ")"
             << ", analytic=" << analytic(imax, jmax)
             << ", fd=" << reference_fd(imax, jmax);
      }
      throw runtime_error(oss.str());
   }
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

   void SaveToStream(std::ostream &os) const
   {
      WriteBinaryPod(os, static_cast<std::int64_t>(states_.size()));
      for (const auto &elem_states : states_)
      {
         WriteBinaryPod(os, static_cast<std::int64_t>(elem_states.size()));
         for (const auto &st : elem_states)
         {
            WriteBinaryVector(os, st.extent);
            WriteBinaryVector(os, st.extent_old);
            WriteBinaryPod(os, st.dt);
         }
      }

      WriteBinaryVector(os, tau_elem_);
      WriteBinaryVector(os, rho_elem_);
      WriteBinaryVector(os, pi_elem_);
      WriteBinaryVector(os, mdot_elem_);

      WriteBinaryPod(os, static_cast<std::int64_t>(extent_elem_.size()));
      for (const auto &extent_per_elem : extent_elem_)
      {
         WriteBinaryVector(os, extent_per_elem);
      }

      WriteBinaryVector(os, degree_char_elem_);
      WriteBinaryVector(os, char_density_fraction_elem_);
   }

   void LoadFromStream(std::istream &is)
   {
      const std::int64_t ne = ReadBinaryPod<std::int64_t>(is);
      if (ne < 0)
      {
         throw runtime_error("Corrupt restart state: negative element count.");
      }

      states_.assign(static_cast<std::size_t>(ne), {});
      for (auto &elem_states : states_)
      {
         const std::int64_t nq = ReadBinaryPod<std::int64_t>(is);
         if (nq < 0)
         {
            throw runtime_error("Corrupt restart state: negative quadrature count.");
         }
         elem_states.resize(static_cast<std::size_t>(nq));
         for (auto &st : elem_states)
         {
            st.extent = ReadBinaryVector<double>(is);
            st.extent_old = ReadBinaryVector<double>(is);
            st.dt = ReadBinaryPod<double>(is);
         }
      }

      tau_elem_ = ReadBinaryVector<double>(is);
      rho_elem_ = ReadBinaryVector<double>(is);
      pi_elem_ = ReadBinaryVector<double>(is);
      mdot_elem_ = ReadBinaryVector<double>(is);

      const std::int64_t nr = ReadBinaryPod<std::int64_t>(is);
      if (nr < 0)
      {
         throw runtime_error("Corrupt restart state: negative reaction count.");
      }
      extent_elem_.assign(static_cast<std::size_t>(nr), {});
      for (auto &extent_per_elem : extent_elem_)
      {
         extent_per_elem = ReadBinaryVector<double>(is);
      }

      degree_char_elem_ = ReadBinaryVector<double>(is);
      char_density_fraction_elem_ = ReadBinaryVector<double>(is);

      const std::size_t expected_ne = states_.size();
      if (tau_elem_.size() != expected_ne ||
          rho_elem_.size() != expected_ne ||
          pi_elem_.size() != expected_ne ||
          mdot_elem_.size() != expected_ne ||
          degree_char_elem_.size() != expected_ne ||
          char_density_fraction_elem_.size() != expected_ne)
      {
         throw runtime_error("Corrupt restart state: diagnostic vector size mismatch.");
      }
      for (const auto &extent_per_elem : extent_elem_)
      {
         if (extent_per_elem.size() != expected_ne)
         {
            throw runtime_error("Corrupt restart state: extent vector size mismatch.");
         }
      }
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
                        const Vector &gravity,
                        const JacobianCheckOptions &jac_check)
      : material_(material),
        state_manager_(state_manager),
        T_old_coeff_(&T_old),
        p_old_coeff_(&p_old),
        quad_order_(quad_order),
        gravity_(gravity),
        jac_check_(jac_check)
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
      ComputeElementGradAnalytic(*el[0], *el[1], Tr, *elfun[0], *elfun[1],
                                 *elmats(0, 0), *elmats(0, 1),
                                 *elmats(1, 0), *elmats(1, 1));
      if (jac_check_.enable && !jacobian_checked_)
      {
         DenseMatrix J00_fd, J01_fd, J10_fd, J11_fd;
         ComputeElementGradFiniteDifference(*el[0], *el[1], Tr, *elfun[0], *elfun[1],
                                            J00_fd, J01_fd, J10_fd, J11_fd);
         VerifyJacobianBlockClose(*elmats(0, 0), J00_fd, jac_check_,
                                  "AblationTPIntegrator", "(0,0)", Tr.ElementNo);
         VerifyJacobianBlockClose(*elmats(0, 1), J01_fd, jac_check_,
                                  "AblationTPIntegrator", "(0,1)", Tr.ElementNo);
         VerifyJacobianBlockClose(*elmats(1, 0), J10_fd, jac_check_,
                                  "AblationTPIntegrator", "(1,0)", Tr.ElementNo);
         VerifyJacobianBlockClose(*elmats(1, 1), J11_fd, jac_check_,
                                  "AblationTPIntegrator", "(1,1)", Tr.ElementNo);
         jacobian_checked_ = true;
      }
   }

private:
   struct QPCoeffs
   {
      double A = 0.0; // storage_p - source_p
      double B = 0.0; // rho_darcy
      double C = 0.0; // rho2_darcy
      double D = 0.0; // solid_storage + gas_storage - pyrolysis_heat_sink
      double E = 0.0; // solid.k
      double F = 0.0; // h_rho_darcy
      double G = 0.0; // h_rho2_darcy
   };

   QPCoeffs EvaluateQPCoeffs(const double T,
                             const double p,
                             const TACOTMaterial::InternalState &old_state,
                             const double T_old,
                             const double p_old,
                             const TACOTMaterial::SolidProperties &solid_old,
                             const TACOTMaterial::GasProperties &gas_old) const
   {
      QPCoeffs out;

      TACOTMaterial::InternalState new_state =
         material_.SolveReactionExtents(T, dt_, old_state);
      const TACOTMaterial::SolidProperties solid =
         material_.EvaluateSolid(T, p, new_state);
      const TACOTMaterial::GasProperties gas =
         material_.EvaluateGas(T, p, new_state);

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

      out.A = storage_p - source_p;
      out.B = rho_darcy;
      out.C = rho2_darcy;
      out.D = solid_storage + gas_storage - solid.pyrolysis_heat_sink;
      out.E = solid.k;
      out.F = h_rho_darcy;
      out.G = h_rho2_darcy;
      return out;
   }

   void ComputeElementGradAnalytic(const FiniteElement &fe_T,
                                   const FiniteElement &fe_p,
                                   ElementTransformation &Tr,
                                   const Vector &elT,
                                   const Vector &elp,
                                   DenseMatrix &J00,
                                   DenseMatrix &J01,
                                   DenseMatrix &J10,
                                   DenseMatrix &J11) const
   {
      const int dof_T = fe_T.GetDof();
      const int dof_p = fe_p.GetDof();
      const int dim = fe_T.GetDim();

      J00.SetSize(dof_T, dof_T);
      J01.SetSize(dof_T, dof_p);
      J10.SetSize(dof_p, dof_T);
      J11.SetSize(dof_p, dof_p);
      J00 = 0.0;
      J01 = 0.0;
      J10 = 0.0;
      J11 = 0.0;

      shape_T_.SetSize(dof_T);
      shape_p_.SetSize(dof_p);
      dshape_T_.SetSize(dof_T, dim);
      dshape_p_.SetSize(dof_p, dim);
      gradT_.SetSize(dim);
      gradp_.SetSize(dim);

      const IntegrationRule &ir = IntRules.Get(fe_T.GetGeomType(), quad_order_);
      MFEM_VERIFY(Tr.ElementNo >= 0, "Invalid element number while assembling gradient.");
      MFEM_VERIFY(ir.GetNPoints() <= state_manager_.NumQPoints(Tr.ElementNo),
                  "Reaction state manager quadrature size mismatch.");

      const double coeff_fd_eps = 1.0e-7;

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
         TACOTMaterial::InternalState old_eval_state = old_state;
         old_eval_state.extent_old = old_eval_state.extent;
         old_eval_state.dt = dt_;
         const TACOTMaterial::SolidProperties solid_old =
            material_.EvaluateSolid(T_old, p_old, old_eval_state);
         const TACOTMaterial::GasProperties gas_old =
            material_.EvaluateGas(T_old, p_old, old_eval_state);

         const QPCoeffs base =
            EvaluateQPCoeffs(T, p, old_state, T_old, p_old, solid_old, gas_old);

         const double hT = coeff_fd_eps * std::max(1.0, std::abs(T));
         const double hp = coeff_fd_eps * std::max(1.0, std::abs(p));
         const QPCoeffs T_pert =
            EvaluateQPCoeffs(T + hT, p, old_state, T_old, p_old, solid_old, gas_old);
         const QPCoeffs p_pert =
            EvaluateQPCoeffs(T, p + hp, old_state, T_old, p_old, solid_old, gas_old);

         const double A_T = (T_pert.A - base.A) / hT;
         const double B_T = (T_pert.B - base.B) / hT;
         const double C_T = (T_pert.C - base.C) / hT;
         const double D_T = (T_pert.D - base.D) / hT;
         const double E_T = (T_pert.E - base.E) / hT;
         const double F_T = (T_pert.F - base.F) / hT;
         const double G_T = (T_pert.G - base.G) / hT;

         const double A_p = (p_pert.A - base.A) / hp;
         const double B_p = (p_pert.B - base.B) / hp;
         const double C_p = (p_pert.C - base.C) / hp;
         const double D_p = (p_pert.D - base.D) / hp;
         const double E_p = (p_pert.E - base.E) / hp;
         const double F_p = (p_pert.F - base.F) / hp;
         const double G_p = (p_pert.G - base.G) / hp;

         const double w = ip.weight * Tr.Weight();

         for (int i = 0; i < dof_p; ++i)
         {
            double Bpi_gradp = 0.0;
            double g_Bpi = 0.0;
            for (int d = 0; d < dim; ++d)
            {
               Bpi_gradp += dshape_p_(i, d) * gradp_[d];
               g_Bpi += gravity_[d] * dshape_p_(i, d);
            }

            for (int j = 0; j < dof_T; ++j)
            {
               J10(i, j) += w * (shape_p_[i] * A_T * shape_T_[j]
                                 + B_T * shape_T_[j] * Bpi_gradp
                                 - C_T * shape_T_[j] * g_Bpi);
            }

            for (int j = 0; j < dof_p; ++j)
            {
               double Bpi_Bpj = 0.0;
               for (int d = 0; d < dim; ++d)
               {
                  Bpi_Bpj += dshape_p_(i, d) * dshape_p_(j, d);
               }
               J11(i, j) += w * (shape_p_[i] * A_p * shape_p_[j]
                                 + B_p * shape_p_[j] * Bpi_gradp
                                 + base.B * Bpi_Bpj
                                 - C_p * shape_p_[j] * g_Bpi);
            }
         }

         for (int i = 0; i < dof_T; ++i)
         {
            double BTi_gradT = 0.0;
            double BTi_gradp = 0.0;
            double g_BTi = 0.0;
            for (int d = 0; d < dim; ++d)
            {
               BTi_gradT += dshape_T_(i, d) * gradT_[d];
               BTi_gradp += dshape_T_(i, d) * gradp_[d];
               g_BTi += gravity_[d] * dshape_T_(i, d);
            }

            for (int j = 0; j < dof_T; ++j)
            {
               double BTi_BTj = 0.0;
               for (int d = 0; d < dim; ++d)
               {
                  BTi_BTj += dshape_T_(i, d) * dshape_T_(j, d);
               }
               J00(i, j) += w * (shape_T_[i] * D_T * shape_T_[j]
                                 + E_T * shape_T_[j] * BTi_gradT
                                 + base.E * BTi_BTj
                                 + F_T * shape_T_[j] * BTi_gradp
                                 - G_T * shape_T_[j] * g_BTi);
            }

            for (int j = 0; j < dof_p; ++j)
            {
               double BTi_Bpj = 0.0;
               for (int d = 0; d < dim; ++d)
               {
                  BTi_Bpj += dshape_T_(i, d) * dshape_p_(j, d);
               }
               J01(i, j) += w * (shape_T_[i] * D_p * shape_p_[j]
                                 + E_p * shape_p_[j] * BTi_gradT
                                 + F_p * shape_p_[j] * BTi_gradp
                                 + base.F * BTi_Bpj
                                 - G_p * shape_p_[j] * g_BTi);
            }
         }
      }
   }

   void ComputeElementGradFiniteDifference(const FiniteElement &fe_T,
                                           const FiniteElement &fe_p,
                                           ElementTransformation &Tr,
                                           const Vector &elT,
                                           const Vector &elp,
                                           DenseMatrix &J00,
                                           DenseMatrix &J01,
                                           DenseMatrix &J10,
                                           DenseMatrix &J11) const
   {
      const int dof_T = fe_T.GetDof();
      const int dof_p = fe_p.GetDof();
      J00.SetSize(dof_T, dof_T);
      J01.SetSize(dof_T, dof_p);
      J10.SetSize(dof_p, dof_T);
      J11.SetSize(dof_p, dof_p);
      J00 = 0.0;
      J01 = 0.0;
      J10 = 0.0;
      J11 = 0.0;

      Vector rT0, rp0;
      ComputeElementResidual(fe_T, fe_p, Tr, elT, elp, rT0, rp0);

      const double fd_eps = 1.0e-7;

      Vector eT(elT);
      Vector ep(elp);
      Vector rT_pert, rp_pert;

      for (int j = 0; j < dof_T; ++j)
      {
         const double h = fd_eps * std::max(1.0, std::abs(eT[j]));
         eT = elT;
         eT[j] += h;
         ComputeElementResidual(fe_T, fe_p, Tr, eT, elp, rT_pert, rp_pert);
         for (int i = 0; i < dof_T; ++i)
         {
            J00(i, j) = (rT_pert[i] - rT0[i]) / h;
         }
         for (int i = 0; i < dof_p; ++i)
         {
            J10(i, j) = (rp_pert[i] - rp0[i]) / h;
         }
      }

      for (int j = 0; j < dof_p; ++j)
      {
         const double h = fd_eps * std::max(1.0, std::abs(ep[j]));
         ep = elp;
         ep[j] += h;
         ComputeElementResidual(fe_T, fe_p, Tr, elT, ep, rT_pert, rp_pert);
         for (int i = 0; i < dof_T; ++i)
         {
            J01(i, j) = (rT_pert[i] - rT0[i]) / h;
         }
         for (int i = 0; i < dof_p; ++i)
         {
            J11(i, j) = (rp_pert[i] - rp0[i]) / h;
         }
      }
   }

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
   JacobianCheckOptions jac_check_;
   bool jacobian_checked_ = false;

   mutable Vector shape_T_;
   mutable Vector shape_p_;
   mutable DenseMatrix dshape_T_;
   mutable DenseMatrix dshape_p_;
   mutable Vector gradT_;
   mutable Vector gradp_;
};

class SurfaceEnergyBalanceIntegrator : public BlockNonlinearFormIntegrator
{
public:
   SurfaceEnergyBalanceIntegrator(const TACOTMaterial &material,
                                  const ReactionStateManager &state_manager,
                                  const BPrimeTable &bprime_table,
                                  const SurfaceBCSchedule &schedule,
                                  const SurfaceFluxModelParams &surface_model,
                                  const Vector &gravity,
                                  const int quad_order,
                                  const JacobianCheckOptions &jac_check,
                                  const ParGridFunction *cooling_temperature_lag)
      : material_(material),
        state_manager_(state_manager),
        bprime_table_(bprime_table),
        schedule_(schedule),
        surface_model_(surface_model),
        gravity_(gravity),
        quad_order_(quad_order),
        jac_check_(jac_check),
        cooling_temperature_lag_(cooling_temperature_lag)
   {}

   void SetTime(const double t) { time_ = t; }

   void AssembleFaceVector(const Array<const FiniteElement *> &el1,
                           const Array<const FiniteElement *> &el2,
                           FaceElementTransformations &Tr,
                           const Array<const Vector *> &elfun,
                           const Array<Vector *> &elvec) override
   {
      (void)el2;
      ComputeFaceResidual(*el1[0], *el1[1], Tr, *elfun[0], *elfun[1], *elvec[0], *elvec[1]);
   }

   void AssembleFaceGrad(const Array<const FiniteElement *> &el1,
                         const Array<const FiniteElement *> &el2,
                         FaceElementTransformations &Tr,
                         const Array<const Vector *> &elfun,
                         const Array2D<DenseMatrix *> &elmats) override
   {
      (void)el2;

      bool face_has_nonsmooth = false;
      ComputeFaceGradAnalytic(*el1[0], *el1[1], Tr, *elfun[0], *elfun[1],
                              *elmats(0, 0), *elmats(0, 1),
                              *elmats(1, 0), *elmats(1, 1),
                              &face_has_nonsmooth);

      const SurfaceBCSchedule::BoundaryState bc_state = schedule_.Eval(time_);
      const SurfaceFluxBranch branch = ClassifySurfaceFluxBranch(bc_state, surface_model_);
      const int branch_idx = SurfaceFluxBranchIndex(branch);
      if (jac_check_.enable && face_has_nonsmooth &&
          !jacobian_checked_branch_[branch_idx])
      {
         jacobian_checked_branch_[branch_idx] = true;
      }
      if (jac_check_.enable && !face_has_nonsmooth &&
          !jacobian_checked_branch_[branch_idx])
      {
         DenseMatrix J00_fd, J01_fd, J10_fd, J11_fd;
         ComputeFaceGradFiniteDifference(*el1[0], *el1[1], Tr, *elfun[0], *elfun[1],
                                         J00_fd, J01_fd, J10_fd, J11_fd);
         const string integrator_name =
            string("SurfaceEnergyBalanceIntegrator[") +
            SurfaceFluxBranchName(branch) + "]";
         VerifyJacobianBlockClose(*elmats(0, 0), J00_fd, jac_check_,
                                  integrator_name, "(0,0)", Tr.Elem1No);
         VerifyJacobianBlockClose(*elmats(0, 1), J01_fd, jac_check_,
                                  integrator_name, "(0,1)", Tr.Elem1No);
         VerifyJacobianBlockClose(*elmats(1, 0), J10_fd, jac_check_,
                                  integrator_name, "(1,0)", Tr.Elem1No);
         VerifyJacobianBlockClose(*elmats(1, 1), J11_fd, jac_check_,
                                  integrator_name, "(1,1)", Tr.Elem1No);
         jacobian_checked_branch_[branch_idx] = true;
      }
   }

private:
   void ComputeFaceGradAnalytic(const FiniteElement &fe_T,
                                const FiniteElement &fe_p,
                                FaceElementTransformations &Tr,
                                const Vector &elT,
                                const Vector &elp,
                                DenseMatrix &J00,
                                DenseMatrix &J01,
                                DenseMatrix &J10,
                                DenseMatrix &J11,
                                bool *has_nonsmooth) const
   {
      const int dof_T = fe_T.GetDof();
      const int dof_p = fe_p.GetDof();
      const int dim = fe_T.GetDim();

      J00.SetSize(dof_T, dof_T);
      J01.SetSize(dof_T, dof_p);
      J10.SetSize(dof_p, dof_T);
      J11.SetSize(dof_p, dof_p);
      J00 = 0.0;
      J01 = 0.0;
      J10 = 0.0;
      J11 = 0.0;
      if (has_nonsmooth) { *has_nonsmooth = false; }

      if (Tr.Elem1No < 0 || Tr.Elem1No >= state_manager_.NumElements())
      {
         return;
      }

      const SurfaceBCSchedule::BoundaryState bc_state = schedule_.Eval(time_);

      shape_T_.SetSize(dof_T);
      shape_p_.SetSize(dof_p);
      dshape_p_.SetSize(dof_p, dim);
      gradp_.SetSize(dim);
      normal_.SetSize(dim);

      const int face_int_order = max(quad_order_,
                                     2 * max(fe_T.GetOrder(), fe_p.GetOrder()) + 2);
      const IntegrationRule &ir_face =
         IntRules.Get(Tr.GetGeometryType(), face_int_order);

      for (int q = 0; q < ir_face.GetNPoints(); ++q)
      {
         const IntegrationPoint &fip = ir_face.IntPoint(q);
         IntegrationPoint eip;
         Tr.Loc1.Transform(fip, eip);

         Tr.Elem1->SetIntPoint(&eip);
         fe_T.CalcPhysShape(*Tr.Elem1, shape_T_);
         fe_p.CalcPhysShape(*Tr.Elem1, shape_p_);
         fe_p.CalcPhysDShape(*Tr.Elem1, dshape_p_);

         const double T_w = shape_T_ * elT;
         const double p_w = shape_p_ * elp;

         gradp_ = 0.0;
         for (int j = 0; j < dof_p; ++j)
         {
            for (int d = 0; d < dim; ++d)
            {
               gradp_[d] += elp[j] * dshape_p_(j, d);
            }
         }

         const TACOTMaterial::InternalState &state = state_manager_.GetState(Tr.Elem1No, 0);
         const TACOTMaterial::SolidSurfaceDerivatives solid_deriv =
            material_.EvaluateSolidSurfaceDerivatives(T_w, p_w, state);
         const TACOTMaterial::GasSurfaceDerivatives gas_deriv =
            material_.EvaluateGasSurfaceDerivatives(T_w, p_w, state);

         const double mu_eff = max(gas_deriv.mu.value, 1.0e-12);
         const double dmu_eff_dT = (gas_deriv.mu.value > 1.0e-12) ? gas_deriv.mu.dT : 0.0;
         const double dmu_eff_dp = (gas_deriv.mu.value > 1.0e-12) ? gas_deriv.mu.dp : 0.0;
         if (has_nonsmooth && gas_deriv.mu.value <= 1.0e-12)
         {
            *has_nonsmooth = true;
         }

         const double rho_darcy = gas_deriv.rho.value * solid_deriv.K.value / mu_eff;
         const double drho_darcy_dT =
            gas_deriv.rho.dT * solid_deriv.K.value / mu_eff +
            gas_deriv.rho.value * solid_deriv.K.dT / mu_eff -
            gas_deriv.rho.value * solid_deriv.K.value /
               (mu_eff * mu_eff) * dmu_eff_dT;
         const double drho_darcy_dp =
            gas_deriv.rho.dp * solid_deriv.K.value / mu_eff +
            gas_deriv.rho.value * solid_deriv.K.dp / mu_eff -
            gas_deriv.rho.value * solid_deriv.K.value /
               (mu_eff * mu_eff) * dmu_eff_dp;

         const double rho2_darcy = gas_deriv.rho.value * rho_darcy;
         const double drho2_darcy_dT =
            gas_deriv.rho.dT * rho_darcy +
            gas_deriv.rho.value * drho_darcy_dT;
         const double drho2_darcy_dp =
            gas_deriv.rho.dp * rho_darcy +
            gas_deriv.rho.value * drho_darcy_dp;

         Tr.Face->SetIntPoint(&fip);
         if (dim == 1)
         {
            normal_[0] = 1.0;
         }
         else
         {
            CalcOrtho(Tr.Face->Jacobian(), normal_);
         }

         const double nmag = normal_.Norml2();
         if (nmag <= 1.0e-20)
         {
            continue;
         }
         const double ds = fip.weight * nmag;

         double gradp_n = 0.0;
         double g_n = 0.0;
         for (int d = 0; d < dim; ++d)
         {
            const double nd = normal_[d] / nmag;
            gradp_n += gradp_[d] * nd;
            g_n += gravity_[d] * nd;
         }

         const double m_dot_g_w = -rho_darcy * gradp_n + rho2_darcy * g_n;
         const double dm_dot_dT = -drho_darcy_dT * gradp_n + drho2_darcy_dT * g_n;
         const double dm_dot_dp = -drho_darcy_dp * gradp_n + drho2_darcy_dp * g_n;

         const double T_eval = T_w;
         const double dT_eval_dT_w = 1.0;

         const double emissivity =
            surface_model_.use_emissivity_override ?
               surface_model_.emissivity :
               solid_deriv.emissivity.value;
         const double absorptivity =
            surface_model_.use_absorptivity_override ?
               surface_model_.absorptivity :
               solid_deriv.absorptivity.value;
         const double reflectivity = solid_deriv.reflectivity.value;
         const double demissivity_dT =
            surface_model_.use_emissivity_override ? 0.0 : solid_deriv.emissivity.dT;
         const double demissivity_dp =
            surface_model_.use_emissivity_override ? 0.0 : solid_deriv.emissivity.dp;
         const double dabsorptivity_dT =
            surface_model_.use_absorptivity_override ? 0.0 : solid_deriv.absorptivity.dT;
         const double dabsorptivity_dp =
            surface_model_.use_absorptivity_override ? 0.0 : solid_deriv.absorptivity.dp;

         const SurfaceFluxLinearization flux = EvaluateSurfaceFluxTermsLinearized(
            m_dot_g_w,
            gas_deriv.h.value,
            T_w,
            T_eval,
            emissivity,
            absorptivity,
            reflectivity,
            bc_state,
            bprime_table_,
            surface_model_);
         if (has_nonsmooth && flux.nonsmooth)
         {
            *has_nonsmooth = true;
         }

         for (int i = 0; i < dof_T; ++i)
         {
            for (int j = 0; j < dof_T; ++j)
            {
               const double dTj = shape_T_[j];
               const double dm_dot_j = dm_dot_dT * dTj;
               const double dh_g_j = gas_deriv.h.dT * dTj;
               const double demissivity_j = demissivity_dT * dTj;
               const double dabsorptivity_j = dabsorptivity_dT * dTj;
               const double dT_eval_j = dT_eval_dT_w * dTj;

               const double dq_j =
                  flux.dq_dm_dot * dm_dot_j +
                  flux.dq_dh_g * dh_g_j +
                  flux.dq_dT_w * dTj +
                  flux.dq_dT_eval * dT_eval_j +
                  flux.dq_demissivity * demissivity_j +
                  flux.dq_dabsorptivity * dabsorptivity_j;
               J00(i, j) += -ds * shape_T_[i] * dq_j;
            }

            for (int j = 0; j < dof_p; ++j)
            {
               double dgradp_n_j = 0.0;
               for (int d = 0; d < dim; ++d)
               {
                  dgradp_n_j += dshape_p_(j, d) * normal_[d] / nmag;
               }
               const double dm_dot_j = dm_dot_dp * shape_p_[j] - rho_darcy * dgradp_n_j;
               const double dh_g_j = gas_deriv.h.dp * shape_p_[j];
               const double demissivity_j = demissivity_dp * shape_p_[j];
               const double dabsorptivity_j = dabsorptivity_dp * shape_p_[j];

               const double dq_j =
                  flux.dq_dm_dot * dm_dot_j +
                  flux.dq_dh_g * dh_g_j +
                  flux.dq_demissivity * demissivity_j +
                  flux.dq_dabsorptivity * dabsorptivity_j;
               J01(i, j) += -ds * shape_T_[i] * dq_j;
            }
         }
      }
   }

   void ComputeFaceGradFiniteDifference(const FiniteElement &fe_T,
                                        const FiniteElement &fe_p,
                                        FaceElementTransformations &Tr,
                                        const Vector &elT,
                                        const Vector &elp,
                                        DenseMatrix &J00,
                                        DenseMatrix &J01,
                                        DenseMatrix &J10,
                                        DenseMatrix &J11) const
   {
      const int dof_T = fe_T.GetDof();
      const int dof_p = fe_p.GetDof();
      J00.SetSize(dof_T, dof_T);
      J01.SetSize(dof_T, dof_p);
      J10.SetSize(dof_p, dof_T);
      J11.SetSize(dof_p, dof_p);
      J00 = 0.0;
      J01 = 0.0;
      J10 = 0.0;
      J11 = 0.0;

      Vector rT0, rp0;
      ComputeFaceResidual(fe_T, fe_p, Tr, elT, elp, rT0, rp0);

      const double fd_eps = 1.0e-7;
      Vector eT(elT);
      Vector ep(elp);
      Vector rT_pert, rp_pert;

      for (int j = 0; j < dof_T; ++j)
      {
         const double h = fd_eps * std::max(1.0, std::abs(eT[j]));
         eT = elT;
         eT[j] += h;
         ComputeFaceResidual(fe_T, fe_p, Tr, eT, elp, rT_pert, rp_pert);
         for (int i = 0; i < dof_T; ++i)
         {
            J00(i, j) = (rT_pert[i] - rT0[i]) / h;
         }
         for (int i = 0; i < dof_p; ++i)
         {
            J10(i, j) = (rp_pert[i] - rp0[i]) / h;
         }
      }

      for (int j = 0; j < dof_p; ++j)
      {
         const double h = fd_eps * std::max(1.0, std::abs(ep[j]));
         ep = elp;
         ep[j] += h;
         ComputeFaceResidual(fe_T, fe_p, Tr, elT, ep, rT_pert, rp_pert);
         for (int i = 0; i < dof_T; ++i)
         {
            J01(i, j) = (rT_pert[i] - rT0[i]) / h;
         }
         for (int i = 0; i < dof_p; ++i)
         {
            J11(i, j) = (rp_pert[i] - rp0[i]) / h;
         }
      }
   }

   void ComputeFaceResidual(const FiniteElement &fe_T,
                            const FiniteElement &fe_p,
                            FaceElementTransformations &Tr,
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

      if (Tr.Elem1No < 0 || Tr.Elem1No >= state_manager_.NumElements())
      {
         return;
      }

      const SurfaceBCSchedule::BoundaryState bc_state = schedule_.Eval(time_);

      shape_T_.SetSize(dof_T);
      shape_p_.SetSize(dof_p);
      dshape_p_.SetSize(dof_p, dim);
      gradp_.SetSize(dim);
      normal_.SetSize(dim);

      const int face_int_order = max(quad_order_,
                                     2 * max(fe_T.GetOrder(), fe_p.GetOrder()) + 2);
      const IntegrationRule &ir_face =
         IntRules.Get(Tr.GetGeometryType(), face_int_order);

      for (int q = 0; q < ir_face.GetNPoints(); ++q)
      {
         const IntegrationPoint &fip = ir_face.IntPoint(q);
         IntegrationPoint eip;
         Tr.Loc1.Transform(fip, eip);

         Tr.Elem1->SetIntPoint(&eip);
         fe_T.CalcPhysShape(*Tr.Elem1, shape_T_);
         fe_p.CalcPhysShape(*Tr.Elem1, shape_p_);
         fe_p.CalcPhysDShape(*Tr.Elem1, dshape_p_);

         const double T_w = shape_T_ * elT;
         const double p_w = shape_p_ * elp;

         gradp_ = 0.0;
         for (int j = 0; j < dof_p; ++j)
         {
            for (int d = 0; d < dim; ++d)
            {
               gradp_[d] += elp[j] * dshape_p_(j, d);
            }
         }

         const TACOTMaterial::InternalState &state = state_manager_.GetState(Tr.Elem1No, 0);
         const TACOTMaterial::SolidProperties solid =
            material_.EvaluateSolid(T_w, p_w, state);
         const TACOTMaterial::GasProperties gas =
            material_.EvaluateGas(T_w, p_w, state);

         const double mu = max(gas.mu, 1.0e-12);
         const double rho_darcy = gas.rho * solid.K / mu;
         const double rho2_darcy = gas.rho * rho_darcy;

         mflux_.SetSize(dim);
         for (int d = 0; d < dim; ++d)
         {
            mflux_[d] = -rho_darcy * gradp_[d] + rho2_darcy * gravity_[d];
         }

         Tr.Face->SetIntPoint(&fip);
         if (dim == 1)
         {
            normal_[0] = 1.0;
         }
         else
         {
            CalcOrtho(Tr.Face->Jacobian(), normal_);
         }

         const double nmag = normal_.Norml2();
         if (nmag <= 1.0e-20)
         {
            continue;
         }
         const double ds = fip.weight * nmag;

         const double m_dot_g_w = (mflux_ * normal_) / nmag;
         const double T_eval = T_w;
         const SurfaceFluxTerms terms = EvaluateSurfaceFluxTerms(m_dot_g_w,
                                                                 gas.h,
                                                                 T_w,
                                                                 T_eval,
                                                                 solid,
                                                                 bc_state,
                                                                 bprime_table_,
                                                                 surface_model_);

         // Residual form: storage*v + flux.grad(v) - q_in*v on the boundary.
         for (int i = 0; i < dof_T; ++i)
         {
            rT[i] -= ds * shape_T_[i] * terms.q_surf;
         }
      }
   }

   const TACOTMaterial &material_;
   const ReactionStateManager &state_manager_;
   const BPrimeTable &bprime_table_;
   const SurfaceBCSchedule &schedule_;
   SurfaceFluxModelParams surface_model_;
   Vector gravity_;
   int quad_order_ = 2;
   double time_ = 0.0;
   JacobianCheckOptions jac_check_;
   mutable std::array<bool, 4> jacobian_checked_branch_ =
      {false, false, false, false};
   const ParGridFunction *cooling_temperature_lag_ = nullptr;

   mutable Vector shape_T_;
   mutable Vector shape_p_;
   mutable DenseMatrix dshape_p_;
   mutable Vector gradp_;
   mutable Vector normal_;
   mutable Vector mflux_;
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

SurfaceBoundaryDiagnostics ComputeTopBoundaryDiagnostics(
   ParMesh &pmesh,
   const ParFiniteElementSpace &fes_T,
   const ParFiniteElementSpace &fes_p,
   const ParGridFunction &T,
   const ParGridFunction &p,
   const TACOTMaterial &material,
   const ReactionStateManager &state_manager,
   const BPrimeTable &bprime_table,
   const SurfaceBCSchedule &schedule,
   const SurfaceFluxModelParams &surface_model,
   const Vector &gravity,
   const int top_bdr_attr,
   const double time,
   const bool compute_surface_terms)
{
   const SurfaceBCSchedule::BoundaryState bc_state = schedule.Eval(time);

   Array<int> dofs_T, dofs_p;
   Vector elT, elp, shape_T, shape_p, gradp, normal;
   DenseMatrix dshape_p;

   const int dim = pmesh.Dimension();
   SurfaceBoundaryDiagnostics local_sum{};
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

         const double nmag = normal.Norml2();
         if (nmag <= 1.0e-20)
         {
            continue;
         }

         const double ds = fip.weight * nmag;
         const double m_dot_g_w = (mflux * normal) / nmag;

         local_sum.m_dot_g_surf += ds * m_dot_g_w;
         if (compute_surface_terms)
         {
            const SurfaceFluxTerms terms = EvaluateSurfaceFluxTerms(m_dot_g_w,
                                                                    gas.h,
                                                                    Tq,
                                                                    Tq,
                                                                    solid,
                                                                    bc_state,
                                                                    bprime_table,
                                                                    surface_model);
            local_sum.BprimeG_surf += ds * terms.BprimeG;
            local_sum.BprimeC_surf += ds * terms.BprimeC;
            local_sum.h_w_surf += ds * terms.h_w;
            local_sum.emissivity_surf += ds * terms.emissivity;
            local_sum.absorptivity_surf += ds * terms.absorptivity;
            local_sum.reflectivity_surf += ds * terms.reflectivity;
            local_sum.blowing_correction_surf += ds * terms.blowing_correction;
            local_sum.q_conv_surf += ds * terms.q_conv;
            local_sum.q_adv_pyro_surf += ds * terms.q_adv_pyro;
            local_sum.q_rad_emit_surf += ds * terms.q_rad_emit;
            local_sum.q_rad_abs_surf += ds * terms.q_rad_abs;
            local_sum.q_surf += ds * terms.q_surf;
         }
         local_area += ds;
      }
   }

   double local_data[14] = {
      local_sum.m_dot_g_surf,
      local_sum.BprimeG_surf,
      local_sum.BprimeC_surf,
      local_sum.h_w_surf,
      local_sum.emissivity_surf,
      local_sum.absorptivity_surf,
      local_sum.reflectivity_surf,
      local_sum.blowing_correction_surf,
      local_sum.q_conv_surf,
      local_sum.q_adv_pyro_surf,
      local_sum.q_rad_emit_surf,
      local_sum.q_rad_abs_surf,
      local_sum.q_surf,
      local_area
   };
   double global_data[14] = {0.0};
   MPI_Allreduce(local_data, global_data, 14, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());

   SurfaceBoundaryDiagnostics out;
   const double area = global_data[13];
   if (area <= 0.0)
   {
      out.m_dot_g_surf = numeric_limits<double>::quiet_NaN();
      out.BprimeG_surf = numeric_limits<double>::quiet_NaN();
      out.BprimeC_surf = numeric_limits<double>::quiet_NaN();
      out.h_w_surf = numeric_limits<double>::quiet_NaN();
      out.emissivity_surf = numeric_limits<double>::quiet_NaN();
      out.absorptivity_surf = numeric_limits<double>::quiet_NaN();
      out.reflectivity_surf = numeric_limits<double>::quiet_NaN();
      out.blowing_correction_surf = numeric_limits<double>::quiet_NaN();
      out.q_conv_surf = numeric_limits<double>::quiet_NaN();
      out.q_adv_pyro_surf = numeric_limits<double>::quiet_NaN();
      out.q_rad_emit_surf = numeric_limits<double>::quiet_NaN();
      out.q_rad_abs_surf = numeric_limits<double>::quiet_NaN();
      out.q_surf = numeric_limits<double>::quiet_NaN();
      return out;
   }

   out.m_dot_g_surf = global_data[0] / area;
   if (!compute_surface_terms)
   {
      out.BprimeG_surf = numeric_limits<double>::quiet_NaN();
      out.BprimeC_surf = numeric_limits<double>::quiet_NaN();
      out.h_w_surf = numeric_limits<double>::quiet_NaN();
      out.emissivity_surf = numeric_limits<double>::quiet_NaN();
      out.absorptivity_surf = numeric_limits<double>::quiet_NaN();
      out.reflectivity_surf = numeric_limits<double>::quiet_NaN();
      out.blowing_correction_surf = numeric_limits<double>::quiet_NaN();
      out.q_conv_surf = numeric_limits<double>::quiet_NaN();
      out.q_adv_pyro_surf = numeric_limits<double>::quiet_NaN();
      out.q_rad_emit_surf = numeric_limits<double>::quiet_NaN();
      out.q_rad_abs_surf = numeric_limits<double>::quiet_NaN();
      out.q_surf = numeric_limits<double>::quiet_NaN();
      return out;
   }

   out.BprimeG_surf = global_data[1] / area;
   out.BprimeC_surf = global_data[2] / area;
   out.h_w_surf = global_data[3] / area;
   out.emissivity_surf = global_data[4] / area;
   out.absorptivity_surf = global_data[5] / area;
   out.reflectivity_surf = global_data[6] / area;
   out.blowing_correction_surf = global_data[7] / area;
   out.q_conv_surf = global_data[8] / area;
   out.q_adv_pyro_surf = global_data[9] / area;
   out.q_rad_emit_surf = global_data[10] / area;
   out.q_rad_abs_surf = global_data[11] / area;
   out.q_surf = global_data[12] / area;
   return out;
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

struct RestartCheckpointInfo
{
   int step = 0;
   double time = 0.0;
};

void SaveRestartCheckpoint(const string &base_path,
                           const int step,
                           const double time,
                           const ParGridFunction &T,
                           const ParGridFunction &p,
                           const ReactionStateManager &state_manager,
                           const int world_size)
{
   if (base_path.empty())
   {
      return;
   }

   const int rank = Mpi::WorldRank();
   const string path = RestartPathForRank(base_path, rank, world_size);
   EnsureParentDirectoryExists(path);

   Vector Ttrue, ptrue;
   T.GetTrueDofs(Ttrue);
   p.GetTrueDofs(ptrue);

   ofstream os(path, ios::binary | ios::trunc);
   if (!os)
   {
      throw runtime_error("Failed to open restart checkpoint for writing: " + path);
   }

   WriteBinaryPod(os, kRestartMagic);
   WriteBinaryPod(os, kRestartVersion);
   WriteBinaryPod(os, static_cast<std::int32_t>(world_size));
   WriteBinaryPod(os, static_cast<std::int64_t>(step));
   WriteBinaryPod(os, time);
   WriteMFEMVector(os, Ttrue);
   WriteMFEMVector(os, ptrue);
   state_manager.SaveToStream(os);
   if (!os)
   {
      throw runtime_error("Failed while finalizing restart checkpoint write: " + path);
   }
}

RestartCheckpointInfo LoadRestartCheckpoint(const string &base_path,
                                            ParGridFunction &T,
                                            ParGridFunction &p,
                                            ReactionStateManager &state_manager,
                                            const int world_size)
{
   const int rank = Mpi::WorldRank();
   const string path = RestartPathForRank(base_path, rank, world_size);

   ifstream is(path, ios::binary);
   if (!is)
   {
      throw runtime_error("Failed to open restart checkpoint for reading: " + path);
   }

   const std::uint64_t magic = ReadBinaryPod<std::uint64_t>(is);
   const std::uint32_t version = ReadBinaryPod<std::uint32_t>(is);
   const std::int32_t file_world_size = ReadBinaryPod<std::int32_t>(is);
   if (magic != kRestartMagic)
   {
      throw runtime_error("Invalid restart checkpoint magic in: " + path);
   }
   if (version != kRestartVersion)
   {
      throw runtime_error("Unsupported restart checkpoint version in: " + path);
   }
   if (file_world_size != world_size)
   {
      throw runtime_error("Restart checkpoint MPI size mismatch for " + path +
                          " (file=" + to_string(file_world_size) +
                          ", run=" + to_string(world_size) + ").");
   }

   const std::int64_t step64 = ReadBinaryPod<std::int64_t>(is);
   const double time = ReadBinaryPod<double>(is);
   if (step64 < 0 || step64 > static_cast<std::int64_t>(std::numeric_limits<int>::max()))
   {
      throw runtime_error("Corrupt restart step index in: " + path);
   }
   if (!std::isfinite(time) || time < 0.0)
   {
      throw runtime_error("Corrupt restart time value in: " + path);
   }

   Vector Ttrue, ptrue;
   ReadMFEMVector(is, Ttrue);
   ReadMFEMVector(is, ptrue);

   const int expected_true_T = T.ParFESpace()->GetTrueVSize();
   const int expected_true_p = p.ParFESpace()->GetTrueVSize();
   if (Ttrue.Size() != expected_true_T || ptrue.Size() != expected_true_p)
   {
      throw runtime_error("Restart true-dof size mismatch in: " + path);
   }

   state_manager.LoadFromStream(is);
   if (!is)
   {
      throw runtime_error("Failed while reading restart checkpoint payload: " + path);
   }
   if (state_manager.NumElements() != T.ParFESpace()->GetNE())
   {
      throw runtime_error("Restart element count mismatch in state manager for: " + path);
   }

   T.SetFromTrueDofs(Ttrue);
   p.SetFromTrueDofs(ptrue);

   RestartCheckpointInfo info;
   info.step = static_cast<int>(step64);
   info.time = time;
   return info;
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
   cout << "  newton_jacobian_rebuild_freq: " << p.newton_jacobian_rebuild_freq
        << endl;
   cout << "  jacobian_check: " << (p.jacobian_check ? "true" : "false") << endl;
   cout << "  jacobian_check_abs_tol: " << p.jacobian_check_abs_tol << endl;
   cout << "  jacobian_check_rel_tol: " << p.jacobian_check_rel_tol << endl;
   cout << "  petsc_options_file: " << p.petsc_options_file << endl;
   cout << "  ksp_prefix: " << p.ksp_prefix << endl;
   cout << "  petsc_ksp_print_level: " << p.petsc_ksp_print_level << endl;
   cout << "  output_every: " << p.output_every << endl;
   cout << "  output_path: " << p.output_path << endl;
   cout << "  collection_name: " << p.collection_name << endl;
   cout << "  timing_step_csv: " << p.timing_step_csv << endl;
   cout << "  timing_summary_csv: " << p.timing_summary_csv << endl;
   cout << "  save_paraview: " << (p.save_paraview ? "true" : "false") << endl;
   cout << "  restart_read_file: "
        << (p.restart_read_file.empty() ? "none" : p.restart_read_file) << endl;
   cout << "  restart_write_file: "
        << (p.restart_write_file.empty() ? "none" : p.restart_write_file) << endl;
   cout << "  restart_write_every: " << p.restart_write_every << endl;
   cout << "  restart_write_at_time: "
        << (std::isfinite(p.restart_write_at_time) ?
               std::to_string(p.restart_write_at_time) :
               "none")
        << endl;
   cout << "  bdr_attr_top: " << p.bdr_attr_top << endl;
   cout << "  bprime_table_file: " << p.bprime_table_file << endl;
   cout << "  boundary_conditions_file: " << p.boundary_conditions_file << endl;
   cout << "  top_thermal_bc: " << p.top_thermal_bc << endl;
   cout << "  top_temperature_value: " << p.top_temperature_value << endl;
   cout << "  top_temperature_file: "
        << (p.top_temperature_file.empty() ? "none" : p.top_temperature_file) << endl;
   cout << "  emissivity_override: "
        << (std::isfinite(p.emissivity) ? std::to_string(p.emissivity) : "none")
        << endl;
   cout << "  absorptivity_override: "
        << (std::isfinite(p.absorptivity) ? std::to_string(p.absorptivity) : "none")
        << endl;
   cout << "  strict_case2_1: " << (p.strict_case2_1 ? "true" : "false") << endl;
   cout << "  pato_compat_mode: " << PatoCompatModeName(p.pato_compat_mode) << endl;
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const int myid = Mpi::WorldRank();
   int world_size = 1;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);

   string input_file = "Input/input_ablation_case2_1.yaml";
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
      const bool use_temperature_dirichlet =
         (params.top_thermal_bc == "temperature_dirichlet");
      TopTemperatureSchedule top_temperature_schedule;
      bool use_top_temperature_schedule = false;

      const auto run_t0 = steady_clock_t::now();
      const auto setup_t0 = run_t0;

      Device device("cpu");
      if (myid == 0) { device.Print(); }

      TACOTMaterial material;
      material.LoadFromYaml(params.material_file);

      BPrimeTable bprime_table;
      bprime_table.LoadFromFile(params.bprime_table_file);

      SurfaceBCSchedule bc_schedule;
      bc_schedule.LoadFromFile(params.boundary_conditions_file);
      if (use_temperature_dirichlet && !params.top_temperature_file.empty())
      {
         top_temperature_schedule.LoadFromFile(params.top_temperature_file);
         use_top_temperature_schedule = true;
      }

      auto top_temperature_at = [&](const double t)
      {
         return use_top_temperature_schedule ?
                   top_temperature_schedule.Eval(t) :
                   params.top_temperature_value;
      };

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
      if (use_temperature_dirichlet)
      {
         ess_bdr_T[params.bdr_attr_top - 1] = 1;
      }
      ess_bdr_p[params.bdr_attr_top - 1] = 1;

      Array<int> ess_tdof_T, ess_tdof_p;
      fes_T.GetEssentialTrueDofs(ess_bdr_T, ess_tdof_T);
      fes_p.GetEssentialTrueDofs(ess_bdr_p, ess_tdof_p);

      ParGridFunction T(&fes_T), p(&fes_p), T_old(&fes_T), p_old(&fes_p);
      ParGridFunction cooling_bc_lag_T(&fes_T);
      T = 300.0;
      p = bc_schedule.Eval(0.0).p_w;
      T_old = T;
      p_old = p;
      cooling_bc_lag_T = T;

      // Apply initial essential pressure values.
      {
         Vector Ttrue, ptrue;
         T.GetTrueDofs(Ttrue);
         p.GetTrueDofs(ptrue);
         if (use_temperature_dirichlet)
         {
            Ttrue.SetSubVector(ess_tdof_T, top_temperature_at(0.0));
         }
         ptrue.SetSubVector(ess_tdof_p, bc_schedule.Eval(0.0).p_w);
         T.SetFromTrueDofs(Ttrue);
         p.SetFromTrueDofs(ptrue);
         T_old = T;
         p_old = p;
         cooling_bc_lag_T = T;
      }

      const int quad_order = max(2, 2 * params.order + 2);

      ReactionStateManager state_manager;
      state_manager.Initialize(fes_T, quad_order, material);
      InitializeDiagnostics(material, state_manager);

      int restart_step = 0;
      double restart_time = 0.0;
      if (!params.restart_read_file.empty())
      {
         const RestartCheckpointInfo restart_info =
            LoadRestartCheckpoint(params.restart_read_file,
                                  T,
                                  p,
                                  state_manager,
                                  world_size);
         restart_step = restart_info.step;
         restart_time = restart_info.time;
         if (restart_step < 0)
         {
            throw runtime_error("Restart step must be >= 0.");
         }
         if (state_manager.NumReactions() != material.NumReactions())
         {
            throw runtime_error(
               "Restart reaction-state count does not match current material.");
         }
         T_old = T;
         p_old = p;
         cooling_bc_lag_T = T;
         if (myid == 0)
         {
            cout << "Loaded restart from " << params.restart_read_file
                 << " at step " << restart_step
                 << ", time " << restart_time << " s." << endl;
         }
      }

      Vector gravity(2);
      gravity[0] = params.gravity_x;
      gravity[1] = params.gravity_y;

      SurfaceFluxModelParams surface_model;
      surface_model.lambda = params.lambda;
      surface_model.q_rad = params.q_rad;
      surface_model.T_background = params.T_background;
      surface_model.T_edge = params.T_edge;
      surface_model.hconv = params.hconv;
      surface_model.use_emissivity_override = std::isfinite(params.emissivity);
      surface_model.use_absorptivity_override = std::isfinite(params.absorptivity);
      surface_model.emissivity =
         surface_model.use_emissivity_override ? params.emissivity : 1.0;
      surface_model.absorptivity =
         surface_model.use_absorptivity_override ? params.absorptivity : 1.0;
      surface_model.stefan_boltzmann = params.stefan_boltzmann;
      surface_model.strict_case2_1 = params.strict_case2_1;
      surface_model.pato_compat_mode = params.pato_compat_mode;

      JacobianCheckOptions jac_check_opts;
      jac_check_opts.enable = params.jacobian_check;
      jac_check_opts.abs_tol = params.jacobian_check_abs_tol;
      jac_check_opts.rel_tol = params.jacobian_check_rel_tol;

      auto *tp_integrator =
         new AblationTPIntegrator(material,
                                  state_manager,
                                  T_old,
                                  p_old,
                                  quad_order,
                                  gravity,
                                  jac_check_opts);
      SurfaceEnergyBalanceIntegrator *surf_integrator = nullptr;
      if (!use_temperature_dirichlet)
      {
         surf_integrator =
            new SurfaceEnergyBalanceIntegrator(material,
                                               state_manager,
                                               bprime_table,
                                               bc_schedule,
                                               surface_model,
                                               gravity,
                                               quad_order,
                                               jac_check_opts,
                                               &cooling_bc_lag_T);
         surf_integrator->SetTime(restart_time);
      }

      ParBlockNonlinearForm block_form(spaces);
      block_form.SetGradientType(Operator::Hypre_ParCSR);
      block_form.AddDomainIntegrator(tp_integrator);

      Array<int> top_bdr_marker(pmesh->bdr_attributes.Max());
      top_bdr_marker = 0;
      top_bdr_marker[params.bdr_attr_top - 1] = 1;
      if (surf_integrator)
      {
         block_form.AddBdrFaceIntegrator(surf_integrator, top_bdr_marker);
      }

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
      newton_cfg.jacobian_rebuild_freq = params.newton_jacobian_rebuild_freq;

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
      ofstream boundary_csv;
      ofstream newton_csv;
      ofstream timing_step_csv;
      if (myid == 0)
      {
         probes_csv.open(filesystem::path(params.output_path) / params.probes_csv);
         mass_csv.open(filesystem::path(params.output_path) / params.mass_csv);
         boundary_csv.open(filesystem::path(params.output_path) / params.boundary_csv);
         newton_csv.open(filesystem::path(params.output_path) / params.newton_csv);
         timing_step_csv.open(filesystem::path(params.output_path) / params.timing_step_csv);

         if (!probes_csv || !mass_csv || !boundary_csv || !newton_csv || !timing_step_csv)
         {
            throw runtime_error("Failed to open one or more CSV output files.");
         }

         probes_csv << "time,wall";
         for (int i = 1; i < static_cast<int>(params.probe_y.size()); ++i)
         {
            probes_csv << ",TC" << i;
         }
         probes_csv << "\n";
         mass_csv << "time,m_dot_g_surf,m_dot_c,front_98_virgin,front_2_char,recession\n";
         boundary_csv << "time,m_dot_g_surf,BprimeG_surf,BprimeC_surf,h_w_surf,"
                      << "emissivity_surf,absorptivity_surf,reflectivity_surf,"
                      << "blowing_correction_surf,q_conv_surf,q_adv_pyro_surf,"
                      << "q_rad_emit_surf,q_rad_abs_surf,q_surf\n";
         newton_csv << "step,time,iter,residual,residual0,rel_residual,"
                    << "update_norm,update0,rel_update,converged\n";
         timing_step_csv << "step,bc_sec,newton_sec,newton_residual_eval_sec,"
                         << "newton_jacobian_sec,newton_linear_sec,"
                         << "newton_update_sec,state_advance_sec,"
                         << "output_sec,step_total_sec\n";

         probes_csv << setprecision(16);
         mass_csv << setprecision(16);
         boundary_csv << setprecision(16);
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

         const double wallT = SampleFieldAtPoint(*pmesh, T, params.probe_x, params.probe_y[0]);
         vector<double> probe_vals;
         probe_vals.reserve(params.probe_y.size() - 1);
         for (int i = 1; i < static_cast<int>(params.probe_y.size()); ++i)
         {
            const double val = SampleFieldAtPoint(*pmesh, T, params.probe_x, params.probe_y[i]);
            probe_vals.push_back(val);
         }

         const SurfaceBoundaryDiagnostics bdiag =
            ComputeTopBoundaryDiagnostics(*pmesh,
                                          fes_T,
                                          fes_p,
                                          T,
                                          p,
                                          material,
                                          state_manager,
                                          bprime_table,
                                          bc_schedule,
                                          surface_model,
                                          gravity,
                                          params.bdr_attr_top,
                                          time,
                                          !use_temperature_dirichlet);
         const double mdot_surf = bdiag.m_dot_g_surf;
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
                     << 0.0 << ","
                     << front98 << ","
                     << front2 << ","
                     << 0.0 << "\n";
            boundary_csv << time << "," << bdiag.m_dot_g_surf << ","
                         << bdiag.BprimeG_surf << ","
                         << bdiag.BprimeC_surf << ","
                         << bdiag.h_w_surf << ","
                         << bdiag.emissivity_surf << ","
                         << bdiag.absorptivity_surf << ","
                         << bdiag.reflectivity_surf << ","
                         << bdiag.blowing_correction_surf << ","
                         << bdiag.q_conv_surf << ","
                         << bdiag.q_adv_pyro_surf << ","
                         << bdiag.q_rad_emit_surf << ","
                         << bdiag.q_rad_abs_surf << ","
                         << bdiag.q_surf << "\n";

            probes_csv.flush();
            mass_csv.flush();
            boundary_csv.flush();
         }

         if (params.save_paraview && (step % params.output_every == 0))
         {
            paraview_dc.SetCycle(step);
            paraview_dc.SetTime(time);
            paraview_dc.Save();
         }

         return bdiag;
      };

      write_outputs(restart_step, restart_time);

      MPI_Barrier(MPI_COMM_WORLD);
      const double setup_time_local = ElapsedSec(setup_t0, steady_clock_t::now());
      double setup_time_global = 0.0;
      MPI_Allreduce(&setup_time_local, &setup_time_global, 1, MPI_DOUBLE,
                    MPI_MAX, MPI_COMM_WORLD);

      const int nsteps_remaining_est = static_cast<int>(
         ceil(std::max(0.0, params.t_final - restart_time) / params.dt - 1.0e-12));
      if (myid == 0)
      {
         cout << "Time steps remaining: " << nsteps_remaining_est
              << ", restart step: " << restart_step
              << ", restart time: " << restart_time
              << ", final time target: " << params.t_final << endl;
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
      bool sign_sanity_logged = false;
      bool cooling_compat_logged = false;

      double time = restart_time;
      int step = restart_step;
      int steps_executed = 0;
      bool restart_write_at_done =
         (!std::isfinite(params.restart_write_at_time) ||
          restart_time >= (params.restart_write_at_time - kRestartTimeTol));
      while (time < (params.t_final - kRestartTimeTol))
      {
         ++step;
         ++steps_executed;
         const auto step_t0 = steady_clock_t::now();
         const double time_prev = time;
         const double t_next = min(params.t_final, time + params.dt);
         const double dt_step = t_next - time;
         time = t_next;

         T_old = T;
         p_old = p;
         tp_integrator->SetTimeStep(dt_step);
         if (surf_integrator)
         {
            surf_integrator->SetTime(time);
         }

         // Build the initial Newton iterate from previous solution with updated BCs.
         const auto bc_t0 = steady_clock_t::now();
         T.GetTrueDofs(xb.GetBlock(0));
         p.GetTrueDofs(xb.GetBlock(1));

         const SurfaceBCSchedule::BoundaryState bc_now = bc_schedule.Eval(time);
         if (!cooling_compat_logged &&
             params.pato_compat_mode == PatoCompatMode::CoolingExact &&
             bc_now.chemistryOn == 0)
         {
            cooling_compat_logged = true;
            if (myid == 0)
            {
               cout << "Activating pato_compat_mode="
                    << PatoCompatModeName(params.pato_compat_mode)
                    << " at t=" << time << " s (chemistryOn=0)." << endl;
            }
         }
         if (use_temperature_dirichlet)
         {
            xb.GetBlock(0).SetSubVector(ess_tdof_T, top_temperature_at(time));
         }
         xb.GetBlock(1).SetSubVector(ess_tdof_p, bc_now.p_w);
         const double step_bc_sec = ElapsedSec(bc_t0, steady_clock_t::now());

         auto enforce_bc = [&](Vector &x_true)
         {
            BlockVector x_true_b(x_true, block_true_offsets);
            if (use_temperature_dirichlet)
            {
               x_true_b.GetBlock(0).SetSubVector(ess_tdof_T, top_temperature_at(time));
            }
            x_true_b.GetBlock(1).SetSubVector(ess_tdof_p, bc_now.p_w);
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
         auto pre_residual_hook = [](const int, Vector &) {};

         const auto newton_t0 = steady_clock_t::now();
         const newton_utils::NewtonSolveResult newton_result =
            newton_solver.Solve(block_form,
                                x,
                                enforce_bc,
                                log_iteration,
                                pre_residual_hook,
                                step);
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
         const SurfaceBoundaryDiagnostics bdiag_now = write_outputs(step, time);
         if (!params.restart_write_file.empty())
         {
            bool write_restart = false;
            if (params.restart_write_every > 0 &&
                (step % params.restart_write_every) == 0)
            {
               write_restart = true;
            }

            if (!restart_write_at_done &&
                std::isfinite(params.restart_write_at_time) &&
                time_prev < (params.restart_write_at_time - kRestartTimeTol) &&
                time >= (params.restart_write_at_time - kRestartTimeTol))
            {
               write_restart = true;
               restart_write_at_done = true;
            }

            if (write_restart)
            {
               SaveRestartCheckpoint(params.restart_write_file,
                                     step,
                                     time,
                                     T,
                                     p,
                                     state_manager,
                                     world_size);
            }
         }
         const double step_output_sec = ElapsedSec(output_t0, steady_clock_t::now());
         const double step_total_sec = ElapsedSec(step_t0, steady_clock_t::now());

         if (!sign_sanity_logged)
         {
            const SurfaceBCSchedule::BoundaryState bc_now = bc_schedule.Eval(time);
            if (!use_temperature_dirichlet &&
                bc_now.chemistryOn && bc_now.rhoeUeCH > 1.0e-12)
            {
               sign_sanity_logged = true;
               if (myid == 0)
               {
                  cout << "Surface-flux sanity at t=" << time << " s: q_surf="
                       << bdiag_now.q_surf << " W/m^2"
                       << (bdiag_now.q_surf > 0.0 ? " (heating)" : " (cooling)") << endl;
               }
            }
         }

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

      if (!params.restart_write_file.empty())
      {
         SaveRestartCheckpoint(params.restart_write_file,
                               step,
                               time,
                               T,
                               p,
                               state_manager,
                               world_size);
      }

      const double run_time_local = ElapsedSec(run_t0, steady_clock_t::now());
      double run_time_global = 0.0;
      MPI_Allreduce(&run_time_local, &run_time_global, 1, MPI_DOUBLE,
                    MPI_MAX, MPI_COMM_WORLD);

      if (myid == 0)
      {
         ofstream tol_csv(filesystem::path(params.output_path) / "amaryllis_error_tolerances.csv");
         tol_csv << "signal,tolerance\n";
         tol_csv << "temperature_rmse_max," << params.tol_temp_rmse << "\n";
         tol_csv << "temperature_max_abs_max," << params.tol_temp_max_abs << "\n";
         tol_csv << "m_dot_g_rmse_max," << params.tol_mdot_rmse << "\n";
         tol_csv << "m_dot_g_max_abs_max," << params.tol_mdot_max_abs << "\n";
         tol_csv << "m_dot_g_peak_rel_error_max," << params.tol_mdot_peak_rel << "\n";
         tol_csv << "m_dot_g_peak_time_error_max," << params.tol_mdot_peak_time << "\n";
         tol_csv << "front98_max_abs_max," << params.tol_front98_max_abs << "\n";
         tol_csv << "front98_rmse_max," << params.tol_front98_rmse << "\n";
         tol_csv << "front2_max_abs_max," << params.tol_front2_max_abs << "\n";
         tol_csv << "front2_rmse_max," << params.tol_front2_rmse << "\n";
         tol_csv << "m_dot_c_max_abs_max," << params.tol_mdot_c_max_abs << "\n";
         tol_csv << "recession_max_abs_max," << params.tol_recession_max_abs << "\n";

         const BPrimeTable::ClampStats clamp_stats = bprime_table.GetClampStats();
         ofstream clamp_csv(filesystem::path(params.output_path) / "bprime_clamp_stats.csv");
         clamp_csv << "axis,clamp_count\n";
         clamp_csv << "pressure," << clamp_stats.p << "\n";
         clamp_csv << "BprimeG," << clamp_stats.bg << "\n";
         clamp_csv << "temperature," << clamp_stats.t << "\n";

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
                            << (timing_sum_step /
                                static_cast<double>(max(1, steps_executed)))
                            << "\n";

         cout << "Timing summary (max over ranks):" << endl
              << "  setup: " << setup_time_global << " s\n"
              << "  run total: " << run_time_global << " s\n"
              << "  step total sum: " << timing_sum_step << " s\n"
              << "  step avg: "
              << (timing_sum_step /
                  static_cast<double>(max(1, steps_executed)))
              << " s\n"
              << "  bc: " << timing_sum_bc << " s\n"
              << "  newton: " << timing_sum_newton << " s\n"
              << "    residual eval: " << timing_sum_newton_res << " s\n"
              << "    jacobian: " << timing_sum_newton_jac << " s\n"
              << "    linear solve: " << timing_sum_newton_lin << " s\n"
              << "    update: " << timing_sum_newton_upd << " s\n"
              << "  state advance: " << timing_sum_state << " s\n"
              << "  output: " << timing_sum_output << " s\n"
              << "B-prime clamp counts: p=" << clamp_stats.p
              << ", B'g=" << clamp_stats.bg
              << ", T=" << clamp_stats.t << endl;
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

#include "mfem.hpp"
#include "bprime_table.hpp"
#include "mesh_recession_handler.hpp"
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
   struct RemapSelfTestConfig
   {
      bool enabled = false;
      double abs_tol = 1.0e-11;
      double rel_tol = 1.0e-10;
   };

   string mesh_file = "Mesh/ablation_strip_tri_uniform.msh";
   string material_file = "Input/material_tacot_case2_2.yaml";

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

   string petsc_options_file = "Input/petsc_ablation_case2_2.opts";
   string ksp_prefix = "ablation22_ls_";
   int petsc_ksp_print_level = 0;
   bool petsc_use_matnest = false;

   int output_every = 10;
   string output_path = "ParaView/ablation_case2_2";
   string collection_name = "ablation_test_case2_2_2D";
   string probes_csv = "temperature_probes.csv";
   string pressure_probes_csv = "pressure_probes.csv";
   string mesh_diagnostics_csv = "mesh_diagnostics.csv";
   string mass_eq_probe_csv = "mass_eq_probe_diagnostics.csv";
   string mass_csv = "mass_metrics.csv";
   string boundary_csv = "boundary_diagnostics.csv";
   string newton_csv = "newton_history_ablation_case2_2_2D.csv";
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
   bool moving_mesh = true;
   bool ale_enabled = true;
   bool ale_mass_enabled = true;
   bool ale_energy_enabled = true;
   bool ale_energy_solid_enabled = true;
   bool ale_energy_gas_enabled = true;
   bool ale_remap_enabled = true;
   string ale_remap_extent_mode = "nearest_qp";
   int ale_remap_extent_l2_order = 1;
   string mesh_smoothing_model = "laplacian";
   string recession_density_mode = "char_surface";
   double recession_density_constant = 1200.0;
   double max_step_recession = 2.0e-4;
   double min_quality_ratio = 0.2;

   string bprime_table_file =
      "/home/quinnchr/Downloads/pato-3.1/data/Environments/Tables/TACOT-Earth-1atm";
   string boundary_conditions_file = "Input/boundary_conditions_ablation_case2_2.dat";
   // Top thermal BC mode: "surface_energy_balance" or "temperature_dirichlet".
   string top_thermal_bc = "surface_energy_balance";
   // Used only when top_thermal_bc == "temperature_dirichlet".
   double top_temperature_value = 300.0;
   // Optional time-temperature table for top_thermal_bc=temperature_dirichlet.
   // Columns: time(s) temperature(K).
   string top_temperature_file = "";
   // Top recession BC mode: "computed_surface_mass_loss" or "recession_file".
   string top_recession_bc = "computed_surface_mass_loss";
   // Optional recession history file for top_recession_bc=recession_file.
   // Supported formats:
   //   1. Columns: time(s) recession(m)
   //   2. A wider file whose first numeric column is time and last numeric
   //      column is recession, e.g. the PATO/Amaryllis mass history files.
   string top_recession_file = "";

   // Surface energy-balance model (Test Case 2.2).
   double lambda = 0.5;
   double q_rad = 0.0;
   double T_background = 300.0;
   double T_edge = 300.0;
   double hconv = 0.0;
   double emissivity = std::numeric_limits<double>::quiet_NaN();
   double absorptivity = std::numeric_limits<double>::quiet_NaN();
   double stefan_boltzmann = 5.670374419e-8;
   bool disable_bprime_c = false;
   PatoCompatMode pato_compat_mode = PatoCompatMode::Off;

   double gravity_x = 0.0;
   double gravity_y = 0.0;

   double probe_x = 0.005;
   vector<double> probe_y = {0.05, 0.049, 0.048, 0.046, 0.042, 0.038, 0.034, 0.026, 0.01};

   string amaryllis_energy_file =
      "/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_2.x/data/ref/Amaryllis/Amaryllis_Energy_TestCase_2.2.txt";
   string amaryllis_mass_file =
      "/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_2.x/data/ref/Amaryllis/Amaryllis_Mass_TestCase_2.2.txt";

   // Acceptance tolerances for compare_ablation_case2_2.py
   double tol_temp_rmse = 300.0;
   double tol_temp_max_abs = 650.0;
   double tol_mdot_rmse = 0.025;
   double tol_mdot_max_abs = 0.08;
   double tol_mdot_peak_rel = 0.5;
   double tol_mdot_peak_time = 10.0;
   double tol_front98_max_abs = 0.01;
   double tol_front98_rmse = 0.01;
   double tol_front2_max_abs = 0.01;
   double tol_front2_rmse = 0.01;
   double tol_mdot_c_rmse = 0.01;
   double tol_mdot_c_peak_rel = 0.35;
   double tol_recession_rmse = 0.0015;
   double tol_recession_final_rel = 0.12;

   RemapSelfTestConfig remap_self_test;
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

string NormalizeTopRecessionBC(string mode)
{
   transform(mode.begin(), mode.end(), mode.begin(),
             [](unsigned char c) { return static_cast<char>(tolower(c)); });

   if (mode == "computed" || mode == "surface_mass_loss" ||
       mode == "computed_surface" || mode == "bprime")
   {
      return "computed_surface_mass_loss";
   }
   if (mode == "file" || mode == "prescribed" || mode == "history" ||
       mode == "recession_history" || mode == "recession")
   {
      return "recession_file";
   }
   return mode;
}

string NormalizeAleRemapExtentMode(string mode)
{
   transform(mode.begin(), mode.end(), mode.begin(),
             [](unsigned char c) { return static_cast<char>(tolower(c)); });

   if (mode == "nearest" || mode == "nearest-qp" || mode == "nearestqp")
   {
      return "nearest_qp";
   }
   if (mode == "l2" || mode == "l2_point" || mode == "l2_point_eval")
   {
      return "l2_point_eval";
   }
   if (mode == "h1" || mode == "h1_point" || mode == "h1_point_eval")
   {
      return "h1_point_eval";
   }
   if (mode == "conservative" || mode == "l2-conservative" ||
       mode == "l2_conservative")
   {
      return "l2_conservative";
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

class TopRecessionSchedule
{
public:
   void LoadFromFile(const string &path)
   {
      times_.clear();
      recession_.clear();

      ifstream in(path);
      if (!in)
      {
         throw runtime_error("Failed to open top recession schedule: " + path);
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
         vector<double> values;
         double value = 0.0;
         while (iss >> value)
         {
            values.push_back(value);
         }
         if (values.size() < 2)
         {
            continue;
         }

         const double t = values.front();
         const double rec = values.back();
         if (t < 0.0)
         {
            throw runtime_error("Negative time in top recession schedule at line " +
                                to_string(line_no) + ": " + path);
         }
         if (!times_.empty() && t < times_.back())
         {
            throw runtime_error(
               "Top recession schedule times must be nondecreasing at line " +
               to_string(line_no) + ": " + path);
         }
         if (!std::isfinite(rec))
         {
            throw runtime_error(
               "Non-finite recession value in top recession schedule at line " +
               to_string(line_no) + ": " + path);
         }

         times_.push_back(t);
         recession_.push_back(std::max(0.0, rec));
      }

      if (times_.empty())
      {
         throw runtime_error("Top recession schedule is empty: " + path);
      }
   }

   bool Empty() const { return times_.empty(); }

   double EvalRecession(const double time) const
   {
      if (times_.empty())
      {
         throw runtime_error("Top recession schedule EvalRecession() called before LoadFromFile().");
      }
      if (time <= times_.front())
      {
         return recession_.front();
      }
      if (time >= times_.back())
      {
         return recession_.back();
      }

      auto it = std::upper_bound(times_.begin(), times_.end(), time);
      const int i1 = static_cast<int>(it - times_.begin());
      const int i0 = i1 - 1;
      const double t0 = times_[i0];
      const double t1 = times_[i1];
      const double r0 = recession_[i0];
      const double r1 = recession_[i1];
      if (std::abs(t1 - t0) < 1.0e-14)
      {
         return r1;
      }
      const double alpha = (time - t0) / (t1 - t0);
      return (1.0 - alpha) * r0 + alpha * r1;
   }

   double AverageRate(const double t0, const double t1) const
   {
      if (!(t1 > t0))
      {
         return 0.0;
      }
      const double r0 = EvalRecession(t0);
      const double r1 = EvalRecession(t1);
      if (!std::isfinite(r0) || !std::isfinite(r1))
      {
         return 0.0;
      }
      return std::max(0.0, (r1 - r0) / (t1 - t0));
   }

private:
   vector<double> times_;
   vector<double> recession_;
};

string EffectiveTopRecessionFile(const DriverParams &p)
{
   if (!p.top_recession_file.empty())
   {
      return p.top_recession_file;
   }
   return p.amaryllis_mass_file;
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
   if (n["petsc_use_matnest"]) { p.petsc_use_matnest = n["petsc_use_matnest"].as<bool>(); }

   if (n["output_every"]) { p.output_every = n["output_every"].as<int>(); }
   if (n["output_path"]) { p.output_path = n["output_path"].as<string>(); }
   if (n["collection_name"]) { p.collection_name = n["collection_name"].as<string>(); }
   if (n["probes_csv"]) { p.probes_csv = n["probes_csv"].as<string>(); }
   if (n["pressure_probes_csv"]) { p.pressure_probes_csv = n["pressure_probes_csv"].as<string>(); }
   if (n["mesh_diagnostics_csv"]) { p.mesh_diagnostics_csv = n["mesh_diagnostics_csv"].as<string>(); }
   if (n["mass_eq_probe_csv"]) { p.mass_eq_probe_csv = n["mass_eq_probe_csv"].as<string>(); }
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
   if (n["moving_mesh"]) { p.moving_mesh = n["moving_mesh"].as<bool>(); }
   if (n["ale_enabled"]) { p.ale_enabled = n["ale_enabled"].as<bool>(); }
   if (n["ale_mass_enabled"]) { p.ale_mass_enabled = n["ale_mass_enabled"].as<bool>(); }
   if (n["ale_energy_enabled"]) { p.ale_energy_enabled = n["ale_energy_enabled"].as<bool>(); }
   if (n["ale_energy_solid_enabled"]) { p.ale_energy_solid_enabled = n["ale_energy_solid_enabled"].as<bool>(); }
   if (n["ale_energy_gas_enabled"]) { p.ale_energy_gas_enabled = n["ale_energy_gas_enabled"].as<bool>(); }
   if (n["ale_remap_enabled"]) { p.ale_remap_enabled = n["ale_remap_enabled"].as<bool>(); }
   if (n["ale_remap_extent_mode"])
   {
      p.ale_remap_extent_mode = n["ale_remap_extent_mode"].as<string>();
   }
   if (n["ale_remap_extent_l2_order"])
   {
      p.ale_remap_extent_l2_order = n["ale_remap_extent_l2_order"].as<int>();
   }
   if (n["mesh_smoothing_model"])
   {
      p.mesh_smoothing_model = n["mesh_smoothing_model"].as<string>();
   }
   if (n["recession_density_mode"])
   {
      p.recession_density_mode = n["recession_density_mode"].as<string>();
   }
   if (n["recession_density_constant"])
   {
      p.recession_density_constant = n["recession_density_constant"].as<double>();
   }
   if (n["max_step_recession"])
   {
      p.max_step_recession = n["max_step_recession"].as<double>();
   }
   if (n["min_quality_ratio"])
   {
      p.min_quality_ratio = n["min_quality_ratio"].as<double>();
   }

   if (n["bprime_table_file"]) { p.bprime_table_file = n["bprime_table_file"].as<string>(); }
   if (n["boundary_conditions_file"]) { p.boundary_conditions_file = n["boundary_conditions_file"].as<string>(); }
   if (n["top_thermal_bc"]) { p.top_thermal_bc = n["top_thermal_bc"].as<string>(); }
   if (n["top_temperature_value"]) { p.top_temperature_value = n["top_temperature_value"].as<double>(); }
   if (n["top_temperature_file"]) { p.top_temperature_file = n["top_temperature_file"].as<string>(); }
   if (n["top_recession_bc"]) { p.top_recession_bc = n["top_recession_bc"].as<string>(); }
   if (n["top_recession_file"]) { p.top_recession_file = n["top_recession_file"].as<string>(); }
   if (n["lambda"]) { p.lambda = n["lambda"].as<double>(); }
   if (n["q_rad"]) { p.q_rad = n["q_rad"].as<double>(); }
   if (n["T_background"]) { p.T_background = n["T_background"].as<double>(); }
   if (n["T_edge"]) { p.T_edge = n["T_edge"].as<double>(); }
   if (n["hconv"]) { p.hconv = n["hconv"].as<double>(); }
   if (n["emissivity"]) { p.emissivity = n["emissivity"].as<double>(); }
   if (n["absorptivity"]) { p.absorptivity = n["absorptivity"].as<double>(); }
   if (n["stefan_boltzmann"]) { p.stefan_boltzmann = n["stefan_boltzmann"].as<double>(); }
   if (n["strict_case2_2"]) { p.disable_bprime_c = n["strict_case2_2"].as<bool>(); }
   if (n["disable_bprime_c"]) { p.disable_bprime_c = n["disable_bprime_c"].as<bool>(); }
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

   if (n["remap_self_test"])
   {
      YAML::Node rst = n["remap_self_test"];
      if (rst["enabled"]) { p.remap_self_test.enabled = rst["enabled"].as<bool>(); }
      if (rst["abs_tol"]) { p.remap_self_test.abs_tol = rst["abs_tol"].as<double>(); }
      if (rst["rel_tol"]) { p.remap_self_test.rel_tol = rst["rel_tol"].as<double>(); }
   }

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
      if (a["m_dot_c_rmse_max"]) { p.tol_mdot_c_rmse = a["m_dot_c_rmse_max"].as<double>(); }
      if (a["m_dot_c_peak_rel_error_max"]) { p.tol_mdot_c_peak_rel = a["m_dot_c_peak_rel_error_max"].as<double>(); }
      if (a["recession_rmse_max"]) { p.tol_recession_rmse = a["recession_rmse_max"].as<double>(); }
      if (a["recession_final_rel_error_max"])
      {
         p.tol_recession_final_rel = a["recession_final_rel_error_max"].as<double>();
      }
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
   if (p.recession_density_constant <= 0.0)
   {
      throw runtime_error("recession_density_constant must be > 0.");
   }
   if (p.max_step_recession < 0.0)
   {
      throw runtime_error("max_step_recession must be >= 0.");
   }
   if (p.min_quality_ratio <= 0.0 || p.min_quality_ratio >= 1.0)
   {
      throw runtime_error("min_quality_ratio must be in (0,1).");
   }
   p.ale_remap_extent_mode =
      NormalizeAleRemapExtentMode(p.ale_remap_extent_mode);
   if (p.ale_remap_extent_mode != "nearest_qp" &&
       p.ale_remap_extent_mode != "l2_point_eval" &&
       p.ale_remap_extent_mode != "h1_point_eval" &&
       p.ale_remap_extent_mode != "l2_conservative")
   {
      throw runtime_error(
         "ale_remap_extent_mode must be \"nearest_qp\", \"l2_point_eval\", "
         "\"h1_point_eval\", or \"l2_conservative\".");
   }
   if (p.ale_remap_extent_l2_order < 0 || p.ale_remap_extent_l2_order > 2)
   {
      throw runtime_error("ale_remap_extent_l2_order must be in [0,2].");
   }
   if (p.remap_self_test.abs_tol < 0.0)
   {
      throw runtime_error("remap_self_test.abs_tol must be >= 0.");
   }
   if (p.remap_self_test.rel_tol < 0.0)
   {
      throw runtime_error("remap_self_test.rel_tol must be >= 0.");
   }

   transform(p.mesh_smoothing_model.begin(), p.mesh_smoothing_model.end(),
             p.mesh_smoothing_model.begin(),
             [](unsigned char c) { return static_cast<char>(tolower(c)); });
   transform(p.recession_density_mode.begin(), p.recession_density_mode.end(),
             p.recession_density_mode.begin(),
             [](unsigned char c) { return static_cast<char>(tolower(c)); });
   if (p.mesh_smoothing_model != "laplacian")
   {
      throw runtime_error("mesh_smoothing_model must be \"laplacian\".");
   }
   if (p.recession_density_mode != "char_surface" &&
       p.recession_density_mode != "constant")
   {
      throw runtime_error(
         "recession_density_mode must be either \"char_surface\" or \"constant\".");
   }

   p.top_thermal_bc = NormalizeTopThermalBC(p.top_thermal_bc);
   if (p.top_thermal_bc != "surface_energy_balance" &&
       p.top_thermal_bc != "temperature_dirichlet")
   {
      throw runtime_error(
         "top_thermal_bc must be either \"surface_energy_balance\" "
         "or \"temperature_dirichlet\".");
   }

   p.top_recession_bc = NormalizeTopRecessionBC(p.top_recession_bc);
   if (p.top_recession_bc != "computed_surface_mass_loss" &&
       p.top_recession_bc != "recession_file")
   {
      throw runtime_error(
         "top_recession_bc must be either \"computed_surface_mass_loss\" "
         "or \"recession_file\".");
   }
   if (p.moving_mesh && p.top_recession_bc == "recession_file")
   {
      const string recession_file = EffectiveTopRecessionFile(p);
      if (recession_file.empty())
      {
         throw runtime_error(
            "top_recession_file or amaryllis_mass_file must be set when "
            "top_recession_bc=recession_file.");
      }
      if (!filesystem::exists(recession_file))
      {
         throw runtime_error("Top recession schedule file not found: " +
                             recession_file);
      }
   }
}

Bounds GetGlobalBounds(const ParMesh &pmesh)
{
   double local_min[2] = {numeric_limits<double>::infinity(),
                          numeric_limits<double>::infinity()};
   double local_max[2] = {-numeric_limits<double>::infinity(),
                          -numeric_limits<double>::infinity()};

   const GridFunction *nodes = pmesh.GetNodes();
   if (nodes != nullptr)
   {
      const FiniteElementSpace *nfes = nodes->FESpace();
      const int ndof = nfes->GetNDofs();
      for (int i = 0; i < ndof; ++i)
      {
         const double x = (*nodes)(nfes->DofToVDof(i, 0));
         const double y = (*nodes)(nfes->DofToVDof(i, 1));
         local_min[0] = min(local_min[0], x);
         local_min[1] = min(local_min[1], y);
         local_max[0] = max(local_max[0], x);
         local_max[1] = max(local_max[1], y);
      }
   }
   else
   {
      for (int i = 0; i < pmesh.GetNV(); ++i)
      {
         const double *v = pmesh.GetVertex(i);
         local_min[0] = min(local_min[0], v[0]);
         local_min[1] = min(local_min[1], v[1]);
         local_max[0] = max(local_max[0], v[0]);
         local_max[1] = max(local_max[1], v[1]);
      }
   }

   Bounds b;
   MPI_Allreduce(local_min, &b.xmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(local_max, &b.xmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(local_min + 1, &b.ymin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(local_max + 1, &b.ymax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return b;
}

void CopyParGridFunctionByTrueDofs(const ParGridFunction &src,
                                   ParGridFunction &dst)
{
   Vector true_dofs;
   src.GetTrueDofs(true_dofs);
   dst.SetFromTrueDofs(true_dofs);
}

double ComputeMinElementQuality(ParMesh &pmesh)
{
   double local_min = std::numeric_limits<double>::infinity();
   for (int e = 0; e < pmesh.GetNE(); ++e)
   {
      ElementTransformation *Tr = pmesh.GetElementTransformation(e);
      const IntegrationRule &ir = IntRules.Get(pmesh.GetElementBaseGeometry(e), 2);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr->SetIntPoint(&ip);
         local_min = std::min(local_min, static_cast<double>(Tr->Weight()));
      }
   }

   double global_min = local_min;
   MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, pmesh.GetComm());
   return global_min;
}

class AleJacobianCoefficient : public Coefficient
{
public:
   explicit AleJacobianCoefficient(const ParGridFunction &ale_displacement)
      : ale_displacement_(ale_displacement) {}

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      T.SetIntPoint(&ip);
      DenseMatrix grad;
      ale_displacement_.GetVectorGradient(T, grad);
      MFEM_VERIFY(grad.Height() == 2 && grad.Width() == 2,
                  "AleJacobianCoefficient expects a 2D displacement field.");

      const double g00 = 1.0 + grad(0, 0);
      const double g01 = grad(0, 1);
      const double g10 = grad(1, 0);
      const double g11 = 1.0 + grad(1, 1);
      return g00 * g11 - g01 * g10;
   }

private:
   const ParGridFunction &ale_displacement_;
};

void BuildAleDeformationMap2D(const DenseMatrix &grad_disp,
                              DenseMatrix &F,
                              DenseMatrix &cofactor,
                              DenseMatrix &invF,
                              double &J)
{
   MFEM_VERIFY(grad_disp.Height() == 2 && grad_disp.Width() == 2,
               "BuildAleDeformationMap2D expects a 2x2 displacement gradient.");

   const double g00 = 1.0 + grad_disp(0, 0);
   const double g01 = grad_disp(0, 1);
   const double g10 = grad_disp(1, 0);
   const double g11 = 1.0 + grad_disp(1, 1);

   F.SetSize(2, 2);
   F(0, 0) = g00;
   F(0, 1) = g01;
   F(1, 0) = g10;
   F(1, 1) = g11;

   J = g00 * g11 - g01 * g10;
   cofactor.SetSize(2, 2);
   cofactor(0, 0) = g11;
   cofactor(0, 1) = -g01;
   cofactor(1, 0) = -g10;
   cofactor(1, 1) = g00;

   invF.SetSize(2, 2);
   invF = cofactor;
   invF *= (1.0 / J);
}

struct AleFaceGeometry2D
{
   Vector x_current;
   Vector area_vector;
   Vector unit_normal;
   double ds = 0.0;
   bool valid = false;
};

void EvaluateAleMap2D(ElementTransformation &Tr,
                      const IntegrationPoint &ip,
                      const ParGridFunction *ale_displacement,
                      Vector &x_current,
                      DenseMatrix &F,
                      DenseMatrix &cofactor,
                      DenseMatrix &invF,
                      double &J)
{
   const int dim = Tr.GetSpaceDim();
   MFEM_VERIFY(dim == 2, "EvaluateAleMap2D expects a 2D element transformation.");

   Tr.SetIntPoint(&ip);
   x_current.SetSize(dim);
   Tr.Transform(ip, x_current);

   F.SetSize(dim, dim);
   cofactor.SetSize(dim, dim);
   invF.SetSize(dim, dim);
   F = 0.0;
   cofactor = 0.0;
   invF = 0.0;
   for (int d = 0; d < dim; ++d)
   {
      F(d, d) = 1.0;
      cofactor(d, d) = 1.0;
      invF(d, d) = 1.0;
   }
   J = 1.0;

   if (!ale_displacement)
   {
      return;
   }

   Vector disp(dim);
   ale_displacement->GetVectorValue(Tr, ip, disp);
   x_current += disp;

   DenseMatrix grad_disp;
   ale_displacement->GetVectorGradient(Tr, grad_disp);
   BuildAleDeformationMap2D(grad_disp, F, cofactor, invF, J);
}

void ApplyInverseTranspose2D(const DenseMatrix &invF,
                             const Vector &grad_ref,
                             Vector &grad_cur)
{
   MFEM_VERIFY(invF.Height() == 2 && invF.Width() == 2,
               "ApplyInverseTranspose2D expects a 2x2 inverse deformation map.");
   MFEM_VERIFY(grad_ref.Size() == 2,
               "ApplyInverseTranspose2D expects a 2D reference gradient.");

   grad_cur.SetSize(2);
   grad_cur[0] = invF(0, 0) * grad_ref[0] + invF(1, 0) * grad_ref[1];
   grad_cur[1] = invF(0, 1) * grad_ref[0] + invF(1, 1) * grad_ref[1];
}

double ComputeCurrentVectorDivergence2D(const DenseMatrix &grad_ref,
                                        const DenseMatrix &invF)
{
   MFEM_VERIFY(grad_ref.Height() == 2 && grad_ref.Width() == 2,
               "ComputeCurrentVectorDivergence2D expects a 2x2 reference gradient.");
   MFEM_VERIFY(invF.Height() == 2 && invF.Width() == 2,
               "ComputeCurrentVectorDivergence2D expects a 2x2 inverse deformation map.");

   double div_cur = 0.0;
   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 2; ++j)
      {
         div_cur += grad_ref(i, j) * invF(j, i);
      }
   }
   return div_cur;
}

bool ReferencePointInside2D(const Geometry::Type geom,
                            const IntegrationPoint &ip,
                            const double tol)
{
   switch (geom)
   {
      case Geometry::TRIANGLE:
         return (ip.x >= -tol &&
                 ip.y >= -tol &&
                 (ip.x + ip.y) <= (1.0 + tol));
      case Geometry::SQUARE:
         return (ip.x >= -tol &&
                 ip.x <= (1.0 + tol) &&
                 ip.y >= -tol &&
                 ip.y <= (1.0 + tol));
      default:
         MFEM_ABORT("ReferencePointInside2D only supports triangles and quads.");
   }
}

vector<IntegrationPoint> ReferenceSeeds2D(const Geometry::Type geom)
{
   vector<IntegrationPoint> seeds;
   auto add_seed = [&](const double x, const double y)
   {
      IntegrationPoint ip;
      ip.x = x;
      ip.y = y;
      ip.z = 0.0;
      seeds.push_back(ip);
   };

   switch (geom)
   {
      case Geometry::TRIANGLE:
         add_seed(1.0 / 3.0, 1.0 / 3.0);
         add_seed(0.0, 0.0);
         add_seed(1.0, 0.0);
         add_seed(0.0, 1.0);
         add_seed(0.5, 0.0);
         add_seed(0.5, 0.5);
         add_seed(0.0, 0.5);
         break;
      case Geometry::SQUARE:
         add_seed(0.5, 0.5);
         add_seed(0.0, 0.0);
         add_seed(1.0, 0.0);
         add_seed(1.0, 1.0);
         add_seed(0.0, 1.0);
         add_seed(0.5, 0.0);
         add_seed(1.0, 0.5);
         add_seed(0.5, 1.0);
         add_seed(0.0, 0.5);
         break;
      default:
         MFEM_ABORT("ReferenceSeeds2D only supports triangles and quads.");
   }
   return seeds;
}

bool Solve2x2(const DenseMatrix &A,
              const Vector &rhs,
              Vector &x)
{
   MFEM_VERIFY(A.Height() == 2 && A.Width() == 2,
               "Solve2x2 expects a 2x2 matrix.");
   MFEM_VERIFY(rhs.Size() == 2,
               "Solve2x2 expects a 2D right-hand side.");

   const double det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
   if (!(std::abs(det) > 1.0e-20))
   {
      return false;
   }

   x.SetSize(2);
   x[0] = ( rhs[0] * A(1, 1) - rhs[1] * A(0, 1)) / det;
   x[1] = (-rhs[0] * A(1, 0) + rhs[1] * A(0, 0)) / det;
   return true;
}

double ComputeMinAleElementQuality(ParMesh &pmesh,
                                   const ParGridFunction *ale_displacement)
{
   double local_min = std::numeric_limits<double>::infinity();
   Vector x_current;
   DenseMatrix F, cofactor, invF;
   for (int e = 0; e < pmesh.GetNE(); ++e)
   {
      ElementTransformation *Tr = pmesh.GetElementTransformation(e);
      const IntegrationRule &ir = IntRules.Get(pmesh.GetElementBaseGeometry(e), 2);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         double J = 1.0;
         EvaluateAleMap2D(*Tr, ip, ale_displacement, x_current, F, cofactor, invF, J);
         local_min = std::min(local_min, J * static_cast<double>(Tr->Weight()));
      }
   }

   double global_min = local_min;
   MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, pmesh.GetComm());
   return global_min;
}

bool EvaluateCurrentFaceGeometry2D(FaceElementTransformations &FT,
                                   const IntegrationPoint &fip,
                                   const IntegrationPoint &eip,
                                   const ParGridFunction *ale_displacement,
                                   AleFaceGeometry2D &face_geom)
{
   Vector x_current;
   DenseMatrix F, cofactor, invF;
   double J = 1.0;
   EvaluateAleMap2D(*FT.Elem1, eip, ale_displacement,
                    x_current, F, cofactor, invF, J);

   Vector ref_area(2);
   FT.Face->SetIntPoint(&fip);
   CalcOrtho(FT.Face->Jacobian(), ref_area);

   face_geom.area_vector.SetSize(2);
   if (ale_displacement)
   {
      // BuildAleDeformationMap2D stores cofactor = J * F^{-1}, so mapped
      // current-area vectors require cofactor^T * ref_area.
      cofactor.MultTranspose(ref_area, face_geom.area_vector);
   }
   else
   {
      face_geom.area_vector = ref_area;
   }

   const double area_mag = face_geom.area_vector.Norml2();
   if (!(area_mag > 1.0e-20))
   {
      face_geom.valid = false;
      face_geom.ds = 0.0;
      return false;
   }

   face_geom.x_current = x_current;
   face_geom.unit_normal.SetSize(2);
   face_geom.unit_normal = face_geom.area_vector;
   face_geom.unit_normal /= area_mag;
   face_geom.ds = fip.weight * area_mag;
   face_geom.valid = true;
   return true;
}

class AlePointLocator2D
{
public:
   struct PointLocation
   {
      int elem = -1;
      IntegrationPoint ip;
      double residual_norm = std::numeric_limits<double>::infinity();
   };

   AlePointLocator2D(ParMesh &pmesh,
                     const ParFiniteElementSpace &fes_state)
      : pmesh_(pmesh),
        fes_state_(fes_state)
   {}

   void Update(const ParGridFunction *ale_displacement)
   {
      ale_displacement_ = ale_displacement;
      BuildElementBoundingBoxes_();
   }

   const ParGridFunction *Displacement() const { return ale_displacement_; }
   MPI_Comm Comm() const { return pmesh_.GetComm(); }
   ParMesh &Mesh() const { return pmesh_; }

   bool FindPointLocal(const Vector &x_target,
                       PointLocation &location) const
   {
      location = PointLocation{};

      vector<int> candidate_elems;
      const double bbox_tol = 1.0e-10;
      for (int e = 0; e < static_cast<int>(element_bboxes_.size()); ++e)
      {
         const ElementBBox &bbox = element_bboxes_[static_cast<size_t>(e)];
         if (x_target[0] >= (bbox.xmin - bbox_tol) &&
             x_target[0] <= (bbox.xmax + bbox_tol) &&
             x_target[1] >= (bbox.ymin - bbox_tol) &&
             x_target[1] <= (bbox.ymax + bbox_tol))
         {
            candidate_elems.push_back(e);
         }
      }

      PointLocation best_location;
      if (TryCandidateElements_(candidate_elems, x_target, best_location))
      {
         location = best_location;
         return true;
      }

      candidate_elems.clear();
      candidate_elems.reserve(fes_state_.GetNE());
      for (int e = 0; e < fes_state_.GetNE(); ++e)
      {
         candidate_elems.push_back(e);
      }
      if (TryCandidateElements_(candidate_elems, x_target, best_location))
      {
         location = best_location;
         return true;
      }

      return false;
   }

   void FindPointsAllGather(const DenseMatrix &local_points,
                            Array<int> &elem_ids,
                            Array<IntegrationPoint> &ref_pts) const
   {
      const int local_npts = local_points.Width();
      const int dim = local_points.Height();
      MFEM_VERIFY(dim == 2, "AlePointLocator2D expects 2D query points.");

      const int myid = Mpi::WorldRank();
      int world_size = 1;
      MPI_Comm_size(pmesh_.GetComm(), &world_size);

      vector<int> point_counts(static_cast<size_t>(world_size), 0);
      MPI_Allgather(&local_npts,
                    1,
                    MPI_INT,
                    point_counts.data(),
                    1,
                    MPI_INT,
                    pmesh_.GetComm());

      vector<int> point_displs(static_cast<size_t>(world_size), 0);
      vector<int> coord_counts(static_cast<size_t>(world_size), 0);
      vector<int> coord_displs(static_cast<size_t>(world_size), 0);
      int total_npts = 0;
      int total_coords = 0;
      for (int rank = 0; rank < world_size; ++rank)
      {
         point_displs[static_cast<size_t>(rank)] = total_npts;
         coord_displs[static_cast<size_t>(rank)] = total_coords;
         total_npts += point_counts[static_cast<size_t>(rank)];
         coord_counts[static_cast<size_t>(rank)] =
            point_counts[static_cast<size_t>(rank)] * dim;
         total_coords += coord_counts[static_cast<size_t>(rank)];
      }

      vector<double> gathered_point_data(static_cast<size_t>(total_coords), 0.0);
      MPI_Allgatherv(local_npts > 0 ? local_points.GetData() : nullptr,
                     local_npts * dim,
                     MPI_DOUBLE,
                     gathered_point_data.data(),
                     coord_counts.data(),
                     coord_displs.data(),
                     MPI_DOUBLE,
                     pmesh_.GetComm());

      DenseMatrix gathered_points(gathered_point_data.data(), dim, total_npts);
      vector<int> owner_candidate(static_cast<size_t>(total_npts), world_size);
      vector<int> local_elem_plus_one(static_cast<size_t>(total_npts), 0);
      vector<double> local_ipx(static_cast<size_t>(total_npts), 0.0);
      vector<double> local_ipy(static_cast<size_t>(total_npts), 0.0);
      vector<double> local_ipz(static_cast<size_t>(total_npts), 0.0);

      Vector x_target(dim);
      for (int k = 0; k < total_npts; ++k)
      {
         for (int d = 0; d < dim; ++d)
         {
            x_target[d] = gathered_points(d, k);
         }

         PointLocation location;
         if (!FindPointLocal(x_target, location))
         {
            continue;
         }

         owner_candidate[static_cast<size_t>(k)] = myid;
         local_elem_plus_one[static_cast<size_t>(k)] = location.elem + 1;
         local_ipx[static_cast<size_t>(k)] = location.ip.x;
         local_ipy[static_cast<size_t>(k)] = location.ip.y;
         local_ipz[static_cast<size_t>(k)] = location.ip.z;
      }

      vector<int> owner_rank(static_cast<size_t>(total_npts), world_size);
      MPI_Allreduce(owner_candidate.data(),
                    owner_rank.data(),
                    total_npts,
                    MPI_INT,
                    MPI_MIN,
                    pmesh_.GetComm());

      for (int k = 0; k < total_npts; ++k)
      {
         if (owner_rank[static_cast<size_t>(k)] != myid)
         {
            local_elem_plus_one[static_cast<size_t>(k)] = 0;
            local_ipx[static_cast<size_t>(k)] = 0.0;
            local_ipy[static_cast<size_t>(k)] = 0.0;
            local_ipz[static_cast<size_t>(k)] = 0.0;
         }
      }

      vector<int> global_elem_plus_one(static_cast<size_t>(total_npts), 0);
      vector<double> global_ipx(static_cast<size_t>(total_npts), 0.0);
      vector<double> global_ipy(static_cast<size_t>(total_npts), 0.0);
      vector<double> global_ipz(static_cast<size_t>(total_npts), 0.0);
      if (total_npts > 0)
      {
         MPI_Allreduce(local_elem_plus_one.data(),
                       global_elem_plus_one.data(),
                       total_npts,
                       MPI_INT,
                       MPI_SUM,
                       pmesh_.GetComm());
         MPI_Allreduce(local_ipx.data(),
                       global_ipx.data(),
                       total_npts,
                       MPI_DOUBLE,
                       MPI_SUM,
                       pmesh_.GetComm());
         MPI_Allreduce(local_ipy.data(),
                       global_ipy.data(),
                       total_npts,
                       MPI_DOUBLE,
                       MPI_SUM,
                       pmesh_.GetComm());
         MPI_Allreduce(local_ipz.data(),
                       global_ipz.data(),
                       total_npts,
                       MPI_DOUBLE,
                       MPI_SUM,
                       pmesh_.GetComm());
      }

      elem_ids.SetSize(local_npts);
      ref_pts.SetSize(local_npts);
      const int local_offset = point_displs[static_cast<size_t>(myid)];
      for (int k = 0; k < local_npts; ++k)
      {
         const int global_k = local_offset + k;
         elem_ids[k] = global_elem_plus_one[static_cast<size_t>(global_k)] - 1;
         ref_pts[k].x = global_ipx[static_cast<size_t>(global_k)];
         ref_pts[k].y = global_ipy[static_cast<size_t>(global_k)];
         ref_pts[k].z = global_ipz[static_cast<size_t>(global_k)];
      }
   }

private:
   struct ElementBBox
   {
      double xmin = std::numeric_limits<double>::infinity();
      double xmax = -std::numeric_limits<double>::infinity();
      double ymin = std::numeric_limits<double>::infinity();
      double ymax = -std::numeric_limits<double>::infinity();
   };

   void BuildElementBoundingBoxes_()
   {
      element_bboxes_.assign(static_cast<size_t>(fes_state_.GetNE()), ElementBBox{});
      Vector x_current;
      DenseMatrix F, cofactor, invF;
      for (int e = 0; e < fes_state_.GetNE(); ++e)
      {
         ElementTransformation *Tr = fes_state_.GetElementTransformation(e);
         const FiniteElement *fe = fes_state_.GetFE(e);
         const int bbox_order = std::max(6, 2 * fe->GetOrder() + 3);
         const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), bbox_order);
         ElementBBox &bbox = element_bboxes_[static_cast<size_t>(e)];
         for (int q = 0; q < ir.GetNPoints(); ++q)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            double J = 1.0;
            EvaluateAleMap2D(*Tr, ip, ale_displacement_, x_current, F, cofactor, invF, J);
            bbox.xmin = std::min(bbox.xmin, static_cast<double>(x_current[0]));
            bbox.xmax = std::max(bbox.xmax, static_cast<double>(x_current[0]));
            bbox.ymin = std::min(bbox.ymin, static_cast<double>(x_current[1]));
            bbox.ymax = std::max(bbox.ymax, static_cast<double>(x_current[1]));
         }

         const double dx = bbox.xmax - bbox.xmin;
         const double dy = bbox.ymax - bbox.ymin;
         const double pad = 1.0e-10 + 1.0e-8 * std::max(1.0, std::max(dx, dy));
         bbox.xmin -= pad;
         bbox.xmax += pad;
         bbox.ymin -= pad;
         bbox.ymax += pad;
      }
   }

   bool TryLocateInElement_(const int elem,
                            const Vector &x_target,
                            PointLocation &best_location) const
   {
      const FiniteElement *fe = fes_state_.GetFE(elem);
      Geometry::Type geom = fe->GetGeomType();
      const vector<IntegrationPoint> seeds = ReferenceSeeds2D(geom);
      ElementTransformation *Tr = fes_state_.GetElementTransformation(elem);
      const ElementBBox &bbox = element_bboxes_[static_cast<size_t>(elem)];
      const double scale = std::max(1.0,
                                    std::sqrt((bbox.xmax - bbox.xmin) * (bbox.xmax - bbox.xmin) +
                                              (bbox.ymax - bbox.ymin) * (bbox.ymax - bbox.ymin)));
      const double residual_tol = 1.0e-8 * scale + 1.0e-12;
      const double inside_tol = 1.0e-8;

      Vector x_current(2), residual(2), delta(2);
      DenseMatrix F, cofactor, invF, Jxi(2), jacobian(2);
      PointLocation best_elem_location;
      for (const IntegrationPoint &seed : seeds)
      {
         IntegrationPoint ip = seed;
         bool finite_iterate = true;
         for (int iter = 0; iter < 15; ++iter)
         {
            double J = 1.0;
            EvaluateAleMap2D(*Tr, ip, ale_displacement_, x_current, F, cofactor, invF, J);
            residual = x_current;
            residual -= x_target;

            if (residual.Norml2() <= residual_tol &&
                ReferencePointInside2D(geom, ip, inside_tol))
            {
               break;
            }

            const DenseMatrix &Jref = Tr->Jacobian();
            Jxi(0, 0) = Jref(0, 0);
            Jxi(0, 1) = Jref(0, 1);
            Jxi(1, 0) = Jref(1, 0);
            Jxi(1, 1) = Jref(1, 1);
            jacobian(0, 0) = F(0, 0) * Jxi(0, 0) + F(0, 1) * Jxi(1, 0);
            jacobian(0, 1) = F(0, 0) * Jxi(0, 1) + F(0, 1) * Jxi(1, 1);
            jacobian(1, 0) = F(1, 0) * Jxi(0, 0) + F(1, 1) * Jxi(1, 0);
            jacobian(1, 1) = F(1, 0) * Jxi(0, 1) + F(1, 1) * Jxi(1, 1);
            if (!Solve2x2(jacobian, residual, delta))
            {
               finite_iterate = false;
               break;
            }

            ip.x -= delta[0];
            ip.y -= delta[1];
            if (!std::isfinite(ip.x) || !std::isfinite(ip.y))
            {
               finite_iterate = false;
               break;
            }
            if (delta.Norml2() <= 1.0e-12)
            {
               break;
            }
         }

         if (!finite_iterate)
         {
            continue;
         }

         double J = 1.0;
         EvaluateAleMap2D(*Tr, ip, ale_displacement_, x_current, F, cofactor, invF, J);
         residual = x_current;
         residual -= x_target;
         const double residual_norm = residual.Norml2();
         if (residual_norm > residual_tol ||
             !ReferencePointInside2D(geom, ip, inside_tol))
         {
            continue;
         }

         if (best_elem_location.elem < 0 ||
             residual_norm < best_elem_location.residual_norm)
         {
            best_elem_location.elem = elem;
            best_elem_location.ip = ip;
            best_elem_location.residual_norm = residual_norm;
         }
      }

      if (best_elem_location.elem >= 0 &&
          (best_location.elem < 0 ||
           best_elem_location.residual_norm < best_location.residual_norm))
      {
         best_location = best_elem_location;
         return true;
      }
      return false;
   }

   bool TryCandidateElements_(const vector<int> &candidate_elems,
                              const Vector &x_target,
                              PointLocation &best_location) const
   {
      bool found = false;
      for (const int elem : candidate_elems)
      {
         found = TryLocateInElement_(elem, x_target, best_location) || found;
      }
      return found;
   }

   ParMesh &pmesh_;
   const ParFiniteElementSpace &fes_state_;
   const ParGridFunction *ale_displacement_ = nullptr;
   vector<ElementBBox> element_bboxes_;
};

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
   bool disable_bprime_c = false;
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
   double dvalue_dBprime = 0.0;
};

struct SurfaceBlowingState
{
   double BprimeG = 0.0;
   double BprimeC = 0.0;
   double dBprimeG_dmdot = 0.0;
   double dBprimeC_dmdot = 0.0;
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
   double m_dot_g_centerline = 0.0;
   double gradp_n_centerline = 0.0;
   double rho_g_centerline = 0.0;
   double mu_g_centerline = 0.0;
   double K_centerline = 0.0;
   double mobility_centerline = 0.0;
   double m_dot_c_surf = 0.0;
   double rho_s_surf = 0.0;
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

BlowingCorrectionValue ComputeBlowingCorrectionWithDerivative(const double Bprime,
                                                              const double lambda)
{
   BlowingCorrectionValue out;

   const double lam = std::max(lambda, 0.0);
   const double Bprime_pos = std::max(Bprime, 0.0);
   const double arg = 2.0 * lam * Bprime_pos;
   if (arg < 1.0e-10)
   {
      const double arg2 = arg * arg;
      out.value = 1.0 - 0.5 * arg + (1.0 / 3.0) * arg2;
      if (Bprime_pos > 0.0)
      {
         const double df_darg = -0.5 + (2.0 / 3.0) * arg;
         out.dvalue_dBprime = df_darg * (2.0 * lam);
      }
      return out;
   }

   out.value = std::log1p(arg) / arg;
   if (Bprime_pos > 0.0)
   {
      const double df_darg = (arg / (1.0 + arg) - std::log1p(arg)) / (arg * arg);
      out.dvalue_dBprime = df_darg * (2.0 * lam);
   }
   return out;
}

double ComputeBlowingCorrection(const double Bprime, const double lambda)
{
   return ComputeBlowingCorrectionWithDerivative(Bprime, lambda).value;
}

SurfaceBlowingState SolveSurfaceBlowingState(const double m_dot_g_w,
                                             const double rhoeUeCH,
                                             const double p_w,
                                             const double T_w,
                                             const BPrimeTable &bprime_table,
                                             const bool chemistry_on,
                                             const bool disable_bprime_c,
                                             const double lambda,
                                             const bool enable_blowing)
{
   SurfaceBlowingState out;
   if (!enable_blowing)
   {
      return out;
   }

   const double rhoeUeCH_eff = std::max(rhoeUeCH, 1.0e-12);
   if (rhoeUeCH <= 1.0e-12) { out.nonsmooth = true; }

   double prev_blowing = out.blowing;
   for (int it = 0; it < 8; ++it)
   {
      const double blowing_eff = std::max(out.blowing, 1.0e-12);
      const bool blowing_clamped = (out.blowing <= 1.0e-12);
      const double dbeff_dm = blowing_clamped ? 0.0 : out.dblowing_dmdot;
      if (blowing_clamped) { out.nonsmooth = true; }

      const double denom = rhoeUeCH_eff * blowing_eff;
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

      if (chemistry_on && !disable_bprime_c)
      {
         const BPrimeTable::LookupDerivatives lookup =
            bprime_table.LookupWithDerivatives(p_w, out.BprimeG, T_w);
         out.BprimeC = lookup.bc;
         out.dBprimeC_dmdot = lookup.dbc_dbg * out.dBprimeG_dmdot;
         out.nonsmooth = out.nonsmooth || lookup.clamped_bg || lookup.clamped_t ||
                         lookup.nonsmooth_bg;
      }
      else
      {
         out.BprimeC = 0.0;
         out.dBprimeC_dmdot = 0.0;
      }

      const double Bprime_total = out.BprimeG + out.BprimeC;
      const BlowingCorrectionValue corr =
         ComputeBlowingCorrectionWithDerivative(Bprime_total, lambda);
      out.blowing = corr.value;
      out.dblowing_dmdot =
         corr.dvalue_dBprime * (out.dBprimeG_dmdot + out.dBprimeC_dmdot);

      const double abs_change = std::abs(out.blowing - prev_blowing);
      const double rel_scale = std::max(1.0, std::abs(prev_blowing));
      if (abs_change <= 1.0e-12 * rel_scale) { break; }
      prev_blowing = out.blowing;
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
      SolveSurfaceBlowingState(m_dot_g_w,
                               rhoeUeCH,
                               bc_state.p_w,
                               T_w,
                               bprime_table,
                               chemistry_on,
                               model.disable_bprime_c,
                               model.lambda,
                               blowing_active);

   const BPrimeTable::LookupDerivatives lookup =
      bprime_table.LookupWithDerivatives(bc_state.p_w, blowing.BprimeG, T_w);
   out.nonsmooth = out.nonsmooth || blowing.nonsmooth ||
                   lookup.clamped_bg || lookup.clamped_t || lookup.nonsmooth_bg;
   const double h_w = chemistry_on ? lookup.hw : 0.0;
   const double dh_w_dT = chemistry_on ? lookup.dhw_dT : 0.0;
   const double dh_w_dmdot =
      chemistry_on ? (lookup.dhw_dbg * blowing.dBprimeG_dmdot) : 0.0;

   out.terms.BprimeG = blowing.BprimeG;
   out.terms.BprimeC = (chemistry_on && !model.disable_bprime_c) ? lookup.bc : 0.0;
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
      pi_qp_.assign(ne, {});
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
         pi_qp_[e].assign(ir.GetNPoints(), 0.0);
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

   double GetQPointPiTotal(const int elem, const int qp) const
   {
      return pi_qp_.at(elem).at(qp);
   }

   void SetQPointPiTotal(const int elem, const int qp, const double pi)
   {
      pi_qp_.at(elem).at(qp) = pi;
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
   // Recompute extent_elem_[r][elem] as the average of the per-QP extents.
   // Called after ALE remapping to keep the diagnostic/output arrays consistent
   // with the authoritative per-QP states_.
   void UpdateExtentAverageFromQPs(const int elem)
   {
      const int nq = static_cast<int>(states_.at(elem).size());
      if (nq == 0) { return; }
      const int nr = static_cast<int>(extent_elem_.size());
      const double inv_nq = 1.0 / static_cast<double>(nq);
      for (int r = 0; r < nr; ++r)
      {
         double sum = 0.0;
         for (int q = 0; q < nq; ++q)
         {
            if (r < static_cast<int>(states_[elem][q].extent.size()))
            {
               sum += states_[elem][q].extent[r];
            }
         }
         extent_elem_[r].at(elem) = sum * inv_nq;
      }
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
      pi_qp_.assign(states_.size(), {});
      for (std::size_t e = 0; e < states_.size(); ++e)
      {
         pi_qp_[e].assign(states_[e].size(), 0.0);
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
   vector<vector<double>> pi_qp_;
   vector<double> tau_elem_;
   vector<double> rho_elem_;
   vector<double> pi_elem_;
   vector<double> mdot_elem_;
   vector<vector<double>> extent_elem_;
   vector<double> degree_char_elem_;
   vector<double> char_density_fraction_elem_;
};

int FindNearestIntegrationPoint(const IntegrationRule &ir,
                                const IntegrationPoint &ip)
{
   MFEM_VERIFY(ir.GetNPoints() > 0,
               "Nearest-integration-point lookup requires a non-empty rule.");

   int nearest_q = 0;
   double min_d2 = numeric_limits<double>::max();
   for (int q = 0; q < ir.GetNPoints(); ++q)
   {
      const IntegrationPoint &iq = ir.IntPoint(q);
      const double dx = ip.x - iq.x;
      const double dy = ip.y - iq.y;
      const double dz = ip.z - iq.z;
      const double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < min_d2)
      {
         min_d2 = d2;
         nearest_q = q;
      }
   }
   return nearest_q;
}

int FindNearestVolumeQuadraturePoint(const IntegrationRule &ir,
                                     const ReactionStateManager &state_manager,
                                     const int elem,
                                     const IntegrationPoint &ip,
                                     const char *context)
{
   MFEM_VERIFY(ir.GetNPoints() == state_manager.NumQPoints(elem),
               string(context) + ": quadrature mismatch while locating face state.");
   return FindNearestIntegrationPoint(ir, ip);
}

static void ProjectElementQPointExtentsToL2Coefficients(
   const ReactionStateManager &state_manager,
   const FiniteElement &fe_state,
   const FiniteElement &fe_l2,
   ElementTransformation &Tr_state,
   const int elem,
   const int quad_order,
   const char *context,
   vector<Vector> &extent_coeffs)
{
   const IntegrationRule &ir = IntRules.Get(fe_state.GetGeomType(), quad_order);
   MFEM_VERIFY(ir.GetNPoints() == state_manager.NumQPoints(elem),
               string(context) + ": quadrature mismatch during face-state projection.");

   const int nr = state_manager.NumReactions();
   const int ndof = fe_l2.GetDof();
   DenseMatrix mass(ndof);
   mass = 0.0;

   Vector shape(ndof);
   vector<Vector> rhs(static_cast<size_t>(nr));
   for (auto &rhs_r : rhs)
   {
      rhs_r.SetSize(ndof);
      rhs_r = 0.0;
   }

   for (int q = 0; q < ir.GetNPoints(); ++q)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Tr_state.SetIntPoint(&ip);
      fe_l2.CalcShape(ip, shape);
      const double weight = ip.weight * Tr_state.Weight();

      for (int i = 0; i < ndof; ++i)
      {
         for (int j = 0; j < ndof; ++j)
         {
            mass(i, j) += weight * shape(i) * shape(j);
         }
      }

      const TACOTMaterial::InternalState &st = state_manager.GetState(elem, q);
      for (int r = 0; r < nr; ++r)
      {
         const double xi =
            (r < static_cast<int>(st.extent.size())) ? st.extent[r] : 0.0;
         for (int i = 0; i < ndof; ++i)
         {
            rhs[static_cast<size_t>(r)](i) += weight * shape(i) * xi;
         }
      }
   }

   extent_coeffs.assign(static_cast<size_t>(nr), Vector());
   DenseMatrixInverse mass_inv(mass);
   for (int r = 0; r < nr; ++r)
   {
      extent_coeffs[static_cast<size_t>(r)].SetSize(ndof);
      mass_inv.Mult(rhs[static_cast<size_t>(r)],
                    extent_coeffs[static_cast<size_t>(r)]);
   }
}

class ElementFaceStateReconstruction
{
public:
   explicit ElementFaceStateReconstruction(const int dim)
      : l2_fec_(1, dim)
   {}

   void Build(const ReactionStateManager &state_manager,
              const FiniteElement &fe_state,
              ElementTransformation &Tr_state,
              const int elem,
              const int quad_order,
              const char *context)
   {
      fe_l2_ = l2_fec_.FiniteElementForGeometry(fe_state.GetGeomType());
      MFEM_VERIFY(fe_l2_ != nullptr,
                  string(context) + ": missing L2 element for face-state reconstruction.");
      ProjectElementQPointExtentsToL2Coefficients(state_manager,
                                                  fe_state,
                                                  *fe_l2_,
                                                  Tr_state,
                                                  elem,
                                                  quad_order,
                                                  context,
                                                  extent_coeffs_);
      state_.extent.assign(static_cast<size_t>(state_manager.NumReactions()), 0.0);
      state_.extent_old.assign(static_cast<size_t>(state_manager.NumReactions()), 0.0);
      state_.dt = 0.0;
      shape_.SetSize(fe_l2_->GetDof());
   }

   const TACOTMaterial::InternalState &Evaluate(const IntegrationPoint &ip)
   {
      MFEM_VERIFY(fe_l2_ != nullptr,
                  "Face-state reconstruction used before Build().");
      fe_l2_->CalcShape(ip, shape_);
      for (int r = 0; r < static_cast<int>(extent_coeffs_.size()); ++r)
      {
         const double xi = max(0.0,
                               min(1.0,
                                   shape_ * extent_coeffs_[static_cast<size_t>(r)]));
         state_.extent[static_cast<size_t>(r)] = xi;
         state_.extent_old[static_cast<size_t>(r)] = xi;
      }
      return state_;
   }

private:
   L2_FECollection l2_fec_;
   const FiniteElement *fe_l2_ = nullptr;
   vector<Vector> extent_coeffs_;
   Vector shape_;
   TACOTMaterial::InternalState state_;
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
                        const bool ale_mass_enabled,
                        const bool ale_energy_solid_enabled,
                        const bool ale_energy_gas_enabled,
                        const JacobianCheckOptions &jac_check)
      : material_(material),
        state_manager_(state_manager),
        T_old_coeff_(&T_old),
        p_old_coeff_(&p_old),
        quad_order_(quad_order),
        gravity_(gravity),
        ale_mass_enabled_(ale_mass_enabled),
        ale_energy_solid_enabled_(ale_energy_solid_enabled),
        ale_energy_gas_enabled_(ale_energy_gas_enabled),
        jac_check_(jac_check)
   {
      dt_ = 1.0;
   }

   void SetTimeStep(const double dt) { dt_ = dt; }

   void SetAleFields(const ParGridFunction *ale_displacement_old,
                     const ParGridFunction *ale_displacement,
                     const ParGridFunction *mesh_velocity)
   {
      ale_displacement_old_gf_ = ale_displacement_old;
      ale_displacement_gf_ = ale_displacement;
      mesh_velocity_gf_ = mesh_velocity;
   }

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
   struct AleGeometry
   {
      double J_old = 1.0;
      double J_new = 1.0;
      double divFw = 0.0;
      DenseMatrix metric;
      Vector Fg;
      Vector Fw;
   };

   struct MassStorageData
   {
      double value = 0.0;
      double dT = 0.0;
      double dp = 0.0;
      double dTT = 0.0;
      double dTp = 0.0;
      double dpp = 0.0;
   };

   struct AleStorageEvaluation
   {
      MassStorageData mass;
      MassStorageData solid_heatcap;
      MassStorageData gas_energy;
   };

   struct MaterialPointEvaluation
   {
      TACOTMaterial::InternalState state;
      TACOTMaterial::SolidProperties solid;
      TACOTMaterial::GasProperties gas;
   };

   struct MaterialDerivativeEvaluation
   {
      vector<double> dextent_dT;
      vector<double> d2extent_dT2;
      TACOTMaterial::SolidBulkDerivatives solid_bulk;
      TACOTMaterial::GasSurfaceDerivatives gas_surface;
   };

   struct QPCoeffs
   {
      double A = 0.0; // storage_p - source_p
      double B = 0.0; // rho_darcy
      double C = 0.0; // rho2_darcy
      double D = 0.0; // solid_storage + gas_storage - pyrolysis_heat_sink
      double E = 0.0; // solid.k
      double F = 0.0; // h_rho_darcy
      double G = 0.0; // h_rho2_darcy
      double H = 0.0; // ALE mass transport storage (eps_g*rho)
      double I = 0.0; // ALE energy transport storage
      double M_T = 0.0; // d(eps_g*rho_g)/dT
      double M_p = 0.0; // d(eps_g*rho_g)/dp
      double solid_heatcap = 0.0; // rho_s * c_ps
      double gas_energy_dT = 0.0; // d(phi(rho_g h_g - p))/dT
      double gas_energy_dp = 0.0; // d(phi(rho_g h_g - p))/dp
   };

   struct QPCoeffDerivatives
   {
      double A_T = 0.0;
      double A_p = 0.0;
      double B_T = 0.0;
      double B_p = 0.0;
      double C_T = 0.0;
      double C_p = 0.0;
      double D_T = 0.0;
      double D_p = 0.0;
      double E_T = 0.0;
      double E_p = 0.0;
      double F_T = 0.0;
      double F_p = 0.0;
      double G_T = 0.0;
      double G_p = 0.0;
      double H_T = 0.0;
      double H_p = 0.0;
      double I_T = 0.0;
      double I_p = 0.0;
      double solid_heatcap_T = 0.0;
      double solid_heatcap_p = 0.0;
      double gas_energy_T = 0.0;
      double gas_energy_p = 0.0;
   };

   bool UseReferenceAleGeometry() const
   {
      return (ale_displacement_gf_ != nullptr);
   }

   bool UseExactAleEnergyForm() const
   {
      return (UseReferenceAleGeometry() &&
              ale_energy_solid_enabled_ &&
              ale_energy_gas_enabled_);
   }

   static void BuildDeformationMap2D(const DenseMatrix &grad_disp,
                                     DenseMatrix &cofactor,
                                     double &J)
   {
      MFEM_VERIFY(grad_disp.Height() == 2 && grad_disp.Width() == 2,
                  "BuildDeformationMap2D expects a 2x2 displacement gradient.");

      const double g00 = 1.0 + grad_disp(0, 0);
      const double g01 = grad_disp(0, 1);
      const double g10 = grad_disp(1, 0);
      const double g11 = 1.0 + grad_disp(1, 1);

      J = g00 * g11 - g01 * g10;
      cofactor.SetSize(2, 2);
      cofactor(0, 0) = g11;
      cofactor(0, 1) = -g01;
      cofactor(1, 0) = -g10;
      cofactor(1, 1) = g00;
   }

   double RowDot(const DenseMatrix &dshape,
                 const int i,
                 const Vector &vec) const
   {
      double out = 0.0;
      for (int d = 0; d < vec.Size(); ++d)
      {
         out += dshape(i, d) * vec[d];
      }
      return out;
   }

   double VectorDot(const Vector &a,
                    const Vector &b) const
   {
      MFEM_VERIFY(a.Size() == b.Size(), "VectorDot size mismatch.");
      double out = 0.0;
      for (int i = 0; i < a.Size(); ++i)
      {
         out += a[i] * b[i];
      }
      return out;
   }

   double MetricDot(const DenseMatrix &metric,
                    const DenseMatrix &dshape,
                    const int i,
                    const Vector &grad) const
   {
      double out = 0.0;
      for (int a = 0; a < metric.Height(); ++a)
      {
         for (int b = 0; b < metric.Width(); ++b)
         {
            out += dshape(i, a) * metric(a, b) * grad[b];
         }
      }
      return out;
   }

   double MetricEntry(const DenseMatrix &metric,
                      const DenseMatrix &dshape_a,
                      const int i,
                      const DenseMatrix &dshape_b,
                      const int j) const
   {
      double out = 0.0;
      for (int a = 0; a < metric.Height(); ++a)
      {
         for (int b = 0; b < metric.Width(); ++b)
         {
            out += dshape_a(i, a) * metric(a, b) * dshape_b(j, b);
         }
      }
      return out;
   }

   void EvaluateAleGeometry(ElementTransformation &Tr,
                            const IntegrationPoint &ip,
                            AleGeometry &geom) const
   {
      const int dim = Tr.GetSpaceDim();
      MFEM_VERIFY(dim == 2, "AblationTPIntegrator ALE geometry expects 2D.");

      geom.metric.SetSize(dim, dim);
      geom.metric = 0.0;
      geom.Fg.SetSize(dim);
      geom.Fg = 0.0;
      geom.Fw.SetSize(dim);
      geom.Fw = 0.0;
      geom.divFw = 0.0;
      geom.J_old = 1.0;
      geom.J_new = 1.0;

      if (!UseReferenceAleGeometry())
      {
         for (int d = 0; d < dim; ++d)
         {
            geom.metric(d, d) = 1.0;
            geom.Fg[d] = gravity_[d];
         }
         if (mesh_velocity_gf_)
         {
            mesh_velocity_q_.SetSize(dim);
            mesh_velocity_gf_->GetVectorValue(Tr, ip, mesh_velocity_q_);
            geom.Fw = mesh_velocity_q_;

            DenseMatrix grad_w;
            mesh_velocity_gf_->GetVectorGradient(Tr, grad_w);
            for (int d = 0; d < dim; ++d)
            {
               geom.divFw += grad_w(d, d);
            }
         }
         return;
      }

      DenseMatrix grad_new;
      DenseMatrix cofactor_new;
      Tr.SetIntPoint(&ip);
      ale_displacement_gf_->GetVectorGradient(Tr, grad_new);
      BuildDeformationMap2D(grad_new, cofactor_new, geom.J_new);
      if (!(geom.J_new > 1.0e-12))
      {
         throw runtime_error("Degenerate ALE Jacobian in AblationTPIntegrator.");
      }

      if (ale_displacement_old_gf_)
      {
         DenseMatrix grad_old;
         DenseMatrix cofactor_old;
         ale_displacement_old_gf_->GetVectorGradient(Tr, grad_old);
         BuildDeformationMap2D(grad_old, cofactor_old, geom.J_old);
         if (!(geom.J_old > 1.0e-12))
         {
            throw runtime_error("Degenerate old ALE Jacobian in AblationTPIntegrator.");
         }
      }

      MultABt(cofactor_new, cofactor_new, geom.metric);
      geom.metric *= (1.0 / geom.J_new);

      cofactor_new.Mult(gravity_, geom.Fg);

      if (mesh_velocity_gf_)
      {
         mesh_velocity_q_.SetSize(dim);
         mesh_velocity_gf_->GetVectorValue(Tr, ip, mesh_velocity_q_);
         cofactor_new.Mult(mesh_velocity_q_, geom.Fw);

         DenseMatrix grad_w;
         mesh_velocity_gf_->GetVectorGradient(Tr, grad_w);
         for (int a = 0; a < dim; ++a)
         {
            for (int b = 0; b < dim; ++b)
            {
               geom.divFw += cofactor_new(a, b) * grad_w(b, a);
            }
         }
      }
   }

   MaterialDerivativeEvaluation EvaluateMaterialDerivatives(
      const double T,
      const double p,
      const TACOTMaterial::InternalState &new_state) const
   {
      MaterialDerivativeEvaluation out;
      out.dextent_dT = material_.EvaluateExtentTemperatureDerivative(T, new_state);
      out.d2extent_dT2 =
         material_.EvaluateExtentTemperatureSecondDerivative(
            T, new_state, out.dextent_dT);
      out.solid_bulk =
         material_.EvaluateSolidBulkDerivatives(
            T, p, new_state, out.dextent_dT, out.d2extent_dT2);
      out.gas_surface =
         material_.EvaluateGasSurfaceDerivatives(T, p, new_state);
      return out;
   }

   AleStorageEvaluation EvaluateAleStorageData(
      const double T,
      const double p,
      const TACOTMaterial::InternalState &old_state,
      const TACOTMaterial::InternalState &new_state,
      const TACOTMaterial::SolidProperties &solid,
      const TACOTMaterial::GasProperties &gas,
      const MaterialDerivativeEvaluation *material_derivs = nullptr) const
   {
      (void)old_state;
      AleStorageEvaluation out;
      out.mass.value = solid.eps_g * gas.rho;
      out.solid_heatcap.value = solid.rho_s * solid.cp;
      out.gas_energy.value = solid.eps_g * (gas.rho * gas.h - p);

      const bool need_mass =
         (ale_mass_enabled_ && UseReferenceAleGeometry());
      const bool need_energy = UseExactAleEnergyForm();
      if (!need_mass && !need_energy)
      {
         return out;
      }

      MaterialDerivativeEvaluation local_derivs;
      if (!material_derivs)
      {
         local_derivs = EvaluateMaterialDerivatives(T, p, new_state);
         material_derivs = &local_derivs;
      }
      const TACOTMaterial::SolidBulkDerivatives &solid_bulk_deriv =
         material_derivs->solid_bulk;
      const TACOTMaterial::GasSurfaceDerivatives &gas_deriv =
         material_derivs->gas_surface;

      if (need_mass)
      {
         out.mass.dT =
            solid_bulk_deriv.eps_g.dT * gas.rho +
            solid.eps_g * gas_deriv.rho.dT;
         out.mass.dp =
            solid_bulk_deriv.eps_g.dp * gas.rho +
            solid.eps_g * gas_deriv.rho.dp;
         out.mass.dTT =
            solid_bulk_deriv.eps_g.dTT * gas.rho +
            2.0 * solid_bulk_deriv.eps_g.dT * gas_deriv.rho.dT +
            solid.eps_g * gas_deriv.rho.dTT;
         out.mass.dTp =
            solid_bulk_deriv.eps_g.dTp * gas.rho +
            solid_bulk_deriv.eps_g.dT * gas_deriv.rho.dp +
            solid_bulk_deriv.eps_g.dp * gas_deriv.rho.dT +
            solid.eps_g * gas_deriv.rho.dTp;
         out.mass.dpp =
            solid_bulk_deriv.eps_g.dpp * gas.rho +
            2.0 * solid_bulk_deriv.eps_g.dp * gas_deriv.rho.dp +
            solid.eps_g * gas_deriv.rho.dpp;
      }

      if (need_energy)
      {
         out.solid_heatcap.dT =
            solid_bulk_deriv.rho_s.dT * solid.cp +
            solid.rho_s * solid_bulk_deriv.cp.dT;
         out.solid_heatcap.dp =
            solid_bulk_deriv.rho_s.dp * solid.cp +
            solid.rho_s * solid_bulk_deriv.cp.dp;
         out.solid_heatcap.dTT =
            solid_bulk_deriv.rho_s.dTT * solid.cp +
            2.0 * solid_bulk_deriv.rho_s.dT * solid_bulk_deriv.cp.dT +
            solid.rho_s * solid_bulk_deriv.cp.dTT;
         out.solid_heatcap.dTp =
            solid_bulk_deriv.rho_s.dTp * solid.cp +
            solid_bulk_deriv.rho_s.dT * solid_bulk_deriv.cp.dp +
            solid_bulk_deriv.rho_s.dp * solid_bulk_deriv.cp.dT +
            solid.rho_s * solid_bulk_deriv.cp.dTp;
         out.solid_heatcap.dpp =
            solid_bulk_deriv.rho_s.dpp * solid.cp +
            2.0 * solid_bulk_deriv.rho_s.dp * solid_bulk_deriv.cp.dp +
            solid.rho_s * solid_bulk_deriv.cp.dpp;

         const double gas_enthalpy_density = gas.rho * gas.h - p;
         const double gas_enthalpy_density_dT =
            gas.h * gas_deriv.rho.dT + gas.rho * gas_deriv.h.dT;
         const double gas_enthalpy_density_dp =
            gas.h * gas_deriv.rho.dp + gas.rho * gas_deriv.h.dp - 1.0;
         const double gas_enthalpy_density_dTT =
            gas_deriv.rho.dTT * gas.h +
            2.0 * gas_deriv.rho.dT * gas_deriv.h.dT +
            gas.rho * gas_deriv.h.dTT;
         const double gas_enthalpy_density_dTp =
            gas_deriv.rho.dTp * gas.h +
            gas_deriv.rho.dT * gas_deriv.h.dp +
            gas_deriv.rho.dp * gas_deriv.h.dT +
            gas.rho * gas_deriv.h.dTp;
         const double gas_enthalpy_density_dpp =
            gas_deriv.rho.dpp * gas.h +
            2.0 * gas_deriv.rho.dp * gas_deriv.h.dp +
            gas.rho * gas_deriv.h.dpp;

         out.gas_energy.dT =
            solid_bulk_deriv.eps_g.dT * gas_enthalpy_density +
            solid.eps_g * gas_enthalpy_density_dT;
         out.gas_energy.dp =
            solid_bulk_deriv.eps_g.dp * gas_enthalpy_density +
            solid.eps_g * gas_enthalpy_density_dp;
         out.gas_energy.dTT =
            solid_bulk_deriv.eps_g.dTT * gas_enthalpy_density +
            2.0 * solid_bulk_deriv.eps_g.dT * gas_enthalpy_density_dT +
            solid.eps_g * gas_enthalpy_density_dTT;
         out.gas_energy.dTp =
            solid_bulk_deriv.eps_g.dTp * gas_enthalpy_density +
            solid_bulk_deriv.eps_g.dT * gas_enthalpy_density_dp +
            solid_bulk_deriv.eps_g.dp * gas_enthalpy_density_dT +
            solid.eps_g * gas_enthalpy_density_dTp;
         out.gas_energy.dpp =
            solid_bulk_deriv.eps_g.dpp * gas_enthalpy_density +
            2.0 * solid_bulk_deriv.eps_g.dp * gas_enthalpy_density_dp +
            solid.eps_g * gas_enthalpy_density_dpp;
      }

      return out;
   }

   MaterialPointEvaluation EvaluateMaterialPoint(
      const double T,
      const double p,
      const TACOTMaterial::InternalState &old_state) const
   {
      MaterialPointEvaluation out;
      out.state = material_.SolveReactionExtents(T, dt_, old_state);
      out.solid = material_.EvaluateSolid(T, p, out.state);
      out.gas = material_.EvaluateGas(T, p, out.state);
      return out;
   }

   MaterialPointEvaluation EvaluateMaterialPointWithState(
      const double T,
      const double p,
      const TACOTMaterial::InternalState &state) const
   {
      MaterialPointEvaluation out;
      out.state = state;
      out.solid = material_.EvaluateSolid(T, p, out.state);
      out.gas = material_.EvaluateGas(T, p, out.state);
      return out;
   }

   QPCoeffs EvaluateQPCoeffs(const double T,
                             const double p,
                             const TACOTMaterial::InternalState &old_state,
                             const double T_old,
                             const double p_old,
                             const TACOTMaterial::SolidProperties &solid_old,
                             const TACOTMaterial::GasProperties &gas_old,
                             const AleGeometry &geom,
                             const MaterialPointEvaluation &eval,
                             const MaterialDerivativeEvaluation *material_derivs = nullptr) const
   {
      QPCoeffs out;
      const TACOTMaterial::InternalState &new_state = eval.state;
      const TACOTMaterial::SolidProperties &solid = eval.solid;
      const TACOTMaterial::GasProperties &gas = eval.gas;

      const double mu = max(gas.mu, 1.0e-12);
      const double darcy = solid.K / mu;
      const double rho_darcy = gas.rho * darcy;
      const double rho2_darcy = gas.rho * rho_darcy;
      const double h_rho_darcy = gas.h * rho_darcy;
      const double h_rho2_darcy = gas.h * rho2_darcy;

      const AleStorageEvaluation ale_storage =
         EvaluateAleStorageData(T, p, old_state, new_state, solid, gas, material_derivs);
      const MassStorageData &mass_storage = ale_storage.mass;
      const MassStorageData &solid_heatcap_data = ale_storage.solid_heatcap;
      const MassStorageData &gas_energy_data = ale_storage.gas_energy;
      const double e_m_new = mass_storage.value;
      const double e_m_old = solid_old.eps_g * gas_old.rho;
      const double solid_heatcap = solid_heatcap_data.value;
      const double e_g_new = gas_energy_data.value;
      const double e_g_old = solid_old.eps_g * (gas_old.rho * gas_old.h - p_old);

      double storage_p = (e_m_new - e_m_old) / dt_;
      double solid_storage = solid_heatcap * ((T - T_old) / dt_);
      double gas_storage = (e_g_new - e_g_old) / dt_;
      double source_p = solid.pi_total;
      double pyro_sink = solid.pyrolysis_heat_sink;

      if (UseReferenceAleGeometry())
      {
         if (ale_mass_enabled_)
         {
            storage_p = (geom.J_new * e_m_new - geom.J_old * e_m_old) / dt_;
         }
         else
         {
            storage_p = geom.J_new * (e_m_new - e_m_old) / dt_;
         }
         source_p = geom.J_new * solid.pi_total;

         if (ale_energy_solid_enabled_)
         {
            solid_storage = solid_heatcap * ((geom.J_new * T - geom.J_old * T_old) / dt_);
         }
         else
         {
            solid_storage = geom.J_new * solid_heatcap * ((T - T_old) / dt_);
         }

         if (ale_energy_gas_enabled_)
         {
            gas_storage = (geom.J_new * e_g_new - geom.J_old * e_g_old) / dt_;
         }
         else
         {
            gas_storage = geom.J_new * ((e_g_new - e_g_old) / dt_);
         }
         pyro_sink = geom.J_new * solid.pyrolysis_heat_sink;
      }

      const double ale_mass_storage = e_m_new;
      const double ale_energy_storage_solid = solid_heatcap * T;
      const double ale_energy_storage_gas = e_g_new;
      const double ale_energy_storage =
         (ale_energy_solid_enabled_ ? ale_energy_storage_solid : 0.0) +
         (ale_energy_gas_enabled_ ? ale_energy_storage_gas : 0.0);

      out.A = storage_p - source_p;
      out.B = rho_darcy;
      out.C = rho2_darcy;
      out.D = solid_storage + gas_storage - pyro_sink;
      out.E = solid.k;
      out.F = h_rho_darcy;
      out.G = h_rho2_darcy;
      out.H = ale_mass_storage;
      out.I = ale_energy_storage;
      out.M_T = mass_storage.dT;
      out.M_p = mass_storage.dp;
      out.solid_heatcap = solid_heatcap;
      out.gas_energy_dT = gas_energy_data.dT;
      out.gas_energy_dp = gas_energy_data.dp;
      return out;
   }

   QPCoeffDerivatives EvaluateQPCoeffDerivatives(
      const double T,
      const double p,
      const double T_old,
      const double T_old_weighted,
      const AleGeometry &geom,
      const MaterialPointEvaluation &eval,
      const MaterialDerivativeEvaluation &material_derivs,
      const AleStorageEvaluation &ale_storage) const
   {
      (void)p;
      (void)T_old;
      QPCoeffDerivatives out;
      const TACOTMaterial::SolidProperties &solid = eval.solid;
      const TACOTMaterial::GasProperties &gas = eval.gas;
      const TACOTMaterial::SolidBulkDerivatives &solid_deriv =
         material_derivs.solid_bulk;
      const TACOTMaterial::GasSurfaceDerivatives &gas_deriv =
         material_derivs.gas_surface;

      const double mu_eff = max(gas.mu, 1.0e-12);
      const double dmu_eff_dT = (gas.mu > 1.0e-12) ? gas_deriv.mu.dT : 0.0;
      const double dmu_eff_dp = (gas.mu > 1.0e-12) ? gas_deriv.mu.dp : 0.0;

      const double darcy = solid.K / mu_eff;
      const double darcy_T =
         solid_deriv.K.dT / mu_eff -
         solid.K / (mu_eff * mu_eff) * dmu_eff_dT;
      const double darcy_p =
         solid_deriv.K.dp / mu_eff -
         solid.K / (mu_eff * mu_eff) * dmu_eff_dp;

      const double rho_darcy = gas.rho * darcy;
      out.B_T = gas_deriv.rho.dT * darcy + gas.rho * darcy_T;
      out.B_p = gas_deriv.rho.dp * darcy + gas.rho * darcy_p;

      const double rho2_darcy = gas.rho * rho_darcy;
      out.C_T = gas_deriv.rho.dT * rho_darcy + gas.rho * out.B_T;
      out.C_p = gas_deriv.rho.dp * rho_darcy + gas.rho * out.B_p;

      out.E_T = solid_deriv.k.dT;
      out.E_p = solid_deriv.k.dp;

      out.F_T = gas_deriv.h.dT * rho_darcy + gas.h * out.B_T;
      out.F_p = gas_deriv.h.dp * rho_darcy + gas.h * out.B_p;

      out.G_T = gas_deriv.h.dT * rho2_darcy + gas.h * out.C_T;
      out.G_p = gas_deriv.h.dp * rho2_darcy + gas.h * out.C_p;

      out.H_T = ale_storage.mass.dT;
      out.H_p = ale_storage.mass.dp;
      out.solid_heatcap_T = ale_storage.solid_heatcap.dT;
      out.solid_heatcap_p = ale_storage.solid_heatcap.dp;
      out.gas_energy_T = ale_storage.gas_energy.dT;
      out.gas_energy_p = ale_storage.gas_energy.dp;

      const double mass_storage_factor =
         UseReferenceAleGeometry() ? (geom.J_new / dt_) : (1.0 / dt_);
      const double source_factor =
         UseReferenceAleGeometry() ? geom.J_new : 1.0;
      out.A_T = mass_storage_factor * out.H_T - source_factor * solid_deriv.pi_total.dT;
      out.A_p = mass_storage_factor * out.H_p - source_factor * solid_deriv.pi_total.dp;

      const double solid_storage_factor =
         UseReferenceAleGeometry() ?
            (ale_energy_solid_enabled_ ?
               ((geom.J_new * T - geom.J_old * T_old_weighted) / dt_) :
               (geom.J_new * (T - T_old_weighted) / dt_)) :
            ((T - T_old_weighted) / dt_);
      const double solid_storage_direct_T =
         UseReferenceAleGeometry() ? (geom.J_new / dt_) : (1.0 / dt_);
      out.D_T = out.solid_heatcap_T * solid_storage_factor +
                ale_storage.solid_heatcap.value * solid_storage_direct_T;
      out.D_p = out.solid_heatcap_p * solid_storage_factor;

      const double gas_storage_factor =
         UseReferenceAleGeometry() ? (geom.J_new / dt_) : (1.0 / dt_);
      out.D_T += gas_storage_factor * out.gas_energy_T;
      out.D_p += gas_storage_factor * out.gas_energy_p;

      const double pyro_factor =
         UseReferenceAleGeometry() ? geom.J_new : 1.0;
      out.D_T -= pyro_factor * solid_deriv.pyrolysis_heat_sink.dT;
      out.D_p -= pyro_factor * solid_deriv.pyrolysis_heat_sink.dp;

      out.I_T = 0.0;
      out.I_p = 0.0;
      if (ale_energy_solid_enabled_)
      {
         out.I_T += out.solid_heatcap_T * T + ale_storage.solid_heatcap.value;
         out.I_p += out.solid_heatcap_p * T;
      }
      if (ale_energy_gas_enabled_)
      {
         out.I_T += out.gas_energy_T;
         out.I_p += out.gas_energy_p;
      }

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

         AleGeometry geom;
         EvaluateAleGeometry(Tr, ip, geom);
         const MaterialPointEvaluation base_eval =
            EvaluateMaterialPoint(T, p, old_state);
         const MaterialDerivativeEvaluation base_material_derivs =
            EvaluateMaterialDerivatives(T, p, base_eval.state);
         const AleStorageEvaluation base_ale_storage =
            EvaluateAleStorageData(T, p, old_state, base_eval.state,
                                   base_eval.solid, base_eval.gas,
                                   &base_material_derivs);
         const QPCoeffs base =
            EvaluateQPCoeffs(T, p, old_state, T_old, p_old, solid_old, gas_old,
                             geom, base_eval, &base_material_derivs);
         const QPCoeffDerivatives coeff_derivs =
            EvaluateQPCoeffDerivatives(T, p, T_old, T_old, geom,
                                       base_eval, base_material_derivs,
                                       base_ale_storage);

         const double A_T = coeff_derivs.A_T;
         const double B_T = coeff_derivs.B_T;
         const double C_T = coeff_derivs.C_T;
         const double D_T = coeff_derivs.D_T;
         const double E_T = coeff_derivs.E_T;
         const double F_T = coeff_derivs.F_T;
         const double G_T = coeff_derivs.G_T;
         const double H_T = coeff_derivs.H_T;
         const double I_T = coeff_derivs.I_T;
         const double solid_heatcap_T = coeff_derivs.solid_heatcap_T;

         const double A_p = coeff_derivs.A_p;
         const double B_p = coeff_derivs.B_p;
         const double C_p = coeff_derivs.C_p;
         const double D_p = coeff_derivs.D_p;
         const double E_p = coeff_derivs.E_p;
         const double F_p = coeff_derivs.F_p;
         const double G_p = coeff_derivs.G_p;
         const double H_p = coeff_derivs.H_p;
         const double I_p = coeff_derivs.I_p;
         const double solid_heatcap_p = coeff_derivs.solid_heatcap_p;

         const double w = ip.weight * Tr.Weight();
         const bool use_exact_ale_mass_form =
            (ale_mass_enabled_ && UseReferenceAleGeometry());
         const bool use_exact_ale_energy_form = UseExactAleEnergyForm();
         const double Fw_gradT = VectorDot(geom.Fw, gradT_);
         const double Fw_gradp = VectorDot(geom.Fw, gradp_);

         double M_T_T = 0.0;
         double M_p_T = 0.0;
         double M_T_p = 0.0;
         double M_p_p = 0.0;
         double gas_energy_dT_T = 0.0;
         double gas_energy_dp_T = 0.0;
         double gas_energy_dT_p = 0.0;
         double gas_energy_dp_p = 0.0;
         if (use_exact_ale_mass_form || use_exact_ale_energy_form)
         {
            M_T_T = base_ale_storage.mass.dTT;
            M_p_T = base_ale_storage.mass.dTp;
            M_T_p = base_ale_storage.mass.dTp;
            M_p_p = base_ale_storage.mass.dpp;
            gas_energy_dT_T = base_ale_storage.gas_energy.dTT;
            gas_energy_dp_T = base_ale_storage.gas_energy.dTp;
            gas_energy_dT_p = base_ale_storage.gas_energy.dTp;
            gas_energy_dp_p = base_ale_storage.gas_energy.dpp;
         }

         for (int i = 0; i < dof_p; ++i)
         {
            const double Bpi_metric_gradp = MetricDot(geom.metric, dshape_p_, i, gradp_);
            const double Fg_Bpi = RowDot(dshape_p_, i, geom.Fg);
            const double Fw_Bpi = RowDot(dshape_p_, i, geom.Fw);

            for (int j = 0; j < dof_T; ++j)
            {
               if (use_exact_ale_mass_form)
               {
                  const double Fw_BTj = RowDot(dshape_T_, j, geom.Fw);
                  const double exact_mass_T =
                     - H_T * shape_T_[j] * geom.divFw
                     - M_T_T * shape_T_[j] * Fw_gradT
                     - base.M_T * Fw_BTj
                     - M_p_T * shape_T_[j] * Fw_gradp;
                  J10(i, j) += w * (shape_p_[i] * (A_T * shape_T_[j] + exact_mass_T)
                                    + B_T * shape_T_[j] * Bpi_metric_gradp);
               }
               else
               {
                  J10(i, j) += w * (shape_p_[i] * A_T * shape_T_[j]
                                    + B_T * shape_T_[j] * Bpi_metric_gradp
                                    - C_T * shape_T_[j] * Fg_Bpi
                                    + (ale_mass_enabled_ ?
                                          (H_T * shape_T_[j] * Fw_Bpi) :
                                          0.0));
               }
            }

            for (int j = 0; j < dof_p; ++j)
            {
               const double Bpi_metric_Bpj =
                  MetricEntry(geom.metric, dshape_p_, i, dshape_p_, j);
               if (use_exact_ale_mass_form)
               {
                  const double Fw_Bpj = RowDot(dshape_p_, j, geom.Fw);
                  const double exact_mass_p =
                     - H_p * shape_p_[j] * geom.divFw
                     - M_T_p * shape_p_[j] * Fw_gradT
                     - M_p_p * shape_p_[j] * Fw_gradp
                     - base.M_p * Fw_Bpj;
                  J11(i, j) += w * (shape_p_[i] * (A_p * shape_p_[j] + exact_mass_p)
                                    + B_p * shape_p_[j] * Bpi_metric_gradp
                                    + base.B * Bpi_metric_Bpj);
               }
               else
               {
                  J11(i, j) += w * (shape_p_[i] * A_p * shape_p_[j]
                                    + B_p * shape_p_[j] * Bpi_metric_gradp
                                    + base.B * Bpi_metric_Bpj
                                    - C_p * shape_p_[j] * Fg_Bpi
                                    + (ale_mass_enabled_ ?
                                          (H_p * shape_p_[j] * Fw_Bpi) :
                                          0.0));
               }
            }
         }

         for (int i = 0; i < dof_T; ++i)
         {
            const double BTi_metric_gradT = MetricDot(geom.metric, dshape_T_, i, gradT_);
            const double BTi_metric_gradp = MetricDot(geom.metric, dshape_T_, i, gradp_);
            const double Fg_BTi = RowDot(dshape_T_, i, geom.Fg);
            const double Fw_BTi = RowDot(dshape_T_, i, geom.Fw);

            for (int j = 0; j < dof_T; ++j)
            {
               const double BTi_metric_BTj =
                  MetricEntry(geom.metric, dshape_T_, i, dshape_T_, j);
               if (use_exact_ale_energy_form)
               {
                  const double Fw_BTj = RowDot(dshape_T_, j, geom.Fw);
                  const double solid_adv_T =
                     -(solid_heatcap_T * shape_T_[j] * (T * geom.divFw + Fw_gradT)
                       + base.solid_heatcap *
                            (shape_T_[j] * geom.divFw + Fw_BTj));
                  const double gas_adv_T =
                     -(base.gas_energy_dT * shape_T_[j] * geom.divFw
                       + gas_energy_dT_T * shape_T_[j] * Fw_gradT
                       + base.gas_energy_dT * Fw_BTj
                       + gas_energy_dp_T * shape_T_[j] * Fw_gradp);
                  J00(i, j) += w * (shape_T_[i] *
                                       (D_T * shape_T_[j] + solid_adv_T + gas_adv_T)
                                    + E_T * shape_T_[j] * BTi_metric_gradT
                                    + base.E * BTi_metric_BTj
                                    + F_T * shape_T_[j] * BTi_metric_gradp);
               }
               else
               {
                  J00(i, j) += w * (shape_T_[i] * D_T * shape_T_[j]
                                    + E_T * shape_T_[j] * BTi_metric_gradT
                                    + base.E * BTi_metric_BTj
                                    + F_T * shape_T_[j] * BTi_metric_gradp
                                    - G_T * shape_T_[j] * Fg_BTi
                                    + ((ale_energy_solid_enabled_ || ale_energy_gas_enabled_)
                                          ? (I_T * shape_T_[j] * Fw_BTi)
                                          : 0.0));
               }
            }

            for (int j = 0; j < dof_p; ++j)
            {
               const double BTi_metric_Bpj =
                  MetricEntry(geom.metric, dshape_T_, i, dshape_p_, j);
               if (use_exact_ale_energy_form)
               {
                  const double Fw_Bpj = RowDot(dshape_p_, j, geom.Fw);
                  const double solid_adv_p =
                     -(solid_heatcap_p * shape_p_[j] * (T * geom.divFw + Fw_gradT));
                  const double gas_adv_p =
                     -(base.gas_energy_dp * shape_p_[j] * geom.divFw
                       + gas_energy_dT_p * shape_p_[j] * Fw_gradT
                       + gas_energy_dp_p * shape_p_[j] * Fw_gradp
                       + base.gas_energy_dp * Fw_Bpj);
                  J01(i, j) += w * (shape_T_[i] *
                                       (D_p * shape_p_[j] + solid_adv_p + gas_adv_p)
                                    + E_p * shape_p_[j] * BTi_metric_gradT
                                    + F_p * shape_p_[j] * BTi_metric_gradp
                                    + base.F * BTi_metric_Bpj);
               }
               else
               {
                  J01(i, j) += w * (shape_T_[i] * D_p * shape_p_[j]
                                    + E_p * shape_p_[j] * BTi_metric_gradT
                                    + F_p * shape_p_[j] * BTi_metric_gradp
                                    + base.F * BTi_metric_Bpj
                                    - G_p * shape_p_[j] * Fg_BTi
                                    + ((ale_energy_solid_enabled_ || ale_energy_gas_enabled_)
                                          ? (I_p * shape_p_[j] * Fw_BTi)
                                          : 0.0));
               }
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

         AleGeometry geom;
         EvaluateAleGeometry(Tr, ip, geom);

         const AleStorageEvaluation ale_storage =
            EvaluateAleStorageData(T, p, old_state, new_state, solid, gas);
         const MassStorageData &mass_storage = ale_storage.mass;
         const MassStorageData &solid_heatcap_data = ale_storage.solid_heatcap;
         const MassStorageData &gas_energy_data = ale_storage.gas_energy;
         const double e_m_new = mass_storage.value;
         const double e_m_old = solid_old.eps_g * gas_old.rho;
         const double solid_heatcap = solid_heatcap_data.value;
         const double e_g_new = gas_energy_data.value;
         const double e_g_old = solid_old.eps_g * (gas_old.rho * gas_old.h - p_old);

         double storage_p = (e_m_new - e_m_old) / dt_;
         double solid_storage = solid_heatcap * ((T - T_old) / dt_);
         double gas_storage = (e_g_new - e_g_old) / dt_;
         double source_p = solid.pi_total;
         double pyro_sink = solid.pyrolysis_heat_sink;

         if (UseReferenceAleGeometry())
         {
            if (ale_mass_enabled_)
            {
               storage_p = (geom.J_new * e_m_new - geom.J_old * e_m_old) / dt_;
            }
            else
            {
               storage_p = geom.J_new * (e_m_new - e_m_old) / dt_;
            }
            source_p = geom.J_new * solid.pi_total;

            if (ale_energy_solid_enabled_)
            {
               solid_storage = solid_heatcap * ((geom.J_new * T - geom.J_old * T_old) / dt_);
            }
            else
            {
               solid_storage = geom.J_new * solid_heatcap * ((T - T_old) / dt_);
            }

            if (ale_energy_gas_enabled_)
            {
               gas_storage = (geom.J_new * e_g_new - geom.J_old * e_g_old) / dt_;
            }
            else
            {
               gas_storage = geom.J_new * ((e_g_new - e_g_old) / dt_);
            }
            pyro_sink = geom.J_new * solid.pyrolysis_heat_sink;
         }

         const double ale_mass_storage = e_m_new;
         const double ale_energy_storage_solid = solid_heatcap * T;
         const double ale_energy_storage_gas = e_g_new;
         const double ale_energy_storage =
            (ale_energy_solid_enabled_ ? ale_energy_storage_solid : 0.0) +
            (ale_energy_gas_enabled_ ? ale_energy_storage_gas : 0.0);

         const double w = ip.weight * Tr.Weight();
         const bool use_exact_ale_mass_form =
            (ale_mass_enabled_ && UseReferenceAleGeometry());
         const bool use_exact_ale_energy_form = UseExactAleEnergyForm();
         const double exact_ale_mass_storage =
            use_exact_ale_mass_form ?
               (e_m_new * geom.divFw +
                mass_storage.dT * VectorDot(geom.Fw, gradT_) +
                mass_storage.dp * VectorDot(geom.Fw, gradp_)) :
               0.0;
         const double exact_ale_solid_energy_transport =
            use_exact_ale_energy_form ?
               (solid_heatcap * (T * geom.divFw + VectorDot(geom.Fw, gradT_))) :
               0.0;
         const double exact_ale_gas_energy_transport =
            use_exact_ale_energy_form ?
               (e_g_new * geom.divFw +
                gas_energy_data.dT * VectorDot(geom.Fw, gradT_) +
                gas_energy_data.dp * VectorDot(geom.Fw, gradp_)) :
               0.0;

         for (int i = 0; i < dof_p; ++i)
         {
            const double metric_gradp = MetricDot(geom.metric, dshape_p_, i, gradp_);
            const double g_dot = RowDot(dshape_p_, i, geom.Fg);
            const double wmesh_dot = RowDot(dshape_p_, i, geom.Fw);

            const double ale_mass_term =
               use_exact_ale_mass_form ?
                  (-shape_p_[i] * exact_ale_mass_storage) :
                  (ale_mass_enabled_ ? (ale_mass_storage * wmesh_dot) : 0.0);
            rp[i] += w * (shape_p_[i] * (storage_p - source_p)
                          + rho_darcy * metric_gradp
                          - (use_exact_ale_mass_form ? 0.0 : rho2_darcy * g_dot)
                          + ale_mass_term);
         }

         for (int i = 0; i < dof_T; ++i)
         {
            const double gradT_dot = MetricDot(geom.metric, dshape_T_, i, gradT_);
            const double gradp_dot = MetricDot(geom.metric, dshape_T_, i, gradp_);
            const double g_dot = RowDot(dshape_T_, i, geom.Fg);
            const double wmesh_dot = RowDot(dshape_T_, i, geom.Fw);

            const double ale_energy_term =
               use_exact_ale_energy_form ?
                  (-shape_T_[i] *
                   (exact_ale_solid_energy_transport + exact_ale_gas_energy_transport)) :
                  ((ale_energy_solid_enabled_ || ale_energy_gas_enabled_) ?
                  (ale_energy_storage * wmesh_dot) :
                  0.0);
            rT[i] += w * (shape_T_[i] * (solid_storage + gas_storage - pyro_sink)
                          + solid.k * gradT_dot
                          + h_rho_darcy * gradp_dot
                          - (use_exact_ale_energy_form ? 0.0 : h_rho2_darcy * g_dot)
                          + ale_energy_term);
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
   bool ale_mass_enabled_ = false;
   bool ale_energy_solid_enabled_ = false;
   bool ale_energy_gas_enabled_ = false;
   JacobianCheckOptions jac_check_;
   bool jacobian_checked_ = false;

   const ParGridFunction *ale_displacement_old_gf_ = nullptr;
   const ParGridFunction *ale_displacement_gf_ = nullptr;
   const ParGridFunction *mesh_velocity_gf_ = nullptr;

   mutable Vector shape_T_;
   mutable Vector shape_p_;
   mutable DenseMatrix dshape_T_;
   mutable DenseMatrix dshape_p_;
   mutable Vector gradT_;
   mutable Vector gradp_;
   mutable Vector mesh_velocity_q_;
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
   void SetAleDisplacement(const ParGridFunction *ale_displacement)
   {
      ale_displacement_ = ale_displacement;
   }

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
   bool ComputeFaceTransportGeometry(FaceElementTransformations &Tr,
                                     const IntegrationPoint &fip,
                                     const IntegrationPoint &eip,
                                     const Vector &gradp_ref,
                                     Vector &gradp_cur,
                                     Vector &unit_normal,
                                     DenseMatrix &invF,
                                     double &ds) const
   {
      const int dim = gradp_ref.Size();
      gradp_cur.SetSize(dim);
      unit_normal.SetSize(dim);
      invF.SetSize(dim, dim);
      invF = 0.0;
      for (int d = 0; d < dim; ++d)
      {
         invF(d, d) = 1.0;
      }

      const bool use_ale_face_geometry = (dim == 2 && ale_displacement_ != nullptr);
      if (!use_ale_face_geometry)
      {
         gradp_cur = gradp_ref;
         Tr.Face->SetIntPoint(&fip);
         if (dim == 1)
         {
            unit_normal[0] = 1.0;
            ds = fip.weight;
            return true;
         }

         CalcOrtho(Tr.Face->Jacobian(), normal_);
         const double nmag = normal_.Norml2();
         if (!(nmag > 1.0e-20))
         {
            ds = 0.0;
            return false;
         }

         unit_normal = normal_;
         unit_normal /= nmag;
         ds = fip.weight * nmag;
         return true;
      }

      Vector x_current;
      DenseMatrix F, cofactor;
      double J = 1.0;
      EvaluateAleMap2D(*Tr.Elem1, eip, ale_displacement_,
                       x_current, F, cofactor, invF, J);
      ApplyInverseTranspose2D(invF, gradp_ref, gradp_cur);

      Vector ref_area(dim);
      Vector area_vector(dim);
      Tr.Face->SetIntPoint(&fip);
      CalcOrtho(Tr.Face->Jacobian(), ref_area);
      // BuildAleDeformationMap2D stores cofactor = J * F^{-1}, so mapped
      // current-area vectors require cofactor^T * ref_area.
      cofactor.MultTranspose(ref_area, area_vector);

      const double area_mag = area_vector.Norml2();
      if (!(area_mag > 1.0e-20))
      {
         ds = 0.0;
         return false;
      }

      unit_normal = area_vector;
      unit_normal /= area_mag;
      ds = fip.weight * area_mag;
      return true;
   }

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

      const int face_int_order = max(quad_order_,
                                     2 * max(fe_T.GetOrder(), fe_p.GetOrder()) + 2);
      const IntegrationRule &ir_face =
         IntRules.Get(Tr.GetGeometryType(), face_int_order);
      ElementFaceStateReconstruction face_state_recon(dim);
      face_state_recon.Build(state_manager_,
                             fe_T,
                             *Tr.Elem1,
                             Tr.Elem1No,
                             quad_order_,
                             "ComputeFaceGradAnalytic");

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

         const TACOTMaterial::InternalState &state = face_state_recon.Evaluate(eip);
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

         Vector gradp_cur;
         Vector unit_normal;
         DenseMatrix invF;
         double ds = 0.0;
         if (!ComputeFaceTransportGeometry(Tr,
                                           fip,
                                           eip,
                                           gradp_,
                                           gradp_cur,
                                           unit_normal,
                                           invF,
                                           ds))
         {
            continue;
         }

         const double gradp_n = gradp_cur * unit_normal;
         const double g_n = gravity_ * unit_normal;

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
               if (dim == 2 && ale_displacement_ != nullptr)
               {
                  Vector dgradp_ref_j(dim);
                  Vector dgradp_cur_j(dim);
                  for (int d = 0; d < dim; ++d)
                  {
                     dgradp_ref_j[d] = dshape_p_(j, d);
                  }
                  ApplyInverseTranspose2D(invF, dgradp_ref_j, dgradp_cur_j);
                  dgradp_n_j = dgradp_cur_j * unit_normal;
               }
               else
               {
                  for (int d = 0; d < dim; ++d)
                  {
                     dgradp_n_j += dshape_p_(j, d) * unit_normal[d];
                  }
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

      const int face_int_order = max(quad_order_,
                                     2 * max(fe_T.GetOrder(), fe_p.GetOrder()) + 2);
      const IntegrationRule &ir_face =
         IntRules.Get(Tr.GetGeometryType(), face_int_order);
      ElementFaceStateReconstruction face_state_recon(dim);
      face_state_recon.Build(state_manager_,
                             fe_T,
                             *Tr.Elem1,
                             Tr.Elem1No,
                             quad_order_,
                             "ComputeFaceResidual");

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

         const TACOTMaterial::InternalState &state = face_state_recon.Evaluate(eip);
         const TACOTMaterial::SolidProperties solid =
            material_.EvaluateSolid(T_w, p_w, state);
         const TACOTMaterial::GasProperties gas =
            material_.EvaluateGas(T_w, p_w, state);

         const double mu = max(gas.mu, 1.0e-12);
         const double rho_darcy = gas.rho * solid.K / mu;
         const double rho2_darcy = gas.rho * rho_darcy;

         Vector gradp_cur;
         Vector unit_normal;
         DenseMatrix invF;
         double ds = 0.0;
         if (!ComputeFaceTransportGeometry(Tr,
                                           fip,
                                           eip,
                                           gradp_,
                                           gradp_cur,
                                           unit_normal,
                                           invF,
                                           ds))
         {
            continue;
         }

         mflux_.SetSize(dim);
         for (int d = 0; d < dim; ++d)
         {
            mflux_[d] = -rho_darcy * gradp_cur[d] + rho2_darcy * gravity_[d];
         }

         const double m_dot_g_w = mflux_ * unit_normal;
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
   const ParGridFunction *ale_displacement_ = nullptr;

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

static void ProjectRhoToDG1Field(const ParFiniteElementSpace &fes_T,
                                 const ParFiniteElementSpace &fes_p,
                                 const ParFiniteElementSpace &fes_rho,
                                 const ParGridFunction &T,
                                 const ParGridFunction &p,
                                 const TACOTMaterial &material,
                                 const ReactionStateManager &state_manager,
                                 const int quad_order,
                                 ParGridFunction &rho_gf)
{
   MFEM_VERIFY(fes_T.GetNE() == fes_p.GetNE() &&
                  fes_T.GetNE() == fes_rho.GetNE(),
               "Rho DG1 projection requires matching element counts.");

   rho_gf = 0.0;

   Array<int> dofs_T, dofs_p, dofs_rho;
   Vector elT, elp;
   Vector shape_T, shape_p, shape_rho, rhs, coeffs;
   DenseMatrix mass;

   for (int e = 0; e < fes_T.GetNE(); ++e)
   {
      const FiniteElement *fe_T = fes_T.GetFE(e);
      const FiniteElement *fe_p = fes_p.GetFE(e);
      const FiniteElement *fe_rho = fes_rho.GetFE(e);
      ElementTransformation *Tr = fes_T.GetElementTransformation(e);
      const IntegrationRule &ir = IntRules.Get(fe_T->GetGeomType(), quad_order);

      MFEM_VERIFY(ir.GetNPoints() == state_manager.NumQPoints(e),
                  "Rho DG1 projection quadrature mismatch.");
      MFEM_VERIFY(fe_rho != nullptr && Tr != nullptr,
                  "Missing DG1 finite element or transformation for rho_s projection.");

      fes_T.GetElementDofs(e, dofs_T);
      fes_p.GetElementDofs(e, dofs_p);
      fes_rho.GetElementDofs(e, dofs_rho);
      T.GetSubVector(dofs_T, elT);
      p.GetSubVector(dofs_p, elp);

      shape_T.SetSize(fe_T->GetDof());
      shape_p.SetSize(fe_p->GetDof());
      shape_rho.SetSize(fe_rho->GetDof());
      rhs.SetSize(fe_rho->GetDof());
      coeffs.SetSize(fe_rho->GetDof());
      mass.SetSize(fe_rho->GetDof());
      rhs = 0.0;
      mass = 0.0;

      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr->SetIntPoint(&ip);

         fe_T->CalcPhysShape(*Tr, shape_T);
         fe_p->CalcPhysShape(*Tr, shape_p);
         fe_rho->CalcShape(ip, shape_rho);

         const double Tq = shape_T * elT;
         const double pq = shape_p * elp;
         const TACOTMaterial::InternalState &state = state_manager.GetState(e, q);
         const TACOTMaterial::SolidProperties solid =
            material.EvaluateSolid(Tq, pq, state);
         const double weight = ip.weight * Tr->Weight();

         for (int i = 0; i < fe_rho->GetDof(); ++i)
         {
            rhs(i) += weight * shape_rho(i) * solid.rho_s;
            for (int j = 0; j < fe_rho->GetDof(); ++j)
            {
               mass(i, j) += weight * shape_rho(i) * shape_rho(j);
            }
         }
      }

      DenseMatrixInverse mass_inv(mass);
      mass_inv.Mult(rhs, coeffs);
      for (int j = 0; j < dofs_rho.Size(); ++j)
      {
         int dof = dofs_rho[j];
         double sign = 1.0;
         if (dof < 0)
         {
            dof = -1 - dof;
            sign = -1.0;
         }
         rho_gf(dof) = sign * coeffs(j);
      }
   }
}

static void ProjectExtentsToL2Fields(
   const ParFiniteElementSpace &fes_state,
   const ParFiniteElementSpace &fes_extent,
   const ReactionStateManager &state_manager,
   const int quad_order,
   vector<unique_ptr<ParGridFunction>> &extent_fields)
{
   const int nr = state_manager.NumReactions();
   MFEM_VERIFY(static_cast<int>(extent_fields.size()) == nr,
               "Extent DG1 projection requires one field per reaction.");
   MFEM_VERIFY(fes_state.GetNE() == fes_extent.GetNE(),
               "Extent DG1 projection requires matching element counts.");

   for (auto &field : extent_fields)
   {
      *field = 0.0;
   }

   Array<int> dofs_extent;
   vector<Vector> extent_coeffs;
   for (int e = 0; e < fes_state.GetNE(); ++e)
   {
      const FiniteElement *fe_state = fes_state.GetFE(e);
      const FiniteElement *fe_extent = fes_extent.GetFE(e);
      ElementTransformation *Tr_state = fes_state.GetElementTransformation(e);
      MFEM_VERIFY(fe_state != nullptr && fe_extent != nullptr && Tr_state != nullptr,
                  "Missing FE or transformation for extent DG1 projection.");

      ProjectElementQPointExtentsToL2Coefficients(state_manager,
                                                  *fe_state,
                                                  *fe_extent,
                                                  *Tr_state,
                                                  e,
                                                  quad_order,
                                                  "Extent DG1 projection",
                                                  extent_coeffs);
      fes_extent.GetElementDofs(e, dofs_extent);
      for (int r = 0; r < nr; ++r)
      {
         for (int j = 0; j < dofs_extent.Size(); ++j)
         {
            int dof = dofs_extent[j];
            double sign = 1.0;
            if (dof < 0)
            {
               dof = -1 - dof;
               sign = -1.0;
            }
            (*extent_fields[static_cast<size_t>(r)])(dof) =
               sign * extent_coeffs[static_cast<size_t>(r)](j);
         }
      }
   }
}

static void ProjectExtentsToDG1Fields(
   const ParFiniteElementSpace &fes_state,
   const ParFiniteElementSpace &fes_extent,
   const ReactionStateManager &state_manager,
   const int quad_order,
   vector<unique_ptr<ParGridFunction>> &extent_fields)
{
   ProjectExtentsToL2Fields(fes_state,
                            fes_extent,
                            state_manager,
                            quad_order,
                            extent_fields);
}

static void ProjectExtentsToH1Fields(
   const ParFiniteElementSpace &fes_state,
   ParFiniteElementSpace &fes_extent,
   const ReactionStateManager &state_manager,
   const int quad_order,
   vector<unique_ptr<ParGridFunction>> &extent_fields)
{
   const int nr = state_manager.NumReactions();
   MFEM_VERIFY(static_cast<int>(extent_fields.size()) == nr,
               "Extent H1 projection requires one field per reaction.");
   MFEM_VERIFY(fes_state.GetNE() == fes_extent.GetNE(),
               "Extent H1 projection requires matching element counts.");

   for (auto &field : extent_fields)
   {
      *field = 0.0;
   }

   Vector dof_weights(fes_extent.GetVSize());
   dof_weights = 0.0;
   Array<int> dofs_extent;
   vector<Vector> extent_coeffs;
   for (int e = 0; e < fes_state.GetNE(); ++e)
   {
      const FiniteElement *fe_state = fes_state.GetFE(e);
      const FiniteElement *fe_extent = fes_extent.GetFE(e);
      ElementTransformation *Tr_state = fes_state.GetElementTransformation(e);
      MFEM_VERIFY(fe_state != nullptr && fe_extent != nullptr && Tr_state != nullptr,
                  "Missing FE or transformation for extent H1 projection.");

      ProjectElementQPointExtentsToL2Coefficients(state_manager,
                                                  *fe_state,
                                                  *fe_extent,
                                                  *Tr_state,
                                                  e,
                                                  quad_order,
                                                  "Extent H1 projection",
                                                  extent_coeffs);
      fes_extent.GetElementDofs(e, dofs_extent);
      for (int j = 0; j < dofs_extent.Size(); ++j)
      {
         int dof = dofs_extent[j];
         double sign = 1.0;
         if (dof < 0)
         {
            dof = -1 - dof;
            sign = -1.0;
         }
         dof_weights(dof) += 1.0;
         for (int r = 0; r < nr; ++r)
         {
            (*extent_fields[static_cast<size_t>(r)])(dof) +=
               sign * extent_coeffs[static_cast<size_t>(r)](j);
         }
      }
   }

   for (int dof = 0; dof < dof_weights.Size(); ++dof)
   {
      const double w = dof_weights(dof);
      if (w <= 0.0)
      {
         continue;
      }
      const double inv_w = 1.0 / w;
      for (int r = 0; r < nr; ++r)
      {
         (*extent_fields[static_cast<size_t>(r)])(dof) *= inv_w;
      }
   }
}

static void ValidateAleRemapExtentReconstructionConfig(
   const DriverParams &params,
   const ParFiniteElementSpace &fes_state,
   const int quad_order)
{
   if (params.ale_remap_extent_mode != "l2_point_eval" &&
       params.ale_remap_extent_mode != "h1_point_eval" &&
       params.ale_remap_extent_mode != "l2_conservative")
   {
      return;
   }

   unique_ptr<FiniteElementCollection> remap_fec;
   if (params.ale_remap_extent_mode == "h1_point_eval")
   {
      remap_fec =
         make_unique<H1_FECollection>(params.ale_remap_extent_l2_order,
                                      fes_state.GetMesh()->Dimension());
   }
   else
   {
      remap_fec =
         make_unique<L2_FECollection>(params.ale_remap_extent_l2_order,
                                      fes_state.GetMesh()->Dimension());
   }
   for (int e = 0; e < fes_state.GetNE(); ++e)
   {
      const FiniteElement *fe_state = fes_state.GetFE(e);
      const FiniteElement *fe_recon =
         remap_fec->FiniteElementForGeometry(fe_state->GetGeomType());
      MFEM_VERIFY(fe_state != nullptr && fe_recon != nullptr,
                  "ALE remap reconstruction requires valid finite elements.");
      const int nq = IntRules.Get(fe_state->GetGeomType(), quad_order).GetNPoints();
      if (fe_recon->GetDof() > nq)
      {
         ostringstream oss;
         oss << "ale_remap_extent_l2_order=" << params.ale_remap_extent_l2_order
             << " is not solvable for remap on element " << e
             << ": reconstruction dofs=" << fe_recon->GetDof()
             << " exceed remap quadrature points=" << nq << ".";
         throw runtime_error(oss.str());
      }
   }
}

struct QuadratureDiagnosticFields
{
   unique_ptr<QuadratureSpace> qspace;
   unique_ptr<QuadratureFunction> tau_qf;
   unique_ptr<QuadratureFunction> rho_s_qf;
   unique_ptr<QuadratureFunction> gas_density_qf;
   unique_ptr<QuadratureFunction> mobility_qf;
   unique_ptr<QuadratureFunction> pi_total_qf;
   unique_ptr<QuadratureFunction> m_dot_g_qf;
   unique_ptr<QuadratureFunction> degree_char_qf;
   unique_ptr<QuadratureFunction> char_density_fraction_qf;
   unique_ptr<QuadratureFunction> ale_displacement_qf;
   vector<unique_ptr<QuadratureFunction>> extent_qf;
   vector<string> extent_field_names;

   bool Enabled() const { return static_cast<bool>(qspace); }
};

static void InitializeQuadratureDiagnosticFields(
   ParMesh &mesh,
   const int quad_order,
   const int num_reactions,
   QuadratureDiagnosticFields &qdiag)
{
   qdiag.qspace = make_unique<QuadratureSpace>(&mesh, quad_order);
   qdiag.tau_qf = make_unique<QuadratureFunction>(*qdiag.qspace);
   qdiag.rho_s_qf = make_unique<QuadratureFunction>(*qdiag.qspace);
   qdiag.gas_density_qf = make_unique<QuadratureFunction>(*qdiag.qspace);
   qdiag.mobility_qf = make_unique<QuadratureFunction>(*qdiag.qspace);
   qdiag.pi_total_qf = make_unique<QuadratureFunction>(*qdiag.qspace);
   qdiag.m_dot_g_qf = make_unique<QuadratureFunction>(*qdiag.qspace);
   qdiag.degree_char_qf = make_unique<QuadratureFunction>(*qdiag.qspace);
   qdiag.char_density_fraction_qf =
      make_unique<QuadratureFunction>(*qdiag.qspace);
   qdiag.ale_displacement_qf =
      make_unique<QuadratureFunction>(*qdiag.qspace, mesh.SpaceDimension());
   qdiag.extent_qf.clear();
   qdiag.extent_field_names.clear();
   qdiag.extent_qf.reserve(num_reactions);
   qdiag.extent_field_names.reserve(num_reactions);
   for (int r = 0; r < num_reactions; ++r)
   {
      qdiag.extent_qf.emplace_back(make_unique<QuadratureFunction>(*qdiag.qspace));
      qdiag.extent_field_names.push_back("X" + to_string(r + 1) + "_qp");
   }
}

static void UpdateQuadratureDiagnosticFields(
   const ParFiniteElementSpace &fes_T,
   const ParFiniteElementSpace &fes_p,
   const ParGridFunction &T,
   const ParGridFunction &p,
   const ParGridFunction *ale_displacement,
   const TACOTMaterial &material,
   const ReactionStateManager &state_manager,
   const int quad_order,
   QuadratureDiagnosticFields &qdiag)
{
   if (!qdiag.Enabled()) { return; }

   const int nr = state_manager.NumReactions();
   MFEM_VERIFY(static_cast<int>(qdiag.extent_qf.size()) == nr,
               "Quadrature diagnostic reaction count mismatch.");

   Array<int> dofs_T, dofs_p;
   Vector elT, elp;
   Vector shape_T, shape_p;

   real_t *tau_data = qdiag.tau_qf->HostWrite();
   real_t *rho_s_data = qdiag.rho_s_qf->HostWrite();
   real_t *gas_density_data = qdiag.gas_density_qf->HostWrite();
   real_t *mobility_data = qdiag.mobility_qf->HostWrite();
   real_t *pi_total_data = qdiag.pi_total_qf->HostWrite();
   real_t *m_dot_g_data = qdiag.m_dot_g_qf->HostWrite();
   real_t *degree_char_data = qdiag.degree_char_qf->HostWrite();
   real_t *char_density_fraction_data =
      qdiag.char_density_fraction_qf->HostWrite();
   if (qdiag.ale_displacement_qf)
   {
      if (ale_displacement)
      {
         VectorGridFunctionCoefficient ale_disp_coeff(ale_displacement);
         ale_disp_coeff.Project(*qdiag.ale_displacement_qf);
      }
      else
      {
         *qdiag.ale_displacement_qf = 0.0;
      }
   }
   vector<real_t *> extent_data(static_cast<size_t>(nr), nullptr);
   for (int r = 0; r < nr; ++r)
   {
      extent_data[static_cast<size_t>(r)] = qdiag.extent_qf[static_cast<size_t>(r)]->HostWrite();
   }

   const double rho_v = material.InitialSolidDensity();
   const double rho_c = material.CharSolidDensity();
   const double rho_den = rho_v - rho_c;

   for (int e = 0; e < fes_T.GetNE(); ++e)
   {
      const FiniteElement *fe_T = fes_T.GetFE(e);
      const FiniteElement *fe_p = fes_p.GetFE(e);
      ElementTransformation *Tr = fes_T.GetElementTransformation(e);
      const IntegrationRule &ir = IntRules.Get(fe_T->GetGeomType(), quad_order);
      const int qoffset = qdiag.qspace->Offset(e);
      const int qcount = qdiag.qspace->Offset(e + 1) - qoffset;

      MFEM_VERIFY(ir.GetNPoints() == state_manager.NumQPoints(e),
                  "Quadrature diagnostic state size mismatch.");
      MFEM_VERIFY(qcount == ir.GetNPoints(),
                  "Quadrature diagnostic space mismatch.");

      fes_T.GetElementDofs(e, dofs_T);
      fes_p.GetElementDofs(e, dofs_p);
      T.GetSubVector(dofs_T, elT);
      p.GetSubVector(dofs_p, elp);

      shape_T.SetSize(fe_T->GetDof());
      shape_p.SetSize(fe_p->GetDof());

      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr->SetIntPoint(&ip);

         fe_T->CalcPhysShape(*Tr, shape_T);
         fe_p->CalcPhysShape(*Tr, shape_p);

         const double Tq = shape_T * elT;
         const double pq = shape_p * elp;
         const TACOTMaterial::InternalState &state = state_manager.GetState(e, q);
         const TACOTMaterial::SolidProperties solid =
            material.EvaluateSolid(Tq, pq, state);
         const TACOTMaterial::GasProperties gas =
            material.EvaluateGas(Tq, pq, state);
         const double mu_eff = max(gas.mu, 1.0e-12);
         const double pi_total_q = state_manager.GetQPointPiTotal(e, q);

         tau_data[qoffset + q] = solid.tau;
         rho_s_data[qoffset + q] = solid.rho_s;
         gas_density_data[qoffset + q] = gas.rho;
         mobility_data[qoffset + q] = solid.K / mu_eff;
         pi_total_data[qoffset + q] = pi_total_q;
         // m_dot_g is defined by the material model as pi_total.
         m_dot_g_data[qoffset + q] = pi_total_q;
         degree_char_data[qoffset + q] =
            std::max(0.0, std::min(1.0, 1.0 - solid.tau));

         double char_density_fraction = 0.0;
         if (std::abs(rho_den) > 1.0e-14)
         {
            char_density_fraction = (rho_v - solid.rho_s) / rho_den;
            char_density_fraction =
               std::max(0.0, std::min(1.0, char_density_fraction));
         }
         char_density_fraction_data[qoffset + q] = char_density_fraction;

         for (int r = 0; r < nr; ++r)
         {
            const double xi =
               (r < static_cast<int>(state.extent.size())) ? state.extent[r] : 0.0;
            extent_data[static_cast<size_t>(r)][qoffset + q] =
               std::max(0.0, std::min(1.0, xi));
         }
      }
   }
}

TACOTMaterial::InternalState ComputeElementRepresentativeState(
   const ReactionStateManager &state_manager,
   const int elem)
{
   const int nq = state_manager.NumQPoints(elem);
   MFEM_VERIFY(nq > 0, "Boundary representative state requires at least one quadrature point.");

   const TACOTMaterial::InternalState &state0 = state_manager.GetState(elem, 0);
   TACOTMaterial::InternalState representative = state0;
   const int nr = static_cast<int>(state0.extent.size());

   representative.extent.assign(static_cast<size_t>(nr), 0.0);
   representative.extent_old.assign(static_cast<size_t>(nr), 0.0);
   representative.dt = 0.0;

   int samples_used = 0;
   for (int q = 0; q < nq; ++q)
   {
      const TACOTMaterial::InternalState &sq = state_manager.GetState(elem, q);
      if (static_cast<int>(sq.extent.size()) != nr ||
          static_cast<int>(sq.extent_old.size()) != nr)
      {
         continue;
      }

      for (int r = 0; r < nr; ++r)
      {
         representative.extent[r] += sq.extent[r];
         representative.extent_old[r] += sq.extent_old[r];
      }
      ++samples_used;
   }

   // Deterministic fallback: use qp=0 state if per-qpoint arrays are inconsistent.
   if (samples_used == 0)
   {
      representative = state0;
      if (representative.extent_old.size() != representative.extent.size())
      {
         representative.extent_old = representative.extent;
      }
      representative.dt = 0.0;
      return representative;
   }

   const double inv_samples = 1.0 / static_cast<double>(samples_used);
   for (int r = 0; r < nr; ++r)
   {
      representative.extent[r] *= inv_samples;
      representative.extent_old[r] *= inv_samples;
   }

   return representative;
}

SurfaceBoundaryDiagnostics ComputeTopBoundaryDiagnostics(
   ParMesh &pmesh,
   const ParFiniteElementSpace &fes_T,
   const ParFiniteElementSpace &fes_p,
   const ParGridFunction &T,
   const ParGridFunction &p,
   const ParGridFunction *ale_displacement,
   const TACOTMaterial &material,
   const ReactionStateManager &state_manager,
   const BPrimeTable &bprime_table,
   const SurfaceBCSchedule &schedule,
   const SurfaceFluxModelParams &surface_model,
   const Vector &gravity,
   const int quad_order,
   const int top_bdr_attr,
   const double x_target,
   const double time,
   const bool compute_surface_terms)
{
   const SurfaceBCSchedule::BoundaryState bc_state = schedule.Eval(time);

   Array<int> dofs_T, dofs_p;
   Vector elT, elp, shape_T, shape_p, gradp, gradp_cur;
   DenseMatrix dshape_p;

   const int dim = pmesh.Dimension();
   SurfaceBoundaryDiagnostics local_sum{};
   double local_area = 0.0;
   double local_centerline_dist = numeric_limits<double>::infinity();
   double local_centerline_flux = 0.0;
   double local_centerline_gradp_n = 0.0;
   double local_centerline_rho_g = 0.0;
   double local_centerline_mu_g = 0.0;
   double local_centerline_K = 0.0;
   double local_centerline_mobility = 0.0;
   double local_max_face_state_diff = 0.0;
   static bool face_state_difference_logged = false;

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

      const int face_int_order = max(2, 2 * max(fe_T->GetOrder(), fe_p->GetOrder()) + 2);
      const IntegrationRule &ir_face = IntRules.Get(FT->GetGeometryType(), face_int_order);
      const IntegrationRule &ir_elem = IntRules.Get(fe_T->GetGeomType(), quad_order);
      ElementFaceStateReconstruction face_state_recon(dim);
      face_state_recon.Build(state_manager,
                             *fe_T,
                             *FT->Elem1,
                             elem,
                             quad_order,
                             "ComputeTopBoundaryDiagnostics");

      Vector face_pos(dim);
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

         Vector x_current;
         DenseMatrix F, cofactor, invF;
         double J = 1.0;
         EvaluateAleMap2D(*FT->Elem1,
                          eip,
                          ale_displacement,
                          x_current,
                          F,
                          cofactor,
                          invF,
                          J);
         ApplyInverseTranspose2D(invF, gradp, gradp_cur);

         const TACOTMaterial::InternalState &state = face_state_recon.Evaluate(eip);
         const TACOTMaterial::SolidProperties solid =
            material.EvaluateSolid(Tq, pq, state);
         const TACOTMaterial::GasProperties gas =
            material.EvaluateGas(Tq, pq, state);

         if (!face_state_difference_logged)
         {
            const int nearest_q =
               FindNearestVolumeQuadraturePoint(ir_elem, state_manager, elem, eip,
                                               "ComputeTopBoundaryDiagnostics");
            const TACOTMaterial::InternalState &nearest_state =
               state_manager.GetState(elem, nearest_q);
            const int nr =
               min(static_cast<int>(state.extent.size()),
                   static_cast<int>(nearest_state.extent.size()));
            for (int r = 0; r < nr; ++r)
            {
               local_max_face_state_diff =
                  max(local_max_face_state_diff,
                      abs(state.extent[static_cast<size_t>(r)] -
                          nearest_state.extent[static_cast<size_t>(r)]));
            }
         }

         const double mu = max(gas.mu, 1.0e-12);
         const double rho_darcy = gas.rho * solid.K / mu;
         const double rho2_darcy = gas.rho * rho_darcy;

         Vector mflux(dim);
         for (int d = 0; d < dim; ++d)
         {
            mflux[d] = -rho_darcy * gradp_cur[d] + rho2_darcy * gravity[d];
         }

         AleFaceGeometry2D face_geom;
         if (!EvaluateCurrentFaceGeometry2D(*FT,
                                            fip,
                                            eip,
                                            ale_displacement,
                                            face_geom))
         {
            continue;
         }

         double gradp_n = 0.0;
         for (int d = 0; d < dim; ++d)
         {
            gradp_n += gradp_cur[d] * face_geom.unit_normal[d];
         }
         const double m_dot_g_w = mflux * face_geom.unit_normal;

         local_sum.m_dot_g_surf += face_geom.ds * m_dot_g_w;
         if (std::isfinite(x_target))
         {
            face_pos = face_geom.x_current;
            const double dist = std::abs(face_pos[0] - x_target);
            if (dist < local_centerline_dist)
            {
               local_centerline_dist = dist;
               local_centerline_flux = m_dot_g_w;
               local_centerline_gradp_n = gradp_n;
               local_centerline_rho_g = gas.rho;
               local_centerline_mu_g = mu;
               local_centerline_K = solid.K;
               local_centerline_mobility = rho_darcy;
            }
         }
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
            const double m_dot_c =
               (bc_state.chemistryOn != 0) ?
                  std::max(0.0, bc_state.rhoeUeCH) * terms.BprimeC *
                     terms.blowing_correction :
                  0.0;
            local_sum.m_dot_c_surf += face_geom.ds * m_dot_c;
            local_sum.rho_s_surf += face_geom.ds * solid.rho_s;
            local_sum.BprimeG_surf += face_geom.ds * terms.BprimeG;
            local_sum.BprimeC_surf += face_geom.ds * terms.BprimeC;
            local_sum.h_w_surf += face_geom.ds * terms.h_w;
            local_sum.emissivity_surf += face_geom.ds * terms.emissivity;
            local_sum.absorptivity_surf += face_geom.ds * terms.absorptivity;
            local_sum.reflectivity_surf += face_geom.ds * terms.reflectivity;
            local_sum.blowing_correction_surf += face_geom.ds * terms.blowing_correction;
            local_sum.q_conv_surf += face_geom.ds * terms.q_conv;
            local_sum.q_adv_pyro_surf += face_geom.ds * terms.q_adv_pyro;
            local_sum.q_rad_emit_surf += face_geom.ds * terms.q_rad_emit;
            local_sum.q_rad_abs_surf += face_geom.ds * terms.q_rad_abs;
            local_sum.q_surf += face_geom.ds * terms.q_surf;
         }
         local_area += face_geom.ds;
      }
   }

   double local_data[16] = {
      local_sum.m_dot_g_surf,
      local_sum.m_dot_c_surf,
      local_sum.rho_s_surf,
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
   double global_data[16] = {0.0};
   MPI_Allreduce(local_data, global_data, 16, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());

    double global_centerline_dist = numeric_limits<double>::infinity();
    MPI_Allreduce(&local_centerline_dist, &global_centerline_dist, 1, MPI_DOUBLE,
                  MPI_MIN, pmesh.GetComm());
    double centerline_flux_sum_local = 0.0;
    double centerline_gradp_n_sum_local = 0.0;
    double centerline_rho_g_sum_local = 0.0;
    double centerline_mu_g_sum_local = 0.0;
    double centerline_K_sum_local = 0.0;
    double centerline_mobility_sum_local = 0.0;
    int centerline_flux_count_local = 0;
    if (std::isfinite(global_centerline_dist))
    {
       const double tol = 1.0e-12 * std::max(1.0, std::abs(x_target));
       if (std::abs(local_centerline_dist - global_centerline_dist) <= tol)
       {
          centerline_flux_sum_local = local_centerline_flux;
          centerline_gradp_n_sum_local = local_centerline_gradp_n;
          centerline_rho_g_sum_local = local_centerline_rho_g;
          centerline_mu_g_sum_local = local_centerline_mu_g;
          centerline_K_sum_local = local_centerline_K;
          centerline_mobility_sum_local = local_centerline_mobility;
          centerline_flux_count_local = 1;
       }
    }
    double centerline_flux_sum_global = 0.0;
    double centerline_gradp_n_sum_global = 0.0;
    double centerline_rho_g_sum_global = 0.0;
    double centerline_mu_g_sum_global = 0.0;
    double centerline_K_sum_global = 0.0;
    double centerline_mobility_sum_global = 0.0;
    int centerline_flux_count_global = 0;
    MPI_Allreduce(&centerline_flux_sum_local, &centerline_flux_sum_global, 1,
                  MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
    MPI_Allreduce(&centerline_gradp_n_sum_local, &centerline_gradp_n_sum_global, 1,
                  MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
    MPI_Allreduce(&centerline_rho_g_sum_local, &centerline_rho_g_sum_global, 1,
                  MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
    MPI_Allreduce(&centerline_mu_g_sum_local, &centerline_mu_g_sum_global, 1,
                  MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
    MPI_Allreduce(&centerline_K_sum_local, &centerline_K_sum_global, 1,
                  MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
    MPI_Allreduce(&centerline_mobility_sum_local, &centerline_mobility_sum_global, 1,
                  MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
    MPI_Allreduce(&centerline_flux_count_local, &centerline_flux_count_global, 1,
                  MPI_INT, MPI_SUM, pmesh.GetComm());

   if (!face_state_difference_logged)
   {
      double global_max_face_state_diff = 0.0;
      MPI_Allreduce(&local_max_face_state_diff,
                    &global_max_face_state_diff,
                    1,
                    MPI_DOUBLE,
                    MPI_MAX,
                    pmesh.GetComm());
      if (global_max_face_state_diff > 1.0e-10)
      {
         int myid = 0;
         MPI_Comm_rank(pmesh.GetComm(), &myid);
         if (myid == 0)
         {
            cout << "Face-state reconstruction check: max |extent_face - extent_nearest_qp| = "
                 << global_max_face_state_diff << endl;
         }
         face_state_difference_logged = true;
      }
   }

   SurfaceBoundaryDiagnostics out;
   if (centerline_flux_count_global > 0)
   {
      out.m_dot_g_centerline =
         centerline_flux_sum_global / static_cast<double>(centerline_flux_count_global);
      out.gradp_n_centerline =
         centerline_gradp_n_sum_global / static_cast<double>(centerline_flux_count_global);
      out.rho_g_centerline =
         centerline_rho_g_sum_global / static_cast<double>(centerline_flux_count_global);
      out.mu_g_centerline =
         centerline_mu_g_sum_global / static_cast<double>(centerline_flux_count_global);
      out.K_centerline =
         centerline_K_sum_global / static_cast<double>(centerline_flux_count_global);
      out.mobility_centerline =
         centerline_mobility_sum_global / static_cast<double>(centerline_flux_count_global);
   }
   else
   {
      out.m_dot_g_centerline = numeric_limits<double>::quiet_NaN();
      out.gradp_n_centerline = numeric_limits<double>::quiet_NaN();
      out.rho_g_centerline = numeric_limits<double>::quiet_NaN();
      out.mu_g_centerline = numeric_limits<double>::quiet_NaN();
      out.K_centerline = numeric_limits<double>::quiet_NaN();
      out.mobility_centerline = numeric_limits<double>::quiet_NaN();
   }
   const double area = global_data[15];
   if (area <= 0.0)
   {
      out.m_dot_g_surf = numeric_limits<double>::quiet_NaN();
      out.m_dot_g_centerline = numeric_limits<double>::quiet_NaN();
      out.gradp_n_centerline = numeric_limits<double>::quiet_NaN();
      out.rho_g_centerline = numeric_limits<double>::quiet_NaN();
      out.mu_g_centerline = numeric_limits<double>::quiet_NaN();
      out.K_centerline = numeric_limits<double>::quiet_NaN();
      out.mobility_centerline = numeric_limits<double>::quiet_NaN();
      out.m_dot_c_surf = numeric_limits<double>::quiet_NaN();
      out.rho_s_surf = numeric_limits<double>::quiet_NaN();
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
   out.m_dot_c_surf = global_data[1] / area;
   out.rho_s_surf = global_data[2] / area;
   if (!compute_surface_terms)
   {
      out.m_dot_c_surf = numeric_limits<double>::quiet_NaN();
      out.rho_s_surf = numeric_limits<double>::quiet_NaN();
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

   out.BprimeG_surf = global_data[3] / area;
   out.BprimeC_surf = global_data[4] / area;
   out.h_w_surf = global_data[5] / area;
   out.emissivity_surf = global_data[6] / area;
   out.absorptivity_surf = global_data[7] / area;
   out.reflectivity_surf = global_data[8] / area;
   out.blowing_correction_surf = global_data[9] / area;
   out.q_conv_surf = global_data[10] / area;
   out.q_adv_pyro_surf = global_data[11] / area;
   out.q_rad_emit_surf = global_data[12] / area;
   out.q_rad_abs_surf = global_data[13] / area;
   out.q_surf = global_data[14] / area;
   return out;
}

void AssembleTopBoundaryRecessionVelocity(
   ParMesh &pmesh,
   const ParFiniteElementSpace &fes_T,
   const ParFiniteElementSpace &fes_p,
   const ParFiniteElementSpace &fes_scalar,
   const ParGridFunction &T,
   const ParGridFunction &p,
   const ParGridFunction *ale_displacement,
   const TACOTMaterial &material,
   const ReactionStateManager &state_manager,
   const BPrimeTable &bprime_table,
   const SurfaceBCSchedule &schedule,
   const SurfaceFluxModelParams &surface_model,
   const Vector &gravity,
   const int quad_order,
   const int top_bdr_attr,
   const double time,
   const string &recession_density_mode,
   const double recession_density_constant,
   Vector &top_recession_velocity_true)
{
   const SurfaceBCSchedule::BoundaryState bc_state = schedule.Eval(time);

   Vector vel_num(fes_scalar.GetVSize());
   Vector vel_den(fes_scalar.GetVSize());
   vel_num = 0.0;
   vel_den = 0.0;

   Array<int> dofs_T, dofs_p, dofs_scalar;
   Vector elT, elp, shape_T, shape_p, shape_scalar, gradp, gradp_cur;
   DenseMatrix dshape_p;
   const int dim = pmesh.Dimension();

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
      const FiniteElement *fe_scalar = fes_scalar.GetFE(elem);

      fes_T.GetElementDofs(elem, dofs_T);
      fes_p.GetElementDofs(elem, dofs_p);
      fes_scalar.GetElementDofs(elem, dofs_scalar);
      T.GetSubVector(dofs_T, elT);
      p.GetSubVector(dofs_p, elp);

      shape_T.SetSize(fe_T->GetDof());
      shape_p.SetSize(fe_p->GetDof());
      shape_scalar.SetSize(fe_scalar->GetDof());
      dshape_p.SetSize(fe_p->GetDof(), dim);
      gradp.SetSize(dim);

      const int face_int_order =
         max(2,
             max(2 * max(fe_T->GetOrder(), fe_p->GetOrder()) + 2,
                 2 * fe_scalar->GetOrder() + 2));
      const IntegrationRule &ir_face = IntRules.Get(FT->GetGeometryType(), face_int_order);
      ElementFaceStateReconstruction face_state_recon(dim);
      face_state_recon.Build(state_manager,
                             *fe_T,
                             *FT->Elem1,
                             elem,
                             quad_order,
                             "AssembleTopBoundaryRecessionVelocity");

      for (int q = 0; q < ir_face.GetNPoints(); ++q)
      {
         const IntegrationPoint &fip = ir_face.IntPoint(q);
         IntegrationPoint eip;
         FT->Loc1.Transform(fip, eip);

         FT->Elem1->SetIntPoint(&eip);
         fe_T->CalcPhysShape(*FT->Elem1, shape_T);
         fe_p->CalcPhysShape(*FT->Elem1, shape_p);
         fe_scalar->CalcPhysShape(*FT->Elem1, shape_scalar);
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

         Vector x_current;
         DenseMatrix F, cofactor, invF;
         double J = 1.0;
         EvaluateAleMap2D(*FT->Elem1,
                          eip,
                          ale_displacement,
                          x_current,
                          F,
                          cofactor,
                          invF,
                          J);
         ApplyInverseTranspose2D(invF, gradp, gradp_cur);

         const TACOTMaterial::InternalState &state = face_state_recon.Evaluate(eip);
         const TACOTMaterial::SolidProperties solid =
            material.EvaluateSolid(Tq, pq, state);
         const TACOTMaterial::GasProperties gas =
            material.EvaluateGas(Tq, pq, state);

         const double mu = max(gas.mu, 1.0e-12);
         const double rho_darcy = gas.rho * solid.K / mu;
         const double rho2_darcy = gas.rho * rho_darcy;

         Vector mflux(dim);
         for (int d = 0; d < dim; ++d)
         {
            mflux[d] = -rho_darcy * gradp_cur[d] + rho2_darcy * gravity[d];
         }

         AleFaceGeometry2D face_geom;
         if (!EvaluateCurrentFaceGeometry2D(*FT,
                                            fip,
                                            eip,
                                            ale_displacement,
                                            face_geom))
         {
            continue;
         }

         const double m_dot_g_w = mflux * face_geom.unit_normal;

         const SurfaceFluxTerms terms = EvaluateSurfaceFluxTerms(m_dot_g_w,
                                                                 gas.h,
                                                                 Tq,
                                                                 Tq,
                                                                 solid,
                                                                 bc_state,
                                                                 bprime_table,
                                                                 surface_model);
         const double m_dot_c =
            (bc_state.chemistryOn != 0) ?
               std::max(0.0, bc_state.rhoeUeCH) * terms.BprimeC *
                  terms.blowing_correction :
               0.0;

         double rho_rec = recession_density_constant;
         if (recession_density_mode == "char_surface")
         {
            rho_rec = solid.rho_s;
         }
         rho_rec = max(rho_rec, 1.0e-8);
         const double v_rec = std::max(0.0, m_dot_c) / rho_rec;

         for (int j = 0; j < dofs_scalar.Size(); ++j)
         {
            int dof = dofs_scalar[j];
            if (dof < 0) { dof = -1 - dof; }
            const double w = face_geom.ds * shape_scalar[j];
            vel_num(dof) += w * v_rec;
            vel_den(dof) += w;
         }
      }
   }

   ParGridFunction vel_num_gf(const_cast<ParFiniteElementSpace *>(&fes_scalar));
   ParGridFunction vel_den_gf(const_cast<ParFiniteElementSpace *>(&fes_scalar));
   vel_num_gf = 0.0;
   vel_den_gf = 0.0;
   for (int i = 0; i < vel_num_gf.Size(); ++i)
   {
      vel_num_gf(i) = vel_num(i);
      vel_den_gf(i) = vel_den(i);
   }

   Vector vel_num_true;
   Vector vel_den_true;
   vel_num_gf.ParallelAssemble(vel_num_true);
   vel_den_gf.ParallelAssemble(vel_den_true);

   top_recession_velocity_true.SetSize(fes_scalar.TrueVSize());
   top_recession_velocity_true = 0.0;
   for (int tdof = 0; tdof < top_recession_velocity_true.Size(); ++tdof)
   {
      if (vel_den_true(tdof) > 1.0e-16)
      {
         double v = vel_num_true(tdof) / vel_den_true(tdof);
         if (!std::isfinite(v) || v < 0.0)
         {
            v = 0.0;
         }
         top_recession_velocity_true(tdof) = v;
      }
   }

   Array<int> top_marker(pmesh.bdr_attributes.Max());
   top_marker = 0;
   if (top_bdr_attr >= 1 && top_bdr_attr <= top_marker.Size())
   {
      top_marker[top_bdr_attr - 1] = 1;
   }
   Array<int> top_tdofs;
   fes_scalar.GetEssentialTrueDofs(top_marker, top_tdofs);

   Vector filtered(top_recession_velocity_true.Size());
   filtered = 0.0;
   for (int i = 0; i < top_tdofs.Size(); ++i)
   {
      const int tdof = top_tdofs[i];
      double v = top_recession_velocity_true(tdof);
      if (!std::isfinite(v) || v < 0.0)
      {
         v = 0.0;
      }
      filtered(tdof) = v;
   }
   top_recession_velocity_true = filtered;
}

double ClampAndAverageTopRecessionVelocity(const Array<int> &top_tdofs,
                                           const double dt,
                                           const double max_step_recession,
                                           Vector &top_recession_velocity_true,
                                           MPI_Comm comm)
{
   double max_velocity = numeric_limits<double>::infinity();
   if (std::isfinite(max_step_recession) &&
       max_step_recession > 0.0 &&
       dt > 0.0)
   {
      max_velocity = max_step_recession / dt;
   }

   Vector filtered(top_recession_velocity_true.Size());
   filtered = 0.0;

   double local_sum = 0.0;
   double local_count = 0.0;
   for (int i = 0; i < top_tdofs.Size(); ++i)
   {
      const int tdof = top_tdofs[i];
      double v = top_recession_velocity_true(tdof);
      if (!std::isfinite(v) || v <= 0.0)
      {
         v = 0.0;
      }
      if (std::isfinite(max_velocity))
      {
         v = std::min(v, max_velocity);
      }
      filtered(tdof) = v;
      local_sum += v;
      local_count += 1.0;
   }
   top_recession_velocity_true = filtered;

   double local_data[2] = {local_sum, local_count};
   double global_data[2] = {0.0, 0.0};
   MPI_Allreduce(local_data, global_data, 2, MPI_DOUBLE, MPI_SUM, comm);
   if (global_data[1] <= 0.0)
   {
      return 0.0;
   }
   return global_data[0] / global_data[1];
}

double SampleFieldAtPoint(const AlePointLocator2D &locator,
                          const ParGridFunction &gf,
                          const double x, const double y)
{
   Vector target(2);
   target[0] = x;
   target[1] = y;

   double local_val = 0.0;
   int local_found = 0;
   AlePointLocator2D::PointLocation location;
   if (locator.FindPointLocal(target, location))
   {
      local_val = gf.GetValue(location.elem, location.ip);
      local_found = 1;
   }

   double global_sum = 0.0;
   int global_count = 0;
   MPI_Allreduce(&local_val, &global_sum, 1, MPI_DOUBLE, MPI_SUM, locator.Comm());
   MPI_Allreduce(&local_found, &global_count, 1, MPI_INT, MPI_SUM, locator.Comm());

   if (global_count == 0)
   {
      return numeric_limits<double>::quiet_NaN();
   }
   return global_sum / static_cast<double>(global_count);
}

struct VectorDivSample
{
   double wx = numeric_limits<double>::quiet_NaN();
   double wy = numeric_limits<double>::quiet_NaN();
   double divw = numeric_limits<double>::quiet_NaN();
};

struct MassEqProbeSample
{
   double pi_total = numeric_limits<double>::quiet_NaN();
   double tau = numeric_limits<double>::quiet_NaN();
   double epsrho = numeric_limits<double>::quiet_NaN();
   double gradp_y = numeric_limits<double>::quiet_NaN();
   double rho_g = numeric_limits<double>::quiet_NaN();
   double mu_g = numeric_limits<double>::quiet_NaN();
   double K = numeric_limits<double>::quiet_NaN();
   double mobility = numeric_limits<double>::quiet_NaN();
   double mflux_y = numeric_limits<double>::quiet_NaN();
};

VectorDivSample SampleVectorFieldAndDivergenceAtPoint(const AlePointLocator2D &locator,
                                                      const ParGridFunction &gf,
                                                      const double x,
                                                      const double y)
{
   Vector target(2);
   target[0] = x;
   target[1] = y;

   double local_wx = 0.0;
   double local_wy = 0.0;
   double local_divw = 0.0;
   int local_found = 0;
   AlePointLocator2D::PointLocation location;
   if (locator.FindPointLocal(target, location))
   {
      ElementTransformation *Tr = locator.Mesh().GetElementTransformation(location.elem);
      MFEM_VERIFY(Tr != nullptr, "Null element transformation in vector point sample.");
      Tr->SetIntPoint(&location.ip);

      Vector w_val(2);
      gf.GetVectorValue(*Tr, location.ip, w_val);
      local_wx = (w_val.Size() > 0) ? w_val[0] : 0.0;
      local_wy = (w_val.Size() > 1) ? w_val[1] : 0.0;

      if (locator.Displacement())
      {
         Vector x_current;
         DenseMatrix F, cofactor, invF, grad_ref;
         double J = 1.0;
         EvaluateAleMap2D(*Tr,
                          location.ip,
                          locator.Displacement(),
                          x_current,
                          F,
                          cofactor,
                          invF,
                          J);
         gf.GetVectorGradient(*Tr, grad_ref);
         local_divw = ComputeCurrentVectorDivergence2D(grad_ref, invF);
      }
      else
      {
         local_divw = gf.GetDivergence(*Tr);
      }
      local_found = 1;
   }

   double sums[3] = {0.0, 0.0, 0.0};
   double local_vals[3] = {local_wx, local_wy, local_divw};
   int global_count = 0;
   MPI_Allreduce(local_vals, sums, 3, MPI_DOUBLE, MPI_SUM, locator.Comm());
   MPI_Allreduce(&local_found, &global_count, 1, MPI_INT, MPI_SUM, locator.Comm());

   VectorDivSample out;
   if (global_count == 0)
   {
      return out;
   }
   const double inv = 1.0 / static_cast<double>(global_count);
   out.wx = sums[0] * inv;
   out.wy = sums[1] * inv;
   out.divw = sums[2] * inv;
   return out;
}

MassEqProbeSample SampleMassEqProbeAtPoint(const AlePointLocator2D &locator,
                                           const ParFiniteElementSpace &fes_T,
                                           const ParFiniteElementSpace &fes_p,
                                           const ParGridFunction &T_gf,
                                           const ParGridFunction &p_gf,
                                           const TACOTMaterial &material,
                                           const ReactionStateManager &state_manager,
                                           const Vector &gravity,
                                           const int quad_order,
                                           const double x,
                                           const double y)
{
   Vector target(2);
   target[0] = x;
   target[1] = y;

   array<double, 9> local_vals{};
   local_vals.fill(0.0);
   int local_found = 0;

   AlePointLocator2D::PointLocation location;
   if (locator.FindPointLocal(target, location))
   {
      const int elem = location.elem;
      const FiniteElement *fe_T = fes_T.GetFE(elem);
      const FiniteElement *fe_p = fes_p.GetFE(elem);
      ElementTransformation *Tr = locator.Mesh().GetElementTransformation(elem);
      MFEM_VERIFY(fe_T && fe_p && Tr, "Null FE/transformation in mass-equation point sample.");

      const IntegrationPoint &ip = location.ip;
      Tr->SetIntPoint(&ip);

      Array<int> dofs_T, dofs_p;
      Vector elT, elp, shape_T, shape_p, gradp, gradp_cur;
      DenseMatrix dshape_p;

      fes_T.GetElementDofs(elem, dofs_T);
      fes_p.GetElementDofs(elem, dofs_p);
      T_gf.GetSubVector(dofs_T, elT);
      p_gf.GetSubVector(dofs_p, elp);

      shape_T.SetSize(fe_T->GetDof());
      shape_p.SetSize(fe_p->GetDof());
      dshape_p.SetSize(fe_p->GetDof(), locator.Mesh().Dimension());
      gradp.SetSize(locator.Mesh().Dimension());

      fe_T->CalcPhysShape(*Tr, shape_T);
      fe_p->CalcPhysShape(*Tr, shape_p);
      fe_p->CalcPhysDShape(*Tr, dshape_p);

      const double Tq = shape_T * elT;
      const double pq = shape_p * elp;
      gradp = 0.0;
      for (int j = 0; j < fe_p->GetDof(); ++j)
      {
         for (int d = 0; d < locator.Mesh().Dimension(); ++d)
         {
            gradp[d] += elp[j] * dshape_p(j, d);
         }
      }

      Vector x_current;
      DenseMatrix F, cofactor, invF;
      double J = 1.0;
      EvaluateAleMap2D(*Tr,
                       ip,
                       locator.Displacement(),
                       x_current,
                       F,
                       cofactor,
                       invF,
                       J);
      ApplyInverseTranspose2D(invF, gradp, gradp_cur);

      const IntegrationRule &ir =
         IntRules.Get(fe_T->GetGeomType(), quad_order);
      MFEM_VERIFY(ir.GetNPoints() == state_manager.NumQPoints(elem),
                  "Quadrature mismatch in mass-equation point sample.");
      int nearest_q = 0;
      double min_d2 = numeric_limits<double>::max();
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint &iq = ir.IntPoint(q);
         const double dx = ip.x - iq.x;
         const double dy = ip.y - iq.y;
         const double dz = ip.z - iq.z;
         const double d2 = dx * dx + dy * dy + dz * dz;
         if (d2 < min_d2)
         {
            min_d2 = d2;
            nearest_q = q;
         }
      }

      const TACOTMaterial::InternalState &state = state_manager.GetState(elem, nearest_q);
      const TACOTMaterial::SolidProperties solid = material.EvaluateSolid(Tq, pq, state);
      const TACOTMaterial::GasProperties gas = material.EvaluateGas(Tq, pq, state);
      const double pi_total_q = state_manager.GetQPointPiTotal(elem, nearest_q);

      const double mu = std::max(gas.mu, 1.0e-12);
      const double darcy = solid.K / mu;
      const double rho_darcy = gas.rho * darcy;
      const double rho2_darcy = gas.rho * rho_darcy;
      const double gy = (gravity.Size() > 1) ? gravity[1] : 0.0;
      const double gradp_y = (gradp_cur.Size() > 1) ? gradp_cur[1] : 0.0;
      const double mflux_y = -rho_darcy * gradp_y + rho2_darcy * gy;

      local_vals[0] = pi_total_q;
      local_vals[1] = solid.tau;
      local_vals[2] = solid.eps_g * gas.rho;
      local_vals[3] = gradp_y;
      local_vals[4] = gas.rho;
      local_vals[5] = gas.mu;
      local_vals[6] = solid.K;
      local_vals[7] = rho_darcy; // rho_g K / mu
      local_vals[8] = mflux_y;
      local_found = 1;
   }

   array<double, 9> sums{};
   sums.fill(0.0);
   int global_count = 0;
   MPI_Allreduce(local_vals.data(), sums.data(), static_cast<int>(sums.size()),
                 MPI_DOUBLE, MPI_SUM, locator.Comm());
   MPI_Allreduce(&local_found, &global_count, 1, MPI_INT, MPI_SUM, locator.Comm());

   MassEqProbeSample out;
   if (global_count == 0)
   {
      return out;
   }

   const double inv = 1.0 / static_cast<double>(global_count);
   out.pi_total = sums[0] * inv;
   out.tau = sums[1] * inv;
   out.epsrho = sums[2] * inv;
   out.gradp_y = sums[3] * inv;
   out.rho_g = sums[4] * inv;
   out.mu_g = sums[5] * inv;
   out.K = sums[6] * inv;
   out.mobility = sums[7] * inv;
   out.mflux_y = sums[8] * inv;
   return out;
}

double ComputeFrontDepth(const AlePointLocator2D &locator,
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
   double vp = SampleFieldAtPoint(locator, tau_gf, x, yp);

   for (int k = 1; k <= ns; ++k)
   {
      const double yc = y0 - k * dy;
      const double vc = SampleFieldAtPoint(locator, tau_gf, x, yc);

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
         state_manager.SetQPointPiTotal(e, q, solid.pi_total);
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

static void CaptureElementQPJacobians(const mfem::ParFiniteElementSpace &fes_T,
                                      const int quad_order,
                                      vector<vector<double>> &J_qp)
{
   const int ne = fes_T.GetNE();
   J_qp.assign(ne, {});
   for (int e = 0; e < ne; ++e)
   {
      const FiniteElement *fe = fes_T.GetFE(e);
      ElementTransformation *Tr = fes_T.GetElementTransformation(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), quad_order);
      vector<double> &elem_vals = J_qp[e];
      elem_vals.resize(ir.GetNPoints(), 0.0);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr->SetIntPoint(&ip);
         elem_vals[q] = Tr->Weight();
      }
   }
}

struct AleRemapWorkspace
{
   DenseMatrix target_pts;
   vector<pair<int, int>> qp_map;
   vector<vector<double>> remapped_extents;
   unique_ptr<L2_FECollection> extent_l2_fec;
   unique_ptr<ParFiniteElementSpace> extent_l2_fes;
   vector<unique_ptr<ParGridFunction>> extent_l2_fields;
   unique_ptr<H1_FECollection> extent_h1_fec;
   unique_ptr<ParFiniteElementSpace> extent_h1_fes;
   vector<unique_ptr<ParGridFunction>> extent_h1_fields;
   vector<int> element_skipped;
   vector<int> element_clipped;
   int total_qps = 0;
   int num_reactions = 0;
   int quad_order = -1;
   int extent_l2_order = -1;
   int extent_h1_order = -1;

   void Initialize(const ReactionStateManager &state_manager,
                   const ParFiniteElementSpace &fes_state,
                   const int new_quad_order)
   {
      const int nr = state_manager.NumReactions();
      int new_total_qps = 0;
      vector<pair<int, int>> new_qp_map;
      for (int e = 0; e < fes_state.GetNE(); ++e)
      {
         const FiniteElement *fe = fes_state.GetFE(e);
         const int nq = IntRules.Get(fe->GetGeomType(), new_quad_order).GetNPoints();
         for (int q = 0; q < nq; ++q)
         {
            new_qp_map.push_back({e, q});
         }
         new_total_qps += nq;
      }

      const bool need_rebuild =
         (quad_order != new_quad_order ||
          num_reactions != nr ||
          total_qps != new_total_qps);
      if (!need_rebuild)
      {
         element_skipped.assign(static_cast<size_t>(fes_state.GetNE()), 0);
         element_clipped.assign(static_cast<size_t>(fes_state.GetNE()), 0);
         return;
      }

      total_qps = new_total_qps;
      num_reactions = nr;
      quad_order = new_quad_order;
      qp_map = std::move(new_qp_map);
      target_pts.SetSize(fes_state.GetParMesh()->SpaceDimension(), total_qps);
      remapped_extents.assign(static_cast<size_t>(total_qps),
                              vector<double>(static_cast<size_t>(nr), 0.0));
      element_skipped.assign(static_cast<size_t>(fes_state.GetNE()), 0);
      element_clipped.assign(static_cast<size_t>(fes_state.GetNE()), 0);
   }

   void InitializeExtentReconstruction(
      const ParFiniteElementSpace &fes_state,
      const int new_l2_order,
      const int nr)
   {
      const bool need_rebuild =
         (!extent_l2_fec ||
          !extent_l2_fes ||
          extent_l2_order != new_l2_order ||
          static_cast<int>(extent_l2_fields.size()) != nr ||
          extent_l2_fes->GetNE() != fes_state.GetNE());
      if (!need_rebuild)
      {
         return;
      }

      extent_l2_order = new_l2_order;
      extent_l2_fec =
         make_unique<L2_FECollection>(new_l2_order, fes_state.GetMesh()->Dimension());
      extent_l2_fes =
         make_unique<ParFiniteElementSpace>(fes_state.GetParMesh(), extent_l2_fec.get());
      extent_l2_fields.clear();
      extent_l2_fields.reserve(static_cast<size_t>(nr));
      for (int r = 0; r < nr; ++r)
      {
         extent_l2_fields.emplace_back(make_unique<ParGridFunction>(extent_l2_fes.get()));
      }
   }

   void InitializeExtentH1Reconstruction(
      const ParFiniteElementSpace &fes_state,
      const int new_h1_order,
      const int new_quad_order,
      const int nr)
   {
      const bool need_rebuild =
         (!extent_h1_fec ||
          !extent_h1_fes ||
          extent_h1_order != new_h1_order ||
          static_cast<int>(extent_h1_fields.size()) != nr ||
          extent_h1_fes->GetNE() != fes_state.GetNE() ||
          quad_order != new_quad_order);
      if (!need_rebuild)
      {
         return;
      }

      extent_h1_order = new_h1_order;
      extent_h1_fec =
         make_unique<H1_FECollection>(new_h1_order, fes_state.GetMesh()->Dimension());
      extent_h1_fes =
         make_unique<ParFiniteElementSpace>(fes_state.GetParMesh(), extent_h1_fec.get());
      extent_h1_fields.clear();
      extent_h1_fields.reserve(static_cast<size_t>(nr));
      for (int r = 0; r < nr; ++r)
      {
         extent_h1_fields.emplace_back(make_unique<ParGridFunction>(extent_h1_fes.get()));
      }
   }
};

struct AleRemapStats
{
   int global_not_found = 0;
   int global_skipped_elements = 0;
   int global_clipped_values = 0;
};

static void ConservativelyProjectSampledExtentsToTargetQPs(
   const ParFiniteElementSpace &fes_state,
   const ParFiniteElementSpace &fes_extent,
   const ParGridFunction &ale_displacement_new,
   const int quad_order,
   const vector<Vector> &sampled_extents,
   const vector<int> &point_found,
   vector<vector<double>> &remapped_extents,
   AleRemapWorkspace &workspace,
   AleRemapStats &stats)
{
   const int nr = static_cast<int>(sampled_extents.size());
   const int ne = fes_state.GetNE();
   int local_skipped_elements = 0;
   int local_clipped_values = 0;
   int col = 0;

   for (int e = 0; e < ne; ++e)
   {
      const FiniteElement *fe_state = fes_state.GetFE(e);
      const FiniteElement *fe_extent = fes_extent.GetFE(e);
      ElementTransformation *Tr_ref = fes_state.GetElementTransformation(e);
      MFEM_VERIFY(fe_state != nullptr && fe_extent != nullptr && Tr_ref != nullptr,
                  "Missing FE or transformation for conservative ALE remap.");
      const IntegrationRule &ir = IntRules.Get(fe_state->GetGeomType(), quad_order);
      const int nq = ir.GetNPoints();
      const int col0 = col;

      bool skip_element = false;
      for (int q = 0; q < nq; ++q, ++col)
      {
         if (point_found[static_cast<size_t>(col)] == 0)
         {
            skip_element = true;
         }
      }

      if (skip_element)
      {
         workspace.element_skipped[static_cast<size_t>(e)] = 1;
         ++local_skipped_elements;
         continue;
      }

      const int ndof = fe_extent->GetDof();
      DenseMatrix mass(ndof);
      mass = 0.0;
      Vector shape(ndof);
      vector<Vector> rhs(static_cast<size_t>(nr));
      vector<Vector> coeffs(static_cast<size_t>(nr));
      for (int r = 0; r < nr; ++r)
      {
         rhs[static_cast<size_t>(r)].SetSize(ndof);
         rhs[static_cast<size_t>(r)] = 0.0;
         coeffs[static_cast<size_t>(r)].SetSize(ndof);
         coeffs[static_cast<size_t>(r)] = 0.0;
      }

      col = col0;
      for (int q = 0; q < nq; ++q, ++col)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr_ref->SetIntPoint(&ip);
         fe_extent->CalcShape(ip, shape);

         Vector x_current;
         DenseMatrix F, cofactor, invF;
         double J_new = 1.0;
         EvaluateAleMap2D(*Tr_ref,
                          ip,
                          &ale_displacement_new,
                          x_current,
                          F,
                          cofactor,
                          invF,
                          J_new);
         const double weight = ip.weight * Tr_ref->Weight() * J_new;

         for (int i = 0; i < ndof; ++i)
         {
            for (int j = 0; j < ndof; ++j)
            {
               mass(i, j) += weight * shape(i) * shape(j);
            }
         }

         for (int r = 0; r < nr; ++r)
         {
            const double xi = sampled_extents[static_cast<size_t>(r)](col);
            for (int i = 0; i < ndof; ++i)
            {
               rhs[static_cast<size_t>(r)](i) += weight * shape(i) * xi;
            }
         }
      }

      DenseMatrixInverse mass_inv(mass);
      for (int r = 0; r < nr; ++r)
      {
         mass_inv.Mult(rhs[static_cast<size_t>(r)], coeffs[static_cast<size_t>(r)]);
      }

      bool clipped_element = false;
      col = col0;
      for (int q = 0; q < nq; ++q, ++col)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         fe_extent->CalcShape(ip, shape);
         for (int r = 0; r < nr; ++r)
         {
            const double xi = shape * coeffs[static_cast<size_t>(r)];
            const double xi_clamped = std::max(0.0, std::min(1.0, xi));
            if (std::abs(xi_clamped - xi) > 0.0)
            {
               clipped_element = true;
               ++local_clipped_values;
            }
            remapped_extents[static_cast<size_t>(col)][r] = xi_clamped;
         }
      }

      if (clipped_element)
      {
         workspace.element_clipped[static_cast<size_t>(e)] = 1;
      }
   }

   MPI_Allreduce(&local_skipped_elements,
                 &stats.global_skipped_elements,
                 1,
                 MPI_INT,
                 MPI_SUM,
                 fes_state.GetParMesh()->GetComm());
   MPI_Allreduce(&local_clipped_values,
                 &stats.global_clipped_values,
                 1,
                 MPI_INT,
                 MPI_SUM,
                 fes_state.GetParMesh()->GetComm());
}

template <typename SampleFn>
static void SampleAleRemapPointValuesAtTargetPoints(
   const AlePointLocator2D &old_locator,
   const ParFiniteElementSpace &fes_state,
   const DenseMatrix &target_pts,
   const int num_values,
   SampleFn &&sample_fn,
   vector<Vector> &sampled_values,
   vector<int> &point_found)
{
   const int local_npts = target_pts.Width();
   const int dim = target_pts.Height();
   const int myid = Mpi::WorldRank();
   int world_size = 1;
   MPI_Comm_size(fes_state.GetParMesh()->GetComm(), &world_size);

   sampled_values.resize(static_cast<size_t>(num_values));
   for (int v = 0; v < num_values; ++v)
   {
      sampled_values[static_cast<size_t>(v)].SetSize(local_npts);
      sampled_values[static_cast<size_t>(v)] = 0.0;
   }
   point_found.assign(static_cast<size_t>(local_npts), 0);

   vector<int> point_counts(static_cast<size_t>(world_size), 0);
   MPI_Allgather(&local_npts,
                 1,
                 MPI_INT,
                 point_counts.data(),
                 1,
                 MPI_INT,
                 fes_state.GetParMesh()->GetComm());
   vector<int> point_displs(static_cast<size_t>(world_size), 0);
   vector<int> coord_counts(static_cast<size_t>(world_size), 0);
   vector<int> coord_displs(static_cast<size_t>(world_size), 0);
   int total_npts = 0;
   int total_coords = 0;
   for (int rank = 0; rank < world_size; ++rank)
   {
      point_displs[static_cast<size_t>(rank)] = total_npts;
      coord_displs[static_cast<size_t>(rank)] = total_coords;
      total_npts += point_counts[static_cast<size_t>(rank)];
      coord_counts[static_cast<size_t>(rank)] =
         point_counts[static_cast<size_t>(rank)] * dim;
      total_coords += coord_counts[static_cast<size_t>(rank)];
   }

   vector<double> gathered_point_data(static_cast<size_t>(total_coords), 0.0);
   MPI_Allgatherv(local_npts > 0 ? target_pts.GetData() : nullptr,
                  local_npts * dim,
                  MPI_DOUBLE,
                  gathered_point_data.data(),
                  coord_counts.data(),
                  coord_displs.data(),
                  MPI_DOUBLE,
                  fes_state.GetParMesh()->GetComm());

   DenseMatrix gathered_points(gathered_point_data.data(), dim, total_npts);
   vector<int> owner_candidate(static_cast<size_t>(total_npts), world_size);
   vector<int> local_elem_ids(static_cast<size_t>(total_npts), -1);
   vector<IntegrationPoint> local_ref_pts(static_cast<size_t>(total_npts));
   Vector x_target(dim);
   for (int k = 0; k < total_npts; ++k)
   {
      for (int d = 0; d < dim; ++d)
      {
         x_target[d] = gathered_points(d, k);
      }
      AlePointLocator2D::PointLocation location;
      if (!old_locator.FindPointLocal(x_target, location))
      {
         continue;
      }
      owner_candidate[static_cast<size_t>(k)] = myid;
      local_elem_ids[static_cast<size_t>(k)] = location.elem;
      local_ref_pts[static_cast<size_t>(k)] = location.ip;
   }

   vector<int> owner_rank(static_cast<size_t>(total_npts), world_size);
   MPI_Allreduce(owner_candidate.data(),
                 owner_rank.data(),
                 total_npts,
                 MPI_INT,
                 MPI_MIN,
                 fes_state.GetParMesh()->GetComm());

   Vector sample_buffer(num_values);
   vector<double> local_values(static_cast<size_t>(total_npts * num_values), 0.0);
   for (int k = 0; k < total_npts; ++k)
   {
      if (owner_rank[static_cast<size_t>(k)] != myid ||
          local_elem_ids[static_cast<size_t>(k)] < 0)
      {
         continue;
      }
      const int elem = local_elem_ids[static_cast<size_t>(k)];
      sample_fn(elem, local_ref_pts[static_cast<size_t>(k)], sample_buffer);
      for (int v = 0; v < num_values; ++v)
      {
         local_values[static_cast<size_t>(v * total_npts + k)] = sample_buffer(v);
      }
   }

   vector<double> global_values(static_cast<size_t>(total_npts * num_values), 0.0);
   if (!local_values.empty())
   {
      MPI_Allreduce(local_values.data(),
                    global_values.data(),
                    total_npts * num_values,
                    MPI_DOUBLE,
                    MPI_SUM,
                    fes_state.GetParMesh()->GetComm());
   }

   const int local_offset = point_displs[static_cast<size_t>(myid)];
   for (int k = 0; k < local_npts; ++k)
   {
      const int global_k = local_offset + k;
      if (owner_rank[static_cast<size_t>(global_k)] >= world_size)
      {
         continue;
      }
      point_found[static_cast<size_t>(k)] = 1;
      for (int v = 0; v < num_values; ++v)
      {
         sampled_values[static_cast<size_t>(v)](k) =
            global_values[static_cast<size_t>(v * total_npts + global_k)];
      }
   }
}

template <typename SampleFn>
static void SampleAleRemapNearestQPointValuesAtTargetPoints(
   const AlePointLocator2D &old_locator,
   const ParFiniteElementSpace &fes_state,
   const int quad_order,
   const DenseMatrix &target_pts,
   const int num_values,
   SampleFn &&sample_fn,
   vector<Vector> &sampled_values,
   vector<int> &point_found)
{
   SampleAleRemapPointValuesAtTargetPoints(
      old_locator,
      fes_state,
      target_pts,
      num_values,
      [&](const int elem_src, const IntegrationPoint &ip_src, Vector &sample_buffer)
      {
         const FiniteElement *fe = fes_state.GetFE(elem_src);
         const IntegrationRule &ir =
            IntRules.Get(fe->GetGeomType(), quad_order);
         const int nearest_q = FindNearestIntegrationPoint(ir, ip_src);
         sample_fn(elem_src, nearest_q, sample_buffer);
      },
      sampled_values,
      point_found);
}

// Pull back the stored internal state from the old ALE map to the new ALE map.
//
// The reference mesh remains fixed. The old material history is queried with an
// ALE point locator configured on the old ALE displacement. For each
// destination reference QP X_q, we evaluate the old internal state at the
// physical point x^{n+1}(X_q) located in the old ALE configuration:
//
//    xi^{n,remap}(X_q) = xi_h^n( (chi^n)^{-1}(chi^{n+1}(X_q)) ).
//
// The source element is located at the pullback source point. Depending on the
// configured mode, the old state is sampled from the nearest source quadrature
// point, from a reconstructed L2 field, or from a conservative target-element
// projection of the pulled-back old field. The remapped values are written into
// both extent and extent_old so the remap itself does not produce artificial
// reaction increments on the following local kinetics update.
static AleRemapStats RemapExtentsALE(ReactionStateManager &state_manager,
                                     ParMesh &reference_pmesh,
                                     const ParFiniteElementSpace &fes_T,
                                     const ParGridFunction &ale_displacement_old,
                                     const ParGridFunction &ale_displacement_new,
                                     const int quad_order,
                                     const string &extent_remap_mode,
                                     const int extent_l2_order,
                                     AlePointLocator2D &old_locator,
                                     AleRemapWorkspace &workspace)
{
   AleRemapStats stats;
   const int ne = fes_T.GetNE();
   workspace.Initialize(state_manager, fes_T, quad_order);
   old_locator.Update(&ale_displacement_old);
   int global_total_qps = 0;
   MPI_Allreduce(&workspace.total_qps,
                 &global_total_qps,
                 1,
                 MPI_INT,
                 MPI_SUM,
                 reference_pmesh.GetComm());
   if (global_total_qps == 0) { return stats; }

   DenseMatrix &target_pts = workspace.target_pts;
   const vector<pair<int, int>> &qp_map = workspace.qp_map;
   vector<vector<double>> &remapped_extents = workspace.remapped_extents;
   const int dim = target_pts.Height();

   for (int col = 0; col < workspace.total_qps; ++col)
   {
      const auto [e, q] = qp_map[static_cast<size_t>(col)];
      const FiniteElement *fe = fes_T.GetFE(e);
      ElementTransformation *Tr_ref = fes_T.GetElementTransformation(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), quad_order);
      const IntegrationPoint &ip = ir.IntPoint(q);
      Vector x_current;
      DenseMatrix F, cofactor, invF;
      double J = 1.0;
      EvaluateAleMap2D(*Tr_ref,
                       ip,
                       &ale_displacement_new,
                       x_current,
                       F,
                       cofactor,
                       invF,
                       J);

      for (int d = 0; d < dim; ++d)
      {
         target_pts(d, col) = x_current(d);
      }
   }

   const int nr = state_manager.NumReactions();
   vector<Vector> sampled_extents;
   vector<int> point_found;
   const bool use_l2_source_sampling =
      (extent_remap_mode == "l2_point_eval" ||
       extent_remap_mode == "l2_conservative");
   const bool use_h1_source_sampling =
      (extent_remap_mode == "h1_point_eval");
   if (use_l2_source_sampling)
   {
      workspace.InitializeExtentReconstruction(fes_T, extent_l2_order, nr);
      ProjectExtentsToL2Fields(fes_T,
                               *workspace.extent_l2_fes,
                               state_manager,
                               quad_order,
                               workspace.extent_l2_fields);
   }
   if (use_h1_source_sampling)
   {
      workspace.InitializeExtentH1Reconstruction(fes_T,
                                                 extent_l2_order,
                                                 quad_order,
                                                 nr);
      ProjectExtentsToH1Fields(fes_T,
                               *workspace.extent_h1_fes,
                               state_manager,
                               quad_order,
                               workspace.extent_h1_fields);
   }

   if (extent_remap_mode == "nearest_qp")
   {
      SampleAleRemapNearestQPointValuesAtTargetPoints(
         old_locator,
         fes_T,
         quad_order,
         target_pts,
         nr,
         [&](const int elem_src, const int q_src, Vector &sample_buffer)
         {
            sample_buffer = 0.0;
            const TACOTMaterial::InternalState &src =
               state_manager.GetState(elem_src, q_src);
            for (int r = 0; r < nr; ++r)
            {
               sample_buffer(r) =
                  (r < static_cast<int>(src.extent.size())) ? src.extent[r] : 0.0;
            }
         },
         sampled_extents,
         point_found);
   }
   else if (extent_remap_mode == "l2_point_eval" ||
            extent_remap_mode == "l2_conservative")
   {
      SampleAleRemapPointValuesAtTargetPoints(
         old_locator,
         fes_T,
         target_pts,
         nr,
         [&](const int elem_src, const IntegrationPoint &ip_src, Vector &sample_buffer)
         {
            sample_buffer = 0.0;
            for (int r = 0; r < nr; ++r)
            {
               sample_buffer(r) =
                  workspace.extent_l2_fields[static_cast<size_t>(r)]->GetValue(elem_src,
                                                                                ip_src);
            }
         },
         sampled_extents,
         point_found);
   }
   else if (extent_remap_mode == "h1_point_eval")
   {
      SampleAleRemapPointValuesAtTargetPoints(
         old_locator,
         fes_T,
         target_pts,
         nr,
         [&](const int elem_src, const IntegrationPoint &ip_src, Vector &sample_buffer)
         {
            sample_buffer = 0.0;
            for (int r = 0; r < nr; ++r)
            {
               sample_buffer(r) =
                  workspace.extent_h1_fields[static_cast<size_t>(r)]->GetValue(elem_src,
                                                                                ip_src);
            }
         },
         sampled_extents,
         point_found);
   }
   else
   {
      MFEM_ABORT("Unsupported ALE extent remap mode.");
   }

   int local_not_found = 0;
   for (int k = 0; k < workspace.total_qps; ++k)
   {
      remapped_extents[static_cast<size_t>(k)] =
         state_manager.GetState(qp_map[static_cast<size_t>(k)].first,
                                qp_map[static_cast<size_t>(k)].second).extent;

      if (point_found[static_cast<size_t>(k)] == 0)
      {
         ++local_not_found;
      }
   }

   if (extent_remap_mode == "l2_conservative")
   {
      ConservativelyProjectSampledExtentsToTargetQPs(fes_T,
                                                     *workspace.extent_l2_fes,
                                                     ale_displacement_new,
                                                     quad_order,
                                                     sampled_extents,
                                                     point_found,
                                                     remapped_extents,
                                                     workspace,
                                                     stats);
   }
   else
   {
      for (int k = 0; k < workspace.total_qps; ++k)
      {
         if (point_found[static_cast<size_t>(k)] == 0)
         {
            continue;
         }

         for (int r = 0; r < nr; ++r)
         {
            const double xi =
               sampled_extents.empty()
                  ? remapped_extents[static_cast<size_t>(k)][r]
                  : sampled_extents[static_cast<size_t>(r)](k);
            remapped_extents[static_cast<size_t>(k)][r] =
               std::max(0.0, std::min(1.0, xi));
         }
      }
   }

   int global_not_found = 0;
   MPI_Allreduce(&local_not_found,
                 &global_not_found,
                 1,
                 MPI_INT,
                 MPI_SUM,
                 reference_pmesh.GetComm());
   stats.global_not_found = global_not_found;
   if (global_not_found > 0)
   {
      MFEM_WARNING("ALE internal-state remap left " << global_not_found
                   << " quadrature points unchanged because the old lookup "
                   << "point could not be found.");
   }
   if (stats.global_skipped_elements > 0)
   {
      MFEM_WARNING("Conservative ALE internal-state remap skipped "
                   << stats.global_skipped_elements
                   << " target elements because at least one source point "
                   << "could not be found on each skipped element.");
   }
   if (stats.global_clipped_values > 0)
   {
      MFEM_WARNING("Conservative ALE internal-state remap clipped "
                   << stats.global_clipped_values
                   << " extent values to [0,1]; strict conservation is not "
                   << "preserved on those clipped values.");
   }

   int col = 0;
   for (int e = 0; e < ne; ++e)
   {
      const FiniteElement *fe = fes_T.GetFE(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), quad_order);
      for (int q = 0; q < ir.GetNPoints(); ++q, ++col)
      {
         TACOTMaterial::InternalState st = state_manager.GetState(e, q);
         st.extent = remapped_extents[static_cast<size_t>(col)];
         st.extent_old = remapped_extents[static_cast<size_t>(col)];
         st.dt = 0.0;
         state_manager.SetState(e, q, std::move(st));
      }
      state_manager.UpdateExtentAverageFromQPs(e);
   }
   return stats;
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
   cout << "  petsc_use_matnest: " << (p.petsc_use_matnest ? "true" : "false") << endl;
   cout << "  output_every: " << p.output_every << endl;
   cout << "  output_path: " << p.output_path << endl;
   cout << "  collection_name: " << p.collection_name << endl;
   cout << "  diagnostics_mode: mass_metrics_only" << endl;
   cout << "  mass_csv: " << p.mass_csv << endl;
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
   cout << "  moving_mesh: " << (p.moving_mesh ? "true" : "false") << endl;
   cout << "  ale_enabled: " << (p.ale_enabled ? "true" : "false") << endl;
   cout << "  ale_mass_enabled: " << (p.ale_mass_enabled ? "true" : "false") << endl;
   cout << "  ale_energy_enabled: " << (p.ale_energy_enabled ? "true" : "false") << endl;
   cout << "  ale_energy_solid_enabled: " << (p.ale_energy_solid_enabled ? "true" : "false") << endl;
   cout << "  ale_energy_gas_enabled: " << (p.ale_energy_gas_enabled ? "true" : "false") << endl;
   cout << "  ale_remap_enabled: " << (p.ale_remap_enabled ? "true" : "false") << endl;
   cout << "  ale_remap_extent_mode: " << p.ale_remap_extent_mode << endl;
   cout << "  ale_remap_extent_l2_order: " << p.ale_remap_extent_l2_order << endl;
   if (p.ale_remap_extent_mode == "h1_point_eval")
   {
      cout << "  ale_remap_extent_note: h1_point_eval reuses "
              "ale_remap_extent_l2_order as the H1 polynomial order"
           << endl;
   }
   if (p.ale_remap_extent_mode == "l2_conservative")
   {
      cout << "  ale_remap_extent_note: clipping to [0,1] can break strict conservation"
           << endl;
   }
   cout << "  mesh_smoothing_model: " << p.mesh_smoothing_model << endl;
   cout << "  recession_density_mode: " << p.recession_density_mode << endl;
   cout << "  recession_density_constant: " << p.recession_density_constant << endl;
   cout << "  max_step_recession: " << p.max_step_recession << endl;
   cout << "  min_quality_ratio: " << p.min_quality_ratio << endl;
   cout << "  bprime_table_file: " << p.bprime_table_file << endl;
   cout << "  boundary_conditions_file: " << p.boundary_conditions_file << endl;
   cout << "  top_thermal_bc: " << p.top_thermal_bc << endl;
   cout << "  top_temperature_value: " << p.top_temperature_value << endl;
   cout << "  top_temperature_file: "
        << (p.top_temperature_file.empty() ? "none" : p.top_temperature_file) << endl;
   cout << "  top_temperature_source: "
        << (p.top_thermal_bc == "temperature_dirichlet" ?
               (p.top_temperature_file.empty() ? "constant_value" :
                                                "time_schedule_file") :
               "surface_energy_balance")
        << endl;
   cout << "  top_recession_bc: " << p.top_recession_bc << endl;
   cout << "  top_recession_file: "
        << ((p.top_recession_bc == "recession_file" && p.moving_mesh) ?
               EffectiveTopRecessionFile(p) :
               "none")
        << endl;
   cout << "  top_recession_source: "
        << (!p.moving_mesh ? "disabled (moving_mesh=false)" :
                             (p.top_recession_bc == "recession_file" ?
                                 "recession_file" :
                                 "computed_surface_mass_loss"))
        << endl;
   cout << "  emissivity_override: "
        << (std::isfinite(p.emissivity) ? std::to_string(p.emissivity) : "none")
        << endl;
   cout << "  absorptivity_override: "
        << (std::isfinite(p.absorptivity) ? std::to_string(p.absorptivity) : "none")
        << endl;
   cout << "  disable_bprime_c: " << (p.disable_bprime_c ? "true" : "false") << endl;
   cout << "  pato_compat_mode: " << PatoCompatModeName(p.pato_compat_mode) << endl;
   cout << "  remap_self_test.enabled: "
        << (p.remap_self_test.enabled ? "true" : "false") << endl;
   cout << "  remap_self_test.abs_tol: " << p.remap_self_test.abs_tol << endl;
   cout << "  remap_self_test.rel_tol: " << p.remap_self_test.rel_tol << endl;
}

double RemapSelfTestManufacturedExtentOfOrder(const Bounds &bounds,
                                              const int reaction_id,
                                              const int num_reactions,
                                              const int poly_order,
                                              const double x,
                                              const double y)
{
   MFEM_VERIFY(num_reactions > 0,
               "Remap self-test requires at least one reaction.");
   const double lx = bounds.xmax - bounds.xmin;
   const double ly = bounds.ymax - bounds.ymin;
   MFEM_VERIFY(lx > 0.0 && ly > 0.0,
               "Remap self-test requires positive domain extents.");

   const double x_n = (x - bounds.xmin) / lx;
   const double y_n = (y - bounds.ymin) / ly;
   const double w_r =
      static_cast<double>(reaction_id + 1) /
      static_cast<double>(num_reactions + 1);

    switch (poly_order)
    {
       case 0:
          return 0.08 + 0.18 * w_r;
       case 1:
          return 0.05 + 0.20 * w_r +
                 0.10 * (1.0 - w_r) * x_n +
                 0.10 * w_r * y_n;
       case 2:
          return 0.03 + 0.12 * w_r +
                 0.04 * (1.0 - w_r) * x_n +
                 0.03 * w_r * y_n +
                 0.025 * (1.0 - 0.5 * w_r) * x_n * x_n +
                 0.020 * w_r * x_n * y_n +
                 0.020 * (0.5 + 0.5 * w_r) * y_n * y_n;
       default:
          MFEM_ABORT("Unsupported remap self-test manufactured polynomial order.");
    }
}

double RemapSelfTestManufacturedExtent(const Bounds &bounds,
                                       const int reaction_id,
                                       const int num_reactions,
                                       const double x,
                                       const double y)
{
   return RemapSelfTestManufacturedExtentOfOrder(bounds,
                                                 reaction_id,
                                                 num_reactions,
                                                 1,
                                                 x,
                                                 y);
}

void RemapSelfTestTargetPoint(const Bounds &bounds,
                              const double x_ref,
                              const double y_ref,
                              double &x_target,
                              double &y_target)
{
   const double cx = 0.5 * (bounds.xmin + bounds.xmax);
   const double cy = 0.5 * (bounds.ymin + bounds.ymax);
   x_target = cx + 0.92 * (x_ref - cx);
   y_target = cy + 0.88 * (y_ref - cy);
}

void SetRemapSelfTestDisplacement(const Bounds &bounds,
                                  const ParGridFunction &reference_nodes,
                                  ParGridFunction &ale_displacement)
{
   auto *fes = ale_displacement.ParFESpace();
   MFEM_VERIFY(fes != nullptr,
               "Remap self-test displacement requires a valid space.");
   const auto *nodes_fes = reference_nodes.ParFESpace();
   MFEM_VERIFY(nodes_fes == fes,
               "Remap self-test displacement requires matching nodal spaces.");
   const int ndof = fes->GetNDofs();
   const int vdim = fes->GetVDim();
   MFEM_VERIFY(vdim >= 2,
               "Remap self-test displacement requires at least 2 components.");

   ale_displacement = 0.0;
   for (int i = 0; i < ndof; ++i)
   {
      const int vdof_x = fes->DofToVDof(i, 0);
      const int vdof_y = fes->DofToVDof(i, 1);
      const double x_ref = reference_nodes(vdof_x);
      const double y_ref = reference_nodes(vdof_y);
      double x_target = 0.0;
      double y_target = 0.0;
      RemapSelfTestTargetPoint(bounds, x_ref, y_ref, x_target, y_target);
      ale_displacement(vdof_x) = x_target - x_ref;
      ale_displacement(vdof_y) = y_target - y_ref;
   }
}

void SetRemapSelfTestTranslatedDisplacement(const Bounds &bounds,
                                            const ParGridFunction &reference_nodes,
                                            const double shift_x,
                                            const double shift_y,
                                            ParGridFunction &ale_displacement)
{
   (void)bounds;
   auto *fes = ale_displacement.ParFESpace();
   MFEM_VERIFY(fes != nullptr,
               "Remap self-test translated displacement requires a valid space.");
   const auto *nodes_fes = reference_nodes.ParFESpace();
   MFEM_VERIFY(nodes_fes == fes,
               "Remap self-test translated displacement requires matching nodal spaces.");
   const int ndof = fes->GetNDofs();
   const int vdim = fes->GetVDim();
   MFEM_VERIFY(vdim >= 2,
               "Remap self-test translated displacement requires at least 2 components.");

   ale_displacement = 0.0;
   for (int i = 0; i < ndof; ++i)
   {
      const int vdof_x = fes->DofToVDof(i, 0);
      const int vdof_y = fes->DofToVDof(i, 1);
      ale_displacement(vdof_x) = shift_x;
      ale_displacement(vdof_y) = shift_y;
   }
}

void SeedRemapSelfTestState(const Bounds &bounds,
                            const ParFiniteElementSpace &fes_T,
                            const int quad_order,
                            const int manufactured_order,
                            ReactionStateManager &state_manager)
{
   const int nr = state_manager.NumReactions();
   const double dt_sentinel = 7.5;
   Vector x_phys(2);
   for (int e = 0; e < fes_T.GetNE(); ++e)
   {
      const FiniteElement *fe = fes_T.GetFE(e);
      ElementTransformation *Tr = fes_T.GetElementTransformation(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), quad_order);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr->SetIntPoint(&ip);
         Tr->Transform(ip, x_phys);

         TACOTMaterial::InternalState st = state_manager.GetState(e, q);
         st.extent.assign(static_cast<size_t>(nr), 0.0);
         st.extent_old.assign(static_cast<size_t>(nr), 0.0);
         for (int r = 0; r < nr; ++r)
         {
            const double xi = RemapSelfTestManufacturedExtentOfOrder(bounds,
                                                                     r,
                                                                     nr,
                                                                     manufactured_order,
                                                                     x_phys(0),
                                                                     x_phys(1));
            st.extent[static_cast<size_t>(r)] = xi;
            st.extent_old[static_cast<size_t>(r)] = xi;
         }
         st.dt = dt_sentinel;
         state_manager.SetState(e, q, std::move(st));
      }
      state_manager.UpdateExtentAverageFromQPs(e);
   }
}

void SeedRemapSelfTestClipStressState(const ParFiniteElementSpace &fes_T,
                                      const int quad_order,
                                      ReactionStateManager &state_manager)
{
   const int nr = state_manager.NumReactions();
   const double dt_sentinel = 7.5;
   for (int e = 0; e < fes_T.GetNE(); ++e)
   {
      const FiniteElement *fe = fes_T.GetFE(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), quad_order);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         TACOTMaterial::InternalState st = state_manager.GetState(e, q);
         st.extent.assign(static_cast<size_t>(nr), 0.0);
         st.extent_old.assign(static_cast<size_t>(nr), 0.0);
         for (int r = 0; r < nr; ++r)
         {
            const bool high = ((e + q + r) % 2) != 0;
            const double xi = high ? 1.0 : 0.0;
            st.extent[static_cast<size_t>(r)] = xi;
            st.extent_old[static_cast<size_t>(r)] = xi;
         }
         st.dt = dt_sentinel;
         state_manager.SetState(e, q, std::move(st));
      }
      state_manager.UpdateExtentAverageFromQPs(e);
   }
}

struct AleRemapSelfTestCase
{
   string name;
   string extent_mode = "nearest_qp";
   int extent_l2_order = 1;
   int manufactured_order = 1;
   bool use_translated_displacement = false;
   double shift_x = 0.0;
   double shift_y = 0.0;
   bool check_pointwise = true;
   bool check_conservative_integrals = false;
   bool expect_some_not_found = false;
   bool expect_some_clipping = false;
   bool use_clip_stress_seed = false;
};

static void SampleReconstructedRemapSelfTestExtentsAtTargetPoints(
   const string &extent_mode,
   const int extent_l2_order,
   const ReactionStateManager &state_manager,
   const AlePointLocator2D &old_locator,
   const ParFiniteElementSpace &fes_T,
   const int quad_order,
   const DenseMatrix &target_pts,
   vector<Vector> &sampled_extents,
   vector<int> &point_found)
{
   const int nr = state_manager.NumReactions();
   if (extent_mode == "nearest_qp")
   {
      SampleAleRemapNearestQPointValuesAtTargetPoints(
         old_locator,
         fes_T,
         quad_order,
         target_pts,
         nr,
         [&](const int elem_src, const int q_src, Vector &sample_buffer)
         {
            sample_buffer = 0.0;
            const TACOTMaterial::InternalState &src =
               state_manager.GetState(elem_src, q_src);
            for (int r = 0; r < nr; ++r)
            {
               sample_buffer(r) =
                  (r < static_cast<int>(src.extent.size())) ? src.extent[r] : 0.0;
            }
         },
         sampled_extents,
         point_found);
      return;
   }

   vector<unique_ptr<ParGridFunction>> extent_fields;
   if (extent_mode == "h1_point_eval")
   {
      H1_FECollection h1_fec(extent_l2_order, fes_T.GetMesh()->Dimension());
      ParFiniteElementSpace fes_extent(fes_T.GetParMesh(), &h1_fec);
      extent_fields.reserve(static_cast<size_t>(nr));
      for (int r = 0; r < nr; ++r)
      {
         extent_fields.emplace_back(make_unique<ParGridFunction>(&fes_extent));
      }
      ProjectExtentsToH1Fields(fes_T,
                               fes_extent,
                               state_manager,
                               quad_order,
                               extent_fields);
      SampleAleRemapPointValuesAtTargetPoints(
         old_locator,
         fes_T,
         target_pts,
         nr,
         [&](const int elem_src, const IntegrationPoint &ip_src, Vector &sample_buffer)
         {
            sample_buffer = 0.0;
            for (int r = 0; r < nr; ++r)
            {
               sample_buffer(r) =
                  extent_fields[static_cast<size_t>(r)]->GetValue(elem_src, ip_src);
            }
         },
         sampled_extents,
         point_found);
      return;
   }

   L2_FECollection l2_fec(extent_l2_order, fes_T.GetMesh()->Dimension());
   ParFiniteElementSpace fes_extent(fes_T.GetParMesh(), &l2_fec);
   extent_fields.reserve(static_cast<size_t>(nr));
   for (int r = 0; r < nr; ++r)
   {
      extent_fields.emplace_back(make_unique<ParGridFunction>(&fes_extent));
   }
   ProjectExtentsToL2Fields(fes_T,
                            fes_extent,
                            state_manager,
                            quad_order,
                            extent_fields);
   SampleAleRemapPointValuesAtTargetPoints(
      old_locator,
      fes_T,
      target_pts,
      nr,
      [&](const int elem_src, const IntegrationPoint &ip_src, Vector &sample_buffer)
      {
         sample_buffer = 0.0;
         for (int r = 0; r < nr; ++r)
         {
            sample_buffer(r) =
               extent_fields[static_cast<size_t>(r)]->GetValue(elem_src, ip_src);
         }
      },
      sampled_extents,
      point_found);
}

template <typename ValueFn>
static void ComputeRemapSelfTestElementPhysicalIntegrals(
   const ParFiniteElementSpace &fes_T,
   const ParGridFunction &ale_displacement_new,
   const int quad_order,
   const int nr,
   ValueFn &&value_fn,
   vector<vector<double>> &elem_integrals)
{
   elem_integrals.assign(static_cast<size_t>(nr),
                         vector<double>(static_cast<size_t>(fes_T.GetNE()), 0.0));
   int col = 0;
   for (int e = 0; e < fes_T.GetNE(); ++e)
   {
      const FiniteElement *fe = fes_T.GetFE(e);
      ElementTransformation *Tr_ref = fes_T.GetElementTransformation(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), quad_order);
      for (int q = 0; q < ir.GetNPoints(); ++q, ++col)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Vector x_current;
         DenseMatrix F, cofactor, invF;
         double J_new = 1.0;
         EvaluateAleMap2D(*Tr_ref,
                          ip,
                          &ale_displacement_new,
                          x_current,
                          F,
                          cofactor,
                          invF,
                          J_new);
         const double weight = ip.weight * Tr_ref->Weight() * J_new;
         for (int r = 0; r < nr; ++r)
         {
            elem_integrals[static_cast<size_t>(r)][static_cast<size_t>(e)] +=
               weight * value_fn(e, q, col, r);
         }
      }
   }
}

static void RunAleRemapSelfTestCase(const DriverParams &params,
                                    const Bounds &bounds,
                                    const TACOTMaterial &material,
                                    const ParFiniteElementSpace &fes_T,
                                    const ParGridFunction &reference_nodes,
                                    ParFiniteElementSpace &ale_vector_fes,
                                    const AleRemapSelfTestCase &test_case)
{
   DriverParams case_params = params;
   case_params.ale_remap_extent_mode = test_case.extent_mode;
   case_params.ale_remap_extent_l2_order = test_case.extent_l2_order;

   const int quad_order = max(2, 2 * params.order + 2);
   ValidateAleRemapExtentReconstructionConfig(case_params, fes_T, quad_order);

   ReactionStateManager state_manager;
   state_manager.Initialize(fes_T, quad_order, material);
   InitializeDiagnostics(material, state_manager);
   if (test_case.use_clip_stress_seed)
   {
      SeedRemapSelfTestClipStressState(fes_T, quad_order, state_manager);
   }
   else
   {
      SeedRemapSelfTestState(bounds,
                             fes_T,
                             quad_order,
                             test_case.manufactured_order,
                             state_manager);
   }
   const ReactionStateManager source_state = state_manager;

   ParGridFunction ale_displacement_old(&ale_vector_fes);
   ParGridFunction ale_displacement_new(&ale_vector_fes);
   ale_displacement_old = 0.0;
   if (test_case.use_translated_displacement)
   {
      SetRemapSelfTestTranslatedDisplacement(bounds,
                                            reference_nodes,
                                            test_case.shift_x,
                                            test_case.shift_y,
                                            ale_displacement_new);
   }
   else
   {
      SetRemapSelfTestDisplacement(bounds, reference_nodes, ale_displacement_new);
   }

   AlePointLocator2D old_locator(*fes_T.GetParMesh(), fes_T);
   AleRemapWorkspace workspace;
   const AleRemapStats remap_stats =
      RemapExtentsALE(state_manager,
                      *fes_T.GetParMesh(),
                      fes_T,
                      ale_displacement_old,
                      ale_displacement_new,
                      quad_order,
                      test_case.extent_mode,
                      test_case.extent_l2_order,
                      old_locator,
                      workspace);

   const int nr = state_manager.NumReactions();
   const int ne = fes_T.GetNE();
   vector<Vector> exact_expected_extents;
   vector<int> exact_point_found;
   if (!test_case.use_clip_stress_seed)
   {
      if (test_case.extent_mode == "nearest_qp")
      {
         SampleAleRemapNearestQPointValuesAtTargetPoints(
            old_locator,
            fes_T,
            quad_order,
            workspace.target_pts,
            nr,
            [&](const int elem_src, const int q_src, Vector &sample_buffer)
            {
               sample_buffer = 0.0;
               const FiniteElement *fe_src = fes_T.GetFE(elem_src);
               ElementTransformation *Tr_src = fes_T.GetElementTransformation(elem_src);
               const IntegrationRule &ir_src =
                  IntRules.Get(fe_src->GetGeomType(), quad_order);
               const IntegrationPoint &ip_src = ir_src.IntPoint(q_src);
               Vector x_src;
               DenseMatrix F, cofactor, invF;
               double J = 1.0;
               EvaluateAleMap2D(*Tr_src,
                                ip_src,
                                &ale_displacement_old,
                                x_src,
                                F,
                                cofactor,
                                invF,
                                J);
               for (int r = 0; r < nr; ++r)
               {
                  sample_buffer(r) = RemapSelfTestManufacturedExtentOfOrder(
                     bounds,
                     r,
                     nr,
                     test_case.manufactured_order,
                     x_src(0),
                     x_src(1));
               }
            },
            exact_expected_extents,
            exact_point_found);
      }
      else
      {
         SampleAleRemapPointValuesAtTargetPoints(
            old_locator,
            fes_T,
            workspace.target_pts,
            nr,
            [&](const int elem_src, const IntegrationPoint &ip_src, Vector &sample_buffer)
            {
               sample_buffer = 0.0;
               ElementTransformation *Tr_src = fes_T.GetElementTransformation(elem_src);
               Vector x_src;
               DenseMatrix F, cofactor, invF;
               double J = 1.0;
               EvaluateAleMap2D(*Tr_src,
                                ip_src,
                                &ale_displacement_old,
                                x_src,
                                F,
                                cofactor,
                                invF,
                                J);
               for (int r = 0; r < nr; ++r)
               {
                  sample_buffer(r) = RemapSelfTestManufacturedExtentOfOrder(
                     bounds,
                     r,
                     nr,
                     test_case.manufactured_order,
                     x_src(0),
                     x_src(1));
               }
            },
            exact_expected_extents,
            exact_point_found);
      }
   }

   vector<Vector> reconstructed_expected_extents;
   vector<int> reconstructed_point_found;
   if (test_case.extent_mode != "nearest_qp" ||
       test_case.check_conservative_integrals ||
       test_case.expect_some_clipping)
   {
      SampleReconstructedRemapSelfTestExtentsAtTargetPoints(test_case.extent_mode,
                                                            test_case.extent_l2_order,
                                                            source_state,
                                                            old_locator,
                                                            fes_T,
                                                            quad_order,
                                                            workspace.target_pts,
                                                            reconstructed_expected_extents,
                                                            reconstructed_point_found);
   }

   const vector<int> &expected_point_found =
      !exact_point_found.empty() ? exact_point_found : reconstructed_point_found;
   vector<int> expected_element_skipped(static_cast<size_t>(ne), 0);
   int local_expected_skipped_elements = 0;
   if (test_case.extent_mode == "l2_conservative")
   {
      for (int k = 0; k < workspace.total_qps; ++k)
      {
         if (!expected_point_found.empty() &&
             expected_point_found[static_cast<size_t>(k)] == 0)
         {
            const int elem = workspace.qp_map[static_cast<size_t>(k)].first;
            expected_element_skipped[static_cast<size_t>(elem)] = 1;
         }
      }
      for (int e = 0; e < ne; ++e)
      {
         local_expected_skipped_elements +=
            expected_element_skipped[static_cast<size_t>(e)];
      }
   }

   vector<vector<double>> expected_elem_sum(static_cast<size_t>(nr),
                                            vector<double>(static_cast<size_t>(ne), 0.0));
   int local_total_qps = 0;
   int local_qp_mismatch = 0;
   int local_qp_old_mismatch = 0;
   int local_dt_mismatch = 0;
   int local_elem_avg_mismatch = 0;
    int local_integral_mismatch = 0;
   int local_expected_not_found = 0;
   int local_bounds_mismatch = 0;
   double local_max_abs_err = 0.0;
   double local_max_rel_err = 0.0;
   int col = 0;

   for (int e = 0; e < ne; ++e)
   {
      const FiniteElement *fe = fes_T.GetFE(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), quad_order);
      for (int q = 0; q < ir.GetNPoints(); ++q, ++col)
      {
         ++local_total_qps;
         const TACOTMaterial::InternalState &st = state_manager.GetState(e, q);
         if (std::abs(st.dt) > 0.0)
         {
            ++local_dt_mismatch;
         }

         if (!expected_point_found.empty() &&
             expected_point_found[static_cast<size_t>(col)] == 0)
         {
            ++local_expected_not_found;
         }

         const double tol =
            10.0 * std::numeric_limits<double>::epsilon();
         for (int r = 0; r < nr; ++r)
         {
            const double actual =
               (r < static_cast<int>(st.extent.size())) ?
                  st.extent[static_cast<size_t>(r)] :
                  0.0;
            const double actual_old =
               (r < static_cast<int>(st.extent_old.size())) ?
                  st.extent_old[static_cast<size_t>(r)] :
                  0.0;
            if (actual < -tol || actual > 1.0 + tol ||
                actual_old < -tol || actual_old > 1.0 + tol)
            {
               ++local_bounds_mismatch;
            }
            const double threshold =
               params.remap_self_test.abs_tol +
               params.remap_self_test.rel_tol * std::max(1.0, std::abs(actual));
            if (std::abs(actual_old - actual) > threshold)
            {
               ++local_qp_old_mismatch;
            }

            if (!test_case.check_pointwise)
            {
               continue;
            }

            const double unchanged =
               (r < static_cast<int>(source_state.GetState(e, q).extent.size())) ?
                  source_state.GetState(e, q).extent[static_cast<size_t>(r)] :
                  0.0;
            double expected = unchanged;
            if (test_case.extent_mode == "l2_conservative")
            {
               if (!expected_element_skipped[static_cast<size_t>(e)])
               {
                  expected = exact_expected_extents[static_cast<size_t>(r)](col);
               }
            }
            else if (!expected_point_found.empty() &&
                     expected_point_found[static_cast<size_t>(col)] != 0)
            {
               expected = exact_expected_extents[static_cast<size_t>(r)](col);
            }

            expected_elem_sum[static_cast<size_t>(r)][static_cast<size_t>(e)] += expected;
            const double abs_err = std::abs(actual - expected);
            const double rel_err =
               abs_err / std::max(1.0, std::abs(expected));
            const double expected_threshold =
               params.remap_self_test.abs_tol +
               params.remap_self_test.rel_tol * std::max(1.0, std::abs(expected));
            local_max_abs_err = std::max(local_max_abs_err, abs_err);
            local_max_rel_err = std::max(local_max_rel_err, rel_err);
            if (abs_err > expected_threshold)
            {
               ++local_qp_mismatch;
            }
         }
      }
   }

   if (test_case.check_pointwise)
   {
      for (int r = 0; r < nr; ++r)
      {
         for (int e = 0; e < ne; ++e)
         {
            const int nq = state_manager.NumQPoints(e);
            MFEM_VERIFY(nq > 0, "ALE remap self-test requires at least one QP per element.");
            const double inv_nq = 1.0 / static_cast<double>(nq);
            const double expected_avg =
               expected_elem_sum[static_cast<size_t>(r)][static_cast<size_t>(e)] * inv_nq;
            const double actual_avg = state_manager.ExtentElement(r)[static_cast<size_t>(e)];
            const double abs_err = std::abs(actual_avg - expected_avg);
            const double threshold =
               params.remap_self_test.abs_tol +
               params.remap_self_test.rel_tol * std::max(1.0, std::abs(expected_avg));
            local_max_abs_err = std::max(local_max_abs_err, abs_err);
            local_max_rel_err = std::max(local_max_rel_err,
                                         abs_err /
                                            std::max(1.0, std::abs(expected_avg)));
            if (abs_err > threshold)
            {
               ++local_elem_avg_mismatch;
            }
         }
      }
   }

   if (test_case.check_conservative_integrals)
   {
      vector<vector<double>> expected_integrals;
      ComputeRemapSelfTestElementPhysicalIntegrals(
         fes_T,
         ale_displacement_new,
         quad_order,
         nr,
         [&](const int e, const int q, const int col_qp, const int r)
         {
            (void)e;
            (void)q;
            return reconstructed_expected_extents[static_cast<size_t>(r)](col_qp);
         },
         expected_integrals);

      vector<vector<double>> actual_integrals;
      ComputeRemapSelfTestElementPhysicalIntegrals(
         fes_T,
         ale_displacement_new,
         quad_order,
         nr,
         [&](const int e, const int q, const int col_qp, const int r)
         {
            (void)col_qp;
            const TACOTMaterial::InternalState &st = state_manager.GetState(e, q);
            return (r < static_cast<int>(st.extent.size())) ?
                      st.extent[static_cast<size_t>(r)] :
                      0.0;
         },
         actual_integrals);

      for (int e = 0; e < ne; ++e)
      {
         if (expected_element_skipped[static_cast<size_t>(e)] != 0 ||
             workspace.element_clipped[static_cast<size_t>(e)] != 0)
         {
            continue;
         }
         for (int r = 0; r < nr; ++r)
         {
            const double expected =
               expected_integrals[static_cast<size_t>(r)][static_cast<size_t>(e)];
            const double actual =
               actual_integrals[static_cast<size_t>(r)][static_cast<size_t>(e)];
            const double abs_err = std::abs(actual - expected);
            const double rel_err =
               abs_err / std::max(1.0, std::abs(expected));
            const double threshold =
               params.remap_self_test.abs_tol +
               params.remap_self_test.rel_tol * std::max(1.0, std::abs(expected));
            local_max_abs_err = std::max(local_max_abs_err, abs_err);
            local_max_rel_err = std::max(local_max_rel_err, rel_err);
            if (abs_err > threshold)
            {
               ++local_integral_mismatch;
            }
         }
      }
   }

   int global_total_qps = 0;
   int global_qp_mismatch = 0;
   int global_qp_old_mismatch = 0;
   int global_dt_mismatch = 0;
   int global_elem_avg_mismatch = 0;
   int global_integral_mismatch = 0;
   int global_expected_not_found = 0;
   int global_expected_skipped_elements = 0;
   int global_bounds_mismatch = 0;
   double global_max_abs_err = 0.0;
   double global_max_rel_err = 0.0;
   MPI_Allreduce(&local_total_qps, &global_total_qps, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&local_qp_mismatch, &global_qp_mismatch, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&local_qp_old_mismatch,
                 &global_qp_old_mismatch,
                 1,
                 MPI_INT,
                 MPI_SUM,
                 MPI_COMM_WORLD);
   MPI_Allreduce(&local_dt_mismatch, &global_dt_mismatch, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&local_elem_avg_mismatch,
                 &global_elem_avg_mismatch,
                 1,
                 MPI_INT,
                 MPI_SUM,
                 MPI_COMM_WORLD);
   MPI_Allreduce(&local_integral_mismatch,
                 &global_integral_mismatch,
                 1,
                 MPI_INT,
                 MPI_SUM,
                 MPI_COMM_WORLD);
   MPI_Allreduce(&local_expected_not_found,
                 &global_expected_not_found,
                 1,
                 MPI_INT,
                 MPI_SUM,
                 MPI_COMM_WORLD);
   MPI_Allreduce(&local_expected_skipped_elements,
                 &global_expected_skipped_elements,
                 1,
                 MPI_INT,
                 MPI_SUM,
                 MPI_COMM_WORLD);
   MPI_Allreduce(&local_bounds_mismatch,
                 &global_bounds_mismatch,
                 1,
                 MPI_INT,
                 MPI_SUM,
                 MPI_COMM_WORLD);
   MPI_Allreduce(&local_max_abs_err,
                 &global_max_abs_err,
                 1,
                 MPI_DOUBLE,
                 MPI_MAX,
                 MPI_COMM_WORLD);
   MPI_Allreduce(&local_max_rel_err,
                 &global_max_rel_err,
                 1,
                 MPI_DOUBLE,
                 MPI_MAX,
                 MPI_COMM_WORLD);

   const bool expect_some_not_found = test_case.expect_some_not_found;
   const bool not_found_behavior_ok =
      expect_some_not_found ?
         (remap_stats.global_not_found > 0 &&
          remap_stats.global_not_found < global_total_qps) :
         (remap_stats.global_not_found == 0);
   const bool skipped_behavior_ok =
      (test_case.extent_mode == "l2_conservative") ?
         (remap_stats.global_skipped_elements == global_expected_skipped_elements) :
         (remap_stats.global_skipped_elements == 0);
   const bool clipping_behavior_ok =
      test_case.expect_some_clipping ?
         (remap_stats.global_clipped_values > 0) :
         (remap_stats.global_clipped_values == 0);

   if (!not_found_behavior_ok ||
       !skipped_behavior_ok ||
       !clipping_behavior_ok ||
       remap_stats.global_not_found != global_expected_not_found ||
       global_qp_mismatch != 0 ||
       global_qp_old_mismatch != 0 ||
       global_dt_mismatch != 0 ||
       global_elem_avg_mismatch != 0 ||
       global_integral_mismatch != 0 ||
       global_bounds_mismatch != 0)
   {
      ostringstream oss;
      oss << "ALE remap self-test case \"" << test_case.name << "\" failed: "
          << "mode=" << test_case.extent_mode
          << ", l2_order=" << test_case.extent_l2_order
          << ", not_found=" << remap_stats.global_not_found
          << ", expected_not_found=" << global_expected_not_found
          << ", skipped_elements=" << remap_stats.global_skipped_elements
          << ", expected_skipped_elements=" << global_expected_skipped_elements
          << ", clipped_values=" << remap_stats.global_clipped_values
          << ", qp_mismatch=" << global_qp_mismatch
          << ", extent_old_mismatch=" << global_qp_old_mismatch
          << ", dt_mismatch=" << global_dt_mismatch
          << ", elem_avg_mismatch=" << global_elem_avg_mismatch
          << ", integral_mismatch=" << global_integral_mismatch
          << ", bounds_mismatch=" << global_bounds_mismatch
          << ", max_abs_err=" << global_max_abs_err
          << ", max_rel_err=" << global_max_rel_err;
      throw runtime_error(oss.str());
   }

   const int myid = Mpi::WorldRank();
   if (myid == 0)
   {
      cout << "ALE remap self-test passed: case=" << test_case.name
           << ", qps=" << global_total_qps
           << ", reactions=" << nr
           << ", not_found=" << remap_stats.global_not_found
           << ", skipped_elements=" << remap_stats.global_skipped_elements
           << ", clipped_values=" << remap_stats.global_clipped_values
           << ", max_abs_err=" << global_max_abs_err
           << ", max_rel_err=" << global_max_rel_err << endl;
   }
}

int RunAleRemapSelfTest(const DriverParams &params)
{
   const int myid = Mpi::WorldRank();

   Device device("cpu");
   if (myid == 0) { device.Print(); }

   TACOTMaterial material;
   material.LoadFromYaml(params.material_file);

   unique_ptr<Mesh> mesh = make_unique<Mesh>(params.mesh_file.c_str(), 1, 1);
   if (mesh->Dimension() != 2)
   {
      throw runtime_error("ALE remap self-test requires a 2D mesh.");
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
   if (!pmesh->GetNodes())
   {
      pmesh->SetCurvature(1, false, pmesh->SpaceDimension(), Ordering::byVDIM);
   }

   const Bounds bounds = GetGlobalBounds(*pmesh);
   H1_FECollection fec(params.order, 2);
   ParFiniteElementSpace fes_T(pmesh.get(), &fec);

   auto *ale_vector_fes =
      dynamic_cast<ParFiniteElementSpace *>(pmesh->GetNodes()->FESpace());
   MFEM_VERIFY(ale_vector_fes != nullptr,
               "ALE remap self-test requires a nodal ParFiniteElementSpace.");
   auto *reference_nodes =
      dynamic_cast<ParGridFunction *>(pmesh->GetNodes());
   MFEM_VERIFY(reference_nodes != nullptr,
               "ALE remap self-test requires nodal mesh coordinates.");

   const double lx = bounds.xmax - bounds.xmin;
   vector<AleRemapSelfTestCase> cases = {
      {"nearest_qp_legacy", "nearest_qp", 1, 1, false, 0.0, 0.0,
       true, false, false, false, false},
      {"l2_point_eval_p0", "l2_point_eval", 0, 0, false, 0.0, 0.0,
       true, false, false, false, false},
      {"l2_point_eval_p1", "l2_point_eval", 1, 1, false, 0.0, 0.0,
       true, false, false, false, false},
      {"l2_point_eval_p2", "l2_point_eval", 2, 2, false, 0.0, 0.0,
       true, false, false, false, false},
      {"h1_point_eval_p1", "h1_point_eval", 1, 1, false, 0.0, 0.0,
       true, false, false, false, false},
      {"h1_point_eval_p2", "h1_point_eval", 2, 2, false, 0.0, 0.0,
       true, false, false, false, false},
      {"l2_point_eval_missing_points", "l2_point_eval", 1, 1, true, 0.6 * lx, 0.0,
       true, false, true, false, false},
      {"h1_point_eval_missing_points", "h1_point_eval", 1, 1, true, 0.6 * lx, 0.0,
       true, false, true, false, false},
      {"l2_conservative_p0_exact", "l2_conservative", 0, 0, false, 0.0, 0.0,
       true, false, false, false, false},
      {"l2_conservative_p1_exact", "l2_conservative", 1, 1, false, 0.0, 0.0,
       true, false, false, false, false},
      {"l2_conservative_p2_exact", "l2_conservative", 2, 2, false, 0.0, 0.0,
       true, false, false, false, false},
      {"l2_conservative_integral_p1_from_p2", "l2_conservative", 1, 2, false, 0.0, 0.0,
       false, true, false, false, false},
      {"l2_conservative_missing_points", "l2_conservative", 1, 1, true, 0.6 * lx, 0.0,
       true, false, true, false, false},
      {"l2_conservative_clipping", "l2_conservative", 2, 1, false, 0.0, 0.0,
       false, false, false, true, true}
   };

   for (const AleRemapSelfTestCase &test_case : cases)
   {
      RunAleRemapSelfTestCase(params,
                              bounds,
                              material,
                              fes_T,
                              *reference_nodes,
                              *ale_vector_fes,
                              test_case);
   }
   return 0;
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const int myid = Mpi::WorldRank();
   int world_size = 1;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);

   string input_file = "Input/input_ablation_case2_2.yaml";
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

   if (params.remap_self_test.enabled)
   {
      try
      {
         return RunAleRemapSelfTest(params);
      }
      catch (const exception &e)
      {
         if (myid == 0) { cerr << "Error: " << e.what() << endl; }
         return 3;
      }
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
      const bool use_recession_history =
         (params.moving_mesh && params.top_recession_bc == "recession_file");
      TopTemperatureSchedule top_temperature_schedule;
      bool use_top_temperature_schedule = false;
      TopRecessionSchedule top_recession_schedule;

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
      if (use_recession_history)
      {
         top_recession_schedule.LoadFromFile(EffectiveTopRecessionFile(params));
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
      L2_FECollection dg1_l2_fec(1, 2);
      ParFiniteElementSpace fes_dg1_diag(pmesh.get(), &dg1_l2_fec);

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
      T = 300.0;
      p = bc_schedule.Eval(0.0).p_w;
      T_old = T;
      p_old = p;

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
      }

      const int quad_order = max(2, 2 * params.order + 2);
      ValidateAleRemapExtentReconstructionConfig(params, fes_T, quad_order);

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
      surface_model.disable_bprime_c = params.disable_bprime_c;
      surface_model.pato_compat_mode = params.pato_compat_mode;

      JacobianCheckOptions jac_check_opts;
      jac_check_opts.enable = params.jacobian_check;
      jac_check_opts.abs_tol = params.jacobian_check_abs_tol;
      jac_check_opts.rel_tol = params.jacobian_check_rel_tol;

      unique_ptr<MeshRecessionHandler> recession_handler;
      Vector top_recession_velocity_true;
      Array<int> top_recession_tdofs;
      unique_ptr<ParGridFunction> ale_displacement;
      unique_ptr<ParGridFunction> ale_displacement_old;
      unique_ptr<ParGridFunction> ale_jacobian_gf;
      unique_ptr<ParGridFunction> recession_gf;
      unique_ptr<AleRemapWorkspace> ale_remap_workspace;
      unique_ptr<AlePointLocator2D> ale_remap_old_locator;
      double diagnostic_initial_min_quality = 1.0;
      const bool ale_mass_active =
         (params.ale_enabled && params.ale_mass_enabled && params.moving_mesh);
      const bool ale_energy_solid_active =
         (params.ale_enabled && params.ale_energy_enabled &&
          params.ale_energy_solid_enabled && params.moving_mesh);
      const bool ale_energy_gas_active =
         (params.ale_enabled && params.ale_energy_enabled &&
          params.ale_energy_gas_enabled && params.moving_mesh);
      if (params.moving_mesh)
      {
         RecessionConfig rec_cfg;
         rec_cfg.bdr_attr_top = params.bdr_attr_top;
         rec_cfg.bdr_attr_bottom = params.bdr_attr_bottom;
         rec_cfg.bdr_attr_sides = params.bdr_attr_sides;
         rec_cfg.mesh_smoothing_model = params.mesh_smoothing_model;
         rec_cfg.max_step_recession = params.max_step_recession;
         rec_cfg.min_quality_ratio = params.min_quality_ratio;
         recession_handler = make_unique<MeshRecessionHandler>(*pmesh, rec_cfg);
         top_recession_velocity_true.SetSize(recession_handler->ScalarSpace().TrueVSize());
         top_recession_velocity_true = 0.0;
         Array<int> top_marker(pmesh->bdr_attributes.Max());
         top_marker = 0;
         if (params.bdr_attr_top >= 1 && params.bdr_attr_top <= top_marker.Size())
         {
            top_marker[params.bdr_attr_top - 1] = 1;
         }
         recession_handler->ScalarSpace().GetEssentialTrueDofs(
            top_marker, top_recession_tdofs);

         auto *ale_vector_fes =
            dynamic_cast<ParFiniteElementSpace *>(pmesh->GetNodes()->FESpace());
         MFEM_VERIFY(ale_vector_fes != nullptr,
                     "Moving-mesh ALE output requires nodal ParFiniteElementSpace.");
         ale_displacement = make_unique<ParGridFunction>(ale_vector_fes);
         ale_displacement_old = make_unique<ParGridFunction>(ale_vector_fes);
         *ale_displacement = 0.0;
         *ale_displacement_old = 0.0;

         ale_jacobian_gf =
            make_unique<ParGridFunction>(
               const_cast<ParFiniteElementSpace *>(&recession_handler->ScalarSpace()));
         recession_gf =
            make_unique<ParGridFunction>(
               const_cast<ParFiniteElementSpace *>(&recession_handler->ScalarSpace()));
         *ale_jacobian_gf = 1.0;
         *recession_gf = 0.0;

         diagnostic_initial_min_quality = ComputeMinAleElementQuality(*pmesh, nullptr);
         if (!(diagnostic_initial_min_quality > 0.0))
         {
            throw runtime_error(
               "Invalid initial ALE mesh quality.");
         }

         ale_remap_workspace = make_unique<AleRemapWorkspace>();
         ale_remap_old_locator = make_unique<AlePointLocator2D>(*pmesh, fes_T);
      }

      auto *tp_integrator =
         new AblationTPIntegrator(material,
                                  state_manager,
                                  T_old,
                                  p_old,
                                  quad_order,
                                  gravity,
                                  ale_mass_active,
                                  ale_energy_solid_active,
                                  ale_energy_gas_active,
                                  jac_check_opts);
      tp_integrator->SetAleFields(ale_displacement_old.get(),
                                  ale_displacement.get(),
                                  recession_handler ?
                                     &recession_handler->MeshVelocity() :
                                     nullptr);
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
                                               nullptr);
         surf_integrator->SetAleDisplacement(ale_displacement.get());
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
      linear_cfg.use_block_matnest = params.petsc_use_matnest;

      newton_utils::PetscNewtonSolver newton_solver(MPI_COMM_WORLD,
                                                    newton_cfg,
                                                    linear_cfg);

      Vector x(block_true_offsets.Last());
      BlockVector xb(x, block_true_offsets);
      T.GetTrueDofs(xb.GetBlock(0));
      p.GetTrueDofs(xb.GetBlock(1));

      ParGridFunction tau_gf(&fes_diag), rho_s_gf(&fes_dg1_diag), pi_total_gf(&fes_diag), mdot_g_gf(&fes_diag);
      ParGridFunction degree_char_gf(&fes_diag), char_density_fraction_gf(&fes_diag);
      QuadratureDiagnosticFields qdiag_fields;
      vector<unique_ptr<ParGridFunction>> extent_gf;
      vector<string> extent_field_names;
      extent_gf.reserve(state_manager.NumReactions());
      extent_field_names.reserve(state_manager.NumReactions());
      for (int r = 0; r < state_manager.NumReactions(); ++r)
      {
         extent_gf.emplace_back(make_unique<ParGridFunction>(&fes_dg1_diag));
         extent_field_names.push_back("X" + to_string(r + 1));
      }
      ApplyElementScalar(fes_diag, state_manager.TauElement(), tau_gf);
      ProjectRhoToDG1Field(fes_T,
                           fes_p,
                           fes_dg1_diag,
                           T,
                           p,
                           material,
                           state_manager,
                           quad_order,
                           rho_s_gf);
      ApplyElementScalar(fes_diag, state_manager.PiElement(), pi_total_gf);
      ApplyElementScalar(fes_diag, state_manager.MdotElement(), mdot_g_gf);
      ApplyElementScalar(fes_diag, state_manager.DegreeCharElement(), degree_char_gf);
      ApplyElementScalar(fes_diag, state_manager.CharDensityFractionElement(), char_density_fraction_gf);
      ProjectExtentsToDG1Fields(fes_T,
                                fes_dg1_diag,
                                state_manager,
                                quad_order,
                                extent_gf);

      std::error_code ec;
      filesystem::create_directories(params.output_path, ec);
      if (ec)
      {
         throw runtime_error("Failed to create output path: " + params.output_path +
                             " (" + ec.message() + ")");
      }

      ofstream mass_csv;
      if (myid == 0)
      {
         mass_csv.open(filesystem::path(params.output_path) / params.mass_csv);

         if (!mass_csv)
         {
            throw runtime_error("Failed to open mass_metrics.csv.");
         }
         mass_csv << "time,m_dot_g_surf,m_dot_g_centerline,m_dot_c,recession\n";
         mass_csv << setprecision(16);
         mass_csv.flush();
      }

      unique_ptr<AleJacobianCoefficient> ale_jacobian_coeff;
      if (ale_displacement)
      {
         ale_jacobian_coeff =
            make_unique<AleJacobianCoefficient>(*ale_displacement);
      }

      ParaViewDataCollection paraview_dc(params.collection_name.c_str(), pmesh.get());
      double recession_total = 0.0;
      if (params.save_paraview)
      {
         InitializeQuadratureDiagnosticFields(*pmesh,
                                              quad_order,
                                              state_manager.NumReactions(),
                                              qdiag_fields);

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
         if (recession_handler)
         {
            auto *mesh_vel =
               const_cast<ParGridFunction *>(&recession_handler->MeshVelocity());
            paraview_dc.RegisterField("mesh_velocity", mesh_vel);
         }
         if (ale_displacement && ale_jacobian_gf && recession_gf)
         {
            paraview_dc.RegisterField("ale_displacement", ale_displacement.get());
            paraview_dc.RegisterField("ale_jacobian", ale_jacobian_gf.get());
            paraview_dc.RegisterField("recession", recession_gf.get());
         }
         for (int r = 0; r < state_manager.NumReactions(); ++r)
         {
            paraview_dc.RegisterField(extent_field_names[r].c_str(), extent_gf[r].get());
         }
         paraview_dc.RegisterQField("tau_qp", qdiag_fields.tau_qf.get());
         paraview_dc.RegisterQField("rho_s_qp", qdiag_fields.rho_s_qf.get());
         paraview_dc.RegisterQField("gas_density_qp", qdiag_fields.gas_density_qf.get());
         paraview_dc.RegisterQField("mobility_qp", qdiag_fields.mobility_qf.get());
         paraview_dc.RegisterQField("pi_total_qp", qdiag_fields.pi_total_qf.get());
         paraview_dc.RegisterQField("m_dot_g_qp", qdiag_fields.m_dot_g_qf.get());
         paraview_dc.RegisterQField("degree_char_qp",
                                    qdiag_fields.degree_char_qf.get());
         paraview_dc.RegisterQField("char_density_fraction_qp",
                                    qdiag_fields.char_density_fraction_qf.get());
         if (ale_displacement)
         {
            paraview_dc.RegisterQField("ale_displacement_qp",
                                       qdiag_fields.ale_displacement_qf.get());
         }
         for (int r = 0; r < state_manager.NumReactions(); ++r)
         {
            paraview_dc.RegisterQField(
               qdiag_fields.extent_field_names[static_cast<size_t>(r)].c_str(),
               qdiag_fields.extent_qf[static_cast<size_t>(r)].get());
         }
      }

      auto write_outputs = [&](const int step, const double time)
      {
         const bool save_paraview_now =
            params.save_paraview && (step % params.output_every == 0);

         if (save_paraview_now)
         {
            ApplyElementScalar(fes_diag, state_manager.TauElement(), tau_gf);
            ProjectRhoToDG1Field(fes_T,
                                 fes_p,
                                 fes_dg1_diag,
                                 T,
                                 p,
                                 material,
                                 state_manager,
                                 quad_order,
                                 rho_s_gf);
            ApplyElementScalar(fes_diag, state_manager.PiElement(), pi_total_gf);
            ApplyElementScalar(fes_diag, state_manager.MdotElement(), mdot_g_gf);
            ApplyElementScalar(fes_diag, state_manager.DegreeCharElement(), degree_char_gf);
            ApplyElementScalar(fes_diag, state_manager.CharDensityFractionElement(),
                               char_density_fraction_gf);
            ProjectExtentsToDG1Fields(fes_T,
                                      fes_dg1_diag,
                                      state_manager,
                                      quad_order,
                                      extent_gf);
            UpdateQuadratureDiagnosticFields(fes_T,
                                             fes_p,
                                             T,
                                             p,
                                             ale_displacement.get(),
                                             material,
                                             state_manager,
                                             quad_order,
                                             qdiag_fields);

            if (ale_displacement && ale_jacobian_gf && recession_gf)
            {
               ale_jacobian_gf->ProjectCoefficient(*ale_jacobian_coeff);
               *recession_gf = recession_total;
            }
         }

         const SurfaceBoundaryDiagnostics bdiag =
            ComputeTopBoundaryDiagnostics(*pmesh,
                                          fes_T,
                                          fes_p,
                                          T,
                                          p,
                                          ale_displacement.get(),
                                          material,
                                          state_manager,
                                          bprime_table,
                                          bc_schedule,
                                          surface_model,
                                          gravity,
                                          quad_order,
                                          params.bdr_attr_top,
                                          xmid,
                                          time,
                                          true);
         const double mdot_surf = bdiag.m_dot_g_surf;

         if (myid == 0)
         {
            mass_csv << time << "," << mdot_surf << ","
                     << bdiag.m_dot_g_centerline << ","
                     << bdiag.m_dot_c_surf << ","
                     << recession_total << "\n";
            mass_csv.flush();
         }

         if (save_paraview_now)
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

         if (recession_handler)
         {
            *ale_displacement_old = *ale_displacement;

            if (use_recession_history)
            {
               const double v_rec_file =
                  top_recession_schedule.AverageRate(time_prev, time);
               top_recession_velocity_true = 0.0;
               for (int i = 0; i < top_recession_tdofs.Size(); ++i)
               {
                  const int tdof = top_recession_tdofs[i];
                  if (tdof >= 0 && tdof < top_recession_velocity_true.Size())
                  {
                     top_recession_velocity_true(tdof) = v_rec_file;
                  }
               }
            }
            else
            {
               AssembleTopBoundaryRecessionVelocity(*pmesh,
                                                    fes_T,
                                                    fes_p,
                                                    recession_handler->ScalarSpace(),
                                                    T,
                                                    p,
                                                    ale_displacement.get(),
                                                    material,
                                                    state_manager,
                                                    bprime_table,
                                                    bc_schedule,
                                                    surface_model,
                                                    gravity,
                                                    quad_order,
                                                    params.bdr_attr_top,
                                                    time,
                                                    params.recession_density_mode,
                                                    params.recession_density_constant,
                                                    top_recession_velocity_true);
            }

            const double applied_top_mean_velocity =
               ClampAndAverageTopRecessionVelocity(top_recession_tdofs,
                                                   dt_step,
                                                   params.max_step_recession,
                                                   top_recession_velocity_true,
                                                   pmesh->GetComm());

            RecessionStepInput rec_input;
            rec_input.dt = dt_step;
            rec_input.top_recession_velocity_true = &top_recession_velocity_true;
            recession_handler->PrepareAdvance(rec_input);

            *ale_displacement = *ale_displacement_old;
            ale_displacement->Add(dt_step, recession_handler->MeshVelocity());
            recession_total += applied_top_mean_velocity * dt_step;

            tp_integrator->SetAleFields(ale_displacement_old.get(),
                                        ale_displacement.get(),
                                        &recession_handler->MeshVelocity());

            if (ale_remap_old_locator && params.ale_remap_enabled)
            {
               RemapExtentsALE(state_manager,
                               *pmesh,
                               fes_T,
                               *ale_displacement_old,
                               *ale_displacement,
                               quad_order,
                               params.ale_remap_extent_mode,
                               params.ale_remap_extent_l2_order,
                               *ale_remap_old_locator,
                               *ale_remap_workspace);
            }

            const double diagnostic_min_quality =
               ComputeMinAleElementQuality(*pmesh, ale_displacement.get());
            if (!(diagnostic_min_quality > 0.0))
            {
               throw runtime_error(
                  "Mesh quality failure: non-positive element Jacobian detected.");
            }

            const double diagnostic_quality_ratio =
               diagnostic_min_quality / diagnostic_initial_min_quality;
            if (diagnostic_quality_ratio < params.min_quality_ratio)
            {
               throw runtime_error(
                  "Mesh quality ratio below configured minimum threshold.");
            }
         }

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
         const BPrimeTable::ClampStats clamp_stats = bprime_table.GetClampStats();

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

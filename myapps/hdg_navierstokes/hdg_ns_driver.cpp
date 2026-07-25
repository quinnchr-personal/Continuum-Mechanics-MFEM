#include "exasim_mesh.hpp"
#include "exasim_reference.hpp"
#include "hdg_newton.hpp"
#include "hdg_ns_operator.hpp"
#include "wall_post.hpp"

#include "mfem.hpp"
#include "yaml-cpp/yaml.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct DriverParams
{
   std::string case_name;
   std::string exasim_directory;
   std::array<int, 3> boundary_conditions;
   hdg_ns::NSParams physics;
   std::string av_mode;
   double av_lambda = 0.0;
   double av_c = 0.0;
   std::string init_mode;
   double damping_c = 0.0;
   hdg_ns::NewtonConfig newton;
   std::string linear_type;
   std::string output_path;
   std::string wall_csv;
   std::string comparison_report;
   int paraview_every = 0;
};

template <typename T>
T Required(const YAML::Node &node, const std::string &key)
{
   if (!node || !node[key])
   {
      throw std::runtime_error("missing required YAML key: " + key);
   }
   return node[key].as<T>();
}

DriverParams LoadParams(const std::string &path)
{
   const YAML::Node root = YAML::LoadFile(path);
   DriverParams params;
   params.case_name = Required<std::string>(root, "case_name");

   const YAML::Node mesh = root["mesh"];
   const std::string mesh_source =
      Required<std::string>(mesh, "source");
   if (mesh_source != "exasim")
   {
      throw std::runtime_error(
         "M2 sanity driver requires mesh.source=exasim");
   }
   params.exasim_directory =
      Required<std::string>(mesh, "exasim_dir");

   const YAML::Node bc = root["bc"];
   const std::vector<int> boundary_conditions =
      Required<std::vector<int>>(bc, "boundaryconditions");
   if (boundary_conditions.size() != 3)
   {
      throw std::runtime_error(
         "bc.boundaryconditions must contain three entries");
   }
   std::copy(boundary_conditions.begin(), boundary_conditions.end(),
             params.boundary_conditions.begin());

   const YAML::Node physics = root["physics"];
   const std::vector<double> mu =
      Required<std::vector<double>>(physics, "mu");
   if (mu.size() != 11)
   {
      throw std::runtime_error("physics.mu must contain 11 entries");
   }
   std::copy(mu.begin(), mu.end(), params.physics.mu);
   params.physics.tau = Required<double>(physics, "tau");

   const YAML::Node av = root["av"];
   params.av_mode = Required<std::string>(av, "mode");
   if (params.av_mode == "tanh")
   {
      params.av_lambda = Required<double>(av, "lambda");
      params.av_c = Required<double>(av, "c");
   }
   else if (params.av_mode != "file")
   {
      throw std::runtime_error("av.mode must be tanh or file");
   }

   const YAML::Node init = root["init"];
   params.init_mode = Required<std::string>(init, "mode");
   if (params.init_mode == "damped_freestream")
   {
      params.damping_c = Required<double>(init, "c");
   }
   else if (params.init_mode != "udg_file")
   {
      throw std::runtime_error(
         "init.mode must be damped_freestream or udg_file");
   }

   const YAML::Node newton = root["newton"];
   params.newton.max_iterations = Required<int>(newton, "max_iters");
   params.newton.tolerance = Required<double>(newton, "tol");

   const YAML::Node ptc = root["ptc"];
   params.newton.pseudo_transient = Required<bool>(ptc, "enabled");
   params.newton.initial_pseudo_time_step =
      Required<double>(ptc, "initial_dt");

   const YAML::Node linear = root["linear"];
   params.linear_type = Required<std::string>(linear, "type");
   if (params.linear_type != "petsc_direct")
   {
      throw std::runtime_error(
         "M2 implementation supports linear.type=petsc_direct");
   }

   const YAML::Node output = root["output"];
   params.output_path = Required<std::string>(output, "path");
   params.paraview_every = Required<int>(output, "paraview_every");
   if (output["wall_csv"])
   {
      params.wall_csv = output["wall_csv"].as<std::string>();
   }
   if (output["comparison_report"])
   {
      params.comparison_report =
         output["comparison_report"].as<std::string>();
   }
   return params;
}

std::string ReferenceCSVPath(const std::string &path)
{
   const std::size_t extension = path.rfind('.');
   if (extension == std::string::npos)
   {
      return path + "_exasim";
   }
   return path.substr(0, extension) + "_exasim" +
          path.substr(extension);
}

void PrintConfig(const DriverParams &params)
{
   std::cout << "HDG configuration:"
             << " case=" << params.case_name
             << " mesh=" << params.exasim_directory
             << " M=" << params.physics.mu[3]
             << " Re=" << params.physics.mu[1]
             << " av_mode=" << params.av_mode;
   if (params.av_mode == "tanh")
   {
      std::cout << " av=" << params.av_lambda
                << "*tanh(" << params.av_c << "*(r-1))";
   }
   std::cout << " init_mode=" << params.init_mode
             << " tau=" << params.physics.tau
             << " NewtonTol=" << params.newton.tolerance
             << " max_iters=" << params.newton.max_iterations
             << " PTC=" << (params.newton.pseudo_transient ? "on" : "off")
             << " initial_dt=" << params.newton.initial_pseudo_time_step
             << " linear=" << params.linear_type << '\n';
}

class ParaViewWriter
{
public:
   ParaViewWriter(const DriverParams &params,
                  hdg_ns::HDGNavierStokesOperator &op,
                  mfem::Mesh &mesh)
      : params_(params),
        op_(op),
        conservative_(
           const_cast<mfem::FiniteElementSpace *>(&op.VolumeSpace())),
        primitive_(
           const_cast<mfem::FiniteElementSpace *>(&op.VolumeSpace())),
        scalar_fec_(hdg_ns::kOrder, 2),
        scalar_fes_(&mesh, &scalar_fec_),
        rho_(&scalar_fes_),
        uvel_(&scalar_fes_),
        vvel_(&scalar_fes_),
        pressure_(&scalar_fes_),
        av_(&scalar_fes_),
        collection_(params.case_name, &mesh)
   {
      collection_.SetPrefixPath(params.output_path);
      collection_.SetLevelsOfDetail(4);
      collection_.SetDataFormat(mfem::VTKFormat::BINARY);
      collection_.SetHighOrderOutput(true);
      collection_.RegisterField("conservative", &conservative_);
      collection_.RegisterField("rho", &rho_);
      collection_.RegisterField("u", &uvel_);
      collection_.RegisterField("v", &vvel_);
      collection_.RegisterField("p", &pressure_);
      collection_.RegisterField("av", &av_);

      op_.FillArtificialViscosityGridFunction(av_);
   }

   void Save(int cycle, const hdg_ns::HDGState &state)
   {
      op_.FillConservativeGridFunction(state, conservative_);
      op_.FillPrimitiveGridFunction(state, primitive_);
      const int scalar_size = scalar_fes_.GetVSize();
      if (primitive_.Size() != 4 * scalar_size)
      {
         throw std::runtime_error(
            "unexpected byVDIM primitive GridFunction layout");
      }
      for (int i = 0; i < scalar_size; ++i)
      {
         rho_[i] = primitive_[4 * i];
         uvel_[i] = primitive_[4 * i + 1];
         vvel_[i] = primitive_[4 * i + 2];
         pressure_[i] = primitive_[4 * i + 3];
      }
      collection_.SetCycle(cycle);
      collection_.SetTime(static_cast<double>(cycle));
      collection_.Save();
   }

private:
   DriverParams params_;
   hdg_ns::HDGNavierStokesOperator &op_;
   mfem::GridFunction conservative_;
   mfem::GridFunction primitive_;
   mfem::L2_FECollection scalar_fec_;
   mfem::FiniteElementSpace scalar_fes_;
   mfem::GridFunction rho_, uvel_, vvel_, pressure_, av_;
   mfem::ParaViewDataCollection collection_;
};

} // namespace

int main(int argc, char *argv[])
{
   int exit_code = EXIT_SUCCESS;
   mfem::Mpi::Init(argc, argv);
   mfem::Hypre::Init();

   const char *input_file = "Input/input_m45_sanity.yaml";
   bool acceptance = false;
   mfem::OptionsParser args(argc, argv);
   args.AddOption(&input_file, "-i", "--input", "Input YAML deck.");
   args.AddOption(&acceptance, "-a", "--acceptance",
                  "-no-a", "--no-acceptance",
                  "Enforce the M2(e) positivity and symmetry gates.");
   args.Parse();
   if (!args.Good())
   {
      if (mfem::Mpi::Root()) { args.PrintUsage(std::cerr); }
      return EXIT_FAILURE;
   }

   mfem::MFEMInitializePetsc(
      &argc, &argv, "Input/petsc.opts", nullptr);
   try
   {
      if (mfem::Mpi::WorldSize() != 1)
      {
         throw std::runtime_error("HDG driver supports np=1 only");
      }
      const DriverParams params = LoadParams(input_file);
      std::cout << std::setprecision(17);
      PrintConfig(params);

      hdg_ns::ExasimMesh converted =
         hdg_ns::BuildExasimMesh(params.exasim_directory);
      hdg_ns::ExasimArray vdg;
      std::unique_ptr<hdg_ns::HDGNavierStokesOperator> op_pointer;
      if (params.av_mode == "file")
      {
         vdg = hdg_ns::ReadExasimArray(
            params.exasim_directory + "/vdg.bin");
         op_pointer =
            std::make_unique<hdg_ns::HDGNavierStokesOperator>(
               *converted.mesh, vdg, converted.orientations,
               params.physics, params.boundary_conditions);
      }
      else
      {
         const auto av_function = [params](const mfem::Vector &x)
         {
            return params.av_lambda *
                   std::tanh(params.av_c *
                             (std::hypot(x[0], x[1]) - 1.0));
         };
         op_pointer =
            std::make_unique<hdg_ns::HDGNavierStokesOperator>(
               *converted.mesh, converted.orientations, av_function,
               params.physics, params.boundary_conditions);
      }
      hdg_ns::HDGNavierStokesOperator &op = *op_pointer;

      hdg_ns::HDGState state;
      if (params.init_mode == "udg_file")
      {
         const hdg_ns::ExasimArray input_udg =
            hdg_ns::ReadExasimArray(
               params.exasim_directory + "/udg.bin");
         op.LoadExasimVolumeState(input_udg, false, state);
         op.InitializeTraceFromInterior(state);
      }
      else
      {
         const auto initial_condition = [params](
            const mfem::Vector &x, double value[4])
         {
            const double distance = std::hypot(x[0], x[1]) - 1.0;
            const double velocity = std::tanh(params.damping_c * distance);
            value[0] = params.physics.mu[4];
            value[1] = value[0] * velocity;
            value[2] = 0.0;
            value[3] =
               value[0] * params.physics.mu[8] *
                  ((params.physics.mu[10] / params.physics.mu[9] - 1.0) *
                      std::exp(-params.damping_c * distance) + 1.0) +
               0.5 * value[0] * velocity * velocity;
         };
         op.ProjectState(initial_condition, state);
      }

      ParaViewWriter writer(params, op, *converted.mesh);
      writer.Save(0, state);
      const hdg_ns::NewtonOutput output =
         [&params, &writer, &op](
            int iteration, const hdg_ns::HDGState &current, double residual)
      {
         std::cout << "Newton " << std::setw(3) << iteration
                   << " residual=" << residual
                   << " min_rho=" << op.MinimumDensity(current)
                   << " min_p=" << op.MinimumPressure(current)
                   << " y_sym=" << op.YSymmetryError(current)
                   << std::endl;
         if (params.paraview_every > 0 &&
             iteration > 0 &&
             iteration % params.paraview_every == 0)
         {
            writer.Save(iteration, current);
         }
      };
      const hdg_ns::NewtonReport report =
         hdg_ns::DampedNewtonSolve(op, state, params.newton, output);
      writer.Save(report.iterations + 1, state);

      const double minimum_density = op.MinimumDensity(state);
      const double minimum_pressure = op.MinimumPressure(state);
      const double symmetry_error = op.YSymmetryError(state);
      const bool mach8_baseline =
         params.av_mode == "file" && params.init_mode == "udg_file";
      int damped_steps = 0;
      double minimum_alpha = 1.0;
      for (const hdg_ns::NewtonIteration &iteration : report.history)
      {
         if (iteration.iteration > 0)
         {
            minimum_alpha = std::min(minimum_alpha, iteration.alpha);
            if (iteration.alpha < 1.0) { ++damped_steps; }
         }
      }
      std::cout << (mach8_baseline ? "M3(b)" : "M2(e)") << " result:"
                << " converged=" << (report.converged ? "yes" : "no")
                << " iterations=" << report.iterations
                << " residual=" << report.residual
                << " min_rho=" << minimum_density
                << " min_p=" << minimum_pressure
                << " y_symmetry=" << symmetry_error
                << " damped_steps=" << damped_steps
                << " min_alpha=" << minimum_alpha
                << " assembly_seconds=" << report.assembly_seconds
                << " linear_seconds=" << report.linear_solve_seconds
                << " total_seconds=" << report.total_seconds << '\n';

      if (acceptance)
      {
         if (!report.converged || report.residual > 1.0e-6)
         {
            throw std::runtime_error(
               (mach8_baseline ? "M3(b)" : "M2(e)") +
               std::string(" Newton did not converge to 1e-6"));
         }
         if (!mach8_baseline &&
             (!(minimum_density > 0.0) || !(minimum_pressure > 0.0)))
         {
            throw std::runtime_error("M2(e) fields are not positive");
         }
         if (!mach8_baseline && symmetry_error > 1.0e-8)
         {
            throw std::runtime_error(
               "M2(e) fields are not y-symmetric to 1e-8");
         }
         std::cout << (mach8_baseline ?
            "PASS M3(b) Mach 8 Newton from udg.bin IC:" :
            "PASS M2(e) M=4.5 Re=1e3 actual-geometry sanity:")
                   << " residual=" << report.residual
                   << " min_rho=" << minimum_density
                   << " min_p=" << minimum_pressure
                   << " y_symmetry=" << symmetry_error << '\n';
         if (mach8_baseline)
         {
            const hdg_ns::ExasimReferenceData reference =
               hdg_ns::ReadExasimReferenceData(
                  params.exasim_directory, converted);
            hdg_ns::HDGState reference_state;
            hdg_ns::LoadExasimReferenceState(
               reference, op, true, reference_state);
            const std::array<double, 4> relative_l2 =
               op.ComponentRelativeL2(state, reference_state);
            std::cout << "M3(c) field relative L2:"
                      << " rho=" << relative_l2[0]
                      << " rhou=" << relative_l2[1]
                      << " rhov=" << relative_l2[2]
                      << " rhoE=" << relative_l2[3] << '\n';
            for (double error : relative_l2)
            {
               if (error > 1.0e-5)
               {
                  throw std::runtime_error(
                     "M3(c) field relative L2 exceeds 1e-5");
               }
            }
            std::cout << "PASS M3(c) field relative L2 comparison\n";

            if (params.wall_csv.empty() ||
                params.comparison_report.empty())
            {
               throw std::runtime_error(
                  "M3 output requires wall_csv and comparison_report");
            }
            const std::vector<hdg_ns::WallSample> wall =
               hdg_ns::ComputeWallSamples(
                  *converted.mesh, op, state, params.physics);
            const std::vector<hdg_ns::WallSample> reference_wall =
               hdg_ns::ComputeWallSamples(
                  *converted.mesh, op, reference_state, params.physics);
            const std::string reference_csv =
               ReferenceCSVPath(params.wall_csv);
            hdg_ns::WriteWallCSV(params.wall_csv, wall);
            hdg_ns::WriteWallCSV(reference_csv, reference_wall);

            const hdg_ns::ShockStandoff shock =
               hdg_ns::ComputeShockStandoff(
                  *converted.mesh, op, state);
            const hdg_ns::ShockStandoff reference_shock =
               hdg_ns::ComputeShockStandoff(
                  *converted.mesh, op, reference_state);
            const hdg_ns::M3Comparison comparison =
               hdg_ns::CompareWallAndShock(
                  wall, reference_wall, shock, reference_shock);
            std::cout << "M3(d) wall/shock comparison:"
                      << " Cp_max_rel="
                      << comparison.cp_max_relative_difference
                      << " Fint_max_rel="
                      << comparison.heat_flux_max_relative_difference
                      << " shock_hdg=" << shock.distance
                      << " shock_exasim=" << reference_shock.distance
                      << " shock_abs_diff="
                      << comparison.shock_standoff_difference
                      << " radial_cell="
                      << reference_shock.radial_cell_width
                      << " csv_hdg=" << params.wall_csv
                      << " csv_exasim=" << reference_csv << '\n';
            if (comparison.cp_max_relative_difference > 1.0e-4)
            {
               throw std::runtime_error(
                  "M3(d) wall Cp difference exceeds 1e-4");
            }
            if (comparison.heat_flux_max_relative_difference > 1.0e-3)
            {
               throw std::runtime_error(
                  "M3(d) wall Fint heat-flux difference exceeds 1e-3");
            }
            if (comparison.shock_standoff_difference >
                reference_shock.radial_cell_width)
            {
               throw std::runtime_error(
                  "M3(d) shock standoff differs by more than one radial cell");
            }
            hdg_ns::WriteM3ComparisonReport(
               params.comparison_report, relative_l2, comparison,
               shock, reference_shock);
            std::cout << "PASS M3(d) wall Cp, Fint heat flux, and shock"
                      << " standoff comparison; report="
                      << params.comparison_report << '\n';
         }
      }
   }
   catch (const std::exception &error)
   {
      std::cerr << "FAIL hdg_ns_driver: " << error.what() << '\n';
      exit_code = EXIT_FAILURE;
   }
   mfem::MFEMFinalizePetsc();
   return exit_code;
}

#include "exasim_mesh.hpp"
#include "hdg_newton.hpp"
#include "hdg_ns_operator.hpp"

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
   double av_lambda = 0.0;
   double av_c = 0.0;
   double damping_c = 0.0;
   hdg_ns::NewtonConfig newton;
   std::string linear_type;
   std::string output_path;
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
   if (Required<std::string>(av, "mode") != "tanh")
   {
      throw std::runtime_error("M2 sanity driver requires av.mode=tanh");
   }
   params.av_lambda = Required<double>(av, "lambda");
   params.av_c = Required<double>(av, "c");

   const YAML::Node init = root["init"];
   if (Required<std::string>(init, "mode") != "damped_freestream")
   {
      throw std::runtime_error(
         "M2 sanity driver requires init.mode=damped_freestream");
   }
   params.damping_c = Required<double>(init, "c");

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
   return params;
}

void PrintConfig(const DriverParams &params)
{
   std::cout << "HDG M2 configuration:"
             << " case=" << params.case_name
             << " mesh=" << params.exasim_directory
             << " M=" << params.physics.mu[3]
             << " Re=" << params.physics.mu[1]
             << " av=" << params.av_lambda
             << "*tanh(" << params.av_c << "*(r-1))"
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

      mfem::FunctionCoefficient av_coefficient(
         [params](const mfem::Vector &x)
         {
            return params.av_lambda *
                   std::tanh(params.av_c *
                             (std::hypot(x[0], x[1]) - 1.0));
         });
      av_.ProjectCoefficient(av_coefficient);
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
         throw std::runtime_error("M2 driver supports np=1 only");
      }
      const DriverParams params = LoadParams(input_file);
      std::cout << std::setprecision(17);
      PrintConfig(params);

      hdg_ns::ExasimMesh converted =
         hdg_ns::BuildExasimMesh(params.exasim_directory);
      const auto av_function = [params](const mfem::Vector &x)
      {
         return params.av_lambda *
                std::tanh(params.av_c *
                          (std::hypot(x[0], x[1]) - 1.0));
      };
      hdg_ns::HDGNavierStokesOperator op(
         *converted.mesh, converted.orientations, av_function,
         params.physics, params.boundary_conditions);

      const auto initial_condition = [params](
         const mfem::Vector &x, double state[4])
      {
         const double distance = std::hypot(x[0], x[1]) - 1.0;
         const double velocity = std::tanh(params.damping_c * distance);
         state[0] = params.physics.mu[4];
         state[1] = state[0] * velocity;
         state[2] = 0.0;
         state[3] =
            state[0] * params.physics.mu[8] *
               ((params.physics.mu[10] / params.physics.mu[9] - 1.0) *
                   std::exp(-params.damping_c * distance) + 1.0) +
            0.5 * state[0] * velocity * velocity;
      };
      hdg_ns::HDGState state;
      op.ProjectState(initial_condition, state);

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
      std::cout << "M2(e) result:"
                << " converged=" << (report.converged ? "yes" : "no")
                << " iterations=" << report.iterations
                << " residual=" << report.residual
                << " min_rho=" << minimum_density
                << " min_p=" << minimum_pressure
                << " y_symmetry=" << symmetry_error << '\n';

      if (acceptance)
      {
         if (!report.converged || report.residual > 1.0e-6)
         {
            throw std::runtime_error(
               "M2(e) Newton did not converge to 1e-6");
         }
         if (!(minimum_density > 0.0) || !(minimum_pressure > 0.0))
         {
            throw std::runtime_error("M2(e) fields are not positive");
         }
         if (symmetry_error > 1.0e-8)
         {
            throw std::runtime_error(
               "M2(e) fields are not y-symmetric to 1e-8");
         }
         std::cout << "PASS M2(e) M=4.5 Re=1e3 actual-geometry sanity:"
                   << " residual=" << report.residual
                   << " min_rho=" << minimum_density
                   << " min_p=" << minimum_pressure
                   << " y_symmetry=" << symmetry_error << '\n';
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

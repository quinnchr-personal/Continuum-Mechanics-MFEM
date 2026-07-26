#include "exasim_mesh.hpp"
#include "exasim_reference.hpp"
#include "hdg_newton.hpp"
#include "hdg_ns_operator.hpp"
#include "wall_post.hpp"

#include "mfem.hpp"
#include "yaml-cpp/yaml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct ContinuationStage
{
   double lambda = 0.0;
   double c = 0.0;
   bool pseudo_transient = false;
   double initial_pseudo_time_step = 0.5;
};

struct DriverParams
{
   std::string case_name;
   std::string mesh_source;
   std::string exasim_directory;
   int analytic_nr = 0;
   int analytic_nc = 0;
   int analytic_order = hdg_ns::kOrder;
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
   bool continuation_enabled = false;
   std::vector<ContinuationStage> continuation_stages;
   std::string continuation_history_csv;
   std::string continuation_summary_csv;
   std::string reference_wall_csv;
   double reference_shock_standoff =
      std::numeric_limits<double>::quiet_NaN();
   bool comparison_gate = false;
   bool comparison_require_matching_coordinates = true;
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
   params.mesh_source = Required<std::string>(mesh, "source");
   if (params.mesh_source == "exasim")
   {
      params.exasim_directory =
         Required<std::string>(mesh, "exasim_dir");
   }
   else if (params.mesh_source == "analytic")
   {
      params.analytic_nr = Required<int>(mesh, "nr");
      params.analytic_nc = Required<int>(mesh, "nc");
      params.analytic_order = Required<int>(mesh, "order");
      if (params.analytic_nr <= 0 || params.analytic_nc <= 0 ||
          params.analytic_order != hdg_ns::kOrder)
      {
         throw std::runtime_error(
            "analytic mesh requires positive nr/nc and order=4");
      }
   }
   else
   {
      throw std::runtime_error(
         "mesh.source must be exasim or analytic");
   }

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
   if (physics["regularization"])
   {
      const std::string regularization =
         physics["regularization"].as<std::string>();
      if (regularization == "floors")
      {
         params.physics.regularized = true;
      }
      else if (regularization != "none")
      {
         throw std::runtime_error(
            "physics.regularization must be floors or none");
      }
   }

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

   const YAML::Node continuation = root["continuation"];
   if (continuation)
   {
      params.continuation_enabled =
         Required<bool>(continuation, "enabled");
      if (params.continuation_enabled)
      {
         const YAML::Node stages = continuation["stages"];
         if (!stages || !stages.IsSequence() || stages.size() == 0)
         {
            throw std::runtime_error(
               "enabled continuation requires a nonempty stages sequence");
         }
         for (const YAML::Node &stage_node : stages)
         {
            ContinuationStage stage;
            stage.lambda = Required<double>(stage_node, "lambda");
            stage.c = Required<double>(stage_node, "c");
            if (!(stage.lambda > 0.0) || !(stage.c > 0.0))
            {
               throw std::runtime_error(
                  "continuation lambda and c must be positive");
            }
            if (stage_node["ptc"])
            {
               const YAML::Node stage_ptc = stage_node["ptc"];
               stage.pseudo_transient =
                  Required<bool>(stage_ptc, "enabled");
               stage.initial_pseudo_time_step =
                  Required<double>(stage_ptc, "initial_dt");
               if (stage.pseudo_transient &&
                   !(stage.initial_pseudo_time_step > 0.0))
               {
                  throw std::runtime_error(
                     "enabled stage PTC requires initial_dt > 0");
               }
            }
            params.continuation_stages.push_back(stage);
         }
         params.continuation_history_csv =
            Required<std::string>(continuation, "history_csv");
         params.continuation_summary_csv =
            Required<std::string>(continuation, "summary_csv");
      }
   }

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

   const YAML::Node comparison = root["comparison"];
   if (comparison)
   {
      params.reference_wall_csv =
         Required<std::string>(comparison, "reference_wall_csv");
      params.reference_shock_standoff =
         Required<double>(comparison, "reference_shock_standoff");
      params.comparison_gate =
         Required<bool>(comparison, "gate");
      if (comparison["require_matching_coordinates"])
      {
         params.comparison_require_matching_coordinates =
            comparison["require_matching_coordinates"].as<bool>();
      }
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
             << " mesh_source=" << params.mesh_source;
   if (params.mesh_source == "exasim")
   {
      std::cout << " mesh=" << params.exasim_directory;
   }
   else
   {
      std::cout << " nr=" << params.analytic_nr
                << " nc=" << params.analytic_nc
                << " order=" << params.analytic_order;
   }
   std::cout
             << " M=" << params.physics.mu[3]
             << " Re=" << params.physics.mu[1]
             << " av_mode=" << params.av_mode;
   if (params.av_mode == "tanh")
   {
      std::cout << " av=" << params.av_lambda
                << "*tanh(" << params.av_c << "*(r-1))";
   }
   std::cout << " init_mode=" << params.init_mode
             << " regularization="
             << (params.physics.regularized ? "floors" : "none")
             << " tau=" << params.physics.tau
             << " NewtonTol=" << params.newton.tolerance
             << " max_iters=" << params.newton.max_iterations
             << " PTC=" << (params.newton.pseudo_transient ? "on" : "off")
             << " initial_dt=" << params.newton.initial_pseudo_time_step
             << " linear=" << params.linear_type
             << " continuation="
             << (params.continuation_enabled ? "on" : "off");
   if (params.continuation_enabled)
   {
      std::cout << " stages=" << params.continuation_stages.size();
   }
   std::cout << '\n';
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
      op_.FillArtificialViscosityGridFunction(av_);
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

hdg_ns::HDGNavierStokesOperator::ScalarFunction
ArtificialViscosity(double lambda, double c)
{
   return [lambda, c](const mfem::Vector &x)
   {
      const double distance = std::hypot(x[0], x[1]) - 1.0;
      return lambda * std::tanh(c * distance);
   };
}

std::vector<hdg_ns::ElementOrientation>
IdentityOrientations(int element_count)
{
   return std::vector<hdg_ns::ElementOrientation>(
      static_cast<std::size_t>(element_count));
}

struct StageResult
{
   ContinuationStage stage;
   hdg_ns::NewtonReport report;
   int damped_steps = 0;
   double minimum_alpha = 1.0;
};

struct BlowupDiagnostic
{
   double minimum_density = std::numeric_limits<double>::infinity();
   int minimum_density_element = -1;
   double minimum_density_x = 0.0;
   double minimum_density_y = 0.0;
   double minimum_pressure = std::numeric_limits<double>::infinity();
   int minimum_pressure_element = -1;
   double minimum_pressure_x = 0.0;
   double minimum_pressure_y = 0.0;
   double maximum_volume_residual = 0.0;
   int maximum_volume_element = -1;
   int maximum_volume_equation = -1;
   int maximum_volume_dof = -1;
   double maximum_volume_x = 0.0;
   double maximum_volume_y = 0.0;
   double maximum_trace_residual = 0.0;
   int maximum_trace_face = -1;
   int maximum_trace_equation = -1;
   int maximum_trace_dof = -1;
   double maximum_trace_x = 0.0;
   double maximum_trace_y = 0.0;
};

BlowupDiagnostic DiagnoseBlowup(
   mfem::Mesh &mesh, const hdg_ns::HDGNavierStokesOperator &op,
   const hdg_ns::HDGState &state, const hdg_ns::NSParams &params)
{
   BlowupDiagnostic diagnostic;
   mfem::Vector physical(2);
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      mfem::ElementTransformation *transformation =
         mesh.GetElementTransformation(element);
      for (int qpoint = 0;
           qpoint < op.VolumeRule().GetNPoints(); ++qpoint)
      {
         const mfem::IntegrationPoint &point =
            op.VolumeRule().IntPoint(qpoint);
         double uq[12];
         op.EvaluateElementState(state, element, point, uq);
         transformation->Transform(point, physical);
         if (uq[0] < diagnostic.minimum_density)
         {
            diagnostic.minimum_density = uq[0];
            diagnostic.minimum_density_element = element;
            diagnostic.minimum_density_x = physical[0];
            diagnostic.minimum_density_y = physical[1];
         }
         const double pressure = hdg_ns::Pressure(uq, params);
         if (pressure < diagnostic.minimum_pressure)
         {
            diagnostic.minimum_pressure = pressure;
            diagnostic.minimum_pressure_element = element;
            diagnostic.minimum_pressure_x = physical[0];
            diagnostic.minimum_pressure_y = physical[1];
         }
      }
   }

   const mfem::Vector &volume_residual = op.VolumeResidual();
   const mfem::IntegrationRule &volume_nodes =
      op.VolumeSpace().GetFE(0)->GetNodes();
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      for (int equation = 0; equation < 4; ++equation)
      {
         for (int dof = 0; dof < 25; ++dof)
         {
            const double magnitude = std::abs(
               volume_residual[
                  element * 100 + dof + 25 * equation]);
            if (magnitude > diagnostic.maximum_volume_residual)
            {
               diagnostic.maximum_volume_residual = magnitude;
               diagnostic.maximum_volume_element = element;
               diagnostic.maximum_volume_equation = equation;
               diagnostic.maximum_volume_dof = dof;
               mesh.GetElementTransformation(element)->Transform(
                  volume_nodes.IntPoint(dof), physical);
               diagnostic.maximum_volume_x = physical[0];
               diagnostic.maximum_volume_y = physical[1];
            }
         }
      }
   }

   const mfem::Vector &trace_residual = op.TraceResidual();
   const mfem::IntegrationRule &trace_nodes =
      op.TraceSpace().GetFaceElement(0)->GetNodes();
   mfem::Array<int> vdofs;
   for (int face = 0; face < mesh.GetNumFaces(); ++face)
   {
      op.TraceSpace().GetFaceVDofs(face, vdofs);
      for (int equation = 0; equation < 4; ++equation)
      {
         for (int dof = 0; dof < 5; ++dof)
         {
            const int vdof = vdofs[dof + 5 * equation];
            const double magnitude = std::abs(trace_residual[vdof]);
            if (magnitude > diagnostic.maximum_trace_residual)
            {
               diagnostic.maximum_trace_residual = magnitude;
               diagnostic.maximum_trace_face = face;
               diagnostic.maximum_trace_equation = equation;
               diagnostic.maximum_trace_dof = dof;
               mfem::FaceElementTransformations *transformation =
                  mesh.GetFaceElementTransformations(face, 31);
               transformation->Face->Transform(
                  trace_nodes.IntPoint(dof), physical);
               diagnostic.maximum_trace_x = physical[0];
               diagnostic.maximum_trace_y = physical[1];
            }
         }
      }
   }
   return diagnostic;
}

StageResult SummarizeStage(const ContinuationStage &stage,
                           hdg_ns::NewtonReport report)
{
   StageResult result;
   result.stage = stage;
   result.report = std::move(report);
   for (const hdg_ns::NewtonIteration &iteration : result.report.history)
   {
      if (iteration.iteration == 0) { continue; }
      result.minimum_alpha =
         std::min(result.minimum_alpha, iteration.alpha);
      if (iteration.alpha < 1.0) { ++result.damped_steps; }
   }
   return result;
}

void WriteContinuationCSVs(
   const DriverParams &params,
   const std::vector<StageResult> &stages)
{
   std::ofstream history(params.continuation_history_csv);
   if (!history)
   {
      throw std::runtime_error(
         "cannot open continuation history CSV: " +
         params.continuation_history_csv);
   }
   history << "stage,lambda,c,ptc_enabled,iteration,residual,alpha,dtau\n"
           << std::setprecision(16);
   for (std::size_t stage_index = 0;
        stage_index < stages.size(); ++stage_index)
   {
      const StageResult &stage = stages[stage_index];
      for (const hdg_ns::NewtonIteration &iteration :
           stage.report.history)
      {
         history << stage_index + 1 << ',' << stage.stage.lambda << ','
                 << stage.stage.c << ','
                 << (stage.stage.pseudo_transient ? 1 : 0) << ','
                 << iteration.iteration << ',' << iteration.residual << ','
                 << iteration.alpha << ','
                 << iteration.pseudo_time_step << '\n';
      }
   }
   if (!history)
   {
      throw std::runtime_error(
         "failed while writing continuation history CSV: " +
         params.continuation_history_csv);
   }

   std::ofstream summary(params.continuation_summary_csv);
   if (!summary)
   {
      throw std::runtime_error(
         "cannot open continuation summary CSV: " +
         params.continuation_summary_csv);
   }
   summary
      << "stage,lambda,c,ptc_enabled,converged,iterations,residual,"
         "damped_steps,min_alpha,assembly_seconds,linear_seconds,total_seconds,"
         "failure\n"
      << std::setprecision(16);
   for (std::size_t stage_index = 0;
        stage_index < stages.size(); ++stage_index)
   {
      const StageResult &stage = stages[stage_index];
      summary << stage_index + 1 << ',' << stage.stage.lambda << ','
              << stage.stage.c << ','
              << (stage.stage.pseudo_transient ? 1 : 0) << ','
              << (stage.report.converged ? 1 : 0) << ','
              << stage.report.iterations << ',' << stage.report.residual
              << ',' << stage.damped_steps << ',' << stage.minimum_alpha
              << ',' << stage.report.assembly_seconds << ','
              << stage.report.linear_solve_seconds << ','
              << stage.report.total_seconds << ','
              << stage.report.failure << '\n';
   }
   if (!summary)
   {
      throw std::runtime_error(
         "failed while writing continuation summary CSV: " +
         params.continuation_summary_csv);
   }
}

void WriteM4ComparisonReport(
   const DriverParams &params,
   const std::vector<StageResult> &stages,
   const hdg_ns::M3Comparison &comparison,
   const hdg_ns::ShockStandoff &shock)
{
   if (params.comparison_report.empty())
   {
      throw std::runtime_error(
         "continuation output requires comparison_report");
   }
   std::ofstream output(params.comparison_report);
   if (!output)
   {
      throw std::runtime_error(
         "cannot open M4 comparison report: " +
         params.comparison_report);
   }
   const double shock_relative_difference =
      comparison.shock_standoff_difference /
      std::max(1.0e-14, std::abs(params.reference_shock_standoff));
   output << std::setprecision(16)
          << "# M4 continuation comparison report\n\n"
          << "Mesh source: `" << params.mesh_source << "`\n\n"
          << "| Stage | lambda | c | PTC | Newton updates | Final residual"
             " | Damped steps | Minimum alpha |\n"
          << "|---:|---:|---:|:---:|---:|---:|---:|---:|\n";
   for (std::size_t index = 0; index < stages.size(); ++index)
   {
      const StageResult &stage = stages[index];
      output << "| " << index + 1 << " | " << stage.stage.lambda
             << " | " << stage.stage.c << " | "
             << (stage.stage.pseudo_transient ? "on" : "off")
             << " | " << stage.report.iterations << " | "
             << stage.report.residual << " | " << stage.damped_steps
             << " | " << stage.minimum_alpha << " |\n";
   }
   output << "\n| Quantity | Result | M4 converted-mesh gate |\n"
          << "|---|---:|---:|\n"
          << "| wall Cp maximum relative difference | "
          << comparison.cp_max_relative_difference << " | report |\n"
          << "| wall Fint heat-flux maximum relative difference | "
          << comparison.heat_flux_max_relative_difference
          << " | <= 1e-2 |\n"
          << "| wall-coordinate maximum absolute difference | "
          << comparison.wall_coordinate_maximum_difference
          << " | geometry diagnostic |\n"
          << "| computed shock standoff | " << shock.distance
          << " | report |\n"
          << "| reference shock standoff | "
          << params.reference_shock_standoff << " | report |\n"
          << "| shock-standoff relative difference | "
          << shock_relative_difference << " | <= 1e-2 |\n";
   if (!output)
   {
      throw std::runtime_error(
         "failed while writing M4 comparison report: " +
         params.comparison_report);
   }
}

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

      hdg_ns::ExasimMesh converted;
      std::unique_ptr<mfem::Mesh> analytic_mesh;
      mfem::Mesh *mesh = nullptr;
      std::vector<hdg_ns::ElementOrientation> orientations;
      if (params.mesh_source == "exasim")
      {
         converted =
            hdg_ns::BuildExasimMesh(params.exasim_directory);
         mesh = converted.mesh.get();
         orientations = converted.orientations;
      }
      else
      {
         analytic_mesh = hdg_ns::BuildAnalyticMesh(
            params.analytic_nr, params.analytic_nc,
            params.analytic_order);
         mesh = analytic_mesh.get();
         orientations = IdentityOrientations(mesh->GetNE());
      }

      hdg_ns::ExasimArray vdg;
      std::unique_ptr<hdg_ns::HDGNavierStokesOperator> op_pointer;
      if (params.av_mode == "file")
      {
         if (params.mesh_source != "exasim" ||
             params.continuation_enabled)
         {
            throw std::runtime_error(
               "av.mode=file requires a non-continuation Exasim mesh run");
         }
         vdg = hdg_ns::ReadExasimArray(
            params.exasim_directory + "/vdg.bin");
         op_pointer =
            std::make_unique<hdg_ns::HDGNavierStokesOperator>(
               *mesh, vdg, orientations,
               params.physics, params.boundary_conditions);
      }
      else
      {
         double initial_lambda = params.av_lambda;
         double initial_c = params.av_c;
         if (params.continuation_enabled)
         {
            initial_lambda = params.continuation_stages.front().lambda;
            initial_c = params.continuation_stages.front().c;
         }
         op_pointer =
            std::make_unique<hdg_ns::HDGNavierStokesOperator>(
               *mesh, orientations,
               ArtificialViscosity(initial_lambda, initial_c),
               params.physics, params.boundary_conditions);
      }
      hdg_ns::HDGNavierStokesOperator &op = *op_pointer;

      hdg_ns::HDGState state;
      if (params.init_mode == "udg_file")
      {
         if (params.mesh_source != "exasim" ||
             params.continuation_enabled)
         {
            throw std::runtime_error(
               "init.mode=udg_file requires a non-continuation Exasim run");
         }
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
            value[0] = 1.0;
            value[1] = velocity;
            value[2] = 0.0;
            value[3] =
               params.physics.TinfFlux() *
                  ((params.physics.mu[10] / params.physics.mu[9] - 1.0) *
                      std::exp(-params.damping_c * distance) + 1.0) +
               0.5 * velocity * velocity;
         };
         op.ProjectState(initial_condition, state);
         // Match the restart semantics used by M3 and Exasim: the trace is
         // taken from the lower-index side and q is then recomputed.
         op.InitializeTraceFromInterior(state);
      }

      ParaViewWriter writer(params, op, *mesh);
      hdg_ns::NewtonReport report;
      std::vector<StageResult> stage_results;
      if (params.continuation_enabled)
      {
         if (params.init_mode != "damped_freestream")
         {
            throw std::runtime_error(
               "continuation requires init.mode=damped_freestream");
         }
         for (std::size_t stage_index = 0;
              stage_index < params.continuation_stages.size();
              ++stage_index)
         {
            const ContinuationStage &stage =
               params.continuation_stages[stage_index];
            op.SetArtificialViscosity(
               ArtificialViscosity(stage.lambda, stage.c));
            if (stage_index > 0)
            {
               // Exasim restart semantics: each stage of pdeapp_ns.m is a
               // fresh run, so the trace restarts from the one-sided
               // interior trace and q is recomputed (the damped-freestream
               // branch already did this for stage 1).
               op.InitializeTraceFromInterior(state);
            }
            const int cycle_base =
               static_cast<int>(1000 * stage_index);
            writer.Save(cycle_base, state);
            hdg_ns::NewtonConfig stage_config = params.newton;
            stage_config.pseudo_transient = stage.pseudo_transient;
            stage_config.initial_pseudo_time_step =
               stage.initial_pseudo_time_step;
            const hdg_ns::NewtonOutput output =
               [&params, &writer, &op, stage_index, cycle_base](
                  int iteration, const hdg_ns::HDGState &current,
                  double residual)
            {
               std::cout << "Stage " << stage_index + 1
                         << " Newton " << std::setw(3) << iteration
                         << " residual=" << residual
                         << " min_rho=" << op.MinimumDensity(current)
                         << " min_p=" << op.MinimumPressure(current)
                         << " y_sym=" << op.YSymmetryError(current)
                         << std::endl;
               if (params.paraview_every > 0 &&
                   iteration > 0 &&
                   iteration % params.paraview_every == 0)
               {
                  writer.Save(cycle_base + iteration, current);
               }
            };
            report = hdg_ns::DampedNewtonSolve(
               op, state, stage_config, output);
            stage_results.push_back(
               SummarizeStage(stage, std::move(report)));
            const StageResult &result = stage_results.back();
            writer.Save(cycle_base + result.report.iterations + 1, state);
            WriteContinuationCSVs(params, stage_results);
            std::cout << "M4 stage " << stage_index + 1
                      << " result: lambda=" << stage.lambda
                      << " c=" << stage.c
                      << " PTC="
                      << (stage.pseudo_transient ? "on" : "off")
                      << " converged="
                      << (result.report.converged ? "yes" : "no")
                      << " iterations=" << result.report.iterations
                      << " residual=" << result.report.residual
                      << " damped_steps=" << result.damped_steps
                      << " min_alpha=" << result.minimum_alpha
                      << " assembly_seconds="
                      << result.report.assembly_seconds
                      << " linear_seconds="
                      << result.report.linear_solve_seconds
                      << " total_seconds="
                      << result.report.total_seconds;
            if (!result.report.failure.empty())
            {
               std::cout << " failure=" << result.report.failure;
            }
            std::cout << '\n';
            if (!result.report.converged ||
                result.report.residual > params.newton.tolerance)
            {
               const BlowupDiagnostic diagnostic =
                  DiagnoseBlowup(*mesh, op, state, params.physics);
               std::cout
                  << "M4 failure localization:"
                  << " min_rho=" << diagnostic.minimum_density
                  << " min_rho_element="
                  << diagnostic.minimum_density_element
                  << " min_rho_xy=(" << diagnostic.minimum_density_x
                  << ',' << diagnostic.minimum_density_y << ')'
                  << " min_p=" << diagnostic.minimum_pressure
                  << " min_p_element="
                  << diagnostic.minimum_pressure_element
                  << " min_p_xy=(" << diagnostic.minimum_pressure_x
                  << ',' << diagnostic.minimum_pressure_y << ')'
                  << " max_Ru=" << diagnostic.maximum_volume_residual
                  << " max_Ru_element="
                  << diagnostic.maximum_volume_element
                  << " max_Ru_equation="
                  << diagnostic.maximum_volume_equation
                  << " max_Ru_dof="
                  << diagnostic.maximum_volume_dof
                  << " max_Ru_xy=(" << diagnostic.maximum_volume_x
                  << ',' << diagnostic.maximum_volume_y << ')'
                  << " max_Rh=" << diagnostic.maximum_trace_residual
                  << " max_Rh_face="
                  << diagnostic.maximum_trace_face
                  << " max_Rh_equation="
                  << diagnostic.maximum_trace_equation
                  << " max_Rh_dof="
                  << diagnostic.maximum_trace_dof
                  << " max_Rh_xy=(" << diagnostic.maximum_trace_x
                  << ',' << diagnostic.maximum_trace_y << ')'
                  << '\n';
               throw std::runtime_error(
                  "M4 continuation stage " +
                  std::to_string(stage_index + 1) +
                  " did not converge; history archived in " +
                  params.continuation_history_csv);
            }
         }
         report = stage_results.back().report;
      }
      else
      {
         writer.Save(0, state);
         const hdg_ns::NewtonOutput output =
            [&params, &writer, &op](
               int iteration, const hdg_ns::HDGState &current,
               double residual)
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
         report =
            hdg_ns::DampedNewtonSolve(op, state, params.newton, output);
         writer.Save(report.iterations + 1, state);
      }

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
      if (!params.continuation_enabled)
      {
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
      }

      if (params.continuation_enabled)
      {
         if (params.wall_csv.empty() ||
             params.reference_wall_csv.empty() ||
             !std::isfinite(params.reference_shock_standoff))
         {
            throw std::runtime_error(
               "M4 comparison requires output.wall_csv and comparison"
               " reference wall/shock data");
         }
         const std::vector<hdg_ns::WallSample> wall =
            hdg_ns::ComputeWallSamples(
               *mesh, op, state, params.physics);
         hdg_ns::WriteWallCSV(params.wall_csv, wall);
         const std::vector<hdg_ns::WallSample> reference_wall =
            hdg_ns::ReadWallCSV(params.reference_wall_csv);
         const hdg_ns::ShockStandoff shock =
            hdg_ns::ComputeShockStandoff(*mesh, op, state);
         hdg_ns::ShockStandoff reference_shock;
         reference_shock.distance = params.reference_shock_standoff;
         reference_shock.radius =
            1.0 + params.reference_shock_standoff;
         const hdg_ns::M3Comparison comparison =
            hdg_ns::CompareWallAndShock(
               wall, reference_wall, shock, reference_shock,
               params.comparison_require_matching_coordinates);
         const double shock_relative_difference =
            comparison.shock_standoff_difference /
            std::max(1.0e-14,
                     std::abs(params.reference_shock_standoff));
         WriteM4ComparisonReport(
            params, stage_results, comparison, shock);
         std::cout << "M4 wall/shock comparison:"
                   << " Fint_max_rel="
                   << comparison.heat_flux_max_relative_difference
                   << " Cp_max_rel="
                   << comparison.cp_max_relative_difference
                   << " coordinate_max_abs="
                   << comparison.wall_coordinate_maximum_difference
                   << " shock=" << shock.distance
                   << " reference_shock="
                   << params.reference_shock_standoff
                   << " shock_rel=" << shock_relative_difference
                   << " wall_csv=" << params.wall_csv
                   << " report=" << params.comparison_report << '\n';
         if (acceptance)
         {
            std::cout << "PASS M4(a/c) "
                      << (params.mesh_source == "exasim" ?
                         "converted-mesh" : "analytic-mesh")
                      << " four-stage continuation: stages="
                      << stage_results.size()
                      << " final_residual=" << report.residual
                      << " history=" << params.continuation_history_csv
                      << " summary=" << params.continuation_summary_csv
                      << '\n';
            if (params.comparison_gate)
            {
               if (comparison.heat_flux_max_relative_difference > 1.0e-2)
               {
                  throw std::runtime_error(
                     "M4(b) wall Fint heat flux differs by more than 1%");
               }
               if (shock_relative_difference > 1.0e-2)
               {
                  throw std::runtime_error(
                     "M4(b) shock standoff differs by more than 1%");
               }
               std::cout << "PASS M4(b) final wall heat flux and shock"
                         << " standoff within 1% of M3";
               if (comparison.heat_flux_max_relative_difference <=
                   1.0e-5 &&
                   shock_relative_difference <= 1.0e-5)
               {
                  std::cout << " (both match to 1e-5 or better)";
               }
               std::cout << '\n';
            }
         }
      }

      if (acceptance && !params.continuation_enabled)
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
                  *mesh, op, state, params.physics);
            const std::vector<hdg_ns::WallSample> reference_wall =
               hdg_ns::ComputeWallSamples(
                  *mesh, op, reference_state, params.physics);
            const std::string reference_csv =
               ReferenceCSVPath(params.wall_csv);
            hdg_ns::WriteWallCSV(params.wall_csv, wall);
            hdg_ns::WriteWallCSV(reference_csv, reference_wall);

            const hdg_ns::ShockStandoff shock =
               hdg_ns::ComputeShockStandoff(
                  *mesh, op, state);
            const hdg_ns::ShockStandoff reference_shock =
               hdg_ns::ComputeShockStandoff(
                  *mesh, op, reference_state);
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

#include "discretization/av_sensor.hpp"
#include "discretization/hdg_operator.hpp"
#include "io/exasim_mesh.hpp"
#include "io/mesh_input.hpp"
#include "physics/perfect_gas_model.hpp"
#include "physics/physics_factory.hpp"
#include "post/surface_post.hpp"
#include "solvers/newton.hpp"

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

struct ComparisonParams
{
   bool enabled = false;
   std::string reference_wall_csv;
   double reference_shock_standoff =
      std::numeric_limits<double>::quiet_NaN();
   bool gate = false;
   bool require_matching_coordinates = true;
   double cp_tolerance = 1.0e-2;
   double heat_flux_tolerance = 2.0e-2;
   double shock_tolerance = 1.0e-2;
};

struct DriverParams
{
   std::string case_name;
   std::string mesh_source;
   std::string exasim_directory;
   int analytic_nr = 0;
   int analytic_nc = 0;
   int analytic_order = hycfd::kOrder;
   std::string mesh_file;
   int curved_order = 0;
   int discretization_order = 4;
   int quadrature_increment = 1;
   struct BoundaryConditionSpec
   {
      std::vector<int> attrs;
      std::string type;
      YAML::Node params;
   };
   std::vector<BoundaryConditionSpec> bc_specs;
   YAML::Node physics_node;
   std::string av_mode;
   double av_lambda = 0.0;
   double av_c = 0.0;
   double av_kappa = 0.25;
   double av_relax = 0.5;
   bool av_bootstrap = false;
   double av_bootstrap_lambda = 0.0;
   double av_bootstrap_c = 0.0;
   int av_bootstrap_decay_iters = 25;
   std::string init_mode;
   double damping_c = 0.0;
   hycfd::NewtonConfig newton;
   std::string linear_type;
   std::string output_path;
   std::string wall_csv;
   std::string comparison_report;
   bool continuation_enabled = false;
   std::vector<ContinuationStage> continuation_stages;
   std::string continuation_history_csv;
   std::string continuation_summary_csv;
   ComparisonParams comparison;
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

template <typename T>
T Optional(const YAML::Node &node, const std::string &key, T fallback)
{
   if (!node || !node[key]) { return fallback; }
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
          params.analytic_order != hycfd::kOrder)
      {
         throw std::runtime_error(
            "analytic mesh requires positive nr/nc and order=4");
      }
   }
   else if (params.mesh_source == "file")
   {
      params.mesh_file = Required<std::string>(mesh, "file");
      params.curved_order = Optional<int>(mesh, "curved_order", 0);
   }
   else
   {
      throw std::runtime_error(
         "mesh.source must be exasim, analytic, or file");
   }

   const YAML::Node discretization = root["discretization"];
   params.discretization_order =
      Optional<int>(discretization, "order", 4);
   params.quadrature_increment =
      Optional<int>(discretization, "quadrature_increment", 1);
   if (params.mesh_source == "exasim" &&
       params.discretization_order != 4)
   {
      throw std::runtime_error(
         "exasim meshes require discretization.order 4");
   }

   const YAML::Node bc = root["bc"];
   if (!bc || !bc.IsSequence() || bc.size() == 0)
   {
      throw std::runtime_error(
         "bc must be a nonempty sequence of {attrs, type} entries");
   }
   for (const YAML::Node &entry : bc)
   {
      DriverParams::BoundaryConditionSpec spec;
      spec.attrs = Required<std::vector<int>>(entry, "attrs");
      spec.type = Required<std::string>(entry, "type");
      if (entry["params"]) { spec.params = entry["params"]; }
      if (spec.attrs.empty())
      {
         throw std::runtime_error("bc entry has an empty attrs list");
      }
      params.bc_specs.push_back(spec);
   }

   params.physics_node = root["physics"];
   if (!params.physics_node)
   {
      throw std::runtime_error("missing required YAML block: physics");
   }

   const YAML::Node av = root["av"];
   params.av_mode = Required<std::string>(av, "mode");
   if (params.av_mode == "tanh")
   {
      params.av_lambda = Required<double>(av, "lambda");
      params.av_c = Required<double>(av, "c");
   }
   else if (params.av_mode == "sensor")
   {
      params.av_lambda = Required<double>(av, "lambda");
      params.av_kappa = Optional<double>(av, "kappa", 0.25);
      params.av_relax = Optional<double>(av, "relax", 0.5);
      if (!(params.av_relax > 0.0) || params.av_relax > 1.0)
      {
         throw std::runtime_error("av.relax must lie in (0, 1]");
      }
      if (av["bootstrap"])
      {
         // Cold starts need AV where the shock will form before the
         // sensor can see it: a tanh profile floors the sensor field and
         // decays away over the first Newton iterations.
         params.av_bootstrap = true;
         params.av_bootstrap_lambda =
            Required<double>(av["bootstrap"], "lambda");
         params.av_bootstrap_c = Required<double>(av["bootstrap"], "c");
         params.av_bootstrap_decay_iters =
            Optional<int>(av["bootstrap"], "decay_iters", 25);
         if (params.av_bootstrap_decay_iters <= 0)
         {
            throw std::runtime_error(
               "av.bootstrap.decay_iters must be positive");
         }
      }
   }
   else if (params.av_mode != "file")
   {
      throw std::runtime_error("av.mode must be tanh, sensor, or file");
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
   params.newton.armijo_c1 =
      Optional<double>(newton, "armijo_c1", params.newton.armijo_c1);
   params.newton.alpha_min =
      Optional<double>(newton, "alpha_min", params.newton.alpha_min);
   params.newton.ptc_off_residual =
      Optional<double>(newton, "ptc_off_residual",
                       params.newton.ptc_off_residual);

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
         "linear.type must be petsc_direct (iterative solvers arrive"
         " with the MPI milestone)");
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
      params.comparison.enabled = true;
      params.comparison.reference_wall_csv =
         Required<std::string>(comparison, "reference_wall_csv");
      params.comparison.reference_shock_standoff =
         Optional<double>(comparison, "reference_shock_standoff",
                          params.comparison.reference_shock_standoff);
      params.comparison.gate = Required<bool>(comparison, "gate");
      params.comparison.require_matching_coordinates =
         Optional<bool>(comparison, "require_matching_coordinates", true);
      params.comparison.cp_tolerance =
         Optional<double>(comparison, "cp_tol",
                          params.comparison.cp_tolerance);
      params.comparison.heat_flux_tolerance =
         Optional<double>(comparison, "heat_flux_tol",
                          params.comparison.heat_flux_tolerance);
      params.comparison.shock_tolerance =
         Optional<double>(comparison, "shock_tol",
                          params.comparison.shock_tolerance);
      if (params.wall_csv.empty())
      {
         throw std::runtime_error(
            "comparison requires output.wall_csv");
      }
   }
   return params;
}

void PrintConfig(const DriverParams &params,
                 const hycfd::PerfectGasParams &gas_params)
{
   std::cout << "hycfd configuration:"
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
             << " M=" << gas_params.mach
             << " Re=" << gas_params.reynolds
             << " Pr=" << gas_params.prandtl
             << " gamma=" << gas_params.gamma
             << " av_mode=" << params.av_mode;
   if (params.av_mode == "tanh")
   {
      std::cout << " av=" << params.av_lambda
                << "*tanh(" << params.av_c << "*(r-1))";
   }
   std::cout << " init_mode=" << params.init_mode
             << " regularization="
             << (gas_params.regularized ? "floors" : "none")
             << " tau=" << gas_params.tau
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
                  hycfd::HDGOperator &op,
                  mfem::ParMesh &mesh)
      : params_(params),
        op_(op),
        conservative_(
           const_cast<mfem::FiniteElementSpace *>(&op.VolumeSpace())),
        primitive_(
           const_cast<mfem::FiniteElementSpace *>(&op.VolumeSpace())),
        scalar_fec_(op.Order(), 2),
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

   void Save(int cycle, const hycfd::HDGState &state)
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
   hycfd::HDGOperator &op_;
   mfem::GridFunction conservative_;
   mfem::GridFunction primitive_;
   mfem::L2_FECollection scalar_fec_;
   mfem::ParFiniteElementSpace scalar_fes_;
   mfem::GridFunction rho_, uvel_, vvel_, pressure_, av_;
   mfem::ParaViewDataCollection collection_;
};

hycfd::HDGOperator::ScalarFunction ScalarZero()
{
   return [](const mfem::Vector &) { return 0.0; };
}

hycfd::HDGOperator::ScalarFunction
ArtificialViscosity(double lambda, double c)
{
   return [lambda, c](const mfem::Vector &x)
   {
      const double distance = std::hypot(x[0], x[1]) - 1.0;
      return lambda * std::tanh(c * distance);
   };
}

std::vector<hycfd::ElementOrientation>
IdentityOrientations(int element_count)
{
   return std::vector<hycfd::ElementOrientation>(
      static_cast<std::size_t>(element_count));
}

struct StageResult
{
   ContinuationStage stage;
   hycfd::NewtonReport report;
   int damped_steps = 0;
   double minimum_alpha = 1.0;
};

StageResult SummarizeStage(const ContinuationStage &stage,
                           hycfd::NewtonReport report)
{
   StageResult result;
   result.stage = stage;
   result.report = std::move(report);
   for (const hycfd::NewtonIteration &iteration : result.report.history)
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
      for (const hycfd::NewtonIteration &iteration :
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

   std::ofstream summary(params.continuation_summary_csv);
   if (!summary)
   {
      throw std::runtime_error(
         "cannot open continuation summary CSV: " +
         params.continuation_summary_csv);
   }
   summary
      << "stage,lambda,c,ptc_enabled,converged,iterations,residual,"
         "damped_steps,min_alpha,assembly_seconds,linear_seconds,"
         "total_seconds,failure\n"
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
}

void WriteComparisonReport(
   const DriverParams &params,
   const std::vector<StageResult> &stages,
   const hycfd::M3Comparison &comparison,
   const hycfd::ShockStandoff &shock,
   double shock_relative_difference)
{
   if (params.comparison_report.empty()) { return; }
   std::ofstream output(params.comparison_report);
   if (!output)
   {
      throw std::runtime_error(
         "cannot open comparison report: " + params.comparison_report);
   }
   output << std::setprecision(16)
          << "# " << params.case_name << " comparison report\n\n"
          << "Mesh source: `" << params.mesh_source << "`\n"
          << "Reference: `" << params.comparison.reference_wall_csv
          << "`\n\n";
   if (!stages.empty())
   {
      output << "| Stage | lambda | c | PTC | Newton updates |"
                " Final residual | Damped steps | Minimum alpha |\n"
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
      output << '\n';
   }
   output << "| Quantity | Result | Gate |\n"
          << "|---|---:|---:|\n"
          << "| wall Cp maximum relative difference | "
          << comparison.cp_max_relative_difference
          << " | <= " << params.comparison.cp_tolerance << " |\n"
          << "| wall heat-flux maximum relative difference | "
          << comparison.heat_flux_max_relative_difference
          << " | <= " << params.comparison.heat_flux_tolerance << " |\n"
          << "| wall-coordinate maximum absolute difference | "
          << comparison.wall_coordinate_maximum_difference
          << " | geometry diagnostic |\n"
          << "| computed shock standoff | " << shock.distance
          << " | report |\n"
          << "| reference shock standoff | "
          << params.comparison.reference_shock_standoff
          << " | report |\n"
          << "| shock-standoff relative difference | "
          << shock_relative_difference
          << " | <= " << params.comparison.shock_tolerance << " |\n";
}

} // namespace

int main(int argc, char *argv[])
{
   int exit_code = EXIT_SUCCESS;
   mfem::Mpi::Init(argc, argv);
   mfem::Hypre::Init();

   const char *input_file = "input/m45_sanity.yaml";
   bool acceptance = false;
   mfem::OptionsParser args(argc, argv);
   args.AddOption(&input_file, "-i", "--input", "Input YAML deck.");
   args.AddOption(&acceptance, "-a", "--acceptance",
                  "-no-a", "--no-acceptance",
                  "Enforce the G1 convergence and comparison gates.");
   args.Parse();
   if (!args.Good())
   {
      if (mfem::Mpi::Root()) { args.PrintUsage(std::cerr); }
      return EXIT_FAILURE;
   }

   mfem::MFEMInitializePetsc(
      &argc, &argv, "input/petsc.opts", nullptr);
   if (!mfem::Mpi::Root())
   {
      // All ranks compute; only the root narrates.
      std::cout.setstate(std::ios::failbit);
   }
   try
   {
      const DriverParams params = LoadParams(input_file);
      std::cout << std::setprecision(17);

      std::unique_ptr<hycfd::PhysicsModel> physics_model =
         hycfd::MakePhysics(params.physics_node);
      const auto *gas_model =
         dynamic_cast<const hycfd::PerfectGasModel *>(physics_model.get());
      if (!gas_model)
      {
         throw std::runtime_error(
            "driver initial conditions and wall post-processing currently"
            " require the perfect_gas model");
      }
      const hycfd::PerfectGasParams &gas_params = gas_model->Params();
      PrintConfig(params, gas_params);

      std::vector<int> attr_to_bcid;
      for (const DriverParams::BoundaryConditionSpec &spec :
           params.bc_specs)
      {
         const int bc_id = physics_model->RegisterBoundaryCondition(
            spec.type, spec.params);
         for (const int attr : spec.attrs)
         {
            if (attr < 1)
            {
               throw std::runtime_error(
                  "bc attrs must be positive boundary attributes");
            }
            if (attr > static_cast<int>(attr_to_bcid.size()))
            {
               attr_to_bcid.resize(static_cast<std::size_t>(attr), -1);
            }
            if (attr_to_bcid[static_cast<std::size_t>(attr - 1)] != -1)
            {
               throw std::runtime_error(
                  "boundary attribute " + std::to_string(attr) +
                  " is assigned to more than one bc entry");
            }
            attr_to_bcid[static_cast<std::size_t>(attr - 1)] = bc_id;
         }
      }

      hycfd::ExasimMesh converted;
      std::unique_ptr<mfem::Mesh> analytic_mesh;
      mfem::Mesh *mesh = nullptr;
      std::vector<hycfd::ElementOrientation> orientations;
      if (params.mesh_source == "exasim")
      {
         converted =
            hycfd::BuildExasimMesh(params.exasim_directory);
         mesh = converted.mesh.get();
         orientations = converted.orientations;
      }
      else if (params.mesh_source == "analytic")
      {
         analytic_mesh = hycfd::BuildAnalyticMesh(
            params.analytic_nr, params.analytic_nc,
            params.analytic_order);
         mesh = analytic_mesh.get();
         orientations = IdentityOrientations(mesh->GetNE());
      }
      else
      {
         analytic_mesh = hycfd::LoadMeshFile(
            params.mesh_file, params.curved_order);
         mesh = analytic_mesh.get();
         orientations = IdentityOrientations(mesh->GetNE());
      }

      // Explicit partitioning so local elements can be mapped back to
      // their serial (Exasim-ordered) indices for vdg/udg data.
      std::unique_ptr<int[]> partitioning(
         mesh->GeneratePartitioning(mfem::Mpi::WorldSize()));
      mfem::ParMesh par_mesh(MPI_COMM_WORLD, *mesh, partitioning.get());
      std::vector<int> serial_element_ids;
      for (int element = 0; element < mesh->GetNE(); ++element)
      {
         if (partitioning[element] == mfem::Mpi::WorldRank())
         {
            serial_element_ids.push_back(element);
         }
      }

      hycfd::HDGOptions hdg_options;
      hdg_options.order = params.discretization_order;
      hdg_options.tau = gas_params.tau;
      hdg_options.quadrature_increment = params.quadrature_increment;

      hycfd::ExasimArray vdg;
      std::unique_ptr<hycfd::HDGOperator> op_pointer;
      if (params.av_mode == "file")
      {
         if (params.mesh_source != "exasim" ||
             params.continuation_enabled)
         {
            throw std::runtime_error(
               "av.mode=file requires a non-continuation Exasim mesh run");
         }
         vdg = hycfd::ReadExasimArray(
            params.exasim_directory + "/vdg.bin");
         op_pointer =
            std::make_unique<hycfd::HDGOperator>(
               par_mesh, vdg, orientations,
               *physics_model, attr_to_bcid, hdg_options,
               serial_element_ids);
      }
      else
      {
         if (params.av_mode == "sensor" && params.continuation_enabled)
         {
            throw std::runtime_error(
               "sensor AV and continuation stages cannot be combined yet");
         }
         double initial_lambda = params.av_lambda;
         double initial_c = params.av_c;
         if (params.continuation_enabled)
         {
            initial_lambda = params.continuation_stages.front().lambda;
            initial_c = params.continuation_stages.front().c;
         }
         op_pointer =
            std::make_unique<hycfd::HDGOperator>(
               par_mesh, orientations,
               params.av_mode == "sensor"
                  ? ScalarZero()
                  : ArtificialViscosity(initial_lambda, initial_c),
               *physics_model, attr_to_bcid, hdg_options);
      }
      hycfd::HDGOperator &op = *op_pointer;

      hycfd::HDGState state;
      if (params.init_mode == "udg_file")
      {
         if (params.mesh_source != "exasim" ||
             params.continuation_enabled)
         {
            throw std::runtime_error(
               "init.mode=udg_file requires a non-continuation Exasim run");
         }
         const hycfd::ExasimArray input_udg =
            hycfd::ReadExasimArray(
               params.exasim_directory + "/udg.bin");
         op.LoadExasimVolumeState(input_udg, false, state);
         op.InitializeTraceFromInterior(state);
      }
      else
      {
         const auto initial_condition = [&params, &gas_params](
            const mfem::Vector &x, double value[4])
         {
            const double distance = std::hypot(x[0], x[1]) - 1.0;
            const double velocity = std::tanh(params.damping_c * distance);
            const hycfd::PerfectGasParams &gas = gas_params;
            value[0] = 1.0;
            value[1] = velocity;
            value[2] = 0.0;
            value[3] =
               gas.TinfND() *
                  ((gas.Twall_K / gas.T_inf_K - 1.0) *
                      std::exp(-params.damping_c * distance) + 1.0) +
               0.5 * velocity * velocity;
         };
         op.ProjectState(initial_condition, state);
         // Restart semantics shared with Exasim: the trace comes from the
         // lower-index side of each face and q is then recomputed.
         op.InitializeTraceFromInterior(state);
      }

      ParaViewWriter writer(params, op, par_mesh);
      hycfd::NewtonReport report;
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
               op.InitializeTraceFromInterior(state);
            }
            const int cycle_base =
               static_cast<int>(1000 * stage_index);
            writer.Save(cycle_base, state);
            hycfd::NewtonConfig stage_config = params.newton;
            stage_config.pseudo_transient = stage.pseudo_transient;
            stage_config.initial_pseudo_time_step =
               stage.initial_pseudo_time_step;
            const hycfd::NewtonOutput output =
               [&params, &writer, &op, stage_index, cycle_base](
                  int iteration, const hycfd::HDGState &current,
                  double residual)
            {
               std::cout << "Stage " << stage_index + 1
                         << " Newton " << std::setw(3) << iteration
                         << " residual=" << residual
                         << " min_rho=" << op.MinimumDensity(current)
                         << " min_p=" << op.MinimumPressure(current)
                         << std::endl;
               if (params.paraview_every > 0 &&
                   iteration > 0 &&
                   iteration % params.paraview_every == 0)
               {
                  writer.Save(cycle_base + iteration, current);
               }
            };
            report = hycfd::DampedNewtonSolve(
               op, state, stage_config, output);
            stage_results.push_back(
               SummarizeStage(stage, std::move(report)));
            const StageResult &result = stage_results.back();
            writer.Save(cycle_base + result.report.iterations + 1, state);
            if (mfem::Mpi::Root())
            {
               WriteContinuationCSVs(params, stage_results);
            }
            std::cout << "Continuation stage " << stage_index + 1
                      << " result: lambda=" << stage.lambda
                      << " c=" << stage.c
                      << " PTC="
                      << (stage.pseudo_transient ? "on" : "off")
                      << " converged="
                      << (result.report.converged ? "yes" : "no")
                      << " iterations=" << result.report.iterations
                      << " residual=" << result.report.residual
                      << " damped_steps=" << result.damped_steps
                      << " min_alpha=" << result.minimum_alpha;
            if (!result.report.failure.empty())
            {
               std::cout << " failure=" << result.report.failure;
            }
            std::cout << '\n';
            if (!result.report.converged)
            {
               throw std::runtime_error(
                  "continuation stage " +
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
         const hycfd::NewtonOutput output =
            [&params, &writer, &op](
               int iteration, const hycfd::HDGState &current,
               double residual)
         {
            std::cout << "Newton " << std::setw(3) << iteration
                      << " residual=" << residual
                      << " min_rho=" << op.MinimumDensity(current)
                      << " min_p=" << op.MinimumPressure(current)
                      << std::endl;
            if (params.paraview_every > 0 &&
                iteration > 0 &&
                iteration % params.paraview_every == 0)
            {
               writer.Save(iteration, current);
            }
         };
         if (params.av_mode == "sensor")
         {
            // Adaptive frozen-per-iteration AV: the dilatation sensor is
            // re-evaluated from the current iterate at the top of every
            // Newton iteration and then frozen through that iteration's
            // Jacobian and line search. The per-solve lagged fixed point
            // this replaces limit-cycled: the shock drifted between
            // solves faster than the outer relaxation could track.
            mfem::H1_FECollection sensor_fec(1, 2);
            mfem::ParFiniteElementSpace sensor_fes(&par_mesh,
                                                   &sensor_fec);
            mfem::ParGridFunction av_field(&sensor_fes);
            mfem::ParGridFunction sensor_field(&sensor_fes);
            mfem::ParGridFunction bootstrap_field(&sensor_fes);
            av_field = 0.0;
            bootstrap_field = 0.0;
            if (params.av_bootstrap)
            {
               mfem::FunctionCoefficient bootstrap_coefficient(
                  ArtificialViscosity(params.av_bootstrap_lambda,
                                      params.av_bootstrap_c));
               bootstrap_field.ProjectCoefficient(bootstrap_coefficient);
            }
            hycfd::DilatationSensorOptions sensor_options;
            sensor_options.lambda = params.av_lambda;
            sensor_options.kappa = params.av_kappa;
            bool have_previous = false;
            const hycfd::NewtonPrepare refresh =
               [&params, &op, &physics_model, &sensor_options,
                &av_field, &sensor_field, &bootstrap_field,
                &have_previous](
                  int iteration, const hycfd::HDGState &current)
            {
               hycfd::ComputeDilatationAV(op, current, *physics_model,
                                          sensor_options, sensor_field);
               // Exponential moving average: damps sensor flicker while
               // the AV and the state co-evolve iteration by iteration.
               if (have_previous)
               {
                  for (int i = 0; i < sensor_field.Size(); ++i)
                  {
                     sensor_field[i] =
                        params.av_relax * sensor_field[i] +
                        (1.0 - params.av_relax) * av_field[i];
                  }
               }
               if (params.av_bootstrap)
               {
                  // Decaying floor: keeps stabilization where the shock
                  // will form until the sensor has locked onto it.
                  const double decay = std::max(
                     0.0, 1.0 - static_cast<double>(iteration) /
                             static_cast<double>(
                                params.av_bootstrap_decay_iters));
                  for (int i = 0; i < sensor_field.Size(); ++i)
                  {
                     sensor_field[i] = std::max(
                        sensor_field[i], decay * bootstrap_field[i]);
                  }
               }
               double change_sumsq = 0.0;
               for (int i = 0; i < sensor_field.Size(); ++i)
               {
                  const double difference = sensor_field[i] - av_field[i];
                  change_sumsq += difference * difference;
               }
               av_field = sensor_field;
               have_previous = true;
               op.SetArtificialViscosityField(av_field);
               const double norm_sumsq = mfem::InnerProduct(
                  MPI_COMM_WORLD, av_field, av_field);
               MPI_Allreduce(MPI_IN_PLACE, &change_sumsq, 1, MPI_DOUBLE,
                             MPI_SUM, MPI_COMM_WORLD);
               std::cout << "AV refresh " << std::setw(3) << iteration
                         << " relative_change="
                         << std::sqrt(change_sumsq /
                                      std::max(1.0e-28, norm_sumsq))
                         << " max_av=" << op.MaximumAbsAV()
                         << std::endl;
            };
            report = hycfd::DampedNewtonSolve(op, state, params.newton,
                                              output, refresh);
         }
         else
         {
            report =
               hycfd::DampedNewtonSolve(op, state, params.newton,
                                        output);
         }
         writer.Save(report.iterations + 1, state);
      }

      const double minimum_density = op.MinimumDensity(state);
      const double minimum_pressure = op.MinimumPressure(state);
      // The reflection-pair symmetry diagnostic needs the full field; it
      // is only evaluated (and gated) on single-rank runs.
      const double symmetry_error =
         mfem::Mpi::WorldSize() == 1 ? op.YSymmetryError(state) : 0.0;
      std::cout << "Solve result:"
                << " converged=" << (report.converged ? "yes" : "no")
                << " iterations=" << report.iterations
                << " residual=" << report.residual
                << " min_rho=" << minimum_density
                << " min_p=" << minimum_pressure
                << " y_symmetry=" << symmetry_error
                << " assembly_seconds=" << report.assembly_seconds
                << " linear_seconds=" << report.linear_solve_seconds
                << " total_seconds=" << report.total_seconds << '\n';
      if (!report.failure.empty())
      {
         std::cout << "Solve failure: " << report.failure << '\n';
      }

      if (!params.wall_csv.empty())
      {
         const std::vector<hycfd::WallSample> wall =
            hycfd::ComputeWallSamples(par_mesh, op, state, gas_params);
         if (mfem::Mpi::Root())
         {
            hycfd::WriteWallCSV(params.wall_csv, wall);
         }
         if (params.comparison.enabled)
         {
            const std::vector<hycfd::WallSample> reference_wall =
               hycfd::ReadWallCSV(params.comparison.reference_wall_csv);
            const hycfd::ShockStandoff shock =
               hycfd::ComputeShockStandoff(par_mesh, op, state);
            hycfd::ShockStandoff reference_shock;
            reference_shock.distance =
               params.comparison.reference_shock_standoff;
            reference_shock.radius =
               1.0 + params.comparison.reference_shock_standoff;
            const hycfd::M3Comparison comparison =
               hycfd::CompareWallAndShock(
                  wall, reference_wall, shock, reference_shock,
                  params.comparison.require_matching_coordinates);
            const bool have_reference_shock =
               std::isfinite(params.comparison.reference_shock_standoff);
            const double shock_relative_difference =
               have_reference_shock ?
               comparison.shock_standoff_difference /
                  std::max(1.0e-14,
                           std::abs(
                              params.comparison.reference_shock_standoff)) :
               0.0;
            if (mfem::Mpi::Root())
            {
               WriteComparisonReport(params, stage_results, comparison,
                                     shock, shock_relative_difference);
            }
            std::cout << "Wall/shock comparison:"
                      << " Cp_max_rel="
                      << comparison.cp_max_relative_difference
                      << " qw_max_rel="
                      << comparison.heat_flux_max_relative_difference
                      << " coordinate_max_abs="
                      << comparison.wall_coordinate_maximum_difference
                      << " shock=" << shock.distance
                      << " reference_shock="
                      << params.comparison.reference_shock_standoff
                      << " shock_rel=" << shock_relative_difference
                      << " wall_csv=" << params.wall_csv << '\n';
            if (acceptance && params.comparison.gate)
            {
               if (comparison.cp_max_relative_difference >
                   params.comparison.cp_tolerance)
               {
                  throw std::runtime_error(
                     "G1 regression: wall Cp differs from the frozen"
                     " reference beyond tolerance");
               }
               if (comparison.heat_flux_max_relative_difference >
                   params.comparison.heat_flux_tolerance)
               {
                  throw std::runtime_error(
                     "G1 regression: wall heat flux differs from the"
                     " frozen reference beyond tolerance");
               }
               if (have_reference_shock &&
                   shock_relative_difference >
                      params.comparison.shock_tolerance)
               {
                  throw std::runtime_error(
                     "G1 regression: shock standoff differs from the"
                     " frozen reference beyond tolerance");
               }
               std::cout << "PASS G1 wall Cp / heat flux / shock standoff"
                         << " within tolerance of the frozen reference\n";
            }
         }
      }

      if (acceptance)
      {
         if (!report.converged ||
             report.residual > params.newton.tolerance)
         {
            throw std::runtime_error(
               "G1 Newton did not converge to tolerance");
         }
         if (!(minimum_density > 0.0) || !(minimum_pressure > 0.0))
         {
            throw std::runtime_error("G1 fields are not positive");
         }
         if (params.init_mode == "damped_freestream" &&
             !params.continuation_enabled && symmetry_error > 1.0e-8)
         {
            throw std::runtime_error(
               "G1 fields are not y-symmetric to 1e-8");
         }
         std::cout << "PASS G1 acceptance:"
                   << " residual=" << report.residual
                   << " min_rho=" << minimum_density
                   << " min_p=" << minimum_pressure
                   << " y_symmetry=" << symmetry_error << '\n';
      }
   }
   catch (const std::exception &error)
   {
      std::cerr << "FAIL hycfd: " << error.what() << '\n';
      exit_code = EXIT_FAILURE;
   }
   mfem::MFEMFinalizePetsc();
   return exit_code;
}

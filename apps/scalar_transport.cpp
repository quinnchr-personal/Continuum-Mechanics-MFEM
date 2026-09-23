// Scalar transport (convection-diffusion-reaction of one unknown), steady
// or in physical time by implicit Euler (a `time` block): YAML in, ParaView
// out, with the errors against an exact expression and the flows through the
// Dirichlet entries per step. This executable only parses input and wires
// library objects together.
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

#include "base/fields.hpp"
#include "base/mesh_input.hpp"
#include "base/output.hpp"
#include "base/probes.hpp"
#include "base/scalar_config.hpp"
#include "mfem.hpp"
#include "physics/scalar_transport.hpp"
#include "solvers/direct_solver.hpp"
#include "solvers/quasi_static.hpp"

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  const cmf::PetscSession petsc;   // solver.linear.type: direct
  const bool root = mfem::Mpi::Root();

  const char *input = "";
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&input, "-i", "--input", "YAML input file (physics: scalar_transport).");
  args.Parse();
  if (!args.Good())
  {
    if (root) { args.PrintUsage(std::cout); }
    return 1;
  }

  try
  {
    const cmf::ScalarAppConfig cfg = cmf::LoadScalarConfig(input);
    std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    std::unique_ptr<cmf::ScalarTransport> problem = cmf::MakeScalarTransport(*pmesh, cfg);
    cmf::ScalarTransport &physics = *problem;
    physics.Finalize();
    const bool in_time = cfg.time.enabled;
    physics.SetPhysicalTime(in_time);

    const HYPRE_BigInt global_ne = pmesh->GetGlobalNE();
    const HYPRE_BigInt global_tdofs = physics.GlobalTrueVSize();
    if (root)
    {
      std::cout << "mesh: " << cfg.mesh.file << ", dim " << pmesh->Dimension() << ", elements "
                << global_ne << ", order " << cfg.mesh.order << ", true dofs " << global_tdofs
                << ", " << physics.Description() << std::endl;
      std::cout << "  element attributes: " << cmf::DescribeAttributes(*pmesh, false)
                << "; boundary attributes: " << cmf::DescribeAttributes(*pmesh, true) << std::endl;
      for (const std::string &line : cmf::DescribeScalarTimeStepping(cfg)) { std::cout << line << std::endl; }
    }

    std::unique_ptr<mfem::Solver> linear = physics.MakeLinearSolver(cfg.solver.linear);
    mfem::Vector u(physics.Height());
    physics.InitialState(u);

    cmf::FieldRegistry fields;
    physics.RegisterFields(fields);
    std::unique_ptr<cmf::ParaViewWriter> writer;
    if (!cfg.output.paraview.empty())
    {
      writer = std::make_unique<cmf::ParaViewWriter>(
        cfg.output.paraview, *pmesh, cfg.mesh.order, cfg.output.high_order);
      writer->RegisterAll(cfg.output, fields);
      physics.SetLoadFactor(0.0);
      physics.UpdateFields(u);
      writer->Save(0, 0.0);
    }

    // Every registered field at every probe point.
    auto print_probes = [&](const std::string &prefix)
    {
      for (const cmf::ProbeConfig &probe : cfg.output.probes)
      {
        for (const std::string &name : fields.Names())
        {
          const std::vector<double> value = cmf::ProbeVector(fields.Get(name), probe.point);
          if (root)
          {
            std::printf("%sprobe %s at (", prefix.c_str(), probe.name.c_str());
            for (std::size_t i = 0; i < probe.point.size(); i++)
            {
              std::printf("%s%g", i ? ", " : "", probe.point[i]);
            }
            std::printf("): %s =", name.c_str());
            for (double v : value) { std::printf(" %.12e", v); }
            std::printf("\n");
          }
        }
      }
    };

    // The errors against output.exact (printed per step, and, with ParaView
    // output, written to <collection>/error_history.csv; step < 0: no row).
    std::unique_ptr<cmf::CsvWriter> error_csv;
    auto print_errors = [&](const std::string &prefix, const mfem::Vector &x, int step, double t)
    {
      if (!physics.HasExact()) { return; }
      const cmf::ScalarErrors e = physics.Errors(x, t);
      if (root && !(prefix.empty() && step == 0))
      {
        std::printf("%serror: l2 = %.12e rel_l2 = %.12e linf_nodal = %.12e\n", prefix.c_str(), e.l2, e.rel_l2,
                    e.linf_nodal);
      }
      if (writer && step >= 0)
      {
        if (!error_csv)
        {
          error_csv = std::make_unique<cmf::CsvWriter>(cfg.output.paraview, "error_history.csv",
                                                       std::vector<std::string>{"l2", "rel_l2", "linf_nodal"}, root);
        }
        error_csv->Append(step, t, {e.l2, e.rel_l2, e.linf_nodal});
      }
    };
    // The flow into the domain through every Dirichlet entry (output.flows),
    // printed per step and written to <collection>/flows.csv.
    std::unique_ptr<cmf::CsvWriter> flow_csv;
    auto print_flows = [&](const std::string &prefix, const mfem::Vector &x, int step, double t)
    {
      if (!cfg.flows) { return; }
      const std::vector<cmf::Flow> flows = physics.Flows(x);
      std::vector<double> values;
      for (const cmf::Flow &f : flows)
      {
        if (root && !(prefix.empty() && step == 0))
        {
          std::printf("%sflow %s: %.12e\n", prefix.c_str(), f.name.c_str(), f.value);
        }
        values.push_back(f.value);
      }
      if (writer && step >= 0)
      {
        if (!flow_csv)
        {
          std::vector<std::string> names;
          for (const cmf::Flow &f : flows) { names.push_back(f.name); }
          flow_csv = std::make_unique<cmf::CsvWriter>(cfg.output.paraview, "flows.csv", names, root);
        }
        flow_csv->Append(step, t, values);
      }
    };
    // The initial state of a transient run: the row of cycle 0 (printed as step 0).
    if (in_time)
    {
      physics.SetLoadFactor(0.0);
      if (physics.HasExact())
      {
        const cmf::ScalarErrors e = physics.Errors(u, 0.0);
        if (root)
        {
          std::printf("step 0 t = 0.000000000e+00 error: l2 = %.12e rel_l2 = %.12e linf_nodal = %.12e\n", e.l2,
                      e.rel_l2, e.linf_nodal);
        }
      }
      print_errors("", u, 0, 0.0);
      print_flows("", u, 0, 0.0);
    }

    // Every output.every-th step is written, and always the last one.
    const double t_end = in_time ? cfg.time.t_final : 1.0;
    const cmf::LoadStepCallback on_step = [&](const cmf::LoadStepReport &step, const mfem::Vector &x)
    {
      if (!step.newton.converged) { return; }
      const bool write = writer && (step.step % cfg.output.every == 0 || step.load_factor >= t_end);
      if (write || cfg.output.probe_every_step) { physics.UpdateFields(x); }
      if (write) { writer->Save(step.step, step.load_factor); }
      char prefix[64];
      std::snprintf(prefix, sizeof(prefix), in_time ? "step %d t = %.9e " : "step %d t = %.6f ",
                    step.step, step.load_factor);
      if (root && in_time)
      {
        std::printf("%snewton iterations = %d residual = %.6e%s\n", prefix, step.newton.iterations,
                    step.newton.residual, step.newton.at_floor ? " (floor)" : "");
      }
      print_errors(prefix, x, step.step, step.load_factor);
      print_flows(prefix, x, step.step, step.load_factor);
      if (cfg.output.probe_every_step) { print_probes(prefix); }
    };
    const cmf::QuasiStaticReport report =
      in_time ? cmf::SolveInTime(physics, *linear, cfg.solver, cfg.time.breakpoints, 0.0, u, on_step)
              : cmf::SolveQuasiStatic(physics, *linear, cfg.solver, u, on_step);

    physics.UpdateFields(u);
    const double u_l2 = physics.NormL2(u);
    const double t_final = report.steps.empty() ? 0.0 : report.steps.back().load_factor;
    if (root)
    {
      if (in_time)
      {
        std::printf("result: converged %s, time steps %zu, t = %.9e, |%s|_L2 = %.12e", report.converged ? "yes" : "no",
                    report.steps.size(), t_final, physics.UnknownName().c_str(), u_l2);
      }
      else
      {
        std::printf("result: converged %s, load steps %zu, |%s|_L2 = %.12e", report.converged ? "yes" : "no",
                    report.steps.size(), physics.UnknownName().c_str(), u_l2);
      }
      int its = 0;
      for (const cmf::LoadStepReport &s : report.steps) { its += s.newton.iterations; }
      std::printf(", newton iterations %d\n", its);
    }
    if (physics.HasExact())
    {
      const cmf::ScalarErrors e = physics.Errors(u, in_time ? t_final : 1.0);
      if (root)
      {
        std::printf("result: error l2 = %.12e rel_l2 = %.12e linf_nodal = %.12e\n", e.l2, e.rel_l2, e.linf_nodal);
      }
    }
    print_probes("");
    print_flows("", u, -1, 0.0);
    if (root && writer) { std::cout << "wrote " << cfg.output.paraview << std::endl; }
    return report.converged ? 0 : 2;
  }
  catch (const std::exception &e)
  {
    if (root) { std::cerr << "error: " << e.what() << std::endl; }
    return 1;
  }
}

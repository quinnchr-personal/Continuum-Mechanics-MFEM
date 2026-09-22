// Total Lagrangian solid mechanics, quasi-static in a pseudo-time, quasi-static
// in physical time (a `time` block: rate-dependent materials) or dynamic (a
// `dynamics` block): YAML in, ParaView out.
// This executable only parses input and wires library objects together.
#include <array>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

#include "base/config.hpp"
#include "base/fields.hpp"
#include "base/mesh_input.hpp"
#include "base/output.hpp"
#include "base/probes.hpp"
#include "mfem.hpp"
#include "physics/dynamic_solid_problem.hpp"
#include "physics/solid_problem.hpp"
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
  args.AddOption(&input, "-i", "--input", "YAML input file.");
  args.Parse();
  if (!args.Good())
  {
    if (root) { args.PrintUsage(std::cout); }
    return 1;
  }

  try
  {
    const cmf::AppConfig cfg = cmf::LoadConfig(input);
    std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*pmesh, cfg);
    cmf::SolidProblem &physics = *problem;
    physics.Finalize();
    // With a dynamics block: the same problem with inertia, stepped in time.
    std::unique_ptr<cmf::DynamicSolidProblem> dynamic;
    if (cfg.dynamics.enabled)
    {
      dynamic = cmf::MakeDynamicSolidProblem(physics, cfg);
      dynamic->TrackExternalWork(cfg.output.energy);
    }
    // With a time block: the quasi-static problem stepped in physical time.
    const bool in_time = cfg.time.enabled;
    if (in_time) { physics.SetPhysicalTime(true); }

    const HYPRE_BigInt global_ne = pmesh->GetGlobalNE();
    const HYPRE_BigInt global_tdofs = physics.GlobalTrueVSize();
    if (root)
    {
      std::cout << "mesh: " << cfg.mesh.file << ", dim " << pmesh->Dimension() << ", elements "
                << global_ne << ", order " << cfg.mesh.order << ", true dofs " << global_tdofs
                << ", " << physics.Description() << std::endl;
      std::cout << "  element attributes: " << cmf::DescribeAttributes(*pmesh, false)
                << "; boundary attributes: " << cmf::DescribeAttributes(*pmesh, true) << std::endl;
      if (dynamic)
      {
        for (const std::string &line : cmf::DescribeDynamics(cfg)) { std::cout << line << std::endl; }
      }
      else if (in_time)
      {
        for (const std::string &line : cmf::DescribeTimeStepping(cfg)) { std::cout << line << std::endl; }
      }
    }

    std::unique_ptr<mfem::Solver> linear = dynamic ? dynamic->MakeLinearSolver(cfg.solver.linear)
                                                   : physics.MakeLinearSolver(cfg.solver.linear);
    mfem::Vector u(physics.Height());
    physics.InitialState(u); // zero, but for the temperature block (theta0) of the coupled formulation

    cmf::FieldRegistry fields;
    if (dynamic) { dynamic->RegisterFields(fields); }
    else { physics.RegisterFields(fields); }
    auto update_fields = [&](const mfem::Vector &x)
    {
      if (dynamic) { dynamic->UpdateFields(x); }
      else { physics.UpdateFields(x); }
    };
    if (dynamic)
    {
      // u_0, the Dirichlet data of t = 0, v_0 and the consistent a_0.
      const double change = dynamic->Initialize(u);
      if (root && change > 1e-12)
      {
        std::printf("  warning: the Dirichlet data at t = 0 changed the initial displacement by up "
                    "to %.3e on the prescribed dofs\n", change);
      }
    }
    std::unique_ptr<cmf::ParaViewWriter> writer;
    if (!cfg.output.paraview.empty())
    {
      writer = std::make_unique<cmf::ParaViewWriter>(
        cfg.output.paraview, *pmesh, cfg.mesh.order, cfg.output.high_order);
      writer->RegisterAll(cfg.output, fields);
      update_fields(u);
      writer->Save(0, 0.0);
    }

    // Every registered grid function at every probe point: the nodal unknowns
    // and the nodal (<name>) and element (<name>_elem) presentations of the
    // quadrature quantities; point clouds cannot be probed. `prefix` marks
    // the per-step lines ("step k t = ..."); the final lines carry none.
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

    // Resultant force and moment of every Dirichlet entry (output.reactions);
    // in a dynamic analysis of the balance with inertia at the accepted state.
    // Printed after every step and at the end, and, with ParaView output, also
    // written to <collection>/reactions.csv per step (step < 0: no row).
    std::unique_ptr<cmf::ReactionWriter> reaction_csv;
    auto print_reactions = [&](const std::string &prefix, const mfem::Vector &x, int step, double t)
    {
      if (!cfg.output.reactions) { return; }
      const std::vector<cmf::Reaction> reactions = dynamic ? dynamic->Reactions() : physics.Reactions(x);
      std::vector<std::array<double, 6>> rows;
      for (const cmf::Reaction &rx : reactions)
      {
        if (root && !(prefix.empty() && step == 0))   // the initial state goes to the CSV only
        {
          std::printf("%sreaction %s: force =", prefix.c_str(), rx.name.c_str());
          for (int d = 0; d < pmesh->Dimension(); d++) { std::printf(" %.12e", rx.force[d]); }
          std::printf(" moment =");
          for (int d = 0; d < 3; d++) { std::printf(" %.12e", rx.moment[d]); }
          std::printf("\n");
        }
        rows.push_back({rx.force[0], rx.force[1], rx.force[2], rx.moment[0], rx.moment[1], rx.moment[2]});
      }
      if (writer && step >= 0)
      {
        if (!reaction_csv)
        {
          std::vector<std::string> names;
          for (const cmf::Reaction &rx : reactions) { names.push_back(rx.name); }
          reaction_csv = std::make_unique<cmf::ReactionWriter>(cfg.output.paraview, names, root);
        }
        reaction_csv->Append(step, t, rows);
      }
    };
    // The initial state, for the row of cycle 0 of the .pvd (not printed).
    if (writer) { print_reactions("", u, 0, 0.0); }

    // Kinetic and internal energy, external work and their balance
    // (output.energy, dynamic analysis).
    double energy_0 = 0.0;
    auto print_energy = [&](const std::string &prefix, const mfem::Vector &x)
    {
      if (!dynamic || !cfg.output.energy) { return; }
      const double kinetic = dynamic->KineticEnergy(), internal = physics.InternalEnergy(x);
      if (dynamic->Steps() == 0) { energy_0 = kinetic + internal; }
      if (root)
      {
        std::printf("%senergy: kinetic = %.12e internal = %.12e external_work = %.12e balance = %.12e\n",
                    prefix.c_str(), kinetic, internal, dynamic->ExternalWork(),
                    kinetic + internal - dynamic->ExternalWork() - energy_0);
        // balance: kinetic + internal - external work - (kinetic + internal)(t = 0)
      }
    };
    if (dynamic) { print_energy("step 0 t = 0.000000000e+00 ", u); }

    // Every output.every-th step is written, and always the last one.
    const double t_end = dynamic ? cfg.dynamics.t_final : in_time ? cfg.time.t_final : 1.0;
    const bool physical_time = dynamic || in_time;
    const cmf::LoadStepCallback on_step = [&](const cmf::LoadStepReport &step, const mfem::Vector &x)
    {
      if (!step.newton.converged) { return; }
      const bool write = writer && (step.step % cfg.output.every == 0 || step.load_factor >= t_end);
      if (write || cfg.output.probe_every_step) { update_fields(x); }
      if (write) { writer->Save(step.step, step.load_factor); }
      char prefix[64];
      std::snprintf(prefix, sizeof(prefix), physical_time ? "step %d t = %.9e " : "step %d t = %.6f ",
                    step.step, step.load_factor);
      if (cfg.output.probe_every_step) { print_probes(prefix); }
      print_reactions(prefix, x, step.step, step.load_factor);
      print_energy(prefix, x);
    };
    const cmf::QuasiStaticReport report =
      dynamic ? cmf::SolveDynamic(*dynamic, *linear, cfg.solver, cfg.dynamics.breakpoints, 0.0, u, on_step)
      : in_time ? cmf::SolveInTime(physics, *linear, cfg.solver, cfg.time.breakpoints, 0.0, u, on_step)
                : cmf::SolveQuasiStatic(physics, *linear, cfg.solver, u, on_step);

    update_fields(u);
    mfem::Vector zero(pmesh->Dimension());
    zero = 0.0;
    mfem::VectorConstantCoefficient zero_coef(zero);
    const double u_l2 = physics.Displacement().ComputeL2Error(zero_coef);
    const double energy = physics.InternalEnergy(u);
    const double kinetic = dynamic ? dynamic->KineticEnergy() : 0.0; // collective
    if (root && dynamic)
    {
      std::printf("result: converged %s, time steps %zu, t = %.9e, |u|_L2 = %.12e, "
                  "internal energy = %.12e, kinetic energy = %.12e\n",
                  report.converged ? "yes" : "no", report.steps.size(), dynamic->Time(), u_l2,
                  energy, kinetic);
    }
    else if (root && in_time)
    {
      std::printf("result: converged %s, time steps %zu, t = %.9e, |u|_L2 = %.12e, "
                  "internal energy = %.12e\n",
                  report.converged ? "yes" : "no", report.steps.size(),
                  report.steps.empty() ? 0.0 : report.steps.back().load_factor, u_l2, energy);
    }
    else if (root)
    {
      std::printf("result: converged %s, load steps %zu, |u|_L2 = %.12e, "
                  "internal energy = %.12e\n",
                  report.converged ? "yes" : "no", report.steps.size(), u_l2, energy);
    }
    print_probes("");
    print_reactions("", u, -1, 0.0);
    if (root && writer) { std::cout << "wrote " << cfg.output.paraview << std::endl; }
    return report.converged ? 0 : 2;
  }
  catch (const std::exception &e)
  {
    if (root) { std::cerr << "error: " << e.what() << std::endl; }
    return 1;
  }
}

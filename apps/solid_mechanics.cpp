// Quasi-static total Lagrangian solid mechanics: YAML in, ParaView out.
// This executable only parses input and wires library objects together.
#include <cstdio>
#include <iostream>
#include <memory>

#include "base/config.hpp"
#include "base/fields.hpp"
#include "base/mesh_input.hpp"
#include "base/output.hpp"
#include "base/probes.hpp"
#include "mfem.hpp"
#include "physics/solid_problem.hpp"
#include "solvers/quasi_static.hpp"

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
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

    const HYPRE_BigInt global_ne = pmesh->GetGlobalNE();
    const HYPRE_BigInt global_tdofs = physics.GlobalTrueVSize();
    if (root)
    {
      std::cout << "mesh: " << cfg.mesh.file << ", dim " << pmesh->Dimension() << ", elements "
                << global_ne << ", order " << cfg.mesh.order << ", true dofs " << global_tdofs
                << ", " << physics.Description() << std::endl;
      std::cout << "  element attributes: " << cmf::DescribeAttributes(*pmesh, false)
                << "; boundary attributes: " << cmf::DescribeAttributes(*pmesh, true) << std::endl;
    }

    std::unique_ptr<mfem::Solver> linear = physics.MakeLinearSolver(cfg.solver.linear);
    mfem::Vector u(physics.Height());
    u = 0.0;

    cmf::FieldRegistry fields;
    physics.RegisterFields(fields);
    std::unique_ptr<cmf::ParaViewWriter> writer;
    if (!cfg.output.paraview.empty())
    {
      writer = std::make_unique<cmf::ParaViewWriter>(
        cfg.output.paraview, *pmesh, cfg.mesh.order, cfg.output.high_order);
      writer->RegisterAll(cfg.output, fields);
      physics.UpdateFields(u);
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

    // Resultant force and moment of every Dirichlet entry (output.reactions).
    auto print_reactions = [&](const std::string &prefix, const mfem::Vector &x)
    {
      if (!cfg.output.reactions) { return; }
      for (const cmf::Reaction &rx : physics.Reactions(x))
      {
        if (root)
        {
          std::printf("%sreaction %s: force =", prefix.c_str(), rx.name.c_str());
          for (int d = 0; d < pmesh->Dimension(); d++) { std::printf(" %.12e", rx.force[d]); }
          std::printf(" moment =");
          for (int d = 0; d < 3; d++) { std::printf(" %.12e", rx.moment[d]); }
          std::printf("\n");
        }
      }
    };

    const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(
      physics, *linear, cfg.solver, u,
      [&](const cmf::LoadStepReport &step, const mfem::Vector &x)
      {
        if (!step.newton.converged) { return; }
        if (writer || cfg.output.probe_every_step) { physics.UpdateFields(x); }
        if (writer) { writer->Save(step.step, step.load_factor); }
        char prefix[64];
        std::snprintf(prefix, sizeof(prefix), "step %d t = %.6f ", step.step, step.load_factor);
        if (cfg.output.probe_every_step) { print_probes(prefix); }
        print_reactions(prefix, x);
      });

    physics.UpdateFields(u);
    mfem::Vector zero(pmesh->Dimension());
    zero = 0.0;
    mfem::VectorConstantCoefficient zero_coef(zero);
    const double u_l2 = physics.Displacement().ComputeL2Error(zero_coef);
    const double energy = physics.InternalEnergy(u);
    if (root)
    {
      std::printf("result: converged %s, load steps %zu, |u|_L2 = %.12e, "
                  "internal energy = %.12e\n",
                  report.converged ? "yes" : "no", report.steps.size(), u_l2, energy);
    }
    print_probes("");
    print_reactions("", u);
    if (root && writer) { std::cout << "wrote " << cfg.output.paraview << std::endl; }
    return report.converged ? 0 : 2;
  }
  catch (const std::exception &e)
  {
    if (root) { std::cerr << "error: " << e.what() << std::endl; }
    return 1;
  }
}

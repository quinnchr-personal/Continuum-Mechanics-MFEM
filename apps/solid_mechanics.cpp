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
      std::cout << "mesh: dim " << pmesh->Dimension() << ", elements " << global_ne
                << ", order " << cfg.mesh.order << ", true dofs " << global_tdofs
                << ", " << physics.Description() << std::endl;
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

    const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(
      physics, *linear, cfg.solver, u,
      [&](const cmf::LoadStepReport &step, const mfem::Vector &x)
      {
        if (writer && step.newton.converged)
        {
          physics.UpdateFields(x);
          writer->Save(step.step, step.load_factor);
        }
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
    for (const cmf::ProbeConfig &probe : cfg.output.probes)
    {
      const std::vector<double> value = cmf::ProbeVector(physics.Displacement(), probe.point);
      if (root)
      {
        std::printf("probe %s at (", probe.name.c_str());
        for (std::size_t i = 0; i < probe.point.size(); i++)
        {
          std::printf("%s%g", i ? ", " : "", probe.point[i]);
        }
        std::printf("): displacement =");
        for (double v : value) { std::printf(" %.12e", v); }
        std::printf("\n");
      }
    }
    if (root && writer) { std::cout << "wrote " << cfg.output.paraview << std::endl; }
    return report.converged ? 0 : 2;
  }
  catch (const std::exception &e)
  {
    if (root) { std::cerr << "error: " << e.what() << std::endl; }
    return 1;
  }
}

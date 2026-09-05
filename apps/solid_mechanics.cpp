// Quasi-static total Lagrangian solid mechanics: YAML in, ParaView out.
// This executable only parses input and wires library objects together.
#include <iostream>
#include <memory>

#include "base/config.hpp"
#include "base/fields.hpp"
#include "base/mesh_input.hpp"
#include "base/output.hpp"
#include "mfem.hpp"

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
    const int dim = pmesh->Dimension();
    mfem::H1_FECollection fec(cfg.mesh.order, dim);
    mfem::ParFiniteElementSpace fes(pmesh.get(), &fec, dim, mfem::Ordering::byVDIM);
    const HYPRE_BigInt global_ne = pmesh->GetGlobalNE();
    const HYPRE_BigInt global_tdofs = fes.GlobalTrueVSize(); // collective
    if (root)
    {
      std::cout << "mesh: dim " << dim << ", elements " << global_ne
                << ", order " << cfg.mesh.order << ", true dofs "
                << global_tdofs << std::endl;
    }

    cmf::FieldRegistry fields;
    mfem::ParGridFunction &u =
      fields.Add("displacement", std::make_unique<mfem::ParGridFunction>(&fes));
    u = 0.0;

    if (!cfg.output.paraview.empty())
    {
      cmf::ParaViewWriter writer(cfg.output.paraview, *pmesh, cfg.mesh.order,
                                 cfg.output.high_order);
      writer.RegisterAll(cfg.output, fields);
      writer.Save();
      if (root) { std::cout << "wrote " << cfg.output.paraview << std::endl; }
    }
  }
  catch (const std::exception &e)
  {
    if (root) { std::cerr << "error: " << e.what() << std::endl; }
    return 1;
  }
  return 0;
}

// S4 gate: parallel consistency. '--write file' runs the reference case and
// records norms; '--check file' reruns (typically under mpirun) and asserts
// they match to 1e-10 relative. Solver tolerances are tightened to 1e-14;
// Newton stalls at the round-off floor of the residual before reaching
// 1e-14 relative, so the achieved reduction is reported and must be <= 1e-12.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "base/config.hpp"
#include "base/mesh_input.hpp"
#include "base/probes.hpp"
#include "materials/materials.hpp"
#include "mfem.hpp"
#include "physics/solid_mechanics_tl.hpp"
#include "solvers/linear_solver.hpp"
#include "solvers/quasi_static.hpp"
#include "test_util.hpp"

namespace
{

struct Norms
{
  double u_l2 = 0.0;
  double corner_ux = 0.0;
  double corner_uy = 0.0;
  double energy = 0.0;
};

Norms RunReference(double &residual_reduction, int &iterations)
{
  cmf::AppConfig cfg = cmf::LoadConfig("apps/input/cook.yaml");
  cfg.output.paraview.clear();
  cfg.mesh.serial_refine = 2;
  cfg.solver.newton.print_level = 0;
  cfg.solver.newton.rtol = 1e-14;
  cfg.solver.newton.atol = 0.0;
  cfg.solver.newton.max_it = 40;
  cfg.solver.linear.rtol = 1e-14;
  cfg.solver.linear.max_it = 2000;
  std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  const cmf::Material material = cmf::MakeMaterial(cfg.material);
  cmf::SolidMechanicsTL physics(*pmesh, cfg, material);
  physics.Finalize();
  cmf::LinearSolver linear(cfg.solver.linear, physics.FESpace());
  mfem::Vector u(physics.FESpace().GetTrueVSize());
  u = 0.0;
  cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(physics, linear, cfg.solver, u);
  const cmf::NewtonReport &newton = report.steps.back().newton;
  residual_reduction = newton.residual / newton.initial_residual;
  iterations = newton.iterations;
  physics.UpdateFields(u);
  Norms n;
  mfem::Vector zero(2);
  zero = 0.0;
  mfem::VectorConstantCoefficient zero_coef(zero);
  n.u_l2 = physics.Displacement().ComputeL2Error(zero_coef);
  const std::vector<double> corner = cmf::ProbeVector(physics.Displacement(), {48.0, 60.0});
  n.corner_ux = corner.at(0);
  n.corner_uy = corner.at(1);
  n.energy = physics.InternalEnergy(u);
  return n;
}

} // namespace

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  const bool root = mfem::Mpi::Root();
  const char *write_path = "";
  const char *check_path = "";
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&write_path, "-w", "--write", "Write reference norms to this file.");
  args.AddOption(&check_path, "-c", "--check", "Check norms against this file.");
  args.Parse();
  if (!args.Good() || (std::string(write_path).empty() == std::string(check_path).empty()))
  {
    if (root) { args.PrintUsage(std::cout); }
    return 1;
  }

  double reduction = 1.0;
  int iterations = 0;
  const Norms n = RunReference(reduction, iterations);
  if (root)
  {
    std::printf("np %d: residual reduced to %.2e relative in %d its, |u|_L2 %.15e, "
                "corner (%.15e, %.15e), energy %.15e\n",
                mfem::Mpi::WorldSize(), reduction, iterations,
                n.u_l2, n.corner_ux, n.corner_uy, n.energy);
  }
  CHECK_MSG(reduction <= 1e-12, "reference case residual reduced to <= 1e-12 relative");

  if (!std::string(write_path).empty())
  {
    if (root)
    {
      std::ofstream out(write_path);
      out.precision(17);
      out << std::scientific << n.u_l2 << " " << n.corner_ux << " " << n.corner_uy
          << " " << n.energy << "\n";
    }
  }
  else
  {
    Norms ref;
    std::ifstream in(check_path);
    CHECK_MSG(bool(in >> ref.u_l2 >> ref.corner_ux >> ref.corner_uy >> ref.energy),
              std::string("read reference file ") + check_path);
    auto rel = [](double a, double b) { return std::abs(a - b) / std::abs(b); };
    const double r1 = rel(n.u_l2, ref.u_l2), r2 = rel(n.corner_uy, ref.corner_uy),
                 r3 = rel(n.corner_ux, ref.corner_ux), r4 = rel(n.energy, ref.energy);
    if (root)
    {
      std::printf("relative differences vs reference: |u|_L2 %.3e, corner ux %.3e, uy %.3e, energy %.3e\n",
                  r1, r3, r2, r4);
    }
    CHECK_MSG(r1 <= 1e-10, "|u|_L2 matches serial reference to 1e-10");
    CHECK_MSG(r2 <= 1e-10, "corner uy matches serial reference to 1e-10");
    CHECK_MSG(r3 <= 1e-10, "corner ux matches serial reference to 1e-10");
    CHECK_MSG(r4 <= 1e-10, "energy matches serial reference to 1e-10");
  }
  int code = cmf_test::Report(root ? "test_parallel" : "test_parallel (rank)");
  int global = 0;
  MPI_Allreduce(&code, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return global;
}

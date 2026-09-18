// S4 gate: parallel consistency. '--write file' runs the reference case and
// records norms; '--check file' reruns (typically under mpirun) and asserts
// they match to 1e-10 relative. The linear solver tolerance is tightened to
// 1e-14 and Newton to 1e-12 relative (its round-off floor is ~3e-13 on these
// problems, so 1e-14 would stall and truncate the load path); the achieved
// reduction is reported. The displacement and the mixed u-p Cook's membrane
// inputs and their small-strain counterparts (linear_elastic) are checked;
// the latter are linear and may be accepted at the round-off floor of their
// residual, just above 1e-12 here.
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
#include "physics/solid_problem.hpp"
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

Norms RunReference(const std::string &input, double &residual_reduction, int &iterations,
                   bool &all_steps_converged, bool &at_floor)
{
  cmf::AppConfig cfg = cmf::LoadConfig(input);
  cfg.output.paraview.clear();
  cfg.mesh.serial_refine = 2;
  cfg.solver.newton.print_level = 0;
  cfg.solver.newton.rtol = 1e-12;
  cfg.solver.newton.atol = 0.0;
  cfg.solver.newton.max_it = 40;
  cfg.solver.linear.rtol = 1e-14;
  cfg.solver.linear.max_it = 2000;
  std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*pmesh, cfg);
  cmf::SolidProblem &physics = *problem;
  physics.Finalize();
  std::unique_ptr<mfem::Solver> linear = physics.MakeLinearSolver(cfg.solver.linear);
  mfem::Vector u(physics.Height());
  u = 0.0;
  cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(physics, *linear, cfg.solver, u);
  const cmf::NewtonReport &newton = report.steps.back().newton;
  residual_reduction = newton.residual / newton.initial_residual;
  iterations = newton.iterations;
  at_floor = newton.at_floor;
  all_steps_converged = report.converged;
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

  const std::vector<std::string> inputs = {"apps/input/finite_elasticity/cooks_membrane/cook.yaml",
                                           "apps/input/finite_elasticity/cooks_membrane/cook_incompressible.yaml",
                                           "apps/input/linear_elasticity/cooks_membrane/cook_linear.yaml",
                                           "apps/input/linear_elasticity/cooks_membrane/cook_linear_incompressible.yaml"};
  std::vector<Norms> norms;
  for (const std::string &input : inputs)
  {
    double reduction = 1.0;
    int iterations = 0;
    bool all_steps = false, at_floor = false;
    const Norms n = RunReference(input, reduction, iterations, all_steps, at_floor);
    norms.push_back(n);
    CHECK_MSG(all_steps, input + ": every load step converged");
    if (root)
    {
      std::printf("%s np %d: residual reduced to %.2e relative in %d its, |u|_L2 %.15e, "
                  "corner (%.15e, %.15e), energy %.15e\n", input.c_str(),
                  mfem::Mpi::WorldSize(), reduction, iterations,
                  n.u_l2, n.corner_ux, n.corner_uy, n.energy);
    }
    // A linear problem may meet the round-off floor of its residual first
    // (NewtonReport::at_floor); the norms below are compared either way.
    CHECK_MSG(reduction <= 1e-12 || (at_floor && reduction <= 1e-10),
              input + ": residual reduced to <= 1e-12 relative, or to the floor of a linear problem");
  }

  if (!std::string(write_path).empty())
  {
    if (root)
    {
      std::ofstream out(write_path);
      out.precision(17);
      for (const Norms &n : norms)
      {
        out << std::scientific << n.u_l2 << " " << n.corner_ux << " " << n.corner_uy
            << " " << n.energy << "\n";
      }
    }
  }
  else
  {
    std::ifstream in(check_path);
    for (std::size_t c = 0; c < inputs.size(); c++)
    {
      Norms ref;
      const Norms &n = norms[c];
      CHECK_MSG(bool(in >> ref.u_l2 >> ref.corner_ux >> ref.corner_uy >> ref.energy),
                std::string("read reference file ") + check_path);
      auto rel = [](double a, double b) { return std::abs(a - b) / std::abs(b); };
      const double r1 = rel(n.u_l2, ref.u_l2), r2 = rel(n.corner_uy, ref.corner_uy),
                   r3 = rel(n.corner_ux, ref.corner_ux), r4 = rel(n.energy, ref.energy);
      if (root)
      {
        std::printf("%s: relative differences vs reference: |u|_L2 %.3e, corner ux %.3e, uy %.3e, energy %.3e\n",
                    inputs[c].c_str(), r1, r3, r2, r4);
      }
      CHECK_MSG(r1 <= 1e-10, inputs[c] + ": |u|_L2 matches serial reference to 1e-10");
      CHECK_MSG(r2 <= 1e-10, inputs[c] + ": corner uy matches serial reference to 1e-10");
      CHECK_MSG(r3 <= 1e-10, inputs[c] + ": corner ux matches serial reference to 1e-10");
      CHECK_MSG(r4 <= 1e-10, inputs[c] + ": energy matches serial reference to 1e-10");
    }
  }
  int code = cmf_test::Report(root ? "test_parallel" : "test_parallel (rank)");
  int global = 0;
  MPI_Allreduce(&code, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return global;
}

// S4 gate: Cook's membrane self-convergence with a frozen regression value,
// and the 3D cantilever small-load linear limit (Euler-Bernoulli sanity and
// NeoHookean vs StVenantKirchhoff agreement). Both are driven by the YAML
// inputs under apps/input through the same library wiring as the app.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "base/config.hpp"
#include "base/expression.hpp"
#include "base/mesh_input.hpp"
#include "base/probes.hpp"
#include "kernels/total_lagrangian.hpp"
#include "materials/materials.hpp"
#include "mfem.hpp"
#include "physics/solid_problem.hpp"
#include "solvers/linear_solver.hpp"
#include "solvers/quasi_static.hpp"
#include "test_util.hpp"

namespace
{

// Frozen regression oracle: Cook's membrane top-right corner vertical
// displacement on the finest self-convergence mesh (64x64, p = 2, traction
// 3.75 per unit length; see CookTest). Set once from the measured value;
// asserted within 1e-8 relative thereafter.
const double kCookCornerFrozen = 4.905891700497e+00;
const int kCookFinestRefine = 4;

// Frozen regression oracle of the mixed u-p, fully incompressible Cook's
// membrane (apps/input/cook_incompressible.yaml, mu = 80.194, resultant 100)
// on its finest self-convergence mesh (32x32, Q2-Q1).
const double kCookIncompressibleCornerFrozen = 6.930412595013e+00;
const int kCookIncompressibleFinestRefine = 3;

struct Run
{
  cmf::QuasiStaticReport report;
  std::vector<double> probe;
  double u_l2 = 0.0;
  double max_grad = 0.0;
  int ndofs = 0;
  std::string amg_mode;
};

// Max |Grad u| over all quadrature points (serial or local + MPI max).
double MaxDisplacementGradient(mfem::ParGridFunction &u)
{
  mfem::ParFiniteElementSpace &fes = *u.ParFESpace();
  mfem::DenseMatrix H;
  double local = 0.0;
  for (int e = 0; e < fes.GetNE(); e++)
  {
    const mfem::FiniteElement &fe = *fes.GetFE(e);
    mfem::ElementTransformation &T = *fes.GetElementTransformation(e);
    const mfem::IntegrationRule &ir =
      mfem::IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder() + 3);
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      T.SetIntPoint(&ir.IntPoint(q));
      u.GetVectorGradient(T, H);
      local = std::max(local, H.MaxMaxNorm());
    }
  }
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, fes.GetComm());
  return global;
}

Run Solve(const cmf::AppConfig &cfg, const std::vector<double> &probe_point)
{
  std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  std::unique_ptr<cmf::SolidProblem> problem = cmf::MakeSolidProblem(*pmesh, cfg);
  cmf::SolidProblem &physics = *problem;
  physics.Finalize();
  std::unique_ptr<mfem::Solver> linear = physics.MakeLinearSolver(cfg.solver.linear);
  mfem::Vector u(physics.Height());
  u = 0.0;
  Run r;
  r.report = cmf::SolveQuasiStatic(physics, *linear, cfg.solver, u);
  r.ndofs = int(physics.GlobalTrueVSize());
  if (auto *ls = dynamic_cast<cmf::LinearSolver *>(linear.get())) { r.amg_mode = ls->ActiveAMG(); }
  physics.UpdateFields(u);
  mfem::Vector zero(pmesh->Dimension());
  zero = 0.0;
  mfem::VectorConstantCoefficient zero_coef(zero);
  r.u_l2 = physics.Displacement().ComputeL2Error(zero_coef);
  r.probe = cmf::ProbeVector(physics.Displacement(), probe_point);
  r.max_grad = MaxDisplacementGradient(physics.Displacement());
  return r;
}

// Mixed u-p, fully incompressible Cook's membrane: monotone self-convergence
// of the corner displacement and a frozen regression value.
void CookIncompressibleTest()
{
  cmf::AppConfig cfg = cmf::LoadConfig("apps/input/cook_incompressible.yaml");
  cfg.output.paraview.clear();
  cfg.solver.newton.print_level = 0;
  cfg.solver.newton.rtol = 1e-11;
  cfg.solver.linear.rtol = 1e-13;
  std::vector<double> corner;
  for (int refine = 0; refine <= kCookIncompressibleFinestRefine; refine++)
  {
    cfg.mesh.serial_refine = refine;
    Run r = Solve(cfg, {48.0, 60.0});
    CHECK_MSG(r.report.converged, "cook incompressible refine " + std::to_string(refine) + " converged");
    corner.push_back(r.probe.at(1));
    std::printf("  cook incompressible refine %d (%dx%d Q2-Q1, %d dofs): corner uy = %.12e "
                "(%.1f%% of 16), |u|_L2 = %.10e, newton its %d\n",
                refine, 4 << refine, 4 << refine, r.ndofs, r.probe.at(1),
                100.0 * r.probe.at(1) / 16.0, r.u_l2, r.report.steps.back().newton.iterations);
  }
  bool monotone = true;
  for (std::size_t k = 0; k + 2 < corner.size(); k++)
  {
    const double d1 = corner[k + 1] - corner[k], d2 = corner[k + 2] - corner[k + 1];
    if (d1 * d2 <= 0.0) { monotone = false; }
    std::printf("  cook incompressible difference %zu: %.3e, ratio to next %.3f\n", k + 1, d1,
                std::abs(d1) / std::abs(d2));
  }
  std::printf("  cook incompressible difference %zu: %.3e\n", corner.size() - 1,
              corner.back() - corner[corner.size() - 2]);
  CHECK_MSG(monotone, "cook incompressible corner displacement converges monotonically");
  const double finest = corner.back();
  std::printf("  cook incompressible finest corner uy = %.12e (frozen %.12e, rel diff %.3e)\n", finest,
              kCookIncompressibleCornerFrozen,
              std::abs(finest - kCookIncompressibleCornerFrozen) / std::abs(kCookIncompressibleCornerFrozen));
  CHECK_MSG(std::abs(finest - kCookIncompressibleCornerFrozen) <=
              1e-8 * std::abs(kCookIncompressibleCornerFrozen),
            "cook incompressible frozen regression value within 1e-8 relative");
}

// With cook_ratio_gate the plan's ">= 3x per refinement" threshold is asserted
// (make test runs this as its final step); otherwise the ratios are printed
// against the threshold and the remaining Cook checks are asserted.
void CookTest(bool cook_ratio_gate)
{
  cmf::AppConfig cfg = cmf::LoadConfig("apps/input/cook.yaml");
  cfg.output.paraview.clear();
  cfg.solver.newton.print_level = 0;
  cfg.solver.newton.rtol = 1e-11;
  cfg.solver.linear.rtol = 1e-13;
  const double traction = cmf::Expression::Parse(cfg.bcs.traction.at(0).expression.at(1)).Eval(0, 0, 0, 1);
  std::printf("  cook: traction %.4f per unit length, resultant %.3f\n", traction, 16.0 * traction);
  // Base 4x4 mesh plus 4 uniform refinements (up to 64x64, p = 2).
  std::vector<double> corner;
  for (int refine = 0; refine <= kCookFinestRefine; refine++)
  {
    cfg.mesh.serial_refine = refine;
    Run r = Solve(cfg, {48.0, 60.0});
    CHECK_MSG(r.report.converged, "cook refine " + std::to_string(refine) + " converged");
    corner.push_back(r.probe.at(1));
    std::printf("  cook refine %d (%dx%d p=2, %d dofs): corner uy = %.12e (%.1f%% of 16), "
                "|u|_L2 = %.10e, newton its %d\n",
                refine, 4 << refine, 4 << refine, r.ndofs, r.probe.at(1),
                100.0 * r.probe.at(1) / 16.0, r.u_l2,
                r.report.steps.back().newton.iterations);
  }
  // Plan gate: monotone convergence with successive differences shrinking by
  // >= 3x per refinement. Uniform refinement of this geometry is limited by
  // the clamped-free corner singularity (exponent ~0.66 at the 108 degree
  // top-left corner, point-value rate ~h^1.3, ratio ~2.5), so the measured
  // ratios are reported against the plan's threshold without relaxing it.
  std::vector<double> diffs;
  for (std::size_t k = 0; k + 1 < corner.size(); k++)
  {
    diffs.push_back(corner[k + 1] - corner[k]);
  }
  bool monotone = true;
  for (std::size_t k = 0; k + 1 < diffs.size(); k++)
  {
    if (diffs[k] * diffs[k + 1] <= 0.0) { monotone = false; }
    const double ratio = std::abs(diffs[k]) / std::abs(diffs[k + 1]);
    std::printf("  cook difference %zu: %.3e, ratio to next %.3f (plan gate >= 3: %s)\n",
                k + 1, diffs[k], ratio, ratio >= 3.0 ? "met" : "NOT MET");
    if (cook_ratio_gate)
    {
      CHECK_MSG(ratio >= 3.0, "GATE NOT MET: cook successive-difference ratio " +
                std::to_string(ratio) + " < 3 (singularity-limited uniform refinement)");
    }
  }
  std::printf("  cook difference %zu: %.3e\n", diffs.size(), diffs.back());
  CHECK_MSG(monotone, "cook corner displacement converges monotonically");
  const double finest = corner.back();
  const double fraction = finest / 16.0;
  CHECK_MSG(fraction >= 0.25 && fraction <= 0.35,
            "cook corner deflection fraction " + std::to_string(fraction) + " in [0.25, 0.35]");
  std::printf("  cook finest corner uy = %.12e (frozen %.12e, rel diff %.3e)\n", finest,
              kCookCornerFrozen, std::abs(finest - kCookCornerFrozen) / std::abs(kCookCornerFrozen));
  CHECK_MSG(std::abs(finest - kCookCornerFrozen) <= 1e-8 * std::abs(kCookCornerFrozen),
            "cook frozen regression value within 1e-8 relative");
}

void CantileverTest()
{
  cmf::AppConfig cfg = cmf::LoadConfig("apps/input/cantilever3d.yaml");
  cfg.output.paraview.clear();
  cfg.solver.newton.print_level = 0;
  cfg.solver.linear.rtol = 1e-13;
  // Beam dimensions from the mesh file's bounding box.
  mfem::Vector bb_min, bb_max;
  cmf::BuildSerialMesh(cfg.mesh)->GetBoundingBox(bb_min, bb_max);
  const double L = bb_max(0) - bb_min(0), w = bb_max(1) - bb_min(1), h = bb_max(2) - bb_min(2);
  const double P = -cmf::Expression::Parse(cfg.bcs.traction.at(0).expression.at(2)).Eval(0, 0, 0, 1) * w * h; // resultant
  const double I = w * h * h * h / 12.0;
  const double euler_bernoulli = P * L * L * L / (3.0 * cfg.material.E * I);

  cfg.material.model = "neo_hookean";
  Run nh = Solve(cfg, {L, 0.5 * w, 0.5 * h});
  cfg.material.model = "st_venant_kirchhoff";
  Run svk = Solve(cfg, {L, 0.5 * w, 0.5 * h});
  CHECK_MSG(nh.report.converged && svk.report.converged, "cantilever solves converged");
  const double tip_nh = -nh.probe.at(2), tip_svk = -svk.probe.at(2);
  const double eb_err = std::abs(tip_nh - euler_bernoulli) / euler_bernoulli;
  const double rel = std::abs(tip_nh - tip_svk) / std::abs(tip_nh);
  std::printf("  cantilever (%d dofs): max |Grad u| = %.3e, tip NH = %.10e, StVK = %.10e, "
              "Euler-Bernoulli = %.10e (NH error %.2f%%), NH vs StVK rel %.3e\n",
              nh.ndofs, nh.max_grad, tip_nh, tip_svk, euler_bernoulli, 100.0 * eb_err, rel);
  CHECK_MSG(nh.max_grad <= 1e-4, "cantilever load is in the linear limit (max |Grad u| <= 1e-4)");
  CHECK_MSG(eb_err <= 0.15, "cantilever tip vs Euler-Bernoulli within 15%");
  CHECK_MSG(rel <= 1e-6, "cantilever NH vs StVK tip within 1e-6 relative");

  // The elasticity (rigid-body-mode) AMG options stall on this mesh; the
  // linear solver must fall back to the systems options and still converge.
  cfg.material.model = "neo_hookean";
  cfg.solver.linear.amg = "elasticity";
  cfg.solver.linear.max_it = 100;
  Run fb = Solve(cfg, {L, 0.5 * w, 0.5 * h});
  std::printf("  cantilever with elasticity AMG requested: active AMG '%s', converged %s, "
              "tip %.10e\n", fb.amg_mode.c_str(), fb.report.converged ? "yes" : "no",
              -fb.probe.at(2));
  CHECK_MSG(fb.report.converged, "cantilever converges after the AMG fallback");
  CHECK_MSG(fb.amg_mode == "systems", "linear solver fell back to systems AMG options");
  CHECK_MSG(std::abs(-fb.probe.at(2) - tip_nh) <= 1e-8 * tip_nh,
            "fallback solve reproduces the tip displacement");
}

} // namespace

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  bool cook_ratio_gate = false;
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&cook_ratio_gate, "-g", "--cook-ratio-gate", "-no-g",
                 "--no-cook-ratio-gate",
                 "Assert the plan's >= 3x successive-difference ratio for Cook's membrane.");
  args.Parse();
  if (!args.Good())
  {
    if (mfem::Mpi::Root()) { args.PrintUsage(std::cout); }
    return 1;
  }
  std::cout << "cook's membrane" << (cook_ratio_gate ? " (ratio gate asserted)" : "") << std::endl;
  CookTest(cook_ratio_gate);
  if (!cook_ratio_gate)
  {
    std::cout << "3d cantilever linear limit" << std::endl;
    CantileverTest();
    std::cout << "cook's membrane, mixed u-p incompressible" << std::endl;
    CookIncompressibleTest();
  }
  const int code = cmf_test::Report(mfem::Mpi::Root() ? "test_benchmarks" : "test_benchmarks (rank)");
  int global = 0;
  MPI_Allreduce(&code, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return global;
}

// Manufactured-solution diffusion driver on a static mesh (no ALE).
//
// Solves:
//   du/dt - alpha * Laplacian(u) = f   on [0,1]^2
// using backward Euler in time on a fixed (non-moving) mesh.
//
// Exact solution:
//   u(x,y,t) = sin(t) * cos(2*(x-0.5)^2 + 2*(y-0.5)^2)
//
// Dirichlet BCs are applied on the entire boundary from the exact solution.
// ParaView output includes the numerical solution, exact solution, and
// pointwise error at every time step.
//
// Configuration is read from a YAML file (see Input/input_diffusion_mms.yaml).

#include "mfem.hpp"

#include <yaml-cpp/yaml.h>

#ifndef MFEM_USE_PETSC
#error "This driver requires MFEM built with PETSc."
#endif
#include <petscksp.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using namespace std;
using namespace mfem;

namespace
{

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

struct DriverParams
{
   string mesh_file;
   int order = 1;
   int serial_ref_levels = 0;
   int par_ref_levels = 0;

   double alpha = 0.1;
   double dt = 0.01;
   double t_final = 2.0;

   string petsc_options_file = "Input/petsc.opts";
   string output_path = "ParaView/diffusion_mms";
   bool save_paraview = true;
};

void LoadParams(const string &path, DriverParams &p)
{
   if (path.empty())
   {
      throw runtime_error("Input YAML file path is empty.");
   }
   if (!filesystem::exists(path))
   {
      throw runtime_error("YAML input file not found: " + path);
   }

   YAML::Node n = YAML::LoadFile(path);

   if (!n["mesh_file"])
   {
      throw runtime_error("Missing required YAML key: mesh_file");
   }
   p.mesh_file = n["mesh_file"].as<string>();
   if (p.mesh_file.empty())
   {
      throw runtime_error("YAML key mesh_file is empty.");
   }

   if (n["order"])             { p.order = n["order"].as<int>(); }
   if (n["serial_ref_levels"]) { p.serial_ref_levels = n["serial_ref_levels"].as<int>(); }
   if (n["par_ref_levels"])    { p.par_ref_levels = n["par_ref_levels"].as<int>(); }
   if (n["alpha"])             { p.alpha = n["alpha"].as<double>(); }
   if (n["dt"])                { p.dt = n["dt"].as<double>(); }
   if (n["t_final"])           { p.t_final = n["t_final"].as<double>(); }
   if (n["petsc_options_file"]){ p.petsc_options_file = n["petsc_options_file"].as<string>(); }
   if (n["output_path"])       { p.output_path = n["output_path"].as<string>(); }
   if (n["save_paraview"])     { p.save_paraview = n["save_paraview"].as<bool>(); }

   if (p.order < 1)
   {
      throw runtime_error("order must be >= 1.");
   }
   if (p.serial_ref_levels < 0 || p.par_ref_levels < 0)
   {
      throw runtime_error("serial_ref_levels and par_ref_levels must be >= 0.");
   }
   if (p.alpha <= 0.0)
   {
      throw runtime_error("alpha must be > 0.");
   }
   if (p.dt <= 0.0)
   {
      throw runtime_error("dt must be > 0.");
   }
   if (p.t_final < 0.0)
   {
      throw runtime_error("t_final must be >= 0.");
   }
}

void PrintConfig(const DriverParams &p)
{
   cout << "Manufactured-solution diffusion driver (backward Euler, no ALE)" << endl;
   cout << "  mesh_file:          " << p.mesh_file << endl;
   cout << "  order:              " << p.order << endl;
   cout << "  serial_ref_levels:  " << p.serial_ref_levels << endl;
   cout << "  par_ref_levels:     " << p.par_ref_levels << endl;
   cout << "  alpha:              " << p.alpha << endl;
   cout << "  dt:                 " << p.dt << endl;
   cout << "  t_final:            " << p.t_final << endl;
   cout << "  petsc_options_file: " << p.petsc_options_file << endl;
   cout << "  output_path:        " << p.output_path << endl;
   cout << "  save_paraview:      " << (p.save_paraview ? "true" : "false") << endl;
}

// ---------------------------------------------------------------------------
// Exact solution and forcing coefficients
// ---------------------------------------------------------------------------
// Exact:   u(x,y,t) = sin(t) * cos(q),  q = 2*(x-0.5)^2 + 2*(y-0.5)^2
// Forcing: f = du/dt - alpha * Lap(u)

class ExactCoefficient : public Coefficient
{
public:
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x;
      T.Transform(ip, x);
      const double t  = GetTime();
      const double dx = x[0] - 0.5;
      const double dy = x[1] - 0.5;
      const double q  = 2.0 * dx * dx + 2.0 * dy * dy;
      return sin(t) * cos(q);
   }
};

class ForcingCoefficient : public Coefficient
{
public:
   explicit ForcingCoefficient(double alpha) : alpha_(alpha) {}

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x;
      T.Transform(ip, x);
      const double t  = GetTime();
      const double dx = x[0] - 0.5;
      const double dy = x[1] - 0.5;
      const double r2 = dx * dx + dy * dy;
      const double q  = 2.0 * r2;

      // du/dt = cos(t) * cos(q)
      const double ut = cos(t) * cos(q);

      // Laplacian(u) = sin(t) * [-16*r2*cos(q) - 8*sin(q)]
      const double lap = sin(t) * (-16.0 * r2 * cos(q) - 8.0 * sin(q));

      // f = du/dt - alpha * Lap(u)
      return ut - alpha_ * lap;
   }

private:
   double alpha_;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Mark all boundary attributes as essential (Dirichlet on entire boundary).
void BuildAllBoundaryMarker(const ParMesh &pmesh, Array<int> &ess_bdr)
{
   const int nbdr = pmesh.bdr_attributes.Max();
   MFEM_VERIFY(nbdr > 0, "Mesh must define boundary attributes.");
   ess_bdr.SetSize(nbdr);
   ess_bdr = 1;
}

} // namespace

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const int myid = Mpi::WorldRank();

   // --- Parse command line ---------------------------------------------------
   string input_file = "Input/input_diffusion_mms.yaml";
   OptionsParser args(argc, argv);
   args.AddOption(&input_file, "-i", "--input", "YAML input file.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // --- Load YAML configuration ---------------------------------------------
   DriverParams params;
   try
   {
      LoadParams(input_file, params);
   }
   catch (const exception &e)
   {
      if (myid == 0) { cerr << e.what() << endl; }
      return 2;
   }
   if (myid == 0) { PrintConfig(params); }

   // --- Initialize PETSc ----------------------------------------------------
   const char *petsc_file_to_use = nullptr;
   if (!params.petsc_options_file.empty())
   {
      ifstream petsc_in(params.petsc_options_file);
      if (petsc_in.good())
      {
         petsc_file_to_use = params.petsc_options_file.c_str();
      }
      else if (myid == 0)
      {
         cerr << "PETSc options file not found: " << params.petsc_options_file
              << ". Proceeding without options file." << endl;
      }
   }
   MFEMInitializePetsc(&argc, &argv, petsc_file_to_use, NULL);

   int exit_code = 0;
   try
   {
      Device device("cpu");
      if (myid == 0) { device.Print(); }

      // --- Mesh --------------------------------------------------------------
      unique_ptr<Mesh> mesh = make_unique<Mesh>(params.mesh_file.c_str(), 1, 1);
      if (mesh->Dimension() != 2)
      {
         throw runtime_error("The mesh must be 2D.");
      }
      for (int l = 0; l < params.serial_ref_levels; l++)
      {
         mesh->UniformRefinement();
      }

      unique_ptr<ParMesh> pmesh = make_unique<ParMesh>(MPI_COMM_WORLD, *mesh);
      mesh.reset();
      for (int l = 0; l < params.par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }

      MFEM_VERIFY(pmesh->bdr_attributes.Size() > 0,
                   "Mesh must define boundary attributes.");

      // --- Finite element space ----------------------------------------------
      H1_FECollection fec(params.order, pmesh->Dimension());
      ParFiniteElementSpace fespace(pmesh.get(), &fec);
      const HYPRE_BigInt global_true_dofs = fespace.GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "Global true dofs: " << global_true_dofs << endl;
      }

      // --- Essential (Dirichlet) boundaries ----------------------------------
      Array<int> ess_bdr;
      BuildAllBoundaryMarker(*pmesh, ess_bdr);
      Array<int> ess_tdof_list;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      // --- Coefficients ------------------------------------------------------
      ExactCoefficient exact_coeff;
      ForcingCoefficient forcing_coeff(params.alpha);
      ConstantCoefficient alpha_dt_coeff(params.alpha * params.dt);

      // --- Mass matrix (constant on static mesh) ----------------------------
      ParBilinearForm mass_form(&fespace);
      mass_form.AddDomainIntegrator(new MassIntegrator());
      mass_form.Assemble();
      mass_form.Finalize();

      // --- LHS bilinear form: M + alpha*dt*K (constant on static mesh) ------
      ParBilinearForm lhs_form(&fespace);
      lhs_form.AddDomainIntegrator(new MassIntegrator());
      lhs_form.AddDomainIntegrator(new DiffusionIntegrator(alpha_dt_coeff));
      lhs_form.Assemble();
      lhs_form.Finalize();

      // --- Grid functions ----------------------------------------------------
      ParGridFunction u(&fespace);
      ParGridFunction u_exact(&fespace);
      ParGridFunction u_error(&fespace);

      // Set initial condition: u(x,y,0) = sin(0)*cos(...) = 0
      exact_coeff.SetTime(0.0);
      u.ProjectCoefficient(exact_coeff);
      u_exact.ProjectCoefficient(exact_coeff);
      u_error = 0.0;

      // --- Integration rules for accurate error computation ------------------
      const int order_quad = max(2, 2 * params.order + 3);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int g = 0; g < Geometry::NumGeom; g++)
      {
         irs[g] = &IntRules.Get(g, order_quad);
      }

      // --- ParaView output ---------------------------------------------------
      unique_ptr<ParaViewDataCollection> paraview_dc;
      if (params.save_paraview)
      {
         error_code ec;
         filesystem::create_directories(params.output_path, ec);
         if (ec)
         {
            throw runtime_error("Failed to create output directory: "
                                + params.output_path + " (" + ec.message() + ")");
         }

         paraview_dc = make_unique<ParaViewDataCollection>("diffusion_mms",
                                                           pmesh.get());
         paraview_dc->SetPrefixPath(params.output_path);
         paraview_dc->SetLevelsOfDetail(params.order);
         paraview_dc->SetDataFormat(VTKFormat::BINARY);
         paraview_dc->SetHighOrderOutput(true);
         paraview_dc->RegisterField("u", &u);
         paraview_dc->RegisterField("u_exact", &u_exact);
         paraview_dc->RegisterField("error", &u_error);
      }

      // --- Error CSV ---------------------------------------------------------
      ofstream err_csv;
      if (myid == 0)
      {
         const filesystem::path csv_path =
            filesystem::path(params.output_path) / "error_history.csv";
         error_code ec;
         filesystem::create_directories(params.output_path, ec);
         err_csv.open(csv_path);
         if (!err_csv)
         {
            throw runtime_error("Failed to open error CSV: " + csv_path.string());
         }
         err_csv << "step,time,l2_error,linf_error\n";
         err_csv << setprecision(16);
      }

      // --- Helpers to write errors and save ParaView -------------------------
      auto compute_and_save = [&](int step, double t)
      {
         exact_coeff.SetTime(t);

         // Compute L2 error
         const double l2_err = u.ComputeL2Error(exact_coeff, irs);

         // Project exact solution and pointwise error for visualization
         u_exact.ProjectCoefficient(exact_coeff);
         // error = u_numerical - u_exact (pointwise on grid function)
         subtract(u, u_exact, u_error);

         // Compute Linf error from the error grid function
         const double local_linf = u_error.Normlinf();
         double linf_err = 0.0;
         MPI_Allreduce(&local_linf, &linf_err, 1, MPI_DOUBLE, MPI_MAX,
                        MPI_COMM_WORLD);

         // Write CSV
         if (myid == 0)
         {
            err_csv << step << "," << t << ","
                    << l2_err << "," << linf_err << "\n";
            err_csv.flush();

            if (step <= 5 || step % 50 == 0)
            {
               cout << "step=" << step << " t=" << t
                    << " L2_error=" << l2_err
                    << " Linf_error=" << linf_err << endl;
            }
         }

         // Save ParaView
         if (paraview_dc)
         {
            paraview_dc->SetCycle(step);
            paraview_dc->SetTime(t);
            paraview_dc->Save();
         }
      };

      // --- Save initial state ------------------------------------------------
      compute_and_save(0, 0.0);

      // --- Time stepping loop ------------------------------------------------
      const int nsteps = static_cast<int>(ceil(params.t_final / params.dt - 1.0e-12));
      if (myid == 0)
      {
         cout << "Time steps: " << nsteps
              << ", dt=" << params.dt
              << ", t_final=" << (nsteps * params.dt) << endl;
      }

      Vector rhs_local(fespace.GetVSize());
      OperatorHandle Ah(Operator::Hypre_ParCSR);
      Vector X, B;

      for (int step = 1; step <= nsteps; step++)
      {
         const double t = step * params.dt;

         // RHS = M * u_old
         mass_form.Mult(u, rhs_local);

         // Add dt * (f^{n+1}, v)
         forcing_coeff.SetTime(t);
         ParLinearForm f_form(&fespace);
         f_form.AddDomainIntegrator(new DomainLFIntegrator(forcing_coeff));
         f_form.Assemble();
         rhs_local.Add(params.dt, f_form);

         // Apply Dirichlet BCs: project exact solution on boundary DOFs
         exact_coeff.SetTime(t);
         u.ProjectBdrCoefficient(exact_coeff, ess_bdr);

         // Form the constrained linear system
         lhs_form.FormLinearSystem(ess_tdof_list, u, rhs_local, Ah, X, B);

         // Solve
         HypreParMatrix *A_hyp = Ah.As<HypreParMatrix>();
         MFEM_VERIFY(A_hyp != nullptr, "Expected HypreParMatrix.");
         PetscParMatrix A_petsc(A_hyp, Operator::PETSC_MATAIJ);
         PetscLinearSolver solver(A_petsc);
         solver.SetPrintLevel(0);
         solver.Mult(B, X);
         MFEM_VERIFY(solver.GetConverged(),
                     "PETSc solver did not converge at step " << step
                     << ". Iterations=" << solver.GetNumIterations()
                     << ", residual=" << solver.GetFinalNorm());

         // Recover FEM solution
         lhs_form.RecoverFEMSolution(X, rhs_local, u);

         // Compute errors and save
         compute_and_save(step, t);
      }

      // --- Final summary -----------------------------------------------------
      if (myid == 0)
      {
         const double t_end = nsteps * params.dt;
         exact_coeff.SetTime(t_end);
         const double final_l2 = u.ComputeL2Error(exact_coeff, irs);
         cout << "\nFinal L2 error at t=" << t_end << ": " << final_l2 << endl;
         cout << "Output written to: " << params.output_path << endl;
      }
   }
   catch (const exception &e)
   {
      if (myid == 0) { cerr << "Error: " << e.what() << endl; }
      exit_code = 3;
   }

   MFEMFinalizePetsc();
   return exit_code;
}

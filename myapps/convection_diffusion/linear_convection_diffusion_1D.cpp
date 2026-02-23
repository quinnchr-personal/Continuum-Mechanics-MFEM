// 2D transient convection-diffusion driver with 3 uncoupled Peclet cases.
//
// Solves:
//   d c / d t + beta · grad(c) - (1/Pe) Delta c = 0 on (0,1)^2
// using backward Euler in time.
//
// Boundary conditions:
//   - Dirichlet (analytical) on x=0 and x=1
//   - Natural zero-Neumann on y=0 and y=1
// where the analytical expression is from Homework_4-4.pdf (Problem 3, Part 1),
// interpreted as a function of x and t (uniform in y).

#include "mfem.hpp"

#include <yaml-cpp/yaml.h>

#ifndef MFEM_USE_PETSC
#error "This driver requires MFEM built with PETSc."
#endif
#include <petscksp.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

using namespace std;
using namespace mfem;

namespace
{

struct DriverParams
{
   string mesh_file;
   int order = 1;
   int serial_ref_levels = 0;
   int par_ref_levels = 0;
   double dt = 1.0e-3;
   double t_final = 1.0;
   array<double, 3> peclet = {1.0, 10.0, 100.0};
   string petsc_options_file = "Input/petsc.opts";
   string output_path = "ParaView";
   string collection_name = "convection_diffusion_3pe";
   string error_csv = "error_history.csv";
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

   if (n["order"]) { p.order = n["order"].as<int>(); }
   if (n["serial_ref_levels"]) { p.serial_ref_levels = n["serial_ref_levels"].as<int>(); }
   if (n["par_ref_levels"]) { p.par_ref_levels = n["par_ref_levels"].as<int>(); }
   if (n["dt"]) { p.dt = n["dt"].as<double>(); }
   if (n["t_final"]) { p.t_final = n["t_final"].as<double>(); }
   if (n["petsc_options_file"]) { p.petsc_options_file = n["petsc_options_file"].as<string>(); }
   if (n["output_path"]) { p.output_path = n["output_path"].as<string>(); }
   if (n["collection_name"]) { p.collection_name = n["collection_name"].as<string>(); }
   if (n["error_csv"]) { p.error_csv = n["error_csv"].as<string>(); }
   if (n["save_paraview"]) { p.save_paraview = n["save_paraview"].as<bool>(); }

   if (n["peclet"])
   {
      const YAML::Node pe = n["peclet"];
      if (!pe.IsSequence() || pe.size() != 3)
      {
         throw runtime_error("YAML key peclet must be a sequence of exactly 3 values.");
      }
      for (int i = 0; i < 3; i++)
      {
         p.peclet[i] = pe[i].as<double>();
      }
   }

   if (p.order < 1)
   {
      throw runtime_error("order must be >= 1.");
   }
   if (p.serial_ref_levels < 0 || p.par_ref_levels < 0)
   {
      throw runtime_error("serial_ref_levels and par_ref_levels must be >= 0.");
   }
   if (p.dt <= 0.0)
   {
      throw runtime_error("dt must be > 0.");
   }
   if (p.t_final < 0.0)
   {
      throw runtime_error("t_final must be >= 0.");
   }
   for (int i = 0; i < 3; i++)
   {
      if (p.peclet[i] <= 0.0)
      {
         throw runtime_error("All Peclet values must be > 0.");
      }
   }
}

double ExpTimesErfc(const double a, const double b)
{
   // Use asymptotic form for large b to avoid inf*0 indeterminate products.
   if (b > 26.0)
   {
      const double inv_b = 1.0 / b;
      const double inv_b2 = inv_b * inv_b;
      const double erfc_asym =
         inv_b / std::sqrt(M_PI) * (1.0 - 0.5 * inv_b2 + 0.75 * inv_b2 * inv_b2);
      const double expo = a - b * b;
      if (expo < -745.0) { return 0.0; }
      if (expo > 709.0) { return numeric_limits<double>::infinity(); }
      return std::exp(expo) * erfc_asym;
   }
   if (a > 709.0) { return numeric_limits<double>::infinity(); }
   return std::exp(a) * std::erfc(b);
}

double ExactConcentration(const double x, const double t, const double pe)
{
   if (t <= 0.0) { return 0.0; }

   const double diff = t / pe;
   const double root = std::sqrt(diff);
   const double arg1 = (x - t) / (2.0 * root);
   const double arg2 = (x + t) / (2.0 * root);
   const double gauss = -((x - t) * (x - t)) / (4.0 * diff);

   const double term1 = 0.5 * std::erfc(arg1);
   const double term2 = std::sqrt(t * pe / M_PI) * std::exp(gauss);
   const double term3 = 0.5 * (1.0 + pe * x + pe * t) * ExpTimesErfc(pe * x, arg2);

   const double c = term1 + term2 - term3;
   if (!std::isfinite(c))
   {
      return 0.0;
   }
   return c;
}

class ExactConcentrationCoefficient : public Coefficient
{
public:
   explicit ExactConcentrationCoefficient(double pe) : pe_(pe) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x;
      T.Transform(ip, x);
      return ExactConcentration(x[0], GetTime(), pe_);
   }

private:
   double pe_;
};

void ValidateUnitSquareMesh(const ParMesh &pmesh, const double tol)
{
   double local_min[2] = {numeric_limits<double>::infinity(),
                          numeric_limits<double>::infinity()};
   double local_max[2] = {-numeric_limits<double>::infinity(),
                          -numeric_limits<double>::infinity()};

   for (int i = 0; i < pmesh.GetNV(); i++)
   {
      const double *v = pmesh.GetVertex(i);
      local_min[0] = std::min(local_min[0], v[0]);
      local_min[1] = std::min(local_min[1], v[1]);
      local_max[0] = std::max(local_max[0], v[0]);
      local_max[1] = std::max(local_max[1], v[1]);
   }

   double global_min[2] = {0.0, 0.0};
   double global_max[2] = {0.0, 0.0};
   MPI_Allreduce(local_min, global_min, 2, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(local_max, global_max, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

   MFEM_VERIFY(std::abs(global_min[0] - 0.0) <= tol &&
               std::abs(global_max[0] - 1.0) <= tol &&
               std::abs(global_min[1] - 0.0) <= tol &&
               std::abs(global_max[1] - 1.0) <= tol,
               "Mesh coordinates must span approximately [0,1]x[0,1]. "
               << "Got x=[" << global_min[0] << "," << global_max[0]
               << "], y=[" << global_min[1] << "," << global_max[1] << "].");
}

void BuildXDirichletBoundaryMarker(ParMesh &pmesh, Array<int> &ess_bdr,
                                   const double tol)
{
   const int nbdr = pmesh.bdr_attributes.Max();
   MFEM_VERIFY(nbdr > 0, "Mesh must define boundary attributes.");

   double local_xmin = numeric_limits<double>::infinity();
   double local_xmax = -numeric_limits<double>::infinity();
   for (int i = 0; i < pmesh.GetNV(); i++)
   {
      const double *v = pmesh.GetVertex(i);
      local_xmin = std::min(local_xmin, v[0]);
      local_xmax = std::max(local_xmax, v[0]);
   }
   double xmin = 0.0, xmax = 0.0;
   MPI_Allreduce(&local_xmin, &xmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(&local_xmax, &xmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

   ess_bdr.SetSize(nbdr);
   ess_bdr = 0;

   Vector x;
   for (int i = 0; i < pmesh.GetNBE(); i++)
   {
      const int attr = pmesh.GetBdrAttribute(i);
      ElementTransformation *T = pmesh.GetBdrElementTransformation(i);
      const IntegrationPoint &ip = Geometries.GetCenter(T->GetGeometryType());
      T->Transform(ip, x);

      if (std::abs(x[0] - xmin) <= tol || std::abs(x[0] - xmax) <= tol)
      {
         ess_bdr[attr - 1] = 1;
      }
   }

   Array<int> global_marker(nbdr);
   global_marker = 0;
   MPI_Allreduce(ess_bdr.GetData(), global_marker.GetData(), nbdr, MPI_INT,
                 MPI_MAX, MPI_COMM_WORLD);
   ess_bdr = global_marker;

   int count = 0;
   for (int a = 0; a < nbdr; a++) { count += ess_bdr[a]; }
   MFEM_VERIFY(count > 0, "Failed to identify Dirichlet boundaries at x-extremes.");
}

void PrintConfig(const DriverParams &p)
{
   cout << "Loaded configuration:" << endl;
   cout << "  mesh_file: " << p.mesh_file << endl;
   cout << "  order: " << p.order << endl;
   cout << "  serial_ref_levels: " << p.serial_ref_levels << endl;
   cout << "  par_ref_levels: " << p.par_ref_levels << endl;
   cout << "  dt: " << p.dt << endl;
   cout << "  t_final: " << p.t_final << endl;
   cout << "  peclet: [" << p.peclet[0] << ", "
        << p.peclet[1] << ", " << p.peclet[2] << "]" << endl;
   cout << "  petsc_options_file: " << p.petsc_options_file << endl;
   cout << "  output_path: " << p.output_path << endl;
   cout << "  collection_name: " << p.collection_name << endl;
   cout << "  error_csv: " << p.error_csv << endl;
   cout << "  save_paraview: " << (p.save_paraview ? "true" : "false") << endl;
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const int myid = Mpi::WorldRank();

   string input_file = "Input/input.yaml";
   OptionsParser args(argc, argv);
   args.AddOption(&input_file, "-i", "--input", "YAML input file.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

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
      ValidateUnitSquareMesh(*pmesh, 1.0e-8);

      H1_FECollection fec(params.order, 2);
      ParFiniteElementSpace fespace(pmesh.get(), &fec);
      const HYPRE_BigInt global_true_dofs = fespace.GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "Global true dofs: " << global_true_dofs << endl;
      }

      Array<int> ess_bdr;
      BuildXDirichletBoundaryMarker(*pmesh, ess_bdr, 1.0e-8);
      Array<int> ess_tdof_list;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      if (myid == 0)
      {
         cout << "Dirichlet boundary marker by attribute: [";
         for (int i = 0; i < ess_bdr.Size(); i++)
         {
            cout << ess_bdr[i] << (i + 1 < ess_bdr.Size() ? ", " : "");
         }
         cout << "] (1=Dirichlet on x-boundaries, 0=natural Neumann)" << endl;
      }

      ParBilinearForm mass_form(&fespace);
      mass_form.AddDomainIntegrator(new MassIntegrator());
      mass_form.Assemble();
      mass_form.Finalize();

      Vector beta_vec(2);
      beta_vec = 0.0;
      beta_vec[0] = 1.0;
      VectorConstantCoefficient beta_coeff(beta_vec);

      array<ConstantCoefficient, 3> diffusion_coeff = {
         ConstantCoefficient(params.dt / params.peclet[0]),
         ConstantCoefficient(params.dt / params.peclet[1]),
         ConstantCoefficient(params.dt / params.peclet[2])
      };

      array<unique_ptr<ParBilinearForm>, 3> forms;
      for (int k = 0; k < 3; k++)
      {
         forms[k] = make_unique<ParBilinearForm>(&fespace);
         forms[k]->AddDomainIntegrator(new MassIntegrator());
         forms[k]->AddDomainIntegrator(new ConvectionIntegrator(beta_coeff, params.dt));
         forms[k]->AddDomainIntegrator(new DiffusionIntegrator(diffusion_coeff[k]));
         forms[k]->Assemble();
         forms[k]->Finalize();
      }

      array<unique_ptr<ParGridFunction>, 3> c;
      for (int k = 0; k < 3; k++)
      {
         c[k] = make_unique<ParGridFunction>(&fespace);
         *(c[k]) = 0.0;
      }
      array<unique_ptr<ParGridFunction>, 3> c_exact;
      for (int k = 0; k < 3; k++)
      {
         c_exact[k] = make_unique<ParGridFunction>(&fespace);
         *(c_exact[k]) = 0.0;
      }

      array<ExactConcentrationCoefficient, 3> exact_coeffs = {
         ExactConcentrationCoefficient(params.peclet[0]),
         ExactConcentrationCoefficient(params.peclet[1]),
         ExactConcentrationCoefficient(params.peclet[2])
      };

      const int true_size = fespace.TrueVSize();
      const bool all_essential = (ess_tdof_list.Size() == true_size);

      array<Vector, 3> rhs_local;
      array<Vector, 3> X_sub;
      array<Vector, 3> B_sub;
      array<OperatorHandle, 3> Ah = {
         OperatorHandle(Operator::Hypre_ParCSR),
         OperatorHandle(Operator::Hypre_ParCSR),
         OperatorHandle(Operator::Hypre_ParCSR)
      };

      const int nsteps = static_cast<int>(std::ceil(params.t_final / params.dt - 1.0e-12));
      if (myid == 0)
      {
         cout << "Time steps: " << nsteps
              << ", nominal final time: " << (nsteps * params.dt) << endl;
         if (all_essential)
         {
            cout << "All true dofs are essential; skipping linear solve."
                 << " Use a refined mesh for non-trivial PETSc solves." << endl;
         }
      }

      std::ofstream err_csv;
      if (myid == 0)
      {
         std::error_code ec;
         filesystem::create_directories(params.output_path, ec);
         if (ec)
         {
            throw runtime_error("Failed to create output directory: " + params.output_path +
                                " (" + ec.message() + ")");
         }
         const filesystem::path csv_path =
            filesystem::path(params.output_path) / params.error_csv;
         err_csv.open(csv_path);
         if (!err_csv)
         {
            throw runtime_error("Failed to open error CSV: " + csv_path.string());
         }
         err_csv << "step,time,abs_l2_pe1,rel_l2_pe1,abs_l2_pe2,rel_l2_pe2,"
                 << "abs_l2_pe3,rel_l2_pe3\n";
         err_csv << std::setprecision(16);
      }

      ParaViewDataCollection paraview_dc(params.collection_name.c_str(), pmesh.get());
      if (params.save_paraview)
      {
         paraview_dc.SetPrefixPath(params.output_path.c_str());
         paraview_dc.SetLevelsOfDetail(params.order);
         paraview_dc.SetDataFormat(VTKFormat::BINARY);
         paraview_dc.SetHighOrderOutput(true);
         paraview_dc.RegisterField("c_pe1", c[0].get());
         paraview_dc.RegisterField("c_pe2", c[1].get());
         paraview_dc.RegisterField("c_pe3", c[2].get());
         paraview_dc.RegisterField("c_exact_pe1", c_exact[0].get());
         paraview_dc.RegisterField("c_exact_pe2", c_exact[1].get());
         paraview_dc.RegisterField("c_exact_pe3", c_exact[2].get());
      }

      int order_quad = std::max(2, 2 * params.order + 3);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int g = 0; g < Geometry::NumGeom; g++)
      {
         irs[g] = &IntRules.Get(g, order_quad);
      }

      auto write_errors = [&](int step, double t)
      {
         array<double, 3> abs_l2 = {0.0, 0.0, 0.0};
         array<double, 3> rel_l2 = {0.0, 0.0, 0.0};
         for (int k = 0; k < 3; k++)
         {
            exact_coeffs[k].SetTime(t);
            abs_l2[k] = c[k]->ComputeL2Error(exact_coeffs[k], irs);
            const double norm_l2 =
               ComputeGlobalLpNorm(2, exact_coeffs[k], *pmesh, irs);
            rel_l2[k] = (norm_l2 > 1.0e-14) ? abs_l2[k] / norm_l2 : 0.0;
         }

         if (myid == 0)
         {
            err_csv << step << "," << t << ","
                    << abs_l2[0] << "," << rel_l2[0] << ","
                    << abs_l2[1] << "," << rel_l2[1] << ","
                    << abs_l2[2] << "," << rel_l2[2] << "\n";
            err_csv.flush();

            if (step <= 10 || step == nsteps || step % 50 == 0)
            {
               cout << "step=" << step << " t=" << t
                    << " relL2=["
                    << rel_l2[0] << ", "
                    << rel_l2[1] << ", "
                    << rel_l2[2] << "]" << endl;
            }
         }
      };

      auto save_fields = [&](int step, double t)
      {
         if (!params.save_paraview) { return; }
         for (int k = 0; k < 3; k++)
         {
            exact_coeffs[k].SetTime(t);
            c_exact[k]->ProjectCoefficient(exact_coeffs[k]);
         }
         paraview_dc.SetCycle(step);
         paraview_dc.SetTime(t);
         paraview_dc.Save();
      };

      write_errors(0, 0.0);
      save_fields(0, 0.0);

      for (int step = 1; step <= nsteps; step++)
      {
         const double t = step * params.dt;

         for (int k = 0; k < 3; k++)
         {
            rhs_local[k].SetSize(fespace.GetVSize());
            mass_form.Mult(*(c[k]), rhs_local[k]);
            exact_coeffs[k].SetTime(t);
            c[k]->ProjectBdrCoefficient(exact_coeffs[k], ess_bdr);
            forms[k]->FormLinearSystem(ess_tdof_list, *(c[k]), rhs_local[k],
                                       Ah[k], X_sub[k], B_sub[k]);
         }

         if (!all_essential)
         {
            for (int k = 0; k < 3; k++)
            {
               HypreParMatrix *Ak = Ah[k].As<HypreParMatrix>();
               MFEM_VERIFY(Ak != nullptr, "Expected HypreParMatrix in block " << k);
               PetscParMatrix A_petsc(Ak, Operator::PETSC_MATAIJ);
               PetscLinearSolver solver(A_petsc);
               solver.SetPrintLevel(0);
               solver.Mult(B_sub[k], X_sub[k]);
               MFEM_VERIFY(solver.GetConverged(),
                           "PETSc solver did not converge at step " << step
                           << ", block " << k
                           << ". Iterations=" << solver.GetNumIterations()
                           << ", residual=" << solver.GetFinalNorm());
            }
         }

         for (int k = 0; k < 3; k++)
         {
            forms[k]->RecoverFEMSolution(X_sub[k], rhs_local[k], *(c[k]));
         }

         write_errors(step, t);
         save_fields(step, t);
      }
   }
   catch (const exception &e)
   {
      if (myid == 0)
      {
         cerr << "Error: " << e.what() << endl;
      }
      exit_code = 3;
   }

   MFEMFinalizePetsc();
   return exit_code;
}

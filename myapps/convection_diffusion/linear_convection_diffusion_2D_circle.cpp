// 2D steady convection-diffusion-reaction driver on a unit circular domain.
//
// Solves equation (7) from 16_930-2.pdf:
//   -kappa * Delta(u) + div(c * u) + s * u = f  in Omega
//   u = 0                                      on Gamma
//
// Uses Table 3 defaults:
//   kappa = 1, s = 1, c = (1, 1)
//
// Uses equation (13) manufactured exact solution:
//   u(r) = (r^2 - 1) * cos(2*pi*r),  r = sqrt(x^2 + y^2)
//
// Forcing is computed analytically from Eq. (7):
//   f = -kappa * Delta(u_exact) + c·grad(u_exact) + s*u_exact.

#include "mfem.hpp"

#include <yaml-cpp/yaml.h>

#ifndef MFEM_USE_PETSC
#error "This driver requires MFEM built with PETSc."
#endif
#include <petscksp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
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

   double kappa = 1.0;
   double reaction = 1.0;
   array<double, 2> convection = {1.0, 1.0};

   string petsc_options_file = "Input/petsc_circle.opts";
   string output_path = "ParaView";
   string collection_name = "convection_diffusion_2D_circle";
   string error_csv = "error_history_2D_circle.csv";
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
   if (n["kappa"]) { p.kappa = n["kappa"].as<double>(); }
   if (n["s"]) { p.reaction = n["s"].as<double>(); }
   if (n["petsc_options_file"]) { p.petsc_options_file = n["petsc_options_file"].as<string>(); }
   if (n["output_path"]) { p.output_path = n["output_path"].as<string>(); }
   if (n["collection_name"]) { p.collection_name = n["collection_name"].as<string>(); }
   if (n["error_csv"]) { p.error_csv = n["error_csv"].as<string>(); }
   if (n["save_paraview"]) { p.save_paraview = n["save_paraview"].as<bool>(); }

   if (n["cx"]) { p.convection[0] = n["cx"].as<double>(); }
   if (n["cy"]) { p.convection[1] = n["cy"].as<double>(); }
   if (n["convection"])
   {
      const YAML::Node c = n["convection"];
      if (!c.IsSequence() || c.size() != 2)
      {
         throw runtime_error("YAML key convection must be a sequence of exactly 2 values.");
      }
      p.convection[0] = c[0].as<double>();
      p.convection[1] = c[1].as<double>();
   }

   if (p.order < 1)
   {
      throw runtime_error("order must be >= 1.");
   }
   if (p.serial_ref_levels < 0 || p.par_ref_levels < 0)
   {
      throw runtime_error("serial_ref_levels and par_ref_levels must be >= 0.");
   }
   if (p.kappa <= 0.0)
   {
      throw runtime_error("kappa must be > 0.");
   }
}

void ValidateUnitCircleMesh(const ParMesh &pmesh, const double tol)
{
   double local_rmax = 0.0;
   for (int i = 0; i < pmesh.GetNV(); i++)
   {
      const double *v = pmesh.GetVertex(i);
      const double r = std::sqrt(v[0] * v[0] + v[1] * v[1]);
      local_rmax = std::max(local_rmax, r);
   }

   double global_rmax = 0.0;
   MPI_Allreduce(&local_rmax, &global_rmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

   MFEM_VERIFY(std::abs(global_rmax - 1.0) <= tol,
               "Expected unit-circle mesh (max radius near 1). Found max radius "
               << global_rmax << ".");
}

constexpr double kAlpha = 2.0 * M_PI;
constexpr double kSmallR = 1.0e-12;

double ExactU(const double r)
{
   return (r * r - 1.0) * std::cos(kAlpha * r);
}

double ExactU_r(const double r)
{
   return 2.0 * r * std::cos(kAlpha * r)
          - kAlpha * (r * r - 1.0) * std::sin(kAlpha * r);
}

double ExactU_rr(const double r)
{
   return 2.0 * std::cos(kAlpha * r)
          - 4.0 * kAlpha * r * std::sin(kAlpha * r)
          - kAlpha * kAlpha * (r * r - 1.0) * std::cos(kAlpha * r);
}

double ExactLaplacian(const double r)
{
   if (r > kSmallR)
   {
      return ExactU_rr(r) + ExactU_r(r) / r;
   }

   // For smooth radial u in 2D: lim_{r->0}(u_rr + u_r/r) = 2*u_rr(0).
   return 2.0 * (2.0 + kAlpha * kAlpha);
}

class ExactSolutionCoefficient : public Coefficient
{
public:
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x;
      T.Transform(ip, x);
      const double r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
      return ExactU(r);
   }
};

class ForcingCoefficient : public Coefficient
{
public:
   ForcingCoefficient(double kappa, double reaction,
                      double cx, double cy)
      : kappa_(kappa), reaction_(reaction), cx_(cx), cy_(cy) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x;
      T.Transform(ip, x);

      const double x0 = x[0];
      const double x1 = x[1];
      const double r = std::sqrt(x0 * x0 + x1 * x1);

      const double u = ExactU(r);
      const double lap_u = ExactLaplacian(r);

      double ux = 0.0;
      double uy = 0.0;
      if (r > kSmallR)
      {
         const double radial_scale = ExactU_r(r) / r;
         ux = radial_scale * x0;
         uy = radial_scale * x1;
      }

      const double convection_term = cx_ * ux + cy_ * uy;
      return -kappa_ * lap_u + convection_term + reaction_ * u;
   }

private:
   double kappa_;
   double reaction_;
   double cx_;
   double cy_;
};

void PrintConfig(const DriverParams &p)
{
   cout << "Loaded configuration:" << endl;
   cout << "  mesh_file: " << p.mesh_file << endl;
   cout << "  order: " << p.order << endl;
   cout << "  serial_ref_levels: " << p.serial_ref_levels << endl;
   cout << "  par_ref_levels: " << p.par_ref_levels << endl;
   cout << "  kappa: " << p.kappa << endl;
   cout << "  s: " << p.reaction << endl;
   cout << "  c: [" << p.convection[0] << ", " << p.convection[1] << "]" << endl;
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

   string input_file = "Input/input_2d_circle.yaml";
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
      ValidateUnitCircleMesh(*pmesh, 1.0e-6);

      H1_FECollection fec(params.order, 2);
      ParFiniteElementSpace fespace(pmesh.get(), &fec);
      const HYPRE_BigInt global_true_dofs = fespace.GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "Global true dofs: " << global_true_dofs << endl;
      }

      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      Array<int> ess_tdof_list;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      ExactSolutionCoefficient exact_coeff;
      ForcingCoefficient forcing_coeff(params.kappa, params.reaction,
                                       params.convection[0], params.convection[1]);

      Vector c_vec(2);
      c_vec[0] = params.convection[0];
      c_vec[1] = params.convection[1];
      VectorConstantCoefficient convection_coeff(c_vec);
      ConstantCoefficient kappa_coeff(params.kappa);
      ConstantCoefficient reaction_coeff(params.reaction);

      ParBilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator(kappa_coeff));
      a.AddDomainIntegrator(new ConvectionIntegrator(convection_coeff));
      a.AddDomainIntegrator(new MassIntegrator(reaction_coeff));
      a.Assemble();

      ParLinearForm b(&fespace);
      b.AddDomainIntegrator(new DomainLFIntegrator(forcing_coeff));
      b.Assemble();

      ParGridFunction u(&fespace);
      u = 0.0;
      u.ProjectBdrCoefficient(exact_coeff, ess_bdr);

      OperatorHandle Ah(Operator::Hypre_ParCSR);
      Vector X, B;
      a.FormLinearSystem(ess_tdof_list, u, b, Ah, X, B);

      const int true_size = fespace.TrueVSize();
      const bool all_essential = (ess_tdof_list.Size() == true_size);
      if (all_essential)
      {
         if (myid == 0)
         {
            cout << "All true dofs are essential; skipping linear solve." << endl;
         }
      }
      else
      {
         HypreParMatrix *A_true = Ah.As<HypreParMatrix>();
         MFEM_VERIFY(A_true != nullptr, "Expected HypreParMatrix from FormLinearSystem.");

         PetscParMatrix A_petsc(MPI_COMM_WORLD, A_true, Operator::PETSC_MATAIJ);
         PetscLinearSolver solver(A_petsc);
         solver.SetPrintLevel(0);
         solver.Mult(B, X);
         MFEM_VERIFY(solver.GetConverged(),
                     "PETSc solver did not converge. Iterations="
                     << solver.GetNumIterations()
                     << ", residual=" << solver.GetFinalNorm());
      }

      a.RecoverFEMSolution(X, b, u);

      ParGridFunction u_exact(&fespace);
      u_exact = 0.0;
      u_exact.ProjectCoefficient(exact_coeff);

      int order_quad = std::max(2, 2 * params.order + 3);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int g = 0; g < Geometry::NumGeom; g++)
      {
         irs[g] = &IntRules.Get(g, order_quad);
      }

      const double abs_l2 = u.ComputeL2Error(exact_coeff, irs);
      const double exact_l2 = ComputeGlobalLpNorm(2, exact_coeff, *pmesh, irs);
      const double rel_l2 = (exact_l2 > 1.0e-14) ? abs_l2 / exact_l2 : 0.0;

      if (myid == 0)
      {
         cout << "L2 error (absolute): " << abs_l2 << endl;
         cout << "L2 error (relative): " << rel_l2 << endl;
      }

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
         ofstream err_csv(csv_path);
         if (!err_csv)
         {
            throw runtime_error("Failed to open error CSV: " + csv_path.string());
         }
         err_csv << "abs_l2,rel_l2\n";
         err_csv << std::setprecision(16) << abs_l2 << "," << rel_l2 << "\n";
      }

      if (params.save_paraview)
      {
         ParaViewDataCollection paraview_dc(params.collection_name.c_str(), pmesh.get());
         paraview_dc.SetPrefixPath(params.output_path.c_str());
         paraview_dc.SetLevelsOfDetail(params.order);
         paraview_dc.SetDataFormat(VTKFormat::BINARY);
         paraview_dc.SetHighOrderOutput(true);
         paraview_dc.RegisterField("u", &u);
         paraview_dc.RegisterField("u_exact", &u_exact);
         paraview_dc.SetCycle(0);
         paraview_dc.SetTime(0.0);
         paraview_dc.Save();
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

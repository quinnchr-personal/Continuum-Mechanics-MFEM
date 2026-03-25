// ALE backward Euler driver for a finite-length Stefan/ablation verification.
//
// This solves the 1D conduction-limited Stefan problem on a finite domain,
// represented as a 2D strip with fields uniform in y:
//   rho*c*dT/dt - k*Delta(T) = 0   on Omega(t) = [x_L, s(t)] x [0,1]
//
// The ALE weak form is assembled on a fixed reference mesh [0,1]^2 using the
// same Eq. 5.7 pattern as diffusion_mms_ale.cpp. The physical domain motion is
// prescribed from the exact similarity solution:
//   s(t) = -2*lambda*sqrt(alpha*t),  alpha = k/(rho*c)
//
// Note on the exact solution:
//   stefan_problem_ablating_material.txt writes a temperature formula that does
//   not satisfy T(s(t), t) = T_a. This driver intentionally uses the corrected
//   similarity form:
//     T(x,t) = T_0 + (T_a - T_0) * erfc(-x/(2*sqrt(alpha*t))) / erfc(lambda)
// for t > 0, and the uniform initial state T_0 at t = 0.

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
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

using namespace std;
using namespace mfem;

namespace
{

constexpr double kPi = 3.141592653589793238462643383279502884;

struct DriverParams
{
   string mesh_file;
   int    order             = 1;
   int    serial_ref_levels = 0;
   int    par_ref_levels    = 0;

   double rho          = 1.0;
   double c            = 1.0;
   double k            = 0.1;
   double L_star       = 10.0;
   double T_0          = 300.0;
   double T_a          = 800.0;
   double domain_length = 1.0;
   double dt           = 0.01;
   double t_final      = 0.25;

   string petsc_options_file = "Input/petsc.opts";
   int    linear_max_iter    = 400;
   double linear_rel_tol     = 1.0e-10;
   double linear_abs_tol     = 0.0;

   string output_path    = "ParaView/stefan_problem_ablating_material_ale";
   bool   save_paraview  = true;
   int    paraview_every = 1;
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

   if (n["order"])              { p.order = n["order"].as<int>(); }
   if (n["serial_ref_levels"])  { p.serial_ref_levels = n["serial_ref_levels"].as<int>(); }
   if (n["par_ref_levels"])     { p.par_ref_levels    = n["par_ref_levels"].as<int>(); }
   if (n["rho"])                { p.rho = n["rho"].as<double>(); }
   if (n["c"])                  { p.c = n["c"].as<double>(); }
   if (n["k"])                  { p.k = n["k"].as<double>(); }
   if (n["L_star"])             { p.L_star = n["L_star"].as<double>(); }
   if (n["T_0"])                { p.T_0 = n["T_0"].as<double>(); }
   if (n["T_a"])                { p.T_a = n["T_a"].as<double>(); }
   if (n["domain_length"])      { p.domain_length = n["domain_length"].as<double>(); }
   if (n["dt"])                 { p.dt = n["dt"].as<double>(); }
   if (n["t_final"])            { p.t_final = n["t_final"].as<double>(); }
   if (n["petsc_options_file"]) { p.petsc_options_file = n["petsc_options_file"].as<string>(); }
   if (n["linear_max_iter"])    { p.linear_max_iter = n["linear_max_iter"].as<int>(); }
   if (n["linear_rel_tol"])     { p.linear_rel_tol = n["linear_rel_tol"].as<double>(); }
   if (n["linear_abs_tol"])     { p.linear_abs_tol = n["linear_abs_tol"].as<double>(); }
   if (n["output_path"])        { p.output_path = n["output_path"].as<string>(); }
   if (n["save_paraview"])      { p.save_paraview = n["save_paraview"].as<bool>(); }
   if (n["paraview_every"])     { p.paraview_every = n["paraview_every"].as<int>(); }

   if (p.order < 1)
   {
      throw runtime_error("order must be >= 1.");
   }
   if (p.serial_ref_levels < 0 || p.par_ref_levels < 0)
   {
      throw runtime_error("Refinement levels must be >= 0.");
   }
   if (p.rho <= 0.0 || p.c <= 0.0 || p.k <= 0.0 || p.L_star <= 0.0)
   {
      throw runtime_error("rho, c, k, and L_star must all be > 0.");
   }
   if (p.T_a <= p.T_0)
   {
      throw runtime_error("Expected T_a > T_0 for this Stefan verification case.");
   }
   if (p.domain_length <= 0.0)
   {
      throw runtime_error("domain_length must be > 0.");
   }
   if (p.dt <= 0.0)
   {
      throw runtime_error("dt must be > 0.");
   }
   if (p.t_final < 0.0)
   {
      throw runtime_error("t_final must be >= 0.");
   }
   if (p.linear_max_iter < 1)
   {
      throw runtime_error("linear_max_iter must be >= 1.");
   }
   if (p.linear_rel_tol <= 0.0 || p.linear_abs_tol < 0.0)
   {
      throw runtime_error("Invalid linear solver tolerances.");
   }
   if (p.paraview_every < 0)
   {
      throw runtime_error("paraview_every must be >= 0.");
   }
}

void PrintConfig(const DriverParams &p)
{
   cout << "Finite-length Stefan ALE driver (Backward Euler, reference configuration)"
        << endl;
   cout << "  mesh_file:          " << p.mesh_file << endl;
   cout << "  order:              " << p.order << endl;
   cout << "  serial_ref_levels:  " << p.serial_ref_levels << endl;
   cout << "  par_ref_levels:     " << p.par_ref_levels << endl;
   cout << "  rho:                " << p.rho << endl;
   cout << "  c:                  " << p.c << endl;
   cout << "  k:                  " << p.k << endl;
   cout << "  L_star:             " << p.L_star << endl;
   cout << "  T_0:                " << p.T_0 << endl;
   cout << "  T_a:                " << p.T_a << endl;
   cout << "  domain_length:      " << p.domain_length << endl;
   cout << "  dt:                 " << p.dt << endl;
   cout << "  t_final:            " << p.t_final << endl;
   cout << "  petsc_options_file: " << p.petsc_options_file << endl;
   cout << "  output_path:        " << p.output_path << endl;
   cout << "  save_paraview:      " << (p.save_paraview ? "true" : "false") << endl;
   cout << "  paraview_every:     " << p.paraview_every
        << " (0=initial/final only)" << endl;
}

double StefanLambdaResidual(const double lambda, const double ste)
{
   return std::sqrt(kPi) * lambda * std::exp(lambda * lambda) * std::erf(lambda)
          - ste;
}

double SolveSimilarityLambda(const double ste)
{
   if (!(ste > 0.0))
   {
      throw runtime_error("Stefan number must be > 0.");
   }

   double lo = 0.0;
   double hi = 1.0;
   double f_lo = StefanLambdaResidual(lo, ste);
   double f_hi = StefanLambdaResidual(hi, ste);

   while (f_hi <= 0.0)
   {
      hi *= 2.0;
      if (hi > 10.0)
      {
         throw runtime_error("Failed to bracket the positive root for lambda.");
      }
      f_hi = StefanLambdaResidual(hi, ste);
   }

   for (int iter = 0; iter < 200; iter++)
   {
      const double mid = 0.5 * (lo + hi);
      const double f_mid = StefanLambdaResidual(mid, ste);
      if (std::abs(f_mid) <= 1.0e-12 || (hi - lo) <= 1.0e-12 * (1.0 + mid))
      {
         return mid;
      }
      if (f_mid > 0.0)
      {
         hi = mid;
         f_hi = f_mid;
      }
      else
      {
         lo = mid;
         f_lo = f_mid;
      }
      (void)f_hi;
      (void)f_lo;
   }

   return 0.5 * (lo + hi);
}

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

class StefanAleMap
{
public:
   StefanAleMap(double x_left, double alpha, double lambda)
      : x_left_(x_left), alpha_(alpha), lambda_(lambda) {}

   double Alpha() const { return alpha_; }
   double Lambda() const { return lambda_; }
   double LeftBoundary() const { return x_left_; }

   double SurfacePosition(const double t) const
   {
      if (t <= 0.0) { return 0.0; }
      return -2.0 * lambda_ * std::sqrt(alpha_ * t);
   }

   double DomainWidth(const double t) const
   {
      return SurfacePosition(t) - x_left_;
   }

   void MapPoint(const Vector &xhat, const double t, Vector &x) const
   {
      MFEM_VERIFY(xhat.Size() == 2, "StefanAleMap expects 2D reference points.");
      x.SetSize(2);
      const double width = DomainWidth(t);
      x[0] = x_left_ + width * xhat[0];
      x[1] = xhat[1];
   }

   void MapGradient(const Vector &xhat, const double t, DenseMatrix &G) const
   {
      (void)xhat;
      const double width = DomainWidth(t);
      G.SetSize(2, 2);
      G = 0.0;
      G(0, 0) = width;
      G(1, 1) = 1.0;
   }

   void MapCofactor(const Vector &xhat, const double t, DenseMatrix &C) const
   {
      (void)xhat;
      const double width = DomainWidth(t);
      C.SetSize(2, 2);
      C = 0.0;
      C(0, 0) = 1.0;
      C(1, 1) = width;
   }

   double JacobianDet(const Vector &xhat, const double t) const
   {
      (void)xhat;
      return DomainWidth(t);
   }

   void IntegratedMappedGridFlux(const Vector &xhat,
                                 const double t0,
                                 const double t1,
                                 Vector &flux_hat,
                                 double &div_flux_hat) const
   {
      MFEM_VERIFY(xhat.Size() == 2, "StefanAleMap expects 2D reference points.");
      const double ds = SurfacePosition(t1) - SurfacePosition(t0);
      flux_hat.SetSize(2);
      flux_hat[0] = xhat[0] * ds;
      flux_hat[1] = 0.0;
      div_flux_hat = ds;
   }

private:
   double x_left_;
   double alpha_;
   double lambda_;
};

class AleJacobianCoefficient : public Coefficient
{
public:
   explicit AleJacobianCoefficient(const StefanAleMap &map) : map_(map) {}

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      T.Transform(ip, xhat_);
      return map_.JacobianDet(xhat_, GetTime());
   }

private:
   const StefanAleMap &map_;
   mutable Vector xhat_;
};

class AleMetricTensorCoefficient : public MatrixCoefficient
{
public:
   AleMetricTensorCoefficient(const StefanAleMap &map, double alpha, double dt)
      : MatrixCoefficient(2), map_(map), alpha_(alpha), dt_(dt) {}

   void Eval(DenseMatrix &M,
             ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      T.Transform(ip, xhat_);
      const double t = GetTime();
      DenseMatrix C(2, 2);
      map_.MapCofactor(xhat_, t, C);
      const double J = map_.JacobianDet(xhat_, t);
      MFEM_VERIFY(std::abs(J) > 1.0e-14,
                  "Degenerate ALE Jacobian at t=" << t);
      M.SetSize(2, 2);
      MultAAt(C, M);
      M *= (alpha_ * dt_ / J);
   }

private:
   const StefanAleMap &map_;
   double alpha_;
   double dt_;
   mutable Vector xhat_;
};

class AleIntegratedFluxConvCoefficient : public VectorCoefficient
{
public:
   AleIntegratedFluxConvCoefficient(const StefanAleMap &map,
                                    double t_old,
                                    double t_new)
      : VectorCoefficient(2), map_(map), t_old_(t_old), t_new_(t_new) {}

   void SetTimes(double t_old, double t_new)
   {
      t_old_ = t_old;
      t_new_ = t_new;
   }

   void Eval(Vector &V,
             ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      T.Transform(ip, xhat_);
      double div_dummy = 0.0;
      map_.IntegratedMappedGridFlux(xhat_, t_old_, t_new_, V, div_dummy);
   }

private:
   const StefanAleMap &map_;
   double t_old_;
   double t_new_;
   mutable Vector xhat_;
};

class AleIntegratedFluxDivCoefficient : public Coefficient
{
public:
   AleIntegratedFluxDivCoefficient(const StefanAleMap &map,
                                   double t_old,
                                   double t_new)
      : map_(map), t_old_(t_old), t_new_(t_new) {}

   void SetTimes(double t_old, double t_new)
   {
      t_old_ = t_old;
      t_new_ = t_new;
   }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      T.Transform(ip, xhat_);
      Vector flux_dummy(2);
      double div_flux = 0.0;
      map_.IntegratedMappedGridFlux(xhat_, t_old_, t_new_, flux_dummy, div_flux);
      return div_flux;
   }

private:
   const StefanAleMap &map_;
   double t_old_;
   double t_new_;
   mutable Vector xhat_;
};

class StefanExactCoefficient : public Coefficient
{
public:
   StefanExactCoefficient(const StefanAleMap &map,
                          const double T_0,
                          const double T_a)
      : map_(map), T_0_(T_0), T_a_(T_a),
        erfc_lambda_(std::erfc(map.Lambda())) {}

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      T.Transform(ip, xhat_);
      const double t = GetTime();
      if (t <= 0.0)
      {
         return T_0_;
      }

      map_.MapPoint(xhat_, t, x_phys_);
      const double denom = 2.0 * std::sqrt(map_.Alpha() * t);
      const double arg = -x_phys_[0] / denom;
      return T_0_ + (T_a_ - T_0_) * std::erfc(arg) / erfc_lambda_;
   }

private:
   const StefanAleMap &map_;
   double T_0_;
   double T_a_;
   double erfc_lambda_;
   mutable Vector xhat_;
   mutable Vector x_phys_;
};

class AleDisplacementCoefficient : public VectorCoefficient
{
public:
   explicit AleDisplacementCoefficient(const StefanAleMap &map)
      : VectorCoefficient(2), map_(map) {}

   void Eval(Vector &V,
             ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      T.Transform(ip, xhat_);
      map_.MapPoint(xhat_, GetTime(), x_phys_);
      V.SetSize(2);
      V[0] = x_phys_[0] - xhat_[0];
      V[1] = x_phys_[1] - xhat_[1];
   }

private:
   const StefanAleMap &map_;
   mutable Vector xhat_;
   mutable Vector x_phys_;
};

class ScaledCoefficient : public Coefficient
{
public:
   ScaledCoefficient(double scale, Coefficient &base)
      : scale_(scale), base_(base) {}

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      return scale_ * base_.Eval(T, ip);
   }

private:
   double scale_;
   Coefficient &base_;
};

void SolveLinearSystem(ParBilinearForm &a,
                       Array<int> &ess_tdof_list,
                       ParGridFunction &u,
                       Vector &rhs,
                       const DriverParams &params,
                       const int myid,
                       const int step)
{
   OperatorHandle Ah(Operator::Hypre_ParCSR);
   Vector X, B;
   a.FormLinearSystem(ess_tdof_list, u, rhs, Ah, X, B);

   HypreParMatrix *A_hyp = Ah.As<HypreParMatrix>();
   MFEM_VERIFY(A_hyp != nullptr, "Expected HypreParMatrix.");

   PetscParMatrix A_petsc(MPI_COMM_WORLD, A_hyp, Operator::PETSC_MATAIJ);
   PetscLinearSolver solver(A_petsc);
   solver.SetRelTol(params.linear_rel_tol);
   solver.SetAbsTol(params.linear_abs_tol);
   solver.SetMaxIter(params.linear_max_iter);
   solver.SetPrintLevel(0);

   KSP ksp = (KSP)solver;
   KSPSetType(ksp, KSPGMRES);
   KSPGMRESSetRestart(ksp, std::min(400, std::max(50, params.linear_max_iter)));
   PC pc = nullptr;
   KSPGetPC(ksp, &pc);
   if (Mpi::WorldSize() == 1)
   {
      PCSetType(pc, PCLU);
   }
   else
   {
      PCSetType(pc, PCBJACOBI);
   }

   X = 0.0;
   solver.Mult(B, X);

   const double rhs_norm = B.Norml2();
   const double final_norm = solver.GetFinalNorm();
   const double effective_tol =
      std::max(params.linear_abs_tol,
               params.linear_rel_tol * std::max(1.0, rhs_norm));

   if (!solver.GetConverged() && !(final_norm <= effective_tol))
   {
      if (myid == 0)
      {
         throw runtime_error(
            "PETSc solver failed at step " + to_string(step)
            + ": iters=" + to_string(solver.GetNumIterations())
            + " residual=" + to_string(final_norm)
            + " tol=" + to_string(effective_tol));
      }
      throw runtime_error("PETSc solver failed on non-root rank.");
   }

   a.RecoverFEMSolution(X, rhs, u);
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const int myid = Mpi::WorldRank();

   string input_file = "Input/input_stefan_problem_ablating_material_ale.yaml";
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

   const char *petsc_file = nullptr;
   if (!params.petsc_options_file.empty())
   {
      ifstream pf(params.petsc_options_file);
      if (pf.good())
      {
         petsc_file = params.petsc_options_file.c_str();
      }
      else if (myid == 0)
      {
         cerr << "PETSc options file not found: " << params.petsc_options_file
              << ". Proceeding without it." << endl;
      }
   }
   MFEMInitializePetsc(&argc, &argv, petsc_file, nullptr);

   int exit_code = 0;
   try
   {
      Device device("cpu");
      if (myid == 0) { device.Print(); }

      const double alpha = params.k / (params.rho * params.c);
      const double ste = params.c * (params.T_a - params.T_0) / params.L_star;
      const double lambda = SolveSimilarityLambda(ste);
      const int nsteps = static_cast<int>(
         std::ceil(params.t_final / params.dt - 1.0e-12));
      const double run_t_final = nsteps * params.dt;

      StefanAleMap ale_map(-params.domain_length, alpha, lambda);
      const double width_final = ale_map.DomainWidth(run_t_final);
      if (width_final <= 0.0)
      {
         throw runtime_error(
            "Domain width becomes non-positive by the requested final time. "
            "Increase domain_length or reduce t_final.");
      }

      if (myid == 0)
      {
         cout << "Derived parameters:" << endl;
         cout << "  alpha:              " << alpha << endl;
         cout << "  Ste:                " << ste << endl;
         cout << "  lambda:             " << lambda << endl;
         cout << "  s(0):               " << ale_map.SurfacePosition(0.0) << endl;
         cout << "  s(t_final_run):     " << ale_map.SurfacePosition(run_t_final) << endl;
         cout << "  width(0):           " << ale_map.DomainWidth(0.0) << endl;
         cout << "  width(t_final_run): " << width_final << endl;
         cout << "  time steps:         " << nsteps << endl;
         cout << "  final run time:     " << run_t_final << endl;
         cout << "  lambda residual:    " << StefanLambdaResidual(lambda, ste) << endl;
      }

      unique_ptr<Mesh> mesh = make_unique<Mesh>(params.mesh_file.c_str(), 1, 1);
      if (mesh->Dimension() != 2)
      {
         throw runtime_error("Mesh must be 2D.");
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

      H1_FECollection fec(params.order, pmesh->Dimension());
      ParFiniteElementSpace fes(pmesh.get(), &fec);
      ParFiniteElementSpace vec_fes(pmesh.get(), &fec, pmesh->Dimension());

      if (myid == 0)
      {
         cout << "Scalar true dofs: " << fes.GlobalTrueVSize() << endl;
      }

      Array<int> ess_bdr;
      BuildXDirichletBoundaryMarker(*pmesh, ess_bdr, 1.0e-8);
      Array<int> ess_tdof_list;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      if (myid == 0)
      {
         cout << "Dirichlet boundary marker by attribute: [";
         for (int i = 0; i < ess_bdr.Size(); i++)
         {
            cout << ess_bdr[i] << (i + 1 < ess_bdr.Size() ? ", " : "");
         }
         cout << "] (1=Dirichlet on x-boundaries, 0=natural Neumann)" << endl;
      }

      const int order_quad = max(4, 2 * params.order + 6);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int g = 0; g < Geometry::NumGeom; g++)
      {
         irs[g] = &IntRules.Get(g, order_quad);
      }

      AleJacobianCoefficient           j_coeff(ale_map);
      AleMetricTensorCoefficient       metric_coeff(ale_map, alpha, params.dt);
      AleIntegratedFluxConvCoefficient flux_conv_coeff(ale_map, 0.0, 0.0);
      AleIntegratedFluxDivCoefficient  flux_div_coeff(ale_map, 0.0, 0.0);
      ScaledCoefficient                neg_flux_div_coeff(-1.0, flux_div_coeff);
      StefanExactCoefficient           exact_coeff(ale_map, params.T_0, params.T_a);
      AleDisplacementCoefficient       disp_coeff(ale_map);

      ParGridFunction temperature(&fes);
      ParGridFunction temperature_exact(&fes);
      ParGridFunction temperature_error(&fes);
      ParGridFunction ale_jacobian(&fes);
      ParGridFunction ale_displacement(&vec_fes);

      ConstantCoefficient initial_coeff(params.T_0);
      temperature.ProjectCoefficient(initial_coeff);
      temperature_exact.ProjectCoefficient(initial_coeff);
      temperature_error = 0.0;

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

         paraview_dc = make_unique<ParaViewDataCollection>(
            "stefan_problem_ablating_material_ale", pmesh.get());
         paraview_dc->SetPrefixPath(params.output_path);
         paraview_dc->SetLevelsOfDetail(params.order);
         paraview_dc->SetDataFormat(VTKFormat::BINARY);
         paraview_dc->SetHighOrderOutput(true);
         paraview_dc->RegisterField("temperature", &temperature);
         paraview_dc->RegisterField("temperature_exact", &temperature_exact);
         paraview_dc->RegisterField("temperature_error", &temperature_error);
         paraview_dc->RegisterField("ale_displacement", &ale_displacement);
         paraview_dc->RegisterField("ale_jacobian", &ale_jacobian);
      }

      ofstream err_csv;
      if (myid == 0)
      {
         filesystem::create_directories(params.output_path);
         const filesystem::path csv_path =
            filesystem::path(params.output_path) / "error_history.csv";
         err_csv.open(csv_path);
         if (!err_csv)
         {
            throw runtime_error("Failed to open error CSV: " + csv_path.string());
         }
         err_csv << "step,time,surface_position,domain_width,l2_error,linf_error\n"
                 << setprecision(16);
      }

      auto compute_and_save = [&](int step, double t, bool force_save)
      {
         exact_coeff.SetTime(t);
         disp_coeff.SetTime(t);
         j_coeff.SetTime(t);

         const double l2_err =
            temperature.ComputeLpError(2.0, exact_coeff, &j_coeff, irs);

         temperature_exact.ProjectCoefficient(exact_coeff);
         subtract(temperature, temperature_exact, temperature_error);
         ale_displacement.ProjectCoefficient(disp_coeff);
         ale_jacobian.ProjectCoefficient(j_coeff);

         const double local_linf = temperature_error.Normlinf();
         double linf_err = 0.0;
         MPI_Allreduce(&local_linf, &linf_err, 1, MPI_DOUBLE, MPI_MAX,
                       MPI_COMM_WORLD);

         const double s = ale_map.SurfacePosition(t);
         const double width = ale_map.DomainWidth(t);

         if (myid == 0)
         {
            err_csv << step << "," << t << "," << s << "," << width << ","
                    << l2_err << "," << linf_err << "\n";
            err_csv.flush();

            if (step == 0 || step <= 3 || step % 20 == 0 || step == nsteps)
            {
               cout << "step=" << step
                    << "  t=" << fixed << setprecision(6) << t
                    << "  s=" << s
                    << "  width=" << width
                    << "  L2_error=" << scientific << setprecision(6) << l2_err
                    << "  Linf_error=" << linf_err
                    << defaultfloat << endl;
            }
         }

         if (paraview_dc)
         {
            const bool save_this =
               force_save
               || (params.paraview_every > 0
                   && (step % params.paraview_every == 0));
            if (save_this)
            {
               paraview_dc->SetCycle(step);
               paraview_dc->SetTime(t);
               paraview_dc->Save();
            }
         }
      };

      compute_and_save(0, 0.0, /*force_save=*/true);

      double t = 0.0;
      for (int step = 1; step <= nsteps; step++)
      {
         const double t_old = t;
         const double t_new = t + params.dt;

         j_coeff.SetTime(t_old);
         ParBilinearForm m_old(&fes);
         m_old.AddDomainIntegrator(new MassIntegrator(j_coeff));
         m_old.Assemble();
         m_old.Finalize();

         Vector rhs(fes.GetVSize());
         m_old.Mult(temperature, rhs);

         t = t_new;
         j_coeff.SetTime(t_new);
         metric_coeff.SetTime(t_new);
         flux_conv_coeff.SetTimes(t_old, t_new);
         flux_div_coeff.SetTimes(t_old, t_new);
         exact_coeff.SetTime(t_new);

         ParBilinearForm a(&fes);
         a.AddDomainIntegrator(new MassIntegrator(j_coeff));
         a.AddDomainIntegrator(new DiffusionIntegrator(metric_coeff));
         a.AddDomainIntegrator(new ConvectionIntegrator(flux_conv_coeff, -1.0));
         a.AddDomainIntegrator(new MassIntegrator(neg_flux_div_coeff));
         a.Assemble();
         a.Finalize();

         temperature.ProjectBdrCoefficient(exact_coeff, ess_bdr);
         SolveLinearSystem(a, ess_tdof_list, temperature, rhs, params, myid, step);

         const bool is_final = (step == nsteps);
         compute_and_save(step, t, is_final);
      }

      if (myid == 0)
      {
         exact_coeff.SetTime(t);
         j_coeff.SetTime(t);
         const double final_l2 =
            temperature.ComputeLpError(2.0, exact_coeff, &j_coeff, irs);
         cout << "\nFinal L2 error at t=" << t << ": " << final_l2 << endl;
         cout << "Output written to:   " << params.output_path << endl;
         if (params.save_paraview)
         {
            cout << "\nParaView tip:" << endl;
            cout << "  1. Open: " << params.output_path
                 << "/stefan_problem_ablating_material_ale.pvd" << endl;
            cout << "  2. Filters -> Warp By Vector -> select 'ale_displacement'"
                 << endl;
            cout << "  3. Color by 'temperature', 'temperature_exact', or "
                 << "'temperature_error'" << endl;
         }
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

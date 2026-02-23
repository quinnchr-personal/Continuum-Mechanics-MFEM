// 2D transient nonlinear diffusion driver on the unit square.
//
// Solves (no forcing, no convection):
//   m(u) * (u^{n+1} - u^n) / dt - div(a(u^{n+1}) * grad(u^{n+1})) = 0
//
// with backward Euler in time and full Newton solves per step.
//
// Boundary conditions:
//   - left/right (x boundaries): Neumann flux from analytical solution
//   - top/bottom (y boundaries): natural zero-Neumann (no explicit boundary term)
//
// The analytical solution is from nonlinear_heat.m, extended uniformly in y.

#include "mfem.hpp"
#include "newton_petsc_solver.hpp"

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

   double dt = 1.0e-3;
   double t_final = 1.0e-1;

   // Nonlinear material model:
   // a(u) = a0 + a1 * (u - u_ref)
   // m(u) = m0 + m1 * (u - u_ref)
   double a0 = 10.0;
   double a1 = 0.09;
   double m0 = 8000.0 * 500.0;
   double m1 = 8000.0 * 4.5;
   double u_ref = 300.0;

   // nonlinear_heat.m analytical-solution constants
   double alpha = 2.5e-6;
   double kappa1 = 10.0;
   double kappa2 = 100.0;
   double T0 = 300.0;
   double T1 = 300.0;
   double T2 = 1300.0;
   double qbar = 7.5e5;
   double L = 1.0;
   int series_terms = 400;

   // Newton parameters
   double newton_abs_tol = 1.0e-10;
   double newton_rel_tol = 1.0e-8;
   int newton_max_iter = 20;
   int newton_print_level = 1;

   // PETSc
   string petsc_options_file = "Input/petsc_nonlinear.opts";
   string ksp_prefix = "newton_ls_";
   int petsc_ksp_print_level = 0;

   // Output
   string output_path = "ParaView";
   string collection_name = "nonlinear_convection_diffusion_1D";
   string error_csv = "error_history_nonlinear_1D.csv";
   string newton_csv = "newton_history_nonlinear_1D.csv";
   bool save_paraview = true;
};

struct ExactParams
{
   double alpha = 2.5e-6;
   double kappa1 = 10.0;
   double kappa2 = 100.0;
   double T0 = 300.0;
   double T1 = 300.0;
   double T2 = 1300.0;
   double qbar = 7.5e5;
   double L = 1.0;
   int series_terms = 400;
};

struct Bounds
{
   double xmin = 0.0;
   double xmax = 1.0;
   double ymin = 0.0;
   double ymax = 1.0;
};

struct ExactResult
{
   double u = 0.0;
   double ux = 0.0;
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

   if (n["a0"]) { p.a0 = n["a0"].as<double>(); }
   if (n["a1"]) { p.a1 = n["a1"].as<double>(); }
   if (n["m0"]) { p.m0 = n["m0"].as<double>(); }
   if (n["m1"]) { p.m1 = n["m1"].as<double>(); }
   if (n["u_ref"]) { p.u_ref = n["u_ref"].as<double>(); }

   if (n["alpha"]) { p.alpha = n["alpha"].as<double>(); }
   if (n["kappa1"]) { p.kappa1 = n["kappa1"].as<double>(); }
   if (n["kappa2"]) { p.kappa2 = n["kappa2"].as<double>(); }
   if (n["T0"]) { p.T0 = n["T0"].as<double>(); }
   if (n["T1"]) { p.T1 = n["T1"].as<double>(); }
   if (n["T2"]) { p.T2 = n["T2"].as<double>(); }
   if (n["qbar"]) { p.qbar = n["qbar"].as<double>(); }
   if (n["L"]) { p.L = n["L"].as<double>(); }
   if (n["series_terms"]) { p.series_terms = n["series_terms"].as<int>(); }

   if (n["newton_abs_tol"]) { p.newton_abs_tol = n["newton_abs_tol"].as<double>(); }
   if (n["newton_rel_tol"]) { p.newton_rel_tol = n["newton_rel_tol"].as<double>(); }
   if (n["newton_max_iter"]) { p.newton_max_iter = n["newton_max_iter"].as<int>(); }
   if (n["newton_print_level"]) { p.newton_print_level = n["newton_print_level"].as<int>(); }

   if (n["petsc_options_file"]) { p.petsc_options_file = n["petsc_options_file"].as<string>(); }
   if (n["ksp_prefix"]) { p.ksp_prefix = n["ksp_prefix"].as<string>(); }
   if (n["petsc_ksp_print_level"])
   {
      p.petsc_ksp_print_level = n["petsc_ksp_print_level"].as<int>();
   }

   if (n["output_path"]) { p.output_path = n["output_path"].as<string>(); }
   if (n["collection_name"]) { p.collection_name = n["collection_name"].as<string>(); }
   if (n["error_csv"]) { p.error_csv = n["error_csv"].as<string>(); }
   if (n["newton_csv"]) { p.newton_csv = n["newton_csv"].as<string>(); }
   if (n["save_paraview"]) { p.save_paraview = n["save_paraview"].as<bool>(); }

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
   if (p.series_terms <= 0)
   {
      throw runtime_error("series_terms must be > 0.");
   }
   if (p.L <= 0.0)
   {
      throw runtime_error("L must be > 0.");
   }
   if (p.kappa1 <= 0.0)
   {
      throw runtime_error("kappa1 must be > 0.");
   }
   if (std::abs(p.kappa2 - p.kappa1) <= 1.0e-14)
   {
      throw runtime_error("kappa2 and kappa1 must be different.");
   }
   if (std::abs(p.T2 - p.T1) <= 1.0e-14)
   {
      throw runtime_error("T2 and T1 must be different.");
   }
   if (p.newton_max_iter < 1)
   {
      throw runtime_error("newton_max_iter must be >= 1.");
   }
   if (p.newton_abs_tol <= 0.0 || p.newton_rel_tol <= 0.0)
   {
      throw runtime_error("newton_abs_tol and newton_rel_tol must be > 0.");
   }
}

void ValidateSquareMesh(const ParMesh &pmesh, const double tol)
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

   const double lx = global_max[0] - global_min[0];
   const double ly = global_max[1] - global_min[1];
   const double scale = std::max(std::max(std::abs(lx), std::abs(ly)), 1.0);

   MFEM_VERIFY(lx > tol && ly > tol,
               "Mesh bounding box must have positive lengths. "
               << "Got x=[" << global_min[0] << "," << global_max[0]
               << "], y=[" << global_min[1] << "," << global_max[1] << "].");
   MFEM_VERIFY(std::abs(lx - ly) <= (1.0e-8 * scale + tol),
               "Mesh must be square (equal x/y extents). "
               << "Got lx=" << lx << ", ly=" << ly << ".");
}

void BuildXYBoundaryMarkers(ParMesh &pmesh, Array<int> &x_bdr, Array<int> &y_bdr,
                            Bounds &bounds, const double tol)
{
   const int nbdr = pmesh.bdr_attributes.Max();
   MFEM_VERIFY(nbdr > 0, "Mesh must define boundary attributes.");

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
   MPI_Allreduce(local_min, &bounds.xmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(local_max, &bounds.xmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(local_min + 1, &bounds.ymin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(local_max + 1, &bounds.ymax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

   x_bdr.SetSize(nbdr);
   y_bdr.SetSize(nbdr);
   x_bdr = 0;
   y_bdr = 0;

   Vector x;
   for (int i = 0; i < pmesh.GetNBE(); i++)
   {
      const int attr = pmesh.GetBdrAttribute(i);
      ElementTransformation *T = pmesh.GetBdrElementTransformation(i);
      const IntegrationPoint &ip = Geometries.GetCenter(T->GetGeometryType());
      T->Transform(ip, x);

      if (std::abs(x[0] - bounds.xmin) <= tol || std::abs(x[0] - bounds.xmax) <= tol)
      {
         x_bdr[attr - 1] = 1;
      }
      if (std::abs(x[1] - bounds.ymin) <= tol || std::abs(x[1] - bounds.ymax) <= tol)
      {
         y_bdr[attr - 1] = 1;
      }
   }

   Array<int> x_global(nbdr), y_global(nbdr);
   MPI_Allreduce(x_bdr.GetData(), x_global.GetData(), nbdr, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(y_bdr.GetData(), y_global.GetData(), nbdr, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
   x_bdr = x_global;
   y_bdr = y_global;

   int x_count = 0;
   int y_count = 0;
   for (int i = 0; i < nbdr; i++)
   {
      x_count += x_bdr[i];
      y_count += y_bdr[i];
   }
   MFEM_VERIFY(x_count > 0, "Failed to identify x-boundary attributes.");
   MFEM_VERIFY(y_count > 0, "Failed to identify y-boundary attributes.");
}

ExactResult EvaluateNonlinearHeatExact(const double x, const double t,
                                       const ExactParams &p)
{
   const double pi = M_PI;
   const double L = p.L;
   const double inv_L = 1.0 / L;
   const double inv_L2 = inv_L * inv_L;

   double sum_cos_over_n2 = 0.0;
   double sum_sin_over_n = 0.0;
   const double decay_pref = pi * pi * p.alpha * t * inv_L2;

   for (int n = 1; n <= p.series_terms; n++)
   {
      const double nn = static_cast<double>(n);
      const double exp_term = std::exp(-nn * nn * decay_pref);
      const double arg = nn * pi * x * inv_L;
      sum_cos_over_n2 += exp_term * std::cos(arg) / (nn * nn);
      sum_sin_over_n += exp_term * std::sin(arg) / nn;
   }

   const double f = p.alpha * t * inv_L2 + 1.0 / 3.0 - x * inv_L
                    + 0.5 * x * x * inv_L2 - 2.0 / (pi * pi) * sum_cos_over_n2;
   const double fx = -inv_L + x * inv_L2 + 2.0 / (pi * L) * sum_sin_over_n;

   const double theta0 = (p.T0 - p.T1) +
                         (p.kappa2 - p.kappa1) / (p.T2 - p.T1) / (2.0 * p.kappa1) *
                         (p.T0 - p.T1) * (p.T0 - p.T1);
   const double theta = f * p.qbar * L / p.kappa1 + theta0;

   const double gamma = 2.0 * (p.kappa2 - p.kappa1) / ((p.T2 - p.T1) * p.kappa1);
   const double sqrt_arg = std::max(1.0e-14, 1.0 + gamma * theta);
   const double sqrt_val = std::sqrt(sqrt_arg);

   const double u = p.T1 + (p.T2 - p.T1) * (p.kappa1 / (p.kappa2 - p.kappa1)) *
                    (-1.0 + sqrt_val);

   const double theta_x = p.qbar * L / p.kappa1 * fx;
   const double ux = theta_x / sqrt_val;

   ExactResult out;
   out.u = u;
   out.ux = ux;
   return out;
}

class ExactSolutionCoefficient : public Coefficient
{
public:
   explicit ExactSolutionCoefficient(const ExactParams &p) : p_(p) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x;
      T.Transform(ip, x);
      return EvaluateNonlinearHeatExact(x[0], GetTime(), p_).u;
   }

private:
   ExactParams p_;
};

class ExactFluxXCoefficient : public Coefficient
{
public:
   ExactFluxXCoefficient(const ExactParams &p, double a0, double a1, double u_ref,
                         double xmin, double xmax, double tol)
      : p_(p), a0_(a0), a1_(a1), u_ref_(u_ref),
        xmin_(xmin), xmax_(xmax), tol_(tol) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x;
      T.Transform(ip, x);
      const ExactResult ex = EvaluateNonlinearHeatExact(x[0], GetTime(), p_);
      const double a = a0_ + a1_ * (ex.u - u_ref_);

      double nx = 0.0;
      if (std::abs(x[0] - xmin_) <= tol_) { nx = -1.0; }
      else if (std::abs(x[0] - xmax_) <= tol_) { nx = 1.0; }

      return a * ex.ux * nx;
   }

private:
   ExactParams p_;
   double a0_ = 10.0;
   double a1_ = 0.09;
   double u_ref_ = 300.0;
   double xmin_ = 0.0;
   double xmax_ = 1.0;
   double tol_ = 1.0e-8;
};

class NonlinearMassBEIntegrator : public NonlinearFormIntegrator
{
public:
   NonlinearMassBEIntegrator(double dt, double m0, double m1, double u_ref,
                             Coefficient &u_old_coeff)
      : dt_(dt), m0_(m0), m1_(m1), u_ref_(u_ref), u_old_(&u_old_coeff) { }

   void SetTimeStep(double dt) { dt_ = dt; }

   void AssembleElementVector(const FiniteElement &el, ElementTransformation &T,
                              const Vector &elfun, Vector &elvect) override
   {
      const int dof = el.GetDof();
      elvect.SetSize(dof);
      elvect = 0.0;
      shape_.SetSize(dof);

      const IntegrationRule *ir = IntRule;
      if (!ir)
      {
         const int order = 2 * el.GetOrder() + T.OrderW() + 2;
         ir = &IntRules.Get(el.GetGeomType(), order);
      }

      for (int q = 0; q < ir->GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir->IntPoint(q);
         T.SetIntPoint(&ip);
         el.CalcPhysShape(T, shape_);

         const double u = shape_ * elfun;
         const double u_old = u_old_->Eval(T, ip);
         const double m = m0_ + m1_ * (u - u_ref_);
         const double r_q = m * (u - u_old) / dt_;
         const double w = ip.weight * T.Weight();

         for (int i = 0; i < dof; i++)
         {
            elvect[i] += w * shape_[i] * r_q;
         }
      }
   }

   void AssembleElementGrad(const FiniteElement &el, ElementTransformation &T,
                            const Vector &elfun, DenseMatrix &elmat) override
   {
      const int dof = el.GetDof();
      elmat.SetSize(dof);
      elmat = 0.0;
      shape_.SetSize(dof);

      const IntegrationRule *ir = IntRule;
      if (!ir)
      {
         const int order = 2 * el.GetOrder() + T.OrderW() + 2;
         ir = &IntRules.Get(el.GetGeomType(), order);
      }

      for (int q = 0; q < ir->GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir->IntPoint(q);
         T.SetIntPoint(&ip);
         el.CalcPhysShape(T, shape_);

         const double u = shape_ * elfun;
         const double u_old = u_old_->Eval(T, ip);
         const double m = m0_ + m1_ * (u - u_ref_);
         const double dmdu = m1_;

         const double fac = (m / dt_) + dmdu * (u - u_old) / dt_;
         const double w = ip.weight * T.Weight();

         for (int i = 0; i < dof; i++)
         {
            for (int j = 0; j < dof; j++)
            {
               elmat(i, j) += w * fac * shape_[i] * shape_[j];
            }
         }
      }
   }

private:
   double dt_ = 1.0;
   double m0_ = 1.0;
   double m1_ = 0.0;
   double u_ref_ = 0.0;
   Coefficient *u_old_ = nullptr;
   mutable Vector shape_;
};

class NonlinearDiffusionIntegrator : public NonlinearFormIntegrator
{
public:
   NonlinearDiffusionIntegrator(double a0, double a1, double u_ref)
      : a0_(a0), a1_(a1), u_ref_(u_ref) { }

   void AssembleElementVector(const FiniteElement &el, ElementTransformation &T,
                              const Vector &elfun, Vector &elvect) override
   {
      const int dof = el.GetDof();
      const int dim = el.GetDim();
      elvect.SetSize(dof);
      elvect = 0.0;
      shape_.SetSize(dof);
      dshape_.SetSize(dof, dim);
      grad_u_.SetSize(dim);

      const IntegrationRule *ir = IntRule;
      if (!ir)
      {
         const int order = 2 * el.GetOrder() + T.OrderW() + 2;
         ir = &IntRules.Get(el.GetGeomType(), order);
      }

      for (int q = 0; q < ir->GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir->IntPoint(q);
         T.SetIntPoint(&ip);
         el.CalcPhysShape(T, shape_);
         el.CalcPhysDShape(T, dshape_);

         const double u = shape_ * elfun;
         const double a = a0_ + a1_ * (u - u_ref_);
         const double w = ip.weight * T.Weight();

         grad_u_ = 0.0;
         for (int j = 0; j < dof; j++)
         {
            const double uj = elfun[j];
            for (int d = 0; d < dim; d++)
            {
               grad_u_[d] += uj * dshape_(j, d);
            }
         }

         for (int i = 0; i < dof; i++)
         {
            double dot_i_u = 0.0;
            for (int d = 0; d < dim; d++)
            {
               dot_i_u += dshape_(i, d) * grad_u_[d];
            }
            elvect[i] += w * a * dot_i_u;
         }
      }
   }

   void AssembleElementGrad(const FiniteElement &el, ElementTransformation &T,
                            const Vector &elfun, DenseMatrix &elmat) override
   {
      const int dof = el.GetDof();
      const int dim = el.GetDim();
      elmat.SetSize(dof);
      elmat = 0.0;
      shape_.SetSize(dof);
      dshape_.SetSize(dof, dim);
      grad_u_.SetSize(dim);
      dot_grad_v_grad_u_.SetSize(dof);

      const IntegrationRule *ir = IntRule;
      if (!ir)
      {
         const int order = 2 * el.GetOrder() + T.OrderW() + 2;
         ir = &IntRules.Get(el.GetGeomType(), order);
      }

      for (int q = 0; q < ir->GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir->IntPoint(q);
         T.SetIntPoint(&ip);
         el.CalcPhysShape(T, shape_);
         el.CalcPhysDShape(T, dshape_);

         const double u = shape_ * elfun;
         const double a = a0_ + a1_ * (u - u_ref_);
         const double da = a1_;
         const double w = ip.weight * T.Weight();

         grad_u_ = 0.0;
         for (int j = 0; j < dof; j++)
         {
            const double uj = elfun[j];
            for (int d = 0; d < dim; d++)
            {
               grad_u_[d] += uj * dshape_(j, d);
            }
         }

         for (int i = 0; i < dof; i++)
         {
            double dot_i_u = 0.0;
            for (int d = 0; d < dim; d++)
            {
               dot_i_u += dshape_(i, d) * grad_u_[d];
            }
            dot_grad_v_grad_u_[i] = dot_i_u;
         }

         for (int i = 0; i < dof; i++)
         {
            for (int j = 0; j < dof; j++)
            {
               double dot_i_j = 0.0;
               for (int d = 0; d < dim; d++)
               {
                  dot_i_j += dshape_(i, d) * dshape_(j, d);
               }
               const double term1 = a * dot_i_j;
               const double term2 = da * shape_[j] * dot_grad_v_grad_u_[i];
               elmat(i, j) += w * (term1 + term2);
            }
         }
      }
   }

private:
   double a0_ = 1.0;
   double a1_ = 0.0;
   double u_ref_ = 0.0;
   mutable Vector shape_;
   mutable Vector grad_u_;
   mutable Vector dot_grad_v_grad_u_;
   mutable DenseMatrix dshape_;
};

class ShiftedResidualOperator : public Operator
{
public:
   explicit ShiftedResidualOperator(ParNonlinearForm &residual_form)
      : Operator(residual_form.Height(), residual_form.Width()),
        residual_form_(residual_form) { }

   void SetBoundaryRHS(const Vector &rhs_true) { rhs_true_ = &rhs_true; }

   void Mult(const Vector &x, Vector &y) const override
   {
      residual_form_.Mult(x, y);
      if (rhs_true_ && rhs_true_->Size() == y.Size())
      {
         y -= *rhs_true_;
      }
   }

   Operator &GetGradient(const Vector &x) const override
   {
      return residual_form_.GetGradient(x);
   }

private:
   ParNonlinearForm &residual_form_;
   const Vector *rhs_true_ = nullptr;
};

void PrintConfig(const DriverParams &p)
{
   cout << "Loaded configuration:" << endl;
   cout << "  mesh_file: " << p.mesh_file << endl;
   cout << "  order: " << p.order << endl;
   cout << "  serial_ref_levels: " << p.serial_ref_levels << endl;
   cout << "  par_ref_levels: " << p.par_ref_levels << endl;
   cout << "  dt: " << p.dt << endl;
   cout << "  t_final: " << p.t_final << endl;
   cout << "  a(u): a0 + a1*(u-u_ref) = " << p.a0 << " + " << p.a1
        << "*(u-" << p.u_ref << ")" << endl;
   cout << "  m(u): m0 + m1*(u-u_ref) = " << p.m0 << " + " << p.m1
        << "*(u-" << p.u_ref << ")" << endl;
   cout << "  exact series_terms: " << p.series_terms << endl;
   cout << "  newton_abs_tol: " << p.newton_abs_tol << endl;
   cout << "  newton_rel_tol: " << p.newton_rel_tol << endl;
   cout << "  newton_max_iter: " << p.newton_max_iter << endl;
   cout << "  newton_print_level: " << p.newton_print_level << endl;
   cout << "  petsc_options_file: " << p.petsc_options_file << endl;
   cout << "  ksp_prefix: " << p.ksp_prefix << endl;
   cout << "  petsc_ksp_print_level: " << p.petsc_ksp_print_level << endl;
   cout << "  output_path: " << p.output_path << endl;
   cout << "  collection_name: " << p.collection_name << endl;
   cout << "  error_csv: " << p.error_csv << endl;
   cout << "  newton_csv: " << p.newton_csv << endl;
   cout << "  save_paraview: " << (p.save_paraview ? "true" : "false") << endl;
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const int myid = Mpi::WorldRank();

   string input_file = "Input/input_nonlinear_1d.yaml";
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

   std::string petsc_options_path_storage;
   const char *petsc_file_to_use =
      newton_utils::ResolvePetscOptionsFile(params.petsc_options_file,
                                            myid,
                                            petsc_options_path_storage);
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
      ValidateSquareMesh(*pmesh, 1.0e-8);

      H1_FECollection fec(params.order, 2);
      ParFiniteElementSpace fespace(pmesh.get(), &fec);
      const int true_size = fespace.TrueVSize();
      const HYPRE_BigInt global_true_dofs = fespace.GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "Global true dofs: " << global_true_dofs << endl;
      }

      Bounds bounds;
      Array<int> x_bdr, y_bdr;
      BuildXYBoundaryMarkers(*pmesh, x_bdr, y_bdr, bounds, 1.0e-8);
      if (myid == 0)
      {
         cout << "Boundary marker by attribute for x-boundaries (Neumann exact flux): [";
         for (int i = 0; i < x_bdr.Size(); i++)
         {
            cout << x_bdr[i] << (i + 1 < x_bdr.Size() ? ", " : "");
         }
         cout << "]" << endl;
         cout << "Boundary marker by attribute for y-boundaries (natural zero-Neumann): [";
         for (int i = 0; i < y_bdr.Size(); i++)
         {
            cout << y_bdr[i] << (i + 1 < y_bdr.Size() ? ", " : "");
         }
         cout << "]" << endl;
      }

      ExactParams exact_params;
      exact_params.alpha = params.alpha;
      exact_params.kappa1 = params.kappa1;
      exact_params.kappa2 = params.kappa2;
      exact_params.T0 = params.T0;
      exact_params.T1 = params.T1;
      exact_params.T2 = params.T2;
      exact_params.qbar = params.qbar;
      exact_params.L = params.L;
      exact_params.series_terms = params.series_terms;

      ExactSolutionCoefficient exact_coeff(exact_params);
      ExactFluxXCoefficient flux_coeff(exact_params, params.a0, params.a1, params.u_ref,
                                       bounds.xmin, bounds.xmax, 1.0e-8);

      ParGridFunction u(&fespace);
      ParGridFunction u_old(&fespace);
      ParGridFunction u_exact(&fespace);
      exact_coeff.SetTime(0.0);
      u.ProjectCoefficient(exact_coeff);
      u_old = u;
      u_exact.ProjectCoefficient(exact_coeff);

      GridFunctionCoefficient u_old_coeff(&u_old);

      auto *mass_nl = new NonlinearMassBEIntegrator(params.dt, params.m0, params.m1,
                                                     params.u_ref, u_old_coeff);
      auto *diff_nl = new NonlinearDiffusionIntegrator(params.a0, params.a1, params.u_ref);

      ParNonlinearForm residual_form(&fespace);
      residual_form.SetGradientType(Operator::Hypre_ParCSR);
      residual_form.AddDomainIntegrator(mass_nl);
      residual_form.AddDomainIntegrator(diff_nl);
      Array<int> ess_tdof_list; // pure Neumann setup
      residual_form.SetEssentialTrueDofs(ess_tdof_list);

      ParLinearForm neumann_form(&fespace);
      neumann_form.AddBoundaryIntegrator(new BoundaryLFIntegrator(flux_coeff), x_bdr);
      Vector neumann_true(true_size);
      neumann_true = 0.0;

      ShiftedResidualOperator residual_op(residual_form);
      residual_op.SetBoundaryRHS(neumann_true);

      newton_utils::NewtonConfig newton_cfg;
      newton_cfg.abs_tol = params.newton_abs_tol;
      newton_cfg.rel_tol = params.newton_rel_tol;
      newton_cfg.max_iter = params.newton_max_iter;

      newton_utils::PetscLinearConfig linear_cfg;
      linear_cfg.ksp_prefix = params.ksp_prefix;
      linear_cfg.ksp_print_level = params.petsc_ksp_print_level;

      newton_utils::PetscNewtonSolver newton_solver(MPI_COMM_WORLD,
                                                    newton_cfg,
                                                    linear_cfg);

      const int nsteps = static_cast<int>(std::ceil(params.t_final / params.dt - 1.0e-12));
      if (myid == 0)
      {
         cout << "Time steps: " << nsteps
              << ", nominal final time: " << (nsteps * params.dt) << endl;
      }

      std::ofstream err_csv;
      std::ofstream newton_csv;
      if (myid == 0)
      {
         std::error_code ec;
         filesystem::create_directories(params.output_path, ec);
         if (ec)
         {
            throw runtime_error("Failed to create output directory: " + params.output_path +
                                " (" + ec.message() + ")");
         }

         const filesystem::path err_path =
            filesystem::path(params.output_path) / params.error_csv;
         err_csv.open(err_path);
         if (!err_csv)
         {
            throw runtime_error("Failed to open error CSV: " + err_path.string());
         }
         err_csv << "step,time,abs_l2,rel_l2,newton_iters,final_residual\n";
         err_csv << std::setprecision(16);

         const filesystem::path newton_path =
            filesystem::path(params.output_path) / params.newton_csv;
         newton_csv.open(newton_path);
         if (!newton_csv)
         {
            throw runtime_error("Failed to open Newton CSV: " + newton_path.string());
         }
         newton_csv << "step,time,iter,residual,residual0,rel_residual,"
                    << "update_norm,update0,rel_update,converged\n";
         newton_csv << std::setprecision(16);
      }

      ParaViewDataCollection paraview_dc(params.collection_name.c_str(), pmesh.get());
      if (params.save_paraview)
      {
         paraview_dc.SetPrefixPath(params.output_path.c_str());
         paraview_dc.SetLevelsOfDetail(params.order);
         paraview_dc.SetDataFormat(VTKFormat::BINARY);
         paraview_dc.SetHighOrderOutput(true);
         paraview_dc.RegisterField("u", &u);
         paraview_dc.RegisterField("u_exact", &u_exact);
      }

      int order_quad = std::max(2, 2 * params.order + 3);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int g = 0; g < Geometry::NumGeom; g++)
      {
         irs[g] = &IntRules.Get(g, order_quad);
      }

      auto write_step_output = [&](int step, double t, int newton_iters, double final_residual)
      {
         exact_coeff.SetTime(t);
         u_exact.ProjectCoefficient(exact_coeff);

         const double abs_l2 = u.ComputeL2Error(exact_coeff, irs);
         const double exact_l2 = ComputeGlobalLpNorm(2, exact_coeff, *pmesh, irs);
         const double rel_l2 = (exact_l2 > 1.0e-14) ? abs_l2 / exact_l2 : 0.0;

         if (myid == 0)
         {
            err_csv << step << "," << t << ","
                    << abs_l2 << "," << rel_l2 << ","
                    << newton_iters << "," << final_residual << "\n";
            err_csv.flush();

            if (step <= 10 || step == nsteps || step % 25 == 0)
            {
               cout << "step=" << step << " t=" << t
                    << " newton_iters=" << newton_iters
                    << " relL2=" << rel_l2 << endl;
            }
         }

         if (params.save_paraview)
         {
            paraview_dc.SetCycle(step);
            paraview_dc.SetTime(t);
            paraview_dc.Save();
         }
      };

      write_step_output(0, 0.0, 0, 0.0);

      for (int step = 1; step <= nsteps; step++)
      {
         const double t = step * params.dt;

         u_old = u;
         mass_nl->SetTimeStep(params.dt);

         flux_coeff.SetTime(t);
         neumann_form = 0.0;
         neumann_form.Assemble();
         neumann_form.ParallelAssemble(neumann_true);

         Vector x_true;
         u.GetTrueDofs(x_true);

         auto enforce_bc = [](Vector &) { };
         auto log_iteration = [&](const newton_utils::NewtonIterationInfo &it)
         {
            if (myid == 0)
            {
               newton_csv << step << "," << t << "," << it.iter << ","
                          << it.residual_norm << "," << it.residual_norm0 << ","
                          << it.relative_residual << ","
                          << it.update_norm << "," << it.update_norm0 << ","
                          << it.relative_update << ","
                          << (it.converged ? 1 : 0) << "\n";
               if (params.newton_print_level > 0 && !it.converged)
               {
                  cout << "NR iteration " << it.iter << ":\n"
                       << "|R|/|R0|= " << it.relative_residual << "\n"
                       << "|R|= " << it.residual_norm << "\n"
                       << "|du|/|du0|= " << it.relative_update << "\n"
                       << "|du|= " << it.update_norm << endl;
               }
            }
         };

         const newton_utils::NewtonSolveResult newton_result =
            newton_solver.Solve(residual_op, x_true, enforce_bc, log_iteration, step);

         if (!newton_result.converged)
         {
            throw runtime_error("Newton did not converge at step " + std::to_string(step) +
                                ", t=" + std::to_string(t) +
                                ", final residual=" +
                                std::to_string(newton_result.final_residual));
         }

         u.SetFromTrueDofs(x_true);
         write_step_output(step,
                           t,
                           newton_result.iterations,
                           newton_result.final_residual);
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

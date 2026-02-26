// ALE backward Euler diffusion driver on a FIXED referent mesh — Example 7.3.
//
// Implements the SCL-preserving ALE BE scheme (Eq. 5.7) from:
//   Ivančić, Sheu, Solovchuk, SIAM J. Sci. Comput. (final version in SIAM_ALE.pdf)
//   [local preprint copy also present as arXiv:1809.06553v1].
//
// PDE (physical domain):
//   du/dt - alpha * Laplacian(u) = f   on Omega(t) x (0,T)
//
// Example 7.3 setup:
//   alpha = 0.1,  T = 2,  Omega(t) = Omega_0 = [0,1]^2  for all t
//   (domain is physically fixed, but the interior grid moves)
//   Exact: u(x,t) = sin(t) * cos(2*(x-0.5)^2 + 2*(y-0.5)^2)
//
// Key design choice: the MFEM mesh stays in the REFERENT configuration
// throughout. The ALE displacement d(xhat,t) = A(xhat,t) - xhat is stored as
// a ParaView vector field. Apply "Warp By Vector" with ale_displacement in
// ParaView to visualize the physical (Eulerian) domain.
//
// ALE maps supported:
//   identity   — no motion; result should match diffusion_mms.cpp exactly
//   accuracy_a — Map A from Example 7.3 (SIAM final / arXiv v1 preprint)
//   accuracy_b — Map B from Example 7.3
//
// Important notation note (paper symbol overload):
//   Eq. (1.7) uses the ALE map gradient G = dA/dxhat.
//   Eqs. (2.9), (2.11), (5.x) then use F to denote the 2D cofactor matrix
//   F = cof(G) = J * G^{-T}. This driver uses explicit names (GradA, CofA)
//   to avoid mixing them up.
//
// Bilinear forms assembled on the FIXED reference mesh, each step:
//   LHS:
//     + MassIntegrator(J_{n+1})                       new mass
//     + DiffusionIntegrator((alpha*dt/J_{n+1})*CofA*CofA^T) diffusion (eq 5.4)
//     + ConvectionIntegrator(phi_hat, -1.0)            -M_{n,n+1} conv part (eq 5.6)
//     + MassIntegrator(-div_phi_hat)                   -M_{n,n+1} div part  (eq 5.6)
//   RHS:
//     = M_old * u   (MassIntegrator(J_n) applied to u_old)
//     + dt * DomainLFIntegrator(f(A(xhat,t)) * J_{n+1})  forcing (eq 5.5)
//
// phi_hat = int_0^dt cof(GradA_hat(t))*w_hat(t) dt  is the integrated mapped
// grid flux,
// computed analytically via AleMap::IntegratedMappedGridFlux.
// Because the mesh stays at reference positions, T.Transform(ip) returns xhat
// directly — no map inversion is ever needed.
//
// Configuration: see Input/input_diffusion_mms_ale.yaml.

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

constexpr double kPi = 3.141592653589793238462643383279502884;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

struct DriverParams
{
   string mesh_file;
   int    order             = 1;
   int    serial_ref_levels = 0;
   int    par_ref_levels    = 0;

   double alpha   = 0.1;
   double dt      = 0.05;
   double t_final = 2.0;

   string ale_map = "accuracy_a";   // identity | accuracy_a | accuracy_b

   string petsc_options_file = "Input/petsc.opts";
   int    linear_max_iter    = 400;
   double linear_rel_tol     = 1.0e-10;
   double linear_abs_tol     = 0.0;

   string output_path    = "ParaView/diffusion_mms_ale";
   bool   save_paraview  = true;
   int    paraview_every = 1;   // 0 = initial/final only; N = every N steps
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
   if (n["alpha"])              { p.alpha   = n["alpha"].as<double>(); }
   if (n["dt"])                 { p.dt      = n["dt"].as<double>(); }
   if (n["t_final"])            { p.t_final = n["t_final"].as<double>(); }
   if (n["ale_map"])            { p.ale_map = n["ale_map"].as<string>(); }
   if (n["petsc_options_file"]) { p.petsc_options_file = n["petsc_options_file"].as<string>(); }
   if (n["linear_max_iter"])    { p.linear_max_iter    = n["linear_max_iter"].as<int>(); }
   if (n["linear_rel_tol"])     { p.linear_rel_tol     = n["linear_rel_tol"].as<double>(); }
   if (n["linear_abs_tol"])     { p.linear_abs_tol     = n["linear_abs_tol"].as<double>(); }
   if (n["output_path"])        { p.output_path    = n["output_path"].as<string>(); }
   if (n["save_paraview"])      { p.save_paraview  = n["save_paraview"].as<bool>(); }
   if (n["paraview_every"])     { p.paraview_every = n["paraview_every"].as<int>(); }

   if (p.order < 1)
   {
      throw runtime_error("order must be >= 1.");
   }
   if (p.serial_ref_levels < 0 || p.par_ref_levels < 0)
   {
      throw runtime_error("Refinement levels must be >= 0.");
   }
   if (p.alpha   <= 0.0) { throw runtime_error("alpha must be > 0."); }
   if (p.dt      <= 0.0) { throw runtime_error("dt must be > 0."); }
   if (p.t_final  < 0.0) { throw runtime_error("t_final must be >= 0."); }
   if (p.paraview_every < 0) { throw runtime_error("paraview_every must be >= 0."); }
}

void PrintConfig(const DriverParams &p)
{
   cout << "ALE diffusion driver (Backward Euler, reference configuration)" << endl;
   cout << "  paper:              SIAM_ALE.pdf (final), Example 7.3"
        << " [arXiv:1809.06553v1 preprint]" << endl;
   cout << "  mesh_file:          " << p.mesh_file << endl;
   cout << "  order:              " << p.order << endl;
   cout << "  serial_ref_levels:  " << p.serial_ref_levels << endl;
   cout << "  par_ref_levels:     " << p.par_ref_levels << endl;
   cout << "  alpha:              " << p.alpha << endl;
   cout << "  dt:                 " << p.dt << endl;
   cout << "  t_final:            " << p.t_final << endl;
   cout << "  ale_map:            " << p.ale_map << endl;
   cout << "  petsc_options_file: " << p.petsc_options_file << endl;
   cout << "  output_path:        " << p.output_path << endl;
   cout << "  save_paraview:      " << (p.save_paraview ? "true" : "false") << endl;
   cout << "  paraview_every:     " << p.paraview_every
        << " (0=initial/final only)" << endl;
}

// ---------------------------------------------------------------------------
// AleMap — prescribes A(xhat, t): Omegahat -> Omega(t)
//
//   MapPoint(xhat,t,x)                    — x = A(xhat,t)
//   MapGradient(xhat,t,G)                 — G = dA/dxhat  (2x2)
//   MapCofactor(xhat,t,C)                 — C = cof(G)    (2x2)
//   JacobianDet(xhat,t)                   — J = det(G)
//   IntegratedMappedGridFlux(xhat,t0,t1,  — analytically integrated
//     flux_hat, div_flux_hat)               grid flux phi_hat
//
// All three ALE maps from Example 7.3 of the paper are implemented.
// ---------------------------------------------------------------------------

enum class AleMapKind { Identity, AccuracyA, AccuracyB };

AleMapKind ParseAleMapKind(string s)
{
   for (char &c : s)
   {
      c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
   }
   if (s == "identity" || s == "id")       { return AleMapKind::Identity; }
   if (s == "accuracy_a" || s == "map_a")  { return AleMapKind::AccuracyA; }
   if (s == "accuracy_b" || s == "map_b")  { return AleMapKind::AccuracyB; }
   throw runtime_error(
      "ale_map must be: identity | accuracy_a | accuracy_b. Got: " + s);
}

const char *AleMapName(AleMapKind k)
{
   switch (k)
   {
      case AleMapKind::Identity:  return "identity";
      case AleMapKind::AccuracyA: return "accuracy_a";
      case AleMapKind::AccuracyB: return "accuracy_b";
   }
   return "unknown";
}

class AleMap
{
public:
   explicit AleMap(AleMapKind kind) : kind_(kind) {}

   const char *Name() const { return AleMapName(kind_); }

   // x = A(xhat, t)
   void MapPoint(const Vector &xhat, const double t, Vector &x) const
   {
      MFEM_VERIFY(xhat.Size() == 2, "AleMap expects 2D.");
      x.SetSize(2);
      const double xh = xhat[0], yh = xhat[1];

      switch (kind_)
      {
         case AleMapKind::Identity:
            x = xhat;
            return;

         case AleMapKind::AccuracyA:
         {
            const double amp = AccuracyAAmp_(t);
            x[0] = xh + amp * AccuracyAShape_(xh);
            x[1] = yh + amp * AccuracyAShape_(yh);
            return;
         }
         case AleMapKind::AccuracyB:
         {
            const double amp = AccuracyBAmp_(t);
            const double q   = xh*(1.0-xh)*yh*(1.0-yh);
            x[0] = xh + amp * q;
            x[1] = yh + amp * q;
            return;
         }
      }
   }

   // G = dA/dxhat  (2x2 ALE map gradient / deformation gradient)
   void MapGradient(const Vector &xhat, const double t, DenseMatrix &G) const
   {
      MFEM_VERIFY(xhat.Size() == 2, "AleMap expects 2D.");
      G.SetSize(2, 2);
      G = 0.0;
      const double xh = xhat[0], yh = xhat[1];

      switch (kind_)
      {
         case AleMapKind::Identity:
            G(0,0) = 1.0;
            G(1,1) = 1.0;
            return;

         case AleMapKind::AccuracyA:
         {
            const double a = AccuracyAAmp_(t);
            G(0,0) = 1.0 + a * AccuracyAShapeD1_(xh);
            G(1,1) = 1.0 + a * AccuracyAShapeD1_(yh);
            return;
         }
         case AleMapKind::AccuracyB:
         {
            const double a   = AccuracyBAmp_(t);
            const double ax  = xh*(1.0-xh);
            const double ay  = yh*(1.0-yh);
            const double dax = 1.0 - 2.0*xh;
            const double day = 1.0 - 2.0*yh;
            G(0,0) = 1.0 + a*dax*ay;
            G(0,1) = a*ax*day;
            G(1,0) = a*dax*ay;
            G(1,1) = 1.0 + a*ax*day;
            return;
         }
      }
   }

   // C = cof(G) for G = dA/dxhat. In 2D, cof([[a,b],[c,d]]) = [[d,-b],[-c,a]].
   void MapCofactor(const Vector &xhat, const double t, DenseMatrix &C) const
   {
      DenseMatrix G(2,2);
      MapGradient(xhat, t, G);
      C.SetSize(2,2);
      C(0,0) = G(1,1);
      C(0,1) = -G(0,1);
      C(1,0) = -G(1,0);
      C(1,1) = G(0,0);
   }

   // J = det(dA/dxhat)
   double JacobianDet(const Vector &xhat, const double t) const
   {
      MFEM_VERIFY(xhat.Size() == 2, "AleMap expects 2D.");
      const double xh = xhat[0], yh = xhat[1];

      switch (kind_)
      {
         case AleMapKind::Identity:
            return 1.0;

         case AleMapKind::AccuracyA:
         {
            const double a = AccuracyAAmp_(t);
            return (1.0 + a*AccuracyAShapeD1_(xh))
                 * (1.0 + a*AccuracyAShapeD1_(yh));
         }
         case AleMapKind::AccuracyB:
         {
            const double a   = AccuracyBAmp_(t);
            const double ax  = xh*(1.0-xh);
            const double ay  = yh*(1.0-yh);
            const double dax = 1.0 - 2.0*xh;
            const double day = 1.0 - 2.0*yh;
            // det [[1+a*dax*ay, a*ax*day], [a*dax*ay, 1+a*ax*day]]
            return (1.0 + a*dax*ay)*(1.0 + a*ax*day) - (a*ax*day)*(a*dax*ay);
         }
      }
      return 1.0;
   }

   // Analytically compute the integrated mapped grid flux (paper eq. 5.6):
   //   phi_hat(xhat) = int_{t0}^{t1} cof(dA/dxhat)(xhat,t) * w(xhat,t) dt
   //   div_phi_hat   = div_xhat [ phi_hat(xhat) ]
   // The closed-form expressions below are written in terms of endpoint ALE-map
   // amplitudes and are exact for the Section 4.1 (piecewise-constant-in-time)
   // grid velocity construction for these maps.
   void IntegratedMappedGridFlux(const Vector &xhat,
                                  const double t0,
                                  const double t1,
                                  Vector &phi_hat,
                                  double &div_phi_hat) const
   {
      MFEM_VERIFY(xhat.Size() == 2, "AleMap expects 2D.");
      phi_hat.SetSize(2);

      switch (kind_)
      {
         case AleMapKind::Identity:
         {
            phi_hat     = 0.0;
            div_phi_hat = 0.0;
            return;
         }
         case AleMapKind::AccuracyA:
         {
            // GradA = diag(1+a*g'x, 1+a*g'y), CofA = diag(1+a*g'y, 1+a*g'x).
            // With w = [a_dot*gx, a_dot*gy] (or the endpoint-equivalent Section
            // 4.1 step interpolation), the integrated flux becomes:
            //   phi_x = gx*(i1 + i2*g'y)
            //   phi_y = gy*(i1 + i2*g'x)
            // where i1 = a1-a0, i2 = 0.5*(a1^2-a0^2).
            const double a0 = AccuracyAAmp_(t0);
            const double a1 = AccuracyAAmp_(t1);
            const double i1 = a1 - a0;
            const double i2 = 0.5*(a1*a1 - a0*a0);

            const double xh   = xhat[0], yh = xhat[1];
            const double gx   = AccuracyAShape_(xh);
            const double gxp  = AccuracyAShapeD1_(xh);
            const double gy   = AccuracyAShape_(yh);
            const double gyp  = AccuracyAShapeD1_(yh);

            phi_hat[0] = gx*(i1 + i2*gyp);
            phi_hat[1] = gy*(i1 + i2*gxp);
            div_phi_hat =
               i1*(gxp + gyp) +
               2.0*i2*gxp*gyp;
            return;
         }
         case AleMapKind::AccuracyB:
         {
            // GradA = [[1+a*qx, a*qy], [a*qx, 1+a*qy]]
            // CofA  = [[1+a*qy, -a*qy], [-a*qx, 1+a*qx]]
            // With w = [a_dot*q, a_dot*q], the a-dependent terms cancel:
            //   CofA * w = [a_dot*q, a_dot*q].
            // Therefore phi_hat = (a1-a0) * [q, q].
            const double a0 = AccuracyBAmp_(t0);
            const double a1 = AccuracyBAmp_(t1);
            const double i1 = a1 - a0;

            const double xh  = xhat[0], yh = xhat[1];
            const double ax  = xh*(1.0-xh);
            const double ay  = yh*(1.0-yh);
            const double dax = 1.0 - 2.0*xh;
            const double day = 1.0 - 2.0*yh;
            const double q   = ax*ay;
            const double qx  = dax*ay;
            const double qy  = ax*day;

            phi_hat[0] = i1*q;
            phi_hat[1] = i1*q;
            div_phi_hat = i1*(qx + qy);
            return;
         }
      }
   }

private:
   // Map A amplitude: a(t) = 0.5*sin(pi*t)
   static double AccuracyAAmp_(const double t)
   {
      return 0.5 * std::sin(kPi*t);
   }

   // Map A shape function g(z) = sin(pi * h(z)),  h = z*(1-z)*(z-0.5)
   static double AccuracyAShape_(const double z)
   {
      const double h = ((-z + 1.5)*z - 0.5)*z;   // = -z^3 + 1.5z^2 - 0.5z
      return std::sin(kPi*h);
   }
   static double AccuracyAShapeD1_(const double z)
   {
      const double h  = ((-z + 1.5)*z - 0.5)*z;
      const double hp = (-3.0*z + 3.0)*z - 0.5;
      return kPi * std::cos(kPi*h) * hp;
   }
   static double AccuracyAShapeD2_(const double z)
   {
      const double h   = ((-z + 1.5)*z - 0.5)*z;
      const double hp  = (-3.0*z + 3.0)*z - 0.5;
      const double hpp = -6.0*z + 3.0;
      return -kPi*kPi * std::sin(kPi*h) * hp*hp + kPi * std::cos(kPi*h) * hpp;
   }

   // Map B amplitude: a(t) = sin(pi*t)
   static double AccuracyBAmp_(const double t)
   {
      return std::sin(kPi*t);
   }

   AleMapKind kind_;
};

// ---------------------------------------------------------------------------
// Coefficient classes — all evaluate at REFERENCE mesh points.
// The mesh never moves, so T.Transform(ip, x) gives xhat directly.
// All time-dependent scalar/vector coefficients use the MFEM Coefficient::time
// pattern (SetTime / GetTime) where a single time value suffices.
// The flux coefficients (which need both t_old and t_new) expose SetTimes().
// ---------------------------------------------------------------------------

// J(xhat, t) = det(F(xhat,t))
// Use SetTime(t) before assembling, GetTime() inside Eval.
class AleJacobianCoefficient : public Coefficient
{
public:
   explicit AleJacobianCoefficient(const AleMap &map) : map_(map) {}

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      T.Transform(ip, xhat_);
      return map_.JacobianDet(xhat_, GetTime());
   }

private:
   const AleMap &map_;
   mutable Vector xhat_;
};

// (alpha * dt / J(xhat,t)) * CofA(xhat,t) * CofA^T(xhat,t)  — 2x2 MatrixCoefficient
// where CofA = cof(dA/dxhat). This is the paper's pulled-back diffusion metric.
// alpha and dt are fixed; use SetTime(t_{n+1}) before assembling.
class AleMetricTensorCoefficient : public MatrixCoefficient
{
public:
   AleMetricTensorCoefficient(const AleMap &map, double alpha, double dt)
      : MatrixCoefficient(2), map_(map), alpha_(alpha), dt_(dt) {}

   void Eval(DenseMatrix &M,
             ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      T.Transform(ip, xhat_);
      const double t = GetTime();
      DenseMatrix C(2,2);
      map_.MapCofactor(xhat_, t, C);
      const double J = map_.JacobianDet(xhat_, t);
      MFEM_VERIFY(std::abs(J) > 1.0e-14,
                  "AleMetricTensorCoefficient: degenerate Jacobian at ("
                  << xhat_[0] << "," << xhat_[1] << ") t=" << t);
      const double scale = alpha_ * dt_ / J;
      M.SetSize(2,2);
      MultAAt(C, M);   // M = CofA * CofA^T
      M *= scale;
   }

private:
   const AleMap &map_;
   double alpha_, dt_;
   mutable Vector xhat_;
};

// phi_hat(xhat) = int_{t_old}^{t_new} cof(dA/dxhat)(xhat,t)*w(xhat,t) dt
// Used in ConvectionIntegrator to assemble the first term of -M_{n,n+1}.
// Call SetTimes(t_old, t_new) before each assembly.
class AleIntegratedFluxConvCoefficient : public VectorCoefficient
{
public:
   AleIntegratedFluxConvCoefficient(const AleMap &map,
                                     double t_old,
                                     double t_new)
      : VectorCoefficient(2), map_(map), t_old_(t_old), t_new_(t_new) {}

   void SetTimes(double t_old, double t_new) { t_old_ = t_old; t_new_ = t_new; }

   void Eval(Vector &V,
             ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      T.Transform(ip, xhat_);
      double div_dummy = 0.0;
      map_.IntegratedMappedGridFlux(xhat_, t_old_, t_new_, V, div_dummy);
   }

private:
   const AleMap &map_;
   double t_old_, t_new_;
   mutable Vector xhat_;
};

// div_xhat(phi_hat(xhat))
// Used in MassIntegrator (with sign -1) to assemble the second term of -M_{n,n+1}.
// Call SetTimes(t_old, t_new) before each assembly.
class AleIntegratedFluxDivCoefficient : public Coefficient
{
public:
   AleIntegratedFluxDivCoefficient(const AleMap &map,
                                    double t_old,
                                    double t_new)
      : map_(map), t_old_(t_old), t_new_(t_new) {}

   void SetTimes(double t_old, double t_new) { t_old_ = t_old; t_new_ = t_new; }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      T.Transform(ip, xhat_);
      Vector phi_dummy(2);
      double div_phi = 0.0;
      map_.IntegratedMappedGridFlux(xhat_, t_old_, t_new_, phi_dummy, div_phi);
      return div_phi;
   }

private:
   const AleMap &map_;
   double t_old_, t_new_;
   mutable Vector xhat_;
};

// Exact solution pulled back to reference configuration:
//   u_exact(xhat, t) = sin(t) * cos(2*(Ax-0.5)^2 + 2*(Ay-0.5)^2)
// where (Ax, Ay) = A(xhat, t).
// Use SetTime(t) before projection.
class AleExactCoefficient : public Coefficient
{
public:
   explicit AleExactCoefficient(const AleMap &map) : map_(map) {}

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      T.Transform(ip, xhat_);
      const double t = GetTime();
      map_.MapPoint(xhat_, t, x_phys_);
      const double dx = x_phys_[0] - 0.5;
      const double dy = x_phys_[1] - 0.5;
      const double q  = 2.0*dx*dx + 2.0*dy*dy;
      return std::sin(t) * std::cos(q);
   }

private:
   const AleMap &map_;
   mutable Vector xhat_, x_phys_;
};

// f(A(xhat,t), t) * J(xhat,t)
// Used in DomainLFIntegrator (multiplied by dt outside) to assemble b_{n,n+1}.
// f = du/dt - alpha*Lap(u) evaluated at the physical point x = A(xhat,t).
// Use SetTime(t_{n+1}) before assembly.
class AleForcingJacobianCoefficient : public Coefficient
{
public:
   AleForcingJacobianCoefficient(const AleMap &map, double alpha)
      : map_(map), alpha_(alpha) {}

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      T.Transform(ip, xhat_);
      const double t = GetTime();
      map_.MapPoint(xhat_, t, x_phys_);
      const double J = map_.JacobianDet(xhat_, t);

      const double dx = x_phys_[0] - 0.5;
      const double dy = x_phys_[1] - 0.5;
      const double r2 = dx*dx + dy*dy;
      const double q  = 2.0*r2;

      const double ut  = std::cos(t) * std::cos(q);
      const double lap = std::sin(t) * (-16.0*r2*std::cos(q) - 8.0*std::sin(q));
      const double f   = ut - alpha_*lap;

      return f * J;
   }

private:
   const AleMap &map_;
   double alpha_;
   mutable Vector xhat_, x_phys_;
};

// ALE displacement for ParaView output: d(xhat, t) = A(xhat, t) - xhat.
// Use SetTime(t) before projecting into a ParGridFunction.
class AleDisplacementCoefficient : public VectorCoefficient
{
public:
   explicit AleDisplacementCoefficient(const AleMap &map)
      : VectorCoefficient(2), map_(map) {}

   void Eval(Vector &V,
             ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      T.Transform(ip, xhat_);
      const double t = GetTime();
      map_.MapPoint(xhat_, t, x_phys_);
      V.SetSize(2);
      V[0] = x_phys_[0] - xhat_[0];
      V[1] = x_phys_[1] - xhat_[1];
   }

private:
   const AleMap &map_;
   mutable Vector xhat_, x_phys_;
};

// ProductCoefficient with a constant factor: returns scale * base.Eval().
// MFEM's ProductCoefficient requires two Coefficient objects; this helper
// class avoids creating a temporary ConstantCoefficient on the heap each step.
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void BuildAllBoundaryMarker(const ParMesh &pmesh, Array<int> &ess_bdr)
{
   const int nbdr = pmesh.bdr_attributes.Max();
   MFEM_VERIFY(nbdr > 0, "Mesh must define boundary attributes.");
   ess_bdr.SetSize(nbdr);
   ess_bdr = 1;
}

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

   const double rhs_norm   = B.Norml2();
   const double final_norm = solver.GetFinalNorm();
   const double effective_tol = std::max(params.linear_abs_tol,
                                          params.linear_rel_tol
                                          * std::max(1.0, rhs_norm));

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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const int myid = Mpi::WorldRank();

   // --- Command line ----------------------------------------------------------
   string input_file = "Input/input_diffusion_mms_ale.yaml";
   OptionsParser args(argc, argv);
   args.AddOption(&input_file, "-i", "--input", "YAML input file.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // --- YAML config -----------------------------------------------------------
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

   // --- PETSc -----------------------------------------------------------------
   const char *petsc_file = nullptr;
   if (!params.petsc_options_file.empty())
   {
      ifstream pf(params.petsc_options_file);
      if (pf.good()) { petsc_file = params.petsc_options_file.c_str(); }
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

      // --- ALE map -----------------------------------------------------------
      const AleMapKind map_kind = ParseAleMapKind(params.ale_map);
      AleMap ale_map(map_kind);
      if (myid == 0)
      {
         cout << "ALE map: " << ale_map.Name() << endl;
      }

      // --- Mesh --------------------------------------------------------------
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

      // --- FE spaces ---------------------------------------------------------
      H1_FECollection fec(params.order, pmesh->Dimension());
      // Scalar space for u, u_exact, u_error, ale_jacobian
      ParFiniteElementSpace fes(pmesh.get(), &fec);
      // Vector space for ale_displacement (vdim = space dimension = 2)
      ParFiniteElementSpace vec_fes(pmesh.get(), &fec, pmesh->Dimension());

      if (myid == 0)
      {
         cout << "Scalar true dofs: " << fes.GlobalTrueVSize() << endl;
      }

      // --- Essential DOFs (Dirichlet on entire boundary) ---------------------
      Array<int> ess_bdr;
      BuildAllBoundaryMarker(*pmesh, ess_bdr);
      Array<int> ess_tdof_list;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      // --- Integration rules -------------------------------------------------
      const int order_quad = max(4, 2*params.order + 6);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int g = 0; g < Geometry::NumGeom; g++)
      {
         irs[g] = &IntRules.Get(g, order_quad);
      }

      // --- Persistent coefficient objects ------------------------------------
      // These are updated by calling SetTime / SetTimes before each assembly.
      AleJacobianCoefficient           j_coeff(ale_map);
      AleMetricTensorCoefficient       metric_coeff(ale_map, params.alpha, params.dt);
      AleIntegratedFluxConvCoefficient flux_conv_coeff(ale_map, 0.0, 0.0);
      AleIntegratedFluxDivCoefficient  flux_div_coeff(ale_map, 0.0, 0.0);
      ScaledCoefficient                neg_flux_div_coeff(-1.0, flux_div_coeff);
      AleExactCoefficient              exact_coeff(ale_map);
      AleForcingJacobianCoefficient    forcing_j_coeff(ale_map, params.alpha);
      AleDisplacementCoefficient       disp_coeff(ale_map);

      // --- Grid functions ----------------------------------------------------
      ParGridFunction u(&fes);
      ParGridFunction u_exact_gf(&fes);
      ParGridFunction u_error_gf(&fes);
      ParGridFunction ale_jac_gf(&fes);
      ParGridFunction ale_disp_gf(&vec_fes);

      // Initial condition: u(x,0) = sin(0)*cos(...) = 0
      exact_coeff.SetTime(0.0);
      u.ProjectCoefficient(exact_coeff);
      u.ProjectBdrCoefficient(exact_coeff, ess_bdr);

      // --- ParaView output setup ---------------------------------------------
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

         paraview_dc = make_unique<ParaViewDataCollection>("diffusion_mms_ale",
                                                            pmesh.get());
         paraview_dc->SetPrefixPath(params.output_path);
         paraview_dc->SetLevelsOfDetail(params.order);
         paraview_dc->SetDataFormat(VTKFormat::BINARY);
         paraview_dc->SetHighOrderOutput(true);
         // Solution fields
         paraview_dc->RegisterField("u",                &u);
         paraview_dc->RegisterField("u_exact",          &u_exact_gf);
         paraview_dc->RegisterField("u_error",          &u_error_gf);
         // ALE geometry fields (use ale_displacement with "Warp By Vector")
         paraview_dc->RegisterField("ale_displacement", &ale_disp_gf);
         paraview_dc->RegisterField("ale_jacobian",     &ale_jac_gf);
      }

      // --- Error CSV ---------------------------------------------------------
      ofstream err_csv;
      if (myid == 0)
      {
         const filesystem::path csv_path =
            filesystem::path(params.output_path) / "error_history.csv";
         filesystem::create_directories(params.output_path);
         err_csv.open(csv_path);
         if (!err_csv)
         {
            throw runtime_error("Failed to open error CSV: " + csv_path.string());
         }
         err_csv << "step,time,l2_error,linf_error\n" << setprecision(16);
      }

      // --- Helper: update output fields and optionally save snapshot ---------
      auto compute_and_save = [&](int step, double t, bool force_save)
      {
         exact_coeff.SetTime(t);
         disp_coeff.SetTime(t);
         j_coeff.SetTime(t);

         // L2 error on the reference domain (not J-weighted physical L2 norm).
         const double l2_err = u.ComputeL2Error(exact_coeff, irs);

         // Project fields for visualization
         u_exact_gf.ProjectCoefficient(exact_coeff);
         subtract(u, u_exact_gf, u_error_gf);
         ale_disp_gf.ProjectCoefficient(disp_coeff);
         ale_jac_gf.ProjectCoefficient(j_coeff);

         // Linf error
         const double local_linf = u_error_gf.Normlinf();
         double linf_err = 0.0;
         MPI_Allreduce(&local_linf, &linf_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

         if (myid == 0)
         {
            err_csv << step << "," << t << ","
                    << l2_err << "," << linf_err << "\n";
            err_csv.flush();

            if (step == 0 || step <= 3 || step % 20 == 0)
            {
               cout << "step=" << step << "  t=" << fixed << setprecision(4) << t
                    << "  L2_error=" << scientific << setprecision(6) << l2_err
                    << "  Linf_error=" << linf_err << defaultfloat << endl;
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

      // Save t=0 snapshot (zero displacement, unit Jacobian, zero error)
      compute_and_save(0, 0.0, /*force=*/true);

      // --- Time stepping loop ------------------------------------------------
      const int nsteps = static_cast<int>(
         std::ceil(params.t_final / params.dt - 1.0e-12));

      if (myid == 0)
      {
         cout << "Time steps: " << nsteps
              << "  dt=" << params.dt
              << "  t_final=" << (nsteps * params.dt) << endl;
      }

      double t = 0.0;
      for (int step = 1; step <= nsteps; step++)
      {
         const double t_old = t;
         const double t_new = t + params.dt;

         // ------------------------------------------------------------------
         // [a] Old mass on J_n:  M_old(J_n) * u_old -> rhs
         //     Must be assembled BEFORE advancing the time used by j_coeff.
         // ------------------------------------------------------------------
         j_coeff.SetTime(t_old);
         ParBilinearForm m_old(&fes);
         m_old.AddDomainIntegrator(new MassIntegrator(j_coeff));
         m_old.Assemble();
         m_old.Finalize();

         Vector rhs(fes.GetVSize());
         m_old.Mult(u, rhs);

         // ------------------------------------------------------------------
         // [b] Advance time and update all time-dependent coefficients.
         // ------------------------------------------------------------------
         t = t_new;
         j_coeff.SetTime(t_new);
         metric_coeff.SetTime(t_new);
         flux_conv_coeff.SetTimes(t_old, t_new);
         flux_div_coeff.SetTimes(t_old, t_new);
         exact_coeff.SetTime(t_new);
         forcing_j_coeff.SetTime(t_new);

         // ------------------------------------------------------------------
         // [c] LHS bilinear form (reassembled every step; J and ALE metric vary):
         //   + MassIntegrator(J_{n+1})
         //   + DiffusionIntegrator((alpha*dt/J_{n+1}) * CofA*CofA^T)
         //   + ConvectionIntegrator(phi_hat, -1)    // -M_{n,n+1} conv part
         //   + MassIntegrator(-div_phi_hat)          // -M_{n,n+1} div part
         // ------------------------------------------------------------------
         ParBilinearForm a(&fes);
         a.AddDomainIntegrator(new MassIntegrator(j_coeff));
         a.AddDomainIntegrator(new DiffusionIntegrator(metric_coeff));
         a.AddDomainIntegrator(new ConvectionIntegrator(flux_conv_coeff, -1.0));
         a.AddDomainIntegrator(new MassIntegrator(neg_flux_div_coeff));
         a.Assemble();
         a.Finalize();

         // ------------------------------------------------------------------
         // [d] RHS forcing: rhs += dt * int_Omegahat f(A(xhat,t)) * J * psi dxhat
         // ------------------------------------------------------------------
         ParLinearForm f_form(&fes);
         f_form.AddDomainIntegrator(new DomainLFIntegrator(forcing_j_coeff));
         f_form.Assemble();
         rhs.Add(params.dt, f_form);

         // ------------------------------------------------------------------
         // [e] Dirichlet BCs: for Maps A and B, A(xhat,t)=xhat on boundary,
         //     so exact_coeff reduces to the static Eulerian exact solution
         //     on all boundary nodes. For Identity, it is identical.
         // ------------------------------------------------------------------
         u.ProjectBdrCoefficient(exact_coeff, ess_bdr);

         // ------------------------------------------------------------------
         // [f] Solve: a * u^{n+1} = rhs
         // ------------------------------------------------------------------
         SolveLinearSystem(a, ess_tdof_list, u, rhs, params, myid, step);

         // ------------------------------------------------------------------
         // [g-i] Project visualization fields, log errors, save ParaView
         // ------------------------------------------------------------------
         const bool is_final = (step == nsteps);
         compute_and_save(step, t, is_final);
      }

      // --- Final summary -----------------------------------------------------
      if (myid == 0)
      {
         exact_coeff.SetTime(t);
         const double final_l2 = u.ComputeL2Error(exact_coeff, irs);
         cout << "\nFinal L2 error at t=" << t << ":  " << final_l2 << endl;
         cout << "Output written to:   " << params.output_path << endl;
         if (params.save_paraview)
         {
            cout << "\nParaView tip:" << endl;
            cout << "  1. Open: " << params.output_path
                 << "/diffusion_mms_ale.pvd" << endl;
            cout << "  2. Filters -> Warp By Vector -> select 'ale_displacement'"
                 << endl;
            cout << "  3. Color by 'u', 'u_exact', or 'u_error'" << endl;
            cout << "  4. 'ale_jacobian' shows grid compression (J<1) / "
                 << "expansion (J>1)" << endl;
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

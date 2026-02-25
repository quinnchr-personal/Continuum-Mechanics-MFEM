// 1D Q* ablation verification with laminar blowing correction (Amar et al.).
//
// This driver solves the steady moving-frame constant-property problem:
//   k T_xx + rho * Cv * sdot * T_x = 0
// on x in [0, L] with:
//   T(0) = T_abl, T(L) = T_inf
// and the surface energy balance:
//   -k T_x(0) = q_aero(sdot) - rho * sdot * Qstar
//
// Aerodynamic heating with blowing correction (paper Eq. 79 + laminar Kays model):
//   q_aero = (rho_e u_e Ch0) * phi_blow * (h_r - h_w)
//   phi_blow = xi / (exp(xi) - 1), xi = 2 lambda (rho sdot)/(rho_e u_e Ch0)
//
// The analytical profile from the paper is:
//   (T(x)-T_inf)/(T_abl-T_inf) = exp(-sdot*x/alpha), alpha = k/(rho Cv)
//
// The driver uses a full Newton solve with an analytical Jacobian for the coupled
// unknowns [T_1 ... T_{N-1}, sdot], then reports errors relative to the analytical
// solution (paper Table 4 / Eqs. 76-80).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using std::cerr;
using std::cout;
using std::endl;
using std::size_t;
using std::string;
using std::vector;

namespace
{

struct Params
{
   int num_elements = 512;
   double L = 3.0e-2;          // m (Table 4)
   double k = 0.2;             // W/m-K (Table 4)
   double rho = 2000.0;        // kg/m^3 (Table 4)
   double Cv = 1000.0;         // J/kg-K (Table 4)
   double Qstar = 2.0e6;       // J/kg (Table 4)
   double q_table = 2.0e6;     // W/m^2 (Table 4)
   double T_abl = 800.0;       // K (Table 4)
   double T_inf = 300.0;       // K (Table 4)

   double lambda = 0.5;        // laminar blowing reduction parameter
   double rhoe_ue_Ch0 = 2.3692465; // kg/m^2-s (paper value for laminar case)

   double Cp_air = 1.00416e3;  // J/kg-K (paper Eq. 80 constants)
   double T_ref = 300.0;       // K
   double T_recovery = 1800.0; // K

   int newton_max_iter = 30;
   double newton_abs_tol = 1.0e-10;
   double newton_rel_tol = 1.0e-10;
   double init_s_factor = 1.0;
   bool verbose = true;

   string output_dir = "ParaView/qstar_ablation_blowing_1D";
   string profile_csv = "qstar_blowing_profile.csv";
   string summary_csv = "qstar_blowing_summary.csv";
};

struct BlowingEval
{
   double phi = 1.0;
   double dphi_dsdot = 0.0;
   double xi = 0.0;
};

struct AeroFluxEval
{
   double q = 0.0;
   double dq_dsdot = 0.0;
   double phi = 1.0;
   double xi = 0.0;
   double hw = 0.0;
   double hr = 0.0;
};

struct NewtonResult
{
   bool converged = false;
   int iterations = 0;
   double residual_inf = std::numeric_limits<double>::infinity();
   double update_rel_inf = std::numeric_limits<double>::infinity();
   vector<double> state; // [T1..T_{N-1}, sdot]
};

struct ErrorMetrics
{
   double linf = 0.0;
   double l2_trap = 0.0;
   double rms_nodes = 0.0;
};

void PrintUsage(const char *prog)
{
   cout << "Usage: " << prog << " [options]\n"
        << "Options:\n"
        << "  --num-elements N      Number of uniform cells (default 512)\n"
        << "  --max-iter N          Newton max iterations (default 30)\n"
        << "  --abs-tol X           Newton absolute residual tolerance\n"
        << "  --rel-tol X           Newton relative update tolerance\n"
        << "  --init-s-factor X     Initial sdot guess factor times Table-4 value\n"
        << "  --lambda X            Blowing reduction parameter (default 0.5)\n"
        << "  --rhoe-ue-ch0 X       Uncorrected heat transfer coefficient\n"
        << "  --output-dir PATH     Output directory for CSV files\n"
        << "  --quiet               Reduce iteration logging\n"
        << "  --help                Show this help\n";
}

bool ParseInt(const string &s, int &out)
{
   char *end = nullptr;
   const long v = std::strtol(s.c_str(), &end, 10);
   if (end == nullptr || *end != '\0') { return false; }
   if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max())
   {
      return false;
   }
   out = static_cast<int>(v);
   return true;
}

bool ParseDouble(const string &s, double &out)
{
   char *end = nullptr;
   out = std::strtod(s.c_str(), &end);
   return end != nullptr && *end == '\0' && std::isfinite(out);
}

void ParseArgs(int argc, char *argv[], Params &p)
{
   for (int i = 1; i < argc; i++)
   {
      const string arg(argv[i]);
      auto need_value = [&](const char *name) -> string
      {
         if (i + 1 >= argc)
         {
            throw std::runtime_error(string("Missing value for ") + name);
         }
         ++i;
         return string(argv[i]);
      };

      if (arg == "--help" || arg == "-h")
      {
         PrintUsage(argv[0]);
         std::exit(0);
      }
      else if (arg == "--quiet")
      {
         p.verbose = false;
      }
      else if (arg == "--num-elements")
      {
         int v = 0;
         if (!ParseInt(need_value("--num-elements"), v))
         {
            throw std::runtime_error("Invalid integer for --num-elements");
         }
         p.num_elements = v;
      }
      else if (arg == "--max-iter")
      {
         int v = 0;
         if (!ParseInt(need_value("--max-iter"), v))
         {
            throw std::runtime_error("Invalid integer for --max-iter");
         }
         p.newton_max_iter = v;
      }
      else if (arg == "--abs-tol")
      {
         double v = 0.0;
         if (!ParseDouble(need_value("--abs-tol"), v))
         {
            throw std::runtime_error("Invalid double for --abs-tol");
         }
         p.newton_abs_tol = v;
      }
      else if (arg == "--rel-tol")
      {
         double v = 0.0;
         if (!ParseDouble(need_value("--rel-tol"), v))
         {
            throw std::runtime_error("Invalid double for --rel-tol");
         }
         p.newton_rel_tol = v;
      }
      else if (arg == "--init-s-factor")
      {
         double v = 0.0;
         if (!ParseDouble(need_value("--init-s-factor"), v))
         {
            throw std::runtime_error("Invalid double for --init-s-factor");
         }
         p.init_s_factor = v;
      }
      else if (arg == "--lambda")
      {
         double v = 0.0;
         if (!ParseDouble(need_value("--lambda"), v))
         {
            throw std::runtime_error("Invalid double for --lambda");
         }
         p.lambda = v;
      }
      else if (arg == "--rhoe-ue-ch0")
      {
         double v = 0.0;
         if (!ParseDouble(need_value("--rhoe-ue-ch0"), v))
         {
            throw std::runtime_error("Invalid double for --rhoe-ue-ch0");
         }
         p.rhoe_ue_Ch0 = v;
      }
      else if (arg == "--output-dir")
      {
         p.output_dir = need_value("--output-dir");
      }
      else
      {
         throw std::runtime_error("Unknown option: " + arg);
      }
   }

   if (p.num_elements < 2)
   {
      throw std::runtime_error("num_elements must be >= 2.");
   }
   if (p.L <= 0.0 || p.k <= 0.0 || p.rho <= 0.0 || p.Cv <= 0.0)
   {
      throw std::runtime_error("Material/domain parameters must be positive.");
   }
   if (p.Qstar < 0.0 || p.q_table <= 0.0)
   {
      throw std::runtime_error("Qstar must be >= 0 and q_table must be > 0.");
   }
   if (p.T_abl <= p.T_inf)
   {
      throw std::runtime_error("Expected T_abl > T_inf for this verification case.");
   }
   if (p.rhoe_ue_Ch0 <= 0.0)
   {
      throw std::runtime_error("rhoe_ue_Ch0 must be > 0.");
   }
   if (p.newton_max_iter < 1 || p.newton_abs_tol <= 0.0 || p.newton_rel_tol <= 0.0)
   {
      throw std::runtime_error("Invalid Newton settings.");
   }
   if (p.init_s_factor <= 0.0)
   {
      throw std::runtime_error("init_s_factor must be > 0.");
   }
}

double Alpha(const Params &p)
{
   return p.k / (p.rho * p.Cv);
}

double WallEnthalpy(const Params &p, const double T)
{
   return p.Cp_air * (T - p.T_ref);
}

double RecoveryEnthalpy(const Params &p)
{
   return p.Cp_air * (p.T_recovery - p.T_ref);
}

BlowingEval EvalLaminarBlowing(const Params &p, const double sdot)
{
   BlowingEval out;
   const double m_dot = p.rho * std::max(sdot, 0.0);
   const double dxi_dsdot = 2.0 * p.lambda * p.rho / p.rhoe_ue_Ch0;
   out.xi = 2.0 * p.lambda * m_dot / p.rhoe_ue_Ch0;

   const double xi = out.xi;
   if (std::abs(xi) < 1.0e-8)
   {
      const double xi2 = xi * xi;
      const double xi3 = xi2 * xi;
      out.phi = 1.0 - 0.5 * xi + xi2 / 12.0 - xi3 * xi / 720.0;
      const double dphi_dxi = -0.5 + xi / 6.0 - xi3 / 180.0;
      out.dphi_dsdot = dphi_dxi * dxi_dsdot;
      return out;
   }

   const double em1 = std::expm1(xi);
   const double ex = em1 + 1.0;
   out.phi = xi / em1;
   const double dphi_dxi = (em1 - xi * ex) / (em1 * em1);
   out.dphi_dsdot = dphi_dxi * dxi_dsdot;
   return out;
}

AeroFluxEval EvalAeroFlux(const Params &p, const double sdot, const double T_wall)
{
   AeroFluxEval out;
   const BlowingEval blow = EvalLaminarBlowing(p, sdot);
   out.phi = blow.phi;
   out.xi = blow.xi;
   out.hw = WallEnthalpy(p, T_wall);
   out.hr = RecoveryEnthalpy(p);
   const double dh = out.hr - out.hw;
   out.q = p.rhoe_ue_Ch0 * blow.phi * dh;
   out.dq_dsdot = p.rhoe_ue_Ch0 * blow.dphi_dsdot * dh;
   return out;
}

double TableAnalyticalRecessionRate(const Params &p)
{
   const double denom = p.rho * (p.Cv * (p.T_abl - p.T_inf) + p.Qstar);
   return p.q_table / denom;
}

double ExactTemperature(const Params &p, const double x, const double sdot)
{
   const double alpha = Alpha(p);
   return p.T_inf + (p.T_abl - p.T_inf) * std::exp(-sdot * x / alpha);
}

double ExactTemperatureDerivative(const Params &p, const double x, const double sdot)
{
   const double alpha = Alpha(p);
   const double coeff = -(sdot / alpha) * (p.T_abl - p.T_inf);
   return coeff * std::exp(-sdot * x / alpha);
}

vector<double> SurfaceD1WeightsUniform(const int N, const double dx)
{
   if (N >= 4)
   {
      return {
         -25.0 / (12.0 * dx),
          48.0 / (12.0 * dx),
         -36.0 / (12.0 * dx),
          16.0 / (12.0 * dx),
          -3.0 / (12.0 * dx)
      };
   }
   if (N >= 3)
   {
      return {
         -11.0 / (6.0 * dx),
          18.0 / (6.0 * dx),
          -9.0 / (6.0 * dx),
           2.0 / (6.0 * dx)
      };
   }
   return {
      -3.0 / (2.0 * dx),
       4.0 / (2.0 * dx),
      -1.0 / (2.0 * dx)
   };
}

double InfNorm(const vector<double> &v)
{
   double nrm = 0.0;
   for (double x : v) { nrm = std::max(nrm, std::abs(x)); }
   return nrm;
}

double RelativeInfUpdate(const vector<double> &u, const vector<double> &du)
{
   double r = 0.0;
   for (size_t i = 0; i < u.size(); i++)
   {
      const double scale = std::max(1.0, std::abs(u[i]));
      r = std::max(r, std::abs(du[i]) / scale);
   }
   return r;
}

bool SolveDenseLinearSystem(vector<double> A, vector<double> b, vector<double> &x)
{
   const int n = static_cast<int>(b.size());
   if (static_cast<int>(A.size()) != n * n) { return false; }

   for (int k = 0; k < n; k++)
   {
      int piv = k;
      double piv_abs = std::abs(A[k * n + k]);
      for (int i = k + 1; i < n; i++)
      {
         const double cand = std::abs(A[i * n + k]);
         if (cand > piv_abs)
         {
            piv_abs = cand;
            piv = i;
         }
      }

      if (piv_abs < 1.0e-30) { return false; }
      if (piv != k)
      {
         for (int j = k; j < n; j++)
         {
            std::swap(A[k * n + j], A[piv * n + j]);
         }
         std::swap(b[k], b[piv]);
      }

      const double Akk = A[k * n + k];
      for (int i = k + 1; i < n; i++)
      {
         const double factor = A[i * n + k] / Akk;
         if (factor == 0.0) { continue; }
         A[i * n + k] = 0.0;
         for (int j = k + 1; j < n; j++)
         {
            A[i * n + j] -= factor * A[k * n + j];
         }
         b[i] -= factor * b[k];
      }
   }

   x.assign(n, 0.0);
   for (int i = n - 1; i >= 0; i--)
   {
      double sum = b[i];
      for (int j = i + 1; j < n; j++)
      {
         sum -= A[i * n + j] * x[j];
      }
      const double Aii = A[i * n + i];
      if (std::abs(Aii) < 1.0e-30) { return false; }
      x[i] = sum / Aii;
   }
   return true;
}

void AssembleSystem(const Params &p, const vector<double> &u,
                    vector<double> &R, vector<double> &J)
{
   const int N = p.num_elements;
   const int nT = N - 1;
   const int nU = nT + 1;
   const double dx = p.L / static_cast<double>(N);
   const double dx2 = dx * dx;
   const double sdot = u[nT];
   const double adv_coeff = p.rho * p.Cv * sdot;

   auto Temp = [&](int i) -> double
   {
      if (i == 0) { return p.T_abl; }
      if (i == N) { return p.T_inf; }
      return u[i - 1];
   };

   R.assign(nU, 0.0);
   J.assign(static_cast<size_t>(nU) * static_cast<size_t>(nU), 0.0);
   const vector<double> d1w = SurfaceD1WeightsUniform(N, dx);

   for (int i = 1; i <= N - 1; i++)
   {
      const int row = i - 1;
      const double Tm = Temp(i - 1);
      const double Tc = Temp(i);
      const double Tp = Temp(i + 1);

      R[row] = p.k * (Tp - 2.0 * Tc + Tm) / dx2 +
               adv_coeff * (Tp - Tm) / (2.0 * dx);

      const double dR_dTm = p.k / dx2 - adv_coeff / (2.0 * dx);
      const double dR_dTc = -2.0 * p.k / dx2;
      const double dR_dTp = p.k / dx2 + adv_coeff / (2.0 * dx);

      if (i - 1 >= 1) { J[static_cast<size_t>(row) * nU + (i - 2)] = dR_dTm; }
      J[static_cast<size_t>(row) * nU + (i - 1)] = dR_dTc;
      if (i + 1 <= N - 1) { J[static_cast<size_t>(row) * nU + i] = dR_dTp; }

      J[static_cast<size_t>(row) * nU + nT] = p.rho * p.Cv * (Tp - Tm) / (2.0 * dx);
   }

   const int row_s = nT;
   double dTdx0 = 0.0;
   for (int j = 0; j < static_cast<int>(d1w.size()); j++)
   {
      dTdx0 += d1w[j] * Temp(j);
   }
   const AeroFluxEval aero = EvalAeroFlux(p, sdot, p.T_abl);

   R[row_s] = -p.k * dTdx0 - aero.q + p.rho * sdot * p.Qstar;

   // Surface derivative uses a one-sided stencil; T0 is fixed Dirichlet.
   for (int j = 1; j < static_cast<int>(d1w.size()); j++)
   {
      if (j <= N - 1)
      {
         J[static_cast<size_t>(row_s) * nU + (j - 1)] += -p.k * d1w[j];
      }
   }
   J[static_cast<size_t>(row_s) * nU + nT] = -aero.dq_dsdot + p.rho * p.Qstar;
}

NewtonResult SolveCoupledNewton(const Params &p)
{
   const int N = p.num_elements;
   const int nT = N - 1;
   const int nU = nT + 1;
   const double dx = p.L / static_cast<double>(N);

   NewtonResult out;
   out.state.assign(nU, 0.0);

   const double s_table = TableAnalyticalRecessionRate(p);
   const double s0 = p.init_s_factor * s_table;
   for (int i = 1; i <= nT; i++)
   {
      const double x = i * dx;
      // Start from a blended guess (linear + analytical) to avoid an "exact" seed.
      const double t_lin = p.T_abl + (p.T_inf - p.T_abl) * (x / p.L);
      const double t_ex = ExactTemperature(p, x, s0);
      out.state[i - 1] = 0.5 * (t_lin + t_ex);
   }
   out.state[nT] = s0;

   vector<double> R, J, du, trial_R, trial_J;
   AssembleSystem(p, out.state, R, J);
   const double r0 = std::max(InfNorm(R), 1.0);

   if (p.verbose)
   {
      cout << "Newton iterations (full coupled solve):\n";
      cout << "  iter 0: |R|inf=" << std::scientific << std::setprecision(6)
           << InfNorm(R) << ", sdot=" << out.state[nT] << "\n";
   }

   for (int it = 1; it <= p.newton_max_iter; it++)
   {
      vector<double> rhs(R.size(), 0.0);
      for (size_t i = 0; i < R.size(); i++) { rhs[i] = -R[i]; }

      if (!SolveDenseLinearSystem(J, rhs, du))
      {
         throw std::runtime_error("Newton linear solve failed (singular Jacobian).");
      }

      double alpha = 1.0;
      vector<double> u_trial = out.state;
      double trial_norm = std::numeric_limits<double>::infinity();
      const double Rn = InfNorm(R);

      for (int ls = 0; ls < 20; ls++)
      {
         for (int i = 0; i < nU; i++)
         {
            u_trial[i] = out.state[i] + alpha * du[i];
         }
         if (u_trial[nT] <= 0.0)
         {
            alpha *= 0.5;
            continue;
         }

         AssembleSystem(p, u_trial, trial_R, trial_J);
         trial_norm = InfNorm(trial_R);
         if (trial_norm <= (1.0 - 1.0e-4 * alpha) * Rn || trial_norm < Rn)
         {
            break;
         }
         alpha *= 0.5;
      }

      vector<double> du_scaled = du;
      for (double &v : du_scaled) { v *= alpha; }
      out.update_rel_inf = RelativeInfUpdate(out.state, du_scaled);

      out.state = u_trial;
      R.swap(trial_R);
      J.swap(trial_J);
      out.residual_inf = InfNorm(R);
      out.iterations = it;

      if (p.verbose)
      {
         cout << "  iter " << it
              << ": |R|inf=" << std::scientific << std::setprecision(6)
              << out.residual_inf
              << ", rel_update=" << out.update_rel_inf
              << ", alpha=" << alpha
              << ", sdot=" << out.state[nT]
              << "\n";
      }

      if (out.residual_inf <= p.newton_abs_tol ||
          out.update_rel_inf <= p.newton_rel_tol ||
          out.residual_inf <= p.newton_abs_tol * r0)
      {
         out.converged = true;
         break;
      }
   }

   if (!out.converged)
   {
      out.residual_inf = InfNorm(R);
   }

   return out;
}

ErrorMetrics ComputeProfileErrors(const Params &p, const vector<double> &state,
                                  const double s_reference)
{
   const int N = p.num_elements;
   const int nT = N - 1;
   const double dx = p.L / static_cast<double>(N);

   auto Temp = [&](int i) -> double
   {
      if (i == 0) { return p.T_abl; }
      if (i == N) { return p.T_inf; }
      return state[i - 1];
   };

   ErrorMetrics e;
   double trap_sum = 0.0;
   double node_sum = 0.0;

   for (int i = 0; i <= N; i++)
   {
      const double x = i * dx;
      const double err = Temp(i) - ExactTemperature(p, x, s_reference);
      e.linf = std::max(e.linf, std::abs(err));
      node_sum += err * err;

      if (i < N)
      {
         const double x1 = (i + 1) * dx;
         const double e0 = err;
         const double e1 = Temp(i + 1) - ExactTemperature(p, x1, s_reference);
         trap_sum += 0.5 * dx * (e0 * e0 + e1 * e1);
      }
   }

   e.l2_trap = std::sqrt(trap_sum);
   e.rms_nodes = std::sqrt(node_sum / static_cast<double>(N + 1));
   return e;
}

void WriteProfileCsv(const Params &p, const vector<double> &state,
                     const double s_exact, const double s_num)
{
   const int N = p.num_elements;
   const double dx = p.L / static_cast<double>(N);
   const std::filesystem::path out_dir(p.output_dir);
   std::filesystem::create_directories(out_dir);

   const std::filesystem::path profile_path = out_dir / p.profile_csv;
   std::ofstream pf(profile_path);
   if (!pf)
   {
      throw std::runtime_error("Failed to open profile CSV: " + profile_path.string());
   }

   auto Temp = [&](int i) -> double
   {
      if (i == 0) { return p.T_abl; }
      if (i == N) { return p.T_inf; }
      return state[i - 1];
   };

   pf << "x_m,T_numeric_K,T_exact_table_K,T_exact_numerical_s_K,abs_err_table_K\n";
   pf << std::scientific << std::setprecision(16);
   for (int i = 0; i <= N; i++)
   {
      const double x = i * dx;
      const double T_num = Temp(i);
      const double T_ex_table = ExactTemperature(p, x, s_exact);
      const double T_ex_num = ExactTemperature(p, x, s_num);
      pf << x << "," << T_num << "," << T_ex_table << "," << T_ex_num << ","
         << std::abs(T_num - T_ex_table) << "\n";
   }

   const std::filesystem::path summary_path = out_dir / p.summary_csv;
   const bool write_header =
      !std::filesystem::exists(summary_path) || std::filesystem::file_size(summary_path) == 0;
   std::ofstream sf(summary_path, std::ios::app);
   if (!sf)
   {
      throw std::runtime_error("Failed to open summary CSV: " + summary_path.string());
   }
   if (write_header)
   {
      sf << "num_elements,L_m,k_W_mK,rho_kg_m3,Cv_J_kgK,Qstar_J_kg,q_table_W_m2,"
            "lambda,rhoe_ue_Ch0,sdot_table_m_s,sdot_numerical_m_s,sdot_rel_error\n";
   }
   const double s_table = s_exact;
   sf << std::scientific << std::setprecision(16)
      << p.num_elements << "," << p.L << "," << p.k << "," << p.rho << ","
      << p.Cv << "," << p.Qstar << "," << p.q_table << "," << p.lambda << ","
      << p.rhoe_ue_Ch0 << "," << s_table << "," << s_num << ","
      << (s_num - s_table) / s_table << "\n";
}

} // namespace

int main(int argc, char *argv[])
{
   try
   {
      Params p;
      ParseArgs(argc, argv, p);

      const double s_table = TableAnalyticalRecessionRate(p);
      const AeroFluxEval aero_at_table = EvalAeroFlux(p, s_table, p.T_abl);
      const double alpha = Alpha(p);
      const double q_cond_table = p.rho * p.Cv * s_table * (p.T_abl - p.T_inf);
      const double q_eff_table = p.rhoe_ue_Ch0 * aero_at_table.phi;
      const double rhoe_ue_Ch_corrected_from_table =
         p.q_table / (aero_at_table.hr - aero_at_table.hw);

      cout << std::setprecision(8) << std::scientific;
      cout << "Q* ablation with blowing correction verification (paper Table 4 defaults)\n";
      cout << "  N elements                    = " << p.num_elements << "\n";
      cout << "  domain length L [m]           = " << p.L << "\n";
      cout << "  alpha = k/(rho*Cv) [m^2/s]    = " << alpha << "\n";
      cout << "  Table-4 analytical sdot [m/s] = " << s_table << "\n";
      cout << "  q_table [W/m^2]               = " << p.q_table << "\n";
      cout << "  q_cond(Table-4) [W/m^2]       = " << q_cond_table << "\n";
      cout << "  rho*sdot*Q* [W/m^2]           = " << (p.rho * s_table * p.Qstar) << "\n";
      cout << "  h_r - h_w(Tabl) [J/kg]        = " << (aero_at_table.hr - aero_at_table.hw) << "\n";
      cout << "  phi_blow(Table-4) [-]         = " << aero_at_table.phi
           << " (xi=" << aero_at_table.xi << ")\n";
      cout << "  rhoe*ue*Ch0 (input) [kg/m^2-s]= " << p.rhoe_ue_Ch0 << "\n";
      cout << "  rhoe*ue*Ch_eff(Table-4)       = " << q_eff_table
           << " (paper corrected value ~1.9917145e+00)\n";
      cout << "  corrected coeff from q_table  = " << rhoe_ue_Ch_corrected_from_table << "\n";
      cout << "  q_aero(Table-4 sdot) [W/m^2]  = " << aero_at_table.q
           << " (delta vs q_table=" << (aero_at_table.q - p.q_table) << ")\n";

      NewtonResult nr = SolveCoupledNewton(p);
      if (!nr.converged)
      {
         throw std::runtime_error("Newton did not converge within max iterations.");
      }

      const int nT = p.num_elements - 1;
      const double s_num = nr.state[nT];
      const double dx = p.L / static_cast<double>(p.num_elements);
      const vector<double> d1w = SurfaceD1WeightsUniform(p.num_elements, dx);
      auto Temp = [&](int i) -> double
      {
         if (i == 0) { return p.T_abl; }
         if (i == p.num_elements) { return p.T_inf; }
         return nr.state[i - 1];
      };
      double dTdx0_num = 0.0;
      for (int j = 0; j < static_cast<int>(d1w.size()); j++)
      {
         dTdx0_num += d1w[j] * Temp(j);
      }
      const double q_cond_num = -p.k * dTdx0_num;
      const AeroFluxEval aero_num = EvalAeroFlux(p, s_num, p.T_abl);
      const double q_net_num = aero_num.q - p.rho * s_num * p.Qstar;

      const ErrorMetrics err_vs_table = ComputeProfileErrors(p, nr.state, s_table);
      const ErrorMetrics err_vs_num_exact = ComputeProfileErrors(p, nr.state, s_num);
      const double s_rel_err_table = (s_num - s_table) / s_table;
      const double dTdx0_exact_table = ExactTemperatureDerivative(p, 0.0, s_table);

      cout << "\nConverged solution\n";
      cout << "  Newton iterations             = " << nr.iterations << "\n";
      cout << "  Final |R|inf                  = " << nr.residual_inf << "\n";
      cout << "  Final relative update inf     = " << nr.update_rel_inf << "\n";
      cout << "  Numerical sdot [m/s]          = " << s_num << "\n";
      cout << "  Relative sdot error vs Table4 = " << s_rel_err_table << "\n";
      cout << "  q_cond(numerical) [W/m^2]     = " << q_cond_num << "\n";
      cout << "  q_aero(numerical) [W/m^2]     = " << aero_num.q << "\n";
      cout << "  q_aero - rho*sdot*Q* [W/m^2]  = " << q_net_num << "\n";
      cout << "  Surface balance residual [W/m^2] = "
           << (q_cond_num - q_net_num) << "\n";
      cout << "  dTdx(0) exact(Table4) [K/m]   = " << dTdx0_exact_table << "\n";
      cout << "  dTdx(0) numerical [K/m]       = " << dTdx0_num << "\n";
      cout << "  Profile error vs Table4 exact: Linf[K]=" << err_vs_table.linf
           << ", L2_trap[K*sqrt(m)]=" << err_vs_table.l2_trap
           << ", RMS_nodes[K]=" << err_vs_table.rms_nodes << "\n";
      cout << "  Profile error vs exact(s_num): Linf[K]=" << err_vs_num_exact.linf
           << ", L2_trap[K*sqrt(m)]=" << err_vs_num_exact.l2_trap
           << ", RMS_nodes[K]=" << err_vs_num_exact.rms_nodes << "\n";

      WriteProfileCsv(p, nr.state, s_table, s_num);
      cout << "\nWrote CSV outputs in " << p.output_dir << "\n";
      return 0;
   }
   catch (const std::exception &e)
   {
      cerr << "Error: " << e.what() << endl;
      return 1;
   }
}

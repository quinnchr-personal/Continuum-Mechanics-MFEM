#include "tacot_material.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

double SafePow(const double base, const double exponent)
{
   if (exponent == 0.0)
   {
      return 1.0;
   }
   return std::pow(std::max(base, 1.0e-14), exponent);
}

} // namespace

bool TACOTMaterial::LoadFromYaml(const std::string &path)
{
   YAML::Node root = YAML::LoadFile(path);

   if (!root["constants"] || !root["phases"] || !root["transport"] ||
       !root["reactions"] || !root["tables"])
   {
      throw std::runtime_error(
         "Material YAML must define constants, phases, transport, reactions, and tables.");
   }

   const YAML::Node constants = root["constants"];
   const YAML::Node phases = root["phases"];
   const YAML::Node transport = root["transport"];

   R_ = constants["R"].as<double>();
   if (constants["min_pi_pyro"])
   {
      min_pi_pyro_ = constants["min_pi_pyro"].as<double>();
   }

   rhoI_.clear();
   epsI_.clear();
   for (const YAML::Node &v : phases["rhoI"])
   {
      rhoI_.push_back(v.as<double>());
   }
   for (const YAML::Node &v : phases["epsI"])
   {
      epsI_.push_back(v.as<double>());
   }
   if (rhoI_.size() < 2 || epsI_.size() < 2)
   {
      throw std::runtime_error("Material YAML phases.rhoI and phases.epsI must have at least 2 entries.");
   }

   K_v_ = transport["K_v"].as<double>();
   K_c_ = transport["K_c"].as<double>();
   eps_g_v_ = transport["eps_g_v"].as<double>();
   eps_g_c_ = transport["eps_g_c"].as<double>();

   reactions_.clear();
   sum_F_ = 0.0;
   for (const YAML::Node &reac_node : root["reactions"])
   {
      Reaction r;
      if (reac_node["phase_index"])
      {
         r.phase_index = reac_node["phase_index"].as<int>();
      }
      else if (reac_node["phase"])
      {
         r.phase_index = reac_node["phase"].as<int>() - 1;
      }
      else
      {
         r.phase_index = std::min(1, static_cast<int>(rhoI_.size()) - 1);
      }
      r.F = reac_node["F"].as<double>();
      r.A = reac_node["A"].as<double>();
      r.E = reac_node["E"].as<double>();
      r.m = reac_node["m"].as<double>();
      r.n = reac_node["n"].as<double>();
      r.T_threshold = reac_node["T_threshold"].as<double>();
      r.h = reac_node["h"].as<double>();
      reactions_.push_back(r);
      sum_F_ += r.F;
   }
   if (reactions_.empty())
   {
      throw std::runtime_error("Material YAML must contain at least one reaction.");
   }
   if (sum_F_ <= 0.0)
   {
      throw std::runtime_error("Sum of reaction F coefficients must be positive.");
   }

   const YAML::Node tables = root["tables"];
   virgin_.Load(tables["virgin"], 3, "tables.virgin");
   char_.Load(tables["char"], 3, "tables.char");
   gas_.Load(tables["gas"], 3, "tables.gas");

   has_surface_optics_ = false;
   if (root["source"] && root["source"]["path"])
   {
      const std::string source_dir = root["source"]["path"].as<std::string>();
      const bool ok_v =
         LoadSurfaceOpticsTableFromSource(source_dir + "/virgin", virgin_optics_);
      const bool ok_c =
         LoadSurfaceOpticsTableFromSource(source_dir + "/char", char_optics_);
      has_surface_optics_ = (ok_v && ok_c);
   }

   return true;
}

bool TACOTMaterial::LoadSurfaceOpticsTableFromSource(const std::string &path,
                                                     MultiTable2D &table_out) const
{
   std::ifstream in(path);
   if (!in)
   {
      return false;
   }

   // Parse: p, T, cp, h, ki, kj, kk, emissivity, reflectivity.
   std::map<double, std::vector<std::array<double, 3>>> grouped;
   std::string line;
   while (std::getline(in, line))
   {
      const std::size_t cpos = line.find("//");
      if (cpos != std::string::npos)
      {
         line = line.substr(0, cpos);
      }

      std::istringstream iss(line);
      double p = 0.0;
      double T = 0.0;
      double cp = 0.0;
      double h = 0.0;
      double ki = 0.0;
      double kj = 0.0;
      double kk = 0.0;
      double emissivity = 0.0;
      double reflectivity = 0.0;
      if (!(iss >> p >> T >> cp >> h >> ki >> kj >> kk >> emissivity >> reflectivity))
      {
         continue;
      }

      grouped[p].push_back({T, emissivity, reflectivity});
   }

   if (grouped.empty())
   {
      return false;
   }

   YAML::Node table_node;
   YAML::Node pressure_tables(YAML::NodeType::Sequence);

   for (auto &pr : grouped)
   {
      auto &rows = pr.second;
      std::sort(rows.begin(), rows.end(),
                [](const std::array<double, 3> &a, const std::array<double, 3> &b)
                {
                   return a[0] < b[0];
                });

      YAML::Node ptable;
      ptable["p"] = pr.first;
      YAML::Node yrows(YAML::NodeType::Sequence);
      for (const auto &row : rows)
      {
         YAML::Node yrow(YAML::NodeType::Sequence);
         yrow.push_back(row[0]);
         yrow.push_back(row[1]);
         yrow.push_back(row[2]);
         yrows.push_back(yrow);
      }
      ptable["rows"] = yrows;
      pressure_tables.push_back(ptable);
   }

   table_node["pressure_tables"] = pressure_tables;
   table_out.Load(table_node, 2, "surface_optics");
   return true;
}

TACOTMaterial::InternalState TACOTMaterial::CreateInitialState() const
{
   InternalState s;
   s.extent.assign(reactions_.size(), 0.0);
   s.extent_old = s.extent;
   s.dt = 0.0;
   return s;
}

TACOTMaterial::InternalState TACOTMaterial::SolveReactionExtents(
   const double T,
   const double dt,
   const InternalState &old_state) const
{
   InternalState state;
   state.extent = old_state.extent;
   if (state.extent.size() != reactions_.size())
   {
      state.extent.assign(reactions_.size(), 0.0);
   }
   state.extent_old = state.extent;
   state.dt = dt;

   if (dt <= 0.0)
   {
      return state;
   }

   const double T_clamped = std::max(T, 1.0);
   for (int i = 0; i < static_cast<int>(reactions_.size()); ++i)
   {
      const Reaction &r = reactions_[i];
      const double x_old = Clamp(state.extent_old[i], 0.0, 1.0);

      if (T_clamped < r.T_threshold)
      {
         state.extent[i] = x_old;
         continue;
      }

      const double arrhenius = r.A * std::exp(-r.E / (R_ * T_clamped));
      if (arrhenius <= 0.0)
      {
         state.extent[i] = x_old;
         continue;
      }

      const double tpow = SafePow(T_clamped, r.n);

      double x = x_old;
      for (int it = 0; it < 30; ++it)
      {
         const double one_minus_x = std::max(1.0 - x, 1.0e-14);
         const double rate = arrhenius * SafePow(one_minus_x, r.m) * tpow;
         const double f = x - x_old - dt * rate;
         if (std::abs(f) < 1.0e-13)
         {
            break;
         }

         const double dfdx = 1.0 + dt * arrhenius * r.m *
                             SafePow(one_minus_x, r.m - 1.0) * tpow;

         if (std::abs(dfdx) < 1.0e-14)
         {
            break;
         }

         const double dx = f / dfdx;
         x = Clamp(x - dx, x_old, 1.0);

         if (std::abs(dx) < 1.0e-12)
         {
            break;
         }
      }

      state.extent[i] = Clamp(x, x_old, 1.0);
   }

   return state;
}

TACOTMaterial::SolidProperties TACOTMaterial::EvaluateSolid(
   const double T,
   const double p,
   const InternalState &internal_state) const
{
   if (internal_state.extent.size() != reactions_.size())
   {
      throw std::runtime_error("InternalState.extent size does not match reaction count.");
   }

   SolidProperties out;
   out.pi_i.assign(reactions_.size(), 0.0);

   const double tau = ComputeTau(internal_state.extent);
   out.tau = Clamp(tau, 0.0, 1.0);

   const int nph = std::min(rhoI_.size(), epsI_.size());
   std::vector<double> rho_eps0(nph, 0.0);
   for (int ph = 0; ph < nph; ++ph)
   {
      rho_eps0[ph] = rhoI_[ph] * epsI_[ph];
   }

   double rho_v = 0.0;
   for (int ph = 0; ph < nph; ++ph)
   {
      rho_v += rho_eps0[ph];
   }

   double rho_c = rho_v;
   for (const Reaction &r : reactions_)
   {
      const int ph = static_cast<int>(Clamp(r.phase_index, 0, nph - 1));
      rho_c -= rho_eps0[ph] * r.F;
   }
   rho_c = std::max(rho_c, 1.0e-14);

   std::vector<double> phase_factor(nph, 1.0);
   for (int i = 0; i < static_cast<int>(reactions_.size()); ++i)
   {
      const int ph = static_cast<int>(Clamp(reactions_[i].phase_index, 0, nph - 1));
      const double x = Clamp(internal_state.extent[i], 0.0, 1.0);
      phase_factor[ph] -= reactions_[i].F * x;
   }
   for (int ph = 0; ph < nph; ++ph)
   {
      phase_factor[ph] = std::max(0.0, phase_factor[ph]);
   }

   out.rho_s = 0.0;
   for (int ph = 0; ph < nph; ++ph)
   {
      out.rho_s += rho_eps0[ph] * phase_factor[ph];
   }

   const double cp_v = virgin_.Eval(0, p, T);
   const double h_v = virgin_.Eval(1, p, T);
   const double k_v = virgin_.Eval(2, p, T);

   const double cp_c = char_.Eval(0, p, T);
   const double h_c = char_.Eval(1, p, T);
   const double k_c = char_.Eval(2, p, T);

   const double rho_ref = std::max(out.rho_s, rho_c);
   const double virgin_weight = (rho_ref > 0.0) ? (out.tau * rho_v / rho_ref) : out.tau;

   out.cp = cp_v * virgin_weight + cp_c * (1.0 - virgin_weight);
   out.h = h_v * virgin_weight + h_c * (1.0 - virgin_weight);
   out.k = k_v * virgin_weight + k_c * (1.0 - virgin_weight);
   if (has_surface_optics_)
   {
      const double eps_v = virgin_optics_.Eval(0, p, T);
      const double refl_v = virgin_optics_.Eval(1, p, T);
      const double eps_c = char_optics_.Eval(0, p, T);
      const double refl_c = char_optics_.Eval(1, p, T);
      out.emissivity = eps_v * virgin_weight + eps_c * (1.0 - virgin_weight);
      out.reflectivity = refl_v * virgin_weight + refl_c * (1.0 - virgin_weight);
   }
   else
   {
      // Conservative fallback when TACOT optics source tables are unavailable.
      out.emissivity = 0.85;
      out.reflectivity = 0.15;
   }
   out.emissivity = Clamp(out.emissivity, 0.0, 1.0);
   out.reflectivity = Clamp(out.reflectivity, 0.0, 1.0);
   out.absorptivity = Clamp(1.0 - out.reflectivity, 0.0, 1.0);
   out.K = BlendTau(K_v_, K_c_, out.tau);
   out.eps_g = BlendTau(eps_g_v_, eps_g_c_, out.tau);

   if (internal_state.extent_old.size() == reactions_.size() && internal_state.dt > 0.0)
   {
      for (int i = 0; i < static_cast<int>(reactions_.size()); ++i)
      {
         const double x = Clamp(internal_state.extent[i], 0.0, 1.0);
         const double x_old = Clamp(internal_state.extent_old[i], 0.0, 1.0);
         const double dX = std::max(0.0, x - x_old);
         const int ph = static_cast<int>(Clamp(reactions_[i].phase_index, 0, nph - 1));
         double pi = rho_eps0[ph] * reactions_[i].F * dX / internal_state.dt;
         pi = std::max(0.0, pi);
         if (pi < min_pi_pyro_)
         {
            pi = 0.0;
         }
         out.pi_i[i] = pi;
         out.pi_total += pi;
      }
   }

   double h_bar = h_v;
   if (std::abs(rho_v - rho_c) > 1.0e-14)
   {
      h_bar = (rho_v * h_v - rho_c * h_c) / (rho_v - rho_c);
   }
   // Energy residual uses (... - pyrolysis_heat_sink), so this must store +h_bar*piTotal
   // to recover PATO's +pyrolysisFlux with pyrolysisFlux = -h_bar*piTotal.
   out.pyrolysis_heat_sink = h_bar * out.pi_total;

   out.m_dot_g = out.pi_total;
   return out;
}

TACOTMaterial::GasProperties TACOTMaterial::EvaluateGas(
   const double T,
   const double p,
   const InternalState &internal_state) const
{
   (void)internal_state;

   GasProperties out;
   out.M = gas_.Eval(0, p, T);
   out.h = gas_.Eval(1, p, T);
   out.mu = gas_.Eval(2, p, T);

   const double T_clamped = std::max(T, 1.0);
   out.rho = p * out.M / (R_ * T_clamped);
   return out;
}

TACOTMaterial::SolidSurfaceDerivatives TACOTMaterial::EvaluateSolidSurfaceDerivatives(
   const double T,
   const double p,
   const InternalState &internal_state) const
{
   if (internal_state.extent.size() != reactions_.size())
   {
      throw std::runtime_error("InternalState.extent size does not match reaction count.");
   }

   SolidSurfaceDerivatives out;

   const double tau = Clamp(ComputeTau(internal_state.extent), 0.0, 1.0);
   out.K.value = BlendTau(K_v_, K_c_, tau);
   out.K.dT = 0.0;
   out.K.dp = 0.0;

   const int nph = std::min(rhoI_.size(), epsI_.size());
   std::vector<double> rho_eps0(nph, 0.0);
   for (int ph = 0; ph < nph; ++ph)
   {
      rho_eps0[ph] = rhoI_[ph] * epsI_[ph];
   }

   double rho_v = 0.0;
   for (int ph = 0; ph < nph; ++ph)
   {
      rho_v += rho_eps0[ph];
   }

   double rho_c = rho_v;
   for (const Reaction &r : reactions_)
   {
      const int ph = static_cast<int>(Clamp(r.phase_index, 0, nph - 1));
      rho_c -= rho_eps0[ph] * r.F;
   }
   rho_c = std::max(rho_c, 1.0e-14);

   std::vector<double> phase_factor(nph, 1.0);
   for (int i = 0; i < static_cast<int>(reactions_.size()); ++i)
   {
      const int ph = static_cast<int>(Clamp(reactions_[i].phase_index, 0, nph - 1));
      const double x = Clamp(internal_state.extent[i], 0.0, 1.0);
      phase_factor[ph] -= reactions_[i].F * x;
   }
   for (int ph = 0; ph < nph; ++ph)
   {
      phase_factor[ph] = std::max(0.0, phase_factor[ph]);
   }

   double rho_s = 0.0;
   for (int ph = 0; ph < nph; ++ph)
   {
      rho_s += rho_eps0[ph] * phase_factor[ph];
   }
   const double rho_ref = std::max(rho_s, rho_c);
   const double virgin_weight = (rho_ref > 0.0) ? (tau * rho_v / rho_ref) : tau;

   if (has_surface_optics_)
   {
      const MultiTable2D::EvalResult eps_v = virgin_optics_.EvalWithDerivatives(0, p, T);
      const MultiTable2D::EvalResult refl_v = virgin_optics_.EvalWithDerivatives(1, p, T);
      const MultiTable2D::EvalResult eps_c = char_optics_.EvalWithDerivatives(0, p, T);
      const MultiTable2D::EvalResult refl_c = char_optics_.EvalWithDerivatives(1, p, T);

      const double raw_eps = eps_v.value * virgin_weight + eps_c.value * (1.0 - virgin_weight);
      const double raw_refl = refl_v.value * virgin_weight + refl_c.value * (1.0 - virgin_weight);
      const double raw_eps_dT = eps_v.dT * virgin_weight + eps_c.dT * (1.0 - virgin_weight);
      const double raw_eps_dp = eps_v.dp * virgin_weight + eps_c.dp * (1.0 - virgin_weight);
      const double raw_refl_dT = refl_v.dT * virgin_weight + refl_c.dT * (1.0 - virgin_weight);
      const double raw_refl_dp = refl_v.dp * virgin_weight + refl_c.dp * (1.0 - virgin_weight);

      out.emissivity.value = Clamp(raw_eps, 0.0, 1.0);
      if (raw_eps > 0.0 && raw_eps < 1.0)
      {
         out.emissivity.dT = raw_eps_dT;
         out.emissivity.dp = raw_eps_dp;
      }

      out.reflectivity.value = Clamp(raw_refl, 0.0, 1.0);
      if (raw_refl > 0.0 && raw_refl < 1.0)
      {
         out.reflectivity.dT = raw_refl_dT;
         out.reflectivity.dp = raw_refl_dp;
      }
   }
   else
   {
      out.emissivity.value = 0.85;
      out.reflectivity.value = 0.15;
   }

   const double raw_abs = 1.0 - out.reflectivity.value;
   out.absorptivity.value = Clamp(raw_abs, 0.0, 1.0);
   if (raw_abs > 0.0 && raw_abs < 1.0)
   {
      out.absorptivity.dT = -out.reflectivity.dT;
      out.absorptivity.dp = -out.reflectivity.dp;
   }

   return out;
}

TACOTMaterial::GasSurfaceDerivatives TACOTMaterial::EvaluateGasSurfaceDerivatives(
   const double T,
   const double p,
   const InternalState &internal_state) const
{
   (void)internal_state;

   GasSurfaceDerivatives out;
   const MultiTable2D::EvalResult M_eval = gas_.EvalWithDerivatives(0, p, T);
   const MultiTable2D::EvalResult h_eval = gas_.EvalWithDerivatives(1, p, T);
   const MultiTable2D::EvalResult mu_eval = gas_.EvalWithDerivatives(2, p, T);

   out.M.value = M_eval.value;
   out.M.dT = M_eval.dT;
   out.M.dp = M_eval.dp;

   out.h.value = h_eval.value;
   out.h.dT = h_eval.dT;
   out.h.dp = h_eval.dp;

   out.mu.value = mu_eval.value;
   out.mu.dT = mu_eval.dT;
   out.mu.dp = mu_eval.dp;

   const double T_eff = std::max(T, 1.0);
   const double dT_eff_dT = (T > 1.0) ? 1.0 : 0.0;
   const double rho_scale = p / (R_ * T_eff);

   out.rho.value = rho_scale * out.M.value;
   out.rho.dT = rho_scale * out.M.dT -
                (p * out.M.value / (R_ * T_eff * T_eff)) * dT_eff_dT;
   out.rho.dp = (out.M.value + p * out.M.dp) / (R_ * T_eff);

   return out;
}

double TACOTMaterial::InitialSolidDensity() const
{
   return rhoI_[0] * epsI_[0] + rhoI_[1] * epsI_[1];
}

double TACOTMaterial::CharSolidDensity() const
{
   const double matrix_remaining = Clamp(1.0 - sum_F_, 0.0, 1.0);
   return rhoI_[0] * epsI_[0] + rhoI_[1] * epsI_[1] * matrix_remaining;
}

void TACOTMaterial::CubicSpline1D::Build(const std::vector<double> &x,
                                         const std::vector<double> &y)
{
   if (x.size() != y.size() || x.empty())
   {
      throw std::runtime_error("Invalid spline data: x/y sizes must match and be non-empty.");
   }

   x_ = x;
   y_ = y;

   if (x_.size() == 1)
   {
      y2_.assign(1, 0.0);
      return;
   }

   for (int i = 1; i < static_cast<int>(x_.size()); ++i)
   {
      if (x_[i] <= x_[i - 1])
      {
         throw std::runtime_error("Spline x values must be strictly increasing.");
      }
   }

   const int n = static_cast<int>(x_.size());
   y2_.assign(n, 0.0);
   std::vector<double> u(n - 1, 0.0);

   y2_[0] = 0.0;
   u[0] = 0.0;

   for (int i = 1; i < n - 1; ++i)
   {
      const double sig = (x_[i] - x_[i - 1]) / (x_[i + 1] - x_[i - 1]);
      const double p = sig * y2_[i - 1] + 2.0;
      y2_[i] = (sig - 1.0) / p;

      const double dy1 = (y_[i + 1] - y_[i]) / (x_[i + 1] - x_[i]);
      const double dy0 = (y_[i] - y_[i - 1]) / (x_[i] - x_[i - 1]);
      const double rhs = (6.0 * (dy1 - dy0) / (x_[i + 1] - x_[i - 1]) - sig * u[i - 1]);
      u[i] = rhs / p;
   }

   y2_[n - 1] = 0.0;
   for (int k = n - 2; k >= 0; --k)
   {
      y2_[k] = y2_[k] * y2_[k + 1] + u[k];
   }
}

double TACOTMaterial::CubicSpline1D::EvalClamp(const double xq) const
{
   return EvalClampWithDerivative(xq).value;
}

TACOTMaterial::CubicSpline1D::EvalResult TACOTMaterial::CubicSpline1D::EvalClampWithDerivative(
   const double xq) const
{
   if (x_.empty())
   {
      throw std::runtime_error("Attempted to evaluate an empty spline.");
   }

   EvalResult out;
   if (x_.size() == 1)
   {
      out.value = y_[0];
      return out;
   }

   if (xq <= x_.front())
   {
      out.value = y_.front();
      out.clamped = true;
      return out;
   }
   if (xq >= x_.back())
   {
      out.value = y_.back();
      out.clamped = true;
      return out;
   }

   auto it = std::lower_bound(x_.begin(), x_.end(), xq);
   int khi = static_cast<int>(it - x_.begin());
   if (khi <= 0) { khi = 1; }
   if (khi >= static_cast<int>(x_.size())) { khi = static_cast<int>(x_.size()) - 1; }
   const int klo = khi - 1;

   const double h = x_[khi] - x_[klo];
   const double a = (x_[khi] - xq) / h;
   const double b = (xq - x_[klo]) / h;

   out.value = a * y_[klo] + b * y_[khi] +
               ((a * a * a - a) * y2_[klo] + (b * b * b - b) * y2_[khi]) *
                  (h * h) / 6.0;

   out.deriv = (y_[khi] - y_[klo]) / h +
               (h / 6.0) *
                  (-(3.0 * a * a - 1.0) * y2_[klo] + (3.0 * b * b - 1.0) * y2_[khi]);
   return out;
}

double TACOTMaterial::CubicSpline1D::MinX() const
{
   if (x_.empty())
   {
      return 0.0;
   }
   return x_.front();
}

double TACOTMaterial::CubicSpline1D::MaxX() const
{
   if (x_.empty())
   {
      return 0.0;
   }
   return x_.back();
}

void TACOTMaterial::MultiTable2D::Load(const YAML::Node &table_node,
                                       const int expected_props,
                                       const std::string &label)
{
   if (!table_node || !table_node["pressure_tables"])
   {
      throw std::runtime_error("Missing " + label + ".pressure_tables in material YAML.");
   }

   levels_.clear();

   for (const YAML::Node &ptable : table_node["pressure_tables"])
   {
      PressureLevel level;
      level.p = ptable["p"].as<double>();
      level.props.resize(expected_props);

      std::vector<double> T;
      std::vector<std::vector<double>> values(expected_props);

      for (const YAML::Node &row : ptable["rows"])
      {
         if (!row.IsSequence() || row.size() < static_cast<std::size_t>(expected_props + 1))
         {
            throw std::runtime_error("Malformed row in " + label + ".rows");
         }

         T.push_back(row[0].as<double>());
         for (int j = 0; j < expected_props; ++j)
         {
            values[j].push_back(row[j + 1].as<double>());
         }
      }

      if (T.empty())
      {
         throw std::runtime_error("Empty pressure table in " + label);
      }

      std::vector<int> perm(T.size());
      for (int i = 0; i < static_cast<int>(T.size()); ++i)
      {
         perm[i] = i;
      }
      std::sort(perm.begin(), perm.end(), [&](const int a, const int b) { return T[a] < T[b]; });

      std::vector<double> Ts(T.size(), 0.0);
      std::vector<std::vector<double>> sorted_values(expected_props,
                                                     std::vector<double>(T.size(), 0.0));
      for (int i = 0; i < static_cast<int>(T.size()); ++i)
      {
         Ts[i] = T[perm[i]];
         for (int j = 0; j < expected_props; ++j)
         {
            sorted_values[j][i] = values[j][perm[i]];
         }
      }

      for (int j = 0; j < expected_props; ++j)
      {
         level.props[j].Build(Ts, sorted_values[j]);
      }

      levels_.push_back(level);
   }

   std::sort(levels_.begin(), levels_.end(),
             [](const PressureLevel &a, const PressureLevel &b) { return a.p < b.p; });
}

double TACOTMaterial::MultiTable2D::Eval(const int prop_idx,
                                         const double p,
                                         const double T) const
{
   return EvalWithDerivatives(prop_idx, p, T).value;
}

TACOTMaterial::MultiTable2D::EvalResult TACOTMaterial::MultiTable2D::EvalWithDerivatives(
   const int prop_idx,
   const double p,
   const double T) const
{
   EvalResult out;
   if (levels_.empty())
   {
      throw std::runtime_error("Attempted to evaluate an empty 2D table.");
   }

   if (levels_.size() == 1)
   {
      const CubicSpline1D::EvalResult s = levels_[0].props[prop_idx].EvalClampWithDerivative(T);
      out.value = s.value;
      out.dT = s.deriv;
      out.clamped_T = s.clamped;
      return out;
   }

   double p_clamped = p;
   if (p_clamped <= levels_.front().p)
   {
      p_clamped = levels_.front().p;
      out.clamped_p = true;
   }
   else if (p_clamped >= levels_.back().p)
   {
      p_clamped = levels_.back().p;
      out.clamped_p = true;
   }

   auto it = std::lower_bound(
      levels_.begin(), levels_.end(), p_clamped,
      [](const PressureLevel &lvl, const double pval) { return lvl.p < pval; });

   if (it == levels_.begin())
   {
      const CubicSpline1D::EvalResult s =
         levels_.front().props[prop_idx].EvalClampWithDerivative(T);
      out.value = s.value;
      out.dT = s.deriv;
      out.clamped_T = s.clamped;
      out.dp = 0.0;
      return out;
   }
   if (it == levels_.end())
   {
      const CubicSpline1D::EvalResult s =
         levels_.back().props[prop_idx].EvalClampWithDerivative(T);
      out.value = s.value;
      out.dT = s.deriv;
      out.clamped_T = s.clamped;
      out.dp = 0.0;
      return out;
   }

   int hi = static_cast<int>(it - levels_.begin());
   int lo = hi - 1;
   const double p_tol = 1.0e-12 * std::max(1.0, std::abs(p_clamped));
   if (!out.clamped_p &&
       it != levels_.end() &&
       std::abs(it->p - p_clamped) <= p_tol)
   {
      const int idx = static_cast<int>(it - levels_.begin());
      if (idx > 0 && idx < static_cast<int>(levels_.size()) - 1)
      {
         lo = idx;
         hi = idx + 1;
      }
   }

   const double p0 = levels_[lo].p;
   const double p1 = levels_[hi].p;
   if (std::abs(p1 - p0) < 1.0e-14)
   {
      const CubicSpline1D::EvalResult s =
         levels_[lo].props[prop_idx].EvalClampWithDerivative(T);
      out.value = s.value;
      out.dT = s.deriv;
      out.clamped_T = s.clamped;
      out.dp = 0.0;
      return out;
   }

   const CubicSpline1D::EvalResult v0 =
      levels_[lo].props[prop_idx].EvalClampWithDerivative(T);
   const CubicSpline1D::EvalResult v1 =
      levels_[hi].props[prop_idx].EvalClampWithDerivative(T);
   const double w = (p_clamped - p0) / (p1 - p0);
   out.value = (1.0 - w) * v0.value + w * v1.value;
   out.dT = (1.0 - w) * v0.deriv + w * v1.deriv;
   out.dp = out.clamped_p ? 0.0 : (v1.value - v0.value) / (p1 - p0);
   out.clamped_T = (v0.clamped || v1.clamped);
   return out;
}

double TACOTMaterial::Clamp(const double x, const double lo, const double hi)
{
   return std::max(lo, std::min(hi, x));
}

double TACOTMaterial::BlendTau(const double virgin, const double ch, const double tau) const
{
   const double tc = Clamp(tau, 0.0, 1.0);
   return tc * virgin + (1.0 - tc) * ch;
}

double TACOTMaterial::ComputeTau(const std::vector<double> &extent) const
{
   const int nph = std::min(rhoI_.size(), epsI_.size());
   if (nph == 0)
   {
      return 1.0;
   }

   std::vector<double> rho_eps0(nph, 0.0);
   for (int ph = 0; ph < nph; ++ph)
   {
      rho_eps0[ph] = rhoI_[ph] * epsI_[ph];
   }

   double norm_sum = 0.0;
   for (const Reaction &r : reactions_)
   {
      const int ph = static_cast<int>(Clamp(r.phase_index, 0, nph - 1));
      norm_sum += r.F * rho_eps0[ph];
   }
   if (norm_sum <= 0.0)
   {
      return 1.0;
   }

   double tau = 0.0;
   for (int i = 0; i < static_cast<int>(reactions_.size()); ++i)
   {
      const int ph = static_cast<int>(Clamp(reactions_[i].phase_index, 0, nph - 1));
      const double w = reactions_[i].F * rho_eps0[ph] / norm_sum;
      tau += w * (1.0 - Clamp(extent[i], 0.0, 1.0));
   }
   return Clamp(tau, 0.0, 1.0);
}

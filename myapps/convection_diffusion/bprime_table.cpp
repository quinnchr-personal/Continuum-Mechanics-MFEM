#include "bprime_table.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace
{

double ToLogPressure(const double p)
{
   return std::log(std::max(p, 1.0e-30));
}

} // namespace

void BPrimeTable::CubicSpline1D::Build(const std::vector<double> &x,
                                       const std::vector<double> &y)
{
   if (x.size() != y.size() || x.empty())
   {
      throw std::runtime_error("Invalid spline input data.");
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
      u[i] = (6.0 * (dy1 - dy0) / (x_[i + 1] - x_[i - 1]) - sig * u[i - 1]) / p;
   }

   y2_[n - 1] = 0.0;
   for (int k = n - 2; k >= 0; --k)
   {
      y2_[k] = y2_[k] * y2_[k + 1] + u[k];
   }
}

double BPrimeTable::CubicSpline1D::EvalClamp(const double xq, bool *clamped) const
{
   const EvalResult out = EvalClampWithDerivative(xq);
   if (clamped) { *clamped = out.clamped; }
   return out.value;
}

BPrimeTable::CubicSpline1D::EvalResult BPrimeTable::CubicSpline1D::EvalClampWithDerivative(
   const double xq) const
{
   if (x_.empty())
   {
      throw std::runtime_error("Attempted to evaluate empty spline.");
   }

   EvalResult out;
   if (x_.size() == 1)
   {
      out.value = y_[0];
      return out;
   }

   double q = xq;
   if (q <= x_.front())
   {
      q = x_.front();
      out.clamped = true;
   }
   else if (q >= x_.back())
   {
      q = x_.back();
      out.clamped = true;
   }

   auto it = std::lower_bound(x_.begin(), x_.end(), q);
   int khi = static_cast<int>(it - x_.begin());
   if (khi == 0) { khi = 1; }
   if (khi >= static_cast<int>(x_.size())) { khi = static_cast<int>(x_.size()) - 1; }
   const int klo = khi - 1;

   const double h = x_[khi] - x_[klo];
   const double a = (x_[khi] - q) / h;
   const double b = (q - x_[klo]) / h;

   out.value = a * y_[klo] + b * y_[khi]
               + ((a * a * a - a) * y2_[klo] + (b * b * b - b) * y2_[khi]) *
                    (h * h) / 6.0;

   if (!out.clamped)
   {
      out.deriv = (y_[khi] - y_[klo]) / h +
                  (h / 6.0) *
                     (-(3.0 * a * a - 1.0) * y2_[klo] +
                      (3.0 * b * b - 1.0) * y2_[khi]);
   }

   return out;
}

double BPrimeTable::CubicSpline1D::MinX() const
{
   return x_.empty() ? 0.0 : x_.front();
}

double BPrimeTable::CubicSpline1D::MaxX() const
{
   return x_.empty() ? 0.0 : x_.back();
}

double BPrimeTable::Clamp(const double x, const double lo, const double hi)
{
   return std::max(lo, std::min(hi, x));
}

void BPrimeTable::LoadFromFile(const std::string &path)
{
   std::ifstream in(path);
   if (!in)
   {
      throw std::runtime_error("Failed to open B-prime table: " + path);
   }

   // Group as p -> bg -> list of rows (T, bc, hw).
   using RowTH = std::tuple<double, double, double>;
   std::map<double, std::map<double, std::vector<RowTH>>> grouped;

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
      double bg = 0.0;
      double bc = 0.0;
      double T = 0.0;
      double hw = 0.0;
      if (!(iss >> p >> bg >> bc >> T >> hw))
      {
         continue;
      }

      grouped[p][bg].emplace_back(T, bc, hw);
   }

   if (grouped.empty())
   {
      throw std::runtime_error("B-prime table contains no readable data: " + path);
   }

   levels_.clear();
   levels_.reserve(grouped.size());

   for (const auto &p_pair : grouped)
   {
      PressureLevel pl;
      pl.p = p_pair.first;

      const auto &bg_map = p_pair.second;
      pl.bg_levels.reserve(bg_map.size());
      for (const auto &bg_pair : bg_map)
      {
         BgLevel bl;
         bl.bg = bg_pair.first;

         std::vector<RowTH> rows = bg_pair.second;
         std::sort(rows.begin(), rows.end(),
                   [](const RowTH &a, const RowTH &b)
                   {
                      return std::get<0>(a) < std::get<0>(b);
                   });

         std::vector<double> Ts;
         std::vector<double> bcs;
         std::vector<double> hws;
         Ts.reserve(rows.size());
         bcs.reserve(rows.size());
         hws.reserve(rows.size());

         for (const RowTH &row : rows)
         {
            const double t = std::get<0>(row);
            if (!Ts.empty() && std::abs(t - Ts.back()) < 1.0e-12)
            {
               bcs.back() = std::get<1>(row);
               hws.back() = std::get<2>(row);
               continue;
            }
            Ts.push_back(t);
            bcs.push_back(std::get<1>(row));
            hws.push_back(std::get<2>(row));
         }

         bl.bc_spline.Build(Ts, bcs);
         bl.hw_spline.Build(Ts, hws);
         pl.bg_levels.push_back(std::move(bl));
      }

      std::sort(pl.bg_levels.begin(), pl.bg_levels.end(),
                [](const BgLevel &a, const BgLevel &b)
                {
                   return a.bg < b.bg;
                });

      levels_.push_back(std::move(pl));
   }

   std::sort(levels_.begin(), levels_.end(),
             [](const PressureLevel &a, const PressureLevel &b)
             {
                return a.p < b.p;
             });

   clamp_stats_ = {};
}

BPrimeTable::BgLookup BPrimeTable::EvalAtPressureLevel(const PressureLevel &pl,
                                                        const double bg,
                                                        const double T_k) const
{
   BgLookup out;

   if (pl.bg_levels.empty())
   {
      throw std::runtime_error("Invalid B-prime table pressure level with no B'g rows.");
   }

   const double bg_min = pl.bg_levels.front().bg;
   const double bg_max = pl.bg_levels.back().bg;

   double bg_q = bg;
   if (bg_q < bg_min)
   {
      bg_q = bg_min;
      out.clamped_bg = true;
      out.nonsmooth_bg = true;
   }
   if (bg_q > bg_max)
   {
      bg_q = bg_max;
      out.clamped_bg = true;
      out.nonsmooth_bg = true;
   }

   if (pl.bg_levels.size() == 1)
   {
      const CubicSpline1D::EvalResult bc0 =
         pl.bg_levels[0].bc_spline.EvalClampWithDerivative(T_k);
      const CubicSpline1D::EvalResult hw0 =
         pl.bg_levels[0].hw_spline.EvalClampWithDerivative(T_k);
      out.bc = bc0.value;
      out.hw = hw0.value;
      out.dbc_dbg = 0.0;
      out.dhw_dbg = 0.0;
      out.dbc_dT = bc0.deriv;
      out.dhw_dT = hw0.deriv;
      out.clamped_t = (bc0.clamped || hw0.clamped);
      return out;
   }

   auto it_hi = std::lower_bound(pl.bg_levels.begin(), pl.bg_levels.end(), bg_q,
                                 [](const BgLevel &a, const double val)
                                 {
                                    return a.bg < val;
                                 });

   int hi = static_cast<int>(it_hi - pl.bg_levels.begin());
   if (hi <= 0) { hi = 1; }
   if (hi >= static_cast<int>(pl.bg_levels.size()))
   {
      hi = static_cast<int>(pl.bg_levels.size()) - 1;
   }
   int lo = hi - 1;
   const double bg_tol_hit = 1.0e-12 * std::max(1.0, std::abs(bg_q));
   if (!out.clamped_bg &&
       it_hi != pl.bg_levels.end() &&
       std::abs(it_hi->bg - bg_q) <= bg_tol_hit)
   {
      const int idx = static_cast<int>(it_hi - pl.bg_levels.begin());
      if (idx > 0 && idx < static_cast<int>(pl.bg_levels.size()) - 1)
      {
         lo = idx;
         hi = idx + 1;
      }
   }

   const double bg0 = pl.bg_levels[lo].bg;
   const double bg1 = pl.bg_levels[hi].bg;
   const double bg_tol = 1.0e-12 * std::max(1.0, std::abs(bg_q));
   if (std::abs(bg_q - bg0) <= bg_tol || std::abs(bg_q - bg1) <= bg_tol)
   {
      out.nonsmooth_bg = true;
   }

   const CubicSpline1D::EvalResult bc0 =
      pl.bg_levels[lo].bc_spline.EvalClampWithDerivative(T_k);
   const CubicSpline1D::EvalResult hw0 =
      pl.bg_levels[lo].hw_spline.EvalClampWithDerivative(T_k);
   const CubicSpline1D::EvalResult bc1 =
      pl.bg_levels[hi].bc_spline.EvalClampWithDerivative(T_k);
   const CubicSpline1D::EvalResult hw1 =
      pl.bg_levels[hi].hw_spline.EvalClampWithDerivative(T_k);

   out.clamped_t = (bc0.clamped || hw0.clamped || bc1.clamped || hw1.clamped);

   if (std::abs(bg1 - bg0) < 1.0e-14)
   {
      out.bc = bc0.value;
      out.hw = hw0.value;
      out.dbc_dbg = 0.0;
      out.dhw_dbg = 0.0;
      out.dbc_dT = bc0.deriv;
      out.dhw_dT = hw0.deriv;
   }
   else
   {
      const double w = (bg_q - bg0) / (bg1 - bg0);
      out.bc = (1.0 - w) * bc0.value + w * bc1.value;
      out.hw = (1.0 - w) * hw0.value + w * hw1.value;
      out.dbc_dT = (1.0 - w) * bc0.deriv + w * bc1.deriv;
      out.dhw_dT = (1.0 - w) * hw0.deriv + w * hw1.deriv;
      if (out.clamped_bg)
      {
         out.dbc_dbg = 0.0;
         out.dhw_dbg = 0.0;
      }
      else
      {
         out.dbc_dbg = (bc1.value - bc0.value) / (bg1 - bg0);
         out.dhw_dbg = (hw1.value - hw0.value) / (bg1 - bg0);
      }
   }

   return out;
}

BPrimeTable::LookupResult BPrimeTable::Lookup(const double p_pa,
                                              const double bg,
                                              const double T_k) const
{
   const LookupDerivatives deriv = LookupWithDerivatives(p_pa, bg, T_k);

   LookupResult out;
   out.bc = deriv.bc;
   out.hw = deriv.hw;
   out.clamped_p = deriv.clamped_p;
   out.clamped_bg = deriv.clamped_bg;
   out.clamped_t = deriv.clamped_t;

   if (out.clamped_p) { clamp_stats_.p++; }
   if (out.clamped_bg) { clamp_stats_.bg++; }
   if (out.clamped_t) { clamp_stats_.t++; }

   return out;
}

BPrimeTable::LookupDerivatives BPrimeTable::LookupWithDerivatives(const double p_pa,
                                                                  const double bg,
                                                                  const double T_k) const
{
   if (levels_.empty())
   {
      throw std::runtime_error("BPrimeTable::LookupWithDerivatives called before loading data.");
   }

   LookupDerivatives out;

   const double p_min = levels_.front().p;
   const double p_max = levels_.back().p;

   double p_q = p_pa;
   if (p_q < p_min)
   {
      p_q = p_min;
      out.clamped_p = true;
   }
   if (p_q > p_max)
   {
      p_q = p_max;
      out.clamped_p = true;
   }

   if (levels_.size() == 1)
   {
      const BgLookup val = EvalAtPressureLevel(levels_[0], bg, T_k);
      out.bc = val.bc;
      out.hw = val.hw;
      out.dbc_dbg = val.dbc_dbg;
      out.dbc_dT = val.dbc_dT;
      out.dhw_dbg = val.dhw_dbg;
      out.dhw_dT = val.dhw_dT;
      out.clamped_bg = val.clamped_bg;
      out.clamped_t = val.clamped_t;
      out.nonsmooth_bg = val.nonsmooth_bg;
   }
   else
   {
      auto it_hi = std::lower_bound(levels_.begin(), levels_.end(), p_q,
                                    [](const PressureLevel &a, const double val)
                                    {
                                       return a.p < val;
                                    });

      int hi = static_cast<int>(it_hi - levels_.begin());
      if (hi <= 0) { hi = 1; }
      if (hi >= static_cast<int>(levels_.size()))
      {
         hi = static_cast<int>(levels_.size()) - 1;
      }
      const int lo = hi - 1;

      const double p0 = levels_[lo].p;
      const double p1 = levels_[hi].p;

      const BgLookup v0 = EvalAtPressureLevel(levels_[lo], bg, T_k);
      const BgLookup v1 = EvalAtPressureLevel(levels_[hi], bg, T_k);

      out.clamped_bg = (v0.clamped_bg || v1.clamped_bg);
      out.clamped_t = (v0.clamped_t || v1.clamped_t);
      out.nonsmooth_bg = (v0.nonsmooth_bg || v1.nonsmooth_bg);

      if (std::abs(p1 - p0) < 1.0e-14)
      {
         out.bc = v0.bc;
         out.hw = v0.hw;
         out.dbc_dbg = v0.dbc_dbg;
         out.dbc_dT = v0.dbc_dT;
         out.dhw_dbg = v0.dhw_dbg;
         out.dhw_dT = v0.dhw_dT;
      }
      else
      {
         const double lp0 = ToLogPressure(p0);
         const double lp1 = ToLogPressure(p1);
         const double lpq = ToLogPressure(p_q);
         const double w = (lpq - lp0) / (lp1 - lp0);
         out.bc = (1.0 - w) * v0.bc + w * v1.bc;
         out.hw = (1.0 - w) * v0.hw + w * v1.hw;
         out.dbc_dbg = (1.0 - w) * v0.dbc_dbg + w * v1.dbc_dbg;
         out.dbc_dT = (1.0 - w) * v0.dbc_dT + w * v1.dbc_dT;
         out.dhw_dbg = (1.0 - w) * v0.dhw_dbg + w * v1.dhw_dbg;
         out.dhw_dT = (1.0 - w) * v0.dhw_dT + w * v1.dhw_dT;
      }
   }

   return out;
}

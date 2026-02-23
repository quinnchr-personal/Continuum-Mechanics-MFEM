#pragma once

#include <cstddef>
#include <string>
#include <vector>

class BPrimeTable
{
public:
   struct LookupResult
   {
      double bc = 0.0;
      double hw = 0.0;
      bool clamped_p = false;
      bool clamped_bg = false;
      bool clamped_t = false;
   };

   struct ClampStats
   {
      long long p = 0;
      long long bg = 0;
      long long t = 0;
   };

   struct LookupDerivatives
   {
      double bc = 0.0;
      double hw = 0.0;
      double dbc_dbg = 0.0;
      double dbc_dT = 0.0;
      double dhw_dbg = 0.0;
      double dhw_dT = 0.0;
      bool clamped_p = false;
      bool clamped_bg = false;
      bool clamped_t = false;
      bool nonsmooth_bg = false;
   };

   void LoadFromFile(const std::string &path);
   LookupResult Lookup(double p_pa, double bg, double T_k) const;
   LookupDerivatives LookupWithDerivatives(double p_pa, double bg, double T_k) const;

   ClampStats GetClampStats() const { return clamp_stats_; }

private:
   class CubicSpline1D
   {
   public:
      struct EvalResult
      {
         double value = 0.0;
         double deriv = 0.0;
         bool clamped = false;
      };

      void Build(const std::vector<double> &x, const std::vector<double> &y);
      double EvalClamp(double xq, bool *clamped = nullptr) const;
      EvalResult EvalClampWithDerivative(double xq) const;
      double MinX() const;
      double MaxX() const;

   private:
      std::vector<double> x_;
      std::vector<double> y_;
      std::vector<double> y2_;
   };

   struct BgLevel
   {
      double bg = 0.0;
      CubicSpline1D bc_spline;
      CubicSpline1D hw_spline;
   };

   struct PressureLevel
   {
      double p = 0.0;
      std::vector<BgLevel> bg_levels;
   };

   struct BgLookup
   {
      double bc = 0.0;
      double hw = 0.0;
      double dbc_dbg = 0.0;
      double dbc_dT = 0.0;
      double dhw_dbg = 0.0;
      double dhw_dT = 0.0;
      bool clamped_bg = false;
      bool clamped_t = false;
      bool nonsmooth_bg = false;
   };

   static double Clamp(double x, double lo, double hi);
   BgLookup EvalAtPressureLevel(const PressureLevel &pl,
                                double bg,
                                double T_k) const;

   std::vector<PressureLevel> levels_;
   mutable ClampStats clamp_stats_;
};

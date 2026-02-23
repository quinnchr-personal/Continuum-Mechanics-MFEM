#pragma once

#include <yaml-cpp/yaml.h>

#include <string>
#include <vector>

class TACOTMaterial
{
public:
   struct Reaction
   {
      int phase_index = 1; // 0-based solid phase index (default: matrix phase)
      double F = 0.0;
      double A = 0.0;
      double E = 0.0;
      double m = 1.0;
      double n = 0.0;
      double T_threshold = 0.0;
      double h = 0.0;
   };

   struct InternalState
   {
      std::vector<double> extent;
      std::vector<double> extent_old;
      double dt = 0.0;
   };

   struct SolidProperties
   {
      double cp = 0.0;
      double h = 0.0;
      double k = 0.0;
      double emissivity = 1.0;
      double reflectivity = 0.0;
      double absorptivity = 1.0;
      double K = 0.0;
      double eps_g = 0.0;
      double rho_s = 0.0;
      double tau = 1.0;
      double pi_total = 0.0;
      double m_dot_g = 0.0;
      double pyrolysis_heat_sink = 0.0;
      std::vector<double> pi_i;
   };

   struct GasProperties
   {
      double M = 0.0;
      double h = 0.0;
      double mu = 0.0;
      double rho = 0.0;
   };

   struct ScalarDerivatives
   {
      double value = 0.0;
      double dT = 0.0;
      double dp = 0.0;
   };

   struct SolidSurfaceDerivatives
   {
      ScalarDerivatives K;
      ScalarDerivatives emissivity;
      ScalarDerivatives absorptivity;
      ScalarDerivatives reflectivity;
   };

   struct GasSurfaceDerivatives
   {
      ScalarDerivatives M;
      ScalarDerivatives h;
      ScalarDerivatives mu;
      ScalarDerivatives rho;
   };

   bool LoadFromYaml(const std::string &path);

   InternalState CreateInitialState() const;
   InternalState SolveReactionExtents(double T, double dt,
                                      const InternalState &old_state) const;
   SolidProperties EvaluateSolid(double T, double p,
                                 const InternalState &internal_state) const;
   GasProperties EvaluateGas(double T, double p,
                             const InternalState &internal_state) const;
   SolidSurfaceDerivatives EvaluateSolidSurfaceDerivatives(
      double T, double p, const InternalState &internal_state) const;
   GasSurfaceDerivatives EvaluateGasSurfaceDerivatives(
      double T, double p, const InternalState &internal_state) const;

   int NumReactions() const { return static_cast<int>(reactions_.size()); }
   double GasConstant() const { return R_; }

   double InitialSolidDensity() const;
   double CharSolidDensity() const;

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
      double EvalClamp(double xq) const;
      EvalResult EvalClampWithDerivative(double xq) const;
      double MinX() const;
      double MaxX() const;

   private:
      std::vector<double> x_;
      std::vector<double> y_;
      std::vector<double> y2_;
   };

   struct PressureLevel
   {
      double p = 0.0;
      std::vector<CubicSpline1D> props;
   };

   class MultiTable2D
   {
   public:
      struct EvalResult
      {
         double value = 0.0;
         double dp = 0.0;
         double dT = 0.0;
         bool clamped_p = false;
         bool clamped_T = false;
      };

      void Load(const YAML::Node &table_node, int expected_props,
                const std::string &label);
      double Eval(int prop_idx, double p, double T) const;
      EvalResult EvalWithDerivatives(int prop_idx, double p, double T) const;
      bool Empty() const { return levels_.empty(); }

   private:
      std::vector<PressureLevel> levels_;
   };

   static double Clamp(double x, double lo, double hi);
   bool LoadSurfaceOpticsTableFromSource(const std::string &path,
                                         MultiTable2D &table_out) const;

   double BlendTau(double virgin, double ch, double tau) const;
   double ComputeTau(const std::vector<double> &extent) const;

   double R_ = 8.31446261815324;
   std::vector<double> rhoI_;
   std::vector<double> epsI_;
   double K_v_ = 0.0;
   double K_c_ = 0.0;
   double eps_g_v_ = 0.0;
   double eps_g_c_ = 0.0;

   std::vector<Reaction> reactions_;
   double sum_F_ = 1.0;
   double min_pi_pyro_ = 1.0e-9;

   MultiTable2D virgin_;
   MultiTable2D char_;
   MultiTable2D gas_;
   MultiTable2D virgin_optics_;
   MultiTable2D char_optics_;
   bool has_surface_optics_ = false;
};

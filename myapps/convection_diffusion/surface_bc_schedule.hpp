#pragma once

#include <string>
#include <vector>

class SurfaceBCSchedule
{
public:
   struct BoundaryState
   {
      double p_w = 101325.0;
      double rhoeUeCH = 0.0;
      double h_r = 0.0;
      double hconv = 0.0;
      double Tedge = 300.0;
      bool has_hconv = false;
      bool has_Tedge = false;
      int chemistryOn = 1;
   };

   void LoadFromFile(const std::string &path);
   BoundaryState Eval(double time) const;

private:
   struct Row
   {
      double t = 0.0;
      double p_w = 101325.0;
      double rhoeUeCH = 0.0;
      double h_r = 0.0;
      double hconv = 0.0;
      double Tedge = 300.0;
      bool has_hconv = false;
      bool has_Tedge = false;
      int chemistryOn = 1;
   };

   std::vector<Row> rows_;
};

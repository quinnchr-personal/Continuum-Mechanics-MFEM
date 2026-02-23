#include "surface_bc_schedule.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

void SurfaceBCSchedule::LoadFromFile(const std::string &path)
{
   std::ifstream in(path);
   if (!in)
   {
      throw std::runtime_error("Failed to open surface BC schedule file: " + path);
   }

   rows_.clear();

   std::string line;
   int line_no = 0;
   int expected_columns = 0;
   while (std::getline(in, line))
   {
      ++line_no;
      const std::size_t cpos = line.find("//");
      if (cpos != std::string::npos)
      {
         line = line.substr(0, cpos);
      }

      std::istringstream iss(line);
      std::vector<double> cols;
      double value = 0.0;
      while (iss >> value)
      {
         cols.push_back(value);
      }
      if (iss.fail() && !iss.eof())
      {
         throw std::runtime_error("Invalid numeric value in surface BC schedule at line " +
                                  std::to_string(line_no) + ": " + path);
      }
      if (cols.empty())
      {
         continue;
      }
      if (cols.size() != 5 && cols.size() != 7)
      {
         throw std::runtime_error("Surface BC schedule line " + std::to_string(line_no) +
                                  " must have 5 or 7 numeric columns: " + path);
      }
      if (expected_columns == 0)
      {
         expected_columns = static_cast<int>(cols.size());
      }
      else if (static_cast<int>(cols.size()) != expected_columns)
      {
         throw std::runtime_error("Inconsistent surface BC schedule column count at line " +
                                  std::to_string(line_no) + ": " + path);
      }

      Row r;
      r.t = cols[0];
      r.p_w = cols[1];
      r.rhoeUeCH = cols[2];
      r.h_r = cols[3];

      double chemistry = cols[4];
      if (cols.size() == 7)
      {
         r.hconv = cols[4];
         r.Tedge = cols[5];
         r.has_hconv = true;
         r.has_Tedge = true;
         chemistry = cols[6];
      }

      r.chemistryOn = (chemistry >= 0.5) ? 1 : 0;
      rows_.push_back(r);
   }

   if (rows_.empty())
   {
      throw std::runtime_error("Surface BC schedule has no valid rows: " + path);
   }

   std::sort(rows_.begin(), rows_.end(),
             [](const Row &a, const Row &b)
             {
                return a.t < b.t;
             });
}

SurfaceBCSchedule::BoundaryState SurfaceBCSchedule::Eval(const double time) const
{
   if (rows_.empty())
   {
      throw std::runtime_error("SurfaceBCSchedule::Eval called before loading schedule.");
   }

   if (time <= rows_.front().t)
   {
      return {rows_.front().p_w,
              rows_.front().rhoeUeCH,
              rows_.front().h_r,
              rows_.front().hconv,
              rows_.front().Tedge,
              rows_.front().has_hconv,
              rows_.front().has_Tedge,
              rows_.front().chemistryOn};
   }
   if (time >= rows_.back().t)
   {
      return {rows_.back().p_w,
              rows_.back().rhoeUeCH,
              rows_.back().h_r,
              rows_.back().hconv,
              rows_.back().Tedge,
              rows_.back().has_hconv,
              rows_.back().has_Tedge,
              rows_.back().chemistryOn};
   }

   int hi = 1;
   while (hi < static_cast<int>(rows_.size()) && rows_[hi].t < time)
   {
      hi++;
   }
   const int lo = hi - 1;

   const Row &a = rows_[lo];
   const Row &b = rows_[hi];

   const double dt = b.t - a.t;
   double w = 0.0;
   if (dt > 1.0e-14)
   {
      w = (time - a.t) / dt;
   }

   BoundaryState out;
   out.p_w = (1.0 - w) * a.p_w + w * b.p_w;
   out.rhoeUeCH = (1.0 - w) * a.rhoeUeCH + w * b.rhoeUeCH;
   out.h_r = (1.0 - w) * a.h_r + w * b.h_r;
   if (a.has_hconv && b.has_hconv)
   {
      out.hconv = (1.0 - w) * a.hconv + w * b.hconv;
      out.has_hconv = true;
   }
   else
   {
      out.hconv = 0.0;
      out.has_hconv = false;
   }
   if (a.has_Tedge && b.has_Tedge)
   {
      out.Tedge = (1.0 - w) * a.Tedge + w * b.Tedge;
      out.has_Tedge = true;
   }
   else
   {
      out.Tedge = 300.0;
      out.has_Tedge = false;
   }

   // Piecewise-left-constant in time for chemistry toggle.
   out.chemistryOn = a.chemistryOn;
   return out;
}

// Eigen-decomposition of a symmetric 3x3 tensor by cyclic Jacobi rotations,
//   A = sum_a w_a q_a (x) q_a,  q_a = column a of Q,
// to machine precision in a handful of sweeps. Used by the principal-stretch
// materials (Ogden).
#pragma once

#include <cmath>

#include "base/tensor.hpp"

namespace cmf
{

struct SymmetricEigen3
{
  tensor<double, 3> values;
  tensor<double, 3, 3> vectors; // column a is the unit eigenvector of values(a)
};

inline SymmetricEigen3 EigenSymmetric3(const tensor<double, 3, 3> &A)
{
  double a[3][3], v[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    {
      a[i][j] = 0.5 * (A(i, j) + A(j, i));
      v[i][j] = i == j ? 1.0 : 0.0;
    }
  for (int sweep = 0; sweep < 60; sweep++)
  {
    const double off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    const double diag = a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2];
    if (off == 0.0 || off <= 1e-34 * diag) { break; }
    for (int p = 0; p < 2; p++)
      for (int q = p + 1; q < 3; q++)
      {
        if (a[p][q] == 0.0) { continue; }
        const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                         (std::abs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        const double apq = a[p][q];
        a[p][p] -= t * apq;
        a[q][q] += t * apq;
        a[p][q] = a[q][p] = 0.0;
        const int r = 3 - p - q;
        const double arp = a[r][p], arq = a[r][q];
        a[r][p] = a[p][r] = c * arp - s * arq;
        a[r][q] = a[q][r] = s * arp + c * arq;
        for (int k = 0; k < 3; k++)
        {
          const double vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
  }
  SymmetricEigen3 e;
  for (int i = 0; i < 3; i++)
  {
    e.values(i) = a[i][i];
    for (int j = 0; j < 3; j++) { e.vectors(i, j) = v[i][j]; }
  }
  return e;
}

} // namespace cmf

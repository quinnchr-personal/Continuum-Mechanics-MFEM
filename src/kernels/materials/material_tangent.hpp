// Quadrature-point tangent A_ijkl = dP_ij/dF_kl by forward-mode dual seeding:
// one PK1 evaluation per perturbed component of F. Materials never see duals
// explicitly; they are only templated on the scalar type.
#pragma once

#include "base/dual.hpp"
#include "base/tensor.hpp"

namespace cmf
{

// Seeds the dim x dim in-plane block of F (dim = 3: 9 evaluations, dim = 2:
// 4 evaluations for plane strain, where the out-of-plane entries of F are
// fixed and their tangent entries are left zero).
template <typename Material>
inline tensor<double, 3, 3, 3, 3> MaterialTangent(const Material &material,
                                                  const tensor<double, 3, 3> &F,
                                                  int dim = 3)
{
  tensor<double, 3, 3, 3, 3> A;
  tensor<dual, 3, 3> Fd;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) { Fd(i, j) = dual(F(i, j), 0.0); }
  for (int k = 0; k < dim; k++)
  {
    for (int l = 0; l < dim; l++)
    {
      Fd(k, l).d = 1.0;
      const tensor<dual, 3, 3> P = material.PK1(Fd);
      Fd(k, l).d = 0.0;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) { A(i, j, k, l) = P(i, j).d; }
    }
  }
  return A;
}

template <typename Material>
inline tensor<double, 3, 3> MaterialStress(const Material &material,
                                           const tensor<double, 3, 3> &F)
{
  return material.PK1(F);
}

} // namespace cmf

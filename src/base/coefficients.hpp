// Boundary data built from the YAML boundary conditions: constant vectors,
// or affine fields u(X) = value + gradient X in the reference coordinates
// (the data of homogeneous deformation tests).
#pragma once

#include <memory>
#include <string>

#include "base/config.hpp"
#include "mfem.hpp"

namespace cmf
{

class AffineVectorCoefficient : public mfem::VectorCoefficient
{
public:
  AffineVectorCoefficient(const mfem::Vector &value, const mfem::DenseMatrix &gradient)
    : mfem::VectorCoefficient(value.Size()), value_(value), gradient_(gradient),
      X_(value.Size()) {}

  void Eval(mfem::Vector &v, mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip) override
  {
    T.Transform(ip, X_);
    v.SetSize(vdim);
    gradient_.Mult(X_, v);
    v += value_;
  }

private:
  mfem::Vector value_;
  mfem::DenseMatrix gradient_;
  mfem::Vector X_;
};

// The coefficient of one boundary condition entry, checked against the space
// dimension; `what` names the YAML key in error messages.
inline std::unique_ptr<mfem::VectorCoefficient>
MakeBCCoefficient(const BoundaryCondition &bc, int dim, const std::string &what)
{
  if (int(bc.value.size()) != dim)
  {
    throw ConfigError(what + ".value has " + std::to_string(bc.value.size()) +
                      " components, mesh dimension is " + std::to_string(dim));
  }
  mfem::Vector v(dim);
  for (int i = 0; i < dim; i++) { v(i) = bc.value[i]; }
  if (bc.gradient.empty())
  {
    return std::make_unique<mfem::VectorConstantCoefficient>(v);
  }
  if (int(bc.gradient.size()) != dim)
  {
    throw ConfigError(what + ".gradient has " + std::to_string(bc.gradient.size()) +
                      " rows, mesh dimension is " + std::to_string(dim));
  }
  mfem::DenseMatrix G(dim);
  for (int i = 0; i < dim; i++)
  {
    if (int(bc.gradient[i].size()) != dim)
    {
      throw ConfigError(what + ".gradient row " + std::to_string(i) + " has " +
                        std::to_string(bc.gradient[i].size()) +
                        " entries, mesh dimension is " + std::to_string(dim));
    }
    for (int j = 0; j < dim; j++) { G(i, j) = bc.gradient[i][j]; }
  }
  return std::make_unique<AffineVectorCoefficient>(v, G);
}

} // namespace cmf

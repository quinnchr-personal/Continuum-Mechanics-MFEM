// Boundary data built from the YAML boundary conditions: constant vectors,
// affine fields u(X) = value + gradient X in the reference coordinates (the
// data of homogeneous deformation tests), and expressions f(x, y, z, t) per
// component (base/expression.hpp; t is the coefficient's time).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "base/config.hpp"
#include "base/expression.hpp"
#include "mfem.hpp"

namespace cmf
{

class ExpressionCoefficient : public mfem::Coefficient
{
public:
  explicit ExpressionCoefficient(const Expression &f) : f_(f) {}
  explicit ExpressionCoefficient(const std::string &text) : f_(Expression::Parse(text)) {}

  mfem::real_t Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    T.Transform(ip, X_);
    return f_.Eval(X_(0), X_.Size() > 1 ? X_(1) : 0.0, X_.Size() > 2 ? X_(2) : 0.0, GetTime());
  }

  bool UsesTime() const { return f_.UsesTime(); }
  const Expression &Function() const { return f_; }

private:
  Expression f_;
  mfem::Vector X_;
};

class ExpressionVectorCoefficient : public mfem::VectorCoefficient
{
public:
  explicit ExpressionVectorCoefficient(const std::vector<Expression> &f)
    : mfem::VectorCoefficient(int(f.size())), f_(f) {}
  explicit ExpressionVectorCoefficient(const std::vector<std::string> &text)
    : mfem::VectorCoefficient(int(text.size()))
  {
    for (const std::string &e : text) { f_.push_back(Expression::Parse(e)); }
  }

  void Eval(mfem::Vector &v, mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip) override
  {
    T.Transform(ip, X_);
    const double x = X_(0), y = X_.Size() > 1 ? X_(1) : 0.0, z = X_.Size() > 2 ? X_(2) : 0.0;
    v.SetSize(vdim);
    for (int i = 0; i < vdim; i++) { v(i) = f_[i].Eval(x, y, z, GetTime()); }
  }

  bool UsesTime() const
  {
    for (const Expression &e : f_) { if (e.UsesTime()) { return true; } }
    return false;
  }

private:
  std::vector<Expression> f_;
  mfem::Vector X_;
};

// Whether the data of a YAML entry depends on t through an expression.
inline bool ExpressionsUseTime(const std::vector<std::string> &expression)
{
  for (const std::string &e : expression)
  {
    if (Expression::Parse(e).UsesTime()) { return true; }
  }
  return false;
}

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

// The vector coefficient of one boundary condition entry, checked against
// the space dimension; `what` names the YAML key in error messages.
inline std::unique_ptr<mfem::VectorCoefficient>
MakeBCCoefficient(const BoundaryCondition &bc, int dim, const std::string &what)
{
  if (!bc.expression.empty())
  {
    if (int(bc.expression.size()) != dim)
    {
      throw ConfigError(what + ".expression has " + std::to_string(bc.expression.size()) +
                        " components, mesh dimension is " + std::to_string(dim));
    }
    std::vector<Expression> f;
    for (const std::string &e : bc.expression) { f.push_back(Expression::Parse(e)); }
    return std::make_unique<ExpressionVectorCoefficient>(f);
  }
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

// The scalar coefficient of a pressure entry (type pressure or
// follower_pressure): a constant or an expression.
inline std::unique_ptr<mfem::Coefficient>
MakeScalarBCCoefficient(const BoundaryCondition &bc, const std::string &what)
{
  if (!bc.expression.empty())
  {
    if (bc.expression.size() != 1)
    {
      throw ConfigError(what + ".expression must be a single string for a pressure");
    }
    return std::make_unique<ExpressionCoefficient>(Expression::Parse(bc.expression[0]));
  }
  if (bc.value.size() != 1)
  {
    throw ConfigError(what + ".value must be a single number for a pressure");
  }
  return std::make_unique<mfem::ConstantCoefficient>(bc.value[0]);
}

} // namespace cmf

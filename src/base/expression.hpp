// Scalar expressions f(x, y, z, t) for YAML boundary data: parsed once into
// a flat postfix program, evaluated per point without allocation.
//
// Grammar: numbers, the variables x y z t (z is 0 in 2D) and pi, the binary
// operators + - * / ^ (^ binds tightest and associates to the right, unary
// minus binds looser than ^ so -x^2 = -(x^2)), parentheses, the functions
// sin cos tan exp log sqrt abs pow(a, b) min(a, b) max(a, b), and
// if(cond, a, b) where the comparisons < <= > >= == != evaluate to 1 or 0.
#pragma once

#include <string>
#include <vector>

namespace cmf
{

class Expression
{
public:
  Expression() = default;

  // Throws ConfigError naming the column on a syntax error.
  static Expression Parse(const std::string &text);

  double Eval(double x, double y, double z, double t) const;
  // The program refers to t (so its value changes with the pseudo-time).
  bool UsesTime() const { return uses_time_; }
  const std::string &Text() const { return text_; }

private:
  enum class Op : unsigned char
  {
    Const, X, Y, Z, T, Neg, Add, Sub, Mul, Div, Pow,
    Lt, Le, Gt, Ge, Eq, Ne,
    Sin, Cos, Tan, Exp, Log, Sqrt, Abs, Min, Max, If
  };
  struct Instruction
  {
    Op op;
    double value; // Const only
  };

  class Parser;
  std::vector<Instruction> program_;
  int stack_size_ = 0;
  bool uses_time_ = false;
  std::string text_;
};

} // namespace cmf

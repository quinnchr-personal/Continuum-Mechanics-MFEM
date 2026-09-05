// Forward-mode dual number: value plus one directional derivative. Arithmetic
// and the elementary functions materials need; tensor<dual, ...> works through
// the templates in tensor.hpp.
#pragma once

#include <cmath>
#include <iosfwd>

namespace cmf
{

struct dual
{
  double v = 0.0; // value
  double d = 0.0; // derivative

  constexpr dual() = default;
  constexpr dual(double value) : v(value), d(0.0) {}
  constexpr dual(double value, double deriv) : v(value), d(deriv) {}
};

// Pull the double overloads into this namespace so unqualified calls inside
// templates resolve to std:: for double and to the overloads below for dual.
using std::abs;
using std::cos;
using std::exp;
using std::log;
using std::pow;
using std::sin;
using std::sqrt;

constexpr dual operator+(dual a, dual b) { return {a.v + b.v, a.d + b.d}; }
constexpr dual operator-(dual a, dual b) { return {a.v - b.v, a.d - b.d}; }
constexpr dual operator*(dual a, dual b)
{
  return {a.v * b.v, a.d * b.v + a.v * b.d};
}
constexpr dual operator/(dual a, dual b)
{
  return {a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v)};
}
constexpr dual operator-(dual a) { return {-a.v, -a.d}; }
constexpr dual operator+(dual a) { return a; }

// Mixed double/dual overloads keep the arithmetic exact and unambiguous.
constexpr dual operator+(dual a, double b) { return {a.v + b, a.d}; }
constexpr dual operator+(double a, dual b) { return {a + b.v, b.d}; }
constexpr dual operator-(dual a, double b) { return {a.v - b, a.d}; }
constexpr dual operator-(double a, dual b) { return {a - b.v, -b.d}; }
constexpr dual operator*(dual a, double b) { return {a.v * b, a.d * b}; }
constexpr dual operator*(double a, dual b) { return {a * b.v, a * b.d}; }
constexpr dual operator/(dual a, double b) { return {a.v / b, a.d / b}; }
constexpr dual operator/(double a, dual b)
{
  return {a / b.v, -a * b.d / (b.v * b.v)};
}

inline dual &operator+=(dual &a, dual b) { a = a + b; return a; }
inline dual &operator-=(dual &a, dual b) { a = a - b; return a; }
inline dual &operator*=(dual &a, dual b) { a = a * b; return a; }
inline dual &operator/=(dual &a, dual b) { a = a / b; return a; }

constexpr bool operator==(dual a, dual b) { return a.v == b.v; }
constexpr bool operator!=(dual a, dual b) { return a.v != b.v; }
constexpr bool operator<(dual a, dual b) { return a.v < b.v; }
constexpr bool operator<=(dual a, dual b) { return a.v <= b.v; }
constexpr bool operator>(dual a, dual b) { return a.v > b.v; }
constexpr bool operator>=(dual a, dual b) { return a.v >= b.v; }

inline dual log(dual a) { return {std::log(a.v), a.d / a.v}; }
inline dual exp(dual a)
{
  const double e = std::exp(a.v);
  return {e, a.d * e};
}
inline dual sqrt(dual a)
{
  const double s = std::sqrt(a.v);
  return {s, 0.5 * a.d / s};
}
inline dual pow(dual a, double p)
{
  return {std::pow(a.v, p), p * std::pow(a.v, p - 1.0) * a.d};
}
inline dual pow(dual a, dual b)
{
  const double f = std::pow(a.v, b.v);
  return {f, f * (b.d * std::log(a.v) + b.v * a.d / a.v)};
}
inline dual abs(dual a) { return a.v < 0.0 ? -a : a; }
inline dual sin(dual a) { return {std::sin(a.v), a.d * std::cos(a.v)}; }
inline dual cos(dual a) { return {std::cos(a.v), -a.d * std::sin(a.v)}; }

// Value and derivative extraction that also work on plain doubles, so code
// templated on the scalar type can be written once.
constexpr double value(double x) { return x; }
constexpr double value(dual x) { return x.v; }
constexpr double derivative(double) { return 0.0; }
constexpr double derivative(dual x) { return x.d; }

} // namespace cmf

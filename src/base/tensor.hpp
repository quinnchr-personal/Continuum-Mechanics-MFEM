// Fixed-size stack tensors for quadrature-point code. Templated on the scalar
// type so dual numbers flow through unchanged; no virtual dispatch, no heap.
// Every function here is a plain inline callable that could later be marked
// MFEM_HOST_DEVICE without change.
#pragma once

#include <cmath>
#include <type_traits>
#include <utility>

namespace cmf
{

template <typename T, int... n> struct tensor;

template <typename T> struct is_tensor : std::false_type {};
template <typename T, int... n>
struct is_tensor<tensor<T, n...>> : std::true_type {};

template <typename S, typename T>
using product_t = decltype(std::declval<S>() * std::declval<T>());
template <typename S, typename T>
using sum_t = decltype(std::declval<S>() + std::declval<T>());

// Rank 1
template <typename T, int n>
struct tensor<T, n>
{
  T v[n] = {};
  static constexpr int rank = 1;
  T &operator()(int i) { return v[i]; }
  const T &operator()(int i) const { return v[i]; }
  T &operator[](int i) { return v[i]; }
  const T &operator[](int i) const { return v[i]; }
};

// Rank 2
template <typename T, int m, int n>
struct tensor<T, m, n>
{
  T v[m][n] = {};
  static constexpr int rank = 2;
  T &operator()(int i, int j) { return v[i][j]; }
  const T &operator()(int i, int j) const { return v[i][j]; }
};

// Rank 4 (material tangents A_ijkl)
template <typename T, int m, int n, int p, int q>
struct tensor<T, m, n, p, q>
{
  T v[m][n][p][q] = {};
  static constexpr int rank = 4;
  T &operator()(int i, int j, int k, int l) { return v[i][j][k][l]; }
  const T &operator()(int i, int j, int k, int l) const { return v[i][j][k][l]; }
};

// ---------------------------------------------------------------- identity

template <int n, typename T = double>
inline tensor<T, n, n> I()
{
  tensor<T, n, n> id;
  for (int i = 0; i < n; i++) { id(i, i) = T(1); }
  return id;
}

// ---------------------------------------------------------- rank-1 algebra

template <typename S, typename T, int n>
inline tensor<sum_t<S, T>, n> operator+(const tensor<S, n> &a, const tensor<T, n> &b)
{
  tensor<sum_t<S, T>, n> c;
  for (int i = 0; i < n; i++) { c(i) = a(i) + b(i); }
  return c;
}

template <typename S, typename T, int n>
inline tensor<sum_t<S, T>, n> operator-(const tensor<S, n> &a, const tensor<T, n> &b)
{
  tensor<sum_t<S, T>, n> c;
  for (int i = 0; i < n; i++) { c(i) = a(i) - b(i); }
  return c;
}

template <typename T, int n>
inline tensor<T, n> operator-(const tensor<T, n> &a)
{
  tensor<T, n> c;
  for (int i = 0; i < n; i++) { c(i) = -a(i); }
  return c;
}

template <typename S, typename T, int n,
          typename = std::enable_if_t<!is_tensor<S>::value>>
inline tensor<product_t<S, T>, n> operator*(const S &s, const tensor<T, n> &a)
{
  tensor<product_t<S, T>, n> c;
  for (int i = 0; i < n; i++) { c(i) = s * a(i); }
  return c;
}

template <typename S, typename T, int n,
          typename = std::enable_if_t<!is_tensor<S>::value>>
inline tensor<product_t<T, S>, n> operator*(const tensor<T, n> &a, const S &s)
{
  tensor<product_t<T, S>, n> c;
  for (int i = 0; i < n; i++) { c(i) = a(i) * s; }
  return c;
}

template <typename S, typename T, int n,
          typename = std::enable_if_t<!is_tensor<S>::value>>
inline tensor<product_t<T, S>, n> operator/(const tensor<T, n> &a, const S &s)
{
  tensor<product_t<T, S>, n> c;
  for (int i = 0; i < n; i++) { c(i) = a(i) / s; }
  return c;
}

template <typename T, int n>
inline tensor<T, n> &operator+=(tensor<T, n> &a, const tensor<T, n> &b)
{
  for (int i = 0; i < n; i++) { a(i) += b(i); }
  return a;
}

template <typename T, int n>
inline tensor<T, n> &operator-=(tensor<T, n> &a, const tensor<T, n> &b)
{
  for (int i = 0; i < n; i++) { a(i) -= b(i); }
  return a;
}

// ---------------------------------------------------------- rank-2 algebra

template <typename S, typename T, int m, int n>
inline tensor<sum_t<S, T>, m, n> operator+(const tensor<S, m, n> &a,
                                           const tensor<T, m, n> &b)
{
  tensor<sum_t<S, T>, m, n> c;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) { c(i, j) = a(i, j) + b(i, j); }
  return c;
}

template <typename S, typename T, int m, int n>
inline tensor<sum_t<S, T>, m, n> operator-(const tensor<S, m, n> &a,
                                           const tensor<T, m, n> &b)
{
  tensor<sum_t<S, T>, m, n> c;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) { c(i, j) = a(i, j) - b(i, j); }
  return c;
}

template <typename T, int m, int n>
inline tensor<T, m, n> operator-(const tensor<T, m, n> &a)
{
  tensor<T, m, n> c;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) { c(i, j) = -a(i, j); }
  return c;
}

template <typename S, typename T, int m, int n,
          typename = std::enable_if_t<!is_tensor<S>::value>>
inline tensor<product_t<S, T>, m, n> operator*(const S &s, const tensor<T, m, n> &a)
{
  tensor<product_t<S, T>, m, n> c;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) { c(i, j) = s * a(i, j); }
  return c;
}

template <typename S, typename T, int m, int n,
          typename = std::enable_if_t<!is_tensor<S>::value>>
inline tensor<product_t<T, S>, m, n> operator*(const tensor<T, m, n> &a, const S &s)
{
  tensor<product_t<T, S>, m, n> c;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) { c(i, j) = a(i, j) * s; }
  return c;
}

template <typename S, typename T, int m, int n,
          typename = std::enable_if_t<!is_tensor<S>::value>>
inline tensor<product_t<T, S>, m, n> operator/(const tensor<T, m, n> &a, const S &s)
{
  tensor<product_t<T, S>, m, n> c;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) { c(i, j) = a(i, j) / s; }
  return c;
}

template <typename T, int m, int n>
inline tensor<T, m, n> &operator+=(tensor<T, m, n> &a, const tensor<T, m, n> &b)
{
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) { a(i, j) += b(i, j); }
  return a;
}

template <typename T, int m, int n>
inline tensor<T, m, n> &operator-=(tensor<T, m, n> &a, const tensor<T, m, n> &b)
{
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) { a(i, j) -= b(i, j); }
  return a;
}

// ------------------------------------------------------------ contractions

// a_i b_i
template <typename S, typename T, int n>
inline product_t<S, T> dot(const tensor<S, n> &a, const tensor<T, n> &b)
{
  product_t<S, T> s{};
  for (int i = 0; i < n; i++) { s += a(i) * b(i); }
  return s;
}

// A_ij b_j
template <typename S, typename T, int m, int n>
inline tensor<product_t<S, T>, m> dot(const tensor<S, m, n> &A, const tensor<T, n> &b)
{
  tensor<product_t<S, T>, m> c;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) { c(i) += A(i, j) * b(j); }
  return c;
}

// a_i A_ij
template <typename S, typename T, int m, int n>
inline tensor<product_t<S, T>, n> dot(const tensor<S, m> &a, const tensor<T, m, n> &A)
{
  tensor<product_t<S, T>, n> c;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) { c(j) += a(i) * A(i, j); }
  return c;
}

// A_ik B_kj
template <typename S, typename T, int m, int n, int p>
inline tensor<product_t<S, T>, m, p> dot(const tensor<S, m, n> &A,
                                         const tensor<T, n, p> &B)
{
  tensor<product_t<S, T>, m, p> C;
  for (int i = 0; i < m; i++)
    for (int k = 0; k < n; k++)
      for (int j = 0; j < p; j++) { C(i, j) += A(i, k) * B(k, j); }
  return C;
}

template <typename S, typename T, int m, int n, int p>
inline tensor<product_t<S, T>, m, p> operator*(const tensor<S, m, n> &A,
                                               const tensor<T, n, p> &B)
{
  return dot(A, B);
}

template <typename S, typename T, int m, int n>
inline tensor<product_t<S, T>, m> operator*(const tensor<S, m, n> &A,
                                            const tensor<T, n> &b)
{
  return dot(A, b);
}

// A_ij B_ij
template <typename S, typename T, int m, int n>
inline product_t<S, T> ddot(const tensor<S, m, n> &A, const tensor<T, m, n> &B)
{
  product_t<S, T> s{};
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) { s += A(i, j) * B(i, j); }
  return s;
}

// A_ijkl B_kl
template <typename S, typename T, int m, int n, int p, int q>
inline tensor<product_t<S, T>, m, n> ddot(const tensor<S, m, n, p, q> &A,
                                          const tensor<T, p, q> &B)
{
  tensor<product_t<S, T>, m, n> C;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < p; k++)
        for (int l = 0; l < q; l++) { C(i, j) += A(i, j, k, l) * B(k, l); }
  return C;
}

// B_ij A_ijkl
template <typename S, typename T, int m, int n, int p, int q>
inline tensor<product_t<S, T>, p, q> ddot(const tensor<S, m, n> &B,
                                          const tensor<T, m, n, p, q> &A)
{
  tensor<product_t<S, T>, p, q> C;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < p; k++)
        for (int l = 0; l < q; l++) { C(k, l) += B(i, j) * A(i, j, k, l); }
  return C;
}

// a_i b_j
template <typename S, typename T, int m, int n>
inline tensor<product_t<S, T>, m, n> outer(const tensor<S, m> &a, const tensor<T, n> &b)
{
  tensor<product_t<S, T>, m, n> C;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) { C(i, j) = a(i) * b(j); }
  return C;
}

// A_ij B_kl
template <typename S, typename T, int m, int n, int p, int q>
inline tensor<product_t<S, T>, m, n, p, q> outer(const tensor<S, m, n> &A,
                                                 const tensor<T, p, q> &B)
{
  tensor<product_t<S, T>, m, n, p, q> C;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < p; k++)
        for (int l = 0; l < q; l++) { C(i, j, k, l) = A(i, j) * B(k, l); }
  return C;
}

// ------------------------------------------------------ rank-2 operations

template <typename T, int m, int n>
inline tensor<T, n, m> transpose(const tensor<T, m, n> &A)
{
  tensor<T, n, m> B;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) { B(j, i) = A(i, j); }
  return B;
}

template <typename T, int n>
inline T tr(const tensor<T, n, n> &A)
{
  T s{};
  for (int i = 0; i < n; i++) { s += A(i, i); }
  return s;
}

template <typename T, int n>
inline tensor<T, n, n> sym(const tensor<T, n, n> &A)
{
  tensor<T, n, n> S;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) { S(i, j) = 0.5 * (A(i, j) + A(j, i)); }
  return S;
}

template <typename T, int n>
inline tensor<T, n, n> dev(const tensor<T, n, n> &A)
{
  tensor<T, n, n> D = A;
  const T mean = tr(A) / double(n);
  for (int i = 0; i < n; i++) { D(i, i) -= mean; }
  return D;
}

template <typename T, int m, int n>
inline T norm_squared(const tensor<T, m, n> &A) { return ddot(A, A); }

template <typename T, int n>
inline T norm_squared(const tensor<T, n> &a) { return dot(a, a); }

template <typename T, int n>
inline T det(const tensor<T, n, n> &A)
{
  static_assert(n >= 1 && n <= 3, "det: only 1x1, 2x2, 3x3 are supported");
  if constexpr (n == 1)
  {
    return A(0, 0);
  }
  else if constexpr (n == 2)
  {
    return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
  }
  else
  {
    return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1))
         - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0))
         + A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
  }
}

template <typename T, int n>
inline tensor<T, n, n> inv(const tensor<T, n, n> &A)
{
  static_assert(n >= 1 && n <= 3, "inv: only 1x1, 2x2, 3x3 are supported");
  tensor<T, n, n> B;
  const T d = det(A);
  if constexpr (n == 1)
  {
    B(0, 0) = 1.0 / d;
  }
  else if constexpr (n == 2)
  {
    B(0, 0) = A(1, 1) / d;
    B(0, 1) = -A(0, 1) / d;
    B(1, 0) = -A(1, 0) / d;
    B(1, 1) = A(0, 0) / d;
  }
  else
  {
    B(0, 0) = (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) / d;
    B(0, 1) = (A(0, 2) * A(2, 1) - A(0, 1) * A(2, 2)) / d;
    B(0, 2) = (A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1)) / d;
    B(1, 0) = (A(1, 2) * A(2, 0) - A(1, 0) * A(2, 2)) / d;
    B(1, 1) = (A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0)) / d;
    B(1, 2) = (A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2)) / d;
    B(2, 0) = (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0)) / d;
    B(2, 1) = (A(0, 1) * A(2, 0) - A(0, 0) * A(2, 1)) / d;
    B(2, 2) = (A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0)) / d;
  }
  return B;
}

} // namespace cmf

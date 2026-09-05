// Minimal check macros shared by the test executables.
#pragma once

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

namespace cmf_test
{

inline int &Failures()
{
  static int failures = 0;
  return failures;
}

inline int &Checks()
{
  static int checks = 0;
  return checks;
}

inline void Check(bool ok, const std::string &what, const char *file, int line)
{
  Checks()++;
  if (!ok)
  {
    Failures()++;
    std::cout << "  FAIL " << file << ":" << line << ": " << what << std::endl;
  }
}

inline void CheckClose(double got, double want, double tol,
                       const std::string &what, const char *file, int line)
{
  const double err = std::abs(got - want);
  const bool ok = err <= tol;
  Checks()++;
  if (!ok)
  {
    Failures()++;
    std::printf("  FAIL %s:%d: %s: got %.16e want %.16e (|diff| %.3e > %.3e)\n",
                file, line, what.c_str(), got, want, err, tol);
  }
}

inline int Report(const char *name)
{
  std::cout << name << ": " << Checks() - Failures() << "/" << Checks()
            << " checks passed" << std::endl;
  return Failures() == 0 ? 0 : 1;
}

} // namespace cmf_test

#define CHECK(cond) cmf_test::Check((cond), #cond, __FILE__, __LINE__)
#define CHECK_MSG(cond, msg) cmf_test::Check((cond), msg, __FILE__, __LINE__)
#define CHECK_CLOSE(got, want, tol) \
  cmf_test::CheckClose((got), (want), (tol), #got, __FILE__, __LINE__)
#define CHECK_THROWS(expr, ExceptionType, needle)                              \
  do                                                                           \
  {                                                                            \
    bool threw_ = false;                                                       \
    std::string msg_;                                                          \
    try { expr; }                                                              \
    catch (const ExceptionType &e) { threw_ = true; msg_ = e.what(); }         \
    catch (const std::exception &e) { msg_ = std::string("other: ") + e.what(); } \
    CHECK_MSG(threw_, std::string(#expr) + " should throw " #ExceptionType +   \
              " (got: " + msg_ + ")");                                         \
    CHECK_MSG(msg_.find(needle) != std::string::npos,                          \
              std::string("message '") + msg_ + "' should mention '" + needle + "'"); \
  } while (0)

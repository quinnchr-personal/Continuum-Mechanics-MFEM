#include "base/expression.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>

#include "base/config.hpp"

namespace cmf
{

// Recursive descent over the grammar of expression.hpp, emitting postfix
// instructions and tracking the evaluation stack depth.
class Expression::Parser
{
public:
  Parser(const std::string &text, Expression &out) : s_(text), out_(out) {}

  void Run()
  {
    SkipSpace();
    if (Eof()) { Fail("empty expression"); }
    ParseComparison();
    SkipSpace();
    if (!Eof()) { Fail("unexpected '" + std::string(1, s_[pos_]) + "'"); }
    if (depth_ != 1) { Fail("internal error: stack depth " + std::to_string(depth_)); }
  }

private:
  bool Eof() const { return pos_ >= s_.size(); }
  void SkipSpace() { while (!Eof() && std::isspace(static_cast<unsigned char>(s_[pos_]))) { pos_++; } }
  char Peek() { SkipSpace(); return Eof() ? '\0' : s_[pos_]; }
  bool Accept(const char *token)
  {
    SkipSpace();
    const std::size_t n = std::char_traits<char>::length(token);
    if (s_.compare(pos_, n, token) == 0) { pos_ += n; return true; }
    return false;
  }

  [[noreturn]] void Fail(const std::string &what) const
  {
    throw ConfigError("expression '" + s_ + "': " + what + " at column " +
                      std::to_string(pos_ + 1));
  }

  void Emit(Op op, double value = 0.0, int pops = 0)
  {
    depth_ -= pops;
    depth_ += 1;
    if (depth_ > out_.stack_size_) { out_.stack_size_ = depth_; }
    out_.program_.push_back({op, value});
  }

  void ParseComparison()
  {
    ParseAdditive();
    SkipSpace();
    struct { const char *tok; Op op; } ops[] = {
      {"<=", Op::Le}, {">=", Op::Ge}, {"==", Op::Eq}, {"!=", Op::Ne}, {"<", Op::Lt}, {">", Op::Gt}};
    for (const auto &o : ops)
    {
      if (Accept(o.tok))
      {
        ParseAdditive();
        Emit(o.op, 0.0, 2);
        SkipSpace();
        if (Peek() == '<' || Peek() == '>' || Peek() == '=' || Peek() == '!')
        {
          Fail("comparisons do not chain; use if(...) or parentheses");
        }
        return;
      }
    }
  }

  void ParseAdditive()
  {
    ParseTerm();
    for (;;)
    {
      if (Accept("+")) { ParseTerm(); Emit(Op::Add, 0.0, 2); }
      else if (Accept("-")) { ParseTerm(); Emit(Op::Sub, 0.0, 2); }
      else { return; }
    }
  }

  void ParseTerm()
  {
    ParseUnary();
    for (;;)
    {
      if (Accept("*")) { ParseUnary(); Emit(Op::Mul, 0.0, 2); }
      else if (Accept("/")) { ParseUnary(); Emit(Op::Div, 0.0, 2); }
      else { return; }
    }
  }

  void ParseUnary()
  {
    if (Accept("-")) { ParseUnary(); Emit(Op::Neg, 0.0, 1); return; }
    if (Accept("+")) { ParseUnary(); return; }
    ParsePower();
  }

  void ParsePower()
  {
    ParsePrimary();
    if (Accept("^"))
    {
      ParseUnary(); // right associative; the exponent may carry a sign
      Emit(Op::Pow, 0.0, 2);
    }
  }

  void ParseArguments(int count, const std::string &name)
  {
    if (!Accept("(")) { Fail("expected '(' after '" + name + "'"); }
    for (int i = 0; i < count; i++)
    {
      if (i > 0 && !Accept(",")) { Fail("'" + name + "' takes " + std::to_string(count) + " arguments"); }
      ParseComparison();
    }
    if (!Accept(")")) { Fail("expected ')' closing '" + name + "'"); }
  }

  void ParsePrimary()
  {
    SkipSpace();
    if (Eof()) { Fail("unexpected end of expression"); }
    const char c = s_[pos_];
    if (std::isdigit(static_cast<unsigned char>(c)) || c == '.')
    {
      const char *begin = s_.c_str() + pos_;
      char *end = nullptr;
      const double v = std::strtod(begin, &end);
      if (end == begin) { Fail("bad number"); }
      pos_ += std::size_t(end - begin);
      Emit(Op::Const, v);
      return;
    }
    if (c == '(')
    {
      pos_++;
      ParseComparison();
      if (!Accept(")")) { Fail("expected ')'"); }
      return;
    }
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_')
    {
      const std::size_t start = pos_;
      while (!Eof() && (std::isalnum(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '_')) { pos_++; }
      const std::string name = s_.substr(start, pos_ - start);
      if (name == "x") { Emit(Op::X); return; }
      if (name == "y") { Emit(Op::Y); return; }
      if (name == "z") { Emit(Op::Z); return; }
      if (name == "t") { Emit(Op::T); out_.uses_time_ = true; return; }
      if (name == "pi") { Emit(Op::Const, M_PI); return; }
      struct { const char *n; Op op; int args; } fns[] = {
        {"sin", Op::Sin, 1}, {"cos", Op::Cos, 1}, {"tan", Op::Tan, 1}, {"exp", Op::Exp, 1},
        {"log", Op::Log, 1}, {"sqrt", Op::Sqrt, 1}, {"abs", Op::Abs, 1}, {"pow", Op::Pow, 2},
        {"min", Op::Min, 2}, {"max", Op::Max, 2}, {"if", Op::If, 3}};
      for (const auto &f : fns)
      {
        if (name == f.n)
        {
          ParseArguments(f.args, name);
          Emit(f.op, 0.0, f.args);
          return;
        }
      }
      pos_ = start;
      Fail("unknown identifier '" + name + "'");
    }
    Fail(std::string("unexpected '") + c + "'");
  }

  const std::string &s_;
  Expression &out_;
  std::size_t pos_ = 0;
  int depth_ = 0;
};

Expression Expression::Parse(const std::string &text)
{
  Expression e;
  e.text_ = text;
  Parser(text, e).Run();
  if (e.stack_size_ > 64)
  {
    throw ConfigError("expression '" + text + "' is too deeply nested");
  }
  return e;
}

double Expression::Eval(double x, double y, double z, double t) const
{
  constexpr int kMaxStack = 64;
  double stack[kMaxStack];
  int n = 0;
  for (const Instruction &ins : program_)
  {
    switch (ins.op)
    {
      case Op::Const: stack[n++] = ins.value; break;
      case Op::X: stack[n++] = x; break;
      case Op::Y: stack[n++] = y; break;
      case Op::Z: stack[n++] = z; break;
      case Op::T: stack[n++] = t; break;
      case Op::Neg: stack[n - 1] = -stack[n - 1]; break;
      case Op::Add: n--; stack[n - 1] += stack[n]; break;
      case Op::Sub: n--; stack[n - 1] -= stack[n]; break;
      case Op::Mul: n--; stack[n - 1] *= stack[n]; break;
      case Op::Div: n--; stack[n - 1] /= stack[n]; break;
      case Op::Pow: n--; stack[n - 1] = std::pow(stack[n - 1], stack[n]); break;
      case Op::Lt: n--; stack[n - 1] = stack[n - 1] < stack[n] ? 1.0 : 0.0; break;
      case Op::Le: n--; stack[n - 1] = stack[n - 1] <= stack[n] ? 1.0 : 0.0; break;
      case Op::Gt: n--; stack[n - 1] = stack[n - 1] > stack[n] ? 1.0 : 0.0; break;
      case Op::Ge: n--; stack[n - 1] = stack[n - 1] >= stack[n] ? 1.0 : 0.0; break;
      case Op::Eq: n--; stack[n - 1] = stack[n - 1] == stack[n] ? 1.0 : 0.0; break;
      case Op::Ne: n--; stack[n - 1] = stack[n - 1] != stack[n] ? 1.0 : 0.0; break;
      case Op::Sin: stack[n - 1] = std::sin(stack[n - 1]); break;
      case Op::Cos: stack[n - 1] = std::cos(stack[n - 1]); break;
      case Op::Tan: stack[n - 1] = std::tan(stack[n - 1]); break;
      case Op::Exp: stack[n - 1] = std::exp(stack[n - 1]); break;
      case Op::Log: stack[n - 1] = std::log(stack[n - 1]); break;
      case Op::Sqrt: stack[n - 1] = std::sqrt(stack[n - 1]); break;
      case Op::Abs: stack[n - 1] = std::abs(stack[n - 1]); break;
      case Op::Min: n--; stack[n - 1] = std::min(stack[n - 1], stack[n]); break;
      case Op::Max: n--; stack[n - 1] = std::max(stack[n - 1], stack[n]); break;
      case Op::If: n -= 2; stack[n - 1] = stack[n - 1] != 0.0 ? stack[n] : stack[n + 1]; break;
    }
  }
  return n == 1 ? stack[0] : 0.0;
}

} // namespace cmf

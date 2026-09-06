// S1 gate: tensor algebra vs hand values, dual derivatives vs central finite
// differences, and YAML validation errors that name the bad key.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>

#include "base/config.hpp"
#include "base/dual.hpp"
#include "base/tensor.hpp"
#include "test_util.hpp"

using cmf::dual;
using cmf::tensor;

namespace
{

// Richardson-extrapolated central difference: O(h^4) truncation error.
double CentralDifference(const std::function<double(double)> &f, double x, double h)
{
  const double d1 = (f(x + h) - f(x - h)) / (2.0 * h);
  const double d2 = (f(x + 0.5 * h) - f(x - 0.5 * h)) / h;
  return (4.0 * d2 - d1) / 3.0;
}

void TestTensor2x2()
{
  tensor<double, 2, 2> A;
  A(0, 0) = 4.0; A(0, 1) = 7.0;
  A(1, 0) = 2.0; A(1, 1) = 6.0;
  CHECK_CLOSE(det(A), 10.0, 1e-14);
  const tensor<double, 2, 2> B = inv(A);
  CHECK_CLOSE(B(0, 0), 0.6, 1e-15);
  CHECK_CLOSE(B(0, 1), -0.7, 1e-15);
  CHECK_CLOSE(B(1, 0), -0.2, 1e-15);
  CHECK_CLOSE(B(1, 1), 0.4, 1e-15);
  const tensor<double, 2, 2> AB = A * B;
  const tensor<double, 2, 2> Id = cmf::I<2>();
  CHECK_CLOSE(cmf::norm_squared(AB - Id), 0.0, 1e-28);
  CHECK_CLOSE(cmf::ddot(A, A), 16.0 + 49.0 + 4.0 + 36.0, 1e-13);
  CHECK_CLOSE(cmf::tr(A), 10.0, 1e-15);
  const tensor<double, 2, 2> At = cmf::transpose(A);
  CHECK_CLOSE(At(0, 1), 2.0, 0.0);
  CHECK_CLOSE(At(1, 0), 7.0, 0.0);

  tensor<double, 2> u, w;
  u(0) = 1.0; u(1) = 2.0;
  w(0) = -3.0; w(1) = 0.5;
  const tensor<double, 2> Au = A * u;                  // (4+14, 2+12)
  CHECK_CLOSE(Au(0), 18.0, 0.0);
  CHECK_CLOSE(Au(1), 14.0, 0.0);
  CHECK_CLOSE(cmf::dot(u, w), -2.0, 0.0);
  const tensor<double, 2, 2> uw = cmf::outer(u, w);
  CHECK_CLOSE(uw(1, 0), -6.0, 0.0);
  CHECK_CLOSE(uw(0, 1), 0.5, 0.0);
  const tensor<double, 2> uA = cmf::dot(u, A);          // (4+4, 7+12)
  CHECK_CLOSE(uA(0), 8.0, 0.0);
  CHECK_CLOSE(uA(1), 19.0, 0.0);
  const tensor<double, 2, 2> S = cmf::sym(A);
  CHECK_CLOSE(S(0, 1), 4.5, 0.0);
  const tensor<double, 2, 2> D = cmf::dev(A);
  CHECK_CLOSE(cmf::tr(D), 0.0, 1e-15);
  const tensor<double, 2, 2> C = 2.0 * A - A * 0.5 + A / 4.0;
  CHECK_CLOSE(C(0, 1), 7.0 * 1.75, 1e-14);
}

void TestTensor3x3()
{
  tensor<double, 3, 3> A;
  A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
  A(1, 0) = 0.0; A(1, 1) = 1.0; A(1, 2) = 4.0;
  A(2, 0) = 5.0; A(2, 1) = 6.0; A(2, 2) = 0.0;
  CHECK_CLOSE(det(A), 1.0, 1e-13);
  const tensor<double, 3, 3> B = inv(A);
  const double want[3][3] = {{-24.0, 18.0, 5.0}, {20.0, -15.0, -4.0}, {-5.0, 4.0, 1.0}};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    {
      CHECK_CLOSE(B(i, j), want[i][j], 1e-12);
    }
  const tensor<double, 3, 3> AB = A * B;
  CHECK_CLOSE(cmf::norm_squared(AB - cmf::I<3>()), 0.0, 1e-24);
  CHECK_CLOSE(cmf::ddot(A, B), -24.0 + 36.0 + 15.0 - 15.0 - 16.0 - 25.0 + 24.0, 1e-12);
  CHECK_CLOSE(det(cmf::transpose(A)), 1.0, 1e-13);
  CHECK_CLOSE(det(cmf::I<3>()), 1.0, 0.0);

  // Rank-4 contractions: A_ijkl = delta_ik delta_jl gives ddot(A4, B) = B.
  tensor<double, 3, 3, 3, 3> Id4;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) { Id4(i, j, i, j) = 1.0; }
  const tensor<double, 3, 3> AA = cmf::ddot(Id4, A);
  CHECK_CLOSE(cmf::norm_squared(AA - A), 0.0, 0.0);
  const tensor<double, 3, 3, 3, 3> O = cmf::outer(A, B);
  CHECK_CLOSE(O(2, 1, 0, 2), A(2, 1) * B(0, 2), 0.0);
}

void TestDualScalar()
{
  auto f = [](auto x)
  {
    using cmf::exp; using cmf::log; using cmf::pow; using cmf::sqrt;
    return log(x * x + 1.0) * sqrt(x) + pow(x, 2.5) / (1.0 + x)
           + exp(-x) * x * x * x - 3.0 / x;
  };
  const double xs[] = {0.3, 0.9, 1.7, 2.6};
  for (double x : xs)
  {
    const dual y = f(dual(x, 1.0));
    const double fd = CentralDifference([&](double t) { return f(t); }, x, 1e-3);
    CHECK_CLOSE(y.v, f(x), 0.0);
    const double rel = std::abs(y.d - fd) / std::abs(fd);
    CHECK_MSG(rel <= 1e-10, "dual derivative at x=" + std::to_string(x) +
              " relative error " + std::to_string(rel));
  }
  // Mixed arithmetic and comparisons.
  const dual a(2.0, 1.0);
  CHECK_CLOSE((3.0 / a).d, -0.75, 1e-15);
  CHECK_CLOSE((a * a * a).d, 12.0, 1e-15);
  CHECK(a > 1.0 && a < 3.0 && a == dual(2.0));
  CHECK_CLOSE(cmf::value(a), 2.0, 0.0);
  CHECK_CLOSE(cmf::derivative(a), 1.0, 0.0);
  CHECK_CLOSE(cmf::derivative(2.0), 0.0, 0.0);
}

// F(t) = I + t H; derivatives of det, inv, and a composite through the tensor
// templates with dual entries, vs finite differences.
void TestDualTensor()
{
  tensor<double, 3, 3> H;
  H(0, 0) = 0.2; H(0, 1) = -0.3; H(0, 2) = 0.1;
  H(1, 0) = 0.5; H(1, 1) = 0.4; H(1, 2) = -0.2;
  H(2, 0) = -0.1; H(2, 1) = 0.3; H(2, 2) = 0.6;
  auto F_of = [&](auto t)
  {
    using T = decltype(t);
    tensor<T, 3, 3> F = cmf::I<3>() + t * H;
    return F;
  };
  auto g = [&](auto t)
  {
    using cmf::log;
    auto F = F_of(t);
    auto J = det(F);
    auto Finv = inv(F);
    // A scalar composite touching every operation used by the materials.
    return log(J) * cmf::ddot(F, cmf::transpose(Finv)) + cmf::tr(F * F)
           + cmf::norm_squared(Finv);
  };
  const double t0 = 0.7;
  const dual gd = g(dual(t0, 1.0));
  const double fd = CentralDifference([&](double t) { return g(t); }, t0, 1e-3);
  CHECK_CLOSE(gd.v, g(t0), 0.0);
  CHECK_MSG(std::abs(gd.d - fd) <= 1e-10 * std::abs(fd),
            "tensor<dual,3,3> composite derivative: got " + std::to_string(gd.d) +
            " fd " + std::to_string(fd));
  // Element-wise: d(det F)/dt = det(F) tr(F^{-1} H).
  const tensor<dual, 3, 3> Fd = F_of(dual(t0, 1.0));
  const dual Jd = det(Fd);
  const tensor<double, 3, 3> F0 = F_of(t0);
  CHECK_CLOSE(Jd.d, det(F0) * cmf::tr(inv(F0) * H), 1e-13);
  // 2x2 embedded: dual determinant of a 2x2 block.
  tensor<dual, 2, 2> F2;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) { F2(i, j) = Fd(i, j); }
  tensor<double, 2, 2> F20, H2;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) { F20(i, j) = F0(i, j); H2(i, j) = H(i, j); }
  CHECK_CLOSE(det(F2).d, det(F20) * cmf::tr(inv(F20) * H2), 1e-14);
}

std::string WriteTemp(const std::string &name, const std::string &text)
{
  const std::string path = "build/tests/out/" + name;
  std::ofstream out(path);
  out << text;
  return path;
}

const char *kGoodYaml = R"(
mesh: { cartesian: { nx: 4, ny: 4, sx: 48.0, sy: 1.0 }, serial_refine: 1, parallel_refine: 0, order: 2,
        corners: [[0.0, 0.0], [48.0, 44.0], [48.0, 60.0], [0.0, 44.0]] }
material: { model: neo_hookean, E: 250.0, nu: 0.3, rho0: 1.0 }
bcs:
  dirichlet: [ { attr: [4], value: [0.0, 0.0] } ]
  traction:  [ { attr: [2], value: [0.0, 6.25] } ]
body_force: [0.0, 0.0]
solver: { load_steps: 1, newton: { rtol: 1e-10, atol: 1e-12, max_it: 25 },
          linear: { type: gmres_amg, rtol: 1e-12, max_it: 500 } }
output: { paraview: out/cook, fields: [displacement, vonmises] }
)";

void TestYaml()
{
  std::filesystem::create_directories("build/tests/out");
  // The schema example of the plan parses with the expected values.
  cmf::AppConfig cfg = cmf::ParseConfig(YAML::Load(kGoodYaml));
  CHECK(cfg.mesh.cartesian && cfg.mesh.box.nx == 4 && cfg.mesh.order == 2);
  CHECK(cfg.mesh.corners.size() == 4 && cfg.mesh.corners[2][1] == 60.0);
  CHECK(cfg.material.model == "neo_hookean");
  CHECK_CLOSE(cfg.material.E, 250.0, 0.0);
  CHECK(cfg.bcs.dirichlet.size() == 1 && cfg.bcs.dirichlet[0].attr[0] == 4);
  CHECK(cfg.bcs.traction.size() == 1 && cfg.bcs.traction[0].value[1] == 6.25);
  CHECK(cfg.body_force.size() == 2);
  CHECK(cfg.solver.newton.max_it == 25 && cfg.solver.linear.type == "gmres_amg");
  CHECK_CLOSE(cfg.solver.linear.rtol, 1e-12, 0.0);
  CHECK(cfg.output.paraview == "out/cook" && cfg.output.fields.size() == 2);
  // Defaults for omitted sections.
  cmf::AppConfig minimal = cmf::ParseConfig(YAML::Load(
    "mesh: { cartesian: { nx: 2, ny: 2 } }\nmaterial: { model: st_venant_kirchhoff, E: 1.0, nu: 0.25 }\n"));
  CHECK(minimal.solver.load_steps == 1 && minimal.solver.newton.max_it == 25);
  CHECK(minimal.bcs.dirichlet.empty() && minimal.output.paraview.empty());
  CHECK_CLOSE(minimal.material.rho0, 1.0, 0.0);

  // Malformed YAML: a clear error, not a crash.
  const std::string bad = WriteTemp("bad_syntax.yaml",
                                    "mesh: { cartesian: { nx: 2, ny: 2 }\nmaterial: [unclosed\n");
  CHECK_THROWS(cmf::LoadConfig(bad), cmf::ConfigError, "YAML syntax error");
  CHECK_THROWS(cmf::LoadConfig("build/tests/out/does_not_exist.yaml"), cmf::ConfigError,
               "cannot open");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load("just a scalar")), cmf::ConfigError, "mesh");

  // Missing keys name the full path.
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { cartesian: { nx: 2, ny: 2 } }\nmaterial: { model: neo_hookean, nu: 0.3 }\n")),
    cmf::ConfigError, "material.E");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { cartesian: { nx: 2 } }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "mesh.cartesian.ny");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "material: { model: neo_hookean, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "mesh");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { order: 1 }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "'file' or 'cartesian'");

  // Wrong types name the key and the expected type.
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { cartesian: { nx: 2, ny: 2 } }\nmaterial: { model: neo_hookean, E: abc, nu: 0.3 }\n")),
    cmf::ConfigError, "material.E");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { cartesian: { nx: 2, ny: 2 } }\nmaterial: { model: neo_hookean, E: abc, nu: 0.3 }\n")),
    cmf::ConfigError, "expected a number");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { cartesian: { nx: 2.5, ny: 2 } }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "mesh.cartesian.nx");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { cartesian: { nx: 2, ny: 2 } }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n"
    "bcs: { dirichlet: [ { attr: 1, value: [0, 0] } ] }\n")),
    cmf::ConfigError, "bcs.dirichlet[0].attr");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { cartesian: { nx: 2, ny: 2 } }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n"
    "bcs: { traction: [ { attr: [2] } ] }\n")),
    cmf::ConfigError, "bcs.traction[0].value");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { cartesian: { nx: 2, ny: 2 } }\nmaterial: 42\n")),
    cmf::ConfigError, "'material' must be a map");

  // Unknown keys and bad enumerations are rejected by name.
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { cartesian: { nx: 2, ny: 2 } }\nmaterial: { modle: neo_hookean, model: neo_hookean, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "unknown key 'material.modle'");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { cartesian: { nx: 2, ny: 2 } }\nmaterial: { model: mooney, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "unknown model 'mooney'");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { cartesian: { nx: 2, ny: 2 } }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n"
    "solver: { linear: { type: mumps } }\n")),
    cmf::ConfigError, "solver.linear.type");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { cartesian: { nx: 2, ny: 2 } }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n"
    "output: { fields: [displacement, stress] }\n")),
    cmf::ConfigError, "unknown field 'stress'");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { cartesian: { nx: 2, ny: 2 } }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.6 }\n")),
    cmf::ConfigError, "material.nu");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { cartesian: { nx: 2, ny: 2 }, corners: [[0,0],[1,0]] }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "mesh.corners");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { cartesian: { nx: 2, ny: 2 }, tetris: 1 }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "unknown key 'mesh.tetris'");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { cartesian: { nx: 2, ny: 2 }, perturb: 0.3 }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "mesh.perturb");
}

} // namespace

int main()
{
  TestTensor2x2();
  TestTensor3x3();
  TestDualScalar();
  TestDualTensor();
  TestYaml();
  return cmf_test::Report("test_base");
}

// S1 gate: tensor algebra vs hand values, dual derivatives vs central finite
// differences, and YAML validation errors that name the bad key.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>

#include "base/config.hpp"
#include "base/expression.hpp"
#include "base/mesh_input.hpp"
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
mesh: { file: apps/mesh/cook.msh, serial_refine: 1, parallel_refine: 0, order: 2 }
material: { model: neo_hookean, E: 250.0, nu: 0.3, rho0: 1.0 }
bcs:
  dirichlet: [ { attr: [4], expression: ["0", "0"] } ]
  traction:  [ { attr: [2], expression: ["0", "6.25"] } ]
body_force: { expression: ["0", "0"] }
solver: { load_steps: 1, newton: { rtol: 1e-10, atol: 1e-12, max_it: 25 },
          linear: { type: gmres_amg, rtol: 1e-12, max_it: 500 } }
output: { paraview: out/cook, fields: [displacement, vonmises] }
)";

void TestYaml()
{
  std::filesystem::create_directories("build/tests/out");
  // The schema example of the plan parses with the expected values.
  cmf::AppConfig cfg = cmf::ParseConfig(YAML::Load(kGoodYaml));
  CHECK(cfg.mesh.file == "apps/mesh/cook.msh" && cfg.mesh.serial_refine == 1 && cfg.mesh.order == 2);
  CHECK(cfg.material.model == "neo_hookean");
  CHECK_CLOSE(cfg.material.E, 250.0, 0.0);
  CHECK(cfg.bcs.dirichlet.size() == 1 && cfg.bcs.dirichlet[0].attr[0] == 4);
  CHECK(cfg.bcs.traction.size() == 1 && cfg.bcs.traction[0].expression[1] == "6.25");
  CHECK(cfg.body_force.expression.size() == 2);
  CHECK(cfg.solver.newton.max_it == 25 && cfg.solver.linear.type == "gmres_amg");
  CHECK_CLOSE(cfg.solver.linear.rtol, 1e-12, 0.0);
  CHECK(cfg.output.paraview == "out/cook" && cfg.output.fields.size() == 2);
  // Defaults for omitted sections.
  cmf::AppConfig minimal = cmf::ParseConfig(YAML::Load(
    "mesh: { file: square.msh }\nmaterial: { model: st_venant_kirchhoff, E: 1.0, nu: 0.25 }\n"));
  CHECK(minimal.solver.load_steps == 1 && minimal.solver.newton.max_it == 25);
  CHECK(minimal.bcs.dirichlet.empty() && minimal.output.paraview.empty());
  CHECK_CLOSE(minimal.material.rho0, 1.0, 0.0);

  // Malformed YAML: a clear error, not a crash.
  const std::string bad = WriteTemp("bad_syntax.yaml",
                                    "mesh: { file: square.msh\nmaterial: [unclosed\n");
  CHECK_THROWS(cmf::LoadConfig(bad), cmf::ConfigError, "YAML syntax error");
  CHECK_THROWS(cmf::LoadConfig("build/tests/out/does_not_exist.yaml"), cmf::ConfigError,
               "cannot open");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load("just a scalar")), cmf::ConfigError, "mesh");

  // Missing keys name the full path.
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { file: square.msh }\nmaterial: { model: neo_hookean, nu: 0.3 }\n")),
    cmf::ConfigError, "material.E");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { cartesian: { nx: 3, ny: 3 } }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "Gmsh");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "material: { model: neo_hookean, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "mesh");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { order: 1 }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "mesh.file");

  // Wrong types name the key and the expected type.
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { file: square.msh }\nmaterial: { model: neo_hookean, E: abc, nu: 0.3 }\n")),
    cmf::ConfigError, "material.E");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { file: square.msh }\nmaterial: { model: neo_hookean, E: abc, nu: 0.3 }\n")),
    cmf::ConfigError, "expected a number");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { file: [a, b] }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "mesh.file");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { file: square.msh }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n"
    "bcs: { dirichlet: [ { attr: 1, expression: [\"0\", \"0\"] } ] }\n")),
    cmf::ConfigError, "bcs.dirichlet[0].attr");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { file: square.msh }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n"
    "bcs: { traction: [ { attr: [2] } ] }\n")),
    cmf::ConfigError, "bcs.traction[0].expression");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { file: square.msh }\nmaterial: 42\n")),
    cmf::ConfigError, "'material' must be a map");

  // Unknown keys and bad enumerations are rejected by name.
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { file: square.msh }\nmaterial: { modle: neo_hookean, model: neo_hookean, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "unknown key 'material.modle'");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { file: square.msh }\nmaterial: { model: mooney, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "unknown model 'mooney'");
  // linear_elastic (small strain): mu or (E, nu), one bulk-modulus key, no volumetric law.
  {
    const std::string mesh = "mesh: { file: square.msh }\n";
    const cmf::AppConfig c = cmf::ParseConfig(YAML::Load(mesh + "material: { model: linear_elastic, E: 2.0, nu: 0.25 }\n"));
    CHECK(c.material.model == "linear_elastic" && c.material.E == 2.0 && c.material.nu == 0.25);
    CHECK(cmf::ParseConfig(YAML::Load(mesh + "material: { model: linear_elastic, mu: 1.0, kappa: 3.0 }\n")).material.kappa == 3.0);
    CHECK_THROWS(cmf::ParseConfig(YAML::Load(mesh + "material: { model: linear_elastic, E: 2.0, nu: 0.25, volumetric: logarithmic }\n")),
                 cmf::ConfigError, "'material.volumetric' is not used by model 'linear_elastic'");
    CHECK_THROWS(cmf::ParseConfig(YAML::Load(mesh + "material: { model: linear_elastic, E: 2.0 }\n")),
                 cmf::ConfigError, "model 'linear_elastic' needs mu, or E and nu");
    CHECK_THROWS(cmf::ParseConfig(YAML::Load(mesh + "material: { model: linear_elastic, mu: 1.0 }\n")),
                 cmf::ConfigError, "needs exactly one of");
    CHECK_THROWS(cmf::ParseConfig(YAML::Load(mesh + "material: { model: linear_elastic, mu: 1.0, nu: 0.3, c1: 1.0 }\n")),
                 cmf::ConfigError, "'material.c1' is not used by model 'linear_elastic'");
  }
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { file: square.msh }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n"
    "solver: { linear: { type: mumps } }\n")),
    cmf::ConfigError, "solver.linear.type");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { file: square.msh }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n"
    "output: { fields: [displacement, stress] }\n")),
    cmf::ConfigError, "unknown field 'stress'");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "plane: shell\nmesh: { file: square.msh }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "key 'plane'");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "formulation: mixed\nplane: stress\nmesh: { file: square.msh, order: 2 }\n"
    "material: { model: iso_neo_hookean, mu: 1.0, incompressible: true }\n")),
    cmf::ConfigError, "formulation: displacement");
  CHECK(cmf::ParseConfig(YAML::Load(
    "plane: stress\nmesh: { file: square.msh }\n"
    "material: { model: iso_neo_hookean, mu: 1.0, incompressible: true }\n")).plane == "stress");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { file: square.msh }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n"
    "output: { fields: [displacement, vonmises], quadrature_at: [cells] }\n")),
    cmf::ConfigError, "output.quadrature_at");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { file: square.msh }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n"
    "output: { fields: [displacement, vonmises], nodal_projection: lumped }\n")),
    cmf::ConfigError, "output.nodal_projection");
  {
    const cmf::AppConfig c = cmf::ParseConfig(YAML::Load(
      "mesh: { file: square.msh }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n"
      "output: { fields: [displacement, energy_density], quadrature_at: [elements, quadrature_points], "
      "nodal_projection: projected }\n"));
    CHECK(c.output.quadrature_at.size() == 2 && c.output.quadrature_at[1] == "quadrature_points");
    CHECK(c.output.nodal_projection == "projected");
  }
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { file: square.msh }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.6 }\n")),
    cmf::ConfigError, "material.nu");

  // Boundary attributes by number and by physical-group name.
  {
    const cmf::AppConfig c = cmf::ParseConfig(YAML::Load(
      "mesh: { file: square.msh }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n"
      "bcs: { dirichlet: [ { attr: [left, 2, top], expression: [\"0\", \"0\"] } ] }\n"));
    const cmf::BoundaryCondition &bc = c.bcs.dirichlet.at(0);
    CHECK(bc.attr.size() == 1 && bc.attr[0] == 2);
    CHECK(bc.attr_names.size() == 2 && bc.attr_names[0] == "left" && bc.attr_names[1] == "top");
    CHECK_THROWS(cmf::ParseConfig(YAML::Load(
      "mesh: { file: square.msh }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n"
      "bcs: { dirichlet: [ { attr: [], expression: [\"0\", \"0\"] } ] }\n")),
      cmf::ConfigError, "bcs.dirichlet[0].attr");
    // Resolution against a mesh: numbers must exist, names come from the
    // boundary attribute sets (Gmsh physical names).
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(2, 2, mfem::Element::QUADRILATERAL, true);
    mfem::Array<int> left(1), top(1), lid(2);
    left[0] = 4;
    top[0] = 3;
    lid[0] = 1; lid[1] = 3;
    mesh.bdr_attribute_sets.SetAttributeSet("left", left);
    mesh.bdr_attribute_sets.SetAttributeSet("top", top);
    mesh.bdr_attribute_sets.SetAttributeSet("lid", lid);
    CHECK(cmf::DescribeAttributes(mesh, true) == "1 (lid), 2, 3 (lid, top), 4 (left)");
    // [left, 2, top] -> {4, 2, 3}, sorted and unique.
    CHECK(cmf::ResolveBoundaryAttributes(mesh, bc, "bcs.dirichlet[0]") == std::vector<int>({2, 3, 4}));
    cmf::BoundaryCondition lidbc;
    lidbc.attr_names = {"lid"};
    lidbc.attr = {4};
    CHECK(cmf::ResolveBoundaryAttributes(mesh, lidbc, "x") == std::vector<int>({1, 3, 4}));
    cmf::BoundaryCondition bad;
    bad.attr_names = {"front"};
    CHECK_THROWS(cmf::ResolveBoundaryAttributes(mesh, bad, "bcs.traction[0]"), cmf::ConfigError,
                 "no boundary physical group named 'front' (boundary attributes: 1 (lid), 2, 3 (lid, top), 4 (left))");
    bad.attr_names.clear();
    bad.attr = {7};
    CHECK_THROWS(cmf::ResolveBoundaryAttributes(mesh, bad, "bcs.traction[0]"), cmf::ConfigError,
                 "boundary attribute 7 is not in the mesh");
  }
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { file: square.msh, corners: [[0,0],[1,0]] }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "mesh.corners");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { file: square.msh, tetris: 1 }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "unknown key 'mesh.tetris'");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(
    "mesh: { file: square.msh, perturb: 0.3 }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n")),
    cmf::ConfigError, "mesh.perturb");
}

} // namespace

// Load schedules: evaluation and the YAML of schedules, components, the
// body-force map form, step segments and bisection control.
void TestLoading()
{
  const cmf::Schedule ramp;
  CHECK_CLOSE(ramp.Eval(0.0), 0.0, 0.0);
  CHECK_CLOSE(ramp.Eval(0.25), 0.25, 1e-15);
  CHECK_CLOSE(ramp.Eval(1.0), 1.0, 0.0);
  CHECK_CLOSE(ramp.Eval(1.5), 1.0, 0.0);
  const cmf::Schedule late = cmf::Schedule::Ramp(0.5, 1.0);
  CHECK_CLOSE(late.Eval(0.5), 0.0, 0.0);
  CHECK_CLOSE(late.Eval(0.75), 0.5, 1e-15);
  CHECK_CLOSE(late.Eval(0.2), 0.0, 0.0);
  const cmf::Schedule constant = cmf::Schedule::Constant();
  CHECK_CLOSE(constant.Eval(0.0), 0.0, 0.0);
  CHECK_CLOSE(constant.Eval(1e-9), 1.0, 0.0);
  const cmf::Schedule table = cmf::Schedule::Table({0.0, 0.5, 1.0}, {0.0, 1.0, 0.0});
  CHECK_CLOSE(table.Eval(0.25), 0.5, 1e-15);
  CHECK_CLOSE(table.Eval(0.5), 1.0, 0.0);
  CHECK_CLOSE(table.Eval(0.875), 0.25, 1e-15);
  CHECK_CLOSE(table.Eval(2.0), 0.0, 0.0);
  const cmf::Schedule offset = cmf::Schedule::Table({0.2, 0.6}, {2.0, 4.0});
  CHECK_CLOSE(offset.Eval(0.0), 2.0, 0.0);
  CHECK_CLOSE(offset.Eval(0.4), 3.0, 1e-15);

  const std::string head =
    "mesh: { file: square.msh }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n";
  {
    const cmf::AppConfig c = cmf::ParseConfig(YAML::Load(head +
      "bcs:\n"
      "  dirichlet:\n"
      "    - { attr: [left], expression: [\"0\", \"0\"], components: [x] }\n"
      "    - { attr: [bottom], expression: [\"0\", \"0\"], components: [1], schedule: { type: ramp, from: 0.5 } }\n"
      "  traction:\n"
      "    - { attr: [right], expression: [\"0\", \"1\"], schedule: { type: table, t: [0, 0.5, 1], s: [0, 1, 0] } }\n"
      "body_force: { expression: [\"0\", \"-1\"], schedule: { type: constant } }\n"
      "solver: { steps: [ { to: 0.5, n: 2 }, { to: 1.0, n: 3 } ], "
      "substep: { on_failure: true, max_bisections: 3, min_dt: 0.01 } }\n"));
    CHECK(c.bcs.dirichlet[0].components == std::vector<int>({0}));
    CHECK(c.bcs.dirichlet[1].components == std::vector<int>({1}));
    CHECK(c.bcs.dirichlet[1].schedule.kind == cmf::Schedule::Kind::Ramp);
    CHECK_CLOSE(c.bcs.dirichlet[1].schedule.from, 0.5, 0.0);
    CHECK(c.bcs.traction[0].schedule.kind == cmf::Schedule::Kind::Table);
    CHECK_CLOSE(c.bcs.traction[0].schedule.Eval(0.75), 0.5, 1e-15);
    CHECK(c.body_force.schedule.kind == cmf::Schedule::Kind::Constant);
    CHECK(c.body_force.expression.size() == 2);
    CHECK(c.solver.load_steps == 5);
    const std::vector<double> want = {0.25, 0.5, 0.5 + 1.0 / 6.0, 0.5 + 2.0 / 6.0, 1.0};
    CHECK(c.solver.breakpoints.size() == 5);
    for (std::size_t i = 0; i < want.size() && i < c.solver.breakpoints.size(); i++)
    {
      CHECK_CLOSE(c.solver.breakpoints[i], want[i], 1e-15);
    }
    CHECK(c.solver.substep.on_failure && c.solver.substep.max_bisections == 3);
    CHECK_CLOSE(c.solver.substep.min_dt, 0.01, 0.0);
  }
  // The predictor key of the load stepper.
  {
    const std::string head2 = "mesh: { file: square.msh }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n";
    CHECK(cmf::ParseConfig(YAML::Load(head2)).solver.predictor == "none");
    CHECK(cmf::ParseConfig(YAML::Load(head2 + "solver: { predictor: tangent }\n")).solver.predictor == "tangent");
    CHECK_THROWS(cmf::ParseConfig(YAML::Load(head2 + "solver: { predictor: secant }\n")), cmf::ConfigError,
                 "'solver.predictor': expected none or tangent");
  }
  // Defaults: ramp schedules, all components, no breakpoints, no bisection.
  {
    const cmf::AppConfig c = cmf::ParseConfig(YAML::Load(head +
      "bcs: { dirichlet: [ { attr: [left], expression: [\"0\", \"0\"] } ] }\nbody_force: { expression: [\"0\", \"-1\"] }\n"
      "solver: { load_steps: 4 }\n"));
    CHECK(c.bcs.dirichlet[0].components.empty());
    CHECK(c.bcs.dirichlet[0].schedule.kind == cmf::Schedule::Kind::Ramp);
    CHECK(c.body_force.schedule.kind == cmf::Schedule::Kind::Ramp);
    CHECK(c.solver.breakpoints.empty() && c.solver.load_steps == 4);
    CHECK(!c.solver.substep.on_failure);
  }
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "bcs: { dirichlet: [ { attr: [left], expression: [\"0\", \"0\"], components: [x, x] } ] }\n")),
    cmf::ConfigError, "bcs.dirichlet[0].components");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "bcs: { dirichlet: [ { attr: [left], expression: [\"0\", \"0\"], components: [w] } ] }\n")),
    cmf::ConfigError, "components[0]");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "bcs: { traction: [ { attr: [left], expression: [\"0\", \"0\"], components: [x] } ] }\n")),
    cmf::ConfigError, "Dirichlet entries only");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "bcs: { dirichlet: [ { attr: [left], expression: [\"0\", \"0\"], schedule: { type: sine } } ] }\n")),
    cmf::ConfigError, "unknown schedule 'sine'");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "bcs: { dirichlet: [ { attr: [left], expression: [\"0\", \"0\"], schedule: { type: ramp, from: 0.7, to: 0.2 } } ] }\n")),
    cmf::ConfigError, "schedule.from");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "bcs: { dirichlet: [ { attr: [left], expression: [\"0\", \"0\"], schedule: { type: table, t: [0, 1], s: [1] } } ] }\n")),
    cmf::ConfigError, "schedule.t");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "bcs: { dirichlet: [ { attr: [left], expression: [\"0\", \"0\"], schedule: { type: table, t: [0.5, 0.2], s: [1, 2] } } ] }\n")),
    cmf::ConfigError, "increase strictly");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "solver: { steps: [ { to: 0.5, n: 2 } ] }\n")),
    cmf::ConfigError, "end at to: 1.0");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "solver: { load_steps: 2, steps: [ { to: 1.0, n: 2 } ] }\n")),
    cmf::ConfigError, "give one, not both");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "solver: { steps: [ { to: 0.5, n: 0 }, { to: 1.0, n: 1 } ] }\n")),
    cmf::ConfigError, "solver.steps[0].n");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "solver: { substep: { min_dt: 0 } }\n")),
    cmf::ConfigError, "solver.substep.min_dt");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head + "body_force: { schedule: { type: constant } }\n")),
    cmf::ConfigError, "body_force.expression");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head + "body_force: [0, 1]\n")),
    cmf::ConfigError, "'body_force' must be a map");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head + "body_force: { expression: [\"0\", \"1\"], scale: 2 }\n")),
    cmf::ConfigError, "unknown key 'body_force.scale'");
}

// Expression parser: precedence, functions, comparisons, errors with columns.
// The dynamics block: time steps, schemes, schedules in physical time and the
// keys that belong to one analysis only.
void TestDynamicsConfig()
{
  const std::string head =
    "mesh: { file: square.msh }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3, rho0: 2.0 }\n";
  auto parse = [&head](const std::string &rest) { return cmf::ParseConfig(YAML::Load(head + rest)); };

  // Absent: quasi-static, nothing about time changes.
  {
    const cmf::AppConfig c = parse("bcs: { traction: [ { attr: [2], expression: [\"0\", \"1\"] } ] }\n");
    CHECK(!c.dynamics.enabled && c.dynamics.breakpoints.empty());
    CHECK(c.bcs.traction[0].schedule.kind == cmf::Schedule::Kind::Ramp);
    CHECK(c.output.every == 1 && !c.output.energy);
    CHECK_CLOSE(c.solver.substep.min_dt, 1e-4, 0.0);
  }
  // dt: equal steps ending on t_final exactly; a dt that does not divide it is shortened.
  {
    const cmf::AppConfig c = parse("dynamics: { t_final: 0.02, dt: 1.0e-5 }\n");
    CHECK(c.dynamics.enabled && c.dynamics.breakpoints.size() == 2000);
    CHECK(c.dynamics.breakpoints.back() == 0.02 && c.dynamics.scheme == "newmark");
    CHECK_CLOSE(c.dynamics.breakpoints[0], 1e-5, 1e-20);
    CHECK_CLOSE(c.solver.substep.min_dt, 1e-8, 1e-20); // 1e-3 of the smallest planned step
    const cmf::AppConfig d = parse("dynamics: { t_final: 1.0, dt: 0.3 }\n");
    CHECK(d.dynamics.breakpoints.size() == 4 && d.dynamics.breakpoints.back() == 1.0);
    CHECK_CLOSE(d.dynamics.breakpoints[0], 0.25, 1e-15);
  }
  // steps: segments in physical time.
  {
    const cmf::AppConfig c = parse(
      "dynamics: { t_final: 2.0, steps: [ { to: 0.5, n: 5 }, { to: 2.0, n: 3 } ], scheme: generalized_alpha, rho_inf: 0.8,\n"
      "            initial: { displacement: [\"0\", \"0\"], velocity: [\"0\", \"1.5*x\"] } }\n"
      "solver: { substep: { min_dt: 0.01 } }\n");
    CHECK(c.dynamics.breakpoints.size() == 8 && c.dynamics.breakpoints.back() == 2.0);
    CHECK_CLOSE(c.dynamics.breakpoints[4], 0.5, 1e-15);
    CHECK_CLOSE(c.dynamics.breakpoints[5], 1.0, 1e-15);
    CHECK_CLOSE(c.dynamics.rho_inf, 0.8, 0.0);
    CHECK(c.dynamics.initial_velocity.size() == 2 && c.dynamics.initial_velocity[1] == "1.5*x");
    CHECK(c.solver.substep.on_failure);
    CHECK_CLOSE(c.solver.substep.min_dt, 0.01, 0.0);
  }
  // Schedules in physical time: the default is constant (on from t = 0), a
  // ramp defaults to [0, t_final], tables reach t_final.
  {
    const cmf::AppConfig c = parse(
      "dynamics: { t_final: 5.0, dt: 0.5 }\n"
      "bcs:\n"
      "  dirichlet: [ { attr: [4], expression: [\"0\", \"0\"] } ]\n"
      "  traction:\n"
      "    - { attr: [2], expression: [\"0\", \"1\"] }\n"
      "    - { attr: [3], expression: [\"0\", \"sin(3*t)\"] }\n"
      "    - { attr: [1], type: pressure, expression: \"2\", schedule: { type: ramp } }\n"
      "    - { attr: [1], type: pressure, expression: \"2\", schedule: { type: table, t: [0, 2.5, 5], s: [0, 1, 0] } }\n"
      "body_force: { expression: [\"0\", \"-9.81\"] }\n"
      "output: { fields: [displacement, velocity, acceleration], every: 10, energy: true }\n");
    CHECK(c.bcs.dirichlet[0].schedule.kind == cmf::Schedule::Kind::Constant);
    CHECK(c.bcs.traction[0].schedule.kind == cmf::Schedule::Kind::Constant);
    CHECK(c.bcs.traction[1].schedule.kind == cmf::Schedule::Kind::Constant);
    CHECK(c.bcs.traction[2].schedule.kind == cmf::Schedule::Kind::Ramp);
    CHECK_CLOSE(c.bcs.traction[2].schedule.to, 5.0, 0.0);
    CHECK_CLOSE(c.bcs.traction[3].schedule.Eval(1.25, true), 0.5, 1e-15);
    CHECK(c.body_force.schedule.kind == cmf::Schedule::Kind::Constant);
    CHECK(c.output.every == 10 && c.output.energy && c.output.fields.size() == 3);
    CHECK(cmf::DescribeTimeDependence(c.bcs.traction[0].schedule, c.bcs.traction[0].expression) ==
          "constant in time (on from t = 0)");
    CHECK(cmf::DescribeTimeDependence(c.bcs.traction[1].schedule, c.bcs.traction[1].expression) ==
          "its expression in t");
    CHECK(cmf::DescribeTimeDependence(c.bcs.traction[2].schedule, c.bcs.traction[2].expression) ==
          "ramp over [0, 5]");
  }
  // The right limit at t = 0: a constant schedule is a step load that is on at once.
  const cmf::Schedule constant = cmf::Schedule::Constant();
  CHECK_CLOSE(constant.Eval(0.0, true), 1.0, 0.0);
  CHECK_CLOSE(constant.Eval(0.0), 0.0, 0.0);

  // Errors name the key.
  CHECK_THROWS(parse("dynamics: { dt: 0.1 }\n"), cmf::ConfigError, "dynamics.t_final");
  CHECK_THROWS(parse("dynamics: { t_final: 1.0 }\n"), cmf::ConfigError, "exactly one");
  CHECK_THROWS(parse("dynamics: { t_final: 1.0, dt: 0.1, steps: [ { to: 1.0, n: 2 } ] }\n"),
               cmf::ConfigError, "exactly one");
  CHECK_THROWS(parse("dynamics: { t_final: 1.0, dt: 2.0 }\n"), cmf::ConfigError, "exceeds t_final");
  CHECK_THROWS(parse("dynamics: { t_final: -1.0, dt: 0.1 }\n"), cmf::ConfigError, "dynamics.t_final");
  CHECK_THROWS(parse("dynamics: { t_final: 2.0, steps: [ { to: 1.0, n: 2 } ] }\n"), cmf::ConfigError,
               "must end at to: t_final");
  CHECK_THROWS(parse("dynamics: { t_final: 2.0, steps: [ { to: 3.0, n: 2 } ] }\n"), cmf::ConfigError,
               "end at t_final");
  CHECK_THROWS(parse("dynamics: { t_final: 1.0, dt: 0.1, scheme: leapfrog }\n"), cmf::ConfigError,
               "unknown scheme 'leapfrog'");
  CHECK_THROWS(parse("dynamics: { t_final: 1.0, dt: 0.1, rho_inf: 0.5 }\n"), cmf::ConfigError,
               "'dynamics.rho_inf' is not used by scheme 'newmark'");
  CHECK_THROWS(parse("dynamics: { t_final: 1.0, dt: 0.1, scheme: hht, beta: 0.3 }\n"), cmf::ConfigError,
               "'dynamics.beta' is not used by scheme 'hht'");
  CHECK_THROWS(parse("dynamics: { t_final: 1.0, dt: 0.1, scheme: generalized_alpha, rho_inf: 1.2 }\n"),
               cmf::ConfigError, "dynamics.rho_inf");
  CHECK_THROWS(parse("dynamics: { t_final: 1.0, dt: 0.1, scheme: hht, alpha: 0.4 }\n"), cmf::ConfigError,
               "dynamics.alpha");
  CHECK_THROWS(parse("dynamics: { t_final: 1.0, dt: 0.1, beta: 0.0 }\n"), cmf::ConfigError, "dynamics.beta");
  CHECK_THROWS(parse("dynamics: { t_final: 1.0, dt: 0.1, gamma: 0.4 }\n"), cmf::ConfigError, "dynamics.gamma");
  CHECK_THROWS(parse("dynamics: { t_final: 1.0, dt: 0.1, damping: 0.1 }\n"), cmf::ConfigError,
               "unknown key 'dynamics.damping'");
  CHECK_THROWS(parse("dynamics: { t_final: 1.0, dt: 0.1, initial: { velocity: [\"t\", \"0\"] } }\n"),
               cmf::ConfigError, "dynamics.initial.velocity[0]");
  CHECK_THROWS(parse("dynamics: { t_final: 1.0, dt: 0.1, initial: { acceleration: [\"0\", \"0\"] } }\n"),
               cmf::ConfigError, "unknown key 'dynamics.initial.acceleration'");
  for (const std::string key : {"load_steps: 4", "steps: [ { to: 1.0, n: 2 } ]", "predictor: tangent"})
  {
    CHECK_THROWS(parse("dynamics: { t_final: 1.0, dt: 0.1 }\nsolver: { " + key + " }\n"), cmf::ConfigError,
                 "is not used in a dynamic analysis");
  }
  CHECK_THROWS(parse("dynamics: { t_final: 1.0, dt: 0.1 }\nsolver: { substep: { min_dt: 0.0 } }\n"),
               cmf::ConfigError, "solver.substep.min_dt");
  CHECK_THROWS(parse("dynamics: { t_final: 2.0, dt: 0.1 }\n"
                     "bcs: { traction: [ { attr: [2], expression: [\"0\", \"1\"], schedule: { type: ramp, to: 3.0 } } ] }\n"),
               cmf::ConfigError, "<= t_final");
  CHECK_THROWS(parse("dynamics: { t_final: 2.0, dt: 0.1 }\n"
                     "body_force: { expression: [\"0\", \"1\"], schedule: { type: table, t: [0, 4], s: [0, 1] } }\n"),
               cmf::ConfigError, "within [0, t_final]");
  // Without the block the pseudo-time bounds and the dynamic output keys are errors.
  CHECK_THROWS(parse("bcs: { traction: [ { attr: [2], expression: [\"0\", \"1\"], schedule: { type: ramp, to: 3.0 } } ] }\n"),
               cmf::ConfigError, "<= 1");
  CHECK_THROWS(parse("output: { fields: [displacement, velocity] }\n"), cmf::ConfigError,
               "'velocity' needs a dynamic analysis");
  CHECK_THROWS(parse("output: { energy: true }\n"), cmf::ConfigError, "needs a dynamic analysis");
  CHECK_THROWS(parse("output: { every: 0 }\n"), cmf::ConfigError, "output.every");
  CHECK(parse("output: { every: 5 }\n").output.every == 5);
}

void TestExpression()
{
  auto ev = [](const char *text, double x = 0.0, double y = 0.0, double z = 0.0, double t = 0.0)
  {
    return cmf::Expression::Parse(text).Eval(x, y, z, t);
  };
  struct Row { const char *text; double x, y, z, t, want; };
  const Row rows[] = {
    {"1 + 2 * 3", 0, 0, 0, 0, 7.0},
    {"(1 + 2) * 3", 0, 0, 0, 0, 9.0},
    {"1 - 2 - 3", 0, 0, 0, 0, -4.0},
    {"2 / 4 / 2", 0, 0, 0, 0, 0.25},
    {"2^3^2", 0, 0, 0, 0, 512.0},
    {"-x^2", 3, 0, 0, 0, -9.0},
    {"(-x)^2", 3, 0, 0, 0, 9.0},
    {"2^-1", 0, 0, 0, 0, 0.5},
    {"-2 + 3", 0, 0, 0, 0, 1.0},
    {"+x", 1.5, 0, 0, 0, 1.5},
    {"x*y - z + t", 2, 3, 4, 5, 7.0},
    {"sin(pi/2) + cos(0)", 0, 0, 0, 0, 2.0},
    {"exp(log(3.5))", 0, 0, 0, 0, 3.5},
    {"sqrt(16) + abs(-2)", 0, 0, 0, 0, 6.0},
    {"tan(pi/4)", 0, 0, 0, 0, 1.0},
    {"pow(2, 10)", 0, 0, 0, 0, 1024.0},
    {"min(x, y) + max(x, y)", -1, 4, 0, 0, 3.0},
    {"if(t < 0.5, 2*t, 1)", 0, 0, 0, 0.25, 0.5},
    {"if(t < 0.5, 2*t, 1)", 0, 0, 0, 0.75, 1.0},
    {"if(x <= 1, 1, 0) + if(x >= 1, 1, 0) + if(x == 1, 1, 0) + if(x != 1, 1, 0) + if(x > 1, 1, 0)", 1, 0, 0, 0, 3.0},
    {"1e-3 * 2.5E2", 0, 0, 0, 0, 0.25},
    {".5 + 1.", 0, 0, 0, 0, 1.5},
    {"0.1*t*sin(pi*x)", 0.5, 0, 0, 0.3, 0.03},
    {"x*(1-y)*exp(-z)", 2, 0.5, 0, 0, 1.0},
    {" 3 ", 0, 0, 0, 0, 3.0},
  };
  for (const Row &r : rows)
  {
    const double got = ev(r.text, r.x, r.y, r.z, r.t);
    CHECK_MSG(std::abs(got - r.want) <= 1e-13 * std::max(1.0, std::abs(r.want)),
              std::string("expression '") + r.text + "' = " + std::to_string(got) +
              ", want " + std::to_string(r.want));
  }
  CHECK(cmf::Expression::Parse("x + t").UsesTime());
  CHECK(!cmf::Expression::Parse("x + y*z").UsesTime());
  CHECK(!cmf::Expression::Parse("tan(x)").UsesTime());

  CHECK_THROWS(cmf::Expression::Parse(""), cmf::ConfigError, "empty expression");
  CHECK_THROWS(cmf::Expression::Parse("(1 + 2"), cmf::ConfigError, "expected ')' at column 7");
  CHECK_THROWS(cmf::Expression::Parse("1 + 2)"), cmf::ConfigError, "unexpected ')' at column 6");
  CHECK_THROWS(cmf::Expression::Parse("2 * foo"), cmf::ConfigError, "unknown identifier 'foo' at column 5");
  CHECK_THROWS(cmf::Expression::Parse("1 +"), cmf::ConfigError, "unexpected end of expression at column 4");
  CHECK_THROWS(cmf::Expression::Parse("sin(1, 2)"), cmf::ConfigError, "closing 'sin'");
  CHECK_THROWS(cmf::Expression::Parse("min(1)"), cmf::ConfigError, "'min' takes 2 arguments");
  CHECK_THROWS(cmf::Expression::Parse("sin 1"), cmf::ConfigError, "expected '(' after 'sin'");
  CHECK_THROWS(cmf::Expression::Parse("1 < 2 < 3"), cmf::ConfigError, "do not chain");
  CHECK_THROWS(cmf::Expression::Parse("x $ 2"), cmf::ConfigError, "unexpected '$' at column 3");

  // YAML: expression entries, default schedules, errors by key.
  const std::string head =
    "mesh: { file: square.msh }\nmaterial: { model: neo_hookean, E: 1.0, nu: 0.3 }\n";
  {
    const cmf::AppConfig c = cmf::ParseConfig(YAML::Load(head +
      "bcs:\n"
      "  dirichlet: [ { attr: [top], expression: [\"0.1*t*sin(pi*x)\", \"0\"] } ]\n"
      "  traction:\n"
      "    - { attr: [right], expression: [\"0\", \"if(t < 0.5, 2*t, 1)\"], schedule: { type: ramp } }\n"
      "    - { attr: [left], type: pressure, expression: \"0.3*t\" }\n"
      "    - { attr: [bottom], type: follower_pressure, expression: \"0.2\" }\n"
      "body_force: { expression: [\"0\", \"-9.81*t\"] }\n"));
    CHECK(c.bcs.dirichlet[0].expression.size() == 2);
    CHECK(c.bcs.dirichlet[0].schedule.kind == cmf::Schedule::Kind::Constant);
    CHECK(c.bcs.traction[0].schedule.kind == cmf::Schedule::Kind::Ramp);
    CHECK(c.bcs.traction[1].type == "pressure" && c.bcs.traction[1].expression.size() == 1);
    CHECK(c.bcs.traction[2].type == "follower_pressure" && c.bcs.traction[2].expression == std::vector<std::string>({"0.2"}));
    CHECK(c.bcs.traction[2].schedule.kind == cmf::Schedule::Kind::Ramp); // no t in the data: ramp
    CHECK(c.body_force.expression.size() == 2 && c.body_force.schedule.kind == cmf::Schedule::Kind::Constant);
  }
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "bcs: { dirichlet: [ { attr: [top], value: [0, 0] } ] }\n")),
    cmf::ConfigError, "missing key 'bcs.dirichlet[0].expression'");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head + "bcs: { dirichlet: [ { attr: [top] } ] }\n")),
    cmf::ConfigError, "missing key 'bcs.dirichlet[0].expression'");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "bcs: { dirichlet: [ { attr: [top], expression: [\"0\", \"2 *\"] } ] }\n")),
    cmf::ConfigError, "bcs.dirichlet[0].expression[1]");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "bcs: { dirichlet: [ { attr: [top], expression: \"x\" } ] }\n")),
    cmf::ConfigError, "one string per component");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "bcs: { traction: [ { attr: [top], type: pressure, expression: [\"1\", \"2\"] } ] }\n")),
    cmf::ConfigError, "must be a string");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "bcs: { traction: [ { attr: [top], type: suction, expression: \"1\" } ] }\n")),
    cmf::ConfigError, "unknown type 'suction'");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "bcs: { dirichlet: [ { attr: [top], type: pressure, expression: [\"1\", \"0\"] } ] }\n")),
    cmf::ConfigError, "unknown key 'bcs.dirichlet[0].type'");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "bcs: { traction: [ { attr: [top], expression: [\"1\", \"0\"], gradient: [[1, 0], [0, 1]] } ] }\n")),
    cmf::ConfigError, "unknown key 'bcs.traction[0].gradient'");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head + "body_force: { expression: [\"0\", \"y +\"] }\n")),
    cmf::ConfigError, "body_force.expression[1]");
}

// Material regions: inheritance from the base, bulk-key replacement, errors.
void TestMaterialRegions()
{
  const std::string head = "mesh: { file: square.msh }\n";
  {
    const cmf::AppConfig c = cmf::ParseConfig(YAML::Load(head +
      "material:\n"
      "  model: arruda_boyce\n"
      "  mu: 280.0\n"
      "  N: 26.2144\n"
      "  kappa: 280000.0\n"
      "  regions:\n"
      "    - { attr: [inclusion, 3], mu: 2800.0, kappa: 2800000.0 }\n"
      "    - { attr: [2], N: 9.0 }\n"));
    CHECK(c.material.regions.size() == 2);
    const cmf::MaterialConfig &r0 = c.material.regions[0];
    CHECK(r0.model == "arruda_boyce" && r0.attr_names == std::vector<std::string>({"inclusion"}) &&
          r0.attr == std::vector<int>({3}));
    CHECK_CLOSE(r0.mu, 2800.0, 0.0);
    CHECK_CLOSE(r0.N, 26.2144, 0.0);       // inherited
    CHECK_CLOSE(r0.kappa, 2800000.0, 0.0);
    CHECK(r0.regions.empty());
    const cmf::MaterialConfig &r1 = c.material.regions[1];
    CHECK_CLOSE(r1.mu, 280.0, 0.0);        // inherited
    CHECK_CLOSE(r1.N, 9.0, 0.0);
    CHECK_CLOSE(r1.kappa, 280000.0, 0.0);  // inherited
    CHECK(c.material.attr.empty() && c.material.attr_names.empty());
  }
  // The volumetric law is a key of the decoupled models and is inherited by regions.
  {
    const cmf::AppConfig c = cmf::ParseConfig(YAML::Load(head +
      "material: { model: iso_neo_hookean, mu: 1.0, kappa: 100.0, volumetric: simo_taylor, "
      "regions: [ { attr: [2], mu: 2.0 } ] }\n"));
    CHECK(c.material.volumetric == "simo_taylor" && c.material.regions[0].volumetric == "simo_taylor");
    CHECK(cmf::ParseConfig(YAML::Load(head + "material: { model: yeoh, c10: 1.0, nu: 0.3 }\n"))
            .material.volumetric == "quadratic");
    CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
      "material: { model: iso_neo_hookean, mu: 1.0, kappa: 100.0, volumetric: cubic }\n")),
      cmf::ConfigError, "'material.volumetric': unknown volumetric law 'cubic'");
    CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
      "material: { model: neo_hookean, E: 1.0, nu: 0.3, volumetric: logarithmic }\n")),
      cmf::ConfigError, "'material.volumetric' is not used by model 'neo_hookean'");
  }
  // A region may switch the bulk specification (nu instead of kappa).
  {
    const cmf::AppConfig c = cmf::ParseConfig(YAML::Load(head +
      "material: { model: iso_neo_hookean, mu: 1.0, kappa: 100.0, regions: [ { attr: [2], nu: 0.3 } ] }\n"));
    CHECK(std::isnan(c.material.regions[0].kappa));
    CHECK_CLOSE(c.material.regions[0].nu, 0.3, 0.0);
  }
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "material: { model: iso_neo_hookean, mu: 1.0, kappa: 100.0, regions: [ { attr: [2], model: gent, Jm: 3 } ] }\n")),
    cmf::ConfigError, "must use the base model");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "material: { model: iso_neo_hookean, mu: 1.0, kappa: 100.0, regions: [ { mu: 2.0 } ] }\n")),
    cmf::ConfigError, "material.regions[0].attr");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "material: { model: iso_neo_hookean, mu: 1.0, kappa: 100.0, regions: [ { attr: [2], incompressible: true } ] }\n")),
    cmf::ConfigError, "incompressible or none");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "material: { model: iso_neo_hookean, mu: 1.0, kappa: 100.0, regions: [ { attr: [2], mu: -2.0 } ] }\n")),
    cmf::ConfigError, "material.regions[0].mu");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "material: { model: neo_hookean, E: 1.0, nu: 0.3, regions: [ { attr: [2], kappa: 3.0 } ] }\n")),
    cmf::ConfigError, "material.regions[0].nu");
  CHECK_THROWS(cmf::ParseConfig(YAML::Load(head +
    "material: { model: iso_neo_hookean, mu: 1.0, kappa: 100.0, regions: [ { attr: [2], mu: 2.0, colour: red } ] }\n")),
    cmf::ConfigError, "unknown key 'material.regions[0].colour'");
}

int main()
{
  TestTensor2x2();
  TestTensor3x3();
  TestDualScalar();
  TestDualTensor();
  TestYaml();
  TestLoading();
  TestDynamicsConfig();
  TestExpression();
  TestMaterialRegions();
  return cmf_test::Report("test_base");
}

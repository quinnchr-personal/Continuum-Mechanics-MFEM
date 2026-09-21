// Homogeneous deformation gates for the incompressible models. Every
// decoupled material at kappa = inf must reproduce the closed-form states of
// doc/verification_manual.tex, Appendix A:
//   1. at the material point, the code's Cauchy stress at F = diag(lambda_a)
//      with the analytic mean stress equals the analytic principal stresses
//      (plane-strain extension, uniaxial, equibiaxial, pure shear), and the
//      deviatoric stress in simple shear equals its spectral closed form;
//   2. as the finite element solution of the mixed u-p formulation with the
//      affine displacement prescribed on the two end faces and the lateral
//      faces free (plane-strain extension in 2D, uniaxial tension in 3D), and
//      of the plane-stress displacement formulation (plane: stress) for the
//      sheet states of that appendix: uniaxial (end faces prescribed, lateral
//      free), equibiaxial, pure shear and simple shear (affine displacement on
//      the whole boundary, well posed since the thickness stretch and the
//      pressure are eliminated pointwise):
//      displacement, pressure, and every quadrature quantity (Cauchy and
//      first Piola-Kirchhoff stress, J, von Mises, energy density, thickness
//      stretch) in all three presentations: nodal (both projections),
//      element average, and the raw quadrature-point values. The affine
//      field and the constant pressure lie in the Taylor-Hood spaces, so the
//      discrete solution is exact to solver tolerance.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "base/coefficients.hpp"
#include "base/fields.hpp"
#include "base/mesh_input.hpp"
#include "base/probes.hpp"
#include "physics/solid_problem.hpp"
#include "kernels/mixed_total_lagrangian.hpp"
#include "materials/materials.hpp"
#include "mfem.hpp"
#include "physics/mixed_solid_mechanics_tl.hpp"
#include "physics/solid_mechanics_tl.hpp"
#include "solvers/quasi_static.hpp"
#include "test_util.hpp"

using cmf::tensor;

namespace
{

using Mat3 = tensor<double, 3, 3>;
const double kInf = std::numeric_limits<double>::infinity();

struct Case
{
  std::string name;
  cmf::MixedMaterial material;
};

// The models with rubber-like parameters (shear modulus of order 1).
std::vector<Case> Models()
{
  return {
    {"neo_hookean", cmf::IsoNeoHookean(1.0, kInf)},
    {"mooney_rivlin", cmf::MooneyRivlin(0.4, 0.1, kInf)},
    {"yeoh", cmf::Yeoh(0.5, -0.05, 0.01, kInf)},
    {"gent", cmf::Gent(1.0, 10.0, kInf)},
    {"arruda_boyce", cmf::ArrudaBoyce(1.0, 5.0, kInf)},   // Pade inverse Langevin (the default)
    {"arruda_boyce_series", cmf::ArrudaBoyce(1.0, 5.0, kInf, cmf::InverseLangevin::Series)},
    {"ogden", cmf::Ogden({0.63, 0.0012, -0.01}, {1.3, 5.0, -2.0}, kInf)},
  };
}

// beta_a = lambda_a dPsi/dlambda_a up to an isotropic term, for principal
// stretches with lambda_1 lambda_2 lambda_3 = 1 (the appendix's principal form):
//   invariant models  beta_a = 2 Psi_1 lambda_a^2 - 2 Psi_2 lambda_a^{-2},
//   Ogden             beta_a = sum_r mu_r lambda_a^alpha_r.
// Principal stress differences are sigma_a - sigma_b = beta_a - beta_b.
std::array<double, 3> Beta(const cmf::MixedMaterial &material, const std::array<double, 3> &lam)
{
  const double I1 = lam[0] * lam[0] + lam[1] * lam[1] + lam[2] * lam[2];
  return std::visit([&](const auto &m) -> std::array<double, 3>
  {
    using M = std::decay_t<decltype(m)>;
    std::array<double, 3> beta{};
    if constexpr (cmf::is_small_strain<M>::value)
    {
      // linear_elastic shares the variant but is not one of the rubber models
      // of this test (tests/test_linear_elasticity.cpp, tests/test_mixed.cpp).
      MFEM_ABORT("Beta: closed forms in the principal stretches need a finite-strain model");
    }
    else if constexpr (std::is_same_v<M, cmf::Ogden>)
    {
      for (int a = 0; a < 3; a++)
        for (int r = 0; r < m.terms; r++) { beta[a] += m.mu[r] * std::pow(lam[a], m.alpha[r]); }
    }
    else
    {
      double psi1 = 0.0, psi2 = 0.0;
      if constexpr (std::is_same_v<M, cmf::IsoNeoHookean>) { psi1 = 0.5 * m.mu; }
      else if constexpr (std::is_same_v<M, cmf::MooneyRivlin>) { psi1 = m.c1; psi2 = m.c2; }
      else { psi1 = m.DPsiDI1(I1); }
      for (int a = 0; a < 3; a++)
      {
        beta[a] = 2.0 * psi1 * lam[a] * lam[a] - 2.0 * psi2 / (lam[a] * lam[a]);
      }
    }
    return beta;
  }, material);
}

struct Analytic
{
  std::string label;
  std::array<double, 3> lam;
  std::array<double, 3> sigma; // principal Cauchy stresses
  double p;                    // mean stress = the pressure of the mixed formulation
  double vonmises;
};

// Principal stretches with one traction-free principal direction.
Analytic Homogeneous(const std::string &label, const cmf::MixedMaterial &material,
                     const std::array<double, 3> &lam, int free_axis)
{
  Analytic a;
  a.label = label;
  a.lam = lam;
  const std::array<double, 3> beta = Beta(material, lam);
  for (int i = 0; i < 3; i++) { a.sigma[i] = beta[i] - beta[free_axis]; }
  a.p = (a.sigma[0] + a.sigma[1] + a.sigma[2]) / 3.0;
  double s = 0.0;
  for (int i = 0; i < 3; i++) { s += (a.sigma[i] - a.p) * (a.sigma[i] - a.p); }
  a.vonmises = std::sqrt(1.5 * s);
  return a;
}

Analytic PlaneStrain(const cmf::MixedMaterial &m, double l)
{
  return Homogeneous("plane strain", m, {l, 1.0 / l, 1.0}, 1);
}
Analytic Uniaxial(const cmf::MixedMaterial &m, double l)
{
  return Homogeneous("uniaxial", m, {l, 1.0 / std::sqrt(l), 1.0 / std::sqrt(l)}, 1);
}
Analytic Equibiaxial(const cmf::MixedMaterial &m, double l)
{
  return Homogeneous("equibiaxial", m, {l, l, 1.0 / (l * l)}, 2);
}
Analytic PureShear(const cmf::MixedMaterial &m, double l)
{
  return Homogeneous("pure shear", m, {l, 1.0, 1.0 / l}, 2);
}

Mat3 CodeCauchyStress(const cmf::MixedMaterial &material, const Mat3 &H, double p)
{
  return std::visit([&](const auto &m)
  {
    return cmf::QPointMixedCauchyStress<std::decay_t<decltype(m)>, 3>(m, H, p);
  }, material);
}

// With F = diag(lambda_a) and p the analytic mean stress, the code's
// sigma = J^{-1} P_iso F^T + p I must be diag(sigma_a).
void MaterialPointTest(const std::string &name, const cmf::MixedMaterial &material,
                       const Analytic &a)
{
  Mat3 H;
  for (int i = 0; i < 3; i++) { H(i, i) = a.lam[i] - 1.0; }
  const Mat3 sigma = CodeCauchyStress(material, H, a.p);
  double err = 0.0, scale = 0.0;
  for (int i = 0; i < 3; i++)
  {
    scale = std::max(scale, std::abs(a.sigma[i]));
    for (int j = 0; j < 3; j++)
    {
      err = std::max(err, std::abs(sigma(i, j) - (i == j ? a.sigma[i] : 0.0)));
    }
  }
  std::printf("  %-13s %-13s sigma = (% .6f, % .6f, % .6f)  p = % .6f  vm = %.6f  error %.1e\n",
              name.c_str(), a.label.c_str(), a.sigma[0], a.sigma[1], a.sigma[2], a.p,
              a.vonmises, err);
  CHECK_MSG(err <= 1e-12 * scale, name + " " + a.label + " material-point stress (" +
            std::to_string(err) + ")");
}

// Simple shear F = I + gamma e1 (x) e2: principal stretches lambda, 1/lambda, 1
// with gamma = lambda - 1/lambda and in-plane principal directions
// (lambda, 1)/|.| and (-1, lambda)/|.|. The deviatoric stress of the code must
// equal dev(sum_a beta_a n_a (x) n_a), and sigma_11 - sigma_22 = gamma sigma_12.
void SimpleShearTest(const std::string &name, const cmf::MixedMaterial &material, double gamma)
{
  const double l = 0.5 * (gamma + std::sqrt(gamma * gamma + 4.0));
  const std::array<double, 3> beta = Beta(material, {l, 1.0 / l, 1.0});
  const double nrm = std::sqrt(1.0 + l * l);
  const double n1[3] = {l / nrm, 1.0 / nrm, 0.0}, n2[3] = {-1.0 / nrm, l / nrm, 0.0},
               n3[3] = {0.0, 0.0, 1.0};
  Mat3 spectral;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    {
      spectral(i, j) = beta[0] * n1[i] * n1[j] + beta[1] * n2[i] * n2[j] + beta[2] * n3[i] * n3[j];
    }
  const Mat3 exact = cmf::dev(spectral);
  Mat3 H;
  H(0, 1) = gamma;
  const Mat3 code = cmf::dev(CodeCauchyStress(material, H, 0.0));
  double err = 0.0, scale = 0.0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    {
      err = std::max(err, std::abs(code(i, j) - exact(i, j)));
      scale = std::max(scale, std::abs(exact(i, j)));
    }
  const double universal = std::abs((code(0, 0) - code(1, 1)) - gamma * code(0, 1));
  std::printf("  %-13s simple shear  sigma'_12 = % .6f  sigma'_11 = % .6f  sigma'_22 = % .6f  "
              "sigma'_33 = % .6f  error %.1e  (s11 - s22 - gamma s12: %.1e)\n",
              name.c_str(), code(0, 1), code(0, 0), code(1, 1), code(2, 2), err, universal);
  CHECK_MSG(err <= 1e-12 * scale, name + " simple shear deviatoric stress (" + std::to_string(err) + ")");
  CHECK_MSG(universal <= 1e-12 * scale, name + " simple shear universal relation");
}

// Attributes 1 and 2 for the faces X_1 = 0 and X_1 = L, 3 for the rest,
// independent of the mesh generator's numbering.
void LabelEndFaces(mfem::ParMesh &mesh, double L)
{
  for (int b = 0; b < mesh.GetNBE(); b++)
  {
    mfem::Array<int> verts;
    mesh.GetBdrElementVertices(b, verts);
    double x = 0.0;
    for (int v : verts) { x += mesh.GetVertex(v)[0]; }
    x /= verts.Size();
    int attr = 3;
    if (x < 1e-12) { attr = 1; }
    else if (x > L - 1e-12) { attr = 2; }
    mesh.GetBdrElement(b)->SetAttribute(attr);
  }
  mesh.SetAttributes();
}

struct FEResult
{
  double u_err = 0.0, p_err = 0.0;
  double nodal_err = 0.0, elem_err = 0.0, qp_err = 0.0; // over all quadrature quantities
  cmf::QuasiStaticReport report;
  int ndofs = 0;
};

// Expected packed values of the quadrature quantities of a homogeneous state.
struct Expected
{
  std::vector<std::pair<std::string, std::vector<double>>> values;
};

double ExpectedEnergy(const cmf::MixedMaterial &material, const Mat3 &F)
{
  return std::visit([&](const auto &m) { return m.EnergyIso(F); }, material);
}

// The largest deviation of the nodal and element presentations at `point`
// and of the raw quadrature values from the expected packed values.
void PresentationErrors(const cmf::FieldRegistry &fields, const Expected &x,
                        const std::vector<double> &point, double &nodal, double &elem, double &qp)
{
  nodal = elem = qp = 0.0;
  for (const auto &kv : x.values)
  {
    const std::vector<double> &want = kv.second;
    const std::vector<double> gn = cmf::ProbeVector(fields.Get(kv.first), point);
    const std::vector<double> ge = cmf::ProbeVector(fields.Get(kv.first + "_elem"), point);
    const mfem::QuadratureFunction &qf = fields.GetQ(kv.first + "_qp");
    for (std::size_t c = 0; c < want.size(); c++)
    {
      nodal = std::max(nodal, std::abs(gn[c] - want[c]));
      elem = std::max(elem, std::abs(ge[c] - want[c]));
    }
    for (int i = 0; i < qf.Size(); i++) { qp = std::max(qp, std::abs(qf(i) - want[i % want.size()])); }
  }
}

// Exact packed quantities of a diagonal state: Cauchy (xx, yy, zz, xy, yz,
// xz), PK1 and F (row-major) with P_aa = sigma_a / lambda_a, J = 1, von Mises,
// and the energy density.
Expected ExpectedDiagonal(const cmf::MixedMaterial &material, const Analytic &a)
{
  Expected x;
  x.values.push_back({"cauchy_stress", {a.sigma[0], a.sigma[1], a.sigma[2], 0.0, 0.0, 0.0}});
  std::vector<double> P(9, 0.0), F(9, 0.0);
  Mat3 Fm;
  for (int i = 0; i < 3; i++)
  {
    P[4 * i] = a.sigma[i] / a.lam[i];
    F[4 * i] = a.lam[i];
    Fm(i, i) = a.lam[i];
  }
  x.values.push_back({"pk1_stress", P});
  x.values.push_back({"deformation_gradient", F});
  x.values.push_back({"jacobian", {1.0}});
  x.values.push_back({"vonmises", {a.vonmises}});
  x.values.push_back({"energy_density", {ExpectedEnergy(material, Fm)}});
  return x;
}

// Mixed u-p solve on a unit box (Q2/Q1), affine displacement on the end
// faces X_1 = 0, L, lateral faces free; compared with the analytic state.
FEResult SolveHomogeneous(const cmf::MixedMaterial &material, const Analytic &a, int dim,
                          int load_steps)
{
  cmf::AppConfig cfg;
  cfg.formulation = "mixed";
  cfg.mesh.cartesian = true;
  cfg.mesh.box.dim = dim;
  cfg.mesh.box.nx = cfg.mesh.box.ny = dim == 2 ? 4 : 2;
  cfg.mesh.box.nz = dim == 2 ? 1 : 2;
  cfg.mesh.box.element = dim == 2 ? "quad" : "hex";
  cfg.mesh.order = 2;
  cfg.output.fields = {"displacement", "pressure", "cauchy_stress", "pk1_stress",
                       "deformation_gradient", "jacobian", "vonmises", "energy_density"};
  cfg.output.quadrature_at = {"nodes", "elements", "quadrature_points"};
  cfg.output.nodal_projection = dim == 2 ? "projected" : "averaged";
  cfg.solver.load_steps = load_steps;
  cfg.solver.newton.rtol = 1e-12;
  cfg.solver.newton.atol = 1e-14;
  cfg.solver.newton.max_it = 40;
  cfg.solver.newton.print_level = 0;
  cfg.solver.linear.rtol = 1e-13;
  cfg.solver.linear.max_it = 400;
  cfg.solver.linear.krylov_dim = 100;
  cfg.solver.linear.inner_rtol = 1e-4;
  cfg.solver.linear.inner_max_it = 100;

  std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  LabelEndFaces(*pmesh, 1.0);
  cmf::MixedSolidMechanicsTL physics(*pmesh, cfg, material);
  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::DenseMatrix G(dim);
  G = 0.0;
  for (int i = 0; i < dim; i++) { G(i, i) = a.lam[i] - 1.0; }
  cmf::AffineVectorCoefficient exact_u(zero, G);
  physics.AddDirichlet({1, 2}, exact_u);
  physics.Finalize();

  std::unique_ptr<mfem::Solver> linear = physics.MakeLinearSolver(cfg.solver.linear);
  mfem::Vector x(physics.Height());
  x = 0.0;
  FEResult r;
  r.report = cmf::SolveQuasiStatic(physics, *linear, cfg.solver, x);
  r.ndofs = int(physics.GlobalTrueVSize());
  physics.UpdateFields(x);

  mfem::ParGridFunction iu(&physics.DisplacementSpace());
  iu.ProjectCoefficient(exact_u);
  iu -= physics.Displacement();
  r.u_err = iu.Normlinf();
  mfem::ConstantCoefficient pc(a.p);
  mfem::ParGridFunction ip(&physics.PressureSpace());
  ip.ProjectCoefficient(pc);
  ip -= physics.Pressure();
  r.p_err = ip.Normlinf();
  cmf::FieldRegistry fields;
  physics.RegisterFields(fields);
  const std::vector<double> center(dim, 0.5);
  PresentationErrors(fields, ExpectedDiagonal(material, a), center, r.nodal_err, r.elem_err, r.qp_err);
  return r;
}

void FETest(const Case &c, const Analytic &a, int dim, int load_steps)
{
  const FEResult r = SolveHomogeneous(c.material, a, dim, load_steps);
  const cmf::NewtonReport &last = r.report.steps.back().newton;
  std::printf("  %-13s %-13s dofs %5d: errors u %.1e p %.1e, quantities nodal %.1e elem %.1e qp %.1e, newton its %d, |R|:",
              c.name.c_str(), a.label.c_str(), r.ndofs, r.u_err, r.p_err,
              r.nodal_err, r.elem_err, r.qp_err, last.iterations);
  for (const cmf::NewtonIteration &it : last.history)
  {
    std::printf(" %.1e%s", it.residual, it.iteration > 0 && it.alpha < 1.0 ? "*" : "");
  }
  std::printf("\n");
  const double sscale = std::max(1.0, std::abs(a.vonmises));
  CHECK_MSG(r.report.converged, c.name + " " + a.label + " converged");
  CHECK_MSG(r.u_err <= 1e-9, c.name + " " + a.label + " displacement (" + std::to_string(r.u_err) + ")");
  CHECK_MSG(r.p_err <= 1e-8 * sscale, c.name + " " + a.label + " pressure (" + std::to_string(r.p_err) + ")");
  CHECK_MSG(r.nodal_err <= 1e-8 * sscale, c.name + " " + a.label + " nodal presentations (" + std::to_string(r.nodal_err) + ")");
  CHECK_MSG(r.elem_err <= 1e-8 * sscale, c.name + " " + a.label + " element presentations (" + std::to_string(r.elem_err) + ")");
  CHECK_MSG(r.qp_err <= 1e-8 * sscale, c.name + " " + a.label + " quadrature-point values (" + std::to_string(r.qp_err) + ")");
}

// Plane-stress sheet states: the affine in-plane displacement gradient H, the
// faces that carry it, and the exact Cauchy stress (sigma_i3 = 0) and
// thickness stretch.
struct SheetState
{
  std::string label;
  tensor<double, 2, 2> H;
  std::vector<int> dirichlet_attrs; // 1, 2: X1 = 0, L; 3: the rest
  Mat3 sigma;
  double lambda3;
};

SheetState DiagonalSheet(const std::string &label, const Analytic &a, const std::vector<int> &attrs)
{
  SheetState s;
  s.label = label;
  s.H(0, 0) = a.lam[0] - 1.0;
  s.H(1, 1) = a.lam[1] - 1.0;
  s.dirichlet_attrs = attrs;
  for (int i = 0; i < 3; i++) { s.sigma(i, i) = a.sigma[i]; }
  s.lambda3 = a.lam[2];
  return s;
}

// Thin-sheet simple shear (sigma_33 = 0): sigma = sum_a beta_a n_a (x) n_a - beta_3 I.
SheetState SimpleShearSheet(const cmf::MixedMaterial &material, double gamma)
{
  const double l = 0.5 * (gamma + std::sqrt(gamma * gamma + 4.0));
  const std::array<double, 3> beta = Beta(material, {l, 1.0 / l, 1.0});
  const double nrm = std::sqrt(1.0 + l * l);
  const double n1[3] = {l / nrm, 1.0 / nrm, 0.0}, n2[3] = {-1.0 / nrm, l / nrm, 0.0},
               n3[3] = {0.0, 0.0, 1.0};
  SheetState s;
  s.label = "simple shear";
  s.H(0, 1) = gamma;
  s.dirichlet_attrs = {1, 2, 3};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    {
      s.sigma(i, j) = beta[0] * n1[i] * n1[j] + beta[1] * n2[i] * n2[j] +
                      beta[2] * n3[i] * n3[j] - (i == j ? beta[2] : 0.0);
    }
  s.lambda3 = 1.0;
  return s;
}

// Expected packed quantities of a sheet state with J = 1: P = sigma F^{-T}.
Expected ExpectedSheet(const cmf::MixedMaterial &material, const SheetState &s)
{
  Mat3 F = cmf::I<3>();
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) { F(i, j) += s.H(i, j); }
  F(2, 2) = s.lambda3;
  const Mat3 P = s.sigma * cmf::transpose(cmf::inv(F));
  Expected x;
  x.values.push_back({"cauchy_stress", {s.sigma(0, 0), s.sigma(1, 1), s.sigma(2, 2),
                                        s.sigma(0, 1), s.sigma(1, 2), s.sigma(0, 2)}});
  std::vector<double> Pv(9), Fv(9);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) { Pv[3 * i + j] = P(i, j); Fv[3 * i + j] = F(i, j); }
  x.values.push_back({"pk1_stress", Pv});
  x.values.push_back({"deformation_gradient", Fv});
  x.values.push_back({"jacobian", {1.0}});
  x.values.push_back({"vonmises", {cmf::VonMises(s.sigma)}});
  x.values.push_back({"energy_density", {ExpectedEnergy(material, F)}});
  x.values.push_back({"thickness_stretch", {s.lambda3}});
  return x;
}

struct SheetResult
{
  double u_err = 0.0, nodal_err = 0.0, elem_err = 0.0, qp_err = 0.0;
  cmf::QuasiStaticReport report;
};

// Displacement formulation with the plane-stress adapter on a 4x4 Q2 square.
SheetResult SolveSheet(const cmf::MixedMaterial &base, const SheetState &s, int load_steps)
{
  const cmf::Material material = std::visit([](const auto &m) -> cmf::Material
  { return cmf::PlaneStress<std::decay_t<decltype(m)>>(m); }, base);
  cmf::AppConfig cfg;
  cfg.plane = "stress";
  cfg.mesh.cartesian = true;
  cfg.mesh.box.nx = cfg.mesh.box.ny = 4;
  cfg.mesh.order = 2;
  cfg.output.fields = {"displacement", "cauchy_stress", "pk1_stress", "deformation_gradient",
                       "jacobian", "vonmises", "energy_density", "thickness_stretch"};
  cfg.output.quadrature_at = {"nodes", "elements", "quadrature_points"};
  cfg.output.nodal_projection = s.label == "simple shear" ? "projected" : "averaged";
  cfg.solver.load_steps = load_steps;
  cfg.solver.newton.rtol = 1e-12;
  cfg.solver.newton.atol = 1e-14;
  cfg.solver.newton.max_it = 40;
  cfg.solver.newton.print_level = 0;
  cfg.solver.linear.rtol = 1e-13;
  cfg.solver.linear.max_it = 400;
  cfg.solver.linear.krylov_dim = 100;

  std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
  LabelEndFaces(*pmesh, 1.0);
  cmf::SolidMechanicsTL physics(*pmesh, cfg, material);
  mfem::Vector zero(2);
  zero = 0.0;
  mfem::DenseMatrix G(2);
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) { G(i, j) = s.H(i, j); }
  cmf::AffineVectorCoefficient exact_u(zero, G);
  physics.AddDirichlet(s.dirichlet_attrs, exact_u);
  physics.Finalize();
  std::unique_ptr<mfem::Solver> linear = physics.MakeLinearSolver(cfg.solver.linear);
  mfem::Vector x(physics.Height());
  x = 0.0;
  SheetResult r;
  r.report = cmf::SolveQuasiStatic(physics, *linear, cfg.solver, x);
  physics.UpdateFields(x);

  mfem::ParGridFunction iu(&physics.DisplacementSpace());
  iu.ProjectCoefficient(exact_u);
  iu -= physics.Displacement();
  r.u_err = iu.Normlinf();
  cmf::FieldRegistry fields;
  physics.RegisterFields(fields);
  const std::vector<double> center(2, 0.5);
  PresentationErrors(fields, ExpectedSheet(base, s), center, r.nodal_err, r.elem_err, r.qp_err);
  return r;
}

void SheetTest(const Case &c, const SheetState &s, int load_steps)
{
  const SheetResult r = SolveSheet(c.material, s, load_steps);
  const cmf::NewtonReport &last = r.report.steps.back().newton;
  std::printf("  %-13s %-13s sigma = (% .4f, % .4f, % .4f; s12 % .4f) l3 %.4f: errors u %.1e, quantities nodal %.1e elem %.1e qp %.1e, newton its %d\n",
              c.name.c_str(), s.label.c_str(), s.sigma(0, 0), s.sigma(1, 1), s.sigma(2, 2), s.sigma(0, 1),
              s.lambda3, r.u_err, r.nodal_err, r.elem_err, r.qp_err, last.iterations);
  double sscale = 1.0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) { sscale = std::max(sscale, std::abs(s.sigma(i, j))); }
  CHECK_MSG(r.report.converged, c.name + " sheet " + s.label + " converged");
  CHECK_MSG(r.u_err <= 1e-9, c.name + " sheet " + s.label + " displacement (" + std::to_string(r.u_err) + ")");
  CHECK_MSG(r.nodal_err <= 1e-8 * sscale, c.name + " sheet " + s.label + " nodal presentations (" + std::to_string(r.nodal_err) + ")");
  CHECK_MSG(r.elem_err <= 1e-8 * sscale, c.name + " sheet " + s.label + " element presentations (" + std::to_string(r.elem_err) + ")");
  CHECK_MSG(r.qp_err <= 1e-8 * sscale, c.name + " sheet " + s.label + " quadrature-point values (" + std::to_string(r.qp_err) + ")");
}

// Gmsh workflow: the example meshes of apps/mesh, their physical-group names
// (checked geometrically), and a solve driven entirely by an AppConfig that
// names the faces, as the YAML inputs do.
void MeshFileTest()
{
  struct Face { const char *name; int axis; double coord; };
  struct Case { const char *file; int dim; std::vector<Face> faces; };
  const std::vector<Case> cases = {
    {"apps/mesh/square.msh", 2, {{"bottom", 1, 0.0}, {"right", 0, 1.0}, {"top", 1, 1.0}, {"left", 0, 0.0}}},
    {"apps/mesh/cube.msh", 3, {{"bottom", 2, 0.0}, {"front", 1, 0.0}, {"right", 0, 1.0},
                                {"back", 1, 1.0}, {"left", 0, 0.0}, {"top", 2, 1.0}}},
  };
  for (const Case &c : cases)
  {
    cmf::MeshConfig mcfg;
    mcfg.file = c.file;
    mcfg.order = 2;
    std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, mcfg);
    std::printf("  %s: dim %d, %d elements, boundary attributes %s\n", c.file, pmesh->Dimension(),
                pmesh->GetNE(), cmf::DescribeAttributes(*pmesh, true).c_str());
    CHECK_MSG(pmesh->Dimension() == c.dim, std::string(c.file) + " dimension");
    CHECK_MSG(pmesh->attribute_sets.AttributeSetExists("domain"), std::string(c.file) + " has 'domain'");
    for (const Face &f : c.faces)
    {
      cmf::BoundaryCondition bc;
      bc.attr_names = {f.name};
      const std::vector<int> attrs = cmf::ResolveBoundaryAttributes(*pmesh, bc, f.name);
      int count = 0;
      double worst = 0.0;
      mfem::Array<int> verts;
      for (int b = 0; b < pmesh->GetNBE(); b++)
      {
        if (std::find(attrs.begin(), attrs.end(), pmesh->GetBdrAttribute(b)) == attrs.end()) { continue; }
        pmesh->GetBdrElementVertices(b, verts);
        for (int v : verts) { worst = std::max(worst, std::abs(pmesh->GetVertex(v)[f.axis] - f.coord)); }
        count++;
      }
      CHECK_MSG(count > 0 && worst <= 1e-12, std::string(c.file) + " group '" + f.name +
                "' lies on its face (" + std::to_string(count) + " faces, offset " + std::to_string(worst) + ")");
    }
  }

  // Plane-strain extension of the square and uniaxial tension of the cube from
  // an AppConfig with named faces, neo-Hookean, mixed formulation.
  const cmf::MixedMaterial nh = cmf::IsoNeoHookean(1.0, kInf);
  for (int dim = 2; dim <= 3; dim++)
  {
    const Analytic a = dim == 2 ? PlaneStrain(nh, 1.6) : Uniaxial(nh, 1.5);
    cmf::AppConfig cfg;
    cfg.formulation = "mixed";
    cfg.mesh.file = dim == 2 ? "apps/mesh/square.msh" : "apps/mesh/cube.msh";
    cfg.mesh.order = 2;
    cfg.material.model = "iso_neo_hookean";
    cfg.material.mu = 1.0;
    cfg.material.incompressible = true;
    cmf::BoundaryCondition bc;
    bc.attr_names = {"left", "right"};
    for (int i = 0; i < dim; i++)
    {
      // u_i = (lambda_i - 1) X_i, written as the YAML expression string.
      char buf[64];
      std::snprintf(buf, sizeof(buf), "%.17g*%c", a.lam[i] - 1.0, "xyz"[i]);
      bc.expression.push_back(buf);
    }
    cfg.bcs.dirichlet.push_back(bc);
    cfg.output.fields = {"displacement", "pressure"};
    cfg.solver.load_steps = dim == 2 ? 6 : 5;
    cfg.solver.newton.rtol = 1e-12;
    cfg.solver.newton.atol = 1e-14;
    cfg.solver.newton.max_it = 40;
    cfg.solver.newton.print_level = 0;
    cfg.solver.linear.rtol = 1e-13;
    cfg.solver.linear.max_it = 400;
    cfg.solver.linear.krylov_dim = 100;
    cfg.solver.linear.inner_rtol = 1e-4;
    cfg.solver.linear.inner_max_it = 100;
    std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    std::unique_ptr<cmf::SolidProblem> physics = cmf::MakeSolidProblem(*pmesh, cfg);
    physics->Finalize();
    std::unique_ptr<mfem::Solver> linear = physics->MakeLinearSolver(cfg.solver.linear);
    mfem::Vector x(physics->Height());
    x = 0.0;
    const cmf::QuasiStaticReport report = cmf::SolveQuasiStatic(*physics, *linear, cfg.solver, x);
    physics->UpdateFields(x);
    mfem::Vector zero(dim);
    zero = 0.0;
    mfem::DenseMatrix G(dim);
    G = 0.0;
    for (int i = 0; i < dim; i++) { G(i, i) = a.lam[i] - 1.0; }
    cmf::AffineVectorCoefficient exact_u(zero, G);
    mfem::ParGridFunction iu(&physics->DisplacementSpace());
    iu.ProjectCoefficient(exact_u);
    iu -= physics->Displacement();
    cmf::FieldRegistry fields;
    physics->RegisterFields(fields);
    const std::vector<double> center(dim, 0.5);
    const double p_err = std::abs(cmf::ProbeVector(fields.Get("pressure"), center)[0] - a.p);
    std::printf("  %s with faces [left, right]: errors u %.1e p %.1e, newton its %d\n", cfg.mesh.file.c_str(),
                iu.Normlinf(), p_err, report.steps.back().newton.iterations);
    CHECK_MSG(report.converged, cfg.mesh.file + " converged");
    CHECK_MSG(iu.Normlinf() <= 1e-9, cfg.mesh.file + " displacement (" + std::to_string(iu.Normlinf()) + ")");
    CHECK_MSG(p_err <= 1e-8 * std::max(1.0, std::abs(a.p)), cfg.mesh.file + " pressure (" + std::to_string(p_err) + ")");
  }
  // A misspelt group name is reported with what the mesh offers.
  {
    cmf::AppConfig cfg;
    cfg.mesh.file = "apps/mesh/square.msh";
    cfg.mesh.order = 2;
    cfg.material.model = "neo_hookean";
    cfg.material.E = 1.0;
    cfg.material.nu = 0.3;
    cmf::BoundaryCondition bc;
    bc.attr_names = {"lefft"};
    bc.expression = {"0", "0"};
    cfg.bcs.dirichlet.push_back(bc);
    std::unique_ptr<mfem::ParMesh> pmesh = cmf::BuildParMesh(MPI_COMM_WORLD, cfg.mesh);
    CHECK_THROWS(cmf::MakeSolidProblem(*pmesh, cfg), cmf::ConfigError,
                 "bcs.dirichlet[0]: the mesh has no boundary physical group named 'lefft' "
                 "(boundary attributes: 1 (bottom), 2 (right), 3 (top), 4 (left))");
  }
}

} // namespace

int main(int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  if (mfem::Mpi::WorldSize() != 1)
  {
    if (mfem::Mpi::Root()) { std::cout << "test_homogeneous is a serial test" << std::endl; }
    return 1;
  }
  const std::vector<Case> models = Models();

  std::cout << "gmsh meshes of apps/mesh: physical groups and a named-face solve" << std::endl;
  MeshFileTest();

  std::cout << "material-point closed forms (lambda = 1.8, gamma = 0.7)" << std::endl;
  for (const Case &c : models)
  {
    MaterialPointTest(c.name, c.material, PlaneStrain(c.material, 1.8));
    MaterialPointTest(c.name, c.material, Uniaxial(c.material, 1.8));
    MaterialPointTest(c.name, c.material, Equibiaxial(c.material, 1.4));
    MaterialPointTest(c.name, c.material, PureShear(c.material, 1.8));
    SimpleShearTest(c.name, c.material, 0.7);
  }

  std::cout << "mixed formulation: plane-strain extension, lambda = 1.6, 4x4 Q2/Q1 (nodal: projected)" << std::endl;
  for (const Case &c : models) { FETest(c, PlaneStrain(c.material, 1.6), 2, 6); }

  std::cout << "mixed formulation: uniaxial tension, lambda = 1.5, 2x2x2 Q2/Q1 (nodal: averaged)" << std::endl;
  for (const Case &c : models) { FETest(c, Uniaxial(c.material, 1.5), 3, 5); }

  std::cout << "plane stress (displacement formulation, 4x4 Q2): sheet states of Section 4" << std::endl;
  for (const Case &c : models)
  {
    SheetTest(c, DiagonalSheet("uniaxial", Uniaxial(c.material, 1.6), {1, 2}), 6);
    SheetTest(c, DiagonalSheet("equibiaxial", Equibiaxial(c.material, 1.3), {1, 2, 3}), 4);
    SheetTest(c, DiagonalSheet("pure shear", PureShear(c.material, 1.6), {1, 2, 3}), 4);
    // Shear increments of 0.05: each load step starts with the boundary moved
    // and the interior lagging, and the plane-stress energy (1/det F2D terms)
    // makes Newton leave the affine branch when a step shears a boundary
    // layer of elements by much more than that.
    SheetTest(c, SimpleShearSheet(c.material, 0.7), 14);
  }

  return cmf_test::Report("test_homogeneous");
}

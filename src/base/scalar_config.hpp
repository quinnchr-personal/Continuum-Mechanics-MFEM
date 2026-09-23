// YAML input schema of the scalar transport executable (apps/scalar_transport.cpp,
// physics/scalar_transport.hpp): the mesh, time, solver and output sections of
// the solid schema (base/config.hpp) plus the transport laws, the initial
// condition and the scalar boundary conditions. A top-level `physics:
// scalar_transport` key marks the input as this executable's.
//
//   physics: scalar_transport
//   mesh: { file: apps/mesh/square_tri.msh, serial_refine: 0, order: 3 }
//   transport:
//     unknown: c                     # name of the nodal field (default u)
//     capacity: 1.0                  # c(u): a number, or { value, slope, reference }: value + slope (u - reference)
//     conductivity: 0.01             # kappa(u): the same forms; positive at the reference
//     velocity: ["1", "0"]           # beta(x, y, z, t), one expression per space dimension; omit for none
//     convection: nonconservative    # nonconservative (beta . grad u, default) | conservative (-div(beta u))
//     reaction: 0.0                  # s: the term s u
//     source: "..."                  # f(x, y, z, t); omit for none
//     quadrature_order: 9            # rule of the kernel and the source (default 2 k + 3)
//   initial: "300"                   # u(x, y, z) at t = 0 (default 0); needs the time block
//   time: { t_final: 1.0, dt: 1.0e-3 }   # implicit Euler in physical time; absent: steady
//   bcs:
//     dirichlet: [ { attr: [left], name: ends, expression: "..." }, { point: [0, 0], expression: "0" } ]
//     flux:      [ { attr: [left], name: heated, expression: "7.5e5" } ]   # inward flux g = kappa du/dn
//   solver: { predictor, substep, newton, linear } as for the solid (linear.amg: scalar)
//   output:
//     paraview, fields ([u, u_exact, u_error, flux]), quadrature_at, nodal_projection, high_order,
//     probes, probe_every_step, every as for the solid, plus
//     exact: "..."                   # u_ex(x, y, z, t): errors per step, error_history.csv, the two fields
//     flows: true                    # the flow through every Dirichlet entry per step, flows.csv
#pragma once

#include <string>
#include <vector>

#include "base/config.hpp"

namespace cmf
{

// a(u) = value + slope (u - reference); a bare number in the input is slope 0.
struct LawConfig
{
  double value = 1.0;
  double slope = 0.0;
  double reference = 0.0;
};

struct TransportConfig
{
  std::string unknown = "u";
  LawConfig capacity;
  LawConfig conductivity;
  std::vector<std::string> velocity;   // expressions, one per space dimension; empty: none
  std::string convection = "nonconservative"; // nonconservative | conservative
  double reaction = 0.0;
  std::string source;                  // expression; empty: none
  int quadrature_order = 0;            // 0: 2 k + 3
};

// A Dirichlet entry (faces by attribute, or a point) or a flux entry (faces):
// one expression f(x, y, z, t), a schedule (the data is schedule(t) * f; the
// default rule of the solid schema), a name (flows output).
struct ScalarCondition
{
  std::string name;
  std::vector<int> attr;
  std::vector<std::string> attr_names;
  std::vector<double> point;           // Dirichlet only
  std::string expression;
  Schedule schedule;
  bool IsPoint() const { return !point.empty(); }
};

struct ScalarBCConfig
{
  std::vector<ScalarCondition> dirichlet;
  std::vector<ScalarCondition> flux;
};

struct ScalarAppConfig
{
  MeshConfig mesh;
  TransportConfig transport;
  std::string initial;                 // expression; empty: zero
  TimeConfig time;
  ScalarBCConfig bcs;
  SolverConfig solver;
  OutputConfig output;
  std::string exact;                   // expression; empty: no error output
  bool flows = false;
};

// Parse the schema above; unknown keys, missing keys and wrong types throw
// ConfigError with the full key path.
ScalarAppConfig ParseScalarConfig(const YAML::Node &root);
ScalarAppConfig LoadScalarConfig(const std::string &path);

} // namespace cmf

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace mfem;

#ifndef MFEM_USE_PETSC
#error "This driver requires MFEM built with PETSc."
#endif
#include <petscksp.h>

namespace
{

constexpr double kPi = 3.141592653589793238462643383279502884;

std::string DefaultValidationMode()
{
#ifdef ALE_VALIDATION_FIXED_MODE
   return ALE_VALIDATION_FIXED_MODE;
#else
   return "all";
#endif
}

bool HasFixedValidationMode()
{
#ifdef ALE_VALIDATION_FIXED_MODE
   return true;
#else
   return false;
#endif
}

std::string FixedValidationMode()
{
#ifdef ALE_VALIDATION_FIXED_MODE
   return ALE_VALIDATION_FIXED_MODE;
#else
   return "";
#endif
}

enum class ValidationSelection
{
   All,
   Stability,
   Convergence,
   Accuracy
};

enum class AleMapKind
{
   Identity,
   Scale20Pi,
   Scale10Pi,
   AccuracyA,
   AccuracyB
};

enum class FieldModel
{
   Zero,
   StabilityInitial,
   ConvergenceExact,
   ConvergenceForcing,
   AccuracyExact,
   AccuracyForcing
};

struct DriverParams
{
   string validation = DefaultValidationMode();
   string mesh_file = "Mesh/unit_square.msh";
   string output_path = "ParaView/ale_validation_be";

   int serial_ref_levels = 3;
   int par_ref_levels = 0;

   int stability_order = 1;
   int convergence_order = 2;
   int accuracy_order = 1;

   string stability_dt_list = "1.0e-4,1.0e-3,1.0e-2,2.0e-2";
   string convergence_dt_list = "0.05,0.01,0.005,0.001";
   string accuracy_dt_list = "0.1,0.05,0.02,0.01,0.005";

   int linear_max_iter = 400;
   double linear_rel_tol = 1.0e-10;
   double linear_abs_tol = 0.0;
   int linear_print_level = 0;

   bool save_paraview = true;
   int paraview_every = 0; // 0 => save only initial/final snapshots per run
};

struct ScenarioSpec
{
   string name;
   AleMapKind map_kind = AleMapKind::Identity;
   double alpha = 0.1;
   double t_final = 1.0;
   int order = 1;

   FieldModel initial_model = FieldModel::Zero;
   FieldModel forcing_model = FieldModel::Zero;
   FieldModel exact_model = FieldModel::Zero;

   bool zero_dirichlet = true;
   bool record_l2_history = false;
};

struct SimulationResult
{
   vector<double> time;
   vector<double> l2;
   double final_error_l2 = 0.0;
};

void ToLowerInPlace(string &s)
{
   transform(s.begin(), s.end(), s.begin(),
             [](unsigned char c) { return static_cast<char>(tolower(c)); });
}

ValidationSelection ParseValidationSelection(string s)
{
   ToLowerInPlace(s);
   if (s == "all") { return ValidationSelection::All; }
   if (s == "stability") { return ValidationSelection::Stability; }
   if (s == "convergence") { return ValidationSelection::Convergence; }
   if (s == "accuracy") { return ValidationSelection::Accuracy; }
   throw runtime_error("validation must be one of: all, stability, convergence, accuracy");
}

vector<double> ParseDoubleList(const string &text)
{
   vector<double> values;
   string token;
   std::stringstream ss(text);
   while (std::getline(ss, token, ','))
   {
      // also allow whitespace-separated single entry chunks
      std::stringstream inner(token);
      double v = 0.0;
      while (inner >> v)
      {
         values.push_back(v);
      }
   }
   if (values.empty())
   {
      throw runtime_error("Expected at least one time step value in list: " + text);
   }
   for (double v : values)
   {
      if (!(v > 0.0))
      {
         throw runtime_error("Time step values must be > 0.");
      }
   }
   sort(values.begin(), values.end(), std::greater<double>());
   return values;
}

string SanitizeToken(string s)
{
   for (char &c : s)
   {
      const unsigned char uc = static_cast<unsigned char>(c);
      if (!std::isalnum(uc))
      {
         c = '_';
      }
   }

   // Collapse repeated underscores for cleaner collection names.
   string out;
   out.reserve(s.size());
   bool prev_us = false;
   for (char c : s)
   {
      const bool is_us = (c == '_');
      if (!(is_us && prev_us))
      {
         out.push_back(c);
      }
      prev_us = is_us;
   }
   while (!out.empty() && out.back() == '_') { out.pop_back(); }
   if (out.empty()) { out = "run"; }
   return out;
}

string DtTag(const double dt)
{
   std::ostringstream oss;
   oss << std::scientific << std::setprecision(3) << dt;
   return SanitizeToken(oss.str());
}

void EnsureOutputDir(const string &path, const int myid)
{
   if (myid == 0)
   {
      filesystem::create_directories(path);
   }
   MPI_Barrier(MPI_COMM_WORLD);
}

void ValidateUnitSquareMesh(const ParMesh &pmesh, const double tol)
{
   double local_min[2] = {numeric_limits<double>::infinity(),
                          numeric_limits<double>::infinity()};
   double local_max[2] = {-numeric_limits<double>::infinity(),
                          -numeric_limits<double>::infinity()};

   for (int i = 0; i < pmesh.GetNV(); ++i)
   {
      const double *v = pmesh.GetVertex(i);
      local_min[0] = std::min(local_min[0], v[0]);
      local_min[1] = std::min(local_min[1], v[1]);
      local_max[0] = std::max(local_max[0], v[0]);
      local_max[1] = std::max(local_max[1], v[1]);
   }

   double global_min[2] = {0.0, 0.0};
   double global_max[2] = {0.0, 0.0};
   MPI_Allreduce(local_min, global_min, 2, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(local_max, global_max, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

   MFEM_VERIFY(std::abs(global_min[0]) <= tol &&
               std::abs(global_min[1]) <= tol &&
               std::abs(global_max[0] - 1.0) <= tol &&
               std::abs(global_max[1] - 1.0) <= tol,
               "Expected a unit-square mesh. Got x=[" << global_min[0] << ", "
               << global_max[0] << "], y=[" << global_min[1] << ", "
               << global_max[1] << "].");
}

void BuildAllBoundaryMarker(const ParMesh &pmesh, Array<int> &ess_bdr)
{
   MFEM_VERIFY(pmesh.bdr_attributes.Size() > 0, "Mesh must define boundary attributes.");
   ess_bdr.SetSize(pmesh.bdr_attributes.Max());
   ess_bdr = 1;
}

void BuildIntegrationRules(const int quad_order,
                           const IntegrationRule *irs[Geometry::NumGeom])
{
   for (int g = 0; g < Geometry::NumGeom; ++g)
   {
      irs[g] = &IntRules.Get(g, quad_order);
   }
}

class AleMap
{
public:
   explicit AleMap(const AleMapKind kind) : kind_(kind) { }

   const char *Name() const
   {
      switch (kind_)
      {
         case AleMapKind::Identity: return "fixed";
         case AleMapKind::Scale20Pi: return "scale_20pi";
         case AleMapKind::Scale10Pi: return "scale_10pi";
         case AleMapKind::AccuracyA: return "map_A";
         case AleMapKind::AccuracyB: return "map_B";
      }
      return "unknown";
   }

   void MapPoint(const Vector &xref, const double t, Vector &x) const
   {
      MFEM_VERIFY(xref.Size() == 2, "AleMap expects 2D points.");
      x.SetSize(2);

      const double xh = xref[0];
      const double yh = xref[1];
      switch (kind_)
      {
         case AleMapKind::Identity:
         {
            x = xref;
            return;
         }
         case AleMapKind::Scale20Pi:
         case AleMapKind::Scale10Pi:
         {
            const double omega = (kind_ == AleMapKind::Scale20Pi) ? (20.0 * kPi) : (10.0 * kPi);
            const double s = 2.0 - std::cos(omega * t);
            x[0] = s * xh;
            x[1] = s * yh;
            return;
         }
         case AleMapKind::AccuracyA:
         {
            const double amp = 0.5 * std::sin(kPi * t);
            const double gx = std::sin(kPi * xh * (1.0 - xh) * (xh - 0.5));
            const double gy = std::sin(kPi * yh * (1.0 - yh) * (yh - 0.5));
            x[0] = xh + amp * gx;
            x[1] = yh + amp * gy;
            return;
         }
         case AleMapKind::AccuracyB:
         {
            const double q = xh * (1.0 - xh) * yh * (1.0 - yh);
            const double amp = std::sin(kPi * t) * q;
            x[0] = xh + amp;
            x[1] = yh + amp;
            return;
         }
      }
   }

private:
   AleMapKind kind_;
};

class PrescribedMeshMotion
{
public:
   explicit PrescribedMeshMotion(ParMesh &pmesh)
      : pmesh_(pmesh)
   {
      if (pmesh_.GetNodes() == nullptr)
      {
         pmesh_.SetCurvature(1, false, pmesh_.SpaceDimension(), Ordering::byVDIM);
      }

      auto *nodes = dynamic_cast<ParGridFunction *>(pmesh_.GetNodes());
      MFEM_VERIFY(nodes != nullptr, "PrescribedMeshMotion requires ParGridFunction mesh nodes.");
      auto *nodes_fes = dynamic_cast<ParFiniteElementSpace *>(nodes->FESpace());
      MFEM_VERIFY(nodes_fes != nullptr, "PrescribedMeshMotion requires ParFiniteElementSpace nodes.");
      MFEM_VERIFY(nodes_fes->GetVDim() == 2,
                  "PrescribedMeshMotion currently supports only 2D meshes.");

      nodes_ = nodes;
      nodes_fes_ = nodes_fes;
      ref_nodes_ = std::make_unique<ParGridFunction>(nodes_fes_);
      mapped_nodes_ = std::make_unique<ParGridFunction>(nodes_fes_);
      mesh_velocity_ = std::make_unique<ParGridFunction>(nodes_fes_);

      *ref_nodes_ = *nodes_;
      *mapped_nodes_ = *nodes_;
      *mesh_velocity_ = 0.0;
   }

   void SetTime(const AleMap &map, const double t)
   {
      ComputeMappedNodes_(map, t, *mapped_nodes_);
      Vector disp(nodes_->Size());
      disp = *mapped_nodes_;
      disp -= *nodes_;
      pmesh_.MoveNodes(disp);
      *mesh_velocity_ = 0.0;
      current_time_ = t;
   }

   void AdvanceTo(const AleMap &map, const double t_new)
   {
      const double dt = t_new - current_time_;
      MFEM_VERIFY(dt >= 0.0, "Mesh motion time must be nondecreasing.");

      ComputeMappedNodes_(map, t_new, *mapped_nodes_);
      *mesh_velocity_ = *mapped_nodes_;
      *mesh_velocity_ -= *nodes_;

      Vector disp(nodes_->Size());
      disp = *mesh_velocity_;

      if (dt > 0.0)
      {
         *mesh_velocity_ /= dt;
      }
      else
      {
         *mesh_velocity_ = 0.0;
      }

      pmesh_.MoveNodes(disp);
      current_time_ = t_new;
   }

   ParGridFunction &Velocity() { return *mesh_velocity_; }
   const ParGridFunction &Velocity() const { return *mesh_velocity_; }
   double CurrentTime() const { return current_time_; }

private:
   void ComputeMappedNodes_(const AleMap &map,
                            const double t,
                            ParGridFunction &nodes_out) const
   {
      const int ndofs = nodes_fes_->GetNDofs();
      Vector xref(2), x(2);
      for (int i = 0; i < ndofs; ++i)
      {
         const int xvdof = nodes_fes_->DofToVDof(i, 0);
         const int yvdof = nodes_fes_->DofToVDof(i, 1);
         xref[0] = (*ref_nodes_)(xvdof);
         xref[1] = (*ref_nodes_)(yvdof);
         map.MapPoint(xref, t, x);
         nodes_out(xvdof) = x[0];
         nodes_out(yvdof) = x[1];
      }
   }

   ParMesh &pmesh_;
   ParGridFunction *nodes_ = nullptr;
   ParFiniteElementSpace *nodes_fes_ = nullptr;
   std::unique_ptr<ParGridFunction> ref_nodes_;
   std::unique_ptr<ParGridFunction> mapped_nodes_;
   std::unique_ptr<ParGridFunction> mesh_velocity_;
   double current_time_ = 0.0;
};

class DivergenceGridFunctionCoefficient : public Coefficient
{
public:
   explicit DivergenceGridFunctionCoefficient(ParGridFunction &v) : v_(v) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      T.SetIntPoint(&ip);
      grad_.SetSize(T.GetSpaceDim(), T.GetSpaceDim());
      v_.GetVectorGradient(T, grad_);
      double div = 0.0;
      const int d = std::min(grad_.Height(), grad_.Width());
      for (int i = 0; i < d; ++i)
      {
         div += grad_(i, i);
      }
      return div;
   }

private:
   ParGridFunction &v_;
   mutable DenseMatrix grad_;
};

class ScaledCoefficient : public Coefficient
{
public:
   ScaledCoefficient(Coefficient &base, const double scale)
      : base_(base), scale_(scale) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      return scale_ * base_.Eval(T, ip);
   }

private:
   Coefficient &base_;
   double scale_;
};

class StabilityInitialCoefficient : public Coefficient
{
public:
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x;
      T.Transform(ip, x);
      return 1600.0 * x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]);
   }
};

class ConvergenceExactCoefficient : public Coefficient
{
public:
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x;
      T.Transform(ip, x);
      const double t = GetTime();
      const double s = 2.0 - std::cos(10.0 * kPi * t);
      const double xi = x[0] / s;
      const double eta = x[1] / s;
      const double amp = 16.0 * (1.0 + 0.5 * std::sin(5.0 * kPi * t));
      return amp * xi * (1.0 - xi) * eta * (1.0 - eta);
   }
};

class ConvergenceForcingCoefficient : public Coefficient
{
public:
   explicit ConvergenceForcingCoefficient(const double alpha) : alpha_(alpha) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x;
      T.Transform(ip, x);
      const double t = GetTime();

      const double s = 2.0 - std::cos(10.0 * kPi * t);
      const double sdot = 10.0 * kPi * std::sin(10.0 * kPi * t);
      const double xi = x[0] / s;
      const double eta = x[1] / s;

      const double amp = 16.0 * (1.0 + 0.5 * std::sin(5.0 * kPi * t));
      const double amp_t = 40.0 * kPi * std::cos(5.0 * kPi * t);

      const double p = xi * (1.0 - xi) * eta * (1.0 - eta);
      const double pxi = (1.0 - 2.0 * xi) * eta * (1.0 - eta);
      const double peta = (1.0 - 2.0 * eta) * xi * (1.0 - xi);
      const double pxxi = -2.0 * eta * (1.0 - eta);
      const double petaeta = -2.0 * xi * (1.0 - xi);

      const double xi_t = -(sdot / s) * xi;
      const double eta_t = -(sdot / s) * eta;
      const double ut = amp_t * p + amp * (pxi * xi_t + peta * eta_t);

      const double lap = (amp / (s * s)) * (pxxi + petaeta);

      return ut - alpha_ * lap;
   }

private:
   double alpha_;
};

class AccuracyExactCoefficient : public Coefficient
{
public:
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x;
      T.Transform(ip, x);
      const double t = GetTime();
      const double dx = x[0] - 0.5;
      const double dy = x[1] - 0.5;
      const double q = 2.0 * dx * dx + 2.0 * dy * dy;
      return std::sin(t) * std::cos(q);
   }
};

class AccuracyForcingCoefficient : public Coefficient
{
public:
   explicit AccuracyForcingCoefficient(const double alpha) : alpha_(alpha) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x;
      T.Transform(ip, x);
      const double t = GetTime();
      const double dx = x[0] - 0.5;
      const double dy = x[1] - 0.5;
      const double r2 = dx * dx + dy * dy;
      const double q = 2.0 * r2;

      const double ut = std::cos(t) * std::cos(q);
      const double lap = -std::sin(t) * (16.0 * r2 * std::cos(q) + 8.0 * std::sin(q));
      return ut - alpha_ * lap;
   }

private:
   double alpha_;
};

Coefficient *SelectCoefficient(FieldModel model,
                               ConstantCoefficient &zero,
                               StabilityInitialCoefficient &stability_ic,
                               ConvergenceExactCoefficient &conv_exact,
                               ConvergenceForcingCoefficient &conv_force,
                               AccuracyExactCoefficient &acc_exact,
                               AccuracyForcingCoefficient &acc_force)
{
   switch (model)
   {
      case FieldModel::Zero: return &zero;
      case FieldModel::StabilityInitial: return &stability_ic;
      case FieldModel::ConvergenceExact: return &conv_exact;
      case FieldModel::ConvergenceForcing: return &conv_force;
      case FieldModel::AccuracyExact: return &acc_exact;
      case FieldModel::AccuracyForcing: return &acc_force;
   }
   return &zero;
}

void SetCoefficientTimeIfNeeded(Coefficient *coeff, const double t)
{
   if (coeff)
   {
      coeff->SetTime(t);
   }
}

bool IsMonotoneNonIncreasing(const vector<double> &vals, const double rel_tol = 1.0e-10)
{
   for (size_t i = 1; i < vals.size(); ++i)
   {
      const double limit = vals[i - 1] * (1.0 + rel_tol) + 1.0e-14;
      if (vals[i] > limit)
      {
         return false;
      }
   }
   return true;
}

unique_ptr<ParMesh> BuildUnitSquareParMesh(const DriverParams &params)
{
   auto mesh = std::make_unique<Mesh>(params.mesh_file.c_str(), 1, 1);
   for (int l = 0; l < params.serial_ref_levels; ++l)
   {
      mesh->UniformRefinement();
   }

   auto pmesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, *mesh);
   mesh.reset();
   for (int l = 0; l < params.par_ref_levels; ++l)
   {
      pmesh->UniformRefinement();
   }

   ValidateUnitSquareMesh(*pmesh, 1.0e-8);
   return pmesh;
}

void SolveTrueSystem(HypreParMatrix &A,
                     Vector &X,
                     Vector &B,
                     const DriverParams &params,
                     const int myid,
                     const string &tag,
                     const int step)
{
   PetscParMatrix A_petsc(MPI_COMM_WORLD, &A, Operator::PETSC_MATAIJ);
   PetscLinearSolver solver(A_petsc);
   solver.SetRelTol(params.linear_rel_tol);
   solver.SetAbsTol(params.linear_abs_tol);
   solver.SetMaxIter(params.linear_max_iter);
   solver.SetPrintLevel(params.linear_print_level);

   KSP ksp = (KSP)solver;
   KSPSetType(ksp, KSPGMRES);
   KSPGMRESSetRestart(ksp, std::min(400, std::max(50, params.linear_max_iter)));
   PC pc = nullptr;
   KSPGetPC(ksp, &pc);
   if (Mpi::WorldSize() == 1)
   {
      PCSetType(pc, PCLU);
   }
   else
   {
      PCSetType(pc, PCBJACOBI);
   }

   X = 0.0;
   solver.Mult(B, X);

   const double rhs_norm = B.Norml2();
   const double final_norm = solver.GetFinalNorm();
   const double effective_rel_target =
      params.linear_rel_tol * std::max(1.0, rhs_norm);
   const double effective_tol =
      std::max(params.linear_abs_tol, effective_rel_target);

   if (!solver.GetConverged() && !(final_norm <= effective_tol))
   {
      if (myid == 0)
      {
         std::ostringstream oss;
         oss << "PETSc solve failed in " << tag << " at step " << step
             << ". iterations=" << solver.GetNumIterations()
             << ", residual=" << final_norm
             << ", tol=" << effective_tol;
         throw runtime_error(oss.str());
      }
      throw runtime_error("PETSc solve failed on non-root rank.");
   }
}

SimulationResult RunScenario(const ScenarioSpec &spec,
                             const DriverParams &params,
                             const double dt_nominal,
                             const int myid)
{
   auto pmesh = BuildUnitSquareParMesh(params);
   AleMap ale_map(spec.map_kind);
   PrescribedMeshMotion motion(*pmesh);
   motion.SetTime(ale_map, 0.0);

   H1_FECollection fec(spec.order, 2);
   ParFiniteElementSpace fes(pmesh.get(), &fec);

   if (myid == 0)
   {
      cout << "[" << spec.name << "] map=" << ale_map.Name()
           << " dt=" << dt_nominal
           << " order=" << spec.order
           << " true_dofs=" << fes.GlobalTrueVSize() << endl;
   }

   Array<int> ess_bdr;
   BuildAllBoundaryMarker(*pmesh, ess_bdr);
   Array<int> ess_tdof_list;
   fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ConstantCoefficient zero_coeff(0.0);
   StabilityInitialCoefficient stability_ic;
   ConvergenceExactCoefficient conv_exact;
   ConvergenceForcingCoefficient conv_force(spec.alpha);
   AccuracyExactCoefficient acc_exact;
   AccuracyForcingCoefficient acc_force(spec.alpha);

   Coefficient *init_coeff = SelectCoefficient(spec.initial_model,
                                               zero_coeff,
                                               stability_ic,
                                               conv_exact,
                                               conv_force,
                                               acc_exact,
                                               acc_force);
   Coefficient *forcing_coeff = SelectCoefficient(spec.forcing_model,
                                                  zero_coeff,
                                                  stability_ic,
                                                  conv_exact,
                                                  conv_force,
                                                  acc_exact,
                                                  acc_force);
   Coefficient *exact_coeff = SelectCoefficient(spec.exact_model,
                                                zero_coeff,
                                                stability_ic,
                                                conv_exact,
                                                conv_force,
                                                acc_exact,
                                                acc_force);

   std::unique_ptr<ParGridFunction> u_exact_vis;
   std::unique_ptr<ParGridFunction> u_error_vis;
   std::unique_ptr<ParaViewDataCollection> paraview_dc;

   ParGridFunction u(&fes);
   u = 0.0;
   SetCoefficientTimeIfNeeded(init_coeff, 0.0);
   u.ProjectCoefficient(*init_coeff);
   if (spec.zero_dirichlet)
   {
      u.ProjectBdrCoefficient(zero_coeff, ess_bdr);
   }
   else
   {
      SetCoefficientTimeIfNeeded(exact_coeff, 0.0);
      u.ProjectBdrCoefficient(*exact_coeff, ess_bdr);
   }

   const int quad_order = std::max(4, 2 * spec.order + 6);
   const IntegrationRule *irs[Geometry::NumGeom];
   BuildIntegrationRules(quad_order, irs);

   if (params.save_paraview)
   {
      std::ostringstream cname;
      cname << "ale_" << spec.name
            << "_" << ale_map.Name()
            << "_dt_" << DtTag(dt_nominal)
            << "_p" << spec.order;
      const string collection_name = SanitizeToken(cname.str());
      paraview_dc =
         std::make_unique<ParaViewDataCollection>(collection_name.c_str(), pmesh.get());
      paraview_dc->SetPrefixPath(params.output_path.c_str());
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->RegisterField("u", &u);
      paraview_dc->RegisterField("mesh_velocity", &motion.Velocity());

      if (spec.exact_model != FieldModel::Zero)
      {
         u_exact_vis = std::make_unique<ParGridFunction>(&fes);
         u_error_vis = std::make_unique<ParGridFunction>(&fes);
         *u_exact_vis = 0.0;
         *u_error_vis = 0.0;
         paraview_dc->RegisterField("u_exact", u_exact_vis.get());
         paraview_dc->RegisterField("u_error", u_error_vis.get());
      }
   }

   SimulationResult result;
   auto save_paraview_snapshot = [&](const int cycle,
                                     const double time,
                                     const bool force)
   {
      if (!paraview_dc) { return; }

      if (!force)
      {
         if (params.paraview_every <= 0) { return; }
         if ((cycle % params.paraview_every) != 0) { return; }
      }

      if (u_exact_vis && u_error_vis)
      {
         SetCoefficientTimeIfNeeded(exact_coeff, time);
         u_exact_vis->ProjectCoefficient(*exact_coeff);
         *u_error_vis = u;
         *u_error_vis -= *u_exact_vis;
      }

      paraview_dc->SetCycle(cycle);
      paraview_dc->SetTime(time);
      paraview_dc->Save();
   };

   save_paraview_snapshot(0, 0.0, true);

   if (spec.record_l2_history)
   {
      result.time.push_back(0.0);
      result.l2.push_back(u.ComputeL2Error(zero_coeff, irs));
   }

   double t = 0.0;
   int step = 0;
   while (t < spec.t_final - 1.0e-14)
   {
      ++step;
      const double dt = std::min(dt_nominal, spec.t_final - t);
      const double t_new = t + dt;

      // Conservative ALE BE update uses old mass contribution on the old mesh.
      ParBilinearForm m_old(&fes);
      m_old.AddDomainIntegrator(new MassIntegrator());
      m_old.Assemble();
      m_old.Finalize();

      Vector rhs_local(fes.GetVSize());
      m_old.Mult(u, rhs_local);

      motion.AdvanceTo(ale_map, t_new);

      VectorGridFunctionCoefficient w_coeff(&motion.Velocity());
      DivergenceGridFunctionCoefficient div_w_coeff(motion.Velocity());
      ConstantCoefficient alpha_dt(spec.alpha * dt);
      ScaledCoefficient dt_div_w_coeff(div_w_coeff, dt);

      ParBilinearForm a(&fes);
      a.AddDomainIntegrator(new MassIntegrator());
      a.AddDomainIntegrator(new DiffusionIntegrator(alpha_dt));
      a.AddDomainIntegrator(new ConvectionIntegrator(w_coeff, dt));
      a.AddDomainIntegrator(new MassIntegrator(dt_div_w_coeff));
      a.Assemble();
      a.Finalize();

      ParLinearForm f(&fes);
      SetCoefficientTimeIfNeeded(forcing_coeff, t_new);
      f.AddDomainIntegrator(new DomainLFIntegrator(*forcing_coeff));
      f.Assemble();
      rhs_local.Add(dt, f);

      if (spec.zero_dirichlet)
      {
         u.ProjectBdrCoefficient(zero_coeff, ess_bdr);
      }
      else
      {
         SetCoefficientTimeIfNeeded(exact_coeff, t_new);
         u.ProjectBdrCoefficient(*exact_coeff, ess_bdr);
      }

      OperatorHandle Ah(Operator::Hypre_ParCSR);
      Vector X, B;
      a.FormLinearSystem(ess_tdof_list, u, rhs_local, Ah, X, B);

      const bool all_essential = (ess_tdof_list.Size() == fes.TrueVSize());
      if (!all_essential)
      {
         HypreParMatrix *A_true = Ah.As<HypreParMatrix>();
         MFEM_VERIFY(A_true != nullptr, "Expected HypreParMatrix from FormLinearSystem.");
         SolveTrueSystem(*A_true, X, B, params, myid, spec.name, step);
      }

      a.RecoverFEMSolution(X, rhs_local, u);
      t = t_new;

      const bool is_final = (t >= spec.t_final - 1.0e-14);
      save_paraview_snapshot(step, t, is_final);

      if (spec.record_l2_history)
      {
         result.time.push_back(t);
         result.l2.push_back(u.ComputeL2Error(zero_coeff, irs));
      }
   }

   if (spec.exact_model != FieldModel::Zero)
   {
      SetCoefficientTimeIfNeeded(exact_coeff, t);
      result.final_error_l2 = u.ComputeL2Error(*exact_coeff, irs);
   }

   return result;
}

double ObservedOrder(const double dt_prev,
                     const double err_prev,
                     const double dt_curr,
                     const double err_curr)
{
   if (dt_prev <= 0.0 || dt_curr <= 0.0 || err_prev <= 0.0 || err_curr <= 0.0)
   {
      return numeric_limits<double>::quiet_NaN();
   }
   return std::log(err_prev / err_curr) / std::log(dt_prev / dt_curr);
}

void RunStabilityValidation(const DriverParams &params, const int myid)
{
   const vector<double> dt_list = ParseDoubleList(params.stability_dt_list);

   ScenarioSpec spec;
   spec.name = "stability";
   spec.map_kind = AleMapKind::Scale20Pi;
   spec.alpha = 0.01;
   spec.t_final = 0.4;
   spec.order = params.stability_order;
   spec.initial_model = FieldModel::StabilityInitial;
   spec.forcing_model = FieldModel::Zero;
   spec.exact_model = FieldModel::Zero;
   spec.zero_dirichlet = true;
   spec.record_l2_history = true;

   const string csv_path = params.output_path + "/stability_l2_history.csv";
   std::ofstream csv;
   if (myid == 0)
   {
      csv.open(csv_path);
      csv << "dt,step,time,l2_norm\n";
   }

   for (double dt : dt_list)
   {
      const SimulationResult res = RunScenario(spec, params, dt, myid);
      const bool monotone = IsMonotoneNonIncreasing(res.l2);

      if (myid == 0)
      {
         for (size_t i = 0; i < res.time.size(); ++i)
         {
            csv << std::setprecision(16)
                << dt << "," << i << "," << res.time[i] << "," << res.l2[i] << "\n";
         }
         csv.flush();
         cout << "[stability] dt=" << dt
              << " final_L2=" << (res.l2.empty() ? 0.0 : res.l2.back())
              << " monotone_nonincreasing=" << (monotone ? "true" : "false")
              << endl;
      }
   }
}

void RunConvergenceValidation(const DriverParams &params, const int myid)
{
   const vector<double> dt_list = ParseDoubleList(params.convergence_dt_list);

   ScenarioSpec spec;
   spec.name = "convergence";
   spec.map_kind = AleMapKind::Scale10Pi;
   spec.alpha = 0.1;
   spec.t_final = 0.3;
   spec.order = params.convergence_order;
   spec.initial_model = FieldModel::ConvergenceExact;
   spec.forcing_model = FieldModel::ConvergenceForcing;
   spec.exact_model = FieldModel::ConvergenceExact;
   spec.zero_dirichlet = true;
   spec.record_l2_history = false;

   vector<double> errors;
   errors.reserve(dt_list.size());

   for (double dt : dt_list)
   {
      const SimulationResult res = RunScenario(spec, params, dt, myid);
      errors.push_back(res.final_error_l2);
      if (myid == 0)
      {
         cout << "[convergence] dt=" << dt << " L2_error=" << res.final_error_l2 << endl;
      }
   }

   if (myid == 0)
   {
      const string csv_path = params.output_path + "/convergence_errors.csv";
      std::ofstream csv(csv_path);
      csv << "dt,l2_error,observed_order_vs_prev\n";
      for (size_t i = 0; i < dt_list.size(); ++i)
      {
         double p = numeric_limits<double>::quiet_NaN();
         if (i > 0)
         {
            p = ObservedOrder(dt_list[i - 1], errors[i - 1], dt_list[i], errors[i]);
         }
         csv << std::setprecision(16) << dt_list[i] << "," << errors[i] << ",";
         if (std::isfinite(p)) { csv << p; }
         csv << "\n";
      }
      csv.flush();
   }
}

void RunAccuracyValidation(const DriverParams &params, const int myid)
{
   const vector<double> dt_list = ParseDoubleList(params.accuracy_dt_list);
   const std::array<AleMapKind, 3> maps = {
      AleMapKind::Identity,
      AleMapKind::AccuracyA,
      AleMapKind::AccuracyB
   };

   ScenarioSpec spec;
   spec.name = "accuracy";
   spec.alpha = 0.1;
   spec.t_final = 2.0;
   spec.order = params.accuracy_order;
   spec.initial_model = FieldModel::AccuracyExact;
   spec.forcing_model = FieldModel::AccuracyForcing;
   spec.exact_model = FieldModel::AccuracyExact;
   spec.zero_dirichlet = false;
   spec.record_l2_history = false;

   const string csv_path = params.output_path + "/accuracy_errors.csv";
   std::ofstream csv;
   if (myid == 0)
   {
      csv.open(csv_path);
      csv << "map,dt,l2_error\n";
   }

   for (const AleMapKind map_kind : maps)
   {
      spec.map_kind = map_kind;
      AleMap map(map_kind);
      for (double dt : dt_list)
      {
         const SimulationResult res = RunScenario(spec, params, dt, myid);
         if (myid == 0)
         {
            csv << map.Name() << "," << std::setprecision(16) << dt
                << "," << res.final_error_l2 << "\n";
            csv.flush();
            cout << "[accuracy] map=" << map.Name()
                 << " dt=" << dt
                 << " L2_error=" << res.final_error_l2 << endl;
         }
      }
   }
}

void PrintConfig(const DriverParams &p, const int myid)
{
   if (myid != 0) { return; }
   cout << "ALE validation driver (Backward Euler, MFEM)" << endl;
   if (HasFixedValidationMode())
   {
      cout << "  fixed_validation_mode: " << FixedValidationMode() << endl;
   }
   cout << "  validation: " << p.validation << endl;
   cout << "  mesh_file: " << p.mesh_file << endl;
   cout << "  output_path: " << p.output_path << endl;
   cout << "  serial_ref_levels: " << p.serial_ref_levels << endl;
   cout << "  par_ref_levels: " << p.par_ref_levels << endl;
   cout << "  stability_order: " << p.stability_order << endl;
   cout << "  convergence_order: " << p.convergence_order << endl;
   cout << "  accuracy_order: " << p.accuracy_order << endl;
   cout << "  stability_dt_list: " << p.stability_dt_list << endl;
   cout << "  convergence_dt_list: " << p.convergence_dt_list << endl;
   cout << "  accuracy_dt_list: " << p.accuracy_dt_list << endl;
   cout << "  linear_max_iter: " << p.linear_max_iter << endl;
   cout << "  linear_rel_tol: " << p.linear_rel_tol << endl;
   cout << "  linear_abs_tol: " << p.linear_abs_tol << endl;
   cout << "  linear_print_level: " << p.linear_print_level << endl;
   cout << "  save_paraview: " << (p.save_paraview ? "true" : "false") << endl;
   cout << "  paraview_every: " << p.paraview_every
        << " (0 means initial/final only)" << endl;
   cout << "  paper benchmark source: 1809.06553v1.pdf (Section 7)" << endl;
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
#ifdef MFEM_USE_PETSC
   MFEMInitializePetsc(&argc, &argv, nullptr, nullptr);
#endif
   const int myid = Mpi::WorldRank();

   DriverParams params;
   OptionsParser args(argc, argv);
   args.AddOption(&params.validation, "-v", "--validation",
                  "Validation to run: all | stability | convergence | accuracy.");
   args.AddOption(&params.mesh_file, "-m", "--mesh",
                  "Unit-square mesh file (paper validations use [0,1]^2). ");
   args.AddOption(&params.output_path, "-o", "--output",
                  "Output directory for CSV summaries.");
   args.AddOption(&params.serial_ref_levels, "-rs", "--refine-serial",
                  "Number of serial uniform refinements.");
   args.AddOption(&params.par_ref_levels, "-rp", "--refine-parallel",
                  "Number of parallel uniform refinements.");
   args.AddOption(&params.stability_order, "-so", "--stability-order",
                  "Finite-element order for stability benchmark (paper uses P1). ");
   args.AddOption(&params.convergence_order, "-co", "--convergence-order",
                  "Finite-element order for convergence benchmark (paper uses P2). ");
   args.AddOption(&params.accuracy_order, "-ao", "--accuracy-order",
                  "Finite-element order for accuracy benchmark (paper figures use P1-style grids). ");
   args.AddOption(&params.stability_dt_list, "-sdt", "--stability-dt",
                  "Comma-separated time steps for stability sweep.");
   args.AddOption(&params.convergence_dt_list, "-cdt", "--convergence-dt",
                  "Comma-separated time steps for convergence sweep.");
   args.AddOption(&params.accuracy_dt_list, "-adt", "--accuracy-dt",
                  "Comma-separated time steps for accuracy sweep.");
   args.AddOption(&params.linear_max_iter, "-lmi", "--lin-max-iter",
                  "GMRES maximum iterations per step.");
   args.AddOption(&params.linear_rel_tol, "-lrt", "--lin-rtol",
                  "GMRES relative tolerance.");
   args.AddOption(&params.linear_abs_tol, "-lat", "--lin-atol",
                  "GMRES absolute tolerance.");
   args.AddOption(&params.linear_print_level, "-lpr", "--lin-print",
                  "GMRES print level.");
   args.AddOption(&params.save_paraview,
                  "-pv", "--paraview",
                  "-no-pv", "--no-paraview",
                  "Enable/disable ParaView output (VTU/PVTU snapshots).");
   args.AddOption(&params.paraview_every, "-pve", "--paraview-every",
                  "Save ParaView every N steps per run; 0 means initial/final only.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }

   if (HasFixedValidationMode())
   {
      const string fixed_mode = FixedValidationMode();
      if (params.validation != fixed_mode && myid == 0)
      {
         cout << "Ignoring --validation=" << params.validation
              << " because this executable is fixed to "
              << fixed_mode << "." << endl;
      }
      params.validation = fixed_mode;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   int exit_code = 0;
   try
   {
      if (params.serial_ref_levels < 0 || params.par_ref_levels < 0)
      {
         throw runtime_error("Refinement levels must be >= 0.");
      }
      if (params.stability_order < 1 || params.convergence_order < 1 || params.accuracy_order < 1)
      {
         throw runtime_error("All finite-element orders must be >= 1.");
      }
      if (params.linear_max_iter < 1)
      {
         throw runtime_error("linear_max_iter must be >= 1.");
      }
      if (params.paraview_every < 0)
      {
         throw runtime_error("paraview_every must be >= 0.");
      }

      const ValidationSelection sel = ParseValidationSelection(params.validation);
      EnsureOutputDir(params.output_path, myid);
      PrintConfig(params, myid);

      if (sel == ValidationSelection::All || sel == ValidationSelection::Stability)
      {
         RunStabilityValidation(params, myid);
      }
      if (sel == ValidationSelection::All || sel == ValidationSelection::Convergence)
      {
         RunConvergenceValidation(params, myid);
      }
      if (sel == ValidationSelection::All || sel == ValidationSelection::Accuracy)
      {
         RunAccuracyValidation(params, myid);
      }
   }
   catch (const std::exception &e)
   {
      if (myid == 0)
      {
         cerr << "Error: " << e.what() << endl;
      }
      exit_code = 2;
   }

#ifdef MFEM_USE_PETSC
   MFEMFinalizePetsc();
#endif
   return exit_code;
}

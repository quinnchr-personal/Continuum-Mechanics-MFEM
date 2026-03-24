// Coupled two-domain transient heat-conduction driver.
//
// Two adjacent unit squares are solved with a Dirichlet-Neumann partitioned
// interface iteration at every time step:
//   - Left domain uses Dirichlet temperature on the shared interface.
//   - Left domain exports interface heat flux to the right domain.
//   - Right domain uses that interface flux as Neumann data.
//   - Right domain exports interface temperature back to the left.
//
// Time discretization:
//   (u^{n+1} - u^n)/dt - div(k grad u^{n+1}) = 0
// with backward Euler in time.
//
// ParaView output supports:
//   - converged snapshots per time step
//   - all interface sub-iterations (monotonic global sub-iteration cycle)

#include "mfem.hpp"

#include <yaml-cpp/yaml.h>

#ifndef MFEM_USE_PETSC
#error "This driver requires MFEM built with PETSc."
#endif
#include <petscksp.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace mfem;

namespace
{

struct DriverParams
{
   string mesh_file = "Mesh/unit_square.msh";
   int order = 1;
   int serial_ref_levels = 0;
   int par_ref_levels = 0;

   double k_left = 1.0;
   double k_right = 1.0;
   double left_dirichlet_value = 1.0;
   double right_dirichlet_value = 0.0;
   double dt = 1.0e-2;
   double t_final = 1.0;

   double interface_initial_temperature = 0.5;
   double relaxation = 0.8;
   double coupling_tol = 1.0e-8;
   int coupling_max_iters = 200;
   bool use_flux_convergence = false;
   double flux_tol = 1.0e-8;
   bool use_flux_change_convergence = false;
   double flux_change_tol = 1.0e-8;

   string petsc_options_file = "Input/petsc.opts";
   string output_path = "ParaView/two_domain_poisson_coupled_transient";
   bool save_paraview = true;
   bool save_csv = true;
};

struct Bounds
{
   double xmin = 0.0;
   double xmax = 0.0;
   double ymin = 0.0;
   double ymax = 0.0;
};

void LoadParams(const string &path, DriverParams &p)
{
   if (path.empty())
   {
      throw runtime_error("Input YAML file path is empty.");
   }
   if (!filesystem::exists(path))
   {
      throw runtime_error("YAML input file not found: " + path);
   }

   const YAML::Node n = YAML::LoadFile(path);

   if (n["mesh_file"]) { p.mesh_file = n["mesh_file"].as<string>(); }
   if (n["order"]) { p.order = n["order"].as<int>(); }
   if (n["serial_ref_levels"]) { p.serial_ref_levels = n["serial_ref_levels"].as<int>(); }
   if (n["par_ref_levels"]) { p.par_ref_levels = n["par_ref_levels"].as<int>(); }

   if (n["k_left"]) { p.k_left = n["k_left"].as<double>(); }
   if (n["k_right"]) { p.k_right = n["k_right"].as<double>(); }
   if (n["left_dirichlet_value"])
   {
      p.left_dirichlet_value = n["left_dirichlet_value"].as<double>();
   }
   if (n["right_dirichlet_value"])
   {
      p.right_dirichlet_value = n["right_dirichlet_value"].as<double>();
   }
   if (n["dt"]) { p.dt = n["dt"].as<double>(); }
   if (n["t_final"]) { p.t_final = n["t_final"].as<double>(); }

   if (n["interface_initial_temperature"])
   {
      p.interface_initial_temperature = n["interface_initial_temperature"].as<double>();
   }
   if (n["relaxation"]) { p.relaxation = n["relaxation"].as<double>(); }
   if (n["coupling_tol"]) { p.coupling_tol = n["coupling_tol"].as<double>(); }
   if (n["coupling_max_iters"]) { p.coupling_max_iters = n["coupling_max_iters"].as<int>(); }
   if (n["use_flux_convergence"])
   {
      p.use_flux_convergence = n["use_flux_convergence"].as<bool>();
   }
   if (n["flux_tol"]) { p.flux_tol = n["flux_tol"].as<double>(); }
   if (n["use_flux_change_convergence"])
   {
      p.use_flux_change_convergence = n["use_flux_change_convergence"].as<bool>();
   }
   if (n["flux_change_tol"]) { p.flux_change_tol = n["flux_change_tol"].as<double>(); }

   if (n["petsc_options_file"]) { p.petsc_options_file = n["petsc_options_file"].as<string>(); }
   if (n["output_path"]) { p.output_path = n["output_path"].as<string>(); }
   if (n["save_paraview"]) { p.save_paraview = n["save_paraview"].as<bool>(); }
   if (n["save_csv"]) { p.save_csv = n["save_csv"].as<bool>(); }

   if (p.mesh_file.empty())
   {
      throw runtime_error("mesh_file must not be empty.");
   }
   if (p.order < 1)
   {
      throw runtime_error("order must be >= 1.");
   }
   if (p.serial_ref_levels < 0 || p.par_ref_levels < 0)
   {
      throw runtime_error("serial_ref_levels and par_ref_levels must be >= 0.");
   }
   if (p.k_left <= 0.0 || p.k_right <= 0.0)
   {
      throw runtime_error("k_left and k_right must be > 0.");
   }
   if (p.dt <= 0.0)
   {
      throw runtime_error("dt must be > 0.");
   }
   if (p.t_final < 0.0)
   {
      throw runtime_error("t_final must be >= 0.");
   }
   if (p.relaxation <= 0.0 || p.relaxation > 1.0)
   {
      throw runtime_error("relaxation must satisfy 0 < relaxation <= 1.");
   }
   if (p.coupling_tol <= 0.0)
   {
      throw runtime_error("coupling_tol must be > 0.");
   }
   if (p.coupling_max_iters < 1)
   {
      throw runtime_error("coupling_max_iters must be >= 1.");
   }
   if (p.flux_tol <= 0.0)
   {
      throw runtime_error("flux_tol must be > 0.");
   }
   if (p.flux_change_tol <= 0.0)
   {
      throw runtime_error("flux_change_tol must be > 0.");
   }
}

void PrintConfig(const DriverParams &p)
{
   cout << "Two-domain coupled transient heat driver (BE + D-N iteration)" << endl;
   cout << "  mesh_file:                    " << p.mesh_file << endl;
   cout << "  order:                        " << p.order << endl;
   cout << "  serial_ref_levels:            " << p.serial_ref_levels << endl;
   cout << "  par_ref_levels:               " << p.par_ref_levels << endl;
   cout << "  k_left:                       " << p.k_left << endl;
   cout << "  k_right:                      " << p.k_right << endl;
   cout << "  left_dirichlet_value:         " << p.left_dirichlet_value << endl;
   cout << "  right_dirichlet_value:        " << p.right_dirichlet_value << endl;
   cout << "  dt:                           " << p.dt << endl;
   cout << "  t_final:                      " << p.t_final << endl;
   cout << "  interface_initial_temperature:" << p.interface_initial_temperature << endl;
   cout << "  relaxation:                   " << p.relaxation << endl;
   cout << "  coupling_tol:                 " << p.coupling_tol << endl;
   cout << "  coupling_max_iters:           " << p.coupling_max_iters << endl;
   cout << "  use_flux_convergence:         " << (p.use_flux_convergence ? "true" : "false") << endl;
   cout << "  flux_tol:                     " << p.flux_tol << endl;
   cout << "  use_flux_change_convergence:  "
        << (p.use_flux_change_convergence ? "true" : "false") << endl;
   cout << "  flux_change_tol:              " << p.flux_change_tol << endl;
   cout << "  petsc_options_file:           " << p.petsc_options_file << endl;
   cout << "  output_path:                  " << p.output_path << endl;
   cout << "  save_paraview:                " << (p.save_paraview ? "true" : "false") << endl;
   cout << "  save_csv:                     " << (p.save_csv ? "true" : "false") << endl;
}

Bounds ComputeBounds(const ParMesh &pmesh)
{
   Bounds b;
   b.xmin = numeric_limits<double>::infinity();
   b.ymin = numeric_limits<double>::infinity();
   b.xmax = -numeric_limits<double>::infinity();
   b.ymax = -numeric_limits<double>::infinity();

   for (int i = 0; i < pmesh.GetNV(); i++)
   {
      const double *v = pmesh.GetVertex(i);
      b.xmin = std::min(b.xmin, v[0]);
      b.xmax = std::max(b.xmax, v[0]);
      b.ymin = std::min(b.ymin, v[1]);
      b.ymax = std::max(b.ymax, v[1]);
   }

   Bounds out = b;
   MPI_Allreduce(&b.xmin, &out.xmin, 1, MPI_DOUBLE, MPI_MIN, pmesh.GetComm());
   MPI_Allreduce(&b.xmax, &out.xmax, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
   MPI_Allreduce(&b.ymin, &out.ymin, 1, MPI_DOUBLE, MPI_MIN, pmesh.GetComm());
   MPI_Allreduce(&b.ymax, &out.ymax, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
   return out;
}

void BuildBoundaryMarkerAtX(ParMesh &pmesh, const double x_target,
                            const double tol, Array<int> &marker)
{
   const int nbdr = pmesh.bdr_attributes.Max();
   MFEM_VERIFY(nbdr > 0, "Mesh must define boundary attributes.");
   marker.SetSize(nbdr);
   marker = 0;

   Vector x;
   for (int be = 0; be < pmesh.GetNBE(); be++)
   {
      const int attr = pmesh.GetBdrAttribute(be);
      ElementTransformation *T = pmesh.GetBdrElementTransformation(be);
      const IntegrationPoint &ip = Geometries.GetCenter(T->GetGeometryType());
      T->Transform(ip, x);
      if (std::abs(x[0] - x_target) <= tol)
      {
         marker[attr - 1] = 1;
      }
   }

   Array<int> marker_global(nbdr);
   marker_global = 0;
   MPI_Allreduce(marker.GetData(), marker_global.GetData(), nbdr, MPI_INT,
                 MPI_MAX, pmesh.GetComm());
   marker = marker_global;

   int count = 0;
   for (int i = 0; i < marker.Size(); i++) { count += marker[i]; }
   MFEM_VERIFY(count > 0, "No boundary attributes found near x=" << x_target << ".");
}

vector<double> CollectBoundaryCenterYSamples(ParMesh &pmesh,
                                             const Array<int> &marker,
                                             const double unique_tol)
{
   vector<double> ys;
   Vector x;
   for (int be = 0; be < pmesh.GetNBE(); be++)
   {
      const int attr = pmesh.GetBdrAttribute(be);
      if (attr < 1 || attr > marker.Size() || marker[attr - 1] == 0) { continue; }

      ElementTransformation *T = pmesh.GetBdrElementTransformation(be);
      const IntegrationPoint &ip = Geometries.GetCenter(T->GetGeometryType());
      T->Transform(ip, x);
      ys.push_back(x[1]);
   }

   sort(ys.begin(), ys.end());
   vector<double> unique_ys;
   unique_ys.reserve(ys.size());
   for (const double y : ys)
   {
      if (unique_ys.empty() || std::abs(y - unique_ys.back()) > unique_tol)
      {
         unique_ys.push_back(y);
      }
   }
   return unique_ys;
}

class InterfaceProfileCoefficient : public Coefficient
{
public:
   InterfaceProfileCoefficient(const vector<double> &ys, const vector<double> &vals)
      : ys_(ys), vals_(vals)
   {
      Validate_();
   }

   void SetValues(const vector<double> &vals)
   {
      MFEM_VERIFY(vals.size() == vals_.size(),
                  "InterfaceProfileCoefficient value size mismatch.");
      vals_ = vals;
   }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x;
      T.Transform(ip, x);
      return EvaluateAtY_(x[1]);
   }

private:
   vector<double> ys_;
   vector<double> vals_;

   void Validate_() const
   {
      MFEM_VERIFY(!ys_.empty(), "Interface profile y-samples must not be empty.");
      MFEM_VERIFY(ys_.size() == vals_.size(),
                  "Interface profile y/value vectors must have the same size.");
      for (size_t i = 1; i < ys_.size(); i++)
      {
         MFEM_VERIFY(ys_[i] > ys_[i - 1],
                     "Interface profile y-samples must be strictly increasing.");
      }
   }

   double EvaluateAtY_(const double y) const
   {
      if (ys_.size() == 1) { return vals_[0]; }
      if (y <= ys_.front()) { return vals_.front(); }
      if (y >= ys_.back()) { return vals_.back(); }

      const auto it = std::upper_bound(ys_.begin(), ys_.end(), y);
      const int i1 = static_cast<int>(it - ys_.begin());
      const int i0 = i1 - 1;

      const double y0 = ys_[i0];
      const double y1 = ys_[i1];
      const double t = (y - y0) / (y1 - y0);
      return (1.0 - t) * vals_[i0] + t * vals_[i1];
   }
};

class SteadyTwoDomainExactCoefficient : public Coefficient
{
public:
   SteadyTwoDomainExactCoefficient(const double x_left_min,
                                   const double x_interface,
                                   const double x_right_max,
                                   const double k_left,
                                   const double k_right,
                                   const double left_temp,
                                   const double right_temp)
      : x_left_min_(x_left_min),
        x_interface_(x_interface),
        x_right_max_(x_right_max),
        k_left_(k_left),
        k_right_(k_right),
        left_temp_(left_temp),
        right_temp_(right_temp)
   {
      const double left_width = x_interface_ - x_left_min_;
      const double right_width = x_right_max_ - x_interface_;
      MFEM_VERIFY(left_width > 0.0 && right_width > 0.0,
                  "Invalid domain extents for analytical solution.");

      const double denom = left_width / k_left_ + right_width / k_right_;
      MFEM_VERIFY(denom > 0.0, "Invalid analytical solution denominator.");

      // Constant interface-normal heat flux for the piecewise-linear steady state.
      q_ = (left_temp_ - right_temp_) / denom;
      interface_temp_ = left_temp_ - (q_ / k_left_) * left_width;
   }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x;
      T.Transform(ip, x);
      const double xv = x[0];

      if (xv <= x_interface_)
      {
         return left_temp_ - (q_ / k_left_) * (xv - x_left_min_);
      }

      return interface_temp_ - (q_ / k_right_) * (xv - x_interface_);
   }

private:
   double x_left_min_ = 0.0;
   double x_interface_ = 0.0;
   double x_right_max_ = 0.0;
   double k_left_ = 1.0;
   double k_right_ = 1.0;
   double left_temp_ = 0.0;
   double right_temp_ = 0.0;
   double q_ = 0.0;
   double interface_temp_ = 0.0;
};

void LocatePoints(ParMesh &pmesh, const vector<double> &ys, const double x_query,
                  Array<int> &elem_ids, Array<IntegrationPoint> &ips)
{
   DenseMatrix pts(2, static_cast<int>(ys.size()));
   for (int i = 0; i < static_cast<int>(ys.size()); i++)
   {
      pts(0, i) = x_query;
      pts(1, i) = ys[static_cast<size_t>(i)];
   }

   pmesh.FindPoints(pts, elem_ids, ips, false);
   MFEM_VERIFY(elem_ids.Size() == static_cast<int>(ys.size()),
               "Unexpected FindPoints result size.");
   for (int i = 0; i < elem_ids.Size(); i++)
   {
      MFEM_VERIFY(elem_ids[i] >= 0,
                  "FindPoints failed for sample " << i
                  << " at x=" << x_query << ", y=" << ys[static_cast<size_t>(i)] << ".");
   }
}

vector<double> EvaluateFieldAtLocatedPoints(const ParGridFunction &u,
                                            const Array<int> &elem_ids,
                                            const Array<IntegrationPoint> &ips)
{
   MFEM_VERIFY(elem_ids.Size() == ips.Size(),
               "Element-id and integration-point arrays must have same size.");
   vector<double> values(static_cast<size_t>(elem_ids.Size()), 0.0);
   for (int i = 0; i < elem_ids.Size(); i++)
   {
      values[static_cast<size_t>(i)] = u.GetValue(elem_ids[i], ips[i]);
   }
   return values;
}

vector<double> EvaluateGradXAtLocatedPoints(ParMesh &pmesh,
                                            const ParFiniteElementSpace &fes,
                                            const ParGridFunction &u,
                                            const Array<int> &elem_ids,
                                            const Array<IntegrationPoint> &ips)
{
   MFEM_VERIFY(elem_ids.Size() == ips.Size(),
               "Element-id and integration-point arrays must have same size.");
   const int dim = pmesh.Dimension();
   MFEM_VERIFY(dim == 2, "This driver expects 2D meshes.");

   vector<double> grad_x(static_cast<size_t>(elem_ids.Size()), 0.0);
   Array<int> dofs;
   Vector elvals;
   DenseMatrix dshape;

   for (int i = 0; i < elem_ids.Size(); i++)
   {
      const int elem = elem_ids[i];
      const FiniteElement *fe = fes.GetFE(elem);
      ElementTransformation *T = pmesh.GetElementTransformation(elem);

      T->SetIntPoint(&ips[i]);
      dshape.SetSize(fe->GetDof(), dim);
      fe->CalcPhysDShape(*T, dshape);

      fes.GetElementDofs(elem, dofs);
      MFEM_VERIFY(dofs.Size() == fe->GetDof(),
                  "Unexpected element dof count mismatch.");
      elvals.SetSize(dofs.Size());
      u.GetSubVector(dofs, elvals);

      double gx = 0.0;
      for (int j = 0; j < dofs.Size(); j++)
      {
         gx += elvals[j] * dshape(j, 0);
      }
      grad_x[static_cast<size_t>(i)] = gx;
   }

   return grad_x;
}

double L2Norm(const vector<double> &v)
{
   double sum = 0.0;
   for (const double x : v) { sum += x * x; }
   return std::sqrt(sum);
}

double L2NormDifference(const vector<double> &a, const vector<double> &b)
{
   MFEM_VERIFY(a.size() == b.size(), "Vector size mismatch in L2NormDifference.");
   double sum = 0.0;
   for (size_t i = 0; i < a.size(); i++)
   {
      const double d = a[i] - b[i];
      sum += d * d;
   }
   return std::sqrt(sum);
}

double Average(const vector<double> &v)
{
   MFEM_VERIFY(!v.empty(), "Cannot average an empty vector.");
   const double s = std::accumulate(v.begin(), v.end(), 0.0);
   return s / static_cast<double>(v.size());
}

double AverageAbsDifference(const vector<double> &a, const vector<double> &b)
{
   MFEM_VERIFY(a.size() == b.size(), "Vector size mismatch in AverageAbsDifference.");
   MFEM_VERIFY(!a.empty(), "Cannot average empty vectors.");
   double sum = 0.0;
   for (size_t i = 0; i < a.size(); i++)
   {
      sum += std::abs(a[i] - b[i]);
   }
   return sum / static_cast<double>(a.size());
}

double AverageAbs(const vector<double> &v)
{
   MFEM_VERIFY(!v.empty(), "Cannot average an empty vector.");
   double sum = 0.0;
   for (const double x : v)
   {
      sum += std::abs(x);
   }
   return sum / static_cast<double>(v.size());
}

void SolveLinearSystem(OperatorHandle &A_handle, Vector &X, Vector &B,
                       const string &tag)
{
   HypreParMatrix *A_hypre = A_handle.As<HypreParMatrix>();
   MFEM_VERIFY(A_hypre != nullptr,
               "Expected HypreParMatrix in " << tag << " solve.");

   PetscParMatrix A_petsc(A_hypre, Operator::PETSC_MATAIJ);
   PetscLinearSolver solver(A_petsc);
   solver.SetPrintLevel(0);
   solver.Mult(B, X);

   MFEM_VERIFY(solver.GetConverged(),
               "PETSc solve failed in " << tag
               << ". Iterations=" << solver.GetNumIterations()
               << ", final residual=" << solver.GetFinalNorm());
}

void UpdateErrorField(const ParGridFunction &u,
                      const ParGridFunction &u_exact,
                      ParGridFunction &u_error)
{
   u_error = u;
   u_error -= u_exact;
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const int myid = Mpi::WorldRank();

   string input_file = "Input/input_two_domain_poisson_coupled_transient.yaml";
   OptionsParser args(argc, argv);
   args.AddOption(&input_file, "-i", "--input", "YAML input file.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   DriverParams params;
   try
   {
      LoadParams(input_file, params);
   }
   catch (const exception &e)
   {
      if (myid == 0) { cerr << e.what() << endl; }
      return 2;
   }
   if (myid == 0) { PrintConfig(params); }

   const char *petsc_file_to_use = nullptr;
   if (!params.petsc_options_file.empty())
   {
      ifstream petsc_in(params.petsc_options_file);
      if (petsc_in.good())
      {
         petsc_file_to_use = params.petsc_options_file.c_str();
      }
      else if (myid == 0)
      {
         cerr << "PETSc options file not found: " << params.petsc_options_file
              << ". Proceeding without options file." << endl;
      }
   }
   MFEMInitializePetsc(&argc, &argv, petsc_file_to_use, NULL);

   int exit_code = 0;
   try
   {
      if (Mpi::WorldSize() != 1)
      {
         throw runtime_error(
            "two_domain_poisson_coupled_transient currently supports only -np 1.");
      }

      Device device("cpu");
      if (myid == 0) { device.Print(); }

      unique_ptr<Mesh> left_mesh = make_unique<Mesh>(params.mesh_file.c_str(), 1, 1);
      unique_ptr<Mesh> right_mesh = make_unique<Mesh>(params.mesh_file.c_str(), 1, 1);
      MFEM_VERIFY(left_mesh->Dimension() == 2 && right_mesh->Dimension() == 2,
                  "The input mesh must be 2D.");

      for (int l = 0; l < params.serial_ref_levels; l++)
      {
         left_mesh->UniformRefinement();
         right_mesh->UniformRefinement();
      }

      MFEM_VERIFY(right_mesh->GetNodes() == nullptr,
                  "Nodal meshes are not supported in this driver.");
      for (int i = 0; i < right_mesh->GetNV(); i++)
      {
         double *v = right_mesh->GetVertex(i);
         v[0] += 1.0;
      }

      unique_ptr<ParMesh> pmesh_left = make_unique<ParMesh>(MPI_COMM_WORLD, *left_mesh);
      unique_ptr<ParMesh> pmesh_right = make_unique<ParMesh>(MPI_COMM_WORLD, *right_mesh);
      left_mesh.reset();
      right_mesh.reset();

      for (int l = 0; l < params.par_ref_levels; l++)
      {
         pmesh_left->UniformRefinement();
         pmesh_right->UniformRefinement();
      }

      MFEM_VERIFY(pmesh_left->bdr_attributes.Size() > 0 &&
                  pmesh_right->bdr_attributes.Size() > 0,
                  "Both meshes must define boundary attributes.");

      const Bounds bounds_left = ComputeBounds(*pmesh_left);
      const Bounds bounds_right = ComputeBounds(*pmesh_right);
      const double tol = 1.0e-8;
      MFEM_VERIFY(std::abs(bounds_left.xmax - bounds_right.xmin) <= tol,
                  "Shifted meshes do not share the expected interface at x=1.");

      Array<int> left_outer_bdr, left_iface_bdr, right_outer_bdr, right_iface_bdr;
      BuildBoundaryMarkerAtX(*pmesh_left, bounds_left.xmin, tol, left_outer_bdr);
      BuildBoundaryMarkerAtX(*pmesh_left, bounds_left.xmax, tol, left_iface_bdr);
      BuildBoundaryMarkerAtX(*pmesh_right, bounds_right.xmin, tol, right_iface_bdr);
      BuildBoundaryMarkerAtX(*pmesh_right, bounds_right.xmax, tol, right_outer_bdr);

      Array<int> left_ess_bdr(left_outer_bdr.Size());
      left_ess_bdr = 0;
      for (int i = 0; i < left_ess_bdr.Size(); i++)
      {
         left_ess_bdr[i] = (left_outer_bdr[i] || left_iface_bdr[i]) ? 1 : 0;
      }
      Array<int> right_ess_bdr = right_outer_bdr;

      H1_FECollection fec(params.order, 2);
      ParFiniteElementSpace fes_left(pmesh_left.get(), &fec);
      ParFiniteElementSpace fes_right(pmesh_right.get(), &fec);
      if (myid == 0)
      {
         cout << "Left global true dofs:  " << fes_left.GlobalTrueVSize() << endl;
         cout << "Right global true dofs: " << fes_right.GlobalTrueVSize() << endl;
      }

      Array<int> ess_tdof_left, ess_tdof_right;
      fes_left.GetEssentialTrueDofs(left_ess_bdr, ess_tdof_left);
      fes_right.GetEssentialTrueDofs(right_ess_bdr, ess_tdof_right);
      const bool left_all_essential = (ess_tdof_left.Size() == fes_left.TrueVSize());
      const bool right_all_essential = (ess_tdof_right.Size() == fes_right.TrueVSize());

      const vector<double> y_samples =
         CollectBoundaryCenterYSamples(*pmesh_left, left_iface_bdr, 1.0e-12);
      MFEM_VERIFY(!y_samples.empty(), "No interface sample points were found.");
      if (myid == 0)
      {
         cout << "Interface sample count: " << y_samples.size() << endl;
      }

      vector<double> interface_guess(y_samples.size(),
                                     params.interface_initial_temperature);
      vector<double> g_old(y_samples.size(), params.interface_initial_temperature);
      vector<double> g_new(y_samples.size(), params.interface_initial_temperature);
      vector<double> q_right_profile(y_samples.size(), 0.0);

      InterfaceProfileCoefficient interface_temp_coeff(y_samples, g_old);
      InterfaceProfileCoefficient interface_flux_coeff(y_samples, q_right_profile);

      ConstantCoefficient left_outer_temp(params.left_dirichlet_value);
      ConstantCoefficient right_outer_temp(params.right_dirichlet_value);
      ConstantCoefficient dt_k_left_coeff(params.dt * params.k_left);
      ConstantCoefficient dt_k_right_coeff(params.dt * params.k_right);

      ParBilinearForm mass_left(&fes_left);
      mass_left.AddDomainIntegrator(new MassIntegrator());
      mass_left.Assemble();
      mass_left.Finalize();

      ParBilinearForm mass_right(&fes_right);
      mass_right.AddDomainIntegrator(new MassIntegrator());
      mass_right.Assemble();
      mass_right.Finalize();

      ParBilinearForm lhs_left(&fes_left);
      lhs_left.AddDomainIntegrator(new MassIntegrator());
      lhs_left.AddDomainIntegrator(new DiffusionIntegrator(dt_k_left_coeff));
      lhs_left.Assemble();
      lhs_left.Finalize();

      ParBilinearForm lhs_right(&fes_right);
      lhs_right.AddDomainIntegrator(new MassIntegrator());
      lhs_right.AddDomainIntegrator(new DiffusionIntegrator(dt_k_right_coeff));
      lhs_right.Assemble();
      lhs_right.Finalize();

      ParLinearForm neumann_form_right(&fes_right);
      neumann_form_right.AddBoundaryIntegrator(
         new BoundaryLFIntegrator(interface_flux_coeff), right_iface_bdr);

      ParGridFunction T_left(&fes_left);
      ParGridFunction T_right(&fes_right);
      ParGridFunction T_left_old(&fes_left);
      ParGridFunction T_right_old(&fes_right);
      ParGridFunction T_left_exact(&fes_left);
      ParGridFunction T_right_exact(&fes_right);
      ParGridFunction T_left_error(&fes_left);
      ParGridFunction T_right_error(&fes_right);

      // Initial condition at t=0.
      T_left = params.left_dirichlet_value;
      T_right = params.right_dirichlet_value;
      T_left_old = T_left;
      T_right_old = T_right;

      SteadyTwoDomainExactCoefficient exact_coeff(bounds_left.xmin,
                                                  bounds_left.xmax,
                                                  bounds_right.xmax,
                                                  params.k_left,
                                                  params.k_right,
                                                  params.left_dirichlet_value,
                                                  params.right_dirichlet_value);
      T_left_exact.ProjectCoefficient(exact_coeff);
      T_right_exact.ProjectCoefficient(exact_coeff);
      UpdateErrorField(T_left, T_left_exact, T_left_error);
      UpdateErrorField(T_right, T_right_exact, T_right_error);

      const double x_iface = bounds_left.xmax;
      const double eps = 1.0e-10;
      Array<int> left_elem_ids, right_elem_ids;
      Array<IntegrationPoint> left_ips, right_ips;
      LocatePoints(*pmesh_left, y_samples, x_iface - eps, left_elem_ids, left_ips);
      LocatePoints(*pmesh_right, y_samples, x_iface + eps, right_elem_ids, right_ips);

      if ((params.save_csv || params.save_paraview) && myid == 0)
      {
         std::error_code ec;
         filesystem::create_directories(params.output_path, ec);
         if (ec)
         {
            throw runtime_error("Failed to create output directory: " + params.output_path +
                                " (" + ec.message() + ")");
         }
      }

      ofstream step_csv;
      ofstream iter_csv;
      if (params.save_csv && myid == 0)
      {
         const filesystem::path step_csv_path =
            filesystem::path(params.output_path) / "time_step_history.csv";
         step_csv.open(step_csv_path);
         if (!step_csv)
         {
            throw runtime_error("Failed to open CSV: " + step_csv_path.string());
         }
         step_csv << "step,time,subiters_used,step_converged,rel_change_final,"
                  << "flux_jump_l2_rel_final,rel_flux_change_final\n";
         step_csv << std::setprecision(16);

         const filesystem::path iter_csv_path =
            filesystem::path(params.output_path) / "interface_iteration_history.csv";
         iter_csv.open(iter_csv_path);
         if (!iter_csv)
         {
            throw runtime_error("Failed to open CSV: " + iter_csv_path.string());
         }
         iter_csv << "step,time,iter,rel_change,interface_l2,interface_avg,temp_jump_avg,"
                  << "flux_left_avg,flux_right_avg,flux_jump_avg_abs,flux_jump_l2_rel,"
                  << "rel_flux_change\n";
         iter_csv << std::setprecision(16);
      }

      ParaViewDataCollection pv_left_steps("two_domain_left_timesteps", pmesh_left.get());
      ParaViewDataCollection pv_right_steps("two_domain_right_timesteps", pmesh_right.get());
      ParaViewDataCollection pv_left_subiters("two_domain_left_subiterations",
                                              pmesh_left.get());
      ParaViewDataCollection pv_right_subiters("two_domain_right_subiterations",
                                               pmesh_right.get());

      if (params.save_paraview)
      {
         pv_left_steps.SetPrefixPath(params.output_path.c_str());
         pv_left_steps.SetLevelsOfDetail(params.order);
         pv_left_steps.SetDataFormat(VTKFormat::BINARY);
         pv_left_steps.SetHighOrderOutput(true);
         pv_left_steps.RegisterField("T_left", &T_left);
         pv_left_steps.RegisterField("T_left_exact", &T_left_exact);
         pv_left_steps.RegisterField("T_left_error", &T_left_error);

         pv_right_steps.SetPrefixPath(params.output_path.c_str());
         pv_right_steps.SetLevelsOfDetail(params.order);
         pv_right_steps.SetDataFormat(VTKFormat::BINARY);
         pv_right_steps.SetHighOrderOutput(true);
         pv_right_steps.RegisterField("T_right", &T_right);
         pv_right_steps.RegisterField("T_right_exact", &T_right_exact);
         pv_right_steps.RegisterField("T_right_error", &T_right_error);

         pv_left_subiters.SetPrefixPath(params.output_path.c_str());
         pv_left_subiters.SetLevelsOfDetail(params.order);
         pv_left_subiters.SetDataFormat(VTKFormat::BINARY);
         pv_left_subiters.SetHighOrderOutput(true);
         pv_left_subiters.RegisterField("T_left", &T_left);
         pv_left_subiters.RegisterField("T_left_exact", &T_left_exact);
         pv_left_subiters.RegisterField("T_left_error", &T_left_error);

         pv_right_subiters.SetPrefixPath(params.output_path.c_str());
         pv_right_subiters.SetLevelsOfDetail(params.order);
         pv_right_subiters.SetDataFormat(VTKFormat::BINARY);
         pv_right_subiters.SetHighOrderOutput(true);
         pv_right_subiters.RegisterField("T_right", &T_right);
         pv_right_subiters.RegisterField("T_right_exact", &T_right_exact);
         pv_right_subiters.RegisterField("T_right_error", &T_right_error);
      }

      int global_subiter_cycle = 0;
      if (params.save_paraview)
      {
         pv_left_steps.SetCycle(0);
         pv_left_steps.SetTime(0.0);
         pv_left_steps.Save();
         pv_right_steps.SetCycle(0);
         pv_right_steps.SetTime(0.0);
         pv_right_steps.Save();

         pv_left_subiters.SetCycle(0);
         pv_left_subiters.SetTime(0.0);
         pv_left_subiters.Save();
         pv_right_subiters.SetCycle(0);
         pv_right_subiters.SetTime(0.0);
         pv_right_subiters.Save();
      }

      const int nsteps = static_cast<int>(ceil(params.t_final / params.dt - 1.0e-12));
      if (myid == 0)
      {
         cout << "Time steps: " << nsteps
              << ", dt=" << params.dt
              << ", nominal final time=" << (nsteps * params.dt) << endl;
      }

      bool any_unconverged_step = false;
      vector<int> unconverged_steps;

      for (int step = 1; step <= nsteps; step++)
      {
         const double t = step * params.dt;

         Vector rhs_left_base(fes_left.GetVSize());
         Vector rhs_right_base(fes_right.GetVSize());
         mass_left.Mult(T_left_old, rhs_left_base);
         mass_right.Mult(T_right_old, rhs_right_base);

         g_old = interface_guess;
         g_new = g_old;

         bool step_converged = false;
         int subiters_used = 0;
         double final_rel_change = numeric_limits<double>::infinity();
         double final_flux_jump_l2_rel = numeric_limits<double>::infinity();
         double final_rel_flux_change = numeric_limits<double>::infinity();
         vector<double> q_left_prev(y_samples.size(), 0.0);
         bool have_prev_flux = false;

         for (int iter = 0; iter < params.coupling_max_iters; iter++)
         {
            subiters_used = iter + 1;
            interface_temp_coeff.SetValues(g_old);

            // Left solve.
            Vector rhs_left(rhs_left_base);
            T_left = T_left_old;
            T_left.ProjectBdrCoefficient(left_outer_temp, left_outer_bdr);
            T_left.ProjectBdrCoefficient(interface_temp_coeff, left_iface_bdr);

            OperatorHandle A_left(Operator::Hypre_ParCSR);
            Vector X_left, B_left;
            lhs_left.FormLinearSystem(ess_tdof_left, T_left, rhs_left, A_left, X_left, B_left);
            if (!left_all_essential)
            {
               SolveLinearSystem(A_left, X_left, B_left, "left");
            }
            lhs_left.RecoverFEMSolution(X_left, rhs_left, T_left);

            const vector<double> T_left_iface =
               EvaluateFieldAtLocatedPoints(T_left, left_elem_ids, left_ips);
            const vector<double> dTdx_left =
               EvaluateGradXAtLocatedPoints(*pmesh_left, fes_left, T_left,
                                            left_elem_ids, left_ips);

            vector<double> q_left(y_samples.size(), 0.0);
            for (size_t i = 0; i < y_samples.size(); i++)
            {
               q_left[i] = params.k_left * dTdx_left[i];
               q_right_profile[i] = -q_left[i];
            }
            interface_flux_coeff.SetValues(q_right_profile);

            // Right solve.
            Vector rhs_right(rhs_right_base);
            neumann_form_right = 0.0;
            neumann_form_right.Assemble();
            rhs_right.Add(params.dt, neumann_form_right);

            T_right = T_right_old;
            T_right.ProjectBdrCoefficient(right_outer_temp, right_outer_bdr);

            OperatorHandle A_right(Operator::Hypre_ParCSR);
            Vector X_right, B_right;
            lhs_right.FormLinearSystem(ess_tdof_right, T_right, rhs_right,
                                       A_right, X_right, B_right);
            if (!right_all_essential)
            {
               SolveLinearSystem(A_right, X_right, B_right, "right");
            }
            lhs_right.RecoverFEMSolution(X_right, rhs_right, T_right);

            const vector<double> T_right_iface =
               EvaluateFieldAtLocatedPoints(T_right, right_elem_ids, right_ips);
            const vector<double> dTdx_right =
               EvaluateGradXAtLocatedPoints(*pmesh_right, fes_right, T_right,
                                            right_elem_ids, right_ips);

            for (size_t i = 0; i < y_samples.size(); i++)
            {
               g_new[i] = (1.0 - params.relaxation) * g_old[i]
                        + params.relaxation * T_right_iface[i];
            }

            vector<double> q_right_solved(y_samples.size(), 0.0);
            vector<double> q_jump(y_samples.size(), 0.0);
            for (size_t i = 0; i < y_samples.size(); i++)
            {
               q_right_solved[i] = -params.k_right * dTdx_right[i];
               q_jump[i] = q_left[i] + q_right_solved[i];
            }

            const double diff_norm = L2NormDifference(g_new, g_old);
            const double new_norm = std::max(L2Norm(g_new), 1.0e-14);
            const double rel_change = diff_norm / new_norm;
            const double interface_l2 = L2Norm(g_new);
            const double interface_avg = Average(g_new);
            const double temp_jump_avg = AverageAbsDifference(T_left_iface, T_right_iface);
            const double flux_left_avg = Average(q_left);
            const double flux_right_avg = Average(q_right_solved);
            const double flux_jump_avg_abs = AverageAbs(q_jump);
            const double flux_jump_l2_rel =
               L2Norm(q_jump) / std::max(L2Norm(q_left), 1.0e-14);
            const double rel_flux_change =
               have_prev_flux
                  ? L2NormDifference(q_left, q_left_prev) / std::max(L2Norm(q_left), 1.0e-14)
                  : numeric_limits<double>::infinity();

            final_rel_change = rel_change;
            final_flux_jump_l2_rel = flux_jump_l2_rel;
            final_rel_flux_change = rel_flux_change;

            const bool temp_ok = (rel_change < params.coupling_tol);
            const bool flux_ok = (!params.use_flux_convergence)
                                 || (flux_jump_l2_rel < params.flux_tol);
            const bool flux_change_ok = (!params.use_flux_change_convergence)
                                        || (rel_flux_change < params.flux_change_tol);

            if (myid == 0)
            {
               cout << "step=" << step
                    << " iter=" << iter
                    << " rel_change=" << rel_change
                    << " flux_jump_l2_rel=" << flux_jump_l2_rel
                    << " rel_flux_change=" << rel_flux_change << endl;
               if (params.save_csv)
               {
                  iter_csv << step << ","
                           << t << ","
                           << iter << ","
                           << rel_change << ","
                           << interface_l2 << ","
                           << interface_avg << ","
                           << temp_jump_avg << ","
                           << flux_left_avg << ","
                           << flux_right_avg << ","
                           << flux_jump_avg_abs << ","
                           << flux_jump_l2_rel << ","
                           << rel_flux_change << "\n";
                  iter_csv.flush();
               }
            }

            if (params.save_paraview)
            {
               UpdateErrorField(T_left, T_left_exact, T_left_error);
               UpdateErrorField(T_right, T_right_exact, T_right_error);

               global_subiter_cycle++;
               pv_left_subiters.SetCycle(global_subiter_cycle);
               pv_left_subiters.SetTime(static_cast<double>(global_subiter_cycle));
               pv_left_subiters.Save();

               pv_right_subiters.SetCycle(global_subiter_cycle);
               pv_right_subiters.SetTime(static_cast<double>(global_subiter_cycle));
               pv_right_subiters.Save();
            }

            q_left_prev = q_left;
            have_prev_flux = true;

            if (temp_ok && flux_ok && flux_change_ok)
            {
               step_converged = true;
               g_old = g_new;
               break;
            }

            g_old = g_new;
         }

         // Carry accepted interface state and accepted fields to next time step.
         interface_guess = g_old;
         T_left_old = T_left;
         T_right_old = T_right;

         if (!step_converged)
         {
            any_unconverged_step = true;
            unconverged_steps.push_back(step);
         }

         if (myid == 0)
         {
            cout << "Completed time step " << step
                 << " at t=" << t
                 << " with subiters=" << subiters_used
                 << " converged=" << (step_converged ? 1 : 0)
                 << " rel_change_final=" << final_rel_change
                 << " flux_jump_l2_rel_final=" << final_flux_jump_l2_rel
                 << " rel_flux_change_final=" << final_rel_flux_change << endl;
            if (params.save_csv)
            {
               step_csv << step << ","
                        << t << ","
                        << subiters_used << ","
                        << (step_converged ? 1 : 0) << ","
                        << final_rel_change << ","
                        << final_flux_jump_l2_rel << ","
                        << final_rel_flux_change << "\n";
               step_csv.flush();
            }
         }

         if (params.save_paraview)
         {
            UpdateErrorField(T_left, T_left_exact, T_left_error);
            UpdateErrorField(T_right, T_right_exact, T_right_error);

            pv_left_steps.SetCycle(step);
            pv_left_steps.SetTime(t);
            pv_left_steps.Save();

            pv_right_steps.SetCycle(step);
            pv_right_steps.SetTime(t);
            pv_right_steps.Save();
         }
      }

      if (myid == 0)
      {
         if (any_unconverged_step)
         {
            cerr << "Warning: " << unconverged_steps.size()
                 << " time step(s) were unconverged:";
            for (size_t i = 0; i < unconverged_steps.size(); i++)
            {
               cerr << " " << unconverged_steps[i];
            }
            cerr << endl;
         }
         else
         {
            cout << "All time steps converged by the requested coupling criteria." << endl;
         }
      }

      if (any_unconverged_step)
      {
         exit_code = 4;
      }
   }
   catch (const exception &e)
   {
      if (myid == 0)
      {
         cerr << "Error: " << e.what() << endl;
      }
      exit_code = 3;
   }

   MFEMFinalizePetsc();
   return exit_code;
}

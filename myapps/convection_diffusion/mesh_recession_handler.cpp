#include "mesh_recession_handler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace mfem;

MeshRecessionHandler::MeshRecessionHandler(ParMesh &pmesh,
                                           const RecessionConfig &config)
   : pmesh_(pmesh),
     config_(config)
{
   EnsureNodalMesh_();

   GridFunction *nodes = pmesh_.GetNodes();
   MFEM_VERIFY(nodes != nullptr,
               "MeshRecessionHandler requires a nodal mesh representation.");

   FiniteElementSpace *nodes_fes = nodes->FESpace();
   MFEM_VERIFY(nodes_fes != nullptr,
               "MeshRecessionHandler requires nodal finite-element space.");

   auto *nodes_pf = dynamic_cast<ParFiniteElementSpace *>(nodes_fes);
   MFEM_VERIFY(nodes_pf != nullptr,
               "MeshRecessionHandler expected ParFiniteElementSpace for nodes.");
   MFEM_VERIFY(nodes_pf->GetVDim() >= 2,
               "MeshRecessionHandler expects at least a 2D nodal coordinate space.");

   scalar_fes_ = std::make_unique<ParFiniteElementSpace>(&pmesh_,
                                                         nodes_pf->FEColl(),
                                                         1,
                                                         nodes_pf->GetOrdering());
   vector_fes_ = nodes_pf;

   mesh_velocity_ = std::make_unique<ParGridFunction>(vector_fes_);
   recession_field_ = std::make_unique<ParGridFunction>(scalar_fes_.get());

   *mesh_velocity_ = 0.0;
   *recession_field_ = 0.0;

   AssembleBoundaryMaps_();

   initial_min_quality_ = ComputeMinElementQuality_();
   if (!std::isfinite(initial_min_quality_) || initial_min_quality_ <= 0.0)
   {
      throw std::runtime_error(
         "Invalid initial mesh quality for moving-mesh recession handling.");
   }
}

void MeshRecessionHandler::EnsureNodalMesh_()
{
   if (pmesh_.GetNodes())
   {
      return;
   }
   pmesh_.SetCurvature(1, false, pmesh_.SpaceDimension(), Ordering::byVDIM);
}

void MeshRecessionHandler::AssembleBoundaryMaps_()
{
   const int nbdr_attr = pmesh_.bdr_attributes.Max();
   Array<int> top_bdr(nbdr_attr);
   Array<int> bottom_bdr(nbdr_attr);
   top_bdr = 0;
   bottom_bdr = 0;

   if (config_.bdr_attr_top >= 1 && config_.bdr_attr_top <= nbdr_attr)
   {
      top_bdr[config_.bdr_attr_top - 1] = 1;
   }
   if (config_.bdr_attr_bottom >= 1 && config_.bdr_attr_bottom <= nbdr_attr)
   {
      bottom_bdr[config_.bdr_attr_bottom - 1] = 1;
   }

   scalar_fes_->GetEssentialTrueDofs(top_bdr, top_scalar_tdofs_);
   scalar_fes_->GetEssentialTrueDofs(bottom_bdr, bottom_scalar_tdofs_);

   Array<int> top_x_true;
   Array<int> top_y_true;
   Array<int> bottom_x_true;
   Array<int> bottom_y_true;
   vector_fes_->GetEssentialTrueDofs(top_bdr, top_x_true, 0);
   vector_fes_->GetEssentialTrueDofs(top_bdr, top_y_true, 1);
   vector_fes_->GetEssentialTrueDofs(bottom_bdr, bottom_x_true, 0);
   vector_fes_->GetEssentialTrueDofs(bottom_bdr, bottom_y_true, 1);

   std::vector<int> ess_tdofs;
   ess_tdofs.reserve(top_x_true.Size() + top_y_true.Size() +
                     bottom_x_true.Size() + bottom_y_true.Size());
   for (int i = 0; i < top_x_true.Size(); ++i)
   {
      ess_tdofs.push_back(top_x_true[i]);
   }
   for (int i = 0; i < top_y_true.Size(); ++i)
   {
      ess_tdofs.push_back(top_y_true[i]);
   }
   for (int i = 0; i < bottom_x_true.Size(); ++i)
   {
      ess_tdofs.push_back(bottom_x_true[i]);
   }
   for (int i = 0; i < bottom_y_true.Size(); ++i)
   {
      ess_tdofs.push_back(bottom_y_true[i]);
   }
   std::sort(ess_tdofs.begin(), ess_tdofs.end());
   ess_tdofs.erase(std::unique(ess_tdofs.begin(), ess_tdofs.end()),
                   ess_tdofs.end());

   ess_vector_tdofs_.SetSize(static_cast<int>(ess_tdofs.size()));
   for (int i = 0; i < ess_vector_tdofs_.Size(); ++i)
   {
      ess_vector_tdofs_[i] = ess_tdofs[static_cast<std::size_t>(i)];
   }

   top_bc_map_.clear();
   top_bc_map_.reserve(top_scalar_tdofs_.Size());
   for (int i = 0; i < top_scalar_tdofs_.Size(); ++i)
   {
      const int scalar_tdof = top_scalar_tdofs_[i];
      const int scalar_ldof = scalar_fes_->GetLocalTDofNumber(scalar_tdof);
      if (scalar_ldof < 0)
      {
         continue;
      }

      TopBCMap entry;
      entry.scalar_tdof = scalar_tdof;
      entry.x_tdof = vector_fes_->DofToVDof(scalar_ldof, 0);
      entry.y_tdof = vector_fes_->DofToVDof(scalar_ldof, 1);
      top_bc_map_.push_back(entry);
   }

   bottom_x_tdofs_.clear();
   bottom_y_tdofs_.clear();
   bottom_x_tdofs_.reserve(bottom_scalar_tdofs_.Size());
   bottom_y_tdofs_.reserve(bottom_scalar_tdofs_.Size());
   for (int i = 0; i < bottom_scalar_tdofs_.Size(); ++i)
   {
      const int scalar_tdof = bottom_scalar_tdofs_[i];
      const int scalar_ldof = scalar_fes_->GetLocalTDofNumber(scalar_tdof);
      if (scalar_ldof < 0)
      {
         continue;
      }
      bottom_x_tdofs_.push_back(vector_fes_->DofToVDof(scalar_ldof, 0));
      bottom_y_tdofs_.push_back(vector_fes_->DofToVDof(scalar_ldof, 1));
   }
}

void MeshRecessionHandler::ClampTopVelocity_(const Vector &top_velocity_true,
                                             const double dt,
                                             Vector &clamped) const
{
   MFEM_VERIFY(top_velocity_true.Size() == scalar_fes_->TrueVSize(),
               "Top recession velocity vector has incompatible true-dof size.");

   clamped.SetSize(top_velocity_true.Size());
   clamped = 0.0;

   double max_velocity = std::numeric_limits<double>::infinity();
   if (std::isfinite(config_.max_step_recession) &&
       config_.max_step_recession > 0.0 &&
       dt > 0.0)
   {
      max_velocity = config_.max_step_recession / dt;
   }

   for (int i = 0; i < top_scalar_tdofs_.Size(); ++i)
   {
      const int tdof = top_scalar_tdofs_[i];
      double v = top_velocity_true(tdof);
      if (!std::isfinite(v) || v <= 0.0)
      {
         v = 0.0;
      }
      if (std::isfinite(max_velocity))
      {
         v = std::min(v, max_velocity);
      }
      clamped(tdof) = v;
   }
}

double MeshRecessionHandler::ComputeTopMeanVelocity_(
   const Vector &top_velocity_true) const
{
   double local_sum = 0.0;
   double local_count = 0.0;
   for (int i = 0; i < top_scalar_tdofs_.Size(); ++i)
   {
      const int tdof = top_scalar_tdofs_[i];
      const double v = std::max(0.0, top_velocity_true(tdof));
      local_sum += v;
      local_count += 1.0;
   }

   double local_data[2] = {local_sum, local_count};
   double global_data[2] = {0.0, 0.0};
   MPI_Allreduce(local_data,
                 global_data,
                 2,
                 MPI_DOUBLE,
                 MPI_SUM,
                 pmesh_.GetComm());

   if (global_data[1] <= 0.0)
   {
      return 0.0;
   }
   return global_data[0] / global_data[1];
}

void MeshRecessionHandler::SolveVelocityLaplacian_(const Vector &top_velocity_true)
{
   if (config_.mesh_smoothing_model != "laplacian")
   {
      throw std::runtime_error(
         "Only mesh_smoothing_model=laplacian is currently supported.");
   }

   ConstantCoefficient one(1.0);
   ParBilinearForm a(vector_fes_);
   a.AddDomainIntegrator(new VectorDiffusionIntegrator(one));
   a.Assemble();
   a.Finalize();

   ParLinearForm b(vector_fes_);
   b = 0.0;
   b.Assemble();

   *mesh_velocity_ = 0.0;

   for (const TopBCMap &bc : top_bc_map_)
   {
      double top_v = 0.0;
      if (bc.scalar_tdof >= 0 && bc.scalar_tdof < top_velocity_true.Size())
      {
         top_v = top_velocity_true(bc.scalar_tdof);
      }
      if (!std::isfinite(top_v) || top_v <= 0.0)
      {
         top_v = 0.0;
      }

      // Top normal is +y for this strip. Recession moves inward.
      (*mesh_velocity_)(bc.x_tdof) = 0.0;
      (*mesh_velocity_)(bc.y_tdof) = -top_v;
   }

   for (const int vdof : bottom_x_tdofs_)
   {
      (*mesh_velocity_)(vdof) = 0.0;
   }
   for (const int vdof : bottom_y_tdofs_)
   {
      (*mesh_velocity_)(vdof) = 0.0;
   }

   OperatorPtr A;
   Vector X;
   Vector B;
   a.FormLinearSystem(ess_vector_tdofs_, *mesh_velocity_, b, A, X, B);

   CGSolver cg(pmesh_.GetComm());
   cg.SetRelTol(1.0e-12);
   cg.SetAbsTol(0.0);
   cg.SetMaxIter(500);
   cg.SetPrintLevel(0);
   cg.SetOperator(*A);
   cg.Mult(B, X);

   a.RecoverFEMSolution(X, b, *mesh_velocity_);
}

void MeshRecessionHandler::MoveMesh_(const double dt,
                                     double &max_node_disp)
{
   GridFunction *nodes = pmesh_.GetNodes();
   MFEM_VERIFY(nodes != nullptr,
               "MeshRecessionHandler update requires nodal coordinates.");

   Vector mesh_disp(nodes->Size());
   mesh_disp = 0.0;

   max_node_disp = 0.0;
   for (int i = 0; i < mesh_disp.Size(); ++i)
   {
      const double dx = dt * (*mesh_velocity_)(i);
      mesh_disp(i) = dx;
      max_node_disp = std::max(max_node_disp, std::abs(dx));
   }

   pmesh_.MoveNodes(mesh_disp);
}

double MeshRecessionHandler::ComputeMinElementQuality_() const
{
   double local_min = std::numeric_limits<double>::infinity();
   for (int e = 0; e < pmesh_.GetNE(); ++e)
   {
      ElementTransformation *Tr = pmesh_.GetElementTransformation(e);
      const IntegrationRule &ir = IntRules.Get(pmesh_.GetElementBaseGeometry(e), 2);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr->SetIntPoint(&ip);
         local_min = std::min(local_min, static_cast<double>(Tr->Weight()));
      }
   }

   double global_min = local_min;
   MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, pmesh_.GetComm());
   return global_min;
}

RecessionStepOutput MeshRecessionHandler::Advance(const RecessionStepInput &input)
{
   RecessionStepOutput out;
   out.total_recession = total_recession_;

   if (input.dt <= 0.0 || input.top_recession_velocity_true == nullptr)
   {
      *mesh_velocity_ = 0.0;
      *recession_field_ = total_recession_;
      out.min_quality = ComputeMinElementQuality_() / initial_min_quality_;
      return out;
   }

   Vector top_velocity_true;
   ClampTopVelocity_(*input.top_recession_velocity_true, input.dt, top_velocity_true);

   const double top_mean_velocity = ComputeTopMeanVelocity_(top_velocity_true);

   double delta = 0.0;
   if (top_mean_velocity > 0.0)
   {
      SolveVelocityLaplacian_(top_velocity_true);
      MoveMesh_(input.dt, out.max_node_disp);
      delta = top_mean_velocity * input.dt;
      total_recession_ += delta;
   }
   else
   {
      *mesh_velocity_ = 0.0;
   }

   *recession_field_ = total_recession_;

   const double min_quality_raw = ComputeMinElementQuality_();
   if (min_quality_raw <= 0.0)
   {
      throw std::runtime_error(
         "Mesh quality failure: non-positive element Jacobian detected.");
   }

   out.delta_recession = delta;
   out.total_recession = total_recession_;
   out.min_quality = min_quality_raw / initial_min_quality_;
   if (out.min_quality < config_.min_quality_ratio)
   {
      throw std::runtime_error(
         "Mesh quality ratio below configured minimum threshold.");
   }

   return out;
}

const ParGridFunction &MeshRecessionHandler::MeshVelocity() const
{
   return *mesh_velocity_;
}

const ParGridFunction &MeshRecessionHandler::RecessionField() const
{
   return *recession_field_;
}

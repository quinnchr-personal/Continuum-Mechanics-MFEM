#pragma once

#include "mfem.hpp"

#include <memory>
#include <string>
#include <vector>

struct RecessionConfig
{
   int bdr_attr_top = 1;
   int bdr_attr_bottom = 2;
   int bdr_attr_sides = 3;
   std::string mesh_smoothing_model = "laplacian";
   double max_step_recession = 1.0e-3;
   double min_quality_ratio = 0.2;
};

struct RecessionStepInput
{
   double dt = 0.0;
   const mfem::Vector *top_recession_velocity_true = nullptr;
};

struct RecessionStepOutput
{
   double delta_recession = 0.0;
   double total_recession = 0.0;
   double max_node_disp = 0.0;
   double min_quality = 1.0;
};

class MeshRecessionHandler
{
public:
   MeshRecessionHandler(mfem::ParMesh &pmesh,
                        const RecessionConfig &config);

   RecessionStepOutput Advance(const RecessionStepInput &input);

   // Two-phase API: PrepareAdvance computes and stores the mesh velocity via
   // Laplacian smoothing but does NOT move the mesh.  CommitAdvance then moves
   // the mesh and finalises recession bookkeeping.  Call PrepareAdvance first,
   // optionally do work that needs the velocity (e.g. ALE remapping of internal
   // state), then call CommitAdvance.  Advance() is equivalent to calling both
   // in sequence.
   void PrepareAdvance(const RecessionStepInput &input);
   RecessionStepOutput CommitAdvance();

   const mfem::ParFiniteElementSpace &ScalarSpace() const { return *scalar_fes_; }
   const mfem::ParGridFunction &MeshVelocity() const;
   const mfem::ParGridFunction &RecessionField() const;
   double TotalRecession() const { return total_recession_; }

private:
   void EnsureNodalMesh_();
   void AssembleBoundaryMaps_();
   void ClampTopVelocity_(const mfem::Vector &top_velocity_true,
                          const double dt,
                          mfem::Vector &clamped) const;
   double ComputeTopMeanVelocity_(const mfem::Vector &top_velocity_true) const;
   void SolveVelocityLaplacian_(const mfem::Vector &top_velocity_true);
   void MoveMesh_(const double dt,
                  double &max_node_disp);
   double ComputeMinElementQuality_() const;

   mfem::ParMesh &pmesh_;
   RecessionConfig config_;

   std::unique_ptr<mfem::ParFiniteElementSpace> scalar_fes_;
   mfem::ParFiniteElementSpace *vector_fes_ = nullptr; // non-owning, nodal FES
   std::unique_ptr<mfem::ParGridFunction> mesh_velocity_;
   std::unique_ptr<mfem::ParGridFunction> recession_field_;

   struct TopBCMap
   {
      int scalar_tdof = -1;
      int x_tdof = -1;
      int y_tdof = -1;
   };

   mfem::Array<int> top_scalar_tdofs_;
   mfem::Array<int> bottom_scalar_tdofs_;
   mfem::Array<int> ess_vector_tdofs_;
   std::vector<TopBCMap> top_bc_map_;
   std::vector<int> bottom_x_tdofs_;
   std::vector<int> bottom_y_tdofs_;

   double initial_min_quality_ = 1.0;
   double total_recession_ = 0.0;

   // State carried between PrepareAdvance and CommitAdvance.
   double pending_dt_ = 0.0;
   double pending_top_mean_velocity_ = 0.0;
};

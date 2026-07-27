#include "discretization/av_sensor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hycfd
{
namespace
{

// Compact-support smoothstep: 0 for x <= 0.2, 1 for x >= 0.8.
double Ramp(double x)
{
   const double t = (x - 0.2) / 0.6;
   if (t <= 0.0) { return 0.0; }
   if (t >= 1.0) { return 1.0; }
   return t * t * (3.0 - 2.0 * t);
}

} // namespace

void ComputeDilatationAV(const HDGOperator &op, const HDGState &state,
                         const PhysicsModel &physics,
                         const DilatationSensorOptions &options,
                         mfem::ParGridFunction &av)
{
   mfem::ParFiniteElementSpace *space = av.ParFESpace();
   if (!space || space->GetVDim() != 1 ||
       space->FEColl()->GetOrder() != 1)
   {
      throw std::runtime_error(
         "dilatation AV requires an H1 order-1 scalar field");
   }
   mfem::Mesh *mesh = space->GetMesh();
   if (mesh->GetNE() != op.Elements())
   {
      throw std::runtime_error(
         "dilatation AV field lives on a different mesh");
   }
   if (physics.NumComponents() != 4 || physics.Dim() != 2)
   {
      throw std::runtime_error(
         "dilatation sensor is implemented for 2D 4-component flow");
   }

   av = 0.0;
   mfem::Array<int> dofs;
   double uq[12];
   for (int element = 0; element < op.Elements(); ++element)
   {
      const double h = mesh->GetElementSize(element);
      const mfem::IntegrationRule &rule = op.VolumeRule(element);
      double sensor = 0.0;
      for (int qpoint = 0; qpoint < rule.GetNPoints(); ++qpoint)
      {
         op.EvaluateElementState(state, element, rule.IntPoint(qpoint),
                                 uq);
         if (!(uq[0] > 0.0)) { continue; }
         const double inv_rho = 1.0 / uq[0];
         const double u = uq[1] * inv_rho;
         const double v = uq[2] * inv_rho;
         // Physical gradients: slots 4..7 are x-derivatives of the
         // conservative state, 8..11 the y-derivatives.
         const double divergence =
            (uq[5] - u * uq[4]) * inv_rho +
            (uq[10] - v * uq[8]) * inv_rho;
         const double wave_speed = physics.MaxWaveSpeed(uq);
         const double sound =
            std::max(1.0e-12, wave_speed - std::hypot(u, v));
         const double ratio =
            -divergence * h /
            ((op.Order() + 1) * options.kappa * sound);
         sensor = std::max(sensor, wave_speed * Ramp(ratio));
      }
      const double element_av = options.lambda * h * sensor;
      space->GetElementDofs(element, dofs);
      for (int i = 0; i < dofs.Size(); ++i)
      {
         av[dofs[i]] = std::max(av[dofs[i]], element_av);
      }
   }

   // Vertex-max across ranks so the smoothed field is globally continuous.
   if (space->GetParMesh()->GetNRanks() > 1)
   {
      mfem::GroupCommunicator &communicator = space->GroupComm();
      communicator.Reduce<double>(av.GetData(),
                                  mfem::GroupCommunicator::Max);
      communicator.Bcast(av.GetData());
   }
}

} // namespace hycfd

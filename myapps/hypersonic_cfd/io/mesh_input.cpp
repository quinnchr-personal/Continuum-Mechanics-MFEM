#include "io/mesh_input.hpp"

#include <fstream>
#include <stdexcept>

namespace hycfd
{

std::unique_ptr<mfem::Mesh> LoadMeshFile(const std::string &path,
                                         int curved_order)
{
   std::ifstream input(path);
   if (!input)
   {
      throw std::runtime_error("cannot open mesh file: " + path);
   }
   auto mesh = std::make_unique<mfem::Mesh>(input, 1, 1, true);
   if (mesh->Dimension() != 2 || mesh->SpaceDimension() != 2)
   {
      throw std::runtime_error("mesh file must be 2D: " + path);
   }
   if (mesh->GetNE() <= 0)
   {
      throw std::runtime_error("mesh file has no elements: " + path);
   }
   if (mesh->bdr_attributes.Size() == 0)
   {
      throw std::runtime_error(
         "mesh file has no boundary attributes: " + path);
   }
   if (curved_order > 0)
   {
      mesh->SetCurvature(curved_order);
   }
   return mesh;
}

} // namespace hycfd

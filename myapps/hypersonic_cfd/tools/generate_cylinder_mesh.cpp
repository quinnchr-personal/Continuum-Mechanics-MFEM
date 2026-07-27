// Writes the analytic half-annulus cylinder mesh (Q4 geometry) to an MFEM
// mesh file for the file-mesh driver path.
#include "io/exasim_mesh.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char *argv[])
{
   int nr = 6, nc = 12;
   std::string path = "input/cylinder_nr6_nc12_q4.mesh";
   if (argc > 1) { nr = std::atoi(argv[1]); }
   if (argc > 2) { nc = std::atoi(argv[2]); }
   if (argc > 3) { path = argv[3]; }
   std::unique_ptr<mfem::Mesh> mesh = hycfd::BuildAnalyticMesh(nr, nc, 4);
   mesh->Save(path.c_str());
   std::cout << "saved " << path << " ne=" << mesh->GetNE() << '\n';
   return 0;
}

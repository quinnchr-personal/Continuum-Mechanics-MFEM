// Unit disk meshed with straight-sided unstructured triangles of size lc
// (OpenCASCADE), the mesh of the disk driver of myapps/convection_diffusion
// (their Mesh/unit_circle.geo): one boundary group "boundary" (1), the
// surface "domain" (1). The tracked disk_tri.msh is a copy of the myapps
// file (3056 triangles, 128 boundary segments). Its polygon approximates the
// circle to O(lc^2) and uniform refinement keeps it; the disk case poses its
// boundary data as the exact solution on the polygon, so that problem has
// no geometric error, and disk.geo provides curved meshes of the exact
// circle for the rate study on the true domain.
//   gmsh -2 -format msh22 -o disk_tri.msh disk_tri.geo
SetFactory("OpenCASCADE");
DefineConstant[ lc = {0.05, Name "Mesh size"} ];
Point(1) = {0.0, 0.0, 0.0, lc};  // centre
Point(2) = {1.0, 0.0, 0.0, lc};
Point(3) = {0.0, 1.0, 0.0, lc};
Point(4) = {-1.0, 0.0, 0.0, lc};
Point(5) = {0.0, -1.0, 0.0, lc};
Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Physical Curve("boundary", 1) = {1, 2, 3, 4};
Physical Surface("domain", 1) = {1};
Mesh.MshFileVersion = 2.2;

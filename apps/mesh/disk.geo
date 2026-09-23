// Unit disk meshed with unstructured triangles of size lc (OpenCASCADE) for
// the curved meshes of the disk case of apps/input/scalar_transport: meshed
// with third-order geometry (gmsh -order 3), so that the boundary follows the
// circle to O(lc^4) and a study over the sizes lc = 0.1, 0.05, 0.025
// (disk_p3_1.msh, disk_p3_2.msh, disk_p3_3.msh; generated, not refined,
// since MFEM's refinement interpolates the coarse geometry) shows the rates
// of the finite element order on the exact circle. Groups: "boundary" (1),
// "domain" (1).
//   gmsh -2 -order 3 -format msh22 -setnumber lc 0.1 -o disk_p3_1.msh disk.geo
SetFactory("OpenCASCADE");
DefineConstant[ lc = {0.1, Name "Mesh size"} ];
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

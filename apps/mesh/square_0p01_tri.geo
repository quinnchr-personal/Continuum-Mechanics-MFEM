// The square [0, 0.01]^2 meshed with unstructured triangles of size lc
// (OpenCASCADE), the mesh of the nonlinear diffusion driver of
// myapps/convection_diffusion (their Mesh/square_0p01.geo). Physical groups
// as for square_tri.geo. The tracked square_0p01_tri.msh is a copy of the
// myapps file (948 triangles); regenerating it with another Gmsh version may
// change the triangulation.
//   gmsh -2 -format msh22 -o square_0p01_tri.msh square_0p01_tri.geo
SetFactory("OpenCASCADE");
DefineConstant[ lc = {0.0005, Name "Mesh size"} ];
L = 0.01;
Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {L, 0.0, 0.0, lc};
Point(3) = {L, L, 0.0, lc};
Point(4) = {0.0, L, 0.0, lc};
Line(1) = {1, 2}; // bottom
Line(2) = {2, 3}; // right
Line(3) = {3, 4}; // top
Line(4) = {4, 1}; // left
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Physical Curve("bottom", 1) = {1};
Physical Curve("right", 2) = {2};
Physical Curve("top", 3) = {3};
Physical Curve("left", 4) = {4};
Physical Surface("domain", 1) = {1};
Mesh.MshFileVersion = 2.2;

// Quarter of a plate [0, 3R]^2 with a quarter hole of radius R = 0.01 centred at the origin,
// unstructured straight-sided triangles of size 0.01 * 3R: the geometry of the exercise
// myapps/plate_with_hole/plate_with_a_hole.geo, and apps/mesh/plate_with_a_hole.msh is that
// exercise's mesh file itself (10900 nodes), so the runs of apps/input/plate_with_hole can be
// compared with the exercise on the same triangulation. Regenerating gives another one:
//   gmsh -2 -format msh22 -o plate_with_a_hole.msh plate_with_a_hole.geo
// (and with -order 2 a curved hole; the straight-sided mesh carries a geometric error of
// R dtheta^2 / 8 = 1e-4 R on the hole that uniform refinement does not remove).
// Physical groups: curves bottom 1, hole 2, right 3, top 4, left 5; surface "plate" (1).
SetFactory("OpenCASCADE");
// =====================
// Parameters
// =====================
R  = 0.01;   // radius of the quarter hole (center at origin)
Lx = 3*R;   // plate length in x
Ly = 3*R;   // plate height in y
lc = 0.01*Lx;  // target element size

// Required for MFEM to read .msh directly
Mesh.MshFileVersion = 2.2;

// =====================
// Geometry (2D)
// Plate quadrant with quarter-hole cut-out at bottom-left (hole center at origin)
// =====================
Point(1) = {0,  0, 0, lc};   // arc center
Point(2) = {R,  0, 0, lc};   // arc on bottom
Point(3) = {0,  R, 0, lc};   // arc on left
Point(4) = {Lx, 0, 0, lc};   // bottom-right
Point(5) = {Lx, Ly, 0, lc};  // top-right
Point(6) = {0,  Ly, 0, lc};  // top-left

Line(1) = {2, 4};      // bottom
Line(3) = {4, 5};      // right
Line(4) = {5, 6};      // top
Line(5) = {6, 3};      // left
Circle(2) = {3, 1, 2}; // hole arc

Curve Loop(1) = {1, 3, 4, 5, 2};
Plane Surface(1) = {1};

// =====================
// Physical groups
// =====================
Physical Surface("plate", 1) = {1};
Physical Curve("bottom", 1) = {1};
Physical Curve("hole",   2) = {2};
Physical Curve("right",  3) = {3};
Physical Curve("top",    4) = {4};
Physical Curve("left",   5) = {5};

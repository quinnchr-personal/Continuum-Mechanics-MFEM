// Unit square meshed with n x n quadrilaterals (transfinite, recombined).
// Physical groups carry MFEM's Cartesian numbering so numeric attributes stay
// valid: bottom 1, right 2, top 3, left 4; the surface is "domain" (1).
//   gmsh -2 -format msh22 -setnumber n 4 -o square.msh square.geo
DefineConstant[ n = {4, Name "Elements per side"} ];
L = 1.0;
Point(1) = {0, 0, 0};
Point(2) = {L, 0, 0};
Point(3) = {L, L, 0};
Point(4) = {0, L, 0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Transfinite Curve {1, 2, 3, 4} = n + 1;
Transfinite Surface {1};
Recombine Surface {1};
Physical Curve("bottom", 1) = {1};
Physical Curve("right", 2) = {2};
Physical Curve("top", 3) = {3};
Physical Curve("left", 4) = {4};
Physical Surface("domain", 1) = {1};

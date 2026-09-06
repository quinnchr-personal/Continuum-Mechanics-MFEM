// Cook's membrane: the quadrilateral (0,0)-(48,44)-(48,60)-(0,44) meshed with
// n x n quadrilaterals. The transfinite interpolation of straight sides is the
// bilinear map of the unit square, so the mesh coincides with the former
// cartesian + corners input. Physical groups: bottom 1, right 2 (loaded edge),
// top 3, left 4 (clamped edge); surface "domain" (1).
//   gmsh -2 -format msh22 -setnumber n 4 -o cook.msh cook.geo
DefineConstant[ n = {4, Name "Elements per side"} ];
Point(1) = {0, 0, 0};
Point(2) = {48, 44, 0};
Point(3) = {48, 60, 0};
Point(4) = {0, 44, 0};
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

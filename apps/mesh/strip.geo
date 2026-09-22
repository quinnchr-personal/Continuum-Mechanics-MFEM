// Rectangle A <= x <= B, 0 <= y <= H meshed with nr x nz quadrilaterals
// (transfinite, recombined): the meridian section of a thick-walled tube in
// an axisymmetric analysis (x = r, y = z), apps/input/finite_elasticity/
// verification/rivlin_cylinder_axisymmetric.yaml.
//   gmsh -2 -format msh22 -o strip.msh strip.geo
// Physical groups with MFEM's Cartesian numbering: bottom (y = 0) 1, outer
// (x = B) 2, top (y = H) 3, inner (x = A) 4; the surface is "domain" (1).
DefineConstant[ A = {1.0}, B = {2.0}, H = {0.5}, nr = {16}, nz = {2} ];
Point(1) = {A, 0, 0};
Point(2) = {B, 0, 0};
Point(3) = {B, H, 0};
Point(4) = {A, H, 0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Transfinite Curve {1, 3} = nr + 1;
Transfinite Curve {2, 4} = nz + 1;
Transfinite Surface {1};
Recombine Surface {1};
Physical Curve("bottom", 1) = {1};
Physical Curve("outer", 2) = {2};
Physical Curve("top", 3) = {3};
Physical Curve("inner", 4) = {4};
Physical Surface("domain", 1) = {1};

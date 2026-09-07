// Quarter annulus A <= r <= B, x, y >= 0, nr x nt quadrilaterals (transfinite,
// second-order geometry so that uniform refinement follows the arcs):
//   gmsh -2 -order 2 -format msh22 -o annulus.msh annulus.geo
// Physical groups: bottom (y = 0) 1, outer (r = B) 2, left (x = 0) 3,
// inner (r = A) 4; surface "domain" (1). The thick-walled cylinder under
// internal pressure of apps/input/finite_elasticity/cylinder_inflation.yaml.
DefineConstant[ A = {1.0}, B = {2.0}, nr = {4}, nt = {8} ];
Point(1) = {0, 0, 0};
Point(2) = {A, 0, 0};
Point(3) = {B, 0, 0};
Point(4) = {0, B, 0};
Point(5) = {0, A, 0};
Line(1) = {2, 3};
Circle(2) = {3, 1, 4};
Line(3) = {4, 5};
Circle(4) = {5, 1, 2};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Transfinite Curve {1, 3} = nr + 1;
Transfinite Curve {2, 4} = nt + 1;
Transfinite Surface {1};
Recombine Surface {1};
Physical Curve("bottom", 1) = {1};
Physical Curve("outer", 2) = {2};
Physical Curve("left", 3) = {3};
Physical Curve("inner", 4) = {4};
Physical Surface("domain", 1) = {1};

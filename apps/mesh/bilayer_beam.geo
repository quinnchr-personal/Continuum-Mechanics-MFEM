// Bilayer beam of apps/input/anand_coupled_theories/finite_thermoelasticity/04_bilayer_actuator
// (after the reference's bilayer_beam.geo, TE04): L x H, two layers of H/2 with nx x n
// quadrilaterals each (transfinite, recombined; the reference has 200 x 2 crossed
// triangles per layer). Physical groups with MFEM's Cartesian numbering: bottom 1, right 2
// (both layers), top 3, left 4 (both layers); surfaces "bottom_layer" (1) and "top_layer" (2).
//   gmsh -2 -format msh22 -o bilayer_beam.msh bilayer_beam.geo
DefineConstant[ L = {100.0}, H = {1.0}, nx = {200}, n = {2} ];
Point(1) = {0, 0, 0};
Point(2) = {L, 0, 0};
Point(3) = {L, H / 2, 0};
Point(4) = {L, H, 0};
Point(5) = {0, H / 2, 0};
Point(6) = {0, H, 0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 5};
Line(4) = {5, 1};
Line(5) = {3, 4};
Line(6) = {4, 6};
Line(7) = {6, 5};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Curve Loop(2) = {-3, 5, 6, 7};
Plane Surface(2) = {2};
Transfinite Curve {1, 3, 6} = nx + 1;
Transfinite Curve {2, 4, 5, 7} = n + 1;
Transfinite Surface {1};
Transfinite Surface {2};
Recombine Surface {1};
Recombine Surface {2};
Physical Curve("bottom", 1) = {1};
Physical Curve("right", 2) = {2, 5};
Physical Curve("top", 3) = {6};
Physical Curve("left", 4) = {4, 7};
Physical Surface("bottom_layer", 1) = {1};
Physical Surface("top_layer", 2) = {2};

// Rectangle [0, Lx] x [0, Ly] meshed with nx x ny quadrilaterals (transfinite,
// recombined). Physical groups carry MFEM's Cartesian numbering so numeric
// attributes stay valid: bottom (y = 0) 1, right (x = Lx) 2, top (y = Ly) 3,
// left (x = 0) 4; the surface is "domain" (1). In an axisymmetric analysis
// x = r: "left" is the axis.
//   gmsh -2 -format msh22 -setnumber Lx 10 -setnumber Ly 10 -setnumber nx 6 -setnumber ny 6 -o thermo_block.msh rect.geo
DefineConstant[ Lx = {1.0}, Ly = {1.0}, nx = {4}, ny = {4} ];
Point(1) = {0, 0, 0};
Point(2) = {Lx, 0, 0};
Point(3) = {Lx, Ly, 0};
Point(4) = {0, Ly, 0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Transfinite Curve {1, 3} = nx + 1;
Transfinite Curve {2, 4} = ny + 1;
Transfinite Surface {1};
Recombine Surface {1};
Physical Curve("bottom", 1) = {1};
Physical Curve("right", 2) = {2};
Physical Curve("top", 3) = {3};
Physical Curve("left", 4) = {4};
Physical Surface("domain", 1) = {1};

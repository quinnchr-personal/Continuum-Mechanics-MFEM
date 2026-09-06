// Box [0, Lx] x [0, Ly] x [0, Lz] meshed with nx x ny x nz hexahedra
// (transfinite base, extruded with recombination). Physical groups carry
// MFEM's Cartesian numbering: bottom (z = 0) 1, front (y = 0) 2, right
// (x = Lx) 3, back (y = Ly) 4, left (x = 0) 5, top (z = Lz) 6; volume
// "domain" (1).
//   unit cube:  gmsh -3 -format msh22 -setnumber nx 2 -setnumber ny 2 -setnumber nz 2 -o cube.msh box.geo
//   cantilever: gmsh -3 -format msh22 -setnumber Lx 10 -setnumber nx 20 -setnumber ny 2 -setnumber nz 2 -o beam.msh box.geo
DefineConstant[ Lx = {1.0}, Ly = {1.0}, Lz = {1.0}, nx = {2}, ny = {2}, nz = {2} ];
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
// ext[0]: top surface, ext[1]: volume, ext[2..5]: lateral surfaces swept from lines 1..4
ext[] = Extrude {0, 0, Lz} { Surface{1}; Layers{nz}; Recombine; };
Physical Surface("bottom", 1) = {1};
Physical Surface("front", 2) = {ext[2]};
Physical Surface("right", 3) = {ext[3]};
Physical Surface("back", 4) = {ext[4]};
Physical Surface("left", 5) = {ext[5]};
Physical Surface("top", 6) = {ext[0]};
Physical Volume("domain", 1) = {ext[1]};

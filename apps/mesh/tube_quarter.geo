// Quarter of a thick-walled tube Ri <= r <= Ro, 0 <= z <= L (tetrahedra,
// second-order geometry); the inflation example
// apps/input/anand_coupled_theories/finite_elasticity/05_cylinder_inflation.yaml (after the FEniCSx
// example cylinder_inflate.geo of solidmechanicscoupledtheories.github.io):
//   gmsh -3 -order 2 -format msh22 -o tube_quarter.msh tube_quarter.geo
// Physical groups: xplane (x = 0) 1, yplane (y = 0) 2, zbot (z = 0) 3,
// ztop (z = L) 4, inner (r = Ri) 5, outer (r = Ro) 6; volume "domain" (1).
SetFactory("OpenCASCADE");
DefineConstant[ Ri = {10.0}, Ro = {11.0}, L = {5.0}, h = {0.5} ];
Cylinder(1) = {0, 0, 0, 0, 0, L, Ro, Pi/2};
Cylinder(2) = {0, 0, 0, 0, 0, L, Ri, Pi/2};
BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };
MeshSize{ PointsOf{ Volume{3}; } } = h;
eps = 1e-6;
xplane() = Surface In BoundingBox{-eps, -eps, -eps, eps, Ro+eps, L+eps};
yplane() = Surface In BoundingBox{-eps, -eps, -eps, Ro+eps, eps, L+eps};
zbot() = Surface In BoundingBox{-eps, -eps, -eps, Ro+eps, Ro+eps, eps};
ztop() = Surface In BoundingBox{-eps, -eps, L-eps, Ro+eps, Ro+eps, L+eps};
inner() = Surface In BoundingBox{-eps, -eps, -eps, Ri+eps, Ri+eps, L+eps};
outer() = Surface In BoundingBox{-eps, -eps, -eps, Ro+eps, Ro+eps, L+eps};
outer() -= {xplane(), yplane(), zbot(), ztop(), inner()};
Physical Surface("xplane", 1) = {xplane()};
Physical Surface("yplane", 2) = {yplane()};
Physical Surface("zbot", 3) = {zbot()};
Physical Surface("ztop", 4) = {ztop()};
Physical Surface("inner", 5) = {inner()};
Physical Surface("outer", 6) = {outer()};
Physical Volume("domain", 1) = {3};

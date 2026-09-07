// Octant of a thick-walled sphere Ri <= r <= Ro (tetrahedra, second-order
// geometry); the inflation example
// apps/input/finite_elasticity/06_sphere_inflation.yaml (after the FEniCSx
// example spherical_shell.geo of solidmechanicscoupledtheories.github.io):
//   gmsh -3 -order 2 -format msh22 -o sphere_octant.msh sphere_octant.geo
// Physical groups: xplane (x = 0) 1, yplane (y = 0) 2, zplane (z = 0) 3,
// inner (r = Ri) 4, outer (r = Ro) 5; volume "domain" (1).
SetFactory("OpenCASCADE");
DefineConstant[ Ri = {10.0}, Ro = {11.0}, h = {0.75} ];
Sphere(1) = {0, 0, 0, Ro, 0, Pi/2, Pi/2};
Sphere(2) = {0, 0, 0, Ri, 0, Pi/2, Pi/2};
BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };
MeshSize{ PointsOf{ Volume{3}; } } = h;
eps = 1e-6;
xplane() = Surface In BoundingBox{-eps, -eps, -eps, eps, Ro+eps, Ro+eps};
yplane() = Surface In BoundingBox{-eps, -eps, -eps, Ro+eps, eps, Ro+eps};
zplane() = Surface In BoundingBox{-eps, -eps, -eps, Ro+eps, Ro+eps, eps};
inner() = Surface In BoundingBox{-eps, -eps, -eps, Ri+eps, Ri+eps, Ri+eps};
outer() = Surface In BoundingBox{-eps, -eps, -eps, Ro+eps, Ro+eps, Ro+eps};
outer() -= {xplane(), yplane(), zplane(), inner()};
Physical Surface("xplane", 1) = {xplane()};
Physical Surface("yplane", 2) = {yplane()};
Physical Surface("zplane", 3) = {zplane()};
Physical Surface("inner", 4) = {inner()};
Physical Surface("outer", 5) = {outer()};
Physical Volume("domain", 1) = {3};

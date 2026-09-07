// Solid cylinder of radius R and length L along z (tetrahedra, second-order
// geometry), the fixed-end torsion example of
// apps/input/finite_elasticity/03_cylinder_torsion.yaml (after the FEniCSx
// example of solidmechanicscoupledtheories.github.io):
//   gmsh -3 -order 2 -format msh22 -o cylinder_torsion.msh cylinder_torsion.geo
// Physical groups: bottom (z = 0) 1, top (z = L) 2, wall 3; volume "domain" (1).
SetFactory("OpenCASCADE");
DefineConstant[ R = {12.7}, L = {25.4}, h = {2.5} ];
Cylinder(1) = {0, 0, 0, 0, 0, L, R, 2*Pi};
MeshSize{ PointsOf{ Volume{1}; } } = h;
eps = 1e-6;
bot() = Surface In BoundingBox{-R-eps, -R-eps, -eps, R+eps, R+eps, eps};
top() = Surface In BoundingBox{-R-eps, -R-eps, L-eps, R+eps, R+eps, L+eps};
Physical Surface("bottom", 1) = {bot()};
Physical Surface("top", 2) = {top()};
Physical Surface("wall", 3) = {1};
Physical Volume("domain", 1) = {1};

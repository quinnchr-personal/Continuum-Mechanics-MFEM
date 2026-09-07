// Quarter of a plate with a hole (L0 x W0 x t0 = 15 x 10 x 1, hole radius a = 3
// at the corner), tetrahedra with second-order geometry; the tension example
// apps/input/anand_coupled_theories/finite_elasticity/04_plate_with_hole.yaml (after the FEniCSx
// example 3D_hip_v2.geo of solidmechanicscoupledtheories.github.io):
//   gmsh -3 -order 2 -format msh22 -o plate_hole.msh plate_hole.geo
// Physical groups: xbot (x = 0) 1, ybot (y = 0) 2, xtop (x = L0) 3, ytop (y = W0) 4,
// hole 5, zbot (z = 0) 6, ztop (z = t0) 7; volume "domain" (1).
SetFactory("OpenCASCADE");
DefineConstant[ L0 = {15.0}, W0 = {10.0}, t0 = {1.0}, a = {3.0}, h = {1.0} ];
Rectangle(1) = {0, 0, 0, L0, W0};
Disk(2) = {0, 0, 0, a};
BooleanDifference(3) = { Surface{1}; Delete; }{ Surface{2}; Delete; };
ext[] = Extrude {0, 0, t0} { Surface{3}; Layers{1}; };
MeshSize{ PointsOf{ Volume{ext[1]}; } } = h;
eps = 1e-6;
xbot() = Surface In BoundingBox{-eps, a-eps, -eps, eps, W0+eps, t0+eps};
ybot() = Surface In BoundingBox{a-eps, -eps, -eps, L0+eps, eps, t0+eps};
xtop() = Surface In BoundingBox{L0-eps, -eps, -eps, L0+eps, W0+eps, t0+eps};
ytop() = Surface In BoundingBox{-eps, W0-eps, -eps, L0+eps, W0+eps, t0+eps};
hole() = Surface In BoundingBox{-eps, -eps, -eps, a+eps, a+eps, t0+eps};
zbot() = Surface In BoundingBox{-eps, -eps, -eps, L0+eps, W0+eps, eps};
ztop() = Surface In BoundingBox{-eps, -eps, t0-eps, L0+eps, W0+eps, t0+eps};
Physical Surface("xbot", 1) = {xbot()};
Physical Surface("ybot", 2) = {ybot()};
Physical Surface("xtop", 3) = {xtop()};
Physical Surface("ytop", 4) = {ytop()};
Physical Surface("hole", 5) = {hole()};
Physical Surface("zbot", 6) = {zbot()};
Physical Surface("ztop", 7) = {ztop()};
Physical Volume("domain", 1) = {ext[1]};

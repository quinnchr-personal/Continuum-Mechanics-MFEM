// Quarter of a square plate of half-width W with a central hole of radius a
// (quadrilaterals, second-order geometry, size ha at the hole growing to hw
// at the outer edges); Kirsch's stress concentration at small strain
// (apps/input/finite_elasticity/verification/kirsch_plate_with_hole.yaml):
//   gmsh -2 -order 2 -format msh22 -o plate_hole_2d.msh plate_hole_2d.geo
// Physical groups: bottom (y = 0) 1, right (x = W) 2, top (y = W) 3,
// left (x = 0) 4, hole 5; surface "domain" (1).
SetFactory("OpenCASCADE");
DefineConstant[ W = {20.0}, a = {1.0}, ha = {0.04}, hw = {1.5} ];
Rectangle(1) = {0, 0, 0, W, W};
Disk(2) = {0, 0, 0, a};
BooleanDifference(3) = { Surface{1}; Delete; }{ Surface{2}; Delete; };
eps = 1e-6;
hole() = Curve In BoundingBox{-eps, -eps, -eps, a+eps, a+eps, eps};
outer() = Curve In BoundingBox{-eps, -eps, -eps, W+eps, W+eps, eps};
outer() -= hole();
MeshSize{ PointsOf{ Curve{outer()}; } } = hw;
MeshSize{ PointsOf{ Curve{hole()}; } } = ha;   // last: the axis lines share the hole's end points
Recombine Surface{3};
Physical Curve("bottom", 1) = {Curve In BoundingBox{a-eps, -eps, -eps, W+eps, eps, eps}};
Physical Curve("right", 2) = {Curve In BoundingBox{W-eps, -eps, -eps, W+eps, W+eps, eps}};
Physical Curve("top", 3) = {Curve In BoundingBox{-eps, W-eps, -eps, W+eps, W+eps, eps}};
Physical Curve("left", 4) = {Curve In BoundingBox{-eps, a-eps, -eps, eps, W+eps, eps}};
Physical Curve("hole", 5) = {Curve In BoundingBox{-eps, -eps, -eps, a+eps, a+eps, eps}};
Physical Surface("domain", 1) = {3};

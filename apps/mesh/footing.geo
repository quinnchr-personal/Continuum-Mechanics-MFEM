// Cube of side L with the quarter patch x, y <= L/2 of its top face as a
// separate surface (tetrahedra); the partial-face pressure example
// apps/input/anand_coupled_theories/finite_elasticity/07_cube_footing.yaml (after the FEniCSx
// example of solidmechanicscoupledtheories.github.io):
//   gmsh -3 -format msh22 -o footing.msh footing.geo
// Physical groups: left (x = 0) 1, front (y = 0) 2, bottom (z = 0) 3,
// patch (z = L, x, y <= L/2) 4, top_rest 5, right (x = L) 6, back (y = L) 7;
// volume "domain" (1).
SetFactory("OpenCASCADE");
DefineConstant[ L = {50.0}, h = {5.0} ];
Box(1) = {0, 0, 0, L, L, L};
Rectangle(100) = {0, 0, L, L/2, L/2};
BooleanFragments{ Volume{1}; Delete; }{ Surface{100}; Delete; }
MeshSize{ PointsOf{ Volume{1}; } } = h;
eps = 1e-6;
left() = Surface In BoundingBox{-eps, -eps, -eps, eps, L+eps, L+eps};
front() = Surface In BoundingBox{-eps, -eps, -eps, L+eps, eps, L+eps};
bottom() = Surface In BoundingBox{-eps, -eps, -eps, L+eps, L+eps, eps};
right() = Surface In BoundingBox{L-eps, -eps, -eps, L+eps, L+eps, L+eps};
back() = Surface In BoundingBox{-eps, L-eps, -eps, L+eps, L+eps, L+eps};
patch() = Surface In BoundingBox{-eps, -eps, L-eps, L/2+eps, L/2+eps, L+eps};
top() = Surface In BoundingBox{-eps, -eps, L-eps, L+eps, L+eps, L+eps};
top() -= {patch()};
Physical Surface("left", 1) = {left()};
Physical Surface("front", 2) = {front()};
Physical Surface("bottom", 3) = {bottom()};
Physical Surface("patch", 4) = {patch()};
Physical Surface("top_rest", 5) = {top()};
Physical Surface("right", 6) = {right()};
Physical Surface("back", 7) = {back()};
Physical Volume("domain", 1) = {1};

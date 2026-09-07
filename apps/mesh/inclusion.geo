// Octant of a cube of side L with a spherical inclusion of radius a centred at
// the corner (the eighth-symmetry model of a 2L cube with a centred sphere),
// tetrahedra graded towards the inclusion; the two-material example
// apps/input/anand_coupled_theories/finite_elasticity/09_spherical_inclusion.yaml (after the FEniCSx
// example sphere_inclusion.geo of solidmechanicscoupledtheories.github.io):
//   gmsh -3 -format msh22 -o inclusion.msh inclusion.geo
// Physical groups: volumes inclusion 1, matrix 2; surfaces left (x = 0) 1,
// front (y = 0) 2, bottom (z = 0) 3, right (x = L) 4, back (y = L) 5, top (z = L) 6.
SetFactory("OpenCASCADE");
DefineConstant[ L = {10.0}, a = {5.0}, hin = {1.0}, hout = {2.5} ];
Box(1) = {0, 0, 0, L, L, L};
Sphere(2) = {0, 0, 0, a, 0, Pi/2, Pi/2};
BooleanFragments{ Volume{1}; Delete; }{ Volume{2}; Delete; }
eps = 1e-6;
inc() = Volume In BoundingBox{-eps, -eps, -eps, a+eps, a+eps, a+eps};
all() = Volume{:};
mat() = all();
mat() -= inc();
MeshSize{ PointsOf{ Volume{inc()}; } } = hin;
MeshSize{ PointsOf{ Volume{mat()}; } } = hout;
MeshSize{ PointsOf{ Volume{inc()}; } } = hin;
left() = Surface In BoundingBox{-eps, -eps, -eps, eps, L+eps, L+eps};
front() = Surface In BoundingBox{-eps, -eps, -eps, L+eps, eps, L+eps};
bottom() = Surface In BoundingBox{-eps, -eps, -eps, L+eps, L+eps, eps};
right() = Surface In BoundingBox{L-eps, -eps, -eps, L+eps, L+eps, L+eps};
back() = Surface In BoundingBox{-eps, L-eps, -eps, L+eps, L+eps, L+eps};
top() = Surface In BoundingBox{-eps, -eps, L-eps, L+eps, L+eps, L+eps};
Physical Volume("inclusion", 1) = {inc()};
Physical Volume("matrix", 2) = {mat()};
Physical Surface("left", 1) = {left()};
Physical Surface("front", 2) = {front()};
Physical Surface("bottom", 3) = {bottom()};
Physical Surface("right", 4) = {right()};
Physical Surface("back", 5) = {back()};
Physical Surface("top", 6) = {top()};

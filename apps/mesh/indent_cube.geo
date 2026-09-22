// Cube of side L indented by a rigid sphere at the corner of its top face
// (solidmechanicscoupledtheories.github.io, finite viscoelasticity FV12, after the
// reference's sphere_indent.geo): tetrahedra of size h_in in the box x, y <= L/5,
// z >= 4L/5 under the indenter and h_out elsewhere.
//   gmsh -3 -format msh22 -o indent_cube.msh indent_cube.geo
// Physical groups: left (x = 0) 1, front (y = 0) 2, bottom (z = 0) 3, top (z = L) 4,
// right (x = L) 5, back (y = L) 6; volume "domain" (1).
SetFactory("OpenCASCADE");
DefineConstant[ L = {50.0}, h_in = {1.0}, h_out = {10.0} ];
Box(1) = {0, 0, 0, L, L, L};
MeshSize{ PointsOf{ Volume{1}; } } = h_out;
Field[1] = Box;
Field[1].Thickness = L;
Field[1].VIn = h_in;
Field[1].VOut = h_out;
Field[1].XMin = 0; Field[1].XMax = L/5;
Field[1].YMin = 0; Field[1].YMax = L/5;
Field[1].ZMin = 4*L/5; Field[1].ZMax = L;
Background Field = 1;
eps = 1e-6;
left() = Surface In BoundingBox{-eps, -eps, -eps, eps, L+eps, L+eps};
front() = Surface In BoundingBox{-eps, -eps, -eps, L+eps, eps, L+eps};
bottom() = Surface In BoundingBox{-eps, -eps, -eps, L+eps, L+eps, eps};
top() = Surface In BoundingBox{-eps, -eps, L-eps, L+eps, L+eps, L+eps};
right() = Surface In BoundingBox{L-eps, -eps, -eps, L+eps, L+eps, L+eps};
back() = Surface In BoundingBox{-eps, L-eps, -eps, L+eps, L+eps, L+eps};
Physical Surface("left", 1) = {left()};
Physical Surface("front", 2) = {front()};
Physical Surface("bottom", 3) = {bottom()};
Physical Surface("top", 4) = {top()};
Physical Surface("right", 5) = {right()};
Physical Surface("back", 6) = {back()};
Physical Volume("domain", 1) = {1};

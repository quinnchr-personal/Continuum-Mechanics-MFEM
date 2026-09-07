// Unit cube meshed with tetrahedra (mesh size h), for the three-dimensional
// manufactured solutions on simplices (apps/input/finite_elasticity/verification/
// manufactured_solutions/mms_3d_tet.yaml):
//   gmsh -3 -format msh22 -o cube_tet.msh cube_tet.geo
// Physical groups as box.geo: bottom (z = 0) 1, front (y = 0) 2, right (x = 1) 3,
// back (y = 1) 4, left (x = 0) 5, top (z = 1) 6; volume "domain" (1).
SetFactory("OpenCASCADE");
DefineConstant[ h = {0.5} ];
Box(1) = {0, 0, 0, 1, 1, 1};
MeshSize{ PointsOf{ Volume{1}; } } = h;
eps = 1e-6;
Physical Surface("bottom", 1) = {Surface In BoundingBox{-eps, -eps, -eps, 1+eps, 1+eps, eps}};
Physical Surface("front", 2) = {Surface In BoundingBox{-eps, -eps, -eps, 1+eps, eps, 1+eps}};
Physical Surface("right", 3) = {Surface In BoundingBox{1-eps, -eps, -eps, 1+eps, 1+eps, 1+eps}};
Physical Surface("back", 4) = {Surface In BoundingBox{-eps, 1-eps, -eps, 1+eps, 1+eps, 1+eps}};
Physical Surface("left", 5) = {Surface In BoundingBox{-eps, -eps, -eps, eps, 1+eps, 1+eps}};
Physical Surface("top", 6) = {Surface In BoundingBox{-eps, -eps, 1-eps, 1+eps, 1+eps, 1+eps}};
Physical Volume("domain", 1) = {1};

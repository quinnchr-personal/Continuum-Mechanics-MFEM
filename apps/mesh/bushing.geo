// Rubber bushing of the viscoelastic shear example (solidmechanicscoupledtheories.github.io,
// finite viscoelasticity FV09, after the reference's bushing.geo): the solid of revolution
// about the y axis of the rectangle 0 <= x <= R, 0 <= y <= L whose outer edge is the
// circular arc through (R, 0) and (R, L) centred at (R + 6.8, L/2), a waist of radius
// R - 5 at mid-height. Second-order tetrahedra of size h:
//   gmsh -3 -order 2 -format msh22 -o bushing.msh bushing.geo
// Physical groups: top (y = L) 1, bottom (y = 0) 2, lateral 3; volume "domain" (1).
SetFactory("OpenCASCADE");
DefineConstant[ R = {22.5}, L = {19.3}, h = {3.0} ];
Point(1) = {0, 0, 0, h};
Point(2) = {0, L, 0, h};
Point(3) = {R, L, 0, h};
Point(4) = {R, 0, 0, h};
Point(5) = {R + 6.8, L/2, 0, h};
Line(1) = {4, 1};
Line(2) = {1, 2};
Line(3) = {2, 3};
Circle(4) = {3, 5, 4};
Curve Loop(1) = {3, 4, 1, 2};
Plane Surface(1) = {1};
Extrude {{0, 1, 0}, {0, 0, 0}, 2*Pi} { Surface{1}; }
MeshSize{ PointsOf{ Volume{1}; } } = h;
eps = 1e-6;
top() = Surface In BoundingBox{-R-eps, L-eps, -R-eps, R+eps, L+eps, R+eps};
bottom() = Surface In BoundingBox{-R-eps, -eps, -R-eps, R+eps, eps, R+eps};
all() = Boundary{ Volume{1}; };
lateral() = all();
lateral() -= {top()};
lateral() -= {bottom()};
Physical Surface("top", 1) = {top()};
Physical Surface("bottom", 2) = {bottom()};
Physical Surface("lateral", 3) = {lateral()};
Physical Volume("domain", 1) = {1};

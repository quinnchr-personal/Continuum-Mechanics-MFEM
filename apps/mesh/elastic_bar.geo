// Bar L x L x H = 0.3 x 0.3 x 1 along z, unstructured tetrahedra of size h = 0.05: the
// geometry of the exercise myapps/elastic_bar/mesh.geo, and apps/mesh/elastic_bar.msh is
// that exercise's mesh file itself (954 nodes), so the runs of
// apps/input/elastic_bar/*.yaml can be compared with its results degree of freedom for
// degree of freedom. Regenerating with another Gmsh version gives another triangulation:
//   gmsh -3 -format msh22 -o elastic_bar.msh elastic_bar.geo
// Physical groups: surfaces back (z = 0) 1, right 2, bottom 3, left 4, top 5,
// front (z = H) 6; volume "vol" (7).
H = 1;
L = 0.3;
h = 0.05;

Point(1) = {0.0,0.0,0.0,h};
Point(2) = {L,0.0,0.0,h};
Point(3) = {L,L,0.0,h};
Point(4) = {0,L,0.0,h};
Line(1) = {4,3};
Line(2) = {3,2};
Line(3) = {2,1};
Line(4) = {1,4};
Line Loop(5) = {2,3,4,1};
Plane Surface(6) = {5};
Extrude {0,0.0,H} {
  Surface{6};
}
Physical Surface("back") = {6};
Physical Surface("right") = {15};
Physical Surface("bottom") = {19};
Physical Surface("left") = {23};
Physical Surface("top") = {27};
Physical Surface("front") = {28};
Physical Volume("vol")={1};

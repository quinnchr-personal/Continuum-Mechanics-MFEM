SetFactory("OpenCASCADE");

lc = 0.05;

Point(1) = {0.0, 0.0, 0.0, lc};  // center
Point(2) = {1.0, 0.0, 0.0, lc};  // east
Point(3) = {0.0, 1.0, 0.0, lc};  // north
Point(4) = {-1.0, 0.0, 0.0, lc}; // west
Point(5) = {0.0, -1.0, 0.0, lc}; // south

Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Physical Curve("boundary", 1) = {1, 2, 3, 4};
Physical Surface("domain", 1) = {1};

Mesh.MshFileVersion = 2.2;

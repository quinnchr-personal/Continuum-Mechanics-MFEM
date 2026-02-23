SetFactory("OpenCASCADE");

Lx = 0.005;
Ly = 0.05;
h1 = 0.0001;
h2 = 0.01;

Point(1) = {0.0, 0.0, 0.0, h2};
Point(2) = {Lx, 0.0, 0.0, h2};
Point(3) = {Lx, Ly, 0.0, h1};
Point(4) = {0.0, Ly, 0.0, h1};

Line(1) = {1, 2}; // bottom
Line(2) = {2, 3}; // right
Line(3) = {3, 4}; // top
Line(4) = {4, 1}; // left

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// No recombination -> triangular elements.
Physical Curve("top", 1) = {3};
Physical Curve("bottom", 2) = {1};
Physical Curve("sides", 3) = {2, 4};
Physical Surface("domain", 1) = {1};

Mesh.MshFileVersion = 2.2;

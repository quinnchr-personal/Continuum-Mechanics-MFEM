SetFactory("OpenCASCADE");

Lx = 0.0025;
Ly = 0.05;

Point(1) = {0.0, 0.0, 0.0, 1.0};
Point(2) = {Lx, 0.0, 0.0, 1.0};
Point(3) = {Lx, Ly, 0.0, 1.0};
Point(4) = {0.0, Ly, 0.0, 1.0};

Line(1) = {1, 2}; // bottom
Line(2) = {2, 3}; // right
Line(3) = {3, 4}; // top
Line(4) = {4, 1}; // left

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

nx = 2;
ny = 100;
grade = 1.03;

Transfinite Curve {1, 3} = nx Using Progression 1.0;
Transfinite Curve {2} = ny Using Progression (1.0 / grade); // refined near top
Transfinite Curve {4} = ny Using Progression grade;         // refined near top
Transfinite Surface {1};
Recombine Surface {1};

Physical Curve("top", 1) = {3};
Physical Curve("bottom", 2) = {1};
Physical Curve("sides", 3) = {2, 4};
Physical Surface("domain", 1) = {1};

Mesh.MshFileVersion = 2.2;

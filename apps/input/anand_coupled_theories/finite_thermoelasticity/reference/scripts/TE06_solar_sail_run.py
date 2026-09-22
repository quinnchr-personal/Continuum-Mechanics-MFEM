# Patched from the reference notebook: headless (pyvista/IPython stubbed), histories saved to hist/<case>.csv.
import sys
from unittest.mock import MagicMock
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot
sys.modules["pyvista"] = MagicMock()
# Import FEnicSx/dolfinx
import dolfinx

# For numerical arrays
import numpy as np

# For MPI-based parallelization
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# PETSc solvers
from petsc4py import PETSc

# Specific functions from dolfinx modules
from dolfinx import fem, mesh, io, plot, log
from dolfinx.fem import (Constant, dirichletbc, Function, functionspace, Expression )
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import VTXWriter, XDMFFile

# Specific functions from ufl modules
import ufl
from ufl import (TestFunctions, TrialFunction, Identity, grad, det, div, dev, inv, tr, sqrt, conditional ,\
                 gt, dx, inner, derivative, dot, ln, split, exp, eq, cos, sin, acos, ge, le, outer, tanh,\
                 cosh, atan, atan2)

# basix finite elements (necessary for dolfinx v0.8.0)
import basix
from basix.ufl import element, mixed_element

# Matplotlib for plotting
import matplotlib.pyplot as plt
plt.close('all')

# For timing the code
from datetime import datetime

# Set level of detail for log messages (integer)
# Guide:
# CRITICAL  = 50, // errors that may lead to data corruption
# ERROR     = 40, // things that HAVE gone wrong
# WARNING   = 30, // things that MAY go wrong later
# INFO      = 20, // information of general interest (includes solver info)
# PROGRESS  = 16, // what's happening (broadly)
# TRACE     = 13, // what's happening (in detail)
# DBG       = 10  // sundry
#
log.set_log_level(log.LogLevel.WARNING)

# %% ----
# Create mesh 
L0 = 100.0  # Length of plate, mm
T0 = 1.0    # Thickness of plate, mm
domain = mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L0, L0, T0]],\
                         [10, 10, 2], mesh.CellType.tetrahedron)

x = ufl.SpatialCoordinate(domain)

# %% ----
# Identify the planar boundaries of the  box mesh
def xBot(x):
    return np.isclose(x[0], 0)
def xTop(x):
    return np.isclose(x[0], L0)
def yBot(x):
    return np.isclose(x[1], 0)
def yTop(x):
    return np.isclose(x[1], L0)
def zBot(x):
    return np.isclose(x[2], 0)
def zTop(x):
    return  np.isclose(x[2], T0) 


# Identify the corners of the domain which will be fixed 
def pin1(x):
    return np.logical_and(np.isclose(x[0], 0), np.isclose(x[1], 0))

def pin2(x):
    return np.logical_and(np.isclose(x[0], L0), np.isclose(x[1], L0))

# Mark the sub-domains
boundaries = [(1,xBot),(2,xTop),(3,yBot),(4,yTop),(5,zBot),(6,zTop)]

# build collections of facets on each subdomain and mark them appropriately.
facet_indices, facet_markers = [], [] # initalize empty collections of indices and markers.
fdim = domain.topology.dim - 1 # geometric dimension of the facet (mesh dimension - 1)
for (marker, locator) in boundaries:
    facets = mesh.locate_entities(domain, fdim, locator) # an array of all the facets in a 
                                                         # given subdomain ("locator")
    facet_indices.append(facets)                         # add these facets to the collection.
    facet_markers.append(np.full_like(facets, marker))   # mark them with the appropriate index.

# Format the facet indices and markers as required for use in dolfinx.
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
# 
# Add these marked facets as "mesh tags" for later use in BCs.
facet_tags = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

# %% ----
top_imap = domain.topology.index_map(2)      # index map of 2D entities in domain (facets)
values = np.zeros(top_imap.size_global)      # an array of zeros of the same size as number of 2D entities
values[facet_tags.indices]=facet_tags.values # populating the array with facet tag index numbers
print(np.unique(facet_tags.values))          # printing the unique indices

# Surface numbering:
# boundaries = [(1,xBot),(2,xTop),(3,yBot),(4,yTop),(5,zBot),(6,zTop)]

# %% ----
import pyvista
pyvista.set_jupyter_backend('html')
from dolfinx.plot import vtk_mesh
pyvista.start_xvfb()

# initialize a plotter
plotter = pyvista.Plotter()

# Add the mesh.
topology, cell_types, geometry = plot.vtk_mesh(domain, domain.topology.dim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
plotter.add_mesh(grid, show_edges=True)#, opacity=0.25) 

# Add colored 2D surfaces for the named surfaces
xBot_surf = pyvista.UnstructuredGrid(*vtk_mesh(domain, domain.topology.dim-1,facet_tags.indices[facet_tags.values==1]) )
yBot_surf = pyvista.UnstructuredGrid(*vtk_mesh(domain, domain.topology.dim-1,facet_tags.indices[facet_tags.values==3]) )
zBot_surf = pyvista.UnstructuredGrid(*vtk_mesh(domain, domain.topology.dim-1,facet_tags.indices[facet_tags.values==5]) )
#
actor  = plotter.add_mesh(xBot_surf, show_edges=True,color="yellow") # xBot face is blue
actor2 = plotter.add_mesh(yBot_surf, show_edges=True,color="red")    # yBot is red
actor3 = plotter.add_mesh(zBot_surf, show_edges=True,color="blue")   # zBot is green


labels = dict(xlabel='X', ylabel='Y',zlabel='Z')
plotter.add_axes(**labels)

plotter.screenshot("results/plate_mesh.png")

Image = lambda *a, **k: None
Image(filename='results/plate_mesh.png') 

# #Use the following  commands for a  zoom-able  view
# if not pyvista.OFF_SCREEN:
#     plotter.show()
# else:
#     plotter.screenshot("mesh.png")

# %% ----
# Define the boundary integration measure "ds" using the facet tags,
# also specify the number of surface quadrature points.
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags, metadata={'quadrature_degree':2})

# Define the volume integration measure "dx" 
# also specify the number of volume quadrature points.
dx = ufl.Measure('dx', domain=domain, metadata={'quadrature_degree': 2})

#  Define facet normal
# n = ufl.FacetNormal(domain)
n = ufl.FacetNormal(domain)

# %% ----
k_B      = Constant(domain,PETSc.ScalarType(1.38E-17))             # Boltzmann's constant
theta0   = Constant(domain,PETSc.ScalarType(273))                  # Initial temperature
Gshear_0 = Constant(domain,PETSc.ScalarType(280))                  # Ground sate shear modulus
N_R      = Constant(domain,PETSc.ScalarType(Gshear_0/(k_B*theta0)))# Number polymer chains per unit ref. volume
lambdaL  = Constant(domain,PETSc.ScalarType(5.12))                 # Locking stretch
Kbulk    = Constant(domain,PETSc.ScalarType(1000*Gshear_0))        # Bulk modulus
alpha    = Constant(domain,PETSc.ScalarType(180.0E-6))             # Coefficient of thermal expansion
c_v      = Constant(domain,PETSc.ScalarType(1839))                 # Specific heat
k_therm  = Constant(domain,PETSc.ScalarType(0.16E3))               # Thermal conductivity

# %% ----
t    = 0.0           
Ttot = 100
dt   = 1.0         # Fixed  time step size

# Temperature change
deltaTheta = Constant(domain,PETSc.ScalarType(50))  # temperature increase of 50 K

def thetaRamp(t):
    return theta0 + deltaTheta*t/Ttot

# Maximum pressure
press_max = Constant(domain,PETSc.ScalarType(10.0E-3))  # kPa

def pressRamp(t):
    return press_max*t/Ttot

# %% ----
# Define function space, both vectorial and scalar
#  
U2 = element("Lagrange", domain.basix_cell(), 2, shape=(3,)) # For displacement
P1 = element("Lagrange", domain.basix_cell(), 1)             # For  pressure and temperature                                        
#
TH = mixed_element([U2, P1, P1])  # Taylor-Hood style mixed element
ME = functionspace(domain, TH)    # Total space for all DOFs

# Define actual functions with the required DOFs
w = Function(ME)
u, p, theta = split(w)  # displacement u, pressure p, temperature theta

# A copy of functions to store values in the previous step for time-stepping
w_old = Function(ME)
u_old,  p_old, theta_old = split(w_old)   

# Define test functions    
u_test, p_test,  theta_test = TestFunctions(ME)  

# Define trial functions needed for automatic differentiation
dw = TrialFunction(ME)                  


# %% ----

# Assign initial  temperature  theta0 to the whole domain 
w.sub(2).interpolate(lambda x: np.full((x.shape[1],),  theta0))
w_old.sub(2).interpolate(lambda x: np.full((x.shape[1],),  theta0))   

# %% ----
# Deformation gradient 
def F_calc(u):
    Id = Identity(3) 
    F  = Id + grad(u)
    return F

def lambdaBar_calc(u):
    F = F_calc(u)
    C = F.T*F
    Cdis = J**(-2/3)*C
    I1 = tr(Cdis)
    lambdaBar = sqrt(I1/3.0)
    return lambdaBar

def zeta_calc(u):
    lambdaBar = lambdaBar_calc(u)
    # Use Pade approximation of Langevin inverse
    z    = lambdaBar/lambdaL
    z    = conditional(gt(z,0.95), 0.95, z) # Keep simulation from blowing up
    beta = z*(3.0 - z**2.0)/(1.0 - z**2.0)
    zeta = (lambdaL/(3*lambdaBar))*beta
    return zeta

#----------------------------------------------
# Subroutine for calculating the Piola  stress
#----------------------------------------------
def Piola_calc(u, p, theta):
    F = F_calc(u)
    J = det(F)
    C = F.T*F
    #
    zeta = zeta_calc(u)
    Gshear  = N_R * k_B * theta * zeta
    #
    Piola = J**(-2/3) * Gshear * (F - (1/3)*tr(C)*inv(F.T) ) - J * p * inv(F.T)
    return Piola

#--------------------------------------------------------------
# Subroutine for calculating the stress temperature tensor
#--------------------------------------------------------------
def M_calc(u):
    Id  = Identity(3)         
    F   = F_calc(u) 
    #
    C  = F.T*F
    Cinv = inv(C) 
    J = det(F)
    zeta = zeta_calc(u)
    #
    fac1 = N_R * k_B * zeta
    fac2 = (3*Kbulk*alpha)/J
    #
    M =  J**(-2/3) * fac1 * (Id - (1/3)*tr(C)*Cinv)  - J * fac2 * Cinv
    return M

#--------------------------------------------------------------
# Subroutine for calculating the heat flux
#--------------------------------------------------------------
#  Heat flux
def Heat_flux_calc(u, theta):
    F = F_calc(u) 
    J = det(F)         
    #
    Cinv = inv(F.T*F) 
    #
    Tcond = J * k_therm * Cinv # Thermal conductivity tensor
    #
    Qmat = - Tcond * grad(theta)
    return Qmat

#--------------------------------------------------------------------------------
# Subroutine for calculating the principal Cauchy stress  for visualization if needed
#--------------------------------------------------------------------------------
def tensor_eigs(T):
    # invariants of T
    I1 = tr(T) 
    I2 = (1/2)*(tr(T)**2 - tr(T*T))
    I3 = det(T)
    
    # Intermediate quantities b, c, d
    b = -I1
    c = I2
    d = -I3
    
    # intermediate quantities E, F, G
    E = (3*c - b*b)/3
    F = (2*(b**3) - 9*b*c + 27*d)/27
    G = (F**2)/4 + (E**3)/27
    
    # Intermediate quantities H, I, J, K, L
    H = sqrt(-(E**3)/27)
    I = H**(1/3)
    J = acos(-F/(2*H))
    K = cos(J/3)
    L = sqrt(3)*sin(J/3)
    
    # Finally, the (not necessarily ordered) eigenvalues
    t1 = 2*I*K - b/3
    t2 = -I*(K+L) - b/3
    t3 = -I*(K-L) - b/3
    
    # Order the eigenvalues using conditionals
    #
    T1_temp = conditional(lt(t1, t3), t3, t1 ) # returns the larger of t1 and t3.
    T1 = conditional(lt(T1_temp, t2), t2, T1_temp ) # returns the larger of T1_temp and t2.
    #
    T3_temp = conditional(gt(t3, t1), t1, t3 ) # returns the smaller of t1 and t3.
    T3 = conditional(gt(T3_temp, t2), t2, T1_temp ) # returns the smaller of T3_temp and t2.
    #
    # use the trace to report the middle eigenvalue.
    T2 = I1 - T1 - T3
    
    return T1, T2, T3

# %% ----
# Kinematics
F = F_calc(u)
J = det(F)   
#
lambdaBar = lambdaBar_calc(u)
#
F_old = F_calc(u_old)
J_old = det(F_old)   
#
C     = F.T*F
C_old = F_old.T*F_old

#  Piola stress
Piola = Piola_calc(u, p, theta)

# Calculate the stress-temperature tensor
M = M_calc(u)

# Calculate the heat flux
Qmat = Heat_flux_calc(u,theta)

# %% ----
# Residuals:
# Res_0: Balance of forces (test fxn: u)
# Res_1: Coupling pressure (test fxn: p)
# Res_2: Balance of energy  (test fxn: theta)

# boundaries = [(1,xBot),(2,xTop),(3,yBot),(4,yTop),(5,zBot),(6,zTop)]

# Time step constant within domain
dk = Constant(domain, PETSc.ScalarType(dt)) 

# Create a constant for the value of the pressure
press_cons = Constant(domain,PETSc.ScalarType(pressRamp(0)))

# Cofactor of F
Fcof = J*inv(F.T)

# Configuration-dependent traction
traction = - press_cons * dot(Fcof,n)

# The weak form for the equilibrium equation
Res_0 = inner(Piola , grad(u_test) )*dx  - dot(traction, u_test)*ds(6) 

# The weak form for the pressure
fac_p =  ( ln(J) - 3*alpha*(theta-theta0) )/J
#
Res_1 = dot( (p/Kbulk + fac_p), p_test)*dx
      

# The weak form for heat equation
Res_2 = dot( c_v*(theta - theta_old), theta_test)*dx \
        -  (1/2)*theta * inner(M, (C - C_old)) * theta_test*dx \
        -  dk* dot(Qmat , grad(theta_test) )*dx   
        
# Total weak form
Res = Res_0 + Res_1 + Res_2 

# Automatic differentiation tangent:
a = derivative(Res, w, dw)

# %% ----
# results file name
results_name = "3D_thermoelas_solar_sail"

# Function space for projection of results
U1 = element("DG", domain.basix_cell(), 1, shape=(3,)) # For displacement
P0 = element("DG", domain.basix_cell(), 1)             # For  pressure and temperature  
T1 = element("DG", domain.basix_cell(), 1, shape=(3,3))# For stress tensor

V1 = fem.functionspace(domain, P0) # Scalar function space
V2 = fem.functionspace(domain, U1) # Vector function space
V3 = fem.functionspace(domain, T1) # Tensor function space

# fields to write to output file
u_vis = Function(V2)
u_vis.name = "disp"

p_vis = Function(V1)
p_vis.name = "p"

theta_vis = Function(V1)
theta_vis.name = "theta"

J_vis = Function(V1)
J_vis.name = "J"
J_expr = Expression(J,V1.element.interpolation_points())

lambdaBar_vis = Function(V1)
lambdaBar_vis.name = "lambdaBar"
lambdaBar_expr = Expression(lambdaBar,V1.element.interpolation_points())

P11 = Function(V1)
P11.name = "P11"
P11_expr = Expression(Piola[0,0],V1.element.interpolation_points())
#
P22 = Function(V1)
P22.name = "P22"
P22_expr = Expression(Piola[1,1],V1.element.interpolation_points())
#
P33 = Function(V1)
P33.name = "P33"
P33_expr = Expression(Piola[2,2],V1.element.interpolation_points())

# Mises stress
T   = Piola*F.T/J
T0   = T - (1/3)*tr(T)*Identity(3)
Mises = sqrt((3/2)*inner(T0, T0))
Mises_vis= Function(V1,name="Mises")
Mises_expr = Expression(Mises,V1.element.interpolation_points())

# set up the output VTX files.
file_results = VTXWriter(
    MPI.COMM_WORLD,
    "results/" + results_name + ".bp",
    [  # put the functions here you wish to write to output
        u_vis, p_vis, theta_vis, J_vis, P11, P22, P33, 
        lambdaBar_vis,Mises_vis,
    ],
    engine="BP4",
)

def writeResults(t):
       # Output field interpolation 
       u_vis.interpolate(w.sub(0))
       p_vis.interpolate(w.sub(1))
       theta_vis.interpolate(w.sub(2))
       J_vis.interpolate(J_expr)
       P11.interpolate(P11_expr)
       P22.interpolate(P22_expr)
       P33.interpolate(P33_expr)
       lambdaBar_vis.interpolate(lambdaBar_expr)
       Mises_vis.interpolate(Mises_expr)

       # Write output fields
       file_results.write(t) 
        

# %% ----
# Identify point A
pointForDisp_A = np.array([70, 60,0.0]) 

bb_tree = dolfinx.geometry.bb_tree(domain,domain.topology.dim)
cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, pointForDisp_A)
colliding_cells_A = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, pointForDisp_A).array

# Identify point B
pointForDisp_B = np.array([L0, 0.0,0.0])

bb_tree = dolfinx.geometry.bb_tree(domain,domain.topology.dim)
cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, pointForDisp_B)
colliding_cells_B = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, pointForDisp_B).array


# %% ----
# Give the step a descriptive name
step = "Press/temp loading"

# %% ----
# Zero displacement boundary conditions for the two fixed points
#
V0, submap = ME.sub(0).collapse()
#
fixed1_u_dofs = fem.locate_dofs_geometrical((ME.sub(0), V0), pin1)
fixed1_disp   = Function(V0)
fixed1_disp.interpolate(lambda x: np.stack(( np.zeros(x.shape[1]), np.zeros(x.shape[1]),  np.zeros(x.shape[1]) ) ) )
#
fixed2_u_dofs = fem.locate_dofs_geometrical((ME.sub(0), V0), pin2)
fixed2_disp   = Function(V0)
fixed2_disp.interpolate(lambda x: np.stack(( np.zeros(x.shape[1]), np.zeros(x.shape[1]), np.zeros(x.shape[1]) ) ) )
#
bcs_1 = dirichletbc(fixed1_disp, fixed1_u_dofs, ME.sub(0))        # u fix -  left corner node pin1
#
bcs_2 = dirichletbc(fixed2_disp, fixed2_u_dofs, ME.sub(0))        # u fix -  right corner node pin2



# Recall the sub-domains names and numbers
# boundaries = [(1,xBot),(2,xTop),(3,yBot),(4,yTop),(5,zBot),(6,zTop)]

# Find the specific DOFs which will be constrained.
xBot_u1_dofs = fem.locate_dofs_topological(ME.sub(0).sub(0), facet_tags.dim, facet_tags.find(1))
#xTop_u1_dofs = fem.locate_dofs_topological(ME.sub(0).sub(0), facet_tags.dim, facet_tags.find(2))
yBot_u2_dofs = fem.locate_dofs_topological(ME.sub(0).sub(1), facet_tags.dim, facet_tags.find(3))
#yTop_u2_dofs = fem.locate_dofs_topological(ME.sub(0).sub(1), facet_tags.dim, facet_tags.find(4))
#zBot_u3_dofs = fem.locate_dofs_topological(ME.sub(0).sub(2), facet_tags.dim, facet_tags.find(5))
#zTop_u3_dofs = fem.locate_dofs_topological(ME.sub(0).sub(2), facet_tags.dim, facet_tags.find(6))

#xBot_theta_dofs = fem.locate_dofs_topological(ME.sub(2), facet_tags.dim, facet_tags.find(1))
xTop_theta_dofs = fem.locate_dofs_topological(ME.sub(2), facet_tags.dim, facet_tags.find(2))
#yBot_theta_dofs = fem.locate_dofs_topological(ME.sub(2), facet_tags.dim, facet_tags.find(3))
yTop_theta_dofs = fem.locate_dofs_topological(ME.sub(2), facet_tags.dim, facet_tags.find(4))
zBot_theta_dofs = fem.locate_dofs_topological(ME.sub(2), facet_tags.dim, facet_tags.find(5))
zTop_theta_dofs = fem.locate_dofs_topological(ME.sub(2), facet_tags.dim, facet_tags.find(6))

# Constant for applied temperature
theta_cons = Constant(domain,PETSc.ScalarType(thetaRamp(0)))

# Dirichlet BCs for displacement
bcs_3 = dirichletbc(0.0, xBot_u1_dofs, ME.sub(0).sub(0))  # u1 fix - xBot
bcs_4 = dirichletbc(0.0, yBot_u2_dofs, ME.sub(0).sub(1))  # u2 fix - yBot
#
# Dirichlet BCs for temperature
bcs_5 = dirichletbc(theta_cons, zBot_theta_dofs, ME.sub(2))  # Theta prescribe - zBot
bcs_6 = dirichletbc(theta_cons, zTop_theta_dofs, ME.sub(2))  # Theta prescribe - zTop
bcs_7 = dirichletbc(theta_cons, xTop_theta_dofs, ME.sub(2))  # Theta prescribe - xTop
bcs_8 = dirichletbc(theta_cons, yTop_theta_dofs, ME.sub(2))  # Theta prescribe - yTop

# Complete set of Dirichlet bcs
bcs = [bcs_1, bcs_2, bcs_3, bcs_4, bcs_5, bcs_6, bcs_7, bcs_8]


# %% ----
# Set up nonlinear problem
problem = NonlinearProblem(Res, w, bcs, a)

# The global newton solver and params
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-8
solver.atol = 1e-8
solver.max_it = 50
solver.report = True

#  The Krylov solver parameters.
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly" # "preonly" works equally well
opts[f"{option_prefix}pc_type"] = "lu" # do not use 'gamg' pre-conditioner
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
opts[f"{option_prefix}ksp_max_it"] = 30
ksp.setFromOptions()

# %% ----
# Arrays for storing output history
totSteps = 100000
timeHist0 = np.zeros(shape=[totSteps])
timeHist1 = np.zeros(shape=[totSteps]) 
timeHist2 = np.zeros(shape=[totSteps]) 
# timeHist3 = np.zeros(shape=[totSteps]) 
# timeHist4 = np.zeros(shape=[totSteps]) 
#
# timeHist3[0] = theta0 # Initialize the temperature
# timeHist4[0] = theta0 # Initialize the temperature

# Initialize a counter for reporting data
ii=0

# Write initial state to file
writeResults(t=0.0) 

# %% ----
# Print message for simulation start
print("------------------------------------")
print("Simulation Start")
print("------------------------------------")
# Store start time 
startTime = datetime.now()

# Time-stepping solution procedure loop
while (round(t + dt, 9) <= Ttot):
     
    # increment time
    t += dt 
    # increment counter
    ii += 1
    
    # update time variables in time-dependent BCs 
    theta_cons.value = float(thetaRamp(t))
    press_cons.value = float(pressRamp(t))
    
    # Solve the problem
    try:
        (iter, converged) = solver.solve(w)
    except: # Break the loop if solver fails
        print("Ended Early")
        break
    
    # Collect results from MPI ghost processes
    w.x.scatter_forward()
    
    # Write output to file
    writeResults(t)
    
    # Update DOFs for next step
    w_old.x.array[:] = w.x.array
   
    # Store  output history at a particular point  at current time
    #
    timeHist0[ii] = t                                                            # time
    #
    timeHist1[ii] = w.sub(0).sub(2).eval([70,60,0.0],colliding_cells_A[0])[0]     # u_3 displacement location A
    #
    timeHist2[ii] = w.sub(0).sub(2).eval([L0,0.0,0.0],colliding_cells_B[0])[0]    # u_3 displacement location B

    # Print progress of calculation
    if ii%1 == 0:      
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Step: {} | Increment: {}, Iterations: {}".\
              format(step, ii, iter))
        print("      Simulation Time: {} s  of  {} s".\
              format(round(t,4), Ttot))
        print()  
    

# ---- histories (added)
import numpy as _np
_h = {k: v for k, v in sorted(globals().items()) if k.startswith("timeHist") and hasattr(v, "shape")}
if MPI.COMM_WORLD.rank == 0:
    _n = min(min(len(v) for v in _h.values()), ii + 1)   # ii: last increment stored (runs that end early)
    _np.savetxt("hist/TE06_solar_sail.csv", _np.column_stack([v[:_n] for v in _h.values()]), delimiter=",", header=",".join(_h), comments="")


# %% ----
# Set up font size, initialize colors array
font = {'size'   : 14}
plt.rc('font', **font)
#
prop_cycle = plt.rcParams['axes.prop_cycle']
colors     = prop_cycle.by_key()['color']

# Only plot as far as we have time history data
ind = np.argmax(timeHist0[:])
   
# Create figure for surface temperature versus time curve.
#
fig = plt.figure() 
ax=fig.gca()  
#---------------------------------------------------------------------------------------------
plt.plot( (10E-3*timeHist0[0:ind]/Ttot)*1000, timeHist1[0:ind], c='b', linewidth=2.0, label='Location A')
plt.plot( (10E-3*timeHist0[0:ind]/Ttot)*1000, timeHist2[0:ind], c='r', linewidth=2.0, label='Location B')
#---------------------------------------------------------------------------------------------
plt.xlim(0,10)
plt.ylim(-20,0)
plt.grid(linestyle="--", linewidth=0.5, color='b')
ax.set_xlabel(r"Pressure, Pa",size=14)
ax.set_ylabel(r"$u_3$, mm",size=14)
ax.set_title(r"Membrane displacement versus pressure curve", size=14, weight='normal')
#
from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
#plt.legend()
import matplotlib.ticker as ticker
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))

plt.legend()
# Save figure
fig = plt.gcf()
fig.set_size_inches(7,5)
plt.tight_layout()
plt.savefig("results/3D_solar_sail_plotting.png", dpi=600) 

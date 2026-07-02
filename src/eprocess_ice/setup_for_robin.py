import numpy as np
import time

from dolfinx import fem
from ufl import ds, dx, grad, inner
import ufl

from mpi4py import MPI

from petsc4py import PETSc

from dolfinx import cpp as _cpp
from dolfinx import plot
from dolfinx.fem import (Constant, Function, FunctionSpace, dirichletbc,
                         extract_function_spaces, form, Expression,
                         locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType, GhostMode, create_rectangle, locate_entities,
                          locate_entities_boundary, meshtags)

from dolfinx import geometry
from ufl import div, dx, grad, inner, Measure

from mpi4py import MPI
from petsc4py import PETSc
import dolfinx.fem.petsc


# AUXILIARY FUNCTIONS FOR FORWARD MAP

# Based on the following code:
# F. Seizilles, Bayesian Approach to Inverse Robin Problems, 
# https://github.com/seizillesf/RobinBC, 2023.

# Create mesh and facet tags for relevant boundaries
def create_mesh_and_tags(size_msh,domain_length=5):
    """
    Creates the mesh and facet tags for the PDE problem.
    
    Keyword arguments:
    size_msh -- int, size of mesh considered
    domain_length -- float, ratio of the rectangular considered (x in [0,1], y in [0, 1/domain_length])
    
    Returns:
    msh -- mesh object
    facet_tag -- facets based on specified boundary conditions
    """
    
    size_msh_x = size_msh*domain_length
    size_msh_y = size_msh

    msh = create_rectangle(MPI.COMM_WORLD,
                           [np.array([0, 0]), np.array([1, 1/domain_length])],
                           [size_msh_x, size_msh_y],
                           CellType.triangle)


    #Create the subdomains and space for Robin boundary condition in this problem
    tol = 1E-14 # tolerance (we cannot use strict equalities)
    boundaries = [(1, lambda x: abs(x[1])<= tol),   # Robin BC, at the bottom
                  (2, lambda x: abs(x[1]-1/domain_length)<= tol), # Neumann BC, at  the surface
                  (3, lambda x: abs(x[0])<= tol),   # Dirichlet BC, on the left
                  (4, lambda x: abs(x[0]-1)<= tol)]  # Dirichlet BC, on the right


    # Loop through all the boundary conditions and create MeshTags identifying the facets for each boundary condition
    facet_indices, facet_markers = [], []
    fdim = msh.topology.dim - 1

    for (marker, locator) in boundaries:
        facets = locate_entities(msh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = meshtags(msh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
    
    return msh, facet_tag


def define_weak_form_stokes(theta, msh, facet_tag, W, V, Q):
    """
    Creates the bilinear and linear forms for the weak form associated to the Robin Laplace problem
    
    Keyword arguments:
    theta -- array of size K, coefficients in the expansion
    msh -- mesh on which we solve the problem
    facet_tag -- facet tags, marking the boundaries
    W, V, Q -- mixed function space / function space for velocity / function space for pressure
    
    Returns
    a -- bilinear form in the weak formulation
    L -- linear form in the weak formulation
    bcs -- Dirichlet boundary conditions for the problem
    """
    
    ds = Measure("ds", domain=msh, subdomain_data=facet_tag)
    fdim = msh.topology.dim - 1

    
    # Define the class of boundary conditions 

    class BoundaryCondition():
        def __init__(self, type, marker, values):
            self._type = type
            if type == "Dirichlet":
                u_D = Function(V)
                u_D.interpolate(values)
                facets = facet_tag.find(marker)
                dofs = locate_dofs_topological(V, fdim, facets)
                self._bc = dirichletbc(u_D, dofs)

            elif type == "Neumann":
                self._bc = inner(values, v) * ds(marker)

            elif type == "Robin":
              #self._bc = values[0] * inner(u-values[1], v)* ds(marker)
              # slight modification: returns 2 integrals, one for the bilinear form a and one for the linear form L
                self._bc = values[0] * inner(u,v)* ds(marker), values[0] * inner(values[1], v)* ds(marker)
            else:
                raise TypeError("Unknown boundary condition: {0:s}".format(type))

        @property
        def bc(self):
            return self._bc

        @property
        def type(self):
            return self._type


    version = "sum"


    # We now define the bilinear and linear forms corresponding to the weak
    # mixed formulation of the Stokes equations in a blocked structure:

    # Define variational problem: Trial and test functions
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    # Define the source terms (based on tunable parameters at the top)
    f = Constant(msh, (PETSc.ScalarType(rho*gx), PETSc.ScalarType(rho*gy)))


    # Define the bilinear form
    bilinear = inner(grad(u), grad(v)) * dx - inner(p, div(v)) * dx + inner(div(u), q) * dx

    # Define the linear form
    L = inner(f, v) * dx + inner(Constant(msh, PETSc.ScalarType(0)), q) * dx


    #-----------------SET THE BOUNDARY CONDITIONS FOR THE PROBLEM----------------------------------------------

    # Set the values for Neumann BC at the surface
    tau_fct = lambda x: (10*(np.sin(12*np.pi*x[0])+1), 0*x[1])
    tau = Function(V)
    tau.interpolate(tau_fct)
    values_boundary_neumann = tau

    # Set the values for Robin BC at the bottom
    beta = build_beta(theta)
    r = Function(Q)
    r.interpolate(beta)
    s = Constant(msh, (PETSc.ScalarType(0), PETSc.ScalarType(0)))
    values_boundary_robin = (r,s)


    # Gather the Boundary conditions
    boundary_conditions = [BoundaryCondition("Robin", 1, values_boundary_robin),
                        BoundaryCondition("Neumann", 2, values_boundary_neumann)]

    bcs = []
    for condition in boundary_conditions:
        if condition.type == "Dirichlet":
            bcs.append(condition.bc)

        elif condition.type == "Neumann":
            linear_term = condition.bc
            L+= linear_term

        elif condition.type == "Robin":

            bilinear_term, linear_term = condition.bc

            if version == "sum":
                a = bilinear + bilinear_term

            else:
                a[0].append(bilinear_term) # add the modification to bilinear form


            L+= linear_term   # add the modification to linear form

        else: 
            print("Unhandled condition type")
    return a, L, bcs
    

def solve_stokes(theta, msh, facet_tag):
    """
    Maps the theta coefficients to the solution of the Stokes PDE problem

    Keyword arguments:
    theta -- array of size K, coefficients in the expansion of the basal drag function beta
    
    Returns:
    uh -- solution of the PDE problem equations 
    """
    
    #------------------------------------PREPARE THE SOLVE------------------------------------------------
        
    # We define the finite elements function space (Taylor Woods method)
    P2 = ufl.VectorElement("Lagrange", msh.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
    mixed = ufl.MixedElement([P2, P1])

    V, Q = FunctionSpace(msh, P2), FunctionSpace(msh, P1)
    W = FunctionSpace(msh, mixed) # Defined Mixed Function space - needed for solving divergence at same time
    

    
    #-----------------GET THE WEAK FORM----------------------------------------------
    a,L, bcs = define_weak_form_stokes(theta, msh,facet_tag, W, V, Q)
     
    
    #-----------------------------ASSEMBLE AND SOLVE-----------------------------------------------

    # Assemble LHS matrix and RHS vector
    a,L = form(a),form(L)

    A = fem.petsc.assemble_matrix(a, bcs=bcs)
    A.assemble()
    b = fem.petsc.assemble_vector(L)

    fem.petsc.apply_lifting(b, [a], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set Dirichlet boundary condition values in the RHS
    fem.petsc.set_bc(b, bcs)

    # Create and configure solver
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("superlu_dist")

    # Compute the solution
    U = Function(W)
    
    start_time = time.time()
    ksp.solve(b, U.vector)
    solve_time = time.time()-start_time

    # Split the mixed solution and collapse
    uh = U.sub(0).collapse()
    ph = U.sub(1).collapse()
    
    return uh


def evaluate(solution, covariates, msh, domain_length=5):
    """
    Evaluates the solution of the PDE at specified covariate points
    
    Keyword arguments:
    solution -- solution of the PDE problem equations, solved by FEM
    covariates -- list of coordinates on which the evaluation is performed. Does not have to match mesh points
    msh -- mesh
    domain_length -- float
    
    Returns:
    uv -- list, evaluations of the solution at covariates points
    """
    
    # bb_tree = geometry.BoundingBoxTree(msh, msh.topology.dim)
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)

    points = [[c, 1/domain_length, 0] for c in covariates] # from surface covariates to points

    # Find cells which bounding-box collide with the the point
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(msh, cell_candidates, points)    

    #cell= colliding_cell.links(0)
    uv=solution.eval(points, colliding_cells.array)
    
    return uv


# FORWARD MAP - Putting together auxiliary functions

def forward_map(problem, theta, size_msh, covariates):
    
    msh, facet_tag = create_mesh_and_tags(size_msh)
        
    if problem=='Stokes':
        uh = solve_stokes(theta, msh, facet_tag)
        
    else:
        raise Exception("Unknown problem")
        
    u_on_covariates = evaluate(uh, covariates, msh)
    
    return u_on_covariates

def build_beta(coeffs):
    """
    Auxiliary function to build a function beta based on the theta coefficients in the arguments.
    We exponentiate because beta must be a positive function.

    Keyword arguments:
    coeffs -- float list of size K, the coefficients weigthing the truncated Fourier expansion functions

    Returns:
    beta -- one-dimensional function representing the basal drag factor
    """ 
    K = len(coeffs)
    functions = [1]+[np.cos,np.sin]*(K//2) # 
    wavenumbers = [0]+[1+k//2 for k in range(K-1)] # wavenumbers inside the (co)sinus function[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,...]
    beta = lambda x: np.exp((coeffs[0] + sum([coeffs[k]*functions[k](wavenumbers[k]*2*np.pi*x[0]) for k in range(1,K)])))
    
    return beta



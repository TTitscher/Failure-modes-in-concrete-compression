from mpi4py import MPI
import numpy as np
import ufl
import dolfinx
import basix
from petsc4py import PETSc

import pyvista as pv
import pyvistaqt as pvqt

import helper as h

import sys


def test_assembly_into_quadrature_function():

    try:
        N = int(sys.argv[1])
    except IndexError:
        N = 10

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = dolfinx.fem.VectorFunctionSpace(mesh, ("P", 1))
    num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    print(f"{num_dofs_global = }")


    def left(x):
        return np.isclose(x[0], 0)

    def right(x):
        return np.isclose(x[0], 1)
    
    def origin(x):
        a = np.isclose(x[0], 0)
        b = np.isclose(x[1], 0)
        return np.logical_and(a, b)

    dim = mesh.topology.dim - 1
    b_facets_l = dolfinx.mesh.locate_entities_boundary(mesh, dim, left)
    b_facets_r = dolfinx.mesh.locate_entities_boundary(mesh, dim, right)
    b_facets_o = dolfinx.mesh.locate_entities_boundary(mesh, dim-1, origin)
    
    b_dofs_l = dolfinx.fem.locate_dofs_topological(V.sub(0), dim, b_facets_l)
    b_dofs_r = dolfinx.fem.locate_dofs_topological(V.sub(0), dim, b_facets_r)
    b_dofs_o = dolfinx.fem.locate_dofs_topological(V.sub(1), dim, b_facets_o)
    # print(b_dofs_o)

    u_bc = dolfinx.fem.Constant(mesh, 0.)

    bcs = []
    bcs.append(dolfinx.fem.dirichletbc(PETSc.ScalarType(0), b_dofs_l, V.sub(0)))
    bcs.append(dolfinx.fem.dirichletbc(u_bc, b_dofs_r, V.sub(0)))
    bcs.append(dolfinx.fem.dirichletbc(PETSc.ScalarType(0), b_dofs_o, V.sub(1)))

    u_bc.value = 0.1 # can be used to modify that later, e.g. for time dependency


    quadrature_degree = 1
    quadrature_points, wts = basix.make_quadrature(
        basix.CellType.triangle, quadrature_degree
    )

    QV = ufl.VectorElement(
        "Quadrature", ufl.triangle, quadrature_degree, quad_scheme="default", dim=3
    )
    QT = ufl.TensorElement(
        "Quadrature",
        ufl.triangle,
        quadrature_degree,
        quad_scheme="default",
        shape=(3, 3),
    )
    VQV = dolfinx.fem.FunctionSpace(mesh, QV)
    VQT = dolfinx.fem.FunctionSpace(mesh, QT)

    u = dolfinx.fem.Function(V)
    u.x.array[:] = np.random.random(len(u.x.array[:]))
    # u.interpolate(lambda x: (x[0], -2*x[1]))

    def eps(u):
        e = ufl.sym(ufl.grad(u))
        return ufl.as_vector((e[0, 0], e[1, 1], 2 * e[0, 1]))

    strain_expr = dolfinx.fem.Expression(eps(u), quadrature_points)

    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    strain_eval = strain_expr.eval(cells)

    # Assemble into Function
    q_strain = dolfinx.fem.Function(VQV)
    with q_strain.vector.localForm() as q_strain_local:
        q_strain_local.setBlockSize(q_strain.function_space.dofmap.bs)
        q_strain_local.setValuesBlocked(
            VQV.dofmap.list.array, strain_eval, addv=PETSc.InsertMode.INSERT
        )
    q_strain2 = dolfinx.fem.Function(VQV)
    q_strain2.x.array[:] = strain_eval.flatten()
    q_strain2.x.scatter_forward()

    # assert np.linalg.norm(q_strain.x.array[:] - q_strain2.x.array[:]) < 1.0e-10

    q_sigma = dolfinx.fem.Function(VQV)
    q_dsigma = dolfinx.fem.Function(VQT)

    E, nu = 20000, 0.2
    C11 = E / (1.0 - nu * nu)
    C12 = C11 * nu
    C33 = C11 * 0.5 * (1.0 - nu)
    C = np.array([[C11, C12, 0.0], [C12, C11, 0.0], [0.0, 0.0, C33]])

    n_gauss = len(q_sigma.x.array[:]) // 3
    print(f"{n_gauss = }")
    C_values = np.tile(C.flatten(), n_gauss)
    q_dsigma.x.array[:] = C_values
    q_dsigma.x.scatter_forward()

    metadata = {"quadrature_degree": quadrature_degree, "quadrature_scheme": "default"}
    dxm = ufl.dx(metadata=metadata)

    f = ufl.as_vector((0, 0))
    # dx = ufl.dx()
    # print(q_strain.vector[:])
    u_, du = ufl.TestFunction(V), ufl.TrialFunction(V)
    R = dolfinx.fem.form((-ufl.inner(eps(u_), q_sigma) + ufl.inner(f, u_)) * dxm) 
    dR = dolfinx.fem.form(ufl.inner(eps(du), ufl.dot(q_dsigma, eps(u_))) * dxm)

    # problem = dolfinx.fem.petsc.NonlinearProblem(R)
    # solver = dolfinx.nls.petsc.NewtonSolver

    A = dolfinx.fem.petsc.assemble_matrix(dR, bcs=bcs)
    A.assemble()

    b = dolfinx.fem.petsc.assemble_vector(R)
    dolfinx.fem.apply_lifting(b, [dR], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)
    
    null_space = h.nullspace_2d(V)
    A.setNearNullSpace(null_space)

    opts = PETSc.Options()
    opts["ksp_type"] = "gmres"
    opts["ksp_rtol"] = 1.0e-10
    opts["pc_type"] = "gamg"

    # Use Chebyshev smoothing for multigrid
    opts["mg_levels_ksp_type"] = "chebyshev"
    opts["mg_levels_pc_type"] = "jacobi"

    # Improve estimate of eigenvalues for Chebyshev smoothing
    opts["mg_levels_esteig_ksp_type"] = "gmres"
    opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20

    solver = PETSc.KSP().create(mesh.comm)
    solver.setFromOptions()
    solver.setOperators(A)

    solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
    solver.solve(b, u.vector)
    # solver.view()

    with dolfinx.io.XDMFFile(mesh.comm, "displacements.xdmf", "w") as f:
        f.write_mesh(mesh)
        f.write_function(u)
    #
    p = pv.Plotter()
    topology, cells, geometry = dolfinx.plot.create_vtk_mesh(u.function_space)
    grid = pv.UnstructuredGrid(topology, cells, geometry)
    actor = p.add_mesh(grid, style="wireframe", color="w")
    values = np.zeros((geometry.shape[0], 3))
    values[:, :len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
    grid["u"] = values
    grid.set_active_vectors("u")
    warped = grid.warp_by_vector("u", factor=10)

    actor = p.add_mesh(warped, style="surface")
    p.show_axes()
    p.show()

    # grid.point_data["u"] = u.x.array.real

test_assembly_into_quadrature_function()

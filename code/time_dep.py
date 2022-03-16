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


def eval_function_at_points(f, points):
    """
    From https://jorgensd.github.io/dolfinx-tutorial/chapter1/membrane_code.html
    """
    mesh = f.function_space.mesh
    tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cell_candidates = dolfinx.geometry.compute_collisions(tree, points)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        mesh, cell_candidates, points
    )

    points_on_proc = []
    cells = []
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    return f.eval(points_on_proc, cells), points_on_proc


class UnitSquareExperiment:
    def __init__(self, N):
        self.mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
        self.u_bc = dolfinx.fem.Constant(self.mesh, 0.0)

    def bcs(self, V):
        def left(x):
            return np.isclose(x[0], 0)

        def right(x):
            return np.isclose(x[0], 1)

        def origin(x):
            a = np.isclose(x[0], 0)
            b = np.isclose(x[1], 0)
            return np.logical_and(a, b)

        mesh = self.mesh
        dim = self.mesh.topology.dim - 1
        b_facets_l = dolfinx.mesh.locate_entities_boundary(mesh, dim, left)
        b_facets_r = dolfinx.mesh.locate_entities_boundary(mesh, dim, right)
        b_facets_o = dolfinx.mesh.locate_entities_boundary(mesh, dim - 1, origin)

        b_dofs_l = dolfinx.fem.locate_dofs_topological(V.sub(0), dim, b_facets_l)
        b_dofs_r = dolfinx.fem.locate_dofs_topological(V.sub(0), dim, b_facets_r)
        b_dofs_o = dolfinx.fem.locate_dofs_topological(V.sub(1), dim - 1, b_facets_o)
        # print(b_dofs_o)

        return [
            dolfinx.fem.dirichletbc(PETSc.ScalarType(0), b_dofs_l, V.sub(0)),
            dolfinx.fem.dirichletbc(self.u_bc, b_dofs_r, V.sub(0)),
            dolfinx.fem.dirichletbc(PETSc.ScalarType(0), b_dofs_o, V.sub(1)),
        ]

    def set_bcs(self, value):
        self.u_bc.value = value


class HookesLaw:
    def __init__(self, E, nu):
        self.E, self.nu = E, nu
        C11 = E / (1.0 - nu * nu)
        C12 = C11 * nu
        C33 = C11 * 0.5 * (1.0 - nu)
        self.C = np.array([[C11, C12, 0.0], [C12, C11, 0.0], [0.0, 0.0, C33]])

    def evaluate(self, strains):
        n_gauss = len(strains)
        return (strains @ self.C).flat, np.tile(self.C.flatten(), n_gauss)


class MechanicsProblem:
    def __init__(self, experiment, material, deg=1):
        self.experiment, self.material, self.deg = experiment, material, deg
        mesh = experiment.mesh

        # define function spaces
        q_deg = deg

        self.V = dolfinx.fem.VectorFunctionSpace(mesh, ("P", deg))
        QV = ufl.VectorElement(
            "Quadrature", ufl.triangle, q_deg, quad_scheme="default", dim=3
        )
        QT = ufl.TensorElement(
            "Quadrature",
            ufl.triangle,
            q_deg,
            quad_scheme="default",
            shape=(3, 3),
        )
        VQV = dolfinx.fem.FunctionSpace(mesh, QV)
        VQT = dolfinx.fem.FunctionSpace(mesh, QT)

        # define functions
        u_, du = ufl.TestFunction(self.V), ufl.TrialFunction(self.V)
        self.u = dolfinx.fem.Function(self.V)
        self.q_sigma = dolfinx.fem.Function(VQV)
        self.q_dsigma = dolfinx.fem.Function(VQT)

        # define form
        self.metadata = {"quadrature_degree": q_deg, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)

        # f = ufl.as_vector((0, 0))

        eps = self.eps
        R = -ufl.inner(eps(u_), self.q_sigma) * self.dxm
        dR = ufl.inner(eps(du), ufl.dot(self.q_dsigma, eps(u_))) * self.dxm

        self.R, self.dR = dolfinx.fem.form(R), dolfinx.fem.form(dR)

        # prepare strain evaluation
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        assert map_c.num_ghosts == 0  # no ghost cells, right?
        self.cells = np.arange(0, num_cells, dtype=np.int32)

        q_points, wts = basix.make_quadrature(basix.CellType.triangle, q_deg)
        self.strain_expr = dolfinx.fem.Expression(self.eps(self.u), q_points)
        self.evaluate_constitutive_law()

        # bcs and stuff
        self.nullspace = h.nullspace_2d(self.V)
        self.bcs = self.experiment.bcs(self.V)


    def eps(self, u):
        e = ufl.sym(ufl.grad(u))
        return ufl.as_vector((e[0, 0], e[1, 1], 2 * e[0, 1]))

    def evaluate_constitutive_law(self):
        strain = self.strain_expr.eval(self.cells)
        self.q_sigma.x.array[:], self.q_dsigma.x.array[:] = self.material.evaluate(
            strain
        )

    def J(self, snes, x: PETSc.Vec, A: PETSc.Mat, P):
        """Assemble the Jacobian matrix.  """
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, self.dR, bcs=self.bcs)
        A.assemble()
        A.setNearNullSpace(self.nullspace)

    def F(self, snes, x: PETSc.Vec, b: PETSc.Vec):
        """Assemble the residual F into the vector b.  """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        self.evaluate_constitutive_law()

        # Reset the residual vector
        with b.localForm() as b_local:
            b_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector(b, self.R)

        # Apply boundary condition
        dolfinx.fem.apply_lifting(b, [self.dR], bcs=[self.bcs], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b, self.bcs, x, -1.0)

def create_solver(problem):
    b = dolfinx.la.create_petsc_vector(problem.V.dofmap.index_map, problem.V.dofmap.index_map_bs)
    J = dolfinx.fem.petsc.create_matrix(problem.dR)
    # exit()
    # pass
    snes = PETSc.SNES().create()
    opts = PETSc.Options()

    opts["solve_ksp_type"] = "cg"
    opts["ksp_rtol"] = 1.0e-10
    opts["pc_type"] = "gamg"

    # Use Chebyshev smoothing for multigrid
    opts["mg_levels_ksp_type"] = "chebyshev"
    opts["mg_levels_pc_type"] = "jacobi"

    # Improve estimate of eigenvalues for Chebyshev smoothing
    opts["mg_levels_esteig_ksp_type"] = "cg"
    opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20

    opts["snes_linesearch_type"] = "bt"
    snes.setFromOptions()

    snes.setFunction(problem.F, b)
    snes.setJacobian(problem.J, J)
    snes.setMonitor(
        lambda _, its, rnorm: print(f"Newton Iteration: {its}, rel. residual: {rnorm}")
    )
    snes.getKSP().setMonitor(
        lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}")
    )

    snes.setTolerances(rtol=1.0e-9, max_it=10)
    # snes.getKSP().setType("preonly")
    # snes.getKSP().setTolerances(rtol=1.0e-9)
    # snes.getKSP().getPC().setType("lu")
    return snes


experiment = UnitSquareExperiment(10)
mat = HookesLaw(20000, 0.2)
u_bc = 0.1
experiment.set_bcs(u_bc)
problem = MechanicsProblem(experiment, mat)

solver = create_solver(problem)
solver.solve(None, problem.u.vector)
u = problem.u

# check solution
points = np.array([[0.0, 0, 0.0], [0.5, 0.5, 0.0], [1.0, 1.0, 0.0]])
u_values, points = eval_function_at_points(u, points)

for x, u_fem in zip(points, u_values):
    u_correct = x[0] * u_bc, -mat.nu * x[1] * u_bc
    if np.linalg.norm(u_fem - u_correct) > 1.0e-10:
        print(u_correct, u_fem)

with dolfinx.io.XDMFFile(experiment.mesh.comm, "displacements.xdmf", "w") as f:
    f.write_mesh(experiment.mesh)
    f.write_function(u)
#
p = pv.Plotter()
topology, cells, geometry = dolfinx.plot.create_vtk_mesh(u.function_space)
grid = pv.UnstructuredGrid(topology, cells, geometry)
actor = p.add_mesh(grid, style="wireframe", color="w")
values = np.zeros((geometry.shape[0], 3))
values[:, : len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
grid["u"] = values
grid.set_active_vectors("u")
warped = grid.warp_by_vector("u", factor=10)

actor = p.add_mesh(warped, style="surface")
p.show_axes()
p.show()


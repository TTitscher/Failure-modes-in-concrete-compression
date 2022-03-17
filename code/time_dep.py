from mpi4py import MPI
import numpy as np
import ufl
import dolfinx as df
import basix
from petsc4py import PETSc

import pyvista as pv
import pyvistaqt as pvqt

import helper as h

import sys

class UnitSquareExperiment:
    def __init__(self, N):
        self.mesh = df.mesh.create_unit_square(
            MPI.COMM_WORLD, N, N, df.mesh.CellType.triangle
        )
        self.u_bc = df.fem.Constant(self.mesh, 0.0)

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
        b_facets_l = df.mesh.locate_entities_boundary(mesh, dim, left)
        b_facets_r = df.mesh.locate_entities_boundary(mesh, dim, right)
        b_facets_o = df.mesh.locate_entities_boundary(mesh, dim - 1, origin)

        b_dofs_l = df.fem.locate_dofs_topological(V.sub(0), dim, b_facets_l)
        b_dofs_r = df.fem.locate_dofs_topological(V.sub(0), dim, b_facets_r)
        b_dofs_o = df.fem.locate_dofs_topological(V.sub(1), dim - 1, b_facets_o)
        # print(b_dofs_o)

        self.bcs = [
            df.fem.dirichletbc(PETSc.ScalarType(0), b_dofs_l, V.sub(0)),
            df.fem.dirichletbc(self.u_bc, b_dofs_r, V.sub(0)),
            df.fem.dirichletbc(PETSc.ScalarType(0), b_dofs_o, V.sub(1)),
        ]

        self.load_dofs = b_dofs_r
        return self.bcs

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
        strains = strains.reshape((-1, 3))
        n_gauss = len(strains)
        return (strains @ self.C).flat, np.tile(self.C.flatten(), n_gauss)


class MechanicsProblem:
    def __init__(self, experiment, material, deg=1):
        self.experiment, self.material, self.deg = experiment, material, deg
        mesh = experiment.mesh

        # define function spaces
        q_deg = deg + 1

        self.V = df.fem.VectorFunctionSpace(mesh, ("P", deg))
        QV = ufl.VectorElement(
            "Quadrature", mesh.ufl_cell(), q_deg, quad_scheme="default", dim=3
        )
        QT = ufl.TensorElement(
            "Quadrature",
            mesh.ufl_cell(),
            q_deg,
            quad_scheme="default",
            shape=(3, 3),
        )
        VQV = df.fem.FunctionSpace(mesh, QV)
        VQT = df.fem.FunctionSpace(mesh, QT)

        # define functions
        u_, du = ufl.TestFunction(self.V), ufl.TrialFunction(self.V)
        self.u = df.fem.Function(self.V)
        self.q_sigma = df.fem.Function(VQV)
        self.q_dsigma = df.fem.Function(VQT)

        # define form
        self.metadata = {"quadrature_degree": q_deg, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)

        # f = ufl.as_vector((0, 0))

        eps = self.eps
        R = -ufl.inner(eps(u_), self.q_sigma) * self.dxm
        dR = ufl.inner(eps(du), ufl.dot(self.q_dsigma, eps(u_))) * self.dxm

        self.R, self.dR = df.fem.form(R), df.fem.form(dR)

        # prepare strain evaluation
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        self.cells = np.arange(0, num_cells, dtype=np.int32)

        basix_celltype = getattr(basix.CellType, mesh.topology.cell_type.name)
        q_points, wts = basix.make_quadrature(basix_celltype, q_deg)
        self.strain_expr = df.fem.Expression(self.eps(self.u), q_points)
        self.evaluate_constitutive_law()

        # bcs and stuff
        self.nullspace = h.nullspace_2d(self.V)
        self.bcs = self.experiment.bcs(self.V)

        self.residual = df.fem.petsc.create_vector(self.R) # stored

    def eps(self, u):
        e = ufl.sym(ufl.grad(u))
        return ufl.as_vector((e[0, 0], e[1, 1], 2 * e[0, 1]))

    def evaluate_constitutive_law(self):
        strain = self.strain_expr.eval(self.cells)
        self.q_sigma.x.array[:], self.q_dsigma.x.array[:] = self.material.evaluate(
            strain
        )

    def J(self, snes, x: PETSc.Vec, A: PETSc.Mat, P):
        """Assemble the Jacobian matrix."""
        A.zeroEntries()
        df.fem.petsc.assemble_matrix(A, self.dR, bcs=self.bcs)
        A.assemble()
        A.setNearNullSpace(self.nullspace)

    def F(self, snes, x: PETSc.Vec, b: PETSc.Vec):
        """Assemble the residual F into the vector b."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        self.evaluate_constitutive_law()

        # Reset the residual vector
        with b.localForm() as b_local:
            b_local.set(0.0)
        df.fem.petsc.assemble_vector(b, self.R)

        b.copy(self.residual)

        df.fem.apply_lifting(b, [self.dR], bcs=[self.bcs], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        df.fem.set_bc(b, self.bcs, x, -1.0)


try:
    N = int(sys.argv[1])
except IndexError:
    N = 10

experiment = UnitSquareExperiment(N)
mat = HookesLaw(20000, 0.2)
problem = MechanicsProblem(experiment, mat)

solver = h.create_solver(problem)


f = df.io.XDMFFile(experiment.mesh.comm, "displacements.xdmf", "w")
f.write_mesh(experiment.mesh)


class LoadSensor:
    def __init__(self, dofs, residual, name="load"):
        self.dofs, self.residual, name = dofs, residual, name
        self.comm = MPI.COMM_WORLD

    def measure(self):
        with self.residual.localForm() as lf:
            local_load = np.sum(lf[self.dofs])

        return self.comm.reduce(local_load, op=MPI.SUM, root=0)

class DisplacementSensor:
    def __init__(self, bc, name="disp"):
        self.bc, name = bc, name

    def measure(self):
        return float(self.bc.g.value) # .value itself is a 0-dim np.array


load_sensor = LoadSensor(experiment.load_dofs, problem.residual)
disp_sensor = DisplacementSensor(experiment.bcs[1])

for i, u_bc in enumerate(np.linspace(0, 0.1, 11)):
    print(f"load step {i} with {u_bc = }")
    experiment.set_bcs(u_bc)
    # solver.setInitialGuess(problem.u.vector)
    solver.solve(None, problem.u.vector)
    f.write_function(problem.u, u_bc)

    # problem.F(None, problem.u.vector, residual, apply_bc=False)
    load = load_sensor.measure()
    disp = disp_sensor.measure()

    if MPI.COMM_WORLD.rank == 0:
        print(f"{load = }")
        print(f"{disp = }")

u = problem.u


# check solution
points = np.array([[0.0, 0, 0.0], [0.5, 0.5, 0.0], [1.0, 1.0, 0.0]])
u_values, points = h.eval_function_at_points(u, points)

for x, u_fem in zip(points, u_values):
    u_correct = x[0] * u_bc, -mat.nu * x[1] * u_bc
    if np.linalg.norm(u_fem - u_correct) > 1.0e-10:
        print(u_correct, u_fem)

#


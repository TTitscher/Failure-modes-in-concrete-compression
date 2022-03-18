from mpi4py import MPI
import numpy as np
import ufl
import dolfinx as df
import basix
from petsc4py import PETSc
import helper as _h  # underscore indicates custom stuff


class MechanicsProblem:
    def __init__(self, experiment, material, deg=1, q_deg=None):
        self.experiment, self.material, self.deg = experiment, material, deg
        mesh = experiment.mesh

        # define function spaces
        q_deg = deg + 1 if q_deg is None else q_deg

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

        eps = material.eps
        R = ufl.inner(eps(u_), self.q_sigma) * self.dxm
        dR = ufl.inner(eps(du), ufl.dot(self.q_dsigma, eps(u_))) * self.dxm

        self.R, self.dR = df.fem.form(R), df.fem.form(dR)

        # prepare strain evaluation
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        self.cells = np.arange(0, num_cells, dtype=np.int32)

        basix_celltype = getattr(basix.CellType, mesh.topology.cell_type.name)
        self.q_points, wts = basix.make_quadrature(basix_celltype, q_deg)
        self.strain_expr = df.fem.Expression(eps(self.u), self.q_points)
        self.evaluate_constitutive_law()

        # bcs and stuff
        self.nullspace = _h.nullspace_2d(self.V)
        self.bcs = self.experiment.get_bcs(self.V)

        self.residual = df.fem.petsc.create_vector(self.R)  # stored
        self.solver = None

    def evaluate_constitutive_law(self):
        self.strain = self.strain_expr.eval(self.cells)
        self.q_sigma.x.array[:], self.q_dsigma.x.array[:] = (
            sigma,
            dsigma,
        ) = self.material.evaluate(self.strain)

    def update(self):
        self.strain = self.strain_expr.eval(self.cells)
        self.material.update(self.strain)

    def J(self, snes, x: PETSc.Vec, A: PETSc.Mat, P=None):
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

    def solve(self):
        if self.solver is None:
            self.solver = _h.create_solver(self)

        self.solver.solve(None, self.u.vector)
        return self.solver.its, self.solver.converged

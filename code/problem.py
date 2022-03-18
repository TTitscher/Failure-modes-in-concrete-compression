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
        q_deg = deg+1 if q_deg is None else q_deg

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
        self.strain_expr = df.fem.Expression(self.material.eps(self.u), self.q_points)
        self.strain = self.strain_expr.eval(self.cells)
        sigma, dsigma = self.material.evaluate(self.strain)
        #
        # VSigma = self.q_sigma.function_space
        # with self.q_sigma.vector.localForm() as sigma_local:
        #     sigma_local.setBlockSize(VSigma.dofmap.bs)
        #     sigma_local.setValuesBlocked(VSigma.dofmap.list.array, sigma, addv=PETSc.InsertMode.INSERT)
        #
        # VdSigma = self.q_dsigma.function_space
        # with self.q_dsigma.vector.localForm() as dsigma_local:
        #     dsigma_local.setBlockSize(VdSigma.dofmap.bs)
        #     dsigma_local.setValuesBlocked(VdSigma.dofmap.list.array, dsigma, addv=PETSc.InsertMode.INSERT)

        self.q_sigma.x.array[:] = sigma
        self.q_dsigma.x.array[:] = dsigma
        # self.q_sigma.x.scatter_forward()
        # self.q_dsigma.x.scatter_forward()

    def update(self):
        self.strain = self.strain_expr.eval(self.cells)
        self.material.update(self.strain)

    def J(self, x: PETSc.Vec, A: PETSc.Mat, P=None):
        """Assemble the Jacobian matrix."""
        A.zeroEntries()
        df.fem.petsc.assemble_matrix(A, self.dR, bcs=self.bcs)
        A.assemble()
        A.setNearNullSpace(self.nullspace)

    def form(self, x: PETSc.Vec):
        """This function is called before the residual or Jacobian is
        computed. This is usually used to update ghost values.
        Parameters
        ----------
        x
            The vector containing the latest solution
        """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, x: PETSc.Vec, b: PETSc.Vec):
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
        # self.solver = None
        if self.solver is None:
            self.a = self.dR
            self.L = self.R
            self.solver = df.nls.petsc.NewtonSolver(self.experiment.mesh.comm, self)
            self.solver.error_on_nonconvergence=False
            self.solver.max_it = 10
            # self.solver.rtol = 1.e-9
            # self.solver.atol = 1.e-9
            ksp = self.solver.krylov_solver
            # ksp.setType("gmres")
            # ksp.setTolerances(rtol=1.e-10)
            # ksp.getPC().setType("gamg")
            # ksp.setMonitor(
                # lambda _, its, rnorm: print(f"    krylov: {its}, |R|/|R0| = {rnorm:6.3e}")
        # )

        return self.solver.solve(self.u)

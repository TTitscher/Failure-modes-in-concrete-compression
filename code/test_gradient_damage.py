import sys

import dolfinx as df
import numpy as np

import helper as _h  # underscore indicates custom stuff
import experiments as _exp
import materials as _mat
from petsc4py import PETSc
import ufl
import basix

import matplotlib.pyplot as plt

from timestepping import TimeStepper

class MechanicsProblem:
    def __init__(self, experiment, mat, deg=1, q_deg=None):
        self.experiment, self.mat, self.deg = experiment, mat, deg
        mesh = experiment.mesh

        # define function spaces
        q_deg = 2*deg if q_deg is None else q_deg

        Ed = ufl.VectorElement("CG", mesh.ufl_cell(), degree=deg)
        Ee = ufl.FiniteElement("CG", mesh.ufl_cell(), degree=deg)
        self.V = df.fem.FunctionSpace(mesh, (Ed * Ee))
        
        self.Vd, self.Ve = self.V.sub(0), self.V.sub(1)
        self.u = df.fem.Function(self.V)

        q = "Quadrature"
        cell = mesh.ufl_cell()
        QF = ufl.FiniteElement(q, cell, q_deg, quad_scheme="default")
        QV = ufl.VectorElement(q, cell, q_deg, quad_scheme="default", dim=mat.qdim)
        QT = ufl.TensorElement(q, cell, q_deg, quad_scheme="default", shape=(mat.qdim, mat.qdim))

        VQF, VQV, VQT = [df.fem.FunctionSpace(mesh, Q) for Q in [QF, QV, QT]]
        self.q_sigma = df.fem.Function(VQV)
        self.q_e = df.fem.Function(VQF)
        self.q_eeq = df.fem.Function(VQF)

        self.q_dsigma_deps = df.fem.Function(VQT)
        self.q_dsigma_de = df.fem.Function(VQV)
        self.q_deeq_deps = df.fem.Function(VQV)

        # define functions
        dd, de = ufl.TrialFunctions(self.V)
        d_, e_ =  ufl.TestFunctions(self.V)

        self.d, self.e = ufl.split(self.u)


        # define form
        self.metadata = {"quadrature_degree": q_deg, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)

        # f = ufl.as_vector((0, 0))

        eps = mat.eps

        R = ufl.inner(eps(d_), self.q_sigma) * self.dxm
        R += e_ * (self.e - self.q_eeq) * self.dxm
        R += ufl.dot(ufl.grad(e_), mat.l**2 * ufl.grad(self.e)) * self.dxm

        dR = ufl.inner(eps(d_), self.q_dsigma_deps* eps(dd)) * self.dxm
        dR += ufl.inner(eps(d_), self.q_dsigma_de * de) * self.dxm
        dR += e_ * (de - ufl.dot(self.q_deeq_deps, eps(dd))) * self.dxm
        dR += ufl.dot(ufl.grad(e_), mat.l**2 * ufl.grad(de)) * self.dxm

        self.R, self.dR = df.fem.form(R), df.fem.form(dR)

        # prepare strain evaluation
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        self.cells = np.arange(0, num_cells, dtype=np.int32)

        basix_celltype = getattr(basix.CellType, mesh.topology.cell_type.name)
        self.q_points, wts = basix.make_quadrature(basix_celltype, q_deg)
        self.strain_expr = df.fem.Expression(eps(self.d), self.q_points)
        self.e_expr = df.fem.Expression(self.e, self.q_points)

        self.evaluate_constitutive_law()

        # bcs and stuff
        # self.nullspace = _h.nullspace_gdm(self.Vd, self.u)
        self.bcs = self.experiment.get_bcs(self.Vd)

        self.residual = df.fem.petsc.create_vector(self.R)  # stored
        self.solver = None

    def evaluate_constitutive_law(self):
        with df.common.Timer("compute strains"):
            strain = self.strain_expr.eval(self.cells)
            e = self.e_expr.eval(self.cells)

        with df.common.Timer("evaluate constitutive law"):
            self.mat.evaluate(strain, e)
        
        with df.common.Timer("assign q space"):
            self.q_sigma.x.array[:] = self.mat.sigma.flat
            self.q_dsigma_deps.x.array[:] = self.mat.dsigma_deps.flat
            self.q_dsigma_de.x.array[:] = self.mat.dsigma_de.flat
            self.q_eeq.x.array[:] = self.mat.eeq.flat
            self.q_deeq_deps.x.array[:] = self.mat.deeq.flat

    def update(self):
        # strain = self.strain_expr.eval(self.cells)
        e = self.e_expr.eval(self.cells)
        self.mat.update(e)

    def form(self, x: PETSc.Vec):
        """This function is called before the residual or Jacobian is
        computed. This is usually used to update ghost values.
        Parameters
        ----------
        x
            The vector containing the latest solution
        """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        self.evaluate_constitutive_law()

    def J(self, x: PETSc.Vec, A: PETSc.Mat, P=None):
        """Assemble the Jacobian matrix."""
        A.zeroEntries()
        df.fem.petsc.assemble_matrix(A, self.dR, bcs=self.bcs)
        A.assemble()
        # A.setNearNullSpace(self.nullspace)

    def F(self, x: PETSc.Vec, b: PETSc.Vec):
        """Assemble the residual F into the vector b."""

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
            self.a = self.dR
            self.L = self.R
            self.solver = df.nls.petsc.NewtonSolver(self.experiment.mesh.comm, self)

        return self.solver.solve(self.u)


experiment = _exp.BendingThreePoint()
mat = _mat.GradientDamage(l=200**0.5, nu=0.2, ft=2., k=10, alpha=0.99, beta=100.)
problem = MechanicsProblem(experiment, mat, deg=2, q_deg=4)

storage = _h.MeasurementSystem()
storage.add(_exp.LoadSensor(experiment.load_dofs, problem.residual, name="load"))
storage.add(_exp.DisplacementSensor(experiment.load_bc, name="disp"))

f = df.io.XDMFFile(experiment.mesh.comm, "gdm.xdmf", "w")
f.write_mesh(experiment.mesh)


# snes, solve_method = _h.create_solver(problem, iterative=False, monitor_newton=True, linesearch="bt")
snes, solve_method = _h.create_solver(problem, iterative=False, monitor_newton=True, linesearch="basic")
# solve_method = problem.solve

u_bc = 3.0

def solve(t, dt):
    experiment.set_bcs(-t*u_bc)
    return solve_method()

def pp(t):
    problem.update()
    f.write_function(problem.u.sub(0), t)
    storage.measure()

t = TimeStepper(solve, pp, problem.u)
t.adaptive(t_end=1, dt=0.1, show_bar=True)

# plt.plot(storage["disp"], storage["load"], "-kx")
# plt.show()

# d
# solve(0.01, None)

_h.list_timings()

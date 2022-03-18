from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

import dolfinx as df

class LoadSensor:
    def __init__(self, dofs, residual, name="load"):
        self.dofs, self.residual, self.name = dofs, residual, name
        self.comm = MPI.COMM_WORLD

    def measure(self):
        with self.residual.localForm() as lf:
            local_load = np.sum(lf[self.dofs])

        return self.comm.reduce(local_load, op=MPI.SUM, root=0)

class DisplacementSensor:
    def __init__(self, bc, name="disp"):
        self.bc, self.name = bc, name

    def measure(self):
        return float(self.bc.g.value) # .value itself is a 0-dim np.array

class Experiment:
    pass

class UnitSquareExperiment(Experiment):
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

if __name__ == "__main__":
    all_experiments = []
    all_experiments.append(UnitSquareExperiment(5))

    for exp in all_experiments:
        V = df.fem.VectorFunctionSpace(exp.mesh, ("P", 1))
        bcs = exp.bcs(V)
        assert len(exp.load_dofs) > 0



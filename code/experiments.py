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
        return float(self.bc.g.value)  # .value itself is a 0-dim np.array


class Experiment:
    pass

def _point_at(p):
    p = np.atleast_1d(p)
    def boundary(x):
        condition = np.isclose(x[0], p[0])
        # check remaining coordinates
        for i in range(1, len(p)):
            condition = np.logical_and(condition, np.isclose(x[i], p[i]))
        return condition

    return boundary

class UniaxialTruss(Experiment):
    def __init__(self, N=10, L=1.):
        self.L = L
        self.mesh = df.mesh.create_interval(MPI.COMM_WORLD, N, [0., L])
        self.u_bc = df.fem.Constant(self.mesh, 0.0)

    def get_bcs(self, V):
        mesh = self.mesh
        b_facets_l = df.mesh.locate_entities_boundary(mesh, 0, _point_at(0))
        b_facets_r = df.mesh.locate_entities_boundary(mesh, 0, _point_at(self.L))

        b_dofs_l = df.fem.locate_dofs_topological(V.sub(0), 0, b_facets_l)
        b_dofs_r = df.fem.locate_dofs_topological(V.sub(0), 0, b_facets_r)
        # print(b_dofs_o)

        self.bcs = [
            df.fem.dirichletbc(PETSc.ScalarType(0), b_dofs_l, V.sub(0)),
            df.fem.dirichletbc(self.u_bc, b_dofs_r, V.sub(0)),
        ]

        self.load_dofs = b_dofs_r
        self.load_bc = self.bcs[1]
        return self.bcs

    def set_bcs(self, value):
        self.u_bc.value = value

class UnitSquareExperiment(Experiment):
    def __init__(self, N, cell_type=df.mesh.CellType.triangle):
        self.mesh = df.mesh.create_unit_square(
            MPI.COMM_WORLD, N, N, cell_type
        )
        self.u_bc = df.fem.Constant(self.mesh, 0.0)

    def get_bcs(self, V):
        def left(x):
            return np.isclose(x[0], 0)

        def right(x):
            return np.isclose(x[0], 1)


        mesh = self.mesh
        dim = self.mesh.topology.dim - 1
        b_facets_l = df.mesh.locate_entities_boundary(mesh, dim, left)
        b_facets_r = df.mesh.locate_entities_boundary(mesh, dim, right)
        b_facets_o = df.mesh.locate_entities_boundary(mesh, dim - 1, _point_at((0,0)))

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
        self.load_bc = self.bcs[1]
        return self.bcs

    def set_bcs(self, value):
        self.u_bc.value = value


class BendingThreePoint(Experiment):
    def __init__(self, lx=200, ly=30, nx=100, ny=15):
        self.lx, self.ly = lx, ly
        self.mesh = df.mesh.create_rectangle(
            MPI.COMM_WORLD, [[0, 0], [lx, ly]], [nx, ny], df.mesh.CellType.triangle, diagonal=df.mesh.DiagonalType.crossed
        )
        self.u_bc = df.fem.Constant(self.mesh, 0.0)

    def get_bcs(self, V):

        mesh = self.mesh
        # l,r,t = left, right, top
        b_facets_l = df.mesh.locate_entities_boundary(mesh, 0, _point_at((0,0)))
        b_facets_r = df.mesh.locate_entities_boundary(mesh, 0, _point_at((self.lx, 0)))
        b_facets_t = df.mesh.locate_entities_boundary(mesh, 0, _point_at((self.lx/2, self.ly)))

        b_dofs_lx = df.fem.locate_dofs_topological(V.sub(0), 0, b_facets_l)
        b_dofs_ly = df.fem.locate_dofs_topological(V.sub(1), 0, b_facets_l)
        b_dofs_r = df.fem.locate_dofs_topological(V.sub(1), 0, b_facets_r)
        b_dofs_t = df.fem.locate_dofs_topological(V.sub(1), 0, b_facets_t)
        # print(b_dofs_o)

        assert len(b_dofs_lx) == 1
        assert len(b_dofs_ly) == 1
        assert len(b_dofs_r) == 1
        assert len(b_dofs_t) == 1

        self.bcs = [
            df.fem.dirichletbc(PETSc.ScalarType(0), b_dofs_lx, V.sub(0)),
            df.fem.dirichletbc(PETSc.ScalarType(0), b_dofs_ly, V.sub(1)),
            df.fem.dirichletbc(PETSc.ScalarType(0), b_dofs_r, V.sub(1)),
            df.fem.dirichletbc(self.u_bc, b_dofs_t, V.sub(1)),
        ]

        self.load_dofs = b_dofs_t
        self.load_bc = self.bcs[3]
        return self.bcs

    def set_bcs(self, value):
        self.u_bc.value = value


if __name__ == "__main__":
    all_experiments = []
    all_experiments.append(UnitSquareExperiment(5))
    all_experiments.append(BendingThreePoint())
    all_experiments.append(UniaxialTruss())

    for exp in all_experiments:

        V = df.fem.VectorFunctionSpace(exp.mesh, ("P", 1))
        bcs = exp.get_bcs(V)
        assert len(exp.load_dofs) > 0

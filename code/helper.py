import dolfinx as df
from petsc4py import PETSc
from mpi4py import MPI
from contextlib import ExitStack
import numpy as np
from enum import Enum

def print0(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)

def nullspace(V):
    f = [nullspace_1d, nullspace_2d, nullspace_3d]
    return f[V.dofmap.index_map_bs - 1](V)

def nullspace_1d(V):
    """Build PETSc nullspace for elasticity"""

    # Create list of vectors for building nullspace
    index_map = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    ns = [df.la.create_petsc_vector(index_map, bs)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in ns]
        basis = [np.asarray(x) for x in vec_local]

        # Get dof indices for each subspace (x and y dofs)
        dofs = [V.sub(0).dofmap.list.array]

        # Build the translational rigid body mode
        basis[0][dofs[0]] = 1.0

    # Orthonormalise the three vectors
    df.la.orthonormalize(ns)
    assert df.la.is_orthonormal(ns)

    return PETSc.NullSpace().create(vectors=ns)


def nullspace_2d(V):
    """Build PETSc nullspace for elasticity"""

    # Create list of vectors for building nullspace
    index_map = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    ns = [df.la.create_petsc_vector(index_map, bs) for i in range(3)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in ns]
        basis = [np.asarray(x) for x in vec_local]

        # Get dof indices for each subspace (x and y dofs)
        dofs = [V.sub(i).dofmap.list.array for i in range(2)]

        # Build the three translational rigid body modes
        for i in range(2):
            basis[i][dofs[i]] = 1.0

        # Build the rotational rigid body mode
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.array
        x0, x1, = (
            x[dofs_block, 0],
            x[dofs_block, 1],
        )
        basis[2][dofs[0]] = -x1
        basis[2][dofs[1]] = x0

    # Orthonormalise the three vectors
    df.la.orthonormalize(ns)
    assert df.la.is_orthonormal(ns)

    return PETSc.NullSpace().create(vectors=ns)


def nullspace_3d(V):
    """Build PETSc nullspace for 3D elasticity"""

    # Create list of vectors for building nullspace
    index_map = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    ns = [la.create_petsc_vector(index_map, bs) for i in range(6)]
    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in ns]
        basis = [np.asarray(x) for x in vec_local]

        # Get dof indices for each subspace (x, y and z dofs)
        dofs = [V.sub(i).dofmap.list.array for i in range(3)]

        # Build the three translational rigid body modes
        for i in range(3):
            basis[i][dofs[i]] = 1.0

        # Build the three rotational rigid body modes
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.array
        x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
        basis[3][dofs[0]] = -x1
        basis[3][dofs[1]] = x0
        basis[4][dofs[0]] = x2
        basis[4][dofs[2]] = -x0
        basis[5][dofs[2]] = x1
        basis[5][dofs[1]] = -x2

    # Orthonormalise the six vectors
    la.orthonormalize(ns)
    assert la.is_orthonormal(ns)

    return PETSc.NullSpace().create(vectors=ns)


def eval_function_at_points(f, points):
    """
    From https://jorgensd.github.io/df-tutorial/chapter1/membrane_code.html
    """
    mesh = f.function_space.mesh
    tree = df.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cell_candidates = df.geometry.compute_collisions(tree, points)
    colliding_cells = df.geometry.compute_colliding_cells(mesh, cell_candidates, points)

    points_on_proc = []
    cells = []
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    return f.eval(points_on_proc, cells), points_on_proc


def create_solver(problem, iterative=False, linesearch="basic", monitor_krylov=False, monitor_newton=True):
    snes = PETSc.SNES().create()

    def snes_J(snes, x, A, P):
        """Assemble the Jacobian matrix."""
        problem.J(x, A)

    def snes_F(snes, x, b):
        problem.form(x)
        problem.F(x, b)
 
    b = df.fem.petsc.create_vector(problem.R)
    J = df.fem.petsc.create_matrix(problem.dR)
    snes.setFunction(snes_F, b)
    snes.setJacobian(snes_J, J)

    opts = PETSc.Options()

    opts["snes_linesearch_type"] = linesearch
    if iterative:
        opts["solve_ksp_type"] = "gmres"
        opts["ksp_rtol"] = 1.0e-10
        opts["pc_type"] = "gamg"

        # Use Chebyshev smoothing for multigrid
        opts["mg_levels_ksp_type"] = "chebyshev"
        opts["mg_levels_pc_type"] = "jacobi"

        # Improve estimate of eigenvalues for Chebyshev smoothing
        opts["mg_levels_esteig_ksp_type"] = "gmres"
        opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20

    snes.setFromOptions()
    snes.setTolerances(atol=1.e-10, rtol=1.0e-10, max_it=10)

    if not iterative:
        snes.getKSP().setType("preonly")
        snes.getKSP().setTolerances(rtol=1.e-10, atol=1.e-10)
        snes.getKSP().getPC().setType("lu")

    if monitor_newton:
        snes.setMonitor(
            lambda _, its, rnorm: print0(f"  Newton: {its}, |R|/|R0| = {rnorm:6.3e}")
        )
    if monitor_krylov:
        snes.getKSP().setMonitor(
            lambda _, its, rnorm: print0(f"    krylov: {its}, |R|/|R0| = {rnorm:6.3e}")
        )

    def solve():
        with df.common.Timer("SNES solve"):
            snes.solve(None, problem.u.vector)
        return snes.its, snes.converged


    return snes, solve

class MeasurementSystem(dict):
    def __init__(self):
        super().__init__()
        self._sensors = []

    def add(self, sensor):
        assert sensor.name not in self
        self._sensors.append(sensor)
        self[sensor.name] = []

    def measure(self):
        for sensor in self._sensors:
            self[sensor.name].append(sensor.measure())

def list_timings():
    df.common.list_timings(MPI.COMM_WORLD, [df.common.TimingType.wall], df.common.Reduction.average)

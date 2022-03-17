import dolfinx as df
from petsc4py import PETSc
from contextlib import ExitStack
import numpy as np
from enum import Enum


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


def create_solver(problem, iterative=False, linesearch="basic", monitor_krylov=False):
    snes = PETSc.SNES().create()

    b = df.fem.petsc.create_vector(problem.R)
    J = df.fem.petsc.create_matrix(problem.dR)
    snes.setFunction(problem.F, b)
    snes.setJacobian(problem.J, J)

    opts = PETSc.Options()

    opts["snes_linesearch_type"] = linesearch
    if iterative:
        opts["solve_ksp_type"] = "gmres"
        opts["ksp_rtol"] = 1.0e-14
        opts["pc_type"] = "gamg"

        # Use Chebyshev smoothing for multigrid
        opts["mg_levels_ksp_type"] = "chebyshev"
        opts["mg_levels_pc_type"] = "jacobi"

        # Improve estimate of eigenvalues for Chebyshev smoothing
        opts["mg_levels_esteig_ksp_type"] = "gmres"
        opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20

    snes.setFromOptions()
    snes.setTolerances(atol=1.e-10, rtol=1.0e-10, max_it=20)
    # snes.getKSP().setTolerances(rtol=1.e-10)

    if not iterative:
        snes.getKSP().setType("preonly")
        snes.getKSP().getPC().setType("lu")

    snes.setMonitor(
        lambda _, its, rnorm: print(f"  Newton: {its}, |R|/|R0| = {rnorm:6.3e}")
    )
    if monitor_krylov:
        snes.getKSP().setMonitor(
            lambda _, its, rnorm: print(f"    krylov: {its}, |R|/|R0| = {rnorm:6.3e}")
        )

    return snes

import dolfinx as df
import numpy as np

import helper as _h  # underscore indicates custom stuff
import experiments as _exp
import materials as _mat
from problem import MechanicsProblem

import pytest

E, nu = 20000, 0.2
u_bc = 42.

def get_problem(cell_type, deg=1):
    experiment = _exp.UnitSquareExperiment(20, cell_type)
    experiment.set_bcs(u_bc)
    mat = _mat.HookesLaw(E, nu)
    return MechanicsProblem(experiment, mat, deg=deg)

def check_solution(u, u_bc):
    x_check = np.linspace(0, 1, 42)
    x, y = np.meshgrid(x_check, x_check)
    xs = np.vstack([x.flat, y.flat, np.zeros(len(x.flat))]).T
    u_fem, xs = _h.eval_function_at_points(u, xs)

    u_ref = np.array([xs[:, 0] * u_bc, -nu * xs[:, 1] * u_bc]).T
    np.testing.assert_array_almost_equal(u_fem, u_ref)

@pytest.mark.parametrize("cell_type", (df.mesh.CellType.triangle, df.mesh.CellType.quadrilateral))
@pytest.mark.parametrize("deg", (1, 2, 3))
def test_df_newtonsolver(cell_type, deg):
    problem = get_problem(cell_type, deg)
    converged, it = problem.solve()
    assert converged
    assert it == 1
    check_solution(problem.u, u_bc)

@pytest.mark.parametrize("cell_type", (df.mesh.CellType.triangle, df.mesh.CellType.quadrilateral))
@pytest.mark.parametrize("iterative", (True, False))
def test_snes(cell_type, iterative):
    problem = get_problem(cell_type)
    snes, solve_method = _h.create_solver(problem, iterative)
    converged, it = solve_method()
    assert converged
    assert it == 1
    check_solution(problem.u, u_bc)

def test_small_increments():
    problem = get_problem(df.mesh.CellType.quadrilateral)
    for u_bc_value in np.linspace(0.1, 0.1001, 10):
        problem.experiment.set_bcs(u_bc_value)
        converged, it = problem.solve()
        assert converged
        assert it == 1
    check_solution(problem.u, u_bc_value)


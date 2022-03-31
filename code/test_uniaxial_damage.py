import dolfinx as df
import numpy as np

import pytest

import helper as _h  # underscore indicates custom stuff
from helper import print0
import experiments as _exp
import materials as _mat
from problem import MechanicsProblem


def test_full_damage():

    experiment = _exp.UniaxialTruss(L=1, N=5)
    mat = _mat.LocalDamage(
        E=20000.0,
        nu=0.0,
        ft=4,
        beta=200,
        constraint=_mat.Constraint.UNIAXIAL_STRESS,
        alpha=1.0,
    )

    problem = MechanicsProblem(experiment, mat, deg=1, q_deg=1)

    storage = _h.MeasurementSystem()
    storage.add(_exp.LoadSensor(experiment.load_dofs, problem.residual, name="load"))
    storage.add(_exp.DisplacementSensor(experiment.load_bc, name="disp"))

    snes, snes_solve = _h.create_solver(
        problem, iterative=False, monitor_krylov=True, linesearch="basic"
    )

    for u_bc in np.linspace(0, 400 * mat.k0, 400):

        experiment.set_bcs(u_bc)

        it, converged = snes_solve()
        assert converged

        problem.update()
        storage.measure()

    Gf = np.trapz(storage["load"], storage["disp"])
    Gf_ref = 0.5 * mat.k0 * mat.ft + mat.Gf

    assert Gf_ref == pytest.approx(Gf, rel=1e-3)


import dolfinx as df
import numpy as np

import pyvista as pv
import pyvistaqt as pvqt

import helper as _h  # underscore indicates custom stuff
from helper import print0
import experiments as _exp
import materials as _mat
from problem import MechanicsProblem

import matplotlib.pyplot as plt

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

def main():

    experiment = _exp.UnitSquareExperiment(3)
    mat = _mat.LocalDamage(E=20000., nu=0.0,ft=4,beta=350, constraint=_mat.Constraint.PLANE_STRAIN)
    problem = MechanicsProblem(experiment, mat, deg=1)

    solver = _h.create_solver(problem, linesearch="bt")


    storage = MeasurementSystem()
    storage.add(_exp.LoadSensor(experiment.load_dofs, problem.residual, name="load"))
    storage.add(_exp.DisplacementSensor(experiment.bcs[1], name="disp"))

    experiment.set_bcs(mat.ft/mat.E)
    solver.solve(None, problem.u.vector)
    problem.update()
    
    experiment.set_bcs(mat.ft/mat.E*1.01)
    solver.solve(None, problem.u.vector)
    return

    for i, u_bc in enumerate(np.linspace(0, 0.001, 51)):
        print(f"load step {i} with {u_bc = }")
        experiment.set_bcs(u_bc)
        # solver.setInitialGuess(problem.u)
        solver.solve(None, problem.u.vector)
        if solver.its == 20:
            exit()
        problem.update()

        # problem.F(None, problem.u.vector, residual, apply_bc=False)
        storage.measure()

    plt.plot(storage["disp"], storage["load"])
    plt.show()

if __name__ == "__main__":
    main()

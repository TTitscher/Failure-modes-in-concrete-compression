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

    experiment = _exp.UnitSquareExperiment(10)
    mat = _mat.LocalDamage(E=20000., nu=0.2,ft=2,beta=100, constraint=_mat.Constraint.PLANE_STRAIN)
    # mat = _mat.HookesLaw(E=20000., nu=0.2, constraint=_mat.Constraint.PLANE_STRESS)
    problem = MechanicsProblem(experiment, mat, deg=1)
    problem.update()
    print(mat.kappa.shape)
    mat.kappa[300] = mat.ft/mat.E * 3
    # return

    storage = MeasurementSystem()
    storage.add(_exp.LoadSensor(experiment.load_dofs, problem.residual, name="load"))
    storage.add(_exp.DisplacementSensor(experiment.bcs[1], name="disp"))

    f = df.io.XDMFFile(experiment.mesh.comm, "displacements.xdmf", "w")
    f.write_mesh(experiment.mesh)
    
    for i, u_bc in enumerate(np.linspace(0., 0.01, 10)):
    # for i, u_bc in enumerate(np.linspace(0.06, 0.07, 11)):
        print0(f"load step {i} with {u_bc = }")
        # problem.u.x.array[:] = 0.
        experiment.set_bcs(u_bc)

        # solver.setInitialGuess(problem.u)
        it, converged = problem.solve()
        problem.u.x.scatter_forward()
        if not converged:
            break
        # print(r)
        # if solver.its == 20:
            # exit()
        f.write_function(problem.u, u_bc)
        # problem.update()
        storage.measure()

    if experiment.mesh.comm.rank == 0:
        plt.plot(storage["disp"], storage["load"], "-kx")
        plt.show()

if __name__ == "__main__":
    main()

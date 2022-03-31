import dolfinx as df
import numpy as np

import helper as _h  # underscore indicates custom stuff
from helper import print0
import experiments as _exp
import materials as _mat
from problem import MechanicsProblem

import matplotlib.pyplot as plt

def main():

    experiment = _exp.BendingThreePoint()
    # experiment = _exp.UnitSquareExperiment(10)
    mat = _mat.LocalDamage(E=20000., nu=0.2,ft=4,beta=10, constraint=_mat.Constraint.PLANE_STRAIN)
    # mat = _mat.HookesLaw(E=20000., nu=0.2, constraint=_mat.Constraint.PLANE_STRESS)
    problem = MechanicsProblem(experiment, mat, deg=1, q_deg=1)
    # return

    storage = _h.MeasurementSystem()
    storage.add(_exp.LoadSensor(experiment.load_dofs, problem.residual, name="load"))
    storage.add(_exp.DisplacementSensor(experiment.load_bc, name="disp"))

    f = df.io.XDMFFile(experiment.mesh.comm, "displacements.xdmf", "w")
    f.write_mesh(experiment.mesh)

    snes, snes_solve = _h.create_solver(problem, iterative=False, monitor_krylov=True, linesearch="basic")
 
    u_bu = problem.u.copy()

    # scalar_plot =

    dt = 0.001
    t = dt
    t_end = 1.
    while t < t_end:
        # try step t+dt

        u_bc = -(t+dt)
        # u_bc = t+dt
        print0(f"load step {t}+{dt} with {u_bc = }")
        experiment.set_bcs(u_bc)

        # it, converged = problem.solve()
        it, converged = snes_solve()
        if converged:
            t += dt
            print(f"Converged after {it} iterations")
            u_bu.x.array[:] = problem.u.x.array[:]
            problem.update()
            
            # pp
            storage.measure()
            f.write_function(problem.u, t)

            if it < 4:
                dt *= 1.5

            dt = min(dt, t_end - t)

        else:
            # restore solution and decrease time step
            problem.u.x.array[:] = u_bu.x.array[:]
            dt *= 0.5

            if dt < 1.e-8:
                break

        print(np.max(problem.material.kappa))

    if experiment.mesh.comm.rank == 0:
        plt.plot(storage["disp"], storage["load"], "-kx")
        plt.show()

if __name__ == "__main__":
    main()

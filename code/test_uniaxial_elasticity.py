import sys

import dolfinx as df
import numpy as np

import pyvista as pv
import pyvistaqt as pvqt

import helper as _h  # underscore indicates custom stuff
import experiments as _exp
import materials as _mat
from problem import MechanicsProblem


experiment = _exp.UnitSquareExperiment(10)
mat = _mat.HookesLaw(20000, 0.2)
problem = MechanicsProblem(experiment, mat)

solver = _h.create_solver(problem)

u_bc = 42.0
experiment.set_bcs(u_bc)
solver.solve(None, problem.u.vector)

u = problem.u


# build a coordinate grid `xs` at which u_fem and u_ref (analytical) are 
# compared
x_check = np.linspace(0, 1, 42)
x, y = np.meshgrid(x_check, x_check)
xs = np.vstack([x.flat, y.flat, np.zeros(len(x.flat))]).T
u_fem, points = _h.eval_function_at_points(u, xs)

u_ref = np.array([xs[:, 0] * u_bc, -mat.nu * xs[:, 1] * u_bc]).T

assert np.linalg.norm(u_fem - u_ref) < 1.0e-10

f = df.io.XDMFFile(experiment.mesh.comm, "displacements.xdmf", "w")
f.write_mesh(experiment.mesh)
f.write_function(u)

import random
import unittest
from mpi4py import MPI
import dolfinx as df
import ufl
import numpy as np
from hypothesis import given, reproduce_failure
import hypothesis.strategies as st

from timestepping import TimeStepper

class DeterministicSolve:
    """
    Returns the same (randomly created) boolean for the same combination of
    (t, dt).
    Keeps track of how many times the same value was requested to find
    infinite loops.
    """
    def __init__(self):
        self.memory = {}
        random.seed(6174)

    def __call__(self, t, dt):
        value = self.memory.get((t, dt))

        if value is not None:
            return 3, value

        value = random.choice([True, True, True, True, False])
        if dt < 1.e-5:
            value = True
        self.memory[(t, dt)] = value

        return 3, value


class TestEquidistant(unittest.TestCase):
    def setUp(self):
        solve = lambda t, dt: (3, True)
        pp = lambda t: None
        self.stepper = TimeStepper(solve, pp)

    def test_run(self):
        self.assertTrue(self.stepper.equidistant(5.0, 1.0))

    def test_run_failed(self):
        self.stepper._solve = lambda t, dt: (3, False)
        self.assertFalse(self.stepper.equidistant(100.0, 1.0, show_bar=True))

    def test_invalid_solve(self):
        self.stepper._solve = lambda t, dt: (3, "False")
        self.assertRaises(Exception, self.stepper.equidistant, 5.0, 1.0)

    @given(st.lists(st.floats(0.0, 5.0)))
    def test_checkpoints(self, checkpoints):
        visited_timesteps = []
        pp = lambda t: visited_timesteps.append(t)
        self.stepper._post_process = pp
        self.assertTrue(self.stepper.equidistant(5.0, 1.0, checkpoints=checkpoints))
        for checkpoint in checkpoints:
            self.assertIn(checkpoint, visited_timesteps)

        self.assertIn(0.0, visited_timesteps)
        self.assertIn(5.0, visited_timesteps)

        # check for duplicates
        self.assertEqual(len(visited_timesteps), np.unique(visited_timesteps).size)

    @given(st.lists(st.floats(max_value=-0.0, exclude_max=True), min_size=1))
    def test_too_low_checkpoints(self, checkpoints):
        with self.assertRaises(RuntimeError) as cm:
            self.stepper.equidistant(5.0, 1.0, checkpoints=checkpoints)

    @given(st.lists(st.floats(min_value=5.0, exclude_min=True), min_size=1))
    def test_too_high_checkpoints(self, checkpoints):
        with self.assertRaises(RuntimeError) as cm:
            self.stepper.equidistant(5.0, 1.0, checkpoints=checkpoints)


class TestAdaptive(unittest.TestCase):
    def setUp(self):
        solve = DeterministicSolve()
        # solve = lambda t: (3, random.choice([True, True, True, True, False]))
        pp = lambda t: None
        u = df.fem.Function(df.fem.FunctionSpace(df.mesh.create_unit_interval(MPI.COMM_WORLD, 10), ("P", 1)))
        self.stepper = TimeStepper(solve, pp, u)

    def test_run(self):
        self.assertTrue(self.stepper.adaptive(1.5, 1.0))

    def test_run_nicely(self):
        visited_timesteps = []
        pp = lambda t: visited_timesteps.append(t)
        self.stepper._post_process = pp
        self.stepper._solve = lambda t, dt: (7, True)
        self.assertTrue(self.stepper.adaptive(1.0))
        self.assertAlmostEqual(visited_timesteps[0], 0)
        self.assertAlmostEqual(visited_timesteps[-1], 1)

    def test_dt_chaining(self):
        special_stepper = self.stepper.dt_max(0.5).dt_min(0.1)
        self.assertAlmostEqual(special_stepper._dt_max, 0.5)
        self.assertAlmostEqual(special_stepper._dt_min, 0.1)

    def test_checkpoint_step_fails(self):
        cps = [0.5]
        self.stepper.dt_max = 1.
        # first time step 1.0 is bigger than the first checkpoint. 
        # So dt --> 0.5, dt0--> 0.1
        self.stepper._solve.memory[(0, cps[0])] = False
        self.stepper.adaptive(1.0, checkpoints=cps)


    def test_invalid_solve_first_str(self):
        self.stepper._solve = lambda t, dt: ("3", True)
        self.assertRaises(Exception, self.stepper.adaptive, 1.5, 1.0)

    def test_invalid_solve_first_bool(self):
        # You may switch arguments and put the bool first. We have to handle
        # this case since booleans are some kind of subclass of int.
        # So False > 5 will not produce errors.
        self.stepper._solve = lambda t, dt: (False, True)
        self.assertRaises(Exception, self.stepper.adaptive, 1.5, 1.0)

    def test_invalid_solve_second(self):
        self.stepper._solve = lambda t, dt: (3, "False")
        self.assertRaises(Exception, self.stepper.adaptive, 1.5, 1.0)

    @given(st.lists(st.floats(0.0, 1.5)))
    def test_checkpoints(self, checkpoints):
        visited_timesteps = []
        pp = lambda t: visited_timesteps.append(t)
        self.stepper._post_process = pp
        self.assertTrue(self.stepper.adaptive(1.5, dt=1.0, checkpoints=checkpoints))

        eps = np.finfo(float).eps  # eps float
        for checkpoint in checkpoints:
            self.assertTrue(np.isclose(visited_timesteps, checkpoint, atol=eps).any())

        self.assertTrue(np.isclose(visited_timesteps, 0, atol=eps).any())
        self.assertTrue(np.isclose(visited_timesteps, 1.5, atol=eps).any())

        # check for duplicates
        self.assertEqual(len(visited_timesteps), np.unique(visited_timesteps).size)

    @given(st.lists(st.floats(max_value=-0.0, exclude_max=True), min_size=1))
    def test_too_low_checkpoints(self, checkpoints):
        with self.assertRaises(RuntimeError) as cm:
            self.stepper.adaptive(1.5, dt=1.0, checkpoints=checkpoints)

    @given(st.lists(st.floats(min_value=1.5, exclude_min=True), min_size=1))
    def test_too_high_checkpoints(self, checkpoints):
        with self.assertRaises(RuntimeError) as cm:
            self.stepper.adaptive(1.5, dt=1.0, checkpoints=checkpoints)

class TestAdaptiveMixed(unittest.TestCase):
    def test_mixed(self):
        solve = DeterministicSolve()
        # solve = lambda t: (3, random.choice([True, True, True, True, False]))
        pp = lambda t: None
        mesh = df.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
        elem_1 = ufl.VectorElement('P', mesh.ufl_cell(), 1, dim=2)
        elem_2 = ufl.FiniteElement('P', mesh.ufl_cell(), 1)
        V = df.fem.FunctionSpace(mesh, elem_1 * elem_2)
        u = df.fem.Function(V)

        u_ref0 = u.split()[0]
        # check for the link between u and u_ref0
        u.vector.array[:] = 42.
        self.assertAlmostEqual(u_ref0.vector.array[0], 42.)

        stepper = TimeStepper(solve, pp, u)
        stepper.adaptive(1.)
        
        # make sure that the link between u and u_ref0 still remains!
        u.vector.array[0] = 6174.
        self.assertAlmostEqual(u_ref0.vector.array[0], 6174.)

if __name__ == "__main__":
    unittest.main()

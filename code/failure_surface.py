"""
Recreate the typical failure surface plots (in 2D) for a
given material law.
"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from util import principal_components


@dataclass
class Material:
    E: float = 20000.0  # [MPa] Young's modulus
    ft: float = 4.0  # [MPa] tensile strength
    nu: float = 0.2  # [-] Poission's ratio
    k: float = 10.0  # [-] compressive to tensile strength ratio

    @property
    def C_plane_stress(self):
        C11 = self.E / (1.0 - self.nu * self.nu)
        C12 = C11 * self.nu
        C33 = C11 * 0.5 * (1.0 - self.nu)
        C = np.array([[C11, C12, 0.0], [C12, C11, 0.0], [0.0, 0.0, C33]])


def modified_mises(mat, eps):
    eps = np.asarray(eps)
    nu, k = mat.nu, mat.k

    K1 = (k - 1.0) / (2.0 * k * (1.0 - 2.0 * nu))
    K2 = 3.0 / (k * (1.0 + nu) ** 2)

    exx, eyy, exy = eps[0::3], eps[1::3], eps[2::3]
    I1 = exx + eyy
    J2 = 1.0 / 6.0 * ((exx - eyy) ** 2 + exx ** 2 + eyy ** 2) + (0.5 * exy) ** 2

    A = np.sqrt(K1 ** 2 * I1 ** 2 + K2 * J2) + 1.0e-14
    eeq = K1 * I1 + A
    return eeq


def iterate_failure_surface(f, mat, N=1000):

    thetas = np.linspace(0, 2 * np.pi, N)
    eps = np.array([np.cos(thetas), np.sin(thetas), np.zeros(N)]).T

    x0 = np.ones(N)

    def failure(factor):
        e = eps * factor[:, np.newaxis]
        return f(mat, e.flatten()) - 1

    r = scipy.optimize.newton(failure, x0=x0)

    points = eps * r[:, np.newaxis]
    return points[:,0], points[:,1]


def main():
    mat = Material(nu=0.0)
    points = iterate_failure_surface(modified_mises, mat)

    ax = plt.gca()
    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(*points)
    plt.grid()
    plt.show()

    # s1, s2 = 0.5, 0.


if __name__ == "__main__":
    main()

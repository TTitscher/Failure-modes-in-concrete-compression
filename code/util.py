import numpy as np
import pytest


def principal_components(s):
    """
    s: vector (3,) or matrix (N,3) that represents
    """
    N = 1
    if len(s.shape) == 2:
        N = len(s)
        s = s.T

    p = -0.5 * (s[0] + s[1])  #
    q = s[0] * s[1] - s[2] ** 2  # det(s)

    D = p ** 2 - q
    assert np.all(D >= -1.0e-15)
    sqrtD = np.sqrt(D)

    return -p + sqrtD, -p - sqrtD

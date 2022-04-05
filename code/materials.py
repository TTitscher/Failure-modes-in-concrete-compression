import numpy as np
from enum import Enum
import ufl
from dataclasses import dataclass


class Constraint(Enum):
    UNIAXIAL_STRAIN = 1
    UNIAXIAL_STRESS = 2
    PLANE_STRAIN = 3
    PLANE_STRESS = 4
    FULL = 5


@dataclass
class HookesLaw:

    E: float = 20000.0
    nu: float = 0.2
    constraint: Constraint = Constraint.PLANE_STRESS

    @property
    def qdim(self):
        return {
            Constraint.UNIAXIAL_STRAIN: 1,
            Constraint.UNIAXIAL_STRESS: 1,
            Constraint.PLANE_STRAIN: 3,
            Constraint.PLANE_STRESS: 3,
            Constraint.FULL: 6,
        }[self.constraint]

    @property
    def C(self):
        E, nu = self.E, self.nu

        if self.constraint in [Constraint.UNIAXIAL_STRAIN, Constraint.UNIAXIAL_STRESS]:
            return np.array([[E]])

        if self.constraint == Constraint.PLANE_STRESS:
            C11 = E / (1.0 - nu * nu)
            C12 = C11 * nu
            C33 = C11 * 0.5 * (1.0 - nu)
            return np.array([[C11, C12, 0.0], [C12, C11, 0.0], [0.0, 0.0, C33]])

        if self.constraint == Constraint.PLANE_STRAIN:
            l = E * nu / (1 + nu) / (1 - 2 * nu)
            m = E / (2.0 * (1 + nu))
            return np.array([[2 * m + l, l, 0], [l, 2 * m + l, 0], [0, 0, m]])

        raise NotImplementedError()

    def eps(self, u):
        e = ufl.sym(ufl.grad(u))
        if self.constraint in [Constraint.PLANE_STRESS, Constraint.PLANE_STRAIN]:
            return ufl.as_vector([e[0, 0], e[1, 1], 2 * e[0, 1]])
        if self.constraint in [Constraint.UNIAXIAL_STRAIN, Constraint.UNIAXIAL_STRESS]:
            return ufl.as_vector([e[0, 0]])
        if self.constraint == Constraint.FULL:
            return ufl.as_vector(
                [e[0, 0], e[1, 1], e[2, 2], 2 * e[1, 2], 2 * e[0, 2], 2 * e[0, 1]]
            )
        raise NotImplementedError()

    def evaluate(self, strains):
        C = self.C
        assert C.shape == (self.qdim, self.qdim)

        eps = strains.reshape((-1, self.qdim))
        n_gauss = len(eps)
        return (eps @ self.C).flatten(), np.tile(self.C.flatten(), n_gauss)

    def update(self, strains):
        pass


def damage_exponential(mat, k):
    k0 = mat.ft / mat.E
    a = mat.alpha
    b = mat.beta

    w = 1.0 - k0 / k * (1.0 - a + a * np.exp(b * (k0 - k)))
    dw = k0 / k * ((1.0 / k + b) * a * np.exp(b * (k0 - k)) + (1.0 - a) / k)

    return w, dw


def modified_mises_strain_norm(mat, eps):
    if mat.qdim == 1:
        return eps.flatten(), np.ones((len(eps), 1))


    nu, k = mat.nu, mat.k

    K1 = (k - 1.0) / (2.0 * k * (1.0 - 2.0 * nu))
    K2 = 3.0 / (k * (1.0 + nu) ** 2)

    exx, eyy, exy = eps[:, 0], eps[:, 1], eps[:, 2]
    I1 = exx + eyy
    J2 = 1.0 / 6.0 * ((exx - eyy) ** 2 + exx ** 2 + eyy ** 2) + (0.5 * exy) ** 2

    A = np.sqrt(K1 ** 2 * I1 ** 2 + K2 * J2) + 1.0e-14
    eeq = K1 * I1 + A

    dJ2dexx = 1.0 / 3.0 * (2 * exx - eyy)
    dJ2deyy = 1.0 / 3.0 * (2 * eyy - exx)
    dJ2dexy = 0.5 * exy

    deeq = np.empty_like(eps)
    deeq[:, 0] = K1 + 1.0 / (2 * A) * (2 * K1 * K1 * I1 + K2 * dJ2dexx)
    deeq[:, 1] = K1 + 1.0 / (2 * A) * (2 * K1 * K1 * I1 + K2 * dJ2deyy)
    deeq[:, 2] = 1.0 / (2 * A) * (K2 * dJ2dexy)
    return eeq, deeq


@dataclass
class LocalDamage(HookesLaw):
    # tensile strength          [N/mm²]
    ft: float = 10.0
    # compressive-tensile ratio   [-]
    k: float = 10.0
    # residual strength factor   [-]
    alpha: float = 0.99
    # fracture energy parameters [-]
    beta: float = 100.0
    # history variable           [-]
    kappa: None = None
    # damage variable           [-]
    w: None = None
    # damage law
    dmg: None = damage_exponential

    @property
    def k0(self):
        return self.ft / self.E

    @property
    def Gf(self):
        return self.ft / self.beta
    
    def evaluate(self, strains):
        """
        sigma = (1-w(k(|eps|))) * C : eps

        dsigma/deps = (1-w) C - C : eps * dw/dk * dk/d|eps| X d|eps|/deps
                                ---------------------------   -----------
                                           P1 \in R3x1          \in R3x1
                                -----------------------------------------
                                              row_wise_outer \in R3x3
        """
        eps = strains.reshape(-1, self.qdim)
        n_gauss = len(eps)

        eeq, deeq = modified_mises_strain_norm(self, eps)
        kappa, dkappa = self.kappa_kkt(eeq)
        w, dw = self.dmg(self, kappa)

        C = self.C
        sigma = eps @ C * (1 - w)[:, None]
        dsigma = np.tile(C.flatten(), (n_gauss, 1)) * (1 - w)[:, None]

        P1 = eps @ C * dw[:, None] * dkappa[:, None]

        # dont ask... https://stackoverflow.com/questions/48498662/numpy-row-wise-outer-product-of-two-matrices
        row_wise_outer = np.matmul(deeq[:, :, None], P1[:, None, :])
        # row_wise_outer = np.matmul(P1[:, :, None], deeq[:, None, :])

        # print(row_wise_outer.shape)
        return sigma.flat, dsigma.flatten() - row_wise_outer.flat

    def kappa_kkt(self, strain_norm):
        if self.kappa is None:
            self.kappa = self.ft / self.E

        kappa = np.maximum(strain_norm, self.kappa)
        dkappa = (strain_norm >= kappa).astype(int)
        return kappa, dkappa

    def update(self, strains):
        eps = strains.reshape(-1, self.qdim)
        eeq, deeq = modified_mises_strain_norm(self, eps)
        self.kappa, dkappa = self.kappa_kkt(eeq)

@dataclass
class GradientDamage(LocalDamage):
    l: float = 2.

    def evaluate(self, strains, e_):
        eps = strains.reshape(-1, self.qdim)
        e = e_.flatten()
        
        kappa = self.kappa_kkt(e)
        dkappa_de = (e >= kappa).astype(int)
        
        w, dw = self.dmg(self, kappa)
    
        C = self.C
        self.sigma = eps @ C * (1 - w)[:, None]

        self.dsigma_deps = np.tile(C.flatten(), (len(kappa), 1)) * (1 - w)[:, None]
        self.dsigma_de = -eps @ C * dw[:, None] * dkappa_de[:, None]
        self.eeq, self.deeq = modified_mises_strain_norm(self, eps)


    def kappa_kkt(self, e):
        if self.kappa is None:
            self.kappa = self.ft / self.E

        return np.maximum(e, self.kappa)

    def update(self, e):
        self.kappa = self.kappa_kkt(e)



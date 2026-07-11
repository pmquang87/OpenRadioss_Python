"""
LAW1 — linear elastic material (/MAT/LAW1, /MAT/ELAST).

Fortran origin: ``engine/source/materials/mat/mat001/sigeps01.F`` (solids)
and ``sigeps01c.F`` (shells, the 'c' suffix means 'coque' = shell).

Theory
------
Hypoelastic isotropic law: the *rate* of Cauchy stress is linear in the
rate of deformation D,

    dsigma/dt = lambda * tr(D) * I  +  2 * mu * D

integrated over the time step with the strain increment
``deps = D * dt`` (the objective Jaumann rotation of the pre-existing
stress is applied by the element kernel before this call). For the small
strain increments of an explicit code this is the standard formulation —
it is NOT hyperelastic, so a closed strain cycle at large strain does not
return exactly to zero stress; identical behaviour to the original.

Voigt convention (see package docstring): engineering shear increments,
i.e. deps[:, 3] = 2*eps_xy*dt = gamma_xy increment, so the shear update is
simply  dsig_xy = G * dgamma_xy.
"""

from __future__ import annotations

import numpy as np


def solid_update(mat, sig: np.ndarray, deps: np.ndarray) -> np.ndarray:
    """3-D stress update. sig, deps: (n, 6) Voigt arrays. In-place on sig."""
    G = mat.G
    lam = mat.K - 2.0 * G / 3.0        # Lamé lambda = K - 2G/3
    tr = deps[:, 0] + deps[:, 1] + deps[:, 2]
    sig[:, 0] += lam * tr + 2.0 * G * deps[:, 0]
    sig[:, 1] += lam * tr + 2.0 * G * deps[:, 1]
    sig[:, 2] += lam * tr + 2.0 * G * deps[:, 2]
    sig[:, 3:] += G * deps[:, 3:]      # engineering shear: tau = G*gamma
    return sig


def shell_update(mat, sig: np.ndarray, deps: np.ndarray) -> np.ndarray:
    """Plane-stress update. sig, deps: (n, 3) = [xx, yy, xy].

    Plane stress (sigma_zz = 0) condenses Hooke's law to

        dsig_xx = E/(1-nu^2) * (deps_xx + nu * deps_yy)
        dsig_yy = E/(1-nu^2) * (deps_yy + nu * deps_xx)
        dsig_xy = G * dgamma_xy
    """
    E, nu, G = mat.E, mat.nu, mat.G
    c = E / (1.0 - nu * nu)
    dxx, dyy = deps[:, 0].copy(), deps[:, 1].copy()
    sig[:, 0] += c * (dxx + nu * dyy)
    sig[:, 1] += c * (dyy + nu * dxx)
    sig[:, 2] += G * deps[:, 2]
    return sig


# ----------------------------------------------------------------------------
# Consistent tangents for the implicit solver (M8)
# ----------------------------------------------------------------------------
# For a linear-elastic law the "consistent (algorithmic) tangent" is trivially
# the constant elasticity matrix C itself — the stress is a linear function of
# strain, so dsigma/deps = C exactly. These builders return C in the SAME
# Voigt / engineering-shear convention the element B-matrices use (shear
# strains are engineering, gamma = 2*eps, so the shear rows carry G directly,
# consistent with the rate updates above and with solid_hexa8._exact_dt_factor).


def solid_tangent(mat) -> np.ndarray:
    """(6, 6) isotropic elastic tangent C for solids (Voigt, engineering
    shear: rows/cols [xx, yy, zz, xy, yz, zx]).

    C = lambda*(1 (x) 1) + 2G on the deviatoric/shear structure, i.e. the
    standard Lame form with G on the engineering-shear diagonal."""
    G = mat.G
    lam = mat.K - 2.0 * G / 3.0
    C = np.array([
        [lam + 2 * G, lam, lam, 0, 0, 0],
        [lam, lam + 2 * G, lam, 0, 0, 0],
        [lam, lam, lam + 2 * G, 0, 0, 0],
        [0, 0, 0, G, 0, 0],
        [0, 0, 0, 0, G, 0],
        [0, 0, 0, 0, 0, G],
    ], dtype=float)
    return C


def shell_membrane_tangent(mat) -> np.ndarray:
    """(3, 3) plane-stress elastic tangent for shells (Voigt [xx, yy, xy],
    engineering shear). This is the membrane AND (scaled by z^2) the bending
    constitutive matrix of the Belytschko-Tsay resultant formulation."""
    E, nu, G = mat.E, mat.nu, mat.G
    Ep = E / (1.0 - nu * nu)
    return np.array([[Ep, nu * Ep, 0.0],
                     [nu * Ep, Ep, 0.0],
                     [0.0, 0.0, G]], dtype=float)

"""
Tests for Milestone M566: /MAT/LAW92 (/MAT/ARRUDA_BOYCE) Arruda-Boyce Hyperelastic Model
Fortran Reference Parity Test vs sigeps92.F.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law92_arruda_boyce import (
    C_LANGEVIN,
    initial_shear_modulus,
    solid_update,
)


def fortran_sigeps92_oracle(
    eps_6: np.ndarray,
    mu: float,
    d: float,
    lam: float,
    rho: float,
    nu: float = 0.495,
) -> tuple[np.ndarray, float, float]:
    """Pure Python bit-faithful reference oracle replicating engine/source/materials/mat/mat092/sigeps92.F.

    Returns (sig_cauchy_6, sound_speed, w_mullins).
    """
    beta = 1.0 / (lam * lam) if lam > 0.0 else 1.0 / 49.0
    c = C_LANGEVIN

    # Initial shear modulus
    poly = (
        1.0
        + 0.6 * beta
        + (99.0 / 175.0) * (beta ** 2)
        + (513.0 / 875.0) * (beta ** 3)
        + (42039.0 / 67375.0) * (beta ** 4)
    )
    g = mu * poly

    if d > 0.0:
        rbulk = 2.0 / d
    else:
        rbulk = (2.0 / 3.0) * (1.0 + nu) * g / (1.0 - 2.0 * nu)

    # 1. Spectral decomposition of strain tensor
    eps_mat = np.array([
        [eps_6[0], 0.5 * eps_6[3], 0.5 * eps_6[5]],
        [0.5 * eps_6[3], eps_6[1], 0.5 * eps_6[4]],
        [0.5 * eps_6[5], 0.5 * eps_6[4], eps_6[2]],
    ], dtype=np.float64)

    evv, dirprv = np.linalg.eigh(eps_mat)

    # 2. Logarithmic stretch (ISMSTR = 0)
    ev = np.exp(evv)

    # 3. Relative volume RV = J = det(F)
    rv = ev[0] * ev[1] * ev[2]
    rv_1 = 1.0 / rv

    # 4. Isochoric stretch EVD = EV * RV^(-1/3)
    rvd = math.exp((-1.0 / 3.0) * math.log(rv))
    evd = ev * rvd
    trace = np.sum(evd ** 2)

    di1lam = np.zeros(3, dtype=np.float64)
    for k in range(3):
        di1lam[k] = 2.0 * (evd[k] ** 2) - (2.0 / 3.0) * trace

    t = np.zeros(3, dtype=np.float64)
    clam = np.zeros(5, dtype=np.float64)
    for j_idx in range(1, 6):
        bb = 1.0 / (lam ** (2 * j_idx - 2))
        aa = j_idx * c[j_idx - 1]
        clam[j_idx - 1] = bb * c[j_idx - 1]
        cc = aa * bb * (trace ** (j_idx - 1))
        for k in range(3):
            t[k] += cc * di1lam[k]

    # Pressure from volumetric compressibility
    p = rv_1 * rbulk * (rv ** 2 - 1.0) / 2.0
    for k in range(3):
        t[k] = (mu * t[k] + p) * rv_1

    # Mullins energy
    w_mullins = mu * (
        c[0] * (trace - 3.0)
        + c[1] * beta * (trace ** 2 - 9.0)
        + c[2] * (beta ** 2) * (trace ** 3 - 27.0)
        + c[3] * (beta ** 3) * (trace ** 4 - 81.0)
        + c[4] * (beta ** 4) * (trace ** 5 - 243.0)
    )

    # Tangent GTMAX and RKMAX
    gtmax = g
    rkmax = rbulk / 2.0
    cii = np.zeros(3, dtype=np.float64)
    for ii in range(1, 6):
        clp = 4.0 * ii * clam[ii - 1]
        lam_2 = evd ** 2
        lam_4 = lam_2 ** 2
        aa_term = (1.0 / 9.0) * ii * (trace ** ii)
        bb_term = (1.0 / 3.0) * (3.0 - ii) * (trace ** (ii - 1)) if ii > 1 else 0.0
        cc_term = (ii - 1) * (trace ** (ii - 2)) if ii > 2 else 0.0
        for k in range(3):
            cii[k] += clp * (aa_term + bb_term * lam_2[k] + cc_term * lam_4[k])

    amax = np.max(cii)
    eti = max(1.0, amax * 0.81)
    gtmax = g * eti
    rkmax = rkmax * (1.0 + rv_1 * rv_1)
    rkmax = max(rbulk, rkmax)

    sound_speed = math.sqrt(((4.0 / 3.0) * gtmax + rkmax) / rho)

    # Rotate principal stresses back to global coordinate system
    sig_mat = np.zeros((3, 3), dtype=np.float64)
    for k in range(3):
        v = dirprv[:, k]
        sig_mat += t[k] * np.outer(v, v)

    sig_6 = np.array([
        sig_mat[0, 0],
        sig_mat[1, 1],
        sig_mat[2, 2],
        sig_mat[0, 1],
        sig_mat[1, 2],
        sig_mat[0, 2],
    ], dtype=np.float64)

    return sig_6, sound_speed, w_mullins


# ============================================================================
# Parity Test Cases
# ============================================================================

@pytest.mark.parametrize("lam_limit", [4.0, 6.0, 7.5])
@pytest.mark.parametrize("d_val", [1e-8, 5e-7])
def test_fortran_parity_uniaxial_tension(lam_limit, d_val):
    """Verify bit-level parity for uniaxial extension."""
    mu = 1.2e6
    rho = 1050.0
    eps = np.array([0.25, -0.10, -0.10, 0.0, 0.0, 0.0], dtype=np.float64)

    sig_oracle, c_oracle, w_oracle = fortran_sigeps92_oracle(
        eps, mu=mu, d=d_val, lam=lam_limit, rho=rho
    )
    sig_py, hist_py, c_py = solid_update(
        eps, mu=mu, d=d_val, lam=lam_limit, rho=rho
    )

    np.testing.assert_allclose(sig_py, sig_oracle, rtol=1e-10, atol=1e-10)
    assert math.isclose(c_py, c_oracle, rel_tol=1e-10)
    assert math.isclose(hist_py["w_mullins"], w_oracle, rel_tol=1e-10)


def test_fortran_parity_pure_shear():
    """Verify bit-level parity for pure shear deformation."""
    mu = 2.0e6
    lam = 5.0
    d = 2.0e-8
    rho = 1100.0
    eps = np.array([0.0, 0.0, 0.0, 0.15, 0.0, 0.0], dtype=np.float64)

    sig_oracle, c_oracle, w_oracle = fortran_sigeps92_oracle(
        eps, mu=mu, d=d, lam=lam, rho=rho
    )
    sig_py, hist_py, c_py = solid_update(
        eps, mu=mu, d=d, lam=lam, rho=rho
    )

    np.testing.assert_allclose(sig_py, sig_oracle, rtol=1e-10, atol=1e-10)
    assert math.isclose(c_py, c_oracle, rel_tol=1e-10)
    assert math.isclose(hist_py["w_mullins"], w_oracle, rel_tol=1e-10)


def test_fortran_parity_general_3d_triaxial():
    """Verify bit-level parity for arbitrary 3D multi-axial strain state."""
    mu = 8.5e5
    lam = 6.2
    d = 3.3e-8
    rho = 980.0
    eps = np.array([0.12, -0.07, 0.03, 0.05, -0.04, 0.06], dtype=np.float64)

    sig_oracle, c_oracle, w_oracle = fortran_sigeps92_oracle(
        eps, mu=mu, d=d, lam=lam, rho=rho
    )
    sig_py, hist_py, c_py = solid_update(
        eps, mu=mu, d=d, lam=lam, rho=rho
    )

    np.testing.assert_allclose(sig_py, sig_oracle, rtol=1e-10, atol=1e-10)
    assert math.isclose(c_py, c_oracle, rel_tol=1e-10)
    assert math.isclose(hist_py["w_mullins"], w_oracle, rel_tol=1e-10)


def test_fortran_parity_hydrostatic_compression():
    """Verify bit-level parity under extreme volumetric compression."""
    mu = 1.0e6
    lam = 5.0
    d = 1.0e-8
    rho = 1200.0
    eps = np.array([-0.05, -0.05, -0.05, 0.0, 0.0, 0.0], dtype=np.float64)

    sig_oracle, c_oracle, w_oracle = fortran_sigeps92_oracle(
        eps, mu=mu, d=d, lam=lam, rho=rho
    )
    sig_py, hist_py, c_py = solid_update(
        eps, mu=mu, d=d, lam=lam, rho=rho
    )

    np.testing.assert_allclose(sig_py, sig_oracle, rtol=1e-10, atol=1e-10)
    assert math.isclose(c_py, c_oracle, rel_tol=1e-10)

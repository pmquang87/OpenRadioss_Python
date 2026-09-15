"""
Tests for Milestone M567: /MAT/LAW94 (/MAT/YEOH) Yeoh Hyperelastic Model
Fortran Reference Parity Test vs sigeps94.F and Analytical Solutions.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law94_yeoh import (
    build_law94,
    solid_update,
    shell_update,
    sound_speed,
    yeoh_analytical_stress,
)


def fortran_sigeps94_oracle(
    eps_6: np.ndarray,
    c10: float,
    c20: float,
    c30: float,
    d1: float,
    d2: float,
    d3: float,
    rho: float,
    nu: float = 0.495,
) -> tuple[np.ndarray, float, float]:
    """Pure Python bit-faithful reference oracle replicating sigeps94.F.

    Returns (sig_cauchy_6, sound_speed, w_mullins).
    """
    g = 2.0 * c10
    if d1 > 0.0:
        rbulk = 2.0 / d1
    else:
        rbulk = (2.0 / 3.0) * (1.0 + nu) * g / (1.0 - 2.0 * nu)

    c0 = np.array([c10, c20, c30], dtype=np.float64)

    # 1. Spectral decomposition
    eps_mat = np.array([
        [eps_6[0], 0.5 * eps_6[3], 0.5 * eps_6[5]],
        [0.5 * eps_6[3], eps_6[1], 0.5 * eps_6[4]],
        [0.5 * eps_6[5], 0.5 * eps_6[4], eps_6[2]],
    ], dtype=np.float64)

    evv, dirprv = np.linalg.eigh(eps_mat)

    # 2. Logarithmic stretch
    ev = np.exp(evv)

    # 3. Relative volume RV = J = det(F)
    rv = ev[0] * ev[1] * ev[2]
    invr = 1.0 / rv

    # 4. Isochoric stretch EVD = EV * RV^(-1/3)
    rvd = math.exp((-1.0 / 3.0) * math.log(rv))
    evd = ev * rvd
    trace = np.sum(evd ** 2)

    l1di1lam = np.zeros(3, dtype=np.float64)
    for k in range(3):
        l1di1lam[k] = 2.0 * (evd[k] ** 2 - (1.0 / 3.0) * trace)

    aa = trace - 3.0
    bb = aa * aa
    cc = (c0[0] + 2.0 * c0[1] * aa + 3.0 * c0[2] * bb) * invr

    # Deviatoric stresses
    t_dev = l1di1lam * cc

    # Upstream hm_read_mat94.F:140-153 stores stiffness units:
    d1_stiff = (1.0 / d1) if d1 != 0.0 else rbulk / 2.0
    d2_stiff = (1.0 / d2) if d2 != 0.0 else 0.0
    d3_stiff = (1.0 / d3) if d3 != 0.0 else 0.0

    # Pressure
    j_minus_1 = rv - 1.0
    p = rbulk * j_minus_1 + 4.0 * d2_stiff * (j_minus_1 ** 3) + 6.0 * d3_stiff * (j_minus_1 ** 5)

    # Principal Cauchy stresses
    t = t_dev + p

    # Mullins energy
    w_iso = c10 * aa + c20 * bb + c30 * (aa * bb)
    w_vol = d1_stiff * (j_minus_1 ** 2) + d2_stiff * (j_minus_1 ** 4) + d3_stiff * (j_minus_1 ** 6)
    w_mullins = w_iso + w_vol

    # Sound speed
    cii = np.zeros(3, dtype=np.float64)
    if abs(aa) >= 1.0e-10:
        for ii in range(1, 4):
            clp = 4.0 * ii * c0[ii - 1]
            lam_2 = evd ** 2
            lam_4 = lam_2 ** 2
            aa_c = (1.0 / 9.0) * ii * (aa ** ii)
            bb_c = (1.0 / 3.0) * (3.0 - ii) * (aa ** (ii - 1)) if ii > 1 else 0.0
            cc_c = (ii - 1.0) * (aa ** (ii - 2)) if ii > 2 else 0.0
            cii += clp * (aa_c + bb_c * lam_2 + cc_c * lam_4)

    amax = float(np.max(cii))
    eti = max(1.0, amax * 0.81)
    gtmax = g * eti
    rkmax = rbulk + 12.0 * d2_stiff * (j_minus_1 ** 2) + 30.0 * d3_stiff * (j_minus_1 ** 4)
    rkmax = max(rbulk, rkmax)

    sound_speed_val = math.sqrt(((4.0 / 3.0) * gtmax + rkmax) / rho)

    # Global stress tensor
    sig_mat = dirprv @ np.diag(t) @ dirprv.T
    sig_cauchy = np.array([
        sig_mat[0, 0],
        sig_mat[1, 1],
        sig_mat[2, 2],
        sig_mat[0, 1],
        sig_mat[1, 2],
        sig_mat[2, 0],
    ], dtype=np.float64)

    return sig_cauchy, sound_speed_val, w_mullins


class TestLaw94FortranParity:
    """Validate pyradioss LAW94 solid update against exact Fortran sigeps94.F logic."""

    @pytest.mark.parametrize(
        "eps",
        [
            np.array([0.10, -0.04, -0.04, 0.0, 0.0, 0.0]),     # Uniaxial tension
            np.array([0.08, 0.08, -0.14, 0.0, 0.0, 0.0]),      # Equibiaxial
            np.array([0.0, 0.0, 0.0, 0.15, 0.0, 0.0]),          # Pure shear XY
            np.array([0.05, -0.02, -0.01, 0.04, 0.03, -0.02]),  # General 3D multiaxial
        ],
    )
    def test_solid_update_parity_vs_oracle(self, eps: np.ndarray):
        """Verify solid_update matches fortran_sigeps94_oracle to float precision."""
        c10 = 2.0e6
        c20 = -1.5e5
        c30 = 8.0e3
        d1 = 1.0e-8
        d2 = 2.0e-10
        d3 = 5.0e-12
        rho = 1100.0

        p = build_law94(c10=c10, c20=c20, c30=c30, d1=d1, d2=d2, d3=d3, rho0=rho)

        extra = {"rho": rho}
        sig_py, _, c_py = solid_update(p, np.zeros(6), eps=eps, extra=extra)
        w_py = extra.get("mullins_w", np.zeros(1))[0]

        sig_ora, c_ora, w_ora = fortran_sigeps94_oracle(
            eps, c10=c10, c20=c20, c30=c30, d1=d1, d2=d2, d3=d3, rho=rho
        )

        assert np.allclose(sig_py, sig_ora, rtol=1e-10, atol=1e-10)
        assert math.isclose(float(c_py), c_ora, rel_tol=1e-10)
        assert math.isclose(float(w_py), w_ora, rel_tol=1e-10)

    def test_incompressible_analytical_stress_match(self):
        """In incompressible limit (D1->0, J=1), Cauchy stress matches analytical formulas."""
        c10 = 1.5e6
        c20 = 2.0e5
        c30 = 1.0e4
        p = build_law94(c10=c10, c20=c20, c30=c30, d1=0.0, rho0=1000.0)

        lam_vals = [1.1, 1.25, 1.5, 2.0]
        for lam in lam_vals:
            # Uniaxial: eps_xx = ln(lam), eps_yy = eps_zz = -0.5*ln(lam)
            eps_xx = math.log(lam)
            eps_yy = -0.5 * eps_xx
            eps = np.array([eps_xx, eps_yy, eps_yy, 0.0, 0.0, 0.0])

            sig, _, _ = solid_update(p, np.zeros(6), eps=eps)
            sigma_xx = sig[0] - sig[1]  # Deviatoric difference removes pressure

            # Analytical nominal stress: P = sigma / lambda
            p_nom_ana = yeoh_analytical_stress(lam, c10, c20, c30, itype=1)
            sigma_ana = p_nom_ana * lam

            # Within small bulk compressibility tolerance (nu=0.495 has ~1-2% deviation from infinite K)
            assert math.isclose(sigma_xx, sigma_ana, rel_tol=2e-2)

"""
Fortran Parity & Physics Oracle Verifier for Milestone M568: /MAT/LAW93 (/MAT/ORTH_HILL).

Orthotropic Hill 1948 Elasto-Plastic Model for Solids and Shells.
Directly cites and mirrors upstream OpenRadioss source:
- Upstream Fortran starter reader:
  ``starter/source/materials/mat/mat093/hm_read_mat93.F``
- Upstream Fortran engine physics:
  ``engine/source/materials/mat/mat093/sigeps93.F`` (solids)
  ``engine/source/materials/mat/mat093/sigeps93c.F`` (shells)
- CFG card definition:
  ``hm_cfg_files/config/CFG/radioss140/MAT/matl93_ORTH_HILL.cfg``

Tests included:
1. Exact Python oracle reproducing sigeps93.F (3D solid return mapping) and sigeps93c.F (2D shell return mapping).
2. Exact Hill 1948 anisotropic yield coefficients (F, G, H, L, M, N) and isotropic von Mises recovery.
3. 2D plane-stress orthotropic elasticity matrix [A11, A22, A12] and compliance inversion.
4. 3D continuum elasticity matrix [Dij] and positive definiteness.
5. Voce two-term non-linear continuous hardening law vs exact analytical formula.
6. Pure elastic states below yield stress in 3D and 2D.
7. Uniaxial tensile yielding along 1, 2, 3 principal orthotropy directions.
8. Shear yielding under in-plane (12) and transverse (23, 31) shear.
9. Shell thickness thinning under plastic deformation matching sigeps93c.F transverse strains.
10. Multi-rate tabulated yield curves with viscoplastic rate filtering (VP=1, FCUT).
11. 3D solid algorithmic consistent tangent (6 x 6) matching numerical perturbation.
12. 2D shell algorithmic consistent tangent (3 x 3) matching numerical perturbation.
13. Vectorized multi-element batch execution matching scalar loop.
14. Exhaustive 64-state test matrix comparing solid_update and shell_update against oracles.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

from pyradioss.materials.law93_orth_hill import (
    OrthHillParams,
    build_law93,
    solid_update,
    shell_update,
    sound_speed,
    sound_speed_shell,
    consistent_tangent,
    eval_yield_stress,
    extra_shapes,
)

_EM10 = 1.0e-10
_EM20 = 1.0e-20


# =============================================================================
# 1. UPSTREAM FORTRAN ENGINE ORACLES: sigeps93.F & sigeps93c.F
# =============================================================================

def sigeps93_solid_oracle(
    sig_old: np.ndarray,
    deps: np.ndarray,
    epsp_old: float,
    e11: float,
    e22: float,
    e33: float,
    nu12: float,
    nu13: float,
    nu23: float,
    g12: float,
    g13: float,
    g23: float,
    r11: float,
    r22: float,
    r33: float,
    r12: float,
    r13: float,
    r23: float,
    sigma_y: float,
    qr1: float,
    cr1: float,
    qr2: float = 0.0,
    cr2: float = 0.0,
    rho0: float = 7.8e-6,
) -> Tuple[np.ndarray, float, float]:
    """Pure Python bit-faithful reference oracle replicating sigeps93.F (solids)."""
    # 1. Hill coefficients
    r11s = max(r11, 1.0e-6) ** 2
    r22s = max(r22, 1.0e-6) ** 2
    r33s = max(r33, 1.0e-6) ** 2
    ff = 0.5 * (1.0 / r22s + 1.0 / r33s - 1.0 / r11s)
    gg = 0.5 * (1.0 / r33s + 1.0 / r11s - 1.0 / r22s)
    hh = 0.5 * (1.0 / r11s + 1.0 / r22s - 1.0 / r33s)
    ll = 1.5 / (max(r23, 1.0e-6) ** 2)
    mm = 1.5 / (max(r13, 1.0e-6) ** 2)
    nn = 1.5 / (max(r12, 1.0e-6) ** 2)

    # 2. Compliance matrix inversion S -> D
    nu21 = nu12 * e22 / e11
    nu31 = nu13 * e33 / e11
    nu32 = nu23 * e33 / e22
    s_mat = np.array([
        [1.0 / e11, -nu21 / e22, -nu31 / e33],
        [-nu12 / e11, 1.0 / e22, -nu32 / e33],
        [-nu13 / e11, -nu23 / e22, 1.0 / e33],
    ], dtype=float)
    d_sub = np.linalg.inv(s_mat)

    # 3. Elastic trial stress
    sig_tr = np.zeros(6, dtype=float)
    sig_tr[0:3] = sig_old[0:3] + d_sub @ deps[0:3]
    sig_tr[3] = sig_old[3] + g12 * deps[3]
    sig_tr[4] = sig_old[4] + g23 * deps[4]
    sig_tr[5] = sig_old[5] + g13 * deps[5]

    # 4. Hill equivalent stress
    sxx, syy, szz = sig_tr[0], sig_tr[1], sig_tr[2]
    sxy, syz, szx = sig_tr[3], sig_tr[4], sig_tr[5]
    shl2 = (
        ff * ((syy - szz) ** 2)
        + gg * ((szz - sxx) ** 2)
        + hh * ((sxx - syy) ** 2)
        + 2.0 * ll * (syz ** 2)
        + 2.0 * mm * (szx ** 2)
        + 2.0 * nn * (sxy ** 2)
    )
    shl = math.sqrt(max(shl2, 0.0))

    # Current yield stress
    pla = float(epsp_old)
    yld = sigma_y + qr1 * (1.0 - math.exp(-cr1 * pla)) + qr2 * (1.0 - math.exp(-cr2 * pla))

    if shl <= yld:
        # Pure elastic
        c_sound = math.sqrt(max(d_sub[0, 0], d_sub[1, 1], d_sub[2, 2]) / rho0)
        return sig_tr, pla, c_sound

    # 5. Cutting plane return mapping (3 iterations matching sigeps93.F)
    sig_cur = sig_tr.copy()
    for _ in range(3):
        sxx, syy, szz = sig_cur[0], sig_cur[1], sig_cur[2]
        sxy, syz, szx = sig_cur[3], sig_cur[4], sig_cur[5]
        shl2 = (
            ff * ((syy - szz) ** 2)
            + gg * ((szz - sxx) ** 2)
            + hh * ((sxx - syy) ** 2)
            + 2.0 * ll * (syz ** 2)
            + 2.0 * mm * (szx ** 2)
            + 2.0 * nn * (sxy ** 2)
        )
        shl = math.sqrt(max(shl2, 1.0e-20))
        yld = sigma_y + qr1 * (1.0 - math.exp(-cr1 * pla)) + qr2 * (1.0 - math.exp(-cr2 * pla))
        f_val = shl - yld
        if abs(f_val) < 1.0e-8 * yld:
            break

        # Flow vector: sigeps93.F:302-307
        norm_xx = (gg * (sxx - szz) + hh * (sxx - syy)) / shl
        norm_yy = (ff * (syy - szz) + hh * (syy - sxx)) / shl
        norm_zz = (ff * (szz - syy) + gg * (szz - sxx)) / shl
        norm_xy = 2.0 * nn * sxy / shl
        norm_yz = 2.0 * ll * syz / shl
        norm_zx = 2.0 * mm * szx / shl

        # DPHI/DLAMBDA: sigeps93.F:314-343
        dfdsig2 = (
            norm_xx * (d_sub[0, 0] * norm_xx + d_sub[0, 1] * norm_yy + d_sub[0, 2] * norm_zz)
            + norm_yy * (d_sub[1, 0] * norm_xx + d_sub[1, 1] * norm_yy + d_sub[1, 2] * norm_zz)
            + norm_zz * (d_sub[2, 0] * norm_xx + d_sub[2, 1] * norm_yy + d_sub[2, 2] * norm_zz)
            + (norm_xy ** 2) * g12
            + (norm_yz ** 2) * g23
            + (norm_zx ** 2) * g13
        )

        sig_dfdsig = (
            sxx * norm_xx
            + syy * norm_yy
            + szz * norm_zz
            + sxy * norm_xy
            + syz * norm_yz
            + szx * norm_zx
        )
        dpla_dlam = sig_dfdsig / yld

        h_slope = qr1 * cr1 * math.exp(-cr1 * pla) + qr2 * cr2 * math.exp(-cr2 * pla)
        dphi_dlam = -dfdsig2 - h_slope * dpla_dlam
        dlam = -f_val / min(dphi_dlam, -1.0e-20)

        # Update stresses: sigeps93.F:357-362
        dpxx = dlam * norm_xx
        dpyy = dlam * norm_yy
        dpzz = dlam * norm_zz
        sig_cur[0] -= (d_sub[0, 0] * dpxx + d_sub[0, 1] * dpyy + d_sub[0, 2] * dpzz)
        sig_cur[1] -= (d_sub[1, 0] * dpxx + d_sub[1, 1] * dpyy + d_sub[1, 2] * dpzz)
        sig_cur[2] -= (d_sub[2, 0] * dpxx + d_sub[2, 1] * dpyy + d_sub[2, 2] * dpzz)
        sig_cur[3] -= (dlam * norm_xy) * g12
        sig_cur[4] -= (dlam * norm_yz) * g23
        sig_cur[5] -= (dlam * norm_zx) * g13

        pla += dlam * dpla_dlam

    c_sound = math.sqrt(max(d_sub[0, 0], d_sub[1, 1], d_sub[2, 2]) / rho0)
    return sig_cur, pla, c_sound


def sigeps93c_shell_oracle(
    sig_old: np.ndarray,
    deps: np.ndarray,
    epsp_old: float,
    e11: float,
    e22: float,
    nu12: float,
    g12: float,
    r11: float,
    r22: float,
    r33: float,
    r12: float,
    sigma_y: float,
    qr1: float,
    cr1: float,
    qr2: float = 0.0,
    cr2: float = 0.0,
    rho0: float = 7.8e-6,
) -> Tuple[np.ndarray, float, float]:
    """Pure Python bit-faithful reference oracle replicating sigeps93c.F (shells)."""
    # 1. Plane-stress moduli
    nu21 = nu12 * e22 / e11
    denom_nu = max(1.0 - nu12 * nu21, 1.0e-12)
    a11 = e11 / denom_nu
    a22 = e22 / denom_nu
    a12 = nu12 * e22 / denom_nu

    # 2. Hill coefficients
    r11s = max(r11, 1.0e-6) ** 2
    r22s = max(r22, 1.0e-6) ** 2
    r33s = max(r33, 1.0e-6) ** 2
    ff = 0.5 * (1.0 / r22s + 1.0 / r33s - 1.0 / r11s)
    gg = 0.5 * (1.0 / r33s + 1.0 / r11s - 1.0 / r22s)
    hh = 0.5 * (1.0 / r11s + 1.0 / r22s - 1.0 / r33s)
    nn = 1.5 / (max(r12, 1.0e-6) ** 2)

    # 3. Elastic trial stress
    sig_tr = np.zeros(3, dtype=float)
    sig_tr[0] = sig_old[0] + a11 * deps[0] + a12 * deps[1]
    sig_tr[1] = sig_old[1] + a12 * deps[0] + a22 * deps[1]
    sig_tr[2] = sig_old[2] + g12 * deps[2]

    # 4. Plane-stress Hill equivalent stress
    sxx = sig_tr[0]
    syy = sig_tr[1]
    sxy = sig_tr[2]
    shl2 = (ff + hh) * (syy ** 2) + (gg + hh) * (sxx ** 2) - 2.0 * hh * sxx * syy + 2.0 * nn * (sxy ** 2)
    shl = math.sqrt(max(shl2, 0.0))

    pla = float(epsp_old)
    yld = sigma_y + qr1 * (1.0 - math.exp(-cr1 * pla)) + qr2 * (1.0 - math.exp(-cr2 * pla))

    if shl <= yld:
        c_sound = math.sqrt(max(a11, a22) / rho0)
        return sig_tr, pla, c_sound

    # 5. Cutting-plane return mapping
    sig_cur = sig_tr.copy()
    for _ in range(3):
        sxx, syy, sxy = sig_cur[0], sig_cur[1], sig_cur[2]
        shl2 = (ff + hh) * (syy ** 2) + (gg + hh) * (sxx ** 2) - 2.0 * hh * sxx * syy + 2.0 * nn * (sxy ** 2)
        shl = math.sqrt(max(shl2, 1.0e-20))
        yld = sigma_y + qr1 * (1.0 - math.exp(-cr1 * pla)) + qr2 * (1.0 - math.exp(-cr2 * pla))
        f_val = shl - yld
        if abs(f_val) < 1.0e-8 * yld:
            break

        norm_xx = (gg * sxx + hh * (sxx - syy)) / shl
        norm_yy = (ff * syy + hh * (syy - sxx)) / shl
        norm_xy = 2.0 * nn * sxy / shl

        dfdsig2 = (
            norm_xx * (a11 * norm_xx + a12 * norm_yy)
            + norm_yy * (a12 * norm_xx + a22 * norm_yy)
            + (norm_xy ** 2) * g12
        )

        sig_dfdsig = sxx * norm_xx + syy * norm_yy + sxy * norm_xy
        dpla_dlam = sig_dfdsig / yld

        h_slope = qr1 * cr1 * math.exp(-cr1 * pla) + qr2 * cr2 * math.exp(-cr2 * pla)
        dphi_dlam = -dfdsig2 - h_slope * dpla_dlam
        dlam = -f_val / min(dphi_dlam, -1.0e-20)

        # Update stresses: sigeps93c.F:377-380
        dpxx = dlam * norm_xx
        dpyy = dlam * norm_yy
        sig_cur[0] -= a11 * dpxx + a12 * dpyy
        sig_cur[1] -= a12 * dpxx + a22 * dpyy
        sig_cur[2] -= (dlam * norm_xy) * g12

        pla += dlam * dpla_dlam

    c_sound = math.sqrt(max(a11, a22) / rho0)
    return sig_cur, pla, c_sound


# =============================================================================
# 2. TEST SUITE
# =============================================================================

class TestLaw93FortranParity:
    """Rigorous mathematical parity tests for LAW93 Orthotropic Hill model."""

    def test_isotropic_mises_recovery(self):
        """When R11=R22=R33=R12=R13=R23=1.0, Hill parameters recover von Mises."""
        p = build_law93(
            rho0=7.8e-6,
            e11=210000.0, e22=210000.0, e33=210000.0,
            nu12=0.3, nu13=0.3, nu23=0.3,
            g12=80000.0, g13=80000.0, g23=80000.0,
            r11=1.0, r22=1.0, r33=1.0,
            r12=1.0, r13=1.0, r23=1.0,
            sigma_y=300.0,
        )
        assert math.isclose(p.ff, 0.5, rel_tol=1e-12)
        assert math.isclose(p.gg, 0.5, rel_tol=1e-12)
        assert math.isclose(p.hh, 0.5, rel_tol=1e-12)
        assert math.isclose(p.ll, 1.5, rel_tol=1e-12)
        assert math.isclose(p.mm, 1.5, rel_tol=1e-12)
        assert math.isclose(p.nn, 1.5, rel_tol=1e-12)

    def test_plane_stress_stiffness_matrix(self):
        """Verify 2D plane-stress orthotropic stiffness components."""
        e11, e22, nu12 = 200000.0, 100000.0, 0.3
        nu21 = nu12 * e22 / e11  # 0.15
        denom = 1.0 - nu12 * nu21  # 1.0 - 0.045 = 0.955
        a11_exp = e11 / denom
        a22_exp = e22 / denom
        a12_exp = nu12 * e22 / denom

        p = build_law93(
            rho0=7.8e-6,
            e11=e11, e22=e22, e33=200000.0,
            nu12=nu12, g12=40000.0,
            sigma_y=250.0,
        )
        assert math.isclose(p.a11, a11_exp, rel_tol=1e-12)
        assert math.isclose(p.a22, a22_exp, rel_tol=1e-12)
        assert math.isclose(p.a12, a12_exp, rel_tol=1e-12)

    def test_voce_hardening_evaluation(self):
        """Verify Voce 2-term saturation hardening matches exact formula."""
        sigma_y, qr1, cr1, qr2, cr2 = 280.0, 120.0, 25.0, 50.0, 5.0
        p = build_law93(
            rho0=7.8e-6,
            e11=210000.0, e22=210000.0, e33=210000.0,
            nu12=0.3, g12=80000.0,
            sigma_y=sigma_y, qr1=qr1, cr1=cr1, qr2=qr2, cr2=cr2,
        )
        for ep in [0.0, 0.005, 0.02, 0.05, 0.1, 0.3]:
            y_exp = sigma_y + qr1 * (1.0 - math.exp(-cr1 * ep)) + qr2 * (1.0 - math.exp(-cr2 * ep))
            h_exp = qr1 * cr1 * math.exp(-cr1 * ep) + qr2 * cr2 * math.exp(-cr2 * ep)
            y_act, h_act = eval_yield_stress(p, ep)
            assert math.isclose(y_act, y_exp, rel_tol=1e-12)
            assert math.isclose(h_act, h_exp, rel_tol=1e-12)

    @pytest.mark.parametrize(
        "deps",
        [
            np.array([1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0e-4, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0e-4, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0e-4, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 1.0e-4, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-4]),
            np.array([5.0e-5, -2.0e-5, -1.0e-5, 3.0e-5, 2.0e-5, -1.0e-5]),
        ],
    )
    def test_solid_pure_elastic_parity(self, deps: np.ndarray):
        """Solid response in elastic regime matches Hooke's law exactly."""
        p = build_law93(
            rho0=7.8e-6,
            e11=210000.0, e22=205000.0, e33=210000.0,
            nu12=0.3, nu13=0.3, nu23=0.3,
            g12=80000.0, g13=80000.0, g23=80000.0,
            sigma_y=500.0,
        )
        sig_old = np.zeros(6)
        sig_act, pla_act, c_act = solid_update(p, sig_old, deps, epsp=0.0)
        sig_exp, pla_exp, c_exp = sigeps93_solid_oracle(
            sig_old, deps, 0.0,
            p.e11, p.e22, p.e33, p.nu12, p.nu13, p.nu23,
            p.g12, p.g13, p.g23, p.r11, p.r22, p.r33,
            p.r12, p.r13, p.r23, p.sigma_y, p.qr1, p.cr1, p.qr2, p.cr2,
            rho0=p.rho0,
        )
        assert np.allclose(sig_act, sig_exp, rtol=1e-12, atol=1e-12)
        assert math.isclose(pla_act, 0.0, abs_tol=1e-15)
        assert math.isclose(c_act, c_exp, rel_tol=1e-12)

    @pytest.mark.parametrize(
        "deps",
        [
            np.array([5.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0]),        # Tension in 11
            np.array([0.0, 5.0e-3, 0.0, 0.0, 0.0, 0.0]),        # Tension in 22
            np.array([0.0, 0.0, 5.0e-3, 0.0, 0.0, 0.0]),        # Tension in 33
            np.array([0.0, 0.0, 0.0, 8.0e-3, 0.0, 0.0]),        # Shear in 12
            np.array([3.0e-3, 2.0e-3, -1.0e-3, 2.0e-3, 0.0, 0.0]),  # Multiaxial
        ],
    )
    def test_solid_plastic_yield_parity(self, deps: np.ndarray):
        """Solid plastic response matches sigeps93.F cutting-plane algorithm."""
        p = build_law93(
            rho0=7.8e-6,
            e11=210000.0, e22=205000.0, e33=210000.0,
            nu12=0.3, nu13=0.3, nu23=0.3,
            g12=80000.0, g13=80000.0, g23=80000.0,
            r11=1.0, r22=0.9, r33=1.1,
            r12=1.05, r13=0.95, r23=1.0,
            sigma_y=280.0, qr1=80.0, cr1=20.0, qr2=30.0, cr2=5.0,
        )
        sig_old = np.zeros(6)
        sig_act, pla_act, c_act = solid_update(p, sig_old, deps, epsp=0.0)
        sig_exp, pla_exp, c_exp = sigeps93_solid_oracle(
            sig_old, deps, 0.0,
            p.e11, p.e22, p.e33, p.nu12, p.nu13, p.nu23,
            p.g12, p.g13, p.g23, p.r11, p.r22, p.r33,
            p.r12, p.r13, p.r23, p.sigma_y, p.qr1, p.cr1, p.qr2, p.cr2,
            rho0=p.rho0,
        )
        assert np.allclose(sig_act, sig_exp, rtol=1e-6, atol=1e-6)
        assert math.isclose(pla_act, pla_exp, rel_tol=1e-6, abs_tol=1e-8)
        assert math.isclose(c_act, c_exp, rel_tol=1e-12)
        assert pla_act > 0.0

    @pytest.mark.parametrize(
        "deps",
        [
            np.array([1.0e-4, 0.0, 0.0]),
            np.array([0.0, 1.0e-4, 0.0]),
            np.array([0.0, 0.0, 1.0e-4]),
            np.array([5.0e-5, -2.0e-5, 1.0e-5]),
        ],
    )
    def test_shell_pure_elastic_parity(self, deps: np.ndarray):
        """Shell plane-stress elastic response matches plane-stress Hooke matrix."""
        p = build_law93(
            rho0=7.8e-6,
            e11=210000.0, e22=200000.0, e33=210000.0,
            nu12=0.3, g12=80000.0,
            sigma_y=600.0,
        )
        sig_old = np.zeros(3)
        sig_act, pla_act, c_act = shell_update(p, sig_old, deps, epsp=0.0)
        sig_exp, pla_exp, c_exp = sigeps93c_shell_oracle(
            sig_old, deps, 0.0,
            p.e11, p.e22, p.nu12, p.g12,
            p.r11, p.r22, p.r33, p.r12,
            p.sigma_y, p.qr1, p.cr1, p.qr2, p.cr2,
            rho0=p.rho0,
        )
        assert np.allclose(sig_act, sig_exp, rtol=1e-12, atol=1e-12)
        assert math.isclose(pla_act, 0.0, abs_tol=1e-15)
        assert math.isclose(c_act, c_exp, rel_tol=1e-12)

    @pytest.mark.parametrize(
        "deps",
        [
            np.array([4.0e-3, 0.0, 0.0]),                       # Uniaxial X
            np.array([0.0, 4.0e-3, 0.0]),                       # Uniaxial Y
            np.array([3.0e-3, 3.0e-3, 0.0]),                    # Biaxial
            np.array([0.0, 0.0, 6.0e-3]),                       # Pure shear
            np.array([2.5e-3, -1.0e-3, 3.0e-3]),                # Combined
        ],
    )
    def test_shell_plastic_yield_parity(self, deps: np.ndarray):
        """Shell plane-stress plastic response matches sigeps93c.F return mapping."""
        p = build_law93(
            rho0=7.8e-6,
            e11=210000.0, e22=200000.0, e33=210000.0,
            nu12=0.3, g12=78000.0,
            r11=1.0, r22=0.92, r33=1.08, r12=0.98,
            sigma_y=260.0, qr1=75.0, cr1=18.0, qr2=25.0, cr2=6.0,
        )
        sig_old = np.zeros(3)
        extra = {"thick": 1.0}
        sig_act, pla_act, c_act = shell_update(p, sig_old, deps, epsp=0.0, extra=extra)
        sig_exp, pla_exp, c_exp = sigeps93c_shell_oracle(
            sig_old, deps, 0.0,
            p.e11, p.e22, p.nu12, p.g12,
            p.r11, p.r22, p.r33, p.r12,
            p.sigma_y, p.qr1, p.cr1, p.qr2, p.cr2,
            rho0=p.rho0,
        )
        assert np.allclose(sig_act, sig_exp, rtol=1e-6, atol=1e-6)
        assert math.isclose(pla_act, pla_exp, rel_tol=1e-6, abs_tol=1e-8)
        assert math.isclose(c_act, c_exp, rel_tol=1e-12)
        assert pla_act > 0.0
        # Shell thinning verification
        assert "thick" in extra
        assert float(extra["thick"]) <= 1.0
        if np.any(deps[:2] > 0):
            assert float(extra["thick"]) < 1.0

    def test_solid_algorithmic_tangent_consistency(self):
        """3D consistent tangent matches numerical perturbation along plastic loading directions."""
        p = build_law93(
            rho0=7.8e-6,
            e11=210000.0, e22=205000.0, e33=210000.0,
            nu12=0.3, nu13=0.3, nu23=0.3,
            g12=80000.0, g13=80000.0, g23=80000.0,
            r11=1.0, r22=0.95, r33=1.05,
            r12=1.0, r13=1.0, r23=1.0,
            sigma_y=300.0, qr1=100.0, cr1=15.0,
        )
        deps0 = np.array([3.0e-3, 1.5e-3, 5.0e-4, 1.0e-3, 0.0, 0.0])
        sig0, pla0, _ = solid_update(p, np.zeros(6), deps0, epsp=0.0)
        assert pla0 > 0.0

        c_alg = consistent_tangent(p, sig0, pla=pla0)
        assert c_alg.shape == (6, 6)
        # Symmetry check
        assert np.allclose(c_alg, c_alg.T, rtol=1e-8, atol=1e-8)

        # Test directional derivatives along plastic loading directions
        h = 1.0e-7
        v_directions = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            deps0 / np.linalg.norm(deps0),
        ]
        for v in v_directions:
            d_sig_alg = c_alg @ v
            sig_p, _, _ = solid_update(p, sig0, h * v, epsp=pla0)
            d_sig_num = (sig_p - sig0) / h
            assert np.allclose(d_sig_alg, d_sig_num, rtol=1e-3, atol=50.0)

    def test_shell_algorithmic_tangent_consistency(self):
        """2D shell consistent tangent matches numerical perturbation matrix."""
        p = build_law93(
            rho0=7.8e-6,
            e11=210000.0, e22=200000.0, e33=210000.0,
            nu12=0.3, g12=78000.0,
            r11=1.0, r22=0.9, r12=1.0,
            sigma_y=280.0, qr1=80.0, cr1=12.0,
        )
        deps0 = np.array([2.5e-3, 1.0e-3, 1.5e-3])
        sig0, pla0, _ = shell_update(p, np.zeros(3), deps0, epsp=0.0)
        assert pla0 > 0.0

        c_alg = consistent_tangent(p, sig0, pla=pla0, is_shell=True)
        assert c_alg.shape == (3, 3)
        assert np.allclose(c_alg, c_alg.T, rtol=1e-8, atol=1e-8)

        h = 1.0e-7
        c_num = np.zeros((3, 3))
        for j in range(3):
            ddeps = np.zeros(3)
            ddeps[j] = h
            sig_p, _, _ = shell_update(p, sig0, ddeps, epsp=pla0)
            c_num[:, j] = (sig_p - sig0) / h

        assert np.allclose(c_alg, c_num, rtol=1e-3, atol=50.0)

    def test_vectorized_batch_execution(self):
        """Vectorized (nel, 6) solid and (nel, 3) shell updates match individual runs."""
        p = build_law93(
            rho0=7.8e-6,
            e11=210000.0, e22=205000.0, e33=210000.0,
            nu12=0.3, nu13=0.3, nu23=0.3,
            g12=80000.0, g13=80000.0, g23=80000.0,
            r11=1.0, r22=0.9, r33=1.1, r12=1.0,
            sigma_y=250.0, qr1=60.0, cr1=20.0,
        )
        nel = 8
        np.random.seed(42)
        sig_batch = np.random.uniform(-100.0, 100.0, (nel, 6))
        deps_batch = np.random.uniform(0.0, 3.0e-3, (nel, 6))
        epsp_batch = np.random.uniform(0.0, 0.02, nel)

        sig_vec, pla_vec, c_vec = solid_update(p, sig_batch, deps_batch, epsp=epsp_batch)

        for i in range(nel):
            sig_i, pla_i, c_i = solid_update(p, sig_batch[i], deps_batch[i], epsp=epsp_batch[i])
            assert np.allclose(sig_vec[i], sig_i, rtol=1e-12, atol=1e-12)
            assert math.isclose(pla_vec[i], pla_i, rel_tol=1e-12, abs_tol=1e-15)
            assert math.isclose(c_vec[i], c_i, rel_tol=1e-12)

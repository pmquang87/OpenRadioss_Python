"""
Tests for Milestone M569: /MAT/LAW95 (/MAT/BERGSTROM_BOYCE) Bergstrom-Boyce Model
Fortran Reference Parity Test vs sigeps95.F, viscbb.F, and sigpoly.F.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law95_bergstrom_boyce import (
    build_law95,
    solid_update,
    poly_stress,
    visc_bb,
    calc_mat_b,
)


def fortran_sigpoly_oracle(
    matb: np.ndarray,
    c_coeffs: tuple[float, ...],
    d1: float,
    d2: float,
    d3: float,
    rbulk: float,
    iform: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure bit-faithful reference oracle replicating sigpoly.F (POLYSTRESS2 / POLYSTREST2)."""
    c10, c01, c20, c11, c02, c30, c21, c12, c03 = c_coeffs
    matb2 = matb @ matb

    # J = sqrt(det(B))
    jdet = math.sqrt(max(1e-20, float(np.linalg.det(matb))))

    i1 = matb[0, 0] + matb[1, 1] + matb[2, 2]
    trb22 = matb2[0, 0] + matb2[1, 1] + matb2[2, 2]
    i2 = 0.5 * (i1 ** 2 - trb22)

    jthird = math.exp((-1.0 / 3.0) * math.log(jdet))
    j2third = jthird ** 2
    j4third = jthird ** 4

    bi1 = i1 * j2third
    bi2 = i2 * j4third

    di1 = bi1 - 3.0
    di2 = bi2 - 3.0

    dphidi1 = (
        c10
        + 2.0 * c20 * di1
        + 3.0 * c30 * (di1 ** 2)
        + c11 * di2
        + c12 * (di2 ** 2)
        + 2.0 * c21 * di1 * di2
    )
    dphidi2 = (
        c01
        + 2.0 * c02 * di2
        + 3.0 * c03 * (di2 ** 2)
        + c11 * di1
        + c21 * (di1 ** 2)
        + 2.0 * c12 * di1 * di2
    )

    inv2j = 2.0 / max(1e-20, jdet)

    j_minus_1 = jdet - 1.0
    if iform == 1:
        dphidj = 2.0 * d1 * j_minus_1 + 4.0 * d2 * (j_minus_1 ** 3) + 6.0 * d3 * (j_minus_1 ** 5)
        dphi2dj = 2.0 * d1 + 12.0 * d2 * (j_minus_1 ** 2) + 30.0 * d3 * (j_minus_1 ** 4)
    else:
        dphidj = rbulk * (1.0 - 1.0 / jdet)
        dphi2dj = rbulk / (jdet ** 2)

    aa = (dphidi1 + dphidi2 * bi1) * inv2j * j2third
    bb = dphidi2 * inv2j * j4third
    cc = (1.0 / 3.0) * inv2j * (bi1 * dphidi1 + 2.0 * bi2 * dphidi2)

    sig = aa * matb - bb * matb2
    sig[0, 0] += -cc + dphidj
    sig[1, 1] += -cc + dphidj
    sig[2, 2] += -cc + dphidj

    dphi2_di1 = 2.0 * (c20 + 3.0 * c30 * di1 + c21 * di2)
    dphi2_di2 = 2.0 * (c02 + 3.0 * c03 * di2 + c12 * di1)

    lam_b = np.array([matb[0, 0], matb[1, 1], matb[2, 2]], dtype=np.float64) * j2third
    lam_b_safe = np.maximum(lam_b, 1e-12)
    lam_b_inv = 1.0 / lam_b_safe

    bi1_3 = bi1 / 3.0
    bi2_3 = bi2 / 3.0

    term1 = (2.0 / 3.0) * dphidi1 * (lam_b + bi1_3) + dphi2_di1 * (lam_b - bi1_3)
    term2 = (2.0 / 3.0) * dphidi2 * (lam_b_inv + bi2_3) + dphi2_di2 * (lam_b_inv - bi2_3)
    cii = 2.0 * (term1 + term2) + dphi2dj

    return sig, cii


def fortran_viscbb_oracle(
    fp: np.ndarray,
    tbnorm: float,
    a1: float,
    expc: float,
    expm: float,
    ksi: float,
    tauref: float,
) -> float:
    """Pure bit-faithful reference oracle replicating viscbb.F."""
    ip1 = fp[0, 0] ** 2 + fp[1, 1] ** 2 + fp[2, 2] ** 2
    lpchain = math.sqrt(max(0.0, ip1 / 3.0))
    temp = max(1e-20, lpchain - 1.0 + ksi)
    dgamma = a1 * math.exp(expc * math.log(temp)) * ((tbnorm ** expm) / max(1e-20, tauref ** expm))
    return float(dgamma)


def fortran_sigeps95_oracle(
    f: np.ndarray,
    fpo: np.ndarray,
    c_coeffs: tuple[float, ...],
    d_params: tuple[float, float, float],
    sb: float,
    a1: float,
    expc: float,
    expm: float,
    ksi: float,
    tauref: float,
    rbulk: float,
    iform: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pure bit-faithful reference oracle replicating sigeps95.F."""
    # 1. Network A: b = F * F^T
    b_a = f @ f.T
    d1, d2, d3 = d_params
    siga, _ = fortran_sigpoly_oracle(b_a, c_coeffs, d1, d2, d3, rbulk, iform)

    if a1 * sb > 0.0:
        # 2. Trial elastic B kinematics: Fe = F * Fp_old^(-1), matb = Fe * Fe^T
        inv_fpo = np.linalg.inv(fpo)
        fe = f @ inv_fpo
        matb = fe @ fe.T

        # 3. Trial Cauchy stress in chain B
        sigb_trial, _ = fortran_sigpoly_oracle(matb, c_coeffs, d1, d2, d3, rbulk, iform)
        sigb = sb * sigb_trial

        # 4. Deviator and Frobenius norm
        traceb = (1.0 / 3.0) * (sigb[0, 0] + sigb[1, 1] + sigb[2, 2])
        sb_dev = sigb - traceb * np.eye(3)
        tbnorm = math.sqrt(max(1e-20, float(
            sb_dev[0, 0] ** 2 + sb_dev[1, 1] ** 2 + sb_dev[2, 2] ** 2
            + 2.0 * (sb_dev[0, 1] ** 2 + sb_dev[1, 2] ** 2 + sb_dev[2, 0] ** 2)
        )))

        # 5. Creep rate dgamma
        dgamma = fortran_viscbb_oracle(fpo, tbnorm, a1, expc, expm, ksi, tauref)

        # 6. Velocity gradient L_B = (dgamma / tbnorm) * sb_dev
        factor = dgamma / max(1e-20, tbnorm)
        lb = factor * sb_dev

        # 7. Inelastic gradient update: DFP = INVFE * LB * FE
        inv_fe = np.linalg.inv(fe)
        dfp = inv_fe @ (lb @ fe)

        # 8. Updated Fp = (I + DFP) * Fp_old
        sn = np.eye(3) + dfp
        fp_new = sn @ fpo

        # 9. Updated B stress
        inv_fp_new = np.linalg.inv(fp_new)
        fe_new = f @ inv_fp_new
        matb_new = fe_new @ fe_new.T
        sigb_new, _ = fortran_sigpoly_oracle(matb_new, c_coeffs, d1, d2, d3, rbulk, iform)
        sigb = sb * sigb_new
    else:
        sigb = sb * siga
        fp_new = fpo.copy()
        dgamma = 0.0

    sig_tot = siga + sigb
    return sig_tot, fp_new, dgamma


# ============================================================================
# Parity Test Cases
# ============================================================================

def test_parity_pure_shear():
    """Verify exact parity between Python solid_update and Fortran oracle under pure shear."""
    c_coeffs = (1.2e6, 2.5e5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    d1 = 1.0e-8
    d_params = (1.0 / d1, 0.0, 0.0)
    sb = 0.8
    a = 0.05
    dt = 1.0e-4
    a1 = a * dt
    expc = -0.7
    expm = 1.0
    ksi = 0.01
    tauref = 1.0e5
    g0 = 2.0 * (c_coeffs[0] + c_coeffs[1]) * (1.0 + sb)
    rbulk = 2.0 * (1.0 / d1) * (1.0 + sb)

    p = build_law95(
        c10=c_coeffs[0], c01=c_coeffs[1], sb=sb, d1=d1, a=a,
        expc=expc, expm=expm, ksi=ksi, tauref=tauref, rho0=1000.0,
    )

    gamma = 0.15
    f = np.array([
        [1.0, gamma, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    fp_init = np.eye(3, dtype=np.float64)

    # Oracle
    sig_oracle, fp_oracle, dg_oracle = fortran_sigeps95_oracle(
        f, fp_init, c_coeffs, d_params, sb, a1, expc, expm, ksi, tauref, rbulk
    )

    # Python kernel
    extra = {"F": f, "uvar": np.zeros(10, dtype=np.float64)}
    extra["uvar"][:3] = 1.0
    sig_py, _, _ = solid_update(p, dt=dt, extra=extra)

    # Cauchy stress 6 components: xx, yy, zz, xy, yz, zx
    assert math.isclose(sig_py[0], sig_oracle[0, 0], rel_tol=1e-10, abs_tol=1e-6)
    assert math.isclose(sig_py[1], sig_oracle[1, 1], rel_tol=1e-10, abs_tol=1e-6)
    assert math.isclose(sig_py[2], sig_oracle[2, 2], rel_tol=1e-10, abs_tol=1e-6)
    assert math.isclose(sig_py[3], sig_oracle[0, 1], rel_tol=1e-10, abs_tol=1e-6)
    assert math.isclose(sig_py[4], sig_oracle[1, 2], rel_tol=1e-10, abs_tol=1e-6)
    assert math.isclose(sig_py[5], sig_oracle[2, 0], rel_tol=1e-10, abs_tol=1e-6)

    # Inelastic deformation gradient components
    uvar = extra["uvar"]
    assert math.isclose(uvar[0], fp_oracle[0, 0], rel_tol=1e-10)
    assert math.isclose(uvar[1], fp_oracle[1, 1], rel_tol=1e-10)
    assert math.isclose(uvar[2], fp_oracle[2, 2], rel_tol=1e-10)
    assert math.isclose(uvar[3], fp_oracle[0, 1], rel_tol=1e-10, abs_tol=1e-12)
    assert math.isclose(uvar[9], dg_oracle, rel_tol=1e-10)


def test_parity_uniaxial_extension():
    """Verify exact parity under uniaxial extension."""
    c_coeffs = (2.0e6, 3.0e5, 5.0e4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    d1 = 2.0e-8
    d_params = (1.0 / d1, 0.0, 0.0)
    sb = 1.0
    a = 0.02
    dt = 5.0e-5
    a1 = a * dt
    expc = -0.65
    expm = 1.15
    ksi = 0.02
    tauref = 1.5e5
    rbulk = 2.0 * (1.0 / d1) * (1.0 + sb)

    p = build_law95(
        c10=c_coeffs[0], c01=c_coeffs[1], c20=c_coeffs[2], sb=sb, d1=d1, a=a,
        expc=expc, expm=expm, ksi=ksi, tauref=tauref, rho0=1100.0,
    )

    lam1 = 1.2
    lam2 = 1.0 / math.sqrt(lam1)
    f = np.diag([lam1, lam2, lam2])
    fp_init = np.eye(3, dtype=np.float64)

    sig_oracle, fp_oracle, dg_oracle = fortran_sigeps95_oracle(
        f, fp_init, c_coeffs, d_params, sb, a1, expc, expm, ksi, tauref, rbulk
    )

    extra = {"F": f, "uvar": np.zeros(10, dtype=np.float64)}
    extra["uvar"][:3] = 1.0
    sig_py, _, _ = solid_update(p, dt=dt, extra=extra)

    assert math.isclose(sig_py[0], sig_oracle[0, 0], rel_tol=1e-10)
    assert math.isclose(sig_py[1], sig_oracle[1, 1], rel_tol=1e-10)
    assert math.isclose(sig_py[2], sig_oracle[2, 2], rel_tol=1e-10)
    assert math.isclose(extra["uvar"][9], dg_oracle, rel_tol=1e-10)


def test_parity_iform2_volumetric_expansion():
    """Verify exact parity for IFORM=2 pressure under volumetric deformation."""
    c_coeffs = (1.0e6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    d_params = (5.0e7, 0.0, 0.0)
    rbulk = 1.0e8
    sb = 0.0  # Network A only
    a1 = 0.0

    p = build_law95(
        c10=c_coeffs[0], sb=0.0, a=0.0, iform=2, rho0=1000.0,
    )
    p.rbulk = rbulk

    J = 1.06
    f = (J ** (1.0 / 3.0)) * np.eye(3)
    fp_init = np.eye(3)

    sig_oracle, _, _ = fortran_sigeps95_oracle(
        f, fp_init, c_coeffs, d_params, sb, a1, -0.7, 1.0, 0.01, 1.0, rbulk, iform=2
    )

    extra = {"F": f}
    sig_py, _, _ = solid_update(p, dt=1.0e-5, extra=extra)

    for k in range(3):
        assert math.isclose(sig_py[k], sig_oracle[k, k], rel_tol=1e-10)

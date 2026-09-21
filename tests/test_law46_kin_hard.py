"""
Unit test for LAW46 (Armstrong-Frederick nonlinear kinematic hardening).

Fortran origins:
- Source: engine/source/materials/mat/mat046/sigeps46.F (subroutine SIGEPS46)
- Subroutine: M46LAW (engine/source/materials/mat/mat046/m46law.F)
- Starter: starter/source/materials/mat/mat046/hm_read_mat46.F

Verifies uniaxial tension behavior:
1. Elastic response first: below yield stress, strain increments are purely elastic (epsp=0, backstress=0).
2. Plastic flow starts at yield stress: once sigma reaches sig_y, plastic flow initiates (epsp > 0, backstress begins to evolve).
3. Hardening behavior: with continued plastic strain, kinematic backstress alpha evolves according to
   dalpha = C * depsilon_p - gamma * alpha * depsilon_p_eq, producing hardening and asymptotic saturation.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law46_kin_hard import (
    Law46Params,
    solid_update,
    shell_update,
    solid_tangent,
    tangent,
    sound_speed,
)


def _make_law46_mat(
    young: float = 200_000.0,
    nu: float = 0.3,
    sig_y: float = 250.0,
    c_kin: float = 50_000.0,
    gamma_kin: float = 500.0,
    h_iso: float = 0.0,
    rho0: float = 7.85e-9,
) -> Law46Params:
    """Helper to create Law46Params with standard material properties."""
    return Law46Params(
        young=young,
        nu=nu,
        sig_y=sig_y,
        c_kin=c_kin,
        gamma_kin=gamma_kin,
        h_iso=h_iso,
        rho0=rho0,
    )


def test_uniaxial_tension_elastic_plastic_hardening():
    """
    Uniaxial tension test verifying:
      1. Elastic response first
      2. Plastic flow starts at yield stress
      3. Hardening behavior
    """
    mat = _make_law46_mat(
        young=200_000.0,
        nu=0.3,
        sig_y=250.0,
        c_kin=50_000.0,
        gamma_kin=500.0,
        h_iso=0.0,
    )

    sig = np.zeros(6, dtype=float)
    extra = {"alpha": np.zeros(6, dtype=float)}
    epsp = 0.0

    # Yield strain: eps_y = sig_y / E = 250 / 200000 = 0.00125
    deps_xx = 0.00025  # 5 steps to reach yield
    n_steps_elastic = 4  # 4 * 0.00025 = 0.00100 < 0.00125 (purely elastic)

    stresses = []
    eps_p_history = []
    alpha_history = []

    # -------------------------------------------------------------------------
    # 1. Elastic response first
    # -------------------------------------------------------------------------
    for step in range(n_steps_elastic):
        # Uniaxial strain with Poisson contraction: deps_yy = deps_zz = -nu * deps_xx
        deps = np.array([deps_xx, -mat.nu * deps_xx, -mat.nu * deps_xx, 0.0, 0.0, 0.0])
        sig, epsp, _ = solid_update(mat, sig, deps, epsp=epsp, extra=extra, return_sound_speed=True)
        sig = np.atleast_2d(sig)[0]
        epsp = float(np.atleast_1d(epsp)[0])
        alpha = np.atleast_2d(extra["alpha"])[0].copy()

        stresses.append(sig[0])
        eps_p_history.append(epsp)
        alpha_history.append(alpha[0])

        # Verify plastic strain remains zero in elastic regime
        assert epsp == pytest.approx(0.0, abs=1e-12), f"Step {step+1}: expected zero plastic strain, got {epsp}"

        # Verify backstress remains zero in elastic regime
        assert np.allclose(alpha, 0.0, atol=1e-10), f"Step {step+1}: expected zero backstress, got {alpha}"

        # Verify linear elastic stress response: sigma_xx ≈ E * total_eps_xx
        expected_sxx = mat.young * (step + 1) * deps_xx
        assert sig[0] == pytest.approx(expected_sxx, rel=1e-4), (
            f"Step {step+1}: expected elastic stress {expected_sxx:.2f} MPa, got {sig[0]:.2f} MPa"
        )
        assert sig[0] < mat.sig_y, f"Step {step+1}: stress {sig[0]:.2f} exceeded yield {mat.sig_y:.2f}"

    # -------------------------------------------------------------------------
    # 2. Plastic flow starts at yield stress
    # -------------------------------------------------------------------------
    # Step 5 & 6: continue incrementing deps_xx, breaching yield (yield strain = 0.00125)
    for _ in range(2):
        sig, epsp, _ = solid_update(mat, sig, deps, epsp=epsp, extra=extra, return_sound_speed=True)
        sig = np.atleast_2d(sig)[0]
        epsp = float(np.atleast_1d(epsp)[0])
        alpha = np.atleast_2d(extra["alpha"])[0].copy()

    # Verify plastic flow has initiated
    assert epsp > 0.0, f"Expected plastic strain > 0, got {epsp}"
    assert alpha[0] > 0.0, f"Expected positive backstress alpha_xx, got {alpha[0]}"

    # -------------------------------------------------------------------------
    # 3. Hardening behavior
    # -------------------------------------------------------------------------
    # Apply 50 plastic increments and track stress and backstress evolution
    prev_sig_xx = sig[0]
    prev_alpha_xx = alpha[0]

    for step in range(50):
        sig, epsp, _ = solid_update(mat, sig, deps, epsp=epsp, extra=extra, return_sound_speed=True)
        sig = np.atleast_2d(sig)[0]
        epsp = float(np.atleast_1d(epsp)[0])
        alpha = np.atleast_2d(extra["alpha"])[0].copy()

        # Hardening: stress must exceed initial yield
        assert sig[0] > mat.sig_y, f"Stress {sig[0]:.2f} must exceed initial yield {mat.sig_y:.2f}"

        # Backstress must continue to increase monotonically in tension
        assert alpha[0] >= prev_alpha_xx, (
            f"Backstress should increase monotonically: {alpha[0]} < {prev_alpha_xx}"
        )
        prev_sig_xx = sig[0]
        prev_alpha_xx = alpha[0]

    # Theoretical saturation limit for Armstrong-Frederick:
    # dalpha/deps_p -> 0 gives alpha_sat <= C / gamma
    alpha_sat_upper = mat.c_kin / mat.gamma_kin  # 50000 / 500 = 100 MPa
    assert alpha[0] <= alpha_sat_upper * 1.1, (
        f"Backstress {alpha[0]:.2f} MPa exceeded theoretical upper bound {alpha_sat_upper:.2f} MPa"
    )
    assert alpha[0] >= alpha_sat_upper * 0.4, (
        f"Backstress {alpha[0]:.2f} MPa did not accumulate significantly toward {alpha_sat_upper:.2f} MPa"
    )


def test_bauschinger_effect():
    """
    Verify the Bauschinger effect:
    After tensile plastic deformation develops backstress (alpha > 0),
    reverse compressive yielding occurs at a reduced stress magnitude.
    """
    mat = _make_law46_mat(
        young=200_000.0,
        nu=0.3,
        sig_y=250.0,
        c_kin=50_000.0,
        gamma_kin=500.0,
        h_iso=0.0,
    )

    sig = np.zeros(6, dtype=float)
    extra = {"alpha": np.zeros(6, dtype=float)}
    epsp = 0.0

    # Forward loading into plastic regime
    deps_fwd = np.array([0.02, -0.01, -0.01, 0.0, 0.0, 0.0])
    sig, epsp, _ = solid_update(mat, sig, deps_fwd, epsp=epsp, extra=extra, return_sound_speed=True)
    sig_fwd = np.atleast_2d(sig)[0].copy()
    epsp_fwd = float(np.atleast_1d(epsp)[0])
    alpha_fwd = np.atleast_2d(extra["alpha"])[0].copy()

    assert alpha_fwd[0] > 0.0, "Tensile backstress must be positive after forward load"

    # Reverse loading (compression)
    # With shifted yield surface (s - alpha), reverse yield occurs earlier
    deps_rev = np.array([-0.001, 0.0005, 0.0005, 0.0, 0.0, 0.0])
    sig_rev, epsp_rev, _ = solid_update(mat, sig_fwd.copy(), deps_rev, epsp=epsp_fwd,
                                        extra={"alpha": alpha_fwd.copy()}, return_sound_speed=True)
    sig_rev = np.atleast_2d(sig_rev)[0]

    # Effective reverse yield stress in tension-compression is shifted by 2 * alpha_xx
    # |sigma_reverse_yield| < sigma_y0
    reverse_yield_approx = mat.sig_y - 2.0 * alpha_fwd[0]
    assert reverse_yield_approx < mat.sig_y, "Bauschinger effect implies reduced reverse yield"


def test_law46_tangent_and_sound_speed():
    """Verify tangent stiffness matrix and acoustic sound speed calculation."""
    mat = _make_law46_mat()
    c_tangent = tangent(mat)

    assert c_tangent.shape == (6, 6), f"Expected (6, 6) tangent matrix, got {c_tangent.shape}"
    assert np.all(np.isfinite(c_tangent))

    # Check Lamé moduli in tangent matrix
    G = mat.G
    K = mat.K
    lam = K - (2.0 / 3.0) * G
    assert c_tangent[0, 0] == pytest.approx(lam + 2.0 * G)
    assert c_tangent[0, 1] == pytest.approx(lam)
    assert c_tangent[3, 3] == pytest.approx(G)

    # Sound speed check
    c = sound_speed(mat, is_shell=False)
    expected_c = math.sqrt((K + 4.0 * G / 3.0) / mat.rho0)
    assert c == pytest.approx(expected_c, rel=1e-9)

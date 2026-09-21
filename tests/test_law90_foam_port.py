"""Tests for LAW90 tabulated foam with hysteresis and crush locking (sigeps90.F).

Verifies:
  1. Elastic loading: small strains follow initial Young's modulus / curve slope.
  2. Load-unload hysteresis: unloading follows a different, lower stress path due to hysteretic energy dissipation.
  3. Crush locking behavior: dramatic stress increase in the high-compression densification regime.
  4. Strain-rate dependence: dynamic stiffening with rate interpolation.
  5. Tangent stiffness operator and material law conventions.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law90_foam import (
    Law90Params,
    build_law90,
    solid_update,
    solid_step,
    sound_speed,
    solid_tangent,
    tangent,
)


def test_law90_elastic_loading():
    """Verify small compressive strains follow the elastic loading curve."""
    # Define linear curve: strain [0.0, 0.05], stress [0.0, 0.5] -> slope = 10.0 MPa
    c_x = np.array([0.0, 0.05, 0.2, 0.5])
    c_y = np.array([0.0, 0.5, 1.0, 2.0])

    mat = build_law90({
        "E0": 10.0,
        "nu": 0.0,
        "hys": 0.5,
        "curves": [(c_x, c_y)],
    })

    sig0 = np.zeros(6, dtype=float)
    extra = {}
    # Apply small compressive strain: eps_xx = -0.01 (log strain ~ -0.01, lambda = exp(-0.01) ~ 0.99005)
    # Eng strain e = 1 - lambda ~ 0.00995 -> nominal stress S ~ 0.0995
    # Cauchy stress sigma_xx = -S / (lambda_y * lambda_z) = -0.0995 / 1.0 = -0.0995 MPa
    deps = np.array([-0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig1, epsp1, c1 = solid_update(mat, sig0, deps, extra=extra)

    assert sig1[0] < 0.0  # Compressive stress is negative
    assert abs(sig1[0]) == pytest.approx(0.0995, rel=0.01)
    assert c1 > 0.0
    assert epsp1 > 0.0


def test_law90_unloading_different_path():
    """Verify unloading returns along a different, lower stress path (hysteretic energy dissipation)."""
    # Define foam curve: elastic slope then plateau then densification
    c_x = np.array([0.0, 0.05, 0.2, 0.4, 0.6])
    c_y = np.array([0.0, 1.0, 2.0, 3.0, 10.0])

    # HYS = 0.4 means significant hysteretic dissipation on unloading
    mat = build_law90({
        "E0": 20.0,
        "nu": 0.0,
        "hys": 0.4,
        "shape": 1.5,
        "alpha": 1.0,
        "curves": [(c_x, c_y)],
    })

    extra = {}
    sig = np.zeros(6, dtype=float)

    # Step 1: Load to eps_xx = -0.1 (log strain)
    d_load1 = np.array([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_load1, _, _ = solid_update(mat, sig, d_load1, extra=extra)
    load_stress_at_01 = abs(sig_load1[0])

    # Step 2: Continue loading to eps_xx = -0.2 (further into plateau)
    d_load2 = np.array([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_load2, _, _ = solid_update(mat, sig_load1, d_load2, extra=extra)
    load_stress_at_02 = abs(sig_load2[0])
    assert load_stress_at_02 > load_stress_at_01

    # Step 3: Unload back to eps_xx = -0.15 (positive strain increment = expansion/unloading)
    d_unload1 = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_unl1, _, _ = solid_update(mat, sig_load2, d_unload1, extra=extra)

    # Step 4: Unload further back to eps_xx = -0.1 (the exact same strain as Step 1)
    d_unload2 = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_unl2, _, _ = solid_update(mat, sig_unl1, d_unload2, extra=extra)
    unload_stress_at_01 = abs(sig_unl2[0])

    # The unloading stress at the same strain must be strictly LOWER than loading stress
    # Demonstrating hysteresis loop / different return path
    assert unload_stress_at_01 < load_stress_at_01
    ratio = unload_stress_at_01 / load_stress_at_01
    assert 0.4 <= ratio <= 0.8  # Consistent with HYS = 0.4


def test_law90_crush_locking_high_compression():
    """Verify crush locking behavior (densification) at high compressive strains."""
    # Typical foam curve: plateau around 2.0 MPa, then locking/densification after strain 0.6
    c_x = np.array([0.0, 0.1, 0.3, 0.6, 0.8, 0.9])
    c_y = np.array([0.0, 1.5, 2.0, 2.5, 25.0, 100.0])

    mat = build_law90({
        "E0": 15.0,
        "nu": 0.0,
        "curves": [(c_x, c_y)],
    })

    extra1 = {}
    sig0 = np.zeros(6, dtype=float)
    # Plateau strain: e ~ 0.3 (lambda = 0.7 -> log strain = ln(0.7) ~ -0.3567)
    deps_plateau = np.array([math.log(0.7), 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_plateau, _, c_plateau = solid_update(mat, sig0, deps_plateau, extra=extra1)
    stress_plateau = abs(sig_plateau[0])
    assert 1.5 < stress_plateau < 3.0

    # Crush locking strain: e ~ 0.8 (lambda = 0.2 -> log strain = ln(0.2) ~ -1.6094)
    extra2 = {}
    deps_crush = np.array([math.log(0.2), 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_crush, _, c_crush = solid_update(mat, sig0, deps_crush, extra=extra2)
    stress_crush = abs(sig_crush[0])

    # Stress at crush locking is an order of magnitude higher than the plateau
    assert stress_crush >= 25.0
    assert stress_crush > 8.0 * stress_plateau


def test_law90_strain_rate_dependence():
    """Verify strain rate stiffening across multiple rate-dependent curves."""
    c_x = np.array([0.0, 0.1, 0.3, 0.5])
    # Static curve (rate = 0.0)
    c_y_stat = np.array([0.0, 1.0, 1.5, 2.0])
    # Dynamic curve (rate = 100.0 /s) - 2x higher stresses
    c_y_dyn = np.array([0.0, 2.0, 3.0, 4.0])

    mat = build_law90({
        "E0": 10.0,
        "nu": 0.0,
        "nl": 2,
        "eps_dots": [0.0, 100.0],
        "curves": [(c_x, c_y_stat), (c_x, c_y_dyn)],
    })

    deps = np.array([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig0 = np.zeros(6, dtype=float)

    # Quasi-static rate: dt = 1.0 s -> rate ~ 0.1 /s
    extra_stat = {}
    sig_stat, _, _ = solid_update(mat, sig0, deps, dt=1.0, extra=extra_stat)

    # High dynamic rate: dt = 0.001 s -> rate ~ 100 /s
    extra_dyn = {}
    sig_dyn, _, _ = solid_update(mat, sig0, deps, dt=0.001, extra=extra_dyn)

    # Dynamic stress must be significantly higher due to rate-dependent stiffening
    assert abs(sig_dyn[0]) > 1.5 * abs(sig_stat[0])


def test_law90_tangent_operator():
    """Verify consistent tangent matrix generation for LAW90."""
    mat = build_law90({
        "E0": 50.0,
        "nu": 0.25,
    })
    C = tangent(mat)
    assert C.shape == (6, 6)
    assert np.all(np.isfinite(C))
    # Check diagonal positivity
    assert np.all(np.diag(C) > 0.0)
    # Check symmetry
    assert np.allclose(C, C.T)

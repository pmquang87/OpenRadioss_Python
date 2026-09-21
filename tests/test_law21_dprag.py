"""Unit test suite for LAW21 (Drucker-Prager concrete / geomaterial model with compaction EOS).

Upstream OpenRadioss Fortran reference:
  C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\materials\\mat\\mat021\\m21law.F
  (historically also referenced as sigeps21.F)
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law21_dprag import (
    build_law21,
    extra_shapes,
    needs_defgrad,
    solid_update,
    sound_speed,
    tangent,
    consistent_solid_tangent,
)
from pyradioss.materials import (
    solid_update as mat_solid_update,
    solid_tangent as mat_solid_tangent,
)


def test_law21_cap_hardening_volumetric_compaction():
    """Verify volumetric compaction EOS and cap hardening hysteretic unloading.

    Difference from LAW81:
    LAW21 tracks maximum historical compaction mu_bak, and upon unloading
    follows a stiffer modulus K_unload = alpha * B_max + (1 - alpha) * B_min.
    """
    mat_def = {
        "id": 1,
        "MAT_RHO": 2400.0,
        "MAT_G": 10000.0,     # Shear modulus G = 10 GPa
        "MAT_C1": 15000.0,    # Minimum / initial bulk modulus B_min = 15 GPa
        "MAT_BUNL": 45000.0,  # Maximum unloading bulk modulus B_max = 45 GPa
        "MAT_MUMAX": 0.1,     # Max compaction limit mu_max = 0.1
        "MAT_A0": 1000.0,     # Large yield envelope to isolate volumetric EOS
        "MAT_A1": 5.0,
        "MAT_A2": 0.0,
        "MAT_Amax": 1e6,
        "MAT_PMIN": -100.0,
    }
    mat = build_law21(mat_def)

    # 1. Loading step: compressive volumetric strain delta_eps_vol = -0.05
    # (in engineering mechanics, compressive tr(deps) < 0 increases density and mu)
    sig0 = np.zeros(6, dtype=float)
    deps_load = np.array([-0.05 / 3.0, -0.05 / 3.0, -0.05 / 3.0, 0.0, 0.0, 0.0])
    extra = {"mu": 0.0, "mu_bak": 0.0}

    sig_1, _, _ = solid_update(mat, sig0, deps=deps_load, dt=1e-3, extra=extra, return_tuple=True)

    # During linear loading: P = C1 * mu = 15000 * 0.05 = 750 MPa
    # In Radioss, Cauchy stress sig = s - P*I, so sig_xx = -P = -750 MPa
    p_load = -(sig_1[0] + sig_1[1] + sig_1[2]) / 3.0
    assert math.isclose(p_load, 750.0, rel_tol=1e-3)
    # Check historical compaction memory mu_bak is updated
    assert math.isclose(float(np.atleast_1d(extra["mu_bak"])[0]), 0.05, rel_tol=1e-3)

    # 2. Unloading step: tensile volumetric strain increment +0.02 (mu decreases to 0.03)
    # For mu_bak = 0.05, alpha = 0.05 / 0.1 = 0.5
    # K_unload = 0.5 * 45000 + 0.5 * 15000 = 30000 MPa (twice the loading stiffness!)
    deps_unload = np.array([0.02 / 3.0, 0.02 / 3.0, 0.02 / 3.0, 0.0, 0.0, 0.0])
    sig_2, _, _ = solid_update(mat, sig_1, deps=deps_unload, dt=1e-3, extra=extra, return_tuple=True)

    p_unl = -(sig_2[0] + sig_2[1] + sig_2[2]) / 3.0
    # Expected: P = P_load(0.05) - (0.05 - 0.03) * K_unload = 750 - 0.02 * 30000 = 750 - 600 = 150 MPa
    assert math.isclose(p_unl, 150.0, rel_tol=1e-3)
    # mu_bak remains 0.05 (compacted memory retained)
    assert math.isclose(float(np.atleast_1d(extra["mu_bak"])[0]), 0.05, rel_tol=1e-3)


def test_law21_drucker_prager_yield_surface_and_radial_return():
    """Verify Drucker-Prager parabolic envelope and radial return projection."""
    mat_def = {
        "id": 2,
        "MAT_RHO": 2000.0,
        "MAT_G": 8000.0,
        "MAT_C1": 12000.0,
        "MAT_A0": 25.0,       # Cohesion squared: G0(0) = 25 MPa^2
        "MAT_A1": 0.5,        # Friction slope
        "MAT_A2": 0.0,
        "MAT_Amax": 200.0,
        "MAT_PMIN": -10.0,
    }
    mat = build_law21(mat_def)

    # Pure shear loading with zero pressure: trial tau_xy = 10 MPa -> J2 = tau^2 = 100 > A0 = 25
    sig0 = np.zeros(6, dtype=float)
    # Elastic shear strain: deps_xy = tau / G = 10 / 8000 = 0.00125
    deps = np.array([0.0, 0.0, 0.0, 10.0 / 8000.0, 0.0, 0.0])
    extra = {}

    sig_out, epsp_out, c = solid_update(mat, sig0, deps=deps, dt=1e-3, extra=extra, return_tuple=True)

    # At P = 0, G0 = 25 MPa^2. J2_trial = 100.
    # Scale factor: ratio = sqrt(G0 / J2) = sqrt(25 / 100) = 0.5.
    # Projected tau_xy = ratio * tau_trial = 0.5 * 10 = 5.0 MPa.
    assert math.isclose(sig_out[3], 5.0, rel_tol=1e-4)

    # Check plastic strain accumulation
    assert float(epsp_out) > 0.0


def test_law21_tensile_cutoff():
    """Verify that tensile pressure below PMIN collapses yield envelope G0 to zero."""
    mat_def = {
        "id": 3,
        "MAT_RHO": 2000.0,
        "MAT_G": 8000.0,
        "MAT_C1": 12000.0,
        "MAT_A0": 50.0,
        "MAT_A1": 1.0,
        "MAT_A2": 0.0,
        "MAT_PMIN": -5.0,    # Tensile cutoff pressure
    }
    mat = build_law21(mat_def)

    # Large tensile volume expansion: tr(deps) = +0.01 -> P = C1 * (-0.01) = -120 MPa < PMIN (-5 MPa)
    sig0 = np.zeros(6, dtype=float)
    deps = np.array([0.01 / 3.0, 0.01 / 3.0, 0.01 / 3.0, 0.001, 0.0, 0.0])

    sig_out, _, _ = solid_update(mat, sig0, deps=deps, dt=1e-3, extra={}, return_tuple=True)

    # Pressure clamped to PMIN = -5 MPa -> sig_normal = -(-5) = +5 MPa
    p_out = -(sig_out[0] + sig_out[1] + sig_out[2]) / 3.0
    assert math.isclose(p_out, -5.0, rel_tol=1e-3)
    # Shear stress should collapse to 0 because G0 = 0 when P <= PMIN
    assert math.isclose(sig_out[3], 0.0, abs_tol=1e-6)


def test_law21_sound_speed_and_metadata():
    """Verify sound speed computation and model metadata."""
    mat_def = {
        "id": 4,
        "MAT_RHO": 2500.0,
        "MAT_G": 12000.0,
        "MAT_C1": 20000.0,
    }
    mat = build_law21(mat_def)

    c = sound_speed(mat)
    # c = sqrt((4/3 * G + K) / rho) = sqrt((16000 + 20000) / 2500) = sqrt(36000 / 2500) = sqrt(14.4) ~ 3.7947 km/s
    expected_c = math.sqrt((4.0 / 3.0 * 12000.0 + 20000.0) / 2500.0)
    assert math.isclose(c, expected_c, rel_tol=1e-4)

    assert needs_defgrad(mat) is False
    shapes = extra_shapes(mat)
    assert "mu_bak" in shapes
    assert "epxe" in shapes


def test_law21_consistent_tangent():
    """Verify consistent solid tangent calculation."""
    mat_def = {
        "id": 5,
        "MAT_RHO": 2400.0,
        "MAT_G": 10000.0,
        "MAT_C1": 15000.0,
        "MAT_A0": 100.0,
        "MAT_A1": 1.0,
    }
    mat = build_law21(mat_def)

    sig = np.zeros(6, dtype=float)
    deps = np.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    c_tan = tangent(mat, sig=sig, deps=deps, dt=1e-4)
    assert c_tan.shape == (6, 6) or c_tan.shape == (1, 6, 6)
    tan_mat = c_tan[0] if c_tan.ndim == 3 else c_tan
    assert tan_mat[0, 0] > 0.0
    # Major symmetry
    assert np.allclose(tan_mat, tan_mat.T, atol=1e-4)


def test_law21_dispatcher_integration():
    """Verify LAW21 integrates through central material dispatcher."""
    mat_def = {
        "id": 21,
        "MAT_RHO": 2200.0,
        "MAT_G": 9000.0,
        "MAT_C1": 14000.0,
        "MAT_A0": 40.0,
    }
    mat = build_law21(mat_def)

    sig = np.zeros(6, dtype=float)
    deps = np.array([-1e-4, -1e-4, -1e-4, 0.0, 0.0, 0.0])

    sig_out, ep_out, c_val = mat_solid_update(mat, sig, deps, dt=1e-4)
    assert c_val is not None
    assert sig_out[0] < 0.0  # Compression negative in Cauchy stress (P positive)

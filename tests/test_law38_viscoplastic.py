"""Unit test suite for LAW38 tabulated viscoplastic model with radial return mapping.

Tests:
  1. Tabulated yield stress lookup Y(eps_p) on a single curve
  2. Multi-curve strain-rate interpolation Y(eps_p, eps_p_dot)
  3. Temperature-dependent yield scaling Y(eps_p, eps_p_dot, T)
  4. Radial return mapping below and above yield
  5. Integration with solid_update in viscoplastic mode
  6. Algorithmic consistent tangent stiffness
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law38_visc_tab import (
    build_law38,
    calc_yield_stress,
    interpolate_yield_table,
    radial_return_mapping,
    solid_update,
    consistent_solid_tangent,
)


def test_law38_single_curve_yield_lookup():
    """Verify single-curve yield stress lookup and slope."""
    curve = ([0.0, 0.1, 0.5], [100.0, 150.0, 250.0])
    mat_def = {
        "id": 1,
        "MAT_E": 10000.0,
        "MAT_NU": 0.3,
        "yield_curves": [curve],
    }
    mat = build_law38(mat_def)

    # At eps_p = 0
    y0, h0 = calc_yield_stress(mat, eps_p=0.0)
    assert math.isclose(y0, 100.0, rel_tol=1e-6)
    assert math.isclose(h0, 500.0, rel_tol=1e-5)  # (150-100)/0.1

    # Mid-segment: eps_p = 0.05
    y_mid, h_mid = calc_yield_stress(mat, eps_p=0.05)
    assert math.isclose(y_mid, 125.0, rel_tol=1e-6)
    assert math.isclose(h_mid, 500.0, rel_tol=1e-5)

    # Second segment: eps_p = 0.3
    y_seg2, h_seg2 = calc_yield_stress(mat, eps_p=0.3)
    # y = 150 + (0.3 - 0.1) * (250-150)/0.4 = 150 + 0.2 * 250 = 200
    assert math.isclose(y_seg2, 200.0, rel_tol=1e-6)
    assert math.isclose(h_seg2, 250.0, rel_tol=1e-5)


def test_law38_multi_curve_strain_rate_interpolation():
    """Verify strain rate interpolation between multiple rate-dependent curves."""
    # Curve at rate = 0
    c0 = ([0.0, 0.2], [100.0, 140.0])
    # Curve at rate = 100
    c1 = ([0.0, 0.2], [150.0, 210.0])

    mat_def = {
        "id": 2,
        "MAT_E": 10000.0,
        "MAT_NU": 0.3,
        "yield_curves": [c0, c1],
        "rates": [0.0, 100.0],
    }
    mat = build_law38(mat_def)

    # At rate = 0
    y_r0, _ = calc_yield_stress(mat, eps_p=0.1, eps_p_dot=0.0)
    assert math.isclose(y_r0, 120.0, rel_tol=1e-5)

    # At rate = 100
    y_r100, _ = calc_yield_stress(mat, eps_p=0.1, eps_p_dot=100.0)
    assert math.isclose(y_r100, 180.0, rel_tol=1e-5)

    # At rate = 50 (linear interpolation: 50% between 120 and 180 = 150)
    y_r50, _ = calc_yield_stress(mat, eps_p=0.1, eps_p_dot=50.0)
    assert math.isclose(y_r50, 150.0, rel_tol=1e-5)

    # Below min rate: clamped to c0
    y_neg, _ = calc_yield_stress(mat, eps_p=0.1, eps_p_dot=-10.0)
    assert math.isclose(y_neg, 120.0, rel_tol=1e-5)

    # Above max rate: clamped to c1
    y_high, _ = calc_yield_stress(mat, eps_p=0.1, eps_p_dot=500.0)
    assert math.isclose(y_high, 180.0, rel_tol=1e-5)


def test_law38_temperature_dependent_yield():
    """Verify optional temperature dependence scaling factor."""
    curve = ([0.0, 0.5], [200.0, 300.0])
    # Thermal scaling: factor 1.0 at 300K, 0.75 at 400K, 0.5 at 500K
    temp_curve = ([300.0, 400.0, 500.0], [1.0, 0.75, 0.5])

    mat_def = {
        "id": 3,
        "MAT_E": 10000.0,
        "MAT_NU": 0.3,
        "yield_curves": [curve],
        "fun_temp": temp_curve,
    }
    mat = build_law38(mat_def)

    # Reference temp 300K: factor 1.0 -> y = 200
    y_300, _ = calc_yield_stress(mat, eps_p=0.0, temp=300.0)
    assert math.isclose(y_300, 200.0, rel_tol=1e-5)

    # Elevated temp 400K: factor 0.75 -> y = 150
    y_400, _ = calc_yield_stress(mat, eps_p=0.0, temp=400.0)
    assert math.isclose(y_400, 150.0, rel_tol=1e-5)

    # Elevated temp 500K: factor 0.50 -> y = 100
    y_500, _ = calc_yield_stress(mat, eps_p=0.0, temp=500.0)
    assert math.isclose(y_500, 100.0, rel_tol=1e-5)


def test_law38_radial_return_below_and_above_yield():
    """Verify radial return mapping: elastic trial below yield, plastic return above yield."""
    curve = ([0.0, 0.1], [100.0, 150.0])
    mat_def = {
        "id": 4,
        "MAT_E": 10000.0,
        "MAT_NU": 0.25,
        "yield_curves": [curve],
    }
    mat = build_law38(mat_def)

    # Elastic trial below yield (von Mises = 50 MPa < 100 MPa)
    sig_elastic = np.array([50.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    deps = np.zeros(6)
    sig_out, epsp_out, q_out = radial_return_mapping(
        mat, sig_elastic, deps, epsp_old=0.0, dt=1e-3
    )
    assert math.isclose(float(epsp_out), 0.0, abs_tol=1e-9)
    assert math.isclose(q_out, 50.0, rel_tol=1e-5)
    assert np.allclose(sig_out, sig_elastic)

    # Plastic trial above yield: pure shear trial tau_xy = 100 MPa -> Q = sqrt(3)*100 = 173.2 MPa
    sig_plastic = np.array([0.0, 0.0, 0.0, 100.0, 0.0, 0.0])
    sig_out_p, epsp_out_p, q_out_p = radial_return_mapping(
        mat, sig_plastic, deps, epsp_old=0.0, dt=1e-3
    )
    # Plastic strain must have accumulated
    assert float(epsp_out_p) > 0.0
    # Equivalent stress must equal the yield stress at the updated plastic strain
    expected_yield, _ = calc_yield_stress(mat, float(epsp_out_p))
    assert math.isclose(float(q_out_p), expected_yield, rel_tol=1e-5)
    # Shear stress should be scaled down proportionally
    assert sig_out_p[3] < 100.0
    assert math.isclose(sig_out_p[3] * math.sqrt(3.0), expected_yield, rel_tol=1e-5)


def test_law38_viscoplastic_solid_update():
    """Verify solid_update executes radial return when viscoplastic mode is active."""
    curve = ([0.0, 0.2], [120.0, 160.0])
    mat_def = {
        "id": 5,
        "MAT_E": 20000.0,
        "MAT_NU": 0.3,
        "MAT_RHO": 1.5,
        "viscoplastic": True,
        "yield_curves": [curve],
    }
    mat = build_law38(mat_def)

    # Apply shear strain increment deps_xy = 0.02 -> trial tau = G*0.02 = 7692*0.02 = 153.8 MPa -> Q = 266.4 MPa > 120 MPa
    sig0 = np.zeros(6, dtype=float)
    deps = np.array([0.0, 0.0, 0.0, 0.02, 0.0, 0.0])

    sig_new, epsp_new, c_sound = solid_update(mat, sig0, deps, epsp=np.zeros(1), dt=1e-4)

    assert float(epsp_new) > 0.0
    expected_yield, _ = calc_yield_stress(mat, float(epsp_new), eps_p_dot=float(epsp_new)/1e-4)
    q_actual = math.sqrt(3.0 * sig_new[3]**2)
    assert math.isclose(q_actual, expected_yield, rel_tol=1e-4)
    assert float(c_sound) > 0.0


def test_law38_consistent_tangent():
    """Verify consistent solid tangent calculation for viscoplastic mode."""
    curve = ([0.0, 0.2], [120.0, 160.0])
    mat_def = {
        "id": 6,
        "MAT_E": 20000.0,
        "MAT_NU": 0.3,
        "viscoplastic": True,
        "yield_curves": [curve],
    }
    mat = build_law38(mat_def)

    sig = np.zeros(6, dtype=float)
    deps = np.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    c_tan = consistent_solid_tangent(mat, sig=sig, deps=deps, dt=1e-4)
    assert c_tan.shape == (1, 6, 6)
    # Check symmetry C_ijkl = C_klij
    assert np.allclose(c_tan[0], c_tan[0].T, atol=1e-6)
    # Elastic longitudinal modulus C11 should be positive
    assert c_tan[0, 0, 0] > 0.0

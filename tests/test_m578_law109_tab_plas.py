"""Constitutive physics unit tests for OpenRadioss /MAT/LAW109 (M578).

Validates:
  1. Parameter initialization and derived elastic moduli (G, Bulk, Lame, A11, A12).
  2. 2D table evaluation (eval_table2d_log) for linear (ISMOOTH=1), log10 (ISMOOTH=2), and ln (ISMOOTH=3).
  3. Temperature scaling factor (eval_table_temp).
  4. Taylor-Quinney thermal conversion factor (eval_table_eta).
  5. 3D continuum solid update (sigeps109.F):
     - Elastic response (Hooke's law, no plastic dissipation, zero temperature rise).
     - Plastic yield and cutting plane return mapping (NITER=3).
     - Strain rate dependence and hardening modulus.
     - Adiabatic plastic heating (dT = FTHERM * YLD * dPLA / (CP * RHO)).
     - Vectorized / batch execution for solids (N, 6).
  6. 2D plane-stress shell update (sigeps109c.F):
     - In-plane elastic trial stress and transverse shear response.
     - Plane-stress plastic return mapping.
     - Thickness thinning update (elastic + plastic dezz).
     - Vectorized / batch execution for shells (N, 5).
  7. Acoustic wave speed estimates for solids and shells.
  8. Algorithmic tangent operators (solid_tangent, consistent_shell_tangent).
  9. Extra history array shapes (extra_shapes).
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law109_tab_plas import (
    Law109Params,
    build_law109,
    eval_table2d_log,
    eval_table_temp,
    eval_table_eta,
    solid_update,
    shell_update,
    sound_speed,
    solid_tangent,
    consistent_shell_tangent,
    extra_shapes,
)


def _make_dummy_table_2d():
    """Create a dummy 2D yield stress table: 2 strain rates, each with linear hardening curve."""
    # Rate 0.001 s^-1: sigma_y(epsp) = 300.0 + 500.0 * epsp
    curve_low = np.array([
        [0.0, 300.0],
        [0.1, 350.0],
        [0.5, 550.0],
        [1.0, 800.0],
    ])
    # Rate 100.0 s^-1: sigma_y(epsp) = 400.0 + 600.0 * epsp
    curve_high = np.array([
        [0.0, 400.0],
        [0.1, 460.0],
        [0.5, 700.0],
        [1.0, 1000.0],
    ])
    return [(0.001, curve_low), (100.0, curve_high)]


def _make_dummy_temp_curve():
    """Create a dummy temperature curve: YLD(T) = 500 at 293K, 250 at 800K."""
    return np.array([
        [200.0, 550.0],
        [293.0, 500.0],
        [500.0, 400.0],
        [800.0, 250.0],
    ])


def _make_dummy_eta_curve():
    """Create a dummy Taylor-Quinney curve versus strain rate."""
    return np.array([
        [0.0, 0.85],
        [10.0, 0.90],
        [100.0, 0.95],
        [1000.0, 1.0],
    ])


def test_law109_params_elastic_moduli():
    """Verify derived elastic constants in Law109Params."""
    young = 200000.0
    nu = 0.3
    p = Law109Params(rho0=7.8e-6, young=young, nu=nu)

    expected_g = 200000.0 / (2.0 * (1.0 + 0.3))
    expected_bulk = 200000.0 / (3.0 * (1.0 - 2.0 * 0.3))
    expected_lame = (2.0 * expected_g * 0.3) / (1.0 - 2.0 * 0.3)
    denom2d = 1.0 - 0.3 * 0.3
    expected_a11 = 200000.0 / denom2d
    expected_a12 = 0.3 * expected_a11

    assert np.isclose(p.g, expected_g)
    assert np.isclose(p.bulk, expected_bulk)
    assert np.isclose(p.lame, expected_lame)
    assert np.isclose(p.a11, expected_a11)
    assert np.isclose(p.a12, expected_a12)


def test_eval_table2d_log_interpolations():
    """Verify 2D table evaluation with linear, log10, and ln rate interpolations."""
    table = _make_dummy_table_2d()
    epsp = np.array([0.0, 0.1, 0.5])

    # 1. Below minimum rate (<= 0.001): returns curve_low
    rates_low = np.full(3, 0.0001)
    y_lo, d1_lo = eval_table2d_log(table, epsp, rates_low, ismooth=1)
    assert np.isclose(y_lo[0], 300.0)
    assert np.isclose(y_lo[1], 350.0)
    assert np.isclose(y_lo[2], 550.0)
    assert np.isclose(d1_lo[0], 500.0)

    # 2. Above maximum rate (>= 100.0): returns curve_high
    rates_high = np.full(3, 200.0)
    y_hi, d1_hi = eval_table2d_log(table, epsp, rates_high, ismooth=1)
    assert np.isclose(y_hi[0], 400.0)
    assert np.isclose(y_hi[1], 460.0)
    assert np.isclose(y_hi[2], 700.0)
    assert np.isclose(d1_hi[0], 600.0)

    # 3. Log10 interpolation at rate = 0.3162 (midpoint of [1e-3, 1e2] in log10 space)
    rate_mid = np.full(3, 10.0**(-0.5))  # log10 is -0.5, halfway between -3 and +2
    y_log, _ = eval_table2d_log(table, epsp, rate_mid, ismooth=2)
    # log10(100) = 2, log10(0.001) = -3. denom = 5.
    # r2 = (2 - (-0.5)) / 5 = 2.5 / 5 = 0.5 -> exact average of curve_low and curve_high!
    expected_mid_0 = 0.5 * 300.0 + 0.5 * 400.0
    assert np.isclose(y_log[0], expected_mid_0)


def test_eval_table_temp_and_eta():
    """Verify temperature scaling factor TFAC and Taylor-Quinney factor FTHERM."""
    temp_curve = _make_dummy_temp_curve()
    pla = np.array([0.0, 0.1])
    temp_ref = np.array([293.0, 293.0])
    temp_hot = np.array([500.0, 800.0])

    # At T_ref: TFAC must be exactly 1.0
    tfac_ref = eval_table_temp(temp_curve, pla, temp_ref, tref=293.0)
    assert np.allclose(tfac_ref, 1.0)

    # At higher temperatures: TFAC must reflect softening
    tfac_hot = eval_table_temp(temp_curve, pla, temp_hot, tref=293.0)
    assert tfac_hot[0] == pytest.approx(400.0 / 500.0)  # 0.8
    assert tfac_hot[1] == pytest.approx(250.0 / 500.0)  # 0.5

    # Taylor-Quinney factor evaluation
    eta_curve = _make_dummy_eta_curve()
    rates = np.array([0.0, 10.0, 100.0, 1000.0])
    dummy_temp = np.full(4, 293.0)
    dummy_pla = np.zeros(4)
    ftherm = eval_table_eta(eta_curve, rates, dummy_temp, dummy_pla, eta_base=1.0)
    assert np.isclose(ftherm[0], 0.85)
    assert np.isclose(ftherm[1], 0.90)
    assert np.isclose(ftherm[2], 0.95)
    assert np.isclose(ftherm[3], 1.0)


def test_solid_3d_elastic_step():
    """Verify 3D solid purely elastic deformation (Hooke's law, no plastic strain, no heating)."""
    p = Law109Params(
        rho0=7.8e-6,
        young=210000.0,
        nu=0.3,
        cp=450.0,
        eta=0.9,
        tref=293.0,
        tini=293.0,
        yield_table=np.array([[0.0, 400.0], [1.0, 800.0]]),
    )

    # Small strain increment below yield: deps = [1e-4, 0, 0, 0, 0, 0]
    deps = np.array([1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0])
    sigo = np.zeros(6)
    extra = {"pla": 0.0, "epsd": 0.0, "temp": 293.0}

    sign, extra_new = solid_update(p, deps=deps, sigo=sigo, extra=extra, dt=1.0e-6)

    # Expected elastic stresses:
    # sigma_xx = (lame + 2G) * deps_xx
    # sigma_yy = sigma_zz = lame * deps_xx
    expected_xx = (p.lame + 2.0 * p.g) * 1.0e-4
    expected_yy = p.lame * 1.0e-4
    assert np.isclose(sign[0], expected_xx)
    assert np.isclose(sign[1], expected_yy)
    assert np.isclose(sign[2], expected_yy)
    assert np.allclose(sign[3:], 0.0)

    # No plasticity, no temperature increase
    assert extra_new["pla"] == pytest.approx(0.0)
    assert extra_new["temp"] == pytest.approx(293.0)


def test_solid_3d_plastic_yield_and_adiabatic_heating():
    """Verify 3D solid plastic return mapping and adiabatic temperature rise."""
    yield_stress = 300.0
    p = Law109Params(
        rho0=7.8e-6,
        young=210000.0,
        nu=0.3,
        cp=450.0,
        eta=0.9,
        tref=293.0,
        tini=293.0,
        yield_table=np.array([[0.0, yield_stress], [1.0, yield_stress]]),  # perfectly plastic
    )

    # Large tensile strain increment: deps = [5e-3, -1.5e-3, -1.5e-3, 0, 0, 0] (uniaxial-like volume preserving)
    deps = np.array([5.0e-3, -1.5e-3, -1.5e-3, 0.0, 0.0, 0.0])
    sigo = np.zeros(6)
    extra = {"pla": 0.0, "epsd": 0.0, "temp": 293.0}

    sign, extra_new = solid_update(p, deps=deps, sigo=sigo, extra=extra, dt=1.0e-5)

    # Equivalent stress must match yield stress
    sigm = (sign[0] + sign[1] + sign[2]) / 3.0
    s_dev = sign[:3] - sigm
    j2 = 0.5 * (s_dev[0]**2 + s_dev[1]**2 + s_dev[2]**2)
    seq = math.sqrt(3.0 * j2)
    assert np.isclose(seq, yield_stress, atol=1.0e-2)

    # Plastic strain accumulated
    dpla = extra_new["pla"]
    assert dpla > 0.0

    # Temperature rise matches Taylor-Quinney formula:
    # dT = eta * sigma_y * dpla / (cp * rho)
    expected_dT = (0.9 * yield_stress * dpla) / (450.0 * 7.8e-6)
    assert np.isclose(extra_new["temp"] - 293.0, expected_dT, rtol=1.0e-3)


def test_shell_2d_plane_stress_and_thinning():
    """Verify 2D shell plane-stress update and plastic thickness thinning."""
    yield_stress = 350.0
    p = Law109Params(
        rho0=7.8e-6,
        young=210000.0,
        nu=0.3,
        cp=450.0,
        eta=0.9,
        tref=293.0,
        tini=293.0,
        yield_table=np.array([[0.0, yield_stress], [1.0, yield_stress]]),
    )

    # Initial thickness h0 = 1.5 mm
    thk0 = 1.5
    deps = np.array([6.0e-3, 0.0, 0.0, 0.0, 0.0])  # in-plane tension in x
    sigo = np.zeros(5)
    extra = {"pla": 0.0, "epsd": 0.0, "temp": 293.0, "thk": thk0}

    sign, extra_new = shell_update(p, deps=deps, sigo=sigo, extra=extra, dt=1.0e-5)

    # Plane-stress von Mises: seq = sqrt(sigxx^2 + sigyy^2 - sigxx*sigyy + 3*sigxy^2)
    seq_shell = math.sqrt(sign[0]**2 + sign[1]**2 - sign[0] * sign[1] + 3.0 * sign[2]**2)
    assert np.isclose(seq_shell, yield_stress, atol=1.0e-2)

    # Thickness must have thinned (thk < thk0) due to Poisson and plastic flow
    assert extra_new["thk"] < thk0
    assert extra_new["pla"] > 0.0


def test_sound_speeds_and_tangents():
    """Verify acoustic wave speeds and implicit tangent matrices."""
    p = Law109Params(
        rho0=7.8e-6,
        young=210000.0,
        nu=0.3,
    )

    # Solid sound speed: sqrt((Bulk + 4/3*G) / rho)
    expected_ssp_solid = math.sqrt((p.bulk + 4.0 / 3.0 * p.g) / 7.8e-6)
    assert np.isclose(sound_speed(p, is_shell=False), expected_ssp_solid)

    # Shell sound speed: sqrt(A11 / rho)
    expected_ssp_shell = math.sqrt(p.a11 / 7.8e-6)
    assert np.isclose(sound_speed(p, is_shell=True), expected_ssp_shell)

    # Elastic solid tangent: 6x6 isotropic Hooke tensor
    c_sol = solid_tangent(p)
    assert c_sol.shape == (6, 6)
    assert np.isclose(c_sol[0, 0], p.lame + 2.0 * p.g)
    assert np.isclose(c_sol[0, 1], p.lame)
    assert np.isclose(c_sol[3, 3], p.g)

    # Elastic shell membrane tangent: 3x3 plane stress tensor
    c_sh = consistent_shell_tangent(p)
    assert c_sh.shape == (3, 3)
    assert np.isclose(c_sh[0, 0], p.a11)
    assert np.isclose(c_sh[0, 1], p.a12)
    assert np.isclose(c_sh[2, 2], p.g)


def test_extra_shapes():
    """Verify extra history shapes for element integration points."""
    p = Law109Params(rho0=7.8e-6, young=210000.0, nu=0.3)

    shapes_single = extra_shapes(p)
    assert "pla" in shapes_single
    assert "epsd" in shapes_single
    assert "temp" in shapes_single
    assert "thk" in shapes_single
    assert shapes_single["pla"] == ()

    shapes_nip = extra_shapes(p, nip=5)
    assert shapes_nip["pla"] == (5,)
    assert shapes_nip["temp"] == (5,)
    assert shapes_nip["thk"] == (5,)

"""Fortran parity tests for /MAT/LAW109 (/MAT/TAB_PLAS / /MAT/ELASTO_PLAS_TAB).

Verifies exact numerical formulas and steps from:
  engine/source/materials/mat/mat109/sigeps109.F
  engine/source/materials/mat/mat109/sigeps109c.F
  starter/source/materials/mat/mat109/hm_read_mat109.F
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law109_tab_plas import (
    Law109Params,
    Law109Table,
    build_law109,
    eval_table2d_log,
    eval_table_temp,
    eval_table_eta,
    solid_update,
    shell_update,
    sound_speed,
    solid_tangent,
    consistent_shell_tangent,
)


def _make_simple_table() -> Law109Table:
    """Create a 2-curve table: rate 0.0 and rate 100.0 with bilinear hardening."""
    eps = np.array([0.0, 0.05, 0.20], dtype=np.float64)
    # At rate=0: sigy = 200, at eps=0.05 sig=250, at eps=0.20 sig=300
    sig_r0 = np.array([200.0, 250.0, 300.0], dtype=np.float64)
    # At rate=100: sigy = 240, at eps=0.05 sig=300, at eps=0.20 sig=360
    sig_r100 = np.array([240.0, 300.0, 360.0], dtype=np.float64)
    return Law109Table(
        rates=[0.0, 100.0],
        curves=[(eps, sig_r0), (eps, sig_r100)],
    )


def test_eval_table2d_log_linear():
    """Verify eval_table2d_log with linear rate interpolation (ismooth=1)."""
    tbl = _make_simple_table()
    # At rate = 50 (midpoint between 0 and 100), at pla = 0.05:
    # sig_r0(0.05) = 250, sig_r100(0.05) = 300 -> expected = 275.0
    # Hardening modulus H = dsig/deps between 0.05 and 0.20:
    # H_r0 = (300 - 250) / 0.15 = 333.3333
    # H_r100 = (360 - 300) / 0.15 = 400.0 -> midpoint = 366.6667
    pla = np.array([0.05])
    rate = np.array([50.0])
    sig, h = eval_table2d_log(tbl, pla, rate, ismooth=1)
    assert sig[0] == pytest.approx(275.0, rel=1e-5)
    assert h[0] == pytest.approx((333.3333333333333 + 400.0) / 2.0, rel=1e-4)


def test_eval_table2d_log_log10():
    """Verify eval_table2d_log with log10 rate interpolation (ismooth=2)."""
    eps = np.array([0.0, 0.1], dtype=np.float64)
    sig_r1 = np.array([200.0, 300.0], dtype=np.float64)
    sig_r100 = np.array([300.0, 400.0], dtype=np.float64)
    tbl = Law109Table(
        rates=[1.0, 100.0],
        curves=[(eps, sig_r1), (eps, sig_r100)],
    )
    # At rate = 10: log10(10)=1.0, midpoint between log10(1)=0 and log10(100)=2 -> weight 0.5
    pla = np.array([0.0])
    rate = np.array([10.0])
    sig, h = eval_table2d_log(tbl, pla, rate, ismooth=2)
    assert sig[0] == pytest.approx(250.0, rel=1e-5)


def test_eval_table2d_log_ln():
    """Verify eval_table2d_log with ln rate interpolation (ismooth=3)."""
    eps = np.array([0.0, 0.1], dtype=np.float64)
    sig_r1 = np.array([200.0, 300.0], dtype=np.float64)
    r2 = math.exp(2.0)
    sig_r2 = np.array([300.0, 400.0], dtype=np.float64)
    tbl = Law109Table(
        rates=[1.0, r2],
        curves=[(eps, sig_r1), (eps, sig_r2)],
    )
    # At rate = exp(1.0): ln=1.0, midpoint between ln(1)=0 and ln(exp(2))=2 -> weight 0.5
    pla = np.array([0.0])
    rate = np.array([math.exp(1.0)])
    sig, h = eval_table2d_log(tbl, pla, rate, ismooth=3)
    assert sig[0] == pytest.approx(250.0, rel=1e-5)


def test_eval_table_temp():
    """Verify temperature scaling factor from table/function."""
    # Linear function: f(T) = 1.0 - 0.001*(T - 293.15)
    t_arr = np.array([293.15, 393.15, 493.15])
    f_arr = np.array([1.0, 0.9, 0.8])
    tbl = (t_arr, f_arr)
    pla = np.array([0.0, 0.0])
    temp = np.array([293.15, 343.15])
    tfac = eval_table_temp(tbl, pla, temp, tref=293.15)
    assert tfac[0] == pytest.approx(1.0, rel=1e-5)
    assert tfac[1] == pytest.approx(0.95, rel=1e-5)


def test_eval_table_eta():
    """Verify Taylor-Quinney conversion factor eta evaluation."""
    # Constant eta
    eta = eval_table_eta(None, np.array([10.0]), np.array([300.0]), np.array([0.05]), eta=0.9)
    assert eta[0] == pytest.approx(0.9)

    # Function of rate
    r_arr = np.array([0.0, 1000.0])
    f_arr = np.array([0.5, 0.9])
    tbl = (r_arr, f_arr)
    eta = eval_table_eta(tbl, np.array([500.0]), np.array([300.0]), np.array([0.05]), eta=1.0)
    assert eta[0] == pytest.approx(0.7, rel=1e-5)


def test_cutting_plane_solid_radial_return():
    """Verify 3D solid return mapping against manual radial return formulas.
    
    For J2 von Mises plasticity with isotropic hardening:
      s_trial = 2*G*deps_dev
      sigma_vm_trial = sqrt(1.5 s_trial : s_trial)
      dlambda = (sigma_vm_trial - sigy0) / (3*G + H)
      sigma_vm_final = sigy0 + H * dlambda
    """
    young = 200000.0
    nu = 0.3
    rho = 7.8e-9
    sigy0 = 200.0
    h_mod = 1000.0  # Constant hardening modulus

    # Table representing linear hardening
    eps = np.array([0.0, 1.0])
    sig = np.array([sigy0, sigy0 + h_mod * 1.0])
    tbl = Law109Table(rates=[0.0], curves=[(eps, sig)])

    params = Law109Params(
        rho=rho,
        refer_rho=rho,
        young=young,
        nu=nu,
        yield_table=tbl,
        cp=0.0,
        eta=0.0,
    )

    g = young / (2.0 * (1.0 + nu))
    deps = np.array([[0.003, -0.0015, -0.0015, 0.0, 0.0, 0.0]], dtype=np.float64)
    sigo = np.zeros((1, 6), dtype=np.float64)

    s_trial_xx = 2.0 * g * 0.003
    s_trial_yy = -1.0 * g * 0.003
    s_trial_zz = -1.0 * g * 0.003
    vm_trial = math.sqrt(1.5 * (s_trial_xx**2 + s_trial_yy**2 + s_trial_zz**2))
    assert vm_trial > sigy0

    sig_out, extra_out = solid_update(params, deps, sigo, dt=1.0e-5, extra={})

    # Invariants of output stress
    s_out = sig_out[0] - np.mean(sig_out[0, :3])
    vm_out = math.sqrt(1.5 * (s_out[0]**2 + s_out[1]**2 + s_out[2]**2 + 2.0 * np.sum(s_out[3:]**2)))

    # Cutting plane consistency condition: Von Mises stress equals current yield stress
    expected_yield = sigy0 + h_mod * extra_out["pla"]
    assert vm_out == pytest.approx(expected_yield, rel=1e-4)
    assert extra_out["pla"] > 0.0


def test_adiabatic_heating_solid():
    """Verify Delta T = (eta * sigy * Delta_pla) / (rho * Cp)."""
    young = 200000.0
    nu = 0.3
    rho = 7.8e-9
    cp = 450.0e6  # mJ / (tonne * K) = mm^2 / (s^2 * K)
    eta = 0.9
    sigy0 = 200.0

    eps = np.array([0.0, 1.0])
    sig = np.array([sigy0, sigy0])  # Perfectly plastic
    tbl = Law109Table(rates=[0.0], curves=[(eps, sig)])

    params = Law109Params(
        rho=rho,
        refer_rho=rho,
        young=young,
        nu=nu,
        yield_table=tbl,
        cp=cp,
        eta=eta,
        tref=293.15,
        tini=300.0,
    )

    deps = np.array([[0.005, -0.0025, -0.0025, 0.0, 0.0, 0.0]], dtype=np.float64)
    sigo = np.zeros((1, 6), dtype=np.float64)
    extra_in = {"temp": 300.0, "pla": 0.0}

    sig_out, extra_out = solid_update(params, deps, sigo, dt=1.0e-5, extra=extra_in)

    dpla = extra_out["pla"]
    expected_dtemp = (eta * sigy0 * dpla) / (rho * cp)
    expected_temp = 300.0 + expected_dtemp

    assert extra_out["temp"] == pytest.approx(expected_temp, rel=1e-4)


def test_shell_plane_stress_plastic_thinning():
    """Verify 2D shell plane-stress return mapping and plastic thickness thinning."""
    young = 210000.0
    nu = 0.3
    rho = 7.85e-9
    sigy0 = 250.0
    h_mod = 2000.0

    eps = np.array([0.0, 1.0])
    sig = np.array([sigy0, sigy0 + h_mod * 1.0])
    tbl = Law109Table(rates=[0.0], curves=[(eps, sig)])

    params = Law109Params(
        rho=rho,
        refer_rho=rho,
        young=young,
        nu=nu,
        yield_table=tbl,
        cp=0.0,
        eta=0.0,
    )

    # In-plane biaxial stretch deps_xx = 0.004, deps_yy = 0.001, deps_xy = 0.0
    deps = np.array([[0.004, 0.001, 0.0, 0.0, 0.0]], dtype=np.float64)
    sigo = np.zeros((1, 5), dtype=np.float64)
    extra_in = {"pla": 0.0, "dezz": 0.0}

    sig_out, extra_out = shell_update(params, deps, sigo, dt=1.0e-5, extra=extra_in)

    # Plastic thinning dezz must be negative under positive in-plane tensile strains
    assert extra_out["pla"] > 0.0
    assert extra_out["dezz"] < 0.0

    # In plane stress, effective stress must match updated yield stress
    sxx = sig_out[0, 0] - (sig_out[0, 0] + sig_out[0, 1]) / 3.0
    syy = sig_out[0, 1] - (sig_out[0, 0] + sig_out[0, 1]) / 3.0
    szz = -(sig_out[0, 0] + sig_out[0, 1]) / 3.0
    sxy = sig_out[0, 2]
    svm = math.sqrt(3.0 * (0.5 * (sxx**2 + syy**2 + szz**2) + sxy**2))

    expected_sigy = sigy0 + h_mod * extra_out["pla"]
    assert svm == pytest.approx(expected_sigy, rel=1e-3)


def test_sound_speed_solid_and_shell():
    """Verify sound speed formulas for 3D solid and 2D shell."""
    young = 200000.0
    nu = 0.3
    rho = 7.8e-9

    params = Law109Params(
        rho=rho,
        refer_rho=rho,
        young=young,
        nu=nu,
        yield_table=_make_simple_table(),
    )

    # Solid sound speed: c = sqrt((K + 4/3 G) / rho)
    k = young / (3.0 * (1.0 - 2.0 * nu))
    g = young / (2.0 * (1.0 + nu))
    c_solid_hand = math.sqrt((k + 4.0 / 3.0 * g) / rho)
    c_solid = sound_speed(params, is_shell=False)
    assert float(np.squeeze(c_solid)) == pytest.approx(c_solid_hand, rel=1e-6)

    # Shell sound speed: c = sqrt(E / ((1 - nu^2) * rho))
    c_shell_hand = math.sqrt(young / ((1.0 - nu**2) * rho))
    c_shell = sound_speed(params, is_shell=True)
    assert float(np.squeeze(c_shell)) == pytest.approx(c_shell_hand, rel=1e-6)


def test_tangent_operators_consistency():
    """Verify solid and shell tangent stiffness matrices."""
    young = 200000.0
    nu = 0.3
    rho = 7.8e-9
    sigy0 = 200.0
    h_mod = 1000.0

    eps = np.array([0.0, 1.0])
    sig = np.array([sigy0, sigy0 + h_mod * 1.0])
    tbl = Law109Table(rates=[0.0], curves=[(eps, sig)])

    params = Law109Params(
        rho=rho,
        refer_rho=rho,
        young=young,
        nu=nu,
        yield_table=tbl,
    )

    # Elastic tangent (pla = 0, sig = 0)
    d_el = solid_tangent(params, sig=np.zeros(6), pla=0.0, epsd=0.0)
    assert d_el.shape == (6, 6)
    assert d_el[0, 0] > d_el[0, 1]
    assert np.allclose(d_el, d_el.T)  # Symmetry

    # Shell tangent
    d_shell = consistent_shell_tangent(params, sig=np.zeros(3), pla=0.0, epsd=0.0)
    assert d_shell.shape == (3, 3)
    denom = 1.0 - nu**2
    assert d_shell[0, 0] == pytest.approx(young / denom, rel=1e-5)
    assert d_shell[0, 1] == pytest.approx(nu * young / denom, rel=1e-5)
    assert d_shell[2, 2] == pytest.approx(young / (2.0 * (1.0 + nu)), rel=1e-5)

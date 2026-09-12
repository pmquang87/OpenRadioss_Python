"""Fortran parity test for /MAT/LAW105 (/MAT/POWDER_BURN) against OpenRadioss sigeps105.F.

Verifies exact numerical formulas and steps from:
  engine/source/materials/mat/mat105/sigeps105.F
  starter/source/materials/mat/mat105/m105init.F
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law105_powder_burn import (
    PowderBurnParams,
    build_law105,
    solid_update,
    solid_update_single,
    sound_speed,
)
from pyradioss.model.entities import MaterialLaw105


def test_fortran_parity_unburnt_hydrostatic():
    """Verify unburnt state reproduces exact acoustic and hydrostatic equations."""
    rho0 = 1.65e-6
    bulk = 14000.0
    p0 = 0.2
    psh = 0.05
    mat = PowderBurnParams(
        rho0=rho0,
        bulk=bulk,
        p0=p0,
        psh=psh,
        gas_d=0.5e-6,
        gas_eg=4200.0,
        gas_gam=1.3,
        gr=0.07,
        c=0.8,
        alpha=1.0,
        c1=2500.0,
        compac=0.93,
    )

    sig = np.zeros(6, dtype=np.float64)
    deps = np.zeros(6, dtype=np.float64)
    extra = {
        "initial_step": True,
        "volume": 100.0,
        "time": 0.0,
        "t_burn": 10.0,  # Not yet ignited
    }

    sig_out, epsp, ssp = solid_update_single(mat, sig, deps, dt=1e-5, extra=extra)

    # In step 0 (initial), P = P0 = 0.2
    expected_p = p0
    expected_sig = -(expected_p - psh)
    assert np.allclose(sig_out[:3], expected_sig)
    assert np.allclose(sig_out[3:], 0.0)
    assert ssp == pytest.approx(math.sqrt(bulk / rho0))


def test_fortran_parity_flame_front_propagation():
    """Verify flame front ignition velocity formula bfrac = 2/3 * C1 * (t - tb) / dx."""
    mat = PowderBurnParams(
        rho0=1.65e-6,
        bulk=14000.0,
        p0=0.1,
        psh=0.0,
        gas_d=0.5e-6,
        gas_eg=4000.0,
        gas_gam=1.3,
        gr=0.0,  # Turn off grain burn to isolate flame front
        c=1.0,
        alpha=1.0,
        c1=3000.0,  # Ignition velocity = 3000 mm/s (or m/s)
        compac=0.93,
    )

    # dx = 10 mm, t_burn = 0.001 s, t = 0.002 s
    # (t - tb) = 0.001 s
    # C1 * (t - tb) = 3000 * 0.001 = 3.0
    # bfrac = (2/3) * 3.0 / 10.0 = 2.0 / 10.0 = 0.20
    dx = 10.0
    volume = dx ** 3  # 1000 mm^3
    t_burn = 0.001
    current_time = 0.002

    sig = np.zeros(6, dtype=np.float64)
    deps = np.zeros(6, dtype=np.float64)
    extra = {
        "time": current_time,
        "t_burn": t_burn,
        "volume": volume,
        "dx": dx,
        "uvar": np.array([0.1, 0.1, 1.65e-6, 0.0, 0.1, 1.0, 1.65e-3], dtype=np.float64),
    }

    sig_out, epsp, ssp = solid_update_single(mat, sig, deps, dt=1e-5, extra=extra)

    # bfrac should match exact analytical result 0.20
    assert extra["bfrac"] == pytest.approx(0.20, rel=1e-5)
    # Since f_burn was initialized to 1.0 and gr=0, total_bfrac = 0.20 * 1.0 = 0.20
    assert extra["total_bfrac"] == pytest.approx(0.20, rel=1e-5)


def test_fortran_parity_grain_burning_growth():
    """Verify grain burning kinetics delta_F = Gr * (1 - alpha * F)^C * fb(P) * dt."""
    gr = 0.05
    c_pow = 0.75
    alpha = 0.9
    scale_b = 2.5
    dt = 0.0001
    f_burn_old = 0.3

    mat = PowderBurnParams(
        rho0=1.65e-6,
        bulk=14000.0,
        p0=0.1,
        psh=0.0,
        gas_d=0.5e-6,
        gas_eg=4000.0,
        gas_gam=1.3,
        gr=gr,
        c=c_pow,
        alpha=alpha,
        c1=5000.0,
        scale_b=scale_b,
        compac=0.93,
    )

    # Hand-calculate expected delta_F
    f_term = 1.0 - alpha * f_burn_old  # 1.0 - 0.9 * 0.3 = 0.73
    d_f_hand = gr * (f_term ** c_pow) * scale_b * dt  # 0.05 * (0.73**0.75) * 2.5 * 1e-4
    f_burn_expected = f_burn_old + d_f_hand

    sig = np.zeros(6, dtype=np.float64)
    deps = np.zeros(6, dtype=np.float64)
    extra = {
        "time": 0.01,
        "t_burn": 0.0,  # already ignited, bfrac = 1.0
        "volume": 1.0,
        "dx": 0.01,
        "uvar": np.array([0.1, 0.1, 1.65e-6, 0.0, 0.1, f_burn_old, 1.65e-6], dtype=np.float64),
    }

    sig_out, epsp, ssp = solid_update_single(mat, sig, deps, dt=dt, extra=extra)

    uvar_out = extra["uvar"]
    assert uvar_out[5] == pytest.approx(f_burn_expected, rel=1e-5)


def test_fortran_parity_gas_exponential_eos():
    """Verify gas EOS equation Pg = P0 + rho_g * eg * exp(rho_g / D)."""
    p0 = 0.15
    gas_d = 0.45e-6
    gas_eg = 3800.0
    rho_g = 0.8e-6

    # Analytical Fortran formula:
    pg_expected = p0 + rho_g * gas_eg * math.exp(rho_g / gas_d)

    mat = PowderBurnParams(
        rho0=1.65e-6,
        bulk=14000.0,
        p0=p0,
        psh=0.0,
        gas_d=gas_d,
        gas_eg=gas_eg,
        gas_gam=1.25,
        gr=0.0,
        c=1.0,
        alpha=1.0,
        c1=5000.0,
        compac=0.93,
    )

    # Completely burnt: bfrac = 1.0, f_burn = 1.0
    vol = 100.0
    mass0 = rho_g * vol
    extra = {
        "time": 0.1,
        "t_burn": 0.0,
        "volume": vol,
        "dx": 0.01,
        "uvar": np.array([p0, p0, 1.65e-6, rho_g, p0, 1.0, mass0], dtype=np.float64),
    }

    sig = np.zeros(6, dtype=np.float64)
    deps = np.zeros(6, dtype=np.float64)
    sig_out, epsp, ssp = solid_update_single(mat, sig, deps, dt=1e-5, extra=extra)

    # In 100% gas phase, P = Pg
    assert extra["pressure"] == pytest.approx(pg_expected, rel=1e-3)
    assert np.allclose(sig_out[:3], -pg_expected, rtol=1e-3)

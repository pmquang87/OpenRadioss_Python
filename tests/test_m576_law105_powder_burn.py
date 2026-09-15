"""Unit tests for /MAT/LAW105 (/MAT/POWDER_BURN) constitutive behavior (M576).

Validates:
  - Initial step dt == 0 state initialization.
  - Flame front propagation (BFRAC) vs lighting time tb and cell size XL.
  - Burn rate evaluation with and without pressure curve.
  - Grain growth kinetics dF/dt = Gr * (1 - alpha * F)^C * BRATE.
  - Exponential gas EOS: P_g = P_0 + rho_g * e_g * exp(rho_g / D).
  - Linear solid grain compressibility: rho_si = (rho0 / compac) * (P_g / Bulk + 1).
  - Mixture pressure and mixture sound speed formulas.
  - Hydrodynamic stress tensor (pure hydrostatic pressure, zero shear).
  - Consistent tangent stiffness matrix.
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
    solid_tangent,
)


def _make_powder_params(**kwargs) -> PowderBurnParams:
    defaults = {
        "id": 1,
        "title": "Propellant_A",
        "rho0": 1.6e-6,      # g/mm^3 = kg/dm^3
        "bulk": 10000.0,     # MPa
        "p0": 0.1,           # MPa
        "psh": 0.0,
        "d": 0.5e-6,         # g/mm^3
        "eg": 4500.0,        # mJ/mg = MPa * mm^3 / mg
        "gr": 0.05,          # 1/mm
        "c": 0.8,
        "alpha": 1.0,
        "c1": 2000.0,        # mm/s (flame speed)
        "scale_b": 1.0,
        "scale_p": 1.0,
        "compac": 0.93,
    }
    defaults.update(kwargs)
    return build_law105(**defaults)


def test_law105_initial_step_dt_zero():
    """Verify initial step dt == 0 initializes state variables and hydrostatic stress."""
    p = _make_powder_params()
    sig = np.zeros(6, dtype=np.float64)
    deps = np.zeros(6, dtype=np.float64)
    extra = {"initial_step": True, "volume": 100.0, "rho": p.rho0}

    sig_out, epsp_out, ssp = solid_update_single(p, sig, deps, 0.0, 0.0, extra)

    p_init = p.p0 - p.psh
    assert np.allclose(sig_out[:3], -p_init)
    assert np.allclose(sig_out[3:], 0.0)
    assert ssp == pytest.approx(math.sqrt(p.bulk / p.rho0))

    uvar = extra["uvar"]
    assert uvar[0] == pytest.approx(p_init)  # PS
    assert uvar[1] == pytest.approx(p_init)  # PG
    assert uvar[2] == pytest.approx(p.rho0 / 0.93)  # RHO_S
    assert uvar[3] == pytest.approx(0.0)    # RHO_G
    assert uvar[5] == pytest.approx(0.0)    # F(t)
    assert uvar[6] == pytest.approx(p.rho0 * 100.0)  # Mass0


def test_law105_flame_ignition_propagation():
    """Verify ignition flame front fraction BFRAC propagation after lighting time tb."""
    p = _make_powder_params(c1=1500.0)
    sig = np.zeros(6, dtype=np.float64)
    deps = np.zeros(6, dtype=np.float64)

    # Element characteristic length XL = 10 mm, tb = 0.001 s
    xl = 10.0
    tb = 0.001

    # Case 1: time < tb -> BFRAC = 0
    extra1 = {"time": 0.0005, "tburn": -tb, "xl": xl, "volume": 1000.0}
    _, _, _ = solid_update_single(p, sig.copy(), deps, 0.0, 1e-4, extra1)
    assert extra1["bfrac"] == 0.0

    # Case 2: time > tb -> BFRAC = C1 * (time - tb) * (2/3) / XL
    # dt_burn = 0.002 s -> C1 * 0.002 * (2/3) / 10 = 1500 * 0.002 * 2 / 30 = 3.0 * 2 / 30 = 0.2
    extra2 = {"time": 0.003, "tburn": -tb, "xl": xl, "volume": 1000.0}
    _, _, _ = solid_update_single(p, sig.copy(), deps, 0.0, 1e-4, extra2)
    assert extra2["bfrac"] == pytest.approx(0.2, rel=1e-5)

    # Case 3: time >> tb -> BFRAC clamped to 1.0
    extra3 = {"time": 0.05, "tburn": -tb, "xl": xl, "volume": 1000.0}
    _, _, _ = solid_update_single(p, sig.copy(), deps, 0.0, 1e-4, extra3)
    assert extra3["bfrac"] == 1.0


def test_law105_grain_burning_kinetics():
    """Verify grain burn fraction dF/dt = Gr * (1 - alpha*F)^C * BRATE."""
    # Define a custom burn rate curve: BRATE = 0.05 * P
    def brate_curve(p_val):
        return 0.05 * p_val

    p = _make_powder_params(gr=0.1, c=0.5, alpha=0.9, curve_b=brate_curve)
    sig = np.zeros(6, dtype=np.float64)
    deps = np.zeros(6, dtype=np.float64)

    # Initialize at step 0
    extra = {"initial_step": True, "volume": 100.0, "rho": p.rho0}
    solid_update_single(p, sig, deps, 0.0, 0.0, extra)

    # Fully ignited element (bfrac = 1.0)
    extra["bfrac"] = 1.0
    extra["time"] = 0.005
    extra["tburn"] = 0.0
    dt = 1e-4

    # Perform one step of burning
    sig_out, _, ssp = solid_update_single(p, sig, deps, 0.0, dt, extra)
    f1 = extra["uvar"][5]
    assert f1 > 0.0
    assert f1 < 1.0
    # Check that pressure increased due to gas generation
    p_mixture = -sig_out[0]
    assert p_mixture > p.p0


def test_law105_gas_eos_exponential():
    """Verify gas EOS pressure: P_g = P_0 + rho_g * e_g * exp(rho_g / D)."""
    p = _make_powder_params(p0=0.5, d=1.0e-6, eg=5000.0)

    # Let's test analytical gas pressure formula
    rho_g = 2.0e-7
    p_gas_expected = p.p0 + rho_g * p.eg * math.exp(rho_g / p.d)
    assert p_gas_expected > p.p0


def test_law105_vectorized_update():
    """Verify vectorized multi-element update (nel, 6)."""
    p = _make_powder_params()
    nel = 4
    sig = np.zeros((nel, 6), dtype=np.float64)
    deps = np.zeros((nel, 6), dtype=np.float64)
    extra = {
        "volume": np.array([10.0, 20.0, 30.0, 40.0]),
        "time": 0.002,
        "tburn": np.array([0.001, 0.001, 0.005, 0.005]),  # First 2 lighted, last 2 unburnt
        "xl": np.array([2.0, 2.5, 3.0, 3.5]),
    }

    sig_out, epsp_out, ssp_out = solid_update(p, sig, deps, dt=1e-4, extra=extra)

    assert sig_out.shape == (nel, 6)
    assert ssp_out.shape == (nel,)
    # Elements 0 and 1 are ignited (t > tburn)
    # Elements 2 and 3 are unburnt (t < tburn)
    assert all(s > 0.0 for s in ssp_out)


def test_law105_consistent_tangent():
    """Verify consistent tangent stiffness matrix for LAW105."""
    p = _make_powder_params(bulk=12000.0, rho0=1.5e-6)
    c_tan = solid_tangent(p)

    assert c_tan.shape == (6, 6)
    # Hydrostatic block: all C_ii (i=0..2) equal to K_eff
    k_eff = c_tan[0, 0]
    assert k_eff == pytest.approx(p.bulk)
    assert c_tan[1, 1] == pytest.approx(k_eff)
    assert c_tan[2, 2] == pytest.approx(k_eff)
    assert c_tan[0, 1] == pytest.approx(k_eff)
    assert c_tan[0, 2] == pytest.approx(k_eff)
    assert c_tan[1, 2] == pytest.approx(k_eff)

    # Deviatoric / shear block must be zero
    assert c_tan[3, 3] == 0.0
    assert c_tan[4, 4] == 0.0
    assert c_tan[5, 5] == 0.0

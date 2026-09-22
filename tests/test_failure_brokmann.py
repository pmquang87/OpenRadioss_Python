"""Unit tests for Brokmann subcritical crack growth failure model and Newman-Raju SIF.

Upstream Fortran reference:
  ``engine/source/materials/fail/alter/fail_brokmann.F``
  ``common_source/fail/newman_raju.F90``

Verifies:
  - Newman-Raju stress intensity factor newman_raju_K at phi=pi/2 gives expected value.
  - SIF at phi=0 (surface point) matches analytical geometry correction.
  - brokmann_check failure criterion triggers when crack reaches critical size (a >= a_crit or K >= K_IC).
  - Sub-critical crack growth under cyclic loading (Paris law) leads to critical size failure.
  - Registration with failure dispatcher (pyradioss.failure.solid_step, shell_step, FAILURE_MODELS).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss import failure
from pyradioss.failure.brokmann import (
    BrokmannGroup,
    brokmann_check,
    newman_raju,
    newman_raju_K,
    newman_raju_vec,
    paris_law_da,
    register,
    shell_step,
    solid_step,
)


class _FailStub:
    """Minimal fail object for testing dispatcher integration."""

    def __init__(self, ftype: str = "BROKMANN", params: dict | None = None) -> None:
        self.type = ftype
        self.params = params or {
            "exp_n": 16.0,
            "k_ic": 1.0e6,
            "k_th": 0.0,
            "v0": 1.0,
            "alpha": 0.9,
            "sig_ini": 0.0,
            "fac_m": 1.0,
            "fac_l": 1.0,
            "fac_t": 1.0,
        }
        self.state = {}


def test_newman_raju_K_deepest_point():
    """Verify newman_raju_K at phi=pi/2 (deepest point) gives expected value."""
    # Test parameters: semi-elliptical surface crack in a plate
    a = 0.002   # crack depth 2 mm
    c = 0.004   # crack half-length 4 mm
    t = 0.010   # plate thickness 10 mm
    b = 0.050   # plate half-width 50 mm
    sigma = 100.0e6  # applied tensile stress 100 MPa
    phi = math.pi / 2.0  # deepest point

    # Analytical evaluation matching common_source/fail/newman_raju.F90 lines 73-99:
    ac = a / c  # 0.5
    at = a / t  # 0.2
    q = 1.0 + 1.464 * (ac ** 1.65)
    m1 = 1.13 - 0.09 * ac
    m2 = -0.54 + 0.89 / (0.2 + ac)
    m3 = 0.5 - 1.0 / (0.65 + ac) + 14.0 * ((1.0 - ac) ** 24)
    # At phi = pi/2: sinp = 1.0, cosp = 0.0
    sinp = 1.0
    g = 1.0 + (0.1 + 0.35 * (at ** 2)) * ((1.0 - sinp) ** 2)  # exactly 1.0
    fphi = 1.0  # (0 + 1)^0.25 = 1.0
    fw = math.cos(math.pi * c / (2.0 * b) * math.sqrt(at))
    f = (m1 + m2 * (at ** 2) + m3 * (at ** 4)) * fphi * g / math.sqrt(abs(fw))
    F_expected = math.sqrt(1.0 / q) * f
    K_expected = sigma * math.sqrt(math.pi * a) * F_expected

    # Evaluate newman_raju_K with explicit phi=pi/2
    K_calc = newman_raju_K(a, c, t, b, sigma, phi=math.pi / 2.0)
    assert K_calc == pytest.approx(K_expected, rel=1e-6)

    # Verify default phi is pi/2 (deepest point)
    K_default = newman_raju_K(a, c, t, b, sigma)
    assert K_default == pytest.approx(K_expected, rel=1e-6)

    # Verify consistency with upstream newman_raju(c, a, t, b, fpi=0.5)
    F_nr = newman_raju(c, a, t, b, fpi=0.5)
    assert K_calc == pytest.approx(sigma * math.sqrt(math.pi * a) * F_nr, rel=1e-12)

    # Verify expected magnitude: ~7.295 MPa*sqrt(m)
    assert 7.2e6 < K_calc < 7.4e6


def test_newman_raju_K_surface_point():
    """Verify newman_raju_K at phi=0 (surface point) matches upstream geometry factor."""
    a = 0.002
    c = 0.004
    t = 0.010
    b = 0.050
    sigma = 100.0e6

    K_surf = newman_raju_K(a, c, t, b, sigma, phi=0.0)
    F_surf = newman_raju(c, a, t, b, fpi=0.0)
    expected_surf = sigma * math.sqrt(math.pi * a) * F_surf

    assert K_surf == pytest.approx(expected_surf, rel=1e-12)
    # Deepest point K is higher than surface point K for this geometry
    K_deep = newman_raju_K(a, c, t, b, sigma, phi=math.pi / 2.0)
    assert K_deep > K_surf


def test_brokmann_check_triggers_at_critical_crack_size():
    """Verify failure triggers when crack reaches critical size (a >= a_crit or K >= K_IC)."""
    # 1. Direct critical crack size a_crit check
    group1 = {
        "a": 0.003,
        "c": 0.006,
        "t": 0.020,
        "b": 0.100,
        "a_crit": 0.005,
        "k_ic": 100.0e6,  # high so K_IC won't trigger first
    }
    # Crack is below critical size (3 mm < 5 mm) -> False
    assert not brokmann_check(group1, stress_eq=50.0e6)

    # Crack grows to exactly critical size (5 mm == 5 mm) -> True
    group1["a"] = 0.005
    assert brokmann_check(group1, stress_eq=50.0e6)

    # Crack exceeds critical size (6 mm > 5 mm) -> True
    group1["a"] = 0.006
    assert brokmann_check(group1, stress_eq=50.0e6)

    # 2. Fracture toughness K_IC check: critical crack size reached when K_I >= K_IC
    # Under sigma = 100 MPa and K_IC = 10 MPa*sqrt(m):
    group2 = BrokmannGroup(a=0.001, c=0.002, t=0.020, b=0.100, k_ic=10.0e6)
    # a = 1 mm -> K_I ~ 5.16 MPa*sqrt(m) < 10 MPa*sqrt(m) -> not failed
    assert not brokmann_check(group2, stress_eq=100.0e6)
    assert not group2.failed

    # Crack depth reaches critical size a = 4 mm -> K_I ~ 10.3 MPa*sqrt(m) >= 10 -> triggers failure!
    group2.a = 0.004
    group2.c = 0.008
    assert brokmann_check(group2, stress_eq=100.0e6)
    assert group2.failed


def test_brokmann_check_through_thickness_penetration():
    """Verify failure triggers when crack penetrates thickness (a >= t)."""
    group = {"a": 0.009, "c": 0.015, "t": 0.010, "b": 0.050, "k_ic": 1e9}
    # Below thickness (9 mm < 10 mm)
    assert not brokmann_check(group, stress_eq=1.0)

    # At or through thickness (10 mm >= 10 mm)
    group["a"] = 0.010
    assert brokmann_check(group, stress_eq=1.0)


def test_brokmann_fatigue_crack_growth_to_failure():
    """Verify sub-critical crack growth via Paris law leads to critical failure."""
    # Start with a sub-critical crack
    # Initial depth a = 1 mm, thickness t = 15 mm, K_IC = 15 MPa*sqrt(m)
    group = BrokmannGroup(
        a=0.001,
        c=0.002,
        t=0.015,
        b=0.050,
        k_ic=15.0e6,
        C=1.0e-11,
        m=3.0,
    )
    delta_stress = 80.0e6  # cyclic stress range 80 MPa

    # Initial state: not failed
    assert not group.check(delta_stress)

    # Grow crack under cyclic loading
    cycles_applied = 0
    max_cycles = 10000
    while not group.failed and cycles_applied < max_cycles:
        group.grow(delta_stress, dN=100.0)
        cycles_applied += 100

    # Crack must have grown and reached critical size
    assert group.failed
    assert group.a > 0.001
    assert group.K >= 15.0e6


def test_newman_raju_vectorized():
    """Verify vectorized newman_raju_K matches scalar evaluations."""
    a = np.array([0.001, 0.002, 0.003])
    c = np.array([0.002, 0.004, 0.006])
    t = np.array([0.010, 0.010, 0.010])
    b = np.array([0.050, 0.050, 0.050])
    sigma = np.array([50.0e6, 100.0e6, 150.0e6])

    K_vec = newman_raju_K(a, c, t, b, sigma, phi=math.pi / 2.0)
    assert isinstance(K_vec, np.ndarray)
    assert K_vec.shape == (3,)

    for i in range(3):
        K_scalar = newman_raju_K(a[i], c[i], t[i], b[i], sigma[i], phi=math.pi / 2.0)
        assert K_vec[i] == pytest.approx(K_scalar, rel=1e-6)


def test_brokmann_check_vectorized():
    """Verify brokmann_check handles array inputs across multiple elements."""
    group = {
        "a": np.array([0.001, 0.003, 0.008]),
        "c": np.array([0.002, 0.006, 0.016]),
        "t": np.array([0.020, 0.020, 0.020]),
        "b": np.array([0.100, 0.100, 0.100]),
        "k_ic": 12.0e6,
    }
    stress_eq = np.array([100.0e6, 100.0e6, 100.0e6])

    failed = brokmann_check(group, stress_eq)
    assert isinstance(failed, np.ndarray)
    assert failed.dtype == bool
    # Small crack (1 mm) doesn't fail, large crack (8 mm) fails
    assert not failed[0]
    assert failed[2]


def test_dispatcher_registration():
    """Verify Brokmann failure model is registered in failure model dispatchers."""
    # Check registration dictionary
    assert "BROKMANN" in failure.FAILURE_MODELS
    assert "FAIL_BROKMANN" in failure.FAILURE_MODELS
    assert failure.FAILURE_MODELS["BROKMANN"] is failure.brokmann

    # Check solid_step dispatch
    fail_solid = _FailStub(ftype="BROKMANN")
    sig_solid = np.zeros((1, 6))
    d_epsp = np.zeros(1)
    deps = np.zeros((1, 6))
    dama = np.zeros(1)
    rupt_solid = failure.solid_step(fail_solid, sig_solid, d_epsp, deps, dt=1e-6, dama=dama)
    assert isinstance(rupt_solid, np.ndarray)

    # Check shell_step dispatch
    fail_shell = _FailStub(ftype="FAIL_BROKMANN")
    sig_shell = np.zeros((1, 3))
    rupt_shell = failure.shell_step(fail_shell, sig_shell, d_epsp, deps[:, :3], dt=1e-6, dama=dama)
    assert isinstance(rupt_shell, np.ndarray)

    # Verify custom register function
    custom_dict = {}
    register(custom_dict)
    assert "BROKMANN" in custom_dict
    assert "FAIL_BROKMANN" in custom_dict


def test_brokmann_step_fortran_state_update():
    """Verify brokmann_step matches Fortran fail_brokmann.F crack growth and rupture."""
    from pyradioss.failure.brokmann import brokmann_step

    nel = 1
    uvar = np.zeros((nel, 21), dtype=float)
    # State variables (Python 0-based indexing matching Fortran UVAR(15..21)):
    uvar[0, 14] = 0.0      # UVAR(15): FAIL_B = 0 (alive/not yet failed)
    uvar[0, 15] = 2000.0   # UVAR(16): CR_LEN = 2000 um (2 mm)
    uvar[0, 16] = 4000.0   # UVAR(17): CR_DEPTH = 4000 um (4 mm)
    uvar[0, 17] = 0.0      # UVAR(18): CR_ANG = 0
    uvar[0, 18] = 10000.0  # UVAR(19): THK0 = 10000 um (10 mm)
    uvar[0, 19] = 50000.0  # UVAR(20): ALDT0 = 50000 um (50 mm)
    uvar[0, 20] = 0.0      # UVAR(21): SIG_COS previous = 0

    off = np.ones(nel, dtype=float)
    tdel = np.zeros(nel, dtype=float)
    uparam = np.zeros(35, dtype=float)
    uparam[0] = 16.0       # EXP_N
    uparam[5] = 10.0e6     # K_IC = 10 MPa*sqrt(m)
    uparam[6] = 0.0        # K_TH = 0
    uparam[7] = 1.0        # V0 = 1.0 m/s
    uparam[9] = 1.0        # ALPHA = 1.0 (no lag filter)
    uparam[29] = 0.0       # SIG_INI = 0
    uparam[32] = 1.0       # FAC_M = 1
    uparam[33] = 1.0       # FAC_L = 1
    uparam[34] = 1.0       # FAC_T = 1

    # Apply moderate tensile stress (CR_ANG=0 -> crack open by SIGNXX)
    signxx = np.array([50.0e6])
    signyy = np.array([0.0])
    signxy = np.array([0.0])

    # Advance 1 step
    dt = 1.0e-3
    failed = brokmann_step(uvar, off, signxx, signyy, signxy, uparam, dt, time=0.001, tdel=tdel)

    assert not failed[0]
    assert uvar[0, 14] == 0.0
    # Crack should have grown (DA > 0 and DC > 0)
    assert uvar[0, 15] > 2000.0
    assert uvar[0, 16] > 4000.0

    # Now apply high tensile stress exceeding fracture toughness K_IC
    signxx = np.array([200.0e6])
    failed_rupt = brokmann_step(uvar, off, signxx, signyy, signxy, uparam, dt, time=0.002, tdel=tdel)
    assert failed_rupt[0]
    assert uvar[0, 14] == 1.0
    assert tdel[0] == 0.002


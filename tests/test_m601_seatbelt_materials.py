"""
Unit tests for Milestone M601:
Seatbelt Material Models Suite:
- /MAT/LAW114 (/MAT/SPR_SEATBELT) — 1D Spring Seatbelt Material
- /MAT/LAW119 (/MAT/SH_SEATBELT) — 2D Shell Seatbelt Material
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import FunctTable
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.materials.law114_seatbelt import Law114Seatbelt, build_law114
from pyradioss.materials.law119_seatbelt import (
    Law119Seatbelt,
    build_law119,
    extra_shapes as law119_extra_shapes,
    shell_membrane_tangent,
    consistent_shell_tangent,
    shell_update as law119_shell_update,
    solid_update as law119_solid_update,
    sound_speed as law119_sound_speed,
)
import pyradioss.materials as mats


def _parse_deck(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    assert not log.errors, f"Parse errors: {log.errors}"
    return model, log


# ============================================================================
# 1. LAW114 1D Spring Seatbelt Tests
# ============================================================================

def test_law114_tension_only_and_compression():
    """Verify LAW114 with YOUNG=0 has zero compressive stiffness (tension-only)."""
    mat = Law114Seatbelt(
        id=1,
        rho0=7.8e-6,
        params={
            "STIFF1": 2000.0,
            "DAMP1": 0.0,
            "YOUNG": 0.0,  # tension only
            "LMIN": 10.0,
        },
    )

    state = None
    L0 = 10.0

    # 1. Tension: delta = +2.0 -> force > 0
    F_tens, k_tan, state = mat.spring_update(L=12.0, L0=L0, state=state, dt=0.001)
    assert F_tens > 0.0
    assert pytest.approx(F_tens) == 2000.0 * 2.0
    assert pytest.approx(k_tan) == 2000.0

    # 2. Compression: delta = -2.0 -> force must be 0 (slack/buckling)
    F_comp, k_comp, state = mat.spring_update(L=8.0, L0=L0, state=state, dt=0.001)
    assert pytest.approx(F_comp) == 0.0
    assert pytest.approx(k_comp) == 0.0

    # 3. Further compression: delta = -5.0 -> force remains 0
    F_comp2, k_comp2, state = mat.spring_update(L=5.0, L0=L0, state=state, dt=0.001)
    assert pytest.approx(F_comp2) == 0.0
    assert pytest.approx(k_comp2) == 0.0


def test_law114_compression_with_young():
    """Verify LAW114 with YOUNG > 0 provides compressive stiffness."""
    mat = Law114Seatbelt(
        id=2,
        rho0=7.8e-6,
        params={
            "STIFF1": 2000.0,
            "YOUNG": 5000.0,
            "SHEAR_AREA": 1.5,
            "LMIN": 10.0,
        },
    )

    state = None
    L0 = 10.0

    # Compression: delta = -1.0
    # K_comp = YOUNG * SHEAR_AREA / L0 = 5000.0 * 1.5 / 10.0 = 750.0
    F_comp, k_comp, state = mat.spring_update(L=9.0, L0=L0, state=state, dt=0.001)
    assert F_comp < 0.0
    expected_k = 5000.0 * 1.5 / 10.0
    assert pytest.approx(k_comp) == expected_k
    assert pytest.approx(F_comp) == -expected_k * 1.0


def test_law114_curve_with_hysteresis():
    """Verify LAW114 loading with FUN_L and unloading with hysteresis along FUN_UL."""
    # Loading curve: non-linear force vs strain
    # strain: 0.0, 0.05, 0.10, 0.20
    # force:  0.0,  500, 1500, 4000
    curve_l = FunctTable(fct_id=10, x=[0.0, 0.05, 0.10, 0.20], y=[0.0, 500.0, 1500.0, 4000.0])
    # Unloading curve: steeper initial unloading slope
    curve_ul = FunctTable(fct_id=11, x=[0.0, 0.05, 0.10, 0.20], y=[0.0, 1000.0, 2500.0, 6000.0])

    functions = {10: curve_l, 11: curve_ul}

    mat = Law114Seatbelt(
        id=3,
        rho0=7.8e-6,
        params={
            "FUN_L": 10,
            "FUN_UL": 11,
            "Fcoeft1": 1.0,
            "Xcoeft1": 1.0,
            "LMIN": 10.0,
            "YOUNG": 0.0,
        },
    )

    state = None
    L0 = 10.0

    # Step 1: Load to eps = 0.10 (L = 11.0, delta = 1.0)
    F1, k1, state = mat.spring_update(L=11.0, L0=L0, state=state, dt=0.001, functions=functions)
    assert pytest.approx(F1) == 1500.0
    assert state["yield_f"] == 1500.0
    assert pytest.approx(state["eps_max"]) == 0.10

    # Step 2: Load further to eps = 0.15 (L = 11.5, delta = 1.5)
    # Between 0.10 and 0.20, slope = (4000 - 1500) / 0.10 = 25000. At 0.15, F = 1500 + 25000*0.05 = 2750
    F2, k2, state = mat.spring_update(L=11.5, L0=L0, state=state, dt=0.001, functions=functions)
    assert pytest.approx(F2) == 2750.0
    assert state["yield_f"] == 2750.0

    # Step 3: Unload partially to L = 11.2 (eps = 0.12, delta = 1.2)
    # Force must decrease below peak yield_f
    F3, k3, state = mat.spring_update(L=11.2, L0=L0, state=state, dt=0.001, functions=functions)
    assert F3 < F2
    assert state["yield_f"] == 2750.0  # peak preserved

    # Step 4: Complete unload to L = 10.0 (delta = 0.0)
    F4, k4, state = mat.spring_update(L=10.0, L0=L0, state=state, dt=0.001, functions=functions)
    assert pytest.approx(F4) == 0.0
    # Permanent offset dpx should be positive (hysteresis / plasticity)
    assert state["dpx"] > 0.0


def test_law114_strain_rate_sensitivity():
    """Verify LAW114 strain-rate scaling on loading curve."""
    curve = FunctTable(fct_id=20, x=[0.0, 0.1, 0.2], y=[0.0, 1000.0, 2000.0])
    functions = {20: curve}

    # Material with rate sensitivity: c_rate = 0.1, eps0 = 1.0
    mat = Law114Seatbelt(
        id=4,
        rho0=7.8e-6,
        params={
            "FUN_L": 20,
            "C_RATE": 0.15,
            "EPS0": 1.0,
            "LMIN": 1.0,
        },
    )

    L0 = 1.0
    # Quasi-static run (v_rel = 0.0)
    state_qs = None
    F_qs, _, _ = mat.spring_update(L=1.1, L0=L0, v_rel=0.0, state=state_qs, dt=0.01, functions=functions)

    # Dynamic run with high strain rate (eps_dot = 100 s^-1 => v_rel = 100.0)
    state_dyn = None
    F_dyn, _, _ = mat.spring_update(L=1.1, L0=L0, v_rel=100.0, state=state_dyn, dt=0.001, functions=functions)

    # Dynamic force should exceed quasi-static by (1 + 0.15 * ln(100))
    expected_factor = 1.0 + 0.15 * math.log(100.0)
    assert pytest.approx(F_dyn / F_qs, rel=1e-3) == expected_factor


def test_law114_damping_in_tension():
    """Verify viscous damping adds force in tension."""
    mat = Law114Seatbelt(
        id=5,
        rho0=7.8e-6,
        params={
            "STIFF1": 1000.0,
            "DAMP1": 50.0,
            "LMIN": 1.0,
        },
    )
    L0 = 1.0
    state = None
    # delta = 0.1, v_rel = 2.0 -> F_elas = 100, F_damp = 50 * 2 = 100 -> Total = 200
    F, _, state = mat.spring_update(L=1.1, L0=L0, v_rel=2.0, state=state, dt=0.01)
    assert pytest.approx(F) == 200.0


# ============================================================================
# 2. LAW119 2D Shell Seatbelt Tests
# ============================================================================

def test_law119_orthotropic_biaxial_tension():
    """Verify LAW119 membrane response under orthotropic biaxial tension."""
    mat = Law119Seatbelt(
        id=10,
        rho0=1.2e-6,
        params={
            "E11": 10000.0,
            "E22": 5000.0,
            "NU12": 0.2,
            "G12": 2500.0,
            "RE": 0.05,
            "Fcoeft22": 1.0,
        },
    )

    sig = np.zeros((1, 3))
    deps = np.array([[0.01, 0.005, 0.02]])

    sig_out, _ = law119_shell_update(mat, sig, deps)

    # Hand calculation:
    # nu21 = 0.2 * 1.0 = 0.2
    # det = 1 / (1 - 0.2 * 0.2) = 1 / 0.96 = 1.04166667
    # A11 = 10000 * det = 10416.6667
    # A22 = 10416.6667
    # A12 = 10416.6667 * 0.2 = 2083.3333
    # G12 = 2500
    # Tension check: S = 0.0075, D = 0.0025, R = sqrt(0.02^2 + 0.0025^2) = 0.020156
    # P1 = 0.0075 + 0.020156 > 0 -> Tension (beta = 1.0)
    # sig_xx = 10416.6667 * 0.01 + 2083.3333 * 0.005 = 114.5833
    # sig_yy = 2083.3333 * 0.01 + 10416.6667 * 0.005 = 72.9167
    # sig_xy = 2500 * 0.02 = 50.0
    assert pytest.approx(sig_out[0, 0], rel=1e-4) == 114.5833
    assert pytest.approx(sig_out[0, 1], rel=1e-4) == 72.9167
    assert pytest.approx(sig_out[0, 2], rel=1e-4) == 50.0


def test_law119_compression_reduction_rcomp():
    """Verify LAW119 scales compressive stresses by RCOMP to simulate yarn buckling/wrinkling."""
    mat = Law119Seatbelt(
        id=11,
        rho0=1.2e-6,
        params={
            "E11": 10000.0,
            "E22": 10000.0,
            "NU12": 0.2,
            "G12": 2500.0,
            "RE": 0.02,  # RCOMP = 0.02 (98% stiffness reduction under compression)
            "Fcoeft22": 1.0,
        },
    )

    sig = np.zeros((1, 3))
    # Biaxial compression: eps_xx = -0.01, eps_yy = -0.01
    deps = np.array([[-0.01, -0.01, 0.0]])

    sig_out, _ = law119_shell_update(mat, sig, deps)

    # Without RCOMP, sig_xx would be: (A11 * -0.01 + A12 * -0.01) = -125.0
    # With RCOMP = 0.02, sig_xx = -125.0 * 0.02 = -2.5
    unreduced = (mat.a11 * (-0.01) + mat.a12 * (-0.01))
    assert pytest.approx(sig_out[0, 0]) == unreduced * 0.02
    assert pytest.approx(sig_out[0, 1]) == unreduced * 0.02


def test_law119_rate_sensitivity_and_nonlinear_curve():
    """Verify LAW119 rate sensitivity with non-linear loading curve."""
    curve = FunctTable(fct_id=30, x=[0.0, 0.05, 0.1], y=[0.0, 50.0, 120.0])
    functions = {30: curve}

    mat = Law119Seatbelt(
        id=12,
        rho0=1.2e-6,
        params={
            "FUN_L": 30,
            "E11": 1000.0,
            "E22": 1000.0,
            "NU12": 0.2,
            "G12": 500.0,
            "C_RATE": 0.2,
            "EPS0": 1.0,
        },
    )

    extra_qs = {"functions": functions}
    extra_dyn = {"functions": functions}

    deps = np.array([[0.04, 0.02, 0.0]])

    # Quasi-static (dt = 0)
    sig_qs, _ = law119_shell_update(mat, np.zeros((1, 3)), deps, dt=0.0, extra=extra_qs)

    # Dynamic with high rate (dt = 1e-4 => strain rate ~ 400 s^-1)
    sig_dyn, _ = law119_shell_update(mat, np.zeros((1, 3)), deps, dt=1e-4, extra=extra_dyn)

    # Dynamic stress must be higher due to rate sensitivity
    assert sig_dyn[0, 0] > sig_qs[0, 0]
    assert sig_dyn[0, 1] > sig_qs[0, 1]


def test_law119_sound_speed_and_tangent():
    """Verify LAW119 acoustic sound speed and consistent tangent tensor."""
    mat = Law119Seatbelt(
        id=13,
        rho0=1.2e-6,
        params={
            "E11": 10000.0,
            "E22": 5000.0,
            "NU12": 0.2,
            "G12": 2500.0,
            "RE": 0.01,
            "Fcoeft22": 1.0,
        },
    )

    c_sound = law119_sound_speed(mat)
    # c = sqrt(C1 / rho0) where C1 = max(E11, E22) / (1 - nu12*nu21) = 10000 / 0.96 = 10416.667
    expected_c = math.sqrt(10416.666667 / 1.2e-6)
    assert pytest.approx(c_sound, rel=1e-4) == expected_c

    # Membrane tangent
    C_tan = shell_membrane_tangent(mat)
    assert C_tan.shape == (3, 3)
    assert pytest.approx(C_tan[0, 0]) == mat.a11
    assert pytest.approx(C_tan[1, 1]) == mat.a22
    assert pytest.approx(C_tan[0, 1]) == mat.a12
    assert pytest.approx(C_tan[2, 2]) == mat.g12

    # Solid update rejection
    with pytest.raises(NotImplementedError):
        law119_solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)))


# ============================================================================
# 3. Energy Conservation in Dynamic Simulation
# ============================================================================

def test_law114_energy_conservation_oscillator():
    """Verify internal energy accounting and energy conservation in 1D spring-mass oscillator."""
    # Spring-mass system: M = 2.0 kg, K = 5000.0 N/m, L0 = 1.0 m
    # Initial stretch: x0 = 0.05 m (L = 1.05 m), v0 = 0.0
    # Analytical energy: E_total = 0.5 * K * x0^2 = 0.5 * 5000 * 0.05^2 = 6.25 J
    mat = Law114Seatbelt(
        id=50,
        rho0=7.8e-6,
        params={
            "STIFF1": 5000.0,
            "DAMP1": 0.0,   # undamped for energy conservation
            "YOUNG": 5000.0,
            "LMIN": 1.0,
        },
    )

    mass = 2.0
    L0 = 1.0
    dt = 1e-4
    n_steps = 1000

    # Initial state
    pos = 1.05
    vel = 0.0
    state = None

    # Step 0: establish initial force and potential energy
    F, _, state = mat.spring_update(L=pos, L0=L0, v_rel=vel, state=state, dt=dt)
    E_initial = 0.5 * 5000.0 * (pos - L0)**2

    # Central difference (Velocity Verlet) time integration
    for step in range(n_steps):
        # Acceleration
        acc = -F / mass
        # Half step velocity
        v_half = vel + 0.5 * acc * dt
        # Full step position
        pos = pos + v_half * dt
        # Update spring force at new position
        F, _, state = mat.spring_update(L=pos, L0=L0, v_rel=v_half, state=state, dt=dt)
        # New acceleration and full step velocity
        acc_new = -F / mass
        vel = v_half + 0.5 * acc_new * dt

    E_kin = 0.5 * mass * vel**2
    E_int = state["eint"]
    E_total = E_kin + E_int

    # Energy should be conserved to within < 0.2% numerical integration error
    assert pytest.approx(E_total, rel=2e-3) == E_initial
    assert E_kin > 0.0  # system actively oscillating
    assert E_int > 0.0


# ============================================================================
# 4. Starter Deck Parsing Tests (Fixed & Free Formats)
# ============================================================================

def test_deck_parsing_law114_spr_seatbelt(tmp_path: Path):
    """Test parsing /MAT/LAW114 fixed and /MAT/SPR_SEATBELT free format cards."""
    # 1. Fixed format
    deck_fixed = """/BEGIN
LAW114_FIXED_TEST
                  00
/MAT/LAW114/1140
Seatbelt 1D Spring Mat Fixed
                7.8e-6                15.0
               12000.0               150.0
         5         6               1.5                 2.0
              210000.0                10.0                20.0             50000.0             10000.0
                  25.0                 0.5
/END
"""
    model_fixed, log_fixed = _parse_deck(tmp_path, deck_fixed)
    assert 1140 in model_fixed.materials
    m_fixed = model_fixed.materials[1140]
    assert m_fixed.law == 114
    assert pytest.approx(m_fixed.rho0) == 7.8e-6
    assert pytest.approx(m_fixed.params["k"]) == 12000.0
    assert pytest.approx(m_fixed.params["c"]) == 150.0
    assert m_fixed.params["fun_l"] == 5
    assert m_fixed.params["fun_ul"] == 6
    assert pytest.approx(m_fixed.params["xscale"]) == 1.5
    assert pytest.approx(m_fixed.params["fscale"]) == 2.0
    assert pytest.approx(m_fixed.params["young"]) == 210000.0
    assert pytest.approx(m_fixed.params["fmax"]) == 50000.0
    assert pytest.approx(m_fixed.params["mmax"]) == 10000.0

    # 2. Free format
    deck_free = """#RADIOSS STARTER
/BEGIN
LAW114_FREE_TEST
0 0
/MAT/SPR_SEATBELT/1141
Seatbelt 1D Spring Mat Free
7.85e-6 12.0
15000.0 200.0
7 8 1.2 1.8
200000.0 8.0 16.0 45000.0 9000.0
30.0 0.8
/END
"""
    model_free, log_free = _parse_deck(tmp_path, deck_free)
    assert 1141 in model_free.materials
    m_free = model_free.materials[1141]
    assert m_free.law == 114
    assert pytest.approx(m_free.rho0) == 7.85e-6
    assert pytest.approx(m_free.params["k"]) == 15000.0
    assert pytest.approx(m_free.params["c"]) == 200.0
    assert m_free.params["fun_l"] == 7
    assert m_free.params["fun_ul"] == 8
    assert pytest.approx(m_free.params["xscale"]) == 1.2
    assert pytest.approx(m_free.params["fscale"]) == 1.8
    assert pytest.approx(m_free.params["young"]) == 200000.0


def test_deck_parsing_law119_sh_seatbelt(tmp_path: Path):
    """Test parsing /MAT/LAW119 fixed and /MAT/SH_SEATBELT free format cards."""
    # 1. Fixed format
    deck_fixed = """/BEGIN
LAW119_FIXED_TEST
                  00
/MAT/LAW119/1190
Shell Seatbelt Mat Fixed
                1.2e-6                20.0
               10000.0               100.0                50.0
         1         2                 1.1                 1.2         1
               15000.0                 0.3              5000.0                 1.0
                2000.0                 0.4                 0.1
/END
"""
    model_fixed, log_fixed = _parse_deck(tmp_path, deck_fixed)
    assert 1190 in model_fixed.materials
    m_fixed = model_fixed.materials[1190]
    assert m_fixed.law == 119
    assert pytest.approx(m_fixed.rho0) == 1.2e-6
    assert pytest.approx(m_fixed.params["lmin"]) == 20.0
    assert pytest.approx(m_fixed.params["stiff1"]) == 10000.0
    assert pytest.approx(m_fixed.params["re"]) == 50.0
    assert m_fixed.params["fun_l"] == 1
    assert m_fixed.params["fun_ul"] == 2
    assert pytest.approx(m_fixed.params["fscale1"]) == 1.1
    assert pytest.approx(m_fixed.params["fscale2"]) == 1.2
    assert m_fixed.params["ireload"] == 1
    assert pytest.approx(m_fixed.params["e22"]) == 15000.0
    assert pytest.approx(m_fixed.params["nu12"]) == 0.3
    assert pytest.approx(m_fixed.params["g12"]) == 5000.0
    assert pytest.approx(m_fixed.params["ecoat"]) == 2000.0
    assert pytest.approx(m_fixed.params["nucoat"]) == 0.4
    assert pytest.approx(m_fixed.params["tcoat"]) == 0.1

    # 2. Free format
    deck_free = """#RADIOSS STARTER
/BEGIN
LAW119_FREE_TEST
0 0
/MAT/SH_SEATBELT/1191
Shell Seatbelt Mat Free
1.3e-6 25.0
12000.0 120.0 60.0
3 4 1.3 1.4 2
18000.0 0.35 6000.0 1.2
2500.0 0.42 0.15
/END
"""
    model_free, log_free = _parse_deck(tmp_path, deck_free)
    assert 1191 in model_free.materials
    m_free = model_free.materials[1191]
    assert m_free.law == 119
    assert pytest.approx(m_free.rho0) == 1.3e-6
    assert pytest.approx(m_free.params["stiff1"]) == 12000.0
    assert pytest.approx(m_free.params["re"]) == 60.0
    assert m_free.params["fun_l"] == 3
    assert m_free.params["fun_ul"] == 4
    assert pytest.approx(m_free.params["fscale1"]) == 1.3
    assert pytest.approx(m_free.params["fscale2"]) == 1.4
    assert m_free.params["ireload"] == 2
    assert pytest.approx(m_free.params["e22"]) == 18000.0
    assert pytest.approx(m_free.params["nu12"]) == 0.35
    assert pytest.approx(m_free.params["g12"]) == 6000.0
    assert pytest.approx(m_free.params["ecoat"]) == 2500.0
    assert pytest.approx(m_free.params["nucoat"]) == 0.42
    assert pytest.approx(m_free.params["tcoat"]) == 0.15

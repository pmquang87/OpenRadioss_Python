"""
Tests for Milestone M604: Dynamic Relaxation & Adaptive Static Equilibrium (/RELAX, /DYREL, /ADYREL).

Upstream references:
- engine/source/general_controls/damping/static.F (STATIC, E_PERIOD, ENER_W0)
- engine/source/tools/univ/butterworth.F (BUTTERWORTH)
- engine/source/input/freform.F (KEREL, DYREL, ADYREL, RELAX)
"""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.engine.damping import DynamicRelaxation, butterworth_filter
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.model.model import EngineControls, Model, NodeGroup


class MockModel:
    """Mock model for testing relaxation controls."""
    def __init__(self, n=4, mass_val=2.5, inertia_val=0.4):
        self.numnod = n
        self.mass = np.ones(n, dtype=np.float64) * mass_val
        self.inertia = np.ones(n, dtype=np.float64) * inertia_val
        self.node_groups = {}


# ============================================================================
# 1. Butterworth Digital Filter Tests (butterworth.F)
# ============================================================================

def test_butterworth_dc_passthrough():
    """Verify that a constant DC signal passes through the Butterworth filter with unity gain."""
    dt = 0.001
    freq = 50.0  # Hz
    val = 42.0

    # Initialize filter state with DC value
    x2, x1, x = val, val, val
    fx2, fx1 = val, val

    for _ in range(50):
        fx = butterworth_filter(dt, freq, x2, x1, x, fx2, fx1)
        # Advance state
        x2, x1 = x1, x
        fx2, fx1 = fx1, fx

    assert fx == pytest.approx(val, rel=1e-6)


def test_butterworth_high_frequency_attenuation():
    """Verify that frequencies well above the cutoff are significantly attenuated."""
    dt = 1e-4
    f_cutoff = 10.0  # Hz cutoff
    f_high = 500.0   # Hz high frequency disturbance (50x cutoff)

    # Apply sinusoidal signal at high frequency
    t_vals = np.arange(0, 0.05, dt)
    raw_signal = np.sin(2.0 * np.pi * f_high * t_vals)

    filtered_vals = []
    x2, x1 = 0.0, 0.0
    fx2, fx1 = 0.0, 0.0

    for x in raw_signal:
        fx = butterworth_filter(dt, f_cutoff, x2, x1, x, fx2, fx1)
        filtered_vals.append(fx)
        x2, x1 = x1, x
        fx2, fx1 = fx1, fx

    # Raw amplitude is 1.0; filtered steady-state amplitude must be << 0.1
    attenuated_amplitude = np.max(np.abs(filtered_vals[-100:]))
    assert attenuated_amplitude < 0.05


def test_butterworth_exact_fortran_formula():
    """Verify exact formula match against manual evaluation of butterworth.F."""
    dt = 0.002
    freq = 25.0
    dt2 = dt / 2.0
    wd = np.sqrt(2.0) * np.pi * freq * (5.0 / 3.0)
    wa = np.tan(wd * dt2)
    wa2 = wa * wa
    c1 = 1.0 + np.sqrt(2.0) * wa + wa2
    a0 = wa2 / c1
    a1 = 2.0 * a0
    a2 = a0
    b1 = -2.0 * (wa2 - 1.0) / c1
    b2 = (-1.0 + np.sqrt(2.0) * wa - wa2) / c1

    x, x1, x2 = 3.5, 2.1, 1.0
    fx1, fx2 = 2.0, 0.9
    expected_fx = a0 * x + a1 * x1 + a2 * x2 + b1 * fx1 + b2 * fx2

    actual_fx = butterworth_filter(dt, freq, x2, x1, x, fx2, fx1)
    assert actual_fx == pytest.approx(expected_fx, rel=1e-12)


# ============================================================================
# 2. E_PERIOD Peak Detection Tests (static.F:211)
# ============================================================================

def test_e_period_peak_detection_synthetic():
    """Verify that E_PERIOD correctly tracks energy peaks and records maximum period."""
    model = MockModel(1)
    controls = EngineControls()
    controls.adyrel_active = True
    controls.adyrel_freq_c = 0.0  # unfiltered for exact peak tracking

    relax = DynamicRelaxation(model, controls)
    dt = 0.005
    T = 1.0  # period of 1 second (frequency 1 Hz)
    omega = 2.0 * np.pi / T

    peak_ke_count = 0
    peak_ie_count = 0

    # Simulate 2 full seconds
    for step in range(400):
        t = step * dt
        # KE peaks when sin^2 is maximum
        ke = float(10.0 * np.sin(omega * t) ** 2)
        # IE peaks when cos^2 is maximum
        ie = float(10.0 * np.cos(omega * t) ** 2)

        ipi, ipc = relax.e_period(dt, ke, ie, t)
        if ipc == 1:
            peak_ke_count += 1
        if ipi == 1:
            peak_ie_count += 1

    # Peaks must be detected multiple times
    assert peak_ke_count >= 4
    assert peak_ie_count >= 4
    # The maximum recorded ascent period pcmax must be approximately T/4 = 0.25s (within 3 steps)
    assert relax.pcmax == pytest.approx(0.25, abs=3 * dt)
    assert relax.pimax == pytest.approx(0.25, abs=3 * dt)


# ============================================================================
# 3. ENER_W0 Frequency Estimation & Adaptation (static.F:312)
# ============================================================================

def test_ener_w0_frequency_adaptation():
    """Verify that ENER_W0 automatically identifies oscillation frequency and adapts betate."""
    model = MockModel(1)
    controls = EngineControls()
    controls.adyrel_active = True
    controls.adyrel_freq_c = 0.0

    relax = DynamicRelaxation(model, controls)
    # With dt = 0.001, F_MAX = 0.01 / 0.001 = 10.0 Hz
    dt = 0.001
    T_target = 0.4  # seconds -> half period = 0.2s, ascent = 0.1s
    omega = 2.0 * np.pi / T_target

    # Initial betate is 0
    assert relax.adyrel_betate == 0.0

    # Run past nc_act (200 cycles) to enable frequency adaptation
    for cycle in range(1, 400):
        t = cycle * dt
        ke = float(5.0 * np.sin(omega * t) ** 2)
        ie = float(5.0 * np.cos(omega * t) ** 2)
        relax.update_adaptive_frequency(t, dt, cycle, e_int=ie, e_kin=ke)

    # After cycle 200 and observing peaks, betate must adapt to structural frequency
    assert relax.adyrel_betate > 0.0
    # Ascent time is 0.1s, so estimated frequency 1/Pmax is ~10.0 Hz, bounded by F_MAX = 10.0
    assert relax.adyrel_betate == pytest.approx(10.0, rel=0.1)


# ============================================================================
# 4. 1-DOF Spring-Mass Oscillator Static Equilibrium with /ADYREL
# ============================================================================

def test_harmonic_oscillator_adyrel_convergence():
    """Simulate a mass-spring system under constant force, verifying /ADYREL reaches static equilibrium."""
    m = 1.0
    k = 100.0  # omega_n = 10 rad/s -> T_n = 0.6283 s
    f_const = 50.0  # Expected static displacement x_static = F/k = 0.5
    x_static = f_const / k
    dt = 0.002
    t_end = 3.0

    model = MockModel(1)
    model.mass[0] = m
    controls = EngineControls()
    controls.adyrel_active = True
    controls.adyrel_freq_c = 0.0
    controls.t_end = t_end

    relax = DynamicRelaxation(model, controls)

    x = 0.0
    v = np.zeros((1, 3))
    vr = np.zeros((1, 3))
    total_de = 0.0

    n_steps = int(t_end / dt)
    x_history = []

    for step in range(n_steps):
        t = step * dt
        # Internal spring force + external applied force
        f_spring = -k * x
        a = (f_spring + f_const) / m

        v[0, 0] += a * dt
        de = relax.apply(t, dt, v, vr, model.mass, model.inertia)
        total_de += de
        x += v[0, 0] * dt

        # Energy update at end of cycle
        e_int = 0.5 * k * (x ** 2)
        e_kin = 0.5 * m * (v[0, 0] ** 2)
        relax.update_adaptive_frequency(t, dt, step, e_int, e_kin)
        x_history.append(x)

    # Must converge to exact static equilibrium x = 0.5 within tight tolerance
    assert x == pytest.approx(x_static, abs=0.02)
    # Final velocity should be near zero (settled structure)
    assert abs(v[0, 0]) < 0.05
    assert total_de > 0.0


# ============================================================================
# 5. Energy Conservation & Machine-Precision Attenuation
# ============================================================================

def test_adyrel_energy_conservation_precision():
    """Verify that dissipated energy de in /ADYREL matches kinetic energy loss to machine precision."""
    model = MockModel(6)
    controls = EngineControls()
    controls.adyrel_active = True
    controls.adyrel_tstart = 0.0
    controls.adyrel_tstop = 1.0

    relax = DynamicRelaxation(model, controls)
    relax.adyrel_betate = 15.0  # set active betate
    dt = 0.004

    np.random.seed(123)
    v = np.random.randn(6, 3) * 5.0
    vr = np.random.randn(6, 3) * 2.0

    ke_before = float(
        0.5 * np.sum(model.mass[:, None] * v ** 2)
        + 0.5 * np.sum(model.inertia[:, None] * vr ** 2)
    )

    de = relax.apply(0.1, dt, v, vr, model.mass, model.inertia)

    ke_after = float(
        0.5 * np.sum(model.mass[:, None] * v ** 2)
        + 0.5 * np.sum(model.inertia[:, None] * vr ** 2)
    )

    # Machine-precision conservation of the energy drop
    assert (ke_before - ke_after) == pytest.approx(de, rel=1e-14, abs=1e-14)


# ============================================================================
# 6. Keyword Parsing: /ADYREL, /ADYREL/FREQ_, /RELAX, /RELAX/SYSTEM, /RELAX/DYNA
# ============================================================================

def test_engine_keywords_adyrel_parsing(tmp_path):
    """Verify parsing of /ADYREL and /ADYREL/FREQ_ with cutoff frequency and time bounds."""
    deck = """/RUN/ADYREL_TEST/1
1.0
/ADYREL/FREQ_
50.0
0.01 0.85
/END
"""
    p = tmp_path / "ADY_0001.rad"
    p.write_text(deck, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    ec = parse_engine_deck(blocks, log)
    assert len(log.errors) == 0

    assert ec.adyrel_active is True
    assert ec.adyrel_freq_c == pytest.approx(50.0)
    assert ec.adyrel_tstart == pytest.approx(0.01)
    assert ec.adyrel_tstop == pytest.approx(0.85)

    model = MockModel(2)
    relax = DynamicRelaxation(model, ec, log)
    assert relax.active is True
    assert relax.adyrel_active is True
    assert relax.adyrel_freq_c == pytest.approx(50.0)
    assert relax.adyrel_tstart == pytest.approx(0.01)
    assert relax.adyrel_tstop == pytest.approx(0.85)


def test_engine_keywords_relax_system_and_dyna(tmp_path):
    """Verify parsing of /RELAX/SYSTEM (default KEREL) and /RELAX/DYNA (DYREL)."""
    deck_kerel = """/RUN/RELAX_TEST/1
1.0
/RELAX/SYSTEM
0.05 0.90
/END
"""
    p1 = tmp_path / "REL1_0001.rad"
    p1.write_text(deck_kerel, encoding="ascii")
    blocks1 = read_deck(str(p1))
    ec1 = parse_engine_deck(blocks1, MessageLog())
    assert ec1.kerel_active is True
    assert ec1.kerel_tstart == pytest.approx(0.05)
    assert ec1.kerel_tstop == pytest.approx(0.90)

    deck_dyna = """/RUN/RELAX_DYNA/1
1.0
/RELAX/DYNA
0.8 0.04
/END
"""
    p2 = tmp_path / "REL2_0001.rad"
    p2.write_text(deck_dyna, encoding="ascii")
    blocks2 = read_deck(str(p2))
    ec2 = parse_engine_deck(blocks2, MessageLog())
    assert ec2.dyrel_active is True
    assert ec2.dyrel_beta == pytest.approx(0.8)
    assert ec2.dyrel_period == pytest.approx(0.04)


def test_engine_keywords_group_id_in_slashes(tmp_path):
    """Verify group ID parsing in /ADYREL/15, /KEREL/20, /DYREL/25."""
    deck = """/RUN/GRP_TEST/1
1.0
/KEREL/20
0.02 0.5
/DYREL/25
0.9 0.03
/ADYREL/15
0.01 0.4
/END
"""
    p = tmp_path / "GRP_0001.rad"
    p.write_text(deck, encoding="ascii")
    blocks = read_deck(str(p))
    ec = parse_engine_deck(blocks, MessageLog())

    assert ec.kerel_active is True
    assert ec.kerel_istatg == 20
    assert ec.dyrel_active is True
    assert ec.dyrel_istatg == 25
    assert ec.adyrel_active is True
    assert ec.adyrel_istatg == 15


# ============================================================================
# 7. End-to-End Starter + Engine Simulations (/KEREL, /ADYREL, /RELAX)
# ============================================================================

_BASE_SPRING_STARTER = """\
/BEGIN
spring pair
/NODE
1 0 0 0
2 10 0 0
/SPRING/1
1 1 2
/PART/1
spring
1 1
/MAT/LAW1/1
dummy
7.8e-6
210. 0.3
/PROP/SPRING/1
k=4 M=1
1.0 4.0 0.0
/GRNOD/NODE/1
left
1
/GRNOD/NODE/2
right
2
/BCS/1
left yz
011 111 0 1
/BCS/2
right yz
011 111 0 2
/INIVEL/TRA/1
left out
-0.1 0 0 1
/INIVEL/TRA/2
right out
0.1 0 0 2
/END
"""


def test_end_to_end_spring_pair_kerel(make_deck):
    """Verify full engine run with /KEREL arrests velocity at the kinetic energy peak."""
    from pyradioss.starter.starter import run_starter
    from pyradioss.engine.engine import run_engine

    T = np.pi / 2.0
    engine = f"""\
/RUN/SPR_KEREL/1
{T}
/DT
0.02 0
/KEREL
0.0 {T}
/PRINT/-10000
/END
"""
    s, e = make_deck("SPR_KEREL", _BASE_SPRING_STARTER, engine)
    run_starter(s)
    model = run_engine(e)

    # After KE peak, velocities must have been arrested to near zero
    assert np.allclose(model.v, 0.0, atol=1e-3)


def test_end_to_end_spring_pair_adyrel(make_deck):
    """Verify full engine run with /ADYREL executes adaptive damping successfully."""
    from pyradioss.starter.starter import run_starter
    from pyradioss.engine.engine import run_engine

    T = np.pi / 2.0
    engine = f"""\
/RUN/SPR_ADYREL/1
{T}
/DT
0.02 0
/ADYREL
0.0 {T}
/PRINT/-10000
/END
"""
    s, e = make_deck("SPR_ADYREL", _BASE_SPRING_STARTER, engine)
    run_starter(s)
    model = run_engine(e)

    # Damped run must finish without errors and maintain stable kinematics
    assert np.all(np.isfinite(model.v))
    assert np.all(np.isfinite(model.x))


def test_end_to_end_spring_pair_relax(make_deck):
    """Verify full engine run with /RELAX directive runs cleanly to completion."""
    from pyradioss.starter.starter import run_starter
    from pyradioss.engine.engine import run_engine

    T = np.pi / 2.0
    engine = f"""\
/RUN/SPR_RELAX/1
{T}
/DT
0.02 0
/RELAX
0.0 {T}
/PRINT/-10000
/END
"""
    s, e = make_deck("SPR_RELAX", _BASE_SPRING_STARTER, engine)
    run_starter(s)
    model = run_engine(e)

    assert np.all(np.isfinite(model.v))
    assert np.all(np.isfinite(model.x))


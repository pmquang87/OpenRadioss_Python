"""
Tests for Dynamic Relaxation (/DYREL, /KEREL) — M580.
Upstream reference: engine/source/general_controls/damping/static.F
"""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.engine.damping import DynamicRelaxation
from pyradioss.model.model import EngineControls, Model, NodeGroup


class DummyModel:
    """Minimal model for unit testing DynamicRelaxation."""
    def __init__(self, n=4):
        self.numnod = n
        self.mass = np.ones(n, dtype=np.float64) * 2.0
        self.inertia = np.ones(n, dtype=np.float64) * 0.5
        self.node_groups = {}


def test_dyrel_initialization():
    """Verify initialization and period/beta calculation."""
    model = DummyModel(3)
    controls = EngineControls()
    controls.dyrel_active = True
    controls.dyrel_beta = 0.8
    controls.dyrel_period = 0.04
    controls.dyrel_istatg = 0

    relax = DynamicRelaxation(model, controls)
    assert relax.active is True
    assert len(relax) == 1
    assert relax.betate == pytest.approx(0.8 / 0.04)  # 20.0
    assert len(relax.dyrel_idx) == 3


def test_dyrel_viscous_damping_unit():
    """Verify viscous relaxation velocity decay and energy calculation."""
    model = DummyModel(2)
    controls = EngineControls()
    controls.dyrel_active = True
    controls.dyrel_beta = 1.0
    controls.dyrel_period = 0.1  # betate = 10.0

    relax = DynamicRelaxation(model, controls)
    dt = 0.01  # omega = 0.1
    # fac = 1 - 2*0.1 = 0.8
    v = np.array([[10.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float64)
    vr = np.array([[0.0, 2.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    v_orig = v.copy()
    vr_orig = vr.copy()

    de = relax.apply(0.0, dt, v, vr, model.mass, model.inertia)

    # v should be scaled by 0.8
    assert np.allclose(v, v_orig * 0.8)
    assert np.allclose(vr, vr_orig * 0.8)

    # Expected kinetic energy drop:
    # translational: 0.5 * (1 - 0.8^2) * (2.0 * 10^2 + 2.0 * 5^2) = 0.5 * 0.36 * (200 + 50) = 45.0
    # rotational: 0.5 * (1 - 0.8^2) * (0.5 * 2^2 + 0.5 * 1^2) = 0.5 * 0.36 * (2.0 + 0.5) = 0.45
    expected_de = 45.0 + 0.45
    assert de == pytest.approx(expected_de)


def test_dyrel_energy_conservation():
    """Verify that dissipated energy de matches actual kinetic energy reduction to machine precision."""
    model = DummyModel(5)
    controls = EngineControls()
    controls.dyrel_active = True
    controls.dyrel_beta = 0.5
    controls.dyrel_period = 0.05  # betate = 10.0

    relax = DynamicRelaxation(model, controls)
    dt = 0.005

    np.random.seed(42)
    v = np.random.randn(5, 3)
    vr = np.random.randn(5, 3)

    ke_before = float(
        0.5 * np.sum(model.mass[:, None] * v ** 2)
        + 0.5 * np.sum(model.inertia[:, None] * vr ** 2)
    )

    de = relax.apply(0.0, dt, v, vr, model.mass, model.inertia)

    ke_after = float(
        0.5 * np.sum(model.mass[:, None] * v ** 2)
        + 0.5 * np.sum(model.inertia[:, None] * vr ** 2)
    )

    # Kinetic energy reduction must equal de
    assert (ke_before - ke_after) == pytest.approx(de, rel=1e-14, abs=1e-14)


def test_kerel_peak_zeroing_unit():
    """Verify /KEREL kinetic energy peak zeroing."""
    model = DummyModel(2)
    controls = EngineControls()
    controls.kerel_active = True
    controls.kerel_tstart = 0.0
    controls.kerel_tstop = 1.0

    relax = DynamicRelaxation(model, controls)
    assert relax.active is True
    assert relax.ke_prev == 0.0

    v = np.zeros((2, 3), dtype=np.float64)
    vr = np.zeros((2, 3), dtype=np.float64)

    # Step 1: KE = 10.0 (increasing)
    v[0, 0] = 3.16227766  # 0.5 * 2.0 * v^2 = 10.0
    de1 = relax.apply(0.01, 0.01, v, vr, model.mass, model.inertia)
    assert de1 == 0.0
    assert relax.ke_prev == pytest.approx(10.0)
    assert v[0, 0] > 0.0

    # Step 2: KE = 15.0 (increasing further)
    v[0, 0] = 3.87298335  # 0.5 * 2.0 * v^2 = 15.0
    de2 = relax.apply(0.02, 0.01, v, vr, model.mass, model.inertia)
    assert de2 == 0.0
    assert relax.ke_prev == pytest.approx(15.0)

    # Step 3: KE = 12.0 (dropped! peak passed)
    v[0, 0] = 3.46410162  # 0.5 * 2.0 * v^2 = 12.0
    de3 = relax.apply(0.03, 0.01, v, vr, model.mass, model.inertia)
    # Peak detected: velocities zeroed!
    assert np.allclose(v, 0.0)
    assert de3 == pytest.approx(12.0)  # dissipated annihilated current KE (static.F:116)
    assert relax.ke_prev == 0.0

    # Step 4: Next cycle with v=0
    de4 = relax.apply(0.04, 0.01, v, vr, model.mass, model.inertia)
    assert de4 == 0.0
    assert np.allclose(v, 0.0)


def test_node_group_filtering():
    """Verify that istatg > 0 confines relaxation to the specified node group."""
    model = DummyModel(4)
    grp = NodeGroup(id=10, title="G10")
    grp.node_idx = np.array([0, 2], dtype=np.int64)
    model.node_groups[10] = grp

    controls = EngineControls()
    controls.dyrel_active = True
    controls.dyrel_beta = 1.0
    controls.dyrel_period = 0.1
    controls.dyrel_istatg = 10

    relax = DynamicRelaxation(model, controls)
    assert np.array_equal(relax.dyrel_idx, np.array([0, 2]))

    v = np.ones((4, 3), dtype=np.float64) * 5.0
    vr = np.zeros((4, 3), dtype=np.float64)
    dt = 0.01  # omega = 0.1, fac = 0.8

    relax.apply(0.0, dt, v, vr, model.mass, model.inertia)

    # Nodes 0 and 2 must be scaled by 0.8 (5.0 * 0.8 = 4.0)
    assert np.allclose(v[0], 4.0)
    assert np.allclose(v[2], 4.0)
    # Nodes 1 and 3 must be untouched (5.0)
    assert np.allclose(v[1], 5.0)
    assert np.allclose(v[3], 5.0)


def test_kerel_time_window():
    """Verify that KEREL is only active between tstart and tstop."""
    model = DummyModel(2)
    controls = EngineControls()
    controls.kerel_active = True
    controls.kerel_tstart = 0.1
    controls.kerel_tstop = 0.5

    relax = DynamicRelaxation(model, controls)
    v = np.ones((2, 3), dtype=np.float64) * 2.0
    vr = np.zeros((2, 3), dtype=np.float64)

    # t = 0.05 (before tstart): should be inactive
    de = relax.apply(0.05, 0.01, v, vr, model.mass, model.inertia)
    assert de == 0.0
    assert relax.ke_prev == 0.0
    assert np.allclose(v, 2.0)

    # t = 0.15 (within window): should track KE
    de = relax.apply(0.15, 0.01, v, vr, model.mass, model.inertia)
    assert de == 0.0
    assert relax.ke_prev > 0.0


def test_harmonic_oscillator_dyrel_simulation():
    """Simulate a 1-DOF spring-mass system with /DYREL and verify amplitude decay."""
    m = 2.0
    k = 200.0  # omega_n = 10 rad/s, T_n = 2*pi/10 = 0.6283 s
    dt = 0.002
    t_end = 2.0

    # Test 1: Undamped run
    x_undamped = 1.0
    v_undamped = 0.0
    history_undamped = []

    for _ in range(int(t_end / dt)):
        # Central-difference leapfrog
        a = (-k * x_undamped) / m
        v_undamped += a * dt
        x_undamped += v_undamped * dt
        history_undamped.append(abs(x_undamped))

    # Test 2: Damped with DYREL (beta=1.0, period=T_n)
    model = DummyModel(1)
    model.mass[0] = m
    controls = EngineControls()
    controls.dyrel_active = True
    controls.dyrel_beta = 1.0
    controls.dyrel_period = 0.6283
    controls.t_end = t_end

    relax = DynamicRelaxation(model, controls)

    x_damped = 1.0
    v_damped = np.zeros((1, 3))
    vr = np.zeros((1, 3))
    total_dissipated = 0.0
    history_damped = []

    for step in range(int(t_end / dt)):
        t = step * dt
        a = (-k * x_damped) / m
        v_damped[0, 0] += a * dt
        de = relax.apply(t, dt, v_damped, vr, model.mass, model.inertia)
        total_dissipated += de
        x_damped += v_damped[0, 0] * dt
        history_damped.append(abs(x_damped))

    # Undamped system still oscillates with peak near 1.0
    assert max(history_undamped[-200:]) > 0.8
    # Damped system amplitude must decay significantly (decay to < 0.05)
    assert history_damped[-1] < 0.05
    assert total_dissipated > 0.0


def test_harmonic_oscillator_kerel_simulation():
    """Simulate a 1-DOF spring-mass system with /KEREL and verify velocity zeroing at peaks."""
    m = 2.0
    k = 200.0  # omega_n = 10 rad/s
    dt = 0.002
    t_end = 1.0

    model = DummyModel(1)
    model.mass[0] = m
    controls = EngineControls()
    controls.kerel_active = True
    controls.kerel_tstart = 0.0
    controls.kerel_tstop = t_end
    controls.t_end = t_end

    relax = DynamicRelaxation(model, controls)

    x = 1.0
    v = np.zeros((1, 3))
    vr = np.zeros((1, 3))
    total_dissipated = 0.0
    peak_count = 0

    for step in range(int(t_end / dt)):
        t = step * dt
        a = (-k * x) / m
        v[0, 0] += a * dt
        v_before = v[0, 0]
        de = relax.apply(t, dt, v, vr, model.mass, model.inertia)
        if de > 0.0:
            total_dissipated += de
            peak_count += 1
            # Velocity must have been zeroed
            assert v[0, 0] == 0.0
        x += v[0, 0] * dt

    assert peak_count >= 1
    assert total_dissipated > 0.0
    # After freezing at KE peak, kinetic energy remains small
    assert abs(v[0, 0]) < 0.5


def test_engine_deck_parsing_and_model_wiring(tmp_path):
    """Verify that /DYREL and /KEREL engine decks wire cleanly into DynamicRelaxation."""
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.engine_keywords import parse_engine_deck

    deck = """/RUN/TEST/1
0.5
/DYREL
0.9 0.05
/KEREL
0.01 0.4
/END
"""
    p = tmp_path / "TEST_0001.rad"
    p.write_text(deck, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    ec = parse_engine_deck(blocks, log)
    assert len(log.errors) == 0

    assert ec.dyrel_active is True
    assert ec.dyrel_beta == pytest.approx(0.9)
    assert ec.dyrel_period == pytest.approx(0.05)

    assert ec.kerel_active is True
    assert ec.kerel_tstart == pytest.approx(0.01)
    assert ec.kerel_tstop == pytest.approx(0.4)

    model = DummyModel(2)
    relax = DynamicRelaxation(model, ec, log)
    assert relax.active is True
    assert relax.dyrel_active is True
    assert relax.kerel_active is True
    assert relax.betate == pytest.approx(0.9 / 0.05)


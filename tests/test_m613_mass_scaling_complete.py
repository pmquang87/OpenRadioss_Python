"""M613 — Mass Scaling & Nodal Time Stepping Complete Parity.

Fortran origin:
  common_source/tools/time_step/find_dt_target.F
  engine/source/time_step/dtnoda.F
  engine/source/time_step/dtnodarayl.F
  engine/source/time_step/find_dt_for_targeted_added_mass.F
  engine/source/input/freform.F
  hm_cfg_files/config/CFG/radioss2018/CARDS/eng_stop.cfg
  hm_cfg_files/config/CFG/radioss2017/CARDS/eng_dt_noda.cfg

Tests verify:
  1. 1.00001 factor preventing repeated micro-additions.
  2. Target added mass auto-tuning matching find_dt_target.F.
  3. Rayleigh damping stiffness scaling reducing critical dt.
  4. Critical entity tracking (crit_node_id, crit_elem_type).
  5. /DT/NODA/STOP halting when dt < dt_min.
  6. /DT/NODA/SET acceleration scaling.
  7. Top mass nodes reporting (get_top_mass_nodes and summary).
  8. Parsing of /STOP (Emax, Mmax, Nmax) and /DT/NODA/CST/<grnd>.
"""

from __future__ import annotations

import io
import contextlib
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.engine.mass_scaling import NodalTimeStep, compute_target_dt
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.model.model import EngineControls, Model


class _MockGroup:
    def __init__(self, conn, mass, dt_iner=None):
        self.conn = conn
        self.state = {"mass": mass}
        if dt_iner is not None:
            self.state["dt_iner"] = dt_iner


class _MockModel:
    def __init__(self, numnod, mass, groups=None, node_ids=None):
        self.numnod = numnod
        self.mass = mass.astype(np.float64)
        self._groups = groups or {}
        self.node_ids = node_ids
        self.inertia = np.zeros(numnod, dtype=np.float64)
        self.v = np.zeros((numnod, 3), dtype=np.float64)
        self.x0 = np.zeros((numnod, 3), dtype=np.float64)

    def element_groups(self):
        return self._groups.items()


def _make_noda(dt_noda="CST", dt_scale=0.9, dt_min=0.05, numnod=4, node_mass=1.0, node_ids=None):
    conn = np.array([[0, 1, 2, 3]])
    elem_mass = np.array([float(node_mass * 4)])
    group = _MockGroup(conn, elem_mass)
    mass = np.full(numnod, node_mass, dtype=np.float64)
    model = _MockModel(numnod=numnod, mass=mass, groups={"shells": group}, node_ids=node_ids)
    controls = EngineControls(dt_noda=dt_noda, dt_scale=dt_scale, dt_min=dt_min)
    log = MessageLog()
    noda = NodalTimeStep(model, controls, log)
    return model, noda, log


# ======================================================================
# 1. 1.00001 factor preventing repeated micro-additions
# ======================================================================

def test_1_00001_factor_prevents_repeated_micro_additions():
    """Fortran dtnoda.F applies 1.00001 (ONEP00001) factor to mass additions.

    On cycle 1, mass is scaled up to 1.00001 * K * (dt_min/dt_sca)^2 / 2.
    On cycle 2, with identical stiffness, m_req - mass <= 0 strictly,
    so exactly 0 additional mass is added (preventing repeated micro-additions).
    """
    dt_min, dt_sca = 0.05, 1.0
    model, noda, log = _make_noda(dt_noda="CST", dt_scale=dt_sca, dt_min=dt_min, node_mass=1.0)
    
    # Element critical dt = 0.01 -> K = 2 * 1.0 / 0.01^2 = 20000.0 per node
    dt_e = np.array([0.01])
    noda.assemble([dt_e])

    mass_eff = model.mass.copy()
    inv_mass = 1.0 / mass_eff
    v = np.zeros((4, 3))

    # Cycle 1: mass scaling occurs
    dt1 = noda.apply(mass_eff, inv_mass, v, t=0.0)
    mass_after_c1 = model.mass.copy()
    added_c1 = noda.mass_added
    assert added_c1 > 0.0

    # Required mass should be 1.00001 * 20000 * 0.05^2 / 2 = 25.00025
    expected_m = 1.00001 * 20000.0 * (dt_min / dt_sca) ** 2 / 2.0
    for i in range(4):
        assert model.mass[i] == pytest.approx(expected_m, rel=1e-12)
        assert mass_eff[i] == pytest.approx(expected_m, rel=1e-12)

    # Cycle 2: assemble again with exact same stiffness
    noda.assemble([dt_e])
    dt2 = noda.apply(mass_eff, inv_mass, v, t=dt1)
    added_c2 = noda.mass_added

    # Zero micro-additions on cycle 2
    assert added_c2 == added_c1
    np.testing.assert_array_equal(model.mass, mass_after_c1)
    assert dt2 == pytest.approx(dt1, rel=1e-12)


# ======================================================================
# 2. Target added mass auto-tuning matching find_dt_target.F
# ======================================================================

def test_compute_target_dt_matching_find_dt_target():
    """Matching common_source/tools/time_step/find_dt_target.F:

    compute_target_dt solves for dt* such that added mass equals
    target_percent_addmass * total_mass.
    """
    ms = np.array([1.0, 2.0, 3.0, 4.0])
    stifn = np.array([10000.0, 15000.0, 12000.0, 8000.0])
    total_mass = float(ms.sum())  # 10.0
    target_percent = 0.05         # 5% added mass -> 0.5 mass target
    dt_scale = 0.9

    dt_target = compute_target_dt(target_percent, dt_scale=dt_scale,
                                  total_mass=total_mass, ms=ms, stifn=stifn)
    assert dt_target > 0.0

    # Verify that applying dt_target produces the requested added mass
    tau_star = 0.5 * (dt_target / dt_scale) ** 2
    dm_node = np.maximum(0.0, stifn * tau_star - ms)
    total_dm = float(dm_node.sum())

    # Ratio total_dm / total_mass should equal target_percent
    assert (total_dm / total_mass) == pytest.approx(target_percent, rel=1e-10)

    # Test via NodalTimeStep method
    model, noda, log = _make_noda(dt_noda="CST", dt_scale=dt_scale, dt_min=0.01)
    noda.stifn = stifn.copy()
    model.mass = ms.copy()
    noda.mass0 = total_mass
    noda.free = np.ones(4, dtype=bool)

    dt_method = noda.compute_target_dt(target_percent)
    assert dt_method == pytest.approx(dt_target, rel=1e-12)


def test_compute_target_dt_single_node():
    """Single-node model where target_dt is derived analytically."""
    ms = np.array([2.0])
    stifn = np.array([50000.0])
    total_mass = 2.0
    threshold = 0.10  # 10%
    dt_scale = 1.0

    # For 1 node: dt_target = dt_scale * sqrt(2 * (total_mass * threshold + M) / K)
    expected_dt = dt_scale * np.sqrt(2.0 * (total_mass * threshold + ms[0]) / stifn[0])
    dt = compute_target_dt(threshold, dt_scale=dt_scale, total_mass=total_mass,
                           ms=ms, stifn=stifn)
    assert dt == pytest.approx(expected_dt, rel=1e-12)


# ======================================================================
# 3. Rayleigh damping stiffness scaling reducing critical dt
# ======================================================================

def test_rayleigh_damping_stiffness_scaling_reduces_dt():
    """Matching engine/source/time_step/dtnodarayl.F:

    BB = beta / dt0 + 0.5 * alpha * dt0
    FAC = sqrt(BB^2 + 1) - BB < 1
    COEFF = 1 / FAC^2 > 1
    K <- K * COEFF
    The critical nodal dt is reduced by factor FAC.
    """
    model, noda, log = _make_noda(dt_noda="NODA", dt_scale=1.0, dt_min=0.0, node_mass=2.0)
    noda.stifn[0] = 10000.0
    noda.free[0] = True

    dt0 = np.sqrt(2.0 * model.mass[0] / noda.stifn[0])  # sqrt(4 / 10000) = 0.02
    alpha = 40.0
    beta = 2e-4

    bb = (beta / dt0) + 0.5 * alpha * dt0
    fac = np.sqrt(bb**2 + 1.0) - bb
    coeff = 1.0 / (fac**2)
    expected_k = 10000.0 * coeff
    expected_dt = fac * dt0

    noda.apply_rayleigh_damping_stiffness(alpha=alpha, beta=beta)
    assert noda.stifn[0] == pytest.approx(expected_k, rel=1e-12)
    assert noda.stifn[0] > 10000.0  # stiffness increased

    # Apply returns reduced dt
    mass_eff = model.mass.copy()
    dt = noda.apply(mass_eff, 1.0 / mass_eff, model.v, t=0.0)
    assert dt == pytest.approx(expected_dt, rel=1e-12)
    assert dt < dt0  # critical dt is reduced by damping


# ======================================================================
# 4. Critical entity tracking
# ======================================================================

def test_critical_entity_tracking():
    """Verify self.crit_node_id and self.crit_elem_type track the worst node."""
    node_ids = np.array([101, 102, 103, 104], dtype=np.int64)
    model, noda, log = _make_noda(dt_noda="NODA", numnod=4, node_mass=1.0, node_ids=node_ids)

    # Node 2 (ID 103) has the highest stiffness -> smallest dt
    noda.stifn = np.array([1000.0, 2000.0, 50000.0, 3000.0])
    noda.free = np.ones(4, dtype=bool)

    mass_eff = model.mass.copy()
    dt = noda.apply(mass_eff, 1.0 / mass_eff, model.v, t=0.0)

    assert noda.crit_elem_type == "NODE"
    assert noda.crit_node_id == 103
    assert dt == pytest.approx(np.sqrt(2.0 * 1.0 / 50000.0), rel=1e-12)


def test_critical_entity_tracking_fallback_index():
    """Without node_ids, crit_node_id defaults to 1-based index."""
    model, noda, log = _make_noda(dt_noda="NODA", numnod=3, node_mass=1.0, node_ids=None)
    noda.stifn = np.array([1000.0, 80000.0, 2000.0])  # index 1 is smallest
    noda.free = np.ones(3, dtype=bool)

    mass_eff = model.mass.copy()
    noda.apply(mass_eff, 1.0 / mass_eff, model.v, t=0.0)
    assert noda.crit_node_id == 2  # 1-based index (0 + 2 = 2)


# ======================================================================
# 5. /DT/NODA/STOP halting when dt < dt_min
# ======================================================================

def test_dt_noda_stop_halting():
    """In STOP mode, do NOT add mass. If dt < dt_min, flag stop and report node."""
    node_ids = np.array([501, 502, 503, 504], dtype=np.int64)
    model, noda, log = _make_noda(dt_noda="STOP", dt_scale=1.0, dt_min=0.05,
                                  numnod=4, node_mass=1.0, node_ids=node_ids)

    # Element dt = 0.01 < dt_min=0.05
    noda.assemble([np.array([0.01])])

    mass_eff = model.mass.copy()
    dt = noda.apply(mass_eff, 1.0 / mass_eff, model.v, t=0.0)

    assert noda.stopped is True
    assert noda.stop_node in node_ids
    assert noda.mass_added == 0.0  # NO mass added in STOP mode
    np.testing.assert_array_equal(model.mass, np.ones(4))


def test_dt_noda_stop_no_halt_when_above_min():
    """When dt >= dt_min, stopped flag remains False."""
    model, noda, log = _make_noda(dt_noda="STOP", dt_scale=1.0, dt_min=0.005,
                                  numnod=4, node_mass=1.0)
    # Element dt = 0.01 >= dt_min=0.005
    noda.assemble([np.array([0.01])])

    mass_eff = model.mass.copy()
    dt = noda.apply(mass_eff, 1.0 / mass_eff, model.v, t=0.0)

    assert noda.stopped is False
    assert noda.stop_node == 0


# ======================================================================
# 6. /DT/NODA/SET acceleration scaling
# ======================================================================

def test_dt_noda_set_acceleration_scaling():
    """In SET mode, scale acceleration directly: acc[idx] *= (model.mass[idx] / m_req)."""
    model, noda, log = _make_noda(dt_noda="SET", dt_scale=1.0, dt_min=0.05,
                                  numnod=4, node_mass=1.0)
    
    # Element dt = 0.01 -> K = 2 * 1.0 / 0.01^2 = 20000.0
    # m_req = K * dt_min^2 / 2 = 20000 * 0.05^2 / 2 = 25.0
    # Factor = 1.0 / 25.0 = 0.04
    noda.assemble([np.array([0.01])])

    mass_eff = model.mass.copy()
    dt = noda.apply(mass_eff, 1.0 / mass_eff, model.v, t=0.0)

    assert noda.mass_added == 0.0  # NO mass added

    acc = np.full((4, 3), 100.0)
    noda.apply_set_acceleration(acc)

    # Every node was scaled by factor 0.04
    for i in range(4):
        assert acc[i, 0] == pytest.approx(4.0, rel=1e-12)
        assert acc[i, 1] == pytest.approx(4.0, rel=1e-12)
        assert acc[i, 2] == pytest.approx(4.0, rel=1e-12)


# ======================================================================
# 7. Top mass nodes reporting
# ======================================================================

def test_top_mass_nodes_reporting():
    """get_top_mass_nodes returns sorted top n nodes: (node_id, delta_m, delta_m / m0)."""
    node_ids = np.array([10, 20, 30, 40], dtype=np.int64)
    model, noda, log = _make_noda(dt_noda="CST", dt_scale=1.0, dt_min=0.05,
                                  numnod=4, node_mass=1.0, node_ids=node_ids)

    # Asymmetrical stiffness
    noda.stifn = np.array([10000.0, 30000.0, 5000.0, 20000.0])
    noda.free = np.ones(4, dtype=bool)

    mass_eff = model.mass.copy()
    noda.apply(mass_eff, 1.0 / mass_eff, model.v, t=0.0)

    top = noda.get_top_mass_nodes(n=3)
    assert len(top) == 3

    # Node 20 had the highest stiffness (30000) -> highest added mass
    assert top[0][0] == 20
    # Node 40 had second highest stiffness (20000) -> second highest added mass
    assert top[1][0] == 40
    # Node 10 had third highest stiffness (10000)
    assert top[2][0] == 10

    # Ensure sorted descending
    assert top[0][1] >= top[1][1] >= top[2][1]

    # Check relative mass increase delta_m / m0
    for nid, dm, rel in top:
        assert rel == pytest.approx(dm / 1.0, rel=1e-12)

    # Test summary log
    out_log = MessageLog()
    out_buf = io.StringIO()
    out_log.attach_listing(out_buf)
    with contextlib.redirect_stdout(io.StringIO()):
        noda.summary(out_log)
    text = out_buf.getvalue()
    assert "TOP NODES BY ADDED MASS" in text
    assert "NODE       20" in text


# ======================================================================
# 8. Parsing of /STOP and /DT/NODA/CST/<grnd>
# ======================================================================

def _parse_engine_text(tmp_path, text):
    p = tmp_path / "TEST_0001.rad"
    p.write_text(text)
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        ec = parse_engine_deck(read_deck(str(p)), log)
    return ec


def test_stop_card_parsing(tmp_path):
    """Card format: Emax Mmax Nmax.

    vals[0] -> energy_error_stop
    vals[1] -> mass_error_stop
    vals[2] -> nodal_mass_error_stop
    stop_nstep must NOT be overwritten by vals[2].
    """
    deck = (
        "/RUN/TEST/1\n"
        "1.0\n"
        "/STOP\n"
        "25.0 5.0 1.5 1 1\n"
    )
    ec = _parse_engine_text(tmp_path, deck)
    assert ec.energy_error_stop == 25.0
    assert ec.mass_error_stop == 5.0
    assert ec.nodal_mass_error_stop == 1.5
    assert ec.stop_nstep == 0


def test_stop_card_parsing_zero_keeps_default(tmp_path):
    """When Emax is 0.0, default 15.0 stays."""
    deck = (
        "/RUN/TEST/1\n"
        "1.0\n"
        "/STOP\n"
        "0.0 10.0 3.0\n"
    )
    ec = _parse_engine_text(tmp_path, deck)
    assert ec.energy_error_stop == 15.0
    assert ec.mass_error_stop == 10.0
    assert ec.nodal_mass_error_stop == 3.0


def test_dt_noda_cst_grnod_header_parsing(tmp_path):
    """Header format /DT/NODA/CST/<grnd>."""
    deck = (
        "/RUN/TEST/1\n"
        "1.0\n"
        "/DT/NODA/CST/105\n"
        "0.85 1e-6 0.03\n"
    )
    ec = _parse_engine_text(tmp_path, deck)
    assert ec.dt_noda == "CST"
    assert ec.dt_noda_grnod == 105
    assert ec.dt_scale == 0.85
    assert ec.dt_min == 1e-6
    assert ec.dt_noda_percent_addmass == 0.03


def test_dt_noda_cst_grnod_card2_parsing(tmp_path):
    """Card 2 contains the node group id."""
    deck = (
        "/RUN/TEST/1\n"
        "1.0\n"
        "/DT/NODA/CST\n"
        "0.9 2e-6 0.02\n"
        "201\n"
    )
    ec = _parse_engine_text(tmp_path, deck)
    assert ec.dt_noda == "CST"
    assert ec.dt_noda_grnod == 201
    assert ec.dt_scale == 0.9
    assert ec.dt_min == 2e-6
    assert ec.dt_noda_percent_addmass == 0.02


def test_dt_noda_stop_and_set_parsing(tmp_path):
    """Check /DT/NODA/STOP and /DT/NODA/SET parsing."""
    deck_stop = (
        "/RUN/TEST/1\n"
        "1.0\n"
        "/DT/NODA/STOP\n"
        "0.9 5e-6\n"
    )
    ec_stop = _parse_engine_text(tmp_path, deck_stop)
    assert ec_stop.dt_noda == "STOP"
    assert ec_stop.dt_scale == 0.9
    assert ec_stop.dt_min == 5e-6

    deck_set = (
        "/RUN/TEST/1\n"
        "1.0\n"
        "/DT/NODA/SET\n"
        "0.9 5e-6\n"
    )
    ec_set = _parse_engine_text(tmp_path, deck_set)
    assert ec_set.dt_noda == "SET"
    assert ec_set.dt_scale == 0.9
    assert ec_set.dt_min == 5e-6

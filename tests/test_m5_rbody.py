"""Tests for M5 Rigid Body, Coloring, and Engine keywords fixes."""
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.engine.coloring import compute_element_colors
from pyradioss.engine.kinematics import LoadsAndConstraints
from pyradioss.engine.rigid_body import RigidBodyEngine
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.model.entities import NodeGroup, RigidBody
from pyradioss.model.model import Model


class _DummyFunc:
    def __init__(self, val: float = 1.0, slope: float = 0.0):
        self.val = float(val)
        self.slope = float(slope)

    def eval(self, t: float) -> float:
        return self.val + self.slope * float(t)


def test_rbody_impvel_time_window_and_facx():
    """Verify translational /IMPVEL on rigid body master respects facx and [tstart, tstop]."""
    model = Model()
    num_nodes = 3
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    model.x = model.x0.copy()
    model.v = np.zeros((num_nodes, 3), dtype=np.float64)
    model.vr = np.zeros((num_nodes, 3), dtype=np.float64)
    model.mass = np.array([10.0, 5.0, 5.0], dtype=np.float64)
    model.mass0 = model.mass.copy()
    model.inertia = np.full(num_nodes, 0.1, dtype=np.float64)

    slaves = np.array([1, 2], dtype=np.int64)
    rb = RigidBody(
        id=1,
        master_id=1,
        grnod_id=10,
        master=0,
        slaves=slaves,
        mass_total=20.0,
        xg=model.x0[0].copy(),
        J=np.eye(3),
        kind="RBODY",
    )
    model.rbodies = [rb]
    model.node_groups[10] = NodeGroup(id=10, node_ids=model.node_ids[slaves], node_idx=slaves)

    log = MessageLog()
    loads = LoadsAndConstraints(model, log)
    # Impvel on master node 0: dof=0 (X), scale=2.0, facx=3.0, tstart=0.1, tstop=0.5
    fct = _DummyFunc(val=1.0, slope=2.0)  # eval(t) = 1.0 + 2.0*t
    # (idx, dof, fct, scale, facx, tstart, tstop)
    loads.impvel = [(np.array([0]), 0, fct, 2.0, 3.0, 0.1, 0.5)]

    rbe = RigidBodyEngine(rb, model, loads, log)
    assert len(rbe.drives) == 1
    # Check stored tuple has 6 elements (dof, fct, scale, facx, t0, t1)
    dof, f, scale, facx, t0, t1 = rbe.drives[0]
    assert dof == 0
    assert scale == 2.0
    assert facx == 3.0
    assert t0 == 0.1
    assert t1 == 0.5

    fint = np.zeros((num_nodes, 3))
    fext = np.zeros((num_nodes, 3))
    fcont = np.zeros((num_nodes, 3))
    mint = np.zeros((num_nodes, 3))
    v = np.zeros((num_nodes, 3))
    vr = np.zeros((num_nodes, 3))
    x = model.x0.copy()
    dt = 0.01

    # Cycle at t_next = 0.05 (< t0 = 0.1): drive must NOT be active
    rbe.v_ref = np.zeros(3)
    rbe.advance(fint, fext, fcont, mint, v, vr, x, dt, t_next=0.05)
    assert rbe.v_ref[0] == 0.0

    # Cycle at t_next = 0.2 (within [0.1, 0.5]): drive active with facx scaling
    # Wave 2 fix: evaluated at t_mid = t_next - 0.5*dt = 0.2 - 0.005 = 0.195
    # vimp = scale * fct.eval(t_mid * facx) = 2.0 * (1.0 + 2.0 * (0.195 * 3.0)) = 4.34
    rbe.advance(fint, fext, fcont, mint, v, vr, x, dt, t_next=0.2)
    assert np.isclose(rbe.v_ref[0], 4.34)

    # Cycle at t_next = 0.6 (> t1 = 0.5): drive must NOT be active
    rbe.v_ref[0] = 0.0
    rbe.advance(fint, fext, fcont, mint, v, vr, x, dt, t_next=0.6)
    assert rbe.v_ref[0] == 0.0


def test_coloring_negative_indices():
    """Verify compute_element_colors handles negative node indices without error."""
    conn = np.array([
        [0, 1, 2, -1],
        [1, 2, 3, -1],
        [2, 3, 4, 100],
    ], dtype=np.int32)
    num_nodes = 5
    color_indices, color_offsets = compute_element_colors(conn, num_nodes)
    assert len(color_indices) == 3
    assert color_offsets[0] == 0
    assert color_offsets[-1] == 3


def test_dt_parser_state_leak(tmp_path):
    """Verify /DT/AMS does not leak into subsequent /DT keywords like /DT/BRICK."""
    deck = """/DT/AMS
0.9 1.0e-6
0.01
100
/DT/BRICK/CST
0.8 1.0e-7
"""
    p = tmp_path / "deck.rad"
    p.write_text(deck)
    blocks = read_deck(str(p))
    log = MessageLog()
    ec = parse_engine_deck(blocks, log)
    assert ec.dt_ams is True
    assert np.isclose(ec.dt_ams_tol, 0.01)
    assert ec.dt_ams_itmax == 100
    assert "BRICK" in ec.dt_controls
    assert np.isclose(ec.dt_controls["BRICK"]["scale"], 0.8)
    assert np.isclose(ec.dt_controls["BRICK"]["dt_min"], 1.0e-7)
    assert np.isclose(ec.dt_ams_tol, 0.01)
    assert ec.dt_ams_itmax == 100

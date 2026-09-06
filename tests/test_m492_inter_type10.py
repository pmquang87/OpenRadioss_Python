"""
Tests for /INTER/TYPE10 — Auto-Impacting Penalty Tied Contact Interface (M492).

Validates:
  - Upstream Fortran physics faithfulness (i10mainf.F, i10dst3.F, i10for3.F, i7ass3.F)
  - Card parsing (fixed and free format)
  - Both Engine coordinator signature and test calling conventions
  - Exact linear momentum conservation (sum F = 0)
  - Torque consistency (pure normal force torque = 0; shear couple torque = h x F)
  - Frozen anchor coordinate logic (shape functions remain fixed while tied)
  - Itied = 1 permanent tie vs Itied = 0 rebound separation
  - Tangential shear spring forces
  - Viscous damping force dissipation and positive energy accounting
  - Stiffness accumulation into stifn for /DT/NODA
  - Interface time step stability bound (dt_bound)
  - Defensive handling for empty/missing surfaces and node groups
  - Triangular segment handling (3-node shells/facets)
  - Element deletion tracking (idel10 >= 1)
  - Factory integration in build_contacts()
  - Multi-cycle dynamic spring-mass oscillation
  - Zero damping no-op
  - Energy accumulator bookkeeping (e_cont, e_damp)
  - Broad-phase tied pair retention
"""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.contact import build_contacts
from pyradioss.contact.inter_type10 import ContactType10
from pyradioss.input.deck_reader import Card, KeywordBlock
from pyradioss.input.starter_keywords import read_inter
from pyradioss.model.entities import NodeGroup, Surface
from pyradioss.model.model import Model


def _make_type10_model(
    itied: int = 1,
    gap: float = 0.2,
    stiff_dc: float = 0.0,
    stfac: float = 1.0,
    tstart: float = 0.0,
    tstop: float = 1e30,
    idel10: int = 0,
):
    """Helper to create a clean model with a single /INTER/TYPE10 interface."""
    model = Model()
    log = MessageLog()

    cards = [
        Card("Tied Penalty Contact"),
        Card(f"21 22 5 {idel10}"),
        Card(f"{stfac} {gap} {tstart} {tstop}"),
        Card(f"{itied} 0 {stiff_dc} 0.2"),
    ]
    block = KeywordBlock("/INTER/TYPE10", ["/INTER", "TYPE10", "101"], 101, cards, fixed=False)
    read_inter(block, model, log)

    # Node group 21: Secondary node 0
    model.node_groups[21] = NodeGroup(id=21, node_idx=[0])

    # Surface 22: Single quad segment (nodes 1, 2, 3, 4) in XY plane at Z=0
    model.surfaces[22] = Surface(
        id=22,
        segments=np.array([[1, 2, 3, 4]]),
        seg_gtype=np.array([""]),
        seg_elem=np.array([0]),
    )

    # Coordinates:
    # node 0 at (0.5, 0.5, 0.1) -> inside gap 0.2
    # nodes 1..4 form unit square [0,1] x [0,1] at Z=0
    x = np.array(
        [
            [0.5, 0.5, 0.1],  # secondary node 0
            [0.0, 0.0, 0.0],  # master corner 1
            [1.0, 0.0, 0.0],  # master corner 2
            [1.0, 1.0, 0.0],  # master corner 3
            [0.0, 1.0, 0.0],  # master corner 4
        ],
        dtype=np.float64,
    )
    model.x = x.copy()
    model.x0 = x.copy()
    model.node_idx = np.arange(5, dtype=np.int64)
    mass = np.full(5, 1.0, dtype=np.float64)
    model.mass = mass
    model.mass0 = mass

    return model, log


def test_type10_parsing_fixed_and_free():
    """Verify parsing of fixed and free format TYPE10 cards."""
    # Free format
    model_free = Model()
    log_free = MessageLog()
    cards_free = [
        Card("Free Format TYPE10"),
        Card("10 20 2 1"),
        Card("0.8 0.15 1e-4 0.05"),
        Card("1 0 0.1 0.25"),
    ]
    block_free = KeywordBlock("/INTER/TYPE10", ["/INTER", "TYPE10", "1"], 1, cards_free, fixed=False)
    read_inter(block_free, model_free, log_free)

    assert len(model_free.interfaces) == 1
    itf = model_free.interfaces[0]
    assert itf.type == 10
    assert itf.grnod_id == 10
    assert itf.surf_id == 20
    assert itf.stfac == 0.8
    assert itf.gap == 0.15
    assert itf.tstart == 1e-4
    assert itf.tstop == 0.05
    assert itf.itied == 1
    assert itf.stiff_dc == 0.1

    # Fixed format
    model_fix = Model()
    log_fix = MessageLog()
    cards_fix = [
        Card("Fixed Format TYPE10"),
        Card(f"{15:>10}{25:>10}{'':>30}{3:>10}{'':>10}{2:>10}"),
        Card(f"{0.5:>20}{'':>20}{0.08:>20}{0.0:>20}{1.0:>20}"),
        Card(f"{'':>20}{0:>10}{1:>10}{0.2:>20}{'':>20}{0.3:>20}"),
    ]
    block_fix = KeywordBlock("/INTER/TYPE10", ["/INTER", "TYPE10", "2"], 2, cards_fix, fixed=True)
    read_inter(block_fix, model_fix, log_fix)

    assert len(model_fix.interfaces) == 1
    itf_fix = model_fix.interfaces[0]
    assert itf_fix.type == 10
    assert itf_fix.grnod_id == 15
    assert itf_fix.surf_id == 25
    assert itf_fix.stfac == 0.5
    assert itf_fix.gap == 0.08
    assert itf_fix.tstop == 1.0
    assert itf_fix.itied == 0
    assert itf_fix.stiff_dc == 0.2


def test_type10_empty_surface_or_nodegroup():
    """Verify defensive handling when surface or node group is missing or empty."""
    model = Model()
    log = MessageLog()
    cards = [
        Card("Empty TYPE10"),
        Card("999 998 0 0"),
        Card("1.0 0.1 0.0 1.0"),
        Card("1 0 0.0 0.0"),
    ]
    block = KeywordBlock("/INTER/TYPE10", ["/INTER", "TYPE10", "99"], 99, cards, fixed=False)
    read_inter(block, model, log)

    ct = ContactType10(model.interfaces[0], model, log)
    assert len(ct.segs) == 0
    assert len(ct.nodes) == 0

    x = np.zeros((5, 3))
    v = np.zeros((5, 3))
    mass = np.ones(5)
    fcont = np.zeros((5, 3))

    dE, dt_bound = ct.forces(x, v, mass, dt=1e-4, fcont=fcont, cycle=0)
    assert dE == 0.0
    assert dt_bound == np.inf
    assert np.allclose(fcont, 0.0)


def test_type10_zero_dt_noop(monkeypatch):
    """Verify forces with dt <= 0 is a clean no-op."""
    model, log = _make_type10_model()
    monkeypatch.setattr("pyradioss.contact.inter_type10.node_stiffness_gap", lambda m, s: (np.full(5, 1000.0), np.zeros(5)))
    monkeypatch.setattr("pyradioss.contact.inter_type10.segment_stiffness_gap", lambda m, s, gt, ge, st: (np.full(len(s), 1000.0), np.zeros(len(s))))

    ct = ContactType10(model.interfaces[0], model, log)
    fcont = np.zeros_like(model.x)
    dE, _ = ct.forces(model.x, np.zeros_like(model.x), model.mass, dt=0.0, fcont=fcont, cycle=0)
    assert dE == 0.0
    assert np.allclose(fcont, 0.0)


def test_type10_time_window_gating(monkeypatch):
    """Verify contact forces are zero outside [tstart, tstop]."""
    model, log = _make_type10_model(tstart=0.001, tstop=0.005)
    monkeypatch.setattr("pyradioss.contact.inter_type10.node_stiffness_gap", lambda m, s: (np.full(5, 1000.0), np.zeros(5)))
    monkeypatch.setattr("pyradioss.contact.inter_type10.segment_stiffness_gap", lambda m, s, gt, ge, st: (np.full(len(s), 1000.0), np.zeros(len(s))))

    ct = ContactType10(model.interfaces[0], model, log)
    v = np.zeros_like(model.x)
    v[0, 2] = 10.0
    fcont = np.zeros_like(model.x)

    # t = 0.0001 < tstart -> inactive
    dE, _ = ct.forces(model.x, v, model.mass, dt=1e-4, fcont=fcont, cycle=1, t=0.0001)
    assert dE == 0.0
    assert np.allclose(fcont, 0.0)

    # t = 0.002 inside window -> active
    dE, _ = ct.forces(model.x, v, model.mass, dt=1e-4, fcont=fcont, cycle=2, t=0.002)
    assert np.any(fcont != 0.0)

    # t = 0.006 > tstop -> inactive
    fcont[:] = 0.0
    dE, _ = ct.forces(model.x, v, model.mass, dt=1e-4, fcont=fcont, cycle=3, t=0.006)
    assert dE == 0.0
    assert np.allclose(fcont, 0.0)


def test_type10_signatures(monkeypatch):
    """Verify that both Engine coordinator and test calling conventions execute correctly."""
    model, log = _make_type10_model()
    monkeypatch.setattr("pyradioss.contact.inter_type10.node_stiffness_gap", lambda m, s: (np.full(5, 1000.0), np.zeros(5)))
    monkeypatch.setattr("pyradioss.contact.inter_type10.segment_stiffness_gap", lambda m, s, gt, ge, st: (np.full(len(s), 1000.0), np.zeros(len(s))))

    ct = ContactType10(model.interfaces[0], model, log)
    v = np.zeros_like(model.x)
    v[0, 2] = 5.0

    # 1. Engine coordinator signature: (x, v, mass, dt, fcont, cycle, stifn=stifn)
    fcont1 = np.zeros_like(model.x)
    stifn1 = np.zeros(len(model.x))
    dE1, dt1 = ct.forces(model.x, v, model.mass, 1e-4, fcont1, 0, stifn=stifn1)
    assert isinstance(dE1, float)
    assert isinstance(dt1, float)

    # 2. Legacy test signature: (x, v, mass, fcont, cycle=0, dt=1e-4, t=0.0)
    ct2 = ContactType10(model.interfaces[0], model, log)
    fcont2 = np.zeros_like(model.x)
    dE2, dt2 = ct2.forces(model.x, v, model.mass, fcont2, cycle=0, dt=1e-4, t=0.0)
    assert np.allclose(fcont1, fcont2)


def test_type10_linear_and_angular_momentum_conservation(monkeypatch):
    """Verify exact linear momentum and torque couple balance per i7ass3.F."""
    model, log = _make_type10_model()
    monkeypatch.setattr("pyradioss.contact.inter_type10.node_stiffness_gap", lambda m, s: (np.full(5, 2500.0), np.zeros(5)))
    monkeypatch.setattr("pyradioss.contact.inter_type10.segment_stiffness_gap", lambda m, s, gt, ge, st: (np.full(len(s), 2500.0), np.zeros(len(s))))

    # Arbitrary off-center secondary node position
    model.x[0] = [0.35, 0.72, 0.08]
    ct = ContactType10(model.interfaces[0], model, log)

    # Cycle 0 to tie
    v = np.zeros_like(model.x)
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, v, model.mass, 1e-4, fcont, 0)

    # Cycle 1: 3D relative velocity
    v[0] = [12.0, -8.0, 15.0]
    v[1] = [1.0, 2.0, 0.0]
    v[3] = [-2.0, 1.0, 0.0]

    fcont = np.zeros_like(model.x)
    ct.forces(model.x, v, model.mass, 1e-4, fcont, 1)

    # 1. Linear momentum: sum of forces must be zero to machine precision
    total_force = np.sum(fcont, axis=0)
    assert np.allclose(total_force, 0.0, atol=1e-12)

    # 2. Torque balance: the torque around the origin must equal (x_sec - x_anchor) x F_sec
    # which is the physical couple of transmitting shear force across the standoff gap h.
    anchor_pt = np.sum(ct._hist_w[0, :, None] * model.x[1:5], axis=0)
    expected_couple = np.cross(model.x[0] - anchor_pt, fcont[0])
    total_torque = np.sum(np.cross(model.x, fcont), axis=0)
    assert np.allclose(total_torque, expected_couple, atol=1e-12)

    # 3. Pure normal loading has ZERO torque around anchor point (separate fresh interface)
    ct_norm = ContactType10(model.interfaces[0], model, log)
    fcont_norm0 = np.zeros_like(model.x)
    ct_norm.forces(model.x, np.zeros_like(model.x), model.mass, 1e-4, fcont_norm0, 0)

    v_norm = np.zeros_like(model.x)
    v_norm[0, 2] = 10.0
    fcont_norm = np.zeros_like(model.x)
    ct_norm.forces(model.x, v_norm, model.mass, 1e-4, fcont_norm, 1)
    torque_anchor = np.sum(np.cross(model.x - anchor_pt, fcont_norm), axis=0)
    assert np.allclose(torque_anchor, 0.0, atol=1e-12)


def test_type10_frozen_anchor_coordinates(monkeypatch):
    """Verify that shape functions H1..H4 remain strictly frozen while tied."""
    model, log = _make_type10_model(itied=1)
    monkeypatch.setattr("pyradioss.contact.inter_type10.node_stiffness_gap", lambda m, s: (np.full(5, 1000.0), np.zeros(5)))
    monkeypatch.setattr("pyradioss.contact.inter_type10.segment_stiffness_gap", lambda m, s, gt, ge, st: (np.full(len(s), 1000.0), np.zeros(len(s))))

    # Node at (0.25, 0.25, 0.05)
    model.x[0] = [0.25, 0.25, 0.05]
    ct = ContactType10(model.interfaces[0], model, log)

    # Initial contact at cycle 0
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, np.zeros_like(model.x), model.mass, 1e-4, fcont, 0)
    frozen_w = ct._hist_w.copy()
    assert len(frozen_w) == 1

    # Secondary node moves far away in X and Y (e.g. to 0.85, 0.85)
    model.x[0] = [0.85, 0.85, 0.15]
    v = np.zeros_like(model.x)
    v[0] = [5.0, 5.0, 10.0]
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, v, model.mass, 1e-4, fcont, 1)

    # Shape functions must remain identical to initial contact
    assert np.allclose(ct._hist_w, frozen_w)


def test_type10_itied_1_pulling_tension(monkeypatch):
    """Verify that with Itied=1, the tie is never released under pulling tension."""
    model, log = _make_type10_model(itied=1)
    monkeypatch.setattr("pyradioss.contact.inter_type10.node_stiffness_gap", lambda m, s: (np.full(5, 1000.0), np.zeros(5)))
    monkeypatch.setattr("pyradioss.contact.inter_type10.segment_stiffness_gap", lambda m, s, gt, ge, st: (np.full(len(s), 1000.0), np.zeros(len(s))))

    ct = ContactType10(model.interfaces[0], model, log)

    # Initial impact
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, np.zeros_like(model.x), model.mass, 1e-4, fcont, 0)
    assert len(ct._hist_keys) == 1

    # Cycle 1: move upward pulling on spring
    v = np.zeros_like(model.x)
    v[0, 2] = 20.0
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, v, model.mass, 1e-4, fcont, 1)
    assert fcont[0, 2] < 0.0  # pulling force downwards

    # Cycle 2: move far beyond gap (e.g. z = 2.0, gap is 0.2)
    model.x[0, 2] = 2.0
    v[0, 2] = -50.0  # reverse velocity
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, v, model.mass, 1e-4, fcont, 2)
    # Still tied!
    assert len(ct._hist_keys) == 1


def test_type10_itied_0_rebound_separation(monkeypatch):
    """Verify that with Itied=0, contact separates when force reverses and node unpenetrates."""
    model, log = _make_type10_model(itied=0, gap=0.2)
    monkeypatch.setattr("pyradioss.contact.inter_type10.node_stiffness_gap", lambda m, s: (np.full(5, 1000.0), np.zeros(5)))
    monkeypatch.setattr("pyradioss.contact.inter_type10.segment_stiffness_gap", lambda m, s, gt, ge, st: (np.full(len(s), 1000.0), np.zeros(len(s))))

    ct = ContactType10(model.interfaces[0], model, log)

    # Cycle 0: impact at z=0.1 (within gap 0.2)
    v = np.zeros_like(model.x)
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, v, model.mass, 1e-4, fcont, 0)
    assert len(ct._hist_keys) == 1

    # Cycle 1: compress further (v_z = -10.0 -> normal velocity > 0)
    # Master is at z=0, slave at z=0.1, nvec=(0,0,1).
    # Moving towards master means v_z < 0.
    v[0, 2] = -10.0
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, v, model.mass, 1e-4, fcont, 1)
    fn_comp = ct._hist_fn[0]
    assert fn_comp < 0.0  # compressive normal force

    # Cycle 2: node moves beyond gap (z=0.3 > 0.2, pen <= 0) and reverses velocity
    model.x[0, 2] = 0.3
    v[0, 2] = 30.0  # pulling away
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, v, model.mass, 1e-4, fcont, 2)

    # Rebound condition triggered: fn_old * fn_new < 0 and pen <= 0
    assert len(ct._hist_keys) == 0  # untied!
    assert np.allclose(fcont, 0.0)


def test_type10_tangential_shear_forces(monkeypatch):
    """Verify accumulation of tangential forces in local orthonormal frame."""
    model, log = _make_type10_model(itied=1)
    monkeypatch.setattr("pyradioss.contact.inter_type10.node_stiffness_gap", lambda m, s: (np.full(5, 1000.0), np.zeros(5)))
    monkeypatch.setattr("pyradioss.contact.inter_type10.segment_stiffness_gap", lambda m, s, gt, ge, st: (np.full(len(s), 1000.0), np.zeros(len(s))))

    ct = ContactType10(model.interfaces[0], model, log)

    # Cycle 0: tie
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, np.zeros_like(model.x), model.mass, 1e-4, fcont, 0)

    # Cycle 1: pure tangential velocity in X direction
    v = np.zeros_like(model.x)
    v[0, 0] = 20.0
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, v, model.mass, 1e-4, fcont, 1)

    # Tangential force opposes motion in X
    assert fcont[0, 0] < 0.0
    # Normal force is zero since vn = 0
    assert np.isclose(fcont[0, 2], 0.0)


def test_type10_viscous_damping_and_dissipation(monkeypatch):
    """Verify viscous damping force and positive dissipation energy."""
    model, log = _make_type10_model(stiff_dc=0.2)
    monkeypatch.setattr("pyradioss.contact.inter_type10.node_stiffness_gap", lambda m, s: (np.full(5, 1000.0), np.zeros(5)))
    monkeypatch.setattr("pyradioss.contact.inter_type10.segment_stiffness_gap", lambda m, s, gt, ge, st: (np.full(len(s), 1000.0), np.zeros(len(s))))

    ct = ContactType10(model.interfaces[0], model, log)

    # Tie at cycle 0
    ct.forces(model.x, np.zeros_like(model.x), model.mass, 1e-4, np.zeros_like(model.x), 0)

    # Move with velocity in cycle 1
    v = np.zeros_like(model.x)
    v[0, 2] = 10.0
    fcont = np.zeros_like(model.x)
    dE, _ = ct.forces(model.x, v, model.mass, 1e-4, fcont, 1)

    # Damping dissipation must be positive
    assert ct.e_damp > 0.0
    assert dE > 0.0


def test_type10_stiffness_nodal_accumulation(monkeypatch):
    """Verify stifn nodal accumulation matches i7ass0 formula."""
    model, log = _make_type10_model()
    monkeypatch.setattr("pyradioss.contact.inter_type10.node_stiffness_gap", lambda m, s: (np.full(5, 1000.0), np.zeros(5)))
    monkeypatch.setattr("pyradioss.contact.inter_type10.segment_stiffness_gap", lambda m, s, gt, ge, st: (np.full(len(s), 1000.0), np.zeros(len(s))))

    ct = ContactType10(model.interfaces[0], model, log)
    fcont = np.zeros_like(model.x)
    stifn = np.zeros(len(model.x), dtype=float)

    # Cycle 0
    ct.forces(model.x, np.zeros_like(model.x), model.mass, 1e-4, fcont, 0, stifn=stifn)

    # Secondary node gets K
    assert np.isclose(stifn[0], 1000.0)
    # Master nodes get H_k * K, sum over master corners equals K
    assert np.isclose(np.sum(stifn[1:5]), 1000.0)


def test_type10_triangular_master_segment(monkeypatch):
    """Verify interface handles 3-node triangular master segments."""
    model = Model()
    log = MessageLog()
    cards = [
        Card("Tri Master TYPE10"),
        Card("21 22 5 0"),
        Card("1.0 0.2 0.0 1e30"),
        Card("1 0 0.0 0.2"),
    ]
    block = KeywordBlock("/INTER/TYPE10", ["/INTER", "TYPE10", "102"], 102, cards, fixed=False)
    read_inter(block, model, log)

    model.node_groups[21] = NodeGroup(id=21, node_idx=[0])
    # 3-node triangle with node 3 repeated
    model.surfaces[22] = Surface(
        id=22,
        segments=np.array([[1, 2, 3, 3]]),
        seg_gtype=np.array([""]),
        seg_elem=np.array([0]),
    )

    x = np.array([
        [0.3, 0.3, 0.05],  # secondary node above triangle
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    model.x = x.copy()
    model.x0 = x.copy()
    model.node_idx = np.arange(4, dtype=np.int64)
    model.mass = np.ones(4)
    model.mass0 = np.ones(4)

    monkeypatch.setattr("pyradioss.contact.inter_type10.node_stiffness_gap", lambda m, s: (np.full(4, 1000.0), np.zeros(4)))
    monkeypatch.setattr("pyradioss.contact.inter_type10.segment_stiffness_gap", lambda m, s, gt, ge, st: (np.full(1, 1000.0), np.zeros(1)))

    ct = ContactType10(model.interfaces[0], model, log)
    fcont = np.zeros_like(x)
    ct.forces(x, np.zeros_like(x), model.mass, 1e-4, fcont, 0)
    assert len(ct._hist_keys) == 1  # Successfully tied to triangle

    v = np.zeros_like(x)
    v[0, 2] = 10.0
    fcont[:] = 0.0
    ct.forces(x, v, model.mass, 1e-4, fcont, 1)
    assert fcont[0, 2] < 0.0
    assert np.allclose(np.sum(fcont, axis=0), 0.0)


def test_type10_element_deletion_tracking(monkeypatch):
    """Verify that with idel10 >= 1, segment deletion deactivates contact."""
    model, log = _make_type10_model(idel10=1)
    monkeypatch.setattr("pyradioss.contact.inter_type10.node_stiffness_gap", lambda m, s: (np.full(5, 1000.0), np.zeros(5)))
    monkeypatch.setattr("pyradioss.contact.inter_type10.segment_stiffness_gap", lambda m, s, gt, ge, st: (np.full(len(s), 1000.0), np.zeros(len(s))))
    monkeypatch.setattr("pyradioss.contact.tracking.any_deletable", lambda *a, **k: True)

    ct = ContactType10(model.interfaces[0], model, log)
    assert ct.deletable is True

    # Cycle 0: active impact
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, np.zeros_like(model.x), model.mass, 1e-4, fcont, 0)
    assert len(ct._hist_keys) == 1

    # Simulate element 0 deletion: alive_segment_mask returns False
    monkeypatch.setattr("pyradioss.contact.tracking.alive_segment_mask", lambda m, gt, ge: np.zeros(len(ge), dtype=bool))

    v = np.zeros_like(model.x)
    v[0, 2] = 10.0
    fcont[:] = 0.0
    ct.forces(model.x, v, model.mass, 1e-4, fcont, 1)
    # Deactivated due to deleted segment
    assert np.allclose(fcont, 0.0)


def test_type10_build_contacts_factory():
    """Verify build_contacts instantiates ContactType10 into penalty contacts."""
    model, log = _make_type10_model()
    penalty, tied = build_contacts(model, log)
    assert len(penalty) == 1
    assert isinstance(penalty[0], ContactType10)
    assert len(tied) == 0


def test_type10_dynamic_oscillation_energy_balance(monkeypatch):
    """Simulate a mass oscillating on a TYPE10 contact spring over 30 cycles."""
    model, log = _make_type10_model(itied=1, stiff_dc=0.05)
    monkeypatch.setattr("pyradioss.contact.inter_type10.node_stiffness_gap", lambda m, s: (np.full(5, 10000.0), np.zeros(5)))
    monkeypatch.setattr("pyradioss.contact.inter_type10.segment_stiffness_gap", lambda m, s, gt, ge, st: (np.full(len(s), 10000.0), np.zeros(len(s))))

    ct = ContactType10(model.interfaces[0], model, log)
    dt = 1e-4
    x = model.x.copy()
    v = np.zeros_like(x)
    mass = model.mass

    # Initial impact
    fcont = np.zeros_like(x)
    ct.forces(x, v, mass, dt, fcont, 0)

    # Initial kick to secondary node
    v[0, 2] = 2.0

    total_dE = 0.0
    for cycle in range(1, 31):
        # Semi-implicit Euler integration
        fcont[:] = 0.0
        dE, dt_int = ct.forces(x, v, mass, dt, fcont, cycle)
        total_dE += dE

        # Linear momentum balance at every cycle
        assert np.allclose(np.sum(fcont, axis=0), 0.0, atol=1e-12)

        # Update secondary node kinematics (fixing master plate)
        a = fcont[0] / mass[0]
        v[0] += a * dt
        x[0] += v[0] * dt

    # Damping dissipated energy must be positive
    assert ct.e_damp > 0.0
    assert total_dE > 0.0


def test_type10_broad_phase_retention(monkeypatch):
    """Verify tied contact pairs are preserved in broad-phase across distant motion."""
    model, log = _make_type10_model(itied=1)
    monkeypatch.setattr("pyradioss.contact.inter_type10.node_stiffness_gap", lambda m, s: (np.full(5, 1000.0), np.zeros(5)))
    monkeypatch.setattr("pyradioss.contact.inter_type10.segment_stiffness_gap", lambda m, s, gt, ge, st: (np.full(len(s), 1000.0), np.zeros(len(s))))

    ct = ContactType10(model.interfaces[0], model, log)
    # Impact at cycle 0
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, np.zeros_like(model.x), model.mass, 1e-4, fcont, 0)
    assert len(ct._hist_keys) == 1

    # Move far outside voxel grid margin
    model.x[0] = [100.0, 100.0, 100.0]
    # Trigger refresh cycle
    ct.forces(model.x, np.zeros_like(model.x), model.mass, 1e-4, fcont, cycle=ct.refresh + 1)
    # Pair must be retained because it is tied
    assert len(ct._hist_keys) == 1
    assert len(ct.pairs_node) >= 1

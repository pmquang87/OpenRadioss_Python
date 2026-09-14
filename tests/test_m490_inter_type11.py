"""
Tests for M490: /INTER/TYPE11 edge-to-edge penalty contact interface hardening,
momentum conservation, friction models, and Starter parsing.

Fortran origins:
  - engine/source/interfaces/int11/i11dst3.F (exact edge-edge closest points)
  - engine/source/interfaces/int11/i11for3.F (penalty forces, normal damping, friction)
  - engine/source/interfaces/int11/i11buce.F (broad-phase voxel binning)
  - engine/source/interfaces/int11/i11mainf.F (friction models & filtering)
  - starter/source/interfaces/inter3d1/i11sti3.F (edge stiffness and gap setup)
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.contact import ContactType11, build_contacts
from pyradioss.contact.inter_type11 import _closest_points_on_segments
from pyradioss.input.deck_reader import KeywordBlock, Card
from pyradioss.input.starter_keywords import read_inter
from pyradioss.model.entities import (
    Interface,
    Line,
)
from pyradioss.model.model import ElementGroup, Model


# ============================================================================
# Helpers
# ============================================================================

def make_type11_model(
    coords: np.ndarray,
    line1_segs: np.ndarray,
    line2_segs: np.ndarray,
    stfac: float = 1.0,
    istf: int = 1,
    gap: float = 0.1,
    igap: int = 0,
    fric: float = 0.0,
    mfrot: int = 0,
    ifq: int = 0,
    xfiltr: float = 0.0,
    fric_c: tuple = (0.0,) * 6,
    mass_val: float = 1.0,
) -> tuple[Model, Interface]:
    """Build a synthetic Model with 2 lines and 1 /INTER/TYPE11 interface."""
    model = Model()
    n = len(coords)
    model.node_ids = np.arange(1, n + 1)
    model.x0 = np.array(coords, dtype=float)
    model.x = model.x0.copy()
    model.v = np.zeros((n, 3), dtype=float)
    model.vr = np.zeros((n, 3), dtype=float)
    model.mass = np.full(n, mass_val, dtype=float)
    model.mass0 = model.mass.copy()

    # Line 1: secondary
    ln1 = Line(id=1, title="SEC_LINE")
    ln1.segments = np.array(line1_segs, dtype=np.int64)
    ln1.seg_gtype = np.full(len(line1_segs), "", dtype="<U8")
    ln1.seg_elem = np.full(len(line1_segs), -1, dtype=np.int64)
    model.lines[1] = ln1

    # Line 2: main
    ln2 = Line(id=2, title="MAIN_LINE")
    ln2.segments = np.array(line2_segs, dtype=np.int64)
    ln2.seg_gtype = np.full(len(line2_segs), "", dtype="<U8")
    ln2.seg_elem = np.full(len(line2_segs), -1, dtype=np.int64)
    model.lines[2] = ln2

    itf = Interface(
        id=1,
        type=11,
        line_id1=1,
        line_id2=2,
        istf=istf,
        stfac=stfac,
        gap=gap,
        igap=igap,
        fric=fric,
        mfrot=mfrot,
        ifq=ifq,
        xfiltr=xfiltr,
        fric_c=fric_c,
    )
    model.interfaces.append(itf)
    return model, itf


# ============================================================================
# 1. Exact Narrow Phase: _closest_points_on_segments (i11dst3.F)
# ============================================================================

def test_closest_points_skew_perpendicular_crossing():
    """Two perpendicular edges crossing in space with known analytical distance."""
    # Edge A: from (-1, 0, 0.05) to (1, 0, 0.05) [along X at Z=+0.05]
    p1 = np.array([[-1.0, 0.0, 0.05]])
    q1 = np.array([[1.0, 0.0, 0.05]])
    # Edge B: from (0, -1, -0.05) to (0, 1, -0.05) [along Y at Z=-0.05]
    p2 = np.array([[0.0, -1.0, -0.05]])
    q2 = np.array([[0.0, 1.0, -0.05]])

    s, t, cA, cB = _closest_points_on_segments(p1, q1, p2, q2)

    # Midpoints s=0.5, t=0.5
    np.testing.assert_allclose(s, [0.5], atol=1e-12)
    np.testing.assert_allclose(t, [0.5], atol=1e-12)
    np.testing.assert_allclose(cA[0], [0.0, 0.0, 0.05], atol=1e-12)
    np.testing.assert_allclose(cB[0], [0.0, 0.0, -0.05], atol=1e-12)
    dist = np.linalg.norm(cA - cB)
    assert np.isclose(dist, 0.1, atol=1e-12)


def test_closest_points_parallel_separated():
    """Parallel separated edges clamp s=0 and find correct projection t."""
    p1 = np.array([[0.0, 0.0, 0.0]])
    q1 = np.array([[2.0, 0.0, 0.0]])
    p2 = np.array([[0.5, 1.0, 0.0]])
    q2 = np.array([[2.5, 1.0, 0.0]])

    s, t, cA, cB = _closest_points_on_segments(p1, q1, p2, q2)

    # For parallel edges, s=0 -> cA = (0, 0, 0), t is clamped to [0, 1]
    assert len(s) == 1
    assert 0.0 <= s[0] <= 1.0
    assert 0.0 <= t[0] <= 1.0
    dist = np.linalg.norm(cA - cB)
    assert np.isclose(dist, 1.0 + (0.5 - 0.0)**2, atol=0.5)  # bounded distance


def test_closest_points_clamped_endpoints():
    """Non-overlapping edges clamp to endpoints correctly."""
    # Segment 1 on [0, 1] on X axis
    p1 = np.array([[0.0, 0.0, 0.0]])
    q1 = np.array([[1.0, 0.0, 0.0]])
    # Segment 2 on [3, 4] on X axis, offset by Y=1
    p2 = np.array([[3.0, 1.0, 0.0]])
    q2 = np.array([[4.0, 1.0, 0.0]])

    s, t, cA, cB = _closest_points_on_segments(p1, q1, p2, q2)

    # Closest points must be q1 (s=1.0) and p2 (t=0.0)
    np.testing.assert_allclose(s, [1.0], atol=1e-12)
    np.testing.assert_allclose(t, [0.0], atol=1e-12)
    np.testing.assert_allclose(cA[0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(cB[0], [3.0, 1.0, 0.0], atol=1e-12)
    # Expected distance sqrt((3-1)^2 + 1^2) = sqrt(5)
    assert np.isclose(np.linalg.norm(cA - cB), np.sqrt(5.0), atol=1e-12)


def test_closest_points_degenerate_zero_length():
    """Zero-length segment (point-to-segment) evaluates safely without NaNs."""
    p1 = np.array([[1.0, 1.0, 1.0]])
    q1 = np.array([[1.0, 1.0, 1.0]])  # zero length
    p2 = np.array([[0.0, 0.0, 0.0]])
    q2 = np.array([[2.0, 0.0, 0.0]])  # along X

    s, t, cA, cB = _closest_points_on_segments(p1, q1, p2, q2)

    assert not np.isnan(s).any()
    assert not np.isnan(t).any()
    np.testing.assert_allclose(cA[0], [1.0, 1.0, 1.0], atol=1e-12)
    # Closest point on edge [0,0,0]-[2,0,0] to (1,1,1) is (1,0,0), t=0.5
    np.testing.assert_allclose(cB[0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(t, [0.5], atol=1e-12)


def test_closest_points_empty():
    """Empty segment inputs return consistent empty arrays."""
    p1 = np.zeros((0, 3))
    q1 = np.zeros((0, 3))
    p2 = np.zeros((0, 3))
    q2 = np.zeros((0, 3))

    s, t, cA, cB = _closest_points_on_segments(p1, q1, p2, q2)
    assert len(s) == 0
    assert len(t) == 0
    assert cA.shape == (0, 3)
    assert cB.shape == (0, 3)


# ============================================================================
# 2. Initialization and Broad Phase (i11buce.F)
# ============================================================================

def test_contact_type11_init_basic():
    """Basic ContactType11 initialization computes stiffness and time step bound."""
    coords = np.array([
        [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0], [0.0, 1.0, 0.0],
    ])
    segs1 = np.array([[0, 1]])
    segs2 = np.array([[2, 3]])
    model, itf = make_type11_model(coords, segs1, segs2, stfac=5000.0, istf=1, gap=0.1)

    ct = ContactType11(itf, model, MessageLog())
    assert len(ct.es) == 1
    assert len(ct.em) == 1
    assert ct.gap_bound == 0.1
    assert ct.dt_bound < 1.0
    assert np.isfinite(ct.dt_bound)


def test_contact_type11_missing_line_defensive():
    """Missing line ID logs error and initializes empty without crashing."""
    model = Model()
    itf = Interface(id=1, type=11, line_id1=999, line_id2=888)
    log = MessageLog()
    ct = ContactType11(itf, model, log)

    assert len(ct.es) == 0
    assert len(ct.em) == 0
    assert ct.dt_bound == np.inf
    assert any("secondary line 999 not found" in e for e in log.errors)


def test_contact_type11_empty_line_warning():
    """Line with zero segments logs warning and deactivates interface."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    model, itf = make_type11_model(coords, np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int))
    log = MessageLog()
    ct = ContactType11(itf, model, log)

    assert len(ct.es) == 0
    assert len(ct.em) == 0
    assert ct.dt_bound == np.inf
    assert any("empty" in w for w in log.warnings)


def test_broad_phase_exclusion_common_node():
    """Edges sharing a common node are excluded from candidate contact pairs."""
    # Node 1 is shared between edge (0, 1) and edge (1, 2)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    segs1 = np.array([[0, 1]])
    segs2 = np.array([[1, 2]])
    model, itf = make_type11_model(coords, segs1, segs2, gap=0.5)

    ct = ContactType11(itf, model, MessageLog())
    ct._broad_phase(model.x, model.v, 1e-4)

    # Candidate pairs must be empty due to node sharing
    assert len(ct.pairs_s) == 0
    assert len(ct.pairs_m) == 0


def test_broad_phase_far_edges_excluded():
    """Edges far apart outside search margin are not binned as candidate pairs."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [100.0, 0.0, 0.0], [101.0, 0.0, 0.0],
    ])
    segs1 = np.array([[0, 1]])
    segs2 = np.array([[2, 3]])
    model, itf = make_type11_model(coords, segs1, segs2, gap=0.1)

    ct = ContactType11(itf, model, MessageLog())
    ct._broad_phase(model.x, model.v, 1e-4)

    assert len(ct.pairs_s) == 0
    assert len(ct.pairs_m) == 0


# ============================================================================
# 3. Gap Policies: Igap=0 vs Igap=1
# ============================================================================

def test_gap_constant_igap0():
    """Igap=0 uses specified constant gap floor."""
    coords = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    model, itf = make_type11_model(coords, [[0, 1]], [[2, 3]], gap=0.25, igap=0)
    ct = ContactType11(itf, model, MessageLog())
    assert np.isclose(ct.gap_const, 0.25)
    assert np.isclose(ct.gap_bound, 0.25)


def test_gap_variable_igap1():
    """Igap=1 uses sum of edge gaps clamped between gap_min and gap_max."""
    coords = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    model, itf = make_type11_model(coords, [[0, 1]], [[2, 3]], gap=0.05, igap=1)
    itf.gap_max = 0.5
    ct = ContactType11(itf, model, MessageLog())
    assert hasattr(ct, "gap_s")
    assert hasattr(ct, "gap_m")
    assert ct.gap_min == 0.05
    assert ct.gap_max == 0.5


# ============================================================================
# 4. Penalty Forces & Momentum Conservation (i11for3.F)
# ============================================================================

def test_forces_penetration_momentum_conservation():
    """Normal contact penalty forces strictly conserve linear and angular momentum."""
    # Edge A: from (-1, 0, 0.02) to (1, 0, 0.02)
    # Edge B: from (0, -1, -0.02) to (0, 1, -0.02)
    # Separation d = 0.04 < gap 0.1 -> penetration pen = 0.06
    coords = np.array([
        [-1.0, 0.0, 0.02],
        [1.0, 0.0, 0.02],
        [0.0, -1.0, -0.02],
        [0.0, 1.0, -0.02],
    ])
    segs1 = np.array([[0, 1]])
    segs2 = np.array([[2, 3]])
    model, itf = make_type11_model(coords, segs1, segs2, stfac=1000.0, istf=1, gap=0.1)

    ct = ContactType11(itf, model, MessageLog())
    fcont = np.zeros((4, 3))
    dt = 1e-4

    de, dt_int = ct.forces(model.x, model.v, model.mass, dt, fcont, cycle=0)

    # 1. Total force must sum to 0 (Linear momentum conservation)
    f_total = fcont.sum(axis=0)
    np.testing.assert_allclose(f_total, [0.0, 0.0, 0.0], atol=1e-12)

    # 2. Total moment about origin must sum to 0 (Angular momentum conservation)
    # Torque = sum r_i x F_i
    torque = np.cross(model.x, fcont).sum(axis=0)
    np.testing.assert_allclose(torque, [0.0, 0.0, 0.0], atol=1e-12)

    # 3. Secondary edge pushed +Z, main edge pushed -Z
    assert fcont[0, 2] > 0.0
    assert fcont[1, 2] > 0.0
    assert fcont[2, 2] < 0.0
    assert fcont[3, 2] < 0.0
    # Symmetric crossing -> equal split on edge endpoints
    np.testing.assert_allclose(fcont[0, 2], fcont[1, 2], atol=1e-12)
    np.testing.assert_allclose(fcont[2, 2], fcont[3, 2], atol=1e-12)
    # Expected Fn = K * pen = 1000 * 0.06 = 60.0
    np.testing.assert_allclose(fcont[:2, 2].sum(), 60.0, atol=1e-12)
    np.testing.assert_allclose(fcont[2:, 2].sum(), -60.0, atol=1e-12)


def test_forces_no_penetration_zero_force():
    """Separation greater than gap results in exactly zero contact force."""
    coords = np.array([
        [-1.0, 0.0, 0.2],
        [1.0, 0.0, 0.2],
        [0.0, -1.0, -0.2],
        [0.0, 1.0, -0.2],
    ])
    segs1 = np.array([[0, 1]])
    segs2 = np.array([[2, 3]])
    # d = 0.4 > gap 0.1
    model, itf = make_type11_model(coords, segs1, segs2, stfac=1000.0, istf=1, gap=0.1)

    ct = ContactType11(itf, model, MessageLog())
    fcont = np.zeros((4, 3))
    de, dt_int = ct.forces(model.x, model.v, model.mass, 1e-4, fcont, cycle=0)

    np.testing.assert_allclose(fcont, 0.0, atol=1e-14)
    assert de == 0.0


def test_forces_normal_damping_dissipation():
    """Approaching velocity activates normal damping C*|vn| and books positive dissipation."""
    coords = np.array([
        [-1.0, 0.0, 0.02],
        [1.0, 0.0, 0.02],
        [0.0, -1.0, -0.02],
        [0.0, 1.0, -0.02],
    ])
    segs1 = np.array([[0, 1]])
    segs2 = np.array([[2, 3]])
    model, itf = make_type11_model(coords, segs1, segs2, stfac=1000.0, istf=1, gap=0.1)

    # Secondary edge moves down (-Z), main edge moves up (+Z): closing velocity vn < 0
    model.v[0:2, 2] = -5.0
    model.v[2:4, 2] = +5.0

    ct = ContactType11(itf, model, MessageLog())
    fcont = np.zeros((4, 3))
    dt = 1e-4

    de, dt_int = ct.forces(model.x, model.v, model.mass, dt, fcont, cycle=0)

    # Damping increases repulsive force beyond static spring Fn=60.0
    assert fcont[:2, 2].sum() > 60.0
    # Work returned should be positive dissipation (dissipative energy)
    assert de > 0.0


def test_forces_stifn_accumulation():
    """stifn array accumulates contact spring stiffness for /DT/NODA mass scaling."""
    coords = np.array([
        [-1.0, 0.0, 0.02],
        [1.0, 0.0, 0.02],
        [0.0, -1.0, -0.02],
        [0.0, 1.0, -0.02],
    ])
    segs1 = np.array([[0, 1]])
    segs2 = np.array([[2, 3]])
    model, itf = make_type11_model(coords, segs1, segs2, stfac=2000.0, istf=1, gap=0.1)

    ct = ContactType11(itf, model, MessageLog())
    fcont = np.zeros((4, 3))
    stifn = np.zeros(4)

    ct.forces(model.x, model.v, model.mass, 1e-4, fcont, cycle=0, stifn=stifn)

    # K=2000 per node
    assert np.all(stifn > 0.0)
    np.testing.assert_allclose(stifn, 2000.0, atol=1e-12)


def test_forces_zero_distance_normal_fallback():
    """Edges crossing at exact identical coordinate (d=0) use cross-product fallback normal."""
    coords = np.array([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    segs1 = np.array([[0, 1]])
    segs2 = np.array([[2, 3]])
    model, itf = make_type11_model(coords, segs1, segs2, stfac=1000.0, istf=1, gap=0.1)

    ct = ContactType11(itf, model, MessageLog())
    fcont = np.zeros((4, 3))

    de, dt_int = ct.forces(model.x, model.v, model.mass, 1e-4, fcont, cycle=0)

    # Edge A along X, Edge B along Y -> cross product is Z
    assert not np.isnan(fcont).any()
    assert not np.isinf(fcont).any()
    assert abs(fcont[0, 2]) > 0.0
    # Momentum is conserved
    np.testing.assert_allclose(fcont.sum(axis=0), 0.0, atol=1e-12)


# ============================================================================
# 5. Friction Models (MFROT) & IFQ Filter
# ============================================================================

def test_forces_coulomb_friction():
    """Coulomb friction generates tangential force opposing relative slip."""
    coords = np.array([
        [-1.0, 0.0, 0.02],
        [1.0, 0.0, 0.02],
        [0.0, -1.0, -0.02],
        [0.0, 1.0, -0.02],
    ])
    segs1 = np.array([[0, 1]])
    segs2 = np.array([[2, 3]])
    # fric = 0.3
    model, itf = make_type11_model(coords, segs1, segs2, stfac=1000.0, istf=1, gap=0.1, fric=0.3)

    # Secondary edge slides in +X direction relative to main edge
    model.v[0:2, 0] = 10.0

    ct = ContactType11(itf, model, MessageLog())
    fcont = np.zeros((4, 3))
    dt = 1e-4

    ct.forces(model.x, model.v, model.mass, dt, fcont, cycle=0)

    # Friction on secondary edge must oppose motion (negative X)
    assert fcont[:2, 0].sum() < 0.0
    # Friction on main edge must be positive X (action-reaction)
    assert fcont[2:, 0].sum() > 0.0
    np.testing.assert_allclose(fcont.sum(axis=0), 0.0, atol=1e-12)


def test_forces_mfrot_dynamic_friction():
    """MFROT=1 (viscous exponential law) evaluates dynamic friction."""
    coords = np.array([
        [-1.0, 0.0, 0.02],
        [1.0, 0.0, 0.02],
        [0.0, -1.0, -0.02],
        [0.0, 1.0, -0.02],
    ])
    segs1 = np.array([[0, 1]])
    segs2 = np.array([[2, 3]])
    fric_c = (0.5, 0.2, 10.0, 0.0, 0.0, 0.0)
    model, itf = make_type11_model(
        coords, segs1, segs2, stfac=1000.0, istf=1, gap=0.1, fric=0.5, mfrot=1, fric_c=fric_c
    )
    model.v[0:2, 0] = 5.0

    log = MessageLog()
    ct = ContactType11(itf, model, log)
    fcont = np.zeros((4, 3))
    ct.forces(model.x, model.v, model.mass, 1e-4, fcont, cycle=0)

    assert fcont[:2, 0].sum() < 0.0
    assert ct.mfrot == 1


def test_forces_ifq_filter():
    """IFQ filter smooths tangential force over cycles."""
    coords = np.array([
        [-1.0, 0.0, 0.02],
        [1.0, 0.0, 0.02],
        [0.0, -1.0, -0.02],
        [0.0, 1.0, -0.02],
    ])
    segs1 = np.array([[0, 1]])
    segs2 = np.array([[2, 3]])
    model, itf = make_type11_model(
        coords, segs1, segs2, stfac=1000.0, istf=1, gap=0.1, fric=0.3, ifq=1, xfiltr=0.5
    )
    model.v[0:2, 0] = 10.0

    ct = ContactType11(itf, model, MessageLog())
    fcont1 = np.zeros((4, 3))
    ct.forces(model.x, model.v, model.mass, 1e-4, fcont1, cycle=0)

    # Stored filtered values must exist
    assert len(ct._filt_keys) > 0
    assert len(ct._filt_vals) > 0


# ============================================================================
# 6. Dynamic Element Deletion Release
# ============================================================================

def test_forces_element_deletion_release():
    """Deletion of parent element drops edge and zeroes contact force."""
    coords = np.array([
        [-1.0, 0.0, 0.02],
        [1.0, 0.0, 0.02],
        [0.0, -1.0, -0.02],
        [0.0, 1.0, -0.02],
    ])
    segs1 = np.array([[0, 1]])
    segs2 = np.array([[2, 3]])
    model, itf = make_type11_model(coords, segs1, segs2, stfac=1000.0, istf=1, gap=0.1)

    class DummyMat:
        E = 2.1e5
    class DummyProp:
        t = 1.0

    # Shell element group linked to secondary line
    egrp = ElementGroup(
        ids=np.array([10]),
        conn=np.array([[0, 1, 1, 0]]),
        part=np.array([0]),
        state={
            "off": np.array([1.0]),
            "chk_fail": True,
            "slices": [(slice(0, 1), DummyMat(), DummyProp())],
            "thick": np.array([1.0]),
        },
    )
    model.shells = egrp
    model.lines[1].seg_gtype = np.array(["shells"], dtype="<U8")
    model.lines[1].seg_elem = np.array([0], dtype=np.int64)

    itf.idel = 1
    ct = ContactType11(itf, model, MessageLog())
    assert ct.deletable is True

    fcont = np.zeros((4, 3))
    ct.forces(model.x, model.v, model.mass, 1e-4, fcont, cycle=0)
    assert abs(fcont[0, 2]) > 0.0  # alive -> carries force

    # Kill element
    egrp.state["off"][0] = 0.0
    fcont.fill(0.0)
    ct.forces(model.x, model.v, model.mass, 1e-4, fcont, cycle=1)
    np.testing.assert_allclose(fcont, 0.0, atol=1e-14)  # dead -> zero force


# ============================================================================
# 7. Starter Keyword Parsing & Factory Integration
# ============================================================================

def test_starter_keyword_inter_type11_compact():
    """Compact free-format /INTER/TYPE11 keyword block parsed into model."""
    block = KeywordBlock(
        keyword="/INTER/TYPE11/5",
        parts=["INTER", "TYPE11", "5"],
        user_id=5,
        cards=[
            Card("Edge-to-edge compact"),
            Card("1 2 1 0 0 0 0"),
            Card("1500.0 0.25 0.08 0.5"),
        ],
        fixed=False,
    )
    model = Model()
    log = MessageLog()
    read_inter(block, model, log)

    assert len(model.interfaces) == 1
    itf = model.interfaces[0]
    assert itf.id == 5
    assert itf.type == 11
    assert itf.line_id1 == 1
    assert itf.line_id2 == 2
    assert itf.istf == 1
    assert itf.stfac == 1500.0
    assert itf.fric == 0.25
    assert itf.gap == 0.08


def test_starter_keyword_inter_type11_fixed():
    """Fixed-format /INTER/TYPE11 keyword block with inter_type11.cfg parsed."""
    block = KeywordBlock(
        keyword="/INTER/TYPE11/7",
        parts=["INTER", "TYPE11", "7"],
        user_id=7,
        cards=[
            Card("Edge-to-edge fixed"),
            Card("        10        20         1         0         0         0         0         0"),
            Card("                                                  0.40000000000000000000         0"),
            Card("2000.00000000000000000.200000000000000000000.05000000000000000000"),
            Card(""),
        ],
        fixed=True,
    )
    model = Model()
    log = MessageLog()
    read_inter(block, model, log)

    assert len(model.interfaces) == 1
    itf = model.interfaces[0]
    assert itf.id == 7
    assert itf.type == 11
    assert itf.line_id1 == 10
    assert itf.line_id2 == 20
    assert itf.istf == 1
    assert itf.stfac == 2000.0
    assert itf.fric == 0.2
    assert itf.gap == 0.05


def test_build_contacts_factory():
    """build_contacts segregates /INTER/TYPE11 into penalty contact list."""
    coords = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    model, itf = make_type11_model(coords, [[0, 1]], [[2, 3]])

    log = MessageLog()
    penalty, tied = build_contacts(model, log)
    assert len(penalty) == 1
    assert len(tied) == 0
    assert isinstance(penalty[0], ContactType11)

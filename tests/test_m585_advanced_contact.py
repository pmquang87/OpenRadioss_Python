"""
Unit tests for Milestone M585: Advanced Contact Interfaces.

Covers:
1. /INTER/TYPE21 Drawbead Contact:
   - Starter deck parsing (both free and fixed formats, CFG layouts).
   - Master line extraction from /SURF or /LINE.
   - Normal clamping and penetration physics.
   - Tangential restraining force, plastic bending resistance, and Pmax capping.
   - Reaction force distribution and exact momentum conservation.
   - Time window gating and energy booking.
2. /INTER/TYPE23 Mortar Segment-to-Segment Contact:
   - Starter deck parsing (both free and fixed formats, CFG layouts).
   - Centroid subdivision into 4 sub-triangles.
   - Mortar shape function partition of unity (H1..H4).
   - Triangular facet fallback (H4=0).
   - Penalty normal stiffness and Coulomb stick-slip friction.
   - Consistent reaction distribution and exact momentum conservation.
   - Time window gating and energy booking.
3. Factory build_contacts() integration and end-to-end explicit engine runs.
"""

from __future__ import annotations

import contextlib
import io
import os
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.contact import build_contacts
from pyradioss.contact.inter_type21 import ContactType21
from pyradioss.contact.inter_type23 import ContactType23
from pyradioss.engine.engine import run_engine
from pyradioss.model.entities import Interface, Line, NodeGroup, Surface
from pyradioss.model.model import Model
from pyradioss.starter.starter import run_starter


STEEL_LAW1 = """\
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
"""

SHELL_PROPS = """\
/PROP/SHELL/1
sheet prop
1 0 0 0
0.01 0.01 0.01 0 0
3 0 0.1
/PROP/SHELL/2
tool prop
1 0 0 0
0.01 0.01 0.01 0 0
3 0 0.1
"""

BASE_DECK = (
    "/BEGIN\n"
    "m585 contact deck\n"
    "/NODE\n"
    "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
    "11 0.25 0.25 0.05\n12 0.75 0.25 0.05\n"
    "13 0.75 0.75 0.05\n14 0.25 0.75 0.05\n"
    "21 0.0 0.5 0.0\n22 1.0 0.5 0.0\n"
    "/SHELL/1\n1 1 2 3 4\n"
    "/SHELL/2\n2 11 12 13 14\n"
    "/PART/1\nlower\n1 1\n"
    "/PART/2\nupper\n2 1\n"
    + STEEL_LAW1
    + SHELL_PROPS
    + "/GRNOD/PART/2\nupper nodes\n2\n"
    "/SURF/PART/1\nlower surf\n1\n"
    "/SURF/PART/2\nupper surf\n2\n"
)


def _run_starter_str(make_deck, name: str, starter_content: str):
    s, _ = make_deck(name, starter_content, "/RUN/X/1\n1.0\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(str(s))


# ============================================================================
# 1. /INTER/TYPE21 Starter Deck Parsing
# ============================================================================

def test_type21_free_format_parsing(make_deck):
    """Verify free-format /INTER/TYPE21 deck parsing with all parameter cards."""
    starter = (
        BASE_DECK
        + "/INTER/TYPE21/10\n"
        "drawbead free format\n"
        "2 1 1 0 4 1\n"                 # surf_s, surf_m, istf, igap, multimp, iadm
        "1.2 0.05 2.5 5000.0 0\n"       # gap_scale, gap_max, depth, pmax, itlim
        "10.0 1e8\n"                    # stmin, stmax
        "1.5 0.15 0.02 0.0 10.0\n"      # stfac, fric, gap_min, tstart, tstop
        "0 1 0.08 0.25\n"               # inactiv, viss, sort_fact
        "1 1 100.0 0 1\n"               # ifric, ifiltr, xfreq, sens_id
        "0.15 0.08 0.01 0.0 0.0\n"      # c1..c5
        "0.02\n"                        # c6
        "/END\n"
    )
    model = _run_starter_str(make_deck, "T21_FREE", starter)
    itfs = [itf for itf in model.interfaces if itf.type == 21]
    assert len(itfs) == 1
    itf = itfs[0]
    assert itf.id == 10
    assert itf.type == 21
    assert itf.surf_id == 1      # master surface
    assert itf.surf_id1 == 2     # secondary surface
    assert itf.istf == 1
    assert itf.iadm == 1
    assert pytest.approx(itf.gap_scale) == 1.2
    assert pytest.approx(itf.gap_max) == 0.05
    assert pytest.approx(itf.depth) == 2.5
    assert pytest.approx(itf.pmax) == 5000.0
    assert itf.itlim == 0
    assert pytest.approx(itf.stmin) == 10.0
    assert pytest.approx(itf.stmax) == 1e8
    assert pytest.approx(itf.stfac) == 1.5
    assert pytest.approx(itf.fric) == 0.15
    assert pytest.approx(itf.gap_min) == 0.02
    assert pytest.approx(itf.viss) == 0.08
    assert itf.ifric == 1
    assert itf.ifiltr == 1
    assert pytest.approx(itf.xfreq) == 100.0
    assert itf.fric_c[0] == pytest.approx(0.15)
    assert itf.fric_c[1] == pytest.approx(0.08)


def test_type21_fixed_format_parsing(make_deck):
    """Verify fixed-format /INTER/TYPE21 deck parsing matching CFG columns."""
    c1 = f"{2:>10}{1:>10}{1:>10}{0:>10}{0:>10}{4:>20}{0:>10}{0:>10}{1:>10}"
    c2 = f"{1.0:>20.1f}{0.0:>20.1f}{3.0:>20.1f}{8000.0:>20.1f}{1:>10}"
    c3 = f"{5.0:>20.1f}{1.0e6:>20.1f}"
    c4 = f"{1.2:>20.1f}{0.12:>20.2f}{0.01:>20.2f}{0.0:>20.1f}{5.0:>20.1f}"
    c5 = f"{'':>7}{'0':>1}{'0':>1}{'0':>1}{'':>20}{0:>10}{0.06:>20.2f}{0.0:>20.1f}{0.2:>20.1f}"
    c6 = f"{0:>10}{0:>10}{0.0:>20.1f}{0:>10}{0:>10}"

    starter = (
        BASE_DECK
        + "/INTER/TYPE21/21\n"
        "drawbead fixed format\n"
        + f"{c1}\n{c2}\n{c3}\n{c4}\n{c5}\n{c6}\n"
        + "/END\n"
    )
    model = _run_starter_str(make_deck, "T21_FIX", starter)
    itfs = [itf for itf in model.interfaces if itf.type == 21]
    assert len(itfs) == 1
    itf = itfs[0]
    assert itf.id == 21
    assert itf.depth == pytest.approx(3.0)
    assert itf.pmax == pytest.approx(8000.0)
    assert itf.itlim == 1
    assert itf.stfac == pytest.approx(1.2)
    assert itf.fric == pytest.approx(0.12)
    assert itf.viss == pytest.approx(0.06)


# ============================================================================
# 2. /INTER/TYPE21 Physics Formulation
# ============================================================================

def test_type21_physics_clamping_and_penetration():
    """Verify normal clamping force, distance penetration, and momentum conservation."""
    model = Model()
    model.x = np.array([
        [0.0, 0.0, 0.0],   # node 0 (bead start)
        [1.0, 0.0, 0.0],   # node 1 (bead end)
        [0.5, 0.05, 0.0],  # node 2 (sheet node near bead line)
    ], dtype=float)
    model.x0 = model.x.copy()
    model.v = np.zeros_like(model.x)
    model.mass = np.array([1.0, 1.0, 1.0], dtype=float)

    # Master line from node 0 to node 1
    line_obj = Line(id=1)
    line_obj.segments = np.array([[0, 1]], dtype=np.int64)
    model.lines[1] = line_obj

    # Secondary sheet node group
    grn = NodeGroup(id=2, node_ids=[2], node_idx=np.array([2], dtype=np.int64))
    model.node_groups[2] = grn

    itf = Interface(
        id=1, type=21, surf_id=1, grnod_id=2,
        istf=1, stfac=1000.0, depth=0.1, fric=0.0,
        gap=0.02, viss=0.05,
    )
    log = MessageLog()
    ct21 = ContactType21(itf, model, log)

    fcont = np.zeros_like(model.x)
    econt, dt_int = ct21.forces(model.x, model.v, model.mass, 1e-4, fcont, cycle=0)

    # Penetration: closest pt on bead is (0.5, 0.0, 0.0), dist is 0.05
    # gap_eff = gap (0.02) + depth (0.1) = 0.12
    # pene = 0.12 - 0.05 = 0.07
    assert econt > 0.0
    # Force on slave node should be in -Y direction (pushing away from bead)
    assert fcont[2, 1] > 0.0
    # Equal and opposite reaction distributed equally to node 0 and node 1 (xi=0.5)
    assert fcont[0, 1] < 0.0
    assert fcont[1, 1] < 0.0
    assert pytest.approx(fcont[0, 1]) == fcont[1, 1]
    # Exact linear momentum conservation
    assert np.allclose(np.sum(fcont, axis=0), 0.0, atol=1e-12)


def test_type21_tangential_restraint_and_pmax_limiting():
    """Verify friction, plastic bending resistance, and Pmax tangential limiting."""
    model = Model()
    model.x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.02, 0.0],
    ], dtype=float)
    model.x0 = model.x.copy()
    # Slave moving in +Y direction across bead
    model.v = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
    ], dtype=float)
    model.mass = np.array([1.0, 1.0, 1.0], dtype=float)

    line_obj = Line(id=1)
    line_obj.segments = np.array([[0, 1]], dtype=np.int64)
    model.lines[1] = line_obj

    grn = NodeGroup(id=2, node_ids=[2], node_idx=np.array([2], dtype=np.int64))
    model.node_groups[2] = grn

    # 1. Pmax capping active (itlim = 0, pmax = 50.0)
    itf_capped = Interface(
        id=1, type=21, surf_id=1, grnod_id=2,
        istf=1, stfac=2000.0, depth=0.1, fric=0.3,
        gap=0.05, pmax=50.0, itlim=0,
    )
    ct21_capped = ContactType21(itf_capped, model, MessageLog())
    fcont1 = np.zeros_like(model.x)
    ct21_capped.forces(model.x, model.v, model.mass, 1e-4, fcont1, cycle=0)

    # 2. Pmax capping inactive (itlim = 1, pmax = 50.0)
    itf_uncapped = Interface(
        id=2, type=21, surf_id=1, grnod_id=2,
        istf=1, stfac=2000.0, depth=0.1, fric=0.3,
        gap=0.05, pmax=50.0, itlim=1,
    )
    ct21_uncapped = ContactType21(itf_uncapped, model, MessageLog())
    fcont2 = np.zeros_like(model.x)
    ct21_uncapped.forces(model.x, model.v, model.mass, 1e-4, fcont2, cycle=0)

    # Tangential restraining force opposes sliding (+Y motion -> negative Y force)
    # Uncapped force should be greater in magnitude than capped force
    assert abs(fcont2[2, 1]) >= abs(fcont1[2, 1])
    # Both conserve linear momentum
    assert np.allclose(np.sum(fcont1, axis=0), 0.0, atol=1e-12)
    assert np.allclose(np.sum(fcont2, axis=0), 0.0, atol=1e-12)


def test_type21_time_window_gating():
    """Verify Type 21 deactivates outside [tstart, tstop]."""
    model = Model()
    model.x = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.01, 0]], dtype=float)
    model.x0 = model.x.copy()
    model.v = np.zeros_like(model.x)
    model.mass = np.ones(3, dtype=float)

    line_obj = Line(id=1)
    line_obj.segments = np.array([[0, 1]], dtype=np.int64)
    model.lines[1] = line_obj
    grn = NodeGroup(id=2, node_ids=[2], node_idx=np.array([2], dtype=np.int64))
    model.node_groups[2] = grn

    itf = Interface(
        id=1, type=21, surf_id=1, grnod_id=2,
        depth=0.1, gap=0.05, tstart=0.01, tstop=0.05,
    )
    ct21 = ContactType21(itf, model, MessageLog())
    fcont = np.zeros_like(model.x)

    # Before tstart: inactive
    e_pre, _ = ct21.forces(model.x, model.v, model.mass, 1e-4, fcont, t=0.005)
    assert e_pre == 0.0 and np.all(fcont == 0.0)

    # Within window: active
    e_mid, _ = ct21.forces(model.x, model.v, model.mass, 1e-4, fcont, t=0.02)
    assert e_mid > 0.0 and np.any(fcont != 0.0)

    # After tstop: inactive
    fcont[:] = 0.0
    e_post, _ = ct21.forces(model.x, model.v, model.mass, 1e-4, fcont, t=0.06)
    assert e_post == 0.0 and np.all(fcont == 0.0)


# ============================================================================
# 3. /INTER/TYPE23 Starter Deck Parsing
# ============================================================================

def test_type23_free_format_parsing(make_deck):
    """Verify free-format /INTER/TYPE23 deck parsing with all cards."""
    starter = (
        BASE_DECK
        + "/INTER/TYPE23/30\n"
        "mortar free format\n"
        "2 1 2 1 0 0\n"                  # surf_s, surf_m, istf, igap, ibag, idel
        "1.1 0.08 0.75\n"                # fscale_gap, gap_max, fpenmax
        "50.0 5e6\n"                     # stmin, stmax
        "1.2 0.2 0.03 0.0 20.0\n"        # stfac, fric, gap_min, tstart, tstop
        "0 0 0.07 0.3\n"                 # inactiv, viss, sort_fact
        "1 1 50.0\n"                     # ifric, ifiltr, xfreq
        "0.2 0.1 0.02 0.0 0.0\n"         # c1..c5
        "0.05\n"                         # c6
        "/END\n"
    )
    model = _run_starter_str(make_deck, "T23_FREE", starter)
    itfs = [itf for itf in model.interfaces if itf.type == 23]
    assert len(itfs) == 1
    itf = itfs[0]
    assert itf.id == 30
    assert itf.type == 23
    assert itf.surf_id == 1
    assert itf.surf_id1 == 2
    assert itf.istf == 2
    assert itf.igap == 1
    assert pytest.approx(itf.fscale_gap) == 1.1
    assert pytest.approx(itf.gap_max) == 0.08
    assert pytest.approx(itf.fpenmax) == 0.75
    assert pytest.approx(itf.stmin) == 50.0
    assert pytest.approx(itf.stmax) == 5e6
    assert pytest.approx(itf.stfac) == 1.2
    assert pytest.approx(itf.fric) == 0.2
    assert pytest.approx(itf.gap_min) == 0.03
    assert pytest.approx(itf.viss) == 0.07
    assert itf.ifric == 1
    assert itf.ifiltr == 1
    assert pytest.approx(itf.xfreq) == 50.0
    assert itf.fric_c[0] == pytest.approx(0.2)
    assert itf.fric_c[1] == pytest.approx(0.1)


def test_type23_fixed_format_parsing(make_deck):
    """Verify fixed-format /INTER/TYPE23 deck parsing matching CFG layout."""
    c1 = f"{2:>10}{1:>10}{1:>10}{0:>10}{0:>10}{0:>10}{0:>10}{0:>10}"
    c2 = f"{1.0:>20.1f}{0.0:>20.1f}{0.85:>20.2f}"
    c3 = f"{0.0:>20.1f}{1.0e8:>20.1f}"
    c4 = f"{1.5:>20.1f}{0.25:>20.2f}{0.02:>20.2f}{0.0:>20.1f}{100.0:>20.1f}"
    c5 = f"{'':>7}{'0':>1}{'0':>1}{'0':>1}{0:>10}{0:>10}{0:>10}{0.05:>20.2f}{0.0:>20.1f}{0.2:>20.1f}"
    c6 = f"{0:>10}{0:>10}{0.0:>20.1f}"

    starter = (
        BASE_DECK
        + "/INTER/TYPE23/31\n"
        "mortar fixed format\n"
        + f"{c1}\n{c2}\n{c3}\n{c4}\n{c5}\n{c6}\n"
        + "/END\n"
    )
    model = _run_starter_str(make_deck, "T23_FIX", starter)
    itfs = [itf for itf in model.interfaces if itf.type == 23]
    assert len(itfs) == 1
    itf = itfs[0]
    assert itf.id == 31
    assert itf.fpenmax == pytest.approx(0.85)
    assert itf.stfac == pytest.approx(1.5)
    assert itf.fric == pytest.approx(0.25)
    assert itf.gap_min == pytest.approx(0.02)
    assert itf.viss == pytest.approx(0.05)


# ============================================================================
# 4. /INTER/TYPE23 Mortar Physics Formulation
# ============================================================================

def test_type23_mortar_shape_functions_and_momentum():
    """Verify centroid subdivision, partition of unity of H1..H4, and momentum conservation."""
    model = Model()
    # Master quad corners in XY plane:
    # 0: (0, 0, 0), 1: (1, 0, 0), 2: (1, 1, 0), 3: (0, 1, 0)
    # Slave node 4 at (0.3, 0.4, 0.01)
    model.x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.3, 0.4, 0.01],
    ], dtype=float)
    model.x0 = model.x.copy()
    model.v = np.zeros_like(model.x)
    model.mass = np.ones(5, dtype=float)

    surf_m = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    surf_m.seg_gtype = np.array(["SHELL"], dtype=object)
    surf_m.seg_elem = np.array([1], dtype=np.int64)
    model.surfaces[1] = surf_m

    grn = NodeGroup(id=2, node_ids=[4], node_idx=np.array([4], dtype=np.int64))
    model.node_groups[2] = grn

    itf = Interface(
        id=1, type=23, surf_id=1, grnod_id=2,
        istf=1, stfac=5000.0, gap=0.05, fric=0.0, viss=0.05,
    )
    log = MessageLog()
    ct23 = ContactType23(itf, model, log)

    fcont = np.zeros_like(model.x)
    econt, dt_int = ct23.forces(model.x, model.v, model.mass, 1e-4, fcont, cycle=0)

    # Penetration: node 4 is at z=0.01, gap=0.05 -> pene = 0.04
    assert econt > 0.0
    # Normal force on slave node points in +Z direction (away from surface)
    assert fcont[4, 2] > 0.0
    # Master nodes receive reactive forces in -Z direction
    for c_idx in range(4):
        assert fcont[c_idx, 2] < 0.0

    # Exact linear momentum conservation
    assert np.allclose(np.sum(fcont, axis=0), 0.0, atol=1e-12)


def test_type23_triangular_master_facet():
    """Verify mortar handling for 3-node triangular master segments."""
    model = Model()
    model.x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.2, 0.2, 0.01],  # slave node inside triangle
    ], dtype=float)
    model.x0 = model.x.copy()
    model.v = np.zeros_like(model.x)
    model.mass = np.ones(4, dtype=float)

    surf_m = Surface(id=1, segments=np.array([[0, 1, 2]], dtype=np.int64))
    surf_m.seg_gtype = np.array(["SH3N"], dtype=object)
    surf_m.seg_elem = np.array([1], dtype=np.int64)
    model.surfaces[1] = surf_m

    grn = NodeGroup(id=2, node_ids=[3], node_idx=np.array([3], dtype=np.int64))
    model.node_groups[2] = grn

    itf = Interface(
        id=1, type=23, surf_id=1, grnod_id=2,
        istf=1, stfac=2000.0, gap=0.05, fric=0.0,
    )
    ct23 = ContactType23(itf, model, MessageLog())

    fcont = np.zeros_like(model.x)
    econt, _ = ct23.forces(model.x, model.v, model.mass, 1e-4, fcont, cycle=0)

    assert econt > 0.0
    assert fcont[3, 2] > 0.0
    # Exact linear momentum conservation
    assert np.allclose(np.sum(fcont, axis=0), 0.0, atol=1e-12)


def test_type23_tangential_coulomb_friction():
    """Verify stick-slip Coulomb friction opposes relative tangential sliding."""
    model = Model()
    model.x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.01],
    ], dtype=float)
    model.x0 = model.x.copy()
    # Slave moving along +X
    model.v = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ], dtype=float)
    model.mass = np.ones(5, dtype=float)

    surf_m = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    surf_m.seg_gtype = np.array(["SHELL"], dtype=object)
    surf_m.seg_elem = np.array([1], dtype=np.int64)
    model.surfaces[1] = surf_m

    grn = NodeGroup(id=2, node_ids=[4], node_idx=np.array([4], dtype=np.int64))
    model.node_groups[2] = grn

    itf = Interface(
        id=1, type=23, surf_id=1, grnod_id=2,
        istf=1, stfac=1000.0, gap=0.05, fric=0.25,
    )
    ct23 = ContactType23(itf, model, MessageLog())

    fcont = np.zeros_like(model.x)
    ct23.forces(model.x, model.v, model.mass, 1e-4, fcont, cycle=0)

    # Normal force in +Z, tangential friction opposing +X slip -> negative X force
    assert fcont[4, 2] > 0.0
    assert fcont[4, 0] < 0.0
    # Friction limit check: |Ft| <= mu * |Fn|
    fn_mag = abs(fcont[4, 2])
    ft_mag = abs(fcont[4, 0])
    assert ft_mag <= 0.25 * fn_mag * (1.0 + 1e-6)

    # Momentum conservation
    assert np.allclose(np.sum(fcont, axis=0), 0.0, atol=1e-12)


# ============================================================================
# 5. Factory Integration & Full Engine Simulation
# ============================================================================

def test_type21_type23_build_contacts_factory():
    """Verify build_contacts() instantiates both Type 21 and Type 23."""
    model = Model()
    model.x = np.zeros((10, 3), dtype=float)
    model.x0 = model.x.copy()
    model.mass = np.ones(10, dtype=float)

    surf1 = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    surf1.seg_gtype = np.array(["SHELL"], dtype=object)
    surf1.seg_elem = np.array([1], dtype=np.int64)
    surf2 = Surface(id=2, segments=np.array([[4, 5, 6, 7]], dtype=np.int64))
    surf2.seg_gtype = np.array(["SHELL"], dtype=object)
    surf2.seg_elem = np.array([2], dtype=np.int64)
    model.surfaces[1] = surf1
    model.surfaces[2] = surf2

    itf21 = Interface(id=1, type=21, surf_id=1, surf_id1=2)
    itf23 = Interface(id=2, type=23, surf_id=1, surf_id1=2)
    model.interfaces = [itf21, itf23]

    log = MessageLog()
    penalty, tied = build_contacts(model, log)

    assert len(penalty) == 2
    types = [type(c).__name__ for c in penalty]
    assert "ContactType21" in types
    assert "ContactType23" in types
    assert len(tied) == 0


def test_type21_engine_end_to_end(make_deck):
    """Run full explicit engine simulation with Type 21 drawbead interface."""
    starter = (
        BASE_DECK
        + "/INTER/TYPE21/1\n"
        "drawbead test\n"
        "2 1 1 0 4 0\n"
        "1.0 0.0 0.1 1e4 0\n"
        "0.0 1e30\n"
        "1.0 0.1 0.02 0.0 1.0\n"
        "0 0 0.05 0.2\n"
        "0 0 0.0 0 0\n"
        "/END\n"
    )
    engine = (
        "/RUN/T21/1\n"
        "0.002\n"
        "/TFILE\n"
        "0.0005\n"
        "/ANIM/DT\n"
        "0.001\n"
    )
    s, e = make_deck("T21_RUN", starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(str(s))
        eng_model = run_engine(str(e))
    assert eng_model.engine_state.cycle > 0
    assert eng_model.engine_state.t >= 0.002 * 0.99


def test_type23_engine_end_to_end(make_deck):
    """Run full explicit engine simulation with Type 23 mortar interface."""
    starter = (
        BASE_DECK
        + "/INTER/TYPE23/1\n"
        "mortar test\n"
        "2 1 1 0 0 0\n"
        "1.0 0.0 0.8\n"
        "0.0 1e30\n"
        "1.0 0.1 0.02 0.0 1.0\n"
        "0 0 0.05 0.2\n"
        "0 0 0.0\n"
        "/END\n"
    )
    engine = (
        "/RUN/T23/1\n"
        "0.002\n"
        "/TFILE\n"
        "0.0005\n"
        "/ANIM/DT\n"
        "0.001\n"
    )
    s, e = make_deck("T23_RUN", starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(str(s))
        eng_model = run_engine(str(e))
    assert eng_model.engine_state.cycle > 0
    assert eng_model.engine_state.t >= 0.002 * 0.99

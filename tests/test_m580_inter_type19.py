"""
Tests for M580: /INTER/TYPE19 compound contact interface auto-expansion into
ContactType7 (node-to-surface penalty) and ContactType11 (edge-to-edge penalty).

Fortran origins:
  - starter/source/starter/starter0.F (CALL HM_CONVERT_INTER_TYPE19)
  - starter/source/devtools/hm_reader/hm_convert_inter_type19.F (TYPE19 -> TYPE7 + TYPE11)
  - starter/source/interfaces/int07/hm_read_inter_type07.F (ID_TYPE19 handling)
  - starter/source/interfaces/int11/hm_read_inter_type11.F (ID_TYPE19 and line deactivation)
  - engine/source/interfaces/int07/ (penalty node-to-surface)
  - engine/source/interfaces/int11/ (penalty edge-to-edge)
"""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.contact import ContactType7, ContactType11, build_contacts
from pyradioss.engine.engine import run_engine
from pyradioss.input.deck_reader import Card, KeywordBlock
from pyradioss.input.starter_keywords import read_inter
from pyradioss.model.entities import Interface, Line, NodeGroup, Surface
from pyradioss.model.model import Model
from pyradioss.starter.starter import run_starter


STEEL_LAW1 = """\
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
"""


def _run(make_deck, name: str, starter: str, engine: str):
    s, e = make_deck(name, starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    return model, e.replace("_0001.rad", "_0001.out")


def _starter_only(make_deck, name: str, starter: str):
    s, _ = make_deck(name, starter, "/RUN/X/1\n1.0\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(s)


def _final_summary(out_path: str):
    """Parse engine termination block."""
    text = open(out_path).read()
    err_match = re.search(r"ENERGY ERROR\s*\.[ .]*:\s*([-\d.Ee+]+)", text)
    ce_match = re.search(r"CONTACT ENERGY\s*\.[ .]*:\s*([-\d.Ee+]+)", text)
    err = float(err_match.group(1)) if err_match else 0.0
    ce = float(ce_match.group(1)) if ce_match else 0.0
    return {
        "ERR": err,
        "CE": ce,
        "NORMAL": "ENGINE TERMINATION : NORMAL" in text,
    }


# ============================================================================
# 1. Unit Tests: build_contacts() Factory Auto-Expansion
# ============================================================================

def test_type19_build_contacts_surface_only():
    """TYPE19 with no edge lines (line_id1=0, line_id2=0) expands to ContactType7 only."""
    model = Model()
    model.node_ids = np.array([1, 2, 3, 4, 5])
    model.x0 = np.zeros((5, 3))
    model.x = model.x0.copy()
    model.v = np.zeros((5, 3))
    model.mass = np.ones(5)

    surf = Surface(id=1, title="TARGET_SURF")
    surf.segments = np.array([[0, 1, 2, 3]], dtype=np.int64)
    surf.seg_gtype = np.full(1, "", dtype="<U8")
    surf.seg_elem = np.full(1, -1, dtype=np.int64)
    model.surfaces[1] = surf
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([4], dtype=np.int64))

    itf = Interface(
        id=1,
        type=19,
        grnod_id=1,
        surf_id=1,
        line_id1=0,
        line_id2=0,
        istf=1,
        stfac=1000.0,
        gap=0.1,
    )
    model.interfaces.append(itf)

    log = MessageLog()
    penalty, tied = build_contacts(model, log)

    assert len(tied) == 0
    assert len(penalty) == 1
    assert isinstance(penalty[0], ContactType7)
    assert penalty[0].itf.id == 1


def test_type19_build_contacts_compound_surface_and_edges():
    """TYPE19 with defined lines (line_id1>0, line_id2>0) expands to BOTH ContactType7 and ContactType11."""
    model = Model()
    model.node_ids = np.arange(1, 9)
    model.x0 = np.zeros((8, 3))
    model.x = model.x0.copy()
    model.v = np.zeros((8, 3))
    model.mass = np.ones(8)

    # Surface + Node Group for Type 7
    surf = Surface(id=1, title="MAIN_SURF")
    surf.segments = np.array([[0, 1, 2, 3]], dtype=np.int64)
    surf.seg_gtype = np.full(1, "", dtype="<U8")
    surf.seg_elem = np.full(1, -1, dtype=np.int64)
    model.surfaces[1] = surf
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([4], dtype=np.int64))

    # Lines for Type 11
    ln1 = Line(id=1, title="SEC_LINE")
    ln1.segments = np.array([[4, 5]], dtype=np.int64)
    model.lines[1] = ln1

    ln2 = Line(id=2, title="MAIN_LINE")
    ln2.segments = np.array([[6, 7]], dtype=np.int64)
    model.lines[2] = ln2

    itf = Interface(
        id=10,
        type=19,
        grnod_id=1,
        surf_id=1,
        line_id1=1,
        line_id2=2,
        istf=1,
        stfac=500.0,
        gap=0.05,
    )
    model.interfaces.append(itf)

    log = MessageLog()
    penalty, tied = build_contacts(model, log)

    assert len(tied) == 0
    assert len(penalty) == 2
    assert isinstance(penalty[0], ContactType7)
    assert isinstance(penalty[1], ContactType11)
    assert penalty[0].itf.id == 10
    assert penalty[1].itf.id == 10


def test_type19_build_contacts_partial_lines_deactivated():
    """TYPE19 with only one line defined does not activate Type 11 (requires both line_id1 > 0 and line_id2 > 0)."""
    model = Model()
    model.node_ids = np.arange(1, 6)
    model.x0 = np.zeros((5, 3))
    model.x = model.x0.copy()
    model.v = np.zeros((5, 3))
    model.mass = np.ones(5)

    surf = Surface(id=1, title="SURF")
    surf.segments = np.array([[0, 1, 2, 3]], dtype=np.int64)
    surf.seg_gtype = np.full(1, "", dtype="<U8")
    surf.seg_elem = np.full(1, -1, dtype=np.int64)
    model.surfaces[1] = surf

    # Case A: line_id1 defined, line_id2 = 0
    itf_a = Interface(id=1, type=19, surf_id=1, line_id1=1, line_id2=0)
    model.interfaces = [itf_a]
    penalty, _ = build_contacts(model, MessageLog())
    assert len(penalty) == 1
    assert isinstance(penalty[0], ContactType7)

    # Case B: line_id1 = 0, line_id2 defined
    itf_b = Interface(id=2, type=19, surf_id=1, line_id1=0, line_id2=2)
    model.interfaces = [itf_b]
    penalty, _ = build_contacts(model, MessageLog())
    assert len(penalty) == 1
    assert isinstance(penalty[0], ContactType7)


# ============================================================================
# 2. Starter Keyword Parsing Tests
# ============================================================================

def test_starter_keyword_inter_type19_free():
    """Free format /INTER/TYPE19 keyword correctly populates Interface attributes including line_id1 and line_id2."""
    block = KeywordBlock(
        keyword="/INTER/TYPE19/42",
        parts=["INTER", "TYPE19", "42"],
        user_id=42,
        cards=[
            Card("Compound interface test"),
            Card("10 20 2 1 0 0 0 0"),
            Card("1.1 0.05"),
            Card("100.0 10000.0"),
            Card("5 6"),
            Card("1.5 0.25 0.02 0.0 100.0"),
        ],
        fixed=False,
    )
    model = Model()
    log = MessageLog()
    read_inter(block, model, log)

    assert len(model.interfaces) == 1
    itf = model.interfaces[0]
    assert itf.id == 42
    assert itf.type == 19
    assert itf.grnod_id == 10
    assert itf.surf_id == 20
    assert itf.istf == 2
    assert itf.igap == 1
    assert itf.line_id1 == 5
    assert itf.line_id2 == 6
    assert itf.gap_scale == pytest.approx(1.1)
    assert itf.gap_max == pytest.approx(0.05)
    assert itf.stmin == pytest.approx(100.0)
    assert itf.stmax == pytest.approx(10000.0)
    assert itf.stfac == pytest.approx(1.5)
    assert itf.fric == pytest.approx(0.25)
    assert itf.gap == pytest.approx(0.02)
    assert itf.tstop == pytest.approx(100.0)


def test_starter_keyword_inter_type19_fixed():
    """Fixed format /INTER/TYPE19 keyword parses into Model with all fields preserved."""
    card0 = "        15        25         1                   0         0         0         1         0"
    card1 = "                 1.0                 0.1"
    card2 = "                50.0              5000.0"
    card3 = "         7         8"
    card4 = "                 2.0                 0.1                0.01                 0.0                50.0"

    block = KeywordBlock(
        keyword="/INTER/TYPE19/99",
        parts=["INTER", "TYPE19", "99"],
        user_id=99,
        cards=[
            Card("Fixed compound interface"),
            Card(card0),
            Card(card1),
            Card(card2),
            Card(card3),
            Card(card4),
        ],
        fixed=True,
    )
    model = Model()
    log = MessageLog()
    read_inter(block, model, log)

    assert len(model.interfaces) == 1
    itf = model.interfaces[0]
    assert itf.id == 99
    assert itf.type == 19
    assert itf.grnod_id == 15
    assert itf.surf_id == 25
    assert itf.line_id1 == 7
    assert itf.line_id2 == 8
    assert itf.istf == 1
    assert itf.stfac == pytest.approx(2.0)
    assert itf.fric == pytest.approx(0.1)
    assert itf.gap == pytest.approx(0.01)


# ============================================================================
# 3. Contact Forces Evaluation & Momentum Balance
# ============================================================================

def test_type19_forces_evaluation_both_active():
    """Evaluate contact forces on both ContactType7 and ContactType11 components of a TYPE19 interface."""
    model = Model()
    # 4 quad nodes (0..3), 1 secondary surface node (4), 2 edge nodes line1 (5..6), 2 edge nodes line2 (7..8)
    coords = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 1.0,  1.0, 0.0],
        [-1.0,  1.0, 0.0],
        [ 0.0,  0.0, 0.04],   # node 4 penetrating surface gap (0.1) at z=0.04
        [-1.0,  0.0, 0.03],   # line 1 edge along X at z=0.03
        [ 1.0,  0.0, 0.03],
        [ 0.0, -1.0, 0.0],    # line 2 edge along Y at z=0.0
        [ 0.0,  1.0, 0.0],
    ])
    n = len(coords)
    model.node_ids = np.arange(1, n + 1)
    model.x0 = coords.copy()
    model.x = coords.copy()
    model.v = np.zeros((n, 3))
    model.mass = np.ones(n)

    # Master surface (quad 0-1-2-3) and slave node group (node 4)
    surf = Surface(id=1, title="SURF1")
    surf.segments = np.array([[0, 1, 2, 3]], dtype=np.int64)
    surf.seg_gtype = np.full(1, "", dtype="<U8")
    surf.seg_elem = np.full(1, -1, dtype=np.int64)
    model.surfaces[1] = surf
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([4], dtype=np.int64))

    # Crossing lines (nodes 5-6 along X, nodes 7-8 along Y)
    ln1 = Line(id=1, title="LINE1")
    ln1.segments = np.array([[5, 6]], dtype=np.int64)
    ln1.seg_gtype = np.full(1, "", dtype="<U8")
    ln1.seg_elem = np.full(1, -1, dtype=np.int64)
    model.lines[1] = ln1

    ln2 = Line(id=2, title="LINE2")
    ln2.segments = np.array([[7, 8]], dtype=np.int64)
    ln2.seg_gtype = np.full(1, "", dtype="<U8")
    ln2.seg_elem = np.full(1, -1, dtype=np.int64)
    model.lines[2] = ln2

    itf = Interface(
        id=1,
        type=19,
        grnod_id=1,
        surf_id=1,
        line_id1=1,
        line_id2=2,
        istf=1,
        stfac=1000.0,
        gap=0.1,
    )
    model.interfaces.append(itf)

    log = MessageLog()
    penalty, _ = build_contacts(model, log)
    assert len(penalty) == 2
    ct7, ct11 = penalty[0], penalty[1]

    # Evaluate Type 7 forces
    fcont7 = np.zeros((n, 3))
    _, dt_7 = ct7.forces(model.x, model.v, model.mass, 1e-4, fcont7, cycle=0)
    # Secondary node 4 pushed upwards (+Z)
    assert fcont7[4, 2] > 0.0
    # Quad corners pushed downwards (action-reaction)
    assert np.all(fcont7[0:4, 2] <= 0.0)
    assert fcont7[0:4, 2].sum() < 0.0
    # Newton's 3rd law: sum of forces is 0
    np.testing.assert_allclose(fcont7.sum(axis=0), 0.0, atol=1e-12)

    # Evaluate Type 11 forces
    fcont11 = np.zeros((n, 3))
    _, dt_11 = ct11.forces(model.x, model.v, model.mass, 1e-4, fcont11, cycle=0)
    # Line 1 edge pushed upwards (+Z)
    assert fcont11[5, 2] > 0.0 and fcont11[6, 2] > 0.0
    # Line 2 edge pushed downwards (-Z)
    assert fcont11[7, 2] < 0.0 and fcont11[8, 2] < 0.0
    # Newton's 3rd law: sum of forces is 0
    np.testing.assert_allclose(fcont11.sum(axis=0), 0.0, atol=1e-12)


# ============================================================================
# 4. Dynamic Simulation: Surface Impact with /INTER/TYPE19
# ============================================================================

def test_type19_dynamic_surface_impact(make_deck):
    """Full starter+engine dynamic run of self/surface impact using /INTER/TYPE19:
    Upper plate impacts lower plate; striker is arrested/bounces, contact energy accumulates,
    and energy conservation holds with low error."""
    nodes, shells = [], []
    # lower plate at z=0, upper plate at z=0.8
    for z, base in ((0.0, 0), (0.8, 100)):
        for j in range(3):
            for i in range(3):
                nodes.append(f"{base + 1 + i + 4 * j} {5.0 * i} {5.0 * j} {z}")
    for base, eb in ((0, 0), (100, 10)):
        for j in range(2):
            for i in range(2):
                n1 = base + 1 + i + 4 * j
                shells.append(f"{eb + 1 + i + 2 * j} {n1} {n1 + 1} {n1 + 5} {n1 + 4}")
    upper = " ".join(str(100 + 1 + i + 4 * j) for j in range(3) for i in range(3))

    starter = (
        "/BEGIN\ntype19 surface impact\n"
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/SHELL/1\n" + "\n".join(shells) + "\n"
        "/PART/1\nboth plates\n1 1\n" + STEEL_LAW1 +
        "/PROP/SHELL/1\nt=0.4\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.4\n"
        f"/GRNOD/NODE/1\nupper plate\n{upper}\n"
        "/INIVEL/TRA/1\nfall\n0 0 -0.5 1\n"
        "/SURF/PART/1\nself\n1\n"
        "/INTER/TYPE19/1\ncompound surface only\n"
        "0 1 2 1 0 0 0 0\n"
        "1.0 0.0\n"
        "0.0 0.0\n"
        "0 0\n"
        "1.0 0.0 0.0 0.0 1.0e30\n"
        "/END\n"
    )
    engine = "/RUN/T19S/1\n2.0\n/DT\n0.9 0\n/PRINT/-100000\n/STOP\n20.0\n"
    model, out = _run(make_deck, "T19S", starter, engine)

    lower = model.node_indices([1 + i + 4 * j for j in range(3) for i in range(3)])
    upperi = model.node_indices([100 + 1 + i + 4 * j for j in range(3) for i in range(3)])

    # Striker did not penetrate through lower plate
    assert model.x[upperi, 2].min() > model.x[lower, 2].max() - 1e-9
    # Striker arrested / bouncing
    assert model.v[upperi, 2].mean() > -0.4
    s = _final_summary(out)
    assert s["NORMAL"], f"Engine failed: {s}"
    assert abs(s["ERR"]) < 5.0, f"Energy error too large: {s['ERR']}%"


# ============================================================================
# 5. Dynamic Simulation: Edge Impact with /INTER/TYPE19
# ============================================================================

def test_type19_dynamic_edge_impact(make_deck):
    """Full starter+engine dynamic run of edge crossing impact using /INTER/TYPE19:
    Line-to-line contact arrests falling strip; momentum and total energy are conserved."""
    starter = (
        "/BEGIN\ntype19 edge impact\n"
        "/NODE\n"
        # main strip along x (3 elements, width 2, z = 0)
        "1 -3 -1 0\n2 -1 -1 0\n3 1 -1 0\n4 3 -1 0\n"
        "5 -3 1 0\n6 -1 1 0\n7 1 1 0\n8 3 1 0\n"
        # secondary strip along y (3 elements, width 2, z = 1.2)
        "11 -1 -3 1.2\n12 -1 -1 1.2\n13 -1 1 1.2\n14 -1 3 1.2\n"
        "15 1 -3 1.2\n16 1 -1 1.2\n17 1 1 1.2\n18 1 3 1.2\n"
        "/SHELL/1\n1 1 2 6 5\n2 2 3 7 6\n3 3 4 8 7\n"
        "/SHELL/2\n11 11 12 16 15\n12 12 13 17 16\n13 13 14 18 17\n"
        "/PART/1\nmain strip\n1 1\n"
        "/PART/2\nflyer strip\n2 1\n" + STEEL_LAW1 +
        "/PROP/SHELL/1\nstrip\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.6\n"
        "/PROP/SHELL/2\nstrip\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.6\n"
        "/GRNOD/PART/2\nflyer\n2\n"
        "/INIVEL/TRA/1\nfall\n0 0 -1.0 2\n"
        "/SURF/PART/1\nmain surf\n1\n"
        "/SURF/PART/2\nflyer surf\n2\n"
        "/LINE/SURF/1\nflyer edges\n2\n"
        "/LINE/SURF/2\nmain edges\n1\n"
        "/INTER/TYPE19/1\nedge impact\n"
        "0 1 2 1 0 0 0 0\n"
        "1.0 0.0\n"
        "0.0 0.0\n"
        "1 2\n"
        "1.0 0.0 0.0 0.0 1.0e30\n"
        "/END\n"
    )
    engine = "/RUN/T19E/1\n3.0\n/DT\n0.9 0\n/PRINT/-100000\n/STOP\n20.0\n"
    model, out = _run(make_deck, "T19E", starter, engine)

    real = model.mass < 1e29
    pz = float((model.mass[real] * model.v[real, 2]).sum())
    flyer = model.node_indices([11, 12, 13, 14, 15, 16, 17, 18])
    m_flyer = model.mass[flyer].sum()
    # Total momentum is conserved exactly
    assert pz == pytest.approx(-m_flyer * 1.0, rel=1e-9)

    # Edge-to-edge contact prevented crossing
    flyer_mid = model.node_indices([12, 13, 16, 17])
    main_mid = model.node_indices([2, 3, 6, 7])
    assert model.x[flyer_mid, 2].min() > model.x[main_mid, 2].max() + 0.2

    # Center of mass velocity indicates arrest
    mf = model.mass[flyer]
    vz_com = float((mf * model.v[flyer, 2]).sum() / mf.sum())
    assert vz_com > -0.5

    s = _final_summary(out)
    assert s["NORMAL"]
    assert abs(s["ERR"]) < 5.0


# ============================================================================
# 6. Dynamic Simulation: Compound Impact (Both Surface & Edges Active)
# ============================================================================

def test_type19_dynamic_compound_surface_and_edges(make_deck):
    """Full starter+engine dynamic run of compound impact where /INTER/TYPE19 handles
    both surface contact and edge contact simultaneously in one interface."""
    starter = (
        "/BEGIN\ncompound impact\n"
        "/NODE\n"
        # Master plate (z=0)
        "1 -2 -2 0\n2 2 -2 0\n3 2 2 0\n4 -2 2 0\n"
        # Striker shell above (z=0.8)
        "5 -1 -1 0.8\n6 1 -1 0.8\n7 1 1 0.8\n8 -1 1 0.8\n"
        "/SHELL/1\n1 1 2 3 4\n"
        "/SHELL/2\n2 5 6 7 8\n"
        "/PART/1\nmain plate\n1 1\n"
        "/PART/2\nstriker\n2 1\n" + STEEL_LAW1 +
        "/PROP/SHELL/1\nplate\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.5\n"
        "/PROP/SHELL/2\nstriker\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.5\n"
        "/GRNOD/PART/2\nstriker nodes\n2\n"
        "/INIVEL/TRA/1\nfall\n0 0 -0.8 2\n"
        "/SURF/PART/1\nmain surf\n1\n"
        "/SURF/PART/2\nstriker surf\n2\n"
        "/LINE/SURF/1\nstriker edges\n2\n"
        "/LINE/SURF/2\nmain edges\n1\n"
        "/INTER/TYPE19/1\ncompound full\n"
        "2 1 2 1 0 0 0 0\n"
        "1.0 0.0\n"
        "0.0 0.0\n"
        "1 2\n"
        "1.0 0.0 0.0 0.0 1.0e30\n"
        "/END\n"
    )
    engine = "/RUN/T19C/1\n2.0\n/DT\n0.9 0\n/PRINT/-100000\n/STOP\n20.0\n"
    model, out = _run(make_deck, "T19C", starter, engine)

    striker_nodes = model.node_indices([5, 6, 7, 8])
    main_nodes = model.node_indices([1, 2, 3, 4])

    # Striker bounced/arrested and stayed above master surface
    assert model.x[striker_nodes, 2].min() > model.x[main_nodes, 2].max() - 1e-6
    assert model.v[striker_nodes, 2].mean() > -0.6

    s = _final_summary(out)
    assert s["NORMAL"]
    assert abs(s["ERR"]) < 5.0

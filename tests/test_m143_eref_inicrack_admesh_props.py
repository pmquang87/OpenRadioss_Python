"""Tests for Milestone M143: Extended Initial References (/EREF), X-FEM Cracks (/INICRACK),
Rivet & X-FEM Properties (/PROP/TYPE5, /PROP/TYPE28), and Adaptive Meshing Controls (/ADMESH/*).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m143_eref_elements(tmp_path):
    deck = """/BEGIN
Test Eref Elements
/EREF/SHELL/10
Reference Shell Configuration
1 0.0 0.0 0.0
2 10.0 0.0 0.0
3 10.0 10.0 0.0
4 0.0 10.0 0.0
/EREF/SOLID/20
Reference Solid Configuration
101 0.0 0.0 0.0
102 5.0 5.0 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 10 in model.eref_elements
    ref10 = model.eref_elements[10]
    assert ref10.elem_type == "SHELL"
    assert ref10.part_id == 10
    assert 1 in ref10.node_coords
    assert np.allclose(ref10.node_coords[2], [10.0, 0.0, 0.0])

    assert 20 in model.eref_elements
    ref20 = model.eref_elements[20]
    assert ref20.elem_type == "SOLID"
    assert np.allclose(ref20.node_coords[102], [5.0, 5.0, 5.0])


def test_m143_inicrack_geometry(tmp_path):
    deck = """/BEGIN
Test IniCrack Geometry
/INICRACK/1
Planar Initial Crack
100 1
0.0 5.0 0.0
10.0 5.0 0.0
0.0 0.0 1.0
/INICRACK/2
Segment Initial Crack
101 102 0.5
102 103 0.25
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.ini_cracks
    cr1 = model.ini_cracks[1]
    assert cr1.grsh_id == 100
    assert cr1.open_flag == 1
    assert np.allclose(cr1.p1, [0.0, 5.0, 0.0])
    assert np.allclose(cr1.p2, [10.0, 5.0, 0.0])
    assert np.allclose(cr1.norm, [0.0, 0.0, 1.0])

    assert 2 in model.ini_cracks
    cr2 = model.ini_cracks[2]
    assert len(cr2.segments) == 2
    assert cr2.segments[0].node_id1 == 101
    assert cr2.segments[0].ratio == pytest.approx(0.5)


def test_m143_prop_rivet_and_xelem(tmp_path):
    deck = """/BEGIN
Test Rivet and X-FEM Properties
/PROP/RIVET/5
Spotweld Rivet Fastener
0.001 50000.0 1200.0 800.0
/PROP/TYPE28/28
Cohesive X-FEM Element
1 2 0.35
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 5 in model.prop_rivets
    riv = model.prop_rivets[5]
    assert riv.mass == pytest.approx(0.001)
    assert riv.stiffness == pytest.approx(50000.0)
    assert riv.fn_fail == pytest.approx(1200.0)
    assert riv.ft_fail == pytest.approx(800.0)

    assert 28 in model.prop_xelems
    xe = model.prop_xelems[28]
    assert xe.itip == 1
    assert xe.isurf == 2
    assert xe.alpha == pytest.approx(0.35)


def test_m143_admesh_controls(tmp_path):
    deck = """/BEGIN
Test Adaptive Meshing Controls
/ADMESH/GLOBAL
Global Adaptivity
3 2 0.001 1
/ADMESH/PART/10
Part Adaptivity Refinement
5 2 0.05 1.5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert model.admesh_global is not None
    assert model.admesh_global.level_max == 3
    assert model.admesh_global.iadm_rule == 2
    assert model.admesh_global.t_delay == pytest.approx(0.001)

    assert 10 in model.admesh_controls
    adm = model.admesh_controls[10]
    assert adm.subtype == "PART"
    assert adm.part_id == 5
    assert adm.crit_level == 2
    assert adm.h_min == pytest.approx(0.05)
    assert adm.h_max == pytest.approx(1.5)

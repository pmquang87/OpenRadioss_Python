"""Tests for Milestone M130: Extended Loads and Preloads Suite
(/LOAD/PCYL, /PLOAD/PCYL, /PRELOAD/AXIAL, /LOAD/LASER, /MERGE/NODE).
"""
from __future__ import annotations

from pathlib import Path
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


def test_m130_pcyl_load(tmp_path):
    deck = """/BEGIN
Test PCYL Load
/LOAD/PCYL/1
Cylindrical Pressure Load
10 2 3
100 1.5 2.0 50.0
/PLOAD/PCYL/2
Cylindrical Follower
20 0 0
200 1.0 1.0 100.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.pcyl_loads
    assert 2 in model.pcyl_loads

    load1 = model.pcyl_loads[1]
    assert load1.surf_id == 10
    assert load1.sens_id == 2
    assert load1.frame_id == 3
    assert load1.table_id == 100
    assert load1.xscale_r == pytest.approx(1.5)
    assert load1.xscale_t == pytest.approx(2.0)
    assert load1.yscale_p == pytest.approx(50.0)

    load2 = model.pcyl_loads[2]
    assert load2.surf_id == 20
    assert load2.table_id == 200
    assert load2.yscale_p == pytest.approx(100.0)


def test_m130_preload_axial_multicard(tmp_path):
    deck = """/BEGIN
Test Preload Axial Multi-card
/PRELOAD/AXIAL/1
Bolt Axial Preload
5 2 10
5000.0 0.05
/LOAD/PRELOAD_AXIAL/2
Spring Preload
6 0 11
2500.0 0.02
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.preload_axials
    assert 2 in model.preload_axials

    p1 = model.preload_axials[1]
    assert p1.grpart_id == 5
    assert p1.sens_id == 2
    assert p1.fct_id == 10
    assert p1.preload == pytest.approx(5000.0)
    assert p1.damp == pytest.approx(0.05)

    p2 = model.preload_axials[2]
    assert p2.grpart_id == 6
    assert p2.fct_id == 11
    assert p2.preload == pytest.approx(2500.0)
    assert p2.damp == pytest.approx(0.02)


def test_m130_preload_axial_singlecard(tmp_path):
    deck = """/BEGIN
Test Preload Axial Single-card
/PRELOAD/AXIAL/3
Single Card Preload
7 0 12 1000.0 0.1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 3 in model.preload_axials

    p3 = model.preload_axials[3]
    assert p3.grpart_id == 7
    assert p3.fct_id == 12
    assert p3.preload == pytest.approx(1000.0)
    assert p3.damp == pytest.approx(0.1)


def test_m130_laser_load(tmp_path):
    deck = """/BEGIN
Test Laser Load
/LOAD/LASER/1
Laser Beam Impact
1000.0 1 0.0 500.0 2
10.0 20.0 30.0 40.0 50.0
2 3
101 102
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.laser_loads

    las = model.laser_loads[1]
    assert las.magnitude == pytest.approx(1000.0)
    assert las.curve_id == 1
    assert las.hn == pytest.approx(10.0)
    assert las.vcp == pytest.approx(20.0)
    assert las.k0 == pytest.approx(30.0)
    assert las.rd == pytest.approx(40.0)
    assert las.ks == pytest.approx(50.0)
    assert las.np == 2
    assert las.nc == 3
    assert las.plasma_elements == [101, 102]


def test_m130_merge_node(tmp_path):
    deck = """/BEGIN
Test Node Merge Option
/MERGE/NODE/1
Merge Tolerance Options
0.05 10 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.node_merges

    m = model.node_merges[1]
    assert m.tol == pytest.approx(0.05)
    assert m.grnod_id == 10
    assert m.merge_type == 1

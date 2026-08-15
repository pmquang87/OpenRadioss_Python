"""Tests for Milestone M139: Multi-Dimensional Tabular Functions, Lagrange Multiplier Contacts & Eulerian Mass Distributions Suite
(/TABLE/2, /TABLE/3, /INTER/LAGMUL/SPOTWELD, /INTER/LAGMUL/SURF, /INTER/LAGMUL/PART, /INTER/LAGMUL/BEAM,
 /EULER/VOID, /ADMAS/TOTAL_PART, /ADMAS/TOTAL_SURF, /ADMAS/TOTAL_BOX).
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


def test_m139_table_2d_and_3d(tmp_path):
    deck = """/BEGIN
Test Tables 2D and 3D
/TABLE/2/10
2D Yield Surface vs Strain Rate
2
0.001
0.0 100.0
0.1 150.0
0.2 180.0
10.0
0.0 200.0
0.1 260.0
0.2 300.0
/TABLE/3/20
3D Multivariate Lookup
2
293.0
0.0 100.0
0.1 120.0
500.0
0.0 50.0
0.1 60.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 10 in model.tables
    t2 = model.tables[10]
    assert t2.id == 10
    assert t2.dim == 2
    assert len(t2.curves) == 2

    rate1, x1, z1 = t2.curves[0]
    assert rate1 == pytest.approx(0.001)
    assert np.allclose(x1, [0.0, 0.1, 0.2])
    assert np.allclose(z1, [100.0, 150.0, 180.0])

    rate2, x2, z2 = t2.curves[1]
    assert rate2 == pytest.approx(10.0)
    assert np.allclose(x2, [0.0, 0.1, 0.2])
    assert np.allclose(z2, [200.0, 260.0, 300.0])

    assert 20 in model.tables
    t3 = model.tables[20]
    assert t3.id == 20
    assert t3.dim == 3
    assert len(t3.curves) == 2


def test_m139_inter_lagmul_qualifiers(tmp_path):
    deck = """/BEGIN
Test Lagrange Multiplier Contacts
/INTER/LAGMUL/SPOTWELD/1
Spotweld Interface
100 200
0 1
/INTER/LAGMUL/SURF/2
Surface Lagrange Contact
10 20 0 0 0 1.5
0 0 0 0 0 0.5
/INTER/LAGMUL/PART/3
Part Lagrange Tied Interface
30 40 0 0.25
/INTER/LAGMUL/BEAM/4
Beam Edge Lagrange Interface
50 60
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.interfaces) == 4

    i1 = model.interfaces[0]
    assert i1.id == 1
    assert i1.type == 16
    assert i1.lagmul is True
    assert i1.grnod_id == 100
    assert i1.grbric_id1 == 200
    assert i1.itied == 1

    i2 = model.interfaces[1]
    assert i2.id == 2
    assert i2.type == 7
    assert i2.lagmul is True
    assert i2.grnod_id == 10
    assert i2.surf_id == 20
    assert i2.gap == pytest.approx(0.5)

    i3 = model.interfaces[2]
    assert i3.id == 3
    assert i3.type == 2
    assert i3.lagmul is True
    assert i3.grnod_id == 30
    assert i3.surf_id == 40
    assert i3.dsearch == pytest.approx(0.25)

    i4 = model.interfaces[3]
    assert i4.id == 4
    assert i4.type == 11
    assert i4.lagmul is True
    assert i4.line_id1 == 50
    assert i4.line_id2 == 60


def test_m139_euler_void_and_admas_total(tmp_path):
    deck = """/BEGIN
Test Euler Void and Admas Total
/EULER/VOID/1
Euler Void Phase
/ADMAS/TOTAL_PART/10
Total Distributed Mass on Part
500.0 100
/ADMAS/TOTAL_SURF/20
Total Distributed Mass on Surface
250.0 200
/ADMAS/TOTAL_BOX/30
Total Distributed Mass in Box
100.0 300
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.admas) == 3

    m1 = model.admas[0]
    assert m1.id == 10
    assert m1.mass == pytest.approx(500.0)
    assert m1.grnod_id == 100
    assert m1.mass_type == 1

    m2 = model.admas[1]
    assert m2.id == 20
    assert m2.mass == pytest.approx(250.0)
    assert m2.grnod_id == 200
    assert m2.mass_type == 2

    m3 = model.admas[2]
    assert m3.id == 30
    assert m3.mass == pytest.approx(100.0)
    assert m3.grnod_id == 300
    assert m3.mass_type == 3

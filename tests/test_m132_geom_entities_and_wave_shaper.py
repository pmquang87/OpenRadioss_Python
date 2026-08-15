"""Tests for Milestone M132: Advanced Geometric Entities & Detonation Shaping
(/BOX/BOX, /SURF/PLANE, /SURF/ELLIPSE, /DFS/WAVE_SHAPER, /SET/NODENS, /BOX aliases).
"""
from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np

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


def test_m132_box_box_combination(tmp_path):
    deck = """/BEGIN
Test BOX/BOX
/BOX/RECTA/1
Sub Box 1
0.0 0.0 0.0
10.0 10.0 10.0
/BOX/RECTA/2
Sub Box 2
20.0 20.0 20.0
30.0 30.0 30.0
/BOX/RECTA/3
Sub Box 3 (to subtract)
5.0 5.0 5.0
6.0 6.0 6.0
/BOX/BOX/100
Combined Box
2 1
1 2
3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 100 in model.boxes

    bbox = model.boxes[100]
    assert bbox.kind == "BOX"
    assert bbox.box_ids == [1, 2, -3]


def test_m132_box_aliases(tmp_path):
    deck = """/BEGIN
Test BOX aliases
/BOX/RECT/1
Rect Alias
0.0 0.0 0.0 10.0 10.0 10.0
/BOX/CYL/2
Cyl Alias
0 0 5.0
0.0 0.0 0.0
0.0 0.0 10.0
/BOX/SPH/3
Sph Alias
0 8.0
1.0 2.0 3.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    b1 = model.boxes[1]
    assert b1.kind == "RECTA"
    assert np.allclose(b1.corner_min, [0, 0, 0])
    assert np.allclose(b1.corner_max, [10, 10, 10])

    b2 = model.boxes[2]
    assert b2.kind == "CYLIN"
    assert b2.diameter == pytest.approx(5.0)
    assert np.allclose(b2.p1, [0, 0, 0])
    assert np.allclose(b2.p2, [0, 0, 10])

    b3 = model.boxes[3]
    assert b3.kind == "SPHER"
    assert b3.diameter == pytest.approx(8.0)
    assert np.allclose(b3.p1, [1, 2, 3])


def test_m132_surf_plane(tmp_path):
    deck = """/BEGIN
Test SURF PLANE
/SURF/PLANE/1
Planar Boundary
0.0 0.0 0.0
0.0 0.0 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.surfaces

    surf = model.surfaces[1]
    assert surf.plane_p1 is not None
    assert np.allclose(surf.plane_p1, [0, 0, 0])
    assert surf.plane_p2 is not None
    assert np.allclose(surf.plane_p2, [0, 0, 1])


def test_m132_surf_ellipse(tmp_path):
    deck = """/BEGIN
Test SURF ELLIPSE
/SURF/ELLIPSE/1
Ellipsoidal Surface
10 0
0.0 0.0 0.0
5.0 4.0 3.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.surfaces

    surf = model.surfaces[1]
    assert surf.ellipse_skew == 10
    assert surf.ellipse_center is not None
    assert np.allclose(surf.ellipse_center, [0, 0, 0])
    assert surf.ellipse_semiaxes is not None
    assert np.allclose(surf.ellipse_semiaxes, [5, 4, 3])


def test_m132_dfs_wave_shaper(tmp_path):
    deck = """/BEGIN
Test Wave Shaper
/DFS/WAVE_SHAPER/1
Wave Shaper Barrier 1
100 2 0.005 1.0e-5
/INIT/DET/WAVE_SHAPER/2
Wave Shaper Barrier 2
200 3 0.010 2.0e-5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.wave_shapers
    ws1 = model.wave_shapers[1]
    assert ws1.surf_id == 100
    assert ws1.mat_id == 2
    assert ws1.thick == pytest.approx(0.005)
    assert ws1.delay == pytest.approx(1.0e-5)

    assert 2 in model.wave_shapers
    ws2 = model.wave_shapers[2]
    assert ws2.surf_id == 200
    assert ws2.mat_id == 3
    assert ws2.thick == pytest.approx(0.010)
    assert ws2.delay == pytest.approx(2.0e-5)


def test_m132_set_nodens(tmp_path):
    deck = """/BEGIN
Test SET NODENS
/SET/NODENS/1
Node Set from Reference
10 20 30
/SET/NODE/NS/2
Node Set NS
40 50 60
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.node_groups
    assert model.node_groups[1].node_ids == [10, 20, 30]

    assert 2 in model.node_groups
    assert model.node_groups[2].node_ids == [40, 50, 60]

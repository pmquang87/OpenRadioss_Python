"""Tests for Milestone M137: Extended Initial Velocities, Detonation Fronts, State Mapping & Set Routing Suite
(/INIVEL/PART, /INIVEL/SPH, /DFS/DETLINE, /DFS/DETCIRC, /INIMAP/3D, /INIMAP3D, /SET/PART, /SET/MAT, /SET/PROP, /SET/SUB).
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


def test_m137_inivel_part_and_sph(tmp_path):
    deck = """/BEGIN
Test Inivel Part and SPH
/INIVEL/PART/1
Initial Part Velocity
10 10.0 20.0 30.0 5.0 1
0.001 2
/INIVEL/SPH/2
Initial SPH Velocity
100 1.0 2.0 3.0 0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.inivel_parts
    ip = model.inivel_parts[1]
    assert ip.id == 1
    assert ip.part_id == 10
    assert ip.vx == pytest.approx(10.0)
    assert ip.vy == pytest.approx(20.0)
    assert ip.vz == pytest.approx(30.0)
    assert ip.vr == pytest.approx(5.0)
    assert ip.skew_id == 1
    assert ip.tstart == pytest.approx(0.001)
    assert ip.sens_id == 2

    assert 2 in model.inivel_sphs
    is_ = model.inivel_sphs[2]
    assert is_.id == 2
    assert is_.grsph_id == 100
    assert is_.vx == pytest.approx(1.0)
    assert is_.vy == pytest.approx(2.0)
    assert is_.vz == pytest.approx(3.0)


def test_m137_dfs_detline_and_detcirc(tmp_path):
    deck = """/BEGIN
Test DFS Detonation Lines and Circles
/DFS/DETLINE/1
Detonation Line
0.0 0.0 0.0 10.0 0.0 0.0 0.001 7000.0
/DFS/DETCIRC/2
Detonation Ring
0.0 0.0 5.0 0.0 0.0 1.0 25.0 0.002 8000.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.det_lines
    dl = model.det_lines[1]
    assert dl.id == 1
    assert dl.p1 == (0.0, 0.0, 0.0)
    assert dl.p2 == (10.0, 0.0, 0.0)
    assert dl.t0 == pytest.approx(0.001)
    assert dl.dvel == pytest.approx(7000.0)

    assert 2 in model.det_circs
    dc = model.det_circs[2]
    assert dc.id == 2
    assert dc.center == (0.0, 0.0, 5.0)
    assert dc.axis == (0.0, 0.0, 1.0)
    assert dc.radius == pytest.approx(25.0)
    assert dc.t0 == pytest.approx(0.002)
    assert dc.dvel == pytest.approx(8000.0)


def test_m137_inimap3d(tmp_path):
    deck = """/BEGIN
Test INIMAP3D
/INIMAP/3D/1
3D solution mapping
1 100 200 300 0 1.5
res_3d.h5
/INIMAP3D/2
Direct 3D Map
2 101 201 301 0 1.0
map2.dat
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.ini_map3ds
    m1 = model.ini_map3ds[1]
    assert m1.id == 1
    assert m1.map_type == 1
    assert m1.grbric_id == 100
    assert m1.grquad_id == 200
    assert m1.grsh3n_id == 300
    assert m1.fscale_v == pytest.approx(1.5)
    assert m1.filename == "res_3d.h5"

    assert 2 in model.ini_map3ds
    m2 = model.ini_map3ds[2]
    assert m2.id == 2
    assert m2.map_type == 2
    assert m2.grbric_id == 101
    assert m2.grquad_id == 201
    assert m2.grsh3n_id == 301
    assert m2.filename == "map2.dat"


def test_m137_set_routing(tmp_path):
    deck = """/BEGIN
Test Sets Routing
/SET/PART/1
Part Set
10 20 30
/SET/MAT/2
Material Set
1 2 3
/SET/PROP/3
Property Set
100 200
/SET/SUB/4
Subset Collection
1 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert "PART" in model.egroups
    assert 1 in model.egroups["PART"]
    assert model.egroups["PART"][1].part_ids == [10, 20, 30]

    assert "MAT" in model.generic_sets
    assert 2 in model.generic_sets["MAT"]
    assert model.generic_sets["MAT"][2].ids == [1, 2, 3]

    assert "PROP" in model.generic_sets
    assert 3 in model.generic_sets["PROP"]
    assert model.generic_sets["PROP"][3].ids == [100, 200]

    assert "SUB" in model.generic_sets
    assert 4 in model.generic_sets["SUB"]
    assert model.generic_sets["SUB"][4].ids == [1, 2]

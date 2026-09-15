"""Tests for Milestone M163: Detonation Cord Ignition, Advanced Pressure Loading & ALE/Euler Directives Suite."""

from pathlib import Path
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import DfsDetcord, LoadPressure, AleMat, EulerMat


def _parse_deck(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_dfs_detcord_fixed_format(tmp_path: Path):
    """Test /DFS/DETCORD in fixed format."""
    deck_text = """/BEGIN
TEST_DFS_DETCORD_FIXED
      2022         0
/DFS/DETCORD/101
Detonation Cord 101
       102                0.05                7500         3         2
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 101 in model.dfs_detcords
    dc = model.dfs_detcords[101]
    assert dc.id == 101
    assert dc.grnd_id == 102
    assert abs(dc.t_det - 0.05) < 1e-6
    assert abs(dc.v_cj - 7500.0) < 1e-6
    assert dc.iopt == 3
    assert dc.mat_id == 2


def test_dfs_detcord_free_format(tmp_path: Path):
    """Test /DFS/DETCORD in free format."""
    deck_text = """/BEGIN
TEST_DFS_DETCORD_FREE
/DFS/DETCORD/102
Detonation Cord 102
201 0.002 6800.0 3 4
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 102 in model.dfs_detcords
    dc = model.dfs_detcords[102]
    assert dc.id == 102
    assert dc.grnd_id == 201
    assert abs(dc.t_det - 0.002) < 1e-6
    assert abs(dc.v_cj - 6800.0) < 1e-6
    assert dc.iopt == 3
    assert dc.mat_id == 4


def test_dfs_detcord_node_format(tmp_path: Path):
    """Test /DFS/DETCORD/NODE with explicit node numbers."""
    deck_text = """/BEGIN
TEST_DFS_DETCORD_NODE
/DFS/DETCORD/NODE/103
Detonation Cord Node List
10 20 30 40 50
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 103 in model.dfs_detcords
    dc = model.dfs_detcords[103]
    assert dc.id == 103
    assert dc.nodes == [10, 20, 30, 40, 50]


def test_load_pressure_fixed_format(tmp_path: Path):
    """Test /LOAD/PRESSURE in fixed format with interface cards."""
    c2 = f"{4:10d}{'':10s}{1.5:20.1f}{2.0:20.1f}"
    c3 = f"{7:10d}{'':10s}{0.25:20.2f}"
    deck_text = f"""/BEGIN
TEST_LOAD_PRESSURE_FIXED
      2022         0
/LOAD/PRESSURE/1
Hydroforming Pressure Load
        10         1         5         2         Z         3
{c2}
{c3}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 1 in model.load_pressures
    lp = model.load_pressures[1]
    assert lp.id == 1
    assert lp.surf_id == 10
    assert lp.iload == 1
    assert lp.sens_id == 5
    assert lp.inorm == 2
    assert lp.direction == "Z"
    assert lp.skew_id == 3
    assert lp.fct_id == 4
    assert abs(lp.xscale_p - 1.5) < 1e-6
    assert abs(lp.yscale_p - 2.0) < 1e-6
    assert lp.inter_ids == [7]
    assert abs(lp.gap_shifts[0] - 0.25) < 1e-6


def test_load_pressure_free_format(tmp_path: Path):
    """Test /LOAD/PRESSURE in free format."""
    deck_text = """/BEGIN
TEST_LOAD_PRESSURE_FREE
/LOAD/PRESSURE/2
Pressure Load 2
15 1 0 1 X 0
8 1.0 5.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 2 in model.load_pressures
    lp = model.load_pressures[2]
    assert lp.id == 2
    assert lp.surf_id == 15
    assert lp.inorm == 1
    assert lp.direction == "X"
    assert lp.fct_id == 8
    assert abs(lp.xscale_p - 1.0) < 1e-6
    assert abs(lp.yscale_p - 5.0) < 1e-6


def test_load_pressure_aliases(tmp_path: Path):
    """Test /PLOAD/PRESSURE and /PRESSURE aliases."""
    deck_text = """/BEGIN
TEST_LOAD_PRESSURE_ALIASES
/PRESSURE/3
Direct Pressure Keyword
25 1 2 1 Y 0
6 1.2 3.4
/PLOAD/PRESSURE/4
Pload Pressure Load
35 1 0 1 Z 0
9 1.0 2.5
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 3 in model.load_pressures
    assert 4 in model.load_pressures
    assert model.load_pressures[3].surf_id == 25
    assert model.load_pressures[4].surf_id == 35


def test_ale_mat_fixed_and_free(tmp_path: Path):
    """Test /ALE/MAT in fixed and free formats."""
    deck_text = """/BEGIN
TEST_ALE_MAT
      2022         0
/ALE/MAT/1
                 0.8
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 1 in model.ale_mats
    assert abs(model.ale_mats[1].ale_flrd - 0.8) < 1e-6


def test_euler_mat_fixed_and_free(tmp_path: Path):
    """Test /EULER/MAT in fixed and free formats."""
    deck_text = """/BEGIN
TEST_EULER_MAT
/EULER/MAT/2
0.65
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 2 in model.euler_mats
    assert abs(model.euler_mats[2].euler_flrd - 0.65) < 1e-6

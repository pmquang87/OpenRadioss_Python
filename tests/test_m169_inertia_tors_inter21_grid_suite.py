"""Tests for Milestone M169: Part Inertia Modifiers, Torsion Springs, Sub-Surface Contacts, Rigid Body Sensors & ALE Grid Directives Suite."""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_deck(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_inertia_part_fixed(tmp_path: Path):
    """Test /INERTIA/PART in fixed format."""
    c1 = f"{101:10d}{2:10d}{1:10d}"
    c2 = f"{50.0:20.4f}{10.0:20.4f}{20.0:20.4f}{30.0:20.4f}"
    c3 = f"{1000.0:20.4f}{2000.0:20.4f}{3000.0:20.4f}{50.0:20.4f}{60.0:20.4f}{70.0:20.4f}"
    deck_text = f"""/BEGIN
TEST_INERTIA_PART_FIXED
      2022         0
/INERTIA/PART/1
Inertia for Part 101
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors: {log.errors}"
    assert 1 in model.inertia_parts
    inp = model.inertia_parts[1]
    assert inp.part_id == 101
    assert inp.skew_id == 2
    assert inp.iflag == 1
    assert abs(inp.mass - 50.0) < 1e-4
    assert abs(inp.xg - 10.0) < 1e-4
    assert abs(inp.yg - 20.0) < 1e-4
    assert abs(inp.zg - 30.0) < 1e-4
    assert abs(inp.ixx - 1000.0) < 1e-3
    assert abs(inp.iyy - 2000.0) < 1e-3
    assert abs(inp.izz - 3000.0) < 1e-3
    assert abs(inp.ixy - 50.0) < 1e-4
    assert abs(inp.iyz - 60.0) < 1e-4
    assert abs(inp.izx - 70.0) < 1e-4


def test_inertia_part_free(tmp_path: Path):
    """Test /INERTIA/PART in free format."""
    deck_text = """/BEGIN
TEST_INERTIA_PART_FREE
/INERTIA/PART/2
Inertia Part Free
201 0 0
120.0 5.0 15.0 25.0
500.0 600.0 700.0 0.0 0.0 0.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors: {log.errors}"
    assert 2 in model.inertia_parts
    inp = model.inertia_parts[2]
    assert inp.part_id == 201
    assert abs(inp.mass - 120.0) < 1e-4
    assert abs(inp.xg - 5.0) < 1e-4


def test_prop_spr_tors_fixed(tmp_path: Path):
    """Test /PROP/SPR_TORS (/PROP/TYPE19) in fixed format."""
    c1 = f"{0.05:20.4f}{50000.0:20.4f}{150.0:20.4f}"
    c2 = f"{11:10d}{12:10d}{1.0:20.4f}{1.0:20.4f}"
    deck_text = f"""/BEGIN
TEST_PROP_SPR_TORS_FIXED
      2022         0
/PROP/SPR_TORS/19
Torsion Spring Property
{c1}
{c2}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors: {log.errors}"
    assert 19 in model.properties
    p = model.properties[19]
    assert p.type == 19
    assert abs(p.params["mass"] - 0.05) < 1e-6
    assert abs(p.params["k_tors"] - 50000.0) < 1e-3
    assert abs(p.params["c_tors"] - 150.0) < 1e-3
    assert p.params["fct_id_k"] == 11
    assert p.params["fct_id_c"] == 12


def test_prop_type19_free(tmp_path: Path):
    """Test /PROP/TYPE19 in free format."""
    deck_text = """/BEGIN
TEST_PROP_TYPE19_FREE
/PROP/TYPE19/20
Torsion Spring Free
0.01 25000.0 80.0
0 0 1.0 1.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors: {log.errors}"
    assert 20 in model.properties
    p = model.properties[20]
    assert p.type == 19
    assert abs(p.params["k_tors"] - 25000.0) < 1e-3


def test_inter_sub_surf(tmp_path: Path):
    """Test /INTER/SUB_SURF (routed as TYPE21)."""
    deck_text = """/BEGIN
TEST_INTER_SUB_SURF
/INTER/SUB_SURF/21
Sub Surface Contact
10 20 2 0 1 0
1.0 0.5 2.0
0.0 0.0 1.2 0.25
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors: {log.errors}"
    assert len(model.interfaces) == 1
    inter = model.interfaces[0]
    assert inter.type == 21
    assert inter.surf_id == 10
    assert inter.surf_id1 == 20
    assert abs(inter.stfac - 1.2) < 1e-4
    assert abs(inter.fric - 0.25) < 1e-4


def test_sensor_rbody_2card(tmp_path: Path):
    """Test /SENSOR/RBODY with 2-card format (card 1: node_id, rbody_id, itype, idir; card 2: fmin, fmax, tmin)."""
    c1 = f"{501:10d}{10:10d}{1:10d}{'FX':>10s}"
    c2 = f"{-1000.0:20.4f}{1000.0:20.4f}{1.0e-3:20.4f}"
    deck_text = f"""/BEGIN
TEST_SENSOR_RBODY_2CARD
      2022         0
/SENSOR/RBODY/5
Rigid Body Kinematics Sensor
                 0.0
{c1}
{c2}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors: {log.errors}"
    matching = [s for s in model.sensors if s.id == 5]
    assert len(matching) == 1
    s = matching[0]
    assert s.kind == "RBODY"
    assert s.node_id == 501
    assert s.rbody_id == 10
    assert s.dir == "FX"
    assert abs(s.fmin - (-1000.0)) < 1e-3
    assert abs(s.fmax - 1000.0) < 1e-3
    assert abs(s.tmin - 1.0e-3) < 1e-6


def test_ale_grid_disp_and_vel(tmp_path: Path):
    """Test /ALE/GRID/DISP and /ALE/GRID/VEL nodal boundary constraints."""
    c1 = f"{101:10d}{5:10d}{0:10d}{'111000':>10s}"
    c2 = f"{2.5:20.4f}{0.0:20.4f}{5.0e-3:20.4f}"
    v1 = f"{201:10d}{6:10d}{0:10d}{'111000':>10s}"
    v2 = f"{1.5:20.4f}{0.0:20.4f}{1.0e-2:20.4f}"
    deck_text = f"""/BEGIN
TEST_ALE_GRID_DISP_AND_VEL
      2022         0
/ALE/GRID/DISP/1
{c1}
{c2}
/ALE/GRID/VEL/2
{v1}
{v2}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors: {log.errors}"
    assert 1 in model.ale_grid_constraints
    assert 2 in model.ale_grid_constraints
    g1 = model.ale_grid_constraints[1]
    assert g1.kind == "DISP"
    assert g1.grnod_id == 101
    assert g1.fun_id == 5
    assert g1.tra_code == "111000"
    assert abs(g1.scale - 2.5) < 1e-4
    assert abs(g1.tstop - 5.0e-3) < 1e-6

    g2 = model.ale_grid_constraints[2]
    assert g2.kind == "VEL"
    assert g2.grnod_id == 201
    assert g2.fun_id == 6
    assert abs(g2.scale - 1.5) < 1e-4

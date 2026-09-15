"""Tests for Milestone M165: Sub-Laminate Ply Stacks, Neck Injury / Thermal Sensor Enhancements & Composite Draping Plies Suite."""

from pathlib import Path
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import SubLaminate, SubLaminatePly, Sensor


def _parse_deck(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_sub_laminate_fixed_format(tmp_path: Path):
    """Test /SUBLAMINATE in fixed format."""
    c1 = f"SUB       {101:10d}{2:10d}"
    c_ply1 = f"{11:10d}{45.0:20.4f}{0.125:20.4f}{0.05:20.4f}{0.95:20.4f}"
    c_ply2 = f"{12:10d}{-45.0:20.4f}{0.250:20.4f}{0.05:20.4f}{0.95:20.4f}"
    deck_text = f"""/BEGIN
TEST_SUBLAMINATE_FIXED
      2022         0
/SUBLAMINATE/101
Quasi-isotropic sublaminate 101
{c1}
Quasi-isotropic sublaminate title
{c_ply1}
{c_ply2}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 101 in model.sub_laminates
    sub = model.sub_laminates[101]
    assert sub.id == 101
    assert len(sub.plies) == 2
    assert sub.plies[0].ply_id == 11
    assert abs(sub.plies[0].phi - 45.0) < 1e-6
    assert abs(sub.plies[0].zi - 0.125) < 1e-6
    assert abs(sub.plies[0].p_thick_fail - 0.05) < 1e-6
    assert abs(sub.plies[0].f_weight - 0.95) < 1e-6

    assert sub.plies[1].ply_id == 12
    assert abs(sub.plies[1].phi - (-45.0)) < 1e-6
    assert abs(sub.plies[1].zi - 0.250) < 1e-6


def test_sub_laminate_free_format(tmp_path: Path):
    """Test /SUBLAMINATE in free format."""
    deck_text = """/BEGIN
TEST_SUBLAMINATE_FREE
/SUBLAMINATE/102
Sublaminate 102 Free
SUB 102 2
Sublaminate 102 Title
21 0.0 0.2 0.1 1.0
22 90.0 0.4 0.1 1.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 102 in model.sub_laminates
    sub = model.sub_laminates[102]
    assert sub.id == 102
    assert len(sub.plies) == 2
    assert sub.plies[0].ply_id == 21
    assert abs(sub.plies[0].phi - 0.0) < 1e-6
    assert sub.plies[1].ply_id == 22
    assert abs(sub.plies[1].phi - 90.0) < 1e-6


def test_stack_with_embedded_sublaminate(tmp_path: Path):
    """Test /STACK containing embedded SUB definitions."""
    c1 = f"{1:10d}{1:10d}{1:10d}{1:10d}          {0.0:20.4f}"
    c2 = f"{0.01:20.4f}{0.01:20.4f}{0.01:20.4f}{0.0:20.4f}{0.0:20.4f}"
    c3 = f"          {1:10d}{0.833333:20.4f}          {0:10d}          {0:10d}"
    c4 = f"{1.0:20.4f}{0.0:20.4f}{0.0:20.4f}{0:10d}{0:10d}{0:10d}{0:10d}"
    c_sub = f"SUB       {201:10d}{2:10d}"
    c_p1 = f"{31:10d}{0.0:20.4f}{0.1:20.4f}{0.0:20.4f}{1.0:20.4f}"
    c_p2 = f"{32:10d}{45.0:20.4f}{0.2:20.4f}{0.0:20.4f}{1.0:20.4f}"
    deck_text = f"""/BEGIN
TEST_STACK_EMBEDDED_SUB
      2022         0
/STACK/501
Composite Stack with SubLaminate
{c1}
{c2}
{c3}
{c4}
{c_sub}
{c_p1}
{c_p2}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 501 in model.stacks
    st = model.stacks[501]
    assert len(st.plies) == 2
    assert st.plies[0].ply_id == 31
    assert st.plies[1].ply_id == 32

    assert 201 in model.sub_laminates
    sub = model.sub_laminates[201]
    assert len(sub.plies) == 2
    assert sub.plies[0].ply_id == 31
    assert sub.plies[1].ply_id == 32


def test_sensor_nic_fixed_format(tmp_path: Path):
    """Test /SENSOR/NIC in 4-card fixed format."""
    c1 = f"{0.005:20.4f}"
    c2 = f"{1.0:20.4f}{1500.0:20.4f}{2000.0:20.4f}{300.0:20.4f}{250.0:20.4f}"
    c3 = f"{12:10d}{3:10d}         X         Y"
    c4 = f"{0.001:20.4f}{2.0775:20.4f}{600.0:20.4f}"
    deck_text = f"""/BEGIN
TEST_SENSOR_NIC_FIXED
      2022         0
/SENSOR/NIC/301
Neck Injury Criterion Sensor 301
{c1}
{c2}
{c3}
{c4}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert len(model.sensors) == 1
    s = model.sensors[0]
    assert s.id == 301
    assert s.kind == "NIC"
    assert abs(s.tdelay - 0.005) < 1e-6
    assert abs(s.nij_max - 1.0) < 1e-6
    assert abs(s.fint_tens - 1500.0) < 1e-6
    assert abs(s.fint_comp - 2000.0) < 1e-6
    assert abs(s.mint_flex - 300.0) < 1e-6
    assert abs(s.mint_ext - 250.0) < 1e-6
    assert s.spring_id == 12
    assert s.skew_id == 3
    assert s.ax_dir == "X"
    assert s.bend_dir == "Y"
    assert abs(s.tmin - 0.001) < 1e-6
    assert abs(s.alpha - 2.0775) < 1e-6
    assert abs(s.cfc - 600.0) < 1e-6


def test_sensor_nic_free_format(tmp_path: Path):
    """Test /SENSOR/NIC in free format."""
    deck_text = """/BEGIN
TEST_SENSOR_NIC_FREE
/SENSOR/NIC/302
Neck Injury Sensor Free
0.002
1.2 1800.0 2200.0 350.0 280.0
15 4 Z X
0.0015 2.0 1000.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert len(model.sensors) == 1
    s = model.sensors[0]
    assert s.id == 302
    assert s.kind == "NIC"
    assert abs(s.tdelay - 0.002) < 1e-6
    assert abs(s.nij_max - 1.2) < 1e-6
    assert s.spring_id == 15
    assert s.skew_id == 4
    assert s.ax_dir == "Z"
    assert s.bend_dir == "X"


def test_sensor_temp_fixed_format(tmp_path: Path):
    """Test /SENSOR/TEMP in fixed format with blank spacer."""
    c1 = f"{0.01:20.4f}"
    c2 = f"{50:10d}          {1200.0:20.4f}{250.0:20.4f}{600.0:20.4f}{0.005:20.4f}"
    deck_text = f"""/BEGIN
TEST_SENSOR_TEMP_FIXED
      2022         0
/SENSOR/TEMP/401
Thermal Node Group Sensor 401
{c1}
{c2}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert len(model.sensors) == 1
    s = model.sensors[0]
    assert s.id == 401
    assert s.kind == "TEMP"
    assert abs(s.tdelay - 0.01) < 1e-6
    assert s.grnod_id == 50
    assert abs(s.tempmax - 1200.0) < 1e-6
    assert abs(s.tempmin - 250.0) < 1e-6
    assert abs(s.tempmean - 600.0) < 1e-6
    assert abs(s.tmin - 0.005) < 1e-6


def test_sensor_dist_with_dflag(tmp_path: Path):
    """Test /SENSOR/DIST with deactivation flag dflag."""
    c1 = f"{0.002:20.4f}"
    c2 = f"{101:10d}{102:10d}{0.05:20.4f}{0.50:20.4f}{0.001:20.4f}{1:10d}"
    deck_text = f"""/BEGIN
TEST_SENSOR_DIST_DFLAG
      2022         0
/SENSOR/DIST/601
Distance Sensor with Dflag
{c1}
{c2}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert len(model.sensors) == 1
    s = model.sensors[0]
    assert s.id == 601
    assert s.kind == "DIST"
    assert s.node_id1 == 101
    assert s.node_id2 == 102
    assert abs(s.dmin - 0.05) < 1e-6
    assert abs(s.dmax - 0.50) < 1e-6
    assert abs(s.tmin - 0.001) < 1e-6
    assert s.dflag == 1


def test_drape_ply_slice_table(tmp_path: Path):
    """Test /DRAPE/PLY_SLICE and /TABLE/DRAPE/PLY_SLICE."""
    deck_text = """/BEGIN
TEST_DRAPE_PLY_SLICE
/DRAPE/PLY_SLICE/701
Drape Ply Slice 701
SHELL 1001 0.95 15.0 2 4
SH3N 2001 0.90 -10.0 3 3
GRSHEL 3001 0.85 45.0 4 2
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 701 in model.drapes
    dr = model.drapes[701]
    assert dr.id == 701
    assert len(dr.slices) == 3
    assert dr.slices[0]["entity_type"] == "SHELL"
    assert dr.slices[0]["entity_id"] == 1001
    assert abs(dr.slices[0]["thinning"] - 0.95) < 1e-6
    assert abs(dr.slices[0]["theta_slice"] - 15.0) < 1e-6
    assert dr.slices[0]["mat_id"] == 2
    assert dr.slices[0]["npt_slice"] == 4

"""Tests for Milestone M167: External Coupling Links, Multi-Fluid Mapping Formulations & User-Defined Element Properties Suite."""

from pathlib import Path
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


def test_extern_link_fixed(tmp_path: Path):
    """Test /EXTERN/LINK in fixed format."""
    c1 = f"{101:10d}"
    deck_text = f"""/BEGIN
TEST_EXTERN_LINK_FIXED
      2022         0
/EXTERN/LINK/1
External Link 1
{c1}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 1 in model.ext_links
    link = model.ext_links[1]
    assert link.id == 1
    assert link.title == "External Link 1"
    assert link.grnod_id == 101


def test_extlnk_free(tmp_path: Path):
    """Test /EXTLNK in free format."""
    deck_text = """/BEGIN
TEST_EXTLNK_FREE
/EXTLNK/2
External Link Free
202
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 2 in model.ext_links
    link = model.ext_links[2]
    assert link.id == 2
    assert link.title == "External Link Free"
    assert link.grnod_id == 202


def test_inimap1d_vp_fixed(tmp_path: Path):
    """Test /INIMAP1D/VP in multi-card fixed format."""
    c2 = f"{3:10d}{1001:10d}{0:10d}{501:10d}{0:10d}{0:10d}{1.0:20.4f}"
    c3 = f"{11:10d}{1.25:20.4f}"
    c4 = f"{2:10d}"
    c5 = f"{101:10d}{102:10d}{1.0:20.4f}{103:10d}{1.5:20.4f}"
    c6 = f"{201:10d}{202:10d}{1.1:20.4f}{203:10d}{2.0:20.4f}"
    deck_text = f"""/BEGIN
TEST_INIMAP1D_VP_FIXED
      2022         0
/INIMAP1D/VP/10
Mapped Field 1D VP
{c2}
{c3}
{c4}
{c5}
{c6}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 10 in model.ini_map1ds
    m = model.ini_map1ds[10]
    assert m.id == 10
    assert m.formulation == "VP"
    assert m.map_type == 3
    assert m.node_id1 == 1001
    assert m.grbric_id == 501
    assert m.func_vel == 11
    assert abs(m.fac_vel - 1.25) < 1e-6
    assert m.nb_mat == 2
    assert m.func_alpha == [101, 201]
    assert m.func_rho == [102, 202]
    assert m.fac_rho == [1.0, 1.1]
    assert m.func_pres_ener == [103, 203]
    assert m.fac_pres_ener == [1.5, 2.0]


def test_inimap1d_ve_free(tmp_path: Path):
    """Test /INIMAP/1D/VE in free format."""
    deck_text = """/BEGIN
TEST_INIMAP1D_VE_FREE
/INIMAP/1D/VE/20
Mapped Field 1D VE Free
1 101 102 301 0 0 1.0
15 2.0
1
301 302 1.0 303 1.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 20 in model.ini_map1ds
    m = model.ini_map1ds[20]
    assert m.id == 20
    assert m.formulation == "VE"
    assert m.map_type == 1
    assert m.node_id1 == 101
    assert m.node_id2 == 102
    assert m.grbric_id == 301
    assert m.func_vel == 15
    assert abs(m.fac_vel - 2.0) < 1e-6
    assert m.nb_mat == 1
    assert m.func_alpha == [301]
    assert m.func_rho == [302]
    assert m.func_pres_ener == [303]


def test_inimap1d_file(tmp_path: Path):
    """Test /INIMAP1D/FILE with external file."""
    deck_text = """/BEGIN
TEST_INIMAP1D_FILE
/INIMAP1D/FILE/30
Mapped Field File
3 1001 0 501 0 0 1.0
mapped_data_1d.dat
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 30 in model.ini_map1ds
    m = model.ini_map1ds[30]
    assert m.formulation == "FILE"
    assert m.filename == "mapped_data_1d.dat"


def test_inimap2d_vp_fixed(tmp_path: Path):
    """Test /INIMAP2D/VP in multi-card fixed format."""
    c2 = f"{1:10d}{101:10d}{102:10d}{103:10d}{601:10d}{1.0:20.4f}"
    c3 = f"{25:10d}{1.5:20.4f}"
    c4 = f"{1:10d}"
    c5 = f"{401:10d}{402:10d}{1.0:20.4f}{403:10d}{1.2:20.4f}"
    deck_text = f"""/BEGIN
TEST_INIMAP2D_VP_FIXED
      2022         0
/INIMAP2D/VP/40
Mapped Field 2D VP
{c2}
{c3}
{c4}
{c5}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 40 in model.ini_map2ds
    m = model.ini_map2ds[40]
    assert m.id == 40
    assert m.formulation == "VP"
    assert m.map_type == 1
    assert m.node_id1 == 101
    assert m.node_id2 == 102
    assert m.node_id3 == 103
    assert m.grbric_id == 601
    assert m.func_vel == 25
    assert abs(m.fac_vel - 1.5) < 1e-6
    assert m.nb_mat == 1
    assert m.func_alpha == [401]
    assert m.func_rho == [402]
    assert m.func_pres_ener == [403]


def test_prop_user_spring_fixed(tmp_path: Path):
    """Test /PROP/USER_SPRING in fixed format."""
    c1 = f"{12:10d}{250000.0:20.4f}{3:10d}"
    deck_text = f"""/BEGIN
TEST_PROP_USER_SPRING
      2022         0
/PROP/USER_SPRING/101
User Spring Property
{c1}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 101 in model.properties
    p = model.properties[101]
    assert p.type == 4
    assert p.params["nuvar"] == 12
    assert abs(p.params["stif_inter"] - 250000.0) < 1e-3
    assert p.params["skew_id"] == 3


def test_prop_user_solid_free(tmp_path: Path):
    """Test /PROP/USER_SOLID in free format."""
    deck_text = """/BEGIN
TEST_PROP_USER_SOLID
/PROP/USER_SOLID/201
User Solid Property
24 5
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 201 in model.properties
    p = model.properties[201]
    assert p.type == 34
    assert p.params["nuvar"] == 24
    assert p.params["skew_id"] == 5

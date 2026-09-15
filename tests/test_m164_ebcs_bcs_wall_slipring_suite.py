"""Tests for Milestone M164: Eulerian & Multifluid Boundary Conditions, Sliding Boundary Walls & Extended Seatbelt Sliprings Suite."""

from pathlib import Path
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import EbcsNrf, BcsWall, SlipringShell, EbcsPres, EbcsVel, EbcsInlet


def _parse_deck(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_ebcs_nrf_fixed_format(tmp_path: Path):
    """Test /EBCS/NRF in fixed format with relaxation times."""
    c2 = f"{0.05:20.4f}{0.12:20.4f}"
    deck_text = f"""/BEGIN
TEST_EBCS_NRF_FIXED
      2022         0
/EBCS/NRF/101
Non-reflecting frontier 101
        10
{c2}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 101 in model.ebcs_nrfs
    nrf = model.ebcs_nrfs[101]
    assert nrf.id == 101
    assert nrf.surf_id == 10
    assert abs(nrf.tcar_p - 0.05) < 1e-6
    assert abs(nrf.tcar_vf - 0.12) < 1e-6


def test_ebcs_nrf_free_format(tmp_path: Path):
    """Test /EBCS/NRF in free format."""
    deck_text = """/BEGIN
TEST_EBCS_NRF_FREE
/EBCS/NRF/102
Non-reflecting frontier 102
20
0.002 0.005
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 102 in model.ebcs_nrfs
    nrf = model.ebcs_nrfs[102]
    assert nrf.id == 102
    assert nrf.surf_id == 20
    assert abs(nrf.tcar_p - 0.002) < 1e-6
    assert abs(nrf.tcar_vf - 0.005) < 1e-6


def test_bcs_nrf_multi_card(tmp_path: Path):
    """Test /BCS/NRF with relaxation time parameters."""
    c2 = f"{0.01:20.4f}{0.02:20.4f}"
    deck_text = f"""/BEGIN
TEST_BCS_NRF_MULTI_CARD
      2022         0
/BCS/NRF/103
BCS Non-reflecting frontier
        30
{c2}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 103 in model.ebcs_nrfs
    nrf = model.ebcs_nrfs[103]
    assert nrf.id == 103
    assert nrf.surf_id == 30
    assert abs(nrf.tcar_p - 0.01) < 1e-6
    assert abs(nrf.tcar_vf - 0.02) < 1e-6


def test_bcs_wall_fixed_format(tmp_path: Path):
    """Test /BCS/WALL in fixed format."""
    c2 = f"{0.001:20.4f}{10.0:20.4f}"
    deck_text = f"""/BEGIN
TEST_BCS_WALL_FIXED
      2022         0
/BCS/WALL/201
Sliding Boundary Wall 201
        15         3
{c2}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 201 in model.bcs_walls
    wall = model.bcs_walls[201]
    assert wall.id == 201
    assert wall.set_id == 15
    assert wall.sensor_id == 3
    assert abs(wall.tstart - 0.001) < 1e-6
    assert abs(wall.tstop - 10.0) < 1e-6


def test_bcs_wall_free_format(tmp_path: Path):
    """Test /BCS/WALL in free format."""
    deck_text = """/BEGIN
TEST_BCS_WALL_FREE
/BCS/WALL/202
Sliding Wall 202
25 0
0.0 5.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 202 in model.bcs_walls
    wall = model.bcs_walls[202]
    assert wall.id == 202
    assert wall.set_id == 25
    assert wall.sensor_id == 0
    assert abs(wall.tstart - 0.0) < 1e-6
    assert abs(wall.tstop - 5.0) < 1e-6


def test_slipring_shell_fixed_format(tmp_path: Path):
    """Test /SLIPRING/SHELL in fixed format."""
    c1 = f"{10:10d}{20:10d}{30:10d}{1:10d}{2:10d}{0.5:20.4f}{15.0:20.4f}"
    c2 = f"{101:10d}{102:10d}{0.35:20.4f}{1.0:20.4f}{1.5:20.4f}{2.0:20.4f}"
    c3 = f"{103:10d}{104:10d}{0.45:20.4f}{1.0:20.4f}{1.2:20.4f}{1.8:20.4f}"
    deck_text = f"""/BEGIN
TEST_SLIPRING_SHELL_FIXED
      2022         0
/SLIPRING/SHELL/301
2D Shell Slipring 301
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 301 in model.slipring_shells
    sr = model.slipring_shells[301]
    assert sr.id == 301
    assert sr.el_set1 == 10
    assert sr.el_set2 == 20
    assert sr.node_set == 30
    assert sr.sens_id == 1
    assert sr.flow_flag == 2
    assert abs(sr.a - 0.5) < 1e-6
    assert abs(sr.ed_factor - 15.0) < 1e-6
    assert sr.fct_id1 == 101
    assert sr.fct_id2 == 102
    assert abs(sr.fricd - 0.35) < 1e-6
    assert abs(sr.fric_d - 0.35) < 1e-6
    assert abs(sr.yscale2 - 1.5) < 1e-6
    assert abs(sr.xscale2 - 2.0) < 1e-6
    assert sr.fct_id3 == 103
    assert sr.fct_id4 == 104
    assert abs(sr.frics - 0.45) < 1e-6
    assert abs(sr.fric_s - 0.45) < 1e-6
    assert abs(sr.yscale4 - 1.2) < 1e-6
    assert abs(sr.xscale4 - 1.8) < 1e-6


def test_slipring_shell_free_format(tmp_path: Path):
    """Test /SLIPRING/SHELL in free format."""
    deck_text = """/BEGIN
TEST_SLIPRING_SHELL_FREE
/SLIPRING/SHELL/302
2D Shell Slipring Free
11 21 31 0 1 0.25 10.0
201 202 0.28 1.0 1.0 1.0
203 204 0.38 1.0 1.0 1.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 302 in model.slipring_shells
    sr = model.slipring_shells[302]
    assert sr.id == 302
    assert sr.el_set1 == 11
    assert sr.el_set2 == 21
    assert sr.node_set == 31
    assert abs(sr.a - 0.25) < 1e-6
    assert abs(sr.ed_factor - 10.0) < 1e-6
    assert abs(sr.fricd - 0.28) < 1e-6
    assert abs(sr.frics - 0.38) < 1e-6


def test_ebcs_pres_and_vel_and_inlet(tmp_path: Path):
    """Test Eulerian /EBCS/PRES, /EBCS/VEL, and /EBCS/INLET boundary conditions."""
    deck_text = """/BEGIN
TEST_EBCS_PRES_VEL_INLET
/EBCS/PRES/401
Eulerian Pressure BC
10 0.0 5 1.0e5 0 1.0 0 1.0 0.0 0.0 0.0
/EBCS/VEL/402
Eulerian Velocity BC
20 0.0 6 50.0 0 0.0 0 0.0 0 1.0 0 1.0 0.0 0.0 0.0
/EBCS/INLET/403
Eulerian Inlet BC
30 1.2 50.0 0.0 0.0 2.5e5 7
/END
"""
    model, log = _parse_deck(tmp_path, deck_text)
    assert not log.errors, f"Errors in parsing: {log.errors}"
    assert 401 in model.ebcs_pres
    pbc = model.ebcs_pres[401]
    assert pbc.surf_id == 10
    assert pbc.fct_pres == 5
    assert abs(pbc.scale_pres - 1.0e5) < 1e-6

    assert 402 in model.ebcs_vel
    vbc = model.ebcs_vel[402]
    assert vbc.surf_id == 20
    assert vbc.fct_vx == 6
    assert abs(vbc.scale_vx - 50.0) < 1e-6

    assert 403 in model.ebcs_inlets
    ibc = model.ebcs_inlets[403]
    assert ibc.surf_id == 30
    assert abs(ibc.density - 1.2) < 1e-6
    assert abs(ibc.vx - 50.0) < 1e-6
    assert ibc.fct_id == 7

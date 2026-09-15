# pyradioss - Python port of OpenRadioss explicit FEM solver
# Milestone M178: Cyclic Symmetry, Cylindrical Pressure, Monitored Volume Boundary, Table Initial States & Work/Dist-Surf Sensors Suite
from pathlib import Path
import numpy as np
import pytest
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.common.messages import MessageLog
from pyradioss.model.model import Model


def _parse_deck(tmp_path: Path, deck_text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck_text, encoding="ascii")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(p))
    parse_starter_deck(blocks, model, log)
    return model, log


def test_bcs_cyclic_fixed_and_free(tmp_path: Path):
    """Test /BCS/CYCLIC and /CYCLIC in fixed and free formats."""
    # Fixed format
    fixed_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Fixed BCS/CYCLIC
/BCS/CYCLIC/101
Cyclic Symmetry Boundary Fixed
#  skew_ID  grnd_ID1  grnd_ID2
         5        21        22
/END
"""
    model, log = _parse_deck(tmp_path, fixed_deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 101 in model.bcs_cyclics
    assert 101 in model.cyclic_bcs
    bc1 = model.bcs_cyclics[101]
    assert bc1.skew_id == 5
    assert bc1.grnd_id1 == 21
    assert bc1.grnd_id2 == 22
    assert bc1.title == "Cyclic Symmetry Boundary Fixed"

    # Free format /CYCLIC
    free_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Free CYCLIC
/CYCLIC/102
Cyclic Symmetry Boundary Free
6 31 32
/END
"""
    model, log = _parse_deck(tmp_path, free_deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 102 in model.bcs_cyclics
    bc2 = model.bcs_cyclics[102]
    assert bc2.skew_id == 6
    assert bc2.grnd_id1 == 31
    assert bc2.grnd_id2 == 32


def test_load_pcyl_fixed_and_free(tmp_path: Path):
    """Test /LOAD/PCYL and /PCYL in fixed and free formats."""
    # Fixed format
    fixed_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Fixed LOAD/PCYL
/LOAD/PCYL/201
Cylindrical Pressure Load Fixed
#  surf_ID sensor_ID  frame_ID
        15         3         4
# tableIDT                      Ascale_r            Ascale_t            Fscale_y
        99                          1.25                0.85               500.0
/END
"""
    model, log = _parse_deck(tmp_path, fixed_deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 201 in model.pcyl_loads
    load1 = model.pcyl_loads[201]
    assert load1.surf_id == 15
    assert load1.sens_id == 3
    assert load1.frame_id == 4
    assert load1.table_id == 99
    assert load1.xscale_r == pytest.approx(1.25)
    assert load1.xscale_t == pytest.approx(0.85)
    assert load1.yscale_p == pytest.approx(500.0)

    # Free format /PCYL
    free_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Free PCYL
/PCYL/202
Cylindrical Pressure Load Free
16 5 6
105 1.10 0.90 600.0
/END
"""
    model, log = _parse_deck(tmp_path, free_deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 202 in model.pcyl_loads
    load2 = model.pcyl_loads[202]
    assert load2.surf_id == 16
    assert load2.sens_id == 5
    assert load2.frame_id == 6
    assert load2.table_id == 105
    assert load2.xscale_r == pytest.approx(1.10)
    assert load2.xscale_t == pytest.approx(0.90)
    assert load2.yscale_p == pytest.approx(600.0)


def test_ebcs_monvol_fixed_and_free(tmp_path: Path):
    """Test /EBCS/MONVOL and /MONVOL in fixed and free formats."""
    # Fixed format
    fixed_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Fixed EBCS/MONVOL
/EBCS/MONVOL/301
Monitored Volume Connection Fixed
#  surf_ID   sens_ID monvol_ID              Fscale
        50         2         8                1.05
/END
"""
    model, log = _parse_deck(tmp_path, fixed_deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 301 in model.ebcs_monvols
    ebcs1 = model.ebcs_monvols[301]
    assert ebcs1.surf_id == 50
    assert ebcs1.sens_id == 2
    assert ebcs1.monvol_id == 8
    assert ebcs1.fscale == pytest.approx(1.05)

    # Free format /MONVOL
    free_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Free MONVOL
/MONVOL/302
Monitored Volume Connection Free
55 3 9 1.15
/END
"""
    model, log = _parse_deck(tmp_path, free_deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 302 in model.ebcs_monvols
    ebcs2 = model.ebcs_monvols[302]
    assert ebcs2.surf_id == 55
    assert ebcs2.sens_id == 3
    assert ebcs2.monvol_id == 9
    assert ebcs2.fscale == pytest.approx(1.15)


def test_table_initial_states(tmp_path: Path):
    """Test /TABLE/INIBRI, /TABLE/INISHE, and /TABLE/INISH3 routing."""
    # /TABLE/INIBRI/STRA_FGLO and /TABLE/INIBRI/STRS_FGLO
    deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Table Initial States
/TABLE/INIBRI/STRA_FGLO
Initial Brick Strains Table
#  elem_ID               eps11               eps22               eps33
       101               0.001               0.002               0.003
#                eps12               eps23               eps31
                0.0004              0.0005              0.0006
/TABLE/INISHE/EPSP
Initial Shell Plastic Strain Table
       201                0.05
/TABLE/INISH3/DENS
Initial Triangle Shell Density Table
       301             7.8e-09
/END
"""
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 101 in model.ini_bricks
    br = model.ini_bricks[101]
    assert np.allclose(br.eps, [0.001, 0.002, 0.003, 0.0004, 0.0005, 0.0006])
    assert 201 in model.ini_shells
    sh = model.ini_shells[201]
    assert sh.epsp == pytest.approx(0.05)
    assert 301 in model.ini_shells
    sh3 = model.ini_shells[301]
    assert sh3.rho == pytest.approx(7.8e-09)


def test_sensor_work_fixed_and_free(tmp_path: Path):
    """Test /SENSOR/WORK and /SENSOR/TYPE13 in fixed and free formats."""
    # Fixed format /SENSOR/WORK
    fixed_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Fixed Sensor Work
/SENSOR/WORK/401
Sensor Work Fixed
#             Tdelay
                0.01
# node_ID1  node_ID2                Wmax                Tmin
       100       101              5000.0              0.0005
#  sect_ID    int_ID  rbody_ID  rwall_ID
        10        11        12        13
/END
"""
    model, log = _parse_deck(tmp_path, fixed_deck)
    assert not log.errors, f"Errors: {log.errors}"
    s1 = next(s for s in model.sensors if s.id == 401)
    assert s1.kind == "WORK"
    assert s1.tdelay == pytest.approx(0.01)
    assert s1.node_id1 == 100
    assert s1.node_id2 == 101
    assert s1.work_max == pytest.approx(5000.0)
    assert s1.tmin == pytest.approx(0.0005)
    assert s1.sect_id == 10
    assert s1.int_id == 11
    assert s1.rbody_id == 12
    assert s1.rwall_id == 13

    # Free format /SENSOR/TYPE13
    free_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Free Sensor TYPE13
/SENSOR/TYPE13/402
Sensor TYPE13 Free
0.02
200 201 6000.0 0.0006
20 21 22 23
/END
"""
    model, log = _parse_deck(tmp_path, free_deck)
    assert not log.errors, f"Errors: {log.errors}"
    s2 = next(s for s in model.sensors if s.id == 402)
    assert s2.kind == "WORK"
    assert s2.tdelay == pytest.approx(0.02)
    assert s2.node_id1 == 200
    assert s2.node_id2 == 201
    assert s2.work_max == pytest.approx(6000.0)
    assert s2.tmin == pytest.approx(0.0006)
    assert s2.sect_id == 20
    assert s2.int_id == 21
    assert s2.rbody_id == 22
    assert s2.rwall_id == 23


def test_sensor_dist_surf_fixed_and_free(tmp_path: Path):
    """Test /SENSOR/DIST_SURF and /SENSOR/TYPE17 in fixed and free formats."""
    # Fixed format /SENSOR/DIST_SURF
    fixed_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Fixed Sensor Dist Surf
/SENSOR/DIST_SURF/501
Sensor Dist Surf Fixed
#             Tdelay
                0.015
#  node_ID              node_ID1  node_ID2  node_ID3
       500                  51        52        53
#                 Dmin                Dmax                                        Tmin
                   0.5                 2.5                                      0.0002
/END
"""
    model, log = _parse_deck(tmp_path, fixed_deck)
    assert not log.errors, f"Errors: {log.errors}"
    s1 = next(s for s in model.sensors if s.id == 501)
    assert s1.kind == "DIST_SURF"
    assert s1.tdelay == pytest.approx(0.015)
    assert s1.node_id == 500
    assert s1.node_id2 == 51
    assert s1.node_id3 == 52
    assert s1.node_id4 == 53
    assert s1.dmin == pytest.approx(0.5)
    assert s1.dmax == pytest.approx(2.5)
    assert s1.tmin == pytest.approx(0.0002)

    # Free format /SENSOR/TYPE17
    free_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Free Sensor TYPE17
/SENSOR/TYPE17/502
Sensor TYPE17 Free
0.025
600 0 61 62 63
0.8 3.2 0.0004
/END
"""
    model, log = _parse_deck(tmp_path, free_deck)
    assert not log.errors, f"Errors: {log.errors}"
    s2 = next(s for s in model.sensors if s.id == 502)
    assert s2.kind == "DIST_SURF"
    assert s2.tdelay == pytest.approx(0.025)
    assert s2.node_id == 600
    assert s2.node_id2 == 61
    assert s2.node_id3 == 62
    assert s2.node_id4 == 63
    assert s2.dmin == pytest.approx(0.8)
    assert s2.dmax == pytest.approx(3.2)
    assert s2.tmin == pytest.approx(0.0004)

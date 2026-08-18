"""Tests for Milestone M207:
- Pull/Push Directional Springs: /PROP/TYPE47, /PROP/SPR_PULL, /PROP/TYPE48, /PROP/SPR_PUSH
- Energy Ratio & Shear Lock Sensors: /SENSOR/RATIO, /SENSOR/ENERGY_RATIO, /SENSOR/SHEAR_LOCK
- Parallelepiped & Truncated Cone Rigid Walls: /RWALL/PARALLELEPIPED, /RWALL/TRUNC_CONE
- Interface Time Step Controls: /DT/INTER/DEL, /DT/NODA/CFL
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m207_prop_spr_pull_and_push(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Pull and Push Springs Test
1 1
/PROP/TYPE47/101
Tension Only Pull Spring
                 0.1              1000.0                50.0             50000.0
/PROP/SPR_PUSH/102
Compression Only Push Spring
                 0.2              2000.0               100.0             80000.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 101 in model.props_type47
    p47 = model.props_type47[101]
    assert pytest.approx(p47.mass) == 0.1
    assert pytest.approx(p47.stiffness_k) == 1000.0
    assert pytest.approx(p47.k) == 1000.0
    assert pytest.approx(p47.damping_c) == 50.0
    assert pytest.approx(p47.c) == 50.0
    assert pytest.approx(p47.fmax) == 50000.0
    assert model.props_spr_pull is model.props_type47

    assert 102 in model.props_type48
    p48 = model.props_type48[102]
    assert pytest.approx(p48.mass) == 0.2
    assert pytest.approx(p48.stiffness_k) == 2000.0
    assert pytest.approx(p48.k) == 2000.0
    assert pytest.approx(p48.damping_c) == 100.0
    assert pytest.approx(p48.c) == 100.0
    assert pytest.approx(p48.fmax) == 80000.0
    assert model.props_spr_push is model.props_type48


def test_m207_sensor_ratio_and_shearlock(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Sensors Ratio and Shear Lock Test
1 1
/SENSOR/RATIO/201
Energy Ratio Sensor
         1                 0.0                 0.1                1e-3                0.01
/SENSOR/SHEAR_LOCK/202
Shear Locking Sensor
         5                 0.2                1e-4                0.02
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 201 in model.sensors_ratio
    sr = model.sensors_ratio[201]
    assert sr.ratio_type == 1
    assert pytest.approx(sr.val_min) == 0.0
    assert pytest.approx(sr.val_max) == 0.1
    assert pytest.approx(sr.tmin) == 1e-3
    assert pytest.approx(sr.tdelay) == 0.01
    assert model.sensors_energy_ratio is model.sensors_ratio

    assert 202 in model.sensors_shear_lock
    ssl = model.sensors_shear_lock[202]
    assert ssl.part_id == 5
    assert pytest.approx(ssl.val_max) == 0.2
    assert pytest.approx(ssl.tmin) == 1e-4
    assert pytest.approx(ssl.tdelay) == 0.02


def test_m207_rwall_parallelepiped_and_trunc_cone(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Rigid Wall Aliases Test
1 1
/RWALL/PARALLELEPIPED/301
Parallelepiped Wall
         0         2        10         0
                 0.0                 0.0                 0.0               100.0                50.0                20.0
/RWALL/TRUNC_CONE/302
Truncated Cone Wall
         0         1        20         0
                 0.0                 0.0                 0.0                 0.0                 0.0                 1.0                45.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 301 in model.rwall_boxes
    wb = model.rwall_boxes[301]
    assert wb.slide == 2
    assert wb.grnod_id == 10
    assert pytest.approx(wb.p1) == (0.0, 0.0, 0.0)
    assert pytest.approx(wb.p2) == (100.0, 50.0, 20.0)

    assert 302 in model.rwall_cones
    wc = model.rwall_cones[302]
    assert wc.slide == 1
    assert wc.grnod_id == 20
    assert pytest.approx(wc.apex) == (0.0, 0.0, 0.0)
    assert pytest.approx(wc.axis) == (0.0, 0.0, 1.0)
    assert pytest.approx(wc.angle) == 45.0


def test_m207_dt_inter_del_and_noda_cfl(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
DT Controls Test
1 1
/DT/INTER/DEL
              1.0e-8
/DT/NODA/CFL
                0.67
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert model.dt_inter_del is True
    assert pytest.approx(model.dt_inter_del_val) == 1.0e-8
    assert pytest.approx(model.dt_noda_cfl) == 0.67

"""Tests for Milestone M227:
- Wood Orthotropic Failure Criterion: /FAIL/WOOD, /FAIL/TIMBER
- Engine Shell Thickness Directive: /SH_THICK, /ENG/SH_THICK
- Slider Line & Cardan Joint Aliases: /LAGMUL/SLIDER_LINE, /LAGMUL/CARDAN_JOINT
- Solid Strain Sensor: /SENSOR/SOLID_STRAIN, /SENSOR/STRAIN_SOLID
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


def test_m227_fail_wood(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Wood Orthotropic Failure Test
1 1
/FAIL/WOOD/88
Wood Orthotropic Failure Criterion
                50.0                15.0                30.0                10.0
                 8.0         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 88 in model.fail_woods
    fw = model.fail_woods[88]
    assert pytest.approx(fw.sigma_t11) == 50.0
    assert pytest.approx(fw.sigma_t22) == 15.0
    assert pytest.approx(fw.sigma_c11) == 30.0
    assert pytest.approx(fw.sigma_c22) == 10.0
    assert pytest.approx(fw.tau_12) == 8.0
    assert fw.ifail_sh == 1


def test_m227_eng_sh_thick(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Shell Thickness Directive Test
1 1
/ENG/SH_THICK/1
Shell Thickness Tracking Control
               0.001         5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_sh_thicks
    est = model.eng_sh_thicks[1]
    assert pytest.approx(est.dt_thick) == 0.001
    assert est.sens_id == 5


def test_m227_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Slider Line and Cardan Joint Aliases Test
1 1
/LAGMUL/SLIDER_LINE/21
Slider Line Joint Constraint
        15        25         2
/LAGMUL/CARDAN_JOINT/22
Cardan Joint Constraint
        35        45         3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 21 in model.slider_joints
    assert model.slider_joints[21].node1 == 15
    assert model.slider_joints[21].node2 == 25

    assert 22 in model.cardan_joints
    assert model.cardan_joints[22].node1 == 35
    assert model.cardan_joints[22].node2 == 45


def test_m227_sensor_solid_strain(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Solid Plastic Strain Sensor Test
1 1
/SENSOR/SOLID_STRAIN/110
Solid Strain Threshold Sensor
       200                 0.22         4               0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 110 in model.sensor_solid_strains
    assert any(s.id == 110 and s.kind == "SOLID_STRAIN" for s in model.sensors)
    sss = model.sensor_solid_strains[110]
    assert sss.solid_id == 200
    assert pytest.approx(sss.eps_max) == 0.22
    assert sss.ip == 4
    assert pytest.approx(sss.t_delay) == 0.003

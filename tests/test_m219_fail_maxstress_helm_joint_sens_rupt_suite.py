"""Tests for Milestone M219:
- Maximum Stress Failure Criterion: /FAIL/MAX_STRESS, /FAIL/MAXSTRESS, /FAIL/MAX_TENS
- Helmholtz Acoustic Output Directive: /HELM, /ENG/HELM
- Hinge & Translational Kinematic Joint Aliases: /LAGMUL/HINGE, /LAGMUL/TRANSLATIONAL
- Element Rupture / Failure Sensor: /SENSOR/RUPT, /SENSOR/SHELL_FAIL
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


def test_m219_fail_maxstress(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Maximum Stress Failure Test
1 1
/FAIL/MAX_STRESS/55
Max Stress Material Failure
               450.0               350.0                80.0               120.0                55.0         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 55 in model.fail_maxstresses
    fms = model.fail_maxstresses[55]
    assert pytest.approx(fms.sig_t1) == 450.0
    assert pytest.approx(fms.sig_c1) == 350.0
    assert pytest.approx(fms.sig_t2) == 80.0
    assert pytest.approx(fms.sig_c2) == 120.0
    assert pytest.approx(fms.tau_12) == 55.0
    assert fms.ifail_sh == 2


def test_m219_eng_helm(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Helmholtz Acoustic Directive Test
1 1
/ENG/HELM/1
Acoustic Frequency Domain Response
                20.0              2000.0        50
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_helms
    helm = model.eng_helms[1]
    assert pytest.approx(helm.freq_start) == 20.0
    assert pytest.approx(helm.freq_end) == 2000.0
    assert helm.n_step == 50


def test_m219_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Kinematic Joint Aliases Test
1 1
/LAGMUL/HINGE/3
Hinge Revolute Constraint
         5         6
/LAGMUL/TRANSLATIONAL/4
Translational Slider Constraint
         7         8
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 3 in model.pin_joints
    assert model.pin_joints[3].node1 == 5
    assert model.pin_joints[3].node2 == 6
    assert 4 in model.slider_joints
    assert model.slider_joints[4].node1 == 7
    assert model.slider_joints[4].node2 == 8


def test_m219_sensor_rupture(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Element Rupture Sensor Test
1 1
/SENSOR/RUPT/15
Target Shell Failure Sensor
      1054         1               0.005
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 15 in model.sensor_ruptures
    assert any(s.id == 15 and s.kind == "RUPT" for s in model.sensors)
    sr = model.sensor_ruptures[15]
    assert sr.elem_id == 1054
    assert sr.itype == 1
    assert pytest.approx(sr.t_delay) == 0.005

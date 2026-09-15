"""Tests for Milestone M217:
- Slot Kinematic Joint Constraint: /LAGMUL/SLOT, /SLOT
- Orthotropic Failure Criterion: /FAIL/ORTHO, /FAIL/ORTHOTROPIC
- Engine FFT Frequency Spectrum Output Directive: /FXFREQ, /ENG/FXFREQ
- Energy Ratio Sensor Trigger: /SENSOR/ENERGY_RATIO, /SENSOR/ENG_RATIO
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


def test_m217_slot_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Slot Kinematic Joint Test
1 1
/LAGMUL/SLOT/10
Slot Kinematic Joint 10
         1         2         3         2               -10.0                50.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 10 in model.slot_joints
    sj = model.slot_joints[10]
    assert sj.node1 == 1
    assert sj.node2 == 2
    assert sj.skew_id == 3
    assert sj.axis_dir == 2
    assert pytest.approx(sj.d_min) == -10.0
    assert pytest.approx(sj.d_max) == 50.0


def test_m217_fail_ortho(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Orthotropic Failure Criterion Test
1 1
/FAIL/ORTHO/88
Orthotropic Lamina Failure
              1200.0               800.0                50.0               200.0                80.0         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 88 in model.fail_orthos
    fo = model.fail_orthos[88]
    assert pytest.approx(fo.xt) == 1200.0
    assert pytest.approx(fo.xc) == 800.0
    assert pytest.approx(fo.yt) == 50.0
    assert pytest.approx(fo.yc) == 200.0
    assert pytest.approx(fo.s) == 80.0
    assert fo.ifail_sh == 2


def test_m217_eng_fxfreq(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine FXFREQ Test
1 1
/ENG/FXFREQ/1
Fast Fourier Transform Frequency Spectrum
              2500.0       500                0.01                 0.1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_fxfreqs
    fxf = model.eng_fxfreqs[1]
    assert pytest.approx(fxf.f_max) == 2500.0
    assert fxf.n_freq == 500
    assert pytest.approx(fxf.t_start) == 0.01
    assert pytest.approx(fxf.t_end) == 0.1


def test_m217_sensor_energy_ratio(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Energy Ratio Sensor Test
1 1
/SENSOR/ENERGY_RATIO/66
Energy Ratio Sensor 66
                1.15                0.85               0.005
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 66 in model.sensor_energy_ratios
    assert any(s.id == 66 and s.kind == "ENERGY_RATIO" for s in model.sensors)
    ser = model.sensor_energy_ratios[66]
    assert pytest.approx(ser.ratio_max) == 1.15
    assert pytest.approx(ser.ratio_min) == 0.85
    assert pytest.approx(ser.t_delay) == 0.005

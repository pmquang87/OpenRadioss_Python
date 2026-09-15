"""Tests for Milestone M280: Mullins Failure Model, EngFsiEnergy, SliderSlotJoint, and SensorSpringTranslationalWork."""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, deck_text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck_text, encoding="utf-8")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(p))
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m280_fail_mullins_fixed(tmp_path: Path):
    c1 = f"{1.25:>20.4f}{0.15:>20.4f}{0.85:>20.4f}"
    c2 = f"{860:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Mullins Failure Model Fixed Format Test
2022 0
/FAIL/MULLINS/860
Mullins Elastomer Damage Model
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 860 in model.fail_mullins
    fm = model.fail_mullins[860]
    assert pytest.approx(fm.coefr) == 1.25
    assert pytest.approx(fm.beta) == 0.15
    assert pytest.approx(fm.coefm) == 0.85
    assert fm.fail_id == 860
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "MULLINS"


def test_m280_fail_mullins_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Mullins Free Format Test
/FAIL/MULLINS/861
1.40, 0.20, 0.75
861
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 861 in model.fail_mullins
    fm = model.fail_mullins[861]
    assert pytest.approx(fm.coefr) == 1.40
    assert pytest.approx(fm.beta) == 0.20
    assert pytest.approx(fm.coefm) == 0.75
    assert fm.fail_id == 861


def test_m280_fail_mullins_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Mullins Aliases Test
/FAIL/MULLINS_OR/862
1.1, 0.1, 0.9
/FAIL/MULLINS_MODEL/863
1.1, 0.1, 0.9
/FAIL/MULLINS_LAW/864
1.1, 0.1, 0.9
/FAIL/MULLINS_DAMAGE/865
1.1, 0.1, 0.9
/FAIL/ELASTOMER_DAMAGE/866
1.1, 0.1, 0.9
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 862 in model.fail_mullins
    assert 863 in model.fail_mullins
    assert 864 in model.fail_mullins
    assert 865 in model.fail_mullins
    assert 866 in model.fail_mullins
    assert len(model.raw_fails) == 5


def test_m280_eng_fsi_energy(tmp_path: Path):
    c1 = f"{0.0001:>20.6f}{4:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
FSI Energy Fixed and Free Format Test
2022 0
/ENG/FSI_ENERGY/1
Fixed FSI Energy
{c1}
/ENG/FSI_ENERGY/2
Free FSI Energy
0.0003, 8
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_fsi_energies
    assert 2 in model.eng_fsi_energies
    fe1 = model.eng_fsi_energies[1]
    assert pytest.approx(fe1.dt_fsi) == 0.0001
    assert fe1.sens_id == 4
    fe2 = model.eng_fsi_energies[2]
    assert pytest.approx(fe2.dt_fsi) == 0.0003
    assert fe2.sens_id == 8


def test_m280_eng_fsi_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
FSI Energy Aliases Test
/ENG/FSI_WORK/11
0.001, 10
/ENG/EFSI/12
0.002, 11
/ENG/FSI_ENER/13
0.003, 12
/ENG/FSI_INTERNAL_ENERGY/14
0.004, 13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.eng_fsi_energies
    assert 12 in model.eng_fsi_energies
    assert 13 in model.eng_fsi_energies
    assert 14 in model.eng_fsi_energies
    assert pytest.approx(model.eng_fsi_energies[11].dt_fsi) == 0.001
    assert pytest.approx(model.eng_fsi_energies[12].dt_fsi) == 0.002
    assert pytest.approx(model.eng_fsi_energies[13].dt_fsi) == 0.003
    assert pytest.approx(model.eng_fsi_energies[14].dt_fsi) == 0.004


def test_m280_slider_slot_joint(tmp_path: Path):
    c1 = f"{111:>10d}{112:>10d}{113:>10d}{3.0e6:>20.4f}{6:>10d}{5.0e-6:>20.6f}"
    c2 = f"{0.0:>20.4f}{1.0:>20.4f}{0.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Slider Slot Joint Fixed and Free Format Test
2022 0
/LAGMUL/SLIDER_SLOT_JOINT/1
Fixed Slider Slot Joint
{c1}
{c2}
/LAGMUL/SLIDER_SLOT_JOINT/2
Free Slider Slot Joint
211, 212, 213, 2.5e6, 7, 8.0e-6
0.0, 0.0, 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_slider_slot_joints
    assert 2 in model.lagmul_slider_slot_joints
    ss1 = model.lagmul_slider_slot_joints[1]
    assert ss1.node1 == 111
    assert ss1.node2 == 112
    assert ss1.node3 == 113
    assert pytest.approx(ss1.stiff) == 3.0e6
    assert ss1.skew_id == 6
    assert pytest.approx(ss1.tol) == 5.0e-6
    assert pytest.approx(ss1.axis_x) == 0.0
    assert pytest.approx(ss1.axis_y) == 1.0
    assert pytest.approx(ss1.axis_z) == 0.0

    ss2 = model.lagmul_slider_slot_joints[2]
    assert ss2.node1 == 211
    assert ss2.node2 == 212
    assert ss2.node3 == 213
    assert pytest.approx(ss2.stiff) == 2.5e6
    assert ss2.skew_id == 7
    assert pytest.approx(ss2.tol) == 8.0e-6
    assert pytest.approx(ss2.axis_x) == 0.0
    assert pytest.approx(ss2.axis_y) == 0.0
    assert pytest.approx(ss2.axis_z) == 1.0


def test_m280_slider_slot_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Slider Slot Joint Aliases Test
/SLIDER_SLOT_JOINT/41
401, 402, 403, 1.0e6, 0, 1.0e-6
1.0, 0.0, 0.0
/LAGMUL/SLIDER_SLOT/42
404, 405, 406, 1.0e6, 0, 1.0e-6
0.0, 1.0, 0.0
/SLIDER_SLOT/43
407, 408, 409, 1.0e6, 0, 1.0e-6
0.0, 0.0, 1.0
/SLIDER_SLOT_MECHANISM/44
410, 411, 412, 1.0e6, 0, 1.0e-6
1.0, 0.0, 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 41 in model.lagmul_slider_slot_joints
    assert 42 in model.lagmul_slider_slot_joints
    assert 43 in model.lagmul_slider_slot_joints
    assert 44 in model.lagmul_slider_slot_joints
    assert model.lagmul_slider_slot_joints[41].node1 == 401
    assert model.lagmul_slider_slot_joints[42].node1 == 404
    assert model.lagmul_slider_slot_joints[43].node1 == 407
    assert model.lagmul_slider_slot_joints[44].node1 == 410


def test_m280_sensor_spring_translational_work(tmp_path: Path):
    c1 = f"{701:>10d}{4500.0:>20.4f}{0.003:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Translational Work Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TRANSLATIONAL_WORK/1
Fixed Spring Translational Work Sensor
{c1}
/SENSOR/SPRING_TRANSLATIONAL_WORK/2
Free Spring Translational Work Sensor
702, 6800.0, 0.005
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_translational_works
    assert 2 in model.sensor_spring_translational_works
    s1 = model.sensor_spring_translational_works[1]
    assert s1.spring_id == 701
    assert pytest.approx(s1.w_trans_max) == 4500.0
    assert pytest.approx(s1.t_delay) == 0.003

    s2 = model.sensor_spring_translational_works[2]
    assert s2.spring_id == 702
    assert pytest.approx(s2.w_trans_max) == 6800.0
    assert pytest.approx(s2.t_delay) == 0.005

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TRANSLATIONAL_WORK"
    assert model.sensors[1].kind == "SPRING_TRANSLATIONAL_WORK"


def test_m280_sensor_spring_translational_work_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Translational Work Aliases Test
/SENSOR/SPRING_TRANS_WORK/11
801, 7200.0, 0.001
/SENSOR/SPRING_WORK_TRANS/12
802, 8300.0, 0.002
/SENSOR/TRANSLATIONAL_WORK_SPRING/13
803, 9400.0, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.sensor_spring_translational_works
    assert 12 in model.sensor_spring_translational_works
    assert 13 in model.sensor_spring_translational_works
    assert model.sensor_spring_translational_works[11].spring_id == 801
    assert pytest.approx(model.sensor_spring_translational_works[11].w_trans_max) == 7200.0
    assert model.sensor_spring_translational_works[12].spring_id == 802
    assert model.sensor_spring_translational_works[13].spring_id == 803

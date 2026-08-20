"""Tests for Milestone M279: TButcher Failure Model, EngAleEnergy, PinInSlotJoint, and SensorSpringRotationalWork."""

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


def test_m279_fail_tbutcher_fixed(tmp_path: Path):
    c1 = f"{1.5e-4:>20.6f}{2.0:>20.4f}{350.0:>20.4f}{1:>10d}{2:>10d}{1:>10d}{0:>10d}"
    c2 = f"{0.05:>20.4f}{0.1:>20.4f}{0.01:>20.4f}"
    c3 = f"{850:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
TButcher Failure Model Fixed Format Test
2022 0
/FAIL/TBUTCHER/850
TButcher Dynamic Damage Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 850 in model.fail_tbutchers
    tb = model.fail_tbutchers[850]
    assert pytest.approx(tb.lam) == 1.5e-4
    assert pytest.approx(tb.k) == 2.0
    assert pytest.approx(tb.sigma_r) == 350.0
    assert tb.ifail_sh == 1
    assert tb.ifail_so == 2
    assert tb.iduct == 1
    assert tb.ixfem == 0
    assert pytest.approx(tb.a) == 0.05
    assert pytest.approx(tb.b) == 0.1
    assert pytest.approx(tb.dadv) == 0.01
    assert tb.fail_id == 850
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "TBUTCHER"


def test_m279_fail_tbutcher_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
TButcher Free Format Test
/FAIL/TBUTCHER/851
2.5e-4, 1.8, 420.0, 1, 1, 0, 0
0.04, 0.08, 0.02
851
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 851 in model.fail_tbutchers
    tb = model.fail_tbutchers[851]
    assert pytest.approx(tb.lam) == 2.5e-4
    assert pytest.approx(tb.k) == 1.8
    assert pytest.approx(tb.sigma_r) == 420.0
    assert tb.ifail_sh == 1
    assert tb.ifail_so == 1
    assert tb.iduct == 0
    assert tb.ixfem == 0
    assert pytest.approx(tb.a) == 0.04
    assert pytest.approx(tb.b) == 0.08
    assert pytest.approx(tb.dadv) == 0.02
    assert tb.fail_id == 851


def test_m279_fail_tbutcher_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
TButcher Aliases Test
/FAIL/TULER_BUTCHER_CRITERION/852
1.0e-4, 2.0, 300.0, 1, 1, 0, 0
/FAIL/TBUTCHER_MODEL/853
1.0e-4, 2.0, 300.0, 1, 1, 0, 0
/FAIL/TBUTCHER_LAW/854
1.0e-4, 2.0, 300.0, 1, 1, 0, 0
/FAIL/TULER_BUTCHER_DAMAGE/855
1.0e-4, 2.0, 300.0, 1, 1, 0, 0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 852 in model.fail_tbutchers
    assert 853 in model.fail_tbutchers
    assert 854 in model.fail_tbutchers
    assert 855 in model.fail_tbutchers
    assert len(model.raw_fails) == 4


def test_m279_eng_ale_energy(tmp_path: Path):
    c1 = f"{0.0002:>20.6f}{5:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
ALE Energy Fixed and Free Format Test
2022 0
/ENG/ALE_ENERGY/1
Fixed ALE Energy
{c1}
/ENG/ALE_ENERGY/2
Free ALE Energy
0.0005, 10
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_ale_energies
    assert 2 in model.eng_ale_energies
    ae1 = model.eng_ale_energies[1]
    assert pytest.approx(ae1.dt_ale) == 0.0002
    assert ae1.sens_id == 5
    ae2 = model.eng_ale_energies[2]
    assert pytest.approx(ae2.dt_ale) == 0.0005
    assert ae2.sens_id == 10


def test_m279_eng_ale_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
ALE Energy Aliases Test
/ENG/ALE_WORK/11
0.001, 10
/ENG/EALE/12
0.002, 11
/ENG/ALE_ENER/13
0.003, 12
/ENG/ALE_INTERNAL_ENERGY/14
0.004, 13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.eng_ale_energies
    assert 12 in model.eng_ale_energies
    assert 13 in model.eng_ale_energies
    assert 14 in model.eng_ale_energies
    assert pytest.approx(model.eng_ale_energies[11].dt_ale) == 0.001
    assert pytest.approx(model.eng_ale_energies[12].dt_ale) == 0.002
    assert pytest.approx(model.eng_ale_energies[13].dt_ale) == 0.003
    assert pytest.approx(model.eng_ale_energies[14].dt_ale) == 0.004


def test_m279_pin_in_slot_joint(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{2.0e6:>20.4f}{4:>10d}{1.0e-5:>20.6f}"
    c2 = f"{1.0:>20.4f}{0.0:>20.4f}{0.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Pin In Slot Joint Fixed and Free Format Test
2022 0
/LAGMUL/PIN_IN_SLOT_JOINT/1
Fixed Pin In Slot Joint
{c1}
{c2}
/LAGMUL/PIN_IN_SLOT_JOINT/2
Free Pin In Slot Joint
201, 202, 203, 1.5e6, 5, 2.0e-5
0.0, 1.0, 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_pin_in_slot_joints
    assert 2 in model.lagmul_pin_in_slot_joints
    ps1 = model.lagmul_pin_in_slot_joints[1]
    assert ps1.node1 == 101
    assert ps1.node2 == 102
    assert ps1.node3 == 103
    assert pytest.approx(ps1.stiff) == 2.0e6
    assert ps1.skew_id == 4
    assert pytest.approx(ps1.tol) == 1.0e-5
    assert pytest.approx(ps1.axis_x) == 1.0
    assert pytest.approx(ps1.axis_y) == 0.0
    assert pytest.approx(ps1.axis_z) == 0.0

    ps2 = model.lagmul_pin_in_slot_joints[2]
    assert ps2.node1 == 201
    assert ps2.node2 == 202
    assert ps2.node3 == 203
    assert pytest.approx(ps2.stiff) == 1.5e6
    assert ps2.skew_id == 5
    assert pytest.approx(ps2.tol) == 2.0e-5
    assert pytest.approx(ps2.axis_x) == 0.0
    assert pytest.approx(ps2.axis_y) == 1.0
    assert pytest.approx(ps2.axis_z) == 0.0


def test_m279_pin_in_slot_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Pin In Slot Joint Aliases Test
/PIN_IN_SLOT_JOINT/31
301, 302, 303, 1.0e6, 0, 1.0e-6
0.0, 0.0, 1.0
/LAGMUL/PIN_IN_SLOT/32
304, 305, 306, 1.0e6, 0, 1.0e-6
1.0, 0.0, 0.0
/PIN_IN_SLOT/33
307, 308, 309, 1.0e6, 0, 1.0e-6
0.0, 1.0, 0.0
/PIN_IN_SLOT_MECHANISM/34
310, 311, 312, 1.0e6, 0, 1.0e-6
0.0, 0.0, 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 31 in model.lagmul_pin_in_slot_joints
    assert 32 in model.lagmul_pin_in_slot_joints
    assert 33 in model.lagmul_pin_in_slot_joints
    assert 34 in model.lagmul_pin_in_slot_joints
    assert model.lagmul_pin_in_slot_joints[31].node1 == 301
    assert model.lagmul_pin_in_slot_joints[32].node1 == 304
    assert model.lagmul_pin_in_slot_joints[33].node1 == 307
    assert model.lagmul_pin_in_slot_joints[34].node1 == 310


def test_m279_sensor_spring_rotational_work(tmp_path: Path):
    c1 = f"{501:>10d}{3200.0:>20.4f}{0.002:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Rotational Work Fixed and Free Format Test
2022 0
/SENSOR/SPRING_ROTATIONAL_WORK/1
Fixed Spring Rotational Work Sensor
{c1}
/SENSOR/SPRING_ROTATIONAL_WORK/2
Free Spring Rotational Work Sensor
502, 5400.0, 0.004
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_rotational_works
    assert 2 in model.sensor_spring_rotational_works
    s1 = model.sensor_spring_rotational_works[1]
    assert s1.spring_id == 501
    assert pytest.approx(s1.w_rot_max) == 3200.0
    assert pytest.approx(s1.t_delay) == 0.002

    s2 = model.sensor_spring_rotational_works[2]
    assert s2.spring_id == 502
    assert pytest.approx(s2.w_rot_max) == 5400.0
    assert pytest.approx(s2.t_delay) == 0.004

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_ROTATIONAL_WORK"
    assert model.sensors[1].kind == "SPRING_ROTATIONAL_WORK"


def test_m279_sensor_spring_rotational_work_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Rotational Work Aliases Test
/SENSOR/SPRING_ROT_WORK/11
601, 6200.0, 0.001
/SENSOR/SPRING_WORK_ROT/12
602, 7300.0, 0.002
/SENSOR/ROTATIONAL_WORK_SPRING/13
603, 8400.0, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.sensor_spring_rotational_works
    assert 12 in model.sensor_spring_rotational_works
    assert 13 in model.sensor_spring_rotational_works
    assert model.sensor_spring_rotational_works[11].spring_id == 601
    assert pytest.approx(model.sensor_spring_rotational_works[11].w_rot_max) == 6200.0
    assert model.sensor_spring_rotational_works[12].spring_id == 602
    assert model.sensor_spring_rotational_works[13].spring_id == 603

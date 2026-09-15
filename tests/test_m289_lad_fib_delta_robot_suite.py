"""Tests for Milestone M289: LadFib Failure Model, EngJouleHeatEnergy, DeltaRobotJoint, and SensorSpringShearAcceleration."""

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


def test_m289_fail_lad_fib_fixed(tmp_path: Path):
    c1 = f"{0.018:>20.4f}{0.012:>20.4f}{2400.0:>20.4f}{1600.0:>20.4f}{0.250:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{995:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Fiber Failure Model Fixed Format Test
2022 0
/FAIL/LAD_FIB/995
Ladeveze Longitudinal Fiber Rupture Criterion
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 995 in model.fail_ladfibs
    flf = model.fail_ladfibs[995]
    assert pytest.approx(flf.eps_ft) == 0.018
    assert pytest.approx(flf.eps_fc) == 0.012
    assert pytest.approx(flf.sigma_ft) == 2400.0
    assert pytest.approx(flf.sigma_fc) == 1600.0
    assert pytest.approx(flf.gamma_fib) == 0.250
    assert flf.ifail_sh == 1
    assert flf.ifail_so == 2
    assert flf.fail_id == 995
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_FIB"


def test_m289_fail_lad_fib_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Fiber Free Format Test
/FAIL/LAD_FIB/996
0.015, 0.010, 2200.0, 1500.0, 0.220
1, 1
996
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 996 in model.fail_ladfibs
    flf = model.fail_ladfibs[996]
    assert pytest.approx(flf.eps_ft) == 0.015
    assert pytest.approx(flf.gamma_fib) == 0.220
    assert flf.fail_id == 996


def test_m289_fail_lad_fib_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Fiber Aliases Test
/FAIL/LADEVEZE_FIBER/997
0.016, 0.011, 2300.0, 1550.0, 0.23
1, 1
/FAIL/LAD_FIBER/998
0.016, 0.011, 2300.0, 1550.0, 0.23
1, 1
/FAIL/LAD_FIB_MODEL/999
0.016, 0.011, 2300.0, 1550.0, 0.23
1, 1
/FAIL/LAD_FIB_LAW/1000
0.016, 0.011, 2300.0, 1550.0, 0.23
1, 1
/FAIL/LADEVEZE_FIBER_BRITTLE/1001
0.016, 0.011, 2300.0, 1550.0, 0.23
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 997 in model.fail_ladfibs
    assert 998 in model.fail_ladfibs
    assert 999 in model.fail_ladfibs
    assert 1000 in model.fail_ladfibs
    assert 1001 in model.fail_ladfibs
    assert len(model.raw_fails) == 5


def test_m289_eng_joule_heat_energy(tmp_path: Path):
    c1 = f"{0.00085:>20.6f}{13:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Joule Heat Energy Fixed and Free Format Test
2022 0
/ENG/JOULE_HEAT_ENERGY/1
Fixed Joule Heat Energy Output
{c1}
/ENG/JOULE_HEAT_ENERGY/2
Free Joule Heat Energy Output
0.0017, 26
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_joule_heat_energies
    assert 2 in model.eng_joule_heat_energies
    je1 = model.eng_joule_heat_energies[1]
    assert pytest.approx(je1.dt_joule) == 0.00085
    assert je1.sens_id == 13
    je2 = model.eng_joule_heat_energies[2]
    assert pytest.approx(je2.dt_joule) == 0.0017
    assert je2.sens_id == 26


def test_m289_eng_joule_heat_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Joule Heat Energy Aliases Test
/ENG/JOULE_HEAT_WORK/11
0.001, 10
/ENG/EJOULE/12
0.002, 11
/ENG/JOULE_HEATING_ENERGY/13
0.003, 12
/ENG/EM_JOULE_HEAT/14
0.004, 13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.eng_joule_heat_energies
    assert 12 in model.eng_joule_heat_energies
    assert 13 in model.eng_joule_heat_energies
    assert 14 in model.eng_joule_heat_energies
    assert pytest.approx(model.eng_joule_heat_energies[11].dt_joule) == 0.001
    assert pytest.approx(model.eng_joule_heat_energies[12].dt_joule) == 0.002
    assert pytest.approx(model.eng_joule_heat_energies[13].dt_joule) == 0.003
    assert pytest.approx(model.eng_joule_heat_energies[14].dt_joule) == 0.004


def test_m289_delta_robot_joint(tmp_path: Path):
    c1 = f"{301:>10d}{302:>10d}{303:>10d}{7.5e6:>20.4f}{1:>10d}{1.8e-6:>20.6e}"
    c2 = f"{320.0:>20.4f}{650.0:>20.4f}{180.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Delta Robot Joint Fixed and Free Format Test
2022 0
/LAGMUL/DELTA_ROBOT_JOINT/1
Fixed Delta Robot Joint
{c1}
{c2}
/LAGMUL/DELTA_ROBOT_JOINT/2
Free Delta Robot Joint
401, 402, 403, 8.2e6, 2, 2.1e-6
350.0, 700.0, 200.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_delta_robot_joints
    assert 2 in model.lagmul_delta_robot_joints
    dj1 = model.lagmul_delta_robot_joints[1]
    assert dj1.node1 == 301
    assert dj1.node2 == 302
    assert dj1.node3 == 303
    assert pytest.approx(dj1.stiff) == 7.5e6
    assert dj1.skew_id == 1
    assert pytest.approx(dj1.tol) == 1.8e-6
    assert pytest.approx(dj1.upper_arm_len) == 320.0
    assert pytest.approx(dj1.forearm_len) == 650.0
    assert pytest.approx(dj1.base_radius) == 180.0

    dj2 = model.lagmul_delta_robot_joints[2]
    assert dj2.node1 == 401
    assert dj2.node2 == 402
    assert dj2.node3 == 403
    assert pytest.approx(dj2.stiff) == 8.2e6
    assert dj2.skew_id == 2
    assert pytest.approx(dj2.tol) == 2.1e-6
    assert pytest.approx(dj2.upper_arm_len) == 350.0
    assert pytest.approx(dj2.forearm_len) == 700.0
    assert pytest.approx(dj2.base_radius) == 200.0


def test_m289_delta_robot_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Delta Robot Joint Aliases Test
/DELTA_ROBOT_JOINT/61
961, 962, 963, 1.0e6, 0, 1.0e-6
300.0, 600.0, 150.0
/LAGMUL/DELTA_ROBOT/62
964, 965, 966, 1.0e6, 0, 1.0e-6
300.0, 600.0, 150.0
/DELTA_ROBOT/63
967, 968, 969, 1.0e6, 0, 1.0e-6
300.0, 600.0, 150.0
/DELTA_PARALLEL_ROBOT/64
970, 971, 972, 1.0e6, 0, 1.0e-6
300.0, 600.0, 150.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 61 in model.lagmul_delta_robot_joints
    assert 62 in model.lagmul_delta_robot_joints
    assert 63 in model.lagmul_delta_robot_joints
    assert 64 in model.lagmul_delta_robot_joints
    assert model.lagmul_delta_robot_joints[61].node1 == 961
    assert model.lagmul_delta_robot_joints[62].node1 == 964
    assert model.lagmul_delta_robot_joints[63].node1 == 967
    assert model.lagmul_delta_robot_joints[64].node1 == 970


def test_m289_sensor_spring_shear_acceleration(tmp_path: Path):
    c1 = f"{851:>10d}{38000.0:>20.4f}{0.0045:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Shear Acceleration Fixed and Free Format Test
2022 0
/SENSOR/SPRING_SHEAR_ACCELERATION/1
Fixed Spring Shear Acceleration Sensor
{c1}
/SENSOR/SPRING_SHEAR_ACCELERATION/2
Free Spring Shear Acceleration Sensor
852, 46000.0, 0.0065
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_shear_accelerations
    assert 2 in model.sensor_spring_shear_accelerations
    s1 = model.sensor_spring_shear_accelerations[1]
    assert s1.spring_id == 851
    assert pytest.approx(s1.accs_max) == 38000.0
    assert pytest.approx(s1.t_delay) == 0.0045

    s2 = model.sensor_spring_shear_accelerations[2]
    assert s2.spring_id == 852
    assert pytest.approx(s2.accs_max) == 46000.0
    assert pytest.approx(s2.t_delay) == 0.0065

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_SHEAR_ACCELERATION"
    assert model.sensors[1].kind == "SPRING_SHEAR_ACCELERATION"


def test_m289_sensor_spring_shear_acceleration_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Shear Acceleration Aliases Test
/SENSOR/SPRING_SHEAR_ACC/61
951, 30000.0, 0.001
/SENSOR/SPRING_ACC_SHEAR/62
952, 32000.0, 0.002
/SENSOR/SHEAR_ACCELERATION_SPRING/63
953, 34000.0, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 61 in model.sensor_spring_shear_accelerations
    assert 62 in model.sensor_spring_shear_accelerations
    assert 63 in model.sensor_spring_shear_accelerations
    assert model.sensor_spring_shear_accelerations[61].spring_id == 951
    assert pytest.approx(model.sensor_spring_shear_accelerations[61].accs_max) == 30000.0
    assert model.sensor_spring_shear_accelerations[62].spring_id == 952
    assert model.sensor_spring_shear_accelerations[63].spring_id == 953

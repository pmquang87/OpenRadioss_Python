"""Tests for Milestone M285: LouHuo Failure Model, EngCoriolisEnergy, CablePulleyJoint, and SensorSpringAngularVelocity."""

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


def test_m285_fail_lou_huo_fixed(tmp_path: Path):
    c1 = f"{1.25:>20.4f}{0.85:>20.4f}{2.10:>20.4f}{1.50:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{960:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Lou-Huo Failure Model Fixed Format Test
2022 0
/FAIL/LOU_HUO/960
Lou-Huo Ductile Fracture Criterion
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 960 in model.fail_louhuos
    flh = model.fail_louhuos[960]
    assert pytest.approx(flh.c1) == 1.25
    assert pytest.approx(flh.c2) == 0.85
    assert pytest.approx(flh.c3) == 2.10
    assert pytest.approx(flh.l_param) == 1.50
    assert flh.ifail_sh == 1
    assert flh.ifail_so == 2
    assert flh.fail_id == 960
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LOU_HUO"


def test_m285_fail_lou_huo_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Lou-Huo Free Format Test
/FAIL/LOU_HUO/961
1.15, 0.75, 1.95, 1.35
1, 1
961
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 961 in model.fail_louhuos
    flh = model.fail_louhuos[961]
    assert pytest.approx(flh.c1) == 1.15
    assert pytest.approx(flh.l_param) == 1.35
    assert flh.fail_id == 961


def test_m285_fail_lou_huo_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Lou-Huo Aliases Test
/FAIL/LOU_HUO_YANG/962
1.0, 0.5, 2.0, 1.0
1, 1
/FAIL/LOUHUO/963
1.0, 0.5, 2.0, 1.0
1, 1
/FAIL/LOU_HUO_MODEL/964
1.0, 0.5, 2.0, 1.0
1, 1
/FAIL/LOU_HUO_LAW/965
1.0, 0.5, 2.0, 1.0
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 962 in model.fail_louhuos
    assert 963 in model.fail_louhuos
    assert 964 in model.fail_louhuos
    assert 965 in model.fail_louhuos
    assert len(model.raw_fails) == 4


def test_m285_eng_coriolis_energy(tmp_path: Path):
    c1 = f"{0.00045:>20.6f}{9:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Coriolis Energy Fixed and Free Format Test
2022 0
/ENG/CORIOLIS_ENERGY/1
Fixed Coriolis Energy Output
{c1}
/ENG/CORIOLIS_ENERGY/2
Free Coriolis Energy Output
0.0009, 18
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_coriolis_energies
    assert 2 in model.eng_coriolis_energies
    ce1 = model.eng_coriolis_energies[1]
    assert pytest.approx(ce1.dt_coriolis) == 0.00045
    assert ce1.sens_id == 9
    ce2 = model.eng_coriolis_energies[2]
    assert pytest.approx(ce2.dt_coriolis) == 0.0009
    assert ce2.sens_id == 18


def test_m285_eng_coriolis_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Coriolis Energy Aliases Test
/ENG/CORIOLIS_WORK/11
0.001, 10
/ENG/ECORIOLIS/12
0.002, 11
/ENG/CORIOLIS_ENER/13
0.003, 12
/ENG/ROTATIONAL_CORIOLIS_ENERGY/14
0.004, 13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.eng_coriolis_energies
    assert 12 in model.eng_coriolis_energies
    assert 13 in model.eng_coriolis_energies
    assert 14 in model.eng_coriolis_energies
    assert pytest.approx(model.eng_coriolis_energies[11].dt_coriolis) == 0.001
    assert pytest.approx(model.eng_coriolis_energies[12].dt_coriolis) == 0.002
    assert pytest.approx(model.eng_coriolis_energies[13].dt_coriolis) == 0.003
    assert pytest.approx(model.eng_coriolis_energies[14].dt_coriolis) == 0.004


def test_m285_cable_pulley_joint(tmp_path: Path):
    c1 = f"{171:>10d}{172:>10d}{173:>10d}{6.5e6:>20.4f}{3:>10d}{1.5e-6:>20.6e}"
    c2 = f"{35.0:>20.4f}{180.0:>20.4f}{1.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Cable Pulley Joint Fixed and Free Format Test
2022 0
/LAGMUL/CABLE_PULLEY_JOINT/1
Fixed Cable Pulley Joint
{c1}
{c2}
/LAGMUL/CABLE_PULLEY_JOINT/2
Free Cable Pulley Joint
271, 272, 273, 7.5e6, 4, 2.0e-6
45.0, 135.0, 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_cable_pulley_joints
    assert 2 in model.lagmul_cable_pulley_joints
    cp1 = model.lagmul_cable_pulley_joints[1]
    assert cp1.node1 == 171
    assert cp1.node2 == 172
    assert cp1.node3 == 173
    assert pytest.approx(cp1.stiff) == 6.5e6
    assert cp1.skew_id == 3
    assert pytest.approx(cp1.tol) == 1.5e-6
    assert pytest.approx(cp1.pulley_radius) == 35.0
    assert pytest.approx(cp1.wrap_angle) == 180.0
    assert pytest.approx(cp1.axis_z) == 1.0

    cp2 = model.lagmul_cable_pulley_joints[2]
    assert cp2.node1 == 271
    assert cp2.node2 == 272
    assert cp2.node3 == 273
    assert pytest.approx(cp2.stiff) == 7.5e6
    assert cp2.skew_id == 4
    assert pytest.approx(cp2.tol) == 2.0e-6
    assert pytest.approx(cp2.pulley_radius) == 45.0
    assert pytest.approx(cp2.wrap_angle) == 135.0
    assert pytest.approx(cp2.axis_z) == 1.0


def test_m285_cable_pulley_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Cable Pulley Joint Aliases Test
/CABLE_PULLEY_JOINT/91
901, 902, 903, 1.0e6, 0, 1.0e-6
30.0, 180.0, 1.0
/LAGMUL/CABLE_PULLEY/92
904, 905, 906, 1.0e6, 0, 1.0e-6
30.0, 180.0, 1.0
/CABLE_PULLEY/93
907, 908, 909, 1.0e6, 0, 1.0e-6
30.0, 180.0, 1.0
/CABLE_PULLEY_MECHANISM/94
910, 911, 912, 1.0e6, 0, 1.0e-6
30.0, 180.0, 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 91 in model.lagmul_cable_pulley_joints
    assert 92 in model.lagmul_cable_pulley_joints
    assert 93 in model.lagmul_cable_pulley_joints
    assert 94 in model.lagmul_cable_pulley_joints
    assert model.lagmul_cable_pulley_joints[91].node1 == 901
    assert model.lagmul_cable_pulley_joints[92].node1 == 904
    assert model.lagmul_cable_pulley_joints[93].node1 == 907
    assert model.lagmul_cable_pulley_joints[94].node1 == 910


def test_m285_sensor_spring_angular_velocity(tmp_path: Path):
    c1 = f"{811:>10d}{125.0:>20.4f}{0.0075:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Angular Velocity Fixed and Free Format Test
2022 0
/SENSOR/SPRING_ANGULAR_VELOCITY/1
Fixed Spring Angular Velocity Sensor
{c1}
/SENSOR/SPRING_ANGULAR_VELOCITY/2
Free Spring Angular Velocity Sensor
812, 150.0, 0.0095
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_angular_velocities
    assert 2 in model.sensor_spring_angular_velocities
    s1 = model.sensor_spring_angular_velocities[1]
    assert s1.spring_id == 811
    assert pytest.approx(s1.omega_max) == 125.0
    assert pytest.approx(s1.t_delay) == 0.0075

    s2 = model.sensor_spring_angular_velocities[2]
    assert s2.spring_id == 812
    assert pytest.approx(s2.omega_max) == 150.0
    assert pytest.approx(s2.t_delay) == 0.0095

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_ANGULAR_VELOCITY"
    assert model.sensors[1].kind == "SPRING_ANGULAR_VELOCITY"


def test_m285_sensor_spring_angular_velocity_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Angular Velocity Aliases Test
/SENSOR/SPRING_ANG_VEL/21
911, 100.0, 0.001
/SENSOR/SPRING_OMEGA/22
912, 120.0, 0.002
/SENSOR/ANGULAR_VELOCITY_SPRING/23
913, 140.0, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 21 in model.sensor_spring_angular_velocities
    assert 22 in model.sensor_spring_angular_velocities
    assert 23 in model.sensor_spring_angular_velocities
    assert model.sensor_spring_angular_velocities[21].spring_id == 911
    assert pytest.approx(model.sensor_spring_angular_velocities[21].omega_max) == 100.0
    assert model.sensor_spring_angular_velocities[22].spring_id == 912
    assert model.sensor_spring_angular_velocities[23].spring_id == 913

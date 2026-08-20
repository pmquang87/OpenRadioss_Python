"""Tests for Milestone M286: LadStr Failure Model, EngMagneticEnergy, SwashPlateJoint, and SensorSpringAngularAcceleration."""

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


def test_m286_fail_lad_str_fixed(tmp_path: Path):
    c1 = f"{120.0:>20.4f}{45.0:>20.4f}{350.0:>20.4f}{110.0:>20.4f}{0.42:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{970:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Stress Failure Model Fixed Format Test
2022 0
/FAIL/LAD_STR/970
Ladeveze Stress Damage Criterion
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 970 in model.fail_ladstrs
    fls = model.fail_ladstrs[970]
    assert pytest.approx(fls.r0_1) == 120.0
    assert pytest.approx(fls.r0_2) == 45.0
    assert pytest.approx(fls.rc_1) == 350.0
    assert pytest.approx(fls.rc_2) == 110.0
    assert pytest.approx(fls.b_lad) == 0.42
    assert fls.ifail_sh == 1
    assert fls.ifail_so == 2
    assert fls.fail_id == 970
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_STR"


def test_m286_fail_lad_str_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Stress Free Format Test
/FAIL/LAD_STR/971
115.0, 40.0, 320.0, 105.0, 0.38
1, 1
971
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 971 in model.fail_ladstrs
    fls = model.fail_ladstrs[971]
    assert pytest.approx(fls.r0_1) == 115.0
    assert pytest.approx(fls.b_lad) == 0.38
    assert fls.fail_id == 971


def test_m286_fail_lad_str_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Stress Aliases Test
/FAIL/LADEVEZE_STRESS/972
100.0, 30.0, 300.0, 90.0, 0.35
1, 1
/FAIL/LAD_STR_MODEL/973
100.0, 30.0, 300.0, 90.0, 0.35
1, 1
/FAIL/LAD_STR_LAW/974
100.0, 30.0, 300.0, 90.0, 0.35
1, 1
/FAIL/LADEVEZE_STRESS_DAMAGE/975
100.0, 30.0, 300.0, 90.0, 0.35
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 972 in model.fail_ladstrs
    assert 973 in model.fail_ladstrs
    assert 974 in model.fail_ladstrs
    assert 975 in model.fail_ladstrs
    assert len(model.raw_fails) == 4


def test_m286_eng_magnetic_energy(tmp_path: Path):
    c1 = f"{0.00055:>20.6f}{10:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Magnetic Energy Fixed and Free Format Test
2022 0
/ENG/MAGNETIC_ENERGY/1
Fixed Magnetic Energy Output
{c1}
/ENG/MAGNETIC_ENERGY/2
Free Magnetic Energy Output
0.0011, 20
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_magnetic_energies
    assert 2 in model.eng_magnetic_energies
    me1 = model.eng_magnetic_energies[1]
    assert pytest.approx(me1.dt_mag) == 0.00055
    assert me1.sens_id == 10
    me2 = model.eng_magnetic_energies[2]
    assert pytest.approx(me2.dt_mag) == 0.0011
    assert me2.sens_id == 20


def test_m286_eng_magnetic_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Magnetic Energy Aliases Test
/ENG/MAGNETIC_WORK/11
0.001, 10
/ENG/EMAGNETIC/12
0.002, 11
/ENG/MAG_ENERGY/13
0.003, 12
/ENG/ELECTROMAGNETIC_ENERGY/14
0.004, 13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.eng_magnetic_energies
    assert 12 in model.eng_magnetic_energies
    assert 13 in model.eng_magnetic_energies
    assert 14 in model.eng_magnetic_energies
    assert pytest.approx(model.eng_magnetic_energies[11].dt_mag) == 0.001
    assert pytest.approx(model.eng_magnetic_energies[12].dt_mag) == 0.002
    assert pytest.approx(model.eng_magnetic_energies[13].dt_mag) == 0.003
    assert pytest.approx(model.eng_magnetic_energies[14].dt_mag) == 0.004


def test_m286_swash_plate_joint(tmp_path: Path):
    c1 = f"{181:>10d}{182:>10d}{183:>10d}{7.2e6:>20.4f}{2:>10d}{1.4e-6:>20.6e}"
    c2 = f"{60.0:>20.4f}{12.5:>20.4f}{1.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Swash Plate Joint Fixed and Free Format Test
2022 0
/LAGMUL/SWASH_PLATE_JOINT/1
Fixed Swash Plate Joint
{c1}
{c2}
/LAGMUL/SWASH_PLATE_JOINT/2
Free Swash Plate Joint
281, 282, 283, 8.4e6, 3, 1.8e-6
75.0, 15.0, 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_swash_plate_joints
    assert 2 in model.lagmul_swash_plate_joints
    sp1 = model.lagmul_swash_plate_joints[1]
    assert sp1.node1 == 181
    assert sp1.node2 == 182
    assert sp1.node3 == 183
    assert pytest.approx(sp1.stiff) == 7.2e6
    assert sp1.skew_id == 2
    assert pytest.approx(sp1.tol) == 1.4e-6
    assert pytest.approx(sp1.plate_radius) == 60.0
    assert pytest.approx(sp1.tilt_angle) == 12.5
    assert pytest.approx(sp1.axis_z) == 1.0

    sp2 = model.lagmul_swash_plate_joints[2]
    assert sp2.node1 == 281
    assert sp2.node2 == 282
    assert sp2.node3 == 283
    assert pytest.approx(sp2.stiff) == 8.4e6
    assert sp2.skew_id == 3
    assert pytest.approx(sp2.tol) == 1.8e-6
    assert pytest.approx(sp2.plate_radius) == 75.0
    assert pytest.approx(sp2.tilt_angle) == 15.0
    assert pytest.approx(sp2.axis_z) == 1.0


def test_m286_swash_plate_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Swash Plate Joint Aliases Test
/SWASH_PLATE_JOINT/95
921, 922, 923, 1.0e6, 0, 1.0e-6
50.0, 10.0, 1.0
/LAGMUL/SWASH_PLATE/96
924, 925, 926, 1.0e6, 0, 1.0e-6
50.0, 10.0, 1.0
/SWASH_PLATE/97
927, 928, 929, 1.0e6, 0, 1.0e-6
50.0, 10.0, 1.0
/SWASH_PLATE_MECHANISM/98
930, 931, 932, 1.0e6, 0, 1.0e-6
50.0, 10.0, 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 95 in model.lagmul_swash_plate_joints
    assert 96 in model.lagmul_swash_plate_joints
    assert 97 in model.lagmul_swash_plate_joints
    assert 98 in model.lagmul_swash_plate_joints
    assert model.lagmul_swash_plate_joints[95].node1 == 921
    assert model.lagmul_swash_plate_joints[96].node1 == 924
    assert model.lagmul_swash_plate_joints[97].node1 == 927
    assert model.lagmul_swash_plate_joints[98].node1 == 930


def test_m286_sensor_spring_angular_acceleration(tmp_path: Path):
    c1 = f"{821:>10d}{1500.0:>20.4f}{0.0085:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Angular Acceleration Fixed and Free Format Test
2022 0
/SENSOR/SPRING_ANGULAR_ACCELERATION/1
Fixed Spring Angular Acceleration Sensor
{c1}
/SENSOR/SPRING_ANGULAR_ACCELERATION/2
Free Spring Angular Acceleration Sensor
822, 1800.0, 0.0105
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_angular_accelerations
    assert 2 in model.sensor_spring_angular_accelerations
    s1 = model.sensor_spring_angular_accelerations[1]
    assert s1.spring_id == 821
    assert pytest.approx(s1.alpha_max) == 1500.0
    assert pytest.approx(s1.t_delay) == 0.0085

    s2 = model.sensor_spring_angular_accelerations[2]
    assert s2.spring_id == 822
    assert pytest.approx(s2.alpha_max) == 1800.0
    assert pytest.approx(s2.t_delay) == 0.0105

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_ANGULAR_ACCELERATION"
    assert model.sensors[1].kind == "SPRING_ANGULAR_ACCELERATION"


def test_m286_sensor_spring_angular_acceleration_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Angular Acceleration Aliases Test
/SENSOR/SPRING_ANG_ACC/31
921, 1200.0, 0.001
/SENSOR/SPRING_ALPHA/32
922, 1400.0, 0.002
/SENSOR/ANGULAR_ACCELERATION_SPRING/33
923, 1600.0, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 31 in model.sensor_spring_angular_accelerations
    assert 32 in model.sensor_spring_angular_accelerations
    assert 33 in model.sensor_spring_angular_accelerations
    assert model.sensor_spring_angular_accelerations[31].spring_id == 921
    assert pytest.approx(model.sensor_spring_angular_accelerations[31].alpha_max) == 1200.0
    assert model.sensor_spring_angular_accelerations[32].spring_id == 922
    assert model.sensor_spring_angular_accelerations[33].spring_id == 923

"""Tests for Milestone M290: LadMicro Failure Model, EngLorentzForceEnergy, SphericalWristJoint, and SensorSpringResultantAcceleration."""

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


def test_m290_fail_lad_micro_fixed(tmp_path: Path):
    c1 = f"{0.025:>20.4f}{0.850:>20.4f}{1.250:>20.4f}{0.650:>20.4f}{0.0035:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{998:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Micromechanics Failure Model Fixed Format Test
2022 0
/FAIL/LAD_MICRO/998
Ladeveze Micromechanical Damage Evolution
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 998 in model.fail_ladmicros
    flm = model.fail_ladmicros[998]
    assert pytest.approx(flm.d0_micro) == 0.025
    assert pytest.approx(flm.dc_micro) == 0.850
    assert pytest.approx(flm.alpha_micro) == 1.250
    assert pytest.approx(flm.beta_micro) == 0.650
    assert pytest.approx(flm.s_micro) == 0.0035
    assert flm.ifail_sh == 1
    assert flm.ifail_so == 2
    assert flm.fail_id == 998
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_MICRO"


def test_m290_fail_lad_micro_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Micromechanics Free Format Test
/FAIL/LAD_MICRO/999
0.020, 0.800, 1.200, 0.600, 0.0030
1, 1
999
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 999 in model.fail_ladmicros
    flm = model.fail_ladmicros[999]
    assert pytest.approx(flm.d0_micro) == 0.020
    assert pytest.approx(flm.s_micro) == 0.0030
    assert flm.fail_id == 999


def test_m290_fail_lad_micro_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Micromechanics Aliases Test
/FAIL/LADEVEZE_MICRO/1002
0.022, 0.82, 1.22, 0.62, 0.0032
1, 1
/FAIL/LAD_MICROMECHANICS/1003
0.022, 0.82, 1.22, 0.62, 0.0032
1, 1
/FAIL/LAD_MICRO_MODEL/1004
0.022, 0.82, 1.22, 0.62, 0.0032
1, 1
/FAIL/LAD_MICRO_LAW/1005
0.022, 0.82, 1.22, 0.62, 0.0032
1, 1
/FAIL/LADEVEZE_MICROMECHANICAL_DAMAGE/1006
0.022, 0.82, 1.22, 0.62, 0.0032
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1002 in model.fail_ladmicros
    assert 1003 in model.fail_ladmicros
    assert 1004 in model.fail_ladmicros
    assert 1005 in model.fail_ladmicros
    assert 1006 in model.fail_ladmicros
    assert len(model.raw_fails) == 5


def test_m290_eng_lorentz_force_energy(tmp_path: Path):
    c1 = f"{0.00095:>20.6f}{14:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Lorentz Force Energy Fixed and Free Format Test
2022 0
/ENG/LORENTZ_FORCE_ENERGY/1
Fixed Lorentz Force Energy Output
{c1}
/ENG/LORENTZ_FORCE_ENERGY/2
Free Lorentz Force Energy Output
0.0019, 28
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_lorentz_force_energies
    assert 2 in model.eng_lorentz_force_energies
    le1 = model.eng_lorentz_force_energies[1]
    assert pytest.approx(le1.dt_lorentz) == 0.00095
    assert le1.sens_id == 14
    le2 = model.eng_lorentz_force_energies[2]
    assert pytest.approx(le2.dt_lorentz) == 0.0019
    assert le2.sens_id == 28


def test_m290_eng_lorentz_force_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Lorentz Force Energy Aliases Test
/ENG/LORENTZ_WORK/11
0.001, 10
/ENG/ELORENTZ/12
0.002, 11
/ENG/LORENTZ_ENERGY/13
0.003, 12
/ENG/EM_LORENTZ_WORK/14
0.004, 13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.eng_lorentz_force_energies
    assert 12 in model.eng_lorentz_force_energies
    assert 13 in model.eng_lorentz_force_energies
    assert 14 in model.eng_lorentz_force_energies
    assert pytest.approx(model.eng_lorentz_force_energies[11].dt_lorentz) == 0.001
    assert pytest.approx(model.eng_lorentz_force_energies[12].dt_lorentz) == 0.002
    assert pytest.approx(model.eng_lorentz_force_energies[13].dt_lorentz) == 0.003
    assert pytest.approx(model.eng_lorentz_force_energies[14].dt_lorentz) == 0.004


def test_m290_spherical_wrist_joint(tmp_path: Path):
    c1 = f"{501:>10d}{502:>10d}{503:>10d}{6.5e6:>20.4f}{1:>10d}{1.6e-6:>20.6e}"
    c2 = f"{180.0:>20.4f}{90.0:>20.4f}{135.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spherical Wrist Joint Fixed and Free Format Test
2022 0
/LAGMUL/SPHERICAL_WRIST_JOINT/1
Fixed Spherical Wrist Joint
{c1}
{c2}
/LAGMUL/SPHERICAL_WRIST_JOINT/2
Free Spherical Wrist Joint
601, 602, 603, 7.2e6, 2, 2.0e-6
200.0, 100.0, 150.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_spherical_wrist_joints
    assert 2 in model.lagmul_spherical_wrist_joints
    wj1 = model.lagmul_spherical_wrist_joints[1]
    assert wj1.node1 == 501
    assert wj1.node2 == 502
    assert wj1.node3 == 503
    assert pytest.approx(wj1.stiff) == 6.5e6
    assert wj1.skew_id == 1
    assert pytest.approx(wj1.tol) == 1.6e-6
    assert pytest.approx(wj1.roll_limit) == 180.0
    assert pytest.approx(wj1.pitch_limit) == 90.0
    assert pytest.approx(wj1.yaw_limit) == 135.0

    wj2 = model.lagmul_spherical_wrist_joints[2]
    assert wj2.node1 == 601
    assert wj2.node2 == 602
    assert wj2.node3 == 603
    assert pytest.approx(wj2.stiff) == 7.2e6
    assert wj2.skew_id == 2
    assert pytest.approx(wj2.tol) == 2.0e-6
    assert pytest.approx(wj2.roll_limit) == 200.0
    assert pytest.approx(wj2.pitch_limit) == 100.0
    assert pytest.approx(wj2.yaw_limit) == 150.0


def test_m290_spherical_wrist_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spherical Wrist Joint Aliases Test
/SPHERICAL_WRIST_JOINT/51
971, 972, 973, 1.0e6, 0, 1.0e-6
120.0, 60.0, 90.0
/LAGMUL/SPHERICAL_WRIST/52
974, 975, 976, 1.0e6, 0, 1.0e-6
120.0, 60.0, 90.0
/SPHERICAL_WRIST/53
977, 978, 979, 1.0e6, 0, 1.0e-6
120.0, 60.0, 90.0
/ROBOTIC_WRIST_JOINT/54
980, 981, 982, 1.0e6, 0, 1.0e-6
120.0, 60.0, 90.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 51 in model.lagmul_spherical_wrist_joints
    assert 52 in model.lagmul_spherical_wrist_joints
    assert 53 in model.lagmul_spherical_wrist_joints
    assert 54 in model.lagmul_spherical_wrist_joints
    assert model.lagmul_spherical_wrist_joints[51].node1 == 971
    assert model.lagmul_spherical_wrist_joints[52].node1 == 974
    assert model.lagmul_spherical_wrist_joints[53].node1 == 977
    assert model.lagmul_spherical_wrist_joints[54].node1 == 980


def test_m290_sensor_spring_resultant_acceleration(tmp_path: Path):
    c1 = f"{861:>10d}{58000.0:>20.4f}{0.0035:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Resultant Acceleration Fixed and Free Format Test
2022 0
/SENSOR/SPRING_RESULTANT_ACCELERATION/1
Fixed Spring Resultant Acceleration Sensor
{c1}
/SENSOR/SPRING_RESULTANT_ACCELERATION/2
Free Spring Resultant Acceleration Sensor
862, 64000.0, 0.0055
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_resultant_accelerations
    assert 2 in model.sensor_spring_resultant_accelerations
    s1 = model.sensor_spring_resultant_accelerations[1]
    assert s1.spring_id == 861
    assert pytest.approx(s1.accr_max) == 58000.0
    assert pytest.approx(s1.t_delay) == 0.0035

    s2 = model.sensor_spring_resultant_accelerations[2]
    assert s2.spring_id == 862
    assert pytest.approx(s2.accr_max) == 64000.0
    assert pytest.approx(s2.t_delay) == 0.0055

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_RESULTANT_ACCELERATION"
    assert model.sensors[1].kind == "SPRING_RESULTANT_ACCELERATION"


def test_m290_sensor_spring_resultant_acceleration_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Resultant Acceleration Aliases Test
/SENSOR/SPRING_RES_ACC/71
961, 40000.0, 0.001
/SENSOR/SPRING_ACC_RESULTANT/72
962, 42000.0, 0.002
/SENSOR/RESULTANT_ACCELERATION_SPRING/73
963, 44000.0, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 71 in model.sensor_spring_resultant_accelerations
    assert 72 in model.sensor_spring_resultant_accelerations
    assert 73 in model.sensor_spring_resultant_accelerations
    assert model.sensor_spring_resultant_accelerations[71].spring_id == 961
    assert pytest.approx(model.sensor_spring_resultant_accelerations[71].accr_max) == 40000.0
    assert model.sensor_spring_resultant_accelerations[72].spring_id == 962
    assert model.sensor_spring_resultant_accelerations[73].spring_id == 963

"""Tests for Milestone M299: LadFatigue Failure Model, EngThermomagneticEnergy, SarrusLinkageJoint, and SensorSpringTotalAngularJerk."""

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


def test_m299_fail_lad_fatigue_fixed(tmp_path: Path):
    c1 = f"{1.6:>20.4f}{18.5:>20.4f}{2.2:>20.4f}{0.45:>20.4f}{0.985:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1085:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Fatigue Damage Model Fixed Format Test
2022 0
/FAIL/LAD_FATIGUE/1085
Ladeveze Cyclic Fatigue Damage Criterion
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1085 in model.fail_ladfatigues
    flf = model.fail_ladfatigues[1085]
    assert pytest.approx(flf.y0_fatigue) == 1.6
    assert pytest.approx(flf.yc_fatigue) == 18.5
    assert pytest.approx(flf.beta_fatigue) == 2.2
    assert pytest.approx(flf.alpha_fatigue) == 0.45
    assert pytest.approx(flf.d_fat_max) == 0.985
    assert flf.ifail_sh == 1
    assert flf.ifail_so == 2
    assert flf.fail_id == 1085
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_FATIGUE"


def test_m299_fail_lad_fatigue_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Fatigue Damage Free Format Test
/FAIL/LAD_FATIGUE/1086
2.0, 22.0, 2.5, 0.50, 0.960
1, 1
1086
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1086 in model.fail_ladfatigues
    flf = model.fail_ladfatigues[1086]
    assert pytest.approx(flf.y0_fatigue) == 2.0
    assert pytest.approx(flf.yc_fatigue) == 22.0
    assert pytest.approx(flf.beta_fatigue) == 2.5
    assert pytest.approx(flf.alpha_fatigue) == 0.50
    assert pytest.approx(flf.d_fat_max) == 0.960
    assert flf.fail_id == 1086


def test_m299_fail_lad_fatigue_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Fatigue Damage Aliases Test
/FAIL/LADEVEZE_HIGH_CYCLE_FATIGUE/1087
1.2, 16.0, 2.0, 0.40, 0.97
1, 1
/FAIL/LAD_HCF/1088
1.2, 16.0, 2.0, 0.40, 0.97
1, 1
/FAIL/LAD_FATIGUE_MODEL/1089
1.2, 16.0, 2.0, 0.40, 0.97
1, 1
/FAIL/LAD_FATIGUE_LAW/1090
1.2, 16.0, 2.0, 0.40, 0.97
1, 1
/FAIL/LADEVEZE_CYCLIC_DAMAGE/1091
1.2, 16.0, 2.0, 0.40, 0.97
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1087 in model.fail_ladfatigues
    assert 1088 in model.fail_ladfatigues
    assert 1089 in model.fail_ladfatigues
    assert 1090 in model.fail_ladfatigues
    assert 1091 in model.fail_ladfatigues
    assert len(model.raw_fails) == 5


def test_m299_eng_thermomagnetic_energy(tmp_path: Path):
    c1 = f"{0.00025:>20.6f}{42:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Thermomagnetic Energy Fixed and Free Format Test
2022 0
/ENG/THERMOMAGNETIC_ENERGY/1
Fixed Thermomagnetic Energy Output
{c1}
/ENG/THERMOMAGNETIC_ENERGY/2
Free Thermomagnetic Energy Output
0.0005, 84
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_thermomagnetic_energies
    assert 2 in model.eng_thermomagnetic_energies
    tme1 = model.eng_thermomagnetic_energies[1]
    assert pytest.approx(tme1.dt_thermomagnetic) == 0.00025
    assert tme1.sens_id == 42
    tme2 = model.eng_thermomagnetic_energies[2]
    assert pytest.approx(tme2.dt_thermomagnetic) == 0.0005
    assert tme2.sens_id == 84


def test_m299_eng_thermomagnetic_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Thermomagnetic Energy Aliases Test
/ENG/TM_WORK/41
0.001, 40
/ENG/ETHERMOMAGNETIC/42
0.002, 41
/ENG/THERMOMAGNETIC_DISSIPATION/43
0.003, 42
/ENG/EM_THERMOMAGNETIC/44
0.004, 43
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 41 in model.eng_thermomagnetic_energies
    assert 42 in model.eng_thermomagnetic_energies
    assert 43 in model.eng_thermomagnetic_energies
    assert 44 in model.eng_thermomagnetic_energies
    assert pytest.approx(model.eng_thermomagnetic_energies[41].dt_thermomagnetic) == 0.001
    assert pytest.approx(model.eng_thermomagnetic_energies[42].dt_thermomagnetic) == 0.002
    assert pytest.approx(model.eng_thermomagnetic_energies[43].dt_thermomagnetic) == 0.003
    assert pytest.approx(model.eng_thermomagnetic_energies[44].dt_thermomagnetic) == 0.004


def test_m299_sarrus_linkage_joint(tmp_path: Path):
    c1 = f"{861:>10d}{862:>10d}{863:>10d}{8.8e6:>20.4f}{1:>10d}{1.1e-6:>20.6e}"
    c2 = f"{75.0:>20.4f}{75.0:>20.4f}{90.0:>20.4f}{150.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sarrus Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/SARRUS_LINKAGE_JOINT/1
Fixed Sarrus Linkage Joint
{c1}
{c2}
/LAGMUL/SARRUS_LINKAGE_JOINT/2
Free Sarrus Linkage Joint
961, 962, 963, 9.4e6, 2, 1.6e-6
85.0, 85.0, 90.0, 170.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_sarrus_linkage_joints
    assert 2 in model.lagmul_sarrus_linkage_joints
    sj1 = model.lagmul_sarrus_linkage_joints[1]
    assert sj1.node1 == 861
    assert sj1.node2 == 862
    assert sj1.node3 == 863
    assert pytest.approx(sj1.stiff) == 8.8e6
    assert sj1.skew_id == 1
    assert pytest.approx(sj1.tol) == 1.1e-6
    assert pytest.approx(sj1.plate_len1) == 75.0
    assert pytest.approx(sj1.plate_len2) == 75.0
    assert pytest.approx(sj1.hinge_angle) == 90.0
    assert pytest.approx(sj1.guide_travel) == 150.0

    sj2 = model.lagmul_sarrus_linkage_joints[2]
    assert sj2.node1 == 961
    assert sj2.node2 == 962
    assert sj2.node3 == 963
    assert pytest.approx(sj2.stiff) == 9.4e6
    assert sj2.skew_id == 2
    assert pytest.approx(sj2.tol) == 1.6e-6
    assert pytest.approx(sj2.plate_len1) == 85.0
    assert pytest.approx(sj2.plate_len2) == 85.0
    assert pytest.approx(sj2.hinge_angle) == 90.0
    assert pytest.approx(sj2.guide_travel) == 170.0


def test_m299_sarrus_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sarrus Linkage Joint Aliases Test
/SARRUS_LINKAGE_JOINT/51
971, 972, 973, 4.0e6, 0, 1.0e-6
60.0, 60.0, 90.0, 120.0
/LAGMUL/SARRUS_LINKAGE/52
974, 975, 976, 4.0e6, 0, 1.0e-6
60.0, 60.0, 90.0, 120.0
/SARRUS_LINKAGE/53
977, 978, 979, 4.0e6, 0, 1.0e-6
60.0, 60.0, 90.0, 120.0
/SARRUS_SPATIAL_MECHANISM/54
980, 981, 982, 4.0e6, 0, 1.0e-6
60.0, 60.0, 90.0, 120.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 51 in model.lagmul_sarrus_linkage_joints
    assert 52 in model.lagmul_sarrus_linkage_joints
    assert 53 in model.lagmul_sarrus_linkage_joints
    assert 54 in model.lagmul_sarrus_linkage_joints
    assert model.lagmul_sarrus_linkage_joints[51].node1 == 971
    assert model.lagmul_sarrus_linkage_joints[52].node1 == 974
    assert model.lagmul_sarrus_linkage_joints[53].node1 == 977
    assert model.lagmul_sarrus_linkage_joints[54].node1 == 980


def test_m299_sensor_spring_total_angular_jerk(tmp_path: Path):
    c1 = f"{948:>10d}{2.2e7:>20.4f}{0.0065:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Angular Jerk Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_JERK/1
Fixed Spring Total Angular Jerk Sensor
{c1}
/SENSOR/SPRING_TOTAL_ANGULAR_JERK/2
Free Spring Total Angular Jerk Sensor
949, 3.8e7, 0.0085
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_angular_jerks
    assert 2 in model.sensor_spring_total_angular_jerks
    s1 = model.sensor_spring_total_angular_jerks[1]
    assert s1.spring_id == 948
    assert pytest.approx(s1.jtota_max) == 2.2e7
    assert pytest.approx(s1.t_delay) == 0.0065

    s2 = model.sensor_spring_total_angular_jerks[2]
    assert s2.spring_id == 949
    assert pytest.approx(s2.jtota_max) == 3.8e7
    assert pytest.approx(s2.t_delay) == 0.0085

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_JERK"
    assert model.sensors[1].kind == "SPRING_TOTAL_ANGULAR_JERK"


def test_m299_sensor_spring_total_angular_jerk_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Angular Jerk Aliases Test
/SENSOR/SPRING_TOT_ANG_JERK/81
991, 1.7e7, 0.0022
/SENSOR/SPRING_JERK_ANG_TOT/82
992, 1.9e7, 0.0032
/SENSOR/TOTAL_ANGULAR_JERK_SPRING/83
993, 2.1e7, 0.0042
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 81 in model.sensor_spring_total_angular_jerks
    assert 82 in model.sensor_spring_total_angular_jerks
    assert 83 in model.sensor_spring_total_angular_jerks
    assert model.sensor_spring_total_angular_jerks[81].spring_id == 991
    assert pytest.approx(model.sensor_spring_total_angular_jerks[81].jtota_max) == 1.7e7
    assert model.sensor_spring_total_angular_jerks[82].spring_id == 992
    assert model.sensor_spring_total_angular_jerks[83].spring_id == 993

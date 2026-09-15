"""Tests for Milestone M295: LadViscoDamage Failure Model, EngElectrocaloricEnergy, EvansLinkageJoint, and SensorSpringShearJerk."""

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


def test_m295_fail_lad_visco_damage_fixed(tmp_path: Path):
    c1 = f"{0.00125:>20.6f}{4.5e4:>20.4f}{1.85:>20.4f}{0.150:>20.4f}{0.965:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1048:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Visco Damage Model Fixed Format Test
2022 0
/FAIL/LAD_VISCO_DAMAGE/1048
Ladeveze Rate-Dependent Microdamage Evolution
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1048 in model.fail_ladviscodamages
    flvd = model.fail_ladviscodamages[1048]
    assert pytest.approx(flvd.tau_c) == 0.00125
    assert pytest.approx(flvd.a_vd) == 4.5e4
    assert pytest.approx(flvd.n_vd) == 1.85
    assert pytest.approx(flvd.d_vd_crit) == 0.150
    assert pytest.approx(flvd.d_vd_max) == 0.965
    assert flvd.ifail_sh == 1
    assert flvd.ifail_so == 2
    assert flvd.fail_id == 1048
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_VISCO_DAMAGE"


def test_m295_fail_lad_visco_damage_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Visco Damage Free Format Test
/FAIL/LAD_VISCO_DAMAGE/1049
0.0025, 5.2e4, 2.10, 0.180, 0.940
1, 1
1049
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1049 in model.fail_ladviscodamages
    flvd = model.fail_ladviscodamages[1049]
    assert pytest.approx(flvd.tau_c) == 0.0025
    assert pytest.approx(flvd.a_vd) == 5.2e4
    assert pytest.approx(flvd.n_vd) == 2.10
    assert pytest.approx(flvd.d_vd_crit) == 0.180
    assert pytest.approx(flvd.d_vd_max) == 0.940
    assert flvd.fail_id == 1049


def test_m295_fail_lad_visco_damage_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Visco Damage Aliases Test
/FAIL/LADEVEZE_VISCO_DAMAGE/1050
0.001, 4.0e4, 1.5, 0.12, 0.95
1, 1
/FAIL/LAD_VD/1051
0.001, 4.0e4, 1.5, 0.12, 0.95
1, 1
/FAIL/LAD_VD_MODEL/1052
0.001, 4.0e4, 1.5, 0.12, 0.95
1, 1
/FAIL/LAD_VD_LAW/1053
0.001, 4.0e4, 1.5, 0.12, 0.95
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_MICRODAMAGE/1054
0.001, 4.0e4, 1.5, 0.12, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1050 in model.fail_ladviscodamages
    assert 1051 in model.fail_ladviscodamages
    assert 1052 in model.fail_ladviscodamages
    assert 1053 in model.fail_ladviscodamages
    assert 1054 in model.fail_ladviscodamages
    assert len(model.raw_fails) == 5


def test_m295_eng_electrocaloric_energy(tmp_path: Path):
    c1 = f"{0.00035:>20.6f}{32:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Electrocaloric Energy Fixed and Free Format Test
2022 0
/ENG/ELECTROCALORIC_ENERGY/1
Fixed Electrocaloric Energy Output
{c1}
/ENG/ELECTROCALORIC_ENERGY/2
Free Electrocaloric Energy Output
0.0007, 64
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrocaloric_energies
    assert 2 in model.eng_electrocaloric_energies
    ee1 = model.eng_electrocaloric_energies[1]
    assert pytest.approx(ee1.dt_electrocaloric) == 0.00035
    assert ee1.sens_id == 32
    ee2 = model.eng_electrocaloric_energies[2]
    assert pytest.approx(ee2.dt_electrocaloric) == 0.0007
    assert ee2.sens_id == 64


def test_m295_eng_electrocaloric_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Electrocaloric Energy Aliases Test
/ENG/EC_WORK/11
0.001, 10
/ENG/EELECTROCALORIC/12
0.002, 11
/ENG/ELECTROCALORIC_DISSIPATION/13
0.003, 12
/ENG/EM_ELECTROCALORIC/14
0.004, 13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.eng_electrocaloric_energies
    assert 12 in model.eng_electrocaloric_energies
    assert 13 in model.eng_electrocaloric_energies
    assert 14 in model.eng_electrocaloric_energies
    assert pytest.approx(model.eng_electrocaloric_energies[11].dt_electrocaloric) == 0.001
    assert pytest.approx(model.eng_electrocaloric_energies[12].dt_electrocaloric) == 0.002
    assert pytest.approx(model.eng_electrocaloric_energies[13].dt_electrocaloric) == 0.003
    assert pytest.approx(model.eng_electrocaloric_energies[14].dt_electrocaloric) == 0.004


def test_m295_evans_linkage_joint(tmp_path: Path):
    c1 = f"{781:>10d}{782:>10d}{783:>10d}{5.5e6:>20.4f}{1:>10d}{1.2e-6:>20.6e}"
    c2 = f"{60.0:>20.4f}{30.0:>20.4f}{90.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Evans Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/EVANS_LINKAGE_JOINT/1
Fixed Evans Linkage Joint
{c1}
{c2}
/LAGMUL/EVANS_LINKAGE_JOINT/2
Free Evans Linkage Joint
881, 882, 883, 6.8e6, 2, 1.5e-6
80.0, 40.0, 120.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_evans_linkage_joints
    assert 2 in model.lagmul_evans_linkage_joints
    ev1 = model.lagmul_evans_linkage_joints[1]
    assert ev1.node1 == 781
    assert ev1.node2 == 782
    assert ev1.node3 == 783
    assert pytest.approx(ev1.stiff) == 5.5e6
    assert ev1.skew_id == 1
    assert pytest.approx(ev1.tol) == 1.2e-6
    assert pytest.approx(ev1.ground_len) == 60.0
    assert pytest.approx(ev1.crank_len) == 30.0
    assert pytest.approx(ev1.arm_len) == 90.0

    ev2 = model.lagmul_evans_linkage_joints[2]
    assert ev2.node1 == 881
    assert ev2.node2 == 882
    assert ev2.node3 == 883
    assert pytest.approx(ev2.stiff) == 6.8e6
    assert ev2.skew_id == 2
    assert pytest.approx(ev2.tol) == 1.5e-6
    assert pytest.approx(ev2.ground_len) == 80.0
    assert pytest.approx(ev2.crank_len) == 40.0
    assert pytest.approx(ev2.arm_len) == 120.0


def test_m295_evans_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Evans Linkage Joint Aliases Test
/EVANS_LINKAGE_JOINT/11
981, 982, 983, 1.0e6, 0, 1.0e-6
50.0, 25.0, 75.0
/LAGMUL/EVANS_LINKAGE/12
984, 985, 986, 1.0e6, 0, 1.0e-6
50.0, 25.0, 75.0
/EVANS_LINKAGE/13
987, 988, 989, 1.0e6, 0, 1.0e-6
50.0, 25.0, 75.0
/EVANS_STRAIGHT_LINE_MECHANISM/14
990, 991, 992, 1.0e6, 0, 1.0e-6
50.0, 25.0, 75.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.lagmul_evans_linkage_joints
    assert 12 in model.lagmul_evans_linkage_joints
    assert 13 in model.lagmul_evans_linkage_joints
    assert 14 in model.lagmul_evans_linkage_joints
    assert model.lagmul_evans_linkage_joints[11].node1 == 981
    assert model.lagmul_evans_linkage_joints[12].node1 == 984
    assert model.lagmul_evans_linkage_joints[13].node1 == 987
    assert model.lagmul_evans_linkage_joints[14].node1 == 990


def test_m295_sensor_spring_shear_jerk(tmp_path: Path):
    c1 = f"{898:>10d}{1.8e8:>20.4f}{0.0032:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Shear Jerk Fixed and Free Format Test
2022 0
/SENSOR/SPRING_SHEAR_JERK/1
Fixed Spring Shear Jerk Sensor
{c1}
/SENSOR/SPRING_SHEAR_JERK/2
Free Spring Shear Jerk Sensor
899, 2.6e8, 0.0052
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_shear_jerks
    assert 2 in model.sensor_spring_shear_jerks
    s1 = model.sensor_spring_shear_jerks[1]
    assert s1.spring_id == 898
    assert pytest.approx(s1.js_max) == 1.8e8
    assert pytest.approx(s1.t_delay) == 0.0032

    s2 = model.sensor_spring_shear_jerks[2]
    assert s2.spring_id == 899
    assert pytest.approx(s2.js_max) == 2.6e8
    assert pytest.approx(s2.t_delay) == 0.0052

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_SHEAR_JERK"
    assert model.sensors[1].kind == "SPRING_SHEAR_JERK"


def test_m295_sensor_spring_shear_jerk_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Shear Jerk Aliases Test
/SENSOR/SPRING_SHR_JERK/51
995, 1.1e8, 0.001
/SENSOR/SPRING_JERK_SHEAR/52
996, 1.2e8, 0.002
/SENSOR/SHEAR_JERK_SPRING/53
997, 1.3e8, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 51 in model.sensor_spring_shear_jerks
    assert 52 in model.sensor_spring_shear_jerks
    assert 53 in model.sensor_spring_shear_jerks
    assert model.sensor_spring_shear_jerks[51].spring_id == 995
    assert pytest.approx(model.sensor_spring_shear_jerks[51].js_max) == 1.1e8
    assert model.sensor_spring_shear_jerks[52].spring_id == 996
    assert model.sensor_spring_shear_jerks[53].spring_id == 997

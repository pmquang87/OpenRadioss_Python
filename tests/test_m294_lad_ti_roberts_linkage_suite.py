"""Tests for Milestone M294: LadTransIsotropic Failure Model, EngMagnetostrictionEnergy, RobertsLinkageJoint, and SensorSpringNormalJerk."""

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


def test_m294_fail_lad_trans_isotropic_fixed(tmp_path: Path):
    c1 = f"{0.980:>20.4f}{0.960:>20.4f}{0.940:>20.4f}{450.0:>20.4f}{280.0:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1038:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transversely Isotropic Failure Model Fixed Format Test
2022 0
/FAIL/LAD_TRANS_ISOTROPIC/1038
Ladeveze Transverse Isotropic Ply Damage Evolution
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1038 in model.fail_ladtransisotropics
    flti = model.fail_ladtransisotropics[1038]
    assert pytest.approx(flti.d1_max) == 0.980
    assert pytest.approx(flti.d2_max) == 0.960
    assert pytest.approx(flti.d3_max) == 0.940
    assert pytest.approx(flti.y1_crit) == 450.0
    assert pytest.approx(flti.y2_crit) == 280.0
    assert flti.ifail_sh == 1
    assert flti.ifail_so == 2
    assert flti.fail_id == 1038
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANS_ISOTROPIC"


def test_m294_fail_lad_trans_isotropic_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transversely Isotropic Free Format Test
/FAIL/LAD_TRANS_ISOTROPIC/1039
0.950, 0.920, 0.900, 420.0, 260.0
1, 1
1039
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1039 in model.fail_ladtransisotropics
    flti = model.fail_ladtransisotropics[1039]
    assert pytest.approx(flti.d1_max) == 0.950
    assert pytest.approx(flti.d2_max) == 0.920
    assert pytest.approx(flti.d3_max) == 0.900
    assert flti.fail_id == 1039


def test_m294_fail_lad_trans_isotropic_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transversely Isotropic Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_ISOTROPIC/1040
0.96, 0.93, 0.91, 430.0, 270.0
1, 1
/FAIL/LAD_TI/1041
0.96, 0.93, 0.91, 430.0, 270.0
1, 1
/FAIL/LAD_TI_MODEL/1042
0.96, 0.93, 0.91, 430.0, 270.0
1, 1
/FAIL/LAD_TI_LAW/1043
0.96, 0.93, 0.91, 430.0, 270.0
1, 1
/FAIL/LADEVEZE_TRANS_ISOTROPIC_DAMAGE/1044
0.96, 0.93, 0.91, 430.0, 270.0
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1040 in model.fail_ladtransisotropics
    assert 1041 in model.fail_ladtransisotropics
    assert 1042 in model.fail_ladtransisotropics
    assert 1043 in model.fail_ladtransisotropics
    assert 1044 in model.fail_ladtransisotropics
    assert len(model.raw_fails) == 5


def test_m294_eng_magnetostriction_energy(tmp_path: Path):
    c1 = f"{0.00045:>20.6f}{28:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Magnetostriction Energy Fixed and Free Format Test
2022 0
/ENG/MAGNETOSTRICTION_ENERGY/1
Fixed Magnetostriction Energy Output
{c1}
/ENG/MAGNETOSTRICTION_ENERGY/2
Free Magnetostriction Energy Output
0.0009, 56
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_magnetostriction_energies
    assert 2 in model.eng_magnetostriction_energies
    me1 = model.eng_magnetostriction_energies[1]
    assert pytest.approx(me1.dt_magnetostriction) == 0.00045
    assert me1.sens_id == 28
    me2 = model.eng_magnetostriction_energies[2]
    assert pytest.approx(me2.dt_magnetostriction) == 0.0009
    assert me2.sens_id == 56


def test_m294_eng_magnetostriction_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Magnetostriction Energy Aliases Test
/ENG/MAG_STRICT_WORK/11
0.001, 10
/ENG/EMAGSTRICT/12
0.002, 11
/ENG/MAGNETOSTRICTIVE_ENERGY/13
0.003, 12
/ENG/EM_MAGNETOSTRICTION/14
0.004, 13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.eng_magnetostriction_energies
    assert 12 in model.eng_magnetostriction_energies
    assert 13 in model.eng_magnetostriction_energies
    assert 14 in model.eng_magnetostriction_energies
    assert pytest.approx(model.eng_magnetostriction_energies[11].dt_magnetostriction) == 0.001
    assert pytest.approx(model.eng_magnetostriction_energies[12].dt_magnetostriction) == 0.002
    assert pytest.approx(model.eng_magnetostriction_energies[13].dt_magnetostriction) == 0.003
    assert pytest.approx(model.eng_magnetostriction_energies[14].dt_magnetostriction) == 0.004


def test_m294_roberts_linkage_joint(tmp_path: Path):
    c1 = f"{771:>10d}{772:>10d}{773:>10d}{6.5e6:>20.4f}{1:>10d}{1.4e-6:>20.6e}"
    c2 = f"{120.0:>20.4f}{100.0:>20.4f}{80.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Roberts Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/ROBERTS_LINKAGE_JOINT/1
Fixed Roberts Linkage Joint
{c1}
{c2}
/LAGMUL/ROBERTS_LINKAGE_JOINT/2
Free Roberts Linkage Joint
871, 872, 873, 7.2e6, 2, 1.8e-6
150.0, 125.0, 100.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_roberts_linkage_joints
    assert 2 in model.lagmul_roberts_linkage_joints
    rb1 = model.lagmul_roberts_linkage_joints[1]
    assert rb1.node1 == 771
    assert rb1.node2 == 772
    assert rb1.node3 == 773
    assert pytest.approx(rb1.stiff) == 6.5e6
    assert rb1.skew_id == 1
    assert pytest.approx(rb1.tol) == 1.4e-6
    assert pytest.approx(rb1.base_len) == 120.0
    assert pytest.approx(rb1.arm_len) == 100.0
    assert pytest.approx(rb1.coupler_height) == 80.0

    rb2 = model.lagmul_roberts_linkage_joints[2]
    assert rb2.node1 == 871
    assert rb2.node2 == 872
    assert rb2.node3 == 873
    assert pytest.approx(rb2.stiff) == 7.2e6
    assert rb2.skew_id == 2
    assert pytest.approx(rb2.tol) == 1.8e-6
    assert pytest.approx(rb2.base_len) == 150.0
    assert pytest.approx(rb2.arm_len) == 125.0
    assert pytest.approx(rb2.coupler_height) == 100.0


def test_m294_roberts_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Roberts Linkage Joint Aliases Test
/ROBERTS_LINKAGE_JOINT/11
971, 972, 973, 1.0e6, 0, 1.0e-6
100.0, 80.0, 60.0
/LAGMUL/ROBERTS_LINKAGE/12
974, 975, 976, 1.0e6, 0, 1.0e-6
100.0, 80.0, 60.0
/ROBERTS_LINKAGE/13
977, 978, 979, 1.0e6, 0, 1.0e-6
100.0, 80.0, 60.0
/ROBERTS_STRAIGHT_LINE_MECHANISM/14
980, 981, 982, 1.0e6, 0, 1.0e-6
100.0, 80.0, 60.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.lagmul_roberts_linkage_joints
    assert 12 in model.lagmul_roberts_linkage_joints
    assert 13 in model.lagmul_roberts_linkage_joints
    assert 14 in model.lagmul_roberts_linkage_joints
    assert model.lagmul_roberts_linkage_joints[11].node1 == 971
    assert model.lagmul_roberts_linkage_joints[12].node1 == 974
    assert model.lagmul_roberts_linkage_joints[13].node1 == 977
    assert model.lagmul_roberts_linkage_joints[14].node1 == 980


def test_m294_sensor_spring_normal_jerk(tmp_path: Path):
    c1 = f"{896:>10d}{1.5e8:>20.4f}{0.0025:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Jerk Fixed and Free Format Test
2022 0
/SENSOR/SPRING_NORMAL_JERK/1
Fixed Spring Normal Jerk Sensor
{c1}
/SENSOR/SPRING_NORMAL_JERK/2
Free Spring Normal Jerk Sensor
897, 2.2e8, 0.0045
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_jerks
    assert 2 in model.sensor_spring_normal_jerks
    s1 = model.sensor_spring_normal_jerks[1]
    assert s1.spring_id == 896
    assert pytest.approx(s1.jn_max) == 1.5e8
    assert pytest.approx(s1.t_delay) == 0.0025

    s2 = model.sensor_spring_normal_jerks[2]
    assert s2.spring_id == 897
    assert pytest.approx(s2.jn_max) == 2.2e8
    assert pytest.approx(s2.t_delay) == 0.0045

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_NORMAL_JERK"
    assert model.sensors[1].kind == "SPRING_NORMAL_JERK"


def test_m294_sensor_spring_normal_jerk_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Jerk Aliases Test
/SENSOR/SPRING_NORM_JERK/61
996, 1.2e8, 0.001
/SENSOR/SPRING_JERK_NORMAL/62
997, 1.3e8, 0.002
/SENSOR/NORMAL_JERK_SPRING/63
998, 1.4e8, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 61 in model.sensor_spring_normal_jerks
    assert 62 in model.sensor_spring_normal_jerks
    assert 63 in model.sensor_spring_normal_jerks
    assert model.sensor_spring_normal_jerks[61].spring_id == 996
    assert pytest.approx(model.sensor_spring_normal_jerks[61].jn_max) == 1.2e8
    assert model.sensor_spring_normal_jerks[62].spring_id == 997
    assert model.sensor_spring_normal_jerks[63].spring_id == 998

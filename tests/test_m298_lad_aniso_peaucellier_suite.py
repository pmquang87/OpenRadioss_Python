"""Tests for Milestone M298: LadAniso Failure Model, EngPyroelectricEnergy, PeaucellierLinkageJoint, and SensorSpringBendingJerk."""

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


def test_m298_fail_lad_aniso_fixed(tmp_path: Path):
    c1 = f"{1.4:>20.4f}{16.5:>20.4f}{2.8:>20.4f}{30.0:>20.4f}{0.975:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1075:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Anisotropic Damage Model Fixed Format Test
2022 0
/FAIL/LAD_ANISO/1075
Ladeveze 3D Anisotropic Damage Criterion
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1075 in model.fail_ladanisos
    fla = model.fail_ladanisos[1075]
    assert pytest.approx(fla.y0_1) == 1.4
    assert pytest.approx(fla.yc_1) == 16.5
    assert pytest.approx(fla.y0_2) == 2.8
    assert pytest.approx(fla.yc_2) == 30.0
    assert pytest.approx(fla.d_aniso_max) == 0.975
    assert fla.ifail_sh == 1
    assert fla.ifail_so == 2
    assert fla.fail_id == 1075
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_ANISO"


def test_m298_fail_lad_aniso_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Anisotropic Damage Free Format Test
/FAIL/LAD_ANISO/1076
1.8, 20.0, 3.5, 36.0, 0.950
1, 1
1076
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1076 in model.fail_ladanisos
    fla = model.fail_ladanisos[1076]
    assert pytest.approx(fla.y0_1) == 1.8
    assert pytest.approx(fla.yc_1) == 20.0
    assert pytest.approx(fla.y0_2) == 3.5
    assert pytest.approx(fla.yc_2) == 36.0
    assert pytest.approx(fla.d_aniso_max) == 0.950
    assert fla.fail_id == 1076


def test_m298_fail_lad_aniso_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Anisotropic Damage Aliases Test
/FAIL/LADEVEZE_ANISOTROPIC_DAMAGE/1077
1.0, 14.0, 2.5, 26.0, 0.96
1, 1
/FAIL/LAD_ANISOTROPIC/1078
1.0, 14.0, 2.5, 26.0, 0.96
1, 1
/FAIL/LAD_ANISO_MODEL/1079
1.0, 14.0, 2.5, 26.0, 0.96
1, 1
/FAIL/LAD_ANISO_LAW/1080
1.0, 14.0, 2.5, 26.0, 0.96
1, 1
/FAIL/LADEVEZE_ANISO_DAMAGE/1081
1.0, 14.0, 2.5, 26.0, 0.96
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1077 in model.fail_ladanisos
    assert 1078 in model.fail_ladanisos
    assert 1079 in model.fail_ladanisos
    assert 1080 in model.fail_ladanisos
    assert 1081 in model.fail_ladanisos
    assert len(model.raw_fails) == 5


def test_m298_eng_pyroelectric_energy(tmp_path: Path):
    c1 = f"{0.00035:>20.6f}{48:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Pyroelectric Energy Fixed and Free Format Test
2022 0
/ENG/PYROELECTRIC_ENERGY/1
Fixed Pyroelectric Energy Output
{c1}
/ENG/PYROELECTRIC_ENERGY/2
Free Pyroelectric Energy Output
0.0007, 96
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_pyroelectric_energies
    assert 2 in model.eng_pyroelectric_energies
    pe1 = model.eng_pyroelectric_energies[1]
    assert pytest.approx(pe1.dt_pyroelectric) == 0.00035
    assert pe1.sens_id == 48
    pe2 = model.eng_pyroelectric_energies[2]
    assert pytest.approx(pe2.dt_pyroelectric) == 0.0007
    assert pe2.sens_id == 96


def test_m298_eng_pyroelectric_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Pyroelectric Energy Aliases Test
/ENG/PYRO_WORK/31
0.001, 30
/ENG/EPYROELECTRIC/32
0.002, 31
/ENG/PYROELECTRIC_DISSIPATION/33
0.003, 32
/ENG/EM_PYROELECTRIC/34
0.004, 33
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 31 in model.eng_pyroelectric_energies
    assert 32 in model.eng_pyroelectric_energies
    assert 33 in model.eng_pyroelectric_energies
    assert 34 in model.eng_pyroelectric_energies
    assert pytest.approx(model.eng_pyroelectric_energies[31].dt_pyroelectric) == 0.001
    assert pytest.approx(model.eng_pyroelectric_energies[32].dt_pyroelectric) == 0.002
    assert pytest.approx(model.eng_pyroelectric_energies[33].dt_pyroelectric) == 0.003
    assert pytest.approx(model.eng_pyroelectric_energies[34].dt_pyroelectric) == 0.004


def test_m298_peaucellier_linkage_joint(tmp_path: Path):
    c1 = f"{851:>10d}{852:>10d}{853:>10d}{9.0e6:>20.4f}{1:>10d}{1.2e-6:>20.6e}"
    c2 = f"{60.0:>20.4f}{45.0:>20.4f}{110.0:>20.4f}{10075.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Peaucellier Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/PEAUCELLIER_LINKAGE_JOINT/1
Fixed Peaucellier Linkage Joint
{c1}
{c2}
/LAGMUL/PEAUCELLIER_LINKAGE_JOINT/2
Free Peaucellier Linkage Joint
951, 952, 953, 9.8e6, 2, 1.5e-6
70.0, 50.0, 130.0, 14400.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_peaucellier_linkage_joints
    assert 2 in model.lagmul_peaucellier_linkage_joints
    pj1 = model.lagmul_peaucellier_linkage_joints[1]
    assert pj1.node1 == 851
    assert pj1.node2 == 852
    assert pj1.node3 == 853
    assert pytest.approx(pj1.stiff) == 9.0e6
    assert pj1.skew_id == 1
    assert pytest.approx(pj1.tol) == 1.2e-6
    assert pytest.approx(pj1.base_len) == 60.0
    assert pytest.approx(pj1.rhombus_len) == 45.0
    assert pytest.approx(pj1.long_link_len) == 110.0
    assert pytest.approx(pj1.inversor_k) == 10075.0

    pj2 = model.lagmul_peaucellier_linkage_joints[2]
    assert pj2.node1 == 951
    assert pj2.node2 == 952
    assert pj2.node3 == 953
    assert pytest.approx(pj2.stiff) == 9.8e6
    assert pj2.skew_id == 2
    assert pytest.approx(pj2.tol) == 1.5e-6
    assert pytest.approx(pj2.base_len) == 70.0
    assert pytest.approx(pj2.rhombus_len) == 50.0
    assert pytest.approx(pj2.long_link_len) == 130.0
    assert pytest.approx(pj2.inversor_k) == 14400.0


def test_m298_peaucellier_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Peaucellier Linkage Joint Aliases Test
/PEAUCELLIER_LINKAGE_JOINT/41
961, 962, 963, 3.0e6, 0, 1.0e-6
50.0, 40.0, 100.0, 8400.0
/LAGMUL/PEAUCELLIER_LINKAGE/42
964, 965, 966, 3.0e6, 0, 1.0e-6
50.0, 40.0, 100.0, 8400.0
/PEAUCELLIER_LINKAGE/43
967, 968, 969, 3.0e6, 0, 1.0e-6
50.0, 40.0, 100.0, 8400.0
/PEAUCELLIER_LIPKIN_INVERSOR/44
970, 971, 972, 3.0e6, 0, 1.0e-6
50.0, 40.0, 100.0, 8400.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 41 in model.lagmul_peaucellier_linkage_joints
    assert 42 in model.lagmul_peaucellier_linkage_joints
    assert 43 in model.lagmul_peaucellier_linkage_joints
    assert 44 in model.lagmul_peaucellier_linkage_joints
    assert model.lagmul_peaucellier_linkage_joints[41].node1 == 961
    assert model.lagmul_peaucellier_linkage_joints[42].node1 == 964
    assert model.lagmul_peaucellier_linkage_joints[43].node1 == 967
    assert model.lagmul_peaucellier_linkage_joints[44].node1 == 970


def test_m298_sensor_spring_bending_jerk(tmp_path: Path):
    c1 = f"{938:>10d}{1.8e7:>20.4f}{0.0055:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Jerk Fixed and Free Format Test
2022 0
/SENSOR/SPRING_BENDING_JERK/1
Fixed Spring Bending Jerk Sensor
{c1}
/SENSOR/SPRING_BENDING_JERK/2
Free Spring Bending Jerk Sensor
939, 3.2e7, 0.0075
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_jerks
    assert 2 in model.sensor_spring_bending_jerks
    s1 = model.sensor_spring_bending_jerks[1]
    assert s1.spring_id == 938
    assert pytest.approx(s1.jbend_max) == 1.8e7
    assert pytest.approx(s1.t_delay) == 0.0055

    s2 = model.sensor_spring_bending_jerks[2]
    assert s2.spring_id == 939
    assert pytest.approx(s2.jbend_max) == 3.2e7
    assert pytest.approx(s2.t_delay) == 0.0075

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_BENDING_JERK"
    assert model.sensors[1].kind == "SPRING_BENDING_JERK"


def test_m298_sensor_spring_bending_jerk_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Jerk Aliases Test
/SENSOR/SPRING_BEND_JERK/71
985, 1.4e7, 0.0018
/SENSOR/SPRING_JERK_BEND/72
986, 1.6e7, 0.0028
/SENSOR/BENDING_JERK_SPRING/73
987, 1.8e7, 0.0038
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 71 in model.sensor_spring_bending_jerks
    assert 72 in model.sensor_spring_bending_jerks
    assert 73 in model.sensor_spring_bending_jerks
    assert model.sensor_spring_bending_jerks[71].spring_id == 985
    assert pytest.approx(model.sensor_spring_bending_jerks[71].jbend_max) == 1.4e7
    assert model.sensor_spring_bending_jerks[72].spring_id == 986
    assert model.sensor_spring_bending_jerks[73].spring_id == 987

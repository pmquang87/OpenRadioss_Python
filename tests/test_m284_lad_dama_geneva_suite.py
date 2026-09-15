"""Tests for Milestone M284: LadDama Failure Model, EngInternalPressure, GenevaJoint, and SensorSpringTotalMoment."""

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


def test_m284_fail_lad_dama_fixed(tmp_path: Path):
    c1 = f"{12000.0:>20.4f}{14000.0:>20.4f}{16000.0:>20.4f}{0.15:>20.4f}{0.25:>20.4f}"
    c2 = f"{5.0:>20.4f}{50.0:>20.4f}{0.5:>20.4f}{1.2:>20.4f}{450.0:>20.4f}"
    c3 = f"{1:>10d}{2:>10d}"
    c4 = f"{940:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
LadDama Failure Model Fixed Format Test
2022 0
/FAIL/LAD_DAMA/940
Ladeveze Damage Failure Model
{c1}
{c2}
{c3}
{c4}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 940 in model.fail_laddamas
    fd = model.fail_laddamas[940]
    assert pytest.approx(fd.k1) == 12000.0
    assert pytest.approx(fd.k2) == 14000.0
    assert pytest.approx(fd.k3) == 16000.0
    assert pytest.approx(fd.gamma1) == 0.15
    assert pytest.approx(fd.gamma2) == 0.25
    assert pytest.approx(fd.y0) == 5.0
    assert pytest.approx(fd.yc) == 50.0
    assert pytest.approx(fd.k_lad) == 0.5
    assert pytest.approx(fd.a_dama) == 1.2
    assert pytest.approx(fd.tau_max) == 450.0
    assert fd.ifail_sh == 1
    assert fd.ifail_so == 2
    assert fd.fail_id == 940
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DAMA"


def test_m284_fail_lad_dama_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
LadDama Free Format Test
/FAIL/LAD_DAMA/941
10000.0, 11000.0, 12000.0, 0.1, 0.2
4.0, 45.0, 0.4, 1.1, 400.0
1, 1
941
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 941 in model.fail_laddamas
    fd = model.fail_laddamas[941]
    assert pytest.approx(fd.k1) == 10000.0
    assert pytest.approx(fd.tau_max) == 400.0
    assert fd.fail_id == 941


def test_m284_fail_lad_dama_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
LadDama Aliases Test
/FAIL/LADEVEZE_DAMAGE/942
1000.0, 1000.0, 1000.0, 0.1, 0.1
1.0, 10.0, 0.1, 1.0, 100.0
1, 1
/FAIL/LAD_DAMA_MODEL/943
1000.0, 1000.0, 1000.0, 0.1, 0.1
1.0, 10.0, 0.1, 1.0, 100.0
1, 1
/FAIL/LAD_DAMA_LAW/944
1000.0, 1000.0, 1000.0, 0.1, 0.1
1.0, 10.0, 0.1, 1.0, 100.0
1, 1
/FAIL/LADEVEZE_DELAMINATION/945
1000.0, 1000.0, 1000.0, 0.1, 0.1
1.0, 10.0, 0.1, 1.0, 100.0
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 942 in model.fail_laddamas
    assert 943 in model.fail_laddamas
    assert 944 in model.fail_laddamas
    assert 945 in model.fail_laddamas
    assert len(model.raw_fails) == 4


def test_m284_eng_internal_pressure(tmp_path: Path):
    c1 = f"{0.00045:>20.6f}{9:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Internal Pressure Fixed and Free Format Test
2022 0
/ENG/INTERNAL_PRESSURE/1
Fixed Internal Pressure
{c1}
/ENG/INTERNAL_PRESSURE/2
Free Internal Pressure
0.0009, 18
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_internal_pressures
    assert 2 in model.eng_internal_pressures
    ip1 = model.eng_internal_pressures[1]
    assert pytest.approx(ip1.dt_pres) == 0.00045
    assert ip1.sens_id == 9
    ip2 = model.eng_internal_pressures[2]
    assert pytest.approx(ip2.dt_pres) == 0.0009
    assert ip2.sens_id == 18


def test_m284_eng_internal_pressure_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Internal Pressure Aliases Test
/ENG/INT_PRESSURE/11
0.001, 10
/ENG/EINT_PRES/12
0.002, 11
/ENG/INTERNAL_PRES/13
0.003, 12
/ENG/HYDROSTATIC_INT_PRESSURE/14
0.004, 13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.eng_internal_pressures
    assert 12 in model.eng_internal_pressures
    assert 13 in model.eng_internal_pressures
    assert 14 in model.eng_internal_pressures
    assert pytest.approx(model.eng_internal_pressures[11].dt_pres) == 0.001
    assert pytest.approx(model.eng_internal_pressures[12].dt_pres) == 0.002
    assert pytest.approx(model.eng_internal_pressures[13].dt_pres) == 0.003
    assert pytest.approx(model.eng_internal_pressures[14].dt_pres) == 0.004


def test_m284_geneva_joint(tmp_path: Path):
    c1 = f"{151:>10d}{152:>10d}{153:>10d}{7.5e6:>20.4f}{3:>10d}{1.5e-6:>20.6e}"
    c2 = f"{6:>10d}{120.0:>20.4f}{1.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Geneva Joint Fixed and Free Format Test
2022 0
/LAGMUL/GENEVA_JOINT/1
Fixed Geneva Joint
{c1}
{c2}
/LAGMUL/GENEVA_JOINT/2
Free Geneva Joint
251, 252, 253, 8.2e6, 4, 2.5e-6
8, 150.0, 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_geneva_joints
    assert 2 in model.lagmul_geneva_joints
    gj1 = model.lagmul_geneva_joints[1]
    assert gj1.node1 == 151
    assert gj1.node2 == 152
    assert gj1.node3 == 153
    assert pytest.approx(gj1.stiff) == 7.5e6
    assert gj1.skew_id == 3
    assert pytest.approx(gj1.tol) == 1.5e-6
    assert gj1.num_slots == 6
    assert pytest.approx(gj1.crank_radius) == 120.0
    assert pytest.approx(gj1.axis_z) == 1.0

    gj2 = model.lagmul_geneva_joints[2]
    assert gj2.node1 == 251
    assert gj2.node2 == 252
    assert gj2.node3 == 253
    assert pytest.approx(gj2.stiff) == 8.2e6
    assert gj2.skew_id == 4
    assert pytest.approx(gj2.tol) == 2.5e-6
    assert gj2.num_slots == 8
    assert pytest.approx(gj2.crank_radius) == 150.0
    assert pytest.approx(gj2.axis_z) == 1.0


def test_m284_geneva_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Geneva Joint Aliases Test
/GENEVA_JOINT/71
701, 702, 703, 1.0e6, 0, 1.0e-6
4, 50.0, 1.0
/LAGMUL/GENEVA_INDEXING/72
704, 705, 706, 1.0e6, 0, 1.0e-6
4, 50.0, 1.0
/GENEVA_INDEXING/73
707, 708, 709, 1.0e6, 0, 1.0e-6
4, 50.0, 1.0
/GENEVA_INDEXING_MECHANISM/74
710, 711, 712, 1.0e6, 0, 1.0e-6
4, 50.0, 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 71 in model.lagmul_geneva_joints
    assert 72 in model.lagmul_geneva_joints
    assert 73 in model.lagmul_geneva_joints
    assert 74 in model.lagmul_geneva_joints
    assert model.lagmul_geneva_joints[71].node1 == 701
    assert model.lagmul_geneva_joints[72].node1 == 704
    assert model.lagmul_geneva_joints[73].node1 == 707
    assert model.lagmul_geneva_joints[74].node1 == 710


def test_m284_sensor_spring_total_moment(tmp_path: Path):
    c1 = f"{781:>10d}{25000.0:>20.4f}{0.0055:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Moment Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TOTAL_MOMENT/1
Fixed Spring Total Moment Sensor
{c1}
/SENSOR/SPRING_TOTAL_MOMENT/2
Free Spring Total Moment Sensor
782, 32000.0, 0.0075
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_moments
    assert 2 in model.sensor_spring_total_moments
    s1 = model.sensor_spring_total_moments[1]
    assert s1.spring_id == 781
    assert pytest.approx(s1.m_tot_max) == 25000.0
    assert pytest.approx(s1.t_delay) == 0.0055

    s2 = model.sensor_spring_total_moments[2]
    assert s2.spring_id == 782
    assert pytest.approx(s2.m_tot_max) == 32000.0
    assert pytest.approx(s2.t_delay) == 0.0075

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TOTAL_MOMENT"
    assert model.sensors[1].kind == "SPRING_TOTAL_MOMENT"


def test_m284_sensor_spring_total_moment_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Moment Aliases Test
/SENSOR/SPRING_TOT_MOMENT/11
881, 24000.0, 0.001
/SENSOR/SPRING_MOMENT_TOTAL/12
882, 28000.0, 0.002
/SENSOR/TOTAL_MOMENT_SPRING/13
883, 35000.0, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.sensor_spring_total_moments
    assert 12 in model.sensor_spring_total_moments
    assert 13 in model.sensor_spring_total_moments
    assert model.sensor_spring_total_moments[11].spring_id == 881
    assert pytest.approx(model.sensor_spring_total_moments[11].m_tot_max) == 24000.0
    assert model.sensor_spring_total_moments[12].spring_id == 882
    assert model.sensor_spring_total_moments[13].spring_id == 883

"""Tests for Milestone M288: LadInter Failure Model, EngMaxwellStressEnergy, ParallelogramJoint, and SensorSpringNormalAcceleration."""

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


def test_m288_fail_lad_inter_fixed(tmp_path: Path):
    c1 = f"{1.2e5:>20.4e}{8.5e4:>20.4e}{0.085:>20.4f}{0.450:>20.4f}{1.500:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{990:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Interfacial Failure Model Fixed Format Test
2022 0
/FAIL/LAD_INTER/990
Ladeveze Interfacial Delamination Criterion
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 990 in model.fail_ladinters
    fli = model.fail_ladinters[990]
    assert pytest.approx(fli.k_n) == 1.2e5
    assert pytest.approx(fli.k_s) == 8.5e4
    assert pytest.approx(fli.y0_inter) == 0.085
    assert pytest.approx(fli.yc_inter) == 0.450
    assert pytest.approx(fli.eta_inter) == 1.500
    assert fli.ifail_sh == 1
    assert fli.ifail_so == 2
    assert fli.fail_id == 990
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_INTER"


def test_m288_fail_lad_inter_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Interfacial Free Format Test
/FAIL/LAD_INTER/991
1.1e5, 7.8e4, 0.075, 0.420, 1.450
1, 1
991
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 991 in model.fail_ladinters
    fli = model.fail_ladinters[991]
    assert pytest.approx(fli.k_n) == 1.1e5
    assert pytest.approx(fli.eta_inter) == 1.450
    assert fli.fail_id == 991


def test_m288_fail_lad_inter_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Interfacial Aliases Test
/FAIL/LADEVEZE_INTER/992
1.0e5, 7.0e4, 0.07, 0.40, 1.40
1, 1
/FAIL/LAD_INTERFACIAL/993
1.0e5, 7.0e4, 0.07, 0.40, 1.40
1, 1
/FAIL/LAD_INTER_MODEL/994
1.0e5, 7.0e4, 0.07, 0.40, 1.40
1, 1
/FAIL/LAD_INTER_LAW/995
1.0e5, 7.0e4, 0.07, 0.40, 1.40
1, 1
/FAIL/LADEVEZE_INTERFACIAL_DAMAGE/996
1.0e5, 7.0e4, 0.07, 0.40, 1.40
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 992 in model.fail_ladinters
    assert 993 in model.fail_ladinters
    assert 994 in model.fail_ladinters
    assert 995 in model.fail_ladinters
    assert 996 in model.fail_ladinters
    assert len(model.raw_fails) == 5


def test_m288_eng_maxwell_stress_energy(tmp_path: Path):
    c1 = f"{0.00075:>20.6f}{12:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Maxwell Stress Energy Fixed and Free Format Test
2022 0
/ENG/MAXWELL_STRESS_ENERGY/1
Fixed Maxwell Stress Energy Output
{c1}
/ENG/MAXWELL_STRESS_ENERGY/2
Free Maxwell Stress Energy Output
0.0015, 24
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_maxwell_stress_energies
    assert 2 in model.eng_maxwell_stress_energies
    me1 = model.eng_maxwell_stress_energies[1]
    assert pytest.approx(me1.dt_maxwell) == 0.00075
    assert me1.sens_id == 12
    me2 = model.eng_maxwell_stress_energies[2]
    assert pytest.approx(me2.dt_maxwell) == 0.0015
    assert me2.sens_id == 24


def test_m288_eng_maxwell_stress_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Maxwell Stress Energy Aliases Test
/ENG/MAXWELL_WORK/11
0.001, 10
/ENG/EMAXWELL/12
0.002, 11
/ENG/MAXWELL_STRESS_TENSOR_WORK/13
0.003, 12
/ENG/EM_MAXWELL_ENERGY/14
0.004, 13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.eng_maxwell_stress_energies
    assert 12 in model.eng_maxwell_stress_energies
    assert 13 in model.eng_maxwell_stress_energies
    assert 14 in model.eng_maxwell_stress_energies
    assert pytest.approx(model.eng_maxwell_stress_energies[11].dt_maxwell) == 0.001
    assert pytest.approx(model.eng_maxwell_stress_energies[12].dt_maxwell) == 0.002
    assert pytest.approx(model.eng_maxwell_stress_energies[13].dt_maxwell) == 0.003
    assert pytest.approx(model.eng_maxwell_stress_energies[14].dt_maxwell) == 0.004


def test_m288_parallelogram_joint(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{8.8e6:>20.4f}{1:>10d}{1.5e-6:>20.6e}"
    c2 = f"{200.0:>20.4f}{80.0:>20.4f}{1.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Parallelogram Joint Fixed and Free Format Test
2022 0
/LAGMUL/PARALLELOGRAM_JOINT/1
Fixed Parallelogram Joint
{c1}
{c2}
/LAGMUL/PARALLELOGRAM_JOINT/2
Free Parallelogram Joint
201, 202, 203, 9.6e6, 2, 1.9e-6
250.0, 100.0, 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_parallelogram_joints
    assert 2 in model.lagmul_parallelogram_joints
    pj1 = model.lagmul_parallelogram_joints[1]
    assert pj1.node1 == 101
    assert pj1.node2 == 102
    assert pj1.node3 == 103
    assert pytest.approx(pj1.stiff) == 8.8e6
    assert pj1.skew_id == 1
    assert pytest.approx(pj1.tol) == 1.5e-6
    assert pytest.approx(pj1.link_length) == 200.0
    assert pytest.approx(pj1.link_width) == 80.0
    assert pytest.approx(pj1.axis_z) == 1.0

    pj2 = model.lagmul_parallelogram_joints[2]
    assert pj2.node1 == 201
    assert pj2.node2 == 202
    assert pj2.node3 == 203
    assert pytest.approx(pj2.stiff) == 9.6e6
    assert pj2.skew_id == 2
    assert pytest.approx(pj2.tol) == 1.9e-6
    assert pytest.approx(pj2.link_length) == 250.0
    assert pytest.approx(pj2.link_width) == 100.0
    assert pytest.approx(pj2.axis_z) == 1.0


def test_m288_parallelogram_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Parallelogram Joint Aliases Test
/PARALLELOGRAM_JOINT/71
951, 952, 953, 1.0e6, 0, 1.0e-6
150.0, 60.0, 1.0
/LAGMUL/PARALLELOGRAM_LINKAGE/72
954, 955, 956, 1.0e6, 0, 1.0e-6
150.0, 60.0, 1.0
/PARALLELOGRAM_LINKAGE/73
957, 958, 959, 1.0e6, 0, 1.0e-6
150.0, 60.0, 1.0
/PARALLELOGRAM_MECHANISM/74
960, 961, 962, 1.0e6, 0, 1.0e-6
150.0, 60.0, 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 71 in model.lagmul_parallelogram_joints
    assert 72 in model.lagmul_parallelogram_joints
    assert 73 in model.lagmul_parallelogram_joints
    assert 74 in model.lagmul_parallelogram_joints
    assert model.lagmul_parallelogram_joints[71].node1 == 951
    assert model.lagmul_parallelogram_joints[72].node1 == 954
    assert model.lagmul_parallelogram_joints[73].node1 == 957
    assert model.lagmul_parallelogram_joints[74].node1 == 960


def test_m288_sensor_spring_normal_acceleration(tmp_path: Path):
    c1 = f"{841:>10d}{45000.0:>20.4f}{0.0055:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Acceleration Fixed and Free Format Test
2022 0
/SENSOR/SPRING_NORMAL_ACCELERATION/1
Fixed Spring Normal Acceleration Sensor
{c1}
/SENSOR/SPRING_NORMAL_ACCELERATION/2
Free Spring Normal Acceleration Sensor
842, 52000.0, 0.0075
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_accelerations
    assert 2 in model.sensor_spring_normal_accelerations
    s1 = model.sensor_spring_normal_accelerations[1]
    assert s1.spring_id == 841
    assert pytest.approx(s1.accn_max) == 45000.0
    assert pytest.approx(s1.t_delay) == 0.0055

    s2 = model.sensor_spring_normal_accelerations[2]
    assert s2.spring_id == 842
    assert pytest.approx(s2.accn_max) == 52000.0
    assert pytest.approx(s2.t_delay) == 0.0075

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_NORMAL_ACCELERATION"
    assert model.sensors[1].kind == "SPRING_NORMAL_ACCELERATION"


def test_m288_sensor_spring_normal_acceleration_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Acceleration Aliases Test
/SENSOR/SPRING_NORM_ACC/51
941, 35000.0, 0.001
/SENSOR/SPRING_ACC_NORMAL/52
942, 38000.0, 0.002
/SENSOR/NORMAL_ACCELERATION_SPRING/53
943, 41000.0, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 51 in model.sensor_spring_normal_accelerations
    assert 52 in model.sensor_spring_normal_accelerations
    assert 53 in model.sensor_spring_normal_accelerations
    assert model.sensor_spring_normal_accelerations[51].spring_id == 941
    assert pytest.approx(model.sensor_spring_normal_accelerations[51].accn_max) == 35000.0
    assert model.sensor_spring_normal_accelerations[52].spring_id == 942
    assert model.sensor_spring_normal_accelerations[53].spring_id == 943

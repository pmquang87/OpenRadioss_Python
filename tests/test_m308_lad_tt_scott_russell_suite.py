"""Tests for Milestone M308: LadTransverseTension Failure Model, EngMagnonPolaritonEnergy, ScottRussellLinkageJoint, and SensorSpringTransverseAccelerationJerk."""

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


def test_m308_fail_lad_transverse_tension_fixed(tmp_path: Path):
    c1 = f"{1.95:>20.4f}{25.5:>20.4f}{1.35:>20.4f}{65.0:>20.4f}{0.984:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1175:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Tension Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_TENSION/1175
Ladeveze Transverse Tensile Microcracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1175 in model.fail_ladtransversetensions
    fltt = model.fail_ladtransversetensions[1175]
    assert pytest.approx(fltt.y0_tt) == 1.95
    assert pytest.approx(fltt.yc_tt) == 25.5
    assert pytest.approx(fltt.eta_tt) == 1.35
    assert pytest.approx(fltt.sigma_tt_max) == 65.0
    assert pytest.approx(fltt.d_tt_max) == 0.984
    assert fltt.ifail_sh == 1
    assert fltt.ifail_so == 2
    assert fltt.fail_id == 1175
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_TENSION"


def test_m308_fail_lad_transverse_tension_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Tension Free Format Test
/FAIL/LAD_TRANSVERSE_TENSION/1176
2.6, 31.0, 1.5, 75.0, 0.962
1, 1
1176
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1176 in model.fail_ladtransversetensions
    fltt = model.fail_ladtransversetensions[1176]
    assert pytest.approx(fltt.y0_tt) == 2.6
    assert pytest.approx(fltt.yc_tt) == 31.0
    assert pytest.approx(fltt.eta_tt) == 1.5
    assert pytest.approx(fltt.sigma_tt_max) == 75.0
    assert pytest.approx(fltt.d_tt_max) == 0.962
    assert fltt.fail_id == 1176


def test_m308_fail_lad_transverse_tension_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Tension Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_TENSION/1177
1.5, 20.0, 1.2, 55.0, 0.97
1, 1
/FAIL/LAD_TT/1178
1.5, 20.0, 1.2, 55.0, 0.97
1, 1
/FAIL/LAD_TT_MODEL/1179
1.5, 20.0, 1.2, 55.0, 0.97
1, 1
/FAIL/LAD_TT_LAW/1180
1.5, 20.0, 1.2, 55.0, 0.97
1, 1
/FAIL/LADEVEZE_TRANSVERSE_MICROCRACKING/1181
1.5, 20.0, 1.2, 55.0, 0.97
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1177 in model.fail_ladtransversetensions
    assert 1178 in model.fail_ladtransversetensions
    assert 1179 in model.fail_ladtransversetensions
    assert 1180 in model.fail_ladtransversetensions
    assert 1181 in model.fail_ladtransversetensions
    assert len(model.raw_fails) == 5


def test_m308_eng_magnon_polariton_energy(tmp_path: Path):
    c1 = f"{0.00019:>20.6f}{39:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Magnon Polariton Energy Fixed and Free Format Test
2022 0
/ENG/MAGNON_POLARITON_ENERGY/1
Fixed Magnon Polariton Energy Output
{c1}
/ENG/MAGNON_POLARITON_ENERGY/2
Free Magnon Polariton Energy Output
0.00038, 78
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_magnon_polariton_energies
    assert 2 in model.eng_magnon_polariton_energies
    mpe1 = model.eng_magnon_polariton_energies[1]
    assert pytest.approx(mpe1.dt_mp) == 0.00019
    assert mpe1.sens_id == 39
    mpe2 = model.eng_magnon_polariton_energies[2]
    assert pytest.approx(mpe2.dt_mp) == 0.00038
    assert mpe2.sens_id == 78


def test_m308_eng_magnon_polariton_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Magnon Polariton Energy Aliases Test
/ENG/MP_WORK/131
0.001, 130
/ENG/EMAGNON_POLARITON/132
0.002, 131
/ENG/MAGNON_POLARITON_DISSIPATION/133
0.003, 132
/ENG/EM_MAGNON_POLARITON/134
0.004, 133
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 131 in model.eng_magnon_polariton_energies
    assert 132 in model.eng_magnon_polariton_energies
    assert 133 in model.eng_magnon_polariton_energies
    assert 134 in model.eng_magnon_polariton_energies
    assert pytest.approx(model.eng_magnon_polariton_energies[131].dt_mp) == 0.001
    assert pytest.approx(model.eng_magnon_polariton_energies[132].dt_mp) == 0.002
    assert pytest.approx(model.eng_magnon_polariton_energies[133].dt_mp) == 0.003
    assert pytest.approx(model.eng_magnon_polariton_energies[134].dt_mp) == 0.004


def test_m308_scott_russell_linkage_joint(tmp_path: Path):
    c1 = f"{951:>10d}{952:>10d}{953:>10d}{9.1e6:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{50.0:>20.4f}{100.0:>20.4f}{160.0:>20.4f}{0.02:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Scott Russell Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/SCOTT_RUSSELL_LINKAGE_JOINT/1
Fixed Scott Russell Linkage Joint
{c1}
{c2}
/LAGMUL/SCOTT_RUSSELL_LINKAGE_JOINT/2
Free Scott Russell Linkage Joint
1051, 1052, 1053, 9.8e6, 2, 1.9e-6
60.0, 120.0, 190.0, 0.03
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_scott_russell_linkage_joints
    assert 2 in model.lagmul_scott_russell_linkage_joints
    srlj1 = model.lagmul_scott_russell_linkage_joints[1]
    assert srlj1.node1 == 951
    assert srlj1.node2 == 952
    assert srlj1.node3 == 953
    assert pytest.approx(srlj1.stiff) == 9.1e6
    assert srlj1.skew_id == 1
    assert pytest.approx(srlj1.tol) == 1.0e-6
    assert pytest.approx(srlj1.crank_len) == 50.0
    assert pytest.approx(srlj1.coupler_len) == 100.0
    assert pytest.approx(srlj1.stroke_travel) == 160.0
    assert pytest.approx(srlj1.slider_friction) == 0.02

    srlj2 = model.lagmul_scott_russell_linkage_joints[2]
    assert srlj2.node1 == 1051
    assert srlj2.node2 == 1052
    assert srlj2.node3 == 1053
    assert pytest.approx(srlj2.stiff) == 9.8e6
    assert srlj2.skew_id == 2
    assert pytest.approx(srlj2.tol) == 1.9e-6
    assert pytest.approx(srlj2.crank_len) == 60.0
    assert pytest.approx(srlj2.coupler_len) == 120.0
    assert pytest.approx(srlj2.stroke_travel) == 190.0
    assert pytest.approx(srlj2.slider_friction) == 0.03


def test_m308_scott_russell_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Scott Russell Linkage Joint Aliases Test
/SCOTT_RUSSELL_LINKAGE_JOINT/135
1101, 1102, 1103, 6.6e6, 0, 1.0e-6
40.0, 80.0, 120.0, 0.0
/LAGMUL/SCOTT_RUSSELL_LINKAGE/136
1104, 1105, 1106, 6.6e6, 0, 1.0e-6
40.0, 80.0, 120.0, 0.0
/SCOTT_RUSSELL_LINKAGE/137
1107, 1108, 1109, 6.6e6, 0, 1.0e-6
40.0, 80.0, 120.0, 0.0
/SCOTT_RUSSELL_EXACT_STRAIGHT_LINE_MECHANISM/138
1110, 1111, 1112, 6.6e6, 0, 1.0e-6
40.0, 80.0, 120.0, 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 135 in model.lagmul_scott_russell_linkage_joints
    assert 136 in model.lagmul_scott_russell_linkage_joints
    assert 137 in model.lagmul_scott_russell_linkage_joints
    assert 138 in model.lagmul_scott_russell_linkage_joints
    assert model.lagmul_scott_russell_linkage_joints[135].node1 == 1101
    assert model.lagmul_scott_russell_linkage_joints[136].node1 == 1104
    assert model.lagmul_scott_russell_linkage_joints[137].node1 == 1107
    assert model.lagmul_scott_russell_linkage_joints[138].node1 == 1110


def test_m308_sensor_spring_transverse_acceleration_jerk(tmp_path: Path):
    c1 = f"{1038:>10d}{3.9e7:>20.4f}{0.0112:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Acceleration Jerk Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_ACCELERATION_JERK/1
Fixed Spring Transverse Acceleration Jerk Sensor
{c1}
/SENSOR/SPRING_TRANSVERSE_ACCELERATION_JERK/2
Free Spring Transverse Acceleration Jerk Sensor
1039, 6.1e7, 0.0152
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_acceleration_jerks
    assert 2 in model.sensor_spring_transverse_acceleration_jerks
    s1 = model.sensor_spring_transverse_acceleration_jerks[1]
    assert s1.spring_id == 1038
    assert pytest.approx(s1.jtrans_rate_max) == 3.9e7
    assert pytest.approx(s1.t_delay) == 0.0112

    s2 = model.sensor_spring_transverse_acceleration_jerks[2]
    assert s2.spring_id == 1039
    assert pytest.approx(s2.jtrans_rate_max) == 6.1e7
    assert pytest.approx(s2.t_delay) == 0.0152

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_ACCELERATION_JERK"
    assert model.sensors[1].kind == "SPRING_TRANSVERSE_ACCELERATION_JERK"


def test_m308_sensor_spring_transverse_acceleration_jerk_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Acceleration Jerk Aliases Test
/SENSOR/SPRING_TRANS_ACC_JERK/116
1068, 2.65e7, 0.0032
/SENSOR/SPRING_JERK_ACC_TRANS/117
1069, 2.85e7, 0.0042
/SENSOR/TRANSVERSE_ACCELERATION_JERK_SPRING/118
1070, 3.05e7, 0.0052
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 116 in model.sensor_spring_transverse_acceleration_jerks
    assert 117 in model.sensor_spring_transverse_acceleration_jerks
    assert 118 in model.sensor_spring_transverse_acceleration_jerks
    assert model.sensor_spring_transverse_acceleration_jerks[116].spring_id == 1068
    assert pytest.approx(model.sensor_spring_transverse_acceleration_jerks[116].jtrans_rate_max) == 2.65e7
    assert model.sensor_spring_transverse_acceleration_jerks[117].spring_id == 1069
    assert model.sensor_spring_transverse_acceleration_jerks[118].spring_id == 1070

"""Tests for Milestone M307: LadFiberMatrixInteraction Failure Model, EngExcitonPolaritonEnergy, WattParallelMotionJoint, and SensorSpringNormalAccelerationJerk."""

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


def test_m307_fail_lad_fiber_matrix_interaction_fixed(tmp_path: Path):
    c1 = f"{2.15:>20.4f}{28.5:>20.4f}{1.45:>20.4f}{1.65:>20.4f}{0.982:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1165:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Fiber-Matrix Interaction Fixed Format Test
2022 0
/FAIL/LAD_FIBER_MATRIX_INTERACTION/1165
Ladeveze Multiaxial Interaction Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1165 in model.fail_ladfibermatrixinteractions
    flfmi = model.fail_ladfibermatrixinteractions[1165]
    assert pytest.approx(flfmi.y0_fmi) == 2.15
    assert pytest.approx(flfmi.yc_fmi) == 28.5
    assert pytest.approx(flfmi.alpha_fmi_trans) == 1.45
    assert pytest.approx(flfmi.beta_fmi_shear) == 1.65
    assert pytest.approx(flfmi.d_fmi_max) == 0.982
    assert flfmi.ifail_sh == 1
    assert flfmi.ifail_so == 2
    assert flfmi.fail_id == 1165
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_FIBER_MATRIX_INTERACTION"


def test_m307_fail_lad_fiber_matrix_interaction_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Fiber-Matrix Interaction Free Format Test
/FAIL/LAD_FIBER_MATRIX_INTERACTION/1166
2.8, 35.0, 1.8, 2.0, 0.955
1, 1
1166
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1166 in model.fail_ladfibermatrixinteractions
    flfmi = model.fail_ladfibermatrixinteractions[1166]
    assert pytest.approx(flfmi.y0_fmi) == 2.8
    assert pytest.approx(flfmi.yc_fmi) == 35.0
    assert pytest.approx(flfmi.alpha_fmi_trans) == 1.8
    assert pytest.approx(flfmi.beta_fmi_shear) == 2.0
    assert pytest.approx(flfmi.d_fmi_max) == 0.955
    assert flfmi.fail_id == 1166


def test_m307_fail_lad_fiber_matrix_interaction_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Fiber-Matrix Interaction Aliases Test
/FAIL/LADEVEZE_FIBER_MATRIX_INTERACTION/1167
1.6, 22.0, 1.2, 1.3, 0.97
1, 1
/FAIL/LAD_FMI/1168
1.6, 22.0, 1.2, 1.3, 0.97
1, 1
/FAIL/LAD_FMI_MODEL/1169
1.6, 22.0, 1.2, 1.3, 0.97
1, 1
/FAIL/LAD_FMI_LAW/1170
1.6, 22.0, 1.2, 1.3, 0.97
1, 1
/FAIL/LADEVEZE_MULTIAXIAL_INTERACTION/1171
1.6, 22.0, 1.2, 1.3, 0.97
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1167 in model.fail_ladfibermatrixinteractions
    assert 1168 in model.fail_ladfibermatrixinteractions
    assert 1169 in model.fail_ladfibermatrixinteractions
    assert 1170 in model.fail_ladfibermatrixinteractions
    assert 1171 in model.fail_ladfibermatrixinteractions
    assert len(model.raw_fails) == 5


def test_m307_eng_exciton_polariton_energy(tmp_path: Path):
    c1 = f"{0.00018:>20.6f}{38:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Exciton Polariton Energy Fixed and Free Format Test
2022 0
/ENG/EXCITON_POLARITON_ENERGY/1
Fixed Exciton Polariton Energy Output
{c1}
/ENG/EXCITON_POLARITON_ENERGY/2
Free Exciton Polariton Energy Output
0.00036, 76
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_exciton_polariton_energies
    assert 2 in model.eng_exciton_polariton_energies
    epe1 = model.eng_exciton_polariton_energies[1]
    assert pytest.approx(epe1.dt_ep) == 0.00018
    assert epe1.sens_id == 38
    epe2 = model.eng_exciton_polariton_energies[2]
    assert pytest.approx(epe2.dt_ep) == 0.00036
    assert epe2.sens_id == 76


def test_m307_eng_exciton_polariton_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Exciton Polariton Energy Aliases Test
/ENG/EP_WORK/121
0.001, 120
/ENG/EEXCITON_POLARITON/122
0.002, 121
/ENG/EXCITON_POLARITON_DISSIPATION/123
0.003, 122
/ENG/EM_EXCITON_POLARITON/124
0.004, 123
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 121 in model.eng_exciton_polariton_energies
    assert 122 in model.eng_exciton_polariton_energies
    assert 123 in model.eng_exciton_polariton_energies
    assert 124 in model.eng_exciton_polariton_energies
    assert pytest.approx(model.eng_exciton_polariton_energies[121].dt_ep) == 0.001
    assert pytest.approx(model.eng_exciton_polariton_energies[122].dt_ep) == 0.002
    assert pytest.approx(model.eng_exciton_polariton_energies[123].dt_ep) == 0.003
    assert pytest.approx(model.eng_exciton_polariton_energies[124].dt_ep) == 0.004


def test_m307_watt_parallel_motion_joint(tmp_path: Path):
    c1 = f"{941:>10d}{942:>10d}{943:>10d}{9.0e6:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{120.0:>20.4f}{60.0:>20.4f}{150.0:>20.4f}{25.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Watt Parallel Motion Joint Fixed and Free Format Test
2022 0
/LAGMUL/WATT_PARALLEL_MOTION_JOINT/1
Fixed Watt Parallel Motion Joint
{c1}
{c2}
/LAGMUL/WATT_PARALLEL_MOTION_JOINT/2
Free Watt Parallel Motion Joint
1041, 1042, 1043, 9.7e6, 2, 1.8e-6
140.0, 70.0, 180.0, 30.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_watt_parallel_motion_joints
    assert 2 in model.lagmul_watt_parallel_motion_joints
    wpmj1 = model.lagmul_watt_parallel_motion_joints[1]
    assert wpmj1.node1 == 941
    assert wpmj1.node2 == 942
    assert wpmj1.node3 == 943
    assert pytest.approx(wpmj1.stiff) == 9.0e6
    assert wpmj1.skew_id == 1
    assert pytest.approx(wpmj1.tol) == 1.0e-6
    assert pytest.approx(wpmj1.arm_length_main) == 120.0
    assert pytest.approx(wpmj1.arm_length_sub) == 60.0
    assert pytest.approx(wpmj1.stroke_travel) == 150.0
    assert pytest.approx(wpmj1.offset_dist) == 25.0

    wpmj2 = model.lagmul_watt_parallel_motion_joints[2]
    assert wpmj2.node1 == 1041
    assert wpmj2.node2 == 1042
    assert wpmj2.node3 == 1043
    assert pytest.approx(wpmj2.stiff) == 9.7e6
    assert wpmj2.skew_id == 2
    assert pytest.approx(wpmj2.tol) == 1.8e-6
    assert pytest.approx(wpmj2.arm_length_main) == 140.0
    assert pytest.approx(wpmj2.arm_length_sub) == 70.0
    assert pytest.approx(wpmj2.stroke_travel) == 180.0
    assert pytest.approx(wpmj2.offset_dist) == 30.0


def test_m307_watt_parallel_motion_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Watt Parallel Motion Joint Aliases Test
/WATT_PARALLEL_MOTION_JOINT/125
1091, 1092, 1093, 6.5e6, 0, 1.0e-6
100.0, 50.0, 120.0, 20.0
/LAGMUL/WATT_PARALLEL_MOTION/126
1094, 1095, 1096, 6.5e6, 0, 1.0e-6
100.0, 50.0, 120.0, 20.0
/WATT_PARALLEL_MOTION/127
1097, 1098, 1099, 6.5e6, 0, 1.0e-6
100.0, 50.0, 120.0, 20.0
/WATT_DOUBLE_PARALLELOGRAM_MECHANISM/128
1100, 1101, 1102, 6.5e6, 0, 1.0e-6
100.0, 50.0, 120.0, 20.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 125 in model.lagmul_watt_parallel_motion_joints
    assert 126 in model.lagmul_watt_parallel_motion_joints
    assert 127 in model.lagmul_watt_parallel_motion_joints
    assert 128 in model.lagmul_watt_parallel_motion_joints
    assert model.lagmul_watt_parallel_motion_joints[125].node1 == 1091
    assert model.lagmul_watt_parallel_motion_joints[126].node1 == 1094
    assert model.lagmul_watt_parallel_motion_joints[127].node1 == 1097
    assert model.lagmul_watt_parallel_motion_joints[128].node1 == 1100


def test_m307_sensor_spring_normal_acceleration_jerk(tmp_path: Path):
    c1 = f"{1028:>10d}{3.7e7:>20.4f}{0.0104:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Acceleration Jerk Fixed and Free Format Test
2022 0
/SENSOR/SPRING_NORMAL_ACCELERATION_JERK/1
Fixed Spring Normal Acceleration Jerk Sensor
{c1}
/SENSOR/SPRING_NORMAL_ACCELERATION_JERK/2
Free Spring Normal Acceleration Jerk Sensor
1029, 5.9e7, 0.0142
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_acceleration_jerks
    assert 2 in model.sensor_spring_normal_acceleration_jerks
    s1 = model.sensor_spring_normal_acceleration_jerks[1]
    assert s1.spring_id == 1028
    assert pytest.approx(s1.jnorm_rate_max) == 3.7e7
    assert pytest.approx(s1.t_delay) == 0.0104

    s2 = model.sensor_spring_normal_acceleration_jerks[2]
    assert s2.spring_id == 1029
    assert pytest.approx(s2.jnorm_rate_max) == 5.9e7
    assert pytest.approx(s2.t_delay) == 0.0142

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_NORMAL_ACCELERATION_JERK"
    assert model.sensors[1].kind == "SPRING_NORMAL_ACCELERATION_JERK"


def test_m307_sensor_spring_normal_acceleration_jerk_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Acceleration Jerk Aliases Test
/SENSOR/SPRING_NORM_ACC_JERK/113
1058, 2.55e7, 0.0031
/SENSOR/SPRING_JERK_ACC_NORM/114
1059, 2.75e7, 0.0041
/SENSOR/NORMAL_ACCELERATION_JERK_SPRING/115
1060, 2.95e7, 0.0051
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 113 in model.sensor_spring_normal_acceleration_jerks
    assert 114 in model.sensor_spring_normal_acceleration_jerks
    assert 115 in model.sensor_spring_normal_acceleration_jerks
    assert model.sensor_spring_normal_acceleration_jerks[113].spring_id == 1058
    assert pytest.approx(model.sensor_spring_normal_acceleration_jerks[113].jnorm_rate_max) == 2.55e7
    assert model.sensor_spring_normal_acceleration_jerks[114].spring_id == 1059
    assert model.sensor_spring_normal_acceleration_jerks[115].spring_id == 1060

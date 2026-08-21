"""Tests for Milestone M306: LadInterlaminarShear Failure Model, EngPhononPolaritonEnergy, PantographLinkageJoint, and SensorSpringTotalAccelerationJerk."""

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


def test_m306_fail_lad_interlaminar_shear_fixed(tmp_path: Path):
    c1 = f"{1.75:>20.4f}{26.0:>20.4f}{4200.0:>20.4f}{85.0:>20.4f}{0.985:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1155:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Interlaminar Shear Model Fixed Format Test
2022 0
/FAIL/LAD_INTERLAMINAR_SHEAR/1155
Ladeveze Interlaminar Shear Microcracking Criterion
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1155 in model.fail_ladinterlaminarshears
    flils = model.fail_ladinterlaminarshears[1155]
    assert pytest.approx(flils.y0_ils) == 1.75
    assert pytest.approx(flils.yc_ils) == 26.0
    assert pytest.approx(flils.gamma_ils_p) == 4200.0
    assert pytest.approx(flils.tau_ils_max) == 85.0
    assert pytest.approx(flils.d_ils_max) == 0.985
    assert flils.ifail_sh == 1
    assert flils.ifail_so == 2
    assert flils.fail_id == 1155
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_INTERLAMINAR_SHEAR"


def test_m306_fail_lad_interlaminar_shear_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Interlaminar Shear Free Format Test
/FAIL/LAD_INTERLAMINAR_SHEAR/1156
2.4, 32.0, 4800.0, 95.0, 0.960
1, 1
1156
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1156 in model.fail_ladinterlaminarshears
    flils = model.fail_ladinterlaminarshears[1156]
    assert pytest.approx(flils.y0_ils) == 2.4
    assert pytest.approx(flils.yc_ils) == 32.0
    assert pytest.approx(flils.gamma_ils_p) == 4800.0
    assert pytest.approx(flils.tau_ils_max) == 95.0
    assert pytest.approx(flils.d_ils_max) == 0.960
    assert flils.fail_id == 1156


def test_m306_fail_lad_interlaminar_shear_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Interlaminar Shear Aliases Test
/FAIL/LADEVEZE_INTERLAMINAR_SHEAR/1157
1.4, 21.0, 3600.0, 75.0, 0.97
1, 1
/FAIL/LAD_ILS/1158
1.4, 21.0, 3600.0, 75.0, 0.97
1, 1
/FAIL/LAD_ILS_MODEL/1159
1.4, 21.0, 3600.0, 75.0, 0.97
1, 1
/FAIL/LAD_ILS_LAW/1160
1.4, 21.0, 3600.0, 75.0, 0.97
1, 1
/FAIL/LADEVEZE_INTERLAMINAR_SHEAR_DAMAGE/1161
1.4, 21.0, 3600.0, 75.0, 0.97
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1157 in model.fail_ladinterlaminarshears
    assert 1158 in model.fail_ladinterlaminarshears
    assert 1159 in model.fail_ladinterlaminarshears
    assert 1160 in model.fail_ladinterlaminarshears
    assert 1161 in model.fail_ladinterlaminarshears
    assert len(model.raw_fails) == 5


def test_m306_eng_phonon_polariton_energy(tmp_path: Path):
    c1 = f"{0.00016:>20.6f}{36:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Phonon Polariton Energy Fixed and Free Format Test
2022 0
/ENG/PHONON_POLARITON_ENERGY/1
Fixed Phonon Polariton Energy Output
{c1}
/ENG/PHONON_POLARITON_ENERGY/2
Free Phonon Polariton Energy Output
0.00032, 72
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_phonon_polariton_energies
    assert 2 in model.eng_phonon_polariton_energies
    ppe1 = model.eng_phonon_polariton_energies[1]
    assert pytest.approx(ppe1.dt_pp) == 0.00016
    assert ppe1.sens_id == 36
    ppe2 = model.eng_phonon_polariton_energies[2]
    assert pytest.approx(ppe2.dt_pp) == 0.00032
    assert ppe2.sens_id == 72


def test_m306_eng_phonon_polariton_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Phonon Polariton Energy Aliases Test
/ENG/PP_WORK/111
0.001, 110
/ENG/EPHONON_POLARITON/112
0.002, 111
/ENG/PHONON_POLARITON_DISSIPATION/113
0.003, 112
/ENG/EM_PHONON_POLARITON/114
0.004, 113
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 111 in model.eng_phonon_polariton_energies
    assert 112 in model.eng_phonon_polariton_energies
    assert 113 in model.eng_phonon_polariton_energies
    assert 114 in model.eng_phonon_polariton_energies
    assert pytest.approx(model.eng_phonon_polariton_energies[111].dt_pp) == 0.001
    assert pytest.approx(model.eng_phonon_polariton_energies[112].dt_pp) == 0.002
    assert pytest.approx(model.eng_phonon_polariton_energies[113].dt_pp) == 0.003
    assert pytest.approx(model.eng_phonon_polariton_energies[114].dt_pp) == 0.004


def test_m306_pantograph_linkage_joint(tmp_path: Path):
    c1 = f"{931:>10d}{932:>10d}{933:>10d}{8.9e6:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{3.0:>20.4f}{40.0:>20.4f}{60.0:>20.4f}{0.785:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Pantograph Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/PANTOGRAPH_LINKAGE_JOINT/1
Fixed Pantograph Linkage Joint
{c1}
{c2}
/LAGMUL/PANTOGRAPH_LINKAGE_JOINT/2
Free Pantograph Linkage Joint
1031, 1032, 1033, 9.6e6, 2, 1.7e-6
2.5, 48.0, 72.0, 0.523
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_pantograph_linkage_joints
    assert 2 in model.lagmul_pantograph_linkage_joints
    plj1 = model.lagmul_pantograph_linkage_joints[1]
    assert plj1.node1 == 931
    assert plj1.node2 == 932
    assert plj1.node3 == 933
    assert pytest.approx(plj1.stiff) == 8.9e6
    assert plj1.skew_id == 1
    assert pytest.approx(plj1.tol) == 1.0e-6
    assert pytest.approx(plj1.scale_factor) == 3.0
    assert pytest.approx(plj1.arm_length_a) == 40.0
    assert pytest.approx(plj1.arm_length_b) == 60.0
    assert pytest.approx(plj1.cross_angle_0) == 0.785

    plj2 = model.lagmul_pantograph_linkage_joints[2]
    assert plj2.node1 == 1031
    assert plj2.node2 == 1032
    assert plj2.node3 == 1033
    assert pytest.approx(plj2.stiff) == 9.6e6
    assert plj2.skew_id == 2
    assert pytest.approx(plj2.tol) == 1.7e-6
    assert pytest.approx(plj2.scale_factor) == 2.5
    assert pytest.approx(plj2.arm_length_a) == 48.0
    assert pytest.approx(plj2.arm_length_b) == 72.0
    assert pytest.approx(plj2.cross_angle_0) == 0.523


def test_m306_pantograph_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Pantograph Linkage Joint Aliases Test
/PANTOGRAPH_LINKAGE_JOINT/115
1081, 1082, 1083, 6.4e6, 0, 1.0e-6
2.0, 30.0, 45.0, 0.0
/LAGMUL/PANTOGRAPH_LINKAGE/116
1084, 1085, 1086, 6.4e6, 0, 1.0e-6
2.0, 30.0, 45.0, 0.0
/PANTOGRAPH_LINKAGE/117
1087, 1088, 1089, 6.4e6, 0, 1.0e-6
2.0, 30.0, 45.0, 0.0
/PANTOGRAPH_SCALING_MECHANISM/118
1090, 1091, 1092, 6.4e6, 0, 1.0e-6
2.0, 30.0, 45.0, 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 115 in model.lagmul_pantograph_linkage_joints
    assert 116 in model.lagmul_pantograph_linkage_joints
    assert 117 in model.lagmul_pantograph_linkage_joints
    assert 118 in model.lagmul_pantograph_linkage_joints
    assert model.lagmul_pantograph_linkage_joints[115].node1 == 1081
    assert model.lagmul_pantograph_linkage_joints[116].node1 == 1084
    assert model.lagmul_pantograph_linkage_joints[117].node1 == 1087
    assert model.lagmul_pantograph_linkage_joints[118].node1 == 1090


def test_m306_sensor_spring_total_acceleration_jerk(tmp_path: Path):
    c1 = f"{1018:>10d}{3.5e7:>20.4f}{0.0098:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Acceleration Jerk Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TOTAL_ACCELERATION_JERK/1
Fixed Spring Total Acceleration Jerk Sensor
{c1}
/SENSOR/SPRING_TOTAL_ACCELERATION_JERK/2
Free Spring Total Acceleration Jerk Sensor
1019, 5.7e7, 0.0132
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_acceleration_jerks
    assert 2 in model.sensor_spring_total_acceleration_jerks
    s1 = model.sensor_spring_total_acceleration_jerks[1]
    assert s1.spring_id == 1018
    assert pytest.approx(s1.jtot_comb_max) == 3.5e7
    assert pytest.approx(s1.t_delay) == 0.0098

    s2 = model.sensor_spring_total_acceleration_jerks[2]
    assert s2.spring_id == 1019
    assert pytest.approx(s2.jtot_comb_max) == 5.7e7
    assert pytest.approx(s2.t_delay) == 0.0132

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TOTAL_ACCELERATION_JERK"
    assert model.sensors[1].kind == "SPRING_TOTAL_ACCELERATION_JERK"


def test_m306_sensor_spring_total_acceleration_jerk_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Acceleration Jerk Aliases Test
/SENSOR/SPRING_TOT_ACC_JERK/110
1048, 2.45e7, 0.0030
/SENSOR/SPRING_JERK_ACC_TOT/111
1049, 2.65e7, 0.0040
/SENSOR/TOTAL_ACCELERATION_JERK_SPRING/112
1050, 2.85e7, 0.0050
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 110 in model.sensor_spring_total_acceleration_jerks
    assert 111 in model.sensor_spring_total_acceleration_jerks
    assert 112 in model.sensor_spring_total_acceleration_jerks
    assert model.sensor_spring_total_acceleration_jerks[110].spring_id == 1048
    assert pytest.approx(model.sensor_spring_total_acceleration_jerks[110].jtot_comb_max) == 2.45e7
    assert model.sensor_spring_total_acceleration_jerks[111].spring_id == 1049
    assert model.sensor_spring_total_acceleration_jerks[112].spring_id == 1050

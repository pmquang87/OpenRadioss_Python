"""Tests for Milestone M326: LadCouplePlasticity Failure Model, EngFlexomagneticEnergy, DietmaierLinkageJoint, and SensorSpringTransversePopRate."""

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


def test_m326_fail_lad_couple_plasticity_fixed(tmp_path: Path):
    c1 = f"{120.0:>20.4f}{450.0:>20.4f}{0.85:>20.4f}{15.0:>20.4f}{0.980:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1370:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Couple Plasticity Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_PLASTICITY/1370
Ladeveze Coupled Damage Plasticity Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1370 in model.fail_ladcoupleplasticitys
    fcp = model.fail_ladcoupleplasticitys[1370]
    assert pytest.approx(fcp.r_p0) == 120.0
    assert pytest.approx(fcp.k_p) == 450.0
    assert pytest.approx(fcp.m_p) == 0.85
    assert pytest.approx(fcp.gamma_d) == 15.0
    assert pytest.approx(fcp.d_cp_max) == 0.980
    assert fcp.ifail_sh == 1
    assert fcp.ifail_so == 2
    assert fcp.fail_id == 1370
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_PLASTICITY"


def test_m326_fail_lad_couple_plasticity_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Couple Plasticity Free Format Test
/FAIL/LAD_COUPLE_PLASTICITY/1371
150.0, 520.0, 0.90, 20.0, 0.965
1, 1
1371
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1371 in model.fail_ladcoupleplasticitys
    fcp = model.fail_ladcoupleplasticitys[1371]
    assert pytest.approx(fcp.r_p0) == 150.0
    assert pytest.approx(fcp.k_p) == 520.0
    assert pytest.approx(fcp.m_p) == 0.90
    assert pytest.approx(fcp.gamma_d) == 20.0
    assert pytest.approx(fcp.d_cp_max) == 0.965
    assert fcp.fail_id == 1371


def test_m326_fail_lad_couple_plasticity_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Couple Plasticity Aliases Test
/FAIL/LADEVEZE_COUPLED_PLASTICITY/1372
100.0, 400.0, 1.0, 10.0, 0.95
1, 1
/FAIL/LAD_CP/1373
100.0, 400.0, 1.0, 10.0, 0.95
1, 1
/FAIL/LAD_CP_MODEL/1374
100.0, 400.0, 1.0, 10.0, 0.95
1, 1
/FAIL/LAD_CP_LAW/1375
100.0, 400.0, 1.0, 10.0, 0.95
1, 1
/FAIL/LADEVEZE_DAMAGE_PLASTICITY_COUPLING/1376
100.0, 400.0, 1.0, 10.0, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1372 in model.fail_ladcoupleplasticitys
    assert 1373 in model.fail_ladcoupleplasticitys
    assert 1374 in model.fail_ladcoupleplasticitys
    assert 1375 in model.fail_ladcoupleplasticitys
    assert 1376 in model.fail_ladcoupleplasticitys
    assert len(model.raw_fails) == 5


def test_m326_fail_lad_couple_plasticity_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_COUPLE_PLASTICITY/1377
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m326_eng_flexomagnetic_energy(tmp_path: Path):
    c1 = f"{0.00055:>20.6f}{105:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexomagnetic Energy Fixed and Free Format Test
2022 0
/ENG/FLEXOMAGNETIC_ENERGY/1
Fixed Flexomagnetic Energy Output
{c1}
/ENG/FLEXOMAGNETIC_ENERGY/2
Free Flexomagnetic Energy Output
0.00095, 210
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexomagnetic_energies
    assert 2 in model.eng_flexomagnetic_energies
    fme1 = model.eng_flexomagnetic_energies[1]
    assert pytest.approx(fme1.dt_fm) == 0.00055
    assert fme1.sens_id == 105
    fme2 = model.eng_flexomagnetic_energies[2]
    assert pytest.approx(fme2.dt_fm) == 0.00095
    assert fme2.sens_id == 210


def test_m326_eng_flexomagnetic_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexomagnetic Energy Aliases Test
/ENG/FLEXOMAG_WORK/3
0.00025, 60
/ENG/EFLEXOMAGNETIC/4
0.00030, 70
/ENG/FLEXOMAGNETIC_DISSIPATION/5
0.00035, 80
/ENG/EM_FLEXOMAGNETIC/6
0.00040, 90
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexomagnetic_energies
    assert 4 in model.eng_flexomagnetic_energies
    assert 5 in model.eng_flexomagnetic_energies
    assert 6 in model.eng_flexomagnetic_energies


def test_m326_eng_flexomagnetic_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOMAGNETIC_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m326_lagmul_dietmaier_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{121:>10d}{122:>10d}{123:>10d}{7.5e6:>20.1f}{22:>10d}{1.8e-5:>20.6e}"
    c2 = f"{58.0:>20.4f}{52.5:>20.4f}{48.0:>20.4f}{36.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Dietmaier Linkage Joint Fixed Format Test
2022 0
/DIETMAIER_LINKAGE_JOINT/85
Dietmaier Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 85 in model.lagmul_dietmaier_linkage_joints
    joint = model.lagmul_dietmaier_linkage_joints[85]
    assert joint.node1 == 121
    assert joint.node2 == 122
    assert joint.node3 == 123
    assert pytest.approx(joint.stiff) == 7.5e6
    assert joint.skew_id == 22
    assert pytest.approx(joint.tol) == 1.8e-5
    assert pytest.approx(joint.link_len_a) == 58.0
    assert pytest.approx(joint.link_len_b) == 52.5
    assert pytest.approx(joint.twist_angle_alpha) == 48.0
    assert pytest.approx(joint.link_angle_theta) == 36.0


def test_m326_lagmul_dietmaier_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Dietmaier Linkage Joint Free Format Test
/LAGMUL/DIETMAIER_LINKAGE_JOINT/86
221, 222, 223, 5.5e6, 35, 3.0e-5
62.0, 56.0, 50.0, 42.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 86 in model.lagmul_dietmaier_linkage_joints
    joint = model.lagmul_dietmaier_linkage_joints[86]
    assert joint.node1 == 221
    assert joint.node2 == 222
    assert joint.node3 == 223
    assert pytest.approx(joint.stiff) == 5.5e6
    assert joint.skew_id == 35
    assert pytest.approx(joint.tol) == 3.0e-5
    assert pytest.approx(joint.link_len_a) == 62.0
    assert pytest.approx(joint.link_len_b) == 56.0
    assert pytest.approx(joint.twist_angle_alpha) == 50.0
    assert pytest.approx(joint.link_angle_theta) == 42.0


def test_m326_lagmul_dietmaier_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Dietmaier Linkage Joint Aliases Test
/LAGMUL/DIETMAIER_LINKAGE/87
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 45.0
/DIETMAIER_LINKAGE/88
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 45.0
/DIETMAIER_6R_MECHANISM/89
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 45.0
/DIETMAIER_SPATIAL_MECHANISM/90
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 45.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 87 in model.lagmul_dietmaier_linkage_joints
    assert 88 in model.lagmul_dietmaier_linkage_joints
    assert 89 in model.lagmul_dietmaier_linkage_joints
    assert 90 in model.lagmul_dietmaier_linkage_joints


def test_m326_lagmul_dietmaier_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/DIETMAIER_LINKAGE_JOINT/91
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m326_sensor_spring_transverse_pop_rate_fixed(tmp_path: Path):
    c1 = f"{781:>10d}{1.05e9:>20.1f}{0.0045:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Transverse Pop Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_POP_RATE/118
Spring Transverse Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 118 in model.sensor_spring_transverse_pop_rates
    sensor = model.sensor_spring_transverse_pop_rates[118]
    assert sensor.spring_id == 781
    assert pytest.approx(sensor.jtrans_pop_max) == 1.05e9
    assert pytest.approx(sensor.t_delay) == 0.0045
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_POP_RATE"


def test_m326_sensor_spring_transverse_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Transverse Pop Rate Sensor Free Format Test
/SENSOR/SPRING_TRANSVERSE_POP_RATE/119
782, 1.15e9, 0.0065
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 119 in model.sensor_spring_transverse_pop_rates
    sensor = model.sensor_spring_transverse_pop_rates[119]
    assert sensor.spring_id == 782
    assert pytest.approx(sensor.jtrans_pop_max) == 1.15e9
    assert pytest.approx(sensor.t_delay) == 0.0065


def test_m326_sensor_spring_transverse_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Transverse Pop Rate Sensor Aliases Test
/SENSOR/SPRING_TRANS_POP_RATE/120
783, 1.5e9, 0.001
/SENSOR/SPRING_RATE_POP_TRANS/121
784, 1.5e9, 0.001
/SENSOR/TRANSVERSE_POP_RATE_SPRING/122
785, 1.5e9, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 120 in model.sensor_spring_transverse_pop_rates
    assert 121 in model.sensor_spring_transverse_pop_rates
    assert 122 in model.sensor_spring_transverse_pop_rates
    assert len(model.sensors) == 3


def test_m326_sensor_spring_transverse_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TRANSVERSE_POP_RATE/123
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

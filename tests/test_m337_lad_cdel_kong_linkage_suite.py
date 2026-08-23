"""Tests for Milestone M337: LadCoupleDelaminationRate Failure Model, EngFlexomagnetoacousticResonanceEnergy, KongLinkageJoint, and SensorSpringNormalDropRate."""

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


def test_m337_fail_lad_couple_delamination_rate_fixed(tmp_path: Path):
    c1 = f"{186.0:>20.4f}{510.0:>20.4f}{39.5:>20.4f}{1.78:>20.4f}{0.960:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1530:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Delamination Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_DELAMINATION_RATE/1530
Ladeveze Coupled Delamination Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1530 in model.fail_ladcoupledelaminationrates
    fcdr = model.fail_ladcoupledelaminationrates[1530]
    assert pytest.approx(fcdr.sigma_cdel0) == 186.0
    assert pytest.approx(fcdr.sigma_cdelc) == 510.0
    assert pytest.approx(fcdr.gamma_cdel) == 39.5
    assert pytest.approx(fcdr.p_cdel) == 1.78
    assert pytest.approx(fcdr.d_cdel_max) == 0.960
    assert fcdr.ifail_sh == 1
    assert fcdr.ifail_so == 2
    assert fcdr.fail_id == 1530
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_DELAMINATION_RATE"


def test_m337_fail_lad_couple_delamination_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Delamination Rate Free Format Test
/FAIL/LAD_COUPLE_DELAMINATION_RATE/1531
200.0, 575.0, 44.0, 1.90, 0.935
1, 1
1531
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1531 in model.fail_ladcoupledelaminationrates
    fcdr = model.fail_ladcoupledelaminationrates[1531]
    assert pytest.approx(fcdr.sigma_cdel0) == 200.0
    assert pytest.approx(fcdr.sigma_cdelc) == 575.0
    assert pytest.approx(fcdr.gamma_cdel) == 44.0
    assert pytest.approx(fcdr.p_cdel) == 1.90
    assert pytest.approx(fcdr.d_cdel_max) == 0.935
    assert fcdr.fail_id == 1531


def test_m337_fail_lad_couple_delamination_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Delamination Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_DELAMINATION_RATE/1532
150.0, 425.0, 30.0, 1.50, 0.93
1, 1
/FAIL/LAD_CDELAM_RATE/1533
150.0, 425.0, 30.0, 1.50, 0.93
1, 1
/FAIL/LAD_CDELAM_RATE_MODEL/1534
150.0, 425.0, 30.0, 1.50, 0.93
1, 1
/FAIL/LAD_CDELAM_RATE_LAW/1535
150.0, 425.0, 30.0, 1.50, 0.93
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_DELAMINATION/1536
150.0, 425.0, 30.0, 1.50, 0.93
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1532 in model.fail_ladcoupledelaminationrates
    assert 1533 in model.fail_ladcoupledelaminationrates
    assert 1534 in model.fail_ladcoupledelaminationrates
    assert 1535 in model.fail_ladcoupledelaminationrates
    assert 1536 in model.fail_ladcoupledelaminationrates
    assert len(model.raw_fails) == 5


def test_m337_fail_lad_couple_delamination_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_COUPLE_DELAMINATION_RATE/1537
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m337_eng_flexomagnetoacoustic_resonance_energy(tmp_path: Path):
    c1 = f"{0.00152:>20.6f}{215:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexomagnetoacoustic Resonance Energy Fixed and Free Format Test
2022 0
/ENG/FLEXOMAGNETOACOUSTIC_RESONANCE_ENERGY/1
Fixed Flexomagnetoacoustic Resonance Energy Output
{c1}
/ENG/FLEXOMAGNETOACOUSTIC_RESONANCE_ENERGY/2
Free Flexomagnetoacoustic Resonance Energy Output
0.00198, 370
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexomagnetoacoustic_resonance_energies
    assert 2 in model.eng_flexomagnetoacoustic_resonance_energies
    fmar1 = model.eng_flexomagnetoacoustic_resonance_energies[1]
    assert pytest.approx(fmar1.dt_fmar) == 0.00152
    assert fmar1.sens_id == 215
    fmar2 = model.eng_flexomagnetoacoustic_resonance_energies[2]
    assert pytest.approx(fmar2.dt_fmar) == 0.00198
    assert fmar2.sens_id == 370


def test_m337_eng_flexomagnetoacoustic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexomagnetoacoustic Resonance Energy Aliases Test
/ENG/FLEXOMAG_AC_RES_WORK/3
0.00045, 102
/ENG/EFLEXOMAGNETOACOUSTICRESONANCE/4
0.00050, 112
/ENG/FLEXOMAGNETOACOUSTIC_RESONANCE_DISSIPATION/5
0.00055, 122
/ENG/EM_FLEXOMAGNETOACOUSTIC_RESONANCE/6
0.00060, 132
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexomagnetoacoustic_resonance_energies
    assert 4 in model.eng_flexomagnetoacoustic_resonance_energies
    assert 5 in model.eng_flexomagnetoacoustic_resonance_energies
    assert 6 in model.eng_flexomagnetoacoustic_resonance_energies


def test_m337_eng_flexomagnetoacoustic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOMAGNETOACOUSTIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m337_lagmul_kong_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{231:>10d}{232:>10d}{233:>10d}{1.85e7:>20.1f}{58:>10d}{2.8e-5:>20.6e}"
    c2 = f"{112.0:>20.4f}{102.5:>20.4f}{88.0:>20.4f}{45.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Kong Linkage Joint Fixed Format Test
2022 0
/KONG_LINKAGE_JOINT/195
Kong Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 195 in model.lagmul_kong_linkage_joints
    joint = model.lagmul_kong_linkage_joints[195]
    assert joint.node1 == 231
    assert joint.node2 == 232
    assert joint.node3 == 233
    assert pytest.approx(joint.stiff) == 1.85e7
    assert joint.skew_id == 58
    assert pytest.approx(joint.tol) == 2.8e-5
    assert pytest.approx(joint.link_len_a) == 112.0
    assert pytest.approx(joint.link_len_b) == 102.5
    assert pytest.approx(joint.twist_angle_alpha) == 88.0
    assert pytest.approx(joint.offset_distance_f) == 45.0


def test_m337_lagmul_kong_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Kong Linkage Joint Free Format Test
/LAGMUL/KONG_LINKAGE_JOINT/196
331, 332, 333, 9.0e6, 78, 3.8e-5
115.0, 104.0, 90.0, 48.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 196 in model.lagmul_kong_linkage_joints
    joint = model.lagmul_kong_linkage_joints[196]
    assert joint.node1 == 331
    assert joint.node2 == 332
    assert joint.node3 == 333
    assert pytest.approx(joint.stiff) == 9.0e6
    assert joint.skew_id == 78
    assert pytest.approx(joint.tol) == 3.8e-5
    assert pytest.approx(joint.link_len_a) == 115.0
    assert pytest.approx(joint.link_len_b) == 104.0
    assert pytest.approx(joint.twist_angle_alpha) == 90.0
    assert pytest.approx(joint.offset_distance_f) == 48.0


def test_m337_lagmul_kong_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Kong Linkage Joint Aliases Test
/LAGMUL/KONG_LINKAGE/197
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/KONG_LINKAGE/198
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/KONG_SPATIAL_MECHANISM/199
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/KONG_6R_MECHANISM/200
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/KONG_OVERCONSTRAINED_MECHANISM/201
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 197 in model.lagmul_kong_linkage_joints
    assert 198 in model.lagmul_kong_linkage_joints
    assert 199 in model.lagmul_kong_linkage_joints
    assert 200 in model.lagmul_kong_linkage_joints
    assert 201 in model.lagmul_kong_linkage_joints


def test_m337_lagmul_kong_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/KONG_LINKAGE_JOINT/202
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m337_sensor_spring_normal_drop_rate_fixed(tmp_path: Path):
    c1 = f"{991:>10d}{3.45e9:>20.1f}{0.0098:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Normal Drop Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_DROP_RATE/228
Spring Normal Drop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 228 in model.sensor_spring_normal_drop_rates
    sensor = model.sensor_spring_normal_drop_rates[228]
    assert sensor.spring_id == 991
    assert pytest.approx(sensor.jnorm_drop_max) == 3.45e9
    assert pytest.approx(sensor.t_delay) == 0.0098
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_DROP_RATE"


def test_m337_sensor_spring_normal_drop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Normal Drop Rate Sensor Free Format Test
/SENSOR/SPRING_NORMAL_DROP_RATE/229
992, 3.55e9, 0.0125
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 229 in model.sensor_spring_normal_drop_rates
    sensor = model.sensor_spring_normal_drop_rates[229]
    assert sensor.spring_id == 992
    assert pytest.approx(sensor.jnorm_drop_max) == 3.55e9
    assert pytest.approx(sensor.t_delay) == 0.0125


def test_m337_sensor_spring_normal_drop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Normal Drop Rate Sensor Aliases Test
/SENSOR/SPRING_NORM_DROP_RATE/230
993, 3.8e9, 0.0040
/SENSOR/SPRING_RATE_DROP_NORM/231
994, 3.8e9, 0.0040
/SENSOR/NORMAL_DROP_RATE_SPRING/232
995, 3.8e9, 0.0040
/SENSOR/SPRING_DROP_NORM/233
996, 3.8e9, 0.0040
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 230 in model.sensor_spring_normal_drop_rates
    assert 231 in model.sensor_spring_normal_drop_rates
    assert 232 in model.sensor_spring_normal_drop_rates
    assert 233 in model.sensor_spring_normal_drop_rates
    assert len(model.sensors) == 4


def test_m337_sensor_spring_normal_drop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_NORMAL_DROP_RATE/234
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

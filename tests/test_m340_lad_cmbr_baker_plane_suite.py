"""Tests for Milestone M340: LadCoupleMicrobucklingRate Failure Model, EngFlexothermomagnetoacousticResonanceEnergy, BakerPlaneLinkageJoint, and SensorSpringTorsionalDropRate."""

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


def test_m340_fail_lad_couple_microbuckling_rate_fixed(tmp_path: Path):
    c1 = f"{196.0:>20.4f}{540.0:>20.4f}{44.0:>20.4f}{1.85:>20.4f}{0.945:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1560:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Microbuckling Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_MICROBUCKLING_RATE/1560
Ladeveze Coupled Microbuckling Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1560 in model.fail_ladcouplemicrobucklingrates
    fcmbr = model.fail_ladcouplemicrobucklingrates[1560]
    assert pytest.approx(fcmbr.sigma_cmbr0) == 196.0
    assert pytest.approx(fcmbr.sigma_cmbrc) == 540.0
    assert pytest.approx(fcmbr.gamma_cmbr) == 44.0
    assert pytest.approx(fcmbr.p_cmbr) == 1.85
    assert pytest.approx(fcmbr.d_cmbr_max) == 0.945
    assert fcmbr.ifail_sh == 1
    assert fcmbr.ifail_so == 2
    assert fcmbr.fail_id == 1560
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_MICROBUCKLING_RATE"


def test_m340_fail_lad_couple_microbuckling_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Microbuckling Rate Free Format Test
/FAIL/LAD_COUPLE_MICROBUCKLING_RATE/1561
215.0, 615.0, 50.0, 2.02, 0.920
1, 1
1561
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1561 in model.fail_ladcouplemicrobucklingrates
    fcmbr = model.fail_ladcouplemicrobucklingrates[1561]
    assert pytest.approx(fcmbr.sigma_cmbr0) == 215.0
    assert pytest.approx(fcmbr.sigma_cmbrc) == 615.0
    assert pytest.approx(fcmbr.gamma_cmbr) == 50.0
    assert pytest.approx(fcmbr.p_cmbr) == 2.02
    assert pytest.approx(fcmbr.d_cmbr_max) == 0.920
    assert fcmbr.fail_id == 1561


def test_m340_fail_lad_couple_microbuckling_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Microbuckling Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_MICROBUCKLING_RATE/1562
165.0, 455.0, 36.0, 1.65, 0.945
1, 1
/FAIL/LAD_CMBR/1563
165.0, 455.0, 36.0, 1.65, 0.945
1, 1
/FAIL/LAD_CMBR_MODEL/1564
165.0, 455.0, 36.0, 1.65, 0.945
1, 1
/FAIL/LAD_CMBR_LAW/1565
165.0, 455.0, 36.0, 1.65, 0.945
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_MICROBUCKLING/1566
165.0, 455.0, 36.0, 1.65, 0.945
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1562 in model.fail_ladcouplemicrobucklingrates
    assert 1563 in model.fail_ladcouplemicrobucklingrates
    assert 1564 in model.fail_ladcouplemicrobucklingrates
    assert 1565 in model.fail_ladcouplemicrobucklingrates
    assert 1566 in model.fail_ladcouplemicrobucklingrates
    assert len(model.raw_fails) == 5


def test_m340_fail_lad_couple_microbuckling_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_COUPLE_MICROBUCKLING_RATE/1567
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m340_eng_flexothermomagnetoacoustic_resonance_energy(tmp_path: Path):
    c1 = f"{0.00162:>20.6f}{245:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermomagnetoacoustic Resonance Energy Fixed and Free Format Test
2022 0
/ENG/FLEXOTHERMOMAGNETOACOUSTIC_RESONANCE_ENERGY/1
Fixed Flexothermomagnetoacoustic Resonance Energy Output
{c1}
/ENG/FLEXOTHERMOMAGNETOACOUSTIC_RESONANCE_ENERGY/2
Free Flexothermomagnetoacoustic Resonance Energy Output
0.00218, 395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermomagnetoacoustic_resonance_energies
    assert 2 in model.eng_flexothermomagnetoacoustic_resonance_energies
    ftmar1 = model.eng_flexothermomagnetoacoustic_resonance_energies[1]
    assert pytest.approx(ftmar1.dt_ftmar) == 0.00162
    assert ftmar1.sens_id == 245
    ftmar2 = model.eng_flexothermomagnetoacoustic_resonance_energies[2]
    assert pytest.approx(ftmar2.dt_ftmar) == 0.00218
    assert ftmar2.sens_id == 395


def test_m340_eng_flexothermomagnetoacoustic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermomagnetoacoustic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_MA_RES_WORK/3
0.00052, 112
/ENG/EFLEXOTHERMOMAGNETOACOUSTICRESONANCE/4
0.00058, 122
/ENG/FLEXOTHERMOMAGNETOACOUSTIC_RESONANCE_DISSIPATION/5
0.00062, 132
/ENG/EM_FLEXOTHERMOMAGNETOACOUSTIC_RESONANCE/6
0.00068, 142
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermomagnetoacoustic_resonance_energies
    assert 4 in model.eng_flexothermomagnetoacoustic_resonance_energies
    assert 5 in model.eng_flexothermomagnetoacoustic_resonance_energies
    assert 6 in model.eng_flexothermomagnetoacoustic_resonance_energies


def test_m340_eng_flexothermomagnetoacoustic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOMAGNETOACOUSTIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m340_lagmul_baker_plane_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{261:>10d}{262:>10d}{263:>10d}{2.15e7:>20.1f}{68:>10d}{3.5e-5:>20.6e}"
    c2 = f"{122.0:>20.4f}{112.0:>20.4f}{98.0:>20.4f}{55.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Baker Plane Linkage Joint Fixed Format Test
2022 0
/BAKER_PLANE_LINKAGE_JOINT/225
Baker Plane Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 225 in model.lagmul_baker_plane_linkage_joints
    joint = model.lagmul_baker_plane_linkage_joints[225]
    assert joint.node1 == 261
    assert joint.node2 == 262
    assert joint.node3 == 263
    assert pytest.approx(joint.stiff) == 2.15e7
    assert joint.skew_id == 68
    assert pytest.approx(joint.tol) == 3.5e-5
    assert pytest.approx(joint.link_len_a) == 122.0
    assert pytest.approx(joint.link_len_b) == 112.0
    assert pytest.approx(joint.twist_angle_alpha) == 98.0
    assert pytest.approx(joint.offset_distance_f) == 55.0


def test_m340_lagmul_baker_plane_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Baker Plane Linkage Joint Free Format Test
/LAGMUL/BAKER_PLANE_LINKAGE_JOINT/226
361, 362, 363, 10.2e6, 88, 4.5e-5
125.0, 115.0, 100.0, 58.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 226 in model.lagmul_baker_plane_linkage_joints
    joint = model.lagmul_baker_plane_linkage_joints[226]
    assert joint.node1 == 361
    assert joint.node2 == 362
    assert joint.node3 == 363
    assert pytest.approx(joint.stiff) == 10.2e6
    assert joint.skew_id == 88
    assert pytest.approx(joint.tol) == 4.5e-5
    assert pytest.approx(joint.link_len_a) == 125.0
    assert pytest.approx(joint.link_len_b) == 115.0
    assert pytest.approx(joint.twist_angle_alpha) == 100.0
    assert pytest.approx(joint.offset_distance_f) == 58.0


def test_m340_lagmul_baker_plane_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Baker Plane Linkage Joint Aliases Test
/LAGMUL/BAKER_PLANE_LINKAGE/227
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/BAKER_PLANE_LINKAGE/228
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/BAKER_PLANE_SYMMETRIC_MECHANISM/229
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/BAKER_PLANE_6R_MECHANISM/230
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/BAKER_PLANE_OVERCONSTRAINED_MECHANISM/231
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 227 in model.lagmul_baker_plane_linkage_joints
    assert 228 in model.lagmul_baker_plane_linkage_joints
    assert 229 in model.lagmul_baker_plane_linkage_joints
    assert 230 in model.lagmul_baker_plane_linkage_joints
    assert 231 in model.lagmul_baker_plane_linkage_joints


def test_m340_lagmul_baker_plane_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/BAKER_PLANE_LINKAGE_JOINT/232
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m340_sensor_spring_torsional_drop_rate_fixed(tmp_path: Path):
    c1 = f"{1005:>10d}{4.15e9:>20.1f}{0.0125:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Torsional Drop Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_DROP_RATE/258
Spring Torsional Drop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 258 in model.sensor_spring_torsional_drop_rates
    sensor = model.sensor_spring_torsional_drop_rates[258]
    assert sensor.spring_id == 1005
    assert pytest.approx(sensor.jtors_drop_max) == 4.15e9
    assert pytest.approx(sensor.t_delay) == 0.0125
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_DROP_RATE"


def test_m340_sensor_spring_torsional_drop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Torsional Drop Rate Sensor Free Format Test
/SENSOR/SPRING_TORSIONAL_DROP_RATE/259
1006, 4.25e9, 0.0150
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 259 in model.sensor_spring_torsional_drop_rates
    sensor = model.sensor_spring_torsional_drop_rates[259]
    assert sensor.spring_id == 1006
    assert pytest.approx(sensor.jtors_drop_max) == 4.25e9
    assert pytest.approx(sensor.t_delay) == 0.0150


def test_m340_sensor_spring_torsional_drop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Torsional Drop Rate Sensor Aliases Test
/SENSOR/SPRING_TORS_DROP_RATE/260
1007, 4.3e9, 0.0055
/SENSOR/SPRING_RATE_DROP_TORS/261
1008, 4.3e9, 0.0055
/SENSOR/TORSIONAL_DROP_RATE_SPRING/262
1009, 4.3e9, 0.0055
/SENSOR/SPRING_DROP_TORS/263
1010, 4.3e9, 0.0055
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 260 in model.sensor_spring_torsional_drop_rates
    assert 261 in model.sensor_spring_torsional_drop_rates
    assert 262 in model.sensor_spring_torsional_drop_rates
    assert 263 in model.sensor_spring_torsional_drop_rates
    assert len(model.sensors) == 4


def test_m340_sensor_spring_torsional_drop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TORSIONAL_DROP_RATE/264
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

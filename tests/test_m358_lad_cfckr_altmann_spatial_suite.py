"""Tests for Milestone M358: LadCoupleFiberCompressionKinkingRate Failure Model, EngFlexothermoplasmonphononmagnonpolaritonicResonanceEnergy, AltmannSpatialLinkageJoint, and SensorSpringTorsionalPopRate."""

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


def test_m358_fail_lad_couple_fiber_compression_kinking_rate_fixed(tmp_path: Path):
    c1 = f"{140.0:>20.4f}{420.0:>20.4f}{52.0:>20.4f}{2.40:>20.4f}{0.984:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1740:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Fiber Compression Kinking Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_FIBER_COMPRESSION_KINKING_RATE/1740
Ladeveze Coupled Fiber Compression Kinking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1740 in model.fail_ladcouplefibercompressionkinkingrates
    fcfckr = model.fail_ladcouplefibercompressionkinkingrates[1740]
    assert pytest.approx(fcfckr.sigma_cfckr0) == 140.0
    assert pytest.approx(fcfckr.sigma_cfckrc) == 420.0
    assert pytest.approx(fcfckr.gamma_cfckr) == 52.0
    assert pytest.approx(fcfckr.p_cfckr) == 2.40
    assert pytest.approx(fcfckr.d_cfckr_max) == 0.984
    assert fcfckr.ifail_sh == 1
    assert fcfckr.ifail_so == 2
    assert fcfckr.fail_id == 1740
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_FIBER_COMPRESSION_KINKING_RATE"


def test_m358_fail_lad_couple_fiber_compression_kinking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Fiber Compression Kinking Rate Free Format Test
/FAIL/LAD_COUPLE_FIBER_COMPRESSION_KINKING_RATE/1741
150.0, 450.0, 58.0, 2.60, 0.970
1, 1
1741
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1741 in model.fail_ladcouplefibercompressionkinkingrates
    fcfckr = model.fail_ladcouplefibercompressionkinkingrates[1741]
    assert pytest.approx(fcfckr.sigma_cfckr0) == 150.0
    assert pytest.approx(fcfckr.sigma_cfckrc) == 450.0
    assert pytest.approx(fcfckr.gamma_cfckr) == 58.0
    assert pytest.approx(fcfckr.p_cfckr) == 2.60
    assert pytest.approx(fcfckr.d_cfckr_max) == 0.970
    assert fcfckr.fail_id == 1741


def test_m358_fail_lad_couple_fiber_compression_kinking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Fiber Compression Kinking Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_FIBER_COMPRESSION_KINKING_RATE/1742
130.0, 390.0, 46.0, 2.10, 0.990
1, 1
/FAIL/LAD_CFCKR/1743
130.0, 390.0, 46.0, 2.10, 0.990
1, 1
/FAIL/LAD_CFCKR_MODEL/1744
130.0, 390.0, 46.0, 2.10, 0.990
1, 1
/FAIL/LAD_CFCKR_LAW/1745
130.0, 390.0, 46.0, 2.10, 0.990
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_FIBER_COMPRESSION_KINKING/1746
130.0, 390.0, 46.0, 2.10, 0.990
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1742 in model.fail_ladcouplefibercompressionkinkingrates
    assert 1743 in model.fail_ladcouplefibercompressionkinkingrates
    assert 1744 in model.fail_ladcouplefibercompressionkinkingrates
    assert 1745 in model.fail_ladcouplefibercompressionkinkingrates
    assert 1746 in model.fail_ladcouplefibercompressionkinkingrates
    assert len(model.raw_fails) == 5


def test_m358_fail_lad_couple_fiber_compression_kinking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_COUPLE_FIBER_COMPRESSION_KINKING_RATE/1747
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m358_eng_flexothermoplasmonphononmagnonpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00150:>20.6f}{240:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoplasmonphononmagnonpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPLASMONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY/1
Flexothermoplasmonphononmagnonpolaritonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoplasmonphononmagnonpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonphononmagnonpolaritonic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftppmpr) == 0.00150
    assert eng.sens_id == 240


def test_m358_eng_flexothermoplasmonphononmagnonpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermoplasmonphononmagnonpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPLASMONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY/2
0.00170, 250
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermoplasmonphononmagnonpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonphononmagnonpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftppmpr) == 0.00170
    assert eng.sens_id == 250


def test_m358_eng_flexothermoplasmonphononmagnonpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoplasmonphononmagnonpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PLASMON_PHONON_MAGNON_POLARITON_RES_WORK/3
0.00110, 180
/ENG/EFLEXOTHERMOPLASMONPHONONMAGNONPOLARITONICRESONANCE/4
0.00114, 186
/ENG/FLEXOTHERMOPLASMONPHONONMAGNONPOLARITONIC_RESONANCE_DISSIPATION/5
0.00120, 196
/ENG/EM_FLEXOTHERMOPLASMONPHONONMAGNONPOLARITONIC_RESONANCE/6
0.00128, 206
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoplasmonphononmagnonpolaritonic_resonance_energies
    assert 4 in model.eng_flexothermoplasmonphononmagnonpolaritonic_resonance_energies
    assert 5 in model.eng_flexothermoplasmonphononmagnonpolaritonic_resonance_energies
    assert 6 in model.eng_flexothermoplasmonphononmagnonpolaritonic_resonance_energies


def test_m358_eng_flexothermoplasmonphononmagnonpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOPLASMONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m358_lagmul_altmann_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{441:>10d}{442:>10d}{443:>10d}{3.95e7:>20.1f}{124:>10d}{3.6e-5:>20.6e}"
    c2 = f"{205.0:>20.4f}{195.0:>20.4f}{175.0:>20.4f}{125.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Altmann Spatial Linkage Joint Fixed Format Test
2022 0
/ALTMANN_SPATIAL_LINKAGE_JOINT/405
Altmann Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 405 in model.lagmul_altmann_spatial_linkage_joints
    joint = model.lagmul_altmann_spatial_linkage_joints[405]
    assert joint.node1 == 441
    assert joint.node2 == 442
    assert joint.node3 == 443
    assert pytest.approx(joint.stiff) == 3.95e7
    assert joint.skew_id == 124
    assert pytest.approx(joint.tol) == 3.6e-5
    assert pytest.approx(joint.link_len_a) == 205.0
    assert pytest.approx(joint.link_len_b) == 195.0
    assert pytest.approx(joint.twist_angle_alpha) == 175.0
    assert pytest.approx(joint.offset_distance_s) == 125.0


def test_m358_lagmul_altmann_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Altmann Spatial Linkage Joint Free Format Test
/LAGMUL/ALTMANN_SPATIAL_LINKAGE_JOINT/406
541, 542, 543, 25.0e6, 144, 4.6e-5
208.0, 198.0, 178.0, 128.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 406 in model.lagmul_altmann_spatial_linkage_joints
    joint = model.lagmul_altmann_spatial_linkage_joints[406]
    assert joint.node1 == 541
    assert joint.node2 == 542
    assert joint.node3 == 543
    assert pytest.approx(joint.stiff) == 25.0e6
    assert joint.skew_id == 144
    assert pytest.approx(joint.tol) == 4.6e-5
    assert pytest.approx(joint.link_len_a) == 208.0
    assert pytest.approx(joint.link_len_b) == 198.0
    assert pytest.approx(joint.twist_angle_alpha) == 178.0
    assert pytest.approx(joint.offset_distance_s) == 128.0


def test_m358_lagmul_altmann_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Altmann Spatial Linkage Joint Aliases Test
/LAGMUL/ALTMANN_SPATIAL_LINKAGE/407
1, 2, 3, 1e6, 0, 1e-6
52.0, 52.0, 98.0, 38.0
/ALTMANN_SPATIAL_LINKAGE/408
1, 2, 3, 1e6, 0, 1e-6
52.0, 52.0, 98.0, 38.0
/ALTMANN_SPATIAL_MULTI_LOOP_MECHANISM/409
1, 2, 3, 1e6, 0, 1e-6
52.0, 52.0, 98.0, 38.0
/ALTMANN_SPATIAL_LINE_SYMMETRIC_MECHANISM/410
1, 2, 3, 1e6, 0, 1e-6
52.0, 52.0, 98.0, 38.0
/ALTMANN_SPATIAL_6R_MECHANISM/411
1, 2, 3, 1e6, 0, 1e-6
52.0, 52.0, 98.0, 38.0
/ALTMANN_SPATIAL_OVERCONSTRAINED_MECHANISM/412
1, 2, 3, 1e6, 0, 1e-6
52.0, 52.0, 98.0, 38.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 407 in model.lagmul_altmann_spatial_linkage_joints
    assert 408 in model.lagmul_altmann_spatial_linkage_joints
    assert 409 in model.lagmul_altmann_spatial_linkage_joints
    assert 410 in model.lagmul_altmann_spatial_linkage_joints
    assert 411 in model.lagmul_altmann_spatial_linkage_joints
    assert 412 in model.lagmul_altmann_spatial_linkage_joints


def test_m358_lagmul_altmann_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ALTMANN_SPATIAL_LINKAGE_JOINT/413
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m358_sensor_spring_torsional_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1185:>10d}{6.85e9:>20.1f}{0.0305:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Torsional Pop Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_POP_RATE/438
Spring Torsional Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 438 in model.sensor_spring_torsional_pop_rates
    sensor = model.sensor_spring_torsional_pop_rates[438]
    assert sensor.spring_id == 1185
    assert pytest.approx(sensor.jtors_pop_max) == 6.85e9
    assert pytest.approx(sensor.t_delay) == 0.0305
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_POP_RATE"


def test_m358_sensor_spring_torsional_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Torsional Pop Rate Sensor Free Format Test
/SENSOR/SPRING_TORSIONAL_POP_RATE/439
1186, 6.95e9, 0.0330
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 439 in model.sensor_spring_torsional_pop_rates
    sensor = model.sensor_spring_torsional_pop_rates[439]
    assert sensor.spring_id == 1186
    assert pytest.approx(sensor.jtors_pop_max) == 6.95e9
    assert pytest.approx(sensor.t_delay) == 0.0330


def test_m358_sensor_spring_torsional_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Torsional Pop Rate Sensor Aliases Test
/SENSOR/SPRING_TORS_POP_RATE/440
1187, 7.0e9, 0.0235
/SENSOR/SPRING_RATE_POP_TORS/441
1188, 7.0e9, 0.0235
/SENSOR/TORSIONAL_POP_RATE_SPRING/442
1189, 7.0e9, 0.0235
/SENSOR/SPRING_POP_TORS/443
1190, 7.0e9, 0.0235
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 440 in model.sensor_spring_torsional_pop_rates
    assert 441 in model.sensor_spring_torsional_pop_rates
    assert 442 in model.sensor_spring_torsional_pop_rates
    assert 443 in model.sensor_spring_torsional_pop_rates
    assert len(model.sensors) == 4


def test_m358_sensor_spring_torsional_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TORSIONAL_POP_RATE/444
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

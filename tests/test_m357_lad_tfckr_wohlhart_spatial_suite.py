"""Tests for Milestone M357: LadTransverseFiberCompressionKinkingRate Failure Model, EngFlexothermoplasmonexcitonmagnonpolaritonicResonanceEnergy, WohlhartSpatialLinkageJoint, and SensorSpringTotalPopRate."""

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


def test_m357_fail_lad_transverse_fiber_compression_kinking_rate_fixed(tmp_path: Path):
    c1 = f"{138.0:>20.4f}{414.0:>20.4f}{50.0:>20.4f}{2.35:>20.4f}{0.986:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1730:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Fiber Compression Kinking Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSION_KINKING_RATE/1730
Ladeveze Transverse Fiber Compression Kinking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1730 in model.fail_ladtransversefibercompressionkinkingrates
    ftfckr = model.fail_ladtransversefibercompressionkinkingrates[1730]
    assert pytest.approx(ftfckr.sigma_tfckr0) == 138.0
    assert pytest.approx(ftfckr.sigma_tfckrc) == 414.0
    assert pytest.approx(ftfckr.gamma_tfckr) == 50.0
    assert pytest.approx(ftfckr.p_tfckr) == 2.35
    assert pytest.approx(ftfckr.d_tfckr_max) == 0.986
    assert ftfckr.ifail_sh == 1
    assert ftfckr.ifail_so == 2
    assert ftfckr.fail_id == 1730
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_FIBER_COMPRESSION_KINKING_RATE"


def test_m357_fail_lad_transverse_fiber_compression_kinking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Fiber Compression Kinking Rate Free Format Test
/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSION_KINKING_RATE/1731
148.0, 444.0, 56.0, 2.55, 0.972
1, 1
1731
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1731 in model.fail_ladtransversefibercompressionkinkingrates
    ftfckr = model.fail_ladtransversefibercompressionkinkingrates[1731]
    assert pytest.approx(ftfckr.sigma_tfckr0) == 148.0
    assert pytest.approx(ftfckr.sigma_tfckrc) == 444.0
    assert pytest.approx(ftfckr.gamma_tfckr) == 56.0
    assert pytest.approx(ftfckr.p_tfckr) == 2.55
    assert pytest.approx(ftfckr.d_tfckr_max) == 0.972
    assert ftfckr.fail_id == 1731


def test_m357_fail_lad_transverse_fiber_compression_kinking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Fiber Compression Kinking Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_FIBER_COMPRESSION_KINKING_RATE/1732
128.0, 384.0, 44.0, 2.08, 0.988
1, 1
/FAIL/LAD_TFCKR/1733
128.0, 384.0, 44.0, 2.08, 0.988
1, 1
/FAIL/LAD_TFCKR_MODEL/1734
128.0, 384.0, 44.0, 2.08, 0.988
1, 1
/FAIL/LAD_TFCKR_LAW/1735
128.0, 384.0, 44.0, 2.08, 0.988
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_FIBER_COMPRESSION_KINKING/1736
128.0, 384.0, 44.0, 2.08, 0.988
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1732 in model.fail_ladtransversefibercompressionkinkingrates
    assert 1733 in model.fail_ladtransversefibercompressionkinkingrates
    assert 1734 in model.fail_ladtransversefibercompressionkinkingrates
    assert 1735 in model.fail_ladtransversefibercompressionkinkingrates
    assert 1736 in model.fail_ladtransversefibercompressionkinkingrates
    assert len(model.raw_fails) == 5


def test_m357_fail_lad_transverse_fiber_compression_kinking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSION_KINKING_RATE/1737
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m357_eng_flexothermoplasmonexcitonmagnonpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00145:>20.6f}{236:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoplasmonexcitonmagnonpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPLASMONEXCITONMAGNONPOLARITONIC_RESONANCE_ENERGY/1
Flexothermoplasmonexcitonmagnonpolaritonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoplasmonexcitonmagnonpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonexcitonmagnonpolaritonic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftpempr) == 0.00145
    assert eng.sens_id == 236


def test_m357_eng_flexothermoplasmonexcitonmagnonpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermoplasmonexcitonmagnonpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPLASMONEXCITONMAGNONPOLARITONIC_RESONANCE_ENERGY/2
0.00165, 246
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermoplasmonexcitonmagnonpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonexcitonmagnonpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftpempr) == 0.00165
    assert eng.sens_id == 246


def test_m357_eng_flexothermoplasmonexcitonmagnonpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoplasmonexcitonmagnonpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK/3
0.00108, 178
/ENG/EFLEXOTHERMOPLASMONEXCITONMAGNONPOLARITONICRESONANCE/4
0.00112, 184
/ENG/FLEXOTHERMOPLASMONEXCITONMAGNONPOLARITONIC_RESONANCE_DISSIPATION/5
0.00118, 194
/ENG/EM_FLEXOTHERMOPLASMONEXCITONMAGNONPOLARITONIC_RESONANCE/6
0.00126, 204
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoplasmonexcitonmagnonpolaritonic_resonance_energies
    assert 4 in model.eng_flexothermoplasmonexcitonmagnonpolaritonic_resonance_energies
    assert 5 in model.eng_flexothermoplasmonexcitonmagnonpolaritonic_resonance_energies
    assert 6 in model.eng_flexothermoplasmonexcitonmagnonpolaritonic_resonance_energies


def test_m357_eng_flexothermoplasmonexcitonmagnonpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOPLASMONEXCITONMAGNONPOLARITONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m357_lagmul_wohlhart_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{431:>10d}{432:>10d}{433:>10d}{3.85e7:>20.1f}{122:>10d}{3.8e-5:>20.6e}"
    c2 = f"{200.0:>20.4f}{190.0:>20.4f}{170.0:>20.4f}{120.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Wohlhart Spatial Linkage Joint Fixed Format Test
2022 0
/WOHLHART_SPATIAL_LINKAGE_JOINT/395
Wohlhart Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 395 in model.lagmul_wohlhart_spatial_linkage_joints
    joint = model.lagmul_wohlhart_spatial_linkage_joints[395]
    assert joint.node1 == 431
    assert joint.node2 == 432
    assert joint.node3 == 433
    assert pytest.approx(joint.stiff) == 3.85e7
    assert joint.skew_id == 122
    assert pytest.approx(joint.tol) == 3.8e-5
    assert pytest.approx(joint.link_len_a) == 200.0
    assert pytest.approx(joint.link_len_b) == 190.0
    assert pytest.approx(joint.twist_angle_alpha) == 170.0
    assert pytest.approx(joint.offset_distance_s) == 120.0


def test_m357_lagmul_wohlhart_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Wohlhart Spatial Linkage Joint Free Format Test
/LAGMUL/WOHLHART_SPATIAL_LINKAGE_JOINT/396
531, 532, 533, 24.0e6, 142, 4.8e-5
202.0, 192.0, 172.0, 122.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 396 in model.lagmul_wohlhart_spatial_linkage_joints
    joint = model.lagmul_wohlhart_spatial_linkage_joints[396]
    assert joint.node1 == 531
    assert joint.node2 == 532
    assert joint.node3 == 533
    assert pytest.approx(joint.stiff) == 24.0e6
    assert joint.skew_id == 142
    assert pytest.approx(joint.tol) == 4.8e-5
    assert pytest.approx(joint.link_len_a) == 202.0
    assert pytest.approx(joint.link_len_b) == 192.0
    assert pytest.approx(joint.twist_angle_alpha) == 172.0
    assert pytest.approx(joint.offset_distance_s) == 122.0


def test_m357_lagmul_wohlhart_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Wohlhart Spatial Linkage Joint Aliases Test
/LAGMUL/WOHLHART_SPATIAL_LINKAGE/397
1, 2, 3, 1e6, 0, 1e-6
50.0, 50.0, 95.0, 35.0
/WOHLHART_SPATIAL_LINKAGE/398
1, 2, 3, 1e6, 0, 1e-6
50.0, 50.0, 95.0, 35.0
/WOHLHART_SPATIAL_MULTI_LOOP_MECHANISM/399
1, 2, 3, 1e6, 0, 1e-6
50.0, 50.0, 95.0, 35.0
/WOHLHART_SPATIAL_SPHERICAL_DOUBLE_SUBGROUP_MECHANISM/400
1, 2, 3, 1e6, 0, 1e-6
50.0, 50.0, 95.0, 35.0
/WOHLHART_SPATIAL_6R_MECHANISM/401
1, 2, 3, 1e6, 0, 1e-6
50.0, 50.0, 95.0, 35.0
/WOHLHART_SPATIAL_OVERCONSTRAINED_MECHANISM/402
1, 2, 3, 1e6, 0, 1e-6
50.0, 50.0, 95.0, 35.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 397 in model.lagmul_wohlhart_spatial_linkage_joints
    assert 398 in model.lagmul_wohlhart_spatial_linkage_joints
    assert 399 in model.lagmul_wohlhart_spatial_linkage_joints
    assert 400 in model.lagmul_wohlhart_spatial_linkage_joints
    assert 401 in model.lagmul_wohlhart_spatial_linkage_joints
    assert 402 in model.lagmul_wohlhart_spatial_linkage_joints


def test_m357_lagmul_wohlhart_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/WOHLHART_SPATIAL_LINKAGE_JOINT/403
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m357_sensor_spring_total_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1175:>10d}{6.65e9:>20.1f}{0.0295:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Total Pop Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_POP_RATE/428
Spring Total Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 428 in model.sensor_spring_total_pop_rates
    sensor = model.sensor_spring_total_pop_rates[428]
    assert sensor.spring_id == 1175
    assert pytest.approx(sensor.jtot_pop_max) == 6.65e9
    assert pytest.approx(sensor.t_delay) == 0.0295
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_POP_RATE"


def test_m357_sensor_spring_total_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Total Pop Rate Sensor Free Format Test
/SENSOR/SPRING_TOTAL_POP_RATE/429
1176, 6.75e9, 0.0320
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 429 in model.sensor_spring_total_pop_rates
    sensor = model.sensor_spring_total_pop_rates[429]
    assert sensor.spring_id == 1176
    assert pytest.approx(sensor.jtot_pop_max) == 6.75e9
    assert pytest.approx(sensor.t_delay) == 0.0320


def test_m357_sensor_spring_total_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Total Pop Rate Sensor Aliases Test
/SENSOR/SPRING_TOT_POP_RATE/430
1177, 6.8e9, 0.0225
/SENSOR/SPRING_RATE_POP_TOT/431
1178, 6.8e9, 0.0225
/SENSOR/TOTAL_POP_RATE_SPRING/432
1179, 6.8e9, 0.0225
/SENSOR/SPRING_POP_TOT/433
1180, 6.8e9, 0.0225
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 430 in model.sensor_spring_total_pop_rates
    assert 431 in model.sensor_spring_total_pop_rates
    assert 432 in model.sensor_spring_total_pop_rates
    assert 433 in model.sensor_spring_total_pop_rates
    assert len(model.sensors) == 4


def test_m357_sensor_spring_total_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TOTAL_POP_RATE/434
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

"""Tests for Milestone M356: LadDynamicFiberCompressionKinkingRate Failure Model, EngFlexothermoplasmonexcitonphononpolaritonicResonanceEnergy, DelassusSpatialLinkageJoint, and SensorSpringTransversePopRate."""

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


def test_m356_fail_lad_dynamic_fiber_compression_kinking_rate_fixed(tmp_path: Path):
    c1 = f"{136.0:>20.4f}{408.0:>20.4f}{48.0:>20.4f}{2.30:>20.4f}{0.988:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1720:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Fiber Compression Kinking Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_FIBER_COMPRESSION_KINKING_RATE/1720
Ladeveze Dynamic Fiber Compression Kinking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1720 in model.fail_laddynamicfibercompressionkinkingrates
    fdfckr = model.fail_laddynamicfibercompressionkinkingrates[1720]
    assert pytest.approx(fdfckr.sigma_dfckr0) == 136.0
    assert pytest.approx(fdfckr.sigma_dfckrc) == 408.0
    assert pytest.approx(fdfckr.gamma_dfckr) == 48.0
    assert pytest.approx(fdfckr.p_dfckr) == 2.30
    assert pytest.approx(fdfckr.d_dfckr_max) == 0.988
    assert fdfckr.ifail_sh == 1
    assert fdfckr.ifail_so == 2
    assert fdfckr.fail_id == 1720
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_FIBER_COMPRESSION_KINKING_RATE"


def test_m356_fail_lad_dynamic_fiber_compression_kinking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Fiber Compression Kinking Rate Free Format Test
/FAIL/LAD_DYNAMIC_FIBER_COMPRESSION_KINKING_RATE/1721
146.0, 438.0, 54.0, 2.50, 0.974
1, 1
1721
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1721 in model.fail_laddynamicfibercompressionkinkingrates
    fdfckr = model.fail_laddynamicfibercompressionkinkingrates[1721]
    assert pytest.approx(fdfckr.sigma_dfckr0) == 146.0
    assert pytest.approx(fdfckr.sigma_dfckrc) == 438.0
    assert pytest.approx(fdfckr.gamma_dfckr) == 54.0
    assert pytest.approx(fdfckr.p_dfckr) == 2.50
    assert pytest.approx(fdfckr.d_dfckr_max) == 0.974
    assert fdfckr.fail_id == 1721


def test_m356_fail_lad_dynamic_fiber_compression_kinking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Fiber Compression Kinking Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_FIBER_COMPRESSION_KINKING_RATE/1722
126.0, 378.0, 42.0, 2.05, 0.986
1, 1
/FAIL/LAD_DFCKR/1723
126.0, 378.0, 42.0, 2.05, 0.986
1, 1
/FAIL/LAD_DFCKR_MODEL/1724
126.0, 378.0, 42.0, 2.05, 0.986
1, 1
/FAIL/LAD_DFCKR_LAW/1725
126.0, 378.0, 42.0, 2.05, 0.986
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_FIBER_COMPRESSION_KINKING/1726
126.0, 378.0, 42.0, 2.05, 0.986
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1722 in model.fail_laddynamicfibercompressionkinkingrates
    assert 1723 in model.fail_laddynamicfibercompressionkinkingrates
    assert 1724 in model.fail_laddynamicfibercompressionkinkingrates
    assert 1725 in model.fail_laddynamicfibercompressionkinkingrates
    assert 1726 in model.fail_laddynamicfibercompressionkinkingrates
    assert len(model.raw_fails) == 5


def test_m356_fail_lad_dynamic_fiber_compression_kinking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_DYNAMIC_FIBER_COMPRESSION_KINKING_RATE/1727
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m356_eng_flexothermoplasmonexcitonphononpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00140:>20.6f}{232:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoplasmonexcitonphononpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPLASMONEXCITONPHONONPOLARITONIC_RESONANCE_ENERGY/1
Flexothermoplasmonexcitonphononpolaritonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoplasmonexcitonphononpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonexcitonphononpolaritonic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftpeppr) == 0.00140
    assert eng.sens_id == 232


def test_m356_eng_flexothermoplasmonexcitonphononpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermoplasmonexcitonphononpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPLASMONEXCITONPHONONPOLARITONIC_RESONANCE_ENERGY/2
0.00160, 242
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermoplasmonexcitonphononpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonexcitonphononpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftpeppr) == 0.00160
    assert eng.sens_id == 242


def test_m356_eng_flexothermoplasmonexcitonphononpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoplasmonexcitonphononpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PLASMON_EXCITON_PHONON_POLARITON_RES_WORK/3
0.00106, 175
/ENG/EFLEXOTHERMOPLASMONEXCITONPHONONPOLARITONICRESONANCE/4
0.00111, 182
/ENG/FLEXOTHERMOPLASMONEXCITONPHONONPOLARITONIC_RESONANCE_DISSIPATION/5
0.00116, 192
/ENG/EM_FLEXOTHERMOPLASMONEXCITONPHONONPOLARITONIC_RESONANCE/6
0.00124, 202
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoplasmonexcitonphononpolaritonic_resonance_energies
    assert 4 in model.eng_flexothermoplasmonexcitonphononpolaritonic_resonance_energies
    assert 5 in model.eng_flexothermoplasmonexcitonphononpolaritonic_resonance_energies
    assert 6 in model.eng_flexothermoplasmonexcitonphononpolaritonic_resonance_energies


def test_m356_eng_flexothermoplasmonexcitonphononpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOPLASMONEXCITONPHONONPOLARITONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m356_lagmul_delassus_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{421:>10d}{422:>10d}{423:>10d}{3.75e7:>20.1f}{120:>10d}{4.0e-5:>20.6e}"
    c2 = f"{195.0:>20.4f}{185.0:>20.4f}{165.0:>20.4f}{115.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Delassus Spatial Linkage Joint Fixed Format Test
2022 0
/DELASSUS_SPATIAL_LINKAGE_JOINT/385
Delassus Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 385 in model.lagmul_delassus_spatial_linkage_joints
    joint = model.lagmul_delassus_spatial_linkage_joints[385]
    assert joint.node1 == 421
    assert joint.node2 == 422
    assert joint.node3 == 423
    assert pytest.approx(joint.stiff) == 3.75e7
    assert joint.skew_id == 120
    assert pytest.approx(joint.tol) == 4.0e-5
    assert pytest.approx(joint.link_len_a) == 195.0
    assert pytest.approx(joint.link_len_b) == 185.0
    assert pytest.approx(joint.twist_angle_alpha) == 165.0
    assert pytest.approx(joint.offset_distance_s) == 115.0


def test_m356_lagmul_delassus_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Delassus Spatial Linkage Joint Free Format Test
/LAGMUL/DELASSUS_SPATIAL_LINKAGE_JOINT/386
521, 522, 523, 23.0e6, 140, 5.0e-5
196.0, 186.0, 166.0, 116.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 386 in model.lagmul_delassus_spatial_linkage_joints
    joint = model.lagmul_delassus_spatial_linkage_joints[386]
    assert joint.node1 == 521
    assert joint.node2 == 522
    assert joint.node3 == 523
    assert pytest.approx(joint.stiff) == 23.0e6
    assert joint.skew_id == 140
    assert pytest.approx(joint.tol) == 5.0e-5
    assert pytest.approx(joint.link_len_a) == 196.0
    assert pytest.approx(joint.link_len_b) == 186.0
    assert pytest.approx(joint.twist_angle_alpha) == 166.0
    assert pytest.approx(joint.offset_distance_s) == 116.0


def test_m356_lagmul_delassus_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Delassus Spatial Linkage Joint Aliases Test
/LAGMUL/DELASSUS_SPATIAL_LINKAGE/387
1, 2, 3, 1e6, 0, 1e-6
48.0, 48.0, 92.0, 32.0
/DELASSUS_SPATIAL_LINKAGE/388
1, 2, 3, 1e6, 0, 1e-6
48.0, 48.0, 92.0, 32.0
/DELASSUS_SPATIAL_MULTI_LOOP_MECHANISM/389
1, 2, 3, 1e6, 0, 1e-6
48.0, 48.0, 92.0, 32.0
/DELASSUS_SPATIAL_CYLINDRICAL_MECHANISM/390
1, 2, 3, 1e6, 0, 1e-6
48.0, 48.0, 92.0, 32.0
/DELASSUS_SPATIAL_6R_MECHANISM/391
1, 2, 3, 1e6, 0, 1e-6
48.0, 48.0, 92.0, 32.0
/DELASSUS_SPATIAL_OVERCONSTRAINED_MECHANISM/392
1, 2, 3, 1e6, 0, 1e-6
48.0, 48.0, 92.0, 32.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 387 in model.lagmul_delassus_spatial_linkage_joints
    assert 388 in model.lagmul_delassus_spatial_linkage_joints
    assert 389 in model.lagmul_delassus_spatial_linkage_joints
    assert 390 in model.lagmul_delassus_spatial_linkage_joints
    assert 391 in model.lagmul_delassus_spatial_linkage_joints
    assert 392 in model.lagmul_delassus_spatial_linkage_joints


def test_m356_lagmul_delassus_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/DELASSUS_SPATIAL_LINKAGE_JOINT/393
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m356_sensor_spring_transverse_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1165:>10d}{6.45e9:>20.1f}{0.0285:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Transverse Pop Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_POP_RATE/418
Spring Transverse Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 418 in model.sensor_spring_transverse_pop_rates
    sensor = model.sensor_spring_transverse_pop_rates[418]
    assert sensor.spring_id == 1165
    assert pytest.approx(sensor.jtrans_pop_max) == 6.45e9
    assert pytest.approx(sensor.t_delay) == 0.0285
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_POP_RATE"


def test_m356_sensor_spring_transverse_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Transverse Pop Rate Sensor Free Format Test
/SENSOR/SPRING_TRANSVERSE_POP_RATE/419
1166, 6.55e9, 0.0310
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 419 in model.sensor_spring_transverse_pop_rates
    sensor = model.sensor_spring_transverse_pop_rates[419]
    assert sensor.spring_id == 1166
    assert pytest.approx(sensor.jtrans_pop_max) == 6.55e9
    assert pytest.approx(sensor.t_delay) == 0.0310


def test_m356_sensor_spring_transverse_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Transverse Pop Rate Sensor Aliases Test
/SENSOR/SPRING_TRANS_POP_RATE/420
1167, 6.6e9, 0.0215
/SENSOR/SPRING_RATE_POP_TRANS/421
1168, 6.6e9, 0.0215
/SENSOR/TRANSVERSE_POP_RATE_SPRING/422
1169, 6.6e9, 0.0215
/SENSOR/SPRING_POP_TRANS/423
1170, 6.6e9, 0.0215
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 420 in model.sensor_spring_transverse_pop_rates
    assert 421 in model.sensor_spring_transverse_pop_rates
    assert 422 in model.sensor_spring_transverse_pop_rates
    assert 423 in model.sensor_spring_transverse_pop_rates
    assert len(model.sensors) == 4


def test_m356_sensor_spring_transverse_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TRANSVERSE_POP_RATE/424
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

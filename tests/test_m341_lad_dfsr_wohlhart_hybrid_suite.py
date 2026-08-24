"""Tests for Milestone M341: LadDynamicFiberSplittingRate Failure Model, EngFlexothermoelectromagnetoacousticResonanceEnergy, WohlhartHybridLinkageJoint, and SensorSpringBendingDropRate."""

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


def test_m341_fail_lad_dynamic_fiber_splitting_rate_fixed(tmp_path: Path):
    c1 = f"{210.0:>20.4f}{580.0:>20.4f}{48.0:>20.4f}{1.92:>20.4f}{0.950:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1570:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Fiber Splitting Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_FIBER_SPLITTING_RATE/1570
Ladeveze Dynamic Fiber Splitting Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1570 in model.fail_laddynamicfibersplittingrates
    fdfsr = model.fail_laddynamicfibersplittingrates[1570]
    assert pytest.approx(fdfsr.sigma_dfsr0) == 210.0
    assert pytest.approx(fdfsr.sigma_dfsrc) == 580.0
    assert pytest.approx(fdfsr.gamma_dfsr) == 48.0
    assert pytest.approx(fdfsr.p_dfsr) == 1.92
    assert pytest.approx(fdfsr.d_dfsr_max) == 0.950
    assert fdfsr.ifail_sh == 1
    assert fdfsr.ifail_so == 2
    assert fdfsr.fail_id == 1570
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_FIBER_SPLITTING_RATE"


def test_m341_fail_lad_dynamic_fiber_splitting_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Fiber Splitting Rate Free Format Test
/FAIL/LAD_DYNAMIC_FIBER_SPLITTING_RATE/1571
225.0, 630.0, 52.0, 2.10, 0.930
1, 1
1571
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1571 in model.fail_laddynamicfibersplittingrates
    fdfsr = model.fail_laddynamicfibersplittingrates[1571]
    assert pytest.approx(fdfsr.sigma_dfsr0) == 225.0
    assert pytest.approx(fdfsr.sigma_dfsrc) == 630.0
    assert pytest.approx(fdfsr.gamma_dfsr) == 52.0
    assert pytest.approx(fdfsr.p_dfsr) == 2.10
    assert pytest.approx(fdfsr.d_dfsr_max) == 0.930
    assert fdfsr.fail_id == 1571


def test_m341_fail_lad_dynamic_fiber_splitting_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Fiber Splitting Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_FIBER_SPLITTING_RATE/1572
175.0, 480.0, 38.0, 1.70, 0.940
1, 1
/FAIL/LAD_DFSR/1573
175.0, 480.0, 38.0, 1.70, 0.940
1, 1
/FAIL/LAD_DFSR_MODEL/1574
175.0, 480.0, 38.0, 1.70, 0.940
1, 1
/FAIL/LAD_DFSR_LAW/1575
175.0, 480.0, 38.0, 1.70, 0.940
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_FIBER_SPLITTING/1576
175.0, 480.0, 38.0, 1.70, 0.940
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1572 in model.fail_laddynamicfibersplittingrates
    assert 1573 in model.fail_laddynamicfibersplittingrates
    assert 1574 in model.fail_laddynamicfibersplittingrates
    assert 1575 in model.fail_laddynamicfibersplittingrates
    assert 1576 in model.fail_laddynamicfibersplittingrates
    assert len(model.raw_fails) == 5


def test_m341_fail_lad_dynamic_fiber_splitting_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_DYNAMIC_FIBER_SPLITTING_RATE/1577
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m341_eng_flexothermoelectromagnetoacoustic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00072:>20.6f}{150:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoelectromagnetoacoustic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOELECTROMAGNETOACOUSTIC_RESONANCE_ENERGY/1
Flexothermoelectromagnetoacoustic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoelectromagnetoacoustic_resonance_energies
    eng = model.eng_flexothermoelectromagnetoacoustic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftemmar) == 0.00072
    assert eng.sens_id == 150


def test_m341_eng_flexothermoelectromagnetoacoustic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermoelectromagnetoacoustic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOELECTROMAGNETOACOUSTIC_RESONANCE_ENERGY/2
0.00085, 160
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermoelectromagnetoacoustic_resonance_energies
    eng = model.eng_flexothermoelectromagnetoacoustic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftemmar) == 0.00085
    assert eng.sens_id == 160


def test_m341_eng_flexothermoelectromagnetoacoustic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoelectromagnetoacoustic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_EMA_RES_WORK/3
0.00055, 120
/ENG/EFLEXOTHERMOELECTROMAGNETOACOUSTICRESONANCE/4
0.00058, 125
/ENG/FLEXOTHERMOELECTROMAGNETOACOUSTIC_RESONANCE_DISSIPATION/5
0.00065, 135
/ENG/EM_FLEXOTHERMOELECTROMAGNETOACOUSTIC_RESONANCE/6
0.00070, 145
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoelectromagnetoacoustic_resonance_energies
    assert 4 in model.eng_flexothermoelectromagnetoacoustic_resonance_energies
    assert 5 in model.eng_flexothermoelectromagnetoacoustic_resonance_energies
    assert 6 in model.eng_flexothermoelectromagnetoacoustic_resonance_energies


def test_m341_eng_flexothermoelectromagnetoacoustic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOELECTROMAGNETOACOUSTIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m341_lagmul_wohlhart_hybrid_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{271:>10d}{272:>10d}{273:>10d}{2.25e7:>20.1f}{72:>10d}{3.8e-5:>20.6e}"
    c2 = f"{128.0:>20.4f}{118.0:>20.4f}{102.0:>20.4f}{60.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Wohlhart Hybrid Linkage Joint Fixed Format Test
2022 0
/WOHLHART_HYBRID_LINKAGE_JOINT/235
Wohlhart Hybrid Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 235 in model.lagmul_wohlhart_hybrid_linkage_joints
    joint = model.lagmul_wohlhart_hybrid_linkage_joints[235]
    assert joint.node1 == 271
    assert joint.node2 == 272
    assert joint.node3 == 273
    assert pytest.approx(joint.stiff) == 2.25e7
    assert joint.skew_id == 72
    assert pytest.approx(joint.tol) == 3.8e-5
    assert pytest.approx(joint.link_len_a) == 128.0
    assert pytest.approx(joint.link_len_b) == 118.0
    assert pytest.approx(joint.twist_angle_alpha) == 102.0
    assert pytest.approx(joint.offset_distance_h) == 60.0


def test_m341_lagmul_wohlhart_hybrid_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Wohlhart Hybrid Linkage Joint Free Format Test
/LAGMUL/WOHLHART_HYBRID_LINKAGE_JOINT/236
371, 372, 373, 11.5e6, 92, 4.8e-5
130.0, 120.0, 105.0, 62.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 236 in model.lagmul_wohlhart_hybrid_linkage_joints
    joint = model.lagmul_wohlhart_hybrid_linkage_joints[236]
    assert joint.node1 == 371
    assert joint.node2 == 372
    assert joint.node3 == 373
    assert pytest.approx(joint.stiff) == 11.5e6
    assert joint.skew_id == 92
    assert pytest.approx(joint.tol) == 4.8e-5
    assert pytest.approx(joint.link_len_a) == 130.0
    assert pytest.approx(joint.link_len_b) == 120.0
    assert pytest.approx(joint.twist_angle_alpha) == 105.0
    assert pytest.approx(joint.offset_distance_h) == 62.0


def test_m341_lagmul_wohlhart_hybrid_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Wohlhart Hybrid Linkage Joint Aliases Test
/LAGMUL/WOHLHART_HYBRID_LINKAGE/237
1, 2, 3, 1e6, 0, 1e-6
12.0, 12.0, 35.0, 6.0
/WOHLHART_HYBRID_LINKAGE/238
1, 2, 3, 1e6, 0, 1e-6
12.0, 12.0, 35.0, 6.0
/WOHLHART_HYBRID_MECHANISM/239
1, 2, 3, 1e6, 0, 1e-6
12.0, 12.0, 35.0, 6.0
/WOHLHART_HYBRID_6R_MECHANISM/240
1, 2, 3, 1e6, 0, 1e-6
12.0, 12.0, 35.0, 6.0
/WOHLHART_OVERCONSTRAINED_MECHANISM/241
1, 2, 3, 1e6, 0, 1e-6
12.0, 12.0, 35.0, 6.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 237 in model.lagmul_wohlhart_hybrid_linkage_joints
    assert 238 in model.lagmul_wohlhart_hybrid_linkage_joints
    assert 239 in model.lagmul_wohlhart_hybrid_linkage_joints
    assert 240 in model.lagmul_wohlhart_hybrid_linkage_joints
    assert 241 in model.lagmul_wohlhart_hybrid_linkage_joints


def test_m341_lagmul_wohlhart_hybrid_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/WOHLHART_HYBRID_LINKAGE_JOINT/242
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m341_sensor_spring_bending_drop_rate_fixed(tmp_path: Path):
    c1 = f"{1015:>10d}{4.35e9:>20.1f}{0.0135:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Bending Drop Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_DROP_RATE/268
Spring Bending Drop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 268 in model.sensor_spring_bending_drop_rates
    sensor = model.sensor_spring_bending_drop_rates[268]
    assert sensor.spring_id == 1015
    assert pytest.approx(sensor.jbend_drop_max) == 4.35e9
    assert pytest.approx(sensor.t_delay) == 0.0135
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_DROP_RATE"


def test_m341_sensor_spring_bending_drop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Bending Drop Rate Sensor Free Format Test
/SENSOR/SPRING_BENDING_DROP_RATE/269
1016, 4.45e9, 0.0160
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 269 in model.sensor_spring_bending_drop_rates
    sensor = model.sensor_spring_bending_drop_rates[269]
    assert sensor.spring_id == 1016
    assert pytest.approx(sensor.jbend_drop_max) == 4.45e9
    assert pytest.approx(sensor.t_delay) == 0.0160


def test_m341_sensor_spring_bending_drop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Bending Drop Rate Sensor Aliases Test
/SENSOR/SPRING_BEND_DROP_RATE/270
1017, 4.5e9, 0.0065
/SENSOR/SPRING_RATE_DROP_BEND/271
1018, 4.5e9, 0.0065
/SENSOR/BENDING_DROP_RATE_SPRING/272
1019, 4.5e9, 0.0065
/SENSOR/SPRING_DROP_BEND/273
1020, 4.5e9, 0.0065
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 270 in model.sensor_spring_bending_drop_rates
    assert 271 in model.sensor_spring_bending_drop_rates
    assert 272 in model.sensor_spring_bending_drop_rates
    assert 273 in model.sensor_spring_bending_drop_rates
    assert len(model.sensors) == 4


def test_m341_sensor_spring_bending_drop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_BENDING_DROP_RATE/274
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

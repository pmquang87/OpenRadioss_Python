"""Tests for Milestone M389: LadDynamicFiberMicroBucklingRate Failure Model, EngFlexomagnetophononicplasmonicmagnonicpolaritonicResonanceEnergy, WohlhartHybridSpatialLinkageJoint, and SensorSpringBendingDropRate."""

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


def test_m389_fail_lad_dynamic_fiber_micro_buckling_rate_fixed(tmp_path: Path):
    c1 = f"{335.0:>20.4f}{1005.0:>20.4f}{175.0:>20.4f}{4.85:>20.4f}{0.975:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2030:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Fiber Micro-Buckling Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_FIBER_MICRO_BUCKLING_RATE/2030
Ladeveze Dynamic Fiber Micro-Buckling Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2030 in model.fail_laddynamicfibermicrobucklingrates
    fdfmbr = model.fail_laddynamicfibermicrobucklingrates[2030]
    assert pytest.approx(fdfmbr.sigma_dfmbr0) == 335.0
    assert pytest.approx(fdfmbr.sigma_dfmbrc) == 1005.0
    assert pytest.approx(fdfmbr.gamma_dfmbr) == 175.0
    assert pytest.approx(fdfmbr.p_dfmbr) == 4.85
    assert pytest.approx(fdfmbr.d_dfmbr_max) == 0.975
    assert fdfmbr.ifail_sh == 1
    assert fdfmbr.ifail_so == 2
    assert fdfmbr.fail_id == 2030
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_FIBER_MICRO_BUCKLING_RATE"


def test_m389_fail_lad_dynamic_fiber_micro_buckling_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Fiber Micro-Buckling Rate Free Format Test
/FAIL/LAD_DYNAMIC_FIBER_MICRO_BUCKLING_RATE/2031
345.0, 1035.0, 180.0, 5.05, 0.950
1, 1
2031
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2031 in model.fail_laddynamicfibermicrobucklingrates
    fdfmbr = model.fail_laddynamicfibermicrobucklingrates[2031]
    assert pytest.approx(fdfmbr.sigma_dfmbr0) == 345.0
    assert pytest.approx(fdfmbr.sigma_dfmbrc) == 1035.0
    assert pytest.approx(fdfmbr.gamma_dfmbr) == 180.0
    assert pytest.approx(fdfmbr.p_dfmbr) == 5.05
    assert pytest.approx(fdfmbr.d_dfmbr_max) == 0.950
    assert fdfmbr.fail_id == 2031


def test_m389_fail_lad_dynamic_fiber_micro_buckling_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Fiber Micro-Buckling Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_FIBER_MICRO_BUCKLING_RATE/2032
325.0, 975.0, 166.0, 4.55, 0.995
1, 1
/FAIL/LAD_DFMBR/2033
325.0, 975.0, 166.0, 4.55, 0.995
1, 1
/FAIL/LAD_DFMBR_MODEL/2034
325.0, 975.0, 166.0, 4.55, 0.995
1, 1
/FAIL/LAD_DFMBR_LAW/2035
325.0, 975.0, 166.0, 4.55, 0.995
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_FIBER_MICRO_BUCKLING/2036
325.0, 975.0, 166.0, 4.55, 0.995
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2032 in model.fail_laddynamicfibermicrobucklingrates
    assert 2033 in model.fail_laddynamicfibermicrobucklingrates
    assert 2034 in model.fail_laddynamicfibermicrobucklingrates
    assert 2035 in model.fail_laddynamicfibermicrobucklingrates
    assert 2036 in model.fail_laddynamicfibermicrobucklingrates
    assert len(model.raw_fails) == 5


def test_m389_fail_lad_dynamic_fiber_micro_buckling_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_FIBER_MICRO_BUCKLING_RATE/2037
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m389_eng_flexomagnetophononicplasmonicmagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0090:>20.4f}{335:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetophononicplasmonicmagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPHONONICPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/235
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 235 in model.eng_flexomagnetophononicplasmonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetophononicplasmonicmagnonicpolaritonic_resonance_energies[235]
    assert pytest.approx(eng.dt_fmppmpr) == 0.0090
    assert eng.sens_id == 335


def test_m389_eng_flexomagnetophononicplasmonicmagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetophononicplasmonicmagnonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPHONONICPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/236
0.0100, 336
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 236 in model.eng_flexomagnetophononicplasmonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetophononicplasmonicmagnonicpolaritonic_resonance_energies[236]
    assert pytest.approx(eng.dt_fmppmpr) == 0.0100
    assert eng.sens_id == 336


def test_m389_eng_flexomagnetophononicplasmonicmagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetophononicplasmonicmagnonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PHONON_PLASMON_MAGNON_POLARITON_RES_WORK/237
0.0110, 337
/ENG/EFLEXOMAGNETOPHONONICPLASMONICMAGNONICPOLARITONICRESONANCE/238
0.0120, 338
/ENG/FLEXOMAGNETOPHONONICPLASMONICMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/239
0.0130, 339
/ENG/EM_FLEXOMAGNETOPHONONICPLASMONICMAGNONICPOLARITONIC_RESONANCE/240
0.0140, 340
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 237 in model.eng_flexomagnetophononicplasmonicmagnonicpolaritonic_resonance_energies
    assert 238 in model.eng_flexomagnetophononicplasmonicmagnonicpolaritonic_resonance_energies
    assert 239 in model.eng_flexomagnetophononicplasmonicmagnonicpolaritonic_resonance_energies
    assert 240 in model.eng_flexomagnetophononicplasmonicmagnonicpolaritonic_resonance_energies


def test_m389_eng_flexomagnetophononicplasmonicmagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPHONONICPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/241
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m389_lagmul_wohlhart_hybrid_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{321:>10d}{322:>10d}{323:>10d}{1.76e7:>20.4f}{39:>10d}{8.0e-5:>20.4e}"
    c2 = f"{110.0:>20.4f}{94.0:>20.4f}{120.0:>20.4f}{48.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Wohlhart Hybrid Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT/285
Wohlhart Hybrid Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 285 in model.lagmul_wohlhart_hybrid_spatial_linkage_joints
    joint = model.lagmul_wohlhart_hybrid_spatial_linkage_joints[285]
    assert joint.node1 == 321
    assert joint.node2 == 322
    assert joint.node3 == 323
    assert pytest.approx(joint.stiff) == 1.76e7
    assert joint.skew_id == 39
    assert pytest.approx(joint.tol) == 8.0e-5
    assert pytest.approx(joint.link_len_a) == 110.0
    assert pytest.approx(joint.link_len_b) == 94.0
    assert pytest.approx(joint.twist_angle_alpha) == 120.0
    assert pytest.approx(joint.offset_distance_s) == 48.0
    assert pytest.approx(joint.offset_distance_r) == 48.0
    assert pytest.approx(joint.offset_distance_v) == 48.0
    assert pytest.approx(joint.offset_distance_h) == 48.0
    assert pytest.approx(joint.offset_distance_u) == 48.0


def test_m389_lagmul_wohlhart_hybrid_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Wohlhart Hybrid Spatial Linkage Joint Free Format Test
/LAGMUL/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT/286
421, 422, 423, 2.00e7, 40, 9.0e-5
130.0, 100.0, 140.0, 60.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 286 in model.lagmul_wohlhart_hybrid_spatial_linkage_joints
    joint = model.lagmul_wohlhart_hybrid_spatial_linkage_joints[286]
    assert joint.node1 == 421
    assert joint.node2 == 422
    assert joint.node3 == 423
    assert pytest.approx(joint.stiff) == 2.00e7
    assert joint.skew_id == 40
    assert pytest.approx(joint.tol) == 9.0e-5
    assert pytest.approx(joint.link_len_a) == 130.0
    assert pytest.approx(joint.link_len_b) == 100.0
    assert pytest.approx(joint.twist_angle_alpha) == 140.0
    assert pytest.approx(joint.offset_distance_s) == 60.0


def test_m389_lagmul_wohlhart_hybrid_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Wohlhart Hybrid Spatial Linkage Joint Aliases Test
/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT/287
521, 522, 523, 1.0e6, 0, 1.0e-6
76.0, 66.0, 76.0, 29.0
/LAGMUL/WOHLHART_HYBRID_SPATIAL_LINKAGE/288
521, 522, 523, 1.0e6, 0, 1.0e-6
76.0, 66.0, 76.0, 29.0
/WOHLHART_HYBRID_SPATIAL_LINKAGE/289
521, 522, 523, 1.0e6, 0, 1.0e-6
76.0, 66.0, 76.0, 29.0
/WOHLHART_HYBRID_SPATIAL_MULTI_LOOP_MECHANISM/290
521, 522, 523, 1.0e6, 0, 1.0e-6
76.0, 66.0, 76.0, 29.0
/WOHLHART_HYBRID_SPATIAL_SYMMETRIC_MECHANISM/291
521, 522, 523, 1.0e6, 0, 1.0e-6
76.0, 66.0, 76.0, 29.0
/WOHLHART_HYBRID_SPATIAL_6R_MECHANISM/292
521, 522, 523, 1.0e6, 0, 1.0e-6
76.0, 66.0, 76.0, 29.0
/WOHLHART_HYBRID_SPATIAL_OVERCONSTRAINED_MECHANISM/293
521, 522, 523, 1.0e6, 0, 1.0e-6
76.0, 66.0, 76.0, 29.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(287, 294):
        assert jid in model.lagmul_wohlhart_hybrid_spatial_linkage_joints


def test_m389_lagmul_wohlhart_hybrid_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT/294
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m389_sensor_spring_bending_drop_rate_fixed(tmp_path: Path):
    c1 = f"{1368:>10d}{2.88e8:>20.4f}{0.0875:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Drop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_DROP_RATE/1
Fixed Spring Bending Drop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_drop_rates
    s1 = model.sensor_spring_bending_drop_rates[1]
    assert s1.spring_id == 1368
    assert pytest.approx(s1.jbend_drop_max) == 2.88e8
    assert pytest.approx(s1.jbend_lock_max) == 2.88e8
    assert pytest.approx(s1.jbend_pop_max) == 2.88e8
    assert pytest.approx(s1.jbend_snp_max) == 2.88e8
    assert pytest.approx(s1.jbend_crackle_max) == 2.88e8
    assert pytest.approx(s1.t_delay) == 0.0875
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_DROP_RATE"


def test_m389_sensor_spring_bending_drop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Bending Drop Rate Free Format Test
/SENSOR/SPRING_BENDING_DROP_RATE/2
Free Spring Bending Drop Rate Sensor
1369, 3.48e8, 0.1025
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_bending_drop_rates
    s2 = model.sensor_spring_bending_drop_rates[2]
    assert s2.spring_id == 1369
    assert pytest.approx(s2.jbend_drop_max) == 3.48e8
    assert pytest.approx(s2.t_delay) == 0.1025


def test_m389_sensor_spring_bending_drop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Drop Rate Aliases Test
/SENSOR/SPRING_BEND_DROP_RATE/390
1389, 1.195e8, 0.0195
/SENSOR/SPRING_RATE_DROP_BEND/391
1390, 1.215e8, 0.0205
/SENSOR/BENDING_DROP_RATE_SPRING/392
1391, 1.235e8, 0.0215
/SENSOR/SPRING_DROP_BEND/393
1392, 1.255e8, 0.0225
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(390, 394):
        assert sid in model.sensor_spring_bending_drop_rates


def test_m389_sensor_spring_bending_drop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_BENDING_DROP_RATE/395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

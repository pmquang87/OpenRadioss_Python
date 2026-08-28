"""Tests for Milestone M395: LadDynamicDelaminationRate Failure Model, EngFlexothermoplasmonicmagnonicpolaritonicResonanceEnergy, KirsonSpatialLinkageJoint, and SensorSpringBendingShotRate."""

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


def test_m395_fail_lad_dynamic_delamination_rate_fixed(tmp_path: Path):
    c1 = f"{510.0:>20.4f}{1530.0:>20.4f}{275.0:>20.4f}{6.55:>20.4f}{0.935:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2090:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Delamination Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_DELAMINATION_RATE/2090
Ladeveze Dynamic Delamination Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2090 in model.fail_laddynamicdelaminationrates
    fddlr = model.fail_laddynamicdelaminationrates[2090]
    assert pytest.approx(fddlr.sigma_ddlr0) == 510.0
    assert pytest.approx(fddlr.sigma_ddlrc) == 1530.0
    assert pytest.approx(fddlr.gamma_ddlr) == 275.0
    assert pytest.approx(fddlr.p_ddlr) == 6.55
    assert pytest.approx(fddlr.d_ddlr_max) == 0.935
    assert fddlr.ifail_sh == 1
    assert fddlr.ifail_so == 2
    assert fddlr.fail_id == 2090
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_DELAMINATION_RATE"


def test_m395_fail_lad_dynamic_delamination_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Delamination Rate Free Format Test
/FAIL/LAD_DYNAMIC_DELAMINATION_RATE/2091
520.0, 1560.0, 285.0, 6.75, 0.915
1, 1
2091
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2091 in model.fail_laddynamicdelaminationrates
    fddlr = model.fail_laddynamicdelaminationrates[2091]
    assert pytest.approx(fddlr.sigma_ddlr0) == 520.0
    assert pytest.approx(fddlr.sigma_ddlrc) == 1560.0
    assert pytest.approx(fddlr.gamma_ddlr) == 285.0
    assert pytest.approx(fddlr.p_ddlr) == 6.75
    assert pytest.approx(fddlr.d_ddlr_max) == 0.915
    assert fddlr.fail_id == 2091


def test_m395_fail_lad_dynamic_delamination_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Delamination Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_DELAMINATION_RATE/2092
500.0, 1500.0, 270.0, 6.35, 0.945
1, 1
/FAIL/LAD_DDLR/2093
500.0, 1500.0, 270.0, 6.35, 0.945
1, 1
/FAIL/LAD_DDLR_MODEL/2094
500.0, 1500.0, 270.0, 6.35, 0.945
1, 1
/FAIL/LAD_DDLR_LAW/2095
500.0, 1500.0, 270.0, 6.35, 0.945
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_DELAMINATION/2096
500.0, 1500.0, 270.0, 6.35, 0.945
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2092 in model.fail_laddynamicdelaminationrates
    assert 2093 in model.fail_laddynamicdelaminationrates
    assert 2094 in model.fail_laddynamicdelaminationrates
    assert 2095 in model.fail_laddynamicdelaminationrates
    assert 2096 in model.fail_laddynamicdelaminationrates
    assert len(model.raw_fails) == 5


def test_m395_fail_lad_dynamic_delamination_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_DELAMINATION_RATE/2097
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m395_eng_flexothermoplasmonicmagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0175:>20.4f}{395:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexothermoplasmonicmagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/295
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 295 in model.eng_flexothermoplasmonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonicmagnonicpolaritonic_resonance_energies[295]
    assert pytest.approx(eng.dt_ftpmp) == 0.0175
    assert eng.sens_id == 395


def test_m395_eng_flexothermoplasmonicmagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexothermoplasmonicmagnonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/296
0.0185, 396
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 296 in model.eng_flexothermoplasmonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonicmagnonicpolaritonic_resonance_energies[296]
    assert pytest.approx(eng.dt_ftpmp) == 0.0185
    assert eng.sens_id == 396


def test_m395_eng_flexothermoplasmonicmagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexothermoplasmonicmagnonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PLASMONIC_MAGNONIC_POLARITON_RES_WORK/297
0.0195, 397
/ENG/EFLEXOTHERMOPLASMONICMAGNONICPOLARITONICRESONANCE/298
0.0205, 398
/ENG/FLEXOTHERMOPLASMONICMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/299
0.0215, 399
/ENG/ET_FLEXOTHERMOPLASMONICMAGNONICPOLARITONIC_RESONANCE/300
0.0225, 400
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 297 in model.eng_flexothermoplasmonicmagnonicpolaritonic_resonance_energies
    assert 298 in model.eng_flexothermoplasmonicmagnonicpolaritonic_resonance_energies
    assert 299 in model.eng_flexothermoplasmonicmagnonicpolaritonic_resonance_energies
    assert 300 in model.eng_flexothermoplasmonicmagnonicpolaritonic_resonance_energies


def test_m395_eng_flexothermoplasmonicmagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOTHERMOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/301
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m395_lagmul_kirson_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{381:>10d}{382:>10d}{383:>10d}{2.36e7:>20.4f}{99:>10d}{1.35e-4:>20.4e}"
    c2 = f"{165.0:>20.4f}{148.0:>20.4f}{175.0:>20.4f}{98.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Kirson Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/KIRSON_SPATIAL_LINKAGE_JOINT/345
Kirson Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 345 in model.lagmul_kirson_spatial_linkage_joints
    joint = model.lagmul_kirson_spatial_linkage_joints[345]
    assert joint.node1 == 381
    assert joint.node2 == 382
    assert joint.node3 == 383
    assert pytest.approx(joint.stiff) == 2.36e7
    assert joint.skew_id == 99
    assert pytest.approx(joint.tol) == 1.35e-4
    assert pytest.approx(joint.link_len_a) == 165.0
    assert pytest.approx(joint.link_len_b) == 148.0
    assert pytest.approx(joint.twist_angle_alpha) == 175.0
    assert pytest.approx(joint.offset_distance_s) == 98.0
    assert pytest.approx(joint.offset_distance_r) == 98.0
    assert pytest.approx(joint.offset_distance_v) == 98.0
    assert pytest.approx(joint.offset_distance_h) == 98.0
    assert pytest.approx(joint.offset_distance_u) == 98.0


def test_m395_lagmul_kirson_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Kirson Spatial Linkage Joint Free Format Test
/LAGMUL/KIRSON_SPATIAL_LINKAGE_JOINT/346
481, 482, 483, 2.60e7, 100, 1.45e-4
185.0, 155.0, 195.0, 110.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 346 in model.lagmul_kirson_spatial_linkage_joints
    joint = model.lagmul_kirson_spatial_linkage_joints[346]
    assert joint.node1 == 481
    assert joint.node2 == 482
    assert joint.node3 == 483
    assert pytest.approx(joint.stiff) == 2.60e7
    assert joint.skew_id == 100
    assert pytest.approx(joint.tol) == 1.45e-4
    assert pytest.approx(joint.link_len_a) == 185.0
    assert pytest.approx(joint.link_len_b) == 155.0
    assert pytest.approx(joint.twist_angle_alpha) == 195.0
    assert pytest.approx(joint.offset_distance_s) == 110.0


def test_m395_lagmul_kirson_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Kirson Spatial Linkage Joint Aliases Test
/KIRSON_SPATIAL_LINKAGE_JOINT/347
581, 582, 583, 1.0e6, 0, 1.0e-6
125.0, 115.0, 125.0, 75.0
/LAGMUL/KIRSON_SPATIAL_LINKAGE/348
581, 582, 583, 1.0e6, 0, 1.0e-6
125.0, 115.0, 125.0, 75.0
/KIRSON_SPATIAL_LINKAGE/349
581, 582, 583, 1.0e6, 0, 1.0e-6
125.0, 115.0, 125.0, 75.0
/KIRSON_SPATIAL_MULTI_LOOP_MECHANISM/350
581, 582, 583, 1.0e6, 0, 1.0e-6
125.0, 115.0, 125.0, 75.0
/KIRSON_SPATIAL_SYMMETRIC_MECHANISM/351
581, 582, 583, 1.0e6, 0, 1.0e-6
125.0, 115.0, 125.0, 75.0
/KIRSON_SPATIAL_6R_MECHANISM/352
581, 582, 583, 1.0e6, 0, 1.0e-6
125.0, 115.0, 125.0, 75.0
/KIRSON_SPATIAL_OVERCONSTRAINED_MECHANISM/353
581, 582, 583, 1.0e6, 0, 1.0e-6
125.0, 115.0, 125.0, 75.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(347, 354):
        assert jid in model.lagmul_kirson_spatial_linkage_joints


def test_m395_lagmul_kirson_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/KIRSON_SPATIAL_LINKAGE_JOINT/354
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m395_sensor_spring_bending_shot_rate_fixed(tmp_path: Path):
    c1 = f"{1428:>10d}{3.58e8:>20.4f}{0.1105:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Shot Rate Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_SHOT_RATE/1
Fixed Spring Bending Shot Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_shot_rates
    s1 = model.sensor_spring_bending_shot_rates[1]
    assert s1.spring_id == 1428
    assert pytest.approx(s1.jbend_shot_max) == 3.58e8
    assert pytest.approx(s1.jbend_drop_max) == 3.58e8
    assert pytest.approx(s1.jbend_lock_max) == 3.58e8
    assert pytest.approx(s1.jbend_pop_max) == 3.58e8
    assert pytest.approx(s1.jbend_snp_max) == 3.58e8
    assert pytest.approx(s1.jbend_crackle_max) == 3.58e8
    assert pytest.approx(s1.t_delay) == 0.1105
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_SHOT_RATE"


def test_m395_sensor_spring_bending_shot_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Bending Shot Rate Free Format Test
/SENSOR/SPRING_BENDING_SHOT_RATE/2
Free Spring Bending Shot Rate Sensor
1429, 4.18e8, 0.1495
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_bending_shot_rates
    s2 = model.sensor_spring_bending_shot_rates[2]
    assert s2.spring_id == 1429
    assert pytest.approx(s2.jbend_shot_max) == 4.18e8
    assert pytest.approx(s2.t_delay) == 0.1495


def test_m395_sensor_spring_bending_shot_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Shot Rate Aliases Test
/SENSOR/SPRING_BEND_SHOT_RATE/391
1449, 1.795e8, 0.0665
/SENSOR/SPRING_RATE_SHOT_BEND/392
1450, 1.815e8, 0.0675
/SENSOR/BENDING_SHOT_RATE_SPRING/393
1451, 1.835e8, 0.0685
/SENSOR/SPRING_SHOT_BEND/394
1452, 1.855e8, 0.0695
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(391, 395):
        assert sid in model.sensor_spring_bending_shot_rates


def test_m395_sensor_spring_bending_shot_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_BENDING_SHOT_RATE/395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

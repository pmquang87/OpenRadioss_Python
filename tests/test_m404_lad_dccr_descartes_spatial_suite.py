"""Tests for Milestone M404: LadDynamicCoreCrushingRate Failure Model, EngFlexothermomagnonicphononicpolaritonicResonanceEnergy, DescartesSpatialLinkageJoint, and SensorSpringTransverseCrackleRate."""

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


def test_m404_fail_lad_dynamic_core_crushing_rate_fixed(tmp_path: Path):
    c1 = f"{580.0:>20.4f}{1740.0:>20.4f}{355.0:>20.4f}{7.45:>20.4f}{0.845:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2180:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Core Crushing Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_CORE_CRUSHING_RATE/2180
Ladeveze Dynamic Core Crushing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2180 in model.fail_laddynamiccorecrushingrates
    fdccr = model.fail_laddynamiccorecrushingrates[2180]
    assert pytest.approx(fdccr.sigma_dccr0) == 580.0
    assert pytest.approx(fdccr.sigma_dccrc) == 1740.0
    assert pytest.approx(fdccr.gamma_dccr) == 355.0
    assert pytest.approx(fdccr.p_dccr) == 7.45
    assert pytest.approx(fdccr.d_dccr_max) == 0.845
    assert fdccr.ifail_sh == 1
    assert fdccr.ifail_so == 2
    assert fdccr.fail_id == 2180
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_CORE_CRUSHING_RATE"


def test_m404_fail_lad_dynamic_core_crushing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Core Crushing Rate Free Format Test
/FAIL/LAD_DYNAMIC_CORE_CRUSHING_RATE/2181
590.0, 1770.0, 365.0, 7.65, 0.835
1, 1
2181
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2181 in model.fail_laddynamiccorecrushingrates
    fdccr = model.fail_laddynamiccorecrushingrates[2181]
    assert pytest.approx(fdccr.sigma_dccr0) == 590.0
    assert pytest.approx(fdccr.sigma_dccrc) == 1770.0
    assert pytest.approx(fdccr.gamma_dccr) == 365.0
    assert pytest.approx(fdccr.p_dccr) == 7.65
    assert pytest.approx(fdccr.d_dccr_max) == 0.835
    assert fdccr.fail_id == 2181


def test_m404_fail_lad_dynamic_core_crushing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Core Crushing Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_CORE_CRUSHING_RATE/2182
570.0, 1710.0, 350.0, 7.25, 0.865
1, 1
/FAIL/LAD_DCCR/2183
570.0, 1710.0, 350.0, 7.25, 0.865
1, 1
/FAIL/LAD_DCCR_MODEL/2184
570.0, 1710.0, 350.0, 7.25, 0.865
1, 1
/FAIL/LAD_DCCR_LAW/2185
570.0, 1710.0, 350.0, 7.25, 0.865
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_CORE_CRUSHING/2186
570.0, 1710.0, 350.0, 7.25, 0.865
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2182 in model.fail_laddynamiccorecrushingrates
    assert 2183 in model.fail_laddynamiccorecrushingrates
    assert 2184 in model.fail_laddynamiccorecrushingrates
    assert 2185 in model.fail_laddynamiccorecrushingrates
    assert 2186 in model.fail_laddynamiccorecrushingrates
    assert len(model.raw_fails) == 5


def test_m404_fail_lad_dynamic_core_crushing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_CORE_CRUSHING_RATE/2187
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m404_eng_flexothermomagnonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0265:>20.4f}{485:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexothermomagnonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/385
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 385 in model.eng_flexothermomagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_flexothermomagnonicphononicpolaritonic_resonance_energies[385]
    assert pytest.approx(eng.dt_ftmpp) == 0.0265
    assert eng.sens_id == 485


def test_m404_eng_flexothermomagnonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexothermomagnonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/386
0.0275, 486
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 386 in model.eng_flexothermomagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_flexothermomagnonicphononicpolaritonic_resonance_energies[386]
    assert pytest.approx(eng.dt_ftmpp) == 0.0275
    assert eng.sens_id == 486


def test_m404_eng_flexothermomagnonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexothermomagnonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_MAGNONIC_PHONONIC_POLARITON_RES_WORK/387
0.0285, 487
/ENG/EFLEXOTHERMOMAGNONICPHONONICPOLARITONICRESONANCE/388
0.0295, 488
/ENG/FLEXOTHERMOMAGNONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/389
0.0305, 489
/ENG/ET_FLEXOTHERMOMAGNONICPHONONICPOLARITONIC_RESONANCE/390
0.0315, 490
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 387 in model.eng_flexothermomagnonicphononicpolaritonic_resonance_energies
    assert 388 in model.eng_flexothermomagnonicphononicpolaritonic_resonance_energies
    assert 389 in model.eng_flexothermomagnonicphononicpolaritonic_resonance_energies
    assert 390 in model.eng_flexothermomagnonicphononicpolaritonic_resonance_energies


def test_m404_eng_flexothermomagnonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOTHERMOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/391
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m404_lagmul_descartes_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{471:>10d}{472:>10d}{473:>10d}{3.26e7:>20.4f}{189:>10d}{2.25e-4:>20.4e}"
    c2 = f"{255.0:>20.4f}{238.0:>20.4f}{265.0:>20.4f}{188.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Descartes Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/DESCARTES_SPATIAL_LINKAGE_JOINT/435
Descartes Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 435 in model.lagmul_descartes_spatial_linkage_joints
    joint = model.lagmul_descartes_spatial_linkage_joints[435]
    assert joint.node1 == 471
    assert joint.node2 == 472
    assert joint.node3 == 473
    assert pytest.approx(joint.stiff) == 3.26e7
    assert joint.skew_id == 189
    assert pytest.approx(joint.tol) == 2.25e-4
    assert pytest.approx(joint.link_len_a) == 255.0
    assert pytest.approx(joint.link_len_b) == 238.0
    assert pytest.approx(joint.twist_angle_alpha) == 265.0
    assert pytest.approx(joint.offset_distance_s) == 188.0
    assert pytest.approx(joint.offset_distance_r) == 188.0
    assert pytest.approx(joint.offset_distance_v) == 188.0
    assert pytest.approx(joint.offset_distance_h) == 188.0
    assert pytest.approx(joint.offset_distance_u) == 188.0


def test_m404_lagmul_descartes_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Descartes Spatial Linkage Joint Free Format Test
/LAGMUL/DESCARTES_SPATIAL_LINKAGE_JOINT/436
571, 572, 573, 3.50e7, 190, 2.35e-4
275.0, 245.0, 285.0, 200.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 436 in model.lagmul_descartes_spatial_linkage_joints
    joint = model.lagmul_descartes_spatial_linkage_joints[436]
    assert joint.node1 == 571
    assert joint.node2 == 572
    assert joint.node3 == 573
    assert pytest.approx(joint.stiff) == 3.50e7
    assert joint.skew_id == 190
    assert pytest.approx(joint.tol) == 2.35e-4
    assert pytest.approx(joint.link_len_a) == 275.0
    assert pytest.approx(joint.link_len_b) == 245.0
    assert pytest.approx(joint.twist_angle_alpha) == 285.0
    assert pytest.approx(joint.offset_distance_s) == 200.0


def test_m404_lagmul_descartes_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Descartes Spatial Linkage Joint Aliases Test
/DESCARTES_SPATIAL_LINKAGE_JOINT/437
671, 672, 673, 1.0e6, 0, 1.0e-6
215.0, 205.0, 215.0, 165.0
/LAGMUL/DESCARTES_SPATIAL_LINKAGE/438
671, 672, 673, 1.0e6, 0, 1.0e-6
215.0, 205.0, 215.0, 165.0
/DESCARTES_SPATIAL_LINKAGE/439
671, 672, 673, 1.0e6, 0, 1.0e-6
215.0, 205.0, 215.0, 165.0
/DESCARTES_SPATIAL_MULTI_LOOP_MECHANISM/440
671, 672, 673, 1.0e6, 0, 1.0e-6
215.0, 205.0, 215.0, 165.0
/DESCARTES_SPATIAL_SYMMETRIC_MECHANISM/441
671, 672, 673, 1.0e6, 0, 1.0e-6
215.0, 205.0, 215.0, 165.0
/DESCARTES_SPATIAL_6R_MECHANISM/442
671, 672, 673, 1.0e6, 0, 1.0e-6
215.0, 205.0, 215.0, 165.0
/DESCARTES_SPATIAL_OVERCONSTRAINED_MECHANISM/443
671, 672, 673, 1.0e6, 0, 1.0e-6
215.0, 205.0, 215.0, 165.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(437, 444):
        assert jid in model.lagmul_descartes_spatial_linkage_joints


def test_m404_lagmul_descartes_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/DESCARTES_SPATIAL_LINKAGE_JOINT/444
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m404_sensor_spring_transverse_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{1518:>10d}{4.48e8:>20.4f}{0.1465:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Crackle Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE/1
Fixed Spring Transverse Crackle Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_crackle_rates
    s1 = model.sensor_spring_transverse_crackle_rates[1]
    assert s1.spring_id == 1518
    assert pytest.approx(s1.jtrans_crk_max) == 4.48e8
    assert pytest.approx(s1.jtrans_shot_max) == 4.48e8
    assert pytest.approx(s1.jtrans_drop_max) == 4.48e8
    assert pytest.approx(s1.jtrans_lock_max) == 4.48e8
    assert pytest.approx(s1.jtrans_pop_max) == 4.48e8
    assert pytest.approx(s1.jtrans_snp_max) == 4.48e8
    assert pytest.approx(s1.t_delay) == 0.1465
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_CRACKLE_RATE"


def test_m404_sensor_spring_transverse_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Transverse Crackle Rate Free Format Test
/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE/2
Free Spring Transverse Crackle Rate Sensor
1519, 5.08e8, 0.2395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_transverse_crackle_rates
    s2 = model.sensor_spring_transverse_crackle_rates[2]
    assert s2.spring_id == 1519
    assert pytest.approx(s2.jtrans_crk_max) == 5.08e8
    assert pytest.approx(s2.t_delay) == 0.2395


def test_m404_sensor_spring_transverse_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Crackle Rate Aliases Test
/SENSOR/SPRING_TRANS_CRACKLE_RATE/391
1539, 2.695e8, 0.1565
/SENSOR/SPRING_RATE_CRACKLE_TRANS/392
1540, 2.715e8, 0.1575
/SENSOR/TRANSVERSE_CRACKLE_RATE_SPRING/393
1541, 2.735e8, 0.1585
/SENSOR/SPRING_CRACKLE_TRANS/394
1542, 2.755e8, 0.1595
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(391, 395):
        assert sid in model.sensor_spring_transverse_crackle_rates


def test_m404_sensor_spring_transverse_crackle_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE/395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

"""Tests for Milestone M410: LadDynamicCoreDebondingRate Failure Model, EngElectrothermoflexoexcitonicphononicpolaritonicResonanceEnergy, LaplaceSpatialLinkageJoint, and SensorSpringTransversePopRate."""

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


def test_m410_fail_lad_dynamic_core_debonding_rate_fixed(tmp_path: Path):
    c1 = f"{620.0:>20.4f}{1860.0:>20.4f}{400.0:>20.4f}{8.35:>20.4f}{0.810:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2240:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Core Debonding Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_CORE_DEBONDING_RATE/2240
Ladeveze Dynamic Core Debonding Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2240 in model.fail_laddynamiccoredebondingrates
    fdcd = model.fail_laddynamiccoredebondingrates[2240]
    assert pytest.approx(fdcd.sigma_dcd0) == 620.0
    assert pytest.approx(fdcd.sigma_dcdc) == 1860.0
    assert pytest.approx(fdcd.gamma_dcd) == 400.0
    assert pytest.approx(fdcd.p_dcd) == 8.35
    assert pytest.approx(fdcd.d_dcd_max) == 0.810
    assert fdcd.ifail_sh == 1
    assert fdcd.ifail_so == 2
    assert fdcd.fail_id == 2240
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_CORE_DEBONDING_RATE"


def test_m410_fail_lad_dynamic_core_debonding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Core Debonding Rate Free Format Test
/FAIL/LAD_DYNAMIC_CORE_DEBONDING_RATE/2241
630.0, 1890.0, 410.0, 8.55, 0.800
1, 1
2241
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2241 in model.fail_laddynamiccoredebondingrates
    fdcd = model.fail_laddynamiccoredebondingrates[2241]
    assert pytest.approx(fdcd.sigma_dcd0) == 630.0
    assert pytest.approx(fdcd.sigma_dcdc) == 1890.0
    assert pytest.approx(fdcd.gamma_dcd) == 410.0
    assert pytest.approx(fdcd.p_dcd) == 8.55
    assert pytest.approx(fdcd.d_dcd_max) == 0.800
    assert fdcd.fail_id == 2241


def test_m410_fail_lad_dynamic_core_debonding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Core Debonding Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_CORE_DEBONDING_RATE/2242
610.0, 1830.0, 395.0, 8.15, 0.820
1, 1
/FAIL/LAD_DCDR/2243
610.0, 1830.0, 395.0, 8.15, 0.820
1, 1
/FAIL/LAD_DCDR_MODEL/2244
610.0, 1830.0, 395.0, 8.15, 0.820
1, 1
/FAIL/LAD_DCDR_LAW/2245
610.0, 1830.0, 395.0, 8.15, 0.820
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_CORE_DEBONDING/2246
610.0, 1830.0, 395.0, 8.15, 0.820
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2242 in model.fail_laddynamiccoredebondingrates
    assert 2243 in model.fail_laddynamiccoredebondingrates
    assert 2244 in model.fail_laddynamiccoredebondingrates
    assert 2245 in model.fail_laddynamiccoredebondingrates
    assert 2246 in model.fail_laddynamiccoredebondingrates
    assert len(model.raw_fails) == 5


def test_m410_fail_lad_dynamic_core_debonding_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_CORE_DEBONDING_RATE/2247
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m410_eng_electrothermoflexoexcitonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0320:>20.4f}{545:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexoexcitonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/445
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 445 in model.eng_electrothermoflexoexcitonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexoexcitonicphononicpolaritonic_resonance_energies[445]
    assert pytest.approx(eng.dt_etfepp) == 0.0320
    assert eng.sens_id == 545


def test_m410_eng_electrothermoflexoexcitonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexoexcitonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/446
0.0330, 546
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 446 in model.eng_electrothermoflexoexcitonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexoexcitonicphononicpolaritonic_resonance_energies[446]
    assert pytest.approx(eng.dt_etfepp) == 0.0330
    assert eng.sens_id == 546


def test_m410_eng_electrothermoflexoexcitonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexoexcitonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_EXCITON_PHONON_POLARITON_RES_WORK/447
0.0340, 547
/ENG/EELECTROTHERMOFLEXOEXCITONICPHONONICPOLARITONICRESONANCE/448
0.0350, 548
/ENG/ELECTROTHERMOFLEXOEXCITONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/449
0.0360, 549
/ENG/ET_ELECTROTHERMOFLEXOEXCITONICPHONONICPOLARITONIC_RESONANCE/450
0.0370, 550
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 447 in model.eng_electrothermoflexoexcitonicphononicpolaritonic_resonance_energies
    assert 448 in model.eng_electrothermoflexoexcitonicphononicpolaritonic_resonance_energies
    assert 449 in model.eng_electrothermoflexoexcitonicphononicpolaritonic_resonance_energies
    assert 450 in model.eng_electrothermoflexoexcitonicphononicpolaritonic_resonance_energies


def test_m410_eng_electrothermoflexoexcitonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/451
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m410_lagmul_laplace_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{531:>10d}{532:>10d}{533:>10d}{3.85e7:>20.4f}{249:>10d}{2.80e-4:>20.4e}"
    c2 = f"{310.0:>20.4f}{290.0:>20.4f}{320.0:>20.4f}{240.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Laplace Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/LAPLACE_SPATIAL_LINKAGE_JOINT/495
Laplace Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 495 in model.lagmul_laplace_spatial_linkage_joints
    joint = model.lagmul_laplace_spatial_linkage_joints[495]
    assert joint.node1 == 531
    assert joint.node2 == 532
    assert joint.node3 == 533
    assert pytest.approx(joint.stiff) == 3.85e7
    assert joint.skew_id == 249
    assert pytest.approx(joint.tol) == 2.80e-4
    assert pytest.approx(joint.link_len_a) == 310.0
    assert pytest.approx(joint.link_len_b) == 290.0
    assert pytest.approx(joint.twist_angle_alpha) == 320.0
    assert pytest.approx(joint.offset_distance_s) == 240.0
    assert pytest.approx(joint.offset_distance_r) == 240.0
    assert pytest.approx(joint.offset_distance_v) == 240.0
    assert pytest.approx(joint.offset_distance_h) == 240.0
    assert pytest.approx(joint.offset_distance_u) == 240.0


def test_m410_lagmul_laplace_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Laplace Spatial Linkage Joint Free Format Test
/LAGMUL/LAPLACE_SPATIAL_LINKAGE_JOINT/496
631, 632, 633, 4.10e7, 250, 2.90e-4
330.0, 300.0, 340.0, 255.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 496 in model.lagmul_laplace_spatial_linkage_joints
    joint = model.lagmul_laplace_spatial_linkage_joints[496]
    assert joint.node1 == 631
    assert joint.node2 == 632
    assert joint.node3 == 633
    assert pytest.approx(joint.stiff) == 4.10e7
    assert joint.skew_id == 250
    assert pytest.approx(joint.tol) == 2.90e-4
    assert pytest.approx(joint.link_len_a) == 330.0
    assert pytest.approx(joint.link_len_b) == 300.0
    assert pytest.approx(joint.twist_angle_alpha) == 340.0
    assert pytest.approx(joint.offset_distance_s) == 255.0


def test_m410_lagmul_laplace_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Laplace Spatial Linkage Joint Aliases Test
/LAPLACE_SPATIAL_LINKAGE_JOINT/497
731, 732, 733, 1.0e6, 0, 1.0e-6
270.0, 260.0, 270.0, 220.0
/LAGMUL/LAPLACE_SPATIAL_LINKAGE/498
731, 732, 733, 1.0e6, 0, 1.0e-6
270.0, 260.0, 270.0, 220.0
/LAPLACE_SPATIAL_LINKAGE/499
731, 732, 733, 1.0e6, 0, 1.0e-6
270.0, 260.0, 270.0, 220.0
/LAPLACE_SPATIAL_MULTI_LOOP_MECHANISM/500
731, 732, 733, 1.0e6, 0, 1.0e-6
270.0, 260.0, 270.0, 220.0
/LAPLACE_SPATIAL_SYMMETRIC_MECHANISM/501
731, 732, 733, 1.0e6, 0, 1.0e-6
270.0, 260.0, 270.0, 220.0
/LAPLACE_SPATIAL_6R_MECHANISM/502
731, 732, 733, 1.0e6, 0, 1.0e-6
270.0, 260.0, 270.0, 220.0
/LAPLACE_SPATIAL_OVERCONSTRAINED_MECHANISM/503
731, 732, 733, 1.0e6, 0, 1.0e-6
270.0, 260.0, 270.0, 220.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(497, 504):
        assert jid in model.lagmul_laplace_spatial_linkage_joints


def test_m410_lagmul_laplace_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/LAPLACE_SPATIAL_LINKAGE_JOINT/504
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m410_sensor_spring_transverse_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1578:>10d}{5.08e8:>20.4f}{0.1645:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_POP_RATE/1
Fixed Spring Transverse Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_pop_rates
    s1 = model.sensor_spring_transverse_pop_rates[1]
    assert s1.spring_id == 1578
    assert pytest.approx(s1.jtrans_pop_max) == 5.08e8
    assert pytest.approx(s1.jtrans_snp_max) == 5.08e8
    assert pytest.approx(s1.jtrans_crackle_max) == 5.08e8
    assert pytest.approx(s1.jtrans_shot_max) == 5.08e8
    assert pytest.approx(s1.jtrans_drop_max) == 5.08e8
    assert pytest.approx(s1.jtrans_lock_max) == 5.08e8
    assert pytest.approx(s1.jtrans_crk_max) == 5.08e8
    assert pytest.approx(s1.t_delay) == 0.1645
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_POP_RATE"


def test_m410_sensor_spring_transverse_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Transverse Pop Rate Free Format Test
/SENSOR/SPRING_TRANSVERSE_POP_RATE/2
Free Spring Transverse Pop Rate Sensor
1579, 5.68e8, 0.2635
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_transverse_pop_rates
    s2 = model.sensor_spring_transverse_pop_rates[2]
    assert s2.spring_id == 1579
    assert pytest.approx(s2.jtrans_pop_max) == 5.68e8
    assert pytest.approx(s2.t_delay) == 0.2635


def test_m410_sensor_spring_transverse_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Pop Rate Aliases Test
/SENSOR/SPRING_TRANS_POP_RATE/441
1599, 3.295e8, 0.2165
/SENSOR/SPRING_RATE_POP_TRANS/442
1600, 3.315e8, 0.2175
/SENSOR/TRANSVERSE_POP_RATE_SPRING/443
1601, 3.335e8, 0.2185
/SENSOR/SPRING_POP_TRANS/444
1602, 3.355e8, 0.2195
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(441, 445):
        assert sid in model.sensor_spring_transverse_pop_rates


def test_m410_sensor_spring_transverse_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TRANSVERSE_POP_RATE/445
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

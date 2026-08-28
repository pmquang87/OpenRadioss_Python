"""Tests for Milestone M406: LadCoupleCoreCrushingRate Failure Model, EngFlexothermoplasmonicmagnonicphononicpolaritonicResonanceEnergy, NewtonSpatialLinkageJoint, and SensorSpringTorsionalCrackleRate."""

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


def test_m406_fail_lad_couple_core_crushing_rate_fixed(tmp_path: Path):
    c1 = f"{585.0:>20.4f}{1755.0:>20.4f}{365.0:>20.4f}{7.65:>20.4f}{0.835:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2200:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Core Crushing Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_CORE_CRUSHING_RATE/2200
Ladeveze Coupled Core Crushing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2200 in model.fail_ladcouplecorecrushingrates
    fcccr = model.fail_ladcouplecorecrushingrates[2200]
    assert pytest.approx(fcccr.sigma_cccr0) == 585.0
    assert pytest.approx(fcccr.sigma_cccrc) == 1755.0
    assert pytest.approx(fcccr.gamma_cccr) == 365.0
    assert pytest.approx(fcccr.p_cccr) == 7.65
    assert pytest.approx(fcccr.d_cccr_max) == 0.835
    assert fcccr.ifail_sh == 1
    assert fcccr.ifail_so == 2
    assert fcccr.fail_id == 2200
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_CORE_CRUSHING_RATE"


def test_m406_fail_lad_couple_core_crushing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Core Crushing Rate Free Format Test
/FAIL/LAD_COUPLE_CORE_CRUSHING_RATE/2201
595.0, 1785.0, 375.0, 7.85, 0.825
1, 1
2201
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2201 in model.fail_ladcouplecorecrushingrates
    fcccr = model.fail_ladcouplecorecrushingrates[2201]
    assert pytest.approx(fcccr.sigma_cccr0) == 595.0
    assert pytest.approx(fcccr.sigma_cccrc) == 1785.0
    assert pytest.approx(fcccr.gamma_cccr) == 375.0
    assert pytest.approx(fcccr.p_cccr) == 7.85
    assert pytest.approx(fcccr.d_cccr_max) == 0.825
    assert fcccr.fail_id == 2201


def test_m406_fail_lad_couple_core_crushing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Core Crushing Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_CORE_CRUSHING_RATE/2202
575.0, 1725.0, 360.0, 7.45, 0.855
1, 1
/FAIL/LAD_CCCR/2203
575.0, 1725.0, 360.0, 7.45, 0.855
1, 1
/FAIL/LAD_CCCR_MODEL/2204
575.0, 1725.0, 360.0, 7.45, 0.855
1, 1
/FAIL/LAD_CCCR_LAW/2205
575.0, 1725.0, 360.0, 7.45, 0.855
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_CORE_CRUSHING/2206
575.0, 1725.0, 360.0, 7.45, 0.855
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2202 in model.fail_ladcouplecorecrushingrates
    assert 2203 in model.fail_ladcouplecorecrushingrates
    assert 2204 in model.fail_ladcouplecorecrushingrates
    assert 2205 in model.fail_ladcouplecorecrushingrates
    assert 2206 in model.fail_ladcouplecorecrushingrates
    assert len(model.raw_fails) == 5


def test_m406_fail_lad_couple_core_crushing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_CORE_CRUSHING_RATE/2207
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m406_eng_flexothermoplasmonicmagnonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0285:>20.4f}{505:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexothermoplasmonicmagnonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/405
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 405 in model.eng_flexothermoplasmonicmagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonicmagnonicphononicpolaritonic_resonance_energies[405]
    assert pytest.approx(eng.dt_ftpmpp) == 0.0285
    assert eng.sens_id == 505


def test_m406_eng_flexothermoplasmonicmagnonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexothermoplasmonicmagnonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/406
0.0295, 506
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 406 in model.eng_flexothermoplasmonicmagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonicmagnonicphononicpolaritonic_resonance_energies[406]
    assert pytest.approx(eng.dt_ftpmpp) == 0.0295
    assert eng.sens_id == 506


def test_m406_eng_flexothermoplasmonicmagnonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexothermoplasmonicmagnonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PLASMONIC_MAGNONIC_PHONONIC_POLARITON_RES_WORK/407
0.0305, 507
/ENG/EFLEXOTHERMOPLASMONICMAGNONICPHONONICPOLARITONICRESONANCE/408
0.0315, 508
/ENG/FLEXOTHERMOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/409
0.0325, 509
/ENG/ET_FLEXOTHERMOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE/410
0.0335, 510
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 407 in model.eng_flexothermoplasmonicmagnonicphononicpolaritonic_resonance_energies
    assert 408 in model.eng_flexothermoplasmonicmagnonicphononicpolaritonic_resonance_energies
    assert 409 in model.eng_flexothermoplasmonicmagnonicphononicpolaritonic_resonance_energies
    assert 410 in model.eng_flexothermoplasmonicmagnonicphononicpolaritonic_resonance_energies


def test_m406_eng_flexothermoplasmonicmagnonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOTHERMOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/411
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m406_lagmul_newton_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{491:>10d}{492:>10d}{493:>10d}{3.46e7:>20.4f}{209:>10d}{2.45e-4:>20.4e}"
    c2 = f"{275.0:>20.4f}{258.0:>20.4f}{285.0:>20.4f}{208.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Newton Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/NEWTON_SPATIAL_LINKAGE_JOINT/455
Newton Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 455 in model.lagmul_newton_spatial_linkage_joints
    joint = model.lagmul_newton_spatial_linkage_joints[455]
    assert joint.node1 == 491
    assert joint.node2 == 492
    assert joint.node3 == 493
    assert pytest.approx(joint.stiff) == 3.46e7
    assert joint.skew_id == 209
    assert pytest.approx(joint.tol) == 2.45e-4
    assert pytest.approx(joint.link_len_a) == 275.0
    assert pytest.approx(joint.link_len_b) == 258.0
    assert pytest.approx(joint.twist_angle_alpha) == 285.0
    assert pytest.approx(joint.offset_distance_s) == 208.0
    assert pytest.approx(joint.offset_distance_r) == 208.0
    assert pytest.approx(joint.offset_distance_v) == 208.0
    assert pytest.approx(joint.offset_distance_h) == 208.0
    assert pytest.approx(joint.offset_distance_u) == 208.0


def test_m406_lagmul_newton_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Newton Spatial Linkage Joint Free Format Test
/LAGMUL/NEWTON_SPATIAL_LINKAGE_JOINT/456
591, 592, 593, 3.70e7, 210, 2.55e-4
295.0, 265.0, 305.0, 220.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 456 in model.lagmul_newton_spatial_linkage_joints
    joint = model.lagmul_newton_spatial_linkage_joints[456]
    assert joint.node1 == 591
    assert joint.node2 == 592
    assert joint.node3 == 593
    assert pytest.approx(joint.stiff) == 3.70e7
    assert joint.skew_id == 210
    assert pytest.approx(joint.tol) == 2.55e-4
    assert pytest.approx(joint.link_len_a) == 295.0
    assert pytest.approx(joint.link_len_b) == 265.0
    assert pytest.approx(joint.twist_angle_alpha) == 305.0
    assert pytest.approx(joint.offset_distance_s) == 220.0


def test_m406_lagmul_newton_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Newton Spatial Linkage Joint Aliases Test
/NEWTON_SPATIAL_LINKAGE_JOINT/457
691, 692, 693, 1.0e6, 0, 1.0e-6
235.0, 225.0, 235.0, 185.0
/LAGMUL/NEWTON_SPATIAL_LINKAGE/458
691, 692, 693, 1.0e6, 0, 1.0e-6
235.0, 225.0, 235.0, 185.0
/NEWTON_SPATIAL_LINKAGE/459
691, 692, 693, 1.0e6, 0, 1.0e-6
235.0, 225.0, 235.0, 185.0
/NEWTON_SPATIAL_MULTI_LOOP_MECHANISM/460
691, 692, 693, 1.0e6, 0, 1.0e-6
235.0, 225.0, 235.0, 185.0
/NEWTON_SPATIAL_SYMMETRIC_MECHANISM/461
691, 692, 693, 1.0e6, 0, 1.0e-6
235.0, 225.0, 235.0, 185.0
/NEWTON_SPATIAL_6R_MECHANISM/462
691, 692, 693, 1.0e6, 0, 1.0e-6
235.0, 225.0, 235.0, 185.0
/NEWTON_SPATIAL_OVERCONSTRAINED_MECHANISM/463
691, 692, 693, 1.0e6, 0, 1.0e-6
235.0, 225.0, 235.0, 185.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(457, 464):
        assert jid in model.lagmul_newton_spatial_linkage_joints


def test_m406_lagmul_newton_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/NEWTON_SPATIAL_LINKAGE_JOINT/464
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m406_sensor_spring_torsional_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{1538:>10d}{4.68e8:>20.4f}{0.1515:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Crackle Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/1
Fixed Spring Torsional Crackle Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_crackle_rates
    s1 = model.sensor_spring_torsional_crackle_rates[1]
    assert s1.spring_id == 1538
    assert pytest.approx(s1.jtors_crk_max) == 4.68e8
    assert pytest.approx(s1.jtors_shot_max) == 4.68e8
    assert pytest.approx(s1.jtors_drop_max) == 4.68e8
    assert pytest.approx(s1.jtors_lock_max) == 4.68e8
    assert pytest.approx(s1.jtors_pop_max) == 4.68e8
    assert pytest.approx(s1.jtors_snp_max) == 4.68e8
    assert pytest.approx(s1.t_delay) == 0.1515
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_CRACKLE_RATE"


def test_m406_sensor_spring_torsional_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Torsional Crackle Rate Free Format Test
/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/2
Free Spring Torsional Crackle Rate Sensor
1539, 5.28e8, 0.2505
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_torsional_crackle_rates
    s2 = model.sensor_spring_torsional_crackle_rates[2]
    assert s2.spring_id == 1539
    assert pytest.approx(s2.jtors_crk_max) == 5.28e8
    assert pytest.approx(s2.t_delay) == 0.2505


def test_m406_sensor_spring_torsional_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Crackle Rate Aliases Test
/SENSOR/SPRING_TORS_CRACKLE_RATE/401
1559, 2.895e8, 0.1765
/SENSOR/SPRING_RATE_CRACKLE_TORS/402
1560, 2.915e8, 0.1775
/SENSOR/TORSIONAL_CRACKLE_RATE_SPRING/403
1561, 2.935e8, 0.1785
/SENSOR/SPRING_CRACKLE_TORS/404
1562, 2.955e8, 0.1795
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(401, 405):
        assert sid in model.sensor_spring_torsional_crackle_rates


def test_m406_sensor_spring_torsional_crackle_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/405
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

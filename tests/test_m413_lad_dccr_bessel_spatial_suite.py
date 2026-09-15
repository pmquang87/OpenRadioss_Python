"""Tests for Milestone M413: LadDynamicCoreCrackingRate Failure Model, EngElectrothermoflexoplasmonicmagnonicphononicpolaritonicResonanceEnergy, BesselSpatialLinkageJoint, and SensorSpringBendingPopRate."""

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


def test_m413_fail_lad_dynamic_core_cracking_rate_fixed(tmp_path: Path):
    c1 = f"{665.0:>20.4f}{1995.0:>20.4f}{445.0:>20.4f}{9.25:>20.4f}{0.765:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2270:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Core Cracking Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_CORE_CRACKING_RATE/2270
Ladeveze Dynamic Core Cracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2270 in model.fail_laddynamiccorecrackingrates
    fdcc = model.fail_laddynamiccorecrackingrates[2270]
    assert pytest.approx(fdcc.sigma_dcc0) == 665.0
    assert pytest.approx(fdcc.sigma_dccc) == 1995.0
    assert pytest.approx(fdcc.gamma_dcc) == 445.0
    assert pytest.approx(fdcc.p_dcc) == 9.25
    assert pytest.approx(fdcc.d_dcc_max) == 0.765
    assert fdcc.ifail_sh == 1
    assert fdcc.ifail_so == 2
    assert fdcc.fail_id == 2270
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_CORE_CRACKING_RATE"


def test_m413_fail_lad_dynamic_core_cracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Core Cracking Rate Free Format Test
/FAIL/LAD_DYNAMIC_CORE_CRACKING_RATE/2271
675.0, 2025.0, 455.0, 9.45, 0.755
1, 1
2271
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2271 in model.fail_laddynamiccorecrackingrates
    fdcc = model.fail_laddynamiccorecrackingrates[2271]
    assert pytest.approx(fdcc.sigma_dcc0) == 675.0
    assert pytest.approx(fdcc.sigma_dccc) == 2025.0
    assert pytest.approx(fdcc.gamma_dcc) == 455.0
    assert pytest.approx(fdcc.p_dcc) == 9.45
    assert pytest.approx(fdcc.d_dcc_max) == 0.755
    assert fdcc.fail_id == 2271


def test_m413_fail_lad_dynamic_core_cracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Core Cracking Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_CORE_CRACKING_RATE/2272
655.0, 1965.0, 440.0, 9.05, 0.775
1, 1
/FAIL/LAD_DCCKR/2273
655.0, 1965.0, 440.0, 9.05, 0.775
1, 1
/FAIL/LAD_DCCKR_MODEL/2274
655.0, 1965.0, 440.0, 9.05, 0.775
1, 1
/FAIL/LAD_DCCKR_LAW/2275
655.0, 1965.0, 440.0, 9.05, 0.775
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_CORE_CRACKING/2276
655.0, 1965.0, 440.0, 9.05, 0.775
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2272 in model.fail_laddynamiccorecrackingrates
    assert 2273 in model.fail_laddynamiccorecrackingrates
    assert 2274 in model.fail_laddynamiccorecrackingrates
    assert 2275 in model.fail_laddynamiccorecrackingrates
    assert 2276 in model.fail_laddynamiccorecrackingrates
    assert len(model.raw_fails) == 5


def test_m413_fail_lad_dynamic_core_cracking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_CORE_CRACKING_RATE/2277
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m413_eng_electrothermoflexoplasmonicmagnonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0365:>20.4f}{595:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexoplasmonicmagnonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/495
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 495 in model.eng_electrothermoflexoplasmonicmagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexoplasmonicmagnonicphononicpolaritonic_resonance_energies[495]
    assert pytest.approx(eng.dt_etfpmpp) == 0.0365
    assert eng.sens_id == 595


def test_m413_eng_electrothermoflexoplasmonicmagnonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexoplasmonicmagnonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/496
0.0375, 596
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 496 in model.eng_electrothermoflexoplasmonicmagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexoplasmonicmagnonicphononicpolaritonic_resonance_energies[496]
    assert pytest.approx(eng.dt_etfpmpp) == 0.0375
    assert eng.sens_id == 596


def test_m413_eng_electrothermoflexoplasmonicmagnonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexoplasmonicmagnonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_PLASMON_MAGNON_PHONON_POLARITON_RES_WORK/497
0.0385, 597
/ENG/EELECTROTHERMOFLEXOPLASMONICMAGNONICPHONONICPOLARITONICRESONANCE/498
0.0395, 598
/ENG/ELECTROTHERMOFLEXOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/499
0.0405, 599
/ENG/ET_ELECTROTHERMOFLEXOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE/500
0.0415, 600
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 497 in model.eng_electrothermoflexoplasmonicmagnonicphononicpolaritonic_resonance_energies
    assert 498 in model.eng_electrothermoflexoplasmonicmagnonicphononicpolaritonic_resonance_energies
    assert 499 in model.eng_electrothermoflexoplasmonicmagnonicphononicpolaritonic_resonance_energies
    assert 500 in model.eng_electrothermoflexoplasmonicmagnonicphononicpolaritonic_resonance_energies


def test_m413_eng_electrothermoflexoplasmonicmagnonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/501
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m413_lagmul_bessel_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{581:>10d}{582:>10d}{583:>10d}{4.35e7:>20.4f}{275:>10d}{3.45e-4:>20.4e}"
    c2 = f"{335.0:>20.4f}{315.0:>20.4f}{345.0:>20.4f}{265.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Bessel Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/BESSEL_SPATIAL_LINKAGE_JOINT/545
Bessel Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 545 in model.lagmul_bessel_spatial_linkage_joints
    joint = model.lagmul_bessel_spatial_linkage_joints[545]
    assert joint.node1 == 581
    assert joint.node2 == 582
    assert joint.node3 == 583
    assert pytest.approx(joint.stiff) == 4.35e7
    assert joint.skew_id == 275
    assert pytest.approx(joint.tol) == 3.45e-4
    assert pytest.approx(joint.link_len_a) == 335.0
    assert pytest.approx(joint.link_len_b) == 315.0
    assert pytest.approx(joint.twist_angle_alpha) == 345.0
    assert pytest.approx(joint.offset_distance_s) == 265.0
    assert pytest.approx(joint.offset_distance_r) == 265.0
    assert pytest.approx(joint.offset_distance_v) == 265.0
    assert pytest.approx(joint.offset_distance_h) == 265.0
    assert pytest.approx(joint.offset_distance_u) == 265.0


def test_m413_lagmul_bessel_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Bessel Spatial Linkage Joint Free Format Test
/LAGMUL/BESSEL_SPATIAL_LINKAGE_JOINT/546
681, 682, 683, 4.65e7, 276, 3.55e-4
355.0, 325.0, 365.0, 280.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 546 in model.lagmul_bessel_spatial_linkage_joints
    joint = model.lagmul_bessel_spatial_linkage_joints[546]
    assert joint.node1 == 681
    assert joint.node2 == 682
    assert joint.node3 == 683
    assert pytest.approx(joint.stiff) == 4.65e7
    assert joint.skew_id == 276
    assert pytest.approx(joint.tol) == 3.55e-4
    assert pytest.approx(joint.link_len_a) == 355.0
    assert pytest.approx(joint.link_len_b) == 325.0
    assert pytest.approx(joint.twist_angle_alpha) == 365.0
    assert pytest.approx(joint.offset_distance_s) == 280.0


def test_m413_lagmul_bessel_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Bessel Spatial Linkage Joint Aliases Test
/BESSEL_SPATIAL_LINKAGE_JOINT/547
781, 782, 783, 1.0e6, 0, 1.0e-6
295.0, 285.0, 295.0, 245.0
/LAGMUL/BESSEL_SPATIAL_LINKAGE/548
781, 782, 783, 1.0e6, 0, 1.0e-6
295.0, 285.0, 295.0, 245.0
/BESSEL_SPATIAL_LINKAGE/549
781, 782, 783, 1.0e6, 0, 1.0e-6
295.0, 285.0, 295.0, 245.0
/BESSEL_SPATIAL_MULTI_LOOP_MECHANISM/550
781, 782, 783, 1.0e6, 0, 1.0e-6
295.0, 285.0, 295.0, 245.0
/BESSEL_SPATIAL_SYMMETRIC_MECHANISM/551
781, 782, 783, 1.0e6, 0, 1.0e-6
295.0, 285.0, 295.0, 245.0
/BESSEL_SPATIAL_6R_MECHANISM/552
781, 782, 783, 1.0e6, 0, 1.0e-6
295.0, 285.0, 295.0, 245.0
/BESSEL_SPATIAL_OVERCONSTRAINED_MECHANISM/553
781, 782, 783, 1.0e6, 0, 1.0e-6
295.0, 285.0, 295.0, 245.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(547, 554):
        assert jid in model.lagmul_bessel_spatial_linkage_joints


def test_m413_lagmul_bessel_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/BESSEL_SPATIAL_LINKAGE_JOINT/554
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m413_sensor_spring_bending_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1628:>10d}{5.58e8:>20.4f}{0.1945:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_POP_RATE/1
Fixed Spring Bending Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_pop_rates
    s1 = model.sensor_spring_bending_pop_rates[1]
    assert s1.spring_id == 1628
    assert pytest.approx(s1.jbend_pop_max) == 5.58e8
    assert pytest.approx(s1.jbend_snp_max) == 5.58e8
    assert pytest.approx(s1.jbend_crackle_max) == 5.58e8
    assert pytest.approx(s1.jbend_shot_max) == 5.58e8
    assert pytest.approx(s1.jbend_drop_max) == 5.58e8
    assert pytest.approx(s1.jbend_lock_max) == 5.58e8
    assert pytest.approx(s1.jbend_crk_max) == 5.58e8
    assert pytest.approx(s1.t_delay) == 0.1945
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_POP_RATE"


def test_m413_sensor_spring_bending_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Bending Pop Rate Free Format Test
/SENSOR/SPRING_BENDING_POP_RATE/2
Free Spring Bending Pop Rate Sensor
1629, 6.18e8, 0.2935
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_bending_pop_rates
    s2 = model.sensor_spring_bending_pop_rates[2]
    assert s2.spring_id == 1629
    assert pytest.approx(s2.jbend_pop_max) == 6.18e8
    assert pytest.approx(s2.t_delay) == 0.2935


def test_m413_sensor_spring_bending_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Pop Rate Aliases Test
/SENSOR/SPRING_BEND_POP_RATE/491
1649, 3.595e8, 0.2465
/SENSOR/SPRING_RATE_POP_BEND/492
1650, 3.615e8, 0.2475
/SENSOR/BENDING_POP_RATE_SPRING/493
1651, 3.635e8, 0.2485
/SENSOR/SPRING_POP_BEND/494
1652, 3.655e8, 0.2495
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(491, 495):
        assert sid in model.sensor_spring_bending_pop_rates


def test_m413_sensor_spring_bending_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_BENDING_POP_RATE/495
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

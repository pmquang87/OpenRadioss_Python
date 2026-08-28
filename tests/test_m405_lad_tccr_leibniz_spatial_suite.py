"""Tests for Milestone M405: LadTransverseCoreCrushingRate Failure Model, EngFlexothermoplasmonicexcitonicphononicpolaritonicResonanceEnergy, LeibnizSpatialLinkageJoint, and SensorSpringTotalCrackleRate."""

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


def test_m405_fail_lad_transverse_core_crushing_rate_fixed(tmp_path: Path):
    c1 = f"{565.0:>20.4f}{1695.0:>20.4f}{345.0:>20.4f}{7.35:>20.4f}{0.855:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2190:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Core Crushing Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_CORE_CRUSHING_RATE/2190
Ladeveze Transverse Core Crushing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2190 in model.fail_ladtransversecorecrushingrates
    ftccr = model.fail_ladtransversecorecrushingrates[2190]
    assert pytest.approx(ftccr.sigma_tccr0) == 565.0
    assert pytest.approx(ftccr.sigma_tccrc) == 1695.0
    assert pytest.approx(ftccr.gamma_tccr) == 345.0
    assert pytest.approx(ftccr.p_tccr) == 7.35
    assert pytest.approx(ftccr.d_tccr_max) == 0.855
    assert ftccr.ifail_sh == 1
    assert ftccr.ifail_so == 2
    assert ftccr.fail_id == 2190
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_CORE_CRUSHING_RATE"


def test_m405_fail_lad_transverse_core_crushing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Core Crushing Rate Free Format Test
/FAIL/LAD_TRANSVERSE_CORE_CRUSHING_RATE/2191
575.0, 1725.0, 355.0, 7.55, 0.845
1, 1
2191
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2191 in model.fail_ladtransversecorecrushingrates
    ftccr = model.fail_ladtransversecorecrushingrates[2191]
    assert pytest.approx(ftccr.sigma_tccr0) == 575.0
    assert pytest.approx(ftccr.sigma_tccrc) == 1725.0
    assert pytest.approx(ftccr.gamma_tccr) == 355.0
    assert pytest.approx(ftccr.p_tccr) == 7.55
    assert pytest.approx(ftccr.d_tccr_max) == 0.845
    assert ftccr.fail_id == 2191


def test_m405_fail_lad_transverse_core_crushing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Core Crushing Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_CORE_CRUSHING_RATE/2192
555.0, 1665.0, 340.0, 7.15, 0.875
1, 1
/FAIL/LAD_TCCR/2193
555.0, 1665.0, 340.0, 7.15, 0.875
1, 1
/FAIL/LAD_TCCR_MODEL/2194
555.0, 1665.0, 340.0, 7.15, 0.875
1, 1
/FAIL/LAD_TCCR_LAW/2195
555.0, 1665.0, 340.0, 7.15, 0.875
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_CORE_CRUSHING/2196
555.0, 1665.0, 340.0, 7.15, 0.875
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2192 in model.fail_ladtransversecorecrushingrates
    assert 2193 in model.fail_ladtransversecorecrushingrates
    assert 2194 in model.fail_ladtransversecorecrushingrates
    assert 2195 in model.fail_ladtransversecorecrushingrates
    assert 2196 in model.fail_ladtransversecorecrushingrates
    assert len(model.raw_fails) == 5


def test_m405_fail_lad_transverse_core_crushing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_CORE_CRUSHING_RATE/2197
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m405_eng_flexothermoplasmonicexcitonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0275:>20.4f}{495:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexothermoplasmonicexcitonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/395
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 395 in model.eng_flexothermoplasmonicexcitonicphononicpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonicexcitonicphononicpolaritonic_resonance_energies[395]
    assert pytest.approx(eng.dt_ftpepp) == 0.0275
    assert eng.sens_id == 495


def test_m405_eng_flexothermoplasmonicexcitonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexothermoplasmonicexcitonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/396
0.0285, 496
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 396 in model.eng_flexothermoplasmonicexcitonicphononicpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonicexcitonicphononicpolaritonic_resonance_energies[396]
    assert pytest.approx(eng.dt_ftpepp) == 0.0285
    assert eng.sens_id == 496


def test_m405_eng_flexothermoplasmonicexcitonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexothermoplasmonicexcitonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PLASMON_EXCITON_PHONON_POLARITON_RES_WORK/397
0.0295, 497
/ENG/EFLEXOTHERMOPLASMONICEXCITONICPHONONICPOLARITONICRESONANCE/398
0.0305, 498
/ENG/FLEXOTHERMOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/399
0.0315, 499
/ENG/ET_FLEXOTHERMOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE/400
0.0325, 500
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 397 in model.eng_flexothermoplasmonicexcitonicphononicpolaritonic_resonance_energies
    assert 398 in model.eng_flexothermoplasmonicexcitonicphononicpolaritonic_resonance_energies
    assert 399 in model.eng_flexothermoplasmonicexcitonicphononicpolaritonic_resonance_energies
    assert 400 in model.eng_flexothermoplasmonicexcitonicphononicpolaritonic_resonance_energies


def test_m405_eng_flexothermoplasmonicexcitonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOTHERMOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/401
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m405_lagmul_leibniz_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{481:>10d}{482:>10d}{483:>10d}{3.36e7:>20.4f}{199:>10d}{2.35e-4:>20.4e}"
    c2 = f"{265.0:>20.4f}{248.0:>20.4f}{275.0:>20.4f}{198.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Leibniz Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/LEIBNIZ_SPATIAL_LINKAGE_JOINT/445
Leibniz Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 445 in model.lagmul_leibniz_spatial_linkage_joints
    joint = model.lagmul_leibniz_spatial_linkage_joints[445]
    assert joint.node1 == 481
    assert joint.node2 == 482
    assert joint.node3 == 483
    assert pytest.approx(joint.stiff) == 3.36e7
    assert joint.skew_id == 199
    assert pytest.approx(joint.tol) == 2.35e-4
    assert pytest.approx(joint.link_len_a) == 265.0
    assert pytest.approx(joint.link_len_b) == 248.0
    assert pytest.approx(joint.twist_angle_alpha) == 275.0
    assert pytest.approx(joint.offset_distance_s) == 198.0
    assert pytest.approx(joint.offset_distance_r) == 198.0
    assert pytest.approx(joint.offset_distance_v) == 198.0
    assert pytest.approx(joint.offset_distance_h) == 198.0
    assert pytest.approx(joint.offset_distance_u) == 198.0


def test_m405_lagmul_leibniz_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Leibniz Spatial Linkage Joint Free Format Test
/LAGMUL/LEIBNIZ_SPATIAL_LINKAGE_JOINT/446
581, 582, 583, 3.60e7, 200, 2.45e-4
285.0, 255.0, 295.0, 210.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 446 in model.lagmul_leibniz_spatial_linkage_joints
    joint = model.lagmul_leibniz_spatial_linkage_joints[446]
    assert joint.node1 == 581
    assert joint.node2 == 582
    assert joint.node3 == 583
    assert pytest.approx(joint.stiff) == 3.60e7
    assert joint.skew_id == 200
    assert pytest.approx(joint.tol) == 2.45e-4
    assert pytest.approx(joint.link_len_a) == 285.0
    assert pytest.approx(joint.link_len_b) == 255.0
    assert pytest.approx(joint.twist_angle_alpha) == 295.0
    assert pytest.approx(joint.offset_distance_s) == 210.0


def test_m405_lagmul_leibniz_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Leibniz Spatial Linkage Joint Aliases Test
/LEIBNIZ_SPATIAL_LINKAGE_JOINT/447
681, 682, 683, 1.0e6, 0, 1.0e-6
225.0, 215.0, 225.0, 175.0
/LAGMUL/LEIBNIZ_SPATIAL_LINKAGE/448
681, 682, 683, 1.0e6, 0, 1.0e-6
225.0, 215.0, 225.0, 175.0
/LEIBNIZ_SPATIAL_LINKAGE/449
681, 682, 683, 1.0e6, 0, 1.0e-6
225.0, 215.0, 225.0, 175.0
/LEIBNIZ_SPATIAL_MULTI_LOOP_MECHANISM/450
681, 682, 683, 1.0e6, 0, 1.0e-6
225.0, 215.0, 225.0, 175.0
/LEIBNIZ_SPATIAL_SYMMETRIC_MECHANISM/451
681, 682, 683, 1.0e6, 0, 1.0e-6
225.0, 215.0, 225.0, 175.0
/LEIBNIZ_SPATIAL_6R_MECHANISM/452
681, 682, 683, 1.0e6, 0, 1.0e-6
225.0, 215.0, 225.0, 175.0
/LEIBNIZ_SPATIAL_OVERCONSTRAINED_MECHANISM/453
681, 682, 683, 1.0e6, 0, 1.0e-6
225.0, 215.0, 225.0, 175.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(447, 454):
        assert jid in model.lagmul_leibniz_spatial_linkage_joints


def test_m405_lagmul_leibniz_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/LEIBNIZ_SPATIAL_LINKAGE_JOINT/454
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m405_sensor_spring_total_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{1528:>10d}{4.58e8:>20.4f}{0.1505:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Crackle Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_CRACKLE_RATE/1
Fixed Spring Total Crackle Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_crackle_rates
    s1 = model.sensor_spring_total_crackle_rates[1]
    assert s1.spring_id == 1528
    assert pytest.approx(s1.jtot_crk_max) == 4.58e8
    assert pytest.approx(s1.jtot_shot_max) == 4.58e8
    assert pytest.approx(s1.jtot_drop_max) == 4.58e8
    assert pytest.approx(s1.jtot_lock_max) == 4.58e8
    assert pytest.approx(s1.jtot_pop_max) == 4.58e8
    assert pytest.approx(s1.jtot_snp_max) == 4.58e8
    assert pytest.approx(s1.t_delay) == 0.1505
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_CRACKLE_RATE"


def test_m405_sensor_spring_total_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Crackle Rate Free Format Test
/SENSOR/SPRING_TOTAL_CRACKLE_RATE/2
Free Spring Total Crackle Rate Sensor
1529, 5.18e8, 0.2495
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_crackle_rates
    s2 = model.sensor_spring_total_crackle_rates[2]
    assert s2.spring_id == 1529
    assert pytest.approx(s2.jtot_crk_max) == 5.18e8
    assert pytest.approx(s2.t_delay) == 0.2495


def test_m405_sensor_spring_total_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Crackle Rate Aliases Test
/SENSOR/SPRING_TOT_CRACKLE_RATE/391
1549, 2.795e8, 0.1665
/SENSOR/SPRING_RATE_CRACKLE_TOT/392
1550, 2.815e8, 0.1675
/SENSOR/TOTAL_CRACKLE_RATE_SPRING/393
1551, 2.835e8, 0.1685
/SENSOR/SPRING_CRACKLE_TOT/394
1552, 2.855e8, 0.1695
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(391, 395):
        assert sid in model.sensor_spring_total_crackle_rates


def test_m405_sensor_spring_total_crackle_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_CRACKLE_RATE/395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

"""Tests for Milestone M408: LadTransverseCoreShearingRate Failure Model, EngFlexothermoplasmonicexcitonicmagnonicphononicpolaritonicResonanceEnergy, EulerSpatialLinkageJoint, and SensorSpringTotalAngularCrackleRate."""

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


def test_m408_fail_lad_transverse_core_shearing_rate_fixed(tmp_path: Path):
    c1 = f"{605.0:>20.4f}{1815.0:>20.4f}{385.0:>20.4f}{8.05:>20.4f}{0.815:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2220:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Core Shearing Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_CORE_SHEARING_RATE/2220
Ladeveze Transverse Core Shearing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2220 in model.fail_ladtransversecoreshearingrates
    ftcsr = model.fail_ladtransversecoreshearingrates[2220]
    assert pytest.approx(ftcsr.sigma_tcsr0) == 605.0
    assert pytest.approx(ftcsr.sigma_tcsrc) == 1815.0
    assert pytest.approx(ftcsr.gamma_tcsr) == 385.0
    assert pytest.approx(ftcsr.p_tcsr) == 8.05
    assert pytest.approx(ftcsr.d_tcsr_max) == 0.815
    assert ftcsr.ifail_sh == 1
    assert ftcsr.ifail_so == 2
    assert ftcsr.fail_id == 2220
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_CORE_SHEARING_RATE"


def test_m408_fail_lad_transverse_core_shearing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Core Shearing Rate Free Format Test
/FAIL/LAD_TRANSVERSE_CORE_SHEARING_RATE/2221
615.0, 1845.0, 395.0, 8.25, 0.805
1, 1
2221
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2221 in model.fail_ladtransversecoreshearingrates
    ftcsr = model.fail_ladtransversecoreshearingrates[2221]
    assert pytest.approx(ftcsr.sigma_tcsr0) == 615.0
    assert pytest.approx(ftcsr.sigma_tcsrc) == 1845.0
    assert pytest.approx(ftcsr.gamma_tcsr) == 395.0
    assert pytest.approx(ftcsr.p_tcsr) == 8.25
    assert pytest.approx(ftcsr.d_tcsr_max) == 0.805
    assert ftcsr.fail_id == 2221


def test_m408_fail_lad_transverse_core_shearing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Core Shearing Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_CORE_SHEARING_RATE/2222
595.0, 1785.0, 380.0, 7.85, 0.835
1, 1
/FAIL/LAD_TCSR/2223
595.0, 1785.0, 380.0, 7.85, 0.835
1, 1
/FAIL/LAD_TCSR_MODEL/2224
595.0, 1785.0, 380.0, 7.85, 0.835
1, 1
/FAIL/LAD_TCSR_LAW/2225
595.0, 1785.0, 380.0, 7.85, 0.835
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_CORE_SHEARING/2226
595.0, 1785.0, 380.0, 7.85, 0.835
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2222 in model.fail_ladtransversecoreshearingrates
    assert 2223 in model.fail_ladtransversecoreshearingrates
    assert 2224 in model.fail_ladtransversecoreshearingrates
    assert 2225 in model.fail_ladtransversecoreshearingrates
    assert 2226 in model.fail_ladtransversecoreshearingrates
    assert len(model.raw_fails) == 5


def test_m408_fail_lad_transverse_core_shearing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_CORE_SHEARING_RATE/2227
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m408_eng_flexothermoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0305:>20.4f}{525:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexothermoplasmonicexcitonicmagnonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/425
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 425 in model.eng_flexothermoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies[425]
    assert pytest.approx(eng.dt_ftpempp) == 0.0305
    assert eng.sens_id == 525


def test_m408_eng_flexothermoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexothermoplasmonicexcitonicmagnonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/426
0.0315, 526
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 426 in model.eng_flexothermoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies[426]
    assert pytest.approx(eng.dt_ftpempp) == 0.0315
    assert eng.sens_id == 526


def test_m408_eng_flexothermoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexothermoplasmonicexcitonicmagnonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PLASMONIC_EXCITONIC_MAGNONIC_PHONONIC_POLARITON_RES_WORK/427
0.0325, 527
/ENG/EFLEXOTHERMOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONICRESONANCE/428
0.0335, 528
/ENG/FLEXOTHERMOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/429
0.0345, 529
/ENG/ET_FLEXOTHERMOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE/430
0.0355, 530
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 427 in model.eng_flexothermoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies
    assert 428 in model.eng_flexothermoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies
    assert 429 in model.eng_flexothermoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies
    assert 430 in model.eng_flexothermoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies


def test_m408_eng_flexothermoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOTHERMOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/431
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m408_lagmul_euler_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{511:>10d}{512:>10d}{513:>10d}{3.66e7:>20.4f}{229:>10d}{2.65e-4:>20.4e}"
    c2 = f"{295.0:>20.4f}{278.0:>20.4f}{305.0:>20.4f}{228.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Euler Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/EULER_SPATIAL_LINKAGE_JOINT/475
Euler Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 475 in model.lagmul_euler_spatial_linkage_joints
    joint = model.lagmul_euler_spatial_linkage_joints[475]
    assert joint.node1 == 511
    assert joint.node2 == 512
    assert joint.node3 == 513
    assert pytest.approx(joint.stiff) == 3.66e7
    assert joint.skew_id == 229
    assert pytest.approx(joint.tol) == 2.65e-4
    assert pytest.approx(joint.link_len_a) == 295.0
    assert pytest.approx(joint.link_len_b) == 278.0
    assert pytest.approx(joint.twist_angle_alpha) == 305.0
    assert pytest.approx(joint.offset_distance_s) == 228.0
    assert pytest.approx(joint.offset_distance_r) == 228.0
    assert pytest.approx(joint.offset_distance_v) == 228.0
    assert pytest.approx(joint.offset_distance_h) == 228.0
    assert pytest.approx(joint.offset_distance_u) == 228.0


def test_m408_lagmul_euler_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Euler Spatial Linkage Joint Free Format Test
/LAGMUL/EULER_SPATIAL_LINKAGE_JOINT/476
611, 612, 613, 3.90e7, 230, 2.75e-4
315.0, 285.0, 325.0, 240.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 476 in model.lagmul_euler_spatial_linkage_joints
    joint = model.lagmul_euler_spatial_linkage_joints[476]
    assert joint.node1 == 611
    assert joint.node2 == 612
    assert joint.node3 == 613
    assert pytest.approx(joint.stiff) == 3.90e7
    assert joint.skew_id == 230
    assert pytest.approx(joint.tol) == 2.75e-4
    assert pytest.approx(joint.link_len_a) == 315.0
    assert pytest.approx(joint.link_len_b) == 285.0
    assert pytest.approx(joint.twist_angle_alpha) == 325.0
    assert pytest.approx(joint.offset_distance_s) == 240.0


def test_m408_lagmul_euler_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Euler Spatial Linkage Joint Aliases Test
/EULER_SPATIAL_LINKAGE_JOINT/477
711, 712, 713, 1.0e6, 0, 1.0e-6
255.0, 245.0, 255.0, 205.0
/LAGMUL/EULER_SPATIAL_LINKAGE/478
711, 712, 713, 1.0e6, 0, 1.0e-6
255.0, 245.0, 255.0, 205.0
/EULER_SPATIAL_LINKAGE/479
711, 712, 713, 1.0e6, 0, 1.0e-6
255.0, 245.0, 255.0, 205.0
/EULER_SPATIAL_MULTI_LOOP_MECHANISM/480
711, 712, 713, 1.0e6, 0, 1.0e-6
255.0, 245.0, 255.0, 205.0
/EULER_SPATIAL_SYMMETRIC_MECHANISM/481
711, 712, 713, 1.0e6, 0, 1.0e-6
255.0, 245.0, 255.0, 205.0
/EULER_SPATIAL_6R_MECHANISM/482
711, 712, 713, 1.0e6, 0, 1.0e-6
255.0, 245.0, 255.0, 205.0
/EULER_SPATIAL_OVERCONSTRAINED_MECHANISM/483
711, 712, 713, 1.0e6, 0, 1.0e-6
255.0, 245.0, 255.0, 205.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(477, 484):
        assert jid in model.lagmul_euler_spatial_linkage_joints


def test_m408_lagmul_euler_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/EULER_SPATIAL_LINKAGE_JOINT/484
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m408_sensor_spring_total_angular_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{1558:>10d}{4.88e8:>20.4f}{0.1535:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Angular Crackle Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE/1
Fixed Spring Total Angular Crackle Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_angular_crackle_rates
    s1 = model.sensor_spring_total_angular_crackle_rates[1]
    assert s1.spring_id == 1558
    assert pytest.approx(s1.jtota_crk_max) == 4.88e8
    assert pytest.approx(s1.jtota_shot_max) == 4.88e8
    assert pytest.approx(s1.jtota_drop_max) == 4.88e8
    assert pytest.approx(s1.jtota_lock_max) == 4.88e8
    assert pytest.approx(s1.jtota_pop_max) == 4.88e8
    assert pytest.approx(s1.jtota_snp_max) == 4.88e8
    assert pytest.approx(s1.t_delay) == 0.1535
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_CRACKLE_RATE"


def test_m408_sensor_spring_total_angular_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Angular Crackle Rate Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE/2
Free Spring Total Angular Crackle Rate Sensor
1559, 5.48e8, 0.2525
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_angular_crackle_rates
    s2 = model.sensor_spring_total_angular_crackle_rates[2]
    assert s2.spring_id == 1559
    assert pytest.approx(s2.jtota_crk_max) == 5.48e8
    assert pytest.approx(s2.t_delay) == 0.2525


def test_m408_sensor_spring_total_angular_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Angular Crackle Rate Aliases Test
/SENSOR/SPRING_TOT_ANG_CRACKLE_RATE/421
1579, 3.095e8, 0.1965
/SENSOR/SPRING_RATE_CRACKLE_ANG_TOT/422
1580, 3.115e8, 0.1975
/SENSOR/TOTAL_ANGULAR_CRACKLE_RATE_SPRING/423
1581, 3.135e8, 0.1985
/SENSOR/SPRING_CRACKLE_ANG_TOT/424
1582, 3.155e8, 0.1995
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(421, 425):
        assert sid in model.sensor_spring_total_angular_crackle_rates


def test_m408_sensor_spring_total_angular_crackle_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE/425
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

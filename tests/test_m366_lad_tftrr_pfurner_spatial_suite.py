"""Tests for Milestone M366: LadTransverseFiberTensionRuptureRate Failure Model, EngFlexomagnetoplasmonicexcitonResonanceEnergy, PfurnerSpatialLinkageJoint, and SensorSpringTotalAngularCrackleRate."""

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


def test_m366_fail_lad_transverse_fiber_tension_rupture_rate_fixed(tmp_path: Path):
    c1 = f"{158.0:>20.4f}{474.0:>20.4f}{68.0:>20.4f}{2.80:>20.4f}{0.968:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1820:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Fiber Tension Rupture Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_FIBER_TENSION_RUPTURE_RATE/1820
Ladeveze Transverse Fiber Tension Rupture Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1820 in model.fail_ladtransversefibertensionrupturerates
    ftftrr = model.fail_ladtransversefibertensionrupturerates[1820]
    assert pytest.approx(ftftrr.sigma_tftrr0) == 158.0
    assert pytest.approx(ftftrr.sigma_tftrrc) == 474.0
    assert pytest.approx(ftftrr.gamma_tftrr) == 68.0
    assert pytest.approx(ftftrr.p_tftrr) == 2.80
    assert pytest.approx(ftftrr.d_tftrr_max) == 0.968
    assert ftftrr.ifail_sh == 1
    assert ftftrr.ifail_so == 2
    assert ftftrr.fail_id == 1820
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_FIBER_TENSION_RUPTURE_RATE"


def test_m366_fail_lad_transverse_fiber_tension_rupture_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Fiber Tension Rupture Rate Free Format Test
/FAIL/LAD_TRANSVERSE_FIBER_TENSION_RUPTURE_RATE/1821
168.0, 504.0, 74.0, 3.00, 0.952
1, 1
1821
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1821 in model.fail_ladtransversefibertensionrupturerates
    ftftrr = model.fail_ladtransversefibertensionrupturerates[1821]
    assert pytest.approx(ftftrr.sigma_tftrr0) == 168.0
    assert pytest.approx(ftftrr.sigma_tftrrc) == 504.0
    assert pytest.approx(ftftrr.gamma_tftrr) == 74.0
    assert pytest.approx(ftftrr.p_tftrr) == 3.00
    assert pytest.approx(ftftrr.d_tftrr_max) == 0.952
    assert ftftrr.fail_id == 1821


def test_m366_fail_lad_transverse_fiber_tension_rupture_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Fiber Tension Rupture Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_FIBER_TENSION_RUPTURE_RATE/1822
148.0, 444.0, 62.0, 2.30, 0.992
1, 1
/FAIL/LAD_TFTRR/1823
148.0, 444.0, 62.0, 2.30, 0.992
1, 1
/FAIL/LAD_TFTRR_MODEL/1824
148.0, 444.0, 62.0, 2.30, 0.992
1, 1
/FAIL/LAD_TFTRR_LAW/1825
148.0, 444.0, 62.0, 2.30, 0.992
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_FIBER_TENSION_RUPTURE/1826
148.0, 444.0, 62.0, 2.30, 0.992
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1822 in model.fail_ladtransversefibertensionrupturerates
    assert 1823 in model.fail_ladtransversefibertensionrupturerates
    assert 1824 in model.fail_ladtransversefibertensionrupturerates
    assert 1825 in model.fail_ladtransversefibertensionrupturerates
    assert 1826 in model.fail_ladtransversefibertensionrupturerates
    assert len(model.raw_fails) == 5


def test_m366_fail_lad_transverse_fiber_tension_rupture_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_TRANSVERSE_FIBER_TENSION_RUPTURE_RATE/1827
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m366_eng_flexomagnetoplasmonicexciton_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00190:>20.6f}{280:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexomagnetoplasmonicexciton Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPLASMONICEXCITON_RESONANCE_ENERGY/1
Flexomagnetoplasmonicexciton Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexomagnetoplasmonicexciton_resonance_energies
    eng = model.eng_flexomagnetoplasmonicexciton_resonance_energies[1]
    assert pytest.approx(eng.dt_fmper) == 0.00190
    assert eng.sens_id == 280


def test_m366_eng_flexomagnetoplasmonicexciton_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexomagnetoplasmonicexciton Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPLASMONICEXCITON_RESONANCE_ENERGY/2
0.00210, 290
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexomagnetoplasmonicexciton_resonance_energies
    eng = model.eng_flexomagnetoplasmonicexciton_resonance_energies[2]
    assert pytest.approx(eng.dt_fmper) == 0.00210
    assert eng.sens_id == 290


def test_m366_eng_flexomagnetoplasmonicexciton_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexomagnetoplasmonicexciton Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PLASMON_EXCITON_RES_WORK/3
0.00150, 215
/ENG/EFLEXOMAGNETOPLASMONICEXCITONRESONANCE/4
0.00154, 220
/ENG/FLEXOMAGNETOPLASMONICEXCITON_RESONANCE_DISSIPATION/5
0.00160, 230
/ENG/EM_FLEXOMAGNETOPLASMONICEXCITON_RESONANCE/6
0.00168, 240
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexomagnetoplasmonicexciton_resonance_energies
    assert 4 in model.eng_flexomagnetoplasmonicexciton_resonance_energies
    assert 5 in model.eng_flexomagnetoplasmonicexciton_resonance_energies
    assert 6 in model.eng_flexomagnetoplasmonicexciton_resonance_energies


def test_m366_eng_flexomagnetoplasmonicexciton_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOMAGNETOPLASMONICEXCITON_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m366_lagmul_pfurner_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{521:>10d}{522:>10d}{523:>10d}{4.75e7:>20.1f}{140:>10d}{2.0e-5:>20.6e}"
    c2 = f"{245.0:>20.4f}{235.0:>20.4f}{215.0:>20.4f}{165.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Pfurner Spatial Linkage Joint Fixed Format Test
2022 0
/PFURNER_SPATIAL_LINKAGE_JOINT/485
Pfurner Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 485 in model.lagmul_pfurner_spatial_linkage_joints
    joint = model.lagmul_pfurner_spatial_linkage_joints[485]
    assert joint.node1 == 521
    assert joint.node2 == 522
    assert joint.node3 == 523
    assert pytest.approx(joint.stiff) == 4.75e7
    assert joint.skew_id == 140
    assert pytest.approx(joint.tol) == 2.0e-5
    assert pytest.approx(joint.link_len_a) == 245.0
    assert pytest.approx(joint.link_len_b) == 235.0
    assert pytest.approx(joint.twist_angle_alpha) == 215.0
    assert pytest.approx(joint.offset_distance_s) == 165.0


def test_m366_lagmul_pfurner_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Pfurner Spatial Linkage Joint Free Format Test
/LAGMUL/PFURNER_SPATIAL_LINKAGE_JOINT/486
621, 622, 623, 33.0e6, 160, 3.0e-5
250.0, 240.0, 220.0, 170.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 486 in model.lagmul_pfurner_spatial_linkage_joints
    joint = model.lagmul_pfurner_spatial_linkage_joints[486]
    assert joint.node1 == 621
    assert joint.node2 == 622
    assert joint.node3 == 623
    assert pytest.approx(joint.stiff) == 33.0e6
    assert joint.skew_id == 160
    assert pytest.approx(joint.tol) == 3.0e-5
    assert pytest.approx(joint.link_len_a) == 250.0
    assert pytest.approx(joint.link_len_b) == 240.0
    assert pytest.approx(joint.twist_angle_alpha) == 220.0
    assert pytest.approx(joint.offset_distance_s) == 170.0


def test_m366_lagmul_pfurner_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Pfurner Spatial Linkage Joint Aliases Test
/LAGMUL/PFURNER_SPATIAL_LINKAGE/487
1, 2, 3, 1e6, 0, 1e-6
68.0, 68.0, 114.0, 54.0
/PFURNER_SPATIAL_LINKAGE/488
1, 2, 3, 1e6, 0, 1e-6
68.0, 68.0, 114.0, 54.0
/PFURNER_SPATIAL_MULTI_LOOP_MECHANISM/489
1, 2, 3, 1e6, 0, 1e-6
68.0, 68.0, 114.0, 54.0
/PFURNER_SPATIAL_PARALLEL_MECHANISM/490
1, 2, 3, 1e6, 0, 1e-6
68.0, 68.0, 114.0, 54.0
/PFURNER_SPATIAL_6R_MECHANISM/491
1, 2, 3, 1e6, 0, 1e-6
68.0, 68.0, 114.0, 54.0
/PFURNER_SPATIAL_OVERCONSTRAINED_MECHANISM/492
1, 2, 3, 1e6, 0, 1e-6
68.0, 68.0, 114.0, 54.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 487 in model.lagmul_pfurner_spatial_linkage_joints
    assert 488 in model.lagmul_pfurner_spatial_linkage_joints
    assert 489 in model.lagmul_pfurner_spatial_linkage_joints
    assert 490 in model.lagmul_pfurner_spatial_linkage_joints
    assert 491 in model.lagmul_pfurner_spatial_linkage_joints
    assert 492 in model.lagmul_pfurner_spatial_linkage_joints


def test_m366_lagmul_pfurner_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/PFURNER_SPATIAL_LINKAGE_JOINT/493
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m366_sensor_spring_total_angular_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{1265:>10d}{8.45e9:>20.1f}{0.0385:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Total Angular Crackle Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE/518
Spring Total Angular Crackle Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 518 in model.sensor_spring_total_angular_crackle_rates
    sensor = model.sensor_spring_total_angular_crackle_rates[518]
    assert sensor.spring_id == 1265
    assert pytest.approx(sensor.jtot_ang_crk_max) == 8.45e9
    assert pytest.approx(sensor.t_delay) == 0.0385
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_CRACKLE_RATE"


def test_m366_sensor_spring_total_angular_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Total Angular Crackle Rate Sensor Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE/519
1266, 8.55e9, 0.0410
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 519 in model.sensor_spring_total_angular_crackle_rates
    sensor = model.sensor_spring_total_angular_crackle_rates[519]
    assert sensor.spring_id == 1266
    assert pytest.approx(sensor.jtot_ang_crk_max) == 8.55e9
    assert pytest.approx(sensor.t_delay) == 0.0410


def test_m366_sensor_spring_total_angular_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Total Angular Crackle Rate Sensor Aliases Test
/SENSOR/SPRING_TOT_ANG_CRACKLE_RATE/520
1267, 8.6e9, 0.0315
/SENSOR/SPRING_RATE_CRACKLE_ANG_TOT/521
1268, 8.6e9, 0.0315
/SENSOR/TOTAL_ANGULAR_CRACKLE_RATE_SPRING/522
1269, 8.6e9, 0.0315
/SENSOR/SPRING_CRACKLE_ANG_TOT/523
1270, 8.6e9, 0.0315
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 520 in model.sensor_spring_total_angular_crackle_rates
    assert 521 in model.sensor_spring_total_angular_crackle_rates
    assert 522 in model.sensor_spring_total_angular_crackle_rates
    assert 523 in model.sensor_spring_total_angular_crackle_rates
    assert len(model.sensors) == 4


def test_m366_sensor_spring_total_angular_crackle_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE/524
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

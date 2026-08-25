"""Tests for Milestone M365: LadDynamicFiberTensionRuptureRate Failure Model, EngFlexomagnetoplasmonicphononResonanceEnergy, KonnokSpatialLinkageJoint, and SensorSpringBendingCrackleRate."""

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


def test_m365_fail_lad_dynamic_fiber_tension_rupture_rate_fixed(tmp_path: Path):
    c1 = f"{155.0:>20.4f}{465.0:>20.4f}{66.0:>20.4f}{2.75:>20.4f}{0.970:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1810:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Fiber Tension Rupture Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_FIBER_TENSION_RUPTURE_RATE/1810
Ladeveze Dynamic Fiber Tension Rupture Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1810 in model.fail_laddynamicfibertensionrupturerates
    fdftrr = model.fail_laddynamicfibertensionrupturerates[1810]
    assert pytest.approx(fdftrr.sigma_dftrr0) == 155.0
    assert pytest.approx(fdftrr.sigma_dftrrc) == 465.0
    assert pytest.approx(fdftrr.gamma_dftrr) == 66.0
    assert pytest.approx(fdftrr.p_dftrr) == 2.75
    assert pytest.approx(fdftrr.d_dftrr_max) == 0.970
    assert fdftrr.ifail_sh == 1
    assert fdftrr.ifail_so == 2
    assert fdftrr.fail_id == 1810
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_FIBER_TENSION_RUPTURE_RATE"


def test_m365_fail_lad_dynamic_fiber_tension_rupture_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Fiber Tension Rupture Rate Free Format Test
/FAIL/LAD_DYNAMIC_FIBER_TENSION_RUPTURE_RATE/1811
165.0, 495.0, 72.0, 2.95, 0.955
1, 1
1811
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1811 in model.fail_laddynamicfibertensionrupturerates
    fdftrr = model.fail_laddynamicfibertensionrupturerates[1811]
    assert pytest.approx(fdftrr.sigma_dftrr0) == 165.0
    assert pytest.approx(fdftrr.sigma_dftrrc) == 495.0
    assert pytest.approx(fdftrr.gamma_dftrr) == 72.0
    assert pytest.approx(fdftrr.p_dftrr) == 2.95
    assert pytest.approx(fdftrr.d_dftrr_max) == 0.955
    assert fdftrr.fail_id == 1811


def test_m365_fail_lad_dynamic_fiber_tension_rupture_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Fiber Tension Rupture Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_FIBER_TENSION_RUPTURE_RATE/1812
145.0, 435.0, 60.0, 2.25, 0.995
1, 1
/FAIL/LAD_DFTRR/1813
145.0, 435.0, 60.0, 2.25, 0.995
1, 1
/FAIL/LAD_DFTRR_MODEL/1814
145.0, 435.0, 60.0, 2.25, 0.995
1, 1
/FAIL/LAD_DFTRR_LAW/1815
145.0, 435.0, 60.0, 2.25, 0.995
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_FIBER_TENSION_RUPTURE/1816
145.0, 435.0, 60.0, 2.25, 0.995
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1812 in model.fail_laddynamicfibertensionrupturerates
    assert 1813 in model.fail_laddynamicfibertensionrupturerates
    assert 1814 in model.fail_laddynamicfibertensionrupturerates
    assert 1815 in model.fail_laddynamicfibertensionrupturerates
    assert 1816 in model.fail_laddynamicfibertensionrupturerates
    assert len(model.raw_fails) == 5


def test_m365_fail_lad_dynamic_fiber_tension_rupture_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_DYNAMIC_FIBER_TENSION_RUPTURE_RATE/1817
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m365_eng_flexomagnetoplasmonicphonon_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00185:>20.6f}{275:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexomagnetoplasmonicphonon Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPLASMONICPHONON_RESONANCE_ENERGY/1
Flexomagnetoplasmonicphonon Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexomagnetoplasmonicphonon_resonance_energies
    eng = model.eng_flexomagnetoplasmonicphonon_resonance_energies[1]
    assert pytest.approx(eng.dt_fmppr) == 0.00185
    assert eng.sens_id == 275


def test_m365_eng_flexomagnetoplasmonicphonon_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexomagnetoplasmonicphonon Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPLASMONICPHONON_RESONANCE_ENERGY/2
0.00205, 285
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexomagnetoplasmonicphonon_resonance_energies
    eng = model.eng_flexomagnetoplasmonicphonon_resonance_energies[2]
    assert pytest.approx(eng.dt_fmppr) == 0.00205
    assert eng.sens_id == 285


def test_m365_eng_flexomagnetoplasmonicphonon_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexomagnetoplasmonicphonon Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PLASMON_PHONON_RES_WORK/3
0.00145, 210
/ENG/EFLEXOMAGNETOPLASMONICPHONONRESONANCE/4
0.00149, 215
/ENG/FLEXOMAGNETOPLASMONICPHONON_RESONANCE_DISSIPATION/5
0.00155, 225
/ENG/EM_FLEXOMAGNETOPLASMONICPHONON_RESONANCE/6
0.00162, 235
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexomagnetoplasmonicphonon_resonance_energies
    assert 4 in model.eng_flexomagnetoplasmonicphonon_resonance_energies
    assert 5 in model.eng_flexomagnetoplasmonicphonon_resonance_energies
    assert 6 in model.eng_flexomagnetoplasmonicphonon_resonance_energies


def test_m365_eng_flexomagnetoplasmonicphonon_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOMAGNETOPLASMONICPHONON_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m365_lagmul_konnok_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{511:>10d}{512:>10d}{513:>10d}{4.65e7:>20.1f}{138:>10d}{2.2e-5:>20.6e}"
    c2 = f"{240.0:>20.4f}{230.0:>20.4f}{210.0:>20.4f}{160.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Konnok Spatial Linkage Joint Fixed Format Test
2022 0
/KONNOK_SPATIAL_LINKAGE_JOINT/475
Konnok Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 475 in model.lagmul_konnok_spatial_linkage_joints
    joint = model.lagmul_konnok_spatial_linkage_joints[475]
    assert joint.node1 == 511
    assert joint.node2 == 512
    assert joint.node3 == 513
    assert pytest.approx(joint.stiff) == 4.65e7
    assert joint.skew_id == 138
    assert pytest.approx(joint.tol) == 2.2e-5
    assert pytest.approx(joint.link_len_a) == 240.0
    assert pytest.approx(joint.link_len_b) == 230.0
    assert pytest.approx(joint.twist_angle_alpha) == 210.0
    assert pytest.approx(joint.offset_distance_s) == 160.0


def test_m365_lagmul_konnok_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Konnok Spatial Linkage Joint Free Format Test
/LAGMUL/KONNOK_SPATIAL_LINKAGE_JOINT/476
611, 612, 613, 32.0e6, 158, 3.2e-5
245.0, 235.0, 215.0, 165.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 476 in model.lagmul_konnok_spatial_linkage_joints
    joint = model.lagmul_konnok_spatial_linkage_joints[476]
    assert joint.node1 == 611
    assert joint.node2 == 612
    assert joint.node3 == 613
    assert pytest.approx(joint.stiff) == 32.0e6
    assert joint.skew_id == 158
    assert pytest.approx(joint.tol) == 3.2e-5
    assert pytest.approx(joint.link_len_a) == 245.0
    assert pytest.approx(joint.link_len_b) == 235.0
    assert pytest.approx(joint.twist_angle_alpha) == 215.0
    assert pytest.approx(joint.offset_distance_s) == 165.0


def test_m365_lagmul_konnok_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Konnok Spatial Linkage Joint Aliases Test
/LAGMUL/KONNOK_SPATIAL_LINKAGE/477
1, 2, 3, 1e6, 0, 1e-6
66.0, 66.0, 112.0, 52.0
/KONNOK_SPATIAL_LINKAGE/478
1, 2, 3, 1e6, 0, 1e-6
66.0, 66.0, 112.0, 52.0
/KONNOK_SPATIAL_MULTI_LOOP_MECHANISM/479
1, 2, 3, 1e6, 0, 1e-6
66.0, 66.0, 112.0, 52.0
/KONNOK_SPATIAL_SYMMETRIC_MECHANISM/480
1, 2, 3, 1e6, 0, 1e-6
66.0, 66.0, 112.0, 52.0
/KONNOK_SPATIAL_6R_MECHANISM/481
1, 2, 3, 1e6, 0, 1e-6
66.0, 66.0, 112.0, 52.0
/KONNOK_SPATIAL_OVERCONSTRAINED_MECHANISM/482
1, 2, 3, 1e6, 0, 1e-6
66.0, 66.0, 112.0, 52.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 477 in model.lagmul_konnok_spatial_linkage_joints
    assert 478 in model.lagmul_konnok_spatial_linkage_joints
    assert 479 in model.lagmul_konnok_spatial_linkage_joints
    assert 480 in model.lagmul_konnok_spatial_linkage_joints
    assert 481 in model.lagmul_konnok_spatial_linkage_joints
    assert 482 in model.lagmul_konnok_spatial_linkage_joints


def test_m365_lagmul_konnok_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/KONNOK_SPATIAL_LINKAGE_JOINT/483
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m365_sensor_spring_bending_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{1255:>10d}{8.25e9:>20.1f}{0.0375:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Bending Crackle Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_CRACKLE_RATE/508
Spring Bending Crackle Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 508 in model.sensor_spring_bending_crackle_rates
    sensor = model.sensor_spring_bending_crackle_rates[508]
    assert sensor.spring_id == 1255
    assert pytest.approx(sensor.jbend_crk_max) == 8.25e9
    assert pytest.approx(sensor.t_delay) == 0.0375
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_CRACKLE_RATE"


def test_m365_sensor_spring_bending_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Bending Crackle Rate Sensor Free Format Test
/SENSOR/SPRING_BENDING_CRACKLE_RATE/509
1256, 8.35e9, 0.0400
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 509 in model.sensor_spring_bending_crackle_rates
    sensor = model.sensor_spring_bending_crackle_rates[509]
    assert sensor.spring_id == 1256
    assert pytest.approx(sensor.jbend_crk_max) == 8.35e9
    assert pytest.approx(sensor.t_delay) == 0.0400


def test_m365_sensor_spring_bending_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Bending Crackle Rate Sensor Aliases Test
/SENSOR/SPRING_BEND_CRACKLE_RATE/510
1257, 8.4e9, 0.0305
/SENSOR/SPRING_RATE_CRACKLE_BEND/511
1258, 8.4e9, 0.0305
/SENSOR/BENDING_CRACKLE_RATE_SPRING/512
1259, 8.4e9, 0.0305
/SENSOR/SPRING_CRACKLE_BEND/513
1260, 8.4e9, 0.0305
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 510 in model.sensor_spring_bending_crackle_rates
    assert 511 in model.sensor_spring_bending_crackle_rates
    assert 512 in model.sensor_spring_bending_crackle_rates
    assert 513 in model.sensor_spring_bending_crackle_rates
    assert len(model.sensors) == 4


def test_m365_sensor_spring_bending_crackle_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_BENDING_CRACKLE_RATE/514
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

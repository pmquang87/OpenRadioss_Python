"""Tests for Milestone M362: LadDynamicMatrixShearDegradationRate Failure Model, EngFlexomagnetophononicResonanceEnergy, HuntSpatialLinkageJoint, and SensorSpringTransverseCrackleRate."""

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


def test_m362_fail_lad_dynamic_matrix_shear_degradation_rate_fixed(tmp_path: Path):
    c1 = f"{148.0:>20.4f}{444.0:>20.4f}{60.0:>20.4f}{2.60:>20.4f}{0.976:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1780:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Matrix Shear Degradation Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_MATRIX_SHEAR_DEGRADATION_RATE/1780
Ladeveze Dynamic Matrix Shear Degradation Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1780 in model.fail_laddynamicmatrixsheardegradationrates
    fdmsdr = model.fail_laddynamicmatrixsheardegradationrates[1780]
    assert pytest.approx(fdmsdr.sigma_dmsdr0) == 148.0
    assert pytest.approx(fdmsdr.sigma_dmsdrc) == 444.0
    assert pytest.approx(fdmsdr.gamma_dmsdr) == 60.0
    assert pytest.approx(fdmsdr.p_dmsdr) == 2.60
    assert pytest.approx(fdmsdr.d_dmsdr_max) == 0.976
    assert fdmsdr.ifail_sh == 1
    assert fdmsdr.ifail_so == 2
    assert fdmsdr.fail_id == 1780
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_MATRIX_SHEAR_DEGRADATION_RATE"


def test_m362_fail_lad_dynamic_matrix_shear_degradation_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Matrix Shear Degradation Rate Free Format Test
/FAIL/LAD_DYNAMIC_MATRIX_SHEAR_DEGRADATION_RATE/1781
158.0, 474.0, 66.0, 2.80, 0.962
1, 1
1781
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1781 in model.fail_laddynamicmatrixsheardegradationrates
    fdmsdr = model.fail_laddynamicmatrixsheardegradationrates[1781]
    assert pytest.approx(fdmsdr.sigma_dmsdr0) == 158.0
    assert pytest.approx(fdmsdr.sigma_dmsdrc) == 474.0
    assert pytest.approx(fdmsdr.gamma_dmsdr) == 66.0
    assert pytest.approx(fdmsdr.p_dmsdr) == 2.80
    assert pytest.approx(fdmsdr.d_dmsdr_max) == 0.962
    assert fdmsdr.fail_id == 1781


def test_m362_fail_lad_dynamic_matrix_shear_degradation_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Matrix Shear Degradation Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_MATRIX_SHEAR_DEGRADATION_RATE/1782
138.0, 414.0, 54.0, 2.18, 0.998
1, 1
/FAIL/LAD_DMSDR/1783
138.0, 414.0, 54.0, 2.18, 0.998
1, 1
/FAIL/LAD_DMSDR_MODEL/1784
138.0, 414.0, 54.0, 2.18, 0.998
1, 1
/FAIL/LAD_DMSDR_LAW/1785
138.0, 414.0, 54.0, 2.18, 0.998
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_MATRIX_SHEAR_DEGRADATION/1786
138.0, 414.0, 54.0, 2.18, 0.998
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1782 in model.fail_laddynamicmatrixsheardegradationrates
    assert 1783 in model.fail_laddynamicmatrixsheardegradationrates
    assert 1784 in model.fail_laddynamicmatrixsheardegradationrates
    assert 1785 in model.fail_laddynamicmatrixsheardegradationrates
    assert 1786 in model.fail_laddynamicmatrixsheardegradationrates
    assert len(model.raw_fails) == 5


def test_m362_fail_lad_dynamic_matrix_shear_degradation_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_DYNAMIC_MATRIX_SHEAR_DEGRADATION_RATE/1787
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m362_eng_flexomagnetophononic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00170:>20.6f}{260:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexomagnetophononic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPHONONIC_RESONANCE_ENERGY/1
Flexomagnetophononic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexomagnetophononic_resonance_energies
    eng = model.eng_flexomagnetophononic_resonance_energies[1]
    assert pytest.approx(eng.dt_fmpr) == 0.00170
    assert eng.sens_id == 260


def test_m362_eng_flexomagnetophononic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexomagnetophononic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPHONONIC_RESONANCE_ENERGY/2
0.00190, 270
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexomagnetophononic_resonance_energies
    eng = model.eng_flexomagnetophononic_resonance_energies[2]
    assert pytest.approx(eng.dt_fmpr) == 0.00190
    assert eng.sens_id == 270


def test_m362_eng_flexomagnetophononic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexomagnetophononic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PHONON_RES_WORK/3
0.00130, 195
/ENG/EFLEXOMAGNETOPHONONICRESONANCE/4
0.00134, 200
/ENG/FLEXOMAGNETOPHONONIC_RESONANCE_DISSIPATION/5
0.00140, 210
/ENG/EM_FLEXOMAGNETOPHONONIC_RESONANCE/6
0.00148, 220
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexomagnetophononic_resonance_energies
    assert 4 in model.eng_flexomagnetophononic_resonance_energies
    assert 5 in model.eng_flexomagnetophononic_resonance_energies
    assert 6 in model.eng_flexomagnetophononic_resonance_energies


def test_m362_eng_flexomagnetophononic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOMAGNETOPHONONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m362_lagmul_hunt_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{481:>10d}{482:>10d}{483:>10d}{4.35e7:>20.1f}{132:>10d}{2.8e-5:>20.6e}"
    c2 = f"{225.0:>20.4f}{215.0:>20.4f}{195.0:>20.4f}{145.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Hunt Spatial Linkage Joint Fixed Format Test
2022 0
/HUNT_SPATIAL_LINKAGE_JOINT/445
Hunt Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 445 in model.lagmul_hunt_spatial_linkage_joints
    joint = model.lagmul_hunt_spatial_linkage_joints[445]
    assert joint.node1 == 481
    assert joint.node2 == 482
    assert joint.node3 == 483
    assert pytest.approx(joint.stiff) == 4.35e7
    assert joint.skew_id == 132
    assert pytest.approx(joint.tol) == 2.8e-5
    assert pytest.approx(joint.link_len_a) == 225.0
    assert pytest.approx(joint.link_len_b) == 215.0
    assert pytest.approx(joint.twist_angle_alpha) == 195.0
    assert pytest.approx(joint.offset_distance_s) == 145.0


def test_m362_lagmul_hunt_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Hunt Spatial Linkage Joint Free Format Test
/LAGMUL/HUNT_SPATIAL_LINKAGE_JOINT/446
581, 582, 583, 29.0e6, 152, 3.8e-5
228.0, 218.0, 198.0, 148.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 446 in model.lagmul_hunt_spatial_linkage_joints
    joint = model.lagmul_hunt_spatial_linkage_joints[446]
    assert joint.node1 == 581
    assert joint.node2 == 582
    assert joint.node3 == 583
    assert pytest.approx(joint.stiff) == 29.0e6
    assert joint.skew_id == 152
    assert pytest.approx(joint.tol) == 3.8e-5
    assert pytest.approx(joint.link_len_a) == 228.0
    assert pytest.approx(joint.link_len_b) == 218.0
    assert pytest.approx(joint.twist_angle_alpha) == 198.0
    assert pytest.approx(joint.offset_distance_s) == 148.0


def test_m362_lagmul_hunt_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Hunt Spatial Linkage Joint Aliases Test
/LAGMUL/HUNT_SPATIAL_LINKAGE/447
1, 2, 3, 1e6, 0, 1e-6
60.0, 60.0, 106.0, 46.0
/HUNT_SPATIAL_LINKAGE/448
1, 2, 3, 1e6, 0, 1e-6
60.0, 60.0, 106.0, 46.0
/HUNT_SPATIAL_MULTI_LOOP_MECHANISM/449
1, 2, 3, 1e6, 0, 1e-6
60.0, 60.0, 106.0, 46.0
/HUNT_SPATIAL_SPECIAL_SYMMETRIC_MECHANISM/450
1, 2, 3, 1e6, 0, 1e-6
60.0, 60.0, 106.0, 46.0
/HUNT_SPATIAL_6R_MECHANISM/451
1, 2, 3, 1e6, 0, 1e-6
60.0, 60.0, 106.0, 46.0
/HUNT_SPATIAL_OVERCONSTRAINED_MECHANISM/452
1, 2, 3, 1e6, 0, 1e-6
60.0, 60.0, 106.0, 46.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 447 in model.lagmul_hunt_spatial_linkage_joints
    assert 448 in model.lagmul_hunt_spatial_linkage_joints
    assert 449 in model.lagmul_hunt_spatial_linkage_joints
    assert 450 in model.lagmul_hunt_spatial_linkage_joints
    assert 451 in model.lagmul_hunt_spatial_linkage_joints
    assert 452 in model.lagmul_hunt_spatial_linkage_joints


def test_m362_lagmul_hunt_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/HUNT_SPATIAL_LINKAGE_JOINT/453
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m362_sensor_spring_transverse_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{1225:>10d}{7.65e9:>20.1f}{0.0345:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Transverse Crackle Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE/478
Spring Transverse Crackle Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 478 in model.sensor_spring_transverse_crackle_rates
    sensor = model.sensor_spring_transverse_crackle_rates[478]
    assert sensor.spring_id == 1225
    assert pytest.approx(sensor.jtrans_crk_max) == 7.65e9
    assert pytest.approx(sensor.t_delay) == 0.0345
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_CRACKLE_RATE"


def test_m362_sensor_spring_transverse_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Transverse Crackle Rate Sensor Free Format Test
/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE/479
1226, 7.75e9, 0.0370
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 479 in model.sensor_spring_transverse_crackle_rates
    sensor = model.sensor_spring_transverse_crackle_rates[479]
    assert sensor.spring_id == 1226
    assert pytest.approx(sensor.jtrans_crk_max) == 7.75e9
    assert pytest.approx(sensor.t_delay) == 0.0370


def test_m362_sensor_spring_transverse_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Transverse Crackle Rate Sensor Aliases Test
/SENSOR/SPRING_TRANS_CRACKLE_RATE/480
1227, 7.8e9, 0.0275
/SENSOR/SPRING_RATE_CRACKLE_TRANS/481
1228, 7.8e9, 0.0275
/SENSOR/TRANSVERSE_CRACKLE_RATE_SPRING/482
1229, 7.8e9, 0.0275
/SENSOR/SPRING_CRACKLE_TRANS/483
1230, 7.8e9, 0.0275
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 480 in model.sensor_spring_transverse_crackle_rates
    assert 481 in model.sensor_spring_transverse_crackle_rates
    assert 482 in model.sensor_spring_transverse_crackle_rates
    assert 483 in model.sensor_spring_transverse_crackle_rates
    assert len(model.sensors) == 4


def test_m362_sensor_spring_transverse_crackle_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE/484
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

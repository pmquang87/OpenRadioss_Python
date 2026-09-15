"""Tests for Milestone M364: LadCoupleMatrixShearDegradationRate Failure Model, EngFlexomagnetopolaritonicResonanceEnergy, WunderlichSpatialLinkageJoint, and SensorSpringTorsionalCrackleRate."""

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


def test_m364_fail_lad_couple_matrix_shear_degradation_rate_fixed(tmp_path: Path):
    c1 = f"{152.0:>20.4f}{456.0:>20.4f}{64.0:>20.4f}{2.70:>20.4f}{0.972:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1800:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Matrix Shear Degradation Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_MATRIX_SHEAR_DEGRADATION_RATE/1800
Ladeveze Coupled Matrix Shear Degradation Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1800 in model.fail_ladcouplematrixsheardegradationrates
    fcmsdr = model.fail_ladcouplematrixsheardegradationrates[1800]
    assert pytest.approx(fcmsdr.sigma_cmsdr0) == 152.0
    assert pytest.approx(fcmsdr.sigma_cmsdrc) == 456.0
    assert pytest.approx(fcmsdr.gamma_cmsdr) == 64.0
    assert pytest.approx(fcmsdr.p_cmsdr) == 2.70
    assert pytest.approx(fcmsdr.d_cmsdr_max) == 0.972
    assert fcmsdr.ifail_sh == 1
    assert fcmsdr.ifail_so == 2
    assert fcmsdr.fail_id == 1800
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_MATRIX_SHEAR_DEGRADATION_RATE"


def test_m364_fail_lad_couple_matrix_shear_degradation_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Matrix Shear Degradation Rate Free Format Test
/FAIL/LAD_COUPLE_MATRIX_SHEAR_DEGRADATION_RATE/1801
162.0, 486.0, 70.0, 2.90, 0.958
1, 1
1801
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1801 in model.fail_ladcouplematrixsheardegradationrates
    fcmsdr = model.fail_ladcouplematrixsheardegradationrates[1801]
    assert pytest.approx(fcmsdr.sigma_cmsdr0) == 162.0
    assert pytest.approx(fcmsdr.sigma_cmsdrc) == 486.0
    assert pytest.approx(fcmsdr.gamma_cmsdr) == 70.0
    assert pytest.approx(fcmsdr.p_cmsdr) == 2.90
    assert pytest.approx(fcmsdr.d_cmsdr_max) == 0.958
    assert fcmsdr.fail_id == 1801


def test_m364_fail_lad_couple_matrix_shear_degradation_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Matrix Shear Degradation Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_MATRIX_SHEAR_DEGRADATION_RATE/1802
142.0, 426.0, 58.0, 2.22, 0.997
1, 1
/FAIL/LAD_CMSDR/1803
142.0, 426.0, 58.0, 2.22, 0.997
1, 1
/FAIL/LAD_CMSDR_MODEL/1804
142.0, 426.0, 58.0, 2.22, 0.997
1, 1
/FAIL/LAD_CMSDR_LAW/1805
142.0, 426.0, 58.0, 2.22, 0.997
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_MATRIX_SHEAR_DEGRADATION/1806
142.0, 426.0, 58.0, 2.22, 0.997
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1802 in model.fail_ladcouplematrixsheardegradationrates
    assert 1803 in model.fail_ladcouplematrixsheardegradationrates
    assert 1804 in model.fail_ladcouplematrixsheardegradationrates
    assert 1805 in model.fail_ladcouplematrixsheardegradationrates
    assert 1806 in model.fail_ladcouplematrixsheardegradationrates
    assert len(model.raw_fails) == 5


def test_m364_fail_lad_couple_matrix_shear_degradation_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_COUPLE_MATRIX_SHEAR_DEGRADATION_RATE/1807
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m364_eng_flexomagnetopolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00180:>20.6f}{270:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexomagnetopolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPOLARITONIC_RESONANCE_ENERGY/1
Flexomagnetopolaritonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexomagnetopolaritonic_resonance_energies
    eng = model.eng_flexomagnetopolaritonic_resonance_energies[1]
    assert pytest.approx(eng.dt_fmpor) == 0.00180
    assert eng.sens_id == 270


def test_m364_eng_flexomagnetopolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexomagnetopolaritonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPOLARITONIC_RESONANCE_ENERGY/2
0.00200, 280
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexomagnetopolaritonic_resonance_energies
    eng = model.eng_flexomagnetopolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_fmpor) == 0.00200
    assert eng.sens_id == 280


def test_m364_eng_flexomagnetopolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexomagnetopolaritonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_POLARITON_RES_WORK/3
0.00140, 205
/ENG/EFLEXOMAGNETOPOLARITONICRESONANCE/4
0.00144, 210
/ENG/FLEXOMAGNETOPOLARITONIC_RESONANCE_DISSIPATION/5
0.00150, 220
/ENG/EM_FLEXOMAGNETOPOLARITONIC_RESONANCE/6
0.00158, 230
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexomagnetopolaritonic_resonance_energies
    assert 4 in model.eng_flexomagnetopolaritonic_resonance_energies
    assert 5 in model.eng_flexomagnetopolaritonic_resonance_energies
    assert 6 in model.eng_flexomagnetopolaritonic_resonance_energies


def test_m364_eng_flexomagnetopolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOMAGNETOPOLARITONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m364_lagmul_wunderlich_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{501:>10d}{502:>10d}{503:>10d}{4.55e7:>20.1f}{136:>10d}{2.4e-5:>20.6e}"
    c2 = f"{235.0:>20.4f}{225.0:>20.4f}{205.0:>20.4f}{155.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Wunderlich Spatial Linkage Joint Fixed Format Test
2022 0
/WUNDERLICH_SPATIAL_LINKAGE_JOINT/465
Wunderlich Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 465 in model.lagmul_wunderlich_spatial_linkage_joints
    joint = model.lagmul_wunderlich_spatial_linkage_joints[465]
    assert joint.node1 == 501
    assert joint.node2 == 502
    assert joint.node3 == 503
    assert pytest.approx(joint.stiff) == 4.55e7
    assert joint.skew_id == 136
    assert pytest.approx(joint.tol) == 2.4e-5
    assert pytest.approx(joint.link_len_a) == 235.0
    assert pytest.approx(joint.link_len_b) == 225.0
    assert pytest.approx(joint.twist_angle_alpha) == 205.0
    assert pytest.approx(joint.offset_distance_s) == 155.0


def test_m364_lagmul_wunderlich_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Wunderlich Spatial Linkage Joint Free Format Test
/LAGMUL/WUNDERLICH_SPATIAL_LINKAGE_JOINT/466
601, 602, 603, 31.0e6, 156, 3.4e-5
240.0, 230.0, 210.0, 160.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 466 in model.lagmul_wunderlich_spatial_linkage_joints
    joint = model.lagmul_wunderlich_spatial_linkage_joints[466]
    assert joint.node1 == 601
    assert joint.node2 == 602
    assert joint.node3 == 603
    assert pytest.approx(joint.stiff) == 31.0e6
    assert joint.skew_id == 156
    assert pytest.approx(joint.tol) == 3.4e-5
    assert pytest.approx(joint.link_len_a) == 240.0
    assert pytest.approx(joint.link_len_b) == 230.0
    assert pytest.approx(joint.twist_angle_alpha) == 210.0
    assert pytest.approx(joint.offset_distance_s) == 160.0


def test_m364_lagmul_wunderlich_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Wunderlich Spatial Linkage Joint Aliases Test
/LAGMUL/WUNDERLICH_SPATIAL_LINKAGE/467
1, 2, 3, 1e6, 0, 1e-6
64.0, 64.0, 110.0, 50.0
/WUNDERLICH_SPATIAL_LINKAGE/468
1, 2, 3, 1e6, 0, 1e-6
64.0, 64.0, 110.0, 50.0
/WUNDERLICH_SPATIAL_MULTI_LOOP_MECHANISM/469
1, 2, 3, 1e6, 0, 1e-6
64.0, 64.0, 110.0, 50.0
/WUNDERLICH_SPATIAL_BISTABLE_MECHANISM/470
1, 2, 3, 1e6, 0, 1e-6
64.0, 64.0, 110.0, 50.0
/WUNDERLICH_SPATIAL_6R_MECHANISM/471
1, 2, 3, 1e6, 0, 1e-6
64.0, 64.0, 110.0, 50.0
/WUNDERLICH_SPATIAL_OVERCONSTRAINED_MECHANISM/472
1, 2, 3, 1e6, 0, 1e-6
64.0, 64.0, 110.0, 50.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 467 in model.lagmul_wunderlich_spatial_linkage_joints
    assert 468 in model.lagmul_wunderlich_spatial_linkage_joints
    assert 469 in model.lagmul_wunderlich_spatial_linkage_joints
    assert 470 in model.lagmul_wunderlich_spatial_linkage_joints
    assert 471 in model.lagmul_wunderlich_spatial_linkage_joints
    assert 472 in model.lagmul_wunderlich_spatial_linkage_joints


def test_m364_lagmul_wunderlich_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/WUNDERLICH_SPATIAL_LINKAGE_JOINT/473
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m364_sensor_spring_torsional_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{1245:>10d}{8.05e9:>20.1f}{0.0365:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Torsional Crackle Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/498
Spring Torsional Crackle Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 498 in model.sensor_spring_torsional_crackle_rates
    sensor = model.sensor_spring_torsional_crackle_rates[498]
    assert sensor.spring_id == 1245
    assert pytest.approx(sensor.jtors_crk_max) == 8.05e9
    assert pytest.approx(sensor.t_delay) == 0.0365
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_CRACKLE_RATE"


def test_m364_sensor_spring_torsional_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Torsional Crackle Rate Sensor Free Format Test
/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/499
1246, 8.15e9, 0.0390
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 499 in model.sensor_spring_torsional_crackle_rates
    sensor = model.sensor_spring_torsional_crackle_rates[499]
    assert sensor.spring_id == 1246
    assert pytest.approx(sensor.jtors_crk_max) == 8.15e9
    assert pytest.approx(sensor.t_delay) == 0.0390


def test_m364_sensor_spring_torsional_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Torsional Crackle Rate Sensor Aliases Test
/SENSOR/SPRING_TORS_CRACKLE_RATE/500
1247, 8.2e9, 0.0295
/SENSOR/SPRING_RATE_CRACKLE_TORS/501
1248, 8.2e9, 0.0295
/SENSOR/TORSIONAL_CRACKLE_RATE_SPRING/502
1249, 8.2e9, 0.0295
/SENSOR/SPRING_CRACKLE_TORS/503
1250, 8.2e9, 0.0295
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 500 in model.sensor_spring_torsional_crackle_rates
    assert 501 in model.sensor_spring_torsional_crackle_rates
    assert 502 in model.sensor_spring_torsional_crackle_rates
    assert 503 in model.sensor_spring_torsional_crackle_rates
    assert len(model.sensors) == 4


def test_m364_sensor_spring_torsional_crackle_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/504
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

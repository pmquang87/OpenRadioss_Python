"""Tests for Milestone M363: LadTransverseMatrixShearDegradationRate Failure Model, EngFlexomagnetoexcitonicResonanceEnergy, ChenSpatialLinkageJoint, and SensorSpringTotalCrackleRate."""

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


def test_m363_fail_lad_transverse_matrix_shear_degradation_rate_fixed(tmp_path: Path):
    c1 = f"{150.0:>20.4f}{450.0:>20.4f}{62.0:>20.4f}{2.65:>20.4f}{0.974:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1790:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Matrix Shear Degradation Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_MATRIX_SHEAR_DEGRADATION_RATE/1790
Ladeveze Transverse Matrix Shear Degradation Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1790 in model.fail_ladtransversematrixsheardegradationrates
    ftmsdr = model.fail_ladtransversematrixsheardegradationrates[1790]
    assert pytest.approx(ftmsdr.sigma_tmsdr0) == 150.0
    assert pytest.approx(ftmsdr.sigma_tmsdrc) == 450.0
    assert pytest.approx(ftmsdr.gamma_tmsdr) == 62.0
    assert pytest.approx(ftmsdr.p_tmsdr) == 2.65
    assert pytest.approx(ftmsdr.d_tmsdr_max) == 0.974
    assert ftmsdr.ifail_sh == 1
    assert ftmsdr.ifail_so == 2
    assert ftmsdr.fail_id == 1790
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_MATRIX_SHEAR_DEGRADATION_RATE"


def test_m363_fail_lad_transverse_matrix_shear_degradation_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Matrix Shear Degradation Rate Free Format Test
/FAIL/LAD_TRANSVERSE_MATRIX_SHEAR_DEGRADATION_RATE/1791
160.0, 480.0, 68.0, 2.85, 0.960
1, 1
1791
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1791 in model.fail_ladtransversematrixsheardegradationrates
    ftmsdr = model.fail_ladtransversematrixsheardegradationrates[1791]
    assert pytest.approx(ftmsdr.sigma_tmsdr0) == 160.0
    assert pytest.approx(ftmsdr.sigma_tmsdrc) == 480.0
    assert pytest.approx(ftmsdr.gamma_tmsdr) == 68.0
    assert pytest.approx(ftmsdr.p_tmsdr) == 2.85
    assert pytest.approx(ftmsdr.d_tmsdr_max) == 0.960
    assert ftmsdr.fail_id == 1791


def test_m363_fail_lad_transverse_matrix_shear_degradation_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Matrix Shear Degradation Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_MATRIX_SHEAR_DEGRADATION_RATE/1792
140.0, 420.0, 56.0, 2.20, 0.999
1, 1
/FAIL/LAD_TMSDR/1793
140.0, 420.0, 56.0, 2.20, 0.999
1, 1
/FAIL/LAD_TMSDR_MODEL/1794
140.0, 420.0, 56.0, 2.20, 0.999
1, 1
/FAIL/LAD_TMSDR_LAW/1795
140.0, 420.0, 56.0, 2.20, 0.999
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_MATRIX_SHEAR_DEGRADATION/1796
140.0, 420.0, 56.0, 2.20, 0.999
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1792 in model.fail_ladtransversematrixsheardegradationrates
    assert 1793 in model.fail_ladtransversematrixsheardegradationrates
    assert 1794 in model.fail_ladtransversematrixsheardegradationrates
    assert 1795 in model.fail_ladtransversematrixsheardegradationrates
    assert 1796 in model.fail_ladtransversematrixsheardegradationrates
    assert len(model.raw_fails) == 5


def test_m363_fail_lad_transverse_matrix_shear_degradation_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_TRANSVERSE_MATRIX_SHEAR_DEGRADATION_RATE/1797
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m363_eng_flexomagnetoexcitonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00175:>20.6f}{265:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexomagnetoexcitonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOEXCITONIC_RESONANCE_ENERGY/1
Flexomagnetoexcitonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexomagnetoexcitonic_resonance_energies
    eng = model.eng_flexomagnetoexcitonic_resonance_energies[1]
    assert pytest.approx(eng.dt_fmer) == 0.00175
    assert eng.sens_id == 265


def test_m363_eng_flexomagnetoexcitonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexomagnetoexcitonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOEXCITONIC_RESONANCE_ENERGY/2
0.00195, 275
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexomagnetoexcitonic_resonance_energies
    eng = model.eng_flexomagnetoexcitonic_resonance_energies[2]
    assert pytest.approx(eng.dt_fmer) == 0.00195
    assert eng.sens_id == 275


def test_m363_eng_flexomagnetoexcitonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexomagnetoexcitonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_EXCITON_RES_WORK/3
0.00135, 200
/ENG/EFLEXOMAGNETOEXCITONICRESONANCE/4
0.00139, 205
/ENG/FLEXOMAGNETOEXCITONIC_RESONANCE_DISSIPATION/5
0.00145, 215
/ENG/EM_FLEXOMAGNETOEXCITONIC_RESONANCE/6
0.00152, 225
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexomagnetoexcitonic_resonance_energies
    assert 4 in model.eng_flexomagnetoexcitonic_resonance_energies
    assert 5 in model.eng_flexomagnetoexcitonic_resonance_energies
    assert 6 in model.eng_flexomagnetoexcitonic_resonance_energies


def test_m363_eng_flexomagnetoexcitonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOMAGNETOEXCITONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m363_lagmul_chen_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{491:>10d}{492:>10d}{493:>10d}{4.45e7:>20.1f}{134:>10d}{2.6e-5:>20.6e}"
    c2 = f"{230.0:>20.4f}{220.0:>20.4f}{200.0:>20.4f}{150.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Chen Spatial Linkage Joint Fixed Format Test
2022 0
/CHEN_SPATIAL_LINKAGE_JOINT/455
Chen Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 455 in model.lagmul_chen_spatial_linkage_joints
    joint = model.lagmul_chen_spatial_linkage_joints[455]
    assert joint.node1 == 491
    assert joint.node2 == 492
    assert joint.node3 == 493
    assert pytest.approx(joint.stiff) == 4.45e7
    assert joint.skew_id == 134
    assert pytest.approx(joint.tol) == 2.6e-5
    assert pytest.approx(joint.link_len_a) == 230.0
    assert pytest.approx(joint.link_len_b) == 220.0
    assert pytest.approx(joint.twist_angle_alpha) == 200.0
    assert pytest.approx(joint.offset_distance_s) == 150.0


def test_m363_lagmul_chen_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Chen Spatial Linkage Joint Free Format Test
/LAGMUL/CHEN_SPATIAL_LINKAGE_JOINT/456
591, 592, 593, 30.0e6, 154, 3.6e-5
234.0, 224.0, 204.0, 154.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 456 in model.lagmul_chen_spatial_linkage_joints
    joint = model.lagmul_chen_spatial_linkage_joints[456]
    assert joint.node1 == 591
    assert joint.node2 == 592
    assert joint.node3 == 593
    assert pytest.approx(joint.stiff) == 30.0e6
    assert joint.skew_id == 154
    assert pytest.approx(joint.tol) == 3.6e-5
    assert pytest.approx(joint.link_len_a) == 234.0
    assert pytest.approx(joint.link_len_b) == 224.0
    assert pytest.approx(joint.twist_angle_alpha) == 204.0
    assert pytest.approx(joint.offset_distance_s) == 154.0


def test_m363_lagmul_chen_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Chen Spatial Linkage Joint Aliases Test
/LAGMUL/CHEN_SPATIAL_LINKAGE/457
1, 2, 3, 1e6, 0, 1e-6
62.0, 62.0, 108.0, 48.0
/CHEN_SPATIAL_LINKAGE/458
1, 2, 3, 1e6, 0, 1e-6
62.0, 62.0, 108.0, 48.0
/CHEN_SPATIAL_MULTI_LOOP_MECHANISM/459
1, 2, 3, 1e6, 0, 1e-6
62.0, 62.0, 108.0, 48.0
/CHEN_SPATIAL_SYMMETRIC_MECHANISM/460
1, 2, 3, 1e6, 0, 1e-6
62.0, 62.0, 108.0, 48.0
/CHEN_SPATIAL_6R_MECHANISM/461
1, 2, 3, 1e6, 0, 1e-6
62.0, 62.0, 108.0, 48.0
/CHEN_SPATIAL_OVERCONSTRAINED_MECHANISM/462
1, 2, 3, 1e6, 0, 1e-6
62.0, 62.0, 108.0, 48.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 457 in model.lagmul_chen_spatial_linkage_joints
    assert 458 in model.lagmul_chen_spatial_linkage_joints
    assert 459 in model.lagmul_chen_spatial_linkage_joints
    assert 460 in model.lagmul_chen_spatial_linkage_joints
    assert 461 in model.lagmul_chen_spatial_linkage_joints
    assert 462 in model.lagmul_chen_spatial_linkage_joints


def test_m363_lagmul_chen_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/CHEN_SPATIAL_LINKAGE_JOINT/463
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m363_sensor_spring_total_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{1235:>10d}{7.85e9:>20.1f}{0.0355:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Total Crackle Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_CRACKLE_RATE/488
Spring Total Crackle Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 488 in model.sensor_spring_total_crackle_rates
    sensor = model.sensor_spring_total_crackle_rates[488]
    assert sensor.spring_id == 1235
    assert pytest.approx(sensor.jtot_crk_max) == 7.85e9
    assert pytest.approx(sensor.t_delay) == 0.0355
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_CRACKLE_RATE"


def test_m363_sensor_spring_total_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Total Crackle Rate Sensor Free Format Test
/SENSOR/SPRING_TOTAL_CRACKLE_RATE/489
1236, 7.95e9, 0.0380
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 489 in model.sensor_spring_total_crackle_rates
    sensor = model.sensor_spring_total_crackle_rates[489]
    assert sensor.spring_id == 1236
    assert pytest.approx(sensor.jtot_crk_max) == 7.95e9
    assert pytest.approx(sensor.t_delay) == 0.0380


def test_m363_sensor_spring_total_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Total Crackle Rate Sensor Aliases Test
/SENSOR/SPRING_TOT_CRACKLE_RATE/490
1237, 8.0e9, 0.0285
/SENSOR/SPRING_RATE_CRACKLE_TOT/491
1238, 8.0e9, 0.0285
/SENSOR/TOTAL_CRACKLE_RATE_SPRING/492
1239, 8.0e9, 0.0285
/SENSOR/SPRING_CRACKLE_TOT/493
1240, 8.0e9, 0.0285
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 490 in model.sensor_spring_total_crackle_rates
    assert 491 in model.sensor_spring_total_crackle_rates
    assert 492 in model.sensor_spring_total_crackle_rates
    assert 493 in model.sensor_spring_total_crackle_rates
    assert len(model.sensors) == 4


def test_m363_sensor_spring_total_crackle_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TOTAL_CRACKLE_RATE/494
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

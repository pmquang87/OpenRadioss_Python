"""Tests for Milestone M354: LadTransverseMatrixMicrocrackingRate Failure Model, EngFlexothermoexcitonmagnonpolaritonicResonanceEnergy, GoldbergSpatialLinkageJoint, and SensorSpringTotalAngularSurgeRate."""

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


def test_m354_fail_lad_transverse_matrix_microcracking_rate_fixed(tmp_path: Path):
    c1 = f"{128.0:>20.4f}{384.0:>20.4f}{44.0:>20.4f}{2.20:>20.4f}{0.992:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1700:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Matrix Microcracking Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_MATRIX_MICROCRACKING_RATE/1700
Ladeveze Transverse Matrix Microcracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1700 in model.fail_ladtransversematrixmicrocrackingrates
    ftmmr = model.fail_ladtransversematrixmicrocrackingrates[1700]
    assert pytest.approx(ftmmr.sigma_tmmr0) == 128.0
    assert pytest.approx(ftmmr.sigma_tmmrc) == 384.0
    assert pytest.approx(ftmmr.gamma_tmmr) == 44.0
    assert pytest.approx(ftmmr.p_tmmr) == 2.20
    assert pytest.approx(ftmmr.d_tmmr_max) == 0.992
    assert ftmmr.ifail_sh == 1
    assert ftmmr.ifail_so == 2
    assert ftmmr.fail_id == 1700
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_MATRIX_MICROCRACKING_RATE"


def test_m354_fail_lad_transverse_matrix_microcracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Matrix Microcracking Rate Free Format Test
/FAIL/LAD_TRANSVERSE_MATRIX_MICROCRACKING_RATE/1701
138.0, 414.0, 50.0, 2.40, 0.978
1, 1
1701
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1701 in model.fail_ladtransversematrixmicrocrackingrates
    ftmmr = model.fail_ladtransversematrixmicrocrackingrates[1701]
    assert pytest.approx(ftmmr.sigma_tmmr0) == 138.0
    assert pytest.approx(ftmmr.sigma_tmmrc) == 414.0
    assert pytest.approx(ftmmr.gamma_tmmr) == 50.0
    assert pytest.approx(ftmmr.p_tmmr) == 2.40
    assert pytest.approx(ftmmr.d_tmmr_max) == 0.978
    assert ftmmr.fail_id == 1701


def test_m354_fail_lad_transverse_matrix_microcracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Matrix Microcracking Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_MATRIX_MICROCRACKING_RATE/1702
118.0, 354.0, 39.0, 1.98, 0.982
1, 1
/FAIL/LAD_TMMR/1703
118.0, 354.0, 39.0, 1.98, 0.982
1, 1
/FAIL/LAD_TMMR_MODEL/1704
118.0, 354.0, 39.0, 1.98, 0.982
1, 1
/FAIL/LAD_TMMR_LAW/1705
118.0, 354.0, 39.0, 1.98, 0.982
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_MATRIX_MICROCRACKING/1706
118.0, 354.0, 39.0, 1.98, 0.982
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1702 in model.fail_ladtransversematrixmicrocrackingrates
    assert 1703 in model.fail_ladtransversematrixmicrocrackingrates
    assert 1704 in model.fail_ladtransversematrixmicrocrackingrates
    assert 1705 in model.fail_ladtransversematrixmicrocrackingrates
    assert 1706 in model.fail_ladtransversematrixmicrocrackingrates
    assert len(model.raw_fails) == 5


def test_m354_fail_lad_transverse_matrix_microcracking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_TRANSVERSE_MATRIX_MICROCRACKING_RATE/1707
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m354_eng_flexothermoexcitonmagnonpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00130:>20.6f}{225:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoexcitonmagnonpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOEXCITONMAGNONPOLARITONIC_RESONANCE_ENERGY/1
Flexothermoexcitonmagnonpolaritonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoexcitonmagnonpolaritonic_resonance_energies
    eng = model.eng_flexothermoexcitonmagnonpolaritonic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftempr) == 0.00130
    assert eng.sens_id == 225


def test_m354_eng_flexothermoexcitonmagnonpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermoexcitonmagnonpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOEXCITONMAGNONPOLARITONIC_RESONANCE_ENERGY/2
0.00150, 235
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermoexcitonmagnonpolaritonic_resonance_energies
    eng = model.eng_flexothermoexcitonmagnonpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftempr) == 0.00150
    assert eng.sens_id == 235


def test_m354_eng_flexothermoexcitonmagnonpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoexcitonmagnonpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_EXCITON_MAGNON_POLARITON_RES_WORK/3
0.00103, 172
/ENG/EFLEXOTHERMOEXCITONMAGNONPOLARITONICRESONANCE/4
0.00108, 178
/ENG/FLEXOTHERMOEXCITONMAGNONPOLARITONIC_RESONANCE_DISSIPATION/5
0.00112, 188
/ENG/EM_FLEXOTHERMOEXCITONMAGNONPOLARITONIC_RESONANCE/6
0.00120, 198
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoexcitonmagnonpolaritonic_resonance_energies
    assert 4 in model.eng_flexothermoexcitonmagnonpolaritonic_resonance_energies
    assert 5 in model.eng_flexothermoexcitonmagnonpolaritonic_resonance_energies
    assert 6 in model.eng_flexothermoexcitonmagnonpolaritonic_resonance_energies


def test_m354_eng_flexothermoexcitonmagnonpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOEXCITONMAGNONPOLARITONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m354_lagmul_goldberg_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{401:>10d}{402:>10d}{403:>10d}{3.55e7:>20.1f}{115:>10d}{4.5e-5:>20.6e}"
    c2 = f"{185.0:>20.4f}{175.0:>20.4f}{155.0:>20.4f}{105.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Goldberg Spatial Linkage Joint Fixed Format Test
2022 0
/GOLDBERG_SPATIAL_LINKAGE_JOINT/365
Goldberg Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 365 in model.lagmul_goldberg_spatial_linkage_joints
    joint = model.lagmul_goldberg_spatial_linkage_joints[365]
    assert joint.node1 == 401
    assert joint.node2 == 402
    assert joint.node3 == 403
    assert pytest.approx(joint.stiff) == 3.55e7
    assert joint.skew_id == 115
    assert pytest.approx(joint.tol) == 4.5e-5
    assert pytest.approx(joint.link_len_a) == 185.0
    assert pytest.approx(joint.link_len_b) == 175.0
    assert pytest.approx(joint.twist_angle_alpha) == 155.0
    assert pytest.approx(joint.offset_distance_s) == 105.0


def test_m354_lagmul_goldberg_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Goldberg Spatial Linkage Joint Free Format Test
/LAGMUL/GOLDBERG_SPATIAL_LINKAGE_JOINT/366
501, 502, 503, 21.0e6, 135, 5.5e-5
186.0, 176.0, 156.0, 106.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 366 in model.lagmul_goldberg_spatial_linkage_joints
    joint = model.lagmul_goldberg_spatial_linkage_joints[366]
    assert joint.node1 == 501
    assert joint.node2 == 502
    assert joint.node3 == 503
    assert pytest.approx(joint.stiff) == 21.0e6
    assert joint.skew_id == 135
    assert pytest.approx(joint.tol) == 5.5e-5
    assert pytest.approx(joint.link_len_a) == 186.0
    assert pytest.approx(joint.link_len_b) == 176.0
    assert pytest.approx(joint.twist_angle_alpha) == 156.0
    assert pytest.approx(joint.offset_distance_s) == 106.0


def test_m354_lagmul_goldberg_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Goldberg Spatial Linkage Joint Aliases Test
/LAGMUL/GOLDBERG_SPATIAL_LINKAGE/367
1, 2, 3, 1e6, 0, 1e-6
42.0, 42.0, 88.0, 26.0
/GOLDBERG_SPATIAL_LINKAGE/368
1, 2, 3, 1e6, 0, 1e-6
42.0, 42.0, 88.0, 26.0
/GOLDBERG_SPATIAL_MULTI_LOOP_MECHANISM/369
1, 2, 3, 1e6, 0, 1e-6
42.0, 42.0, 88.0, 26.0
/GOLDBERG_SPATIAL_5R_6R_MECHANISM/370
1, 2, 3, 1e6, 0, 1e-6
42.0, 42.0, 88.0, 26.0
/GOLDBERG_SPATIAL_VARIABLE_ANGLE_MECHANISM/371
1, 2, 3, 1e6, 0, 1e-6
42.0, 42.0, 88.0, 26.0
/GOLDBERG_SPATIAL_OVERCONSTRAINED_MECHANISM/372
1, 2, 3, 1e6, 0, 1e-6
42.0, 42.0, 88.0, 26.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 367 in model.lagmul_goldberg_spatial_linkage_joints
    assert 368 in model.lagmul_goldberg_spatial_linkage_joints
    assert 369 in model.lagmul_goldberg_spatial_linkage_joints
    assert 370 in model.lagmul_goldberg_spatial_linkage_joints
    assert 371 in model.lagmul_goldberg_spatial_linkage_joints
    assert 372 in model.lagmul_goldberg_spatial_linkage_joints


def test_m354_lagmul_goldberg_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/GOLDBERG_SPATIAL_LINKAGE_JOINT/373
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m354_sensor_spring_total_angular_surge_rate_fixed(tmp_path: Path):
    c1 = f"{1145:>10d}{5.85e9:>20.1f}{0.0265:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Total Angular Surge Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_SURGE_RATE/398
Spring Total Angular Surge Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 398 in model.sensor_spring_total_angular_surge_rates
    sensor = model.sensor_spring_total_angular_surge_rates[398]
    assert sensor.spring_id == 1145
    assert pytest.approx(sensor.jrot_tot_surge_max) == 5.85e9
    assert pytest.approx(sensor.t_delay) == 0.0265
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_SURGE_RATE"


def test_m354_sensor_spring_total_angular_surge_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Total Angular Surge Rate Sensor Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_SURGE_RATE/399
1146, 5.95e9, 0.0290
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 399 in model.sensor_spring_total_angular_surge_rates
    sensor = model.sensor_spring_total_angular_surge_rates[399]
    assert sensor.spring_id == 1146
    assert pytest.approx(sensor.jrot_tot_surge_max) == 5.95e9
    assert pytest.approx(sensor.t_delay) == 0.0290


def test_m354_sensor_spring_total_angular_surge_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Total Angular Surge Rate Sensor Aliases Test
/SENSOR/SPRING_TOT_ANG_SURGE_RATE/400
1147, 6.0e9, 0.0195
/SENSOR/SPRING_RATE_SURGE_ANG_TOT/401
1148, 6.0e9, 0.0195
/SENSOR/TOTAL_ANGULAR_SURGE_RATE_SPRING/402
1149, 6.0e9, 0.0195
/SENSOR/SPRING_SURGE_ANG_TOT/403
1150, 6.0e9, 0.0195
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 400 in model.sensor_spring_total_angular_surge_rates
    assert 401 in model.sensor_spring_total_angular_surge_rates
    assert 402 in model.sensor_spring_total_angular_surge_rates
    assert 403 in model.sensor_spring_total_angular_surge_rates
    assert len(model.sensors) == 4


def test_m354_sensor_spring_total_angular_surge_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TOTAL_ANGULAR_SURGE_RATE/404
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

"""Tests for Milestone M355: LadCoupleMatrixMicrocrackingRate Failure Model, EngFlexothermophononmagnonpolaritonicResonanceEnergy, SarrusSpatialLinkageJoint, and SensorSpringNormalPopRate."""

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


def test_m355_fail_lad_couple_matrix_microcracking_rate_fixed(tmp_path: Path):
    c1 = f"{132.0:>20.4f}{396.0:>20.4f}{46.0:>20.4f}{2.25:>20.4f}{0.990:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1710:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Matrix Microcracking Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_MATRIX_MICROCRACKING_RATE/1710
Ladeveze Coupled Matrix Microcracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1710 in model.fail_ladcouplematrixmicrocrackingrates
    fcmmr = model.fail_ladcouplematrixmicrocrackingrates[1710]
    assert pytest.approx(fcmmr.sigma_cmmr0) == 132.0
    assert pytest.approx(fcmmr.sigma_cmmrc) == 396.0
    assert pytest.approx(fcmmr.gamma_cmmr) == 46.0
    assert pytest.approx(fcmmr.p_cmmr) == 2.25
    assert pytest.approx(fcmmr.d_cmmr_max) == 0.990
    assert fcmmr.ifail_sh == 1
    assert fcmmr.ifail_so == 2
    assert fcmmr.fail_id == 1710
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_MATRIX_MICROCRACKING_RATE"


def test_m355_fail_lad_couple_matrix_microcracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Matrix Microcracking Rate Free Format Test
/FAIL/LAD_COUPLE_MATRIX_MICROCRACKING_RATE/1711
142.0, 426.0, 52.0, 2.45, 0.976
1, 1
1711
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1711 in model.fail_ladcouplematrixmicrocrackingrates
    fcmmr = model.fail_ladcouplematrixmicrocrackingrates[1711]
    assert pytest.approx(fcmmr.sigma_cmmr0) == 142.0
    assert pytest.approx(fcmmr.sigma_cmmrc) == 426.0
    assert pytest.approx(fcmmr.gamma_cmmr) == 52.0
    assert pytest.approx(fcmmr.p_cmmr) == 2.45
    assert pytest.approx(fcmmr.d_cmmr_max) == 0.976
    assert fcmmr.fail_id == 1711


def test_m355_fail_lad_couple_matrix_microcracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Matrix Microcracking Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_MATRIX_MICROCRACKING_RATE/1712
122.0, 366.0, 40.0, 2.02, 0.984
1, 1
/FAIL/LAD_CMMR/1713
122.0, 366.0, 40.0, 2.02, 0.984
1, 1
/FAIL/LAD_CMMR_MODEL/1714
122.0, 366.0, 40.0, 2.02, 0.984
1, 1
/FAIL/LAD_CMMR_LAW/1715
122.0, 366.0, 40.0, 2.02, 0.984
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_MATRIX_MICROCRACKING/1716
122.0, 366.0, 40.0, 2.02, 0.984
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1712 in model.fail_ladcouplematrixmicrocrackingrates
    assert 1713 in model.fail_ladcouplematrixmicrocrackingrates
    assert 1714 in model.fail_ladcouplematrixmicrocrackingrates
    assert 1715 in model.fail_ladcouplematrixmicrocrackingrates
    assert 1716 in model.fail_ladcouplematrixmicrocrackingrates
    assert len(model.raw_fails) == 5


def test_m355_fail_lad_couple_matrix_microcracking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_COUPLE_MATRIX_MICROCRACKING_RATE/1717
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m355_eng_flexothermophononmagnonpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00135:>20.6f}{228:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermophononmagnonpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY/1
Flexothermophononmagnonpolaritonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermophononmagnonpolaritonic_resonance_energies
    eng = model.eng_flexothermophononmagnonpolaritonic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftpmpr) == 0.00135
    assert eng.sens_id == 228


def test_m355_eng_flexothermophononmagnonpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermophononmagnonpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY/2
0.00155, 238
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermophononmagnonpolaritonic_resonance_energies
    eng = model.eng_flexothermophononmagnonpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftpmpr) == 0.00155
    assert eng.sens_id == 238


def test_m355_eng_flexothermophononmagnonpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermophononmagnonpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PHONON_MAGNON_POLARITON_RES_WORK/3
0.00105, 174
/ENG/EFLEXOTHERMOPHONONMAGNONPOLARITONICRESONANCE/4
0.00110, 180
/ENG/FLEXOTHERMOPHONONMAGNONPOLARITONIC_RESONANCE_DISSIPATION/5
0.00115, 190
/ENG/EM_FLEXOTHERMOPHONONMAGNONPOLARITONIC_RESONANCE/6
0.00122, 200
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermophononmagnonpolaritonic_resonance_energies
    assert 4 in model.eng_flexothermophononmagnonpolaritonic_resonance_energies
    assert 5 in model.eng_flexothermophononmagnonpolaritonic_resonance_energies
    assert 6 in model.eng_flexothermophononmagnonpolaritonic_resonance_energies


def test_m355_eng_flexothermophononmagnonpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m355_lagmul_sarrus_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{411:>10d}{412:>10d}{413:>10d}{3.65e7:>20.1f}{118:>10d}{4.2e-5:>20.6e}"
    c2 = f"{190.0:>20.4f}{180.0:>20.4f}{160.0:>20.4f}{110.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sarrus Spatial Linkage Joint Fixed Format Test
2022 0
/SARRUS_SPATIAL_LINKAGE_JOINT/375
Sarrus Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 375 in model.lagmul_sarrus_spatial_linkage_joints
    joint = model.lagmul_sarrus_spatial_linkage_joints[375]
    assert joint.node1 == 411
    assert joint.node2 == 412
    assert joint.node3 == 413
    assert pytest.approx(joint.stiff) == 3.65e7
    assert joint.skew_id == 118
    assert pytest.approx(joint.tol) == 4.2e-5
    assert pytest.approx(joint.link_len_a) == 190.0
    assert pytest.approx(joint.link_len_b) == 180.0
    assert pytest.approx(joint.twist_angle_alpha) == 160.0
    assert pytest.approx(joint.offset_distance_s) == 110.0


def test_m355_lagmul_sarrus_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sarrus Spatial Linkage Joint Free Format Test
/LAGMUL/SARRUS_SPATIAL_LINKAGE_JOINT/376
511, 512, 513, 22.0e6, 138, 5.2e-5
192.0, 182.0, 162.0, 112.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 376 in model.lagmul_sarrus_spatial_linkage_joints
    joint = model.lagmul_sarrus_spatial_linkage_joints[376]
    assert joint.node1 == 511
    assert joint.node2 == 512
    assert joint.node3 == 513
    assert pytest.approx(joint.stiff) == 22.0e6
    assert joint.skew_id == 138
    assert pytest.approx(joint.tol) == 5.2e-5
    assert pytest.approx(joint.link_len_a) == 192.0
    assert pytest.approx(joint.link_len_b) == 182.0
    assert pytest.approx(joint.twist_angle_alpha) == 162.0
    assert pytest.approx(joint.offset_distance_s) == 112.0


def test_m355_lagmul_sarrus_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sarrus Spatial Linkage Joint Aliases Test
/LAGMUL/SARRUS_SPATIAL_LINKAGE/377
1, 2, 3, 1e6, 0, 1e-6
45.0, 45.0, 90.0, 30.0
/SARRUS_SPATIAL_LINKAGE/378
1, 2, 3, 1e6, 0, 1e-6
45.0, 45.0, 90.0, 30.0
/SARRUS_SPATIAL_MULTI_LOOP_MECHANISM/379
1, 2, 3, 1e6, 0, 1e-6
45.0, 45.0, 90.0, 30.0
/SARRUS_SPATIAL_RECTILINEAR_MECHANISM/380
1, 2, 3, 1e6, 0, 1e-6
45.0, 45.0, 90.0, 30.0
/SARRUS_SPATIAL_6R_MECHANISM/381
1, 2, 3, 1e6, 0, 1e-6
45.0, 45.0, 90.0, 30.0
/SARRUS_SPATIAL_OVERCONSTRAINED_MECHANISM/382
1, 2, 3, 1e6, 0, 1e-6
45.0, 45.0, 90.0, 30.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 377 in model.lagmul_sarrus_spatial_linkage_joints
    assert 378 in model.lagmul_sarrus_spatial_linkage_joints
    assert 379 in model.lagmul_sarrus_spatial_linkage_joints
    assert 380 in model.lagmul_sarrus_spatial_linkage_joints
    assert 381 in model.lagmul_sarrus_spatial_linkage_joints
    assert 382 in model.lagmul_sarrus_spatial_linkage_joints


def test_m355_lagmul_sarrus_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SARRUS_SPATIAL_LINKAGE_JOINT/383
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m355_sensor_spring_normal_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1155:>10d}{6.25e9:>20.1f}{0.0275:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Normal Pop Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_POP_RATE/408
Spring Normal Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 408 in model.sensor_spring_normal_pop_rates
    sensor = model.sensor_spring_normal_pop_rates[408]
    assert sensor.spring_id == 1155
    assert pytest.approx(sensor.jnorm_pop_max) == 6.25e9
    assert pytest.approx(sensor.t_delay) == 0.0275
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_POP_RATE"


def test_m355_sensor_spring_normal_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Normal Pop Rate Sensor Free Format Test
/SENSOR/SPRING_NORMAL_POP_RATE/409
1156, 6.35e9, 0.0300
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 409 in model.sensor_spring_normal_pop_rates
    sensor = model.sensor_spring_normal_pop_rates[409]
    assert sensor.spring_id == 1156
    assert pytest.approx(sensor.jnorm_pop_max) == 6.35e9
    assert pytest.approx(sensor.t_delay) == 0.0300


def test_m355_sensor_spring_normal_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Normal Pop Rate Sensor Aliases Test
/SENSOR/SPRING_NORM_POP_RATE/410
1157, 6.4e9, 0.0205
/SENSOR/SPRING_RATE_POP_NORM/411
1158, 6.4e9, 0.0205
/SENSOR/NORMAL_POP_RATE_SPRING/412
1159, 6.4e9, 0.0205
/SENSOR/SPRING_POP_NORM/413
1160, 6.4e9, 0.0205
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 410 in model.sensor_spring_normal_pop_rates
    assert 411 in model.sensor_spring_normal_pop_rates
    assert 412 in model.sensor_spring_normal_pop_rates
    assert 413 in model.sensor_spring_normal_pop_rates
    assert len(model.sensors) == 4


def test_m355_sensor_spring_normal_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_NORMAL_POP_RATE/414
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

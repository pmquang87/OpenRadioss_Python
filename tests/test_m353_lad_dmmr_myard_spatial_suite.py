"""Tests for Milestone M353: LadDynamicMatrixMicrocrackingRate Failure Model, EngFlexothermoexcitonphononpolaritonicResonanceEnergy, MyardSpatialLinkageJoint, and SensorSpringBendingSurgeRate."""

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


def test_m353_fail_lad_dynamic_matrix_microcracking_rate_fixed(tmp_path: Path):
    c1 = f"{125.0:>20.4f}{375.0:>20.4f}{42.0:>20.4f}{2.15:>20.4f}{0.995:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1690:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Matrix Microcracking Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_MATRIX_MICROCRACKING_RATE/1690
Ladeveze Dynamic Matrix Microcracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1690 in model.fail_laddynamicmatrixmicrocrackingrates
    fdmmr = model.fail_laddynamicmatrixmicrocrackingrates[1690]
    assert pytest.approx(fdmmr.sigma_dmmr0) == 125.0
    assert pytest.approx(fdmmr.sigma_dmmrc) == 375.0
    assert pytest.approx(fdmmr.gamma_dmmr) == 42.0
    assert pytest.approx(fdmmr.p_dmmr) == 2.15
    assert pytest.approx(fdmmr.d_dmmr_max) == 0.995
    assert fdmmr.ifail_sh == 1
    assert fdmmr.ifail_so == 2
    assert fdmmr.fail_id == 1690
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_MATRIX_MICROCRACKING_RATE"


def test_m353_fail_lad_dynamic_matrix_microcracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Matrix Microcracking Rate Free Format Test
/FAIL/LAD_DYNAMIC_MATRIX_MICROCRACKING_RATE/1691
135.0, 405.0, 48.0, 2.35, 0.975
1, 1
1691
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1691 in model.fail_laddynamicmatrixmicrocrackingrates
    fdmmr = model.fail_laddynamicmatrixmicrocrackingrates[1691]
    assert pytest.approx(fdmmr.sigma_dmmr0) == 135.0
    assert pytest.approx(fdmmr.sigma_dmmrc) == 405.0
    assert pytest.approx(fdmmr.gamma_dmmr) == 48.0
    assert pytest.approx(fdmmr.p_dmmr) == 2.35
    assert pytest.approx(fdmmr.d_dmmr_max) == 0.975
    assert fdmmr.fail_id == 1691


def test_m353_fail_lad_dynamic_matrix_microcracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Matrix Microcracking Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_MATRIX_MICROCRACKING_RATE/1692
115.0, 345.0, 38.0, 1.95, 0.980
1, 1
/FAIL/LAD_DMMR/1693
115.0, 345.0, 38.0, 1.95, 0.980
1, 1
/FAIL/LAD_DMMR_MODEL/1694
115.0, 345.0, 38.0, 1.95, 0.980
1, 1
/FAIL/LAD_DMMR_LAW/1695
115.0, 345.0, 38.0, 1.95, 0.980
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_MATRIX_MICROCRACKING/1696
115.0, 345.0, 38.0, 1.95, 0.980
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1692 in model.fail_laddynamicmatrixmicrocrackingrates
    assert 1693 in model.fail_laddynamicmatrixmicrocrackingrates
    assert 1694 in model.fail_laddynamicmatrixmicrocrackingrates
    assert 1695 in model.fail_laddynamicmatrixmicrocrackingrates
    assert 1696 in model.fail_laddynamicmatrixmicrocrackingrates
    assert len(model.raw_fails) == 5


def test_m353_fail_lad_dynamic_matrix_microcracking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_DYNAMIC_MATRIX_MICROCRACKING_RATE/1697
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m353_eng_flexothermoexcitonphononpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00125:>20.6f}{220:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoexcitonphononpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOEXCITONPHONONPOLARITONIC_RESONANCE_ENERGY/1
Flexothermoexcitonphononpolaritonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoexcitonphononpolaritonic_resonance_energies
    eng = model.eng_flexothermoexcitonphononpolaritonic_resonance_energies[1]
    assert pytest.approx(eng.dt_fteppr) == 0.00125
    assert eng.sens_id == 220


def test_m353_eng_flexothermoexcitonphononpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermoexcitonphononpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOEXCITONPHONONPOLARITONIC_RESONANCE_ENERGY/2
0.00145, 230
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermoexcitonphononpolaritonic_resonance_energies
    eng = model.eng_flexothermoexcitonphononpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_fteppr) == 0.00145
    assert eng.sens_id == 230


def test_m353_eng_flexothermoexcitonphononpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoexcitonphononpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_EXCITON_PHONON_POLARITON_RES_WORK/3
0.00101, 170
/ENG/EFLEXOTHERMOEXCITONPHONONPOLARITONICRESONANCE/4
0.00106, 175
/ENG/FLEXOTHERMOEXCITONPHONONPOLARITONIC_RESONANCE_DISSIPATION/5
0.00110, 185
/ENG/EM_FLEXOTHERMOEXCITONPHONONPOLARITONIC_RESONANCE/6
0.00118, 195
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoexcitonphononpolaritonic_resonance_energies
    assert 4 in model.eng_flexothermoexcitonphononpolaritonic_resonance_energies
    assert 5 in model.eng_flexothermoexcitonphononpolaritonic_resonance_energies
    assert 6 in model.eng_flexothermoexcitonphononpolaritonic_resonance_energies


def test_m353_eng_flexothermoexcitonphononpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOEXCITONPHONONPOLARITONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m353_lagmul_myard_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{391:>10d}{392:>10d}{393:>10d}{3.45e7:>20.1f}{112:>10d}{4.8e-5:>20.6e}"
    c2 = f"{180.0:>20.4f}{170.0:>20.4f}{150.0:>20.4f}{100.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Myard Spatial Linkage Joint Fixed Format Test
2022 0
/MYARD_SPATIAL_LINKAGE_JOINT/355
Myard Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 355 in model.lagmul_myard_spatial_linkage_joints
    joint = model.lagmul_myard_spatial_linkage_joints[355]
    assert joint.node1 == 391
    assert joint.node2 == 392
    assert joint.node3 == 393
    assert pytest.approx(joint.stiff) == 3.45e7
    assert joint.skew_id == 112
    assert pytest.approx(joint.tol) == 4.8e-5
    assert pytest.approx(joint.link_len_a) == 180.0
    assert pytest.approx(joint.link_len_b) == 170.0
    assert pytest.approx(joint.twist_angle_alpha) == 150.0
    assert pytest.approx(joint.offset_distance_s) == 100.0


def test_m353_lagmul_myard_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Myard Spatial Linkage Joint Free Format Test
/LAGMUL/MYARD_SPATIAL_LINKAGE_JOINT/356
491, 492, 493, 20.0e6, 132, 5.8e-5
182.0, 172.0, 152.0, 102.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 356 in model.lagmul_myard_spatial_linkage_joints
    joint = model.lagmul_myard_spatial_linkage_joints[356]
    assert joint.node1 == 491
    assert joint.node2 == 492
    assert joint.node3 == 493
    assert pytest.approx(joint.stiff) == 20.0e6
    assert joint.skew_id == 132
    assert pytest.approx(joint.tol) == 5.8e-5
    assert pytest.approx(joint.link_len_a) == 182.0
    assert pytest.approx(joint.link_len_b) == 172.0
    assert pytest.approx(joint.twist_angle_alpha) == 152.0
    assert pytest.approx(joint.offset_distance_s) == 102.0


def test_m353_lagmul_myard_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Myard Spatial Linkage Joint Aliases Test
/LAGMUL/MYARD_SPATIAL_LINKAGE/357
1, 2, 3, 1e6, 0, 1e-6
40.0, 40.0, 85.0, 25.0
/MYARD_SPATIAL_LINKAGE/358
1, 2, 3, 1e6, 0, 1e-6
40.0, 40.0, 85.0, 25.0
/MYARD_SPATIAL_MULTI_LOOP_MECHANISM/359
1, 2, 3, 1e6, 0, 1e-6
40.0, 40.0, 85.0, 25.0
/MYARD_SPATIAL_PLANE_SYMMETRIC_MECHANISM/360
1, 2, 3, 1e6, 0, 1e-6
40.0, 40.0, 85.0, 25.0
/MYARD_SPATIAL_5R_6R_MECHANISM/361
1, 2, 3, 1e6, 0, 1e-6
40.0, 40.0, 85.0, 25.0
/MYARD_SPATIAL_OVERCONSTRAINED_MECHANISM/362
1, 2, 3, 1e6, 0, 1e-6
40.0, 40.0, 85.0, 25.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 357 in model.lagmul_myard_spatial_linkage_joints
    assert 358 in model.lagmul_myard_spatial_linkage_joints
    assert 359 in model.lagmul_myard_spatial_linkage_joints
    assert 360 in model.lagmul_myard_spatial_linkage_joints
    assert 361 in model.lagmul_myard_spatial_linkage_joints
    assert 362 in model.lagmul_myard_spatial_linkage_joints


def test_m353_lagmul_myard_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/MYARD_SPATIAL_LINKAGE_JOINT/363
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m353_sensor_spring_bending_surge_rate_fixed(tmp_path: Path):
    c1 = f"{1135:>10d}{5.75e9:>20.1f}{0.0255:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Bending Surge Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_SURGE_RATE/388
Spring Bending Surge Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 388 in model.sensor_spring_bending_surge_rates
    sensor = model.sensor_spring_bending_surge_rates[388]
    assert sensor.spring_id == 1135
    assert pytest.approx(sensor.jbend_surge_max) == 5.75e9
    assert pytest.approx(sensor.t_delay) == 0.0255
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_SURGE_RATE"


def test_m353_sensor_spring_bending_surge_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Bending Surge Rate Sensor Free Format Test
/SENSOR/SPRING_BENDING_SURGE_RATE/389
1136, 5.85e9, 0.0280
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 389 in model.sensor_spring_bending_surge_rates
    sensor = model.sensor_spring_bending_surge_rates[389]
    assert sensor.spring_id == 1136
    assert pytest.approx(sensor.jbend_surge_max) == 5.85e9
    assert pytest.approx(sensor.t_delay) == 0.0280


def test_m353_sensor_spring_bending_surge_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Bending Surge Rate Sensor Aliases Test
/SENSOR/SPRING_BEND_SURGE_RATE/390
1137, 5.9e9, 0.0185
/SENSOR/SPRING_RATE_SURGE_BEND/391
1138, 5.9e9, 0.0185
/SENSOR/BENDING_SURGE_RATE_SPRING/392
1139, 5.9e9, 0.0185
/SENSOR/SPRING_SURGE_BEND/393
1140, 5.9e9, 0.0185
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 390 in model.sensor_spring_bending_surge_rates
    assert 391 in model.sensor_spring_bending_surge_rates
    assert 392 in model.sensor_spring_bending_surge_rates
    assert 393 in model.sensor_spring_bending_surge_rates
    assert len(model.sensors) == 4


def test_m353_sensor_spring_bending_surge_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_BENDING_SURGE_RATE/394
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

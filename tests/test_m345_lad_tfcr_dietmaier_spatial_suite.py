"""Tests for Milestone M345: LadTransverseFiberCrushingRate Failure Model, EngFlexothermomagnonicResonanceEnergy, DietmaierSpatialLinkageJoint, and SensorSpringTotalDriftRate."""

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


def test_m345_fail_lad_transverse_fiber_crushing_rate_fixed(tmp_path: Path):
    c1 = f"{238.0:>20.4f}{635.0:>20.4f}{53.5:>20.4f}{2.05:>20.4f}{0.970:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1610:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Fiber Crushing Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_FIBER_CRUSHING_RATE/1610
Ladeveze Transverse Fiber Crushing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1610 in model.fail_ladtransversefibercrushingrates
    ftfcr = model.fail_ladtransversefibercrushingrates[1610]
    assert pytest.approx(ftfcr.sigma_tfcr0) == 238.0
    assert pytest.approx(ftfcr.sigma_tfcrc) == 635.0
    assert pytest.approx(ftfcr.gamma_tfcr) == 53.5
    assert pytest.approx(ftfcr.p_tfcr) == 2.05
    assert pytest.approx(ftfcr.d_tfcr_max) == 0.970
    assert ftfcr.ifail_sh == 1
    assert ftfcr.ifail_so == 2
    assert ftfcr.fail_id == 1610
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_FIBER_CRUSHING_RATE"


def test_m345_fail_lad_transverse_fiber_crushing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Fiber Crushing Rate Free Format Test
/FAIL/LAD_TRANSVERSE_FIBER_CRUSHING_RATE/1611
248.0, 685.0, 58.0, 2.30, 0.950
1, 1
1611
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1611 in model.fail_ladtransversefibercrushingrates
    ftfcr = model.fail_ladtransversefibercrushingrates[1611]
    assert pytest.approx(ftfcr.sigma_tfcr0) == 248.0
    assert pytest.approx(ftfcr.sigma_tfcrc) == 685.0
    assert pytest.approx(ftfcr.gamma_tfcr) == 58.0
    assert pytest.approx(ftfcr.p_tfcr) == 2.30
    assert pytest.approx(ftfcr.d_tfcr_max) == 0.950
    assert ftfcr.fail_id == 1611


def test_m345_fail_lad_transverse_fiber_crushing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Fiber Crushing Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_FIBER_CRUSHING_RATE/1612
195.0, 520.0, 44.0, 1.85, 0.955
1, 1
/FAIL/LAD_TFCR/1613
195.0, 520.0, 44.0, 1.85, 0.955
1, 1
/FAIL/LAD_TFCR_MODEL/1614
195.0, 520.0, 44.0, 1.85, 0.955
1, 1
/FAIL/LAD_TFCR_LAW/1615
195.0, 520.0, 44.0, 1.85, 0.955
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_FIBER_CRUSHING/1616
195.0, 520.0, 44.0, 1.85, 0.955
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1612 in model.fail_ladtransversefibercrushingrates
    assert 1613 in model.fail_ladtransversefibercrushingrates
    assert 1614 in model.fail_ladtransversefibercrushingrates
    assert 1615 in model.fail_ladtransversefibercrushingrates
    assert 1616 in model.fail_ladtransversefibercrushingrates
    assert len(model.raw_fails) == 5


def test_m345_fail_lad_transverse_fiber_crushing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_TRANSVERSE_FIBER_CRUSHING_RATE/1617
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m345_eng_flexothermomagnonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00085:>20.6f}{170:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermomagnonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOMAGNONIC_RESONANCE_ENERGY/1
Flexothermomagnonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermomagnonic_resonance_energies
    eng = model.eng_flexothermomagnonic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftmr) == 0.00085
    assert eng.sens_id == 170


def test_m345_eng_flexothermomagnonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermomagnonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOMAGNONIC_RESONANCE_ENERGY/2
0.00098, 180
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermomagnonic_resonance_energies
    eng = model.eng_flexothermomagnonic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftmr) == 0.00098
    assert eng.sens_id == 180


def test_m345_eng_flexothermomagnonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermomagnonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_MAGNON_RES_WORK/3
0.00068, 134
/ENG/EFLEXOTHERMOMAGNONICRESONANCE/4
0.00072, 140
/ENG/FLEXOTHERMOMAGNONIC_RESONANCE_DISSIPATION/5
0.00078, 150
/ENG/EM_FLEXOTHERMOMAGNONIC_RESONANCE/6
0.00084, 160
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermomagnonic_resonance_energies
    assert 4 in model.eng_flexothermomagnonic_resonance_energies
    assert 5 in model.eng_flexothermomagnonic_resonance_energies
    assert 6 in model.eng_flexothermomagnonic_resonance_energies


def test_m345_eng_flexothermomagnonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOMAGNONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m345_lagmul_dietmaier_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{311:>10d}{312:>10d}{313:>10d}{2.65e7:>20.1f}{86:>10d}{4.8e-5:>20.6e}"
    c2 = f"{144.0:>20.4f}{134.0:>20.4f}{116.0:>20.4f}{72.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Dietmaier Spatial Linkage Joint Fixed Format Test
2022 0
/DIETMAIER_SPATIAL_LINKAGE_JOINT/275
Dietmaier Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 275 in model.lagmul_dietmaier_spatial_linkage_joints
    joint = model.lagmul_dietmaier_spatial_linkage_joints[275]
    assert joint.node1 == 311
    assert joint.node2 == 312
    assert joint.node3 == 313
    assert pytest.approx(joint.stiff) == 2.65e7
    assert joint.skew_id == 86
    assert pytest.approx(joint.tol) == 4.8e-5
    assert pytest.approx(joint.link_len_a) == 144.0
    assert pytest.approx(joint.link_len_b) == 134.0
    assert pytest.approx(joint.twist_angle_alpha) == 116.0
    assert pytest.approx(joint.offset_distance_t) == 72.0


def test_m345_lagmul_dietmaier_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Dietmaier Spatial Linkage Joint Free Format Test
/LAGMUL/DIETMAIER_SPATIAL_LINKAGE_JOINT/276
411, 412, 413, 13.5e6, 106, 5.8e-5
146.0, 136.0, 118.0, 74.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 276 in model.lagmul_dietmaier_spatial_linkage_joints
    joint = model.lagmul_dietmaier_spatial_linkage_joints[276]
    assert joint.node1 == 411
    assert joint.node2 == 412
    assert joint.node3 == 413
    assert pytest.approx(joint.stiff) == 13.5e6
    assert joint.skew_id == 106
    assert pytest.approx(joint.tol) == 5.8e-5
    assert pytest.approx(joint.link_len_a) == 146.0
    assert pytest.approx(joint.link_len_b) == 136.0
    assert pytest.approx(joint.twist_angle_alpha) == 118.0
    assert pytest.approx(joint.offset_distance_t) == 74.0


def test_m345_lagmul_dietmaier_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Dietmaier Spatial Linkage Joint Aliases Test
/LAGMUL/DIETMAIER_SPATIAL_LINKAGE/277
1, 2, 3, 1e6, 0, 1e-6
22.0, 22.0, 55.0, 10.0
/DIETMAIER_SPATIAL_LINKAGE/278
1, 2, 3, 1e6, 0, 1e-6
22.0, 22.0, 55.0, 10.0
/DIETMAIER_SPATIAL_VARIABLE_GEOMETRY_MECHANISM/279
1, 2, 3, 1e6, 0, 1e-6
22.0, 22.0, 55.0, 10.0
/DIETMAIER_SPATIAL_MULTI_LOOP_MECHANISM/280
1, 2, 3, 1e6, 0, 1e-6
22.0, 22.0, 55.0, 10.0
/DIETMAIER_SPATIAL_6R_MECHANISM/281
1, 2, 3, 1e6, 0, 1e-6
22.0, 22.0, 55.0, 10.0
/DIETMAIER_SPATIAL_OVERCONSTRAINED_MECHANISM/282
1, 2, 3, 1e6, 0, 1e-6
22.0, 22.0, 55.0, 10.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 277 in model.lagmul_dietmaier_spatial_linkage_joints
    assert 278 in model.lagmul_dietmaier_spatial_linkage_joints
    assert 279 in model.lagmul_dietmaier_spatial_linkage_joints
    assert 280 in model.lagmul_dietmaier_spatial_linkage_joints
    assert 281 in model.lagmul_dietmaier_spatial_linkage_joints
    assert 282 in model.lagmul_dietmaier_spatial_linkage_joints


def test_m345_lagmul_dietmaier_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/DIETMAIER_SPATIAL_LINKAGE_JOINT/283
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m345_sensor_spring_total_drift_rate_fixed(tmp_path: Path):
    c1 = f"{1055:>10d}{4.95e9:>20.1f}{0.0175:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Total Drift Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_DRIFT_RATE/308
Spring Total Drift Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 308 in model.sensor_spring_total_drift_rates
    sensor = model.sensor_spring_total_drift_rates[308]
    assert sensor.spring_id == 1055
    assert pytest.approx(sensor.jtot_drift_max) == 4.95e9
    assert pytest.approx(sensor.t_delay) == 0.0175
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_DRIFT_RATE"


def test_m345_sensor_spring_total_drift_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Total Drift Rate Sensor Free Format Test
/SENSOR/SPRING_TOTAL_DRIFT_RATE/309
1056, 5.05e9, 0.0200
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 309 in model.sensor_spring_total_drift_rates
    sensor = model.sensor_spring_total_drift_rates[309]
    assert sensor.spring_id == 1056
    assert pytest.approx(sensor.jtot_drift_max) == 5.05e9
    assert pytest.approx(sensor.t_delay) == 0.0200


def test_m345_sensor_spring_total_drift_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Total Drift Rate Sensor Aliases Test
/SENSOR/SPRING_TOT_DRIFT_RATE/310
1057, 5.1e9, 0.0105
/SENSOR/SPRING_RATE_DRIFT_TOT/311
1058, 5.1e9, 0.0105
/SENSOR/TOTAL_DRIFT_RATE_SPRING/312
1059, 5.1e9, 0.0105
/SENSOR/SPRING_DRIFT_TOT/313
1060, 5.1e9, 0.0105
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 310 in model.sensor_spring_total_drift_rates
    assert 311 in model.sensor_spring_total_drift_rates
    assert 312 in model.sensor_spring_total_drift_rates
    assert 313 in model.sensor_spring_total_drift_rates
    assert len(model.sensors) == 4


def test_m345_sensor_spring_total_drift_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TOTAL_DRIFT_RATE/314
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

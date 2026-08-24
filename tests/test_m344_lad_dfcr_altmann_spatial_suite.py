"""Tests for Milestone M344: LadDynamicFiberCrushingRate Failure Model, EngFlexothermoexcitonicResonanceEnergy, AltmannSpatialLinkageJoint, and SensorSpringTransverseDriftRate."""

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


def test_m344_fail_lad_dynamic_fiber_crushing_rate_fixed(tmp_path: Path):
    c1 = f"{232.0:>20.4f}{625.0:>20.4f}{52.5:>20.4f}{2.02:>20.4f}{0.965:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1600:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Fiber Crushing Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_FIBER_CRUSHING_RATE/1600
Ladeveze Dynamic Fiber Crushing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1600 in model.fail_laddynamicfibercrushingrates
    fdfcr = model.fail_laddynamicfibercrushingrates[1600]
    assert pytest.approx(fdfcr.sigma_dfcr0) == 232.0
    assert pytest.approx(fdfcr.sigma_dfcrc) == 625.0
    assert pytest.approx(fdfcr.gamma_dfcr) == 52.5
    assert pytest.approx(fdfcr.p_dfcr) == 2.02
    assert pytest.approx(fdfcr.d_dfcr_max) == 0.965
    assert fdfcr.ifail_sh == 1
    assert fdfcr.ifail_so == 2
    assert fdfcr.fail_id == 1600
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_FIBER_CRUSHING_RATE"


def test_m344_fail_lad_dynamic_fiber_crushing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Fiber Crushing Rate Free Format Test
/FAIL/LAD_DYNAMIC_FIBER_CRUSHING_RATE/1601
242.0, 675.0, 56.0, 2.25, 0.945
1, 1
1601
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1601 in model.fail_laddynamicfibercrushingrates
    fdfcr = model.fail_laddynamicfibercrushingrates[1601]
    assert pytest.approx(fdfcr.sigma_dfcr0) == 242.0
    assert pytest.approx(fdfcr.sigma_dfcrc) == 675.0
    assert pytest.approx(fdfcr.gamma_dfcr) == 56.0
    assert pytest.approx(fdfcr.p_dfcr) == 2.25
    assert pytest.approx(fdfcr.d_dfcr_max) == 0.945
    assert fdfcr.fail_id == 1601


def test_m344_fail_lad_dynamic_fiber_crushing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Fiber Crushing Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_FIBER_CRUSHING_RATE/1602
190.0, 510.0, 42.0, 1.80, 0.950
1, 1
/FAIL/LAD_DFCR/1603
190.0, 510.0, 42.0, 1.80, 0.950
1, 1
/FAIL/LAD_DFCR_MODEL/1604
190.0, 510.0, 42.0, 1.80, 0.950
1, 1
/FAIL/LAD_DFCR_LAW/1605
190.0, 510.0, 42.0, 1.80, 0.950
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_FIBER_CRUSHING/1606
190.0, 510.0, 42.0, 1.80, 0.950
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1602 in model.fail_laddynamicfibercrushingrates
    assert 1603 in model.fail_laddynamicfibercrushingrates
    assert 1604 in model.fail_laddynamicfibercrushingrates
    assert 1605 in model.fail_laddynamicfibercrushingrates
    assert 1606 in model.fail_laddynamicfibercrushingrates
    assert len(model.raw_fails) == 5


def test_m344_fail_lad_dynamic_fiber_crushing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_DYNAMIC_FIBER_CRUSHING_RATE/1607
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m344_eng_flexothermoexcitonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00082:>20.6f}{165:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoexcitonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOEXCITONIC_RESONANCE_ENERGY/1
Flexothermoexcitonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoexcitonic_resonance_energies
    eng = model.eng_flexothermoexcitonic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftexr) == 0.00082
    assert eng.sens_id == 165


def test_m344_eng_flexothermoexcitonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermoexcitonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOEXCITONIC_RESONANCE_ENERGY/2
0.00096, 175
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermoexcitonic_resonance_energies
    eng = model.eng_flexothermoexcitonic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftexr) == 0.00096
    assert eng.sens_id == 175


def test_m344_eng_flexothermoexcitonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoexcitonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_EXCITON_RES_WORK/3
0.00065, 130
/ENG/EFLEXOTHERMOEXCITONICRESONANCE/4
0.00068, 136
/ENG/FLEXOTHERMOEXCITONIC_RESONANCE_DISSIPATION/5
0.00075, 146
/ENG/EM_FLEXOTHERMOEXCITONIC_RESONANCE/6
0.00080, 156
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoexcitonic_resonance_energies
    assert 4 in model.eng_flexothermoexcitonic_resonance_energies
    assert 5 in model.eng_flexothermoexcitonic_resonance_energies
    assert 6 in model.eng_flexothermoexcitonic_resonance_energies


def test_m344_eng_flexothermoexcitonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOEXCITONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m344_lagmul_altmann_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{301:>10d}{302:>10d}{303:>10d}{2.55e7:>20.1f}{82:>10d}{4.5e-5:>20.6e}"
    c2 = f"{140.0:>20.4f}{130.0:>20.4f}{112.0:>20.4f}{68.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Altmann Spatial Linkage Joint Fixed Format Test
2022 0
/ALTMANN_SPATIAL_LINKAGE_JOINT/265
Altmann Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 265 in model.lagmul_altmann_spatial_linkage_joints
    joint = model.lagmul_altmann_spatial_linkage_joints[265]
    assert joint.node1 == 301
    assert joint.node2 == 302
    assert joint.node3 == 303
    assert pytest.approx(joint.stiff) == 2.55e7
    assert joint.skew_id == 82
    assert pytest.approx(joint.tol) == 4.5e-5
    assert pytest.approx(joint.link_len_a) == 140.0
    assert pytest.approx(joint.link_len_b) == 130.0
    assert pytest.approx(joint.twist_angle_alpha) == 112.0
    assert pytest.approx(joint.offset_distance_s) == 68.0


def test_m344_lagmul_altmann_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Altmann Spatial Linkage Joint Free Format Test
/LAGMUL/ALTMANN_SPATIAL_LINKAGE_JOINT/266
401, 402, 403, 13.0e6, 102, 5.5e-5
142.0, 132.0, 115.0, 70.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 266 in model.lagmul_altmann_spatial_linkage_joints
    joint = model.lagmul_altmann_spatial_linkage_joints[266]
    assert joint.node1 == 401
    assert joint.node2 == 402
    assert joint.node3 == 403
    assert pytest.approx(joint.stiff) == 13.0e6
    assert joint.skew_id == 102
    assert pytest.approx(joint.tol) == 5.5e-5
    assert pytest.approx(joint.link_len_a) == 142.0
    assert pytest.approx(joint.link_len_b) == 132.0
    assert pytest.approx(joint.twist_angle_alpha) == 115.0
    assert pytest.approx(joint.offset_distance_s) == 70.0


def test_m344_lagmul_altmann_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Altmann Spatial Linkage Joint Aliases Test
/LAGMUL/ALTMANN_SPATIAL_LINKAGE/267
1, 2, 3, 1e6, 0, 1e-6
20.0, 20.0, 50.0, 9.0
/ALTMANN_SPATIAL_LINKAGE/268
1, 2, 3, 1e6, 0, 1e-6
20.0, 20.0, 50.0, 9.0
/ALTMANN_SPATIAL_MULTI_LOOP_MECHANISM/269
1, 2, 3, 1e6, 0, 1e-6
20.0, 20.0, 50.0, 9.0
/ALTMANN_SPATIAL_NONSPHERICAL_MECHANISM/270
1, 2, 3, 1e6, 0, 1e-6
20.0, 20.0, 50.0, 9.0
/ALTMANN_SPATIAL_6R_MECHANISM/271
1, 2, 3, 1e6, 0, 1e-6
20.0, 20.0, 50.0, 9.0
/ALTMANN_SPATIAL_OVERCONSTRAINED_MECHANISM/272
1, 2, 3, 1e6, 0, 1e-6
20.0, 20.0, 50.0, 9.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 267 in model.lagmul_altmann_spatial_linkage_joints
    assert 268 in model.lagmul_altmann_spatial_linkage_joints
    assert 269 in model.lagmul_altmann_spatial_linkage_joints
    assert 270 in model.lagmul_altmann_spatial_linkage_joints
    assert 271 in model.lagmul_altmann_spatial_linkage_joints
    assert 272 in model.lagmul_altmann_spatial_linkage_joints


def test_m344_lagmul_altmann_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ALTMANN_SPATIAL_LINKAGE_JOINT/272
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m344_sensor_spring_transverse_drift_rate_fixed(tmp_path: Path):
    c1 = f"{1045:>10d}{4.85e9:>20.1f}{0.0165:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Transverse Drift Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_DRIFT_RATE/298
Spring Transverse Drift Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 298 in model.sensor_spring_transverse_drift_rates
    sensor = model.sensor_spring_transverse_drift_rates[298]
    assert sensor.spring_id == 1045
    assert pytest.approx(sensor.jtrans_drift_max) == 4.85e9
    assert pytest.approx(sensor.t_delay) == 0.0165
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_DRIFT_RATE"


def test_m344_sensor_spring_transverse_drift_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Transverse Drift Rate Sensor Free Format Test
/SENSOR/SPRING_TRANSVERSE_DRIFT_RATE/299
1046, 4.95e9, 0.0190
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 299 in model.sensor_spring_transverse_drift_rates
    sensor = model.sensor_spring_transverse_drift_rates[299]
    assert sensor.spring_id == 1046
    assert pytest.approx(sensor.jtrans_drift_max) == 4.95e9
    assert pytest.approx(sensor.t_delay) == 0.0190


def test_m344_sensor_spring_transverse_drift_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Transverse Drift Rate Sensor Aliases Test
/SENSOR/SPRING_TRANS_DRIFT_RATE/300
1047, 5.0e9, 0.0095
/SENSOR/SPRING_RATE_DRIFT_TRANS/301
1048, 5.0e9, 0.0095
/SENSOR/TRANSVERSE_DRIFT_RATE_SPRING/302
1049, 5.0e9, 0.0095
/SENSOR/SPRING_DRIFT_TRANS/303
1050, 5.0e9, 0.0095
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 300 in model.sensor_spring_transverse_drift_rates
    assert 301 in model.sensor_spring_transverse_drift_rates
    assert 302 in model.sensor_spring_transverse_drift_rates
    assert 303 in model.sensor_spring_transverse_drift_rates
    assert len(model.sensors) == 4


def test_m344_sensor_spring_transverse_drift_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TRANSVERSE_DRIFT_RATE/304
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

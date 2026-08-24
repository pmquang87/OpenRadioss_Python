"""Tests for Milestone M346: LadCoupleFiberCrushingRate Failure Model, EngFlexothermomagnonpolaritonicResonanceEnergy, WohlhartSpatialLinkageJoint, and SensorSpringTorsionalDriftRate."""

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


def test_m346_fail_lad_couple_fiber_crushing_rate_fixed(tmp_path: Path):
    c1 = f"{245.0:>20.4f}{645.0:>20.4f}{55.0:>20.4f}{2.10:>20.4f}{0.975:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1620:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Fiber Crushing Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_FIBER_CRUSHING_RATE/1620
Ladeveze Coupled Fiber Crushing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1620 in model.fail_ladcouplefibercrushingrates
    fcfcr = model.fail_ladcouplefibercrushingrates[1620]
    assert pytest.approx(fcfcr.sigma_cfcr0) == 245.0
    assert pytest.approx(fcfcr.sigma_cfcrc) == 645.0
    assert pytest.approx(fcfcr.gamma_cfcr) == 55.0
    assert pytest.approx(fcfcr.p_cfcr) == 2.10
    assert pytest.approx(fcfcr.d_cfcr_max) == 0.975
    assert fcfcr.ifail_sh == 1
    assert fcfcr.ifail_so == 2
    assert fcfcr.fail_id == 1620
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_FIBER_CRUSHING_RATE"


def test_m346_fail_lad_couple_fiber_crushing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Fiber Crushing Rate Free Format Test
/FAIL/LAD_COUPLE_FIBER_CRUSHING_RATE/1621
255.0, 695.0, 60.0, 2.35, 0.955
1, 1
1621
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1621 in model.fail_ladcouplefibercrushingrates
    fcfcr = model.fail_ladcouplefibercrushingrates[1621]
    assert pytest.approx(fcfcr.sigma_cfcr0) == 255.0
    assert pytest.approx(fcfcr.sigma_cfcrc) == 695.0
    assert pytest.approx(fcfcr.gamma_cfcr) == 60.0
    assert pytest.approx(fcfcr.p_cfcr) == 2.35
    assert pytest.approx(fcfcr.d_cfcr_max) == 0.955
    assert fcfcr.fail_id == 1621


def test_m346_fail_lad_couple_fiber_crushing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Fiber Crushing Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_FIBER_CRUSHING_RATE/1622
200.0, 530.0, 46.0, 1.90, 0.960
1, 1
/FAIL/LAD_CFCR/1623
200.0, 530.0, 46.0, 1.90, 0.960
1, 1
/FAIL/LAD_CFCR_MODEL/1624
200.0, 530.0, 46.0, 1.90, 0.960
1, 1
/FAIL/LAD_CFCR_LAW/1625
200.0, 530.0, 46.0, 1.90, 0.960
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_FIBER_CRUSHING/1626
200.0, 530.0, 46.0, 1.90, 0.960
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1622 in model.fail_ladcouplefibercrushingrates
    assert 1623 in model.fail_ladcouplefibercrushingrates
    assert 1624 in model.fail_ladcouplefibercrushingrates
    assert 1625 in model.fail_ladcouplefibercrushingrates
    assert 1626 in model.fail_ladcouplefibercrushingrates
    assert len(model.raw_fails) == 5


def test_m346_fail_lad_couple_fiber_crushing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_COUPLE_FIBER_CRUSHING_RATE/1627
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m346_eng_flexothermomagnonpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00088:>20.6f}{175:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermomagnonpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOMAGNONPOLARITONIC_RESONANCE_ENERGY/1
Flexothermomagnonpolaritonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermomagnonpolaritonic_resonance_energies
    eng = model.eng_flexothermomagnonpolaritonic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftmpr) == 0.00088
    assert eng.sens_id == 175


def test_m346_eng_flexothermomagnonpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermomagnonpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOMAGNONPOLARITONIC_RESONANCE_ENERGY/2
0.00102, 185
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermomagnonpolaritonic_resonance_energies
    eng = model.eng_flexothermomagnonpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftmpr) == 0.00102
    assert eng.sens_id == 185


def test_m346_eng_flexothermomagnonpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermomagnonpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_MAGNON_POLARITON_RES_WORK/3
0.00070, 138
/ENG/EFLEXOTHERMOMAGNONPOLARITONICRESONANCE/4
0.00075, 144
/ENG/FLEXOTHERMOMAGNONPOLARITONIC_RESONANCE_DISSIPATION/5
0.00082, 154
/ENG/EM_FLEXOTHERMOMAGNONPOLARITONIC_RESONANCE/6
0.00088, 164
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermomagnonpolaritonic_resonance_energies
    assert 4 in model.eng_flexothermomagnonpolaritonic_resonance_energies
    assert 5 in model.eng_flexothermomagnonpolaritonic_resonance_energies
    assert 6 in model.eng_flexothermomagnonpolaritonic_resonance_energies


def test_m346_eng_flexothermomagnonpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOMAGNONPOLARITONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m346_lagmul_wohlhart_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{321:>10d}{322:>10d}{323:>10d}{2.75e7:>20.1f}{90:>10d}{5.0e-5:>20.6e}"
    c2 = f"{148.0:>20.4f}{138.0:>20.4f}{120.0:>20.4f}{76.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Wohlhart Spatial Linkage Joint Fixed Format Test
2022 0
/WOHLHART_SPATIAL_LINKAGE_JOINT/285
Wohlhart Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 285 in model.lagmul_wohlhart_spatial_linkage_joints
    joint = model.lagmul_wohlhart_spatial_linkage_joints[285]
    assert joint.node1 == 321
    assert joint.node2 == 322
    assert joint.node3 == 323
    assert pytest.approx(joint.stiff) == 2.75e7
    assert joint.skew_id == 90
    assert pytest.approx(joint.tol) == 5.0e-5
    assert pytest.approx(joint.link_len_a) == 148.0
    assert pytest.approx(joint.link_len_b) == 138.0
    assert pytest.approx(joint.twist_angle_alpha) == 120.0
    assert pytest.approx(joint.offset_distance_u) == 76.0


def test_m346_lagmul_wohlhart_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Wohlhart Spatial Linkage Joint Free Format Test
/LAGMUL/WOHLHART_SPATIAL_LINKAGE_JOINT/286
421, 422, 423, 14.0e6, 110, 6.0e-5
150.0, 140.0, 122.0, 78.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 286 in model.lagmul_wohlhart_spatial_linkage_joints
    joint = model.lagmul_wohlhart_spatial_linkage_joints[286]
    assert joint.node1 == 421
    assert joint.node2 == 422
    assert joint.node3 == 423
    assert pytest.approx(joint.stiff) == 14.0e6
    assert joint.skew_id == 110
    assert pytest.approx(joint.tol) == 6.0e-5
    assert pytest.approx(joint.link_len_a) == 150.0
    assert pytest.approx(joint.link_len_b) == 140.0
    assert pytest.approx(joint.twist_angle_alpha) == 122.0
    assert pytest.approx(joint.offset_distance_u) == 78.0


def test_m346_lagmul_wohlhart_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Wohlhart Spatial Linkage Joint Aliases Test
/LAGMUL/WOHLHART_SPATIAL_LINKAGE/287
1, 2, 3, 1e6, 0, 1e-6
25.0, 25.0, 60.0, 12.0
/WOHLHART_SPATIAL_LINKAGE/288
1, 2, 3, 1e6, 0, 1e-6
25.0, 25.0, 60.0, 12.0
/WOHLHART_SPATIAL_MULTI_LOOP_MECHANISM/289
1, 2, 3, 1e6, 0, 1e-6
25.0, 25.0, 60.0, 12.0
/WOHLHART_SPATIAL_SKEW_SYMMETRIC_MECHANISM/290
1, 2, 3, 1e6, 0, 1e-6
25.0, 25.0, 60.0, 12.0
/WOHLHART_SPATIAL_6R_MECHANISM/291
1, 2, 3, 1e6, 0, 1e-6
25.0, 25.0, 60.0, 12.0
/WOHLHART_SPATIAL_OVERCONSTRAINED_MECHANISM/292
1, 2, 3, 1e6, 0, 1e-6
25.0, 25.0, 60.0, 12.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 287 in model.lagmul_wohlhart_spatial_linkage_joints
    assert 288 in model.lagmul_wohlhart_spatial_linkage_joints
    assert 289 in model.lagmul_wohlhart_spatial_linkage_joints
    assert 290 in model.lagmul_wohlhart_spatial_linkage_joints
    assert 291 in model.lagmul_wohlhart_spatial_linkage_joints
    assert 292 in model.lagmul_wohlhart_spatial_linkage_joints


def test_m346_lagmul_wohlhart_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/WOHLHART_SPATIAL_LINKAGE_JOINT/293
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m346_sensor_spring_torsional_drift_rate_fixed(tmp_path: Path):
    c1 = f"{1065:>10d}{5.05e9:>20.1f}{0.0185:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Torsional Drift Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_DRIFT_RATE/318
Spring Torsional Drift Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 318 in model.sensor_spring_torsional_drift_rates
    sensor = model.sensor_spring_torsional_drift_rates[318]
    assert sensor.spring_id == 1065
    assert pytest.approx(sensor.jtors_drift_max) == 5.05e9
    assert pytest.approx(sensor.t_delay) == 0.0185
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_DRIFT_RATE"


def test_m346_sensor_spring_torsional_drift_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Torsional Drift Rate Sensor Free Format Test
/SENSOR/SPRING_TORSIONAL_DRIFT_RATE/319
1066, 5.15e9, 0.0210
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 319 in model.sensor_spring_torsional_drift_rates
    sensor = model.sensor_spring_torsional_drift_rates[319]
    assert sensor.spring_id == 1066
    assert pytest.approx(sensor.jtors_drift_max) == 5.15e9
    assert pytest.approx(sensor.t_delay) == 0.0210


def test_m346_sensor_spring_torsional_drift_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Torsional Drift Rate Sensor Aliases Test
/SENSOR/SPRING_TORS_DRIFT_RATE/320
1067, 5.2e9, 0.0115
/SENSOR/SPRING_RATE_DRIFT_TORS/321
1068, 5.2e9, 0.0115
/SENSOR/TORSIONAL_DRIFT_RATE_SPRING/322
1069, 5.2e9, 0.0115
/SENSOR/SPRING_DRIFT_TORS/323
1070, 5.2e9, 0.0115
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 320 in model.sensor_spring_torsional_drift_rates
    assert 321 in model.sensor_spring_torsional_drift_rates
    assert 322 in model.sensor_spring_torsional_drift_rates
    assert 323 in model.sensor_spring_torsional_drift_rates
    assert len(model.sensors) == 4


def test_m346_sensor_spring_torsional_drift_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TORSIONAL_DRIFT_RATE/324
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

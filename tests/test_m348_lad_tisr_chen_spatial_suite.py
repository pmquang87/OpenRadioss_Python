"""Tests for Milestone M348: LadTransverseInterlaminarShearRate Failure Model, EngFlexothermoexcitonpolaritonicResonanceEnergy, ChenSpatialLinkageJoint, and SensorSpringTotalAngularDriftRate."""

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


def test_m348_fail_lad_transverse_interlaminar_shear_rate_fixed(tmp_path: Path):
    c1 = f"{140.0:>20.4f}{395.0:>20.4f}{44.0:>20.4f}{2.05:>20.4f}{0.970:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1640:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Interlaminar Shear Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_SHEAR_RATE/1640
Ladeveze Transverse Interlaminar Shear Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1640 in model.fail_ladtransverseinterlaminarshearrates
    ftisr = model.fail_ladtransverseinterlaminarshearrates[1640]
    assert pytest.approx(ftisr.tau_tisr0) == 140.0
    assert pytest.approx(ftisr.tau_tisrc) == 395.0
    assert pytest.approx(ftisr.gamma_tisr) == 44.0
    assert pytest.approx(ftisr.p_tisr) == 2.05
    assert pytest.approx(ftisr.d_tisr_max) == 0.970
    assert ftisr.ifail_sh == 1
    assert ftisr.ifail_so == 2
    assert ftisr.fail_id == 1640
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_INTERLAMINAR_SHEAR_RATE"


def test_m348_fail_lad_transverse_interlaminar_shear_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Interlaminar Shear Rate Free Format Test
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_SHEAR_RATE/1641
150.0, 425.0, 50.0, 2.25, 0.950
1, 1
1641
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1641 in model.fail_ladtransverseinterlaminarshearrates
    ftisr = model.fail_ladtransverseinterlaminarshearrates[1641]
    assert pytest.approx(ftisr.tau_tisr0) == 150.0
    assert pytest.approx(ftisr.tau_tisrc) == 425.0
    assert pytest.approx(ftisr.gamma_tisr) == 50.0
    assert pytest.approx(ftisr.p_tisr) == 2.25
    assert pytest.approx(ftisr.d_tisr_max) == 0.950
    assert ftisr.fail_id == 1641


def test_m348_fail_lad_transverse_interlaminar_shear_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Interlaminar Shear Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_SHEAR_RATE/1642
125.0, 350.0, 38.0, 1.80, 0.955
1, 1
/FAIL/LAD_TISR/1643
125.0, 350.0, 38.0, 1.80, 0.955
1, 1
/FAIL/LAD_TISR_MODEL/1644
125.0, 350.0, 38.0, 1.80, 0.955
1, 1
/FAIL/LAD_TISR_LAW/1645
125.0, 350.0, 38.0, 1.80, 0.955
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_INTERLAMINAR_SHEAR/1646
125.0, 350.0, 38.0, 1.80, 0.955
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1642 in model.fail_ladtransverseinterlaminarshearrates
    assert 1643 in model.fail_ladtransverseinterlaminarshearrates
    assert 1644 in model.fail_ladtransverseinterlaminarshearrates
    assert 1645 in model.fail_ladtransverseinterlaminarshearrates
    assert 1646 in model.fail_ladtransverseinterlaminarshearrates
    assert len(model.raw_fails) == 5


def test_m348_fail_lad_transverse_interlaminar_shear_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_SHEAR_RATE/1647
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m348_eng_flexothermoexcitonpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00095:>20.6f}{188:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoexcitonpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOEXCITONPOLARITONIC_RESONANCE_ENERGY/1
Flexothermoexcitonpolaritonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoexcitonpolaritonic_resonance_energies
    eng = model.eng_flexothermoexcitonpolaritonic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftepr) == 0.00095
    assert eng.sens_id == 188


def test_m348_eng_flexothermoexcitonpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermoexcitonpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOEXCITONPOLARITONIC_RESONANCE_ENERGY/2
0.00112, 198
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermoexcitonpolaritonic_resonance_energies
    eng = model.eng_flexothermoexcitonpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftepr) == 0.00112
    assert eng.sens_id == 198


def test_m348_eng_flexothermoexcitonpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoexcitonpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_EXCITON_POLARITON_RES_WORK/3
0.00078, 146
/ENG/EFLEXOTHERMOEXCITONPOLARITONICRESONANCE/4
0.00083, 152
/ENG/FLEXOTHERMOEXCITONPOLARITONIC_RESONANCE_DISSIPATION/5
0.00090, 162
/ENG/EM_FLEXOTHERMOEXCITONPOLARITONIC_RESONANCE/6
0.00096, 172
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoexcitonpolaritonic_resonance_energies
    assert 4 in model.eng_flexothermoexcitonpolaritonic_resonance_energies
    assert 5 in model.eng_flexothermoexcitonpolaritonic_resonance_energies
    assert 6 in model.eng_flexothermoexcitonpolaritonic_resonance_energies


def test_m348_eng_flexothermoexcitonpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOEXCITONPOLARITONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m348_lagmul_chen_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{341:>10d}{342:>10d}{343:>10d}{2.95e7:>20.1f}{98:>10d}{5.5e-5:>20.6e}"
    c2 = f"{156.0:>20.4f}{146.0:>20.4f}{128.0:>20.4f}{84.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Chen Spatial Linkage Joint Fixed Format Test
2022 0
/CHEN_SPATIAL_LINKAGE_JOINT/305
Chen Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 305 in model.lagmul_chen_spatial_linkage_joints
    joint = model.lagmul_chen_spatial_linkage_joints[305]
    assert joint.node1 == 341
    assert joint.node2 == 342
    assert joint.node3 == 343
    assert pytest.approx(joint.stiff) == 2.95e7
    assert joint.skew_id == 98
    assert pytest.approx(joint.tol) == 5.5e-5
    assert pytest.approx(joint.link_len_a) == 156.0
    assert pytest.approx(joint.link_len_b) == 146.0
    assert pytest.approx(joint.twist_angle_alpha) == 128.0
    assert pytest.approx(joint.offset_distance_w) == 84.0


def test_m348_lagmul_chen_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Chen Spatial Linkage Joint Free Format Test
/LAGMUL/CHEN_SPATIAL_LINKAGE_JOINT/306
441, 442, 443, 15.0e6, 118, 6.5e-5
158.0, 148.0, 130.0, 86.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 306 in model.lagmul_chen_spatial_linkage_joints
    joint = model.lagmul_chen_spatial_linkage_joints[306]
    assert joint.node1 == 441
    assert joint.node2 == 442
    assert joint.node3 == 443
    assert pytest.approx(joint.stiff) == 15.0e6
    assert joint.skew_id == 118
    assert pytest.approx(joint.tol) == 6.5e-5
    assert pytest.approx(joint.link_len_a) == 158.0
    assert pytest.approx(joint.link_len_b) == 148.0
    assert pytest.approx(joint.twist_angle_alpha) == 130.0
    assert pytest.approx(joint.offset_distance_w) == 86.0


def test_m348_lagmul_chen_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Chen Spatial Linkage Joint Aliases Test
/LAGMUL/CHEN_SPATIAL_LINKAGE/307
1, 2, 3, 1e6, 0, 1e-6
30.0, 30.0, 70.0, 16.0
/CHEN_SPATIAL_LINKAGE/308
1, 2, 3, 1e6, 0, 1e-6
30.0, 30.0, 70.0, 16.0
/CHEN_SPATIAL_MULTI_LOOP_MECHANISM/309
1, 2, 3, 1e6, 0, 1e-6
30.0, 30.0, 70.0, 16.0
/CHEN_SPATIAL_LARGE_DISPLACEMENT_MECHANISM/310
1, 2, 3, 1e6, 0, 1e-6
30.0, 30.0, 70.0, 16.0
/CHEN_SPATIAL_6R_MECHANISM/311
1, 2, 3, 1e6, 0, 1e-6
30.0, 30.0, 70.0, 16.0
/CHEN_SPATIAL_OVERCONSTRAINED_MECHANISM/312
1, 2, 3, 1e6, 0, 1e-6
30.0, 30.0, 70.0, 16.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 307 in model.lagmul_chen_spatial_linkage_joints
    assert 308 in model.lagmul_chen_spatial_linkage_joints
    assert 309 in model.lagmul_chen_spatial_linkage_joints
    assert 310 in model.lagmul_chen_spatial_linkage_joints
    assert 311 in model.lagmul_chen_spatial_linkage_joints
    assert 312 in model.lagmul_chen_spatial_linkage_joints


def test_m348_lagmul_chen_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/CHEN_SPATIAL_LINKAGE_JOINT/313
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m348_sensor_spring_total_angular_drift_rate_fixed(tmp_path: Path):
    c1 = f"{1085:>10d}{5.25e9:>20.1f}{0.0205:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Total Angular Drift Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_DRIFT_RATE/338
Spring Total Angular Drift Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 338 in model.sensor_spring_total_angular_drift_rates
    sensor = model.sensor_spring_total_angular_drift_rates[338]
    assert sensor.spring_id == 1085
    assert pytest.approx(sensor.jtot_ang_drift_max) == 5.25e9
    assert pytest.approx(sensor.t_delay) == 0.0205
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_DRIFT_RATE"


def test_m348_sensor_spring_total_angular_drift_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Total Angular Drift Rate Sensor Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_DRIFT_RATE/339
1086, 5.35e9, 0.0230
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 339 in model.sensor_spring_total_angular_drift_rates
    sensor = model.sensor_spring_total_angular_drift_rates[339]
    assert sensor.spring_id == 1086
    assert pytest.approx(sensor.jtot_ang_drift_max) == 5.35e9
    assert pytest.approx(sensor.t_delay) == 0.0230


def test_m348_sensor_spring_total_angular_drift_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Total Angular Drift Rate Sensor Aliases Test
/SENSOR/SPRING_TOT_ANG_DRIFT_RATE/340
1087, 5.4e9, 0.0135
/SENSOR/SPRING_RATE_DRIFT_ANG_TOT/341
1088, 5.4e9, 0.0135
/SENSOR/TOTAL_ANGULAR_DRIFT_RATE_SPRING/342
1089, 5.4e9, 0.0135
/SENSOR/SPRING_DRIFT_ANG_TOT/343
1090, 5.4e9, 0.0135
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 340 in model.sensor_spring_total_angular_drift_rates
    assert 341 in model.sensor_spring_total_angular_drift_rates
    assert 342 in model.sensor_spring_total_angular_drift_rates
    assert 343 in model.sensor_spring_total_angular_drift_rates
    assert len(model.sensors) == 4


def test_m348_sensor_spring_total_angular_drift_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TOTAL_ANGULAR_DRIFT_RATE/344
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

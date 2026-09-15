"""Tests for Milestone M347: LadDynamicInterlaminarShearRate Failure Model, EngFlexothermoplasmonpolaritonicResonanceEnergy, HuntSpatialLinkageJoint, and SensorSpringBendingDriftRate."""

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


def test_m347_fail_lad_dynamic_interlaminar_shear_rate_fixed(tmp_path: Path):
    c1 = f"{135.0:>20.4f}{385.0:>20.4f}{42.0:>20.4f}{1.95:>20.4f}{0.965:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1630:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Interlaminar Shear Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_INTERLAMINAR_SHEAR_RATE/1630
Ladeveze Dynamic Interlaminar Shear Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1630 in model.fail_laddynamicinterlaminarshearrates
    fdisr = model.fail_laddynamicinterlaminarshearrates[1630]
    assert pytest.approx(fdisr.tau_disr0) == 135.0
    assert pytest.approx(fdisr.tau_disrc) == 385.0
    assert pytest.approx(fdisr.gamma_disr) == 42.0
    assert pytest.approx(fdisr.p_disr) == 1.95
    assert pytest.approx(fdisr.d_disr_max) == 0.965
    assert fdisr.ifail_sh == 1
    assert fdisr.ifail_so == 2
    assert fdisr.fail_id == 1630
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_INTERLAMINAR_SHEAR_RATE"


def test_m347_fail_lad_dynamic_interlaminar_shear_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Interlaminar Shear Rate Free Format Test
/FAIL/LAD_DYNAMIC_INTERLAMINAR_SHEAR_RATE/1631
145.0, 415.0, 48.0, 2.15, 0.945
1, 1
1631
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1631 in model.fail_laddynamicinterlaminarshearrates
    fdisr = model.fail_laddynamicinterlaminarshearrates[1631]
    assert pytest.approx(fdisr.tau_disr0) == 145.0
    assert pytest.approx(fdisr.tau_disrc) == 415.0
    assert pytest.approx(fdisr.gamma_disr) == 48.0
    assert pytest.approx(fdisr.p_disr) == 2.15
    assert pytest.approx(fdisr.d_disr_max) == 0.945
    assert fdisr.fail_id == 1631


def test_m347_fail_lad_dynamic_interlaminar_shear_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Interlaminar Shear Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_SHEAR_RATE/1632
120.0, 340.0, 36.0, 1.75, 0.950
1, 1
/FAIL/LAD_DISR/1633
120.0, 340.0, 36.0, 1.75, 0.950
1, 1
/FAIL/LAD_DISR_MODEL/1634
120.0, 340.0, 36.0, 1.75, 0.950
1, 1
/FAIL/LAD_DISR_LAW/1635
120.0, 340.0, 36.0, 1.75, 0.950
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_INTERLAMINAR_SHEAR/1636
120.0, 340.0, 36.0, 1.75, 0.950
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1632 in model.fail_laddynamicinterlaminarshearrates
    assert 1633 in model.fail_laddynamicinterlaminarshearrates
    assert 1634 in model.fail_laddynamicinterlaminarshearrates
    assert 1635 in model.fail_laddynamicinterlaminarshearrates
    assert 1636 in model.fail_laddynamicinterlaminarshearrates
    assert len(model.raw_fails) == 5


def test_m347_fail_lad_dynamic_interlaminar_shear_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_DYNAMIC_INTERLAMINAR_SHEAR_RATE/1637
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m347_eng_flexothermoplasmonpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00092:>20.6f}{182:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoplasmonpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPLASMONPOLARITONIC_RESONANCE_ENERGY/1
Flexothermoplasmonpolaritonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoplasmonpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonpolaritonic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftppr) == 0.00092
    assert eng.sens_id == 182


def test_m347_eng_flexothermoplasmonpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermoplasmonpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPLASMONPOLARITONIC_RESONANCE_ENERGY/2
0.00108, 192
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermoplasmonpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftppr) == 0.00108
    assert eng.sens_id == 192


def test_m347_eng_flexothermoplasmonpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoplasmonpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PLASMON_POLARITON_RES_WORK/3
0.00074, 142
/ENG/EFLEXOTHERMOPLASMONPOLARITONICRESONANCE/4
0.00079, 148
/ENG/FLEXOTHERMOPLASMONPOLARITONIC_RESONANCE_DISSIPATION/5
0.00086, 158
/ENG/EM_FLEXOTHERMOPLASMONPOLARITONIC_RESONANCE/6
0.00092, 168
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoplasmonpolaritonic_resonance_energies
    assert 4 in model.eng_flexothermoplasmonpolaritonic_resonance_energies
    assert 5 in model.eng_flexothermoplasmonpolaritonic_resonance_energies
    assert 6 in model.eng_flexothermoplasmonpolaritonic_resonance_energies


def test_m347_eng_flexothermoplasmonpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOPLASMONPOLARITONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m347_lagmul_hunt_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{331:>10d}{332:>10d}{333:>10d}{2.85e7:>20.1f}{94:>10d}{5.2e-5:>20.6e}"
    c2 = f"{152.0:>20.4f}{142.0:>20.4f}{124.0:>20.4f}{80.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Hunt Spatial Linkage Joint Fixed Format Test
2022 0
/HUNT_SPATIAL_LINKAGE_JOINT/295
Hunt Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 295 in model.lagmul_hunt_spatial_linkage_joints
    joint = model.lagmul_hunt_spatial_linkage_joints[295]
    assert joint.node1 == 331
    assert joint.node2 == 332
    assert joint.node3 == 333
    assert pytest.approx(joint.stiff) == 2.85e7
    assert joint.skew_id == 94
    assert pytest.approx(joint.tol) == 5.2e-5
    assert pytest.approx(joint.link_len_a) == 152.0
    assert pytest.approx(joint.link_len_b) == 142.0
    assert pytest.approx(joint.twist_angle_alpha) == 124.0
    assert pytest.approx(joint.offset_distance_v) == 80.0


def test_m347_lagmul_hunt_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Hunt Spatial Linkage Joint Free Format Test
/LAGMUL/HUNT_SPATIAL_LINKAGE_JOINT/296
431, 432, 433, 14.5e6, 114, 6.2e-5
154.0, 144.0, 126.0, 82.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 296 in model.lagmul_hunt_spatial_linkage_joints
    joint = model.lagmul_hunt_spatial_linkage_joints[296]
    assert joint.node1 == 431
    assert joint.node2 == 432
    assert joint.node3 == 433
    assert pytest.approx(joint.stiff) == 14.5e6
    assert joint.skew_id == 114
    assert pytest.approx(joint.tol) == 6.2e-5
    assert pytest.approx(joint.link_len_a) == 154.0
    assert pytest.approx(joint.link_len_b) == 144.0
    assert pytest.approx(joint.twist_angle_alpha) == 126.0
    assert pytest.approx(joint.offset_distance_v) == 82.0


def test_m347_lagmul_hunt_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Hunt Spatial Linkage Joint Aliases Test
/LAGMUL/HUNT_SPATIAL_LINKAGE/297
1, 2, 3, 1e6, 0, 1e-6
28.0, 28.0, 65.0, 14.0
/HUNT_SPATIAL_LINKAGE/298
1, 2, 3, 1e6, 0, 1e-6
28.0, 28.0, 65.0, 14.0
/HUNT_SPATIAL_MULTI_LOOP_MECHANISM/299
1, 2, 3, 1e6, 0, 1e-6
28.0, 28.0, 65.0, 14.0
/HUNT_SPATIAL_SCREW_AXIS_MECHANISM/300
1, 2, 3, 1e6, 0, 1e-6
28.0, 28.0, 65.0, 14.0
/HUNT_SPATIAL_6R_MECHANISM/301
1, 2, 3, 1e6, 0, 1e-6
28.0, 28.0, 65.0, 14.0
/HUNT_SPATIAL_OVERCONSTRAINED_MECHANISM/302
1, 2, 3, 1e6, 0, 1e-6
28.0, 28.0, 65.0, 14.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 297 in model.lagmul_hunt_spatial_linkage_joints
    assert 298 in model.lagmul_hunt_spatial_linkage_joints
    assert 299 in model.lagmul_hunt_spatial_linkage_joints
    assert 300 in model.lagmul_hunt_spatial_linkage_joints
    assert 301 in model.lagmul_hunt_spatial_linkage_joints
    assert 302 in model.lagmul_hunt_spatial_linkage_joints


def test_m347_lagmul_hunt_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/HUNT_SPATIAL_LINKAGE_JOINT/303
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m347_sensor_spring_bending_drift_rate_fixed(tmp_path: Path):
    c1 = f"{1075:>10d}{5.15e9:>20.1f}{0.0195:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Bending Drift Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_DRIFT_RATE/328
Spring Bending Drift Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 328 in model.sensor_spring_bending_drift_rates
    sensor = model.sensor_spring_bending_drift_rates[328]
    assert sensor.spring_id == 1075
    assert pytest.approx(sensor.jbend_drift_max) == 5.15e9
    assert pytest.approx(sensor.t_delay) == 0.0195
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_DRIFT_RATE"


def test_m347_sensor_spring_bending_drift_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Bending Drift Rate Sensor Free Format Test
/SENSOR/SPRING_BENDING_DRIFT_RATE/329
1076, 5.25e9, 0.0220
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 329 in model.sensor_spring_bending_drift_rates
    sensor = model.sensor_spring_bending_drift_rates[329]
    assert sensor.spring_id == 1076
    assert pytest.approx(sensor.jbend_drift_max) == 5.25e9
    assert pytest.approx(sensor.t_delay) == 0.0220


def test_m347_sensor_spring_bending_drift_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Bending Drift Rate Sensor Aliases Test
/SENSOR/SPRING_BEND_DRIFT_RATE/330
1077, 5.3e9, 0.0125
/SENSOR/SPRING_RATE_DRIFT_BEND/331
1078, 5.3e9, 0.0125
/SENSOR/BENDING_DRIFT_RATE_SPRING/332
1079, 5.3e9, 0.0125
/SENSOR/SPRING_DRIFT_BEND/333
1080, 5.3e9, 0.0125
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 330 in model.sensor_spring_bending_drift_rates
    assert 331 in model.sensor_spring_bending_drift_rates
    assert 332 in model.sensor_spring_bending_drift_rates
    assert 333 in model.sensor_spring_bending_drift_rates
    assert len(model.sensors) == 4


def test_m347_sensor_spring_bending_drift_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_BENDING_DRIFT_RATE/334
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

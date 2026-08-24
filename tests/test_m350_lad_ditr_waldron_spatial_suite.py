"""Tests for Milestone M350: LadDynamicInterlaminarTensionRate Failure Model, EngFlexothermoplasmonphononpolaritonicResonanceEnergy, WaldronSpatialLinkageJoint, and SensorSpringTransverseSurgeRate."""

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


def test_m350_fail_lad_dynamic_interlaminar_tension_rate_fixed(tmp_path: Path):
    c1 = f"{115.0:>20.4f}{345.0:>20.4f}{36.0:>20.4f}{1.95:>20.4f}{0.980:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1660:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Interlaminar Tension Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_INTERLAMINAR_TENSION_RATE/1660
Ladeveze Dynamic Interlaminar Tension Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1660 in model.fail_laddynamicinterlaminartensionrates
    fditr = model.fail_laddynamicinterlaminartensionrates[1660]
    assert pytest.approx(fditr.sigma_ditr0) == 115.0
    assert pytest.approx(fditr.sigma_ditrc) == 345.0
    assert pytest.approx(fditr.gamma_ditr) == 36.0
    assert pytest.approx(fditr.p_ditr) == 1.95
    assert pytest.approx(fditr.d_ditr_max) == 0.980
    assert fditr.ifail_sh == 1
    assert fditr.ifail_so == 2
    assert fditr.fail_id == 1660
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_INTERLAMINAR_TENSION_RATE"


def test_m350_fail_lad_dynamic_interlaminar_tension_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Interlaminar Tension Rate Free Format Test
/FAIL/LAD_DYNAMIC_INTERLAMINAR_TENSION_RATE/1661
125.0, 375.0, 42.0, 2.15, 0.960
1, 1
1661
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1661 in model.fail_laddynamicinterlaminartensionrates
    fditr = model.fail_laddynamicinterlaminartensionrates[1661]
    assert pytest.approx(fditr.sigma_ditr0) == 125.0
    assert pytest.approx(fditr.sigma_ditrc) == 375.0
    assert pytest.approx(fditr.gamma_ditr) == 42.0
    assert pytest.approx(fditr.p_ditr) == 2.15
    assert pytest.approx(fditr.d_ditr_max) == 0.960
    assert fditr.fail_id == 1661


def test_m350_fail_lad_dynamic_interlaminar_tension_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Interlaminar Tension Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_TENSION_RATE/1662
105.0, 315.0, 32.0, 1.75, 0.965
1, 1
/FAIL/LAD_DITR/1663
105.0, 315.0, 32.0, 1.75, 0.965
1, 1
/FAIL/LAD_DITR_MODEL/1664
105.0, 315.0, 32.0, 1.75, 0.965
1, 1
/FAIL/LAD_DITR_LAW/1665
105.0, 315.0, 32.0, 1.75, 0.965
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_INTERLAMINAR_TENSION/1666
105.0, 315.0, 32.0, 1.75, 0.965
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1662 in model.fail_laddynamicinterlaminartensionrates
    assert 1663 in model.fail_laddynamicinterlaminartensionrates
    assert 1664 in model.fail_laddynamicinterlaminartensionrates
    assert 1665 in model.fail_laddynamicinterlaminartensionrates
    assert 1666 in model.fail_laddynamicinterlaminartensionrates
    assert len(model.raw_fails) == 5


def test_m350_fail_lad_dynamic_interlaminar_tension_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_DYNAMIC_INTERLAMINAR_TENSION_RATE/1667
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m350_eng_flexothermoplasmonphononpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00105:>20.6f}{205:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoplasmonphononpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPLASMONPHONONPOLARITONIC_RESONANCE_ENERGY/1
Flexothermoplasmonphononpolaritonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoplasmonphononpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonphononpolaritonic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftpppr) == 0.00105
    assert eng.sens_id == 205


def test_m350_eng_flexothermoplasmonphononpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermoplasmonphononpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPLASMONPHONONPOLARITONIC_RESONANCE_ENERGY/2
0.00125, 215
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermoplasmonphononpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonphononpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftpppr) == 0.00125
    assert eng.sens_id == 215


def test_m350_eng_flexothermoplasmonphononpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoplasmonphononpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PLASMON_PHONON_POLARITON_RES_WORK/3
0.00088, 155
/ENG/EFLEXOTHERMOPLASMONPHONONPOLARITONICRESONANCE/4
0.00092, 162
/ENG/FLEXOTHERMOPLASMONPHONONPOLARITONIC_RESONANCE_DISSIPATION/5
0.00098, 172
/ENG/EM_FLEXOTHERMOPLASMONPHONONPOLARITONIC_RESONANCE/6
0.00104, 182
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoplasmonphononpolaritonic_resonance_energies
    assert 4 in model.eng_flexothermoplasmonphononpolaritonic_resonance_energies
    assert 5 in model.eng_flexothermoplasmonphononpolaritonic_resonance_energies
    assert 6 in model.eng_flexothermoplasmonphononpolaritonic_resonance_energies


def test_m350_eng_flexothermoplasmonphononpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOPLASMONPHONONPOLARITONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m350_lagmul_waldron_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{361:>10d}{362:>10d}{363:>10d}{3.15e7:>20.1f}{106:>10d}{6.0e-5:>20.6e}"
    c2 = f"{165.0:>20.4f}{155.0:>20.4f}{135.0:>20.4f}{92.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Waldron Spatial Linkage Joint Fixed Format Test
2022 0
/WALDRON_SPATIAL_LINKAGE_JOINT/325
Waldron Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 325 in model.lagmul_waldron_spatial_linkage_joints
    joint = model.lagmul_waldron_spatial_linkage_joints[325]
    assert joint.node1 == 361
    assert joint.node2 == 362
    assert joint.node3 == 363
    assert pytest.approx(joint.stiff) == 3.15e7
    assert joint.skew_id == 106
    assert pytest.approx(joint.tol) == 6.0e-5
    assert pytest.approx(joint.link_len_a) == 165.0
    assert pytest.approx(joint.link_len_b) == 155.0
    assert pytest.approx(joint.twist_angle_alpha) == 135.0
    assert pytest.approx(joint.offset_distance_s) == 92.0


def test_m350_lagmul_waldron_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Waldron Spatial Linkage Joint Free Format Test
/LAGMUL/WALDRON_SPATIAL_LINKAGE_JOINT/326
461, 462, 463, 16.0e6, 126, 7.0e-5
168.0, 158.0, 138.0, 94.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 326 in model.lagmul_waldron_spatial_linkage_joints
    joint = model.lagmul_waldron_spatial_linkage_joints[326]
    assert joint.node1 == 461
    assert joint.node2 == 462
    assert joint.node3 == 463
    assert pytest.approx(joint.stiff) == 16.0e6
    assert joint.skew_id == 126
    assert pytest.approx(joint.tol) == 7.0e-5
    assert pytest.approx(joint.link_len_a) == 168.0
    assert pytest.approx(joint.link_len_b) == 158.0
    assert pytest.approx(joint.twist_angle_alpha) == 138.0
    assert pytest.approx(joint.offset_distance_s) == 94.0


def test_m350_lagmul_waldron_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Waldron Spatial Linkage Joint Aliases Test
/LAGMUL/WALDRON_SPATIAL_LINKAGE/327
1, 2, 3, 1e6, 0, 1e-6
34.0, 34.0, 78.0, 20.0
/WALDRON_SPATIAL_LINKAGE/328
1, 2, 3, 1e6, 0, 1e-6
34.0, 34.0, 78.0, 20.0
/WALDRON_SPATIAL_MULTI_LOOP_MECHANISM/329
1, 2, 3, 1e6, 0, 1e-6
34.0, 34.0, 78.0, 20.0
/WALDRON_SPATIAL_HYBRID_MECHANISM/330
1, 2, 3, 1e6, 0, 1e-6
34.0, 34.0, 78.0, 20.0
/WALDRON_SPATIAL_6R_MECHANISM/331
1, 2, 3, 1e6, 0, 1e-6
34.0, 34.0, 78.0, 20.0
/WALDRON_SPATIAL_OVERCONSTRAINED_MECHANISM/332
1, 2, 3, 1e6, 0, 1e-6
34.0, 34.0, 78.0, 20.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 327 in model.lagmul_waldron_spatial_linkage_joints
    assert 328 in model.lagmul_waldron_spatial_linkage_joints
    assert 329 in model.lagmul_waldron_spatial_linkage_joints
    assert 330 in model.lagmul_waldron_spatial_linkage_joints
    assert 331 in model.lagmul_waldron_spatial_linkage_joints
    assert 332 in model.lagmul_waldron_spatial_linkage_joints


def test_m350_lagmul_waldron_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/WALDRON_SPATIAL_LINKAGE_JOINT/333
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m350_sensor_spring_transverse_surge_rate_fixed(tmp_path: Path):
    c1 = f"{1105:>10d}{5.45e9:>20.1f}{0.0225:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Transverse Surge Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_SURGE_RATE/358
Spring Transverse Surge Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 358 in model.sensor_spring_transverse_surge_rates
    sensor = model.sensor_spring_transverse_surge_rates[358]
    assert sensor.spring_id == 1105
    assert pytest.approx(sensor.jtrans_surge_max) == 5.45e9
    assert pytest.approx(sensor.t_delay) == 0.0225
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_SURGE_RATE"


def test_m350_sensor_spring_transverse_surge_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Transverse Surge Rate Sensor Free Format Test
/SENSOR/SPRING_TRANSVERSE_SURGE_RATE/359
1106, 5.55e9, 0.0250
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 359 in model.sensor_spring_transverse_surge_rates
    sensor = model.sensor_spring_transverse_surge_rates[359]
    assert sensor.spring_id == 1106
    assert pytest.approx(sensor.jtrans_surge_max) == 5.55e9
    assert pytest.approx(sensor.t_delay) == 0.0250


def test_m350_sensor_spring_transverse_surge_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Transverse Surge Rate Sensor Aliases Test
/SENSOR/SPRING_TRANS_SURGE_RATE/360
1107, 5.6e9, 0.0155
/SENSOR/SPRING_RATE_SURGE_TRANS/361
1108, 5.6e9, 0.0155
/SENSOR/TRANSVERSE_SURGE_RATE_SPRING/362
1109, 5.6e9, 0.0155
/SENSOR/SPRING_SURGE_TRANS/363
1110, 5.6e9, 0.0155
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 360 in model.sensor_spring_transverse_surge_rates
    assert 361 in model.sensor_spring_transverse_surge_rates
    assert 362 in model.sensor_spring_transverse_surge_rates
    assert 363 in model.sensor_spring_transverse_surge_rates
    assert len(model.sensors) == 4


def test_m350_sensor_spring_transverse_surge_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TRANSVERSE_SURGE_RATE/364
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

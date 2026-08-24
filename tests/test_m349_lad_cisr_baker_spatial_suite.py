"""Tests for Milestone M349: LadCoupleInterlaminarShearRate Failure Model, EngFlexothermophononpolaritonicResonanceEnergy, BakerSpatialLinkageJoint, and SensorSpringNormalSurgeRate."""

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


def test_m349_fail_lad_couple_interlaminar_shear_rate_fixed(tmp_path: Path):
    c1 = f"{145.0:>20.4f}{405.0:>20.4f}{46.0:>20.4f}{2.15:>20.4f}{0.975:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1650:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Interlaminar Shear Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_INTERLAMINAR_SHEAR_RATE/1650
Ladeveze Coupled Interlaminar Shear Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1650 in model.fail_ladcoupleinterlaminarshearrates
    fcisr = model.fail_ladcoupleinterlaminarshearrates[1650]
    assert pytest.approx(fcisr.tau_cisr0) == 145.0
    assert pytest.approx(fcisr.tau_cisrc) == 405.0
    assert pytest.approx(fcisr.gamma_cisr) == 46.0
    assert pytest.approx(fcisr.p_cisr) == 2.15
    assert pytest.approx(fcisr.d_cisr_max) == 0.975
    assert fcisr.ifail_sh == 1
    assert fcisr.ifail_so == 2
    assert fcisr.fail_id == 1650
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_INTERLAMINAR_SHEAR_RATE"


def test_m349_fail_lad_couple_interlaminar_shear_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Interlaminar Shear Rate Free Format Test
/FAIL/LAD_COUPLE_INTERLAMINAR_SHEAR_RATE/1651
155.0, 435.0, 52.0, 2.35, 0.955
1, 1
1651
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1651 in model.fail_ladcoupleinterlaminarshearrates
    fcisr = model.fail_ladcoupleinterlaminarshearrates[1651]
    assert pytest.approx(fcisr.tau_cisr0) == 155.0
    assert pytest.approx(fcisr.tau_cisrc) == 435.0
    assert pytest.approx(fcisr.gamma_cisr) == 52.0
    assert pytest.approx(fcisr.p_cisr) == 2.35
    assert pytest.approx(fcisr.d_cisr_max) == 0.955
    assert fcisr.fail_id == 1651


def test_m349_fail_lad_couple_interlaminar_shear_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Interlaminar Shear Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_SHEAR_RATE/1652
130.0, 360.0, 40.0, 1.85, 0.960
1, 1
/FAIL/LAD_CISR/1653
130.0, 360.0, 40.0, 1.85, 0.960
1, 1
/FAIL/LAD_CISR_MODEL/1654
130.0, 360.0, 40.0, 1.85, 0.960
1, 1
/FAIL/LAD_CISR_LAW/1655
130.0, 360.0, 40.0, 1.85, 0.960
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_INTERLAMINAR_SHEAR/1656
130.0, 360.0, 40.0, 1.85, 0.960
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1652 in model.fail_ladcoupleinterlaminarshearrates
    assert 1653 in model.fail_ladcoupleinterlaminarshearrates
    assert 1654 in model.fail_ladcoupleinterlaminarshearrates
    assert 1655 in model.fail_ladcoupleinterlaminarshearrates
    assert 1656 in model.fail_ladcoupleinterlaminarshearrates
    assert len(model.raw_fails) == 5


def test_m349_fail_lad_couple_interlaminar_shear_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_COUPLE_INTERLAMINAR_SHEAR_RATE/1657
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m349_eng_flexothermophononpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00098:>20.6f}{194:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermophononpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPHONONPOLARITONIC_RESONANCE_ENERGY/1
Flexothermophononpolaritonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermophononpolaritonic_resonance_energies
    eng = model.eng_flexothermophononpolaritonic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftphpr) == 0.00098
    assert eng.sens_id == 194


def test_m349_eng_flexothermophononpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermophononpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPHONONPOLARITONIC_RESONANCE_ENERGY/2
0.00116, 204
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermophononpolaritonic_resonance_energies
    eng = model.eng_flexothermophononpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftphpr) == 0.00116
    assert eng.sens_id == 204


def test_m349_eng_flexothermophononpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermophononpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PHONON_POLARITON_RES_WORK/3
0.00082, 150
/ENG/EFLEXOTHERMOPHONONPOLARITONICRESONANCE/4
0.00087, 156
/ENG/FLEXOTHERMOPHONONPOLARITONIC_RESONANCE_DISSIPATION/5
0.00094, 166
/ENG/EM_FLEXOTHERMOPHONONPOLARITONIC_RESONANCE/6
0.00100, 176
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermophononpolaritonic_resonance_energies
    assert 4 in model.eng_flexothermophononpolaritonic_resonance_energies
    assert 5 in model.eng_flexothermophononpolaritonic_resonance_energies
    assert 6 in model.eng_flexothermophononpolaritonic_resonance_energies


def test_m349_eng_flexothermophononpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOPHONONPOLARITONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m349_lagmul_baker_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{351:>10d}{352:>10d}{353:>10d}{3.05e7:>20.1f}{102:>10d}{5.8e-5:>20.6e}"
    c2 = f"{160.0:>20.4f}{150.0:>20.4f}{132.0:>20.4f}{88.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Baker Spatial Linkage Joint Fixed Format Test
2022 0
/BAKER_SPATIAL_LINKAGE_JOINT/315
Baker Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 315 in model.lagmul_baker_spatial_linkage_joints
    joint = model.lagmul_baker_spatial_linkage_joints[315]
    assert joint.node1 == 351
    assert joint.node2 == 352
    assert joint.node3 == 353
    assert pytest.approx(joint.stiff) == 3.05e7
    assert joint.skew_id == 102
    assert pytest.approx(joint.tol) == 5.8e-5
    assert pytest.approx(joint.link_len_a) == 160.0
    assert pytest.approx(joint.link_len_b) == 150.0
    assert pytest.approx(joint.twist_angle_alpha) == 132.0
    assert pytest.approx(joint.offset_distance_r) == 88.0


def test_m349_lagmul_baker_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Baker Spatial Linkage Joint Free Format Test
/LAGMUL/BAKER_SPATIAL_LINKAGE_JOINT/316
451, 452, 453, 15.5e6, 122, 6.8e-5
162.0, 152.0, 134.0, 90.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 316 in model.lagmul_baker_spatial_linkage_joints
    joint = model.lagmul_baker_spatial_linkage_joints[316]
    assert joint.node1 == 451
    assert joint.node2 == 452
    assert joint.node3 == 453
    assert pytest.approx(joint.stiff) == 15.5e6
    assert joint.skew_id == 122
    assert pytest.approx(joint.tol) == 6.8e-5
    assert pytest.approx(joint.link_len_a) == 162.0
    assert pytest.approx(joint.link_len_b) == 152.0
    assert pytest.approx(joint.twist_angle_alpha) == 134.0
    assert pytest.approx(joint.offset_distance_r) == 90.0


def test_m349_lagmul_baker_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Baker Spatial Linkage Joint Aliases Test
/LAGMUL/BAKER_SPATIAL_LINKAGE/317
1, 2, 3, 1e6, 0, 1e-6
32.0, 32.0, 75.0, 18.0
/BAKER_SPATIAL_LINKAGE/318
1, 2, 3, 1e6, 0, 1e-6
32.0, 32.0, 75.0, 18.0
/BAKER_SPATIAL_MULTI_LOOP_MECHANISM/319
1, 2, 3, 1e6, 0, 1e-6
32.0, 32.0, 75.0, 18.0
/BAKER_SPATIAL_VARIABLE_GEOMETRY_MECHANISM/320
1, 2, 3, 1e6, 0, 1e-6
32.0, 32.0, 75.0, 18.0
/BAKER_SPATIAL_6R_MECHANISM/321
1, 2, 3, 1e6, 0, 1e-6
32.0, 32.0, 75.0, 18.0
/BAKER_SPATIAL_OVERCONSTRAINED_MECHANISM/322
1, 2, 3, 1e6, 0, 1e-6
32.0, 32.0, 75.0, 18.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 317 in model.lagmul_baker_spatial_linkage_joints
    assert 318 in model.lagmul_baker_spatial_linkage_joints
    assert 319 in model.lagmul_baker_spatial_linkage_joints
    assert 320 in model.lagmul_baker_spatial_linkage_joints
    assert 321 in model.lagmul_baker_spatial_linkage_joints
    assert 322 in model.lagmul_baker_spatial_linkage_joints


def test_m349_lagmul_baker_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/BAKER_SPATIAL_LINKAGE_JOINT/323
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m349_sensor_spring_normal_surge_rate_fixed(tmp_path: Path):
    c1 = f"{1095:>10d}{5.35e9:>20.1f}{0.0215:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Normal Surge Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_SURGE_RATE/348
Spring Normal Surge Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 348 in model.sensor_spring_normal_surge_rates
    sensor = model.sensor_spring_normal_surge_rates[348]
    assert sensor.spring_id == 1095
    assert pytest.approx(sensor.jnorm_surge_max) == 5.35e9
    assert pytest.approx(sensor.t_delay) == 0.0215
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_SURGE_RATE"


def test_m349_sensor_spring_normal_surge_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Normal Surge Rate Sensor Free Format Test
/SENSOR/SPRING_NORMAL_SURGE_RATE/349
1096, 5.45e9, 0.0240
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 349 in model.sensor_spring_normal_surge_rates
    sensor = model.sensor_spring_normal_surge_rates[349]
    assert sensor.spring_id == 1096
    assert pytest.approx(sensor.jnorm_surge_max) == 5.45e9
    assert pytest.approx(sensor.t_delay) == 0.0240


def test_m349_sensor_spring_normal_surge_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Normal Surge Rate Sensor Aliases Test
/SENSOR/SPRING_NORM_SURGE_RATE/350
1097, 5.5e9, 0.0145
/SENSOR/SPRING_RATE_SURGE_NORM/351
1098, 5.5e9, 0.0145
/SENSOR/NORMAL_SURGE_RATE_SPRING/352
1099, 5.5e9, 0.0145
/SENSOR/SPRING_SURGE_NORM/353
1100, 5.5e9, 0.0145
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 350 in model.sensor_spring_normal_surge_rates
    assert 351 in model.sensor_spring_normal_surge_rates
    assert 352 in model.sensor_spring_normal_surge_rates
    assert 353 in model.sensor_spring_normal_surge_rates
    assert len(model.sensors) == 4


def test_m349_sensor_spring_normal_surge_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_NORMAL_SURGE_RATE/354
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

"""Tests for Milestone M352: LadCoupleInterlaminarTensionRate Failure Model, EngFlexothermoplasmonmagnonpolaritonicResonanceEnergy, BennettSpatialLinkageJoint, and SensorSpringTorsionalSurgeRate."""

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


def test_m352_fail_lad_couple_interlaminar_tension_rate_fixed(tmp_path: Path):
    c1 = f"{122.0:>20.4f}{366.0:>20.4f}{40.0:>20.4f}{2.10:>20.4f}{0.990:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1680:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Interlaminar Tension Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_INTERLAMINAR_TENSION_RATE/1680
Ladeveze Coupled Interlaminar Tension Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1680 in model.fail_ladcoupleinterlaminartensionrates
    fcitr = model.fail_ladcoupleinterlaminartensionrates[1680]
    assert pytest.approx(fcitr.sigma_citr0) == 122.0
    assert pytest.approx(fcitr.sigma_citrc) == 366.0
    assert pytest.approx(fcitr.gamma_citr) == 40.0
    assert pytest.approx(fcitr.p_citr) == 2.10
    assert pytest.approx(fcitr.d_citr_max) == 0.990
    assert fcitr.ifail_sh == 1
    assert fcitr.ifail_so == 2
    assert fcitr.fail_id == 1680
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_INTERLAMINAR_TENSION_RATE"


def test_m352_fail_lad_couple_interlaminar_tension_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Interlaminar Tension Rate Free Format Test
/FAIL/LAD_COUPLE_INTERLAMINAR_TENSION_RATE/1681
132.0, 396.0, 46.0, 2.30, 0.970
1, 1
1681
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1681 in model.fail_ladcoupleinterlaminartensionrates
    fcitr = model.fail_ladcoupleinterlaminartensionrates[1681]
    assert pytest.approx(fcitr.sigma_citr0) == 132.0
    assert pytest.approx(fcitr.sigma_citrc) == 396.0
    assert pytest.approx(fcitr.gamma_citr) == 46.0
    assert pytest.approx(fcitr.p_citr) == 2.30
    assert pytest.approx(fcitr.d_citr_max) == 0.970
    assert fcitr.fail_id == 1681


def test_m352_fail_lad_couple_interlaminar_tension_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Interlaminar Tension Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_TENSION_RATE/1682
112.0, 336.0, 36.0, 1.90, 0.975
1, 1
/FAIL/LAD_CITR/1683
112.0, 336.0, 36.0, 1.90, 0.975
1, 1
/FAIL/LAD_CITR_MODEL/1684
112.0, 336.0, 36.0, 1.90, 0.975
1, 1
/FAIL/LAD_CITR_LAW/1685
112.0, 336.0, 36.0, 1.90, 0.975
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_INTERLAMINAR_TENSION/1686
112.0, 336.0, 36.0, 1.90, 0.975
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1682 in model.fail_ladcoupleinterlaminartensionrates
    assert 1683 in model.fail_ladcoupleinterlaminartensionrates
    assert 1684 in model.fail_ladcoupleinterlaminartensionrates
    assert 1685 in model.fail_ladcoupleinterlaminartensionrates
    assert 1686 in model.fail_ladcoupleinterlaminartensionrates
    assert len(model.raw_fails) == 5


def test_m352_fail_lad_couple_interlaminar_tension_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_COUPLE_INTERLAMINAR_TENSION_RATE/1687
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m352_eng_flexothermoplasmonmagnonpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00120:>20.6f}{215:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoplasmonmagnonpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPLASMONMAGNONPOLARITONIC_RESONANCE_ENERGY/1
Flexothermoplasmonmagnonpolaritonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoplasmonmagnonpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonmagnonpolaritonic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftpmpr) == 0.00120
    assert eng.sens_id == 215


def test_m352_eng_flexothermoplasmonmagnonpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermoplasmonmagnonpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPLASMONMAGNONPOLARITONIC_RESONANCE_ENERGY/2
0.00140, 225
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermoplasmonmagnonpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonmagnonpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftpmpr) == 0.00140
    assert eng.sens_id == 225


def test_m352_eng_flexothermoplasmonmagnonpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoplasmonmagnonpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PLASMON_MAGNON_POLARITON_RES_WORK/3
0.00098, 165
/ENG/EFLEXOTHERMOPLASMONMAGNONPOLARITONICRESONANCE/4
0.00102, 172
/ENG/FLEXOTHERMOPLASMONMAGNONPOLARITONIC_RESONANCE_DISSIPATION/5
0.00108, 182
/ENG/EM_FLEXOTHERMOPLASMONMAGNONPOLARITONIC_RESONANCE/6
0.00115, 192
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoplasmonmagnonpolaritonic_resonance_energies
    assert 4 in model.eng_flexothermoplasmonmagnonpolaritonic_resonance_energies
    assert 5 in model.eng_flexothermoplasmonmagnonpolaritonic_resonance_energies
    assert 6 in model.eng_flexothermoplasmonmagnonpolaritonic_resonance_energies


def test_m352_eng_flexothermoplasmonmagnonpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOPLASMONMAGNONPOLARITONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m352_lagmul_bennett_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{381:>10d}{382:>10d}{383:>10d}{3.35e7:>20.1f}{110:>10d}{5.0e-5:>20.6e}"
    c2 = f"{175.0:>20.4f}{165.0:>20.4f}{145.0:>20.4f}{98.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Bennett Spatial Linkage Joint Fixed Format Test
2022 0
/BENNETT_SPATIAL_LINKAGE_JOINT/345
Bennett Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 345 in model.lagmul_bennett_spatial_linkage_joints
    joint = model.lagmul_bennett_spatial_linkage_joints[345]
    assert joint.node1 == 381
    assert joint.node2 == 382
    assert joint.node3 == 383
    assert pytest.approx(joint.stiff) == 3.35e7
    assert joint.skew_id == 110
    assert pytest.approx(joint.tol) == 5.0e-5
    assert pytest.approx(joint.link_len_a) == 175.0
    assert pytest.approx(joint.link_len_b) == 165.0
    assert pytest.approx(joint.twist_angle_alpha) == 145.0
    assert pytest.approx(joint.offset_distance_s) == 98.0


def test_m352_lagmul_bennett_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Bennett Spatial Linkage Joint Free Format Test
/LAGMUL/BENNETT_SPATIAL_LINKAGE_JOINT/346
481, 482, 483, 19.0e6, 130, 6.0e-5
176.0, 166.0, 146.0, 98.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 346 in model.lagmul_bennett_spatial_linkage_joints
    joint = model.lagmul_bennett_spatial_linkage_joints[346]
    assert joint.node1 == 481
    assert joint.node2 == 482
    assert joint.node3 == 483
    assert pytest.approx(joint.stiff) == 19.0e6
    assert joint.skew_id == 130
    assert pytest.approx(joint.tol) == 6.0e-5
    assert pytest.approx(joint.link_len_a) == 176.0
    assert pytest.approx(joint.link_len_b) == 166.0
    assert pytest.approx(joint.twist_angle_alpha) == 146.0
    assert pytest.approx(joint.offset_distance_s) == 98.0


def test_m352_lagmul_bennett_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Bennett Spatial Linkage Joint Aliases Test
/LAGMUL/BENNETT_SPATIAL_LINKAGE/347
1, 2, 3, 1e6, 0, 1e-6
38.0, 38.0, 82.0, 24.0
/BENNETT_SPATIAL_LINKAGE/348
1, 2, 3, 1e6, 0, 1e-6
38.0, 38.0, 82.0, 24.0
/BENNETT_SPATIAL_MULTI_LOOP_MECHANISM/349
1, 2, 3, 1e6, 0, 1e-6
38.0, 38.0, 82.0, 24.0
/BENNETT_SPATIAL_SKEW_MECHANISM/350
1, 2, 3, 1e6, 0, 1e-6
38.0, 38.0, 82.0, 24.0
/BENNETT_SPATIAL_4R_6R_MECHANISM/351
1, 2, 3, 1e6, 0, 1e-6
38.0, 38.0, 82.0, 24.0
/BENNETT_SPATIAL_OVERCONSTRAINED_MECHANISM/352
1, 2, 3, 1e6, 0, 1e-6
38.0, 38.0, 82.0, 24.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 347 in model.lagmul_bennett_spatial_linkage_joints
    assert 348 in model.lagmul_bennett_spatial_linkage_joints
    assert 349 in model.lagmul_bennett_spatial_linkage_joints
    assert 350 in model.lagmul_bennett_spatial_linkage_joints
    assert 351 in model.lagmul_bennett_spatial_linkage_joints
    assert 352 in model.lagmul_bennett_spatial_linkage_joints


def test_m352_lagmul_bennett_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/BENNETT_SPATIAL_LINKAGE_JOINT/353
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m352_sensor_spring_torsional_surge_rate_fixed(tmp_path: Path):
    c1 = f"{1125:>10d}{5.65e9:>20.1f}{0.0245:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Torsional Surge Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_SURGE_RATE/378
Spring Torsional Surge Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 378 in model.sensor_spring_torsional_surge_rates
    sensor = model.sensor_spring_torsional_surge_rates[378]
    assert sensor.spring_id == 1125
    assert pytest.approx(sensor.jrot_surge_max) == 5.65e9
    assert pytest.approx(sensor.t_delay) == 0.0245
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_SURGE_RATE"


def test_m352_sensor_spring_torsional_surge_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Torsional Surge Rate Sensor Free Format Test
/SENSOR/SPRING_TORSIONAL_SURGE_RATE/379
1126, 5.75e9, 0.0270
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 379 in model.sensor_spring_torsional_surge_rates
    sensor = model.sensor_spring_torsional_surge_rates[379]
    assert sensor.spring_id == 1126
    assert pytest.approx(sensor.jrot_surge_max) == 5.75e9
    assert pytest.approx(sensor.t_delay) == 0.0270


def test_m352_sensor_spring_torsional_surge_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Torsional Surge Rate Sensor Aliases Test
/SENSOR/SPRING_TORS_SURGE_RATE/380
1127, 5.8e9, 0.0175
/SENSOR/SPRING_RATE_SURGE_TORS/381
1128, 5.8e9, 0.0175
/SENSOR/TORSIONAL_SURGE_RATE_SPRING/382
1129, 5.8e9, 0.0175
/SENSOR/SPRING_SURGE_TORS/383
1130, 5.8e9, 0.0175
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 380 in model.sensor_spring_torsional_surge_rates
    assert 381 in model.sensor_spring_torsional_surge_rates
    assert 382 in model.sensor_spring_torsional_surge_rates
    assert 383 in model.sensor_spring_torsional_surge_rates
    assert len(model.sensors) == 4


def test_m352_sensor_spring_torsional_surge_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TORSIONAL_SURGE_RATE/384
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

"""Tests for Milestone M330: LadCoupleDamageViscoelasticity Failure Model, EngElectromagnetomechanicalResonanceEnergy, WunderlichLinkageJoint, and SensorSpringTotalAngularPopRate."""

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


def test_m330_fail_lad_couple_damage_viscoelasticity_fixed(tmp_path: Path):
    c1 = f"{0.280:>20.4f}{0.0045:>20.4f}{0.65:>20.4f}{14.5:>20.4f}{0.980:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1450:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Couple Damage Viscoelasticity Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_DAMAGE_VISCOELASTICITY/1450
Ladeveze Coupled Damage Viscoelasticity Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1450 in model.fail_ladcoupledamageviscoelasticitys
    fcdve = model.fail_ladcoupledamageviscoelasticitys[1450]
    assert pytest.approx(fcdve.g_inf) == 0.280
    assert pytest.approx(fcdve.tau_ve) == 0.0045
    assert pytest.approx(fcdve.beta_ve) == 0.65
    assert pytest.approx(fcdve.gamma_ve) == 14.5
    assert pytest.approx(fcdve.d_cdve_max) == 0.980
    assert fcdve.ifail_sh == 1
    assert fcdve.ifail_so == 2
    assert fcdve.fail_id == 1450
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_DAMAGE_VISCOELASTICITY"


def test_m330_fail_lad_couple_damage_viscoelasticity_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Couple Damage Viscoelasticity Free Format Test
/FAIL/LAD_COUPLE_DAMAGE_VISCOELASTICITY/1451
0.320, 0.0060, 0.70, 16.5, 0.960
1, 1
1451
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1451 in model.fail_ladcoupledamageviscoelasticitys
    fcdve = model.fail_ladcoupledamageviscoelasticitys[1451]
    assert pytest.approx(fcdve.g_inf) == 0.320
    assert pytest.approx(fcdve.tau_ve) == 0.0060
    assert pytest.approx(fcdve.beta_ve) == 0.70
    assert pytest.approx(fcdve.gamma_ve) == 16.5
    assert pytest.approx(fcdve.d_cdve_max) == 0.960
    assert fcdve.fail_id == 1451


def test_m330_fail_lad_couple_damage_viscoelasticity_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Couple Damage Viscoelasticity Aliases Test
/FAIL/LADEVEZE_COUPLED_DAMAGE_VISCOELASTICITY/1452
0.25, 0.005, 0.60, 12.0, 0.95
1, 1
/FAIL/LAD_CDVE/1453
0.25, 0.005, 0.60, 12.0, 0.95
1, 1
/FAIL/LAD_CDVE_MODEL/1454
0.25, 0.005, 0.60, 12.0, 0.95
1, 1
/FAIL/LAD_CDVE_LAW/1455
0.25, 0.005, 0.60, 12.0, 0.95
1, 1
/FAIL/LADEVEZE_DAMAGE_VISCOELASTICITY_COUPLING/1456
0.25, 0.005, 0.60, 12.0, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1452 in model.fail_ladcoupledamageviscoelasticitys
    assert 1453 in model.fail_ladcoupledamageviscoelasticitys
    assert 1454 in model.fail_ladcoupledamageviscoelasticitys
    assert 1455 in model.fail_ladcoupledamageviscoelasticitys
    assert 1456 in model.fail_ladcoupledamageviscoelasticitys
    assert len(model.raw_fails) == 5


def test_m330_fail_lad_couple_damage_viscoelasticity_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_COUPLE_DAMAGE_VISCOELASTICITY/1457
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m330_eng_electromagnetomechanical_resonance_energy(tmp_path: Path):
    c1 = f"{0.00095:>20.6f}{145:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Electromagnetomechanical Resonance Energy Fixed and Free Format Test
2022 0
/ENG/ELECTROMAGNETOMECHANICAL_RESONANCE_ENERGY/1
Fixed Electromagnetomechanical Resonance Energy Output
{c1}
/ENG/ELECTROMAGNETOMECHANICAL_RESONANCE_ENERGY/2
Free Electromagnetomechanical Resonance Energy Output
0.00135, 290
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electromagnetomechanical_resonance_energies
    assert 2 in model.eng_electromagnetomechanical_resonance_energies
    emmr1 = model.eng_electromagnetomechanical_resonance_energies[1]
    assert pytest.approx(emmr1.dt_emmr) == 0.00095
    assert emmr1.sens_id == 145
    emmr2 = model.eng_electromagnetomechanical_resonance_energies[2]
    assert pytest.approx(emmr2.dt_emmr) == 0.00135
    assert emmr2.sens_id == 290


def test_m330_eng_electromagnetomechanical_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Electromagnetomechanical Resonance Energy Aliases Test
/ENG/EMM_RES_WORK/3
0.00025, 75
/ENG/EEMMRESONANCE/4
0.00030, 85
/ENG/EMM_RESONANCE_DISSIPATION/5
0.00035, 95
/ENG/EM_ELECTROMAGNETOMECHANICAL_RESONANCE/6
0.00040, 105
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_electromagnetomechanical_resonance_energies
    assert 4 in model.eng_electromagnetomechanical_resonance_energies
    assert 5 in model.eng_electromagnetomechanical_resonance_energies
    assert 6 in model.eng_electromagnetomechanical_resonance_energies


def test_m330_eng_electromagnetomechanical_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/ELECTROMAGNETOMECHANICAL_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m330_lagmul_wunderlich_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{161:>10d}{162:>10d}{163:>10d}{1.15e7:>20.1f}{35:>10d}{2.8e-5:>20.6e}"
    c2 = f"{82.0:>20.4f}{76.5:>20.4f}{68.0:>20.4f}{26.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Wunderlich Linkage Joint Fixed Format Test
2022 0
/WUNDERLICH_LINKAGE_JOINT/125
Wunderlich Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 125 in model.lagmul_wunderlich_linkage_joints
    joint = model.lagmul_wunderlich_linkage_joints[125]
    assert joint.node1 == 161
    assert joint.node2 == 162
    assert joint.node3 == 163
    assert pytest.approx(joint.stiff) == 1.15e7
    assert joint.skew_id == 35
    assert pytest.approx(joint.tol) == 2.8e-5
    assert pytest.approx(joint.link_len_a) == 82.0
    assert pytest.approx(joint.link_len_b) == 76.5
    assert pytest.approx(joint.twist_angle_alpha) == 68.0
    assert pytest.approx(joint.offset_distance_e) == 26.0


def test_m330_lagmul_wunderlich_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Wunderlich Linkage Joint Free Format Test
/LAGMUL/WUNDERLICH_LINKAGE_JOINT/126
261, 262, 263, 8.2e6, 55, 4.2e-5
86.0, 80.0, 72.0, 28.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 126 in model.lagmul_wunderlich_linkage_joints
    joint = model.lagmul_wunderlich_linkage_joints[126]
    assert joint.node1 == 261
    assert joint.node2 == 262
    assert joint.node3 == 263
    assert pytest.approx(joint.stiff) == 8.2e6
    assert joint.skew_id == 55
    assert pytest.approx(joint.tol) == 4.2e-5
    assert pytest.approx(joint.link_len_a) == 86.0
    assert pytest.approx(joint.link_len_b) == 80.0
    assert pytest.approx(joint.twist_angle_alpha) == 72.0
    assert pytest.approx(joint.offset_distance_e) == 28.0


def test_m330_lagmul_wunderlich_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Wunderlich Linkage Joint Aliases Test
/LAGMUL/WUNDERLICH_LINKAGE/127
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/WUNDERLICH_LINKAGE/128
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/WUNDERLICH_SPATIAL_MECHANISM/129
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/WUNDERLICH_6R_MECHANISM/130
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 127 in model.lagmul_wunderlich_linkage_joints
    assert 128 in model.lagmul_wunderlich_linkage_joints
    assert 129 in model.lagmul_wunderlich_linkage_joints
    assert 130 in model.lagmul_wunderlich_linkage_joints


def test_m330_lagmul_wunderlich_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/WUNDERLICH_LINKAGE_JOINT/131
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m330_sensor_spring_total_angular_pop_rate_fixed(tmp_path: Path):
    c1 = f"{901:>10d}{1.85e9:>20.1f}{0.0065:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Total Angular Pop Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/158
Spring Total Angular Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 158 in model.sensor_spring_total_angular_pop_rates
    sensor = model.sensor_spring_total_angular_pop_rates[158]
    assert sensor.spring_id == 901
    assert pytest.approx(sensor.jtot_ang_pop_max) == 1.85e9
    assert pytest.approx(sensor.t_delay) == 0.0065
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_POP_RATE"


def test_m330_sensor_spring_total_angular_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Total Angular Pop Rate Sensor Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/159
902, 1.95e9, 0.0085
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 159 in model.sensor_spring_total_angular_pop_rates
    sensor = model.sensor_spring_total_angular_pop_rates[159]
    assert sensor.spring_id == 902
    assert pytest.approx(sensor.jtot_ang_pop_max) == 1.95e9
    assert pytest.approx(sensor.t_delay) == 0.0085


def test_m330_sensor_spring_total_angular_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Total Angular Pop Rate Sensor Aliases Test
/SENSOR/SPRING_TOT_ANG_POP_RATE/160
903, 2.5e9, 0.001
/SENSOR/SPRING_POP_ANG_TOT/161
904, 2.5e9, 0.001
/SENSOR/TOTAL_ANGULAR_POP_RATE_SPRING/162
905, 2.5e9, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 160 in model.sensor_spring_total_angular_pop_rates
    assert 161 in model.sensor_spring_total_angular_pop_rates
    assert 162 in model.sensor_spring_total_angular_pop_rates
    assert len(model.sensors) == 3


def test_m330_sensor_spring_total_angular_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/163
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

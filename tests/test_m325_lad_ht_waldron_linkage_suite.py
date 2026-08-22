"""Tests for Milestone M325: LadHygrothermal Failure Model, EngThermoflexoelectricEnergy, WaldronLinkageJoint, and SensorSpringNormalPopRate."""

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


def test_m325_fail_lad_hygrothermal_fixed(tmp_path: Path):
    c1 = f"{0.0250:>20.4f}{0.00045:>20.6f}{420.0:>20.2f}{0.1250:>20.4f}{0.985:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1350:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Hygrothermal Fixed Format Test
2022 0
/FAIL/LAD_HYGROTHERMAL/1350
Ladeveze Hygrothermal Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1350 in model.fail_ladhygrothermals
    fht = model.fail_ladhygrothermals[1350]
    assert pytest.approx(fht.c_moist) == 0.0250
    assert pytest.approx(fht.beta_exp) == 0.00045
    assert pytest.approx(fht.t_glass) == 420.0
    assert pytest.approx(fht.d_ht_rate) == 0.1250
    assert pytest.approx(fht.d_ht_max) == 0.985
    assert fht.ifail_sh == 1
    assert fht.ifail_so == 2
    assert fht.fail_id == 1350
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_HYGROTHERMAL"


def test_m325_fail_lad_hygrothermal_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Hygrothermal Free Format Test
/FAIL/LAD_HYGROTHERMAL/1351
0.0350, 0.00060, 450.0, 0.150, 0.975
1, 1
1351
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1351 in model.fail_ladhygrothermals
    fht = model.fail_ladhygrothermals[1351]
    assert pytest.approx(fht.c_moist) == 0.0350
    assert pytest.approx(fht.beta_exp) == 0.00060
    assert pytest.approx(fht.t_glass) == 450.0
    assert pytest.approx(fht.d_ht_rate) == 0.150
    assert pytest.approx(fht.d_ht_max) == 0.975
    assert fht.fail_id == 1351


def test_m325_fail_lad_hygrothermal_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Hygrothermal Aliases Test
/FAIL/LADEVEZE_HYGROTHERMAL/1352
0.02, 0.0005, 400.0, 0.1, 0.95
1, 1
/FAIL/LAD_HT/1353
0.02, 0.0005, 400.0, 0.1, 0.95
1, 1
/FAIL/LAD_HYGRO_MODEL/1354
0.02, 0.0005, 400.0, 0.1, 0.95
1, 1
/FAIL/LAD_HYGRO_LAW/1355
0.02, 0.0005, 400.0, 0.1, 0.95
1, 1
/FAIL/LADEVEZE_HYGROTHERMAL_DAMAGE/1356
0.02, 0.0005, 400.0, 0.1, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1352 in model.fail_ladhygrothermals
    assert 1353 in model.fail_ladhygrothermals
    assert 1354 in model.fail_ladhygrothermals
    assert 1355 in model.fail_ladhygrothermals
    assert 1356 in model.fail_ladhygrothermals
    assert len(model.raw_fails) == 5


def test_m325_fail_lad_hygrothermal_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_HYGROTHERMAL/1357
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m325_eng_thermoflexoelectric_energy(tmp_path: Path):
    c1 = f"{0.00045:>20.6f}{95:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Thermoflexoelectric Energy Fixed and Free Format Test
2022 0
/ENG/THERMOFLEXOELECTRIC_ENERGY/1
Fixed Thermoflexoelectric Energy Output
{c1}
/ENG/THERMOFLEXOELECTRIC_ENERGY/2
Free Thermoflexoelectric Energy Output
0.00085, 190
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_thermoflexoelectric_energies
    assert 2 in model.eng_thermoflexoelectric_energies
    tfe1 = model.eng_thermoflexoelectric_energies[1]
    assert pytest.approx(tfe1.dt_tfe) == 0.00045
    assert tfe1.sens_id == 95
    tfe2 = model.eng_thermoflexoelectric_energies[2]
    assert pytest.approx(tfe2.dt_tfe) == 0.00085
    assert tfe2.sens_id == 190


def test_m325_eng_thermoflexoelectric_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Thermoflexoelectric Energy Aliases Test
/ENG/THERMOFLEXO_WORK/3
0.00025, 55
/ENG/ETHERMOFLEXOELECTRIC/4
0.00030, 65
/ENG/THERMOFLEXOELECTRIC_DISSIPATION/5
0.00035, 75
/ENG/EM_THERMOFLEXOELECTRIC/6
0.00040, 85
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_thermoflexoelectric_energies
    assert 4 in model.eng_thermoflexoelectric_energies
    assert 5 in model.eng_thermoflexoelectric_energies
    assert 6 in model.eng_thermoflexoelectric_energies


def test_m325_eng_thermoflexoelectric_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/THERMOFLEXOELECTRIC_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m325_lagmul_waldron_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{111:>10d}{112:>10d}{113:>10d}{6.5e6:>20.1f}{18:>10d}{1.5e-5:>20.6e}"
    c2 = f"{52.5:>20.4f}{45.0:>20.4f}{40.0:>20.4f}{15.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Waldron Linkage Joint Fixed Format Test
2022 0
/WALDRON_LINKAGE_JOINT/75
Waldron Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 75 in model.lagmul_waldron_linkage_joints
    joint = model.lagmul_waldron_linkage_joints[75]
    assert joint.node1 == 111
    assert joint.node2 == 112
    assert joint.node3 == 113
    assert pytest.approx(joint.stiff) == 6.5e6
    assert joint.skew_id == 18
    assert pytest.approx(joint.tol) == 1.5e-5
    assert pytest.approx(joint.link_len_a) == 52.5
    assert pytest.approx(joint.link_len_b) == 45.0
    assert pytest.approx(joint.twist_angle_alpha) == 40.0
    assert pytest.approx(joint.offset_distance_s) == 15.5


def test_m325_lagmul_waldron_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Waldron Linkage Joint Free Format Test
/LAGMUL/WALDRON_LINKAGE_JOINT/76
211, 212, 213, 4.2e6, 30, 2.5e-5
55.0, 50.5, 45.0, 18.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 76 in model.lagmul_waldron_linkage_joints
    joint = model.lagmul_waldron_linkage_joints[76]
    assert joint.node1 == 211
    assert joint.node2 == 212
    assert joint.node3 == 213
    assert pytest.approx(joint.stiff) == 4.2e6
    assert joint.skew_id == 30
    assert pytest.approx(joint.tol) == 2.5e-5
    assert pytest.approx(joint.link_len_a) == 55.0
    assert pytest.approx(joint.link_len_b) == 50.5
    assert pytest.approx(joint.twist_angle_alpha) == 45.0
    assert pytest.approx(joint.offset_distance_s) == 18.0


def test_m325_lagmul_waldron_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Waldron Linkage Joint Aliases Test
/LAGMUL/WALDRON_LINKAGE/77
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/WALDRON_LINKAGE/78
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/WALDRON_HYBRID_MECHANISM/79
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/WALDRON_SPATIAL_MECHANISM/80
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/WALDRON_6R_MECHANISM/81
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 77 in model.lagmul_waldron_linkage_joints
    assert 78 in model.lagmul_waldron_linkage_joints
    assert 79 in model.lagmul_waldron_linkage_joints
    assert 80 in model.lagmul_waldron_linkage_joints
    assert 81 in model.lagmul_waldron_linkage_joints


def test_m325_lagmul_waldron_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/WALDRON_LINKAGE_JOINT/82
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m325_sensor_spring_normal_pop_rate_fixed(tmp_path: Path):
    c1 = f"{751:>10d}{9.2e8:>20.1f}{0.0040:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Normal Pop Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_POP_RATE/108
Spring Normal Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 108 in model.sensor_spring_normal_pop_rates
    sensor = model.sensor_spring_normal_pop_rates[108]
    assert sensor.spring_id == 751
    assert pytest.approx(sensor.jnorm_pop_max) == 9.2e8
    assert pytest.approx(sensor.t_delay) == 0.0040
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_POP_RATE"


def test_m325_sensor_spring_normal_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Normal Pop Rate Sensor Free Format Test
/SENSOR/SPRING_NORMAL_POP_RATE/109
752, 9.8e8, 0.0060
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 109 in model.sensor_spring_normal_pop_rates
    sensor = model.sensor_spring_normal_pop_rates[109]
    assert sensor.spring_id == 752
    assert pytest.approx(sensor.jnorm_pop_max) == 9.8e8
    assert pytest.approx(sensor.t_delay) == 0.0060


def test_m325_sensor_spring_normal_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Normal Pop Rate Sensor Aliases Test
/SENSOR/SPRING_NORM_POP_RATE/110
753, 1.2e9, 0.001
/SENSOR/SPRING_RATE_POP_NORM/111
754, 1.2e9, 0.001
/SENSOR/NORMAL_POP_RATE_SPRING/112
755, 1.2e9, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 110 in model.sensor_spring_normal_pop_rates
    assert 111 in model.sensor_spring_normal_pop_rates
    assert 112 in model.sensor_spring_normal_pop_rates
    assert len(model.sensors) == 3


def test_m325_sensor_spring_normal_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_NORMAL_POP_RATE/113
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

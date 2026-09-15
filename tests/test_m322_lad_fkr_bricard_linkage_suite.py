"""Tests for Milestone M322: LadFiberKinkingRate Failure Model, EngFlexoelectricEnergy, BricardLinkageJoint, and SensorSpringTorsionalCrackleRate."""

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


def test_m322_fail_lad_fiber_kinking_rate_fixed(tmp_path: Path):
    c1 = f"{350.0:>20.4f}{3.5:>20.4f}{1.45:>20.4f}{4200.0:>20.4f}{0.965:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1309:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Fiber Kinking Rate Fixed Format Test
2022 0
/FAIL/LAD_FIBER_KINKING_RATE/1309
Ladeveze Fiber Kinking Rate Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1309 in model.fail_ladfiberkinkingrates
    ffkr = model.fail_ladfiberkinkingrates[1309]
    assert pytest.approx(ffkr.sigma_kink) == 350.0
    assert pytest.approx(ffkr.phi_kink0) == 3.5
    assert pytest.approx(ffkr.gamma_kink) == 1.45
    assert pytest.approx(ffkr.c_kink) == 4200.0
    assert pytest.approx(ffkr.d_kink_max) == 0.965
    assert ffkr.ifail_sh == 1
    assert ffkr.ifail_so == 2
    assert ffkr.fail_id == 1309
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_FIBER_KINKING_RATE"


def test_m322_fail_lad_fiber_kinking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Fiber Kinking Rate Free Format Test
/FAIL/LAD_FIBER_KINKING_RATE/1310
420.0, 4.0, 1.85, 4800.0, 0.950
1, 1
1310
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1310 in model.fail_ladfiberkinkingrates
    ffkr = model.fail_ladfiberkinkingrates[1310]
    assert pytest.approx(ffkr.sigma_kink) == 420.0
    assert pytest.approx(ffkr.phi_kink0) == 4.0
    assert pytest.approx(ffkr.gamma_kink) == 1.85
    assert pytest.approx(ffkr.c_kink) == 4800.0
    assert pytest.approx(ffkr.d_kink_max) == 0.950
    assert ffkr.fail_id == 1310


def test_m322_fail_lad_fiber_kinking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Fiber Kinking Rate Aliases Test
/FAIL/LADEVEZE_FIBER_KINKING_RATE/1311
300.0, 3.0, 1.2, 3800.0, 0.95
1, 1
/FAIL/LAD_FKR/1312
300.0, 3.0, 1.2, 3800.0, 0.95
1, 1
/FAIL/LAD_FKR_MODEL/1313
300.0, 3.0, 1.2, 3800.0, 0.95
1, 1
/FAIL/LAD_FKR_LAW/1314
300.0, 3.0, 1.2, 3800.0, 0.95
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_KINKING/1315
300.0, 3.0, 1.2, 3800.0, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1311 in model.fail_ladfiberkinkingrates
    assert 1312 in model.fail_ladfiberkinkingrates
    assert 1313 in model.fail_ladfiberkinkingrates
    assert 1314 in model.fail_ladfiberkinkingrates
    assert 1315 in model.fail_ladfiberkinkingrates
    assert len(model.raw_fails) == 5


def test_m322_eng_flexoelectric_energy(tmp_path: Path):
    c1 = f"{0.00035:>20.6f}{65:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexoelectric Energy Fixed and Free Format Test
2022 0
/ENG/FLEXOELECTRIC_ENERGY/1
Fixed Flexoelectric Energy Output
{c1}
/ENG/FLEXOELECTRIC_ENERGY/2
Free Flexoelectric Energy Output
0.00085, 145
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexoelectric_energies
    assert 2 in model.eng_flexoelectric_energies
    flx1 = model.eng_flexoelectric_energies[1]
    assert pytest.approx(flx1.dt_flx) == 0.00035
    assert flx1.sens_id == 65
    flx2 = model.eng_flexoelectric_energies[2]
    assert pytest.approx(flx2.dt_flx) == 0.00085
    assert flx2.sens_id == 145


def test_m322_eng_flexoelectric_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexoelectric Energy Aliases Test
/ENG/FLEXO_WORK/271
0.0019, 270
/ENG/EFLEXOELECTRIC/272
0.0029, 271
/ENG/FLEXOELECTRIC_DISSIPATION/273
0.0039, 272
/ENG/EM_FLEXOELECTRIC/274
0.0049, 273
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 271 in model.eng_flexoelectric_energies
    assert 272 in model.eng_flexoelectric_energies
    assert 273 in model.eng_flexoelectric_energies
    assert 274 in model.eng_flexoelectric_energies
    assert pytest.approx(model.eng_flexoelectric_energies[271].dt_flx) == 0.0019
    assert pytest.approx(model.eng_flexoelectric_energies[272].dt_flx) == 0.0029
    assert pytest.approx(model.eng_flexoelectric_energies[273].dt_flx) == 0.0039
    assert pytest.approx(model.eng_flexoelectric_energies[274].dt_flx) == 0.0049


def test_m322_bricard_linkage_joint(tmp_path: Path):
    c1 = f"{1071:>10d}{1072:>10d}{1073:>10d}{1.72e7:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{60.0:>20.4f}{45.0:>20.4f}{12.5:>20.4f}{30.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Bricard Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/BRICARD_LINKAGE_JOINT/1
Fixed Bricard Spatial Mechanism Joint
{c1}
{c2}
/LAGMUL/BRICARD_LINKAGE_JOINT/2
Free Bricard Spatial Mechanism Joint
1171, 1172, 1173, 1.92e7, 2, 2.7e-6
70.0, 60.0, 15.0, 45.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_bricard_linkage_joints
    assert 2 in model.lagmul_bricard_linkage_joints
    blj1 = model.lagmul_bricard_linkage_joints[1]
    assert blj1.node1 == 1071
    assert blj1.node2 == 1072
    assert blj1.node3 == 1073
    assert pytest.approx(blj1.stiff) == 1.72e7
    assert blj1.skew_id == 1
    assert pytest.approx(blj1.tol) == 1.0e-6
    assert pytest.approx(blj1.link_len) == 60.0
    assert pytest.approx(blj1.twist_angle) == 45.0
    assert pytest.approx(blj1.offset_dist) == 12.5
    assert pytest.approx(blj1.sym_angle) == 30.0

    blj2 = model.lagmul_bricard_linkage_joints[2]
    assert blj2.node1 == 1171
    assert blj2.node2 == 1172
    assert blj2.node3 == 1173
    assert pytest.approx(blj2.stiff) == 1.92e7
    assert blj2.skew_id == 2
    assert pytest.approx(blj2.tol) == 2.7e-6
    assert pytest.approx(blj2.link_len) == 70.0
    assert pytest.approx(blj2.twist_angle) == 60.0
    assert pytest.approx(blj2.offset_dist) == 15.0
    assert pytest.approx(blj2.sym_angle) == 45.0


def test_m322_bricard_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Bricard Linkage Joint Aliases Test
/BRICARD_LINKAGE_JOINT/275
1271, 1272, 1273, 9.9e6, 0, 1.0e-6
50.0, 30.0, 10.0, 20.0
/LAGMUL/BRICARD_LINKAGE/276
1274, 1275, 1276, 9.9e6, 0, 1.0e-6
50.0, 30.0, 10.0, 20.0
/BRICARD_LINKAGE/277
1277, 1278, 1279, 9.9e6, 0, 1.0e-6
50.0, 30.0, 10.0, 20.0
/BRICARD_SPATIAL_MECHANISM/278
1280, 1281, 1282, 9.9e6, 0, 1.0e-6
50.0, 30.0, 10.0, 20.0
/BRICARD_6R_MECHANISM/279
1283, 1284, 1285, 9.9e6, 0, 1.0e-6
50.0, 30.0, 10.0, 20.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 275 in model.lagmul_bricard_linkage_joints
    assert 276 in model.lagmul_bricard_linkage_joints
    assert 277 in model.lagmul_bricard_linkage_joints
    assert 278 in model.lagmul_bricard_linkage_joints
    assert 279 in model.lagmul_bricard_linkage_joints
    assert model.lagmul_bricard_linkage_joints[275].node1 == 1271
    assert model.lagmul_bricard_linkage_joints[276].node1 == 1274
    assert model.lagmul_bricard_linkage_joints[277].node1 == 1277
    assert model.lagmul_bricard_linkage_joints[278].node1 == 1280
    assert model.lagmul_bricard_linkage_joints[279].node1 == 1283


def test_m322_sensor_spring_torsional_crackle_rate(tmp_path: Path):
    c1 = f"{1178:>10d}{8.4e7:>20.4f}{0.0225:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Crackle Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/1
Fixed Spring Torsional Crackle Rate Sensor
{c1}
/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/2
Free Spring Torsional Crackle Rate Sensor
1179, 1.05e8, 0.0325
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_crackle_rates
    assert 2 in model.sensor_spring_torsional_crackle_rates
    s1 = model.sensor_spring_torsional_crackle_rates[1]
    assert s1.spring_id == 1178
    assert pytest.approx(s1.jtors_pop_max) == 8.4e7
    assert pytest.approx(s1.t_delay) == 0.0225

    s2 = model.sensor_spring_torsional_crackle_rates[2]
    assert s2.spring_id == 1179
    assert pytest.approx(s2.jtors_pop_max) == 1.05e8
    assert pytest.approx(s2.t_delay) == 0.0325

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TORSIONAL_CRACKLE_RATE"
    assert model.sensors[1].kind == "SPRING_TORSIONAL_CRACKLE_RATE"


def test_m322_sensor_spring_torsional_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Crackle Rate Aliases Test
/SENSOR/SPRING_TORS_CRACKLE_RATE/193
1208, 6.85e7, 0.0072
/SENSOR/SPRING_RATE_CRACKLE_TORS/194
1209, 7.05e7, 0.0082
/SENSOR/TORSIONAL_CRACKLE_RATE_SPRING/195
1210, 7.25e7, 0.0092
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 193 in model.sensor_spring_torsional_crackle_rates
    assert 194 in model.sensor_spring_torsional_crackle_rates
    assert 195 in model.sensor_spring_torsional_crackle_rates
    assert model.sensor_spring_torsional_crackle_rates[193].spring_id == 1208
    assert pytest.approx(model.sensor_spring_torsional_crackle_rates[193].jtors_pop_max) == 6.85e7
    assert model.sensor_spring_torsional_crackle_rates[194].spring_id == 1209
    assert model.sensor_spring_torsional_crackle_rates[195].spring_id == 1210

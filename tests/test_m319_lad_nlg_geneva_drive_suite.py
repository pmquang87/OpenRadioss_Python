"""Tests for Milestone M319: LadNonlocalGradient Failure Model, EngElectrorheologicalEnergy, GenevaDriveMechanismJoint, and SensorSpringNormalCrackleRate."""

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


def test_m319_fail_lad_nonlocal_gradient_fixed(tmp_path: Path):
    c1 = f"{3.85:>20.4f}{58.0:>20.4f}{1.25:>20.4f}{1.75:>20.4f}{0.985:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1279:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Nonlocal Gradient Damage Fixed Format Test
2022 0
/FAIL/LAD_NONLOCAL_GRADIENT/1279
Ladeveze Nonlocal Gradient Damage Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1279 in model.fail_ladnonlocalgradients
    flnlg = model.fail_ladnonlocalgradients[1279]
    assert pytest.approx(flnlg.y0_nlg) == 3.85
    assert pytest.approx(flnlg.yc_nlg) == 58.0
    assert pytest.approx(flnlg.lc_char) == 1.25
    assert pytest.approx(flnlg.p_nlg) == 1.75
    assert pytest.approx(flnlg.d_nlg_max) == 0.985
    assert flnlg.ifail_sh == 1
    assert flnlg.ifail_so == 2
    assert flnlg.fail_id == 1279
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_NONLOCAL_GRADIENT"


def test_m319_fail_lad_nonlocal_gradient_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Nonlocal Gradient Damage Free Format Test
/FAIL/LAD_NONLOCAL_GRADIENT/1280
4.65, 65.5, 1.85, 2.05, 0.965
1, 1
1280
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1280 in model.fail_ladnonlocalgradients
    flnlg = model.fail_ladnonlocalgradients[1280]
    assert pytest.approx(flnlg.y0_nlg) == 4.65
    assert pytest.approx(flnlg.yc_nlg) == 65.5
    assert pytest.approx(flnlg.lc_char) == 1.85
    assert pytest.approx(flnlg.p_nlg) == 2.05
    assert pytest.approx(flnlg.d_nlg_max) == 0.965
    assert flnlg.fail_id == 1280


def test_m319_fail_lad_nonlocal_gradient_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Nonlocal Gradient Damage Aliases Test
/FAIL/LADEVEZE_NONLOCAL_GRADIENT/1281
3.2, 42.0, 1.1, 1.5, 0.95
1, 1
/FAIL/LAD_NLG/1282
3.2, 42.0, 1.1, 1.5, 0.95
1, 1
/FAIL/LAD_NLG_MODEL/1283
3.2, 42.0, 1.1, 1.5, 0.95
1, 1
/FAIL/LAD_NLG_LAW/1284
3.2, 42.0, 1.1, 1.5, 0.95
1, 1
/FAIL/LADEVEZE_GRADIENT_DAMAGE/1285
3.2, 42.0, 1.1, 1.5, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1281 in model.fail_ladnonlocalgradients
    assert 1282 in model.fail_ladnonlocalgradients
    assert 1283 in model.fail_ladnonlocalgradients
    assert 1284 in model.fail_ladnonlocalgradients
    assert 1285 in model.fail_ladnonlocalgradients
    assert len(model.raw_fails) == 5


def test_m319_eng_electrorheological_energy(tmp_path: Path):
    c1 = f"{0.00065:>20.6f}{85:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Electrorheological Energy Fixed and Free Format Test
2022 0
/ENG/ELECTRORHEOLOGICAL_ENERGY/1
Fixed Electrorheological Energy Output
{c1}
/ENG/ELECTRORHEOLOGICAL_ENERGY/2
Free Electrorheological Energy Output
0.00130, 170
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrorheological_energies
    assert 2 in model.eng_electrorheological_energies
    er1 = model.eng_electrorheological_energies[1]
    assert pytest.approx(er1.dt_er) == 0.00065
    assert er1.sens_id == 85
    er2 = model.eng_electrorheological_energies[2]
    assert pytest.approx(er2.dt_er) == 0.00130
    assert er2.sens_id == 170


def test_m319_eng_electrorheological_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Electrorheological Energy Aliases Test
/ENG/ER_WORK/241
0.0018, 240
/ENG/EELECTRORHEOLOGICAL/242
0.0028, 241
/ENG/ELECTRORHEOLOGICAL_DISSIPATION/243
0.0038, 242
/ENG/EM_ELECTRORHEOLOGICAL/244
0.0048, 243
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 241 in model.eng_electrorheological_energies
    assert 242 in model.eng_electrorheological_energies
    assert 243 in model.eng_electrorheological_energies
    assert 244 in model.eng_electrorheological_energies
    assert pytest.approx(model.eng_electrorheological_energies[241].dt_er) == 0.0018
    assert pytest.approx(model.eng_electrorheological_energies[242].dt_er) == 0.0028
    assert pytest.approx(model.eng_electrorheological_energies[243].dt_er) == 0.0038
    assert pytest.approx(model.eng_electrorheological_energies[244].dt_er) == 0.0048


def test_m319_geneva_drive_mechanism_joint(tmp_path: Path):
    c1 = f"{1041:>10d}{1042:>10d}{1043:>10d}{1.55e7:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{40.0:>20.4f}{80.0:>20.4f}{6:>10d}{92.38:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Geneva Drive Mechanism Joint Fixed and Free Format Test
2022 0
/LAGMUL/GENEVA_DRIVE_MECHANISM_JOINT/1
Fixed Geneva Drive Mechanism Joint
{c1}
{c2}
/LAGMUL/GENEVA_DRIVE_MECHANISM_JOINT/2
Free Geneva Drive Mechanism Joint
1141, 1142, 1143, 1.75e7, 2, 2.8e-6
50.0, 100.0, 4, 141.42
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_geneva_drive_mechanism_joints
    assert 2 in model.lagmul_geneva_drive_mechanism_joints
    gdj1 = model.lagmul_geneva_drive_mechanism_joints[1]
    assert gdj1.node1 == 1041
    assert gdj1.node2 == 1042
    assert gdj1.node3 == 1043
    assert pytest.approx(gdj1.stiff) == 1.55e7
    assert gdj1.skew_id == 1
    assert pytest.approx(gdj1.tol) == 1.0e-6
    assert pytest.approx(gdj1.drive_radius) == 40.0
    assert pytest.approx(gdj1.wheel_radius) == 80.0
    assert gdj1.num_slots == 6
    assert pytest.approx(gdj1.center_dist) == 92.38

    gdj2 = model.lagmul_geneva_drive_mechanism_joints[2]
    assert gdj2.node1 == 1141
    assert gdj2.node2 == 1142
    assert gdj2.node3 == 1143
    assert pytest.approx(gdj2.stiff) == 1.75e7
    assert gdj2.skew_id == 2
    assert pytest.approx(gdj2.tol) == 2.8e-6
    assert pytest.approx(gdj2.drive_radius) == 50.0
    assert pytest.approx(gdj2.wheel_radius) == 100.0
    assert gdj2.num_slots == 4
    assert pytest.approx(gdj2.center_dist) == 141.42


def test_m319_geneva_drive_mechanism_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Geneva Drive Mechanism Joint Aliases Test
/GENEVA_DRIVE_MECHANISM_JOINT/245
1221, 1222, 1223, 8.8e6, 0, 1.0e-6
35.0, 70.0, 4, 98.99
/LAGMUL/GENEVA_DRIVE_MECHANISM/246
1224, 1225, 1226, 8.8e6, 0, 1.0e-6
35.0, 70.0, 4, 98.99
/GENEVA_DRIVE_MECHANISM/247
1227, 1228, 1229, 8.8e6, 0, 1.0e-6
35.0, 70.0, 4, 98.99
/LAGMUL/MALTESE_CROSS_MECHANISM/248
1230, 1231, 1232, 8.8e6, 0, 1.0e-6
35.0, 70.0, 4, 98.99
/MALTESE_CROSS_MECHANISM/249
1233, 1234, 1235, 8.8e6, 0, 1.0e-6
35.0, 70.0, 4, 98.99
/LAGMUL/GENEVA_INDEXING_JOINT/250
1236, 1237, 1238, 8.8e6, 0, 1.0e-6
35.0, 70.0, 4, 98.99
/GENEVA_INDEXING_JOINT/251
1239, 1240, 1241, 8.8e6, 0, 1.0e-6
35.0, 70.0, 4, 98.99
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 245 in model.lagmul_geneva_drive_mechanism_joints
    assert 246 in model.lagmul_geneva_drive_mechanism_joints
    assert 247 in model.lagmul_geneva_drive_mechanism_joints
    assert 248 in model.lagmul_geneva_drive_mechanism_joints
    assert 249 in model.lagmul_geneva_drive_mechanism_joints
    assert 250 in model.lagmul_geneva_drive_mechanism_joints
    assert 251 in model.lagmul_geneva_drive_mechanism_joints
    assert model.lagmul_geneva_drive_mechanism_joints[245].node1 == 1221
    assert model.lagmul_geneva_drive_mechanism_joints[246].node1 == 1224
    assert model.lagmul_geneva_drive_mechanism_joints[247].node1 == 1227
    assert model.lagmul_geneva_drive_mechanism_joints[248].node1 == 1230
    assert model.lagmul_geneva_drive_mechanism_joints[249].node1 == 1233
    assert model.lagmul_geneva_drive_mechanism_joints[250].node1 == 1236
    assert model.lagmul_geneva_drive_mechanism_joints[251].node1 == 1239


def test_m319_sensor_spring_normal_crackle_rate(tmp_path: Path):
    c1 = f"{1148:>10d}{8.8e7:>20.4f}{0.0245:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Crackle Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_NORMAL_CRACKLE_RATE/1
Fixed Spring Normal Crackle Rate Sensor
{c1}
/SENSOR/SPRING_NORMAL_CRACKLE_RATE/2
Free Spring Normal Crackle Rate Sensor
1149, 1.15e8, 0.0345
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_crackle_rates
    assert 2 in model.sensor_spring_normal_crackle_rates
    s1 = model.sensor_spring_normal_crackle_rates[1]
    assert s1.spring_id == 1148
    assert pytest.approx(s1.jnorm_pop_max) == 8.8e7
    assert pytest.approx(s1.t_delay) == 0.0245

    s2 = model.sensor_spring_normal_crackle_rates[2]
    assert s2.spring_id == 1149
    assert pytest.approx(s2.jnorm_pop_max) == 1.15e8
    assert pytest.approx(s2.t_delay) == 0.0345

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_NORMAL_CRACKLE_RATE"
    assert model.sensors[1].kind == "SPRING_NORMAL_CRACKLE_RATE"


def test_m319_sensor_spring_normal_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Crackle Rate Aliases Test
/SENSOR/SPRING_NORM_CRACKLE_RATE/163
1178, 5.55e7, 0.0062
/SENSOR/SPRING_RATE_CRACKLE_NORM/164
1179, 5.75e7, 0.0072
/SENSOR/NORMAL_CRACKLE_RATE_SPRING/165
1180, 5.95e7, 0.0082
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 163 in model.sensor_spring_normal_crackle_rates
    assert 164 in model.sensor_spring_normal_crackle_rates
    assert 165 in model.sensor_spring_normal_crackle_rates
    assert model.sensor_spring_normal_crackle_rates[163].spring_id == 1178
    assert pytest.approx(model.sensor_spring_normal_crackle_rates[163].jnorm_pop_max) == 5.55e7
    assert model.sensor_spring_normal_crackle_rates[164].spring_id == 1179
    assert model.sensor_spring_normal_crackle_rates[165].spring_id == 1180

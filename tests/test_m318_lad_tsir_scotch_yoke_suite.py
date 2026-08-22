"""Tests for Milestone M318: LadTransverseShearInteractionRate Failure Model, EngMagnetorheologicalEnergy, ScotchYokeMechanismJoint, and SensorSpringTotalSnapRate."""

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


def test_m318_fail_lad_transverse_shear_interaction_rate_fixed(tmp_path: Path):
    c1 = f"{4.25:>20.4f}{68.0:>20.4f}{0.085:>20.4f}{1.65:>20.4f}{0.975:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1272:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse-Shear Interaction Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_SHEAR_INTERACTION_RATE/1272
Ladeveze Transverse-Shear Interaction Rate Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1272 in model.fail_ladtransverseshearinteractionrates
    fltsir = model.fail_ladtransverseshearinteractionrates[1272]
    assert pytest.approx(fltsir.y0_tsir) == 4.25
    assert pytest.approx(fltsir.yc_tsir) == 68.0
    assert pytest.approx(fltsir.gamma_rate_tsir) == 0.085
    assert pytest.approx(fltsir.p_rate_tsir) == 1.65
    assert pytest.approx(fltsir.d_tsir_max) == 0.975
    assert fltsir.ifail_sh == 1
    assert fltsir.ifail_so == 2
    assert fltsir.fail_id == 1272
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_SHEAR_INTERACTION_RATE"


def test_m318_fail_lad_transverse_shear_interaction_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse-Shear Interaction Rate Free Format Test
/FAIL/LAD_TRANSVERSE_SHEAR_INTERACTION_RATE/1273
5.15, 76.5, 0.105, 1.95, 0.955
1, 1
1273
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1273 in model.fail_ladtransverseshearinteractionrates
    fltsir = model.fail_ladtransverseshearinteractionrates[1273]
    assert pytest.approx(fltsir.y0_tsir) == 5.15
    assert pytest.approx(fltsir.yc_tsir) == 76.5
    assert pytest.approx(fltsir.gamma_rate_tsir) == 0.105
    assert pytest.approx(fltsir.p_rate_tsir) == 1.95
    assert pytest.approx(fltsir.d_tsir_max) == 0.955
    assert fltsir.fail_id == 1273


def test_m318_fail_lad_transverse_shear_interaction_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse-Shear Interaction Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_SHEAR_INTERACTION_RATE/1274
3.5, 45.0, 0.065, 1.35, 0.94
1, 1
/FAIL/LAD_TSIR/1275
3.5, 45.0, 0.065, 1.35, 0.94
1, 1
/FAIL/LAD_TSIR_MODEL/1276
3.5, 45.0, 0.065, 1.35, 0.94
1, 1
/FAIL/LAD_TSIR_LAW/1277
3.5, 45.0, 0.065, 1.35, 0.94
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLING/1278
3.5, 45.0, 0.065, 1.35, 0.94
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1274 in model.fail_ladtransverseshearinteractionrates
    assert 1275 in model.fail_ladtransverseshearinteractionrates
    assert 1276 in model.fail_ladtransverseshearinteractionrates
    assert 1277 in model.fail_ladtransverseshearinteractionrates
    assert 1278 in model.fail_ladtransverseshearinteractionrates
    assert len(model.raw_fails) == 5


def test_m318_eng_magnetorheological_energy(tmp_path: Path):
    c1 = f"{0.00055:>20.6f}{75:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Magnetorheological Energy Fixed and Free Format Test
2022 0
/ENG/MAGNETORHEOLOGICAL_ENERGY/1
Fixed Magnetorheological Energy Output
{c1}
/ENG/MAGNETORHEOLOGICAL_ENERGY/2
Free Magnetorheological Energy Output
0.00110, 150
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_magnetorheological_energies
    assert 2 in model.eng_magnetorheological_energies
    mr1 = model.eng_magnetorheological_energies[1]
    assert pytest.approx(mr1.dt_mr) == 0.00055
    assert mr1.sens_id == 75
    mr2 = model.eng_magnetorheological_energies[2]
    assert pytest.approx(mr2.dt_mr) == 0.00110
    assert mr2.sens_id == 150


def test_m318_eng_magnetorheological_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Magnetorheological Energy Aliases Test
/ENG/MR_WORK/231
0.0015, 230
/ENG/EMAGNETORHEOLOGICAL/232
0.0025, 231
/ENG/MAGNETORHEOLOGICAL_DISSIPATION/233
0.0035, 232
/ENG/EM_MAGNETORHEOLOGICAL/234
0.0045, 233
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 231 in model.eng_magnetorheological_energies
    assert 232 in model.eng_magnetorheological_energies
    assert 233 in model.eng_magnetorheological_energies
    assert 234 in model.eng_magnetorheological_energies
    assert pytest.approx(model.eng_magnetorheological_energies[231].dt_mr) == 0.0015
    assert pytest.approx(model.eng_magnetorheological_energies[232].dt_mr) == 0.0025
    assert pytest.approx(model.eng_magnetorheological_energies[233].dt_mr) == 0.0035
    assert pytest.approx(model.eng_magnetorheological_energies[234].dt_mr) == 0.0045


def test_m318_scotch_yoke_mechanism_joint(tmp_path: Path):
    c1 = f"{1031:>10d}{1032:>10d}{1033:>10d}{1.45e7:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{35.0:>20.4f}{20.0:>20.4f}{120.0:>20.4f}{45.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Scotch Yoke Mechanism Joint Fixed and Free Format Test
2022 0
/LAGMUL/SCOTCH_YOKE_MECHANISM_JOINT/1
Fixed Scotch Yoke Mechanism Joint
{c1}
{c2}
/LAGMUL/SCOTCH_YOKE_MECHANISM_JOINT/2
Free Scotch Yoke Mechanism Joint
1131, 1132, 1133, 1.65e7, 2, 2.5e-6
42.0, 24.0, 150.0, 30.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_scotch_yoke_mechanism_joints
    assert 2 in model.lagmul_scotch_yoke_mechanism_joints
    syj1 = model.lagmul_scotch_yoke_mechanism_joints[1]
    assert syj1.node1 == 1031
    assert syj1.node2 == 1032
    assert syj1.node3 == 1033
    assert pytest.approx(syj1.stiff) == 1.45e7
    assert syj1.skew_id == 1
    assert pytest.approx(syj1.tol) == 1.0e-6
    assert pytest.approx(syj1.crank_radius) == 35.0
    assert pytest.approx(syj1.slot_width) == 20.0
    assert pytest.approx(syj1.stroke_limit) == 120.0
    assert pytest.approx(syj1.yoke_angle) == 45.0

    syj2 = model.lagmul_scotch_yoke_mechanism_joints[2]
    assert syj2.node1 == 1131
    assert syj2.node2 == 1132
    assert syj2.node3 == 1133
    assert pytest.approx(syj2.stiff) == 1.65e7
    assert syj2.skew_id == 2
    assert pytest.approx(syj2.tol) == 2.5e-6
    assert pytest.approx(syj2.crank_radius) == 42.0
    assert pytest.approx(syj2.slot_width) == 24.0
    assert pytest.approx(syj2.stroke_limit) == 150.0
    assert pytest.approx(syj2.yoke_angle) == 30.0


def test_m318_scotch_yoke_mechanism_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Scotch Yoke Mechanism Joint Aliases Test
/SCOTCH_YOKE_MECHANISM_JOINT/235
1201, 1202, 1203, 9.5e6, 0, 1.0e-6
30.0, 15.0, 100.0, 0.0
/LAGMUL/SCOTCH_YOKE_MECHANISM/236
1204, 1205, 1206, 9.5e6, 0, 1.0e-6
30.0, 15.0, 100.0, 0.0
/SCOTCH_YOKE_MECHANISM/237
1207, 1208, 1209, 9.5e6, 0, 1.0e-6
30.0, 15.0, 100.0, 0.0
/SCOTCH_YOKE_JOINT/238
1210, 1211, 1212, 9.5e6, 0, 1.0e-6
30.0, 15.0, 100.0, 0.0
/SLOTTED_LINK_MECHANISM/239
1213, 1214, 1215, 9.5e6, 0, 1.0e-6
30.0, 15.0, 100.0, 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 235 in model.lagmul_scotch_yoke_mechanism_joints
    assert 236 in model.lagmul_scotch_yoke_mechanism_joints
    assert 237 in model.lagmul_scotch_yoke_mechanism_joints
    assert 238 in model.lagmul_scotch_yoke_mechanism_joints
    assert 239 in model.lagmul_scotch_yoke_mechanism_joints
    assert model.lagmul_scotch_yoke_mechanism_joints[235].node1 == 1201
    assert model.lagmul_scotch_yoke_mechanism_joints[236].node1 == 1204
    assert model.lagmul_scotch_yoke_mechanism_joints[237].node1 == 1207
    assert model.lagmul_scotch_yoke_mechanism_joints[238].node1 == 1210
    assert model.lagmul_scotch_yoke_mechanism_joints[239].node1 == 1213


def test_m318_sensor_spring_total_snap_rate(tmp_path: Path):
    c1 = f"{1138:>10d}{7.5e7:>20.4f}{0.0215:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Snap Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TOTAL_SNAP_RATE/1
Fixed Spring Total Snap Rate Sensor
{c1}
/SENSOR/SPRING_TOTAL_SNAP_RATE/2
Free Spring Total Snap Rate Sensor
1139, 9.8e7, 0.0315
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_snap_rates
    assert 2 in model.sensor_spring_total_snap_rates
    s1 = model.sensor_spring_total_snap_rates[1]
    assert s1.spring_id == 1138
    assert pytest.approx(s1.jtot_crackle_max) == 7.5e7
    assert pytest.approx(s1.t_delay) == 0.0215

    s2 = model.sensor_spring_total_snap_rates[2]
    assert s2.spring_id == 1139
    assert pytest.approx(s2.jtot_crackle_max) == 9.8e7
    assert pytest.approx(s2.t_delay) == 0.0315

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TOTAL_SNAP_RATE"
    assert model.sensors[1].kind == "SPRING_TOTAL_SNAP_RATE"


def test_m318_sensor_spring_total_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Snap Rate Aliases Test
/SENSOR/SPRING_TOT_SNAP_RATE/153
1168, 4.55e7, 0.0055
/SENSOR/SPRING_RATE_SNAP_TOT/154
1169, 4.75e7, 0.0065
/SENSOR/TOTAL_SNAP_RATE_SPRING/155
1170, 4.95e7, 0.0075
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 153 in model.sensor_spring_total_snap_rates
    assert 154 in model.sensor_spring_total_snap_rates
    assert 155 in model.sensor_spring_total_snap_rates
    assert model.sensor_spring_total_snap_rates[153].spring_id == 1168
    assert pytest.approx(model.sensor_spring_total_snap_rates[153].jtot_crackle_max) == 4.55e7
    assert model.sensor_spring_total_snap_rates[154].spring_id == 1169
    assert model.sensor_spring_total_snap_rates[155].spring_id == 1170

"""Tests for Milestone M321: LadNonlocalGradientRate Failure Model, EngFerroelectricEnergy, BennettLinkageJoint, and SensorSpringTotalCrackleRate."""

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


def test_m321_fail_lad_nonlocal_gradient_rate_fixed(tmp_path: Path):
    c1 = f"{4.15:>20.4f}{64.0:>20.4f}{1.35:>20.4f}{0.0025:>20.4f}{0.970:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1299:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Nonlocal Gradient Rate Fixed Format Test
2022 0
/FAIL/LAD_NONLOCAL_GRADIENT_RATE/1299
Ladeveze Nonlocal Gradient Rate Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1299 in model.fail_ladnonlocalgradientrates
    fnlgr = model.fail_ladnonlocalgradientrates[1299]
    assert pytest.approx(fnlgr.y0_nlgr) == 4.15
    assert pytest.approx(fnlgr.yc_nlgr) == 64.0
    assert pytest.approx(fnlgr.lc_char) == 1.35
    assert pytest.approx(fnlgr.tau_nlgr) == 0.0025
    assert pytest.approx(fnlgr.d_nlgr_max) == 0.970
    assert fnlgr.ifail_sh == 1
    assert fnlgr.ifail_so == 2
    assert fnlgr.fail_id == 1299
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_NONLOCAL_GRADIENT_RATE"


def test_m321_fail_lad_nonlocal_gradient_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Nonlocal Gradient Rate Free Format Test
/FAIL/LAD_NONLOCAL_GRADIENT_RATE/1300
5.25, 72.0, 1.95, 0.0045, 0.955
1, 1
1300
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1300 in model.fail_ladnonlocalgradientrates
    fnlgr = model.fail_ladnonlocalgradientrates[1300]
    assert pytest.approx(fnlgr.y0_nlgr) == 5.25
    assert pytest.approx(fnlgr.yc_nlgr) == 72.0
    assert pytest.approx(fnlgr.lc_char) == 1.95
    assert pytest.approx(fnlgr.tau_nlgr) == 0.0045
    assert pytest.approx(fnlgr.d_nlgr_max) == 0.955
    assert fnlgr.fail_id == 1300


def test_m321_fail_lad_nonlocal_gradient_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Nonlocal Gradient Rate Aliases Test
/FAIL/LADEVEZE_NONLOCAL_GRADIENT_RATE/1301
3.4, 46.0, 1.2, 0.0015, 0.95
1, 1
/FAIL/LAD_NLGR/1302
3.4, 46.0, 1.2, 0.0015, 0.95
1, 1
/FAIL/LAD_NLGR_MODEL/1303
3.4, 46.0, 1.2, 0.0015, 0.95
1, 1
/FAIL/LAD_NLGR_LAW/1304
3.4, 46.0, 1.2, 0.0015, 0.95
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_GRADIENT/1305
3.4, 46.0, 1.2, 0.0015, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1301 in model.fail_ladnonlocalgradientrates
    assert 1302 in model.fail_ladnonlocalgradientrates
    assert 1303 in model.fail_ladnonlocalgradientrates
    assert 1304 in model.fail_ladnonlocalgradientrates
    assert 1305 in model.fail_ladnonlocalgradientrates
    assert len(model.raw_fails) == 5


def test_m321_eng_ferroelectric_energy(tmp_path: Path):
    c1 = f"{0.00055:>20.6f}{80:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ferroelectric Energy Fixed and Free Format Test
2022 0
/ENG/FERROELECTRIC_ENERGY/1
Fixed Ferroelectric Energy Output
{c1}
/ENG/FERROELECTRIC_ENERGY/2
Free Ferroelectric Energy Output
0.00115, 160
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_ferroelectric_energies
    assert 2 in model.eng_ferroelectric_energies
    fe1 = model.eng_ferroelectric_energies[1]
    assert pytest.approx(fe1.dt_fe) == 0.00055
    assert fe1.sens_id == 80
    fe2 = model.eng_ferroelectric_energies[2]
    assert pytest.approx(fe2.dt_fe) == 0.00115
    assert fe2.sens_id == 160


def test_m321_eng_ferroelectric_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ferroelectric Energy Aliases Test
/ENG/FE_WORK/261
0.0017, 260
/ENG/EFERROELECTRIC/262
0.0027, 261
/ENG/FERROELECTRIC_DISSIPATION/263
0.0037, 262
/ENG/EM_FERROELECTRIC/264
0.0047, 263
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 261 in model.eng_ferroelectric_energies
    assert 262 in model.eng_ferroelectric_energies
    assert 263 in model.eng_ferroelectric_energies
    assert 264 in model.eng_ferroelectric_energies
    assert pytest.approx(model.eng_ferroelectric_energies[261].dt_fe) == 0.0017
    assert pytest.approx(model.eng_ferroelectric_energies[262].dt_fe) == 0.0027
    assert pytest.approx(model.eng_ferroelectric_energies[263].dt_fe) == 0.0037
    assert pytest.approx(model.eng_ferroelectric_energies[264].dt_fe) == 0.0047


def test_m321_bennett_linkage_joint(tmp_path: Path):
    c1 = f"{1061:>10d}{1062:>10d}{1063:>10d}{1.68e7:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{55.0:>20.4f}{75.0:>20.4f}{30.0:>20.4f}{45.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Bennett Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/BENNETT_LINKAGE_JOINT/1
Fixed Bennett Spatial Mechanism Joint
{c1}
{c2}
/LAGMUL/BENNETT_LINKAGE_JOINT/2
Free Bennett Spatial Mechanism Joint
1161, 1162, 1163, 1.88e7, 2, 2.6e-6
65.0, 85.0, 35.0, 50.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_bennett_linkage_joints
    assert 2 in model.lagmul_bennett_linkage_joints
    bj1 = model.lagmul_bennett_linkage_joints[1]
    assert bj1.node1 == 1061
    assert bj1.node2 == 1062
    assert bj1.node3 == 1063
    assert pytest.approx(bj1.stiff) == 1.68e7
    assert bj1.skew_id == 1
    assert pytest.approx(bj1.tol) == 1.0e-6
    assert pytest.approx(bj1.link_len_a) == 55.0
    assert pytest.approx(bj1.link_len_b) == 75.0
    assert pytest.approx(bj1.twist_angle_alpha) == 30.0
    assert pytest.approx(bj1.twist_angle_beta) == 45.0

    bj2 = model.lagmul_bennett_linkage_joints[2]
    assert bj2.node1 == 1161
    assert bj2.node2 == 1162
    assert bj2.node3 == 1163
    assert pytest.approx(bj2.stiff) == 1.88e7
    assert bj2.skew_id == 2
    assert pytest.approx(bj2.tol) == 2.6e-6
    assert pytest.approx(bj2.link_len_a) == 65.0
    assert pytest.approx(bj2.link_len_b) == 85.0
    assert pytest.approx(bj2.twist_angle_alpha) == 35.0
    assert pytest.approx(bj2.twist_angle_beta) == 50.0


def test_m321_bennett_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Bennett Linkage Joint Aliases Test
/BENNETT_LINKAGE_JOINT/265
1261, 1262, 1263, 9.8e6, 0, 1.0e-6
50.0, 70.0, 25.0, 40.0
/LAGMUL/BENNETT_LINKAGE/266
1264, 1265, 1266, 9.8e6, 0, 1.0e-6
50.0, 70.0, 25.0, 40.0
/BENNETT_LINKAGE/267
1267, 1268, 1269, 9.8e6, 0, 1.0e-6
50.0, 70.0, 25.0, 40.0
/BENNETT_SPATIAL_MECHANISM/268
1270, 1271, 1272, 9.8e6, 0, 1.0e-6
50.0, 70.0, 25.0, 40.0
/BENNETT_4R_MECHANISM/269
1273, 1274, 1275, 9.8e6, 0, 1.0e-6
50.0, 70.0, 25.0, 40.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 265 in model.lagmul_bennett_linkage_joints
    assert 266 in model.lagmul_bennett_linkage_joints
    assert 267 in model.lagmul_bennett_linkage_joints
    assert 268 in model.lagmul_bennett_linkage_joints
    assert 269 in model.lagmul_bennett_linkage_joints
    assert model.lagmul_bennett_linkage_joints[265].node1 == 1261
    assert model.lagmul_bennett_linkage_joints[266].node1 == 1264
    assert model.lagmul_bennett_linkage_joints[267].node1 == 1267
    assert model.lagmul_bennett_linkage_joints[268].node1 == 1270
    assert model.lagmul_bennett_linkage_joints[269].node1 == 1273


def test_m321_sensor_spring_total_crackle_rate(tmp_path: Path):
    c1 = f"{1168:>10d}{9.2e7:>20.4f}{0.0215:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Crackle Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TOTAL_CRACKLE_RATE/1
Fixed Spring Total Crackle Rate Sensor
{c1}
/SENSOR/SPRING_TOTAL_CRACKLE_RATE/2
Free Spring Total Crackle Rate Sensor
1169, 1.25e8, 0.0315
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_crackle_rates
    assert 2 in model.sensor_spring_total_crackle_rates
    s1 = model.sensor_spring_total_crackle_rates[1]
    assert s1.spring_id == 1168
    assert pytest.approx(s1.jtot_pop_max) == 9.2e7
    assert pytest.approx(s1.t_delay) == 0.0215

    s2 = model.sensor_spring_total_crackle_rates[2]
    assert s2.spring_id == 1169
    assert pytest.approx(s2.jtot_pop_max) == 1.25e8
    assert pytest.approx(s2.t_delay) == 0.0315

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TOTAL_CRACKLE_RATE"
    assert model.sensors[1].kind == "SPRING_TOTAL_CRACKLE_RATE"


def test_m321_sensor_spring_total_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Crackle Rate Aliases Test
/SENSOR/SPRING_TOT_CRACKLE_RATE/183
1198, 7.25e7, 0.0068
/SENSOR/SPRING_RATE_CRACKLE_TOT/184
1199, 7.45e7, 0.0078
/SENSOR/TOTAL_CRACKLE_RATE_SPRING/185
1200, 7.65e7, 0.0088
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 183 in model.sensor_spring_total_crackle_rates
    assert 184 in model.sensor_spring_total_crackle_rates
    assert 185 in model.sensor_spring_total_crackle_rates
    assert model.sensor_spring_total_crackle_rates[183].spring_id == 1198
    assert pytest.approx(model.sensor_spring_total_crackle_rates[183].jtot_pop_max) == 7.25e7
    assert model.sensor_spring_total_crackle_rates[184].spring_id == 1199
    assert model.sensor_spring_total_crackle_rates[185].spring_id == 1200

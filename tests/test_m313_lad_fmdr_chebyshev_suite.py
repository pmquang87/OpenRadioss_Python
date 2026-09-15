"""Tests for Milestone M313: LadFiberMatrixDebondRate Failure Model, EngMagnetogalvanicEnergy, ChebyshevLambdaLinkageJoint, and SensorSpringTransverseJerkRate."""

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


def test_m313_fail_lad_fiber_matrix_debond_rate_fixed(tmp_path: Path):
    c1 = f"{2.15:>20.4f}{31.5:>20.4f}{0.045:>20.4f}{1.25:>20.4f}{0.985:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1225:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Fiber Matrix Debond Rate Fixed Format Test
2022 0
/FAIL/LAD_FIBER_MATRIX_DEBOND_RATE/1225
Ladeveze Fiber Matrix Debond Rate Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1225 in model.fail_ladfibermatrixdebondrates
    flfmdr = model.fail_ladfibermatrixdebondrates[1225]
    assert pytest.approx(flfmdr.y0_fmdr) == 2.15
    assert pytest.approx(flfmdr.yc_fmdr) == 31.5
    assert pytest.approx(flfmdr.c_rate_fmdr) == 0.045
    assert pytest.approx(flfmdr.p_rate_fmdr) == 1.25
    assert pytest.approx(flfmdr.d_fmdr_max) == 0.985
    assert flfmdr.ifail_sh == 1
    assert flfmdr.ifail_so == 2
    assert flfmdr.fail_id == 1225
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_FIBER_MATRIX_DEBOND_RATE"


def test_m313_fail_lad_fiber_matrix_debond_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Fiber Matrix Debond Rate Free Format Test
/FAIL/LAD_FIBER_MATRIX_DEBOND_RATE/1226
2.95, 42.0, 0.065, 1.4, 0.972
1, 1
1226
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1226 in model.fail_ladfibermatrixdebondrates
    flfmdr = model.fail_ladfibermatrixdebondrates[1226]
    assert pytest.approx(flfmdr.y0_fmdr) == 2.95
    assert pytest.approx(flfmdr.yc_fmdr) == 42.0
    assert pytest.approx(flfmdr.c_rate_fmdr) == 0.065
    assert pytest.approx(flfmdr.p_rate_fmdr) == 1.4
    assert pytest.approx(flfmdr.d_fmdr_max) == 0.972
    assert flfmdr.fail_id == 1226


def test_m313_fail_lad_fiber_matrix_debond_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Fiber Matrix Debond Rate Aliases Test
/FAIL/LADEVEZE_FIBER_MATRIX_DEBOND_RATE/1227
1.8, 26.0, 0.035, 1.1, 0.95
1, 1
/FAIL/LAD_FMDR/1228
1.8, 26.0, 0.035, 1.1, 0.95
1, 1
/FAIL/LAD_FMDR_MODEL/1229
1.8, 26.0, 0.035, 1.1, 0.95
1, 1
/FAIL/LAD_FMDR_LAW/1230
1.8, 26.0, 0.035, 1.1, 0.95
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DEBONDING/1231
1.8, 26.0, 0.035, 1.1, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1227 in model.fail_ladfibermatrixdebondrates
    assert 1228 in model.fail_ladfibermatrixdebondrates
    assert 1229 in model.fail_ladfibermatrixdebondrates
    assert 1230 in model.fail_ladfibermatrixdebondrates
    assert 1231 in model.fail_ladfibermatrixdebondrates
    assert len(model.raw_fails) == 5


def test_m313_eng_magnetogalvanic_energy(tmp_path: Path):
    c1 = f"{0.00032:>20.6f}{52:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Magnetogalvanic Energy Fixed and Free Format Test
2022 0
/ENG/MAGNETOGALVANIC_ENERGY/1
Fixed Magnetogalvanic Energy Output
{c1}
/ENG/MAGNETOGALVANIC_ENERGY/2
Free Magnetogalvanic Energy Output
0.00064, 104
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_magnetogalvanic_energies
    assert 2 in model.eng_magnetogalvanic_energies
    mge1 = model.eng_magnetogalvanic_energies[1]
    assert pytest.approx(mge1.dt_mge) == 0.00032
    assert mge1.sens_id == 52
    mge2 = model.eng_magnetogalvanic_energies[2]
    assert pytest.approx(mge2.dt_mge) == 0.00064
    assert mge2.sens_id == 104


def test_m313_eng_magnetogalvanic_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Magnetogalvanic Energy Aliases Test
/ENG/MGE_WORK/181
0.001, 180
/ENG/EMAGNETOGALVANIC/182
0.002, 181
/ENG/MAGNETOGALVANIC_DISSIPATION/183
0.003, 182
/ENG/EM_MAGNETOGALVANIC/184
0.004, 183
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 181 in model.eng_magnetogalvanic_energies
    assert 182 in model.eng_magnetogalvanic_energies
    assert 183 in model.eng_magnetogalvanic_energies
    assert 184 in model.eng_magnetogalvanic_energies
    assert pytest.approx(model.eng_magnetogalvanic_energies[181].dt_mge) == 0.001
    assert pytest.approx(model.eng_magnetogalvanic_energies[182].dt_mge) == 0.002
    assert pytest.approx(model.eng_magnetogalvanic_energies[183].dt_mge) == 0.003
    assert pytest.approx(model.eng_magnetogalvanic_energies[184].dt_mge) == 0.004


def test_m313_chebyshev_lambda_linkage_joint(tmp_path: Path):
    c1 = f"{995:>10d}{996:>10d}{997:>10d}{9.8e6:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{100.0:>20.4f}{50.0:>20.4f}{125.0:>20.4f}{125.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Chebyshev Lambda Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/CHEBYSHEV_LAMBDA_LINKAGE_JOINT/1
Fixed Chebyshev Lambda Linkage Joint
{c1}
{c2}
/LAGMUL/CHEBYSHEV_LAMBDA_LINKAGE_JOINT/2
Free Chebyshev Lambda Linkage Joint
1095, 1096, 1097, 1.15e7, 2, 2.5e-6
120.0, 60.0, 150.0, 150.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_chebyshev_lambda_linkage_joints
    assert 2 in model.lagmul_chebyshev_lambda_linkage_joints
    cllj1 = model.lagmul_chebyshev_lambda_linkage_joints[1]
    assert cllj1.node1 == 995
    assert cllj1.node2 == 996
    assert cllj1.node3 == 997
    assert pytest.approx(cllj1.stiff) == 9.8e6
    assert cllj1.skew_id == 1
    assert pytest.approx(cllj1.tol) == 1.0e-6
    assert pytest.approx(cllj1.ground_dist) == 100.0
    assert pytest.approx(cllj1.crank_len) == 50.0
    assert pytest.approx(cllj1.coupler_len) == 125.0
    assert pytest.approx(cllj1.rocker_len) == 125.0

    cllj2 = model.lagmul_chebyshev_lambda_linkage_joints[2]
    assert cllj2.node1 == 1095
    assert cllj2.node2 == 1096
    assert cllj2.node3 == 1097
    assert pytest.approx(cllj2.stiff) == 1.15e7
    assert cllj2.skew_id == 2
    assert pytest.approx(cllj2.tol) == 2.5e-6
    assert pytest.approx(cllj2.ground_dist) == 120.0
    assert pytest.approx(cllj2.crank_len) == 60.0
    assert pytest.approx(cllj2.coupler_len) == 150.0
    assert pytest.approx(cllj2.rocker_len) == 150.0


def test_m313_chebyshev_lambda_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Chebyshev Lambda Linkage Joint Aliases Test
/CHEBYSHEV_LAMBDA_LINKAGE_JOINT/185
1151, 1152, 1153, 7.3e6, 0, 1.0e-6
80.0, 40.0, 100.0, 100.0
/LAGMUL/CHEBYSHEV_LAMBDA_LINKAGE/186
1154, 1155, 1156, 7.3e6, 0, 1.0e-6
80.0, 40.0, 100.0, 100.0
/CHEBYSHEV_LAMBDA_LINKAGE/187
1157, 1158, 1159, 7.3e6, 0, 1.0e-6
80.0, 40.0, 100.0, 100.0
/CHEBYSHEV_WALKING_MECHANISM/188
1160, 1161, 1162, 7.3e6, 0, 1.0e-6
80.0, 40.0, 100.0, 100.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 185 in model.lagmul_chebyshev_lambda_linkage_joints
    assert 186 in model.lagmul_chebyshev_lambda_linkage_joints
    assert 187 in model.lagmul_chebyshev_lambda_linkage_joints
    assert 188 in model.lagmul_chebyshev_lambda_linkage_joints
    assert model.lagmul_chebyshev_lambda_linkage_joints[185].node1 == 1151
    assert model.lagmul_chebyshev_lambda_linkage_joints[186].node1 == 1154
    assert model.lagmul_chebyshev_lambda_linkage_joints[187].node1 == 1157
    assert model.lagmul_chebyshev_lambda_linkage_joints[188].node1 == 1160


def test_m313_sensor_spring_transverse_jerk_rate(tmp_path: Path):
    c1 = f"{1088:>10d}{5.1e7:>20.4f}{0.0155:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Jerk Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_JERK_RATE/1
Fixed Spring Transverse Jerk Rate Sensor
{c1}
/SENSOR/SPRING_TRANSVERSE_JERK_RATE/2
Free Spring Transverse Jerk Rate Sensor
1089, 7.2e7, 0.0210
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_jerk_rates
    assert 2 in model.sensor_spring_transverse_jerk_rates
    s1 = model.sensor_spring_transverse_jerk_rates[1]
    assert s1.spring_id == 1088
    assert pytest.approx(s1.jtrans_snap_max) == 5.1e7
    assert pytest.approx(s1.t_delay) == 0.0155

    s2 = model.sensor_spring_transverse_jerk_rates[2]
    assert s2.spring_id == 1089
    assert pytest.approx(s2.jtrans_snap_max) == 7.2e7
    assert pytest.approx(s2.t_delay) == 0.0210

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_JERK_RATE"
    assert model.sensors[1].kind == "SPRING_TRANSVERSE_JERK_RATE"


def test_m313_sensor_spring_transverse_jerk_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Jerk Rate Aliases Test
/SENSOR/SPRING_TRANS_JERK_RATE/131
1118, 3.15e7, 0.0039
/SENSOR/SPRING_RATE_JERK_TRANS/132
1119, 3.35e7, 0.0049
/SENSOR/TRANSVERSE_JERK_RATE_SPRING/133
1120, 3.55e7, 0.0059
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 131 in model.sensor_spring_transverse_jerk_rates
    assert 132 in model.sensor_spring_transverse_jerk_rates
    assert 133 in model.sensor_spring_transverse_jerk_rates
    assert model.sensor_spring_transverse_jerk_rates[131].spring_id == 1118
    assert pytest.approx(model.sensor_spring_transverse_jerk_rates[131].jtrans_snap_max) == 3.15e7
    assert model.sensor_spring_transverse_jerk_rates[132].spring_id == 1119
    assert model.sensor_spring_transverse_jerk_rates[133].spring_id == 1120

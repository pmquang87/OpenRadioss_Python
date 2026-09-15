"""Tests for Milestone M315: LadTransverseCompressionRate Failure Model, EngThermophononicEnergy, FourBarDoubleCrankJoint, and SensorSpringBendingSnapRate."""

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


def test_m315_fail_lad_transverse_compression_rate_fixed(tmp_path: Path):
    c1 = f"{2.55:>20.4f}{36.5:>20.4f}{0.062:>20.4f}{1.42:>20.4f}{0.984:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1245:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Compression Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_COMPRESSION_RATE/1245
Ladeveze Transverse Compression Rate Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1245 in model.fail_ladtransversecompressionrates
    fltcr = model.fail_ladtransversecompressionrates[1245]
    assert pytest.approx(fltcr.y0_tcr) == 2.55
    assert pytest.approx(fltcr.yc_tcr) == 36.5
    assert pytest.approx(fltcr.c_rate_tcr) == 0.062
    assert pytest.approx(fltcr.p_rate_tcr) == 1.42
    assert pytest.approx(fltcr.d_tcr_max) == 0.984
    assert fltcr.ifail_sh == 1
    assert fltcr.ifail_so == 2
    assert fltcr.fail_id == 1245
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_COMPRESSION_RATE"


def test_m315_fail_lad_transverse_compression_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Compression Rate Free Format Test
/FAIL/LAD_TRANSVERSE_COMPRESSION_RATE/1246
3.35, 47.0, 0.082, 1.6, 0.968
1, 1
1246
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1246 in model.fail_ladtransversecompressionrates
    fltcr = model.fail_ladtransversecompressionrates[1246]
    assert pytest.approx(fltcr.y0_tcr) == 3.35
    assert pytest.approx(fltcr.yc_tcr) == 47.0
    assert pytest.approx(fltcr.c_rate_tcr) == 0.082
    assert pytest.approx(fltcr.p_rate_tcr) == 1.6
    assert pytest.approx(fltcr.d_tcr_max) == 0.968
    assert fltcr.fail_id == 1246


def test_m315_fail_lad_transverse_compression_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Compression Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_COMPRESSION_RATE/1247
2.1, 29.0, 0.045, 1.3, 0.95
1, 1
/FAIL/LAD_TCR/1248
2.1, 29.0, 0.045, 1.3, 0.95
1, 1
/FAIL/LAD_TCR_MODEL/1249
2.1, 29.0, 0.045, 1.3, 0.95
1, 1
/FAIL/LAD_TCR_LAW/1250
2.1, 29.0, 0.045, 1.3, 0.95
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_CRUSH/1251
2.1, 29.0, 0.045, 1.3, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1247 in model.fail_ladtransversecompressionrates
    assert 1248 in model.fail_ladtransversecompressionrates
    assert 1249 in model.fail_ladtransversecompressionrates
    assert 1250 in model.fail_ladtransversecompressionrates
    assert 1251 in model.fail_ladtransversecompressionrates
    assert len(model.raw_fails) == 5


def test_m315_eng_thermophononic_energy(tmp_path: Path):
    c1 = f"{0.00038:>20.6f}{58:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Thermophononic Energy Fixed and Free Format Test
2022 0
/ENG/THERMOPHONONIC_ENERGY/1
Fixed Thermophononic Energy Output
{c1}
/ENG/THERMOPHONONIC_ENERGY/2
Free Thermophononic Energy Output
0.00076, 116
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_thermophononic_energies
    assert 2 in model.eng_thermophononic_energies
    tph1 = model.eng_thermophononic_energies[1]
    assert pytest.approx(tph1.dt_tph) == 0.00038
    assert tph1.sens_id == 58
    tph2 = model.eng_thermophononic_energies[2]
    assert pytest.approx(tph2.dt_tph) == 0.00076
    assert tph2.sens_id == 116


def test_m315_eng_thermophononic_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Thermophononic Energy Aliases Test
/ENG/TPH_WORK/201
0.001, 200
/ENG/ETHERMOPHONONIC/202
0.002, 201
/ENG/THERMOPHONONIC_DISSIPATION/203
0.003, 202
/ENG/EM_THERMOPHONONIC/204
0.004, 203
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 201 in model.eng_thermophononic_energies
    assert 202 in model.eng_thermophononic_energies
    assert 203 in model.eng_thermophononic_energies
    assert 204 in model.eng_thermophononic_energies
    assert pytest.approx(model.eng_thermophononic_energies[201].dt_tph) == 0.001
    assert pytest.approx(model.eng_thermophononic_energies[202].dt_tph) == 0.002
    assert pytest.approx(model.eng_thermophononic_energies[203].dt_tph) == 0.003
    assert pytest.approx(model.eng_thermophononic_energies[204].dt_tph) == 0.004


def test_m315_four_bar_double_crank_joint(tmp_path: Path):
    c1 = f"{1005:>10d}{1006:>10d}{1007:>10d}{1.06e7:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{70.0:>20.4f}{110.0:>20.4f}{85.0:>20.4f}{50.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Four-Bar Double Crank Joint Fixed and Free Format Test
2022 0
/LAGMUL/FOUR_BAR_DOUBLE_CRANK_JOINT/1
Fixed Four-Bar Double Crank Joint
{c1}
{c2}
/LAGMUL/FOUR_BAR_DOUBLE_CRANK_JOINT/2
Free Four-Bar Double Crank Joint
1105, 1106, 1107, 1.22e7, 2, 2.7e-6
85.0, 130.0, 105.0, 60.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_four_bar_double_crank_joints
    assert 2 in model.lagmul_four_bar_double_crank_joints
    fdcj1 = model.lagmul_four_bar_double_crank_joints[1]
    assert fdcj1.node1 == 1005
    assert fdcj1.node2 == 1006
    assert fdcj1.node3 == 1007
    assert pytest.approx(fdcj1.stiff) == 1.06e7
    assert fdcj1.skew_id == 1
    assert pytest.approx(fdcj1.tol) == 1.0e-6
    assert pytest.approx(fdcj1.driving_crank_len) == 70.0
    assert pytest.approx(fdcj1.coupler_len) == 110.0
    assert pytest.approx(fdcj1.driven_crank_len) == 85.0
    assert pytest.approx(fdcj1.ground_len) == 50.0

    fdcj2 = model.lagmul_four_bar_double_crank_joints[2]
    assert fdcj2.node1 == 1105
    assert fdcj2.node2 == 1106
    assert fdcj2.node3 == 1107
    assert pytest.approx(fdcj2.stiff) == 1.22e7
    assert fdcj2.skew_id == 2
    assert pytest.approx(fdcj2.tol) == 2.7e-6
    assert pytest.approx(fdcj2.driving_crank_len) == 85.0
    assert pytest.approx(fdcj2.coupler_len) == 130.0
    assert pytest.approx(fdcj2.driven_crank_len) == 105.0
    assert pytest.approx(fdcj2.ground_len) == 60.0


def test_m315_four_bar_double_crank_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Four-Bar Double Crank Joint Aliases Test
/FOUR_BAR_DOUBLE_CRANK_JOINT/205
1171, 1172, 1173, 7.8e6, 0, 1.0e-6
65.0, 95.0, 75.0, 45.0
/LAGMUL/FOUR_BAR_DOUBLE_CRANK/206
1174, 1175, 1176, 7.8e6, 0, 1.0e-6
65.0, 95.0, 75.0, 45.0
/FOUR_BAR_DOUBLE_CRANK/207
1177, 1178, 1179, 7.8e6, 0, 1.0e-6
65.0, 95.0, 75.0, 45.0
/DRAG_LINK_MECHANISM/208
1180, 1181, 1182, 7.8e6, 0, 1.0e-6
65.0, 95.0, 75.0, 45.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 205 in model.lagmul_four_bar_double_crank_joints
    assert 206 in model.lagmul_four_bar_double_crank_joints
    assert 207 in model.lagmul_four_bar_double_crank_joints
    assert 208 in model.lagmul_four_bar_double_crank_joints
    assert model.lagmul_four_bar_double_crank_joints[205].node1 == 1171
    assert model.lagmul_four_bar_double_crank_joints[206].node1 == 1174
    assert model.lagmul_four_bar_double_crank_joints[207].node1 == 1177
    assert model.lagmul_four_bar_double_crank_joints[208].node1 == 1180


def test_m315_sensor_spring_bending_snap_rate(tmp_path: Path):
    c1 = f"{1108:>10d}{5.7e7:>20.4f}{0.0175:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Snap Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_BENDING_SNAP_RATE/1
Fixed Spring Bending Snap Rate Sensor
{c1}
/SENSOR/SPRING_BENDING_SNAP_RATE/2
Free Spring Bending Snap Rate Sensor
1109, 7.8e7, 0.0240
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_snap_rates
    assert 2 in model.sensor_spring_bending_snap_rates
    s1 = model.sensor_spring_bending_snap_rates[1]
    assert s1.spring_id == 1108
    assert pytest.approx(s1.jbend_crackle_max) == 5.7e7
    assert pytest.approx(s1.t_delay) == 0.0175

    s2 = model.sensor_spring_bending_snap_rates[2]
    assert s2.spring_id == 1109
    assert pytest.approx(s2.jbend_crackle_max) == 7.8e7
    assert pytest.approx(s2.t_delay) == 0.0240

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_BENDING_SNAP_RATE"
    assert model.sensors[1].kind == "SPRING_BENDING_SNAP_RATE"


def test_m315_sensor_spring_bending_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Snap Rate Aliases Test
/SENSOR/SPRING_BEND_SNAP_RATE/137
1138, 3.35e7, 0.0043
/SENSOR/SPRING_RATE_SNAP_BEND/138
1139, 3.55e7, 0.0053
/SENSOR/BENDING_SNAP_RATE_SPRING/139
1140, 3.75e7, 0.0063
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 137 in model.sensor_spring_bending_snap_rates
    assert 138 in model.sensor_spring_bending_snap_rates
    assert 139 in model.sensor_spring_bending_snap_rates
    assert model.sensor_spring_bending_snap_rates[137].spring_id == 1138
    assert pytest.approx(model.sensor_spring_bending_snap_rates[137].jbend_crackle_max) == 3.35e7
    assert model.sensor_spring_bending_snap_rates[138].spring_id == 1139
    assert model.sensor_spring_bending_snap_rates[139].spring_id == 1140

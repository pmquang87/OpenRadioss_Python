"""Tests for Milestone M314: LadInplaneShearRate Failure Model, EngElastocaloricEnergy, FourBarCrankRockerJoint, and SensorSpringTorsionalSnapRate."""

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


def test_m314_fail_lad_inplane_shear_rate_fixed(tmp_path: Path):
    c1 = f"{2.35:>20.4f}{34.5:>20.4f}{0.055:>20.4f}{1.35:>20.4f}{0.982:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1235:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze In-plane Shear Rate Fixed Format Test
2022 0
/FAIL/LAD_INPLANE_SHEAR_RATE/1235
Ladeveze In-plane Shear Rate Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1235 in model.fail_ladinplaneshearrates
    flipsr = model.fail_ladinplaneshearrates[1235]
    assert pytest.approx(flipsr.y0_ipsr) == 2.35
    assert pytest.approx(flipsr.yc_ipsr) == 34.5
    assert pytest.approx(flipsr.gamma_rate_ipsr) == 0.055
    assert pytest.approx(flipsr.n_rate_ipsr) == 1.35
    assert pytest.approx(flipsr.d_ipsr_max) == 0.982
    assert flipsr.ifail_sh == 1
    assert flipsr.ifail_so == 2
    assert flipsr.fail_id == 1235
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_INPLANE_SHEAR_RATE"


def test_m314_fail_lad_inplane_shear_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze In-plane Shear Rate Free Format Test
/FAIL/LAD_INPLANE_SHEAR_RATE/1236
3.15, 45.0, 0.075, 1.5, 0.970
1, 1
1236
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1236 in model.fail_ladinplaneshearrates
    flipsr = model.fail_ladinplaneshearrates[1236]
    assert pytest.approx(flipsr.y0_ipsr) == 3.15
    assert pytest.approx(flipsr.yc_ipsr) == 45.0
    assert pytest.approx(flipsr.gamma_rate_ipsr) == 0.075
    assert pytest.approx(flipsr.n_rate_ipsr) == 1.5
    assert pytest.approx(flipsr.d_ipsr_max) == 0.970
    assert flipsr.fail_id == 1236


def test_m314_fail_lad_inplane_shear_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze In-plane Shear Rate Aliases Test
/FAIL/LADEVEZE_INPLANE_SHEAR_RATE/1237
1.9, 28.0, 0.040, 1.2, 0.95
1, 1
/FAIL/LAD_IPSR/1238
1.9, 28.0, 0.040, 1.2, 0.95
1, 1
/FAIL/LAD_IPSR_MODEL/1239
1.9, 28.0, 0.040, 1.2, 0.95
1, 1
/FAIL/LAD_IPSR_LAW/1240
1.9, 28.0, 0.040, 1.2, 0.95
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_SHEAR/1241
1.9, 28.0, 0.040, 1.2, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1237 in model.fail_ladinplaneshearrates
    assert 1238 in model.fail_ladinplaneshearrates
    assert 1239 in model.fail_ladinplaneshearrates
    assert 1240 in model.fail_ladinplaneshearrates
    assert 1241 in model.fail_ladinplaneshearrates
    assert len(model.raw_fails) == 5


def test_m314_eng_elastocaloric_energy(tmp_path: Path):
    c1 = f"{0.00035:>20.6f}{55:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Elastocaloric Energy Fixed and Free Format Test
2022 0
/ENG/ELASTOCALORIC_ENERGY/1
Fixed Elastocaloric Energy Output
{c1}
/ENG/ELASTOCALORIC_ENERGY/2
Free Elastocaloric Energy Output
0.00070, 110
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_elastocaloric_energies
    assert 2 in model.eng_elastocaloric_energies
    elc1 = model.eng_elastocaloric_energies[1]
    assert pytest.approx(elc1.dt_elc) == 0.00035
    assert elc1.sens_id == 55
    elc2 = model.eng_elastocaloric_energies[2]
    assert pytest.approx(elc2.dt_elc) == 0.00070
    assert elc2.sens_id == 110


def test_m314_eng_elastocaloric_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Elastocaloric Energy Aliases Test
/ENG/ELC_WORK/191
0.001, 190
/ENG/EELASTOCALORIC/192
0.002, 191
/ENG/ELASTOCALORIC_DISSIPATION/193
0.003, 192
/ENG/EM_ELASTOCALORIC/194
0.004, 193
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 191 in model.eng_elastocaloric_energies
    assert 192 in model.eng_elastocaloric_energies
    assert 193 in model.eng_elastocaloric_energies
    assert 194 in model.eng_elastocaloric_energies
    assert pytest.approx(model.eng_elastocaloric_energies[191].dt_elc) == 0.001
    assert pytest.approx(model.eng_elastocaloric_energies[192].dt_elc) == 0.002
    assert pytest.approx(model.eng_elastocaloric_energies[193].dt_elc) == 0.003
    assert pytest.approx(model.eng_elastocaloric_energies[194].dt_elc) == 0.004


def test_m314_four_bar_crank_rocker_joint(tmp_path: Path):
    c1 = f"{1001:>10d}{1002:>10d}{1003:>10d}{1.02e7:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{40.0:>20.4f}{120.0:>20.4f}{90.0:>20.4f}{140.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Four-Bar Crank Rocker Joint Fixed and Free Format Test
2022 0
/LAGMUL/FOUR_BAR_CRANK_ROCKER_JOINT/1
Fixed Four-Bar Crank Rocker Joint
{c1}
{c2}
/LAGMUL/FOUR_BAR_CRANK_ROCKER_JOINT/2
Free Four-Bar Crank Rocker Joint
1101, 1102, 1103, 1.18e7, 2, 2.6e-6
50.0, 140.0, 110.0, 160.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_four_bar_crank_rocker_joints
    assert 2 in model.lagmul_four_bar_crank_rocker_joints
    fcrj1 = model.lagmul_four_bar_crank_rocker_joints[1]
    assert fcrj1.node1 == 1001
    assert fcrj1.node2 == 1002
    assert fcrj1.node3 == 1003
    assert pytest.approx(fcrj1.stiff) == 1.02e7
    assert fcrj1.skew_id == 1
    assert pytest.approx(fcrj1.tol) == 1.0e-6
    assert pytest.approx(fcrj1.crank_len) == 40.0
    assert pytest.approx(fcrj1.coupler_len) == 120.0
    assert pytest.approx(fcrj1.rocker_len) == 90.0
    assert pytest.approx(fcrj1.ground_len) == 140.0

    fcrj2 = model.lagmul_four_bar_crank_rocker_joints[2]
    assert fcrj2.node1 == 1101
    assert fcrj2.node2 == 1102
    assert fcrj2.node3 == 1103
    assert pytest.approx(fcrj2.stiff) == 1.18e7
    assert fcrj2.skew_id == 2
    assert pytest.approx(fcrj2.tol) == 2.6e-6
    assert pytest.approx(fcrj2.crank_len) == 50.0
    assert pytest.approx(fcrj2.coupler_len) == 140.0
    assert pytest.approx(fcrj2.rocker_len) == 110.0
    assert pytest.approx(fcrj2.ground_len) == 160.0


def test_m314_four_bar_crank_rocker_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Four-Bar Crank Rocker Joint Aliases Test
/FOUR_BAR_CRANK_ROCKER_JOINT/195
1161, 1162, 1163, 7.5e6, 0, 1.0e-6
35.0, 100.0, 80.0, 120.0
/LAGMUL/FOUR_BAR_CRANK_ROCKER/196
1164, 1165, 1166, 7.5e6, 0, 1.0e-6
35.0, 100.0, 80.0, 120.0
/FOUR_BAR_CRANK_ROCKER/197
1167, 1168, 1169, 7.5e6, 0, 1.0e-6
35.0, 100.0, 80.0, 120.0
/GRASHOF_CRANK_ROCKER_MECHANISM/198
1170, 1171, 1172, 7.5e6, 0, 1.0e-6
35.0, 100.0, 80.0, 120.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 195 in model.lagmul_four_bar_crank_rocker_joints
    assert 196 in model.lagmul_four_bar_crank_rocker_joints
    assert 197 in model.lagmul_four_bar_crank_rocker_joints
    assert 198 in model.lagmul_four_bar_crank_rocker_joints
    assert model.lagmul_four_bar_crank_rocker_joints[195].node1 == 1161
    assert model.lagmul_four_bar_crank_rocker_joints[196].node1 == 1164
    assert model.lagmul_four_bar_crank_rocker_joints[197].node1 == 1167
    assert model.lagmul_four_bar_crank_rocker_joints[198].node1 == 1170


def test_m314_sensor_spring_torsional_snap_rate(tmp_path: Path):
    c1 = f"{1098:>10d}{5.4e7:>20.4f}{0.0165:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Snap Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_SNAP_RATE/1
Fixed Spring Torsional Snap Rate Sensor
{c1}
/SENSOR/SPRING_TORSIONAL_SNAP_RATE/2
Free Spring Torsional Snap Rate Sensor
1099, 7.5e7, 0.0225
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_snap_rates
    assert 2 in model.sensor_spring_torsional_snap_rates
    s1 = model.sensor_spring_torsional_snap_rates[1]
    assert s1.spring_id == 1098
    assert pytest.approx(s1.jtors_crackle_max) == 5.4e7
    assert pytest.approx(s1.t_delay) == 0.0165

    s2 = model.sensor_spring_torsional_snap_rates[2]
    assert s2.spring_id == 1099
    assert pytest.approx(s2.jtors_crackle_max) == 7.5e7
    assert pytest.approx(s2.t_delay) == 0.0225

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TORSIONAL_SNAP_RATE"
    assert model.sensors[1].kind == "SPRING_TORSIONAL_SNAP_RATE"


def test_m314_sensor_spring_torsional_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Snap Rate Aliases Test
/SENSOR/SPRING_TORS_SNAP_RATE/134
1128, 3.25e7, 0.0041
/SENSOR/SPRING_RATE_SNAP_TORS/135
1129, 3.45e7, 0.0051
/SENSOR/TORSIONAL_SNAP_RATE_SPRING/136
1130, 3.65e7, 0.0061
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 134 in model.sensor_spring_torsional_snap_rates
    assert 135 in model.sensor_spring_torsional_snap_rates
    assert 136 in model.sensor_spring_torsional_snap_rates
    assert model.sensor_spring_torsional_snap_rates[134].spring_id == 1128
    assert pytest.approx(model.sensor_spring_torsional_snap_rates[134].jtors_crackle_max) == 3.25e7
    assert model.sensor_spring_torsional_snap_rates[135].spring_id == 1129
    assert model.sensor_spring_torsional_snap_rates[136].spring_id == 1130

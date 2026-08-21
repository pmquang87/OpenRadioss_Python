"""Tests for Milestone M310: LadInplaneShear Failure Model, EngBarocaloricEnergy, StephensonLinkageJoint, and SensorSpringBendingJerkRate."""

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


def test_m310_fail_lad_inplane_shear_fixed(tmp_path: Path):
    c1 = f"{2.25:>20.4f}{32.5:>20.4f}{0.0055:>20.4f}{450.0:>20.4f}{0.985:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1195:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze In-plane Shear Fixed Format Test
2022 0
/FAIL/LAD_INPLANE_SHEAR/1195
Ladeveze In-plane Shear Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1195 in model.fail_ladinplaneshears
    flips = model.fail_ladinplaneshears[1195]
    assert pytest.approx(flips.y0_ips) == 2.25
    assert pytest.approx(flips.yc_ips) == 32.5
    assert pytest.approx(flips.gamma_plastic_0) == 0.0055
    assert pytest.approx(flips.alpha_ips_slip) == 450.0
    assert pytest.approx(flips.d_ips_max) == 0.985
    assert flips.ifail_sh == 1
    assert flips.ifail_so == 2
    assert flips.fail_id == 1195
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_INPLANE_SHEAR"


def test_m310_fail_lad_inplane_shear_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze In-plane Shear Free Format Test
/FAIL/LAD_INPLANE_SHEAR/1196
3.1, 45.0, 0.0075, 520.0, 0.968
1, 1
1196
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1196 in model.fail_ladinplaneshears
    flips = model.fail_ladinplaneshears[1196]
    assert pytest.approx(flips.y0_ips) == 3.1
    assert pytest.approx(flips.yc_ips) == 45.0
    assert pytest.approx(flips.gamma_plastic_0) == 0.0075
    assert pytest.approx(flips.alpha_ips_slip) == 520.0
    assert pytest.approx(flips.d_ips_max) == 0.968
    assert flips.fail_id == 1196


def test_m310_fail_lad_inplane_shear_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze In-plane Shear Aliases Test
/FAIL/LADEVEZE_INPLANE_SHEAR/1197
2.0, 28.0, 0.0045, 400.0, 0.97
1, 1
/FAIL/LAD_IPS/1198
2.0, 28.0, 0.0045, 400.0, 0.97
1, 1
/FAIL/LAD_IPS_MODEL/1199
2.0, 28.0, 0.0045, 400.0, 0.97
1, 1
/FAIL/LAD_IPS_LAW/1200
2.0, 28.0, 0.0045, 400.0, 0.97
1, 1
/FAIL/LADEVEZE_SHEAR_DAMAGE/1201
2.0, 28.0, 0.0045, 400.0, 0.97
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1197 in model.fail_ladinplaneshears
    assert 1198 in model.fail_ladinplaneshears
    assert 1199 in model.fail_ladinplaneshears
    assert 1200 in model.fail_ladinplaneshears
    assert 1201 in model.fail_ladinplaneshears
    assert len(model.raw_fails) == 5


def test_m310_eng_barocaloric_energy(tmp_path: Path):
    c1 = f"{0.00023:>20.6f}{43:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Barocaloric Energy Fixed and Free Format Test
2022 0
/ENG/BAROCALORIC_ENERGY/1
Fixed Barocaloric Energy Output
{c1}
/ENG/BAROCALORIC_ENERGY/2
Free Barocaloric Energy Output
0.00046, 86
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_barocaloric_energies
    assert 2 in model.eng_barocaloric_energies
    bce1 = model.eng_barocaloric_energies[1]
    assert pytest.approx(bce1.dt_bce) == 0.00023
    assert bce1.sens_id == 43
    bce2 = model.eng_barocaloric_energies[2]
    assert pytest.approx(bce2.dt_bce) == 0.00046
    assert bce2.sens_id == 86


def test_m310_eng_barocaloric_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Barocaloric Energy Aliases Test
/ENG/BCE_WORK/151
0.001, 150
/ENG/EBAROCALORIC/152
0.002, 151
/ENG/BAROCALORIC_DISSIPATION/153
0.003, 152
/ENG/EM_BAROCALORIC/154
0.004, 153
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 151 in model.eng_barocaloric_energies
    assert 152 in model.eng_barocaloric_energies
    assert 153 in model.eng_barocaloric_energies
    assert 154 in model.eng_barocaloric_energies
    assert pytest.approx(model.eng_barocaloric_energies[151].dt_bce) == 0.001
    assert pytest.approx(model.eng_barocaloric_energies[152].dt_bce) == 0.002
    assert pytest.approx(model.eng_barocaloric_energies[153].dt_bce) == 0.003
    assert pytest.approx(model.eng_barocaloric_energies[154].dt_bce) == 0.004


def test_m310_stephenson_linkage_joint(tmp_path: Path):
    c1 = f"{971:>10d}{972:>10d}{973:>10d}{9.3e6:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{45.0:>20.4f}{110.0:>20.4f}{95.0:>20.4f}{135.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Stephenson Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/STEPHENSON_LINKAGE_JOINT/1
Fixed Stephenson Linkage Joint
{c1}
{c2}
/LAGMUL/STEPHENSON_LINKAGE_JOINT/2
Free Stephenson Linkage Joint
1071, 1072, 1073, 1.05e7, 2, 2.1e-6
55.0, 130.0, 115.0, 160.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_stephenson_linkage_joints
    assert 2 in model.lagmul_stephenson_linkage_joints
    slj1 = model.lagmul_stephenson_linkage_joints[1]
    assert slj1.node1 == 971
    assert slj1.node2 == 972
    assert slj1.node3 == 973
    assert pytest.approx(slj1.stiff) == 9.3e6
    assert slj1.skew_id == 1
    assert pytest.approx(slj1.tol) == 1.0e-6
    assert pytest.approx(slj1.link_len_a) == 45.0
    assert pytest.approx(slj1.link_len_b) == 110.0
    assert pytest.approx(slj1.link_len_c) == 95.0
    assert pytest.approx(slj1.link_len_d) == 135.0

    slj2 = model.lagmul_stephenson_linkage_joints[2]
    assert slj2.node1 == 1071
    assert slj2.node2 == 1072
    assert slj2.node3 == 1073
    assert pytest.approx(slj2.stiff) == 1.05e7
    assert slj2.skew_id == 2
    assert pytest.approx(slj2.tol) == 2.1e-6
    assert pytest.approx(slj2.link_len_a) == 55.0
    assert pytest.approx(slj2.link_len_b) == 130.0
    assert pytest.approx(slj2.link_len_c) == 115.0
    assert pytest.approx(slj2.link_len_d) == 160.0


def test_m310_stephenson_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Stephenson Linkage Joint Aliases Test
/STEPHENSON_LINKAGE_JOINT/155
1121, 1122, 1123, 6.8e6, 0, 1.0e-6
40.0, 100.0, 80.0, 120.0
/LAGMUL/STEPHENSON_LINKAGE/156
1124, 1125, 1126, 6.8e6, 0, 1.0e-6
40.0, 100.0, 80.0, 120.0
/STEPHENSON_LINKAGE/157
1127, 1128, 1129, 6.8e6, 0, 1.0e-6
40.0, 100.0, 80.0, 120.0
/STEPHENSON_SIXBAR_MECHANISM/158
1130, 1131, 1132, 6.8e6, 0, 1.0e-6
40.0, 100.0, 80.0, 120.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 155 in model.lagmul_stephenson_linkage_joints
    assert 156 in model.lagmul_stephenson_linkage_joints
    assert 157 in model.lagmul_stephenson_linkage_joints
    assert 158 in model.lagmul_stephenson_linkage_joints
    assert model.lagmul_stephenson_linkage_joints[155].node1 == 1121
    assert model.lagmul_stephenson_linkage_joints[156].node1 == 1124
    assert model.lagmul_stephenson_linkage_joints[157].node1 == 1127
    assert model.lagmul_stephenson_linkage_joints[158].node1 == 1130


def test_m310_sensor_spring_bending_jerk_rate(tmp_path: Path):
    c1 = f"{1058:>10d}{4.3e7:>20.4f}{0.0128:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Jerk Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_BENDING_JERK_RATE/1
Fixed Spring Bending Jerk Rate Sensor
{c1}
/SENSOR/SPRING_BENDING_JERK_RATE/2
Free Spring Bending Jerk Rate Sensor
1059, 6.5e7, 0.0172
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_jerk_rates
    assert 2 in model.sensor_spring_bending_jerk_rates
    s1 = model.sensor_spring_bending_jerk_rates[1]
    assert s1.spring_id == 1058
    assert pytest.approx(s1.jbend_snap_max) == 4.3e7
    assert pytest.approx(s1.t_delay) == 0.0128

    s2 = model.sensor_spring_bending_jerk_rates[2]
    assert s2.spring_id == 1059
    assert pytest.approx(s2.jbend_snap_max) == 6.5e7
    assert pytest.approx(s2.t_delay) == 0.0172

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_BENDING_JERK_RATE"
    assert model.sensors[1].kind == "SPRING_BENDING_JERK_RATE"


def test_m310_sensor_spring_bending_jerk_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Jerk Rate Aliases Test
/SENSOR/SPRING_BEND_JERK_RATE/122
1088, 2.85e7, 0.0034
/SENSOR/SPRING_RATE_JERK_BEND/123
1089, 3.05e7, 0.0044
/SENSOR/BENDING_JERK_RATE_SPRING/124
1090, 3.25e7, 0.0054
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 122 in model.sensor_spring_bending_jerk_rates
    assert 123 in model.sensor_spring_bending_jerk_rates
    assert 124 in model.sensor_spring_bending_jerk_rates
    assert model.sensor_spring_bending_jerk_rates[122].spring_id == 1088
    assert pytest.approx(model.sensor_spring_bending_jerk_rates[122].jbend_snap_max) == 2.85e7
    assert model.sensor_spring_bending_jerk_rates[123].spring_id == 1089
    assert model.sensor_spring_bending_jerk_rates[124].spring_id == 1090

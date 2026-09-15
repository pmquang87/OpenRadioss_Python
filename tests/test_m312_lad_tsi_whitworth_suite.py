"""Tests for Milestone M312: LadTransverseShearInteraction Failure Model, EngElectrohydrodynamicEnergy, WhitworthQuickReturnJoint, and SensorSpringNormalJerkRate."""

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


def test_m312_fail_lad_transverse_shear_interaction_fixed(tmp_path: Path):
    c1 = f"{2.85:>20.4f}{38.5:>20.4f}{1.75:>20.4f}{350.0:>20.4f}{0.988:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1215:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Shear Interaction Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_SHEAR_INTERACTION/1215
Ladeveze Transverse Shear Interaction Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1215 in model.fail_ladtransverseshearinteractions
    fltsi = model.fail_ladtransverseshearinteractions[1215]
    assert pytest.approx(fltsi.y0_tsi) == 2.85
    assert pytest.approx(fltsi.yc_tsi) == 38.5
    assert pytest.approx(fltsi.gamma_coupling_exp) == 1.75
    assert pytest.approx(fltsi.sigma_tsi_max) == 350.0
    assert pytest.approx(fltsi.d_tsi_max) == 0.988
    assert fltsi.ifail_sh == 1
    assert fltsi.ifail_so == 2
    assert fltsi.fail_id == 1215
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_SHEAR_INTERACTION"


def test_m312_fail_lad_transverse_shear_interaction_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Shear Interaction Free Format Test
/FAIL/LAD_TRANSVERSE_SHEAR_INTERACTION/1216
3.4, 48.0, 1.85, 420.0, 0.975
1, 1
1216
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1216 in model.fail_ladtransverseshearinteractions
    fltsi = model.fail_ladtransverseshearinteractions[1216]
    assert pytest.approx(fltsi.y0_tsi) == 3.4
    assert pytest.approx(fltsi.yc_tsi) == 48.0
    assert pytest.approx(fltsi.gamma_coupling_exp) == 1.85
    assert pytest.approx(fltsi.sigma_tsi_max) == 420.0
    assert pytest.approx(fltsi.d_tsi_max) == 0.975
    assert fltsi.fail_id == 1216


def test_m312_fail_lad_transverse_shear_interaction_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Shear Interaction Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_SHEAR_INTERACTION/1217
2.2, 30.0, 1.5, 310.0, 0.96
1, 1
/FAIL/LAD_TSI/1218
2.2, 30.0, 1.5, 310.0, 0.96
1, 1
/FAIL/LAD_TSI_MODEL/1219
2.2, 30.0, 1.5, 310.0, 0.96
1, 1
/FAIL/LAD_TSI_LAW/1220
2.2, 30.0, 1.5, 310.0, 0.96
1, 1
/FAIL/LADEVEZE_COUPLING_DAMAGE/1221
2.2, 30.0, 1.5, 310.0, 0.96
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1217 in model.fail_ladtransverseshearinteractions
    assert 1218 in model.fail_ladtransverseshearinteractions
    assert 1219 in model.fail_ladtransverseshearinteractions
    assert 1220 in model.fail_ladtransverseshearinteractions
    assert 1221 in model.fail_ladtransverseshearinteractions
    assert len(model.raw_fails) == 5


def test_m312_eng_electrohydrodynamic_energy(tmp_path: Path):
    c1 = f"{0.00028:>20.6f}{48:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Electrohydrodynamic Energy Fixed and Free Format Test
2022 0
/ENG/ELECTROHYDRODYNAMIC_ENERGY/1
Fixed Electrohydrodynamic Energy Output
{c1}
/ENG/ELECTROHYDRODYNAMIC_ENERGY/2
Free Electrohydrodynamic Energy Output
0.00056, 96
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrohydrodynamic_energies
    assert 2 in model.eng_electrohydrodynamic_energies
    ehd1 = model.eng_electrohydrodynamic_energies[1]
    assert pytest.approx(ehd1.dt_ehd) == 0.00028
    assert ehd1.sens_id == 48
    ehd2 = model.eng_electrohydrodynamic_energies[2]
    assert pytest.approx(ehd2.dt_ehd) == 0.00056
    assert ehd2.sens_id == 96


def test_m312_eng_electrohydrodynamic_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Electrohydrodynamic Energy Aliases Test
/ENG/EHD_WORK/171
0.001, 170
/ENG/EELECTROHYDRODYNAMIC/172
0.002, 171
/ENG/ELECTROHYDRODYNAMIC_DISSIPATION/173
0.003, 172
/ENG/EM_ELECTROHYDRODYNAMIC/174
0.004, 173
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 171 in model.eng_electrohydrodynamic_energies
    assert 172 in model.eng_electrohydrodynamic_energies
    assert 173 in model.eng_electrohydrodynamic_energies
    assert 174 in model.eng_electrohydrodynamic_energies
    assert pytest.approx(model.eng_electrohydrodynamic_energies[171].dt_ehd) == 0.001
    assert pytest.approx(model.eng_electrohydrodynamic_energies[172].dt_ehd) == 0.002
    assert pytest.approx(model.eng_electrohydrodynamic_energies[173].dt_ehd) == 0.003
    assert pytest.approx(model.eng_electrohydrodynamic_energies[174].dt_ehd) == 0.004


def test_m312_whitworth_quick_return_joint(tmp_path: Path):
    c1 = f"{991:>10d}{992:>10d}{993:>10d}{9.6e6:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{50.0:>20.4f}{40.0:>20.4f}{180.0:>20.4f}{120.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Whitworth Quick Return Joint Fixed and Free Format Test
2022 0
/LAGMUL/WHITWORTH_QUICK_RETURN_JOINT/1
Fixed Whitworth Quick Return Joint
{c1}
{c2}
/LAGMUL/WHITWORTH_QUICK_RETURN_JOINT/2
Free Whitworth Quick Return Joint
1091, 1092, 1093, 1.12e7, 2, 2.4e-6
65.0, 52.0, 210.0, 145.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_whitworth_quick_return_joints
    assert 2 in model.lagmul_whitworth_quick_return_joints
    wqrj1 = model.lagmul_whitworth_quick_return_joints[1]
    assert wqrj1.node1 == 991
    assert wqrj1.node2 == 992
    assert wqrj1.node3 == 993
    assert pytest.approx(wqrj1.stiff) == 9.6e6
    assert wqrj1.skew_id == 1
    assert pytest.approx(wqrj1.tol) == 1.0e-6
    assert pytest.approx(wqrj1.driving_crank_len) == 50.0
    assert pytest.approx(wqrj1.pivot_offset) == 40.0
    assert pytest.approx(wqrj1.slotted_arm_len) == 180.0
    assert pytest.approx(wqrj1.connecting_rod_len) == 120.0

    wqrj2 = model.lagmul_whitworth_quick_return_joints[2]
    assert wqrj2.node1 == 1091
    assert wqrj2.node2 == 1092
    assert wqrj2.node3 == 1093
    assert pytest.approx(wqrj2.stiff) == 1.12e7
    assert wqrj2.skew_id == 2
    assert pytest.approx(wqrj2.tol) == 2.4e-6
    assert pytest.approx(wqrj2.driving_crank_len) == 65.0
    assert pytest.approx(wqrj2.pivot_offset) == 52.0
    assert pytest.approx(wqrj2.slotted_arm_len) == 210.0
    assert pytest.approx(wqrj2.connecting_rod_len) == 145.0


def test_m312_whitworth_quick_return_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Whitworth Quick Return Joint Aliases Test
/WHITWORTH_QUICK_RETURN_JOINT/175
1141, 1142, 1143, 7.1e6, 0, 1.0e-6
45.0, 35.0, 160.0, 105.0
/LAGMUL/WHITWORTH_QUICK_RETURN/176
1144, 1145, 1146, 7.1e6, 0, 1.0e-6
45.0, 35.0, 160.0, 105.0
/WHITWORTH_QUICK_RETURN/177
1147, 1148, 1149, 7.1e6, 0, 1.0e-6
45.0, 35.0, 160.0, 105.0
/WHITWORTH_SLOTTED_CRANK_MECHANISM/178
1150, 1151, 1152, 7.1e6, 0, 1.0e-6
45.0, 35.0, 160.0, 105.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 175 in model.lagmul_whitworth_quick_return_joints
    assert 176 in model.lagmul_whitworth_quick_return_joints
    assert 177 in model.lagmul_whitworth_quick_return_joints
    assert 178 in model.lagmul_whitworth_quick_return_joints
    assert model.lagmul_whitworth_quick_return_joints[175].node1 == 1141
    assert model.lagmul_whitworth_quick_return_joints[176].node1 == 1144
    assert model.lagmul_whitworth_quick_return_joints[177].node1 == 1147
    assert model.lagmul_whitworth_quick_return_joints[178].node1 == 1150


def test_m312_sensor_spring_normal_jerk_rate(tmp_path: Path):
    c1 = f"{1078:>10d}{4.8e7:>20.4f}{0.0142:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Jerk Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_NORMAL_JERK_RATE/1
Fixed Spring Normal Jerk Rate Sensor
{c1}
/SENSOR/SPRING_NORMAL_JERK_RATE/2
Free Spring Normal Jerk Rate Sensor
1079, 6.9e7, 0.0195
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_jerk_rates
    assert 2 in model.sensor_spring_normal_jerk_rates
    s1 = model.sensor_spring_normal_jerk_rates[1]
    assert s1.spring_id == 1078
    assert pytest.approx(s1.jnorm_snap_max) == 4.8e7
    assert pytest.approx(s1.t_delay) == 0.0142

    s2 = model.sensor_spring_normal_jerk_rates[2]
    assert s2.spring_id == 1079
    assert pytest.approx(s2.jnorm_snap_max) == 6.9e7
    assert pytest.approx(s2.t_delay) == 0.0195

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_NORMAL_JERK_RATE"
    assert model.sensors[1].kind == "SPRING_NORMAL_JERK_RATE"


def test_m312_sensor_spring_normal_jerk_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Jerk Rate Aliases Test
/SENSOR/SPRING_NORM_JERK_RATE/128
1108, 3.05e7, 0.0038
/SENSOR/SPRING_RATE_JERK_NORM/129
1109, 3.25e7, 0.0048
/SENSOR/NORMAL_JERK_RATE_SPRING/130
1110, 3.45e7, 0.0058
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 128 in model.sensor_spring_normal_jerk_rates
    assert 129 in model.sensor_spring_normal_jerk_rates
    assert 130 in model.sensor_spring_normal_jerk_rates
    assert model.sensor_spring_normal_jerk_rates[128].spring_id == 1108
    assert pytest.approx(model.sensor_spring_normal_jerk_rates[128].jnorm_snap_max) == 3.05e7
    assert model.sensor_spring_normal_jerk_rates[129].spring_id == 1109
    assert model.sensor_spring_normal_jerk_rates[130].spring_id == 1110

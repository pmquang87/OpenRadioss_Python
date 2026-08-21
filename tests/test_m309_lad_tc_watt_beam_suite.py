"""Tests for Milestone M309: LadTransverseCompression Failure Model, EngPiezomagneticEnergy, WattBeamEngineJoint, and SensorSpringTorsionalJerkRate."""

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


def test_m309_fail_lad_transverse_compression_fixed(tmp_path: Path):
    c1 = f"{3.45:>20.4f}{42.5:>20.4f}{0.28:>20.4f}{145.0:>20.4f}{0.988:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1185:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Compression Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_COMPRESSION/1185
Ladeveze Transverse Compressive Crushing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1185 in model.fail_ladtransversecompressions
    fltc = model.fail_ladtransversecompressions[1185]
    assert pytest.approx(fltc.y0_tc) == 3.45
    assert pytest.approx(fltc.yc_tc) == 42.5
    assert pytest.approx(fltc.mu_fric_tc) == 0.28
    assert pytest.approx(fltc.sigma_tc_max) == 145.0
    assert pytest.approx(fltc.d_tc_max) == 0.988
    assert fltc.ifail_sh == 1
    assert fltc.ifail_so == 2
    assert fltc.fail_id == 1185
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_COMPRESSION"


def test_m309_fail_lad_transverse_compression_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Compression Free Format Test
/FAIL/LAD_TRANSVERSE_COMPRESSION/1186
4.2, 55.0, 0.35, 175.0, 0.965
1, 1
1186
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1186 in model.fail_ladtransversecompressions
    fltc = model.fail_ladtransversecompressions[1186]
    assert pytest.approx(fltc.y0_tc) == 4.2
    assert pytest.approx(fltc.yc_tc) == 55.0
    assert pytest.approx(fltc.mu_fric_tc) == 0.35
    assert pytest.approx(fltc.sigma_tc_max) == 175.0
    assert pytest.approx(fltc.d_tc_max) == 0.965
    assert fltc.fail_id == 1186


def test_m309_fail_lad_transverse_compression_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Compression Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_COMPRESSION/1187
2.8, 38.0, 0.25, 130.0, 0.97
1, 1
/FAIL/LAD_TRANS_COMP/1188
2.8, 38.0, 0.25, 130.0, 0.97
1, 1
/FAIL/LAD_TRANS_COMP_MODEL/1189
2.8, 38.0, 0.25, 130.0, 0.97
1, 1
/FAIL/LAD_TRANS_COMP_LAW/1190
2.8, 38.0, 0.25, 130.0, 0.97
1, 1
/FAIL/LADEVEZE_TRANSVERSE_CRUSH/1191
2.8, 38.0, 0.25, 130.0, 0.97
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1187 in model.fail_ladtransversecompressions
    assert 1188 in model.fail_ladtransversecompressions
    assert 1189 in model.fail_ladtransversecompressions
    assert 1190 in model.fail_ladtransversecompressions
    assert 1191 in model.fail_ladtransversecompressions
    assert len(model.raw_fails) == 5


def test_m309_eng_piezomagnetic_energy(tmp_path: Path):
    c1 = f"{0.00021:>20.6f}{41:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Piezomagnetic Energy Fixed and Free Format Test
2022 0
/ENG/PIEZOMAGNETIC_ENERGY/1
Fixed Piezomagnetic Energy Output
{c1}
/ENG/PIEZOMAGNETIC_ENERGY/2
Free Piezomagnetic Energy Output
0.00042, 82
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_piezomagnetic_energies
    assert 2 in model.eng_piezomagnetic_energies
    pze1 = model.eng_piezomagnetic_energies[1]
    assert pytest.approx(pze1.dt_pzm) == 0.00021
    assert pze1.sens_id == 41
    pze2 = model.eng_piezomagnetic_energies[2]
    assert pytest.approx(pze2.dt_pzm) == 0.00042
    assert pze2.sens_id == 82


def test_m309_eng_piezomagnetic_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Piezomagnetic Energy Aliases Test
/ENG/PZM_WORK/141
0.001, 140
/ENG/EPIEZOMAGNETIC/142
0.002, 141
/ENG/PIEZOMAGNETIC_DISSIPATION/143
0.003, 142
/ENG/EM_PIEZOMAGNETIC/144
0.004, 143
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 141 in model.eng_piezomagnetic_energies
    assert 142 in model.eng_piezomagnetic_energies
    assert 143 in model.eng_piezomagnetic_energies
    assert 144 in model.eng_piezomagnetic_energies
    assert pytest.approx(model.eng_piezomagnetic_energies[141].dt_pzm) == 0.001
    assert pytest.approx(model.eng_piezomagnetic_energies[142].dt_pzm) == 0.002
    assert pytest.approx(model.eng_piezomagnetic_energies[143].dt_pzm) == 0.003
    assert pytest.approx(model.eng_piezomagnetic_energies[144].dt_pzm) == 0.004


def test_m309_watt_beam_engine_joint(tmp_path: Path):
    c1 = f"{961:>10d}{962:>10d}{963:>10d}{9.2e6:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{75.0:>20.4f}{85.0:>20.4f}{120.0:>20.4f}{0.35:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Watt Beam Engine Joint Fixed and Free Format Test
2022 0
/LAGMUL/WATT_BEAM_ENGINE_JOINT/1
Fixed Watt Beam Engine Joint
{c1}
{c2}
/LAGMUL/WATT_BEAM_ENGINE_JOINT/2
Free Watt Beam Engine Joint
1061, 1062, 1063, 9.9e6, 2, 2.0e-6
90.0, 100.0, 150.0, 0.45
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_watt_beam_engine_joints
    assert 2 in model.lagmul_watt_beam_engine_joints
    wbej1 = model.lagmul_watt_beam_engine_joints[1]
    assert wbej1.node1 == 961
    assert wbej1.node2 == 962
    assert wbej1.node3 == 963
    assert pytest.approx(wbej1.stiff) == 9.2e6
    assert wbej1.skew_id == 1
    assert pytest.approx(wbej1.tol) == 1.0e-6
    assert pytest.approx(wbej1.beam_half_length_a) == 75.0
    assert pytest.approx(wbej1.beam_half_length_b) == 85.0
    assert pytest.approx(wbej1.stroke_travel) == 120.0
    assert pytest.approx(wbej1.beam_tilt_max) == 0.35

    wbej2 = model.lagmul_watt_beam_engine_joints[2]
    assert wbej2.node1 == 1061
    assert wbej2.node2 == 1062
    assert wbej2.node3 == 1063
    assert pytest.approx(wbej2.stiff) == 9.9e6
    assert wbej2.skew_id == 2
    assert pytest.approx(wbej2.tol) == 2.0e-6
    assert pytest.approx(wbej2.beam_half_length_a) == 90.0
    assert pytest.approx(wbej2.beam_half_length_b) == 100.0
    assert pytest.approx(wbej2.stroke_travel) == 150.0
    assert pytest.approx(wbej2.beam_tilt_max) == 0.45


def test_m309_watt_beam_engine_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Watt Beam Engine Joint Aliases Test
/WATT_BEAM_ENGINE_JOINT/145
1111, 1112, 1113, 6.7e6, 0, 1.0e-6
60.0, 70.0, 100.0, 0.3
/LAGMUL/WATT_BEAM_ENGINE/146
1114, 1115, 1116, 6.7e6, 0, 1.0e-6
60.0, 70.0, 100.0, 0.3
/WATT_BEAM_ENGINE/147
1117, 1118, 1119, 6.7e6, 0, 1.0e-6
60.0, 70.0, 100.0, 0.3
/WATT_ROCKING_BEAM_MECHANISM/148
1120, 1121, 1122, 6.7e6, 0, 1.0e-6
60.0, 70.0, 100.0, 0.3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 145 in model.lagmul_watt_beam_engine_joints
    assert 146 in model.lagmul_watt_beam_engine_joints
    assert 147 in model.lagmul_watt_beam_engine_joints
    assert 148 in model.lagmul_watt_beam_engine_joints
    assert model.lagmul_watt_beam_engine_joints[145].node1 == 1111
    assert model.lagmul_watt_beam_engine_joints[146].node1 == 1114
    assert model.lagmul_watt_beam_engine_joints[147].node1 == 1117
    assert model.lagmul_watt_beam_engine_joints[148].node1 == 1120


def test_m309_sensor_spring_torsional_jerk_rate(tmp_path: Path):
    c1 = f"{1048:>10d}{4.1e7:>20.4f}{0.0120:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Jerk Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_JERK_RATE/1
Fixed Spring Torsional Jerk Rate Sensor
{c1}
/SENSOR/SPRING_TORSIONAL_JERK_RATE/2
Free Spring Torsional Jerk Rate Sensor
1049, 6.3e7, 0.0162
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_jerk_rates
    assert 2 in model.sensor_spring_torsional_jerk_rates
    s1 = model.sensor_spring_torsional_jerk_rates[1]
    assert s1.spring_id == 1048
    assert pytest.approx(s1.jtors_snap_max) == 4.1e7
    assert pytest.approx(s1.t_delay) == 0.0120

    s2 = model.sensor_spring_torsional_jerk_rates[2]
    assert s2.spring_id == 1049
    assert pytest.approx(s2.jtors_snap_max) == 6.3e7
    assert pytest.approx(s2.t_delay) == 0.0162

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TORSIONAL_JERK_RATE"
    assert model.sensors[1].kind == "SPRING_TORSIONAL_JERK_RATE"


def test_m309_sensor_spring_torsional_jerk_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Jerk Rate Aliases Test
/SENSOR/SPRING_TORS_JERK_RATE/119
1078, 2.75e7, 0.0033
/SENSOR/SPRING_RATE_JERK_TORS/120
1079, 2.95e7, 0.0043
/SENSOR/TORSIONAL_JERK_RATE_SPRING/121
1080, 3.15e7, 0.0053
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 119 in model.sensor_spring_torsional_jerk_rates
    assert 120 in model.sensor_spring_torsional_jerk_rates
    assert 121 in model.sensor_spring_torsional_jerk_rates
    assert model.sensor_spring_torsional_jerk_rates[119].spring_id == 1078
    assert pytest.approx(model.sensor_spring_torsional_jerk_rates[119].jtors_snap_max) == 2.75e7
    assert model.sensor_spring_torsional_jerk_rates[120].spring_id == 1079
    assert model.sensor_spring_torsional_jerk_rates[121].spring_id == 1080

"""Tests for Milestone M317: LadInterfacialDelaminationRate Failure Model, EngThermomagneticGeneratorEnergy, SliderRockerInversionJoint, and SensorSpringTransverseSnapRate."""

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


def test_m317_fail_lad_interfacial_delamination_rate_fixed(tmp_path: Path):
    c1 = f"{2.95:>20.4f}{40.5:>20.4f}{0.072:>20.4f}{1.52:>20.4f}{0.988:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1265:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Interfacial Delamination Rate Fixed Format Test
2022 0
/FAIL/LAD_INTERFACIAL_DELAMINATION_RATE/1265
Ladeveze Interfacial Delamination Rate Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1265 in model.fail_ladinterfacialdelaminationrates
    flifdr = model.fail_ladinterfacialdelaminationrates[1265]
    assert pytest.approx(flifdr.y0_ifdr) == 2.95
    assert pytest.approx(flifdr.yc_ifdr) == 40.5
    assert pytest.approx(flifdr.c_rate_ifdr) == 0.072
    assert pytest.approx(flifdr.p_rate_ifdr) == 1.52
    assert pytest.approx(flifdr.d_ifdr_max) == 0.988
    assert flifdr.ifail_sh == 1
    assert flifdr.ifail_so == 2
    assert flifdr.fail_id == 1265
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_INTERFACIAL_DELAMINATION_RATE"


def test_m317_fail_lad_interfacial_delamination_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Interfacial Delamination Rate Free Format Test
/FAIL/LAD_INTERFACIAL_DELAMINATION_RATE/1266
3.75, 52.0, 0.092, 1.8, 0.962
1, 1
1266
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1266 in model.fail_ladinterfacialdelaminationrates
    flifdr = model.fail_ladinterfacialdelaminationrates[1266]
    assert pytest.approx(flifdr.y0_ifdr) == 3.75
    assert pytest.approx(flifdr.yc_ifdr) == 52.0
    assert pytest.approx(flifdr.c_rate_ifdr) == 0.092
    assert pytest.approx(flifdr.p_rate_ifdr) == 1.8
    assert pytest.approx(flifdr.d_ifdr_max) == 0.962
    assert flifdr.fail_id == 1266


def test_m317_fail_lad_interfacial_delamination_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Interfacial Delamination Rate Aliases Test
/FAIL/LADEVEZE_INTERFACIAL_DELAMINATION_RATE/1267
2.5, 33.0, 0.055, 1.45, 0.95
1, 1
/FAIL/LAD_IFDR/1268
2.5, 33.0, 0.055, 1.45, 0.95
1, 1
/FAIL/LAD_IFDR_MODEL/1269
2.5, 33.0, 0.055, 1.45, 0.95
1, 1
/FAIL/LAD_IFDR_LAW/1270
2.5, 33.0, 0.055, 1.45, 0.95
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DELAMINATION/1271
2.5, 33.0, 0.055, 1.45, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1267 in model.fail_ladinterfacialdelaminationrates
    assert 1268 in model.fail_ladinterfacialdelaminationrates
    assert 1269 in model.fail_ladinterfacialdelaminationrates
    assert 1270 in model.fail_ladinterfacialdelaminationrates
    assert 1271 in model.fail_ladinterfacialdelaminationrates
    assert len(model.raw_fails) == 5


def test_m317_eng_thermomagnetic_generator_energy(tmp_path: Path):
    c1 = f"{0.00045:>20.6f}{65:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Thermomagnetic Generator Energy Fixed and Free Format Test
2022 0
/ENG/THERMOMAGNETIC_GENERATOR_ENERGY/1
Fixed Thermomagnetic Generator Energy Output
{c1}
/ENG/THERMOMAGNETIC_GENERATOR_ENERGY/2
Free Thermomagnetic Generator Energy Output
0.00090, 130
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_thermomagnetic_generator_energies
    assert 2 in model.eng_thermomagnetic_generator_energies
    tmg1 = model.eng_thermomagnetic_generator_energies[1]
    assert pytest.approx(tmg1.dt_tmg) == 0.00045
    assert tmg1.sens_id == 65
    tmg2 = model.eng_thermomagnetic_generator_energies[2]
    assert pytest.approx(tmg2.dt_tmg) == 0.00090
    assert tmg2.sens_id == 130


def test_m317_eng_thermomagnetic_generator_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Thermomagnetic Generator Energy Aliases Test
/ENG/TMG_WORK/221
0.001, 220
/ENG/ETHERMOMAGNETIC_GENERATOR/222
0.002, 221
/ENG/THERMOMAGNETIC_GENERATOR_DISSIPATION/223
0.003, 222
/ENG/EM_THERMOMAGNETIC_GENERATOR/224
0.004, 223
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 221 in model.eng_thermomagnetic_generator_energies
    assert 222 in model.eng_thermomagnetic_generator_energies
    assert 223 in model.eng_thermomagnetic_generator_energies
    assert 224 in model.eng_thermomagnetic_generator_energies
    assert pytest.approx(model.eng_thermomagnetic_generator_energies[221].dt_tmg) == 0.001
    assert pytest.approx(model.eng_thermomagnetic_generator_energies[222].dt_tmg) == 0.002
    assert pytest.approx(model.eng_thermomagnetic_generator_energies[223].dt_tmg) == 0.003
    assert pytest.approx(model.eng_thermomagnetic_generator_energies[224].dt_tmg) == 0.004


def test_m317_slider_rocker_inversion_joint(tmp_path: Path):
    c1 = f"{1021:>10d}{1022:>10d}{1023:>10d}{1.12e7:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{45.0:>20.4f}{120.0:>20.4f}{5.0:>20.4f}{150.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Slider-Rocker Inversion Joint Fixed and Free Format Test
2022 0
/LAGMUL/SLIDER_ROCKER_INVERSION_JOINT/1
Fixed Slider-Rocker Inversion Joint
{c1}
{c2}
/LAGMUL/SLIDER_ROCKER_INVERSION_JOINT/2
Free Slider-Rocker Inversion Joint
1121, 1122, 1123, 1.28e7, 2, 2.9e-6
55.0, 140.0, 8.0, 180.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_slider_rocker_inversion_joints
    assert 2 in model.lagmul_slider_rocker_inversion_joints
    srij1 = model.lagmul_slider_rocker_inversion_joints[1]
    assert srij1.node1 == 1021
    assert srij1.node2 == 1022
    assert srij1.node3 == 1023
    assert pytest.approx(srij1.stiff) == 1.12e7
    assert srij1.skew_id == 1
    assert pytest.approx(srij1.tol) == 1.0e-6
    assert pytest.approx(srij1.crank_len) == 45.0
    assert pytest.approx(srij1.frame_dist) == 120.0
    assert pytest.approx(srij1.piston_offset) == 5.0
    assert pytest.approx(srij1.stroke_limit) == 150.0

    srij2 = model.lagmul_slider_rocker_inversion_joints[2]
    assert srij2.node1 == 1121
    assert srij2.node2 == 1122
    assert srij2.node3 == 1123
    assert pytest.approx(srij2.stiff) == 1.28e7
    assert srij2.skew_id == 2
    assert pytest.approx(srij2.tol) == 2.9e-6
    assert pytest.approx(srij2.crank_len) == 55.0
    assert pytest.approx(srij2.frame_dist) == 140.0
    assert pytest.approx(srij2.piston_offset) == 8.0
    assert pytest.approx(srij2.stroke_limit) == 180.0


def test_m317_slider_rocker_inversion_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Slider-Rocker Inversion Joint Aliases Test
/SLIDER_ROCKER_INVERSION_JOINT/225
1191, 1192, 1193, 8.4e6, 0, 1.0e-6
40.0, 100.0, 0.0, 130.0
/LAGMUL/SLIDER_ROCKER_INVERSION/226
1194, 1195, 1196, 8.4e6, 0, 1.0e-6
40.0, 100.0, 0.0, 130.0
/SLIDER_ROCKER_INVERSION/227
1197, 1198, 1199, 8.4e6, 0, 1.0e-6
40.0, 100.0, 0.0, 130.0
/OSCILLATING_CYLINDER_MECHANISM/228
1200, 1201, 1202, 8.4e6, 0, 1.0e-6
40.0, 100.0, 0.0, 130.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 225 in model.lagmul_slider_rocker_inversion_joints
    assert 226 in model.lagmul_slider_rocker_inversion_joints
    assert 227 in model.lagmul_slider_rocker_inversion_joints
    assert 228 in model.lagmul_slider_rocker_inversion_joints
    assert model.lagmul_slider_rocker_inversion_joints[225].node1 == 1191
    assert model.lagmul_slider_rocker_inversion_joints[226].node1 == 1194
    assert model.lagmul_slider_rocker_inversion_joints[227].node1 == 1197
    assert model.lagmul_slider_rocker_inversion_joints[228].node1 == 1200


def test_m317_sensor_spring_transverse_snap_rate(tmp_path: Path):
    c1 = f"{1128:>10d}{6.1e7:>20.4f}{0.0195:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Snap Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_SNAP_RATE/1
Fixed Spring Transverse Snap Rate Sensor
{c1}
/SENSOR/SPRING_TRANSVERSE_SNAP_RATE/2
Free Spring Transverse Snap Rate Sensor
1129, 8.4e7, 0.0265
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_snap_rates
    assert 2 in model.sensor_spring_transverse_snap_rates
    s1 = model.sensor_spring_transverse_snap_rates[1]
    assert s1.spring_id == 1128
    assert pytest.approx(s1.jtrans_crackle_max) == 6.1e7
    assert pytest.approx(s1.t_delay) == 0.0195

    s2 = model.sensor_spring_transverse_snap_rates[2]
    assert s2.spring_id == 1129
    assert pytest.approx(s2.jtrans_crackle_max) == 8.4e7
    assert pytest.approx(s2.t_delay) == 0.0265

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_SNAP_RATE"
    assert model.sensors[1].kind == "SPRING_TRANSVERSE_SNAP_RATE"


def test_m317_sensor_spring_transverse_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Snap Rate Aliases Test
/SENSOR/SPRING_TRANS_SNAP_RATE/143
1158, 3.55e7, 0.0047
/SENSOR/SPRING_RATE_SNAP_TRANS/144
1159, 3.75e7, 0.0057
/SENSOR/TRANSVERSE_SNAP_RATE_SPRING/145
1160, 3.95e7, 0.0067
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 143 in model.sensor_spring_transverse_snap_rates
    assert 144 in model.sensor_spring_transverse_snap_rates
    assert 145 in model.sensor_spring_transverse_snap_rates
    assert model.sensor_spring_transverse_snap_rates[143].spring_id == 1158
    assert pytest.approx(model.sensor_spring_transverse_snap_rates[143].jtrans_crackle_max) == 3.55e7
    assert model.sensor_spring_transverse_snap_rates[144].spring_id == 1159
    assert model.sensor_spring_transverse_snap_rates[145].spring_id == 1160

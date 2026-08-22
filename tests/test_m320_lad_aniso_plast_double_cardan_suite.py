"""Tests for Milestone M320: LadAnisotropicPlasticity Failure Model, EngThermoacousticEnergy, DoubleCardanJoint, and SensorSpringTransverseCrackleRate."""

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


def test_m320_fail_lad_anisotropic_plasticity_fixed(tmp_path: Path):
    c1 = f"{4.25:>20.4f}{62.5:>20.4f}{150.0:>20.4f}{35.0:>20.4f}{0.975:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1289:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Anisotropic Plasticity Failure Fixed Format Test
2022 0
/FAIL/LAD_ANISOTROPIC_PLASTICITY/1289
Ladeveze Anisotropic Plasticity Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1289 in model.fail_ladanisotropicplasticities
    flap = model.fail_ladanisotropicplasticities[1289]
    assert pytest.approx(flap.y0_aniso) == 4.25
    assert pytest.approx(flap.yc_aniso) == 62.5
    assert pytest.approx(flap.r0_hard) == 150.0
    assert pytest.approx(flap.beta_hard) == 35.0
    assert pytest.approx(flap.d_aniso_max) == 0.975
    assert flap.ifail_sh == 1
    assert flap.ifail_so == 2
    assert flap.fail_id == 1289
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_ANISOTROPIC_PLASTICITY"


def test_m320_fail_lad_anisotropic_plasticity_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Anisotropic Plasticity Failure Free Format Test
/FAIL/LAD_ANISOTROPIC_PLASTICITY/1290
5.15, 70.0, 180.0, 42.0, 0.960
1, 1
1290
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1290 in model.fail_ladanisotropicplasticities
    flap = model.fail_ladanisotropicplasticities[1290]
    assert pytest.approx(flap.y0_aniso) == 5.15
    assert pytest.approx(flap.yc_aniso) == 70.0
    assert pytest.approx(flap.r0_hard) == 180.0
    assert pytest.approx(flap.beta_hard) == 42.0
    assert pytest.approx(flap.d_aniso_max) == 0.960
    assert flap.fail_id == 1290


def test_m320_fail_lad_anisotropic_plasticity_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Anisotropic Plasticity Failure Aliases Test
/FAIL/LADEVEZE_ANISOTROPIC_PLASTICITY/1291
3.5, 45.0, 120.0, 25.0, 0.95
1, 1
/FAIL/LAD_ANISO_PLAST/1292
3.5, 45.0, 120.0, 25.0, 0.95
1, 1
/FAIL/LAD_ANISO_PLAST_MODEL/1293
3.5, 45.0, 120.0, 25.0, 0.95
1, 1
/FAIL/LAD_ANISO_PLAST_LAW/1294
3.5, 45.0, 120.0, 25.0, 0.95
1, 1
/FAIL/LADEVEZE_DAMAGE_PLASTICITY/1295
3.5, 45.0, 120.0, 25.0, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1291 in model.fail_ladanisotropicplasticities
    assert 1292 in model.fail_ladanisotropicplasticities
    assert 1293 in model.fail_ladanisotropicplasticities
    assert 1294 in model.fail_ladanisotropicplasticities
    assert 1295 in model.fail_ladanisotropicplasticities
    assert len(model.raw_fails) == 5


def test_m320_eng_thermoacoustic_energy(tmp_path: Path):
    c1 = f"{0.00045:>20.6f}{75:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Thermoacoustic Energy Fixed and Free Format Test
2022 0
/ENG/THERMOACOUSTIC_ENERGY/1
Fixed Thermoacoustic Energy Output
{c1}
/ENG/THERMOACOUSTIC_ENERGY/2
Free Thermoacoustic Energy Output
0.00095, 155
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_thermoacoustic_energies
    assert 2 in model.eng_thermoacoustic_energies
    ta1 = model.eng_thermoacoustic_energies[1]
    assert pytest.approx(ta1.dt_ta) == 0.00045
    assert ta1.sens_id == 75
    ta2 = model.eng_thermoacoustic_energies[2]
    assert pytest.approx(ta2.dt_ta) == 0.00095
    assert ta2.sens_id == 155


def test_m320_eng_thermoacoustic_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Thermoacoustic Energy Aliases Test
/ENG/TA_WORK/251
0.0016, 250
/ENG/ETHERMOACOUSTIC/252
0.0026, 251
/ENG/THERMOACOUSTIC_DISSIPATION/253
0.0036, 252
/ENG/EM_THERMOACOUSTIC/254
0.0046, 253
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 251 in model.eng_thermoacoustic_energies
    assert 252 in model.eng_thermoacoustic_energies
    assert 253 in model.eng_thermoacoustic_energies
    assert 254 in model.eng_thermoacoustic_energies
    assert pytest.approx(model.eng_thermoacoustic_energies[251].dt_ta) == 0.0016
    assert pytest.approx(model.eng_thermoacoustic_energies[252].dt_ta) == 0.0026
    assert pytest.approx(model.eng_thermoacoustic_energies[253].dt_ta) == 0.0036
    assert pytest.approx(model.eng_thermoacoustic_energies[254].dt_ta) == 0.0046


def test_m320_double_cardan_joint(tmp_path: Path):
    c1 = f"{1051:>10d}{1052:>10d}{1053:>10d}{1.65e7:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{75.0:>20.4f}{45.0:>20.4f}{0.0:>20.4f}{0.05:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Double Cardan Joint Fixed and Free Format Test
2022 0
/LAGMUL/DOUBLE_CARDAN_JOINT/1
Fixed Double Cardan Mechanism Joint
{c1}
{c2}
/LAGMUL/DOUBLE_CARDAN_JOINT/2
Free Double Cardan Mechanism Joint
1151, 1152, 1153, 1.85e7, 2, 2.5e-6
85.0, 50.0, 15.0, 0.08
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_double_cardan_joints
    assert 2 in model.lagmul_double_cardan_joints
    dcj1 = model.lagmul_double_cardan_joints[1]
    assert dcj1.node1 == 1051
    assert dcj1.node2 == 1052
    assert dcj1.node3 == 1053
    assert pytest.approx(dcj1.stiff) == 1.65e7
    assert dcj1.skew_id == 1
    assert pytest.approx(dcj1.tol) == 1.0e-6
    assert pytest.approx(dcj1.center_yoke_len) == 75.0
    assert pytest.approx(dcj1.max_bend_angle) == 45.0
    assert pytest.approx(dcj1.phase_offset) == 0.0
    assert pytest.approx(dcj1.friction_coeff) == 0.05

    dcj2 = model.lagmul_double_cardan_joints[2]
    assert dcj2.node1 == 1151
    assert dcj2.node2 == 1152
    assert dcj2.node3 == 1153
    assert pytest.approx(dcj2.stiff) == 1.85e7
    assert dcj2.skew_id == 2
    assert pytest.approx(dcj2.tol) == 2.5e-6
    assert pytest.approx(dcj2.center_yoke_len) == 85.0
    assert pytest.approx(dcj2.max_bend_angle) == 50.0
    assert pytest.approx(dcj2.phase_offset) == 15.0
    assert pytest.approx(dcj2.friction_coeff) == 0.08


def test_m320_double_cardan_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Double Cardan Joint Aliases Test
/DOUBLE_CARDAN_JOINT/255
1251, 1252, 1253, 9.5e6, 0, 1.0e-6
60.0, 40.0, 0.0, 0.02
/LAGMUL/DOUBLE_CARDAN_COUPLING/256
1254, 1255, 1256, 9.5e6, 0, 1.0e-6
60.0, 40.0, 0.0, 0.02
/DOUBLE_CARDAN_COUPLING/257
1257, 1258, 1259, 9.5e6, 0, 1.0e-6
60.0, 40.0, 0.0, 0.02
/DOUBLE_CARDAN_MECHANISM/258
1260, 1261, 1262, 9.5e6, 0, 1.0e-6
60.0, 40.0, 0.0, 0.02
/DOUBLE_UNIVERSAL_JOINT/259
1263, 1264, 1265, 9.5e6, 0, 1.0e-6
60.0, 40.0, 0.0, 0.02
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 255 in model.lagmul_double_cardan_joints
    assert 256 in model.lagmul_double_cardan_joints
    assert 257 in model.lagmul_double_cardan_joints
    assert 258 in model.lagmul_double_cardan_joints
    assert 259 in model.lagmul_double_cardan_joints
    assert model.lagmul_double_cardan_joints[255].node1 == 1251
    assert model.lagmul_double_cardan_joints[256].node1 == 1254
    assert model.lagmul_double_cardan_joints[257].node1 == 1257
    assert model.lagmul_double_cardan_joints[258].node1 == 1260
    assert model.lagmul_double_cardan_joints[259].node1 == 1263


def test_m320_sensor_spring_transverse_crackle_rate(tmp_path: Path):
    c1 = f"{1158:>10d}{7.5e7:>20.4f}{0.0195:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Crackle Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE/1
Fixed Spring Transverse Crackle Rate Sensor
{c1}
/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE/2
Free Spring Transverse Crackle Rate Sensor
1159, 9.5e7, 0.0295
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_crackle_rates
    assert 2 in model.sensor_spring_transverse_crackle_rates
    s1 = model.sensor_spring_transverse_crackle_rates[1]
    assert s1.spring_id == 1158
    assert pytest.approx(s1.jtrans_pop_max) == 7.5e7
    assert pytest.approx(s1.t_delay) == 0.0195

    s2 = model.sensor_spring_transverse_crackle_rates[2]
    assert s2.spring_id == 1159
    assert pytest.approx(s2.jtrans_pop_max) == 9.5e7
    assert pytest.approx(s2.t_delay) == 0.0295

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_CRACKLE_RATE"
    assert model.sensors[1].kind == "SPRING_TRANSVERSE_CRACKLE_RATE"


def test_m320_sensor_spring_transverse_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Crackle Rate Aliases Test
/SENSOR/SPRING_TRANS_CRACKLE_RATE/173
1188, 6.25e7, 0.0055
/SENSOR/SPRING_RATE_CRACKLE_TRANS/174
1189, 6.45e7, 0.0065
/SENSOR/TRANSVERSE_CRACKLE_RATE_SPRING/175
1190, 6.65e7, 0.0075
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 173 in model.sensor_spring_transverse_crackle_rates
    assert 174 in model.sensor_spring_transverse_crackle_rates
    assert 175 in model.sensor_spring_transverse_crackle_rates
    assert model.sensor_spring_transverse_crackle_rates[173].spring_id == 1188
    assert pytest.approx(model.sensor_spring_transverse_crackle_rates[173].jtrans_pop_max) == 6.25e7
    assert model.sensor_spring_transverse_crackle_rates[174].spring_id == 1189
    assert model.sensor_spring_transverse_crackle_rates[175].spring_id == 1190

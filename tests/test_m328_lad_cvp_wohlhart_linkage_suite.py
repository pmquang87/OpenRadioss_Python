"""Tests for Milestone M328: LadCoupleViscoplasticity Failure Model, EngThermomagneticResonanceEnergy, WohlhartLinkageJoint, and SensorSpringTorsionalPopRate."""

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


def test_m328_fail_lad_couple_viscoplasticity_fixed(tmp_path: Path):
    c1 = f"{350.0:>20.4f}{4.20:>20.4f}{125.0:>20.2f}{25.5:>20.4f}{0.970:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1410:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Couple Viscoplasticity Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_VISCOPLASTICITY/1410
Ladeveze Coupled Damage Viscoplasticity Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1410 in model.fail_ladcoupleviscoplasticitys
    fcvp = model.fail_ladcoupleviscoplasticitys[1410]
    assert pytest.approx(fcvp.k_vp) == 350.0
    assert pytest.approx(fcvp.n_vp) == 4.20
    assert pytest.approx(fcvp.r_vp0) == 125.0
    assert pytest.approx(fcvp.gamma_vp) == 25.5
    assert pytest.approx(fcvp.d_cvp_max) == 0.970
    assert fcvp.ifail_sh == 1
    assert fcvp.ifail_so == 2
    assert fcvp.fail_id == 1410
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_VISCOPLASTICITY"


def test_m328_fail_lad_couple_viscoplasticity_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Couple Viscoplasticity Free Format Test
/FAIL/LAD_COUPLE_VISCOPLASTICITY/1411
420.0, 4.8, 140.0, 28.0, 0.955
1, 1
1411
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1411 in model.fail_ladcoupleviscoplasticitys
    fcvp = model.fail_ladcoupleviscoplasticitys[1411]
    assert pytest.approx(fcvp.k_vp) == 420.0
    assert pytest.approx(fcvp.n_vp) == 4.8
    assert pytest.approx(fcvp.r_vp0) == 140.0
    assert pytest.approx(fcvp.gamma_vp) == 28.0
    assert pytest.approx(fcvp.d_cvp_max) == 0.955
    assert fcvp.fail_id == 1411


def test_m328_fail_lad_couple_viscoplasticity_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Couple Viscoplasticity Aliases Test
/FAIL/LADEVEZE_COUPLED_VISCOPLASTICITY/1412
300.0, 4.0, 100.0, 20.0, 0.95
1, 1
/FAIL/LAD_CVP/1413
300.0, 4.0, 100.0, 20.0, 0.95
1, 1
/FAIL/LAD_CVP_MODEL/1414
300.0, 4.0, 100.0, 20.0, 0.95
1, 1
/FAIL/LAD_CVP_LAW/1415
300.0, 4.0, 100.0, 20.0, 0.95
1, 1
/FAIL/LADEVEZE_DAMAGE_VISCOPLASTICITY_COUPLING/1416
300.0, 4.0, 100.0, 20.0, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1412 in model.fail_ladcoupleviscoplasticitys
    assert 1413 in model.fail_ladcoupleviscoplasticitys
    assert 1414 in model.fail_ladcoupleviscoplasticitys
    assert 1415 in model.fail_ladcoupleviscoplasticitys
    assert 1416 in model.fail_ladcoupleviscoplasticitys
    assert len(model.raw_fails) == 5


def test_m328_fail_lad_couple_viscoplasticity_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_COUPLE_VISCOPLASTICITY/1417
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m328_eng_thermomagnetic_resonance_energy(tmp_path: Path):
    c1 = f"{0.00075:>20.6f}{125:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Thermomagnetic Resonance Energy Fixed and Free Format Test
2022 0
/ENG/THERMOMAGNETIC_RESONANCE_ENERGY/1
Fixed Thermomagnetic Resonance Energy Output
{c1}
/ENG/THERMOMAGNETIC_RESONANCE_ENERGY/2
Free Thermomagnetic Resonance Energy Output
0.00115, 250
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_thermomagnetic_resonance_energies
    assert 2 in model.eng_thermomagnetic_resonance_energies
    tmre1 = model.eng_thermomagnetic_resonance_energies[1]
    assert pytest.approx(tmre1.dt_tmr) == 0.00075
    assert tmre1.sens_id == 125
    tmre2 = model.eng_thermomagnetic_resonance_energies[2]
    assert pytest.approx(tmre2.dt_tmr) == 0.00115
    assert tmre2.sens_id == 250


def test_m328_eng_thermomagnetic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Thermomagnetic Resonance Energy Aliases Test
/ENG/THERMOMAG_RES_WORK/3
0.00025, 68
/ENG/ETHERMOMAGRESONANCE/4
0.00030, 78
/ENG/THERMOMAGNETIC_RESONANCE_DISSIPATION/5
0.00035, 88
/ENG/EM_THERMOMAGNETIC_RESONANCE/6
0.00040, 98
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_thermomagnetic_resonance_energies
    assert 4 in model.eng_thermomagnetic_resonance_energies
    assert 5 in model.eng_thermomagnetic_resonance_energies
    assert 6 in model.eng_thermomagnetic_resonance_energies


def test_m328_eng_thermomagnetic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/THERMOMAGNETIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m328_lagmul_wohlhart_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{141:>10d}{142:>10d}{143:>10d}{9.5e6:>20.1f}{28:>10d}{2.2e-5:>20.6e}"
    c2 = f"{70.0:>20.4f}{64.5:>20.4f}{56.0:>20.4f}{18.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Wohlhart Linkage Joint Fixed Format Test
2022 0
/WOHLHART_LINKAGE_JOINT/105
Wohlhart Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 105 in model.lagmul_wohlhart_linkage_joints
    joint = model.lagmul_wohlhart_linkage_joints[105]
    assert joint.node1 == 141
    assert joint.node2 == 142
    assert joint.node3 == 143
    assert pytest.approx(joint.stiff) == 9.5e6
    assert joint.skew_id == 28
    assert pytest.approx(joint.tol) == 2.2e-5
    assert pytest.approx(joint.link_len_a) == 70.0
    assert pytest.approx(joint.link_len_b) == 64.5
    assert pytest.approx(joint.twist_angle_alpha) == 56.0
    assert pytest.approx(joint.offset_distance_s) == 18.0


def test_m328_lagmul_wohlhart_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Wohlhart Linkage Joint Free Format Test
/LAGMUL/WOHLHART_LINKAGE_JOINT/106
241, 242, 243, 6.8e6, 45, 3.8e-5
74.0, 68.0, 60.0, 20.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 106 in model.lagmul_wohlhart_linkage_joints
    joint = model.lagmul_wohlhart_linkage_joints[106]
    assert joint.node1 == 241
    assert joint.node2 == 242
    assert joint.node3 == 243
    assert pytest.approx(joint.stiff) == 6.8e6
    assert joint.skew_id == 45
    assert pytest.approx(joint.tol) == 3.8e-5
    assert pytest.approx(joint.link_len_a) == 74.0
    assert pytest.approx(joint.link_len_b) == 68.0
    assert pytest.approx(joint.twist_angle_alpha) == 60.0
    assert pytest.approx(joint.offset_distance_s) == 20.0


def test_m328_lagmul_wohlhart_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Wohlhart Linkage Joint Aliases Test
/LAGMUL/WOHLHART_LINKAGE/107
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/WOHLHART_LINKAGE/108
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/WOHLHART_SPATIAL_MECHANISM/109
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/WOHLHART_6R_MECHANISM/110
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 107 in model.lagmul_wohlhart_linkage_joints
    assert 108 in model.lagmul_wohlhart_linkage_joints
    assert 109 in model.lagmul_wohlhart_linkage_joints
    assert 110 in model.lagmul_wohlhart_linkage_joints


def test_m328_lagmul_wohlhart_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/WOHLHART_LINKAGE_JOINT/111
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m328_sensor_spring_torsional_pop_rate_fixed(tmp_path: Path):
    c1 = f"{841:>10d}{1.45e9:>20.1f}{0.0055:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Torsional Pop Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_POP_RATE/138
Spring Torsional Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 138 in model.sensor_spring_torsional_pop_rates
    sensor = model.sensor_spring_torsional_pop_rates[138]
    assert sensor.spring_id == 841
    assert pytest.approx(sensor.jtors_pop_max) == 1.45e9
    assert pytest.approx(sensor.t_delay) == 0.0055
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_POP_RATE"


def test_m328_sensor_spring_torsional_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Torsional Pop Rate Sensor Free Format Test
/SENSOR/SPRING_TORSIONAL_POP_RATE/139
842, 1.55e9, 0.0075
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 139 in model.sensor_spring_torsional_pop_rates
    sensor = model.sensor_spring_torsional_pop_rates[139]
    assert sensor.spring_id == 842
    assert pytest.approx(sensor.jtors_pop_max) == 1.55e9
    assert pytest.approx(sensor.t_delay) == 0.0075


def test_m328_sensor_spring_torsional_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Torsional Pop Rate Sensor Aliases Test
/SENSOR/SPRING_TORS_POP_RATE/140
843, 2.0e9, 0.001
/SENSOR/SPRING_RATE_POP_TORS/141
844, 2.0e9, 0.001
/SENSOR/TORSIONAL_POP_RATE_SPRING/142
845, 2.0e9, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 140 in model.sensor_spring_torsional_pop_rates
    assert 141 in model.sensor_spring_torsional_pop_rates
    assert 142 in model.sensor_spring_torsional_pop_rates
    assert len(model.sensors) == 3


def test_m328_sensor_spring_torsional_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TORSIONAL_POP_RATE/143
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

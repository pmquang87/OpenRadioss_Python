"""Tests for Milestone M351: LadTransverseInterlaminarTensionRate Failure Model, EngFlexothermoplasmonexcitonpolaritonicResonanceEnergy, BricardSpatialLinkageJoint, and SensorSpringTotalSurgeRate."""

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


def test_m351_fail_lad_transverse_interlaminar_tension_rate_fixed(tmp_path: Path):
    c1 = f"{118.0:>20.4f}{354.0:>20.4f}{38.0:>20.4f}{2.05:>20.4f}{0.985:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1670:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Interlaminar Tension Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_TENSION_RATE/1670
Ladeveze Transverse Interlaminar Tension Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1670 in model.fail_ladtransverseinterlaminartensionrates
    ftitr = model.fail_ladtransverseinterlaminartensionrates[1670]
    assert pytest.approx(ftitr.sigma_titr0) == 118.0
    assert pytest.approx(ftitr.sigma_titrc) == 354.0
    assert pytest.approx(ftitr.gamma_titr) == 38.0
    assert pytest.approx(ftitr.p_titr) == 2.05
    assert pytest.approx(ftitr.d_titr_max) == 0.985
    assert ftitr.ifail_sh == 1
    assert ftitr.ifail_so == 2
    assert ftitr.fail_id == 1670
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_INTERLAMINAR_TENSION_RATE"


def test_m351_fail_lad_transverse_interlaminar_tension_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Interlaminar Tension Rate Free Format Test
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_TENSION_RATE/1671
128.0, 384.0, 44.0, 2.25, 0.965
1, 1
1671
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1671 in model.fail_ladtransverseinterlaminartensionrates
    ftitr = model.fail_ladtransverseinterlaminartensionrates[1671]
    assert pytest.approx(ftitr.sigma_titr0) == 128.0
    assert pytest.approx(ftitr.sigma_titrc) == 384.0
    assert pytest.approx(ftitr.gamma_titr) == 44.0
    assert pytest.approx(ftitr.p_titr) == 2.25
    assert pytest.approx(ftitr.d_titr_max) == 0.965
    assert ftitr.fail_id == 1671


def test_m351_fail_lad_transverse_interlaminar_tension_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Interlaminar Tension Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_TENSION_RATE/1672
108.0, 324.0, 34.0, 1.85, 0.970
1, 1
/FAIL/LAD_TITR/1673
108.0, 324.0, 34.0, 1.85, 0.970
1, 1
/FAIL/LAD_TITR_MODEL/1674
108.0, 324.0, 34.0, 1.85, 0.970
1, 1
/FAIL/LAD_TITR_LAW/1675
108.0, 324.0, 34.0, 1.85, 0.970
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_INTERLAMINAR_TENSION/1676
108.0, 324.0, 34.0, 1.85, 0.970
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1672 in model.fail_ladtransverseinterlaminartensionrates
    assert 1673 in model.fail_ladtransverseinterlaminartensionrates
    assert 1674 in model.fail_ladtransverseinterlaminartensionrates
    assert 1675 in model.fail_ladtransverseinterlaminartensionrates
    assert 1676 in model.fail_ladtransverseinterlaminartensionrates
    assert len(model.raw_fails) == 5


def test_m351_fail_lad_transverse_interlaminar_tension_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_TENSION_RATE/1677
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m351_eng_flexothermoplasmonexcitonpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00115:>20.6f}{210:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoplasmonexcitonpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPLASMONEXCITONPOLARITONIC_RESONANCE_ENERGY/1
Flexothermoplasmonexcitonpolaritonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoplasmonexcitonpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonexcitonpolaritonic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftpepr) == 0.00115
    assert eng.sens_id == 210


def test_m351_eng_flexothermoplasmonexcitonpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermoplasmonexcitonpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPLASMONEXCITONPOLARITONIC_RESONANCE_ENERGY/2
0.00135, 220
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermoplasmonexcitonpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonexcitonpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftpepr) == 0.00135
    assert eng.sens_id == 220


def test_m351_eng_flexothermoplasmonexcitonpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoplasmonexcitonpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PLASMON_EXCITON_POLARITON_RES_WORK/3
0.00095, 160
/ENG/EFLEXOTHERMOPLASMONEXCITONPOLARITONICRESONANCE/4
0.00099, 168
/ENG/FLEXOTHERMOPLASMONEXCITONPOLARITONIC_RESONANCE_DISSIPATION/5
0.00105, 178
/ENG/EM_FLEXOTHERMOPLASMONEXCITONPOLARITONIC_RESONANCE/6
0.00112, 188
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoplasmonexcitonpolaritonic_resonance_energies
    assert 4 in model.eng_flexothermoplasmonexcitonpolaritonic_resonance_energies
    assert 5 in model.eng_flexothermoplasmonexcitonpolaritonic_resonance_energies
    assert 6 in model.eng_flexothermoplasmonexcitonpolaritonic_resonance_energies


def test_m351_eng_flexothermoplasmonexcitonpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOPLASMONEXCITONPOLARITONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m351_lagmul_bricard_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{371:>10d}{372:>10d}{373:>10d}{3.25e7:>20.1f}{108:>10d}{5.5e-5:>20.6e}"
    c2 = f"{170.0:>20.4f}{160.0:>20.4f}{140.0:>20.4f}{95.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Bricard Spatial Linkage Joint Fixed Format Test
2022 0
/BRICARD_SPATIAL_LINKAGE_JOINT/335
Bricard Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 335 in model.lagmul_bricard_spatial_linkage_joints
    joint = model.lagmul_bricard_spatial_linkage_joints[335]
    assert joint.node1 == 371
    assert joint.node2 == 372
    assert joint.node3 == 373
    assert pytest.approx(joint.stiff) == 3.25e7
    assert joint.skew_id == 108
    assert pytest.approx(joint.tol) == 5.5e-5
    assert pytest.approx(joint.link_len_a) == 170.0
    assert pytest.approx(joint.link_len_b) == 160.0
    assert pytest.approx(joint.twist_angle_alpha) == 140.0
    assert pytest.approx(joint.offset_distance_s) == 95.0


def test_m351_lagmul_bricard_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Bricard Spatial Linkage Joint Free Format Test
/LAGMUL/BRICARD_SPATIAL_LINKAGE_JOINT/336
471, 472, 473, 18.0e6, 128, 6.5e-5
172.0, 162.0, 142.0, 96.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 336 in model.lagmul_bricard_spatial_linkage_joints
    joint = model.lagmul_bricard_spatial_linkage_joints[336]
    assert joint.node1 == 471
    assert joint.node2 == 472
    assert joint.node3 == 473
    assert pytest.approx(joint.stiff) == 18.0e6
    assert joint.skew_id == 128
    assert pytest.approx(joint.tol) == 6.5e-5
    assert pytest.approx(joint.link_len_a) == 172.0
    assert pytest.approx(joint.link_len_b) == 162.0
    assert pytest.approx(joint.twist_angle_alpha) == 142.0
    assert pytest.approx(joint.offset_distance_s) == 96.0


def test_m351_lagmul_bricard_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Bricard Spatial Linkage Joint Aliases Test
/LAGMUL/BRICARD_SPATIAL_LINKAGE/337
1, 2, 3, 1e6, 0, 1e-6
36.0, 36.0, 80.0, 22.0
/BRICARD_SPATIAL_LINKAGE/338
1, 2, 3, 1e6, 0, 1e-6
36.0, 36.0, 80.0, 22.0
/BRICARD_SPATIAL_MULTI_LOOP_MECHANISM/339
1, 2, 3, 1e6, 0, 1e-6
36.0, 36.0, 80.0, 22.0
/BRICARD_SPATIAL_TRIAXIAL_MECHANISM/340
1, 2, 3, 1e6, 0, 1e-6
36.0, 36.0, 80.0, 22.0
/BRICARD_SPATIAL_6R_MECHANISM/341
1, 2, 3, 1e6, 0, 1e-6
36.0, 36.0, 80.0, 22.0
/BRICARD_SPATIAL_OVERCONSTRAINED_MECHANISM/342
1, 2, 3, 1e6, 0, 1e-6
36.0, 36.0, 80.0, 22.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 337 in model.lagmul_bricard_spatial_linkage_joints
    assert 338 in model.lagmul_bricard_spatial_linkage_joints
    assert 339 in model.lagmul_bricard_spatial_linkage_joints
    assert 340 in model.lagmul_bricard_spatial_linkage_joints
    assert 341 in model.lagmul_bricard_spatial_linkage_joints
    assert 342 in model.lagmul_bricard_spatial_linkage_joints


def test_m351_lagmul_bricard_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/BRICARD_SPATIAL_LINKAGE_JOINT/343
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m351_sensor_spring_total_surge_rate_fixed(tmp_path: Path):
    c1 = f"{1115:>10d}{5.55e9:>20.1f}{0.0235:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Total Surge Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_SURGE_RATE/368
Spring Total Surge Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 368 in model.sensor_spring_total_surge_rates
    sensor = model.sensor_spring_total_surge_rates[368]
    assert sensor.spring_id == 1115
    assert pytest.approx(sensor.jtot_surge_max) == 5.55e9
    assert pytest.approx(sensor.t_delay) == 0.0235
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_SURGE_RATE"


def test_m351_sensor_spring_total_surge_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Total Surge Rate Sensor Free Format Test
/SENSOR/SPRING_TOTAL_SURGE_RATE/369
1116, 5.65e9, 0.0260
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 369 in model.sensor_spring_total_surge_rates
    sensor = model.sensor_spring_total_surge_rates[369]
    assert sensor.spring_id == 1116
    assert pytest.approx(sensor.jtot_surge_max) == 5.65e9
    assert pytest.approx(sensor.t_delay) == 0.0260


def test_m351_sensor_spring_total_surge_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Total Surge Rate Sensor Aliases Test
/SENSOR/SPRING_TOT_SURGE_RATE/370
1117, 5.7e9, 0.0165
/SENSOR/SPRING_RATE_SURGE_TOT/371
1118, 5.7e9, 0.0165
/SENSOR/TOTAL_SURGE_RATE_SPRING/372
1119, 5.7e9, 0.0165
/SENSOR/SPRING_SURGE_TOT/373
1120, 5.7e9, 0.0165
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 370 in model.sensor_spring_total_surge_rates
    assert 371 in model.sensor_spring_total_surge_rates
    assert 372 in model.sensor_spring_total_surge_rates
    assert 373 in model.sensor_spring_total_surge_rates
    assert len(model.sensors) == 4


def test_m351_sensor_spring_total_surge_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TOTAL_SURGE_RATE/374
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

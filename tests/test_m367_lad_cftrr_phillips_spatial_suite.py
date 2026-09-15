"""Tests for Milestone M367: LadCoupleFiberTensionRuptureRate Failure Model, EngFlexomagnetoplasmonicmagnonResonanceEnergy, PhillipsSpatialLinkageJoint, and SensorSpringNormalSnapRate."""

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


def test_m367_fail_lad_couple_fiber_tension_rupture_rate_fixed(tmp_path: Path):
    c1 = f"{162.0:>20.4f}{486.0:>20.4f}{72.0:>20.4f}{2.85:>20.4f}{0.965:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1825:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Fiber Tension Rupture Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_FIBER_TENSION_RUPTURE_RATE/1825
Ladeveze Coupled Fiber Tension Rupture Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1825 in model.fail_ladcouplefibertensionrupturerates
    fcftrr = model.fail_ladcouplefibertensionrupturerates[1825]
    assert pytest.approx(fcftrr.sigma_cftrr0) == 162.0
    assert pytest.approx(fcftrr.sigma_cftrrc) == 486.0
    assert pytest.approx(fcftrr.gamma_cftrr) == 72.0
    assert pytest.approx(fcftrr.p_cftrr) == 2.85
    assert pytest.approx(fcftrr.d_cftrr_max) == 0.965
    assert fcftrr.ifail_sh == 1
    assert fcftrr.ifail_so == 2
    assert fcftrr.fail_id == 1825
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_FIBER_TENSION_RUPTURE_RATE"


def test_m367_fail_lad_couple_fiber_tension_rupture_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Fiber Tension Rupture Rate Free Format Test
/FAIL/LAD_COUPLE_FIBER_TENSION_RUPTURE_RATE/1826
172.0, 516.0, 78.0, 3.05, 0.948
1, 1
1826
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1826 in model.fail_ladcouplefibertensionrupturerates
    fcftrr = model.fail_ladcouplefibertensionrupturerates[1826]
    assert pytest.approx(fcftrr.sigma_cftrr0) == 172.0
    assert pytest.approx(fcftrr.sigma_cftrrc) == 516.0
    assert pytest.approx(fcftrr.gamma_cftrr) == 78.0
    assert pytest.approx(fcftrr.p_cftrr) == 3.05
    assert pytest.approx(fcftrr.d_cftrr_max) == 0.948
    assert fcftrr.fail_id == 1826


def test_m367_fail_lad_couple_fiber_tension_rupture_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Fiber Tension Rupture Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_FIBER_TENSION_RUPTURE_RATE/1827
152.0, 456.0, 66.0, 2.35, 0.990
1, 1
/FAIL/LAD_CFTRR/1828
152.0, 456.0, 66.0, 2.35, 0.990
1, 1
/FAIL/LAD_CFTRR_MODEL/1829
152.0, 456.0, 66.0, 2.35, 0.990
1, 1
/FAIL/LAD_CFTRR_LAW/1830
152.0, 456.0, 66.0, 2.35, 0.990
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_FIBER_TENSION_RUPTURE/1831
152.0, 456.0, 66.0, 2.35, 0.990
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1827 in model.fail_ladcouplefibertensionrupturerates
    assert 1828 in model.fail_ladcouplefibertensionrupturerates
    assert 1829 in model.fail_ladcouplefibertensionrupturerates
    assert 1830 in model.fail_ladcouplefibertensionrupturerates
    assert 1831 in model.fail_ladcouplefibertensionrupturerates
    assert len(model.raw_fails) == 5


def test_m367_fail_lad_couple_fiber_tension_rupture_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_FIBER_TENSION_RUPTURE_RATE/1832
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m367_eng_flexomagnetoplasmonicmagnon_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0035:>20.4f}{115:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetoplasmonicmagnon Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPLASMONICMAGNON_RESONANCE_ENERGY/15
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 15 in model.eng_flexomagnetoplasmonicmagnon_resonance_energies
    eng = model.eng_flexomagnetoplasmonicmagnon_resonance_energies[15]
    assert pytest.approx(eng.dt_fmpmr) == 0.0035
    assert eng.sens_id == 115


def test_m367_eng_flexomagnetoplasmonicmagnon_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetoplasmonicmagnon Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPLASMONICMAGNON_RESONANCE_ENERGY/16
0.0045, 116
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 16 in model.eng_flexomagnetoplasmonicmagnon_resonance_energies
    eng = model.eng_flexomagnetoplasmonicmagnon_resonance_energies[16]
    assert pytest.approx(eng.dt_fmpmr) == 0.0045
    assert eng.sens_id == 116


def test_m367_eng_flexomagnetoplasmonicmagnon_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetoplasmonicmagnon Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PLASMON_MAGNON_RES_WORK/17
0.0055, 117
/ENG/EFLEXOMAGNETOPLASMONICMAGNONRESONANCE/18
0.0065, 118
/ENG/FLEXOMAGNETOPLASMONICMAGNON_RESONANCE_DISSIPATION/19
0.0075, 119
/ENG/EM_FLEXOMAGNETOPLASMONICMAGNON_RESONANCE/20
0.0085, 120
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 17 in model.eng_flexomagnetoplasmonicmagnon_resonance_energies
    assert 18 in model.eng_flexomagnetoplasmonicmagnon_resonance_energies
    assert 19 in model.eng_flexomagnetoplasmonicmagnon_resonance_energies
    assert 20 in model.eng_flexomagnetoplasmonicmagnon_resonance_energies


def test_m367_eng_flexomagnetoplasmonicmagnon_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPLASMONICMAGNON_RESONANCE_ENERGY/21
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m367_lagmul_phillips_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{5.0e6:>20.4f}{3:>10d}{1.0e-5:>20.4e}"
    c2 = f"{48.0:>20.4f}{36.0:>20.4f}{45.0:>20.4f}{12.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Phillips Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/PHILLIPS_SPATIAL_LINKAGE_JOINT/55
Phillips Spatial 6R Multi-Loop Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 55 in model.lagmul_phillips_spatial_linkage_joints
    joint = model.lagmul_phillips_spatial_linkage_joints[55]
    assert joint.node1 == 101
    assert joint.node2 == 102
    assert joint.node3 == 103
    assert pytest.approx(joint.stiff) == 5.0e6
    assert joint.skew_id == 3
    assert pytest.approx(joint.tol) == 1.0e-5
    assert pytest.approx(joint.link_len_a) == 48.0
    assert pytest.approx(joint.link_len_b) == 36.0
    assert pytest.approx(joint.twist_angle_alpha) == 45.0
    assert pytest.approx(joint.offset_distance_s) == 12.0


def test_m367_lagmul_phillips_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Phillips Spatial Linkage Joint Free Format Test
/LAGMUL/PHILLIPS_SPATIAL_LINKAGE_JOINT/56
201, 202, 203, 6.5e6, 5, 2.0e-5
52.0, 42.0, 60.0, 15.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 56 in model.lagmul_phillips_spatial_linkage_joints
    joint = model.lagmul_phillips_spatial_linkage_joints[56]
    assert joint.node1 == 201
    assert joint.node2 == 202
    assert joint.node3 == 203
    assert pytest.approx(joint.stiff) == 6.5e6
    assert joint.skew_id == 5
    assert pytest.approx(joint.tol) == 2.0e-5
    assert pytest.approx(joint.link_len_a) == 52.0
    assert pytest.approx(joint.link_len_b) == 42.0
    assert pytest.approx(joint.twist_angle_alpha) == 60.0
    assert pytest.approx(joint.offset_distance_s) == 15.0


def test_m367_lagmul_phillips_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Phillips Spatial Linkage Joint Aliases Test
/PHILLIPS_SPATIAL_LINKAGE_JOINT/57
301, 302, 303, 1.0e6, 0, 1.0e-6
30.0, 20.0, 30.0, 5.0
/LAGMUL/PHILLIPS_SPATIAL_LINKAGE/58
301, 302, 303, 1.0e6, 0, 1.0e-6
30.0, 20.0, 30.0, 5.0
/PHILLIPS_SPATIAL_LINKAGE/59
301, 302, 303, 1.0e6, 0, 1.0e-6
30.0, 20.0, 30.0, 5.0
/PHILLIPS_SPATIAL_MULTI_LOOP_MECHANISM/60
301, 302, 303, 1.0e6, 0, 1.0e-6
30.0, 20.0, 30.0, 5.0
/PHILLIPS_SPATIAL_SYMMETRIC_MECHANISM/61
301, 302, 303, 1.0e6, 0, 1.0e-6
30.0, 20.0, 30.0, 5.0
/PHILLIPS_SPATIAL_6R_MECHANISM/62
301, 302, 303, 1.0e6, 0, 1.0e-6
30.0, 20.0, 30.0, 5.0
/PHILLIPS_SPATIAL_OVERCONSTRAINED_MECHANISM/63
301, 302, 303, 1.0e6, 0, 1.0e-6
30.0, 20.0, 30.0, 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(57, 64):
        assert jid in model.lagmul_phillips_spatial_linkage_joints


def test_m367_lagmul_phillips_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/PHILLIPS_SPATIAL_LINKAGE_JOINT/64
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m367_sensor_spring_normal_snap_rate_fixed(tmp_path: Path):
    c1 = f"{1148:>10d}{9.2e7:>20.4f}{0.0265:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_SNAP_RATE/1
Fixed Spring Normal Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_snap_rates
    s1 = model.sensor_spring_normal_snap_rates[1]
    assert s1.spring_id == 1148
    assert pytest.approx(s1.jnorm_snp_max) == 9.2e7
    assert pytest.approx(s1.t_delay) == 0.0265
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_SNAP_RATE"


def test_m367_sensor_spring_normal_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Normal Snap Rate Free Format Test
/SENSOR/SPRING_NORMAL_SNAP_RATE/2
Free Spring Normal Snap Rate Sensor
1149, 1.25e8, 0.0385
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_normal_snap_rates
    s2 = model.sensor_spring_normal_snap_rates[2]
    assert s2.spring_id == 1149
    assert pytest.approx(s2.jnorm_snp_max) == 1.25e8
    assert pytest.approx(s2.t_delay) == 0.0385


def test_m367_sensor_spring_normal_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Snap Rate Aliases Test
/SENSOR/SPRING_NORM_SNAP_RATE/170
1181, 6.25e7, 0.0075
/SENSOR/SPRING_RATE_SNAP_NORM/171
1182, 6.45e7, 0.0085
/SENSOR/NORMAL_SNAP_RATE_SPRING/172
1183, 6.65e7, 0.0095
/SENSOR/SPRING_SNAP_NORM/173
1184, 6.85e7, 0.0105
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(170, 174):
        assert sid in model.sensor_spring_normal_snap_rates


def test_m367_sensor_spring_normal_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_NORMAL_SNAP_RATE/175
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

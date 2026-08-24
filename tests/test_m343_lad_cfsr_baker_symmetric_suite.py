"""Tests for Milestone M343: LadCoupleFiberSplittingRate Failure Model, EngFlexothermoplasmonicResonanceEnergy, BakerSymmetricLinkageJoint, and SensorSpringNormalDriftRate."""

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


def test_m343_fail_lad_couple_fiber_splitting_rate_fixed(tmp_path: Path):
    c1 = f"{225.0:>20.4f}{610.0:>20.4f}{51.0:>20.4f}{1.98:>20.4f}{0.960:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1590:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Fiber Splitting Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_FIBER_SPLITTING_RATE/1590
Ladeveze Coupled Fiber Splitting Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1590 in model.fail_ladcouplefibersplittingrates
    fcfsr = model.fail_ladcouplefibersplittingrates[1590]
    assert pytest.approx(fcfsr.sigma_cfsr0) == 225.0
    assert pytest.approx(fcfsr.sigma_cfsrc) == 610.0
    assert pytest.approx(fcfsr.gamma_cfsr) == 51.0
    assert pytest.approx(fcfsr.p_cfsr) == 1.98
    assert pytest.approx(fcfsr.d_cfsr_max) == 0.960
    assert fcfsr.ifail_sh == 1
    assert fcfsr.ifail_so == 2
    assert fcfsr.fail_id == 1590
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_FIBER_SPLITTING_RATE"


def test_m343_fail_lad_couple_fiber_splitting_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Fiber Splitting Rate Free Format Test
/FAIL/LAD_COUPLE_FIBER_SPLITTING_RATE/1591
235.0, 660.0, 54.0, 2.20, 0.940
1, 1
1591
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1591 in model.fail_ladcouplefibersplittingrates
    fcfsr = model.fail_ladcouplefibersplittingrates[1591]
    assert pytest.approx(fcfsr.sigma_cfsr0) == 235.0
    assert pytest.approx(fcfsr.sigma_cfsrc) == 660.0
    assert pytest.approx(fcfsr.gamma_cfsr) == 54.0
    assert pytest.approx(fcfsr.p_cfsr) == 2.20
    assert pytest.approx(fcfsr.d_cfsr_max) == 0.940
    assert fcfsr.fail_id == 1591


def test_m343_fail_lad_couple_fiber_splitting_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Fiber Splitting Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_FIBER_SPLITTING_RATE/1592
185.0, 500.0, 40.0, 1.75, 0.945
1, 1
/FAIL/LAD_CFSR/1593
185.0, 500.0, 40.0, 1.75, 0.945
1, 1
/FAIL/LAD_CFSR_MODEL/1594
185.0, 500.0, 40.0, 1.75, 0.945
1, 1
/FAIL/LAD_CFSR_LAW/1595
185.0, 500.0, 40.0, 1.75, 0.945
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_FIBER_SPLITTING/1596
185.0, 500.0, 40.0, 1.75, 0.945
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1592 in model.fail_ladcouplefibersplittingrates
    assert 1593 in model.fail_ladcouplefibersplittingrates
    assert 1594 in model.fail_ladcouplefibersplittingrates
    assert 1595 in model.fail_ladcouplefibersplittingrates
    assert 1596 in model.fail_ladcouplefibersplittingrates
    assert len(model.raw_fails) == 5


def test_m343_fail_lad_couple_fiber_splitting_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_COUPLE_FIBER_SPLITTING_RATE/1597
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m343_eng_flexothermoplasmonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00078:>20.6f}{160:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoplasmonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPLASMONIC_RESONANCE_ENERGY/1
Flexothermoplasmonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoplasmonic_resonance_energies
    eng = model.eng_flexothermoplasmonic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftplr) == 0.00078
    assert eng.sens_id == 160


def test_m343_eng_flexothermoplasmonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermoplasmonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPLASMONIC_RESONANCE_ENERGY/2
0.00092, 170
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermoplasmonic_resonance_energies
    eng = model.eng_flexothermoplasmonic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftplr) == 0.00092
    assert eng.sens_id == 170


def test_m343_eng_flexothermoplasmonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoplasmonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PLASMON_RES_WORK/3
0.00062, 126
/ENG/EFLEXOTHERMOPLASMONICRESONANCE/4
0.00065, 132
/ENG/FLEXOTHERMOPLASMONIC_RESONANCE_DISSIPATION/5
0.00072, 142
/ENG/EM_FLEXOTHERMOPLASMONIC_RESONANCE/6
0.00076, 152
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoplasmonic_resonance_energies
    assert 4 in model.eng_flexothermoplasmonic_resonance_energies
    assert 5 in model.eng_flexothermoplasmonic_resonance_energies
    assert 6 in model.eng_flexothermoplasmonic_resonance_energies


def test_m343_eng_flexothermoplasmonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOPLASMONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m343_lagmul_baker_symmetric_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{291:>10d}{292:>10d}{293:>10d}{2.45e7:>20.1f}{78:>10d}{4.2e-5:>20.6e}"
    c2 = f"{136.0:>20.4f}{126.0:>20.4f}{108.0:>20.4f}{65.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Baker Symmetric Linkage Joint Fixed Format Test
2022 0
/BAKER_SYMMETRIC_LINKAGE_JOINT/255
Baker Symmetric Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 255 in model.lagmul_baker_symmetric_linkage_joints
    joint = model.lagmul_baker_symmetric_linkage_joints[255]
    assert joint.node1 == 291
    assert joint.node2 == 292
    assert joint.node3 == 293
    assert pytest.approx(joint.stiff) == 2.45e7
    assert joint.skew_id == 78
    assert pytest.approx(joint.tol) == 4.2e-5
    assert pytest.approx(joint.link_len_a) == 136.0
    assert pytest.approx(joint.link_len_b) == 126.0
    assert pytest.approx(joint.twist_angle_alpha) == 108.0
    assert pytest.approx(joint.offset_distance_r) == 65.0


def test_m343_lagmul_baker_symmetric_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Baker Symmetric Linkage Joint Free Format Test
/LAGMUL/BAKER_SYMMETRIC_LINKAGE_JOINT/256
391, 392, 393, 12.5e6, 98, 5.2e-5
138.0, 128.0, 110.0, 68.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 256 in model.lagmul_baker_symmetric_linkage_joints
    joint = model.lagmul_baker_symmetric_linkage_joints[256]
    assert joint.node1 == 391
    assert joint.node2 == 392
    assert joint.node3 == 393
    assert pytest.approx(joint.stiff) == 12.5e6
    assert joint.skew_id == 98
    assert pytest.approx(joint.tol) == 5.2e-5
    assert pytest.approx(joint.link_len_a) == 138.0
    assert pytest.approx(joint.link_len_b) == 128.0
    assert pytest.approx(joint.twist_angle_alpha) == 110.0
    assert pytest.approx(joint.offset_distance_r) == 68.0


def test_m343_lagmul_baker_symmetric_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Baker Symmetric Linkage Joint Aliases Test
/LAGMUL/BAKER_SYMMETRIC_LINKAGE/257
1, 2, 3, 1e6, 0, 1e-6
18.0, 18.0, 45.0, 8.0
/BAKER_SYMMETRIC_LINKAGE/258
1, 2, 3, 1e6, 0, 1e-6
18.0, 18.0, 45.0, 8.0
/BAKER_SYMMETRIC_MECHANISM/259
1, 2, 3, 1e6, 0, 1e-6
18.0, 18.0, 45.0, 8.0
/BAKER_SYMMETRIC_6R_MECHANISM/260
1, 2, 3, 1e6, 0, 1e-6
18.0, 18.0, 45.0, 8.0
/BAKER_SYMMETRIC_OVERCONSTRAINED_MECHANISM/261
1, 2, 3, 1e6, 0, 1e-6
18.0, 18.0, 45.0, 8.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 257 in model.lagmul_baker_symmetric_linkage_joints
    assert 258 in model.lagmul_baker_symmetric_linkage_joints
    assert 259 in model.lagmul_baker_symmetric_linkage_joints
    assert 260 in model.lagmul_baker_symmetric_linkage_joints
    assert 261 in model.lagmul_baker_symmetric_linkage_joints


def test_m343_lagmul_baker_symmetric_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/BAKER_SYMMETRIC_LINKAGE_JOINT/262
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m343_sensor_spring_normal_drift_rate_fixed(tmp_path: Path):
    c1 = f"{1035:>10d}{4.75e9:>20.1f}{0.0155:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Normal Drift Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_DRIFT_RATE/288
Spring Normal Drift Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 288 in model.sensor_spring_normal_drift_rates
    sensor = model.sensor_spring_normal_drift_rates[288]
    assert sensor.spring_id == 1035
    assert pytest.approx(sensor.jnorm_drift_max) == 4.75e9
    assert pytest.approx(sensor.t_delay) == 0.0155
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_DRIFT_RATE"


def test_m343_sensor_spring_normal_drift_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Normal Drift Rate Sensor Free Format Test
/SENSOR/SPRING_NORMAL_DRIFT_RATE/289
1036, 4.85e9, 0.0180
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 289 in model.sensor_spring_normal_drift_rates
    sensor = model.sensor_spring_normal_drift_rates[289]
    assert sensor.spring_id == 1036
    assert pytest.approx(sensor.jnorm_drift_max) == 4.85e9
    assert pytest.approx(sensor.t_delay) == 0.0180


def test_m343_sensor_spring_normal_drift_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Normal Drift Rate Sensor Aliases Test
/SENSOR/SPRING_NORM_DRIFT_RATE/290
1037, 4.9e9, 0.0085
/SENSOR/SPRING_RATE_DRIFT_NORM/291
1038, 4.9e9, 0.0085
/SENSOR/NORMAL_DRIFT_RATE_SPRING/292
1039, 4.9e9, 0.0085
/SENSOR/SPRING_DRIFT_NORM/293
1040, 4.9e9, 0.0085
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 290 in model.sensor_spring_normal_drift_rates
    assert 291 in model.sensor_spring_normal_drift_rates
    assert 292 in model.sensor_spring_normal_drift_rates
    assert 293 in model.sensor_spring_normal_drift_rates
    assert len(model.sensors) == 4


def test_m343_sensor_spring_normal_drift_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_NORMAL_DRIFT_RATE/294
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

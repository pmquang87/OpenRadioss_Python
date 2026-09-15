"""Tests for Milestone M409: LadCoupleCoreShearingRate Failure Model, EngElectrothermoflexoplasmonicphononicpolaritonicResonanceEnergy, LagrangeSpatialLinkageJoint, and SensorSpringNormalPopRate."""

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


def test_m409_fail_lad_couple_core_shearing_rate_fixed(tmp_path: Path):
    c1 = f"{615.0:>20.4f}{1845.0:>20.4f}{395.0:>20.4f}{8.25:>20.4f}{0.805:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2230:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Core Shearing Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_CORE_SHEARING_RATE/2230
Ladeveze Coupled Core Shearing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2230 in model.fail_ladcouplecoreshearingrates
    fccsr = model.fail_ladcouplecoreshearingrates[2230]
    assert pytest.approx(fccsr.sigma_ccsr0) == 615.0
    assert pytest.approx(fccsr.sigma_ccsrc) == 1845.0
    assert pytest.approx(fccsr.gamma_ccsr) == 395.0
    assert pytest.approx(fccsr.p_ccsr) == 8.25
    assert pytest.approx(fccsr.d_ccsr_max) == 0.805
    assert fccsr.ifail_sh == 1
    assert fccsr.ifail_so == 2
    assert fccsr.fail_id == 2230
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_CORE_SHEARING_RATE"


def test_m409_fail_lad_couple_core_shearing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Core Shearing Rate Free Format Test
/FAIL/LAD_COUPLE_CORE_SHEARING_RATE/2231
625.0, 1875.0, 405.0, 8.45, 0.795
1, 1
2231
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2231 in model.fail_ladcouplecoreshearingrates
    fccsr = model.fail_ladcouplecoreshearingrates[2231]
    assert pytest.approx(fccsr.sigma_ccsr0) == 625.0
    assert pytest.approx(fccsr.sigma_ccsrc) == 1875.0
    assert pytest.approx(fccsr.gamma_ccsr) == 405.0
    assert pytest.approx(fccsr.p_ccsr) == 8.45
    assert pytest.approx(fccsr.d_ccsr_max) == 0.795
    assert fccsr.fail_id == 2231


def test_m409_fail_lad_couple_core_shearing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Core Shearing Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_CORE_SHEARING_RATE/2232
605.0, 1815.0, 390.0, 8.05, 0.825
1, 1
/FAIL/LAD_CCSR/2233
605.0, 1815.0, 390.0, 8.05, 0.825
1, 1
/FAIL/LAD_CCSR_MODEL/2234
605.0, 1815.0, 390.0, 8.05, 0.825
1, 1
/FAIL/LAD_CCSR_LAW/2235
605.0, 1815.0, 390.0, 8.05, 0.825
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_CORE_SHEARING/2236
605.0, 1815.0, 390.0, 8.05, 0.825
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2232 in model.fail_ladcouplecoreshearingrates
    assert 2233 in model.fail_ladcouplecoreshearingrates
    assert 2234 in model.fail_ladcouplecoreshearingrates
    assert 2235 in model.fail_ladcouplecoreshearingrates
    assert 2236 in model.fail_ladcouplecoreshearingrates
    assert len(model.raw_fails) == 5


def test_m409_fail_lad_couple_core_shearing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_CORE_SHEARING_RATE/2237
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m409_eng_electrothermoflexoplasmonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0315:>20.4f}{535:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexoplasmonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY/435
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 435 in model.eng_electrothermoflexoplasmonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexoplasmonicphononicpolaritonic_resonance_energies[435]
    assert pytest.approx(eng.dt_etfppp) == 0.0315
    assert eng.sens_id == 535


def test_m409_eng_electrothermoflexoplasmonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexoplasmonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY/436
0.0325, 536
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 436 in model.eng_electrothermoflexoplasmonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexoplasmonicphononicpolaritonic_resonance_energies[436]
    assert pytest.approx(eng.dt_etfppp) == 0.0325
    assert eng.sens_id == 536


def test_m409_eng_electrothermoflexoplasmonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexoplasmonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_PLASMON_PHONON_POLARITON_RES_WORK/437
0.0335, 537
/ENG/EELECTROTHERMOFLEXOPLASMONICPHONONICPOLARITONICRESONANCE/438
0.0345, 538
/ENG/ELECTROTHERMOFLEXOPLASMONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/439
0.0355, 539
/ENG/ET_ELECTROTHERMOFLEXOPLASMONICPHONONICPOLARITONIC_RESONANCE/440
0.0365, 540
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 437 in model.eng_electrothermoflexoplasmonicphononicpolaritonic_resonance_energies
    assert 438 in model.eng_electrothermoflexoplasmonicphononicpolaritonic_resonance_energies
    assert 439 in model.eng_electrothermoflexoplasmonicphononicpolaritonic_resonance_energies
    assert 440 in model.eng_electrothermoflexoplasmonicphononicpolaritonic_resonance_energies


def test_m409_eng_electrothermoflexoplasmonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY/441
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m409_lagmul_lagrange_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{521:>10d}{522:>10d}{523:>10d}{3.76e7:>20.4f}{239:>10d}{2.75e-4:>20.4e}"
    c2 = f"{305.0:>20.4f}{288.0:>20.4f}{315.0:>20.4f}{238.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Lagrange Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/LAGRANGE_SPATIAL_LINKAGE_JOINT/485
Lagrange Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 485 in model.lagmul_lagrange_spatial_linkage_joints
    joint = model.lagmul_lagrange_spatial_linkage_joints[485]
    assert joint.node1 == 521
    assert joint.node2 == 522
    assert joint.node3 == 523
    assert pytest.approx(joint.stiff) == 3.76e7
    assert joint.skew_id == 239
    assert pytest.approx(joint.tol) == 2.75e-4
    assert pytest.approx(joint.link_len_a) == 305.0
    assert pytest.approx(joint.link_len_b) == 288.0
    assert pytest.approx(joint.twist_angle_alpha) == 315.0
    assert pytest.approx(joint.offset_distance_s) == 238.0
    assert pytest.approx(joint.offset_distance_r) == 238.0
    assert pytest.approx(joint.offset_distance_v) == 238.0
    assert pytest.approx(joint.offset_distance_h) == 238.0
    assert pytest.approx(joint.offset_distance_u) == 238.0


def test_m409_lagmul_lagrange_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Lagrange Spatial Linkage Joint Free Format Test
/LAGMUL/LAGRANGE_SPATIAL_LINKAGE_JOINT/486
621, 622, 623, 4.00e7, 240, 2.85e-4
325.0, 295.0, 335.0, 250.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 486 in model.lagmul_lagrange_spatial_linkage_joints
    joint = model.lagmul_lagrange_spatial_linkage_joints[486]
    assert joint.node1 == 621
    assert joint.node2 == 622
    assert joint.node3 == 623
    assert pytest.approx(joint.stiff) == 4.00e7
    assert joint.skew_id == 240
    assert pytest.approx(joint.tol) == 2.85e-4
    assert pytest.approx(joint.link_len_a) == 325.0
    assert pytest.approx(joint.link_len_b) == 295.0
    assert pytest.approx(joint.twist_angle_alpha) == 335.0
    assert pytest.approx(joint.offset_distance_s) == 250.0


def test_m409_lagmul_lagrange_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Lagrange Spatial Linkage Joint Aliases Test
/LAGRANGE_SPATIAL_LINKAGE_JOINT/487
721, 722, 723, 1.0e6, 0, 1.0e-6
265.0, 255.0, 265.0, 215.0
/LAGMUL/LAGRANGE_SPATIAL_LINKAGE/488
721, 722, 723, 1.0e6, 0, 1.0e-6
265.0, 255.0, 265.0, 215.0
/LAGRANGE_SPATIAL_LINKAGE/489
721, 722, 723, 1.0e6, 0, 1.0e-6
265.0, 255.0, 265.0, 215.0
/LAGRANGE_SPATIAL_MULTI_LOOP_MECHANISM/490
721, 722, 723, 1.0e6, 0, 1.0e-6
265.0, 255.0, 265.0, 215.0
/LAGRANGE_SPATIAL_SYMMETRIC_MECHANISM/491
721, 722, 723, 1.0e6, 0, 1.0e-6
265.0, 255.0, 265.0, 215.0
/LAGRANGE_SPATIAL_6R_MECHANISM/492
721, 722, 723, 1.0e6, 0, 1.0e-6
265.0, 255.0, 265.0, 215.0
/LAGRANGE_SPATIAL_OVERCONSTRAINED_MECHANISM/493
721, 722, 723, 1.0e6, 0, 1.0e-6
265.0, 255.0, 265.0, 215.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(487, 494):
        assert jid in model.lagmul_lagrange_spatial_linkage_joints


def test_m409_lagmul_lagrange_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/LAGRANGE_SPATIAL_LINKAGE_JOINT/494
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m409_sensor_spring_normal_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1568:>10d}{4.98e8:>20.4f}{0.1545:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_POP_RATE/1
Fixed Spring Normal Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_pop_rates
    s1 = model.sensor_spring_normal_pop_rates[1]
    assert s1.spring_id == 1568
    assert pytest.approx(s1.jnorm_pop_max) == 4.98e8
    assert pytest.approx(s1.jnorm_snp_max) == 4.98e8
    assert pytest.approx(s1.jnorm_crackle_max) == 4.98e8
    assert pytest.approx(s1.jnorm_shot_max) == 4.98e8
    assert pytest.approx(s1.jnorm_drop_max) == 4.98e8
    assert pytest.approx(s1.jnorm_lock_max) == 4.98e8
    assert pytest.approx(s1.jnorm_crk_max) == 4.98e8
    assert pytest.approx(s1.t_delay) == 0.1545
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_POP_RATE"


def test_m409_sensor_spring_normal_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Normal Pop Rate Free Format Test
/SENSOR/SPRING_NORMAL_POP_RATE/2
Free Spring Normal Pop Rate Sensor
1569, 5.58e8, 0.2535
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_normal_pop_rates
    s2 = model.sensor_spring_normal_pop_rates[2]
    assert s2.spring_id == 1569
    assert pytest.approx(s2.jnorm_pop_max) == 5.58e8
    assert pytest.approx(s2.t_delay) == 0.2535


def test_m409_sensor_spring_normal_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Pop Rate Aliases Test
/SENSOR/SPRING_NORM_POP_RATE/431
1589, 3.195e8, 0.2065
/SENSOR/SPRING_RATE_POP_NORM/432
1590, 3.215e8, 0.2075
/SENSOR/NORMAL_POP_RATE_SPRING/433
1591, 3.235e8, 0.2085
/SENSOR/SPRING_POP_NORM/434
1592, 3.255e8, 0.2095
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(431, 435):
        assert sid in model.sensor_spring_normal_pop_rates


def test_m409_sensor_spring_normal_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_NORMAL_POP_RATE/435
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0

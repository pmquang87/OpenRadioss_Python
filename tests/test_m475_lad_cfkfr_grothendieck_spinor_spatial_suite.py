"""Tests for Milestone M475:
- /FAIL/LAD_COUPLED_FIBER_KINKING_FAILURE_RATE (and aliases)
- /ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANTISKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY (and aliases)
- /LAGMUL/GROTHENDIECK_SPINOR_SPATIAL_LINKAGE_JOINT / /GROTHENDIECK_SPINOR_SPATIAL_LINKAGE_JOINT (and aliases)
- /SENSOR/SPRING_BENDING_CRACKLE_RATE (and aliases)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailLadCoupledFiberKinkingFailureRate,
    EngElectrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonicResonanceEnergy,
    LagmulGrothendieckSpinorSpatialLinkageJoint,
    SensorSpringBendingCrackleRate,
)


def _parse_starter(tmp_path: Path, deck_text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck_text, encoding="utf-8")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(p))
    parse_starter_deck(blocks, model, log)
    return model, log


# ============================================================================
# 1. FAIL_LAD_COUPLED_FIBER_KINKING_FAILURE_RATE Tests
# ============================================================================

def test_m475_fail_lad_coupled_fiber_kinking_failure_rate_fixed(tmp_path: Path):
    c1 = f"{135.0:>20.4f}{435.0:>20.4f}{0.36:>20.4f}{1.58:>20.4f}{0.988:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M475 Fail Lad Coupled Fiber Kinking Failure Rate Fixed
2022 0
/FAIL/LAD_COUPLED_FIBER_KINKING_FAILURE_RATE/1
Coupled Fiber Kinking Failure Rate Card Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.fail_ladcoupledfiberkinkingfailurerates
    fm = model.fail_ladcoupledfiberkinkingfailurerates[1]
    assert isinstance(fm, FailLadCoupledFiberKinkingFailureRate)
    assert fm.mat_id == 1
    assert pytest.approx(fm.sigma_cfkfr0) == 135.0
    assert pytest.approx(fm.sigma_cfkfrc) == 435.0
    assert pytest.approx(fm.gamma_cfkfr) == 0.36
    assert pytest.approx(fm.p_cfkfr) == 1.58
    assert pytest.approx(fm.d_cfkfr_max) == 0.988
    assert fm.ifail_sh == 2
    assert fm.ifail_so == 1

    # Properties aliases
    assert pytest.approx(fm.sigma_cfkf0) == 135.0
    assert pytest.approx(fm.sigma_cfkfc) == 435.0
    assert pytest.approx(fm.gamma_cfkf) == 0.36
    assert pytest.approx(fm.p_cfkf) == 1.58
    assert pytest.approx(fm.d_cfkf_max) == 0.988

    assert pytest.approx(fm.sigma_cfk0) == 135.0
    assert pytest.approx(fm.sigma_cfkc) == 435.0
    assert pytest.approx(fm.gamma_cfk) == 0.36
    assert pytest.approx(fm.p_cfk) == 1.58
    assert pytest.approx(fm.d_cfk_max) == 0.988


def test_m475_fail_lad_coupled_fiber_kinking_failure_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M475 Fail Lad Coupled Fiber Kinking Failure Rate Free
/FAIL/LAD_COUPLED_FIBER_KINKING_FAILURE_RATE/2
145.0, 465.0, 0.39, 1.68, 0.992
1, 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.fail_ladcoupledfiberkinkingfailurerates
    fm = model.fail_ladcoupledfiberkinkingfailurerates[2]
    assert pytest.approx(fm.sigma_cfkfr0) == 145.0
    assert pytest.approx(fm.sigma_cfkfrc) == 465.0
    assert pytest.approx(fm.gamma_cfkfr) == 0.39
    assert pytest.approx(fm.p_cfkfr) == 1.68
    assert pytest.approx(fm.d_cfkfr_max) == 0.992
    assert fm.ifail_sh == 1
    assert fm.ifail_so == 2


def test_m475_fail_lad_coupled_fiber_kinking_failure_rate_aliases(tmp_path: Path):
    aliases = [
        "/FAIL/LADEVEZE_COUPLED_FIBER_KINKING_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_FIBER_KINKING_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_FIBER_KINKING_FAILURE",
        "/FAIL/LAD_COUPLED_FIBER_KINK_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLED_FIBER_KINK_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_FIBER_KINK_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_FIBER_KINK_FAILURE",
        "/FAIL/LAD_COUPLED_FIBER_COMPRESSIVE_KINKING_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLED_FIBER_COMPRESSIVE_KINKING_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_FIBER_COMPRESSIVE_KINKING_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_FIBER_COMPRESSIVE_KINKING_FAILURE",
        "/FAIL/LAD_COUPLED_FIBER_MICROBUCKLING_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLED_FIBER_MICROBUCKLING_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_FIBER_MICROBUCKLING_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_FIBER_MICROBUCKLING_FAILURE",
        "/FAIL/LAD_COUPLED_FIBER_MICRO_BUCKLING_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLED_FIBER_MICRO_BUCKLING_FAILURE_RATE",
        "/FAIL/LAD_CFDFKFR",
        "/FAIL/LAD_CFDFKFR_MODEL",
        "/FAIL/LAD_CFDFKFR_LAW",
        "/FAIL/LAD_CFKFR",
        "/FAIL/LAD_CFKFR_MODEL",
        "/FAIL/LAD_CFKFR_LAW",
        "/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_FIBER_KINKING_FAILURE",
        "/FAIL/LAD_COUPLED_FIBER_KINKING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_COUPLED_FIBER_KINKING_DAMAGE_RATE",
        "/FAIL/LAD_COUPLED_FIBER_KINK_DAMAGE_RATE",
        "/FAIL/LADEVEZE_COUPLED_FIBER_KINK_DAMAGE_RATE",
        "/FAIL/LAD_COUPLED_FIBER_MICROBUCKLING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_COUPLED_FIBER_MICROBUCKLING_DAMAGE_RATE",
    ]
    deck_lines = ["/BEGIN", "Test M475 Fail Lad Coupled Fiber Kinking Failure Rate Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            f"{float(idx) * 10.0}, {float(idx) * 30.0}, 0.25, 1.4, 0.95",
            "1, 1"
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.fail_ladcoupledfiberkinkingfailurerates
        fm = model.fail_ladcoupledfiberkinkingfailurerates[idx]
        assert pytest.approx(fm.sigma_cfkfr0) == float(idx) * 10.0
        assert pytest.approx(fm.sigma_cfkfrc) == float(idx) * 30.0


def test_m475_fail_lad_coupled_fiber_kinking_failure_rate_missing_card(tmp_path: Path):
    deck = """/BEGIN
Test M475 Fail Missing Card
/FAIL/LAD_COUPLED_FIBER_KINKING_FAILURE_RATE/99
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
    assert "missing data card" in log.errors[0]
    assert 99 not in model.fail_ladcoupledfiberkinkingfailurerates


def test_m475_fail_lad_coupled_fiber_kinking_failure_rate_properties():
    fm = FailLadCoupledFiberKinkingFailureRate(mat_id=5)
    fm.sigma_cfkf0 = 155.0
    fm.sigma_cfkfc = 495.0
    fm.gamma_cfkf = 0.45
    fm.p_cfkf = 1.79
    fm.d_cfkf_max = 0.997

    assert pytest.approx(fm.sigma_cfkfr0) == 155.0
    assert pytest.approx(fm.sigma_cfkfrc) == 495.0
    assert pytest.approx(fm.gamma_cfkfr) == 0.45
    assert pytest.approx(fm.p_cfkfr) == 1.79
    assert pytest.approx(fm.d_cfkfr_max) == 0.997

    fm.sigma_cfk0 = 165.0
    fm.sigma_cfkc = 525.0
    fm.gamma_cfk = 0.48
    fm.p_cfk = 1.84
    fm.d_cfk_max = 0.998

    assert pytest.approx(fm.sigma_cfkfr0) == 165.0
    assert pytest.approx(fm.sigma_cfkfrc) == 525.0
    assert pytest.approx(fm.gamma_cfkfr) == 0.48
    assert pytest.approx(fm.p_cfkfr) == 1.84
    assert pytest.approx(fm.d_cfkfr_max) == 0.998


# ============================================================================
# 2. ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALANTISKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY Tests
# ============================================================================

def test_m475_eng_electrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0042:>20.4f}{18:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M475 Eng Chiral Antiskyrmion Resonance Energy Fixed
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANTISKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Eng Chiral Antiskyrmion Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonic_resonance_energies[1]
    assert isinstance(eng, EngElectrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonicResonanceEnergy)
    assert pytest.approx(eng.dt_etfcantiskyrmionplp) == 0.0042
    assert eng.sens_id == 18


def test_m475_eng_electrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M475 Eng Chiral Antiskyrmion Resonance Energy Free
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANTISKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.0052, 19
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfcantiskyrmionplp) == 0.0052
    assert eng.sens_id == 19


def test_m475_eng_electrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    aliases = [
        "/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_ANTISKYRMION_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALANTISKYRMIONPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANTISKYRMIONPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALANTISKYRMIONPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOANTISKYRMIONCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_ANTISKYRMION_CHIRAL_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOANTISKYRMIONCHIRALPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOANTISKYRMIONCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOANTISKYRMIONCHIRALPLASMONICPOLARITONIC_RESONANCE",
    ]
    deck_lines = ["/BEGIN", "Test M475 Eng Aliases"]
    for idx, kw in enumerate(aliases, start=20):
        deck_lines.extend([
            f"{kw}/{idx}",
            f"{0.001 * idx}, {idx}"
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(20, 20 + len(aliases)):
        assert idx in model.eng_electrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonic_resonance_energies
        eng = model.eng_electrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonic_resonance_energies[idx]
        assert pytest.approx(eng.dt_etfcantiskyrmionplp) == 0.001 * idx
        assert eng.sens_id == idx


def test_m475_eng_electrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """/BEGIN
Test Eng Missing Card
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANTISKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY/99
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
    assert "missing data card" in log.errors[0]
    assert 99 not in model.eng_electrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonic_resonance_energies


def test_m475_eng_electrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonic_resonance_energy_cascade():
    # Test fallback from preceding polaritonic directive (skyrmionium)
    eng1 = EngElectrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonicResonanceEnergy(dt_etfcskyrmioniumplp=0.0068)
    assert pytest.approx(eng1.dt_etfcantiskyrmionplp) == 0.0068

    # Test forward propagation from antiskyrmion to other polaritonic directives
    eng2 = EngElectrothermoflexomagnetochiralantiskyrmionplasmonicpolaritonicResonanceEnergy(dt_etfcantiskyrmionplp=0.0082)
    assert pytest.approx(eng2.dt_etfplp) == 0.0082
    assert pytest.approx(eng2.dt_etfcblochpointplp) == 0.0082
    assert pytest.approx(eng2.dt_etfcbobberplp) == 0.0082
    assert pytest.approx(eng2.dt_etfcskyrmioniumplp) == 0.0082


# ============================================================================
# 3. LAGMUL_GROTHENDIECK_SPINOR_SPATIAL_LINKAGE_JOINT Tests
# ============================================================================

def test_m475_lagmul_grothendieck_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{114:>10d}{115:>10d}{116:>10d}{3.1e6:>20.4f}{8:>10d}{2.0e-5:>20.4e}"
    c2 = f"{17.5:>20.4f}{27.5:>20.4f}{52.0:>20.4f}{7.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M475 Grothendieck Spinor Spatial Linkage Joint Fixed
/GROTHENDIECK_SPINOR_SPATIAL_LINKAGE_JOINT/1
Grothendieck Spinor Joint Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_grothendieck_spinor_spatial_linkage_joints
    j = model.lagmul_grothendieck_spinor_spatial_linkage_joints[1]
    assert isinstance(j, LagmulGrothendieckSpinorSpatialLinkageJoint)
    assert j.node1 == 114
    assert j.node2 == 115
    assert j.node3 == 116
    assert pytest.approx(j.stiff) == 3.1e6
    assert j.skew_id == 8
    assert pytest.approx(j.tol) == 2.0e-5
    assert pytest.approx(j.link_len_a) == 17.5
    assert pytest.approx(j.link_len_b) == 27.5
    assert pytest.approx(j.twist_angle_alpha) == 52.0
    assert pytest.approx(j.offset_distance_s) == 7.5

    # Offset distance properties
    assert pytest.approx(j.offset_distance_r) == 7.5
    assert pytest.approx(j.offset_distance_v) == 7.5
    assert pytest.approx(j.offset_distance_h) == 7.5
    assert pytest.approx(j.offset_distance_u) == 7.5
    assert pytest.approx(j.offset_distance_f) == 7.5


def test_m475_lagmul_grothendieck_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M475 Grothendieck Spinor Spatial Linkage Joint Free
/LAGMUL/GROTHENDIECK_SPINOR_SPATIAL_LINKAGE_JOINT/2
221, 222, 223, 3.8e6, 9, 2.5e-5
21.0, 31.0, 68.0, 9.5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_grothendieck_spinor_spatial_linkage_joints
    j = model.lagmul_grothendieck_spinor_spatial_linkage_joints[2]
    assert j.node1 == 221
    assert j.node2 == 222
    assert j.node3 == 223
    assert pytest.approx(j.stiff) == 3.8e6
    assert j.skew_id == 9
    assert pytest.approx(j.tol) == 2.5e-5
    assert pytest.approx(j.link_len_a) == 21.0
    assert pytest.approx(j.link_len_b) == 31.0
    assert pytest.approx(j.twist_angle_alpha) == 68.0
    assert pytest.approx(j.offset_distance_s) == 9.5


def test_m475_lagmul_grothendieck_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    aliases = [
        "/GROTHENDIECK_SPINOR_SPATIAL_LINKAGE",
        "/LAGMUL/GROTHENDIECK_SPINOR_SPATIAL_LINKAGE",
        "/GROTHENDIECK_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM",
        "/GROTHENDIECK_SPINOR_SPATIAL_SYMMETRIC_MECHANISM",
        "/GROTHENDIECK_SPINOR_SPATIAL_6R_MECHANISM",
        "/GROTHENDIECK_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM",
        "/LAGMUL/GROTHENDIECK_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/GROTHENDIECK_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/GROTHENDIECK_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/GROTHENDIECK_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/GROTHENDIECK_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/GROTHENDIECK_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/GROTHENDIECK_RIEMANN_ROCH_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/GROTHENDIECK_RIEMANN_ROCH_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/GROTHENDIECK_RIEMANN_ROCH_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/GROTHENDIECK_RIEMANN_ROCH_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/GROTHENDIECK_MOTIVIC_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/GROTHENDIECK_MOTIVIC_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/GROTHENDIECK_MOTIVIC_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/GROTHENDIECK_MOTIVIC_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/GROTHENDIECK_TOPOS_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/GROTHENDIECK_TOPOS_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/GROTHENDIECK_TOPOS_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/GROTHENDIECK_TOPOS_TWISTOR_SPATIAL_LINKAGE_JOINT",
    ]
    deck_lines = ["/BEGIN", "Test M475 Grothendieck Joint Aliases"]
    for idx, kw in enumerate(aliases, start=30):
        deck_lines.extend([
            f"{kw}/{idx}",
            f"{idx * 10}, {idx * 10 + 1}, {idx * 10 + 2}, 1.0e6, 0, 1.0e-6",
            "10.0, 20.0, 30.0, 2.0"
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(30, 30 + len(aliases)):
        assert idx in model.lagmul_grothendieck_spinor_spatial_linkage_joints
        j = model.lagmul_grothendieck_spinor_spatial_linkage_joints[idx]
        assert j.node1 == idx * 10
        assert j.node2 == idx * 10 + 1
        assert j.node3 == idx * 10 + 2


def test_m475_lagmul_grothendieck_spinor_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """/BEGIN
Test Joint Missing Card
/GROTHENDIECK_SPINOR_SPATIAL_LINKAGE_JOINT/99
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
    assert "missing data card 1" in log.errors[0]
    assert 99 not in model.lagmul_grothendieck_spinor_spatial_linkage_joints


def test_m475_lagmul_grothendieck_spinor_spatial_linkage_joint_properties():
    j = LagmulGrothendieckSpinorSpatialLinkageJoint(id=9)
    j.offset_distance_r = 14.5
    assert pytest.approx(j.offset_distance_s) == 14.5

    j.offset_distance_v = 16.5
    assert pytest.approx(j.offset_distance_s) == 16.5

    j.offset_distance_h = 18.5
    assert pytest.approx(j.offset_distance_s) == 18.5

    j.offset_distance_u = 20.5
    assert pytest.approx(j.offset_distance_s) == 20.5

    j.offset_distance_f = 22.5
    assert pytest.approx(j.offset_distance_s) == 22.5


# ============================================================================
# 4. SENSOR_SPRING_BENDING_CRACKLE_RATE Tests
# ============================================================================

def test_m475_sensor_spring_bending_crackle_rate(tmp_path: Path):
    # Fixed format
    c1 = f"{75:>10d}{8.5e5:>20.4e}{0.0030:>20.4e}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M475 Sensor Spring Bending Crackle Rate
/SENSOR/SPRING_BENDING_CRACKLE_RATE/1
{c1}
/SENSOR/SPRING_BEND_CRACKLE_RATE/2
76, 9.5e5, 0.0040
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_crackle_rates
    s1 = model.sensor_spring_bending_crackle_rates[1]
    assert isinstance(s1, SensorSpringBendingCrackleRate)
    assert s1.spring_id == 75
    assert pytest.approx(s1.jbend_pop_max) == 8.5e5
    assert pytest.approx(s1.t_delay) == 0.0030

    # M475 25th rate-of-change property accessors
    assert pytest.approx(s1.jbend_crackle_max) == 8.5e5
    assert pytest.approx(s1.j_bend_crk_max) == 8.5e5
    assert pytest.approx(s1.j_bend_crackle_max) == 8.5e5
    assert pytest.approx(s1.j_bending_crk_max) == 8.5e5
    assert pytest.approx(s1.j_bending_crackle_max) == 8.5e5
    assert pytest.approx(s1.j_rot_crk_max) == 8.5e5
    assert pytest.approx(s1.j_rot_crackle_max) == 8.5e5
    assert pytest.approx(s1.j_rotational_crk_max) == 8.5e5
    assert pytest.approx(s1.j_rotational_crackle_max) == 8.5e5
    assert pytest.approx(s1.j_crk_bend_max) == 8.5e5
    assert pytest.approx(s1.j_crackle_bend_max) == 8.5e5
    assert pytest.approx(s1.j_crk_rot_max) == 8.5e5
    assert pytest.approx(s1.j_crackle_rot_max) == 8.5e5
    assert pytest.approx(s1.jbend_crackle_rate_max) == 8.5e5
    assert pytest.approx(s1.jbend_snap_rate_max) == 8.5e5
    assert pytest.approx(s1.jbend_pop_rate_max) == 8.5e5
    assert pytest.approx(s1.jbend_lock_rate_max) == 8.5e5

    # Free format
    assert 2 in model.sensor_spring_bending_crackle_rates
    s2 = model.sensor_spring_bending_crackle_rates[2]
    assert s2.spring_id == 76
    assert pytest.approx(s2.jbend_pop_max) == 9.5e5
    assert pytest.approx(s2.t_delay) == 0.0040

    # Setters
    s1.jbend_crackle_max = 1.1e6
    assert pytest.approx(s1.jbend_pop_max) == 1.1e6
    s1.j_rot_crk_max = 1.25e6
    assert pytest.approx(s1.jbend_pop_max) == 1.25e6
    s1.j_bending_crackle_max = 1.35e6
    assert pytest.approx(s1.jbend_pop_max) == 1.35e6


def test_m475_sensor_spring_bending_crackle_rate_aliases(tmp_path: Path):
    aliases = [
        "/SENSOR/SPRING_BEND_CRACKLE_RATE",
        "/SENSOR/SPRING_RATE_BENDING_CRACKLE",
        "/SENSOR/SPRING_RATE_BEND_CRACKLE",
        "/SENSOR/SPRING_BENDING_CRACKLE_RATE_SENSOR",
        "/SENSOR/SPRING_BEND_CRACKLE_RATE_SENSOR",
        "/SENSOR/BENDING_CRACKLE_RATE_SPRING_SENSOR",
        "/SENSOR/BEND_CRACKLE_RATE_SPRING_SENSOR",
        "/SENSOR/SPRING_CRACKLE_RATE_BENDING_SENSOR",
        "/SENSOR/SPRING_CRACKLE_RATE_BEND_SENSOR",
        "/SENSOR/SPRING_RATE_CRACKLE_BENDING",
        "/SENSOR/SPRING_RATE_CRACKLE_BEND",
        "/SENSOR/SPRING_RATE_BENDING_CRK",
        "/SENSOR/SPRING_RATE_BEND_CRK",
        "/SENSOR/SPRING_CRACKLE_BENDING",
        "/SENSOR/SPRING_CRACKLE_BEND",
        "/SENSOR/SPRING_SNAP_RATE_BENDING_CRACKLE",
        "/SENSOR/SPRING_SNAP_RATE_BEND_CRACKLE",
        "/SENSOR/SPRING_LOCK_RATE_BENDING_CRACKLE",
        "/SENSOR/SPRING_LOCK_RATE_BEND_CRACKLE",
        "/SENSOR/SPRING_ROT_CRACKLE_RATE",
        "/SENSOR/SPRING_ROTATIONAL_CRACKLE_RATE",
        "/SENSOR/SPRING_RATE_ROT_CRACKLE",
        "/SENSOR/SPRING_RATE_ROTATIONAL_CRACKLE",
        "/SENSOR/SPRING_ROT_CRACKLE_RATE_SENSOR",
        "/SENSOR/SPRING_ROTATIONAL_CRACKLE_RATE_SENSOR",
        "/SENSOR/ROT_CRACKLE_RATE_SPRING_SENSOR",
        "/SENSOR/ROTATIONAL_CRACKLE_RATE_SPRING_SENSOR",
        "/SENSOR/SPRING_CRACKLE_RATE_ROT_SENSOR",
        "/SENSOR/SPRING_CRACKLE_RATE_ROTATIONAL_SENSOR",
        "/SENSOR/SPRING_RATE_CRACKLE_ROT",
        "/SENSOR/SPRING_RATE_CRACKLE_ROTATIONAL",
        "/SENSOR/SPRING_RATE_ROT_CRK",
        "/SENSOR/SPRING_RATE_ROTATIONAL_CRK",
        "/SENSOR/SPRING_CRACKLE_ROT",
        "/SENSOR/SPRING_CRACKLE_ROTATIONAL",
        "/SENSOR/SPRING_SNAP_RATE_ROT_CRACKLE",
        "/SENSOR/SPRING_SNAP_RATE_ROTATIONAL_CRACKLE",
        "/SENSOR/SPRING_LOCK_RATE_ROT_CRACKLE",
        "/SENSOR/SPRING_LOCK_RATE_ROTATIONAL_CRACKLE",
    ]
    deck_lines = ["/BEGIN", "Test M475 Sensor Aliases"]
    for idx, kw in enumerate(aliases, start=70):
        deck_lines.extend([
            f"{kw}/{idx}",
            f"{idx + 100}, {idx * 1e4}, 0.001"
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(70, 70 + len(aliases)):
        assert idx in model.sensor_spring_bending_crackle_rates
        s = model.sensor_spring_bending_crackle_rates[idx]
        assert s.spring_id == idx + 100
        assert pytest.approx(s.jbend_pop_max) == idx * 1e4

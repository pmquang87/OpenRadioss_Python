"""Tests for Milestone M463:
- /FAIL/LAD_COUPLED_INTERLAMINAR_SHEAR_FAILURE_RATE (and aliases)
- /ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBIMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY (and aliases)
- /LAGMUL/JACOBI_SPINOR_SPATIAL_LINKAGE_JOINT / /JACOBI_SPINOR_SPATIAL_LINKAGE_JOINT (and aliases)
- /SENSOR/SPRING_BENDING_LOCK_RATE (and aliases)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailLadCoupledInterlaminarShearFailureRate,
    EngElectrothermoflexomagnetochiralbimeronplasmonicpolaritonicResonanceEnergy,
    LagmulJacobiSpinorSpatialLinkageJoint,
    SensorSpringBendingLockRate,
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
# 1. FAIL_LAD_COUPLED_INTERLAMINAR_SHEAR_FAILURE_RATE Tests
# ============================================================================

def test_m463_fail_lad_coupled_interlaminar_shear_failure_rate_fixed(tmp_path: Path):
    c1 = f"{80.0:>20.4f}{300.0:>20.4f}{0.15:>20.4f}{1.35:>20.4f}{0.93:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M463 Fail Lad Coupled Interlaminar Shear Failure Rate Fixed
2022 0
/FAIL/LAD_COUPLED_INTERLAMINAR_SHEAR_FAILURE_RATE/1
Coupled Interlaminar Shear Failure Rate Card Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.fail_ladcoupledinterlaminarshearfailurerates
    fm = model.fail_ladcoupledinterlaminarshearfailurerates[1]
    assert isinstance(fm, FailLadCoupledInterlaminarShearFailureRate)
    assert fm.mat_id == 1
    assert pytest.approx(fm.sigma_cisfr0) == 80.0
    assert pytest.approx(fm.sigma_cisfrc) == 300.0
    assert pytest.approx(fm.gamma_cisfr) == 0.15
    assert pytest.approx(fm.p_cisfr) == 1.35
    assert pytest.approx(fm.d_cisfr_max) == 0.93
    assert fm.ifail_sh == 2
    assert fm.ifail_so == 1

    # Properties
    assert pytest.approx(fm.sigma_cisf0) == 80.0
    assert pytest.approx(fm.sigma_cisfc) == 300.0
    assert pytest.approx(fm.gamma_cisf) == 0.15
    assert pytest.approx(fm.p_cisf) == 1.35
    assert pytest.approx(fm.d_cisf_max) == 0.93

    assert pytest.approx(fm.sigma_cis0) == 80.0
    assert pytest.approx(fm.sigma_cisc) == 300.0
    assert pytest.approx(fm.gamma_cis) == 0.15
    assert pytest.approx(fm.p_cis) == 1.35
    assert pytest.approx(fm.d_cis_max) == 0.93

    assert pytest.approx(fm.sigma_cir0) == 80.0
    assert pytest.approx(fm.sigma_circ) == 300.0
    assert pytest.approx(fm.gamma_cir) == 0.15
    assert pytest.approx(fm.p_cir) == 1.35
    assert pytest.approx(fm.d_cir_max) == 0.93


def test_m463_fail_lad_coupled_interlaminar_shear_failure_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M463 Fail Lad Coupled Interlaminar Shear Failure Rate Free
/FAIL/LAD_COUPLED_INTERLAMINAR_SHEAR_FAILURE_RATE/2
90.0, 320.0, 0.18, 1.45, 0.95
1, 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.fail_ladcoupledinterlaminarshearfailurerates
    fm = model.fail_ladcoupledinterlaminarshearfailurerates[2]
    assert pytest.approx(fm.sigma_cisfr0) == 90.0
    assert pytest.approx(fm.sigma_cisfrc) == 320.0
    assert pytest.approx(fm.gamma_cisfr) == 0.18
    assert pytest.approx(fm.p_cisfr) == 1.45
    assert pytest.approx(fm.d_cisfr_max) == 0.95
    assert fm.ifail_sh == 1
    assert fm.ifail_so == 2


def test_m463_fail_lad_coupled_interlaminar_shear_failure_rate_aliases(tmp_path: Path):
    aliases = [
        "/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_SHEAR_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_INTERLAMINAR_SHEAR_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_SHEAR_FAILURE",
        "/FAIL/LAD_COUPLE_INTERLAMINAR_SHEAR_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLE_INTERLAMINAR_SHEAR_FAILURE_RATE",
        "/FAIL/LAD_COUPLE_INTERLAMINAR_SHEAR_FAILURE",
        "/FAIL/LADEVEZE_COUPLE_INTERLAMINAR_SHEAR_FAILURE",
        "/FAIL/LAD_COUPLED_INTERLAMINAR_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_FAILURE",
        "/FAIL/LAD_COUPLE_INTERLAMINAR_FAILURE",
        "/FAIL/LADEVEZE_COUPLE_INTERLAMINAR_FAILURE",
        "/FAIL/LAD_COUPLED_INTERLAMINAR_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_FAILURE_RATE",
        "/FAIL/LAD_COUPLE_INTERLAMINAR_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLE_INTERLAMINAR_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_INTERLAMINAR_DELAMINATION_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_DELAMINATION_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_INTERLAMINAR_SHEAR_DELAMINATION_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_SHEAR_DELAMINATION_FAILURE_RATE",
        "/FAIL/LAD_CISFR",
        "/FAIL/LAD_CISFR_MODEL",
        "/FAIL/LAD_CISFR_LAW",
        "/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_INTERLAMINAR_SHEAR_FAILURE",
        "/FAIL/LAD_COUPLED_INTERLAMINAR_SHEAR_DAMAGE_RATE",
        "/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_SHEAR_DAMAGE_RATE",
        "/FAIL/LAD_COUPLE_INTERLAMINAR_SHEAR_DAMAGE_RATE",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test M463 Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "72.0, 270.0, 0.13, 1.20, 0.96",
            "1, 1",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.fail_ladcoupledinterlaminarshearfailurerates
        fm = model.fail_ladcoupledinterlaminarshearfailurerates[idx]
        assert pytest.approx(fm.sigma_cisfr0) == 72.0


# ============================================================================
# 2. ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALBIMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY Tests
# ============================================================================

def test_m463_eng_energy_fixed(tmp_path: Path):
    c1 = f"{2.1e-4:>20.6f}{17:>10d}"
    deck = f"""# RADIOSS ENGINE DECK
/BEGIN
Test M463 Eng Output Fixed
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBIMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Bimeron Resonance Energy Directive Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralbimeronplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralbimeronplasmonicpolaritonic_resonance_energies[1]
    assert isinstance(eng, EngElectrothermoflexomagnetochiralbimeronplasmonicpolaritonicResonanceEnergy)
    assert pytest.approx(eng.dt_etfcbimeronplp) == 2.1e-4
    assert eng.sens_id == 17


def test_m463_eng_energy_free(tmp_path: Path):
    deck = """# RADIOSS ENGINE DECK
/BEGIN
Test M463 Eng Output Free
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBIMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
3.1e-4, 20
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralbimeronplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralbimeronplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfcbimeronplp) == 3.1e-4
    assert eng.sens_id == 20


def test_m463_eng_energy_aliases(tmp_path: Path):
    aliases = [
        "/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_BIMERON_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALBIMERONPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBIMERONPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALBIMERONPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOBIMERONCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_BIMERON_CHIRAL_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOBIMERONCHIRALPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOBIMERONCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOBIMERONCHIRALPLASMONICPOLARITONIC_RESONANCE",
    ]
    deck_lines = ["# RADIOSS ENGINE DECK", "/BEGIN", "Test Eng Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "4.1e-4, 27",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.eng_electrothermoflexomagnetochiralbimeronplasmonicpolaritonic_resonance_energies
        eng = model.eng_electrothermoflexomagnetochiralbimeronplasmonicpolaritonic_resonance_energies[idx]
        assert pytest.approx(eng.dt_etfcbimeronplp) == 4.1e-4


# ============================================================================
# 3. LAGMUL_JACOBI_SPINOR_SPATIAL_LINKAGE_JOINT Tests
# ============================================================================

def test_m463_jacobi_spinor_joint_fixed(tmp_path: Path):
    c1 = f"{120:>10d}{121:>10d}{122:>10d}{3.0e6:>20.4f}{10:>10d}{1.5e-5:>20.6f}"
    c2 = f"{19.5:>20.4f}{26.0:>20.4f}{65.0:>20.4f}{12.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M463 Jacobi Joint Fixed
2022 0
/JACOBI_SPINOR_SPATIAL_LINKAGE_JOINT/1
Jacobi Spinor Joint Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_jacobi_spinor_spatial_linkage_joints
    j = model.lagmul_jacobi_spinor_spatial_linkage_joints[1]
    assert isinstance(j, LagmulJacobiSpinorSpatialLinkageJoint)
    assert j.node1 == 120
    assert j.node2 == 121
    assert j.node3 == 122
    assert pytest.approx(j.stiff) == 3.0e6
    assert j.skew_id == 10
    assert pytest.approx(j.tol) == 1.5e-5
    assert pytest.approx(j.link_len_a) == 19.5
    assert pytest.approx(j.link_len_b) == 26.0
    assert pytest.approx(j.twist_angle_alpha) == 65.0
    assert pytest.approx(j.offset_distance_s) == 12.5

    # Offset aliases
    assert pytest.approx(j.offset_distance_r) == 12.5
    assert pytest.approx(j.offset_distance_v) == 12.5
    assert pytest.approx(j.offset_distance_h) == 12.5
    assert pytest.approx(j.offset_distance_u) == 12.5
    assert pytest.approx(j.offset_distance_f) == 12.5


def test_m463_jacobi_spinor_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M463 Jacobi Joint Free
/LAGMUL/JACOBI_SPINOR_SPATIAL_LINKAGE_JOINT/2
220, 221, 222, 4.0e6, 11, 2.5e-5
22.0, 28.5, 80.0, 14.2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_jacobi_spinor_spatial_linkage_joints
    j = model.lagmul_jacobi_spinor_spatial_linkage_joints[2]
    assert j.node1 == 220
    assert j.node2 == 221
    assert j.node3 == 222
    assert pytest.approx(j.stiff) == 4.0e6
    assert j.skew_id == 11
    assert pytest.approx(j.tol) == 2.5e-5
    assert pytest.approx(j.link_len_a) == 22.0
    assert pytest.approx(j.link_len_b) == 28.5
    assert pytest.approx(j.twist_angle_alpha) == 80.0
    assert pytest.approx(j.offset_distance_s) == 14.2


def test_m463_jacobi_spinor_joint_aliases(tmp_path: Path):
    aliases = [
        "/JACOBI_SPINOR_SPATIAL_LINKAGE",
        "/LAGMUL/JACOBI_SPINOR_SPATIAL_LINKAGE",
        "/JACOBI_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM",
        "/JACOBI_SPINOR_SPATIAL_SYMMETRIC_MECHANISM",
        "/JACOBI_SPINOR_SPATIAL_6R_MECHANISM",
        "/JACOBI_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM",
        "/LAGMUL/JACOBI_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/JACOBI_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/JACOBI_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/JACOBI_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/JACOBI_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/JACOBI_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/NAMBU_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/NAMBU_SPINOR_SPATIAL_LINKAGE_JOINT",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test Jacobi Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "320, 321, 322, 1.0e6, 0, 1e-6",
            "16.0, 16.0, 0.0, 0.0",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.lagmul_jacobi_spinor_spatial_linkage_joints
        j = model.lagmul_jacobi_spinor_spatial_linkage_joints[idx]
        assert j.node1 == 320


# ============================================================================
# 4. SENSOR_SPRING_BENDING_LOCK_RATE Tests
# ============================================================================

def test_m463_sensor_spring_bending_lock_rate_fixed(tmp_path: Path):
    c1 = f"{14:>10d}{1.74e5:>20.4f}{0.0026:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Lock Rate Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_LOCK_RATE/1
Bending Lock Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_lock_rates
    s = model.sensor_spring_bending_lock_rates[1]
    assert isinstance(s, SensorSpringBendingLockRate)
    assert s.spring_id == 14
    assert pytest.approx(s.jbend_lock_max) == 1.74e5
    assert pytest.approx(s.t_delay) == 0.0026

    # Check property aliases
    assert pytest.approx(s.j_lock_bend_max) == 1.74e5
    assert pytest.approx(s.j_bend_lock_max) == 1.74e5
    assert pytest.approx(s.jbend_snap_max) == 1.74e5
    assert pytest.approx(s.jbend_rate_max) == 1.74e5
    assert pytest.approx(s.jbend_roc_rate_max) == 1.74e5
    assert pytest.approx(s.jbend_drop_rate_max) == 1.74e5
    assert pytest.approx(s.jbend_crk_rate_max) == 1.74e5
    assert pytest.approx(s.jbend_pop_rate_max) == 1.74e5
    assert pytest.approx(s.jbend_lock_rate_max) == 1.74e5
    assert pytest.approx(s.jbending_lock_max) == 1.74e5
    assert pytest.approx(s.j_bending_lock_max) == 1.74e5
    assert pytest.approx(s.j_lock_bending_max) == 1.74e5
    assert pytest.approx(s.jbending_angular_lock_max) == 1.74e5
    assert pytest.approx(s.j_bending_angular_lock_max) == 1.74e5
    assert pytest.approx(s.j_lock_bending_angular_max) == 1.74e5


def test_m463_sensor_spring_bending_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Bending Lock Rate Free Format Test
/SENSOR/SPRING_BENDING_LOCK_RATE/2
24, 2.74e5, 0.0036
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_bending_lock_rates
    s = model.sensor_spring_bending_lock_rates[2]
    assert s.spring_id == 24
    assert pytest.approx(s.jbend_lock_max) == 2.74e5
    assert pytest.approx(s.t_delay) == 0.0036


def test_m463_sensor_spring_bending_lock_rate_aliases(tmp_path: Path):
    aliases = [
        "/SENSOR/SPRING_BEND_LOCK_RATE",
        "/SENSOR/SPRING_BENDING_LOCK",
        "/SENSOR/SPRING_BEND_LOCK",
        "/SENSOR/SPRING_ANGULAR_BENDING_LOCK_RATE",
        "/SENSOR/SPRING_ANGULAR_BEND_LOCK_RATE",
        "/SENSOR/SPRING_POP_RATE_BENDING_LOCK",
        "/SENSOR/SPRING_LOCK_RATE_BENDING",
        "/SENSOR/SPRING_LOCK_RATE_BEND",
        "/SENSOR/SPRING_LOCK_RATE_ANGULAR_BEND",
        "/SENSOR/BEND_LOCK_RATE_SPRING",
        "/SENSOR/ANGULAR_BEND_LOCK_RATE_SPRING",
        "/SENSOR/SPRING_LOCK_BENDING",
        "/SENSOR/SPRING_LOCK_ANGULAR_BEND",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Sensor Spring Bending Lock Rate Aliases Test"]
    for sid, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{sid}",
            "37, 3.6e5, 0.0014",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(10, 10 + len(aliases)):
        assert sid in model.sensor_spring_bending_lock_rates
        s = model.sensor_spring_bending_lock_rates[sid]
        assert s.spring_id == 37
        assert pytest.approx(s.jbend_lock_max) == 3.6e5


# ============================================================================
# 5. Error handling tests
# ============================================================================

def test_m463_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_COUPLED_INTERLAMINAR_SHEAR_FAILURE_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBIMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/JACOBI_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_BENDING_LOCK_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4

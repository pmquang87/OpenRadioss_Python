"""Tests for Milestone M462:
- /FAIL/LAD_TRANSVERSE_INTERLAMINAR_SHEAR_FAILURE_RATE (and aliases)
- /ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY (and aliases)
- /LAGMUL/DIRAC_SPINOR_SPATIAL_LINKAGE_JOINT / /DIRAC_SPINOR_SPATIAL_LINKAGE_JOINT (and aliases)
- /SENSOR/SPRING_TRANSVERSE_LOCK_RATE (and aliases)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailLadTransverseInterlaminarShearFailureRate,
    EngElectrothermoflexomagnetochiralmeronplasmonicpolaritonicResonanceEnergy,
    LagmulDiracSpinorSpatialLinkageJoint,
    SensorSpringTransverseLockRate,
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
# 1. FAIL_LAD_TRANSVERSE_INTERLAMINAR_SHEAR_FAILURE_RATE Tests
# ============================================================================

def test_m462_fail_lad_transverse_interlaminar_shear_failure_rate_fixed(tmp_path: Path):
    c1 = f"{65.0:>20.4f}{240.0:>20.4f}{0.12:>20.4f}{1.28:>20.4f}{0.89:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M462 Fail Lad Transverse Interlaminar Shear Failure Rate Fixed
2022 0
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_SHEAR_FAILURE_RATE/1
Transverse Interlaminar Shear Failure Rate Card Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.fail_ladtransverseinterlaminarshearfailurerates
    fm = model.fail_ladtransverseinterlaminarshearfailurerates[1]
    assert isinstance(fm, FailLadTransverseInterlaminarShearFailureRate)
    assert fm.mat_id == 1
    assert pytest.approx(fm.sigma_tisfr0) == 65.0
    assert pytest.approx(fm.sigma_tisfrc) == 240.0
    assert pytest.approx(fm.gamma_tisfr) == 0.12
    assert pytest.approx(fm.p_tisfr) == 1.28
    assert pytest.approx(fm.d_tisfr_max) == 0.89
    assert fm.ifail_sh == 2
    assert fm.ifail_so == 1

    # Properties
    assert pytest.approx(fm.sigma_tisf0) == 65.0
    assert pytest.approx(fm.sigma_tisfc) == 240.0
    assert pytest.approx(fm.gamma_tisf) == 0.12
    assert pytest.approx(fm.p_tisf) == 1.28
    assert pytest.approx(fm.d_tisf_max) == 0.89

    assert pytest.approx(fm.sigma_tis0) == 65.0
    assert pytest.approx(fm.sigma_tisc) == 240.0
    assert pytest.approx(fm.gamma_tis) == 0.12
    assert pytest.approx(fm.p_tis) == 1.28
    assert pytest.approx(fm.d_tis_max) == 0.89

    assert pytest.approx(fm.sigma_tir0) == 65.0
    assert pytest.approx(fm.sigma_tirc) == 240.0
    assert pytest.approx(fm.gamma_tir) == 0.12
    assert pytest.approx(fm.p_tir) == 1.28
    assert pytest.approx(fm.d_tir_max) == 0.89


def test_m462_fail_lad_transverse_interlaminar_shear_failure_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M462 Fail Lad Transverse Interlaminar Shear Failure Rate Free
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_SHEAR_FAILURE_RATE/2
78.0, 275.0, 0.16, 1.38, 0.92
1, 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.fail_ladtransverseinterlaminarshearfailurerates
    fm = model.fail_ladtransverseinterlaminarshearfailurerates[2]
    assert pytest.approx(fm.sigma_tisfr0) == 78.0
    assert pytest.approx(fm.sigma_tisfrc) == 275.0
    assert pytest.approx(fm.gamma_tisfr) == 0.16
    assert pytest.approx(fm.p_tisfr) == 1.38
    assert pytest.approx(fm.d_tisfr_max) == 0.92
    assert fm.ifail_sh == 1
    assert fm.ifail_so == 2


def test_m462_fail_lad_transverse_interlaminar_shear_failure_rate_aliases(tmp_path: Path):
    aliases = [
        "/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_SHEAR_FAILURE_RATE",
        "/FAIL/LAD_TRANSVERSE_INTERLAMINAR_SHEAR_FAILURE",
        "/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_SHEAR_FAILURE",
        "/FAIL/LAD_TRANSVERSE_INTERLAMINAR_FAILURE",
        "/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_FAILURE",
        "/FAIL/LAD_TRANSVERSE_INTERLAMINAR_FAILURE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_FAILURE_RATE",
        "/FAIL/LAD_TRANSVERSE_INTERLAMINAR_DELAMINATION_FAILURE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_DELAMINATION_FAILURE_RATE",
        "/FAIL/LAD_TRANSVERSE_INTERLAMINAR_SHEAR_DELAMINATION_FAILURE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_SHEAR_DELAMINATION_FAILURE_RATE",
        "/FAIL/LAD_TISFR",
        "/FAIL/LAD_TISFR_MODEL",
        "/FAIL/LAD_TISFR_LAW",
        "/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_INTERLAMINAR_SHEAR_FAILURE",
        "/FAIL/LAD_TRANSVERSE_INTERLAMINAR_SHEAR_DAMAGE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_SHEAR_DAMAGE_RATE",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test M462 Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "60.0, 230.0, 0.10, 1.10, 0.95",
            "1, 1",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.fail_ladtransverseinterlaminarshearfailurerates
        fm = model.fail_ladtransverseinterlaminarshearfailurerates[idx]
        assert pytest.approx(fm.sigma_tisfr0) == 60.0


# ============================================================================
# 2. ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY Tests
# ============================================================================

def test_m462_eng_energy_fixed(tmp_path: Path):
    c1 = f"{1.9e-4:>20.6f}{16:>10d}"
    deck = f"""# RADIOSS ENGINE DECK
/BEGIN
Test M462 Eng Output Fixed
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Meron Resonance Energy Directive Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralmeronplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralmeronplasmonicpolaritonic_resonance_energies[1]
    assert isinstance(eng, EngElectrothermoflexomagnetochiralmeronplasmonicpolaritonicResonanceEnergy)
    assert pytest.approx(eng.dt_etfcmeronplp) == 1.9e-4
    assert eng.sens_id == 16


def test_m462_eng_energy_free(tmp_path: Path):
    deck = """# RADIOSS ENGINE DECK
/BEGIN
Test M462 Eng Output Free
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
2.9e-4, 19
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralmeronplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralmeronplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfcmeronplp) == 2.9e-4
    assert eng.sens_id == 19


def test_m462_eng_energy_aliases(tmp_path: Path):
    aliases = [
        "/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_MERON_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALMERONPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMERONPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALMERONPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOMERONCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_MERON_CHIRAL_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOMERONCHIRALPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOMERONCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOMERONCHIRALPLASMONICPOLARITONIC_RESONANCE",
    ]
    deck_lines = ["# RADIOSS ENGINE DECK", "/BEGIN", "Test Eng Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "3.9e-4, 26",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.eng_electrothermoflexomagnetochiralmeronplasmonicpolaritonic_resonance_energies
        eng = model.eng_electrothermoflexomagnetochiralmeronplasmonicpolaritonic_resonance_energies[idx]
        assert pytest.approx(eng.dt_etfcmeronplp) == 3.9e-4


# ============================================================================
# 3. LAGMUL_DIRAC_SPINOR_SPATIAL_LINKAGE_JOINT Tests
# ============================================================================

def test_m462_dirac_spinor_joint_fixed(tmp_path: Path):
    c1 = f"{117:>10d}{118:>10d}{119:>10d}{2.9e6:>20.4f}{9:>10d}{1.4e-5:>20.6f}"
    c2 = f"{18.5:>20.4f}{25.0:>20.4f}{60.0:>20.4f}{11.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M462 Dirac Joint Fixed
2022 0
/DIRAC_SPINOR_SPATIAL_LINKAGE_JOINT/1
Dirac Spinor Joint Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_dirac_spinor_spatial_linkage_joints
    j = model.lagmul_dirac_spinor_spatial_linkage_joints[1]
    assert isinstance(j, LagmulDiracSpinorSpatialLinkageJoint)
    assert j.node1 == 117
    assert j.node2 == 118
    assert j.node3 == 119
    assert pytest.approx(j.stiff) == 2.9e6
    assert j.skew_id == 9
    assert pytest.approx(j.tol) == 1.4e-5
    assert pytest.approx(j.link_len_a) == 18.5
    assert pytest.approx(j.link_len_b) == 25.0
    assert pytest.approx(j.twist_angle_alpha) == 60.0
    assert pytest.approx(j.offset_distance_s) == 11.5

    # Offset aliases
    assert pytest.approx(j.offset_distance_r) == 11.5
    assert pytest.approx(j.offset_distance_v) == 11.5
    assert pytest.approx(j.offset_distance_h) == 11.5
    assert pytest.approx(j.offset_distance_u) == 11.5
    assert pytest.approx(j.offset_distance_f) == 11.5


def test_m462_dirac_spinor_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M462 Dirac Joint Free
/LAGMUL/DIRAC_SPINOR_SPATIAL_LINKAGE_JOINT/2
217, 218, 219, 3.9e6, 10, 2.4e-5
21.0, 27.5, 75.0, 13.2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_dirac_spinor_spatial_linkage_joints
    j = model.lagmul_dirac_spinor_spatial_linkage_joints[2]
    assert j.node1 == 217
    assert j.node2 == 218
    assert j.node3 == 219
    assert pytest.approx(j.stiff) == 3.9e6
    assert j.skew_id == 10
    assert pytest.approx(j.tol) == 2.4e-5
    assert pytest.approx(j.link_len_a) == 21.0
    assert pytest.approx(j.link_len_b) == 27.5
    assert pytest.approx(j.twist_angle_alpha) == 75.0
    assert pytest.approx(j.offset_distance_s) == 13.2


def test_m462_dirac_spinor_joint_aliases(tmp_path: Path):
    aliases = [
        "/DIRAC_SPINOR_SPATIAL_LINKAGE",
        "/LAGMUL/DIRAC_SPINOR_SPATIAL_LINKAGE",
        "/DIRAC_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM",
        "/DIRAC_SPINOR_SPATIAL_SYMMETRIC_MECHANISM",
        "/DIRAC_SPINOR_SPATIAL_6R_MECHANISM",
        "/DIRAC_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM",
        "/LAGMUL/DIRAC_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/DIRAC_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/DIRAC_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/DIRAC_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/DIRAC_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/DIRAC_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/COURANT_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/COURANT_SPINOR_SPATIAL_LINKAGE_JOINT",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test Dirac Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "317, 318, 319, 1.0e6, 0, 1e-6",
            "15.0, 15.0, 0.0, 0.0",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.lagmul_dirac_spinor_spatial_linkage_joints
        j = model.lagmul_dirac_spinor_spatial_linkage_joints[idx]
        assert j.node1 == 317


# ============================================================================
# 4. SENSOR_SPRING_TRANSVERSE_LOCK_RATE Tests
# ============================================================================

def test_m462_sensor_spring_transverse_lock_rate_fixed(tmp_path: Path):
    c1 = f"{13:>10d}{1.84e5:>20.4f}{0.0025:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Lock Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_LOCK_RATE/1
Transverse Lock Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_lock_rates
    s = model.sensor_spring_transverse_lock_rates[1]
    assert isinstance(s, SensorSpringTransverseLockRate)
    assert s.spring_id == 13
    assert pytest.approx(s.jtrans_lock_max) == 1.84e5
    assert pytest.approx(s.t_delay) == 0.0025

    # Check property aliases
    assert pytest.approx(s.j_lock_trans_max) == 1.84e5
    assert pytest.approx(s.j_trans_lock_max) == 1.84e5
    assert pytest.approx(s.jtrans_snap_max) == 1.84e5
    assert pytest.approx(s.jtrans_rate_max) == 1.84e5
    assert pytest.approx(s.jtrans_roc_rate_max) == 1.84e5
    assert pytest.approx(s.jtrans_drop_rate_max) == 1.84e5
    assert pytest.approx(s.jtrans_crk_rate_max) == 1.84e5
    assert pytest.approx(s.jtrans_pop_rate_max) == 1.84e5
    assert pytest.approx(s.jtrans_lock_rate_max) == 1.84e5
    assert pytest.approx(s.jshear_lock_max) == 1.84e5
    assert pytest.approx(s.j_shear_lock_max) == 1.84e5
    assert pytest.approx(s.j_lock_shear_max) == 1.84e5
    assert pytest.approx(s.jtransverse_lock_max) == 1.84e5
    assert pytest.approx(s.j_transverse_lock_max) == 1.84e5
    assert pytest.approx(s.j_lock_transverse_max) == 1.84e5


def test_m462_sensor_spring_transverse_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Transverse Lock Rate Free Format Test
/SENSOR/SPRING_TRANSVERSE_LOCK_RATE/2
23, 2.84e5, 0.0035
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_transverse_lock_rates
    s = model.sensor_spring_transverse_lock_rates[2]
    assert s.spring_id == 23
    assert pytest.approx(s.jtrans_lock_max) == 2.84e5
    assert pytest.approx(s.t_delay) == 0.0035


def test_m462_sensor_spring_transverse_lock_rate_aliases(tmp_path: Path):
    aliases = [
        "/SENSOR/SPRING_TRANS_LOCK_RATE",
        "/SENSOR/SPRING_TRANSVERSE_LOCK",
        "/SENSOR/SPRING_TRANS_LOCK",
        "/SENSOR/SPRING_SHEAR_LOCK_RATE",
        "/SENSOR/SPRING_SHEAR_LOCK",
        "/SENSOR/SPRING_POP_RATE_TRANSVERSE_LOCK",
        "/SENSOR/SPRING_LOCK_RATE_TRANSVERSE",
        "/SENSOR/SPRING_LOCK_RATE_TRANS",
        "/SENSOR/SPRING_LOCK_RATE_SHEAR",
        "/SENSOR/TRANS_LOCK_RATE_SPRING",
        "/SENSOR/SHEAR_LOCK_RATE_SPRING",
        "/SENSOR/SPRING_LOCK_TRANSVERSE",
        "/SENSOR/SPRING_LOCK_SHEAR",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Sensor Spring Transverse Lock Rate Aliases Test"]
    for sid, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{sid}",
            "36, 3.5e5, 0.0013",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(10, 10 + len(aliases)):
        assert sid in model.sensor_spring_transverse_lock_rates
        s = model.sensor_spring_transverse_lock_rates[sid]
        assert s.spring_id == 36
        assert pytest.approx(s.jtrans_lock_max) == 3.5e5


# ============================================================================
# 5. Error handling tests
# ============================================================================

def test_m462_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_SHEAR_FAILURE_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/DIRAC_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TRANSVERSE_LOCK_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4

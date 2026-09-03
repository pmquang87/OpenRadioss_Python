"""Tests for Milestone M469:
- /FAIL/LAD_COUPLED_MATRIX_MICROCRACKING_FAILURE_RATE (and aliases)
- /ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPHALERONPLASMONICPOLARITONIC_RESONANCE_ENERGY (and aliases)
- /LAGMUL/BEREZIN_SPINOR_SPATIAL_LINKAGE_JOINT / /BEREZIN_SPINOR_SPATIAL_LINKAGE_JOINT (and aliases)
- /SENSOR/SPRING_BENDING_SNAP_RATE (and aliases)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailLadCoupledMatrixMicrocrackingFailureRate,
    EngElectrothermoflexomagnetochiralsphaleronplasmonicpolaritonicResonanceEnergy,
    LagmulBerezinSpinorSpatialLinkageJoint,
    SensorSpringBendingSnapRate,
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
# 1. FAIL_LAD_COUPLED_MATRIX_MICROCRACKING_FAILURE_RATE Tests
# ============================================================================

def test_m469_fail_lad_coupled_matrix_microcracking_failure_rate_fixed(tmp_path: Path):
    c1 = f"{92.0:>20.4f}{330.0:>20.4f}{0.21:>20.4f}{1.52:>20.4f}{0.97:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M469 Fail Lad Coupled Matrix Microcracking Failure Rate Fixed
2022 0
/FAIL/LAD_COUPLED_MATRIX_MICROCRACKING_FAILURE_RATE/1
Coupled Matrix Microcracking Failure Rate Card Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.fail_ladcoupledmatrixmicrocrackingfailurerates
    fm = model.fail_ladcoupledmatrixmicrocrackingfailurerates[1]
    assert isinstance(fm, FailLadCoupledMatrixMicrocrackingFailureRate)
    assert fm.mat_id == 1
    assert pytest.approx(fm.sigma_cmmcfr0) == 92.0
    assert pytest.approx(fm.sigma_cmmcfrc) == 330.0
    assert pytest.approx(fm.gamma_cmmcfr) == 0.21
    assert pytest.approx(fm.p_cmmcfr) == 1.52
    assert pytest.approx(fm.d_cmmcfr_max) == 0.97
    assert fm.ifail_sh == 2
    assert fm.ifail_so == 1

    # Properties
    assert pytest.approx(fm.sigma_cmmcf0) == 92.0
    assert pytest.approx(fm.sigma_cmmcfc) == 330.0
    assert pytest.approx(fm.gamma_cmmcf) == 0.21
    assert pytest.approx(fm.p_cmmcf) == 1.52
    assert pytest.approx(fm.d_cmmcf_max) == 0.97

    assert pytest.approx(fm.sigma_cmmc0) == 92.0
    assert pytest.approx(fm.sigma_cmmcc) == 330.0
    assert pytest.approx(fm.gamma_cmmc) == 0.21
    assert pytest.approx(fm.p_cmmc) == 1.52
    assert pytest.approx(fm.d_cmmc_max) == 0.97

    assert pytest.approx(fm.sigma_cmc0) == 92.0
    assert pytest.approx(fm.sigma_cmcc) == 330.0
    assert pytest.approx(fm.gamma_cmc) == 0.21
    assert pytest.approx(fm.p_cmc) == 1.52
    assert pytest.approx(fm.d_cmc_max) == 0.97


def test_m469_fail_lad_coupled_matrix_microcracking_failure_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M469 Fail Lad Coupled Matrix Microcracking Failure Rate Free
/FAIL/LAD_COUPLED_MATRIX_MICROCRACKING_FAILURE_RATE/2
105.0, 350.0, 0.24, 1.62, 0.98
1, 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.fail_ladcoupledmatrixmicrocrackingfailurerates
    fm = model.fail_ladcoupledmatrixmicrocrackingfailurerates[2]
    assert pytest.approx(fm.sigma_cmmcfr0) == 105.0
    assert pytest.approx(fm.sigma_cmmcfrc) == 350.0
    assert pytest.approx(fm.gamma_cmmcfr) == 0.24
    assert pytest.approx(fm.p_cmmcfr) == 1.62
    assert pytest.approx(fm.d_cmmcfr_max) == 0.98
    assert fm.ifail_sh == 1
    assert fm.ifail_so == 2


def test_m469_fail_lad_coupled_matrix_microcracking_failure_rate_aliases(tmp_path: Path):
    aliases = [
        "/FAIL/LADEVEZE_COUPLED_MATRIX_MICROCRACKING_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_MATRIX_MICROCRACKING_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_MICROCRACKING_FAILURE",
        "/FAIL/LAD_COUPLE_MATRIX_MICROCRACKING_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLE_MATRIX_MICROCRACKING_FAILURE_RATE",
        "/FAIL/LAD_COUPLE_MATRIX_MICROCRACKING_FAILURE",
        "/FAIL/LADEVEZE_COUPLE_MATRIX_MICROCRACKING_FAILURE",
        "/FAIL/LAD_COUPLED_MATRIX_CRACKING_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_CRACKING_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_MATRIX_CRACKING_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_CRACKING_FAILURE",
        "/FAIL/LAD_COUPLE_MATRIX_CRACKING_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLE_MATRIX_CRACKING_FAILURE_RATE",
        "/FAIL/LAD_COUPLE_MATRIX_CRACKING_FAILURE",
        "/FAIL/LADEVEZE_COUPLE_MATRIX_CRACKING_FAILURE",
        "/FAIL/LAD_COUPLED_MATRIX_MICRO_CRACKING_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_MICRO_CRACKING_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_MATRIX_MICRO_CRACKING_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_MICRO_CRACKING_FAILURE",
        "/FAIL/LAD_CMMCFR",
        "/FAIL/LAD_CMMCFR_MODEL",
        "/FAIL/LAD_CMMCFR_LAW",
        "/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_MATRIX_MICROCRACKING_FAILURE",
        "/FAIL/LAD_COUPLED_MATRIX_MICROCRACKING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_MICROCRACKING_DAMAGE_RATE",
        "/FAIL/LAD_COUPLED_MATRIX_CRACKING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_CRACKING_DAMAGE_RATE",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test M469 Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "85.0, 310.0, 0.18, 1.40, 0.96",
            "1, 1",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.fail_ladcoupledmatrixmicrocrackingfailurerates
        fm = model.fail_ladcoupledmatrixmicrocrackingfailurerates[idx]
        assert pytest.approx(fm.sigma_cmmcfr0) == 85.0


# ============================================================================
# 2. ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALSPHALERONPLASMONICPOLARITONIC_RESONANCE_ENERGY Tests
# ============================================================================

def test_m469_eng_energy_fixed(tmp_path: Path):
    c1 = f"{3.1e-4:>20.6f}{24:>10d}"
    deck = f"""# RADIOSS ENGINE DECK
/BEGIN
Test M469 Eng Output Fixed
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPHALERONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Sphaleron Resonance Energy Directive Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralsphaleronplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralsphaleronplasmonicpolaritonic_resonance_energies[1]
    assert isinstance(eng, EngElectrothermoflexomagnetochiralsphaleronplasmonicpolaritonicResonanceEnergy)
    assert pytest.approx(eng.dt_etfcsphaleronplp) == 3.1e-4
    assert eng.sens_id == 24


def test_m469_eng_energy_free(tmp_path: Path):
    deck = """# RADIOSS ENGINE DECK
/BEGIN
Test M469 Eng Output Free
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPHALERONPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
4.1e-4, 28
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralsphaleronplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralsphaleronplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfcsphaleronplp) == 4.1e-4
    assert eng.sens_id == 28


def test_m469_eng_energy_aliases(tmp_path: Path):
    aliases = [
        "/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_SPHALERON_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALSPHALERONPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPHALERONPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALSPHALERONPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOSPHALERONCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_SPHALERON_CHIRAL_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOSPHALERONCHIRALPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOSPHALERONCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOSPHALERONCHIRALPLASMONICPOLARITONIC_RESONANCE",
    ]
    deck_lines = ["# RADIOSS ENGINE DECK", "/BEGIN", "Test Eng Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "5.1e-4, 35",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.eng_electrothermoflexomagnetochiralsphaleronplasmonicpolaritonic_resonance_energies
        eng = model.eng_electrothermoflexomagnetochiralsphaleronplasmonicpolaritonic_resonance_energies[idx]
        assert pytest.approx(eng.dt_etfcsphaleronplp) == 5.1e-4


# ============================================================================
# 3. LAGMUL_BEREZIN_SPINOR_SPATIAL_LINKAGE_JOINT Tests
# ============================================================================

def test_m469_berezin_spinor_joint_fixed(tmp_path: Path):
    c1 = f"{180:>10d}{181:>10d}{182:>10d}{4.3e6:>20.4f}{21:>10d}{2.2e-5:>20.6f}"
    c2 = f"{25.5:>20.4f}{32.0:>20.4f}{88.0:>20.4f}{18.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M469 Berezin Joint Fixed
2022 0
/BEREZIN_SPINOR_SPATIAL_LINKAGE_JOINT/1
Berezin Spinor Joint Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_berezin_spinor_spatial_linkage_joints
    j = model.lagmul_berezin_spinor_spatial_linkage_joints[1]
    assert isinstance(j, LagmulBerezinSpinorSpatialLinkageJoint)
    assert j.node1 == 180
    assert j.node2 == 181
    assert j.node3 == 182
    assert pytest.approx(j.stiff) == 4.3e6
    assert j.skew_id == 21
    assert pytest.approx(j.tol) == 2.2e-5
    assert pytest.approx(j.link_len_a) == 25.5
    assert pytest.approx(j.link_len_b) == 32.0
    assert pytest.approx(j.twist_angle_alpha) == 88.0
    assert pytest.approx(j.offset_distance_s) == 18.0

    # Offset aliases
    assert pytest.approx(j.offset_distance_r) == 18.0
    assert pytest.approx(j.offset_distance_v) == 18.0
    assert pytest.approx(j.offset_distance_h) == 18.0
    assert pytest.approx(j.offset_distance_u) == 18.0
    assert pytest.approx(j.offset_distance_f) == 18.0


def test_m469_berezin_spinor_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M469 Berezin Joint Free
/LAGMUL/BEREZIN_SPINOR_SPATIAL_LINKAGE_JOINT/2
280, 281, 282, 5.3e6, 22, 3.2e-5
28.0, 34.5, 101.0, 20.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_berezin_spinor_spatial_linkage_joints
    j = model.lagmul_berezin_spinor_spatial_linkage_joints[2]
    assert j.node1 == 280
    assert j.node2 == 281
    assert j.node3 == 282
    assert pytest.approx(j.stiff) == 5.3e6
    assert j.skew_id == 22
    assert pytest.approx(j.tol) == 3.2e-5
    assert pytest.approx(j.link_len_a) == 28.0
    assert pytest.approx(j.link_len_b) == 34.5
    assert pytest.approx(j.twist_angle_alpha) == 101.0
    assert pytest.approx(j.offset_distance_s) == 20.0


def test_m469_berezin_spinor_joint_aliases(tmp_path: Path):
    aliases = [
        "/BEREZIN_SPINOR_SPATIAL_LINKAGE",
        "/LAGMUL/BEREZIN_SPINOR_SPATIAL_LINKAGE",
        "/BEREZIN_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM",
        "/BEREZIN_SPINOR_SPATIAL_SYMMETRIC_MECHANISM",
        "/BEREZIN_SPINOR_SPATIAL_6R_MECHANISM",
        "/BEREZIN_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM",
        "/LAGMUL/BEREZIN_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/BEREZIN_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/BEREZIN_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/BEREZIN_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/BEREZIN_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/BEREZIN_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/FROBENIUS_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/FROBENIUS_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/FROBENIUS_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/FROBENIUS_TWISTOR_SPATIAL_LINKAGE_JOINT",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test Berezin Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "380, 381, 382, 1.8e6, 0, 1e-6",
            "22.0, 22.0, 0.0, 0.0",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.lagmul_berezin_spinor_spatial_linkage_joints
        j = model.lagmul_berezin_spinor_spatial_linkage_joints[idx]
        assert j.node1 == 380


# ============================================================================
# 4. SENSOR_SPRING_BENDING_SNAP_RATE Tests
# ============================================================================

def test_m469_sensor_spring_bending_snap_rate_fixed(tmp_path: Path):
    c1 = f"{21:>10d}{2.45e5:>20.4f}{0.0038:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_SNAP_RATE/1
Bending Snap Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_snap_rates
    s = model.sensor_spring_bending_snap_rates[1]
    assert isinstance(s, SensorSpringBendingSnapRate)
    assert s.spring_id == 21
    assert pytest.approx(s.jbnd_snp_max) == 2.45e5
    assert pytest.approx(s.t_delay) == 0.0038

    # Check property aliases
    assert pytest.approx(s.jbnd_snap_max) == 2.45e5
    assert pytest.approx(s.jbend_snap_max) == 2.45e5
    assert pytest.approx(s.jbending_snap_max) == 2.45e5
    assert pytest.approx(s.j_bnd_snp_max) == 2.45e5
    assert pytest.approx(s.j_bend_snp_max) == 2.45e5
    assert pytest.approx(s.j_bending_snp_max) == 2.45e5
    assert pytest.approx(s.j_snap_bend_max) == 2.45e5
    assert pytest.approx(s.j_snap_bending_max) == 2.45e5
    assert pytest.approx(s.j_bending_snap_max) == 2.45e5
    assert pytest.approx(s.jbnd_rate_max) == 2.45e5
    assert pytest.approx(s.jbnd_roc_rate_max) == 2.45e5
    assert pytest.approx(s.jbnd_drop_rate_max) == 2.45e5
    assert pytest.approx(s.jbnd_crk_rate_max) == 2.45e5
    assert pytest.approx(s.jbnd_pop_rate_max) == 2.45e5
    assert pytest.approx(s.jbnd_lock_rate_max) == 2.45e5
    assert pytest.approx(s.jbnd_snap_rate_max) == 2.45e5


def test_m469_sensor_spring_bending_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Bending Snap Rate Free Format Test
/SENSOR/SPRING_BENDING_SNAP_RATE/2
31, 3.45e5, 0.0048
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_bending_snap_rates
    s = model.sensor_spring_bending_snap_rates[2]
    assert s.spring_id == 31
    assert pytest.approx(s.jbnd_snp_max) == 3.45e5
    assert pytest.approx(s.t_delay) == 0.0048


def test_m469_sensor_spring_bending_snap_rate_aliases(tmp_path: Path):
    aliases = [
        "/SENSOR/SPRING_BEND_SNAP_RATE",
        "/SENSOR/SPRING_BENDING_SNAP",
        "/SENSOR/SPRING_BEND_SNAP",
        "/SENSOR/SPRING_ANGULAR_BENDING_SNAP_RATE",
        "/SENSOR/SPRING_ANGULAR_BEND_SNAP_RATE",
        "/SENSOR/SPRING_ANGULAR_BENDING_SNAP",
        "/SENSOR/SPRING_ANGULAR_BEND_SNAP",
        "/SENSOR/SPRING_FLEXURAL_SNAP_RATE",
        "/SENSOR/SPRING_FLEX_SNAP_RATE",
        "/SENSOR/SPRING_FLEXURAL_SNAP",
        "/SENSOR/SPRING_FLEX_SNAP",
        "/SENSOR/SPRING_POP_RATE_BENDING_SNAP",
        "/SENSOR/SPRING_LOCK_RATE_BENDING_SNAP",
        "/SENSOR/SPRING_SNAP_RATE_BENDING",
        "/SENSOR/SPRING_SNAP_RATE_BEND",
        "/SENSOR/BENDING_SNAP_RATE_SPRING",
        "/SENSOR/BEND_SNAP_RATE_SPRING",
        "/SENSOR/SPRING_SNAP_BENDING",
        "/SENSOR/SPRING_SNAP_BEND",
        "/SENSOR/SPRING_RATE_BENDING_SNAP",
        "/SENSOR/SPRING_RATE_BEND_SNAP",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Sensor Spring Bending Snap Rate Aliases Test"]
    for sid, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{sid}",
            "44, 4.5e5, 0.0024",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(10, 10 + len(aliases)):
        assert sid in model.sensor_spring_bending_snap_rates
        s = model.sensor_spring_bending_snap_rates[sid]
        assert s.spring_id == 44
        assert pytest.approx(s.jbnd_snp_max) == 4.5e5


# ============================================================================
# 5. Error handling tests
# ============================================================================

def test_m469_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_COUPLED_MATRIX_MICROCRACKING_FAILURE_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPHALERONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/BEREZIN_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_BENDING_SNAP_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4

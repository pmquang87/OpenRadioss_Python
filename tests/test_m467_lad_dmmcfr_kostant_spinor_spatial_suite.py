"""Tests for Milestone M467:
- /FAIL/LAD_DYNAMIC_MATRIX_MICROCRACKING_FAILURE_RATE (and aliases)
- /ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHOPFIONPLASMONICPOLARITONIC_RESONANCE_ENERGY (and aliases)
- /LAGMUL/KOSTANT_SPINOR_SPATIAL_LINKAGE_JOINT / /KOSTANT_SPINOR_SPATIAL_LINKAGE_JOINT (and aliases)
- /SENSOR/SPRING_NORMAL_SNAP_RATE (and aliases)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailLadDynamicMatrixMicrocrackingFailureRate,
    EngElectrothermoflexomagnetochiralhopfionplasmonicpolaritonicResonanceEnergy,
    LagmulKostantSpinorSpatialLinkageJoint,
    SensorSpringNormalSnapRate,
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
# 1. FAIL_LAD_DYNAMIC_MATRIX_MICROCRACKING_FAILURE_RATE Tests
# ============================================================================

def test_m467_fail_lad_dynamic_matrix_microcracking_failure_rate_fixed(tmp_path: Path):
    c1 = f"{85.0:>20.4f}{310.0:>20.4f}{0.18:>20.4f}{1.42:>20.4f}{0.95:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M467 Fail Lad Dynamic Matrix Microcracking Failure Rate Fixed
2022 0
/FAIL/LAD_DYNAMIC_MATRIX_MICROCRACKING_FAILURE_RATE/1
Dynamic Matrix Microcracking Failure Rate Card Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.fail_laddynamicmatrixmicrocrackingfailurerates
    fm = model.fail_laddynamicmatrixmicrocrackingfailurerates[1]
    assert isinstance(fm, FailLadDynamicMatrixMicrocrackingFailureRate)
    assert fm.mat_id == 1
    assert pytest.approx(fm.sigma_dmmcfr0) == 85.0
    assert pytest.approx(fm.sigma_dmmcfrc) == 310.0
    assert pytest.approx(fm.gamma_dmmcfr) == 0.18
    assert pytest.approx(fm.p_dmmcfr) == 1.42
    assert pytest.approx(fm.d_dmmcfr_max) == 0.95
    assert fm.ifail_sh == 2
    assert fm.ifail_so == 1

    # Properties
    assert pytest.approx(fm.sigma_dmmcf0) == 85.0
    assert pytest.approx(fm.sigma_dmmcfc) == 310.0
    assert pytest.approx(fm.gamma_dmmcf) == 0.18
    assert pytest.approx(fm.p_dmmcf) == 1.42
    assert pytest.approx(fm.d_dmmcf_max) == 0.95

    assert pytest.approx(fm.sigma_dmmc0) == 85.0
    assert pytest.approx(fm.sigma_dmmcc) == 310.0
    assert pytest.approx(fm.gamma_dmmc) == 0.18
    assert pytest.approx(fm.p_dmmc) == 1.42
    assert pytest.approx(fm.d_dmmc_max) == 0.95

    assert pytest.approx(fm.sigma_dmc0) == 85.0
    assert pytest.approx(fm.sigma_dmcc) == 310.0
    assert pytest.approx(fm.gamma_dmc) == 0.18
    assert pytest.approx(fm.p_dmc) == 1.42
    assert pytest.approx(fm.d_dmc_max) == 0.95


def test_m467_fail_lad_dynamic_matrix_microcracking_failure_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M467 Fail Lad Dynamic Matrix Microcracking Failure Rate Free
/FAIL/LAD_DYNAMIC_MATRIX_MICROCRACKING_FAILURE_RATE/2
95.0, 335.0, 0.20, 1.55, 0.98
1, 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.fail_laddynamicmatrixmicrocrackingfailurerates
    fm = model.fail_laddynamicmatrixmicrocrackingfailurerates[2]
    assert pytest.approx(fm.sigma_dmmcfr0) == 95.0
    assert pytest.approx(fm.sigma_dmmcfrc) == 335.0
    assert pytest.approx(fm.gamma_dmmcfr) == 0.20
    assert pytest.approx(fm.p_dmmcfr) == 1.55
    assert pytest.approx(fm.d_dmmcfr_max) == 0.98
    assert fm.ifail_sh == 1
    assert fm.ifail_so == 2


def test_m467_fail_lad_dynamic_matrix_microcracking_failure_rate_aliases(tmp_path: Path):
    aliases = [
        "/FAIL/LADEVEZE_DYNAMIC_MATRIX_MICROCRACKING_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_MATRIX_MICROCRACKING_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_MATRIX_MICROCRACKING_FAILURE",
        "/FAIL/LAD_DYNAMIC_MATRIX_CRACKING_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_MATRIX_CRACKING_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_MATRIX_CRACKING_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_MATRIX_CRACKING_FAILURE",
        "/FAIL/LAD_DYNAMIC_TRANSVERSE_MATRIX_MICROCRACKING_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_TRANSVERSE_MATRIX_MICROCRACKING_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_TRANSVERSE_MATRIX_MICROCRACKING_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_TRANSVERSE_MATRIX_MICROCRACKING_FAILURE",
        "/FAIL/LAD_DYNAMIC_TRANSVERSE_MATRIX_CRACKING_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_TRANSVERSE_MATRIX_CRACKING_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_MATRIX_MICRO_CRACKING_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_MATRIX_MICRO_CRACKING_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_MATRIX_MICRO_CRACKING_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_MATRIX_MICRO_CRACKING_FAILURE",
        "/FAIL/LAD_DMMCFR",
        "/FAIL/LAD_DMMCFR_MODEL",
        "/FAIL/LAD_DMMCFR_LAW",
        "/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_MATRIX_MICROCRACKING_FAILURE",
        "/FAIL/LAD_DYNAMIC_MATRIX_MICROCRACKING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_MATRIX_MICROCRACKING_DAMAGE_RATE",
        "/FAIL/LAD_DYNAMIC_MATRIX_CRACKING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_MATRIX_CRACKING_DAMAGE_RATE",
        "/FAIL/LAD_DYNAMIC_MATRIX_MICRO_CRACKING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_MATRIX_MICRO_CRACKING_DAMAGE_RATE",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test M467 Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "79.0, 290.0, 0.15, 1.30, 0.97",
            "1, 1",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.fail_laddynamicmatrixmicrocrackingfailurerates
        fm = model.fail_laddynamicmatrixmicrocrackingfailurerates[idx]
        assert pytest.approx(fm.sigma_dmmcfr0) == 79.0


# ============================================================================
# 2. ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALHOPFIONPLASMONICPOLARITONIC_RESONANCE_ENERGY Tests
# ============================================================================

def test_m467_eng_energy_fixed(tmp_path: Path):
    c1 = f"{2.8e-4:>20.6f}{21:>10d}"
    deck = f"""# RADIOSS ENGINE DECK
/BEGIN
Test M467 Eng Output Fixed
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHOPFIONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Hopfion Resonance Energy Directive Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralhopfionplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralhopfionplasmonicpolaritonic_resonance_energies[1]
    assert isinstance(eng, EngElectrothermoflexomagnetochiralhopfionplasmonicpolaritonicResonanceEnergy)
    assert pytest.approx(eng.dt_etfchopfionplp) == 2.8e-4
    assert eng.sens_id == 21


def test_m467_eng_energy_free(tmp_path: Path):
    deck = """# RADIOSS ENGINE DECK
/BEGIN
Test M467 Eng Output Free
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHOPFIONPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
3.8e-4, 24
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralhopfionplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralhopfionplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfchopfionplp) == 3.8e-4
    assert eng.sens_id == 24


def test_m467_eng_energy_aliases(tmp_path: Path):
    aliases = [
        "/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_HOPFION_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALHOPFIONPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHOPFIONPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALHOPFIONPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOHOPFIONCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_HOPFION_CHIRAL_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOHOPFIONCHIRALPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOHOPFIONCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOHOPFIONCHIRALPLASMONICPOLARITONIC_RESONANCE",
    ]
    deck_lines = ["# RADIOSS ENGINE DECK", "/BEGIN", "Test Eng Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "4.8e-4, 32",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.eng_electrothermoflexomagnetochiralhopfionplasmonicpolaritonic_resonance_energies
        eng = model.eng_electrothermoflexomagnetochiralhopfionplasmonicpolaritonic_resonance_energies[idx]
        assert pytest.approx(eng.dt_etfchopfionplp) == 4.8e-4


# ============================================================================
# 3. LAGMUL_KOSTANT_SPINOR_SPATIAL_LINKAGE_JOINT Tests
# ============================================================================

def test_m467_kostant_spinor_joint_fixed(tmp_path: Path):
    c1 = f"{160:>10d}{161:>10d}{162:>10d}{3.9e6:>20.4f}{18:>10d}{1.9e-5:>20.6f}"
    c2 = f"{23.5:>20.4f}{30.0:>20.4f}{85.0:>20.4f}{16.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M467 Kostant Joint Fixed
2022 0
/KOSTANT_SPINOR_SPATIAL_LINKAGE_JOINT/1
Kostant Spinor Joint Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_kostant_spinor_spatial_linkage_joints
    j = model.lagmul_kostant_spinor_spatial_linkage_joints[1]
    assert isinstance(j, LagmulKostantSpinorSpatialLinkageJoint)
    assert j.node1 == 160
    assert j.node2 == 161
    assert j.node3 == 162
    assert pytest.approx(j.stiff) == 3.9e6
    assert j.skew_id == 18
    assert pytest.approx(j.tol) == 1.9e-5
    assert pytest.approx(j.link_len_a) == 23.5
    assert pytest.approx(j.link_len_b) == 30.0
    assert pytest.approx(j.twist_angle_alpha) == 85.0
    assert pytest.approx(j.offset_distance_s) == 16.5

    # Offset aliases
    assert pytest.approx(j.offset_distance_r) == 16.5
    assert pytest.approx(j.offset_distance_v) == 16.5
    assert pytest.approx(j.offset_distance_h) == 16.5
    assert pytest.approx(j.offset_distance_u) == 16.5
    assert pytest.approx(j.offset_distance_f) == 16.5


def test_m467_kostant_spinor_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M467 Kostant Joint Free
/LAGMUL/KOSTANT_SPINOR_SPATIAL_LINKAGE_JOINT/2
260, 261, 262, 4.9e6, 19, 2.9e-5
26.0, 32.5, 98.0, 18.2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_kostant_spinor_spatial_linkage_joints
    j = model.lagmul_kostant_spinor_spatial_linkage_joints[2]
    assert j.node1 == 260
    assert j.node2 == 261
    assert j.node3 == 262
    assert pytest.approx(j.stiff) == 4.9e6
    assert j.skew_id == 19
    assert pytest.approx(j.tol) == 2.9e-5
    assert pytest.approx(j.link_len_a) == 26.0
    assert pytest.approx(j.link_len_b) == 32.5
    assert pytest.approx(j.twist_angle_alpha) == 98.0
    assert pytest.approx(j.offset_distance_s) == 18.2


def test_m467_kostant_spinor_joint_aliases(tmp_path: Path):
    aliases = [
        "/KOSTANT_SPINOR_SPATIAL_LINKAGE",
        "/LAGMUL/KOSTANT_SPINOR_SPATIAL_LINKAGE",
        "/KOSTANT_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM",
        "/KOSTANT_SPINOR_SPATIAL_SYMMETRIC_MECHANISM",
        "/KOSTANT_SPINOR_SPATIAL_6R_MECHANISM",
        "/KOSTANT_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM",
        "/LAGMUL/KOSTANT_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/KOSTANT_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/KOSTANT_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/KOSTANT_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/KOSTANT_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/KOSTANT_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/SEGAL_SHALE_WEIL_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/SEGAL_SHALE_WEIL_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/SEGAL_SHALE_WEIL_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/SEGAL_SHALE_WEIL_TWISTOR_SPATIAL_LINKAGE_JOINT",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test Kostant Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "360, 361, 362, 1.5e6, 0, 1e-6",
            "20.0, 20.0, 0.0, 0.0",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.lagmul_kostant_spinor_spatial_linkage_joints
        j = model.lagmul_kostant_spinor_spatial_linkage_joints[idx]
        assert j.node1 == 360


# ============================================================================
# 4. SENSOR_SPRING_NORMAL_SNAP_RATE Tests
# ============================================================================

def test_m467_sensor_spring_normal_snap_rate_fixed(tmp_path: Path):
    c1 = f"{18:>10d}{2.15e5:>20.4f}{0.0035:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_SNAP_RATE/1
Normal Snap Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_snap_rates
    s = model.sensor_spring_normal_snap_rates[1]
    assert isinstance(s, SensorSpringNormalSnapRate)
    assert s.spring_id == 18
    assert pytest.approx(s.jnorm_snp_max) == 2.15e5
    assert pytest.approx(s.t_delay) == 0.0035

    # Check property aliases
    assert pytest.approx(s.jnorm_snap_max) == 2.15e5
    assert pytest.approx(s.j_norm_snp_max) == 2.15e5
    assert pytest.approx(s.j_snap_norm_max) == 2.15e5
    assert pytest.approx(s.j_norm_snap_max) == 2.15e5
    assert pytest.approx(s.j_axial_snap_max) == 2.15e5
    assert pytest.approx(s.jnorm_rate_max) == 2.15e5
    assert pytest.approx(s.jnorm_roc_rate_max) == 2.15e5
    assert pytest.approx(s.jnorm_drop_rate_max) == 2.15e5
    assert pytest.approx(s.jnorm_crk_rate_max) == 2.15e5
    assert pytest.approx(s.jnorm_pop_rate_max) == 2.15e5
    assert pytest.approx(s.jnorm_lock_rate_max) == 2.15e5
    assert pytest.approx(s.jnorm_snap_rate_max) == 2.15e5
    assert pytest.approx(s.j_normal_snap_max) == 2.15e5


def test_m467_sensor_spring_normal_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Normal Snap Rate Free Format Test
/SENSOR/SPRING_NORMAL_SNAP_RATE/2
28, 3.15e5, 0.0045
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_normal_snap_rates
    s = model.sensor_spring_normal_snap_rates[2]
    assert s.spring_id == 28
    assert pytest.approx(s.jnorm_snp_max) == 3.15e5
    assert pytest.approx(s.t_delay) == 0.0045


def test_m467_sensor_spring_normal_snap_rate_aliases(tmp_path: Path):
    aliases = [
        "/SENSOR/SPRING_NORM_SNAP_RATE",
        "/SENSOR/SPRING_NORMAL_SNAP",
        "/SENSOR/SPRING_NORM_SNAP",
        "/SENSOR/SPRING_POP_RATE_NORMAL_SNAP",
        "/SENSOR/SPRING_LOCK_RATE_NORMAL_SNAP",
        "/SENSOR/SPRING_SNAP_RATE_NORMAL",
        "/SENSOR/SPRING_SNAP_RATE_NORM",
        "/SENSOR/NORM_SNAP_RATE_SPRING",
        "/SENSOR/SPRING_SNAP_NORMAL",
        "/SENSOR/SPRING_RATE_NORMAL_SNAP",
        "/SENSOR/SPRING_RATE_NORM_SNAP",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Sensor Spring Normal Snap Rate Aliases Test"]
    for sid, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{sid}",
            "41, 4.2e5, 0.0020",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(10, 10 + len(aliases)):
        assert sid in model.sensor_spring_normal_snap_rates
        s = model.sensor_spring_normal_snap_rates[sid]
        assert s.spring_id == 41
        assert pytest.approx(s.jnorm_snp_max) == 4.2e5


# ============================================================================
# 5. Error handling tests
# ============================================================================

def test_m467_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_DYNAMIC_MATRIX_MICROCRACKING_FAILURE_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHOPFIONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/KOSTANT_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_NORMAL_SNAP_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4

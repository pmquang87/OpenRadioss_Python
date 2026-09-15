"""Tests for Milestone M468:
- /FAIL/LAD_TRANSVERSE_MATRIX_MICROCRACKING_FAILURE_RATE (and aliases)
- /ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMONOPOLEPLASMONICPOLARITONIC_RESONANCE_ENERGY (and aliases)
- /LAGMUL/SOURIAU_SPINOR_SPATIAL_LINKAGE_JOINT / /SOURIAU_SPINOR_SPATIAL_LINKAGE_JOINT (and aliases)
- /SENSOR/SPRING_TRANSVERSE_SNAP_RATE (and aliases)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailLadTransverseMatrixMicrocrackingFailureRate,
    EngElectrothermoflexomagnetochiralmonopoleplasmonicpolaritonicResonanceEnergy,
    LagmulSouriauSpinorSpatialLinkageJoint,
    SensorSpringTransverseSnapRate,
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
# 1. FAIL_LAD_TRANSVERSE_MATRIX_MICROCRACKING_FAILURE_RATE Tests
# ============================================================================

def test_m468_fail_lad_transverse_matrix_microcracking_failure_rate_fixed(tmp_path: Path):
    c1 = f"{88.0:>20.4f}{320.0:>20.4f}{0.19:>20.4f}{1.45:>20.4f}{0.96:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M468 Fail Lad Transverse Matrix Microcracking Failure Rate Fixed
2022 0
/FAIL/LAD_TRANSVERSE_MATRIX_MICROCRACKING_FAILURE_RATE/1
Transverse Matrix Microcracking Failure Rate Card Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.fail_ladtransversematrixmicrocrackingfailurerates
    fm = model.fail_ladtransversematrixmicrocrackingfailurerates[1]
    assert isinstance(fm, FailLadTransverseMatrixMicrocrackingFailureRate)
    assert fm.mat_id == 1
    assert pytest.approx(fm.sigma_tmmcfr0) == 88.0
    assert pytest.approx(fm.sigma_tmmcfrc) == 320.0
    assert pytest.approx(fm.gamma_tmmcfr) == 0.19
    assert pytest.approx(fm.p_tmmcfr) == 1.45
    assert pytest.approx(fm.d_tmmcfr_max) == 0.96
    assert fm.ifail_sh == 2
    assert fm.ifail_so == 1

    # Properties
    assert pytest.approx(fm.sigma_tmmcf0) == 88.0
    assert pytest.approx(fm.sigma_tmmcfc) == 320.0
    assert pytest.approx(fm.gamma_tmmcf) == 0.19
    assert pytest.approx(fm.p_tmmcf) == 1.45
    assert pytest.approx(fm.d_tmmcf_max) == 0.96

    assert pytest.approx(fm.sigma_tmmc0) == 88.0
    assert pytest.approx(fm.sigma_tmmcc) == 320.0
    assert pytest.approx(fm.gamma_tmmc) == 0.19
    assert pytest.approx(fm.p_tmmc) == 1.45
    assert pytest.approx(fm.d_tmmc_max) == 0.96

    assert pytest.approx(fm.sigma_tmc0) == 88.0
    assert pytest.approx(fm.sigma_tmcc) == 320.0
    assert pytest.approx(fm.gamma_tmc) == 0.19
    assert pytest.approx(fm.p_tmc) == 1.45
    assert pytest.approx(fm.d_tmc_max) == 0.96


def test_m468_fail_lad_transverse_matrix_microcracking_failure_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M468 Fail Lad Transverse Matrix Microcracking Failure Rate Free
/FAIL/LAD_TRANSVERSE_MATRIX_MICROCRACKING_FAILURE_RATE/2
98.0, 340.0, 0.22, 1.58, 0.97
1, 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.fail_ladtransversematrixmicrocrackingfailurerates
    fm = model.fail_ladtransversematrixmicrocrackingfailurerates[2]
    assert pytest.approx(fm.sigma_tmmcfr0) == 98.0
    assert pytest.approx(fm.sigma_tmmcfrc) == 340.0
    assert pytest.approx(fm.gamma_tmmcfr) == 0.22
    assert pytest.approx(fm.p_tmmcfr) == 1.58
    assert pytest.approx(fm.d_tmmcfr_max) == 0.97
    assert fm.ifail_sh == 1
    assert fm.ifail_so == 2


def test_m468_fail_lad_transverse_matrix_microcracking_failure_rate_aliases(tmp_path: Path):
    aliases = [
        "/FAIL/LADEVEZE_TRANSVERSE_MATRIX_MICROCRACKING_FAILURE_RATE",
        "/FAIL/LAD_TRANSVERSE_MATRIX_MICROCRACKING_FAILURE",
        "/FAIL/LADEVEZE_TRANSVERSE_MATRIX_MICROCRACKING_FAILURE",
        "/FAIL/LAD_TRANSVERSE_MATRIX_CRACKING_FAILURE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_MATRIX_CRACKING_FAILURE_RATE",
        "/FAIL/LAD_TRANSVERSE_MATRIX_CRACKING_FAILURE",
        "/FAIL/LADEVEZE_TRANSVERSE_MATRIX_CRACKING_FAILURE",
        "/FAIL/LAD_TRANSVERSE_MATRIX_MICRO_CRACKING_FAILURE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_MATRIX_MICRO_CRACKING_FAILURE_RATE",
        "/FAIL/LAD_TRANSVERSE_MATRIX_MICRO_CRACKING_FAILURE",
        "/FAIL/LADEVEZE_TRANSVERSE_MATRIX_MICRO_CRACKING_FAILURE",
        "/FAIL/LAD_TMMCFR",
        "/FAIL/LAD_TMMCFR_MODEL",
        "/FAIL/LAD_TMMCFR_LAW",
        "/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_MATRIX_MICROCRACKING_FAILURE",
        "/FAIL/LAD_TRANSVERSE_MATRIX_MICROCRACKING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_MATRIX_MICROCRACKING_DAMAGE_RATE",
        "/FAIL/LAD_TRANSVERSE_MATRIX_CRACKING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_MATRIX_CRACKING_DAMAGE_RATE",
        "/FAIL/LAD_TRANSVERSE_MATRIX_MICRO_CRACKING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_MATRIX_MICRO_CRACKING_DAMAGE_RATE",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test M468 Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "82.0, 300.0, 0.16, 1.35, 0.95",
            "1, 1",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.fail_ladtransversematrixmicrocrackingfailurerates
        fm = model.fail_ladtransversematrixmicrocrackingfailurerates[idx]
        assert pytest.approx(fm.sigma_tmmcfr0) == 82.0


# ============================================================================
# 2. ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALMONOPOLEPLASMONICPOLARITONIC_RESONANCE_ENERGY Tests
# ============================================================================

def test_m468_eng_energy_fixed(tmp_path: Path):
    c1 = f"{2.9e-4:>20.6f}{22:>10d}"
    deck = f"""# RADIOSS ENGINE DECK
/BEGIN
Test M468 Eng Output Fixed
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMONOPOLEPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Monopole Resonance Energy Directive Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralmonopoleplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralmonopoleplasmonicpolaritonic_resonance_energies[1]
    assert isinstance(eng, EngElectrothermoflexomagnetochiralmonopoleplasmonicpolaritonicResonanceEnergy)
    assert pytest.approx(eng.dt_etfcmonopoleplp) == 2.9e-4
    assert eng.sens_id == 22


def test_m468_eng_energy_free(tmp_path: Path):
    deck = """# RADIOSS ENGINE DECK
/BEGIN
Test M468 Eng Output Free
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMONOPOLEPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
3.9e-4, 25
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralmonopoleplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralmonopoleplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfcmonopoleplp) == 3.9e-4
    assert eng.sens_id == 25


def test_m468_eng_energy_aliases(tmp_path: Path):
    aliases = [
        "/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_MONOPOLE_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALMONOPOLEPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMONOPOLEPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALMONOPOLEPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOMONOPOLECHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_MONOPOLE_CHIRAL_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOMONOPOLECHIRALPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOMONOPOLECHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOMONOPOLECHIRALPLASMONICPOLARITONIC_RESONANCE",
    ]
    deck_lines = ["# RADIOSS ENGINE DECK", "/BEGIN", "Test Eng Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "4.9e-4, 33",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.eng_electrothermoflexomagnetochiralmonopoleplasmonicpolaritonic_resonance_energies
        eng = model.eng_electrothermoflexomagnetochiralmonopoleplasmonicpolaritonic_resonance_energies[idx]
        assert pytest.approx(eng.dt_etfcmonopoleplp) == 4.9e-4


# ============================================================================
# 3. LAGMUL_SOURIAU_SPINOR_SPATIAL_LINKAGE_JOINT Tests
# ============================================================================

def test_m468_souriau_spinor_joint_fixed(tmp_path: Path):
    c1 = f"{170:>10d}{171:>10d}{172:>10d}{4.1e6:>20.4f}{19:>10d}{2.0e-5:>20.6f}"
    c2 = f"{24.5:>20.4f}{31.0:>20.4f}{86.0:>20.4f}{17.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M468 Souriau Joint Fixed
2022 0
/SOURIAU_SPINOR_SPATIAL_LINKAGE_JOINT/1
Souriau Spinor Joint Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_souriau_spinor_spatial_linkage_joints
    j = model.lagmul_souriau_spinor_spatial_linkage_joints[1]
    assert isinstance(j, LagmulSouriauSpinorSpatialLinkageJoint)
    assert j.node1 == 170
    assert j.node2 == 171
    assert j.node3 == 172
    assert pytest.approx(j.stiff) == 4.1e6
    assert j.skew_id == 19
    assert pytest.approx(j.tol) == 2.0e-5
    assert pytest.approx(j.link_len_a) == 24.5
    assert pytest.approx(j.link_len_b) == 31.0
    assert pytest.approx(j.twist_angle_alpha) == 86.0
    assert pytest.approx(j.offset_distance_s) == 17.0

    # Offset aliases
    assert pytest.approx(j.offset_distance_r) == 17.0
    assert pytest.approx(j.offset_distance_v) == 17.0
    assert pytest.approx(j.offset_distance_h) == 17.0
    assert pytest.approx(j.offset_distance_u) == 17.0
    assert pytest.approx(j.offset_distance_f) == 17.0


def test_m468_souriau_spinor_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M468 Souriau Joint Free
/LAGMUL/SOURIAU_SPINOR_SPATIAL_LINKAGE_JOINT/2
270, 271, 272, 5.1e6, 20, 3.0e-5
27.0, 33.5, 99.0, 19.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_souriau_spinor_spatial_linkage_joints
    j = model.lagmul_souriau_spinor_spatial_linkage_joints[2]
    assert j.node1 == 270
    assert j.node2 == 271
    assert j.node3 == 272
    assert pytest.approx(j.stiff) == 5.1e6
    assert j.skew_id == 20
    assert pytest.approx(j.tol) == 3.0e-5
    assert pytest.approx(j.link_len_a) == 27.0
    assert pytest.approx(j.link_len_b) == 33.5
    assert pytest.approx(j.twist_angle_alpha) == 99.0
    assert pytest.approx(j.offset_distance_s) == 19.0


def test_m468_souriau_spinor_joint_aliases(tmp_path: Path):
    aliases = [
        "/SOURIAU_SPINOR_SPATIAL_LINKAGE",
        "/LAGMUL/SOURIAU_SPINOR_SPATIAL_LINKAGE",
        "/SOURIAU_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM",
        "/SOURIAU_SPINOR_SPATIAL_SYMMETRIC_MECHANISM",
        "/SOURIAU_SPINOR_SPATIAL_6R_MECHANISM",
        "/SOURIAU_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM",
        "/LAGMUL/SOURIAU_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/SOURIAU_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/SOURIAU_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/SOURIAU_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/SOURIAU_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/SOURIAU_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/KIRILLOV_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/KIRILLOV_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/KIRILLOV_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/KIRILLOV_TWISTOR_SPATIAL_LINKAGE_JOINT",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test Souriau Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "370, 371, 372, 1.6e6, 0, 1e-6",
            "21.0, 21.0, 0.0, 0.0",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.lagmul_souriau_spinor_spatial_linkage_joints
        j = model.lagmul_souriau_spinor_spatial_linkage_joints[idx]
        assert j.node1 == 370


# ============================================================================
# 4. SENSOR_SPRING_TRANSVERSE_SNAP_RATE Tests
# ============================================================================

def test_m468_sensor_spring_transverse_snap_rate_fixed(tmp_path: Path):
    c1 = f"{19:>10d}{2.25e5:>20.4f}{0.0036:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_SNAP_RATE/1
Transverse Snap Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_snap_rates
    s = model.sensor_spring_transverse_snap_rates[1]
    assert isinstance(s, SensorSpringTransverseSnapRate)
    assert s.spring_id == 19
    assert pytest.approx(s.jtrans_snp_max) == 2.25e5
    assert pytest.approx(s.t_delay) == 0.0036

    # Check property aliases
    assert pytest.approx(s.jtrans_snap_max) == 2.25e5
    assert pytest.approx(s.j_trans_snp_max) == 2.25e5
    assert pytest.approx(s.j_snap_trans_max) == 2.25e5
    assert pytest.approx(s.j_trans_snap_max) == 2.25e5
    assert pytest.approx(s.j_shear_snap_max) == 2.25e5
    assert pytest.approx(s.jtrans_rate_max) == 2.25e5
    assert pytest.approx(s.jtrans_roc_rate_max) == 2.25e5
    assert pytest.approx(s.jtrans_drop_rate_max) == 2.25e5
    assert pytest.approx(s.jtrans_crk_rate_max) == 2.25e5
    assert pytest.approx(s.jtrans_pop_rate_max) == 2.25e5
    assert pytest.approx(s.jtrans_lock_rate_max) == 2.25e5
    assert pytest.approx(s.jtrans_snap_rate_max) == 2.25e5
    assert pytest.approx(s.j_transverse_snap_max) == 2.25e5


def test_m468_sensor_spring_transverse_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Transverse Snap Rate Free Format Test
/SENSOR/SPRING_TRANSVERSE_SNAP_RATE/2
29, 3.25e5, 0.0046
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_transverse_snap_rates
    s = model.sensor_spring_transverse_snap_rates[2]
    assert s.spring_id == 29
    assert pytest.approx(s.jtrans_snp_max) == 3.25e5
    assert pytest.approx(s.t_delay) == 0.0046


def test_m468_sensor_spring_transverse_snap_rate_aliases(tmp_path: Path):
    aliases = [
        "/SENSOR/SPRING_TRANS_SNAP_RATE",
        "/SENSOR/SPRING_TRANSVERSE_SNAP",
        "/SENSOR/SPRING_TRANS_SNAP",
        "/SENSOR/SPRING_SHEAR_SNAP_RATE",
        "/SENSOR/SPRING_SHEAR_SNAP",
        "/SENSOR/SPRING_POP_RATE_TRANSVERSE_SNAP",
        "/SENSOR/SPRING_LOCK_RATE_TRANSVERSE_SNAP",
        "/SENSOR/SPRING_SNAP_RATE_TRANSVERSE",
        "/SENSOR/SPRING_SNAP_RATE_TRANS",
        "/SENSOR/SPRING_SNAP_RATE_SHEAR",
        "/SENSOR/TRANS_SNAP_RATE_SPRING",
        "/SENSOR/SHEAR_SNAP_RATE_SPRING",
        "/SENSOR/SPRING_SNAP_TRANSVERSE",
        "/SENSOR/SPRING_SNAP_SHEAR",
        "/SENSOR/SPRING_RATE_TRANSVERSE_SNAP",
        "/SENSOR/SPRING_RATE_TRANS_SNAP",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Sensor Spring Transverse Snap Rate Aliases Test"]
    for sid, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{sid}",
            "42, 4.3e5, 0.0022",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(10, 10 + len(aliases)):
        assert sid in model.sensor_spring_transverse_snap_rates
        s = model.sensor_spring_transverse_snap_rates[sid]
        assert s.spring_id == 42
        assert pytest.approx(s.jtrans_snp_max) == 4.3e5


# ============================================================================
# 5. Error handling tests
# ============================================================================

def test_m468_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_TRANSVERSE_MATRIX_MICROCRACKING_FAILURE_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMONOPOLEPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/SOURIAU_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TRANSVERSE_SNAP_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4

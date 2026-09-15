"""Tests for Milestone M460:
- /FAIL/LAD_COUPLED_FIBER_COMPRESSION_FAILURE_RATE (and aliases)
- /ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANYONPLASMONICPOLARITONIC_RESONANCE_ENERGY (and aliases)
- /LAGMUL/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT / /SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT (and aliases)
- /SENSOR/SPRING_TOTAL_POP_RATE (and aliases)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailLadCoupledFiberCompressionFailureRate,
    EngElectrothermoflexomagnetochiralanyonplasmonicpolaritonicResonanceEnergy,
    LagmulSymplecticSpinorSpatialLinkageJoint,
    SensorSpringTotalPopRate,
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
# 1. FAIL_LAD_COUPLED_FIBER_COMPRESSION_FAILURE_RATE Tests
# ============================================================================

def test_m460_fail_lad_coupled_fiber_compression_failure_rate_fixed(tmp_path: Path):
    c1 = f"{115.0:>20.4f}{435.0:>20.4f}{0.18:>20.4f}{1.28:>20.4f}{0.93:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M460 Fail Lad Coupled Fiber Compression Failure Rate Fixed
2022 0
/FAIL/LAD_COUPLED_FIBER_COMPRESSION_FAILURE_RATE/1
Coupled Fiber Compression Failure Rate Card Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.fail_ladcoupledfibercompressionfailurerates
    fm = model.fail_ladcoupledfibercompressionfailurerates[1]
    assert isinstance(fm, FailLadCoupledFiberCompressionFailureRate)
    assert fm.mat_id == 1
    assert pytest.approx(fm.sigma_cfcfr0) == 115.0
    assert pytest.approx(fm.sigma_cfcfrc) == 435.0
    assert pytest.approx(fm.gamma_cfcfr) == 0.18
    assert pytest.approx(fm.p_cfcfr) == 1.28
    assert pytest.approx(fm.d_cfcfr_max) == 0.93
    assert fm.ifail_sh == 2
    assert fm.ifail_so == 1

    # Properties
    assert pytest.approx(fm.sigma_cfcf0) == 115.0
    assert pytest.approx(fm.sigma_cfcfc) == 435.0
    assert pytest.approx(fm.gamma_cfcf) == 0.18
    assert pytest.approx(fm.p_cfcf) == 1.28
    assert pytest.approx(fm.d_cfcf_max) == 0.93

    assert pytest.approx(fm.sigma_cfc0) == 115.0
    assert pytest.approx(fm.sigma_cfcc) == 435.0
    assert pytest.approx(fm.gamma_cfc) == 0.18
    assert pytest.approx(fm.p_cfc) == 1.28
    assert pytest.approx(fm.d_cfc_max) == 0.93

    assert pytest.approx(fm.sigma_cfr0) == 115.0
    assert pytest.approx(fm.sigma_cfrc) == 435.0
    assert pytest.approx(fm.gamma_cfr) == 0.18
    assert pytest.approx(fm.p_cfr) == 1.28
    assert pytest.approx(fm.d_cfr_max) == 0.93


def test_m460_fail_lad_coupled_fiber_compression_failure_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M460 Fail Lad Coupled Fiber Compression Failure Rate Free
/FAIL/LAD_COUPLED_FIBER_COMPRESSION_FAILURE_RATE/2
135.0, 490.0, 0.25, 1.48, 0.96
1, 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.fail_ladcoupledfibercompressionfailurerates
    fm = model.fail_ladcoupledfibercompressionfailurerates[2]
    assert pytest.approx(fm.sigma_cfcfr0) == 135.0
    assert pytest.approx(fm.sigma_cfcfrc) == 490.0
    assert pytest.approx(fm.gamma_cfcfr) == 0.25
    assert pytest.approx(fm.p_cfcfr) == 1.48
    assert pytest.approx(fm.d_cfcfr_max) == 0.96
    assert fm.ifail_sh == 1
    assert fm.ifail_so == 2


def test_m460_fail_lad_coupled_fiber_compression_failure_rate_aliases(tmp_path: Path):
    aliases = [
        "/FAIL/LADEVEZE_COUPLED_FIBER_COMPRESSION_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_FIBER_COMPRESSION_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_FIBER_COMPRESSION_FAILURE",
        "/FAIL/LAD_COUPLE_FIBER_COMPRESSION_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLE_FIBER_COMPRESSION_FAILURE_RATE",
        "/FAIL/LAD_COUPLE_FIBER_COMPRESSION_FAILURE",
        "/FAIL/LADEVEZE_COUPLE_FIBER_COMPRESSION_FAILURE",
        "/FAIL/LAD_COUPLED_FIBER_COMPRESSION_RATE",
        "/FAIL/LADEVEZE_COUPLED_FIBER_COMPRESSION_RATE",
        "/FAIL/LAD_COUPLE_FIBER_COMPRESSION_RATE",
        "/FAIL/LADEVEZE_COUPLE_FIBER_COMPRESSION_RATE",
        "/FAIL/LAD_COUPLED_FIBER_COMPRESSIVE_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLED_FIBER_COMPRESSIVE_FAILURE_RATE",
        "/FAIL/LAD_COUPLE_FIBER_COMPRESSIVE_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLE_FIBER_COMPRESSIVE_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_FIBER_COMPRESSIVE_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_FIBER_COMPRESSIVE_FAILURE",
        "/FAIL/LAD_COUPLE_FIBER_COMPRESSIVE_FAILURE",
        "/FAIL/LADEVEZE_COUPLE_FIBER_COMPRESSIVE_FAILURE",
        "/FAIL/LAD_CFCFR",
        "/FAIL/LAD_CFCFR_MODEL",
        "/FAIL/LAD_CFCFR_LAW",
        "/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_FIBER_COMPRESSION_FAILURE",
        "/FAIL/LAD_COUPLED_FIBER_COMPRESSION_DAMAGE_RATE",
        "/FAIL/LADEVEZE_COUPLED_FIBER_COMPRESSION_DAMAGE_RATE",
        "/FAIL/LAD_COUPLE_FIBER_COMPRESSION_DAMAGE_RATE",
        "/FAIL/LADEVEZE_COUPLE_FIBER_COMPRESSION_DAMAGE_RATE",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test M460 Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "105.0, 410.0, 0.12, 1.05, 0.98",
            "1, 1",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.fail_ladcoupledfibercompressionfailurerates
        fm = model.fail_ladcoupledfibercompressionfailurerates[idx]
        assert pytest.approx(fm.sigma_cfcfr0) == 105.0


# ============================================================================
# 2. ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALANYONPLASMONICPOLARITONIC_RESONANCE_ENERGY Tests
# ============================================================================

def test_m460_eng_energy_fixed(tmp_path: Path):
    c1 = f"{1.6e-4:>20.6f}{14:>10d}"
    deck = f"""# RADIOSS ENGINE DECK
/BEGIN
Test M460 Eng Output Fixed
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANYONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Anyon Resonance Energy Directive Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralanyonplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralanyonplasmonicpolaritonic_resonance_energies[1]
    assert isinstance(eng, EngElectrothermoflexomagnetochiralanyonplasmonicpolaritonicResonanceEnergy)
    assert pytest.approx(eng.dt_etfcanyonplp) == 1.6e-4
    assert eng.sens_id == 14


def test_m460_eng_energy_free(tmp_path: Path):
    deck = """# RADIOSS ENGINE DECK
/BEGIN
Test M460 Eng Output Free
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANYONPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
2.6e-4, 16
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralanyonplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralanyonplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfcanyonplp) == 2.6e-4
    assert eng.sens_id == 16


def test_m460_eng_energy_aliases(tmp_path: Path):
    aliases = [
        "/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_ANYON_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALANYONPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANYONPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALANYONPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOANYONCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_ANYON_CHIRAL_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOANYONCHIRALPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOANYONCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOANYONCHIRALPLASMONICPOLARITONIC_RESONANCE",
    ]
    deck_lines = ["# RADIOSS ENGINE DECK", "/BEGIN", "Test Eng Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "3.6e-4, 22",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.eng_electrothermoflexomagnetochiralanyonplasmonicpolaritonic_resonance_energies
        eng = model.eng_electrothermoflexomagnetochiralanyonplasmonicpolaritonic_resonance_energies[idx]
        assert pytest.approx(eng.dt_etfcanyonplp) == 3.6e-4


# ============================================================================
# 3. LAGMUL_SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT Tests
# ============================================================================

def test_m460_symplectic_spinor_joint_fixed(tmp_path: Path):
    c1 = f"{104:>10d}{105:>10d}{106:>10d}{2.6e6:>20.4f}{7:>10d}{1.0e-5:>20.6f}"
    c2 = f"{16.5:>20.4f}{23.0:>20.4f}{50.0:>20.4f}{9.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M460 Symplectic Joint Fixed
2022 0
/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/1
Symplectic Spinor Joint Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_symplectic_spinor_spatial_linkage_joints
    j = model.lagmul_symplectic_spinor_spatial_linkage_joints[1]
    assert isinstance(j, LagmulSymplecticSpinorSpatialLinkageJoint)
    assert j.node1 == 104
    assert j.node2 == 105
    assert j.node3 == 106
    assert pytest.approx(j.stiff) == 2.6e6
    assert j.skew_id == 7
    assert pytest.approx(j.tol) == 1.0e-5
    assert pytest.approx(j.link_len_a) == 16.5
    assert pytest.approx(j.link_len_b) == 23.0
    assert pytest.approx(j.twist_angle_alpha) == 50.0
    assert pytest.approx(j.offset_distance_s) == 9.5

    # Offset aliases
    assert pytest.approx(j.offset_distance_r) == 9.5
    assert pytest.approx(j.offset_distance_v) == 9.5
    assert pytest.approx(j.offset_distance_h) == 9.5
    assert pytest.approx(j.offset_distance_u) == 9.5
    assert pytest.approx(j.offset_distance_f) == 9.5


def test_m460_symplectic_spinor_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M460 Symplectic Joint Free
/LAGMUL/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/2
204, 205, 206, 3.6e6, 8, 2.0e-5
19.0, 25.5, 65.0, 11.2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_symplectic_spinor_spatial_linkage_joints
    j = model.lagmul_symplectic_spinor_spatial_linkage_joints[2]
    assert j.node1 == 204
    assert j.node2 == 205
    assert j.node3 == 206
    assert pytest.approx(j.stiff) == 3.6e6
    assert j.skew_id == 8
    assert pytest.approx(j.tol) == 2.0e-5
    assert pytest.approx(j.link_len_a) == 19.0
    assert pytest.approx(j.link_len_b) == 25.5
    assert pytest.approx(j.twist_angle_alpha) == 65.0
    assert pytest.approx(j.offset_distance_s) == 11.2


def test_m460_symplectic_spinor_joint_aliases(tmp_path: Path):
    aliases = [
        "/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE",
        "/LAGMUL/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE",
        "/SYMPLECTIC_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM",
        "/SYMPLECTIC_SPINOR_SPATIAL_SYMMETRIC_MECHANISM",
        "/SYMPLECTIC_SPINOR_SPATIAL_6R_MECHANISM",
        "/SYMPLECTIC_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM",
        "/LAGMUL/SYMPLECTIC_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/SYMPLECTIC_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/SYMPLECTIC_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/SYMPLECTIC_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/SYMPLECTIC_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/SYMPLECTIC_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test Symplectic Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "304, 305, 306, 1.0e6, 0, 1e-6",
            "12.0, 12.0, 0.0, 0.0",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.lagmul_symplectic_spinor_spatial_linkage_joints
        j = model.lagmul_symplectic_spinor_spatial_linkage_joints[idx]
        assert j.node1 == 304


# ============================================================================
# 4. SENSOR_SPRING_TOTAL_POP_RATE Tests
# ============================================================================

def test_m460_sensor_spring_total_pop_rate_fixed(tmp_path: Path):
    c1 = f"{10:>10d}{1.96e5:>20.4f}{0.0026:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_POP_RATE/1
Total Pop Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_pop_rates
    s = model.sensor_spring_total_pop_rates[1]
    assert isinstance(s, SensorSpringTotalPopRate)
    assert s.spring_id == 10
    assert pytest.approx(s.jtot_pop_max) == 1.96e5
    assert pytest.approx(s.t_delay) == 0.0026

    # Check property aliases
    assert pytest.approx(s.j_pop_tot_max) == 1.96e5
    assert pytest.approx(s.j_tot_pop_max) == 1.96e5
    assert pytest.approx(s.jtot_shot_max) == 1.96e5
    assert pytest.approx(s.jtot_drop_max) == 1.96e5
    assert pytest.approx(s.jtot_lock_max) == 1.96e5
    assert pytest.approx(s.jtot_snp_max) == 1.96e5
    assert pytest.approx(s.jtot_snap_max) == 1.96e5
    assert pytest.approx(s.jtot_rate_max) == 1.96e5
    assert pytest.approx(s.jtot_roc_rate_max) == 1.96e5
    assert pytest.approx(s.jtot_drop_rate_max) == 1.96e5
    assert pytest.approx(s.jtot_crk_rate_max) == 1.96e5
    assert pytest.approx(s.jtot_pop_rate_max) == 1.96e5
    assert pytest.approx(s.jtot_crackle_max) == 1.96e5
    assert pytest.approx(s.jtot_crk_max) == 1.96e5
    assert pytest.approx(s.jres_pop_max) == 1.96e5
    assert pytest.approx(s.j_res_pop_max) == 1.96e5
    assert pytest.approx(s.j_pop_res_max) == 1.96e5


def test_m460_sensor_spring_total_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Pop Rate Free Format Test
/SENSOR/SPRING_TOTAL_POP_RATE/2
20, 2.96e5, 0.0036
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_pop_rates
    s = model.sensor_spring_total_pop_rates[2]
    assert s.spring_id == 20
    assert pytest.approx(s.jtot_pop_max) == 2.96e5
    assert pytest.approx(s.t_delay) == 0.0036


def test_m460_sensor_spring_total_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Pop Rate Aliases Test
/SENSOR/SPRING_TOT_POP_RATE/3
30, 3.6e5, 0.001
/SENSOR/SPRING_TOTAL_POP/4
30, 3.6e5, 0.001
/SENSOR/SPRING_TOT_POP/5
30, 3.6e5, 0.001
/SENSOR/SPRING_RESULTANT_POP_RATE/6
30, 3.6e5, 0.001
/SENSOR/SPRING_RES_POP_RATE/7
30, 3.6e5, 0.001
/SENSOR/SPRING_RESULTANT_POP/8
30, 3.6e5, 0.001
/SENSOR/SPRING_RES_POP/9
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_RATE_TOTAL/10
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_RATE_TOT/11
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_RATE_RESULTANT/12
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_RATE_RES/13
30, 3.6e5, 0.001
/SENSOR/TOTAL_POP_RATE_SPRING/14
30, 3.6e5, 0.001
/SENSOR/TOT_POP_RATE_SPRING/15
30, 3.6e5, 0.001
/SENSOR/RESULTANT_POP_RATE_SPRING/16
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_TOTAL/17
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_TOT/18
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_RESULTANT/19
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_RES/20
30, 3.6e5, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 21):
        assert sid in model.sensor_spring_total_pop_rates
        s = model.sensor_spring_total_pop_rates[sid]
        assert s.spring_id == 30
        assert pytest.approx(s.jtot_pop_max) == 3.6e5


# ============================================================================
# 5. Error handling tests
# ============================================================================

def test_m460_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_COUPLED_FIBER_COMPRESSION_FAILURE_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANYONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TOTAL_POP_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4

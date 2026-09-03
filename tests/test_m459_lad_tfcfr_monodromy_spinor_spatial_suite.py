"""Tests for Milestone M459:
- /FAIL/LAD_TRANSVERSE_FIBER_COMPRESSION_FAILURE_RATE (and aliases)
- /ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMAJORANAPLASMONICPOLARITONIC_RESONANCE_ENERGY (and aliases)
- /LAGMUL/MONODROMY_SPINOR_SPATIAL_LINKAGE_JOINT / /MONODROMY_SPINOR_SPATIAL_LINKAGE_JOINT (and aliases)
- /SENSOR/SPRING_BENDING_POP_RATE (and aliases)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailLadTransverseFiberCompressionFailureRate,
    EngElectrothermoflexomagnetochiralmajoranaplasmonicpolaritonicResonanceEnergy,
    LagmulMonodromySpinorSpatialLinkageJoint,
    SensorSpringBendingPopRate,
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
# 1. FAIL_LAD_TRANSVERSE_FIBER_COMPRESSION_FAILURE_RATE Tests
# ============================================================================

def test_m459_fail_lad_transverse_fiber_compression_failure_rate_fixed(tmp_path: Path):
    c1 = f"{110.0:>20.4f}{420.0:>20.4f}{0.15:>20.4f}{1.25:>20.4f}{0.92:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M459 Fail Lad Transverse Fiber Compression Failure Rate Fixed
2022 0
/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSION_FAILURE_RATE/1
Dynamic Transverse Fiber Compression Failure Rate Card Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.fail_ladtransversefibercompressionfailurerates
    fm = model.fail_ladtransversefibercompressionfailurerates[1]
    assert isinstance(fm, FailLadTransverseFiberCompressionFailureRate)
    assert fm.mat_id == 1
    assert pytest.approx(fm.sigma_tfcfr0) == 110.0
    assert pytest.approx(fm.sigma_tfcfrc) == 420.0
    assert pytest.approx(fm.gamma_tfcfr) == 0.15
    assert pytest.approx(fm.p_tfcfr) == 1.25
    assert pytest.approx(fm.d_tfcfr_max) == 0.92
    assert fm.ifail_sh == 2
    assert fm.ifail_so == 1

    # Properties
    assert pytest.approx(fm.sigma_tfcf0) == 110.0
    assert pytest.approx(fm.sigma_tfcfc) == 420.0
    assert pytest.approx(fm.gamma_tfcf) == 0.15
    assert pytest.approx(fm.p_tfcf) == 1.25
    assert pytest.approx(fm.d_tfcf_max) == 0.92

    assert pytest.approx(fm.sigma_tfc0) == 110.0
    assert pytest.approx(fm.sigma_tfcc) == 420.0
    assert pytest.approx(fm.gamma_tfc) == 0.15
    assert pytest.approx(fm.p_tfc) == 1.25
    assert pytest.approx(fm.d_tfc_max) == 0.92

    assert pytest.approx(fm.sigma_tfr0) == 110.0
    assert pytest.approx(fm.sigma_tfrc) == 420.0
    assert pytest.approx(fm.gamma_tfr) == 0.15
    assert pytest.approx(fm.p_tfr) == 1.25
    assert pytest.approx(fm.d_tfr_max) == 0.92


def test_m459_fail_lad_transverse_fiber_compression_failure_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M459 Fail Lad Transverse Fiber Compression Failure Rate Free
/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSION_FAILURE_RATE/2
130.0, 480.0, 0.22, 1.45, 0.95
1, 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.fail_ladtransversefibercompressionfailurerates
    fm = model.fail_ladtransversefibercompressionfailurerates[2]
    assert pytest.approx(fm.sigma_tfcfr0) == 130.0
    assert pytest.approx(fm.sigma_tfcfrc) == 480.0
    assert pytest.approx(fm.gamma_tfcfr) == 0.22
    assert pytest.approx(fm.p_tfcfr) == 1.45
    assert pytest.approx(fm.d_tfcfr_max) == 0.95
    assert fm.ifail_sh == 1
    assert fm.ifail_so == 2


def test_m459_fail_lad_transverse_fiber_compression_failure_rate_aliases(tmp_path: Path):
    aliases = [
        "/FAIL/LADEVEZE_TRANSVERSE_FIBER_COMPRESSION_FAILURE_RATE",
        "/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSION_FAILURE",
        "/FAIL/LADEVEZE_TRANSVERSE_FIBER_COMPRESSION_FAILURE",
        "/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSION_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_FIBER_COMPRESSION_RATE",
        "/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSIVE_FAILURE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_FIBER_COMPRESSIVE_FAILURE_RATE",
        "/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSIVE_FAILURE",
        "/FAIL/LADEVEZE_TRANSVERSE_FIBER_COMPRESSIVE_FAILURE",
        "/FAIL/LAD_TFCFR",
        "/FAIL/LAD_TFCFR_MODEL",
        "/FAIL/LAD_TFCFR_LAW",
        "/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_FIBER_COMPRESSION_FAILURE",
        "/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSION_DAMAGE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_FIBER_COMPRESSION_DAMAGE_RATE",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test M459 Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "100.0, 400.0, 0.1, 1.0, 0.99",
            "1, 1",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.fail_ladtransversefibercompressionfailurerates
        fm = model.fail_ladtransversefibercompressionfailurerates[idx]
        assert pytest.approx(fm.sigma_tfcfr0) == 100.0


# ============================================================================
# 2. ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALMAJORANAPLASMONICPOLARITONIC_RESONANCE_ENERGY Tests
# ============================================================================

def test_m459_eng_energy_fixed(tmp_path: Path):
    c1 = f"{1.5e-4:>20.6f}{12:>10d}"
    deck = f"""# RADIOSS ENGINE DECK
/BEGIN
Test M459 Eng Output Fixed
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMAJORANAPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Majorana Resonance Energy Directive Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralmajoranaplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralmajoranaplasmonicpolaritonic_resonance_energies[1]
    assert isinstance(eng, EngElectrothermoflexomagnetochiralmajoranaplasmonicpolaritonicResonanceEnergy)
    assert pytest.approx(eng.dt_etfcmajoranaplp) == 1.5e-4
    assert eng.sens_id == 12


def test_m459_eng_energy_free(tmp_path: Path):
    deck = """# RADIOSS ENGINE DECK
/BEGIN
Test M459 Eng Output Free
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMAJORANAPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
2.5e-4, 15
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralmajoranaplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralmajoranaplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfcmajoranaplp) == 2.5e-4
    assert eng.sens_id == 15


def test_m459_eng_energy_aliases(tmp_path: Path):
    aliases = [
        "/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_MAJORANA_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALMAJORANAPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMAJORANAPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALMAJORANAPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOMAJORANACHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_MAJORANA_CHIRAL_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOMAJORANACHIRALPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOMAJORANACHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOMAJORANACHIRALPLASMONICPOLARITONIC_RESONANCE",
    ]
    deck_lines = ["# RADIOSS ENGINE DECK", "/BEGIN", "Test Eng Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "3.5e-4, 20",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.eng_electrothermoflexomagnetochiralmajoranaplasmonicpolaritonic_resonance_energies
        eng = model.eng_electrothermoflexomagnetochiralmajoranaplasmonicpolaritonic_resonance_energies[idx]
        assert pytest.approx(eng.dt_etfcmajoranaplp) == 3.5e-4


# ============================================================================
# 3. LAGMUL_MONODROMY_SPINOR_SPATIAL_LINKAGE_JOINT Tests
# ============================================================================

def test_m459_monodromy_spinor_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{2.5e6:>20.4f}{5:>10d}{1.0e-5:>20.6f}"
    c2 = f"{15.5:>20.4f}{22.0:>20.4f}{45.0:>20.4f}{8.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M459 Monodromy Joint Fixed
2022 0
/MONODROMY_SPINOR_SPATIAL_LINKAGE_JOINT/1
Monodromy Spinor Joint Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_monodromy_spinor_spatial_linkage_joints
    j = model.lagmul_monodromy_spinor_spatial_linkage_joints[1]
    assert isinstance(j, LagmulMonodromySpinorSpatialLinkageJoint)
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 2.5e6
    assert j.skew_id == 5
    assert pytest.approx(j.tol) == 1.0e-5
    assert pytest.approx(j.link_len_a) == 15.5
    assert pytest.approx(j.link_len_b) == 22.0
    assert pytest.approx(j.twist_angle_alpha) == 45.0
    assert pytest.approx(j.offset_distance_s) == 8.5

    # Offset aliases
    assert pytest.approx(j.offset_distance_r) == 8.5
    assert pytest.approx(j.offset_distance_v) == 8.5
    assert pytest.approx(j.offset_distance_h) == 8.5
    assert pytest.approx(j.offset_distance_u) == 8.5
    assert pytest.approx(j.offset_distance_f) == 8.5


def test_m459_monodromy_spinor_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M459 Monodromy Joint Free
/LAGMUL/MONODROMY_SPINOR_SPATIAL_LINKAGE_JOINT/2
201, 202, 203, 3.5e6, 6, 2.0e-5
18.0, 24.5, 60.0, 10.2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_monodromy_spinor_spatial_linkage_joints
    j = model.lagmul_monodromy_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 3.5e6
    assert j.skew_id == 6
    assert pytest.approx(j.tol) == 2.0e-5
    assert pytest.approx(j.link_len_a) == 18.0
    assert pytest.approx(j.link_len_b) == 24.5
    assert pytest.approx(j.twist_angle_alpha) == 60.0
    assert pytest.approx(j.offset_distance_s) == 10.2


def test_m459_monodromy_spinor_joint_aliases(tmp_path: Path):
    aliases = [
        "/MONODROMY_SPINOR_SPATIAL_LINKAGE",
        "/LAGMUL/MONODROMY_SPINOR_SPATIAL_LINKAGE",
        "/MONODROMY_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM",
        "/MONODROMY_SPINOR_SPATIAL_SYMMETRIC_MECHANISM",
        "/MONODROMY_SPINOR_SPATIAL_6R_MECHANISM",
        "/MONODROMY_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM",
        "/LAGMUL/MONODROMY_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/MONODROMY_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/MONODROMY_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/MONODROMY_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/MONODROMY_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/MONODROMY_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test Monodromy Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "301, 302, 303, 1.0e6, 0, 1e-6",
            "10.0, 10.0, 0.0, 0.0",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.lagmul_monodromy_spinor_spatial_linkage_joints
        j = model.lagmul_monodromy_spinor_spatial_linkage_joints[idx]
        assert j.node1 == 301


# ============================================================================
# 4. SENSOR_SPRING_BENDING_POP_RATE Tests
# ============================================================================

def test_m459_sensor_spring_bending_pop_rate_fixed(tmp_path: Path):
    c1 = f"{10:>10d}{1.95e5:>20.4f}{0.0025:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_POP_RATE/1
Bending Pop Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_pop_rates
    s = model.sensor_spring_bending_pop_rates[1]
    assert isinstance(s, SensorSpringBendingPopRate)
    assert s.spring_id == 10
    assert pytest.approx(s.jbend_pop_max) == 1.95e5
    assert pytest.approx(s.t_delay) == 0.0025

    # Check property aliases
    assert pytest.approx(s.j_pop_bend_max) == 1.95e5
    assert pytest.approx(s.j_bend_pop_max) == 1.95e5
    assert pytest.approx(s.jbend_shot_max) == 1.95e5
    assert pytest.approx(s.jbend_drop_max) == 1.95e5
    assert pytest.approx(s.jbend_lock_max) == 1.95e5
    assert pytest.approx(s.jbend_snp_max) == 1.95e5
    assert pytest.approx(s.jbend_snap_max) == 1.95e5
    assert pytest.approx(s.jbend_rate_max) == 1.95e5
    assert pytest.approx(s.jbend_roc_rate_max) == 1.95e5
    assert pytest.approx(s.jbend_drop_rate_max) == 1.95e5
    assert pytest.approx(s.jbend_crk_rate_max) == 1.95e5
    assert pytest.approx(s.jbend_pop_rate_max) == 1.95e5
    assert pytest.approx(s.jbend_crackle_max) == 1.95e5
    assert pytest.approx(s.jbend_crk_max) == 1.95e5
    assert pytest.approx(s.jflex_pop_max) == 1.95e5
    assert pytest.approx(s.j_flex_pop_max) == 1.95e5
    assert pytest.approx(s.j_pop_flex_max) == 1.95e5


def test_m459_sensor_spring_bending_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Bending Pop Rate Free Format Test
/SENSOR/SPRING_BENDING_POP_RATE/2
20, 2.95e5, 0.0035
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_bending_pop_rates
    s = model.sensor_spring_bending_pop_rates[2]
    assert s.spring_id == 20
    assert pytest.approx(s.jbend_pop_max) == 2.95e5
    assert pytest.approx(s.t_delay) == 0.0035


def test_m459_sensor_spring_bending_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Pop Rate Aliases Test
/SENSOR/SPRING_BEND_POP_RATE/3
30, 3.5e5, 0.001
/SENSOR/SPRING_BENDING_POP/4
30, 3.5e5, 0.001
/SENSOR/SPRING_BEND_POP/5
30, 3.5e5, 0.001
/SENSOR/SPRING_POP_RATE_BENDING/6
30, 3.5e5, 0.001
/SENSOR/SPRING_POP_RATE_BEND/7
30, 3.5e5, 0.001
/SENSOR/BENDING_POP_RATE_SPRING/8
30, 3.5e5, 0.001
/SENSOR/SPRING_POP_BENDING/9
30, 3.5e5, 0.001
/SENSOR/SPRING_POP_BEND/10
30, 3.5e5, 0.001
/SENSOR/SPRING_FLEX_POP_RATE/11
30, 3.5e5, 0.001
/SENSOR/SPRING_FLEX_POP/12
30, 3.5e5, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 13):
        assert sid in model.sensor_spring_bending_pop_rates
        s = model.sensor_spring_bending_pop_rates[sid]
        assert s.spring_id == 30
        assert pytest.approx(s.jbend_pop_max) == 3.5e5


# ============================================================================
# 5. Error handling tests
# ============================================================================

def test_m459_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSION_FAILURE_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMAJORANAPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/MONODROMY_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_BENDING_POP_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4

"""Tests for Milestone M474:
- /FAIL/LAD_TRANSVERSE_FIBER_KINKING_FAILURE_RATE (and aliases)
- /ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONIUMPLASMONICPOLARITONIC_RESONANCE_ENERGY (and aliases)
- /LAGMUL/HIRZEBRUCH_SPINOR_SPATIAL_LINKAGE_JOINT / /HIRZEBRUCH_SPINOR_SPATIAL_LINKAGE_JOINT (and aliases)
- /SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE (and aliases)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailLadTransverseFiberKinkingFailureRate,
    EngElectrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonicResonanceEnergy,
    LagmulHirzebruchSpinorSpatialLinkageJoint,
    SensorSpringTransverseCrackleRate,
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
# 1. FAIL_LAD_TRANSVERSE_FIBER_KINKING_FAILURE_RATE Tests
# ============================================================================

def test_m474_fail_lad_transverse_fiber_kinking_failure_rate_fixed(tmp_path: Path):
    c1 = f"{130.0:>20.4f}{420.0:>20.4f}{0.35:>20.4f}{1.55:>20.4f}{0.985:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M474 Fail Lad Transverse Fiber Kinking Failure Rate Fixed
2022 0
/FAIL/LAD_TRANSVERSE_FIBER_KINKING_FAILURE_RATE/1
Transverse Fiber Kinking Failure Rate Card Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.fail_ladtransversefiberkinkingfailurerates
    fm = model.fail_ladtransversefiberkinkingfailurerates[1]
    assert isinstance(fm, FailLadTransverseFiberKinkingFailureRate)
    assert fm.mat_id == 1
    assert pytest.approx(fm.sigma_tfkfr0) == 130.0
    assert pytest.approx(fm.sigma_tfkfrc) == 420.0
    assert pytest.approx(fm.gamma_tfkfr) == 0.35
    assert pytest.approx(fm.p_tfkfr) == 1.55
    assert pytest.approx(fm.d_tfkfr_max) == 0.985
    assert fm.ifail_sh == 2
    assert fm.ifail_so == 1

    # Properties aliases
    assert pytest.approx(fm.sigma_tfkf0) == 130.0
    assert pytest.approx(fm.sigma_tfkfc) == 420.0
    assert pytest.approx(fm.gamma_tfkf) == 0.35
    assert pytest.approx(fm.p_tfkf) == 1.55
    assert pytest.approx(fm.d_tfkf_max) == 0.985

    assert pytest.approx(fm.sigma_tfk0) == 130.0
    assert pytest.approx(fm.sigma_tfk0) == 130.0
    assert pytest.approx(fm.sigma_tfkc) == 420.0
    assert pytest.approx(fm.gamma_tfk) == 0.35
    assert pytest.approx(fm.p_tfk) == 1.55
    assert pytest.approx(fm.d_tfk_max) == 0.985


def test_m474_fail_lad_transverse_fiber_kinking_failure_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M474 Fail Lad Transverse Fiber Kinking Failure Rate Free
/FAIL/LAD_TRANSVERSE_FIBER_KINKING_FAILURE_RATE/2
140.0, 450.0, 0.38, 1.65, 0.99
1, 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.fail_ladtransversefiberkinkingfailurerates
    fm = model.fail_ladtransversefiberkinkingfailurerates[2]
    assert pytest.approx(fm.sigma_tfkfr0) == 140.0
    assert pytest.approx(fm.sigma_tfkfrc) == 450.0
    assert pytest.approx(fm.gamma_tfkfr) == 0.38
    assert pytest.approx(fm.p_tfkfr) == 1.65
    assert pytest.approx(fm.d_tfkfr_max) == 0.99
    assert fm.ifail_sh == 1
    assert fm.ifail_so == 2


def test_m474_fail_lad_transverse_fiber_kinking_failure_rate_aliases(tmp_path: Path):
    aliases = [
        "/FAIL/LADEVEZE_TRANSVERSE_FIBER_KINKING_FAILURE_RATE",
        "/FAIL/LAD_TRANSVERSE_FIBER_KINKING_FAILURE",
        "/FAIL/LADEVEZE_TRANSVERSE_FIBER_KINKING_FAILURE",
        "/FAIL/LAD_TRANSVERSE_FIBER_KINK_FAILURE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_FIBER_KINK_FAILURE_RATE",
        "/FAIL/LAD_TRANSVERSE_FIBER_KINK_FAILURE",
        "/FAIL/LADEVEZE_TRANSVERSE_FIBER_KINK_FAILURE",
        "/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSIVE_KINKING_FAILURE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_FIBER_COMPRESSIVE_KINKING_FAILURE_RATE",
        "/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSIVE_KINKING_FAILURE",
        "/FAIL/LADEVEZE_TRANSVERSE_FIBER_COMPRESSIVE_KINKING_FAILURE",
        "/FAIL/LAD_TRANSVERSE_FIBER_MICROBUCKLING_FAILURE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_FIBER_MICROBUCKLING_FAILURE_RATE",
        "/FAIL/LAD_TRANSVERSE_FIBER_MICROBUCKLING_FAILURE",
        "/FAIL/LADEVEZE_TRANSVERSE_FIBER_MICROBUCKLING_FAILURE",
        "/FAIL/LAD_TRANSVERSE_FIBER_MICRO_BUCKLING_FAILURE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_FIBER_MICRO_BUCKLING_FAILURE_RATE",
        "/FAIL/LAD_TFDFKFR",
        "/FAIL/LAD_TFDFKFR_MODEL",
        "/FAIL/LAD_TFDFKFR_LAW",
        "/FAIL/LAD_TFKFR",
        "/FAIL/LAD_TFKFR_MODEL",
        "/FAIL/LAD_TFKFR_LAW",
        "/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_FIBER_KINKING_FAILURE",
        "/FAIL/LAD_TRANSVERSE_FIBER_KINKING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_FIBER_KINKING_DAMAGE_RATE",
        "/FAIL/LAD_TRANSVERSE_FIBER_KINK_DAMAGE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_FIBER_KINK_DAMAGE_RATE",
        "/FAIL/LAD_TRANSVERSE_FIBER_MICROBUCKLING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_FIBER_MICROBUCKLING_DAMAGE_RATE",
    ]
    deck_lines = ["/BEGIN", "Test M474 Fail Lad Transverse Fiber Kinking Failure Rate Aliases"]
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
        assert idx in model.fail_ladtransversefiberkinkingfailurerates
        fm = model.fail_ladtransversefiberkinkingfailurerates[idx]
        assert pytest.approx(fm.sigma_tfkfr0) == float(idx) * 10.0
        assert pytest.approx(fm.sigma_tfkfrc) == float(idx) * 30.0


def test_m474_fail_lad_transverse_fiber_kinking_failure_rate_missing_card(tmp_path: Path):
    deck = """/BEGIN
Test M474 Fail Missing Card
/FAIL/LAD_TRANSVERSE_FIBER_KINKING_FAILURE_RATE/99
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
    assert "missing data card" in log.errors[0]
    assert 99 not in model.fail_ladtransversefiberkinkingfailurerates


def test_m474_fail_lad_transverse_fiber_kinking_failure_rate_properties():
    fm = FailLadTransverseFiberKinkingFailureRate(mat_id=5)
    fm.sigma_tfkf0 = 150.0
    fm.sigma_tfkfc = 490.0
    fm.gamma_tfkf = 0.44
    fm.p_tfkf = 1.78
    fm.d_tfkf_max = 0.996

    assert pytest.approx(fm.sigma_tfkfr0) == 150.0
    assert pytest.approx(fm.sigma_tfkfrc) == 490.0
    assert pytest.approx(fm.gamma_tfkfr) == 0.44
    assert pytest.approx(fm.p_tfkfr) == 1.78
    assert pytest.approx(fm.d_tfkfr_max) == 0.996

    fm.sigma_tfk0 = 160.0
    fm.sigma_tfkc = 520.0
    fm.gamma_tfk = 0.47
    fm.p_tfk = 1.82
    fm.d_tfk_max = 0.997

    assert pytest.approx(fm.sigma_tfkfr0) == 160.0
    assert pytest.approx(fm.sigma_tfkfrc) == 520.0
    assert pytest.approx(fm.gamma_tfkfr) == 0.47
    assert pytest.approx(fm.p_tfkfr) == 1.82
    assert pytest.approx(fm.d_tfkfr_max) == 0.997


# ============================================================================
# 2. ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONIUMPLASMONICPOLARITONIC_RESONANCE_ENERGY Tests
# ============================================================================

def test_m474_eng_electrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0038:>20.4f}{15:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M474 Eng Chiral Skyrmionium Resonance Energy Fixed
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONIUMPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Eng Chiral Skyrmionium Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonic_resonance_energies[1]
    assert isinstance(eng, EngElectrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonicResonanceEnergy)
    assert pytest.approx(eng.dt_etfcskyrmioniumplp) == 0.0038
    assert eng.sens_id == 15


def test_m474_eng_electrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M474 Eng Chiral Skyrmionium Resonance Energy Free
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONIUMPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.0048, 16
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfcskyrmioniumplp) == 0.0048
    assert eng.sens_id == 16


def test_m474_eng_electrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    aliases = [
        "/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_SKYRMIONIUM_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONIUMPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONIUMPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONIUMPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOSKYRMIONIUMCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_SKYRMIONIUM_CHIRAL_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOSKYRMIONIUMCHIRALPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOSKYRMIONIUMCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOSKYRMIONIUMCHIRALPLASMONICPOLARITONIC_RESONANCE",
    ]
    deck_lines = ["/BEGIN", "Test M474 Eng Aliases"]
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
        assert idx in model.eng_electrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonic_resonance_energies
        eng = model.eng_electrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonic_resonance_energies[idx]
        assert pytest.approx(eng.dt_etfcskyrmioniumplp) == 0.001 * idx
        assert eng.sens_id == idx


def test_m474_eng_electrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """/BEGIN
Test Eng Missing Card
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONIUMPLASMONICPOLARITONIC_RESONANCE_ENERGY/99
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
    assert "missing data card" in log.errors[0]
    assert 99 not in model.eng_electrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonic_resonance_energies


def test_m474_eng_electrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonic_resonance_energy_cascade():
    # Test fallback from preceding polaritonic directive (hedgehog)
    eng1 = EngElectrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonicResonanceEnergy(dt_etfchedgehogplp=0.0065)
    assert pytest.approx(eng1.dt_etfcskyrmioniumplp) == 0.0065

    # Test forward propagation from skyrmionium to other polaritonic directives
    eng2 = EngElectrothermoflexomagnetochiralskyrmioniumplasmonicpolaritonicResonanceEnergy(dt_etfcskyrmioniumplp=0.0078)
    assert pytest.approx(eng2.dt_etfplp) == 0.0078
    assert pytest.approx(eng2.dt_etfcblochpointplp) == 0.0078
    assert pytest.approx(eng2.dt_etfcbobberplp) == 0.0078
    assert pytest.approx(eng2.dt_etfchedgehogplp) == 0.0078


# ============================================================================
# 3. LAGMUL_HIRZEBRUCH_SPINOR_SPATIAL_LINKAGE_JOINT Tests
# ============================================================================

def test_m474_lagmul_hirzebruch_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{111:>10d}{112:>10d}{113:>10d}{2.8e6:>20.4f}{6:>10d}{1.8e-5:>20.4e}"
    c2 = f"{16.5:>20.4f}{26.5:>20.4f}{48.0:>20.4f}{6.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M474 Hirzebruch Spinor Spatial Linkage Joint Fixed
/HIRZEBRUCH_SPINOR_SPATIAL_LINKAGE_JOINT/1
Hirzebruch Spinor Joint Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_hirzebruch_spinor_spatial_linkage_joints
    j = model.lagmul_hirzebruch_spinor_spatial_linkage_joints[1]
    assert isinstance(j, LagmulHirzebruchSpinorSpatialLinkageJoint)
    assert j.node1 == 111
    assert j.node2 == 112
    assert j.node3 == 113
    assert pytest.approx(j.stiff) == 2.8e6
    assert j.skew_id == 6
    assert pytest.approx(j.tol) == 1.8e-5
    assert pytest.approx(j.link_len_a) == 16.5
    assert pytest.approx(j.link_len_b) == 26.5
    assert pytest.approx(j.twist_angle_alpha) == 48.0
    assert pytest.approx(j.offset_distance_s) == 6.5

    # Offset distance properties
    assert pytest.approx(j.offset_distance_r) == 6.5
    assert pytest.approx(j.offset_distance_v) == 6.5
    assert pytest.approx(j.offset_distance_h) == 6.5
    assert pytest.approx(j.offset_distance_u) == 6.5
    assert pytest.approx(j.offset_distance_f) == 6.5


def test_m474_lagmul_hirzebruch_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M474 Hirzebruch Spinor Spatial Linkage Joint Free
/LAGMUL/HIRZEBRUCH_SPINOR_SPATIAL_LINKAGE_JOINT/2
211, 212, 213, 3.5e6, 7, 2.2e-5
19.0, 29.0, 65.0, 8.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_hirzebruch_spinor_spatial_linkage_joints
    j = model.lagmul_hirzebruch_spinor_spatial_linkage_joints[2]
    assert j.node1 == 211
    assert j.node2 == 212
    assert j.node3 == 213
    assert pytest.approx(j.stiff) == 3.5e6
    assert j.skew_id == 7
    assert pytest.approx(j.tol) == 2.2e-5
    assert pytest.approx(j.link_len_a) == 19.0
    assert pytest.approx(j.link_len_b) == 29.0
    assert pytest.approx(j.twist_angle_alpha) == 65.0
    assert pytest.approx(j.offset_distance_s) == 8.0


def test_m474_lagmul_hirzebruch_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    aliases = [
        "/HIRZEBRUCH_SPINOR_SPATIAL_LINKAGE",
        "/LAGMUL/HIRZEBRUCH_SPINOR_SPATIAL_LINKAGE",
        "/HIRZEBRUCH_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM",
        "/HIRZEBRUCH_SPINOR_SPATIAL_SYMMETRIC_MECHANISM",
        "/HIRZEBRUCH_SPINOR_SPATIAL_6R_MECHANISM",
        "/HIRZEBRUCH_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM",
        "/LAGMUL/HIRZEBRUCH_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/HIRZEBRUCH_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/HIRZEBRUCH_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/HIRZEBRUCH_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/HIRZEBRUCH_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/HIRZEBRUCH_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/HIRZEBRUCH_RIEMANN_ROCH_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/HIRZEBRUCH_RIEMANN_ROCH_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/HIRZEBRUCH_RIEMANN_ROCH_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/HIRZEBRUCH_RIEMANN_ROCH_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/HIRZEBRUCH_SIGNATURE_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/HIRZEBRUCH_SIGNATURE_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/HIRZEBRUCH_SIGNATURE_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/HIRZEBRUCH_SIGNATURE_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/HIRZEBRUCH_SURFACE_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/HIRZEBRUCH_SURFACE_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/HIRZEBRUCH_SURFACE_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/HIRZEBRUCH_SURFACE_TWISTOR_SPATIAL_LINKAGE_JOINT",
    ]
    deck_lines = ["/BEGIN", "Test M474 Hirzebruch Joint Aliases"]
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
        assert idx in model.lagmul_hirzebruch_spinor_spatial_linkage_joints
        j = model.lagmul_hirzebruch_spinor_spatial_linkage_joints[idx]
        assert j.node1 == idx * 10
        assert j.node2 == idx * 10 + 1
        assert j.node3 == idx * 10 + 2


def test_m474_lagmul_hirzebruch_spinor_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """/BEGIN
Test Joint Missing Card
/HIRZEBRUCH_SPINOR_SPATIAL_LINKAGE_JOINT/99
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
    assert "missing data card 1" in log.errors[0]
    assert 99 not in model.lagmul_hirzebruch_spinor_spatial_linkage_joints


def test_m474_lagmul_hirzebruch_spinor_spatial_linkage_joint_properties():
    j = LagmulHirzebruchSpinorSpatial_Linkage_Joint = LagmulHirzebruchSpinorSpatialLinkageJoint(id=8)
    j.offset_distance_r = 13.5
    assert pytest.approx(j.offset_distance_s) == 13.5

    j.offset_distance_v = 15.5
    assert pytest.approx(j.offset_distance_s) == 15.5

    j.offset_distance_h = 17.5
    assert pytest.approx(j.offset_distance_s) == 17.5

    j.offset_distance_u = 19.5
    assert pytest.approx(j.offset_distance_s) == 19.5

    j.offset_distance_f = 21.5
    assert pytest.approx(j.offset_distance_s) == 21.5


# ============================================================================
# 4. SENSOR_SPRING_TRANSVERSE_CRACKLE_RATE Tests
# ============================================================================

def test_m474_sensor_spring_transverse_crackle_rate(tmp_path: Path):
    # Fixed format
    c1 = f"{65:>10d}{7.5e5:>20.4e}{0.0025:>20.4e}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M474 Sensor Spring Transverse Crackle Rate
/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE/1
{c1}
/SENSOR/SPRING_TRANS_CRACKLE_RATE/2
66, 8.5e5, 0.0035
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_crackle_rates
    s1 = model.sensor_spring_transverse_crackle_rates[1]
    assert isinstance(s1, SensorSpringTransverseCrackleRate)
    assert s1.spring_id == 65
    assert pytest.approx(s1.jtrans_pop_max) == 7.5e5
    assert pytest.approx(s1.t_delay) == 0.0025

    # M474 25th rate-of-change property accessors
    assert pytest.approx(s1.jtrans_crackle_max) == 7.5e5
    assert pytest.approx(s1.j_trans_crk_max) == 7.5e5
    assert pytest.approx(s1.j_trans_crackle_max) == 7.5e5
    assert pytest.approx(s1.j_transverse_crk_max) == 7.5e5
    assert pytest.approx(s1.j_transverse_crackle_max) == 7.5e5
    assert pytest.approx(s1.j_shear_crk_max) == 7.5e5
    assert pytest.approx(s1.j_shear_crackle_max) == 7.5e5
    assert pytest.approx(s1.j_crk_trans_max) == 7.5e5
    assert pytest.approx(s1.j_crackle_trans_max) == 7.5e5
    assert pytest.approx(s1.j_crk_shear_max) == 7.5e5
    assert pytest.approx(s1.j_crackle_shear_max) == 7.5e5
    assert pytest.approx(s1.jtrans_crackle_rate_max) == 7.5e5
    assert pytest.approx(s1.jtrans_snap_rate_max) == 7.5e5
    assert pytest.approx(s1.jtrans_pop_rate_max) == 7.5e5
    assert pytest.approx(s1.jtrans_lock_rate_max) == 7.5e5

    # Free format
    assert 2 in model.sensor_spring_transverse_crackle_rates
    s2 = model.sensor_spring_transverse_crackle_rates[2]
    assert s2.spring_id == 66
    assert pytest.approx(s2.jtrans_pop_max) == 8.5e5
    assert pytest.approx(s2.t_delay) == 0.0035

    # Setters
    s1.jtrans_crackle_max = 9.8e5
    assert pytest.approx(s1.jtrans_pop_max) == 9.8e5
    s1.j_shear_crk_max = 1.05e6
    assert pytest.approx(s1.jtrans_pop_max) == 1.05e6
    s1.j_transverse_crackle_max = 1.2e6
    assert pytest.approx(s1.jtrans_pop_max) == 1.2e6


def test_m474_sensor_spring_transverse_crackle_rate_aliases(tmp_path: Path):
    aliases = [
        "/SENSOR/SPRING_TRANS_CRACKLE_RATE",
        "/SENSOR/SPRING_TRANSVERSE_CRK_RATE",
        "/SENSOR/SPRING_TRANS_CRK_RATE",
        "/SENSOR/SPRING_RATE_TRANSVERSE_CRACKLE",
        "/SENSOR/SPRING_RATE_TRANS_CRACKLE",
        "/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE_SENSOR",
        "/SENSOR/SPRING_TRANS_CRACKLE_RATE_SENSOR",
        "/SENSOR/TRANSVERSE_CRACKLE_RATE_SPRING_SENSOR",
        "/SENSOR/TRANS_CRACKLE_RATE_SPRING_SENSOR",
        "/SENSOR/SPRING_CRACKLE_RATE_TRANSVERSE_SENSOR",
        "/SENSOR/SPRING_CRACKLE_RATE_TRANS_SENSOR",
        "/SENSOR/SPRING_RATE_CRACKLE_TRANSVERSE",
        "/SENSOR/SPRING_RATE_CRACKLE_TRANS",
        "/SENSOR/SPRING_RATE_TRANSVERSE_CRK",
        "/SENSOR/SPRING_RATE_TRANS_CRK",
        "/SENSOR/SPRING_CRACKLE_TRANSVERSE",
        "/SENSOR/SPRING_CRACKLE_TRANS",
        "/SENSOR/SPRING_SNAP_RATE_TRANSVERSE_CRACKLE",
        "/SENSOR/SPRING_SNAP_RATE_TRANS_CRACKLE",
        "/SENSOR/SPRING_LOCK_RATE_TRANSVERSE_CRACKLE",
        "/SENSOR/SPRING_LOCK_RATE_TRANS_CRACKLE",
        "/SENSOR/SPRING_SHEAR_CRACKLE_RATE",
        "/SENSOR/SPRING_RATE_SHEAR_CRACKLE",
        "/SENSOR/SPRING_SHEAR_CRACKLE_RATE_SENSOR",
        "/SENSOR/SHEAR_CRACKLE_RATE_SPRING_SENSOR",
        "/SENSOR/SPRING_CRACKLE_RATE_SHEAR_SENSOR",
        "/SENSOR/SPRING_RATE_CRACKLE_SHEAR",
        "/SENSOR/SPRING_RATE_SHEAR_CRK",
        "/SENSOR/SPRING_CRACKLE_SHEAR",
        "/SENSOR/SPRING_SNAP_RATE_SHEAR_CRACKLE",
        "/SENSOR/SPRING_LOCK_RATE_SHEAR_CRACKLE",
    ]
    deck_lines = ["/BEGIN", "Test M474 Sensor Aliases"]
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
        assert idx in model.sensor_spring_transverse_crackle_rates
        s = model.sensor_spring_transverse_crackle_rates[idx]
        assert s.spring_id == idx + 100
        assert pytest.approx(s.jtrans_pop_max) == idx * 1e4

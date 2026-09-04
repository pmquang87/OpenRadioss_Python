"""Tests for Milestone M473:
- /FAIL/LAD_DYNAMIC_FIBER_KINKING_FAILURE_RATE (and aliases)
- /ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHEDGEHOGPLASMONICPOLARITONIC_RESONANCE_ENERGY (and aliases)
- /LAGMUL/ATIYAH_SPINOR_SPATIAL_LINKAGE_JOINT / /ATIYAH_SPINOR_SPATIAL_LINKAGE_JOINT (and aliases)
- /SENSOR/SPRING_NORMAL_CRACKLE_RATE (and aliases)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailLadDynamicFiberKinkingFailureRate,
    EngElectrothermoflexomagnetochiralhedgehogplasmonicpolaritonicResonanceEnergy,
    LagmulAtiyahSpinorSpatialLinkageJoint,
    SensorSpringNormalCrackleRate,
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
# 1. FAIL_LAD_DYNAMIC_FIBER_KINKING_FAILURE_RATE Tests
# ============================================================================

def test_m473_fail_lad_dynamic_fiber_kinking_failure_rate_fixed(tmp_path: Path):
    c1 = f"{125.0:>20.4f}{415.0:>20.4f}{0.31:>20.4f}{1.52:>20.4f}{0.98:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M473 Fail Lad Dynamic Fiber Kinking Failure Rate Fixed
2022 0
/FAIL/LAD_DYNAMIC_FIBER_KINKING_FAILURE_RATE/1
Dynamic Fiber Kinking Failure Rate Card Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.fail_laddynamicfiberkinkingfailurerates
    fm = model.fail_laddynamicfiberkinkingfailurerates[1]
    assert isinstance(fm, FailLadDynamicFiberKinkingFailureRate)
    assert fm.mat_id == 1
    assert pytest.approx(fm.sigma_dfkfr0) == 125.0
    assert pytest.approx(fm.sigma_dfkfrc) == 415.0
    assert pytest.approx(fm.gamma_dfkfr) == 0.31
    assert pytest.approx(fm.p_dfkfr) == 1.52
    assert pytest.approx(fm.d_dfkfr_max) == 0.98
    assert fm.ifail_sh == 2
    assert fm.ifail_so == 1

    # Properties
    assert pytest.approx(fm.sigma_dfkf0) == 125.0
    assert pytest.approx(fm.sigma_dfkfc) == 415.0
    assert pytest.approx(fm.gamma_dfkf) == 0.31
    assert pytest.approx(fm.p_dfkf) == 1.52
    assert pytest.approx(fm.d_dfkf_max) == 0.98

    assert pytest.approx(fm.sigma_dfk0) == 125.0
    assert pytest.approx(fm.sigma_dfkc) == 415.0
    assert pytest.approx(fm.gamma_dfk) == 0.31
    assert pytest.approx(fm.p_dfk) == 1.52
    assert pytest.approx(fm.d_dfk_max) == 0.98


def test_m473_fail_lad_dynamic_fiber_kinking_failure_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M473 Fail Lad Dynamic Fiber Kinking Failure Rate Free
/FAIL/LAD_DYNAMIC_FIBER_KINKING_FAILURE_RATE/2
135.0, 440.0, 0.36, 1.62, 0.99
1, 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.fail_laddynamicfiberkinkingfailurerates
    fm = model.fail_laddynamicfiberkinkingfailurerates[2]
    assert pytest.approx(fm.sigma_dfkfr0) == 135.0
    assert pytest.approx(fm.sigma_dfkfrc) == 440.0
    assert pytest.approx(fm.gamma_dfkfr) == 0.36
    assert pytest.approx(fm.p_dfkfr) == 1.62
    assert pytest.approx(fm.d_dfkfr_max) == 0.99
    assert fm.ifail_sh == 1
    assert fm.ifail_so == 2


def test_m473_fail_lad_dynamic_fiber_kinking_failure_rate_aliases(tmp_path: Path):
    aliases = [
        "/FAIL/LADEVEZE_DYNAMIC_FIBER_KINKING_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_FIBER_KINKING_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_FIBER_KINKING_FAILURE",
        "/FAIL/LAD_DYNAMIC_FIBER_KINK_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_FIBER_KINK_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_FIBER_KINK_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_FIBER_KINK_FAILURE",
        "/FAIL/LAD_DYNAMIC_FIBER_COMPRESSIVE_KINKING_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_FIBER_COMPRESSIVE_KINKING_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_FIBER_COMPRESSIVE_KINKING_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_FIBER_COMPRESSIVE_KINKING_FAILURE",
        "/FAIL/LAD_DYNAMIC_FIBER_MICROBUCKLING_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_FIBER_MICROBUCKLING_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_FIBER_MICROBUCKLING_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_FIBER_MICROBUCKLING_FAILURE",
        "/FAIL/LAD_DFKFR",
        "/FAIL/LAD_DFKFR_MODEL",
        "/FAIL/LAD_DFKFR_LAW",
        "/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_FIBER_KINKING_FAILURE",
        "/FAIL/LAD_DYNAMIC_FIBER_KINKING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_FIBER_KINKING_DAMAGE_RATE",
        "/FAIL/LAD_DYNAMIC_FIBER_KINK_DAMAGE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_FIBER_KINK_DAMAGE_RATE",
        "/FAIL/LAD_DYNAMIC_FIBER_MICROBUCKLING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_FIBER_MICROBUCKLING_DAMAGE_RATE",
    ]
    deck_lines = ["/BEGIN", "Test M473 Fail Lad Dynamic Fiber Kinking Failure Rate Aliases"]
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
        assert idx in model.fail_laddynamicfiberkinkingfailurerates
        fm = model.fail_laddynamicfiberkinkingfailurerates[idx]
        assert pytest.approx(fm.sigma_dfkfr0) == float(idx) * 10.0
        assert pytest.approx(fm.sigma_dfkfrc) == float(idx) * 30.0


def test_m473_fail_lad_dynamic_fiber_kinking_failure_rate_missing_card(tmp_path: Path):
    deck = """/BEGIN
Test M473 Fail Missing Card
/FAIL/LAD_DYNAMIC_FIBER_KINKING_FAILURE_RATE/99
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
    assert "missing data card" in log.errors[0]
    assert 99 not in model.fail_laddynamicfiberkinkingfailurerates


def test_m473_fail_lad_dynamic_fiber_kinking_failure_rate_properties(tmp_path: Path):
    fm = FailLadDynamicFiberKinkingFailureRate(mat_id=5)
    fm.sigma_dfkf0 = 145.0
    fm.sigma_dfkfc = 480.0
    fm.gamma_dfkf = 0.42
    fm.p_dfkf = 1.75
    fm.d_dfkf_max = 0.995

    assert pytest.approx(fm.sigma_dfkfr0) == 145.0
    assert pytest.approx(fm.sigma_dfkfrc) == 480.0
    assert pytest.approx(fm.gamma_dfkfr) == 0.42
    assert pytest.approx(fm.p_dfkfr) == 1.75
    assert pytest.approx(fm.d_dfkfr_max) == 0.995

    fm.sigma_dfk0 = 155.0
    fm.sigma_dfkc = 510.0
    fm.gamma_dfk = 0.45
    fm.p_dfk = 1.80
    fm.d_dfk_max = 0.996

    assert pytest.approx(fm.sigma_dfkfr0) == 155.0
    assert pytest.approx(fm.sigma_dfkfrc) == 510.0
    assert pytest.approx(fm.gamma_dfkfr) == 0.45
    assert pytest.approx(fm.p_dfkfr) == 1.80
    assert pytest.approx(fm.d_dfkfr_max) == 0.996


# ============================================================================
# 2. ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALHEDGEHOGPLASMONICPOLARITONIC_RESONANCE_ENERGY Tests
# ============================================================================

def test_m473_eng_electrothermoflexomagnetochiralhedgehogplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0035:>20.4f}{12:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M473 Eng Chiral Hedgehog Resonance Energy Fixed
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHEDGEHOGPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Eng Chiral Hedgehog Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralhedgehogplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralhedgehogplasmonicpolaritonic_resonance_energies[1]
    assert isinstance(eng, EngElectrothermoflexomagnetochiralhedgehogplasmonicpolaritonicResonanceEnergy)
    assert pytest.approx(eng.dt_etfchedgehogplp) == 0.0035
    assert eng.sens_id == 12


def test_m473_eng_electrothermoflexomagnetochiralhedgehogplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M473 Eng Chiral Hedgehog Resonance Energy Free
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHEDGEHOGPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.0045, 14
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralhedgehogplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralhedgehogplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfchedgehogplp) == 0.0045
    assert eng.sens_id == 14


def test_m473_eng_electrothermoflexomagnetochiralhedgehogplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    aliases = [
        "/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_HEDGEHOG_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALHEDGEHOGPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHEDGEHOGPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALHEDGEHOGPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOHEDGEHOGCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_HEDGEHOG_CHIRAL_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOHEDGEHOGCHIRALPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOHEDGEHOGCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOHEDGEHOGCHIRALPLASMONICPOLARITONIC_RESONANCE",
    ]
    deck_lines = ["/BEGIN", "Test M473 Eng Aliases"]
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
        assert idx in model.eng_electrothermoflexomagnetochiralhedgehogplasmonicpolaritonic_resonance_energies
        eng = model.eng_electrothermoflexomagnetochiralhedgehogplasmonicpolaritonic_resonance_energies[idx]
        assert pytest.approx(eng.dt_etfchedgehogplp) == 0.001 * idx
        assert eng.sens_id == idx


def test_m473_eng_electrothermoflexomagnetochiralhedgehogplasmonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """/BEGIN
Test Eng Missing Card
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHEDGEHOGPLASMONICPOLARITONIC_RESONANCE_ENERGY/99
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
    assert "missing data card" in log.errors[0]
    assert 99 not in model.eng_electrothermoflexomagnetochiralhedgehogplasmonicpolaritonic_resonance_energies


def test_m473_eng_electrothermoflexomagnetochiralhedgehogplasmonicpolaritonic_resonance_energy_cascade():
    # Test fallback from preceding polaritonic directive
    eng1 = EngElectrothermoflexomagnetochiralhedgehogplasmonicpolaritonicResonanceEnergy(dt_etfcblochpointplp=0.0062)
    assert pytest.approx(eng1.dt_etfchedgehogplp) == 0.0062

    # Test forward propagation from hedgehog to other polaritonic directives
    eng2 = EngElectrothermoflexomagnetochiralhedgehogplasmonicpolaritonicResonanceEnergy(dt_etfchedgehogplp=0.0075)
    assert pytest.approx(eng2.dt_etfplp) == 0.0075
    assert pytest.approx(eng2.dt_etfcblochpointplp) == 0.0075
    assert pytest.approx(eng2.dt_etfcbobberplp) == 0.0075


# ============================================================================
# 3. LAGMUL_ATIYAH_SPINOR_SPATIAL_LINKAGE_JOINT Tests
# ============================================================================

def test_m473_lagmul_atiyah_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{2.5e6:>20.4f}{4:>10d}{1.5e-5:>20.4e}"
    c2 = f"{15.5:>20.4f}{25.5:>20.4f}{45.0:>20.4f}{5.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M473 Atiyah Spinor Spatial Linkage Joint Fixed
/ATIYAH_SPINOR_SPATIAL_LINKAGE_JOINT/1
Atiyah Spinor Joint Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_atiyah_spinor_spatial_linkage_joints
    j = model.lagmul_atiyah_spinor_spatial_linkage_joints[1]
    assert isinstance(j, LagmulAtiyahSpinorSpatialLinkageJoint)
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 2.5e6
    assert j.skew_id == 4
    assert pytest.approx(j.tol) == 1.5e-5
    assert pytest.approx(j.link_len_a) == 15.5
    assert pytest.approx(j.link_len_b) == 25.5
    assert pytest.approx(j.twist_angle_alpha) == 45.0
    assert pytest.approx(j.offset_distance_s) == 5.5

    # Offset distance properties
    assert pytest.approx(j.offset_distance_r) == 5.5
    assert pytest.approx(j.offset_distance_v) == 5.5
    assert pytest.approx(j.offset_distance_h) == 5.5
    assert pytest.approx(j.offset_distance_u) == 5.5
    assert pytest.approx(j.offset_distance_f) == 5.5


def test_m473_lagmul_atiyah_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M473 Atiyah Spinor Spatial Linkage Joint Free
/LAGMUL/ATIYAH_SPINOR_SPATIAL_LINKAGE_JOINT/2
201, 202, 203, 3.2e6, 5, 2.0e-5
18.0, 28.0, 60.0, 7.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_atiyah_spinor_spatial_linkage_joints
    j = model.lagmul_atiyah_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 3.2e6
    assert j.skew_id == 5
    assert pytest.approx(j.tol) == 2.0e-5
    assert pytest.approx(j.link_len_a) == 18.0
    assert pytest.approx(j.link_len_b) == 28.0
    assert pytest.approx(j.twist_angle_alpha) == 60.0
    assert pytest.approx(j.offset_distance_s) == 7.0


def test_m473_lagmul_atiyah_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    aliases = [
        "/LAGMUL/ATIYAH_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/ATIYAH_SPINOR_SPATIAL_LINKAGE",
        "/LAGMUL/ATIYAH_SPINOR_SPATIAL_LINKAGE",
        "/ATIYAH_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM",
        "/ATIYAH_SPINOR_SPATIAL_SYMMETRIC_MECHANISM",
        "/ATIYAH_SPINOR_SPATIAL_6R_MECHANISM",
        "/ATIYAH_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM",
        "/LAGMUL/ATIYAH_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/ATIYAH_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/ATIYAH_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/ATIYAH_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/ATIYAH_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/ATIYAH_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/ATIYAH_SINGER_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/ATIYAH_SINGER_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/ATIYAH_SINGER_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/ATIYAH_SINGER_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/ATIYAH_PATODI_SINGER_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/ATIYAH_PATODI_SINGER_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/ATIYAH_PATODI_SINGER_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/ATIYAH_PATODI_SINGER_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/ATIYAH_HIRZEBRUCH_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/ATIYAH_HIRZEBRUCH_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/ATIYAH_HIRZEBRUCH_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/ATIYAH_HIRZEBRUCH_TWISTOR_SPATIAL_LINKAGE_JOINT",
    ]
    deck_lines = ["/BEGIN", "Test M473 Atiyah Joint Aliases"]
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
        assert idx in model.lagmul_atiyah_spinor_spatial_linkage_joints
        j = model.lagmul_atiyah_spinor_spatial_linkage_joints[idx]
        assert j.node1 == idx * 10
        assert j.node2 == idx * 10 + 1
        assert j.node3 == idx * 10 + 2


def test_m473_lagmul_atiyah_spinor_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """/BEGIN
Test Joint Missing Card
/ATIYAH_SPINOR_SPATIAL_LINKAGE_JOINT/99
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
    assert "missing data card 1" in log.errors[0]
    assert 99 not in model.lagmul_atiyah_spinor_spatial_linkage_joints


def test_m473_lagmul_atiyah_spinor_spatial_linkage_joint_properties():
    j = LagmulAtiyahSpinorSpatialLinkageJoint(id=7)
    j.offset_distance_r = 12.5
    assert pytest.approx(j.offset_distance_s) == 12.5

    j.offset_distance_v = 14.5
    assert pytest.approx(j.offset_distance_s) == 14.5

    j.offset_distance_h = 16.5
    assert pytest.approx(j.offset_distance_s) == 16.5

    j.offset_distance_u = 18.5
    assert pytest.approx(j.offset_distance_s) == 18.5

    j.offset_distance_f = 20.5
    assert pytest.approx(j.offset_distance_s) == 20.5


# ============================================================================
# 4. SENSOR_SPRING_NORMAL_CRACKLE_RATE Tests
# ============================================================================

def test_m473_sensor_spring_normal_crackle_rate(tmp_path: Path):
    # Fixed format
    c1 = f"{55:>10d}{6.5e5:>20.4e}{0.002:>20.4e}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M473 Sensor Spring Normal Crackle Rate
/SENSOR/SPRING_NORMAL_CRACKLE_RATE/1
{c1}
/SENSOR/SPRING_NORM_CRACKLE_RATE/2
56, 7.5e5, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_crackle_rates
    s1 = model.sensor_spring_normal_crackle_rates[1]
    assert isinstance(s1, SensorSpringNormalCrackleRate)
    assert s1.spring_id == 55
    assert pytest.approx(s1.jnorm_crk_max) == 6.5e5
    assert pytest.approx(s1.t_delay) == 0.002

    # M473 property accessors
    assert pytest.approx(s1.jnorm_crackle_max) == 6.5e5
    assert pytest.approx(s1.j_norm_crk_max) == 6.5e5
    assert pytest.approx(s1.j_norm_crackle_max) == 6.5e5
    assert pytest.approx(s1.j_normal_crk_max) == 6.5e5
    assert pytest.approx(s1.j_normal_crackle_max) == 6.5e5
    assert pytest.approx(s1.j_crk_norm_max) == 6.5e5
    assert pytest.approx(s1.j_crackle_norm_max) == 6.5e5
    assert pytest.approx(s1.j_axial_crk_max) == 6.5e5
    assert pytest.approx(s1.j_axial_crackle_max) == 6.5e5
    assert pytest.approx(s1.jnorm_crackle_rate_max) == 6.5e5

    # Free format
    assert 2 in model.sensor_spring_normal_crackle_rates
    s2 = model.sensor_spring_normal_crackle_rates[2]
    assert s2.spring_id == 56
    assert pytest.approx(s2.jnorm_crk_max) == 7.5e5
    assert pytest.approx(s2.t_delay) == 0.003

    # Setters
    s1.jnorm_crackle_max = 8.8e5
    assert pytest.approx(s1.jnorm_crk_max) == 8.8e5
    s1.j_norm_crk_max = 9.9e5
    assert pytest.approx(s1.jnorm_crk_max) == 9.9e5
    s1.j_axial_crackle_max = 1.1e6
    assert pytest.approx(s1.jnorm_crk_max) == 1.1e6


def test_m473_sensor_spring_normal_crackle_rate_aliases(tmp_path: Path):
    aliases = [
        "/SENSOR/SPRING_RATE_NORMAL_CRACKLE",
        "/SENSOR/SPRING_RATE_NORM_CRACKLE",
        "/SENSOR/SPRING_NORMAL_CRACKLE_RATE_SENSOR",
        "/SENSOR/SPRING_NORM_CRACKLE_RATE_SENSOR",
        "/SENSOR/NORMAL_CRACKLE_RATE_SPRING_SENSOR",
        "/SENSOR/NORM_CRACKLE_RATE_SPRING_SENSOR",
        "/SENSOR/SPRING_CRACKLE_RATE_NORMAL_SENSOR",
        "/SENSOR/SPRING_CRACKLE_RATE_NORM_SENSOR",
        "/SENSOR/SPRING_RATE_CRACKLE_NORMAL",
        "/SENSOR/SPRING_RATE_CRACKLE_NORM",
        "/SENSOR/SPRING_RATE_NORMAL_CRK",
        "/SENSOR/SPRING_RATE_NORM_CRK",
        "/SENSOR/SPRING_CRACKLE_NORMAL",
        "/SENSOR/SPRING_CRACKLE_NORM",
        "/SENSOR/SPRING_SNAP_RATE_NORMAL_CRACKLE",
        "/SENSOR/SPRING_SNAP_RATE_NORM_CRACKLE",
        "/SENSOR/SPRING_LOCK_RATE_NORMAL_CRACKLE",
        "/SENSOR/SPRING_LOCK_RATE_NORM_CRACKLE",
    ]
    deck_lines = ["/BEGIN", "Test M473 Sensor Aliases"]
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
        assert idx in model.sensor_spring_normal_crackle_rates
        s = model.sensor_spring_normal_crackle_rates[idx]
        assert s.spring_id == idx + 100
        assert pytest.approx(s.jnorm_crk_max) == idx * 1e4

"""Tests for Milestone M472:
- /FAIL/LAD_COUPLED_MATRIX_CRUSHING_FAILURE_RATE (and aliases)
- /ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBLOCHPOINTPLASMONICPOLARITONIC_RESONANCE_ENERGY (and aliases)
- /LAGMUL/CHERN_SPINOR_SPATIAL_LINKAGE_JOINT / /CHERN_SPINOR_SPATIAL_LINKAGE_JOINT (and aliases)
- /SENSOR/SPRING_TOTAL_SNAP_RATE (and aliases)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailLadCoupledMatrixCrushingFailureRate,
    EngElectrothermoflexomagnetochiralblochpointplasmonicpolaritonicResonanceEnergy,
    LagmulChernSpinorSpatialLinkageJoint,
    SensorSpringTotalSnapRate,
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
# 1. FAIL_LAD_COUPLED_MATRIX_CRUSHING_FAILURE_RATE Tests
# ============================================================================

def test_m472_fail_lad_coupled_matrix_crushing_failure_rate_fixed(tmp_path: Path):
    c1 = f"{115.0:>20.4f}{395.0:>20.4f}{0.29:>20.4f}{1.48:>20.4f}{0.97:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M472 Fail Lad Coupled Matrix Crushing Failure Rate Fixed
2022 0
/FAIL/LAD_COUPLED_MATRIX_CRUSHING_FAILURE_RATE/1
Coupled Matrix Crushing Failure Rate Card Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.fail_ladcoupledmatrixcrushingfailurerates
    fm = model.fail_ladcoupledmatrixcrushingfailurerates[1]
    assert isinstance(fm, FailLadCoupledMatrixCrushingFailureRate)
    assert fm.mat_id == 1
    assert pytest.approx(fm.sigma_cmcrfr0) == 115.0
    assert pytest.approx(fm.sigma_cmcrfrc) == 395.0
    assert pytest.approx(fm.gamma_cmcrfr) == 0.29
    assert pytest.approx(fm.p_cmcrfr) == 1.48
    assert pytest.approx(fm.d_cmcrfr_max) == 0.97
    assert fm.ifail_sh == 2
    assert fm.ifail_so == 1

    # Properties
    assert pytest.approx(fm.sigma_cmcrf0) == 115.0
    assert pytest.approx(fm.sigma_cmcrfc) == 395.0
    assert pytest.approx(fm.gamma_cmcrf) == 0.29
    assert pytest.approx(fm.p_cmcrf) == 1.48
    assert pytest.approx(fm.d_cmcrf_max) == 0.97

    assert pytest.approx(fm.sigma_cmcr0) == 115.0
    assert pytest.approx(fm.sigma_cmcrc) == 395.0
    assert pytest.approx(fm.gamma_cmcr) == 0.29
    assert pytest.approx(fm.p_cmcr) == 1.48
    assert pytest.approx(fm.d_cmcr_max) == 0.97

    assert pytest.approx(fm.sigma_cmc0) == 115.0
    assert pytest.approx(fm.sigma_cmcc) == 395.0
    assert pytest.approx(fm.gamma_cmc) == 0.29
    assert pytest.approx(fm.p_cmc) == 1.48
    assert pytest.approx(fm.d_cmc_max) == 0.97


def test_m472_fail_lad_coupled_matrix_crushing_failure_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M472 Fail Lad Coupled Matrix Crushing Failure Rate Free
/FAIL/LAD_COUPLED_MATRIX_CRUSHING_FAILURE_RATE/2
130.0, 420.0, 0.34, 1.58, 0.985
1, 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.fail_ladcoupledmatrixcrushingfailurerates
    fm = model.fail_ladcoupledmatrixcrushingfailurerates[2]
    assert pytest.approx(fm.sigma_cmcrfr0) == 130.0
    assert pytest.approx(fm.sigma_cmcrfrc) == 420.0
    assert pytest.approx(fm.gamma_cmcrfr) == 0.34
    assert pytest.approx(fm.p_cmcrfr) == 1.58
    assert pytest.approx(fm.d_cmcrfr_max) == 0.985
    assert fm.ifail_sh == 1
    assert fm.ifail_so == 2


def test_m472_fail_lad_coupled_matrix_crushing_failure_rate_aliases(tmp_path: Path):
    aliases = [
        "/FAIL/LAD_COUPLED_MATRIX_CRUSHING_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_CRUSHING_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_MATRIX_CRUSHING_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_CRUSHING_FAILURE",
        "/FAIL/LAD_COUPLED_MATRIX_CRUSH_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_CRUSH_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_MATRIX_CRUSH_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_CRUSH_FAILURE",
        "/FAIL/LAD_COUPLE_MATRIX_CRUSHING_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLE_MATRIX_CRUSHING_FAILURE_RATE",
        "/FAIL/LAD_COUPLE_MATRIX_CRUSHING_FAILURE",
        "/FAIL/LADEVEZE_COUPLE_MATRIX_CRUSHING_FAILURE",
        "/FAIL/LAD_COUPLE_MATRIX_CRUSH_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLE_MATRIX_CRUSH_FAILURE_RATE",
        "/FAIL/LAD_COUPLE_MATRIX_CRUSH_FAILURE",
        "/FAIL/LADEVEZE_COUPLE_MATRIX_CRUSH_FAILURE",
        "/FAIL/LAD_COUPLED_MATRIX_MICRO_CRUSHING_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_MICRO_CRUSHING_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_MATRIX_MICRO_CRUSHING_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_MICRO_CRUSHING_FAILURE",
        "/FAIL/LAD_COUPLED_MATRIX_MICROCRUSHING_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_MICROCRUSHING_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_MATRIX_MICROCRUSHING_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_MICROCRUSHING_FAILURE",
        "/FAIL/LAD_CMCRFR",
        "/FAIL/LAD_CMCRFR_MODEL",
        "/FAIL/LAD_CMCRFR_LAW",
        "/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_MATRIX_CRUSHING_FAILURE",
        "/FAIL/LAD_COUPLED_MATRIX_CRUSHING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_CRUSHING_DAMAGE_RATE",
        "/FAIL/LAD_COUPLED_MATRIX_CRUSH_DAMAGE_RATE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_CRUSH_DAMAGE_RATE",
        "/FAIL/LAD_COUPLED_MATRIX_MICRO_CRUSHING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_MICRO_CRUSHING_DAMAGE_RATE",
        "/FAIL/LAD_COUPLED_MATRIX_MICROCRUSHING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_COUPLED_MATRIX_MICROCRUSHING_DAMAGE_RATE",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test M472 Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "98.0, 350.0, 0.24, 1.38, 0.955",
            "1, 1",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.fail_ladcoupledmatrixcrushingfailurerates
        fm = model.fail_ladcoupledmatrixcrushingfailurerates[idx]
        assert pytest.approx(fm.sigma_cmcrfr0) == 98.0
        assert pytest.approx(fm.sigma_cmcrfrc) == 350.0


def test_m472_fail_lad_coupled_matrix_crushing_failure_rate_properties():
    fm = FailLadCoupledMatrixCrushingFailureRate(
        mat_id=1,
        sigma_cmcrfr0=105.0,
        sigma_cmcrfrc=360.0,
        gamma_cmcrfr=0.26,
        p_cmcrfr=1.52,
        d_cmcrfr_max=0.99,
    )
    # Test property setters
    fm.sigma_cmcrf0 = 110.0
    assert fm.sigma_cmcrfr0 == 110.0
    fm.sigma_cmcrfc = 370.0
    assert fm.sigma_cmcrfrc == 370.0
    fm.gamma_cmcrf = 0.27
    assert fm.gamma_cmcrfr == 0.27
    fm.p_cmcrf = 1.62
    assert fm.p_cmcrfr == 1.62
    fm.d_cmcrf_max = 0.98
    assert fm.d_cmcrfr_max == 0.98

    fm.sigma_cmcr0 = 115.0
    assert fm.sigma_cmcrfr0 == 115.0
    fm.sigma_cmcrc = 380.0
    assert fm.sigma_cmcrfrc == 380.0
    fm.gamma_cmcr = 0.28
    assert fm.gamma_cmcrfr == 0.28
    fm.p_cmcr = 1.72
    assert fm.p_cmcrfr == 1.72
    fm.d_cmcr_max = 0.97
    assert fm.d_cmcrfr_max == 0.97

    fm.sigma_cmc0 = 120.0
    assert fm.sigma_cmcrfr0 == 120.0
    fm.sigma_cmcc = 390.0
    assert fm.sigma_cmcrfrc == 390.0
    fm.gamma_cmc = 0.29
    assert fm.gamma_cmcrfr == 0.29
    fm.p_cmc = 1.82
    assert fm.p_cmcrfr == 1.82
    fm.d_cmc_max = 0.95
    assert fm.d_cmcrfr_max == 0.95


# ============================================================================
# 2. ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALBLOCHPOINTPLASMONICPOLARITONIC_RESONANCE_ENERGY Tests
# ============================================================================

def test_m472_eng_electrothermoflexomagnetochiralblochpointplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{1.85e-5:>20.8e}{7:>10d}"
    deck = f"""# RADIOSS ENGINE DECK
/BEGIN
Test M472 Eng Directive Fixed
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBLOCHPOINTPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Chiral Bloch Point Plasmon Polariton Resonance Energy Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralblochpointplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralblochpointplasmonicpolaritonic_resonance_energies[1]
    assert isinstance(eng, EngElectrothermoflexomagnetochiralblochpointplasmonicpolaritonicResonanceEnergy)
    assert pytest.approx(eng.dt_etfcblochpointplp) == 1.85e-5
    assert eng.sens_id == 7


def test_m472_eng_electrothermoflexomagnetochiralblochpointplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M472 Eng Directive Free
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBLOCHPOINTPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
2.45e-5, 9
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralblochpointplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralblochpointplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfcblochpointplp) == 2.45e-5
    assert eng.sens_id == 9


def test_m472_eng_electrothermoflexomagnetochiralblochpointplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    aliases = [
        "/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_BLOCH_POINT_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALBLOCHPOINTPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBLOCHPOINTPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALBLOCHPOINTPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOBLOCHPOINTCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_BLOCH_POINT_CHIRAL_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOBLOCHPOINTCHIRALPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOBLOCHPOINTCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOBLOCHPOINTCHIRALPLASMONICPOLARITONIC_RESONANCE",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test M472 Eng Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "3.8e-5, 15",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.eng_electrothermoflexomagnetochiralblochpointplasmonicpolaritonic_resonance_energies
        eng = model.eng_electrothermoflexomagnetochiralblochpointplasmonicpolaritonic_resonance_energies[idx]
        assert pytest.approx(eng.dt_etfcblochpointplp) == 3.8e-5
        assert eng.sens_id == 15


def test_m472_eng_electrothermoflexomagnetochiralblochpointplasmonicpolaritonic_resonance_energy_fallback():
    # Test post_init fallback
    eng = EngElectrothermoflexomagnetochiralblochpointplasmonicpolaritonicResonanceEnergy(
        id=1,
        dt_etfcbobberplp=5.5e-5,
    )
    assert pytest.approx(eng.dt_etfcblochpointplp) == 5.5e-5

    eng2 = EngElectrothermoflexomagnetochiralblochpointplasmonicpolaritonicResonanceEnergy(
        id=2,
        dt_etfcblochpointplp=7.5e-5,
    )
    assert pytest.approx(eng2.dt_etfcbobberplp) == 7.5e-5
    assert pytest.approx(eng2.dt_etfplp) == 7.5e-5


# ============================================================================
# 3. CHERN_SPINOR_SPATIAL_LINKAGE_JOINT Tests
# ============================================================================

def test_m472_lagmul_chern_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{14:>10d}{15:>10d}{16:>10d}{2.6e6:>20.4e}{5:>10d}{1.6e-6:>20.4e}"
    c2 = f"{34.5:>20.4f}{49.5:>20.4f}{58.0:>20.4f}{15.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M472 Chern Spinor Spatial Linkage Joint Fixed
2022 0
/CHERN_SPINOR_SPATIAL_LINKAGE_JOINT/1
Chern Spinor Spatial Linkage Joint Card
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_chern_spinor_spatial_linkage_joints
    j = model.lagmul_chern_spinor_spatial_linkage_joints[1]
    assert isinstance(j, LagmulChernSpinorSpatialLinkageJoint)
    assert j.node1 == 14
    assert j.node2 == 15
    assert j.node3 == 16
    assert pytest.approx(j.stiff) == 2.6e6
    assert j.skew_id == 5
    assert pytest.approx(j.tol) == 1.6e-6
    assert pytest.approx(j.link_len_a) == 34.5
    assert pytest.approx(j.link_len_b) == 49.5
    assert pytest.approx(j.twist_angle_alpha) == 58.0
    assert pytest.approx(j.offset_distance_s) == 15.5

    # Offset distance properties
    assert pytest.approx(j.offset_distance_r) == 15.5
    assert pytest.approx(j.offset_distance_v) == 15.5
    assert pytest.approx(j.offset_distance_h) == 15.5
    assert pytest.approx(j.offset_distance_u) == 15.5
    assert pytest.approx(j.offset_distance_f) == 15.5


def test_m472_lagmul_chern_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M472 Chern Spinor Spatial Linkage Joint Free
/LAGMUL/CHERN_SPINOR_SPATIAL_LINKAGE_JOINT/2
24, 25, 26, 3.5e6, 8, 2.8e-6
39.5, 54.0, 68.0, 19.5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_chern_spinor_spatial_linkage_joints
    j = model.lagmul_chern_spinor_spatial_linkage_joints[2]
    assert j.node1 == 24
    assert j.node2 == 25
    assert j.node3 == 26
    assert pytest.approx(j.stiff) == 3.5e6
    assert j.skew_id == 8
    assert pytest.approx(j.tol) == 2.8e-6
    assert pytest.approx(j.link_len_a) == 39.5
    assert pytest.approx(j.link_len_b) == 54.0
    assert pytest.approx(j.twist_angle_alpha) == 68.0
    assert pytest.approx(j.offset_distance_s) == 19.5


def test_m472_lagmul_chern_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    aliases = [
        "/CHERN_SPINOR_SPATIAL_LINKAGE",
        "/LAGMUL/CHERN_SPINOR_SPATIAL_LINKAGE",
        "/CHERN_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM",
        "/CHERN_SPINOR_SPATIAL_SYMMETRIC_MECHANISM",
        "/CHERN_SPINOR_SPATIAL_6R_MECHANISM",
        "/CHERN_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM",
        "/LAGMUL/CHERN_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/CHERN_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/CHERN_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/CHERN_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/CHERN_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/CHERN_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/SINGER_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/SINGER_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/SINGER_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/SINGER_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/CHERN_SIMONS_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/CHERN_SIMONS_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/CHERN_SIMONS_SPATIAL_LINKAGE_JOINT",
        "/CHERN_SIMONS_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/CHERN_SIMONS_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/CHERN_SIMONS_TWISTOR_SPATIAL_LINKAGE_JOINT",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test M472 Joint Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "34, 35, 36, 1.9e6, 3, 1.2e-6",
            "26.0, 36.0, 46.0, 11.0",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.lagmul_chern_spinor_spatial_linkage_joints
        j = model.lagmul_chern_spinor_spatial_linkage_joints[idx]
        assert j.node1 == 34
        assert pytest.approx(j.stiff) == 1.9e6


# ============================================================================
# 4. SENSOR_SPRING_TOTAL_SNAP_RATE Tests
# ============================================================================

def test_m472_sensor_spring_total_snap_rate_fixed(tmp_path: Path):
    c1 = f"{28:>10d}{2.95e5:>20.4e}{0.0038:>20.4e}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_SNAP_RATE/1
Total Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_snap_rates
    s = model.sensor_spring_total_snap_rates[1]
    assert isinstance(s, SensorSpringTotalSnapRate)
    assert s.spring_id == 28
    assert pytest.approx(s.jtot_snp_max) == 2.95e5
    assert pytest.approx(s.t_delay) == 0.0038

    # Rate properties
    assert pytest.approx(s.jtot_snap_max) == 2.95e5
    assert pytest.approx(s.j_tot_snp_max) == 2.95e5
    assert pytest.approx(s.j_tot_snap_max) == 2.95e5
    assert pytest.approx(s.j_total_snp_max) == 2.95e5
    assert pytest.approx(s.j_total_snap_max) == 2.95e5
    assert pytest.approx(s.j_snap_tot_max) == 2.95e5
    assert pytest.approx(s.j_snap_total_max) == 2.95e5
    assert pytest.approx(s.j_linear_total_snap_max) == 2.95e5
    assert pytest.approx(s.j_linear_tot_snap_max) == 2.95e5
    assert pytest.approx(s.jtot_crk_rate_max) == 2.95e5
    assert pytest.approx(s.jtot_pop_rate_max) == 2.95e5
    assert pytest.approx(s.jtot_lock_rate_max) == 2.95e5


def test_m472_sensor_spring_total_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Snap Rate Free Format Test
/SENSOR/SPRING_TOTAL_SNAP_RATE/2
38, 3.95e5, 0.0055
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_snap_rates
    s = model.sensor_spring_total_snap_rates[2]
    assert s.spring_id == 38
    assert pytest.approx(s.jtot_snp_max) == 3.95e5
    assert pytest.approx(s.t_delay) == 0.0055


def test_m472_sensor_spring_total_snap_rate_aliases(tmp_path: Path):
    aliases = [
        "/SENSOR/SPRING_TOT_SNAP_RATE",
        "/SENSOR/SPRING_TOTAL_SNAP",
        "/SENSOR/SPRING_TOT_SNAP",
        "/SENSOR/SPRING_LINEAR_TOTAL_SNAP_RATE",
        "/SENSOR/SPRING_LINEAR_TOT_SNAP_RATE",
        "/SENSOR/SPRING_RATE_SNAP_TOT",
        "/SENSOR/SPRING_RATE_SNAP_TOTAL",
        "/SENSOR/SPRING_RATE_TOTAL_SNAP",
        "/SENSOR/SPRING_RATE_TOT_SNAP",
        "/SENSOR/SPRING_SNAP_RATE_TOTAL",
        "/SENSOR/SPRING_SNAP_RATE_TOT",
        "/SENSOR/TOTAL_SNAP_RATE_SPRING",
        "/SENSOR/TOT_SNAP_RATE_SPRING",
        "/SENSOR/LINEAR_TOTAL_SNAP_RATE_SPRING",
        "/SENSOR/SPRING_SNAP_TOTAL",
        "/SENSOR/SPRING_SNAP_TOT",
        "/SENSOR/SPRING_POP_RATE_TOTAL_SNAP",
        "/SENSOR/SPRING_LOCK_RATE_TOTAL_SNAP",
        "/SENSOR/RESULTANT_SNAP_RATE_SPRING",
        "/SENSOR/SPRING_RESULTANT_SNAP_RATE",
        "/SENSOR/SPRING_RESULTANT_SNAP",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Sensor Spring Total Snap Rate Aliases Test"]
    for sid, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{sid}",
            "58, 4.95e5, 0.0033",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(10, 10 + len(aliases)):
        assert sid in model.sensor_spring_total_snap_rates
        s = model.sensor_spring_total_snap_rates[sid]
        assert s.spring_id == 58
        assert pytest.approx(s.jtot_snp_max) == 4.95e5


def test_m472_sensor_spring_total_snap_rate_properties():
    s = SensorSpringTotalSnapRate(spring_id=1, jtot_snp_max=1.0e5)
    s.jtot_snap_max = 2.0e5
    assert s.jtot_snp_max == 2.0e5
    s.j_tot_snp_max = 2.1e5
    assert s.jtot_snp_max == 2.1e5
    s.j_tot_snap_max = 2.2e5
    assert s.jtot_snp_max == 2.2e5
    s.j_total_snp_max = 2.3e5
    assert s.jtot_snp_max == 2.3e5
    s.j_total_snap_max = 2.4e5
    assert s.jtot_snp_max == 2.4e5
    s.j_snap_tot_max = 2.5e5
    assert s.jtot_snp_max == 2.5e5
    s.j_snap_total_max = 2.6e5
    assert s.jtot_snp_max == 2.6e5
    s.j_linear_total_snap_max = 2.7e5
    assert s.jtot_snp_max == 2.7e5
    s.j_linear_tot_snap_max = 2.8e5
    assert s.jtot_snp_max == 2.8e5
    s.jtot_crk_rate_max = 2.9e5
    assert s.jtot_snp_max == 2.9e5
    s.jtot_pop_rate_max = 3.0e5
    assert s.jtot_snp_max == 3.0e5
    s.jtot_lock_rate_max = 3.1e5
    assert s.jtot_snp_max == 3.1e5


# ============================================================================
# 5. Error handling tests
# ============================================================================

def test_m472_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_COUPLED_MATRIX_CRUSHING_FAILURE_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBLOCHPOINTPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/CHERN_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TOTAL_SNAP_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4

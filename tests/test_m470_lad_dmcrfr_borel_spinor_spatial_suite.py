"""Tests for Milestone M470:
- /FAIL/LAD_DYNAMIC_MATRIX_CRUSHING_FAILURE_RATE (and aliases)
- /ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALTORONPLASMONICPOLARITONIC_RESONANCE_ENERGY (and aliases)
- /LAGMUL/BOREL_SPINOR_SPATIAL_LINKAGE_JOINT / /BOREL_SPINOR_SPATIAL_LINKAGE_JOINT (and aliases)
- /SENSOR/SPRING_TORSIONAL_SNAP_RATE (and aliases)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailLadDynamicMatrixCrushingFailureRate,
    EngElectrothermoflexomagnetochiraltoronplasmonicpolaritonicResonanceEnergy,
    LagmulBorelSpinorSpatialLinkageJoint,
    SensorSpringTorsionalSnapRate,
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
# 1. FAIL_LAD_DYNAMIC_MATRIX_CRUSHING_FAILURE_RATE Tests
# ============================================================================

def test_m470_fail_lad_dynamic_matrix_crushing_failure_rate_fixed(tmp_path: Path):
    c1 = f"{110.0:>20.4f}{380.0:>20.4f}{0.28:>20.4f}{1.45:>20.4f}{0.96:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M470 Fail Lad Dynamic Matrix Crushing Failure Rate Fixed
2022 0
/FAIL/LAD_DYNAMIC_MATRIX_CRUSHING_FAILURE_RATE/1
Dynamic Matrix Crushing Failure Rate Card Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.fail_laddynamicmatrixcrushingfailurerates
    fm = model.fail_laddynamicmatrixcrushingfailurerates[1]
    assert isinstance(fm, FailLadDynamicMatrixCrushingFailureRate)
    assert fm.mat_id == 1
    assert pytest.approx(fm.sigma_dmcrfr0) == 110.0
    assert pytest.approx(fm.sigma_dmcrfrc) == 380.0
    assert pytest.approx(fm.gamma_dmcrfr) == 0.28
    assert pytest.approx(fm.p_dmcrfr) == 1.45
    assert pytest.approx(fm.d_dmcrfr_max) == 0.96
    assert fm.ifail_sh == 2
    assert fm.ifail_so == 1

    # Properties
    assert pytest.approx(fm.sigma_dmcrf0) == 110.0
    assert pytest.approx(fm.sigma_dmcrfc) == 380.0
    assert pytest.approx(fm.gamma_dmcrf) == 0.28
    assert pytest.approx(fm.p_dmcrf) == 1.45
    assert pytest.approx(fm.d_dmcrf_max) == 0.96

    assert pytest.approx(fm.sigma_dmcr0) == 110.0
    assert pytest.approx(fm.sigma_dmcrc) == 380.0
    assert pytest.approx(fm.gamma_dmcr) == 0.28
    assert pytest.approx(fm.p_dmcr) == 1.45
    assert pytest.approx(fm.d_dmcr_max) == 0.96

    assert pytest.approx(fm.sigma_dmc0) == 110.0
    assert pytest.approx(fm.sigma_dmcc) == 380.0
    assert pytest.approx(fm.gamma_dmc) == 0.28
    assert pytest.approx(fm.p_dmc) == 1.45
    assert pytest.approx(fm.d_dmc_max) == 0.96


def test_m470_fail_lad_dynamic_matrix_crushing_failure_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M470 Fail Lad Dynamic Matrix Crushing Failure Rate Free
/FAIL/LAD_DYNAMIC_MATRIX_CRUSHING_FAILURE_RATE/2
125.0, 410.0, 0.32, 1.55, 0.98
1, 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.fail_laddynamicmatrixcrushingfailurerates
    fm = model.fail_laddynamicmatrixcrushingfailurerates[2]
    assert pytest.approx(fm.sigma_dmcrfr0) == 125.0
    assert pytest.approx(fm.sigma_dmcrfrc) == 410.0
    assert pytest.approx(fm.gamma_dmcrfr) == 0.32
    assert pytest.approx(fm.p_dmcrfr) == 1.55
    assert pytest.approx(fm.d_dmcrfr_max) == 0.98
    assert fm.ifail_sh == 1
    assert fm.ifail_so == 2


def test_m470_fail_lad_dynamic_matrix_crushing_failure_rate_aliases(tmp_path: Path):
    aliases = [
        "/FAIL/LADEVEZE_DYNAMIC_MATRIX_CRUSHING_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_MATRIX_CRUSHING_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_MATRIX_CRUSHING_FAILURE",
        "/FAIL/LAD_DYNAMIC_MATRIX_CRUSH_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_MATRIX_CRUSH_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_MATRIX_CRUSH_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_MATRIX_CRUSH_FAILURE",
        "/FAIL/LAD_DYNAMIC_TRANSVERSE_MATRIX_CRUSHING_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_TRANSVERSE_MATRIX_CRUSHING_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_TRANSVERSE_MATRIX_CRUSHING_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_TRANSVERSE_MATRIX_CRUSHING_FAILURE",
        "/FAIL/LAD_DYNAMIC_TRANSVERSE_MATRIX_CRUSH_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_TRANSVERSE_MATRIX_CRUSH_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_MATRIX_MICRO_CRUSHING_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_MATRIX_MICRO_CRUSHING_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_MATRIX_MICRO_CRUSHING_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_MATRIX_MICRO_CRUSHING_FAILURE",
        "/FAIL/LAD_DMCRFR",
        "/FAIL/LAD_DMCRFR_MODEL",
        "/FAIL/LAD_DMCRFR_LAW",
        "/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_MATRIX_CRUSHING_FAILURE",
        "/FAIL/LAD_DYNAMIC_MATRIX_CRUSHING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_MATRIX_CRUSHING_DAMAGE_RATE",
        "/FAIL/LAD_DYNAMIC_MATRIX_CRUSH_DAMAGE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_MATRIX_CRUSH_DAMAGE_RATE",
        "/FAIL/LAD_DYNAMIC_MATRIX_MICRO_CRUSHING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_MATRIX_MICRO_CRUSHING_DAMAGE_RATE",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test M470 Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "95.0, 340.0, 0.22, 1.35, 0.95",
            "1, 1",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.fail_laddynamicmatrixcrushingfailurerates
        fm = model.fail_laddynamicmatrixcrushingfailurerates[idx]
        assert pytest.approx(fm.sigma_dmcrfr0) == 95.0
        assert pytest.approx(fm.sigma_dmcrfrc) == 340.0


def test_m470_fail_lad_dynamic_matrix_crushing_failure_rate_properties():
    fm = FailLadDynamicMatrixCrushingFailureRate(
        mat_id=1,
        sigma_dmcrfr0=100.0,
        sigma_dmcrfrc=350.0,
        gamma_dmcrfr=0.25,
        p_dmcrfr=1.5,
        d_dmcrfr_max=0.99,
    )
    # Test property setters
    fm.sigma_dmcrf0 = 105.0
    assert fm.sigma_dmcrfr0 == 105.0
    fm.sigma_dmcrfc = 360.0
    assert fm.sigma_dmcrfrc == 360.0
    fm.gamma_dmcrf = 0.26
    assert fm.gamma_dmcrfr == 0.26
    fm.p_dmcrf = 1.6
    assert fm.p_dmcrfr == 1.6
    fm.d_dmcrf_max = 0.98
    assert fm.d_dmcrfr_max == 0.98

    fm.sigma_dmcr0 = 110.0
    assert fm.sigma_dmcrfr0 == 110.0
    fm.sigma_dmcrc = 370.0
    assert fm.sigma_dmcrfrc == 370.0
    fm.gamma_dmcr = 0.27
    assert fm.gamma_dmcrfr == 0.27
    fm.p_dmcr = 1.7
    assert fm.p_dmcrfr == 1.7
    fm.d_dmcr_max = 0.97
    assert fm.d_dmcrfr_max == 0.97

    fm.sigma_dmc0 = 115.0
    assert fm.sigma_dmcrfr0 == 115.0
    fm.sigma_dmcc = 380.0
    assert fm.sigma_dmcrfrc == 380.0
    fm.gamma_dmc = 0.28
    assert fm.gamma_dmcrfr == 0.28
    fm.p_dmc = 1.8
    assert fm.p_dmcrfr == 1.8
    fm.d_dmc_max = 0.95
    assert fm.d_dmcrfr_max == 0.95


# ============================================================================
# 2. ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALTORONPLASMONICPOLARITONIC_RESONANCE_ENERGY Tests
# ============================================================================

def test_m470_eng_electrothermoflexomagnetochiraltoronplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{1.75e-5:>20.8e}{6:>10d}"
    deck = f"""# RADIOSS ENGINE DECK
/BEGIN
Test M470 Eng Directive Fixed
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALTORONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Chiral Toron Plasmon Polariton Resonance Energy Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiraltoronplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiraltoronplasmonicpolaritonic_resonance_energies[1]
    assert isinstance(eng, EngElectrothermoflexomagnetochiraltoronplasmonicpolaritonicResonanceEnergy)
    assert pytest.approx(eng.dt_etfctoronplp) == 1.75e-5
    assert eng.sens_id == 6


def test_m470_eng_electrothermoflexomagnetochiraltoronplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M470 Eng Directive Free
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALTORONPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
2.25e-5, 8
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiraltoronplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiraltoronplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfctoronplp) == 2.25e-5
    assert eng.sens_id == 8


def test_m470_eng_electrothermoflexomagnetochiraltoronplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    aliases = [
        "/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_TORON_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALTORONPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALTORONPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALTORONPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOTORONCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_TORON_CHIRAL_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOTORONCHIRALPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOTORONCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOTORONCHIRALPLASMONICPOLARITONIC_RESONANCE",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test M470 Eng Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "3.5e-5, 12",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.eng_electrothermoflexomagnetochiraltoronplasmonicpolaritonic_resonance_energies
        eng = model.eng_electrothermoflexomagnetochiraltoronplasmonicpolaritonic_resonance_energies[idx]
        assert pytest.approx(eng.dt_etfctoronplp) == 3.5e-5
        assert eng.sens_id == 12


# ============================================================================
# 3. BOREL_SPINOR_SPATIAL_LINKAGE_JOINT Tests
# ============================================================================

def test_m470_lagmul_borel_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{11:>10d}{12:>10d}{13:>10d}{2.4e6:>20.4e}{4:>10d}{1.5e-6:>20.4e}"
    c2 = f"{32.5:>20.4f}{48.0:>20.4f}{55.0:>20.4f}{14.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M470 Borel Spinor Spatial Linkage Joint Fixed
2022 0
/BOREL_SPINOR_SPATIAL_LINKAGE_JOINT/1
Borel Spinor Spatial Linkage Joint Card
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_borel_spinor_spatial_linkage_joints
    j = model.lagmul_borel_spinor_spatial_linkage_joints[1]
    assert isinstance(j, LagmulBorelSpinorSpatialLinkageJoint)
    assert j.node1 == 11
    assert j.node2 == 12
    assert j.node3 == 13
    assert pytest.approx(j.stiff) == 2.4e6
    assert j.skew_id == 4
    assert pytest.approx(j.tol) == 1.5e-6
    assert pytest.approx(j.link_len_a) == 32.5
    assert pytest.approx(j.link_len_b) == 48.0
    assert pytest.approx(j.twist_angle_alpha) == 55.0
    assert pytest.approx(j.offset_distance_s) == 14.5

    # Offset distance properties
    assert pytest.approx(j.offset_distance_r) == 14.5
    assert pytest.approx(j.offset_distance_v) == 14.5
    assert pytest.approx(j.offset_distance_h) == 14.5
    assert pytest.approx(j.offset_distance_u) == 14.5
    assert pytest.approx(j.offset_distance_f) == 14.5


def test_m470_lagmul_borel_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M470 Borel Spinor Spatial Linkage Joint Free
/LAGMUL/BOREL_SPINOR_SPATIAL_LINKAGE_JOINT/2
21, 22, 23, 3.2e6, 7, 2.5e-6
38.0, 52.0, 65.0, 18.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_borel_spinor_spatial_linkage_joints
    j = model.lagmul_borel_spinor_spatial_linkage_joints[2]
    assert j.node1 == 21
    assert j.node2 == 22
    assert j.node3 == 23
    assert pytest.approx(j.stiff) == 3.2e6
    assert j.skew_id == 7
    assert pytest.approx(j.tol) == 2.5e-6
    assert pytest.approx(j.link_len_a) == 38.0
    assert pytest.approx(j.link_len_b) == 52.0
    assert pytest.approx(j.twist_angle_alpha) == 65.0
    assert pytest.approx(j.offset_distance_s) == 18.0


def test_m470_lagmul_borel_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    aliases = [
        "/BOREL_SPINOR_SPATIAL_LINKAGE",
        "/LAGMUL/BOREL_SPINOR_SPATIAL_LINKAGE",
        "/BOREL_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM",
        "/BOREL_SPINOR_SPATIAL_SYMMETRIC_MECHANISM",
        "/BOREL_SPINOR_SPATIAL_6R_MECHANISM",
        "/BOREL_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM",
        "/LAGMUL/BOREL_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/BOREL_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/BOREL_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/BOREL_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/BOREL_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/BOREL_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/WEIL_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/WEIL_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/WEIL_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/WEIL_TWISTOR_SPATIAL_LINKAGE_JOINT",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test M470 Joint Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "31, 32, 33, 1.8e6, 2, 1e-6",
            "25.0, 35.0, 45.0, 10.0",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.lagmul_borel_spinor_spatial_linkage_joints
        j = model.lagmul_borel_spinor_spatial_linkage_joints[idx]
        assert j.node1 == 31
        assert pytest.approx(j.stiff) == 1.8e6


# ============================================================================
# 4. SENSOR_SPRING_TORSIONAL_SNAP_RATE Tests
# ============================================================================

def test_m470_sensor_spring_torsional_snap_rate_fixed(tmp_path: Path):
    c1 = f"{25:>10d}{2.85e5:>20.4e}{0.0035:>20.4e}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_SNAP_RATE/1
Torsional Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_snap_rates
    s = model.sensor_spring_torsional_snap_rates[1]
    assert isinstance(s, SensorSpringTorsionalSnapRate)
    assert s.spring_id == 25
    assert pytest.approx(s.jtor_snp_max) == 2.85e5
    assert pytest.approx(s.t_delay) == 0.0035

    # Rate properties
    assert pytest.approx(s.jtor_snap_max) == 2.85e5
    assert pytest.approx(s.jtors_snap_max) == 2.85e5
    assert pytest.approx(s.jtorsional_snap_max) == 2.85e5
    assert pytest.approx(s.jtwist_snap_max) == 2.85e5
    assert pytest.approx(s.jtwst_snap_max) == 2.85e5
    assert pytest.approx(s.j_tor_snp_max) == 2.85e5
    assert pytest.approx(s.j_tors_snp_max) == 2.85e5
    assert pytest.approx(s.j_torsional_snp_max) == 2.85e5
    assert pytest.approx(s.j_twist_snp_max) == 2.85e5
    assert pytest.approx(s.j_snap_tor_max) == 2.85e5
    assert pytest.approx(s.j_snap_tors_max) == 2.85e5
    assert pytest.approx(s.j_snap_torsional_max) == 2.85e5
    assert pytest.approx(s.j_snap_twist_max) == 2.85e5
    assert pytest.approx(s.jtor_crk_rate_max) == 2.85e5
    assert pytest.approx(s.jtor_pop_rate_max) == 2.85e5
    assert pytest.approx(s.jtor_lock_rate_max) == 2.85e5
    assert pytest.approx(s.jtor_snap_rate_max) == 2.85e5


def test_m470_sensor_spring_torsional_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Torsional Snap Rate Free Format Test
/SENSOR/SPRING_TORSIONAL_SNAP_RATE/2
35, 3.85e5, 0.0052
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_torsional_snap_rates
    s = model.sensor_spring_torsional_snap_rates[2]
    assert s.spring_id == 35
    assert pytest.approx(s.jtor_snp_max) == 3.85e5
    assert pytest.approx(s.t_delay) == 0.0052


def test_m470_sensor_spring_torsional_snap_rate_aliases(tmp_path: Path):
    aliases = [
        "/SENSOR/SPRING_TORS_SNAP_RATE",
        "/SENSOR/SPRING_TORSIONAL_SNAP",
        "/SENSOR/SPRING_TORS_SNAP",
        "/SENSOR/SPRING_ANGULAR_TORSIONAL_SNAP_RATE",
        "/SENSOR/SPRING_ANGULAR_TORS_SNAP_RATE",
        "/SENSOR/SPRING_ANGULAR_TORSIONAL_SNAP",
        "/SENSOR/SPRING_ANGULAR_TORS_SNAP",
        "/SENSOR/SPRING_TWIST_SNAP_RATE",
        "/SENSOR/SPRING_TWIST_SNAP",
        "/SENSOR/SPRING_POP_RATE_TORSIONAL_SNAP",
        "/SENSOR/SPRING_LOCK_RATE_TORSIONAL_SNAP",
        "/SENSOR/SPRING_SNAP_RATE_TORSIONAL",
        "/SENSOR/SPRING_SNAP_RATE_TORS",
        "/SENSOR/TORSIONAL_SNAP_RATE_SPRING",
        "/SENSOR/TORS_SNAP_RATE_SPRING",
        "/SENSOR/SPRING_SNAP_TORSIONAL",
        "/SENSOR/SPRING_SNAP_TORS",
        "/SENSOR/SPRING_RATE_TORSIONAL_SNAP",
        "/SENSOR/SPRING_RATE_TORS_SNAP",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Sensor Spring Torsional Snap Rate Aliases Test"]
    for sid, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{sid}",
            "55, 4.8e5, 0.0031",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(10, 10 + len(aliases)):
        assert sid in model.sensor_spring_torsional_snap_rates
        s = model.sensor_spring_torsional_snap_rates[sid]
        assert s.spring_id == 55
        assert pytest.approx(s.jtor_snp_max) == 4.8e5


# ============================================================================
# 5. Error handling tests
# ============================================================================

def test_m470_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_DYNAMIC_MATRIX_CRUSHING_FAILURE_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALTORONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/BOREL_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TORSIONAL_SNAP_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4

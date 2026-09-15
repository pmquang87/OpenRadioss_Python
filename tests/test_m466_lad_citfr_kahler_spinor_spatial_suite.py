"""Tests for Milestone M466:
- /FAIL/LAD_COUPLED_INTERLAMINAR_TENSION_FAILURE_RATE (and aliases)
- /ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALVORTEXPLASMONICPOLARITONIC_RESONANCE_ENERGY (and aliases)
- /LAGMUL/KAHLER_SPINOR_SPATIAL_LINKAGE_JOINT / /KAHLER_SPINOR_SPATIAL_LINKAGE_JOINT (and aliases)
- /SENSOR/SPRING_TOTAL_LOCK_RATE (and aliases)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailLadCoupledInterlaminarTensionFailureRate,
    EngElectrothermoflexomagnetochiralvortexplasmonicpolaritonicResonanceEnergy,
    LagmulKahlerSpinorSpatialLinkageJoint,
    SensorSpringTotalLockRate,
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
# 1. FAIL_LAD_COUPLED_INTERLAMINAR_TENSION_FAILURE_RATE Tests
# ============================================================================

def test_m466_fail_lad_coupled_interlaminar_tension_failure_rate_fixed(tmp_path: Path):
    c1 = f"{92.0:>20.4f}{330.0:>20.4f}{0.19:>20.4f}{1.48:>20.4f}{0.96:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M466 Fail Lad Coupled Interlaminar Tension Failure Rate Fixed
2022 0
/FAIL/LAD_COUPLED_INTERLAMINAR_TENSION_FAILURE_RATE/1
Coupled Interlaminar Tension Failure Rate Card Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.fail_ladcoupledinterlaminartensionfailurerates
    fm = model.fail_ladcoupledinterlaminartensionfailurerates[1]
    assert isinstance(fm, FailLadCoupledInterlaminarTensionFailureRate)
    assert fm.mat_id == 1
    assert pytest.approx(fm.sigma_citfr0) == 92.0
    assert pytest.approx(fm.sigma_citfrc) == 330.0
    assert pytest.approx(fm.gamma_citfr) == 0.19
    assert pytest.approx(fm.p_citfr) == 1.48
    assert pytest.approx(fm.d_citfr_max) == 0.96
    assert fm.ifail_sh == 2
    assert fm.ifail_so == 1

    # Properties
    assert pytest.approx(fm.sigma_citf0) == 92.0
    assert pytest.approx(fm.sigma_citfc) == 330.0
    assert pytest.approx(fm.gamma_citf) == 0.19
    assert pytest.approx(fm.p_citf) == 1.48
    assert pytest.approx(fm.d_citf_max) == 0.96

    assert pytest.approx(fm.sigma_cit0) == 92.0
    assert pytest.approx(fm.sigma_citc) == 330.0
    assert pytest.approx(fm.gamma_cit) == 0.19
    assert pytest.approx(fm.p_cit) == 1.48
    assert pytest.approx(fm.d_cit_max) == 0.96

    assert pytest.approx(fm.sigma_cin0) == 92.0
    assert pytest.approx(fm.sigma_cinc) == 330.0
    assert pytest.approx(fm.gamma_cin) == 0.19
    assert pytest.approx(fm.p_cin) == 1.48
    assert pytest.approx(fm.d_cin_max) == 0.96


def test_m466_fail_lad_coupled_interlaminar_tension_failure_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M466 Fail Lad Coupled Interlaminar Tension Failure Rate Free
/FAIL/LAD_COUPLED_INTERLAMINAR_TENSION_FAILURE_RATE/2
102.0, 350.0, 0.22, 1.60, 0.99
1, 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.fail_ladcoupledinterlaminartensionfailurerates
    fm = model.fail_ladcoupledinterlaminartensionfailurerates[2]
    assert pytest.approx(fm.sigma_citfr0) == 102.0
    assert pytest.approx(fm.sigma_citfrc) == 350.0
    assert pytest.approx(fm.gamma_citfr) == 0.22
    assert pytest.approx(fm.p_citfr) == 1.60
    assert pytest.approx(fm.d_citfr_max) == 0.99
    assert fm.ifail_sh == 1
    assert fm.ifail_so == 2


def test_m466_fail_lad_coupled_interlaminar_tension_failure_rate_aliases(tmp_path: Path):
    aliases = [
        "/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_TENSION_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_INTERLAMINAR_TENSION_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_TENSION_FAILURE",
        "/FAIL/LAD_COUPLE_INTERLAMINAR_TENSION_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLE_INTERLAMINAR_TENSION_FAILURE_RATE",
        "/FAIL/LAD_COUPLE_INTERLAMINAR_TENSION_FAILURE",
        "/FAIL/LADEVEZE_COUPLE_INTERLAMINAR_TENSION_FAILURE",
        "/FAIL/LAD_COUPLED_INTERLAMINAR_TENSIONAL_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_TENSIONAL_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_INTERLAMINAR_TENSIONAL_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_TENSIONAL_FAILURE",
        "/FAIL/LAD_COUPLE_INTERLAMINAR_TENSIONAL_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLE_INTERLAMINAR_TENSIONAL_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_INTERLAMINAR_NORMAL_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_NORMAL_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_INTERLAMINAR_NORMAL_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_NORMAL_FAILURE",
        "/FAIL/LAD_COUPLED_INTERLAMINAR_NORMAL_PEELING_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_NORMAL_PEELING_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_INTERLAMINAR_PEELING_FAILURE_RATE",
        "/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_PEELING_FAILURE_RATE",
        "/FAIL/LAD_COUPLED_INTERLAMINAR_PEELING_FAILURE",
        "/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_PEELING_FAILURE",
        "/FAIL/LAD_CITFR",
        "/FAIL/LAD_CITFR_MODEL",
        "/FAIL/LAD_CITFR_LAW",
        "/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_INTERLAMINAR_TENSION_FAILURE",
        "/FAIL/LAD_COUPLED_INTERLAMINAR_TENSION_DAMAGE_RATE",
        "/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_TENSION_DAMAGE_RATE",
        "/FAIL/LAD_COUPLE_INTERLAMINAR_TENSION_DAMAGE_RATE",
        "/FAIL/LADEVEZE_COUPLE_INTERLAMINAR_TENSION_DAMAGE_RATE",
        "/FAIL/LAD_COUPLED_INTERLAMINAR_NORMAL_DAMAGE_RATE",
        "/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_NORMAL_DAMAGE_RATE",
        "/FAIL/LAD_COUPLED_INTERLAMINAR_PEELING_DAMAGE_RATE",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test M466 Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "81.0, 300.0, 0.16, 1.35, 0.98",
            "1, 1",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.fail_ladcoupledinterlaminartensionfailurerates
        fm = model.fail_ladcoupledinterlaminartensionfailurerates[idx]
        assert pytest.approx(fm.sigma_citfr0) == 81.0


# ============================================================================
# 2. ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALVORTEXPLASMONICPOLARITONIC_RESONANCE_ENERGY Tests
# ============================================================================

def test_m466_eng_energy_fixed(tmp_path: Path):
    c1 = f"{2.5e-4:>20.6f}{20:>10d}"
    deck = f"""# RADIOSS ENGINE DECK
/BEGIN
Test M466 Eng Output Fixed
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALVORTEXPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Vortex Resonance Energy Directive Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralvortexplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralvortexplasmonicpolaritonic_resonance_energies[1]
    assert isinstance(eng, EngElectrothermoflexomagnetochiralvortexplasmonicpolaritonicResonanceEnergy)
    assert pytest.approx(eng.dt_etfcvortexplp) == 2.5e-4
    assert eng.sens_id == 20


def test_m466_eng_energy_free(tmp_path: Path):
    deck = """# RADIOSS ENGINE DECK
/BEGIN
Test M466 Eng Output Free
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALVORTEXPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
3.5e-4, 23
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralvortexplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralvortexplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfcvortexplp) == 3.5e-4
    assert eng.sens_id == 23


def test_m466_eng_energy_aliases(tmp_path: Path):
    aliases = [
        "/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_VORTEX_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALVORTEXPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALVORTEXPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALVORTEXPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOVORTEXCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_VORTEX_CHIRAL_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOVORTEXCHIRALPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOVORTEXCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOVORTEXCHIRALPLASMONICPOLARITONIC_RESONANCE",
    ]
    deck_lines = ["# RADIOSS ENGINE DECK", "/BEGIN", "Test Eng Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "4.5e-4, 30",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.eng_electrothermoflexomagnetochiralvortexplasmonicpolaritonic_resonance_energies
        eng = model.eng_electrothermoflexomagnetochiralvortexplasmonicpolaritonic_resonance_energies[idx]
        assert pytest.approx(eng.dt_etfcvortexplp) == 4.5e-4


# ============================================================================
# 3. LAGMUL_KAHLER_SPINOR_SPATIAL_LINKAGE_JOINT Tests
# ============================================================================

def test_m466_kahler_spinor_joint_fixed(tmp_path: Path):
    c1 = f"{150:>10d}{151:>10d}{152:>10d}{3.7e6:>20.4f}{16:>10d}{1.8e-5:>20.6f}"
    c2 = f"{22.5:>20.4f}{29.0:>20.4f}{80.0:>20.4f}{15.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M466 Kahler Joint Fixed
2022 0
/KAHLER_SPINOR_SPATIAL_LINKAGE_JOINT/1
Kahler Spinor Joint Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_kahler_spinor_spatial_linkage_joints
    j = model.lagmul_kahler_spinor_spatial_linkage_joints[1]
    assert isinstance(j, LagmulKahlerSpinorSpatialLinkageJoint)
    assert j.node1 == 150
    assert j.node2 == 151
    assert j.node3 == 152
    assert pytest.approx(j.stiff) == 3.7e6
    assert j.skew_id == 16
    assert pytest.approx(j.tol) == 1.8e-5
    assert pytest.approx(j.link_len_a) == 22.5
    assert pytest.approx(j.link_len_b) == 29.0
    assert pytest.approx(j.twist_angle_alpha) == 80.0
    assert pytest.approx(j.offset_distance_s) == 15.5

    # Offset aliases
    assert pytest.approx(j.offset_distance_r) == 15.5
    assert pytest.approx(j.offset_distance_v) == 15.5
    assert pytest.approx(j.offset_distance_h) == 15.5
    assert pytest.approx(j.offset_distance_u) == 15.5
    assert pytest.approx(j.offset_distance_f) == 15.5


def test_m466_kahler_spinor_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M466 Kahler Joint Free
/LAGMUL/KAHLER_SPINOR_SPATIAL_LINKAGE_JOINT/2
250, 251, 252, 4.7e6, 17, 2.8e-5
25.0, 31.5, 95.0, 17.2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_kahler_spinor_spatial_linkage_joints
    j = model.lagmul_kahler_spinor_spatial_linkage_joints[2]
    assert j.node1 == 250
    assert j.node2 == 251
    assert j.node3 == 252
    assert pytest.approx(j.stiff) == 4.7e6
    assert j.skew_id == 17
    assert pytest.approx(j.tol) == 2.8e-5
    assert pytest.approx(j.link_len_a) == 25.0
    assert pytest.approx(j.link_len_b) == 31.5
    assert pytest.approx(j.twist_angle_alpha) == 95.0
    assert pytest.approx(j.offset_distance_s) == 17.2


def test_m466_kahler_spinor_joint_aliases(tmp_path: Path):
    aliases = [
        "/KAHLER_SPINOR_SPATIAL_LINKAGE",
        "/LAGMUL/KAHLER_SPINOR_SPATIAL_LINKAGE",
        "/KAHLER_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM",
        "/KAHLER_SPINOR_SPATIAL_SYMMETRIC_MECHANISM",
        "/KAHLER_SPINOR_SPATIAL_6R_MECHANISM",
        "/KAHLER_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM",
        "/LAGMUL/KAHLER_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/KAHLER_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/KAHLER_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/KAHLER_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/KAHLER_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/KAHLER_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/KAHLER_ATIYAH_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/KAHLER_ATIYAH_SPINOR_SPATIAL_LINKAGE_JOINT",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test Kahler Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "350, 351, 352, 1.4e6, 0, 1e-6",
            "19.0, 19.0, 0.0, 0.0",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.lagmul_kahler_spinor_spatial_linkage_joints
        j = model.lagmul_kahler_spinor_spatial_linkage_joints[idx]
        assert j.node1 == 350


# ============================================================================
# 4. SENSOR_SPRING_TOTAL_LOCK_RATE Tests
# ============================================================================

def test_m466_sensor_spring_total_lock_rate_fixed(tmp_path: Path):
    c1 = f"{17:>10d}{2.04e5:>20.4f}{0.0030:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Lock Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_LOCK_RATE/1
Total Lock Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_lock_rates
    s = model.sensor_spring_total_lock_rates[1]
    assert isinstance(s, SensorSpringTotalLockRate)
    assert s.spring_id == 17
    assert pytest.approx(s.jtot_lock_max) == 2.04e5
    assert pytest.approx(s.t_delay) == 0.0030

    # Check property aliases
    assert pytest.approx(s.j_lock_tot_max) == 2.04e5
    assert pytest.approx(s.j_tot_lock_max) == 2.04e5
    assert pytest.approx(s.jtot_snap_max) == 2.04e5
    assert pytest.approx(s.jtot_rate_max) == 2.04e5
    assert pytest.approx(s.jtot_roc_rate_max) == 2.04e5
    assert pytest.approx(s.jtot_drop_rate_max) == 2.04e5
    assert pytest.approx(s.jtot_crk_rate_max) == 2.04e5
    assert pytest.approx(s.jtot_pop_rate_max) == 2.04e5
    assert pytest.approx(s.jtot_lock_rate_max) == 2.04e5
    assert pytest.approx(s.jtotal_lock_max) == 2.04e5
    assert pytest.approx(s.j_total_lock_max) == 2.04e5
    assert pytest.approx(s.j_lock_total_max) == 2.04e5


def test_m466_sensor_spring_total_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Lock Rate Free Format Test
/SENSOR/SPRING_TOTAL_LOCK_RATE/2
27, 3.04e5, 0.0040
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_lock_rates
    s = model.sensor_spring_total_lock_rates[2]
    assert s.spring_id == 27
    assert pytest.approx(s.jtot_lock_max) == 3.04e5
    assert pytest.approx(s.t_delay) == 0.0040


def test_m466_sensor_spring_total_lock_rate_aliases(tmp_path: Path):
    aliases = [
        "/SENSOR/SPRING_TOT_LOCK_RATE",
        "/SENSOR/SPRING_TOTAL_LOCK",
        "/SENSOR/SPRING_TOT_LOCK",
        "/SENSOR/SPRING_LINEAR_TOTAL_LOCK_RATE",
        "/SENSOR/SPRING_LINEAR_TOT_LOCK_RATE",
        "/SENSOR/SPRING_POP_RATE_TOTAL_LOCK",
        "/SENSOR/SPRING_LOCK_RATE_TOTAL",
        "/SENSOR/SPRING_LOCK_RATE_TOT",
        "/SENSOR/SPRING_LOCK_RATE_LINEAR_TOT",
        "/SENSOR/TOT_LOCK_RATE_SPRING",
        "/SENSOR/LINEAR_TOTAL_LOCK_RATE_SPRING",
        "/SENSOR/SPRING_LOCK_TOTAL",
        "/SENSOR/SPRING_LOCK_LINEAR_TOT",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Sensor Spring Total Lock Rate Aliases Test"]
    for sid, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{sid}",
            "40, 4.0e5, 0.0018",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(10, 10 + len(aliases)):
        assert sid in model.sensor_spring_total_lock_rates
        s = model.sensor_spring_total_lock_rates[sid]
        assert s.spring_id == 40
        assert pytest.approx(s.jtot_lock_max) == 4.0e5


# ============================================================================
# 5. Error handling tests
# ============================================================================

def test_m466_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_COUPLED_INTERLAMINAR_TENSION_FAILURE_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALVORTEXPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/KAHLER_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TOTAL_LOCK_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4

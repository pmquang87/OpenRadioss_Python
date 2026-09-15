"""Tests for Milestone M465:
- /FAIL/LAD_TRANSVERSE_INTERLAMINAR_TENSION_FAILURE_RATE (and aliases)
- /ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSOLITONPLASMONICPOLARITONIC_RESONANCE_ENERGY (and aliases)
- /LAGMUL/MAJORANA_SPINOR_SPATIAL_LINKAGE_JOINT / /MAJORANA_SPINOR_SPATIAL_LINKAGE_JOINT (and aliases)
- /SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE (and aliases)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailLadTransverseInterlaminarTensionFailureRate,
    EngElectrothermoflexomagnetochiralsolitonplasmonicpolaritonicResonanceEnergy,
    LagmulMajoranaSpinorSpatialLinkageJoint,
    SensorSpringTotalAngularLockRate,
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
# 1. FAIL_LAD_TRANSVERSE_INTERLAMINAR_TENSION_FAILURE_RATE Tests
# ============================================================================

def test_m465_fail_lad_transverse_interlaminar_tension_failure_rate_fixed(tmp_path: Path):
    c1 = f"{88.0:>20.4f}{320.0:>20.4f}{0.18:>20.4f}{1.45:>20.4f}{0.95:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M465 Fail Lad Transverse Interlaminar Tension Failure Rate Fixed
2022 0
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_TENSION_FAILURE_RATE/1
Transverse Interlaminar Tension Failure Rate Card Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.fail_ladtransverseinterlaminartensionfailurerates
    fm = model.fail_ladtransverseinterlaminartensionfailurerates[1]
    assert isinstance(fm, FailLadTransverseInterlaminarTensionFailureRate)
    assert fm.mat_id == 1
    assert pytest.approx(fm.sigma_titfr0) == 88.0
    assert pytest.approx(fm.sigma_titfrc) == 320.0
    assert pytest.approx(fm.gamma_titfr) == 0.18
    assert pytest.approx(fm.p_titfr) == 1.45
    assert pytest.approx(fm.d_titfr_max) == 0.95
    assert fm.ifail_sh == 2
    assert fm.ifail_so == 1

    # Properties
    assert pytest.approx(fm.sigma_titf0) == 88.0
    assert pytest.approx(fm.sigma_titfc) == 320.0
    assert pytest.approx(fm.gamma_titf) == 0.18
    assert pytest.approx(fm.p_titf) == 1.45
    assert pytest.approx(fm.d_titf_max) == 0.95

    assert pytest.approx(fm.sigma_tit0) == 88.0
    assert pytest.approx(fm.sigma_titc) == 320.0
    assert pytest.approx(fm.gamma_tit) == 0.18
    assert pytest.approx(fm.p_tit) == 1.45
    assert pytest.approx(fm.d_tit_max) == 0.95

    assert pytest.approx(fm.sigma_tin0) == 88.0
    assert pytest.approx(fm.sigma_tinc) == 320.0
    assert pytest.approx(fm.gamma_tin) == 0.18
    assert pytest.approx(fm.p_tin) == 1.45
    assert pytest.approx(fm.d_tin_max) == 0.95


def test_m465_fail_lad_transverse_interlaminar_tension_failure_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M465 Fail Lad Transverse Interlaminar Tension Failure Rate Free
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_TENSION_FAILURE_RATE/2
98.0, 340.0, 0.21, 1.55, 0.98
1, 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.fail_ladtransverseinterlaminartensionfailurerates
    fm = model.fail_ladtransverseinterlaminartensionfailurerates[2]
    assert pytest.approx(fm.sigma_titfr0) == 98.0
    assert pytest.approx(fm.sigma_titfrc) == 340.0
    assert pytest.approx(fm.gamma_titfr) == 0.21
    assert pytest.approx(fm.p_titfr) == 1.55
    assert pytest.approx(fm.d_titfr_max) == 0.98
    assert fm.ifail_sh == 1
    assert fm.ifail_so == 2


def test_m465_fail_lad_transverse_interlaminar_tension_failure_rate_aliases(tmp_path: Path):
    aliases = [
        "/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_TENSION_FAILURE_RATE",
        "/FAIL/LAD_TRANSVERSE_INTERLAMINAR_TENSION_FAILURE",
        "/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_TENSION_FAILURE",
        "/FAIL/LAD_TRANSVERSE_INTERLAMINAR_TENSIONAL_FAILURE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_TENSIONAL_FAILURE_RATE",
        "/FAIL/LAD_TRANSVERSE_INTERLAMINAR_TENSIONAL_FAILURE",
        "/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_TENSIONAL_FAILURE",
        "/FAIL/LAD_TRANSVERSE_INTERLAMINAR_NORMAL_FAILURE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_NORMAL_FAILURE_RATE",
        "/FAIL/LAD_TRANSVERSE_INTERLAMINAR_NORMAL_FAILURE",
        "/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_NORMAL_FAILURE",
        "/FAIL/LAD_TRANSVERSE_INTERLAMINAR_NORMAL_PEELING_FAILURE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_NORMAL_PEELING_FAILURE_RATE",
        "/FAIL/LAD_TRANSVERSE_INTERLAMINAR_PEELING_FAILURE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_PEELING_FAILURE_RATE",
        "/FAIL/LAD_TRANSVERSE_INTERLAMINAR_PEELING_FAILURE",
        "/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_PEELING_FAILURE",
        "/FAIL/LAD_TITFR",
        "/FAIL/LAD_TITFR_MODEL",
        "/FAIL/LAD_TITFR_LAW",
        "/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_INTERLAMINAR_TENSION_FAILURE",
        "/FAIL/LAD_TRANSVERSE_INTERLAMINAR_TENSION_DAMAGE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_TENSION_DAMAGE_RATE",
        "/FAIL/LAD_TRANSVERSE_INTERLAMINAR_NORMAL_DAMAGE_RATE",
        "/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_NORMAL_DAMAGE_RATE",
        "/FAIL/LAD_TRANSVERSE_INTERLAMINAR_PEELING_DAMAGE_RATE",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test M465 Aliases"]
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
        assert idx in model.fail_ladtransverseinterlaminartensionfailurerates
        fm = model.fail_ladtransverseinterlaminartensionfailurerates[idx]
        assert pytest.approx(fm.sigma_titfr0) == 79.0


# ============================================================================
# 2. ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALSOLITONPLASMONICPOLARITONIC_RESONANCE_ENERGY Tests
# ============================================================================

def test_m465_eng_energy_fixed(tmp_path: Path):
    c1 = f"{2.4e-4:>20.6f}{19:>10d}"
    deck = f"""# RADIOSS ENGINE DECK
/BEGIN
Test M465 Eng Output Fixed
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSOLITONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Soliton Resonance Energy Directive Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralsolitonplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralsolitonplasmonicpolaritonic_resonance_energies[1]
    assert isinstance(eng, EngElectrothermoflexomagnetochiralsolitonplasmonicpolaritonicResonanceEnergy)
    assert pytest.approx(eng.dt_etfcsolitonplp) == 2.4e-4
    assert eng.sens_id == 19


def test_m465_eng_energy_free(tmp_path: Path):
    deck = """# RADIOSS ENGINE DECK
/BEGIN
Test M465 Eng Output Free
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSOLITONPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
3.4e-4, 22
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralsolitonplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralsolitonplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfcsolitonplp) == 3.4e-4
    assert eng.sens_id == 22


def test_m465_eng_energy_aliases(tmp_path: Path):
    aliases = [
        "/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_SOLITON_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALSOLITONPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSOLITONPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALSOLITONPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOSOLITONCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_SOLITON_CHIRAL_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOSOLITONCHIRALPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOSOLITONCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOSOLITONCHIRALPLASMONICPOLARITONIC_RESONANCE",
    ]
    deck_lines = ["# RADIOSS ENGINE DECK", "/BEGIN", "Test Eng Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "4.4e-4, 29",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.eng_electrothermoflexomagnetochiralsolitonplasmonicpolaritonic_resonance_energies
        eng = model.eng_electrothermoflexomagnetochiralsolitonplasmonicpolaritonic_resonance_energies[idx]
        assert pytest.approx(eng.dt_etfcsolitonplp) == 4.4e-4


# ============================================================================
# 3. LAGMUL_MAJORANA_SPINOR_SPATIAL_LINKAGE_JOINT Tests
# ============================================================================

def test_m465_majorana_spinor_joint_fixed(tmp_path: Path):
    c1 = f"{140:>10d}{141:>10d}{142:>10d}{3.6e6:>20.4f}{14:>10d}{1.7e-5:>20.6f}"
    c2 = f"{21.5:>20.4f}{28.0:>20.4f}{75.0:>20.4f}{14.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M465 Majorana Joint Fixed
2022 0
/MAJORANA_SPINOR_SPATIAL_LINKAGE_JOINT/1
Majorana Spinor Joint Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_majorana_spinor_spatial_linkage_joints
    j = model.lagmul_majorana_spinor_spatial_linkage_joints[1]
    assert isinstance(j, LagmulMajoranaSpinorSpatialLinkageJoint)
    assert j.node1 == 140
    assert j.node2 == 141
    assert j.node3 == 142
    assert pytest.approx(j.stiff) == 3.6e6
    assert j.skew_id == 14
    assert pytest.approx(j.tol) == 1.7e-5
    assert pytest.approx(j.link_len_a) == 21.5
    assert pytest.approx(j.link_len_b) == 28.0
    assert pytest.approx(j.twist_angle_alpha) == 75.0
    assert pytest.approx(j.offset_distance_s) == 14.5

    # Offset aliases
    assert pytest.approx(j.offset_distance_r) == 14.5
    assert pytest.approx(j.offset_distance_v) == 14.5
    assert pytest.approx(j.offset_distance_h) == 14.5
    assert pytest.approx(j.offset_distance_u) == 14.5
    assert pytest.approx(j.offset_distance_f) == 14.5


def test_m465_majorana_spinor_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M465 Majorana Joint Free
/LAGMUL/MAJORANA_SPINOR_SPATIAL_LINKAGE_JOINT/2
240, 241, 242, 4.6e6, 15, 2.7e-5
24.0, 30.5, 90.0, 16.2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_majorana_spinor_spatial_linkage_joints
    j = model.lagmul_majorana_spinor_spatial_linkage_joints[2]
    assert j.node1 == 240
    assert j.node2 == 241
    assert j.node3 == 242
    assert pytest.approx(j.stiff) == 4.6e6
    assert j.skew_id == 15
    assert pytest.approx(j.tol) == 2.7e-5
    assert pytest.approx(j.link_len_a) == 24.0
    assert pytest.approx(j.link_len_b) == 30.5
    assert pytest.approx(j.twist_angle_alpha) == 90.0
    assert pytest.approx(j.offset_distance_s) == 16.2


def test_m465_majorana_spinor_joint_aliases(tmp_path: Path):
    aliases = [
        "/MAJORANA_SPINOR_SPATIAL_LINKAGE",
        "/LAGMUL/MAJORANA_SPINOR_SPATIAL_LINKAGE",
        "/MAJORANA_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM",
        "/MAJORANA_SPINOR_SPATIAL_SYMMETRIC_MECHANISM",
        "/MAJORANA_SPINOR_SPATIAL_6R_MECHANISM",
        "/MAJORANA_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM",
        "/LAGMUL/MAJORANA_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/MAJORANA_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/MAJORANA_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/MAJORANA_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/MAJORANA_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/MAJORANA_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/PIN_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/PIN_SPINOR_SPATIAL_LINKAGE_JOINT",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test Majorana Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "340, 341, 342, 1.3e6, 0, 1e-6",
            "18.0, 18.0, 0.0, 0.0",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.lagmul_majorana_spinor_spatial_linkage_joints
        j = model.lagmul_majorana_spinor_spatial_linkage_joints[idx]
        assert j.node1 == 340


# ============================================================================
# 4. SENSOR_SPRING_TOTAL_ANGULAR_LOCK_RATE Tests
# ============================================================================

def test_m465_sensor_spring_total_angular_lock_rate_fixed(tmp_path: Path):
    c1 = f"{16:>10d}{1.94e5:>20.4f}{0.0029:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Angular Lock Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE/1
Total Angular Lock Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_angular_lock_rates
    s = model.sensor_spring_total_angular_lock_rates[1]
    assert isinstance(s, SensorSpringTotalAngularLockRate)
    assert s.spring_id == 16
    assert pytest.approx(s.jtot_ang_lock_max) == 1.94e5
    assert pytest.approx(s.t_delay) == 0.0029

    # Check property aliases
    assert pytest.approx(s.j_lock_tot_ang_max) == 1.94e5
    assert pytest.approx(s.j_tot_ang_lock_max) == 1.94e5
    assert pytest.approx(s.jtot_ang_snap_max) == 1.94e5
    assert pytest.approx(s.jtot_ang_rate_max) == 1.94e5
    assert pytest.approx(s.jtot_ang_roc_rate_max) == 1.94e5
    assert pytest.approx(s.jtot_ang_drop_rate_max) == 1.94e5
    assert pytest.approx(s.jtot_ang_crk_rate_max) == 1.94e5
    assert pytest.approx(s.jtot_ang_pop_rate_max) == 1.94e5
    assert pytest.approx(s.jtot_ang_lock_rate_max) == 1.94e5
    assert pytest.approx(s.jtotal_angular_lock_max) == 1.94e5
    assert pytest.approx(s.j_total_angular_lock_max) == 1.94e5
    assert pytest.approx(s.j_lock_total_angular_max) == 1.94e5
    assert pytest.approx(s.jtotal_lock_max) == 1.94e5
    assert pytest.approx(s.j_total_lock_max) == 1.94e5
    assert pytest.approx(s.j_lock_total_max) == 1.94e5


def test_m465_sensor_spring_total_angular_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Angular Lock Rate Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE/2
26, 2.94e5, 0.0039
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_angular_lock_rates
    s = model.sensor_spring_total_angular_lock_rates[2]
    assert s.spring_id == 26
    assert pytest.approx(s.jtot_ang_lock_max) == 2.94e5
    assert pytest.approx(s.t_delay) == 0.0039


def test_m465_sensor_spring_total_angular_lock_rate_aliases(tmp_path: Path):
    aliases = [
        "/SENSOR/SPRING_TOT_ANG_LOCK_RATE",
        "/SENSOR/SPRING_TOTAL_ANGULAR_LOCK",
        "/SENSOR/SPRING_TOT_ANG_LOCK",
        "/SENSOR/SPRING_ANGULAR_TOTAL_LOCK_RATE",
        "/SENSOR/SPRING_ANGULAR_TOT_LOCK_RATE",
        "/SENSOR/SPRING_POP_RATE_TOTAL_ANGULAR_LOCK",
        "/SENSOR/SPRING_LOCK_RATE_TOTAL_ANGULAR",
        "/SENSOR/SPRING_LOCK_RATE_TOT_ANG",
        "/SENSOR/SPRING_LOCK_RATE_ANGULAR_TOT",
        "/SENSOR/TOT_ANG_LOCK_RATE_SPRING",
        "/SENSOR/ANGULAR_TOTAL_LOCK_RATE_SPRING",
        "/SENSOR/SPRING_LOCK_TOTAL_ANGULAR",
        "/SENSOR/SPRING_LOCK_ANGULAR_TOT",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Sensor Spring Total Angular Lock Rate Aliases Test"]
    for sid, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{sid}",
            "39, 3.9e5, 0.0017",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(10, 10 + len(aliases)):
        assert sid in model.sensor_spring_total_angular_lock_rates
        s = model.sensor_spring_total_angular_lock_rates[sid]
        assert s.spring_id == 39
        assert pytest.approx(s.jtot_ang_lock_max) == 3.9e5


# ============================================================================
# 5. Error handling tests
# ============================================================================

def test_m465_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_TENSION_FAILURE_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSOLITONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/MAJORANA_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4

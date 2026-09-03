"""Tests for Milestone M461:
- /FAIL/LAD_DYNAMIC_INTERLAMINAR_SHEAR_FAILURE_RATE (and aliases)
- /ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY (and aliases)
- /LAGMUL/POISSON_SPINOR_SPATIAL_LINKAGE_JOINT / /POISSON_SPINOR_SPATIAL_LINKAGE_JOINT (and aliases)
- /SENSOR/SPRING_NORMAL_LOCK_RATE (and aliases)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailLadDynamicInterlaminarShearFailureRate,
    EngElectrothermoflexomagnetochiralskyrmionplasmonicpolaritonicResonanceEnergy,
    LagmulPoissonSpinorSpatialLinkageJoint,
    SensorSpringNormalLockRate,
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
# 1. FAIL_LAD_DYNAMIC_INTERLAMINAR_SHEAR_FAILURE_RATE Tests
# ============================================================================

def test_m461_fail_lad_dynamic_interlaminar_shear_failure_rate_fixed(tmp_path: Path):
    c1 = f"{75.0:>20.4f}{280.0:>20.4f}{0.14:>20.4f}{1.32:>20.4f}{0.91:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M461 Fail Lad Dynamic Interlaminar Shear Failure Rate Fixed
2022 0
/FAIL/LAD_DYNAMIC_INTERLAMINAR_SHEAR_FAILURE_RATE/1
Dynamic Interlaminar Shear Failure Rate Card Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.fail_laddynamicinterlaminarshearfailurerates
    fm = model.fail_laddynamicinterlaminarshearfailurerates[1]
    assert isinstance(fm, FailLadDynamicInterlaminarShearFailureRate)
    assert fm.mat_id == 1
    assert pytest.approx(fm.sigma_disfr0) == 75.0
    assert pytest.approx(fm.sigma_disfrc) == 280.0
    assert pytest.approx(fm.gamma_disfr) == 0.14
    assert pytest.approx(fm.p_disfr) == 1.32
    assert pytest.approx(fm.d_disfr_max) == 0.91
    assert fm.ifail_sh == 2
    assert fm.ifail_so == 1

    # Properties
    assert pytest.approx(fm.sigma_disf0) == 75.0
    assert pytest.approx(fm.sigma_disfc) == 280.0
    assert pytest.approx(fm.gamma_disf) == 0.14
    assert pytest.approx(fm.p_disf) == 1.32
    assert pytest.approx(fm.d_disf_max) == 0.91

    assert pytest.approx(fm.sigma_dis0) == 75.0
    assert pytest.approx(fm.sigma_disc) == 280.0
    assert pytest.approx(fm.gamma_dis) == 0.14
    assert pytest.approx(fm.p_dis) == 1.32
    assert pytest.approx(fm.d_dis_max) == 0.91

    assert pytest.approx(fm.sigma_dir0) == 75.0
    assert pytest.approx(fm.sigma_dirc) == 280.0
    assert pytest.approx(fm.gamma_dir) == 0.14
    assert pytest.approx(fm.p_dir) == 1.32
    assert pytest.approx(fm.d_dir_max) == 0.91


def test_m461_fail_lad_dynamic_interlaminar_shear_failure_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M461 Fail Lad Dynamic Interlaminar Shear Failure Rate Free
/FAIL/LAD_DYNAMIC_INTERLAMINAR_SHEAR_FAILURE_RATE/2
85.0, 310.0, 0.19, 1.42, 0.94
1, 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.fail_laddynamicinterlaminarshearfailurerates
    fm = model.fail_laddynamicinterlaminarshearfailurerates[2]
    assert pytest.approx(fm.sigma_disfr0) == 85.0
    assert pytest.approx(fm.sigma_disfrc) == 310.0
    assert pytest.approx(fm.gamma_disfr) == 0.19
    assert pytest.approx(fm.p_disfr) == 1.42
    assert pytest.approx(fm.d_disfr_max) == 0.94
    assert fm.ifail_sh == 1
    assert fm.ifail_so == 2


def test_m461_fail_lad_dynamic_interlaminar_shear_failure_rate_aliases(tmp_path: Path):
    aliases = [
        "/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_SHEAR_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_INTERLAMINAR_SHEAR_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_SHEAR_FAILURE",
        "/FAIL/LAD_DYNAMIC_INTERLAMINAR_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_FAILURE",
        "/FAIL/LAD_DYNAMIC_INTERLAMINAR_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_INTERLAMINAR_DELAMINATION_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_DELAMINATION_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_INTERLAMINAR_SHEAR_DELAMINATION_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_SHEAR_DELAMINATION_FAILURE_RATE",
        "/FAIL/LAD_DISFR",
        "/FAIL/LAD_DISFR_MODEL",
        "/FAIL/LAD_DISFR_LAW",
        "/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_INTERLAMINAR_SHEAR_FAILURE",
        "/FAIL/LAD_DYNAMIC_INTERLAMINAR_SHEAR_DAMAGE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_SHEAR_DAMAGE_RATE",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test M461 Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "70.0, 260.0, 0.11, 1.15, 0.97",
            "1, 1",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.fail_laddynamicinterlaminarshearfailurerates
        fm = model.fail_laddynamicinterlaminarshearfailurerates[idx]
        assert pytest.approx(fm.sigma_disfr0) == 70.0


# ============================================================================
# 2. ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY Tests
# ============================================================================

def test_m461_eng_energy_fixed(tmp_path: Path):
    c1 = f"{1.7e-4:>20.6f}{15:>10d}"
    deck = f"""# RADIOSS ENGINE DECK
/BEGIN
Test M461 Eng Output Fixed
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Skyrmion Resonance Energy Directive Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralskyrmionplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralskyrmionplasmonicpolaritonic_resonance_energies[1]
    assert isinstance(eng, EngElectrothermoflexomagnetochiralskyrmionplasmonicpolaritonicResonanceEnergy)
    assert pytest.approx(eng.dt_etfcskyrmionplp) == 1.7e-4
    assert eng.sens_id == 15


def test_m461_eng_energy_free(tmp_path: Path):
    deck = """# RADIOSS ENGINE DECK
/BEGIN
Test M461 Eng Output Free
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
2.7e-4, 18
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralskyrmionplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralskyrmionplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfcskyrmionplp) == 2.7e-4
    assert eng.sens_id == 18


def test_m461_eng_energy_aliases(tmp_path: Path):
    aliases = [
        "/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_SKYRMION_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOSKYRMIONCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_SKYRMION_CHIRAL_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOSKYRMIONCHIRALPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOSKYRMIONCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOSKYRMIONCHIRALPLASMONICPOLARITONIC_RESONANCE",
    ]
    deck_lines = ["# RADIOSS ENGINE DECK", "/BEGIN", "Test Eng Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "3.7e-4, 25",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.eng_electrothermoflexomagnetochiralskyrmionplasmonicpolaritonic_resonance_energies
        eng = model.eng_electrothermoflexomagnetochiralskyrmionplasmonicpolaritonic_resonance_energies[idx]
        assert pytest.approx(eng.dt_etfcskyrmionplp) == 3.7e-4


# ============================================================================
# 3. LAGMUL_POISSON_SPINOR_SPATIAL_LINKAGE_JOINT Tests
# ============================================================================

def test_m461_poisson_spinor_joint_fixed(tmp_path: Path):
    c1 = f"{114:>10d}{115:>10d}{116:>10d}{2.8e6:>20.4f}{8:>10d}{1.2e-5:>20.6f}"
    c2 = f"{17.5:>20.4f}{24.0:>20.4f}{55.0:>20.4f}{10.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M461 Poisson Joint Fixed
2022 0
/POISSON_SPINOR_SPATIAL_LINKAGE_JOINT/1
Poisson Spinor Joint Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_poisson_spinor_spatial_linkage_joints
    j = model.lagmul_poisson_spinor_spatial_linkage_joints[1]
    assert isinstance(j, LagmulPoissonSpinorSpatialLinkageJoint)
    assert j.node1 == 114
    assert j.node2 == 115
    assert j.node3 == 116
    assert pytest.approx(j.stiff) == 2.8e6
    assert j.skew_id == 8
    assert pytest.approx(j.tol) == 1.2e-5
    assert pytest.approx(j.link_len_a) == 17.5
    assert pytest.approx(j.link_len_b) == 24.0
    assert pytest.approx(j.twist_angle_alpha) == 55.0
    assert pytest.approx(j.offset_distance_s) == 10.5

    # Offset aliases
    assert pytest.approx(j.offset_distance_r) == 10.5
    assert pytest.approx(j.offset_distance_v) == 10.5
    assert pytest.approx(j.offset_distance_h) == 10.5
    assert pytest.approx(j.offset_distance_u) == 10.5
    assert pytest.approx(j.offset_distance_f) == 10.5


def test_m461_poisson_spinor_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M461 Poisson Joint Free
/LAGMUL/POISSON_SPINOR_SPATIAL_LINKAGE_JOINT/2
214, 215, 216, 3.8e6, 9, 2.2e-5
20.0, 26.5, 70.0, 12.2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_poisson_spinor_spatial_linkage_joints
    j = model.lagmul_poisson_spinor_spatial_linkage_joints[2]
    assert j.node1 == 214
    assert j.node2 == 215
    assert j.node3 == 216
    assert pytest.approx(j.stiff) == 3.8e6
    assert j.skew_id == 9
    assert pytest.approx(j.tol) == 2.2e-5
    assert pytest.approx(j.link_len_a) == 20.0
    assert pytest.approx(j.link_len_b) == 26.5
    assert pytest.approx(j.twist_angle_alpha) == 70.0
    assert pytest.approx(j.offset_distance_s) == 12.2


def test_m461_poisson_spinor_joint_aliases(tmp_path: Path):
    aliases = [
        "/POISSON_SPINOR_SPATIAL_LINKAGE",
        "/LAGMUL/POISSON_SPINOR_SPATIAL_LINKAGE",
        "/POISSON_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM",
        "/POISSON_SPINOR_SPATIAL_SYMMETRIC_MECHANISM",
        "/POISSON_SPINOR_SPATIAL_6R_MECHANISM",
        "/POISSON_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM",
        "/LAGMUL/POISSON_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/POISSON_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/POISSON_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/POISSON_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/POISSON_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/POISSON_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/METAPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/METAPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test Poisson Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "314, 315, 316, 1.0e6, 0, 1e-6",
            "14.0, 14.0, 0.0, 0.0",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.lagmul_poisson_spinor_spatial_linkage_joints
        j = model.lagmul_poisson_spinor_spatial_linkage_joints[idx]
        assert j.node1 == 314


# ============================================================================
# 4. SENSOR_SPRING_NORMAL_LOCK_RATE Tests
# ============================================================================

def test_m461_sensor_spring_normal_lock_rate_fixed(tmp_path: Path):
    c1 = f"{12:>10d}{1.94e5:>20.4f}{0.0024:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Lock Rate Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_LOCK_RATE/1
Normal Lock Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_lock_rates
    s = model.sensor_spring_normal_lock_rates[1]
    assert isinstance(s, SensorSpringNormalLockRate)
    assert s.spring_id == 12
    assert pytest.approx(s.jnorm_lock_max) == 1.94e5
    assert pytest.approx(s.t_delay) == 0.0024

    # Check property aliases
    assert pytest.approx(s.j_lock_norm_max) == 1.94e5
    assert pytest.approx(s.j_norm_lock_max) == 1.94e5
    assert pytest.approx(s.jnorm_shot_max) == 1.94e5
    assert pytest.approx(s.jnorm_drop_max) == 1.94e5
    assert pytest.approx(s.jnorm_snp_max) == 1.94e5
    assert pytest.approx(s.jnorm_snap_max) == 1.94e5
    assert pytest.approx(s.jnorm_rate_max) == 1.94e5
    assert pytest.approx(s.jnorm_roc_rate_max) == 1.94e5
    assert pytest.approx(s.jnorm_drop_rate_max) == 1.94e5
    assert pytest.approx(s.jnorm_crk_rate_max) == 1.94e5
    assert pytest.approx(s.jnorm_pop_rate_max) == 1.94e5
    assert pytest.approx(s.jnorm_lock_rate_max) == 1.94e5
    assert pytest.approx(s.jnorm_crackle_max) == 1.94e5
    assert pytest.approx(s.jnorm_crk_max) == 1.94e5
    assert pytest.approx(s.jaxi_lock_max) == 1.94e5
    assert pytest.approx(s.j_axi_lock_max) == 1.94e5
    assert pytest.approx(s.j_lock_axi_max) == 1.94e5
    assert pytest.approx(s.jaxial_lock_max) == 1.94e5
    assert pytest.approx(s.j_axial_lock_max) == 1.94e5
    assert pytest.approx(s.j_lock_axial_max) == 1.94e5


def test_m461_sensor_spring_normal_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Normal Lock Rate Free Format Test
/SENSOR/SPRING_NORMAL_LOCK_RATE/2
22, 2.94e5, 0.0034
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_normal_lock_rates
    s = model.sensor_spring_normal_lock_rates[2]
    assert s.spring_id == 22
    assert pytest.approx(s.jnorm_lock_max) == 2.94e5
    assert pytest.approx(s.t_delay) == 0.0034


def test_m461_sensor_spring_normal_lock_rate_aliases(tmp_path: Path):
    aliases = [
        "/SENSOR/SPRING_NORM_LOCK_RATE",
        "/SENSOR/SPRING_NORMAL_LOCK",
        "/SENSOR/SPRING_NORM_LOCK",
        "/SENSOR/SPRING_AXIAL_LOCK_RATE",
        "/SENSOR/SPRING_AXIAL_LOCK",
        "/SENSOR/SPRING_POP_RATE_NORMAL_LOCK",
        "/SENSOR/SPRING_LOCK_RATE_NORMAL",
        "/SENSOR/SPRING_LOCK_RATE_NORM",
        "/SENSOR/SPRING_LOCK_RATE_AXIAL",
        "/SENSOR/NORMAL_LOCK_RATE_SPRING",
        "/SENSOR/NORM_LOCK_RATE_SPRING",
        "/SENSOR/AXIAL_LOCK_RATE_SPRING",
        "/SENSOR/SPRING_LOCK_NORMAL",
        "/SENSOR/SPRING_LOCK_NORM",
        "/SENSOR/SPRING_LOCK_AXIAL",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Sensor Spring Normal Lock Rate Aliases Test"]
    for sid, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{sid}",
            "35, 3.4e5, 0.0012",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(10, 10 + len(aliases)):
        assert sid in model.sensor_spring_normal_lock_rates
        s = model.sensor_spring_normal_lock_rates[sid]
        assert s.spring_id == 35
        assert pytest.approx(s.jnorm_lock_max) == 3.4e5


# ============================================================================
# 5. Error handling tests
# ============================================================================

def test_m461_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_DYNAMIC_INTERLAMINAR_SHEAR_FAILURE_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/POISSON_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_NORMAL_LOCK_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4

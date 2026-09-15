"""Tests for Milestone M464:
- /FAIL/LAD_DYNAMIC_INTERLAMINAR_TENSION_FAILURE_RATE (and aliases)
- /ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALINSTANTONPLASMONICPOLARITONIC_RESONANCE_ENERGY (and aliases)
- /LAGMUL/CARTAN_SPINOR_SPATIAL_LINKAGE_JOINT / /CARTAN_SPINOR_SPATIAL_LINKAGE_JOINT (and aliases)
- /SENSOR/SPRING_TORSIONAL_LOCK_RATE (and aliases)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailLadDynamicInterlaminarTensionFailureRate,
    EngElectrothermoflexomagnetochiralinstantonplasmonicpolaritonicResonanceEnergy,
    LagmulCartanSpinorSpatialLinkageJoint,
    SensorSpringTorsionalLockRate,
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
# 1. FAIL_LAD_DYNAMIC_INTERLAMINAR_TENSION_FAILURE_RATE Tests
# ============================================================================

def test_m464_fail_lad_dynamic_interlaminar_tension_failure_rate_fixed(tmp_path: Path):
    c1 = f"{85.0:>20.4f}{310.0:>20.4f}{0.16:>20.4f}{1.40:>20.4f}{0.94:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M464 Fail Lad Dynamic Interlaminar Tension Failure Rate Fixed
2022 0
/FAIL/LAD_DYNAMIC_INTERLAMINAR_TENSION_FAILURE_RATE/1
Dynamic Interlaminar Tension Failure Rate Card Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.fail_laddynamicinterlaminartensionfailurerates
    fm = model.fail_laddynamicinterlaminartensionfailurerates[1]
    assert isinstance(fm, FailLadDynamicInterlaminarTensionFailureRate)
    assert fm.mat_id == 1
    assert pytest.approx(fm.sigma_ditfr0) == 85.0
    assert pytest.approx(fm.sigma_ditfrc) == 310.0
    assert pytest.approx(fm.gamma_ditfr) == 0.16
    assert pytest.approx(fm.p_ditfr) == 1.40
    assert pytest.approx(fm.d_ditfr_max) == 0.94
    assert fm.ifail_sh == 2
    assert fm.ifail_so == 1

    # Properties
    assert pytest.approx(fm.sigma_ditf0) == 85.0
    assert pytest.approx(fm.sigma_ditfc) == 310.0
    assert pytest.approx(fm.gamma_ditf) == 0.16
    assert pytest.approx(fm.p_ditf) == 1.40
    assert pytest.approx(fm.d_ditf_max) == 0.94

    assert pytest.approx(fm.sigma_dit0) == 85.0
    assert pytest.approx(fm.sigma_ditc) == 310.0
    assert pytest.approx(fm.gamma_dit) == 0.16
    assert pytest.approx(fm.p_dit) == 1.40
    assert pytest.approx(fm.d_dit_max) == 0.94

    assert pytest.approx(fm.sigma_din0) == 85.0
    assert pytest.approx(fm.sigma_dinc) == 310.0
    assert pytest.approx(fm.gamma_din) == 0.16
    assert pytest.approx(fm.p_din) == 1.40
    assert pytest.approx(fm.d_din_max) == 0.94


def test_m464_fail_lad_dynamic_interlaminar_tension_failure_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M464 Fail Lad Dynamic Interlaminar Tension Failure Rate Free
/FAIL/LAD_DYNAMIC_INTERLAMINAR_TENSION_FAILURE_RATE/2
95.0, 330.0, 0.19, 1.50, 0.97
1, 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.fail_laddynamicinterlaminartensionfailurerates
    fm = model.fail_laddynamicinterlaminartensionfailurerates[2]
    assert pytest.approx(fm.sigma_ditfr0) == 95.0
    assert pytest.approx(fm.sigma_ditfrc) == 330.0
    assert pytest.approx(fm.gamma_ditfr) == 0.19
    assert pytest.approx(fm.p_ditfr) == 1.50
    assert pytest.approx(fm.d_ditfr_max) == 0.97
    assert fm.ifail_sh == 1
    assert fm.ifail_so == 2


def test_m464_fail_lad_dynamic_interlaminar_tension_failure_rate_aliases(tmp_path: Path):
    aliases = [
        "/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_TENSION_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_INTERLAMINAR_TENSION_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_TENSION_FAILURE",
        "/FAIL/LAD_DYNAMIC_INTERLAMINAR_TENSIONAL_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_TENSIONAL_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_INTERLAMINAR_TENSIONAL_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_TENSIONAL_FAILURE",
        "/FAIL/LAD_DYNAMIC_INTERLAMINAR_NORMAL_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_NORMAL_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_INTERLAMINAR_NORMAL_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_NORMAL_FAILURE",
        "/FAIL/LAD_DYNAMIC_INTERLAMINAR_NORMAL_PEELING_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_NORMAL_PEELING_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_INTERLAMINAR_PEELING_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_PEELING_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_INTERLAMINAR_PEELING_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_PEELING_FAILURE",
        "/FAIL/LAD_DITFR",
        "/FAIL/LAD_DITFR_MODEL",
        "/FAIL/LAD_DITFR_LAW",
        "/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_INTERLAMINAR_TENSION_FAILURE",
        "/FAIL/LAD_DYNAMIC_INTERLAMINAR_TENSION_DAMAGE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_TENSION_DAMAGE_RATE",
        "/FAIL/LAD_DYNAMIC_INTERLAMINAR_NORMAL_DAMAGE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_NORMAL_DAMAGE_RATE",
        "/FAIL/LAD_DYNAMIC_INTERLAMINAR_PEELING_DAMAGE_RATE",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test M464 Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "76.0, 280.0, 0.14, 1.25, 0.96",
            "1, 1",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.fail_laddynamicinterlaminartensionfailurerates
        fm = model.fail_laddynamicinterlaminartensionfailurerates[idx]
        assert pytest.approx(fm.sigma_ditfr0) == 76.0


# ============================================================================
# 2. ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALINSTANTONPLASMONICPOLARITONIC_RESONANCE_ENERGY Tests
# ============================================================================

def test_m464_eng_energy_fixed(tmp_path: Path):
    c1 = f"{2.3e-4:>20.6f}{18:>10d}"
    deck = f"""# RADIOSS ENGINE DECK
/BEGIN
Test M464 Eng Output Fixed
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALINSTANTONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Instanton Resonance Energy Directive Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralinstantonplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralinstantonplasmonicpolaritonic_resonance_energies[1]
    assert isinstance(eng, EngElectrothermoflexomagnetochiralinstantonplasmonicpolaritonicResonanceEnergy)
    assert pytest.approx(eng.dt_etfcinstantonplp) == 2.3e-4
    assert eng.sens_id == 18


def test_m464_eng_energy_free(tmp_path: Path):
    deck = """# RADIOSS ENGINE DECK
/BEGIN
Test M464 Eng Output Free
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALINSTANTONPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
3.3e-4, 21
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralinstantonplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralinstantonplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfcinstantonplp) == 3.3e-4
    assert eng.sens_id == 21


def test_m464_eng_energy_aliases(tmp_path: Path):
    aliases = [
        "/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_INSTANTON_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALINSTANTONPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALINSTANTONPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALINSTANTONPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOINSTANTONCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_INSTANTON_CHIRAL_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOINSTANTONCHIRALPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOINSTANTONCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOINSTANTONCHIRALPLASMONICPOLARITONIC_RESONANCE",
    ]
    deck_lines = ["# RADIOSS ENGINE DECK", "/BEGIN", "Test Eng Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "4.3e-4, 28",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.eng_electrothermoflexomagnetochiralinstantonplasmonicpolaritonic_resonance_energies
        eng = model.eng_electrothermoflexomagnetochiralinstantonplasmonicpolaritonic_resonance_energies[idx]
        assert pytest.approx(eng.dt_etfcinstantonplp) == 4.3e-4


# ============================================================================
# 3. LAGMUL_CARTAN_SPINOR_SPATIAL_LINKAGE_JOINT Tests
# ============================================================================

def test_m464_cartan_spinor_joint_fixed(tmp_path: Path):
    c1 = f"{130:>10d}{131:>10d}{132:>10d}{3.5e6:>20.4f}{12:>10d}{1.6e-5:>20.6f}"
    c2 = f"{20.5:>20.4f}{27.0:>20.4f}{70.0:>20.4f}{13.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M464 Cartan Joint Fixed
2022 0
/CARTAN_SPINOR_SPATIAL_LINKAGE_JOINT/1
Cartan Spinor Joint Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_cartan_spinor_spatial_linkage_joints
    j = model.lagmul_cartan_spinor_spatial_linkage_joints[1]
    assert isinstance(j, LagmulCartanSpinorSpatialLinkageJoint)
    assert j.node1 == 130
    assert j.node2 == 131
    assert j.node3 == 132
    assert pytest.approx(j.stiff) == 3.5e6
    assert j.skew_id == 12
    assert pytest.approx(j.tol) == 1.6e-5
    assert pytest.approx(j.link_len_a) == 20.5
    assert pytest.approx(j.link_len_b) == 27.0
    assert pytest.approx(j.twist_angle_alpha) == 70.0
    assert pytest.approx(j.offset_distance_s) == 13.5

    # Offset aliases
    assert pytest.approx(j.offset_distance_r) == 13.5
    assert pytest.approx(j.offset_distance_v) == 13.5
    assert pytest.approx(j.offset_distance_h) == 13.5
    assert pytest.approx(j.offset_distance_u) == 13.5
    assert pytest.approx(j.offset_distance_f) == 13.5


def test_m464_cartan_spinor_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M464 Cartan Joint Free
/LAGMUL/CARTAN_SPINOR_SPATIAL_LINKAGE_JOINT/2
230, 231, 232, 4.5e6, 13, 2.6e-5
23.0, 29.5, 85.0, 15.2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_cartan_spinor_spatial_linkage_joints
    j = model.lagmul_cartan_spinor_spatial_linkage_joints[2]
    assert j.node1 == 230
    assert j.node2 == 231
    assert j.node3 == 232
    assert pytest.approx(j.stiff) == 4.5e6
    assert j.skew_id == 13
    assert pytest.approx(j.tol) == 2.6e-5
    assert pytest.approx(j.link_len_a) == 23.0
    assert pytest.approx(j.link_len_b) == 29.5
    assert pytest.approx(j.twist_angle_alpha) == 85.0
    assert pytest.approx(j.offset_distance_s) == 15.2


def test_m464_cartan_spinor_joint_aliases(tmp_path: Path):
    aliases = [
        "/CARTAN_SPINOR_SPATIAL_LINKAGE",
        "/LAGMUL/CARTAN_SPINOR_SPATIAL_LINKAGE",
        "/CARTAN_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM",
        "/CARTAN_SPINOR_SPATIAL_SYMMETRIC_MECHANISM",
        "/CARTAN_SPINOR_SPATIAL_6R_MECHANISM",
        "/CARTAN_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM",
        "/LAGMUL/CARTAN_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/CARTAN_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/CARTAN_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/CARTAN_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/CARTAN_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/CARTAN_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/WEYL_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/WEYL_SPINOR_SPATIAL_LINKAGE_JOINT",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Test Cartan Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            "330, 331, 332, 1.2e6, 0, 1e-6",
            "17.0, 17.0, 0.0, 0.0",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.lagmul_cartan_spinor_spatial_linkage_joints
        j = model.lagmul_cartan_spinor_spatial_linkage_joints[idx]
        assert j.node1 == 330


# ============================================================================
# 4. SENSOR_SPRING_TORSIONAL_LOCK_RATE Tests
# ============================================================================

def test_m464_sensor_spring_torsional_lock_rate_fixed(tmp_path: Path):
    c1 = f"{15:>10d}{1.84e5:>20.4f}{0.0028:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Lock Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_LOCK_RATE/1
Torsional Lock Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_lock_rates
    s = model.sensor_spring_torsional_lock_rates[1]
    assert isinstance(s, SensorSpringTorsionalLockRate)
    assert s.spring_id == 15
    assert pytest.approx(s.jtors_lock_max) == 1.84e5
    assert pytest.approx(s.t_delay) == 0.0028

    # Check property aliases
    assert pytest.approx(s.j_lock_tors_max) == 1.84e5
    assert pytest.approx(s.j_tors_lock_max) == 1.84e5
    assert pytest.approx(s.jtors_snap_max) == 1.84e5
    assert pytest.approx(s.jtors_rate_max) == 1.84e5
    assert pytest.approx(s.jtors_roc_rate_max) == 1.84e5
    assert pytest.approx(s.jtors_drop_rate_max) == 1.84e5
    assert pytest.approx(s.jtors_crk_rate_max) == 1.84e5
    assert pytest.approx(s.jtors_pop_rate_max) == 1.84e5
    assert pytest.approx(s.jtors_lock_rate_max) == 1.84e5
    assert pytest.approx(s.jtorsional_lock_max) == 1.84e5
    assert pytest.approx(s.j_torsional_lock_max) == 1.84e5
    assert pytest.approx(s.j_lock_torsional_max) == 1.84e5
    assert pytest.approx(s.jtorsional_angular_lock_max) == 1.84e5
    assert pytest.approx(s.j_torsional_angular_lock_max) == 1.84e5
    assert pytest.approx(s.j_lock_torsional_angular_max) == 1.84e5


def test_m464_sensor_spring_torsional_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Torsional Lock Rate Free Format Test
/SENSOR/SPRING_TORSIONAL_LOCK_RATE/2
25, 2.84e5, 0.0038
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_torsional_lock_rates
    s = model.sensor_spring_torsional_lock_rates[2]
    assert s.spring_id == 25
    assert pytest.approx(s.jtors_lock_max) == 2.84e5
    assert pytest.approx(s.t_delay) == 0.0038


def test_m464_sensor_spring_torsional_lock_rate_aliases(tmp_path: Path):
    aliases = [
        "/SENSOR/SPRING_TORS_LOCK_RATE",
        "/SENSOR/SPRING_TORSIONAL_LOCK",
        "/SENSOR/SPRING_TORS_LOCK",
        "/SENSOR/SPRING_ANGULAR_TORSIONAL_LOCK_RATE",
        "/SENSOR/SPRING_ANGULAR_TORS_LOCK_RATE",
        "/SENSOR/SPRING_POP_RATE_TORSIONAL_LOCK",
        "/SENSOR/SPRING_LOCK_RATE_TORSIONAL",
        "/SENSOR/SPRING_LOCK_RATE_TORS",
        "/SENSOR/SPRING_LOCK_RATE_ANGULAR_TORS",
        "/SENSOR/TORS_LOCK_RATE_SPRING",
        "/SENSOR/ANGULAR_TORS_LOCK_RATE_SPRING",
        "/SENSOR/SPRING_LOCK_TORSIONAL",
        "/SENSOR/SPRING_LOCK_ANGULAR_TORS",
    ]
    deck_lines = ["# RADIOSS ALIAS DECK", "/BEGIN", "Sensor Spring Torsional Lock Rate Aliases Test"]
    for sid, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{sid}",
            "38, 3.8e5, 0.0016",
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(10, 10 + len(aliases)):
        assert sid in model.sensor_spring_torsional_lock_rates
        s = model.sensor_spring_torsional_lock_rates[sid]
        assert s.spring_id == 38
        assert pytest.approx(s.jtors_lock_max) == 3.8e5


# ============================================================================
# 5. Error handling tests
# ============================================================================

def test_m464_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_DYNAMIC_INTERLAMINAR_TENSION_FAILURE_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALINSTANTONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/CARTAN_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TORSIONAL_LOCK_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4

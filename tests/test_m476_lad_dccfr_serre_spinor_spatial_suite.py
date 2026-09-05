"""Tests for Milestone M476:
- /FAIL/LAD_DYNAMIC_CORE_CRUSHING_FAILURE_RATE (and aliases)
- /ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANTIMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY (and aliases)
- /LAGMUL/SERRE_SPINOR_SPATIAL_LINKAGE_JOINT / /SERRE_SPINOR_SPATIAL_LINKAGE_JOINT (and aliases)
- /SENSOR/SPRING_TORSIONAL_CRACKLE_RATE (and aliases)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailLadDynamicCoreCrushingFailureRate,
    EngElectrothermoflexomagnetochiralantimeronplasmonicpolaritonicResonanceEnergy,
    LagmulSerreSpinorSpatialLinkageJoint,
    SensorSpringTorsionalCrackleRate,
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
# 1. FAIL_LAD_DYNAMIC_CORE_CRUSHING_FAILURE_RATE Tests
# ============================================================================

def test_m476_fail_lad_dynamic_core_crushing_failure_rate_fixed(tmp_path: Path):
    c1 = f"{145.0:>20.4f}{455.0:>20.4f}{0.42:>20.4f}{1.65:>20.4f}{0.992:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M476 Fail Lad Dynamic Core Crushing Failure Rate Fixed
2022 0
/FAIL/LAD_DYNAMIC_CORE_CRUSHING_FAILURE_RATE/1
Dynamic Core Crushing Failure Rate Card Title
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.fail_laddynamiccorecrushingfailurerates
    fm = model.fail_laddynamiccorecrushingfailurerates[1]
    assert isinstance(fm, FailLadDynamicCoreCrushingFailureRate)
    assert fm.mat_id == 1
    assert pytest.approx(fm.sigma_dccfr0) == 145.0
    assert pytest.approx(fm.sigma_dccfrc) == 455.0
    assert pytest.approx(fm.gamma_dccfr) == 0.42
    assert pytest.approx(fm.p_dccfr) == 1.65
    assert pytest.approx(fm.d_dccfr_max) == 0.992
    assert fm.ifail_sh == 2
    assert fm.ifail_so == 1


def test_m476_fail_lad_numeric_title_matches_alpha_title(tmp_path: Path):
    """BUG-08: A numeric title like '123' in fixed-format must not shift fields into data cards."""
    c1 = f"{145.0:>20.4f}{455.0:>20.4f}{0.42:>20.4f}{1.65:>20.4f}{0.992:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test M476 Fail Lad Dynamic Core Crushing Failure Rate Numeric Title
2022 0
/FAIL/LAD_DYNAMIC_CORE_CRUSHING_FAILURE_RATE/1
123
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.fail_laddynamiccorecrushingfailurerates
    fm = model.fail_laddynamiccorecrushingfailurerates[1]
    assert pytest.approx(fm.sigma_dccfr0) == 145.0
    assert pytest.approx(fm.sigma_dccfrc) == 455.0
    assert pytest.approx(fm.gamma_dccfr) == 0.42
    assert pytest.approx(fm.p_dccfr) == 1.65
    assert pytest.approx(fm.d_dccfr_max) == 0.992
    assert fm.ifail_sh == 2
    assert fm.ifail_so == 1


def test_m476_fail_lad_dynamic_core_crushing_failure_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test M476 Fail Lad Dynamic Core Crushing Failure Rate Free
/FAIL/LAD_DYNAMIC_CORE_CRUSHING_FAILURE_RATE/2
150.0, 480.0, 0.55, 1.75, 0.985
1, 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.fail_laddynamiccorecrushingfailurerates
    fm = model.fail_laddynamiccorecrushingfailurerates[2]
    assert pytest.approx(fm.sigma_dccfr0) == 150.0
    assert pytest.approx(fm.sigma_dccfrc) == 480.0
    assert pytest.approx(fm.gamma_dccfr) == 0.55
    assert pytest.approx(fm.p_dccfr) == 1.75
    assert pytest.approx(fm.d_dccfr_max) == 0.985
    assert fm.ifail_sh == 1
    assert fm.ifail_so == 2


def test_m476_fail_lad_dynamic_core_crushing_failure_rate_properties():
    fm = FailLadDynamicCoreCrushingFailureRate(
        mat_id=1,
        sigma_dccfr0=100.0,
        sigma_dccfrc=300.0,
        gamma_dccfr=0.5,
        p_dccfr=1.5,
        d_dccfr_max=0.95,
    )
    # sigma_dccfr0 getters and setters
    assert pytest.approx(fm.sigma_dccf0) == 100.0
    assert pytest.approx(fm.sigma_dcc0) == 100.0
    assert pytest.approx(fm.sigma_dfdccfr0) == 100.0
    assert pytest.approx(fm.sigma_dfdccf0) == 100.0
    assert pytest.approx(fm.sigma_dfdcc0) == 100.0
    fm.sigma_dccf0 = 110.0
    assert pytest.approx(fm.sigma_dccfr0) == 110.0
    fm.sigma_dcc0 = 120.0
    assert pytest.approx(fm.sigma_dccfr0) == 120.0
    fm.sigma_dfdccfr0 = 130.0
    assert pytest.approx(fm.sigma_dccfr0) == 130.0
    fm.sigma_dfdccf0 = 140.0
    assert pytest.approx(fm.sigma_dccfr0) == 140.0
    fm.sigma_dfdcc0 = 150.0
    assert pytest.approx(fm.sigma_dccfr0) == 150.0

    # sigma_dccfrc getters and setters
    assert pytest.approx(fm.sigma_dccfc) == 300.0
    assert pytest.approx(fm.sigma_dccc) == 300.0
    assert pytest.approx(fm.sigma_dfdccfrc) == 300.0
    assert pytest.approx(fm.sigma_dfdccfc) == 300.0
    assert pytest.approx(fm.sigma_dfdccc) == 300.0
    fm.sigma_dccfc = 310.0
    assert pytest.approx(fm.sigma_dccfrc) == 310.0
    fm.sigma_dccc = 320.0
    assert pytest.approx(fm.sigma_dccfrc) == 320.0
    fm.sigma_dfdccfrc = 330.0
    assert pytest.approx(fm.sigma_dccfrc) == 330.0
    fm.sigma_dfdccfc = 340.0
    assert pytest.approx(fm.sigma_dccfrc) == 340.0
    fm.sigma_dfdccc = 350.0
    assert pytest.approx(fm.sigma_dccfrc) == 350.0

    # gamma_dccfr getters and setters
    assert pytest.approx(fm.gamma_dccf) == 0.5
    assert pytest.approx(fm.gamma_dcc) == 0.5
    assert pytest.approx(fm.gamma_dfdccfr) == 0.5
    assert pytest.approx(fm.gamma_dfdccf) == 0.5
    assert pytest.approx(fm.gamma_dfdcc) == 0.5
    fm.gamma_dccf = 0.6
    assert pytest.approx(fm.gamma_dccfr) == 0.6
    fm.gamma_dcc = 0.7
    assert pytest.approx(fm.gamma_dccfr) == 0.7
    fm.gamma_dfdccfr = 0.8
    assert pytest.approx(fm.gamma_dccfr) == 0.8
    fm.gamma_dfdccf = 0.9
    assert pytest.approx(fm.gamma_dccfr) == 0.9
    fm.gamma_dfdcc = 1.0
    assert pytest.approx(fm.gamma_dccfr) == 1.0

    # p_dccfr getters and setters
    assert pytest.approx(fm.p_dccf) == 1.5
    assert pytest.approx(fm.p_dcc) == 1.5
    assert pytest.approx(fm.p_dfdccfr) == 1.5
    assert pytest.approx(fm.p_dfdccf) == 1.5
    assert pytest.approx(fm.p_dfdcc) == 1.5
    fm.p_dccf = 1.6
    assert pytest.approx(fm.p_dccfr) == 1.6
    fm.p_dcc = 1.7
    assert pytest.approx(fm.p_dccfr) == 1.7
    fm.p_dfdccfr = 1.8
    assert pytest.approx(fm.p_dccfr) == 1.8
    fm.p_dfdccf = 1.9
    assert pytest.approx(fm.p_dccfr) == 1.9
    fm.p_dfdcc = 2.0
    assert pytest.approx(fm.p_dccfr) == 2.0

    # d_dccfr_max getters and setters
    assert pytest.approx(fm.d_dccf_max) == 0.95
    assert pytest.approx(fm.d_dcc_max) == 0.95
    assert pytest.approx(fm.d_dfdccfr_max) == 0.95
    assert pytest.approx(fm.d_dfdccf_max) == 0.95
    assert pytest.approx(fm.d_dfdcc_max) == 0.95
    fm.d_dccf_max = 0.96
    assert pytest.approx(fm.d_dccfr_max) == 0.96
    fm.d_dcc_max = 0.97
    assert pytest.approx(fm.d_dccfr_max) == 0.97
    fm.d_dfdccfr_max = 0.98
    assert pytest.approx(fm.d_dccfr_max) == 0.98
    fm.d_dfdccf_max = 0.985
    assert pytest.approx(fm.d_dccfr_max) == 0.985
    fm.d_dfdcc_max = 0.99
    assert pytest.approx(fm.d_dccfr_max) == 0.99


def test_m476_fail_lad_dynamic_core_crushing_failure_rate_material_binding(tmp_path: Path):
    rho_card = f"{7.8e-6:>20.6e}"
    e_nu_card = f"{210000.0:>20.4f}{0.3:>20.4f}"
    c1 = f"{120.0:>20.4f}{380.0:>20.4f}{0.45:>20.4f}{1.4:>20.4f}{0.99:>20.4f}"
    c2 = f"{1:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test Material Binding
2022 0
/MAT/LAW1/10
steel elastic
{rho_card}
{e_nu_card}
/FAIL/LAD_DYNAMIC_CORE_CRUSHING_FAILURE_RATE/10
Core Crushing Card
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 10 in model.materials
    mat = model.materials[10]
    assert len(mat.fail_models) == 1
    fm = mat.fail_models[0]
    assert isinstance(fm, FailLadDynamicCoreCrushingFailureRate)
    assert getattr(mat, "fm_type", None) == "LAD_DCCFR"


def test_m476_fail_lad_dynamic_core_crushing_failure_rate_aliases(tmp_path: Path):
    aliases = [
        "/FAIL/LADEVEZE_DYNAMIC_CORE_CRUSHING_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_CORE_CRUSHING_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_CORE_CRUSHING_FAILURE",
        "/FAIL/LAD_DYNAMIC_CORE_CRUSH_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_CORE_CRUSH_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_CORE_CRUSH_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_CORE_CRUSH_FAILURE",
        "/FAIL/LAD_DYNAMIC_FIBER_DIRECTION_CORE_CRUSHING_FAILURE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_FIBER_DIRECTION_CORE_CRUSHING_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_FIBER_DIRECTION_CORE_CRUSHING_FAILURE",
        "/FAIL/LADEVEZE_DYNAMIC_FIBER_DIRECTION_CORE_CRUSHING_FAILURE",
        "/FAIL/LAD_FIBER_DIRECTION_DYNAMIC_CORE_CRUSHING_FAILURE_RATE",
        "/FAIL/LADEVEZE_FIBER_DIRECTION_DYNAMIC_CORE_CRUSHING_FAILURE_RATE",
        "/FAIL/LAD_DFDCCFR",
        "/FAIL/LAD_DFDCCFR_MODEL",
        "/FAIL/LAD_DFDCCFR_LAW",
        "/FAIL/LAD_DCCFR",
        "/FAIL/LAD_DCCFR_MODEL",
        "/FAIL/LAD_DCCFR_LAW",
        "/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_CORE_CRUSHING_FAILURE",
        "/FAIL/LAD_DYNAMIC_CORE_CRUSHING_DAMAGE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_CORE_CRUSHING_DAMAGE_RATE",
        "/FAIL/LAD_DYNAMIC_CORE_CRUSH_DAMAGE_RATE",
        "/FAIL/LADEVEZE_DYNAMIC_CORE_CRUSH_DAMAGE_RATE",
        "/FAIL/LAD_DYNAMIC_HONEYCOMB_CORE_CRUSHING_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_FOAM_CORE_CRUSHING_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_SANDWICH_CORE_CRUSHING_FAILURE_RATE",
        "/FAIL/LAD_DYNAMIC_CELLULAR_CORE_CRUSHING_FAILURE_RATE",
    ]
    deck_lines = ["# RADIOSS FREE DECK", "/BEGIN", "Test Failure Aliases"]
    for idx, kw in enumerate(aliases, start=20):
        deck_lines.extend([
            f"{kw}/{idx}",
            f"{idx * 5.0}, {idx * 15.0}, 0.25, 1.2, 0.995",
            "1, 1"
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(20, 20 + len(aliases)):
        assert idx in model.fail_laddynamiccorecrushingfailurerates
        fm = model.fail_laddynamiccorecrushingfailurerates[idx]
        assert pytest.approx(fm.sigma_dccfr0) == idx * 5.0
        assert pytest.approx(fm.sigma_dccfrc) == idx * 15.0


def test_m476_fail_lad_dynamic_core_crushing_failure_rate_missing_card_error(tmp_path: Path):
    deck = """/BEGIN
Test Missing Card
/FAIL/LAD_DYNAMIC_CORE_CRUSHING_FAILURE_RATE/50
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
    assert any("missing data card" in err for err in log.errors)


# ============================================================================
# 2. ENG_ELECTROTHERMOFLEXOMAGNETOCHIRALANTIMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY Tests
# ============================================================================

def test_m476_eng_electrothermoflexomagnetochiralantimeronplasmonicpolaritonic_fixed(tmp_path: Path):
    c1 = f"{0.000125:>20.6f}{4:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test Engine Fixed
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANTIMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Engine Directive Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralantimeronplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralantimeronplasmonicpolaritonic_resonance_energies[1]
    assert isinstance(eng, EngElectrothermoflexomagnetochiralantimeronplasmonicpolaritonicResonanceEnergy)
    assert pytest.approx(eng.dt_etfcantimeronplp) == 0.000125
    assert eng.sens_id == 4


def test_m476_eng_electrothermoflexomagnetochiralantimeronplasmonicpolaritonic_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test Engine Free
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANTIMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00035, 7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralantimeronplasmonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetochiralantimeronplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfcantimeronplp) == 0.00035
    assert eng.sens_id == 7


def test_m476_eng_electrothermoflexomagnetochiralantimeronplasmonicpolaritonic_aliases(tmp_path: Path):
    aliases = [
        "/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_ANTIMERON_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALANTIMERONPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANTIMERONPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALANTIMERONPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOANTIMERONCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_ANTIMERON_CHIRAL_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOANTIMERONCHIRALPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOANTIMERONCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOANTIMERONCHIRALPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBISKYRMIONPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_BISKYRMION_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALBISKYRMIONPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALBISKYRMIONPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALBISKYRMIONPLASMONICPOLARITONIC_RESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOBISKYRMIONCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY",
        "/ENG/ELECTRO_THERM_FLEXO_MAG_BISKYRMION_CHIRAL_PLASMON_POLARITON_RES_WORK",
        "/ENG/EELECTROTHERMOFLEXOMAGNETOBISKYRMIONCHIRALPLASMONICPOLARITONICRESONANCE",
        "/ENG/ELECTROTHERMOFLEXOMAGNETOBISKYRMIONCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION",
        "/ENG/ET_ELECTROTHERMOFLEXOMAGNETOBISKYRMIONCHIRALPLASMONICPOLARITONIC_RESONANCE",
    ]
    deck_lines = ["# RADIOSS FREE DECK", "/BEGIN", "Test Engine Aliases"]
    for idx, kw in enumerate(aliases, start=10):
        deck_lines.extend([
            f"{kw}/{idx}",
            f"{idx * 1e-5}, {idx}"
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(10, 10 + len(aliases)):
        assert idx in model.eng_electrothermoflexomagnetochiralantimeronplasmonicpolaritonic_resonance_energies
        eng = model.eng_electrothermoflexomagnetochiralantimeronplasmonicpolaritonic_resonance_energies[idx]
        assert pytest.approx(eng.dt_etfcantimeronplp) == idx * 1e-5
        assert eng.sens_id == idx


def test_m476_eng_electrothermoflexomagnetochiralantimeronplasmonicpolaritonic_cascade():
    # Test fallback from preceding polaritonic directive (antiskyrmion)
    eng1 = EngElectrothermoflexomagnetochiralantimeronplasmonicpolaritonicResonanceEnergy(dt_etfcantiskyrmionplp=0.0075)
    assert pytest.approx(eng1.dt_etfcantimeronplp) == 0.0075

    # Test forward propagation from antimeron to other polaritonic directives
    eng2 = EngElectrothermoflexomagnetochiralantimeronplasmonicpolaritonicResonanceEnergy(dt_etfcantimeronplp=0.0092)
    assert pytest.approx(eng2.dt_etfplp) == 0.0092
    assert pytest.approx(eng2.dt_etfcblochpointplp) == 0.0092
    assert pytest.approx(eng2.dt_etfcbobberplp) == 0.0092
    assert pytest.approx(eng2.dt_etfcskyrmioniumplp) == 0.0092
    assert pytest.approx(eng2.dt_etfcantiskyrmionplp) == 0.0092
    assert pytest.approx(eng2.dt_etfcbiskyrmionplp) == 0.0092


def test_m476_eng_electrothermoflexomagnetochiralantimeronplasmonicpolaritonic_missing_card_error(tmp_path: Path):
    deck = """/BEGIN
Test Missing Card
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALANTIMERONPLASMONICPOLARITONIC_RESONANCE_ENERGY/99
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
    assert any("missing data card" in err for err in log.errors)


# ============================================================================
# 3. LAGMUL_SERRE_SPINOR_SPATIAL_LINKAGE_JOINT Tests
# ============================================================================

def test_m476_lagmul_serre_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{11:>10d}{12:>10d}{13:>10d}{2.5e6:>20.4f}{5:>10d}{1.5e-5:>20.6f}"
    c2 = f"{120.0:>20.4f}{150.0:>20.4f}{45.0:>20.4f}{15.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test Serre Spinor Spatial Linkage Joint Fixed
2022 0
/SERRE_SPINOR_SPATIAL_LINKAGE_JOINT/1
Serre Spinor Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_serre_spinor_spatial_linkage_joints
    j = model.lagmul_serre_spinor_spatial_linkage_joints[1]
    assert isinstance(j, LagmulSerreSpinorSpatialLinkageJoint)
    assert j.node1 == 11
    assert j.node2 == 12
    assert j.node3 == 13
    assert pytest.approx(j.stiff) == 2.5e6
    assert j.skew_id == 5
    assert pytest.approx(j.tol) == 1.5e-5
    assert pytest.approx(j.link_len_a) == 120.0
    assert pytest.approx(j.link_len_b) == 150.0
    assert pytest.approx(j.twist_angle_alpha) == 45.0
    assert pytest.approx(j.offset_distance_s) == 15.5


def test_m476_lagmul_serre_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Test Serre Spinor Spatial Linkage Joint Free
/LAGMUL/SERRE_SPINOR_SPATIAL_LINKAGE_JOINT/2
Joint Free Format
21, 22, 23, 3.5e6, 8, 2.0e-5
80.0, 95.0, 30.0, 8.5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_serre_spinor_spatial_linkage_joints
    j = model.lagmul_serre_spinor_spatial_linkage_joints[2]
    assert j.node1 == 21
    assert j.node2 == 22
    assert j.node3 == 23
    assert pytest.approx(j.stiff) == 3.5e6
    assert j.skew_id == 8
    assert pytest.approx(j.tol) == 2.0e-5
    assert pytest.approx(j.link_len_a) == 80.0
    assert pytest.approx(j.link_len_b) == 95.0
    assert pytest.approx(j.twist_angle_alpha) == 30.0
    assert pytest.approx(j.offset_distance_s) == 8.5


def test_m476_lagmul_serre_spinor_spatial_linkage_joint_properties():
    j = LagmulSerreSpinorSpatialLinkageJoint(
        id=1, offset_distance_s=12.5
    )
    assert pytest.approx(j.offset_distance_r) == 12.5
    assert pytest.approx(j.offset_distance_v) == 12.5
    assert pytest.approx(j.offset_distance_h) == 12.5
    assert pytest.approx(j.offset_distance_u) == 12.5
    assert pytest.approx(j.offset_distance_f) == 12.5

    j.offset_distance_r = 14.0
    assert pytest.approx(j.offset_distance_s) == 14.0
    j.offset_distance_v = 15.5
    assert pytest.approx(j.offset_distance_s) == 15.5
    j.offset_distance_h = 16.2
    assert pytest.approx(j.offset_distance_s) == 16.2
    j.offset_distance_u = 17.8
    assert pytest.approx(j.offset_distance_s) == 17.8
    j.offset_distance_f = 19.1
    assert pytest.approx(j.offset_distance_s) == 19.1


def test_m476_lagmul_serre_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    aliases = [
        "/SERRE_SPINOR_SPATIAL_LINKAGE",
        "/SERRE_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM",
        "/SERRE_SPINOR_SPATIAL_SYMMETRIC_MECHANISM",
        "/SERRE_SPINOR_SPATIAL_6R_MECHANISM",
        "/SERRE_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM",
        "/SERRE_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/SERRE_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/SERRE_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/SERRE_DUALITY_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/SERRE_DUALITY_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/SERRE_FIBRATION_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/SERRE_FIBRATION_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/SERRE_SPECTRAL_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/SERRE_SPECTRAL_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/SERRE_SPINOR_SPATIAL_LINKAGE",
        "/LAGMUL/SERRE_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/SERRE_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/SERRE_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/SERRE_DUALITY_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/SERRE_DUALITY_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/SERRE_FIBRATION_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/SERRE_FIBRATION_TWISTOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/SERRE_SPECTRAL_SPINOR_SPATIAL_LINKAGE_JOINT",
        "/LAGMUL/SERRE_SPECTRAL_TWISTOR_SPATIAL_LINKAGE_JOINT",
    ]
    deck_lines = ["# RADIOSS FREE DECK", "/BEGIN", "Test Joint Aliases"]
    for idx, kw in enumerate(aliases, start=30):
        deck_lines.extend([
            f"{kw}/{idx}",
            f"{idx}, {idx + 1}, {idx + 2}, 1.0e6, 0, 1.0e-6",
            f"{idx * 2.0}, {idx * 3.0}, 25.0, 5.0"
        ])
    deck_lines.append("/END")
    deck = "\n".join(deck_lines)

    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for idx in range(30, 30 + len(aliases)):
        assert idx in model.lagmul_serre_spinor_spatial_linkage_joints
        j = model.lagmul_serre_spinor_spatial_linkage_joints[idx]
        assert j.node1 == idx
        assert j.node2 == idx + 1
        assert j.node3 == idx + 2
        assert pytest.approx(j.link_len_a) == idx * 2.0


def test_m476_lagmul_serre_spinor_spatial_linkage_joint_missing_card_error(tmp_path: Path):
    deck = """/BEGIN
Test Missing Card Joint
/SERRE_SPINOR_SPATIAL_LINKAGE_JOINT/99
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
    assert any("missing data card 1" in err for err in log.errors)


# ============================================================================
# 4. SENSOR_SPRING_TORSIONAL_CRACKLE_RATE Tests
# ============================================================================

def test_m476_sensor_spring_torsional_crackle_rate_fixed_and_free(tmp_path: Path):
    c1 = f"{75:>10d}{8.5e5:>20.4f}{0.0030:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test Sensor Torsional Crackle Rate
2022 0
/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/1
Torsional Crackle Rate Sensor Fixed
{c1}
/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/2
Free Format
76, 9.5e5, 0.0040
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_crackle_rates
    s1 = model.sensor_spring_torsional_crackle_rates[1]
    assert isinstance(s1, SensorSpringTorsionalCrackleRate)
    assert s1.spring_id == 75
    assert pytest.approx(s1.jtors_crk_max) == 8.5e5
    assert pytest.approx(s1.t_delay) == 0.0030

    # M476 25th rate-of-change property accessors
    assert pytest.approx(s1.j_tors_crackle_max) == 8.5e5
    assert pytest.approx(s1.j_torsional_crk_max) == 8.5e5
    assert pytest.approx(s1.j_torsional_crackle_max) == 8.5e5
    assert pytest.approx(s1.j_twist_crk_max) == 8.5e5
    assert pytest.approx(s1.j_twist_crackle_max) == 8.5e5
    assert pytest.approx(s1.j_twisting_crk_max) == 8.5e5
    assert pytest.approx(s1.j_twisting_crackle_max) == 8.5e5
    assert pytest.approx(s1.j_crk_twist_max) == 8.5e5
    assert pytest.approx(s1.j_crackle_twist_max) == 8.5e5
    assert pytest.approx(s1.jtors_crackle_rate_max) == 8.5e5
    assert pytest.approx(s1.jtors_snap_rate_max) == 8.5e5
    assert pytest.approx(s1.jtors_pop_rate_max) == 8.5e5
    assert pytest.approx(s1.jtors_lock_rate_max) == 8.5e5

    # Free format
    assert 2 in model.sensor_spring_torsional_crackle_rates
    s2 = model.sensor_spring_torsional_crackle_rates[2]
    assert s2.spring_id == 76
    assert pytest.approx(s2.jtors_crk_max) == 9.5e5
    assert pytest.approx(s2.t_delay) == 0.0040

    # Setters
    s1.j_tors_crackle_max = 1.1e6
    assert pytest.approx(s1.jtors_crk_max) == 1.1e6
    s1.j_twist_crk_max = 1.25e6
    assert pytest.approx(s1.jtors_crk_max) == 1.25e6
    s1.j_twisting_crackle_max = 1.35e6
    assert pytest.approx(s1.jtors_crk_max) == 1.35e6


def test_m476_sensor_spring_torsional_crackle_rate_aliases(tmp_path: Path):
    aliases = [
        "/SENSOR/SPRING_TORS_CRK_RATE",
        "/SENSOR/SPRING_TWIST_CRK_RATE",
        "/SENSOR/SPRING_TWISTING_CRACKLE_RATE",
        "/SENSOR/SPRING_TWISTING_CRK_RATE",
        "/SENSOR/SPRING_CRK_TWIST_RATE",
        "/SENSOR/SPRING_CRACKLE_TWIST_RATE",
        "/SENSOR/SPRING_RATE_TORSIONAL_CRACKLE",
        "/SENSOR/SPRING_RATE_TORS_CRACKLE",
        "/SENSOR/SPRING_RATE_TWIST_CRACKLE",
        "/SENSOR/SPRING_RATE_TWISTING_CRACKLE",
        "/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE_SENSOR",
        "/SENSOR/SPRING_TORS_CRACKLE_RATE_SENSOR",
        "/SENSOR/SPRING_TWIST_CRACKLE_RATE_SENSOR",
        "/SENSOR/SPRING_TWISTING_CRACKLE_RATE_SENSOR",
        "/SENSOR/TORSIONAL_CRACKLE_RATE_SPRING_SENSOR",
        "/SENSOR/TORS_CRACKLE_RATE_SPRING_SENSOR",
        "/SENSOR/TWIST_CRACKLE_RATE_SPRING_SENSOR",
        "/SENSOR/TWISTING_CRACKLE_RATE_SPRING_SENSOR",
        "/SENSOR/SPRING_CRACKLE_RATE_TORSIONAL_SENSOR",
        "/SENSOR/SPRING_CRACKLE_RATE_TORS_SENSOR",
        "/SENSOR/SPRING_CRACKLE_RATE_TWIST_SENSOR",
        "/SENSOR/SPRING_CRACKLE_RATE_TWISTING_SENSOR",
        "/SENSOR/SPRING_RATE_CRACKLE_TORSIONAL",
        "/SENSOR/SPRING_RATE_CRACKLE_TORS",
        "/SENSOR/SPRING_RATE_CRACKLE_TWIST",
        "/SENSOR/SPRING_RATE_CRACKLE_TWISTING",
        "/SENSOR/SPRING_RATE_TORS_CRK",
        "/SENSOR/SPRING_RATE_TORSIONAL_CRK",
        "/SENSOR/SPRING_RATE_TWIST_CRK",
        "/SENSOR/SPRING_RATE_TWISTING_CRK",
        "/SENSOR/SPRING_CRACKLE_TORSIONAL",
        "/SENSOR/SPRING_CRACKLE_TORS",
        "/SENSOR/SPRING_CRACKLE_TWIST",
        "/SENSOR/SPRING_CRACKLE_TWISTING",
        "/SENSOR/SPRING_SNAP_RATE_TORSIONAL_CRACKLE",
        "/SENSOR/SPRING_SNAP_RATE_TORS_CRACKLE",
        "/SENSOR/SPRING_SNAP_RATE_TWIST_CRACKLE",
        "/SENSOR/SPRING_LOCK_RATE_TORSIONAL_CRACKLE",
        "/SENSOR/SPRING_LOCK_RATE_TORS_CRACKLE",
        "/SENSOR/SPRING_LOCK_RATE_TWIST_CRACKLE",
    ]
    deck_lines = ["# RADIOSS FREE DECK", "/BEGIN", "Test M476 Sensor Aliases"]
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
        assert idx in model.sensor_spring_torsional_crackle_rates
        s = model.sensor_spring_torsional_crackle_rates[idx]
        assert s.spring_id == idx + 100
        assert pytest.approx(s.jtors_crk_max) == idx * 1e4

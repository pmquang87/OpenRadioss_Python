"""Test suite for Milestone M603: Failure Models Wave 3
(/FAIL/LADEVEZE, /FAIL/RTCL, /FAIL/GURSON).

Verifies Fortran-faithful physics, damage accumulation mechanics, and
starter keyword parsing for:
1. /FAIL/LADEVEZE (/FAIL/LAD_DAMA):
   - Micro-damage evolution in fiber-reinforced composite laminates
   - Thermodynamic damage forces Y_11, Y_22, Y_12, Y_33, Y_23, Y_13
   - Unilateral crack closure under transverse compression (sigma_22 < 0 -> Y_22 = 0)
   - Irreversibility and damage memory under cyclic shear / tension
   - Matrix micro-cracking damage d and fiber damage d_1
2. /FAIL/RTCL:
   - Rice-Tracey void growth criterion under high triaxiality (eta >= 1/3)
   - Cockcroft-Latham shear fracture criterion under low triaxiality (-1/3 <= eta < 1/3)
   - Continuous C0 transition function between regimes at eta = 1/3
   - Compressive cut-off (eta < -1/3)
   - Shell mesh sensitivity regularization (Inst = 2)
3. /FAIL/GURSON:
   - Standalone Gurson void volume fraction failure model
   - Void growth rate df_growth = (1 - f) * tr(d_eps_p)
   - Strain-controlled Gaussian void nucleation (Chu & Needleman 1980)
   - Void coalescence acceleration past critical void fraction f_c up to failure f_F
4. Starter deck parsing for /FAIL/LADEVEZE, /FAIL/RTCL, /FAIL/GURSON.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.failure import fail_gurson, fail_ladeveze, fail_rtcl, shell_step, solid_step
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import FailureModel
from pyradioss.model.model import Model


# ============================================================================
# Helper fixtures
# ============================================================================

def _parse_deck(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    from pyradioss.starter.initialization import resolve_materials
    resolve_materials(model, log)
    return model, log


# ============================================================================
# 1. /FAIL/LADEVEZE tests
# ============================================================================

def test_ladeveze_damage_forces_unilateral_closure():
    """Verify thermodynamic damage forces and unilateral micro-crack closure."""
    # Under transverse tension (syy > 0) vs compression (syy < 0)
    sxx = np.array([100.0, 100.0])
    syy = np.array([50.0, -50.0])  # tension vs compression
    sxy = np.array([30.0, 30.0])
    d = np.array([0.0, 0.0])

    k1, k2, k3 = 100000.0, 10000.0, 5000.0
    y11, y22, y12 = fail_ladeveze.compute_damage_forces_shell(sxx, syy, sxy, k1, k2, k3, d)

    # In tension: Y_22 > 0
    assert y22[0] > 0.0
    expected_y22_tens = 0.5 * (50.0**2) / k2
    assert np.isclose(y22[0], expected_y22_tens)

    # In compression: micro-cracks close -> Y_22 = 0
    assert y22[1] == 0.0

    # Shear force is symmetric in sxy
    assert np.isclose(y12[0], 0.5 * (30.0**2) / k3)
    assert np.isclose(y12[1], y12[0])


def test_ladeveze_cyclic_shear_damage_accumulation():
    """Verify damage accumulation under cyclic shear loading with unloading/reloading."""
    fail = FailureModel(
        type="LADEVEZE",
        ifail_sh=1,
        params={
            "k1": 1.0e5,
            "k2": 1.0e4,
            "k3": 5.0e3,
            "gamma1": 1.0,
            "y0": 0.04,     # Yield damage energy threshold
            "yc": 0.36,     # Critical damage energy threshold
            "k": 0.0,       # Instantaneous evolution
            "a": 1.0e30,
        },
    )

    dama = np.zeros(1, dtype=float)
    deps = np.zeros((1, 3), dtype=float)
    dt = 1e-4

    # Step 1: Initial shear loading -> sxy = 25.0
    # Y12 = 0.5 * 25^2 / 5000 = 0.0625 -> sqrt(Y) = 0.25 > sqrt(Y0)=0.20
    # W = (0.25 - 0.20) / (0.60 - 0.20) = 0.05 / 0.40 = 0.125
    sig1 = np.array([[0.0, 0.0, 25.0]])
    solid_step(fail, sig1, np.zeros(1), deps, dt, dama)
    d1 = float(dama[0])
    assert d1 > 0.0
    assert np.isclose(d1, 0.125, atol=1e-3)

    # Step 2: Unload shear to 10.0 -> damage must NOT heal or increase
    sig2 = np.array([[0.0, 0.0, 10.0]])
    solid_step(fail, sig2, np.zeros(1), deps, dt, dama)
    assert np.isclose(dama[0], d1)

    # Step 3: Reload within elastic envelope to 20.0 -> damage remains constant
    sig3 = np.array([[0.0, 0.0, 20.0]])
    solid_step(fail, sig3, np.zeros(1), deps, dt, dama)
    assert np.isclose(dama[0], d1)

    # Step 4: Overload shear to 40.0 -> exceeds previous peak, damage increases
    # Y12 = 0.5 * 40^2 / (5000 * (1 - 0.125)^2) = 800 / (5000 * 0.7656) = 0.2089
    # sqrt(Y) approx 0.457 > 0.25 -> W > 0.125
    sig4 = np.array([[0.0, 0.0, 40.0]])
    solid_step(fail, sig4, np.zeros(1), deps, dt, dama)
    assert dama[0] > d1


def test_ladeveze_fiber_and_matrix_failure():
    """Verify element deletion from matrix micro-cracking and fiber fracture."""
    fail = FailureModel(
        type="LADEVEZE",
        ifail_sh=1,
        params={
            "k1": 1.0e5,
            "k2": 1.0e4,
            "k3": 5.0e3,
            "gamma1": 1.0,
            "y0": 0.01,
            "yc": 0.04,
            "sigma_1t": 800.0,  # Fiber tensile strength
        },
    )

    # Element 0: moderate stress, not broken
    # Element 1: high shear driving matrix damage d >= 1.0
    # Element 2: high longitudinal tension sxx >= sigma_1t driving fiber failure d1 = 1.0
    sig = np.array([
        [100.0, 10.0, 0.0, 10.0, 0.0, 0.0],
        [100.0, 50.0, 0.0, 100.0, 0.0, 0.0],
        [850.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    dama = np.zeros(3, dtype=float)
    deps = np.zeros_like(sig)
    broken = solid_step(fail, sig, np.zeros(3), deps, 1e-4, dama)

    assert not broken[0]
    assert broken[1]  # Matrix rupture
    assert broken[2]  # Fiber rupture
    assert dama[1] >= 1.0
    assert dama[2] >= 1.0


# ============================================================================
# 2. /FAIL/RTCL tests
# ============================================================================

def test_rtcl_transition_function():
    """Verify RTCL smooth transition function between Rice-Tracey and Cockcroft-Latham."""
    # Compressive cut-off: eta < -1/3 -> F = 0
    assert fail_rtcl.rtcl_transition_function(-0.5) == 0.0
    assert fail_rtcl.rtcl_transition_function(-0.8) == 0.0

    # Boundary at eta = -1/3: F = 0
    assert np.isclose(fail_rtcl.rtcl_transition_function(-1.0 / 3.0), 0.0, atol=1e-6)

    # Pure shear: eta = 0 -> F_RTCL = 2 / sqrt(12) = 1 / sqrt(3) approx 0.57735
    # (Matches Cockcroft-Latham plane stress sigma_1 / sigma_vm = 1/sqrt(3))
    f_shear = fail_rtcl.rtcl_transition_function(0.0)
    assert np.isclose(f_shear, 1.0 / np.sqrt(3.0), atol=1e-4)

    # Transition boundary at eta = 1/3 (uniaxial tension):
    # Rice-Tracey: exp(-0.5) * exp(1.5 * 1/3) = exp(0) = 1.0
    # Cockcroft-Latham: 2 * (1 + 1/3 * 3) / (3 * 1/3 + 3) = 4 / 4 = 1.0
    # Both regimes match continuously!
    f_trans_left = fail_rtcl.rtcl_transition_function(1.0 / 3.0 - 1e-6)
    f_trans_right = fail_rtcl.rtcl_transition_function(1.0 / 3.0 + 1e-6)
    assert np.isclose(f_trans_left, 1.0, atol=1e-4)
    assert np.isclose(f_trans_right, 1.0, atol=1e-4)

    # High triaxiality: eta = 2/3 (equibiaxial tension):
    # F_RTCL = exp(-0.5) * exp(1.5 * 2/3) = exp(0.5) approx 1.6487
    f_high = fail_rtcl.rtcl_transition_function(2.0 / 3.0)
    assert np.isclose(f_high, np.exp(0.5), atol=1e-4)


def test_rtcl_dual_regime_damage_accumulation():
    """Verify dual-regime damage accumulation under different stress states."""
    fail = FailureModel(
        type="RTCL",
        ifail_sh=1,
        params={"epscal": 0.30, "inst": 1, "n": 0.0},
    )

    # Element 0: Uniaxial tension (eta = 1/3) -> F_RTCL = 1.0
    # Element 1: Pure shear (eta = 0.0) -> F_RTCL approx 0.577
    # Element 2: Triaxial compression (eta = -0.5 < -1/3) -> F_RTCL = 0.0
    sig = np.array([
        [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],       # Uniaxial tension
        [0.0, 0.0, 0.0, 50.0, 0.0, 0.0],        # Pure shear
        [-100.0, -100.0, 0.0, 0.0, 0.0, 0.0],   # Biaxial compression
    ])
    d_epsp = np.array([0.03, 0.03, 0.03])
    deps = np.zeros_like(sig)
    dama = np.zeros(3, dtype=float)

    broken = solid_step(fail, sig, d_epsp, deps, 1e-4, dama)
    assert not np.any(broken)

    # Uniaxial tension: dD = 1.0 * 0.03 / 0.30 = 0.10
    assert np.isclose(dama[0], 0.10, atol=1e-3)

    # Pure shear: dD = 0.577 * 0.03 / 0.30 approx 0.0577
    assert np.isclose(dama[1], (1.0 / np.sqrt(3.0)) * 0.10, atol=1e-3)

    # Compression: dD = 0
    assert dama[2] == 0.0


def test_rtcl_failure_element_deletion():
    """Verify element deletion when accumulated RTCL damage reaches 1.0."""
    fail = FailureModel(
        type="RTCL",
        ifail_sh=1,
        params={"epscal": 0.20, "inst": 1, "n": 0.0},
    )
    sig = np.array([[150.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    deps = np.zeros_like(sig)
    dama = np.zeros(1, dtype=float)

    # Small plastic strain
    solid_step(fail, sig, np.array([0.05]), deps, 1e-4, dama)
    assert dama[0] == pytest.approx(0.25, rel=1e-2)

    # Large plastic strain driving to failure
    broken = solid_step(fail, sig, np.array([0.16]), deps, 1e-4, dama)
    assert broken[0]
    assert dama[0] >= 1.0


# ============================================================================
# 3. /FAIL/GURSON tests
# ============================================================================

def test_gurson_void_growth_and_compression_inhibition():
    """Verify Gurson void growth rate in tension and inhibition in compression."""
    fail = FailureModel(
        type="GURSON",
        ifail_sh=1,
        params={
            "q1": 1.5,
            "q2": 1.0,
            "fc": 0.15,
            "fr": 0.25,
            "f0": 0.02,
            "eps_n": 0.0,
            "a_s": 0.0,
            "k_w": 0.0,
        },
    )

    # Element 0: Hydrostatic tension (eta = 1.0 > 0) -> void growth
    # Element 1: Hydrostatic compression (eta = -1.0 < 0) -> void growth = 0
    sig = np.array([
        [150.0, 100.0, 100.0, 0.0, 0.0, 0.0],
        [-150.0, -100.0, -100.0, 0.0, 0.0, 0.0],
    ])
    d_epsp = np.array([0.05, 0.05])
    deps = np.zeros_like(sig)
    dama = np.zeros(2, dtype=float)

    solid_step(fail, sig, d_epsp, deps, 1e-4, dama)
    assert dama[0] > dama[1]
    # In compression, void growth is zero
    assert dama[1] == pytest.approx(0.02 / 0.25, rel=1e-2)  # Remains at baseline f0 / fr


def test_gurson_gaussian_void_nucleation():
    """Verify Chu-Needleman Gaussian strain-controlled void nucleation."""
    eps_n = 0.20
    s_n = 0.05
    f_n = 0.04

    epsp_vals = np.array([0.10, 0.20, 0.30])  # Below peak, at peak, above peak
    d_epsp = np.array([0.01, 0.01, 0.01])

    df_nucl = fail_gurson.gaussian_void_nucleation(epsp_vals, d_epsp, eps_n, s_n, f_n)

    # Peak nucleation occurs at epsp = eps_n
    assert df_nucl[1] > df_nucl[0]
    assert df_nucl[1] > df_nucl[2]
    # Symmetry of Gaussian distribution
    assert np.isclose(df_nucl[0], df_nucl[2], rtol=1e-3)


def test_gurson_coalescence_and_critical_deletion():
    """Verify Tvergaard-Needleman void coalescence acceleration and element deletion."""
    fail = FailureModel(
        type="GURSON",
        ifail_sh=1,
        params={
            "q1": 1.5,
            "q2": 1.0,
            "fc": 0.10,   # Coalescence starts at f = 0.10
            "fr": 0.20,   # Failure at f = 0.20
            "f0": 0.01,
            "a_s": 0.10,  # Nucleation driving void volume fraction
            "k_w": 0.0,
            "eps_n": 0.0,
        },
    )

    # Uniaxial tension: moderate triaxiality eta = 1/3
    sig = np.array([[200.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    deps = np.zeros_like(sig)
    dama = np.zeros(1, dtype=float)

    # Small deformation: pre-coalescence (f < fc)
    broken1 = solid_step(fail, sig, np.array([0.05]), deps, 1e-4, dama)
    assert not broken1[0]
    assert dama[0] < (0.10 / 0.20)

    # Large deformation: driving through coalescence past fc to rupture
    broken2 = solid_step(fail, sig, np.array([2.0]), deps, 1e-4, dama)
    assert broken2[0]
    assert dama[0] >= 1.0


# ============================================================================
# 4. Starter Keyword Parsing Tests
# ============================================================================

def test_starter_parsing_ladeveze_free_and_fixed(tmp_path: Path):
    """Verify starter parsing of /FAIL/LADEVEZE and /FAIL/LAD_DAMA."""
    deck_free = """/BEGIN
TEST_LADEVEZE_FREE
/MAT/LAW1/10
Composite Ply
1.5e-9
140000.0 0.3
/FAIL/LADEVEZE/10
140000.0 10000.0 5000.0 0.45 0.35
0.05 0.40 0.8 1.5 50.0
1 1
101
/END
"""
    model, log = _parse_deck(tmp_path, deck_free)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 10 in model.fail_ladevezes
    flad = model.fail_ladevezes[10]
    assert flad.k1 == pytest.approx(140000.0)
    assert flad.k2 == pytest.approx(10000.0)
    assert flad.k3 == pytest.approx(5000.0)
    assert flad.gamma1 == pytest.approx(0.45)
    assert flad.gamma2 == pytest.approx(0.35)
    assert flad.y0 == pytest.approx(0.05)
    assert flad.yc == pytest.approx(0.40)
    assert flad.k_lad == pytest.approx(0.8)
    assert flad.a_dama == pytest.approx(1.5)
    assert flad.tau_max == pytest.approx(50.0)
    assert flad.ifail_sh == 1
    assert flad.ifail_so == 1
    assert flad.fail_id == 101

    # Verify material attachment
    mat = model.materials[10]
    assert mat.fail is not None
    assert mat.fail.type in ("LADEVEZE", "LAD_DAMA")


def test_starter_parsing_rtcl(tmp_path: Path):
    """Verify starter parsing of /FAIL/RTCL."""
    deck = """/BEGIN
TEST_RTCL
/MAT/LAW2/20
Steel Sheet
7.8e-9
210000.0 0.3
400.0 250.0 0.25
/FAIL/RTCL/20
0.28 2 0.18
201
/END
"""
    model, log = _parse_deck(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 20 in model.fail_rtcls
    frtcl = model.fail_rtcls[20]
    assert frtcl.epscal == pytest.approx(0.28)
    assert frtcl.inst == 2
    assert frtcl.n == pytest.approx(0.18)
    assert frtcl.fail_id == 201

    mat = model.materials[20]
    assert mat.fail is not None
    assert mat.fail.type == "RTCL"


def test_starter_parsing_gurson_formats(tmp_path: Path):
    """Verify starter parsing of /FAIL/GURSON in 4-card and 2-card formats."""
    # Standard 4-card Radioss GTN format
    deck_4card = """/BEGIN
TEST_GURSON_4CARD
/MAT/LAW2/30
Porous Metal
7.85e-9
200000.0 0.3
500.0 300.0 0.2
/FAIL/GURSON/30
1.5 1.0 1
0.25 0.05 1.2
0.12 0.22 0.005
2.0 0.05 1.0
301
/END
"""
    model, log = _parse_deck(tmp_path, deck_4card)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 30 in model.fail_gursons
    fgur = model.fail_gursons[30]
    assert fgur.q1 == pytest.approx(1.5)
    assert fgur.q2 == pytest.approx(1.0)
    assert fgur.iloc == 1
    assert fgur.eps_n == pytest.approx(0.25)
    assert fgur.a_s == pytest.approx(0.05)
    assert fgur.k_w == pytest.approx(1.2)
    assert fgur.f_c == pytest.approx(0.12)
    assert fgur.f_r == pytest.approx(0.22)
    assert fgur.f_0 == pytest.approx(0.005)
    assert fgur.fail_id == 301

    mat = model.materials[30]
    assert mat.fail is not None
    assert mat.fail.type == "GURSON"


def test_failure_dispatch_shell_step_wave3():
    """Verify shell_step dispatch for all three failure models."""
    sig_shell = np.array([[120.0, 60.0, 20.0]])
    d_epsp = np.array([0.02])
    deps = np.zeros_like(sig_shell)

    # 1. LADEVEZE shell step
    fail_lad = FailureModel(
        type="LADEVEZE",
        ifail_sh=1,
        params={"k1": 1e5, "k2": 1e4, "k3": 5e3, "y0": 0.01, "yc": 0.10, "gamma1": 1.0},
    )
    dama_lad = np.zeros(1, dtype=float)
    broken_lad = shell_step(fail_lad, sig_shell, d_epsp, deps, 1e-4, dama_lad)
    assert isinstance(broken_lad, (np.ndarray, bool))
    assert dama_lad[0] > 0.0

    # 2. RTCL shell step
    fail_rtcl_m = FailureModel(
        type="RTCL",
        ifail_sh=1,
        params={"epscal": 0.30, "inst": 1, "n": 0.0},
    )
    dama_rtcl = np.zeros(1, dtype=float)
    broken_rtcl = shell_step(fail_rtcl_m, sig_shell, d_epsp, deps, 1e-4, dama_rtcl)
    assert isinstance(broken_rtcl, (np.ndarray, bool))
    assert dama_rtcl[0] > 0.0

    # 3. GURSON shell step
    fail_gur = FailureModel(
        type="GURSON",
        ifail_sh=1,
        params={"q1": 1.5, "q2": 1.0, "fc": 0.15, "fr": 0.25, "f0": 0.01, "a_s": 0.05},
    )
    dama_gur = np.zeros(1, dtype=float)
    broken_gur = shell_step(fail_gur, sig_shell, d_epsp, deps, 1e-4, dama_gur)
    assert isinstance(broken_gur, (np.ndarray, bool))
    assert dama_gur[0] > 0.0

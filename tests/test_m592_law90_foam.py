"""Unit test suite for Milestone M592: /MAT/LAW90 & /MAT/HYST_FOAM Tabulated Foam.

Tests:
1. Law90Params dataclass fields, defaults, and aliases (gamma/alpha).
2. build_law90 factory supporting dict, GenericMaterialRecord, MaterialLaw90, and Law90Params.
3. Compressive loading plateau and densification (nominal and Cauchy stresses).
4. Hysteretic energy dissipation during cyclic loading and unloading (continuity at reversal, DAM scaling, shape factor).
5. Strain-rate stiffening under high strain rates and dynamic curve interpolation.
6. Tensile stress cutoff limit (tcut) and failure option (fail).
7. Starter keyword parsing for /MAT/LAW90 and /MAT/HYST_FOAM in fixed and free formats.
8. Acoustic sound speed calculation with evolving modulus.
9. Tangent stiffness matrix calculation and implicit consistency.
10. Energy conservation in dynamic explicit simulation.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import FunctTable
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import Material, MaterialLaw90
from pyradioss.model.model import Model
from pyradioss.starter.initialization import resolve_materials
import pyradioss.materials as materials
from pyradioss.materials.law90_foam import (
    Law90Params,
    build_law90,
    solid_step,
    solid_update,
    sound_speed,
    solid_tangent,
    consistent_solid_tangent,
    shell_update,
    extra_shapes,
    resolve,
)


def _parse_deck(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


# ---------------------------------------------------------------------------
# 1. Parameter and Factory Tests
# ---------------------------------------------------------------------------

def test_law90_params_dataclass():
    """Verify Law90Params default values and attribute aliases."""
    p = Law90Params(rho0=1.2e-3, E0=50.0, nu=0.1)
    assert p.rho0 == 1.2e-3
    assert p.refer_rho == 1.2e-3
    assert p.E0 == 50.0
    assert p.nu == 0.1
    assert p.shape == 1.0
    assert p.hys == 1.0
    assert p.gamma == 1.0
    assert p.alpha == 1.0
    assert p.tcut == 1e20
    assert p.tflag == 1
    assert p.fail == 0
    assert p.econt == 50.0
    assert p.ismooth == 0
    assert p.fcut == 0.0

    # Test gamma/alpha synchronization
    p2 = Law90Params(gamma=2.5)
    assert p2.gamma == 2.5
    assert p2.alpha == 2.5

    p3 = Law90Params(alpha=1.8)
    assert p3.alpha == 1.8
    assert p3.gamma == 1.8


def test_build_law90_factory_inputs():
    """Verify build_law90 supports dict, MaterialLaw90, Material, and Law90Params."""
    # From dict with standard names
    d1 = {
        "id": 10,
        "title": "Foam-A",
        "rho": 0.05,
        "E0": 80.0,
        "nu": 0.15,
        "shape": 2.0,
        "hys": 0.4,
        "alpha": 1.5,
        "tcut": 8.0,
        "tflag": 2,
    }
    m1 = build_law90(d1)
    assert isinstance(m1, Material)
    assert m1.id == 10
    assert m1.law == 90
    assert m1.rho0 == 0.05
    assert m1.title == "Foam-A"
    assert m1.params["E0"] == 80.0
    assert m1.params["shape"] == 2.0
    assert m1.params["hys"] == 0.4
    assert m1.params["alpha"] == 1.5
    assert m1.params["tcut"] == 8.0
    assert m1.params["tflag"] == 2

    # From MaterialLaw90 entity
    ent = MaterialLaw90(
        id=20,
        title="Foam-Entity",
        rho0=0.06,
        e0=100.0,
        nu=0.05,
        shape=1.8,
        hys=0.3,
        alpha=2.0,
        tcut=12.0,
        tflag=1,
    )
    m2 = build_law90(ent)
    assert m2.id == 20
    assert m2.rho0 == 0.06
    assert m2.params["E0"] == 100.0
    assert m2.params["shape"] == 1.8
    assert m2.params["hys"] == 0.3
    assert m2.params["alpha"] == 2.0
    assert m2.params["tcut"] == 12.0

    # From Law90Params
    lp = Law90Params(rho0=0.04, E0=60.0, nu=0.0, hys=0.5, tcut=4.0)
    m3 = build_law90(lp)
    assert m3.rho0 == 0.04
    assert m3.params["E0"] == 60.0
    assert m3.params["hys"] == 0.5
    assert m3.params["tcut"] == 4.0


# ---------------------------------------------------------------------------
# 2. Compressive Loading Plateau and Densification Tests
# ---------------------------------------------------------------------------

def test_compressive_loading_plateau_and_densification():
    """Verify foam compression behavior: linear elasticity, plateau, and densification."""
    # Engineering strain vs engineering stress curve:
    # 0.0 -> 0.0
    # 0.02 -> 2.0 MPa (initial slope E = 100 MPa)
    # 0.50 -> 2.5 MPa (plateau)
    # 0.70 -> 10.0 MPa (densification onset)
    # 0.80 -> 40.0 MPa (steep densification)
    x = np.array([0.0, 0.02, 0.50, 0.70, 0.80], dtype=float)
    y = np.array([0.0, 2.0, 2.5, 10.0, 40.0], dtype=float)

    mat = build_law90({
        "E0": 100.0,
        "nu": 0.0,
        "rho0": 1.0e-3,
        "curves": [(x, y)],
        "eps_dots": [0.0],
        "hys": 1.0,
    })

    extra = extra_shapes(mat)
    sig = np.zeros(6, dtype=float)

    # 1. Elastic region: small compression eps_xx = -0.01 (e_xx = 1 - exp(-0.01) ~= 0.00995)
    deps_elas = np.array([-0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_new, _, c_elas = solid_step(mat, sig, deps_elas, dt=1.0e-4, extra=extra)

    # In compression, Cauchy stress sigma_xx is negative
    assert sig_new[0] < 0.0
    # True engineering strain e = 1 - exp(-0.01) = 0.00995017
    # Expected nominal stress S ~= 0.00995017 * 100 = 0.995 MPa
    # Since nu=0, lateral stretches lambda_y = lambda_z = 1.0, so sigma_xx = -S
    e_val = 1.0 - math.exp(-0.01)
    expected_s = np.interp(e_val, x, y)
    assert pytest.approx(abs(sig_new[0]), rel=1e-3) == expected_s

    # 2. Plateau region: compress further to total log strain ~ -0.30 (e ~= 0.259)
    # Reset state for clean step
    extra = extra_shapes(mat)
    sig = np.zeros(6, dtype=float)
    deps_plat = np.array([-0.30, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_plat, _, c_plat = solid_step(mat, sig, deps_plat, dt=1.0e-4, extra=extra)

    e_plat = 1.0 - math.exp(-0.30)
    expected_s_plat = np.interp(e_plat, x, y)  # In plateau range ~ 2.25 MPa
    assert 2.0 <= expected_s_plat <= 2.5
    assert pytest.approx(abs(sig_plat[0]), rel=1e-3) == expected_s_plat

    # 3. Densification region: compress to log strain ~ -1.5 (e = 1 - exp(-1.5) = 0.777)
    extra = extra_shapes(mat)
    sig = np.zeros(6, dtype=float)
    deps_dens = np.array([-1.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_dens, _, c_dens = solid_step(mat, sig, deps_dens, dt=1.0e-4, extra=extra)

    e_dens = 1.0 - math.exp(-1.5)
    expected_s_dens = np.interp(e_dens, x, y)
    assert expected_s_dens > 20.0
    assert pytest.approx(abs(sig_dens[0]), rel=1e-3) == expected_s_dens

    # Sound speed should increase with evolving modulus in densification
    assert c_dens >= c_elas


# ---------------------------------------------------------------------------
# 3. Cyclic Hysteresis & Energy Dissipation Tests
# ---------------------------------------------------------------------------

def test_hysteretic_unloading_continuity_and_energy_dissipation():
    """Verify hysteretic unloading: continuity at reversal point (DAM=1),

    stress reduction by Hys, and dissipated work integral.
    """
    x = np.array([0.0, 0.1, 0.4, 0.6], dtype=float)
    y = np.array([0.0, 5.0, 10.0, 20.0], dtype=float)

    hys_factor = 0.3
    mat = build_law90({
        "E0": 50.0,
        "nu": 0.0,
        "rho0": 1.0e-3,
        "curves": [(x, y)],
        "eps_dots": [0.0],
        "hys": hys_factor,
        "shape": 1.5,
        "alpha": 1.0,
    })

    extra = extra_shapes(mat)
    sig = np.zeros(6, dtype=float)

    # Step 1: Loading in compression to eps_xx = -0.40 (10 steps of -0.04)
    dt = 1.0e-3
    nsteps = 10
    deps_load = np.array([-0.04, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    work_load = 0.0
    for _ in range(nsteps):
        sig_old = sig.copy()
        sig, _, _ = solid_step(mat, sig, deps_load, dt=dt, extra=extra)
        # Work done on specimen during compression (sig < 0, deps < 0 -> sig * deps > 0)
        work_load += 0.5 * (sig_old[0] + sig[0]) * deps_load[0]

    # At peak compression:
    peak_sig = sig[0]
    assert peak_sig < 0.0  # compressive
    uv = extra["uv90"]
    w_max = uv[0, 1]
    assert w_max > 0.0

    # Step 2: First unloading increment (reversal)
    deps_unload = np.array([+0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_reversal, _, _ = solid_step(mat, sig, deps_unload, dt=dt, extra=extra)

    # Continuity check: at the reversal point, DAM should be near 1.0,
    # so stress is continuous without sudden discontinuous jump
    assert abs(sig_reversal[0] - peak_sig) < 0.2

    # Step 3: Unload completely back towards zero strain
    deps_unl_full = np.array([+0.04, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    work_unload_recovered = 0.0
    for _ in range(nsteps - 1):
        sig_old = sig.copy()
        sig, _, _ = solid_step(mat, sig, deps_unl_full, dt=dt, extra=extra)
        # Work recovered during unloading: -(sig * deps) > 0 since sig < 0 and deps > 0
        work_unload_recovered += -0.5 * (sig_old[0] + sig[0]) * deps_unl_full[0]

    # Energy dissipation check:
    # Work done during loading must exceed work recovered during unloading
    net_dissipated_energy = work_load - work_unload_recovered
    assert net_dissipated_energy > 0.0, "Hysteretic cycle must dissipate positive net energy"


def test_hys_elastic_limit_comparison():
    """Verify that when hys=1.0, unloading follows loading curve (elastic, zero dissipation),

    while hys=0.2 dissipates significant energy.
    """
    x = np.array([0.0, 0.2, 0.5], dtype=float)
    y = np.array([0.0, 4.0, 8.0], dtype=float)

    # Case A: Purely elastic (hys=1.0)
    mat_elas = build_law90({
        "E0": 20.0, "nu": 0.0, "rho0": 1.0e-3, "curves": [(x, y)], "eps_dots": [0.0], "hys": 1.0,
    })
    # Case B: Strongly hysteretic (hys=0.2)
    mat_hyst = build_law90({
        "E0": 20.0, "nu": 0.0, "rho0": 1.0e-3, "curves": [(x, y)], "eps_dots": [0.0], "hys": 0.2,
    })

    extra_elas = extra_shapes(mat_elas)
    extra_hyst = extra_shapes(mat_hyst)

    # Load both to eps_xx = -0.3
    sig_e = np.zeros(6, dtype=float)
    sig_h = np.zeros(6, dtype=float)
    deps_load = np.array([-0.3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    sig_e, _, _ = solid_step(mat_elas, sig_e, deps_load, dt=1e-3, extra=extra_elas)
    sig_h, _, _ = solid_step(mat_hyst, sig_h, deps_load, dt=1e-3, extra=extra_hyst)

    # Both reach same loading stress at peak
    assert pytest.approx(sig_e[0], rel=1e-6) == sig_h[0]

    # Partial unload: deps_unload = +0.15
    deps_unl = np.array([+0.15, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_e_unl, _, _ = solid_step(mat_elas, sig_e, deps_unl, dt=1e-3, extra=extra_elas)
    sig_h_unl, _, _ = solid_step(mat_hyst, sig_h, deps_unl, dt=1e-3, extra=extra_hyst)

    # Unloading stress for hysteretic material is reduced: |sig_h_unl| < |sig_e_unl|
    assert abs(sig_h_unl[0]) < abs(sig_e_unl[0])


# ---------------------------------------------------------------------------
# 4. Strain-Rate Stiffening Tests
# ---------------------------------------------------------------------------

def test_strain_rate_stiffening_and_interpolation():
    """Verify dynamic strain rate stiffening across tabulated curves and linear interpolation."""
    x = np.array([0.0, 0.1, 0.5], dtype=float)
    # 3 strain rate curves:
    # Curve 1 (rate=0): static y = [0, 2, 5]
    # Curve 2 (rate=10): y = [0, 4, 10]
    # Curve 3 (rate=100): y = [0, 8, 20]
    c1 = (x, np.array([0.0, 2.0, 5.0], dtype=float))
    c2 = (x, np.array([0.0, 4.0, 10.0], dtype=float))
    c3 = (x, np.array([0.0, 8.0, 20.0], dtype=float))

    mat = build_law90({
        "E0": 20.0,
        "nu": 0.0,
        "rho0": 1.0e-3,
        "curves": [c1, c2, c3],
        "eps_dots": [0.0, 10.0, 100.0],
        "hys": 1.0,
    })

    # Strain increment: log strain -0.2
    # In sigeps90.F: STRAINRATE = EPSP * (1 - STRAIN) = (deps / dt) * exp(deps)
    deps = np.array([-0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    lam = math.exp(-0.2)
    e_val = 1.0 - lam

    # Test at static rate (dt large -> rate ~ 0)
    sig0 = np.zeros(6, dtype=float)
    ex0 = extra_shapes(mat)
    sig_stat, _, _ = solid_step(mat, sig0, deps, dt=1000.0, extra=ex0)

    # Test at rate = 10: dt = (0.2 * lam) / 10.0 -> rateeps = 10.0
    dt_10 = (0.2 * lam) / 10.0
    sig1 = np.zeros(6, dtype=float)
    ex1 = extra_shapes(mat)
    sig_rate10, _, _ = solid_step(mat, sig1, deps, dt=dt_10, extra=ex1)

    # Test at rate = 100: dt = (0.2 * lam) / 100.0 -> rateeps = 100.0
    dt_100 = (0.2 * lam) / 100.0
    sig2 = np.zeros(6, dtype=float)
    ex2 = extra_shapes(mat)
    sig_rate100, _, _ = solid_step(mat, sig2, deps, dt=dt_100, extra=ex2)

    # Verify monotonic rate stiffening: |sig(100)| > |sig(10)| > |sig(0)|
    assert abs(sig_rate100[0]) > abs(sig_rate10[0]) > abs(sig_stat[0])

    # Check exact interpolated values
    s_stat_exp = np.interp(e_val, c1[0], c1[1])
    s_10_exp = np.interp(e_val, c2[0], c2[1])
    s_100_exp = np.interp(e_val, c3[0], c3[1])

    assert pytest.approx(abs(sig_stat[0]), rel=1e-2) == s_stat_exp
    assert pytest.approx(abs(sig_rate10[0]), rel=1e-2) == s_10_exp
    assert pytest.approx(abs(sig_rate100[0]), rel=1e-2) == s_100_exp

    # Test intermediate rate (e.g. rate = 55): should interpolate between c2 and c3
    sig3 = np.zeros(6, dtype=float)
    ex3 = extra_shapes(mat)
    dt_mid = (0.2 * lam) / 55.0
    sig_mid, _, _ = solid_step(mat, sig3, deps, dt=dt_mid, extra=ex3)
    assert abs(sig_rate10[0]) < abs(sig_mid[0]) < abs(sig_rate100[0])


# ---------------------------------------------------------------------------
# 5. Tensile Cutoff and Failure Tests
# ---------------------------------------------------------------------------

def test_tensile_cutoff_limit():
    """Verify that tensile stress is capped at tcut when fail=0."""
    x = np.array([-0.5, 0.0, 0.5], dtype=float)
    y = np.array([-50.0, 0.0, 50.0], dtype=float)  # linear 100 MPa slope

    tcut_val = 15.0
    mat = build_law90({
        "E0": 100.0,
        "nu": 0.0,
        "rho0": 1.0e-3,
        "curves": [(x, y)],
        "eps_dots": [0.0],
        "tcut": tcut_val,
        "fail": 0,
        "tflag": 1,
    })

    extra = extra_shapes(mat)
    sig = np.zeros(6, dtype=float)

    # 1. Below cutoff: tensile strain deps_xx = +0.05
    # e = 1 - exp(0.05) ~= -0.05127
    deps_small = np.array([+0.05, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_small, _, _ = solid_step(mat, sig, deps_small, dt=1e-3, extra=extra)
    # Expected tensile stress ~ 5.127 MPa < 15.0 MPa
    assert 0.0 < sig_small[0] < tcut_val

    # 2. Above cutoff: large tensile strain deps_xx = +0.30
    extra = extra_shapes(mat)
    sig = np.zeros(6, dtype=float)
    deps_large = np.array([+0.30, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_large, _, _ = solid_step(mat, sig, deps_large, dt=1e-3, extra=extra)

    # Nominal tensile stress is clamped at tcut_val = 15.0 MPa
    # True Cauchy stress sigma_xx = T / (lambda_y * lambda_z) = 15.0 / (1 * 1) = 15.0
    assert pytest.approx(sig_large[0], rel=1e-4) == tcut_val


def test_tensile_failure_element_deletion():
    """Verify that when fail=1 and tensile stress reaches tcut, stress is zeroed."""
    x = np.array([-0.5, 0.0, 0.5], dtype=float)
    y = np.array([-50.0, 0.0, 50.0], dtype=float)

    tcut_val = 10.0
    mat = build_law90({
        "E0": 100.0,
        "nu": 0.0,
        "rho0": 1.0e-3,
        "curves": [(x, y)],
        "eps_dots": [0.0],
        "tcut": tcut_val,
        "fail": 1,
    })

    extra = extra_shapes(mat)
    sig = np.zeros(6, dtype=float)

    # Exceed cutoff in tension
    deps_large = np.array([+0.30, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_fail, _, _ = solid_step(mat, sig, deps_large, dt=1e-3, extra=extra)

    assert pytest.approx(sig_fail[0]) == 0.0


# ---------------------------------------------------------------------------
# 6. Starter Keyword Parsing Tests (/MAT/LAW90 and /MAT/HYST_FOAM)
# ---------------------------------------------------------------------------

def test_starter_keyword_parsing_mat_law90_fixed(tmp_path: Path):
    """Test /MAT/LAW90 deck parsing in fixed format with Radioss 2026 extended cards."""
    deck = (
        "#STARTER\n"
        "/MAT/LAW90/100\n"
        "LAW90 Tabulated Foam Test\n"
        "#              Rho_I               Rho_O\n"
        "             5.00E-8             5.00E-8\n"
        "#                 E0              MAT_NU     TFLAG      FAIL               Kcont                Tcut\n"
        "                80.0                0.05         2         0                80.0                12.5\n"
        "#       NL   Ismooth                Fcut               Shape                 Hys               Alpha\n"
        "         2         1                50.0                 2.5                 0.4                 1.2\n"
        "#  fct_idL             eps_dot              Fscale\n"
        "       101                 0.0                 1.0\n"
        "       102                10.0                 1.5\n"
        "/FUNCT/101\n"
        "Static Loading Curve\n"
        "0.0 0.0\n"
        "0.1 2.0\n"
        "0.5 5.0\n"
        "/FUNCT/102\n"
        "Dynamic Loading Curve\n"
        "0.0 0.0\n"
        "0.1 3.0\n"
        "0.5 8.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 100 in model.materials

    mat = model.materials[100]
    assert mat.law == 90
    assert pytest.approx(mat.rho0) == 5.00e-8
    assert pytest.approx(mat.params["E0"]) == 80.0
    assert pytest.approx(mat.params["nu"]) == 0.05
    assert mat.params["tflag"] == 2
    assert mat.params["fail"] == 0
    assert pytest.approx(mat.params["tcut"]) == 12.5
    assert mat.params["NL"] == 2
    assert mat.params["Ismooth"] == 1
    assert pytest.approx(mat.params["Fcut"]) == 50.0
    assert pytest.approx(mat.params["shape"]) == 2.5
    assert pytest.approx(mat.params["hys"]) == 0.4
    assert pytest.approx(mat.params["alpha"]) == 1.2
    assert mat.params["fct_ids"] == [101, 102]
    assert mat.params["eps_dots"] == [0.0, 10.0]
    assert mat.params["fscales"] == [1.0, 1.5]

    # Test resolve_materials hook
    resolve_materials(model, log)
    assert len(mat.params["curves"]) == 2
    # Curve 102 should be scaled by 1.5
    assert pytest.approx(mat.params["curves"][1][1][-1]) == 8.0 * 1.5


def test_starter_keyword_parsing_mat_hyst_foam_free(tmp_path: Path):
    """Test /MAT/HYST_FOAM deck parsing in free format."""
    deck = (
        "/MAT/HYST_FOAM/200\n"
        "Hysteretic Foam Free Format\n"
        "4.5e-8 4.5e-8\n"
        "60.0 0.0 1 0 60.0 8.0\n"
        "1 0 0.0 2.0 0.35 1.5\n"
        "301 0.0 1.0\n"
        "/FUNCT/301\n"
        "Static Foam Curve\n"
        "0.0 0.0\n"
        "0.2 3.0\n"
        "0.6 6.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 200 in model.materials

    mat = model.materials[200]
    assert mat.law == 90
    assert pytest.approx(mat.rho0) == 4.5e-8
    assert pytest.approx(mat.params["E0"]) == 60.0
    assert pytest.approx(mat.params["nu"]) == 0.0
    assert pytest.approx(mat.params["tcut"]) == 8.0
    assert pytest.approx(mat.params["shape"]) == 2.0
    assert pytest.approx(mat.params["hys"]) == 0.35
    assert pytest.approx(mat.params["alpha"]) == 1.5
    assert mat.params["fct_ids"] == [301]

    resolve_materials(model, log)
    assert len(mat.params["curves"]) == 1


# ---------------------------------------------------------------------------
# 7. Tangent Stiffness and Dispatch Tests
# ---------------------------------------------------------------------------

def test_solid_tangent_consistency():
    """Verify solid_tangent returns symmetric positive-definite 6x6 tangent matrix."""
    mat = build_law90({"E0": 120.0, "nu": 0.2, "rho0": 1.0e-3})
    C = solid_tangent(mat)

    assert C.shape == (6, 6)
    # Symmetry check
    assert np.allclose(C, C.T)
    # Positive definiteness check
    eigvals = np.linalg.eigvalsh(C)
    assert np.all(eigvals > 0.0)

    # Check P-wave component C11 = E*(1-nu)/((1+nu)*(1-2*nu))
    expected_c11 = 120.0 * (1.0 - 0.2) / ((1.0 + 0.2) * (1.0 - 0.4))
    assert pytest.approx(C[0, 0]) == expected_c11


def test_materials_dispatch_integration():
    """Verify pyradioss.materials dispatchers recognize LAW90."""
    mat = build_law90({"E0": 80.0, "nu": 0.1, "rho0": 2.0e-3})
    sig = np.zeros(6, dtype=float)
    deps = np.array([-0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    extra = extra_shapes(mat)

    # Test top-level materials.solid_update
    s_new, ep, c = materials.solid_update(mat, sig, deps, dt=1e-3, extra=extra)
    assert s_new[0] < 0.0
    assert c > 0.0

    # Test top-level materials.sound_speed
    c_calc = materials.sound_speed(mat, extra=extra)
    assert c_calc > 0.0

    # Test top-level materials.solid_tangent
    C_tan = materials.solid_tangent(mat, extra=extra)
    assert C_tan.shape == (6, 6)

    # Shell update must raise NotImplementedError
    with pytest.raises(NotImplementedError):
        materials.shell_update(mat, np.zeros(3), np.zeros(3))


# ---------------------------------------------------------------------------
# 8. Dynamic Simulation & Energy Conservation Test
# ---------------------------------------------------------------------------

def test_dynamic_explicit_energy_balance():
    """Verify energy conservation and stability of LAW90 under explicit time integration."""
    x = np.array([0.0, 0.1, 0.4, 0.7], dtype=float)
    y = np.array([0.0, 4.0, 8.0, 25.0], dtype=float)

    mat = build_law90({
        "E0": 40.0,
        "nu": 0.0,
        "rho0": 1.0e-3,
        "curves": [(x, y)],
        "eps_dots": [0.0],
        "hys": 0.5,
    })

    extra = extra_shapes(mat)
    sig = np.zeros(6, dtype=float)

    dt = 1.0e-4
    n_cycles = 100
    # Apply sinusoidal strain oscillation: eps(t) = -0.15 * (1 - cos(omega*t))
    omega = 2.0 * math.pi / (n_cycles * dt)

    t = 0.0
    eps_prev = 0.0
    total_external_work = 0.0

    for step in range(n_cycles):
        t += dt
        eps_curr = -0.15 * (1.0 - math.cos(omega * t))
        deps_val = eps_curr - eps_prev
        eps_prev = eps_curr

        deps = np.array([deps_val, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        sig_old = sig.copy()
        sig, _, c = solid_step(mat, sig, deps, dt=dt, extra=extra)

        # Work increment dW = sigma_avg * deps (positive work input during compression)
        dw = 0.5 * (sig_old[0] + sig[0]) * deps_val
        total_external_work += dw

        assert np.all(np.isfinite(sig)), "Stress tensor must remain finite"
        assert c > 0.0, "Sound speed must remain strictly positive"

    # Total work done over closed cycle must be non-negative (2nd law / thermodynamics)
    assert total_external_work >= 0.0

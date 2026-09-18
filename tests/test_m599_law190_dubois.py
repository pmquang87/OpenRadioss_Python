"""
Unit tests for Milestone M599: Fortran-faithful Du Bois Crushable Foam Material
(/MAT/LAW190, /MAT/FOAM_DUBOIS).

Verifies:
1. Law190Params dataclass fields, defaults, and derived elastic properties.
2. build_law190 factory supporting dict, MatLaw190, Material, and Law190Params.
3. Principal stretch and engineering strain decomposition e_k = 1.0 - lambda_k.
4. Compressive stress-strain response matching tabulated curve (1D, 2D, 3D).
5. Strain-rate stiffening under dynamic loading rates.
6. Cyclic loading-unloading hysteresis and energy dissipation (condamage.F).
7. Tensile stress cutoff clamping (fail=0).
8. Tensile failure and erosion under hydrostatic and principal tension (fail=1).
9. Dilatational sound speed evolution and algorithmic tangent stiffness.
10. Solid-only constraint: shell_update raises NotImplementedError.
11. Starter deck parsing for /MAT/LAW190 and /MAT/FOAM_DUBOIS (fixed & free format).
12. Integration with pyradioss.materials dispatcher functions.
13. Vectorized multi-element group consistency.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.common.tables import FunctTable
from pyradioss.model import Model
from pyradioss.model.entities import Material, MatLaw190
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
import pyradioss.materials as materials
from pyradioss.materials.law190_dubois import (
    Law190Params,
    build_law190,
    solid_step,
    solid_update,
    sound_speed,
    solid_tangent,
    consistent_solid_tangent,
    shell_update,
    shell_tangent,
    extra_shapes,
    resolve,
)


# ---------------------------------------------------------------------------
# 1. Parameter checks and factory tests
# ---------------------------------------------------------------------------

def test_law190_params_defaults_and_derived():
    """Verify Law190Params defaults, derived elastic properties, and nu bounds."""
    # Defaults with E0=120.0, nu=0.0 (typical crushable foam)
    p = Law190Params(rho0=1.2e-3, e0=120.0, nu=0.0)
    assert p.rho0 == 1.2e-3
    assert p.refer_rho == 1.2e-3
    assert p.hu == 1.0
    assert p.hys == 1.0
    assert p.shape == 1.0
    assert p.xscale == 1.0
    assert p.scale == 1.0
    assert p.tcut == 1.0e20
    assert p.fail == 0

    # Derived moduli for nu=0.0:
    # G = E / (2*(1+0)) = 60.0
    # K = E / (3*(1-0)) = 40.0
    # cii = K + 4/3*G = 40 + 80 = 120.0
    # cij = K - 2/3*G = 40 - 40 = 0.0
    # c0 = sqrt(E0 / rho0) = sqrt(120.0 / 1.2e-3) = sqrt(100000) = 316.227766
    assert math.isclose(p.g, 60.0, rel_tol=1e-12)
    assert math.isclose(p.bulk, 40.0, rel_tol=1e-12)
    assert math.isclose(p.cii, 120.0, rel_tol=1e-12)
    assert math.isclose(p.cij, 0.0, rel_tol=1e-12)
    assert math.isclose(p.sound_speed0, math.sqrt(120.0 / 1.2e-3), rel_tol=1e-6)

    # For nu=0.25:
    p_nu = Law190Params(rho0=1.0, e0=100.0, nu=0.25)
    # G = 100 / 2.5 = 40.0
    # K = 100 / (3*0.5) = 66.666667
    # cii = 66.666667 + 53.333333 = 120.0
    # cij = 66.666667 - 26.666667 = 40.0
    assert math.isclose(p_nu.g, 40.0, rel_tol=1e-12)
    assert math.isclose(p_nu.bulk, 100.0 / 1.5, rel_tol=1e-12)
    assert math.isclose(p_nu.cii, 120.0, rel_tol=1e-12)
    assert math.isclose(p_nu.cij, 40.0, rel_tol=1e-12)


def test_build_law190_factory():
    """Verify build_law190 accepts dict, MatLaw190, Material, and Law190Params."""
    # From dict
    d = {
        "id": 1,
        "title": "DuBoisFoamA",
        "rho": 0.05,
        "e0": 150.0,
        "nu": 0.1,
        "hu": 0.4,
        "shape": 1.5,
        "table_id": 10,
        "xscale": 1.2,
        "scale": 1.5,
        "tcut": 2.5,
        "fail": 1,
    }
    m1 = build_law190(d)
    assert isinstance(m1, Material)
    assert m1.id == 1
    assert m1.law == 190
    assert m1.rho0 == 0.05
    assert m1.title == "DuBoisFoamA"
    assert m1.params["e0"] == 150.0
    assert m1.params["hu"] == 0.4
    assert m1.params["shape"] == 1.5
    assert m1.params["table_id"] == 10
    assert m1.params["tcut"] == 2.5
    assert m1.params["fail"] == 1

    # From MatLaw190
    m_ent = MatLaw190(
        id=2, rho=0.08, e0=200.0, nu=0.0, hu=0.5, shape=2.0,
        fun_1=20, xscale_1=1.0, scale_1=1.0, tcut=4.0, fail=0, title="DuBoisB"
    )
    m2 = build_law190(m_ent)
    assert m2.id == 2
    assert m2.params["E0"] == 200.0
    assert m2.params["hu"] == 0.5
    assert m2.params["tcut"] == 4.0

    # From Law190Params
    p = Law190Params(rho0=0.1, e0=300.0, hu=0.3, title="DuBoisC")
    m3 = build_law190(p)
    assert m3.rho0 == 0.1
    assert m3.params["E0"] == 300.0
    assert m3.params["hu"] == 0.3


# ---------------------------------------------------------------------------
# 2. Kinematics and Principal Decomposition
# ---------------------------------------------------------------------------

def test_law190_principal_kinematics():
    """Verify principal stretch and engineering strain decomposition e_k = 1.0 - lambda_k."""
    # Uniaxial compression along X with 30% compression: lambda_1 = 0.7, lambda_2 = lambda_3 = 1.0
    # F = diag(0.7, 1.0, 1.0)
    F = np.diag([0.7, 1.0, 1.0])
    extra = {"F": F[np.newaxis, :, :]}

    mat = Law190Params(rho0=1.0, e0=100.0, nu=0.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))

    sig_out, epsp_out, _ = solid_update(mat, sig, deps, dt=1e-3, extra=extra, return_tuple=True)

    # e_1 = 1 - 0.7 = 0.3; e_2 = 1 - 1.0 = 0.0; e_3 = 0.0
    # norm = sqrt(0.3^2 + 0 + 0) = 0.3
    assert math.isclose(epsp_out[0], 0.3, rel_tol=1e-6)

    # State uv190 should store Green-Lagrange strain:
    # E_11 = 0.5*(0.7^2 - 1) = 0.5*(0.49 - 1) = -0.255
    uv = extra["uv190"]
    assert math.isclose(uv[0, 0], -0.255, rel_tol=1e-6)
    assert math.isclose(uv[0, 1], 0.0, abs_tol=1e-12)
    assert math.isclose(uv[0, 2], 0.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# 3. Compressive Stress vs Tabulated Curve
# ---------------------------------------------------------------------------

def test_law190_compressive_stress_tabulated():
    """Verify compressive stress-strain response matches tabulated 1D curve."""
    # Tabulated crushable foam curve:
    # Linear elastic up to 5% strain, flat plateau at 2.0 MPa up to 50%, densification beyond 50%
    xs = np.array([0.0, 0.05, 0.50, 0.70, 0.80])
    ys = np.array([0.0, 2.00, 2.00, 6.00, 20.0])
    tbl = FunctTable(1, xs, ys)

    mat = Law190Params(rho0=1.0, e0=40.0, nu=0.0, table=tbl)

    # Test at several compression levels:
    test_points = [
        (0.025, 1.00),   # Linear region: slope = 2.0 / 0.05 = 40.0 -> 0.025 * 40 = 1.0
        (0.05, 2.00),    # Yield point
        (0.25, 2.00),    # Plateau region
        (0.50, 2.00),    # End of plateau
        (0.60, 4.00),    # Densification halfway between 2.0 and 6.0
        (0.70, 6.00),    # Densification point
        (0.75, 13.00),   # Densification halfway between 6.0 and 20.0
    ]

    for eng_strain, expected_stress in test_points:
        lam = 1.0 - eng_strain
        F = np.diag([lam, 1.0, 1.0])
        extra = {"F": F[np.newaxis, :, :]}
        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))

        sig_new = solid_update(mat, sig, deps, dt=1e-3, extra=extra)

        # Cauchy stress in compression is negative: sig_xx = - expected_stress
        compressive_stress = - sig_new[0, 0]
        assert math.isclose(compressive_stress, expected_stress, rel_tol=1e-5), (
            f"At strain {eng_strain}: got {compressive_stress}, expected {expected_stress}"
        )
        # Lateral stresses must be zero (nu=0)
        assert abs(sig_new[0, 1]) < 1e-10
        assert abs(sig_new[0, 2]) < 1e-10


# ---------------------------------------------------------------------------
# 4. Strain-Rate Stiffening Under Dynamic Rates
# ---------------------------------------------------------------------------

def test_law190_strain_rate_stiffening():
    """Verify strain-rate stiffening under dynamic loading rates (2D table / curve family)."""
    # 2D table grid:
    # strain: [0.0, 0.1, 0.4, 0.7]
    # strain rates: [0.0, 10.0, 100.0]
    # At rate 0: plateau at 1.0 MPa
    # At rate 10: plateau at 1.5 MPa
    # At rate 100: plateau at 2.5 MPa
    xg = np.array([0.0, 0.1, 0.4, 0.7])
    rates = np.array([0.0, 10.0, 100.0])
    Y = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.5, 2.5],
        [1.0, 1.5, 2.5],
        [5.0, 6.0, 8.0],
    ])
    tbl = (xg, rates, Y)

    mat = Law190Params(rho0=1.0, e0=10.0, nu=0.0, table=tbl)

    # 1. Quasistatic compression at strain 0.25 (plateau region) with rate = 0
    F = np.diag([0.75, 1.0, 1.0])
    extra_static = {"F": F[np.newaxis, :, :], "rates": np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])}
    sig_static = solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-3, extra=extra_static)
    stress_static = - sig_static[0, 0]
    assert math.isclose(stress_static, 1.0, rel_tol=1e-5)

    # 2. Dynamic compression at rate 10.0 s^-1: engineering strain rate = deps_dot * lam = deps_dot * 0.75
    # To get dr = 10.0, set deps_dot = 10.0 / 0.75
    rate_10_xx = 10.0 / 0.75
    extra_rate10 = {"F": F[np.newaxis, :, :], "rates": np.array([[rate_10_xx, 0.0, 0.0, 0.0, 0.0, 0.0]])}
    sig_rate10 = solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-3, extra=extra_rate10)
    stress_rate10 = - sig_rate10[0, 0]
    assert math.isclose(stress_rate10, 1.5, rel_tol=1e-5)

    # 3. Dynamic compression at rate 100.0 s^-1
    rate_100_xx = 100.0 / 0.75
    extra_rate100 = {"F": F[np.newaxis, :, :], "rates": np.array([[rate_100_xx, 0.0, 0.0, 0.0, 0.0, 0.0]])}
    sig_rate100 = solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-3, extra=extra_rate100)
    stress_rate100 = - sig_rate100[0, 0]
    assert math.isclose(stress_rate100, 2.5, rel_tol=1e-5)

    # Clear rate stiffening ordering: static < rate 10 < rate 100
    assert stress_static < stress_rate10 < stress_rate100


# ---------------------------------------------------------------------------
# 5. Cyclic Loading-Unloading Hysteresis and Energy Dissipation
# ---------------------------------------------------------------------------

def test_law190_cyclic_loading_unloading_hysteresis():
    """Verify cyclic loading-unloading hysteresis and energy dissipation (condamage.F)."""
    # 1D linear-plateau curve
    xs = np.array([0.0, 0.1, 0.5, 0.8])
    ys = np.array([0.0, 2.0, 2.0, 10.0])
    tbl = FunctTable(1, xs, ys)

    # Foam with hysteretic unloading parameter HU = 0.2 (80% hysteresis reduction) and SHAPE = 1.0
    mat_hys = Law190Params(rho0=1.0, e0=20.0, nu=0.0, hu=0.2, shape=1.0, table=tbl)

    # Shared persistent state across cycles
    extra = {"uv190": np.zeros((1, 16))}

    # Step 1: Load monotonically from e = 0.0 to e = 0.4
    strains_load = np.linspace(0.0, 0.4, 21)
    stresses_load = []
    for eps in strains_load:
        F = np.diag([1.0 - eps, 1.0, 1.0])
        extra["F"] = F[np.newaxis, :, :]
        sig = solid_update(mat_hys, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-3, extra=extra)
        stresses_load.append(- sig[0, 0])
        # During monotonic loading, damage must be 0
        assert math.isclose(extra["uv190"][0, 13], 0.0, abs_tol=1e-10)

    # Step 2: Unload from e = 0.4 down to e = 0.05
    strains_unload = np.linspace(0.4, 0.05, 15)[1:]  # exclude the reversal point
    stresses_unload = []
    damages_unload = []
    for eps in strains_unload:
        F = np.diag([1.0 - eps, 1.0, 1.0])
        extra["F"] = F[np.newaxis, :, :]
        sig = solid_update(mat_hys, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-3, extra=extra)
        stresses_unload.append(- sig[0, 0])
        dam = extra["uv190"][0, 13]
        damages_unload.append(dam)

    # Verify that during unloading:
    # 1. Damage D > 0 (condamage.F damage active)
    assert all(d > 0.0 for d in damages_unload)
    # 2. Unloading stress is strictly lower than monotonic loading stress at the same strain
    for i, eps_un in enumerate(strains_unload):
        # Monotonic stress at this strain is 2.0 (plateau)
        sig_un = stresses_unload[i]
        assert sig_un < 2.0, f"Unloading stress {sig_un} should be softened compared to monotonic 2.0"

    # 3. As we approach e -> 0, ratio -> 0, damage -> (1 - HU) = 0.8, dam_factor -> 0.2
    assert math.isclose(damages_unload[-1], 0.8, rel_tol=0.25)

    # Step 3: Now compare with HU = 1.0 (elastic, no hysteresis)
    extra_elastic = {"uv190": np.zeros((1, 16))}
    mat_elastic = Law190Params(rho0=1.0, e0=20.0, nu=0.0, hu=1.0, shape=1.0, table=tbl)

    # Load to 0.4
    F_peak = np.diag([0.6, 1.0, 1.0])
    extra_elastic["F"] = F_peak[np.newaxis, :, :]
    solid_update(mat_elastic, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-3, extra=extra_elastic)

    # Unload to 0.2
    F_mid = np.diag([0.8, 1.0, 1.0])
    extra_elastic["F"] = F_mid[np.newaxis, :, :]
    sig_el_mid = solid_update(mat_elastic, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-3, extra=extra_elastic)

    # For HU=1.0, damage must be 0 and stress remains at 2.0
    assert math.isclose(extra_elastic["uv190"][0, 13], 0.0, abs_tol=1e-12)
    assert math.isclose(- sig_el_mid[0, 0], 2.0, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# 6. Tensile Cutoff and Erosion
# ---------------------------------------------------------------------------

def test_law190_tensile_cutoff_clamping():
    """Verify tensile stress cutoff clamps tensile stress when fail=0."""
    mat = Law190Params(rho0=1.0, e0=100.0, nu=0.0, tcut=3.0, fail=0)

    # Apply 10% tension along X: lambda_1 = 1.10 -> e_1 = -0.10
    # Expected elastic tensile stress: E0 * 0.10 = 10.0 MPa
    F = np.diag([1.10, 1.0, 1.0])
    extra = {"F": F[np.newaxis, :, :]}

    sig = solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-3, extra=extra)

    # Tensile stress sig_xx should be clamped to tcut = 3.0 MPa
    assert math.isclose(sig[0, 0], 3.0, rel_tol=1e-5)


def test_law190_tensile_failure_and_erosion():
    """Verify tensile cutoff causes element failure and erosion when fail=1."""
    mat = Law190Params(rho0=1.0, e0=100.0, nu=0.0, tcut=3.0, fail=1)

    extra = {"uv190": np.zeros((1, 16))}

    # Step 1: Moderate tension below tcut: 2% tension -> stress = 2.0 MPa < 3.0
    F1 = np.diag([1.02, 1.0, 1.0])
    extra["F"] = F1[np.newaxis, :, :]
    sig1 = solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-3, extra=extra)
    assert math.isclose(sig1[0, 0], 2.0, rel_tol=1e-2)
    assert extra["uv190"][0, 15] == 0.0  # Not eroded

    # Step 2: High tension exceeding tcut: 5% tension -> stress would be 5.0 > 3.0
    F2 = np.diag([1.05, 1.0, 1.0])
    extra["F"] = F2[np.newaxis, :, :]
    sig2 = solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-3, extra=extra)

    # Element must fail: stress dropped to 0 and marked eroded
    assert np.allclose(sig2, 0.0)
    assert extra["uv190"][0, 15] == 1.0  # Eroded flag set

    # Step 3: Subsequent cycle remains completely eroded
    sig3 = solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-3, extra=extra)
    assert np.allclose(sig3, 0.0)
    assert extra["uv190"][0, 15] == 1.0


def test_law190_hydrostatic_tension_erosion():
    """Verify triaxial hydrostatic tension failure when P < -tcut with fail=1."""
    mat = Law190Params(rho0=1.0, e0=60.0, nu=0.0, tcut=2.0, fail=1)
    extra = {"uv190": np.zeros((1, 16))}

    # Triaxial stretch: lambda_1 = lambda_2 = lambda_3 = 1.05
    F = np.diag([1.05, 1.05, 1.05])
    extra["F"] = F[np.newaxis, :, :]

    sig = solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-3, extra=extra)

    assert np.allclose(sig, 0.0)
    assert extra["uv190"][0, 15] == 1.0


# ---------------------------------------------------------------------------
# 7. Sound Speed and Algorithmic Tangent Stiffness
# ---------------------------------------------------------------------------

def test_law190_sound_speed():
    """Verify dilatational sound speed increases during foam densification (sigeps190.F)."""
    xs = np.array([0.0, 0.1, 0.5, 0.8])
    ys = np.array([0.0, 1.0, 1.0, 100.0])  # Steep densification slope at 80%
    tbl = FunctTable(1, xs, ys)

    mat = Law190Params(rho0=1.0, e0=10.0, nu=0.0, table=tbl)

    # Initial sound speed at zero strain: c = sqrt(E0 / rho) = sqrt(10 / 1) = 3.162277
    c0 = sound_speed(mat, rho=1.0)
    assert math.isclose(c0, math.sqrt(10.0), rel_tol=1e-6)

    # During plateau (e = 0.3): slope = 0.0, so max(slope, E0) = E0 -> c = sqrt(10 / 1)
    F_plat = np.diag([0.7, 1.0, 1.0])
    extra_plat = {"F": F_plat[np.newaxis, :, :]}
    _, _, c_plat = solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-3, extra=extra_plat, return_tuple=True)
    assert math.isclose(c_plat[0], math.sqrt(10.0), rel_tol=1e-6)

    # During densification (e = 0.7): slope = (100 - 1) / (0.8 - 0.5) = 99 / 0.3 = 330.0
    F_dens = np.diag([0.3, 1.0, 1.0])
    extra_dens = {"F": F_dens[np.newaxis, :, :]}
    _, _, c_dens = solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-3, extra=extra_dens, return_tuple=True)
    assert math.isclose(c_dens[0], math.sqrt(330.0), rel_tol=1e-5)
    assert c_dens[0] > c0


def test_law190_tangent_stiffness():
    """Verify algorithmic consistent solid tangent stiffness matrix."""
    mat = Law190Params(rho0=1.0, e0=120.0, nu=0.0)
    C = solid_tangent(mat)

    assert C.shape == (6, 6)
    # Symmetry check
    assert np.allclose(C, C.T)

    # For nu=0: C_11 = 120.0, C_12 = 0.0, C_44 = G = 60.0
    assert math.isclose(C[0, 0], 120.0, rel_tol=1e-12)
    assert math.isclose(C[1, 1], 120.0, rel_tol=1e-12)
    assert math.isclose(C[2, 2], 120.0, rel_tol=1e-12)
    assert math.isclose(C[0, 1], 0.0, abs_tol=1e-12)
    assert math.isclose(C[3, 3], 60.0, rel_tol=1e-12)


def test_law190_shell_update_raises():
    """Verify shell_update and shell_tangent raise NotImplementedError for LAW190."""
    mat = Law190Params(e0=100.0)
    with pytest.raises(NotImplementedError, match="solid elements only"):
        shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)))
    with pytest.raises(NotImplementedError, match="solid elements only"):
        shell_tangent(mat)


# ---------------------------------------------------------------------------
# 8. Starter Deck Parsing for /MAT/LAW190 and /MAT/FOAM_DUBOIS
# ---------------------------------------------------------------------------

def test_law190_starter_deck_parsing_fixed_format():
    """Verify starter fixed-format deck reading for /MAT/LAW190."""
    deck_lines = [
        "/BEGIN",
        "TEST FIXED DECK",
        "/MAT/LAW190/101",
        "Du Bois Crushable Foam",
        "              0.0012",
        "               150.0                 0.1",
        "                0.35                 1.5",
        "        10                 1.0                 1.0",
        "                25.0         1",
        "/END",
    ]
    blocks = read_deck(deck_lines)
    model = parse_starter_deck(blocks)

    assert 101 in model.materials
    mat = model.materials[101]
    assert mat.id == 101
    assert mat.law == 190
    assert math.isclose(mat.rho0, 0.0012, rel_tol=1e-6)
    assert math.isclose(mat.params["E"], 150.0, rel_tol=1e-6)
    assert math.isclose(mat.params["nu"], 0.1, rel_tol=1e-6)
    assert math.isclose(mat.params["hu"], 0.35, rel_tol=1e-6)
    assert math.isclose(mat.params["shape"], 1.5, rel_tol=1e-6)
    assert mat.params["table_id"] == 10
    assert math.isclose(mat.params["tcut"], 25.0, rel_tol=1e-6)
    assert mat.params["fail"] == 1

    assert 101 in model.mat_law190s
    m190 = model.mat_law190s[101]
    assert m190.id == 101
    assert math.isclose(m190.e0, 150.0, rel_tol=1e-6)
    assert math.isclose(m190.hu, 0.35, rel_tol=1e-6)
    assert math.isclose(m190.tcut, 25.0, rel_tol=1e-6)


def test_law190_starter_deck_parsing_free_format_alias():
    """Verify starter free-format deck reading for /MAT/FOAM_DUBOIS alias."""
    deck_lines = [
        "/BEGIN",
        "TEST FREE DECK",
        "/MAT/FOAM_DUBOIS/202",
        "Du Bois Free Format Alias",
        "0.0025",
        "250.0 0.0",
        "0.25 2.0",
        "15 1.0 1.0",
        "50.0 0",
        "/END",
    ]
    blocks = read_deck(deck_lines)
    model = parse_starter_deck(blocks)

    assert 202 in model.materials
    mat = model.materials[202]
    assert mat.id == 202
    assert mat.law == 190
    assert math.isclose(mat.rho0, 0.0025, rel_tol=1e-6)
    assert math.isclose(mat.params["E0"], 250.0, rel_tol=1e-6)
    assert math.isclose(mat.params["hu"], 0.25, rel_tol=1e-6)
    assert math.isclose(mat.params["shape"], 2.0, rel_tol=1e-6)
    assert mat.params["table_id"] == 15
    assert math.isclose(mat.params["tcut"], 50.0, rel_tol=1e-6)
    assert mat.params["fail"] == 0


# ---------------------------------------------------------------------------
# 9. Materials Module Dispatcher Integration
# ---------------------------------------------------------------------------

def test_law190_materials_dispatcher_integration():
    """Verify integration with pyradioss.materials dispatcher functions."""
    xs = np.array([0.0, 0.1, 0.5])
    ys = np.array([0.0, 5.0, 5.0])
    tbl = FunctTable(1, xs, ys)

    mat_obj = build_law190({
        "id": 1,
        "rho": 1.2e-3,
        "e0": 50.0,
        "nu": 0.0,
        "hu": 0.5,
        "shape": 1.0,
        "table": tbl,
    })

    # Dispatch solid_update
    F = np.diag([0.8, 1.0, 1.0])
    extra = {"F": F[np.newaxis, :, :]}
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))

    sig_out, epsp_out, c_out = materials.solid_update(mat_obj, sig, deps, dt=1e-3, extra=extra)
    assert math.isclose(- sig_out[0, 0], 5.0, rel_tol=1e-5)
    assert math.isclose(epsp_out[0], 0.2, rel_tol=1e-5)

    # Dispatch sound_speed
    c_disp = materials.sound_speed(mat_obj, rho=1.2e-3)
    assert c_disp > 0.0

    # Dispatch solid_tangent
    C_disp = materials.solid_tangent(mat_obj)
    assert C_disp.shape == (6, 6)

    # Dispatch extra_shapes
    shapes = materials.extra_shapes(mat_obj)
    assert "uv190" in shapes
    assert shapes["uv190"] == (16,)

    # Dispatch needs_defgrad & needs_env
    assert materials.needs_defgrad(mat_obj) is True
    assert materials.needs_env(mat_obj) is True

    # Dispatch shell_update raises NotImplementedError
    with pytest.raises(NotImplementedError):
        materials.shell_update(mat_obj, np.zeros((1, 3)), np.zeros((1, 3)))


# ---------------------------------------------------------------------------
# 10. Vectorized Multi-Element Group Consistency
# ---------------------------------------------------------------------------

def test_law190_vectorized_multielement():
    """Verify vectorized multi-element group update consistency."""
    xs = np.array([0.0, 0.1, 0.5, 0.8])
    ys = np.array([0.0, 2.0, 2.0, 20.0])
    tbl = FunctTable(1, xs, ys)

    mat = Law190Params(rho0=1.0, e0=20.0, nu=0.0, hu=0.5, table=tbl)

    n_elem = 4
    # 4 elements with different strain states:
    # elem 0: zero strain
    # elem 1: 5% compression (elastic)
    # elem 2: 30% compression (plateau)
    # elem 3: 65% compression (densification)
    strains = [0.0, 0.05, 0.30, 0.65]
    F_stack = np.zeros((n_elem, 3, 3))
    for i, e in enumerate(strains):
        F_stack[i] = np.diag([1.0 - e, 1.0, 1.0])

    extra = {"F": F_stack}
    sig = np.zeros((n_elem, 6))
    deps = np.zeros((n_elem, 6))

    sig_new, epsp_new, c_new = solid_update(mat, sig, deps, dt=1e-3, extra=extra, return_tuple=True)

    assert sig_new.shape == (n_elem, 6)
    assert epsp_new.shape == (n_elem,)
    assert c_new.shape == (n_elem,)

    # Check individual element values against serial updates
    for i, e in enumerate(strains):
        F_single = np.diag([1.0 - e, 1.0, 1.0])[np.newaxis, :, :]
        sig_single, epsp_single, c_single = solid_update(
            mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-3, extra={"F": F_single}, return_tuple=True
        )
        assert np.allclose(sig_new[i], sig_single[0])
        assert math.isclose(epsp_new[i], epsp_single[0], rel_tol=1e-6)
        assert math.isclose(c_new[i], c_single[0], rel_tol=1e-6)

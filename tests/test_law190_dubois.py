"""
Unit test suite for LAW190 Du Bois Crushable Foam Material (/MAT/LAW190, /MAT/FOAM_DUBOIS).

Fortran references:
- engine/source/materials/mat/mat190/sigeps190.F
- engine/source/materials/mat/mat190/conversion.F
- engine/source/materials/mat/mat190/condamage.F
- starter/source/materials/mat/mat190/hm_read_mat190.F
- starter/source/materials/mat/mat190/law190_upd.F90

Verifies:
1. Physics registry of LAW190, FOAM_DUBOIS, DUBOIS.
2. Defensive empty array handling.
3. Linear elastic compression response (fallback when no table is given).
4. Tabulated 1D stress-strain curve evaluation.
5. Strain-rate dependence with 2D table / multi-curve rate stiffening.
6. Hysteretic energy dissipation and unloading damage (condamage.F).
7. Tensile stress cutoff clamping (fail=0) vs erosion (fail=1).
8. Acoustic dilatational sound speed calculation (sigeps190.F lines 406-413).
9. Consistent algorithmic solid tangent matrix (conversion.F).
10. Solid-only constraint: shell_update raises NotImplementedError.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials import law190_dubois
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
from pyradioss.model.entities import Material
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY


# ============================================================================
# 1. Registration tests
# ============================================================================

def test_law190_registry():
    """Verify LAW190 is registered under all standard aliases."""
    assert "LAW190" in MAT_PHYSICS_REGISTRY
    assert "FOAM_DUBOIS" in MAT_PHYSICS_REGISTRY
    assert "DUBOIS" in MAT_PHYSICS_REGISTRY
    assert 190 in MAT_PHYSICS_REGISTRY


# ============================================================================
# 2. Empty array handling
# ============================================================================

def test_law190_empty_arrays():
    """Verify solid_update and tangent gracefully handle empty element arrays."""
    mat = build_law190({"e0": 100.0, "rho": 1.0})
    sig = np.empty((0, 6), dtype=float)
    deps = np.empty((0, 6), dtype=float)
    sig_out, epsp, c = solid_update(mat, sig, deps, None, dt=0.0, return_tuple=True)
    assert sig_out.shape == (0, 6)
    assert epsp.shape == (0,)
    assert c.shape == (0,)

    D = solid_tangent(mat, sig=sig)
    assert D.shape == (0, 6, 6)


# ============================================================================
# 3. Fallback linear elasticity
# ============================================================================

def test_law190_elastic_compression():
    """Without a curve, response follows initial Young's modulus E0."""
    E0 = 150.0
    rho0 = 1.0e-3
    mat = build_law190({"e0": E0, "rho": rho0, "nu": 0.0})

    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    # Uniaxial compressive strain increment of 2% along X
    # Note: engineering strain e = 1 - lambda = 0.02
    deps[0, 0] = -0.02
    F = np.diag([0.98, 1.0, 1.0])
    extra = {"F": F[np.newaxis, :, :]}

    sig_out, epsp, c = solid_update(mat, sig, deps, None, dt=1.0, extra=extra, return_tuple=True)

    # In uniaxial compression with nu=0, J = 0.98, lambda_1 = 0.98
    # Engineering strain e_1 = 1 - 0.98 = 0.02
    # Compressive stress = E0 * e_1 = 150 * 0.02 = 3.0
    # Cauchy stress sigma_xx = -3.0
    assert math.isclose(sig_out[0, 0], -3.0, rel_tol=1e-2)
    assert epsp[0] > 0.0
    assert math.isclose(c[0], math.sqrt(E0 / rho0), rel_tol=1e-3)


# ============================================================================
# 4. Tabulated 1D curve lookup
# ============================================================================

def test_law190_tabulated_1d_curve():
    """Verify piecewise linear curve evaluation matching condomage.F / conversion.F."""
    # Typical foam response: initial elastic (e=0..0.1), plateau (e=0.1..0.5), densification (e>0.5)
    xs = np.array([0.0, 0.1, 0.5, 0.8])
    ys = np.array([0.0, 2.0, 3.0, 15.0])  # Stress (MPa) vs engineering strain (e >= 0 in compression)
    table_1d = (xs, ys)

    mat = build_law190({
        "e0": 20.0,
        "rho": 1.0e-3,
        "table": table_1d,
        "hu": 1.0,  # no hysteresis damage
    })

    # Strain e = 0.3 (in the plateau region between 0.1 and 0.5)
    # Expected stress = 2.0 + (0.3 - 0.1)/(0.5 - 0.1) * (3.0 - 2.0) = 2.0 + 0.5 * 1.0 = 2.5 MPa
    lambda_1 = 0.7  # e_1 = 1.0 - 0.7 = 0.3
    F = np.diag([lambda_1, 1.0, 1.0])
    extra = {"F": F[np.newaxis, :, :]}

    sig = np.zeros((1, 6))
    deps = np.array([[-0.3, 0.0, 0.0, 0.0, 0.0, 0.0]])

    sig_out, epsp, _ = solid_update(mat, sig, deps, None, dt=1.0, extra=extra, return_tuple=True)

    # In uniaxial compression with nu=0: J = 0.7, lambda_1 = 0.7
    # Cauchy stress = - (lambda_1 / J) * sig_dyn = - (0.7 / 0.7) * 2.5 = -2.5 MPa
    assert math.isclose(sig_out[0, 0], -2.5, rel_tol=1e-2)


# ============================================================================
# 5. Strain rate stiffening (2D table)
# ============================================================================

def test_law190_rate_dependent_stiffening():
    """Verify strain rate stiffening across multi-rate table."""
    xs = np.array([0.0, 0.2, 0.5])
    rates = np.array([0.0, 10.0, 100.0])
    # Stresses increase with rate
    Y = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 3.0, 5.0],
        [4.0, 6.0, 10.0],
    ])
    table_2d = (xs, rates, Y)

    mat = build_law190({
        "e0": 50.0,
        "rho": 1.0e-3,
        "table": table_2d,
        "hu": 1.0,
    })

    lambda_1 = 0.8  # e_1 = 0.2
    F = np.diag([lambda_1, 1.0, 1.0])

    # Case 1: Quasistatic rate = 0 (dt = large or rates = 0)
    extra_qs = {"F": F[np.newaxis, :, :], "rates": np.zeros((1, 6))}
    sig_qs, _, _ = solid_update(mat, np.zeros((1, 6)), np.array([[-0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]), None, dt=1.0, extra=extra_qs, return_tuple=True)

    # Case 2: High dynamic rate = 100 /s
    # Engineering strain rate dr = |eps_dot| * lambda_1 = (0.2 / 0.0016) * 0.8 = 125 * 0.8 = 100
    extra_dyn = {"F": F[np.newaxis, :, :], "rates": np.array([[125.0, 0.0, 0.0, 0.0, 0.0, 0.0]])}
    sig_dyn, _, _ = solid_update(mat, np.zeros((1, 6)), np.array([[-0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]), None, dt=0.0016, extra=extra_dyn, return_tuple=True)

    # Quasistatic stress at e=0.2 is Y[1, 0] = 2.0 MPa -> Cauchy = -2.0 MPa
    assert math.isclose(sig_qs[0, 0], -2.0, rel_tol=1e-2)
    # Dynamic stress at e=0.2 and rate=100 is Y[1, 2] = 5.0 MPa -> Cauchy = -5.0 MPa
    assert math.isclose(sig_dyn[0, 0], -5.0, rel_tol=1e-2)
    # Dynamic stress must be strictly stiffer than quasistatic
    assert abs(sig_dyn[0, 0]) > abs(sig_qs[0, 0])


# ============================================================================
# 6. Hysteretic energy & unloading damage (condamage.F)
# ============================================================================

def test_law190_hysteresis_unloading_damage():
    """Verify hysteretic energy tracking and unloading damage formulation."""
    xs = np.array([0.0, 0.2, 0.5])
    ys = np.array([0.0, 4.0, 10.0])
    table_1d = (xs, ys)

    HU = 0.3    # 70% hysteretic loss
    SHAPE = 2.0
    mat = build_law190({
        "e0": 50.0,
        "rho": 1.0e-3,
        "table": table_1d,
        "hu": HU,
        "shape": SHAPE,
    })

    extra = {}

    # Step 1: Load to e_1 = 0.5 (lambda_1 = 0.5)
    F_peak = np.diag([0.5, 1.0, 1.0])
    extra["F"] = F_peak[np.newaxis, :, :]
    sig1, epsp1, _ = solid_update(mat, np.zeros((1, 6)), np.array([[-0.5, 0.0, 0.0, 0.0, 0.0, 0.0]]), None, dt=1.0, extra=extra, return_tuple=True)

    # In loading, damage = 0
    assert math.isclose(extra["uv190"][0, 13], 0.0, abs_tol=1e-6)
    w_max = extra["uv190"][0, 12]
    assert w_max > 0.0

    # Step 2: Unload to e_1 = 0.2 (lambda_1 = 0.8)
    F_unload = np.diag([0.8, 1.0, 1.0])
    extra["F"] = F_unload[np.newaxis, :, :]
    sig2, epsp2, _ = solid_update(mat, sig1, np.array([[0.3, 0.0, 0.0, 0.0, 0.0, 0.0]]), None, dt=1.0, extra=extra, return_tuple=True)

    # In unloading: w_hys < w_max
    w_hys = extra["uv190"][0, 14]
    assert w_hys < w_max
    damage = extra["uv190"][0, 13]
    # damage = (1 - HU) * (1 - (w_hys / w_max)^SHAPE) > 0
    assert damage > 0.0
    # Unloading stress must be softened by (1 - damage)
    expected_dam_factor = 1.0 - damage
    # Quasistatic stress at e=0.2 is 4.0 MPa -> softened Cauchy stress = - 4.0 * (1 - damage)
    assert math.isclose(sig2[0, 0], -4.0 * expected_dam_factor, rel_tol=1e-2)


# ============================================================================
# 7. Tensile cutoff (fail=0 clamping vs fail=1 erosion)
# ============================================================================

def test_law190_tensile_cutoff_clamping():
    """With fail=0, tensile stresses exceeding tcut are clamped to tcut."""
    TCUT = 1.5
    mat = build_law190({
        "e0": 100.0,
        "rho": 1.0e-3,
        "tcut": TCUT,
        "fail": 0,
    })

    # Uniaxial tension along X: lambda_1 = 1.1 (e_1 = -0.1)
    # Linear elastic tension = E0 * e = 100 * 0.1 = 10.0 MPa >> 1.5 MPa
    F = np.diag([1.1, 1.0, 1.0])
    extra = {"F": F[np.newaxis, :, :]}

    sig_out, _, _ = solid_update(mat, np.zeros((1, 6)), np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]), None, dt=1.0, extra=extra, return_tuple=True)

    # Clamped to TCUT
    assert math.isclose(sig_out[0, 0], TCUT, rel_tol=1e-3)


def test_law190_tensile_cutoff_erosion():
    """With fail=1, element is eroded (sig=0, uv190[15]=1.0) when exceeding tcut."""
    TCUT = 1.5
    mat = build_law190({
        "e0": 100.0,
        "rho": 1.0e-3,
        "tcut": TCUT,
        "fail": 1,
    })

    F = np.diag([1.1, 1.0, 1.0])
    extra = {}
    extra["F"] = F[np.newaxis, :, :]

    sig_out, _, _ = solid_update(mat, np.zeros((1, 6)), np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]), None, dt=1.0, extra=extra, return_tuple=True)

    # Eroded: stress zeroed and erosion flag set
    assert np.allclose(sig_out[0], 0.0)
    assert extra["uv190"][0, 15] == 1.0


# ============================================================================
# 8. Dilatational sound speed
# ============================================================================

def test_law190_sound_speed_evolution():
    """Sound speed uses max(E0, max(slopes)) matching sigeps190.F lines 406-413."""
    E0 = 100.0
    rho0 = 1.0e-3
    xs = np.array([0.0, 0.5, 0.8])
    ys = np.array([0.0, 5.0, 100.0])  # slope between 0.5 and 0.8 is (100 - 5)/0.3 = 316.67 >> 100
    mat = build_law190({
        "e0": E0,
        "rho": rho0,
        "table": (xs, ys),
    })

    # At e=0.6, slope is ~316.67 > 100.0
    F = np.diag([0.4, 1.0, 1.0])  # e = 0.6
    extra = {"F": F[np.newaxis, :, :]}
    _, _, c = solid_update(mat, np.zeros((1, 6)), np.array([[-0.6, 0.0, 0.0, 0.0, 0.0, 0.0]]), None, dt=1.0, extra=extra, return_tuple=True)

    expected_slope = (100.0 - 5.0) / 0.3
    expected_c = math.sqrt(expected_slope / rho0)
    assert math.isclose(c[0], expected_c, rel_tol=1e-2)


# ============================================================================
# 9. Tangent stiffness matrix
# ============================================================================

def test_law190_solid_tangent_symmetry_and_psd():
    """Consistent solid tangent must be symmetric and positive definite."""
    mat = build_law190({"e0": 200.0, "rho": 1.0e-3, "nu": 0.2})
    D = solid_tangent(mat, sig=np.zeros((1, 6)))[0]

    # Symmetry
    assert np.allclose(D, D.T)
    # Positive definiteness
    eigvals = np.linalg.eigvalsh(D)
    assert np.all(eigvals > 0.0)


# ============================================================================
# 10. Shell rejection constraint
# ============================================================================

def test_law190_shell_rejection():
    """LAW190 is formulated for 3D solids only; shell updates must raise NotImplementedError."""
    mat = build_law190({"e0": 100.0})
    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        shell_update(mat, np.zeros((1, 6)), np.zeros((1, 6)))
    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        shell_tangent(mat)

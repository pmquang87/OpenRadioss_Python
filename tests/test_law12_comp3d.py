"""Unit tests for OpenRadioss LAW12 (3D Orthotropic Elastic Composite).

References:
  - engine/source/materials/mat/mat012/m12law.F (referenced as sigeps12.F)
  - starter/source/materials/mat/mat012/hm_read_mat12.F
"""

import math
import numpy as np
import pytest

from pyradioss.materials import law12_comp3d


def test_law12_orthotropic_stiffness_matrix():
    """Verify 9-parameter compliance inversion matching hm_read_mat12.F lines 221-255."""
    E11 = 140000.0
    E22 = 12000.0
    E33 = 10000.0
    nu12 = 0.28
    nu23 = 0.35
    nu31 = 0.02
    G12 = 5500.0
    G23 = 3500.0
    G31 = 4500.0
    rho0 = 1.6e-9

    mat = law12_comp3d.build_law12(
        E11=E11,
        E22=E22,
        E33=E33,
        nu12=nu12,
        nu23=nu23,
        nu31=nu31,
        G12=G12,
        G23=G23,
        G31=G31,
        rho0=rho0,
    )

    p = mat.params

    # Compliance matrix from Fortran
    c11 = 1.0 / E11
    c22 = 1.0 / E22
    c33 = 1.0 / E33
    c12 = -nu12 / E11
    c13 = -nu31 / E33
    c23 = -nu23 / E22

    C_mat = np.array([
        [c11, c12, c13],
        [c12, c22, c23],
        [c13, c23, c33],
    ])

    D_mat = np.array([
        [p["D11"], p["D12"], p["D13"]],
        [p["D12"], p["D22"], p["D23"]],
        [p["D13"], p["D23"], p["D33"]],
    ])

    # Check inverse: C_mat @ D_mat == I
    prod = C_mat @ D_mat
    np.testing.assert_allclose(prod, np.eye(3), atol=1e-12)

    # Check shear moduli
    assert p["G12"] == G12
    assert p["G23"] == G23
    assert p["G31"] == G31


def test_law12_pure_elastic_stress_update():
    """Verify 3D orthotropic stress update: Delta sigma = D : Delta eps (m12law.F:347-353)."""
    E11 = 150000.0
    E22 = 15000.0
    E33 = 15000.0
    nu12 = 0.3
    nu23 = 0.35
    nu31 = 0.03
    G12 = 5000.0
    G23 = 4000.0
    G31 = 5000.0
    rho0 = 1.5e-9

    mat = law12_comp3d.build_law12(
        E11=E11,
        E22=E22,
        E33=E33,
        nu12=nu12,
        nu23=nu23,
        nu31=nu31,
        G12=G12,
        G23=G23,
        G31=G31,
        rho0=rho0,
        sigt1=1e10, sigt2=1e10, sigt3=1e10,  # Ensure no damage
        sigyt1=1e10, sigyt2=1e10, sigyt3=1e10,
    )
    p = mat.params
    d11, d12, d13 = p["D11"], p["D12"], p["D13"]
    d22, d23, d33 = p["D22"], p["D23"], p["D33"]

    # 1. Longitudinal strain along fiber direction 1
    deps1 = np.array([1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig1, _, _ = law12_comp3d.solid_update(mat, np.zeros(6), deps1)
    assert sig1[0] == pytest.approx(d11 * 1.0e-3, rel=1e-6)
    assert sig1[1] == pytest.approx(d12 * 1.0e-3, rel=1e-6)
    assert sig1[2] == pytest.approx(d13 * 1.0e-3, rel=1e-6)
    assert np.allclose(sig1[3:], 0.0, atol=1e-12)

    # 2. Transverse strain in direction 2
    deps2 = np.array([0.0, 1.0e-3, 0.0, 0.0, 0.0, 0.0])
    sig2, _, _ = law12_comp3d.solid_update(mat, np.zeros(6), deps2)
    assert sig2[0] == pytest.approx(d12 * 1.0e-3, rel=1e-6)
    assert sig2[1] == pytest.approx(d22 * 1.0e-3, rel=1e-6)
    assert sig2[2] == pytest.approx(d23 * 1.0e-3, rel=1e-6)

    # 3. Transverse strain in direction 3
    deps3 = np.array([0.0, 0.0, 1.0e-3, 0.0, 0.0, 0.0])
    sig3, _, _ = law12_comp3d.solid_update(mat, np.zeros(6), deps3)
    assert sig3[0] == pytest.approx(d13 * 1.0e-3, rel=1e-6)
    assert sig3[1] == pytest.approx(d23 * 1.0e-3, rel=1e-6)
    assert sig3[2] == pytest.approx(d33 * 1.0e-3, rel=1e-6)

    # 4. Pure shear strains
    deps_shear = np.array([0.0, 0.0, 0.0, 2.0e-3, 1.5e-3, 1.0e-3])
    sig_shear, _, _ = law12_comp3d.solid_update(mat, np.zeros(6), deps_shear)
    assert sig_shear[3] == pytest.approx(G12 * 2.0e-3, rel=1e-6)
    assert sig_shear[4] == pytest.approx(G23 * 1.5e-3, rel=1e-6)
    assert sig_shear[5] == pytest.approx(G31 * 1.0e-3, rel=1e-6)


def test_law12_fiber_composite_contribution():
    """Verify fiber strain and fiber stress tracking (m12law.F lines 399-407)."""
    alpha = 0.55     # 55% fiber volume fraction
    efib = 230000.0  # Carbon fiber modulus

    mat = law12_comp3d.build_law12(
        E11=150000.0,
        E22=10000.0,
        E33=10000.0,
        nu12=0.3,
        nu23=0.35,
        nu31=0.02,
        G12=5000.0,
        G23=3500.0,
        G31=5000.0,
        alpha=alpha,
        efib=efib,
        sigt1=1e10, sigt2=1e10, sigt3=1e10,
    )

    extra = {}
    deps = np.array([2.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig, _, _ = law12_comp3d.solid_update(mat, np.zeros(6), deps, extra=extra)

    # Fiber strain and stress
    assert "epsf12" in extra
    assert "sigf12" in extra
    assert extra["epsf12"][0] == pytest.approx(2.0e-3, rel=1e-6)
    assert extra["sigf12"][0] == pytest.approx(efib * 2.0e-3, rel=1e-6)


def test_law12_tangents_and_sound_speed():
    """Verify elastic solid tangent stiffness tensor and acoustic sound speed."""
    E11 = 160000.0
    E22 = 12000.0
    E33 = 12000.0
    nu12 = 0.28
    nu23 = 0.35
    nu31 = 0.025
    G12 = 6000.0
    G23 = 4000.0
    G31 = 6000.0
    rho0 = 1.55e-9

    mat = law12_comp3d.build_law12(
        E11=E11,
        E22=E22,
        E33=E33,
        nu12=nu12,
        nu23=nu23,
        nu31=nu31,
        G12=G12,
        G23=G23,
        G31=G31,
        rho0=rho0,
    )

    # Solid tangent in elastic regime
    c_sol = law12_comp3d.solid_tangent(mat, np.zeros(6))
    assert c_sol.shape == (6, 6)
    np.testing.assert_allclose(c_sol, c_sol.T, rtol=1e-12)
    assert np.all(np.diag(c_sol) > 0.0)

    # Tangent template compliance
    t_res = law12_comp3d.tangent(mat)
    assert t_res is not None
    assert t_res.shape == (6, 6)

    # Sound speed: c = sqrt(max(D11, D22, D33)/rho0)
    p = mat.params
    c_expected = math.sqrt(max(p["D11"], p["D22"], p["D33"]) / rho0)
    c_actual = law12_comp3d.sound_speed(mat)
    assert c_actual == pytest.approx(c_expected, rel=1e-6)

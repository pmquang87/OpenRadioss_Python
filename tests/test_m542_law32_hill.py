"""
Tests for OpenRadioss Material Law 32 (/MAT/LAW32 / /MAT/HILL - Hill Orthotropic Plasticity).

Covers:
  - Parameter extraction, validation, and upstream defaults (hm_read_mat32.F)
  - Elastic in-plane loading and wave speed
  - Anisotropic yield in 0°, 45°, 90° directions with Lankford parameters
  - Hardening curve and maximum stress cap
  - Strain rate sensitivity and clamping
  - Plastic return algorithms: radial return (IPLA=0), iterative (IPLA=1), plane stress projection (IPLA=2)
  - Plastic failure and element deletion
  - Membrane tangent symmetry and algorithmic consistency
  - Vectorized multi-point execution
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials import law32_hill, shell_update, solid_update, sound_speed, shell_membrane_tangent, consistent_shell_tangent


# ============================================================================
# 1. Parameter Extraction & Upstream Defaults
# ============================================================================

def test_law32_defaults_isotropic():
    """Verify default parameters produce isotropic von Mises Hill parameters."""
    mat = law32_hill.build_law32(
        id=10,
        density=7.85e-9,
        E=210000.0,
        nu=0.3,
        A=350.0,
    )
    p = mat.params
    assert mat.id == 10
    assert mat.law == 32
    assert p["E"] == 210000.0
    assert p["nu"] == 0.3
    assert p["A"] == 350.0
    assert p["B"] == 0.0
    assert p["n"] == 1.0
    assert p["eps_max"] == 1e30
    assert p["sig_max"] == 1e30
    assert p["eps0"] == 1.0
    assert p["m"] == 0.0
    assert p["r00"] == 1.0
    assert p["r45"] == 1.0
    assert p["r90"] == 1.0

    # R = 1.0, H = 0.5 -> A11=1, A22=1, A1122=1, A12=3 (exact von Mises)
    assert p["A11"] == pytest.approx(1.0)
    assert p["A22"] == pytest.approx(1.0)
    assert p["A1122"] == pytest.approx(1.0)
    assert p["A12"] == pytest.approx(3.0)


def test_law32_nu_clamped_at_half():
    """nu >= 0.5 is clamped to 0.499 per hm_read_mat32.F."""
    mat = law32_hill.build_law32(
        id=1,
        E=100000.0,
        nu=0.5,
        A=200.0,
    )
    assert mat.params["nu"] == pytest.approx(0.499)


def test_law32_error_on_n_greater_than_one():
    """Hardening exponent n > 1.0 raises ValueError (upstream error 213)."""
    with pytest.raises(ValueError, match="error 213"):
        law32_hill.build_law32(
            id=1,
            E=100000.0,
            nu=0.3,
            A=200.0,
            n=1.2,
        )


def test_law32_solid_update_not_implemented():
    """solid_update raises NotImplementedError (shells only)."""
    mat = law32_hill.build_law32(id=1, E=210000.0, nu=0.3, A=300.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    with pytest.raises(NotImplementedError, match="shell elements only"):
        law32_hill.solid_update(mat, sig, deps)


def test_law32_sound_speed():
    """Longitudinal thin-shell sound speed c = sqrt(E / (rho0 * (1 - nu^2)))."""
    rho = 7.85e-9
    e = 210000.0
    nu = 0.3
    mat = law32_hill.build_law32(id=1, rho0=rho, E=e, nu=nu, A=300.0)
    expected_c = math.sqrt(e / (rho * (1.0 - nu ** 2)))
    assert law32_hill.sound_speed(mat) == pytest.approx(expected_c)
    assert sound_speed(mat) == pytest.approx(expected_c)


# ============================================================================
# 2. Elastic In-Plane Loading
# ============================================================================

def test_law32_elastic_in_plane_loading():
    """Pure elastic deformation matches generalized Hooke plane stress."""
    e = 210000.0
    nu = 0.3
    g = e / (2.0 * (1.0 + nu))
    a1 = e / (1.0 - nu ** 2)
    a2 = nu * a1
    mat = law32_hill.build_law32(id=1, E=e, nu=nu, A=500.0, B=1.0, n=0.0)

    sig = np.zeros((1, 3))
    deps = np.array([[0.0005, -0.0002, 0.0003]])

    s_out, ep_out = law32_hill.shell_update(mat, sig, deps, dt=1e-4)

    expected_sxx = a1 * deps[0, 0] + a2 * deps[0, 1]
    expected_syy = a2 * deps[0, 0] + a1 * deps[0, 1]
    expected_sxy = g * deps[0, 2]

    assert s_out[0, 0] == pytest.approx(expected_sxx)
    assert s_out[0, 1] == pytest.approx(expected_syy)
    assert s_out[0, 2] == pytest.approx(expected_sxy)
    assert ep_out[0] == pytest.approx(0.0)


def test_law32_elastic_5_component_shear():
    """Transverse shear components sigyz and sigzx are updated with 5/6 G."""
    e = 210000.0
    nu = 0.3
    g = e / (2.0 * (1.0 + nu))
    gs = 5.0 / 6.0 * g
    mat = law32_hill.build_law32(id=1, E=e, nu=nu, A=500.0, B=1.0, n=0.0)

    sig = np.zeros((1, 5))
    deps = np.array([[0.0002, 0.0001, 0.0001, 0.0004, 0.0003]])

    s_out, _ = law32_hill.shell_update(mat, sig, deps, dt=1e-4)

    assert s_out[0, 3] == pytest.approx(gs * deps[0, 3])
    assert s_out[0, 4] == pytest.approx(gs * deps[0, 4])


# ============================================================================
# 3. Anisotropic Yield in 0°, 45°, 90° Directions
# ============================================================================

def test_law32_anisotropic_yield_directional():
    """Lankford parameters R00, R45, R90 govern directional yield stresses."""
    e = 200000.0
    nu = 0.3
    a_yield = 300.0
    r00 = 1.6
    r45 = 1.1
    r90 = 2.4

    mat = law32_hill.build_law32(
        id=1, E=e, nu=nu,
        A=a_yield, B=1.0, n=0.0,
        r00=r00, r45=r45, r90=r90,
        i_yield=0
    )
    p = mat.params
    a11 = p["A11"]
    a22 = p["A22"]
    a1122 = p["A1122"]
    a12 = p["A12"]

    # Theoretical yield stresses under uniaxial tension in 0, 90, 45 deg:
    # Theoretical yield stresses under uniaxial tension in 0, 90, 45 deg:
    sig0_th = a_yield / math.sqrt(a11)
    sig90_th = a_yield / math.sqrt(a22)
    sig45_th = 2.0 * a_yield / math.sqrt(a11 + a22 - a1122 + a12)

    # 1. Uniaxial tension along 0 deg (theta = 0)
    # Under plane-stress, pure uniaxial stress sigma_yy = 0 requires deps_yy = -nu * deps_xx
    deps_0 = np.array([[0.005, -nu * 0.005, 0.0]])
    s0_out, _ = law32_hill.shell_update(mat, np.zeros((1, 3)), deps_0, dt=1e-4, extra={"theta": 0.0})
    assert s0_out[0, 0] == pytest.approx(sig0_th, rel=1e-4)
    assert s0_out[0, 1] == pytest.approx(0.0, abs=1e-6)

    # 2. Uniaxial tension along 90 deg (deps_xx = -nu * deps_yy)
    deps_90 = np.array([[-nu * 0.005, 0.005, 0.0]])
    s90_out, _ = law32_hill.shell_update(mat, np.zeros((1, 3)), deps_90, dt=1e-4, extra={"theta": 0.0})
    assert s90_out[0, 1] == pytest.approx(sig90_th, rel=1e-4)
    assert s90_out[0, 0] == pytest.approx(0.0, abs=1e-6)

    # 3. Uniaxial tension along 45 deg (theta = pi/4, deps_yy = -nu * deps_xx in element axes)
    deps_45 = np.array([[0.005, -nu * 0.005, 0.0]])
    s45_out, _ = law32_hill.shell_update(mat, np.zeros((1, 3)), deps_45, dt=1e-4, extra={"theta": math.pi / 4.0})
    assert s45_out[0, 0] == pytest.approx(sig45_th, rel=1e-4)
    assert s45_out[0, 1] == pytest.approx(0.0, abs=1e-6)


def test_law32_iyield_normalization():
    """When i_yield > 0, A11 is normalized to 1.0, so sig0 == A identically."""
    mat = law32_hill.build_law32(
        id=2, E=200000.0, nu=0.3,
        A=300.0, B=1.0, n=0.0,
        r00=1.5, r45=1.2, r90=2.0,
        i_yield=1
    )
    assert mat.params["A11"] == pytest.approx(1.0)


# ============================================================================
# 4. Hardening Curve & Maximum Stress Cap
# ============================================================================

def test_law32_hardening_and_sig_max_cap():
    """Flow stress evolves with plastic strain and is bounded by sig_max."""
    a = 300.0
    b = 0.02
    n = 0.4
    sig_max = 100.0
    mat = law32_hill.build_law32(
        id=1, E=210000.0, nu=0.3,
        A=a, B=b, n=n, sig_max=sig_max
    )

    sig = np.zeros((1, 3))
    epsp = np.zeros(1)

    # Incremental plastic stretching
    epsp_start = 0.0
    for _ in range(10):
        epsp_start = float(epsp[0])
        deps = np.array([[0.001, 0.0, 0.0]])
        sig, epsp = law32_hill.shell_update(mat, sig, deps, epsp, dt=1e-4)

    # Check that flow stress equals Swift formula at the start-of-step plastic strain
    expected_flow = a * ((b + epsp_start) ** n)
    seq = math.sqrt(sig[0, 0]**2 + sig[0, 1]**2 - sig[0, 0]*sig[0, 1] + 3.0*sig[0, 2]**2)
    assert seq == pytest.approx(expected_flow, rel=1e-3)

    # Apply large strain (epsp will exceed 0.05 where uncapped flow stress > 103.5 > 100.0)
    # Note that in explicit radial return, sigy is evaluated at start-of-step epsp,
    # so we step once to cross the cap, and the next step is capped at sig_max.
    deps_large = np.array([[0.05, 0.0, 0.0]])
    sig, epsp = law32_hill.shell_update(mat, sig, deps_large, epsp, dt=1e-4)
    # Next step: start-of-step epsp > 0.05, so sigy is capped at sig_max = 100.0
    deps_test = np.array([[0.001, 0.0, 0.0]])
    sig, epsp = law32_hill.shell_update(mat, sig, deps_test, epsp, dt=1e-4)
    seq_capped = math.sqrt(sig[0, 0]**2 + sig[0, 1]**2 - sig[0, 0]*sig[0, 1] + 3.0*sig[0, 2]**2)
    assert seq_capped == pytest.approx(sig_max, rel=1e-4)


# ============================================================================
# 5. Strain Rate Sensitivity
# ============================================================================

def test_law32_strain_rate_scaling():
    """Yield stress scales with (eps_dot / eps0)^m."""
    mat = law32_hill.build_law32(
        id=1, E=210000.0, nu=0.3,
        A=300.0, B=1.0, n=0.0,
        eps0=1.0, m=0.1
    )

    sig1 = np.zeros((1, 3))
    deps1 = np.array([[0.01, 0.0, 0.0]])
    dt1 = 1e-3  # rate = 10 s^-1
    sig1, _ = law32_hill.shell_update(mat, sig1, deps1, dt=dt1)
    seq1 = math.sqrt(sig1[0, 0]**2 + sig1[0, 1]**2 - sig1[0, 0]*sig1[0, 1])

    sig2 = np.zeros((1, 3))
    deps2 = np.array([[0.01, 0.0, 0.0]])
    dt2 = 1e-4  # rate = 100 s^-1
    sig2, _ = law32_hill.shell_update(mat, sig2, deps2, dt=dt2)
    seq2 = math.sqrt(sig2[0, 0]**2 + sig2[0, 1]**2 - sig2[0, 0]*sig2[0, 1])

    expected_ratio = (100.0 / 10.0) ** 0.1
    assert (seq2 / seq1) == pytest.approx(expected_ratio, rel=1e-3)


def test_law32_strain_rate_clamped_below_eps0():
    """Strain rate below eps0 is clamped to eps0 (no softening)."""
    mat = law32_hill.build_law32(
        id=1, E=210000.0, nu=0.3,
        A=300.0, B=1.0, n=0.0,
        eps0=10.0, m=0.1
    )

    sig = np.zeros((1, 3))
    deps = np.array([[0.005, 0.0, 0.0]])
    dt = 1.0  # rate = 0.005 << 10.0
    sig, _ = law32_hill.shell_update(mat, sig, deps, dt=dt)
    seq = math.sqrt(sig[0, 0]**2 + sig[0, 1]**2 - sig[0, 0]*sig[0, 1])
    assert seq == pytest.approx(300.0, rel=1e-4)


# ============================================================================
# 6. Plastic Return Algorithms (IPLA=0, IPLA=1, IPLA=2)
# ============================================================================

@pytest.mark.parametrize("ipla", [0, 1, 2])
def test_law32_all_ipla_return_algorithms(ipla):
    """All return mapping options project stresses onto the Hill yield surface."""
    mat = law32_hill.build_law32(
        id=1, E=200000.0, nu=0.3,
        A=300.0, B=1.0, n=0.0,
        r00=1.5, r45=1.2, r90=1.8,
        ipla=ipla
    )
    p = mat.params

    sig = np.zeros((1, 3))
    deps = np.array([[0.004, -0.001, 0.002]])
    sig_new, epsp = law32_hill.shell_update(mat, sig, deps, dt=1e-4)

    # In material coordinates:
    a11, a22, a1122, a12 = p["A11"], p["A22"], p["A1122"], p["A12"]
    seq = math.sqrt(a11 * sig_new[0, 0]**2 + a22 * sig_new[0, 1]**2
                    - a1122 * sig_new[0, 0] * sig_new[0, 1] + a12 * sig_new[0, 2]**2)

    tol = 3e-3 if ipla == 1 else 1e-3
    assert seq == pytest.approx(300.0, rel=tol)
    assert epsp[0] > 0.0


# ============================================================================
# 7. Plastic Failure & Element Deletion
# ============================================================================

def test_law32_plastic_failure_deletion():
    """Element deletion occurs when eps_p >= eps_max, setting off32=0 and zeroing stress."""
    eps_max = 0.005
    mat = law32_hill.build_law32(
        id=1, E=210000.0, nu=0.3,
        A=300.0, B=1.0, n=0.0,
        eps_max=eps_max
    )

    sig = np.zeros((1, 3))
    epsp = np.zeros(1)
    extra = {"off32": np.ones(1), "uv32": np.zeros((1, 1))}

    # Step 1: deformation below failure
    deps1 = np.array([[0.003, 0.0, 0.0]])
    sig, epsp = law32_hill.shell_update(mat, sig, deps1, epsp, dt=1e-4, extra=extra)
    assert extra["off32"][0] == 1.0
    assert np.any(sig[0] != 0.0)

    # Step 2: large deformation causing eps_p >= eps_max
    deps2 = np.array([[0.01, 0.0, 0.0]])
    sig, epsp = law32_hill.shell_update(mat, sig, deps2, epsp, dt=1e-4, extra=extra)
    assert epsp[0] >= eps_max
    assert extra["off32"][0] == 0.0
    assert np.all(sig[0] == 0.0)


# ============================================================================
# 8. Consistent Algorithmic Membrane Tangent
# ============================================================================

def test_law32_membrane_tangent_elastic():
    """Elastic membrane tangent matches shell_membrane_tangent."""
    mat = law32_hill.build_law32(id=1, E=210000.0, nu=0.3, A=400.0)
    c_el = law32_hill.shell_membrane_tangent(mat)
    assert c_el.shape == (3, 3)
    assert c_el[0, 0] == c_el[1, 1]
    assert c_el[0, 1] == c_el[1, 0]
    assert c_el[0, 2] == 0.0


def test_law32_consistent_shell_tangent_symmetry():
    """consistent_shell_tangent with symmetric=True produces an exact symmetric matrix."""
    mat = law32_hill.build_law32(
        id=1, E=210000.0, nu=0.3,
        A=300.0, B=1.0, n=0.0,
        r00=1.4, r45=1.2, r90=1.8
    )
    sig = np.array([[220.0, -80.0, 30.0]])
    epsp = np.array([0.002])
    epsp_incr = np.array([0.0005])

    d_sym = law32_hill.consistent_shell_tangent(mat, sig, epsp, epsp_incr, symmetric=True)
    assert np.allclose(d_sym[0], d_sym[0].T, atol=1e-12)


def test_law32_consistent_shell_tangent_directional_derivative():
    """Exact algorithmic derivative matches finite difference to high precision (< 1e-5)."""
    mat = law32_hill.build_law32(
        id=1, E=210000.0, nu=0.3,
        A=300.0, B=1.0, n=0.0,
        ipla=0
    )
    sig_init = np.array([[190.0, 45.0, 15.0]])
    epsp_init = np.array([0.001])
    deps_base = np.array([[0.003, -0.001, 0.002]])

    sig = sig_init.copy()
    epsp = epsp_init.copy()
    law32_hill.shell_update(mat, sig, deps_base, epsp, dt=0.0)
    dep_incr = epsp - epsp_init

    d_tangent = law32_hill.consistent_shell_tangent(mat, sig, epsp, dep_incr, symmetric=False)[0]

    # Test in multiple perturbation directions
    perturbations = [
        np.array([0.0002, -0.0001, 0.0003]),
        np.array([0.0001, 0.0002, -0.0001]),
        np.array([-0.0003, 0.0001, 0.0002]),
    ]
    h = 1e-7

    for delta_eps in perturbations:
        sig_p = sig_init.copy()
        epsp_p = epsp_init.copy()
        law32_hill.shell_update(mat, sig_p, deps_base + h * delta_eps, epsp_p, dt=0.0)

        sig_m = sig_init.copy()
        epsp_m = epsp_init.copy()
        law32_hill.shell_update(mat, sig_m, deps_base - h * delta_eps, epsp_m, dt=0.0)

        d_num = (sig_p[0] - sig_m[0]) / (2.0 * h)
        d_tan = d_tangent @ delta_eps

        rel_err = np.linalg.norm(d_num - d_tan) / np.linalg.norm(d_tan)
        assert rel_err < 1e-5


# ============================================================================
# 9. Batch Vectorization
# ============================================================================

def test_law32_batch_vectorization():
    """Batch evaluation across multiple integration points equals serial evaluations."""
    mat = law32_hill.build_law32(
        id=1, E=210000.0, nu=0.3,
        A=320.0, B=0.01, n=0.3,
        r00=1.5, r45=1.2, r90=1.8
    )

    n_pts = 4
    sig_batch = np.array([
        [0.0, 0.0, 0.0],
        [100.0, -50.0, 10.0],
        [200.0, 100.0, -20.0],
        [0.0, 0.0, 0.0],
    ])
    deps_batch = np.array([
        [0.001, 0.0, 0.0],
        [0.003, -0.001, 0.002],
        [0.005, 0.002, 0.0],
        [0.02, 0.0, 0.0],
    ])
    epsp_batch = np.array([0.0, 0.001, 0.002, 0.0])

    s_bat, ep_bat = law32_hill.shell_update(
        mat, sig_batch.copy(), deps_batch, epsp_batch.copy(), dt=1e-4
    )

    for i in range(n_pts):
        s_ind, ep_ind = law32_hill.shell_update(
            mat, sig_batch[i:i+1].copy(), deps_batch[i:i+1], epsp_batch[i:i+1].copy(), dt=1e-4
        )
        assert np.allclose(s_bat[i], s_ind[0], atol=1e-10)
        assert ep_bat[i] == pytest.approx(ep_ind[0], rel=1e-10)

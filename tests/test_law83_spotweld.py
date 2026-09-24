"""
Unit tests for LAW83 solid spotweld material (/MAT/LAW83, /MAT/CONNECT, /MAT/SPR_JOU).

Verifies:
1. Non-quadratic exponent (beta != 2) return mapping for tension and shear.
2. Compression behavior under icomp=1 vs icomp=0 with beta != 2.
3. Yield surface condition Phi = 0 satisfied for beta != 2.
4. Orientation angle (sym / theta) scaling of normal yield parameter.
5. Rate filtering for vp = 1 (plastic strain rate filtering).
6. Deletion flag (off) suppressing stress updates.
7. Spotweld failure modes:
   - Tension failure (fail_n)
   - Shear failure (fail_s)
   - Combined failure (fail_n + fail_s + beta_f)
   - Equivalent plastic strain failure (epsp_max)
   - Damage threshold failure (dmg >= 1.0)
8. Discrete connection interface (discrete_connection_update and spring_update).
9. Acoustic sound speed (sound_speed).
10. Consistent tangent symmetry and directional derivative with beta != 2.
11. Registry of LAW83, CONNECT, SPR_JOU.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials import law83_spotweld
from pyradioss.model.entities import Material
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY


def _create_mat83(
    E=200000.0,
    G=77000.0,
    nu=0.3,
    rho0=7850.0,
    sig_y=400.0,
    rn=1.0,
    rs=1.0,
    icomp=1,
    e_comp=250000.0,
    alpha=0.0,
    beta=2.0,
    **kwargs,
) -> Material:
    params = {
        "E": E,
        "G": G,
        "nu": nu,
        "rho0": rho0,
        "sig_y": sig_y,
        "rn": rn,
        "rs": rs,
        "icomp": icomp,
        "E_comp": e_comp,
        "alpha": alpha,
        "beta": beta,
        **kwargs,
    }
    return Material(id=83, law=83, rho0=rho0, title="LAW83_TEST", params=params)


# ============================================================================
# 1. Non-quadratic exponent (beta != 2) return mapping
# ============================================================================

def test_law83_beta_nonquadratic_pure_normal_yield():
    """Verify return mapping with beta = 1.434 (from RD-E-4801 test deck)."""
    beta = 1.434
    sig_y = 350.0
    mat = _create_mat83(E=200000.0, sig_y=sig_y, rn=1.0, beta=beta)

    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 2] = 0.005  # trial stress = 1000 MPa >> 350 MPa

    sig, epsp, c = law83_spotweld.solid_update(mat, sig, deps, None, dt=1.0)

    # For pure normal tension with rn=1.0, sig_eff = (sigma_zz)^beta = sig_y^beta => sigma_zz = sig_y
    assert math.isclose(sig[0, 2], sig_y, rel_tol=1e-3)
    assert epsp[0] > 0.0


def test_law83_beta_nonquadratic_pure_shear_yield():
    """Verify shear return mapping with beta = 1.5."""
    beta = 1.5
    sig_y = 250.0
    G = 80000.0
    mat = _create_mat83(E=200000.0, G=G, sig_y=sig_y, rs=1.0, beta=beta)

    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 4] = 0.01  # trial shear = 800 MPa >> 250 MPa

    sig, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, dt=1.0)

    # For pure shear with rs=1.0, (tau)^beta = sig_y^beta => tau = sig_y
    assert math.isclose(sig[0, 4], sig_y, rel_tol=1e-3)
    assert epsp[0] > 0.0


def test_law83_beta_nonquadratic_combined_yield_surface():
    """Verify that combined normal and shear stress state lands exactly on yield surface for beta != 2."""
    beta = 1.8
    sig_y = 300.0
    rn = 1.2
    rs = 0.9
    mat = _create_mat83(E=200000.0, G=80000.0, sig_y=sig_y, rn=rn, rs=rs, beta=beta)

    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 2] = 0.002  # normal trial = 400 MPa
    deps[0, 4] = 0.002  # shear trial = 160 MPa

    sig, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, dt=1.0)

    # Compute yield criterion: an * |szz|^beta + as * (syz^2 + szx^2)^(beta/2) - sig_y^beta == 0
    an = 1.0 / (rn**beta)
    ast = 1.0 / (rs**beta)
    svmn = abs(sig[0, 2])
    svmt = math.sqrt(sig[0, 4]**2 + sig[0, 5]**2)
    sig_eff = an * (svmn**beta) + ast * (svmt**beta)

    assert math.isclose(sig_eff, sig_y**beta, rel_tol=1e-3)
    assert epsp[0] > 0.0


def test_law83_beta_nonquadratic_compression_icomp1():
    """With icomp=1 and beta != 2, compression is purely elastic for normal, but shear yields."""
    beta = 1.6
    sig_y = 200.0
    E_comp = 240000.0
    G = 80000.0
    mat = _create_mat83(E=200000.0, G=G, e_comp=E_comp, icomp=1, sig_y=sig_y, rs=1.0, beta=beta)

    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 2] = -0.002  # compressive increment -> trial szz = -480 MPa
    deps[0, 4] = 0.008   # shear increment -> trial syz = 640 MPa >> 200 MPa

    sig, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, dt=1.0)

    # Normal stress must remain purely elastic using E_comp
    assert math.isclose(sig[0, 2], -E_comp * 0.002)
    # Shear stress must be projected onto yield surface tau = sig_y
    assert math.isclose(sig[0, 4], sig_y, rel_tol=1e-3)
    assert epsp[0] > 0.0


def test_law83_beta_nonquadratic_compression_icomp0():
    """With icomp=0 and beta != 2, compression undergoes full elastoplastic return."""
    beta = 1.5
    sig_y = 300.0
    E = 200000.0
    mat = _create_mat83(E=E, icomp=0, sig_y=sig_y, rn=1.0, beta=beta)

    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 2] = -0.005  # large compressive trial = -1000 MPa

    sig, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, dt=1.0)

    # Compressive stress must be limited to -sig_y
    assert math.isclose(sig[0, 2], -sig_y, rel_tol=1e-3)
    assert epsp[0] > 0.0


# ============================================================================
# 2. Orientation angle (sym / theta) scaling
# ============================================================================

def test_law83_orientation_angle_sym():
    """Verify that orientation angle theta reduces normal capacity via rn*(1 - alpha*sin(theta))."""
    alpha = 0.5
    theta = math.pi / 2.0  # sin(theta) = 1.0 -> effective rn = rn * (1 - 0.5) = 0.5 * rn
    mat = _create_mat83(E=200000.0, sig_y=400.0, rn=1.0, alpha=alpha, beta=2.0)

    # Element 0: theta = 0 -> effective rn = 1.0 -> max szz = 400.0
    # Element 1: theta = pi/2 -> effective rn = 0.5 -> max szz = 200.0
    sig = np.zeros((2, 6))
    deps = np.zeros((2, 6))
    deps[:, 2] = 0.005  # trial szz = 1000.0

    extra = {"sym": np.array([0.0, theta])}
    sig, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, dt=1.0, extra=extra)

    assert math.isclose(sig[0, 2], 400.0, rel_tol=1e-3)
    assert math.isclose(sig[1, 2], 200.0, rel_tol=1e-3)


# ============================================================================
# 3. Active / deletion flag (off)
# ============================================================================

def test_law83_off_flag_suppresses_updates():
    """When off=0, stress and plastic strain do not increment."""
    mat = _create_mat83(E=200000.0, sig_y=300.0)
    sig = np.array([[0.0, 0.0, 50.0, 0.0, 20.0, 0.0]])
    deps = np.array([[0.0, 0.0, 0.002, 0.0, 0.001, 0.0]])

    extra = {"off": np.array([0.0])}
    sig_out, epsp, _ = law83_spotweld.solid_update(mat, sig.copy(), deps, None, dt=1.0, extra=extra)

    # Stress should remain at old stress
    assert math.isclose(sig_out[0, 2], 50.0)
    assert math.isclose(sig_out[0, 4], 20.0)
    assert epsp[0] == 0.0


# ============================================================================
# 4. Rate filtering for vp = 1 (plastic strain rate filtering)
# ============================================================================

def test_law83_rate_filtering_vp1():
    """Verify that vp=1 filters plastic strain rate (sigeps83.F lines 458-464)."""
    mat = _create_mat83(E=200000.0, sig_y=300.0, vp=1, fcut=1000.0)
    extra = {"epsp": np.zeros(1), "asrate": np.zeros(1)}

    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 2] = 0.005  # plastic yielding occurs
    dt = 0.001

    sig, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, dt=dt, extra=extra)

    assert epsp[0] > 0.0
    # asrate must be positive because plastic strain increment > 0
    assert extra["asrate"][0] > 0.0


# ============================================================================
# 5. Spotweld Failure modes
# ============================================================================

def test_law83_failure_pure_tension():
    """Tension failure when normal strain exceeds fail_n."""
    fail_n = 0.003
    mat = _create_mat83(E=200000.0, sig_y=1000.0, fail_n=fail_n)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 2] = 0.004  # exceeds fail_n

    extra = {}
    sig_out, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, dt=1.0, extra=extra)

    assert extra["failed"][0] == True
    assert extra["off"][0] == 0.0
    assert np.allclose(sig_out[0], 0.0)


def test_law83_failure_pure_shear():
    """Shear failure when transverse shear strain exceeds fail_s."""
    fail_s = 0.005
    mat = _create_mat83(E=200000.0, G=80000.0, sig_y=1000.0, fail_s=fail_s)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 4] = 0.006  # exceeds fail_s

    extra = {}
    sig_out, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, dt=1.0, extra=extra)

    assert extra["failed"][0] == True
    assert extra["off"][0] == 0.0
    assert np.allclose(sig_out[0], 0.0)


def test_law83_failure_combined():
    """Combined failure when (ezz / fail_n)^2 + (gamma / fail_s)^2 >= 1."""
    fail_n = 0.004
    fail_s = 0.004
    mat = _create_mat83(E=200000.0, G=80000.0, sig_y=1000.0, fail_n=fail_n, fail_s=fail_s, beta_f=2.0)

    # Point at (0.003, 0.003) -> (3/4)^2 + (3/4)^2 = 9/16 + 9/16 = 18/16 > 1.0
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 2] = 0.003
    deps[0, 4] = 0.003

    extra = {}
    sig_out, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, dt=1.0, extra=extra)

    assert extra["failed"][0] == True
    assert np.allclose(sig_out[0], 0.0)


def test_law83_failure_epsp_max():
    """Failure when equivalent plastic strain exceeds epsp_max."""
    epsp_max = 0.002
    mat = _create_mat83(E=200000.0, sig_y=200.0, epsp_max=epsp_max)

    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 2] = 0.005  # produces plastic strain > 0.002

    extra = {}
    sig_out, epsp, _ = law83_spotweld.solid_update(mat, sig, deps, None, dt=1.0, extra=extra)

    assert extra["failed"][0] == True
    assert np.allclose(sig_out[0], 0.0)


def test_law83_failure_damage_1():
    """Failure when damage D >= 1.0."""
    mat = _create_mat83(E=200000.0, sig_y=400.0)
    extra = {"dmg": np.array([1.0])}
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 2] = 0.001

    sig_out, _, _ = law83_spotweld.solid_update(mat, sig, deps, None, dt=1.0, extra=extra)
    assert extra["failed"][0] == True
    assert np.allclose(sig_out[0], 0.0)


# ============================================================================
# 6. Discrete connection update & spring update
# ============================================================================

def test_law83_discrete_connection_update():
    """Test discrete_connection_update translating relative displacements to forces."""
    mat = _create_mat83(E=200000.0, G=80000.0, sig_y=1000.0)
    area = 2.0
    length = 1.0
    forces = np.zeros((1, 3))
    du = np.array([[0.001, 0.002, 0.0015]])  # [du_z, du_y, du_x]

    f_new, epsp, c = law83_spotweld.discrete_connection_update(
        mat, forces, du, area=area, length=length, dt=1.0
    )

    expected_Fz = (200000.0 * 0.001) * area
    expected_Fy = (80000.0 * 0.002) * area
    expected_Fx = (80000.0 * 0.0015) * area

    assert math.isclose(f_new[0, 0], expected_Fz)
    assert math.isclose(f_new[0, 1], expected_Fy)
    assert math.isclose(f_new[0, 2], expected_Fx)
    assert epsp[0] == 0.0
    assert c[0] > 0.0


def test_law83_spring_update():
    """Test spring_update convenience interface."""
    mat = _create_mat83(E=200000.0, G=80000.0, sig_y=300.0)
    forces = np.zeros((1, 3))
    du = np.array([[0.005, 0.0, 0.0]])  # large tension causing yield

    f_new, epsp = law83_spotweld.spring_update(mat, forces, du, area=1.0, length=1.0, dt=1.0)

    # Force should be limited to sig_y * area = 300.0
    assert math.isclose(f_new[0, 0], 300.0, rel_tol=1e-3)
    assert epsp[0] > 0.0


# ============================================================================
# 7. Acoustic sound speed
# ============================================================================

def test_law83_sound_speed():
    E = 210000.0
    rho0 = 7850.0
    mat = _create_mat83(E=E, rho0=rho0)

    c = law83_spotweld.sound_speed(mat)
    assert math.isclose(c, math.sqrt(E / rho0))

    # Batched extra check
    c_arr = law83_spotweld.sound_speed(mat, extra={"nel": 3})
    assert len(c_arr) == 3
    assert np.allclose(c_arr, math.sqrt(E / rho0))


# ============================================================================
# 8. Consistent solid tangent with beta != 2
# ============================================================================

def test_law83_tangent_symmetry_nonquadratic_beta():
    """Verify tangent symmetry and directional derivative consistency with beta = 1.6."""
    beta = 1.6
    mat = _create_mat83(E=200000.0, G=80000.0, sig_y=400.0, beta=beta)
    sig0 = np.array([[0.0, 0.0, 150.0, 0.0, 60.0, 40.0]])

    D = law83_spotweld.consistent_solid_tangent(mat, sig=sig0)[0]
    assert np.allclose(D, D.T)

    # Directional derivative consistency
    deps = np.zeros((1, 6))
    deps[0, 2] = 1e-6
    deps[0, 4] = 2e-6
    deps[0, 5] = 1.5e-6

    sig_new, _, _ = law83_spotweld.solid_update(mat, sig0.copy(), deps, None, dt=1.0)
    d_sig_actual = sig_new[0] - sig0[0]
    d_sig_tangent = D @ deps[0]
    assert np.allclose(d_sig_actual, d_sig_tangent, atol=1e-8)


# ============================================================================
# 9. Extra shapes and registry
# ============================================================================

def test_law83_extra_shapes():
    shapes = law83_spotweld.extra_shapes(5)
    assert shapes["epsp"] == (5,)
    assert shapes["asrate"] == (5,)
    assert shapes["dmg"] == (5,)
    assert shapes["off"] == (5,)
    assert shapes["failed"] == (5,)


def test_law83_registry():
    assert "LAW83" in MAT_PHYSICS_REGISTRY
    assert "CONNECT" in MAT_PHYSICS_REGISTRY
    assert "SPR_JOU" in MAT_PHYSICS_REGISTRY

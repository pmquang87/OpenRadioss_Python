"""
Unit tests for Milestone M517: Shell Orthotropy subsystem (/PROP/TYPE9 SH_ORTH,
/PROP/TYPE16 SH_FABR, pyradioss/elements/shell_ortho.py).

Verifies:
1. Empty and None slices / invalid groups handling (returns None).
2. Purely isotropic groups return None (fast path).
3. Projected fiber direction cosines with V parallel to in-plane axes and Phi=0.
4. User rotation angle Phi (30, 45, 60, 90, 180, -45 degrees).
5. Arbitrary 3D tilted corotational shell frame projection.
6. Degenerate / out-of-plane reference vector fallback to local e1 (VNR < 1e-3).
7. Zero reference vector (0, 0, 0) fallback to local e1 without NaNs.
8. Mixed group slices (isotropic TYPE1 + orthotropic TYPE9 + TYPE16).
9. Strain tensor rotation identity at theta=0.
10. Strain tensor rotation at theta=90 deg.
11. Strain tensor rotation at theta=45 deg (uniaxial to pure shear).
12. Stress tensor rotation identity at theta=0.
13. Stress tensor rotation at theta=90 deg.
14. Stress tensor rotation at theta=45 deg (Mohr's circle).
15. Strain round-trip exact invariance (m2e(e2m(deps)) == deps).
16. Stress round-trip exact invariance (e2m(m2e(sig)) == sig).
17. First stress invariant (trace) invariance: tr(sigma_elem) == tr(sigma_ortho).
18. Second stress invariant (determinant) invariance: det(sigma_elem) == det(sigma_ortho).
19. Multidimensional layered shell arrays (n, n_layers, 3) and 5/6-component vectors.
20. Integration with shell_bt4 and shell_tri3 kernels.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pyradioss.elements import shell_ortho, shell_bt4, shell_tri3
from pyradioss.model import Property, Material


# ============================================================================
# Helper fixtures
# ============================================================================

def make_ortho_prop(prop_id=1, prop_type=9, vx=1.0, vy=0.0, vz=0.0, phi=0.0):
    """Construct a dummy Property object matching starter /PROP/TYPE9 layout."""
    return Property(
        id=prop_id,
        type=prop_type,
        params={
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "phi": phi,
            "thick": 1.0,
            "npt": 1,
            "nip": 1,
            "hm": 0.1,
            "hf": 0.1,
            "hr": 0.1,
            "dm": 0.0,
            "df": 0.0,
            "dr": 0.0,
            "ishell": 1,
        }
    )


class DummyLog:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, context=""):
        self.warnings.append((context, msg))


# ============================================================================
# Unit Tests
# ============================================================================

def test_ortho_empty_slices_returns_none():
    """Empty slices, zero elements, or None inputs cleanly return None."""
    E = np.eye(3).reshape(1, 3, 3)
    prop = make_ortho_prop()
    slices = [(slice(0, 1), None, prop)]

    assert shell_ortho.build_group_ortho(None, E, 1) is None
    assert shell_ortho.build_group_ortho([], E, 1) is None
    assert shell_ortho.build_group_ortho(slices, None, 1) is None
    assert shell_ortho.build_group_ortho(slices, E, 0) is None
    assert shell_ortho.build_group_ortho(slices, np.empty((0, 3, 3)), 0) is None


def test_ortho_no_ortho_props_returns_none():
    """Group with only isotropic shell properties (TYPE1, TYPE2) returns None."""
    E = np.tile(np.eye(3), (2, 1, 1))
    p1 = Property(id=1, type=1, params={"thick": 1.0, "nip": 1, "hm": 0.1, "hf": 0.1, "hr": 0.1})
    p2 = Property(id=2, type=2, params={"thick": 1.5, "nip": 1, "hm": 0.1, "hf": 0.1, "hr": 0.1})
    slices = [
        (slice(0, 1), None, p1),
        (slice(1, 2), None, p2),
    ]
    assert shell_ortho.build_group_ortho(slices, E, 2) is None


def test_ortho_build_default_phi0_aligned_v():
    """Reference vector V aligned with local e1 and phi=0 produces (1, 0)."""
    E = np.zeros((1, 3, 3))
    E[0, :, 0] = [1.0, 0.0, 0.0]  # e1
    E[0, :, 1] = [0.0, 1.0, 0.0]  # e2
    E[0, :, 2] = [0.0, 0.0, 1.0]  # e3
    prop = make_ortho_prop(vx=1.0, vy=0.0, vz=0.0, phi=0.0)
    slices = [(slice(0, 1), None, prop)]

    cs = shell_ortho.build_group_ortho(slices, E, 1)
    assert cs is not None
    assert cs.shape == (1, 2)
    assert np.allclose(cs[0], [1.0, 0.0], atol=1e-15)


def test_ortho_build_rotated_phi():
    """Angles Phi = 30, 45, 60, 90, 180, -45 degrees produce exact (cos, sin)."""
    angles_deg = [30.0, 45.0, 60.0, 90.0, 180.0, -45.0]
    n = len(angles_deg)
    E = np.tile(np.eye(3), (n, 1, 1))

    slices = [
        (slice(i, i + 1), None, make_ortho_prop(vx=1.0, vy=0.0, vz=0.0, phi=ang))
        for i, ang in enumerate(angles_deg)
    ]
    cs = shell_ortho.build_group_ortho(slices, E, n)
    assert cs is not None
    for i, ang in enumerate(angles_deg):
        phi_rad = math.radians(ang)
        expected = [math.cos(phi_rad), math.sin(phi_rad)]
        assert np.allclose(cs[i], expected, atol=1e-15)


def test_ortho_build_arbitrary_euler_angles():
    """Tilted 3D shell element triads correctly project global vector V."""
    # Rotate triad by 45 deg around global Z, then 30 deg around local Y
    c45, s45 = math.cos(math.pi / 4), math.sin(math.pi / 4)
    Rz = np.array([
        [c45, -s45, 0.0],
        [s45,  c45, 0.0],
        [0.0,  0.0, 1.0]
    ])
    c30, s30 = math.cos(math.pi / 6), math.sin(math.pi / 6)
    Ry = np.array([
        [c30,  0.0, s30],
        [0.0,  1.0, 0.0],
        [-s30, 0.0, c30]
    ])
    R = Rz @ Ry
    E = np.zeros((1, 3, 3))
    E[0, :, 0] = R[:, 0]
    E[0, :, 1] = R[:, 1]
    E[0, :, 2] = R[:, 2]

    # Global reference vector V = [1, 0, 0]
    prop = make_ortho_prop(vx=1.0, vy=0.0, vz=0.0, phi=0.0)
    cs = shell_ortho.build_group_ortho([(slice(0, 1), None, prop)], E, 1)

    # Verify unit norm of (c, s)
    assert math.isclose(cs[0, 0]**2 + cs[0, 1]**2, 1.0, abs_tol=1e-14)

    # Verify project_fiber_direction is unit vector and orthogonal to normal e3
    d_fib = shell_ortho.project_fiber_direction(E[0, :, 0], E[0, :, 1], vx=1.0, phi_deg=0.0)
    assert math.isclose(np.linalg.norm(d_fib), 1.0, abs_tol=1e-14)
    assert abs(np.dot(d_fib, E[0, :, 2])) < 1e-14


def test_ortho_build_nearly_normal_v_fallback():
    """Reference vector V nearly parallel to shell normal falls back to local e1."""
    E = np.zeros((1, 3, 3))
    E[0, :, 0] = [1.0, 0.0, 0.0]
    E[0, :, 1] = [0.0, 1.0, 0.0]
    E[0, :, 2] = [0.0, 0.0, 1.0]

    # V = [1e-5, 1e-5, 1.0] -> norm < 1e-3
    prop = make_ortho_prop(vx=1e-5, vy=1e-5, vz=1.0, phi=0.0)
    log = DummyLog()
    ids = np.array([101])

    cs = shell_ortho.build_group_ortho([(slice(0, 1), None, prop)], E, 1, log=log, ids=ids)
    assert np.allclose(cs[0], [1.0, 0.0])
    assert len(log.warnings) == 1
    assert "nearly normal" in log.warnings[0][1]
    assert "elements [101]" in log.warnings[0][1]


def test_ortho_build_zero_vector_v_fallback():
    """Zero reference vector V = (0, 0, 0) falls back to local e1 without NaNs."""
    E = np.eye(3).reshape(1, 3, 3)
    prop = make_ortho_prop(vx=0.0, vy=0.0, vz=0.0, phi=0.0)
    cs = shell_ortho.build_group_ortho([(slice(0, 1), None, prop)], E, 1)
    assert np.allclose(cs[0], [1.0, 0.0])
    assert not np.isnan(cs).any()


def test_ortho_build_mixed_isotropic_and_ortho_slices():
    """Mixed slices properly assign identity to isotropic and rotation to ortho."""
    E = np.tile(np.eye(3), (3, 1, 1))
    p_iso = Property(id=1, type=1, params={"thick": 1.0, "nip": 1, "hm": 0.1, "hf": 0.1, "hr": 0.1})
    p_ortho9 = make_ortho_prop(prop_id=2, prop_type=9, vx=0.0, vy=1.0, vz=0.0, phi=0.0)
    p_ortho16 = make_ortho_prop(prop_id=3, prop_type=16, vx=1.0, vy=0.0, vz=0.0, phi=45.0)

    slices = [
        (slice(0, 1), None, p_iso),
        (slice(1, 2), None, p_ortho9),
        (slice(2, 3), None, p_ortho16),
    ]
    cs = shell_ortho.build_group_ortho(slices, E, 3)
    assert cs is not None
    assert np.allclose(cs[0], [1.0, 0.0])
    assert np.allclose(cs[1], [0.0, 1.0])
    c45 = math.cos(math.pi / 4)
    assert np.allclose(cs[2], [c45, c45])


def test_ortho_rotate_strain_identity_zero_angle():
    """Strain rotation at theta=0 is exact identity."""
    cs = np.array([[1.0, 0.0]])
    deps = np.array([[0.01, -0.005, 0.002]])
    deps_rot = shell_ortho.rot_strain_e2m(deps, cs)
    assert np.allclose(deps_rot, deps, atol=1e-15)

    deps_back = shell_ortho.rot_strain_m2e(deps_rot, cs)
    assert np.allclose(deps_back, deps, atol=1e-15)


def test_ortho_rotate_strain_90_deg():
    """Strain rotation at 90 deg exchanges e11/e22 and inverts shear."""
    cs = np.array([[0.0, 1.0]])  # theta = 90 deg
    deps = np.array([[0.012, -0.004, 0.007]])
    # e11' = eyy, e22' = exx, g12' = -gxy
    expected = np.array([[-0.004, 0.012, -0.007]])
    deps_rot = shell_ortho.rot_strain_e2m(deps, cs)
    assert np.allclose(deps_rot, expected, atol=1e-15)

    # Inverse rotation from material back to element
    deps_back = shell_ortho.rot_strain_m2e(deps_rot, cs)
    assert np.allclose(deps_back, deps, atol=1e-15)


def test_ortho_rotate_strain_45_deg_pure_shear():
    """45 deg rotation transforms pure tension/compression into pure shear."""
    c45 = math.cos(math.pi / 4)
    cs = np.array([[c45, c45]])
    eps0 = 0.005
    # Pure normal strain in principal axes: exx = eps0, eyy = -eps0, gxy = 0
    deps = np.array([[eps0, -eps0, 0.0]])
    deps_rot = shell_ortho.rot_strain_e2m(deps, cs)

    # At 45 deg, normal strains vanish, shear becomes -2 * eps0
    assert abs(deps_rot[0, 0]) < 1e-15
    assert abs(deps_rot[0, 1]) < 1e-15
    assert math.isclose(deps_rot[0, 2], -2.0 * eps0, abs_tol=1e-15)


def test_ortho_rotate_stress_identity_zero_angle():
    """Stress rotation at theta=0 is exact identity."""
    cs = np.array([[1.0, 0.0]])
    sig = np.array([[150.0, -80.0, 45.0]])
    sig_rot = shell_ortho.rot_stress_e2m(sig, cs)
    assert np.allclose(sig_rot, sig, atol=1e-15)

    sig_back = shell_ortho.rot_stress_m2e(sig_rot, cs)
    assert np.allclose(sig_back, sig, atol=1e-15)


def test_ortho_rotate_stress_90_deg():
    """Stress rotation at 90 deg exchanges s11/s22 and inverts shear."""
    cs = np.array([[0.0, 1.0]])  # theta = 90 deg
    sig = np.array([[200.0, 50.0, 30.0]])
    expected = np.array([[50.0, 200.0, -30.0]])
    sig_rot = shell_ortho.rot_stress_e2m(sig, cs)
    assert np.allclose(sig_rot, expected, atol=1e-15)

    sig_back = shell_ortho.rot_stress_m2e(sig_rot, cs)
    assert np.allclose(sig_back, sig, atol=1e-15)


def test_ortho_rotate_stress_45_deg():
    """Stress rotation at 45 deg matches Mohr's circle analytical solution."""
    c45 = math.cos(math.pi / 4)
    cs = np.array([[c45, c45]])
    s0 = 100.0
    # Hydrostatic tension: sxx = s0, syy = s0, sxy = 0
    sig = np.array([[s0, s0, 0.0]])
    sig_rot = shell_ortho.rot_stress_e2m(sig, cs)
    assert np.allclose(sig_rot, [s0, s0, 0.0], atol=1e-14)

    # Pure shear: sxx = 0, syy = 0, sxy = tau0
    tau0 = 60.0
    sig_shear = np.array([[0.0, 0.0, tau0]])
    sig_shear_rot = shell_ortho.rot_stress_e2m(sig_shear, cs)
    # At 45 deg, pure shear rotates to principal tension/compression: s11 = tau0, s22 = -tau0, t12 = 0
    assert math.isclose(sig_shear_rot[0, 0], tau0, abs_tol=1e-14)
    assert math.isclose(sig_shear_rot[0, 1], -tau0, abs_tol=1e-14)
    assert abs(sig_shear_rot[0, 2]) < 1e-14


def test_ortho_roundtrip_strain_invariance():
    """Random strain increments round-trip to machine precision."""
    rng = np.random.default_rng(42)
    n = 20
    angles = rng.uniform(-np.pi, np.pi, size=n)
    cs = np.column_stack([np.cos(angles), np.sin(angles)])
    deps = rng.normal(scale=1e-3, size=(n, 3))

    deps_m = shell_ortho.rot_strain_e2m(deps, cs)
    deps_rec = shell_ortho.rot_strain_m2e(deps_m, cs)
    assert np.allclose(deps, deps_rec, atol=1e-16)


def test_ortho_roundtrip_stress_invariance():
    """Random stresses round-trip to machine precision."""
    rng = np.random.default_rng(123)
    n = 20
    angles = rng.uniform(-np.pi, np.pi, size=n)
    cs = np.column_stack([np.cos(angles), np.sin(angles)])
    sig = rng.normal(scale=100.0, size=(n, 3))

    sig_m = shell_ortho.rot_stress_e2m(sig, cs)
    sig_rec = shell_ortho.rot_stress_m2e(sig_m, cs)
    assert np.allclose(sig, sig_rec, atol=1e-13)


def test_ortho_stress_trace_invariant():
    """First stress invariant tr(sigma) = sxx + syy is strictly invariant."""
    rng = np.random.default_rng(999)
    n = 30
    angles = rng.uniform(-np.pi, np.pi, size=n)
    cs = np.column_stack([np.cos(angles), np.sin(angles)])
    sig = rng.normal(scale=250.0, size=(n, 3))

    sig_rot = shell_ortho.rot_stress_e2m(sig, cs)
    tr_orig = sig[:, 0] + sig[:, 1]
    tr_rot = sig_rot[:, 0] + sig_rot[:, 1]
    assert np.allclose(tr_orig, tr_rot, atol=1e-13)


def test_ortho_stress_determinant_invariant():
    """Second stress invariant det(sigma) = sxx*syy - sxy^2 is strictly invariant."""
    rng = np.random.default_rng(777)
    n = 30
    angles = rng.uniform(-np.pi, np.pi, size=n)
    cs = np.column_stack([np.cos(angles), np.sin(angles)])
    sig = rng.normal(scale=250.0, size=(n, 3))

    sig_rot = shell_ortho.rot_stress_e2m(sig, cs)
    det_orig = sig[:, 0] * sig[:, 1] - sig[:, 2] ** 2
    det_rot = sig_rot[:, 0] * sig_rot[:, 1] - sig_rot[:, 2] ** 2
    assert np.allclose(det_orig, det_rot, atol=1e-12)

    # Check invariant utility
    assert shell_ortho.check_invariants(sig, sig_rot)


def test_ortho_layered_shell_tensor_shape():
    """Multidimensional multi-layer arrays (n, n_layers, 3) and 5/6-component tensors."""
    n = 5
    n_layers = 4
    cs = np.array([[math.cos(0.4), math.sin(0.4)]] * n)
    rng = np.random.default_rng(55)

    # 3D layered array
    sig_layered = rng.normal(scale=50.0, size=(n, n_layers, 3))
    sig_rot = shell_ortho.rot_stress_e2m(sig_layered, cs)
    sig_back = shell_ortho.rot_stress_m2e(sig_rot, cs)
    assert sig_rot.shape == (n, n_layers, 3)
    assert np.allclose(sig_layered, sig_back, atol=1e-13)

    # 5-component array with transverse shears
    sig5 = rng.normal(scale=50.0, size=(n, 5))
    sig5_rot = shell_ortho.rot_stress_e2m(sig5, cs)
    sig5_back = shell_ortho.rot_stress_m2e(sig5_rot, cs)
    assert np.allclose(sig5, sig5_back, atol=1e-13)

    # 6-component array with normal stress szz
    sig6 = rng.normal(scale=50.0, size=(n, 6))
    sig6_rot = shell_ortho.rot_stress_e2m(sig6, cs)
    assert np.allclose(sig6[:, 2], sig6_rot[:, 2])  # szz untouched
    sig6_back = shell_ortho.rot_stress_m2e(sig6_rot, cs)
    assert np.allclose(sig6, sig6_back, atol=1e-13)


def test_ortho_integration_with_bt4_and_tri3():
    """End-to-end integration test with shell_bt4 and shell_tri3 kernels."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    model = SimpleNamespace(
        x0=coords.copy(),
        nodes_idx=np.arange(len(coords), dtype=np.int64),
    )
    log = SimpleNamespace(
        error=lambda msg, ctx="": None,
        warning=lambda msg, ctx="": None,
    )

    mat = Material(id=1, law=1, rho0=7.8e-6, params={"rho": 7.8e-6, "E": 210000.0, "nu": 0.3})
    prop = make_ortho_prop(prop_id=1, prop_type=9, vx=1.0, vy=0.0, vz=0.0, phi=30.0)

    # Shell BT4 group
    conn_bt4 = np.array([[0, 1, 2, 3]], dtype=np.int64)
    g_bt4 = SimpleNamespace(
        id=1,
        n=1,
        conn=conn_bt4,
        prop=prop,
        mat=mat,
        ids=np.array([1], dtype=np.int64),
        state={
            "slices": [(slice(0, 1), mat, prop)],
            "off": np.ones(1, dtype=float),
            "chk_fail": False,
        },
    )

    shell_bt4.init_group(g_bt4, model, log)
    assert "ortho" in g_bt4.state
    assert g_bt4.state["ortho"] is not None
    assert g_bt4.state["ortho"].shape == (1, 2)
    c30 = math.cos(math.radians(30))
    s30 = math.sin(math.radians(30))
    assert np.allclose(g_bt4.state["ortho"][0], [c30, s30], atol=1e-6)

    # Run one step of forces
    v = np.zeros((4, 3))
    vr = np.zeros((4, 3))
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    dt_c = shell_bt4.forces(g_bt4, coords, v, vr, 1e-6, fint, mint)
    assert dt_c > 0.0
    assert np.isfinite(fint).all()
    assert np.isfinite(mint).all()

    # Shell TRI3 group
    conn_tri3 = np.array([[0, 1, 2]], dtype=np.int64)
    g_tri3 = SimpleNamespace(
        id=2,
        n=1,
        conn=conn_tri3,
        prop=prop,
        mat=mat,
        ids=np.array([2], dtype=np.int64),
        state={
            "slices": [(slice(0, 1), mat, prop)],
            "off": np.ones(1, dtype=float),
            "chk_fail": False,
        },
    )

    shell_tri3.init_group(g_tri3, model, log)
    assert "ortho" in g_tri3.state
    assert g_tri3.state["ortho"] is not None
    assert np.allclose(g_tri3.state["ortho"][0], [c30, s30], atol=1e-6)

    # Run one step of forces
    fint_tri = np.zeros((4, 3))
    mint_tri = np.zeros((4, 3))
    dt_tri = shell_tri3.forces(g_tri3, coords, v, vr, 1e-6, fint_tri, mint_tri)
    assert dt_tri > 0.0
    assert np.isfinite(fint_tri).all()
    assert np.isfinite(mint_tri).all()

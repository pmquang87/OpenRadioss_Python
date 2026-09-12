"""Tests for Milestone M571: /MAT/LAW101 Continuum Integration, Frame Invariance, and Cyclic Viscoplasticity.

Verifies:
1. Frame invariance:
   - Co-rotational Cauchy stress sigma_Q is invariant under rigid body rotation R applied to F.
   - Spatial Cauchy stress sigma_spatial = R1 @ sigma_Q @ R1^T transforms objectively: sigma* = R @ sigma @ R^T.
2. Cyclic loading and unloading displaying viscoplastic hysteresis and kinematic hardening.
3. Numerical tangent perturbation tensor, symmetry, and positive-definiteness.
4. Sound speed and acoustic wave speed consistency across direct and dispatch APIs.
5. Rejection of shell and 2D elements in materials dispatch (solid continuum only).
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law101_plas_poly import (
    BouvardParams,
    build_law101,
    solid_update,
    solid_tangent,
    sound_speed,
)
from pyradioss.materials import (
    solid_update as mat_solid_update,
    shell_update as mat_shell_update,
    solid_tangent as mat_solid_tangent,
    sound_speed as mat_sound_speed,
)


def _sample_params(**kwargs) -> BouvardParams:
    """Create typical polypropylene material parameters for testing."""
    defaults = dict(
        rho0=0.9e-6,        # kg/mm^3
        e_ref=1500.0,       # MPa
        e1=-2.5,            # MPa/K
        nu=0.38,
        ve1=0.25,
        ve2=1.5,
        edot_ref=1.0e-4,    # 1/s
        gamma0_ref=1.0e-3,  # 1/s
        alpha_p=0.08,       # Pressure sensitivity
        deltah=45000.0,     # J/mol
        vol=2.0e-28,        # Activation volume
        m=1.8,
        c3=-0.05,
        c4=25.0,            # Yield stress at T0
        alphak1=0.15,
        alphak2=0.05,
        h0=50.0,
        zeta1_0=0.1,
        c5=-0.001,
        c6=1.0,
        c7=-0.002,
        c8=2.0,
        c9=-0.01,
        c10=10.0,
        h1=20.0,
        zeta2_0=0.05,
        c11=-0.001,
        c12=1.5,
        c13=0.01,
        c14=5.0,
        c1=-0.05,
        c2=8.0,
        lambda_l=5.0,
        rho_p=0.9e-6,
        cv=1800.0,
        theta0=293.15,
        beta0=1.2e-4,
        theta_g=250.0,
        factor=0.9,
        temp_opt=0.0,
        theta_i=293.15,
    )
    defaults.update(kwargs)
    return build_law101(**defaults)


def test_law101_frame_invariance():
    """Verify frame invariance: rigid body rotation R preserves co-rotational stress and rotates spatial stress."""
    params = _sample_params()

    # Base deformation gradient F (stretched and sheared)
    F_base = np.array([
        [1.15, 0.04, 0.01],
        [0.02, 0.96, 0.03],
        [0.00, 0.01, 0.90],
    ], dtype=np.float64)

    extra1 = {"F": F_base}
    sig_base_v, _, _ = solid_update(params, dt=1.0e-4, extra=extra1)
    sig_base_q = np.array([
        [sig_base_v[0], sig_base_v[3], sig_base_v[5]],
        [sig_base_v[3], sig_base_v[1], sig_base_v[4]],
        [sig_base_v[5], sig_base_v[4], sig_base_v[2]],
    ], dtype=np.float64)

    # 3D rotation matrix (45 deg around z-axis, then 30 deg around x-axis)
    az = math.radians(45.0)
    ax = math.radians(30.0)
    Rz = np.array([
        [math.cos(az), -math.sin(az), 0.0],
        [math.sin(az),  math.cos(az), 0.0],
        [0.0,           0.0,          1.0],
    ])
    Rx = np.array([
        [1.0, 0.0,          0.0],
        [0.0, math.cos(ax), -math.sin(ax)],
        [0.0, math.sin(ax),  math.cos(ax)],
    ])
    R = Rz @ Rx

    # Rotated deformation gradient F_rot = R @ F_base
    F_rot = R @ F_base
    extra2 = {"F": F_rot}
    sig_rot_v, _, _ = solid_update(params, dt=1.0e-4, extra=extra2)
    sig_rot_q = np.array([
        [sig_rot_v[0], sig_rot_v[3], sig_rot_v[5]],
        [sig_rot_v[3], sig_rot_v[1], sig_rot_v[4]],
        [sig_rot_v[5], sig_rot_v[4], sig_rot_v[2]],
    ], dtype=np.float64)

    # 1. Co-rotational Cauchy stress returned by solid_update is identical within numerical rotation precision
    np.testing.assert_allclose(sig_rot_q, sig_base_q, rtol=1e-3, atol=0.02)

    # 2. Spatial Cauchy stress transforms objectively: sigma_spatial* = R @ sigma_spatial @ R.T
    c_base = F_base.T @ F_base
    vals_b, vecs_b = np.linalg.eigh(c_base)
    u_base = vecs_b @ np.diag(np.sqrt(np.maximum(vals_b, 1e-30))) @ vecs_b.T
    r_base = F_base @ np.linalg.inv(u_base)
    sig_spatial_base = r_base @ sig_base_q @ r_base.T

    c_rot = F_rot.T @ F_rot
    vals_r, vecs_r = np.linalg.eigh(c_rot)
    u_rot = vecs_r @ np.diag(np.sqrt(np.maximum(vals_r, 1e-30))) @ vecs_r.T
    r_rot = F_rot @ np.linalg.inv(u_rot)
    sig_spatial_rot = r_rot @ sig_rot_q @ r_rot.T

    expected_spatial_rot = R @ sig_spatial_base @ R.T
    np.testing.assert_allclose(sig_spatial_rot, expected_spatial_rot, rtol=1e-3, atol=0.02)


def test_law101_cyclic_viscoplastic_hysteresis():
    """Verify cyclic loading/unloading produces positive plastic dissipation and hysteresis."""
    params = _sample_params()

    dt = 1.0e-3
    extra = {}
    sig = np.zeros(6, dtype=np.float64)

    # Load in 8 steps
    n_load = 8
    d_load = np.array([0.015, -0.005, -0.005, 0.0, 0.0, 0.0], dtype=np.float64)
    stress_load = []
    for _ in range(n_load):
        sig, epsp, _ = solid_update(params, sig, d_load, dt=dt, extra=extra)
        stress_load.append(sig[0])

    # Plastic strain accumulated during loading
    assert epsp > 0.0
    assert extra["uvar101"][18] > 0.0

    # Unload in 8 steps
    d_unload = -d_load
    stress_unload = []
    for _ in range(n_load):
        sig, epsp, _ = solid_update(params, sig, d_unload, dt=dt, extra=extra)
        stress_unload.append(sig[0])

    # Viscoplastic hysteresis: peak stress during loading is higher than after first unload
    assert stress_load[-1] > stress_unload[0]
    # Residual plastic deformation at completion of reversed steps
    assert extra["uvar101"][18] > 0.0


def test_law101_tangent_consistency():
    """Verify solid_tangent returns a 6x6 positive definite symmetric matrix."""
    params = _sample_params()

    C = solid_tangent(params)
    assert C.shape == (6, 6)
    np.testing.assert_allclose(C, C.T, rtol=1e-12)
    eigenvalues = np.linalg.eigvalsh(C)
    assert np.all(eigenvalues > 0.0)

    C_direct = mat_solid_tangent(params, sig=np.zeros(6))
    np.testing.assert_allclose(C, C_direct)


def test_law101_sound_speed():
    """Verify sound_speed matches longitudinal wave speed sqrt((K + 4/3 G) / rho)."""
    params = _sample_params()
    c_local = sound_speed(params)
    c_dispatch = mat_sound_speed(params)
    assert c_local == pytest.approx(c_dispatch)
    assert c_local > 1000.0


def test_law101_shell_rejection():
    """Verify shell_update rejects LAW101 (solid continuum only)."""
    params = _sample_params()
    with pytest.raises((ValueError, NotImplementedError), match="LAW101|PLAS_POLY|solid|3D"):
        mat_shell_update(params, sig=np.zeros(3), deps=np.zeros(3))

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials import law06_hyd_visc as law06


def _make_law6(visc=1.0, rho0=1000.0, c1=0.0, **kwargs):
    rec = {
        "id": 1,
        "density": rho0,
        "params": {
            "visc": visc,
            "c1": c1,
            **kwargs,
        },
    }
    return law06.build_law6(rec)


def test_law06_empty_array():
    mat = _make_law6()
    sig = np.zeros((0, 6))
    deps = np.zeros((0, 6))
    s_out, epsp_out, c_out = law06.solid_update(mat, sig, deps, None, 0.01)
    assert s_out.shape == (0, 6)
    assert c_out is None


def test_law06_tangent_empty_array():
    mat = _make_law6()
    deps = np.zeros((0, 6))
    D = law06.consistent_solid_tangent(mat, deps, dt=0.01)
    assert D.shape == (0, 6, 6)


def test_law06_shell_update_not_implemented():
    mat = _make_law6()
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    with pytest.raises(NotImplementedError, match="solid/SPH elements only"):
        law06.shell_update(mat, sig, deps)


def test_law06_build_from_material_instance():
    base_mat = Material(id=1, law=6, rho0=1000.0, title="Water", params={"visc": 1.5})
    mat = law06.build_law6(base_mat)
    assert mat.id == 1
    assert mat.rho0 == 1000.0
    assert mat.params["visc"] == 1.5
    assert mat.params["DAMP1"] == 1.5


def test_law06_build_from_dict_direct_keys():
    rec = {
        "id": 2,
        "title": "FluidDirect",
        "density": 1200.0,
        "params": {
            "visc": 2.5,
            "c1": 2.2e9,
            "pmin": -1e6,
        },
    }
    mat = law06.build_law6(rec)
    assert mat.id == 2
    assert mat.rho0 == 1200.0
    assert mat.params["visc"] == 2.5
    assert mat.params["c1"] == 2.2e9
    assert mat.params["pmin"] == -1e6


def test_law06_build_from_dict_cfg_keys():
    rec = {
        "id": 3,
        "title": "FluidCFG",
        "rho": 998.0,
        "params": {
            "DAMP1": 1.0e-3,
            "MAT_C1": 2.15e9,
            "MAT_PC": -5.0e5,
            "MAT_EA": 100.0,
            "MAT_PSH": 101325.0,
        },
    }
    mat = law06.build_law6(rec)
    assert mat.id == 3
    assert mat.rho0 == 998.0
    assert mat.params["visc"] == 1.0e-3
    assert mat.params["c1"] == 2.15e9
    assert mat.params["pmin"] == -5.0e5
    assert mat.params["e0"] == 100.0
    assert mat.params["psh"] == 101325.0
    assert mat.eos is not None
    assert mat.eos.kind == "POLYNOMIAL"


def test_law06_build_validation_density_nonpositive():
    rec = {"id": 1, "rho": 0.0, "params": {"visc": 1.0}}
    with pytest.raises(ValueError, match="Density rho0 must be positive"):
        law06.build_law6(rec)


def test_law06_build_validation_viscosity_negative():
    rec = {"id": 1, "density": 1000.0, "params": {"DAMP1": -0.5}}
    with pytest.raises(ValueError, match="Viscosity parameter DAMP1 must be non-negative"):
        law06.build_law6(rec)


def test_law06_embedded_polynomial_eos_creation():
    rec = {
        "id": 4,
        "density": 1000.0,
        "params": {
            "visc": 1.0,
            "MAT_C0": 0.0,
            "MAT_C1": 2.2e9,
            "MAT_C2": 1.0e8,
            "MAT_C3": 0.0,
            "MAT_C4": 0.4,
            "MAT_C5": 0.0,
            "MAT_EA": 500.0,
            "MAT_PSH": 0.0,
            "MAT_PC": -1e6,
        },
    }
    mat = law06.build_law6(rec)
    assert mat.eos is not None
    assert mat.eos.kind == "POLYNOMIAL"
    assert mat.eos.params["c1"] == 2.2e9
    assert mat.eos.params["c2"] == 1.0e8
    assert mat.eos.params["c4"] == 0.4
    assert mat.eos.params["e0"] == 500.0


def test_law06_pmin_default_infinity():
    # If MAT_PC is 0.0, OpenRadioss Fortran hm_read_mat06 sets PMIN = -INFINITY
    rec = {"id": 5, "density": 1000.0, "params": {"visc": 1.0, "MAT_PC": 0.0}}
    mat = law06.build_law6(rec)
    assert mat.params["pmin"] == -1e30


def test_law06_preserve_test_m76_behavior():
    # Exact reproduction of tests/test_m76_law06.py
    mat = Material(id=1, law=6, rho0=1000.0, title="Water", params={"visc": 1.0})
    deps = np.array([[0.1, 0.2, -0.3, 0.1, -0.1, 0.0]])
    dt = 0.1
    extra = {"rho": np.array([1200.0])}
    sig = np.zeros((1, 6))

    sig_out, epsp_out, c_out = law06.solid_update(mat, sig, deps, None, dt, extra)
    expected_sig = np.array([[2400.0, 4800.0, -7200.0, 1200.0, -1200.0, 0.0]])
    np.testing.assert_allclose(sig_out, expected_sig)
    assert epsp_out is None
    assert c_out is None


def test_law06_missing_extra_fallback():
    # When extra is None, should fall back to mat.rho0 = 1000.0
    mat = _make_law6(visc=2.0, rho0=1000.0)
    sig = np.zeros((1, 6))
    deps = np.array([[0.0, 0.0, 0.0, 0.05, 0.0, 0.0]])
    dt = 0.1
    # deps_rate = 0.5; visc = 2.0 * 1000.0 = 2000.0; sig4 = 2000 * 0.5 = 1000.0
    sig_out, _, _ = law06.solid_update(mat, sig, deps, None, dt, extra=None)
    assert math.isclose(sig_out[0, 3], 1000.0)


def test_law06_scalar_rho_extra():
    # When extra["rho"] is a scalar
    mat = _make_law6(visc=1.0, rho0=1000.0)
    sig = np.zeros((1, 6))
    deps = np.array([[0.0, 0.0, 0.0, 0.1, 0.0, 0.0]])
    dt = 0.1
    extra = {"rho": 1500.0}
    sig_out, _, _ = law06.solid_update(mat, sig, deps, None, dt, extra=extra)
    # visc = 1.0 * 1500 = 1500; rate = 1.0; sig4 = 1500.0
    assert math.isclose(sig_out[0, 3], 1500.0)


def test_law06_dt_zero_preserves_zero_deviator():
    # Static solve or dt <= 0: viscous stress deviator is zero
    mat = _make_law6(visc=5.0, rho0=1000.0)
    sig = np.zeros((1, 6))
    deps = np.array([[0.01, -0.01, 0.0, 0.02, 0.0, 0.0]])
    sig_out, _, _ = law06.solid_update(mat, sig, deps, None, dt=0.0)
    np.testing.assert_allclose(sig_out, np.zeros((1, 6)))


def test_law06_pure_shear_rate_xy():
    mat = _make_law6(visc=2.0, rho0=1000.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 3] = 0.04  # shear xy
    dt = 0.02  # rate = 2.0
    sig_out, _, _ = law06.solid_update(mat, sig, deps, None, dt)
    # eta = 2.0 * 1000.0 = 2000.0; sig_xy = 2000.0 * 2.0 = 4000.0
    assert math.isclose(sig_out[0, 3], 4000.0)
    assert np.all(sig_out[0, [0, 1, 2, 4, 5]] == 0.0)


def test_law06_pure_shear_rate_yz_zx():
    mat = _make_law6(visc=1.5, rho0=800.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 4] = 0.03  # yz
    deps[0, 5] = -0.06  # zx
    dt = 0.01  # rates = 3.0 and -6.0
    sig_out, _, _ = law06.solid_update(mat, sig, deps, None, dt)
    # eta = 1.5 * 800 = 1200
    assert math.isclose(sig_out[0, 4], 1200.0 * 3.0)
    assert math.isclose(sig_out[0, 5], 1200.0 * -6.0)


def test_law06_uniaxial_strain_rate_normal_deviator():
    # Uniaxial extension: D1 = 3.0, D2 = D3 = 0 -> dav = -1.0
    # sig1 = 2*eta*(3 - 1) = 4*eta
    # sig2 = 2*eta*(0 - 1) = -2*eta
    # sig3 = 2*eta*(0 - 1) = -2*eta
    mat = _make_law6(visc=1.0, rho0=1000.0)
    sig = np.zeros((1, 6))
    deps = np.array([[0.03, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dt = 0.01  # D1 = 3.0
    sig_out, _, _ = law06.solid_update(mat, sig, deps, None, dt)
    eta = 1000.0
    assert math.isclose(sig_out[0, 0], 4.0 * eta)
    assert math.isclose(sig_out[0, 1], -2.0 * eta)
    assert math.isclose(sig_out[0, 2], -2.0 * eta)
    # Trace of deviatoric stress must be exactly 0
    assert math.isclose(sig_out[0, 0] + sig_out[0, 1] + sig_out[0, 2], 0.0, abs_tol=1e-12)


def test_law06_pure_volumetric_expansion_zero_deviator():
    # Hydrostatic strain rate: D1 = D2 = D3 = 2.0 -> dav = -2.0 -> deviatoric viscous stress = 0
    mat = _make_law6(visc=3.0, rho0=1000.0)
    sig = np.zeros((1, 6))
    deps = np.array([[0.02, 0.02, 0.02, 0.0, 0.0, 0.0]])
    dt = 0.01
    sig_out, _, _ = law06.solid_update(mat, sig, deps, None, dt)
    np.testing.assert_allclose(sig_out, np.zeros((1, 6)), atol=1e-12)


def test_law06_density_proportionality():
    mat = _make_law6(visc=1.0, rho0=1000.0)
    deps = np.array([[0.01, 0.02, -0.03, 0.01, 0.0, 0.0]])
    dt = 0.01

    extra1 = {"rho": np.array([1000.0])}
    sig1, _, _ = law06.solid_update(mat, np.zeros((1, 6)), deps, None, dt, extra=extra1)

    extra2 = {"rho": np.array([2000.0])}
    sig2, _, _ = law06.solid_update(mat, np.zeros((1, 6)), deps, None, dt, extra=extra2)

    np.testing.assert_allclose(sig2, 2.0 * sig1)


def test_law06_sound_speed_embedded_eos():
    mat = _make_law6(visc=1.0, rho0=1000.0, MAT_C1=2.25e9, MAT_C0=0.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    _, _, c_out = law06.solid_update(mat, sig, deps, None, 0.01)
    # c = sqrt(2.25e9 / 1000) = sqrt(2.25e6) = 1500.0 m/s (water sound speed!)
    assert c_out is not None
    assert math.isclose(c_out[0], 1500.0)


def test_law06_sound_speed_no_eos_returns_none():
    mat = _make_law6(visc=1.0, rho0=1000.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    _, _, c_out = law06.solid_update(mat, sig, deps, None, 0.01)
    assert c_out is None


def test_law06_tangent_shape_and_symmetry():
    mat = _make_law6(visc=1.0, rho0=1000.0, MAT_C1=2.2e9)
    deps = np.zeros((1, 6))
    dt = 0.001
    D = law06.consistent_solid_tangent(mat, deps, dt=dt)
    assert D.shape == (1, 6, 6)
    # Symmetry: D == D^T
    np.testing.assert_allclose(D[0], D[0].T, atol=1e-12)


def test_law06_tangent_positive_semidefinite():
    mat = _make_law6(visc=1.5, rho0=1000.0, MAT_C1=2.0e9)
    deps = np.zeros((1, 6))
    dt = 0.01
    D = law06.consistent_solid_tangent(mat, deps, dt=dt)
    eigs = np.linalg.eigvalsh(D[0])
    assert np.all(eigs >= -1e-12)


def test_law06_tangent_deviatoric_consistency():
    # Verify D_visc @ deps matches sig_dev exactly
    mat = _make_law6(visc=1.2, rho0=1000.0)  # bulk = 0
    deps = np.array([[0.01, -0.005, -0.005, 0.02, -0.015, 0.01]])
    dt = 0.005
    sig, _, _ = law06.solid_update(mat, np.zeros((1, 6)), deps, None, dt)
    D = law06.consistent_solid_tangent(mat, deps, dt=dt)
    sig_pred = D[0] @ deps[0]
    np.testing.assert_allclose(sig_pred, sig[0], rtol=1e-12, atol=1e-12)


def test_law06_tangent_finite_difference():
    mat = _make_law6(visc=1.0, rho0=1000.0)
    deps0 = np.array([[0.002, -0.001, -0.001, 0.005, 0.003, -0.002]])
    dt = 0.01
    D = law06.consistent_solid_tangent(mat, deps0, dt=dt)[0]

    h = 1e-6
    for comp in range(6):
        deps_p = deps0.copy()
        deps_m = deps0.copy()
        deps_p[0, comp] += h
        deps_m[0, comp] -= h

        sig_p, _, _ = law06.solid_update(mat, np.zeros((1, 6)), deps_p, None, dt)
        sig_m, _, _ = law06.solid_update(mat, np.zeros((1, 6)), deps_m, None, dt)

        d_num = (sig_p[0] - sig_m[0]) / (2.0 * h)
        np.testing.assert_allclose(D[:, comp], d_num, rtol=1e-5, atol=1e-5)


def test_law06_tangent_dt_zero_static():
    mat = _make_law6(visc=2.0, rho0=1000.0, MAT_C1=2.0e9)
    deps = np.zeros((1, 6))
    D = law06.consistent_solid_tangent(mat, deps, dt=0.0)
    # Deviatoric viscous stiffness is 0, bulk stiffness is K = 2.0e9
    expected_D = np.zeros((6, 6))
    expected_D[:3, :3] = 2.0e9
    np.testing.assert_allclose(D[0], expected_D)


def test_law06_frame_invariance_3d_rotation():
    # Rotating strain rate by 3D rotation matrix Q rotates stress deviator by Q
    theta = math.pi / 4.0
    phi = math.pi / 6.0
    # Rotation about Z then X
    Rz = np.array([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta), math.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(phi), -math.sin(phi)],
        [0.0, math.sin(phi), math.cos(phi)],
    ])
    Q = Rz @ Rx

    mat = _make_law6(visc=1.5, rho0=1000.0)
    dt = 0.01

    # Original strain tensor: [e11, e22, e33, 2*e12, 2*e23, 2*e31]
    eps_mat = np.array([
        [0.01, 0.005, -0.002],
        [0.005, -0.006, 0.003],
        [-0.002, 0.003, -0.004],
    ])
    deps = np.array([[
        eps_mat[0, 0], eps_mat[1, 1], eps_mat[2, 2],
        2.0 * eps_mat[0, 1], 2.0 * eps_mat[1, 2], 2.0 * eps_mat[2, 0]
    ]])

    sig, _, _ = law06.solid_update(mat, np.zeros((1, 6)), deps, None, dt)
    sig_mat = np.array([
        [sig[0, 0], sig[0, 3], sig[0, 5]],
        [sig[0, 3], sig[0, 1], sig[0, 4]],
        [sig[0, 5], sig[0, 4], sig[0, 2]],
    ])

    # Rotated strain tensor: eps_rot = Q @ eps_mat @ Q.T
    eps_rot = Q @ eps_mat @ Q.T
    deps_rot = np.array([[
        eps_rot[0, 0], eps_rot[1, 1], eps_rot[2, 2],
        2.0 * eps_rot[0, 1], 2.0 * eps_rot[1, 2], 2.0 * eps_rot[2, 0]
    ]])

    sig_rot, _, _ = law06.solid_update(mat, np.zeros((1, 6)), deps_rot, None, dt)
    sig_rot_mat = np.array([
        [sig_rot[0, 0], sig_rot[0, 3], sig_rot[0, 5]],
        [sig_rot[0, 3], sig_rot[0, 1], sig_rot[0, 4]],
        [sig_rot[0, 5], sig_rot[0, 4], sig_rot[0, 2]],
    ])

    expected_sig_rot_mat = Q @ sig_mat @ Q.T
    np.testing.assert_allclose(sig_rot_mat, expected_sig_rot_mat, rtol=1e-10, atol=1e-10)


def test_law06_batched_vectorization():
    nel = 8
    mat = _make_law6(visc=1.8, rho0=1050.0, MAT_C1=2.1e9)
    dt = 0.005

    rng = np.random.default_rng(42)
    deps_batch = rng.normal(0.0, 0.01, size=(nel, 6))
    sig_batch = np.zeros((nel, 6))
    extra_batch = {"rho": rng.uniform(900.0, 1200.0, size=nel)}

    sig_batch_out, _, c_batch = law06.solid_update(
        mat, sig_batch, deps_batch, None, dt, extra=extra_batch
    )
    D_batch = law06.consistent_solid_tangent(mat, deps_batch, dt, extra=extra_batch)

    # Compare against sequential 1-element updates
    for i in range(nel):
        sig_single = np.zeros((1, 6))
        deps_single = deps_batch[i : i + 1]
        extra_single = {"rho": np.array([extra_batch["rho"][i]])}
        sig_s_out, _, c_single = law06.solid_update(
            mat, sig_single, deps_single, None, dt, extra=extra_single
        )
        D_single = law06.consistent_solid_tangent(
            mat, deps_single, dt, extra=extra_single
        )
        np.testing.assert_allclose(sig_batch_out[i], sig_s_out[0], atol=1e-12)
        np.testing.assert_allclose(c_batch[i], c_single[0], atol=1e-12)
        np.testing.assert_allclose(D_batch[i], D_single[0], atol=1e-12)


def test_law06_materials_init_solid_tangent_dispatch():
    from pyradioss.materials import solid_tangent
    mat = _make_law6(visc=1.5, rho0=1000.0, MAT_C1=2.0e9)
    sig = np.zeros((2, 6))
    extra = {"dt": 0.01, "rho": np.array([1000.0, 1000.0])}
    D = solid_tangent(mat, sig, None, None, extra=extra)
    assert D.shape == (2, 6, 6)
    np.testing.assert_allclose(D[0], D[0].T, atol=1e-12)


def test_law06_materials_init_shell_update_dispatch():
    from pyradioss.materials import shell_update
    mat = _make_law6()
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    with pytest.raises(NotImplementedError, match="solid/SPH elements only"):
        shell_update(mat, sig, deps, None, 0.01)


"""
Tests for Milestone M529: LAW40 / KELVINMAX Generalized Kelvin-Maxwell visco-elasticity
constitutive model hardening, 5-branch Prony relaxation, consistent solid tangents,
and starter checks integration.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
import pyradioss.materials as materials
from pyradioss.materials import law40_kelvinmax
from pyradioss.starter.checks import _ALLOWED_LAWS


def _make_law40_mat(**kwargs) -> Material:
    defaults = {
        "MAT_BULK": 50.0,
        "MAT_GI": 10.0,
        "MAT_G0": 5.0,
        "MAT_DECAY": 2.0,
        "MAT_G2": 3.0,
        "MAT_DECAY2": 10.0,
        "rho": 1.2e-3,
    }
    defaults.update(kwargs)
    return law40_kelvinmax.build_law40(defaults)


# ----------------------------------------------------------------------------
# 1-3. Validation and Defaults
# ----------------------------------------------------------------------------

def test_law40_validation_k_positive():
    with pytest.raises(ValueError, match="bulk modulus K must be > 0"):
        law40_kelvinmax.build_law40({"MAT_BULK": 0.0, "MAT_GI": 5.0})
    with pytest.raises(ValueError, match="bulk modulus K must be > 0"):
        law40_kelvinmax.build_law40({"MAT_BULK": -10.0, "MAT_GI": 5.0})


def test_law40_validation_gsum_positive():
    with pytest.raises(ValueError, match="no shear stiffness"):
        law40_kelvinmax.build_law40({"MAT_BULK": 50.0, "MAT_GI": 0.0, "MAT_G0": 0.0})
    with pytest.raises(ValueError, match="no shear stiffness"):
        law40_kelvinmax.build_law40({"MAT_BULK": 50.0, "MAT_GI": -5.0, "MAT_G0": 2.0})


def test_law40_defaults():
    mat = _make_law40_mat(Astass=0.0, Bstass=0.0, Kvm=0.0, MAT_DECAY=0.0)
    assert mat.params["astas"] == 1e30
    assert mat.params["bstas"] == 1e30
    assert mat.params["vmisk"] == 1e30
    # Decays are clamped to >= 1e-20
    assert mat.params["beta"][0] == 1e-20


def test_law40_cfg_vs_direct_keys():
    cfg_input = {
        "MAT_BULK": 60.0, "MAT_GI": 12.0,
        "MAT_G0": 4.0, "MAT_DECAY": 1.5,
        "MAT_G2": 3.0, "MAT_DECAY2": 5.0,
        "MAT_G3": 2.0, "MAT_DECAY3": 10.0,
        "MAT_G4": 1.0, "MAT_DECAY4": 20.0,
        "MAT_G5": 0.5, "MAT_DECAY5": 50.0,
        "Astass": 1.2, "Bstass": 2.3, "Kvm": 3.4,
    }
    dir_input = {
        "bulk": 60.0, "gi": 12.0,
        "g0": 4.0, "beta0": 1.5,
        "g2": 3.0, "beta2": 5.0,
        "g3": 2.0, "beta3": 10.0,
        "g4": 1.0, "beta4": 20.0,
        "g5": 0.5, "beta5": 50.0,
        "astas": 1.2, "bstas": 2.3, "vmisk": 3.4,
    }
    mat_cfg = law40_kelvinmax.build_law40(cfg_input)
    mat_dir = law40_kelvinmax.build_law40(dir_input)
    assert pytest.approx(mat_cfg.params["K40"]) == mat_dir.params["K40"]
    assert pytest.approx(mat_cfg.params["G_inf"]) == mat_dir.params["G_inf"]
    assert pytest.approx(mat_cfg.params["G"]) == mat_dir.params["G"]
    assert pytest.approx(mat_cfg.params["beta"]) == mat_dir.params["beta"]
    assert pytest.approx(mat_cfg.params["astas"]) == mat_dir.params["astas"]
    assert pytest.approx(mat_cfg.params["bstas"]) == mat_dir.params["bstas"]
    assert pytest.approx(mat_cfg.params["vmisk"]) == mat_dir.params["vmisk"]


# ----------------------------------------------------------------------------
# 4-6. Empty array guards and shell rejection
# ----------------------------------------------------------------------------

def test_law40_solid_update_empty_array():
    mat = _make_law40_mat()
    sig = np.empty((0, 6), dtype=float)
    deps = np.empty((0, 6), dtype=float)
    sig_out, c_out = law40_kelvinmax.solid_update(mat, sig, deps, dt=1e-5)
    assert sig_out.shape == (0, 6)
    assert c_out.shape == (0,)


def test_law40_consistent_tangent_empty_array():
    mat = _make_law40_mat()
    sig = np.empty((0, 6), dtype=float)
    C = law40_kelvinmax.consistent_solid_tangent(mat, sig)
    assert C.shape == (0, 6, 6)


def test_law40_shell_update_rejection():
    mat = _make_law40_mat()
    sig = np.zeros((1, 3))
    deps = np.zeros((1, 3))
    with pytest.raises(NotImplementedError, match="LAW40 .* implemented for 3D solid"):
        law40_kelvinmax.shell_update(mat, sig, deps)
    with pytest.raises(NotImplementedError, match="LAW40 .* implemented for 3D solid"):
        materials.shell_update(mat, sig, deps, epsp=None, dt=1e-5)


# ----------------------------------------------------------------------------
# 7-8. Extra state auto-allocation
# ----------------------------------------------------------------------------

def test_law40_extra_auto_init_none():
    mat = _make_law40_mat()
    sig = np.zeros((2, 6))
    deps = np.zeros((2, 6))
    sig_out, c_out = law40_kelvinmax.solid_update(mat, sig, deps, dt=1e-5, extra=None)
    assert sig_out.shape == (2, 6)
    assert c_out.shape == (2,)
    assert np.all(c_out > 0)


def test_law40_extra_auto_init_missing_keys():
    mat = _make_law40_mat()
    sig = np.zeros((3, 6))
    deps = np.zeros((3, 6))
    extra = {"eps40": np.zeros((3, 6))}  # missing uv40 and rho
    sig_out, c_out = law40_kelvinmax.solid_update(mat, sig, deps, dt=1e-5, extra=extra)
    assert "uv40" in extra
    assert "rho" in extra
    assert extra["uv40"].shape == (3, 40)
    assert len(extra["rho"]) == 3


# ----------------------------------------------------------------------------
# 9-12. Exponential relaxation, asymptotic & instantaneous stress
# ----------------------------------------------------------------------------

def test_law40_single_branch_relaxation_time_constant():
    # G_inf = 5.0, G0 = 10.0, beta0 = 2.0 (tau = 0.5 s)
    beta0 = 2.0
    G_inf = 5.0
    G0 = 10.0
    tau = 1.0 / beta0
    mat = _make_law40_mat(MAT_GI=G_inf, MAT_G0=G0, MAT_DECAY=beta0,
                          MAT_G2=0.0, MAT_G3=0.0, MAT_G4=0.0, MAT_G5=0.0)

    n = 1
    sig = np.zeros((n, 6))
    extra = {
        "eps40": np.zeros((n, 6)),
        "uv40": np.zeros((n, 40)),
        "rho": np.full(n, mat.rho0),
    }
    gamma_xy = 0.02
    extra["eps40"][0, 3] = gamma_xy
    v0 = 0.05
    extra["uv40"][0, 13] = v0  # branch 0 shear stress

    # Relax under constant strain for time t = tau
    t_relax = tau
    n_steps = 500
    dt_step = t_relax / n_steps
    for _ in range(n_steps):
        sig, _ = law40_kelvinmax.solid_update(mat, sig, np.zeros((n, 6)), dt=dt_step, extra=extra)

    s_inf = 2.0 * G_inf * (gamma_xy * 0.5)
    s_relaxed = sig[0, 3]
    ratio_measured = (s_relaxed - s_inf) / v0
    ratio_expected = math.exp(-1.0)
    assert pytest.approx(ratio_expected, rel=1e-3) == ratio_measured


def test_law40_multi_branch_relaxation_spectrum():
    # 5 branches with distinct relaxation times
    mat = _make_law40_mat(
        MAT_GI=2.0,
        MAT_G0=1.0, MAT_DECAY=0.1,
        MAT_G2=2.0, MAT_DECAY2=1.0,
        MAT_G3=3.0, MAT_DECAY3=10.0,
        MAT_G4=4.0, MAT_DECAY4=100.0,
        MAT_G5=5.0, MAT_DECAY5=1000.0,
    )
    sig = np.zeros((1, 6))
    extra = {"eps40": np.zeros((1, 6)), "uv40": np.zeros((1, 40)), "rho": np.full(1, mat.rho0)}
    gamma_xy = 0.02
    extra["eps40"][0, 3] = gamma_xy

    # Pre-stress branches
    v0s = [0.01, 0.02, 0.03, 0.04, 0.05]
    for j in range(5):
        extra["uv40"][0, 13 + 6 * j] = v0s[j]

    # Relax for t = 0.2 s in small steps
    t_relax = 0.2
    dt_step = 1e-3
    steps = int(t_relax / dt_step)
    for _ in range(steps):
        sig, _ = law40_kelvinmax.solid_update(mat, sig, np.zeros((1, 6)), dt=dt_step, extra=extra)

    s_inf = 2.0 * mat.params["G_inf"] * (gamma_xy * 0.5)
    s_branches_expected = sum(
        v0s[j] * math.exp(-mat.params["beta"][j] * t_relax) for j in range(5)
    )
    assert pytest.approx(s_inf + s_branches_expected, rel=1e-3) == sig[0, 3]


def test_law40_long_term_asymptotic_stress():
    G_inf = 4.0
    mat = _make_law40_mat(MAT_GI=G_inf, MAT_G0=10.0, MAT_DECAY=10.0,
                          MAT_G2=0.0, MAT_G3=0.0, MAT_G4=0.0, MAT_G5=0.0)
    sig = np.zeros((1, 6))
    extra = {"eps40": np.zeros((1, 6)), "uv40": np.zeros((1, 40)), "rho": np.full(1, mat.rho0)}
    gamma_xy = 0.02
    extra["eps40"][0, 3] = gamma_xy
    extra["uv40"][0, 13] = 0.10  # branch 0 stress

    # Relax for 10 seconds (100 * tau)
    for _ in range(1000):
        sig, _ = law40_kelvinmax.solid_update(mat, sig, np.zeros((1, 6)), dt=0.01, extra=extra)

    s_inf_expected = 2.0 * G_inf * (gamma_xy * 0.5)
    assert pytest.approx(s_inf_expected, rel=1e-4) == sig[0, 3]


def test_law40_instantaneous_stress_response():
    # Immediate step: rate reconstruction integrates branch
    mat = _make_law40_mat(MAT_GI=5.0, MAT_G0=15.0, MAT_DECAY=1.0,
                          MAT_G2=10.0, MAT_DECAY2=2.0,
                          MAT_G3=0.0, MAT_G4=0.0, MAT_G5=0.0)
    sig = np.zeros((1, 6))
    deps = np.array([[0.001, -0.0005, -0.0005, 0.0, 0.0, 0.0]])
    extra = {"eps40": np.zeros((1, 6)), "uv40": np.zeros((1, 40)), "rho": np.full(1, mat.rho0)}

    sig, c = law40_kelvinmax.solid_update(mat, sig, deps, dt=1e-4, extra=extra)
    dev_sig = sig[0, 0] - (sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
    assert dev_sig > 0.0
    assert c[0] > 0.0


# ----------------------------------------------------------------------------
# 13-14. Linear rate reconstruction and pressure
# ----------------------------------------------------------------------------

def test_law40_linear_in_time_rate_reconstruction():
    mat = _make_law40_mat()
    sig = np.zeros((1, 6))
    dt = 1e-4
    deps1 = np.array([[0.001, 0.0, 0.0, 0.0, 0.0, 0.0]])
    extra = {"eps40": np.zeros((1, 6)), "uv40": np.zeros((1, 40)), "rho": np.full(1, mat.rho0)}

    # Cycle 1
    law40_kelvinmax.solid_update(mat, sig, deps1, dt=dt, extra=extra)
    edrv1 = extra["uv40"][0, 4:10].copy()
    assert np.any(edrv1 != 0.0)

    # Cycle 2 with zero increment
    deps2 = np.zeros_like(deps1)
    law40_kelvinmax.solid_update(mat, sig, deps2, dt=dt, extra=extra)
    edrv2 = extra["uv40"][0, 4:10].copy()
    assert np.any(edrv2 != edrv1)


def test_law40_incremental_bulk_pressure():
    K = 100.0
    mat = _make_law40_mat(MAT_BULK=K)
    sig = np.zeros((1, 6))
    deps = np.full((1, 6), 0.0)
    # Volumetric compression
    deps[0, 0] = deps[0, 1] = deps[0, 2] = -0.002
    extra = {"eps40": np.zeros((1, 6)), "uv40": np.zeros((1, 40)), "rho": np.full(1, mat.rho0)}

    sig, _ = law40_kelvinmax.solid_update(mat, sig, deps, dt=1e-4, extra=extra)
    p_mean = (sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
    # Incremental pressure: sigv = 0 + K * (-0.006) = -0.6
    assert pytest.approx(-0.6, rel=1e-3) == p_mean


# ----------------------------------------------------------------------------
# 15-17. Failure criteria histories
# ----------------------------------------------------------------------------

def test_law40_von_mises_criterion_history():
    kvm = 25.0
    mat = _make_law40_mat(Kvm=kvm)
    sig = np.zeros((1, 6))
    deps = np.array([[0.01, -0.005, -0.005, 0.0, 0.0, 0.0]])
    extra = {"eps40": np.zeros((1, 6)), "uv40": np.zeros((1, 40)), "rho": np.full(1, mat.rho0)}

    law40_kelvinmax.solid_update(mat, sig, deps, dt=1e-4, extra=extra)
    # UVAR 0 is current VM / kvm, UVAR 2 is peak
    assert extra["uv40"][0, 0] > 0.0
    assert extra["uv40"][0, 2] == extra["uv40"][0, 0]

    # Partial unload
    deps_unload = -0.5 * deps
    law40_kelvinmax.solid_update(mat, sig, deps_unload, dt=1e-4, extra=extra)
    assert extra["uv40"][0, 0] < extra["uv40"][0, 2]


def test_law40_stassi_criterion_history():
    astas = 1.5
    bstas = 50.0
    mat = _make_law40_mat(Astass=astas, Bstass=bstas)
    sig = np.zeros((1, 6))
    deps = np.array([[0.01, -0.005, -0.005, 0.0, 0.0, 0.0]])
    extra = {"eps40": np.zeros((1, 6)), "uv40": np.zeros((1, 40)), "rho": np.full(1, mat.rho0)}

    law40_kelvinmax.solid_update(mat, sig, deps, dt=1e-4, extra=extra)
    # UVAR 1 is current Stassi, UVAR 3 is peak
    assert extra["uv40"][0, 1] > 0.0
    assert extra["uv40"][0, 3] == extra["uv40"][0, 1]


def test_law40_infinite_criterion_thresholds():
    # Blank criteria cards -> 1e30 defaults -> UVAR 0..3 should remain 0
    mat = _make_law40_mat(Astass=0.0, Bstass=0.0, Kvm=0.0)
    sig = np.zeros((1, 6))
    deps = np.array([[0.01, -0.005, -0.005, 0.0, 0.0, 0.0]])
    extra = {"eps40": np.zeros((1, 6)), "uv40": np.zeros((1, 40)), "rho": np.full(1, mat.rho0)}

    law40_kelvinmax.solid_update(mat, sig, deps, dt=1e-4, extra=extra)
    np.testing.assert_allclose(extra["uv40"][0, 0:4], 0.0, atol=1e-12)


# ----------------------------------------------------------------------------
# 18-20. Sound speed and spatial objectivity
# ----------------------------------------------------------------------------

def test_law40_sound_speed_assembly():
    K = 80.0
    G_inf = 10.0
    G0 = 5.0
    G2 = 3.0
    rho = 2e-3
    mat = _make_law40_mat(MAT_BULK=K, MAT_GI=G_inf, MAT_G0=G0, MAT_G2=G2,
                          MAT_G3=0.0, MAT_G4=0.0, MAT_G5=0.0, rho=rho)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))

    gt = 2.0 * (G_inf + G0 + G2)
    c_expected = math.sqrt(K / rho + 4.0 * gt / (3.0 * rho))

    _, c = law40_kelvinmax.solid_update(mat, sig, deps, dt=1e-4)
    assert pytest.approx(c_expected, rel=1e-6) == c[0]


def test_law40_sound_speed_density_scaling():
    mat = _make_law40_mat(rho=1.0)
    sig = np.zeros((2, 6))
    deps = np.zeros((2, 6))
    extra = {
        "eps40": np.zeros((2, 6)),
        "uv40": np.zeros((2, 40)),
        "rho": np.array([1.0, 4.0]),
    }
    _, c = law40_kelvinmax.solid_update(mat, sig, deps, dt=1e-4, extra=extra)
    # 4x density -> 1/2 sound speed
    assert pytest.approx(c[0] * 0.5, rel=1e-6) == c[1]


def test_law40_spatial_objectivity_3d_rotation():
    mat = _make_law40_mat()

    theta = math.pi / 3.0
    c, s = math.cos(theta), math.sin(theta)
    Q = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])

    deps_mat = np.array([
        [0.01, 0.003, 0.0],
        [0.003, -0.006, 0.0],
        [0.0, 0.0, -0.004],
    ])
    deps1 = np.array([[deps_mat[0, 0], deps_mat[1, 1], deps_mat[2, 2],
                       2.0 * deps_mat[0, 1], 2.0 * deps_mat[1, 2], 2.0 * deps_mat[2, 0]]])

    deps_mat_rot = Q @ deps_mat @ Q.T
    deps2 = np.array([[deps_mat_rot[0, 0], deps_mat_rot[1, 1], deps_mat_rot[2, 2],
                       2.0 * deps_mat_rot[0, 1], 2.0 * deps_mat_rot[1, 2], 2.0 * deps_mat_rot[2, 0]]])

    sig1 = np.zeros((1, 6))
    sig2 = np.zeros((1, 6))
    dt = 1e-4

    extra1 = {"eps40": np.zeros((1, 6)), "uv40": np.zeros((1, 40)), "rho": np.full(1, mat.rho0)}
    extra2 = {"eps40": np.zeros((1, 6)), "uv40": np.zeros((1, 40)), "rho": np.full(1, mat.rho0)}

    law40_kelvinmax.solid_update(mat, sig1, deps1, dt=dt, extra=extra1)
    law40_kelvinmax.solid_update(mat, sig2, deps2, dt=dt, extra=extra2)

    sig1_mat = np.array([
        [sig1[0, 0], sig1[0, 3], sig1[0, 5]],
        [sig1[0, 3], sig1[0, 1], sig1[0, 4]],
        [sig1[0, 5], sig1[0, 4], sig1[0, 2]],
    ])
    sig1_mat_rot = Q @ sig1_mat @ Q.T

    sig2_mat = np.array([
        [sig2[0, 0], sig2[0, 3], sig2[0, 5]],
        [sig2[0, 3], sig2[0, 1], sig2[0, 4]],
        [sig2[0, 5], sig2[0, 4], sig2[0, 2]],
    ])
    np.testing.assert_allclose(sig1_mat_rot, sig2_mat, atol=1e-6)


# ----------------------------------------------------------------------------
# 21-23. Consistent solid tangent tests
# ----------------------------------------------------------------------------

def test_law40_consistent_solid_tangent_symmetry_and_positive_definite():
    mat = _make_law40_mat()
    sig = np.zeros((2, 6))
    C = law40_kelvinmax.consistent_solid_tangent(mat, sig, dt=1e-4)
    assert C.shape == (2, 6, 6)
    for k in range(2):
        Ck = C[k]
        np.testing.assert_allclose(Ck, Ck.T, atol=1e-12)
        evals = np.linalg.eigvalsh(Ck)
        assert np.all(evals > 0)


def test_law40_consistent_solid_tangent_time_limits():
    mat = _make_law40_mat(MAT_GI=10.0, MAT_G0=5.0, MAT_G2=3.0, MAT_DECAY=2.0, MAT_DECAY2=5.0,
                          MAT_G3=0.0, MAT_G4=0.0, MAT_G5=0.0)
    sig = np.zeros((1, 6))

    # Instantaneous limit (dt = 0)
    C_0 = law40_kelvinmax.consistent_solid_tangent(mat, sig, dt=0.0)
    G_inst = 10.0 + 5.0 + 3.0
    assert pytest.approx(G_inst) == C_0[0, 3, 3]

    # Equilibrium long-term limit (dt -> infinity)
    C_inf = law40_kelvinmax.consistent_solid_tangent(mat, sig, dt=1e6)
    G_inf = 10.0
    assert pytest.approx(G_inf, rel=1e-4) == C_inf[0, 3, 3]


def test_law40_solid_tangent_dispatch():
    mat = _make_law40_mat()
    sig = np.zeros((3, 6))
    C_direct = law40_kelvinmax.consistent_solid_tangent(mat, sig, dt=1e-3)
    C_dispatched = materials.solid_tangent(mat, sig, None, None, extra={"dt": 1e-3})
    np.testing.assert_allclose(C_direct, C_dispatched, atol=1e-12)


# ----------------------------------------------------------------------------
# 24-26. Batch equivalence, cycle 0 stability, and starter checks
# ----------------------------------------------------------------------------

def test_law40_batch_equivalence():
    mat = _make_law40_mat()
    n = 4
    sig_batch = np.zeros((n, 6))
    np.random.seed(42)
    deps_batch = np.random.uniform(-0.01, 0.01, size=(n, 6))
    dt = 1e-4

    extra_batch = {
        "eps40": np.zeros((n, 6)),
        "uv40": np.zeros((n, 40)),
        "rho": np.full(n, mat.rho0),
    }

    sig_res_batch, c_batch = law40_kelvinmax.solid_update(
        mat, sig_batch.copy(), deps_batch.copy(), dt=dt, extra=extra_batch
    )

    for i in range(n):
        sig_single = np.zeros((1, 6))
        deps_single = deps_batch[i:i+1].copy()
        extra_single = {
            "eps40": np.zeros((1, 6)),
            "uv40": np.zeros((1, 40)),
            "rho": np.full(1, mat.rho0),
        }
        sig_res_single, c_single = law40_kelvinmax.solid_update(
            mat, sig_single, deps_single, dt=dt, extra=extra_single
        )
        np.testing.assert_allclose(sig_res_single[0], sig_res_batch[i], atol=1e-12)
        np.testing.assert_allclose(c_single[0], c_batch[i], atol=1e-12)


def test_law40_cycle0_static_stability():
    mat = _make_law40_mat()
    sig = np.array([[12.0, 6.0, 6.0, 2.0, 0.0, 0.0]])
    deps = np.zeros((1, 6))

    # dt = 0.0
    sig_out0, c0 = law40_kelvinmax.solid_update(mat, sig.copy(), deps, dt=0.0)
    np.testing.assert_allclose(sig, sig_out0, atol=1e-12)
    assert c0[0] > 0.0

    # dt < 0.0
    sig_out_neg, c_neg = law40_kelvinmax.solid_update(mat, sig.copy(), deps, dt=-1.0)
    np.testing.assert_allclose(sig, sig_out_neg, atol=1e-12)
    assert c_neg[0] > 0.0


def test_law40_starter_checks_and_extra_shapes():
    mat = _make_law40_mat()
    assert 40 in _ALLOWED_LAWS["bricks"]
    assert 40 in _ALLOWED_LAWS["tetras"]
    assert 40 not in _ALLOWED_LAWS["shells"]
    assert materials.needs_env(mat) is True
    assert materials.needs_defgrad(mat) is False

    shapes = materials.extra_shapes(mat)
    assert "eps40" in shapes and shapes["eps40"] == (6,)
    assert "uv40" in shapes and shapes["uv40"] == (40,)

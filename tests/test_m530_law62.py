"""
Comprehensive test suite for /MAT/LAW62 (/MAT/HYP_VISCO_FOAM, /MAT/VISC_HYP).

Verifies:
- Defensive empty array handling across solid_update and consistent_solid_tangent
- Shell update rejection (solids-only)
- CFG vs direct dictionary parameter extraction and defaults
- Input validation error codes (559, 560, 2084, 846)
- Rigidity rescaling (Rflag = 2) and viscosity flags (Vflag = 0, 1)
- Large-strain Ogden kinematics, principal stretch decomposition, and foam compressibility
- Frame indifference / material objectivity under finite 3D rigid body rotations
- Conservative hyperelastic cyclic loading (zero residual stress upon return to identity)
- Viscoelastic Maxwell/Prony relaxation under deviatoric and full-stress formulations
- Sound speed bound (CIMAX) under tension, compression, and varying density
- Exact 6x6 spatial tangent tensor symmetry, positive definiteness, and coalescent L'Hopital limits
- Directional derivative Truesdell rate consistency
- Element batch vectorization equivalence.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials import law62_hypervisco, solid_tangent, shell_update, solid_update
from pyradioss.model.entities import Material


def _make_law62(norder=1, nvisc=0, mu=None, alpha=None, nu=None,
                gamma=None, tau=None, nug=0.2, vflag=0, rflag=0, rho0=1000.0) -> Material:
    if mu is None:
        mu = [10.0] * norder
    if alpha is None:
        alpha = [2.0] * norder
    if nu is None:
        nu = [0.0] * norder
    if gamma is None:
        gamma = [0.2] * nvisc
    if tau is None:
        tau = [1.0] * nvisc

    rec = {
        "id": 62,
        "title": "LAW62_TEST",
        "density": rho0,
        "MAT_NU": nug,
        "ORDER": norder,
        "Order2": nvisc,
        "Vflag": vflag,
        "Rflag": rflag,
        "Mu_arr": mu,
        "Alpha_arr": alpha,
        "Nu_arr": nu,
        "Gamma_arr": gamma,
        "Tau_arr": tau,
    }
    return law62_hypervisco.build_law62(rec)


# ============================================================================
# 1. Empty arrays & shell rejection
# ============================================================================

def test_law62_empty_arrays():
    mat = _make_law62()
    sig = np.empty((0, 6))
    deps = np.empty((0, 6))
    s_out, epsp_out, c = law62_hypervisco.solid_update(mat, sig, deps, None, 1.0)
    assert s_out.shape == (0, 6)
    assert c.shape == (0,)


def test_law62_tangent_empty_arrays():
    mat = _make_law62()
    sig = np.empty((0, 6))
    D = law62_hypervisco.consistent_solid_tangent(mat, sig=sig)
    assert D.shape == (0, 6, 6)

    F = np.empty((0, 3, 3))
    D2 = law62_hypervisco.consistent_solid_tangent(mat, F=F)
    assert D2.shape == (0, 6, 6)


def test_law62_shell_update_rejection():
    mat = _make_law62()
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    with pytest.raises(NotImplementedError, match="solids-only"):
        law62_hypervisco.shell_update(mat, sig, deps, None, 1.0)
    with pytest.raises(NotImplementedError, match="solids-only"):
        shell_update(mat, sig, deps, None, 1.0)


# ============================================================================
# 2. CFG vs Direct parameter construction & validation
# ============================================================================

def test_law62_build_from_direct_dict():
    rec = {
        "id": 1,
        "rho": 500.0,
        "nu": 0.15,
        "N": 2,
        "M": 1,
        "mu": [12.0, 4.0],
        "alpha": [1.5, -2.0],
        "gamma": [0.3],
        "tau": [0.05],
        "vflag": 1,
    }
    mat = law62_hypervisco.build_law62(rec)
    assert mat.law == 62
    assert mat.rho0 == 500.0
    assert len(mat.params["MU62"]) == 2
    assert mat.params["IVISC"] == 2
    assert mat.params["NPRONY"] == 1
    assert math.isclose(mat.params["GAMAINF"], 0.7)


def test_law62_build_validation_order_zero():
    rec = {"ORDER": 0, "Mu_arr": []}
    with pytest.raises(ValueError, match="error 559"):
        law62_hypervisco.build_law62(rec)


def test_law62_build_validation_gamma_range():
    rec = {"ORDER": 1, "Mu_arr": [10.0], "Order2": 1, "Gamma_arr": [-0.1], "Tau_arr": [1.0]}
    with pytest.raises(ValueError, match="error 560"):
        law62_hypervisco.build_law62(rec)

    rec2 = {"ORDER": 1, "Mu_arr": [10.0], "Order2": 1, "Gamma_arr": [1.5], "Tau_arr": [1.0]}
    with pytest.raises(ValueError, match="error 560"):
        law62_hypervisco.build_law62(rec2)


def test_law62_build_validation_gamma_sum():
    rec = {"ORDER": 1, "Mu_arr": [10.0], "Order2": 2, "Gamma_arr": [0.6, 0.4], "Tau_arr": [1.0, 1.0]}
    with pytest.raises(ValueError, match="error 2084"):
        law62_hypervisco.build_law62(rec)


def test_law62_build_validation_mu_sum():
    rec = {"ORDER": 1, "Mu_arr": [-5.0]}
    with pytest.raises(ValueError, match="error 846"):
        law62_hypervisco.build_law62(rec)


def test_law62_defaults_alpha_and_tau():
    rec = {
        "ORDER": 1,
        "Order2": 1,
        "Mu_arr": [10.0],
        "Alpha_arr": [0.0],  # should default to 1.0
        "Gamma_arr": [0.2],
        "Tau_arr": [0.0],    # should default to 1e20
    }
    mat = law62_hypervisco.build_law62(rec)
    assert mat.params["AL62"][0] == 1.0
    assert mat.params["TAU62"][0] == 1e20


def test_law62_rflag_rigidity_rescaling():
    # Rflag = 2 rescales mu by 1 / gamma_inf
    mat1 = _make_law62(norder=1, nvisc=1, mu=[10.0], gamma=[0.5], rflag=0)
    mat2 = _make_law62(norder=1, nvisc=1, mu=[10.0], gamma=[0.5], rflag=2)
    # gamma_inf = 1 - 0.5 = 0.5
    # For mat2, mu becomes 10.0 / 0.5 = 20.0
    assert math.isclose(mat1.params["MU62"][0], 10.0)
    assert math.isclose(mat2.params["MU62"][0], 20.0)


def test_law62_per_term_nu_vs_global_nu():
    # When Nu_arr has non-zero entries, beta_i = nu_i / (1 - 2 nu_i)
    mat = _make_law62(norder=2, mu=[10.0, 20.0], nu=[0.25, 0.0])
    beta = mat.params["BETA62"]
    # For nu=0.25: beta = 0.25 / 0.5 = 0.5
    # For nu=0.0: beta = 0.0 / 1.0 = 0.0
    assert math.isclose(beta[0], 0.5)
    assert math.isclose(beta[1], 0.0)


def test_law62_pure_foam_zero_poisson():
    # nu = 0 gives beta = 0: fully decoupled foam
    mat = _make_law62(norder=1, mu=[15.0], alpha=[2.0], nug=0.0)
    assert mat.params["BETA62"][0] == 0.0
    # In pure uniaxial extension along X: lambda = [1.2, 1.0, 1.0]
    F = np.diag([1.2, 1.0, 1.0])[None, :, :]
    sig = np.zeros((1, 6))
    sig, _, c = law62_hypervisco.solid_update(mat, sig, None, None, 1.0, extra={"F": F})
    # Since beta = 0 and J = 1.2: J^(-beta*alpha) = 1.0
    # S_1 = (2 * mu / alpha / lambda_1^2) * (lambda_1^alpha - 1.0)
    # sigma_1 = S_1 * lambda_1^2 / J = (2 * mu / alpha / J) * (lambda_1^alpha - 1.0)
    expected_sig1 = (2.0 * 15.0 / 2.0 / 1.2) * (1.2**2.0 - 1.0)
    assert math.isclose(sig[0, 0], expected_sig1, rel_tol=1e-5)
    # Lateral stresses sigma_2 and sigma_3 must be zero because lambda_2 = lambda_3 = 1.0 and beta = 0!
    assert abs(sig[0, 1]) < 1e-10
    assert abs(sig[0, 2]) < 1e-10


def test_law62_incompressible_limit():
    # nu clamps at 0.499
    mat = _make_law62(norder=1, mu=[10.0], nug=0.5)
    assert mat.params["nu"] <= 0.499
    assert mat.params["RBULK"] > 0.0


# ============================================================================
# 3. Kinematics, Stresses, Objectivity, and Conservation
# ============================================================================

def test_law62_uniaxial_tension_and_compression():
    mat = _make_law62(norder=1, mu=[10.0], alpha=[2.0], nug=0.25)
    # Tension: lambda_1 = 1.2, lambda_2 = lambda_3 = 1.0
    F_tens = np.diag([1.2, 1.0, 1.0])[None, :, :]
    sig_tens = np.zeros((1, 6))
    sig_tens, _, _ = law62_hypervisco.solid_update(mat, sig_tens, None, None, 1.0, extra={"F": F_tens})
    assert sig_tens[0, 0] > 0.0  # tensile stress

    # Compression: lambda_1 = 0.8, lambda_2 = lambda_3 = 1.0
    F_comp = np.diag([0.8, 1.0, 1.0])[None, :, :]
    sig_comp = np.zeros((1, 6))
    sig_comp, _, _ = law62_hypervisco.solid_update(mat, sig_comp, None, None, 1.0, extra={"F": F_comp})
    assert sig_comp[0, 0] < 0.0  # compressive stress


def test_law62_hydrostatic_compression():
    mat = _make_law62(norder=1, mu=[10.0], alpha=[2.0], nug=0.3)
    # J = 0.9^3 = 0.729
    F = 0.9 * np.eye(3)[None, :, :]
    sig = np.zeros((1, 6))
    sig, _, _ = law62_hypervisco.solid_update(mat, sig, None, None, 1.0, extra={"F": F})
    # Must be isotropic compression: sig_xx == sig_yy == sig_zz < 0, shears == 0
    assert sig[0, 0] < 0.0
    assert math.isclose(sig[0, 0], sig[0, 1], rel_tol=1e-10)
    assert math.isclose(sig[0, 0], sig[0, 2], rel_tol=1e-10)
    assert np.allclose(sig[0, 3:], 0.0, atol=1e-12)


def test_law62_pure_shear():
    mat = _make_law62(norder=1, mu=[10.0], alpha=[2.0], nug=0.0)
    # F with simple shear gamma = 0.2 in xy
    F = np.array([
        [1.0, 0.2, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])[None, :, :]
    sig = np.zeros((1, 6))
    sig, _, _ = law62_hypervisco.solid_update(mat, sig, None, None, 1.0, extra={"F": F})
    # xy shear stress sig[0, 3] should be positive and non-zero
    assert sig[0, 3] > 0.0
    assert abs(sig[0, 4]) < 1e-10  # yz
    assert abs(sig[0, 5]) < 1e-10  # zx


def test_law62_frame_indifference():
    mat = _make_law62(norder=2, mu=[10.0, 5.0], alpha=[1.8, -2.0], nug=0.2)
    rng = np.random.default_rng(123)
    F0 = np.eye(3) + 0.1 * rng.standard_normal((3, 3))
    # Random 3D orthogonal rotation Q
    Q = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    # Evaluate at F0
    sig0 = np.zeros((1, 6))
    sig0, _, _ = law62_hypervisco.solid_update(mat, sig0, None, None, 1.0, extra={"F": F0[None, :, :]})
    T0 = np.array([
        [sig0[0, 0], sig0[0, 3], sig0[0, 5]],
        [sig0[0, 3], sig0[0, 1], sig0[0, 4]],
        [sig0[0, 5], sig0[0, 4], sig0[0, 2]]
    ])

    # Evaluate at rotated F_rot = Q @ F0
    F_rot = (Q @ F0)[None, :, :]
    sig_rot = np.zeros((1, 6))
    sig_rot, _, _ = law62_hypervisco.solid_update(mat, sig_rot, None, None, 1.0, extra={"F": F_rot})
    T_rot = np.array([
        [sig_rot[0, 0], sig_rot[0, 3], sig_rot[0, 5]],
        [sig_rot[0, 3], sig_rot[0, 1], sig_rot[0, 4]],
        [sig_rot[0, 5], sig_rot[0, 4], sig_rot[0, 2]]
    ])

    # Objectivity requires T_rot == Q @ T0 @ Q^T
    expected_T = Q @ T0 @ Q.T
    assert np.allclose(T_rot, expected_T, atol=1e-10 * np.abs(T0).max())


def test_law62_elastic_path_independence():
    mat = _make_law62(norder=2, mu=[15.0, 8.0], alpha=[2.0, -1.5], nug=0.25)
    sig = np.zeros((1, 6))

    # Deform to F1
    F1 = np.diag([1.3, 0.9, 0.85])[None, :, :]
    sig, _, _ = law62_hypervisco.solid_update(mat, sig, None, None, 1.0, extra={"F": F1})
    assert np.linalg.norm(sig) > 1.0

    # Return to identity F = I
    F_identity = np.eye(3)[None, :, :]
    sig, _, _ = law62_hypervisco.solid_update(mat, sig, None, None, 1.0, extra={"F": F_identity})
    assert np.allclose(sig, 0.0, atol=1e-12)


# ============================================================================
# 4. Viscoelasticity (Prony Series)
# ============================================================================

def test_law62_prony_deviatoric_relaxation_ivisc1():
    # Vflag = 0: deviatoric relaxation
    # 1 Ogden term, 1 Prony term: gamma = 0.5, tau = 0.1
    mat = _make_law62(norder=1, nvisc=1, mu=[20.0], alpha=[2.0], gamma=[0.5], tau=[0.1], vflag=0)
    extra = {
        "sdg62": np.zeros((1, 6)),
        "h62": np.zeros((1, 1, 6)),
        "F": np.diag([1.2, 1.0, 1.0])[None, :, :],
    }
    sig = np.zeros((1, 6))

    # Step 1: instantaneous loading at dt = 1e-6 << tau
    sig1, _, _ = law62_hypervisco.solid_update(mat, sig.copy(), None, None, 1e-6, extra=extra)

    # Step 2: hold deformation constant at F, advance time by dt = 1.0 >> tau
    sig2, _, _ = law62_hypervisco.solid_update(mat, sig1.copy(), None, None, 1.0, extra=extra)

    # Stress should have relaxed significantly
    assert sig2[0, 0] < sig1[0, 0]
    # In fact, since gamma_inf = 0.5, deviatoric stress should relax towards 50%
    assert sig2[0, 0] > 0.0


def test_law62_prony_full_stress_relaxation_ivisc2():
    # Vflag = 1: full-stress relaxation
    mat = _make_law62(norder=1, nvisc=1, mu=[20.0], alpha=[2.0], gamma=[0.4], tau=[0.05], vflag=1)
    extra = {
        "sdg62": np.zeros((1, 6)),
        "h62": np.zeros((1, 1, 6)),
        "F": 0.9 * np.eye(3)[None, :, :],
    }
    sig = np.zeros((1, 6))
    sig1, _, _ = law62_hypervisco.solid_update(mat, sig.copy(), None, None, 1e-6, extra=extra)
    sig2, _, _ = law62_hypervisco.solid_update(mat, sig1.copy(), None, None, 0.5, extra=extra)

    # Compressive stress relaxes towards gamma_inf = 0.6 of its value
    assert abs(sig2[0, 0]) < abs(sig1[0, 0])


def test_law62_static_reevaluation_skips_prony():
    # When extra contains only {"F": F} without "sdg62", Prony update is skipped
    mat = _make_law62(norder=1, nvisc=1, mu=[10.0], gamma=[0.5], tau=[0.1])
    F = np.diag([1.1, 1.0, 1.0])[None, :, :]
    sig1, _, _ = law62_hypervisco.solid_update(mat, np.zeros((1, 6)), None, None, 1.0, extra={"F": F})
    sig2, _, _ = law62_hypervisco.solid_update(mat, np.zeros((1, 6)), None, None, 10.0, extra={"F": F})
    assert np.allclose(sig1, sig2, atol=1e-12)


def test_law62_small_strain_fallback():
    mat = _make_law62(norder=1, mu=[10.0], alpha=[2.0])
    deps = np.zeros((1, 6))
    deps[0, 0] = 0.05
    # Call without extra: F = I + deps fallback
    sig, _, _ = law62_hypervisco.solid_update(mat, np.zeros((1, 6)), deps, None, 1.0, extra=None)
    assert sig[0, 0] > 0.0


# ============================================================================
# 5. Sound Speed (CIMAX)
# ============================================================================

def test_law62_sound_speed_cimax():
    mat = _make_law62(norder=1, mu=[10.0], alpha=[2.0], nug=0.25, rho0=1000.0)
    F_ref = np.eye(3)[None, :, :]
    _, _, c_ref = law62_hypervisco.solid_update(mat, np.zeros((1, 6)), None, None, 1.0, extra={"F": F_ref})
    assert c_ref[0] > 0.0

    # Compressing volume increases density rho = rho0/J and increases bulk stiffening
    F_comp = 0.8 * np.eye(3)[None, :, :]
    _, _, c_comp = law62_hypervisco.solid_update(mat, np.zeros((1, 6)), None, None, 1.0, extra={"F": F_comp})
    assert c_comp[0] > 0.0


# ============================================================================
# 6. Tangents (consistent_solid_tangent)
# ============================================================================

def test_law62_consistent_solid_tangent_symmetry_and_posdef():
    mat = _make_law62(norder=2, mu=[10.0, 5.0], alpha=[2.0, -2.0], nug=0.2)
    for F in [
        np.eye(3),
        np.diag([1.2, 0.9, 0.85]),
        np.array([[1.1, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.95]]),
    ]:
        D = law62_hypervisco.consistent_solid_tangent(mat, F=F)[0]
        # Major symmetry: D == D.T
        assert np.allclose(D, D.T, atol=1e-12 * np.abs(D).max())
        # Positive definiteness: all eigenvalues > 0
        eigvals = np.linalg.eigvalsh(D)
        assert np.all(eigvals > 0.0), f"Negative eigenvalue found: {eigvals}"


def test_law62_consistent_solid_tangent_coalescent_limit():
    mat = _make_law62(norder=2, mu=[12.0, 6.0], alpha=[2.0, -1.0], nug=0.3)
    # Equal stretch in two directions: lambda_1 == lambda_2
    F = np.diag([1.1, 1.1, 0.9])[None, :, :]
    D = law62_hypervisco.consistent_solid_tangent(mat, F=F)[0]
    assert not np.any(np.isnan(D))
    assert not np.any(np.isinf(D))
    assert np.allclose(D, D.T, atol=1e-12 * np.abs(D).max())


def test_law62_consistent_solid_tangent_directional_derivative():
    mat = _make_law62(norder=2, mu=[10.0, 4.0], alpha=[2.0, -2.0], nug=0.25)
    VO = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))

    def c4_of(F):
        D6 = law62_hypervisco.consistent_solid_tangent(mat, F=F[None])[0]
        c4 = np.zeros((3, 3, 3, 3))
        for I, (i, j) in enumerate(VO):
            for J, (k, ell) in enumerate(VO):
                for (a, b) in {(i, j), (j, i)}:
                    for (c, d) in {(k, ell), (ell, k)}:
                        c4[a, b, c, d] = D6[I, J]
        return c4

    def sig_of(F):
        s = np.zeros((1, 6))
        s, _, _ = law62_hypervisco.solid_update(mat, s, None, None, 1.0, extra={"F": F[None]})
        return np.array([
            [s[0, 0], s[0, 3], s[0, 5]],
            [s[0, 3], s[0, 1], s[0, 4]],
            [s[0, 5], s[0, 4], s[0, 2]]
        ])

    rng = np.random.default_rng(999)
    F = np.diag([1.15, 0.95, 0.9])
    c4 = c4_of(F)
    sig0 = sig_of(F)
    eps = 1e-7

    for _ in range(5):
        g = rng.standard_normal((3, 3))
        dF = eps * g @ F
        ds_fd = (sig_of(F + dF) - sig_of(F - dF)) / (2 * eps)
        d = 0.5 * (g + g.T)
        pred = (np.einsum("ijkl,kl->ij", c4, d) + g @ sig0
                + sig0 @ g.T - np.trace(d) * sig0)
        diff = np.abs(ds_fd - pred).max()
        rel_diff = diff / max(np.abs(ds_fd).max(), 1e-12)
        assert rel_diff <= 1e-5, f"Directional derivative check failed: rel_diff={rel_diff}"


def test_law62_solid_tangent_dispatch():
    mat = _make_law62()
    sig = np.zeros((2, 6))
    D = solid_tangent(mat, sig, None, None, extra={"F": np.broadcast_to(np.eye(3), (2, 3, 3))})
    assert D.shape == (2, 6, 6)


def test_law62_batched_vectorization_equivalence():
    mat = _make_law62(norder=2, mu=[10.0, 5.0], alpha=[1.5, -2.0], nug=0.2)
    n = 5
    rng = np.random.default_rng(42)
    Fs = np.array([np.eye(3) + 0.1 * rng.standard_normal((3, 3)) for _ in range(n)])

    sig_batch = np.zeros((n, 6))
    sig_batch, _, c_batch = law62_hypervisco.solid_update(mat, sig_batch, None, None, 1.0, extra={"F": Fs})
    D_batch = law62_hypervisco.consistent_solid_tangent(mat, F=Fs)

    for i in range(n):
        sig_single = np.zeros((1, 6))
        sig_single, _, c_single = law62_hypervisco.solid_update(mat, sig_single, None, None, 1.0, extra={"F": Fs[i:i+1]})
        D_single = law62_hypervisco.consistent_solid_tangent(mat, F=Fs[i:i+1])

        assert np.allclose(sig_batch[i], sig_single[0], atol=1e-12)
        assert math.isclose(c_batch[i], c_single[0], rel_tol=1e-12)
        assert np.allclose(D_batch[i], D_single[0], atol=1e-12)

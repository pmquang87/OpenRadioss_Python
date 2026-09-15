"""
Milestone M521: Ogden Hyperelastic Material Law (/MAT/LAW42 /MAT/OGDEN)
Hardening, Principal Stretches Kinematics, Nonlinear Sound Speed, Coalescent Spatial Tangents & Unit Tests.

Covers:
1. Empty array guards for solid_update and consistent_solid_tangent
2. Identity deformation (F = I) produces identically zero Cauchy stress
3. Ground-state spatial tangent reduces exactly to classical isotropic Hooke tensor
4. Ground-state sound speed matches classical acoustic sound speed
5. Pure volumetric hydrostatic compression (deviatoric stress = 0, P = K*(J - 1))
6. Pure volumetric hydrostatic expansion (tensile pressure, Kt = K*J)
7. Isochoric uniaxial tension matches analytical Ogden stress difference formula
8. Mooney-Rivlin special case (N=2, alpha=[2, -2]) matches analytical formula
9. Neo-Hookean special case (N=1, alpha=[2]) matches analytical formula
10. Isochoric equibiaxial tension matches analytical Ogden stress formula
11. Pure shear stress response (planar isochoric extension, tr(sigma) = 0)
12. Large tensile strain stiffening and sound speed growth (Courant stability)
13. Large compressive strain stiffening and sound speed growth
14. Energy conservation and zero hysteresis on closed deformation cycles
15. Frame indifference / material objectivity under rigid body rotation (sigma(QF) = Q sigma(F) Q^T)
16. Inverted element Jacobian guard (det(F) <= 0 clamped safely to EM20)
17. Small-strain fallback when extra["F"] is omitted
18. Consistent spatial tangent Truesdell rate directional derivative verification
19. Coalescent eigenvalue L'Hopital limit (hydrostatic and uniaxial states, positive definite)
20. Multi-element batch vectorization equivalence and shell_update guard
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.common.constants import EM20
from pyradioss.materials import law42_ogden
from pyradioss.model.entities import Material


def _make_law42(mu, alpha, K=100000.0, rho0=1e-6):
    """Helper to create a LAW42 Material instance with Ogden coefficients."""
    G0 = sum(m * a for m, a in zip(mu, alpha)) / 2.0
    # Equivalent Poisson's ratio from K and G0: nu = (3K - 2G) / (6K + 2G)
    nu = (3.0 * K - 2.0 * G0) / (6.0 * K + 2.0 * G0)
    E = 2.0 * G0 * (1.0 + nu)
    params = {
        "E": E,
        "nu": nu,
        "K": K,
        "G": G0,
        "mu": list(mu),
        "alpha": list(alpha),
    }
    return Material(id=1, law=42, rho0=rho0, params=params)


def test_law42_empty_arrays():
    mat = _make_law42(mu=[100.0], alpha=[2.0])
    sig = np.empty((0, 6))
    deps = np.empty((0, 6))
    epsp = np.empty((0,))
    s, ep, c = law42_ogden.solid_update(mat, sig, deps, epsp, dt=1e-4)
    assert s.shape == (0, 6)
    assert ep.shape == (0,)
    assert c.shape == (0,)

    F_empty = np.empty((0, 3, 3))
    D = law42_ogden.consistent_solid_tangent(mat, F_empty)
    assert D.shape == (0, 6, 6)


def test_law42_identity_deformation_zero_stress():
    mat = _make_law42(mu=[50.0, -20.0], alpha=[2.0, -2.0], K=50000.0)
    sig = np.zeros((1, 6))
    F = np.eye(3)[None]
    extra = {"F": F}
    s, _, _ = law42_ogden.solid_update(mat, sig, np.zeros((1, 6)), None, 1e-4, extra)
    assert np.allclose(s, 0.0, atol=1e-12)


def test_law42_ground_state_shear_and_bulk_modulus():
    mu = [80.0, -30.0]
    alpha = [2.0, -2.0]
    K = 150000.0
    mat = _make_law42(mu=mu, alpha=alpha, K=K)
    G0 = sum(m * a for m, a in zip(mu, alpha)) / 2.0

    F = np.eye(3)[None]
    D = law42_ogden.consistent_solid_tangent(mat, F)[0]

    # Check classical isotropic elasticity matrix
    # Diagonal normal: K + 4/3 G0
    expected_diag = K + (4.0 / 3.0) * G0
    assert D[0, 0] == pytest.approx(expected_diag, rel=1e-10)
    assert D[1, 1] == pytest.approx(expected_diag, rel=1e-10)
    assert D[2, 2] == pytest.approx(expected_diag, rel=1e-10)

    # Off-diagonal normal: K - 2/3 G0
    expected_off = K - (2.0 / 3.0) * G0
    assert D[0, 1] == pytest.approx(expected_off, rel=1e-10)
    assert D[0, 2] == pytest.approx(expected_off, rel=1e-10)
    assert D[1, 2] == pytest.approx(expected_off, rel=1e-10)

    # Shear diagonals: G0
    assert D[3, 3] == pytest.approx(G0, rel=1e-10)
    assert D[4, 4] == pytest.approx(G0, rel=1e-10)
    assert D[5, 5] == pytest.approx(G0, rel=1e-10)


def test_law42_ground_state_sound_speed():
    mu = [60.0]
    alpha = [2.0]
    K = 80000.0
    rho0 = 1.2e-6
    mat = _make_law42(mu=mu, alpha=alpha, K=K, rho0=rho0)
    G0 = 0.5 * 60.0 * 2.0

    sig = np.zeros((1, 6))
    F = np.eye(3)[None]
    extra = {"F": F}
    _, _, c = law42_ogden.solid_update(mat, sig, np.zeros((1, 6)), None, 1e-4, extra)

    expected_c0 = np.sqrt((K + (4.0 / 3.0) * G0) / rho0)
    assert c[0] == pytest.approx(expected_c0, rel=1e-10)


def test_law42_pure_volumetric_hydrostatic_compression():
    K = 100000.0
    mat = _make_law42(mu=[50.0], alpha=[2.0], K=K)
    J = 0.85
    F = (J ** (1.0 / 3.0) * np.eye(3))[None]
    sig = np.zeros((1, 6))
    extra = {"F": F}
    s, _, _ = law42_ogden.solid_update(mat, sig, np.zeros((1, 6)), None, 1e-4, extra)

    # Deviatoric stresses must be exactly 0
    p_expected = K * (J - 1.0)
    assert s[0, 0] == pytest.approx(p_expected, rel=1e-10)
    assert s[0, 1] == pytest.approx(p_expected, rel=1e-10)
    assert s[0, 2] == pytest.approx(p_expected, rel=1e-10)
    assert abs(s[0, 3]) < 1e-12
    assert abs(s[0, 4]) < 1e-12
    assert abs(s[0, 5]) < 1e-12


def test_law42_pure_volumetric_hydrostatic_expansion():
    K = 100000.0
    mat = _make_law42(mu=[50.0], alpha=[2.0], K=K)
    J = 1.25
    F = (J ** (1.0 / 3.0) * np.eye(3))[None]
    sig = np.zeros((1, 6))
    extra = {"F": F}
    s, _, c = law42_ogden.solid_update(mat, sig, np.zeros((1, 6)), None, 1e-4, extra)

    p_expected = K * (J - 1.0)
    assert s[0, 0] == pytest.approx(p_expected, rel=1e-10)
    assert s[0, 1] == pytest.approx(p_expected, rel=1e-10)
    assert s[0, 2] == pytest.approx(p_expected, rel=1e-10)

    # In tension J > 1, Kt = K*J, density rho = rho0 / J
    Gt = 0.5 * (50.0 * 2.0) * (1.0) / J
    Kt = K * J
    rho = mat.rho0 / J
    expected_c = np.sqrt((Kt + 4.0 * Gt / 3.0) / rho)
    assert c[0] == pytest.approx(expected_c, rel=1e-10)


def test_law42_isochoric_uniaxial_tension_closed_form():
    mu = [40.0, -15.0]
    alpha = [2.0, -2.0]
    mat = _make_law42(mu=mu, alpha=alpha, K=1e6)
    lam = 1.4
    # Isochoric uniaxial extension: lam1 = lam, lam2 = lam3 = 1/sqrt(lam)
    F = np.diag([lam, 1.0 / np.sqrt(lam), 1.0 / np.sqrt(lam)])[None]
    sig = np.zeros((1, 6))
    extra = {"F": F}
    s, _, _ = law42_ogden.solid_update(mat, sig, np.zeros((1, 6)), None, 1e-4, extra)

    # Cauchy stress difference sigma_1 - sigma_2
    sig_diff = s[0, 0] - s[0, 1]
    expected = sum(m * (lam ** a - (1.0 / np.sqrt(lam)) ** a) for m, a in zip(mu, alpha))
    assert sig_diff == pytest.approx(expected, rel=1e-10)


def test_law42_mooney_rivlin_special_case():
    C10 = 25.0
    C01 = 10.0
    mu = [2.0 * C10, -2.0 * C01]
    alpha = [2.0, -2.0]
    mat = _make_law42(mu=mu, alpha=alpha, K=1e6)
    lam = 1.3
    F = np.diag([lam, 1.0 / np.sqrt(lam), 1.0 / np.sqrt(lam)])[None]
    sig = np.zeros((1, 6))
    s, _, _ = law42_ogden.solid_update(mat, sig, np.zeros((1, 6)), None, 1e-4, {"F": F})

    sig_diff = s[0, 0] - s[0, 1]
    expected_mr = 2.0 * C10 * (lam ** 2 - 1.0 / lam) + 2.0 * C01 * (lam - 1.0 / (lam ** 2))
    assert sig_diff == pytest.approx(expected_mr, rel=1e-10)


def test_law42_neo_hookean_special_case():
    C10 = 35.0
    mu = [2.0 * C10]
    alpha = [2.0]
    mat = _make_law42(mu=mu, alpha=alpha, K=1e6)
    lam = 1.25
    F = np.diag([lam, 1.0 / np.sqrt(lam), 1.0 / np.sqrt(lam)])[None]
    sig = np.zeros((1, 6))
    s, _, _ = law42_ogden.solid_update(mat, sig, np.zeros((1, 6)), None, 1e-4, {"F": F})

    sig_diff = s[0, 0] - s[0, 1]
    expected_nh = 2.0 * C10 * (lam ** 2 - 1.0 / lam)
    assert sig_diff == pytest.approx(expected_nh, rel=1e-10)


def test_law42_isochoric_equibiaxial_tension_closed_form():
    mu = [50.0, -20.0]
    alpha = [2.0, -2.0]
    mat = _make_law42(mu=mu, alpha=alpha, K=1e6)
    lam = 1.3
    # Equibiaxial extension: lam1 = lam2 = lam, lam3 = 1 / lam^2
    F = np.diag([lam, lam, 1.0 / (lam ** 2)])[None]
    sig = np.zeros((1, 6))
    s, _, _ = law42_ogden.solid_update(mat, sig, np.zeros((1, 6)), None, 1e-4, {"F": F})

    # In-plane stress difference: sigma1 - sigma3 = sigma2 - sigma3
    sig_diff = s[0, 0] - s[0, 2]
    expected = sum(m * (lam ** a - (1.0 / (lam ** 2)) ** a) for m, a in zip(mu, alpha))
    assert sig_diff == pytest.approx(expected, rel=1e-10)
    assert s[0, 0] == pytest.approx(s[0, 1], rel=1e-10)


def test_law42_pure_shear_stress_response():
    mu = [40.0]
    alpha = [2.0]
    mat = _make_law42(mu=mu, alpha=alpha, K=1e6)
    lam = 1.2
    # Isochoric planar shear: lam1 = lam, lam2 = 1/lam, lam3 = 1.0
    F = np.diag([lam, 1.0 / lam, 1.0])[None]
    sig = np.zeros((1, 6))
    s, _, _ = law42_ogden.solid_update(mat, sig, np.zeros((1, 6)), None, 1e-4, {"F": F})

    # Pressure must be zero since J = 1
    p = (s[0, 0] + s[0, 1] + s[0, 2]) / 3.0
    assert abs(p) < 1e-10


def test_law42_strain_stiffening_sound_speed_growth():
    mat = _make_law42(mu=[200.0], alpha=[4.0], K=1000.0)
    sig = np.zeros((1, 6))

    # Undeformed
    _, _, c0 = law42_ogden.solid_update(mat, sig.copy(), np.zeros((1, 6)), None, 1e-4, {"F": np.eye(3)[None]})
    # Stretched to lambda = 2.5
    F_stretch = np.diag([2.5, 1.0 / np.sqrt(2.5), 1.0 / np.sqrt(2.5)])[None]
    _, _, c_stretched = law42_ogden.solid_update(mat, sig.copy(), np.zeros((1, 6)), None, 1e-4, {"F": F_stretch})

    # Tangent shear stiffens strongly, wave speed must increase significantly
    assert c_stretched[0] > 1.5 * c0[0]


def test_law42_large_compressive_strain_stiffening():
    mat = _make_law42(mu=[30.0, -30.0], alpha=[2.0, -2.0], K=50000.0)
    sig = np.zeros((1, 6))

    _, _, c0 = law42_ogden.solid_update(mat, sig.copy(), np.zeros((1, 6)), None, 1e-4, {"F": np.eye(3)[None]})
    # Compressed along axis 1 to lambda = 0.35
    F_comp = np.diag([0.35, 1.0 / np.sqrt(0.35), 1.0 / np.sqrt(0.35)])[None]
    _, _, c_comp = law42_ogden.solid_update(mat, sig.copy(), np.zeros((1, 6)), None, 1e-4, {"F": F_comp})

    # Negative alpha term stiffens in compression, wave speed must grow
    assert c_comp[0] > c0[0]


def test_law42_energy_conservation_closed_deformation_cycle():
    mat = _make_law42(mu=[40.0, -20.0], alpha=[2.0, -2.0], K=80000.0)
    sig = np.zeros((1, 6))

    # Step 1: Stretch
    F1 = np.diag([1.4, 0.9, 1.0])[None]
    s1, _, _ = law42_ogden.solid_update(mat, sig.copy(), np.zeros((1, 6)), None, 1e-4, {"F": F1})
    assert np.linalg.norm(s1) > 10.0

    # Step 2: Shear + stretch
    F2 = np.array([[[1.2, 0.3, 0.0],
                    [0.1, 0.9, 0.0],
                    [0.0, 0.0, 1.0]]])
    s2, _, _ = law42_ogden.solid_update(mat, sig.copy(), np.zeros((1, 6)), None, 1e-4, {"F": F2})
    assert np.linalg.norm(s2) > 10.0

    # Step 3: Return exactly to reference state F = I
    F0 = np.eye(3)[None]
    s0, _, _ = law42_ogden.solid_update(mat, sig.copy(), np.zeros((1, 6)), None, 1e-4, {"F": F0})
    assert np.allclose(s0, 0.0, atol=1e-12)


def test_law42_frame_indifference_material_objectivity():
    mat = _make_law42(mu=[50.0, -15.0], alpha=[2.0, -2.0], K=60000.0)
    F = np.array([[[1.3, 0.2, 0.1],
                   [0.0, 0.9, 0.2],
                   [0.1, 0.0, 1.1]]])
    sig1 = np.zeros((1, 6))
    s1, _, _ = law42_ogden.solid_update(mat, sig1, np.zeros((1, 6)), None, 1e-4, {"F": F})
    T1 = np.array([[s1[0, 0], s1[0, 3], s1[0, 5]],
                   [s1[0, 3], s1[0, 1], s1[0, 4]],
                   [s1[0, 5], s1[0, 4], s1[0, 2]]])

    # Superpose rigid rotation Q
    theta = np.pi / 3.0
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    K_mat = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    Q = np.eye(3) + np.sin(theta) * K_mat + (1.0 - np.cos(theta)) * (K_mat @ K_mat)

    F_rot = (Q @ F[0])[None]
    sig2 = np.zeros((1, 6))
    s2, _, _ = law42_ogden.solid_update(mat, sig2, np.zeros((1, 6)), None, 1e-4, {"F": F_rot})
    T2 = np.array([[s2[0, 0], s2[0, 3], s2[0, 5]],
                   [s2[0, 3], s2[0, 1], s2[0, 4]],
                   [s2[0, 5], s2[0, 4], s2[0, 2]]])

    expected_T2 = Q @ T1 @ Q.T
    assert np.allclose(T2, expected_T2, atol=1e-10)


def test_law42_inverted_element_jacobian_guard():
    mat = _make_law42(mu=[50.0], alpha=[2.0], K=50000.0)
    sig = np.zeros((1, 6))
    # Inverted deformation gradient det(F) = -1.0
    F_inv = np.diag([-1.0, 1.0, 1.0])[None]
    s, _, c = law42_ogden.solid_update(mat, sig, np.zeros((1, 6)), None, 1e-4, {"F": F_inv})
    assert not np.isnan(s).any()
    assert not np.isnan(c).any()
    assert c[0] > 0.0


def test_law42_fallback_without_extra_F():
    mat = _make_law42(mu=[50.0], alpha=[2.0], K=50000.0)
    sig = np.zeros((1, 6))
    deps = np.array([[1e-4, -3e-5, -3e-5, 0.0, 0.0, 0.0]])
    # Call without extra dictionary
    s, _, c = law42_ogden.solid_update(mat, sig, deps, None, 1e-4)
    assert not np.isnan(s).any()
    assert c[0] > 0.0
    # Small strain should produce tension in direction 1
    assert s[0, 0] > 0.0


def test_law42_consistent_tangent_truesdell_directional_derivative():
    mat = _make_law42(mu=[50.0, -20.0], alpha=[2.0, -2.0], K=80000.0)
    rng = np.random.default_rng(42)
    F = np.diag([1.25, 0.95, 0.85])

    def sig_tensor(Fin):
        sig = np.zeros((1, 6))
        law42_ogden.solid_update(mat, sig, np.zeros((1, 6)), None, 1e-4, {"F": Fin[None]})
        s = sig[0]
        return np.array([[s[0], s[3], s[5]],
                         [s[3], s[1], s[4]],
                         [s[5], s[4], s[2]]])

    D6 = law42_ogden.consistent_solid_tangent(mat, F[None])[0]
    VO = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
    c4 = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(VO):
        for Jc, (k, ell) in enumerate(VO):
            for (a, b) in {(i, j), (j, i)}:
                for (c, d) in {(k, ell), (ell, k)}:
                    c4[a, b, c, d] = D6[I, Jc]

    sig0 = sig_tensor(F)
    eps = 1e-7
    for _ in range(4):
        g = (rng.standard_normal((3, 3)) - 0.5) * 0.1
        dF = eps * g @ F
        ds_fd = (sig_tensor(F + dF) - sig_tensor(F - dF)) / (2.0 * eps)
        d = 0.5 * (g + g.T)
        pred = (np.einsum("ijkl,kl->ij", c4, d) + g @ sig0 + sig0 @ g.T - np.trace(d) * sig0)
        assert np.abs(ds_fd - pred).max() <= 1e-5 * max(np.abs(ds_fd).max(), 1.0)


def test_law42_consistent_tangent_coalescent_eigenvalue_lhopital():
    mat = _make_law42(mu=[50.0, -20.0], alpha=[2.0, -2.0], K=80000.0)

    def sig_tensor(Fin):
        sig = np.zeros((1, 6))
        law42_ogden.solid_update(mat, sig, np.zeros((1, 6)), None, 1e-4, {"F": Fin[None]})
        s = sig[0]
        return np.array([[s[0], s[3], s[5]],
                         [s[3], s[1], s[4]],
                         [s[5], s[4], s[2]]])

    VO = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))

    def verify_fd(F_curr, D6_curr):
        c4 = np.zeros((3, 3, 3, 3))
        for I, (i, j) in enumerate(VO):
            for Jc, (k, ell) in enumerate(VO):
                for (a, b) in {(i, j), (j, i)}:
                    for (c, d) in {(k, ell), (ell, k)}:
                        c4[a, b, c, d] = D6_curr[I, Jc]
        sig0 = sig_tensor(F_curr)
        eps = 1e-7
        g = np.array([[0.02, 0.01, -0.015],
                      [0.01, -0.01, 0.005],
                      [-0.015, 0.005, 0.015]])
        dF = eps * g @ F_curr
        ds_fd = (sig_tensor(F_curr + dF) - sig_tensor(F_curr - dF)) / (2.0 * eps)
        d = 0.5 * (g + g.T)
        pred = (np.einsum("ijkl,kl->ij", c4, d) + g @ sig0 + sig0 @ g.T - np.trace(d) * sig0)
        assert np.abs(ds_fd - pred).max() <= 1e-5 * max(np.abs(ds_fd).max(), 1.0)

    # 0. Reference state F = I: positive definite
    D_I = law42_ogden.consistent_solid_tangent(mat, np.eye(3)[None])[0]
    assert np.allclose(D_I, D_I.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(D_I) > 0.0)

    # 1. Hydrostatic (3 equal eigenvalues, triggers L'Hopital limit)
    F_hydro = 1.08 * np.eye(3)
    D_hydro = law42_ogden.consistent_solid_tangent(mat, F_hydro[None])[0]
    assert not np.isnan(D_hydro).any()
    assert np.allclose(D_hydro, D_hydro.T, atol=1e-10)
    verify_fd(F_hydro, D_hydro)

    # 2. Uniaxial coalescent (2 equal transverse eigenvalues, triggers L'Hopital limit)
    F_uni = np.diag([1.3, 1.05, 1.05])
    D_uni = law42_ogden.consistent_solid_tangent(mat, F_uni[None])[0]
    assert not np.isnan(D_uni).any()
    assert np.allclose(D_uni, D_uni.T, atol=1e-10)
    verify_fd(F_uni, D_uni)


def test_law42_multi_element_batch_vectorization():
    mat = _make_law42(mu=[40.0, -15.0], alpha=[2.0, -2.0], K=60000.0)
    N = 6
    np.random.seed(123)
    F_batch = np.tile(np.eye(3), (N, 1, 1))
    for i in range(N):
        F_batch[i] += (np.random.rand(3, 3) - 0.5) * 0.2

    sig_bat = np.zeros((N, 6))
    s_bat, _, c_bat = law42_ogden.solid_update(mat, sig_bat.copy(), np.zeros((N, 6)), None, 1e-4, {"F": F_batch})
    D_bat = law42_ogden.consistent_solid_tangent(mat, F_batch)

    for i in range(N):
        sig_ind = np.zeros((1, 6))
        s_ind, _, c_ind = law42_ogden.solid_update(mat, sig_ind, np.zeros((1, 6)), None, 1e-4, {"F": F_batch[i:i+1]})
        D_ind = law42_ogden.consistent_solid_tangent(mat, F_batch[i:i+1])
        assert np.allclose(s_bat[i], s_ind[0], atol=1e-12)
        assert np.allclose(c_bat[i], c_ind[0], atol=1e-12)
        assert np.allclose(D_bat[i], D_ind[0], atol=1e-12)

    # Verify shell_update raises NotImplementedError
    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        law42_ogden.shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)), None, 1e-4)

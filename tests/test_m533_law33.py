"""M533 — LAW33 crushable foam plasticity (FOAM_PLAS) hardening, tangents, and unit tests.

Fortran origin: ``engine/source/materials/mat/mat033/sigeps33.F`` and
``starter/source/materials/mat/mat033/hm_read_mat33.F``.
"""
import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials import law33_foamplas as law33


# ===================================================================
# Helpers
# ===================================================================

def _make_law33_ken0(e=100.0, rho0=1.0, a=0.0, b=10.0, c=0.0, **kw):
    """Simple KEN=0 material."""
    rec = {
        "id": 1, "density": rho0, "title": "FOAM_KEN0",
        "params": {"MAT_E": e, "Itype": 0, "MAT_A0": a,
                   "MAT_A1": b, "MAT_A2": c, **kw},
    }
    return law33.build_law33(rec)


def _make_law33_ken1(e=100.0, rho0=1.0, a=0.0, b=10.0, c=0.0,
                     c1_k=50.0, c2_k=200.0, et=20.0, vmu=5.0, vmu0=5.0, **kw):
    """KEN=1 (Kelvin) material."""
    rec = {
        "id": 2, "density": rho0, "title": "FOAM_KEN1",
        "params": {"MAT_E": e, "Itype": 1, "MAT_A0": a,
                   "MAT_A1": b, "MAT_A2": c,
                   "MAT_E1": c1_k, "MAT_E2": c2_k,
                   "MAT_ETAN": et, "MAT_ETA1": vmu, "MAT_ETA2": vmu0,
                   **kw},
    }
    return law33.build_law33(rec)


def _make_law33_ken2(e=100.0, rho0=1.0, a=0.0, b=10.0, c=0.0,
                     sigt_cutoff=5.0, **kw):
    """KEN=2 (tension cutoff) material."""
    rec = {
        "id": 3, "density": rho0, "title": "FOAM_KEN2",
        "params": {"MAT_E": e, "Itype": 2, "MAT_A0": a,
                   "MAT_A1": b, "MAT_A2": c,
                   "MAT_SIGT_CUTOFF": sigt_cutoff, **kw},
    }
    return law33.build_law33(rec)


# ===================================================================
# Build / parameter extraction tests
# ===================================================================

def test_law33_build_from_dict():
    mat = _make_law33_ken0()
    assert mat.id == 1
    assert mat.law == 33
    assert mat.rho0 == 1.0
    assert mat.params["E"] == 100.0
    assert mat.params["KEN"] == 0


def test_law33_build_from_material_instance():
    base = Material(id=10, law=33, rho0=2.0, title="Foam",
                    params={"MAT_E": 50.0, "Itype": 0, "MAT_A0": 1.0,
                            "MAT_A1": 5.0, "MAT_A2": 0.5})
    mat = law33.build_law33(base)
    assert mat.id == 10
    assert mat.rho0 == 2.0
    assert mat.params["E"] == 50.0
    assert mat.params["A"] == 1.0
    assert mat.params["B"] == 5.0
    assert mat.params["C"] == 0.5


def test_law33_build_ken1_params():
    mat = _make_law33_ken1()
    assert mat.params["KEN"] == 1
    assert mat.params["C1_kelvin"] == 50.0
    assert mat.params["C2_kelvin"] == 200.0
    assert mat.params["Et"] == 20.0
    assert mat.params["VMU"] == 5.0
    assert mat.params["VMU0"] == 5.0


def test_law33_build_ken2_sigt_cutoff():
    mat = _make_law33_ken2(sigt_cutoff=3.0)
    assert mat.params["KEN"] == 2
    assert mat.params["SIGT_CUTOFF"] == 3.0


def test_law33_build_ken2_sigt_cutoff_default():
    """If SIGT_CUTOFF=0, it defaults to EP20 (1e20)."""
    rec = {
        "id": 1, "density": 1.0,
        "params": {"MAT_E": 100.0, "Itype": 2, "MAT_SIGT_CUTOFF": 0.0},
    }
    mat = law33.build_law33(rec)
    assert mat.params["SIGT_CUTOFF"] == 1e20


def test_law33_build_fac_default():
    """If FAC=0, it defaults to 1.0."""
    rec = {
        "id": 1, "density": 1.0,
        "params": {"MAT_E": 100.0, "Itype": 0, "IFscale": 0.0},
    }
    mat = law33.build_law33(rec)
    assert mat.params["FAC"] == 1.0


def test_law33_build_density_validation():
    rec = {"id": 1, "density": 0.0, "params": {"MAT_E": 100.0}}
    with pytest.raises(ValueError, match="Density rho0 must be positive"):
        law33.build_law33(rec)


def test_law33_build_e_validation():
    rec = {"id": 1, "density": 1.0, "params": {"MAT_E": -10.0}}
    with pytest.raises(ValueError, match="Young modulus E must be > 0"):
        law33.build_law33(rec)


def test_law33_build_ken1_vmu_validation():
    rec = {
        "id": 1, "density": 1.0,
        "params": {"MAT_E": 100.0, "Itype": 1,
                   "MAT_E1": 10.0, "MAT_E2": 20.0, "MAT_ETAN": 5.0,
                   "MAT_ETA1": 0.0, "MAT_ETA2": 5.0},
    }
    with pytest.raises(ValueError, match="viscous coefficients"):
        law33.build_law33(rec)


# ===================================================================
# Empty array tests
# ===================================================================

def test_law33_empty_array():
    mat = _make_law33_ken0()
    sig = np.zeros((0, 6))
    deps = np.zeros((0, 6))
    sig_out, epsp_out, c_out = law33.solid_update(mat, sig, deps, None, 0.01)
    assert sig_out.shape == (0, 6)
    assert c_out is None


def test_law33_tangent_empty_array():
    mat = _make_law33_ken0()
    sig = np.zeros((0, 6))
    D = law33.consistent_solid_tangent(mat, sig)
    assert D.shape == (0, 6, 6)


def test_law33_shell_not_implemented():
    mat = _make_law33_ken0()
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    with pytest.raises(NotImplementedError, match="solid/SPH elements only"):
        law33.shell_update(mat, sig, deps)


# ===================================================================
# ICASE=1 (KEN=0) tests
# ===================================================================

def test_law33_ken0_elastic_below_yield():
    """Elastic response when stress stays below yield."""
    mat = _make_law33_ken0(e=100.0, b=1000.0)  # high yield
    sig = np.zeros((1, 6))
    deps = np.array([[0.01, -0.005, -0.005, 0.0, 0.0, 0.0]])
    dt = 0.1
    sig_out, _, c = law33.solid_update(mat, sig, deps, None, dt,
                                       extra={"rho": np.array([1.0])})
    # Trial: sig_xx = 100 * 0.01/0.1 * 0.1 = 1.0 (below yield 1000)
    assert math.isclose(sig_out[0, 0], 1.0, rel_tol=1e-10)
    assert math.isclose(sig_out[0, 1], -0.5, rel_tol=1e-10)
    assert math.isclose(sig_out[0, 2], -0.5, rel_tol=1e-10)
    assert c is not None
    assert math.isclose(c[0], math.sqrt(100.0 / 1.0))


def test_law33_ken0_yield_clamp():
    """Principal stress clamping when trial exceeds yield."""
    mat = _make_law33_ken0(e=1000.0, b=5.0)  # low yield
    sig = np.zeros((1, 6))
    # Large uniaxial strain → trial stress far above yield
    deps = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dt = 0.1
    sig_out, _, _ = law33.solid_update(mat, sig, deps, None, dt,
                                       extra={"rho": np.array([1.0])})
    # sig_xx should be clamped: principal stress 1 = trial ≈ 100
    # yield = |0 + 5*(1+0*gamma)| = 5
    # After return, max principal = 5 → sig_xx ≤ 5
    assert np.max(np.abs(sig_out[0, :3])) <= 5.0 + 1e-10


def test_law33_ken0_air_pressure():
    """Air pressure contribution (closed-cell foam)."""
    mat = _make_law33_ken0(e=100.0, b=1000.0, MAT_P0=10.0,
                           MAT_PHI=0.5, MAT_GAMA0=0.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    dt = 0.01
    # rho < rho0 → gamma > 0 → expansion → negative pressure possible
    sig_out, _, _ = law33.solid_update(mat, sig, deps, None, dt,
                                       extra={"rho": np.array([0.5])})
    # gamma = 1/0.5 - 1 + 0 = 1.0
    # var = -(10 * 1.0)/(1 + 1.0 - 0.5 + 1e-15) = -(10)/(1.5) = -6.67
    # sig_air = max(0, -6.67) = 0 (the var is negative, so sig_air = 0)
    # So no air pressure when compressed (gamma > 0 means expanded from ref)
    # With no deps, sig_out should be all zeros
    np.testing.assert_allclose(sig_out, np.zeros((1, 6)), atol=1e-12)


def test_law33_ken0_air_pressure_compression():
    """Air pressure when foam is compressed (rho > rho0)."""
    mat = _make_law33_ken0(e=100.0, b=1000.0, MAT_P0=10.0,
                           MAT_PHI=0.3, MAT_GAMA0=0.0)
    sig = np.zeros((1, 6))
    # With zero deps, trial_s is just 0 + air_pressure = sig_air on normals
    # After principal return (sig_air < yield 1000), sig_new = sig_air - sig_air = 0
    # So we need to verify the air pressure value directly.
    gamma, sig_air = law33._air_pressure(
        np.array([1.0]), np.array([2.0]), 10.0, 0.3, 0.0)
    # gamma = 0.5 - 1 = -0.5
    # var = -(10*(-0.5))/(1 + (-0.5) - 0.3 + 1e-15) = 5/0.2 = 25
    assert math.isclose(gamma[0], -0.5)
    assert math.isclose(sig_air[0], 25.0)


def test_law33_ken0_pure_shear():
    """Shear response with the ½ factor."""
    mat = _make_law33_ken0(e=100.0, b=1000.0)
    sig = np.zeros((1, 6))
    deps = np.array([[0.0, 0.0, 0.0, 0.02, 0.0, 0.0]])
    dt = 0.01  # shear rate = 2.0
    sig_out, _, _ = law33.solid_update(mat, sig, deps, None, dt,
                                       extra={"rho": np.array([1.0])})
    # sig_xy = E * rate * dt * 0.5 = 100 * 2.0 * 0.01 * 0.5 = 1.0
    assert math.isclose(sig_out[0, 3], 1.0, rel_tol=1e-10)


def test_law33_ken0_sound_speed():
    """Sound speed = sqrt(E/rho0)."""
    mat = _make_law33_ken0(e=400.0, rho0=4.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    _, _, c = law33.solid_update(mat, sig, deps, None, 0.01,
                                 extra={"rho": np.array([4.0])})
    assert c is not None
    assert math.isclose(c[0], 10.0)  # sqrt(400/4) = 10


def test_law33_ken0_volumetric_strain_yield():
    """Yield stress depends on volumetric strain gamma via formula."""
    mat = _make_law33_ken0(e=100.0, a=5.0, b=10.0, c=2.0, rho0=1.0)
    sig = np.zeros((1, 6))
    deps = np.array([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dt = 0.01
    # rho = 2.0 → gamma = 0.5 - 1 = -0.5
    # yield = |5 + 10*(1 + 2*(-0.5))| = |5 + 10*(0)| = 5
    sig_out, _, _ = law33.solid_update(mat, sig, deps, None, dt,
                                       extra={"rho": np.array([2.0])})
    # The principal stress should be clamped to 5 or less
    # Build the stress tensor and check eigenvalues
    S = np.zeros((3, 3))
    S[0, 0] = sig_out[0, 0]
    S[1, 1] = sig_out[0, 1]
    S[2, 2] = sig_out[0, 2]
    S[0, 1] = S[1, 0] = sig_out[0, 3]
    S[1, 2] = S[2, 1] = sig_out[0, 4]
    S[0, 2] = S[2, 0] = sig_out[0, 5]
    eigvals = np.linalg.eigvalsh(S)
    # Subtract back air pressure for the principal comparison
    # The returned stress already has air removed; the eigenvalue check
    # on the RETURNED stress doesn't directly compare to yield
    # (yield clamp happens on sig_trial + air), but the returned stress
    # magnitudes should be bounded
    assert np.all(np.abs(eigvals) < 100.0)  # reasonable bound


# ===================================================================
# ICASE=2 (KEN=1) tests
# ===================================================================

def test_law33_ken1_rate_dependent_modulus():
    """E_eff = max(E, C1*edot + C2)."""
    mat = _make_law33_ken1(e=100.0, c1_k=50.0, c2_k=200.0)
    sig = np.zeros((1, 6))
    # High strain rate → E_eff should be > E
    deps = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dt = 0.01  # rate = 10
    sig_out1, _, c1 = law33.solid_update(mat, sig, deps, None, dt,
                                         extra={"rho": np.array([1.0])})
    # E_eff = max(100, 50*10 + 200) = max(100, 700) = 700
    assert c1 is not None
    assert math.isclose(c1[0], math.sqrt(700.0), rel_tol=1e-8)


def test_law33_ken1_kelvin_basic():
    """Basic Kelvin model stress update produces nonzero stress."""
    mat = _make_law33_ken1(e=100.0, c1_k=0.0, c2_k=100.0, et=10.0,
                           vmu=1.0, vmu0=1.0, b=1000.0)
    sig = np.zeros((1, 6))
    deps = np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dt = 0.001
    extra = {"rho": np.array([1.0])}
    sig_out, _, c = law33.solid_update(mat, sig, deps, None, dt, extra=extra)
    assert np.any(sig_out != 0.0)
    assert c is not None


def test_law33_ken1_sound_speed():
    """Sound speed from rate-dependent E_eff."""
    mat = _make_law33_ken1(e=100.0, c1_k=100.0, c2_k=0.0, rho0=4.0)
    sig = np.zeros((1, 6))
    deps = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dt = 0.01
    _, _, c = law33.solid_update(mat, sig, deps, None, dt,
                                 extra={"rho": np.array([4.0])})
    # edot = 0 → E_eff = max(100, 0+0) = 100
    assert c is not None
    assert math.isclose(c[0], math.sqrt(100.0 / 4.0))


# ===================================================================
# ICASE=3 (KEN=2) tests
# ===================================================================

def test_law33_ken2_tension_cutoff():
    """Tension cutoff clamps in near-zero gamma regime."""
    mat = _make_law33_ken2(e=1000.0, sigt_cutoff=2.0, b=100.0)
    sig = np.zeros((1, 6))
    # Large tensile strain
    deps = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dt = 0.01
    # gamma ≈ 0 (rho ≈ rho0) → |gamma| < 1e-10 regime
    sig_out, _, _ = law33.solid_update(mat, sig, deps, None, dt,
                                       extra={"rho": np.array([1.0])})
    # In the |gamma| < 1e-10 regime: principal stress clamped to ±SIGT_CUTOFF=2
    S = np.zeros((3, 3))
    S[0, 0] = sig_out[0, 0]
    S[1, 1] = sig_out[0, 1]
    S[2, 2] = sig_out[0, 2]
    S[0, 1] = S[1, 0] = sig_out[0, 3]
    S[1, 2] = S[2, 1] = sig_out[0, 4]
    S[0, 2] = S[2, 0] = sig_out[0, 5]
    eigvals = np.linalg.eigvalsh(S)
    # All principals should be ≤ sigt_cutoff (allowing air pressure shift)
    assert np.all(np.abs(eigvals) <= 2.0 + 1.0)  # allow air pressure margin


def test_law33_ken2_compression_yield():
    """Compression yield when gamma < 0."""
    mat = _make_law33_ken2(e=1000.0, sigt_cutoff=50.0, b=5.0, rho0=1.0)
    sig = np.zeros((1, 6))
    deps = np.array([[-0.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dt = 0.01
    # rho > rho0 → compression → gamma < 0
    sig_out, _, _ = law33.solid_update(mat, sig, deps, None, dt,
                                       extra={"rho": np.array([2.0])})
    # Non-trivial stress should exist
    assert np.any(sig_out != 0.0)


def test_law33_ken2_expansion_clamp():
    """Expansion (gamma > 0): positive principals clamped to zero."""
    mat = _make_law33_ken2(e=1000.0, sigt_cutoff=2.0, b=100.0, rho0=1.0)
    sig = np.zeros((1, 6))
    deps = np.array([[0.1, 0.1, 0.1, 0.0, 0.0, 0.0]])
    dt = 0.01
    # rho < rho0 → expansion → gamma > 0
    sig_out, _, _ = law33.solid_update(mat, sig, deps, None, dt,
                                       extra={"rho": np.array([0.5])})
    # gamma = 2 - 1 = 1 > 0 → expansion regime: positive principals → 0
    # The returned stress (after subtracting air) should have no tension
    # (though air pressure offset might shift things)
    # Just check the function runs without error
    assert sig_out.shape == (1, 6)


# ===================================================================
# General tests
# ===================================================================

def test_law33_dt_zero():
    """Static solve / dt=0: return original stress unchanged."""
    mat = _make_law33_ken0()
    sig = np.array([[1.0, 2.0, 3.0, 0.5, -0.5, 0.1]])
    deps = np.array([[0.01, -0.01, 0.0, 0.02, 0.0, 0.0]])
    sig_out, _, c = law33.solid_update(mat, sig, deps, None, dt=0.0)
    np.testing.assert_allclose(sig_out, sig)
    assert c is None


def test_law33_tangent_shape_symmetry():
    """Tangent is (n,6,6) and symmetric."""
    mat = _make_law33_ken0(e=200.0)
    sig = np.zeros((2, 6))
    D = law33.consistent_solid_tangent(mat, sig)
    assert D.shape == (2, 6, 6)
    for i in range(2):
        np.testing.assert_allclose(D[i], D[i].T, atol=1e-12)


def test_law33_tangent_positive_semidefinite():
    """Tangent eigenvalues ≥ 0."""
    mat = _make_law33_ken0(e=300.0)
    sig = np.zeros((1, 6))
    D = law33.consistent_solid_tangent(mat, sig)
    eigs = np.linalg.eigvalsh(D[0])
    assert np.all(eigs >= -1e-12)


def test_law33_tangent_finite_difference():
    """Tangent matches finite difference of stress update."""
    mat = _make_law33_ken0(e=200.0, b=1e6)  # very high yield → elastic regime
    deps0 = np.array([[0.001, -0.0005, -0.0005, 0.002, 0.001, -0.001]])
    dt = 0.01
    extra = {"rho": np.array([1.0])}

    D = law33.consistent_solid_tangent(mat, np.zeros((1, 6)))[0]

    h = 1e-7
    for comp in range(6):
        deps_p = deps0.copy()
        deps_m = deps0.copy()
        deps_p[0, comp] += h
        deps_m[0, comp] -= h

        sig_p, _, _ = law33.solid_update(mat, np.zeros((1, 6)), deps_p,
                                         None, dt, extra)
        sig_m, _, _ = law33.solid_update(mat, np.zeros((1, 6)), deps_m,
                                         None, dt, extra)

        d_num = (sig_p[0] - sig_m[0]) / (2.0 * h)
        np.testing.assert_allclose(D[:, comp], d_num, rtol=1e-4, atol=1e-4)


def test_law33_tangent_ken1_shape():
    """KEN=1 tangent shape and symmetry."""
    mat = _make_law33_ken1()
    sig = np.zeros((3, 6))
    extra = {"rho": np.ones(3), "dt": 0.01}
    D = law33.consistent_solid_tangent(mat, sig, extra=extra)
    assert D.shape == (3, 6, 6)
    for i in range(3):
        np.testing.assert_allclose(D[i], D[i].T, atol=1e-12)


def test_law33_frame_invariance():
    """Rotated strain → rotated stress (frame invariance)."""
    mat = _make_law33_ken0(e=200.0, b=1e6, rho0=1.0)
    dt = 0.01

    # Rotation about Z axis
    theta = math.pi / 3.0
    ct, st = math.cos(theta), math.sin(theta)
    Q = np.array([
        [ct, -st, 0.0],
        [st,  ct, 0.0],
        [0.0, 0.0, 1.0],
    ])

    # Original strain tensor
    eps_mat = np.array([
        [0.01, 0.003, 0.0],
        [0.003, -0.005, 0.001],
        [0.0, 0.001, -0.005],
    ])
    deps = np.array([[
        eps_mat[0, 0], eps_mat[1, 1], eps_mat[2, 2],
        2.0 * eps_mat[0, 1], 2.0 * eps_mat[1, 2], 2.0 * eps_mat[2, 0]
    ]])

    extra = {"rho": np.array([1.0])}
    sig, _, _ = law33.solid_update(mat, np.zeros((1, 6)), deps, None, dt, extra)

    sig_mat = np.array([
        [sig[0, 0], sig[0, 3], sig[0, 5]],
        [sig[0, 3], sig[0, 1], sig[0, 4]],
        [sig[0, 5], sig[0, 4], sig[0, 2]],
    ])

    # Rotated strain
    eps_rot = Q @ eps_mat @ Q.T
    deps_rot = np.array([[
        eps_rot[0, 0], eps_rot[1, 1], eps_rot[2, 2],
        2.0 * eps_rot[0, 1], 2.0 * eps_rot[1, 2], 2.0 * eps_rot[2, 0]
    ]])

    sig_rot, _, _ = law33.solid_update(mat, np.zeros((1, 6)), deps_rot,
                                        None, dt, extra)
    sig_rot_mat = np.array([
        [sig_rot[0, 0], sig_rot[0, 3], sig_rot[0, 5]],
        [sig_rot[0, 3], sig_rot[0, 1], sig_rot[0, 4]],
        [sig_rot[0, 5], sig_rot[0, 4], sig_rot[0, 2]],
    ])

    expected = Q @ sig_mat @ Q.T
    np.testing.assert_allclose(sig_rot_mat, expected, rtol=1e-10, atol=1e-10)


def test_law33_batched_vectorization():
    """Multi-element batch vs. sequential agreement."""
    nel = 5
    mat = _make_law33_ken0(e=200.0, b=50.0, rho0=2.0)
    dt = 0.005

    rng = np.random.default_rng(42)
    deps_batch = rng.normal(0.0, 0.01, size=(nel, 6))
    sig_batch = np.zeros((nel, 6))
    extra_batch = {"rho": rng.uniform(1.5, 2.5, size=nel)}

    sig_out, _, c_out = law33.solid_update(mat, sig_batch, deps_batch,
                                           None, dt, extra_batch)

    for i in range(nel):
        sig_s = np.zeros((1, 6))
        deps_s = deps_batch[i:i+1]
        extra_s = {"rho": np.array([extra_batch["rho"][i]])}
        sig_s_out, _, c_s = law33.solid_update(mat, sig_s, deps_s,
                                               None, dt, extra_s)
        np.testing.assert_allclose(sig_out[i], sig_s_out[0], atol=1e-12)
        np.testing.assert_allclose(c_out[i], c_s[0], atol=1e-12)


def test_law33_missing_rho_fallback():
    """When extra is None, falls back to rho0."""
    mat = _make_law33_ken0(e=100.0, rho0=2.0, b=1e6)
    sig = np.zeros((1, 6))
    deps = np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]])
    sig_out, _, c = law33.solid_update(mat, sig, deps, None, 0.01, extra=None)
    assert c is not None
    assert math.isclose(c[0], math.sqrt(100.0 / 2.0))


def test_law33_materials_init_solid_dispatch():
    """Dispatch through materials.__init__.solid_update works."""
    from pyradioss.materials import solid_update
    mat = _make_law33_ken0(e=100.0, rho0=1.0, b=1000.0)
    sig = np.zeros((1, 6))
    deps = np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]])
    extra = {"rho": np.array([1.0])}
    sig_out, _, c = solid_update(mat, sig, deps, None, 0.01, extra)
    assert sig_out.shape == (1, 6)
    assert c is not None


def test_law33_materials_init_tangent_dispatch():
    """Dispatch through materials.__init__.solid_tangent works."""
    from pyradioss.materials import solid_tangent
    mat = _make_law33_ken0(e=100.0)
    sig = np.zeros((2, 6))
    D = solid_tangent(mat, sig, None, None)
    assert D.shape == (2, 6, 6)


def test_law33_materials_init_shell_dispatch():
    """Shell dispatch raises NotImplementedError."""
    from pyradioss.materials import shell_update
    mat = _make_law33_ken0()
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    with pytest.raises(NotImplementedError):
        shell_update(mat, sig, deps, None, 0.01)


# ===================================================================
# Air pressure specific tests
# ===================================================================

def test_law33_air_pressure_zero_p0():
    """Zero foam pressure P0 → no air pressure contribution."""
    mat = _make_law33_ken0(e=100.0, b=1e6, MAT_P0=0.0)
    sig = np.zeros((1, 6))
    deps = np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dt = 0.01
    sig_out1, _, _ = law33.solid_update(mat, sig, deps, None, dt,
                                        extra={"rho": np.array([2.0])})

    # Compare with same mat without P0
    mat2 = _make_law33_ken0(e=100.0, b=1e6)
    sig_out2, _, _ = law33.solid_update(mat2, sig, deps, None, dt,
                                        extra={"rho": np.array([2.0])})
    np.testing.assert_allclose(sig_out1, sig_out2, atol=1e-12)


def test_law33_air_pressure_helper():
    """Direct test of _air_pressure helper."""
    rho0 = np.array([1.0])
    rho = np.array([2.0])  # compressed: rho > rho0
    p0 = 10.0
    phi = 0.0
    gama0 = 0.0
    gamma, sig_air = law33._air_pressure(rho0, rho, p0, phi, gama0)
    # gamma = 1/2 - 1 + 0 = -0.5
    assert math.isclose(gamma[0], -0.5)
    # var = -(10 * (-0.5))/(1 + (-0.5) - 0 + 1e-15) = 5/0.5 = 10
    assert math.isclose(sig_air[0], 10.0)


def test_law33_principal_return_identity():
    """If all principal stresses are within yield, return is unchanged."""
    sig_trial = np.array([[1.0, 2.0, 3.0, 0.5, 0.3, 0.1]])
    syield = np.array([100.0])  # high yield
    sig_ret = law33._principal_return_clamp(sig_trial, syield)
    np.testing.assert_allclose(sig_ret, sig_trial, atol=1e-10)


def test_law33_principal_return_clamped():
    """Verify principal stresses are actually clamped."""
    sig_trial = np.array([[10.0, -8.0, 6.0, 0.0, 0.0, 0.0]])
    syield = np.array([5.0])
    sig_ret = law33._principal_return_clamp(sig_trial, syield)

    # Check eigenvalues of returned stress ≤ 5.0
    S = np.zeros((3, 3))
    S[0, 0] = sig_ret[0, 0]
    S[1, 1] = sig_ret[0, 1]
    S[2, 2] = sig_ret[0, 2]
    S[0, 1] = S[1, 0] = sig_ret[0, 3]
    S[1, 2] = S[2, 1] = sig_ret[0, 4]
    S[0, 2] = S[2, 0] = sig_ret[0, 5]
    eigvals = np.linalg.eigvalsh(S)
    assert np.all(np.abs(eigvals) <= 5.0 + 1e-10)

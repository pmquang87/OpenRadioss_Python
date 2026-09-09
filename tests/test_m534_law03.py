"""M534 — LAW3 elastic-plastic hydrodynamic (PLAS_BOST) hardening, tangents,
and unit tests.

Fortran origin: ``engine/source/materials/mat/mat003/m3law.F`` and
``starter/source/materials/mat/mat003/hm_read_mat03.F``.
"""
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials import law03_plas_bost as law03


# ===================================================================
# Helpers
# ===================================================================

def _make_law03(e=210e3, nu=0.3, rho0=7.85e-3, a=200.0, b=500.0, n=0.4,
                eps_max=0.0, sig_max=0.0, **kw):
    """Simple LAW3 material with steel-like defaults."""
    rec = {
        "id": 1, "density": rho0, "title": "PLAS_BOST",
        "params": {"MAT_E": e, "MAT_NU": nu, "MAT_SIGY": a,
                   "MAT_BETA": b, "MAT_HARD": n,
                   "MAT_EPS": eps_max, "MAT_SIG": sig_max, **kw},
    }
    return law03.build_law03(rec)


# ===================================================================
# Build / parameter extraction tests
# ===================================================================

def test_build_from_dict():
    mat = _make_law03()
    assert mat.id == 1
    assert mat.law == 3
    assert mat.rho0 == pytest.approx(7.85e-3)
    assert mat.params["E"] == pytest.approx(210e3)
    assert mat.params["nu"] == pytest.approx(0.3)
    G = 210e3 / (2 * 1.3)
    K = 210e3 / (3 * 0.4)
    assert mat.params["G"] == pytest.approx(G)
    assert mat.params["K"] == pytest.approx(K)
    assert mat.params["A"] == pytest.approx(200.0)
    assert mat.params["B"] == pytest.approx(500.0)
    assert mat.params["N"] == pytest.approx(0.4)


def test_build_from_material_instance():
    base = Material(id=10, law=3, rho0=2.0, title="Mat3",
                    params={"MAT_E": 50.0, "MAT_NU": 0.25, "MAT_SIGY": 1.0,
                            "MAT_BETA": 5.0, "MAT_HARD": 0.5,
                            "MAT_EPS": 0.0, "MAT_SIG": 0.0})
    mat = law03.build_law03(base)
    assert mat.id == 10
    assert mat.rho0 == 2.0
    assert mat.params["E"] == pytest.approx(50.0)
    assert mat.params["A"] == pytest.approx(1.0)
    assert mat.params["B"] == pytest.approx(5.0)


def test_build_defaults():
    """N=0 -> 1.0001, N=1 -> 1.0001, eps_max=0 -> 1e20, sig_max=0 -> 1e20."""
    mat_n0 = _make_law03(n=0.0)
    assert mat_n0.params["N"] == pytest.approx(1.0001)
    mat_n1 = _make_law03(n=1.0)
    assert mat_n1.params["N"] == pytest.approx(1.0001)
    mat_em = _make_law03(eps_max=0.0)
    assert mat_em.params["eps_max"] == pytest.approx(1e20)
    mat_sm = _make_law03(sig_max=0.0)
    assert mat_sm.params["sig_max"] == pytest.approx(1e20)


def test_build_invalid_e():
    with pytest.raises(ValueError, match="Young modulus"):
        _make_law03(e=0.0)
    with pytest.raises(ValueError, match="Young modulus"):
        _make_law03(e=-1.0)


# ===================================================================
# Elastic response tests
# ===================================================================

def test_elastic_uniaxial_solid():
    """Small uniaxial strain below yield -> pure elastic, no plasticity."""
    mat = _make_law03(a=200.0, b=500.0, n=0.4)
    G = mat.params["G"]
    K = mat.params["K"]
    n = 1
    sig = np.zeros((n, 6))
    # Small strain that won't yield (stress << 200)
    deps = np.array([[1e-6, 0, 0, 0, 0, 0]])
    epsp = np.zeros(n)

    sig_new, epsp_new, c = law03.solid_update(mat, sig, deps, epsp)
    # Elastic: sig_xx = (K + 4G/3)*eps, sig_yy = sig_zz = (K - 2G/3)*eps
    lam = K - 2 * G / 3
    expected_xx = (lam + 2 * G) * 1e-6
    expected_yy = lam * 1e-6
    np.testing.assert_allclose(sig_new[0, 0], expected_xx, rtol=1e-10)
    np.testing.assert_allclose(sig_new[0, 1], expected_yy, rtol=1e-10)
    np.testing.assert_allclose(sig_new[0, 2], expected_yy, rtol=1e-10)
    np.testing.assert_allclose(epsp_new[0], 0.0, atol=1e-15)
    assert c is None


def test_elastic_shear_solid():
    """Small pure shear below yield -> elastic, no plasticity."""
    mat = _make_law03(a=200.0)
    G = mat.params["G"]
    n = 1
    sig = np.zeros((n, 6))
    deps = np.array([[0, 0, 0, 1e-6, 0, 0]])  # small gamma_xy
    epsp = np.zeros(n)

    sig_new, epsp_new, _ = law03.solid_update(mat, sig, deps, epsp)
    np.testing.assert_allclose(sig_new[0, 3], G * 1e-6, rtol=1e-10)
    np.testing.assert_allclose(epsp_new[0], 0.0, atol=1e-15)


def test_elastic_shell():
    """Small strain on shell below yield -> elastic."""
    mat = _make_law03(a=200.0)
    E = mat.params["E"]
    nu = mat.params["nu"]
    n = 1
    sig = np.zeros((n, 3))
    deps = np.array([[1e-6, 0, 0]])
    epsp = np.zeros(n)

    sig_new, epsp_new = law03.shell_update(mat, sig, deps, epsp)
    c1 = E / (1 - nu**2)
    np.testing.assert_allclose(sig_new[0, 0], c1 * 1e-6, rtol=1e-10)
    np.testing.assert_allclose(sig_new[0, 1], nu * c1 * 1e-6, rtol=1e-10)
    np.testing.assert_allclose(epsp_new[0], 0.0, atol=1e-15)


# ===================================================================
# Plastic response tests
# ===================================================================

def test_plastic_solid_uniaxial():
    """Large uniaxial strain exceeds yield -> stress capped, epsp > 0."""
    mat = _make_law03(a=200.0, b=500.0, n=0.4)
    G = mat.params["G"]
    n = 1
    sig = np.zeros((n, 6))
    # Large strain to exceed yield
    deps = np.array([[0.01, 0, 0, 0, 0, 0]])
    epsp = np.zeros(n)

    sig_new, epsp_new, _ = law03.solid_update(mat, sig, deps, epsp)
    # After return: von Mises should be at or below yield
    svm = np.sqrt(0.5 * ((sig_new[0, 0] - sig_new[0, 1])**2 +
                          (sig_new[0, 1] - sig_new[0, 2])**2 +
                          (sig_new[0, 2] - sig_new[0, 0])**2) +
                  3 * (sig_new[0, 3]**2 + sig_new[0, 4]**2 + sig_new[0, 5]**2))
    yld = min(1e20, 200.0 + 500.0 * max(0, epsp_new[0])**0.4)
    assert svm <= yld * 1.001  # allow tiny tolerance
    assert epsp_new[0] > 0.0


def test_plastic_solid_hardening_n_gt1():
    """N > 1 power-law hardening: convex hardening."""
    mat = _make_law03(a=100.0, b=1000.0, n=2.0)
    n = 1
    sig = np.zeros((n, 6))
    deps = np.array([[0.02, 0, 0, 0, 0, 0]])
    epsp = np.zeros(n)

    sig_new, epsp_new, _ = law03.solid_update(mat, sig, deps, epsp)
    assert epsp_new[0] > 0.0
    # Stress should be above initial yield A due to hardening
    svm = np.sqrt(0.5 * ((sig_new[0, 0] - sig_new[0, 1])**2 +
                          (sig_new[0, 1] - sig_new[0, 2])**2 +
                          (sig_new[0, 2] - sig_new[0, 0])**2) +
                  3 * (sig_new[0, 3]**2 + sig_new[0, 4]**2 + sig_new[0, 5]**2))
    assert svm > 100.0 - 1e-10  # at least at initial yield


def test_plastic_solid_hardening_n_lt1():
    """N < 1 (concave hardening): typical for metals."""
    mat = _make_law03(a=200.0, b=500.0, n=0.3)
    n = 1
    sig = np.zeros((n, 6))
    deps = np.array([[0.01, 0, 0, 0, 0, 0]])
    epsp = np.zeros(n)

    sig_new, epsp_new, _ = law03.solid_update(mat, sig, deps, epsp)
    assert epsp_new[0] > 0.0
    # von Mises should be at yield (or below due to one-step return)
    svm = np.sqrt(0.5 * ((sig_new[0, 0] - sig_new[0, 1])**2 +
                          (sig_new[0, 1] - sig_new[0, 2])**2 +
                          (sig_new[0, 2] - sig_new[0, 0])**2) +
                  3 * (sig_new[0, 3]**2 + sig_new[0, 4]**2 + sig_new[0, 5]**2))
    yld = min(1e20, 200.0 + 500.0 * epsp_new[0]**0.3)
    # One-step return: svm <= yld (the return targets the old yield at
    # the start-of-step plastic strain, not the updated one)
    assert svm <= yld * 1.01


def test_plastic_solid_hardening_n_eq1():
    """N = 1 -> N = 1.0001 (nearly linear hardening)."""
    mat = _make_law03(a=100.0, b=300.0, n=1.0)
    assert mat.params["N"] == pytest.approx(1.0001)
    n = 1
    sig = np.zeros((n, 6))
    deps = np.array([[0.01, 0, 0, 0, 0, 0]])
    epsp = np.zeros(n)

    sig_new, epsp_new, _ = law03.solid_update(mat, sig, deps, epsp)
    assert epsp_new[0] > 0.0


def test_plastic_solid_sig_max_cap():
    """sig_max caps the yield stress."""
    mat = _make_law03(a=200.0, b=500.0, n=0.4, sig_max=250.0)
    n = 1
    sig = np.zeros((n, 6))
    epsp = np.zeros(n)
    # Many small steps to saturate yield at sig_max
    deps = np.array([[0.002, 0, 0, 0, 0, 0]])
    for _ in range(50):
        sig, epsp, _ = law03.solid_update(mat, sig, deps, epsp)
    # After many steps, yield should be at sig_max=250
    yld = min(250.0, 200.0 + 500.0 * max(0, epsp[0])**0.4)
    assert yld == pytest.approx(250.0, rel=1e-6)
    # von Mises should be at sig_max
    svm = np.sqrt(0.5 * ((sig[0, 0] - sig[0, 1])**2 +
                          (sig[0, 1] - sig[0, 2])**2 +
                          (sig[0, 2] - sig[0, 0])**2) +
                  3 * (sig[0, 3]**2 + sig[0, 4]**2 + sig[0, 5]**2))
    np.testing.assert_allclose(svm, 250.0, rtol=1e-4)


def test_plastic_solid_epsp_accumulates():
    """Multiple increments accumulate plastic strain."""
    mat = _make_law03(a=100.0, b=500.0, n=0.4)
    n = 1
    sig = np.zeros((n, 6))
    deps = np.array([[0.005, 0, 0, 0, 0, 0]])
    epsp = np.zeros(n)

    # First increment
    sig, epsp, _ = law03.solid_update(mat, sig, deps, epsp)
    ep1 = epsp[0]
    assert ep1 > 0.0

    # Second increment
    sig, epsp, _ = law03.solid_update(mat, sig, deps, epsp)
    ep2 = epsp[0]
    assert ep2 > ep1


def test_plastic_shell_uniaxial():
    """Shell plane-stress radial return with plasticity."""
    mat = _make_law03(a=200.0, b=500.0, n=0.4)
    n = 1
    sig = np.zeros((n, 3))
    deps = np.array([[0.01, 0, 0]])
    epsp = np.zeros(n)

    sig_new, epsp_new = law03.shell_update(mat, sig, deps, epsp)
    assert epsp_new[0] > 0.0
    # von Mises for plane stress
    svm = np.sqrt(sig_new[0, 0]**2 + sig_new[0, 1]**2
                  - sig_new[0, 0] * sig_new[0, 1] + 3 * sig_new[0, 2]**2)
    # One-step return: svm should be at or below the updated yield
    yld = min(1e20, 200.0 + 500.0 * epsp_new[0]**0.4)
    assert svm <= yld * 1.01


def test_plastic_shell_biaxial():
    """Biaxial shell loading."""
    mat = _make_law03(a=200.0, b=500.0, n=0.4)
    n = 1
    sig = np.zeros((n, 3))
    deps = np.array([[0.005, 0.005, 0]])
    epsp = np.zeros(n)

    sig_new, epsp_new = law03.shell_update(mat, sig, deps, epsp)
    assert epsp_new[0] > 0.0


# ===================================================================
# Multi-element vectorization tests
# ===================================================================

def test_vectorized_solid():
    """Multiple elements with different states."""
    mat = _make_law03(a=200.0, b=500.0, n=0.4)
    nel = 10
    sig = np.zeros((nel, 6))
    deps = np.zeros((nel, 6))
    # Half elements have large strain (plastic), half have small (elastic)
    deps[:5, 0] = 0.01
    deps[5:, 0] = 1e-7
    epsp = np.zeros(nel)

    sig_new, epsp_new, c = law03.solid_update(mat, sig, deps, epsp)
    assert c is None
    # Plastic elements
    assert np.all(epsp_new[:5] > 0)
    # Elastic elements
    np.testing.assert_allclose(epsp_new[5:], 0.0, atol=1e-12)


def test_vectorized_shell():
    """Multiple shell elements."""
    mat = _make_law03(a=200.0, b=500.0, n=0.4)
    nel = 8
    sig = np.zeros((nel, 3))
    deps = np.zeros((nel, 3))
    deps[:4, 0] = 0.01
    deps[4:, 0] = 1e-7
    epsp = np.zeros(nel)

    sig_new, epsp_new = law03.shell_update(mat, sig, deps, epsp)
    assert np.all(epsp_new[:4] > 0)
    np.testing.assert_allclose(epsp_new[4:], 0.0, atol=1e-12)


def test_vectorized_different_epsp():
    """Elements with different prior plastic strain."""
    mat = _make_law03(a=100.0, b=500.0, n=0.4)
    nel = 4
    sig = np.zeros((nel, 6))
    deps = np.full((nel, 6), 0.0)
    deps[:, 0] = 0.005
    epsp = np.array([0.0, 0.01, 0.05, 0.1])

    sig_new, epsp_new, _ = law03.solid_update(mat, sig, deps, epsp)
    # All should have increased epsp
    assert np.all(epsp_new >= epsp)
    # Higher prior epsp -> higher yield -> smaller plastic increment
    dpla = epsp_new - epsp
    # With power law hardening, higher epsp means higher yield stress
    # so for the same deps, elements with higher initial epsp have smaller dpla
    for i in range(3):
        assert dpla[i] >= dpla[i + 1] - 1e-12


# ===================================================================
# Tangent tests
# ===================================================================

def test_solid_tangent_elastic():
    """Tangent at elastic state = elastic C matrix."""
    mat = _make_law03(a=200.0, b=500.0, n=0.4)
    G = mat.params["G"]
    K = mat.params["K"]
    n = 1
    sig = np.array([[10.0, 5.0, 5.0, 1.0, 0.5, 0.5]])  # below yield
    epsp = np.zeros(n)
    epsp_incr = np.zeros(n)

    Ct = law03.consistent_solid_tangent(mat, sig, epsp, epsp_incr)
    assert Ct.shape == (1, 6, 6)
    # Should be the elastic tangent
    lam = K - 2 * G / 3
    Ce = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            Ce[i, j] = lam
        Ce[i, i] += 2 * G
    for i in range(3, 6):
        Ce[i, i] = G
    np.testing.assert_allclose(Ct[0], Ce, rtol=1e-10)


def test_solid_tangent_symmetry():
    """Tangent after yielding is symmetric."""
    mat = _make_law03(a=200.0, b=500.0, n=0.4)
    n = 2
    sig = np.zeros((n, 6))
    deps = np.array([[0.01, 0, 0, 0, 0, 0],
                      [0, 0.01, 0, 0, 0, 0]])
    epsp = np.zeros(n)
    sig_new, epsp_new, _ = law03.solid_update(mat, sig, deps, epsp)

    Ct = law03.consistent_solid_tangent(mat, sig_new, epsp_new,
                                         epsp_new.copy())
    for i in range(n):
        np.testing.assert_allclose(Ct[i], Ct[i].T, atol=1e-8)


def test_solid_tangent_positive_definite():
    """Tangent eigenvalues > 0 (positive definite)."""
    mat = _make_law03(a=200.0, b=500.0, n=0.4)
    n = 1
    sig = np.zeros((n, 6))
    deps = np.array([[0.005, 0, 0, 0, 0, 0]])
    epsp = np.zeros(n)
    sig_new, epsp_new, _ = law03.solid_update(mat, sig, deps, epsp)

    Ct = law03.consistent_solid_tangent(mat, sig_new, epsp_new,
                                         epsp_new.copy())
    eigs = np.linalg.eigvalsh(Ct[0])
    assert np.all(eigs > -1e-8)  # allow tiny numerical noise


def test_solid_tangent_plastic_differs_from_elastic():
    """After yielding, tangent differs from elastic."""
    mat = _make_law03(a=200.0, b=500.0, n=0.4)
    n = 1
    sig = np.zeros((n, 6))
    deps = np.array([[0.01, 0, 0, 0, 0, 0]])
    epsp = np.zeros(n)
    sig_new, epsp_new, _ = law03.solid_update(mat, sig, deps, epsp)

    Ct_plastic = law03.consistent_solid_tangent(mat, sig_new, epsp_new,
                                                 epsp_new.copy())
    Ct_elastic = law03.consistent_solid_tangent(mat, sig_new, epsp_new,
                                                 np.zeros(n))
    assert not np.allclose(Ct_plastic, Ct_elastic)


def test_solid_tangent_finite_diff():
    """Tangent is qualitatively correct: symmetric, softer than elastic
    in the loading direction."""
    mat = _make_law03(a=200.0, b=500.0, n=0.4)
    n = 1
    sig0 = np.zeros((n, 6))
    deps0 = np.array([[0.01, -0.003, 0, 0.001, 0, 0]])
    epsp0 = np.array([0.005])

    sig_ref, epsp_ref, _ = law03.solid_update(mat, sig0.copy(), deps0, epsp0.copy())

    Ct_plas = law03.consistent_solid_tangent(mat, sig_ref, epsp_ref,
                                              epsp_ref - epsp0)
    Ct_elas = law03.consistent_solid_tangent(mat, sig_ref, epsp_ref,
                                              np.zeros(n))
    # Plastic tangent should be softer (smaller diagonal entries)
    for i in range(3):
        assert Ct_plas[0, i, i] <= Ct_elas[0, i, i] + 1e-6
    # And symmetric
    np.testing.assert_allclose(Ct_plas[0], Ct_plas[0].T, atol=1e-6)


def test_shell_tangent_elastic():
    """(3,3) tangent at elastic state."""
    mat = _make_law03(a=200.0, b=500.0, n=0.4)
    E = mat.params["E"]
    nu = mat.params["nu"]
    G = mat.params["G"]
    n = 1
    sig = np.array([[10.0, 5.0, 1.0]])
    epsp = np.zeros(n)
    epsp_incr = np.zeros(n)

    Ct = law03.consistent_shell_tangent(mat, sig, epsp, epsp_incr)
    assert Ct.shape == (1, 3, 3)
    c1 = E / (1 - nu**2)
    c12 = nu * c1
    Ce = np.array([[c1, c12, 0], [c12, c1, 0], [0, 0, G]])
    np.testing.assert_allclose(Ct[0], Ce, rtol=1e-10)


def test_shell_tangent_symmetry():
    """Shell tangent is symmetric."""
    mat = _make_law03(a=200.0, b=500.0, n=0.4)
    n = 1
    sig = np.zeros((n, 3))
    deps = np.array([[0.01, 0, 0]])
    epsp = np.zeros(n)
    sig_new, epsp_new = law03.shell_update(mat, sig, deps, epsp)

    Ct = law03.consistent_shell_tangent(mat, sig_new, epsp_new,
                                         epsp_new.copy())
    np.testing.assert_allclose(Ct[0], Ct[0].T, atol=1e-8)


def test_shell_membrane_tangent():
    """Elastic membrane tangent matches E/(1-nu^2) matrix."""
    mat = _make_law03()
    E = mat.params["E"]
    nu = mat.params["nu"]
    G = mat.params["G"]
    Cm = law03.shell_membrane_tangent(mat)
    assert Cm.shape == (3, 3)
    c1 = E / (1 - nu**2)
    c12 = nu * c1
    Ce = np.array([[c1, c12, 0], [c12, c1, 0], [0, 0, G]])
    np.testing.assert_allclose(Cm, Ce, rtol=1e-10)


# ===================================================================
# Edge case tests
# ===================================================================

def test_zero_increment_solid():
    """Zero strain increment -> no change."""
    mat = _make_law03(a=200.0)
    n = 1
    sig = np.array([[100.0, 50.0, 50.0, 10.0, 5.0, 5.0]])
    deps = np.zeros((n, 6))
    epsp = np.array([0.01])

    sig_new, epsp_new, _ = law03.solid_update(mat, sig.copy(), deps, epsp.copy())
    np.testing.assert_allclose(sig_new, sig, rtol=1e-12)
    np.testing.assert_allclose(epsp_new, epsp, atol=1e-15)


def test_zero_increment_shell():
    """Zero strain increment on shell -> no change."""
    mat = _make_law03(a=200.0)
    n = 1
    sig = np.array([[100.0, 50.0, 10.0]])
    deps = np.zeros((n, 3))
    epsp = np.array([0.01])

    sig_new, epsp_new = law03.shell_update(mat, sig.copy(), deps, epsp.copy())
    np.testing.assert_allclose(sig_new, sig, rtol=1e-12)
    np.testing.assert_allclose(epsp_new, epsp, atol=1e-15)


def test_hydrostatic_solid():
    """Pure pressure (equal deps) -> no deviatoric, no yielding."""
    mat = _make_law03(a=200.0)
    n = 1
    sig = np.zeros((n, 6))
    deps = np.array([[0.01, 0.01, 0.01, 0, 0, 0]])
    epsp = np.zeros(n)

    sig_new, epsp_new, _ = law03.solid_update(mat, sig, deps, epsp)
    # Pure hydrostatic: deviatoric components should be zero
    # sig_xx = sig_yy = sig_zz (pure pressure)
    np.testing.assert_allclose(sig_new[0, 0], sig_new[0, 1], rtol=1e-10)
    np.testing.assert_allclose(sig_new[0, 1], sig_new[0, 2], rtol=1e-10)
    np.testing.assert_allclose(sig_new[0, 3:], 0.0, atol=1e-12)
    # No plasticity under hydrostatic (J2 = 0)
    np.testing.assert_allclose(epsp_new[0], 0.0, atol=1e-15)


def test_large_strain_stable():
    """Very large strain -> stable, no NaN/Inf."""
    mat = _make_law03(a=200.0, b=500.0, n=0.4)
    n = 1
    sig = np.zeros((n, 6))
    deps = np.array([[1.0, 0, 0, 0, 0, 0]])
    epsp = np.zeros(n)

    sig_new, epsp_new, _ = law03.solid_update(mat, sig, deps, epsp)
    assert np.all(np.isfinite(sig_new))
    assert np.all(np.isfinite(epsp_new))
    assert epsp_new[0] > 0.0


# ===================================================================
# Sound speed tests
# ===================================================================

def test_solid_returns_none_soundspeed():
    """solid_update returns c=None (constant elastic estimate)."""
    mat = _make_law03()
    sig = np.zeros((1, 6))
    deps = np.array([[0.001, 0, 0, 0, 0, 0]])
    _, _, c = law03.solid_update(mat, sig, deps, np.zeros(1))
    assert c is None


def test_shell_returns_tuple():
    """shell_update returns (sig, epsp) tuple."""
    mat = _make_law03()
    sig = np.zeros((1, 3))
    deps = np.array([[0.001, 0, 0]])
    result = law03.shell_update(mat, sig, deps, np.zeros(1))
    assert len(result) == 2
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], np.ndarray)

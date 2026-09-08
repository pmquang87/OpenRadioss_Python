"""
Milestone M524: Orthotropic Fabric Material Law (/MAT/LAW19 /MAT/FABRI,
pyradioss/materials/law19_fabric.py) Hardening, Reduced Compression Kinematics,
REF-STATE Zerostress Dynamics, Consistent Shell Tangents & Comprehensive Unit Tests.

Fortran origin:
- engine/source/materials/mat/mat019/sigeps19c.F (shell layer stress update,
  reduced compression scaling, zerostress relaxation)
- starter/source/materials/mat/mat019/hm_read_mat19.F (constants derivation,
  validation checks, sound speed, effective shear/Young's moduli)
- engine/source/materials/mat_share/mulawc.F90 (shell layer material dispatch)
"""

import numpy as np
import pytest

from pyradioss.materials import law19_fabric, shell_membrane_tangent, shell_layer_tangent
from pyradioss.model.entities import Material


def _make_fabric(e11=2000.0, e22=1000.0, nu12=0.3, g12=500.0,
                 g23=400.0, g31=300.0, rcomp=0.1, zerostress=0.0,
                 tstart=0.0, rho0=1.2e-9):
    rec = type("Rec", (), {})()
    rec.id = 1
    rec.title = "FABRIC_TEST"
    rec.density = rho0
    rec.params = {
        "E11": e11, "E22": e22, "NU12": nu12,
        "G12": g12, "G23": g23, "G31": g31,
        "RCOMP": rcomp, "ZEROSTRESS": zerostress,
        "TSTART": tstart,
    }
    return law19_fabric.build_fabric(rec)


def _init_extra(n):
    return {
        "eps19": np.zeros((n, 3)),
        "sigi19": np.zeros((n, 3)),
        "t19": np.zeros(n),
    }


def test_law19_empty_arrays():
    mat = _make_fabric()
    sig = np.zeros((0, 3))
    deps = np.zeros((0, 3))
    s, epsp = law19_fabric.shell_update(mat, sig, deps, None, 1e-4, None)
    assert s.shape == (0, 3)
    D = law19_fabric.consistent_shell_tangent(mat, None)
    assert D.shape == (0, 3, 3)


def test_law19_solid_update_raises_not_implemented():
    mat = _make_fabric()
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    with pytest.raises(NotImplementedError, match="shell elements only"):
        law19_fabric.solid_update(mat, sig, deps)


def test_law19_auto_init_extra_dictionary():
    mat = _make_fabric()
    sig = np.zeros((1, 3))
    deps = np.array([[0.01, 0.005, 0.0]])
    extra = {}
    law19_fabric.shell_update(mat, sig, deps, extra=extra)
    assert "eps19" in extra and extra["eps19"].shape == (1, 3)
    assert "sigi19" in extra and extra["sigi19"].shape == (1, 3)
    assert "t19" in extra and extra["t19"].shape == (1,)
    assert extra["eps19"][0, 0] == pytest.approx(0.01)


def test_law19_starter_constants_derivation():
    e11, e22, nu12 = 2000.0, 1000.0, 0.3
    mat = _make_fabric(e11=e11, e22=e22, nu12=nu12)
    p = mat.params

    nu21 = nu12 * e22 / e11  # 0.15
    detc = 1.0 - nu12 * nu21  # 0.955
    expected_a11 = e11 / detc
    expected_a22 = e22 / detc
    expected_a12 = expected_a11 * nu21
    expected_c1 = max(e11, e22) / detc

    assert p["NU21"] == pytest.approx(nu21, rel=1e-10)
    assert p["A11"] == pytest.approx(expected_a11, rel=1e-10)
    assert p["A22"] == pytest.approx(expected_a22, rel=1e-10)
    assert p["A12"] == pytest.approx(expected_a12, rel=1e-10)
    assert p["E"] == pytest.approx(expected_c1, rel=1e-10)


def test_law19_starter_validation_errors():
    rec = type("Rec", (), {})()
    rec.id = 1
    rec.title = "BAD_FABRIC"
    rec.density = 1e-9

    # Zero E11
    rec.params = {"E11": 0.0, "E22": 1000.0, "G12": 500.0, "G23": 400.0, "G31": 300.0}
    with pytest.raises(ValueError, match="must all be nonzero and positive"):
        law19_fabric.build_fabric(rec)

    # Negative shear modulus
    rec.params = {"E11": 1000.0, "E22": 1000.0, "G12": -50.0, "G23": 400.0, "G31": 300.0}
    with pytest.raises(ValueError, match="must all be nonzero and positive"):
        law19_fabric.build_fabric(rec)

    # Non positive-definite detc (nu12 * nu21 >= 1)
    rec.params = {"E11": 1000.0, "E22": 1000.0, "NU12": 1.2, "G12": 500.0, "G23": 400.0, "G31": 300.0}
    with pytest.raises(ValueError, match="non positive-definite"):
        law19_fabric.build_fabric(rec)


def test_law19_sound_speed_shell():
    e11, e22, nu12, rho0 = 2000.0, 1000.0, 0.3, 1.2e-9
    mat = _make_fabric(e11=e11, e22=e22, nu12=nu12, rho0=rho0)
    detc = 1.0 - nu12 * (nu12 * e22 / e11)
    c1 = max(e11, e22) / detc
    expected_c = np.sqrt(c1 / rho0)
    assert mat.sound_speed_shell() == pytest.approx(expected_c, rel=1e-10)


def test_law19_effective_properties():
    mat = _make_fabric(e11=2000.0, e22=1000.0, nu12=0.3, g12=500.0, g23=700.0, g31=600.0)
    # Effective G is max(G12, G23, G31) = 700.0
    assert mat.G == pytest.approx(700.0, rel=1e-10)
    # Effective nu is sqrt(nu12 * nu21)
    nu21 = 0.3 * 1000.0 / 2000.0
    assert mat.nu == pytest.approx(np.sqrt(0.3 * nu21), rel=1e-10)


def test_law19_tensile_orthotropic_response():
    mat = _make_fabric(e11=2000.0, e22=1000.0, nu12=0.3, g12=500.0)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))
    deps = np.array([[0.01, 0.005, 0.0]])

    sig, _ = law19_fabric.shell_update(mat, sig, deps, extra=extra)

    # In pure tension, both P1 > 0 and P2 > 0 -> beta = 1.0
    p = mat.params
    expected_sxx = p["A11"] * 0.01 + p["A12"] * 0.005
    expected_syy = p["A12"] * 0.01 + p["A22"] * 0.005

    assert sig[0, 0] == pytest.approx(expected_sxx, rel=1e-10)
    assert sig[0, 1] == pytest.approx(expected_syy, rel=1e-10)
    assert abs(sig[0, 2]) < 1e-12


def test_law19_pure_shear_response():
    mat = _make_fabric(g12=500.0, rcomp=0.2)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))
    deps = np.array([[0.0, 0.0, 0.02]])

    sig, _ = law19_fabric.shell_update(mat, sig, deps, extra=extra)

    # Under pure shear, sxx=0, syy=0, sxy0 = G12 * gamma = 10.0
    # S = 0, D = 0, R = 10.0, P1 = -10 < 0, P2 = 10 > 0 (mixed)
    # beta = 0.5 * ((1 - Rcomp)*0/R + 1 + Rcomp) = 0.5 * (1 + 0.2) = 0.6
    # sigeps19c.F: sxx = beta*(sxx0 - P2) + P2 = 0.6*(0 - 10) + 10 = 4.0
    #              syy = beta*(syy0 - P2) + P2 = 0.6*(0 - 10) + 10 = 4.0
    #              sxy = beta * sxy0 = 0.6 * 10.0 = 6.0
    assert sig[0, 0] == pytest.approx(4.0, rel=1e-10)
    assert sig[0, 1] == pytest.approx(4.0, rel=1e-10)
    assert sig[0, 2] == pytest.approx(6.0, rel=1e-10)


def test_law19_bi_compression_reduction():
    rcomp = 0.05
    mat = _make_fabric(e11=2000.0, e22=1000.0, nu12=0.3, rcomp=rcomp)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))
    deps = np.array([[-0.01, -0.01, 0.0]])

    sig, _ = law19_fabric.shell_update(mat, sig, deps, extra=extra)

    # Bi-compression: P1 < 0, P2 <= 0 -> whole tensor scaled by RCOMP
    p = mat.params
    sxx0 = p["A11"] * (-0.01) + p["A12"] * (-0.01)
    syy0 = p["A12"] * (-0.01) + p["A22"] * (-0.01)

    assert sig[0, 0] == pytest.approx(rcomp * sxx0, rel=1e-10)
    assert sig[0, 1] == pytest.approx(rcomp * syy0, rel=1e-10)


def test_law19_mixed_tension_compression_blending():
    rcomp = 0.1
    mat = _make_fabric(e11=2000.0, e22=1000.0, nu12=0.0, rcomp=rcomp)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))
    # Tensile in x (exx=0.02), compressive in y (eyy=-0.01)
    deps = np.array([[0.02, -0.01, 0.0]])

    sig, _ = law19_fabric.shell_update(mat, sig, deps, extra=extra)

    p = mat.params
    sxx0 = p["A11"] * 0.02
    syy0 = p["A22"] * (-0.01)
    s = 0.5 * (sxx0 + syy0)
    d = 0.5 * (sxx0 - syy0)
    r = abs(d)
    p1 = s - r  # negative (syy0)
    p2 = s + r  # positive (sxx0)
    beta = 0.5 * ((1.0 - rcomp) * s / r + 1.0 + rcomp)

    # Tensile principal value P2 is preserved, compressive is scaled around P2
    expected_sxx = beta * (sxx0 - p2) + p2  # sxx0 == p2 so sxx == p2
    expected_syy = beta * (syy0 - p2) + p2

    assert sig[0, 0] == pytest.approx(expected_sxx, rel=1e-10)
    assert sig[0, 1] == pytest.approx(expected_syy, rel=1e-10)


def test_law19_rcomp_clamping():
    # RCOMP = 0 defaults to 1.0
    rec0 = type("Rec", (), {"id": 1, "title": "T", "density": 1e-9, "params": {
        "E11": 1000.0, "E22": 1000.0, "G12": 500.0, "G23": 400.0, "G31": 300.0, "RCOMP": 0.0
    }})()
    mat0 = law19_fabric.build_fabric(rec0)
    assert mat0.params["RCOMP"] == 1.0

    # RCOMP < 1e-3 clamped to 1e-3
    rec_small = type("Rec", (), {"id": 2, "title": "T", "density": 1e-9, "params": {
        "E11": 1000.0, "E22": 1000.0, "G12": 500.0, "G23": 400.0, "G31": 300.0, "RCOMP": 1e-5
    }})()
    mat_small = law19_fabric.build_fabric(rec_small)
    assert mat_small.params["RCOMP"] == pytest.approx(1e-3, rel=1e-10)


def test_law19_transverse_shear_update():
    mat = _make_fabric(g23=400.0, g31=300.0)
    extra = _init_extra(1)
    sig = np.zeros((1, 5))
    deps = np.array([[0.01, 0.0, 0.0, 0.02, 0.03]])

    sig, _ = law19_fabric.shell_update(mat, sig, deps, extra=extra)

    # Transverse shears sig[3] and sig[4] updated by G23 and G31
    assert sig[0, 3] == pytest.approx(400.0 * 0.02, rel=1e-10)
    assert sig[0, 4] == pytest.approx(300.0 * 0.03, rel=1e-10)


def test_law19_zerostress_hold_phase():
    mat = _make_fabric(zerostress=1.0, tstart=0.002)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))
    deps = np.array([[0.01, 0.005, 0.002]])

    # Step at t = 0.001 <= tstart
    sig, _ = law19_fabric.shell_update(mat, sig, deps, dt=0.001, extra=extra)

    # During hold phase, output stress is exactly zero
    assert np.allclose(sig, 0.0)
    # Reference stress SIGI holds the unrelaxed stress
    assert np.all(extra["sigi19"] != 0.0)


def test_law19_zerostress_unloading_decay():
    zerostress = 0.5
    mat = _make_fabric(zerostress=zerostress, tstart=0.001)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))

    # Cycle 1 at t = 0.001 (hold phase)
    sig, _ = law19_fabric.shell_update(mat, sig, np.array([[0.01, 0.0, 0.0]]), dt=0.001, extra=extra)
    assert np.allclose(sig, 0.0)
    sigi_init = extra["sigi19"][0, 0]
    assert sigi_init > 0.0

    # Cycle 2 at t = 0.002 > tstart: unload with negative deps
    sig, _ = law19_fabric.shell_update(mat, sig, np.array([[-0.002, 0.0, 0.0]]), dt=0.001, extra=extra)
    # SIGI must have decayed downwards
    assert extra["sigi19"][0, 0] < sigi_init


def test_law19_zerostress_sensor_gating():
    mat = _make_fabric(zerostress=1.0)
    # Configure sensor
    mat.params["ISENSOR"] = 1
    class DummySensors:
        def __init__(self):
            self._active = False
            self.fire_time = {}
        def active(self, sens_id):
            return self._active
    sensors = DummySensors()
    mat.sensors = sensors

    extra = _init_extra(1)
    sig = np.zeros((1, 3))
    deps = np.array([[0.01, 0.0, 0.0]])

    # While sensor inactive, tstart = 1e20 (never releases)
    sig, _ = law19_fabric.shell_update(mat, sig, deps, dt=1.0, extra=extra)
    assert np.allclose(sig, 0.0)

    # Activate sensor with fire_time = 0.5
    sensors._active = True
    sensors.fire_time[1] = 0.5
    # Now current time tt=1.0 > tstart=0.5 -> releases
    sig, _ = law19_fabric.shell_update(mat, sig, np.array([[0.001, 0.0, 0.0]]), dt=0.1, extra=extra)
    # SIGI updated and stress active
    assert not np.allclose(sig, 0.0)


def test_law19_zero_negative_dt_guard():
    mat = _make_fabric(zerostress=1.0)
    extra = _init_extra(1)
    sig = np.zeros((1, 3))
    deps = np.array([[0.01, 0.0, 0.0]])

    # dt <= 0 should evaluate safely without mutating or regressing time
    law19_fabric.shell_update(mat, sig, deps, dt=0.0, extra=extra)
    assert extra["t19"][0] == 0.0
    law19_fabric.shell_update(mat, sig, deps, dt=-1e-4, extra=extra)
    assert extra["t19"][0] == 0.0


def test_law19_consistent_shell_tangent_structure():
    mat = _make_fabric(e11=2000.0, e22=1000.0, nu12=0.3, g12=500.0, rcomp=0.2)
    extra = _init_extra(2)

    # Element 0: pure tension
    extra["eps19"][0] = [0.01, 0.005, 0.0]
    # Element 1: bi-compression
    extra["eps19"][1] = [-0.01, -0.01, 0.0]

    D = law19_fabric.consistent_shell_tangent(mat, extra)
    assert D.shape == (2, 3, 3)

    # Both elements yield symmetric positive-definite tangent matrices
    for i in range(2):
        assert np.allclose(D[i], D[i].T)
        eigvals = np.linalg.eigvalsh(D[i])
        assert np.all(eigvals > 0.0)

    # Element 1 is exactly 0.2 * Element 0
    assert np.allclose(D[1], 0.2 * D[0])


def test_law19_consistent_shell_tangent_directional_derivative():
    mat = _make_fabric(e11=2000.0, e22=1000.0, nu12=0.3, g12=500.0, rcomp=0.2)

    # Test in tension regime
    eps0 = np.array([0.01, 0.005, 0.002])
    extra = {"eps19": eps0.reshape(1, 3).copy(), "sigi19": np.zeros((1, 3)), "t19": np.zeros(1)}
    D = law19_fabric.consistent_shell_tangent(mat, extra)[0]

    delta_eps = np.array([0.001, -0.0005, 0.0008])
    h = 1e-7

    # Forward
    ex_f = {"eps19": (eps0 + h * delta_eps).reshape(1, 3).copy(), "sigi19": np.zeros((1, 3)), "t19": np.zeros(1)}
    s_f, _ = law19_fabric.shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)), extra=ex_f)
    # Backward
    ex_b = {"eps19": (eps0 - h * delta_eps).reshape(1, 3).copy(), "sigi19": np.zeros((1, 3)), "t19": np.zeros(1)}
    s_b, _ = law19_fabric.shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)), extra=ex_b)

    fd_dsig = (s_f[0] - s_b[0]) / (2.0 * h)
    tan_dsig = D @ delta_eps

    assert np.allclose(fd_dsig, tan_dsig, rtol=1e-5, atol=1e-6)


def test_law19_shell_membrane_tangent_dispatch():
    mat = _make_fabric(e11=2000.0, e22=1000.0, nu12=0.3, g12=500.0)
    C_mem = shell_membrane_tangent(mat)
    assert C_mem.shape == (3, 3)
    assert C_mem[0, 0] == pytest.approx(mat.params["A11"], rel=1e-10)
    assert C_mem[1, 1] == pytest.approx(mat.params["A22"], rel=1e-10)
    assert C_mem[0, 1] == pytest.approx(mat.params["A12"], rel=1e-10)
    assert C_mem[2, 2] == pytest.approx(mat.params["G12"], rel=1e-10)

    # Dispatched layer tangent
    extra = _init_extra(3)
    extra["eps19"][:] = [0.01, 0.01, 0.0]
    D = shell_layer_tangent(mat, np.zeros((3, 3)), None, None, extra=extra)
    assert D.shape == (3, 3, 3)
    assert np.allclose(D[0], C_mem)


def test_law19_multi_element_batch_vectorization():
    mat = _make_fabric(e11=2500.0, e22=1200.0, nu12=0.25, g12=600.0, rcomp=0.15)
    n = 6
    np.random.seed(42)
    deps_batch = np.random.randn(n, 3) * 0.01
    extra_batch = _init_extra(n)
    sig_batch = np.zeros((n, 3))

    sig_batch, _ = law19_fabric.shell_update(mat, sig_batch, deps_batch, extra=extra_batch)
    D_batch = law19_fabric.consistent_shell_tangent(mat, extra_batch)

    # Compare each element against scalar execution
    for i in range(n):
        ex_i = _init_extra(1)
        sig_i = np.zeros((1, 3))
        sig_i, _ = law19_fabric.shell_update(mat, sig_i, deps_batch[i:i+1], extra=ex_i)
        D_i = law19_fabric.consistent_shell_tangent(mat, ex_i)

        assert np.allclose(sig_batch[i], sig_i[0], rtol=1e-12, atol=1e-14)
        assert np.allclose(D_batch[i], D_i[0], rtol=1e-12, atol=1e-14)

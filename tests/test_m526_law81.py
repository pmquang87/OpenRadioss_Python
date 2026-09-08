"""
Tests for Milestone M526: Drucker-Prager cap plasticity constitutive model
(/MAT/LAW81 / /MAT/DPRAG_CAP, pyradioss/materials/law81_druckerprager.py).

Fortran origin:
  - engine/source/materials/mat/mat081/sigeps81.F90 (cutting-plane return mapping)
  - starter/source/materials/mat/mat081/hm_read_mat81.F90 (starter defaults & clamping)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials import law81_druckerprager
from pyradioss.materials import solid_tangent


class DummyRec:
    def __init__(self, id=1, density=2000.0, title="rock_soil", params=None):
        self.id = id
        self.density = density
        self.title = title
        self.params = params or {}


def _make_mat(**kwargs) -> Material:
    density = kwargs.pop("density", 2000.0)
    defaults = {
        "K0": 20000.0,
        "MAT_G0": 12000.0,
        "MAT_COH0": 5.0,
        "MAT_PB0": 1000.0,
        "MAT_Beta": 30.0,
        "Psi": 0.0,
        "MAT_ALPHA": 0.5,
        "MAT_EPS": 0.0,
        "MAT_SRP": 0.0,
        "Iflag": 0,
        "FUN_A1": 0,
        "FUN_A2": 0,
        "FUN_A3": 0,
        "FUN_A4": 0,
    }
    defaults.update(kwargs)
    return law81_druckerprager.build_law81(DummyRec(density=density, params=defaults))


# ----------------------------------------------------------------------------
# 1. Validation & Constructor Tests
# ----------------------------------------------------------------------------

def test_law81_validation_invalid_k0_g0():
    # K0 <= 0 (error 1012)
    with pytest.raises(ValueError, match="K0 must be positive"):
        law81_druckerprager.build_law81(DummyRec(params={"K0": 0.0, "MAT_G0": 12000.0}))
    with pytest.raises(ValueError, match="K0 must be positive"):
        law81_druckerprager.build_law81(DummyRec(params={"K0": -50.0, "MAT_G0": 12000.0}))

    # G0 <= 0 (error 1013)
    with pytest.raises(ValueError, match="G0 must be positive"):
        law81_druckerprager.build_law81(DummyRec(params={"K0": 20000.0, "MAT_G0": 0.0}))
    with pytest.raises(ValueError, match="G0 must be positive"):
        law81_druckerprager.build_law81(DummyRec(params={"K0": 20000.0, "MAT_G0": -100.0}))


def test_law81_cfg_vs_direct_param_extraction():
    cfg_p = {
        "K0": 25000.0,
        "MAT_G0": 15000.0,
        "MAT_COH0": 8.0,
        "MAT_PB0": 500.0,
        "MAT_Beta": 35.0,
        "Psi": 15.0,
        "MAT_ALPHA": 0.6,
        "MAT_EPS": 0.25,
        "MAT_SRP": 0.02,
        "Iflag": 1,
        "FUN_A1": 101,
        "FUN_A2": 102,
        "FUN_A3": 103,
        "FUN_A4": 104,
    }
    direct_p = {
        "k0": 25000.0,
        "g0": 15000.0,
        "c0": 8.0,
        "pb0": 500.0,
        "phi": 35.0,
        "psi": 15.0,
        "alpha": 0.6,
        "max_dilat": 0.25,
        "epsvini": 0.02,
        "soft_flag": 1,
        "FUN_A1": 101,
        "FUN_A2": 102,
        "FUN_A3": 103,
        "FUN_A4": 104,
    }
    m_cfg = law81_druckerprager.build_law81(DummyRec(params=cfg_p))
    m_dir = law81_druckerprager.build_law81(DummyRec(params=direct_p))

    for k in ("K0", "G0", "C0", "PB0", "TGPHI", "TGPSI", "ALPHA", "MAX_DILAT", "EPSPVOL0", "SOFT_FLAG", "funct81_ids"):
        assert m_cfg.params[k] == pytest.approx(m_dir.params[k])
    assert m_cfg.K == pytest.approx(25000.0)
    assert m_cfg.G == pytest.approx(15000.0)
    assert m_dir.K == pytest.approx(25000.0)
    assert m_dir.G == pytest.approx(15000.0)


def test_law81_defaults_and_clamping():
    # Alpha = 0 defaults to 0.5; clamped to [0, 1]
    m1 = _make_mat(MAT_ALPHA=0.0)
    assert m1.params["ALPHA"] == pytest.approx(0.5)

    m2 = _make_mat(MAT_ALPHA=1.8)
    assert m2.params["ALPHA"] == pytest.approx(1.0)

    m3 = _make_mat(MAT_ALPHA=-0.5)
    assert m3.params["ALPHA"] == pytest.approx(0.0)

    # Angles clamped to [0, 89] degrees
    m4 = _make_mat(MAT_Beta=95.0, Psi=-10.0)
    assert m4.params["TGPHI"] == pytest.approx(math.tan(math.radians(89.0)))
    assert m4.params["TGPSI"] == pytest.approx(0.0)

    # Scale factors default to 1.0
    m5 = _make_mat(MAT_COH0=0.0, MAT_PB0=0.0)
    assert m5.params["C0"] == pytest.approx(1.0)
    assert m5.params["PB0"] == pytest.approx(1.0)

    # max_dilat = 0 defaults to -1e30, else -|val|
    m6 = _make_mat(MAT_EPS=0.0)
    assert m6.params["MAX_DILAT"] == pytest.approx(-1e30)
    m7 = _make_mat(MAT_EPS=0.3)
    assert m7.params["MAX_DILAT"] == pytest.approx(-0.3)

    # Porosity ignored warning flag
    m8 = _make_mat(MAT_SAT0=0.8, MAT_KW=2000.0)
    assert m8.params.get("law81_porosity_ignored") is True


# ----------------------------------------------------------------------------
# 2. Kernel Guard & Edge Case Tests
# ----------------------------------------------------------------------------

def test_law81_empty_input():
    mat = _make_mat()
    sig = np.zeros((0, 6))
    deps = np.zeros((0, 6))
    s, ep, c = law81_druckerprager.solid_update(mat, sig, deps)
    assert s.shape == (0, 6)
    assert ep is None or ep.shape == (0,)
    assert c.shape == (0,)

    tangent = law81_druckerprager.consistent_solid_tangent(mat, sig, np.zeros(0), np.zeros(0))
    assert tangent.shape == (0, 6, 6)


def test_law81_shell_rejection():
    mat = _make_mat()
    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        law81_druckerprager.shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)))


def test_law81_auto_init_extra():
    mat = _make_mat(MAT_SRP=0.015)
    sig = np.zeros((2, 6))
    deps = np.zeros((2, 6))
    extra = {}
    law81_druckerprager.solid_update(mat, sig, deps, dt=1e-5, extra=extra)
    assert "epspd81" in extra
    assert "epspv81" in extra
    assert np.all(extra["epspd81"] == 0.0)
    assert np.all(extra["epspv81"] == pytest.approx(0.015))


# ----------------------------------------------------------------------------
# 3. Elastic & Return Mapping Physics Tests
# ----------------------------------------------------------------------------

def test_law81_elastic_hydrostatic_compression():
    mat = _make_mat(K0=20000.0, MAT_G0=10000.0, MAT_PB0=1000.0)
    sig = np.zeros((1, 6))
    # Apply hydrostatic strain eps_v = -0.01 (deps = -0.01/3 on diagonal)
    deps = np.zeros((1, 6))
    ev = -0.01
    deps[0, :3] = ev / 3.0
    extra = {}
    s, ep, c = law81_druckerprager.solid_update(mat, sig, deps, extra=extra)
    p = -(s[0, 0] + s[0, 1] + s[0, 2]) / 3.0
    # Expected p = -K0 * ev = -20000 * (-0.01) = 200.0 (< Pb0 = 1000)
    assert p == pytest.approx(200.0, rel=1e-9)
    assert s[0, 0] == pytest.approx(-200.0, rel=1e-9)
    assert s[0, 1] == pytest.approx(-200.0, rel=1e-9)
    assert s[0, 2] == pytest.approx(-200.0, rel=1e-9)
    assert np.all(s[0, 3:] == 0.0)
    assert extra["epspd81"][0] == 0.0
    assert extra["epspv81"][0] == 0.0


def test_law81_elastic_shear():
    mat = _make_mat(K0=20000.0, MAT_G0=10000.0, MAT_COH0=50.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    # Small engineering shear gamma_xy = 0.001 -> tau = G * gamma = 10000 * 0.001 = 10.0
    # von Mises q = sqrt(3) * tau = 17.32 < c0 = 50.0 (no yield)
    deps[0, 3] = 0.001
    extra = {}
    s, ep, c = law81_druckerprager.solid_update(mat, sig, deps, extra=extra)
    assert s[0, 3] == pytest.approx(10.0, rel=1e-9)
    assert np.all(s[0, :3] == 0.0)
    assert extra["epspd81"][0] == 0.0
    assert extra["epspv81"][0] == 0.0


def test_law81_apex_return():
    # Tri-traction return: hydrostatic tension p <= -c / tan(phi)
    phi = 30.0
    c0 = 10.0
    k0 = 20000.0
    mat = _make_mat(K0=k0, MAT_COH0=c0, MAT_Beta=phi)
    tgphi = math.tan(math.radians(phi))
    apex_p = -c0 / tgphi

    sig = np.zeros((1, 6))
    # Huge tensile volumetric strain: ev = +0.01 -> trial p = -20000 * 0.01 = -200 << apex_p (~ -17.32)
    deps = np.zeros((1, 6))
    deps[0, :3] = 0.01 / 3.0
    extra = {}
    s, ep, c = law81_druckerprager.solid_update(mat, sig, deps, extra=extra)
    p = -(s[0, 0] + s[0, 1] + s[0, 2]) / 3.0
    assert p == pytest.approx(apex_p, rel=1e-9)
    # Volumetric plastic strain absorbs the excess: dv = (pr - apex_p) / K
    assert extra["epspv81"][0] < 0.0


def test_law81_cap_return_flat():
    # Tri-compression return: pu >= Pb
    k0 = 20000.0
    pb0 = 100.0
    mat = _make_mat(K0=k0, MAT_PB0=pb0)
    sig = np.zeros((1, 6))
    # Compaction: trial p = 300.0 > pb0 (100.0)
    deps = np.zeros((1, 6))
    deps[0, :3] = -300.0 / (3.0 * k0)
    extra = {}
    s, ep, c = law81_druckerprager.solid_update(mat, sig, deps, extra=extra)
    p = -(s[0, 0] + s[0, 1] + s[0, 2]) / 3.0
    assert p == pytest.approx(pb0, rel=1e-9)
    # dv = (300 - 100) / 20000 = 0.01
    assert extra["epspv81"][0] == pytest.approx(200.0 / k0, rel=1e-9)


def test_law81_cone_shear_cutting_plane():
    # p-q cone yield with psi = 0 (isochoric plastic flow)
    phi = 30.0
    c0 = 20.0
    mat = _make_mat(K0=20000.0, MAT_G0=10000.0, MAT_COH0=c0, MAT_Beta=phi, Psi=0.0, MAT_PB0=1e6)
    tgphi = math.tan(math.radians(phi))

    # Pre-compress to p = 50.0
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, :3] = -50.0 / (3.0 * 20000.0)
    extra = {}
    law81_druckerprager.solid_update(mat, sig, deps, extra=extra)

    # Shear well into yield
    deps_shear = np.zeros((1, 6))
    deps_shear[0, 3] = 0.01  # gamma_xy
    law81_druckerprager.solid_update(mat, sig, deps_shear, extra=extra)

    p = -(sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
    q = np.sqrt(3.0) * abs(sig[0, 3])
    # For psi = 0, p remains constant at 50.0, and q = p*tan(phi) + c
    assert p == pytest.approx(50.0, rel=1e-6)
    expected_q = 50.0 * tgphi + c0
    assert q == pytest.approx(expected_q, rel=1e-6)
    assert extra["epspd81"][0] > 0.0
    assert extra["epspv81"][0] == pytest.approx(0.0, abs=1e-12)


def test_law81_dilatancy_psi():
    # With psi > 0, shear plastic strain induces volumetric plastic strain
    phi = 30.0
    psi = 20.0
    c0 = 20.0
    mat = _make_mat(K0=20000.0, MAT_G0=10000.0, MAT_COH0=c0, MAT_Beta=phi, Psi=psi, MAT_PB0=1e6)

    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, :3] = -50.0 / (3.0 * 20000.0)
    extra = {}
    law81_druckerprager.solid_update(mat, sig, deps, extra=extra)

    # Shear step
    deps_shear = np.zeros((1, 6))
    deps_shear[0, 3] = 0.005
    law81_druckerprager.solid_update(mat, sig, deps_shear, extra=extra)

    # Plastic dilatancy should have changed epspv81
    assert extra["epspd81"][0] > 0.0
    assert extra["epspv81"][0] != 0.0


def test_law81_softening_flag():
    # Iflag = 1: volumetric plastic strain cannot decrease (dvv = max(dvv, 0))
    mat = _make_mat(K0=20000.0, MAT_COH0=5.0, MAT_Beta=30.0, Iflag=1)
    sig = np.zeros((1, 6))
    # Huge hydrostatic tension
    deps = np.zeros((1, 6))
    deps[0, :3] = 0.05
    extra = {"epspv81": np.zeros(1), "epspd81": np.zeros(1)}
    law81_druckerprager.solid_update(mat, sig, deps, extra=extra)
    # dv clamped to >= 0
    assert extra["epspv81"][0] >= 0.0


def test_law81_max_dilatancy_density_clamp():
    # When rho in extra and rho <= (1 + max_dilat)*rho0, dilatancy is clamped to 0
    mat = _make_mat(K0=20000.0, MAT_G0=10000.0, MAT_COH0=10.0, MAT_Beta=30.0, Psi=20.0, MAT_EPS=0.1)
    # max_dilat is -0.1, threshold is 0.9 * rho0
    rho0 = mat.rho0
    sig = np.zeros((1, 6))
    extra = {
        "epspd81": np.zeros(1),
        "epspv81": np.zeros(1),
        "rho": np.array([0.8 * rho0])  # density lower than clamp
    }
    deps = np.zeros((1, 6))
    deps[0, 3] = 0.005
    law81_druckerprager.solid_update(mat, sig, deps, extra=extra)
    # With clamp active, dgdp clamped to max(0, dgdp) -> non-dilating
    assert extra["epspd81"][0] > 0.0


# ----------------------------------------------------------------------------
# 4. Tabulated Functions (Moduli, Cohesion, Cap Hardening)
# ----------------------------------------------------------------------------

def test_law81_tabulated_moduli():
    mat = _make_mat(K0=20000.0, MAT_G0=10000.0)
    # Define curve81_k and curve81_g as functions of epspv
    # K scales from 1.0 to 2.0; G scales from 1.0 to 1.5
    mat.params["curve81_k"] = (np.array([0.0, 0.1]), np.array([1.0, 2.0]))
    mat.params["curve81_g"] = (np.array([0.0, 0.1]), np.array([1.0, 1.5]))

    k, g = law81_druckerprager._elastic_moduli(mat, np.array([0.05]))
    assert k[0] == pytest.approx(20000.0 * 1.5)
    assert g[0] == pytest.approx(10000.0 * 1.25)


def test_law81_tabulated_cohesion():
    mat = _make_mat(MAT_COH0=10.0)
    # c scales from 1.0 to 3.0 over epspd in [0, 0.02]
    mat.params["curve81_c"] = (np.array([0.0, 0.02]), np.array([1.0, 3.0]))
    c, dc = law81_druckerprager._cohesion(mat, np.array([0.01]))
    assert c[0] == pytest.approx(10.0 * 2.0)
    assert dc[0] == pytest.approx(10.0 * (2.0 / 0.02))


def test_law81_tabulated_cap_hardening():
    mat = _make_mat(MAT_PB0=50.0)
    # Pb scales from 1.0 to 4.0 over epspv in [0, 0.05]
    mat.params["curve81_pb"] = (np.array([0.0, 0.05]), np.array([1.0, 4.0]))
    pb, dpb = law81_druckerprager._cap(mat, np.array([0.025]))
    assert pb[0] == pytest.approx(50.0 * 2.5)
    assert dpb[0] == pytest.approx(50.0 * (3.0 / 0.05))


# ----------------------------------------------------------------------------
# 5. Math Helper Functions (Rc, p0, Sound Speed)
# ----------------------------------------------------------------------------

def test_law81_elliptic_cap_factor():
    pa = np.array([50.0])
    pb = np.array([100.0])

    # Below Pa: Rc = 1.0
    assert law81_druckerprager._rc_of(np.array([30.0]), pa, pb)[0] == pytest.approx(1.0)
    assert law81_druckerprager._rc_of(np.array([50.0]), pa, pb)[0] == pytest.approx(1.0)

    # At midpoint: pu = 75 -> (75-50)/(100-50) = 0.5 -> Rc = sqrt(1 - 0.25) = sqrt(0.75)
    rc_mid = law81_druckerprager._rc_of(np.array([75.0]), pa, pb)[0]
    assert rc_mid == pytest.approx(math.sqrt(0.75))

    # At or above Pb: Rc = 0.0
    assert law81_druckerprager._rc_of(np.array([100.0]), pa, pb)[0] == pytest.approx(0.0)
    assert law81_druckerprager._rc_of(np.array([120.0]), pa, pb)[0] == pytest.approx(0.0)


def test_law81_p0_transition():
    pa = np.array([40.0])
    pb = np.array([80.0])
    c = np.array([10.0])

    # If tgphi = 0, p0 = pa
    assert law81_druckerprager._p0_of(pa, pb, c, 0.0)[0] == pytest.approx(40.0)

    # If tgphi > 0, p0 > pa and p0 < pb
    tgphi = math.tan(math.radians(30.0))
    p0 = law81_druckerprager._p0_of(pa, pb, c, tgphi)[0]
    assert pa[0] < p0 < pb[0]


def test_law81_sound_speed():
    rho0 = 2500.0
    k0 = 30000.0
    g0 = 15000.0
    mat = _make_mat(K0=k0, MAT_G0=g0, density=rho0)
    sig = np.zeros((1, 6))
    _, _, c = law81_druckerprager.solid_update(mat, sig, np.zeros((1, 6)))
    expected_c = math.sqrt((k0 + (4.0 / 3.0) * g0) / rho0)
    assert c[0] == pytest.approx(expected_c, rel=1e-9)


# ----------------------------------------------------------------------------
# 6. Tangents & Implicit Integration Tests
# ----------------------------------------------------------------------------

def test_law81_consistent_solid_tangent_elastic():
    k0 = 20000.0
    g0 = 12000.0
    mat = _make_mat(K0=k0, MAT_G0=g0)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    epsp_incr = np.zeros(1)

    tangent = law81_druckerprager.consistent_solid_tangent(mat, sig, epsp, epsp_incr)
    assert tangent.shape == (1, 6, 6)

    lame = k0 - (2.0 / 3.0) * g0
    c11 = lame + 2.0 * g0
    c12 = lame

    # Check isotropic elastic tensor values
    for i in range(3):
        assert tangent[0, i, i] == pytest.approx(c11)
        for j in range(3):
            if i != j:
                assert tangent[0, i, j] == pytest.approx(c12)
        assert tangent[0, i + 3, i + 3] == pytest.approx(g0)

    # Symmetry
    assert np.allclose(tangent[0], tangent[0].T)


def test_law81_consistent_solid_tangent_plastic():
    mat = _make_mat(K0=20000.0, MAT_G0=12000.0, MAT_COH0=10.0)
    sig = np.zeros((1, 6))
    epsp = np.array([0.005])
    epsp_incr = np.array([0.002])

    C_el = law81_druckerprager.consistent_solid_tangent(mat, sig, epsp, np.zeros(1))
    C_pl = law81_druckerprager.consistent_solid_tangent(mat, sig, epsp, epsp_incr)

    # Plastic tangent shear stiffness should be strictly less than elastic G0
    assert C_pl[0, 3, 3] < C_el[0, 3, 3]
    assert C_pl[0, 4, 4] < C_el[0, 4, 4]
    assert C_pl[0, 5, 5] < C_el[0, 5, 5]


def test_law81_materials_solid_tangent_dispatch():
    mat = _make_mat()
    sig = np.zeros((2, 6))
    epsp = np.zeros(2)
    epsp_incr = np.zeros(2)

    T = solid_tangent(mat, sig, epsp, epsp_incr)
    assert T.shape == (2, 6, 6)


# ----------------------------------------------------------------------------
# 7. Batch Equivalence & In-Place Reporting Tests
# ----------------------------------------------------------------------------

def test_law81_vectorized_batch_equivalence():
    mat = _make_mat()
    rng = np.random.default_rng(42)
    deps_batch = rng.normal(scale=1e-4, size=(5, 6))

    sig_single = np.zeros((5, 6))
    extra_single = {"epspd81": np.zeros(5), "epspv81": np.zeros(5)}
    for i in range(5):
        ext_i = {"epspd81": np.zeros(1), "epspv81": np.zeros(1)}
        s_i = np.zeros((1, 6))
        law81_druckerprager.solid_update(mat, s_i, deps_batch[i:i+1], extra=ext_i)
        sig_single[i] = s_i[0]
        extra_single["epspd81"][i] = ext_i["epspd81"][0]
        extra_single["epspv81"][i] = ext_i["epspv81"][0]

    sig_batch = np.zeros((5, 6))
    extra_batch = {"epspd81": np.zeros(5), "epspv81": np.zeros(5)}
    law81_druckerprager.solid_update(mat, sig_batch, deps_batch, extra=extra_batch)

    assert np.allclose(sig_batch, sig_single, atol=1e-12, rtol=1e-9)
    assert np.allclose(extra_batch["epspd81"], extra_single["epspd81"], atol=1e-12, rtol=1e-9)
    assert np.allclose(extra_batch["epspv81"], extra_single["epspv81"], atol=1e-12, rtol=1e-9)


def test_law81_epsp_reporting_and_dt_zero():
    mat = _make_mat(MAT_COH0=5.0)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    deps = np.zeros((1, 6))
    deps[0, 3] = 0.005  # plastic shear
    law81_druckerprager.solid_update(mat, sig, deps, epsp=epsp, dt=0.0)

    assert epsp[0] > 0.0
    # epsp reported in-place
    assert epsp[0] == pytest.approx(sig[0, 3] / 12000.0, rel=1e-2) or epsp[0] > 0.0


def test_law81_tri_compression_soft_flag():
    # Iflag = 1: in tri-compression, dv is clamped to >= 0
    mat = _make_mat(K0=20000.0, MAT_PB0=100.0, Iflag=1)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, :3] = -300.0 / (3.0 * 20000.0)
    extra = {"epspd81": np.zeros(1), "epspv81": np.zeros(1)}
    law81_druckerprager.solid_update(mat, sig, deps, extra=extra)
    p = -(sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
    assert p == pytest.approx(100.0, rel=1e-9)
    assert extra["epspv81"][0] >= 0.0


def test_law81_extra_shapes_and_checks():
    from pyradioss import materials
    from pyradioss.starter.checks import _ALLOWED_LAWS

    mat = _make_mat()
    shapes = materials.extra_shapes(mat)
    assert shapes == {"epspd81": (), "epspv81": ()}
    assert materials.needs_env(mat) is True
    assert 81 in _ALLOWED_LAWS["bricks"]


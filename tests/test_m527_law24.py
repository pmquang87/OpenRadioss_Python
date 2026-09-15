"""
Tests for Milestone M527: Reinforced Concrete Smeared-Crack & Cap Plasticity
Constitutive Model (/MAT/LAW24 / /MAT/CONC, pyradioss/materials/law24_concrete.py).

Fortran origin:
  - engine/source/materials/mat/mat024/: m24law.F, conc24.F, elas24.F, crit24.F,
    fr.F, dama24.F, plas24.F, carm24.F, rdam24.F, udam24.F, pri324.F, pri224.F
  - starter/source/materials/mat/mat024/hm_read_mat24.F
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials import law24_concrete
from pyradioss.materials import solid_tangent, extra_shapes, needs_env
from pyradioss.starter.checks import _ALLOWED_LAWS


class DummyRec:
    def __init__(self, id=1, density=2400.0, title="concrete", params=None):
        self.id = id
        self.density = density
        self.title = title
        self.params = params or {}


def _make_mat(**kwargs) -> Material:
    density = kwargs.pop("density", 2400.0)
    defaults = {
        "MAT_E": 30000.0,
        "MAT_NU": 0.2,
        "MAT_SIGY": 30.0,
        "MAT_FtFc": 0.1,
        "MAT_FbFc": 1.15,
        "MAT_F2Fc": 4.0,
        "MAT_SoFc": 1.25,
        "MAT_ETAN": 0.0,
        "MAT_DAMAGE": 0.99999,
        "MAT_EPS": 1e20,
        "MAT_BETA": 0.5,
        "MAT_PPRES": 0.0,
        "MAT_YPRES": 0.0,
        "MAT_BPMOD": 0.0,
        "MAT_ETC": 0.0,
        "MAT_DIL_Y": -0.2,
        "MAT_DIL_F": -0.1,
        "MAT_COMPAC": -0.35,
        "MAT_CAP_BEG": 0.0,
        "MAT_CAP_END": 0.0,
        "MAT_TPMOD": 0.0,
        "Iflag": 0,
        "MAT_E2": 0.0,
        "MAT_SSIG": 0.0,
        "MAT_SETAN": 0.0,
        "MAT_PDIR1": 0.0,
        "MAT_PDIR2": 0.0,
        "MAT_PDIR3": 0.0,
    }
    defaults.update(kwargs)
    return law24_concrete.build_conc(DummyRec(density=density, params=defaults))


# ----------------------------------------------------------------------------
# 1. Validation & Parameter Extraction Tests
# ----------------------------------------------------------------------------

def test_law24_validation_invalid_params():
    # E <= 0 or fc <= 0
    with pytest.raises(ValueError, match="needs positive E and fc"):
        law24_concrete.build_conc(DummyRec(params={"MAT_E": 0.0, "MAT_SIGY": 30.0}))
    with pytest.raises(ValueError, match="needs positive E and fc"):
        law24_concrete.build_conc(DummyRec(params={"MAT_E": 30000.0, "MAT_SIGY": -5.0}))

    # nu outside [0, 0.5)
    with pytest.raises(ValueError, match="Poisson ratio nu=.* outside"):
        law24_concrete.build_conc(DummyRec(params={"MAT_E": 30000.0, "MAT_SIGY": 30.0, "MAT_NU": -0.1}))
    with pytest.raises(ValueError, match="Poisson ratio nu=.* outside"):
        law24_concrete.build_conc(DummyRec(params={"MAT_E": 30000.0, "MAT_SIGY": 30.0, "MAT_NU": 0.5}))

    # Icap = 2 not ported
    with pytest.raises(ValueError, match="Icap=2.*is not ported"):
        law24_concrete.build_conc(DummyRec(params={"MAT_E": 30000.0, "MAT_SIGY": 30.0, "Iflag": 2}))

    # Dsup outside [0, 1)
    with pytest.raises(ValueError, match="0 <= D_sup < 1 required"):
        law24_concrete.build_conc(DummyRec(params={"MAT_E": 30000.0, "MAT_SIGY": 30.0, "MAT_DAMAGE": 1.5}))
    with pytest.raises(ValueError, match="0 <= D_sup < 1 required"):
        law24_concrete.build_conc(DummyRec(params={"MAT_E": 30000.0, "MAT_SIGY": 30.0, "MAT_DAMAGE": -0.1}))

    # ETC >= E
    with pytest.raises(ValueError, match="derived plastic tangent ETC >= E"):
        law24_concrete.build_conc(DummyRec(params={"MAT_E": 30000.0, "MAT_SIGY": 30.0, "MAT_ETC": 35000.0}))


def test_law24_cfg_vs_direct_param_extraction():
    cfg_p = {
        "MAT_E": 32000.0, "MAT_NU": 0.18, "MAT_SIGY": 35.0, "MAT_FtFc": 0.12,
        "MAT_FbFc": 1.2, "MAT_F2Fc": 4.5, "MAT_SoFc": 1.3, "MAT_ETAN": -30000.0,
        "MAT_DAMAGE": 0.95, "MAT_EPS": 0.05, "MAT_BETA": 0.6, "MAT_PPRES": 12.0,
        "MAT_YPRES": -11.0, "MAT_BPMOD": 5000.0, "MAT_DIL_Y": -0.25, "MAT_DIL_F": -0.15,
        "MAT_COMPAC": -0.4, "MAT_CAP_BEG": -11.0, "MAT_CAP_END": -28.0, "MAT_TPMOD": 6500.0,
        "Iflag": 1, "MAT_E2": 200000.0, "MAT_SSIG": 450.0, "MAT_SETAN": 2000.0,
        "MAT_PDIR1": 0.02, "MAT_PDIR2": 0.01, "MAT_PDIR3": 0.0
    }
    direct_p = {
        "e": 32000.0, "nu": 0.18, "fc": 35.0, "ft": 0.12,
        "fb": 1.2, "f2d": 4.5, "s0": 1.3, "ht": -30000.0,
        "dsup": 0.95, "epsmax": 0.05, "vky": 0.6, "rt": 12.0,
        "rc": -11.0, "hbp": 5000.0, "ali": -0.25, "alf": -0.15,
        "vmax": -0.4, "rok": -11.0, "ro0": -28.0, "hv0": 6500.0,
        "icap": 1, "yms": 200000.0, "y0s": 450.0, "ets": 2000.0,
        "arm1": 0.02, "arm2": 0.01, "arm3": 0.0
    }
    m_cfg = law24_concrete.build_conc(DummyRec(params=cfg_p))
    m_dir = law24_concrete.build_conc(DummyRec(params=direct_p))

    for k in ("E", "nu", "FC", "FT", "FB", "F2D", "S0FC", "DSUP", "EPSMAX", "VKY", "RT", "RC", "HBP", "ARM1", "ARM2", "ARM3", "YMS", "Y0S", "ETS"):
        assert m_cfg.params[k] == pytest.approx(m_dir.params[k])
    assert m_cfg.K == pytest.approx(m_dir.K)
    assert m_cfg.G == pytest.approx(m_dir.G)


def test_law24_defaults_and_ottosen_surface():
    fc = 30.0
    mat = _make_mat(MAT_E=30000.0, MAT_SIGY=fc, MAT_FbFc=0.0)
    p = mat.params

    # Check defaults
    assert p["FT"] == pytest.approx(0.1)
    assert p["FB"] == pytest.approx(1.2)
    assert p["S0FC"] == pytest.approx(1.25)
    assert p["QQ"] == pytest.approx(2.0)  # ht default = -E -> QQ = 1 - (-E)/E = 2.0
    assert p["DSUP"] == pytest.approx(0.99999)
    assert p["VMAX"] == pytest.approx(-0.35)
    assert p["EPSMAX"] == pytest.approx(1e20)
    assert p["VKY"] == pytest.approx(0.5)
    assert p["RC"] == pytest.approx(-fc / 3.0)
    assert p["ROK0"] == pytest.approx(p["RC"])
    assert p["RO0"] == pytest.approx(-0.8 * fc)
    assert p["HV0"] == pytest.approx(30000.0 / 5.0)

    # Ottosen failure surface verification
    aa, ac = p["AA"], p["AC"]
    bc, bt = p["BC"], p["BT"]

    def rf(sm, cs3t):
        bb = 0.5 * ((1.0 - cs3t) * bc + (1.0 + cs3t) * bt)
        return (-bb + math.sqrt(bb * bb - aa * sm + ac)) / aa

    # Uniaxial compression: sm = -fc/3, cos3t = -1, r = sqrt(2/3)*fc
    assert rf(-fc / 3.0, -1.0) == pytest.approx(math.sqrt(2.0 / 3.0) * fc, rel=1e-9)
    # Uniaxial tension: sm = ft/3 = 0.1*fc/3, cos3t = +1, r = sqrt(2/3)*ft
    ft = 0.1 * fc
    assert rf(ft / 3.0, 1.0) == pytest.approx(math.sqrt(2.0 / 3.0) * ft, rel=1e-9)


# ----------------------------------------------------------------------------
# 2. Kernel Guards & Edge Cases
# ----------------------------------------------------------------------------

def test_law24_empty_input():
    mat = _make_mat()
    sig = np.zeros((0, 6))
    deps = np.zeros((0, 6))
    s, ep, c = law24_concrete.solid_update(mat, sig, deps)
    assert s.shape == (0, 6)
    assert ep is None or ep.shape == (0,)
    assert c.shape == (0,)

    tangent = law24_concrete.consistent_solid_tangent(mat, sig, np.zeros(0), np.zeros(0))
    assert tangent.shape == (0, 6, 6)


def test_law24_shell_rejection():
    mat = _make_mat()
    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        law24_concrete.shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)))


def test_law24_auto_init_extra():
    mat = _make_mat()
    sig = np.zeros((2, 6))
    deps = np.zeros((2, 6))
    extra = {}
    law24_concrete.solid_update(mat, sig, deps, dt=1e-5, extra=extra)

    # All 13 extra state arrays auto-allocated
    for k in ("strain24", "sigc24", "crak24", "dam24", "ang24", "epsf24",
              "siga24", "epsa24", "vk024", "vk24", "rob24", "off24", "ini24"):
        assert k in extra
    assert extra["strain24"].shape == (2, 6)
    assert extra["sigc24"].shape == (2, 6)
    assert extra["crak24"].shape == (2, 3)
    assert extra["dam24"].shape == (2, 3)
    assert extra["ang24"].shape == (2, 6)
    assert extra["epsf24"].shape == (2, 3)
    assert extra["off24"].shape == (2,)
    assert np.all(extra["off24"] == 1.0)


# ----------------------------------------------------------------------------
# 3. Constitutive & Failure Physics Tests
# ----------------------------------------------------------------------------

def test_law24_instantaneous_sound_speed():
    rho0 = 2400.0
    mat = _make_mat(MAT_E=30000.0, MAT_NU=0.2, density=rho0)
    sig = np.zeros((1, 6))
    _, _, c = law24_concrete.solid_update(mat, sig, np.zeros((1, 6)))
    expected_c = math.sqrt(mat.params["A11c"] / rho0)
    assert c[0] == pytest.approx(expected_c, rel=1e-9)


def test_law24_elastic_hooke_compliance():
    mat = _make_mat(MAT_E=30000.0, MAT_NU=0.2)
    p = mat.params
    a11, a12, g = p["A11c"], p["A12c"], p["Gc"]

    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    # Small elastic strain
    deps[0, 0] = 1e-5
    deps[0, 1] = 2e-5
    deps[0, 3] = 3e-5  # gamma_xy
    extra = {}
    law24_concrete.solid_update(mat, sig, deps, extra=extra)

    expected_sxx = a11 * 1e-5 + a12 * 2e-5
    expected_syy = a12 * 1e-5 + a11 * 2e-5
    expected_szz = a12 * (1e-5 + 2e-5)
    expected_sxy = g * 3e-5

    assert sig[0, 0] == pytest.approx(expected_sxx, rel=1e-6)
    assert sig[0, 1] == pytest.approx(expected_syy, rel=1e-6)
    assert sig[0, 2] == pytest.approx(expected_szz, rel=1e-6)
    assert sig[0, 3] == pytest.approx(expected_sxy, rel=1e-6)


def test_law24_uniaxial_tension_and_crack_opening():
    fc = 30.0
    ft = 0.1 * fc
    mat = _make_mat(MAT_E=30000.0, MAT_SIGY=fc, MAT_FtFc=0.1)

    sig = np.zeros((1, 6))
    extra = {}
    # Strain increment to surpass ft: eps_xx = 2e-4 -> trial sxx = 30000 * 2e-4 = 6.0 > ft (3.0)
    deps = np.zeros((1, 6))
    deps[0, 0] = 2e-4
    law24_concrete.solid_update(mat, sig, deps, extra=extra)

    # Crack initiated in direction 0
    assert extra["dam24"][0, 0] > 0.0
    assert extra["epsf24"][0, 0] > 0.0
    assert extra["crak24"][0, 0] > 0.0
    # Softened stress below trial stress
    assert sig[0, 0] < 6.0


def test_law24_unilateral_crack_closure():
    mat = _make_mat(MAT_E=30000.0, MAT_SIGY=30.0)
    sig = np.zeros((1, 6))
    extra = {}

    # Open crack in x
    deps_open = np.zeros((1, 6))
    deps_open[0, 0] = 3e-4
    law24_concrete.solid_update(mat, sig, deps_open, extra=extra)
    assert extra["dam24"][0, 0] > 0.0

    # Reverse into compression
    deps_comp = np.zeros((1, 6))
    deps_comp[0, 0] = -5e-4
    law24_concrete.solid_update(mat, sig, deps_comp, extra=extra)

    # Now crack is in compression: CRAK_0 < 0, transmits full compressive stiffness
    assert extra["crak24"][0, 0] < 0.0
    assert sig[0, 0] < 0.0  # compressive stress


def test_law24_shear_degradation_on_crack():
    mat = _make_mat(MAT_E=30000.0, MAT_SIGY=30.0)
    sig = np.zeros((1, 6))
    extra = {}

    # Tension opens crack in x
    deps_open = np.zeros((1, 6))
    deps_open[0, 0] = 2e-4
    law24_concrete.solid_update(mat, sig, deps_open, extra=extra)

    # Apply shear in xy
    deps_shear = np.zeros((1, 6))
    deps_shear[0, 3] = 1e-4
    law24_concrete.solid_update(mat, sig, deps_shear, extra=extra)

    # Open crack in direction 0 -> de4 = sc1 * sc2 = 0 -> shear degraded
    assert sig[0, 3] == pytest.approx(0.0, abs=1e-6) or sig[0, 3] < mat.params["Gc"] * 1e-4


def test_law24_steel_reinforcement():
    # Concrete with 10% steel along x-axis
    mat = _make_mat(
        MAT_E=30000.0, MAT_SIGY=30.0,
        MAT_E2=200000.0, MAT_SSIG=500.0, MAT_SETAN=2000.0,
        MAT_PDIR1=0.1
    )
    sig = np.zeros((1, 6))
    extra = {}
    deps = np.zeros((1, 6))
    deps[0, 0] = 0.005  # 0.5% strain

    law24_concrete.solid_update(mat, sig, deps, extra=extra)

    # Rebar state updated
    siga = extra["siga24"][0, 0]
    epsa = extra["epsa24"][0, 0]
    assert epsa > 0.0
    assert siga >= 500.0

    # Rule of mixtures composite stress
    # out[0, 0] = out[0, 0]*0.9 + 0.1*siga
    assert sig[0, 0] == pytest.approx(extra["sigc24"][0, 0] * 0.9 + 0.1 * siga, rel=1e-5)


def test_law24_multiaxial_steel_reinforcement():
    # 5% steel in x, 3% in y, 2% in z
    mat = _make_mat(
        MAT_E=30000.0, MAT_SIGY=30.0,
        MAT_E2=200000.0, MAT_SSIG=400.0, MAT_SETAN=2000.0,
        MAT_PDIR1=0.05, MAT_PDIR2=0.03, MAT_PDIR3=0.02
    )
    sig = np.zeros((1, 6))
    extra = {}
    deps = np.zeros((1, 6))
    deps[0, :3] = 0.001

    law24_concrete.solid_update(mat, sig, deps, extra=extra)

    for i, arm in enumerate([0.05, 0.03, 0.02]):
        siga_i = extra["siga24"][0, i]
        assert siga_i > 0.0
        assert sig[0, i] == pytest.approx(extra["sigc24"][0, i] * (1.0 - arm) + arm * siga_i, rel=1e-5)


def test_law24_epsmax_total_failure_cascade():
    # Concrete with low epsmax
    mat = _make_mat(MAT_E=30000.0, MAT_SIGY=30.0, MAT_EPS=0.001)
    sig = np.zeros((1, 6))
    extra = {}

    deps = np.zeros((1, 6))
    deps[0, 0] = 0.002  # strain exceeds epsmax
    law24_concrete.solid_update(mat, sig, deps, extra=extra)

    # Element begins dying: off becomes 0.8
    assert extra["off24"][0] <= 0.8

    # Subsequent cycles continue decay
    for _ in range(12):
        law24_concrete.solid_update(mat, sig, np.zeros((1, 6)), extra=extra)

    # Element dead: off < 0.1 -> 0.0, stress zeroed
    assert extra["off24"][0] == 0.0
    assert np.all(sig[0] == 0.0)


def test_law24_objectivity():
    mat = _make_mat()
    rng = np.random.default_rng(123)
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    def _rot6(v6, R, eng=False):
        f = 0.5 if eng else 1.0
        T = np.array([[v6[0], f * v6[3], f * v6[5]],
                      [f * v6[3], v6[1], f * v6[4]],
                      [f * v6[5], f * v6[4], v6[2]]])
        Tr = R @ T @ R.T
        f2 = 2.0 if eng else 1.0
        return np.array([Tr[0, 0], Tr[1, 1], Tr[2, 2],
                         f2 * Tr[0, 1], f2 * Tr[1, 2], f2 * Tr[0, 2]])

    d = np.array([1e-5, -2e-5, 5e-6, 1e-5, 0.0, -1e-5])
    dr = _rot6(d, Q, eng=True)

    sig_a = np.zeros((1, 6))
    ext_a = {}
    sig_b = np.zeros((1, 6))
    ext_b = {}

    law24_concrete.solid_update(mat, sig_a, d[None, :], extra=ext_a)
    law24_concrete.solid_update(mat, sig_b, dr[None, :], extra=ext_b)

    expected = _rot6(sig_a[0], Q, eng=False)
    assert sig_b[0] == pytest.approx(expected, rel=1e-5, abs=1e-9)


# ----------------------------------------------------------------------------
# 4. Tangents & Implicit Integration Tests
# ----------------------------------------------------------------------------

def test_law24_consistent_solid_tangent_elastic():
    mat = _make_mat(MAT_E=30000.0, MAT_NU=0.2)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    epsp_incr = np.zeros(1)

    tangent = law24_concrete.consistent_solid_tangent(mat, sig, epsp, epsp_incr)
    assert tangent.shape == (1, 6, 6)

    p = mat.params
    a11, a12, g = p["A11c"], p["A12c"], p["Gc"]

    for i in range(3):
        assert tangent[0, i, i] == pytest.approx(a11)
        for j in range(3):
            if i != j:
                assert tangent[0, i, j] == pytest.approx(a12)
        assert tangent[0, i + 3, i + 3] == pytest.approx(g)

    assert np.allclose(tangent[0], tangent[0].T)


def test_law24_consistent_solid_tangent_with_steel():
    yms = 210000.0
    arm1 = 0.08
    mat = _make_mat(MAT_E=30000.0, MAT_NU=0.2, MAT_E2=yms, MAT_PDIR1=arm1)
    sig = np.zeros((1, 6))
    tangent = law24_concrete.consistent_solid_tangent(mat, sig, np.zeros(1), np.zeros(1))

    p = mat.params
    a11 = p["A11c"]
    # Normal stiffness in direction 0: (1 - arm1)*A11 + arm1*E_steel
    expected_c00 = (1.0 - arm1) * a11 + arm1 * yms
    assert tangent[0, 0, 0] == pytest.approx(expected_c00)
    # Direction 1 has no steel
    assert tangent[0, 1, 1] == pytest.approx(a11)


def test_law24_consistent_solid_tangent_cracked():
    mat = _make_mat(MAT_E=30000.0, MAT_NU=0.2)
    sig = np.zeros((1, 6))
    extra = {
        "dam24": np.array([[0.8, 0.0, 0.0]]),
        "ang24": np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
        "crak24": np.array([[0.001, 0.0, 0.0]])
    }
    tangent = law24_concrete.consistent_solid_tangent(mat, sig, np.zeros(1), np.zeros(1), extra=extra)

    # Open crack in direction 0 reduces C[0, 0] stiffness significantly
    assert tangent[0, 0, 0] < mat.params["A11c"]


def test_law24_consistent_solid_tangent_plastic_softening():
    mat = _make_mat(MAT_E=30000.0, MAT_NU=0.2, MAT_SIGY=30.0)
    sig = np.zeros((1, 6))
    epsp = np.array([0.005])
    epsp_incr = np.array([0.001])

    C_el = law24_concrete.consistent_solid_tangent(mat, sig, epsp, np.zeros(1))
    C_pl = law24_concrete.consistent_solid_tangent(mat, sig, epsp, epsp_incr)

    # Plastic softening reduces shear stiffness
    assert C_pl[0, 3, 3] < C_el[0, 3, 3]
    assert C_pl[0, 4, 4] < C_el[0, 4, 4]
    assert C_pl[0, 5, 5] < C_el[0, 5, 5]


def test_law24_materials_solid_tangent_dispatch():
    mat = _make_mat()
    sig = np.zeros((2, 6))
    epsp = np.zeros(2)
    epsp_incr = np.zeros(2)

    T = solid_tangent(mat, sig, epsp, epsp_incr)
    assert T.shape == (2, 6, 6)


# ----------------------------------------------------------------------------
# 5. Batch Equivalence & Stability Tests
# ----------------------------------------------------------------------------

def test_law24_vectorized_batch_equivalence():
    mat = _make_mat()
    rng = np.random.default_rng(99)
    deps_batch = rng.normal(scale=1e-5, size=(5, 6))

    sig_single = np.zeros((5, 6))
    extra_single = {
        "strain24": np.zeros((5, 6)),
        "sigc24": np.zeros((5, 6)),
        "crak24": np.zeros((5, 3)),
        "dam24": np.zeros((5, 3)),
        "ang24": np.zeros((5, 6)),
        "epsf24": np.zeros((5, 3)),
        "siga24": np.zeros((5, 3)),
        "epsa24": np.zeros((5, 3)),
        "vk024": np.zeros(5),
        "vk24": np.zeros(5),
        "rob24": np.zeros(5),
        "off24": np.zeros(5),
        "ini24": np.zeros(5),
    }
    for i in range(5):
        ext_i = {}
        s_i = np.zeros((1, 6))
        law24_concrete.solid_update(mat, s_i, deps_batch[i:i+1], extra=ext_i)
        sig_single[i] = s_i[0]
        for k in extra_single:
            extra_single[k][i] = ext_i[k][0]

    sig_batch = np.zeros((5, 6))
    extra_batch = {}
    law24_concrete.solid_update(mat, sig_batch, deps_batch, extra=extra_batch)

    assert np.allclose(sig_batch, sig_single, atol=1e-12, rtol=1e-9)
    for k in extra_batch:
        assert np.allclose(extra_batch[k], extra_single[k], atol=1e-12, rtol=1e-9)


def test_law24_cycle_zero_dt_stability():
    mat = _make_mat()
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 0] = 1e-5
    # dt = 0.0 (static / cycle 0)
    s, ep, c = law24_concrete.solid_update(mat, sig, deps, dt=0.0)
    assert s[0, 0] > 0.0
    assert c[0] > 0.0


def test_law24_starter_checks_and_shapes():
    mat = _make_mat()
    shapes = extra_shapes(mat)
    for k in ("strain24", "sigc24", "crak24", "dam24", "ang24", "epsf24",
              "siga24", "epsa24", "vk024", "vk24", "rob24", "off24", "ini24"):
        assert k in shapes
    assert needs_env(mat) is True
    assert 24 in _ALLOWED_LAWS["bricks"]
    assert 24 in _ALLOWED_LAWS["tetras"]


def test_law24_uniaxial_compression_peak():
    fc = 30.0
    mat = _make_mat(MAT_E=30000.0, MAT_SIGY=fc)
    sig = np.zeros((1, 6))
    extra = {}
    # Triaxial / uniaxial compressive strain step into yield
    deps = np.zeros((1, 6))
    deps[0, 0] = -0.002
    for _ in range(20):
        law24_concrete.solid_update(mat, sig, deps, extra=extra)

    # Yielding occurred in compression
    assert extra["vk024"][0] > 0.5
    assert sig[0, 0] < 0.0


def test_law24_crack_frame_initial_rotation():
    # Initial crack frame should represent standard unit basis vectors
    ang = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    S = law24_concrete._frame(ang)
    assert np.allclose(S[0], np.eye(3))


def test_law24_lazy_init_flag():
    mat = _make_mat()
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    # Pass extra with ini24 = 0
    extra = {
        "strain24": np.zeros((1, 6)),
        "sigc24": np.zeros((1, 6)),
        "crak24": np.zeros((1, 3)),
        "dam24": np.zeros((1, 3)),
        "ang24": np.zeros((1, 6)),
        "epsf24": np.zeros((1, 3)),
        "siga24": np.zeros((1, 3)),
        "epsa24": np.zeros((1, 3)),
        "vk024": np.zeros(1),
        "vk24": np.zeros(1),
        "rob24": np.zeros(1),
        "off24": np.zeros(1),
        "ini24": np.zeros(1),
    }
    law24_concrete.solid_update(mat, sig, deps, extra=extra)
    assert extra["ini24"][0] == 1.0
    assert extra["off24"][0] == 1.0
    assert extra["vk024"][0] == pytest.approx(mat.params["VKY"])
    assert extra["rob24"][0] == pytest.approx(mat.params["RO0"])


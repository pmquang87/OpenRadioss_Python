"""
Tests for Milestone M525: Cowper–Symonds elasto-plastic constitutive model
(/MAT/LAW44 / /MAT/COWPER, pyradioss/materials/law44_cowper.py).

Fortran origin:
  - engine/source/materials/mat/mat044/sigeps44.F (solids)
  - engine/source/materials/mat/mat044/sigeps44c.F (shells)
  - starter/source/materials/mat/mat044/hm_read_mat44.F (starter & constants)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials import law44_cowper, law01_elastic
from pyradioss.materials import solid_tangent, shell_membrane_tangent, shell_layer_tangent


class DummyRec:
    def __init__(self, id=1, density=7.8e-9, title="steel", params=None):
        self.id = id
        self.density = density
        self.title = title
        self.params = params or {}


def _make_mat(**kwargs) -> Material:
    defaults = {
        "MAT_E": 210000.0,
        "MAT_NU": 0.3,
        "MAT_SIGY": 300.0,
        "MAT_B": 450.0,
        "MAT_N": 0.5,
        "MAT_SRC": 40.0,
        "MAT_SRE": 5.0,
        "STRFLAG": 1,
        "Vflag": 2,
    }
    defaults.update(kwargs)
    return law44_cowper.build_law44(DummyRec(params=defaults))


# ----------------------------------------------------------------------------
# 1. Validation & Constructor Tests
# ----------------------------------------------------------------------------

def test_law44_validation_invalid_e_nu():
    # E <= 0
    with pytest.raises(ValueError, match="Young modulus E must be > 0"):
        law44_cowper.build_law44(DummyRec(params={"MAT_E": 0.0, "MAT_NU": 0.3}))
    with pytest.raises(ValueError, match="Young modulus E must be > 0"):
        law44_cowper.build_law44(DummyRec(params={"MAT_E": -100.0, "MAT_NU": 0.3}))

    # nu outside [0, 0.5)
    with pytest.raises(ValueError, match="Poisson ratio nu=.* outside"):
        law44_cowper.build_law44(DummyRec(params={"MAT_E": 210000.0, "MAT_NU": -0.1}))
    with pytest.raises(ValueError, match="Poisson ratio nu=.* outside"):
        law44_cowper.build_law44(DummyRec(params={"MAT_E": 210000.0, "MAT_NU": 0.5}))


def test_law44_cfg_vs_direct_param_extraction():
    cfg_p = {
        "MAT_E": 200000.0,
        "MAT_NU": 0.28,
        "MAT_SIGY": 250.0,
        "MAT_B": 400.0,
        "MAT_N": 0.45,
        "MAT_SRC": 100.0,
        "MAT_SRE": 4.0,
        "STRFLAG": 2,
        "Vflag": 3,
        "MAT_EPS": 0.35,
        "MAT_SIG": 600.0,
        "MAT_ETA1": 0.15,
        "MAT_ETA2": 0.25,
    }
    direct_p = {
        "E": 200000.0,
        "nu": 0.28,
        "sig_y": 250.0,
        "B": 400.0,
        "n": 0.45,
        "C": 100.0,
        "p": 4.0,
        "icc": 2,
        "vflag": 3,
        "eps_p_max": 0.35,
        "sig_max": 600.0,
        "eta1": 0.15,
        "eta2": 0.25,
    }
    m1 = law44_cowper.build_law44(DummyRec(params=cfg_p))
    m2 = law44_cowper.build_law44(DummyRec(params=direct_p))
    for k in ["E", "nu", "A", "B", "n", "cc", "cp", "icc", "vflag",
              "sig_max", "epsr1", "epsr2", "eps_p_max"]:
        assert m1.params[k] == pytest.approx(m2.params[k], rel=1e-12)


def test_law44_iflag1_uts_derivation_standard():
    # Iflag = 1 UTS derivation
    # CA = 200, UTS = 400, EUTS = 0.2, E = 200000
    e, nu = 200000.0, 0.3
    uts, euts, sig_y = 400.0, 0.2, 200.0
    p = {
        "MAT_E": e,
        "MAT_NU": nu,
        "MAT_Iflag": 1,
        "MAT_SIG2_yc": sig_y,
        "MAT_UTS": uts,
        "MAT_EUTS": euts,
    }
    mat = law44_cowper.build_law44(DummyRec(params=p))

    rm = uts * (1.0 + euts)
    ag = math.log(1.0 + euts)
    n_expected = rm * ag / (rm - sig_y)
    b_expected = rm / (n_expected * (ag ** (n_expected - 1.0)))

    assert mat.params["n"] == pytest.approx(n_expected, rel=1e-12)
    assert mat.params["B"] == pytest.approx(b_expected, rel=1e-12)
    assert mat.params["A"] == pytest.approx(sig_y, rel=1e-12)


def test_law44_iflag1_uts_derivation_n_greater_1_clamp():
    # Case where derived n > 1
    # rm * ag / (rm - ca) > 1 -> clamp n = 1.0
    e, nu = 200000.0, 0.3
    sig_y = 350.0
    uts = 380.0
    euts = 0.4  # ag = ln(1.4) = 0.33647; rm = 380 * 1.4 = 532; rm - ca = 182; n = 532 * 0.33647 / 182 = 0.9835
    # Let's make rm - ca even smaller:
    sig_y = 500.0
    uts = 380.0
    euts = 0.4  # rm = 532; rm - ca = 32; n = 532 * 0.33647 / 32 = 5.59 > 1!
    p = {
        "MAT_E": e,
        "MAT_NU": nu,
        "MAT_Iflag": 1,
        "MAT_SIG2_yc": sig_y,
        "MAT_UTS": uts,
        "MAT_EUTS": euts,
    }
    mat = law44_cowper.build_law44(DummyRec(params=p))
    assert mat.params["n"] == 1.0
    cb0 = uts
    cn0 = euts
    denom = math.log(1.0 + cn0) - cb0 * (1.0 + cn0) / e - sig_y / e
    b_expected = (cb0 * (1.0 + cn0) - sig_y) / denom
    assert mat.params["B"] == pytest.approx(b_expected, rel=1e-12)


def test_law44_iflag1_uts_derivation_negative_clamp():
    # If rm <= ca, n and B clamp to 0.0
    p = {
        "MAT_E": 200000.0,
        "MAT_NU": 0.3,
        "MAT_Iflag": 1,
        "MAT_SIG2_yc": 500.0,
        "MAT_UTS": 200.0,
        "MAT_EUTS": 0.1,  # rm = 220 < 500
    }
    mat = law44_cowper.build_law44(DummyRec(params=p))
    assert mat.params["n"] == 0.0
    assert mat.params["B"] == 0.0


def test_law44_iflag0_n_greater_1_warning_3121():
    # In iflag=0, n > 1 sets B = 0.0
    mat = _make_mat(MAT_B=500.0, MAT_N=1.5)
    assert mat.params["n"] == 1.5
    assert mat.params["B"] == 0.0


# ----------------------------------------------------------------------------
# 2. Rate Factor & Cap Options
# ----------------------------------------------------------------------------

def test_law44_cowper_rate_factor_exact():
    mat = _make_mat(MAT_SRC=40.0, MAT_SRE=5.0)
    # RQ = 1 + (epsdot / 40.0) ** (1/5)
    rates = np.array([0.0, 40.0, 1280.0])
    rq = law44_cowper._rate_factor(mat.params, rates)
    assert rq[0] == 1.0
    assert rq[1] == pytest.approx(2.0, rel=1e-12)
    assert rq[2] == pytest.approx(1.0 + (32.0) ** 0.2, rel=1e-12)


def test_law44_rate_factor_icc_options():
    # ICC = 1: smax = sig_max * RQ
    # ICC = 2: smax = sig_max
    mat_icc1 = _make_mat(MAT_SRC=40.0, MAT_SRE=5.0, STRFLAG=1, MAT_SIG=500.0)
    mat_icc2 = _make_mat(MAT_SRC=40.0, MAT_SRE=5.0, STRFLAG=2, MAT_SIG=500.0)

    pla = np.array([0.1])
    rq = np.array([2.0])
    fail = np.array([1.0])

    yld1, _ = law44_cowper._yield44(mat_icc1.params, pla, rq, fail)
    yld2, _ = law44_cowper._yield44(mat_icc2.params, pla, rq, fail)

    # Base yield = (300 + 450 * 0.1^0.5) * 2 = (300 + 142.302) * 2 = 884.6
    # For ICC=1, smax = 500 * 2 = 1000 > 884.6, so yld1 = 884.6
    # For ICC=2, smax = 500 < 884.6, so yld2 = 500.0
    assert yld1[0] == pytest.approx(884.604989, rel=1e-5)
    assert yld2[0] == pytest.approx(500.0, rel=1e-5)


# ----------------------------------------------------------------------------
# 3. Solid Mechanics Tests
# ----------------------------------------------------------------------------

def test_law44_solid_elastic_trial():
    mat = _make_mat()
    sig = np.zeros((1, 6))
    deps = np.array([[1e-4, 0.0, 0.0, 0.0, 0.0, 0.0]])
    epsp = np.zeros(1)
    # When density accounts for tensile uniaxial strain: mu = -1e-4
    extra = {"rho": np.array([7.8e-9 * (1.0 - 1e-4)])}
    sig_out, epsp_out, c = law44_cowper.solid_update(mat, sig, deps, epsp, 1e-4, extra)

    # Elastic: epsp unchanged
    assert epsp_out[0] == 0.0
    # Moduli
    E, nu = 210000.0, 0.3
    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    # sig_xx should match elastic uniaxial strain: E*(1-nu)/((1+nu)*(1-2nu)) * deps
    c11 = K + 4.0 * G / 3.0
    assert sig_out[0, 0] == pytest.approx(c11 * 1e-4, rel=1e-6)
    assert c[0] == pytest.approx(math.sqrt(c11 / 7.8e-9), rel=1e-12)


def test_law44_solid_plastic_yielding():
    mat = _make_mat(MAT_SRC=0.0)  # Rate-independent for clarity
    sig = np.zeros((1, 6))
    # Pure shear to yield: sig_xy
    deps = np.array([[0.0, 0.0, 0.0, 0.01, 0.0, 0.0]])
    epsp = np.zeros(1)
    sig_out, epsp_out, _ = law44_cowper.solid_update(mat, sig, deps, epsp, 1e-3)

    assert epsp_out[0] > 0.0
    vm = math.sqrt(3.0 * sig_out[0, 3] ** 2)
    # At yield: vm equals A + B*epsp^n
    expected_vm = 300.0 + 450.0 * (epsp_out[0] ** 0.5)
    # Since one-step radial return evaluates yield with previous epsp (0.0):
    assert vm == pytest.approx(300.0, rel=1e-9)


def test_law44_solid_eos_pressure_coupling():
    mat = _make_mat(MAT_SRC=0.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    epsp = np.zeros(1)
    # Density increased by 2%
    extra = {"rho": np.array([7.8e-9 * 1.02])}
    sig_out, _, _ = law44_cowper.solid_update(mat, sig, deps, epsp, 1e-3, extra)

    K = 210000.0 / (3.0 * (1.0 - 2.0 * 0.3))
    p_expected = K * 0.02
    assert sig_out[0, 0] == pytest.approx(-p_expected, rel=1e-12)
    assert sig_out[0, 1] == pytest.approx(-p_expected, rel=1e-12)
    assert sig_out[0, 2] == pytest.approx(-p_expected, rel=1e-12)
    assert sig_out[0, 3] == 0.0


def test_law44_solid_vflag1_plastic_rate():
    mat = _make_mat(Vflag=1, Fcut=1000.0, Fsmooth=1)
    sig = np.zeros((1, 6))
    deps = np.array([[0.0, 0.0, 0.0, 0.01, 0.0, 0.0]])
    epsp = np.zeros(1)
    extra = {"epsd44": np.zeros(1)}
    dt = 1e-4

    sig_out, epsp_out, _ = law44_cowper.solid_update(mat, sig, deps, epsp, dt, extra)
    assert epsp_out[0] > 0.0
    # extra["epsd44"] updated with filtered plastic strain rate
    assert extra["epsd44"][0] > 0.0


def test_law44_solid_vflag3_deviatoric_rate():
    mat = _make_mat(Vflag=3)
    sig = np.zeros((1, 6))
    # deps with volumetric + deviatoric: deps_xx = 0.002, deps_yy = -0.001, deps_zz = -0.001
    deps = np.array([[0.002, -0.001, -0.001, 0.0, 0.0, 0.0]])
    epsp = np.zeros(1)
    sig_out, epsp_out, _ = law44_cowper.solid_update(mat, sig, deps, epsp, 1e-3)
    assert epsp_out[0] > 0.0


def test_law44_solid_rate_filtering_exponential():
    mat = _make_mat(Fsmooth=1, Fcut=500.0, Vflag=2)
    sig = np.zeros((1, 6))
    deps = np.array([[1e-3, 0.0, 0.0, 0.0, 0.0, 0.0]])
    epsp = np.zeros(1)
    extra = {"epsd44": np.zeros(1)}
    dt = 1e-4
    raw_rate = 1e-3 / dt  # 10.0

    law44_cowper.solid_update(mat, sig, deps, epsp, dt, extra)
    alpha = min(1.0, 2.0 * math.pi * 500.0 * dt)
    expected_st = alpha * raw_rate
    assert extra["epsd44"][0] == pytest.approx(expected_st, rel=1e-9)


def test_law44_solid_tension_softening():
    mat = _make_mat(MAT_ETA1=0.05, MAT_ETA2=0.15)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    extra = {}
    dt = 1e-3
    # Step total strain to 0.10 (halfway between 0.05 and 0.15 -> FAIL = 0.5)
    deps = np.array([[0.10, 0.0, 0.0, 0.0, 0.0, 0.0]])
    sig_out, epsp_out, _ = law44_cowper.solid_update(mat, sig, deps, epsp, dt, extra)
    # Principal strain tracked in extra["eps44"]
    assert "eps44" in extra
    assert extra["eps44"][0, 0] == 0.10


def test_law44_solid_epsp_max_kill():
    mat = _make_mat(MAT_EPS=0.05)
    sig = np.zeros((1, 6))
    epsp = np.array([0.06])  # Already past eps_max
    deps = np.array([[0.0, 0.0, 0.0, 0.01, 0.0, 0.0]])
    sig_out, epsp_out, _ = law44_cowper.solid_update(mat, sig, deps, epsp, 1e-3)
    # Deviatoric stress should be reduced to 0 because yld=0
    assert np.allclose(sig_out, 0.0)


def test_law44_solid_dt_zero_guard():
    mat = _make_mat()
    sig = np.zeros((1, 6))
    deps = np.array([[1e-4, 0.0, 0.0, 0.0, 0.0, 0.0]])
    epsp = np.zeros(1)
    # dt = 0.0 should not raise ZeroDivisionError
    sig_out, epsp_out, c = law44_cowper.solid_update(mat, sig, deps, epsp, 0.0)
    assert np.all(np.isfinite(sig_out))
    assert np.all(np.isfinite(c))


# ----------------------------------------------------------------------------
# 4. Shell Mechanics Tests
# ----------------------------------------------------------------------------

def test_law44_shell_elastic_trial():
    mat = _make_mat()
    sig = np.zeros((1, 3))
    deps = np.array([[1e-5, 0.0, 0.0]])
    epsp = np.zeros(1)
    sig_out, epsp_out = law44_cowper.shell_update(mat, sig, deps, epsp, 1e-4)

    assert epsp_out[0] == 0.0
    fac = 210000.0 / (1.0 - 0.3 ** 2)
    assert sig_out[0, 0] == pytest.approx(fac * 1e-5, rel=1e-12)
    assert sig_out[0, 1] == pytest.approx(fac * 0.3 * 1e-5, rel=1e-12)
    assert sig_out[0, 2] == 0.0


def test_law44_shell_plane_stress_plastic_return():
    mat = _make_mat(MAT_SRC=0.0)
    sig = np.zeros((1, 3))
    deps = np.array([[0.005, 0.0, 0.0]])
    epsp = np.zeros(1)
    sig_out, epsp_out = law44_cowper.shell_update(mat, sig, deps, epsp, 1e-3)

    assert epsp_out[0] > 0.0
    sxx, syy, sxy = sig_out[0]
    vm = math.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2)
    expected_vm = 300.0 + 450.0 * (epsp_out[0] ** 0.5)
    assert vm == pytest.approx(expected_vm, rel=1e-4)


def test_law44_shell_5comp_transverse_shear():
    mat = _make_mat()
    sig = np.zeros((1, 5))
    deps = np.array([[1e-5, 0.0, 0.0, 1e-4, 2e-4]])
    epsp = np.zeros(1)
    sig_out, epsp_out = law44_cowper.shell_update(mat, sig, deps, epsp, 1e-4)

    G = 210000.0 / (2.0 * 1.3)
    assert sig_out[0, 3] == pytest.approx(G * 1e-4, rel=1e-12)
    assert sig_out[0, 4] == pytest.approx(G * 2e-4, rel=1e-12)


def test_law44_shell_sound_speed():
    mat = _make_mat()
    c = law44_cowper.shell_sound_speed(mat)
    a11 = 210000.0 / (1.0 - 0.09)
    assert c == pytest.approx(math.sqrt(a11 / 7.8e-9), rel=1e-12)


# ----------------------------------------------------------------------------
# 5. Tangent & Implicit Dispatches
# ----------------------------------------------------------------------------

def test_law44_shell_membrane_tangent():
    mat = _make_mat()
    C = shell_membrane_tangent(mat)
    assert C.shape == (3, 3)
    fac = 210000.0 / (1.0 - 0.09)
    G = 210000.0 / 2.6
    assert C[0, 0] == pytest.approx(fac, rel=1e-12)
    assert C[0, 1] == pytest.approx(fac * 0.3, rel=1e-12)
    assert C[2, 2] == pytest.approx(G, rel=1e-12)


def test_law44_consistent_solid_tangent_elastic():
    mat = _make_mat()
    sig = np.zeros((2, 6))
    epsp = np.zeros(2)
    epsp_incr = np.zeros(2)
    D = solid_tangent(mat, sig, epsp, epsp_incr)
    assert D.shape == (2, 6, 6)
    C_el = law01_elastic.solid_tangent(mat)
    assert np.allclose(D[0], C_el)
    assert np.allclose(D[1], C_el)


def test_law44_consistent_solid_tangent_plastic():
    mat = _make_mat(MAT_SRC=0.0)
    sig = np.array([[300.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    epsp = np.array([0.01])
    epsp_incr = np.array([0.001])
    D = solid_tangent(mat, sig, epsp, epsp_incr)
    assert D.shape == (1, 6, 6)
    # Check that plastic tangent is softer than elastic
    C_el = law01_elastic.solid_tangent(mat)
    assert D[0, 0, 0] < C_el[0, 0]
    # Check symmetry
    assert np.allclose(D[0], D[0].T, atol=1e-9)


def test_law44_consistent_shell_tangent_plastic():
    mat = _make_mat(MAT_SRC=0.0)
    sig = np.array([[300.0, 0.0, 0.0]])
    epsp = np.array([0.01])
    epsp_incr = np.array([0.001])
    D = shell_layer_tangent(mat, sig, epsp, epsp_incr)
    assert D.shape == (1, 3, 3)
    C_el = shell_membrane_tangent(mat)
    assert D[0, 0, 0] < C_el[0, 0]


# ----------------------------------------------------------------------------
# 6. Batched Equivalence & Empty Array Defensive Handling
# ----------------------------------------------------------------------------

def test_law44_batched_solid_vectorization():
    mat = _make_mat()
    n = 4
    sig_batch = np.zeros((n, 6))
    deps_batch = np.array([
        [1e-4, -5e-5, -5e-5, 0.0, 0.0, 0.0],
        [5e-3, -2.5e-3, -2.5e-3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1e-4, 0.0, 0.0],
        [0.0, 0.0, 0.0, 5e-3, 0.0, 0.0],
    ])
    epsp_batch = np.zeros(n)

    sig_out, epsp_out, c_out = law44_cowper.solid_update(
        mat, sig_batch.copy(), deps_batch, epsp_batch.copy(), 1e-4)

    for i in range(n):
        s_single, e_single, c_single = law44_cowper.solid_update(
            mat, np.zeros((1, 6)), deps_batch[i:i+1], np.zeros(1), 1e-4)
        assert np.allclose(sig_out[i], s_single[0], rtol=1e-10, atol=1e-12)
        assert epsp_out[i] == pytest.approx(e_single[0], rel=1e-10)
        assert c_out[i] == pytest.approx(c_single[0], rel=1e-10)


def test_law44_batched_shell_vectorization():
    mat = _make_mat()
    n = 3
    sig_batch = np.zeros((n, 5))
    deps_batch = np.array([
        [1e-4, 0.0, 0.0, 1e-5, 0.0],
        [5e-3, 0.0, 0.0, 0.0, 1e-5],
        [0.0, 0.0, 5e-3, 1e-5, 1e-5],
    ])
    epsp_batch = np.zeros(n)

    sig_out, epsp_out = law44_cowper.shell_update(
        mat, sig_batch.copy(), deps_batch, epsp_batch.copy(), 1e-4)

    for i in range(n):
        s_single, e_single = law44_cowper.shell_update(
            mat, np.zeros((1, 5)), deps_batch[i:i+1], np.zeros(1), 1e-4)
        assert np.allclose(sig_out[i], s_single[0], rtol=1e-10, atol=1e-12)
        assert epsp_out[i] == pytest.approx(e_single[0], rel=1e-10)


def test_law44_empty_arrays():
    mat = _make_mat()
    sig_empty = np.zeros((0, 6))
    deps_empty = np.zeros((0, 6))
    epsp_empty = np.zeros(0)

    s_out, e_out, c_out = law44_cowper.solid_update(
        mat, sig_empty, deps_empty, epsp_empty, 1e-4)
    assert s_out.shape == (0, 6)
    assert e_out.shape == (0,)
    assert c_out.shape == (0,)

    s_sh, e_sh = law44_cowper.shell_update(
        mat, np.zeros((0, 3)), np.zeros((0, 3)), epsp_empty, 1e-4)
    assert s_sh.shape == (0, 3)
    assert e_sh.shape == (0,)

    d_sol = law44_cowper.consistent_solid_tangent(
        mat, sig_empty, epsp_empty, epsp_empty)
    assert d_sol.shape == (0, 6, 6)

    d_sh = law44_cowper.consistent_shell_tangent(
        mat, np.zeros((0, 3)), epsp_empty, epsp_empty)
    assert d_sh.shape == (0, 3, 3)

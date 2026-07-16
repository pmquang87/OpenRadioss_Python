"""
M33 validations: NON-GAUSSIAN JOINT-TENSOR DISTRIBUTION
(/IMPL/FATIG/NGAUSS + /JOINT + /WVILLE) — a VECTOR (multivariate) Winterstein-Hermite /
translation-process transform of the CORRELATED 6x6 stress-tensor process, the target
kurtosis (and skewness) imposed JOINTLY on the tensor COMPONENTS (preserving the full
6x6 covariance / cross-PSD), NOT on the already-resolved equivalent scalar. Where
M24/M32 imposed gamma_4 on the von-Mises / critical-plane SCALAR (a scalar Hermite
transform), M33 imposes it on the TENSOR and lets the critical-plane / von-Mises
reduction INHERIT the INDUCED non-Gaussianity from the joint tensor statistics —
applied along the M31/M32 CONTINUOUS Wigner-Ville instantaneous spectrum, reduced per
instant and Palmgren-Miner INTEGRATED, cross-validated by a MULTIVARIATE non-Gaussian
non-stationary Monte-Carlo. Built ALONGSIDE the M20-M32 spectral fatigue (all stay
bit-identical; the M24/M32 equivalent-scalar and the M27/M31 Gaussian tensor paths stay
byte-identical, the JOINT non-Gaussian tensor path is a NEW path ALONGSIDE them).

Every new capability gets at least one ANALYTIC / EXACT-REDUCTION check (the port's
philosophy):

THE M33 <-> M32 / M31 REDUCTIONS (built in, exact)
* gamma_4_c == 3 on EVERY component (a Gaussian tensor) -> every component transform is
  the IDENTITY, the induced projection kurtosis is 3 EXACTLY, lambda_ng == 1, and the
  M33 answer DELEGATES to the M31/M27 Gaussian tensor answer BYTE-IDENTICALLY;
* the SCALAR-EQUIVALENT limit (kurtosis imposed on the RESOLVED scalar rather than the
  tensor components) recovers the M32/M24 correction EXACTLY — M33 reports the M32
  answer as the ``scalar_equivalent`` side-by-side entry by DELEGATION (byte-identical
  to a direct M32 call);
* a UNIAXIAL (single-component) projection's INDUCED kurtosis reduces EXACTLY to the
  M24 realised kurtosis ``hermite_kurtosis`` of that component's transform (the scalar
  Hermite transform recovered as the 1-component special case);
* the induced-kurtosis closed form matches the MULTIVARIATE Monte-Carlo.

THE JOINT POINT (the M33 <-> M32 boundary)
* per-component kurtoses that DIFFER give an induced resolved-plane kurtosis (and
  damage) that DIFFERS from imposing gamma_4 directly on the scalar — the genuinely
  JOINT case, distinct from the M32 equivalent-scalar correction.

CARDS + NO-REGRESSION (the M7 parity contract)
* /IMPL/FATIG/NGAUSS/JOINT/WVILLE with a per-component kurtosis line (cols 4..9 of the
  M24 kurtosis line); a card WITHOUT the per-component line never triggers M33 (the M32
  path is untouched);
* the M24/M32 equivalent-scalar AND the M27/M31 Gaussian tensor answers byte-identical
  whether or not the M33 joint path runs.

See PORTING_GUIDE.md roadmap M33.
"""

import contextlib
import io
import os
import tempfile

import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.starter.starter import run_starter

pytest.importorskip("scipy")   # implicit requires scipy (optional otherwise)

from pyradioss.common.messages import MessageLog                            # noqa: E402
from pyradioss.implicit import joint_nongaussian_fatigue as jng             # noqa: E402
from pyradioss.implicit import nongaussian_wigner_ville_fatigue as ngwv     # noqa: E402
from pyradioss.implicit import wigner_ville_fatigue as wv                   # noqa: E402
from pyradioss.implicit import nongaussian_fatigue as ngf                   # noqa: E402
from pyradioss.implicit import joint_evolutionary_fatigue as jf            # noqa: E402
from pyradioss.implicit.multiaxial_fatigue import (                        # noqa: E402
    tensor_moment_matrices, critical_plane_search, synthesize_multiaxial_history)

from tests.test_m21_multiaxfatig import _brick_deck                        # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _run(starter_text, engine_text, capture=False):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M33_0000.rad")
    ep = os.path.join(d, "M33_0001.rad")
    with open(sp, "w") as f:
        f.write(starter_text)
    with open(ep, "w") as f:
        f.write(engine_text)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_starter(sp)
        model = run_engine(ep)
    return (model, buf.getvalue()) if capture else model


_MISSION = ("/FUNCT/20\nmission\n       0.0       0.5\n"
            "     100.0       1.5\n     200.0       1.5\n"
            "     300.0       0.3\n")


def _with(deck, *extra):
    return deck.replace("/END", "".join(extra) + "/END", 1)


def _tensor_cross_psd(nf=400, fmax=600.0):
    """A synthetic 6x6 stress-tensor cross-PSD with several correlated components (so a
    genuinely JOINT per-component kurtosis has structure to combine through)."""
    f = np.linspace(1e-3, fmax, nf)
    omega = 2.0 * np.pi * f
    S = np.zeros((nf, 6, 6), dtype=complex)
    low = np.exp(-((f - 100.0) ** 2) / (2.0 * 20.0 ** 2))
    high = np.exp(-((f - 450.0) ** 2) / (2.0 * 30.0 ** 2))
    S[:, 0, 0] = low + 0.05
    S[:, 3, 3] = high + 0.02
    S[:, 1, 1] = 0.4 * low + 0.2 * high + 0.01
    S[:, 0, 3] = 0.3 * np.sqrt((low + 0.05) * (high + 0.02))
    S[:, 3, 0] = np.conj(S[:, 0, 3])
    S[:, 0, 1] = 0.4 * low
    S[:, 1, 0] = np.conj(S[:, 0, 1])
    return omega, S


def _parse(text):
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.engine_keywords import parse_engine_deck
    d = tempfile.mkdtemp()
    p = os.path.join(d, "E_0001.rad")
    with open(p, "w") as f:
        f.write(text)
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        ec = parse_engine_deck(read_deck(p), log)
    return ec, log


# ----------------------------------------------------------------------------
# The M33 primitives: the per-component schedule + the diagram moment tables
# ----------------------------------------------------------------------------

def test_component_kurtosis_schedule_forms():
    s = np.linspace(0.0, 1.0, 5)
    # scalar -> same on every component, held constant
    g4, g3 = jng.component_kurtosis_schedule(s, 6.0)
    assert g4.shape == (5, 6) and np.allclose(g4, 6.0) and np.allclose(g3, 0.0)
    # per-component (6,) constant
    kc = np.array([9.0, 5.0, 3.0, 7.0, 3.0, 3.0])
    g4, _ = jng.component_kurtosis_schedule(s, kc)
    assert np.allclose(g4[0], kc) and np.allclose(g4[-1], kc)
    # linear sweep (start, end), same on every component
    g4, _ = jng.component_kurtosis_schedule(s, (3.0, 8.0))
    assert np.allclose(g4[0], 3.0) and np.allclose(g4[-1], 8.0)
    # per-instant per-component grid (nt, 6)
    grid = np.random.default_rng(0).uniform(3.0, 9.0, size=(5, 6))
    g4, _ = jng.component_kurtosis_schedule(s, 3.0, kurt_grid=grid)
    assert np.array_equal(g4, grid)
    # Gaussian detection
    assert jng._is_gaussian_component_schedule(np.full((5, 6), 3.0),
                                               np.zeros((5, 6)))
    assert not jng._is_gaussian_component_schedule(grid, np.zeros((5, 6)))


def test_single_component_induced_kurtosis_equals_m24():
    """A UNIAXIAL projection (one nonzero component) recovers the M24 SCALAR Hermite
    transform EXACTLY: the induced kurtosis IS the M24 realised kurtosis
    ``hermite_kurtosis`` of that component's transform (the scalar Hermite transform as
    the 1-component special case of the joint diagram formula)."""
    M0 = np.diag([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    proj = np.array([1.3, 0.0, 0.0, 0.0, 0.0, 0.0])
    for g3t, g4t in ((0.0, 7.0), (0.0, 5.0), (0.4, 6.0), (-0.3, 4.5)):
        var, sk, ku = jng.induced_projection_moments(M0, proj, g3t, g4t)
        sk24, ku24 = ngf.hermite_kurtosis(g3t, g4t)
        assert var == pytest.approx(1.3 ** 2 * 2.0, rel=1e-12)
        assert ku == pytest.approx(ku24, rel=1e-10)
        assert sk == pytest.approx(sk24, rel=1e-9, abs=1e-9)


def test_gaussian_components_induced_kurtosis_is_three():
    """gamma_4_c == 3, gamma_3_c == 0 on every component -> the induced projection
    kurtosis is EXACTLY 3 and the skewness EXACTLY 0 (a Gaussian tensor projects to a
    Gaussian scalar)."""
    omega, S = _tensor_cross_psd()
    M = tensor_moment_matrices(omega, S, nmax=1)
    cp = critical_plane_search(M, method="shear")
    var, sk, ku = jng.induced_projection_moments(M[0], cp["proj"], 0.0, 3.0)
    assert ku == pytest.approx(3.0, abs=1e-9)
    assert sk == pytest.approx(0.0, abs=1e-9)


@pytest.mark.slow          # measured 76.6 s serial (2026-07 full-suite run)
def test_induced_kurtosis_matches_multivariate_monte_carlo():
    """The closed-form induced (variance, skewness, kurtosis) of the resolved plane
    matches a STATIONARY multivariate non-Gaussian Monte-Carlo: synthesise the
    correlated Gaussian components, push EACH through its own Hermite transform (the
    VECTOR transform), project onto the fixed plane, and compare sample moments."""
    omega, S = _tensor_cross_psd()
    freqs = omega / (2.0 * np.pi)
    M = tensor_moment_matrices(omega, S, nmax=1)
    M0 = M[0]
    cp = critical_plane_search(M, method="shear")
    proj = cp["proj"]
    g4 = np.array([9.0, 5.0, 3.0, 7.0, 3.0, 3.0])
    g3 = np.array([0.0, 0.4, 0.0, -0.3, 0.0, 0.0])
    var, sk, ku = jng.induced_projection_moments(M0, proj, g3, g4)

    t, X = synthesize_multiaxial_history(freqs, S, 6000.0, 17, fs=2400.0)
    sig = np.sqrt(np.clip(np.diag(M0), 0.0, None))
    for c in range(6):
        coeffs = ngf.hermite_coefficients(g3[c], g4[c])
        h3, h4, kappa = coeffs
        if abs(h3) < 1e-15 and abs(h4) < 1e-15 and abs(kappa - 1.0) < 1e-15:
            continue
        s_ = float(np.std(X[:, c]))
        if s_ <= 0.0:
            continue
        z = (X[:, c] - X[:, c].mean()) / s_
        X[:, c] = s_ * ngf.hermite_transform(z, g3[c], g4[c], coeffs=coeffs)
    sp = X @ proj
    spc = sp - sp.mean()
    v = float(np.mean(spc ** 2))
    ku_mc = float(np.mean(spc ** 4) / v ** 2)
    sk_mc = float(np.mean(spc ** 3) / v ** 1.5)
    assert var == pytest.approx(v, rel=0.03)
    assert ku == pytest.approx(ku_mc, rel=0.06)
    assert sk == pytest.approx(sk_mc, rel=0.15, abs=0.02)
    assert ku > 3.5                                  # genuinely leptokurtic


# ----------------------------------------------------------------------------
# The M33 <-> M31 / M32 reductions (byte-identity / exact)
# ----------------------------------------------------------------------------

def test_gaussian_tensor_limit_byte_identical_to_m31():
    """gamma_4_c == 3 on every component: the M33 joint summary DELEGATES to the M31
    continuous Gaussian tensor summary BYTE-IDENTICALLY (per reduction)."""
    omega, S = _tensor_cross_psd()
    m, C = 5.0, 1e4
    durs = [10.0] * 6
    g = jng.joint_nongaussian_tensor_summary(
        omega, S, durs, (120.0, 480.0), (20.0, 40.0), m, C, kurt=3.0, refine=6,
        smooth=0.0)
    gm31 = wv.wigner_ville_tensor_summary(
        omega, S, durs, (120.0, 480.0), (20.0, 40.0), m, C, refine=6, smooth=0.0)
    assert g["delegated"] == "m31_gaussian"
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert g[k]["damage_rate"] == gm31[k]["damage_rate"]


def test_scalar_equivalent_byte_identical_to_m32():
    """The ``scalar_equivalent`` side-by-side entry (kurtosis imposed on the RESOLVED
    scalar) is BYTE-IDENTICAL to a direct M32 tensor call with the representative scalar
    target — the scalar-equivalent limit recovering the M32/M24 correction EXACTLY."""
    omega, S = _tensor_cross_psd()
    m, C = 5.0, 1e4
    durs = [10.0] * 6
    kc = np.array([9.0, 5.0, 3.0, 7.0, 3.0, 3.0])
    g = jng.joint_nongaussian_tensor_summary(
        omega, S, durs, (120.0, 480.0), (20.0, 40.0), m, C, kurt=kc, refine=6,
        smooth=0.0, scalar_equivalent=True)
    k_rep, s_rep = jng._representative_scalar_kurtosis(kc, 0.0)
    m32 = ngwv.nongaussian_wigner_ville_tensor_summary(
        omega, S, durs, (120.0, 480.0), (20.0, 40.0), m, C, k_rep, skew=s_rep,
        refine=6, smooth=0.0)
    se = g["scalar_equivalent"]
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert se[k]["damage_rate"] == m32[k]["damage_rate"]


def test_joint_differs_from_scalar_equivalent():
    """The M33 <-> M32 BOUNDARY: with per-component kurtoses that DIFFER, the induced
    resolved-plane kurtosis (and damage) DIFFERS from imposing the (max) gamma_4
    directly on the scalar — the projection / correlation combine the per-component
    non-Gaussianity into a genuinely JOINT induced kurtosis distinct from the scalar
    surrogate."""
    omega, S = _tensor_cross_psd()
    m, C = 5.0, 1e4
    durs = [10.0] * 6
    kc = np.array([9.0, 5.0, 3.0, 7.0, 3.0, 3.0])
    g = jng.joint_nongaussian_tensor_summary(
        omega, S, durs, (120.0, 480.0), (20.0, 40.0), m, C, kurt=kc, refine=6,
        smooth=0.0, scalar_equivalent=True)
    se = g["scalar_equivalent"]
    # the JOINT resolved-plane damage differs from the equivalent-scalar one
    for k in ("normal_plane", "shear_plane"):
        assert g[k]["damage_rate"] != se[k]["damage_rate"]
    # and the joint path amplifies above the Gaussian baseline (leptokurtic components)
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert g[k]["damage_rate"] >= g[k]["gaussian_damage_rate"]
    assert g["induced_kurt_max"] > 3.0
    assert g["lambda_max"] >= g["lambda_min"] >= 0.9
    assert g["preservation_error"] >= 0.0
    assert not g.get("delegated")


def test_windowed_limit_runs_and_reduces():
    """refine = 1, smooth = 0 (the windowed limit): the per-instant reduction rebuilds
    the M27 per-window Gaussian moment matrices (byte-identical to M27's Gaussian rate)
    before the induced lambda_ng scales it — the joint correction on the M27 windowed
    joint-tensor."""
    omega, S = _tensor_cross_psd()
    m, C = 5.0, 1e4
    durs = [10.0] * 6
    kc = np.array([9.0, 5.0, 3.0, 7.0, 3.0, 3.0])
    g = jng.joint_nongaussian_tensor_summary(
        omega, S, durs, (120.0, 480.0), (20.0, 40.0), m, C, kurt=kc, refine=1,
        smooth=0.0, scalar_equivalent=False)
    m27 = jf.joint_evolutionary_fatigue_summary(
        omega, S, durs, fc=(120.0, 480.0), bw=(20.0, 40.0), m=m, C=C, drift=True)
    for k in ("von_mises", "normal_plane", "shear_plane"):
        # the Gaussian per-window reduction is byte-identical to M27
        assert g[k]["gaussian_damage_rate"] == m27[k]["damage_rate"]
        # the joint non-Gaussian rate scales it up (leptokurtic)
        assert g[k]["damage_rate"] >= g[k]["gaussian_damage_rate"]


# ----------------------------------------------------------------------------
# The multivariate non-Gaussian Monte-Carlo
# ----------------------------------------------------------------------------

def test_mc_gaussian_tensor_limit_bit_identical_to_m27_mc():
    """The Gaussian-tensor non-Gaussian Monte-Carlo (gamma_4_c == 3) is the M27/M31
    multivariate Monte-Carlo damage BIT-IDENTICALLY (every component transform is the
    identity, so the synthesised record is the M27 multivariate record; the same fixed
    plane and whole-record rainflow)."""
    omega, S = _tensor_cross_psd()
    m, C = 5.0, 1e4
    durs = [300.0]
    mc = jng.joint_nongaussian_monte_carlo_damage(
        omega, S, durs, 0.0, 0.0, m, C, seed=5, kurt=3.0, refine=1, smooth=0.0,
        fs=2400.0, reduction="shear_plane")
    m27mc = jf.joint_evolutionary_monte_carlo_damage(
        omega, S, durs, 0.0, 0.0, m, C, seed=5, fs=2400.0, reduction="shear_plane")
    assert mc["damage_rate"] == pytest.approx(m27mc["damage_rate"], rel=1e-12)


def test_mc_joint_induced_kurtosis_leptokurtic():
    """The multivariate non-Gaussian Monte-Carlo's resolved-projection sample kurtosis
    is leptokurtic (well above the Gaussian 3) and its damage exceeds the Gaussian
    multivariate MC when the tensor components carry a strong per-component kurtosis."""
    omega, S = _tensor_cross_psd()
    m, C = 5.0, 1e4
    durs = [300.0]
    kc = np.array([9.0, 7.0, 6.0, 8.0, 5.0, 5.0])
    mc = jng.joint_nongaussian_monte_carlo_damage(
        omega, S, durs, 0.0, 0.0, m, C, seed=5, kurt=kc, refine=1, smooth=0.0,
        fs=2400.0, reduction="shear_plane")
    m27mc = jf.joint_evolutionary_monte_carlo_damage(
        omega, S, durs, 0.0, 0.0, m, C, seed=5, fs=2400.0, reduction="shear_plane")
    assert mc["kurtosis"] > 3.5
    assert mc["damage_rate"] > m27mc["damage_rate"]


# ----------------------------------------------------------------------------
# Cards
# ----------------------------------------------------------------------------

def test_impl_fatig_joint_ngauss_card_parsing():
    """/IMPL/FATIG/NGAUSS/JOINT/WVILLE reads up to 6 PER-COMPONENT target kurtoses on
    the trailing columns 4..9 of the M24 kurtosis line (kurt skew kfunct kurt1 k_xx ..
    k_zx); a card WITHOUT the per-component line leaves impl_fatig_joint_kurt empty (no
    M33 trigger — the M32 path untouched)."""
    ec, _ = _parse(
        "#\n/RUN/A/1\n1.0\n/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS\n"
        "0.0 400 500 10\n5.0 1e4 0.03\n3.0 0.0 0 0 9.0 5.0 3.0 7.0 3.0 3.0\n"
        "50.0 250.0 10.0 40.0 6 6\n/END\n")
    assert ec.impl_fatig_joint and ec.impl_fatig_ngauss and ec.impl_fatig_wville
    assert tuple(ec.impl_fatig_joint_kurt) == (9.0, 5.0, 3.0, 7.0, 3.0, 3.0)
    # fewer than 6 values pad with the scalar kurt (col 0)
    ec, _ = _parse(
        "#\n/RUN/A/1\n1.0\n/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS\n"
        "0.0 400 500 10\n5.0 1e4 0.03\n4.0 0.0 0 0 9.0 5.0\n"
        "50.0 250.0 10.0 40.0 6 6\n/END\n")
    assert tuple(ec.impl_fatig_joint_kurt) == (9.0, 5.0, 4.0, 4.0, 4.0, 4.0)
    # a JOINT card WITHOUT the per-component line (the M32 tensor card) -> empty
    ec, _ = _parse(
        "#\n/RUN/A/1\n1.0\n/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS\n"
        "0.0 400 500 10\n5.0 1e4 0.03\n3.0 0.0 0 8.0\n"
        "50.0 250.0 10.0 40.0 6 8\n/END\n")
    assert tuple(ec.impl_fatig_joint_kurt) == ()
    assert ec.impl_fatig_kurt1 == 8.0


# ----------------------------------------------------------------------------
# End-to-end + reporting (model.implicit_result)
# ----------------------------------------------------------------------------

def test_joint_nongaussian_tensor_end_to_end():
    """/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS with a per-component kurtosis line: the
    continuous JOINT-tensor non-Gaussian distribution with a per-instant plane re-search
    AND the induced-kurtosis lambda_ng, reported ALONGSIDE the M32 equivalent-scalar and
    the M31/M27 Gaussian tensor; the ``wigner_ville`` entry carries a
    ``joint_nongaussian`` sub-entry."""
    eng = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n"
           "/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS/NSTAT\n"
           "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000\n"
           "3.0 0.0 0 0 9.0 5.0 3.0 7.0 3.0 3.0\n"
           "20 6\n50.0 250.0 10.0 40.0 6 6 0.0\n/PRINT/-500\n/STOP\n15.0\n")
    m, out = _run(_with(_brick_deck(nx=3), _MISSION), eng, capture=True)
    wvr = m.implicit_result.fatigue["wigner_ville"]
    jn = wvr["joint_nongaussian"]
    assert jn is not None and jn["tensor"]
    assert tuple(jn["joint_kurt"]) == (9.0, 5.0, 3.0, 7.0, 3.0, 3.0)
    assert jn["induced_kurt_max"] >= 3.0
    assert jn["lambda_max"] >= jn["lambda_min"]
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert jn[key]["damage_rate"] >= jn["gaussian"][key]
    # the M32 equivalent-scalar entry is present and left byte-identical
    assert wvr.get("nongaussian") is not None
    assert jn["scalar_equivalent"] is not None
    assert jn["monte_carlo"] is not None
    assert "JOINT-TENSOR NON-GAUSSIAN DISTRIBUTION" in out
    assert "INDUCED CRITICAL-PLANE" in out
    assert "TIME-VARYING NON-GAUSSIAN INSTANTANEOUS" in out   # M32 alongside


def test_m31_m32_byte_identical_with_without_joint_ng():
    """The M27/M31 Gaussian tensor AND the M32 equivalent-scalar answers are
    BYTE-IDENTICAL whether or not the M33 joint path runs — the M33 path is NEW and
    ALONGSIDE (the M7 parity contract extended to M33)."""
    # both cards carry the SAME M32 equivalent-scalar input (a constant kurtosis 6.0 on
    # cols 0..3); they differ ONLY in the M33 per-component line (cols 4..9), so the M32
    # nongaussian number must be byte-identical
    base = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n"
            "/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS/NSTAT\n"
            "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000\n"
            "6.0 0.0\n"
            "20 6\n50.0 250.0 10.0 40.0 6 6 0.0\n/PRINT/-500\n/STOP\n15.0\n")
    m33 = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n"
           "/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS/NSTAT\n"
           "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000\n"
           "6.0 0.0 0 0 9.0 5.0 3.0 7.0 3.0 3.0\n"
           "20 6\n50.0 250.0 10.0 40.0 6 6 0.0\n/PRINT/-500\n/STOP\n15.0\n")
    deck = _with(_brick_deck(nx=3), _MISSION)
    a = _run(deck, base)
    b = _run(deck, m33)
    fa = a.implicit_result.fatigue
    fb = b.implicit_result.fatigue
    # the M31 Gaussian-continuous tensor byte-identical
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert (fa["wigner_ville"][k]["damage_rate"]
                == fb["wigner_ville"][k]["damage_rate"])
    # the M32 equivalent-scalar (nongaussian) sub-entry byte-identical
    assert (fa["wigner_ville"]["nongaussian"]["damage_rate"]
            == fb["wigner_ville"]["nongaussian"]["damage_rate"])
    # M33 present only in the second run
    assert fa["wigner_ville"].get("joint_nongaussian") is None
    assert fb["wigner_ville"]["joint_nongaussian"] is not None


def test_joint_nongaussian_does_not_mutate_state():
    """The M33 joint non-Gaussian path is read-only in the element state (the M14-M32
    parity contract extended to M33)."""
    starter = _brick_deck(nx=3)
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M33_0000.rad")
    with open(sp, "w") as f:
        f.write(starter)
    with contextlib.redirect_stdout(io.StringIO()):
        m = run_starter(sp)
    before = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in m.bricks.state.items()}
    omega, S = _tensor_cross_psd()
    jng.joint_nongaussian_tensor_summary(
        omega, S, [1.0] * 6, (120.0, 480.0), (20.0, 40.0), 5.0, 1e4,
        kurt=np.array([9.0, 5.0, 3.0, 7.0, 3.0, 3.0]), refine=6, smooth=0.0)
    for k, v in before.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(m.bricks.state[k], v), k

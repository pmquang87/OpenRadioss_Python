"""
M34 validations: EXACT TRANSLATION-PROCESS CORRELATION-DISTORTION INVERSION
(/IMPL/FATIG/NGAUSS + /JOINT + /WVILLE + /EXACT) — the Grigoriu / Nataf / Cario-Nelson
NORTA correlation-matching inversion that solves the underlying-Gaussian correlation
rho^U per component pair so the component-wise Winterstein-Hermite (translation)
transform of the correlated 6x6 stress tensor reproduces the TARGET 6x6 covariance
EXACTLY (not merely to M33's leading order), driving the covariance preservation error to
~0. Applied along the M31/M32 continuous spectrum, reduced per instant and Miner-
INTEGRATED, cross-validated by the M33 multivariate non-Gaussian Monte-Carlo synthesised
with the CORRECTED underlying correlation. Built ALONGSIDE the M20-M33 spectral fatigue
(all stay bit-identical; the M33 leading-order joint path, the M24/M32 equivalent-scalar
and the M27/M31 Gaussian tensor paths stay byte-identical — the EXACT path is a NEW path
ALONGSIDE them).

Every new capability gets at least one ANALYTIC / EXACT-REDUCTION check (the port's
philosophy):

THE M34 <-> M33 / M31 REDUCTIONS (built in, exact)
* the LEADING-ORDER limit (small non-Gaussianity, gamma4 -> 3) recovers the M33
  underlying correlation rho^U -> R EXACTLY (the cubic collapses to phi(rho) = rho);
* gamma4_c == 3 on every component (a Gaussian tensor) -> rho^U == R EXACTLY, the
  preservation error is 0, and the M34 path DELEGATES to the M33/M31/M27 Gaussian answer
  BYTE-IDENTICALLY;
* the transformed covariance matches the target 6x6 to MACHINE PRECISION
  (preservation_error -> ~0) where M33's leading-order distortion is nonzero;
* a UNIAXIAL / SCALAR-equivalent limit recovers the M32/M24 correction (through the
  M33 scalar_equivalent delegation, left byte-identical);
* the corrected Monte-Carlo's SAMPLE covariance matches the target where M33's drifted.

CARDS + NO-REGRESSION (the M7 parity contract)
* /IMPL/FATIG/NGAUSS/JOINT/WVILLE/EXACT (or /NORTA) sets impl_fatig_exact; a card WITHOUT
  the flag leaves it False (no M34 trigger — the M33 leading-order path untouched);
* the M33 leading-order joint, the M32 equivalent-scalar AND the M31/M27 Gaussian answers
  byte-identical whether or not the M34 exact path runs.

See PORTING_GUIDE.md roadmap M34.
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
from pyradioss.implicit import nongaussian_fatigue as ngf                   # noqa: E402
from pyradioss.implicit import joint_evolutionary_fatigue as jf            # noqa: E402
from pyradioss.implicit.multiaxial_fatigue import (                        # noqa: E402
    tensor_moment_matrices, critical_plane_search, synthesize_multiaxial_history)

from tests.test_m21_multiaxfatig import _brick_deck                        # noqa: E402
from tests.test_m33_jointng import _tensor_cross_psd, _MISSION, _with       # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _run(starter_text, engine_text, capture=False):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M34_0000.rad")
    ep = os.path.join(d, "M34_0001.rad")
    with open(sp, "w") as f:
        f.write(starter_text)
    with open(ep, "w") as f:
        f.write(engine_text)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_starter(sp)
        model = run_engine(ep)
    return (model, buf.getvalue()) if capture else model


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
# The M34 primitives: the per-pair cubic solver + Higham repair
# ----------------------------------------------------------------------------

def test_solve_pair_rho_linear_and_cubic():
    """The per-pair correlation-matching solver (theory eq. (2')): for two Gaussian
    components (A2 = A3 = 0) rho^U = target EXACTLY; for the cubic it recovers the
    physical near-identity root and reproduces the target correlation to machine
    precision."""
    # linear (Gaussian pair): phi(rho) = rho
    r, res, ok = jng._solve_pair_rho(1.0, 0.0, 0.0, 0.4)
    assert r == pytest.approx(0.4, abs=1e-12) and ok and abs(res) < 1e-12
    # a leptokurtic pair: solve phi(rho) = target, verify phi(rho) == target
    h3, h4, kappa = jng.component_hermite_coefficients([0.0, 0.0], [9.0, 7.0])
    A1 = kappa[0] * kappa[1]
    A2 = A1 * 2.0 * h3[0] * h3[1]
    A3 = A1 * 6.0 * h4[0] * h4[1]
    for target in (0.2, 0.5, -0.4, 0.8):
        r, res, ok = jng._solve_pair_rho(A1, A2, A3, target)
        phi = A1 * r + A2 * r ** 2 + A3 * r ** 3
        assert -1.0 <= r <= 1.0
        assert phi == pytest.approx(target, abs=1e-10)


def test_higham_nearest_correlation_is_valid_pd():
    """The Higham (2002) nearest-correlation repair returns a symmetric, unit-diagonal,
    POSITIVE-DEFINITE matrix; an already-valid correlation matrix passes through
    essentially unchanged."""
    A = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    H = jng._higham_nearest_correlation(A)
    assert np.allclose(np.diag(H), 1.0)
    assert np.allclose(H, H.T)
    assert np.linalg.eigvalsh(H).min() > 0.0
    # already-valid matrix unchanged
    V = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    assert np.allclose(jng._higham_nearest_correlation(V), V, atol=1e-8)


def test_underlying_correlation_exact_preservation():
    """The NORTA inversion drives the transformed-covariance preservation error to
    MACHINE PRECISION where the M33 leading-order distortion is nonzero — the M34
    payload."""
    omega, S = _tensor_cross_psd()
    M0 = tensor_moment_matrices(omega, S, nmax=1)[0]
    g4 = np.array([9.0, 5.0, 3.0, 7.0, 3.0, 3.0])
    g3 = np.zeros(6)
    sol = jng.solve_underlying_correlation(M0, g3, g4)
    assert sol["preservation_error"] < 1e-12               # exact ~ machine precision
    assert sol["preservation_error_leading"] > 1e-4        # M33 leading-order nonzero
    # rho^U is a valid PD correlation on the support
    idx = np.where(sol["support"])[0]
    sub = sol["rho_u"][np.ix_(idx, idx)]
    assert np.linalg.eigvalsh(sub).min() > 0.0
    assert np.allclose(np.diag(sol["rho_u"]), 1.0)


def test_gaussian_tensor_underlying_equals_target():
    """gamma4_c == 3 on every component -> rho^U == R EXACTLY and the preservation error
    is 0 (the delegation guarantee — every component transform is the identity)."""
    omega, S = _tensor_cross_psd()
    M0 = tensor_moment_matrices(omega, S, nmax=1)[0]
    sol = jng.solve_underlying_correlation(M0, np.zeros(6), np.full(6, 3.0))
    assert np.allclose(sol["rho_u"], sol["target_R"], atol=1e-12)
    assert sol["preservation_error"] < 1e-14
    assert sol["preservation_error_leading"] < 1e-14


def test_leading_order_limit_recovers_m33_correlation():
    """The SMALL-non-Gaussianity limit (gamma4 -> 3) recovers the M33 underlying
    correlation rho^U -> R (the target): as the excess kurtosis shrinks, the NORTA
    solution converges to the target correlation the M33 leading-order model uses."""
    omega, S = _tensor_cross_psd()
    M0 = tensor_moment_matrices(omega, S, nmax=1)[0]
    prev = None
    for eps in (0.5, 0.1, 1e-2, 1e-4):
        g4 = np.full(6, 3.0 + eps)
        sol = jng.solve_underlying_correlation(M0, np.zeros(6), g4)
        idx = np.where(sol["support"])[0]
        dev = np.max(np.abs(sol["rho_u"][np.ix_(idx, idx)]
                            - sol["target_R"][np.ix_(idx, idx)]))
        if prev is not None:
            assert dev < prev                              # monotone convergence to R
        prev = dev
    assert prev < 1e-6                                     # rho^U -> R in the limit


def test_induced_underlying_R_equals_M33():
    """Passing ``underlying_R = R`` (the target correlation) to
    ``induced_projection_moments`` reproduces the M33 leading-order answer BYTE-
    IDENTICALLY — the exact path is a strict generalisation."""
    omega, S = _tensor_cross_psd()
    M = tensor_moment_matrices(omega, S, nmax=1)
    M0 = M[0]
    proj = critical_plane_search(M, method="normal")["proj"]
    g4 = np.array([9.0, 5.0, 3.0, 7.0, 3.0, 3.0])
    g3 = np.zeros(6)
    m33 = jng.induced_projection_moments(M0, proj, g3, g4)
    sol = jng.solve_underlying_correlation(M0, g3, g4)
    same = jng.induced_projection_moments(M0, proj, g3, g4,
                                          underlying_R=sol["target_R"])
    assert same == m33                                     # byte-identical


def test_uniaxial_induced_kurtosis_equals_m24_exact():
    """A UNIAXIAL projection recovers the M24 scalar Hermite transform EXACTLY under the
    exact path too (the correlation is trivial 1x1, so exact == leading == M24)."""
    M0 = np.diag([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    proj = np.array([1.3, 0.0, 0.0, 0.0, 0.0, 0.0])
    for g3t, g4t in ((0.0, 7.0), (0.4, 6.0)):
        sol = jng.solve_underlying_correlation(M0, g3t, g4t)
        _, sk, ku = jng.induced_projection_moments(M0, proj, g3t, g4t,
                                                   underlying_R=sol["rho_u"])
        sk24, ku24 = ngf.hermite_kurtosis(g3t, g4t)
        assert ku == pytest.approx(ku24, rel=1e-10)
        assert sk == pytest.approx(sk24, rel=1e-9, abs=1e-9)


# ----------------------------------------------------------------------------
# The exact-covariance joint-tensor SUMMARY (byte-identity / exact reductions)
# ----------------------------------------------------------------------------

def test_exact_summary_preservation_and_gaussian_delegation():
    """The exact-covariance summary reports a preservation error ~0 (with the M33
    leading-order value alongside), and in the Gaussian-tensor limit DELEGATES to the M31
    Gaussian answer byte-identically whether exact or not."""
    omega, S = _tensor_cross_psd()
    m, C = 5.0, 1e4
    durs = [10.0] * 6
    kc = np.array([9.0, 5.0, 3.0, 7.0, 3.0, 3.0])
    ex = jng.joint_nongaussian_tensor_summary(
        omega, S, durs, (120.0, 480.0), (20.0, 40.0), m, C, kurt=kc, refine=6,
        smooth=0.0, scalar_equivalent=False, exact=True)
    assert ex["exact"] is True
    assert ex["preservation_error"] < 1e-10
    assert ex["preservation_error_leading"] > 1e-4
    # Gaussian limit delegates identically for exact and leading
    for ex_flag in (False, True):
        g = jng.joint_nongaussian_tensor_summary(
            omega, S, durs, (120.0, 480.0), (20.0, 40.0), m, C, kurt=3.0, refine=6,
            smooth=0.0, exact=ex_flag)
        assert g["delegated"] == "m31_gaussian"
        assert g["preservation_error"] == 0.0


def test_exact_leading_gaussian_scalar_byte_identical():
    """The M33 leading-order joint, the M31 Gaussian and the M32 scalar-equivalent
    entries are BYTE-IDENTICAL whether the exact path runs or not — the exact path is a
    NEW path ALONGSIDE them (the M7 parity contract extended to M34)."""
    omega, S = _tensor_cross_psd()
    m, C = 5.0, 1e4
    durs = [10.0] * 6
    kc = np.array([9.0, 5.0, 3.0, 7.0, 3.0, 3.0])
    lead = jng.joint_nongaussian_tensor_summary(
        omega, S, durs, (120.0, 480.0), (20.0, 40.0), m, C, kurt=kc, refine=6,
        smooth=0.0, scalar_equivalent=True, exact=False)
    ex = jng.joint_nongaussian_tensor_summary(
        omega, S, durs, (120.0, 480.0), (20.0, 40.0), m, C, kurt=kc, refine=6,
        smooth=0.0, scalar_equivalent=True, exact=True)
    # the M31 Gaussian tensor reductions byte-identical
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert lead["gaussian"][k]["damage_rate"] == ex["gaussian"][k]["damage_rate"]
    # the M32 scalar-equivalent byte-identical
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert (lead["scalar_equivalent"][k]["damage_rate"]
                == ex["scalar_equivalent"][k]["damage_rate"])


def test_exact_differs_from_leading_on_sensitive_plane():
    """The exact-covariance induced kurtosis / damage DIFFERS from the M33 leading-order
    one on a plane whose resolved scalar is sensitive to the corrected cross-structure —
    the exact path is a genuinely different (covariance-exact) answer, still leptokurtic
    above the Gaussian baseline."""
    omega, S = _tensor_cross_psd()
    m, C = 5.0, 1e4
    durs = [10.0] * 6
    kc = np.array([9.0, 5.0, 3.0, 7.0, 3.0, 3.0])
    kw = dict(refine=6, smooth=0.0, scalar_equivalent=False)
    lead = jng.joint_nongaussian_tensor_summary(
        omega, S, durs, (120.0, 480.0), (20.0, 40.0), m, C, kurt=kc, exact=False, **kw)
    ex = jng.joint_nongaussian_tensor_summary(
        omega, S, durs, (120.0, 480.0), (20.0, 40.0), m, C, kurt=kc, exact=True, **kw)
    # at least one linear plane's induced kurtosis moves under the exact correction
    diffs = [ex[k]["induced_kurt"] != lead[k]["induced_kurt"]
             for k in ("normal_plane", "shear_plane")]
    assert any(diffs)
    # both stay leptokurtic (above the Gaussian baseline)
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert ex[k]["damage_rate"] >= ex[k]["gaussian_damage_rate"]


# ----------------------------------------------------------------------------
# The corrected multivariate Monte-Carlo
# ----------------------------------------------------------------------------

def test_mc_gaussian_limit_bit_identical():
    """The exact-covariance Monte-Carlo in the Gaussian limit (gamma4_c == 3) is
    BIT-IDENTICAL to the M27/M31 multivariate MC (every transform the identity, the
    NORTA ratio 1)."""
    omega, S = _tensor_cross_psd()
    m, C = 5.0, 1e4
    durs = [300.0]
    mc = jng.joint_nongaussian_monte_carlo_damage(
        omega, S, durs, 0.0, 0.0, m, C, seed=5, kurt=3.0, refine=1, smooth=0.0,
        fs=2400.0, reduction="shear_plane", exact=True)
    m27mc = jf.joint_evolutionary_monte_carlo_damage(
        omega, S, durs, 0.0, 0.0, m, C, seed=5, fs=2400.0, reduction="shear_plane")
    assert mc["damage_rate"] == pytest.approx(m27mc["damage_rate"], rel=1e-12)
    assert mc["sample_cov_error"] == 0.0


def test_mc_corrected_covariance_matches_target():
    """The corrected multivariate MC (synthesised with the underlying correlation rho^U)
    has a SAMPLE covariance closer to the TARGET than the M33 leading-order record — the
    covariance-distortion correction confirmed in the time domain."""
    omega, S = _tensor_cross_psd()
    m, C = 5.0, 1e4
    durs = [300.0]
    kc = np.array([9.0, 7.0, 6.0, 8.0, 5.0, 5.0])
    lead = jng.joint_nongaussian_monte_carlo_damage(
        omega, S, durs, 0.0, 0.0, m, C, seed=5, kurt=kc, refine=1, smooth=0.0,
        fs=2400.0, reduction="shear_plane", exact=False)
    ex = jng.joint_nongaussian_monte_carlo_damage(
        omega, S, durs, 0.0, 0.0, m, C, seed=5, kurt=kc, refine=1, smooth=0.0,
        fs=2400.0, reduction="shear_plane", exact=True)
    assert ex["sample_cov_error"] < lead["sample_cov_error"]
    assert ex["kurtosis"] > 3.5                            # still leptokurtic


# ----------------------------------------------------------------------------
# Cards
# ----------------------------------------------------------------------------

def test_impl_fatig_exact_card_parsing():
    """/EXACT (or /NORTA) on the /IMPL/FATIG/NGAUSS/JOINT/WVILLE card sets
    impl_fatig_exact; a card without it leaves it False (the M33 leading-order path
    untouched)."""
    for flag in ("EXACT", "NORTA", "NATAF"):
        ec, _ = _parse(
            "#\n/RUN/A/1\n1.0\n/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS/" + flag + "\n"
            "0.0 400 500 10\n5.0 1e4 0.03\n3.0 0.0 0 0 9.0 5.0 3.0 7.0 3.0 3.0\n"
            "50.0 250.0 10.0 40.0 6 6\n/END\n")
        assert ec.impl_fatig_exact and ec.impl_fatig_joint and ec.impl_fatig_ngauss
    # no flag -> exact off
    ec, _ = _parse(
        "#\n/RUN/A/1\n1.0\n/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS\n"
        "0.0 400 500 10\n5.0 1e4 0.03\n3.0 0.0 0 0 9.0 5.0 3.0 7.0 3.0 3.0\n"
        "50.0 250.0 10.0 40.0 6 6\n/END\n")
    assert ec.impl_fatig_exact is False


# ----------------------------------------------------------------------------
# End-to-end + reporting (model.implicit_result)
# ----------------------------------------------------------------------------

def test_exact_covariance_end_to_end():
    """/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS/EXACT/NSTAT: the covariance-EXACT joint
    tensor, reported as an ``exact_covariance`` sub-entry of the M33 ``joint_nongaussian``
    entry ALONGSIDE the M33 leading-order joint numbers; the exact preservation error is
    ~0 and much smaller than the M33 leading-order one."""
    eng = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n"
           "/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS/EXACT/NSTAT\n"
           "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000\n"
           "9.0 0.0 0 0 9.0 5.0 4.0 8.0 5.0 8.0\n"
           "20 6\n50.0 250.0 10.0 40.0 6 6 0.0\n/PRINT/-500\n/STOP\n15.0\n")
    m, out = _run(_with(_brick_deck(nx=3), _MISSION), eng, capture=True)
    jn = m.implicit_result.fatigue["wigner_ville"]["joint_nongaussian"]
    ex = jn["exact_covariance"]
    assert ex is not None
    # the covariance-exact preservation error is ~0 (the M34 payload); its magnitude vs
    # the M33 leading-order value is exercised numerically in the library tests on a
    # strongly-correlated tensor (this small brick's significant components are weakly
    # correlated, so both are near machine precision here)
    assert ex["preservation_error"] < 1e-6
    assert "preservation_error_leading" in ex
    # exact induced kurtosis reported and leptokurtic
    assert ex["induced_kurt_max"] >= 3.0
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert np.isfinite(ex[key]["damage_rate"])
    assert "EXACT TRANSLATION-PROCESS CORRELATION-DISTORTION INVERSION" in out
    assert "NORTA" in out
    # the M33 leading-order joint block is still present and unchanged
    assert "JOINT-TENSOR NON-GAUSSIAN DISTRIBUTION" in out


def test_m33_byte_identical_with_without_exact():
    """The M33 leading-order joint, the M32 equivalent-scalar and the M31 Gaussian
    answers are BYTE-IDENTICAL whether or not the M34 exact path runs — the M34 path is
    NEW and ALONGSIDE (the M7 parity contract extended to M34)."""
    base = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n"
            "/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS/NSTAT\n"
            "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000\n"
            "9.0 0.0 0 0 9.0 5.0 4.0 8.0 5.0 8.0\n"
            "20 6\n50.0 250.0 10.0 40.0 6 6 0.0\n/PRINT/-500\n/STOP\n15.0\n")
    exact = base.replace("/WVILLE/NGAUSS/NSTAT", "/WVILLE/NGAUSS/EXACT/NSTAT")
    deck = _with(_brick_deck(nx=3), _MISSION)
    a = _run(deck, base)
    b = _run(deck, exact)
    fa = a.implicit_result.fatigue
    fb = b.implicit_result.fatigue
    # the M31 Gaussian-continuous tensor byte-identical
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert (fa["wigner_ville"][k]["damage_rate"]
                == fb["wigner_ville"][k]["damage_rate"])
    # the M33 leading-order joint reductions byte-identical
    ja = fa["wigner_ville"]["joint_nongaussian"]
    jb = fb["wigner_ville"]["joint_nongaussian"]
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert ja[k]["damage_rate"] == jb[k]["damage_rate"]
    # the M32 equivalent-scalar (nongaussian) sub-entry byte-identical
    assert (fa["wigner_ville"]["nongaussian"]["damage_rate"]
            == fb["wigner_ville"]["nongaussian"]["damage_rate"])
    # M34 exact_covariance present only in the second run
    assert ja.get("exact_covariance") is None
    assert jb["exact_covariance"] is not None

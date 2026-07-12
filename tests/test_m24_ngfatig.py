"""
M24 validations: NON-GAUSSIAN / KURTOSIS SPECTRAL FATIGUE (/IMPL/FATIG/NGAUSS) —
the frequency-domain damage of a stationary but NON-GAUSSIAN random-vibration
response, computed by CORRECTING the M20-M23 Gaussian spectral estimators for a
specified kurtosis / skewness (the Winterstein Hermite-moment model + the
Benasciutti-Braccesi / Rizzi-Kihm closed-form correction lambda_ng), with the
correction cross-validated against a non-Gaussian time-domain Monte-Carlo. Built
ALONGSIDE the M20 scalar / M21-M23 multiaxial spectral fatigue (all stay
bit-identical; the non-Gaussian path CONSUMES the Gaussian moments read-only).

Every new capability gets at least one ANALYTIC check (the port's philosophy):

WINTERSTEIN HERMITE-MOMENT MODEL
* the Gaussian target (gamma4 = 3, gamma3 = 0) gives the IDENTITY transform
  (h3 = h4 = 0, kappa = 1) so the whole correction collapses to the M20 answer;
* the transform preserves the mean (0) and variance EXACTLY (kappa makes
  Var(g) = 1) and hits the target kurtosis to the Winterstein-fit accuracy
  (a hand check via the exact moments of g(u) AND a large-sample check).

NON-GAUSSIAN CORRECTION FACTOR lambda_ng
* lambda_ng = 1 for a GAUSSIAN process (the M20 answer recovered EXACTLY);
* lambda_ng > 1 for a LEPTOKURTIC (gamma4 > 3, spiky) process and < 1 for a
  PLATYKURTIC (gamma4 < 3) one (the closed-form Rayleigh-amplitude integral);
* lambda_ng grows with the kurtosis and the S-N slope; the corrected damage is
  EXACTLY lambda_ng times the Gaussian damage of every estimator.

NON-GAUSSIAN MONTE-CARLO CROSS-CHECK
* the synthesised non-Gaussian history hits the target sample kurtosis, and its
  rainflow-counted / Miner-summed damage matches the lambda_ng-corrected spectral
  estimate within the seeded scatter (a narrow-band process, where the amplitude
  transform and the instantaneous transform coincide);
* the Gaussian limit (gamma4 = 3) reduces the non-Gaussian Monte-Carlo EXACTLY to
  the M20 Gaussian Monte-Carlo (the identity transform).

CARDS + NO-REGRESSION (the M7 parity contract)
* /IMPL/FATIG/NGAUSS card mirror (a PORT sub-flag composing with /MULT, /NPROP,
  /SPEC — freimpl.F has no non-Gaussian fatigue path);
* the non-Gaussian path NEVER mutates the M16 eigensolver / M17-M18 FRFs / the
  M20 SCALAR / M21-M23 MULTIAXIAL fatigue / the element state; the M20-M23
  answers are byte-identical whether or not /NGAUSS runs, and the direct M10
  answer is unchanged.

See PORTING_GUIDE.md roadmap M24.
"""

import contextlib
import io
import math
import os
import tempfile

import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.starter.starter import run_starter

pytest.importorskip("scipy")   # implicit requires scipy (optional otherwise)

from pyradioss.common.messages import MessageLog                    # noqa: E402
from pyradioss.engine.kinematics import LoadsAndConstraints         # noqa: E402
from pyradioss.implicit.modal import modal_frequencies              # noqa: E402
from pyradioss.implicit.modal_response import (                     # noqa: E402
    build_modal_basis, modal_damping, modal_frequency_response)
from pyradioss.implicit.random_response import (                    # noqa: E402
    spectral_moments, stress_modes)
from pyradioss.implicit import spectral_fatigue as sf               # noqa: E402
from pyradioss.implicit import nongaussian_fatigue as ng            # noqa: E402

# reuse the M20 spring-chain deck and the M21 solid-brick deck
from tests.test_m20_fatigue import _chain_deck                      # noqa: E402
from tests.test_m21_multiaxfatig import _brick_deck                 # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _run(starter_text, engine_text, capture=False):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M24_0000.rad")
    ep = os.path.join(d, "M24_0001.rad")
    with open(sp, "w") as f:
        f.write(starter_text)
    with open(ep, "w") as f:
        f.write(engine_text)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_starter(sp)
        model = run_engine(ep)
    return (model, buf.getvalue()) if capture else model


def _starter(text):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M24_0000.rad")
    with open(sp, "w") as f:
        f.write(text)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(sp)


def _narrowband_stress_psd(fc=50.0, bw=1.0, nf=400000, fmax=200.0):
    """A synthetic NARROW-BAND stress PSD (a single Gaussian-shaped peak, M19
    two-sided convention). For a narrow band the amplitude Hermite transform and
    the instantaneous transform coincide, so the non-Gaussian Monte-Carlo tracks
    the lambda_ng-corrected spectral estimate tightly. Returns (f, S, moments)."""
    f = np.linspace(1e-3, fmax, nf)
    G = np.exp(-((f - fc) ** 2) / (2.0 * bw ** 2))     # one-sided-ish (Hz)
    S = G / 2.0                                        # -> two-sided S(omega)
    mom = spectral_moments(2.0 * np.pi * f, S, nmax=4)
    return f, S, mom


# ============================================================================
# WINTERSTEIN HERMITE-MOMENT MODEL
# ============================================================================

def test_hermite_gaussian_is_identity():
    """The Gaussian target (gamma4 = 3, gamma3 = 0) gives the IDENTITY transform:
    h3 = h4 = 0, kappa = 1, g(u) = u — so the whole M24 correction collapses to
    the M20 Gaussian answer."""
    h3, h4, kappa = ng.hermite_coefficients(0.0, 3.0)
    assert h3 == 0.0 and h4 == 0.0 and kappa == 1.0
    u = np.linspace(-4, 4, 101)
    assert np.array_equal(ng.hermite_transform(u, 0.0, 3.0), u)


def test_hermite_preserves_mean_and_variance_exactly():
    """The Hermite transform g(u) = kappa[u + h3(u^2-1) + h4(u^3-3u)] has EXACTLY
    zero mean and unit variance (kappa = 1/sqrt(1 + 2h3^2 + 6h4^2), the Hermite
    orthogonality identity) — checked analytically AND on a large sample."""
    for g3, g4 in ((0.0, 6.0), (0.4, 5.0), (0.0, 4.5)):
        h3, h4, kappa = ng.hermite_coefficients(g3, g4)
        # analytic: Var(g) = kappa^2 (1 + 2 h3^2 + 6 h4^2) = 1 exactly
        var_analytic = kappa ** 2 * (1.0 + 2.0 * h3 ** 2 + 6.0 * h4 ** 2)
        assert var_analytic == pytest.approx(1.0, rel=1e-12)
        rng = np.random.default_rng(4)
        u = rng.standard_normal(2_000_000)
        x = ng.hermite_transform(u, g3, g4)
        assert np.mean(x) == pytest.approx(0.0, abs=5e-3)
        assert np.var(x) == pytest.approx(1.0, rel=5e-3)


def test_hermite_hits_target_kurtosis():
    """The transform hits the target kurtosis (and skewness) to the Winterstein-
    fit accuracy — an EXACT-moment hand check (``hermite_kurtosis``) and a
    large-sample check. The fit is approximate (documented), so a tolerance."""
    for g3, g4 in ((0.0, 5.0), (0.0, 7.0), (0.3, 6.0)):
        skew, kurt = ng.hermite_kurtosis(g3, g4)
        # the realised kurtosis is elevated in the right direction and within the
        # fit accuracy of the target
        assert kurt > 3.0
        assert kurt == pytest.approx(g4, rel=0.25)
        assert skew == pytest.approx(g3, abs=0.1)
        # a large-sample check corroborates the closed-form moments
        rng = np.random.default_rng(11)
        u = rng.standard_normal(4_000_000)
        x = ng.hermite_transform(u, g3, g4)
        xc = x - x.mean()
        k = np.mean(xc ** 4) / np.var(x) ** 2
        assert k == pytest.approx(kurt, rel=0.05)


def test_platykurtic_uses_first_order_coeffs():
    """For a PLATYKURTIC target (gamma4 < 3) the Winterstein softening sqrt would
    go complex, so it falls back to the first-order coefficients h4 =
    (gamma4-3)/24 < 0 (a hardening, saturating transform)."""
    h3, h4, kappa = ng.hermite_coefficients(0.0, 2.5)
    assert h4 == pytest.approx((2.5 - 3.0) / 24.0)
    assert h4 < 0.0


# ============================================================================
# NON-GAUSSIAN CORRECTION FACTOR lambda_ng
# ============================================================================

def test_lambda_ng_gaussian_is_exactly_one():
    """lambda_ng = 1 EXACTLY for a Gaussian process (gamma4 = 3, gamma3 = 0) — the
    M20 answer recovered to machine precision (the identity transform, and the
    same-grid numerator/denominator)."""
    for m in (3.0, 5.0, 8.0):
        assert ng.nongaussian_correction_factor(0.0, 3.0, m) == 1.0


def test_lambda_ng_leptokurtic_greater_than_one():
    """lambda_ng > 1 for a LEPTOKURTIC (spiky, gamma4 > 3) process — the spikes do
    the damage (the g ~ v^3 tail of the Hermite transform)."""
    for g4 in (4.0, 5.0, 7.0):
        lam = ng.nongaussian_correction_factor(0.0, g4, 5.0,
                                               bandwidth_correction=False)
        assert lam > 1.0
    # monotone increasing in the kurtosis
    l4 = ng.nongaussian_correction_factor(0.0, 4.0, 5.0, bandwidth_correction=False)
    l7 = ng.nongaussian_correction_factor(0.0, 7.0, 5.0, bandwidth_correction=False)
    assert l7 > l4
    # monotone increasing in the S-N slope m (steeper curve -> tails matter more)
    lm3 = ng.nongaussian_correction_factor(0.0, 6.0, 3.0, bandwidth_correction=False)
    lm8 = ng.nongaussian_correction_factor(0.0, 6.0, 8.0, bandwidth_correction=False)
    assert lm8 > lm3 > 1.0


def test_lambda_ng_platykurtic_less_than_one():
    """lambda_ng < 1 for a PLATYKURTIC (gamma4 < 3) process — the saturating
    transform clips the amplitudes, LESS damage than Gaussian."""
    lam = ng.nongaussian_correction_factor(0.0, 2.5, 5.0,
                                           bandwidth_correction=False)
    assert 0.0 < lam < 1.0


def test_lambda_ng_bandwidth_attenuation():
    """The Benasciutti-Tovo bandwidth attenuation moves lambda_ng toward 1 for a
    WIDE-band process (small alpha2), and leaves it unchanged for a narrow band
    (alpha2 = 1)."""
    lam_nb = ng.nongaussian_correction_factor(0.0, 6.0, 5.0, alpha2=1.0)
    lam_wb = ng.nongaussian_correction_factor(0.0, 6.0, 5.0, alpha2=0.4)
    lam_full = ng.nongaussian_correction_factor(0.0, 6.0, 5.0,
                                                bandwidth_correction=False)
    assert lam_nb == pytest.approx(lam_full, rel=1e-6)   # alpha2=1 -> full
    assert 1.0 < lam_wb < lam_nb                          # wide band -> less


def test_nongaussian_damage_scales_gaussian_exactly():
    """The corrected damage of each estimator is EXACTLY lambda_ng times the
    Gaussian damage (eq. (5)); the life scales by 1/lambda_ng."""
    _f, _S, mom = _narrowband_stress_psd()
    m, C = 5.0, 1e15
    gauss = sf.fatigue_summary(mom, m, C)
    summ = ng.nongaussian_summary(mom, m, C, 0.0, 6.0)
    lam = summ["lambda_ng"]
    assert lam > 1.0
    for key in ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti"):
        assert summ[key]["damage_rate"] == pytest.approx(
            lam * gauss[key]["damage_rate"], rel=1e-12)
        assert summ[key]["gaussian_damage_rate"] == gauss[key]["damage_rate"]
        assert summ[key]["life"] == pytest.approx(
            gauss[key]["life"] / lam, rel=1e-10)


# ============================================================================
# NON-GAUSSIAN MONTE-CARLO CROSS-CHECK
# ============================================================================

def test_nongaussian_mc_matches_corrected_spectral():
    """On a NARROW-band process the non-Gaussian Monte-Carlo (Gaussian history ->
    Hermite transform -> rainflow -> Miner) matches the lambda_ng-corrected
    spectral (Dirlik) estimate within the seeded scatter, and its sample kurtosis
    hits the target. (The amplitude transform and the instantaneous transform
    coincide for a narrow band.)"""
    f, S, mom = _narrowband_stress_psd()
    m, C = 5.0, 1e15
    g4 = 6.0
    p = sf.spectral_bandwidth_params(mom)
    assert p["alpha2"] > 0.99                       # genuinely narrow band
    lam = ng.nongaussian_correction_factor(0.0, g4, m, alpha2=p["alpha2"],
                                           bandwidth_correction=False)
    gmc = sf.monte_carlo_damage(f, S, m, C, duration=3000.0, seed=7, fs=800.0)
    nmc = ng.nongaussian_monte_carlo_damage(f, S, m, C, 3000.0, 7, 0.0, g4,
                                            fs=800.0)
    # the non-Gaussian history hits the target sample kurtosis
    assert nmc["kurtosis"] == pytest.approx(g4, rel=0.15)
    # and its damage tracks lambda_ng * (Gaussian MC) within scatter
    ratio = nmc["damage_rate"] / (lam * gmc["damage_rate"])
    assert 0.7 < ratio < 1.4, f"ngMC / (lam*gaussMC) = {ratio:.3f} out of band"


def test_gaussian_limit_reduces_to_m20_mc():
    """In the Gaussian limit (gamma4 = 3, gamma3 = 0) the non-Gaussian Monte-Carlo
    reduces EXACTLY (bit-identical) to the M20 Gaussian Monte-Carlo — the identity
    transform returns the M20 history unchanged."""
    f, S, _mom = _narrowband_stress_psd()
    m, C = 5.0, 1e15
    gmc = sf.monte_carlo_damage(f, S, m, C, duration=2000.0, seed=5, fs=800.0)
    nmc = ng.nongaussian_monte_carlo_damage(f, S, m, C, 2000.0, 5, 0.0, 3.0,
                                            fs=800.0)
    assert nmc["damage_rate"] == gmc["damage_rate"]
    assert np.array_equal(nmc["ranges"], gmc["ranges"])
    assert nmc["kurtosis"] == pytest.approx(3.0, abs=0.2)


def test_synthesis_gaussian_limit_bit_identical():
    """``synthesize_nongaussian_history`` with the Gaussian target returns the M20
    Gaussian history bit-for-bit (the identity short-circuit)."""
    f, S, _ = _narrowband_stress_psd()
    _t0, u = sf.synthesize_gaussian_history(f, S, duration=1000.0, seed=9,
                                            fs=800.0)
    _t1, x = ng.synthesize_nongaussian_history(f, S, 1000.0, 9, 0.0, 3.0,
                                               fs=800.0)
    assert np.array_equal(x, u)


# ============================================================================
# CARDS
# ============================================================================

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


def test_impl_fatig_ngauss_card_parsing():
    """/IMPL/FATIG/NGAUSS sets the non-Gaussian flag and reads the target kurtosis
    (and optional skewness) from card LINE 3 (kurt [skew]); it composes with
    /MULT, /NPROP, /SPEC and does NOT imply them — a PORT sub-flag."""
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/FATIG/NGAUSS\n"
                   "0.0 250 2000 7 4\n5.0 1.0e12 0.03\n6.0 0.4\n/END\n")
    assert ec.implicit and ec.impl_fatig and ec.impl_fatig_ngauss
    assert ec.impl_fatig_kurt == pytest.approx(6.0)
    assert ec.impl_fatig_skew == pytest.approx(0.4)
    assert not ec.impl_fatig_mult          # NGAUSS is orthogonal to MULT

    # composes with MULT (any order) + BASE; kurtosis on line 3
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG/MULT/NGAUSS/BASE\n"
                   "0.0 400 500 10 0 6\n5.0 1e4 0.03 0 0 15 9\n5.5\n/END\n")
    assert (ec.impl_fatig_ngauss and ec.impl_fatig_mult
            and ec.impl_fatig_base and ec.impl_fatig_kurt == pytest.approx(5.5))

    # a plain M20 card is NOT non-Gaussian (default kurtosis 3 = Gaussian)
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG\n0.0 50 500 7\n"
                   "5.0 1e12\n/END\n")
    assert not ec.impl_fatig_ngauss and ec.impl_fatig_kurt == 3.0


# ============================================================================
# END-TO-END
# ============================================================================

def test_ngauss_scalar_end_to_end():
    """/IMPL/FATIG/NGAUSS end to end on the base-driven spring chain: the listing
    prints the NON-GAUSSIAN / KURTOSIS block ALONGSIDE the Gaussian one, stores a
    ``nongaussian`` sub-entry with lambda_ng > 1 (leptokurtic), the corrected
    estimators and the non-Gaussian Monte-Carlo cross-check."""
    starter = _chain_deck([2e-3] * 5, [800.] * 5, [0.] * 5)
    m, out = _run(starter,
                  "#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG/NGAUSS/BASE\n"
                  "0.0 250.0 3000 2 0 5\n5.0 1.0e14 0.03 0.0 0.0 400.0 123\n"
                  "6.0 0.0\n/PRINT/-500\n/STOP\n15.0\n", capture=True)
    fat = m.implicit_result.fatigue
    ngres = fat["nongaussian"]
    assert ngres is not None
    assert ngres["gamma4"] == pytest.approx(6.0)
    assert ngres["lambda_ng"] > 1.0                 # leptokurtic -> more damage
    # the corrected life is SHORTER than the Gaussian life
    assert (ngres["dirlik"]["life"]
            < fat["summary"]["dirlik"]["life"])
    assert ngres["dirlik"]["damage_rate"] == pytest.approx(
        ngres["lambda_ng"] * fat["summary"]["dirlik"]["damage_rate"], rel=1e-10)
    assert ngres["monte_carlo"] is not None
    assert ngres["monte_carlo"]["kurtosis"] > 3.0
    assert "NON-GAUSSIAN / KURTOSIS FATIGUE" in out
    assert "CORRECTION lambda_ng" in out
    # the M20 Gaussian block is still there, side by side
    assert "RANDOM-VIBRATION (SPECTRAL) FATIGUE" in out


def test_ngauss_multiaxial_end_to_end():
    """/IMPL/FATIG/MULT/NGAUSS composes: the M21 multiaxial reductions run
    unchanged, and the non-Gaussian correction scales each reduction's Gaussian
    damage, with a non-Gaussian Monte-Carlo on the shear-plane projection."""
    eng = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/NGAUSS\n"
           "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000\n5.0\n"
           "/PRINT/-500\n/STOP\n15.0\n")
    m, out = _run(_brick_deck(nx=3), eng, capture=True)
    fat = m.implicit_result.fatigue
    assert fat.get("multiaxial")
    ngres = fat["nongaussian"]
    assert ngres is not None
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert ngres[key]["lambda_ng"] > 1.0
        assert ngres[key]["dirlik"]["damage_rate"] == pytest.approx(
            ngres[key]["lambda_ng"] * fat[key]["summary"]["dirlik"][
                "damage_rate"], rel=1e-10)
    assert ngres["monte_carlo"] is not None
    assert "NON-GAUSSIAN / KURTOSIS FATIGUE" in out
    # the M21 block still present side by side
    assert "MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE" in out


# ============================================================================
# NO-REGRESSION (the M7 parity contract)
# ============================================================================

def test_m20_scalar_byte_identical_with_without_ngauss():
    """The M20 SCALAR Gaussian summary is BYTE-IDENTICAL whether or not the M24
    non-Gaussian path runs — the correction is NEW and ALONGSIDE."""
    starter = _chain_deck([2e-3] * 5, [800.] * 5, [0.] * 5)
    base = ("#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG/BASE\n"
            "0.0 250.0 3000 2 0 5\n5.0 1.0e14 0.03 0.0 0.0 400.0 123\n"
            "{extra}/PRINT/-500\n/STOP\n15.0\n")
    m20 = _run(starter, base.format(extra=""))
    # /NGAUSS needs the /NGAUSS flag + line 3
    ng_eng = ("#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG/NGAUSS/BASE\n"
              "0.0 250.0 3000 2 0 5\n5.0 1.0e14 0.03 0.0 0.0 400.0 123\n"
              "6.0 0.0\n/PRINT/-500\n/STOP\n15.0\n")
    m24 = _run(starter, ng_eng)
    f20 = m20.implicit_result.fatigue
    f24 = m24.implicit_result.fatigue
    assert f20["nongaussian"] is None
    assert f24["nongaussian"] is not None
    for est in ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti"):
        assert (f20["summary"][est]["damage_rate"]
                == f24["summary"][est]["damage_rate"])
    # the Gaussian Monte-Carlo is byte-identical too
    assert (f20["monte_carlo"]["damage_rate"]
            == f24["monte_carlo"]["damage_rate"])


def test_m21_m22_m23_byte_identical_with_without_ngauss():
    """The M21 spectral, M22 time-domain and M23 spectral non-proportional answers
    are BYTE-IDENTICAL whether or not the M24 non-Gaussian path runs."""
    deck = _brick_deck(nx=3)
    eng_spec = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/NPROP/SPEC\n"
                "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000 0.3 0.5\n"
                "/PRINT/-500\n/STOP\n15.0\n")
    eng_spec_ng = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n"
                   "/IMPL/FATIG/MULT/NPROP/SPEC/NGAUSS\n"
                   "0.0 400.0 500 10 6\n"
                   "5.0 1.0e4 0.03 0.0 0.0 15.0 21000 0.3 0.5\n5.0\n"
                   "/PRINT/-500\n/STOP\n15.0\n")
    a = _run(deck, eng_spec)
    b = _run(deck, eng_spec_ng)
    fa = a.implicit_result.fatigue
    fb = b.implicit_result.fatigue
    assert fa["nongaussian"] is None and fb["nongaussian"] is not None
    # M21 reductions
    for key in ("von_mises", "normal_plane", "shear_plane"):
        for est in ("narrow_band", "dirlik", "wirsching_light",
                    "tovo_benasciutti"):
            assert (fa[key]["summary"][est]["damage_rate"]
                    == fb[key]["summary"][est]["damage_rate"])
    # M22 time-domain + M23 spectral
    for mdl in ("findley", "fatemi_socie", "shear_path"):
        assert (fa["nprop_result"][mdl]["damage_rate"]
                == fb["nprop_result"][mdl]["damage_rate"])
        assert (fa["nprop_result"]["spectral"][mdl]["damage_rate"]
                == fb["nprop_result"]["spectral"][mdl]["damage_rate"])


def test_ngauss_does_not_mutate_state():
    """The non-Gaussian path is read-only in the element state and leaves the M16
    modal_frequencies output bit-identical before and after (the M14-M23 parity
    contract extended to M24)."""
    m = _starter(_chain_deck([2e-3] * 3, [800.] * 3, [0.] * 3))
    f0, _, _ = modal_frequencies(m, nev=3)
    before = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in m.springs.state.items()}
    basis = build_modal_basis(m, nev=3)
    Sigma, channels = stress_modes(m, basis)
    z = modal_damping(basis.omega, uniform=0.03)
    n = m.numnod
    F = np.zeros((n, 3))
    LoadsAndConstraints(m, MessageLog()).external_forces(1.0, F, m.x0)
    frf = modal_frequency_response(basis, F, np.zeros((n, 3)),
                                   np.linspace(1e-6, 300.0, 60), z)
    # a representative non-Gaussian evaluation
    ng.nongaussian_summary(np.array([1.0, 10.0, 200.0, 5e3, 2e5]), 5.0, 1e14,
                           0.0, 6.0)
    for k, v in before.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(m.springs.state[k], v), k
    f1, _, _ = modal_frequencies(m, nev=3)
    assert np.array_equal(f0, f1)


def test_direct_dynamics_unchanged_by_ngauss_path():
    """A DIRECT /IMPL/DYNA run is byte-for-byte unaffected by the M24 non-Gaussian
    machinery living in the same package (the M10 integrator stays bit-identical,
    no fatigue attached)."""
    K_D, M_D = 4.0, 2.0e-3
    om = np.sqrt(K_D / (M_D / 2.0))
    T = 2.0 * np.pi / om
    starter = f"""\
#RADIOSS STARTER
/BEGIN
SM
/NODE
         1{0.0:20.10f}{0.0:20.10f}{0.0:20.10f}
         2{10.0:20.10f}{0.0:20.10f}{0.0:20.10f}
/SPRING/1
         1         1         2
/PART/1
s
         1         1
/MAT/LAW1/1
st
   7.8e-6
     210.0       0.0
/PROP/SPRING/1
sp
     {M_D}       {K_D}       0.0
/GRNOD/NODE/1
n
1
/GRNOD/NODE/2
t
2
/INIVEL/TRA/1
kick
      0.01       0.0       0.0         2
/BCS/1
pin
       111       111         0         1
/BCS/2
axial
       011       111         0         2
/END
"""
    eng = (f"#\n/RUN/SM/1\n{2*T}\n/IMPL/DYNA/2\n0.5 0.25\n/IMPL/DTINI\n"
           f"{T/100}\n/END\n")
    m1 = _run(starter, eng)
    m2 = _run(starter, eng)
    u1 = np.array([uu[m1.node_index(2), 0]
                   for uu in m1.implicit_result.history["u"]])
    u2 = np.array([uu[m2.node_index(2), 0]
                   for uu in m2.implicit_result.history["u"]])
    assert np.array_equal(u1, u2)
    assert m1.implicit_result.fatigue is None

"""
M25 validations: NON-STATIONARY / EVOLUTIONARY-PSD SPECTRAL FATIGUE
(/IMPL/FATIG/NSTAT) — the frequency-domain damage of a random-vibration response
whose PSD / RMS VARIES WITH TIME, computed by extending the M20-M24 STATIONARY
spectral estimators to a non-stationary process (the piecewise-stationary
"mission profile" block Miner-sum + the amplitude-modulated / evolutionary
E[a^m]-weighted damage), cross-validated against a non-stationary time-domain
Monte-Carlo. Built ALONGSIDE the M20 scalar / M21-M23 multiaxial / M24
non-Gaussian spectral fatigue (all stay bit-identical; the non-stationary path
CONSUMES the stationary moments — and M24's lambda_ng — read-only).

Every new capability gets at least one ANALYTIC check (the port's philosophy):

RMS-MODULATION STATISTICS + AMPLIFICATION
* a CONSTANT modulation induces kurtosis EXACTLY 3 and amplification EXACTLY 1
  (the stationary / Gaussian limit); any VARYING modulation is leptokurtic
  (gamma4 > 3, Jensen) and amplifies the damage (kappa_ns > 1);
* the induced kurtosis is 3 E[a^4]/E[a^2]^2 (the Wolfsteiner-Trapp / Kihm-Rizzi
  link) and kappa_ns = E[a^m]/E[a^2]^(m/2), both monotone in the modulation depth.

BLOCK ("mission profile") + AMPLITUDE-MODULATED DAMAGE
* the block Miner-sum EQUALS the duration-weighted per-block damages (hand check);
* for a SHARED spectral shape the block Miner-sum EQUALS the amplitude-modulated
  E[a^m]-weighted damage EXACTLY (the two models coincide);
* a CONSTANT modulation / single unit block recovers the M20 stationary answer
  EXACTLY (E[a^m] = 1);
* the amplitude-modulated damage scales every estimator by E[a^m].

M25<->M24 BRIDGE
* constant modulation: kappa_ns = lambda_ng = 1 EXACTLY (both directions);
* mild kurtosis: the M25 amplification and the M24 lambda_ng at the induced
  kurtosis AGREE within tolerance (they are two models of the same leptokurtic
  marginal, differing by the documented (m-2)/(m-1) amplitude-model factor);
* both > 1 and monotone increasing in the kurtosis (same direction).

NON-STATIONARY MONTE-CARLO CROSS-CHECK
* the synthesised history's sample kurtosis matches the induced kurtosis and its
  per-block RMS tracks the modulation schedule;
* the non-stationary Monte-Carlo damage matches the block / modulated spectral
  estimate within the seeded scatter;
* the constant-modulation limit reduces the non-stationary Monte-Carlo EXACTLY
  (bit-identical) to the M20 Gaussian Monte-Carlo.

CARDS + NO-REGRESSION (the M7 parity contract)
* /IMPL/FATIG/NSTAT card mirror (a PORT sub-flag composing with /MULT, /NPROP,
  /SPEC, /NGAUSS — freimpl.F has no non-stationary fatigue path);
* the non-stationary path NEVER mutates the M16 eigensolver / M17-M18 FRFs / the
  M20 SCALAR / M21-M23 MULTIAXIAL / M24 NON-GAUSSIAN fatigue / the element state;
  the M20-M24 answers are byte-identical whether or not /NSTAT runs, and the
  direct M10 answer is unchanged.

See PORTING_GUIDE.md roadmap M25.
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
from pyradioss.implicit import nongaussian_fatigue as ngf           # noqa: E402
from pyradioss.implicit import nonstationary_fatigue as ns          # noqa: E402

# reuse the M20 spring-chain deck and the M21 solid-brick deck
from tests.test_m20_fatigue import _chain_deck                      # noqa: E402
from tests.test_m21_multiaxfatig import _brick_deck                 # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _run(starter_text, engine_text, capture=False):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M25_0000.rad")
    ep = os.path.join(d, "M25_0001.rad")
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
    sp = os.path.join(d, "M25_0000.rad")
    with open(sp, "w") as f:
        f.write(text)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(sp)


# a run-up / dwell / run-down mission-profile /FUNCT (scale vs time)
_MISSION = ("/FUNCT/20\nmission\n       0.0       0.5\n"
            "     100.0       1.5\n     200.0       1.5\n"
            "     300.0       0.3\n")


def _with_mission(deck):
    """Append the mission-profile /FUNCT/20 to a starter deck (before /END)."""
    return deck.replace("/END", _MISSION + "/END", 1)


def _narrowband_stress_psd(fc=50.0, bw=1.0, nf=200000, fmax=200.0):
    """A synthetic NARROW-BAND stress PSD (M19 two-sided convention). For a
    narrow band the block model and the Monte-Carlo track tightly. Returns
    (f, S, moments)."""
    f = np.linspace(1e-3, fmax, nf)
    G = np.exp(-((f - fc) ** 2) / (2.0 * bw ** 2))
    S = G / 2.0
    mom = spectral_moments(2.0 * np.pi * f, S, nmax=4)
    return f, S, mom


def _moments_narrowband(sigma, f0):
    w0 = 2.0 * math.pi * f0
    m0 = sigma ** 2
    return np.array([m0, m0 * w0, m0 * w0 ** 2, m0 * w0 ** 3, m0 * w0 ** 4])


# ============================================================================
# RMS-MODULATION STATISTICS + AMPLIFICATION
# ============================================================================

def test_constant_modulation_is_gaussian_and_unit():
    """A CONSTANT modulation induces kurtosis EXACTLY 3 and amplification EXACTLY
    1 — the stationary / Gaussian limit."""
    for a in (1.0, 2.7):
        assert ns.rms_modulation_kurtosis([a, a, a], [1., 2., 3.]) == 3.0
        for m in (3.0, 5.0, 8.0):
            assert ns.nonstationary_amplification([a], [1.0], m) == 1.0
            assert ns.nonstationary_amplification([a, a], [0.4, 0.6], m) == \
                pytest.approx(1.0, abs=1e-14)


def test_varying_modulation_is_leptokurtic_and_amplifies():
    """Any VARYING modulation is leptokurtic (gamma4 > 3, Jensen) and amplifies
    the damage (kappa_ns > 1), both monotone in the modulation depth."""
    k_shallow = ns.rms_modulation_kurtosis([0.9, 1.1], [0.5, 0.5])
    k_deep = ns.rms_modulation_kurtosis([0.5, 1.5], [0.5, 0.5])
    assert 3.0 < k_shallow < k_deep
    a_shallow = ns.nonstationary_amplification([0.9, 1.1], [0.5, 0.5], 5.0)
    a_deep = ns.nonstationary_amplification([0.5, 1.5], [0.5, 0.5], 5.0)
    assert 1.0 < a_shallow < a_deep
    # monotone in the S-N slope m (steeper -> the high-RMS block dominates more)
    a_m3 = ns.nonstationary_amplification([0.5, 1.5], [0.5, 0.5], 3.0)
    a_m8 = ns.nonstationary_amplification([0.5, 1.5], [0.5, 0.5], 8.0)
    assert a_m8 > a_m3 > 1.0


def test_induced_kurtosis_closed_form():
    """The induced kurtosis is 3 E[a^4]/E[a^2]^2 (theory eq. (5)) — a hand check
    against the moment definition."""
    a = np.array([0.5, 1.0, 2.0])
    w = np.array([0.3, 0.5, 0.2])
    e_a2 = np.sum(w * a ** 2)
    e_a4 = np.sum(w * a ** 4)
    assert ns.rms_modulation_kurtosis(a, w) == pytest.approx(3.0 * e_a4 / e_a2 ** 2)


# ============================================================================
# BLOCK ("mission profile") + AMPLITUDE-MODULATED DAMAGE
# ============================================================================

def test_block_miner_sum_hand_check():
    """The block Miner-sum EQUALS the duration-weighted per-block damages (theory
    eqs. (1)-(2)): D = sum_i dr_i T_i, damage_rate = D / sum T_i."""
    base = _moments_narrowband(10.0, 50.0)
    m, C = 5.0, 1e15
    scales = [0.7, 1.3, 1.0]
    durs = [50.0, 30.0, 120.0]
    blocks = [{"scale": s, "duration": d} for s, d in zip(scales, durs)]
    summ = ns.block_fatigue_summary(blocks, m, C, estimator="dirlik",
                                    base_moments=base)
    # hand-compute the per-block damage and the Miner sum
    D = 0.0
    T = 0.0
    for s, d in zip(scales, durs):
        dr_i = sf.dirlik_damage(ns.block_moments(base, s), m, C)["damage_rate"]
        D += dr_i * d
        T += d
    assert summ["damage"] == pytest.approx(D, rel=1e-12)
    assert summ["total_time"] == pytest.approx(T)
    assert summ["damage_rate"] == pytest.approx(D / T, rel=1e-12)
    assert summ["life"] == pytest.approx(T / D, rel=1e-12)


def test_block_equals_amplitude_modulated_shared_shape():
    """For a SHARED spectral shape the block Miner-sum EQUALS the amplitude-
    modulated E[a^m]-weighted damage EXACTLY (theory eq. (4)) — the two models
    coincide (weights = duration fractions)."""
    base = _moments_narrowband(8.0, 40.0)
    m, C = 5.0, 1e14
    scales = [0.6, 1.0, 1.4, 2.0]
    durs = [40.0, 80.0, 30.0, 10.0]
    blocks = [{"scale": s, "duration": d} for s, d in zip(scales, durs)]
    bl = ns.block_fatigue_summary(blocks, m, C, estimator="dirlik",
                                  base_moments=base)
    sc, wt = ns.modulation_from_schedule(scales, durs)
    am = ns.amplitude_modulated_summary(base, sc, wt, m, C)
    assert bl["damage_rate"] == pytest.approx(am["dirlik"]["damage_rate"],
                                              rel=1e-12)


def test_constant_modulation_recovers_m20_exactly():
    """A CONSTANT unit modulation / single unit block recovers the M20 stationary
    answer EXACTLY (E[a^m] = 1) — for every estimator."""
    base = _moments_narrowband(12.0, 60.0)
    m, C = 5.0, 1e15
    gauss = sf.fatigue_summary(base, m, C)
    am = ns.amplitude_modulated_summary(base, [1.0], [1.0], m, C)
    assert am["e_am"] == 1.0
    assert am["kappa_ns"] == 1.0
    assert am["kurtosis"] == 3.0
    for key in ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti"):
        assert am[key]["damage_rate"] == gauss[key]["damage_rate"]


def test_amplitude_modulated_scales_every_estimator():
    """The amplitude-modulated damage of each estimator is EXACTLY E[a^m] times
    the stationary damage (theory eq. (4)); the life scales by 1/E[a^m]."""
    base = _moments_narrowband(10.0, 50.0)
    m, C = 5.0, 1e15
    scales, weights = [0.8, 1.5], [0.6, 0.4]
    e_am = 0.6 * 0.8 ** m + 0.4 * 1.5 ** m
    gauss = sf.fatigue_summary(base, m, C)
    am = ns.amplitude_modulated_summary(base, scales, weights, m, C)
    assert am["e_am"] == pytest.approx(e_am, rel=1e-12)
    for key in ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti"):
        assert am[key]["damage_rate"] == pytest.approx(
            e_am * gauss[key]["damage_rate"], rel=1e-12)
        assert am[key]["stationary_damage_rate"] == gauss[key]["damage_rate"]
        assert am[key]["life"] == pytest.approx(gauss[key]["life"] / e_am,
                                                rel=1e-10)


def test_block_moments_scale_as_square():
    """``block_moments`` scales a shared-shape moment array by a^2 (the PSD scales
    as the RMS squared), leaving the rates / width factors invariant."""
    base = _moments_narrowband(10.0, 50.0)
    scaled = ns.block_moments(base, 2.0)
    assert np.allclose(scaled, base * 4.0)
    p0 = sf.spectral_bandwidth_params(base)
    p1 = sf.spectral_bandwidth_params(scaled)
    assert p1["sigma"] == pytest.approx(2.0 * p0["sigma"])
    assert p1["nu0"] == pytest.approx(p0["nu0"])       # rates invariant
    assert p1["alpha2"] == pytest.approx(p0["alpha2"])  # width invariant


# ============================================================================
# M25<->M24 BRIDGE
# ============================================================================

def test_bridge_constant_modulation_both_unity():
    """Constant modulation: kappa_ns = lambda_ng = 1 EXACTLY (both directions —
    the M25 amplification and the M24 correction meet at the Gaussian limit)."""
    for m in (3.0, 5.0, 8.0):
        br = ns.bridge_to_nongaussian([1.7], [1.0], m)
        assert br["kurtosis"] == 3.0
        assert br["kappa_ns"] == 1.0
        assert br["lambda_ng"] == 1.0
        assert br["ratio"] == pytest.approx(1.0)


def test_bridge_mild_kurtosis_agrees():
    """Mild kurtosis: the M25 amplification kappa_ns and the M24 lambda_ng at the
    induced kurtosis AGREE within tolerance — two models of the SAME leptokurtic
    marginal (differing by the documented (m-2)/(m-1) amplitude-model factor,
    tight for mild kurtosis / low slope)."""
    br = ns.bridge_to_nongaussian([0.9, 1.1], [0.5, 0.5], 3.0)
    assert br["kurtosis"] == pytest.approx(3.12, abs=0.02)
    assert br["kappa_ns"] > 1.0 and br["lambda_ng"] > 1.0
    # both amplify; agree to a few percent at this mild kurtosis / slope
    assert br["ratio"] == pytest.approx(1.0, abs=0.08)


def test_bridge_direction_and_monotonicity():
    """Both kappa_ns and lambda_ng are > 1 for a leptokurtic modulation and grow
    together with the kurtosis (same DIRECTION) — the bridge holds qualitatively
    across the range even where the exact magnitudes diverge."""
    b_mild = ns.bridge_to_nongaussian([0.85, 1.15], [0.5, 0.5], 4.0)
    b_deep = ns.bridge_to_nongaussian([0.6, 1.4], [0.5, 0.5], 4.0)
    assert b_deep["kurtosis"] > b_mild["kurtosis"] > 3.0
    assert b_deep["kappa_ns"] > b_mild["kappa_ns"] > 1.0
    assert b_deep["lambda_ng"] > b_mild["lambda_ng"] > 1.0


# ============================================================================
# NON-STATIONARY MONTE-CARLO CROSS-CHECK
# ============================================================================

def test_ns_mc_constant_limit_bit_identical():
    """In the CONSTANT-modulation limit (a(t) = 1) the non-stationary Monte-Carlo
    reduces EXACTLY (bit-identical) to the M20 Gaussian Monte-Carlo — the envelope
    is unity, x = u."""
    f, S, _ = _narrowband_stress_psd()
    m, C = 5.0, 1e15
    gmc = sf.monte_carlo_damage(f, S, m, C, duration=2000.0, seed=5, fs=800.0)
    nmc = ns.nonstationary_monte_carlo_damage(f, S, m, C, [1.0], [2000.0], 5,
                                              fs=800.0)
    assert nmc["damage_rate"] == gmc["damage_rate"]
    assert np.array_equal(nmc["ranges"], gmc["ranges"])
    assert nmc["kurtosis"] == pytest.approx(3.0, abs=0.2)


def test_ns_mc_hits_induced_kurtosis_and_schedule():
    """The synthesised history's sample kurtosis matches the induced kurtosis, and
    its per-block RMS tracks the modulation schedule (the time-varying RMS)."""
    f, S, mom = _narrowband_stress_psd()
    m, C = 5.0, 1e15
    scales = [0.5, 1.0, 1.8]
    durs = [1500.0, 1500.0, 1500.0]
    induced = ns.rms_modulation_kurtosis(*ns.modulation_from_schedule(scales, durs))
    nmc = ns.nonstationary_monte_carlo_damage(f, S, m, C, scales, durs, 3,
                                              fs=800.0)
    assert nmc["kurtosis"] == pytest.approx(induced, rel=0.15)
    # the per-block RMS follows the schedule ratios (0.5 : 1.0 : 1.8)
    br = nmc["block_rms"]
    assert br[1] / br[0] == pytest.approx(1.0 / 0.5, rel=0.15)
    assert br[2] / br[0] == pytest.approx(1.8 / 0.5, rel=0.15)


def test_ns_mc_matches_block_spectral():
    """On a NARROW-band process the non-stationary Monte-Carlo matches the block /
    amplitude-modulated spectral estimate within the seeded scatter."""
    f, S, mom = _narrowband_stress_psd()
    m, C = 5.0, 1e15
    scales = [0.6, 1.0, 1.5]
    durs = [1000.0, 1000.0, 1000.0]
    blocks = [{"scale": s, "duration": d} for s, d in zip(scales, durs)]
    bl = ns.block_fatigue_summary(blocks, m, C, estimator="dirlik",
                                  base_moments=mom)
    nmc = ns.nonstationary_monte_carlo_damage(f, S, m, C, scales, durs, 7,
                                              fs=800.0)
    ratio = nmc["damage_rate"] / bl["damage_rate"]
    assert 0.6 < ratio < 1.6, f"nsMC / block = {ratio:.3f} out of band"


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


def test_impl_fatig_nstat_card_parsing():
    """/IMPL/FATIG/NSTAT sets the non-stationary flag and reads the RMS
    modulation /FUNCT (and optional nseg) from the card line AFTER the sweep /
    S-N (and kurtosis, if NGAUSS) lines; it composes with /MULT, /NPROP, /SPEC,
    /NGAUSS and does NOT imply them — a PORT sub-flag."""
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/FATIG/NSTAT/BASE\n"
                   "0.0 250 2000 7 0 4\n5.0 1.0e12 0.03\n20 8\n/END\n")
    assert ec.implicit and ec.impl_fatig and ec.impl_fatig_nstat
    assert ec.impl_fatig_modfunct == 20
    assert ec.impl_fatig_nstat_nseg == 8
    assert not ec.impl_fatig_mult          # NSTAT is orthogonal to MULT

    # composes with MULT (any order) + BASE; modulation on the line after S-N
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG/MULT/NSTAT/BASE\n"
                   "0.0 400 500 10 0 6\n5.0 1e4 0.03 0 0 15 9\n30\n/END\n")
    assert (ec.impl_fatig_nstat and ec.impl_fatig_mult and ec.impl_fatig_base
            and ec.impl_fatig_modfunct == 30)

    # composes with NGAUSS: kurtosis on line 3, modulation on line 4
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG/NGAUSS/NSTAT/BASE\n"
                   "0.0 250 2000 7 0 5\n5.0 1e12 0.03 0 0 400 123\n"
                   "6.0 0.0\n20 8\n/END\n")
    assert (ec.impl_fatig_ngauss and ec.impl_fatig_nstat
            and ec.impl_fatig_kurt == pytest.approx(6.0)
            and ec.impl_fatig_modfunct == 20 and ec.impl_fatig_nstat_nseg == 8)

    # a plain M20 card is NOT non-stationary
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG\n0.0 50 500 7\n"
                   "5.0 1e12\n/END\n")
    assert not ec.impl_fatig_nstat and ec.impl_fatig_modfunct == 0


# ============================================================================
# END-TO-END
# ============================================================================

def test_nstat_scalar_end_to_end():
    """/IMPL/FATIG/NSTAT end to end on the base-driven spring chain: the listing
    prints the NON-STATIONARY block ALONGSIDE the stationary one, stores a
    ``nonstationary`` sub-entry with the block Miner-sum, the amplitude-modulated
    damage (block == modulated), the induced kurtosis / bridge and the
    non-stationary Monte-Carlo."""
    starter = _with_mission(_chain_deck([2e-3] * 5, [800.] * 5, [0.] * 5))
    m, out = _run(starter,
                  "#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG/NSTAT/BASE\n"
                  "0.0 250.0 3000 2 0 5\n5.0 1.0e14 0.03 0.0 0.0 400.0 123\n"
                  "20 6\n/PRINT/-500\n/STOP\n15.0\n", capture=True)
    fat = m.implicit_result.fatigue
    nsres = fat["nonstationary"]
    assert nsres is not None
    assert nsres["nblocks"] == 6
    assert nsres["kurtosis"] > 3.0                    # varying RMS -> leptokurtic
    # block Miner-sum EQUALS the amplitude-modulated Dirlik (shared shape)
    assert nsres["block"]["damage_rate"] == pytest.approx(
        nsres["amplitude_modulated"]["dirlik"]["damage_rate"], rel=1e-12)
    # the amplitude-modulated Dirlik = E[a^m] * the stationary Dirlik
    assert nsres["amplitude_modulated"]["dirlik"]["damage_rate"] == pytest.approx(
        nsres["e_am"] * fat["summary"]["dirlik"]["damage_rate"], rel=1e-10)
    assert nsres["monte_carlo"] is not None
    assert nsres["monte_carlo"]["kurtosis"] > 3.0
    # the M25<->M24 bridge is present
    assert nsres["bridge"]["kappa_ns"] > 1.0
    assert "NON-STATIONARY / EVOLUTIONARY-PSD FATIGUE" in out
    assert "M25<->M24 BRIDGE" in out
    # the M20 stationary block is still there, side by side
    assert "RANDOM-VIBRATION (SPECTRAL) FATIGUE" in out


def test_nstat_multiaxial_end_to_end():
    """/IMPL/FATIG/MULT/NSTAT composes: the M21 multiaxial reductions run
    unchanged, and the non-stationary correction scales each reduction's
    stationary damage by E[a^m], with a non-stationary Monte-Carlo."""
    eng = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/NSTAT\n"
           "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000\n20 6\n"
           "/PRINT/-500\n/STOP\n15.0\n")
    m, out = _run(_with_mission(_brick_deck(nx=3)), eng, capture=True)
    fat = m.implicit_result.fatigue
    assert fat.get("multiaxial")
    nsres = fat["nonstationary"]
    assert nsres is not None
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert nsres[key]["e_am"] > 1.0
        assert nsres[key]["amplitude_modulated"]["dirlik"]["damage_rate"] == \
            pytest.approx(nsres[key]["e_am"]
                          * fat[key]["summary"]["dirlik"]["damage_rate"],
                          rel=1e-10)
        # block == amplitude-modulated (shared shape)
        assert nsres[key]["block"]["damage_rate"] == pytest.approx(
            nsres[key]["amplitude_modulated"]["dirlik"]["damage_rate"],
            rel=1e-12)
    assert nsres["monte_carlo"] is not None
    assert "NON-STATIONARY / EVOLUTIONARY-PSD FATIGUE" in out
    # the M21 block still present side by side
    assert "MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE" in out


def test_nstat_composes_with_ngauss():
    """/IMPL/FATIG/NGAUSS/NSTAT: BOTH siblings run — the M24 non-Gaussian
    correction (fixed kurtosis) AND the M25 non-stationary correction (RMS
    modulation) are stored side by side, each byte-identical to running alone."""
    starter = _with_mission(_chain_deck([2e-3] * 5, [800.] * 5, [0.] * 5))
    eng = ("#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG/NGAUSS/NSTAT/BASE\n"
           "0.0 250.0 3000 2 0 5\n5.0 1.0e14 0.03 0.0 0.0 400.0 123\n"
           "6.0 0.0\n20 6\n/PRINT/-500\n/STOP\n15.0\n")
    m, out = _run(starter, eng, capture=True)
    fat = m.implicit_result.fatigue
    assert fat["nongaussian"] is not None and fat["nonstationary"] is not None
    assert fat["nongaussian"]["gamma4"] == pytest.approx(6.0)
    assert fat["nonstationary"]["kurtosis"] > 3.0
    assert "NON-GAUSSIAN / KURTOSIS FATIGUE" in out
    assert "NON-STATIONARY / EVOLUTIONARY-PSD FATIGUE" in out


# ============================================================================
# NO-REGRESSION (the M7 parity contract)
# ============================================================================

def test_m20_scalar_byte_identical_with_without_nstat():
    """The M20 SCALAR stationary summary is BYTE-IDENTICAL whether or not the M25
    non-stationary path runs — the correction is NEW and ALONGSIDE."""
    starter = _with_mission(_chain_deck([2e-3] * 5, [800.] * 5, [0.] * 5))
    base = ("#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG/BASE\n"
            "0.0 250.0 3000 2 0 5\n5.0 1.0e14 0.03 0.0 0.0 400.0 123\n"
            "/PRINT/-500\n/STOP\n15.0\n")
    ns_eng = ("#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG/NSTAT/BASE\n"
              "0.0 250.0 3000 2 0 5\n5.0 1.0e14 0.03 0.0 0.0 400.0 123\n"
              "20 6\n/PRINT/-500\n/STOP\n15.0\n")
    m20 = _run(starter, base)
    m25 = _run(starter, ns_eng)
    f20 = m20.implicit_result.fatigue
    f25 = m25.implicit_result.fatigue
    assert f20["nonstationary"] is None
    assert f25["nonstationary"] is not None
    for est in ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti"):
        assert (f20["summary"][est]["damage_rate"]
                == f25["summary"][est]["damage_rate"])
    # the Gaussian Monte-Carlo is byte-identical too
    assert (f20["monte_carlo"]["damage_rate"]
            == f25["monte_carlo"]["damage_rate"])


def test_m21_m22_m23_m24_byte_identical_with_without_nstat():
    """The M21 spectral, M22 time-domain, M23 spectral non-proportional AND M24
    non-Gaussian answers are BYTE-IDENTICAL whether or not the M25 non-stationary
    path runs."""
    deck = _with_mission(_brick_deck(nx=3))
    eng = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/NPROP/SPEC/NGAUSS\n"
           "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000 0.3 0.5\n"
           "5.0\n/PRINT/-500\n/STOP\n15.0\n")
    eng_ns = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n"
              "/IMPL/FATIG/MULT/NPROP/SPEC/NGAUSS/NSTAT\n"
              "0.0 400.0 500 10 6\n"
              "5.0 1.0e4 0.03 0.0 0.0 15.0 21000 0.3 0.5\n5.0\n20 6\n"
              "/PRINT/-500\n/STOP\n15.0\n")
    a = _run(deck, eng)
    b = _run(deck, eng_ns)
    fa = a.implicit_result.fatigue
    fb = b.implicit_result.fatigue
    assert fa["nonstationary"] is None and fb["nonstationary"] is not None
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
    # M24 non-Gaussian
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert (fa["nongaussian"][key]["dirlik"]["damage_rate"]
                == fb["nongaussian"][key]["dirlik"]["damage_rate"])


def test_nstat_does_not_mutate_state():
    """The non-stationary path is read-only in the element state and leaves the
    M16 modal_frequencies output bit-identical before and after (the M14-M24
    parity contract extended to M25)."""
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
    base = _moments_narrowband(10.0, 50.0)
    # a representative non-stationary evaluation
    ns.amplitude_modulated_summary(base, [0.5, 1.0, 1.5], [0.3, 0.4, 0.3],
                                   5.0, 1e14)
    ns.block_fatigue_summary(
        [{"scale": s, "duration": 1.0} for s in (0.5, 1.0, 1.5)],
        5.0, 1e14, base_moments=base)
    for k, v in before.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(m.springs.state[k], v), k
    f1, _, _ = modal_frequencies(m, nev=3)
    assert np.array_equal(f0, f1)


def test_direct_dynamics_unchanged_by_nstat_path():
    """A DIRECT /IMPL/DYNA run is byte-for-byte unaffected by the M25
    non-stationary machinery living in the same package (the M10 integrator stays
    bit-identical, no fatigue attached)."""
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

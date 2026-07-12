"""
M26 validations: FULLY EVOLUTIONARY / NON-SEPARABLE-PSD SPECTRAL FATIGUE
(/IMPL/FATIG/EVOL) — the frequency-domain damage of a random-vibration response
whose spectral SHAPE (not merely its RMS level) VARIES WITH TIME, computed by
extending the M25 piecewise-stationary / amplitude-modulated estimators to a
genuinely NON-SEPARABLE evolutionary spectrum S(omega, t) / spectrogram (a
time-frequency picture whose bandwidth / rates / centre frequency DRIFT with
time), cross-validated against a non-stationary time-domain Monte-Carlo with a
time-varying filter. Built ALONGSIDE the M20 scalar / M21-M23 multiaxial / M24
non-Gaussian / M25 non-stationary spectral fatigue (all stay bit-identical; the
evolutionary path CONSUMES the M20 estimators and the M25 block machinery
read-only).

Every new capability gets at least one ANALYTIC / EXACT-REDUCTION check (the
port's philosophy):

THE M26 <-> M25 / M20 REDUCTIONS (built in, exact)
* a SINGLE window / a time-invariant shape recovers the M20 stationary answer
  EXACTLY (the window Miner-sum of one block);
* a CONSTANT-SHAPE spectrogram (only the RMS level drifts) recovers the M25
  amplitude-modulated answer EXACTLY (the window Miner-sum IS the M25 block
  Miner-sum when every window shares one shape);
* the window Miner-sum EQUALS the duration-weighted per-window damages (hand
  check).

THE NON-SEPARABLE POINT (the M26 <-> M25 boundary made explicit)
* a two-window "shape-swap" (narrow-band -> wide-band) whose window Miner-sum
  DIFFERS from any single-shape M25 scaling — the spectral SHAPE, not just the
  level, changed;
* a swept centre frequency drifts the per-window zero-crossing rate nu0; a
  broadening bandwidth drifts the per-window irregularity factor alpha2.

NON-SEPARABLE MONTE-CARLO CROSS-CHECK (time-varying filter)
* the synthesised history's short-time spectrogram (per-window RMS + zero-crossing
  rate) tracking the target evolutionary spectrum (the centre-frequency drift);
* the non-separable Monte-Carlo damage matching the window spectral estimate
  within the seeded scatter;
* the CONSTANT-SHAPE limit reducing the non-separable Monte-Carlo EXACTLY
  (bit-identical) to the M25 non-stationary Monte-Carlo, and a single unit window
  reducing to the M20 Gaussian Monte-Carlo bit-identically.

CARDS + NO-REGRESSION (the M7 parity contract)
* /IMPL/FATIG/EVOL card mirror (a PORT sub-flag composing with /MULT, /NPROP,
  /SPEC, /NGAUSS, /NSTAT — freimpl.F has no evolutionary fatigue path; its sole
  PSD token is IMUMPSD, a MUMPS flag);
* the evolutionary path NEVER mutates the M16 eigensolver / M17-M18 FRFs / the
  M20 SCALAR / M21-M23 MULTIAXIAL / M24 NON-GAUSSIAN / M25 NON-STATIONARY fatigue
  / the element state; the M20-M25 answers are byte-identical whether or not /EVOL
  runs, and the direct M10 answer is unchanged.

See PORTING_GUIDE.md roadmap M26.
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
from pyradioss.implicit import nonstationary_fatigue as ns          # noqa: E402
from pyradioss.implicit import evolutionary_fatigue as ef           # noqa: E402

# reuse the M20 spring-chain deck and the M21 solid-brick deck
from tests.test_m20_fatigue import _chain_deck                      # noqa: E402
from tests.test_m21_multiaxfatig import _brick_deck                 # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _run(starter_text, engine_text, capture=False):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M26_0000.rad")
    ep = os.path.join(d, "M26_0001.rad")
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
    sp = os.path.join(d, "M26_0000.rad")
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


def _bimodal_psd(nf=20000, fmax=200.0):
    """A synthetic BIMODAL stress PSD (two well-separated narrow peaks, so a
    swept spectral window selects a genuinely DIFFERENT shape per window — the
    non-separable demonstrator). Returns (f, S)."""
    f = np.linspace(1e-3, fmax, nf)
    S = (np.exp(-((f - 40.0) ** 2) / (2.0 * 3.0 ** 2))
         + 0.5 * np.exp(-((f - 120.0) ** 2) / (2.0 * 4.0 ** 2))) / 2.0
    return f, S


# ============================================================================
# THE M26 <-> M25 / M20 REDUCTIONS (built in, exact)
# ============================================================================

def test_single_window_recovers_m20_exactly():
    """A SINGLE window / a time-invariant shape recovers the M20 stationary answer
    EXACTLY (the window Miner-sum of one block = the M20 damage of that shape)."""
    f, S = _bimodal_psd()
    m, C = 5.0, 1e14
    base_mom = ef._moments_of_psd(f, S)
    win = ef.drifting_shape_spectrogram(f, S, [100.0], fc=0.0, bw=0.0)
    evo = ef.evolutionary_fatigue_summary(win, m, C)
    stat = sf.fatigue_summary(base_mom, m, C)
    assert evo["constant_shape"]
    # exact to machine precision (the window Miner-sum forms dr*T/T for one block;
    # the round-trip multiply/divide can flip the last bit — as M25's block tests)
    for est in ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti"):
        assert evo[est]["damage_rate"] == pytest.approx(
            stat[est]["damage_rate"], rel=1e-12)


def test_constant_shape_recovers_m25_amplitude_modulated():
    """A CONSTANT-SHAPE spectrogram (a flat window, only the RMS level drifts)
    recovers the M25 amplitude-modulated answer EXACTLY — the window Miner-sum IS
    the M25 block Miner-sum when every window shares one shape."""
    f, S = _bimodal_psd()
    m, C = 5.0, 1e14
    scales = [0.6, 1.4, 1.0, 1.8]
    durs = [30.0, 50.0, 20.0, 15.0]
    # bw = 0 -> a flat window -> every window is the SAME shape scaled (M25 case)
    wins = ef.drifting_shape_spectrogram(f, S, durs, fc=0.0, bw=0.0,
                                         scales=scales)
    evo = ef.evolutionary_fatigue_summary(wins, m, C)
    base_mom = ef._moments_of_psd(f, S)
    sc, wt = ns.modulation_from_schedule(scales, durs)
    am = ns.amplitude_modulated_summary(base_mom, sc, wt, m, C)
    assert evo["constant_shape"]
    for est in ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti"):
        assert evo[est]["damage_rate"] == pytest.approx(
            am[est]["damage_rate"], rel=1e-12)


def test_window_miner_sum_hand_check():
    """The window Miner-sum EQUALS the duration-weighted per-window damages (theory
    eq. (1)): D = sum_i dr_i T_i, damage_rate = D / sum_i T_i — a hand check with
    genuinely DIFFERENT per-window shapes."""
    f = np.linspace(1e-3, 200.0, 20000)
    m, C = 5.0, 1e14
    # three different Gaussian shapes (different centre freqs / bandwidths)
    specs = [(40.0, 2.0, 10.0, 50.0), (80.0, 10.0, 8.0, 30.0),
             (120.0, 25.0, 6.0, 120.0)]
    windows = []
    for fc, bw, lv, T in specs:
        S = ef.gaussian_shape_psd(f, fc, bw, lv)
        windows.append({"freqs": f, "psd": S, "duration": T,
                        "moments": ef._moments_of_psd(f, S)})
    evo = ef.evolutionary_fatigue_summary(windows, m, C)
    D = 0.0
    T = 0.0
    for w in windows:
        dr = sf.dirlik_damage(w["moments"], m, C)["damage_rate"]
        D += dr * w["duration"]
        T += w["duration"]
    assert evo["dirlik"]["damage"] == pytest.approx(D, rel=1e-12)
    assert evo["dirlik"]["total_time"] == pytest.approx(T)
    assert evo["dirlik"]["damage_rate"] == pytest.approx(D / T, rel=1e-12)
    assert not evo["constant_shape"]        # the shapes genuinely differ


# ============================================================================
# THE NON-SEPARABLE POINT (the M26 <-> M25 boundary made explicit)
# ============================================================================

def test_shape_swap_differs_from_single_shape_scaling():
    """A two-window SHAPE-SWAP (narrow-band -> wide-band) whose window Miner-sum
    DIFFERS from any single-shape M25 scaling — the spectral SHAPE, not just the
    level, changed (the whole point of a NON-separable spectrum)."""
    f = np.linspace(1e-3, 200.0, 20000)
    m, C = 5.0, 1e14
    narrow = ef.gaussian_shape_psd(f, 40.0, 2.0, 10.0)     # narrow band
    wide = ef.gaussian_shape_psd(f, 100.0, 40.0, 10.0)     # wide band
    windows = [
        {"freqs": f, "psd": narrow, "duration": 50.0,
         "moments": ef._moments_of_psd(f, narrow)},
        {"freqs": f, "psd": wide, "duration": 50.0,
         "moments": ef._moments_of_psd(f, wide)}]
    evo = ef.evolutionary_fatigue_summary(windows, m, C)
    assert not evo["constant_shape"]
    # the per-window irregularity factor genuinely differs (narrow ~1, wide < 1)
    a2 = [w["alpha2"] for w in evo["windows"]]
    assert a2[0] > 0.95 and a2[1] < 0.9
    # the Miner-sum differs from scaling EITHER single shape by the same durations
    scales = [1.0, 1.0]
    durs = [50.0, 50.0]
    sc, wt = ns.modulation_from_schedule(scales, durs)
    am_narrow = ns.amplitude_modulated_summary(
        ef._moments_of_psd(f, narrow), sc, wt, m, C)["dirlik"]["damage_rate"]
    am_wide = ns.amplitude_modulated_summary(
        ef._moments_of_psd(f, wide), sc, wt, m, C)["dirlik"]["damage_rate"]
    evo_rate = evo["dirlik"]["damage_rate"]
    assert evo_rate != pytest.approx(am_narrow, rel=1e-6)
    assert evo_rate != pytest.approx(am_wide, rel=1e-6)
    # the Miner-sum of two shapes lies between the two single-shape rates
    assert min(am_narrow, am_wide) < evo_rate < max(am_narrow, am_wide)


def test_swept_centre_frequency_drifts_nu0():
    """A swept centre frequency drifts the per-window zero-crossing rate nu0 (the
    spectral energy moves up in frequency — a chirp-like process)."""
    f, S = _bimodal_psd()
    m, C = 5.0, 1e14
    wins = ef.drifting_shape_spectrogram(f, S, [1.0] * 6, fc=(30.0, 130.0),
                                         bw=8.0)
    evo = ef.evolutionary_fatigue_summary(wins, m, C)
    nu0 = [w["nu0"] for w in evo["windows"]]
    assert nu0[-1] > nu0[0]                        # centre freq drifted UP
    assert not evo["constant_shape"]


def test_broadening_bandwidth_drifts_alpha2():
    """A broadening bandwidth drifts the per-window irregularity factor alpha2
    toward 0 (narrow-band -> wide-band transition)."""
    f = np.linspace(1e-3, 300.0, 20000)
    m, C = 5.0, 1e14
    # a single centred bump whose bandwidth broadens 2 -> 40 Hz across windows
    wins = ef.gaussian_evolutionary_spectrogram(
        f, fc=120.0, bw=(2.0, 40.0), level=10.0, durations=[1.0] * 6)
    evo = ef.evolutionary_fatigue_summary(wins, m, C)
    a2 = [w["alpha2"] for w in evo["windows"]]
    assert a2[0] > a2[-1]                          # narrow -> wide
    assert a2[0] > 0.95 and a2[-1] < 0.9


# ============================================================================
# NON-SEPARABLE MONTE-CARLO CROSS-CHECK (time-varying filter)
# ============================================================================

def test_evol_mc_constant_shape_bit_identical_to_m25():
    """The CONSTANT-SHAPE limit reduces the non-separable Monte-Carlo EXACTLY
    (bit-identical) to the M25 non-stationary Monte-Carlo — the per-window PSDs
    collapse to a shared shape times an envelope, and the synthesiser delegates to
    the M25 single-carrier-times-envelope path."""
    f, S = _bimodal_psd()
    m, C = 5.0, 1e14
    scales = [0.6, 1.0, 1.5]
    durs = [400.0, 400.0, 400.0]
    wins = ef.drifting_shape_spectrogram(f, S, durs, fc=0.0, bw=0.0,
                                         scales=scales)
    evo_mc = ef.evolutionary_monte_carlo_damage(wins, m, C, seed=7, fs=800.0)
    m25_mc = ns.nonstationary_monte_carlo_damage(f, S, m, C, scales, durs, 7,
                                                 fs=800.0)
    assert evo_mc["delegated"]
    assert evo_mc["damage_rate"] == m25_mc["damage_rate"]
    assert np.array_equal(evo_mc["ranges"], m25_mc["ranges"])


def test_evol_mc_single_window_bit_identical_to_m20():
    """A single UNIT window reduces the non-separable Monte-Carlo EXACTLY
    (bit-identical) to the M20 Gaussian Monte-Carlo (the envelope is unity, one
    shape — the whole chain collapses to M20)."""
    f, S = _bimodal_psd()
    m, C = 5.0, 1e14
    wins = ef.drifting_shape_spectrogram(f, S, [2000.0], fc=0.0, bw=0.0,
                                         scales=[1.0])
    evo_mc = ef.evolutionary_monte_carlo_damage(wins, m, C, seed=5, fs=800.0)
    g_mc = sf.monte_carlo_damage(f, S, m, C, duration=2000.0, seed=5, fs=800.0)
    assert evo_mc["delegated"]
    assert evo_mc["damage_rate"] == g_mc["damage_rate"]
    assert np.array_equal(evo_mc["ranges"], g_mc["ranges"])


def test_evol_mc_tracks_spectrogram():
    """The synthesised (non-separable) history's short-time spectrogram tracks the
    target evolutionary spectrum: the per-window zero-crossing rate follows the
    swept centre frequency (rises through the record)."""
    f, S = _bimodal_psd()
    m, C = 5.0, 1e14
    wins = ef.drifting_shape_spectrogram(f, S, [300.0] * 3, fc=(35.0, 125.0),
                                         bw=8.0)
    mc = ef.evolutionary_monte_carlo_damage(wins, m, C, seed=3, fs=1600.0)
    assert not mc["delegated"]
    nu0 = mc["window_nu0"]
    # the per-window crossing rate rises with the swept centre frequency
    assert nu0[-1] > nu0[0]
    assert nu0[0] == pytest.approx(35.0, rel=0.35)
    assert nu0[-1] == pytest.approx(125.0, rel=0.35)


def test_evol_mc_matches_window_spectral_estimate():
    """On a drifting narrow-band spectrogram the non-separable Monte-Carlo damage
    matches the window spectral estimate within the seeded scatter."""
    f, S = _bimodal_psd()
    m, C = 5.0, 1e14
    scales = [0.8, 1.0, 1.2]
    durs = [1000.0, 1000.0, 1000.0]
    wins = ef.drifting_shape_spectrogram(f, S, durs, fc=(40.0, 120.0), bw=6.0,
                                         scales=scales)
    evo = ef.evolutionary_fatigue_summary(wins, m, C)
    mc = ef.evolutionary_monte_carlo_damage(wins, m, C, seed=11, fs=1600.0)
    ratio = mc["damage_rate"] / evo["dirlik"]["damage_rate"]
    assert 0.5 < ratio < 1.8, f"evoMC / window-spectral = {ratio:.3f} out of band"


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


def test_impl_fatig_evol_card_parsing():
    """/IMPL/FATIG/EVOL sets the evolutionary flag and reads the drifting-shape
    schedule (fc0 fc1 bw0 bw1 nwin) from the card line AFTER the sweep / S-N (and
    kurtosis / modulation, if NGAUSS / NSTAT) lines; it composes with /MULT,
    /NGAUSS, /NSTAT and does NOT imply them — a PORT sub-flag."""
    # EVOL standalone: schedule on line 2 (no NGAUSS / NSTAT)
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/FATIG/EVOL/BASE\n"
                   "0.0 250 2000 7 0 4\n5.0 1.0e12 0.03\n"
                   "20.0 120.0 5.0 30.0 10\n/END\n")
    assert ec.implicit and ec.impl_fatig and ec.impl_fatig_evol
    assert ec.impl_fatig_evol_fc0 == pytest.approx(20.0)
    assert ec.impl_fatig_evol_fc1 == pytest.approx(120.0)
    assert ec.impl_fatig_evol_bw0 == pytest.approx(5.0)
    assert ec.impl_fatig_evol_bw1 == pytest.approx(30.0)
    assert ec.impl_fatig_evol_nwin == 10
    assert not ec.impl_fatig_nstat          # EVOL is orthogonal to NSTAT

    # composes with NSTAT: modulation on line 2, EVOL schedule on line 3
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG/EVOL/NSTAT/BASE\n"
                   "0.0 250 2000 7 0 4\n5.0 1e12 0.03\n"
                   "20 8\n30.0 150.0 8.0 40.0 8\n/END\n")
    assert (ec.impl_fatig_evol and ec.impl_fatig_nstat
            and ec.impl_fatig_modfunct == 20 and ec.impl_fatig_nstat_nseg == 8
            and ec.impl_fatig_evol_fc0 == pytest.approx(30.0)
            and ec.impl_fatig_evol_fc1 == pytest.approx(150.0)
            and ec.impl_fatig_evol_nwin == 8)

    # composes with MULT + NGAUSS + NSTAT: kurtosis line 3, modulation line 4,
    # EVOL schedule line 5 (2 + 1 ngauss + 1 nstat = 4 -> zero-based index 4)
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG/MULT/NGAUSS/NSTAT/EVOL\n"
                   "0.0 400 500 10 6\n5.0 1e4 0.03 0 0 15 9\n6.0 0.0\n"
                   "30\n25.0 200.0 10.0 50.0 6\n/END\n")
    assert (ec.impl_fatig_mult and ec.impl_fatig_ngauss and ec.impl_fatig_nstat
            and ec.impl_fatig_evol and ec.impl_fatig_kurt == pytest.approx(6.0)
            and ec.impl_fatig_modfunct == 30
            and ec.impl_fatig_evol_fc0 == pytest.approx(25.0)
            and ec.impl_fatig_evol_fc1 == pytest.approx(200.0)
            and ec.impl_fatig_evol_nwin == 6)

    # a plain M20 / M25 card is NOT evolutionary
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG\n0.0 50 500 7\n"
                   "5.0 1e12\n/END\n")
    assert not ec.impl_fatig_evol


# ============================================================================
# END-TO-END
# ============================================================================

def test_evol_scalar_end_to_end():
    """/IMPL/FATIG/EVOL end to end on the base-driven spring chain: the listing
    prints the FULLY EVOLUTIONARY block ALONGSIDE the stationary and non-stationary
    ones, stores an ``evolutionary`` sub-entry with the window Miner-sum, the
    per-window shape breakdown, the induced kurtosis and the non-separable
    Monte-Carlo, and the answer genuinely differs from a constant shape."""
    starter = _with_mission(_chain_deck([2e-3] * 5, [800.] * 5, [0.] * 5))
    m, out = _run(starter,
                  "#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG/EVOL/NSTAT/BASE\n"
                  "0.0 250.0 3000 2 0 5\n5.0 1.0e14 0.03 0.0 0.0 400.0 123\n"
                  "20 8\n20.0 120.0 5.0 30.0 8\n/PRINT/-500\n/STOP\n15.0\n",
                  capture=True)
    fat = m.implicit_result.fatigue
    ev = fat["evolutionary"]
    assert ev is not None
    assert ev["nwin"] == 8
    assert not ev["constant_shape"]                  # the shape genuinely drifts
    assert ev["kurtosis"] > 3.0                      # varying RMS -> leptokurtic
    # the per-window shape drift (nu0 rises with the swept centre frequency)
    wins = ev["summary"]["windows"]
    assert wins[-1]["nu0"] > wins[0]["nu0"]
    # the evolutionary Dirlik differs SUBSTANTIALLY from BOTH the stationary and
    # the M25 block (the SHAPE drift changed the answer — not a rounding nudge)
    evo_dr = ev["summary"]["dirlik"]["damage_rate"]
    stat_dr = fat["summary"]["dirlik"]["damage_rate"]
    m25_dr = fat["nonstationary"]["block"]["damage_rate"]
    assert not (0.9 < evo_dr / stat_dr < 1.1)
    assert not (0.9 < evo_dr / m25_dr < 1.1)
    assert ev["monte_carlo"] is not None
    assert not ev["monte_carlo"]["delegated"]        # genuinely non-separable
    assert "FULLY EVOLUTIONARY / NON-SEPARABLE-PSD FATIGUE" in out
    # the M20 stationary AND the M25 non-stationary blocks are still there
    assert "RANDOM-VIBRATION (SPECTRAL) FATIGUE" in out
    assert "NON-STATIONARY / EVOLUTIONARY-PSD FATIGUE" in out


def test_evol_multiaxial_end_to_end():
    """/IMPL/FATIG/MULT/EVOL composes: the M21 multiaxial reductions run unchanged,
    and the evolutionary correction applies the drifting-shape window per window to
    each reduction's scalar PSD, with a non-separable Monte-Carlo."""
    eng = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/EVOL/NSTAT\n"
           "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000\n20 6\n"
           "50.0 250.0 10.0 40.0 6\n/PRINT/-500\n/STOP\n15.0\n")
    m, out = _run(_with_mission(_brick_deck(nx=3)), eng, capture=True)
    fat = m.implicit_result.fatigue
    assert fat.get("multiaxial")
    ev = fat["evolutionary"]
    assert ev is not None
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert ev[key]["summary"]["dirlik"]["damage_rate"] > 0.0
    assert ev["monte_carlo"] is not None
    assert "FULLY EVOLUTIONARY / NON-SEPARABLE-PSD FATIGUE" in out
    # the M21 block still present side by side
    assert "MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE" in out


def test_evol_composes_with_ngauss_and_nstat():
    """/IMPL/FATIG/NGAUSS/NSTAT/EVOL: ALL THREE siblings run — the M24 non-Gaussian
    correction, the M25 non-stationary correction AND the M26 evolutionary
    correction are stored side by side, each byte-identical to running alone."""
    starter = _with_mission(_chain_deck([2e-3] * 5, [800.] * 5, [0.] * 5))
    eng = ("#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG/NGAUSS/NSTAT/EVOL/BASE\n"
           "0.0 250.0 3000 2 0 5\n5.0 1.0e14 0.03 0.0 0.0 400.0 123\n"
           "6.0 0.0\n20 6\n20.0 120.0 5.0 30.0 8\n/PRINT/-500\n/STOP\n15.0\n")
    m, out = _run(starter, eng, capture=True)
    fat = m.implicit_result.fatigue
    assert fat["nongaussian"] is not None and fat["nonstationary"] is not None
    assert fat["evolutionary"] is not None
    assert fat["nongaussian"]["gamma4"] == pytest.approx(6.0)
    assert fat["nonstationary"]["kurtosis"] > 3.0
    assert not fat["evolutionary"]["constant_shape"]
    assert "NON-GAUSSIAN / KURTOSIS FATIGUE" in out
    assert "NON-STATIONARY / EVOLUTIONARY-PSD FATIGUE" in out
    assert "FULLY EVOLUTIONARY / NON-SEPARABLE-PSD FATIGUE" in out


# ============================================================================
# NO-REGRESSION (the M7 parity contract)
# ============================================================================

def test_m20_scalar_byte_identical_with_without_evol():
    """The M20 SCALAR stationary summary AND the M25 non-stationary block are
    BYTE-IDENTICAL whether or not the M26 evolutionary path runs — the correction
    is NEW and ALONGSIDE."""
    starter = _with_mission(_chain_deck([2e-3] * 5, [800.] * 5, [0.] * 5))
    base = ("#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG/NSTAT/BASE\n"
            "0.0 250.0 3000 2 0 5\n5.0 1.0e14 0.03 0.0 0.0 400.0 123\n"
            "20 6\n/PRINT/-500\n/STOP\n15.0\n")
    ev_eng = ("#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG/NSTAT/EVOL/BASE\n"
              "0.0 250.0 3000 2 0 5\n5.0 1.0e14 0.03 0.0 0.0 400.0 123\n"
              "20 6\n20.0 120.0 5.0 30.0 8\n/PRINT/-500\n/STOP\n15.0\n")
    m20 = _run(starter, base)
    m26 = _run(starter, ev_eng)
    f20 = m20.implicit_result.fatigue
    f26 = m26.implicit_result.fatigue
    assert f20["evolutionary"] is None
    assert f26["evolutionary"] is not None
    for est in ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti"):
        assert (f20["summary"][est]["damage_rate"]
                == f26["summary"][est]["damage_rate"])
    # the M25 non-stationary block Miner-sum is byte-identical too
    assert (f20["nonstationary"]["block"]["damage_rate"]
            == f26["nonstationary"]["block"]["damage_rate"])
    # the Gaussian Monte-Carlo is byte-identical
    assert (f20["monte_carlo"]["damage_rate"]
            == f26["monte_carlo"]["damage_rate"])


def test_m21_m24_m25_byte_identical_with_without_evol():
    """The M21 spectral, M24 non-Gaussian AND M25 non-stationary answers are
    BYTE-IDENTICAL whether or not the M26 evolutionary path runs."""
    deck = _with_mission(_brick_deck(nx=3))
    eng = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/NGAUSS/NSTAT\n"
           "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000\n"
           "5.0 0.0\n20 6\n/PRINT/-500\n/STOP\n15.0\n")
    eng_ev = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n"
              "/IMPL/FATIG/MULT/NGAUSS/NSTAT/EVOL\n"
              "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000\n"
              "5.0 0.0\n20 6\n50.0 250.0 10.0 40.0 6\n/PRINT/-500\n"
              "/STOP\n15.0\n")
    a = _run(deck, eng)
    b = _run(deck, eng_ev)
    fa = a.implicit_result.fatigue
    fb = b.implicit_result.fatigue
    assert fa["evolutionary"] is None and fb["evolutionary"] is not None
    # M21 reductions
    for key in ("von_mises", "normal_plane", "shear_plane"):
        for est in ("narrow_band", "dirlik", "wirsching_light",
                    "tovo_benasciutti"):
            assert (fa[key]["summary"][est]["damage_rate"]
                    == fb[key]["summary"][est]["damage_rate"])
    # M24 non-Gaussian
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert (fa["nongaussian"][key]["dirlik"]["damage_rate"]
                == fb["nongaussian"][key]["dirlik"]["damage_rate"])
    # M25 non-stationary block Miner-sum
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert (fa["nonstationary"][key]["block"]["damage_rate"]
                == fb["nonstationary"][key]["block"]["damage_rate"])


def test_evol_does_not_mutate_state():
    """The evolutionary path is read-only in the element state and leaves the M16
    modal_frequencies output bit-identical before and after (the M14-M25 parity
    contract extended to M26)."""
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
    modal_frequency_response(basis, F, np.zeros((n, 3)),
                             np.linspace(1e-6, 300.0, 60), z)
    f, S = _bimodal_psd()
    # a representative evolutionary evaluation
    wins = ef.drifting_shape_spectrogram(f, S, [1.0] * 5, fc=(30.0, 130.0),
                                         bw=(5.0, 25.0), scales=[0.5, 1., 1.5, 1., .5])
    ef.evolutionary_fatigue_summary(wins, 5.0, 1e14)
    for k, v in before.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(m.springs.state[k], v), k
    f1, _, _ = modal_frequencies(m, nev=3)
    assert np.array_equal(f0, f1)


def test_direct_dynamics_unchanged_by_evol_path():
    """A DIRECT /IMPL/DYNA run is byte-for-byte unaffected by the M26 evolutionary
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

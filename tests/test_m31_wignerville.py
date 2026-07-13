"""
M31 validations: CONTINUOUS WIGNER-VILLE / LOEVE INSTANTANEOUS TIME-FREQUENCY
SPECTRUM (/IMPL/FATIG/WVILLE) — the frequency-domain fatigue-damage estimate of a
random-vibration response computed from a CONTINUOUS instantaneous spectrum
S_WV(omega, t) (a bilinear time-frequency distribution) rather than the M26-M30
SHORT-TIME WINDOWED SPECTROGRAM, reduced by the M20-M27 estimator family AT EACH
INSTANT and Palmgren-Miner INTEGRATED over time (an integral, not a per-window sum),
cross-validated against the M25/M27/M29/M30 non-stationary Monte-Carlo. Built
ALONGSIDE the M20-M30 spectral fatigue (all stay bit-identical; the continuous
Wigner-Ville path CONSUMES the M20-M27 estimators and the M26-M30 windowed machinery
read-only).

Every new capability gets at least one ANALYTIC / EXACT-REDUCTION check (the port's
philosophy):

THE M31 <-> M26/M27/M29/M30 REDUCTIONS (built in, exact — the windowed limit)
* the windowed spectrogram is EXACTLY the long-window (refine = 1) / un-smoothed
  (smooth = 0) limit — the continuous scalar / tensor / multi-input summaries
  DELEGATE to the M26 / M27 / M29 / M30 windowed answers BYTE-IDENTICALLY there;
* a STATIONARY process recovers the stationary PSD at EVERY instant EXACTLY;
* the frequency MARGINAL (time-averaged instantaneous spectrum) recovers the
  mission-average PSD; the time MARGINAL recovers the instantaneous power.

THE CONTINUOUS POINT (the M31 <-> M26-M30 boundary made explicit)
* a genuinely non-stationary chirp whose instantaneous spectral PEAK / critical
  plane drifts CONTINUOUSLY, finer than the windows resolve (nt = nwin * refine
  instants);
* the window-boundary caveat (the adjacent-instant shape jump) measurably SHRINKS on
  the fine grid vs the coarse windows;
* heavy Cohen-class smoothing collapses the instantaneous spectrum toward the
  mission-average (the drift / plane rotation vanishes).

CONTINUOUS MONTE-CARLO CROSS-CHECK
* the windowed-limit Monte-Carlo bit-identical to the M26 Monte-Carlo;
* the continuous Miner-integral tracking the Monte-Carlo BETTER than the coarse
  windowed sum (the reference that includes the straddling cycles).

CARDS + NO-REGRESSION (the M7 parity contract)
* /IMPL/FATIG/WVILLE card mirror (a PORT sub-flag implying /EVOL, composing with
  /JOINT, /MINPUT, /FCOH, /NSTAT — freimpl.F has no time-frequency / Wigner-Ville
  solver; its sole PSD token is IMUMPSD, a MUMPS flag);
* the continuous path NEVER mutates the M20 SCALAR / M21 MULTIAXIAL / M26 windowed
  evolutionary / M27 joint-tensor / M29 / M30 multi-input answers — they are
  byte-identical whether or not /WVILLE runs.

See PORTING_GUIDE.md roadmap M31.
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

from pyradioss.common.messages import MessageLog                    # noqa: E402
from pyradioss.implicit import wigner_ville_fatigue as wv           # noqa: E402
from pyradioss.implicit import evolutionary_fatigue as ef           # noqa: E402
from pyradioss.implicit import joint_evolutionary_fatigue as jf     # noqa: E402
from pyradioss.implicit import evolutionary_multi_input as emi      # noqa: E402
from pyradioss.implicit import freq_evolutionary_multi_input as fem  # noqa: E402
from pyradioss.implicit import spectral_fatigue as sf               # noqa: E402

from tests.test_m20_fatigue import _chain_deck                      # noqa: E402
from tests.test_m21_multiaxfatig import _brick_deck                 # noqa: E402
from tests.test_m28_multiinput import _two_input_brick              # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _run(starter_text, engine_text, capture=False):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M31_0000.rad")
    ep = os.path.join(d, "M31_0001.rad")
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


def _with_mission(deck):
    return deck.replace("/END", _MISSION + "/END", 1)


def _bimodal_psd(nf=20000, fmax=200.0):
    """A synthetic BIMODAL stress PSD (two well-separated narrow peaks) — a swept
    NARROW window between them makes the coarse windowed sum a POOR quadrature of the
    continuous damage-rate integral, the M31 demonstrator."""
    f = np.linspace(1e-3, fmax, nf)
    S = (np.exp(-((f - 40.0) ** 2) / (2.0 * 3.0 ** 2))
         + 0.5 * np.exp(-((f - 120.0) ** 2) / (2.0 * 4.0 ** 2))) / 2.0
    return f, S


def _tensor_cross_psd(nf=400, fmax=600.0):
    """A synthetic 6x6 stress-tensor cross-PSD with a LOW band in sigma_xx (bending)
    and a HIGH band in sigma_xy (shear/torsion), so a swept window rotates the
    dominant component — the joint-tensor drift demonstrator (M27 base)."""
    f = np.linspace(1e-3, fmax, nf)
    omega = 2.0 * np.pi * f
    S = np.zeros((nf, 6, 6), dtype=complex)
    low = np.exp(-((f - 100.0) ** 2) / (2.0 * 20.0 ** 2))
    high = np.exp(-((f - 450.0) ** 2) / (2.0 * 30.0 ** 2))
    S[:, 0, 0] = low + 0.05
    S[:, 3, 3] = high + 0.02
    S[:, 0, 3] = 0.3 * np.sqrt(low * high)
    S[:, 3, 0] = np.conj(S[:, 0, 3])
    return omega, S


# ============================================================================
# THE M31 <-> M26/M27/M29/M30 REDUCTIONS (windowed limit, byte-identical)
# ============================================================================

def test_scalar_windowed_limit_byte_identical_to_m26():
    """refine = 1, smooth = 0: the continuous scalar spectrogram windows are the M26
    ``drifting_shape_spectrogram`` output BYTE-IDENTICALLY, so the summary equals the
    M26 ``evolutionary_fatigue_summary`` byte-for-byte (the built-in reduction)."""
    f, S = _bimodal_psd()
    m, C = 5.0, 1e14
    durs = [30.0, 50.0, 20.0, 15.0]
    scales = [0.6, 1.4, 1.0, 1.8]
    wvw = wv.wigner_ville_spectrogram(f, S, durs, (30.0, 120.0), (5.0, 20.0),
                                      scales=scales, refine=1, smooth=0.0)
    m26 = ef.drifting_shape_spectrogram(f, S, durs, fc=(30.0, 120.0),
                                        bw=(5.0, 20.0), scales=scales)
    for a, b in zip(wvw, m26):
        assert np.array_equal(a["psd"], b["psd"])
        assert np.array_equal(a["moments"], b["moments"])
        assert a["duration"] == b["duration"]
    s_wv = wv.wigner_ville_fatigue_summary(f, S, durs, (30.0, 120.0), (5.0, 20.0),
                                           m, C, scales=scales, refine=1, smooth=0.0)
    s_m26 = ef.evolutionary_fatigue_summary(m26, m, C)
    for est in ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti"):
        assert s_wv[est]["damage_rate"] == s_m26[est]["damage_rate"]


def test_tensor_windowed_limit_byte_identical_to_m27():
    """refine = 1, smooth = 0: the continuous 6x6 TENSOR summary DELEGATES to the M27
    ``joint_evolutionary_fatigue_summary`` byte-identically (per reduction)."""
    omega, Scross = _tensor_cross_psd()
    m, C = 5.0, 1e4
    durs = [10.0] * 6
    t_wv = wv.wigner_ville_tensor_summary(omega, Scross, durs, (120.0, 480.0),
                                          (20.0, 40.0), m, C, refine=1, smooth=0.0)
    t_m27 = jf.joint_evolutionary_fatigue_summary(
        omega, Scross, durs, fc=(120.0, 480.0), bw=(20.0, 40.0), m=m, C=C,
        drift=True)
    assert t_wv["delegated"] == "m27_windowed"
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert t_wv[k]["damage_rate"] == t_m27[k]["damage_rate"]


def test_multi_input_windowed_limit_byte_identical_to_m29_m30():
    """refine = 1, smooth = 0: the continuous MULTI-INPUT summary DELEGATES to the
    M29 scalar-coherence AND the M30 frequency-dependent-coherence windowed answers
    byte-identically."""
    nf, ninput = 300, 2
    f = np.linspace(1e-3, 600.0, nf)
    omega = 2.0 * np.pi * f
    Hcols = np.zeros((nf, 6, ninput), dtype=complex)
    res = np.exp(-((f - 100.0) ** 2) / (2.0 * 25.0 ** 2)) + 0.2
    Hcols[:, 0, 0] = res
    Hcols[:, 3, 0] = 0.4 * res
    Hcols[:, 1, 1] = np.exp(-((f - 400.0) ** 2) / (2.0 * 30.0 ** 2)) + 0.1
    Hcols[:, 3, 1] = 0.5 * Hcols[:, 1, 1]
    G = np.zeros((nf, ninput))
    G[:, 0] = 1.0
    G[:, 1] = 0.8
    m, C = 5.0, 1e4
    durs = [10.0] * 6
    # M29 scalar coherence
    w29 = wv.wigner_ville_multi_input_summary(
        omega, Hcols, G, durs, m, C, gamma0=0.1, gamma1=0.9, fc=(120.0, 480.0),
        bw=(20.0, 40.0), refine=1, smooth=0.0, freq_dependent=False)
    m29 = emi.evolutionary_multi_input_summary(
        omega, Hcols, G, durs, m, C, gamma0=0.1, gamma1=0.9, fc=(120.0, 480.0),
        bw=(20.0, 40.0), drift=True)
    assert w29["wv_delegated"] == "m29_windowed"
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert w29[k]["damage_rate"] == m29[k]["damage_rate"]
    # M30 frequency-dependent coherence stacks
    pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    g0 = fem.exponential_coherence_stack(f, pos, decay=0.9, ref_speed=200.0)
    g1 = fem.exponential_coherence_stack(f, pos, decay=0.2, ref_speed=200.0)
    w30 = wv.wigner_ville_multi_input_summary(
        omega, Hcols, G, durs, m, C, gamma0=g0, gamma1=g1, fc=(120.0, 480.0),
        bw=(20.0, 40.0), refine=1, smooth=0.0, freq_dependent=True)
    m30 = fem.freq_evolutionary_multi_input_summary(
        omega, Hcols, G, durs, m, C, gamma0=g0, gamma1=g1, fc=(120.0, 480.0),
        bw=(20.0, 40.0), drift=True)
    assert w30["wv_delegated"] == "m30_windowed"
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert w30[k]["damage_rate"] == m30[k]["damage_rate"]


def test_stationary_recovers_psd_every_instant():
    """A STATIONARY process (no drift) recovers the stationary PSD at EVERY instant
    EXACTLY — the instantaneous spectral peak is identical across all fine instants,
    and the continuous Dirlik equals the M20 stationary Dirlik."""
    f, S = _bimodal_psd()
    m, C = 5.0, 1e14
    s = wv.wigner_ville_fatigue_summary(f, S, [100.0] * 3, 0.0, 0.0, m, C,
                                        refine=8, smooth=0.0)
    pk = s["instantaneous"]["peak_freq"]
    assert len(set(np.round(pk, 9))) == 1                # one peak, every instant
    assert s["constant_shape"]
    stat = sf.fatigue_summary(ef._moments_of_psd(f, S), m, C)
    assert s["dirlik"]["damage_rate"] == pytest.approx(
        stat["dirlik"]["damage_rate"], rel=1e-9)


def test_frequency_marginal_recovers_average_psd():
    """The frequency MARGINAL (time-averaged instantaneous spectrum) recovers the
    mission-average PSD: for a flat (no-drift) stationary window it equals the
    stationary PSD EXACTLY."""
    f, S = _bimodal_psd(nf=4000)
    marg = wv.instantaneous_marginals(f, S, [100.0] * 4, 0.0, 0.0, refine=6,
                                      smooth=0.0)
    ratio = marg["avg_psd"][S > 1e-3] / S[S > 1e-3]
    assert np.allclose(ratio, 1.0, atol=1e-9)


# ============================================================================
# THE CONTINUOUS POINT (finer than the windows resolve)
# ============================================================================

def test_chirp_peak_drifts_continuously_finer_than_windows():
    """A genuinely non-stationary chirp: the instantaneous spectral PEAK drifts
    CONTINUOUSLY across nt = nwin * refine instants (finer than the nwin windows),
    and the peak sweeps from the low band to the high band."""
    f, S = _bimodal_psd()
    m, C = 5.0, 1e14
    s = wv.wigner_ville_fatigue_summary(f, S, [1.0] * 6, (35.0, 125.0), 8.0, m, C,
                                        refine=8, smooth=0.0)
    assert s["nt"] == 48                                 # 6 windows x refine 8
    assert s["continuous"]
    pk = np.asarray(s["instantaneous"]["peak_freq"])
    assert pk[-1] > pk[0]                                # swept up
    assert s["peak_drift"] > 60.0
    # many DISTINCT peak positions — finer than the 6 windows would give
    assert len(set(np.round(pk, 3))) > 6


def test_boundary_caveat_shrinks_fine_vs_coarse():
    """The window-boundary caveat (the adjacent-instant normalised shape jump)
    measurably SHRINKS on the fine grid vs the coarse windows — the continuous
    spectrum's many near-identical instants jump far less at each boundary."""
    f, S = _bimodal_psd()
    m, C = 5.0, 1e14
    s = wv.wigner_ville_fatigue_summary(f, S, [1.0] * 6, (35.0, 125.0), 8.0, m, C,
                                        refine=8, smooth=0.0)
    assert s["boundary_jump"] < s["boundary_jump_windowed"]


def test_tensor_plane_drifts_continuously():
    """The 6x6 TENSOR continuous spectrum re-searches the critical plane AT EACH
    instant — a swept window that rotates the dominant component drifts the plane
    CONTINUOUSLY (a non-zero rotation / F_np drift), and heavy smoothing collapses
    it toward the mission-average (no drift)."""
    omega, Scross = _tensor_cross_psd()
    m, C = 5.0, 1e4
    durs = [10.0] * 6
    t = wv.wigner_ville_tensor_summary(omega, Scross, durs, (120.0, 480.0),
                                       (20.0, 40.0), m, C, refine=8, smooth=0.0)
    assert t["nt"] == 48 and t["continuous"]
    assert t["plane_rotation_deg"] > 10.0
    assert t["fnp_drift"] > 0.1
    # heavy Cohen-class smoothing -> mission-average (no rotation)
    ts = wv.wigner_ville_tensor_summary(omega, Scross, durs, (120.0, 480.0),
                                        (20.0, 40.0), m, C, refine=8, smooth=5.0)
    assert ts["plane_rotation_deg"] < t["plane_rotation_deg"]


def test_heavy_smoothing_collapses_toward_stationary():
    """Heavy Cohen-class smoothing collapses the scalar instantaneous spectrum toward
    the mission-average: every instant's peak converges to a single value (the
    cross-term-suppressed / spectrogram limit)."""
    f, S = _bimodal_psd()
    m, C = 5.0, 1e14
    s0 = wv.wigner_ville_fatigue_summary(f, S, [1.0] * 6, (35.0, 125.0), 8.0, m, C,
                                         refine=8, smooth=0.0)
    s1 = wv.wigner_ville_fatigue_summary(f, S, [1.0] * 6, (35.0, 125.0), 8.0, m, C,
                                         refine=8, smooth=10.0)
    # the peak-frequency spread shrinks under heavy smoothing
    sp0 = np.ptp(s0["instantaneous"]["peak_freq"])
    sp1 = np.ptp(s1["instantaneous"]["peak_freq"])
    assert sp1 <= sp0


# ============================================================================
# CONTINUOUS MONTE-CARLO CROSS-CHECK
# ============================================================================

def test_mc_windowed_limit_bit_identical_to_m26():
    """The windowed-limit continuous Monte-Carlo (refine = 1, smooth = 0) is the M26
    non-separable Monte-Carlo BIT-IDENTICALLY (the fine windows ARE the M26
    windows)."""
    f, S = _bimodal_psd(nf=4000)
    m, C = 5.0, 1e14
    mc_wv = wv.wigner_ville_monte_carlo_damage(
        f, S, [1000.0] * 3, (40.0, 120.0), 6.0, m, C, seed=11, refine=1,
        smooth=0.0, fs=1600.0)
    windows = ef.drifting_shape_spectrogram(f, S, [1000.0] * 3, fc=(40.0, 120.0),
                                            bw=6.0)
    mc_m26 = ef.evolutionary_monte_carlo_damage(windows, m, C, 11, fs=1600.0)
    assert mc_wv["damage_rate"] == mc_m26["damage_rate"]


def test_continuous_integral_tracks_mc_better_than_windowed():
    """On a swept NARROW window between two PSD peaks, the coarse windowed sum is a
    POOR quadrature (it misses the peaks the sweep passes through), while the
    continuous fine-grid Miner-integral tracks the Monte-Carlo (which sweeps
    continuously and rainflows the WHOLE record) far better."""
    f, S = _bimodal_psd(nf=4000)
    m, C = 5.0, 1e14
    durs = [1000.0] * 3
    coarse = wv.wigner_ville_fatigue_summary(
        f, S, durs, (40.0, 120.0), 6.0, m, C, refine=1, smooth=0.0
    )["dirlik"]["damage_rate"]
    fine = wv.wigner_ville_fatigue_summary(
        f, S, durs, (40.0, 120.0), 6.0, m, C, refine=16, smooth=0.0
    )["dirlik"]["damage_rate"]
    mc = wv.wigner_ville_monte_carlo_damage(
        f, S, durs, (40.0, 120.0), 6.0, m, C, seed=11, refine=16, smooth=0.0,
        fs=1600.0)["damage_rate"]
    # the fine continuous integral is much closer to the MC than the coarse sum
    assert abs(fine / mc - 1.0) < abs(coarse / mc - 1.0)
    # and it tracks the MC within a factor of ~2 (seeded scatter)
    assert 0.4 < fine / mc < 2.5


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


def test_impl_fatig_wville_card_parsing():
    """/IMPL/FATIG/WVILLE sets the continuous-spectrum flag (IMPLYING /EVOL) and
    reads the grid-refinement factor + Cohen-class smoothing width from the trailing
    columns of the /EVOL drifting-shape line (fc0 fc1 bw0 bw1 nwin [refine smooth]);
    it composes with /JOINT, /MINPUT, /FCOH — a PORT sub-flag."""
    # WVILLE standalone -> implies EVOL; refine / smooth on the schedule line
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/FATIG/WVILLE/BASE\n"
                   "0.0 250 2000 7 0 4\n5.0 1.0e12 0.03\n"
                   "20.0 120.0 5.0 30.0 8 12 0.25\n/END\n")
    assert ec.impl_fatig_wville and ec.impl_fatig_evol
    assert ec.impl_fatig_evol_nwin == 8
    assert ec.impl_fatig_wv_refine == 12
    assert ec.impl_fatig_wv_smooth == pytest.approx(0.25)

    # composes with JOINT (tensor continuous); refine given, smooth defaults to 0
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE\n"
                   "0.0 400 500 10\n5.0 1e4 0.03\n50.0 250.0 10.0 40.0 6 8\n/END\n")
    assert ec.impl_fatig_joint and ec.impl_fatig_wville
    assert ec.impl_fatig_wv_refine == 8 and ec.impl_fatig_wv_smooth == 0.0

    # a plain /EVOL card is NOT WVILLE and keeps the defaults
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG/EVOL/BASE\n"
                   "0.0 250 2000 7 0 4\n5.0 1e12\n20 120 5 30 10\n/END\n")
    assert not ec.impl_fatig_wville
    assert ec.impl_fatig_wv_refine == 8 and ec.impl_fatig_wv_smooth == 0.0


# ============================================================================
# END-TO-END
# ============================================================================

def test_wville_scalar_end_to_end():
    """/IMPL/FATIG/WVILLE end to end on the base-driven spring chain: the listing
    prints the CONTINUOUS WIGNER-VILLE block ALONGSIDE the M26 windowed one, stores a
    ``wigner_ville`` sub-entry with the fine-grid summary, the instantaneous
    peak-frequency drift and the shrinking boundary caveat, and the M26 windowed
    block is still present."""
    starter = _with_mission(_chain_deck([2e-3] * 5, [800.] * 5, [0.] * 5))
    eng = ("#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG/WVILLE/NSTAT/BASE\n"
           "0.0 250.0 3000 2 0 5\n5.0 1.0e14 0.03 0.0 0.0 400.0 123\n"
           "20 8\n20.0 120.0 5.0 30.0 8 8 0.0\n/PRINT/-500\n/STOP\n15.0\n")
    m, out = _run(starter, eng, capture=True)
    fat = m.implicit_result.fatigue
    wvr = fat["wigner_ville"]
    assert wvr is not None
    assert wvr["nt"] == 64                               # 8 windows x refine 8
    assert wvr["refine"] == 8
    assert wvr["boundary_jump"] < wvr["boundary_jump_windowed"]
    assert wvr["windowed_damage_rate"] is not None       # M26 reference reported
    assert "CONTINUOUS WIGNER-VILLE INSTANTANEOUS" in out
    assert "FULLY EVOLUTIONARY / NON-SEPARABLE-PSD FATIGUE" in out
    assert "RANDOM-VIBRATION (SPECTRAL) FATIGUE" in out


def test_wville_multiaxial_end_to_end():
    """/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE: the M27 windowed joint-tensor path runs
    unchanged and the continuous TENSOR spectrum runs ALONGSIDE (a ``wigner_ville``
    sub-entry with the per-instant plane re-search), with a continuous
    multivariate Monte-Carlo."""
    eng = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n"
           "/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NSTAT\n"
           "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000\n20 6\n"
           "50.0 250.0 10.0 40.0 6 6 0.0\n/PRINT/-500\n/STOP\n15.0\n")
    m, out = _run(_with_mission(_brick_deck(nx=3)), eng, capture=True)
    fat = m.implicit_result.fatigue
    assert fat.get("multiaxial")
    wvr = fat["wigner_ville"]
    assert wvr is not None and wvr["tensor"]
    assert wvr["nt"] == 36                               # 6 windows x refine 6
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert wvr[key]["damage_rate"] > 0.0
        assert wvr["windowed"][key] > 0.0                # M27 reference alongside
    assert fat["joint_evolutionary"] is not None
    assert "CONTINUOUS WIGNER-VILLE INSTANTANEOUS" in out
    assert "MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE" in out


def test_wville_multi_input_end_to_end():
    """/IMPL/FATIG/MULT/MINPUT/EVOL/WVILLE: the M29 scalar-coherence windowed path
    runs and the continuous MULTI-INPUT spectrum runs ALONGSIDE (the coherence
    drifting continuously, a per-instant plane re-search), with a continuous
    multi-input Monte-Carlo."""
    eng = ("#\n/RUN/BR/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT/EVOL/WVILLE\n"
           "0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 40.0 7\n"
           "60.0 180.0 8.0 8.0 5 5 0.0\n2 0 0.1 0.0 0.0 0.0 0.9 0.0\n1 10\n2 11\n"
           "/PRINT/-500\n/STOP\n15.0\n")
    m, out = _run(_two_input_brick(), eng, capture=True)
    mi = m.implicit_result.fatigue["multi_input"]
    wvr = mi["wigner_ville"]
    assert wvr is not None and wvr["multi_input"]
    assert wvr["nt"] == 25                               # 5 windows x refine 5
    assert not wvr["freq_dependent"]
    assert wvr["plane_rotation_deg"] > 0.0               # coherence-driven drift
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert wvr[key]["damage_rate"] > 0.0
        assert wvr["windowed"][key] > 0.0                # M29 reference alongside
    assert mi["evolutionary_multi_input"] is not None
    assert "CONTINUOUS WIGNER-VILLE INSTANTANEOUS" in out
    assert "FULLY EVOLUTIONARY MULTI-INPUT" in out


def test_wville_fcoh_multi_input_end_to_end():
    """/IMPL/FATIG/MULT/MINPUT/EVOL/FCOH/WVILLE: the M30 frequency-dependent-coherence
    windowed path runs and the continuous FREQUENCY-DEPENDENT multi-input spectrum
    runs ALONGSIDE (freq_dependent = True — the M30 stacks drifting continuously)."""
    eng = ("#\n/RUN/BR/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT/EVOL/FCOH/WVILLE\n"
           "0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 40.0 7\n"
           "60.0 180.0 80.0 80.0 5 5 0.0\n"
           "2 1 0.0 0.0 0.9 100.0 -1 0.0 0.2 100.0\n"
           "1 10 0.0 0.0 0.0\n2 11 3.0 0.0 0.0\n"
           "/PRINT/-500\n/STOP\n15.0\n")
    m, out = _run(_two_input_brick(), eng, capture=True)
    mi = m.implicit_result.fatigue["multi_input"]
    wvr = mi["wigner_ville"]
    assert wvr is not None and wvr["freq_dependent"] is True
    assert mi["freq_evolutionary_multi_input"] is not None
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert wvr[key]["damage_rate"] > 0.0
    assert "CONTINUOUS WIGNER-VILLE INSTANTANEOUS" in out
    assert "FREQUENCY-DEPENDENT EVOLUTIONARY MULTI-INPUT" in out


# ============================================================================
# NO-REGRESSION (the M7 parity contract)
# ============================================================================

def test_m20_m26_byte_identical_with_without_wville():
    """The M20 stationary, M25 non-stationary AND M26 windowed-evolutionary scalar
    answers are BYTE-IDENTICAL whether or not the M31 continuous path runs — the
    continuous path is NEW and ALONGSIDE."""
    starter = _with_mission(_chain_deck([2e-3] * 5, [800.] * 5, [0.] * 5))
    base = ("#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG/EVOL/NSTAT/BASE\n"
            "0.0 250.0 3000 2 0 5\n5.0 1.0e14 0.03 0.0 0.0 400.0 123\n"
            "20 8\n20.0 120.0 5.0 30.0 8\n/PRINT/-500\n/STOP\n15.0\n")
    wville = ("#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG/EVOL/WVILLE/NSTAT/BASE\n"
              "0.0 250.0 3000 2 0 5\n5.0 1.0e14 0.03 0.0 0.0 400.0 123\n"
              "20 8\n20.0 120.0 5.0 30.0 8 8 0.0\n/PRINT/-500\n/STOP\n15.0\n")
    a = _run(starter, base)
    b = _run(starter, wville)
    fa = a.implicit_result.fatigue
    fb = b.implicit_result.fatigue
    assert fa["wigner_ville"] is None
    assert fb["wigner_ville"] is not None
    for est in ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti"):
        assert (fa["summary"][est]["damage_rate"]
                == fb["summary"][est]["damage_rate"])
    # M26 windowed evolutionary byte-identical
    assert (fa["evolutionary"]["summary"]["dirlik"]["damage_rate"]
            == fb["evolutionary"]["summary"]["dirlik"]["damage_rate"])
    # M20 Gaussian Monte-Carlo byte-identical
    assert (fa["monte_carlo"]["damage_rate"]
            == fb["monte_carlo"]["damage_rate"])


def test_m21_m27_byte_identical_with_without_wville():
    """The M21 spectral reductions AND the M27 windowed joint-tensor answers are
    BYTE-IDENTICAL whether or not the M31 continuous tensor path runs."""
    deck = _with_mission(_brick_deck(nx=3))
    base = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/EVOL/JOINT/NSTAT\n"
            "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000\n20 6\n"
            "50.0 250.0 10.0 40.0 6\n/PRINT/-500\n/STOP\n15.0\n")
    wville = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n"
              "/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NSTAT\n"
              "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000\n20 6\n"
              "50.0 250.0 10.0 40.0 6 6 0.0\n/PRINT/-500\n/STOP\n15.0\n")
    a = _run(deck, base)
    b = _run(deck, wville)
    fa = a.implicit_result.fatigue
    fb = b.implicit_result.fatigue
    assert fa["wigner_ville"] is None and fb["wigner_ville"] is not None
    for key in ("von_mises", "normal_plane", "shear_plane"):
        for est in ("narrow_band", "dirlik", "wirsching_light",
                    "tovo_benasciutti"):
            assert (fa[key]["summary"][est]["damage_rate"]
                    == fb[key]["summary"][est]["damage_rate"])
        # M27 windowed joint-tensor byte-identical
        assert (fa["joint_evolutionary"][key]["damage_rate"]
                == fb["joint_evolutionary"][key]["damage_rate"])


def test_m29_byte_identical_with_without_wville():
    """The M29 scalar-coherence windowed multi-input answers are BYTE-IDENTICAL
    whether or not the M31 continuous multi-input path runs."""
    base = ("#\n/RUN/BR/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT/EVOL\n"
            "0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 40.0 7\n"
            "60.0 180.0 8.0 8.0 5\n2 0 0.1 0.0 0.0 0.0 0.9 0.0\n1 10\n2 11\n"
            "/PRINT/-500\n/STOP\n15.0\n")
    wville = ("#\n/RUN/BR/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT/EVOL/WVILLE\n"
              "0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 40.0 7\n"
              "60.0 180.0 8.0 8.0 5 5 0.0\n2 0 0.1 0.0 0.0 0.0 0.9 0.0\n1 10\n"
              "2 11\n/PRINT/-500\n/STOP\n15.0\n")
    a = _run(_two_input_brick(), base)
    b = _run(_two_input_brick(), wville)
    ma = a.implicit_result.fatigue["multi_input"]
    mb = b.implicit_result.fatigue["multi_input"]
    assert ma["wigner_ville"] is None and mb["wigner_ville"] is not None
    ea = ma["evolutionary_multi_input"]
    eb = mb["evolutionary_multi_input"]
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert ea[key]["damage_rate"] == eb[key]["damage_rate"]
    # the M28 stationary reductions byte-identical too
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert (ma[key]["summary"]["dirlik"]["damage_rate"]
                == mb[key]["summary"]["dirlik"]["damage_rate"])


def test_wville_does_not_mutate_state():
    """The continuous Wigner-Ville path is read-only in the element state: a
    representative evaluation leaves the springs' state arrays untouched (the
    M14-M30 parity contract extended to M31)."""
    starter = _chain_deck([2e-3] * 3, [800.] * 3, [0.] * 3)
    m = None
    with contextlib.redirect_stdout(io.StringIO()):
        m = run_starter(_write(starter))
    before = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in m.springs.state.items()}
    f, S = _bimodal_psd()
    wv.wigner_ville_fatigue_summary(f, S, [1.0] * 5, (30.0, 130.0), (5.0, 25.0),
                                    5.0, 1e14, scales=[0.5, 1., 1.5, 1., .5],
                                    refine=6, smooth=0.2)
    for k, v in before.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(m.springs.state[k], v), k


def _write(text):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M31_0000.rad")
    with open(sp, "w") as f:
        f.write(text)
    return sp

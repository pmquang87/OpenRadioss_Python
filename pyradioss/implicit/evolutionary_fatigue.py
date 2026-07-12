"""
Fully evolutionary / non-separable-PSD spectral fatigue — M26: the frequency-
domain fatigue-damage estimate of a random-vibration response whose spectral
SHAPE (not merely its RMS level) VARIES WITH TIME, computed by extending the M25
piecewise-stationary / amplitude-modulated estimators to a genuinely
NON-SEPARABLE evolutionary spectrum S(omega, t) — a time-frequency "spectrogram"
whose bandwidth, modal content and centre frequency DRIFT with time — and
cross-validated against a non-stationary time-domain Monte-Carlo with a
time-varying filter.

WHERE M26 SITS RELATIVE TO M25. M25 lifted stationarity only in the RMS ENVELOPE:
Priestley's SEPARABLE evolutionary spectrum S(omega, t) = |A(t)|^2 S(omega), a
FIXED spectral SHAPE whose amplitude A(t) is modulated. In the separable case the
shape coefficients (the rates nu_0 / nu_p, the width factors alpha_1 / alpha_2,
the Dirlik D1..D3 / Q / R) are TIME-INVARIANT — every time-window is the SAME
shape at a different level, and a single reference moment set m_0..m_4 (scaled by
a_i^2 per block) suffices. M26 lifts exactly that SEPARABILITY assumption: the
spectral SHAPE ITSELF evolves, so each time-window carries its OWN complete moment
set m_0..m_4 (a different bandwidth, different rates, different centre frequency),
NOT just its own scaling of a shared shape.

THE M26 <-> M25 REDUCTION (built in, exact). M26's non-separable spectrogram is
the GENERAL per-window-full-PSD block; M25's RMS-scaled block is the shared-shape
SPECIAL CASE. A non-separable spectrogram whose per-window SHAPE is CONSTANT (only
the level a_i drifts) must recover the M25 amplitude-modulated answer EXACTLY, and
a single window / a time-invariant shape must recover the M20 stationary answer
EXACTLY. Both reductions are structural (the window Miner-sum over shared-shape
windows IS the M25 block Miner-sum, which M25 already proved equals the
amplitude-modulated E[a^m] damage), and both are asserted in the tests. M26
therefore CONSUMES the M25 machinery read-only: ``block_fatigue_summary`` already
accepts per-block DISTINCT moment sets (the ``{"moments": [...], "duration": T}``
block form), so the window Miner-sum is driven straight through it — M26 supplies
the per-window REAL PSDs of a drifting shape rather than a shared shape scaled.

Fortran origin
--------------
There is NONE. ``engine/source/input/freimpl.F`` (the /IMPL reader, re-read line
by line for M16-M25 AND AGAIN for M26, fetched from raw.githubusercontent.com)
parses only /IMPL/DYNA (the DIRECT Newmark/HHT integrator), /IMPL/BUCKL,
/IMPL/DT, /IMPL/NONLIN and /IMPL/ARCL plus the linear-solver housekeeping — there
is no /FATIG, no S-N / Miner branch, no Dirlik / rainflow / narrow-band
estimator, no non-Gaussian / kurtosis machinery, and (as M16-M25 already found)
NOTHING non-stationary / evolutionary / spectrogram / time-frequency of any kind.
The sole ``PSD`` token in the whole file is still ``IMUMPSD`` (line 269), a
MUMPS-solver flag, not a power spectral density. OpenRadioss is a time-domain
crash/impact code: the random-vibration fatigue analysis — stationary Gaussian
(M20-M23), stationary non-Gaussian (M24), separable non-stationary (M25) OR fully
evolutionary / non-separable (M26) — is simply not part of the open-source
solver, exactly as M16 found for the real eigensolver, M17/M18 for the transfer
functions, M19 for the PSD machinery and M20-M25 for the spectral-fatigue
estimators this module extends.

So M26 does exactly what M16-M25 did: it ports evolutionary / non-separable-PSD
fatigue as a clean LIBRARY capability and drives it with a minimal PORT engine
sub-flag (/IMPL/FATIG/EVOL — the evolutionary analogue of the M20 /IMPL/FATIG
Gaussian card, the M24 /IMPL/FATIG/NGAUSS card and the M25 /IMPL/FATIG/NSTAT
card). Nothing in the M10 direct integrator, the M16 REAL eigensolver, the
M17/M18 superposition, the M19 PSD path, the M20 SCALAR fatigue, the M21
MULTIAXIAL SPECTRAL path, the M22 NON-PROPORTIONAL TIME-DOMAIN path, the M23
SPECTRAL NON-PROPORTIONAL path, the M24 NON-GAUSSIAN correction OR the M25
NON-STATIONARY (separable) correction is touched: the evolutionary path is a NEW,
parallel path that CONSUMES the M20 stationary estimators and the M25 block /
amplitude-modulated machinery (read-only) and produces its answer ALONGSIDE the
stationary (M20) and RMS-non-stationary (M25) numbers so a listing shows the
stationary, the RMS-non-stationary and the shape-evolutionary answers side by
side.

Theory — evolutionary / non-separable spectra + the spectrogram / short-time view
---------------------------------------------------------------------------------
(Priestley, "Evolutionary spectra and non-stationary processes", J. Roy. Statist.
Soc. B 27, 1965, AND "Power spectral analysis of non-stationary random
processes", J. Sound Vib. 6, 1967 — the evolutionary spectrum S(omega, t), the
GENERAL non-separable case where the oscillatory amplitude A(omega, t) depends on
BOTH omega and t (M25 used the separable A(t)A(omega) subcase); Mark, "Spectral
analysis of the convolution and filtering of non-stationary stochastic
processes", J. Sound Vib. 11, 1970, and Hammond — non-stationary spectral
analysis and the instantaneous spectrum; Newland, "An Introduction to Random
Vibrations, Spectral & Wavelet Analysis" ch. 5-7, 10 — the short-time / windowed
spectral method (the spectrogram) and the time-frequency view; Bendat & Piersol,
"Random Data" ch. 12 — non-stationary random data; the Wigner-Ville / Loeve
time-frequency distribution as the continuous instantaneous-spectrum limit of the
windowed spectrogram; Palmgren-Miner — the linear window-damage summation; the
M25 piecewise-stationary basis — Wolfsteiner & Trapp, Braccesi-Cianetti-Lori-
Pioli, Kihm-Ferguson-Antoni, Rychlik's switching process.)

THE SPECTROGRAM / PER-WINDOW-FULL-PSD MODEL. The general engineering treatment of
a non-separable evolutionary load is the SPECTROGRAM: partition the record into a
sequence of short time-WINDOWS, and within each window i treat the process as
locally stationary with its OWN COMPLETE stress PSD S_i(omega) (a genuinely
different SHAPE per window — different bandwidth, rates, centre frequency), hence
its own full moment array m_{n,i} = (1/pi) int Omega^n S_i(Omega) dOmega. Within
each window the STATIONARY M20 estimator gives a damage RATE (E[D]/T)_i from
m_{0,i}..m_{4,i}; the window contributes D_i = (E[D]/T)_i T_i and the windows
ACCUMULATE by the Palmgren-Miner linear rule

    D = sum_i D_i = sum_i (E[D]/T)_i T_i ,   E[D]/T = D / sum_i T_i          (1)

with the mission time-to-failure T_f = (sum_i T_i) / D (failure at D = 1). This is
IDENTICAL in FORM to the M25 block Miner-sum (2) — the difference is that M25's
per-block moments are a_i^2 times ONE shared m_n (a scaling), whereas M26's
per-window moments are the moments of GENUINELY DIFFERENT PSDs. Because the M25
``block_fatigue_summary`` already supports per-block distinct moments, M26 drives
it read-only with the real per-window PSDs — the general non-separable extension
of the M25 model, of which M25's RMS-scaled block is the shared-shape reduction.

THE TIME-FREQUENCY / DRIFTING-SHAPE MODEL. A smooth evolutionary spectrum is built
from a spectral parameter that DRIFTS with time — e.g. a Gaussian spectral bump
whose centre frequency f_c(t) SWEEPS (a "chirp-like" random process — the
random-vibration analogue of a swept sine) and/or whose bandwidth b(t) BROADENS
(narrow-band -> wide-band transition):

    S(omega, t) = A(t)^2 exp( -(f - f_c(t))^2 / (2 b(t)^2) ) ,  omega = 2 pi f
                                                                             (2)

Sampling f_c(t), b(t), A(t) at the window mid-times gives the per-window PSDs of
the spectrogram; each window's full moments are recomputed FROM the drifting shape
(NOT scaled from a fixed one), and the damage is the window Miner-sum (1). When
f_c and b are CONSTANT (only A drifts) this collapses to the M25 separable
|A(t)|^2 S(omega) case and the answer reduces to the M25 amplitude-modulated
damage EXACTLY (the built-in M26 <-> M25 reduction).

APPLYING A DRIFTING SHAPE TO A RECOVERED STRESS PSD (the driver's route). The
structure's stress response PSD is S_sigmasigma(f) = |H_sigma(f)|^2 S_aa(f). A
resonance sweep — the excitation energy moving through the modal band with time —
is a time-varying INPUT window W_i(f) on S_aa; but because W_i is a scalar
multiplier at each frequency, W_i(f) S_aa(f) propagates through |H_sigma|^2 to
give W_i(f) S_sigmasigma(f) EXACTLY (the window commutes with the linear |H|^2
map). So the driver forms the M20 stationary stress PSD ONCE and applies the
swept/broadening Gaussian window W_i(f) = exp(-(f - f_c,i)^2 / (2 b_i^2)) per
window to obtain the per-window stress PSDs, then Miner-sums (1). A per-window RMS
level a_i (the M25 modulation, if /NSTAT composes) multiplies on top.

NON-STATIONARY (NON-SEPARABLE) MONTE-CARLO with a TIME-VARYING FILTER. As an
independent time-domain validation the module SYNTHESISES a non-separable history
— per-window spectral-representation blocks (the M20 ``synthesize_gaussian_
history``: an inverse-rFFT of the WINDOW's OWN PSD with seeded phases) CONCATENATED
in time, so the instantaneous spectrum tracks S(omega, t) window by window (the
short-time-stationary approximation — a slowly time-varying filter applied to the
carrier). It is RAINFLOW-counted (the M20 ASTM E1049 counter) and Miner-summed;
its short-time spectrogram (per-window RMS + zero-crossing rate) reproduces the
target evolutionary spectrum, and its damage matches the window spectral estimate
to within the seeded scatter. In the CONSTANT-SHAPE limit the per-window PSDs
collapse to one shared shape times an RMS envelope, and the non-separable
synthesiser DELEGATES to the M25 ``synthesize_nonstationary_history`` (a single
carrier times the envelope) — reducing BIT-IDENTICALLY to the M25 non-stationary
Monte-Carlo (and hence, for a constant envelope, to the M20 Gaussian Monte-Carlo).

    WINDOW-BOUNDARY RAINFLOW CAVEAT (documented, carried from M25). Rainflow
    counting over the concatenated non-separable record naturally handles cycles
    that STRADDLE a window boundary (a peak in one shape paired with a valley in
    the next). The closed-form window Miner-sum (1) counts each window's cycles
    INDEPENDENTLY and so MISSES those boundary cycles — a small, documented
    discrepancy that vanishes as the windows grow long relative to the cycle
    period. The Monte-Carlo is the reference that includes them.

    INSTANTANEOUS vs WINDOWED SPECTRUM (documented). The spectrogram is a WINDOWED
    (short-time) estimate of the evolutionary spectrum; the continuous
    instantaneous-spectrum (Wigner-Ville / Priestley's oscillatory A(omega, t))
    formulation is the fine-window limit. M26 uses the windowed spectrogram (the
    engineering standard), NOT the continuous Wigner-Ville distribution, which is
    DEFERRED (a different, cross-term-laden analysis).

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* LIBRARY-FIRST sub-flag (/IMPL/FATIG/EVOL) — no upstream equivalent, exactly as
  established for M16-M25's PORT cards.
* WINDOWED SPECTROGRAM, not the continuous Wigner-Ville / Loeve instantaneous
  spectrum — the short-time-stationary spectrogram is the engineering standard;
  the continuous time-frequency distribution is DEFERRED.
* SHORT-TIME-STATIONARY approximation: each window is treated as locally
  stationary (the M20 estimators assume stationarity within the window). The
  window must be LONG relative to the carrier period and SHORT relative to the
  shape drift — the standard spectrogram trade-off, documented.
* A full non-stationary MULTIAXIAL JOINT treatment of the evolutionary stress
  TENSOR (a jointly time-varying tensor cross-PSD) is DEFERRED; M26 applies the
  drifting-shape window to the equivalent-scalar (von Mises / critical-plane) PSD
  the M21/M23 reductions already produce, so it COMPOSES with the multiaxial paths
  (each reduction's per-window moments are recomputed) but does not model a
  jointly evolutionary tensor.
* MEAN-STRESS beyond the basic M20/M21 Goodman intercept, CRACK-GROWTH /
  fracture-mechanics fatigue, the COMPLEX-FRF stress recovery, a MULTI-INPUT
  cross-PSD with coherence and the full non-Gaussian MULTIAXIAL joint
  distribution remain DEFERRED (the unchanged M20-M25 tail).
"""

from __future__ import annotations

import math

import numpy as np


# ============================================================================
# Spectral-window / drifting-shape construction (build item 1b support)
# ============================================================================

def spectral_window(freqs, fc, bw):
    """A Gaussian bandpass spectral WINDOW W(f) = exp(-(f - fc)^2 / (2 bw^2)) on
    the frequency grid ``freqs`` [Hz], centred at ``fc`` with width (std) ``bw``.

    Multiplying a base stress PSD by W(f) selects / emphasises a frequency band —
    the mechanism by which the evolutionary spectrogram makes the SHAPE (which
    modes carry energy) drift with time as ``fc`` sweeps and ``bw`` broadens
    (theory eq. (2)). Because the window is a per-frequency SCALAR multiplier it
    commutes with the linear |H_sigma|^2 map, so windowing the stress PSD is
    identical to windowing the input PSD (module docstring).

    A non-positive ``bw`` means NO shape drift — the window is all-ones (a flat
    pass-through), so a spectrogram built with bw <= 0 reduces to the M25 shared
    shape. Returns the window array (same length as ``freqs``)."""
    f = np.asarray(freqs, dtype=float)
    if bw is None or bw <= 0.0:
        return np.ones_like(f)
    return np.exp(-((f - float(fc)) ** 2) / (2.0 * float(bw) ** 2))


def _moments_of_psd(freqs, psd):
    """Spectral moments m_0..m_4 (M19 convention m_n = (1/pi) int Omega^n S dOmega)
    of a stress PSD sampled on ``freqs`` [Hz]. Reuses the M19 quadrature so the
    per-window moments are formed EXACTLY as the M20 stationary path forms them."""
    from .random_response import spectral_moments
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    return spectral_moments(omega, np.clip(np.asarray(psd, dtype=float),
                                           0.0, None), nmax=4)


def gaussian_shape_psd(freqs, fc, bw, level=1.0):
    """An analytic Gaussian narrow-band stress PSD bump S(f) = level^2 exp(-(f -
    fc)^2 / (2 bw^2)) on ``freqs`` [Hz] (theory eq. (2)) — a self-contained
    evolutionary-spectrum building block (a "spectral line" of centre frequency
    ``fc``, bandwidth ``bw`` and RMS scaling ``level``). Used by
    ``gaussian_evolutionary_spectrogram`` to build a drifting-shape spectrogram
    from first principles (independent of any recovered structural PSD)."""
    f = np.asarray(freqs, dtype=float)
    bw = max(float(bw), 1e-12)
    return float(level) ** 2 * np.exp(-((f - float(fc)) ** 2) / (2.0 * bw ** 2))


def drifting_shape_spectrogram(freqs, base_psd, durations, fc, bw, scales=None):
    """Build a NON-SEPARABLE spectrogram by applying a DRIFTING Gaussian window to
    a base stress PSD (theory eq. (2) / the driver's route): partition the mission
    into ``len(durations)`` windows and, per window i, apply the swept-centre /
    broadening window W_i(f) = spectral_window(freqs, fc_i, bw_i) to ``base_psd``
    and scale by the per-window RMS level a_i.

    ``fc`` = (fc0, fc1) — the window centre frequency at the FIRST and LAST window
    mid-time (sweeps linearly; a "chirp"); ``bw`` = (bw0, bw1) — the window
    bandwidth start/end (broadens/narrows). A scalar ``fc`` / ``bw`` means it is
    held constant; ``bw`` <= 0 means NO shape drift (a flat window -> the M25
    shared shape). ``scales`` — the per-window RMS level a_i (the M25 modulation,
    default all-ones); the per-window PSD is a_i^2 W_i(f) base_psd.

    Returns the ``windows`` list the estimators / Monte-Carlo consume: each window
    dict carries ``freqs``, ``psd`` (a_i^2 W_i base_psd), ``duration``,
    ``moments`` (m_0..m_4 of ``psd``), ``fc``, ``bw``, ``scale`` (a_i) and
    ``shape_psd`` (W_i base_psd — the UNIT-scale windowed shape, used for the
    constant-shape detection / M25 delegation in the Monte-Carlo)."""
    f = np.asarray(freqs, dtype=float)
    S0 = np.clip(np.asarray(base_psd, dtype=float), 0.0, None)
    T = np.asarray(durations, dtype=float).ravel()
    nwin = T.size
    if nwin == 0:
        raise ValueError("drifting_shape_spectrogram needs at least one window.")
    if np.any(T <= 0.0):
        raise ValueError("every window duration must be positive.")
    if scales is None:
        a = np.ones(nwin)
    else:
        a = np.asarray(scales, dtype=float).ravel()
        if a.size != nwin:
            raise ValueError("scales and durations must match in length.")
    fc0, fc1 = (fc if isinstance(fc, (tuple, list, np.ndarray)) else (fc, fc))
    bw0, bw1 = (bw if isinstance(bw, (tuple, list, np.ndarray)) else (bw, bw))
    # window mid-time fractions s_i in [0, 1] (equal-duration parameterisation of
    # the drift; a single window sits at the midpoint s = 0.5)
    if nwin == 1:
        s = np.array([0.5])
    else:
        s = (np.arange(nwin) + 0.5) / nwin
    windows = []
    for i in range(nwin):
        fci = float(fc0) + s[i] * (float(fc1) - float(fc0))
        bwi = float(bw0) + s[i] * (float(bw1) - float(bw0))
        W = spectral_window(f, fci, bwi)
        shape = W * S0                              # unit-scale windowed shape
        psd = float(a[i]) ** 2 * shape
        windows.append({"freqs": f, "psd": psd, "duration": float(T[i]),
                        "moments": _moments_of_psd(f, psd), "fc": fci,
                        "bw": bwi, "scale": float(a[i]), "shape_psd": shape})
    return windows


def gaussian_evolutionary_spectrogram(freqs, fc, bw, level, durations):
    """A self-contained analytic DRIFTING-SHAPE spectrogram (theory eq. (2)),
    independent of any recovered structural PSD: per window i a Gaussian bump
    ``gaussian_shape_psd(freqs, fc_i, bw_i, level_i)`` with the centre frequency
    ``fc``, bandwidth ``bw`` and level ``level`` each a scalar (held constant), a
    2-tuple ``(start, end)`` (swept LINEARLY across the windows — a chirp for
    ``fc``, a narrow-band -> wide-band transition for ``bw``) OR a full per-window
    array. Returns the same ``windows`` list shape as ``drifting_shape_spectrogram``
    (``shape_psd`` = the unit-level bump for level i so the constant-shape detector
    still works when only the level array differs)."""
    f = np.asarray(freqs, dtype=float)
    T = np.asarray(durations, dtype=float).ravel()
    nwin = T.size
    if nwin == 0:
        raise ValueError("gaussian_evolutionary_spectrogram needs >= 1 window.")
    # window mid-time fractions (the same parameterisation as the drift builder)
    s = np.array([0.5]) if nwin == 1 else (np.arange(nwin) + 0.5) / nwin

    def _per_window(v):
        arr = np.atleast_1d(np.asarray(v, dtype=float))
        if arr.size == nwin:                       # explicit per-window values
            return arr
        if arr.size == 2:                          # (start, end) linear sweep
            return arr[0] + s * (arr[1] - arr[0])
        return np.full(nwin, float(arr.flat[0]))   # scalar -> constant

    fcs, bws, lvs = _per_window(fc), _per_window(bw), _per_window(level)
    windows = []
    for i in range(nwin):
        psd = gaussian_shape_psd(f, fcs[i], bws[i], lvs[i])
        shape = gaussian_shape_psd(f, fcs[i], bws[i], 1.0)   # unit-level shape
        windows.append({"freqs": f, "psd": psd, "duration": float(T[i]),
                        "moments": _moments_of_psd(f, psd), "fc": float(fcs[i]),
                        "bw": float(bws[i]), "scale": float(lvs[i]),
                        "shape_psd": shape})
    return windows


# ============================================================================
# The evolutionary (window Miner-sum) damage (build item 1a + 1b)
# ============================================================================

def evolutionary_fatigue_summary(windows, m, C, mean_stress=0.0, ultimate=0.0):
    """The FULLY EVOLUTIONARY / NON-SEPARABLE window Miner-sum damage (theory eq.
    (1)): run the M20 estimators PER WINDOW on the window's OWN full moment set and
    Palmgren-Miner SUM the window damages duration-weighted. This is the general
    non-separable extension of the M25 block model — driven straight through the
    M25 ``block_fatigue_summary`` (which already accepts per-block DISTINCT
    moments) with the real per-window PSDs of a drifting shape.

    ``windows`` — the spectrogram: a list of per-window dicts, each with
    ``moments`` [m0..m4] (a genuinely different SHAPE per window) and a
    ``duration`` T_i (plus, for reporting, the optional ``fc`` / ``bw`` / ``scale``
    the drifting-shape builders attach). ``m`` / ``C`` the S-N law.

    Returns a dict with, for EACH of the four M20 estimators (narrow_band / dirlik
    / wirsching_light / tovo_benasciutti), the window Miner-sum
    {damage_rate, life, damage, total_time, s_eq, nu}; plus:
      * ``windows`` — the per-window SHAPE breakdown [{fc, bw, scale, duration,
        sigma, nu0, nup, alpha2, damage_rate, damage}] (the spectrogram RMS /
        bandwidth / rate drift, the point of a NON-separable spectrum);
      * ``damage_rate`` / ``life`` / ``total_time`` — the Dirlik (wide-band
        standard) window Miner-sum, promoted to the top level;
      * ``constant_shape`` — whether every window shares ONE spectral shape (so the
        answer reduces to the M25 amplitude-modulated / M20 stationary case);
      * ``nwin``."""
    from . import spectral_fatigue as sf
    from . import nonstationary_fatigue as nsf
    if not windows:
        raise ValueError("evolutionary_fatigue_summary needs >= 1 window.")
    # drive the M25 block Miner-sum per estimator with the per-window FULL moments
    # (the {"moments", "duration"} block form — read-only reuse of M25)
    blocks = [{"moments": np.asarray(w["moments"], dtype=float),
               "duration": float(w["duration"])} for w in windows]
    out = {"method": "evolutionary_spectrogram", "nwin": len(windows)}
    for est in ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti"):
        out[est] = nsf.block_fatigue_summary(
            blocks, m, C, mean_stress=mean_stress, ultimate=ultimate,
            estimator=est)
    # the per-window SHAPE breakdown (the spectrogram descriptors + Dirlik damage)
    Ceff = sf._goodman_C(C, m, mean_stress, ultimate)
    wout = []
    shapes = []
    for w in windows:
        mom = np.asarray(w["moments"], dtype=float)
        p = sf.spectral_bandwidth_params(mom)
        r = sf.dirlik_damage(mom, m, C, mean_stress, ultimate)
        Ti = float(w["duration"])
        wout.append({"fc": w.get("fc"), "bw": w.get("bw"),
                     "scale": w.get("scale"), "duration": Ti,
                     "sigma": p["sigma"], "nu0": p["nu0"], "nup": p["nup"],
                     "alpha2": p["alpha2"], "damage_rate": float(r["damage_rate"]),
                     "damage": float(r["damage_rate"]) * Ti})
        # the NORMALISED shape (unit-m0) — for the constant-shape detection
        s0 = p["sigma"]
        shapes.append(mom / (s0 ** 2) if s0 > 0 else mom)
    # constant shape: every window's normalised moment set is the same (the
    # spectral shape is invariant, only the level drifts -> the M25 special case)
    const = len(shapes) <= 1 or all(
        np.allclose(shapes[k], shapes[0], rtol=1e-9, atol=1e-12)
        for k in range(1, len(shapes)))
    dk = out["dirlik"]
    out.update({"windows": wout, "constant_shape": bool(const),
                "damage_rate": dk["damage_rate"], "life": dk["life"],
                "total_time": dk["total_time"], "s_eq": dk["s_eq"],
                "nu": dk["nu"]})
    return out


# ============================================================================
# Non-separable Monte-Carlo cross-check — time-varying filter (build item 2)
# ============================================================================

def _windows_share_shape(windows):
    """True when every window shares ONE spectral SHAPE — the same ``freqs`` and
    the same UNIT-scale ``shape_psd`` (so window i's PSD is scale_i^2 shape) — the
    SEPARABLE / constant-shape case in which the non-separable synthesiser
    delegates to the M25 single-carrier-times-envelope path (bit-identical)."""
    if len(windows) <= 1:
        return "shape_psd" in windows[0]
    if any("shape_psd" not in w for w in windows):
        return False
    f0 = np.asarray(windows[0]["freqs"], dtype=float)
    s0 = np.asarray(windows[0]["shape_psd"], dtype=float)
    for w in windows[1:]:
        f = np.asarray(w["freqs"], dtype=float)
        s = np.asarray(w["shape_psd"], dtype=float)
        if f.shape != f0.shape or not np.array_equal(f, f0):
            return False
        if s.shape != s0.shape or not np.array_equal(s, s0):
            return False
    return True


def synthesize_evolutionary_history(windows, seed, fs=None):
    """Synthesise a NON-SEPARABLE (evolutionary) time history whose instantaneous
    spectrum tracks the spectrogram S(omega, t) (theory "NON-SEPARABLE MONTE-CARLO
    with a TIME-VARYING FILTER"):

      * SEPARABLE / CONSTANT-SHAPE case — if every window shares ONE spectral shape
        (scale_i^2 times a common ``shape_psd``), DELEGATE to the M25
        ``synthesize_nonstationary_history`` (a single carrier of the shared shape
        times the piecewise-constant RMS envelope a(t)) — BIT-IDENTICAL to the M25
        non-stationary Monte-Carlo (and, for a constant envelope, to the M20
        Gaussian Monte-Carlo). This is the built-in M26 <-> M25 reduction.
      * GENUINELY NON-SEPARABLE case — synthesise per-window spectral-representation
        BLOCKS (the M20 ``synthesize_gaussian_history`` of the WINDOW's OWN PSD,
        seeded deterministically by ``seed`` + i) and CONCATENATE them in time, so
        the instantaneous spectrum tracks the drifting shape window by window (the
        short-time-stationary approximation — a slowly time-varying filter on the
        carrier). All windows share ONE sampling rate ``fs`` so the concatenated
        record has a uniform time step.

    Returns (t, x, info) with ``info`` = {edges (window-boundary times), fs,
    delegated (bool)}. The window-boundary rainflow caveat (module docstring): the
    concatenated record's rainflow DOES count cycles straddling a window boundary;
    the closed-form window Miner-sum does NOT."""
    from . import spectral_fatigue as sf
    from . import nonstationary_fatigue as nsf
    if not windows:
        raise ValueError("synthesize_evolutionary_history needs >= 1 window.")
    T = np.array([float(w["duration"]) for w in windows], dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(T)])
    # --- separable / constant-shape: delegate to M25 (bit-identical) ------------
    if _windows_share_shape(windows):
        f = np.asarray(windows[0]["freqs"], dtype=float)
        S_base = np.asarray(windows[0]["shape_psd"], dtype=float)
        scales = np.array([float(w.get("scale", 1.0)) for w in windows])
        t, x, _env = nsf.synthesize_nonstationary_history(
            f, S_base, scales, T, seed, fs=fs)
        return t, x, {"edges": edges, "fs": None, "delegated": True}
    # --- genuinely non-separable: per-window blocks, ONE shared fs --------------
    fmax = max(float(np.max(np.asarray(w["freqs"], dtype=float)))
               for w in windows)
    if fs is None:
        fs = 8.0 * fmax                    # comfortably above Nyquist (2 fmax)
    chunks = []
    for i, w in enumerate(windows):
        # each window: its OWN PSD -> its OWN Gaussian carrier over its duration,
        # with a deterministic per-window seed (seed + i); a shared fs keeps the
        # concatenated time step uniform
        _t, xi = sf.synthesize_gaussian_history(
            np.asarray(w["freqs"], dtype=float),
            np.asarray(w["psd"], dtype=float),
            float(w["duration"]), int(seed) + i, fs=fs)
        chunks.append(xi)
    x = np.concatenate(chunks) if chunks else np.zeros(0)
    t = np.arange(x.size) / fs
    return t, x, {"edges": edges, "fs": fs, "delegated": False}


def evolutionary_monte_carlo_damage(windows, m, C, seed, fs=None,
                                    mean_stress=0.0, ultimate=0.0):
    """The TIME-DOMAIN NON-SEPARABLE (evolutionary) damage rate by Monte-Carlo
    (theory "NON-SEPARABLE MONTE-CARLO"): synthesise the non-separable history
    (``synthesize_evolutionary_history`` — per-window spectral-representation blocks
    concatenated, or the M25 delegation in the constant-shape limit), rainflow-count
    it (ASTM E1049, the M20 counter) and Miner-sum the ranges with N = C S^-m — the
    independent time-domain answer the window spectral estimate approximates.

    Returns the M20 Monte-Carlo dict shape plus a short-time SPECTROGRAM check: the
    per-window realised RMS ``window_rms`` and per-window mean zero-up-crossing rate
    ``window_nu0`` (Hz — which tracks the centre-frequency drift of the target
    spectrogram), and the overall sample ``kurtosis``. In the constant-shape limit
    reduces EXACTLY to the M25 non-stationary Monte-Carlo (module docstring)."""
    from . import spectral_fatigue as sf
    Ceff = sf._goodman_C(C, m, mean_stress, ultimate)
    t, x, info = synthesize_evolutionary_history(windows, seed, fs=fs)
    ranges, counts = sf.rainflow_count(x)
    D = float(np.sum(counts * ranges ** m) / Ceff) if ranges.size else 0.0
    T = t[-1] - t[0] if t.size > 1 else float(
        np.sum([w["duration"] for w in windows]))
    dr = D / T if T > 0 else 0.0
    tf, s_eq = sf.life_and_equivalent(dr, ranges.size / T if T > 0 else 0.0,
                                      m, Ceff)
    xc = x - np.mean(x)
    var = float(np.mean(xc * xc))
    kurt = float(np.mean(xc ** 4) / var ** 2) if var > 0 else 3.0
    # per-window short-time spectrogram: realised RMS + mean crossing rate (the
    # zero-up-crossing rate tracks the window centre frequency of S(omega, t))
    edges = info["edges"]
    window_rms = []
    window_nu0 = []
    for i in range(len(windows)):
        seg = x[(t >= edges[i]) & (t < edges[i + 1])]
        window_rms.append(float(np.std(seg)) if seg.size else 0.0)
        if seg.size > 2:
            sc = seg - np.mean(seg)
            # upward zero crossings / duration = the mean zero-up-crossing rate
            ups = np.sum((sc[:-1] < 0.0) & (sc[1:] >= 0.0))
            dur = float(edges[i + 1] - edges[i])
            window_nu0.append(ups / dur if dur > 0 else 0.0)
        else:
            window_nu0.append(0.0)
    return {"method": "evolutionary_monte_carlo", "damage_rate": dr, "life": tf,
            "s_eq": s_eq, "ncycles": float(counts.sum()), "ranges": ranges,
            "counts": counts, "duration": T, "rms": float(np.std(x)),
            "kurtosis": kurt, "window_rms": np.asarray(window_rms),
            "window_nu0": np.asarray(window_nu0),
            "delegated": bool(info.get("delegated"))}

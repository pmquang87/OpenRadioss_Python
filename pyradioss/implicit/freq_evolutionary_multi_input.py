"""
FREQUENCY-DEPENDENT + TIME-VARYING (EVOLUTIONARY) INPUT COHERENCE — M30: the
frequency-domain fatigue-damage estimate of a MULTI-INPUT random-vibration
response whose full INPUT cross-spectral matrix S_ff(omega, t) is driven by a
coherence matrix gamma_ab(f, t) that varies with BOTH FREQUENCY AND TIME — a
measured / modelled coherence SHAPE gamma_ab(f) that DRIFTS window to window
(and/or the M28 exponential/convection coherence field with a TIME-VARYING decay
coefficient / reference speed). Each mission window i carries its OWN FULL
(nf, ninput, ninput) Hermitian coherence stack gamma_ab(f, t_i); the per-window
multi-input stress-tensor cross-PSD S_sigmasigma(omega, t_i) = H_sigma S_ff(t_i)
H_sigma^H drives a per-window critical-plane search whose plane / F_np may DRIFT
as the coherence FREQUENCY-SHAPE evolves, reduced PER WINDOW by the M20-M27
estimator family + the M28/M29 multi-input paths and Palmgren-Miner-summed,
cross-validated against a NON-STATIONARY MULTI-INPUT multivariate Monte-Carlo.

WHERE M30 SITS RELATIVE TO M28 / M29. M30 is the CONVERGENCE of M28's
frequency-dependent coherence and M29's time-varying coherence:
* M28 (multi-input) drove S_sigmasigma from a FULL Hermitian ninput x ninput input
  cross-PSD S_ff whose coherence could be FREQUENCY-DEPENDENT (the
  ``exponential_coherence`` gamma_ab(f) = exp(-decay |x_a - x_b| f / speed)
  Davenport / von Karman convection field, a full (nf, n, n) stack) — but S_ff was
  STATIONARY: the coherence gamma_ab(f) held FIXED in time.
* M29 (evolutionary multi-input) let the input coherence DRIFT with time — but only
  as a per-window SCALAR gamma_ab(t_i) (a start -> end interpolation of a
  frequency-FLAT coherence, constant across frequency within each window). Its
  ``exponential_coherence`` drift was DEFERRED (M29 held the exponential model
  stationary — see its ``_run_evolutionary_multi_input`` docstring and the
  PORTING_GUIDE "Deferred out of M29" list).
M30 lifts exactly those two restrictions AT ONCE: the coherence is a FULL
frequency-dependent stack gamma_ab(f) that ALSO drifts window to window. S_ff(omega,
t_i) is a per-window schedule of Hermitian input cross-spectral matrices whose
OFF-DIAGONAL coherence is a per-pair frequency SHAPE gamma_ab(f, t_i) — a measured
/ modelled shape interpolated across the M26/M27 windows from a START shape
gamma_ab^0(f) to an END shape gamma_ab^1(f), OR the M28 exponential/convection
field with a decay coefficient / reference speed that DRIFTS window to window (a
turbulence field whose DECORRELATION FREQUENCY moves through the mission — the
"decorrelation moves up in frequency" / "convection speed ramps" case). Each
window's S_ff drives its OWN multi-input stress-tensor cross-PSD, whose 6x6 moment
matrices are reduced by the M27 per-window critical-plane search — the plane /
F_np RE-SEARCHED per window as the coherence FREQUENCY-SHAPE drifts.

THE BUILT-IN REDUCTIONS (exact, asserted).
* FREQUENCY-FLAT coherence (gamma_ab(f) constant across frequency, a scalar /
  (n, n) drifting-in-time coherence) -> the M29 scalar-coherence answer EXACTLY.
  A scalar / (n, n) ``gamma0`` / ``gamma1`` is detected as frequency-flat and the
  whole call DELEGATES to ``evolutionary_multi_input.evolutionary_multi_input_
  summary`` so it is BIT-IDENTICAL to M29 (which in turn recovers M28 / M27 in ITS
  special cases). M29 is EXACTLY the frequency-flat special case of M30.
* STATIONARY (single-window / constant-in-time) frequency-dependent coherence ->
  the M28 frequency-dependent answer EXACTLY. A single flat unit window with a
  fixed (nf, n, n) coherence stack gives S_ff,1 = the M28 stationary
  frequency-dependent S_ff and S_sigmasigma,1 = H_sigma S_ff H_sigma^H, reduced by
  the M28 ``multi_input_multiaxial_summary``. This case DELEGATES to M28 so it is
  BIT-IDENTICAL. M28 (with a frequency-dependent coherence) is EXACTLY the
  single-window special case of M30.
The whole POINT of M30 over M28/M29 is the case NEITHER covers: a coherence whose
FREQUENCY SHAPE itself drifts — a decorrelation frequency that MOVES UP as the
mission proceeds, or a convection speed that ramps — whose per-window response
variance AND critical plane genuinely DRIFT as the frequency-shape of the
coherence evolves, not merely its scalar level.

Fortran origin
--------------
There is NONE — re-confirmed for M30. ``engine/source/input/freimpl.F`` (the /IMPL
reader, re-read line by line for M16-M29 AND AGAIN for M30, fetched from
raw.githubusercontent.com) parses only /IMPL/DYNA, /IMPL/BUCKL, /IMPL/DT,
/IMPL/NONLIN and /IMPL/ARCL plus the linear-solver housekeeping — there is no
/FATIG, no S-N / Miner branch, no Dirlik / rainflow / narrow-band estimator, no
von-Mises / critical-plane / stress-tensor cross-PSD machinery, no multi-input /
coherence / cross-spectral matrix, and (as M16-M29 already found) NOTHING
non-stationary / evolutionary / spectrogram / frequency-dependent-time-varying-
coherence of any kind. The sole ``PSD`` token anywhere in the engine is
``IMUMPSD`` (a MUMPS-solver flag, not a power spectral density) — re-confirmed for
M30 in ``freimpl.F`` (line 269, ``IF (ISOLV==3) IMUMPSD=L_LIM``) and
``engine/source/implicit/imp_solv.F`` (line 1796). OpenRadioss is a time-domain
crash/impact code: the random-vibration MULTI-INPUT MULTIAXIAL fatigue analysis —
stationary multi-input (M28), evolutionary scalar-coherence multi-input (M29) OR
fully frequency-dependent evolutionary multi-input (M30) — is simply not part of
the open-source solver, exactly as M16 found for the real eigensolver and M17-M29
for the transfer functions, the PSD machinery, the spectral-fatigue estimators and
the multi-input / evolutionary paths this module extends.

So M30 does exactly what M16-M29 did: it ports frequency-dependent evolutionary
multi-input random fatigue as a clean LIBRARY capability and drives it with a
minimal PORT engine sub-flag (/IMPL/FATIG/MULT/MINPUT/EVOL/FCOH — the
FREQUENCY-DEPENDENT-coherence analogue of the M29 evolutionary-coherence card,
composing with /EVOL / /MINPUT / /JOINT / /NSTAT). Nothing in the M10 direct
integrator, the M16 REAL eigensolver, the M17/M18 superposition, the M19 PSD path,
the M20-M27 fatigue reductions, the M28 stationary multi-input path OR the M29
scalar-coherence evolutionary path is touched: the frequency-dependent
evolutionary multi-input path is a NEW, parallel path that CONSUMES the M28
multi-input synthesiser (``input_cross_psd_matrix`` / ``exponential_coherence`` /
``nearest_psd`` / ``stress_tensor_cross_psd_multi`` / ``measure_coherence``), the
M27 per-window tensor Miner-sum + plane re-search (``reduce_window_tensor``) and
the M29 evolutionary-multi-input machinery (``evolutionary_multi_input_summary`` as
the frequency-flat delegate, ``synthesize_evolutionary_multi_input_stress`` for the
per-window synthesis) — all read-only — and produces its answer ALONGSIDE the M29
scalar-coherence and the M28 frequency-dependent-stationary numbers so a listing
shows all three side by side.

Theory — frequency-dependent evolutionary multi-input fatigue
-------------------------------------------------------------
(Priestley, "Evolutionary spectra and non-stationary processes", J. Roy. Statist.
Soc. B 27, 1965 — the evolutionary spectrum S(omega, t), here the frequency-AND-
time varying coherence MATRIX gamma_ab(f, t); Newland, "An Introduction to Random
Vibrations, Spectral & Wavelet Analysis", 3rd ed., ch. 6-8 — multiple correlated
inputs, the FREQUENCY-DEPENDENT coherence function and the matrix input/output
spectral relation, and ch. 5-7 the short-time / windowed spectral method; Bendat &
Piersol, "Random Data", 4th ed., ch. 5-7 — the cross-spectral / FREQUENCY-DEPENDENT
coherence matrix and the MIMO relation S_yy = H S_xx H^H; Davenport 1961 / von
Karman — the atmospheric-turbulence / convection coherence field
gamma_ab(f) = exp(-decay |x_a - x_b| f / U) whose decorrelation scales with a
convection speed U; Wirsching, Paez & Ortiz 1995 — multi-input random fatigue; the
M28 frequency-dependent-coherence + M29 time-varying-coherence bases this module
converges.)

THE PER-WINDOW FREQUENCY-DEPENDENT INPUT CROSS-PSD (both f and t varying). For
ninput jointly random input processes the input cross-spectral matrix is Hermitian
PSD (M28 eq. (1), now with the coherence a function of BOTH f and t):

    S_ff(omega, t_i)[a,b] = sqrt( G_a(omega, t_i) G_b(omega, t_i) )
                            gamma_ab(f, t_i) exp( i theta_ab(t_i) )          (1)

with the FREQUENCY-DEPENDENT coherence gamma_ab(f, t_i) in [0, 1] a FULL
(nf, n, n) Hermitian stack that varies across frequency AND drifts window to
window — either
  (a) a per-pair measured / modelled SHAPE gamma_ab(f) interpolated across the
      M26/M27 window mid-time fractions s_i = (i + 1/2)/nwin from a START stack
      gamma_ab^0(f) to an END stack gamma_ab^1(f):
          gamma_ab(f, t_i) = gamma_ab^0(f) + s_i (gamma_ab^1(f) - gamma_ab^0(f)), or
  (b) the exponential / convection field gamma_ab(f, t_i) =
          exp( -decay(t_i) |x_a - x_b| f / speed(t_i) )
      with the decay coefficient decay(t_i) and/or reference speed speed(t_i)
      interpolated start -> end (a turbulence field whose DECORRELATION FREQUENCY
      f_dec ~ speed / (decay |x_a - x_b|) drifts through the mission).
The per-window auto-PSDs G_a(omega, t_i) = a_i^2 W_i(f) G_a(omega) carry the M27
drifting-shape window W_i(f) and RMS level a_i (so M30 composes the coherence
frequency-shape drift WITH the M25/M26/M27 level / spectral-shape drift). Each
window's S_ff is projected onto the NEAREST Hermitian PSD matrix (M28
``nearest_psd``) — a distance/frequency-decay coherence MODEL can leave the PSD
band by a rounding margin exactly as the constant one can.

THE PER-WINDOW STRESS-TENSOR CROSS-PSD + REDUCTION. Identical in FORM to M29 (the
M28 matrix triple product S_sigmasigma,i = H_sigma S_ff(omega, t_i) H_sigma^H, its
6x6 spectral-moment matrices, reduced by the M27 ``reduce_window_tensor`` — the
plane / F_np RE-SEARCHED per window — and Miner-summed with the duration weights).
The ONLY difference from M29 is that the window's coherence is a FREQUENCY-DEPENDENT
stack, so the effective coherent combination of the per-input stress columns is
DIFFERENT AT DIFFERENT FREQUENCIES — the low-frequency band may be nearly coherent
while the high-frequency band decorrelates, and that split MOVES through the
mission as the frequency-shape drifts. So the per-window tensor SHAPE (and its
critical plane) drift with the frequency-shape of the coherence, not merely with a
scalar coherence level.

NON-STATIONARY MULTI-INPUT MONTE-CARLO (the independent cross-check). Per-window
BLOCKS of the M28 correlated-input synthesiser (each window's S_ff(t_i)'s per-bin
eigen/Cholesky factor, M28 ``synthesize_multi_input_stress``, reused through M29's
``synthesize_evolutionary_multi_input_stress``) are concatenated in time. Each
window's stress block is PROJECTED onto that WINDOW's OWN critical plane,
rainflow-counted (ASTM E1049, the M20 counter) and Miner-summed. The synthesised
inputs' per-window measured coherence SPECTRUM (M28 ``measure_coherence``, Welch
averaging — the coherence per FREQUENCY BAND, not just a band-mean scalar) tracks
the target gamma_ab(f, t_i) — the confirmation that the drifting FREQUENCY-SHAPE is
faithfully realised.

    DELEGATIONS (the bit-identical reductions).
    * FREQUENCY-FLAT coherence -> M29 ``evolutionary_multi_input_summary`` /
      ``evolutionary_multi_input_monte_carlo_damage`` (bit-identical); M29 then
      recovers M28 / M27 in its own special cases.
    * SINGLE WINDOW / CONSTANT-IN-TIME frequency-dependent coherence -> the M28
      ``multi_input_multiaxial_summary`` / ``monte_carlo_multi_input_damage`` on the
      frequency-dependent stationary S_ff (bit-identical).
    * ninput = 1 -> M29 (which delegates to the M27 single-input path),
      bit-identical.

    WINDOW-BOUNDARY RAINFLOW CAVEAT + SHORT-TIME-STATIONARY approximation
    (documented, carried from M25/M26/M27/M29). The closed-form window Miner-sum
    counts each window's cycles INDEPENDENTLY and so MISSES cycles that STRADDLE a
    window boundary — a small documented discrepancy that vanishes as the windows
    grow long relative to the cycle period. Each window is treated as locally
    stationary with its OWN frequency-dependent coherence (piecewise-constant in
    time, sampled at the window mid-time). The window must be LONG relative to the
    carrier period and SHORT relative to the coherence-shape drift — the standard
    spectrogram trade-off.

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* LIBRARY-FIRST sub-flag (/IMPL/FATIG/MULT/MINPUT/EVOL/FCOH) — no upstream
  equivalent, exactly as established for M16-M29's PORT cards.
* WINDOWED (short-time) evolutionary coherence, NOT a continuous Wigner-Ville /
  Loeve INSTANTANEOUS coherence-matrix spectrum — DEFERRED, exactly as M26/M27/M29
  deferred the scalar / joint-tensor / coherence-matrix Wigner-Ville distributions.
* The two supported frequency-shape schedules are (a) a per-pair measured shape
  gamma_ab(f) interpolated start -> end and (b) the exponential/convection field
  with a drifting decay / speed. A fully ARBITRARY per-pair per-window shape stack
  is accepted by the library (``freq_coherence_stacks``) but the CARD exposes only
  those two schedules.
* A full NON-GAUSSIAN frequency-dependent-coherence joint-tensor evolutionary
  distribution is DEFERRED (M30 composes with the M24 non-Gaussian correction on
  the equivalent scalar but does not model a non-Gaussian frequency-time-varying
  joint-tensor distribution).
* The BASE-ACCELERATION multi-input feed remains DEFERRED (carried from M28/M29 —
  the force-pattern feed is used).
* MEAN-STRESS beyond the basic M20/M21 Goodman intercept, CRACK-GROWTH /
  fracture-mechanics fatigue and the COMPLEX-FRF stress recovery remain DEFERRED
  (the unchanged M20-M29 tail).

See PORTING_GUIDE.md roadmap M30.
"""

from __future__ import annotations

import math

import numpy as np

# reuse the M28 multi-input synthesiser + coherence models (read-only)
from .multi_input_response import (input_cross_psd_matrix, exponential_coherence,
                                   stress_tensor_cross_psd_multi)
# reuse the M27 per-window tensor machinery (read-only)
from .joint_evolutionary_fatigue import (joint_evolutionary_windows,
                                         reduce_window_tensor,
                                         _reduce_window_fixed)
from .multiaxial_fatigue import (tensor_moment_matrices,
                                 equivalent_vonmises_moments)
# reuse the M29 evolutionary-multi-input machinery (read-only): the frequency-flat
# delegate + the per-window synthesiser
from . import evolutionary_multi_input as emi


# ============================================================================
# Frequency-dependent coherence detection + stack builders (build item 1a)
# ============================================================================

def is_frequency_dependent(gamma):
    """True when ``gamma`` is a FULL frequency-dependent coherence stack (a
    (nf, n, n) array), False when it is a frequency-FLAT scalar or (n, n) matrix.

    This is the switch that decides whether a call is a genuine M30
    frequency-dependent-coherence problem or the M29 frequency-flat special case
    (which is DELEGATED bit-identically). ``None`` (mutually incoherent — a
    diagonal S_ff) is frequency-flat."""
    if gamma is None:
        return False
    g = np.asarray(gamma, dtype=float)
    return g.ndim >= 3


def _both_frequency_flat(gamma0, gamma1):
    """True when NEITHER the start nor the end coherence is frequency-dependent —
    the whole schedule is frequency-flat and delegates to M29 exactly."""
    return not (is_frequency_dependent(gamma0) or is_frequency_dependent(gamma1))


def measured_coherence_stack(freqs, shape, ninput, pairs=None):
    """Build a FULL (nf, ninput, ninput) Hermitian coherence stack gamma_ab(f)
    from a per-frequency MEASURED / MODELLED shape ``shape`` (nf,) applied to the
    off-diagonal input pairs (theory eq. (1)(a)).

    ``shape`` is a per-frequency coherence in [0, 1] (a measured gamma(f), e.g. a
    /FUNCT sampled on the sweep grid) applied to every off-diagonal pair by default,
    or only to the pairs in ``pairs`` (a list of (a, b) index tuples) — the others
    held incoherent (gamma = 0). The diagonal is 1. This is the frequency-dependent
    generalisation of the M28 ``constant_coherence`` (a scalar per pair): here the
    per-pair coherence is a frequency SHAPE. Returns (nf, ninput, ninput)."""
    f = np.asarray(freqs, dtype=float)
    nf = f.size
    n = int(ninput)
    sh = np.clip(np.asarray(shape, dtype=float).ravel(), 0.0, 1.0)
    if sh.size != nf:
        raise ValueError("measured_coherence_stack: shape must sample the sweep "
                         f"grid (got {sh.size}, need {nf}).")
    g = np.zeros((nf, n, n))
    for a in range(n):
        g[:, a, a] = 1.0
    prs = pairs if pairs is not None else [(a, b) for a in range(n)
                                           for b in range(a + 1, n)]
    for (a, b) in prs:
        g[:, a, b] = sh
        g[:, b, a] = sh
    return g


def exponential_coherence_stack(freqs, positions, decay, ref_speed=1.0):
    """The M28 exponential / convection coherence field as a FULL (nf, n, n) stack
    gamma_ab(f) = exp(-decay |x_a - x_b| f / ref_speed) (theory eq. (1)(b);
    Davenport / von Karman). A thin wrapper over the M28 ``exponential_coherence``
    that returns ONLY the (nf, n, n) magnitude stack (M30 assembles the phase
    separately from the schedule)."""
    g, _th = exponential_coherence(freqs, positions, decay, ref_speed=ref_speed)
    return g


def freq_coherence_stacks(freqs, nwin, gamma0, gamma1=None):
    """Build the per-window schedule of FULL (nf, n, n) FREQUENCY-DEPENDENT
    coherence stacks gamma_ab(f, t_i) (theory eq. (1)(a)): interpolate the START
    stack ``gamma0`` (nf, n, n) to the END stack ``gamma1`` (nf, n, n) LINEARLY
    across the M26/M27 window mid-time fractions s_i = (i + 1/2)/nwin (the SAME
    parameterisation the M29 ``coherence_schedule`` uses, so the frequency-shape
    drift stays consistent in time with the M27 spectral-shape window). ``gamma1``
    defaults to ``gamma0`` (a STATIONARY frequency-dependent coherence — the M28
    special case). Returns a list of ``nwin`` (nf, n, n) stacks, each clipped to
    [0, 1] with a unit diagonal.

    NOTE this MIRRORS ``coherence_schedule`` (whose broadcasting already handles
    (nf, n, n) stacks) but names the frequency-dependent case explicitly and
    enforces the [0, 1] / unit-diagonal invariants per window (a drifting model can
    briefly overshoot at a boundary bin)."""
    g0 = np.asarray(gamma0, dtype=float)
    if g0.ndim != 3:
        raise ValueError("freq_coherence_stacks needs a (nf, n, n) start stack; "
                         f"got shape {g0.shape}.")
    g1 = g0 if gamma1 is None else np.asarray(gamma1, dtype=float)
    if g1.shape != g0.shape:
        raise ValueError("freq_coherence_stacks: start / end stacks must match "
                         f"in shape ({g0.shape} vs {g1.shape}).")
    n = g0.shape[1]
    s = np.array([0.5]) if nwin <= 1 else (np.arange(nwin) + 0.5) / nwin
    out = []
    for si in s:
        gi = np.clip(g0 + si * (g1 - g0), 0.0, 1.0)
        for a in range(n):
            gi[:, a, a] = 1.0                        # unit diagonal (gamma_aa = 1)
        out.append(gi)
    return out


def exponential_drift_stacks(freqs, positions, nwin, decay0, decay1=None,
                             speed0=1.0, speed1=None):
    """Build the per-window schedule of exponential / convection coherence stacks
    with a TIME-VARYING decay coefficient and/or reference speed (theory eq.
    (1)(b)): interpolate ``decay0`` -> ``decay1`` and ``speed0`` -> ``speed1``
    LINEARLY across the window mid-time fractions and, per window, build
    gamma_ab(f, t_i) = exp(-decay(t_i) |x_a - x_b| f / speed(t_i)).

    This is the M30 lift of the M28 exponential model (which M29 held STATIONARY):
    a turbulence / convection field whose DECORRELATION FREQUENCY
    f_dec ~ speed / (decay |x_a - x_b|) drifts through the mission — the
    decorrelation "moves up in frequency" as decay falls or the convection speed
    ramps. ``decay1`` / ``speed1`` default to the start values (a STATIONARY
    exponential field — the M28 special case). Returns a list of ``nwin``
    (nf, n, n) coherence stacks."""
    d0 = float(decay0)
    d1 = d0 if decay1 is None else float(decay1)
    v0 = float(speed0) if speed0 else 1.0
    v1 = v0 if speed1 is None else (float(speed1) or 1.0)
    s = np.array([0.5]) if nwin <= 1 else (np.arange(nwin) + 0.5) / nwin
    out = []
    for si in s:
        di = d0 + si * (d1 - d0)
        vi = v0 + si * (v1 - v0)
        out.append(exponential_coherence_stack(freqs, positions, di,
                                               ref_speed=vi))
    return out


# ============================================================================
# The per-window frequency-dependent input cross-PSD schedule (build item 1a)
# ============================================================================

def freq_evolutionary_input_windows(freqs, auto_psds, durations, gamma_stacks,
                                    phase0=0.0, phase1=None, fc=0.0, bw=0.0,
                                    scales=None):
    """Build the per-window schedule of Hermitian input cross-spectral matrices
    S_ff(omega, t_i) (theory eq. (1)) from a PRE-BUILT per-window schedule of FULL
    (nf, n, n) frequency-dependent coherence stacks ``gamma_stacks`` (one per
    window, e.g. from ``freq_coherence_stacks`` or ``exponential_drift_stacks``).

    Per window i assemble the M27 windowed / scaled auto-PSDs
    G_a(omega, t_i) = a_i^2 W_i(f) G_a(omega) (the SAME drifting-shape schedule
    ``joint_evolutionary_windows`` builds) and
    S_ff(omega, t_i) = input_cross_psd_matrix(G_i, gamma = gamma_stacks[i],
    phase = phase_i) — the coherence FREQUENCY-STACK for window i, projected onto
    the nearest Hermitian PSD matrix (M28 ``input_cross_psd_matrix`` calls
    ``nearest_psd``). The phase theta_ab(t_i) interpolates ``phase0`` -> ``phase1``
    across the windows (a constant per-window antisymmetric matrix, as M28/M29).

    Returns a list of window dicts {W, fc, bw, scale, duration, gamma_stack, phase,
    G, Sff, projected, min_eig} — the SAME structure M29 ``evolutionary_input_
    windows`` returns (so ``synthesize_evolutionary_multi_input_stress`` consumes it
    UNCHANGED), plus the full per-window ``gamma_stack`` (nf, n, n)."""
    f = np.asarray(freqs, dtype=float)
    G0 = np.clip(np.asarray(auto_psds, dtype=float), 0.0, None)
    if G0.ndim == 1:
        G0 = G0[:, None]
    shape_wins = joint_evolutionary_windows(f, durations, fc, bw, scales)
    nwin = len(shape_wins)
    if len(gamma_stacks) != nwin:
        raise ValueError("freq_evolutionary_input_windows: gamma_stacks length "
                         f"({len(gamma_stacks)}) must match the window count "
                         f"({nwin}).")
    p0 = float(phase0)
    p1 = p0 if phase1 is None else float(phase1)
    s = np.array([0.5]) if nwin == 1 else (np.arange(nwin) + 0.5) / nwin
    out = []
    for i, (w, gstack) in enumerate(zip(shape_wins, gamma_stacks)):
        # the M27 windowed / scaled auto-PSDs (a per-frequency scalar, so it
        # commutes with the frequency-dependent coherence assembly)
        Gi = (w["scale"] ** 2) * w["W"][:, None] * G0            # (nf, ninput)
        phase_i = p0 + float(s[i]) * (p1 - p0)
        mi = input_cross_psd_matrix(Gi, gamma=gstack, phase=phase_i)
        out.append({"W": w["W"], "fc": w["fc"], "bw": w["bw"],
                    "scale": w["scale"], "duration": w["duration"],
                    "gamma_stack": np.asarray(gstack, dtype=float),
                    "phase": phase_i, "G": Gi, "Sff": mi["Sff"],
                    "projected": mi["projected"], "min_eig": mi["min_eig"]})
    return out


# ============================================================================
# Band-resolved coherence diagnostics (build item 1a — the reporting)
# ============================================================================

def representative_pair_spectrum(gamma_stack):
    """The representative off-diagonal coherence SPECTRUM gamma_01(f) of a
    (nf, n, n) coherence stack (the (0, 1) pair, or the mean off-diagonal for
    n > 2) — a (nf,) array in [0, 1]. This is the frequency-resolved coherence the
    band diagnostics and the measured-coherence tracking compare against (NOT a
    band-mean scalar)."""
    g = np.asarray(gamma_stack, dtype=float)
    n = g.shape[1]
    if n < 2:
        return np.ones(g.shape[0])
    iu = np.triu_indices(n, 1)
    # mean over the off-diagonal pairs, per frequency
    return np.clip(np.mean(np.abs(g[:, iu[0], iu[1]]), axis=1), 0.0, 1.0)


def decorrelation_frequency(freqs, gamma_stack, level=0.5):
    """The characteristic DECORRELATION FREQUENCY f_dec of a coherence stack — the
    lowest frequency at which the representative off-diagonal coherence spectrum
    gamma_01(f) crosses below ``level`` (default 0.5). The single scalar that
    summarises WHERE in frequency the inputs decorrelate; its DRIFT window to window
    is the "decorrelation moves up in frequency" diagnostic. Returns ``nan`` if the
    coherence never crosses ``level`` (always coherent, or always incoherent)."""
    f = np.asarray(freqs, dtype=float)
    g = representative_pair_spectrum(gamma_stack)
    below = g < level
    if not np.any(below) or np.all(below):
        return float("nan")
    # first index where it drops below the level
    idx = int(np.argmax(below))
    if idx == 0:
        return float(f[0])
    # linear interpolation between the bracketing bins for a smooth f_dec
    g0, g1 = g[idx - 1], g[idx]
    f0, f1 = f[idx - 1], f[idx]
    if g0 == g1:
        return float(f1)
    frac = (g0 - level) / (g0 - g1)
    return float(f0 + frac * (f1 - f0))


def band_coherence(freqs, gamma_stack, nband=4):
    """The band-mean representative coherence in ``nband`` equal frequency bands —
    a coarse (nband,) summary of the coherence SPECTRUM for the listing (the low
    band coherent, the high band decorrelating, as the frequency shape drifts).
    Returns (band_centres, band_gamma) both (nband,)."""
    f = np.asarray(freqs, dtype=float)
    g = representative_pair_spectrum(gamma_stack)
    edges = np.linspace(f.min(), f.max(), nband + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    vals = np.zeros(nband)
    for k in range(nband):
        sel = (f >= edges[k]) & (f <= edges[k + 1])
        vals[k] = float(np.mean(g[sel])) if np.any(sel) else float("nan")
    return centres, vals


# ============================================================================
# The frequency-dependent evolutionary multi-input window Miner-sum (item 1a+1b)
# ============================================================================

def freq_evolutionary_multi_input_summary(
        omega, Hcols, auto_psds, durations, m, C, gamma0, gamma1=None,
        phase0=0.0, phase1=None, fc=0.0, bw=0.0, scales=None, mean_stress=0.0,
        ultimate=0.0, naz=24, npol=13, drift=True):
    """The FREQUENCY-DEPENDENT + TIME-VARYING (evolutionary) MULTI-INPUT window
    Miner-sum damage (theory eq. (3), M29 FORM) for ONE element, with the coherence
    a FULL frequency-dependent stack gamma_ab(f, t_i) drifting window to window:
    build the per-window frequency-dependent input cross-PSD S_ff(omega, t_i), form
    the per-window multi-input stress-tensor cross-PSD S_sigmasigma,i = H_sigma
    S_ff(t_i) H_sigma^H, recompute its 6x6 moment matrices, reduce EACH window with
    the M27 critical-plane machinery — the plane / F_np RE-SEARCHED from the window's
    OWN tensor (so the plane may ROTATE as the coherence FREQUENCY-SHAPE evolves) —
    and Palmgren-Miner SUM the per-window multiaxial damages duration-weighted.

    ``gamma0`` / ``gamma1`` are the START / END coherence: EITHER scalars / (n, n)
    matrices (FREQUENCY-FLAT — the whole call DELEGATES to the M29
    ``evolutionary_multi_input_summary`` BIT-IDENTICALLY, M29 being exactly the
    frequency-flat special case) OR FULL (nf, n, n) frequency-dependent stacks (the
    genuine M30 path). ``gamma1`` defaults to ``gamma0`` (a STATIONARY coherence).

    ``omega`` (nf,) angular grid, ``Hcols`` (nf, 6, ninput) the element's per-input
    Voigt stress FRF column stack, ``auto_psds`` (nf, ninput) the stationary per-input
    auto-PSDs, ``durations`` the per-window times, ``fc`` / ``bw`` / ``scales`` the
    M27 drifting-shape schedule, ``m`` / ``C`` the S-N law. ``drift`` — if True the
    critical plane is RE-SEARCHED per window (the whole point); if False the plane is
    FIXED at the mission-averaged-coherence stationary plane.

    Delegations (the exact-reduction contract):
    * FREQUENCY-FLAT gamma -> the M29 scalar-coherence answer EXACTLY (delegation);
    * ninput = 1 -> M29 (which delegates to M27), bit-identical;
    * nwin = 1 AND constant-in-time frequency-dependent coherence AND flat unit
      window -> the M28 ``multi_input_multiaxial_summary`` on the frequency-dependent
      stationary S_ff, BIT-IDENTICAL to the M28 frequency-dependent answer.

    Returns a dict with, for each reduction, the window Miner-sum {damage_rate, life,
    damage, total_time}; plus ``windows`` (the per-window frequency-coherence /
    critical-plane / RMS drift, each carrying its ``gamma_spectrum`` and
    ``decorr_freq``), ``decorr_drift`` (the max - min decorrelation frequency across
    the windows — the "decorrelation moves up in frequency" measure),
    ``plane_rotation_deg``, ``fnp_drift``, ``freq_dependent`` (True on the genuine
    M30 path), ``delegated`` / ``delegated_m29``, ``ninput``, ``nwin`` and the
    promoted von-Mises ``damage_rate`` / ``life``."""
    omega = np.asarray(omega, dtype=float)
    Hcols = np.asarray(Hcols)
    G0 = np.clip(np.asarray(auto_psds, dtype=float), 0.0, None)
    if G0.ndim == 1:
        G0 = G0[:, None]
    ninput = Hcols.shape[2]
    T = np.asarray(durations, dtype=float).ravel()
    nwin = T.size
    freqs = omega / (2.0 * np.pi)

    # ----- DELEGATION A: frequency-FLAT coherence -> the M29 answer EXACTLY --------
    # a scalar / (n, n) coherence (or ninput = 1, which has no off-diagonal) is the
    # M29 scalar-coherence problem; delegate so M30 is bit-identical to M29 (which in
    # turn recovers M28 / M27 in its own special cases).
    if ninput == 1 or _both_frequency_flat(gamma0, gamma1):
        summ = emi.evolutionary_multi_input_summary(
            omega, Hcols, auto_psds, durations, m, C, gamma0=gamma0,
            gamma1=gamma1, phase0=phase0, phase1=phase1, fc=fc, bw=bw,
            scales=scales, mean_stress=mean_stress, ultimate=ultimate, naz=naz,
            npol=npol, drift=drift)
        summ = dict(summ)
        summ["method"] = "freq_evolutionary_multi_input"
        summ["freq_dependent"] = False
        summ["delegated_m29"] = True
        summ.setdefault("decorr_drift", 0.0)
        return summ

    # ----- promote the (possibly scalar) start/end to full (nf, n, n) stacks ------
    g0_stack, g1_stack = _promote_stacks(freqs, gamma0, gamma1, ninput)
    const_in_time = np.array_equal(g0_stack, g1_stack)

    # ----- DELEGATION B: single window + constant-in-time freq coherence -> M28 ---
    if (const_in_time and nwin == 1
            and emi._flat_unit_window(fc, bw, scales, nwin)):
        from . import multi_input_response as mir
        Sff = input_cross_psd_matrix(G0, gamma=g0_stack, phase=phase0)["Sff"]
        Scross = stress_tensor_cross_psd_multi(Hcols, Sff)
        m28 = mir.multi_input_multiaxial_summary(
            Scross, omega, m, C, mean_stress=mean_stress, ultimate=ultimate,
            naz=naz, npol=npol)
        return _wrap_m28_single_window(m28, freqs, T, g0_stack, ninput)

    # ----- GENERAL FREQUENCY-DEPENDENT DRIFT PATH ---------------------------------
    gamma_stacks = freq_coherence_stacks(freqs, nwin, g0_stack, g1_stack)
    windows = freq_evolutionary_input_windows(
        freqs, G0, durations, gamma_stacks, phase0=phase0, phase1=phase1,
        fc=fc, bw=bw, scales=scales)

    # the mission-averaged-coherence STATIONARY reduction — the reference the
    # reporting compares against and the FIXED plane for the drift=False branch
    g_mean = 0.5 * (g0_stack + g1_stack)
    p_mean = 0.5 * (float(phase0) + (float(phase0) if phase1 is None
                                     else float(phase1)))
    Sff_stat = input_cross_psd_matrix(G0, gamma=g_mean, phase=p_mean)["Sff"]
    Scross_stat = stress_tensor_cross_psd_multi(Hcols, Sff_stat)
    Mstat = tensor_moment_matrices(omega, Scross_stat, nmax=4)
    stat = reduce_window_tensor(Mstat, m, C, mean_stress, ultimate, naz, npol)

    reductions = ("von_mises", "normal_plane", "shear_plane")
    D = {k: 0.0 for k in reductions}
    Ttot = 0.0
    wout = []
    shear_normals = []
    normal_normals = []
    fnps = []
    decorrs = []
    shapes = []
    for w in windows:
        Scross_i = stress_tensor_cross_psd_multi(Hcols, w["Sff"])   # eq. (2)
        Mi = tensor_moment_matrices(omega, Scross_i, nmax=4)
        if drift:
            red = reduce_window_tensor(Mi, m, C, mean_stress, ultimate, naz, npol)
        else:
            red = _reduce_window_fixed(Mi, stat, m, C, mean_stress, ultimate)
        Ti = w["duration"]
        Ttot += Ti
        for k in reductions:
            D[k] += red[k]["damage_rate"] * Ti
        shear_normals.append(np.asarray(red["shear_plane"]["normal"], dtype=float))
        normal_normals.append(np.asarray(red["normal_plane"]["normal"],
                                         dtype=float))
        fnps.append(float(red["F_np"]))
        gspec = representative_pair_spectrum(w["gamma_stack"])
        fdec = decorrelation_frequency(freqs, w["gamma_stack"])
        decorrs.append(fdec)
        _bc_c, bc_v = band_coherence(freqs, w["gamma_stack"])
        wout.append({
            "fc": w["fc"], "bw": w["bw"], "scale": w["scale"], "duration": Ti,
            "gamma_spectrum": gspec, "band_gamma": bc_v,
            "decorr_freq": fdec,
            "gamma_band_mean": float(np.mean(gspec)),
            "projected": bool(w["projected"]), "min_eig": float(w["min_eig"]),
            "normal_n": np.asarray(red["normal_plane"]["normal"], dtype=float),
            "shear_n": np.asarray(red["shear_plane"]["normal"], dtype=float),
            "normal_proj": np.asarray(red["normal_plane"]["proj"], dtype=float),
            "shear_proj": np.asarray(red["shear_plane"]["proj"], dtype=float),
            "F_np": float(red["F_np"]), "sigma_vm": float(red["sigma_vm"]),
            "vm_rate": float(red["von_mises"]["damage_rate"]),
            "normal_rate": float(red["normal_plane"]["damage_rate"]),
            "shear_rate": float(red["shear_plane"]["damage_rate"])})
        s0 = float(equivalent_vonmises_moments(Mi)[0])
        shapes.append(Mi[0] / s0 if s0 > 0 else Mi[0])

    # constant tensor shape: every window's normalised M_0 is the same (neither the
    # coherence frequency-shape nor the spectral shape drifts the tensor orientation)
    const_shape = len(shapes) <= 1 or all(
        np.allclose(shapes[k], shapes[0], rtol=1e-9, atol=1e-12)
        for k in range(1, len(shapes)))

    def _swing(nrmls):
        n0 = nrmls[0]
        return max((math.degrees(math.acos(abs(float(np.clip(np.dot(n0, nn),
                                                             -1.0, 1.0)))))
                    for nn in nrmls[1:]), default=0.0)

    rot = 0.0
    if not const_shape and shear_normals:
        rot = max(_swing(normal_normals), _swing(shear_normals))
    fnp_drift = (max(fnps) - min(fnps)) if fnps else 0.0
    fin_dec = [d for d in decorrs if np.isfinite(d)]
    decorr_drift = (max(fin_dec) - min(fin_dec)) if len(fin_dec) >= 2 else 0.0

    out = {"method": "freq_evolutionary_multi_input", "ninput": int(ninput),
           "nwin": len(windows), "drift": bool(drift), "delegated": None,
           "delegated_m29": False, "freq_dependent": True,
           "constant_shape": bool(const_shape),
           "constant_coherence": bool(const_in_time),
           "plane_rotation_deg": float(rot), "fnp_drift": float(fnp_drift),
           "decorr_drift": float(decorr_drift), "windows": wout,
           "stationary": stat, "total_time": Ttot}
    for k in reductions:
        dr = D[k] / Ttot if Ttot > 0 else 0.0
        out[k] = {"damage": D[k], "total_time": Ttot, "damage_rate": dr,
                  "life": (math.inf if dr <= 0.0 else 1.0 / dr)}
    vm = out["von_mises"]
    out.update({"damage_rate": vm["damage_rate"], "life": vm["life"]})
    return out


def _promote_stacks(freqs, gamma0, gamma1, ninput):
    """Promote the (start, end) coherence to a pair of full (nf, n, n) stacks. At
    least one of ``gamma0`` / ``gamma1`` is already a (nf, n, n) stack (this is the
    genuine M30 path); a frequency-FLAT partner (scalar / (n, n)) is promoted to a
    frequency-constant (nf, n, n) stack via the M28 ``constant_coherence`` so the
    two ends interpolate consistently."""
    from .multi_input_response import constant_coherence
    f = np.asarray(freqs, dtype=float)
    nf = f.size
    n = int(ninput)

    def _to_stack(g):
        if g is None:
            gm, _th = constant_coherence(n, 0.0)          # incoherent (diagonal)
            return np.broadcast_to(gm, (nf, n, n)).copy()
        g = np.asarray(g, dtype=float)
        if g.ndim >= 3:
            return np.clip(g.astype(float), 0.0, 1.0)
        gm, _th = constant_coherence(n, g)                # scalar / (n,n) -> matrix
        return np.broadcast_to(gm, (nf, n, n)).copy()

    g0 = _to_stack(gamma0)
    g1 = g0 if gamma1 is None else _to_stack(gamma1)
    return g0, g1


def _wrap_m28_single_window(m28, freqs, durations, gamma_stack, ninput):
    """Wrap the M28 ``multi_input_multiaxial_summary`` output into the M30 window
    Miner-sum dict shape for the single-window / constant frequency-coherence
    delegation — so the caller sees the SAME structure whether the answer came from
    the M28 delegation or the general drift path. BIT-IDENTICAL to the M28
    frequency-dependent answer (no re-reduction)."""
    T = np.asarray(durations, dtype=float).ravel()
    Ttot = float(T.sum()) if T.size else 1.0
    f = np.asarray(freqs, dtype=float)
    reductions = ("von_mises", "normal_plane", "shear_plane")
    gspec = representative_pair_spectrum(gamma_stack)
    fdec = decorrelation_frequency(f, gamma_stack)
    _bc_c, bc_v = band_coherence(f, gamma_stack)
    out = {"method": "freq_evolutionary_multi_input", "ninput": int(ninput),
           "nwin": 1, "drift": True, "delegated": "m28_single_window",
           "delegated_m29": False, "freq_dependent": True,
           "constant_shape": True, "constant_coherence": True,
           "plane_rotation_deg": 0.0, "fnp_drift": 0.0, "decorr_drift": 0.0,
           "stationary": None, "total_time": Ttot}
    for k in reductions:
        dr = float(m28[k]["summary"]["dirlik"]["damage_rate"])
        out[k] = {"damage": dr * Ttot, "total_time": Ttot, "damage_rate": dr,
                  "life": (math.inf if dr <= 0.0 else 1.0 / dr)}
    sp = m28["shear_plane"]
    npl = m28["normal_plane"]
    out["windows"] = [{
        "fc": 0.0, "bw": 0.0, "scale": 1.0, "duration": Ttot,
        "gamma_spectrum": gspec, "band_gamma": bc_v, "decorr_freq": fdec,
        "gamma_band_mean": float(np.mean(gspec)),
        "projected": False, "min_eig": 0.0,
        "normal_n": np.asarray(npl["normal"], dtype=float),
        "shear_n": np.asarray(sp["normal"], dtype=float),
        "normal_proj": np.asarray(npl["proj"], dtype=float),
        "shear_proj": np.asarray(sp["proj"], dtype=float),
        "F_np": 0.0,
        "sigma_vm": float(math.sqrt(max(float(m28["von_mises"]["moments"][0]),
                                        0.0))),
        "vm_rate": out["von_mises"]["damage_rate"],
        "normal_rate": out["normal_plane"]["damage_rate"],
        "shear_rate": out["shear_plane"]["damage_rate"]}]
    out["m28"] = m28
    vm = out["von_mises"]
    out.update({"damage_rate": vm["damage_rate"], "life": vm["life"]})
    return out


# ============================================================================
# Non-stationary MULTI-INPUT multivariate Monte-Carlo cross-check (build item 2)
# ============================================================================

def freq_evolutionary_multi_input_monte_carlo_damage(
        omega, Hcols, auto_psds, durations, m, C, seed, gamma0, gamma1=None,
        phase0=0.0, phase1=None, fc=0.0, bw=0.0, scales=None, fs=None,
        mean_stress=0.0, ultimate=0.0, naz=24, npol=13, reduction="shear_plane",
        summary=None, measure=False):
    """The TIME-DOMAIN frequency-dependent evolutionary multi-input damage rate by
    NON-STATIONARY MULTI-INPUT multivariate Monte-Carlo: synthesise the multi-input
    history from per-window BLOCKS of the M28 correlated-input synthesiser (each
    window's frequency-dependent S_ff(t_i)'s per-bin eigen/Cholesky factor, via the
    M29 ``synthesize_evolutionary_multi_input_stress`` reused UNCHANGED), PROJECT
    each window's block onto that WINDOW's OWN critical plane, rainflow-count (ASTM
    E1049) and Palmgren-Miner SUM.

    Delegations:
    * FREQUENCY-FLAT coherence (or ninput = 1) -> the M29
      ``evolutionary_multi_input_monte_carlo_damage`` BIT-IDENTICALLY;
    * single window + constant-in-time frequency coherence -> the M28
      ``monte_carlo_multi_input_damage`` on the frequency-dependent stationary S_ff,
      BIT-IDENTICAL.

    ``measure`` — if True, ALSO measure each window's synthesised-input coherence
    SPECTRUM (M28 ``measure_coherence``, Welch, per frequency band) and return it as
    ``window_gamma_spectrum`` (a (nwin, nband) array) + the band-mean scalar
    ``window_gamma`` (for the coarse listing) so the drift of the FREQUENCY SHAPE —
    not just a band-mean scalar — is confirmed to track the target gamma_ab(f, t_i).

    Returns the M20 Monte-Carlo dict shape plus per-window ``window_rms`` /
    ``window_nu0`` / ``window_decorr`` (the measured decorrelation frequency drift)
    and ``delegated`` / ``delegated_m29``."""
    from . import spectral_fatigue as sf
    omega = np.asarray(omega, dtype=float)
    Hcols = np.asarray(Hcols)
    G0 = np.clip(np.asarray(auto_psds, dtype=float), 0.0, None)
    if G0.ndim == 1:
        G0 = G0[:, None]
    ninput = Hcols.shape[2]
    freqs = omega / (2.0 * np.pi)
    T = np.asarray(durations, dtype=float).ravel()
    nwin = T.size
    Ceff = sf._goodman_C(C, m, mean_stress, ultimate)
    projkey = "shear_proj" if reduction == "shear_plane" else "normal_proj"

    # ----- DELEGATION A: frequency-FLAT (or single input) -> the M29 MC EXACTLY ---
    if ninput == 1 or _both_frequency_flat(gamma0, gamma1):
        out = emi.evolutionary_multi_input_monte_carlo_damage(
            omega, Hcols, auto_psds, durations, m, C, seed=seed, gamma0=gamma0,
            gamma1=gamma1, phase0=phase0, phase1=phase1, fc=fc, bw=bw,
            scales=scales, fs=fs, mean_stress=mean_stress, ultimate=ultimate,
            naz=naz, npol=npol, reduction=reduction,
            summary=(summary if (summary is None
                                 or not summary.get("delegated_m29")) else None),
            measure=measure)
        out = dict(out)
        out["delegated_m29"] = True
        return out

    g0_stack, g1_stack = _promote_stacks(freqs, gamma0, gamma1, ninput)
    const_in_time = np.array_equal(g0_stack, g1_stack)

    # ----- DELEGATION B: single window + constant freq coherence -> the M28 MC ----
    if (const_in_time and nwin == 1
            and emi._flat_unit_window(fc, bw, scales, nwin)):
        from . import multi_input_fatigue as mif
        from . import multi_input_response as mir
        Sff = input_cross_psd_matrix(G0, gamma=g0_stack, phase=phase0)["Sff"]
        if summary is not None and summary.get("windows"):
            proj = np.asarray(summary["windows"][0][projkey], dtype=float)
        else:
            Scross = stress_tensor_cross_psd_multi(Hcols, Sff)
            m28 = mir.multi_input_multiaxial_summary(
                Scross, omega, m, C, mean_stress=mean_stress, ultimate=ultimate,
                naz=naz, npol=npol)
            proj = np.asarray(m28[reduction]["proj"], dtype=float)
        dur = float(T[0]) if T.size else 1.0
        out = mif.monte_carlo_multi_input_damage(
            freqs, Sff, Hcols, proj, m, C, dur, seed, fs=fs,
            mean_stress=mean_stress, ultimate=ultimate)
        out = dict(out)
        out["delegated"] = "m28_single_window"
        out["delegated_m29"] = False
        return out

    # ----- GENERAL DRIFT PATH: per-window blocks, per-window plane projection -----
    if summary is None or summary.get("delegated_m29"):
        summary = freq_evolutionary_multi_input_summary(
            omega, Hcols, auto_psds, durations, m, C, gamma0, gamma1, phase0,
            phase1, fc, bw, scales, mean_stress, ultimate, naz, npol, drift=True)
    gamma_stacks = freq_coherence_stacks(freqs, nwin, g0_stack, g1_stack)
    windows = freq_evolutionary_input_windows(
        freqs, G0, durations, gamma_stacks, phase0=phase0, phase1=phase1,
        fc=fc, bw=bw, scales=scales)
    # reuse the M29 per-window synthesiser UNCHANGED (it only reads w["Sff"] /
    # w["duration"], which our windows carry with the frequency-dependent S_ff)
    t, X, info = emi.synthesize_evolutionary_multi_input_stress(
        freqs, windows, Hcols, seed, fs=fs)
    edges = info["edges"]
    fs = info["fs"]
    projs = [np.asarray(w[projkey], dtype=float) for w in summary["windows"]]

    D = 0.0
    ncyc = 0.0
    window_rms = []
    window_nu0 = []
    window_gamma = []
    window_gamma_spec = []
    window_decorr = []
    for i in range(len(windows)):
        sel = (t >= edges[i]) & (t < edges[i + 1])
        seg = X[sel]
        if seg.shape[0] < 2:
            window_rms.append(0.0)
            window_nu0.append(0.0)
            window_gamma.append(np.nan)
            window_gamma_spec.append(None)
            window_decorr.append(np.nan)
            continue
        s = seg @ projs[min(i, len(projs) - 1)]           # projected scalar
        ranges, counts = sf.rainflow_count(s)
        D += (float(np.sum(counts * ranges ** m) / Ceff) if ranges.size else 0.0)
        ncyc += float(counts.sum())
        window_rms.append(float(np.std(s)))
        sc = s - np.mean(s)
        ups = np.sum((sc[:-1] < 0.0) & (sc[1:] >= 0.0))
        dur = float(edges[i + 1] - edges[i])
        window_nu0.append(ups / dur if dur > 0 else 0.0)
        if measure:
            fband, gband, fdec = _measure_window_coherence_spectrum(
                freqs, windows[i], seed + i, fs)
            window_gamma_spec.append(gband)
            window_gamma.append(float(np.nanmean(gband)))
            window_decorr.append(fdec)

    Ttot = float(edges[-1]) if edges.size else float(np.sum(durations))
    dr = D / Ttot if Ttot > 0 else 0.0
    nu = ncyc / Ttot if Ttot > 0 else 0.0
    tf, s_eq = sf.life_and_equivalent(dr, nu, m, Ceff)
    out = {"method": "freq_evolutionary_multi_input_monte_carlo",
           "damage_rate": dr, "life": tf, "s_eq": s_eq, "ncycles": ncyc,
           "duration": Ttot, "reduction": reduction, "delegated": None,
           "delegated_m29": False, "window_rms": np.asarray(window_rms),
           "window_nu0": np.asarray(window_nu0)}
    if measure:
        out["window_gamma"] = np.asarray(window_gamma)
        out["window_gamma_spectrum"] = window_gamma_spec
        out["window_decorr"] = np.asarray(window_decorr)
    return out


def _measure_window_coherence_spectrum(freqs, window, seed, fs, nband=4):
    """Measure ONE window's synthesised-input coherence SPECTRUM (the representative
    off-diagonal coherence per FREQUENCY BAND, M28 ``measure_coherence`` by Welch
    averaging) — the confirmation that the drifting FREQUENCY-SHAPE gamma_ab(f, t_i)
    is faithfully realised (NOT just a band-mean scalar). Synthesises the window's
    correlated INPUT histories (long enough for a stable Welch average), measures the
    coherence per band and returns (band_centres, band_gamma (nband,),
    decorrelation_freq) so it compares band-by-band to the target frequency shape."""
    from . import multi_input_fatigue as mif
    f = np.asarray(freqs, dtype=float)
    fmax = float(f.max())
    fmin = float(f.min())
    # a dedicated long record for the coherence estimate (the damage segment may be
    # too short for a stable Welch average)
    dur = max(window["duration"], 6000.0 / max(fmax, 1.0))
    _t, F = mif.synthesize_multi_input_forces(f, window["Sff"], dur, int(seed),
                                              fs=fs)
    fsr = 1.0 / (_t[1] - _t[0]) if _t.size > 1 else fs
    nper = min(2048, max(128, F.shape[0] // 8))
    fc_meas, coh = mif.measure_coherence(F, fsr, nperseg=nper)
    # representative off-diagonal (0,1) coherence -> sqrt(coh) to compare to gamma
    gmeas = np.sqrt(np.clip(coh[:, 0, 1], 0.0, 1.0))
    # bin the measured coherence into the SAME nband equal bands as the target
    edges = np.linspace(fmin, fmax, nband + 1)
    band = np.full(nband, np.nan)
    for k in range(nband):
        sel = (fc_meas >= edges[k]) & (fc_meas <= edges[k + 1])
        if np.any(sel):
            band[k] = float(np.mean(gmeas[sel]))
    # a measured decorrelation frequency (first band crossing below 0.5)
    below = np.where(band < 0.5)[0]
    fdec = float(0.5 * (edges[below[0]] + edges[below[0] + 1])) if below.size \
        else float("nan")
    return fc_meas, band, fdec

"""
Fully non-stationary / EVOLUTIONARY MULTI-INPUT random fatigue — M29: the
frequency-domain fatigue-damage estimate of a MULTI-INPUT random-vibration
response whose full INPUT cross-spectral matrix S_ff(omega, t) — the coherence
gamma_ab(t) and phase theta_ab(t) (and/or the auto-PSDs G_a(t)) — DRIFTS with
time, driving a per-window multi-input stress-tensor cross-PSD S_sigmasigma(omega,
t_i) = H_sigma S_ff(t_i) H_sigma^H whose critical plane / F_np may DRIFT as the
input coherence evolves, reduced PER WINDOW by the M20-M27 estimator family + the
M28 multi-input path and Palmgren-Miner-summed, cross-validated against a
NON-STATIONARY MULTI-INPUT multivariate time-domain Monte-Carlo.

WHERE M29 SITS RELATIVE TO M27 / M28. M29 is the CONVERGENCE of M27 (evolutionary
joint-tensor) and M28 (multi-input):
* M27 (evolutionary joint-tensor) let the full 6x6 stress-TENSOR cross-PSD
  S_sigmasigma(omega, t) evolve window to window — but that tensor came from a
  SINGLE scalar input process (a rank-1 H S_ff H^H with a scalar drifting-shape
  window a_i^2 W_i(f) on ONE PSD). The critical plane rotated because the tensor
  ORIENTATION drifted as a window swept the modal band; the excitation was still
  one process.
* M28 (multi-input) drove S_sigmasigma from a FULL Hermitian ninput x ninput input
  cross-PSD S_ff (auto-PSDs on the diagonal, coherence gamma_ab + phase theta_ab
  off the diagonal) via the MIMO relation S_sigmasigma = H_sigma S_ff H_sigma^H —
  but S_ff was STATIONARY (the coherence held fixed; M28 composes with M25/M26/M27
  by modulating the stationary multi-input S_sigmasigma with a SCALAR RMS mission
  profile / drifting-shape window, the coherence itself never drifting).
M29 lifts exactly that last assumption: the INPUT COHERENCE MATRIX itself is
time-varying. S_ff(omega, t_i) is a PER-WINDOW schedule of Hermitian input
cross-spectral matrices — the coherence gamma_ab(t_i) and phase theta_ab(t_i)
(and/or the auto-PSDs G_a(t_i)) interpolated across the M26/M27 time-windows (a
start -> end coherence pair, or a per-input mission profile). Each window's S_ff
drives its OWN multi-input stress-tensor cross-PSD S_sigmasigma,i = H_sigma
S_ff(t_i) H_sigma^H, whose 6x6 moment matrices are reduced by the M27 per-window
critical-plane search — the plane / F_np RE-SEARCHED per window AS THE COHERENCE
DRIFTS, so the critical plane genuinely swings between the M28 incoherent-SUM and
coherent-combination limits window to window. This is the natural consumer of BOTH
the M28 multi-input synthesiser AND the M27 per-window tensor Miner-sum + plane
re-search.

THE BUILT-IN REDUCTIONS (exact, asserted).
* SINGLE INPUT (ninput = 1) -> the M27 scalar/tensor EVOLUTIONARY answer EXACTLY:
  a lone input has no off-diagonal coherence, so S_ff,i = [a_i^2 W_i(f) G] and
  S_sigmasigma,i = |H_sigma|^2 a_i^2 W_i(f) G = a_i^2 W_i(f) S_sigmasigma,stat is
  exactly M27's windowed tensor eq. (1). The ninput = 1 path DELEGATES to
  ``joint_evolutionary_fatigue.joint_evolutionary_fatigue_summary`` so it is
  BIT-IDENTICAL to M27.
* SINGLE WINDOW / CONSTANT COHERENCE (nwin = 1, flat unit window, gamma / phase
  not drifting) -> the M28 STATIONARY multi-input answer EXACTLY: one flat unit
  window with a fixed coherence gives S_ff,1 = the M28 stationary S_ff and
  S_sigmasigma,1 = H_sigma S_ff H_sigma^H, reduced by the M28
  ``multi_input_multiaxial_summary``. This case DELEGATES to M28 so it is
  BIT-IDENTICAL.
The whole POINT of M29 over M27/M28 is the case NEITHER covers: a DRIFTING
incoherent -> coherent schedule (gamma sweeping 0 -> 1 across the windows) whose
per-window response variance and critical plane genuinely DRIFT between the M28
incoherent-SUM limit (gamma = 0, S_sigmasigma = sum_a H_a G_a H_a^H) and the
coherent-combination limit (gamma = 1, the effective combined pattern) — the
input coherence itself evolving, not merely the level or spectral shape.

Fortran origin
--------------
There is NONE. ``engine/source/input/freimpl.F`` (the /IMPL reader, re-read line
by line for M16-M28 AND AGAIN for M29 — fetched from raw.githubusercontent.com)
parses only /IMPL/DYNA, /IMPL/BUCKL, /IMPL/DT, /IMPL/NONLIN and /IMPL/ARCL plus
the linear-solver housekeeping — there is no /FATIG, no S-N / Miner branch, no
Dirlik / rainflow / narrow-band estimator, no von-Mises / critical-plane /
stress-tensor cross-PSD machinery, no multi-input / coherence / cross-spectral
matrix, and (as M16-M28 already found) NOTHING non-stationary / evolutionary /
spectrogram / time-varying-coherence of any kind. The sole ``PSD`` token anywhere
in the engine is ``IMUMPSD`` (a MUMPS-solver flag, not a power spectral density) —
re-confirmed for M29 in ``freimpl.F`` and ``engine/source/implicit/imp_solv.F``.
OpenRadioss is a time-domain crash/impact code: the random-vibration MULTI-INPUT
MULTIAXIAL fatigue analysis — stationary multi-input (M28) OR fully evolutionary
multi-input with a time-varying coherence (M29) — is simply not part of the
open-source solver, exactly as M16 found for the real eigensolver and M17-M28 for
the transfer functions, the PSD machinery, the spectral-fatigue estimators and the
multi-input path this module extends.

So M29 does exactly what M16-M28 did: it ports evolutionary multi-input random
fatigue as a clean LIBRARY capability and drives it with a minimal PORT engine
sub-flag (/IMPL/FATIG/MULT/MINPUT/EVOL — the EVOLUTIONARY-COHERENCE analogue of
the M28 multi-input card, composing with /EVOL / /JOINT). Nothing in the M10 direct
integrator, the M16 REAL eigensolver, the M17/M18 superposition, the M19 PSD path,
the M20-M27 fatigue reductions OR the M28 stationary multi-input path is touched:
the evolutionary multi-input path is a NEW, parallel path that CONSUMES the M28
multi-input synthesiser (``input_cross_psd_matrix`` / ``nearest_psd`` /
``stress_tensor_cross_psd_multi`` / ``synthesize_multi_input_stress`` /
``measure_coherence``) and the M27 per-window tensor Miner-sum + plane re-search
(``reduce_window_tensor`` / ``joint_evolutionary_windows``) — all read-only — and
produces its answer ALONGSIDE the M28 stationary multi-input and the M27
single-input evolutionary numbers so a listing shows all three side by side.

Theory — evolutionary multi-input random fatigue (the time-varying coherence)
-----------------------------------------------------------------------------
(Priestley, "Evolutionary spectra and non-stationary processes", J. Roy. Statist.
Soc. B 27, 1965 — the evolutionary spectrum S(omega, t), here the MATRIX /
COHERENCE-valued case S_ff(omega, t) with a time-varying coherence matrix; Newland,
"An Introduction to Random Vibrations, Spectral & Wavelet Analysis", 3rd ed.,
ch. 6-8 — multiple correlated inputs, the coherence function and the matrix
input/output spectral relation, and ch. 5-7 the short-time / windowed spectral
method; Bendat & Piersol, "Random Data", 4th ed., ch. 5-7 — the cross-spectral /
coherence matrix, the time-varying coherence and the MIMO relation S_yy = H S_xx
H^H; Wirsching, Paez & Ortiz 1995 — multi-input random fatigue and the
spectral-representation synthesis of correlated processes; the M21/M23 multiaxial
+ M25/M26/M27 non-stationary / evolutionary + M28 multi-input bases this module
converges.)

THE PER-WINDOW INPUT CROSS-PSD (the time-varying coherence). For ninput jointly
random input processes the input cross-spectral matrix is Hermitian PSD (M28 eq.
(1))

    S_ff(omega, t_i)[a,b] = sqrt( G_a(omega, t_i) G_b(omega, t_i) )
                            gamma_ab(t_i) exp( i theta_ab(t_i) )              (1)

with the coherence gamma_ab(t_i) in [0, 1] and phase theta_ab(t_i) now DRIFTING
window to window — interpolated linearly across the M26/M27 window mid-time
fractions s_i = (i + 1/2)/nwin from a START pair (gamma_0, theta_0) to an END pair
(gamma_1, theta_1), and the per-window auto-PSDs G_a(omega, t_i) = a_i^2 W_i(f)
G_a(omega) carrying the M27 drifting-shape window W_i(f) and RMS level a_i (so M29
composes the coherence drift WITH the M25/M26/M27 level / spectral-shape drift).
Each window's S_ff is projected onto the NEAREST Hermitian PSD matrix (M28
``nearest_psd``, a no-op on a valid matrix) — a drifting coherence MODEL can leave
the PSD band by a rounding margin exactly as the stationary one can.

THE PER-WINDOW STRESS-TENSOR CROSS-PSD (the MIMO map, per window). The structure's
per-window stress-tensor cross-PSD is the M28 matrix triple product with the
window's OWN input matrix (M28 eq. (3))

    S_sigmasigma,i(omega) = H_sigma(omega) S_ff(omega, t_i) H_sigma(omega)^H   (2)

from the per-input Voigt stress FRF column stack H_sigma (nf, 6, ninput). Its 6x6
spectral-MOMENT matrices M_{n,i} = (1/pi) int omega^n S_sigmasigma,i domega (M21)
are a GENUINELY DIFFERENT matrix per window — NOT a shared shape scaled by a
level, and NOT merely a windowed shape, but a DIFFERENT COHERENT COMBINATION of the
per-input stress columns as gamma_ab drifts. In the gamma = 0 (incoherent) window
S_sigmasigma,i = sum_a H_a G_a H_a^H (the M28 incoherent SUM); in the gamma = 1
(coherent) window it is the rank-1 (H v)(H v)^H of the effective combined pattern
(the M28 coherent combination); a partially-coherent window interpolates. So as the
coherence drifts the critical plane of M_{0,i} ROTATES between those limits.

THE PER-WINDOW REDUCTION (plane RE-SEARCHED as the coherence drifts). Each window's
M_{n,i} is reduced EXACTLY as M27 reduces an evolutionary window (M27
``reduce_window_tensor``): the equivalent von Mises trace(Q M_{n,i}), the max-NORMAL
and max-SHEAR critical planes RE-SEARCHED from M_{0,i} (so the plane may ROTATE as
the coherence evolves), and the M23 spectral non-proportionality F_np,i of the
window's critical plane. The window Miner-sum is the M27 duration-weighted linear
accumulation

    D = sum_i (E[D]/T)_i T_i ,   E[D]/T = D / sum_i T_i                       (3)

with the mission time-to-failure T_f = (sum_i T_i) / D. The FORM is identical to
M27's; the difference is that M27's per-window tensor came from ONE input windowed
in shape, whereas M29's comes from the window's OWN time-varying input COHERENCE.

NON-STATIONARY MULTI-INPUT MONTE-CARLO (the independent cross-check). Per-window
BLOCKS of the M28 correlated-input synthesiser (each window's S_ff(t_i)'s per-bin
eigen/Cholesky factor, M28 ``synthesize_multi_input_stress``) are concatenated in
time, so the instantaneous input cross-spectrum tracks S_ff(omega, t). Each window's
stress block (the correlated inputs driven through the stress FRF columns and
summed) is PROJECTED onto that WINDOW's OWN critical plane, rainflow-counted (ASTM
E1049, the M20 counter) and Miner-summed — the time-domain damage the evolutionary
multi-input window spectral estimate approximates. The synthesised inputs' per-window
MEASURED coherence (M28 ``measure_coherence``, Welch averaging) tracks the target
gamma_ab(t_i) — the confirmation that the drifting coherence is faithfully realised.

    DELEGATIONS (the bit-identical reductions).
    * ninput = 1 -> the M27 non-stationary MULTIVARIATE Monte-Carlo
      (``joint_evolutionary_monte_carlo_damage``) on the rank-1 tensor, bit-identical.
    * SINGLE WINDOW / CONSTANT COHERENCE -> the M28 multi-input Monte-Carlo
      (``monte_carlo_multi_input_damage``), bit-identical.

    WINDOW-BOUNDARY RAINFLOW CAVEAT (documented, carried from M25/M26/M27). The
    closed-form window Miner-sum (3) counts each window's cycles INDEPENDENTLY and so
    MISSES cycles that STRADDLE a window boundary — a small documented discrepancy
    that vanishes as the windows grow long relative to the cycle period. The
    Monte-Carlo over the concatenated record is the reference that includes them.

    SHORT-TIME-STATIONARY / WINDOWED-COHERENCE approximation. Each window is treated
    as locally stationary with its OWN input coherence (piecewise-constant in time);
    the coherence is sampled at the window mid-time. The window must be LONG relative
    to the carrier period and SHORT relative to the coherence drift — the standard
    spectrogram trade-off, documented.

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* LIBRARY-FIRST sub-flag (/IMPL/FATIG/MULT/MINPUT/EVOL) — no upstream equivalent,
  exactly as established for M16-M28's PORT cards.
* WINDOWED (short-time) evolutionary coherence, NOT a continuous Wigner-Ville /
  Loeve INSTANTANEOUS coherence-matrix spectrum — DEFERRED, exactly as M26/M27
  deferred the scalar / joint-tensor Wigner-Ville distributions.
* The coherence schedule is a START -> END interpolation (or a per-input mission
  /FUNCT on the auto-PSDs); a FREQUENCY-DEPENDENT drifting coherence gamma_ab(f, t)
  beyond the supported schedules is DEFERRED (the library accepts a full per-window
  coherence stack, but the card exposes the start/end-pair schedule).
* The BASE-ACCELERATION multi-input feed (per-direction participation column stack)
  remains DEFERRED (carried from M28 — the force-pattern feed is used).
* A full NON-GAUSSIAN JOINT-TENSOR evolutionary distribution is DEFERRED; M29
  composes with the M24 non-Gaussian correction on the equivalent scalar but does
  not model a non-Gaussian time-varying-coherence joint-tensor distribution.
* MEAN-STRESS beyond the basic M20/M21 Goodman intercept, CRACK-GROWTH /
  fracture-mechanics fatigue and the COMPLEX-FRF stress recovery remain DEFERRED
  (the unchanged M20-M28 tail).

See PORTING_GUIDE.md roadmap M29.
"""

from __future__ import annotations

import math

import numpy as np

# reuse the M28 multi-input synthesiser + the M27 per-window tensor machinery,
# read-only (the natural consumer relationship — M29 is the convergence of the M28
# multi-input path and the M27 evolutionary joint-tensor path)
from .multi_input_response import (input_cross_psd_matrix,
                                   stress_tensor_cross_psd_multi)
from .joint_evolutionary_fatigue import (joint_evolutionary_windows,
                                         reduce_window_tensor,
                                         _reduce_window_fixed)
from .multiaxial_fatigue import tensor_moment_matrices


# ============================================================================
# The per-window coherence schedule (build item 1a)
# ============================================================================

def coherence_schedule(nwin, gamma0, gamma1=None, phase0=0.0, phase1=None):
    """Build the per-window coherence schedule (gamma_i, phase_i) — theory eq.
    (1)'s time-varying coherence — interpolated LINEARLY across the M26/M27 window
    mid-time fractions s_i = (i + 1/2)/nwin from a START pair (gamma0, phase0) to
    an END pair (gamma1, phase1).

    ``gamma0`` / ``gamma1`` are either scalars (a coherence applied to every
    off-diagonal input pair) or full (ninput, ninput) coherence matrices; ``phase0``
    / ``phase1`` likewise (radians). ``gamma1`` / ``phase1`` default to the start
    value (a CONSTANT coherence — the M28 stationary special case). Returns a list
    of ``nwin`` tuples (gamma_i, phase_i) — the mid-time coherence of each window,
    the SAME parameterisation ``joint_evolutionary_windows`` uses for the
    drifting-shape window so the coherence and the spectral-shape drift stay
    consistent in time.

    NumPy broadcasting handles scalars and matrices uniformly: the interpolation
    g0 + s_i (g1 - g0) works element-wise whether g0/g1 are 0-d scalars or (n, n)
    arrays."""
    if nwin <= 0:
        raise ValueError("coherence_schedule needs at least one window.")
    g1 = gamma0 if gamma1 is None else gamma1
    p1 = phase0 if phase1 is None else phase1
    g0 = np.asarray(gamma0, dtype=float)
    g1 = np.asarray(g1, dtype=float)
    p0 = np.asarray(phase0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    # equal-duration window mid-time fractions — IDENTICAL to the M27
    # joint_evolutionary_windows parameterisation (so coherence + shape drift align)
    s = np.array([0.5]) if nwin == 1 else (np.arange(nwin) + 0.5) / nwin
    return [(g0 + si * (g1 - g0), p0 + si * (p1 - p0)) for si in s]


def _constant_coherence(gamma0, gamma1, phase0, phase1):
    """True when the coherence schedule does NOT drift (gamma1 / phase1 absent or
    equal to gamma0 / phase0) — the M28 stationary-coherence special case used to
    trigger the bit-identical M28 delegation."""
    gconst = gamma1 is None or np.array_equal(np.asarray(gamma1, dtype=float),
                                              np.asarray(gamma0, dtype=float))
    pconst = phase1 is None or np.array_equal(np.asarray(phase1, dtype=float),
                                              np.asarray(phase0, dtype=float))
    return bool(gconst and pconst)


def _flat_unit_window(fc, bw, scales, nwin):
    """True when the drifting-SHAPE window is a flat, unit-level, single window (no
    M25 level drift, no M26/M27 shape drift) — with a constant coherence this is the
    M28 stationary multi-input special case."""
    bw0, bw1 = (bw if isinstance(bw, (tuple, list, np.ndarray)) else (bw, bw))
    flat = (bw0 is None or bw0 <= 0.0) and (bw1 is None or bw1 <= 0.0)
    unit = scales is None or np.allclose(np.asarray(scales, dtype=float), 1.0)
    return bool(nwin == 1 and flat and unit)


# ============================================================================
# The per-window input cross-PSD schedule S_ff(omega, t_i) (build item 1a)
# ============================================================================

def evolutionary_input_windows(freqs, auto_psds, durations, gamma0, gamma1=None,
                               phase0=0.0, phase1=None, fc=0.0, bw=0.0,
                               scales=None):
    """Build the per-window schedule of Hermitian input cross-spectral matrices
    S_ff(omega, t_i) (theory eq. (1)): partition the mission into ``len(durations)``
    windows and, per window i, assemble

        G_a(omega, t_i) = a_i^2 W_i(f) G_a(omega)      (the M27 windowed auto-PSD)
        S_ff(omega, t_i) = input_cross_psd_matrix(G_i, gamma_i, phase_i)

    with the coherence (gamma_i, phase_i) from ``coherence_schedule`` (drifting
    window to window) and the drifting-shape window W_i(f) / RMS level a_i from the
    M27 ``joint_evolutionary_windows`` (the SAME schedule /EVOL / /JOINT use). Each
    window's S_ff is projected onto the nearest Hermitian PSD matrix (M28
    ``input_cross_psd_matrix`` calls ``nearest_psd``).

    ``freqs`` (nf,) the sweep grid [Hz], ``auto_psds`` (nf, ninput) the stationary
    per-input auto-PSDs G_a(omega), ``durations`` the per-window times; ``gamma0/1``,
    ``phase0/1`` the coherence start/end pair; ``fc``, ``bw``, ``scales`` the
    drifting-shape schedule (as ``joint_evolutionary_windows`` — a scalar ``bw`` <= 0
    means a flat window, no shape drift).

    Returns a list of window dicts {W, fc, bw, scale, duration, gamma, phase, G,
    Sff} where ``Sff`` is the (nf, ninput, ninput) Hermitian PSD input cross-PSD of
    the window and ``G`` its (nf, ninput) windowed auto-PSDs."""
    f = np.asarray(freqs, dtype=float)
    G0 = np.clip(np.asarray(auto_psds, dtype=float), 0.0, None)
    if G0.ndim == 1:
        G0 = G0[:, None]
    # the M27 drifting-shape window schedule (W_i, fc_i, bw_i, scale_i, duration_i)
    shape_wins = joint_evolutionary_windows(f, durations, fc, bw, scales)
    nwin = len(shape_wins)
    # the drifting coherence schedule (gamma_i, phase_i) — the M29 addition
    coh = coherence_schedule(nwin, gamma0, gamma1, phase0, phase1)
    out = []
    for w, (gi, pi) in zip(shape_wins, coh):
        # the M27 windowed / scaled auto-PSDs (a_i^2 W_i(f) applied to every input —
        # a per-frequency scalar, so it commutes with the coherence assembly)
        Gi = (w["scale"] ** 2) * w["W"][:, None] * G0            # (nf, ninput)
        mi = input_cross_psd_matrix(Gi, gamma=gi, phase=pi)
        out.append({"W": w["W"], "fc": w["fc"], "bw": w["bw"],
                    "scale": w["scale"], "duration": w["duration"],
                    "gamma": gi, "phase": pi, "G": Gi, "Sff": mi["Sff"],
                    "projected": mi["projected"], "min_eig": mi["min_eig"]})
    return out


def _representative_coherence(gamma):
    """A single scalar coherence representing a (possibly matrix) per-window gamma —
    the mean of the off-diagonal magnitudes (or the scalar itself) — used only for
    the coherence-DRIFT reporting diagnostic (the physics uses the full matrix)."""
    g = np.asarray(gamma, dtype=float)
    if g.ndim == 0:
        return float(g)
    n = g.shape[0]
    if n < 2:
        return 1.0
    iu = np.triu_indices(n, 1)
    return float(np.mean(np.abs(g[iu]))) if iu[0].size else 1.0


# ============================================================================
# The evolutionary multi-input window Miner-sum (build item 1a + 1b)
# ============================================================================

def evolutionary_multi_input_summary(omega, Hcols, auto_psds, durations, m, C,
                                     gamma0, gamma1=None, phase0=0.0, phase1=None,
                                     fc=0.0, bw=0.0, scales=None, mean_stress=0.0,
                                     ultimate=0.0, naz=24, npol=13, drift=True):
    """The FULLY EVOLUTIONARY MULTI-INPUT window Miner-sum damage (theory eq. (3))
    for ONE element: build the per-window input cross-PSD S_ff(omega, t_i) (the
    coherence DRIFTING window to window), form the per-window multi-input
    stress-tensor cross-PSD S_sigmasigma,i = H_sigma S_ff(t_i) H_sigma^H (eq. (2)),
    recompute the per-window 6x6 moment matrices M_{n,i}, reduce EACH window with the
    M27 critical-plane machinery — the plane / F_np RE-SEARCHED from the window's OWN
    tensor (so the plane may ROTATE as the coherence evolves) — and Palmgren-Miner SUM
    the per-window multiaxial damages duration-weighted.

    ``omega`` (nf,) angular grid, ``Hcols`` (nf, 6, ninput) the element's per-input
    Voigt stress FRF column stack, ``auto_psds`` (nf, ninput) the stationary per-input
    auto-PSDs, ``durations`` the per-window times, ``gamma0/1`` / ``phase0/1`` the
    coherence start/end pair, ``fc`` / ``bw`` / ``scales`` the drifting-shape schedule,
    ``m`` / ``C`` the S-N law.

    ``drift`` — if True (the M29 default) the critical plane is RE-SEARCHED per window
    (it may ROTATE as the coherence drifts — the whole point); if False the plane is
    FIXED at the mission-averaged-coherence stationary critical plane and only the
    windowed scalar drifts (the M26/M27 fixed-reduction check).

    Two BIT-IDENTICAL delegations (the exact-reduction contract):
    * ninput = 1 -> DELEGATE to the M27 ``joint_evolutionary_fatigue_summary`` on the
      rank-1 tensor S_sigmasigma = |H_sigma|^2 G (the single-input evolutionary answer,
      bit-identical);
    * nwin = 1 AND constant coherence AND flat unit window -> DELEGATE to the M28
      ``multi_input_multiaxial_summary`` on the stationary S_sigmasigma (the stationary
      multi-input answer, bit-identical).

    Returns a dict with, for each reduction (von_mises / normal_plane / shear_plane),
    the window Miner-sum {damage_rate, life, damage, total_time}; plus ``windows`` (the
    per-window coherence / critical-plane / RMS drift), ``coherence_drift`` (the max -
    min representative coherence across the windows), ``plane_rotation_deg`` (the max
    max-normal / max-shear plane swing), ``constant_coherence`` / ``constant_shape``,
    ``delegated`` (None / 'm27_single_input' / 'm28_single_window'), ``ninput``,
    ``nwin`` and the promoted von-Mises ``damage_rate`` / ``life``."""
    omega = np.asarray(omega, dtype=float)
    Hcols = np.asarray(Hcols)
    G0 = np.clip(np.asarray(auto_psds, dtype=float), 0.0, None)
    if G0.ndim == 1:
        G0 = G0[:, None]
    ninput = Hcols.shape[2]
    T = np.asarray(durations, dtype=float).ravel()
    nwin = T.size
    freqs = omega / (2.0 * np.pi)

    # ----- DELEGATION 1: single input -> the M27 evolutionary answer EXACTLY ------
    if ninput == 1:
        from . import multiaxial_fatigue as mf
        from . import joint_evolutionary_fatigue as jf
        # the rank-1 stationary tensor |H_sigma|^2 G (delegates internally to the M21
        # rank-1 path so it is bit-identical to the single-input tensor)
        Scross = stress_tensor_cross_psd_multi(Hcols, G0[:, :, None])
        summ = jf.joint_evolutionary_fatigue_summary(
            omega, Scross, durations, fc=fc, bw=bw, m=m, C=C, scales=scales,
            mean_stress=mean_stress, ultimate=ultimate, naz=naz, npol=npol,
            drift=drift)
        summ = dict(summ)
        summ.update({"method": "evolutionary_multi_input", "ninput": 1,
                     "delegated": "m27_single_input", "coherence_drift": 0.0,
                     "constant_coherence": True})
        return summ

    # ----- DELEGATION 2: single window + constant coherence -> M28 EXACTLY --------
    const_coh = _constant_coherence(gamma0, gamma1, phase0, phase1)
    if const_coh and _flat_unit_window(fc, bw, scales, nwin):
        from . import multi_input_response as mir
        Sff = input_cross_psd_matrix(G0, gamma=gamma0, phase=phase0)["Sff"]
        Scross = stress_tensor_cross_psd_multi(Hcols, Sff)
        m28 = mir.multi_input_multiaxial_summary(
            Scross, omega, m, C, mean_stress=mean_stress, ultimate=ultimate,
            naz=naz, npol=npol)
        return _wrap_m28_single_window(m28, Scross, T, gamma0, ninput)

    # ----- GENERAL DRIFT PATH: per-window S_ff -> S_sigmasigma -> reduce ----------
    windows = evolutionary_input_windows(
        freqs, G0, durations, gamma0, gamma1, phase0, phase1, fc, bw, scales)

    # the mission-averaged-coherence STATIONARY reduction — the reference the
    # reporting compares against and the FIXED plane for the drift=False branch
    g_mean, p_mean = _mean_coherence(windows, gamma0, gamma1, phase0, phase1)
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
    cohvals = []
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
        cohvals.append(_representative_coherence(w["gamma"]))
        wout.append({
            "fc": w["fc"], "bw": w["bw"], "scale": w["scale"], "duration": Ti,
            "gamma": _representative_coherence(w["gamma"]),
            "gamma_mat": np.asarray(w["gamma"], dtype=float),
            "projected": bool(w["projected"]), "min_eig": float(w["min_eig"]),
            "normal_n": np.asarray(red["normal_plane"]["normal"], dtype=float),
            "shear_n": np.asarray(red["shear_plane"]["normal"], dtype=float),
            "normal_proj": np.asarray(red["normal_plane"]["proj"], dtype=float),
            "shear_proj": np.asarray(red["shear_plane"]["proj"], dtype=float),
            "F_np": float(red["F_np"]), "sigma_vm": float(red["sigma_vm"]),
            "vm_rate": float(red["von_mises"]["damage_rate"]),
            "normal_rate": float(red["normal_plane"]["damage_rate"]),
            "shear_rate": float(red["shear_plane"]["damage_rate"])})
        # the normalised M_0 shape (unit m0^vm) — the constant-shape detector
        from .multiaxial_fatigue import equivalent_vonmises_moments
        s0 = float(equivalent_vonmises_moments(Mi)[0])
        shapes.append(Mi[0] / s0 if s0 > 0 else Mi[0])

    # constant tensor shape: every window's normalised M_0 is the same (neither the
    # coherence nor the spectral shape drifts the tensor orientation)
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
    coh_drift = (max(cohvals) - min(cohvals)) if cohvals else 0.0

    out = {"method": "evolutionary_multi_input", "ninput": int(ninput),
           "nwin": len(windows), "drift": bool(drift), "delegated": None,
           "constant_shape": bool(const_shape),
           "constant_coherence": bool(const_coh),
           "plane_rotation_deg": float(rot), "fnp_drift": float(fnp_drift),
           "coherence_drift": float(coh_drift), "windows": wout,
           "stationary": stat, "total_time": Ttot}
    for k in reductions:
        dr = D[k] / Ttot if Ttot > 0 else 0.0
        out[k] = {"damage": D[k], "total_time": Ttot, "damage_rate": dr,
                  "life": (math.inf if dr <= 0.0 else 1.0 / dr)}
    vm = out["von_mises"]
    out.update({"damage_rate": vm["damage_rate"], "life": vm["life"]})
    return out


def _mean_coherence(windows, gamma0, gamma1, phase0, phase1):
    """The mission-averaged coherence / phase (the mean over the per-window
    schedule) — the STATIONARY reference coherence for the reporting baseline and the
    fixed plane of the drift=False branch. Returns (gamma_mean, phase_mean) as
    scalars or matrices (matching the input shape)."""
    g0 = np.asarray(gamma0, dtype=float)
    g1 = g0 if gamma1 is None else np.asarray(gamma1, dtype=float)
    p0 = np.asarray(phase0, dtype=float)
    p1 = p0 if phase1 is None else np.asarray(phase1, dtype=float)
    return 0.5 * (g0 + g1), 0.5 * (p0 + p1)


def _wrap_m28_single_window(m28, Scross, durations, gamma0, ninput):
    """Wrap the M28 ``multi_input_multiaxial_summary`` output into the M29 window
    Miner-sum dict shape for the single-window / constant-coherence delegation — so
    the caller sees the SAME structure whether the answer came from the M28
    delegation or the general drift path. The single window carries the whole
    mission time; the damage rate is the M28 stationary Dirlik rate, BIT-IDENTICAL to
    the M28 answer (no re-reduction)."""
    T = np.asarray(durations, dtype=float).ravel()
    Ttot = float(T.sum()) if T.size else 1.0
    reductions = ("von_mises", "normal_plane", "shear_plane")
    out = {"method": "evolutionary_multi_input", "ninput": int(ninput),
           "nwin": 1, "drift": True,
           "delegated": "m28_single_window", "constant_shape": True,
           "constant_coherence": True, "plane_rotation_deg": 0.0,
           "fnp_drift": 0.0, "coherence_drift": 0.0,
           "stationary": None, "total_time": Ttot}
    wout = []
    for k in reductions:
        dr = float(m28[k]["summary"]["dirlik"]["damage_rate"])
        out[k] = {"damage": dr * Ttot, "total_time": Ttot, "damage_rate": dr,
                  "life": (math.inf if dr <= 0.0 else 1.0 / dr)}
    sp = m28["shear_plane"]
    npl = m28["normal_plane"]
    wout.append({
        "fc": 0.0, "bw": 0.0, "scale": 1.0, "duration": Ttot,
        "gamma": _representative_coherence(gamma0),
        "gamma_mat": np.asarray(gamma0, dtype=float),
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
        "shear_rate": out["shear_plane"]["damage_rate"]})
    out["windows"] = wout
    out["m28"] = m28              # the full M28 summary (byte-identical) for reuse
    vm = out["von_mises"]
    out.update({"damage_rate": vm["damage_rate"], "life": vm["life"]})
    return out


# ============================================================================
# Non-stationary MULTI-INPUT multivariate Monte-Carlo cross-check (build item 2)
# ============================================================================

def synthesize_evolutionary_multi_input_stress(freqs, windows, Hcols, seed,
                                               fs=None):
    """Synthesise a NON-STATIONARY MULTI-INPUT stress-tensor history whose
    instantaneous 6x6 cross-spectrum tracks the evolutionary multi-input
    S_sigmasigma(omega, t) (theory "NON-STATIONARY MULTI-INPUT MONTE-CARLO"):
    per-window BLOCKS of the M28 correlated-input synthesiser
    (``synthesize_multi_input_stress`` — the window's OWN S_ff(t_i)'s per-bin
    eigen/Cholesky factor driven through the stress FRF columns and summed),
    concatenated in time so the instantaneous input coherence tracks gamma_ab(t).

    ``freqs`` (nf,) the sweep grid, ``windows`` the ``evolutionary_input_windows``
    schedule, ``Hcols`` (nf, 6, ninput) the element's stress FRF columns. All windows
    share ONE sampling rate ``fs`` (default 8 x fmax, the M28 convention). Returns
    (t, X, info) with ``X`` (nt, 6) the six Voigt stress components and ``info`` =
    {edges (window-boundary times), fs}."""
    from . import multi_input_fatigue as mif
    f = np.asarray(freqs, dtype=float)
    if fs is None:
        fs = 8.0 * float(f.max())
    T = np.array([w["duration"] for w in windows], dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(T)])
    chunks = []
    for i, w in enumerate(windows):
        # window i's OWN input cross-PSD S_ff(t_i) -> its correlated stress block
        _t, Xi = mif.synthesize_multi_input_stress(
            f, w["Sff"], Hcols, w["duration"], int(seed) + i, fs=fs)
        chunks.append(Xi)
    X = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 6))
    t = np.arange(X.shape[0]) / fs
    return t, X, {"edges": edges, "fs": fs}


def evolutionary_multi_input_monte_carlo_damage(
        omega, Hcols, auto_psds, durations, m, C, seed, gamma0, gamma1=None,
        phase0=0.0, phase1=None, fc=0.0, bw=0.0, scales=None, fs=None,
        mean_stress=0.0, ultimate=0.0, naz=24, npol=13, reduction="shear_plane",
        summary=None, measure=False):
    """The TIME-DOMAIN evolutionary multi-input damage rate by NON-STATIONARY
    MULTI-INPUT multivariate Monte-Carlo (theory "NON-STATIONARY MULTI-INPUT
    MONTE-CARLO"): synthesise the multi-input history
    (``synthesize_evolutionary_multi_input_stress`` — per-window blocks of the M28
    correlated-input synthesiser), PROJECT each window's block onto that WINDOW's OWN
    critical plane (the windowed critical-plane path — the plane re-searched as the
    coherence drifts), rainflow-count each window (ASTM E1049, the M20 counter) and
    Palmgren-Miner SUM.

    Two BIT-IDENTICAL delegations:
    * ninput = 1 -> DELEGATE to the M27 ``joint_evolutionary_monte_carlo_damage`` on
      the rank-1 tensor (the single-input evolutionary MC, bit-identical);
    * nwin = 1 AND constant coherence AND flat unit window -> DELEGATE to the M28
      ``monte_carlo_multi_input_damage`` on the stationary S_ff (the stationary
      multi-input MC, bit-identical).

    ``reduction`` — which critical plane to project onto ("shear_plane" default, or
    "normal_plane"). ``summary`` optionally supplies a precomputed
    ``evolutionary_multi_input_summary`` so the per-window planes are shared.
    ``measure`` — if True, ALSO measure each window's synthesised-input coherence
    (M28 ``measure_coherence``) and return it as ``window_gamma`` to confirm it
    tracks the target gamma_ab(t_i).

    Returns the M20 Monte-Carlo dict shape plus per-window ``window_rms`` /
    ``window_nu0`` and ``delegated``. In the single-window / constant-coherence limit
    (or single-input limit) reduces EXACTLY to the M28 (or M27) Monte-Carlo."""
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

    # ----- DELEGATION 1: single input -> the M27 evolutionary MC EXACTLY ----------
    if ninput == 1:
        from . import joint_evolutionary_fatigue as jf
        Scross = stress_tensor_cross_psd_multi(Hcols, G0[:, :, None])
        out = jf.joint_evolutionary_monte_carlo_damage(
            omega, Scross, durations, fc=fc, bw=bw, m=m, C=C, seed=seed,
            scales=scales, fs=fs, mean_stress=mean_stress, ultimate=ultimate,
            naz=naz, npol=npol, reduction=reduction)
        out = dict(out)
        out["delegated"] = "m27_single_input"
        return out

    # ----- DELEGATION 2: single window + constant coherence -> M28 MC EXACTLY -----
    const_coh = _constant_coherence(gamma0, gamma1, phase0, phase1)
    if const_coh and _flat_unit_window(fc, bw, scales, nwin):
        from . import multi_input_fatigue as mif
        Sff = input_cross_psd_matrix(G0, gamma=gamma0, phase=phase0)["Sff"]
        # the projection onto the (stationary) critical plane; reuse the summary's
        # plane if given, else derive it from the M28 stationary reduction
        if summary is not None and summary.get("windows"):
            proj = np.asarray(summary["windows"][0][projkey], dtype=float)
        else:
            from . import multi_input_response as mir
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
        return out

    # ----- GENERAL DRIFT PATH: per-window blocks, per-window plane projection -----
    if summary is None:
        summary = evolutionary_multi_input_summary(
            omega, Hcols, auto_psds, durations, m, C, gamma0, gamma1, phase0,
            phase1, fc, bw, scales, mean_stress, ultimate, naz, npol, drift=True)
    windows = evolutionary_input_windows(
        freqs, G0, durations, gamma0, gamma1, phase0, phase1, fc, bw, scales)
    t, X, info = synthesize_evolutionary_multi_input_stress(
        freqs, windows, Hcols, seed, fs=fs)
    edges = info["edges"]
    fs = info["fs"]
    projs = [np.asarray(w[projkey], dtype=float) for w in summary["windows"]]

    D = 0.0
    ncyc = 0.0
    window_rms = []
    window_nu0 = []
    window_gamma = []
    for i in range(len(windows)):
        sel = (t >= edges[i]) & (t < edges[i + 1])
        seg = X[sel]
        if seg.shape[0] < 2:
            window_rms.append(0.0)
            window_nu0.append(0.0)
            window_gamma.append(np.nan)
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
            window_gamma.append(_measure_window_coherence(freqs, windows[i],
                                                          Hcols, seed + i, fs))

    Ttot = float(edges[-1]) if edges.size else float(np.sum(durations))
    dr = D / Ttot if Ttot > 0 else 0.0
    nu = ncyc / Ttot if Ttot > 0 else 0.0
    tf, s_eq = sf.life_and_equivalent(dr, nu, m, Ceff)
    out = {"method": "evolutionary_multi_input_monte_carlo", "damage_rate": dr,
           "life": tf, "s_eq": s_eq, "ncycles": ncyc, "duration": Ttot,
           "reduction": reduction, "delegated": None,
           "window_rms": np.asarray(window_rms),
           "window_nu0": np.asarray(window_nu0)}
    if measure:
        out["window_gamma"] = np.asarray(window_gamma)
    return out


def _measure_window_coherence(freqs, window, Hcols, seed, fs):
    """Measure ONE window's synthesised-input coherence (the representative
    off-diagonal, M28 ``measure_coherence`` by Welch averaging) — the confirmation
    that the drifting input coherence gamma_ab(t_i) is faithfully realised in the
    time-domain synthesis. Synthesises the window's correlated INPUT histories (long
    enough for the Welch average), measures the coherence and returns the band-mean
    representative off-diagonal sqrt(coh) (so it compares to the target gamma)."""
    from . import multi_input_fatigue as mif
    f = np.asarray(freqs, dtype=float)
    fmax = float(f.max())
    # a dedicated long record for the coherence estimate (the segment used in the
    # damage Miner-sum may be too short for a stable Welch average)
    dur = max(window["duration"], 4000.0 / max(fmax, 1.0))
    _t, F = mif.synthesize_multi_input_forces(f, window["Sff"], dur, int(seed),
                                              fs=fs)
    fsr = 1.0 / (_t[1] - _t[0]) if _t.size > 1 else fs
    nper = min(1024, max(64, F.shape[0] // 8))
    fc, coh = mif.measure_coherence(F, fsr, nperseg=nper)
    band = (fc > 0.05 * fmax) & (fc < 0.9 * fmax)
    if not np.any(band):
        band = slice(None)
    # representative off-diagonal (0,1) coherence -> sqrt to compare to gamma
    return float(np.sqrt(max(coh[band, 0, 1].mean(), 0.0)))

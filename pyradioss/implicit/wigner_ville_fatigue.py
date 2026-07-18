"""
Continuous Wigner-Ville / Loeve INSTANTANEOUS time-frequency spectrum — M31: the
frequency-domain fatigue-damage estimate of a random-vibration response whose
spectral SHAPE varies with time, computed from a CONTINUOUS instantaneous spectrum
S_WV(omega, t) (a bilinear time-frequency distribution) rather than the M26-M30
SHORT-TIME WINDOWED SPECTROGRAM, reduced by the M20-M27 estimator family AT EACH
INSTANT and Palmgren-Miner INTEGRATED over time (an integral, not a per-window sum),
cross-validated against the M25/M27/M29/M30 non-stationary Monte-Carlo.

WHERE M31 SITS RELATIVE TO M26-M30. M26 (scalar), M27 (6x6 joint tensor), M29
(time-varying coherence matrix) and M30 (frequency-AND-time-varying coherence
matrix) all model the non-stationary load with a SHORT-TIME WINDOWED SPECTROGRAM:
partition the mission into ``nwin`` short time-WINDOWS, treat each window as locally
stationary with its OWN complete PSD / tensor cross-PSD S_i(omega), reduce PER
WINDOW and Palmgren-Miner SUM the window damages D = sum_i (dD/dt)_i T_i. That
sampling incurs TWO documented costs the milestones each carried:
  * the SHORT-TIME-STATIONARY trade-off — the window must be LONG relative to the
    carrier period yet SHORT relative to the shape drift, a resolution compromise;
  * the WINDOW-BOUNDARY RAINFLOW CAVEAT — the closed-form window Miner-SUM counts
    each window's cycles INDEPENDENTLY and MISSES cycles that STRADDLE a window
    boundary (a peak in one window's shape paired with a valley in the next), a
    small discrepancy the Monte-Carlo (which rainflows the WHOLE concatenated
    record) does include.
M31 lifts the whole M26 -> M27 -> M29 -> M30 windowed line to a CONTINUOUS
instantaneous spectrum: the drifting shape fc(t) / bw(t) / a(t) (and the M29/M30
coherence gamma_ab(f, t)) is evaluated CONTINUOUSLY in t on a fine instant grid, the
reduction is performed AT EACH INSTANT (the critical plane / F_np drifting
CONTINUOUSLY, not window to window), and the damage is Palmgren-Miner INTEGRATED
D = integral (dD/dt)(t) dt (a Riemann quadrature over the fine grid, of which the
coarse windowed sum is the nwin-point midpoint approximation). Refining the grid
removes the short-time-stationary resolution compromise AND shrinks the
window-boundary caveat (adjacent fine instants carry near-identical shapes, so the
straddling-cycle mismatch at each of the many fine boundaries is far smaller than at
the few coarse window boundaries), so the continuous integral tracks the Monte-Carlo
BETTER than the windowed sum did.

THE M31 <-> M26/M27/M29/M30 REDUCTION (built in, exact). The windowed spectrogram is
EXACTLY the long-window / coarsest-grid / un-smoothed limit of the continuous
instantaneous spectrum: with the grid refinement factor ``refine`` = 1 (one instant
per window) and the smoothing width ``smooth`` = 0, the fine instant grid IS the
M26/M27/M29/M30 window grid (same window mid-time fractions s_i = (i + 1/2)/nwin,
same durations T_i, same drifting Gaussian window W_i(f)), so the continuous
Miner-INTEGRAL sum_j (dD/dt)_j Delta t_j collapses to the window Miner-SUM
sum_i (dD/dt)_i T_i BYTE-IDENTICALLY. M31 therefore DELEGATES to the M26/M27/M29/M30
summaries in that windowed limit (guaranteeing byte-identity, exactly as M27
delegated to M26, M29 to M28, M30 to M29), and only when refine > 1 or smooth > 0
does it evaluate the genuinely CONTINUOUS instantaneous spectrum. A STATIONARY
process (no drift) recovers the stationary PSD at EVERY instant EXACTLY.

Fortran origin
--------------
There is NONE. ``engine/source/input/freimpl.F`` (the /IMPL reader, re-read line by
line for M16-M30 AND AGAIN for M31, fetched from raw.githubusercontent.com) parses
only /IMPL/DYNA (the DIRECT Newmark/HHT integrator), /IMPL/BUCKL, /IMPL/DT,
/IMPL/NONLIN and /IMPL/ARCL plus the linear-solver housekeeping — there is no
/FATIG, no S-N / Miner branch, no PSD / spectral / random-vibration path, and (as
M16-M30 already found, line by line) NOTHING time-frequency / Wigner-Ville / Loeve /
Cohen-class / instantaneous-spectrum / evolutionary of any kind. The sole ``PSD``
token in the whole file is still ``IMUMPSD`` (line 269, ``IF (ISOLV==3)
IMUMPSD=L_LIM``), a MUMPS-solver flag, echoed at ``engine/source/implicit/
imp_solv.F`` line 1796 (``IF (IMUMPSD == 0) IMUMPSD = 1``) — not a power spectral
density. OpenRadioss is a time-domain crash/impact code: the random-vibration
fatigue analysis — stationary (M20-M24), separable non-stationary (M25),
evolutionary / non-separable windowed spectrogram (M26-M30) OR the CONTINUOUS
Wigner-Ville instantaneous spectrum (M31) — is simply not part of the open-source
solver, exactly as every spectral milestone since M16 recorded.

So M31 does exactly what M16-M30 did: it ports the continuous instantaneous
time-frequency fatigue distribution as a clean LIBRARY capability EXTENDING the
M26-M30 evolutionary windowed line, and drives it with a minimal PORT engine
sub-flag (/IMPL/FATIG/.../WVILLE, composing with /EVOL / /JOINT / /MINPUT / /FCOH,
plus a smoothing-kernel / cross-term-control parameter). Nothing in the M10 direct
integrator, the M16 eigensolver, the M17/M18 superposition, the M19 PSD path, the
M20-M27 fatigue reductions OR the M26-M30 windowed evolutionary paths is touched:
the continuous Wigner-Ville path is a NEW, parallel path that CONSUMES the M20-M27
estimators and the M26-M30 windowed machinery (read-only) and produces its answer
ALONGSIDE the M26/M27/M29/M30 windowed numbers — the windowed spectrogram stays
byte-identical (it is EXACTLY the long-window / heavily-smoothed limit, asserted).

Theory — the Wigner-Ville spectrum of a non-stationary random process
--------------------------------------------------------------------
(Wigner, "On the quantum correction for thermodynamic equilibrium", Phys. Rev. 40,
1932 — the Wigner distribution, the original quasi-probability time-frequency /
phase-space density; Ville, "Theorie et applications de la notion de signal
analytique", Cables et Transmission 2A, 1948 — the Wigner-Ville distribution of a
signal as the Fourier transform of its instantaneous autocorrelation; Loeve,
"Probability Theory" — the harmonizable process and its DUAL-frequency (Loeve)
spectrum, of which the Wigner-Ville spectrum is the time-frequency rotation; Mark,
"Spectral analysis of the convolution and filtering of non-stationary stochastic
processes", J. Sound Vib. 11, 1970 — the Wigner-Ville spectrum W_x(t, omega) =
integral R_x(t + tau/2, t - tau/2) exp(-i omega tau) dtau of a non-stationary random
process, the Fourier transform of the TIME-VARYING autocorrelation; Martin &
Flandrin, "Wigner-Ville spectral analysis of nonstationary processes", IEEE Trans.
ASSP 33, 1985 — the estimation theory and the pseudo / smoothed-pseudo-WVD; Cohen,
"Time-frequency distributions — a review", Proc. IEEE 77, 1989 — the CLASS of
bilinear time-frequency distributions parameterised by a kernel Phi(theta, tau) and
the CROSS-TERM smoothing that maps between members (the spectrogram, the WVD, the
smoothed-pseudo-WVD); Priestley, "Evolutionary spectra and non-stationary
processes", J. Roy. Statist. Soc. B 27, 1965 — the evolutionary spectrum S(omega, t)
the windowed spectrogram approximates and M31 makes continuous; the M26/M27/M29/M30
windowed base this module lifts.)

THE WIGNER-VILLE SPECTRUM. For a non-stationary random process x(t) with
time-varying autocorrelation R_x(t1, t2) = E[x(t1) x*(t2)], the Wigner-Ville
spectrum is the Fourier transform of the LOCAL autocorrelation about the instant t:

    W_x(t, omega) = integral R_x(t + tau/2, t - tau/2) exp(-i omega tau) dtau     (1)

— a CONTINUOUS instantaneous power spectrum, the bilinear time-frequency
distribution whose TIME MARGINAL integral W_x(t, omega) domega / (2 pi) = R_x(t, t) =
E[|x(t)|^2] is the instantaneous power and whose (time-integrated) FREQUENCY MARGINAL
recovers the average PSD. For an oscillatory / evolutionary process (Priestley) with
slowly-varying amplitude A(omega, t), W_x(t, omega) coincides (to first order in the
drift rate) with the evolutionary spectrum S(omega, t) = |A(omega, t)|^2 mu'(omega)
— so for the drifting-shape load M26-M30 model, S_WV(f, t) = a(t)^2 W(f, t) S_0(f)
with W(f, t) = exp(-(f - f_c(t))^2 / (2 b(t)^2)) evaluated CONTINUOUSLY in t. M31
builds exactly this continuous S_WV(f, t), reduces it at each instant and
Miner-integrates.

THE COHEN-CLASS CROSS-TERM SMOOTHING (smoothed-pseudo-WVD). The raw bilinear WVD (1)
of a MULTI-COMPONENT process carries CROSS-TERMS — spurious oscillatory interference
half-way between genuine time-frequency components — because the distribution is
QUADRATIC. Cohen's class suppresses them by CONVOLVING the WVD with a smoothing
kernel; the smoothed-pseudo-WVD applies a separable time-smoothing g(t) and
frequency-smoothing h(tau). Heavier smoothing suppresses more cross-term energy at
the cost of time-frequency resolution, and in the LONG-window / heavily-smoothed
limit the smoothed-pseudo-WVD becomes the SPECTROGRAM (Cohen 1989, Flandrin). M31
exposes a tunable TIME-smoothing width ``smooth`` (a normalised Gaussian kernel over
the mission-fraction axis): ``smooth`` = 0 is the raw instantaneous spectrum (full
resolution), increasing ``smooth`` applies Cohen-class cross-term suppression, and in
the heavily-smoothed limit every instant collapses to the mission-AVERAGED
(stationary) spectrum — the frequency marginal. Because the drifting-shape
evolutionary spectrum used here is a SINGLE-component (unimodal, slowly-varying)
model rather than a multi-component analytic signal, its raw WVD is already
cross-term-free; the smoothing is provided as the documented, tunable Cohen-class
control the theory names (and the mechanism by which the continuous spectrum
reproduces the windowed spectrogram in the coarse-grid limit).

THE INSTANTANEOUS EFFECTIVE WINDOW (the implementation seam). Because the smoothing
is LINEAR and S_0(f) (the recovered stationary stress PSD / tensor cross-PSD) is
fixed, smoothing the instantaneous spectrum in time is identical to smoothing the
per-instant scalar window a(t)^2 W(f, t): the smoothed instantaneous spectrum is
S_smooth(f, t_j) = [sum_k g_jk a_k^2 W_k(f)] S_0(f) = W_eff,j(f) S_0(f) with the
EFFECTIVE WINDOW W_eff,j(f) = sum_k g_jk a_k^2 W_k(f). The whole M31 machinery is
therefore built on ONE primitive — the fine-grid effective windows W_eff,j(f) — fed
to the SCALAR (M26), the 6x6 TENSOR (M27) and the MULTI-INPUT (M29/M30) reductions
UNCHANGED (each a per-frequency scalar multiplier that commutes with the linear
|H|^2 / H S_ff H^H map and with the reduction, exactly as M26-M30 established).

CONTINUOUS MONTE-CARLO cross-check. The independent time-domain validation reuses the
M26-M30 non-separable synthesisers on the FINE instant grid (per-instant
spectral-representation blocks concatenated), rainflow-counted (ASTM E1049) over the
WHOLE concatenated record and Miner-summed — the reference that DOES include the
straddling cycles the windowed closed-form misses. As the grid refines the continuous
Miner-integral converges to this Monte-Carlo (and the per-boundary shape jump, the
deterministic proxy for the window-boundary caveat, shrinks). In the windowed limit
the Monte-Carlo DELEGATES to the M26/M27/M29/M30 Monte-Carlo bit-identically.

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* LIBRARY-FIRST sub-flag (/IMPL/FATIG/.../WVILLE) — no upstream equivalent, exactly
  as established for M16-M30's PORT cards.
* SINGLE-COMPONENT evolutionary WVD: the smoothing is exposed as the Cohen-class
  cross-term control the theory names, but the drifting-shape model is unimodal /
  slowly-varying, so its raw WVD is cross-term-free; a full MULTI-component analytic
  WVD with genuine interference terms is not the load model here (the effective
  window is real and non-negative by construction). Documented, not hidden.
* A full NON-GAUSSIAN instantaneous-tensor time-frequency distribution, the
  base-acceleration multi-input feed, multi-directional 100-30-30 response spectra
  and an arbitrary per-pair per-window coherence-shape card beyond M30's schedules
  remain DEFERRED (the unchanged M24-M30 tail); M31 composes with the M24 kurtosis
  correction on the equivalent scalar but does not model a non-Gaussian
  time-frequency joint-tensor distribution.
* MEAN-STRESS beyond the basic M20/M21 Goodman intercept, CRACK-GROWTH /
  fracture-mechanics fatigue and the COMPLEX-FRF stress recovery remain DEFERRED (the
  unchanged M20-M30 tail).
"""

from __future__ import annotations

import numpy as np

from ..common.npcompat import trapezoid


# ============================================================================
# The continuous fine-grid schedule + the Cohen-class smoothing kernel
# (the M31 primitive: the instantaneous effective windows W_eff,j(f))
# ============================================================================

def _pair(v):
    """Coerce a scalar or 2-tuple ``v`` into (start, end) — a scalar is held
    constant (start == end); a (start, end) pair sweeps LINEARLY (the M26-M30
    drifting-shape convention)."""
    if isinstance(v, (tuple, list, np.ndarray)):
        a = np.asarray(v, dtype=float).ravel()
        if a.size >= 2:
            return float(a[0]), float(a[1])
        return float(a[0]), float(a[0])
    return float(v), float(v)


def is_windowed_limit(refine, smooth):
    """True in the WINDOWED limit — one instant per window (``refine`` <= 1) and no
    smoothing (``smooth`` <= 0) — where the continuous instantaneous spectrum
    collapses BYTE-IDENTICALLY onto the M26/M27/M29/M30 short-time windowed
    spectrogram (module docstring, the built-in M31 <-> M26-M30 reduction). In that
    limit the summaries DELEGATE to the windowed milestones."""
    return int(refine) <= 1 and float(smooth) <= 0.0


def instantaneous_schedule(durations, fc, bw, scales=None, refine=8):
    """The CONTINUOUS fine-grid instant schedule: subdivide each of the ``nwin``
    M26-M30 windows into ``refine`` equal sub-instants (so nt = nwin * refine) and
    evaluate the drifting shape CONTINUOUSLY at each fine instant's mission-fraction.

    Window i spans the mission-fraction cell [i/nwin, (i+1)/nwin] and carries
    duration T_i; sub-instant r of window i sits at mission-fraction
    s = (i + (r + 1/2)/refine) / nwin and inherits duration T_i / refine (so the
    total mission time sum_j Delta t_j = sum_i T_i is preserved EXACTLY, and at
    refine = 1 the fine grid IS the window grid: s_j = (i + 1/2)/nwin = the M26-M30
    window mid-time fraction, Delta t_j = T_i).

    ``fc`` / ``bw`` = scalar (held constant) or (start, end) (swept LINEARLY across
    the mission); ``scales`` = the per-WINDOW RMS level a_i (default all-ones),
    interpolated CONTINUOUSLY across the window mid-fractions (so at refine = 1 each
    fine instant recovers a_i exactly, and for refine > 1 the level follows a smooth
    mission envelope between window levels). Returns a dict with the fine-grid arrays
    ``s`` (mission-fraction), ``fc``, ``bw``, ``scale``, ``dur`` (each (nt,)) plus
    ``nwin`` / ``refine`` / ``nt`` and the per-window mid-fractions ``win_mid``."""
    T = np.asarray(durations, dtype=float).ravel()
    nwin = T.size
    if nwin == 0:
        raise ValueError("instantaneous_schedule needs at least one window.")
    if np.any(T <= 0.0):
        raise ValueError("every window duration must be positive.")
    refine = max(1, int(refine))
    nt = nwin * refine
    if scales is None:
        a_win = np.ones(nwin)
    else:
        a_win = np.asarray(scales, dtype=float).ravel()
        if a_win.size != nwin:
            raise ValueError("scales and durations must match in length.")
    fc0, fc1 = _pair(fc)
    bw0, bw1 = _pair(bw)
    # per-window mid-time fractions (the M26-M30 drifting parameterisation), used to
    # interpolate the piecewise-per-window level onto the continuous fine grid
    win_mid = (np.arange(nwin) + 0.5) / nwin if nwin > 1 else np.array([0.5])
    # fine-instant mission fractions: window i sub-instant r -> (i + (r+.5)/refine)/nwin
    idx = np.arange(nt)
    win_of = idx // refine
    sub = idx % refine
    s = (win_of + (sub + 0.5) / refine) / nwin
    fcj = fc0 + s * (fc1 - fc0)
    bwj = bw0 + s * (bw1 - bw0)
    # continuous level: linear interpolation of the per-window levels across the
    # window mid-fractions (exact = a_i at refine = 1 since s_j == win_mid there)
    if nwin > 1:
        aj = np.interp(s, win_mid, a_win)
    else:
        aj = np.full(nt, float(a_win[0]))
    durj = np.repeat(T / refine, refine)
    return {"s": s, "fc": fcj, "bw": bwj, "scale": aj, "dur": durj,
            "nwin": nwin, "refine": refine, "nt": nt, "win_mid": win_mid,
            "win_of": win_of}


def cohen_time_kernel(s, smooth):
    """The Cohen-class TIME-smoothing kernel g_jk over the mission-fraction axis
    ``s`` (nt,) — a normalised Gaussian of width ``smooth`` (std, in mission-fraction
    units) about each instant: g_jk propto exp(-(s_j - s_k)^2 / (2 smooth^2)), ROW
    normalised so sum_k g_jk = 1 (the smoothed instantaneous spectrum is a proper
    time-average of neighbouring instantaneous spectra — it preserves total power).

    ``smooth`` <= 0 returns the IDENTITY (the raw instantaneous WVD, no smoothing).
    As ``smooth`` grows past the mission span the kernel becomes uniform and every
    instant sees the mission-AVERAGE spectrum (the heavily-smoothed / spectrogram
    limit). Returns the (nt, nt) smoothing matrix."""
    s = np.asarray(s, dtype=float).ravel()
    nt = s.size
    if smooth is None or float(smooth) <= 0.0:
        return np.eye(nt)
    sig = float(smooth)
    d = s[:, None] - s[None, :]
    g = np.exp(-(d * d) / (2.0 * sig * sig))
    row = g.sum(axis=1, keepdims=True)
    row[row == 0.0] = 1.0
    return g / row


def instantaneous_effective_windows(freqs, durations, fc, bw, scales=None,
                                    refine=8, smooth=0.0):
    """The M31 PRIMITIVE: the fine-grid instantaneous EFFECTIVE WINDOWS W_eff,j(f)
    (module docstring). Build the continuous schedule (``instantaneous_schedule``),
    the RAW per-instant window R_j(f) = a_j^2 exp(-(f - fc_j)^2 / (2 bw_j^2)) (a flat
    all-ones window when bw_j <= 0 — no shape drift), and the Cohen-class
    time-smoothed W_eff,j(f) = sum_k g_jk R_k(f).

    Returns a dict with the schedule arrays (``s``, ``fc``, ``bw``, ``scale``,
    ``dur``, ``nwin``/``refine``/``nt``) plus ``R`` (nt, nf) the raw instantaneous
    windows, ``Weff`` (nt, nf) the smoothed effective windows, and ``freqs``. At
    refine = 1, smooth = 0, R == Weff and each row is a_i^2 W_i(f) — the M26-M30
    window BYTE-IDENTICALLY."""
    f = np.asarray(freqs, dtype=float)
    sch = instantaneous_schedule(durations, fc, bw, scales=scales, refine=refine)
    fcj, bwj, aj = sch["fc"], sch["bw"], sch["scale"]
    nt = sch["nt"]
    R = np.empty((nt, f.size))
    for j in range(nt):
        if bwj[j] > 0.0:
            W = np.exp(-((f - fcj[j]) ** 2) / (2.0 * bwj[j] ** 2))
        else:
            W = np.ones_like(f)                      # flat window (no shape drift)
        R[j] = aj[j] ** 2 * W
    g = cohen_time_kernel(sch["s"], smooth)
    Weff = g @ R if smooth and float(smooth) > 0.0 else R
    sch.update({"freqs": f, "R": R, "Weff": Weff})
    return sch


def boundary_shape_jump(psds):
    """The deterministic WINDOW-BOUNDARY CAVEAT proxy (module docstring): the mean
    NORMALISED shape discontinuity between adjacent instants / windows,
    mean_i || S_{i+1}/||S_{i+1}|| - S_i/||S_i|| ||. A COARSE grid (few windows) jumps
    a lot at each of its few boundaries; a FINE grid (many near-identical instants)
    jumps far less at each of its many boundaries — the measurable sense in which the
    continuous spectrum shrinks the boundary caveat the windowed Miner-sum carries.
    ``psds`` is an (n, nf) stack of per-instant PSDs. Returns a float (0 for one
    instant)."""
    S = np.asarray(psds, dtype=float)
    if S.ndim != 2 or S.shape[0] < 2:
        return 0.0
    nrm = np.linalg.norm(S, axis=1)
    nrm[nrm == 0.0] = 1.0
    U = S / nrm[:, None]
    return float(np.mean(np.linalg.norm(np.diff(U, axis=0), axis=1)))


# ============================================================================
# (a) SCALAR continuous instantaneous spectrum (the M26 base)
# ============================================================================

def wigner_ville_spectrogram(freqs, base_psd, durations, fc, bw, scales=None,
                             refine=8, smooth=0.0):
    """Build the CONTINUOUS instantaneous SCALAR spectrum S_WV(f, t_j) =
    W_eff,j(f) S_0(f) as the fine-grid ``windows`` list the M26 estimators /
    Monte-Carlo consume UNCHANGED (each instant a per-instant PSD, duration and
    moment set — the continuous analogue of ``evolutionary_fatigue.
    drifting_shape_spectrogram``, with ``refine`` instants per window and the
    Cohen-class ``smooth`` time-kernel).

    At refine = 1, smooth = 0 the returned list is the M26
    ``drifting_shape_spectrogram`` output BYTE-IDENTICALLY (same s_i, same window,
    same durations) — so feeding it to the M26 ``evolutionary_fatigue_summary``
    reproduces the M26 answer exactly. Each instant dict carries ``freqs``, ``psd``,
    ``duration``, ``moments`` (m0..m4), ``fc``, ``bw``, ``scale`` and ``shape_psd``
    (the UNIT-scale effective shape W_eff,j/a_j^2 * S_0 — used by the constant-shape
    detection / M25 delegation in the Monte-Carlo)."""
    from . import evolutionary_fatigue as efm
    from .evolutionary_fatigue import _moments_of_psd
    f = np.asarray(freqs, dtype=float)
    S0 = np.clip(np.asarray(base_psd, dtype=float), 0.0, None)
    # WINDOWED limit: DELEGATE to the M26 drifting-shape spectrogram so the per-
    # instant PSDs / moments are BYTE-IDENTICAL (the same multiply order — otherwise
    # a^2*(W*S0) vs (a^2*W)*S0 can flip the last bit). The built-in M31 <-> M26
    # reduction (module docstring).
    if is_windowed_limit(refine, smooth):
        return efm.drifting_shape_spectrogram(f, S0, durations, fc=fc, bw=bw,
                                              scales=scales)
    eff = instantaneous_effective_windows(f, durations, fc, bw, scales=scales,
                                          refine=refine, smooth=smooth)
    windows = []
    for j in range(eff["nt"]):
        Weff = eff["Weff"][j]
        psd = Weff * S0
        aj = eff["scale"][j]
        # the unit-scale shape (divide the effective window by a_j^2) — keeps the
        # M26 constant-shape detector / M25 delegation working in the flat limit
        shape = (Weff / (aj ** 2)) * S0 if aj != 0.0 else Weff * S0
        windows.append({"freqs": f, "psd": psd, "duration": float(eff["dur"][j]),
                        "moments": _moments_of_psd(f, psd),
                        "fc": float(eff["fc"][j]), "bw": float(eff["bw"][j]),
                        "scale": float(aj), "shape_psd": shape})
    return windows


def instantaneous_marginals(freqs, base_psd, durations, fc, bw, scales=None,
                            refine=8, smooth=0.0):
    """The Wigner-Ville MARGINALS (module docstring / theory eq. (1)) of the scalar
    continuous spectrum, for validation:
      * ``power`` (nt,) — the TIME marginal integral S_WV(f, t_j) df at each instant,
        the instantaneous power / variance sigma^2(t_j) = m0(t_j);
      * ``time`` (nt,) — the fine-instant mission-fraction grid s_j (for plotting);
      * ``avg_psd`` (nf,) — the (duration-weighted) time-AVERAGED PSD
        (1/T) integral S_WV(f, t) dt = the FREQUENCY marginal, which for a STATIONARY
        process recovers the stationary PSD EXACTLY and in general recovers the
        mission-average of the per-window PSDs;
      * ``peak_freq`` (nt,) — the instantaneous spectral PEAK frequency argmax_f
        S_WV(f, t_j) (drifts continuously for a chirp, finer than the windows
        resolve).
    The average PSD is formed from the RAW instantaneous spectra (the true
    evolutionary spectrum), independent of the smoothing (which only reshapes the
    per-instant slices, not the time-integral)."""
    f = np.asarray(freqs, dtype=float)
    S0 = np.clip(np.asarray(base_psd, dtype=float), 0.0, None)
    eff = instantaneous_effective_windows(f, durations, fc, bw, scales=scales,
                                          refine=refine, smooth=smooth)
    dur = eff["dur"]
    # instantaneous PSDs (smoothed slices for the per-instant power / peak) and the
    # RAW ones (for the time-average marginal — the true evolutionary spectrum)
    Ssm = eff["Weff"] * S0[None, :]
    Sraw = eff["R"] * S0[None, :]
    power = trapezoid(Ssm, f, axis=1)
    peak = f[np.argmax(Ssm, axis=1)]
    T = float(np.sum(dur))
    avg = np.sum(Sraw * dur[:, None], axis=0) / T if T > 0 else Sraw.mean(axis=0)
    return {"time": eff["s"], "power": power, "peak_freq": peak, "avg_psd": avg,
            "freqs": f}


def wigner_ville_fatigue_summary(freqs, base_psd, durations, fc, bw, m, C,
                                 scales=None, refine=8, smooth=0.0,
                                 mean_stress=0.0, ultimate=0.0):
    """The CONTINUOUS instantaneous SCALAR damage — the M20 estimators evaluated AT
    EACH fine instant and Palmgren-Miner INTEGRATED over time D = integral (dD/dt)(t)
    dt (the Riemann quadrature over the fine grid, of which the M26 windowed Miner-SUM
    is the coarse-grid midpoint case). Reuses the M26 ``evolutionary_fatigue_summary``
    read-only on the fine-grid instant windows (its window Miner-SUM over the fine
    instants IS the continuous Miner-INTEGRAL).

    In the WINDOWED limit (refine = 1, smooth = 0) DELEGATES to
    ``evolutionary_fatigue_summary`` on the M26 windows (byte-identical). Otherwise
    returns the same dict shape PLUS the continuous diagnostics: ``instantaneous``
    (the marginals: peak-frequency drift, instantaneous power, average PSD),
    ``peak_drift`` (Hz — the total swing of the instantaneous spectral peak),
    ``boundary_jump`` and ``boundary_jump_windowed`` (the caveat proxy on the fine
    grid vs the coarse window grid — the fine value is smaller), ``refine`` /
    ``smooth`` / ``nt`` / ``continuous`` (True)."""
    from . import evolutionary_fatigue as ef
    windows = wigner_ville_spectrogram(freqs, base_psd, durations, fc, bw,
                                       scales=scales, refine=refine, smooth=smooth)
    out = ef.evolutionary_fatigue_summary(windows, m, C, mean_stress=mean_stress,
                                          ultimate=ultimate)
    out["method"] = "wigner_ville_spectrum"
    out["refine"] = int(max(1, refine))
    out["smooth"] = float(smooth)
    out["nt"] = len(windows)
    out["continuous"] = not is_windowed_limit(refine, smooth)
    marg = instantaneous_marginals(freqs, base_psd, durations, fc, bw,
                                   scales=scales, refine=refine, smooth=smooth)
    out["instantaneous"] = marg
    pk = np.asarray(marg["peak_freq"], dtype=float)
    out["peak_drift"] = float(pk.max() - pk.min()) if pk.size else 0.0
    out["boundary_jump"] = boundary_shape_jump(
        np.asarray([w["psd"] for w in windows]))
    # the coarse WINDOW grid caveat proxy (refine = 1) for the side-by-side shrink
    coarse = wigner_ville_spectrogram(freqs, base_psd, durations, fc, bw,
                                      scales=scales, refine=1, smooth=0.0)
    out["boundary_jump_windowed"] = boundary_shape_jump(
        np.asarray([w["psd"] for w in coarse]))
    return out


def wigner_ville_monte_carlo_damage(freqs, base_psd, durations, fc, bw, m, C,
                                    seed, scales=None, refine=8, smooth=0.0,
                                    fs=None, mean_stress=0.0, ultimate=0.0):
    """The CONTINUOUS instantaneous SCALAR damage by Monte-Carlo — reuse the M26
    non-separable synthesiser on the FINE instant grid (per-instant
    spectral-representation blocks concatenated), rainflow-counted over the WHOLE
    record and Miner-summed (``evolutionary_monte_carlo_damage``). The reference that
    includes the straddling cycles the closed-form misses; as the grid refines the
    continuous Miner-integral converges to it. In the windowed limit (refine = 1,
    smooth = 0) the fine windows ARE the M26 windows, so this is the M26 Monte-Carlo
    byte-identically. Returns the M26 Monte-Carlo dict shape."""
    from . import evolutionary_fatigue as ef
    windows = wigner_ville_spectrogram(freqs, base_psd, durations, fc, bw,
                                       scales=scales, refine=refine, smooth=smooth)
    mc = ef.evolutionary_monte_carlo_damage(windows, m, C, seed, fs=fs,
                                            mean_stress=mean_stress,
                                            ultimate=ultimate)
    mc["method"] = "wigner_ville_monte_carlo"
    mc["refine"] = int(max(1, refine))
    mc["smooth"] = float(smooth)
    return mc


# ============================================================================
# (b) 6x6 TENSOR continuous instantaneous spectrum (the M27 base — the critical
#     plane / F_np drifting CONTINUOUSLY, per-instant re-search)
# ============================================================================

def _reduce_instant_tensors(omega, Scross, effwins, durations, m, C,
                            mean_stress, ultimate, naz, npol, drift=True):
    """Reduce a fine-grid instantaneous TENSOR spectrum: per instant form the 6x6
    windowed moment matrices M_{n,j} = (1/pi) integral omega^n W_eff,j(f) S_cross
    domega (``windowed_tensor_moment_matrices`` with the smoothed effective window
    folded in at unit scale), RE-SEARCH the critical plane / F_np from THAT instant's
    tensor (``reduce_window_tensor`` — the plane drifts CONTINUOUSLY), and
    Palmgren-Miner INTEGRATE the per-instant damages. Returns the M27
    ``joint_evolutionary_fatigue_summary`` dict shape (per-reduction damage / life,
    the per-instant plane / F_np / sigma_vm breakdown, plane_rotation_deg,
    fnp_drift). ``effwins`` = the (nt, nf) effective windows; ``durations`` = the
    fine (nt,) durations."""
    from . import spectral_fatigue as sf
    from .joint_evolutionary_fatigue import (windowed_tensor_moment_matrices,
                                             reduce_window_tensor,
                                             _reduce_window_fixed)
    from .multiaxial_fatigue import (tensor_moment_matrices,
                                     equivalent_vonmises_moments)
    Scross = np.asarray(Scross)
    T = np.asarray(durations, dtype=float).ravel()
    nt = len(effwins)
    # stationary reference (the un-windowed tensor) — for the drift=False fixed
    # reduction and the constant-shape detector
    Mstat = tensor_moment_matrices(omega, Scross, nmax=4)
    stat = reduce_window_tensor(Mstat, m, C, mean_stress=mean_stress,
                                ultimate=ultimate, naz=naz, npol=npol)
    keys = ("von_mises", "normal_plane", "shear_plane")
    D = {k: 0.0 for k in keys}
    wout = []
    shear_normals = []
    normal_normals = []
    fnps = []
    shapes = []
    for j in range(nt):
        Mi = windowed_tensor_moment_matrices(omega, Scross, effwins[j],
                                             scale=1.0, nmax=4)
        if drift:
            red = reduce_window_tensor(Mi, m, C, mean_stress=mean_stress,
                                       ultimate=ultimate, naz=naz, npol=npol)
        else:
            red = _reduce_window_fixed(Mi, stat, m, C, mean_stress, ultimate)
        Ti = float(T[j])
        for k in keys:
            D[k] += red[k]["damage_rate"] * Ti
        shear_normals.append(np.asarray(red["shear_plane"]["normal"], dtype=float))
        normal_normals.append(np.asarray(red["normal_plane"]["normal"],
                                         dtype=float))
        fnps.append(float(red["F_np"]))
        m0 = equivalent_vonmises_moments(Mi)[0]
        shapes.append(Mi[0] / m0 if m0 > 0 else Mi[0])
        wout.append({"fc": None, "bw": None, "scale": None, "duration": Ti,
                     "normal_n": normal_normals[-1], "shear_n": shear_normals[-1],
                     "normal_proj": np.asarray(red["normal_plane"]["proj"],
                                               dtype=float),
                     "shear_proj": np.asarray(red["shear_plane"]["proj"],
                                              dtype=float),
                     "F_np": fnps[-1], "sigma_vm": float(red["sigma_vm"]),
                     "vm_rate": red["von_mises"]["damage_rate"],
                     "normal_rate": red["normal_plane"]["damage_rate"],
                     "shear_rate": red["shear_plane"]["damage_rate"]})
    Ttot = float(np.sum(T))
    const = len(shapes) <= 1 or all(
        np.allclose(shapes[k], shapes[0], rtol=1e-9, atol=1e-12)
        for k in range(1, len(shapes)))

    def _rot(normals):
        n0 = normals[0]
        best = 0.0
        for n in normals[1:]:
            c = abs(float(np.dot(n0, n)))
            best = max(best, np.degrees(np.arccos(min(1.0, c))))
        return best
    rot = 0.0 if const else max(_rot(shear_normals), _rot(normal_normals))
    fnp_drift = float(np.max(fnps) - np.min(fnps)) if fnps else 0.0
    out = {"method": "wigner_ville_tensor", "nt": nt, "drift": bool(drift),
           "constant_shape": bool(const), "plane_rotation_deg": rot,
           "fnp_drift": fnp_drift, "windows": wout, "stationary": stat,
           "total_time": Ttot}
    for k in keys:
        dr = D[k] / Ttot if Ttot > 0 else 0.0
        tf, _ = sf.life_and_equivalent(dr, 0.0, m, sf._goodman_C(
            C, m, mean_stress, ultimate))
        out[k] = {"damage": D[k], "total_time": Ttot, "damage_rate": dr,
                  "life": tf}
    out["damage_rate"] = out["von_mises"]["damage_rate"]
    out["life"] = out["von_mises"]["life"]
    return out


def wigner_ville_tensor_summary(omega, Scross, durations, fc, bw, m, C,
                                scales=None, refine=8, smooth=0.0,
                                mean_stress=0.0, ultimate=0.0, naz=24, npol=13,
                                drift=True):
    """The CONTINUOUS instantaneous 6x6 TENSOR damage (the M27 base lifted to a
    continuous spectrum): the FULL stress-tensor cross-PSD S_sigmasigma(omega, t) is
    evaluated CONTINUOUSLY in t, the critical plane / F_np RE-SEARCHED at EACH instant
    (drifting continuously, not window to window), and the per-instant multiaxial
    damages Palmgren-Miner INTEGRATED.

    In the WINDOWED limit (refine = 1, smooth = 0) DELEGATES to the M27
    ``joint_evolutionary_fatigue_summary`` (byte-identical). Otherwise builds the
    fine-grid effective windows and reduces per instant (``_reduce_instant_tensors``).
    Returns the M27 summary dict shape PLUS ``refine`` / ``smooth`` / ``nt`` /
    ``continuous`` and (from ``_reduce_instant_tensors``) the continuous
    plane_rotation_deg / fnp_drift."""
    freqs = np.asarray(omega, dtype=float) / (2.0 * np.pi)
    if is_windowed_limit(refine, smooth):
        from .joint_evolutionary_fatigue import joint_evolutionary_fatigue_summary
        out = joint_evolutionary_fatigue_summary(
            omega, Scross, durations, fc=fc, bw=bw, m=m, C=C, scales=scales,
            mean_stress=mean_stress, ultimate=ultimate, naz=naz, npol=npol,
            drift=drift)
        out["refine"] = 1
        out["smooth"] = 0.0
        out["nt"] = out.get("nwin")
        out["continuous"] = False
        out["delegated"] = "m27_windowed"
        return out
    eff = instantaneous_effective_windows(freqs, durations, fc, bw, scales=scales,
                                          refine=refine, smooth=smooth)
    out = _reduce_instant_tensors(omega, Scross, eff["Weff"], eff["dur"], m, C,
                                  mean_stress, ultimate, naz, npol, drift=drift)
    out["refine"] = int(max(1, refine))
    out["smooth"] = float(smooth)
    out["nt"] = eff["nt"]
    out["continuous"] = True
    out["delegated"] = None
    # continuous peak-frequency drift (of the equivalent von-Mises scalar spectrum)
    Svm = _vonmises_scalar_psd(Scross)
    marg = instantaneous_marginals(freqs, Svm, durations, fc, bw, scales=scales,
                                   refine=refine, smooth=smooth)
    out["instantaneous"] = marg
    pk = np.asarray(marg["peak_freq"], dtype=float)
    out["peak_drift"] = float(pk.max() - pk.min()) if pk.size else 0.0
    return out


def _vonmises_scalar_psd(Scross):
    """The equivalent-von-Mises SCALAR PSD sigma_vm^2(f) = tr(Q S_cross(f)) (the M21
    quadratic von-Mises form on the 6x6 tensor cross-PSD), for the peak-frequency
    marginal only (a reporting descriptor, not the reduction). Reuses the M28
    ``equivalent_vonmises_psd_multi``, clipped >= 0."""
    from .multi_input_response import equivalent_vonmises_psd_multi
    return np.clip(np.real(np.asarray(equivalent_vonmises_psd_multi(Scross),
                                      dtype=float)), 0.0, None)


def wigner_ville_tensor_monte_carlo_damage(omega, Scross, durations, fc, bw, m, C,
                                           seed, scales=None, refine=8, smooth=0.0,
                                           fs=None, mean_stress=0.0, ultimate=0.0,
                                           naz=24, npol=13, reduction="shear_plane",
                                           summary=None):
    """The CONTINUOUS instantaneous TENSOR damage by Monte-Carlo — reuse the M27
    multivariate non-stationary synthesiser on the FINE grid, projecting each
    instant's block onto THAT instant's critical plane, rainflow + Miner. In the
    windowed limit DELEGATES to the M27 ``joint_evolutionary_monte_carlo_damage``.
    For the fine grid it drives the M27 Monte-Carlo with the refined
    (durations, fc, bw, scales) schedule (the fine grid IS a many-window
    joint-evolutionary schedule, so the M27 synthesiser applies unchanged); the
    smoothing enters only the closed-form slices, so the MC uses the raw fine
    schedule (documented). Returns the M27 Monte-Carlo dict shape."""
    from .joint_evolutionary_fatigue import joint_evolutionary_monte_carlo_damage
    if is_windowed_limit(refine, smooth):
        fine_dur = np.asarray(durations, dtype=float)
        fine_scales = scales
    else:
        # refine the schedule: split each window into `refine` equal sub-instants;
        # the M27 synthesiser samples fc/bw/scale at the sub-instant mid-fractions
        sch = instantaneous_schedule(durations, fc, bw, scales=scales,
                                     refine=refine)
        fine_dur = sch["dur"]
        fine_scales = sch["scale"]
    mc = joint_evolutionary_monte_carlo_damage(
        omega, Scross, fine_dur, fc=fc, bw=bw, m=m, C=C, seed=seed,
        scales=fine_scales, fs=fs, mean_stress=mean_stress, ultimate=ultimate,
        naz=naz, npol=npol, reduction=reduction, summary=summary)
    mc["method"] = "wigner_ville_tensor_monte_carlo"
    mc["refine"] = int(max(1, refine))
    mc["smooth"] = float(smooth)
    return mc


# ============================================================================
# (c) MULTI-INPUT continuous instantaneous spectrum (the M29/M30 base — the
#     coherence matrix gamma_ab(f, t) drifting CONTINUOUSLY)
# ============================================================================

def wigner_ville_multi_input_summary(omega, Hcols, auto_psds, durations, m, C,
                                     gamma0, gamma1=None, phase0=0.0, phase1=None,
                                     fc=0.0, bw=0.0, scales=None, refine=8,
                                     smooth=0.0, mean_stress=0.0, ultimate=0.0,
                                     naz=24, npol=13, drift=True,
                                     freq_dependent=False):
    """The CONTINUOUS instantaneous MULTI-INPUT damage (the M29 scalar-coherence /
    M30 frequency-dependent-coherence base lifted to a continuous spectrum): the
    input coherence matrix S_ff(omega, t) is evaluated CONTINUOUSLY in t (the
    coherence gamma_ab / phase, or the M30 frequency stack gamma_ab(f), drifting
    continuously), the per-instant multi-input stress-tensor cross-PSD
    S_sigmasigma(omega, t) = H_sigma S_ff(t) H_sigma^H reduced with a per-instant
    critical-plane re-search, and the per-instant damages Miner-INTEGRATED.

    ``freq_dependent`` selects the M30 (frequency-dependent stack, gamma0/gamma1 are
    (nf, n, n)) vs the M29 (scalar / (n,n) coherence) base. In the WINDOWED limit
    (refine = 1, smooth = 0) DELEGATES to the M30 ``freq_evolutionary_multi_input_
    summary`` (freq_dependent) or the M29 ``evolutionary_multi_input_summary``
    (byte-identical). For smooth = 0, refine > 1 it drives the SAME M29/M30 summary
    with the FINE-grid (durations, scales) — the finer grid is the continuous
    instantaneous spectrum, reusing the milestone's exact reduction. For smooth > 0 it
    runs the Cohen-class own-loop on the auto-PSD effective windows. Returns the
    M29/M30 summary dict shape PLUS ``refine`` / ``smooth`` / ``nt`` / ``continuous``.
    """
    from . import evolutionary_multi_input as emi
    from . import freq_evolutionary_multi_input as fem

    def _tag(out, delegated, nt):
        out = dict(out)
        out["refine"] = int(max(1, refine))
        out["smooth"] = float(smooth)
        out["nt"] = nt
        out["continuous"] = not is_windowed_limit(refine, smooth)
        out["wv_delegated"] = delegated
        return out

    # --- windowed limit OR un-smoothed fine grid: reuse the M29/M30 summary --------
    if float(smooth) <= 0.0:
        if is_windowed_limit(refine, smooth):
            fine_dur = np.asarray(durations, dtype=float)
            fine_scales = scales
            nt = fine_dur.size
        else:
            sch = instantaneous_schedule(durations, fc, bw, scales=scales,
                                         refine=refine)
            fine_dur = sch["dur"]
            fine_scales = sch["scale"]
            nt = sch["nt"]
        if freq_dependent:
            out = fem.freq_evolutionary_multi_input_summary(
                omega, Hcols, auto_psds, fine_dur, m, C, gamma0=gamma0,
                gamma1=gamma1, phase0=phase0, phase1=phase1, fc=fc, bw=bw,
                scales=fine_scales, mean_stress=mean_stress, ultimate=ultimate,
                naz=naz, npol=npol, drift=drift)
            return _tag(out, "m30_fine" if nt != np.asarray(durations).size
                        else "m30_windowed", nt)
        out = emi.evolutionary_multi_input_summary(
            omega, Hcols, auto_psds, fine_dur, m, C, gamma0=gamma0, gamma1=gamma1,
            phase0=phase0, phase1=phase1, fc=fc, bw=bw, scales=fine_scales,
            mean_stress=mean_stress, ultimate=ultimate, naz=naz, npol=npol,
            drift=drift)
        return _tag(out, "m29_fine" if nt != np.asarray(durations).size
                    else "m29_windowed", nt)

    # --- smooth > 0: the Cohen-class own-loop on the auto-PSD effective windows ----
    out = _multi_input_smoothed_loop(
        omega, Hcols, auto_psds, durations, m, C, gamma0, gamma1, phase0, phase1,
        fc, bw, scales, refine, smooth, mean_stress, ultimate, naz, npol, drift,
        freq_dependent)
    return _tag(out, "cohen_smoothed", out["nt"])


def _multi_input_smoothed_loop(omega, Hcols, auto_psds, durations, m, C, gamma0,
                               gamma1, phase0, phase1, fc, bw, scales, refine,
                               smooth, mean_stress, ultimate, naz, npol, drift,
                               freq_dependent):
    """The Cohen-class SMOOTHED multi-input own-loop (smooth > 0): the per-instant
    auto-PSDs get the SMOOTHED effective window G_j(f) = W_eff,j(f) G_0(f) (the
    smoothing is linear in the auto-spectra), the coherence gamma_ab / phase drifts
    continuously, and each instant's S_ff -> S_sigmasigma is reduced with a
    per-instant plane re-search, Miner-integrated. Reuses ``input_cross_psd_matrix``,
    ``stress_tensor_cross_psd_multi``, ``windowed_tensor_moment_matrices``-style
    integration and ``reduce_window_tensor`` (all read-only)."""
    from . import spectral_fatigue as sf
    from .multi_input_response import (input_cross_psd_matrix,
                                       stress_tensor_cross_psd_multi)
    from .joint_evolutionary_fatigue import (reduce_window_tensor,
                                             _reduce_window_fixed,
                                             windowed_tensor_moment_matrices)
    from .multiaxial_fatigue import (tensor_moment_matrices,
                                     equivalent_vonmises_moments)
    freqs = np.asarray(omega, dtype=float) / (2.0 * np.pi)
    G0 = np.clip(np.asarray(auto_psds, dtype=float), 0.0, None)
    ninput = G0.shape[1]
    sch = instantaneous_schedule(durations, fc, bw, scales=scales, refine=refine)
    nt = sch["nt"]
    # the raw per-instant effective window applied to the auto-PSDs (unit level here,
    # a_j folded in below); smoothing over the fine instants
    R = np.empty((nt, freqs.size))
    for j in range(nt):
        if sch["bw"][j] > 0.0:
            W = np.exp(-((freqs - sch["fc"][j]) ** 2) / (2.0 * sch["bw"][j] ** 2))
        else:
            W = np.ones_like(freqs)
        R[j] = sch["scale"][j] ** 2 * W
    Weff = cohen_time_kernel(sch["s"], smooth) @ R
    # continuous coherence schedule (scalar/matrix M29, or frequency-stack M30)
    g0 = gamma0
    g1 = gamma0 if gamma1 is None else gamma1
    p0 = float(phase0)
    p1 = p0 if phase1 is None else float(phase1)
    keys = ("von_mises", "normal_plane", "shear_plane")
    D = {k: 0.0 for k in keys}
    wout = []
    shear_normals = []
    normal_normals = []
    fnps = []
    shapes = []
    T = sch["dur"]
    for j in range(nt):
        sfrac = sch["s"][j]
        Gj = Weff[j][:, None] * G0                        # smoothed auto-PSDs
        gj = _interp_gamma(g0, g1, sfrac)
        pj = p0 + sfrac * (p1 - p0)
        res = input_cross_psd_matrix(Gj, gamma=gj, phase=pj)
        Sff = res["Sff"]
        Scr = stress_tensor_cross_psd_multi(Hcols, Sff)
        Mi = tensor_moment_matrices(omega, Scr, nmax=4)
        if drift:
            red = reduce_window_tensor(Mi, m, C, mean_stress=mean_stress,
                                       ultimate=ultimate, naz=naz, npol=npol)
        else:
            Mstat = Mi
            stat = reduce_window_tensor(Mstat, m, C, mean_stress=mean_stress,
                                        ultimate=ultimate, naz=naz, npol=npol)
            red = stat
        Ti = float(T[j])
        for k in keys:
            D[k] += red[k]["damage_rate"] * Ti
        shear_normals.append(np.asarray(red["shear_plane"]["normal"], dtype=float))
        normal_normals.append(np.asarray(red["normal_plane"]["normal"],
                                         dtype=float))
        fnps.append(float(red["F_np"]))
        m0 = equivalent_vonmises_moments(Mi)[0]
        shapes.append(Mi[0] / m0 if m0 > 0 else Mi[0])
        wout.append({"duration": Ti, "normal_n": normal_normals[-1],
                     "shear_n": shear_normals[-1],
                     "normal_proj": np.asarray(red["normal_plane"]["proj"],
                                               dtype=float),
                     "shear_proj": np.asarray(red["shear_plane"]["proj"],
                                              dtype=float),
                     "F_np": fnps[-1], "sigma_vm": float(red["sigma_vm"]),
                     "vm_rate": red["von_mises"]["damage_rate"],
                     "normal_rate": red["normal_plane"]["damage_rate"],
                     "shear_rate": red["shear_plane"]["damage_rate"]})
    Ttot = float(np.sum(T))
    const = len(shapes) <= 1 or all(
        np.allclose(shapes[k], shapes[0], rtol=1e-9, atol=1e-12)
        for k in range(1, len(shapes)))

    def _rot(normals):
        n0 = normals[0]
        best = 0.0
        for n in normals[1:]:
            best = max(best, np.degrees(np.arccos(
                min(1.0, abs(float(np.dot(n0, n)))))))
        return best
    rot = 0.0 if const else max(_rot(shear_normals), _rot(normal_normals))
    out = {"method": "wigner_ville_multi_input", "ninput": ninput, "nt": nt,
           "drift": bool(drift), "constant_shape": bool(const),
           "plane_rotation_deg": rot,
           "fnp_drift": float(np.max(fnps) - np.min(fnps)) if fnps else 0.0,
           "windows": wout, "total_time": Ttot}
    Ceff = sf._goodman_C(C, m, mean_stress, ultimate)
    for k in keys:
        dr = D[k] / Ttot if Ttot > 0 else 0.0
        tf, _ = sf.life_and_equivalent(dr, 0.0, m, Ceff)
        out[k] = {"damage": D[k], "total_time": Ttot, "damage_rate": dr,
                  "life": tf}
    out["damage_rate"] = out["von_mises"]["damage_rate"]
    out["life"] = out["von_mises"]["life"]
    return out


def _interp_gamma(g0, g1, s):
    """Linear interpolation of a coherence descriptor between the mission ENDS at
    mission-fraction ``s`` — a scalar, an (n, n) matrix or a (nf, n, n) frequency
    stack (all broadcast). Off-diagonal coherence stacks are clipped to [0, 1]."""
    if g0 is None and g1 is None:
        return None
    a0 = np.asarray(g0, dtype=float) if g0 is not None else None
    a1 = np.asarray(g1, dtype=float) if g1 is not None else None
    if a0 is None:
        return a1
    if a1 is None:
        return a0
    return a0 + float(s) * (a1 - a0)


def wigner_ville_multi_input_monte_carlo_damage(omega, Hcols, auto_psds, durations,
                                                m, C, seed, gamma0, gamma1=None,
                                                phase0=0.0, phase1=None, fc=0.0,
                                                bw=0.0, scales=None, refine=8,
                                                smooth=0.0, fs=None, mean_stress=0.0,
                                                ultimate=0.0, naz=24, npol=13,
                                                reduction="shear_plane",
                                                summary=None, measure=False,
                                                freq_dependent=False):
    """The CONTINUOUS instantaneous MULTI-INPUT damage by Monte-Carlo — reuse the
    M29/M30 multi-input synthesiser on the FINE grid (per-instant correlated-input
    blocks concatenated, per-instant plane projection, rainflow + Miner). In the
    windowed limit DELEGATES to the M29/M30 Monte-Carlo. For the fine grid it drives
    the SAME M29/M30 Monte-Carlo with the refined schedule (the smoothing enters only
    the closed-form slices; the MC uses the raw fine schedule, documented). Returns
    the M29/M30 Monte-Carlo dict shape."""
    from . import evolutionary_multi_input as emi
    from . import freq_evolutionary_multi_input as fem
    if is_windowed_limit(refine, smooth):
        fine_dur = np.asarray(durations, dtype=float)
        fine_scales = scales
    else:
        sch = instantaneous_schedule(durations, fc, bw, scales=scales,
                                     refine=refine)
        fine_dur = sch["dur"]
        fine_scales = sch["scale"]
    if freq_dependent:
        mc = fem.freq_evolutionary_multi_input_monte_carlo_damage(
            omega, Hcols, auto_psds, fine_dur, m, C, seed=seed, gamma0=gamma0,
            gamma1=gamma1, phase0=phase0, phase1=phase1, fc=fc, bw=bw,
            scales=fine_scales, fs=fs, mean_stress=mean_stress, ultimate=ultimate,
            naz=naz, npol=npol, reduction=reduction, summary=summary,
            measure=measure)
    else:
        mc = emi.evolutionary_multi_input_monte_carlo_damage(
            omega, Hcols, auto_psds, fine_dur, m, C, seed=seed, gamma0=gamma0,
            gamma1=gamma1, phase0=phase0, phase1=phase1, fc=fc, bw=bw,
            scales=fine_scales, fs=fs, mean_stress=mean_stress, ultimate=ultimate,
            naz=naz, npol=npol, reduction=reduction, summary=summary,
            measure=measure)
    mc = dict(mc)
    mc["method"] = "wigner_ville_multi_input_monte_carlo"
    mc["refine"] = int(max(1, refine))
    mc["smooth"] = float(smooth)
    return mc

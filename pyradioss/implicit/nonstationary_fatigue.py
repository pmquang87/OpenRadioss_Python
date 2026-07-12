"""
Non-stationary / evolutionary-PSD spectral fatigue — M25: the frequency-domain
fatigue-damage estimate of a random-vibration response whose PSD (or its
variance / RMS) VARIES WITH TIME, computed by extending the M20-M24 STATIONARY
spectral estimators to a time-varying spectrum, and cross-validated against a
non-stationary time-domain Monte-Carlo.

Every spectral estimator up to here assumed a STATIONARY process — a single,
time-invariant PSD: M20 (scalar narrow-band / Dirlik / Wirsching-Light /
Tovo-Benasciutti), M21-M23 (multiaxial reductions) and M24 (the non-Gaussian /
kurtosis correction) all evaluate ONE fixed set of spectral moments. M25 lifts
exactly that STATIONARITY assumption: the loading is a SEQUENCE (or a
time-modulated envelope) of stationary states, each with its own PSD / RMS, and
the total damage is the Palmgren-Miner sum of the per-state damages.

M24 and M25 are SIBLINGS and are DEEPLY linked. M24 lifted the GAUSSIAN
assumption at fixed stationarity; M25 lifts the STATIONARITY assumption. They
MEET on one shared case: an RMS-modulated (non-stationary) Gaussian process is
ITSELF non-Gaussian / leptokurtic — a slowly-varying amplitude turns a Gaussian
carrier into a heavy-tailed variance-mixture marginal (the Wolfsteiner-Trapp /
Kihm-Rizzi "non-stationary -> kurtosis" link). So the M25 amplitude-modulated
damage (the M20 stationary damage integrated over the RMS distribution) and the
M24 lambda_ng-corrected damage (at the modulation-induced kurtosis gamma_4) must
AGREE — the built-in M25<->M24 cross-check. This module REUSES
``nongaussian_fatigue`` (M24) READ-ONLY for exactly that bridge.

Fortran origin
--------------
There is NONE. ``engine/source/input/freimpl.F`` (the /IMPL reader, re-read line
by line for M16-M24 AND AGAIN for M25, fetched from raw.githubusercontent.com)
parses only /IMPL/DYNA (the DIRECT Newmark/HHT integrator), /IMPL/BUCKL,
/IMPL/DT, /IMPL/NONLIN and /IMPL/ARCL plus the linear-solver housekeeping — there
is no /FATIG, no S-N / Miner branch, no Dirlik / rainflow / narrow-band
estimator, no non-Gaussian / kurtosis machinery, and (as M16-M24 already found)
NOTHING non-stationary / evolutionary / spectrogram of any kind. The sole ``PSD``
token in the whole file is still ``IMUMPSD`` (line 269), a MUMPS-solver flag, not
a power spectral density. OpenRadioss is a time-domain crash/impact code: the
random-vibration fatigue analysis — stationary Gaussian (M20-M23), stationary
non-Gaussian (M24) OR non-stationary (M25) — is simply not part of the
open-source solver, exactly as M16 found for the real eigensolver, M17/M18 for
the transfer functions, M19 for the PSD machinery and M20-M24 for the
spectral-fatigue estimators this module extends.

So M25 does exactly what M16-M24 did: it ports non-stationary spectral fatigue as
a clean LIBRARY capability and drives it with a minimal PORT engine sub-flag
(/IMPL/FATIG/NSTAT — the non-stationary analogue of the M20 /IMPL/FATIG Gaussian
card and the M24 /IMPL/FATIG/NGAUSS card). Nothing in the M10 direct integrator,
the M16 REAL eigensolver, the M17/M18 superposition, the M19 PSD path, the M20
SCALAR fatigue, the M21 MULTIAXIAL SPECTRAL path, the M22 NON-PROPORTIONAL
TIME-DOMAIN path, the M23 SPECTRAL NON-PROPORTIONAL path OR the M24 NON-GAUSSIAN
correction is touched: the non-stationary path is a NEW, parallel path that
CONSUMES the M20 stationary estimators (and M24's lambda_ng, read-only) and
produces its answer ALONGSIDE the stationary numbers so a listing shows the
stationary and the non-stationary answers side by side.

Theory — non-stationary / evolutionary-PSD fatigue
--------------------------------------------------
(Priestley, "Evolutionary spectra and non-stationary processes", J. Roy. Statist.
Soc. B 27, 1965 — the evolutionary spectrum S(omega, t); Bendat & Piersol,
"Random Data: Analysis and Measurement Procedures", ch. 12 — non-stationary
random data; Wolfsteiner & Breuer / Wolfsteiner & Trapp, "Fatigue life due to
non-stationary vibration" (Int. J. Fatigue) — the "spectrogram" / RMS-mixture
method; Braccesi, Cianetti, Lori & Pioli — non-stationary random-vibration
fatigue in the frequency domain; Kihm, Ferguson & Antoni — non-stationary /
non-Gaussian random vibration fatigue; Rychlik — the switching-process /
Markov-modulated stationary model; Palmgren-Miner — the linear block-damage
summation; Newland, "An Introduction to Random Vibrations…" ch. 5-7 — the
spectral-moment machinery this module scales.)

PIECEWISE-STATIONARY / BLOCK ("mission profile") MODEL. The classic engineering
treatment of a non-stationary load is to partition it into a SEQUENCE of
stationary SEGMENTS (a "mission profile": run-up, dwell, run-down, ...), each
segment i a stationary process with its OWN stress PSD S_i(omega) (hence its own
moment array m_{n,i}) and DURATION T_i. Within each segment the STATIONARY M20
estimator gives a damage RATE (E[D]/T)_i; the segment contributes damage

    D_i = (E[D]/T)_i * T_i                                                  (1)

and the segments ACCUMULATE by the Palmgren-Miner linear rule:

    D = sum_i D_i = sum_i (E[D]/T)_i T_i ,   E[D]/T = D / sum_i T_i         (2)

The mission time-to-failure is T_f = 1 / (E[D]/T) = (sum_i T_i) / D (failure at
D = 1). This is the standard non-stationary-as-a-sequence-of-stationary approach
(``block_fatigue_summary``): each block is a full M20 evaluation, and the block
damages are duration-weighted and summed.

RMS-SCALING OF A SHARED SHAPE. A common and important special case is that every
segment has the SAME spectral SHAPE S(omega) but a different RMS LEVEL — segment
i is an amplitude scaling a_i of a reference process, S_i(omega) = a_i^2 S(omega)
(the PSD scales as the SQUARE of the RMS). Then every moment scales as
m_{n,i} = a_i^2 m_n, so the RMS scales as sigma_i = a_i sigma_0; but every RATE
(nu_0, nu_p, ~ sqrt(m_j/m_k)) and every spectral-WIDTH factor (alpha_1, alpha_2,
x_m, all ratios of moments) is UNCHANGED by the scaling. Because each M20
estimator's damage is (rate)/C * E[S^m] with E[S^m] proportional to sigma^m and
the shape coefficients (Dirlik D1..D3, Q, R; the Wirsching-Light / Tovo-
Benasciutti weights) scale-INVARIANT, the segment damage rate is EXACTLY

    (E[D]/T)_i = a_i^m (E[D]/T)_ref                                         (3)

for ALL four estimators, with (E[D]/T)_ref the damage of the unit-scale reference
shape (``block_moments`` scales the moments; the driver exploits (3)).

AMPLITUDE-MODULATED / EVOLUTIONARY MODEL. Priestley's SEPARABLE evolutionary
spectrum is S(omega, t) = |A(t)|^2 S(omega): a stationary-in-SHAPE process whose
RMS envelope A(t) is a slowly-varying (random) modulation. If the modulation is
slow compared with the carrier period, the process is locally stationary with
instantaneous RMS sigma(t) = A(t) sigma_0, and the total damage is the STATIONARY
damage INTEGRATED over the RMS distribution p(a) of the modulation:

    E[D]/T = integral (E[D]/T)(a) p(a) da = E[a^m] (E[D]/T)_ref            (4)

using (3): the closed-form E[a^m]-weighted damage (``amplitude_modulated_
summary``). This is the continuous limit of the block model with the blocks'
duration fractions as p(a): for equal-duration blocks the block Miner-sum (2)
EQUALS the modulated integral (4). E[a^m] is the m-th moment of the RMS envelope
— a single multiplier on the M20 stationary damage of every estimator, just as
M24's lambda_ng is (they are the two siblings' corrections).

THE RMS-MODULATION <-> KURTOSIS LINK (the M25<->M24 bridge). An amplitude-
modulated Gaussian process X(t) = A(t) G(t) — G a stationary UNIT-variance
Gaussian carrier, A an INDEPENDENT slowly-varying amplitude — is a SCALE MIXTURE
of normals, and its one-point marginal is NON-Gaussian / leptokurtic. With A and
G independent and G Gaussian (E[G^2] = 1, E[G^4] = 3),

    E[X^2] = E[A^2] ,  E[X^4] = 3 E[A^4] ,
    gamma_4 = E[X^4]/E[X^2]^2 = 3 E[A^4] / E[A^2]^2                        (5)

(``rms_modulation_kurtosis``). By Jensen E[A^4] >= E[A^2]^2 so gamma_4 >= 3
ALWAYS (a varying RMS is leptokurtic), = 3 only for a CONSTANT A (the Gaussian /
stationary limit). This is the Wolfsteiner-Trapp / Kihm-Rizzi non-stationary ->
kurtosis relation. The NON-STATIONARY AMPLIFICATION of the damage over the
EQUAL-VARIANCE stationary process (RMS sqrt(E[A^2]) sigma_0) is the
normalization-INVARIANT ratio

    kappa_ns = E[a^m] / E[a^2]^(m/2) = E[ tilde a^m ] ,  tilde a = a/sqrt(E[a^2])
                                                                          (6)

(``nonstationary_amplification``; tilde a is the variance-normalized modulation,
E[tilde a^2] = 1). kappa_ns -> 1 for a constant modulation. The BRIDGE: because
BOTH models describe the SAME leptokurtic marginal (kurtosis (5)), the M25
amplification (6) must AGREE with the M24 correction lambda_ng(gamma_4) evaluated
at the induced kurtosis (5):

    kappa_ns  ~=  lambda_ng( gamma_4 = 3 E[a^4]/E[a^2]^2 )                 (7)

VALIDATED BOTH DIRECTIONS. The two are EXACT and IDENTICAL (= 1) in the constant-
modulation / Gaussian limit, and agree to LEADING ORDER in the excess kurtosis:
expanding (6) with delta = tilde a^2 - 1 (E[delta] = 0, E[delta^2] = (gamma_4-3)/3)
gives kappa_ns ~= 1 + m(m-2)(gamma_4-3)/24, while the M24 amplitude transform
gives lambda_ng ~= 1 + m(m-1)(gamma_4-3)/24 (see ``nongaussian_fatigue``). Both
grow like m^2 (gamma_4 - 3): same DIRECTION, same order, differing only by the
structural (m-2)/(m-1) factor — the honest signature that the SCALE-MIXTURE
amplitude (M25: a leptokurtic Rayleigh mixture) and the HERMITE amplitude (M24:
Winterstein's g(V)) are two DIFFERENT one-parameter models of the same kurtosis.
The agreement is TIGHT for mild kurtosis / low slope and loosens for extreme
kurtosis / high slope (documented; the bridge is validated within tolerance, like
the M20/M24 Monte-Carlo checks).

NON-STATIONARY MONTE-CARLO cross-check. As an independent validation the module
SYNTHESISES a non-stationary time history — the M20 Gaussian spectral-
representation carrier (``synthesize_gaussian_history``) multiplied by a
TIME-VARYING RMS envelope a(t) (a block schedule or a continuous modulation) —
RAINFLOW-counts it (the M20 ASTM E1049 counter) and Miner-sums the ranges
(``nonstationary_monte_carlo_damage``). Its sample kurtosis reproduces (5) and
its time-varying RMS reproduces the schedule; its damage matches the block /
modulated spectral estimate to within the seeded scatter. In the CONSTANT-
modulation limit (a(t) = 1) the envelope is unity, x = u EXACTLY, and the
non-stationary Monte-Carlo reduces BIT-IDENTICALLY to the M20 Gaussian
Monte-Carlo.

    BLOCK-BOUNDARY RAINFLOW CAVEAT (documented). Rainflow counting over the
    concatenated non-stationary record naturally handles cycles that STRADDLE a
    block boundary (a peak in a high-RMS block paired with a valley in the next,
    lower-RMS block). The closed-form block Miner-sum (2) counts each block's
    cycles INDEPENDENTLY and so MISSES those boundary cycles — a small,
    documented discrepancy (bounded by the number of block boundaries relative to
    the cycle count) that vanishes as the blocks grow long relative to the cycle
    period. The Monte-Carlo is the reference that includes them.

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* LIBRARY-FIRST sub-flag (/IMPL/FATIG/NSTAT) — no upstream equivalent, exactly as
  established for M16-M24's PORT cards.
* SEPARABLE evolutionary spectrum ONLY: S(omega, t) = |A(t)|^2 S(omega) — a
  time-varying RMS on a FIXED spectral SHAPE (plus the piecewise-stationary block
  model, where each block may carry a DIFFERENT shape). A fully evolutionary
  NON-SEPARABLE S(omega, t) with a continuously time-varying spectral SHAPE
  (beyond the separable / piecewise-stationary model) is DEFERRED.
* The amplitude-modulated damage (4) and the bridge (7) are the SCALE-MIXTURE
  amplitude model; the leading-order (m-2)/(m-1) structural difference from the
  M24 Hermite amplitude is documented, not hidden — both are principled
  kurtosis-driven amplifications, and the port validates their agreement within
  tolerance rather than claiming an identity that does not hold.
* A full non-stationary MULTIAXIAL joint treatment (a time-varying stress-tensor
  cross-PSD) is DEFERRED; M25 applies the block/modulation scaling to the
  equivalent-scalar (von Mises / critical-plane) PSD the M21/M23 reductions
  already produce, so it COMPOSES with the multiaxial paths (each reduction's
  moments are scaled) but does not model a jointly evolutionary tensor.
* MEAN-STRESS beyond the basic M20/M21 Goodman intercept, CRACK-GROWTH /
  fracture-mechanics fatigue, the COMPLEX-FRF stress recovery and a MULTI-INPUT
  cross-PSD with coherence remain DEFERRED (the unchanged M20-M24 tail).
"""

from __future__ import annotations

import math

import numpy as np


# ============================================================================
# RMS-modulation statistics + the non-stationary amplification (build item 1)
# ============================================================================

def rms_modulation_moments(scales, weights=None):
    """The moments of an RMS MODULATION — the per-segment / per-sample amplitude
    scalings ``scales`` (a_i) with (optional) probability WEIGHTS ``weights``
    (w_i, e.g. the segment DURATION fractions; defaults to equal weights). The
    modulation describes the time-varying RMS envelope A(t) = a of the amplitude-
    modulated (evolutionary) process X = A G, G a unit-variance Gaussian carrier.

    Returns a dict:
      * ``e_a2`` = E[a^2], ``e_a4`` = E[a^4] — the amplitude second/fourth moments;
      * ``kurtosis`` = 3 E[a^4] / E[a^2]^2 — the INDUCED marginal kurtosis of the
        modulated Gaussian process (theory eq. (5), the Wolfsteiner-Trapp /
        Kihm-Rizzi link; >= 3 always, = 3 only for a constant modulation);
      * ``rms_scale`` = sqrt(E[a^2]) — the overall RMS scaling of the modulated
        process relative to the unit-scale shape;
      * ``scales`` / ``weights`` — the normalized inputs (weights sum to 1).

    The kurtosis is normalization-INVARIANT (a ratio of amplitude moments), so a
    modulation need NOT be pre-normalized to unit mean-square."""
    a = np.asarray(scales, dtype=float).ravel()
    if a.size == 0:
        raise ValueError("rms_modulation_moments needs at least one RMS scale.")
    if weights is None:
        w = np.ones_like(a)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != a.size:
            raise ValueError(
                f"scales ({a.size}) and weights ({w.size}) must match in length.")
    wsum = float(w.sum())
    if wsum <= 0.0:
        raise ValueError("RMS-modulation weights must sum to a positive value.")
    w = w / wsum
    e_a2 = float(np.sum(w * a ** 2))
    e_a4 = float(np.sum(w * a ** 4))
    # CONSTANT modulation (all scales equal, any level) is the EXACT stationary /
    # Gaussian limit: kurtosis EXACTLY 3 (short-circuit the float ratio, which
    # would round to 2.9999...); a genuinely varying modulation is leptokurtic.
    if np.all(a == a.flat[0]):
        kurt = 3.0
    else:
        kurt = 3.0 * e_a4 / e_a2 ** 2 if e_a2 > 0.0 else 3.0
    return {"e_a2": e_a2, "e_a4": e_a4, "kurtosis": kurt,
            "rms_scale": math.sqrt(max(e_a2, 0.0)),
            "scales": a, "weights": w}


def rms_modulation_kurtosis(scales, weights=None):
    """The INDUCED marginal kurtosis gamma_4 = 3 E[a^4]/E[a^2]^2 of the amplitude-
    modulated Gaussian process (theory eq. (5)) — a one-line convenience wrapping
    ``rms_modulation_moments``. >= 3 for ANY non-constant modulation (Jensen),
    = 3 EXACTLY for a constant one (the Gaussian / stationary limit)."""
    return rms_modulation_moments(scales, weights)["kurtosis"]


def nonstationary_amplification(scales, weights, m):
    """The NON-STATIONARY DAMAGE AMPLIFICATION kappa_ns of an amplitude-modulated
    process over the EQUAL-VARIANCE stationary process (theory eq. (6)):

        kappa_ns = E[a^m] / E[a^2]^(m/2) = E[ tilde a^m ] ,  tilde a = a/sqrt(E[a^2])

    the m-th moment of the variance-NORMALIZED modulation tilde a (E[tilde a^2] =
    1). This is the factor by which the amplitude-modulated damage EXCEEDS the
    damage of a stationary process with the SAME overall variance — the direct
    counterpart of the M24 correction lambda_ng, and the quantity the M25<->M24
    bridge (eq. (7)) compares. kappa_ns = 1 for a constant modulation; > 1 for any
    varying one (the leptokurtic amplification). ``m`` is the S-N slope."""
    st = rms_modulation_moments(scales, weights)
    a, w, e_a2 = st["scales"], st["weights"], st["e_a2"]
    if e_a2 <= 0.0:
        return 1.0
    # CONSTANT modulation -> amplification EXACTLY 1 (the stationary limit; the
    # float ratio a^m/(a^2)^(m/2) would otherwise round to 1.0000...2)
    if np.all(a == a.flat[0]):
        return 1.0
    e_am = float(np.sum(w * a ** m))
    return e_am / e_a2 ** (m / 2.0)


# ============================================================================
# Piecewise-stationary / block ("mission profile") damage (build item 1a)
# ============================================================================

def block_moments(base_moments, scale):
    """Scale a shared-shape moment array ``base_moments`` = [m0..m4] by an RMS
    scaling ``a`` = ``scale``: because the PSD scales as the SQUARE of the RMS,
    every moment scales as m_n -> a^2 m_n (theory: RMS-scaling of a shared shape).
    The rates and width factors (ratios of moments) are then UNCHANGED, and every
    M20 estimator's damage scales as a^m (theory eq. (3))."""
    return np.asarray(base_moments, dtype=float) * float(scale) ** 2


def block_fatigue_summary(blocks, m, C, mean_stress=0.0, ultimate=0.0,
                          estimator="dirlik", base_moments=None):
    """The PIECEWISE-STATIONARY / BLOCK ("mission profile") damage (theory eqs.
    (1)-(2)): partition the loading into stationary BLOCKS, run the M20 estimator
    per block and Palmgren-Miner SUM the block damages duration-weighted.

    ``blocks`` — a list of per-block dicts, each either
      * ``{"moments": [m0..m4], "duration": T_i}`` — an explicit block PSD (a
        DIFFERENT spectral shape per block is allowed), or
      * ``{"scale": a_i, "duration": T_i}`` — an RMS scaling of a SHARED shape
        ``base_moments`` (``block_moments`` scales it; theory eq. (3)).
    ``estimator`` — which M20 estimator to Miner-sum ("narrow_band" / "dirlik" /
    "wirsching_light" / "tovo_benasciutti"). ``m`` / ``C`` the S-N law.

    Returns a dict:
      * ``blocks`` — per-block [{scale, duration, damage_rate, damage, params}];
      * ``damage`` = sum_i (E[D]/T)_i T_i, ``total_time`` = sum_i T_i;
      * ``damage_rate`` = damage / total_time, ``life`` = total_time / damage
        (Miner: failure at D = 1);
      * ``s_eq`` — the equivalent constant-amplitude range at the mission peak
        rate; ``estimator`` — the estimator used."""
    from . import spectral_fatigue as sf
    est = {"narrow_band": sf.narrow_band_damage, "dirlik": sf.dirlik_damage,
           "wirsching_light": sf.wirsching_light_damage,
           "tovo_benasciutti": sf.tovo_benasciutti_damage}.get(estimator)
    if est is None:
        raise ValueError(f"unknown block estimator {estimator!r}; expected "
                         "narrow_band / dirlik / wirsching_light / "
                         "tovo_benasciutti.")
    if not blocks:
        raise ValueError("block_fatigue_summary needs at least one block.")
    out_blocks = []
    D = 0.0
    T = 0.0
    nu_sum = 0.0                                  # duration-weighted cycle rate
    for b in blocks:
        Ti = float(b.get("duration", 0.0))
        if Ti <= 0.0:
            raise ValueError("every block needs a positive duration.")
        if "moments" in b:
            mom = np.asarray(b["moments"], dtype=float)
            scale = float("nan")
        else:
            if base_moments is None:
                raise ValueError(
                    "a block given by 'scale' needs a shared 'base_moments'.")
            scale = float(b.get("scale", 1.0))
            mom = block_moments(base_moments, scale)
        r = est(mom, m, C, mean_stress, ultimate)
        dri = float(r["damage_rate"])
        Di = dri * Ti
        D += Di
        T += Ti
        nu_sum += float(r["nu"]) * Ti
        out_blocks.append({"scale": scale, "duration": Ti, "damage_rate": dri,
                           "damage": Di, "params": r.get("params"),
                           "nu": float(r["nu"])})
    dr = D / T if T > 0.0 else 0.0
    life = math.inf if dr <= 0.0 else 1.0 / dr
    nu_bar = nu_sum / T if T > 0.0 else 0.0
    Ceff = sf._goodman_C(C, m, mean_stress, ultimate)
    _tf, s_eq = sf.life_and_equivalent(dr, nu_bar, m, Ceff)
    return {"method": "block_miner", "estimator": estimator, "blocks": out_blocks,
            "damage": D, "total_time": T, "damage_rate": dr, "life": life,
            "s_eq": s_eq, "nu": nu_bar}


# ============================================================================
# Amplitude-modulated / evolutionary damage (build item 1b)
# ============================================================================

def amplitude_modulated_damage(gaussian_result, scales, weights, m):
    """Apply the amplitude-modulated correction (theory eqs. (3)-(4)) to a SINGLE
    stationary estimator result ``gaussian_result`` (a dict from
    ``spectral_fatigue``, carrying ``damage_rate`` / ``nu`` / ``s_eq``): scale the
    damage rate by E[a^m] (the m-th moment of the RMS modulation), the closed-form
    integral of the stationary damage over the RMS distribution.

        (E[D]/T)_mod = E[a^m] (E[D]/T)_stationary ,  T_f = T_f,stat / E[a^m] ,
        S_eq,mod = E[a^m]^(1/m) S_eq,stat

    Returns a NEW dict (the stationary one untouched) with the modulated
    ``damage_rate`` / ``life`` / ``s_eq`` plus ``e_am`` = E[a^m]; the stationary
    value is kept under ``stationary_damage_rate``."""
    st = rms_modulation_moments(scales, weights)
    a, w = st["scales"], st["weights"]
    e_am = float(np.sum(w * a ** m))
    drS = float(gaussian_result["damage_rate"])
    dr = e_am * drS
    life = math.inf if dr <= 0.0 else 1.0 / dr
    s_eqS = gaussian_result.get("s_eq", 0.0)
    s_eq = (e_am ** (1.0 / m)) * s_eqS if s_eqS > 0 else 0.0
    out = dict(gaussian_result)
    out.update({"method": gaussian_result.get("method", "?") + "_modulated",
                "damage_rate": dr, "life": life, "s_eq": s_eq, "e_am": e_am,
                "stationary_damage_rate": drS})
    return out


def amplitude_modulated_summary(base_moments, scales, weights, m, C,
                                mean_stress=0.0, ultimate=0.0, stationary=None):
    """The full AMPLITUDE-MODULATED / evolutionary correction of ALL four M20
    estimators on a shared-shape moment array ``base_moments`` (theory eqs.
    (3)-(4)). Computes (or reuses via ``stationary``) the M20 stationary
    ``fatigue_summary`` of the base shape, then scales every estimator by E[a^m]
    (the m-th RMS-modulation moment).

    Returns a dict with:
      * ``e_am`` — the E[a^m] modulation factor (eq. (4)); ``e_a2`` / ``e_a4`` /
        ``kurtosis`` — the modulation moments + INDUCED kurtosis (eq. (5));
      * ``kappa_ns`` — the normalization-invariant amplification E[a^m]/E[a^2]^(m/2)
        (eq. (6), the M25<->M24 bridge quantity);
      * per-estimator modulated results (narrow_band / dirlik / wirsching_light /
        tovo_benasciutti) with the modulated damage_rate / life / s_eq;
      * ``stationary`` — the untouched M20 stationary summary (side-by-side).

    E[a^m] = 1, kappa_ns = 1 (the M20 answer recovered) for a constant modulation."""
    from . import spectral_fatigue as sf
    if stationary is None:
        stationary = sf.fatigue_summary(base_moments, m, C, mean_stress, ultimate)
    st = rms_modulation_moments(scales, weights)
    a, w = st["scales"], st["weights"]
    e_am = float(np.sum(w * a ** m))
    kappa_ns = nonstationary_amplification(scales, weights, m)
    out = {"e_am": e_am, "e_a2": st["e_a2"], "e_a4": st["e_a4"],
           "kurtosis": st["kurtosis"], "rms_scale": st["rms_scale"],
           "kappa_ns": kappa_ns, "scales": a, "weights": w,
           "stationary": stationary}
    for key in ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti"):
        out[key] = amplitude_modulated_damage(stationary[key], scales, weights, m)
    return out


def bridge_to_nongaussian(scales, weights, m, alpha2=1.0,
                          bandwidth_correction=False):
    """The M25<->M24 BRIDGE (theory eq. (7)): compare the M25 non-stationary
    amplification kappa_ns = E[a^m]/E[a^2]^(m/2) (this module) with the M24
    non-Gaussian correction lambda_ng evaluated at the modulation-INDUCED kurtosis
    gamma_4 = 3 E[a^4]/E[a^2]^2 (``nongaussian_fatigue``, READ-ONLY). Both are
    kurtosis-driven damage amplifications of the SAME leptokurtic marginal; they
    are EXACTLY 1 in the constant-modulation limit and agree to leading order in
    the excess kurtosis (differing by the structural (m-2)/(m-1) amplitude-model
    factor — documented). Returns a dict {kurtosis, kappa_ns, lambda_ng, ratio}."""
    from . import nongaussian_fatigue as ngf
    st = rms_modulation_moments(scales, weights)
    gamma4 = st["kurtosis"]
    kappa_ns = nonstationary_amplification(scales, weights, m)
    lam = ngf.nongaussian_correction_factor(
        0.0, gamma4, m, alpha2=alpha2,
        bandwidth_correction=bandwidth_correction)
    ratio = kappa_ns / lam if lam > 0.0 else float("nan")
    return {"kurtosis": gamma4, "kappa_ns": kappa_ns, "lambda_ng": lam,
            "ratio": ratio, "m": float(m)}


# ============================================================================
# Modulation <-> function/schedule helpers (build item 2 support)
# ============================================================================

def modulation_from_schedule(scales, durations):
    """Build a (scales, weights) modulation from a BLOCK SCHEDULE — matched lists
    of per-block RMS ``scales`` (a_i) and ``durations`` (T_i) — with the weights
    the DURATION fractions w_i = T_i / sum(T_i). The time-fraction distribution of
    the mission profile IS the modulation's p(a) (theory: the block model is the
    discrete amplitude-modulation). Returns (scales, weights)."""
    a = np.asarray(scales, dtype=float).ravel()
    T = np.asarray(durations, dtype=float).ravel()
    if a.size != T.size:
        raise ValueError("scales and durations must match in length.")
    if np.any(T <= 0.0):
        raise ValueError("every block duration must be positive.")
    return a, T / float(T.sum())


def sample_function_modulation(func, nseg, t0=0.0, t1=None):
    """Sample a modulation /FUNCT (a piecewise-linear scale-vs-time envelope, its
    (x, y) points giving the RMS scaling a as a function of time) into ``nseg``
    equal-duration blocks over [``t0``, ``t1``] (``t1`` defaults to the function's
    last abscissa). Each block's scale is the function evaluated at the block
    MIDPOINT; the durations are equal. Returns (scales, durations).

    Used by the driver to turn a mission-profile /FUNCT (run-up / dwell /
    run-down) into the block schedule the closed-form estimators and the
    Monte-Carlo consume. ``func`` is any object with an ``eval(x)`` method (the
    deck /FUNCT table) OR a callable."""
    ev = func.eval if hasattr(func, "eval") else func
    xs = getattr(func, "x", None)
    if t1 is None:
        if xs is not None and len(xs):
            t1 = float(np.max(xs))
        else:
            t1 = 1.0
    if t1 <= t0:
        t1 = t0 + 1.0
    nseg = max(1, int(nseg))
    edges = np.linspace(t0, t1, nseg + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    scales = np.array([abs(float(ev(x))) for x in mids], dtype=float)
    durations = np.diff(edges)
    return scales, durations


# ============================================================================
# Non-stationary Monte-Carlo cross-check (build item 2)
# ============================================================================

def synthesize_nonstationary_history(freqs_hz, psd, scales, durations, seed,
                                     fs=None):
    """Synthesise a NON-STATIONARY time history — the M20 Gaussian spectral-
    representation carrier (``synthesize_gaussian_history``: an inverse-rFFT of the
    shared-shape PSD with seeded random phases, RMS sigma_0 = sqrt(m0)) MULTIPLIED
    by a TIME-VARYING RMS envelope a(t) built from the block schedule (``scales``
    a_i over ``durations`` T_i, concatenated in order):

        u(t) = M20 Gaussian carrier of the shape PSD (RMS sigma_0) ,
        x(t) = a(t) u(t) ,   a(t) = a_i  for  t in block i                  (theory)

    The total record length is sum(durations); each block spans its own duration
    (which should be LONG relative to the carrier period so the envelope is 'slow'
    — the locally-stationary assumption). Returns (t, x, envelope).

    In the CONSTANT-modulation limit (a single block of scale 1, or all scales 1)
    the envelope is unity and x = u EXACTLY — the M20 Gaussian history recovered
    bit-for-bit (the identity short-circuit is implicit: a(t) == 1).

    BLOCK-BOUNDARY RAINFLOW CAVEAT (documented, module docstring): the concatenated
    record's rainflow (the caller's job) DOES count cycles that straddle a block
    boundary; the closed-form block Miner-sum does NOT — the small documented
    discrepancy between the Monte-Carlo and the block estimate."""
    from . import spectral_fatigue as sf
    a = np.asarray(scales, dtype=float).ravel()
    T = np.asarray(durations, dtype=float).ravel()
    if a.size != T.size:
        raise ValueError("scales and durations must match in length.")
    if np.any(T <= 0.0):
        raise ValueError("every block duration must be positive.")
    total = float(T.sum())
    t, u = sf.synthesize_gaussian_history(freqs_hz, psd, total, seed, fs=fs)
    # build the piecewise-constant envelope a(t): assign each time sample to its
    # block by the cumulative block-boundary times
    edges = np.concatenate([[0.0], np.cumsum(T)])
    # np.searchsorted maps each time sample to its block index (clip the final
    # sample, which can land exactly on the last edge)
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, a.size - 1)
    env = a[idx]
    x = env * u
    return t, x, env


def nonstationary_monte_carlo_damage(freqs_hz, psd, m, C, scales, durations, seed,
                                     fs=None, mean_stress=0.0, ultimate=0.0):
    """The TIME-DOMAIN NON-STATIONARY damage rate by Monte-Carlo (theory
    "NON-STATIONARY MONTE-CARLO"): synthesise the non-stationary history
    (``synthesize_nonstationary_history`` — the M20 Gaussian carrier times the
    time-varying RMS envelope), rainflow-count it (ASTM E1049, the M20 counter)
    and Miner-sum the ranges with the S-N law N = C S^-m — the independent
    time-domain answer the block / amplitude-modulated spectral estimate
    approximates. Seeded for reproducibility.

    Returns the M20 Monte-Carlo dict shape plus the sample ``kurtosis`` of the
    synthesised history (which reproduces the induced kurtosis, theory eq. (5)) and
    the per-block realised RMS ``block_rms`` (the time-varying RMS check). In the
    CONSTANT-modulation limit reduces EXACTLY to the M20 ``monte_carlo_damage``."""
    from . import spectral_fatigue as sf
    Ceff = sf._goodman_C(C, m, mean_stress, ultimate)
    t, x, env = synthesize_nonstationary_history(freqs_hz, psd, scales,
                                                 durations, seed, fs=fs)
    ranges, counts = sf.rainflow_count(x)
    D = float(np.sum(counts * ranges ** m) / Ceff) if ranges.size else 0.0
    T = t[-1] - t[0] if t.size > 1 else float(np.sum(durations))
    dr = D / T if T > 0 else 0.0
    tf, s_eq = sf.life_and_equivalent(dr, ranges.size / T if T > 0 else 0.0,
                                      m, Ceff)
    xc = x - np.mean(x)
    var = float(np.mean(xc * xc))
    kurt = float(np.mean(xc ** 4) / var ** 2) if var > 0 else 3.0
    # per-block realised RMS (the time-varying envelope check)
    a = np.asarray(scales, dtype=float).ravel()
    Tb = np.asarray(durations, dtype=float).ravel()
    edges = np.concatenate([[0.0], np.cumsum(Tb)])
    block_rms = []
    for i in range(a.size):
        seg = x[(t >= edges[i]) & (t < edges[i + 1])]
        block_rms.append(float(np.std(seg)) if seg.size else 0.0)
    return {"method": "nonstationary_monte_carlo", "damage_rate": dr, "life": tf,
            "s_eq": s_eq, "ncycles": float(counts.sum()), "ranges": ranges,
            "counts": counts, "duration": T, "rms": float(np.std(x)),
            "kurtosis": kurt, "block_rms": np.asarray(block_rms)}

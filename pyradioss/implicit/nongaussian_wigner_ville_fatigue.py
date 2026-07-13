"""
Non-Gaussian instantaneous-tensor time-frequency distribution — M32: a per-instant,
time-VARYING NON-GAUSSIAN (kurtosis / skewness) correction of the CONTINUOUS
Wigner-Ville instantaneous stress spectrum (M31), so the leptokurtic damage
amplification lambda_ng(t) itself DRIFTS with time along the continuous spectrum,
reduced PER INSTANT and Palmgren-Miner INTEGRATED over time, cross-validated by a
non-Gaussian NON-STATIONARY Monte-Carlo.

WHERE M32 SITS — THE CONVERGENCE OF M24 AND M31. Two independent milestones meet
here:
  * M24 (``nongaussian_fatigue.py``) corrected ONE stationary equivalent scalar for a
    FIXED kurtosis gamma_4 (and skewness gamma_3): the closed-form Winterstein-Hermite
    factor lambda_ng SCALES the Gaussian spectral damage of every M20 estimator,
    lambda_ng = E[g(V)^m]/E[V^m], V ~ Rayleigh(1), attenuated toward Gaussian for a
    wide band by the irregularity factor alpha_2 (Benasciutti-Tovo). It is STATIONARY
    non-Gaussian — one process, one kurtosis, one factor.
  * M31 (``wigner_ville_fatigue.py``) gave the Gaussian joint-tensor a CONTINUOUS
    instantaneous spectrum S_WV(omega, t): the drifting shape fc(t)/bw(t)/a(t) (and
    the coherence gamma_ab(f, t)) evaluated CONTINUOUSLY on a fine instant grid,
    reduced (scalar / 6x6 tensor with the per-instant critical-plane re-search) AT
    EACH INSTANT and Miner-INTEGRATED D = integral (dD/dt)(t) dt. It is
    non-stationary but GAUSSIAN at every instant.
M32 lets the kurtosis gamma_4(t) (and skewness gamma_3(t)) of the instantaneous
critical-plane / von-Mises equivalent scalar VARY WITH TIME ALONG the continuous
Wigner-Ville spectrum: at each fine instant t_j it forms the M31 instantaneous
Gaussian reduction (its per-instant moments, its per-instant bandwidth alpha_2(t_j),
its per-instant Gaussian damage rate (dD/dt)_G(t_j)) AND, from that instant's
alpha_2(t_j) and a TIME-VARYING target kurtosis gamma_4(t_j) sampled continuously
onto the fine grid, its instantaneous NON-GAUSSIAN amplification lambda_ng(t_j) (the
M24 closed form re-computed AT EACH INSTANT). The non-Gaussian per-instant damage
rate is lambda_ng(t_j) (dD/dt)_G(t_j) and the mission damage is the Miner INTEGRAL

    D_nG = integral lambda_ng(t) (dD/dt)_G(t) dt
         ~ sum_j lambda_ng(t_j) (dD/dt)_G(t_j) Delta t_j                        (*)

— a NON-GAUSSIAN INSTANTANEOUS spectrum whose leptokurtic amplification DRIFTS
continuously, of which the M24 stationary correction is the constant-kurtosis /
constant-bandwidth special case (lambda_ng(t) == const pulls out of (*)) and the M31
Gaussian continuous integral is the gamma_4(t) == 3 special case (lambda_ng(t) == 1).

THE M32 <-> M24 / M31 REDUCTIONS (built in, exact). Both parent answers are recovered
EXACTLY, and asserted (like every milestone since M14):
  * gamma_4(t) == 3, gamma_3(t) == 0 (Gaussian at every instant) -> lambda_ng(t) == 1
    EXACTLY (the M24 Hermite transform is the identity), so (*) collapses to the M31
    continuous Gaussian Miner-integral BYTE-IDENTICALLY. M32 DELEGATES to the M31
    summary in that limit (guaranteeing byte-identity, exactly as M31 delegated to
    M26/M27, M27 to M26).
  * a CONSTANT (time-invariant) kurtosis with the bandwidth attenuation OFF (or a
    STATIONARY process, where alpha_2(t) is constant) gives a CONSTANT lambda_ng that
    factors out of (*): D_nG = lambda_ng D_G, so the M32 answer is EXACTLY the M24
    correction applied to the M31 continuous Gaussian answer (lambda_ng from the M24
    closed form, D_G the M31 continuous integral). For a stationary process this is
    EXACTLY the M24 stationary answer (M31 recovers the stationary reduction at every
    instant, so D_G is the stationary damage).
  * refine = 1, smooth = 0 (the WINDOWED limit): the fine grid IS the M26-M30 window
    grid, so (*) collapses to the per-WINDOW non-Gaussian Miner-SUM — the M24
    correction applied to the M27 windowed joint-tensor (scalar: the M24 correction
    on the M26 windowed scalar) — recovered EXACTLY (the tensor path rebuilds the M27
    per-window moment matrices so the Gaussian per-window reduction is byte-identical
    to M27 before the per-window lambda_ng scales it).

THE M25/M26 RMS-INDUCED-KURTOSIS BRIDGE. M25 showed that an RMS-MODULATED stationary
Gaussian carrier is itself non-Gaussian — the amplitude modulation induces an
apparent kurtosis gamma_4 = E[a^4]/E[a^2]^2 * 3 (``nonstationary_fatigue.
rms_modulation_kurtosis``), and the M25<->M24 bridge maps the E[a^m] modulation
amplification onto an equivalent lambda_ng. M32 is the natural home of that bridge for
a continuous spectrum: the drifting RMS level a(t) of the M31 continuous spectrum
ALREADY induces a kurtosis, and M32 lets the caller ADD a genuine (measured / target)
non-Gaussian kurtosis gamma_4(t) ON TOP of it — the two compose (the induced kurtosis
is reported alongside the target as a diagnostic, not double-counted: the target
gamma_4(t) is the physical one-point marginal the Hermite transform imposes).

Fortran origin
--------------
There is NONE. ``engine/source/input/freimpl.F`` (the /IMPL reader, re-read line by
line for M16-M31 AND AGAIN for M32 — 639 lines, fetched from
raw.githubusercontent.com) parses only /IMPL/DYNA (the DIRECT Newmark/HHT
integrator), /IMPL/BUCKL, /IMPL/DT, /IMPL/NONLIN and /IMPL/ARCL plus the
linear-solver housekeeping — there is no /FATIG, no S-N / Miner branch, no PSD /
spectral / random-vibration path, and (as M16-M31 already found, line by line)
NOTHING time-frequency / Wigner-Ville / Cohen-class / non-Gaussian / Hermite /
Winterstein / evolutionary of any kind. The sole ``PSD`` token in the whole file is
still ``IMUMPSD`` (line 269, ``IF (ISOLV==3) IMUMPSD=L_LIM``), a MUMPS-solver flag,
NOT a power spectral density. OpenRadioss is a time-domain crash/impact code: the
random-vibration fatigue analysis — stationary Gaussian (M20-M23), stationary
NON-Gaussian (M24), separable non-stationary (M25), evolutionary windowed (M26-M30),
the CONTINUOUS Wigner-Ville instantaneous spectrum (M31) OR the time-VARYING
non-Gaussian instantaneous spectrum (M32) — is simply not part of the open-source
solver, exactly as every spectral milestone since M16 recorded.

So M32 does exactly what M16-M31 did: it ports the time-varying non-Gaussian
instantaneous time-frequency distribution as a clean LIBRARY capability EXTENDING the
M31 continuous line + the M24 non-Gaussian line, and drives it with a minimal PORT
engine sub-flag (/IMPL/FATIG/NGAUSS composing with /WVILLE, plus a kurtosis-vs-time
/FUNCT feeding gamma_4(t)/gamma_3(t)). Nothing in the M10 integrator, the M16
eigensolver, the M17/M18 superposition, the M19 PSD path, the M20-M27 reductions, the
M24 stationary non-Gaussian correction OR the M31 continuous Gaussian spectrum is
touched: the time-varying non-Gaussian instantaneous path is a NEW, parallel path that
CONSUMES the M24 closed form and the M31 continuous machinery (read-only) and produces
its answer ALONGSIDE the M31 Gaussian-continuous and the M24 stationary-non-Gaussian
numbers — both stay byte-identical (the M24 correction is EXACTLY the constant-kurtosis
limit and the M31 Gaussian answer EXACTLY the gamma_4 == 3 limit, asserted).

Theory — the time-varying non-Gaussian instantaneous spectrum
-------------------------------------------------------------
(Winterstein, "Nonlinear vibration models for extremes and fatigue", J. Eng. Mech.
114, 1988 — the Hermite-moment transformation g(u) = kappa[u + h_3(u^2-1) +
h_4(u^3-3u)] and the softening coefficients; Benasciutti & Tovo, "Cycle distribution
and fatigue damage assessment in broad-band non-Gaussian random processes", Prob.
Eng. Mech. 20, 2005 / Int. J. Fatigue 2006 — the bandwidth-dependent non-Gaussian
rainflow correction; Braccesi, Cianetti, Lori & Pioli, Int. J. Fatigue 31, 2009 — the
closed-form non-Gaussian damage coefficient; Rizzi, Kihm, Ferguson, Przekop, Robinson
et al. — the kurtosis-corrected spectral damage; Kihm & Rizzi 2013 — how kurtosis
transfers from input acceleration to stress response, the RMS-induced-kurtosis bridge
M25/M26 uses; Wigner 1932 / Ville 1948 — the Wigner-Ville distribution; Mark 1970 /
Martin & Flandrin 1985 — the Wigner-Ville spectrum of a non-stationary random process;
Cohen 1989 — the class of time-frequency distributions; Priestley 1965 — the
evolutionary spectrum the windowed spectrogram approximates and M31 makes continuous;
the M24 stationary non-Gaussian base and the M31 continuous instantaneous base this
module converges.)

THE INSTANTANEOUS NON-GAUSSIAN AMPLIFICATION lambda_ng(t). At each fine instant t_j of
the M31 continuous spectrum the reduction produces a per-instant Gaussian equivalent
scalar (the M26 scalar PSD, or the per-instant re-searched critical-plane / von-Mises
scalar of the 6x6 tensor) with its OWN spectral moments m_0(t_j)..m_4(t_j) and hence
its OWN irregularity factor alpha_2(t_j) = nu_0/nu_p. A TIME-VARYING target kurtosis
gamma_4(t_j) (and skewness gamma_3(t_j)), sampled CONTINUOUSLY from a kurtosis-vs-time
schedule onto the fine grid, feeds the M24 closed form AT THAT INSTANT:

    lambda_ng(t_j) = nongaussian_correction_factor(gamma_3(t_j), gamma_4(t_j), m,
                                                   alpha_2(t_j))                 (1)

— a per-instant Winterstein-Hermite amplitude correction, wide-band-attenuated by the
INSTANTANEOUS bandwidth alpha_2(t_j). The mission damage is the Miner INTEGRAL (*) of
the non-Gaussian per-instant rate lambda_ng(t_j) (dD/dt)_G(t_j) over time. For a
LEPTOKURTIC instant (gamma_4(t_j) > 3) lambda_ng(t_j) > 1 (the spikes do the damage,
locally); for a Gaussian instant (gamma_4(t_j) = 3) lambda_ng(t_j) = 1 EXACTLY; so a
BURSTY mission whose kurtosis sweeps up during a transient and relaxes to Gaussian
between bursts sees its damage amplification DRIFT continuously with the burst.

THE NON-GAUSSIAN NON-STATIONARY MONTE-CARLO. The independent time-domain validation
reuses the M31 continuous non-separable synthesiser on the fine instant grid
(per-instant spectral-representation blocks concatenated) and pushes EACH per-instant
block through the M24 memoryless Winterstein-Hermite transform to THAT instant's
target (gamma_3(t_j), gamma_4(t_j)) — so the synthesised record's LOCAL kurtosis
tracks gamma_4(t) — then rainflow-counts (ASTM E1049) the WHOLE concatenated record
and Miner-sums. As the grid refines the time-varying non-Gaussian Miner-integral (*)
converges to this Monte-Carlo; in the Gaussian limit (gamma_4(t) == 3) the transform
is the identity and the record reduces EXACTLY to the M31 Gaussian non-separable
Monte-Carlo. The induced sample kurtosis of the record tracks gamma_4(t).

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* LIBRARY-FIRST sub-flag (/IMPL/FATIG/NGAUSS composing with /WVILLE + a
  kurtosis-vs-time /FUNCT) — no upstream equivalent, exactly as established for
  M16-M31's PORT cards.
* SCALAR-equivalent Hermite correction: like M24, M32 imposes the target kurtosis on
  the RESOLVED equivalent scalar (the von-Mises / critical-plane scalar the M21/M27
  reductions produce), NOT jointly on the 6x6 stress tensor. A full NON-GAUSSIAN
  INSTANTANEOUS-tensor JOINT distribution (a vector Hermite transform of the
  correlated tensor at each instant) is DEFERRED (M32 converges the M24 equivalent-
  scalar correction with the M31 continuous tensor, which is the item M31 deferred; a
  joint non-Gaussian tensor distribution is the next step). Documented, not hidden.
* The per-instant amplitude correction (1) is the NARROW-BAND-exact Winterstein
  transform wide-band-attenuated by the instantaneous alpha_2 (Benasciutti-Tovo,
  first-order — the exact Braccesi 2009 empirical constants are DEFERRED, as in M24);
  the memoryless transform preserves the per-instant PSD SHAPE only approximately (a
  static nonlinearity injects harmonics), so the higher-moment shape reuse is the
  documented approximation the M24 correction already carries.
* The non-Gaussian Monte-Carlo always uses the NON-SEPARABLE per-block synthesiser
  (per-instant blocks with a per-instant Hermite transform); the M31 constant-shape
  M25-delegation branch is used only in the Gaussian-schedule short-circuit (where the
  whole M32 path delegates to M31 for byte-identity). Documented.
* The unchanged M24-M31 deferral tail: the base-acceleration multi-input feed,
  multi-directional 100-30-30 response spectra, the arbitrary per-pair per-window
  coherence-shape card beyond M30's schedules; mean-stress beyond Goodman,
  crack-growth / fracture-mechanics fatigue, the complex-FRF stress recovery.
"""

from __future__ import annotations

import math

import numpy as np


# ============================================================================
# The kurtosis-vs-time schedule (the M32 primitive: gamma_4(t), gamma_3(t))
# ============================================================================

def _pair(v):
    """Coerce a scalar or 2-tuple ``v`` into (start, end) — a scalar is held constant
    (start == end); a (start, end) pair sweeps LINEARLY across the mission (the
    M26-M31 drifting-shape convention, reused for the kurtosis sweep)."""
    if isinstance(v, (tuple, list, np.ndarray)):
        a = np.asarray(v, dtype=float).ravel()
        if a.size >= 2:
            return float(a[0]), float(a[1])
        return float(a[0]), float(a[0])
    return float(v), float(v)


def kurtosis_schedule(s, kurt, skew=0.0, kurt_grid=None, skew_grid=None):
    """The time-varying target kurtosis gamma_4(t) / skewness gamma_3(t) sampled
    CONTINUOUSLY onto the fine instant grid (the M32 primitive). ``s`` (nt,) is the
    fine-grid mission-fraction axis (the M31 ``instantaneous_schedule`` ``s``, in
    [0, 1]).

    Each of ``kurt`` / ``skew`` may be
      * a SCALAR — held constant across the mission (the M24 stationary target — the
        constant-kurtosis limit), or
      * a (start, end) PAIR — swept LINEARLY across the mission fraction (a kurtosis
        ramp: Gaussian -> leptokurtic, or a burst rise/decay), or
      * given DIRECTLY as a per-instant array via ``kurt_grid`` / ``skew_grid`` (nt,)
        — the continuous sampling of a kurtosis-vs-time /FUNCT already evaluated on the
        fine grid (the driver samples the /FUNCT at the mission-fraction axis).

    Returns (gamma4 (nt,), gamma3 (nt,)). gamma4 == 3, gamma3 == 0 everywhere is the
    Gaussian schedule (a no-op — lambda_ng(t) == 1)."""
    s = np.asarray(s, dtype=float).ravel()
    nt = s.size
    if kurt_grid is not None:
        g4 = np.asarray(kurt_grid, dtype=float).ravel()
        if g4.size != nt:
            raise ValueError("kurt_grid must have one value per fine instant.")
    else:
        k0, k1 = _pair(kurt)
        g4 = k0 + s * (k1 - k0)
    if skew_grid is not None:
        g3 = np.asarray(skew_grid, dtype=float).ravel()
        if g3.size != nt:
            raise ValueError("skew_grid must have one value per fine instant.")
    else:
        s0, s1 = _pair(skew)
        g3 = s0 + s * (s1 - s0)
    return g4, g3


def _is_gaussian_schedule(gamma4, gamma3):
    """True when the whole schedule is Gaussian (gamma_4 == 3, gamma_3 == 0 at EVERY
    instant) — the exact gamma_4(t) == 3 limit in which the M32 path DELEGATES to the
    M31 Gaussian continuous answer BYTE-IDENTICALLY."""
    g4 = np.asarray(gamma4, dtype=float)
    g3 = np.asarray(gamma3, dtype=float)
    return bool(np.all(g4 == 3.0) and np.all(g3 == 0.0))


def instantaneous_lambda_ng(alpha2, gamma4, gamma3, m, bandwidth_correction=True,
                            model="winterstein"):
    """The per-instant NON-GAUSSIAN amplification lambda_ng(t_j) (theory eq. (1)): the
    M24 closed-form Winterstein-Hermite correction factor evaluated AT EACH INSTANT
    from that instant's bandwidth ``alpha2`` (nt,) and the time-varying target
    ``gamma4`` / ``gamma3`` (nt,). Returns lambda_ng (nt,).

    lambda_ng(t_j) = 1 EXACTLY at any Gaussian instant (gamma_4(t_j) = 3,
    gamma_3(t_j) = 0); > 1 at a leptokurtic instant; wide-band-attenuated by
    alpha_2(t_j) when ``bandwidth_correction`` (the M24 Benasciutti-Tovo model). Reuses
    ``nongaussian_fatigue.nongaussian_correction_factor`` read-only (the same closed
    form M24 uses for the stationary process, re-computed per instant)."""
    from . import nongaussian_fatigue as ngf
    a2 = np.asarray(alpha2, dtype=float).ravel()
    g4 = np.asarray(gamma4, dtype=float).ravel()
    g3 = np.asarray(gamma3, dtype=float).ravel()
    nt = a2.size
    lam = np.ones(nt)
    for j in range(nt):
        lam[j] = ngf.nongaussian_correction_factor(
            g3[j], g4[j], m, alpha2=a2[j],
            bandwidth_correction=bandwidth_correction, model=model)
    return lam


# ============================================================================
# (a) SCALAR time-varying non-Gaussian instantaneous spectrum (M24-on-M31 scalar)
# ============================================================================

_ESTIMATORS = ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti")


def _estimator_fn(name):
    from . import spectral_fatigue as sf
    return {"narrow_band": sf.narrow_band_damage, "dirlik": sf.dirlik_damage,
            "wirsching_light": sf.wirsching_light_damage,
            "tovo_benasciutti": sf.tovo_benasciutti_damage}[name]


def _corrected_estimator(gaussian_est, dr_ng, lam_repr, m, gamma3, gamma4, coeffs):
    """Assemble a corrected per-estimator result dict (the M24 ``nongaussian_damage``
    shape) from a Gaussian estimator result ``gaussian_est`` and the time-INTEGRATED
    non-Gaussian damage rate ``dr_ng`` — the life / equivalent stress follow directly
    (T_f = 1/dr, S_eq scales as (dr_ng/dr_G)^(1/m)). ``lam_repr`` is the representative
    (time-average) amplification for reporting."""
    drG = float(gaussian_est["damage_rate"])
    life = math.inf if dr_ng <= 0.0 else 1.0 / dr_ng
    ratio = (dr_ng / drG) if drG > 0 else 1.0
    s_eqG = gaussian_est.get("s_eq", 0.0)
    s_eq = (ratio ** (1.0 / m)) * s_eqG if s_eqG > 0 else 0.0
    out = dict(gaussian_est)
    out.update({"method": gaussian_est.get("method", "?") + "_nongaussian_wv",
                "damage_rate": dr_ng, "life": life, "s_eq": s_eq,
                "lambda_ng": float(lam_repr), "gaussian_damage_rate": drG,
                "h3": coeffs[0], "h4": coeffs[1], "kappa": coeffs[2],
                "gamma3": float(gamma3), "gamma4": float(gamma4)})
    return out


def nongaussian_wigner_ville_summary(freqs, base_psd, durations, fc, bw, m, C,
                                     kurt, skew=0.0, scales=None, refine=8,
                                     smooth=0.0, kurt_grid=None, skew_grid=None,
                                     bandwidth_correction=True, model="winterstein",
                                     mean_stress=0.0, ultimate=0.0):
    """The SCALAR time-varying NON-GAUSSIAN continuous instantaneous damage (theory
    eq. (*)): the M31 continuous Gaussian scalar spectrum reduced AT EACH INSTANT and
    scaled by the per-instant amplification lambda_ng(t_j), Miner-INTEGRATED over time.

    Builds the M31 continuous Gaussian summary (``wigner_ville_fatigue_summary`` —
    left byte-identical, reported alongside), samples the time-varying target
    gamma_4(t)/gamma_3(t) onto the fine grid (``kurtosis_schedule``), computes the
    per-instant alpha_2(t_j) and Gaussian per-estimator damage rate from the fine-grid
    spectrogram, and Miner-integrates lambda_ng(t_j) (dD/dt)_G(t_j).

    In the GAUSSIAN-schedule limit (gamma_4(t) == 3, gamma_3(t) == 0) DELEGATES to the
    M31 summary BYTE-IDENTICALLY. For a CONSTANT lambda_ng (constant kurtosis with the
    bandwidth attenuation off, or a stationary process) the correction FACTORS out:
    dr_nG = lambda_ng * (dr_G of M31) EXACTLY (the M24 correction on the M31 continuous
    answer). Returns a dict with per-estimator corrected results, ``gaussian`` (the
    M31 continuous summary), ``stationary_ng`` (the M24 stationary answer on the
    mission-average spectrum, for side-by-side), the lambda_ng drift diagnostics
    (``lambda_ng`` array, ``lambda_min``/``lambda_max``/``lambda_mean``), the
    gamma_4(t) schedule (``gamma4``/``gamma3`` arrays, ``gamma4_range``) and
    ``refine``/``smooth``/``nt``/``continuous``."""
    from . import wigner_ville_fatigue as wv
    from . import spectral_fatigue as sf
    from . import nongaussian_fatigue as ngf
    from . import evolutionary_fatigue as ef

    refine = max(1, int(refine))
    # 1. the M31 continuous Gaussian baseline (byte-identical, reported alongside)
    gauss = wv.wigner_ville_fatigue_summary(
        freqs, base_psd, durations, fc, bw, m, C, scales=scales, refine=refine,
        smooth=smooth, mean_stress=mean_stress, ultimate=ultimate)
    # 2. the fine-grid mission-fraction axis + durations (the schedule sampling grid)
    sch = wv.instantaneous_schedule(durations, fc, bw, scales=scales, refine=refine)
    s, T = sch["s"], sch["dur"]
    nt = sch["nt"]
    gamma4, gamma3 = kurtosis_schedule(s, kurt, skew, kurt_grid, skew_grid)

    # 3. GAUSSIAN-schedule short-circuit: DELEGATE to M31 byte-identically -----------
    if _is_gaussian_schedule(gamma4, gamma3):
        return _wrap_gaussian_scalar(gauss, gamma4, gamma3, refine, smooth, nt)

    # 4. the fine-grid Gaussian spectrogram (the same windows M31 reduced) -----------
    windows = wv.wigner_ville_spectrogram(
        freqs, base_psd, durations, fc, bw, scales=scales, refine=refine,
        smooth=smooth)
    # per-instant bandwidth alpha_2(t_j) (one channel per instant -> shared across
    # estimators) and the per-instant lambda_ng(t_j)
    alpha2 = np.array([sf.spectral_bandwidth_params(
        np.asarray(w["moments"], dtype=float))["alpha2"] for w in windows])
    lam = instantaneous_lambda_ng(alpha2, gamma4, gamma3, m,
                                  bandwidth_correction=bandwidth_correction,
                                  model=model)
    Ttot = float(np.sum(T))
    # is lambda_ng CONSTANT across instants? then the correction factors out EXACTLY
    lam_const = bool(np.all(lam == lam[0])) if nt else True

    out = {"method": "nongaussian_wigner_ville", "refine": refine,
           "smooth": float(smooth), "nt": nt, "continuous": nt != len(np.atleast_1d(
               np.asarray(durations))) or float(smooth) > 0.0,
           "gaussian": gauss, "bandwidth_correction": bool(bandwidth_correction),
           "model": model, "lambda_ng": lam, "gamma4": gamma4, "gamma3": gamma3,
           "lambda_min": float(lam.min()), "lambda_max": float(lam.max()),
           "lambda_mean": float(np.mean(lam)),
           "gamma4_range": float(np.max(gamma4) - np.min(gamma4)),
           "gamma3_range": float(np.max(gamma3) - np.min(gamma3))}
    # 5. per estimator: the Miner-INTEGRAL of lambda_ng(t_j) (dD/dt)_G(t_j) ----------
    for est in _ESTIMATORS:
        efn = _estimator_fn(est)
        drG_j = np.array([float(efn(np.asarray(w["moments"], dtype=float), m, C,
                                    mean_stress, ultimate)["damage_rate"])
                          for w in windows])
        if lam_const:
            # EXACT M24 factoring: dr_nG = lambda_ng * (M31 continuous dr_G). Use the
            # M31 aggregate so it is bit-for-bit the M24 correction on the M31 answer.
            dr_ng = float(lam[0]) * float(gauss[est]["damage_rate"])
        else:
            # genuine time-varying Miner integral (eq. (*))
            dr_ng = float(np.sum(lam * drG_j * T) / Ttot) if Ttot > 0 else 0.0
        # representative (time-average) coefficients for reporting
        g4bar = float(np.mean(gamma4))
        g3bar = float(np.mean(gamma3))
        coeffs = ngf.hermite_coefficients(g3bar, g4bar, model=model)
        out[est] = _corrected_estimator(gauss[est], dr_ng, out["lambda_mean"], m,
                                         g3bar, g4bar, coeffs)
    out["damage_rate"] = out["dirlik"]["damage_rate"]
    out["life"] = out["dirlik"]["life"]

    # 6. the M24 STATIONARY answer on the mission-average spectrum (side-by-side) ----
    marg = gauss.get("instantaneous")
    if marg is not None and marg.get("avg_psd") is not None:
        avg_mom = ef._moments_of_psd(np.asarray(freqs, dtype=float),
                                     np.asarray(marg["avg_psd"], dtype=float))
        out["stationary_ng"] = ngf.nongaussian_summary(
            avg_mom, m, C, float(np.mean(gamma3)), float(np.mean(gamma4)),
            mean_stress=mean_stress, ultimate=ultimate,
            bandwidth_correction=bandwidth_correction, model=model)
    else:
        out["stationary_ng"] = None
    return out


def _wrap_gaussian_scalar(gauss, gamma4, gamma3, refine, smooth, nt):
    """Wrap the M31 Gaussian continuous summary as the M32 result in the exact
    gamma_4(t) == 3 limit — lambda_ng(t) == 1 everywhere, so every non-Gaussian
    estimator rate IS the M31 Gaussian rate BYTE-IDENTICALLY (the delegation)."""
    lam = np.ones(int(nt))
    out = {"method": "nongaussian_wigner_ville", "refine": int(refine),
           "smooth": float(smooth), "nt": int(nt), "continuous": False,
           "gaussian": gauss, "bandwidth_correction": True, "model": "winterstein",
           "lambda_ng": lam, "gamma4": np.asarray(gamma4, dtype=float),
           "gamma3": np.asarray(gamma3, dtype=float), "lambda_min": 1.0,
           "lambda_max": 1.0, "lambda_mean": 1.0, "gamma4_range": 0.0,
           "gamma3_range": 0.0, "delegated": "m31_gaussian", "stationary_ng": None}
    for est in _ESTIMATORS:
        g = gauss[est]
        r = dict(g)
        r.update({"lambda_ng": 1.0, "gaussian_damage_rate": float(g["damage_rate"]),
                  "gamma3": 0.0, "gamma4": 3.0})
        out[est] = r
    out["damage_rate"] = out["dirlik"]["damage_rate"]
    out["life"] = out["dirlik"]["life"]
    return out


# ============================================================================
# (b) 6x6 TENSOR time-varying non-Gaussian instantaneous spectrum (M24-on-M31 tensor)
# ============================================================================

def _reduce_instant_tensors_ng(omega, Scross, weights, inst_scales, durations,
                               gamma4, gamma3, m, C, mean_stress, ultimate, naz,
                               npol, bandwidth_correction, model, drift=True):
    """Reduce a fine-grid instantaneous TENSOR spectrum with the per-instant
    non-Gaussian correction: per instant form the 6x6 windowed moment matrices M_{n,j}
    (``windowed_tensor_moment_matrices`` with the per-instant window ``weights[j]`` and
    RMS level ``inst_scales[j]``), RE-SEARCH the critical plane / F_np from THAT
    instant's tensor (``reduce_window_tensor`` — the plane drifts CONTINUOUSLY), and
    for EACH reduction scale the per-instant Gaussian damage rate by that reduction's
    own per-instant lambda_ng(t_j) (from its per-instant alpha_2(t_j) and the target
    gamma_4(t_j)/gamma_3(t_j)) before Miner-INTEGRATING.

    ``weights`` / ``inst_scales`` — the (nt, nf) per-instant window weights and the
    per-instant RMS levels; passing the M31 effective windows with unit scale gives
    the CONTINUOUS reduction, passing the raw M27 per-window weight W_i with scale a_i
    gives the WINDOWED reduction byte-identical to M27. Returns the M27-summary dict
    shape PLUS the per-reduction / per-instant lambda_ng diagnostics."""
    from . import spectral_fatigue as sf
    from .joint_evolutionary_fatigue import (windowed_tensor_moment_matrices,
                                             reduce_window_tensor,
                                             _reduce_window_fixed)
    from .multiaxial_fatigue import (tensor_moment_matrices,
                                     equivalent_vonmises_moments)
    from . import nongaussian_fatigue as ngf
    Scross = np.asarray(Scross)
    T = np.asarray(durations, dtype=float).ravel()
    g4 = np.asarray(gamma4, dtype=float).ravel()
    g3 = np.asarray(gamma3, dtype=float).ravel()
    nt = len(weights)
    # stationary reference (the un-windowed tensor) for the drift=False fixed plane
    Mstat = tensor_moment_matrices(omega, Scross, nmax=4)
    stat = reduce_window_tensor(Mstat, m, C, mean_stress=mean_stress,
                                ultimate=ultimate, naz=naz, npol=npol)
    keys = ("von_mises", "normal_plane", "shear_plane")
    Dng = {k: 0.0 for k in keys}
    Dg = {k: 0.0 for k in keys}
    lam_track = {k: [] for k in keys}
    wout = []
    shear_normals = []
    normal_normals = []
    fnps = []
    shapes = []
    for j in range(nt):
        Mi = windowed_tensor_moment_matrices(omega, Scross, weights[j],
                                             scale=float(inst_scales[j]), nmax=4)
        if drift:
            red = reduce_window_tensor(Mi, m, C, mean_stress=mean_stress,
                                       ultimate=ultimate, naz=naz, npol=npol)
        else:
            red = _reduce_window_fixed(Mi, stat, m, C, mean_stress, ultimate)
        Ti = float(T[j])
        for k in keys:
            summ = red[k]["summary"]
            a2 = float(summ["params"]["alpha2"])
            lam = ngf.nongaussian_correction_factor(
                g3[j], g4[j], m, alpha2=a2,
                bandwidth_correction=bandwidth_correction, model=model)
            drG = float(red[k]["damage_rate"])           # Dirlik per-instant Gaussian
            Dg[k] += drG * Ti
            Dng[k] += lam * drG * Ti
            lam_track[k].append(lam)
        shear_normals.append(np.asarray(red["shear_plane"]["normal"], dtype=float))
        normal_normals.append(np.asarray(red["normal_plane"]["normal"],
                                         dtype=float))
        fnps.append(float(red["F_np"]))
        m0 = equivalent_vonmises_moments(Mi)[0]
        shapes.append(Mi[0] / m0 if m0 > 0 else Mi[0])
        wout.append({"duration": Ti, "F_np": fnps[-1],
                     "sigma_vm": float(red["sigma_vm"]),
                     "vm_rate": float(red["von_mises"]["damage_rate"]),
                     "vm_lambda": lam_track["von_mises"][-1],
                     "normal_rate": float(red["normal_plane"]["damage_rate"]),
                     "shear_rate": float(red["shear_plane"]["damage_rate"])})
    Ttot = float(np.sum(T))
    const = len(shapes) <= 1 or all(
        np.allclose(shapes[k], shapes[0], rtol=1e-9, atol=1e-12)
        for k in range(1, len(shapes)))

    def _rot(normals):
        n0 = normals[0]
        best = 0.0
        for n in normals[1:]:
            c = abs(float(np.dot(n0, n)))
            best = max(best, math.degrees(math.acos(min(1.0, c))))
        return best
    rot = 0.0 if const else max(_rot(shear_normals), _rot(normal_normals))
    fnp_drift = float(np.max(fnps) - np.min(fnps)) if fnps else 0.0
    lam_all = np.array([v for lst in lam_track.values() for v in lst], dtype=float)
    out = {"method": "nongaussian_wigner_ville_tensor", "nt": nt,
           "drift": bool(drift), "constant_shape": bool(const),
           "plane_rotation_deg": rot, "fnp_drift": fnp_drift, "windows": wout,
           "stationary": stat, "total_time": Ttot,
           "lambda_min": float(lam_all.min()) if lam_all.size else 1.0,
           "lambda_max": float(lam_all.max()) if lam_all.size else 1.0,
           "lambda_mean": float(np.mean(lam_all)) if lam_all.size else 1.0}
    Ceff = sf._goodman_C(C, m, mean_stress, ultimate)
    for k in keys:
        drg = Dg[k] / Ttot if Ttot > 0 else 0.0
        lams = np.asarray(lam_track[k], dtype=float)
        # EXACT factoring when this reduction's lambda_ng is CONSTANT across instants
        # (constant kurtosis with the bandwidth attenuation off, or a stationary
        # tensor): dr_nG = lambda_ng * dr_G, bit-for-bit the M24 correction on the
        # M31/M27 Gaussian answer (whose per-instant reduction is byte-identical).
        if lams.size and np.all(lams == lams[0]):
            dr = float(lams[0]) * drg
        else:
            dr = Dng[k] / Ttot if Ttot > 0 else 0.0
        tf, _ = sf.life_and_equivalent(dr, 0.0, m, Ceff)
        out[k] = {"damage": Dng[k], "total_time": Ttot, "damage_rate": dr,
                  "life": tf, "gaussian_damage_rate": drg,
                  "lambda_ng": float(np.mean(lam_track[k])) if lam_track[k] else 1.0}
    out["damage_rate"] = out["von_mises"]["damage_rate"]
    out["life"] = out["von_mises"]["life"]
    return out


def nongaussian_wigner_ville_tensor_summary(omega, Scross, durations, fc, bw, m, C,
                                            kurt, skew=0.0, scales=None, refine=8,
                                            smooth=0.0, kurt_grid=None,
                                            skew_grid=None,
                                            bandwidth_correction=True,
                                            model="winterstein", mean_stress=0.0,
                                            ultimate=0.0, naz=24, npol=13,
                                            drift=True):
    """The 6x6 TENSOR time-varying NON-GAUSSIAN continuous instantaneous damage
    (theory eq. (*)): the M31 continuous Gaussian TENSOR spectrum reduced with a
    per-instant critical-plane re-search AND scaled by each reduction's per-instant
    lambda_ng(t_j), Miner-INTEGRATED.

    Builds the M31 Gaussian tensor summary (``wigner_ville_tensor_summary`` — left
    byte-identical, reported alongside). In the GAUSSIAN-schedule limit DELEGATES to
    it byte-identically. In the WINDOWED limit (refine = 1, smooth = 0) the per-instant
    reduction rebuilds the M27 per-window moment matrices (weight W_i, scale a_i) so
    the Gaussian per-window reduction is byte-identical to M27 before the per-window
    lambda_ng scales it — the M24 correction applied to the M27 windowed answer.
    Otherwise builds the M31 fine-grid effective windows and reduces per instant with
    the per-instant lambda_ng. Returns the M27 summary dict shape PLUS the per-reduction
    gaussian_damage_rate / lambda_ng, the lambda drift diagnostics and the
    gamma_4(t) schedule."""
    from . import wigner_ville_fatigue as wv
    from .joint_evolutionary_fatigue import joint_evolutionary_windows
    freqs = np.asarray(omega, dtype=float) / (2.0 * np.pi)
    refine = max(1, int(refine))
    gauss = wv.wigner_ville_tensor_summary(
        omega, Scross, durations, fc, bw, m, C, scales=scales, refine=refine,
        smooth=smooth, mean_stress=mean_stress, ultimate=ultimate, naz=naz,
        npol=npol, drift=drift)
    sch = wv.instantaneous_schedule(durations, fc, bw, scales=scales, refine=refine)
    s = sch["s"]
    gamma4, gamma3 = kurtosis_schedule(s, kurt, skew, kurt_grid, skew_grid)

    if _is_gaussian_schedule(gamma4, gamma3):
        return _wrap_gaussian_tensor(gauss, gamma4, gamma3, refine, smooth)

    if wv.is_windowed_limit(refine, smooth):
        # WINDOWED limit: rebuild the M27 per-window weight/scale so the Gaussian
        # per-window reduction is byte-identical to M27 before lambda_ng scales it
        win = joint_evolutionary_windows(freqs, durations, fc, bw, scales)
        weights = [np.asarray(w["W"], dtype=float) for w in win]
        inst_scales = [float(w["scale"]) for w in win]
        dur = np.asarray([float(w["duration"]) for w in win], dtype=float)
    else:
        eff = wv.instantaneous_effective_windows(
            freqs, durations, fc, bw, scales=scales, refine=refine, smooth=smooth)
        weights = [eff["Weff"][j] for j in range(eff["nt"])]
        inst_scales = [1.0] * eff["nt"]         # scale folded into Weff (M31)
        dur = eff["dur"]
    out = _reduce_instant_tensors_ng(
        omega, Scross, weights, inst_scales, dur, gamma4, gamma3, m, C, mean_stress,
        ultimate, naz, npol, bandwidth_correction, model, drift=drift)
    out["refine"] = refine
    out["smooth"] = float(smooth)
    out["continuous"] = not wv.is_windowed_limit(refine, smooth)
    out["gaussian"] = gauss
    out["gamma4"] = gamma4
    out["gamma3"] = gamma3
    out["gamma4_range"] = float(np.max(gamma4) - np.min(gamma4))
    out["bandwidth_correction"] = bool(bandwidth_correction)
    out["peak_drift"] = gauss.get("peak_drift", 0.0)
    return out


def _wrap_gaussian_tensor(gauss, gamma4, gamma3, refine, smooth):
    """Wrap the M31 Gaussian continuous tensor summary as the M32 result in the exact
    gamma_4(t) == 3 limit — lambda_ng == 1, every reduction rate IS the M31 Gaussian
    rate BYTE-IDENTICALLY."""
    out = dict(gauss)
    out["method"] = "nongaussian_wigner_ville_tensor"
    out["delegated"] = "m31_gaussian"
    out["refine"] = int(refine)
    out["smooth"] = float(smooth)
    out["gamma4"] = np.asarray(gamma4, dtype=float)
    out["gamma3"] = np.asarray(gamma3, dtype=float)
    out["gamma4_range"] = 0.0
    out["lambda_min"] = 1.0
    out["lambda_max"] = 1.0
    out["lambda_mean"] = 1.0
    out["bandwidth_correction"] = True
    for k in ("von_mises", "normal_plane", "shear_plane"):
        r = dict(gauss[k])
        r["gaussian_damage_rate"] = float(gauss[k]["damage_rate"])
        r["lambda_ng"] = 1.0
        out[k] = r
    return out


# ============================================================================
# (c) The non-Gaussian NON-STATIONARY Monte-Carlo cross-check (build item 2)
# ============================================================================

def synthesize_nongaussian_wigner_ville_history(freqs, base_psd, durations, fc, bw,
                                                seed, kurt, skew=0.0, scales=None,
                                                refine=8, smooth=0.0, kurt_grid=None,
                                                skew_grid=None, fs=None,
                                                model="winterstein"):
    """Synthesise the NON-GAUSSIAN NON-STATIONARY record (theory "NON-GAUSSIAN
    NON-STATIONARY MONTE-CARLO"): the M31 continuous non-separable synthesiser on the
    fine instant grid (per-instant spectral-representation blocks) with EACH block
    pushed through the M24 memoryless Winterstein-Hermite transform to THAT instant's
    target gamma_4(t_j) / gamma_3(t_j), concatenated so the record's LOCAL kurtosis
    tracks gamma_4(t). Returns (t, x, info) with ``info`` = {edges, fs, gamma4,
    gamma3}.

    In the Gaussian-schedule limit every block transform is the identity, so the
    record reduces EXACTLY to the M31 continuous non-separable history (the caller
    delegates to the M31 Monte-Carlo there for bit-identity)."""
    from . import wigner_ville_fatigue as wv
    from . import spectral_fatigue as sf
    from . import nongaussian_fatigue as ngf
    refine = max(1, int(refine))
    windows = wv.wigner_ville_spectrogram(
        freqs, base_psd, durations, fc, bw, scales=scales, refine=refine,
        smooth=smooth)
    sch = wv.instantaneous_schedule(durations, fc, bw, scales=scales, refine=refine)
    gamma4, gamma3 = kurtosis_schedule(sch["s"], kurt, skew, kurt_grid, skew_grid)
    fmax = max(float(np.max(np.asarray(w["freqs"], dtype=float))) for w in windows)
    if fs is None:
        fs = 8.0 * fmax
    T = np.array([float(w["duration"]) for w in windows], dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(T)])
    chunks = []
    for j, w in enumerate(windows):
        _t, xi = sf.synthesize_gaussian_history(
            np.asarray(w["freqs"], dtype=float), np.asarray(w["psd"], dtype=float),
            float(w["duration"]), int(seed) + j, fs=fs)
        coeffs = ngf.hermite_coefficients(float(gamma3[j]), float(gamma4[j]),
                                          model=model)
        h3, h4, kappa = coeffs
        # per-instant memoryless Hermite transform (identity at a Gaussian instant)
        if not (abs(h3) < 1e-15 and abs(h4) < 1e-15 and abs(kappa - 1.0) < 1e-15):
            sig = float(np.std(xi))
            if sig > 0.0:
                z = (xi - float(np.mean(xi))) / sig
                xi = sig * ngf.hermite_transform(z, float(gamma3[j]),
                                                 float(gamma4[j]), coeffs=coeffs)
        chunks.append(xi)
    x = np.concatenate(chunks) if chunks else np.zeros(0)
    t = np.arange(x.size) / fs
    return t, x, {"edges": edges, "fs": fs, "gamma4": gamma4, "gamma3": gamma3}


def nongaussian_wigner_ville_monte_carlo_damage(freqs, base_psd, durations, fc, bw,
                                                m, C, seed, kurt, skew=0.0,
                                                scales=None, refine=8, smooth=0.0,
                                                kurt_grid=None, skew_grid=None,
                                                fs=None, mean_stress=0.0,
                                                ultimate=0.0, model="winterstein"):
    """The SCALAR time-varying NON-GAUSSIAN damage by Monte-Carlo (theory "NON-GAUSSIAN
    NON-STATIONARY MONTE-CARLO"): synthesise the non-Gaussian non-stationary record
    (``synthesize_nongaussian_wigner_ville_history``), rainflow-count the WHOLE
    concatenated record (ASTM E1049) and Miner-sum — the reference the time-varying
    Miner-integral (*) converges to. In the Gaussian-schedule limit DELEGATES to the
    M31 continuous Monte-Carlo BIT-IDENTICALLY. Returns the M31 Monte-Carlo dict shape
    plus the sample ``skewness`` / ``kurtosis`` of the record (tracking gamma_4(t))."""
    from . import wigner_ville_fatigue as wv
    from . import spectral_fatigue as sf
    refine = max(1, int(refine))
    # sample the schedule to detect the Gaussian limit (delegate to M31 there)
    sch = wv.instantaneous_schedule(durations, fc, bw, scales=scales, refine=refine)
    gamma4, gamma3 = kurtosis_schedule(sch["s"], kurt, skew, kurt_grid, skew_grid)
    if _is_gaussian_schedule(gamma4, gamma3):
        mc = wv.wigner_ville_monte_carlo_damage(
            freqs, base_psd, durations, fc, bw, m, C, seed, scales=scales,
            refine=refine, smooth=smooth, fs=fs, mean_stress=mean_stress,
            ultimate=ultimate)
        mc = dict(mc)
        mc["method"] = "nongaussian_wigner_ville_monte_carlo"
        x = None
    else:
        Ceff = sf._goodman_C(C, m, mean_stress, ultimate)
        t, x, info = synthesize_nongaussian_wigner_ville_history(
            freqs, base_psd, durations, fc, bw, seed, kurt, skew=skew,
            scales=scales, refine=refine, smooth=smooth, kurt_grid=kurt_grid,
            skew_grid=skew_grid, fs=fs, model=model)
        ranges, counts = sf.rainflow_count(x)
        D = float(np.sum(counts * ranges ** m) / Ceff) if ranges.size else 0.0
        Ttot = t[-1] - t[0] if t.size > 1 else float(np.sum(durations))
        dr = D / Ttot if Ttot > 0 else 0.0
        tf, s_eq = sf.life_and_equivalent(
            dr, ranges.size / Ttot if Ttot > 0 else 0.0, m, Ceff)
        mc = {"method": "nongaussian_wigner_ville_monte_carlo", "damage_rate": dr,
              "life": tf, "s_eq": s_eq, "ncycles": float(counts.sum()),
              "ranges": ranges, "counts": counts, "duration": Ttot,
              "rms": float(np.std(x))}
    if x is not None:
        xc = x - np.mean(x)
        var = float(np.mean(xc * xc))
        mc["skewness"] = float(np.mean(xc ** 3) / var ** 1.5) if var > 0 else 0.0
        mc["kurtosis"] = float(np.mean(xc ** 4) / var ** 2) if var > 0 else 3.0
    else:
        mc.setdefault("kurtosis", 3.0)
        mc.setdefault("skewness", 0.0)
    mc["refine"] = refine
    mc["smooth"] = float(smooth)
    return mc

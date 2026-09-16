"""
Spectral (frequency-domain) fatigue — M20: the stress-life fatigue-damage
estimate of a stationary random-vibration response, computed DIRECTLY from the
spectral moments (m0..m4) of a stress Power Spectral Density.

Fortran origin
--------------
There is NO frequency-domain / spectral-fatigue solver anywhere in the
open-source OpenRadioss engine. ``engine/source/input/freimpl.F`` (the /IMPL
reader, re-read line by line for M16-M19 AND again for M20) parses only
/IMPL/DYNA (the DIRECT Newmark/HHT integrator), /IMPL/BUCKL, /IMPL/DT,
/IMPL/NONLIN and /IMPL/ARCL plus the linear-solver housekeeping — there is no
/FATIG, no S-N / Miner branch, no Dirlik / rainflow / narrow-band damage
estimator, and (as M19 already found) the sole ``PSD`` token in the whole file
is ``IMUMPSD``, a MUMPS-solver flag, not a power spectral density. OpenRadioss
is a time-domain crash/impact code: the stationary random-vibration fatigue
analysis is simply not part of the open-source solver — exactly the finding
M16 made for the real eigensolver, M17 for mode superposition, M18 for the
complex modes and M19 for the PSD / spectral-moment machinery this module
consumes.

So M20 does exactly what M16-M19 did: it ports spectral fatigue as a clean
LIBRARY capability and drives it with a minimal PORT engine card (/IMPL/FATIG —
the fatigue analogue of M16's /IMPL/EIGV and M19's /IMPL/PSD). Nothing in the
M10 direct integrator, the M16 REAL eigensolver, the M17/M18 superposition or
the M19 PSD / response-spectrum paths is touched: the fatigue path is a NEW,
parallel path that CONSUMES the M19 stress-PSD spectral moments read-only.

Theory — random-vibration (spectral) fatigue
--------------------------------------------
(Bendat, "Probability Functions for Random Responses" NASA CR-33, 1964 — the
narrow-band estimate; Dirlik, "Application of computers in fatigue analysis",
PhD thesis, University of Warwick, 1985 — the empirical rainflow-range PDF;
Wirsching & Light, "Fatigue under wide band random stresses", J. Struct. Div.
ASCE 106, 1980; Benasciutti & Tovo, "Spectral methods for lifetime prediction
under wide-band stationary random processes", Int. J. Fatigue 27, 2005;
Newland, "An Introduction to Random Vibrations…" ch. 5-7; Bishop & Sherratt,
"Finite Element Based Fatigue Calculations", NAFEMS 2000; Halfpenny, "A gentle
introduction to the frequency-domain fatigue analysis", 1999.)

STRESS-LIFE (S-N) + MINER. A material's stress-life curve is the power law

    N = C * S**(-m)          <=>    N * S**m = C                          (1)

(N cycles to failure at a constant stress RANGE S; m the negative log-log slope,
C the fatigue strength coefficient). By the Palmgren-Miner linear-damage rule
the damage of ONE cycle of range S is 1/N = S**m / C, and cycles ACCUMULATE
linearly: failure at cumulative damage D = sum(1/N_i) = 1. For a random stress
history the ranges S are a RANDOM VARIABLE with a probability density p(S), and
the expected damage RATE (damage per unit time) is

    E[D]/T = (nu_p / C) * integral_0^inf S**m p(S) dS
           = (nu_p / C) * E[S**m]                                         (2)

where nu_p is the rate at which cycles occur (the peak / rainflow rate). The
time to failure is then T_f = 1 / (E[D]/T) (Miner: D = 1), and the equivalent
constant-amplitude stress range that produces the same damage at the same
cycle rate is S_eq = (C * (E[D]/T) / nu_p)**(1/m) — the standard fatigue
reporting triple (damage rate, equivalent stress, life).

The whole game of a SPECTRAL method is to get p(S) — the distribution of
rainflow ranges — from the STRESS-PSD spectral moments alone, without ever
building a time history. The moments (M19 convention, this module's input) are

    m_n = (1/pi) integral_0^inf Omega**n S_sigma(Omega) dOmega            (3)

so m_0 = sigma^2 is the stress VARIANCE (RMS sigma = sqrt(m_0)), and the
crossing / peak RATES are Rice's

    nu_0 = (1/2pi) sqrt(m_2/m_0)   [Hz]  (mean zero-up-crossing rate)
    nu_p = (1/2pi) sqrt(m_4/m_2)   [Hz]  (mean peak / local-maxima rate)  (4)

with the two spectral-width / irregularity factors

    alpha_1 = m_1 / sqrt(m_0 m_2) ,   alpha_2 = m_2 / sqrt(m_0 m_4)       (5)

(alpha_2 = nu_0/nu_p is the irregularity factor: 1 for a narrow-band process,
-> 0 for white noise; note (3)'s 1/pi cancels in every ratio (5), so the width
factors are convention-independent — see ``spectral_bandwidth_params``).

NARROW-BAND (Bendat 1964). If the process is NARROW-BAND (alpha_2 -> 1) every
peak is a full cycle, cycles occur at nu_0 = nu_p, and the peaks follow a
RAYLEIGH distribution of parameter sigma. The rainflow RANGE is then twice the
peak amplitude, S = 2A, Rayleigh with p(S) = (S/(4 sigma^2)) exp(-S^2/(8
sigma^2)), whose m-th moment is the CLOSED FORM

    E[S**m] = (2 sqrt(2) sigma)**m * Gamma(1 + m/2)                       (6)

so the narrow-band damage rate is (Bendat; Miles' equation's fatigue sibling)

    (E[D]/T)_NB = (nu_0 / C) * (2 sqrt(2) sigma)**m * Gamma(1 + m/2)      (7)

For a WIDE-BAND process (7) is CONSERVATIVE (it over-counts: every peak is
wrongly paired with the mean, ignoring the small ripples rainflow merges into
larger cycles) — sometimes by an order of magnitude. The corrections below fix
the range distribution.

DIRLIK (1985). Dirlik fitted, by Monte-Carlo over thousands of spectra, an
EMPIRICAL closed-form rainflow-range PDF as one exponential + two Rayleigh
terms. With the normalised range Z = S / (2 sigma), x_m = (m_1/m_0)
sqrt(m_2/m_4), gamma = alpha_2 = m_2/sqrt(m_0 m_4),

    p(S) = 1/(2 sigma) [ D1/Q e^{-Z/Q}
                       + D2 Z/R^2 e^{-Z^2/(2R^2)}
                       + D3 Z e^{-Z^2/2} ]                                (8)
    D1 = 2(x_m - gamma^2)/(1 + gamma^2)
    R  = (gamma - x_m - D1^2)/(1 - gamma - D1 + D1^2)
    D2 = (1 - gamma - D1 + D1^2)/(1 - R)
    D3 = 1 - D1 - D2
    Q  = 1.25 (gamma - D3 - D2 R)/D1

The rainflow rate is the PEAK rate nu_p, and (8)'s m-th moment is again a
closed form (an exponential -> Gamma(1+m), two Rayleigh -> Gamma(1+m/2)):

    E[S**m] = (2 sigma)**m [ D1 Q**m Gamma(1+m)
                           + (sqrt(2) R)**m D2 Gamma(1+m/2)
                           + (sqrt(2))**m   D3 Gamma(1+m/2) ]             (9)

so (E[D]/T)_DK = (nu_p/C) E[S**m]. Dirlik REDUCES to the narrow-band estimate
in the narrow-band limit (gamma -> 1: D1, D2 -> 0, D3 -> 1, the third term
alone survives and (9) -> (2 sigma)**m sqrt(2)**m Gamma(1+m/2) = (2 sqrt(2)
sigma)**m Gamma(1+m/2), with nu_p -> nu_0 — exactly (6)/(7)); it is the de-facto
industry-standard wide-band estimator.

WIRSCHING-LIGHT (1980). A closed-form empirical CORRECTION FACTOR on the
narrow-band damage, as a function of the spectral width epsilon = sqrt(1 -
alpha_2^2) and the slope m:

    lambda_WL = a(m) + [1 - a(m)] (1 - epsilon)**b(m)                     (10)
    a(m) = 0.926 - 0.033 m ,   b(m) = 1.587 m - 2.323

    (E[D]/T)_WL = lambda_WL * (E[D]/T)_NB                                 (11)

lambda_WL -> 1 as epsilon -> 0 (narrow band), and < 1 for a wide band (it
RELAXES the conservative narrow-band estimate). Provided as an independent
cross-check of Dirlik.

TOVO-BENASCIUTTI (2005). A second wide-band correction, a linear combination
of the narrow-band (upper-bound) and range-counting (lower-bound) damages
weighted by a bandwidth coefficient b in [0,1]:

    (E[D]/T)_TB = [ b + (1 - b) alpha_2**(m-1) ] * alpha_2 * (E[D]/T)_NB' (12)

with (E[D]/T)_NB' the narrow-band damage taken at the PEAK rate nu_p and the
approximate weight (Benasciutti & Tovo 2005 eq. for b)

    b = (alpha_1 - alpha_2) *
        [1.112 (1 + alpha_1 alpha_2 - (alpha_1 + alpha_2)) e^{2.11 alpha_2}
         + (alpha_1 - alpha_2)] / (alpha_2 - 1)**2                        (13)

reported as a further cross-check. (12) also collapses to the narrow-band
estimate for alpha_1 = alpha_2 -> 1.

MONTE-CARLO cross-check. As an independent validation the module can SYNTHESISE
a Gaussian time history from the stress PSD (spectral representation: a sum of
harmonics with the PSD-set amplitudes and random phases), RAINFLOW-count it
(ASTM E1049 three-point / four-point algorithm) and Miner-sum the counted
ranges — the time-domain answer the spectral methods approximate. It agrees
with Dirlik to within the (documented, ~10-30 %) empirical scatter of the
method plus the statistical scatter of a finite record. Randomness is seeded
explicitly (a fixed seed argument) so the check is reproducible.

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* LIBRARY-FIRST card (/IMPL/FATIG) — no upstream equivalent, exactly as
  established for M16's /IMPL/EIGV, M17's /IMPL/MODAL, M18's /IMPL/CEIGV and
  M19's /IMPL/PSD.
* UNIAXIAL / SCALAR stress channel. The damage is computed from a SINGLE scalar
  stress (or stress-resultant) component's PSD — the stress-recovery path picks
  the critical element/component. MULTIAXIAL / critical-plane fatigue and
  stress-invariant (von Mises / signed-von-Mises) equivalent PSDs are DEFERRED
  (they need the full stress-tensor cross-PSD matrix and a critical-plane
  search, beyond a scalar component).
* MEAN STRESS. A basic mean-stress option is provided (a Goodman correction of
  the S-N intercept from a static mean stress); Gerber / Soderberg / Walker
  and a per-cycle mean from the rainflow pairing are DEFERRED.
* STATIONARY, GAUSSIAN response only (the |H|^2 S law and every PDF above assume
  a stationary Gaussian process). NON-stationary / evolutionary-PSD fatigue and
  non-Gaussian (kurtosis) corrections are DEFERRED — they build on the same
  moments.
* CRACK-GROWTH / fracture-mechanics fatigue (Paris law, spectral da/dN) is a
  different analysis entirely and is DEFERRED.
"""

from __future__ import annotations

import math

import numpy as np


# ============================================================================
# Spectral-width parameters from the moments (build-order item 1)
# ============================================================================

def spectral_bandwidth_params(moments):
    """The fatigue-relevant spectral descriptors from the moment array
    ``moments`` = [m_0, m_1, m_2, m_3, m_4] (the M19 ``spectral_moments``
    output for ONE channel, convention m_n = (1/pi) int Omega^n S dOmega):

        sigma   = sqrt(m_0)                     RMS stress
        nu_0    = (1/2pi) sqrt(m_2/m_0)   [Hz]  mean zero-up-crossing rate (4)
        nu_p    = (1/2pi) sqrt(m_4/m_2)   [Hz]  mean peak rate (4)
        alpha_1 = m_1 / sqrt(m_0 m_2)           first width factor (5)
        alpha_2 = m_2 / sqrt(m_0 m_4)           irregularity factor (5)
        x_m     = (m_1/m_0) sqrt(m_2/m_4)       Dirlik mean-frequency ratio
        epsilon = sqrt(1 - alpha_2^2)           spectral bandwidth (0=narrow)

    Returned as a dict. The rate/width ratios are convention-independent (the
    1/pi of (3) cancels); only sigma and the rates carry the moment units."""
    m = np.asarray(moments, dtype=float)
    if m.shape[0] < 5:
        raise ValueError(
            "spectral fatigue needs the moments m_0..m_4 (5 values); got "
            f"{m.shape[0]} — call spectral_moments(..., nmax=4).")
    m0, m1, m2, m3, m4 = (float(m[i]) for i in range(5))
    sigma = math.sqrt(max(m0, 0.0))
    # guard the degenerate cases (a null response, or a grid too coarse to
    # carry m_4) with finite, honest fall-backs rather than NaNs
    nu0 = math.sqrt(max(m2 / m0, 0.0)) / (2.0 * math.pi) if m0 > 0 else 0.0
    nup = math.sqrt(max(m4 / m2, 0.0)) / (2.0 * math.pi) if m2 > 0 else 0.0
    alpha1 = m1 / math.sqrt(m0 * m2) if (m0 > 0 and m2 > 0) else 0.0
    alpha2 = m2 / math.sqrt(m0 * m4) if (m0 > 0 and m4 > 0) else 0.0
    x_m = (m1 / m0) * math.sqrt(m2 / m4) if (m0 > 0 and m4 > 0) else 0.0
    # alpha_2 is a correlation-like factor and must live in [0, 1]; round-off
    # on a near-narrow-band spectrum can nudge it a hair past 1
    alpha1 = min(max(alpha1, 0.0), 1.0)
    alpha2 = min(max(alpha2, 0.0), 1.0)
    epsilon = math.sqrt(max(1.0 - alpha2 * alpha2, 0.0))
    return {"sigma": sigma, "nu0": nu0, "nup": nup, "alpha1": alpha1,
            "alpha2": alpha2, "x_m": x_m, "epsilon": epsilon,
            "m0": m0, "m1": m1, "m2": m2, "m3": m3, "m4": m4}


# ============================================================================
# S-N helpers: from a damage RATE to the reported (life, equivalent stress)
# ============================================================================

def _goodman_C(C, m, mean_stress, ultimate):
    """Goodman mean-stress correction of the S-N intercept: a static mean
    stress sigma_m knocks the allowable range down by (1 - sigma_m/S_u), so
    the effective coefficient is C_eff = C (1 - sigma_m/S_u)**m (the amplitude
    axis is scaled, and N = C S^-m scales as C). ``ultimate`` = S_u the
    ultimate tensile strength; a zero/None mean or ultimate leaves C
    unchanged. Basic option only — Gerber/Soderberg/Walker are deferred."""
    if not mean_stress or not ultimate or ultimate <= 0.0:
        return C
    fac = 1.0 - float(mean_stress) / float(ultimate)
    if fac <= 0.0:
        # the mean stress alone exceeds the ultimate -> immediate failure
        return 1e-300
    return C * fac ** m


def life_and_equivalent(damage_rate, nu, m, C):
    """From a damage RATE (damage per unit time) return the reporting triple
    fields time-to-failure and equivalent constant-amplitude stress range:

        T_f   = 1 / damage_rate                 (Miner: failure at D = 1)
        S_eq  = (C * damage_rate / nu)**(1/m)   (the constant range that at the
                cycle rate ``nu`` reproduces the same damage rate)

    ``nu`` is the cycle rate the equivalent stress is referred to (nu_0 for the
    narrow-band estimate, nu_p for the wide-band ones)."""
    dr = float(damage_rate)
    tf = math.inf if dr <= 0.0 else 1.0 / dr
    if nu > 0.0 and dr > 0.0:
        s_eq = (C * dr / nu) ** (1.0 / m)
    else:
        s_eq = 0.0
    return tf, s_eq


# ============================================================================
# The damage estimators (build-order item 2)
# ============================================================================

def narrow_band_damage(moments, m, C, mean_stress=0.0, ultimate=0.0):
    """The NARROW-BAND (Bendat 1964) damage rate (theory eq. (7)):

        E[D]/T = (nu_0 / C) (2 sqrt(2) sigma)**m Gamma(1 + m/2)

    the closed-form Rayleigh-range integral (6). Cycles at the mean
    zero-up-crossing rate nu_0; CONSERVATIVE for a wide-band spectrum. Returns
    a dict {damage_rate, life, s_eq, nu, E_Sm, ...}. ``mean_stress`` applies
    the basic Goodman intercept correction (``ultimate`` = S_u)."""
    p = spectral_bandwidth_params(moments)
    sigma, nu0 = p["sigma"], p["nu0"]
    Ceff = _goodman_C(C, m, mean_stress, ultimate)
    # E[S^m] for the Rayleigh range S = 2A, A ~ Rayleigh(sigma): (6)
    E_Sm = (2.0 * math.sqrt(2.0) * sigma) ** m * math.gamma(1.0 + m / 2.0)
    dr = nu0 * E_Sm / Ceff
    tf, s_eq = life_and_equivalent(dr, nu0, m, Ceff)
    return {"method": "narrow_band", "damage_rate": dr, "life": tf,
            "s_eq": s_eq, "nu": nu0, "E_Sm": E_Sm, "sigma": sigma,
            "alpha2": p["alpha2"], "params": p}


def dirlik_coefficients(moments):
    """The Dirlik (1985) PDF coefficients (D1, D2, D3, Q, R, x_m, gamma) from
    the moments (theory eq. (8)). ``gamma`` = alpha_2 the irregularity factor.
    Guards the narrow-band limit (D1 -> 0) where R/Q are indeterminate."""
    p = spectral_bandwidth_params(moments)
    gamma = p["alpha2"]
    x_m = p["x_m"]
    D1 = 2.0 * (x_m - gamma ** 2) / (1.0 + gamma ** 2)
    D1 = max(D1, 0.0)                       # a density weight is non-negative
    denomR = 1.0 - gamma - D1 + D1 ** 2
    # R and Q are only used through D1 (as D1/Q, D2 Z/R^2 ...); when D1 -> 0
    # (the narrow-band limit) the exponential term vanishes and R/Q drop out.
    if abs(denomR) < 1e-14 or D1 <= 1e-14:
        R = 0.5
        D2 = 0.0
        D3 = 1.0 - D1 - D2
        # the exponential term carries the weight D1 -> 0, so its scale Q drops
        # out; keep Q finite (not 1e30) so D1*Q**m can never form 0*inf -> NaN
        # for a large slope m
        Q = 1.0
    else:
        R = (gamma - x_m - D1 ** 2) / denomR
        R = min(max(R, 1e-6), 1.0 - 1e-9)  # keep R in (0,1) (Dirlik range)
        D2 = (1.0 - gamma - D1 + D1 ** 2) / (1.0 - R)
        D2 = min(max(D2, 0.0), max(1.0 - D1, 0.0))
        D3 = 1.0 - D1 - D2
        D3 = max(D3, 0.0)
        Q = 1.25 * (gamma - D3 - D2 * R) / D1
        Q = max(Q, 1e-9)
    return {"D1": D1, "D2": D2, "D3": D3, "Q": Q, "R": R,
            "x_m": x_m, "gamma": gamma}


def dirlik_range_pdf(S, moments):
    """The Dirlik rainflow-RANGE probability density p(S) (theory eq. (8)),
    evaluated on the stress-range grid ``S`` (same units as sigma). For
    plotting / the Monte-Carlo comparison; the damage uses the closed-form
    moment ``dirlik_damage`` instead."""
    p = spectral_bandwidth_params(moments)
    sigma = p["sigma"]
    c = dirlik_coefficients(moments)
    S = np.asarray(S, dtype=float)
    if sigma <= 0.0:
        return np.zeros_like(S)
    Z = S / (2.0 * sigma)
    D1, D2, D3, Q, R = c["D1"], c["D2"], c["D3"], c["Q"], c["R"]
    pdf = (D1 / Q * np.exp(-Z / Q)
           + D2 * Z / R ** 2 * np.exp(-Z ** 2 / (2.0 * R ** 2))
           + D3 * Z * np.exp(-Z ** 2 / 2.0)) / (2.0 * sigma)
    return pdf


def dirlik_damage(moments, m, C, mean_stress=0.0, ultimate=0.0):
    """The DIRLIK (1985) wide-band damage rate (theory eqs. (2), (9)):

        E[D]/T = (nu_p / C) E[S**m] ,
        E[S**m] = (2 sigma)**m [ D1 Q**m Gamma(1+m)
                               + (sqrt2 R)**m D2 Gamma(1+m/2)
                               + sqrt2**m     D3 Gamma(1+m/2) ]

    cycles at the PEAK rate nu_p. Reduces to the narrow-band estimate as the
    irregularity factor gamma -> 1 (D1, D2 -> 0, D3 -> 1). The de-facto
    industry-standard wide-band estimator. Returns the same dict shape as
    ``narrow_band_damage`` plus the Dirlik coefficients."""
    p = spectral_bandwidth_params(moments)
    sigma, nup = p["sigma"], p["nup"]
    Ceff = _goodman_C(C, m, mean_stress, ultimate)
    c = dirlik_coefficients(moments)
    D1, D2, D3, Q, R = c["D1"], c["D2"], c["D3"], c["Q"], c["R"]
    g1 = math.gamma(1.0 + m)
    g2 = math.gamma(1.0 + m / 2.0)
    # E[S^m] closed form (9)
    E_Sm = (2.0 * sigma) ** m * (
        D1 * Q ** m * g1
        + (math.sqrt(2.0) * R) ** m * D2 * g2
        + (math.sqrt(2.0)) ** m * D3 * g2)
    dr = nup * E_Sm / Ceff
    tf, s_eq = life_and_equivalent(dr, nup, m, Ceff)
    return {"method": "dirlik", "damage_rate": dr, "life": tf, "s_eq": s_eq,
            "nu": nup, "E_Sm": E_Sm, "sigma": sigma, "alpha2": p["alpha2"],
            "coeffs": c, "params": p}


def wirsching_light_damage(moments, m, C, mean_stress=0.0, ultimate=0.0):
    """The WIRSCHING-LIGHT (1980) wide-band damage rate (theory eqs. (10),
    (11)): the narrow-band damage times the empirical rain-flow correction
    lambda_WL = a(m) + (1 - a(m))(1 - epsilon)**b(m), a = 0.926 - 0.033 m,
    b = 1.587 m - 2.323, epsilon = sqrt(1 - alpha_2^2). An independent
    cross-check of Dirlik; lambda_WL -> 1 in the narrow-band limit."""
    nb = narrow_band_damage(moments, m, C, mean_stress, ultimate)
    p = nb["params"]
    eps = p["epsilon"]
    a = 0.926 - 0.033 * m
    b = 1.587 * m - 2.323
    lam = a + (1.0 - a) * (1.0 - eps) ** b
    dr = lam * nb["damage_rate"]
    tf, s_eq = life_and_equivalent(dr, nb["nu"], m, _goodman_C(
        C, m, mean_stress, ultimate))
    return {"method": "wirsching_light", "damage_rate": dr, "life": tf,
            "s_eq": s_eq, "nu": nb["nu"], "lambda_wl": lam, "epsilon": eps,
            "sigma": nb["sigma"], "alpha2": p["alpha2"], "params": p}


def tovo_benasciutti_damage(moments, m, C, mean_stress=0.0, ultimate=0.0):
    """The TOVO-BENASCIUTTI (2005) wide-band damage rate (theory eqs. (12),
    (13)): a bandwidth-weighted combination of the narrow-band (upper-bound)
    and range-counting (lower-bound) damages, taken at the PEAK rate nu_p. A
    second independent cross-check; collapses to narrow band for alpha_1 =
    alpha_2 -> 1."""
    p = spectral_bandwidth_params(moments)
    sigma, nup = p["sigma"], p["nup"]
    a1, a2 = p["alpha1"], p["alpha2"]
    Ceff = _goodman_C(C, m, mean_stress, ultimate)
    # narrow-band damage at the PEAK rate (the Tovo-Benasciutti reference)
    E_Sm = (2.0 * math.sqrt(2.0) * sigma) ** m * math.gamma(1.0 + m / 2.0)
    D_nb_peak = nup * E_Sm / Ceff
    # bandwidth weight b in [0, 1] (eq. (13)); guard a2 -> 1 (narrow band)
    if abs(a2 - 1.0) < 1e-9:
        b = 1.0
    else:
        b = ((a1 - a2)
             * (1.112 * (1.0 + a1 * a2 - (a1 + a2)) * math.exp(2.11 * a2)
                + (a1 - a2)) / (a2 - 1.0) ** 2)
    b = min(max(b, 0.0), 1.0)
    weight = (b + (1.0 - b) * a2 ** (m - 1.0)) * a2
    dr = weight * D_nb_peak
    tf, s_eq = life_and_equivalent(dr, nup, m, Ceff)
    return {"method": "tovo_benasciutti", "damage_rate": dr, "life": tf,
            "s_eq": s_eq, "nu": nup, "b": b, "weight": weight,
            "sigma": sigma, "alpha1": a1, "alpha2": a2, "params": p}


# ============================================================================
# Monte-Carlo rainflow cross-check (build-order item 2, validation)
# ============================================================================

def synthesize_gaussian_history(freqs_hz, psd, duration, seed, fs=None):
    """Synthesise a stationary Gaussian time history whose PSD is ``psd`` (in
    the M19 stress-PSD convention, sampled on ``freqs_hz`` [Hz]), by the
    spectral-representation method (Shinozuka & Deodatis 1991) implemented
    through an inverse real FFT: each positive-frequency bin gets the
    PSD-set amplitude and a UNIFORM random phase seeded by ``seed``, then a
    single ``irfft`` sums the harmonics in O(N log N). Returns (t, x).

    Convention bridge: the M19 module carries the TWO-SIDED PSD S(Omega) with
    variance sigma^2 = (1/pi) int_0^inf S dOmega = 2 int_0^inf S(2 pi f) df, so
    the ONE-SIDED PSD in Hz is G(f) = 2 S(2 pi f). For numpy's irfft
    (x_n = (1/N) sum_k X_k e^{2 pi i k n/N}) a single positive-frequency bin
    k contributes a cosine of amplitude (2/N)|X_k|, whose variance
    (2/N^2)|X_k|^2 must equal G(f_k) df; hence |X_k| = N sqrt(G(f_k) df / 2)
    with a random phase. Summed over the bins this reproduces the variance m_0
    (and thus the same spectral moments the closed forms use)."""
    rng = np.random.default_rng(int(seed))
    f = np.asarray(freqs_hz, dtype=float)
    S = np.clip(np.asarray(psd, dtype=float), 0.0, None)
    fmax = float(f.max()) if len(f) > 0 else 0.0
    if fs is None:
        fs = max(8.0 * fmax, 1.0)                    # comfortably above Nyquist (2 fmax)
    nt = int(max(round(duration * fs), 4))
    if nt % 2:                             # even length keeps rfft bookkeeping simple
        nt += 1
    df = fs / nt
    fft_f = np.fft.rfftfreq(nt, d=1.0 / fs)          # 0 .. fs/2
    # one-sided PSD G(f) = 2 S(2 pi f) interpolated onto the FFT grid (zero
    # outside the supplied band — the response carries no energy there)
    G = 2.0 * np.interp(fft_f, f, S, left=0.0, right=0.0)
    mag = nt * np.sqrt(np.clip(G, 0.0, None) * df / 2.0)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=fft_f.size)
    X = mag * np.exp(1j * phase)
    X[0] = 0.0                             # no DC (a PSD carries no mean)
    if nt % 2 == 0:
        X[-1] = np.abs(X[-1])              # Nyquist bin must be real
    x = np.fft.irfft(X, n=nt)
    t = np.arange(nt) / fs
    return t, x


def rainflow_count(series):
    """ASTM E1049-85 rainflow cycle counting (the three-point / four-point
    algorithm) of a stress time series. Returns an array of stress RANGES
    (peak-to-peak) of the counted cycles (half-cycles counted as 0.5 via
    repetition — here each full cycle contributes one range, each residual
    half-cycle one range with a 0.5 weight folded into ``counts``). Returns
    (ranges, counts) with counts in {0.5, 1.0}.

    Implementation: reduce the signal to its turning points, then apply the
    standard four-point rainflow rule on a running stack (Downing & Socie
    1982) — the reference time-domain counter the spectral PDFs approximate."""
    x = np.asarray(series, dtype=float)
    # 1) reduce to turning points (local extrema); keep the first and last
    if x.size < 3:
        return np.zeros(0), np.zeros(0)
    dx = np.diff(x)
    # indices where the slope changes sign are turning points
    slope = np.sign(dx)
    # drop zero-slope segments by carrying the previous sign forward
    for i in range(1, slope.size):
        if slope[i] == 0:
            slope[i] = slope[i - 1]
    turn = np.ones(x.size, dtype=bool)
    turn[1:-1] = slope[1:] != slope[:-1]
    tp = x[turn]

    ranges = []
    counts = []
    stack = []
    for p in tp:
        stack.append(p)
        # apply the four-point rule while the interior pair is an enclosed cycle
        while len(stack) >= 4:
            s = stack
            r1 = abs(s[-3] - s[-4])
            r2 = abs(s[-2] - s[-3])
            r3 = abs(s[-1] - s[-2])
            if r2 <= r1 and r2 <= r3:
                # the middle range s[-3]..s[-2] is a closed full cycle
                ranges.append(r2)
                counts.append(1.0)
                # remove the two interior points and continue
                del stack[-3:-1]
            else:
                break
    # 2) count the residual stack as half cycles (the ASTM residue rule)
    for i in range(len(stack) - 1):
        ranges.append(abs(stack[i + 1] - stack[i]))
        counts.append(0.5)
    return np.asarray(ranges), np.asarray(counts)


def monte_carlo_damage(freqs_hz, psd, m, C, duration, seed, fs=None,
                       mean_stress=0.0, ultimate=0.0):
    """The TIME-DOMAIN damage rate by Monte-Carlo: synthesise a Gaussian
    history from the stress PSD, rainflow-count it (ASTM E1049), and Miner-sum
    the counted ranges S with the S-N law N = C S**-m:

        D = sum_i count_i * S_i**m / C ,   E[D]/T ~ D / duration

    The independent time-domain answer the spectral estimators approximate;
    seeded for reproducibility. Returns {damage_rate, ncycles, ranges, ...}."""
    Ceff = _goodman_C(C, m, mean_stress, ultimate)
    t, x = synthesize_gaussian_history(freqs_hz, psd, duration, seed, fs=fs)
    ranges, counts = rainflow_count(x)
    D = float(np.sum(counts * ranges ** m) / Ceff) if ranges.size else 0.0
    T = t[-1] - t[0] if t.size > 1 else duration
    dr = D / T if T > 0 else 0.0
    tf, s_eq = life_and_equivalent(dr, float(counts.sum()) / T if T > 0 else 0.0,
                                   m, Ceff)
    return {"method": "monte_carlo", "damage_rate": dr, "life": tf,
            "s_eq": s_eq, "ncycles": float(counts.sum()),
            "ranges": ranges, "counts": counts, "duration": T,
            "rms": float(np.std(x))}


# ============================================================================
# The full fatigue summary (all methods on one channel)
# ============================================================================

def fatigue_summary(moments, m, C, mean_stress=0.0, ultimate=0.0):
    """Evaluate ALL closed-form spectral estimators on one channel's moment
    array and return a dict of per-method results plus the shared spectral
    descriptors (RMS, rates, width factors). The Monte-Carlo cross-check is a
    separate call (it needs the PSD, a duration and a seed)."""
    p = spectral_bandwidth_params(moments)
    return {
        "params": p,
        "narrow_band": narrow_band_damage(moments, m, C, mean_stress,
                                          ultimate),
        "dirlik": dirlik_damage(moments, m, C, mean_stress, ultimate),
        "wirsching_light": wirsching_light_damage(moments, m, C, mean_stress,
                                                  ultimate),
        "tovo_benasciutti": tovo_benasciutti_damage(moments, m, C,
                                                    mean_stress, ultimate),
    }

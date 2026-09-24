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

# ============================================================================
# Mean stress corrections (Goodman, Gerber, Soderberg, Morrow, SWT, Walker)
# ============================================================================

def goodman_correction(s_a, sigma_m, ultimate, ignore_compressive=True):
    """Goodman mean-stress correction of stress amplitude:
        S_eq = S_a / (1 - sigma_m / S_u)
    """
    if not ultimate or ultimate <= 0.0 or not sigma_m:
        return float(s_a)
    if ignore_compressive and float(sigma_m) <= 0.0:
        return float(s_a)
    denom = 1.0 - float(sigma_m) / float(ultimate)
    if denom <= 0.0:
        return math.inf
    return float(s_a) / denom


def gerber_correction(s_a, sigma_m, ultimate, ignore_compressive=True):
    """Gerber mean-stress correction of stress amplitude:
        S_eq = S_a / (1 - (sigma_m / S_u)^2)
    """
    if not ultimate or ultimate <= 0.0 or not sigma_m:
        return float(s_a)
    if ignore_compressive and float(sigma_m) <= 0.0:
        return float(s_a)
    ratio = float(sigma_m) / float(ultimate)
    denom = 1.0 - ratio ** 2
    if denom <= 0.0:
        return math.inf
    return float(s_a) / denom


def soderberg_correction(s_a, sigma_m, yield_strength, ignore_compressive=True):
    """Soderberg mean-stress correction of stress amplitude:
        S_eq = S_a / (1 - sigma_m / S_y)
    """
    if not yield_strength or yield_strength <= 0.0 or not sigma_m:
        return float(s_a)
    if ignore_compressive and float(sigma_m) <= 0.0:
        return float(s_a)
    denom = 1.0 - float(sigma_m) / float(yield_strength)
    if denom <= 0.0:
        return math.inf
    return float(s_a) / denom


def morrow_correction(s_a, sigma_m, sigma_f_prime, ignore_compressive=True):
    """Morrow mean-stress correction of stress amplitude:
        S_eq = S_a / (1 - sigma_m / sigma_f')
    """
    if not sigma_f_prime or sigma_f_prime <= 0.0 or not sigma_m:
        return float(s_a)
    if ignore_compressive and float(sigma_m) <= 0.0:
        return float(s_a)
    denom = 1.0 - float(sigma_m) / float(sigma_f_prime)
    if denom <= 0.0:
        return math.inf
    return float(s_a) / denom


def swt_correction(s_a, sigma_m):
    """Smith-Watson-Topper (SWT) equivalent stress amplitude:
        S_eq = sqrt(sigma_max * S_a) = sqrt((sigma_m + S_a) * S_a)
    """
    sigma_max = float(sigma_m) + float(s_a)
    if sigma_max <= 0.0 or s_a <= 0.0:
        return 0.0
    return math.sqrt(sigma_max * float(s_a))


def walker_correction(s_a, sigma_m, gamma=0.5):
    """Walker equivalent stress amplitude:
        S_eq = sigma_max^(1 - gamma) * S_a^gamma = (sigma_m + S_a)^(1 - gamma) * S_a^gamma
    When gamma = 0.5, Walker collapses to Smith-Watson-Topper (SWT).
    """
    sigma_max = float(sigma_m) + float(s_a)
    if sigma_max <= 0.0 or s_a <= 0.0:
        return 0.0
    return (sigma_max ** (1.0 - float(gamma))) * (float(s_a) ** float(gamma))


def mean_stress_correction(s_a, sigma_m, method="goodman", ultimate=0.0,
                           su=None, yield_strength=0.0, sigma_f_prime=0.0,
                           walker_gamma=0.5, ignore_compressive=True):
    """Evaluate mean stress correction on stress amplitude `s_a` using `method`:
    'goodman', 'gerber', 'soderberg', 'morrow', 'swt', 'walker'.
    """
    ult = su if su is not None else ultimate
    m_lower = str(method).lower().strip()
    if m_lower in ("goodman", "good"):
        return goodman_correction(s_a, sigma_m, ult, ignore_compressive=ignore_compressive)
    elif m_lower in ("gerber", "gerb"):
        return gerber_correction(s_a, sigma_m, ult, ignore_compressive=ignore_compressive)
    elif m_lower in ("soderberg", "soder"):
        return soderberg_correction(s_a, sigma_m, yield_strength, ignore_compressive=ignore_compressive)
    elif m_lower in ("morrow", "morr"):
        return morrow_correction(s_a, sigma_m, sigma_f_prime, ignore_compressive=ignore_compressive)
    elif m_lower in ("swt", "smith_watson_topper"):
        return swt_correction(s_a, sigma_m)
    elif m_lower in ("walker", "walk"):
        return walker_correction(s_a, sigma_m, walker_gamma)
    else:
        return goodman_correction(s_a, sigma_m, ult, ignore_compressive=ignore_compressive)


def effective_sn_coefficient(C, m, mean_stress=0.0, method="goodman",
                             ultimate=0.0, su=None, yield_strength=0.0,
                             sigma_f_prime=0.0, walker_gamma=0.5,
                             ignore_compressive=True):
    """Effective S-N intercept coefficient C_eff = C * fac^m for mean stress
    corrections (scaling the allowable stress range by 1/fac).
    """
    if not mean_stress:
        return float(C)
    sm = float(mean_stress)
    if ignore_compressive and sm <= 0.0:
        return float(C)
    ult = su if su is not None else ultimate
    meth = str(method).lower().strip()
    if meth in ("goodman", "good"):
        if not ult or ult <= 0.0:
            return float(C)
        fac = 1.0 - sm / float(ult)
    elif meth in ("gerber", "gerb"):
        if not ult or ult <= 0.0:
            return float(C)
        fac = 1.0 - (sm / float(ult)) ** 2
    elif meth in ("soderberg", "soder"):
        if not yield_strength or yield_strength <= 0.0:
            return float(C)
        fac = 1.0 - sm / float(yield_strength)
    elif meth in ("morrow", "morr"):
        if not sigma_f_prime or sigma_f_prime <= 0.0:
            return float(C)
        fac = 1.0 - sm / float(sigma_f_prime)
    else:
        # Fall back to Goodman if ultimate is provided
        if ult and ult > 0.0:
            fac = 1.0 - sm / float(ult)
        else:
            return float(C)
    if fac <= 0.0:
        return 1e-300
    return float(C) * (fac ** float(m))


def _goodman_C(C, m, mean_stress, ultimate):
    """Backward compatibility helper for Goodman mean-stress correction."""
    return effective_sn_coefficient(C, m, mean_stress=mean_stress,
                                    method="goodman", ultimate=ultimate)


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


class SteinbergResult(dict):
    """Result dictionary for Steinberg 3-band fatigue damage."""
    def __iter__(self):
        d_tot = self.get("damage")
        if d_tot is None:
            dur = self.get("duration") or 1.0
            d_tot = self["damage_rate"] * dur
        bands_val = self.get("bands_list")
        if bands_val is None:
            bands_val = list(self.get("bands", {}).values())
        return iter((d_tot, self["damage_rate"], bands_val))


class ZhaoBakerCoeffs(dict):
    """Result dictionary for Zhao-Baker PDF parameters."""
    def __iter__(self):
        return iter((self["a"], self["beta"], self["w"]))


class ZhaoBakerResult(dict):
    """Result dictionary for Zhao-Baker spectral fatigue damage."""
    def __iter__(self):
        d_tot = self.get("damage")
        if d_tot is None:
            dur = self.get("duration") or 1.0
            d_tot = self["damage_rate"] * dur
        e_sm = self.get("e_sm", self.get("E_Sm", 0.0))
        return iter((d_tot, self["damage_rate"], e_sm))


def steinberg_damage(*args, **kwargs):
    """The STEINBERG (1988) 3-band random vibration fatigue damage:
    Gaussian cycle partitioning across 3 stress bands:
        - 1*sigma: 68.3% of time, S = 1*sigma
        - 2*sigma: 27.1% of time, S = 2*sigma
        - 3*sigma: 4.33% of time, S = 3*sigma
    Total damage rate:
        E[D]/T = (nu_p / C_eff) * [0.683 (1*sigma)^m + 0.271 (2*sigma)^m + 0.0433 (3*sigma)^m]
    Total damage for duration T:
        D = sum_{i=1}^3 (n_i / N_i) = (nu_p * T / C_eff) * [...]
    Can be called as:
        steinberg_damage(sigma, nu_p, duration, C, m, ...) -> SteinbergResult (iter yields (D, dr, bands))
        steinberg_damage(moments, m, C, ...) -> SteinbergResult
    """
    if len(args) >= 5 and isinstance(args[0], (int, float)):
        sigma = float(args[0])
        nup = float(args[1])
        duration = float(args[2])
        C = float(args[3])
        m = float(args[4])
        mean_stress = kwargs.get("mean_stress", 0.0)
        ultimate = kwargs.get("ultimate", kwargs.get("su", 0.0))
        mean_correction = kwargs.get("mean_correction", "goodman")
        p = {"sigma": sigma, "nup": nup}
    else:
        moments = args[0]
        m = float(args[1])
        C = float(args[2])
        mean_stress = kwargs.get("mean_stress", 0.0)
        ultimate = kwargs.get("ultimate", kwargs.get("su", 0.0))
        duration = kwargs.get("duration", None)
        mean_correction = kwargs.get("mean_correction", "goodman")
        p = spectral_bandwidth_params(moments)
        sigma, nup = p["sigma"], p["nup"]

    yield_strength = kwargs.get("yield_strength", 0.0)
    sigma_f_prime = kwargs.get("sigma_f_prime", 0.0)
    walker_gamma = kwargs.get("walker_gamma", 0.5)
    Ceff = effective_sn_coefficient(C, m, mean_stress=mean_stress,
                                    method=mean_correction, ultimate=ultimate,
                                    yield_strength=yield_strength,
                                    sigma_f_prime=sigma_f_prime,
                                    walker_gamma=walker_gamma)
    term = (0.683 * (1.0 * sigma) ** m
            + 0.271 * (2.0 * sigma) ** m
            + 0.0433 * (3.0 * sigma) ** m)
    dr = nup * term / Ceff if Ceff > 0.0 else 0.0
    tf, s_eq = life_and_equivalent(dr, nup, m, Ceff)
    dur = float(duration) if duration is not None else 1.0
    dmg = dr * dur if duration is not None else None
    bands_dict = {
        "1sigma": {"fraction": 0.683, "stress": 1.0 * sigma, "rate": 0.683 * nup, "damage": (0.683 * nup * dur * ((1.0 * sigma) ** m) / Ceff)},
        "2sigma": {"fraction": 0.271, "stress": 2.0 * sigma, "rate": 0.271 * nup, "damage": (0.271 * nup * dur * ((2.0 * sigma) ** m) / Ceff)},
        "3sigma": {"fraction": 0.0433, "stress": 3.0 * sigma, "rate": 0.0433 * nup, "damage": (0.0433 * nup * dur * ((3.0 * sigma) ** m) / Ceff)},
    }
    bands_list = list(bands_dict.values())
    return SteinbergResult({
        "method": "steinberg",
        "damage_rate": dr,
        "life": tf,
        "s_eq": s_eq,
        "nu": nup,
        "sigma": sigma,
        "damage": dmg if dmg is not None else dr,
        "duration": duration,
        "term": term,
        "bands": bands_dict,
        "bands_list": bands_list,
        "params": p,
    })


def zhao_baker_coefficients(moments_or_alpha2):
    """The Zhao & Baker (1992) Weibull-Rayleigh mixture PDF parameters:
        alpha_2 = m_2 / sqrt(m_0 * m_4)
        a = 8 - 7 * alpha_2
        beta = 1.1 if alpha_2 < 0.9 else 1.1 + 9 * (alpha_2 - 0.9)
        w = (1 - alpha_2) / (1 - sqrt(2/pi) * Gamma(1 + 1/beta) * a^(-1/beta))
    Guards numerical limits and clamps w to [0, 1].
    Accepts either moments array or scalar alpha_2.
    """
    if isinstance(moments_or_alpha2, (int, float)):
        a2 = float(moments_or_alpha2)
    else:
        p = spectral_bandwidth_params(moments_or_alpha2)
        a2 = p["alpha2"]
    a = 8.0 - 7.0 * a2
    a = max(a, 1e-6)
    if a2 < 0.9:
        beta = 1.1
    else:
        beta = 1.1 + 9.0 * (a2 - 0.9)
    gamma_term = math.gamma(1.0 + 1.0 / beta) * (a ** (-1.0 / beta))
    denom = 1.0 - math.sqrt(2.0 / math.pi) * gamma_term
    if abs(denom) < 1e-12:
        w = 0.0
    else:
        w = (1.0 - a2) / denom
    w = min(max(w, 0.0), 1.0)
    return ZhaoBakerCoeffs({"a": a, "beta": beta, "w": w, "alpha2": a2})


def zhao_baker_range_pdf(S, *args):
    """The Zhao-Baker rainflow stress range PDF p(S) on the range grid S:
        p(Z) = w * a * beta * Z^(beta - 1) * exp(-a * Z^beta)
             + (1 - w) * Z * exp(-Z^2 / 2)
    with Z = S / (2 * sigma) and p(S) = p(Z) / (2 * sigma).
    Can be called as:
        zhao_baker_range_pdf(S, moments)
        zhao_baker_range_pdf(S, sigma, alpha2)
    """
    if len(args) == 1:
        moments = args[0]
        p = spectral_bandwidth_params(moments)
        sigma = p["sigma"]
        c = zhao_baker_coefficients(moments)
    elif len(args) >= 2:
        sigma = float(args[0])
        alpha2 = float(args[1])
        c = zhao_baker_coefficients(alpha2)
    else:
        raise ValueError("zhao_baker_range_pdf requires moments or (sigma, alpha2)")
    a, beta, w = c["a"], c["beta"], c["w"]
    S = np.asarray(S, dtype=float)
    if sigma <= 0.0:
        return np.zeros_like(S)
    Z = S / (2.0 * sigma)
    Z_safe = np.maximum(Z, 1e-15)
    weibull = a * beta * (Z_safe ** (beta - 1.0)) * np.exp(-a * (Z_safe ** beta))
    rayleigh = Z * np.exp(-0.5 * (Z ** 2))
    pdf_Z = w * weibull + (1.0 - w) * rayleigh
    return pdf_Z / (2.0 * sigma)


def zhao_baker_damage(*args, **kwargs):
    """The ZHAO-BAKER (1992) Weibull-Rayleigh spectral damage model:
    Closed-form m-th moment of rainflow stress range:
        E[S^m] = (2 sigma)^m [ w a^(-m/beta) Gamma(1 + m/beta)
                             + (1 - w) (sqrt(2))^m Gamma(1 + m/2) ]
        E[D]/T = (nu_p / C_eff) * E[S^m]
    Collapses to narrow-band damage when alpha_2 -> 1 (w -> 0).
    Can be called as:
        zhao_baker_damage(sigma, alpha2, nu_p, duration, C, m, ...)
        zhao_baker_damage(moments, m, C, ...)
    """
    if len(args) >= 6 and isinstance(args[0], (int, float)):
        sigma = float(args[0])
        a2 = float(args[1])
        nup = float(args[2])
        duration = float(args[3])
        C = float(args[4])
        m = float(args[5])
        mean_stress = kwargs.get("mean_stress", 0.0)
        ultimate = kwargs.get("ultimate", kwargs.get("su", 0.0))
        mean_correction = kwargs.get("mean_correction", "goodman")
        p = {"sigma": sigma, "nup": nup, "alpha2": a2}
        c = zhao_baker_coefficients(a2)
    else:
        moments = args[0]
        m = float(args[1])
        C = float(args[2])
        mean_stress = kwargs.get("mean_stress", 0.0)
        ultimate = kwargs.get("ultimate", kwargs.get("su", 0.0))
        duration = kwargs.get("duration", None)
        mean_correction = kwargs.get("mean_correction", "goodman")
        p = spectral_bandwidth_params(moments)
        sigma, nup = p["sigma"], p["nup"]
        c = zhao_baker_coefficients(moments)

    yield_strength = kwargs.get("yield_strength", 0.0)
    sigma_f_prime = kwargs.get("sigma_f_prime", 0.0)
    walker_gamma = kwargs.get("walker_gamma", 0.5)
    Ceff = effective_sn_coefficient(C, m, mean_stress=mean_stress,
                                    method=mean_correction, ultimate=ultimate,
                                    yield_strength=yield_strength,
                                    sigma_f_prime=sigma_f_prime,
                                    walker_gamma=walker_gamma)
    a, beta, w = c["a"], c["beta"], c["w"]
    term_weibull = (a ** (-float(m) / beta)) * math.gamma(1.0 + float(m) / beta)
    term_rayleigh = (math.sqrt(2.0) ** float(m)) * math.gamma(1.0 + float(m) / 2.0)
    E_Sm = ((2.0 * sigma) ** float(m)) * (w * term_weibull + (1.0 - w) * term_rayleigh)
    dr = nup * E_Sm / Ceff if Ceff > 0.0 else 0.0
    tf, s_eq = life_and_equivalent(dr, nup, m, Ceff)
    dur = float(duration) if duration is not None else 1.0
    dmg = dr * dur if duration is not None else None
    return ZhaoBakerResult({
        "method": "zhao_baker",
        "damage_rate": dr,
        "life": tf,
        "s_eq": s_eq,
        "nu": nup,
        "E_Sm": E_Sm,
        "e_sm": E_Sm,
        "sigma": sigma,
        "alpha2": p.get("alpha2", 0.0),
        "damage": dmg if dmg is not None else dr,
        "duration": duration,
        "coeffs": c,
        "params": p,
    })


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
# Strain-Life (epsilon-N) & Notch Plasticity Models (M614)
# ============================================================================

def ramberg_osgood_strain(sigma, E, K_prime, n_prime):
    """Monotonic Ramberg-Osgood stress-strain relation:
        epsilon = sigma / E + (abs(sigma) / K')^(1 / n') * sign(sigma)
    """
    s = float(sigma)
    if s == 0.0:
        return 0.0
    sgn = 1.0 if s > 0.0 else -1.0
    return s / float(E) + sgn * ((abs(s) / float(K_prime)) ** (1.0 / float(n_prime)))


def ramberg_osgood_stress(epsilon, E, K_prime, n_prime, tol=1e-9, max_iter=100):
    """Invert monotonic Ramberg-Osgood curve to find stress sigma for a given strain epsilon."""
    eps = float(epsilon)
    if abs(eps) < 1e-15:
        return 0.0
    # Initial linear elastic guess
    sigma = float(E) * eps
    for _ in range(max_iter):
        e_cur = ramberg_osgood_strain(sigma, E, K_prime, n_prime)
        res = e_cur - eps
        if abs(res) < tol:
            break
        s_abs = max(abs(sigma), 1e-15)
        de_ds = 1.0 / float(E) + (1.0 / (float(n_prime) * float(K_prime))) * ((s_abs / float(K_prime)) ** (1.0 / float(n_prime) - 1.0))
        d_sigma = res / de_ds
        sigma -= d_sigma
        if abs(d_sigma) < tol:
            break
    return sigma


def ramberg_osgood_cyclic_strain(delta_sigma, E, K_prime, n_prime):
    """Cyclic Ramberg-Osgood hysteresis curve (Masing hypothesis):
        Delta epsilon = Delta sigma / E + 2 * (Delta sigma / (2 * K'))^(1 / n')
    """
    ds = float(delta_sigma)
    if ds <= 0.0:
        return 0.0
    return ds / float(E) + 2.0 * ((ds / (2.0 * float(K_prime))) ** (1.0 / float(n_prime)))


def ramberg_osgood_cyclic_stress(delta_epsilon, E, K_prime, n_prime, tol=1e-9, max_iter=100):
    """Invert cyclic Ramberg-Osgood curve to find stress range Delta sigma for a given strain range Delta epsilon."""
    de = float(delta_epsilon)
    if de <= 0.0:
        return 0.0
    ds = float(E) * de
    for _ in range(max_iter):
        de_cur = ramberg_osgood_cyclic_strain(ds, E, K_prime, n_prime)
        res = de_cur - de
        if abs(res) < tol:
            break
        ds_abs = max(ds, 1e-15)
        dde_dds = 1.0 / float(E) + (1.0 / (float(n_prime) * float(K_prime))) * ((ds_abs / (2.0 * float(K_prime))) ** (1.0 / float(n_prime) - 1.0))
        delta = res / dde_dds
        ds = max(ds - delta, 1e-12)
        if abs(delta) < tol:
            break
    return ds


class CoffinMansonResult(float):
    """Result object that acts as a float (cycles Nf) and a dict with details."""
    def __new__(cls, val, data):
        obj = super().__new__(cls, float(val))
        obj._data = data
        return obj

    def __getitem__(self, item):
        return self._data[item]

    def __contains__(self, item):
        return item in self._data

    def get(self, item, default=None):
        return self._data.get(item, default)

    def update(self, d):
        self._data.update(d)

    def __repr__(self):
        return repr(self._data)


def coffin_manson_strain(reversals_2Nf, *args, **kwargs):
    """Coffin-Manson relation for total strain amplitude Delta epsilon / 2:
        Delta epsilon / 2 = ((sigma_f' - sigma_m) / E) * (2N_f)^b + eps_f' * (2N_f)^c
    """
    rev = float(reversals_2Nf)
    if rev <= 0.0:
        return math.inf
    if len(args) >= 5:
        a0, a1 = float(args[0]), float(args[1])
        if a0 > a1:  # a0 is E, a1 is sigf
            E, sigf = a0, a1
        else:
            sigf, E = a0, a1
        b, epsf, c = float(args[2]), float(args[3]), float(args[4])
    else:
        sigf = float(kwargs.get("sigma_f_prime", args[0] if len(args) > 0 else 0.0))
        E = float(kwargs.get("E", args[1] if len(args) > 1 else 0.0))
        b = float(kwargs.get("b", args[2] if len(args) > 2 else 0.0))
        epsf = float(kwargs.get("eps_f_prime", args[3] if len(args) > 3 else 0.0))
        c = float(kwargs.get("c", args[4] if len(args) > 4 else 0.0))
    sigma_m = float(kwargs.get("sigma_m", (args[5] if len(args) > 5 else 0.0)))
    sig_term = (sigf - sigma_m) / E
    return sig_term * (rev ** b) + epsf * (rev ** c)


def coffin_manson_life(strain_input, *args, **kwargs):
    """Solve Coffin-Manson relation for fatigue life (N_f cycles, 2N_f reversals)
    given strain amplitude (eps_a) or range (Delta epsilon).
    Supports Morrow mean stress (via sigma_m/mean_stress) or Smith-Watson-Topper (SWT, via method='swt').
    """
    s_in = float(strain_input)
    if s_in <= 0.0:
        res = {"Nf": math.inf, "reversals_2Nf": math.inf, "delta_eps": s_in, "eps_a": s_in}
        return CoffinMansonResult(math.inf, res)

    if len(args) >= 5:
        a0, a1 = float(args[0]), float(args[1])
        if a0 > a1:  # a0 is E, a1 is sigf
            E, sigf = a0, a1
        else:
            sigf, E = a0, a1
        b, epsf, c = float(args[2]), float(args[3]), float(args[4])
    else:
        sigf = float(kwargs.get("sigma_f_prime", args[0] if len(args) > 0 else 0.0))
        E = float(kwargs.get("E", args[1] if len(args) > 1 else 0.0))
        b = float(kwargs.get("b", args[2] if len(args) > 2 else 0.0))
        epsf = float(kwargs.get("eps_f_prime", args[3] if len(args) > 3 else 0.0))
        c = float(kwargs.get("c", args[4] if len(args) > 4 else 0.0))

    sigma_m = float(kwargs.get("sigma_m", kwargs.get("mean_stress", (args[5] if len(args) > 5 else 0.0))))
    method = kwargs.get("method", (args[6] if len(args) > 6 else None))
    sigma_max = kwargs.get("sigma_max", None)
    tol = kwargs.get("tol", 1e-8)
    max_iter = kwargs.get("max_iter", 100)

    eps_a = s_in
    use_swt = (str(method).lower().strip() == "swt")
    if use_swt and sigma_max is None:
        sigma_max = sigf if sigma_m == 0.0 else (sigf * (1.0 - sigma_m / sigf) if sigf > 0 else 1000.0)

    # Solve in log-space: x = ln(2N_f)
    x = math.log(1e5)  # initial guess 10^5 reversals
    for _ in range(max_iter):
        rev = math.exp(x)
        if use_swt and (sigma_max is not None):
            s_max = float(sigma_max)
            lhs = s_max * eps_a
            t1 = ((sigf ** 2) / E) * (rev ** (2.0 * b))
            t2 = sigf * epsf * (rev ** (b + c))
            f_val = t1 + t2 - lhs
            df_dx = (2.0 * b * t1) + ((b + c) * t2)
        else:
            sig_eff = sigf - sigma_m
            t1 = (sig_eff / E) * (rev ** b)
            t2 = epsf * (rev ** c)
            f_val = t1 + t2 - eps_a
            df_dx = b * t1 + c * t2

        if abs(df_dx) < 1e-20:
            break
        dx = f_val / df_dx
        x = max(min(x - dx, 45.0), 0.0)  # clamp between 1 reversal and ~10^19
        if abs(dx) < tol:
            break

    rev_final = math.exp(x)
    nf_final = rev_final / 2.0
    res = {
        "Nf": nf_final,
        "reversals_2Nf": rev_final,
        "delta_eps": 2.0 * eps_a,
        "eps_a": eps_a,
        "sigma_m": sigma_m,
    }
    return CoffinMansonResult(nf_final, res)


def neuber_notch_analysis(delta_sigma_e, Kt, E, K_prime, n_prime, tol=1e-8, max_iter=100):
    """Neuber's rule for notch plasticity:
        K_t^2 * Delta sigma_e * Delta epsilon_e = Delta sigma * Delta epsilon
    where Delta epsilon_e = Delta sigma_e / E.
    Equivalently:
        Delta sigma * [Delta sigma / E + 2 * (Delta sigma / (2*K'))^(1/n')] = (K_t * Delta sigma_e)^2 / E
    Returns:
        (delta_sigma, delta_epsilon) local elasto-plastic stress and strain ranges.
    """
    dse = float(delta_sigma_e)
    kt = float(Kt)
    if dse <= 0.0 or kt <= 0.0:
        return 0.0, 0.0
    target_energy = ((kt * dse) ** 2) / float(E)
    # Solve for local stress range ds
    ds = kt * dse
    for _ in range(max_iter):
        eps = ramberg_osgood_cyclic_strain(ds, E, K_prime, n_prime)
        f_val = ds * eps - target_energy
        if abs(f_val) < tol * target_energy:
            break
        # df/dds = eps + ds * d(eps)/dds
        ds_abs = max(ds, 1e-15)
        deps_dds = 1.0 / float(E) + (1.0 / (float(n_prime) * float(K_prime))) * ((ds_abs / (2.0 * float(K_prime))) ** (1.0 / float(n_prime) - 1.0))
        df_dds = eps + ds * deps_dds
        if df_dds <= 0.0:
            break
        delta = f_val / df_dds
        ds = max(ds - delta, 1e-12)
        if abs(delta) < tol:
            break
    delta_eps = ramberg_osgood_cyclic_strain(ds, E, K_prime, n_prime)
    return ds, delta_eps


def neuber_strain_life(delta_sigma_e, Kt, E, K_prime, n_prime, sigma_f_prime, b,
                       eps_f_prime, c, sigma_m=0.0, method=None):
    """Notch strain-life analysis combining Neuber's rule with Coffin-Manson equation."""
    ds, de = neuber_notch_analysis(delta_sigma_e, Kt, E, K_prime, n_prime)
    res = coffin_manson_life(de, sigma_f_prime, E, b, eps_f_prime, c,
                             sigma_m=sigma_m, method=method)
    res.update({
        "delta_sigma_local": ds,
        "delta_epsilon_local": de,
        "delta_sigma_elastic": delta_sigma_e,
        "Kt": Kt,
        "rule": "neuber",
    })
    return res


def glinka_notch_analysis(delta_sigma_e, Kt, E, K_prime, n_prime, tol=1e-8, max_iter=100):
    """Glinka's Equivalent Strain Energy Density (ESED) notch plasticity rule:
        W_elastic = W_el-pl
        (K_t * Delta sigma_e)^2 / (4 * E) = Delta sigma^2 / (4 * E) + (Delta sigma / (1 + n')) * (Delta sigma / (2*K'))^(1/n')
    Returns:
        (delta_sigma, delta_epsilon) local elasto-plastic stress and strain ranges.
    """
    dse = float(delta_sigma_e)
    kt = float(Kt)
    if dse <= 0.0 or kt <= 0.0:
        return 0.0, 0.0
    w_elastic = ((kt * dse) ** 2) / (4.0 * float(E))
    ds = kt * dse
    np_val = float(n_prime)
    for _ in range(max_iter):
        ds_abs = max(ds, 1e-15)
        w_plastic = (ds / (1.0 + np_val)) * ((ds_abs / (2.0 * float(K_prime))) ** (1.0 / np_val))
        w_elpl = (ds ** 2) / (4.0 * float(E)) + w_plastic
        f_val = w_elpl - w_elastic
        if abs(f_val) < tol * w_elastic:
            break
        # Derivative dw/dds = ds / (2*E) + (ds/(2K'))^(1/np)
        df_dds = ds / (2.0 * float(E)) + ((ds_abs / (2.0 * float(K_prime))) ** (1.0 / np_val))
        if df_dds <= 0.0:
            break
        delta = f_val / df_dds
        ds = max(ds - delta, 1e-12)
        if abs(delta) < tol:
            break
    delta_eps = ramberg_osgood_cyclic_strain(ds, E, K_prime, n_prime)
    return ds, delta_eps


def glinka_strain_life(delta_sigma_e, Kt, E, K_prime, n_prime, sigma_f_prime, b,
                       eps_f_prime, c, sigma_m=0.0, method=None):
    """Notch strain-life analysis combining Glinka's ESED rule with Coffin-Manson equation."""
    ds, de = glinka_notch_analysis(delta_sigma_e, Kt, E, K_prime, n_prime)
    res = coffin_manson_life(de, sigma_f_prime, E, b, eps_f_prime, c,
                             sigma_m=sigma_m, method=method)
    res.update({
        "delta_sigma_local": ds,
        "delta_epsilon_local": de,
        "delta_sigma_elastic": delta_sigma_e,
        "Kt": Kt,
        "rule": "glinka",
    })
    return res


# ============================================================================
# Multi-slope S-N curve (bi-linear Wöhler with knee point N_k, endurance limit S_e)
# ============================================================================

class SpectralDamageResult(float):
    """Result object that acts as a float (damage or damage rate) and a dict."""
    def __new__(cls, val, data):
        obj = super().__new__(cls, float(val))
        obj._data = data
        return obj

    def __getitem__(self, item):
        return self._data[item]

    def __contains__(self, item):
        return item in self._data

    def get(self, item, default=None):
        return self._data.get(item, default)

    def update(self, d):
        self._data.update(d)

    def __repr__(self):
        return repr(self._data)


class MultiSlopeSN:
    """Bi-linear / multi-slope Wöhler S-N curve with knee point (S_k, N_k),
    slopes m1 (high-stress / low-cycle regime S >= S_k) and m2 (low-stress / high-cycle regime
    S_e <= S < S_k), and endurance limit S_e below which N -> inf.
    """

    def __init__(self, s_knee: float, n_knee: float, m1: float, m2: float,
                 s_endurance: float = 0.0):
        self.s_knee = float(s_knee)
        self.n_knee = float(n_knee)
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.s_endurance = float(s_endurance)
        self.c1 = self.n_knee * (self.s_knee ** self.m1)
        self.c2 = self.n_knee * (self.s_knee ** self.m2)

    def life(self, S: float) -> float:
        """Cycles to failure N at constant stress range S."""
        s = float(S)
        if s <= 0.0:
            return math.inf
        if self.s_endurance > 0.0 and s < self.s_endurance:
            return math.inf
        if s >= self.s_knee:
            return self.c1 * (s ** (-self.m1))
        else:
            return self.c2 * (s ** (-self.m2))

    def damage_per_cycle(self, S: float) -> float:
        """Damage 1/N caused by one cycle of stress range S."""
        n = self.life(S)
        return 0.0 if math.isinf(n) or n <= 0.0 else 1.0 / n

    def damage_spectrum(self, ranges, counts) -> float:
        """Total Miner damage from cycle-counted stress ranges and counts."""
        r = np.asarray(ranges, dtype=float)
        c = np.asarray(counts, dtype=float)
        d = 0.0
        for s_val, count in zip(r, c):
            d += count * self.damage_per_cycle(s_val)
        return d

    def spectral_damage(self, *args, **kwargs):
        """Estimate damage rate by integrating damage_per_cycle over the spectral PDF:
            E[D]/T = nu * integral_0^inf p(S) / N(S) dS
        Can be called as:
            spectral_damage(sigma, nu_p, duration=1.0, method="narrow_band", ...)
            spectral_damage(moments, method="narrow_band", ...)
        """
        method = kwargs.get("method", "narrow_band")
        n_points = kwargs.get("n_points", 1000)
        if len(args) >= 2 and isinstance(args[0], (int, float)):
            sigma = float(args[0])
            nu = float(args[1])
            duration = float(args[2]) if len(args) > 2 else kwargs.get("duration", 1.0)
            moments = None
        else:
            moments = args[0]
            p = spectral_bandwidth_params(moments)
            sigma = p["sigma"]
            nu = p["nu0"] if method == "narrow_band" else p["nup"]
            duration = kwargs.get("duration", 1.0)

        if sigma <= 0.0 or nu <= 0.0:
            res = {"damage": 0.0, "damage_rate": 0.0, "life": math.inf, "nu": nu}
            return SpectralDamageResult(0.0, res)

        s_max = 8.0 * sigma
        s_grid = np.linspace(1e-6 * sigma, s_max, n_points)
        ds = s_grid[1] - s_grid[0]
        if method == "narrow_band" or moments is None:
            # Rayleigh PDF for range S = 2A
            pdf = (s_grid / (4.0 * (sigma ** 2))) * np.exp(- (s_grid ** 2) / (8.0 * (sigma ** 2)))
        elif method == "zhao_baker":
            pdf = zhao_baker_range_pdf(s_grid, moments)
        else:
            pdf = dirlik_range_pdf(s_grid, moments)

        dmg_per_s = np.array([self.damage_per_cycle(s_val) for s_val in s_grid])
        dr = nu * float(np.sum(pdf * dmg_per_s * ds))
        life_est = 1.0 / dr if dr > 0.0 else math.inf
        total_dmg = dr * duration
        res = {
            "damage": total_dmg,
            "damage_rate": dr,
            "life": life_est,
            "nu": nu,
            "method": method,
            "duration": duration,
        }
        return SpectralDamageResult(total_dmg, res)


# ============================================================================
# The full fatigue summary (all methods on one channel)
# ============================================================================

def fatigue_summary(*args, **kwargs):
    """Evaluate ALL closed-form spectral estimators on one channel and return
    a dict of per-method results plus the shared spectral descriptors (RMS,
    rates, width factors).

    Can be called with either:
      1) Spectral moments:
         fatigue_summary(moments, m, C, mean_stress=0.0, ultimate=0.0, ...)
      2) Frequency and PSD arrays:
         fatigue_summary(freqs, psd, m=..., C=..., duration=..., ...)
         fatigue_summary(freqs, psd, m, C, mean_stress=0.0, ultimate=0.0, ...)
    """
    if len(args) >= 2 and (isinstance(args[1], (np.ndarray, list, tuple)) or hasattr(args[1], "__len__")):
        freqs = np.asarray(args[0], dtype=float)
        psd = np.clip(np.asarray(args[1], dtype=float), 0.0, None)
        omega = 2.0 * np.pi * freqs
        from .random_response import spectral_moments
        moments = spectral_moments(omega, psd, nmax=4)
        if len(args) > 2:
            m = float(args[2])
        else:
            m = float(kwargs.get("m", 3.0))
        if len(args) > 3:
            C = float(args[3])
        else:
            C = float(kwargs.get("C", kwargs.get("c", 1.0)))
        mean_stress = float(args[4]) if len(args) > 4 else float(kwargs.get("mean_stress", 0.0))
        ultimate = float(args[5]) if len(args) > 5 else float(kwargs.get("ultimate", kwargs.get("su", 0.0)))
    elif len(args) >= 1:
        moments = args[0]
        if len(args) > 1:
            m = float(args[1])
        else:
            m = float(kwargs.get("m", 3.0))
        if len(args) > 2:
            C = float(args[2])
        else:
            C = float(kwargs.get("C", kwargs.get("c", 1.0)))
        mean_stress = float(args[3]) if len(args) > 3 else float(kwargs.get("mean_stress", 0.0))
        ultimate = float(args[4]) if len(args) > 4 else float(kwargs.get("ultimate", kwargs.get("su", 0.0)))
    else:
        raise TypeError("fatigue_summary() requires at least moments or (freqs, psd)")

    duration = kwargs.get("duration", kwargs.get("t_dur", None))
    if duration is not None:
        duration = float(duration)

    mean_correction = kwargs.get("mean_correction", "goodman")
    yield_strength = kwargs.get("yield_strength", 0.0)
    sigma_f_prime = kwargs.get("sigma_f_prime", 0.0)
    walker_gamma = kwargs.get("walker_gamma", 0.5)

    p = spectral_bandwidth_params(moments)
    nb = narrow_band_damage(moments, m, C, mean_stress=mean_stress, ultimate=ultimate)
    dl = dirlik_damage(moments, m, C, mean_stress=mean_stress, ultimate=ultimate)
    wl = wirsching_light_damage(moments, m, C, mean_stress=mean_stress, ultimate=ultimate)
    tb = tovo_benasciutti_damage(moments, m, C, mean_stress=mean_stress, ultimate=ultimate)
    sb = steinberg_damage(moments, m, C, duration=duration, mean_stress=mean_stress,
                          ultimate=ultimate, mean_correction=mean_correction,
                          yield_strength=yield_strength, sigma_f_prime=sigma_f_prime,
                          walker_gamma=walker_gamma)
    zb = zhao_baker_damage(moments, m, C, duration=duration, mean_stress=mean_stress,
                           ultimate=ultimate, mean_correction=mean_correction,
                           yield_strength=yield_strength, sigma_f_prime=sigma_f_prime,
                           walker_gamma=walker_gamma)

    if duration is not None:
        for model in (nb, dl, wl, tb):
            if "damage_rate" in model and "damage" not in model:
                model["damage"] = model["damage_rate"] * duration
                model["duration"] = duration

    return {
        "params": p,
        "narrow_band": nb,
        "dirlik": dl,
        "wirsching_light": wl,
        "tovo_benasciutti": tb,
        "steinberg": sb,
        "zhao_baker": zb,
    }


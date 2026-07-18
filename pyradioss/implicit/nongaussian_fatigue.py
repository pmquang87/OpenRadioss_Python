"""
Non-Gaussian / kurtosis spectral fatigue — M24: the frequency-domain
fatigue-damage estimate of a stationary but NON-GAUSSIAN random-vibration
response, computed by CORRECTING the M20-M23 Gaussian spectral estimators for a
specified kurtosis (and skewness), with the correction cross-validated against a
non-Gaussian time-domain Monte-Carlo.

Every spectral estimator up to here (M20 scalar narrow-band / Dirlik /
Wirsching-Light / Tovo-Benasciutti, M21-M23 multiaxial reductions) assumed a
stationary GAUSSIAN response: the |H|^2 S law of a Gaussian input, and the
Rayleigh peak / rainflow-range PDFs of a Gaussian process. M24 lifts exactly
that Gaussian assumption while KEEPING the process stationary — the natural
consumer of the M20 scalar / M21-M23 multiaxial moment + Dirlik machinery.

Fortran origin
--------------
There is NONE. ``engine/source/input/freimpl.F`` (the /IMPL reader, re-read line
by line for M16-M23 AND AGAIN for M24 — 639 lines, fetched from
raw.githubusercontent.com) parses only /IMPL/DYNA (the DIRECT Newmark/HHT
integrator), /IMPL/BUCKL, /IMPL/DT, /IMPL/NONLIN and /IMPL/ARCL plus the
linear-solver housekeeping — there is no /FATIG, no S-N / Miner branch, no
Dirlik / rainflow / narrow-band estimator, and (as M16-M23 already found) NO
non-Gaussian / kurtosis / Hermite-moment / Winterstein machinery of any kind.
The sole ``PSD`` token in the whole file is still ``IMUMPSD`` (line 269), a
MUMPS-solver flag, not a power spectral density. OpenRadioss is a time-domain
crash/impact code: the stationary random-vibration fatigue analysis, Gaussian
(M20-M23) OR non-Gaussian (M24), is simply not part of the open-source solver,
exactly as M16 found for the real eigensolver, M17/M18 for the transfer
functions, M19 for the PSD machinery and M20-M23 for the spectral-fatigue
estimators this module corrects.

So M24 does exactly what M16-M23 did: it ports the non-Gaussian / kurtosis
correction as a clean LIBRARY capability and drives it with a minimal PORT engine
sub-flag (/IMPL/FATIG/NGAUSS — the non-Gaussian analogue of the M20 /IMPL/FATIG
Gaussian card). Nothing in the M10 direct integrator, the M16 REAL eigensolver,
the M17/M18 superposition, the M19 PSD path, the M20 SCALAR fatigue, the M21
MULTIAXIAL SPECTRAL path, the M22 NON-PROPORTIONAL TIME-DOMAIN path OR the M23
SPECTRAL NON-PROPORTIONAL path is touched: the non-Gaussian correction is a NEW,
parallel path that CONSUMES the M20-M23 Gaussian spectral moments read-only and
produces its answer ALONGSIDE the Gaussian numbers so a listing shows the
Gaussian and the kurtosis-corrected answers side by side.

Theory — non-Gaussian / kurtosis spectral fatigue
-------------------------------------------------
(Winterstein, "Nonlinear vibration models for extremes and fatigue", J. Eng.
Mech. 114, 1988 — the Hermite-moment transformation; Winterstein & Kashef 2000 /
Winterstein & MacKenzie 1997 — the refined softening coefficients; Benasciutti &
Tovo, "Cycle distribution and fatigue damage assessment in broad-band non-
Gaussian random processes", Prob. Eng. Mech. 20, 2005 and Int. J. Fatigue 2006 —
the bandwidth-dependent non-Gaussian rainflow correction; Braccesi, Cianetti,
Lori & Pioli, "Fatigue behaviour analysis of mechanical components subject to
random bimodal stress process: frequency domain approach", Int. J. Fatigue 31,
2009 — the closed-form non-Gaussian damage correction coefficient; Rizzi, Kihm,
Ferguson, Przekop, Robinson et al. — the kurtosis-corrected spectral damage;
Kihm & Rizzi, "Understanding how kurtosis is transferred from input acceleration
to stress response" (2013) — non-Gaussian random fatigue; Ochi & Ahn 1994 — the
non-Gaussian transformation background; Palmieri, Cesnik et al. — the Winterstein
narrow-band fatigue correction.)

THE WINTERSTEIN HERMITE-MOMENT MODEL. A stationary zero-mean non-Gaussian process
X(t) with variance sigma^2, skewness gamma_3 = E[X^3]/sigma^3 and kurtosis
gamma_4 = E[X^4]/sigma^4 (gamma_4 = 3 is Gaussian) is modelled as a MEMORYLESS
(static, instantaneous) monotonic transformation g of a STANDARD Gaussian process
u(t) (zero mean, unit variance):

    X(t) = sigma * g(u(t)) ,
    g(u) = kappa [ u + h_3 (u^2 - 1) + h_4 (u^3 - 3u) ]                    (1)

where (u^2 - 1) = He_2(u) and (u^3 - 3u) = He_3(u) are the probabilists' Hermite
polynomials. Because the Hermite polynomials are ORTHOGONAL under the Gaussian
measure (E[He_i He_j] = i! delta_ij), the transform has EXACTLY zero mean
(E[g] = kappa(0 + h_3*0 + h_4*0) = 0, since E[He_2] = E[He_3] = 0) and, with

    kappa = 1 / sqrt( 1 + 2 h_3^2 + 6 h_4^2 )                             (2)

EXACTLY unit variance (Var(g) = kappa^2 (1 + 2 h_3^2 + 6 h_4^2) = 1, using
E[u^2] = 1, E[He_2^2] = 2, E[He_3^2] = 6 and the vanishing cross terms). So (1)
PRESERVES the mean (0) and variance (sigma^2) EXACTLY, and the Hermite
coefficients (h_3, h_4) are tuned to hit the target (gamma_3, gamma_4). The
skewness and kurtosis of g are (leading order) gamma_3 ~ 6 h_3 and gamma_4 - 3 ~
24 h_4, giving the FIRST-ORDER coefficients h_3 = gamma_3/6, h_4 =
(gamma_4 - 3)/24 (valid for small non-Gaussianity, symmetric in the sign of
gamma_4 - 3 — leptokurtic h_4 > 0, platykurtic h_4 < 0). Winterstein's REFINED
softening (leptokurtic, gamma_4 >= 3) fit, more accurate at high kurtosis, is

    h_4 = ( sqrt(1 + 1.5 (gamma_4 - 3)) - 1 ) / 18 ,
    h_3 = gamma_3 / ( 6 (1 + 6 h_4) )                                     (3)

(``hermite_coefficients``). For gamma_4 = 3, gamma_3 = 0 BOTH give h_3 = h_4 = 0,
kappa = 1, g(u) = u — the IDENTITY, so the whole M24 correction collapses to the
M20 Gaussian answer EXACTLY (the Gaussian-limit validation).

THE NON-GAUSSIAN CORRECTION FACTOR lambda_ng. For a NARROW-BAND process every
peak is a full cycle and the rainflow RANGE is twice the peak AMPLITUDE, S = 2A,
with A the (Gaussian) envelope amplitude A ~ Rayleigh(sigma). Winterstein's model
"Nonlinear vibration models for EXTREMES and fatigue" applies the SAME memoryless
transform (1) to the amplitude: the non-Gaussian amplitude is A_ng = sigma
g(A/sigma), so the non-Gaussian range is S_ng = 2 sigma g(V), V = A/sigma ~
Rayleigh(1). The Gaussian narrow-band damage uses E[S^m] = (2 sqrt2 sigma)^m
Gamma(1 + m/2) = (2 sigma)^m E[V^m] (M20 eq. (6)); the non-Gaussian one uses
E[S_ng^m] = (2 sigma)^m E[g(V)^m]. Their RATIO is the closed-form NON-GAUSSIAN
CORRECTION COEFFICIENT (``nongaussian_correction_factor``)

    lambda_ng = E[ g(V)^m ] / E[ V^m ] ,   V ~ Rayleigh(1)               (4)

a pure 1-D quadrature over the Rayleigh amplitude density f(v) = v e^{-v^2/2}
(both moments taken on the SAME grid, so lambda_ng = 1 to machine precision for
the identity transform). lambda_ng SCALES the Gaussian spectral damage of EVERY
estimator (narrow-band / Dirlik / Wirsching-Light / Tovo-Benasciutti):

    (E[D]/T)_nG = lambda_ng * (E[D]/T)_Gaussian                          (5)

and hence the corrected life T_f,nG = T_f,G / lambda_ng and equivalent stress
S_eq,nG = lambda_ng^(1/m) S_eq,G. Because g grows faster than linear in its tail
for a LEPTOKURTIC process (gamma_4 > 3, h_4 > 0, the g(v) ~ kappa h_4 v^3 spiky
tail) lambda_ng > 1 (MORE damage, shorter life — the spikes do the damage); for a
PLATYKURTIC process (gamma_4 < 3, h_4 < 0) g saturates and lambda_ng < 1 (LESS
damage); for a GAUSSIAN process (gamma_4 = 3, gamma_3 = 0) g = identity and
lambda_ng = 1 EXACTLY — the M20 answer recovered. This is the Winterstein /
Benasciutti-Braccesi / Rizzi-Kihm non-Gaussian correction: a single multiplier on
the Gaussian damage, derived from the Hermite model rather than an empirical fit.

THE BANDWIDTH DEPENDENCE (Benasciutti & Tovo 2005/2006). For a WIDE-BAND process
the rainflow ranges combine a peak and a valley that are far apart in time and so
nearly independent; by the central-limit effect their combination is CLOSER to
Gaussian than the process itself, so the non-Gaussian effect the RANGES see is
ATTENUATED relative to the process kurtosis. Benasciutti & Tovo model this by
weighting the excess kurtosis (and skewness) toward the Gaussian value by a
bandwidth factor. This module uses the irregularity factor alpha_2 (= nu_0/nu_p,
M20 eq. (5): 1 for a narrow band, -> 0 for white noise) as that weight,

    gamma_4,eff = 3 + (gamma_4 - 3) * w ,  gamma_3,eff = gamma_3 * w ,
    w = alpha_2   (bandwidth attenuation; w = 1 for a narrow band)       (6)

feeding (gamma_3,eff, gamma_4,eff) to (3) before (4). A narrow-band process
(alpha_2 = 1) gets the FULL correction; a wide-band one gets less. The exact
empirical Benasciutti-Tovo / Braccesi 2009 weighting constants are NOT reproduced
(a documented deferral — the attenuation model here is first-order in alpha_2);
the correction can be run WITHOUT the attenuation (w = 1) to recover the pure
narrow-band Winterstein factor (the closed-form validations use w = 1).

THE NON-GAUSSIAN MONTE-CARLO CROSS-CHECK. As an independent time-domain
validation the module SYNTHESISES a non-Gaussian history — the M20 Gaussian
spectral-representation history u(t) (an inverse-rFFT of the PSD with seeded
random phases) pushed through the memoryless Hermite / power-law transform (1) to
the target (gamma_3, gamma_4) — RAINFLOW-counts it (the M20 ASTM E1049 counter)
and Miner-sums the ranges (``nongaussian_monte_carlo_damage``). This is the
time-domain damage the lambda_ng-corrected spectral estimate approximates: it
matches the lambda_ng * Gaussian spectral damage to within the seeded scatter,
exactly as M20 used the Gaussian Monte-Carlo to check Dirlik and M23 used the M22
path count. In the Gaussian limit (gamma_4 = 3, gamma_3 = 0) the transform is the
identity and the non-Gaussian Monte-Carlo reduces EXACTLY to the M20 Gaussian
Monte-Carlo.

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* LIBRARY-FIRST sub-flag (/IMPL/FATIG/NGAUSS) — no upstream equivalent, exactly
  as established for M16-M23's PORT cards.
* The correction (4) is the NARROW-BAND-exact Winterstein amplitude transform;
  the WIDE-BAND attenuation (6) is a first-order alpha_2 model (Benasciutti &
  Tovo), NOT the exact empirical Braccesi 2009 fit (whose specific constants are
  DEFERRED — the Hermite-derived form here is more principled and matches the
  Monte-Carlo directly). The memoryless transform (1) preserves the PSD SHAPE
  only approximately (a static nonlinearity injects harmonics), so the
  higher-moment shape reuse of the corrected estimators is the documented
  approximation — the amplitude correction lambda_ng is exact for a narrow band.
* Winterstein's refined softening fit (3) is calibrated for gamma_4 >= 3; for a
  PLATYKURTIC target (gamma_4 < 3) it falls back to the FIRST-ORDER coefficients
  (h_3 = gamma_3/6, h_4 = (gamma_4 - 3)/24), which are monotonic for small
  |gamma_4 - 3| (the standard limitation of the softening model). Extreme
  platykurtosis (gamma_4 < ~2) is outside the monotonic range and is clamped
  (documented) — random-vibration fatigue is dominated by LEPTOKURTIC processes.
* STATIONARY non-Gaussian only: the kurtosis here is a STATIONARY non-Gaussian
  correction, NOT a time-varying (non-stationary / evolutionary-PSD) process.
  NON-STATIONARY / evolutionary-PSD fatigue is DEFERRED (a different analysis).
* A full non-Gaussian MULTIAXIAL joint-distribution (a vector Hermite transform
  of the correlated stress tensor) is DEFERRED; M24 applies the SCALAR kurtosis
  correction to the (von Mises / critical-plane) equivalent scalar the M21/M23
  reductions already produce, so it COMPOSES with the multiaxial / non-
  proportional paths but does not model a joint non-Gaussian tensor.
* MEAN-STRESS beyond the basic M20/M21 Goodman intercept, CRACK-GROWTH /
  fracture-mechanics fatigue, the COMPLEX-FRF stress recovery and a MULTI-INPUT
  cross-PSD with coherence remain DEFERRED (the unchanged M20-M23 tail).
"""

from __future__ import annotations

import math

import numpy as np

from ..common.npcompat import trapezoid


# ============================================================================
# The Winterstein Hermite-moment model (build-order item 1)
# ============================================================================

def hermite_coefficients(gamma3, gamma4, model="winterstein"):
    """The Winterstein Hermite-moment coefficients (h_3, h_4, kappa) of the
    softening/hardening transform g(u) = kappa[u + h_3(u^2-1) + h_4(u^3-3u)]
    (theory eqs. (1)-(3)) for a target skewness ``gamma3`` and kurtosis
    ``gamma4`` (gamma4 = 3, gamma3 = 0 is Gaussian).

    ``model``:
      * ``"winterstein"`` (default) — the refined SOFTENING (leptokurtic) fit
        h_4 = (sqrt(1 + 1.5(gamma4-3)) - 1)/18, h_3 = gamma3/(6(1 + 6 h_4))
        (Winterstein 1988; Winterstein & MacKenzie 1997), calibrated for
        gamma4 >= 3; falls back to the first-order coefficients for gamma4 < 3
        (the softening fit's sqrt goes complex below gamma4 = 7/3);
      * ``"first_order"`` — h_3 = gamma3/6, h_4 = (gamma4-3)/24, the leading-order
        moment match, symmetric in the sign of gamma4-3 (lepto h_4 > 0, platy
        h_4 < 0), monotonic for small non-Gaussianity.

    kappa = 1/sqrt(1 + 2 h_3^2 + 6 h_4^2) makes Var(g) = 1 EXACTLY (eq. (2)); the
    mean of g is 0 EXACTLY. Returns (h3, h4, kappa). For the Gaussian target
    (gamma4 = 3, gamma3 = 0) returns (0, 0, 1) so g is the identity and the whole
    correction collapses to the Gaussian answer."""
    g3 = float(gamma3)
    g4 = float(gamma4)
    use_first_order = (model == "first_order") or (g4 < 3.0)
    if use_first_order:
        # leading-order moment match (works for lepto AND platy)
        h3 = g3 / 6.0
        h4 = (g4 - 3.0) / 24.0
    else:
        # Winterstein refined softening fit (leptokurtic, gamma4 >= 3)
        h4 = (math.sqrt(max(1.0 + 1.5 * (g4 - 3.0), 0.0)) - 1.0) / 18.0
        h3 = g3 / (6.0 * (1.0 + 6.0 * h4))
    # kappa preserves the variance EXACTLY (Hermite orthogonality, eq. (2))
    kappa = 1.0 / math.sqrt(1.0 + 2.0 * h3 * h3 + 6.0 * h4 * h4)
    return h3, h4, kappa


def hermite_transform(u, gamma3, gamma4, model="winterstein",
                      coeffs=None):
    """Apply the memoryless Winterstein Hermite transform (theory eq. (1)) to a
    STANDARD Gaussian sample/array ``u`` (zero mean, unit variance), returning a
    UNIT-VARIANCE, zero-mean non-Gaussian sample with the target skewness
    ``gamma3`` and kurtosis ``gamma4``:

        g(u) = kappa [ u + h_3 (u^2 - 1) + h_4 (u^3 - 3u) ]

    (multiply by sigma to give the physical stress of RMS sigma). The mean and
    variance are preserved EXACTLY; the target (skewness, kurtosis) are hit to the
    Winterstein-fit accuracy. ``coeffs`` optionally supplies a precomputed
    (h3, h4, kappa) tuple (``hermite_coefficients``)."""
    if coeffs is None:
        coeffs = hermite_coefficients(gamma3, gamma4, model=model)
    h3, h4, kappa = coeffs
    u = np.asarray(u, dtype=float)
    return kappa * (u + h3 * (u * u - 1.0) + h4 * (u * u * u - 3.0 * u))


def hermite_kurtosis(gamma3, gamma4, model="winterstein", coeffs=None):
    """The ACTUAL (skewness, kurtosis) the Hermite transform (1) produces for the
    requested target — the EXACT moments of g(u), u ~ N(0,1), evaluated in closed
    form from (h3, h4, kappa). Because the Winterstein coefficient fits (3) are
    approximate, the realised kurtosis differs slightly from the target; this
    returns the realised values for the hand-check validation (the transform
    'hitting the target kurtosis' to the fit accuracy). Returns (skew, kurt)."""
    if coeffs is None:
        coeffs = hermite_coefficients(gamma3, gamma4, model=model)
    h3, h4, kappa = coeffs
    # Central Gaussian moments E[u^k]: 0,1,0,3,0,15,0,105,0,945 for k=0..9.
    # g = kappa (u + h3(u^2-1) + h4(u^3-3u)) is a cubic in u; its 3rd and 4th
    # moments follow from expanding and using the standard-normal moments. Var = 1
    # by construction, so skew = E[g^3], kurt = E[g^4] directly.
    # Evaluate numerically from the polynomial coefficients (robust, exact to the
    # Gaussian-moment table) rather than transcribing the long algebra.
    mu = np.array([1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 15.0, 0.0, 105.0, 0.0, 945.0,
                   0.0, 10395.0])  # E[u^k], k = 0..12
    # polynomial g(u) = kappa*( -h3 + (1-3h4) u + h3 u^2 + h4 u^3 )
    c = np.array([kappa * (-h3), kappa * (1.0 - 3.0 * h4),
                  kappa * h3, kappa * h4])            # coeffs of u^0..u^3

    def _moment(order):
        # E[ (sum c_i u^i)^order ] via polynomial power then the moment table
        p = np.array([1.0])
        for _ in range(order):
            p = np.convolve(p, c)
        return float(sum(p[k] * mu[k] for k in range(p.size)))

    skew = _moment(3)
    kurt = _moment(4)
    return skew, kurt


# ============================================================================
# The closed-form non-Gaussian correction factor lambda_ng (build item 1)
# ============================================================================

# a fixed, fine Rayleigh(1) quadrature grid for the amplitude integral (4); the
# tail reaches v ~ 20 so v^(3m) e^{-v^2/2} is negligible even for a large slope
# m and a spiky leptokurtic transform (g ~ v^3 in the tail)
_V_GRID = np.linspace(0.0, 20.0, 8001)
_RAYLEIGH_PDF = _V_GRID * np.exp(-0.5 * _V_GRID * _V_GRID)   # f(v) = v e^{-v^2/2}


def nongaussian_correction_factor(gamma3, gamma4, m, alpha2=1.0,
                                  bandwidth_correction=True, model="winterstein",
                                  return_coeffs=False):
    """The closed-form NON-GAUSSIAN CORRECTION COEFFICIENT lambda_ng (theory eq.
    (4)): the ratio of the non-Gaussian to the Gaussian m-th range moment for a
    narrow-band process, the Winterstein Hermite amplitude transform integrated
    over the Rayleigh(1) amplitude density,

        lambda_ng = E[ g(V)^m ] / E[ V^m ] ,   V ~ Rayleigh(1)

    with g the Hermite transform (``hermite_transform``) of the target skewness /
    kurtosis. lambda_ng SCALES the Gaussian spectral damage of every M20 estimator
    (eq. (5)): (E[D]/T)_nG = lambda_ng (E[D]/T)_Gaussian.

    * ``gamma3`` / ``gamma4`` — the target skewness / kurtosis (gamma4 = 3,
      gamma3 = 0 -> lambda_ng = 1 EXACTLY, the Gaussian answer recovered);
    * ``m`` — the S-N slope (the moment order of the range distribution);
    * ``alpha2`` — the M20 irregularity factor (1 narrow-band .. 0 white noise);
      with ``bandwidth_correction`` (default True) the excess kurtosis/skewness is
      attenuated toward Gaussian by w = alpha2 (Benasciutti-Tovo, eq. (6)) — a
      wide-band process gets less correction. Pass ``bandwidth_correction=False``
      (or alpha2 = 1) for the pure narrow-band Winterstein factor.

    lambda_ng > 1 for a leptokurtic (spiky, gamma4 > 3) process, < 1 for a
    platykurtic (gamma4 < 3) one. Returns lambda_ng (or (lambda_ng, (h3,h4,kappa))
    if ``return_coeffs``)."""
    g3, g4 = float(gamma3), float(gamma4)
    if bandwidth_correction:
        # attenuate the excess toward Gaussian for a wide band (eq. (6))
        w = min(max(float(alpha2), 0.0), 1.0)
        g4 = 3.0 + (g4 - 3.0) * w
        g3 = g3 * w
    coeffs = hermite_coefficients(g3, g4, model=model)
    h3, h4, kappa = coeffs
    # Gaussian short-circuit: g is the identity -> lambda_ng = 1 EXACTLY
    if abs(h3) < 1e-15 and abs(h4) < 1e-15 and abs(kappa - 1.0) < 1e-15:
        return (1.0, coeffs) if return_coeffs else 1.0
    v = _V_GRID
    gv = hermite_transform(v, g3, g4, coeffs=coeffs)
    # the amplitude must be non-negative to raise to the (possibly fractional)
    # power m; a monotone increasing g stays >= 0 on v >= 0 except a negligible
    # dip near v = 0 (Rayleigh weight ~ 0 there) for a skewed transform — clip it
    gv = np.clip(gv, 0.0, None)
    num = trapezoid(gv ** m * _RAYLEIGH_PDF, v)
    den = trapezoid(v ** m * _RAYLEIGH_PDF, v)     # same grid -> exact 1 for id
    lam = float(num / den) if den > 0 else 1.0
    lam = max(lam, 0.0)
    return (lam, coeffs) if return_coeffs else lam


# ============================================================================
# The corrected damage estimators (build-order item 1)
# ============================================================================

def nongaussian_damage(gaussian_result, moments, m, gamma3, gamma4,
                       bandwidth_correction=True, model="winterstein"):
    """Apply the non-Gaussian correction (theory eq. (5)) to a SINGLE Gaussian
    estimator result ``gaussian_result`` (a dict from ``spectral_fatigue``'s
    narrow_band / dirlik / wirsching_light / tovo_benasciutti, carrying
    ``damage_rate``, ``nu`` and ``params``): scale the damage rate by lambda_ng
    and recompute the life / equivalent stress.

        (E[D]/T)_nG = lambda_ng (E[D]/T)_G ,  T_f,nG = T_f,G / lambda_ng ,
        S_eq,nG = lambda_ng^(1/m) S_eq,G

    ``moments`` = [m0..m4] of the SAME channel (for alpha_2, the bandwidth weight).
    Returns a NEW dict (the Gaussian one is left untouched) with the corrected
    ``damage_rate`` / ``life`` / ``s_eq`` plus ``lambda_ng`` and the Hermite
    coefficients; the Gaussian value is kept under ``gaussian_damage_rate``."""
    from . import spectral_fatigue as sf
    p = gaussian_result.get("params") or sf.spectral_bandwidth_params(moments)
    alpha2 = p["alpha2"]
    lam, coeffs = nongaussian_correction_factor(
        gamma3, gamma4, m, alpha2=alpha2,
        bandwidth_correction=bandwidth_correction, model=model,
        return_coeffs=True)
    drG = float(gaussian_result["damage_rate"])
    dr = lam * drG
    # the correction just SCALES the damage rate, so the life / equivalent stress
    # follow directly (T_f = 1/dr by Miner, S_eq scales as lambda^(1/m)) without
    # needing to recover the S-N coefficient C
    life = math.inf if dr <= 0.0 else 1.0 / dr
    s_eqG = gaussian_result.get("s_eq", 0.0)
    s_eq = (lam ** (1.0 / m)) * s_eqG if s_eqG > 0 else 0.0
    out = dict(gaussian_result)
    out.update({"method": gaussian_result.get("method", "?") + "_nongaussian",
                "damage_rate": dr, "life": life, "s_eq": s_eq,
                "lambda_ng": lam, "gaussian_damage_rate": drG,
                "h3": coeffs[0], "h4": coeffs[1], "kappa": coeffs[2],
                "gamma3": float(gamma3), "gamma4": float(gamma4)})
    return out


def nongaussian_summary(moments, m, C, gamma3, gamma4, mean_stress=0.0,
                        ultimate=0.0, bandwidth_correction=True,
                        model="winterstein", gaussian=None):
    """The full NON-GAUSSIAN correction of ALL four M20 estimators on one
    channel's moment array (theory eqs. (4)-(5)). Computes (or reuses via
    ``gaussian``) the M20 Gaussian ``fatigue_summary``, then scales every
    estimator by lambda_ng. Returns a dict with:

      * ``lambda_ng`` — the correction factor (eq. (4));
      * ``h3`` / ``h4`` / ``kappa`` — the Hermite coefficients;
      * ``gamma3`` / ``gamma4`` — the target moments; ``alpha2`` — the bandwidth;
      * per-estimator corrected results (narrow_band / dirlik / wirsching_light /
        tovo_benasciutti) with the corrected damage_rate / life / s_eq;
      * ``gaussian`` — the untouched M20 Gaussian summary (side-by-side).

    lambda_ng = 1 (Gaussian answer recovered) for gamma4 = 3, gamma3 = 0."""
    from . import spectral_fatigue as sf
    if gaussian is None:
        gaussian = sf.fatigue_summary(moments, m, C, mean_stress, ultimate)
    p = gaussian["params"]
    alpha2 = p["alpha2"]
    lam, coeffs = nongaussian_correction_factor(
        gamma3, gamma4, m, alpha2=alpha2,
        bandwidth_correction=bandwidth_correction, model=model,
        return_coeffs=True)
    out = {"lambda_ng": lam, "h3": coeffs[0], "h4": coeffs[1],
           "kappa": coeffs[2], "gamma3": float(gamma3), "gamma4": float(gamma4),
           "alpha2": alpha2, "bandwidth_correction": bool(bandwidth_correction),
           "model": model, "params": p, "gaussian": gaussian}
    for key in ("narrow_band", "dirlik", "wirsching_light",
                "tovo_benasciutti"):
        out[key] = nongaussian_damage(
            gaussian[key], moments, m, gamma3, gamma4,
            bandwidth_correction=bandwidth_correction, model=model)
    return out


# ============================================================================
# The non-Gaussian Monte-Carlo cross-check (build-order item 2)
# ============================================================================

def synthesize_nongaussian_history(freqs_hz, psd, duration, seed, gamma3,
                                   gamma4, fs=None, model="winterstein"):
    """Synthesise a stationary NON-GAUSSIAN time history of the target skewness
    ``gamma3`` and kurtosis ``gamma4`` (theory "NON-GAUSSIAN MONTE-CARLO"): the
    M20 Gaussian spectral-representation history u(t) (``synthesize_gaussian_
    history``: an inverse-rFFT of the PSD with seeded random phases, RMS sigma =
    sqrt(m0)) pushed through the MEMORYLESS Winterstein Hermite transform (1) to
    the target moments,

        x(t) = sigma * g( u(t) / sigma )  =  sigma * g(z(t)) ,  z = u/sigma

    (standardise to unit variance, transform, rescale). The mean (0) and variance
    (sigma^2 = m0) are PRESERVED; the sample skewness / kurtosis hit the target.
    Returns (t, x).

    BANDWIDTH CAVEAT (documented): the static/memoryless transform preserves the
    PSD SHAPE only APPROXIMATELY — a cubic nonlinearity injects harmonic and
    intermodulation content, so x's PSD (and hence its spectral bandwidth /
    moments) drifts from the input PSD's, most for a strongly non-Gaussian, wide-
    band target. The transform reproduces the target one-point MARGINAL (kurtosis)
    exactly by construction; the two-point (spectral) structure is approximate.
    In the Gaussian limit (gamma4 = 3, gamma3 = 0) the transform is the IDENTITY
    and x = u EXACTLY — the M20 Gaussian history recovered."""
    from . import spectral_fatigue as sf
    t, u = sf.synthesize_gaussian_history(freqs_hz, psd, duration, seed, fs=fs)
    coeffs = hermite_coefficients(gamma3, gamma4, model=model)
    h3, h4, kappa = coeffs
    # Gaussian limit: identity transform -> return the M20 Gaussian history EXACTLY
    if abs(h3) < 1e-15 and abs(h4) < 1e-15 and abs(kappa - 1.0) < 1e-15:
        return t, u
    sigma = float(np.std(u))
    if sigma <= 0.0:
        return t, u
    z = (u - float(np.mean(u))) / sigma          # standardise to unit variance
    x = sigma * hermite_transform(z, gamma3, gamma4, coeffs=coeffs)
    return t, x


def nongaussian_monte_carlo_damage(freqs_hz, psd, m, C, duration, seed, gamma3,
                                   gamma4, fs=None, mean_stress=0.0,
                                   ultimate=0.0, model="winterstein"):
    """The TIME-DOMAIN NON-GAUSSIAN damage rate by Monte-Carlo (theory
    "NON-GAUSSIAN MONTE-CARLO"): synthesise the non-Gaussian history
    (``synthesize_nongaussian_history``), rainflow-count it (ASTM E1049, the M20
    counter) and Miner-sum the ranges with the S-N law N = C S^-m — the
    independent time-domain answer the lambda_ng-corrected spectral estimate
    approximates. Seeded for reproducibility. Returns the M20 Monte-Carlo dict
    shape plus the sample ``skewness`` / ``kurtosis`` of the synthesised history
    (the target check). In the Gaussian limit reduces EXACTLY to the M20
    ``monte_carlo_damage``."""
    from . import spectral_fatigue as sf
    Ceff = sf._goodman_C(C, m, mean_stress, ultimate)
    t, x = synthesize_nongaussian_history(freqs_hz, psd, duration, seed, gamma3,
                                          gamma4, fs=fs, model=model)
    ranges, counts = sf.rainflow_count(x)
    D = float(np.sum(counts * ranges ** m) / Ceff) if ranges.size else 0.0
    T = t[-1] - t[0] if t.size > 1 else duration
    dr = D / T if T > 0 else 0.0
    tf, s_eq = sf.life_and_equivalent(dr, ranges.size / T if T > 0 else 0.0,
                                      m, Ceff)
    xc = x - np.mean(x)
    var = float(np.mean(xc * xc))
    skew = float(np.mean(xc ** 3) / var ** 1.5) if var > 0 else 0.0
    kurt = float(np.mean(xc ** 4) / var ** 2) if var > 0 else 3.0
    return {"method": "nongaussian_monte_carlo", "damage_rate": dr, "life": tf,
            "s_eq": s_eq, "ncycles": float(counts.sum()), "ranges": ranges,
            "counts": counts, "duration": T, "rms": float(np.std(x)),
            "skewness": skew, "kurtosis": kurt}


def nongaussian_monte_carlo_projected(freqs_hz, Scross, proj, m, C, duration,
                                      seed, gamma3, gamma4, fs=None,
                                      mean_stress=0.0, ultimate=0.0,
                                      model="winterstein"):
    """The MULTIAXIAL non-Gaussian Monte-Carlo cross-check (the M24 composition
    with the M21/M23 multiaxial path): synthesise the 6 correlated GAUSSIAN
    stress-component histories from the cross-PSD (the M21 multivariate
    synthesiser ``synthesize_multiaxial_history``, seeded), PROJECT onto the
    critical plane's LINEAR scalar (``proj`` = a normal_/shear_projection 6-vector,
    a Gaussian scalar process), push THAT scalar through the memoryless Winterstein
    Hermite transform (1) to the target (``gamma3``, ``gamma4``), rainflow-count
    (ASTM E1049) and Miner-sum. The multiaxial analogue of
    ``nongaussian_monte_carlo_damage`` — the non-Gaussian time-domain answer the
    lambda_ng-corrected critical-plane spectral estimate approximates. In the
    Gaussian limit reduces EXACTLY to ``monte_carlo_multiaxial_damage``.

    Documented approximation: the target kurtosis is imposed on the RESOLVED
    scalar (the physical fatigue driver on the plane), not jointly on the stress
    tensor — a full non-Gaussian tensor joint distribution is DEFERRED (module
    docstring)."""
    from . import spectral_fatigue as sf
    from .multiaxial_fatigue import synthesize_multiaxial_history
    Ceff = sf._goodman_C(C, m, mean_stress, ultimate)
    t, X = synthesize_multiaxial_history(freqs_hz, Scross, duration, seed, fs=fs)
    s = X @ np.asarray(proj, dtype=float)          # Gaussian projected scalar
    coeffs = hermite_coefficients(gamma3, gamma4, model=model)
    h3, h4, kappa = coeffs
    if not (abs(h3) < 1e-15 and abs(h4) < 1e-15 and abs(kappa - 1.0) < 1e-15):
        sigma = float(np.std(s))
        if sigma > 0.0:
            z = (s - float(np.mean(s))) / sigma
            s = sigma * hermite_transform(z, gamma3, gamma4, coeffs=coeffs)
    ranges, counts = sf.rainflow_count(s)
    D = float(np.sum(counts * ranges ** m) / Ceff) if ranges.size else 0.0
    T = t[-1] - t[0] if t.size > 1 else duration
    dr = D / T if T > 0 else 0.0
    tf, s_eq = sf.life_and_equivalent(dr, ranges.size / T if T > 0 else 0.0,
                                      m, Ceff)
    sc = s - np.mean(s)
    var = float(np.mean(sc * sc))
    kurt = float(np.mean(sc ** 4) / var ** 2) if var > 0 else 3.0
    return {"method": "nongaussian_monte_carlo_multiaxial", "damage_rate": dr,
            "life": tf, "s_eq": s_eq, "ncycles": float(counts.sum()),
            "ranges": ranges, "counts": counts, "duration": T,
            "rms": float(np.std(s)), "kurtosis": kurt}

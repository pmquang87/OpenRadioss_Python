"""
Non-Gaussian JOINT-TENSOR distribution — M33: a VECTOR (multivariate)
Winterstein-Hermite / translation-process transform of the CORRELATED 6x6
stress-tensor process, so the target kurtosis (and skewness) is imposed JOINTLY on
the stress-tensor COMPONENTS (preserving the full 6x6 covariance / cross-PSD), NOT on
the already-resolved equivalent scalar. Where M24/M32 imposed gamma_4 on the von-Mises
/ critical-plane SCALAR (a scalar Hermite transform), M33 imposes it on the TENSOR and
lets the critical-plane / von-Mises reduction INHERIT the INDUCED non-Gaussianity from
the joint tensor statistics — a genuinely multiaxial non-Gaussian distribution, applied
along the M31/M32 CONTINUOUS Wigner-Ville instantaneous spectrum (composing with M32's
time-varying gamma_4(t) schedule), reduced per instant and Palmgren-Miner INTEGRATED,
cross-validated by a multivariate non-Gaussian non-stationary Monte-Carlo.

WHERE M33 SITS — THE JOINT-TENSOR LIFT OF M32. Two independent lines meet here:
  * M24 (``nongaussian_fatigue.py``) / M32 (``nongaussian_wigner_ville_fatigue.py``)
    imposed the target kurtosis on the RESOLVED EQUIVALENT SCALAR (the von-Mises /
    critical-plane scalar the M21/M27 reductions produce): the closed-form
    Winterstein-Hermite factor lambda_ng SCALES the Gaussian spectral damage, computed
    from the SCALAR's own kurtosis. M32 let that scalar kurtosis gamma_4(t) drift with
    time along the M31 continuous spectrum. Both are SCALAR-equivalent corrections — the
    tensor is reduced FIRST (to a Gaussian scalar), THEN a kurtosis is imposed on it.
  * M27 (``joint_evolutionary_fatigue.py``) / M31 (``wigner_ville_fatigue.py``) gave the
    GAUSSIAN 6x6 joint tensor a windowed / continuous instantaneous spectrum, reduced
    per instant with the critical plane RE-SEARCHED from the window's own tensor.
M33 lifts the correction from the SCALAR to the TENSOR: it applies a COMPONENT-WISE
(vector) Winterstein-Hermite / translation-process transform to the CORRELATED 6x6
stress tensor BEFORE the reduction, imposing a per-component (or joint) target kurtosis
gamma_4_c on each tensor COMPONENT while PRESERVING the marginal variances (and, to
leading order, the full 6x6 covariance / cross-PSD). The critical-plane / von-Mises
reduction then INHERITS an INDUCED kurtosis derived from the JOINT tensor statistics —
because the resolved-plane scalar s = p^T sigma is a LINEAR PROJECTION of a
component-wise-transformed correlated Gaussian, its kurtosis is a closed-form function
of the per-component kurtoses AND the 6x6 correlation, computed EXACTLY here (matching
the multivariate Monte-Carlo) and fed to the M24 lambda_ng closed form. This is the
FIRST item M32 (and M24, M31) deferred: a full non-Gaussian instantaneous-tensor JOINT
distribution, the vector Hermite transform of the correlated tensor at each instant.

THE M33 <-> M32 / M31 REDUCTIONS (built in, exact, asserted like every milestone since
M14):
  * gamma_4_c == 3, gamma_3_c == 0 on EVERY component (a Gaussian tensor) -> every
    component transform is the IDENTITY, the induced projection kurtosis is 3 EXACTLY,
    lambda_ng == 1, and the M33 path DELEGATES to the M31/M27 Gaussian tensor answer
    BYTE-IDENTICALLY (guaranteeing byte-identity, exactly as M32 delegated to M31, M31
    to M27, M27 to M26).
  * the SCALAR-EQUIVALENT limit — imposing the kurtosis on the RESOLVED scalar rather
    than the tensor components — recovers the M32/M24 correction EXACTLY: M33 reports the
    M32 answer (``nongaussian_wigner_ville_tensor_summary`` with the scalar target) as
    the ``scalar_equivalent`` side-by-side entry by DELEGATION, so it is byte-identical
    to a direct M32 call. The genuinely JOINT case — per-component kurtoses that DIFFER,
    so the induced resolved-plane kurtosis (and damage) DIFFERS from imposing gamma_4
    directly on the scalar — is the M33 <-> M32 boundary made explicit.
  * the INDUCED-KURTOSIS closed form matches the multivariate non-Gaussian Monte-Carlo
    (both use the underlying-Gaussian correlation = the target tensor correlation), and
    for a SINGLE-component (uniaxial) projection the induced kurtosis reduces EXACTLY to
    the M24 realised kurtosis ``hermite_kurtosis`` of that component's transform (the
    scalar Hermite transform recovered as the 1-component special case).

Fortran origin
--------------
There is NONE. ``engine/source/input/freimpl.F`` (the /IMPL reader, re-read line by
line for M16-M32 AND AGAIN for M33 — 639 lines, fetched from
raw.githubusercontent.com) parses only /IMPL/DYNA (the DIRECT Newmark/HHT integrator),
/IMPL/BUCKL, /IMPL/DT, /IMPL/NONLIN and /IMPL/ARCL plus the linear-solver housekeeping —
there is no /FATIG, no S-N / Miner branch, no PSD / spectral / random-vibration path,
and (as M16-M32 already found, line by line) NOTHING time-frequency / Wigner-Ville /
non-Gaussian / Hermite / Winterstein / multivariate-translation / joint-tensor of any
kind. The sole ``PSD`` token in the whole file is still ``IMUMPSD`` (line 269,
``IF (ISOLV==3) IMUMPSD=L_LIM``), a MUMPS-solver flag, NOT a power spectral density.
OpenRadioss is a time-domain crash/impact code: the random-vibration fatigue analysis —
stationary Gaussian (M20-M23), stationary non-Gaussian SCALAR (M24), separable
non-stationary (M25), evolutionary windowed (M26-M30), the CONTINUOUS Wigner-Ville
instantaneous spectrum (M31), the time-VARYING non-Gaussian SCALAR instantaneous
spectrum (M32) OR the JOINT-tensor non-Gaussian distribution (M33) — is simply not part
of the open-source solver, exactly as every spectral milestone since M16 recorded.

So M33 does exactly what M16-M32 did: it ports the joint-tensor non-Gaussian
distribution as a clean LIBRARY capability EXTENDING the M32 time-varying line + the
M27/M31 joint-tensor line, and drives it with a minimal PORT engine sub-flag
(/IMPL/FATIG/NGAUSS composing with /JOINT + /WVILLE, plus a per-component or joint
kurtosis target). Nothing in the M10 integrator, the M16 eigensolver, the M17/M18
superposition, the M19 PSD path, the M20-M27 reductions, the M24/M32 SCALAR non-Gaussian
correction OR the M27/M31 Gaussian tensor spectrum is touched: the JOINT non-Gaussian
tensor path is a NEW, parallel path that CONSUMES the M21 tensor / plane machinery, the
M24 closed form and the M31/M32 continuous machinery (all read-only) and produces its
answer ALONGSIDE the M32 equivalent-scalar and the M31/M27 Gaussian numbers so a listing
shows the Gaussian tensor, the equivalent-scalar-non-Gaussian and the JOINT-tensor
non-Gaussian answers side by side (the M24/M32 equivalent-scalar and the M27/M31
Gaussian paths stay byte-identical — the JOINT path is EXACTLY the scalar-equivalent
correction when the kurtosis is imposed on the scalar, and EXACTLY the Gaussian answer
in the gamma_4 == 3 limit, asserted).

Theory — the joint non-Gaussian tensor (translation-process) model
------------------------------------------------------------------
(Grigoriu, "Applied Non-Gaussian Processes", Prentice-Hall 1995, and "Simulation of
stationary non-Gaussian translation processes", J. Eng. Mech. 124, 1998 — the
memoryless (translation) process X(t) = g(U(t)), U Gaussian, and the correlation
DISTORTION between the Gaussian and the translated process; Winterstein, "Nonlinear
vibration models for extremes and fatigue", J. Eng. Mech. 114, 1988 — the Hermite-moment
transform g(u) = kappa[u + h_3(u^2-1) + h_4(u^3-3u)] and the softening coefficients, the
SCALAR base M24 ports; Lutes & Sarkani, "Random Vibrations: Analysis of Structural and
Mechanical Systems", 2004 — the VECTOR translation process and the projection of a
non-Gaussian vector; Sarkani, Kihm, Rizzi — the multiaxial non-Gaussian random fatigue;
the multivariate Hermite / diagram (Wick) moment formula E[He_p(U_i) He_q(U_j) ...] =
sum over matchings without self-contractions of prod R_ij — Isserlis 1918 / Wick 1950 /
the Mehler kernel; Benasciutti & Tovo 2005/2006 / Braccesi 2009 / Rizzi & Kihm 2013 —
the bandwidth-dependent non-Gaussian rainflow correction the resolved scalar inherits;
the M21/M27 critical-plane tensor reduction; the M24 scalar Hermite base; the M32
time-varying base.)

THE VECTOR (COMPONENT-WISE) HERMITE / TRANSLATION TRANSFORM. The Gaussian 6x6 tensor
cross-PSD gives, per instant, the covariance matrix M_0 = E[sigma sigma^T] of the six
Voigt stress components. Standardise: sigma_c = sigma_c^{rms} U_c with U =
(U_1..U_6) ~ N(0, R) a zero-mean UNIT-variance jointly Gaussian VECTOR of correlation
R_cc' = M_0[c,c'] / (sigma_c^{rms} sigma_c'^{rms}). The JOINT non-Gaussian tensor
imposes a per-component target kurtosis gamma_4_c (skewness gamma_3_c) by pushing EACH
component through its OWN memoryless Winterstein-Hermite transform (a VECTOR translation
process, Grigoriu 1998; Lutes & Sarkani):

    X_c(t) = sigma_c^{rms} g_c(U_c(t)) ,
    g_c(u) = kappa_c [ u + h_{3,c}(u^2-1) + h_{4,c}(u^3-3u) ]                 (1)

with (h_{3,c}, h_{4,c}, kappa_c) the M24 ``hermite_coefficients`` of (gamma_3_c,
gamma_4_c). Because each g_c has mean 0 and variance 1 EXACTLY (kappa_c normalisation,
Hermite orthogonality), (1) PRESERVES each component's MARGINAL variance sigma_c^2
EXACTLY and imposes its target kurtosis. The transformed CROSS-covariance is (Mehler /
diagram formula, He_p orthogonality E[He_p(U_c) He_q(U_c')] = delta_pq p! R^p)

    Cov(X_c, X_c') = sigma_c^{rms} sigma_c'^{rms} kappa_c kappa_c'
        [ R_cc' + 2 h_{3,c} h_{3,c'} R_cc'^2 + 6 h_{4,c} h_{4,c'} R_cc'^3 ]   (2)

— its DIAGONAL (c == c', R = 1) is sigma_c^2 EXACTLY, and its off-diagonal equals the
target M_0[c,c'] to LEADING order (the dominant kappa_c kappa_c' R term), with the
h-dependent R^2/R^3 corrections the documented second-order translation DISTORTION
(``translation_process_covariance`` reports the max preservation error). This is the
multiaxial analogue of M24's "the memoryless transform preserves the PSD SHAPE only
approximately"; the MARGINALS (each component's variance AND kurtosis) are exact by
construction, the CROSS-structure is leading-order (Grigoriu's translation correlation
distortion — the exact correlation-inversion that would restore M_0[c,c'] exactly is a
documented deferral).

THE INDUCED KURTOSIS OF THE RESOLVED PLANE (the M33 payload). The critical-plane /
von-Mises reduction resolves a scalar s(t) = p^T sigma(t) — a LINEAR projection of the
Voigt tensor (M21 eqs. (5)-(6): p = p_n normal or p_s shear). Under the joint transform
(1), s = sum_c p_c sigma_c^{rms} g_c(U_c) = sum_c a_c g_c(U_c) with a_c = p_c
sigma_c^{rms} — a LINEAR PROJECTION of a COMPONENT-WISE-transformed CORRELATED Gaussian.
Its moments are CLOSED-FORM. Writing q_c(U_c) = a_c g_c(U_c) = sum_{k=1}^3 e_{c,k}
He_k(U_c) (e_{c,1} = a_c kappa_c, e_{c,2} = a_c kappa_c h_{3,c}, e_{c,3} = a_c kappa_c
h_{4,c}; no He_0 term -> mean 0), the multivariate Hermite (diagram / Wick) moment
formula gives EXACTLY (``induced_projection_moments``):

    E[ prod_v He_{p_v}(U_{i_v}) ] = sum over perfect matchings of the half-edges
        (each vertex v contributes p_v half-edges), NO self-contractions,
        prod over matched pairs (a,b) of R_{i_a i_b}                         (3)

so the variance E[s^2] = sum_{c,c'} sum_p e_{c,p} e_{c',p} p! R_cc'^p, and the 3rd / 4th
moments follow from the 3-vertex / 4-vertex diagram sums (implemented in closed form).
The INDUCED skewness gamma_3^s = E[s^3]/E[s^2]^{3/2} and INDUCED kurtosis gamma_4^s =
E[s^4]/E[s^2]^2 are then a closed-form function of the per-component (gamma_3_c,
gamma_4_c) AND the full 6x6 correlation R — the resolved plane INHERITS the joint tensor
non-Gaussianity. For a UNIAXIAL projection (one nonzero component) gamma_4^s reduces
EXACTLY to the M24 ``hermite_kurtosis`` of that component (the scalar Hermite transform
recovered); for gamma_4_c == 3 everywhere gamma_4^s == 3 (Gaussian). Because the
per-component kurtoses combine THROUGH the correlation, gamma_4^s in general DIFFERS from
imposing any single gamma_4 directly on the scalar — the JOINT vs equivalent-scalar
distinction.

THE PER-INSTANT JOINT REDUCTION + MINER INTEGRAL. Along the M31/M32 CONTINUOUS
instantaneous spectrum, at each fine instant t_j the M27 reduction forms the 6x6
windowed moment matrices M_{n,j}, RE-SEARCHES the critical plane (the plane drifts
continuously) and gives the Gaussian per-instant damage rate (dD/dt)_G(t_j) and
bandwidth alpha_2(t_j). M33 then computes the INDUCED (gamma_3^s(t_j), gamma_4^s(t_j))
of THAT plane's resolved scalar from M_{0,j} and the per-component target
gamma_4_c(t_j) (which may itself drift with time — composing with the M32 schedule),
and scales the Gaussian rate by the M24 amplification lambda_ng(t_j) =
nongaussian_correction_factor(gamma_3^s(t_j), gamma_4^s(t_j), m, alpha_2(t_j)). The
mission damage is the Miner INTEGRAL D_nG = integral lambda_ng(t) (dD/dt)_G(t) dt over
the fine grid — the JOINT-tensor analogue of the M32 equivalent-scalar integral, of
which the M31/M27 Gaussian answer is the gamma_4_c == 3 limit and the M32
equivalent-scalar answer is the "impose gamma_4 on the scalar" special case.

THE MULTIVARIATE NON-GAUSSIAN NON-STATIONARY MONTE-CARLO. The independent time-domain
validation reuses the M21 MULTIVARIATE synthesiser (a per-bin eigen/Cholesky factor of
the 6x6 cross-PSD, the M27/M31 non-stationary per-instant blocks) to generate the six
CORRELATED Gaussian component histories, pushes EACH component through its OWN memoryless
Winterstein-Hermite transform (1) to that component's target gamma_4_c(t_j) (the VECTOR
transform — so the tensor's LOCAL joint marginals track the per-component targets),
PROJECTS onto the per-instant critical plane (a resolved non-Gaussian scalar), and
rainflow-counts (ASTM E1049) the WHOLE record. The induced SAMPLE kurtosis of the
resolved projection tracks the closed-form induced gamma_4^s; the Miner-integral tracks
the Monte-Carlo damage. In the Gaussian limit (gamma_4_c == 3) every component transform
is the identity and the record reduces EXACTLY to the M27/M31 multivariate Monte-Carlo.

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* LIBRARY-FIRST sub-flag (/IMPL/FATIG/NGAUSS composing with /JOINT + /WVILLE + a
  per-component / joint kurtosis target) — no upstream equivalent, exactly as
  established for M16-M32's PORT cards.
* TRANSLATION (memoryless component-wise Hermite) joint model: the MARGINALS (each
  component's variance AND kurtosis) are imposed EXACTLY; the 6x6 CROSS-covariance is
  preserved to LEADING order (eq. (2), the dominant R term), the h-dependent R^2/R^3
  translation distortion reported as a diagnostic. The EXACT Grigoriu correlation
  inversion (solve the underlying Gaussian correlation so the translated covariance
  equals M_0 EXACTLY) and a full non-Gaussian COPULA / non-translation joint
  distribution (beyond the component-wise Hermite translation) are DEFERRED — the
  closed-form induced kurtosis and the Monte-Carlo BOTH use the underlying-Gaussian
  correlation = the target tensor correlation, so they MATCH exactly (the model is
  self-consistent; the distortion is the small deviation of the translated covariance
  from the target).
* The von-Mises equivalent scalar is a QUADRATIC form (no linear projection), so it
  cannot inherit a linear-projection induced kurtosis; the JOINT induced-kurtosis
  correction is applied to the LINEAR critical-plane projections (normal_plane /
  shear_plane — the primary multiaxial fatigue drivers), and the von-Mises reduction
  reuses the max-shear plane's induced kurtosis as its representative (documented). The
  M32 equivalent-scalar von-Mises correction is reported side by side.
* The per-instant amplitude correction lambda_ng is the M24 narrow-band-exact Winterstein
  transform wide-band-attenuated by the instantaneous alpha_2 (Benasciutti-Tovo,
  first-order — the exact Braccesi 2009 empirical constants are DEFERRED, as in M24/M32).
* The MULTI-INPUT (M29/M30) joint non-Gaussian path and the base-acceleration multi-input
  feed are DEFERRED (the driver wires the scalar + JOINT-tensor single-input paths);
  multi-directional 100-30-30 response spectra and the arbitrary per-pair per-window
  coherence-shape card beyond M30's schedules remain DEFERRED (the M28-M32 tail).
* MEAN-STRESS beyond the basic M20/M21 Goodman intercept, CRACK-GROWTH /
  fracture-mechanics fatigue, the COMPLEX-FRF stress recovery, non-proportional hardening
  and the unchanged M10-M32 deferral tail remain DEFERRED.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.optimize import brentq

from ..common.npcompat import trapezoid


# ============================================================================
# The multivariate Hermite (diagram / Wick) moment tables (theory eq. (3))
# ============================================================================
# The induced moments of a linear projection s = sum_c q_c(U_c) of a
# component-wise-transformed correlated Gaussian are closed-form sums over the
# multivariate Hermite (diagram) moment formula: E[prod_v He_{p_v}(U_{i_v})] is a
# sum over perfect matchings of the half-edges (each vertex v carries p_v half-edges)
# WITHOUT self-contractions, each matched pair (i_a, i_b) contributing R_{i_a i_b}.
# For 2 / 3 / 4 vertices this reduces to a finite table of (Hermite orders ->
# correlation-power exponents + integer multiplicity) entries, precomputed ONCE here
# and evaluated per instant by an einsum contraction over the component index space
# (n <= 6). p, q, r, s range over the Hermite orders 1..3 present in the transform.

_FACT = [math.factorial(k) for k in range(8)]


def _build_m3_configs():
    """The 3-vertex diagram table (theory eq. (3), n = 3 vertices i, j, k with
    Hermite orders p, q, r): a matching has x edges i-j, y edges j-k, z edges i-k with
    x + z = p, x + y = q, y + z = r, so x = (p+q-r)/2, y = (q+r-p)/2, z = (p+r-q)/2
    must be NON-NEGATIVE INTEGERS; the multiplicity is p! q! r! / (x! y! z!) and the
    contribution R_ij^x R_jk^y R_ik^z. Returns a list of (p, q, r, x, y, z, coef)."""
    cfg = []
    for p in (1, 2, 3):
        for q in (1, 2, 3):
            for r in (1, 2, 3):
                sx = p + q - r
                sy = q + r - p
                sz = p + r - q
                if sx < 0 or sy < 0 or sz < 0 or sx % 2 or sy % 2 or sz % 2:
                    continue
                x, y, z = sx // 2, sy // 2, sz // 2
                coef = _FACT[p] * _FACT[q] * _FACT[r] / (
                    _FACT[x] * _FACT[y] * _FACT[z])
                cfg.append((p, q, r, x, y, z, coef))
    return cfg


def _build_m4_configs():
    """The 4-vertex diagram table (theory eq. (3), vertices i, j, k, l with Hermite
    orders p, q, r, s): enumerate the six edge counts (n_ij, n_ik, n_il, n_jk, n_jl,
    n_kl) >= 0 with NO self-loops such that each vertex's half-edge count matches its
    order (n_ij+n_ik+n_il = p, n_ij+n_jk+n_jl = q, n_ik+n_jk+n_kl = r, n_il+n_jl+n_kl =
    s); the multiplicity is p! q! r! s! / (prod n_e!) and the contribution
    prod R_e^{n_e}. Returns (p, q, r, s, n_ij, n_ik, n_il, n_jk, n_jl, n_kl, coef)."""
    cfg = []
    for p in (1, 2, 3):
        for q in (1, 2, 3):
            for r in (1, 2, 3):
                for s in (1, 2, 3):
                    if (p + q + r + s) % 2:
                        continue                       # odd total degree -> no matching
                    for nij in range(min(p, q) + 1):
                        for nik in range(min(p, r) + 1):
                            nil = p - nij - nik
                            if nil < 0:
                                continue
                            for njk in range(min(q, r) + 1):
                                njl = q - nij - njk
                                if njl < 0:
                                    continue
                                nkl = r - nik - njk
                                if nkl < 0 or nkl != s - nil - njl:
                                    continue
                                denom = (_FACT[nij] * _FACT[nik] * _FACT[nil]
                                         * _FACT[njk] * _FACT[njl] * _FACT[nkl])
                                coef = (_FACT[p] * _FACT[q] * _FACT[r] * _FACT[s]
                                        / denom)
                                cfg.append((p, q, r, s, nij, nik, nil, njk, njl,
                                            nkl, coef))
    return cfg


_M3_CFG = _build_m3_configs()
_M4_CFG = _build_m4_configs()

_M3_SUBSCRIPTS = "i,j,k,ij,jk,ik->"
_M4_SUBSCRIPTS = "i,j,k,l,ij,ik,il,jk,jl,kl->"
# per-support-size cache of the precomputed einsum contraction paths (the path
# depends only on the operand SHAPES, which are fixed for a given support size n, so
# it is found ONCE per n and reused — the diagram sums run per instant per plane, so
# re-finding the path each call dominated the cost). Keyed by n.
_PATH_CACHE = {}


def _paths_for(n):
    """The precomputed einsum contraction paths for the 3- and 4-vertex diagram sums at
    support size ``n`` (cached — found once per n from dummy operands of the right
    shape). Returns (m3_paths, m4_paths) aligned with ``_M3_CFG`` / ``_M4_CFG``."""
    cached = _PATH_CACHE.get(n)
    if cached is not None:
        return cached
    v = np.ones(n)
    Mn = np.ones((n, n))
    m3_paths = [np.einsum_path(_M3_SUBSCRIPTS, v, v, v, Mn, Mn, Mn,
                               optimize="greedy")[0] for _ in _M3_CFG]
    m4_paths = [np.einsum_path(_M4_SUBSCRIPTS, v, v, v, v, Mn, Mn, Mn, Mn, Mn, Mn,
                               optimize="greedy")[0] for _ in _M4_CFG]
    _PATH_CACHE[n] = (m3_paths, m4_paths)
    return m3_paths, m4_paths


def _projection_central_moments(e, R):
    """The central moments (E[s^2], E[s^3], E[s^4]) of the mean-zero projection
    s = sum_c q_c(U_c), q_c = sum_{k=1..3} e[c,k] He_k(U_c), U ~ N(0, R) — the
    multivariate Hermite (diagram) sums (theory eq. (3)). ``e`` is (n, 4) (column 0
    unused; columns 1..3 the He_1/He_2/He_3 coefficients per component), ``R`` the
    (n, n) UNDERLYING-Gaussian correlation. Evaluated by einsum over the component
    index space (n small), with the contraction paths cached per n. Exact to machine
    precision."""
    e = np.asarray(e, dtype=float)
    R = np.asarray(R, dtype=float)
    n = e.shape[0]
    # elementwise correlation powers R^0..R^3 (R^0 = ones for the "no edge" case)
    Rp = [np.ones((n, n)), R, R * R, R * R * R]
    m3_paths, m4_paths = _paths_for(n)
    # --- 2nd moment: only equal Hermite orders survive across two vertices ---------
    M2 = 0.0
    for p in (1, 2, 3):
        ep = e[:, p]
        M2 += _FACT[p] * float(ep @ Rp[p] @ ep)
    # --- 3rd moment: the 3-vertex diagram sum -------------------------------------
    M3 = 0.0
    for cfg, path in zip(_M3_CFG, m3_paths):
        p, q, r, x, y, z, coef = cfg
        M3 += coef * np.einsum(_M3_SUBSCRIPTS, e[:, p], e[:, q], e[:, r],
                               Rp[x], Rp[y], Rp[z], optimize=path)
    # --- 4th moment: the 4-vertex diagram sum -------------------------------------
    M4 = 0.0
    for cfg, path in zip(_M4_CFG, m4_paths):
        p, q, r, s, nij, nik, nil, njk, njl, nkl, coef = cfg
        M4 += coef * np.einsum(_M4_SUBSCRIPTS, e[:, p], e[:, q], e[:, r], e[:, s],
                               Rp[nij], Rp[nik], Rp[nil], Rp[njk], Rp[njl], Rp[nkl],
                               optimize=path)
    return float(M2), float(M3), float(M4)


# ============================================================================
# The per-component Hermite coefficients + the induced projection moments
# ============================================================================

def component_hermite_coefficients(gamma3, gamma4, model="winterstein"):
    """The per-component Winterstein-Hermite coefficients (h3_c, h4_c, kappa_c) for the
    6 (or n) tensor components (theory eq. (1)), a VECTOR wrapper over the M24
    ``nongaussian_fatigue.hermite_coefficients``. ``gamma3`` / ``gamma4`` are per-
    component arrays (or scalars, broadcast). Returns (h3, h4, kappa) each (n,). A
    Gaussian component (gamma4 = 3, gamma3 = 0) gets (0, 0, 1) — the identity."""
    from . import nongaussian_fatigue as ngf
    g3 = np.atleast_1d(np.asarray(gamma3, dtype=float))
    g4 = np.atleast_1d(np.asarray(gamma4, dtype=float))
    n = max(g3.size, g4.size)
    g3 = np.broadcast_to(g3, (n,))
    g4 = np.broadcast_to(g4, (n,))
    h3 = np.empty(n)
    h4 = np.empty(n)
    kappa = np.empty(n)
    for c in range(n):
        h3[c], h4[c], kappa[c] = ngf.hermite_coefficients(g3[c], g4[c], model=model)
    return h3, h4, kappa


def _hermite_e_coeffs(a, h3, h4, kappa):
    """Assemble the Hermite-basis coefficient array ``e`` (n, 4) of the weighted
    per-component transforms q_c = a_c g_c(U_c) = sum_k e[c,k] He_k(U_c): e[c,1] =
    a_c kappa_c, e[c,2] = a_c kappa_c h3_c, e[c,3] = a_c kappa_c h4_c (column 0 the
    unused He_0). ``a`` = the projection weights a_c = p_c sigma_c^{rms}."""
    a = np.asarray(a, dtype=float)
    n = a.size
    e = np.zeros((n, 4))
    ak = a * kappa
    e[:, 1] = ak
    e[:, 2] = ak * h3
    e[:, 3] = ak * h4
    return e


def induced_projection_moments(M0, proj, gamma3, gamma4, model="winterstein",
                               return_components=False, underlying_R=None,
                               copula="gaussian", copula_params=None):
    """The CLOSED-FORM induced (variance, skewness, kurtosis) of the resolved-plane
    scalar s = p^T sigma under the joint component-wise Hermite transform (theory eq.
    (3)) — the resolved plane INHERITING the joint tensor non-Gaussianity.

    ``M0`` (n, n) — the Gaussian tensor COVARIANCE matrix (M_0 = E[sigma sigma^T], the
    0th spectral-moment matrix; n = 6 Voigt components). ``proj`` (n,) — the linear
    projection 6-vector p (normal_projection / shear_projection). ``gamma3`` / ``gamma4``
    — the per-component target skewness / kurtosis (scalars broadcast, or (n,) arrays).

    Standardises to U ~ N(0, R) with R the correlation of ``M0`` and the component RMS
    sigma_c^{rms} = sqrt(M0[c,c]); forms the weighted Hermite coefficients from a_c =
    p_c sigma_c^{rms}; and evaluates the multivariate Hermite (diagram) moments. Returns
    (var, skew, kurt) — skew = E[s^3]/var^{3/2}, kurt = E[s^4]/var^2 (the INDUCED
    gamma_3^s / gamma_4^s). For a UNIAXIAL projection (one nonzero component) the induced
    kurtosis reduces EXACTLY to the M24 ``hermite_kurtosis`` of that component; for a
    Gaussian tensor (gamma4 = 3) the induced kurtosis is 3 EXACTLY. ``return_components``
    additionally returns the (h3, h4, kappa) per-component coefficients.

    ``underlying_R`` (M34, optional) — the (n, n) UNDERLYING-Gaussian correlation to use
    for U instead of the correlation of ``M0``. In the M33 (leading-order) model the
    underlying correlation is TAKEN as the target correlation R = M0/(sig sig) (default,
    ``underlying_R=None``); in the M34 EXACT-covariance model it is the Grigoriu / NORTA
    correlation-matching solution rho^U (``solve_underlying_correlation``) so the
    transformed component-wise Hermite tensor reproduces the target covariance M0 EXACTLY
    rather than to leading order. The variance is ALWAYS computed on the passed underlying
    correlation, so with rho^U the induced (var, skew, kurt) are the covariance-EXACT
    ones; passing ``underlying_R = R`` reproduces the M33 answer byte-identically."""
    M0 = np.asarray(M0, dtype=float)
    p = np.asarray(proj, dtype=float).ravel()
    n = p.size
    var_c = np.clip(np.diag(M0)[:n], 0.0, None)
    sig = np.sqrt(var_c)
    a = p * sig                                        # effective projection weights
    # RESTRICT to the nonzero support (components that actually enter s): both a
    # nonzero AND a positive variance (a zero-variance component contributes nothing
    # and its correlation row is undefined)
    supp = np.where((np.abs(a) > 0.0) & (var_c > 0.0))[0]
    h3f, h4f, kappaf = component_hermite_coefficients(gamma3, gamma4, model=model)
    # broadcast per-component coeffs to n if they came in as scalars/short arrays
    if h3f.size != n:
        h3f = np.broadcast_to(h3f, (n,)).copy()
        h4f = np.broadcast_to(h4f, (n,)).copy()
        kappaf = np.broadcast_to(kappaf, (n,)).copy()
    if supp.size == 0:
        out = (0.0, 0.0, 3.0)
        return (out + ((h3f, h4f, kappaf),)) if return_components else out
    sig_s = sig[supp]
    if underlying_R is not None:
        # M34 EXACT-covariance: use the supplied underlying-Gaussian correlation rho^U
        # (the NORTA correlation-matching solution) on the support instead of the target
        # correlation — the diagram moments then evaluate the covariance-EXACT statistics
        R = np.asarray(underlying_R, dtype=float)[np.ix_(supp, supp)].copy()
    else:
        # M33 leading-order: the underlying correlation IS the target correlation
        # (R_cc' = M0/(sig_c sig_c'))
        Msub = M0[np.ix_(supp, supp)]
        R = Msub / np.outer(sig_s, sig_s)
    R = np.clip(R, -1.0, 1.0)
    np.fill_diagonal(R, 1.0)
    e = _hermite_e_coeffs(a[supp], h3f[supp], h4f[supp], kappaf[supp])
    M2, M3, M4 = _projection_central_moments(e, R)
    if M2 <= 0.0:
        out = (0.0, 0.0, 3.0)
        return (out + ((h3f, h4f, kappaf),)) if return_components else out
    skew = M3 / (M2 ** 1.5)
    kurt = M4 / (M2 ** 2)
    out = (M2, skew, kurt)
    return (out + ((h3f, h4f, kappaf),)) if return_components else out


def translation_process_covariance(M0, gamma3, gamma4, model="winterstein"):
    """The transformed 6x6 CROSS-covariance Cov(X_c, X_c') of the joint component-wise
    Hermite (translation) process (theory eq. (2)) and its PRESERVATION diagnostic.

    Given the Gaussian covariance ``M0`` (n, n) and per-component (gamma3, gamma4),
    returns a dict with:
      * ``cov`` (n, n) — the transformed covariance (eq. (2)); its DIAGONAL equals
        diag(M0) EXACTLY (marginal variance preserved), its off-diagonal equals M0 to
        leading order;
      * ``preservation_error`` — the max relative off-diagonal deviation
        |cov - M0| / sqrt(diag_c diag_c') (0 for a Gaussian tensor; the documented
        second-order translation distortion, eq. (2));
      * ``h3`` / ``h4`` / ``kappa`` — the per-component coefficients.
    The MARGINALS (variance + kurtosis) are exact; the cross-structure is leading-order
    (Grigoriu translation correlation distortion — the exact inversion is deferred)."""
    M0 = np.asarray(M0, dtype=float)
    n = M0.shape[0]
    var_c = np.clip(np.diag(M0), 0.0, None)
    sig = np.sqrt(var_c)
    h3, h4, kappa = component_hermite_coefficients(gamma3, gamma4, model=model)
    if h3.size != n:
        h3 = np.broadcast_to(h3, (n,)).copy()
        h4 = np.broadcast_to(h4, (n,)).copy()
        kappa = np.broadcast_to(kappa, (n,)).copy()
    R = np.zeros((n, n))
    nz = sig > 0.0
    R[np.ix_(nz, nz)] = M0[np.ix_(nz, nz)] / np.outer(sig[nz], sig[nz])
    R = np.clip(R, -1.0, 1.0)
    # eq. (2): Cov(X_c,X_c') = sig_c sig_c' kappa_c kappa_c'
    #          [R + 2 h3_c h3_c' R^2 + 6 h4_c h4_c' R^3]
    KK = np.outer(kappa, kappa)
    H3 = np.outer(h3, h3)
    H4 = np.outer(h4, h4)
    inner = R + 2.0 * H3 * R ** 2 + 6.0 * H4 * R ** 3
    cov = np.outer(sig, sig) * KK * inner
    np.fill_diagonal(cov, var_c)                       # diagonal EXACT by construction
    # preservation error (max relative off-diagonal deviation)
    denom = np.outer(sig, sig)
    err = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom > 0.0, np.abs(cov - M0) / denom, 0.0)
    np.fill_diagonal(rel, 0.0)
    err = float(np.max(rel)) if rel.size else 0.0
    return {"cov": cov, "preservation_error": err, "h3": h3, "h4": h4,
            "kappa": kappa}


# ============================================================================
# The EXACT translation-process correlation-distortion INVERSION (M34)
# ============================================================================
# THE GRIGORIU / NATAF / NORTA CORRELATION MATCHING. The M33 (leading-order)
# translation model imposes the per-component variances AND kurtoses EXACTLY but
# preserves the 6x6 CROSS-covariance only to LEADING order: it takes the underlying
# Gaussian correlation rho^U_cc' = the TARGET correlation R_cc' and reports the
# residual translation distortion (``translation_process_covariance``'s
# ``preservation_error``) as a diagnostic. M34 INVERTS that distortion. For each
# component pair (c, c') the transformed correlation of the two component-wise
# Winterstein-Hermite transforms g_c, g_c' at underlying-Gaussian correlation rho is
# the Mehler / diagram sum (theory eq. (2), the c != c' off-diagonal; He_p orthogonality
# E[He_p(U_c) He_q(U_c')] = delta_pq p! rho^p):
#
#     phi_cc'(rho) = kappa_c kappa_c'
#         [ rho + 2 h_{3,c} h_{3,c'} rho^2 + 6 h_{4,c} h_{4,c'} rho^3 ]        (2')
#
# — a CUBIC in rho. The exact correlation matching solves phi_cc'(rho^U) = R_cc' for
# the underlying rho^U_cc' PER PAIR (a monotone root-find; the cubic has a
# closed/near-closed real inverse on [-1, 1]), assembles the 6x6 rho^U, and REPAIRS it
# to the nearest valid (positive-definite, unit-diagonal) correlation matrix by the
# Higham 2002 alternating-projections algorithm so the underlying Gaussian is a
# well-defined covariance. Pushing the component-wise Hermite transform through a
# Gaussian of correlation rho^U then reproduces the target covariance M0 EXACTLY (the
# translated ``preservation_error`` -> ~0), where M33's leading-order rho^U = R left the
# documented second-order distortion. This is exactly the Grigoriu translation-process
# correlation distortion (Grigoriu 1995/1998); the Nataf transformation (Nataf 1962; Der
# Kiureghian & Liu 1986 — the underlying-Gaussian correlation of a marginal-transformed
# vector); Cario & Nelson's NORTA (NORmal-To-Anything, 1997 — the same correlation-
# matching root-find for arbitrary marginals); Vale & Maurelli 1983 (the intermediate-
# correlation solve for non-normal multivariate data); Higham 2002 (the nearest
# correlation matrix). In the SMALL-non-Gaussianity limit (h3, h4 -> 0, kappa -> 1) the
# cubic collapses to phi(rho) = rho, so rho^U -> R and M34 recovers the M33 underlying
# correlation EXACTLY; for a Gaussian component (gamma4 = 3) phi_cc'(rho) = rho on that
# pair, so rho^U = R and the whole M34 path DELEGATES to the M33 / M31 / M27 Gaussian
# answer byte-identically.


def _solve_pair_rho(A1, A2, A3, target):
    """Solve the per-pair correlation-matching cubic phi(rho) = A1 rho + A2 rho^2 +
    A3 rho^3 = ``target`` for the underlying-Gaussian correlation rho on [-1, 1] (theory
    eq. (2'), with A1 = kappa_c kappa_c', A2 = 2 kappa_c kappa_c' h3_c h3_c', A3 = 6
    kappa_c kappa_c' h4_c h4_c'). Picks the REAL root in [-1, 1] closest to ``target``
    (the physical near-identity branch — phi is monotone in the valid leptokurtic
    regime, so the branch is unique). If NO root lies in [-1, 1] (the target correlation
    is outside the achievable range [phi(-1), phi(1)] — the NORTA feasibility limit),
    CLAMPS to the endpoint whose phi is closest to the target and flags it. Returns
    (rho, residual, feasible) with residual = phi(rho) - target."""
    def phi(r):
        return A1 * r + A2 * r * r + A3 * r * r * r
    # exact cubic roots (numpy drops leading zeros, so a degenerate quadratic/linear is
    # handled gracefully — e.g. two Gaussian components give A2 = A3 = 0, root target/A1)
    coeffs = [A3, A2, A1, -float(target)]
    while len(coeffs) > 2 and abs(coeffs[0]) < 1e-300:
        coeffs = coeffs[1:]
    if len(coeffs) == 2 and abs(coeffs[0]) < 1e-300:
        # fully degenerate (A1 = A2 = A3 = 0) — a zero-variance / no-correlation pair
        return 0.0, -float(target), True
    roots = np.roots(coeffs)
    real = [r.real for r in np.atleast_1d(roots)
            if abs(r.imag) < 1e-9 and -1.0001 <= r.real <= 1.0001]
    if real:
        r = min(real, key=lambda x: abs(x - target))
        r = max(-1.0, min(1.0, r))
        return float(r), float(phi(r) - target), True
    # infeasible: clamp to the endpoint whose transformed correlation is closest
    r = min((-1.0, 1.0), key=lambda x: abs(phi(x) - target))
    return float(r), float(phi(r) - target), False


def _proj_spd(A, floor=0.0):
    """Project a symmetric matrix onto the positive-semidefinite cone (clamp its
    eigenvalues to >= ``floor``) — the SPD projection step of the Higham 2002 nearest-
    correlation-matrix alternating projections."""
    w, V = np.linalg.eigh((A + A.T) / 2.0)
    w = np.clip(w, floor, None)
    return (V * w) @ V.T


def _higham_nearest_correlation(A, max_iter=200, tol=1e-12, eig_floor=1e-10):
    """The Higham (2002) NEAREST CORRELATION MATRIX of a symmetric ``A`` by Dykstra-
    corrected alternating projections between the positive-semidefinite cone and the
    unit-diagonal set. Returns the closest (in Frobenius norm) valid correlation matrix
    (symmetric, unit diagonal, positive-semidefinite). ``eig_floor`` clamps the smallest
    eigenvalue slightly above 0 so the result is strictly positive-definite (a valid
    Cholesky factor for the Monte-Carlo synthesiser). A matrix that is ALREADY a valid
    correlation matrix passes through essentially unchanged (a few cheap iterations)."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    Y = (A + A.T) / 2.0
    dS = np.zeros_like(Y)
    X = Y.copy()
    for _ in range(max_iter):
        R = Y - dS                        # Dykstra correction
        X = _proj_spd(R, floor=0.0)       # project onto the PSD cone
        dS = X - R
        Yprev = Y
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)          # project onto unit-diagonal set
        if np.linalg.norm(Y - Yprev, "fro") <= tol * max(1.0,
                                                          np.linalg.norm(Y, "fro")):
            break
    # final strict-PD floor (guarantee a Cholesky factor exists)
    w, V = np.linalg.eigh((Y + Y.T) / 2.0)
    if w.min() < eig_floor:
        w = np.clip(w, eig_floor, None)
        Y = (V * w) @ V.T
        d = np.sqrt(np.clip(np.diag(Y), 1e-300, None))
        Y = Y / np.outer(d, d)            # renormalise to unit diagonal
    np.fill_diagonal(Y, 1.0)
    return Y



def _solve_pair_rho_t_copula(target, h3_1, h4_1, kappa_1, h3_2, h4_2, kappa_2, nu, n_samples=100000, seed=42):
    rng = np.random.default_rng(seed)
    Z1_norm = rng.standard_normal(n_samples)
    Z2_indep = rng.standard_normal(n_samples)
    W = rng.chisquare(nu, size=n_samples)
    sqrt_nu_W = np.sqrt(nu / W)
    
    def obj(rho_U):
        Z2_norm = rho_U * Z1_norm + np.sqrt(1.0 - rho_U**2) * Z2_indep
        X1 = Z1_norm * sqrt_nu_W
        X2 = Z2_norm * sqrt_nu_W
        Z1 = stats.norm.ppf(stats.t.cdf(X1, df=nu))
        Z2 = stats.norm.ppf(stats.t.cdf(X2, df=nu))
        q1 = kappa_1 * (Z1 + h3_1*(Z1**2 - 1.0) + h4_1*(Z1**3 - 3.0*Z1))
        q2 = kappa_2 * (Z2 + h3_2*(Z2**2 - 1.0) + h4_2*(Z2**3 - 3.0*Z2))
        return np.corrcoef(q1, q2)[0, 1] - target
        
    try:
        return brentq(obj, -0.999, 0.999)
    except ValueError:
        return np.sign(target) * 0.999

def _induced_projection_moments_t_copula(R, proj_a, h3f, h4f, kappaf, nu, n_samples=100000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(proj_a)
    L = np.linalg.cholesky(R)
    Z_norm = rng.standard_normal((n_samples, n))
    Y = Z_norm @ L.T
    
    W = rng.chisquare(nu, size=(n_samples, 1))
    X = Y * np.sqrt(nu / W)
    
    U = stats.t.cdf(X, df=nu)
    Z = stats.norm.ppf(U)
    
    Z2 = Z**2
    Z3 = Z**3
    q = proj_a * kappaf * (Z + h3f * (Z2 - 1.0) + h4f * (Z3 - 3.0 * Z))
    
    s = np.sum(q, axis=1)
    
    var = np.var(s)
    std = np.sqrt(var)
    if var > 0:
        skew = np.mean((s - np.mean(s))**3) / (std**3)
        kurt = np.mean((s - np.mean(s))**4) / (std**4)
    else:
        skew = 0.0
        kurt = 3.0
    return float(var), float(skew), float(kurt)

def solve_underlying_correlation(M0, gamma3, gamma4, model="winterstein",
                                 repair=True, var_floor=1e-9,
                                 copula="gaussian", copula_params=None):
    """Solve the UNDERLYING-Gaussian correlation rho^U (M34) so the component-wise
    Winterstein-Hermite (translation) transform of a Gaussian tensor of correlation
    rho^U reproduces the TARGET 6x6 covariance ``M0`` EXACTLY — the Grigoriu / Nataf /
    NORTA correlation-distortion INVERSION (theory eq. (2')).

    For each component pair (c, c') solves phi_cc'(rho^U) = R_cc' (the target correlation
    R = M0 / (sig sig)) for rho^U by ``_solve_pair_rho`` (the monotone cubic root-find),
    assembles the 6x6 rho^U (unit diagonal), and — if ``repair`` — projects it to the
    nearest valid correlation matrix by ``_higham_nearest_correlation`` (so the
    underlying Gaussian is a genuine positive-definite covariance).

    ``var_floor`` — the RELATIVE variance floor that defines the MATERIALLY-CONTRIBUTING
    support: a component whose variance is below ``var_floor`` times the largest component
    variance carries numerically no stress, so its "correlation" with the others is
    floating-point noise (a near-degenerate tensor). Such components are EXCLUDED from the
    correlation matching (their rho^U rows are the identity) AND from the preservation-
    error diagnostics — the exact inversion reproduces the covariance of the significant
    components to machine precision, which is all that enters the resolved-plane damage
    (a component with negligible RMS is weighted out of every projection). Without this
    floor a thin-section stress tensor's numerically-zero minor components would inject
    spurious near-+/-1 correlations, some infeasible under NORTA, that no inversion can
    match.

    Returns a dict with
      * ``rho_u`` (n, n) — the (repaired) underlying-Gaussian correlation;
      * ``rho_u_raw`` (n, n) — before the Higham repair;
      * ``target_R`` (n, n) — the target correlation of ``M0``;
      * ``transformed_cov`` (n, n) — eq. (2) evaluated at ``rho_u`` (equals ``M0`` to
        machine precision on the diagonal AND, for feasible targets with no repair,
        off-diagonal);
      * ``support`` (n,) bool — the materially-contributing components;
      * ``preservation_error`` — the EXACT model's max relative off-diagonal deviation
        |transformed_cov - M0| / (sig_c sig_c') over the SUPPORT (~0 for a feasible target
        — the M34 payload, closing M33's ``preservation_error``);
      * ``preservation_error_leading`` — the M33 leading-order deviation (rho^U = R) over
        the SAME support, for the fair side-by-side comparison;
      * ``n_infeasible`` — count of support pairs clamped at the NORTA feasibility bound;
      * ``repair_delta`` — Frobenius norm of the Higham correction (0 if already valid);
      * ``h3`` / ``h4`` / ``kappa`` — the per-component coefficients.

    In the Gaussian limit (gamma4 = 3 on every component) rho^U = R EXACTLY and the
    preservation error is 0 (the delegation guarantee)."""
    M0 = np.asarray(M0, dtype=float)
    n = M0.shape[0]
    var_c = np.clip(np.diag(M0), 0.0, None)
    sig = np.sqrt(var_c)
    vmax = float(np.max(var_c)) if var_c.size else 0.0
    # the materially-contributing support (relative variance floor); everything below is
    # numerically-zero stress whose correlation is floating-point noise
    nz = var_c > max(vmax * float(var_floor), 0.0)
    h3, h4, kappa = component_hermite_coefficients(gamma3, gamma4, model=model)
    if h3.size != n:
        h3 = np.broadcast_to(h3, (n,)).copy()
        h4 = np.broadcast_to(h4, (n,)).copy()
        kappa = np.broadcast_to(kappa, (n,)).copy()
    # target correlation (on the support; degenerate rows left at identity)
    R = np.eye(n)
    R[np.ix_(nz, nz)] = M0[np.ix_(nz, nz)] / np.outer(sig[nz], sig[nz])
    R = np.clip(R, -1.0, 1.0)
    np.fill_diagonal(R, 1.0)
    # per-pair inversion (upper triangle, symmetric) over the support only
    rho = np.eye(n)
    n_infeasible = 0
    for c in range(n):
        for cp in range(c + 1, n):
            if not (nz[c] and nz[cp]):
                rho[c, cp] = rho[cp, c] = 0.0
                continue
            if copula == "t":
                nu = copula_params if copula_params is not None else 4.0
                r = _solve_pair_rho_t_copula(float(R[c, cp]), h3[c], h4[c], kappa[c], h3[cp], h4[cp], kappa[cp], nu)
                # t-copula MC solve doesn't return feasible status, assume feasible unless r is exactly bounded
                feasible = abs(r) < 0.99
            else:
                A1 = kappa[c] * kappa[cp]
                A2 = A1 * 2.0 * h3[c] * h3[cp]
                A3 = A1 * 6.0 * h4[c] * h4[cp]
                r, _res, feasible = _solve_pair_rho(A1, A2, A3, float(R[c, cp]))
            rho[c, cp] = rho[cp, c] = r
            if not feasible:
                n_infeasible += 1
    rho_raw = rho.copy()
    if repair:
        # only the significant sub-block is a meaningful correlation matrix; repair it
        # and leave the negligible-variance rows/cols at their identity
        idx = np.where(nz)[0]
        if idx.size > 1:
            sub = _higham_nearest_correlation(rho[np.ix_(idx, idx)])
            rho[np.ix_(idx, idx)] = sub
    repair_delta = float(np.linalg.norm(rho - rho_raw, "fro"))
    # the transformed covariance at rho_u (eq. (2)) and its preservation diagnostics
    KK = np.outer(kappa, kappa)
    H3 = np.outer(h3, h3)
    H4 = np.outer(h4, h4)
    inner = rho + 2.0 * H3 * rho ** 2 + 6.0 * H4 * rho ** 3
    cov = np.outer(sig, sig) * KK * inner
    np.fill_diagonal(cov, var_c)
    # the M33 leading-order transformed covariance (rho^U = R) on the SAME footing
    inner_l = R + 2.0 * H3 * R ** 2 + 6.0 * H4 * R ** 3
    cov_l = np.outer(sig, sig) * KK * inner_l
    np.fill_diagonal(cov_l, var_c)
    # preservation errors over the significant support only
    def _pres(cov_):
        idx = np.where(nz)[0]
        if idx.size < 2:
            return 0.0
        st = sig[idx]
        rel = np.abs(cov_[np.ix_(idx, idx)] - M0[np.ix_(idx, idx)]) / np.outer(st, st)
        np.fill_diagonal(rel, 0.0)
        return float(np.max(rel))
    pres_exact = _pres(cov)
    pres_lead = _pres(cov_l)
    return {"rho_u": rho, "rho_u_raw": rho_raw, "target_R": R, "support": nz,
            "transformed_cov": cov, "preservation_error": pres_exact,
            "preservation_error_leading": pres_lead,
            "n_infeasible": int(n_infeasible), "repair_delta": repair_delta,
            "h3": h3, "h4": h4, "kappa": kappa}


def joint_lambda_ng(M0, proj, gamma3, gamma4, m, alpha2=1.0,
                    bandwidth_correction=True, model="winterstein",
                    copula="gaussian", copula_params=None):
    """The JOINT non-Gaussian amplification lambda_ng of a resolved plane: compute the
    INDUCED (gamma_3^s, gamma_4^s) of the projection ``proj`` under the joint transform
    (``induced_projection_moments``) and feed those to the M24 closed-form
    ``nongaussian_correction_factor`` at the plane's bandwidth ``alpha2`` (theory — the
    resolved plane inherits the joint tensor kurtosis, then the SAME M24 amplitude
    correction applies). Returns (lambda_ng, gamma3_induced, gamma4_induced). In the
    Gaussian tensor limit the induced kurtosis is 3 and lambda_ng == 1 EXACTLY."""
    from . import nongaussian_fatigue as ngf
    _, g3s, g4s = induced_projection_moments(M0, proj, gamma3, gamma4, model=model,
                                             copula=copula, copula_params=copula_params)
    lam = ngf.nongaussian_correction_factor(
        g3s, g4s, m, alpha2=alpha2, bandwidth_correction=bandwidth_correction,
        model=model)
    return float(lam), float(g3s), float(g4s)


# ============================================================================
# The per-component kurtosis-vs-time / per-component schedule (M33 primitive)
# ============================================================================

def _pair(v):
    """Coerce a scalar or 2-tuple ``v`` into (start, end) — a scalar is held constant
    (start == end); a (start, end) pair sweeps LINEARLY across the mission (the M32
    kurtosis-sweep convention, reused per component)."""
    if isinstance(v, (tuple, list, np.ndarray)):
        a = np.asarray(v, dtype=float).ravel()
        if a.size >= 2:
            return float(a[0]), float(a[1])
        return float(a[0]), float(a[0])
    return float(v), float(v)


def component_kurtosis_schedule(s, kurt, skew=0.0, ncomp=6, kurt_grid=None,
                                skew_grid=None):
    """The per-component, time-varying target kurtosis gamma_4_c(t) / skewness
    gamma_3_c(t) sampled onto the fine instant grid (the M33 primitive — the
    per-COMPONENT generalisation of the M32 ``kurtosis_schedule``). ``s`` (nt,) is the
    M31/M32 fine-grid mission-fraction axis, ``ncomp`` the number of tensor components
    (6 Voigt).

    Each of ``kurt`` / ``skew`` may be
      * a SCALAR — the SAME target on EVERY component, held constant (the M32 scalar
        target lifted to the tensor — the scalar-equivalent-shape limit);
      * an (ncomp,) ARRAY — a PER-COMPONENT constant target (the JOINT case: different
        components carry different kurtoses);
      * a (start, end) PAIR — swept LINEARLY across the mission, same on every component
        (the M32 time-varying target on the tensor);
      * given DIRECTLY per instant via ``kurt_grid`` / ``skew_grid`` of shape (nt,)
        (same on every component) or (nt, ncomp) (per instant AND per component — the
        full joint time-varying schedule).

    Returns (gamma4 (nt, ncomp), gamma3 (nt, ncomp)). gamma4 == 3, gamma3 == 0
    everywhere is the Gaussian tensor (a no-op)."""
    s = np.asarray(s, dtype=float).ravel()
    nt = s.size

    def _grid(val, grid, default):
        if grid is not None:
            g = np.asarray(grid, dtype=float)
            if g.ndim == 1:
                if g.size != nt:
                    raise ValueError("kurt_grid/skew_grid (1-D) must have one value "
                                     "per fine instant.")
                return np.repeat(g[:, None], ncomp, axis=1)
            if g.shape != (nt, ncomp):
                raise ValueError("kurt_grid/skew_grid (2-D) must be (nt, ncomp).")
            return g.copy()
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 0:
            return np.full((nt, ncomp), float(arr))
        arr = arr.ravel()
        if arr.size == ncomp:
            # per-component constant target (broadcast over time)
            return np.broadcast_to(arr, (nt, ncomp)).copy()
        if arr.size == 2:
            # linear sweep, same on every component
            a0, a1 = float(arr[0]), float(arr[1])
            ramp = a0 + s * (a1 - a0)
            return np.repeat(ramp[:, None], ncomp, axis=1)
        if arr.size == 1:
            return np.full((nt, ncomp), float(arr[0]))
        raise ValueError(f"kurt/skew array of size {arr.size} is neither a scalar, a "
                         f"(start,end) pair nor an ({ncomp},) per-component vector.")

    gamma4 = _grid(kurt, kurt_grid, 3.0)
    gamma3 = _grid(skew, skew_grid, 0.0)
    return gamma4, gamma3


def _is_gaussian_component_schedule(gamma4, gamma3):
    """True when the whole per-component schedule is Gaussian (gamma_4_c == 3,
    gamma_3_c == 0 at EVERY instant AND component) — the exact limit in which the M33
    path DELEGATES to the M31/M27 Gaussian tensor answer BYTE-IDENTICALLY."""
    g4 = np.asarray(gamma4, dtype=float)
    g3 = np.asarray(gamma3, dtype=float)
    return bool(np.all(g4 == 3.0) and np.all(g3 == 0.0))


def _representative_scalar_kurtosis(kurt, skew=0.0):
    """A representative SCALAR (kurt, skew) target for the M32 equivalent-scalar
    side-by-side reference from a per-component target: the per-component MAX kurtosis
    (the worst-case component, a conservative scalar surrogate) and the max-magnitude
    skewness. A scalar / (start,end) pair passes through unchanged (so a scalar target
    drives the M32 reference identically). Used ONLY to build the ``scalar_equivalent``
    side-by-side entry; the JOINT path uses the per-component targets throughout."""
    if isinstance(kurt, (tuple, list, np.ndarray)):
        arr = np.asarray(kurt, dtype=float).ravel()
        if arr.size == 2:
            k = (float(arr[0]), float(arr[1]))         # a sweep -> keep the sweep
        else:
            k = float(np.max(arr))                     # per-component -> the max
    else:
        k = float(kurt)
    if isinstance(skew, (tuple, list, np.ndarray)):
        arr = np.asarray(skew, dtype=float).ravel()
        if arr.size == 2:
            sk = (float(arr[0]), float(arr[1]))
        else:
            sk = float(arr[np.argmax(np.abs(arr))]) if arr.size else 0.0
    else:
        sk = float(skew)
    return k, sk


# ============================================================================
# (a) The per-instant JOINT non-Gaussian tensor reduction (build item 1)
# ============================================================================

def _reduce_instant_tensors_joint_ng(omega, Scross, weights, inst_scales, durations,
                                     gamma4, gamma3, m, C, mean_stress, ultimate, naz,
                                     npol, bandwidth_correction, model, drift=True,
                                     exact=False):
    """Reduce a fine-grid instantaneous TENSOR spectrum with the JOINT non-Gaussian
    correction (theory "THE PER-INSTANT JOINT REDUCTION"): per instant form the 6x6
    windowed moment matrices M_{n,j}, RE-SEARCH the critical plane / F_np from THAT
    instant's tensor (``reduce_window_tensor`` — the plane drifts CONTINUOUSLY), compute
    the INDUCED (gamma_3^s, gamma_4^s) of each LINEAR plane's resolved scalar from
    M_{0,j} and the per-component target gamma_4_c(t_j) (``induced_projection_moments``),
    and scale the per-instant Gaussian damage rate by the M24 lambda_ng of that INDUCED
    kurtosis before Miner-INTEGRATING.

    ``weights`` / ``inst_scales`` — the (nt, nf) per-instant window weights and RMS
    levels (M31 effective windows at unit scale for the continuous reduction; the M27
    per-window weight/scale for the windowed limit). ``gamma4`` / ``gamma3`` — the
    (nt, 6) per-instant per-component target arrays. Returns the M27-summary dict shape
    PLUS the per-reduction / per-instant induced-kurtosis + lambda_ng diagnostics.

    ``exact`` (M34) — when False (default, M33), the underlying-Gaussian correlation of
    each instant's tensor is TAKEN as the target correlation R = M_{0,j}/(sig sig), so the
    transformed covariance is preserved only to LEADING order and the induced kurtosis is
    the M33 one. When True, the underlying correlation is the Grigoriu / NORTA
    correlation-matching solution rho^U (``solve_underlying_correlation``) so the
    component-wise Hermite tensor reproduces M_{0,j} EXACTLY (covariance
    preservation_error -> ~0) and the induced kurtosis is computed on the covariance-EXACT
    joint tensor. The reported ``preservation_error`` is then the EXACT (~0) one, with the
    M33 leading-order value alongside as ``preservation_error_leading``.

    The von-Mises reduction (a QUADRATIC form, no linear projection) reuses the
    max-shear plane's induced kurtosis as its representative (documented)."""
    from . import spectral_fatigue as sf
    from .joint_evolutionary_fatigue import (windowed_tensor_moment_matrices,
                                             reduce_window_tensor,
                                             _reduce_window_fixed)
    from .multiaxial_fatigue import (tensor_moment_matrices,
                                     equivalent_vonmises_moments)
    from . import nongaussian_fatigue as ngf
    Scross = np.asarray(Scross)
    T = np.asarray(durations, dtype=float).ravel()
    g4 = np.asarray(gamma4, dtype=float)
    g3 = np.asarray(gamma3, dtype=float)
    nt = len(weights)
    # stationary reference (the un-windowed tensor) for the drift=False fixed plane
    Mstat = tensor_moment_matrices(omega, Scross, nmax=4)
    stat = reduce_window_tensor(Mstat, m, C, mean_stress=mean_stress,
                                ultimate=ultimate, naz=naz, npol=npol)
    keys = ("von_mises", "normal_plane", "shear_plane")
    Dng = {k: 0.0 for k in keys}
    Dg = {k: 0.0 for k in keys}
    lam_track = {k: [] for k in keys}
    kurt_track = {k: [] for k in keys}
    wout = []
    shear_normals = []
    normal_normals = []
    fnps = []
    shapes = []
    preservation = 0.0
    preservation_leading = 0.0        # M34: the M33 leading-order error, alongside
    for j in range(nt):
        Mi = windowed_tensor_moment_matrices(omega, Scross, weights[j],
                                             scale=float(inst_scales[j]), nmax=4)
        if drift:
            red = reduce_window_tensor(Mi, m, C, mean_stress=mean_stress,
                                       ultimate=ultimate, naz=naz, npol=npol)
        else:
            red = _reduce_window_fixed(Mi, stat, m, C, mean_stress, ultimate)
        Ti = float(T[j])
        M0j = Mi[0]
        g4j = g4[j]
        g3j = g3[j]
        # the joint-tensor covariance-preservation diagnostic (max off-diagonal drift).
        # M34 exact: solve the NORTA underlying correlation rho^U so the transformed
        # covariance reproduces M_{0,j} EXACTLY (preservation_error -> ~0), and use rho^U
        # as the underlying correlation for the induced-moment diagram. M33 leading-order:
        # the underlying correlation IS the target R (the documented distortion).
        if exact:
            sol = solve_underlying_correlation(M0j, g3j, g4j, model=model)
            rho_u = sol["rho_u"]
            preservation = max(preservation, sol["preservation_error"])
            preservation_leading = max(preservation_leading,
                                       sol["preservation_error_leading"])
        else:
            rho_u = None
            tp = translation_process_covariance(M0j, g3j, g4j, model=model)
            preservation = max(preservation, tp["preservation_error"])
            preservation_leading = preservation
        # LINEAR critical planes: the induced kurtosis of the resolved scalar (on the
        # covariance-EXACT joint tensor when exact, else the M33 leading-order one)
        plane_kurt = {}
        for k in ("normal_plane", "shear_plane"):
            proj = np.asarray(red[k]["proj"], dtype=float)
            _, g3s, g4s = induced_projection_moments(M0j, proj, g3j, g4j, model=model,
                                                     underlying_R=rho_u)
            plane_kurt[k] = (g3s, g4s)
        # the von-Mises quadratic reuses the max-shear plane's induced kurtosis
        plane_kurt["von_mises"] = plane_kurt["shear_plane"]
        for k in keys:
            summ = red[k]["summary"]
            a2 = float(summ["params"]["alpha2"])
            g3s, g4s = plane_kurt[k]
            lam = ngf.nongaussian_correction_factor(
                g3s, g4s, m, alpha2=a2,
                bandwidth_correction=bandwidth_correction, model=model)
            drG = float(red[k]["damage_rate"])
            Dg[k] += drG * Ti
            Dng[k] += lam * drG * Ti
            lam_track[k].append(lam)
            kurt_track[k].append(g4s)
        shear_normals.append(np.asarray(red["shear_plane"]["normal"], dtype=float))
        normal_normals.append(np.asarray(red["normal_plane"]["normal"],
                                         dtype=float))
        fnps.append(float(red["F_np"]))
        m0 = equivalent_vonmises_moments(Mi)[0]
        shapes.append(Mi[0] / m0 if m0 > 0 else Mi[0])
        wout.append({"duration": Ti, "F_np": fnps[-1],
                     "sigma_vm": float(red["sigma_vm"]),
                     "vm_rate": float(red["von_mises"]["damage_rate"]),
                     "shear_lambda": lam_track["shear_plane"][-1],
                     "shear_kurt": kurt_track["shear_plane"][-1],
                     "normal_lambda": lam_track["normal_plane"][-1],
                     "normal_kurt": kurt_track["normal_plane"][-1],
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
    kurt_all = np.array([v for lst in kurt_track.values() for v in lst], dtype=float)
    out = {"method": "joint_nongaussian_tensor", "nt": nt, "drift": bool(drift),
           "constant_shape": bool(const), "plane_rotation_deg": rot,
           "fnp_drift": fnp_drift, "windows": wout, "stationary": stat,
           "exact": bool(exact),
           "total_time": Ttot, "preservation_error": preservation,
           "preservation_error_leading": preservation_leading,
           "lambda_min": float(lam_all.min()) if lam_all.size else 1.0,
           "lambda_max": float(lam_all.max()) if lam_all.size else 1.0,
           "lambda_mean": float(np.mean(lam_all)) if lam_all.size else 1.0,
           "induced_kurt_min": float(kurt_all.min()) if kurt_all.size else 3.0,
           "induced_kurt_max": float(kurt_all.max()) if kurt_all.size else 3.0,
           "induced_kurt_mean": float(np.mean(kurt_all)) if kurt_all.size else 3.0}
    Ceff = sf._goodman_C(C, m, mean_stress, ultimate)
    for k in keys:
        drg = Dg[k] / Ttot if Ttot > 0 else 0.0
        dr = Dng[k] / Ttot if Ttot > 0 else 0.0
        tf, _ = sf.life_and_equivalent(dr, 0.0, m, Ceff)
        out[k] = {"damage": Dng[k], "total_time": Ttot, "damage_rate": dr,
                  "life": tf, "gaussian_damage_rate": drg,
                  "lambda_ng": float(np.mean(lam_track[k])) if lam_track[k] else 1.0,
                  "induced_kurt": (float(np.mean(kurt_track[k]))
                                   if kurt_track[k] else 3.0)}
    out["damage_rate"] = out["von_mises"]["damage_rate"]
    out["life"] = out["von_mises"]["life"]
    return out


def joint_nongaussian_tensor_summary(omega, Scross, durations, fc, bw, m, C,
                                     kurt, skew=0.0, scales=None, refine=8,
                                     smooth=0.0, kurt_grid=None, skew_grid=None,
                                     bandwidth_correction=True, model="winterstein",
                                     mean_stress=0.0, ultimate=0.0, naz=24, npol=13,
                                     drift=True, scalar_equivalent=True, exact=False):
    """The 6x6 JOINT NON-GAUSSIAN TENSOR continuous instantaneous damage (theory "THE
    PER-INSTANT JOINT REDUCTION"): the M31 continuous Gaussian TENSOR spectrum reduced
    with a per-instant critical-plane re-search, the resolved plane INHERITING the
    INDUCED kurtosis of the joint component-wise Hermite transform, Miner-INTEGRATED.

    Builds the M31 Gaussian tensor summary (``wigner_ville_tensor_summary`` — left
    byte-identical, reported alongside as ``gaussian``). In the GAUSSIAN-tensor limit
    (gamma_4_c == 3 on every component AND instant) DELEGATES to it byte-identically. In
    the WINDOWED limit (refine = 1, smooth = 0) the per-instant reduction rebuilds the
    M27 per-window moment matrices so the Gaussian per-window reduction is byte-identical
    to M27 before the induced lambda_ng scales it. Otherwise builds the M31 fine-grid
    effective windows and reduces per instant with the induced-kurtosis lambda_ng.

    ``kurt`` / ``skew`` — the per-component target (scalar / (6,) per-component / (start,
    end) sweep; or the per-instant ``kurt_grid`` (nt,) or (nt, 6)) — see
    ``component_kurtosis_schedule``. ``scalar_equivalent`` — if True (default) also
    computes the M32 EQUIVALENT-SCALAR tensor answer (kurtosis imposed on the resolved
    scalar) as the ``scalar_equivalent`` side-by-side entry (a direct M32 delegation, so
    it is byte-identical to M32); the JOINT vs equivalent-scalar damage difference is the
    M33 <-> M32 boundary.

    ``exact`` (M34) — when False (default) the underlying-Gaussian correlation is TAKEN
    as the target correlation (M33 leading-order, the transformed covariance preserved to
    leading order); when True the per-instant reduction solves the Grigoriu / NORTA
    underlying correlation rho^U (``solve_underlying_correlation``) so the component-wise
    Hermite tensor reproduces the target covariance EXACTLY (the covariance
    ``preservation_error`` -> ~0, reported with the M33 leading-order value alongside as
    ``preservation_error_leading``) and the induced kurtosis / lambda_ng / damage are
    computed on the covariance-EXACT joint tensor. The Gaussian and scalar-equivalent
    side entries are byte-identical either way (the EXACT path is a NEW path ALONGSIDE
    the M33 leading-order one). Returns the M27 summary dict shape PLUS the per-reduction
    gaussian_damage_rate / lambda_ng / induced_kurt, the induced-kurtosis + lambda drift
    diagnostics, the covariance ``preservation_error`` and the gamma_4_c(t) schedule."""
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
    gamma4, gamma3 = component_kurtosis_schedule(s, kurt, skew, ncomp=6,
                                                 kurt_grid=kurt_grid,
                                                 skew_grid=skew_grid)

    if _is_gaussian_component_schedule(gamma4, gamma3):
        return _wrap_gaussian_tensor(gauss, gamma4, gamma3, refine, smooth, exact=exact)

    if wv.is_windowed_limit(refine, smooth):
        win = joint_evolutionary_windows(freqs, durations, fc, bw, scales)
        weights = [np.asarray(w["W"], dtype=float) for w in win]
        inst_scales = [float(w["scale"]) for w in win]
        dur = np.asarray([float(w["duration"]) for w in win], dtype=float)
    else:
        eff = wv.instantaneous_effective_windows(
            freqs, durations, fc, bw, scales=scales, refine=refine, smooth=smooth)
        weights = [eff["Weff"][j] for j in range(eff["nt"])]
        inst_scales = [1.0] * eff["nt"]
        dur = eff["dur"]
    out = _reduce_instant_tensors_joint_ng(
        omega, Scross, weights, inst_scales, dur, gamma4, gamma3, m, C, mean_stress,
        ultimate, naz, npol, bandwidth_correction, model, drift=drift, exact=exact)
    out["refine"] = refine
    out["smooth"] = float(smooth)
    out["continuous"] = not wv.is_windowed_limit(refine, smooth)
    out["gaussian"] = gauss
    out["gamma4"] = gamma4
    out["gamma3"] = gamma3
    out["gamma4_range"] = float(np.max(gamma4) - np.min(gamma4))
    out["bandwidth_correction"] = bool(bandwidth_correction)
    out["peak_drift"] = gauss.get("peak_drift", 0.0)
    out["joint"] = True

    # the M32 EQUIVALENT-SCALAR tensor answer (kurtosis imposed on the resolved scalar)
    # as the side-by-side reference — a direct M32 delegation (byte-identical to M32)
    if scalar_equivalent:
        from . import nongaussian_wigner_ville_fatigue as ngwv
        k_rep, s_rep = _representative_scalar_kurtosis(kurt, skew)
        out["scalar_equivalent"] = ngwv.nongaussian_wigner_ville_tensor_summary(
            omega, Scross, durations, fc, bw, m, C, k_rep, skew=s_rep, scales=scales,
            refine=refine, smooth=smooth, bandwidth_correction=bandwidth_correction,
            model=model, mean_stress=mean_stress, ultimate=ultimate, naz=naz,
            npol=npol, drift=drift)
    else:
        out["scalar_equivalent"] = None
    return out


def _wrap_gaussian_tensor(gauss, gamma4, gamma3, refine, smooth, exact=False):
    """Wrap the M31 Gaussian continuous tensor summary as the M33 result in the exact
    gamma_4_c == 3 limit — every component transform is the identity, the induced
    kurtosis is 3, lambda_ng == 1, so every reduction rate IS the M31 Gaussian rate
    BYTE-IDENTICALLY (the delegation). In the M34 EXACT limit rho^U = R = the target
    correlation, so the preservation error is 0 and the delegation is identical."""
    out = dict(gauss)
    out["method"] = "joint_nongaussian_tensor"
    out["delegated"] = "m31_gaussian"
    out["joint"] = True
    out["exact"] = bool(exact)
    out["refine"] = int(refine)
    out["smooth"] = float(smooth)
    out["gamma4"] = np.asarray(gamma4, dtype=float)
    out["gamma3"] = np.asarray(gamma3, dtype=float)
    out["gamma4_range"] = 0.0
    out["lambda_min"] = 1.0
    out["lambda_max"] = 1.0
    out["lambda_mean"] = 1.0
    out["induced_kurt_min"] = 3.0
    out["induced_kurt_max"] = 3.0
    out["induced_kurt_mean"] = 3.0
    out["preservation_error"] = 0.0
    out["preservation_error_leading"] = 0.0
    out["bandwidth_correction"] = True
    out["scalar_equivalent"] = None
    for k in ("von_mises", "normal_plane", "shear_plane"):
        r = dict(gauss[k])
        r["gaussian_damage_rate"] = float(gauss[k]["damage_rate"])
        r["lambda_ng"] = 1.0
        r["induced_kurt"] = 3.0
        out[k] = r
    return out


# ============================================================================
# (b) The multivariate NON-GAUSSIAN non-stationary Monte-Carlo (build item 2)
# ============================================================================

def _sample_cov_rel_error(X, M0_target):
    """The max off-diagonal deviation between the SAMPLE correlation of a synthesised
    record ``X`` (nt, n) and the TARGET correlation of ``M0_target`` (n, n) — the M34
    Monte-Carlo covariance-preservation diagnostic. The marginals (diagonal) are
    preserved by construction; this measures the CROSS-structure drift the leptokurtic
    translation induces (large for the M33 leading-order record, ~0 for the M34
    covariance-exact one). Restricted to components with positive target AND sample
    variance."""
    X = np.asarray(X, dtype=float)
    if X.shape[0] < 2:
        return 0.0
    n = X.shape[1]
    Xc = X - X.mean(axis=0, keepdims=True)
    Cs = (Xc.T @ Xc) / X.shape[0]
    dt = np.clip(np.diag(np.asarray(M0_target, dtype=float)), 0.0, None)
    ds = np.clip(np.diag(Cs), 0.0, None)
    # restrict to the SIGNIFICANT (materially-contributing) components: a numerically-zero
    # component's sample correlation is floating-point noise (the same relative-variance
    # floor the NORTA inversion uses)
    tmax = float(np.max(dt)) if dt.size else 0.0
    ok = (dt > tmax * 1e-9) & (ds > 0.0)
    idx = np.where(ok)[0]
    if idx.size < 2:
        return 0.0
    st = np.sqrt(dt[idx])
    ss = np.sqrt(ds[idx])
    Rt = M0_target[np.ix_(idx, idx)] / np.outer(st, st)
    Rs = Cs[np.ix_(idx, idx)] / np.outer(ss, ss)
    dev = np.abs(Rs - Rt)
    np.fill_diagonal(dev, 0.0)
    return float(np.max(dev)) if dev.size else 0.0


def _rescale_block_to_underlying(Sw, freqs, gamma3, gamma4, model, copula="gaussian", copula_params=None):
    """Return a copy of the block cross-PSD ``Sw`` (nf, 6, 6) with its OFF-DIAGONAL
    cross-spectra scaled by the NORTA ratio rho^U_cc' / R_cc' (M34), so a Gaussian
    synthesised from it carries the underlying-Gaussian correlation rho^U. The block's
    0th-moment covariance M_0 = integral of Re(Sw) df fixes the target correlation R and
    (via ``solve_underlying_correlation``) the underlying rho^U; the ratio is applied to
    each Hermitian off-diagonal pair (the diagonal auto-PSDs are left EXACT, so the
    marginals are unchanged). The scaling preserves the cross-spectrum's frequency SHAPE
    and phase; the synthesiser's per-bin PSD-cone clip absorbs any residual loss of
    positive-definiteness from the rescale (a tiny effect for the modest ratios of a
    leptokurtic translation)."""
    Sw = np.asarray(Sw)
    # block covariance (correlation only needs the co-spectrum integral; the constant
    # factor cancels in the ratio). Real part = the zero-lag covariance contribution.
    M0 = trapezoid(Sw.real, freqs, axis=0)
    M0 = 0.5 * (M0 + M0.T)
    sol = solve_underlying_correlation(M0, gamma3, gamma4, model=model, copula=copula, copula_params=copula_params)
    R = sol["target_R"]
    rho_u = sol["rho_u"]
    supp = sol["support"]
    n = R.shape[0]
    ratio = np.ones((n, n))
    for c in range(n):
        for cp in range(n):
            # only rescale pairs of SIGNIFICANT components whose target correlation is
            # meaningfully nonzero (a near-zero R gives rho^U ~ R -> ratio ~ 1 anyway);
            # a degenerate component's row is left untouched (ratio 1)
            if c != cp and supp[c] and supp[cp] and abs(R[c, cp]) > 1e-6:
                r = rho_u[c, cp] / R[c, cp]
                ratio[c, cp] = min(max(r, 0.0), 4.0)   # clip for synthesiser stability
    return Sw * ratio[None, :, :]


def synthesize_joint_nongaussian_history(omega, Scross, durations, fc, bw, seed,
                                         kurt, skew=0.0, scales=None, refine=8,
                                         smooth=0.0, kurt_grid=None, skew_grid=None,
                                         fs=None, model="winterstein", exact=False,
                                         copula="gaussian", copula_params=None):
    """Synthesise the MULTIVARIATE NON-GAUSSIAN NON-STATIONARY stress-tensor record
    (theory "THE MULTIVARIATE NON-GAUSSIAN NON-STATIONARY MONTE-CARLO"): the M27/M31
    multivariate non-separable synthesiser on the fine instant grid (per-instant
    correlated 6-component Gaussian blocks of the WINDOWED tensor) with EACH COMPONENT of
    each block pushed through its OWN memoryless Winterstein-Hermite transform (1) to
    THAT instant's per-component target gamma_4_c(t_j) / gamma_3_c(t_j) — the VECTOR
    transform, so the record's LOCAL joint marginals track the per-component targets.
    Returns (t, X, info) with ``X`` (nt, 6) the six Voigt components, ``info`` = {edges,
    fs, gamma4, gamma3}.

    ``exact`` (M34) — when False (default, M33) each block's correlated Gaussian is
    synthesised from the TARGET cross-PSD, so the component-wise Hermite transform
    DISTORTS the cross-covariance away from the target (the record's sample covariance
    drifts). When True, each block's cross-spectrum off-diagonals are pre-scaled by the
    NORTA ratio rho^U_cc' / R_cc' (``solve_underlying_correlation`` on the block's own
    covariance) so the UNDERLYING Gaussian carries correlation rho^U; the Hermite
    transform then brings the transformed cross-covariance BACK to the target, so the
    corrected record's sample covariance matches the target (the marginals unchanged —
    the diagonal auto-PSDs are untouched). In the Gaussian-tensor limit the ratio is 1
    and the record reduces EXACTLY to the M27/M31 multivariate non-separable history
    either way."""
    from .joint_evolutionary_fatigue import joint_evolutionary_windows
    from .multiaxial_fatigue import synthesize_multiaxial_history
    from . import wigner_ville_fatigue as wv
    from . import nongaussian_fatigue as ngf
    freqs = np.asarray(omega, dtype=float) / (2.0 * np.pi)
    S = np.asarray(Scross)
    refine = max(1, int(refine))
    # the fine-grid schedule + per-instant windows (the M27 per-window blocks refined;
    # smoothing enters only the closed-form slices, so the MC uses the RAW fine grid —
    # the M31/M32 Monte-Carlo convention, documented)
    if wv.is_windowed_limit(refine, smooth):
        win = joint_evolutionary_windows(freqs, durations, fc, bw, scales)
        blocks = [( (float(w["scale"]) ** 2) * w["W"][:, None, None] * S,
                    float(w["duration"]) ) for w in win]
        sch = wv.instantaneous_schedule(durations, fc, bw, scales=scales, refine=1)
    else:
        sch = wv.instantaneous_schedule(durations, fc, bw, scales=scales,
                                        refine=refine)
        blocks = []
        for j in range(sch["nt"]):
            fcj, bwj, aj = sch["fc"][j], sch["bw"][j], sch["scale"][j]
            if bwj > 0.0:
                W = np.exp(-((freqs - fcj) ** 2) / (2.0 * bwj ** 2))
            else:
                W = np.ones_like(freqs)
            blocks.append(((aj ** 2) * W[:, None, None] * S, float(sch["dur"][j])))
    gamma4, gamma3 = component_kurtosis_schedule(sch["s"], kurt, skew, ncomp=6,
                                                 kurt_grid=kurt_grid,
                                                 skew_grid=skew_grid)
    fmax = float(np.max(freqs))
    if fs is None:
        fs = 8.0 * fmax
    T = np.array([b[1] for b in blocks], dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(T)])
    chunks = []
    cov_err = 0.0                     # max per-block sample-covariance drift diagnostic
    for j, (Sw, dur_j) in enumerate(blocks):
        # the block's TARGET covariance (before any rescale) fixes the correlation the
        # transformed record should reproduce — the sample-covariance drift diagnostic
        M0_target = trapezoid(np.asarray(Sw).real, freqs, axis=0)
        M0_target = 0.5 * (M0_target + M0_target.T)
        # M34 EXACT-covariance: pre-scale this block's cross-spectrum off-diagonals by
        # the NORTA ratio rho^U/R (computed from the block's OWN 0th-moment covariance)
        # so the synthesised UNDERLYING Gaussian carries correlation rho^U; the Hermite
        # transform below then restores the transformed cross-covariance to the target.
        # The diagonal auto-PSDs are untouched (the marginals stay exact). In the M33
        # leading-order / Gaussian case the ratio is 1 (a no-op, byte-identical).
        if exact and not _is_gaussian_component_schedule(gamma4[j:j + 1],
                                                         gamma3[j:j + 1]):
            Sw = _rescale_block_to_underlying(Sw, freqs, gamma3[j], gamma4[j], model=model, copula=copula, copula_params=copula_params)
        _t, Xi = synthesize_multiaxial_history(freqs, Sw, dur_j, int(seed) + j, fs=fs)
        
        if copula == "t":
            rng_t = np.random.default_rng(int(seed) + j)
            nu = copula_params if copula_params is not None else 4.0
            W = rng_t.chisquare(nu)
            Xi = Xi * np.sqrt(nu / W)
            U = stats.t.cdf(Xi, df=nu)
            Xi = stats.norm.ppf(U)
            
        # per-component memoryless Hermite transform (VECTOR transform; identity at a
        # Gaussian component) — standardise each component, transform, rescale
        for c in range(6):
            g4c = float(gamma4[j, c])
            g3c = float(gamma3[j, c])
            coeffs = ngf.hermite_coefficients(g3c, g4c, model=model)
            h3, h4, kappa = coeffs
            if abs(h3) < 1e-15 and abs(h4) < 1e-15 and abs(kappa - 1.0) < 1e-15:
                continue
            xc = Xi[:, c]
            sig = float(np.std(xc))
            if sig <= 0.0:
                continue
            z = (xc - float(np.mean(xc))) / sig
            Xi[:, c] = sig * ngf.hermite_transform(z, g3c, g4c, coeffs=coeffs)
        cov_err = max(cov_err, _sample_cov_rel_error(Xi, M0_target))
        chunks.append(Xi)
    X = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 6))
    t = np.arange(X.shape[0]) / fs
    return t, X, {"edges": edges, "fs": fs, "gamma4": gamma4, "gamma3": gamma3,
                  "exact": bool(exact), "sample_cov_error": float(cov_err)}


def joint_nongaussian_monte_carlo_damage(omega, Scross, durations, fc, bw, m, C,
                                         seed, kurt, skew=0.0, scales=None, refine=8,
                                         smooth=0.0, kurt_grid=None, skew_grid=None,
                                         fs=None, mean_stress=0.0, ultimate=0.0,
                                         naz=24, npol=13, model="winterstein",
                                         reduction="shear_plane", summary=None,
                                         exact=False, copula="gaussian", copula_params=None):
    """The JOINT non-Gaussian tensor damage rate by MULTIVARIATE non-stationary
    Monte-Carlo (theory): synthesise the multivariate non-Gaussian record
    (``synthesize_joint_nongaussian_history`` — per-instant correlated 6-component
    blocks, EACH component pushed through its own Hermite transform), PROJECT each
    instant's block onto THAT instant's critical plane (a resolved non-Gaussian scalar),
    rainflow-count (ASTM E1049) the WHOLE record and Miner-sum — the independent
    time-domain answer the JOINT Miner-integral approximates. The induced SAMPLE kurtosis
    of the resolved projection tracks the closed-form induced gamma_4^s.

    ``reduction`` — which critical plane to project onto ("shear_plane" default, the
    primary multiaxial driver; or "normal_plane"). ``summary`` optionally supplies a
    precomputed ``joint_nongaussian_tensor_summary`` so the per-instant planes are
    shared. ``exact`` (M34) — when True the record is synthesised from the NORTA
    underlying correlation rho^U so its sample covariance matches the TARGET (where the
    M33 leading-order record's drifts); the ``sample_cov_error`` returned then measures
    that (near-zero) residual. In the Gaussian-tensor limit reduces EXACTLY to the
    M27/M31 multivariate MC either way. Returns the M27 Monte-Carlo dict shape plus the
    sample ``kurtosis`` / ``skewness`` of the resolved projection (tracking the induced
    value) and the ``sample_cov_error`` (max relative off-diagonal deviation of the
    record's sample covariance from the target)."""
    from . import spectral_fatigue as sf
    from . import wigner_ville_fatigue as wv
    Ceff = sf._goodman_C(C, m, mean_stress, ultimate)
    refine = max(1, int(refine))
    projkey = "shear_proj" if reduction == "shear_plane" else "normal_proj"
    # GAUSSIAN-tensor short-circuit: gamma_4_c == 3 on every component -> every component
    # transform is the identity, so the record IS the M27/M31 multivariate Gaussian
    # record. DELEGATE to the M31 continuous tensor Monte-Carlo (which itself delegates
    # to the M27 windowed MC in the windowed limit) for BIT-IDENTITY — exactly as the
    # M32 MC delegates to the M31 MC in the Gaussian limit.
    sch0 = wv.instantaneous_schedule(durations, fc, bw, scales=scales, refine=refine)
    g4chk, g3chk = component_kurtosis_schedule(sch0["s"], kurt, skew, ncomp=6,
                                               kurt_grid=kurt_grid,
                                               skew_grid=skew_grid)
    if _is_gaussian_component_schedule(g4chk, g3chk):
        mc = wv.wigner_ville_tensor_monte_carlo_damage(
            omega, Scross, durations, fc, bw, m, C, seed=seed, scales=scales,
            refine=refine, smooth=smooth, fs=fs, mean_stress=mean_stress,
            ultimate=ultimate, naz=naz, npol=npol, reduction=reduction)
        mc = dict(mc)
        mc["method"] = "joint_nongaussian_monte_carlo"
        mc.setdefault("kurtosis", 3.0)
        mc.setdefault("skewness", 0.0)
        mc.setdefault("sample_cov_error", 0.0)
        return mc
    t, X, info = synthesize_joint_nongaussian_history(
        omega, Scross, durations, fc, bw, seed, kurt, skew=skew, scales=scales,
        refine=refine, smooth=smooth, kurt_grid=kurt_grid, skew_grid=skew_grid,
        fs=fs, model=model, exact=exact, copula=copula, copula_params=copula_params)
    edges = info["edges"]
    nwin = len(edges) - 1
    # the per-instant critical planes (re-searched from each instant's tensor) — reuse
    # the Gaussian M31 tensor summary's per-instant plane list (the plane geometry is a
    # Gaussian-covariance quantity, matching the closed-form reduction)
    if summary is None:
        gauss = wv.wigner_ville_tensor_summary(
            omega, Scross, durations, fc, bw, m, C, scales=scales, refine=refine,
            smooth=0.0, mean_stress=mean_stress, ultimate=ultimate, naz=naz,
            npol=npol, drift=True)
    else:
        gauss = summary.get("gaussian", summary)
    gwins = gauss.get("windows")
    D = 0.0
    ncyc = 0.0
    proj_all = []
    for i in range(nwin):
        sel = (t >= edges[i]) & (t < edges[i + 1])
        seg = X[sel]
        if seg.shape[0] < 2:
            continue
        if gwins is not None and i < len(gwins) and projkey in gwins[i]:
            proj = np.asarray(gwins[i][projkey], dtype=float)
        else:
            proj = np.asarray(gauss[reduction]["proj"], dtype=float)
        sproj = seg @ proj
        ranges, counts = sf.rainflow_count(sproj)
        D += (float(np.sum(counts * ranges ** m) / Ceff) if ranges.size else 0.0)
        ncyc += float(counts.sum())
        proj_all.append(sproj)
    Ttot = float(edges[-1]) if edges.size else float(np.sum(durations))
    dr = D / Ttot if Ttot > 0 else 0.0
    nu = ncyc / Ttot if Ttot > 0 else 0.0
    tf, s_eq = sf.life_and_equivalent(dr, nu, m, Ceff)
    sfull = np.concatenate(proj_all) if proj_all else np.zeros(0)
    if sfull.size:
        sc = sfull - np.mean(sfull)
        var = float(np.mean(sc * sc))
        kurt_s = float(np.mean(sc ** 4) / var ** 2) if var > 0 else 3.0
        skew_s = float(np.mean(sc ** 3) / var ** 1.5) if var > 0 else 0.0
        rms = float(np.std(sfull))
    else:
        kurt_s, skew_s, rms = 3.0, 0.0, 0.0
    return {"method": "joint_nongaussian_monte_carlo", "damage_rate": dr, "life": tf,
            "s_eq": s_eq, "ncycles": ncyc, "duration": Ttot, "reduction": reduction,
            "kurtosis": kurt_s, "skewness": skew_s, "rms": rms, "refine": refine,
            "smooth": float(smooth), "exact": bool(exact),
            "sample_cov_error": float(info.get("sample_cov_error", 0.0))}

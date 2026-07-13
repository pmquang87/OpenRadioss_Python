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
                               return_components=False):
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
    additionally returns the (h3, h4, kappa) per-component coefficients."""
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
    # underlying-Gaussian correlation on the support (R_cc' = M0/(sig_c sig_c'))
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


def joint_lambda_ng(M0, proj, gamma3, gamma4, m, alpha2=1.0,
                    bandwidth_correction=True, model="winterstein"):
    """The JOINT non-Gaussian amplification lambda_ng of a resolved plane: compute the
    INDUCED (gamma_3^s, gamma_4^s) of the projection ``proj`` under the joint transform
    (``induced_projection_moments``) and feed those to the M24 closed-form
    ``nongaussian_correction_factor`` at the plane's bandwidth ``alpha2`` (theory — the
    resolved plane inherits the joint tensor kurtosis, then the SAME M24 amplitude
    correction applies). Returns (lambda_ng, gamma3_induced, gamma4_induced). In the
    Gaussian tensor limit the induced kurtosis is 3 and lambda_ng == 1 EXACTLY."""
    from . import nongaussian_fatigue as ngf
    _, g3s, g4s = induced_projection_moments(M0, proj, gamma3, gamma4, model=model)
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
                                     npol, bandwidth_correction, model, drift=True):
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
        # the joint-tensor covariance-preservation diagnostic (max off-diagonal drift)
        tp = translation_process_covariance(M0j, g3j, g4j, model=model)
        preservation = max(preservation, tp["preservation_error"])
        # LINEAR critical planes: the induced kurtosis of the resolved scalar
        plane_kurt = {}
        for k in ("normal_plane", "shear_plane"):
            proj = np.asarray(red[k]["proj"], dtype=float)
            _, g3s, g4s = induced_projection_moments(M0j, proj, g3j, g4j, model=model)
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
           "total_time": Ttot, "preservation_error": preservation,
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
                                     drift=True, scalar_equivalent=True):
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
    M33 <-> M32 boundary. Returns the M27 summary dict shape PLUS the per-reduction
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
        return _wrap_gaussian_tensor(gauss, gamma4, gamma3, refine, smooth)

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


def _wrap_gaussian_tensor(gauss, gamma4, gamma3, refine, smooth):
    """Wrap the M31 Gaussian continuous tensor summary as the M33 result in the exact
    gamma_4_c == 3 limit — every component transform is the identity, the induced
    kurtosis is 3, lambda_ng == 1, so every reduction rate IS the M31 Gaussian rate
    BYTE-IDENTICALLY (the delegation)."""
    out = dict(gauss)
    out["method"] = "joint_nongaussian_tensor"
    out["delegated"] = "m31_gaussian"
    out["joint"] = True
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

def synthesize_joint_nongaussian_history(omega, Scross, durations, fc, bw, seed,
                                         kurt, skew=0.0, scales=None, refine=8,
                                         smooth=0.0, kurt_grid=None, skew_grid=None,
                                         fs=None, model="winterstein"):
    """Synthesise the MULTIVARIATE NON-GAUSSIAN NON-STATIONARY stress-tensor record
    (theory "THE MULTIVARIATE NON-GAUSSIAN NON-STATIONARY MONTE-CARLO"): the M27/M31
    multivariate non-separable synthesiser on the fine instant grid (per-instant
    correlated 6-component Gaussian blocks of the WINDOWED tensor) with EACH COMPONENT of
    each block pushed through its OWN memoryless Winterstein-Hermite transform (1) to
    THAT instant's per-component target gamma_4_c(t_j) / gamma_3_c(t_j) — the VECTOR
    transform, so the record's LOCAL joint marginals track the per-component targets.
    Returns (t, X, info) with ``X`` (nt, 6) the six Voigt components, ``info`` = {edges,
    fs, gamma4, gamma3}.

    In the Gaussian-tensor limit every component transform is the identity, so the record
    reduces EXACTLY to the M27/M31 multivariate non-separable history."""
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
    for j, (Sw, dur_j) in enumerate(blocks):
        _t, Xi = synthesize_multiaxial_history(freqs, Sw, dur_j, int(seed) + j, fs=fs)
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
        chunks.append(Xi)
    X = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 6))
    t = np.arange(X.shape[0]) / fs
    return t, X, {"edges": edges, "fs": fs, "gamma4": gamma4, "gamma3": gamma3}


def joint_nongaussian_monte_carlo_damage(omega, Scross, durations, fc, bw, m, C,
                                         seed, kurt, skew=0.0, scales=None, refine=8,
                                         smooth=0.0, kurt_grid=None, skew_grid=None,
                                         fs=None, mean_stress=0.0, ultimate=0.0,
                                         naz=24, npol=13, model="winterstein",
                                         reduction="shear_plane", summary=None):
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
    shared. In the Gaussian-tensor limit reduces EXACTLY to the M27/M31 multivariate MC.
    Returns the M27 Monte-Carlo dict shape plus the sample ``kurtosis`` / ``skewness`` of
    the resolved projection (tracking the induced value)."""
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
        return mc
    t, X, info = synthesize_joint_nongaussian_history(
        omega, Scross, durations, fc, bw, seed, kurt, skew=skew, scales=scales,
        refine=refine, smooth=smooth, kurt_grid=kurt_grid, skew_grid=skew_grid,
        fs=fs, model=model)
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
            "smooth": float(smooth)}

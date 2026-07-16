"""
Spectral non-proportional multiaxial fatigue — M23: the FREQUENCY-DOMAIN
non-proportionality factor and critical-plane damage estimated DIRECTLY from the
stress-tensor cross-PSD spectral-MOMENT matrices, with NO synthesised history.
Where M22 keeps the full 2-D resolved-shear PATH on each candidate plane and
counts (by rainflow) the damage a rotating shear vector accrues over a
SYNTHESISED time record, M23 is the SPECTRAL (no-history) counterpart: it reads
the SAME non-proportionality out of the 2x2 in-plane shear block of the M21
moment matrix M_0 and applies it as a closed-form correction to the M20/M21
spectral estimators — the Pitoiset / Cristofori-Susmel-Tovo spectral method the
M22 time-domain path count deliberately deferred.

Fortran origin
--------------
There is NONE. ``engine/source/input/freimpl.F`` (the /IMPL reader, re-read line
by line for M16-M22 AND AGAIN for M23 — 639 lines, fetched from
raw.githubusercontent.com) parses only /IMPL/DYNA (the DIRECT Newmark/HHT
integrator), /IMPL/BUCKL, /IMPL/DT, /IMPL/NONLIN and /IMPL/ARCL plus the
linear-solver housekeeping — there is no /FATIG, no S-N / Miner branch, no
von-Mises / critical-plane / stress-tensor cross-PSD machinery, and NO
frequency-domain non-proportionality / modified-Wohler / spectral-invariant
fatigue estimator of any kind. The sole ``PSD`` token in the whole file is still
``IMUMPSD`` (line 269), a MUMPS-solver flag, not a power spectral density.
OpenRadioss is a time-domain crash/impact code: the stationary random-vibration
MULTIAXIAL fatigue analysis — proportional (M21) OR non-proportional, time-domain
(M22) OR spectral (M23) — is simply not part of the open-source solver, exactly
as M16 found for the real eigensolver, M17/M18 for the transfer functions, M19
for the PSD machinery, M20 for the scalar spectral fatigue, M21 for the
multiaxial spectral fatigue and M22 for the non-proportional path count.

So M23 does exactly what M16-M22 did: it ports the spectral non-proportional
estimator as a clean LIBRARY capability and drives it with a minimal PORT engine
sub-flag (/IMPL/FATIG/MULT/NPROP/SPEC — the SPECTRAL, no-history analogue of the
M22 /IMPL/FATIG/MULT/NPROP time-domain card). Nothing in the M10 direct
integrator, the M16 REAL eigensolver, the M17/M18 superposition, the M19 PSD
path, the M20 SCALAR fatigue, the M21 MULTIAXIAL SPECTRAL path OR the M22
NON-PROPORTIONAL TIME-DOMAIN path is touched: the spectral non-proportional
estimator is a NEW, parallel path that CONSUMES the M21 moment-matrix machinery
(``tensor_moment_matrices`` / ``normal_projection`` / ``shear_projection`` /
``_inplane_basis`` / ``candidate_normals``) read-only, and produces its answer
ALONGSIDE the M21 spectral and the M22 time-domain answers so a listing can show
all three side by side.

Theory — spectral non-proportional multiaxial fatigue
-----------------------------------------------------
(Pitoiset & Preumont, "Spectral methods for multiaxial random fatigue analysis
of metallic structures", Int. J. Fatigue 22, 2000 — the frequency-domain
multiaxial projection and the cross-PSD eigenstructure; Cristofori, Susmel &
Tovo, "A stress-invariant based spectral method to estimate fatigue life under
multiaxial random loading", Int. J. Fatigue 30, 2008 — the spectral critical-
plane / modified-Wohler estimator this module ports; Susmel & Lazzarin, "A
bi-parametric Wohler curve for high cycle multiaxial fatigue assessment",
Fatigue Fract. Engng Mater. Struct. 25, 2002 — the MODIFIED WOHLER CURVE method
and the stress ratio rho = sigma_a/tau_a; Backstrom & Marquis, "A review of
multiaxial fatigue of weldments: experimental results, design code and critical
plane approaches", FFEMS 24, 2001 — the non-proportionality background;
Carpinteri, Spagnoli, Vantadori & Bagni, "Structural integrity assessment of
metallic components under multiaxial fatigue: the C-S criterion and its
evolution", FFEMS 36, 2013 — the spectral critical-plane weighting; Preumont &
Piefort 1994 for the equivalent-von-Mises frequency-domain projection M21 uses.)

THE SPECTRAL SHEAR-PATH COVARIANCE. The M21 machinery already gives, per element,
the 6x6 stress-tensor cross-PSD S_sigmasigma(Omega) = H_sigma S_ff H_sigma^H and
its spectral-MOMENT matrices M_n = (1/pi) int Omega^n S_sigmasigma dOmega. The
n = 0 matrix M_0 IS the stress COVARIANCE Psi = E[sigma sigma^T] (a stationary
zero-mean Gaussian response carries no DC, so the covariance equals the
mean-square). On a candidate plane of unit normal n with in-plane orthonormal
axes (a, b) (``_inplane_basis``), the two resolved-shear projections p_a =
shear_projection(n, a), p_b = shear_projection(n, b) give the 2x2 IN-PLANE SHEAR
CROSS-SPECTRAL MOMENT MATRIX

    Sigma_tau = [ p_a^T M_0 p_a   p_a^T M_0 p_b ]
                [ p_b^T M_0 p_a   p_b^T M_0 p_b ]                          (1)

— exactly E[ (tau_a, tau_b) (tau_a, tau_b)^T ], the COVARIANCE of the 2-D shear
path (tau_a(t), tau_b(t)) that M22 SYNTHESISES and measures in the time domain.
This is the clean cross-check between the two milestones: because M_0 = E[sigma
sigma^T] is the very covariance the multivariate synthesiser reproduces, the 2x2
block (1) built from the moment matrix EQUALS the 2x2 covariance M22 computes
from the synthesised (tau_a, tau_b) history (``np.cov`` of ``resolved_shear_path``)
— up to the seeded Monte-Carlo scatter of the finite synthesis. M23 gets it with
NO synthesis, directly from the frequency-domain integral.

THE SPECTRAL NON-PROPORTIONALITY FACTOR. The degree of non-proportionality of
the shear path is the same aspect-ratio factor M22 uses, but read spectrally off
(1):

    F_np = sqrt( lambda_2 / lambda_1 ) ,   lambda_1 >= lambda_2 >= 0        (2)

the eigenvalues of Sigma_tau (the principal shear VARIANCES: lambda_1 the
variance along the dominant in-plane shear direction, lambda_2 across it). F_np =
0 for a straight LINE (rank-1 Sigma_tau, lambda_2 = 0 — a PROPORTIONAL / rank-1
stress state, where every stress component scales with one scalar process so the
shear path is a line on EVERY plane) and F_np = 1 for a CIRCLE (isotropic
Sigma_tau, lambda_1 = lambda_2 — an equal-amplitude 90-deg-out-of-phase state).
It is BASIS-INDEPENDENT (a rotation of (a, b) leaves the eigenvalues fixed).
Crucially, F_np EQUALS the M22 TIME-DOMAIN F_np on the same cross-PSD — the
identity M_0 = E[sigma sigma^T] = the covariance the synthesis reproduces —
which is the M23<->M22 cross-validation (asserted).

THE SPECTRAL RESOLVED AMPLITUDES / RATES. Both the resolved SHEAR (in any in-plane
direction p_s) and the resolved NORMAL (p_n = normal_projection(n)) stresses are
LINEAR projections of the tensor, so each is a scalar Gaussian process whose PSD
is the rank-1 quadratic p^T S_sigmasigma(Omega) p and whose spectral MOMENTS are
p^T M_n p (M21 eq. (7)). The DOMINANT resolved-shear scalar is the projection
onto the leading eigenvector of Sigma_tau (the M21 max-shear direction); its
moments m_n^tau = p_dom^T M_n p_dom carry the M20 spectral descriptors (RMS =
sqrt(m_0) = sqrt(lambda_1), the crossing/peak rates nu_0/nu_p, the irregularity
factor alpha_2). The resolved-normal moments m_n^sigma = p_n^T M_n p_n give the
normal-stress RMS and rates. NO time history is formed: the amplitudes are RMS
(variance) measures, and a peak-stress factor (default sqrt(2), the equivalent-
sinusoid amplitude of a narrow-band Gaussian) converts an RMS to an amplitude
where a critical-plane criterion needs a stress AMPLITUDE (sigma_n,a = psf *
sqrt(m_0^sigma)) — this is the documented spectral stand-in for M22's per-record
sigma_n,max (see the deferrals).

THE NON-PROPORTIONAL SPECTRAL CORRECTION (the whole point). A scalar projection
of the shear onto its DOMINANT direction (the M21 max-shear plane) counts only
the variance lambda_1 and is BLIND to the cross-variance lambda_2 that a rotating
path also accrues. The MRH / prismatic-hull argument (Mamiya-Araujo, used by M22)
says the effective non-proportional shear amplitude of an elliptical path of
semi-axes (p, q) is sqrt(p^2 + q^2) — the diagonal of its bounding box —
NOT the major semi-axis p alone. In variance terms (a sinusoid's variance is
half its squared amplitude) p^2 -> lambda_1 and q^2 -> lambda_2, so the effective
non-proportional shear VARIANCE is

    lambda_eff = lambda_1 + lambda_2 = trace(Sigma_tau)                     (3)

and the non-proportional amplitude-scaling factor is the CLOSED FORM

    g = sqrt( lambda_eff / lambda_1 ) = sqrt( 1 + lambda_2/lambda_1 )
      = sqrt( 1 + F_np^2 )                                                  (4)

— g = 1 for a proportional LINE (F_np = 0), g = sqrt(2) for a CIRCLE (F_np = 1,
the exact MRH/radius ratio of M22), g = sqrt(1 + (q/p)^2) = sqrt(p^2+q^2)/p for
an ELLIPSE (exactly MRH/MCC). This is the spectral, closed-form image of M22's
measured path factor g = MRH/(dominant scalar amplitude): the two agree exactly
for a clean ellipse and to within the seeded Monte-Carlo scatter for a general
random path. The NON-PROPORTIONAL SPECTRAL correction therefore scales the
DOMINANT-shear PSD moments by lambda_eff/lambda_1 = 1 + F_np^2,

    m_n^eff = (1 + F_np^2) * m_n^tau                                        (5)

(equivalently the effective shear PSD is S_tau,eff(Omega) = (1 + F_np^2)
S_tau,dom(Omega), preserving the spectral SHAPE — a documented approximation:
the shape of the cross-shear PSD may differ from the dominant one; the amplitude
correction is exact, the higher-moment shape reuse is the approximation). Feeding
m_n^eff to the M20 estimators (narrow-band / Dirlik / Wirsching-Light /
Tovo-Benasciutti) scales the damage by (1 + F_np^2)^(m/2) = g^m over the
uncorrected M21 projected-scalar answer — for a circle exactly 2^(m/2), the SAME
factor M22's MRH path count produces (the two non-proportional methods converge,
the whole point). For F_np = 0 (proportional) m_n^eff = m_n^tau and the estimate
reduces EXACTLY to the M21 max-shear critical-plane spectral answer (no
correction, no spurious damage).

THE MODIFIED WOHLER CURVE / CRITICAL-PLANE MODELS. The critical plane and the
normal-stress effect follow Susmel-Lazzarin's MODIFIED WOHLER CURVE method and
Findley / Fatemi-Socie, computed spectrally (RMS amplitudes, no history):
* the stress ratio rho = sigma_a / tau_a (Susmel-Tovo) — the ratio of the normal-
  stress amplitude sigma_a = psf sqrt(m_0^sigma) to the (non-proportional) shear
  amplitude tau_a = sqrt(lambda_eff) = g sqrt(lambda_1) on the plane — measures
  how normal-stress-dominated the plane is (rho = 0 pure shear, rho large
  tension-dominated); it indexes the modified Wohler curve in Susmel's method and
  is REPORTED per plane;
* FINDLEY (1959, spectral): the equivalent resolved stress is the LINEAR
  combination c(t) = g tau_dom(t) + k sigma_n(t) = (g p_dom + k p_n)^T sigma(t),
  ITSELF a linear projection, so its spectral moments are (g p_dom + k p_n)^T M_n
  (g p_dom + k p_n) DIRECTLY from the moment matrices (no history) — the spectral-
  invariant / linear-combination form of Findley's tau_a + k sigma_n (this folds
  in the shear/normal PHASE correlation the M22 additive record-max form drops;
  see the deferrals);
* FATEMI-SOCIE (1988, spectral): the parameter multiplies the shear amplitude by
  (1 + k sigma_n,a/sigma_y), a per-plane constant, so m_n^FS = (1 + k
  sigma_n,a/sigma_y)^2 m_n^eff;
* SHEAR-PATH: the pure non-proportional shear model m_n^eff of (5), no normal term
  — the model that reduces EXACTLY to the M21 max-shear spectral answer for a
  proportional state (used for the reduction validation).
The critical plane MAXIMISES the model's amplitude-only CRITERION PARAMETER
(tau_a + k sigma_n,a for Findley, tau_a(1 + k sigma_n,a/sigma_y) for FS, tau_a
for shear-path — the spectral, RMS-amplitude image of the textbook definitions),
so the O(n_planes) scan costs no estimator evaluation; only the winning plane is
run through the four M20 estimators.

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* LIBRARY-FIRST sub-flag (/IMPL/FATIG/MULT/NPROP/SPEC) — no upstream equivalent,
  exactly as established for M16-M22's PORT cards.
* The non-proportional amplitude correction (5) uses the EXACT-ellipse factor
  g = sqrt(1 + F_np^2) (the MRH/MCC closed form) and REUSES the dominant-shear
  PSD spectral SHAPE for the higher moments. M22 measures the actual MRH of the
  synthesised hull (which for a general random path differs from the clean
  ellipse by the seeded scatter) and rainflow-counts the real history; the two
  converge for a clean ellipse and agree within the M20 rainflow-vs-Dirlik +
  path scatter otherwise (validated). The exact-ellipse shape reuse is the
  documented spectral approximation.
* The normal-stress AMPLITUDE is a peak-stress factor times the RMS (default psf
  = sqrt(2), the equivalent-sinusoid amplitude of a narrow-band Gaussian), NOT a
  per-record sigma_n,max (M22 uses the record max, which is record-length /
  seed dependent). Spectral methods work in RMS amplitudes; the record-max is a
  time-domain object. MEAN-STRESS corrections beyond the per-plane normal and the
  basic M20/M21 Goodman intercept (Gerber / Soderberg / Walker) remain DEFERRED.
* The spectral Findley uses the LINEAR-COMBINATION (spectral-invariant) form
  g tau + k sigma_n (a single linear projection whose moments come straight from
  M_n, folding in the shear/normal phase), which differs from M22's ADDITIVE
  record-max form g*range + 2 k sigma_n,max — the spectral form is the natural
  frequency-domain object; both reduce to tau_a + k sigma_n,a in amplitude.
* SINGLE scalar random input process (the M19-M22 assumption); a full MULTI-INPUT
  cross-PSD with coherence is DEFERRED. STATIONARY, GAUSSIAN response only; the
  REAL-mode FRF (classical damping). Non-stationary / evolutionary-PSD,
  non-Gaussian (kurtosis) corrections, the complex-FRF stress recovery and
  CRACK-GROWTH / fracture-mechanics fatigue are DEFERRED (the M20/M21/M22 tail).
* NON-PROPORTIONAL HARDENING as a material model is NOT modelled; F_np is reported
  and drives the amplitude correction, but the port's S-N curve is the
  proportional one (the standard high-cycle-fatigue assumption — Socie & Marquis
  note the hardening matters mainly in LOW-cycle fatigue).
"""

from __future__ import annotations

import math

import numpy as np

# reuse the M21 plane / moment machinery read-only (the natural consumer
# relationship — M23 is the SPECTRAL sibling of M22, both built on M21)
from .multiaxial_fatigue import (_inplane_basis, candidate_normals,
                                 normal_projection, shear_projection)


# ============================================================================
# The 2x2 in-plane shear cross-spectral moment matrix + F_np (build item 1)
# ============================================================================

def inplane_shear_covariance(M0, n):
    """The 2x2 IN-PLANE SHEAR cross-spectral moment matrix Sigma_tau (theory
    eq. (1)) on the plane of unit normal ``n``, from the 6x6 spectral-covariance
    matrix ``M0`` = M_0 = E[sigma sigma^T] (the n = 0 tensor moment matrix,
    ``multiaxial_fatigue.tensor_moment_matrices`` output row 0).

    With the M21 in-plane orthonormal basis (a, b) = ``_inplane_basis(n)`` and
    the shear projections p_a = ``shear_projection(n, a)``, p_b =
    ``shear_projection(n, b)``, returns ``(Sigma_tau, (a, b))`` with

        Sigma_tau = [[p_a^T M0 p_a, p_a^T M0 p_b],
                     [p_b^T M0 p_a, p_b^T M0 p_b]]

    — the COVARIANCE of the 2-D resolved-shear path (tau_a, tau_b). This is the
    SPECTRAL image of the M22 time-domain path covariance ``np.cov`` of
    ``resolved_shear_path``: because M_0 = E[sigma sigma^T] is exactly the
    covariance the M21 multivariate synthesiser reproduces, the two 2x2 matrices
    agree (up to the seeded synthesis scatter) — the M23<->M22 identity, computed
    here with NO synthesised history."""
    a, b = _inplane_basis(n)
    pa = shear_projection(n, a)
    pb = shear_projection(n, b)
    M0 = np.asarray(M0)
    c11 = float(pa @ M0 @ pa)
    c22 = float(pb @ M0 @ pb)
    c12 = float(pa @ M0 @ pb)
    return np.array([[c11, c12], [c12, c22]]), (a, b)


def spectral_nonproportionality_factor(M0, n):
    """The SPECTRAL non-proportionality factor F_np of the plane of normal ``n``
    (theory eq. (2)): F_np = sqrt(lambda_2 / lambda_1) of the 2x2 in-plane shear
    cross-spectral moment matrix (``inplane_shear_covariance``), lambda_1 >=
    lambda_2 the principal shear variances. F_np = 0 for a proportional (rank-1)
    state and F_np = 1 for an equal-amplitude 90-deg-out-of-phase (circular)
    state; basis-independent. EQUALS the M22 TIME-DOMAIN F_np on the same
    cross-PSD (the identity M_0 = E[sigma sigma^T]) — computed here with no
    synthesis, straight from the moment matrix."""
    Sig, _ = inplane_shear_covariance(M0, n)
    w = np.linalg.eigvalsh(Sig)                     # ascending, real (symmetric)
    lam1 = float(max(w[-1], 0.0))
    lam2 = float(max(w[0], 0.0))
    if lam1 <= 0.0:
        return 0.0
    return math.sqrt(min(lam2 / lam1, 1.0))


def max_shear_plane_nonproportionality(M0, naz=24, npol=13):
    """The F_np of the MAX-SHEAR CRITICAL PLANE under THIS module's (M23)
    critical-plane convention — the plane maximising the NON-PROPORTIONAL
    EFFECTIVE shear variance tau_a^2 = (1 + F_np^2) lambda_1 = lambda_1 +
    lambda_2 = trace(Sigma_tau) (theory eqs. (3)-(4); exactly the
    ``spectral_critical_plane_damage`` model="shear_path" search parameter).
    Returns ``(F_np, normal)``.

    This is the plane whose F_np the M23 search itself reports, and it is the
    WELL-POSED one: the M21 ``critical_plane_search(method="shear")`` plane
    (argmax of lambda_1 ALONE) is F_np-BLIND — for a uniaxial-dominated state
    its maximum is DEGENERATE (a one-parameter family of 45-deg planes) and the
    first-found member generically carries an EMPTY minor shear axis, so
    evaluating F_np there reports 0 even for a genuinely non-proportional
    (independent bending + torsion) state. Maximising the trace instead folds
    the minor axis in, resolving the degeneracy toward the damage-relevant
    (Susmel-Tovo effective-amplitude) member. The scan needs NO per-plane
    eigendecomposition (the trace is basis-independent); only the winning
    plane's 2x2 block is diagonalised."""
    M0 = np.asarray(M0)
    best_tr = -1.0
    best_n = None
    for n in candidate_normals(naz, npol):
        a, b = _inplane_basis(n)
        pa = shear_projection(n, a)
        pb = shear_projection(n, b)
        tr = float(pa @ M0 @ pa) + float(pb @ M0 @ pb)
        if tr > best_tr:
            best_tr = tr
            best_n = n.copy()
    if best_n is None or best_tr <= 0.0:
        return 0.0, best_n
    return spectral_nonproportionality_factor(M0, best_n), best_n


# ============================================================================
# Per-plane spectral statistics (build item 1)
# ============================================================================

def _plane_spectral_stats(Mmats, n, peak_factor=math.sqrt(2.0)):
    """The per-plane SPECTRAL statistics for the critical-plane search AND the
    damage estimate, computed ONCE per plane from the 6x6 moment-matrix stack
    ``Mmats`` (``tensor_moment_matrices``, shape (nmom, 6, 6)):

      * the 2x2 in-plane shear covariance Sigma_tau, its eigenvalues (lambda_1
        >= lambda_2 the principal shear variances) and F_np = sqrt(lambda_2/
        lambda_1) (theory eqs. (1), (2));
      * the DOMINANT resolved-shear direction (leading eigenvector of Sigma_tau
        — the M21 max-shear direction) and its projection p_dom, plus the
        dominant-shear PSD MOMENTS m_n^tau = p_dom^T M_n p_dom (theory: the
        resolved-shear spectral amplitudes/rates from p^T M_n p);
      * the resolved-NORMAL projection p_n and its moments m_n^sigma = p_n^T M_n
        p_n; the normal-stress AMPLITUDE sigma_n,a = psf * sqrt(m_0^sigma) (the
        spectral peak-stress stand-in for M22's per-record sigma_n,max);
      * the non-proportional EFFECTIVE shear amplitude tau_a = sqrt((1 + F_np^2)
        lambda_1) = sqrt(lambda_1 + lambda_2) (theory eqs. (3)-(4)), the Susmel-
        Tovo stress ratio rho = sigma_n,a / tau_a.

    Returns a dict shared by every model (Findley / Fatemi-Socie / shear-path)
    so the plane search does no per-model estimator evaluation."""
    Mmats = np.asarray(Mmats)
    M0 = Mmats[0]
    Sig, (a, b) = inplane_shear_covariance(M0, n)
    w, V = np.linalg.eigh(Sig)                      # ascending eigenvalues
    lam1 = float(max(w[-1], 0.0))
    lam2 = float(max(w[0], 0.0))
    Fnp = math.sqrt(min(lam2 / lam1, 1.0)) if lam1 > 0.0 else 0.0
    ca, cb = V[:, -1]                               # dominant shear direction
    pa = shear_projection(n, a)
    pb = shear_projection(n, b)
    p_dom = ca * pa + cb * pb
    p_n = normal_projection(n)
    nmom = Mmats.shape[0]
    shear_mom = np.array([float(p_dom @ Mmats[i] @ p_dom) for i in range(nmom)])
    normal_mom = np.array([float(p_n @ Mmats[i] @ p_n) for i in range(nmom)])
    gfac2 = 1.0 + Fnp * Fnp                         # g^2 = lambda_eff/lambda_1
    tau_a = math.sqrt(max(gfac2 * shear_mom[0], 0.0))     # effective shear amp
    sigma_n_a = peak_factor * math.sqrt(max(normal_mom[0], 0.0))
    rho = sigma_n_a / tau_a if tau_a > 1e-300 else 0.0
    return {"F_np": Fnp, "lambda1": lam1, "lambda2": lam2, "gfac2": gfac2,
            "p_dom": p_dom, "p_n": p_n, "shear_mom": shear_mom,
            "normal_mom": normal_mom, "tau_a": tau_a, "sigma_n_a": sigma_n_a,
            "rho": rho, "direction": ca * a + cb * b, "in_plane_basis": (a, b)}


# ============================================================================
# Spectral non-proportional moments per model (build item 2)
# ============================================================================

def modified_wohler_ratio(normal_m0, shear_m0, peak_factor=math.sqrt(2.0)):
    """The Susmel-Lazzarin MODIFIED WOHLER CURVE stress ratio rho = sigma_a /
    tau_a from the resolved-normal variance ``normal_m0`` (m_0^sigma) and the
    (effective) resolved-shear variance ``shear_m0`` (m_0^tau or lambda_eff): the
    ratio of the normal-stress amplitude sigma_a = psf sqrt(m_0^sigma) to the
    shear amplitude tau_a = sqrt(shear_m0). rho indexes the modified Wohler curve
    in Susmel's method (rho = 0 pure shear .. rho large tension-dominated) and is
    reported per plane. A pure RMS-amplitude ratio (Susmel & Lazzarin 2002)."""
    tau = math.sqrt(max(shear_m0, 0.0))
    if tau <= 1e-300:
        return 0.0
    return peak_factor * math.sqrt(max(normal_m0, 0.0)) / tau


def spectral_plane_moments(Mmats, n, model="findley", k=0.3, sigma_y=1.0,
                           peak_factor=math.sqrt(2.0), stats=None):
    """The SPECTRAL equivalent-stress MOMENT array [mu_0..mu_nmax] on the plane
    of normal ``n`` for the requested critical-plane ``model``, ready for the M20
    estimators — computed DIRECTLY from the moment matrices ``Mmats``, no
    synthesised history (theory "THE NON-PROPORTIONAL SPECTRAL CORRECTION" +
    "MODIFIED WOHLER CURVE / CRITICAL-PLANE MODELS"):

      * ``"shear_path"``  : mu_n = (1 + F_np^2) m_n^tau — the pure non-proportional
        shear moments (eq. (5)); reduces to the M21 max-shear moments for F_np = 0;
      * ``"findley"``     : mu_n = p_F^T M_n p_F with p_F = g p_dom + k p_n, g =
        sqrt(1 + F_np^2) — the spectral-invariant linear combination g tau + k
        sigma_n (folds in the shear/normal phase correlation);
      * ``"fatemi_socie"``: mu_n = (1 + k sigma_n,a/sigma_y)^2 (1 + F_np^2) m_n^tau
        — the shear moments amplified by the crack-opening normal stress.

    ``stats`` optionally supplies a precomputed ``_plane_spectral_stats`` dict so
    the search shares the single per-plane eigendecomposition. Returns the moment
    array (nmom,)."""
    if stats is None:
        stats = _plane_spectral_stats(Mmats, n, peak_factor)
    Mmats = np.asarray(Mmats)
    gfac2 = stats["gfac2"]
    if model == "shear_path":
        return gfac2 * stats["shear_mom"]
    if model == "findley":
        g = math.sqrt(gfac2)
        p_F = g * stats["p_dom"] + k * stats["p_n"]
        nmom = Mmats.shape[0]
        return np.array([float(p_F @ Mmats[i] @ p_F) for i in range(nmom)])
    if model == "fatemi_socie":
        fac = 1.0 + k * stats["sigma_n_a"] / sigma_y if sigma_y > 0 else 1.0
        return fac * fac * gfac2 * stats["shear_mom"]
    raise ValueError(f"unknown spectral critical-plane model '{model}' "
                     "(use 'findley', 'fatemi_socie' or 'shear_path').")


def spectral_plane_parameter(stats, model="findley", k=0.3, sigma_y=1.0):
    """The amplitude-only CRITERION PARAMETER the critical plane MAXIMISES
    (theory "MODIFIED WOHLER CURVE / CRITICAL-PLANE MODELS"), from a per-plane
    ``_plane_spectral_stats`` dict — the SPECTRAL, RMS-amplitude image of the
    textbook Findley / Fatemi-Socie critical-plane definitions, so the plane scan
    costs no estimator evaluation:
      * findley      : tau_a + k sigma_n,a
      * fatemi_socie : tau_a (1 + k sigma_n,a/sigma_y)
      * shear_path   : tau_a
    with tau_a = sqrt((1 + F_np^2) lambda_1) the non-proportional effective shear
    amplitude and sigma_n,a = psf sqrt(m_0^sigma) the normal-stress amplitude."""
    tau_a = stats["tau_a"]
    sn_a = stats["sigma_n_a"]
    if model == "findley":
        return tau_a + k * sn_a
    if model == "fatemi_socie":
        return tau_a * (1.0 + k * sn_a / sigma_y if sigma_y > 0 else 1.0)
    if model == "shear_path":
        return tau_a
    raise ValueError(f"unknown spectral critical-plane model '{model}' "
                     "(use 'findley', 'fatemi_socie' or 'shear_path').")


# ============================================================================
# Spectral critical-plane damage (build item 2)
# ============================================================================

def spectral_critical_plane_damage(Mmats, m, C, model="findley", k=0.3,
                                   sigma_y=1.0, mean_stress=0.0, ultimate=0.0,
                                   peak_factor=math.sqrt(2.0), naz=24, npol=13,
                                   normals=None, plane_stats=None):
    """Search the candidate planes for the CRITICAL plane under a SPECTRAL
    critical-plane ``model`` (Findley / Fatemi-Socie / shear-path) and evaluate
    the M20 estimators on its non-proportional equivalent-stress moments — all
    from the 6x6 moment-matrix stack ``Mmats`` (``tensor_moment_matrices``), with
    NO synthesised history.

    The search maximises the model's amplitude-only CRITERION PARAMETER
    (``spectral_plane_parameter``), so the O(n_planes) scan does no estimator
    evaluation; only the winning plane's moments (``spectral_plane_moments``) are
    run through ``spectral_fatigue.fatigue_summary`` (narrow-band / Dirlik /
    Wirsching-Light / Tovo-Benasciutti for N = C S^-m). Returns a dict {normal,
    F_np, rho, tau_a, sigma_n_a, moments, summary, damage_rate (Dirlik), life,
    s_eq, all_param, normals, ...}; ``damage_rate``/``life``/``s_eq`` mirror the
    Dirlik estimator (the wide-band industry standard, matching how M21 ranks).

    ``plane_stats`` optionally supplies a precomputed list of
    ``_plane_spectral_stats`` (one per candidate normal, in ``normals`` order) so
    several models SHARE the single per-plane eigendecomposition."""
    from . import spectral_fatigue as sf

    Mmats = np.asarray(Mmats)
    if normals is None:
        normals = candidate_normals(naz, npol)
    if plane_stats is None:
        plane_stats = [_plane_spectral_stats(Mmats, nn, peak_factor)
                       for nn in normals]
    params = np.array([spectral_plane_parameter(s, model, k, sigma_y)
                       for s in plane_stats])
    icrit = int(np.argmax(params))
    st = plane_stats[icrit]
    nb = normals[icrit].copy()

    moments = spectral_plane_moments(Mmats, nb, model=model, k=k,
                                     sigma_y=sigma_y, peak_factor=peak_factor,
                                     stats=st)
    summ = sf.fatigue_summary(moments, m, C, mean_stress, ultimate)
    dk = summ["dirlik"]
    return {"normal": nb, "in_plane_basis": st["in_plane_basis"],
            "direction": st["direction"], "F_np": st["F_np"], "rho": st["rho"],
            "tau_a": st["tau_a"], "sigma_n_a": st["sigma_n_a"],
            "lambda1": st["lambda1"], "lambda2": st["lambda2"],
            "moments": moments, "summary": summ,
            "damage_rate": dk["damage_rate"], "life": dk["life"],
            "s_eq": dk["s_eq"], "all_param": params, "normals": normals,
            "model": model, "k": k, "sigma_y": sigma_y, "icrit": icrit}


# ============================================================================
# The full spectral non-proportional summary for ONE element — driver API
# ============================================================================

def spectral_nonproportional_summary(Mmats, m, C, k=0.3, sigma_y=1.0,
                                     mean_stress=0.0, ultimate=0.0,
                                     peak_factor=math.sqrt(2.0), naz=24,
                                     npol=13, models=("findley", "fatemi_socie",
                                                      "shear_path")):
    """Evaluate the SPECTRAL non-proportional critical-plane damage of one
    element from its 6x6 stress-tensor spectral-MOMENT matrices ``Mmats`` (the
    M21 ``tensor_moment_matrices`` output — NO synthesised history), searching the
    critical plane for each requested ``model`` (Findley / Fatemi-Socie /
    shear-path) and running the M20 estimators on the non-proportional
    equivalent-stress moments.

    Returns a nested dict: one entry per model (the
    ``spectral_critical_plane_damage`` result — critical plane, F_np, rho, damage
    rate, life) plus the shear-amplitude / F_np / rho summary on the Findley
    critical plane. This is the driver's per-element spectral evaluation
    (``random_response._run_spectral_nonproportional``); it is the SPECTRAL
    (no-history) analogue of the M22 ``nonproportional_summary`` time-domain path
    count, reported ALONGSIDE it and the M21 spectral answer so a listing shows
    all three side by side."""
    Mmats = np.asarray(Mmats)
    normals = candidate_normals(naz, npol)
    # the single per-plane eigendecomposition, SHARED by every model
    plane_stats = [_plane_spectral_stats(Mmats, nn, peak_factor)
                   for nn in normals]

    out = {"method": "spectral", "peak_factor": float(peak_factor),
           "k": k, "sigma_y": sigma_y}
    for mdl in models:
        out[mdl] = spectral_critical_plane_damage(
            Mmats, m, C, model=mdl, k=k, sigma_y=sigma_y,
            mean_stress=mean_stress, ultimate=ultimate, peak_factor=peak_factor,
            normals=normals, plane_stats=plane_stats)

    # the shear-amplitude / F_np / rho summary on the Findley critical plane (the
    # spectral analogue of M22's MCC/chord/MRH amplitude comparison — here the
    # amplitudes are RMS/variance measures, no hull)
    crit = out.get("findley", out[models[0]])
    ncrit = crit["normal"]
    st = _plane_spectral_stats(Mmats, ncrit, peak_factor)
    lam1, lam2 = st["lambda1"], st["lambda2"]
    out["amplitudes"] = {
        # dominant-shear RMS (M21 max-shear amplitude, sqrt lambda_1), the
        # non-proportional effective RMS shear (sqrt(lambda_1+lambda_2)), and the
        # path factor g = MRH/scalar = sqrt(1 + F_np^2)
        "shear_rms_dominant": math.sqrt(max(lam1, 0.0)),
        "shear_rms_effective": math.sqrt(max(lam1 + lam2, 0.0)),
        "g": math.sqrt(1.0 + st["F_np"] ** 2),
        "F_np": st["F_np"], "rho": st["rho"],
        "sigma_n_a": st["sigma_n_a"],
    }
    return out

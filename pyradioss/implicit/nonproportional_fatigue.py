"""
Non-proportional multiaxial fatigue — M22: the CRITICAL-PLANE, TIME-DOMAIN
PATH-COUNTING fatigue-damage estimate of a stationary random-vibration response
for a MULTIAXIAL stress STATE whose principal axes ROTATE in time (a
non-proportional load path). Where M21 reduces the stress-tensor cross-PSD to a
scalar EQUIVALENT-stress PSD and counts damage on a PROJECTED scalar (exact only
for proportional loading — the standard spectral approximation otherwise), M22
keeps the full 2-D resolved-shear PATH on each candidate plane and counts the
damage that a rotating shear vector actually accrues.

Fortran origin
--------------
There is NONE. ``engine/source/input/freimpl.F`` (the /IMPL reader, re-read line
by line for M16-M21 AND AGAIN for M22) parses only /IMPL/DYNA (the DIRECT
Newmark/HHT integrator), /IMPL/BUCKL, /IMPL/DT, /IMPL/NONLIN and /IMPL/ARCL plus
the linear-solver housekeeping — there is no /FATIG, no S-N / Miner branch, no
von-Mises / critical-plane / stress-tensor cross-PSD machinery, and NO
non-proportional path-counting / minimum-circumscribed-circle / Findley /
Fatemi-Socie criterion of any kind. The sole ``PSD`` token in the whole file is
still ``IMUMPSD`` (line 269), a MUMPS-solver flag, not a power spectral density.
OpenRadioss is a time-domain crash/impact code: the stationary random-vibration
MULTIAXIAL fatigue analysis — proportional (M21) OR non-proportional (M22) — is
simply not part of the open-source solver, exactly as M16 found for the real
eigensolver, M17/M18 for the transfer functions, M19 for the PSD machinery, M20
for the scalar spectral fatigue and M21 for the multiaxial spectral fatigue this
module extends.

So M22 does exactly what M16-M21 did: it ports non-proportional multiaxial
fatigue as a clean LIBRARY capability and drives it with a minimal PORT engine
sub-flag (/IMPL/FATIG/MULT/NPROP — the non-proportional analogue of the M21
/IMPL/FATIG/MULT multiaxial card). Nothing in the M10 direct integrator, the M16
REAL eigensolver, the M17/M18 superposition, the M19 PSD path, the M20 SCALAR
fatigue or the M21 MULTIAXIAL SPECTRAL path is touched: the non-proportional
path-counting is a NEW, parallel path that CONSUMES the M21 multivariate
synthesiser (``multiaxial_fatigue.synthesize_multiaxial_history``) and the
candidate-plane machinery (``candidate_normals`` / ``_inplane_basis`` /
``normal_projection`` / ``shear_projection``) read-only.

Theory — non-proportional multiaxial fatigue (critical-plane path counting)
---------------------------------------------------------------------------
(Papadopoulos, "Critical plane approaches in high-cycle fatigue: on the
definition of the amplitude and mean value of the shear stress acting on the
critical plane", Fatigue Fract. Engng Mater. Struct. 21, 1998 — the minimum
circumscribed circle; Mamiya, Araujo & Castro, "Prismatic hull: a new measure of
shear stress amplitude in multiaxial high-cycle fatigue", Int. J. Fatigue 31,
2009 — the maximum rectangular / prismatic hull; Findley, "A theory for the
effect of mean stress on fatigue of metals under combined torsion and axial load
or bending", J. Eng. Ind. 81, 1959 — tau_a + k sigma_n,max; Fatemi & Socie, "A
critical plane approach to multiaxial fatigue damage including out-of-phase
loading", Fatigue Fract. Engng Mater. Struct. 11, 1988 — gamma_a(1 + k
sigma_n,max/sigma_y); Matake 1977; Carpinteri & Spagnoli, Int. J. Fatigue 23,
2001; Itoh, Sakane, Ohnami & Socie, "Nonproportional low cycle fatigue criterion
for type 304 stainless steel", J. Eng. Mater. Technol. 117, 1995 — the
non-proportionality factor F_np; Socie & Marquis, "Multiaxial Fatigue", SAE 2000,
ch. 2-4 — the rotating-principal-axes / non-proportional-hardening background.)

THE ROTATING SHEAR PATH. On a candidate material plane of unit normal n, with two
orthonormal in-plane axes (a, b) (``_inplane_basis``), the resolved shear stress
is the 2-D VECTOR

    tau_vec(t) = ( tau_a(t), tau_b(t) ),
    tau_a(t) = p_s(n,a)^T sigma_voigt(t),  tau_b(t) = p_s(n,b)^T sigma_voigt(t)

(the M21 ``shear_projection`` applied along each in-plane axis of a synthesised
stress-component history), and the resolved normal stress is the SCALAR
sigma_n(t) = p_n(n)^T sigma_voigt(t) (``normal_projection``). For PROPORTIONAL
loading tau_vec(t) traces a straight LINE (the principal axes are fixed and the
shear direction never changes); for NON-PROPORTIONAL loading it traces a 2-D
CURVE (an ellipse for a 90-deg-out-of-phase sinusoid, a general locus otherwise)
— the rotating principal axes. A scalar projection of the shear (M21's max-shear
plane) sees only ONE in-plane direction and UNDER-counts the damage of a rotating
path; the whole point of a non-proportional criterion is to measure the
amplitude of the 2-D path itself.

SHEAR-AMPLITUDE OPERATORS. The "amplitude" of a closed 2-D shear path Psi = {
tau_vec(t) } is defined three standard ways (all reducing to the scalar amplitude
for a line, all closed-form for a circle):
* MINIMUM CIRCUMSCRIBED CIRCLE (MCC, Papadopoulos 1998): the RADIUS of the
  smallest circle enclosing Psi. tau_a = R_mcc. For a line of half-length A,
  R_mcc = A (the M21 scalar amplitude); for a circle of radius r (equal-amplitude
  90-deg-out-of-phase), R_mcc = r EXACTLY.
* LONGEST CHORD / longest projection (Grubisic & Simburger): the maximum distance
  between two points of Psi (its DIAMETER); the amplitude is half the chord. The
  diameter equals the maximum over all directions of the projection RANGE (the
  longest-projection method), so "longest chord" and "longest projection" name the
  same number. For a line, chord = 2A (amplitude A); for a circle, chord = 2r =
  the DIAMETER (amplitude r).
* MAXIMUM RECTANGULAR HULL (MRH / prismatic hull, Mamiya-Araujo-Castro 2009): the
  maximum over in-plane rotations theta of sqrt(a1(theta)^2 + a2(theta)^2), with
  a1, a2 the half-widths of the axis-aligned bounding box of Psi in the rotated
  frame. For a line, MRH = A (only one box side is nonzero, at the aligned angle);
  for a circle, MRH = r sqrt(2) — LARGER than the radius, because the MRH is the
  measure that FEELS the enclosed area of a rotating path. The MRH is the default
  amplitude for the damage models below precisely because it distinguishes a
  circle (r sqrt2) from a line (A = r): the MCC and the longest chord do NOT (a
  circle of radius r and a line of half-length r give the same R_mcc = r and the
  same amplitude r), so a critical-plane model built on the MCC alone is blind to
  the extra damage of a rotating path unless the NORMAL-stress term or the plane
  search supplies it. This is a documented, deliberate choice (Mamiya et al. make
  exactly this argument for the prismatic hull).

NON-PROPORTIONALITY FACTOR. The degree of non-proportionality of the shear path
is measured (Itoh-Kanazawa; the shear-path aspect-ratio form) by

    F_np = sqrt( lambda_2 / lambda_1 ) ,   lambda_1 >= lambda_2 >= 0            (*)

the eigenvalues of the 2x2 COVARIANCE of the mean-removed shear path (tau_a,
tau_b). F_np = 0 for a straight LINE (rank-1 covariance, lambda_2 = 0 — perfectly
proportional) and F_np = 1 for a CIRCLE (isotropic covariance, lambda_1 =
lambda_2 — maximally non-proportional). It is exactly the ratio of the minor to
the major principal axis of the path's inertia ellipse — a clean, closed-form
stand-in for Itoh's rotation-angle integral that shares its endpoints (0 for a
proportional line, 1 for a circle) and is invariant to the in-plane basis choice.
F_np is REPORTED per critical plane and is available as a non-proportional
HARDENING multiplier for the Carpinteri-Spagnoli-style correction (documented
below), but the primary damage measure — the MRH shear amplitude — already
carries the rotating-path effect, so F_np is not double-counted into the default
damage.

CRITICAL-PLANE DAMAGE MODELS. On each candidate plane the record is reduced to a
scalar EQUIVALENT-stress cycle history and Miner-summed against the S-N curve
N = C S^-m. The scalar carries the MAX NORMAL STRESS on the plane, sigma_n,max =
max_t sigma_n(t), which FOLDS IN the per-plane MEAN normal stress (a piece of the
M20/M21 mean-stress deferral — the plane's own mean, not a global Goodman
intercept):
* FINDLEY (1959): the equivalent shear amplitude on the plane is tau_a + k
  sigma_n,max (k the Findley normal-sensitivity constant, ~0.2-0.3 for steels).
  The plane of MAXIMUM (tau_a + k sigma_n,max) is the Findley critical plane. The
  damage counts the rainflowed shear cycles at the plane's (non-proportional) MRH
  amplitude with the constant normal-stress boost 2 k sigma_n,max on each range.
* FATEMI-SOCIE (1988): the parameter is gamma_a (1 + k sigma_n,max / sigma_y)
  with gamma_a = tau_a / G the shear-strain amplitude, k the FS constant and
  sigma_y the yield stress. In this stress-based port the G cancels against an
  S-N curve expressed in stress, leaving the equivalent stress range tau_range (1
  + k sigma_n,max/sigma_y) per cycle — a shear range AMPLIFIED by the normal
  stress that opens the crack faces (the mechanism FS captures that Findley's
  additive form approximates). FS is the standard model for ductile,
  shear-dominated non-proportional fatigue.
(MATAKE is Findley with k on the plane of maximum shear rather than maximum
Findley parameter; CARPINTERI-SPAGNOLI weights the normal and shear damage on the
weighted-mean-principal plane with an F_np correction — both are provided as
options of the same machinery, documented at the functions.)

TIME-DOMAIN PATH COUNTING. Over the M21-synthesised CORRELATED Gaussian
stress-component histories (the multivariate spectral-representation method,
seeded), for each candidate plane:
  1. resolve the normal history sigma_n(t) and the 2-D shear path (tau_a, tau_b);
  2. measure the shear-path amplitude (MCC / longest chord / MRH) and F_np;
  3. rainflow-count (ASTM E1049, the M20 counter) the DOMINANT resolved shear
     scalar (the max-variance in-plane direction — the M21 max-shear projection)
     to get the cycle COUNT and per-cycle range structure, then SCALE each shear
     range by the non-proportional path factor g = (MRH amplitude)/(dominant
     scalar amplitude) so a rotating path counts its full 2-D amplitude while a
     line counts g = 1 (unchanged);
  4. form the model's equivalent-stress range per cycle (Findley / Fatemi-Socie,
     with sigma_n,max), Miner-sum, divide by the record duration.
The critical plane is the plane of MAXIMUM damage. For PROPORTIONAL loading the
shear path is a line, g = 1, F_np = 0 and (with k = 0, i.e. the pure shear-path
damage) the count reduces EXACTLY to the M21 max-shear critical-plane rainflow —
so the non-proportional path-counting damage reduces to the M21 spectral answer
within the seeded Monte-Carlo scatter (validated). For a 90-deg-OUT-OF-PHASE
biaxial case the MRH shear amplitude is sqrt(2) larger than the projected scalar
and F_np ~ 1, so the non-proportional damage is HIGHER than the proportional case
at the same channel amplitudes — the extra damage the scalar projection and the
spectral method miss (the whole point).

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* LIBRARY-FIRST sub-flag (/IMPL/FATIG/MULT/NPROP) — no upstream equivalent,
  exactly as established for M16-M21's PORT cards.
* The damage is TIME-DOMAIN path counting over a SYNTHESISED history (the M21
  multivariate synthesiser). A purely FREQUENCY-DOMAIN non-proportionality factor
  that avoids a synthesised history (the Cristofori-Susmel-Tovo / Pitoiset
  spectral non-proportionality factor) is DEFERRED — it is a different (spectral)
  estimator, not a path count.
* The default non-proportional shear amplitude for the damage models is the MRH
  (it distinguishes a circle from a line; the MCC does not). MCC and longest
  chord are computed and reported for comparison / the closed-form validations.
* The normal-stress term uses the per-plane MAX normal stress sigma_n,max (which
  folds in that plane's mean). MEAN-STRESS corrections beyond the per-plane normal
  and the basic M20/M21 Goodman intercept (Gerber / Soderberg / Walker) remain
  DEFERRED.
* SINGLE scalar random input process (the M19-M21 assumption); a full MULTI-INPUT
  cross-PSD with coherence is DEFERRED. STATIONARY, GAUSSIAN response only; the
  REAL-mode FRF (classical damping). Non-stationary / evolutionary-PSD,
  non-Gaussian (kurtosis) corrections, the complex-FRF stress recovery and
  CRACK-GROWTH / fracture-mechanics fatigue are DEFERRED (the M20/M21 tail).
* NON-PROPORTIONAL HARDENING as a material model (the extra cyclic hardening a
  rotating path induces in the flow curve itself) is NOT modelled; F_np is
  reported and available as a Carpinteri-Spagnoli damage multiplier, but the
  port's S-N curve is the proportional one (the standard high-cycle-fatigue
  assumption — Socie & Marquis note the hardening matters mainly in LOW-cycle
  fatigue).
"""

from __future__ import annotations

import math

import numpy as np

# reuse the M21 plane machinery read-only (the natural consumer relationship)
from .multiaxial_fatigue import (_inplane_basis, candidate_normals,
                                 normal_projection, shear_projection,
                                 synthesize_multiaxial_history)


# ============================================================================
# Resolved histories on a plane (build-order item 1)
# ============================================================================

def resolved_normal_history(X, n):
    """The resolved NORMAL-stress scalar history sigma_n(t) = p_n^T sigma(t) on
    the plane of unit normal ``n``, from a synthesised Voigt stress-component
    history ``X`` (nt, 6). Returns (nt,)."""
    return np.asarray(X) @ normal_projection(n)


def resolved_shear_path(X, n):
    """The 2-D resolved-SHEAR path (tau_a(t), tau_b(t)) on the plane of unit
    normal ``n``, in the in-plane orthonormal basis (a, b) = ``_inplane_basis(n)``
    — the M21 in-plane basis and shear projection lifted to a time history.

    Returns ``(P, (a, b))`` with ``P`` of shape (nt, 2), column 0 = tau along a,
    column 1 = tau along b. For PROPORTIONAL loading the columns are perfectly
    correlated (P traces a line); for NON-PROPORTIONAL loading they carry a phase
    lag (P traces a 2-D curve). This 2-D locus is the object every shear-amplitude
    operator below measures."""
    a, b = _inplane_basis(n)
    ta = np.asarray(X) @ shear_projection(n, a)
    tb = np.asarray(X) @ shear_projection(n, b)
    return np.column_stack([ta, tb]), (a, b)


# ============================================================================
# 2-D geometry helpers (convex hull, used by every amplitude operator)
# ============================================================================

def _convex_hull(P):
    """The convex hull of a 2-D point set ``P`` (n, 2). Returns the hull VERTICES
    as an (h, 2) array. Every shear-amplitude operator (MCC, longest chord, MRH)
    is realised on hull vertices, so reducing to the hull first turns the O(nt)
    synthesised path into a handful of points — the operators then cost nothing.

    Uses SciPy's C-level ``ConvexHull`` when available (the fatigue path already
    requires SciPy — ``random_response.require_scipy``); the O(nt) synthesised
    paths make a pure-Python monotone chain far too slow. Falls back to Andrew's
    monotone chain (numpy) when SciPy is absent or the set is degenerate
    (collinear / < 3 points), where ConvexHull raises."""
    pts = np.asarray(P, dtype=float)
    if pts.shape[0] <= 2:
        return np.unique(pts, axis=0)
    try:
        from scipy.spatial import ConvexHull
        # ConvexHull handles duplicate points itself — skipping the O(nt log nt)
        # np.unique(axis=0) here is the single biggest speed-up on the O(nt)
        # synthesised paths (measured ~7x on a 1e6-point path)
        hull = ConvexHull(pts)
        return pts[hull.vertices]
    except Exception:
        # degenerate (collinear) or SciPy absent -> monotone chain fallback
        return _monotone_chain(np.unique(pts, axis=0))


def _monotone_chain(pts):
    """Andrew's monotone-chain convex hull (numpy fallback for _convex_hull):
    the O(n log n) hull used when SciPy's ConvexHull is unavailable or refuses a
    degenerate (collinear) set."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    return hull if hull.shape[0] else pts


def _circle_from_2(a, b):
    c = 0.5 * (a + b)
    return c, float(np.linalg.norm(a - c))


def _circle_from_3(a, b, c):
    """The circumscribed circle of three points (the unique circle through them);
    returns (center, radius) or (None, inf) if collinear."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-300:
        return None, math.inf
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / d
    center = np.array([ux, uy])
    return center, float(np.linalg.norm(a - center))


def min_circumscribed_circle(P, hull=None):
    """The MINIMUM CIRCUMSCRIBED CIRCLE (Papadopoulos 1998) of a 2-D shear path
    ``P`` (n, 2): the smallest circle enclosing every point, by Welzl's algorithm
    (expected linear) run on the convex hull. Returns ``(center, radius)``; the
    radius is the MCC shear amplitude. For a straight LINE of half-length A the
    MCC radius is A (the M21 scalar amplitude); for a CIRCLE of radius r it is r
    EXACTLY (both asserted in the M22 validation). ``hull`` may be a precomputed
    convex hull (``_convex_hull(P)``) to avoid recomputing it."""
    if hull is None:
        hull = _convex_hull(P)
    if hull.shape[0] == 0:
        return np.zeros(2), 0.0
    if hull.shape[0] == 1:
        return hull[0].copy(), 0.0
    # Welzl (iterative, move-to-front on the hull points — small set)
    pts = hull.copy()
    # a deterministic shuffle (no RNG in this module — reproducibility) by a
    # fixed permutation stride so the expected-linear behaviour still holds
    idx = np.arange(pts.shape[0])
    if pts.shape[0] > 3:
        idx = (idx * 2654435761) % pts.shape[0]
        _, keep = np.unique(idx, return_index=True)
        order = np.argsort(idx)
        # fall back to identity if the stride collides (small sets)
        pts = pts[order] if len(set(idx.tolist())) == pts.shape[0] else pts

    c = np.zeros(2)
    r = -1.0
    n = pts.shape[0]
    for i in range(n):
        if r >= 0 and np.linalg.norm(pts[i] - c) <= r + 1e-12:
            continue
        c, r = pts[i].copy(), 0.0
        for j in range(i):
            if np.linalg.norm(pts[j] - c) <= r + 1e-12:
                continue
            c, r = _circle_from_2(pts[i], pts[j])
            for k in range(j):
                if np.linalg.norm(pts[k] - c) <= r + 1e-12:
                    continue
                cc, rr = _circle_from_3(pts[i], pts[j], pts[k])
                if cc is not None:
                    c, r = cc, rr
    return c, max(r, 0.0)


def longest_chord(P, hull=None):
    """The LONGEST CHORD (the DIAMETER) of a 2-D shear path ``P`` (n, 2): the
    maximum distance between any two points, computed on the convex hull (the
    diameter is always realised by two hull vertices). Returns the chord LENGTH;
    the shear amplitude is half of it. The diameter equals the maximum over all
    in-plane directions of the projection RANGE (so this is also the
    longest-PROJECTION amplitude). For a CIRCLE of radius r the longest chord is
    2 r (the diameter), amplitude r; for a line of half-length A it is 2 A.
    ``hull`` may be a precomputed convex hull to avoid recomputing it."""
    if hull is None:
        hull = _convex_hull(P)
    if hull.shape[0] <= 1:
        return 0.0
    # brute force over hull vertices (few) — clearer than rotating calipers and
    # the hull is tiny after the reduction
    d2 = 0.0
    h = hull
    for i in range(h.shape[0]):
        dd = np.sum((h - h[i]) ** 2, axis=1)
        d2 = max(d2, float(dd.max()))
    return math.sqrt(d2)


def max_rectangular_hull(P, ntheta=90, hull=None):
    """The MAXIMUM RECTANGULAR / PRISMATIC HULL amplitude (Mamiya-Araujo-Castro
    2009) of a 2-D shear path ``P`` (n, 2):

        MRH = max_theta sqrt( a1(theta)^2 + a2(theta)^2 )

    with a1, a2 the half-widths of the axis-aligned bounding box of ``P`` in the
    frame rotated by ``theta`` (scanned over [0, pi/2), ``ntheta`` steps). Run on
    the convex hull. For a LINE of half-length A the MRH is A (one box side
    vanishes at the aligned angle); for a CIRCLE of radius r it is r sqrt(2)
    (every rotated bounding box is the 2r-square, a1 = a2 = r) — the measure that
    distinguishes a rotating path from a line, which is why the damage models use
    it by default (module docstring). ``hull`` may be a precomputed convex hull to
    avoid recomputing it."""
    if hull is None:
        hull = _convex_hull(P)
    if hull.shape[0] <= 1:
        return 0.0
    thetas = np.linspace(0.0, math.pi / 2.0, int(ntheta), endpoint=False)
    best = 0.0
    for th in thetas:
        c, s = math.cos(th), math.sin(th)
        u1 = hull[:, 0] * c + hull[:, 1] * s          # projection on axis 1
        u2 = -hull[:, 0] * s + hull[:, 1] * c         # projection on axis 2
        a1 = 0.5 * (u1.max() - u1.min())
        a2 = 0.5 * (u2.max() - u2.min())
        best = max(best, math.hypot(a1, a2))
    return best


def shear_amplitude(P, method="mrh", ntheta=90, hull=None):
    """The shear-path amplitude of ``P`` (n, 2) by the requested ``method``:
    ``"mcc"`` (minimum circumscribed circle radius, Papadopoulos), ``"chord"`` /
    ``"lp"`` (half the longest chord = the longest projection), or ``"mrh"``
    (maximum rectangular hull, Mamiya-Araujo — the default). All three reduce to
    the scalar amplitude for a proportional line; they DIFFER for a rotating path
    (a circle of radius r: MCC = r, chord/2 = r, MRH = r sqrt2). ``hull`` may be
    a precomputed convex hull to avoid recomputing it."""
    if hull is None:
        hull = _convex_hull(P)
    if method == "mcc":
        _, r = min_circumscribed_circle(P, hull=hull)
        return r
    if method in ("chord", "lp", "longest_chord", "longest_projection"):
        return 0.5 * longest_chord(P, hull=hull)
    if method == "mrh":
        return max_rectangular_hull(P, ntheta=ntheta, hull=hull)
    raise ValueError(f"unknown shear-amplitude method '{method}' "
                     "(use 'mcc', 'chord' or 'mrh').")


# ============================================================================
# Non-proportionality factor (build-order item 1)
# ============================================================================

def nonproportionality_factor(P, cov=None):
    """The non-proportionality factor F_np of a 2-D shear path ``P`` (n, 2)
    (theory eq. (*)): F_np = sqrt(lambda_2 / lambda_1) with lambda_1 >= lambda_2
    the eigenvalues of the mean-removed 2x2 path covariance — the ratio of the
    minor to the major principal axis of the path's inertia ellipse. F_np = 0 for
    a straight LINE (proportional; the covariance is rank-1) and F_np = 1 for a
    CIRCLE (maximally non-proportional; isotropic covariance). Basis-independent.
    The shear-path aspect-ratio form of the Itoh-Kanazawa factor (module
    docstring). ``cov`` may be a precomputed 2x2 path covariance."""
    Pc = np.asarray(P, dtype=float)
    if Pc.shape[0] < 2:
        return 0.0
    if cov is None:
        cov = np.cov(Pc.T)
    if not np.all(np.isfinite(cov)):
        return 0.0
    w = np.linalg.eigvalsh(cov)                    # ascending, real (symmetric)
    lam1 = float(max(w[-1], 0.0))
    lam2 = float(max(w[0], 0.0))
    if lam1 <= 0.0:
        return 0.0
    return math.sqrt(min(lam2 / lam1, 1.0))


# ============================================================================
# Critical-plane time-domain damage (build-order item 2)
# ============================================================================

def _dominant_shear_scalar(P, cov=None):
    """The DOMINANT resolved-shear scalar tau_dom(t): the projection of the 2-D
    shear path onto its maximum-variance in-plane direction (the leading
    eigenvector of the path covariance) — exactly the M21 max-shear plane's
    resolved scalar. Rainflowing THIS carries the cycle count and temporal range
    structure; the non-proportional path factor g then scales its amplitude to the
    full 2-D (MRH) amplitude. ``cov`` may be a precomputed 2x2 path covariance.
    Returns (tau_dom (nt,), scalar_amplitude)."""
    Pc = np.asarray(P, dtype=float)
    if cov is None:
        cov = np.cov(Pc.T)
    w, V = np.linalg.eigh(cov)
    d = V[:, -1]                                   # max-variance direction
    tau = Pc @ d
    amp = 0.5 * (tau.max() - tau.min())            # half peak-to-peak (line amp)
    return tau, float(amp)


def plane_parameter(tau_a, sn_max, model="findley", k=0.3, sigma_y=1.0):
    """The critical-plane CRITERION PARAMETER on a plane, from its shear-path
    amplitude ``tau_a`` and max normal stress ``sn_max`` (theory
    "CRITICAL-PLANE DAMAGE MODELS"). This is the quantity the critical plane
    MAXIMISES — the textbook definition of each model's critical plane (Findley:
    max of tau_a + k sigma_n,max; Fatemi-Socie: max of tau_a(1 + k
    sigma_n,max/sigma_y)) — and it is amplitude-only, so the plane search costs
    no rainflow (the damage is then counted on the winning plane alone):
      * findley      : tau_a + k sigma_n,max
      * fatemi_socie : tau_a (1 + k sigma_n,max/sigma_y)
      * shear_path   : tau_a  (pure shear amplitude, no normal term)."""
    if model == "findley":
        return tau_a + k * sn_max
    if model == "fatemi_socie":
        return tau_a * (1.0 + k * sn_max / sigma_y if sigma_y > 0 else 1.0)
    if model == "shear_path":
        return tau_a
    raise ValueError(f"unknown critical-plane model '{model}' "
                     "(use 'findley', 'fatemi_socie' or 'shear_path').")


def plane_damage(P, sigma_n, m, C, model="findley", k=0.3, sigma_y=1.0,
                 amp_method="mrh", duration=None, stats=None):
    """The TIME-DOMAIN critical-plane damage on ONE plane from its 2-D shear path
    ``P`` (nt, 2) and resolved normal history ``sigma_n`` (nt,), for the S-N curve
    N = C S^-m under a Miner sum.

    Steps (theory "TIME-DOMAIN PATH COUNTING"):
      1. sigma_n,max = max_t sigma_n(t) (folds in the plane's mean normal stress);
      2. the non-proportional shear amplitude tau_a = shear_amplitude(P,
         amp_method) (default MRH) and the dominant scalar amplitude a_dom;
      3. g = tau_a / a_dom, the non-proportional path factor (1 for a line);
      4. rainflow (ASTM E1049, the M20 counter) the dominant shear scalar ->
         (ranges, counts); each shear range is scaled by g;
      5. the model's equivalent-stress range per cycle, Miner-summed / duration.

    ``model``:
      * ``"findley"``   : S_i = g*range_i + 2 k sigma_n,max  (tau_a + k sigma_n,max
                          in range form; Findley 1959);
      * ``"fatemi_socie"``: S_i = g*range_i (1 + k sigma_n,max/sigma_y)  (the FS
                          shear range amplified by the crack-opening normal
                          stress; the shear modulus G cancels in the stress-based
                          port, module docstring);
      * ``"shear_path"`` : S_i = g*range_i  (the pure non-proportional shear-path
                          damage, no normal term — the model that reduces EXACTLY
                          to the M21 max-shear rainflow for a proportional line at
                          g = 1; used for the reduction validation).

    ``stats`` optionally supplies a precomputed ``_plane_stats`` result (tau_a,
    sn_max, tau_dom, a_dom, F_np) so the search does not recompute the shear
    amplitude. Returns a dict {damage, damage_rate, ncycles, sn_max, tau_a, g,
    F_np, ...}."""
    from . import spectral_fatigue as sf

    if stats is None:
        stats = _plane_stats(P, sigma_n, amp_method)
    sn_max = stats["sn_max"]
    tau_a = stats["tau_a"]
    a_dom = stats["a_dom"]
    tau_dom = stats["tau_dom"]
    g = tau_a / a_dom if a_dom > 1e-300 else 1.0
    Fnp = stats["F_np"]

    ranges, counts = sf.rainflow_count(tau_dom)
    if ranges.size == 0:
        return {"damage": 0.0, "damage_rate": 0.0, "ncycles": 0.0,
                "sn_max": sn_max, "tau_a": tau_a, "g": g, "F_np": Fnp,
                "model": model}

    sr = g * ranges                                # non-proportional shear ranges
    if model == "findley":
        S = sr + 2.0 * k * sn_max
    elif model == "fatemi_socie":
        S = sr * (1.0 + k * sn_max / sigma_y if sigma_y > 0 else 1.0)
    elif model == "shear_path":
        S = sr
    else:
        raise ValueError(f"unknown critical-plane model '{model}' "
                         "(use 'findley', 'fatemi_socie' or 'shear_path').")
    S = np.clip(S, 0.0, None)
    D = float(np.sum(counts * S ** m) / C)
    T = duration if (duration and duration > 0) else 1.0
    return {"damage": D, "damage_rate": D / T, "ncycles": float(counts.sum()),
            "sn_max": sn_max, "tau_a": tau_a, "g": g, "F_np": Fnp,
            "model": model, "ranges": sr, "counts": counts}


def _plane_stats(P, sigma_n, amp_method="mrh"):
    """The per-plane statistics needed by the critical-plane search AND the
    damage count, computed ONCE per plane (the expensive part is the shear-path
    amplitude's convex hull): the non-proportional shear amplitude ``tau_a``, the
    max normal stress ``sn_max`` (folds in the plane mean), the dominant shear
    scalar ``tau_dom`` + its amplitude ``a_dom`` (for the rainflow), and the
    non-proportionality factor ``F_np``. Shared so the plane search does no
    rainflow and the winning plane is rainflowed once."""
    sn = np.asarray(sigma_n, dtype=float)
    sn_max = float(sn.max()) if sn.size else 0.0
    # compute the convex hull and the 2x2 path covariance ONCE, then derive the
    # shear amplitude, the dominant scalar and F_np from them (the hull is the
    # single expensive per-plane operation)
    hull = _convex_hull(P)
    Pc = np.asarray(P, dtype=float)
    Pm = Pc - Pc.mean(axis=0, keepdims=True)
    cov = (Pm.T @ Pm) / max(Pc.shape[0] - 1, 1)    # matmul cov (far faster)
    tau_a = shear_amplitude(P, amp_method, hull=hull)
    tau_dom, a_dom = _dominant_shear_scalar(P, cov=cov)
    Fnp = nonproportionality_factor(P, cov=cov)
    return {"tau_a": tau_a, "sn_max": sn_max, "tau_dom": tau_dom,
            "a_dom": a_dom, "F_np": Fnp}


def critical_plane_damage(X, m, C, model="findley", k=0.3, sigma_y=1.0,
                          amp_method="mrh", duration=None, naz=24, npol=13,
                          normals=None, plane_stats=None):
    """Search the candidate planes for the CRITICAL plane under a critical-plane
    ``model`` (Findley / Fatemi-Socie / shear-path), from the synthesised Voigt
    stress-component history ``X`` (nt, 6), then count the TIME-DOMAIN damage on
    that plane.

    The search maximises the model's CRITERION PARAMETER (``plane_parameter`` —
    tau_a + k sigma_n,max for Findley, etc.: the textbook definition of the
    critical plane, amplitude-only), so the O(n_planes) scan does NO rainflow —
    only the winning plane is rainflowed (``plane_damage``). For constant
    amplitude the parameter-max plane IS the maximum-damage plane; for a random
    record it is the standard, far cheaper critical-plane definition (a full
    per-plane rainflow-damage search is O(n_planes) rainflows over the whole
    synthesised history — prohibitive).

    ``plane_stats`` optionally supplies a precomputed list of ``_plane_stats``
    (one per candidate normal, in ``normals`` order) so several models SHARE the
    single expensive amplitude scan. Returns a dict {normal, in_plane_basis,
    damage, damage_rate, life, sn_max, tau_a, g, F_np, all_param (per-plane),
    normals, ...}. ``life`` = 1 / damage_rate (Miner: failure at D = 1)."""
    Xa = np.asarray(X)
    if normals is None:
        normals = candidate_normals(naz, npol)
    # per-plane amplitude scan (shared across models when plane_stats is given)
    if plane_stats is None:
        plane_stats = []
        for n in normals:
            P, _ = resolved_shear_path(Xa, n)
            sn = resolved_normal_history(Xa, n)
            plane_stats.append(_plane_stats(P, sn, amp_method))
    params = np.array([plane_parameter(s["tau_a"], s["sn_max"], model, k,
                                       sigma_y) for s in plane_stats])
    icrit = int(np.argmax(params))
    nb = normals[icrit].copy()
    P, basis = resolved_shear_path(Xa, nb)
    sn = resolved_normal_history(Xa, nb)
    d = plane_damage(P, sn, m, C, model=model, k=k, sigma_y=sigma_y,
                     amp_method=amp_method, duration=duration,
                     stats=plane_stats[icrit])
    dr = d["damage_rate"]
    life = math.inf if dr <= 0.0 else 1.0 / dr
    out = dict(d)
    out.update({"normal": nb, "in_plane_basis": basis, "life": life,
                "all_param": params, "normals": normals, "amp_method":
                amp_method, "k": k, "sigma_y": sigma_y, "icrit": icrit})
    return out


# ============================================================================
# The full non-proportional summary for ONE element (all models) — driver API
# ============================================================================

def nonproportional_summary(freqs_hz, Scross, m, C, duration, seed,
                            k=0.3, sigma_y=1.0, amp_method="mrh", fs=None,
                            naz=24, npol=13, models=("findley", "fatemi_socie",
                                                     "shear_path")):
    """Evaluate the NON-PROPORTIONAL critical-plane path-counting damage of one
    element from its stress-tensor cross-PSD ``Scross`` (nf, 6, 6) — the M21
    matrix — by synthesising the CORRELATED Gaussian stress-component histories
    once (``synthesize_multiaxial_history``, seeded) and searching the critical
    plane for each requested ``model`` (Findley / Fatemi-Socie / shear-path).

    Returns a nested dict: one entry per model (the ``critical_plane_damage``
    result — critical plane, damage rate, life, sigma_n,max, tau_a, F_np) plus the
    shared synthesised RMS-per-component and the amplitude comparison (MCC vs
    chord vs MRH) on the Findley critical plane. This is the driver's per-element
    evaluation (``random_response._run_nonproportional``)."""
    t, X = synthesize_multiaxial_history(freqs_hz, Scross, duration, seed, fs=fs)
    T = float(t[-1] - t[0]) if t.size > 1 else float(duration)
    normals = candidate_normals(naz, npol)

    # the single expensive amplitude scan over all candidate planes — SHARED by
    # every model (they differ only in how tau_a and sigma_n,max combine into the
    # criterion parameter, which is cheap)
    plane_stats = []
    for n in normals:
        P, _ = resolved_shear_path(X, n)
        sn = resolved_normal_history(X, n)
        plane_stats.append(_plane_stats(P, sn, amp_method))

    out = {"duration": T, "seed": int(seed), "amp_method": amp_method,
           "k": k, "sigma_y": sigma_y, "rms_components": np.std(X, axis=0)}
    for mdl in models:
        out[mdl] = critical_plane_damage(
            X, m, C, model=mdl, k=k, sigma_y=sigma_y, amp_method=amp_method,
            duration=T, normals=normals, plane_stats=plane_stats)

    # amplitude comparison (MCC / chord / MRH) on the Findley critical plane
    ncrit = out.get("findley", out[models[0]])["normal"]
    P, _ = resolved_shear_path(X, ncrit)
    out["amplitudes"] = {
        "mcc": shear_amplitude(P, "mcc"),
        "chord": shear_amplitude(P, "chord"),
        "mrh": shear_amplitude(P, "mrh"),
        "F_np": nonproportionality_factor(P),
    }
    return out

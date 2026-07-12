"""
Multiaxial / critical-plane spectral fatigue — M21: the frequency-domain
fatigue-damage estimate of a stationary random-vibration response for a
MULTIAXIAL stress STATE (a full stress tensor), not just a single scalar
channel. Built DIRECTLY on top of M20: it reduces the 6x6 stress-tensor
cross-PSD to a scalar EQUIVALENT-stress PSD and then runs the M20 spectral
estimators (narrow-band / Dirlik / Wirsching-Light / Tovo-Benasciutti) on its
moments.

Fortran origin
--------------
There is NO frequency-domain / spectral-fatigue solver of ANY kind (scalar or
multiaxial) anywhere in the open-source OpenRadioss engine. ``engine/source/
input/freimpl.F`` (the /IMPL reader, re-read line by line for M16-M20 AND
AGAIN for M21) parses only /IMPL/DYNA (the DIRECT Newmark/HHT integrator),
/IMPL/BUCKL, /IMPL/DT, /IMPL/NONLIN and /IMPL/ARCL plus the linear-solver
housekeeping — there is no /FATIG, no S-N / Miner branch, no von-Mises /
critical-plane multiaxial-fatigue machinery, and (as every milestone since M19
has confirmed) the sole ``PSD`` token in the whole file is ``IMUMPSD``, a
MUMPS-solver flag, not a power spectral density. OpenRadioss is a time-domain
crash/impact code: the stationary random-vibration multiaxial fatigue analysis
is simply not part of the open-source solver — exactly the finding M16 made for
the real eigensolver, M17/M18 for the transfer functions, M19 for the PSD
machinery and M20 for the scalar-channel spectral fatigue this module extends.

So M21 does exactly what M16-M20 did: it ports multiaxial spectral fatigue as a
clean LIBRARY capability and drives it with a minimal PORT engine sub-card
(/IMPL/FATIG/MULT — the multiaxial analogue of the M20 /IMPL/FATIG scalar
card). Nothing in the M10 direct integrator, the M16 REAL eigensolver, the
M17/M18 superposition, the M19 PSD path or the M20 SCALAR-channel fatigue path
is touched: the multiaxial path is a NEW, parallel path that CONSUMES the M20
vector stress modes (the full 6-component Voigt stress FRF per element)
read-only.

Theory — multiaxial random (spectral) fatigue
---------------------------------------------
(Preumont & Piefort, "Predicting random high-cycle fatigue life with finite
elements", J. Sound Vib. 168, 1994 — the equivalent-von-Mises frequency-domain
projection; Pitoiset & Preumont, "Spectral methods for multiaxial random
fatigue analysis of metallic structures", Int. J. Fatigue 22, 2000 — the
trace(Q S) form and the critical-plane comparison; Carpinteri & Spagnoli,
"Multiaxial high-cycle fatigue criterion for hard metals", Int. J. Fatigue 23,
2001 — the maximum-principal / weighted-mean-principal critical-plane
direction; Cristofori, Susmel & Tovo, "A stress-invariant based spectral
method to estimate fatigue life under multiaxial random loading", Int. J.
Fatigue 30, 2008; Benasciutti, Sherratt & Cristofori — the projection-by-
direction spectral form; Socie & Marquis, "Multiaxial Fatigue", SAE 2000, for
the time-domain critical-plane background the spectral methods approximate.)

The M20 stress recovery already gives, per solid/shell element, the full
6-component Voigt stress FRF

    H_sigma(Omega) = [ H_xx, H_yy, H_zz, H_xy, H_yz, H_zx ](Omega)   (nf, 6)

(the stress modes sigma_i = StressOp . phi_i combined with the SAME modal
coordinates q_i(Omega) the displacement FRF uses — random_response.stress_frf).
For a SINGLE scalar random input process f(t) of PSD S_ff(Omega) (a force
pattern or one base-acceleration direction, the M19/M20 assumption), the
stress-tensor CROSS-PSD is the rank-1 Hermitian matrix

    S_sigmasigma(Omega) = H_sigma(Omega) S_ff(Omega) H_sigma(Omega)^H  (nf,6,6)

(the |H|^2 S law of eq. (1) written for the VECTOR transfer function — Newland;
Pitoiset & Preumont 2000 eq. 3). Its diagonal entries S[c,c] = |H_c|^2 S_ff are
exactly the M20 per-component scalar channel PSDs; the off-diagonals carry the
cross-spectra (the phase relationship between the components) a scalar channel
throws away. The whole game of a MULTIAXIAL spectral method is to reduce this
6x6 matrix to ONE scalar EQUIVALENT-stress PSD whose spectral moments the M20
estimators consume.

EQUIVALENT VON MISES (Preumont & Piefort 1994; Pitoiset & Preumont 2000). The
von Mises equivalent stress is a QUADRATIC form of the stress tensor,

    sigma_vm^2 = sigma^T Q sigma ,   sigma = [xx yy zz xy yz zx]^T        (2)

with the symmetric 6x6 von Mises operator

          [  1  -1/2 -1/2  0  0  0 ]
          [-1/2   1  -1/2  0  0  0 ]
    Q  =  [-1/2 -1/2   1   0  0  0 ]
          [  0    0    0   3  0  0 ]
          [  0    0    0   0  3  0 ]
          [  0    0    0   0  0  3 ]

so sigma_vm^2 = xx^2+yy^2+zz^2 - xx yy - yy zz - zz xx + 3(xy^2+yz^2+zx^2), the
textbook von Mises invariant. Because expectation is linear and Q constant, the
mean-square equivalent stress is E[sigma_vm^2] = E[sigma^T Q sigma] = trace(Q
E[sigma sigma^T]) = trace(Q Psi) with Psi = int S_sigmasigma dOmega the
covariance matrix. Extending this PER FREQUENCY defines the EQUIVALENT VON MISES
PSD (Pitoiset & Preumont 2000 eq. 8)

    S_vm(Omega) = trace( Q S_sigmasigma(Omega) )                          (3)

a REAL scalar PSD (Q symmetric real, S Hermitian => trace real) whose 0th moment
is exactly E[sigma_vm^2]. For a rank-1 cross-PSD (single input) (3) collapses to
the cheap quadratic scalar

    S_vm(Omega) = ( H_sigma^H Q H_sigma )(Omega) . S_ff(Omega)            (4)

— no 6x6 matrix needed at run time (``equivalent_vonmises_psd``). For a
UNIAXIAL state (only H_xx nonzero) (3)/(4) reduce to Q_xx,xx |H_xx|^2 S_ff =
|H_xx|^2 S_ff = the M20 scalar sigma_xx channel PSD — asserted in the M21
validation. The M20 estimators then run on S_vm's moments, giving the
multiaxial damage rate / equivalent stress / life.

CRITICAL PLANE (Carpinteri-Spagnoli; Cristofori-Susmel-Tovo). The alternative
family projects the stress tensor onto a searched set of candidate material
PLANES and counts damage on the plane that maximises a resolved scalar stress.
For a plane of unit normal n the NORMAL stress is the LINEAR projection

    sigma_n(t) = n^T sigma(t) n = p_n^T sigma_voigt(t) ,
    p_n = [nx^2, ny^2, nz^2, 2 nx ny, 2 ny nz, 2 nz nx]                   (5)

and the resolved SHEAR stress in an in-plane direction m (m . n = 0) is

    tau(t) = m^T sigma(t) n = p_s^T sigma_voigt(t) ,
    p_s = [mx nx, my ny, mz nz, mx ny+my nx, my nz+mz ny, mx nz+mz nx]    (6)

Both projections being LINEAR, the projected scalar's PSD is the rank-1 quadratic

    S_p(Omega) = p^T S_sigmasigma(Omega) p = |H_sigma(Omega) . p|^2 S_ff  (7)

(a real scalar — S Hermitian, p real, so the imaginary part p^T Im(S) p = 0),
and its moments are p^T M_n p with M_n = (1/pi) int Omega^n S_sigmasigma dOmega
the 6x6 spectral-moment matrices. The MAX-NORMAL-STRESS critical plane searches
n for the maximum normal-stress variance (m0 = p_n^T M_0 p_n); the
MAX-SHEAR-STRESS critical plane searches n and, on each plane, takes the
in-plane direction of maximum shear variance (the leading eigenvector of the 2x2
in-plane covariance). The M20 estimators then run on the critical plane's
scalar moments (``critical_plane_search``). For a PURE-SHEAR state sigma_xy(t)
the max-normal plane is at 45deg (normal in the xy-plane at 45deg, where the
resolved normal stress equals the shear amplitude — the principal plane), and
the max-shear plane is a coordinate plane (normal along x or y, where the
resolved shear equals the shear amplitude) — both recovered in the M21
validation. (Textbook: in pure shear the principal axes are at 45deg to x, and
the max-shear planes bisect the principal axes.)

MONTE-CARLO multiaxial cross-check. As an independent time-domain validation
the module SYNTHESISES the CORRELATED Gaussian stress-component histories from
the 6x6 cross-PSD (the multivariate spectral-representation method: a per-bin
eigendecomposition of S_sigmasigma(Omega) — a general Cholesky-type factor that
handles the rank-1 single-input case gracefully — with independent random
phases, then an inverse real FFT per component, the M20 synthesiser generalised
to a vector process), PROJECTS them onto the critical plane's scalar stress
(the linear projection of eq. (5)/(6), so the projected history is exactly
Gaussian with PSD (7)), RAINFLOW-counts it (the M20 ASTM E1049 counter) and
Miner-sums — the time-domain damage the spectral critical-plane estimate
approximates. Seeded (a fixed seed argument) for reproducibility. The check is
run on the LINEAR critical-plane scalar (not the quadratic von Mises, whose
rainflow of a non-Gaussian signal a Gaussian-PDF estimator would not match —
see the deferrals).

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* LIBRARY-FIRST sub-card (/IMPL/FATIG/MULT) — no upstream equivalent, exactly as
  established for M16-M20's PORT cards.
* SINGLE scalar random input process (a force pattern OR one base-acceleration
  direction), so the cross-PSD is the rank-1 H S_ff H^H of eq. (1). A full
  MULTI-INPUT cross-PSD with coherence (off-diagonal input cross-spectra) is
  DEFERRED (the same M19/M20 deferral — it needs the H S_ff H^H matrix triple
  product with a non-diagonal S_ff and a coherence model).
* PROJECTED-EQUIVALENT multiaxial methods only (equivalent von Mises + normal /
  shear critical plane). NON-PROPORTIONAL cycle counting beyond a projected
  scalar — the full tensor rainflow / minimum-circumscribed-circle (Papadopoulos)
  shear-amplitude path counting, and the rotating-principal-axes non-proportional
  hardening correction — is DEFERRED.
* The MONTE-CARLO cross-check validates the LINEAR critical-plane projection.
  A von-Mises (quadratic, hence non-Gaussian) time-domain rainflow cross-check
  is DEFERRED (the spectral von-Mises PDF is itself an approximation — Pitoiset
  & Preumont 2000 discuss the bias).
* MEAN STRESS: the basic M20 Goodman intercept option is carried through
  (applied to the equivalent scalar); Gerber / Soderberg / Walker and a
  per-plane mean remain DEFERRED (M20).
* STATIONARY, GAUSSIAN response only; the REAL-mode FRF (classical damping).
  Non-stationary / evolutionary-PSD fatigue, non-Gaussian (kurtosis) corrections
  and the complex-FRF stress recovery are DEFERRED (M20 tail).
* CRACK-GROWTH / fracture-mechanics fatigue is a different analysis entirely and
  is DEFERRED.
"""

from __future__ import annotations

import math

import numpy as np


# ============================================================================
# The von Mises quadratic operator Q (build-order item 2, equivalent stress)
# ============================================================================

def von_mises_operator():
    """The symmetric 6x6 von Mises quadratic operator Q of theory eq. (2), in
    the Voigt order [xx, yy, zz, xy, yz, zx] (the port's ``sig`` convention):

        sigma_vm^2 = sigma^T Q sigma
                   = xx^2+yy^2+zz^2 - xx yy - yy zz - zz xx
                     + 3 (xy^2 + yz^2 + zx^2)

    so the normal-normal block is (I - 1/2 (J - I)) with off-diagonal -1/2 and
    the shear block is 3 I (engineering shear — the factor 3 = 2*(3/2), the
    deviatoric-norm weight on each shear component). Constant, cached by the
    caller."""
    Q = np.zeros((6, 6))
    # normal-stress 3x3 block: diagonal 1, off-diagonal -1/2
    Q[:3, :3] = -0.5
    for i in range(3):
        Q[i, i] = 1.0
    # shear block: 3 on the diagonal (xy, yz, zx)
    for i in range(3, 6):
        Q[i, i] = 3.0
    return Q


_Q_VM = von_mises_operator()


# ============================================================================
# The stress-tensor cross-PSD (build-order item 1)
# ============================================================================

def stress_tensor_cross_psd(Hvoigt, input_psd):
    """The 6x6 stress-tensor cross-PSD S_sigmasigma(Omega) = H_sigma S_ff
    H_sigma^H (theory eq. (1)) for ONE element, from its 6-component Voigt
    stress FRF ``Hvoigt`` (nf, 6) complex and the scalar input PSD
    ``input_psd`` (nf,) sampled on the SAME grid.

    Returns a (nf, 6, 6) complex Hermitian array: entry [f, a, b] = H[f,a]
    conj(H[f,b]) S_ff[f]. The DIAGONAL S[f,c,c] = |H[f,c]|^2 S_ff[f] is exactly
    the M20 per-component scalar channel PSD (asserted in the validation); the
    off-diagonals carry the component cross-spectra the scalar path discards.

    For a single scalar input the matrix is RANK ONE (an outer product times a
    scalar) — materialised in full here for validation / reporting / the
    Monte-Carlo synthesis; the damage path uses the cheaper scalar quadratics
    (``equivalent_vonmises_psd`` / ``project_frf``) that never form it."""
    H = np.asarray(Hvoigt)
    Sff = np.asarray(input_psd, dtype=float)
    if H.ndim != 2 or H.shape[1] != 6:
        raise ValueError("stress_tensor_cross_psd needs a (nf, 6) Voigt "
                         f"stress FRF; got shape {H.shape}.")
    if Sff.shape[0] != H.shape[0]:
        raise ValueError(
            f"input PSD has {Sff.shape[0]} samples but the stress FRF sweep "
            f"has {H.shape[0]} — evaluate the PSD on the sweep grid first.")
    # H H^H per frequency (outer product), scaled by the scalar S_ff (eq. (1))
    S = H[:, :, None] * np.conj(H[:, None, :])       # (nf, 6, 6)
    return S * Sff[:, None, None]


def tensor_moment_matrices(omega, Scross, nmax=4):
    """The 6x6 spectral-MOMENT matrices M_n = (1/pi) int_0^inf Omega^n
    S_sigmasigma dOmega, n = 0..nmax (theory below eq. (7)), by the trapezoidal
    rule over the swept angular-frequency grid — the M19/M20 moment convention
    lifted to a matrix. ``Scross`` is (nf, 6, 6). Returns the REAL symmetric part
    as a (nmax+1, 6, 6) array (a real projection p^T M_n p uses only Re(M_n),
    since p^T Im(M_n) p = 0 for the antisymmetric Im of a Hermitian matrix).

    M_0 is the covariance matrix Psi = E[sigma sigma^T]; trace(Q M_0) = E[
    sigma_vm^2] the von Mises variance; p^T M_n p are the moments of any linearly
    projected scalar stress (eq. (7))."""
    omega = np.asarray(omega, dtype=float)
    S = np.asarray(Scross)
    order = np.argsort(omega)
    w = omega[order]
    Sw = S[order]
    out = np.zeros((nmax + 1, 6, 6))
    for n in range(nmax + 1):
        integrand = (w ** n)[:, None, None] * Sw          # (nf, 6, 6)
        # trapezoid over frequency; keep the real (symmetric) part
        out[n] = np.trapezoid(integrand.real, w, axis=0) / np.pi
    return out


# ============================================================================
# Equivalent von Mises PSD (build-order item 2)
# ============================================================================

def equivalent_vonmises_psd(Hvoigt, input_psd):
    """The EQUIVALENT VON MISES PSD S_vm(Omega) = trace(Q S_sigmasigma) (theory
    eq. (3)), evaluated through the cheap rank-1 quadratic form (eq. (4))

        S_vm(Omega) = ( H_sigma^H Q H_sigma )(Omega) . S_ff(Omega)

    from the 6-component Voigt stress FRF ``Hvoigt`` (nf, 6) and the scalar input
    PSD (nf,). Returns S_vm (nf,) REAL. Reduces to the M20 scalar sigma_xx
    channel PSD for a uniaxial state (only H_xx nonzero => Q_xx,xx |H_xx|^2
    S_ff = |H_xx|^2 S_ff). No 6x6 matrix is formed — the whole reduction is one
    (nf,6)x(6,6)x(nf,6) contraction."""
    H = np.asarray(Hvoigt)
    Sff = np.asarray(input_psd, dtype=float)
    # H^H Q H per frequency: conj(H) . Q . H, a real (Hermitian-form) scalar
    quad = np.einsum("fa,ab,fb->f", np.conj(H), _Q_VM, H).real
    quad = np.clip(quad, 0.0, None)          # Q is PSD on the deviator (guard)
    return quad * Sff


def equivalent_vonmises_moments(Mmats):
    """The spectral moments [m_0..m_nmax] of the equivalent von Mises PSD from
    the 6x6 moment matrices ``Mmats`` (tensor_moment_matrices output): m_n^vm =
    trace(Q M_n) (theory eq. (3) integrated). The matrix route (used to CHECK
    the direct ``equivalent_vonmises_psd`` + spectral_moments answer — the
    trace / quadratic-operator identity of the validation)."""
    Mmats = np.asarray(Mmats)
    return np.array([float(np.tensordot(_Q_VM, Mmats[n], axes=([0, 1],
                                                               [1, 0])))
                     for n in range(Mmats.shape[0])])


# ============================================================================
# Linear projections + critical-plane search (build-order item 2)
# ============================================================================

def normal_projection(n):
    """The 6-vector p_n mapping the Voigt stress to the NORMAL stress on a plane
    of unit normal ``n`` (theory eq. (5)): sigma_n = p_n^T sigma_voigt,
    p_n = [nx^2, ny^2, nz^2, 2 nx ny, 2 ny nz, 2 nz nx]."""
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
    return np.array([nx * nx, ny * ny, nz * nz,
                     2.0 * nx * ny, 2.0 * ny * nz, 2.0 * nz * nx])


def shear_projection(n, m):
    """The 6-vector p_s mapping the Voigt stress to the resolved SHEAR stress in
    the in-plane direction ``m`` on a plane of normal ``n`` (theory eq. (6)):
    tau = m^T sigma n = p_s^T sigma_voigt,
    p_s = [mx nx, my ny, mz nz, mx ny+my nx, my nz+mz ny, mx nz+mz nx]."""
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
    mx, my, mz = float(m[0]), float(m[1]), float(m[2])
    return np.array([mx * nx, my * ny, mz * nz,
                     mx * ny + my * nx, my * nz + mz * ny, mx * nz + mz * nx])


def candidate_normals(naz=24, npol=13):
    """A grid of candidate plane NORMALS over the unit sphere: ``npol`` polar
    angles theta in [0, pi] and ``naz`` azimuths phi in [0, 2pi). The defaults
    (naz=24 -> 15deg, npol=13 -> 15deg) land exactly on the equator (theta=90,
    normals in the xy-plane) and on phi=45deg, so the pure-shear 45deg plane is
    in the set. Opposite normals (same physical plane) are de-duplicated by
    keeping the hemisphere with a non-negative leading nonzero component.
    Returns an (K, 3) array of unit normals."""
    normals = []
    seen = set()
    for it in range(npol):
        theta = math.pi * it / (npol - 1)
        st, ct = math.sin(theta), math.cos(theta)
        for ia in range(naz):
            phi = 2.0 * math.pi * ia / naz
            v = np.array([st * math.cos(phi), st * math.sin(phi), ct])
            nrm = np.linalg.norm(v)
            if nrm < 1e-12:
                continue
            v = v / nrm
            # fold opposite normals together (n and -n are the same plane)
            lead = v[np.argmax(np.abs(v) > 1e-9)] if np.any(
                np.abs(v) > 1e-9) else 0.0
            if lead < 0:
                v = -v
            key = tuple(np.round(v, 6))
            if key in seen:
                continue
            seen.add(key)
            normals.append(v)
    return np.asarray(normals)


def _inplane_basis(n):
    """Two orthonormal in-plane directions (u1, u2) spanning the plane of normal
    ``n`` (u1, u2, n right-handed). Picks a reference axis not parallel to n."""
    n = np.asarray(n, dtype=float)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array(
        [0.0, 1.0, 0.0])
    u1 = ref - np.dot(ref, n) * n
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(n, u1)
    return u1, u2


def critical_plane_search(Mmats, method="normal", naz=24, npol=13):
    """Search the candidate planes for the CRITICAL plane under the requested
    resolved-stress ``method`` and return its scalar spectral moments (theory
    eqs. (5)-(7)).

    method = "normal": the plane of maximum NORMAL-stress variance m0 = p_n^T
        M_0 p_n; the reported scalar is sigma_n (eq. (5)).
    method = "shear": the plane of maximum resolved-SHEAR variance; on each
        plane the in-plane direction is the leading eigenvector of the 2x2
        in-plane shear covariance (the direction of maximum shear variance), and
        the reported scalar is tau (eq. (6)).

    ``Mmats`` is the (>=1, 6, 6) moment-matrix stack (tensor_moment_matrices).
    Returns a dict {normal, direction (shear only), proj (the 6-vector p),
    moments (nmom,), m0, all_m0 (per-plane variance), normals}. The full moment
    array feeds the M20 estimators."""
    normals = candidate_normals(naz, npol)
    M0 = Mmats[0]
    best = None
    all_m0 = np.zeros(normals.shape[0])
    for k, n in enumerate(normals):
        if method == "normal":
            p = normal_projection(n)
            m0 = float(p @ M0 @ p)
            direction = None
        elif method == "shear":
            u1, u2 = _inplane_basis(n)
            p1 = shear_projection(n, u1)
            p2 = shear_projection(n, u2)
            # 2x2 in-plane shear covariance; its max eigenvalue is the plane's
            # maximum resolved-shear variance, the eigenvector its direction
            c11 = float(p1 @ M0 @ p1)
            c22 = float(p2 @ M0 @ p2)
            c12 = float(p1 @ M0 @ p2)
            cov = np.array([[c11, c12], [c12, c22]])
            evals, evecs = np.linalg.eigh(cov)
            m0 = float(evals[-1])
            a, b = evecs[:, -1]
            p = a * p1 + b * p2                      # that direction's proj
            direction = a * u1 + b * u2
        else:
            raise ValueError(f"unknown critical-plane method '{method}' "
                             "(use 'normal' or 'shear').")
        all_m0[k] = m0
        if best is None or m0 > best[0]:
            best = (m0, n.copy(), p.copy(), direction)
    m0b, nb, pb, db = best
    moments = np.array([float(pb @ Mmats[n] @ pb)
                        for n in range(Mmats.shape[0])])
    out = {"normal": nb, "proj": pb, "moments": moments, "m0": m0b,
           "all_m0": all_m0, "normals": normals, "method": method}
    if db is not None:
        out["direction"] = db
    return out


def project_frf(Hvoigt, proj):
    """The scalar projected stress FRF H_p(Omega) = H_sigma(Omega) . p (nf,)
    complex, from the 6-component Voigt stress FRF and a projection 6-vector
    (normal_projection / shear_projection). The projected scalar PSD is then
    |H_p|^2 S_ff (theory eq. (7)) — exactly the M20 |H|^2 S form, so the M20
    stress-PSD / moment / estimator machinery consumes it unchanged."""
    return np.asarray(Hvoigt) @ np.asarray(proj, dtype=float)


# ============================================================================
# Monte-Carlo multiaxial cross-check (build-order item 2, validation)
# ============================================================================

def synthesize_multiaxial_history(freqs_hz, Scross, duration, seed, fs=None):
    """Synthesise CORRELATED Gaussian stress-component histories whose
    cross-PSD is ``Scross`` (nf, 6, 6, the M20 two-sided convention, sampled on
    ``freqs_hz`` [Hz]), by the MULTIVARIATE spectral-representation method
    (Shinozuka & Deodatis 1996 — the vector generalisation of the M20 scalar
    synthesiser). Each positive-frequency bin's 6x6 cross-spectral matrix is
    factored S = V diag(lam) V^H (an eigendecomposition — a general Cholesky-type
    factor L = V sqrt(lam) with L L^H = S that handles the RANK-1 single-input
    case gracefully, where a plain Cholesky would fail on the semi-definite
    matrix), each nonzero eigen-direction gets an independent uniform random
    phase, and one inverse real FFT per component sums the harmonics. Returns
    (t, X) with X of shape (nt, 6).

    Convention bridge (identical to the M20 scalar synthesiser): the two-sided
    S(Omega) maps to the one-sided G(f) = 2 S(2 pi f); a positive-frequency
    irfft bin k of magnitude |X_k| contributes variance (2/N^2)|X_k|^2 = G df,
    so the eigen-amplitude is N sqrt(2 lam(f) df / 2) = N sqrt(lam df) per unit
    eigenvector, with lam the one-sided eigenvalue 2*lam_two_sided."""
    rng = np.random.default_rng(int(seed))
    f = np.asarray(freqs_hz, dtype=float)
    S = np.asarray(Scross)
    fmax = float(f.max())
    if fs is None:
        fs = 8.0 * fmax
    nt = int(max(round(duration * fs), 4))
    if nt % 2:
        nt += 1
    df = fs / nt
    fft_f = np.fft.rfftfreq(nt, d=1.0 / fs)
    nbin = fft_f.size
    # interpolate each 6x6 entry onto the FFT grid (zero outside the band), then
    # form the one-sided Hermitian cross-spectral stack G(f) = 2 S(2 pi f)
    Gstack = np.zeros((nbin, 6, 6), dtype=complex)
    for a in range(6):
        for b in range(6):
            re = np.interp(fft_f, f, S[:, a, b].real, left=0.0, right=0.0)
            im = np.interp(fft_f, f, S[:, a, b].imag, left=0.0, right=0.0)
            Gstack[:, a, b] = 2.0 * (re + 1j * im)
    # BATCHED Hermitian eigendecomposition (np.linalg.eigh vectorizes over the
    # leading axis): the general Cholesky-type factor of every bin at once —
    # far faster than a per-bin Python loop, and it handles the rank-1
    # single-input semi-definite matrices gracefully
    lam, V = np.linalg.eigh(Gstack)               # lam (nbin,6), V (nbin,6,6)
    lam = np.clip(lam.real, 0.0, None)
    amp = nt * np.sqrt(lam * df / 2.0)            # (nbin, 6) per eigen-direction
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(nbin, 6))
    coeff = amp * np.exp(1j * phase)             # (nbin, 6)
    Xspec = np.einsum("kab,kb->ka", V, coeff)     # (nbin, 6) component spectra
    Xspec[0] = 0.0                                # no DC (a PSD carries no mean)
    if nt % 2 == 0:
        Xspec[-1] = Xspec[-1].real                # Nyquist bin real
    X = np.fft.irfft(Xspec, n=nt, axis=0)          # (nt, 6)
    t = np.arange(nt) / fs
    return t, X


def monte_carlo_multiaxial_damage(freqs_hz, Scross, proj, m, C, duration, seed,
                                  fs=None, mean_stress=0.0, ultimate=0.0):
    """The TIME-DOMAIN multiaxial damage rate by Monte-Carlo: synthesise the 6
    correlated Gaussian stress-component histories from the cross-PSD, PROJECT
    onto the critical plane's LINEAR scalar (``proj`` = a normal_/shear_
    projection 6-vector), rainflow-count (ASTM E1049, the M20 counter) and
    Miner-sum. The independent time-domain answer the spectral critical-plane
    estimate approximates; seeded for reproducibility. Returns the M20
    Monte-Carlo dict shape plus ``rms`` of the projected scalar."""
    from . import spectral_fatigue as sf
    t, X = synthesize_multiaxial_history(freqs_hz, Scross, duration, seed,
                                         fs=fs)
    proj = np.asarray(proj, dtype=float)
    s = X @ proj                                   # the projected scalar history
    Ceff = sf._goodman_C(C, m, mean_stress, ultimate)
    ranges, counts = sf.rainflow_count(s)
    D = float(np.sum(counts * ranges ** m) / Ceff) if ranges.size else 0.0
    T = t[-1] - t[0] if t.size > 1 else duration
    dr = D / T if T > 0 else 0.0
    tf, s_eq = sf.life_and_equivalent(
        dr, ranges.size / T if T > 0 else 0.0, m, Ceff)
    return {"method": "monte_carlo_multiaxial", "damage_rate": dr, "life": tf,
            "s_eq": s_eq, "ncycles": float(counts.sum()), "ranges": ranges,
            "counts": counts, "duration": T, "rms": float(np.std(s))}


# ============================================================================
# The full multiaxial fatigue summary for ONE element (all reductions)
# ============================================================================

def multiaxial_fatigue_summary(Hvoigt, input_psd, omega, m, C,
                               mean_stress=0.0, ultimate=0.0, naz=24, npol=13):
    """Evaluate ALL multiaxial reductions on ONE element's 6-component Voigt
    stress FRF ``Hvoigt`` (nf, 6) under the scalar input PSD ``input_psd`` (nf,)
    on the angular grid ``omega`` (nf,): the equivalent VON MISES PSD, the
    MAX-NORMAL-stress critical plane and the MAX-SHEAR-stress critical plane —
    each reduced to a scalar PSD, its M20 spectral moments and the four M20
    damage estimators (narrow-band / Dirlik / Wirsching-Light / Tovo-Benasciutti)
    for the S-N curve N = C S^-m. Returns a nested dict; the cross-PSD 6x6 moment
    matrices are included for reporting / the trace identity."""
    from . import spectral_fatigue as sf
    from .random_response import spectral_moments

    Sff = np.clip(np.asarray(input_psd, dtype=float), 0.0, None)
    omega = np.asarray(omega, dtype=float)

    # --- equivalent von Mises (cheap rank-1 quadratic; eq. (4)) -------------
    Svm = equivalent_vonmises_psd(Hvoigt, Sff)
    mom_vm = spectral_moments(omega, Svm, nmax=4)
    vm = {"psd": Svm, "moments": mom_vm,
          "summary": sf.fatigue_summary(mom_vm, m, C, mean_stress, ultimate)}

    # --- critical planes: need the 6x6 moment matrices (eq. (7)) ------------
    Scross = stress_tensor_cross_psd(Hvoigt, Sff)
    Mmats = tensor_moment_matrices(omega, Scross, nmax=4)

    def _plane(method):
        cp = critical_plane_search(Mmats, method=method, naz=naz, npol=npol)
        cp["summary"] = sf.fatigue_summary(cp["moments"], m, C, mean_stress,
                                           ultimate)
        return cp

    normal_cp = _plane("normal")
    shear_cp = _plane("shear")

    return {"von_mises": vm, "normal_plane": normal_cp,
            "shear_plane": shear_cp, "Mmats": Mmats, "Scross": Scross,
            "omega": omega}

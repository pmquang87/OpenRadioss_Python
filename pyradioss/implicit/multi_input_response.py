"""
Multi-input / partially-coherent random-vibration response — M28: the
stationary response (and stress-tensor) cross-PSD driven by SEVERAL
simultaneous random input processes with a full Hermitian input
cross-spectral matrix S_ff(Omega), propagated through the VECTOR FRF by the
MIMO relation S_uu = H S_ff H^H and S_sigmasigma = H_sigma S_ff H_sigma^H,
then reduced by the whole M20-M27 estimator family UNCHANGED.

Where M19/M20/M21 assumed ONE scalar random input process f(t) of PSD
S_ff(Omega) (a single force pattern or one base-acceleration direction) and
formed the RANK-1 Hermitian cross-PSD H S_ff H^H, M28 generalises S_ff to a
NON-DIAGONAL ninput x ninput Hermitian matrix — auto-PSDs on the diagonal,
coherence gamma_ab and phase theta_ab off the diagonal — and contracts it with
a per-input FRF COLUMN stack H(Omega) (nf, ndof, ninput), one column per input
load pattern. The M19/M20/M21 single-input answer is EXACTLY the 1x1
(diagonal / rank-1) special case of this matrix path (asserted, and delegated
to the M21 rank-1 routines so it is bit-identical).

Fortran origin
--------------
There is NO frequency-domain / spectral / random-vibration solver of ANY kind
in the open-source OpenRadioss engine — no PSD path, no cross-spectral /
coherence matrix, no MIMO response machinery. ``engine/source/implicit/`` is a
time-domain crash/impact code (re-checked for M28: ``imp_solv.F`` /
``freform.F`` carry no ``coherence`` / ``cross_psd`` / multi-input token, and
the sole ``PSD`` token anywhere is ``IMUMPSD``, a MUMPS-solver flag — exactly
the finding M16-M27 recorded line by line). So M28, like every spectral
milestone since M16, ports multi-input coherent random vibration as a clean
LIBRARY capability and drives it with minimal PORT engine sub-flags
(/IMPL/PSD/MULTI and /IMPL/FATIG/MINPUT — the multi-input analogues of the M19
/IMPL/PSD and M20/M21 /IMPL/FATIG cards). Nothing in the M10 integrator, the
M16 eigensolver, the M17/M18 superposition, the M19 PSD path or the M20-M27
fatigue reductions is touched: the multi-input path is a NEW, parallel path
that CONSUMES the M17/M19 modal FRF and the M20 stress modes read-only.

Theory — multiple correlated inputs and the coherence matrix
------------------------------------------------------------
(Newland, "An Introduction to Random Vibrations, Spectral & Wavelet Analysis",
3rd ed., ch. 6-8 — multiple correlated inputs, the coherence function and the
matrix input/output spectral relation; Bendat & Piersol, "Random Data",
4th ed., ch. 5-7 — the cross-spectral density / coherence matrix and the
MULTIPLE-INPUT/MULTIPLE-OUTPUT (MIMO) relation S_yy = H S_xx H^H; Wirsching,
Paez & Ortiz, "Random Vibrations: Theory and Practice", 1995 — multi-input
random fatigue and the spectral-representation synthesis of correlated
processes; the M19/M20/M21 single-input base this module generalises.)

For ninput jointly-stationary random input processes f_a(t), a = 1..ninput,
the INPUT CROSS-SPECTRAL MATRIX is the Hermitian, positive-semidefinite matrix

    S_ff(Omega)[a,b] = sqrt( G_a(Omega) G_b(Omega) ) gamma_ab(Omega)
                       exp( i theta_ab(Omega) )                          (1)

with the AUTO-PSDs G_a(Omega) = S_ff[a,a] on the diagonal (gamma_aa = 1,
theta_aa = 0) and, off the diagonal, the ORDINARY COHERENCE
gamma_ab(Omega) in [0, 1] (gamma = 0 mutually incoherent, gamma = 1 fully
coherent) and the PHASE theta_ab(Omega) = -theta_ba (Newland eq. 6.x;
Bendat & Piersol eq. 5.x). |S_ff[a,b]|^2 = gamma_ab^2 G_a G_b is the classic
coherence identity. A physical cross-spectral matrix is Hermitian PSD; a
coherence MODEL (a clipped constant table, or a distance/frequency decay) can
violate PSD-ness by a rounding margin or an over-large off-diagonal, so (1) is
projected onto the NEAREST Hermitian PSD matrix (Higham 1988 — clip the
negative eigenvalues to zero), a documented no-op on an already-valid matrix.

The structure LINEAR system carries each input f_a(t) through its OWN transfer
function: the FRF COLUMN H_a(Omega) (nf, ndof) is the M17/M19 modal-
superposition FRF of input load pattern a, stacked into

    H(Omega) = [ H_1 | H_2 | ... | H_ninput ](Omega)      (nf, ndof, ninput)

(one column per input pattern). The stationary RESPONSE CROSS-PSD is the MIMO
matrix triple product (Bendat & Piersol eq. 7.x; Newland ch. 8)

    S_uu(Omega) = H(Omega) S_ff(Omega) H(Omega)^H          (nf, ndof, ndof)  (2)

whose DIAGONAL S_uu[j,j] = sum_{a,b} H[j,a] S_ff[a,b] conj(H[j,b]) is the
response PSD of DOF j. Likewise the per-element 6-component Voigt stress FRF
column stack H_sigma(Omega) (nf, 6, ninput) gives the STRESS-TENSOR CROSS-PSD

    S_sigmasigma(Omega) = H_sigma(Omega) S_ff(Omega) H_sigma(Omega)^H  (nf,6,6) (3)

— exactly the matrix the M21-M27 reductions consume, so the multi-input
S_sigmasigma flows straight into the equivalent-von-Mises / critical-plane /
non-proportional / non-Gaussian / non-stationary / evolutionary machinery
UNCHANGED (they read S_sigmasigma and its 6x6 spectral-moment matrices
read-only). The single-input case (ninput = 1, S_ff = [G]) makes (2)-(3)
collapse to the rank-1 |H|^2 G law of M19/M20/M21 exactly.

Two limiting reductions, both proven in the validation:
* a DIAGONAL S_ff (mutually incoherent inputs, gamma_ab = 0) makes (3) the SUM
  of the per-input single-input cross-PSDs, S_sigmasigma = sum_a H_a G_a H_a^H,
  so every response/stress PSD is the SUM of the single-input answers;
* a RANK-1 fully-coherent S_ff = v v^H (one common source split across the
  patterns, gamma_ab = 1) makes H S_ff H^H = (H v)(H v)^H the single-input
  answer for the EFFECTIVE combined pattern H v, so the whole multi-input path
  reduces to the M19/M20/M21 single-input path for the summed force pattern.
Partial coherence interpolates monotonically between these two.

See PORTING_GUIDE.md roadmap M28.
"""

from __future__ import annotations

import numpy as np


# ============================================================================
# Nearest Hermitian PSD projection (build-order item 1a)
# ============================================================================

def nearest_psd(S, tol=0.0):
    """Project each Hermitian matrix in the stack ``S`` (..., n, n) onto the
    NEAREST positive-semidefinite Hermitian matrix by clipping its negative
    eigenvalues to zero (Higham 1988 — the eigenvalue-clipping projection in
    the Frobenius norm for a matrix already Hermitian).

    A physical input cross-spectral matrix S_ff(Omega) is Hermitian PSD, but a
    coherence MODEL (a constant table with an over-large off-diagonal, or a
    distance/frequency decay) can produce a matrix that is Hermitian but has a
    slightly negative eigenvalue. This projects it back, per frequency bin
    (``np.linalg.eigh`` vectorises over the leading axis), and reports the
    smallest eigenvalue seen and whether any clipping happened.

    Returns ``(Sproj, info)`` with ``info = {min_eig, clipped, nclipped}``.
    On an ALREADY-VALID (PSD to within ``tol``) matrix the projection is a
    numerical no-op: the eigenvalues are non-negative so nothing is clipped and
    ``Sproj`` reconstructs ``S`` to round-off (asserted in the validation)."""
    A = np.asarray(S)
    # Hermitian-symmetrise defensively (average with the conjugate transpose):
    # a coherence-model assembly is Hermitian by construction, but this guards
    # against a stray asymmetric rounding term before the eigendecomposition
    swap = tuple(range(A.ndim - 2)) + (A.ndim - 1, A.ndim - 2)
    Ah = 0.5 * (A + np.conj(np.transpose(A, swap)))
    w, V = np.linalg.eigh(Ah)                       # ascending real eigenvalues
    min_eig = float(w.min()) if w.size else 0.0
    wc = np.clip(w, 0.0, None)
    nclipped = int(np.count_nonzero(w < -abs(tol)))
    if nclipped == 0:
        # nothing negative beyond tol -> a no-op; return the Hermitian-symmetric
        # input unchanged (up to the defensive symmetrise) so a valid matrix is
        # reproduced to round-off
        return Ah, {"min_eig": min_eig, "clipped": False, "nclipped": 0}
    # rebuild V diag(wc) V^H per bin
    Sproj = np.einsum("...ij,...j,...kj->...ik", V, wc, np.conj(V))
    # re-Hermitianise the reconstruction (kills asymmetric round-off)
    Sproj = 0.5 * (Sproj + np.conj(np.transpose(Sproj, swap)))
    return Sproj, {"min_eig": min_eig, "clipped": True, "nclipped": nclipped}


def matrix_rank_psd(S, rtol=1e-9):
    """The numerical rank of a Hermitian PSD matrix (or per-bin stack) from its
    eigenvalues (count above ``rtol`` times the largest). Used to detect the
    RANK-1 fully-coherent input matrix (one common source) so the multi-input
    Monte-Carlo can delegate to the M21 single-input path for a bit-identical
    reduction."""
    A = np.asarray(S)
    w = np.linalg.eigvalsh(0.5 * (A + np.conj(np.swapaxes(A, -1, -2))))
    wmax = np.abs(w).max(axis=-1, keepdims=True)
    thr = rtol * np.where(wmax > 0, wmax, 1.0)
    return np.count_nonzero(w > thr, axis=-1)


# ============================================================================
# Coherence models + input cross-PSD matrix assembly (build-order item 1a)
# ============================================================================

def constant_coherence(ninput, gamma, phase=0.0):
    """A CONSTANT (frequency-independent) coherence matrix gamma_ab and phase
    matrix theta_ab for ``ninput`` inputs.

    ``gamma`` is either a scalar applied to every off-diagonal pair (diagonal
    forced to 1) or a full (ninput, ninput) symmetric matrix in [0, 1].
    ``phase`` (radians) is either a scalar theta applied with an
    anti-symmetric sign convention theta_ab = +theta for a < b, -theta for
    a > b (so the matrix is Hermitian), or a full antisymmetric matrix.
    Returns ``(gamma_mat, phase_mat)`` both (ninput, ninput)."""
    n = int(ninput)
    g = np.asarray(gamma, dtype=float)
    if g.ndim == 0:
        gm = np.full((n, n), float(g))
        np.fill_diagonal(gm, 1.0)
    else:
        gm = g.astype(float).copy()
        gm = 0.5 * (gm + gm.T)                       # symmetrise
        np.fill_diagonal(gm, 1.0)
    gm = np.clip(gm, 0.0, 1.0)                        # a coherence lives in [0,1]
    p = np.asarray(phase, dtype=float)
    if p.ndim == 0:
        th = np.zeros((n, n))
        iu = np.triu_indices(n, 1)
        th[iu] = float(p)
        th = th - th.T                               # antisymmetric
    else:
        th = p.astype(float).copy()
        th = 0.5 * (th - th.T)                        # antisymmetrise
    return gm, th


def exponential_coherence(freqs_hz, positions, decay, ref_speed=1.0, phase=0.0):
    """A frequency-dependent EXPONENTIAL / DECAY coherence model for spatially
    DISTRIBUTED loads (the wind / turbulence / pressure-field coherence of
    Davenport-type models; Bendat & Piersol sec. 5.x):

        gamma_ab(f) = exp( -decay * |x_a - x_b| * f / ref_speed )         (4)

    from per-input scalar positions ``positions`` (ninput,) [or (ninput, k)
    vector positions], a decay coefficient ``decay`` (>= 0) and a reference
    convection speed ``ref_speed``. Coherence falls off with SEPARATION and
    FREQUENCY (near inputs / low frequencies stay coherent; far inputs / high
    frequencies decorrelate), always in [0, 1]. Returns ``(gamma_stack,
    phase_mat)`` with ``gamma_stack`` (nf, ninput, ninput) and a constant
    phase matrix (a propagation phase theta_ab(f) = 2 pi f d_ab / ref_speed is
    a documented refinement; the base model uses ``phase`` as a constant lag)."""
    f = np.asarray(freqs_hz, dtype=float)
    X = np.asarray(positions, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    # pairwise Euclidean separation d_ab
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))   # (n, n)
    speed = float(ref_speed) if ref_speed else 1.0
    # gamma_ab(f) = exp(-decay d f / speed), broadcast over frequency
    arg = -abs(float(decay)) * d[None, :, :] * f[:, None, None] / speed
    gamma = np.exp(arg)
    gamma = np.clip(gamma, 0.0, 1.0)
    for a in range(n):
        gamma[:, a, a] = 1.0
    _gm, th = constant_coherence(n, 0.0, phase)
    return gamma, th


def input_cross_psd_matrix(auto_psds, gamma=None, phase=0.0):
    """Assemble the Hermitian input cross-spectral matrix S_ff(Omega)
    (nf, ninput, ninput) from the per-input AUTO-PSDs and a coherence model
    (theory eq. (1)), then project it onto the nearest Hermitian PSD matrix
    (``nearest_psd``).

    Parameters
    ----------
    auto_psds : (nf, ninput)
        the input auto-PSDs G_a(Omega) >= 0, one column per input, sampled on
        the FRF sweep grid.
    gamma : None | scalar | (ninput,ninput) | (nf,ninput,ninput)
        the ordinary coherence gamma_ab in [0, 1]. ``None`` (default) means the
        IDENTITY (mutually incoherent inputs — a diagonal S_ff, the SUM
        reduction). A scalar or (n,n) matrix is a constant coherence
        (``constant_coherence``); a full (nf,n,n) stack is a
        frequency-dependent model (``exponential_coherence``).
    phase : scalar | (ninput,ninput)
        the phase theta_ab (radians) — a scalar becomes the antisymmetric
        Hermitian phase; a full matrix is antisymmetrised. Ignored on the
        diagonal.

    Returns a dict ``{Sff, gamma, phase, auto_psds, min_eig, projected,
    nclipped}``. ``Sff`` is the (nf, ninput, ninput) complex Hermitian PSD
    cross-spectral stack. For ninput = 1 the result is exactly [[G]] (the M19
    single-input PSD, a bit-identical special case)."""
    G = np.clip(np.asarray(auto_psds, dtype=float), 0.0, None)
    if G.ndim == 1:
        G = G[:, None]
    nf, n = G.shape
    # coherence stack gamma_ab(f) (nf, n, n) and phase matrix theta_ab (n, n)
    if gamma is None:
        gstack = np.broadcast_to(np.eye(n), (nf, n, n))
        _g, th = constant_coherence(n, 0.0, phase)
    else:
        g = np.asarray(gamma, dtype=float)
        if g.ndim <= 2:
            gm, th = constant_coherence(n, g, phase)
            gstack = np.broadcast_to(gm, (nf, n, n))
        else:
            gstack = np.clip(g, 0.0, 1.0)
            _g, th = constant_coherence(n, 0.0, phase)
    # sqrt(G_a G_b) outer product per frequency
    sqrtG = np.sqrt(G)                                 # (nf, n)
    outer = sqrtG[:, :, None] * sqrtG[:, None, :]      # (nf, n, n)
    Sff = outer * gstack * np.exp(1j * th)[None, :, :]  # eq. (1)
    # the diagonal is exactly G_a (gamma_aa = 1, theta_aa = 0) — force it so
    # round-off in the sqrt/outer never perturbs the auto-PSDs
    for a in range(n):
        Sff[:, a, a] = G[:, a]
    Sff, info = nearest_psd(Sff)
    return {"Sff": Sff, "gamma": gstack, "phase": th, "auto_psds": G,
            "min_eig": info["min_eig"], "projected": info["clipped"],
            "nclipped": info["nclipped"]}


# ============================================================================
# Per-input FRF column stacks (build-order item 1b)
# ============================================================================

def stress_frf_columns(q_cols, Sigma):
    """The per-input STRESS FRF column stack H_sigma(Omega) (nf, nchan, ninput)
    from the per-input modal-coordinate columns ``q_cols`` (nf, nmode, ninput)
    and the M20 stress modes ``Sigma`` (nmode, nchan): column a is the stress
    FRF of input pattern a, H_sigma[:, :, a] = q_cols[:, :, a] @ Sigma (the M20
    stress-FRF product, one per input). Stress is a LINEAR operator on
    displacement, so it commutes with the modal superposition exactly as in the
    single-input M20 path."""
    q = np.asarray(q_cols)
    return np.einsum("fma,mc->fca", q, np.asarray(Sigma))   # (nf, nchan, ninput)


def element_voigt_frf_columns(Hs_cols, cols):
    """The 6-component Voigt stress FRF column stack H_sigma(Omega) (nf, 6,
    ninput) of ONE element, slicing its 6 channel columns (``cols`` from
    ``random_response.element_voigt_blocks``) out of the full per-input stress
    FRF ``Hs_cols`` (nf, nchan, ninput)."""
    return np.asarray(Hs_cols)[:, np.asarray(cols, dtype=int), :]  # (nf,6,nin)


def displacement_frf_columns(q_cols, Phi):
    """The per-input physical (equation-space) FRF column stack U(Omega)
    (nf, ndof, ninput) from the per-input modal columns and the mode matrix
    ``Phi`` (ndof, nmode): U[:, :, a] = q_cols[:, :, a] @ Phi^T."""
    q = np.asarray(q_cols)
    return np.einsum("fma,jm->fja", q, np.asarray(Phi))   # (nf, ndof, ninput)


# ============================================================================
# The MIMO matrix triple products (build-order item 1b)
# ============================================================================

def stress_tensor_cross_psd_multi(Hcols, Sff):
    """The 6x6 stress-tensor cross-PSD S_sigmasigma(Omega) = H_sigma S_ff
    H_sigma^H (theory eq. (3)) for ONE element, from its per-input Voigt stress
    FRF column stack ``Hcols`` (nf, 6, ninput) and the Hermitian input
    cross-PSD ``Sff`` (nf, ninput, ninput).

    Returns a (nf, 6, 6) complex Hermitian array
    S[f,c,d] = sum_{a,b} H[f,c,a] Sff[f,a,b] conj(H[f,d,b]). For ninput = 1
    (a single scalar input, S_ff = [G]) this DELEGATES to the M21 rank-1
    ``stress_tensor_cross_psd`` so the single-input case is BIT-IDENTICAL to the
    M19/M20/M21 answer (the diagonal / rank-1 special case). For a DIAGONAL
    S_ff it equals the SUM of the per-input rank-1 cross-PSDs; for a rank-1
    coherent S_ff it equals the single-input cross-PSD of the effective
    combined pattern."""
    H = np.asarray(Hcols)
    S = np.asarray(Sff)
    if H.ndim != 3 or H.shape[1] != 6:
        raise ValueError("stress_tensor_cross_psd_multi needs a (nf, 6, ninput)"
                         f" Voigt stress FRF column stack; got {H.shape}.")
    if H.shape[2] == 1:
        # single input: the M21 rank-1 path EXACTLY (bit-identical special case)
        from .multiaxial_fatigue import stress_tensor_cross_psd
        return stress_tensor_cross_psd(H[:, :, 0], S[:, 0, 0].real)
    # H S_ff H^H per frequency (the MIMO matrix triple product)
    return np.einsum("fca,fab,fdb->fcd", H, S, np.conj(H))


def response_cross_psd_diagonal(Ucols, Sff):
    """The DIAGONAL of the response cross-PSD, S_uu[j,j](Omega) = sum_{a,b}
    U[j,a] S_ff[a,b] conj(U[j,b]) (theory eq. (2)) — the response PSD of every
    equation DOF, (nf, ndof) REAL. From the per-input physical FRF column stack
    ``Ucols`` (nf, ndof, ninput) and the input cross-PSD ``Sff``. For ninput = 1
    this DELEGATES to the M19 rank-1 |U|^2 G law (bit-identical); for a diagonal
    S_ff it is the SUM of the per-input response PSDs."""
    U = np.asarray(Ucols)
    S = np.asarray(Sff)
    if U.shape[2] == 1:
        return (np.abs(U[:, :, 0]) ** 2) * S[:, 0, 0].real[:, None]   # M19 law
    return np.einsum("fja,fab,fjb->fj", U, S, np.conj(U)).real


def response_cross_psd(Ucols, Sff, dofs=None):
    """The FULL response cross-PSD S_uu(Omega) = U S_ff U^H (theory eq. (2)) for
    a SELECTED set of equation DOFs ``dofs`` (default: all — beware the (nf,
    ndof, ndof) memory for a large model). Returns (nf, k, k) complex Hermitian
    for the ``k`` selected DOFs. The off-diagonals carry the response
    cross-spectra / coherence the diagonal discards."""
    U = np.asarray(Ucols)
    if dofs is not None:
        U = U[:, np.asarray(dofs, dtype=int), :]
    S = np.asarray(Sff)
    return np.einsum("fja,fab,fkb->fjk", U, S, np.conj(U))


def equivalent_vonmises_psd_multi(Scross):
    """The EQUIVALENT VON MISES PSD S_vm(Omega) = trace(Q S_sigmasigma)(Omega)
    (M21 eq. (3)) of a multi-input stress-tensor cross-PSD ``Scross`` (nf, 6, 6),
    a REAL scalar PSD (Q symmetric real, S Hermitian => trace real). This is the
    matrix-trace form that works for ANY input matrix — it does NOT assume the
    rank-1 |H|^2 S single-input shortcut, so it is the multi-input generalisation
    of ``multiaxial_fatigue.equivalent_vonmises_psd``. Clipped >= 0 (Q is PSD on
    the deviator; guards round-off)."""
    from .multiaxial_fatigue import _Q_VM
    S = np.asarray(Scross)
    Svm = np.einsum("ab,fab->f", _Q_VM, S).real
    return np.clip(Svm, 0.0, None)


# ============================================================================
# The full multi-input multiaxial summary for ONE element (all reductions)
# ============================================================================

def multi_input_multiaxial_summary(Scross, omega, m, C, mean_stress=0.0,
                                    ultimate=0.0, naz=24, npol=13):
    """Evaluate ALL M21 multiaxial reductions on ONE element's multi-input
    stress-tensor cross-PSD ``Scross`` (nf, 6, 6) on the angular grid ``omega``:
    the equivalent VON MISES PSD, the MAX-NORMAL and MAX-SHEAR critical planes —
    each reduced to a scalar PSD, its M20 spectral moments and the four M20
    damage estimators. Returns the SAME nested-dict shape as
    ``multiaxial_fatigue.multiaxial_fatigue_summary`` so the M22-M27 correction
    helpers (which read ``von_mises`` / ``normal_plane`` / ``shear_plane`` /
    ``Mmats`` / ``Scross`` read-only) compose with it UNCHANGED — the whole
    point of feeding the multi-input S_sigmasigma into the estimator family.

    The ONLY difference from the single-input summary is HOW S_sigmasigma is
    formed (the MIMO triple product H S_ff H^H instead of the rank-1 |H|^2 S):
    every reduction below is byte-for-byte the M21 machinery."""
    from . import multiaxial_fatigue as mf
    from . import spectral_fatigue as sf
    from .random_response import spectral_moments

    omega = np.asarray(omega, dtype=float)
    S = np.asarray(Scross)

    # equivalent von Mises PSD (the trace form; works for any input matrix)
    Svm = equivalent_vonmises_psd_multi(S)
    mom_vm = spectral_moments(omega, Svm, nmax=4)
    vm = {"psd": Svm, "moments": mom_vm,
          "summary": sf.fatigue_summary(mom_vm, m, C, mean_stress, ultimate)}

    # critical planes: the 6x6 spectral-moment matrices (M21 eq. (7)) — the
    # SAME real-symmetric moment machinery, consuming the multi-input Scross
    Mmats = mf.tensor_moment_matrices(omega, S, nmax=4)

    def _plane(method):
        cp = mf.critical_plane_search(Mmats, method=method, naz=naz, npol=npol)
        cp["summary"] = sf.fatigue_summary(cp["moments"], m, C, mean_stress,
                                           ultimate)
        return cp

    normal_cp = _plane("normal")
    shear_cp = _plane("shear")
    return {"von_mises": vm, "normal_plane": normal_cp,
            "shear_plane": shear_cp, "Mmats": Mmats, "Scross": S,
            "omega": omega}


# ============================================================================
# Cross-spectral / coherence diagnostics of the response (reporting)
# ============================================================================

def response_coherence(Suu):
    """The response ordinary-coherence matrix gamma_jk^2(Omega) = |S_uu[j,k]|^2 /
    (S_uu[j,j] S_uu[k,k]) from a (nf, k, k) response cross-PSD ``Suu`` — the
    output-side coherence diagnostic (Bendat & Piersol sec. 5.x). Returns the
    (nf, k, k) real coherence stack in [0, 1] (diagonal 1)."""
    S = np.asarray(Suu)
    diag = np.einsum("fjj->fj", S).real                  # (nf, k)
    denom = np.sqrt(np.clip(diag[:, :, None] * diag[:, None, :], 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.where(denom > 0.0, (np.abs(S) ** 2) / (denom ** 2), 0.0)
    return np.clip(coh, 0.0, 1.0)

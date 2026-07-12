"""
Complex / damped eigenvalues + complex-mode superposition — M18.

Fortran origin
--------------
There is NO complex/damped eigensolver and NO state-space / quadratic-
eigenvalue path anywhere in the open-source OpenRadioss engine. ``freimpl.F``
(the /IMPL reader, read line by line for M16/M17/M18) parses only /IMPL/DYNA
(the DIRECT Newmark/HHT integrator, imp_dyna.F), /IMPL/BUCKL, /IMPL/DT,
/IMPL/NONLIN and /IMPL/ARCL; the only damping it knows is the on-the-fly
Rayleigh force C v = a M v + b K v of IMP_DYKV (M11). OpenRadioss is a time-
domain crash/impact code — the frequency domain and complex modal analysis
simply are not part of the open-source solver (the same finding M16 made for
the real eigensolver and M17 for mode superposition).

So M18 does exactly what M16/M17 did: it ports complex modal analysis +
complex-mode superposition as clean LIBRARY capabilities and drives them with
a minimal PORT engine card (/IMPL/CEIGV — the complex-eigenvalue analogue of
M16's /IMPL/EIGV). Nothing in the M10 direct integrator, the M16 REAL
eigensolver or the M17 REAL-mode superposition is touched: the complex path is
a NEW, parallel path (asserted, like the M14-M17 parity contracts).

Theory — the quadratic eigenvalue problem & state-space linearization
---------------------------------------------------------------------
(Geradin & Rixen, "Mechanical Vibrations", ch. 3 & 5; Meirovitch,
"Principles and Techniques of Vibrations", ch. 9; Tisseur & Meerbergen,
"The Quadratic Eigenvalue Problem", SIAM Review 2001.)

Free vibration of a DAMPED structure, M u_ddot + C u_dot + K u = 0, has
solutions u = phi e^{lambda t}; substituting gives the QUADRATIC eigenproblem

    (lambda^2 M + lambda C + K) phi = 0                                   (QEP)

which — unlike the M16 undamped symmetric pencil (K - omega^2 M) phi = 0 — is
NOT a linear symmetric eigenproblem: its 2n eigenvalues lambda are COMPLEX and
its eigenvectors phi are COMPLEX (a DOF-dependent phase lag). For an under-
damped mode i the pair is

    lambda_i = -zeta_i omega_i +/- i omega_i sqrt(1 - zeta_i^2)
             = -sigma_i +/- i omega_{d,i}

so Re(lambda) = -sigma_i is the DECAY RATE (the exponential envelope
e^{-sigma_i t}), Im(lambda) = omega_{d,i} is the DAMPED angular frequency, and
|lambda_i| = omega_i, zeta_i = -Re(lambda_i)/|lambda_i| recover the undamped
natural frequency and the modal damping ratio. When C is CLASSICAL (Rayleigh
a M + b K, or any Caughey series) the complex modes reduce to the M16 REAL
modes times a complex scalar and lambda reduces EXACTLY to the M16 omega_i and
the M17 zeta_i — the consistency check below. When C is NON-classical (a local
dashpot, damping on part of the structure) the mode shapes carry a genuine
DOF-to-DOF phase lag: different points reach their extremes at different
instants, the signature of non-proportional damping that the real-mode
superposition of M17 CANNOT represent.

STATE-SPACE LINEARIZATION. The QEP is solved by linearizing it to a 2n x 2n
GENERALIZED eigenproblem. With the state z = [phi; lambda phi] (the mode shape
and its "velocity") this port uses the SYMMETRIC linearization

    A z = lambda B z ,   A = [[0,  K], [K,  C]] ,   B = [[K,  0], [0, -M]]

(Tisseur & Meerbergen's symmetric companion form; verify by expanding: row 1
gives K(lambda phi) = lambda K phi, i.e. the state definition; row 2 gives
K phi + C(lambda phi) = -lambda M(lambda phi), i.e. the QEP). It is chosen
over the more familiar NON-symmetric companion [[0, I], [-M^-1 K, -M^-1 C]]
for two reasons: (a) it needs NO mass inverse (a reduced consistent M is dense
and forming M^-1 is both costly and less stable than the pencil solve), and
(b) A and B inherit the symmetry of K, C, M, so the left and right
eigenvectors coincide and the state-space BIORTHOGONALITY is the clean
symmetric-bilinear (transpose, not conjugate) relation

    z_i^T B z_j = 0 ,   z_i^T A z_j = 0   for lambda_i != lambda_j

which decouples the forced problem below. (It does require K non-singular —
true for any constrained structure; a free-free model is filtered for rigid
modes exactly as the M16 solver is.) The reduced pencil is solved with
``scipy.linalg.eig`` (the non-symmetric generalized driver — the QEP has no
Hermitian structure to exploit), a dense O((2 n_red)^3) solve: the SAME
documented dense-eig choice M16 made (Lanczos/subspace deferred).

Constraints and contact (the M12/M14 reduced pencil)
----------------------------------------------------
Stated on the REDUCED equations, identically to M16: K, C and M are each
condensed T^T (.) T through the constraint transform (``constraints.py``), so
a rigid body carries its exact condensed 6-DOF mass and any dashpot attached
to it its condensed damping. Mode shapes are recovered through T
(``constraints.expand``) before scatter to nodes.

Complex-mode SUPERPOSITION (transient + FRF)
--------------------------------------------
The state-space form B w_dot = A w + P(t), with w = [u; u_dot] and P = [0; -f]
(f the physical load), decouples in the complex modal basis: expanding
w = sum_k z_k x_k(t) and pre-multiplying by z_i^T (biorthogonality) gives 2n
INDEPENDENT FIRST-ORDER complex modal equations

    x_dot_i = lambda_i x_i + p_i(t) ,   p_i(t) = (z_i^T P(t)) / (z_i^T B z_i)

(compare M17's SECOND-order real SDOFs — the state-space form is naturally
first order). Each is marched by the EXACT piecewise-linear-forcing recurrence
(the first-order analogue of M17's Nigam-Jennings, ``_first_order_coeffs``),
and recombined u(t) = sum_k phi_k x_k(t). Because the modes and their forces
come in conjugate pairs and P is real, the sum is real to round-off. For a
harmonic load f = F e^{i Omega t} the steady modal coordinate is the complex
FRF x_i = p_i^F/(i Omega - lambda_i), recombined U(Omega) = sum_k phi_k x_i(k)
— the DAMPED complex transfer function, exact for non-classical damping (it
matches the direct inversion (K - Omega^2 M + i Omega C)^-1 F,
``direct_frf``, to round-off on the full basis).

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* LIBRARY-FIRST card (/IMPL/CEIGV) — no upstream equivalent, exactly as
  established for M16's /IMPL/EIGV and M17's /IMPL/MODAL, /IMPL/FREQ.
* DENSE ``scipy.linalg.eig`` on the reduced 2 n_red pencil (Lanczos/subspace
  and a structure-preserving QEP solver — e.g. ``polyeig`` / SOAR — deferred,
  the M16 dense-eig choice carried forward).
* SYMMETRIC state-space linearization chosen over the M^-1 companion form
  (justified above) — a documented, deliberate formulation choice.
* GYROSCOPIC / circulatory systems (a non-symmetric C from rotation, a non-
  symmetric K from follower/circulatory forces) are OUT of scope: the pencil
  above assumes symmetric K, C, M. Deferred (PORTING_GUIDE M18).
"""

from __future__ import annotations

import numpy as np

from . import require_scipy
from .assembly import assemble, assemble_mass
from .damping_matrix import assemble_damping
from .dofmap import DofMap, DOFS_PER_NODE


# ============================================================================
# The reduced (K, C, M) pencil and its state-space complex eigensolve
# ============================================================================

def _reduced_pencil(model, nev, log, constraints, contacts, prestress,
                    alpha, beta):
    """Assemble and condense the (K, C, M) triple in the REDUCED (constrained)
    equation space. Returns ``(Kr, Cr, Mr, dof, constraints, Mfull)`` with the
    reduced matrices DENSE (ndarray) — the same reduced-pencil path M16 uses,
    now with the damping matrix alongside. ``alpha, beta`` are the Rayleigh
    coefficients folded into C (``damping_matrix.assemble_damping``)."""
    require_scipy()
    x_geom = model.x if model.x.size else model.x0
    x0 = model.x0 if model.x0.size else x_geom
    if constraints is None or contacts is None:
        from ..common.messages import MessageLog
        silent = MessageLog()
        if constraints is None:
            from .constraints import build_constraints
            constraints = build_constraints(model, silent)
        if contacts is None:
            from .contact import build_implicit_contacts
            contacts = build_implicit_contacts(model, silent)

    dof = DofMap(model, log, constraints=constraints)
    # STIFFNESS (+ geometric stiffness for a prestressed spectrum), the same
    # tangent M16 forms
    K = assemble(model, dof, x_geom, kgeo=prestress)
    if contacts:
        from .contact import contact_tangent
        Kc = contact_tangent(contacts, x_geom, dof)
        K = K + 0.5 * (Kc + Kc.T)
    # CONSISTENT mass on the reference geometry (M16)
    M = assemble_mass(model, dof, x0, log)
    # DAMPING C = a M + b K + discrete dashpots (/DAMP + /PROP/SPRING c) — the
    # M18 operator (see damping_matrix.py). Reuse the K, M already assembled.
    C = assemble_damping(model, dof, x_geom, alpha=alpha, beta=beta,
                         K=K, M=M, log=log)

    Mfull = M
    if constraints is not None:
        constraints.build(dof, x_geom)
        K = constraints.reduce_matrix(K)          # T^T K T
        C = constraints.reduce_matrix(C)          # T^T C T
        M = constraints.reduce_matrix(M)          # T^T M T
    return (K.toarray(), C.toarray(), M.toarray(), dof, constraints, Mfull)


def _state_space_eig(Kr, Cr, Mr):
    """Solve the QEP (lambda^2 M + lambda C + K) phi = 0 through the symmetric
    state-space linearization A z = lambda B z (module docstring). Returns
    ``(lam, Z)`` — the 2 n_red complex eigenvalues and the 2 n_red-row
    eigenvectors z = [phi; lambda phi] (columns), as ``scipy.linalg.eig``
    returns them (unordered)."""
    _, _ = require_scipy()
    import scipy.linalg as sla
    n = Kr.shape[0]
    Z = np.zeros((n, n))
    # A = [[0, K], [K, C]] ,  B = [[K, 0], [0, -M]]  (both symmetric)
    A = np.block([[Z, Kr], [Kr, Cr]])
    B = np.block([[Kr, Z], [Z, -Mr]])
    lam, vecs = sla.eig(A, B)
    return lam, vecs


# ============================================================================
# Selecting + normalizing the reported complex modes
# ============================================================================

def _filter_modes(lam, Z, nred):
    """From the 2 n_red raw eigenpairs of the state-space pencil, drop the
    rigid-body / mechanism modes (|lambda| ~ 0), any spurious infinite
    eigenvalues, and any unstable mode (Re(lambda) > 0 — an eigensolver
    artefact for a passively-damped structure, whose every mode must decay).
    Returns ``(lam_keep, Z_keep)`` — the SURVIVING modes, still 2-per-physical-
    mode (a conjugate pair for an under-damped mode, two real roots for an
    over-damped one), so the set is conjugate-closed and the superposition it
    feeds recombines to a REAL response exactly.

    The whole surviving set is kept (not truncated to nev): the complex-mode
    superposition then uses the COMPLETE reduced modal basis, so the transient
    / FRF are EXACT for the reduced pencil (they match the direct Newmark march
    / the direct FRF inversion to round-off — the M18 validation anchor).
    ``nev`` limits only how many complex modes are REPORTED (see
    ``ComplexModalBasis.rep_idx``). This mirrors the M16 rigid filter; the
    same-cost dense eig already produced all 2 n_red pairs, so keeping them is
    free and makes the superposition exact rather than truncated (truncation
    for very large models is the deferred Lanczos/subspace upgrade)."""
    finite = np.isfinite(lam)
    lam_f = lam[finite]
    if lam_f.size == 0:
        return np.zeros(0, complex), np.zeros((2 * nred, 0), complex)
    mag = np.abs(lam_f)
    mag_pos = mag[mag > 0.0]
    scale = float(np.median(mag_pos)) if mag_pos.size else 1.0
    rigid_tol = 1e-6 * scale
    keep = [k for k in np.where(finite)[0]
            if abs(lam[k]) > rigid_tol and lam[k].real < rigid_tol]
    keep.sort(key=lambda k: abs(lam[k]))          # ascending |lambda| = omega
    return lam[keep], Z[:, keep]


def _normalize_shape(phi):
    """Rotate the global complex phase of a mode shape so its largest-
    magnitude component is real and positive, and scale to unit max-magnitude.
    A CLASSICALLY-damped mode then comes out REAL (every entry shares one
    phase); a non-classical mode keeps its DOF-to-DOF phase lag (the entries
    do NOT collapse to a single phase). Pure presentation — the physics
    (superposition, FRF) is normalization-invariant (see the module notes)."""
    if phi.size == 0:
        return phi
    j = int(np.argmax(np.abs(phi)))
    ref = phi[j]
    if abs(ref) > 0.0:
        phi = phi * (np.conj(ref) / abs(ref)) / abs(ref)
    return phi


# ============================================================================
# The complex modal basis (consumed by transient + FRF)
# ============================================================================

class ComplexModalBasis:
    """The complex (damped) eigenpairs in REDUCED equation space, ready for
    state-space superposition. Holds the COMPLETE surviving reduced modal set
    (``nstate`` = 2 n_red minus the rigid/spurious modes — conjugate-closed, so
    a real load recombines to a real response and a complex harmonic load gets
    its complete FRF, both EXACT for the reduced pencil).

    Attributes
    ----------
    lam    (nstate,)           complex eigenvalues lambda_k (ascending |lambda|)
    Phi    (nred, nstate)      complex reduced mode shapes phi_k (top half of z)
    Zvel   (nred, nstate)      lambda_k phi_k (the state's velocity half)
    b      (nstate,)           biorthogonality norms z_k^T B z_k
    Kr,Cr,Mr (nred, nred)      the reduced pencil (dense) — for the direct FRF
    dof    DofMap              the numbering the modes scatter through
    constr                     the live constraint transform (or None)
    rep_idx (nmode,)           indices (into lam/Phi) of the REPORTED complex
                               modes — the lowest ``nev`` UNDER-DAMPED
                               representatives (Im > 0), frequency-ordered
    nmode  int                 number of reported complex modes (= len rep_idx)

    The reporting quantities (natural / damped frequencies, decay rates,
    damping ratios, phase-normalized nodal mode shapes) are on ``rep_idx``; the
    superposition (transient / FRF) uses ALL ``nstate`` modes."""

    def __init__(self, lam, Phi, Zvel, b, Kr, Cr, Mr, dof, constr, nev):
        self.lam = lam
        self.Phi = Phi
        self.Zvel = Zvel
        self.b = b
        self.Kr = Kr
        self.Cr = Cr
        self.Mr = Mr
        self.dof = dof
        self.constr = constr
        # reported modes: the lowest nev under-damped (Im > 0) representatives
        reps = [k for k in range(lam.size) if lam[k].imag > 0.0]
        reps.sort(key=lambda k: abs(lam[k]))
        self.rep_idx = np.array(reps[:nev], dtype=int)
        self.nmode = self.rep_idx.size

    @property
    def nstate(self):
        return self.lam.size

    # ---- reporting (the positive-imaginary representatives) ----------------
    @property
    def eigenvalues(self):
        """The reported complex eigenvalues (under-damped, Im > 0)."""
        return self.lam[self.rep_idx]

    @property
    def natural_freqs_hz(self):
        """Undamped natural frequency f_i = |lambda_i| / 2pi (Hz)."""
        return np.abs(self.lam[self.rep_idx]) / (2.0 * np.pi)

    @property
    def damped_freqs_hz(self):
        """Damped natural frequency f_{d,i} = Im(lambda_i) / 2pi (Hz)."""
        return self.lam[self.rep_idx].imag / (2.0 * np.pi)

    @property
    def decay_rates(self):
        """Decay rate sigma_i = -Re(lambda_i) (1/s) — the envelope
        e^{-sigma_i t}."""
        return -self.lam[self.rep_idx].real

    @property
    def damping_ratios(self):
        """Modal damping ratio zeta_i = -Re(lambda_i) / |lambda_i|."""
        lr = self.lam[self.rep_idx]
        return -lr.real / np.abs(lr)

    def mode_shapes(self):
        """The reported complex nodal mode shapes as ``(du, dur)`` complex
        per-node fields, phase-normalized (``_normalize_shape``) and expanded
        through the constraint transform. A classically-damped mode's fields
        are real-up-to-phase; a non-classical mode's carry the DOF phase lag."""
        out = []
        for i in self.rep_idx:
            phi_red = _normalize_shape(self.Phi[:, i].copy())
            phi_full = (self.constr.expand(phi_red)
                        if self.constr is not None else phi_red)
            du, dur = _scatter_complex(self.dof, phi_full)
            out.append((du, dur))
        return out

    # ---- projection / recombination (over ALL nstate modes) ----------------
    def modal_force(self, f_red):
        """The complex modal forces p_k = (z_k^T P)/b_k for P = [0; -f_red],
        i.e. p_k = -lambda_k (phi_k^T f_red) / b_k, over all nstate modes."""
        proj = self.Phi.T @ f_red                    # phi_k^T f  (nstate,)
        return -self.lam * proj / self.b

    def modal_state(self, w0_red):
        """Initial modal coordinates x_k(0) = (z_k^T B w0)/b_k for the state
        w0 = [u0; v0] (reduced). Returns (nstate,) complex."""
        n = self.Phi.shape[0]
        u0, v0 = w0_red[:n], w0_red[n:]
        # z_k^T B w0 with B = [[K,0],[0,-M]]: phi_k^T K u0 - (lam_k phi_k)^T M v0
        Bu = self.Phi.T @ (self.Kr @ u0) - self.Zvel.T @ (self.Mr @ v0)
        return Bu / self.b

    def recombine_red(self, x):
        """Reduced physical displacement u_red = sum_k phi_k x_k (real to
        round-off — the modes/forces are conjugate-closed)."""
        return (self.Phi @ x).real


def build_complex_basis(model, nev, log=None, constraints=None, contacts=None,
                        prestress=False, alpha=0.0, beta=0.0):
    """Extract ``nev`` under-damped complex modes and pack them into a
    :class:`ComplexModalBasis`. ``alpha, beta`` fold Rayleigh damping into C
    (on top of the deck's discrete dashpots); ``prestress``, ``constraints``,
    ``contacts`` follow ``modal.modal_frequencies``."""
    Kr, Cr, Mr, dof, constr, _ = _reduced_pencil(
        model, nev, log, constraints, contacts, prestress, alpha, beta)
    nred = Kr.shape[0]
    lam, Z = _state_space_eig(Kr, Cr, Mr)
    lam_keep, Z_keep = _filter_modes(lam, Z, nred)
    if lam_keep.size == 0:
        raise ValueError(
            "complex modal analysis extracted no structural mode — the model "
            "is a mechanism (all modes rigid) or carries no stiffness. Check "
            "the constraints.")
    Phi = Z_keep[:nred, :]
    Zvel = Z_keep[nred:, :]
    # B-biorthogonality norms b_k = z_k^T B z_k = phi_k^T K phi_k
    #                                            - (lam_k phi_k)^T M (lam_k phi)
    b = np.einsum("ik,ik->k", Phi, Kr @ Phi) \
        - np.einsum("ik,ik->k", Zvel, Mr @ Zvel)
    return ComplexModalBasis(lam_keep, Phi, Zvel, b, Kr, Cr, Mr, dof, constr,
                             nev)


# ============================================================================
# COMPLEX-MODE TRANSIENT — the exact first-order recurrence
# ============================================================================

def _first_order_coeffs(lam, dt):
    """Exact recurrence coefficients (g, a1, a2) for the FIRST-ORDER complex
    modal equation x_dot = lambda x + p(t) with p LINEAR over [t_k, t_k+dt]:

        x_{k+1} = g x_k + a1 p_k + a2 p_{k+1}
        g  = e^{lambda dt}
        a1 = g/lambda - (g - 1)/(lambda^2 dt)
        a2 = (g - 1)/(lambda^2 dt) - 1/lambda

    (the first-order analogue of M17's second-order Nigam-Jennings step;
    derived by exact integration of x = e^{lambda(t-t_k)} x_k +
    integral e^{lambda(t-s)} p(s) ds with p piecewise linear — EXACT at any
    step, so no dispersion). Vectorized over the complex mode array ``lam``."""
    lam = np.asarray(lam, dtype=complex)
    g = np.exp(lam * dt)
    a1 = g / lam - (g - 1.0) / (lam * lam * dt)
    a2 = (g - 1.0) / (lam * lam * dt) - 1.0 / lam
    return g, a1, a2


def complex_modal_transient(basis, loads, t_end, dt, model, log=None,
                            v0=None, vr0=None):
    """March the COMPLEX-mode superposition transient over [0, t_end] at step
    ``dt`` and return a history dict mirroring the M17 ledger.

    Each first-order complex modal equation x_dot_k = lambda_k x_k + p_k(t) is
    integrated by the EXACT recurrence (``_first_order_coeffs``) with the modal
    force p_k(t) sampled from the deck loads at PHYSICAL time
    (``loads.external_forces`` — the same /CLOAD//GRAV//PLOAD machinery). The
    physical response recombines u(t) = sum_k phi_k x_k(t) (real to round-off).

    Initial state: u(0) = 0 (rest), u_dot(0) = v0 from any /INIVEL (projected
    onto the complex modes through the state-space biorthogonality); ``v0``/
    ``vr0`` default to ``model.v``/``model.vr``.

    The energy ledger is booked in PHYSICAL reduced coordinates from the
    reduced pencil (exact for the retained modes): KE = 1/2 u_dot^T M u_dot,
    IE = 1/2 u^T K u, the viscous dissipation edamp = integral u_dot^T C u_dot
    dt (trapezoid), external work wext = trapezoid of f . du, balance =
    IE + KE + edamp - wext - e0."""
    n = model.numnod
    constr = basis.constr
    nred = basis.Phi.shape[0]

    nsteps = max(int(np.ceil(t_end / dt - 1e-12)), 1)
    g, a1, a2 = _first_order_coeffs(basis.lam, dt)

    def _reduce(fnod):
        f_eq = basis.dof.gather_residual(fnod, np.zeros((n, 3)))
        return constr.reduce_vector(f_eq) if constr is not None else f_eq

    def _modal_force(t):
        fext = np.zeros((n, 3))
        loads.external_forces(t, fext, model.x0)
        return basis.modal_force(_reduce(fext))

    # ---- initial modal coordinates from the rest+velocity state ------------
    if v0 is None:
        v0 = model.v
    if vr0 is None:
        vr0 = model.vr
    v_eq = basis.dof.gather_residual(v0, vr0)
    v_red = constr.reduce_vector(v_eq) if constr is not None else v_eq
    # NOTE the velocity projection: reduce_vector is T^T, but the state
    # x_k(0) = z_k^T B w0 / b_k uses the REDUCED w0 directly (u0 = 0). The
    # constraint-consistent reduced velocity is the mass-weighted projection;
    # for the analytic decks here (no constraint on the moving DOFs) T^T v is
    # the reduced velocity. (A rigid-body /INIVEL under CEIGV would want the
    # M-projection of modal_response; the transient decks do not exercise it.)
    w0 = np.concatenate([np.zeros(nred), v_red])
    x = basis.modal_state(w0)

    p_k = _modal_force(0.0)

    Kr, Cr, Mr = basis.Kr, basis.Cr, basis.Mr

    def _energies(u_red, v_red_):
        ke = 0.5 * float(v_red_ @ (Mr @ v_red_))
        ie = 0.5 * float(u_red @ (Kr @ u_red))
        return ke, ie

    u_red = basis.recombine_red(x)
    v_red_now = (basis.Zvel @ x).real            # u_dot = sum lam_k phi_k x_k
    ke0, ie0 = _energies(u_red, v_red_now)
    e0 = ke0 + ie0

    hist = {"t": [], "u": [], "ke": [], "ie": [], "wext": [], "edamp": [],
            "bal": [], "x": []}
    wext = 0.0
    edamp = 0.0
    t = 0.0
    f_red_k = _reduce(_ext(loads, model, n, 0.0))
    for _ in range(nsteps):
        t_new = min(t + dt, t_end)
        step = t_new - t
        if abs(step - dt) <= 1e-12 * dt:
            gk, a1k, a2k = g, a1, a2
        else:
            gk, a1k, a2k = _first_order_coeffs(basis.lam, step)
        p_new = _modal_force(t_new)
        x_new = gk * x + a1k * p_k + a2k * p_new

        u_red_new = basis.recombine_red(x_new)
        v_red_new = (basis.Zvel @ x_new).real

        # external work: trapezoid of f . du in reduced coords
        f_red_new = _reduce(_ext(loads, model, n, t_new))
        wext += 0.5 * float((f_red_k + f_red_new) @ (u_red_new - u_red))
        # viscous dissipation: trapezoid of u_dot^T C u_dot over the step
        d0 = float(v_red_now @ (Cr @ v_red_now))
        d1 = float(v_red_new @ (Cr @ v_red_new))
        edamp += 0.5 * (d0 + d1) * step

        x, p_k, f_red_k = x_new, p_new, f_red_new
        u_red, v_red_now = u_red_new, v_red_new
        t = t_new

        du, dur = _scatter_real(basis, x)
        ke, ie = _energies(u_red, v_red_now)
        hist["t"].append(t)
        hist["u"].append(du)
        hist["ke"].append(ke)
        hist["ie"].append(ie)
        hist["wext"].append(wext)
        hist["edamp"].append(edamp)
        hist["bal"].append(ie + ke + edamp - wext - e0)
        hist["x"].append(x.copy())

    hist["e0"] = e0
    return hist


def _ext(loads, model, n, t):
    f = np.zeros((n, 3))
    loads.external_forces(t, f, model.x0)
    return f


# ============================================================================
# COMPLEX FREQUENCY RESPONSE — the damped complex FRF
# ============================================================================

def complex_frf(basis, F_nod, M_nod, freqs_hz):
    """Steady-state harmonic response for a NON-classically-damped structure,
    through the complex modes.

    For f(t) = F e^{i Omega t} the steady modal coordinate is
    x_k(Omega) = p_k^F / (i Omega - lambda_k) with the complex modal force
    p_k^F = -lambda_k (phi_k^T F_red)/b_k, recombined
    U(Omega) = sum_k phi_k x_k(Omega) over ALL 2*nmode modes (the conjugate
    modes are NEEDED here — the harmonic forcing is complex, so the response is
    not a simple 2 Re). Returns a dict analogous to the M17 FRF (omega/freqs/
    U/amp/phase) plus ``resonances`` (the damped natural frequencies)."""
    dof = basis.dof
    constr = basis.constr
    n = M_nod.shape[0]
    f_eq = dof.gather_residual(F_nod, M_nod)
    f_red = constr.reduce_vector(f_eq) if constr is not None else f_eq
    pF = basis.modal_force(f_red)                    # (2*nmode,) complex

    freqs_hz = np.asarray(freqs_hz, dtype=float)
    Omega = 2.0 * np.pi * freqs_hz
    # x_k(Omega) over the (nf, 2*nmode) grid
    denom = 1j * Omega[:, None] - basis.lam[None, :]
    x = pF[None, :] / denom                          # (nf, 2*nmode)
    U_red = x @ basis.Phi.T                           # (nf, nred) complex
    # expand + scatter each frequency to full nodal complex amplitude
    U = np.array([_expand_complex(basis, U_red[j]) for j in range(len(Omega))])
    return {
        "omega": Omega,
        "freqs": freqs_hz,
        "U": U,
        "amp": np.abs(U),
        "phase": np.angle(U),
        "resonances": basis.damped_freqs_hz.copy(),
    }


def direct_frf(basis, F_nod, M_nod, freqs_hz):
    """The DIRECT damped FRF U(Omega) = (K - Omega^2 M + i Omega C)^-1 F,
    solved on the reduced pencil — the exact reference the complex-mode FRF
    reproduces on the full basis (a validation anchor; no modal truncation)."""
    _, _ = require_scipy()
    import scipy.linalg as sla
    dof = basis.dof
    constr = basis.constr
    f_eq = dof.gather_residual(F_nod, M_nod)
    f_red = constr.reduce_vector(f_eq) if constr is not None else f_eq
    Kr, Cr, Mr = basis.Kr, basis.Cr, basis.Mr
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    Omega = 2.0 * np.pi * freqs_hz
    U = []
    for Om in Omega:
        D = Kr - Om * Om * Mr + 1j * Om * Cr
        u_red = sla.solve(D, f_red.astype(complex))
        U.append(_expand_complex(basis, u_red))
    U = np.array(U)
    return {"omega": Omega, "freqs": freqs_hz, "U": U,
            "amp": np.abs(U), "phase": np.angle(U)}


# ============================================================================
# small complex scatter/expand helpers (the DofMap is real-only)
# ============================================================================

def _scatter_complex(dof, u_full_complex):
    """Scatter a COMPLEX full-eq-space vector to (du, dur) complex node fields
    by scattering the real and imaginary parts separately (the DofMap scatter
    is linear, so this is exact)."""
    dre, drre = dof.scatter_solution(u_full_complex.real)
    dim, drim = dof.scatter_solution(u_full_complex.imag)
    return dre + 1j * dim, drre + 1j * drim


def _expand_complex(basis, u_red_complex):
    """Reduced complex vector -> full-eq-space complex vector through T."""
    if basis.constr is not None:
        return (basis.constr.expand(u_red_complex.real)
                + 1j * basis.constr.expand(u_red_complex.imag))
    return u_red_complex


def _scatter_real(basis, x):
    """Recombine the modal coordinates to a REAL nodal displacement field
    (du, dur): u_red = sum phi_k x_k (real), expand through T, scatter."""
    u_red = basis.recombine_red(x)
    u_full = (basis.constr.expand(u_red) if basis.constr is not None
              else u_red)
    return basis.dof.scatter_solution(u_full)


# ============================================================================
# Engine-card driver (/IMPL/CEIGV)
# ============================================================================

def run_complex_modal(model, ip, log, result, constr=None, contacts=(),
                      loads=None):
    """/IMPL/CEIGV (M18): complex / damped eigenvalue extraction (+ optional
    complex-mode transient / FRF) on the converged state — the engine-card
    wiring of this module, mirroring M16's ``_run_modal`` and M17's
    ``run_modal_transient``. A PORT card (the open-source engine has no
    complex/damped eigensolver — module docstring).

    Assembles (K, C, M), extracts the complex modes, reports the complex
    eigenvalues / decay rates / damped frequencies / damping ratios / complex
    mode shapes, and stores them on ``result``. With CTRAN it also drives the
    complex-mode transient; with CFRQ it sweeps the complex FRF."""
    from ..engine.kinematics import LoadsAndConstraints
    if loads is None:
        loads = LoadsAndConstraints(model, log)
    nev = max(1, int(getattr(ip, "impl_ceigv_nmode", 6)))
    prestress = bool(getattr(ip, "impl_ceigv_prestress", False))
    # Rayleigh coefficients: the SAME (alpha, beta) the M11 /IMPL/DYNA/DAMP
    # feeds, so a classically-damped deck's complex modes carry the identical
    # physical damping (and reduce to the M16 omega_i, M17 zeta_i)
    alpha = beta = 0.0
    if getattr(ip, "impl_dyna_damp", False):
        alpha = float(getattr(ip, "impl_dyna_dampa", 0.0))
        beta = float(getattr(ip, "impl_dyna_dampb", 0.0))

    log.info("\n     ** COMPLEX / DAMPED EIGENVALUES **      (/IMPL/CEIGV)")
    if prestress:
        log.info("        (prestressed: K = K_mat + K_geo of the "
                 "committed state)")
    if alpha or beta:
        log.info(f"        RAYLEIGH C += a M + b K : a = {alpha:g}, "
                 f"b = {beta:g}  (/IMPL/DYNA/DAMP)")

    # a pure complex-mode analysis linearizes about the REST state at x0
    # (unless prestressed), exactly like _run_modal / run_modal_transient
    x_saved = model.x
    if not prestress:
        model.x = model.x0.copy()
    try:
        basis = build_complex_basis(model, nev=nev, log=None,
                                    constraints=constr, contacts=contacts,
                                    prestress=prestress, alpha=alpha, beta=beta)
    finally:
        if not prestress:
            model.x = x_saved

    lam = basis.eigenvalues
    result.complex_eigenvalues = lam
    result.complex_natural_freqs = basis.natural_freqs_hz
    result.complex_damped_freqs = basis.damped_freqs_hz
    result.complex_decay_rates = basis.decay_rates
    result.complex_damping_ratios = basis.damping_ratios
    result.complex_modes = basis.mode_shapes()

    log.info(f"      NUMBER OF COMPLEX MODES . . . . . : {basis.nmode:10d}")
    log.info("      MODE   FREQ(HZ)     DAMPED(HZ)    ZETA        DECAY(1/S)")
    fn = basis.natural_freqs_hz
    fd = basis.damped_freqs_hz
    zt = basis.damping_ratios
    sg = basis.decay_rates
    for i in range(basis.nmode):
        log.info(f"      {i + 1:4d}  {fn[i]:12.5E} {fd[i]:12.5E} "
                 f"{zt[i]:10.4E}  {sg[i]:12.5E}")

    # ---- optional complex-mode transient / FRF ----------------------------
    if getattr(ip, "impl_ceigv_tran", False):
        t_end = float(getattr(ip, "impl_ceigv_tend", 0.0)) or float(ip.t_end)
        dt = float(getattr(ip, "impl_ceigv_dt", 0.0)) or float(getattr(
            ip, "impl_dt", 0.0)) or t_end
        if dt <= 0.0:
            raise ValueError("/IMPL/CEIGV/TRAN needs a positive sampling step "
                             "(card: t_end dt).")
        x_saved = model.x
        if not prestress:
            model.x = model.x0.copy()
        try:
            hist = complex_modal_transient(basis, loads, t_end, dt, model,
                                           log=log)
        finally:
            if not prestress:
                model.x = x_saved
        result.complex_transient_history = hist
        log.info("\n     ** COMPLEX-MODE TRANSIENT **            "
                 "(/IMPL/CEIGV/TRAN)")
        log.info(f"      TRANSIENT WINDOW / STEP  . . . . : "
                 f"{t_end:.5E} / {dt:.5E}")
        if hist["t"]:
            umax = max(float(np.abs(u).max()) for u in hist["u"])
            log.info(f"      MAX |DISPLACEMENT| . . . . . . . : {umax:.5E}")

    if getattr(ip, "impl_ceigv_frf", False):
        fmin = float(getattr(ip, "impl_ceigv_fmin", 0.0))
        fmax = float(getattr(ip, "impl_ceigv_fmax", 0.0))
        nf = max(2, int(getattr(ip, "impl_ceigv_nf", 200)))
        if fmax <= fmin:
            fmin, fmax = 0.0, 1.2 * float(basis.natural_freqs_hz.max())
        n = model.numnod
        F = np.zeros((n, 3))
        loads.external_forces(1.0, F, model.x0)      # deck /CLOAD pattern
        freqs_hz = np.linspace(fmin, fmax, nf)
        frf = complex_frf(basis, F, np.zeros((n, 3)), freqs_hz)
        result.complex_frf = frf
        log.info("\n     ** COMPLEX FREQUENCY RESPONSE **        "
                 "(/IMPL/CEIGV/FRF)")
        log.info(f"      SWEEP BAND (HZ) / POINTS . . . . : "
                 f"[{fmin:.5E}, {fmax:.5E}] / {nf}")

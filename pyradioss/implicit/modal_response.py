"""
Modal-superposition dynamics — M17: transient response history, harmonic /
frequency response, and modal damping.

Fortran origin
--------------
There is NO modal-superposition or frequency-response path in the open-source
OpenRadioss engine. ``engine/source/input/freimpl.F`` (the /IMPL reader, read
line by line for M17) parses only /IMPL/DYNA (imp_dyna.F — the DIRECT
Newmark/HHT time integrator M10 ports), /IMPL/BUCKL (imp_buck.F), /IMPL/DT,
/IMPL/NONLIN and /IMPL/ARCL; there is no /FREQ, no /IMPL/MODAL, no
mode-superposition transient, and no harmonic-response driver anywhere in the
open tree. OpenRadioss is a time-domain crash/impact code — the frequency
domain simply is not part of the open-source solver.

So M17 does exactly what M16 did for the modal eigensolver: it ports
modal-superposition transient + harmonic response as clean LIBRARY
capabilities that CONSUME the M16 eigenpairs (``implicit/modal.py``), and adds
minimal PORT engine cards to drive them (/IMPL/MODAL/DYNA and /IMPL/FREQ — see
``statics._run_modal_transient`` / ``_run_freqresponse``), the same way M11
added the thin /IMPL/BUCKL card over the M9 buckling library and M16 added
/IMPL/EIGV over this module's eigensolver. Nothing in the DIRECT M10
Newmark/HHT integrator or the M16 eigensolver is touched — modal
superposition is a NEW, parallel path.

Theory — mode superposition (Clough & Penzien, "Dynamics of Structures",
ch. 12-13; Chopra, "Dynamics of Structures", ch. 10-13; Craig & Kurdila,
"Fundamentals of Structural Dynamics", ch. 11)
-----------------------------------------------------------------------------
The M16 eigensolver returns the natural angular frequencies omega_i and the
MASS-NORMALIZED mode shapes phi_i of the undamped structure:

    (K - omega_i^2 M) phi_i = 0 ,     phi_i^T M phi_j = delta_ij ,
                                      phi_i^T K phi_j = omega_i^2 delta_ij

Expanding the physical response in the modal basis, u(t) = sum_i phi_i q_i(t),
and pre-multiplying the semi-discrete equation of motion M u_ddot + C u_dot +
K u = f(t) by phi_i^T, the M- and K-orthogonality of the modes DECOUPLES the
system into independent single-DOF (SDOF) oscillators — one per mode:

    q_i_ddot + 2 zeta_i omega_i q_i_dot + omega_i^2 q_i = r_i(t) ,
    r_i(t) = phi_i^T f(t)            (the "modal" / generalized force)

PROVIDED the damping is CLASSICAL (phi_i^T C phi_j diagonal — Rayleigh damping
C = a M + b K is, and so is any modal damping assigned per mode). Each modal
coordinate q_i is then a damped SDOF with unit modal mass, natural frequency
omega_i and damping ratio zeta_i. The physical response recombines
u(t) = sum_i phi_i q_i(t) (a few low modes usually suffice — the truncation
question the mode-acceleration correction below addresses).

This module delivers, in the milestone's build order:

1. MODAL DAMPING (``modal_damping`` + helpers): the per-mode zeta_i, from
   (a) a uniform zeta, (b) a table (frequency, zeta) interpolated onto the
   extracted omega_i, or (c) the Rayleigh map zeta_i = 1/2 (a/omega_i +
   b omega_i) so a /IMPL/DYNA/DAMP a,b becomes modal damping CONSISTENTLY
   with the M11 direct-integration Rayleigh damping (same a, b, same C).
2. MODAL TRANSIENT (``modal_transient``): project f(t) onto the modes, march
   each decoupled SDOF with the EXACT piecewise-linear-forcing recurrence
   (Nigam-Jennings — see below), recombine u(t). The M10 energy ledger is
   reproduced in modal coordinates.
3. HARMONIC / FREQUENCY RESPONSE (``modal_frequency_response``): the complex
   modal FRF q_i(Omega) = (phi_i^T F) / (omega_i^2 - Omega^2 +
   2 i zeta_i omega_i Omega), swept over a band, recombined to the complex
   transfer function u(Omega) = sum_i phi_i q_i(Omega), reported as amplitude
   and phase.

The exact recurrence (Nigam & Jennings 1969; Chopra Table 5.2.1)
----------------------------------------------------------------
For the SDOF q_ddot + 2 zeta omega q_dot + omega^2 q = r(t) with r(t)
LINEARLY INTERPOLATED between the sampled instants t_k (r piecewise linear —
the natural assumption for a sampled load), the step from t_k to t_{k+1} =
t_k + dt has a CLOSED-FORM recurrence

    [ q_{k+1} ]   [ A  B ] [ q_k ]   [ C  D ] [ r_k     ]
    [ qd_{k+1}] = [ A' B'] [ qd_k] + [ C' D'] [ r_{k+1} ]

whose eight coefficients (functions of omega, zeta, dt only) are given in
``_nigam_jennings_coeffs`` below. This is EXACT for piecewise-linear forcing
at ANY step size (no period elongation, no amplitude error) — the modal
transient reproduces the closed-form SDOF response to round-off, which is why
the M17 validation can assert the impulse/step response POINTWISE rather than
"to O(dt^2)" as the direct Newmark integrator must. (The alternative, a
per-mode Newmark march, would inherit the (omega dt)^2/12 dispersion of the
direct integrator; the exact recurrence is chosen deliberately — the whole
point of decoupling into SDOFs is that each one then has a closed form.)
Requires under-critical modes (zeta < 1); an over-damped modal ratio (rare
for a structure) is refused loudly rather than silently mis-integrated.

Mode-acceleration / residual-flexibility correction (documented deviation)
--------------------------------------------------------------------------
Truncating the sum at N modes drops the quasi-static contribution of the
higher modes to a low-frequency load — the "mode-displacement" recombination
u ~= sum_{i<=N} phi_i q_i converges only as fast as the modal forces decay.
The MODE-ACCELERATION correction (Craig & Kurdila sec. 11.6; Cornwell,
Craig & Johnson 1983) adds back the static response of the truncated tail:

    u(t) = sum_{i<=N} phi_i q_i(t)
         + [ K^-1 - sum_{i<=N} phi_i phi_i^T / omega_i^2 ] f(t)

The bracket is the RESIDUAL FLEXIBILITY (the exact static flexibility K^-1
minus the part already captured by the retained modes). It makes the retained
modes carry only the DYNAMIC amplification while the truncated tail is
represented STATICALLY — exact for a static load, and dramatically faster
convergence for a low-frequency transient. It costs one extra linear solve
K u = f per step (the same factorization the M8 statics uses). This is a
deliberate, documented ADDITION over the plain mode-displacement sum: it is
off by default (mode_acceleration=False) and enabled by /IMPL/MODAL/DYNA's
MACC flag; the M17 validation shows the truncation error collapsing when it
is on.

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* REAL modes only. The port superposes the M16 REAL (undamped, symmetric)
  eigenmodes and assumes CLASSICAL damping (diagonal modal C). Non-classical
  damping needs the complex (state-space / quadratic-eigenvalue) modes, which
  M16 already defers — so this module defers them too (a heavily/locally
  damped structure whose C is not a-M-plus-b-K would have coupled modal
  equations the SDOF decoupling does not capture).
* LIBRARY-FIRST cards. /IMPL/MODAL/DYNA and /IMPL/FREQ are PORT cards (no
  upstream equivalent, as established above) — minimal, exactly like M16's
  /IMPL/EIGV.
* Random / spectral (PSD) response, response spectra, base-excitation beyond
  the simple participation-factor feed, AMLS / substructuring: DEFERRED
  (PORTING_GUIDE M17), building on this same real-modes machinery.
"""

from __future__ import annotations

import numpy as np

from . import require_scipy
from .assembly import assemble, assemble_mass
from .dofmap import DofMap, DOFS_PER_NODE
from .modal import modal_frequencies, _rigid_body_vectors


# ============================================================================
# MODAL DAMPING — per-mode zeta_i (build-order item 1)
# ============================================================================

def rayleigh_ratios(omega, alpha, beta):
    """Per-mode damping ratios of Rayleigh damping C = alpha M + beta K,
    mapped onto the extracted natural frequencies:

        zeta_i = 1/2 (alpha / omega_i + beta omega_i)

    (Clough & Penzien eq. 12-36 / Chopra eq. 11.4.9). This is the EXACT modal
    ratio a Rayleigh C produces — the mass term dominates the low modes and
    the stiffness term the high modes, with a minimum at
    omega = sqrt(alpha/beta). Feeding the SAME (alpha, beta) as the M11
    direct-integration /IMPL/DYNA/DAMP makes the modal transient and the
    direct Newmark run carry IDENTICAL physical damping — the consistency the
    M17 validation asserts (the damped decay envelope matches between the two
    solvers). ``omega`` in rad/s."""
    omega = np.asarray(omega, dtype=float)
    z = np.zeros_like(omega)
    nz = omega > 0.0
    z[nz] = 0.5 * (alpha / omega[nz] + beta * omega[nz])
    return z


def table_ratios(omega, freq_hz_tab, zeta_tab):
    """Per-mode damping ratios from a table of (frequency [Hz], zeta) pairs,
    LINEARLY INTERPOLATED onto the extracted frequencies f_i = omega_i/2pi
    (a common experimental input — a measured damping-vs-frequency curve).
    Below the first / above the last tabulated frequency the end value is held
    (np.interp's clamp), the standard flat extrapolation for a damping table.
    ``omega`` in rad/s; the table frequencies in Hz."""
    omega = np.asarray(omega, dtype=float)
    f_hz = omega / (2.0 * np.pi)
    ft = np.asarray(freq_hz_tab, dtype=float)
    zt = np.asarray(zeta_tab, dtype=float)
    order = np.argsort(ft)
    return np.interp(f_hz, ft[order], zt[order])


def modal_damping(omega, uniform=None, table=None, rayleigh=None, log=None):
    """Resolve the per-mode damping-ratio vector zeta_i for the natural
    frequencies ``omega`` (rad/s). Exactly one source is used, checked in this
    order (the /IMPL/MODAL/DAMP card precedence):

    * ``uniform``  — a single zeta applied to every mode (constant modal
      damping, the textbook default);
    * ``table``    — an ``(freq_hz, zeta)`` pair of sequences interpolated
      onto omega_i (``table_ratios``);
    * ``rayleigh`` — an ``(alpha, beta)`` pair mapped by
      ``rayleigh_ratios`` (the M11-consistent Rayleigh map);
    * none of the above -> UNDAMPED (zeta = 0, a conservative free / forced
      response).

    Ratios >= 1 (over-critical) are refused: the exact SDOF recurrence and the
    complex FRF both assume an under-damped mode (a structure with an
    over-damped mode is pathological — a mis-typed zeta far more likely)."""
    omega = np.asarray(omega, dtype=float)
    if uniform is not None and float(uniform) != 0.0:
        z = np.full(omega.shape, float(uniform))
    elif table is not None:
        z = table_ratios(omega, table[0], table[1])
    elif rayleigh is not None:
        z = rayleigh_ratios(omega, rayleigh[0], rayleigh[1])
    else:
        z = np.zeros_like(omega)
    if np.any(z >= 1.0):
        bad = np.where(z >= 1.0)[0]
        raise ValueError(
            f"modal damping ratio >= 1 (over-critical) on mode(s) "
            f"{[int(b) + 1 for b in bad]} (zeta = "
            f"{[float(z[b]) for b in bad]}); the mode-superposition recurrence "
            f"and the complex FRF assume under-damped modes (zeta < 1). Check "
            f"the damping input.")
    if np.any(z < 0.0):
        raise ValueError("negative modal damping ratio — this would INJECT "
                         "energy (check alpha/beta or the damping table).")
    if log is not None:
        log.info("      MODAL DAMPING RATIOS (zeta_i): "
                 + ", ".join(f"{zz:.4E}" for zz in z))
    return z


# ============================================================================
# The modal basis in EQUATION space (consumed by transient + FRF)
# ============================================================================

class ModalBasis:
    """The M16 eigenpairs reconstructed in EQUATION space, ready for
    superposition. Built by :func:`build_modal_basis` — it re-uses
    ``modal.modal_frequencies`` unchanged (the M16 eigensolver stays
    bit-identical) and packs its (du, dur) mode fields back into the
    equation-space modal matrix ``Phi`` (ndof, nmode) through the SAME DofMap.

    Because the reported modes are mass-normalized in the reduced space and
    expanded through the constraint transform T (phi_full = T phi_red), they
    are M_full-orthonormal in the full equation space too
    (phi_full^T M_full phi_full = phi_red^T (T^T M T) phi_red = 1) — verified
    by the M16 orthogonality test. So projection/recombination can work
    entirely in full equation space:

        r_i(t) = phi_i^T f_eq(t)               (generalized / modal force)
        u_eq(t) = Phi q(t)                     (recombination)

    Attributes:
        omega   (nmode,)        natural angular frequencies (rad/s)
        freqs   (nmode,)        natural frequencies (Hz)
        Phi     (ndof, nmode)   M-orthonormal mode shapes, equation space
        dof     DofMap          the numbering Phi lives in
        M       (ndof, ndof)    the CONSISTENT mass (CSR) — for initial-state
                                projection and the effective-mass feed
        eff     (nmode, 6)      the M16 modal effective mass / participation
        constr                  the live constraint transform (or None)
    """

    def __init__(self, omega, freqs, Phi, dof, M, eff, constr):
        self.omega = omega
        self.freqs = freqs
        self.Phi = Phi
        self.dof = dof
        self.M = M
        self.eff = eff
        self.constr = constr

    @property
    def nmode(self):
        return self.Phi.shape[1]

    def project_force(self, fnod, mnod):
        """Modal (generalized) force r_i = phi_i^T f_eq for a nodal force /
        moment field (numnod, 3) each -> (nmode,)."""
        f_eq = self.dof.gather_residual(fnod, mnod)
        return self.Phi.T @ f_eq

    def project_state(self, unod, urnod):
        """Modal coordinate of a nodal displacement/velocity state:
        q_i = phi_i^T M u_eq (the M-orthonormal projection). Used to seed the
        initial modal coordinates from /INIVEL and any committed prestress
        offset."""
        u_eq = self.dof.gather_residual(unod, urnod)
        return self.Phi.T @ (self.M @ u_eq)

    def recombine(self, q):
        """Physical nodal displacement of a modal-coordinate vector q
        (nmode,) -> (du, dur) per node. u_eq = Phi q, scattered."""
        return self.dof.scatter_solution(self.Phi @ np.asarray(q))


def build_modal_basis(model, nev, log=None, constraints=None, contacts=None,
                      prestress=False):
    """Extract ``nev`` modes (reusing ``modal.modal_frequencies`` unchanged)
    and pack them into a :class:`ModalBasis` in equation space. ``prestress``,
    ``constraints`` and ``contacts`` follow ``modal_frequencies`` exactly."""
    require_scipy()
    # resolve the same constraint/contact objects modal_frequencies will use,
    # so the DofMap we build to re-pack Phi matches the one it scattered with
    if constraints is None or contacts is None:
        from ..common.messages import MessageLog
        silent = MessageLog()
        if constraints is None:
            from .constraints import build_constraints
            constraints = build_constraints(model, silent)
        if contacts is None:
            from .contact import build_implicit_contacts
            contacts = build_implicit_contacts(model, silent)

    freqs, modes, eff = modal_frequencies(
        model, nev=nev, log=log, constraints=constraints, contacts=contacts,
        prestress=prestress)
    if len(freqs) == 0:
        raise ValueError(
            "modal superposition needs at least one natural frequency, but "
            "none were extracted — the model is either a mechanism (all modes "
            "rigid) or carries no stiffness. Check the constraints.")

    # the SAME DofMap modal_frequencies used internally (built the same way),
    # so gather_residual inverts its scatter_solution exactly
    dof = DofMap(model, log, constraints=constraints)
    M = assemble_mass(model, dof, model.x0 if model.x0.size else model.x, log)

    ndof = dof.ndof
    Phi = np.zeros((ndof, len(freqs)))
    for i, (du, dur) in enumerate(modes):
        Phi[:, i] = dof.gather_residual(du, dur)

    omega = 2.0 * np.pi * np.asarray(freqs)
    return ModalBasis(omega, np.asarray(freqs), Phi, dof, M, eff, constraints)


# ============================================================================
# MODAL TRANSIENT — Nigam-Jennings exact recurrence (build-order item 2)
# ============================================================================

def _nigam_jennings_coeffs(omega, zeta, dt):
    """The eight exact recurrence coefficients (A, B, C, D, A', B', C', D')
    for the SDOF q_ddot + 2 zeta omega q_dot + omega^2 q = r(t) with r LINEAR
    over [t_k, t_k+dt], vectorized over the mode arrays ``omega``/``zeta``
    (Nigam & Jennings 1969; Chopra "Dynamics of Structures" Table 5.2.1 —
    unit mass, so k = omega^2, c = 2 zeta omega). Under-damped modes only
    (zeta < 1)."""
    omega = np.asarray(omega, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    w = omega
    w2 = w * w
    sq = np.sqrt(1.0 - zeta * zeta)         # sqrt(1 - zeta^2)
    wd = w * sq                             # damped angular frequency
    e = np.exp(-zeta * w * dt)
    s = np.sin(wd * dt)
    c = np.cos(wd * dt)
    zs = zeta / sq                          # zeta / sqrt(1 - zeta^2)

    # displacement row
    A = e * (zs * s + c)
    B = e * (s / wd)
    C = (1.0 / w2) * (
        2.0 * zeta / (w * dt)
        + e * (((1.0 - 2.0 * zeta * zeta) / (wd * dt) - zs) * s
               - (1.0 + 2.0 * zeta / (w * dt)) * c))
    D = (1.0 / w2) * (
        1.0 - 2.0 * zeta / (w * dt)
        + e * (-(1.0 - 2.0 * zeta * zeta) / (wd * dt) * s
               + 2.0 * zeta / (w * dt) * c))

    # velocity row
    Ap = -e * (w / sq) * s
    Bp = e * (c - zs * s)
    Cp = (1.0 / w2) * (
        -1.0 / dt
        + e * ((w / sq + zs / dt) * s + (1.0 / dt) * c))
    Dp = (1.0 / (w2 * dt)) * (1.0 - e * (zs * s + c))
    return A, B, C, D, Ap, Bp, Cp, Dp


def modal_transient(basis, loads, t_end, dt, zeta, model, log=None,
                    mode_acceleration=False, solver=None, v0=None, vr0=None):
    """March the mode-superposition transient response over [0, t_end] at step
    ``dt`` and return a history dict mirroring the M10 dynamics ledger.

    Each decoupled SDOF q_i_ddot + 2 zeta_i omega_i q_i_dot + omega_i^2 q_i =
    r_i(t) is integrated by the EXACT Nigam-Jennings recurrence (module
    docstring), with the modal force r_i(t_k) = phi_i^T f_eq(t_k) sampled from
    the deck loads at PHYSICAL time (``loads.external_forces`` — the same
    /CLOAD//GRAV//PLOAD machinery, evaluated at t rather than a load factor).
    The physical response recombines u(t) = sum_i phi_i q_i(t).

    Initial conditions: q_i(0) = 0 (rest), q_i_dot(0) = phi_i^T M v0 from any
    /INIVEL (mass-weighted projection onto the modes — ``v0``/``vr0`` default
    to ``model.v``/``model.vr``).

    ``mode_acceleration`` adds the residual-flexibility static correction
    (module docstring) — one extra linear solve K u = f per step (``solver``
    and the model's reduced K); documented ADDITION, off by default.

    The energy ledger is booked in MODAL coordinates (exact for the retained
    modes, M-orthonormal so the modal sums equal the physical energies):
    KE = 1/2 sum q_i_dot^2, IE = 1/2 sum omega_i^2 q_i^2, the Rayleigh/modal
    dissipation edamp = sum 2 zeta_i omega_i integral q_i_dot^2 dt, external
    work wext = trapezoid of r . dq, balance = IE + KE + edamp - wext - e0."""
    n = model.numnod
    Phi = basis.Phi
    omega = basis.omega
    nmode = basis.nmode
    zeta = np.asarray(zeta, dtype=float)
    if zeta.shape != omega.shape:
        zeta = np.broadcast_to(zeta, omega.shape).copy()

    nsteps = int(np.ceil(t_end / dt - 1e-12))
    nsteps = max(nsteps, 1)
    A, B, C, D, Ap, Bp, Cp, Dp = _nigam_jennings_coeffs(omega, zeta, dt)

    # ---- residual-flexibility static correction operator (mode-accel) ------
    # G f = [K^-1 - sum phi_i phi_i^T / omega_i^2] f, evaluated per step. The
    # K^-1 part is one reduced linear solve; the modal part is a small
    # (nmode) contraction. Built lazily below.
    macc = bool(mode_acceleration)
    if macc:
        if solver is None:
            from .linsolve import LinearSolver
            solver = LinearSolver("", log)
        dof = basis.dof
        K = assemble(model, dof, model.x0 if model.x0.size else model.x)
        constr = basis.constr
        Kred = constr.reduce_matrix(K) if constr is not None else K

    def _modal_force(t):
        fext = np.zeros((n, 3))
        loads.external_forces(t, fext, model.x0)
        return basis.project_force(fext, np.zeros((n, 3))), fext

    def _static_correction(fext):
        """[K^-1 - sum phi phi^T/omega^2] f in equation space (mode-accel)."""
        f_eq = basis.dof.gather_residual(fext, np.zeros((n, 3)))
        if basis.constr is not None:
            us = basis.constr.expand(solver.solve(Kred,
                                                  basis.constr.reduce_vector(f_eq)))
        else:
            us = solver.solve(Kred, f_eq)
        # subtract the retained-mode quasi-static part sum phi (phi^T f)/w^2
        rf = Phi.T @ f_eq
        us = us - Phi @ (rf / (omega * omega))
        return us

    # ---- initial state ------------------------------------------------------
    if v0 is None:
        v0 = model.v
    if vr0 is None:
        vr0 = model.vr
    q = np.zeros(nmode)
    qd = basis.project_state(v0, vr0)            # q_dot(0) = phi^T M v0

    r_k, fext_k = _modal_force(0.0)

    # energy seed: KE(0) + IE(0) in modal coordinates
    e0 = 0.5 * float(qd @ qd) + 0.5 * float(((omega * omega) * (q * q)).sum())

    hist = {"t": [], "u": [], "ke": [], "ie": [], "wext": [], "edamp": [],
            "bal": [], "q": []}
    wext = 0.0
    edamp = 0.0
    t = 0.0
    for _ in range(nsteps):
        t_new = min(t + dt, t_end)
        step = t_new - t
        # the recurrence coefficients are exact for the STEP length; the
        # common uniform-dt case reuses the precomputed set, a final clipped
        # step (t_end not a multiple of dt) recomputes so it stays exact too
        if abs(step - dt) <= 1e-12 * dt:
            Ak, Bk, Ck, Dk, Apk, Bpk, Cpk, Dpk = A, B, C, D, Ap, Bp, Cp, Dp
        else:
            Ak, Bk, Ck, Dk, Apk, Bpk, Cpk, Dpk = _nigam_jennings_coeffs(
                omega, zeta, step)
        r_new, fext_new = _modal_force(t_new)
        # exact SDOF step (vectorized over modes)
        q_new = Ak * q + Bk * qd + Ck * r_k + Dk * r_new
        qd_new = Apk * q + Bpk * qd + Cpk * r_k + Dpk * r_new

        # ---- energy booking (modal coordinates) ---------------------------
        # external work: trapezoid of the generalized force over dq
        wext += 0.5 * float(((r_k + r_new) * (q_new - q)).sum())
        # modal (Rayleigh/classical) dissipation: 2 zeta w integral qd^2 dt,
        # trapezoid on qd^2 over the step
        edamp += float((2.0 * zeta * omega
                        * 0.5 * (qd * qd + qd_new * qd_new)
                        * (t_new - t)).sum())

        q, qd, r_k, fext_k = q_new, qd_new, r_new, fext_new
        t = t_new

        du, dur = basis.recombine(q)
        if macc:
            us = _static_correction(fext_k)
            dus, durs = basis.dof.scatter_solution(us)
            du = du + dus
            dur = dur + durs

        ke = 0.5 * float(qd @ qd)
        ie = 0.5 * float(((omega * omega) * (q * q)).sum())
        hist["t"].append(t)
        hist["u"].append(du.copy())
        hist["ke"].append(ke)
        hist["ie"].append(ie)
        hist["wext"].append(wext)
        hist["edamp"].append(edamp)
        hist["bal"].append(ie + ke + edamp - wext - e0)
        hist["q"].append(q.copy())

    hist["e0"] = e0
    return hist


# ============================================================================
# HARMONIC / FREQUENCY RESPONSE — the complex FRF (build-order item 3)
# ============================================================================

def modal_frequency_response(basis, F_nod, M_nod, freqs_hz, zeta,
                             base_excitation=False, base_dir=None):
    """Steady-state harmonic response over a frequency sweep.

    For a harmonic force f(t) = F e^{i Omega t} the steady modal coordinate is
    the complex FRF

        q_i(Omega) = (phi_i^T F) / (omega_i^2 - Omega^2 + 2 i zeta_i omega_i
                                    Omega)

    (the classic damped-SDOF transfer function — Chopra eq. 3.2.3), and the
    physical complex response recombines u(Omega) = sum_i phi_i q_i(Omega).
    At an undamped resonance Omega -> omega_i the denominator collapses to
    2 i zeta_i omega_i^2, giving the resonant amplification 1/(2 zeta_i) that
    the SDOF-peak validation checks, with the half-power bandwidth
    Delta_Omega/omega = 2 zeta.

    Parameters
    ----------
    F_nod, M_nod : (numnod, 3)
        the real (or complex) harmonic force / moment amplitude pattern.
    freqs_hz : (nf,)
        the forcing frequencies (Hz) to sweep.
    zeta : (nmode,)
        the per-mode damping ratios.
    base_excitation : bool
        if True, the excitation is a rigid-base ACCELERATION in direction
        ``base_dir`` (a 0..5 rigid direction index): the modal force becomes
        r_i = -(phi_i^T M r_dir) = -Gamma_i, the M16 participation factor —
        so the effective-mass participation feeds the FRF directly (a shaker
        table). ``F_nod`` is then ignored.

    Returns a dict:
        omega   (nf,)            swept angular frequencies (rad/s)
        freqs   (nf,)            swept frequencies (Hz)
        q       (nf, nmode)      complex modal FRF
        U       (nf, ndof)       complex physical FRF (equation space)
        amp     (nf, ndof)       |U| amplitude
        phase   (nf, ndof)       arg(U) phase (rad)
        resonances (nmode,)      the natural frequencies (Hz) — FRF peaks
    """
    omega = basis.omega
    Phi = basis.Phi
    zeta = np.broadcast_to(np.asarray(zeta, float), omega.shape)
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    Omega = 2.0 * np.pi * freqs_hz

    if base_excitation:
        # r_i = -phi_i^T M r_dir = -Gamma_i (participation factor); reuse the
        # M16 rigid-body influence vector of the requested direction
        d = 0 if base_dir is None else int(base_dir)
        R = _rigid_body_vectors(basis.dof.model, basis.dof)
        r_static = -(Phi.T @ (basis.M @ R[:, d]))       # (nmode,) real
    else:
        r_static = basis.project_force(F_nod, M_nod).astype(complex)

    # the complex modal FRF q_i(Omega), vectorized over the (nf, nmode) grid
    denom = (omega * omega)[None, :] - (Omega * Omega)[:, None] \
        + 2j * (zeta * omega)[None, :] * Omega[:, None]
    q = r_static[None, :] / denom                        # (nf, nmode)

    U = q @ Phi.T                                        # (nf, ndof) complex
    return {
        "omega": Omega,
        "freqs": freqs_hz,
        "q": q,
        "U": U,
        "amp": np.abs(U),
        "phase": np.angle(U),
        "resonances": basis.freqs.copy(),
    }


# ============================================================================
# Engine-card drivers (the /IMPL/MODAL/DYNA and /IMPL/FREQ wiring)
# ============================================================================

def _resolve_modal_zeta(basis, ip, log):
    """Per-mode zeta from the controls, in the /IMPL/MODAL/DAMP precedence:
    a uniform zeta if given, else the M11-consistent Rayleigh map when
    /IMPL/DYNA/DAMP supplied (alpha, beta), else undamped."""
    uniform = float(getattr(ip, "impl_modal_zeta", 0.0))
    rayleigh = None
    if getattr(ip, "impl_dyna_damp", False):
        rayleigh = (float(getattr(ip, "impl_dyna_dampa", 0.0)),
                    float(getattr(ip, "impl_dyna_dampb", 0.0)))
    return modal_damping(basis.omega, uniform=(uniform or None),
                         rayleigh=rayleigh, log=log)


def run_modal_transient(model, ip, log, result, constr=None, contacts=(),
                        loads=None, solver=None):
    """/IMPL/MODAL/DYNA (M17): mode-superposition transient response history.
    Extract the modes (reuse modal.py), project the deck loads onto them,
    integrate the decoupled SDOFs with the exact recurrence and recombine —
    storing the response history on ``result`` (mirroring how ``_run_modal``
    stores the eigenpairs). A PORT card: the open-source engine has no
    mode-superposition path (module docstring)."""
    from ..engine.kinematics import LoadsAndConstraints
    if loads is None:
        loads = LoadsAndConstraints(model, log)
    nev = max(1, int(getattr(ip, "impl_modal_nmode", 6)))
    prestress = bool(getattr(ip, "impl_modal_prestress", False))
    macc = bool(getattr(ip, "impl_modal_macc", False))
    t_end = float(getattr(ip, "impl_modal_tend", 0.0)) or float(ip.t_end)
    dt = float(getattr(ip, "impl_modal_dt", 0.0)) or float(getattr(
        ip, "impl_dt", 0.0)) or t_end
    if dt <= 0.0:
        raise ValueError("/IMPL/MODAL/DYNA needs a positive sampling step "
                         "(card: t_end dt) or an /IMPL/DTINI value.")

    log.info("\n     ** MODAL TRANSIENT (mode superposition) **   "
             "(/IMPL/MODAL/DYNA)")
    if prestress:
        log.info("        (prestressed modes: K = K_mat + K_geo of the "
                 "committed state)")
    # A pure (non-prestressed) transient linearizes and starts from the REST
    # state at x0: the modal response u(t)=sum phi q(t) is measured from the
    # unloaded reference regardless of what the (usually trivial) static step
    # left in model.x, so extract the modes on x0. Prestress keeps the
    # committed (loaded/stressed) geometry.
    x_saved = model.x
    if not prestress:
        model.x = model.x0.copy()
    try:
        basis = build_modal_basis(model, nev=nev, log=None,
                                  constraints=constr, contacts=contacts,
                                  prestress=prestress)
    finally:
        if not prestress:
            model.x = x_saved
    zeta = _resolve_modal_zeta(basis, ip, log)
    hist = modal_transient(basis, loads, t_end, dt, zeta, model, log=log,
                           mode_acceleration=macc, solver=solver)

    result.modal_frequencies = basis.freqs
    result.modal_modes = [basis.dof.scatter_solution(basis.Phi[:, i])
                          for i in range(basis.nmode)]
    result.modal_damping = zeta
    result.modal_transient_history = hist
    log.info(f"      MODES RETAINED . . . . . . . . . : {basis.nmode}")
    log.info(f"      TRANSIENT WINDOW / STEP  . . . . : "
             f"{t_end:.5E} / {dt:.5E}")
    if macc:
        log.info("      MODE-ACCELERATION CORRECTION . . : ON "
                 "(residual flexibility)")
    if hist["t"]:
        umax = max(float(np.abs(u).max()) for u in hist["u"])
        log.info(f"      MAX |MODAL DISPLACEMENT| . . . . : {umax:.5E}")
        ref = max(abs(hist["e0"]), abs(hist["ke"][-1] + hist["ie"][-1]),
                  abs(hist["wext"][-1]), 1e-30)
        log.info(f"      ENERGY BALANCE . . . . . . . . . : "
                 f"{hist['bal'][-1] / ref * 100.0:8.2f} %")


def run_freq_response(model, ip, log, result, constr=None, contacts=(),
                      loads=None):
    """/IMPL/FREQ (M17): harmonic / steady-state frequency response. Extract
    the modes, take the deck's /CLOAD pattern as the harmonic force amplitude,
    sweep the requested band and store the complex FRF on ``result``. A PORT
    card (module docstring)."""
    from ..engine.kinematics import LoadsAndConstraints
    if loads is None:
        loads = LoadsAndConstraints(model, log)
    nev = max(1, int(getattr(ip, "impl_modal_nmode", 6)))
    prestress = bool(getattr(ip, "impl_modal_prestress", False))
    fmin = float(getattr(ip, "impl_freq_fmin", 0.0))
    fmax = float(getattr(ip, "impl_freq_fmax", 0.0))
    nf = max(2, int(getattr(ip, "impl_freq_nf", 200)))
    zeta_u = float(getattr(ip, "impl_freq_zeta", 0.02))

    log.info("\n     ** FREQUENCY RESPONSE (harmonic) **          (/IMPL/FREQ)")
    x_saved = model.x
    if not prestress:
        model.x = model.x0.copy()
    try:
        basis = build_modal_basis(model, nev=nev, log=None,
                                  constraints=constr, contacts=contacts,
                                  prestress=prestress)
    finally:
        if not prestress:
            model.x = x_saved
    if fmax <= fmin:
        # default band: a little past the highest retained resonance
        fmin, fmax = 0.0, 1.2 * float(basis.freqs.max())
    zeta = modal_damping(basis.omega, uniform=(zeta_u or None), log=None)

    # the harmonic force amplitude pattern = the deck's static /CLOAD pattern
    n = model.numnod
    F = np.zeros((n, 3))
    loads.external_forces(1.0, F, model.x0)          # unit-amplitude pattern

    freqs_hz = np.linspace(fmin, fmax, nf)
    frf = modal_frequency_response(basis, F, np.zeros((n, 3)), freqs_hz, zeta)

    result.modal_frequencies = basis.freqs
    result.modal_damping = zeta
    result.freq_response = frf
    log.info(f"      MODES RETAINED . . . . . . . . . : {basis.nmode}")
    log.info(f"      SWEEP BAND (HZ) / POINTS . . . . : "
             f"[{fmin:.5E}, {fmax:.5E}] / {nf}")
    log.info("      RESONANCES (NATURAL FREQUENCIES, HZ):")
    for i, f in enumerate(basis.freqs):
        log.info(f"          {i + 1:10d}  {f:14.6E}")

"""
Modal (free-vibration eigenvalue) analysis — M16; sparse Lanczos & Sturm sequence check (M614).

Fortran origin
--------------
There is no self-contained modal CARD in the open-source OpenRadioss engine:
``engine/source/input/freimpl.F`` reads only /IMPL/DYNA (imp_dyna.F, lumped
mass) and /IMPL/BUCKL (imp_buck.F, the geometric-stiffness eigenproblem), with
no /IMPL/EIGV branch. The natural-frequency extraction of the commercial
solver runs the SAME generalized-eigenproblem machinery imp_buck.F uses for
buckling (the EIGBUCKP / Lanczos family, ``imp_lanz.F``), with the CONSISTENT MASS matrix
in place of the geometric stiffness.

This module provides:
* Dense ``scipy.linalg.eigh`` generalized eigensolver for small/medium models;
* Sparse shift-and-invert Lanczos eigensolver (``scipy.sparse.linalg.eigsh``)
  mirroring ``imp_lanz.F`` / ``EIGBUCKP`` with spectral shift-and-invert:
  (K - sigma M)^-1 M phi = mu phi, where lambda = sigma + 1/mu;
* Automatic Sturm sequence check / Sylvester's inertia theorem to verify
  that no eigenvalues were missed below a target cutoff frequency;
* Computation of rigid-body modes (omega ~= 0) for unconstrained / free-free systems;
* Modal participation factors and effective modal mass fractions in all 6
  rigid-body directions (Tx, Ty, Tz, Rx, Ry, Rz), verifying mass completeness
  sum m_eff = M_total.

Theory
------
Free vibration of an undamped structure, M ü + K u = 0, has solutions
u = phi e^{i omega t}; substituting gives the symmetric generalized
eigenproblem

    (K - omega^2 M) phi = 0

for the natural angular frequencies omega and mode shapes phi. K is the
tangent stiffness (``assembly.assemble`` — the same material+hourglass tangent
the statics Newton uses, so a plastified/geometrically-nonlinear state is
linearized about its committed configuration) and M is the CONSISTENT element
mass (``assembly.assemble_mass``, M16 — NOT the lumped diagonal the explicit
and implicit-dynamics paths use). The frequency in Hz is f = omega / (2 pi).

PRESTRESSED modes (``prestress=True``): the stiffness is the FULL tangent of a
loaded/spinning state, K = K_mat + K_geo (``assemble(..., kgeo=True)`` — the
imp_kgeo initial-stress term of /IMPL/NONLIN). A tensile prestress stiffens the
structure and raises its frequencies (a tightened guitar string, a spinning
blade); a compressive prestress softens it, and the fundamental frequency
drops to zero exactly at the buckling load (the modal and buckling
eigenproblems meet there — K_mat + mu_cr K_geo is singular). Run an implicit
static analysis first to leave the committed stress in the element buffers.

Constraints and contact (the M12/M14 reduced pencil)
----------------------------------------------------
Stated on the REDUCED equations, identically to buckling:

    ( T^T K T ) phi_red = omega^2 ( T^T M T ) phi_red

with T the constraint transform (``constraints.py`` — /RBODY, /RBE2, tied,
/RBE3, /MPC and chains). Because the inertia term is reduced the SAME way as
the stiffness, a rigid body automatically carries its EXACT condensed 6-DOF
mass (total mass, parallel-axis inertia, m*skew(r) COG coupling) at the
master — the T^T M T identity the M12 dynamics already relies on, now with the
CONSISTENT element mass feeding it. Mode shapes are recovered through T
(``constraints.expand`` — the RECUKIN analogue), so a constrained mode never
violates a kinematic tie. Contact (a body resting on a closed stop) adds the
converged active set's gap tangent to K, exactly as ``buckling.py`` documents.

Reporting and Participation
---------------------------
``modal_frequencies`` returns ``(freqs_hz, modes, effective_mass)``:

* ``freqs_hz`` — natural frequencies (Hz), ascending;
* ``modes``    — MASS-NORMALIZED mode shapes (phi^T M phi = 1), each a
  ``(du, dur)`` pair of per-node (numnod, 3) translation/rotation fields;
* ``effective_mass`` — a :class:`ModalEffectiveMass` (nev, 6) array carrying:
  - ``participation_factors``: Gamma_kd = phi_k^T M r_d
  - ``mass_fractions``: f_kd = m_eff_kd / M_tot_d
  - ``total_mass``: M_tot_d = r_d^T M r_d in each of the 6 rigid-body directions
  - ``cumulative_fractions``: sum_{j<=k} f_jd
  - ``rigid_modes``: list of rigid body mode shapes (omega ~= 0)
  - ``sturm_info``: Sturm sequence verification result (if requested).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from . import require_scipy
from .assembly import assemble, assemble_mass
from .dofmap import DofMap, DOFS_PER_NODE


class ModalEffectiveMass(np.ndarray):
    """ndarray subclass representing modal effective mass (nev, 6) with metadata:
    participation_factors, mass_fractions, total_mass, cumulative_fractions,
    rigid_modes, sturm_info."""

    def __new__(cls, input_array, participation_factors=None, mass_fractions=None,
                total_mass=None, cumulative_fractions=None, rigid_modes=None,
                sturm_info=None):
        obj = np.asarray(input_array, dtype=float).view(cls)
        obj.participation_factors = (np.asarray(participation_factors, dtype=float)
                                     if participation_factors is not None else None)
        obj.mass_fractions = (np.asarray(mass_fractions, dtype=float)
                              if mass_fractions is not None else None)
        obj.total_mass = (np.asarray(total_mass, dtype=float)
                          if total_mass is not None else None)
        obj.cumulative_fractions = (np.asarray(cumulative_fractions, dtype=float)
                                    if cumulative_fractions is not None else None)
        obj.rigid_modes = rigid_modes or []
        obj.sturm_info = sturm_info
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.participation_factors = getattr(obj, "participation_factors", None)
        self.mass_fractions = getattr(obj, "mass_fractions", None)
        self.total_mass = getattr(obj, "total_mass", None)
        self.cumulative_fractions = getattr(obj, "cumulative_fractions", None)
        self.rigid_modes = getattr(obj, "rigid_modes", [])
        self.sturm_info = getattr(obj, "sturm_info", None)


@dataclass
class ModalAnalysisResult:
    """Comprehensive modal analysis results."""
    freqs_hz: np.ndarray
    modes: List[Tuple[np.ndarray, np.ndarray]]
    effective_mass: ModalEffectiveMass
    participation_factors: np.ndarray
    mass_fractions: np.ndarray
    total_mass: np.ndarray
    cumulative_fractions: np.ndarray
    rigid_modes: List[Tuple[float, Tuple[np.ndarray, np.ndarray], np.ndarray]]
    sturm_info: Optional[Dict[str, Any]] = None


def sparse_lanczos_eigen(K, M, nev=6, sigma=None, tol=0.0, which="LM", maxiter=None):
    """Sparse shift-and-invert Lanczos eigensolver mirroring ``imp_lanz.F`` / ``EIGBUCKP``.
    
    Transforms the symmetric generalized eigenproblem K phi = lambda M phi into:
        (K - sigma M)^-1 M phi = mu phi
    where lambda = sigma + 1 / mu.
    
    Parameters
    ----------
    K : scipy.sparse matrix or ndarray
        Stiffness matrix
    M : scipy.sparse matrix or ndarray
        Mass matrix (symmetric positive definite)
    nev : int
        Number of eigenvalues/eigenvectors to extract
    sigma : float, optional
        Spectral shift. If None, chooses a small negative shift to guarantee
        that (K - sigma M) is strictly positive definite even with rigid-body zero modes.
    tol : float
        Convergence tolerance for Lanczos iterations
    which : str
        ARPACK eigenvalue selection criterion ('LM' for shift-and-invert)
    maxiter : int, optional
        Maximum number of Lanczos iterations
        
    Returns
    -------
    lam : ndarray of shape (nev,)
        Eigenvalues lambda = omega^2 in ascending order
    vecs : ndarray of shape (n, nev)
        M-orthonormal eigenvectors (phi^T M phi = I)
    """
    require_scipy()
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    import scipy.linalg as sla

    n = K.shape[0]
    if nev >= n - 1 or n <= 6:
        # ARPACK requires nev < n - 1; fallback to dense eigh for small matrices
        Kd = K.toarray() if sp.issparse(K) else np.asarray(K, dtype=float)
        Md = M.toarray() if sp.issparse(M) else np.asarray(M, dtype=float)
        lam, vecs = sla.eigh(Kd, Md)
        k_take = min(nev, len(lam))
        return lam[:k_take], vecs[:, :k_take]

    if sigma is None:
        diag_k = K.diagonal() if sp.issparse(K) else np.diag(K)
        diag_pos = np.abs(diag_k[np.abs(diag_k) > 1e-12])
        k_scale = float(np.median(diag_pos)) if diag_pos.size else 1.0
        # A small negative shift guarantees (K - sigma M) is strictly positive definite
        sigma = -1e-5 * k_scale

    Ks = sp.csc_matrix(K, dtype=float)
    Ms = sp.csc_matrix(M, dtype=float)

    lam, vecs = spla.eigsh(Ks, k=nev, M=Ms, sigma=sigma, which=which,
                            tol=tol, maxiter=maxiter)

    order = np.argsort(lam)
    lam = lam[order]
    vecs = vecs[:, order]

    # Ensure exact M-orthonormality: phi_i^T M phi_i = 1.0
    for i in range(vecs.shape[1]):
        v = vecs[:, i]
        v_mv = float(v.T @ (Ms @ v))
        if v_mv > 1e-30:
            vecs[:, i] = v / np.sqrt(v_mv)

    return lam, vecs


def sturm_sequence_check(K, M, cutoff_freq_hz=None, shift=None, nev_found=None):
    """Automatic Sturm sequence check using Sylvester's law of inertia.
    
    Verifies that no eigenvalues were missed below a target cutoff frequency.
    By Sylvester's inertia theorem, the number of negative eigenvalues in the
    congruent factorization A = K - sigma M = L D L^T is EXACTLY equal to the
    number of eigenvalues of the pencil (K, M) strictly below sigma = (2*pi*f_cut)^2.
    
    Parameters
    ----------
    K : scipy.sparse matrix or ndarray
        Stiffness matrix
    M : scipy.sparse matrix or ndarray
        Mass matrix
    cutoff_freq_hz : float, optional
        Target cutoff frequency in Hz
    shift : float, optional
        Target spectral shift sigma = omega^2
    nev_found : int, optional
        Number of eigenpairs already found by the solver below the cutoff
        
    Returns
    -------
    dict with:
        "n_below": int, exact count of eigenvalues below cutoff
        "cutoff_freq_hz": float or None
        "shift": float
        "missed": int or None (n_below - nev_found if nev_found given)
        "passed": bool (True if missed == 0)
    """
    require_scipy()
    import scipy.sparse as sp
    import scipy.linalg as sla

    if cutoff_freq_hz is not None:
        sigma = float((2.0 * np.pi * cutoff_freq_hz) ** 2)
    elif shift is not None:
        sigma = float(shift)
    else:
        raise ValueError("Either cutoff_freq_hz or shift must be provided for Sturm sequence check.")

    A = K - sigma * M
    Ad = A.toarray() if sp.issparse(A) else np.asarray(A, dtype=float)

    # LDL^T factorization gives block diagonal D congruent to A
    _, d, _ = sla.ldl(Ad)
    d_eigs = np.linalg.eigvalsh(d)
    n_below = int(np.sum(d_eigs < -1e-12))

    missed = None
    passed = True
    if nev_found is not None:
        missed = max(0, n_below - int(nev_found))
        passed = (missed == 0)

    return {
        "n_below": n_below,
        "cutoff_freq_hz": cutoff_freq_hz,
        "shift": sigma,
        "missed": missed,
        "passed": passed,
    }


def modal_participation(modes, M_full, dof: DofMap, model=None, rigid=None):
    """Compute modal participation factors, effective modal masses, total masses,
    and mass fractions in all 6 rigid-body directions (Tx, Ty, Tz, Rx, Ry, Rz).
    
    Parameters
    ----------
    modes : list of (du, dur)
        Mass-normalized mode shapes (phi^T M phi = 1).
    M_full : scipy.sparse matrix or ndarray
        Full equation-space mass matrix.
    dof : DofMap
        Equation-to-node DOF mapping.
    model : Model, optional
        Model instance.
    rigid : ndarray of shape (ndof, 6), optional
        Rigid-body influence vectors.
        
    Returns
    -------
    dict with keys:
        "participation_factors": (n_modes, 6) ndarray
        "effective_mass": (n_modes, 6) ndarray
        "total_mass": (6,) ndarray
        "mass_fractions": (n_modes, 6) ndarray
        "cumulative_fractions": (n_modes, 6) ndarray
    """
    if rigid is None:
        if model is None:
            raise ValueError("Either rigid vectors or model must be provided.")
        rigid = _rigid_body_vectors(model, dof)

    n_modes = len(modes)
    if n_modes == 0:
        return {
            "participation_factors": np.zeros((0, 6)),
            "effective_mass": np.zeros((0, 6)),
            "total_mass": np.zeros(6),
            "mass_fractions": np.zeros((0, 6)),
            "cumulative_fractions": np.zeros((0, 6)),
        }

    total_mass = np.zeros(6)
    for d in range(6):
        rd = rigid[:, d]
        total_mass[d] = float(rd @ (M_full @ rd))

    part_factors = np.zeros((n_modes, 6))
    eff_mass = np.zeros((n_modes, 6))

    for k, (du, dur) in enumerate(modes):
        phi_full = dof.gather_residual(du, dur)
        Mphi = M_full @ phi_full
        for d in range(6):
            gamma_kd = float(rigid[:, d] @ Mphi)
            part_factors[k, d] = gamma_kd
            eff_mass[k, d] = gamma_kd ** 2

    safe_tot = np.where(total_mass > 1e-15, total_mass, 1.0)
    mass_fractions = np.where(total_mass > 1e-15, eff_mass / safe_tot, 0.0)
    cumulative_fractions = np.cumsum(mass_fractions, axis=0)

    return {
        "participation_factors": part_factors,
        "effective_mass": eff_mass,
        "total_mass": total_mass,
        "mass_fractions": mass_fractions,
        "cumulative_fractions": cumulative_fractions,
    }


def verify_modal_mass_completeness(effective_mass, total_mass, rtol=1e-3, atol=1e-8):
    """Verify that sum m_eff = M_total in all active rigid-body directions (Tx, Ty, Tz, Rx, Ry, Rz).
    
    Returns (passed, relative_error_vector).
    """
    sum_eff = np.sum(np.asarray(effective_mass, dtype=float), axis=0)
    total = np.asarray(total_mass, dtype=float)
    active = total > atol
    if not np.any(active):
        return True, np.zeros(6)
    rel_err = np.zeros(6)
    rel_err[active] = np.abs(sum_eff[active] - total[active]) / total[active]
    passed = bool(np.all(rel_err[active] <= rtol))
    return passed, rel_err


def compute_rigid_body_modes(model, constraints=None, contacts=None, dof=None,
                             M_full=None, tol=1e-8):
    """Compute rigid body modes (zero-frequency modes omega ~= 0) of the model.
    
    Returns
    -------
    rigid_freqs_hz : ndarray of shape (n_rigid,)
        Frequencies (all 0.0 Hz)
    rigid_modes : list of (du, dur)
        Mass-normalized rigid-body mode shapes
    rigid_effective_mass : ndarray of shape (n_rigid, 6)
        Effective mass in all 6 rigid directions
    participation_factors : ndarray of shape (n_rigid, 6)
        Modal participation factors
    """
    require_scipy()
    import scipy.linalg as sla

    x_geom = model.x if model.x.size else model.x0
    if constraints is None or contacts is None:
        from ..common.messages import MessageLog
        silent = MessageLog()
        if constraints is None:
            from .constraints import build_constraints
            constraints = build_constraints(model, silent)
        if contacts is None:
            from .contact import build_implicit_contacts
            contacts = build_implicit_contacts(model, silent)

    if not constraints:
        constraints = None

    if dof is None:
        dof = DofMap(model, constraints=constraints)

    K = assemble(model, dof, x_geom, kgeo=False)
    if M_full is None:
        M_full = assemble_mass(model, dof, model.x0 if model.x0.size else x_geom)
    M = M_full

    if constraints is not None:
        constraints.build(dof, x_geom)
        K = constraints.reduce_matrix(K)
        M = constraints.reduce_matrix(M)

    Kd = K.toarray()
    Md = M.toarray()
    lam, vecs = sla.eigh(Kd, Md)

    lam_pos = lam[lam > 1e-12]
    lam_scale = float(np.median(lam_pos)) if lam_pos.size else 1.0
    rigid_tol = tol * lam_scale

    rigid_indices = np.where(lam <= rigid_tol)[0]
    rigid_modes = []
    rigid_freqs = []

    rigid = _rigid_body_vectors(model, dof)

    for idx in rigid_indices:
        phi_red = vecs[:, idx]
        phi_full = constraints.expand(phi_red) if constraints is not None else phi_red
        du, dur = dof.scatter_solution(phi_full)
        rigid_modes.append((du, dur))
        rigid_freqs.append(0.0)

    part_info = modal_participation(rigid_modes, M_full, dof, model=model, rigid=rigid)
    return (np.array(rigid_freqs), rigid_modes, part_info["effective_mass"],
            part_info["participation_factors"])


def modal_frequencies(model, nev=6, log=None, constraints=None, contacts=None,
                      prestress=False, sparse=False, sigma=None, tol=0.0,
                      check_sturm=False, cutoff_freq=None, include_rigid=False,
                      return_details=False):
    """Natural frequencies (Hz) and mode shapes from the consistent-mass generalized
    eigenproblem (K - omega^2 M) phi = 0 (see module docstring).
    
    Supports both dense ``scipy.linalg.eigh`` and sparse shift-and-invert Lanczos
    (``imp_lanz.F`` / ``scipy.sparse.linalg.eigsh``).
    
    Parameters
    ----------
    model : Model
        Model to analyze
    nev : int
        Number of modes to extract (default 6)
    log : MessageLog, optional
        Log instance
    constraints : Constraints, optional
        Kinematic constraints
    contacts : list, optional
        Active contact interfaces
    prestress : bool
        If True, includes geometric initial-stress stiffness K_geo
    sparse : bool
        If True, uses sparse shift-and-invert Lanczos eigensolver (scipy.sparse.linalg.eigsh)
    sigma : float, optional
        Spectral shift for shift-and-invert Lanczos
    tol : float
        Tolerance for eigensolver
    check_sturm : bool
        If True, runs automatic Sturm sequence check below cutoff_freq
    cutoff_freq : float, optional
        Target cutoff frequency (Hz) for Sturm sequence check
    include_rigid : bool
        If True, includes rigid body modes (omega ~= 0) in the returned frequencies
    return_details : bool
        If True, returns a :class:`ModalAnalysisResult` dataclass; if False,
        returns (freqs_hz, modes, effective_mass) where effective_mass is
        a :class:`ModalEffectiveMass` ndarray carrying participation metadata.
    """
    require_scipy()
    import scipy.linalg as sla

    x_geom = model.x if model.x.size else model.x0
    if constraints is None or contacts is None:
        from ..common.messages import MessageLog
        silent = MessageLog()
        if constraints is None:
            from .constraints import build_constraints
            constraints = build_constraints(model, silent)
        if contacts is None:
            from .contact import build_implicit_contacts
            contacts = build_implicit_contacts(model, silent)

    if not constraints:
        constraints = None

    dof = DofMap(model, log, constraints=constraints)
    K = assemble(model, dof, x_geom, kgeo=prestress)
    if contacts:
        from .contact import contact_tangent
        Kc = contact_tangent(contacts, x_geom, dof)
        K = K + 0.5 * (Kc + Kc.T)
    M = assemble_mass(model, dof, model.x0 if model.x0.size else x_geom, log)

    M_full = M
    if constraints is not None:
        constraints.build(dof, x_geom)
        K = constraints.reduce_matrix(K)              # T^T K T
        M = constraints.reduce_matrix(M)              # T^T M T

    nred = K.shape[0]

    # Eigensolve: sparse Lanczos or dense eigh
    if sparse and nred > 6 and nev < nred - 1:
        # Request extra modes to cover any rigid body modes when include_rigid=False
        k_req = min(nev + (0 if include_rigid else 6), nred - 2)
        lam, vecs = sparse_lanczos_eigen(K, M, nev=k_req, sigma=sigma, tol=tol)
    else:
        Kd = K.toarray()
        Md = M.toarray()
        lam, vecs = sla.eigh(Kd, Md)

    # RIGID-BODY / MECHANISM threshold
    lam_pos = lam[lam > 1e-12]
    lam_scale = float(np.median(lam_pos)) if lam_pos.size else 1.0
    rigid_tol = 1e-8 * lam_scale

    rigid = _rigid_body_vectors(model, dof)

    all_modes = []
    all_freqs = []
    rigid_modes_list = []

    for k in range(len(lam)):
        lk = lam[k]
        phi_red = vecs[:, k]
        phi_full = (constraints.expand(phi_red)
                    if constraints is not None else phi_red)
        du, dur = dof.scatter_solution(phi_full)
        Mphi = M_full @ phi_full
        eff_k = np.array([(rigid[:, d] @ Mphi) ** 2 for d in range(6)])

        is_rigid = (lk <= rigid_tol) or (abs(lk) <= 1e-12)
        if is_rigid:
            rigid_modes_list.append((0.0, (du, dur), eff_k))
            if include_rigid:
                all_freqs.append(0.0)
                all_modes.append((du, dur))
        else:
            omega = np.sqrt(max(0.0, lk))
            f_hz = omega / (2.0 * np.pi)
            all_freqs.append(f_hz)
            all_modes.append((du, dur))

        if len(all_freqs) >= nev:
            break

    freqs_arr = np.array(all_freqs[:nev])
    modes_list = all_modes[:nev]

    part_info = modal_participation(modes_list, M_full, dof, model=model, rigid=rigid)

    sturm_info = None
    if check_sturm or cutoff_freq is not None:
        c_freq = cutoff_freq if cutoff_freq is not None else (freqs_arr[-1] if len(freqs_arr) else 100.0)
        sturm_info = sturm_sequence_check(K, M, cutoff_freq_hz=c_freq, nev_found=len(freqs_arr))

    eff_obj = ModalEffectiveMass(
        part_info["effective_mass"],
        participation_factors=part_info["participation_factors"],
        mass_fractions=part_info["mass_fractions"],
        total_mass=part_info["total_mass"],
        cumulative_fractions=part_info["cumulative_fractions"],
        rigid_modes=rigid_modes_list,
        sturm_info=sturm_info,
    )

    if log is not None and len(freqs_arr):
        log.info(" NATURAL FREQUENCIES (HZ) . . . . . . : "
                 + ", ".join(f"{f:.6E}" for f in freqs_arr))

    if return_details:
        return ModalAnalysisResult(
            freqs_hz=freqs_arr,
            modes=modes_list,
            effective_mass=eff_obj,
            participation_factors=part_info["participation_factors"],
            mass_fractions=part_info["mass_fractions"],
            total_mass=part_info["total_mass"],
            cumulative_fractions=part_info["cumulative_fractions"],
            rigid_modes=rigid_modes_list,
            sturm_info=sturm_info,
        )

    return freqs_arr, modes_list, eff_obj


def _rigid_body_vectors(model, dof: DofMap):
    """The 6 rigid-body influence vectors (ndof, 6) about the origin — the
    directions the modal effective mass is projected on. Columns 0-2 = unit
    translations, 3-5 = unit rotations (translation arm r x e_d + the unit
    rotation itself). Only DOFs with an equation are filled."""
    n = model.numnod
    x = model.x0 if model.x0.size else model.x
    R = np.zeros((dof.ndof, 6))
    node = np.arange(n)
    eq = dof.eq
    tr_eq = np.stack([eq[node * DOFS_PER_NODE + c] for c in range(3)], axis=1)
    rot_eq = np.stack([eq[node * DOFS_PER_NODE + 3 + c] for c in range(3)],
                      axis=1)
    for c in range(3):
        act = tr_eq[:, c] >= 0
        R[tr_eq[act, c], c] = 1.0                      # unit translation
    for d in range(3):
        e = np.zeros(3); e[d] = 1.0
        arm = np.cross(np.broadcast_to(e, (n, 3)), x)  # (n,3) = e_d x r
        for c in range(3):
            act = tr_eq[:, c] >= 0
            R[tr_eq[act, c], 3 + d] += arm[act, c]
        act = rot_eq[:, d] >= 0
        R[rot_eq[act, d], 3 + d] += 1.0
    return R

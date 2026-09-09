"""
LAW3 — elastic-plastic hydrodynamic (/MAT/LAW3, /MAT/PLAS_BOST).

Fortran origin: ``engine/source/materials/mat/mat003/m3law.F`` (solids)
and ``m3law8.F`` (shells — the port maps it onto the same Iplas=2 radial
projection the LAW2/LAW36/LAW44 shell ports use), constants from
``starter/source/materials/mat/mat003/hm_read_mat03.F``.

Theory
------
J2 (von Mises) plasticity with **power-law isotropic hardening**::

    sigma_y(eps_p) = min(sig_max,  A + B * eps_p^N)

where A is the initial yield stress, B is the hardening coefficient, and
N is the hardening exponent.  No strain-rate sensitivity (that's LAW44's
Cowper-Symonds addition to the same base model).

The Fortran clamps ``N = 1.0001`` when the input is 0 or 1 (to keep the
power-law derivative finite), ``eps_max = 1e20`` when zero, ``sig_max =
1e20`` when zero — reproduced here exactly.

The stress integration is the classic **radial return** (Wilkins 1964)
identical in structure to the LAW2 port:

1. *Elastic trial*: integrate deviatoric strain increment elastically.
2. Compute von Mises equivalent ``sig_eq = sqrt(3 J2)``.
3. Yield stress ``YLD = min(sig_max, A + B * eps_p^N)``.
4. If ``sig_eq > YLD``: scale deviator by ``R = YLD / sig_eq``, update
   plastic strain ``dpla = (1 - R) * sig_eq / (3G + QH)`` with QH the
   hardening modulus ``d(sigma_y)/d(eps_p)``.

Hardening modulus QH (m3law.F lines 123-139)::

    N == 1:  QH = B
    N >  1:  QH = B * N * eps_p^(N-1)
    N <  1:  QH = B * N / eps_p^(1-N)   [eps_p > 0; QH = 0 otherwise]

Sound speed: ``c = sqrt((dP/drho + 4G/3) / rho0)`` — the law returns
``c = None`` and lets the kernel use the constant elastic estimate.

Ductile failure: not handled in the constitutive law; the Fortran
``OFF *= 0.8 when EPSP >= EPSM`` is the kernel's element deletion
plumbing (already implemented in the pyradioss element kernels).
"""

from __future__ import annotations

import numpy as np

from ..model.entities import Material

_EM15 = 1e-15
_INF = 1e20


# ===================================================================
# Physics constructor
# ===================================================================

def build_law03(rec) -> Material:
    """Physics constructor for the cfg-parsed /MAT/LAW3 record (cfg
    ``matl3_plas_bost.cfg`` / hm_read_mat03.F)."""
    if isinstance(rec, dict):
        p = rec.get("params", rec)
        _id = rec.get("id", 0)
        _rho = float(rec.get("density", rec.get("rho0", 1.0)))
        _title = rec.get("title", "LAW3")
    else:
        _id = rec.id
        _rho = rec.rho0
        _title = getattr(rec, "title", "LAW3")
        p = rec.params

    e = float(p.get("MAT_E") or p.get("E") or 0.0)
    nu = float(p.get("MAT_NU") or p.get("nu") or 0.0)
    if e <= 0.0:
        raise ValueError(f"/MAT/LAW3/{_id}: Young modulus E must be > 0")
    if not (0.0 <= nu < 0.5):
        raise ValueError(f"/MAT/LAW3/{_id}: Poisson ratio nu={nu:g} "
                         f"outside [0, 0.5)")

    g = e / (2.0 * (1.0 + nu))
    k = e / (3.0 * (1.0 - 2.0 * nu))

    a = float(p.get("MAT_SIGY") or p.get("A") or 0.0)
    b = float(p.get("MAT_BETA") or p.get("B") or 0.0)
    n = float(p.get("MAT_HARD") or p.get("N") or 0.0)
    eps_max = float(p.get("MAT_EPS") or p.get("eps_max") or 0.0)
    sig_max = float(p.get("MAT_SIG") or p.get("sig_max") or 0.0)

    # Fortran defaults (hm_read_mat03.F lines 186-188)
    if n == 0.0 or n == 1.0:
        n = 1.0001
    if eps_max == 0.0:
        eps_max = _INF
    if sig_max == 0.0:
        sig_max = _INF

    return Material(
        id=_id, law=3, rho0=_rho, title=_title,
        params={
            "E": e, "nu": nu, "G": g, "K": k,
            "A": a, "B": b, "N": n,
            "eps_max": eps_max, "sig_max": sig_max,
        },
    )


def _register():
    """Register the law-3 physics constructor."""
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["LAW3"] = build_law03
    MAT_PHYSICS_REGISTRY["PLAS_BOST"] = build_law03
    MAT_PHYSICS_REGISTRY["BOSTEELS"] = build_law03


_register()


# ===================================================================
# Solid stress update (m3law.F, IPLA=0 path)
# ===================================================================

def solid_update(mat, sig: np.ndarray, deps: np.ndarray,
                 epsp=None, dt: float = 0.0, extra: dict = None):
    """Vectorized 3D solid stress update — J2 radial return.

    Ports ``m3law.F`` lines 92-231 (IPLA=0 branch).

    Parameters
    ----------
    sig : (n, 6)  old (Jaumann-rotated) stress [xx, yy, zz, xy, yz, zx]
    deps : (n, 6) strain increment (engineering shear)
    epsp : (n,) accumulated equivalent plastic strain
    dt : float — time step (unused; rates not needed for LAW3)

    Returns
    -------
    (sig, epsp, None) — None means constant elastic sound speed.
    """
    p = mat.params
    G = p["G"]
    A = p["A"]
    B = p["B"]
    N = p["N"]
    sig_max = p["sig_max"]

    nel = sig.shape[0]
    if epsp is None:
        epsp = np.zeros(nel)
    epsp = epsp.copy()
    sig = sig.copy()

    # ---- strip old pressure and compute deviatoric trial ----
    # m3law.F lines 92-95: P = -1/3*(sig_xx+sig_yy+sig_zz)
    # This is the NEGATIVE mean stress; +P strips it from normal components.
    p_old = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    tr3 = (deps[:, 0] + deps[:, 1] + deps[:, 2]) / 3.0

    # ---- elastic trial on deviatoric (m3law.F lines 106-113) ----
    # Convert to deviatoric: s = sig - p_old*I, then add 2G*deps_dev
    s = sig.copy()
    s[:, 0] += 2.0 * G * (deps[:, 0] - tr3) - p_old
    s[:, 1] += 2.0 * G * (deps[:, 1] - tr3) - p_old
    s[:, 2] += 2.0 * G * (deps[:, 2] - tr3) - p_old
    s[:, 3] += G * deps[:, 3]
    s[:, 4] += G * deps[:, 4]
    s[:, 5] += G * deps[:, 5]

    # ---- new pressure (hypoelastic bulk modulus) ----
    K = mat.params["K"]
    if extra is not None and "rho" in extra:
        p_new = -K * (extra["rho"] / mat.rho0 - 1.0)
    else:
        p_new = p_old + K * 3.0 * tr3

    # ---- yield stress (m3law.F lines 117-119) ----
    yld = np.minimum(sig_max, A + B * np.maximum(0.0, epsp) ** N)

    # ---- hardening modulus QH (m3law.F lines 123-139) ----
    qh = _hardening_modulus(B, N, epsp, nel)

    # ---- von Mises norm (m3law.F lines 148-151) ----
    # Working on deviatoric stress s (pressure already stripped)
    aj2 = (0.5 * (s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2)
           + s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    sj2 = np.sqrt(3.0 * aj2)

    # ---- radial return, IPLA=0 (m3law.F lines 160-172) ----
    scale = np.minimum(1.0, yld / np.maximum(sj2, _EM15))
    active = yld > 0.0
    for c in range(6):
        s[active, c] *= scale[active]
    dpla = np.zeros(nel)
    dpla[active] = ((1.0 - scale[active]) * sj2[active]
                    / np.maximum(3.0 * G + qh[active], _EM15))
    epsp += dpla

    # ---- add pressure back (LAW2 convention: total stress output) ----
    sig[:, :] = s
    sig[:, 0] += p_new
    sig[:, 1] += p_new
    sig[:, 2] += p_new

    return sig, epsp, None


# ===================================================================
# Shell stress update (m3law8.F — Iplas=2 radial projection)
# ===================================================================

def shell_update(mat, sig: np.ndarray, deps: np.ndarray,
                 epsp=None, dt: float = 0.0, extra: dict = None):
    """Vectorized plane-stress shell update — J2 radial return.

    Ports ``m3law8.F`` mapped to the same Iplas=2 radial projection the
    LAW2/LAW36/LAW44 shell ports use.

    Parameters
    ----------
    sig : (n, 3) or (n, 5)  old stress [xx, yy, xy (, yz, zx)]
    deps : (n, 3) strain increment (in-plane, engineering shear)

    Returns
    -------
    (sig, epsp)
    """
    p = mat.params
    E = p["E"]
    nu = p["nu"]
    G = p["G"]
    A = p["A"]
    B = p["B"]
    N = p["N"]
    sig_max = p["sig_max"]

    nel = sig.shape[0]
    if epsp is None:
        epsp = np.zeros(nel)
    epsp = epsp.copy()
    sig = sig.copy()

    # ---- plane-stress elastic trial ----
    c1 = E / (1.0 - nu * nu)
    c12 = nu * c1
    sig[:, 0] += c1 * deps[:, 0] + c12 * deps[:, 1]
    sig[:, 1] += c12 * deps[:, 0] + c1 * deps[:, 1]
    sig[:, 2] += G * deps[:, 2]

    # ---- yield stress ----
    yld = np.minimum(sig_max, A + B * np.maximum(0.0, epsp) ** N)

    # ---- hardening modulus QH ----
    qh = _hardening_modulus(B, N, epsp, nel)

    # ---- von Mises equivalent (plane stress) ----
    svm = np.sqrt(sig[:, 0] ** 2 + sig[:, 1] ** 2
                  - sig[:, 0] * sig[:, 1] + 3.0 * sig[:, 2] ** 2)

    # ---- radial return ----
    scale = np.minimum(1.0, yld / np.maximum(svm, _EM15))
    active = yld > 0.0
    for c in range(min(3, sig.shape[1])):
        sig[active, c] *= scale[active]
    dpla = np.zeros(nel)
    dpla[active] = ((1.0 - scale[active]) * svm[active]
                    / np.maximum(3.0 * G + qh[active], _EM15))
    epsp += dpla

    return sig, epsp


# ===================================================================
# Helpers
# ===================================================================

def _hardening_modulus(B, N, epsp, nel):
    """Compute QH = d(sigma_y)/d(eps_p) for the power-law hardening.

    m3law.F lines 123-139: three branches on N vs 1."""
    qh = np.zeros(nel)
    if N == 1.0:
        qh[:] = B
    elif N > 1.0:
        qh[:] = B * N * np.maximum(0.0, epsp) ** (N - 1.0)
    else:
        mask = epsp > 0.0
        if np.any(mask):
            qh[mask] = B * N / np.maximum(epsp[mask], _EM15) ** (1.0 - N)
    return qh


# ===================================================================
# Consistent tangent for the implicit solver
# ===================================================================

def consistent_solid_tangent(mat, sig, epsp, epsp_incr, extra=None):
    """(n, 6, 6) algorithmic tangent — same J2 radial-return tangent
    formula as LAW2 (law02_johnson_cook.consistent_solid_tangent).

    C_ep = C_el - (6G^2 / (3G + H)) * (n otimes n)

    where n = s_dev / ||s_dev||_F is the normalized deviatoric direction
    and H is the hardening slope."""
    p = mat.params
    G = p["G"]
    K = p["K"]
    B = p["B"]
    N = p["N"]

    nel = sig.shape[0]

    # Isotropic elastic tangent C_el (Voigt notation)
    lam = K - 2.0 * G / 3.0
    Ce = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            Ce[i, j] = lam
        Ce[i, i] += 2.0 * G
    for i in range(3, 6):
        Ce[i, i] = G

    Ct = np.tile(Ce, (nel, 1, 1))  # (n, 6, 6)

    # Hardening modulus
    qh = _hardening_modulus(B, N, epsp, nel)

    # Plasticity correction for yielding elements
    plastic = epsp_incr > 0.0
    if np.any(plastic):
        idx = np.where(plastic)[0]
        for ii in idx:
            s = sig[ii, :].copy()
            pm = (s[0] + s[1] + s[2]) / 3.0
            s[0] -= pm
            s[1] -= pm
            s[2] -= pm
            # Frobenius norm of deviatoric stress
            snorm = np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2
                            + 2.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2))
            if snorm < _EM15:
                continue
            ndir = s / snorm
            H = qh[ii]
            factor = 6.0 * G * G / max(3.0 * G + H, _EM15)
            Ct[ii] -= factor * np.outer(ndir, ndir)

    return Ct


def consistent_shell_tangent(mat, sig, epsp, epsp_incr, extra=None):
    """(n, 3, 3) consistent plane-stress tangent for the shell implicit
    tangents."""
    p = mat.params
    E = p["E"]
    nu = p["nu"]
    G = p["G"]
    B = p["B"]
    N = p["N"]

    nel = sig.shape[0]

    # Elastic plane-stress tangent
    c1 = E / (1.0 - nu * nu)
    c12 = nu * c1
    Ce = np.array([[c1, c12, 0.0],
                    [c12, c1, 0.0],
                    [0.0, 0.0, G]])

    Ct = np.tile(Ce, (nel, 1, 1))

    # Hardening modulus
    qh = _hardening_modulus(B, N, epsp, nel)

    # von Mises from current plane stress
    svm = np.sqrt(np.maximum(
        sig[:, 0] ** 2 + sig[:, 1] ** 2
        - sig[:, 0] * sig[:, 1] + 3.0 * sig[:, 2] ** 2, 0.0))

    plastic = (epsp_incr > 0.0) & (svm > _EM15)
    if np.any(plastic):
        idx = np.where(plastic)[0]
        for ii in idx:
            s = sig[ii, :3].copy()
            svm_ii = svm[ii]
            if svm_ii < _EM15:
                continue
            # Flow direction d(svm)/d(sigma) for plane-stress von Mises
            ndir = np.array([
                (2.0 * s[0] - s[1]) / (2.0 * svm_ii),
                (2.0 * s[1] - s[0]) / (2.0 * svm_ii),
                3.0 * s[2] / (2.0 * svm_ii),
            ])
            H = qh[ii]
            # Standard consistent tangent: C_ep = C_el - (Ce@n)(Ce@n)^T / (n^T Ce n + H')
            Cen = Ce @ ndir
            denom = ndir @ Cen + H
            if abs(denom) > _EM15:
                Ct[ii] -= np.outer(Cen, Cen) / denom

    return Ct


def shell_membrane_tangent(mat):
    """(3, 3) elastic plane-stress membrane tangent."""
    p = mat.params
    E = p["E"]
    nu = p["nu"]
    G = p["G"]
    c1 = E / (1.0 - nu * nu)
    c12 = nu * c1
    return np.array([[c1, c12, 0.0],
                      [c12, c1, 0.0],
                      [0.0, 0.0, G]])

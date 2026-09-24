"""
6-node Thick Shell Wedge Element (/THICK_SHELL wedge / solide6c).

Fortran origin: ``engine/source/elements/thickshell/solide6c/``
    s6cforc3.F  driver: gather coords/velocities, through-thickness integration
    s6cderi3.F  triangular in-plane and thickness shape function derivatives
    s6cfint3.F  internal force assembly for 6-node thick shell wedge

Theory notes:
* 6-node prismatic geometry (3 bottom nodes, 3 top nodes) treated with shell kinematics.
* Employs through-thickness integration (typically NIP=2..5 integration points along zeta)
  to accurately capture bending gradients, plasticity through thickness, and transverse shear.
* No rotational DOFs are required at the nodes; bending moments are represented directly
  by translational force couples between the top and bottom triangular faces.
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import det_inv33, scatter_add3
from .solid_penta6 import _DN_DXI, _FACES, _char_length, _geometry


def init_group(group, model, log):
    """Starter initialization for 6-node thick shell wedge element group."""
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        group.state.update(
            sig=np.zeros((0, 2, 6)),
            epsp=np.zeros((0, 2)),
            vol0=np.zeros(0),
            vol_g0=np.zeros((0, 2)),
            mass=np.zeros(0),
            eint=np.zeros(0),
            ehour=np.zeros(0),
            off=np.zeros(0),
            qvw_pend=np.zeros(0),
            dtfac=np.zeros(0),
            chk_fail=False,
            dama=np.zeros(0),
        )
        return np.zeros(0, dtype=np.int64), np.zeros(0), None

    xe = model.x0[conn]
    dndx0, vol_g0, vol0 = _geometry(xe)

    bad = vol0 <= EM20
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/THICK_SHELL(WEDGE6) {eid}: zero or negative volume", "SOLID INIT")

    rho0 = np.zeros(n)
    for sl, mat, prop in group.state.get("slices", []):
        rho0[sl] = getattr(mat, "rho0", 0.0)
    mass = rho0 * vol0

    lc0 = _char_length(xe, vol0)
    dtfac = np.full(n, 0.9)

    group.state.update(
        sig=np.zeros((n, 2, 6)),
        epsp=np.zeros((n, 2)),
        vol0=vol0.copy(),
        vol_g0=vol_g0.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),
        off=np.ones(n),
        qvw_pend=np.zeros(n),
        dtfac=dtfac,
        chk_fail=False,
        dama=np.zeros(n),
    )

    for sl, mat, prop in group.state.get("slices", []):
        if getattr(mat, "fail_models", None) or getattr(mat, "eps_p_max", 0.0) > 0.0:
            group.state["chk_fail"] = True

    node_idx = conn.reshape(-1)
    mass_c = np.repeat(mass / 6.0, 6)
    return node_idx, mass_c, None


def forces(group, x, v, vr, dt, fint, mint):
    """Engine explicit cycle force kernel for 6-node thick shell wedge elements."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)
    if dt is None or dt < 0.0:
        return np.full(n, EP30)

    xe = x[conn]
    ve = v[conn] if v is not None else np.zeros_like(xe)

    dndx, vol_g, vol_tot = _geometry(xe)
    vol_tot_safe = np.maximum(vol_tot, EM20)
    lc = _char_length(xe, vol_tot_safe)

    # Probe cycle 0
    if dt == 0.0 or v is None:
        rho = st["mass"] / vol_tot_safe
        c = np.zeros(n)
        for sl, mat, _ in st.get("slices", []):
            K_sl = getattr(mat, "K", 0.0)
            G_sl = getattr(mat, "G", 0.0)
            c[sl] = np.sqrt(np.maximum(K_sl + 4.0 * G_sl / 3.0, 0.0) / np.maximum(rho[sl], EM20))
        return np.where(c > 0.0, st["dtfac"] * lc / c, EP30)

    sig = st["sig"]
    sig_old = sig.copy()
    alive = st["off"] > 0.0

    # Rate of deformation and spin at 2 through-thickness integration points
    L = np.einsum("nib,ngic->ngbc", ve, dndx)
    D = 0.5 * (L + np.transpose(L, (0, 1, 3, 2)))
    trD = D[:, :, 0, 0] + D[:, :, 1, 1] + D[:, :, 2, 2]

    W = 0.5 * (L - np.transpose(L, (0, 1, 3, 2)))

    deps = np.empty((n, 2, 6), dtype=np.float64)
    deps[:, :, 0] = D[:, :, 0, 0] * dt
    deps[:, :, 1] = D[:, :, 1, 1] * dt
    deps[:, :, 2] = D[:, :, 2, 2] * dt
    deps[:, :, 3] = 2.0 * D[:, :, 0, 1] * dt
    deps[:, :, 4] = 2.0 * D[:, :, 1, 2] * dt
    deps[:, :, 5] = 2.0 * D[:, :, 0, 2] * dt

    if not np.all(alive):
        deps[~alive] = 0.0
        trD[~alive] = 0.0

    # Jaumann stress rotation
    wxy = W[:, :, 0, 1] * dt
    wyz = W[:, :, 1, 2] * dt
    wxz = W[:, :, 0, 2] * dt

    sxx = sig[:, :, 0].copy()
    syy = sig[:, :, 1].copy()
    szz = sig[:, :, 2].copy()
    sxy = sig[:, :, 3].copy()
    syz = sig[:, :, 4].copy()
    szx = sig[:, :, 5].copy()

    sig[:, :, 0] += 2.0 * (wxy * sxy + wxz * szx)
    sig[:, :, 1] += 2.0 * (-wxy * sxy + wyz * syz)
    sig[:, :, 2] += 2.0 * (-wxz * szx - wyz * syz)
    sig[:, :, 3] += wxy * (syy - sxx) + wxz * syz + wyz * szx
    sig[:, :, 4] += wyz * (szz - syy) - wxy * szx - wxz * sxy
    sig[:, :, 5] += wxz * (szz - sxx) + wxy * syz - wyz * sxy

    # Constitutive update
    c_sound = np.zeros(n)
    qa = np.zeros(n)
    qb = np.zeros(n)
    rho = st["mass"] / vol_tot_safe

    for sl, mat, prop in st.get("slices", []):
        qa[sl] = getattr(prop, "qa", 1.1) if prop else 1.1
        qb[sl] = getattr(prop, "qb", 0.05) if prop else 0.05

        law = getattr(mat, "law", 1)
        if law == 0:
            sig[sl] = 0.0
            continue

        K_sl = getattr(mat, "K", 0.0)
        G_sl = getattr(mat, "G", 0.0)
        c_sound[sl] = np.sqrt(np.maximum(K_sl + 4.0 * G_sl / 3.0, 0.0) / np.maximum(rho[sl], EM20))

        for g in range(2):
            materials.solid_update(mat, sig[sl, g], deps[sl, g], st["epsp"][sl, g], dt, None)

    # Bulk viscosity
    compressing = (trD < 0.0) & alive[:, None]
    qvisc = np.where(
        compressing,
        rho[:, None] * lc[:, None] * (qa[:, None]**2 * lc[:, None] * trD**2 - qb[:, None] * c_sound[:, None] * trD),
        0.0)

    sig_tot = sig.copy()
    sig_tot[:, :, 0] -= qvisc
    sig_tot[:, :, 1] -= qvisc
    sig_tot[:, :, 2] -= qvisc

    S = np.empty((n, 2, 3, 3), dtype=np.float64)
    S[:, :, 0, 0], S[:, :, 1, 1], S[:, :, 2, 2] = sig_tot[:, :, 0], sig_tot[:, :, 1], sig_tot[:, :, 2]
    S[:, :, 0, 1] = S[:, :, 1, 0] = sig_tot[:, :, 3]
    S[:, :, 1, 2] = S[:, :, 2, 1] = sig_tot[:, :, 4]
    S[:, :, 0, 2] = S[:, :, 2, 0] = sig_tot[:, :, 5]

    fe = -np.einsum("ng,ngbc,ngic->nib", vol_g, S, dndx)

    # Energy bookkeeping
    sig_mid = 0.5 * (sig_old + sig)
    w_visc = np.sum(vol_g * 0.5 * qvisc * (-trD * dt), axis=1) + st["qvw_pend"] * np.mean(-trD, axis=1)
    deint0 = np.sum(vol_g * np.einsum("nga,nga->ng", sig_mid, deps), axis=1) + w_visc
    st["qvw_pend"] = np.sum(vol_g * 0.5 * qvisc * dt, axis=1)
    st["eint"] += deint0

    # Courant time step
    trD_min = np.min(trD, axis=1)
    Q = np.where(trD_min < 0.0, qb * c_sound + qa * lc * np.abs(trD_min), 0.0)
    denom = Q + np.sqrt(Q * Q + c_sound * c_sound)
    safe_denom = np.where(denom > 0.0, denom, 1.0)
    dt_crit = np.where(denom > 0.0, st["dtfac"] * lc / safe_denom, EP30)
    dt_crit = np.where(alive, dt_crit, EP30)

    if fint is not None:
        scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3))

    return dt_crit


# ----------------------------------------------------------------------------
# Implicit element matrices: tangent, kgeo, consistent_mass (M614 Component 1B)
# ----------------------------------------------------------------------------
# Fortran origin:
#   engine/source/elements/thickshell/solide6c/s6cforc3.F (driver)
#   engine/source/elements/thickshell/solide6c/s6cderi3.F (derivatives)
#   engine/source/elements/thickshell/solide6c/s6cfint3.F (internal forces)
#   engine/source/implicit/assem_s6.F (implicit assembly for 6-node thick shell wedges)
# ----------------------------------------------------------------------------

def _edofs(conn: np.ndarray) -> np.ndarray:
    """(n, 18) global scalar translation DOF indices in node-major order."""
    n = len(conn)
    if n == 0:
        return np.zeros((0, 18), dtype=np.int64)
    ix = np.arange(6)
    edofs = np.full((n, 18), -1, dtype=np.int64)
    valid = conn >= 0
    for c in range(3):
        edofs[:, 3 * ix + c] = np.where(valid, conn * 6 + c, -1)
    return edofs


def tangent(group, x, epsp_incr=None):
    """Element tangent stiffness for the 6-node thick shell wedge group (n, 18, 18).

    Fortran origin: engine/source/elements/thickshell/solide6c/s6cforc3.F,
    s6cderi3.F, s6cfint3.F; engine/source/implicit/assem_s6.F.

    Returns (ke, edofs):
      ke: (n, 18, 18)
      edofs: (n, 18)
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 18, 18), dtype=float), np.zeros((0, 18), dtype=np.int64)

    xe = x[conn]
    dndx, vol_g, vol_tot = _geometry(xe)

    ke = np.zeros((n, 18, 18), dtype=np.float64)
    epi = np.zeros((n, 2)) if epsp_incr is None else (
        epsp_incr if np.ndim(epsp_incr) == 2 else np.repeat(epsp_incr[:, None], 2, axis=1)
    )

    for g in range(2):
        B = np.zeros((n, 6, 18), dtype=np.float64)
        gx = dndx[:, g, :, 0]
        gy = dndx[:, g, :, 1]
        gz = dndx[:, g, :, 2]
        ix = np.arange(6)

        B[:, 0, 3 * ix + 0] = gx
        B[:, 1, 3 * ix + 1] = gy
        B[:, 2, 3 * ix + 2] = gz
        B[:, 3, 3 * ix + 0] = gy
        B[:, 3, 3 * ix + 1] = gx
        B[:, 4, 3 * ix + 1] = gz
        B[:, 4, 3 * ix + 2] = gy
        B[:, 5, 3 * ix + 0] = gz
        B[:, 5, 3 * ix + 2] = gx

        for sl, mat, prop in st.get("slices", []):
            if getattr(mat, "law", 1) == 0:
                continue
            D = materials.solid_tangent(mat, st["sig"][sl, g], st["epsp"][sl, g], epi[sl, g], None)
            Bs = B[sl]
            DB = np.einsum("mij,mjk->mik", D, Bs)
            ke[sl] += vol_g[sl, g][:, None, None] * np.einsum("mji,mjk->mik", Bs, DB)

    dead = (st["off"] <= 0.0)
    if np.any(dead):
        ke[dead] = 0.0

    return ke, _edofs(conn)


def kgeo(group, x):
    """Geometric (initial-stress) element stiffness for the 6-node thick shell wedge.

    Fortran origin: engine/source/elements/thickshell/solide6c/s6cfint3.F,
    engine/source/implicit/assem_s6.F.

    Returns (ke, edofs): ke (n, 18, 18), edofs (n, 18).
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 18, 18), dtype=float), np.zeros((0, 18), dtype=np.int64)

    xe = x[conn]
    dndx, vol_g, vol_tot = _geometry(xe)

    ke = np.zeros((n, 18, 18), dtype=np.float64)
    sig = st["sig"]

    for g in range(2):
        s_g = sig[:, g]
        S = np.empty((n, 3, 3), dtype=np.float64)
        S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = s_g[:, 0], s_g[:, 1], s_g[:, 2]
        S[:, 0, 1] = S[:, 1, 0] = s_g[:, 3]
        S[:, 1, 2] = S[:, 2, 1] = s_g[:, 4]
        S[:, 0, 2] = S[:, 2, 0] = s_g[:, 5]

        g_mat = vol_g[:, g, None, None] * np.einsum("nac,ncd,nbd->nab", dndx[:, g], S, dndx[:, g])

        ix = np.arange(6)
        for b in range(3):
            rows = (3 * ix + b)[:, None]
            cols = (3 * ix + b)[None, :]
            ke[:, rows, cols] += g_mat

    dead = (st["off"] <= 0.0)
    if np.any(dead):
        ke[dead] = 0.0

    return ke, _edofs(conn)


_M_WEDGE6 = np.array([
    [4.0, 2.0, 2.0, 2.0, 1.0, 1.0],
    [2.0, 4.0, 2.0, 1.0, 2.0, 1.0],
    [2.0, 2.0, 4.0, 1.0, 1.0, 2.0],
    [2.0, 1.0, 1.0, 4.0, 2.0, 2.0],
    [1.0, 2.0, 1.0, 2.0, 4.0, 2.0],
    [1.0, 1.0, 2.0, 2.0, 2.0, 4.0],
], dtype=np.float64) / 72.0


def consistent_mass(group, x=None):
    """Consistent element mass matrix for 6-node thick shell wedge (18x18).

    Fortran origin: starter/source/elements/solid/solide/smass3.F,
    engine/source/implicit/assem_s6.F.

    Returns (me, edofs): me (n, 18, 18), edofs (n, 18).
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 18, 18), dtype=float), np.zeros((0, 18), dtype=np.int64)

    m = st["mass"]
    me = np.zeros((n, 18, 18), dtype=np.float64)
    for a in range(6):
        for b in range(6):
            val = m * _M_WEDGE6[a, b]
            for c in range(3):
                me[:, 3 * a + c, 3 * b + c] = val

    dead = (st["off"] <= 0.0)
    if np.any(dead):
        me[dead] = 0.0

    return me, _edofs(conn)


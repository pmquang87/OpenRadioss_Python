"""
5-node Pyramid Solid Element (/PYRA5 or collapsed /BRICK with IDEGE=6).

Fortran origin:
    starter/source/elements/solid/solide/degenes8.F  (IDEGE=6 degenerate pyramid detection)
    engine/source/elements/solid/solide/sdlen_dege.F (DELTAX FAC=1/9 time step scaling)
    engine/source/elements/solid/solide/sforc3.F     (pyramid kinematics and internal forces)

Theory notes:
* 5-node pyramid with quadrilateral base (nodes 1, 2, 3, 4) and apex (node 5).
* In OpenRadioss, pyramids can be represented as 5-node elements or as degenerate 8-node
  bricks where the top face collapses to a single apex node (nodes 5 = 6 = 7 = 8).
* Volume of a pyramid: V = 1/3 * A_base * h.
* Mass distribution (degenes8.F):
    Base nodes 1..4 receive 1/8 of the element mass each (total 4/8 = 1/2).
    Apex node 5 receives 4/8 = 1/2 of the element mass.
* Characteristic length and Courant time step (sdlen_dege.F):
    The collapsed apex concentrates gradients, scaling the critical time step by:
        FAC = (1/3)^2 = 1/9 ~ 0.111111
    to maintain unconditional numerical stability in explicit integration.
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import cross3, det_inv33, norm3, scatter_add3


def _pyramid_geometry(xe: np.ndarray):
    """Geometry for 5-node pyramid (or 8-node degenerate pyramid).

    xe: (n, 5, 3) or (n, 8, 3) nodal coordinates.
    Returns:
        dndx: (n, 5, 3) shape function gradients at centroid
        vol: (n,) element volume
        lc: (n,) characteristic length
    """
    n = len(xe)
    if n == 0:
        return np.zeros((0, 5, 3)), np.zeros(0), np.zeros(0)

    # Extract 5 key nodes
    p1 = xe[:, 0]
    p2 = xe[:, 1]
    p3 = xe[:, 2]
    p4 = xe[:, 3]
    p5 = xe[:, 4]

    # Base center and normal
    base_center = 0.25 * (p1 + p2 + p3 + p4)
    d13 = p3 - p1
    d24 = p4 - p2
    cross_base = np.cross(d13, d24)
    a_base2 = np.linalg.norm(cross_base, axis=1)
    a_base = 0.5 * a_base2

    safe_abase2 = np.where(a_base2 > EM20, a_base2, 1.0)
    n_base = cross_base / safe_abase2[:, None]

    # Height from apex p5 to base
    h = np.abs(np.sum((p5 - base_center) * n_base, axis=1))

    # Exact pyramid volume V = 1/3 * A_base * h
    vol = (1.0 / 3.0) * a_base * h
    vol_safe = np.maximum(vol, EM20)

    # Characteristic length: lc = V / max(face_area) scaled by FAC = 1/9 (sdlen_dege.F)
    # 4 triangular side faces: (1,2,5), (2,3,5), (3,4,5), (4,1,5)
    a_side1 = 0.5 * np.linalg.norm(np.cross(p2 - p1, p5 - p1), axis=1)
    a_side2 = 0.5 * np.linalg.norm(np.cross(p3 - p2, p5 - p2), axis=1)
    a_side3 = 0.5 * np.linalg.norm(np.cross(p4 - p3, p5 - p3), axis=1)
    a_side4 = 0.5 * np.linalg.norm(np.cross(p1 - p4, p5 - p4), axis=1)
    max_area = np.maximum(np.maximum.reduce([a_base, a_side1, a_side2, a_side3, a_side4]), EM20)

    # Characteristic length with sdlen_dege scaling
    lc = (vol_safe / max_area) * (1.0 / 9.0)

    # Centroid shape function gradients
    # Centroid is at z = 1/4 h above base center
    # Approximate linear gradients:
    # dN_apex / dz = 1 / h; dN_base / dz = - 1 / (4 h)
    dndx = np.zeros((n, 5, 3), dtype=np.float64)

    # Base in-plane gradients
    dndx[:, 0] = 0.5 * np.cross(p4 - p2, n_base) / safe_abase2[:, None]
    dndx[:, 1] = 0.5 * np.cross(p1 - p3, n_base) / safe_abase2[:, None]
    dndx[:, 2] = 0.5 * np.cross(p2 - p4, n_base) / safe_abase2[:, None]
    dndx[:, 3] = 0.5 * np.cross(p3 - p1, n_base) / safe_abase2[:, None]

    # Normal gradients
    safe_h = np.where(h > EM20, h, 1.0)
    grad_z = n_base / safe_h[:, None]
    dndx[:, 4] += grad_z
    for i in range(4):
        dndx[:, i] -= 0.25 * grad_z

    return dndx, vol_safe, lc


def init_group(group, model, log):
    """Starter initialization for 5-node pyramid element group."""
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        group.state.update(
            sig=np.zeros((0, 6)),
            epsp=np.zeros(0),
            vol0=np.zeros(0),
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
    dndx0, vol0, lc0 = _pyramid_geometry(xe)

    bad = vol0 <= EM20
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/PYRA5 {eid}: zero or negative volume", "SOLID INIT")

    rho0 = np.zeros(n)
    for sl, mat, prop in group.state.get("slices", []):
        rho0[sl] = getattr(mat, "rho0", 0.0)
    mass = rho0 * vol0

    group.state.update(
        sig=np.zeros((n, 6)),
        epsp=np.zeros(n),
        vol0=vol0.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),
        off=np.ones(n),
        qvw_pend=np.zeros(n),
        dtfac=np.ones(n),
        chk_fail=False,
        dama=np.zeros(n),
    )

    for sl, mat, prop in group.state.get("slices", []):
        if getattr(mat, "fail_models", None) or getattr(mat, "eps_p_max", 0.0) > 0.0:
            group.state["chk_fail"] = True

    # Nodal mass lumping (degenes8.F):
    # If 5 nodes: nodes 0..3 get mass/8, node 4 gets 4*mass/8 = mass/2
    # If 8 nodes: each node gets mass/8 (apex is repeated 4 times, automatically summing to mass/2)
    n_nodes = conn.shape[1]
    if n_nodes == 5:
        m_nodes = np.zeros((n, 5), dtype=np.float64)
        m_nodes[:, 0:4] = 0.125 * mass[:, None]
        m_nodes[:, 4] = 0.5 * mass
        node_idx = conn.reshape(-1)
        mass_c = m_nodes.reshape(-1)
    else:
        node_idx = conn.reshape(-1)
        mass_c = np.repeat(mass / 8.0, 8)

    return node_idx, mass_c, None


def forces(group, x, v, vr, dt, fint, mint):
    """Engine explicit cycle force kernel for 5-node pyramid solid elements."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)
    if dt is None or dt < 0.0:
        return np.full(n, EP30)

    xe = x[conn]
    ve = v[conn] if v is not None else np.zeros_like(xe)

    # Use first 5 nodes
    xe5 = xe[:, 0:5]
    ve5 = ve[:, 0:5]
    dndx, vol, lc = _pyramid_geometry(xe5)
    vol_safe = np.maximum(vol, EM20)

    # Probe cycle 0
    if dt == 0.0 or v is None:
        rho = st["mass"] / vol_safe
        c = np.zeros(n)
        for sl, mat, _ in st.get("slices", []):
            K_sl = getattr(mat, "K", 0.0)
            G_sl = getattr(mat, "G", 0.0)
            c[sl] = np.sqrt(np.maximum(K_sl + 4.0 * G_sl / 3.0, 0.0) / np.maximum(rho[sl], EM20))
        return np.where(c > 0.0, st["dtfac"] * lc / c, EP30)

    sig = st["sig"]
    sig_old = sig.copy()
    alive = st["off"] > 0.0

    # Rate of deformation
    L = np.einsum("nib,nic->nbc", ve5, dndx)
    D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
    trD = D[:, 0, 0] + D[:, 1, 1] + D[:, 2, 2]

    W = 0.5 * (L - np.transpose(L, (0, 2, 1)))

    deps = np.empty((n, 6), dtype=np.float64)
    deps[:, 0] = D[:, 0, 0] * dt
    deps[:, 1] = D[:, 1, 1] * dt
    deps[:, 2] = D[:, 2, 2] * dt
    deps[:, 3] = 2.0 * D[:, 0, 1] * dt
    deps[:, 4] = 2.0 * D[:, 1, 2] * dt
    deps[:, 5] = 2.0 * D[:, 0, 2] * dt

    if not np.all(alive):
        deps[~alive] = 0.0
        trD[~alive] = 0.0

    # Jaumann stress rotation
    wxy = W[:, 0, 1] * dt
    wyz = W[:, 1, 2] * dt
    wxz = W[:, 0, 2] * dt

    sxx, syy, szz = sig[:, 0].copy(), sig[:, 1].copy(), sig[:, 2].copy()
    sxy, syz, szx = sig[:, 3].copy(), sig[:, 4].copy(), sig[:, 5].copy()

    sig[:, 0] += 2.0 * (wxy * sxy + wxz * szx)
    sig[:, 1] += 2.0 * (-wxy * sxy + wyz * syz)
    sig[:, 2] += 2.0 * (-wxz * szx - wyz * syz)
    sig[:, 3] += wxy * (syy - sxx) + wxz * syz + wyz * szx
    sig[:, 4] += wyz * (szz - syy) - wxy * szx - wxz * sxy
    sig[:, 5] += wxz * (szz - sxx) + wxy * syz - wyz * sxy

    # Constitutive update
    c_sound = np.zeros(n)
    qa = np.zeros(n)
    qb = np.zeros(n)
    rho = st["mass"] / vol_safe

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

        materials.solid_update(mat, sig[sl], deps[sl], st["epsp"][sl], dt, None)

    # Bulk viscosity
    compressing = (trD < 0.0) & alive
    qvisc = np.where(
        compressing,
        rho * lc * (qa**2 * lc * trD**2 - qb * c_sound * trD),
        0.0)

    sig_tot = sig.copy()
    sig_tot[:, 0] -= qvisc
    sig_tot[:, 1] -= qvisc
    sig_tot[:, 2] -= qvisc

    S = np.empty((n, 3, 3), dtype=np.float64)
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = sig_tot[:, 0], sig_tot[:, 1], sig_tot[:, 2]
    S[:, 0, 1] = S[:, 1, 0] = sig_tot[:, 3]
    S[:, 1, 2] = S[:, 2, 1] = sig_tot[:, 4]
    S[:, 0, 2] = S[:, 2, 0] = sig_tot[:, 5]

    # Internal force assembly: f_i = - vol * sigma . gradN_i
    fe5 = -vol[:, None, None] * np.einsum("nbc,nic->nib", S, dndx)

    # Energy bookkeeping
    sig_mid = 0.5 * (sig_old + sig)
    w_visc = vol * 0.5 * qvisc * (-trD * dt) + st["qvw_pend"] * (-trD)
    deint0 = vol * np.einsum("na,na->n", sig_mid, deps) + w_visc
    st["qvw_pend"] = vol * 0.5 * qvisc * dt
    st["eint"] += deint0

    # Courant time step with FAC=1/9
    Q = np.where(compressing, qb * c_sound + qa * lc * np.abs(trD), 0.0)
    denom = Q + np.sqrt(Q * Q + c_sound * c_sound)
    safe_denom = np.where(denom > 0.0, denom, 1.0)
    dt_crit = np.where(denom > 0.0, st["dtfac"] * lc / safe_denom, EP30)
    dt_crit = np.where(alive, dt_crit, EP30)

    # Scatter forces to fint
    if fint is not None:
        n_nodes = conn.shape[1]
        if n_nodes == 5:
            scatter_add3(fint, conn.reshape(-1), fe5.reshape(-1, 3))
        else:
            # 8-node collapsed brick representation: distribute apex force 4 ways
            fe8 = np.zeros((n, 8, 3), dtype=np.float64)
            fe8[:, 0:4] = fe5[:, 0:4]
            for j in range(4, 8):
                fe8[:, j] = 0.25 * fe5[:, 4]
            scatter_add3(fint, conn.reshape(-1), fe8.reshape(-1, 3))

    return dt_crit

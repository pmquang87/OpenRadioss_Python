"""
4-node Smoothed Finite Element (NS-FEM / SFEM) tetrahedral solid element (Itetra4=3).

Fortran origin: ``engine/source/elements/solid/solide4_sfem/``
    s4lagsfem.F  driver: smoothed strain rates, constitutive update, and internal forces
    s4volnod3.F  nodal smoothing domain volume calculation (V_nod = sum 1/4 V_e)
    s4voln_m.F   smoothed shape function gradients over nodal domains
    sdlen3.F     characteristic length -> critical time step

Theory notes (Liu & Nguyen-Xuan 2010; OpenRadioss SFEM):
* Standard 4-node linear tetrahedral elements (CST) suffer from severe volumetric
  locking in plastic / incompressible regimes and are overly stiff in bending.
* Node-based Smoothed FEM (NS-FEM) constructs smoothing domains Omega_k around each
  mesh node k by dividing each connected tetrahedron into 4 sub-volumes of volume V_e / 4.
* The total smoothed domain volume for node k is:
      V_k^smooth = sum_{e in S_k} 1/4 * V_e
* The smoothed gradient operator at node k is:
      grad_tilde N_k = 1 / V_k^smooth * sum_{e in S_k} 1/4 * V_e * grad N_k^e
* This completely eliminates volumetric locking, softens the overly stiff CST tetrahedra,
  and delivers high accuracy in crash and large-strain metal forming simulations.
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import cross3, det_inv33, norm3, scatter_add3
from .solid_tetra4 import _char_length, _geometry


def init_group(group, model, log):
    """Starter initialization for 4-node SFEM tetrahedral group (Itetra4=3)."""
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

    xe = model.x0[conn]  # (n, 4, 3)
    dndx0, vol0 = _geometry(xe)

    bad = vol0 <= EM20
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/TETRA4(SFEM) {eid}: zero or negative volume", "SOLID INIT")

    rho0 = np.zeros(n)
    for sl, mat, prop in group.state.get("slices", []):
        rho0[sl] = getattr(mat, "rho0", 0.0)
    mass = rho0 * vol0

    lc0 = _char_length(xe, vol0)
    dtfac = np.full(n, 1.0)

    group.state.update(
        sig=np.zeros((n, 6)),
        epsp=np.zeros(n),
        vol0=vol0.copy(),
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
    mass_c = np.repeat(mass / 4.0, 4)
    return node_idx, mass_c, None


def _smooth_gradients(dndx: np.ndarray, vol: np.ndarray, conn: np.ndarray, num_nodes: int):
    """Compute smoothed shape function gradients across nodal smoothing domains (s4volnod3.F).

    dndx: (n, 4, 3) element shape gradients
    vol: (n,) element volumes
    conn: (n, 4) element connectivity
    Returns:
        dndx_smoothed: (n, 4, 3) smoothed shape function gradients per element
    """
    n = len(dndx)
    if n == 0:
        return np.zeros((0, 4, 3))

    # Nodal smoothing volume: V_nod[k] = sum_{e in S_k} 0.25 * V_e
    sub_vol = 0.25 * vol  # (n,)
    v_nod = np.zeros(num_nodes, dtype=np.float64)
    # Scatter sub-volumes to the 4 nodes of each element
    for i in range(4):
        np.add.at(v_nod, conn[:, i], sub_vol)

    safe_vnod = np.where(v_nod > EM20, v_nod, 1.0)

    # Nodal smoothed gradients: grad_nod[k, dim] = sum_{e in S_k} 0.25 * V_e * gradN_i^e
    grad_nod = np.zeros((num_nodes, 3), dtype=np.float64)
    for i in range(4):
        weighted_grad = sub_vol[:, None] * dndx[:, i, :]  # (n, 3)
        for dim in range(3):
            np.add.at(grad_nod[:, dim], conn[:, i], weighted_grad[:, dim])

    # Normalized nodal smoothed gradients
    grad_nod_norm = grad_nod / safe_vnod[:, None]

    # Gather smoothed gradients back to element nodes
    # dndx_smooth[e, i, :] = grad_nod_norm[conn[e, i], :]
    dndx_smooth = np.empty((n, 4, 3), dtype=np.float64)
    for i in range(4):
        dndx_smooth[:, i, :] = grad_nod_norm[conn[:, i], :]

    return dndx_smooth


def forces(group, x, v, vr, dt, fint, mint):
    """Engine explicit cycle force kernel for 4-node SFEM smoothed tetrahedrals."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)
    if dt is None or dt < 0.0:
        return np.full(n, EP30)

    xe = x[conn]
    ve = v[conn] if v is not None else np.zeros_like(xe)

    dndx, vol = _geometry(xe)
    vol_safe = np.maximum(vol, EM20)
    lc = _char_length(xe, vol_safe)

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
    num_nodes = len(x)

    # Smoothed shape function gradients over nodal smoothing domains
    dndx_smooth = _smooth_gradients(dndx, vol, conn, num_nodes)

    # Velocity gradient computed with smoothed gradients: L = sum_i v_i (x) grad_smooth_N_i
    L = np.einsum("nib,nic->nbc", ve, dndx_smooth)
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

    # Internal force assembly with smoothed gradients
    S = np.empty((n, 3, 3), dtype=np.float64)
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = sig_tot[:, 0], sig_tot[:, 1], sig_tot[:, 2]
    S[:, 0, 1] = S[:, 1, 0] = sig_tot[:, 3]
    S[:, 1, 2] = S[:, 2, 1] = sig_tot[:, 4]
    S[:, 0, 2] = S[:, 2, 0] = sig_tot[:, 5]

    # fe[n, i, b] = - vol[n] * sum_c S[n, b, c] * dndx_smooth[n, i, c]
    fe = -vol[:, None, None] * np.einsum("nbc,nic->nib", S, dndx_smooth)

    # Energy bookkeeping
    sig_mid = 0.5 * (sig_old + sig)
    w_visc = vol * 0.5 * qvisc * (-trD * dt) + st["qvw_pend"] * (-trD)
    deint0 = vol * np.einsum("na,na->n", sig_mid, deps) + w_visc
    st["qvw_pend"] = vol * 0.5 * qvisc * dt
    st["eint"] += deint0

    # Courant time step
    Q = np.where(compressing, qb * c_sound + qa * lc * np.abs(trD), 0.0)
    denom = Q + np.sqrt(Q * Q + c_sound * c_sound)
    safe_denom = np.where(denom > 0.0, denom, 1.0)
    dt_crit = np.where(denom > 0.0, st["dtfac"] * lc / safe_denom, EP30)
    dt_crit = np.where(alive, dt_crit, EP30)

    if fint is not None:
        scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3))

    return dt_crit

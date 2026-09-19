"""
6-node Rotation-Free Discrete Kirchhoff Triangle Shell Element (coquedk6, /SH3N DKT6).

Fortran origin: ``engine/source/elements/sh3n/coquedk6/``
    cdk6forc3.F  driver: macro-patch kinematics, rotation-free bending, translational forces
    cdk6deri3.F  patch curvature and Discrete Kirchhoff constraint operators
    cdk6fint3.F  internal translational force assembly on 3 primary + 3 neighbor nodes

Theory notes (Phung-Van et al. 2015; Bletzinger et al. 2000; OpenRadioss DKT6):
* 6-node macro-patch consisting of 3 primary vertices (nodes 1, 2, 3) and 3 neighboring
  patch vertices (nodes 4, 5, 6) adjacent to edges (2-3, 3-1, 1-2).
* Possesses ONLY translational DOFs (3 DOFs per node, 18 DOFs per element).
  Completely avoids rotational DOFs, drilling degrees of freedom, and rotational inertia issues.
* Kinematics:
    - Membrane strain: standard constant-strain triangle on the central vertices 1, 2, 3.
    - Bending curvatures: evaluated from the relative deflections of the 3 neighboring
      vertices across the 3 shared edges via Discrete Kirchhoff constraints.
* Internal forces:
    - Membrane stresses yield in-plane translational forces on nodes 1, 2, 3.
    - Bending moments produce self-equilibrating out-of-plane force couples distributed
      between the central vertices and the neighbor nodes.
    - Nodal moments mint are identically zero!
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import cross3, det_inv33, norm3, scatter_add3
from .shell_tri3 import _char_length, _local_geometry


def _patch_geometry(xe: np.ndarray):
    """Compute central triangle geometry, local corotational frame, and bending curvatures.

    xe: (n, 6, 3) or (n, 3, 3) nodal coordinates.
    Returns:
        r: (n, 3, 3) local triad [e1, e2, e3] with e3 as unit normal
        area: (n,) area of central triangle
        dndx_tri: (n, 3, 2) in-plane shape gradients on central triangle
        lc: (n,) characteristic length
    """
    n = len(xe)
    if n == 0:
        return np.zeros((0, 3, 3)), np.zeros(0), np.zeros((0, 3, 2)), np.zeros(0)

    E, xl, area, B1, B2 = _local_geometry(xe[:, 0:3])
    # E has columns [e1, e2, e3], transpose to get rows [e1, e2, e3]
    r = np.transpose(E, (0, 2, 1))

    dndx_tri = np.zeros((n, 3, 2), dtype=np.float64)
    dndx_tri[:, :, 0] = B1
    dndx_tri[:, :, 1] = B2
    lc = _char_length(xl, area)

    return r, area, dndx_tri, lc


def init_group(group, model, log):
    """Starter initialization for 6-node rotation-free DKT shell group."""
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        group.state.update(
            sig=np.zeros((0, 3)),       # membrane stresses [Nxx, Nyy, Nxy]
            sig_b=np.zeros((0, 3)),     # bending stresses [Mxx, Myy, Mxy]
            epsp=np.zeros(0),
            thick=np.zeros(0),
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
    r, area, _, lc0 = _patch_geometry(xe)

    bad = area <= EM20
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/SH3N(DKT6) {eid}: zero or negative area", "SHELL INIT")

    thick = np.zeros(n)
    rho0 = np.zeros(n)
    for sl, mat, prop in group.state.get("slices", []):
        rho0[sl] = getattr(mat, "rho0", 0.0)
        thick[sl] = getattr(prop, "thick", 1.0e-3) if prop else 1.0e-3

    mass = rho0 * area * thick
    dtfac = np.full(n, 0.9)

    group.state.update(
        sig=np.zeros((n, 3)),
        sig_b=np.zeros((n, 3)),
        epsp=np.zeros(n),
        thick=thick,
        vol0=area * thick,
        area0=area.copy(),
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

    # Nodal mass lumping: mass / 3 to each of the 3 primary triangle vertices
    # If 6 nodes provided, primary 3 carry the mass, neighbor nodes get mass from their own primary elements
    n_nodes = conn.shape[1]
    node_idx = conn[:, 0:3].reshape(-1)
    mass_c = np.repeat(mass / 3.0, 3)

    return node_idx, mass_c, None


def forces(group, x, v, vr, dt, fint, mint):
    """Engine explicit cycle force kernel for 6-node rotation-free DKT shells."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)
    if dt is None or dt < 0.0:
        return np.full(n, EP30)

    xe = x[conn]
    ve = v[conn] if v is not None else np.zeros_like(xe)

    r, area, dndx_tri, lc = _patch_geometry(xe)
    thick = st["thick"]
    vol = area * thick
    vol_safe = np.maximum(vol, EM20)


    # Probe cycle 0
    if dt == 0.0 or v is None:
        rho = st["mass"] / vol_safe
        c = np.zeros(n)
        for sl, mat, _ in st.get("slices", []):
            E_sl = getattr(mat, "E", 0.0)
            c[sl] = np.sqrt(np.maximum(E_sl, 0.0) / np.maximum(rho[sl], EM20))
        return np.where(c > 0.0, st["dtfac"] * lc / c, EP30)

    sig = st["sig"]
    sig_b = st["sig_b"]
    sig_old = sig.copy()
    alive = st["off"] > 0.0

    # Project velocity into local triad for nodes 1, 2, 3
    # v_loc[n, i, a] = sum_b ve[n, i, b] * r[n, a, b]
    v_loc = np.einsum("nib,nab->nia", ve[:, 0:3], r)

    # In-plane membrane rate of deformation D_m [xx, yy, 2xy]
    # L_ab = sum_i v_loc[i, a] * dndx_tri[i, b]
    L = np.einsum("nia,nib->nab", v_loc[:, :, 0:2], dndx_tri)
    D_m = np.empty((n, 3), dtype=np.float64)
    D_m[:, 0] = L[:, 0, 0]
    D_m[:, 1] = L[:, 1, 1]
    D_m[:, 2] = L[:, 0, 1] + L[:, 1, 0]
    trD = D_m[:, 0] + D_m[:, 1]

    # Bending curvature rate: computed from relative out-of-plane velocity
    # If 6 nodes available, use relative normal velocity differences between neighbors and primary nodes
    kappa_dot = np.zeros((n, 3), dtype=np.float64)
    if conn.shape[1] >= 6:
        # v_norm: out-of-plane velocity v . n for all 6 nodes
        v_norm = np.einsum("nib,nb->ni", ve[:, 0:6], r[:, 2])
        # Curvature rates across the 3 edges
        # Edge 1 (nodes 2-3 vs neighbor 4): (v_norm_4 - 0.5*(v_norm_2 + v_norm_3)) / h^2
        h_char = np.maximum(lc, EM20)
        k1 = (v_norm[:, 3] - 0.5 * (v_norm[:, 1] + v_norm[:, 2])) / (h_char**2)
        k2 = (v_norm[:, 4] - 0.5 * (v_norm[:, 2] + v_norm[:, 0])) / (h_char**2)
        k3 = (v_norm[:, 5] - 0.5 * (v_norm[:, 0] + v_norm[:, 1])) / (h_char**2)
        kappa_dot[:, 0] = k1
        kappa_dot[:, 1] = k2
        kappa_dot[:, 2] = 0.5 * (k1 + k2 - k3)

    deps_m = D_m * dt
    dkappa = kappa_dot * dt

    # Update membrane stresses and bending moments using shell section stiffness
    c_sound = np.zeros(n)
    qa = np.zeros(n)
    qb = np.zeros(n)
    rho = st["mass"] / vol_safe

    for sl, mat, prop in st.get("slices", []):
        qa[sl] = getattr(prop, "qa", 1.1) if prop else 1.1
        qb[sl] = getattr(prop, "qb", 0.05) if prop else 0.05

        E_sl = getattr(mat, "E", 1.0e10)
        nu_sl = getattr(mat, "nu", 0.3)
        c_sound[sl] = np.sqrt(np.maximum(E_sl / (1.0 - nu_sl**2), 0.0) / np.maximum(rho[sl], EM20))

        # Plane stress elastic stiffness
        fac = E_sl / (1.0 - nu_sl**2)
        C11 = fac
        C12 = fac * nu_sl
        C33 = fac * 0.5 * (1.0 - nu_sl)

        # Membrane increment
        sig[sl, 0] += (C11 * deps_m[sl, 0] + C12 * deps_m[sl, 1])
        sig[sl, 1] += (C12 * deps_m[sl, 0] + C11 * deps_m[sl, 1])
        sig[sl, 2] += C33 * deps_m[sl, 2]

        # Bending moment increment: M = D_b * dkappa, D_b = E * t^3 / (12 * (1 - nu^2))
        Db = fac * (thick[sl]**3) / 12.0
        sig_b[sl, 0] += Db * (dkappa[sl, 0] + nu_sl * dkappa[sl, 1])
        sig_b[sl, 1] += Db * (nu_sl * dkappa[sl, 0] + dkappa[sl, 1])
        sig_b[sl, 2] += Db * 0.5 * (1.0 - nu_sl) * dkappa[sl, 2]

    # Membrane internal forces on primary 3 nodes (in local frame, then rotated to global)
    # f_loc[i, a] = - area * thick * sum_b sigma[a, b] * dndx_tri[i, b]
    f_loc = np.zeros((n, 3, 3), dtype=np.float64)
    # S_loc 2x2: [[sxx, sxy], [sxy, syy]]
    for i in range(3):
        f_loc[:, i, 0] = -area * thick * (sig[:, 0] * dndx_tri[:, i, 0] + sig[:, 2] * dndx_tri[:, i, 1])
        f_loc[:, i, 1] = -area * thick * (sig[:, 2] * dndx_tri[:, i, 0] + sig[:, 1] * dndx_tri[:, i, 1])

    # Rotate local in-plane forces back to global coordinates: f_glob = sum_a f_loc[a] * r[a]
    fe = np.zeros((n, conn.shape[1], 3), dtype=np.float64)
    for i in range(3):
        fe[:, i] = np.einsum("na,nab->nb", f_loc[:, i], r)

    # Bending couple forces distributed to primary nodes and neighbor nodes
    if conn.shape[1] >= 6:
        # Bending moments produce self-equilibrating transverse forces along the normal r[:, 2]
        # Neighbor nodes receive f_bend * n, central nodes receive -f_bend * n
        fb_scale = (sig_b[:, 0] + sig_b[:, 1]) / np.maximum(lc, EM20)  # (n,)
        for j in range(3, 6):
            fe[:, j] += (fb_scale[:, None] * r[:, 2]) / 3.0
            fe[:, j - 3] -= (fb_scale[:, None] * r[:, 2]) / 3.0

    # Energy bookkeeping
    sig_mid = 0.5 * (sig_old + sig)
    deint0 = vol * np.sum(sig_mid * deps_m, axis=1)
    st["eint"] += deint0

    # Courant time step
    compressing = (trD < 0.0) & alive
    Q = np.where(compressing, qb * c_sound + qa * lc * np.abs(trD), 0.0)
    denom = Q + np.sqrt(Q * Q + c_sound * c_sound)
    safe_denom = np.where(denom > 0.0, denom, 1.0)
    dt_crit = np.where(denom > 0.0, st["dtfac"] * lc / safe_denom, EP30)
    dt_crit = np.where(alive, dt_crit, EP30)

    # Scatter forces to fint (mint is untouched because DKT6 is rotation-free!)
    if fint is not None:
        scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3))

    return dt_crit


# ----------------------------------------------------------------------------
# Implicit element matrices: tangent, kgeo, consistent_mass (M614 Component 1B)
# ----------------------------------------------------------------------------
# Fortran origin:
#   engine/source/elements/sh3n/coquedk6/cdk6forc3.F (driver)
#   engine/source/elements/sh3n/coquedk6/cdk6deri3.F (patch curvature & kinematics)
#   engine/source/elements/sh3n/coquedk6/cdk6fint3.F (internal force assembly)
# ----------------------------------------------------------------------------

def _edofs(conn: np.ndarray) -> np.ndarray:
    """(n, 18) global scalar translation DOF indices in node-major order."""
    n = len(conn)
    if n == 0:
        return np.zeros((0, 18), dtype=np.int64)
    ix = np.arange(min(conn.shape[1], 6))
    edofs = np.full((n, 18), -1, dtype=np.int64)
    valid = conn[:, :len(ix)] >= 0
    for c in range(3):
        edofs[:, 3 * ix + c] = np.where(valid, conn[:, :len(ix)] * 6 + c, -1)
    return edofs


def tangent(group, x, epsp_incr=None):
    """Element tangent stiffness for 6-node rotation-free DKT shell group (n, 18, 18).

    Fortran origin: engine/source/elements/sh3n/coquedk6/cdk6forc3.F,
    cdk6deri3.F, cdk6fint3.F.

    Returns (ke, edofs):
      ke: (n, 18, 18) translational element stiffness matrix
      edofs: (n, 18) global translational DOF indices
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 18, 18), dtype=float), np.zeros((0, 18), dtype=np.int64)

    xe = x[conn]
    r, area, dndx_tri, lc = _patch_geometry(xe)
    thick = st["thick"]
    vol = np.maximum(area * thick, EM20)

    ke = np.zeros((n, 18, 18), dtype=np.float64)

    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            continue
        E_sl = getattr(mat, "E", 1.0e10) or 1.0e10
        nu_sl = getattr(mat, "nu", 0.3) or 0.3

        # 1. Membrane stiffness on primary triangle nodes 0, 1, 2 (9x9 local)
        fac = E_sl / max(1.0 - nu_sl**2, 1e-12)
        Cm = np.array([
            [fac, fac * nu_sl, 0.0],
            [fac * nu_sl, fac, 0.0],
            [0.0, 0.0, fac * 0.5 * (1.0 - nu_sl)],
        ], dtype=np.float64)

        Bm = np.zeros((n, 3, 9), dtype=np.float64)
        for i in range(3):
            b1 = dndx_tri[:, i, 0]
            b2 = dndx_tri[:, i, 1]
            Bm[:, 0, 3 * i + 0] = b1
            Bm[:, 1, 3 * i + 1] = b2
            Bm[:, 2, 3 * i + 0] = b2
            Bm[:, 2, 3 * i + 1] = b1

        Km_loc = vol[sl, None, None] * np.einsum("nai,ab,nbj->nij", Bm[sl], Cm, Bm[sl])

        T_m = np.zeros((len(Km_loc), 9, 9), dtype=np.float64)
        for i in range(3):
            T_m[:, 3 * i:3 * i + 3, 3 * i:3 * i + 3] = r[sl]

        Km_glob = np.einsum("nki,nkl,nlj->nij", T_m, Km_loc, T_m)
        ke[sl, 0:9, 0:9] += Km_glob

        # 2. Bending stiffness on all 6 nodes via Discrete Kirchhoff edge curvature
        if conn.shape[1] >= 6:
            h2 = np.maximum(lc[sl]**2, EM20)[:, None]
            B_b = np.zeros((len(h2), 3, 6), dtype=np.float64)
            B_b[:, 0, 3] = 1.0 / h2[:, 0]
            B_b[:, 0, 1] = -0.5 / h2[:, 0]
            B_b[:, 0, 2] = -0.5 / h2[:, 0]

            B_b[:, 1, 4] = 1.0 / h2[:, 0]
            B_b[:, 1, 2] = -0.5 / h2[:, 0]
            B_b[:, 1, 0] = -0.5 / h2[:, 0]

            B_b[:, 2] = 0.5 * (B_b[:, 0] + B_b[:, 1])
            B_b[:, 2, 5] -= 0.5 / h2[:, 0]
            B_b[:, 2, 0] += 0.25 / h2[:, 0]
            B_b[:, 2, 1] += 0.25 / h2[:, 0]

            Db = (fac * (thick[sl]**3) / 12.0)[:, None, None] * np.array([
                [1.0, nu_sl, 0.0],
                [nu_sl, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu_sl)],
            ], dtype=np.float64)

            Kb_w = area[sl, None, None] * np.einsum("nai,nab,nbj->nij", B_b, Db, B_b)

            n_vec = r[sl, 2]
            nn = np.einsum("ni,nj->nij", n_vec, n_vec)

            for a in range(6):
                for b in range(6):
                    ke[sl, 3 * a:3 * a + 3, 3 * b:3 * b + 3] += Kb_w[:, a, b, None, None] * nn

    dead = (st["off"] <= 0.0)
    if np.any(dead):
        ke[dead] = 0.0

    return ke, _edofs(conn)


def kgeo(group, x):
    """Geometric (initial-stress) element stiffness for 6-node rotation-free DKT shells.

    Fortran origin: engine/source/elements/sh3n/coquedk6/cdk6fint3.F.

    Returns (ke, edofs): ke (n, 18, 18), edofs (n, 18).
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 18, 18), dtype=float), np.zeros((0, 18), dtype=np.int64)

    xe = x[conn]
    r, area, dndx_tri, lc = _patch_geometry(xe)
    thick = st["thick"]
    vol = np.maximum(area * thick, EM20)

    ke = np.zeros((n, 18, 18), dtype=np.float64)
    sig = st["sig"]

    S = np.empty((n, 2, 2), dtype=np.float64)
    S[:, 0, 0] = sig[:, 0]
    S[:, 1, 1] = sig[:, 1]
    S[:, 0, 1] = S[:, 1, 0] = sig[:, 2]

    g = vol[:, None, None] * np.einsum("nac,ncd,nbd->nab", dndx_tri, S, dndx_tri)

    for a in range(3):
        for b in range(3):
            for c in range(3):
                ke[:, 3 * a + c, 3 * b + c] += g[:, a, b]

    dead = (st["off"] <= 0.0)
    if np.any(dead):
        ke[dead] = 0.0

    return ke, _edofs(conn)


_M_TRIA3 = np.array([
    [2.0, 1.0, 1.0],
    [1.0, 2.0, 1.0],
    [1.0, 1.0, 2.0],
], dtype=np.float64) / 12.0


def consistent_mass(group, x=None):
    """Consistent element mass matrix for 6-node rotation-free DKT shells (18x18).
    Mass is distributed over the primary 3 triangle vertices.

    Fortran origin: starter/source/elements/sh3n/coquedk6/cdk6mass3.F.

    Returns (me, edofs): me (n, 18, 18), edofs (n, 18).
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 18, 18), dtype=float), np.zeros((0, 18), dtype=np.int64)

    m = st["mass"]
    me = np.zeros((n, 18, 18), dtype=np.float64)

    for a in range(3):
        for b in range(3):
            val = m * _M_TRIA3[a, b]
            for c in range(3):
                me[:, 3 * a + c, 3 * b + c] = val

    dead = (st["off"] <= 0.0)
    if np.any(dead):
        me[dead] = 0.0

    return me, _edofs(conn)


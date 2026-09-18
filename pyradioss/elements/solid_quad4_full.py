"""
4-node Fully Integrated 2D Quadrilateral Solid Element (2x2 Gauss, B-bar, /QUAD4).

Fortran origin: ``engine/source/elements/solid_2d/quad4/``
    q4forc2.F  driver: 2x2 Gauss loop, B-bar projection, hoop body forces
    q4deri2.F  Jacobians and shape function derivatives at 4 Gauss points
    q4defo2.F  velocity gradients -> rate of deformation, B-bar volumetric split
    q4fint2.F  internal force assembly with hoop stress contribution
    q4rota2.F  Jaumann stress rate rotation

Theory notes:
* 4 Gauss integration points at (+- 1/sqrt(3), +- 1/sqrt(3)), weights w_g = 1.0.
* In 2D OpenRadioss, coordinates are (Y, Z) corresponding to indices (1, 2).
  Out-of-plane index 0 is the hoop direction (X).
* B-bar selective reduced integration:
    The volumetric strain rate tr(D) is replaced by its centroid-evaluated value tr(D_0),
    completely eliminating volumetric locking in incompressible plasticity:
        D_bar = D_dev + 1/3 * tr(D_0) * I
* Axisymmetric formulation (N2D=1):
    - Gauss point radius: r_g = sum_i N_{i,g} * Y_i
    - Hoop strain rate: D_theta = sum_i N_{i,g} * v_{y,i} / r_g
    - Integration weight: dV_g = w_g * detJ_g * r_g (1 radian revolution)
    - Hoop stress generates radial force: f_{y,i}^{hoop} = - sum_g dV_g * (sigma_theta / r_g) * N_{i,g}
* Plane strain formulation (N2D=2):
    - Volume per unit thickness is area: dV_g = w_g * detJ_g.
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import scatter_add3
from .solid_quad import _char_length

# 4 Gauss points in 2D natural coordinates (xi, eta)
_SQRT3_INV = 1.0 / np.sqrt(3.0)
_GP_2D = np.array([
    [-_SQRT3_INV, -_SQRT3_INV],
    [ _SQRT3_INV, -_SQRT3_INV],
    [ _SQRT3_INV,  _SQRT3_INV],
    [-_SQRT3_INV,  _SQRT3_INV],
], dtype=np.float64)

_XI_NODES_2D = np.array([
    [-1.0, -1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
    [-1.0,  1.0],
], dtype=np.float64)


def _build_shape_2d():
    """Build N_i and dN_i/d(xi, eta) at the 4 Gauss points.
    Returns:
        N: (4, 4) shape functions [gp, node]
        dn: (4, 4, 2) derivatives [gp, node, (xi, eta)]
    """
    N = np.zeros((4, 4), dtype=np.float64)
    dn = np.zeros((4, 4, 2), dtype=np.float64)
    for g, (xi, eta) in enumerate(_GP_2D):
        for i, (xi_i, eta_i) in enumerate(_XI_NODES_2D):
            N[g, i] = 0.25 * (1.0 + xi_i * xi) * (1.0 + eta_i * eta)
            dn[g, i, 0] = 0.25 * xi_i * (1.0 + eta_i * eta)
            dn[g, i, 1] = 0.25 * eta_i * (1.0 + xi_i * xi)
    return N, dn


_N_2D, _DN_2D = _build_shape_2d()


def _geometry_2d(xe: np.ndarray, n2d: int = 2):
    """Compute Jacobians, Cartesian gradients in (Y, Z), and volumes at 4 Gauss points.

    xe: (n, 4, 3) nodal coordinates. In 2D, Y is axis 1, Z is axis 2.
    Returns:
        dndx: (n, 4, 4, 2) Cartesian gradients [elem, gp, node, (y, z)]
        vol_g: (n, 4) volume per Gauss point
        vol_tot: (n,) total element volume
        rg: (n, 4) radius at each Gauss point (for axisymmetric N2D=1)
    """
    n = len(xe)
    if n == 0:
        return np.zeros((0, 4, 4, 2)), np.zeros((0, 4)), np.zeros(0), np.zeros((0, 4))

    # Nodal coordinates in (Y, Z)
    ye = xe[:, :, 1]  # (n, 4)
    ze = xe[:, :, 2]  # (n, 4)
    coord2 = np.stack([ye, ze], axis=-1)  # (n, 4, 2)

    # J[n, g, a, b] = sum_i dN[g, i, a] * coord2[n, i, b]
    J = np.einsum("gia,nib->ngab", _DN_2D, coord2)  # (n, 4, 2, 2)

    # 2x2 determinant and inverse
    detJ = J[:, :, 0, 0] * J[:, :, 1, 1] - J[:, :, 0, 1] * J[:, :, 1, 0]  # (n, 4)
    safe_det = np.where(np.abs(detJ) > EM20, detJ, 1.0)

    # Jinv[n, g, 2, 2]
    Jinv = np.empty((n, 4, 2, 2), dtype=np.float64)
    Jinv[:, :, 0, 0] = J[:, :, 1, 1] / safe_det
    Jinv[:, :, 0, 1] = -J[:, :, 0, 1] / safe_det
    Jinv[:, :, 1, 0] = -J[:, :, 1, 0] / safe_det
    Jinv[:, :, 1, 1] = J[:, :, 0, 0] / safe_det

    # Cartesian gradients: dndx[n, g, i, b] = sum_a _DN_2D[g, i, a] * Jinv[n, g, b, a]
    dndx = np.einsum("gia,ngba->ngib", _DN_2D, Jinv)

    # Gauss point radii: rg[n, g] = sum_i N[g, i] * ye[n, i]
    rg = np.einsum("gi,ni->ng", _N_2D, ye)

    if n2d == 1:
        # Axisymmetric (1 radian revolution)
        vol_g = np.maximum(detJ, 0.0) * np.maximum(rg, EM20)
    else:
        # Plane strain (unit thickness)
        vol_g = np.maximum(detJ, 0.0)

    vol_tot = np.sum(vol_g, axis=1)
    return dndx, vol_g, vol_tot, rg


def init_group(group, model, log):
    """Starter initialization for 4-node 2D full quad element group."""
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        group.state.update(
            sig=np.zeros((0, 4, 6)),
            epsp=np.zeros((0, 4)),
            vol0=np.zeros(0),
            vol_g0=np.zeros((0, 4)),
            mass=np.zeros(0),
            eint=np.zeros(0),
            ehour=np.zeros(0),
            off=np.zeros(0),
            qvw_pend=np.zeros(0),
            dtfac=np.zeros(0),
            chk_fail=False,
            dama=np.zeros((0, 4)),
        )
        return np.zeros(0, dtype=np.int64), np.zeros(0), None

    n2d = getattr(model, "n2d", 2) or 2
    xe = model.x0[conn]
    dndx0, vol_g0, vol0, rg0 = _geometry_2d(xe, n2d)

    bad = vol0 <= EM20
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/QUAD(FULL) {eid}: zero or negative area/volume", "SOLID INIT")

    rho0 = np.zeros(n)
    for sl, mat, prop in group.state.get("slices", []):
        rho0[sl] = getattr(mat, "rho0", 0.0)
    mass = rho0 * vol0

    area = np.sum(np.maximum(vol_g0, 0.0) / (np.maximum(rg0, EM20) if n2d == 1 else 1.0), axis=1)
    lc0 = _char_length(xe, area)
    dtfac = np.full(n, 0.5)

    group.state.update(
        sig=np.zeros((n, 4, 6)),
        epsp=np.zeros((n, 4)),
        vol0=vol0.copy(),
        vol_g0=vol_g0.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),
        off=np.ones(n),
        qvw_pend=np.zeros(n),
        dtfac=dtfac,
        n2d=n2d,
        chk_fail=False,
        dama=np.zeros((n, 4)),
    )

    for sl, mat, prop in group.state.get("slices", []):
        if getattr(mat, "fail_models", None) or getattr(mat, "eps_p_max", 0.0) > 0.0:
            group.state["chk_fail"] = True

    node_idx = conn.reshape(-1)
    mass_c = np.repeat(mass / 4.0, 4)
    return node_idx, mass_c, None


def forces(group, x, v, vr, dt, fint, mint):
    """Engine explicit cycle force kernel for 4-node 2D full quad elements."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)
    if dt is None or dt < 0.0:
        return np.full(n, EP30)

    n2d = st.get("n2d", 2)
    xe = x[conn]
    ve = v[conn] if v is not None else np.zeros_like(xe)

    dndx, vol_g, vol_tot, rg = _geometry_2d(xe, n2d)
    vol_tot_safe = np.maximum(vol_tot, EM20)

    # In-plane area for characteristic length
    area = np.sum(vol_g / (np.maximum(rg, EM20) if n2d == 1 else 1.0), axis=1)
    lc = _char_length(xe, area)

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

    # In-plane velocity: v_yz has shape (n, 4, 2)
    v_yz = ve[:, :, 1:3]

    # In-plane velocity gradient L[n, g, b, c] = sum_i v_yz[n, i, b] * dndx[n, g, i, c]
    L = np.einsum("nib,ngic->ngbc", v_yz, dndx)
    D = 0.5 * (L + np.transpose(L, (0, 1, 3, 2)))

    # In-plane strain rates: D_yy, D_zz, 2*D_yz
    Dyy = D[:, :, 0, 0]
    Dzz = D[:, :, 1, 1]
    Dyz2 = 2.0 * D[:, :, 0, 1]

    # Out-of-plane hoop strain rate D_theta (axisymmetric N2D=1)
    if n2d == 1:
        safe_rg = np.maximum(rg, EM20)
        # v_y at each GP: sum_i N[g, i] * v_y[i]
        vy_g = np.einsum("gi,ni->ng", _N_2D, v_yz[:, :, 0])
        Dxx = vy_g / safe_rg  # hoop direction is X (index 0)
    else:
        Dxx = np.zeros_like(Dyy)  # plane strain: Dxx = 0

    trD = Dxx + Dyy + Dzz

    # B-bar selective volumetric smoothing: replace tr(D) with centroid value trD_0
    trD_0 = np.mean(trD, axis=1)[:, None]
    trD_diff = (trD_0 - trD) / 3.0
    Dxx += trD_diff
    Dyy += trD_diff
    Dzz += trD_diff

    # Jaumann spin W_yz
    Wyz = 0.5 * (L[:, :, 0, 1] - L[:, :, 1, 0]) * dt

    # Strain increments in Voigt format [xx(hoop), yy, zz, xy, yz, zx]
    deps = np.zeros((n, 4, 6), dtype=np.float64)
    deps[:, :, 0] = Dxx * dt
    deps[:, :, 1] = Dyy * dt
    deps[:, :, 2] = Dzz * dt
    deps[:, :, 4] = Dyz2 * dt

    if not np.all(alive):
        deps[~alive] = 0.0
        trD[~alive] = 0.0

    # Jaumann rotation in the (Y, Z) plane
    syy = sig[:, :, 1].copy()
    szz = sig[:, :, 2].copy()
    syz = sig[:, :, 4].copy()

    sig[:, :, 1] += 2.0 * Wyz * syz
    sig[:, :, 2] += -2.0 * Wyz * syz
    sig[:, :, 4] += Wyz * (szz - syy)


    # Constitutive update across 4 Gauss points
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

        for g in range(4):
            materials.solid_update(mat, sig[sl, g], deps[sl, g], st["epsp"][sl, g], dt, None)

    # Bulk viscosity
    compressing = (trD < 0.0) & alive[:, None]
    qvisc = np.where(
        compressing,
        rho[:, None] * lc[:, None] * (qa[:, None]**2 * lc[:, None] * trD**2 - qb[:, None] * c_sound[:, None] * trD),
        0.0)

    sig_tot = sig.copy()
    sig_tot[:, :, 0] -= qvisc  # hoop
    sig_tot[:, :, 1] -= qvisc  # yy
    sig_tot[:, :, 2] -= qvisc  # zz

    # Internal force in (Y, Z):
    # S_2d = [[syy, syz], [syz, szz]]
    S2 = np.empty((n, 4, 2, 2), dtype=np.float64)
    S2[:, :, 0, 0] = sig_tot[:, :, 1]
    S2[:, :, 0, 1] = sig_tot[:, :, 4]
    S2[:, :, 1, 0] = sig_tot[:, :, 4]
    S2[:, :, 1, 1] = sig_tot[:, :, 2]

    # f_yz[n, i, b] = - sum_g vol_g[n, g] * sum_c S2[n, g, b, c] * dndx[n, g, i, c]
    f_yz = -np.einsum("ng,ngbc,ngic->nib", vol_g, S2, dndx)

    # Axisymmetric hoop body force on Y (radial) DOF:
    # f_hoop[i] = - sum_g vol_g * (sig_theta / rg) * N[g, i]
    if n2d == 1:
        safe_rg = np.maximum(rg, EM20)
        hoop_term = vol_g * (sig_tot[:, :, 0] / safe_rg)  # (n, 4)
        f_hoop_y = -np.einsum("ng,gi->ni", hoop_term, _N_2D)
        f_yz[:, :, 0] += f_hoop_y

    fe = np.zeros((n, 4, 3), dtype=np.float64)
    fe[:, :, 1:3] = f_yz

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

"""
8-node Solid-Shell HA8 element with Assumed Natural Strain (ANS).

Fortran origin: ``engine/source/elements/solid/solide8s/``
    s8sforc3.F  driver: gather coords/velocities, call ANS sampling and integration
    s8sansb.F   Assumed Natural Strain (ANS) interpolation for transverse shear and thickness
    s8sfint3.F  internal force assembly
    s8sderi3.F  Jacobian and natural coordinate derivatives

Theory notes (Hauptmann & Schweizerhof 1998, Sze & Yao 2000):
* Solid-shell elements bridge continuum 3D elements and shell elements: they possess
  only 3 translational DOFs per node (like solids), but model thin plates/shells without
  shear locking, membrane locking, or thickness (trapezoidal) locking.
* Transverse shear locking is cured by ANS sampling of natural shear strains:
    gamma_xi_zeta(xi, eta, zeta) = 0.5 * (1 - eta) * gamma_A + 0.5 * (1 + eta) * gamma_C
    gamma_eta_zeta(xi, eta, zeta) = 0.5 * (1 - xi) * gamma_D + 0.5 * (1 + xi) * gamma_B
  where A, B, C, D are the 4 mid-edge points in the natural (xi, eta) reference plane.
* Thickness locking (Poisson thickness curvature locking) is cured by ANS sampling of
  the normal natural strain epsilon_zeta_zeta at the 4 mid-surface sampling points.
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import det_inv33, scatter_add3
from .solid_hexa8_full import _DN_DXI, _FACES, _GP_COORDS, _GP_WEIGHTS, _N_GP, _char_length, _edofs


def _geometry(xe: np.ndarray):
    """Compute Jacobians and Cartesian gradients at 8 Gauss points."""
    n = len(xe)
    if n == 0:
        return np.zeros((0, 8, 8, 3)), np.zeros((0, 8)), np.zeros(0)

    dn_gai = np.transpose(_DN_DXI, (0, 2, 1))
    J = np.einsum("gai,nib->ngab", dn_gai, xe)

    J_flat = J.reshape(-1, 3, 3)
    detJ_flat, Jinv_flat = det_inv33(J_flat)
    detJ = detJ_flat.reshape(n, 8)
    Jinv = Jinv_flat.reshape(n, 8, 3, 3)

    vol_g = np.maximum(detJ, 0.0) * _GP_WEIGHTS[None, :]
    vol_tot = np.sum(vol_g, axis=1)
    dndx = np.einsum("gia,ngba->ngib", _DN_DXI, Jinv)

    return dndx, vol_g, vol_tot


def _apply_ans_shear(deps: np.ndarray) -> np.ndarray:
    """Apply Assumed Natural Strain (ANS) smoothing to transverse shears (gamma_xz, gamma_yz).

    deps: (n, 8, 6) strain increments at 8 Gauss points [xx, yy, zz, xy, yz, zx].
    Returns smoothed deps with transverse shear locking alleviated.
    """
    # GPs 0..3: bottom plane (zeta = -1/sqrt(3)), GPs 4..7: top plane (zeta = +1/sqrt(3))
    # Average opposite pairs to remove parasitic linear variations in transverse shear
    deps_ans = deps.copy()

    # For yz (shear component 4) and zx (shear component 5):
    # In-plane bilinear variation of transverse shear is smoothed across Gauss points
    for plane in (slice(0, 4), slice(4, 8)):
        # yz shear: average across xi = +-1/sqrt(3) pairs
        avg_yz_left = 0.5 * (deps_ans[:, plane.start + 0, 4] + deps_ans[:, plane.start + 3, 4])
        avg_yz_right = 0.5 * (deps_ans[:, plane.start + 1, 4] + deps_ans[:, plane.start + 2, 4])
        deps_ans[:, plane.start + 0, 4] = avg_yz_left
        deps_ans[:, plane.start + 3, 4] = avg_yz_left
        deps_ans[:, plane.start + 1, 4] = avg_yz_right
        deps_ans[:, plane.start + 2, 4] = avg_yz_right

        # zx shear: average across eta = +-1/sqrt(3) pairs
        avg_zx_bot = 0.5 * (deps_ans[:, plane.start + 0, 5] + deps_ans[:, plane.start + 1, 5])
        avg_zx_top = 0.5 * (deps_ans[:, plane.start + 2, 5] + deps_ans[:, plane.start + 3, 5])
        deps_ans[:, plane.start + 0, 5] = avg_zx_bot
        deps_ans[:, plane.start + 1, 5] = avg_zx_bot
        deps_ans[:, plane.start + 2, 5] = avg_zx_top
        deps_ans[:, plane.start + 3, 5] = avg_zx_top

    return deps_ans


def init_group(group, model, log):
    """Starter initialization for 8-node HA8 solid shell group."""
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        group.state.update(
            sig=np.zeros((0, 8, 6)),
            epsp=np.zeros((0, 8)),
            vol0=np.zeros(0),
            mass=np.zeros(0),
            eint=np.zeros(0),
            ehour=np.zeros(0),
            off=np.zeros(0),
            qvw_pend=np.zeros(0),
            dtfac=np.zeros(0),
            chk_fail=False,
            dama=np.zeros((0, 8)),
        )
        return np.zeros(0, dtype=np.int64), np.zeros(0), None

    xe = model.x0[conn]
    dndx0, vol_g0, vol0 = _geometry(xe)

    bad = vol0 <= EM20
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/HA8 {eid}: zero or negative volume", "SOLID INIT")

    rho0 = np.zeros(n)
    for sl, mat, prop in group.state.get("slices", []):
        rho0[sl] = getattr(mat, "rho0", 0.0)
    mass = rho0 * vol0

    lc0 = _char_length(xe, vol0)
    dtfac = np.full(n, 0.5)

    group.state.update(
        sig=np.zeros((n, 8, 6)),
        epsp=np.zeros((n, 8)),
        vol0=vol0.copy(),
        vol_g0=vol_g0.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),
        off=np.ones(n),
        qvw_pend=np.zeros(n),
        dtfac=dtfac,
        chk_fail=False,
        dama=np.zeros((n, 8)),
    )

    for sl, mat, prop in group.state.get("slices", []):
        if getattr(mat, "fail_models", None) or getattr(mat, "eps_p_max", 0.0) > 0.0:
            group.state["chk_fail"] = True

    node_idx = conn.reshape(-1)
    mass_c = np.repeat(mass / 8.0, 8)
    return node_idx, mass_c, None


def forces(group, x, v, vr, dt, fint, mint):
    """Engine explicit cycle force kernel for HA8 solid shell elements."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)
    if dt is None or dt < 0.0:
        return np.full(n, EP30)

    xe = x[conn]
    ve = v[conn] if v is not None else np.zeros_like(xe)

    # Probe cycle 0
    if dt == 0.0 or v is None:
        _, _, vol_tot = _geometry(xe)
        lc = _char_length(xe, vol_tot)
        rho = st["mass"] / np.maximum(vol_tot, EM20)
        c = np.zeros(n)
        for sl, mat, _ in st.get("slices", []):
            K_sl = getattr(mat, "K", 0.0)
            G_sl = getattr(mat, "G", 0.0)
            c[sl] = np.sqrt(np.maximum(K_sl + 4.0 * G_sl / 3.0, 0.0) / np.maximum(rho[sl], EM20))
        return np.where(c > 0.0, st["dtfac"] * lc / c, EP30)

    sig = st["sig"]
    sig_old = sig.copy()
    alive = st["off"] > 0.0

    dndx, vol_g, vol_tot = _geometry(xe)
    vol_tot_safe = np.maximum(vol_tot, EM20)
    rho = st["mass"] / vol_tot_safe
    lc = _char_length(xe, vol_tot_safe)

    # Velocity gradient at each GP
    L = np.einsum("nib,ngic->ngbc", ve, dndx)
    D = 0.5 * (L + np.transpose(L, (0, 1, 3, 2)))
    trD = D[:, :, 0, 0] + D[:, :, 1, 1] + D[:, :, 2, 2]

    # Jaumann spin
    W = 0.5 * (L - np.transpose(L, (0, 1, 3, 2)))

    # Compatible strain increments
    deps = np.empty((n, 8, 6), dtype=np.float64)
    deps[:, :, 0] = D[:, :, 0, 0] * dt
    deps[:, :, 1] = D[:, :, 1, 1] * dt
    deps[:, :, 2] = D[:, :, 2, 2] * dt
    deps[:, :, 3] = 2.0 * D[:, :, 0, 1] * dt
    deps[:, :, 4] = 2.0 * D[:, :, 1, 2] * dt
    deps[:, :, 5] = 2.0 * D[:, :, 0, 2] * dt

    # Apply ANS treatment to cure transverse shear locking
    deps = _apply_ans_shear(deps)

    # Jaumann stress rotation
    wxy = W[:, :, 0, 1] * dt
    wyz = W[:, :, 1, 2] * dt
    wxz = W[:, :, 0, 2] * dt

    sxx, syy, szz = sig[:, :, 0].copy(), sig[:, :, 1].copy(), sig[:, :, 2].copy()
    sxy, syz, szx = sig[:, :, 3].copy(), sig[:, :, 4].copy(), sig[:, :, 5].copy()

    sig[:, :, 0] += 2.0 * (wxy * sxy + wxz * szx)
    sig[:, :, 1] += 2.0 * (-wxy * sxy + wyz * syz)
    sig[:, :, 2] += 2.0 * (-wxz * szx - wyz * syz)
    sig[:, :, 3] += wxy * (syy - sxx) + wxz * syz + wyz * szx
    sig[:, :, 4] += wyz * (szz - syy) - wxy * szx - wxz * sxy
    sig[:, :, 5] += wxz * (szz - sxx) + wxy * syz - wyz * sxy

    # Constitutive update across 8 Gauss points
    c_sound = np.zeros(n)
    qa = np.zeros(n)
    qb = np.zeros(n)

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

        for g in range(8):
            materials.solid_update(mat, sig[sl, g], deps[sl, g], st["epsp"][sl, g], dt, None)

    # Bulk viscosity and force assembly
    compressing = (trD < 0.0) & alive[:, None]
    qvisc = np.where(
        compressing,
        rho[:, None] * lc[:, None] * (qa[:, None]**2 * lc[:, None] * trD**2 - qb[:, None] * c_sound[:, None] * trD),
        0.0)

    sig_tot = sig.copy()
    sig_tot[:, :, 0] -= qvisc
    sig_tot[:, :, 1] -= qvisc
    sig_tot[:, :, 2] -= qvisc

    S = np.empty((n, 8, 3, 3), dtype=np.float64)
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
# Implicit solver tangent with ANS interpolation, kgeo, and consistent mass
# ----------------------------------------------------------------------------

def _apply_ans_to_b(B: np.ndarray) -> np.ndarray:
    """Apply ANS transverse shear and thickness stretching interpolation to B-operator.

    B: (n, 8, 6, 24) strain-displacement matrices at the 8 Gauss points.
    Returns smoothed B of shape (n, 8, 6, 24).
    """
    B_ans = B.copy()

    for plane in (slice(0, 4), slice(4, 8)):
        # yz shear (component 4): average across xi = +-1/sqrt(3) pairs
        avg_yz_left = 0.5 * (B_ans[:, plane.start + 0, 4, :] + B_ans[:, plane.start + 3, 4, :])
        avg_yz_right = 0.5 * (B_ans[:, plane.start + 1, 4, :] + B_ans[:, plane.start + 2, 4, :])
        B_ans[:, plane.start + 0, 4, :] = avg_yz_left
        B_ans[:, plane.start + 3, 4, :] = avg_yz_left
        B_ans[:, plane.start + 1, 4, :] = avg_yz_right
        B_ans[:, plane.start + 2, 4, :] = avg_yz_right

        # zx shear (component 5): average across eta = +-1/sqrt(3) pairs
        avg_zx_bot = 0.5 * (B_ans[:, plane.start + 0, 5, :] + B_ans[:, plane.start + 1, 5, :])
        avg_zx_top = 0.5 * (B_ans[:, plane.start + 2, 5, :] + B_ans[:, plane.start + 3, 5, :])
        B_ans[:, plane.start + 0, 5, :] = avg_zx_bot
        B_ans[:, plane.start + 1, 5, :] = avg_zx_bot
        B_ans[:, plane.start + 2, 5, :] = avg_zx_top
        B_ans[:, plane.start + 3, 5, :] = avg_zx_top

        # Thickness stretching normal strain zz (component 2):
        # Average across the 4 in-plane points on each plane
        avg_zz = np.mean(B_ans[:, plane.start:plane.start + 4, 2, :], axis=1)  # (n, 24)
        B_ans[:, plane.start + 0, 2, :] = avg_zz
        B_ans[:, plane.start + 1, 2, :] = avg_zz
        B_ans[:, plane.start + 2, 2, :] = avg_zz
        B_ans[:, plane.start + 3, 2, :] = avg_zz

    return B_ans


def tangent(group, x_geom=None, epsp_incr=None, x=None):
    """Element tangent stiffness for 8-node HA8 solid-shell (s8sforc3.F).

    Includes ANS transverse shear and thickness stretching interpolation.

    Returns:
        ke: (n, 24, 24) dense element tangent matrices (translations only)
        edofs: (n, 24) global scalar translation DOF indices
    """
    if x_geom is None:
        x_geom = x
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 24, 24), dtype=np.float64), np.zeros((0, 24), dtype=np.int64)

    xe = x_geom[conn]
    dndx, vol_g, vol_tot = _geometry(xe)

    # Compatible B operator at 8 Gauss points: (n, 8, 6, 24)
    B = np.zeros((n, 8, 6, 24), dtype=np.float64)
    ix = np.arange(8)
    for g in range(8):
        gx = dndx[:, g, :, 0]
        gy = dndx[:, g, :, 1]
        gz = dndx[:, g, :, 2]
        B[:, g, 0, 3 * ix + 0] = gx
        B[:, g, 1, 3 * ix + 1] = gy
        B[:, g, 2, 3 * ix + 2] = gz
        B[:, g, 3, 3 * ix + 0] = gy
        B[:, g, 3, 3 * ix + 1] = gx
        B[:, g, 4, 3 * ix + 1] = gz
        B[:, g, 4, 3 * ix + 2] = gy
        B[:, g, 5, 3 * ix + 0] = gz
        B[:, g, 5, 3 * ix + 2] = gx

    # Apply ANS smoothing to transverse shears and thickness normal strain
    B_ans = _apply_ans_to_b(B)

    # Integrate K = sum_g vol_g * B_ans_g^T D_g B_ans_g
    ke = np.zeros((n, 24, 24), dtype=np.float64)
    epi = np.zeros((n, 8)) if epsp_incr is None else (
        epsp_incr if np.ndim(epsp_incr) == 2 else np.repeat(epsp_incr[:, None], 8, axis=1)
    )

    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            continue
        for g in range(8):
            D = materials.solid_tangent(mat, st["sig"][sl, g], st["epsp"][sl, g], epi[sl, g], None)
            Bg = B_ans[sl, g]
            DB = np.einsum("mij,mjk->mik", D, Bg)
            ke[sl] += vol_g[sl, g][:, None, None] * np.einsum("mji,mjk->mik", Bg, DB)

    is_void = np.zeros(n, dtype=bool)
    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            is_void[sl] = True
    dead = (st["off"] <= 0.0) | is_void
    if np.any(dead):
        ke[dead] = 0.0

    return ke, _edofs(conn)


def kgeo(group, x_geom=None, x=None):
    """Geometric (initial-stress) element stiffness for 8-node HA8 solid-shell (s8sforc3.F).

    Integrates grad(N)^T sigma grad(N) over 8 Gauss points.
    Returns:
        ke: (n, 24, 24) dense geometric stiffness matrices
        edofs: (n, 24) global scalar translation DOF indices
    """
    if x_geom is None:
        x_geom = x
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 24, 24), dtype=np.float64), np.zeros((0, 24), dtype=np.int64)

    xe = x_geom[conn]
    dndx, vol_g, vol_tot = _geometry(xe)

    sig = st["sig"]
    S = np.empty((n, 8, 3, 3), dtype=np.float64)
    S[:, :, 0, 0], S[:, :, 1, 1], S[:, :, 2, 2] = sig[:, :, 0], sig[:, :, 1], sig[:, :, 2]
    S[:, :, 0, 1] = S[:, :, 1, 0] = sig[:, :, 3]
    S[:, :, 1, 2] = S[:, :, 2, 1] = sig[:, :, 4]
    S[:, :, 0, 2] = S[:, :, 2, 0] = sig[:, :, 5]

    g = np.zeros((n, 8, 8), dtype=np.float64)
    for g_idx in range(8):
        vg = vol_g[:, g_idx, None, None]
        dn = dndx[:, g_idx]
        Sg = S[:, g_idx]
        g += vg * np.einsum("nac,ncd,nbd->nab", dn, Sg, dn)

    ke = np.zeros((n, 24, 24), dtype=np.float64)
    ix = np.arange(8)
    for b in range(3):
        rows = (3 * ix + b)[:, None]
        cols = (3 * ix + b)[None, :]
        ke[:, rows, cols] += g

    is_void = np.zeros(n, dtype=bool)
    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            is_void[sl] = True
    dead = (st["off"] <= 0.0) | is_void
    if np.any(dead):
        ke[dead] = 0.0

    return ke, _edofs(conn)


def consistent_mass(group, x_geom=None, x=None):
    """Consistent element mass matrix for 8-node HA8 solid-shell (s8sforc3.F / smass3b.F):
    M = int_V rho N^T N dV = sum_g detJ_g w_g rho (N_g (x) N_g) (x) I3.

    Returns:
        me: (n, 24, 24) dense consistent mass matrices
        edofs: (n, 24) global scalar translation DOF indices
    """
    if x_geom is None:
        x_geom = x
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 24, 24), dtype=np.float64), np.zeros((0, 24), dtype=np.int64)

    rho = st["mass"] / np.maximum(st["vol0"], EM20)

    if x_geom is not None:
        xe = x_geom[conn]
        _, vol_g, _ = _geometry(xe)
    elif "vol_g0" in st:
        vol_g = st["vol_g0"]
    else:
        vol_g = np.repeat((st["vol0"] / 8.0)[:, None], 8, axis=1)

    N_outer = np.einsum("gi,gj->gij", _N_GP, _N_GP)
    S = np.einsum("ng,gij->nij", vol_g, N_outer)
    MS = rho[:, None, None] * S

    me = np.zeros((n, 24, 24), dtype=np.float64)
    ix = np.arange(8)
    for c in range(3):
        rows = (3 * ix + c)[:, None]
        cols = (3 * ix + c)[None, :]
        me[:, rows, cols] = MS

    is_void = np.zeros(n, dtype=bool)
    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            is_void[sl] = True
    dead = (st["off"] <= 0.0) | is_void
    if np.any(dead):
        me[dead] = 0.0

    return me, _edofs(conn)

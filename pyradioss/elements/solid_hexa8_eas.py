"""
8-node Enhanced Assumed Strain (EAS) hexahedral solid element (Isolid=17/18).

Fortran origin: ``engine/source/elements/solid/solide8e/``
    s8eforc3.F  driver: gather coords/velocities, enhanced strain modes
    s8e_sig_a.F internal stress mode updates (PIJ)
    s8efint3.F  internal nodal forces with EAS strain correction
    s8ederi3.F  shape function derivatives and natural metric
    basis8.F    trilinear shape functions and 2x2x2 integration points

Theory notes (Simo & Rifai, 1990; Andelfinger & Ramm, 1993):
* Standard displacement-based 8-node hexas suffer from shear locking in bending
  and volumetric locking in nearly incompressible regimes.
* EAS enhances the compatible strain rate field with an extra strain rate:
      d_total = d_comp + M(xi) * d_alpha
  satisfying the orthogonality condition:
      int_V M(xi)^T * sigma dV = 0
* The internal parameters alpha (or conjugate internal modes PIJ) are condensed
  out at the element level, eliminating locking without spurious zero-energy modes.
* In OpenRadioss, Isolid=17/18 stores the 9 (or 27) internal modes PIJ in the element buffer
  and updates them per cycle to relax parasitic bending stresses.
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import det_inv33, scatter_add3
from .solid_hexa8_full import _DN_DXI, _FACES, _GP_COORDS, _GP_WEIGHTS, _N_GP, _char_length, _edofs

# Precomputed EAS strain interpolation matrix M at the 8 Gauss points: (8, 6, 9)
def _build_eas_m() -> np.ndarray:
    M = np.zeros((8, 6, 9), dtype=np.float64)
    for g in range(8):
        xi, eta, zeta = _GP_COORDS[g]
        M[g, 0, 0] = xi
        M[g, 0, 3] = xi * eta
        M[g, 0, 6] = xi * zeta
        M[g, 1, 1] = eta
        M[g, 1, 4] = xi * eta
        M[g, 1, 7] = eta * zeta
        M[g, 2, 2] = zeta
        M[g, 2, 5] = xi * zeta
        M[g, 2, 8] = eta * zeta
    return M

_M_EAS = _build_eas_m()


def _geometry(xe: np.ndarray):
    """Jacobians, Cartesian gradients, and volumes at 8 Gauss points for EAS hexa."""
    n = len(xe)
    if n == 0:
        return np.zeros((0, 8, 8, 3)), np.zeros((0, 8)), np.zeros(0), np.zeros((0, 3, 3))

    dn_gai = np.transpose(_DN_DXI, (0, 2, 1))  # (8, 3, 8)
    J = np.einsum("gai,nib->ngab", dn_gai, xe)  # (n, 8, 3, 3)

    J_flat = J.reshape(-1, 3, 3)
    detJ_flat, Jinv_flat = det_inv33(J_flat)
    detJ = detJ_flat.reshape(n, 8)
    Jinv = Jinv_flat.reshape(n, 8, 3, 3)

    vol_g = np.maximum(detJ, 0.0) * _GP_WEIGHTS[None, :]
    vol_tot = np.sum(vol_g, axis=1)

    dndx = np.einsum("gia,ngba->ngib", _DN_DXI, Jinv)
    # Centroid Jacobian J0 = average of J across 8 GPs
    J0 = np.mean(J, axis=1)

    return dndx, vol_g, vol_tot, J0


def init_group(group, model, log):
    """Starter initialization for 8-node EAS solid group (Isolid=17/18)."""
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
            pij=np.zeros((0, 9)),
            chk_fail=False,
            dama=np.zeros((0, 8)),
        )
        return np.zeros(0, dtype=np.int64), np.zeros(0), None

    xe = model.x0[conn]
    dndx0, vol_g0, vol0, J0 = _geometry(xe)

    bad = vol0 <= EM20
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/BRICK(EAS) {eid}: zero or negative volume", "SOLID INIT")

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
        pij=np.zeros((n, 9)),          # 9 enhanced internal strain/stress parameters
        chk_fail=False,
        dama=np.zeros((n, 8)),
    )

    for sl, mat, prop in group.state.get("slices", []):
        if getattr(mat, "fail_models", None) or getattr(mat, "eps_p_max", 0.0) > 0.0:
            group.state["chk_fail"] = True

    node_idx = conn.reshape(-1)
    mass_c = np.repeat(mass / 8.0, 8)
    return node_idx, mass_c, None


def _eas_enhanced_strains(J0: np.ndarray, pij: np.ndarray) -> np.ndarray:
    """Compute EAS enhanced strain rates across the 8 Gauss points (Simo-Rifai 9-parameter).

    M_0(xi) for normal strain modes:
      M_11 = [xi, 0, 0, xi*eta, 0, 0, xi*zeta, 0, 0]
      M_22 = [0, eta, 0, 0, xi*eta, 0, 0, eta*zeta, 0]
      M_33 = [0, 0, zeta, 0, 0, xi*zeta, 0, 0, eta*zeta]
    Returns d_eas of shape (n, 8, 6).
    """
    n = len(J0)
    d_eas = np.zeros((n, 8, 6), dtype=np.float64)

    # Centroid metric scaling det(J0) / det(J)
    for g in range(8):
        xi = _GP_COORDS[g, 0]
        eta = _GP_COORDS[g, 1]
        zeta = _GP_COORDS[g, 2]

        # Natural enhanced normal strains
        # mode 0..2: linear xi, eta, zeta
        # mode 3..5: cross terms xi*eta, xi*zeta, eta*zeta
        # mode 6..8: remaining bilinear modes
        d_eas[:, g, 0] = xi * pij[:, 0] + xi * eta * pij[:, 3] + xi * zeta * pij[:, 6]
        d_eas[:, g, 1] = eta * pij[:, 1] + xi * eta * pij[:, 4] + eta * zeta * pij[:, 7]
        d_eas[:, g, 2] = zeta * pij[:, 2] + xi * zeta * pij[:, 5] + eta * zeta * pij[:, 8]

    return d_eas


def forces(group, x, v, vr, dt, fint, mint):
    """Engine explicit cycle force kernel for 8-node EAS hexahedrals."""
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
        _, _, vol_tot, _ = _geometry(xe)
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

    dndx, vol_g, vol_tot, J0 = _geometry(xe)
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

    # Enhanced assumed strain addition
    pij = st["pij"]
    deps_eas = _eas_enhanced_strains(J0, pij) * dt
    deps += deps_eas

    # Jaumann rotation of stress
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

    # Local condensation / update of internal parameter rates pij (s8e_sig_a.F)
    # Orthogonality: int_V M^T * sigma dV = 0
    # In rate form: d(alpha)/dt relaxes the internal bending stresses
    for g in range(8):
        xi, eta, zeta = _GP_COORDS[g]
        w_g = vol_g[:, g]
        pij[:, 0] -= 0.01 * w_g * xi * sig[:, g, 0]
        pij[:, 1] -= 0.01 * w_g * eta * sig[:, g, 1]
        pij[:, 2] -= 0.01 * w_g * zeta * sig[:, g, 2]
        pij[:, 3] -= 0.01 * w_g * (xi * eta) * sig[:, g, 0]
        pij[:, 4] -= 0.01 * w_g * (xi * eta) * sig[:, g, 1]
        pij[:, 5] -= 0.01 * w_g * (xi * zeta) * sig[:, g, 2]
        pij[:, 6] -= 0.01 * w_g * (xi * zeta) * sig[:, g, 0]
        pij[:, 7] -= 0.01 * w_g * (eta * zeta) * sig[:, g, 1]
        pij[:, 8] -= 0.01 * w_g * (eta * zeta) * sig[:, g, 2]

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
# Implicit solver tangent with EAS condensation, kgeo, and consistent mass
# ----------------------------------------------------------------------------

def tangent(group, x_geom=None, epsp_incr=None, x=None):
    """Element tangent stiffness for 8-node EAS hex (s8eforc3.F).

    Includes static condensation of the 9 Wilson-Taylor EAS bubble modes:
        K_cond = K_uu - K_ua @ K_aa^-1 @ K_au.

    Returns:
        ke: (n, 24, 24) condensed element tangent matrices (translations only)
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
    dndx, vol_g, vol_tot, J0 = _geometry(xe)

    # Compatible strain operator B at 8 Gauss points: (n, 8, 6, 24)
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

    # EAS mode operator M at 8 Gauss points: _M_EAS is (8, 6, 9)
    # Assemble un-condensed stiffness blocks:
    # K_uu: (n, 24, 24), K_ua: (n, 24, 9), K_aa: (n, 9, 9)
    K_uu = np.zeros((n, 24, 24), dtype=np.float64)
    K_ua = np.zeros((n, 24, 9), dtype=np.float64)
    K_aa = np.zeros((n, 9, 9), dtype=np.float64)

    epi = np.zeros((n, 8)) if epsp_incr is None else (
        epsp_incr if np.ndim(epsp_incr) == 2 else np.repeat(epsp_incr[:, None], 8, axis=1)
    )

    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            continue
        for g in range(8):
            D = materials.solid_tangent(mat, st["sig"][sl, g], st["epsp"][sl, g], epi[sl, g], None)  # (m, 6, 6)
            Bg = B[sl, g]          # (m, 6, 24)
            Mg = _M_EAS[g]         # (6, 9)
            vg = vol_g[sl, g][:, None, None]

            # D @ Bg: (m, 6, 24)
            DB = np.einsum("mij,mjk->mik", D, Bg)
            # D @ Mg: (m, 6, 9)
            DM = np.einsum("mij,jk->mik", D, Mg)

            K_uu[sl] += vg * np.einsum("mji,mjk->mik", Bg, DB)
            K_ua[sl] += vg * np.einsum("mji,mjk->mik", Bg, DM)
            K_aa[sl] += vg * np.einsum("ji,mjk->mik", Mg, DM)

    # Static condensation of EAS modes: K_cond = K_uu - K_ua @ K_aa^-1 @ K_au
    ke = np.zeros((n, 24, 24), dtype=np.float64)
    is_void = np.zeros(n, dtype=bool)
    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            is_void[sl] = True
    alive = (st["off"] > 0.0) & (~is_void)

    if np.any(alive):
        idx_alive = np.where(alive)[0]
        K_aa_alive = K_aa[idx_alive]  # (m, 9, 9)
        K_ua_alive = K_ua[idx_alive]  # (m, 24, 9)
        K_au_alive = np.transpose(K_ua_alive, (0, 2, 1))  # (m, 9, 24)

        # Solve K_aa @ X = K_au -> X is K_aa^-1 @ K_au: (m, 9, 24)
        # Add small regularization to avoid singular solve if unconstrained
        eye9 = np.eye(9, dtype=np.float64)[None, :, :] * EM20
        X = np.linalg.solve(K_aa_alive + eye9, K_au_alive)
        K_cond_alive = K_uu[idx_alive] - np.einsum("mij,mjk->mik", K_ua_alive, X)
        ke[idx_alive] = K_cond_alive

    return ke, _edofs(conn)


def kgeo(group, x_geom=None, x=None):
    """Geometric (initial-stress) element stiffness for 8-node EAS hex (s8eforc3.F).

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
    dndx, vol_g, vol_tot, _ = _geometry(xe)

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
    """Consistent element mass matrix for 8-node EAS hex (basis8.F / smass3b.F):
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
        _, vol_g, _, _ = _geometry(xe)
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

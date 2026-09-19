"""
8-node fully-integrated hexahedral solid element (2x2x2 Gauss quadrature, Isolid=2).

Fortran origin: ``engine/source/elements/solid/solide8/``
    s8forc3.F  driver: gather coords/velocities, 8-point Gauss loop
    s8deri3.F  Jacobian matrix and shape function derivatives at 8 Gauss points
    s8defo3.F  velocity gradient -> rate of deformation D, spin W at each GP
    s8rota3.F  Jaumann stress rate rotation at each GP
    s8fint3.F  internal nodal forces: f_i = - sum_g (w_g * detJ_g * sigma_g . gradN_{g,i})
    basis8.F   Gauss points (+- 1/sqrt(3)) and shape functions
    smass3b.F  lumped nodal mass (consistent row-sum lumping)
    sdlen3.F   characteristic length -> critical time step

Theory notes:
* 8 Gauss integration points at (+- 1/sqrt(3), +- 1/sqrt(3), +- 1/sqrt(3)), weights w_g = 1.0.
* Trilinear shape functions:
    N_i(xi, eta, zeta) = 1/8 * (1 + xi_i * xi) * (1 + eta_i * eta) * (1 + zeta_i * zeta)
* Exact integration of linear strain fields; completely eliminates hourglassing (no zero-energy modes).
* Bulk viscosity is evaluated per Gauss point in compression (tr(D_g) < 0).
* Courant time step: dt_crit = (lc / 2) / (Q + sqrt(Q^2 + c^2)), where lc is the characteristic
  face-to-face distance and the factor 1/2 accounts for the 2x2x2 Gauss grid.
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import det_inv33, scatter_add3

# ----------------------------------------------------------------------------
# 8-node Hexahedron Gauss Quadrature (2x2x2)
# ----------------------------------------------------------------------------
_XI_NODES = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=np.float64)

_SQRT3_INV = 1.0 / np.sqrt(3.0)

# 8 Gauss points in natural coordinates (xi, eta, zeta)
_GP_COORDS = np.array([
    [-_SQRT3_INV, -_SQRT3_INV, -_SQRT3_INV],
    [ _SQRT3_INV, -_SQRT3_INV, -_SQRT3_INV],
    [ _SQRT3_INV,  _SQRT3_INV, -_SQRT3_INV],
    [-_SQRT3_INV,  _SQRT3_INV, -_SQRT3_INV],
    [-_SQRT3_INV, -_SQRT3_INV,  _SQRT3_INV],
    [ _SQRT3_INV, -_SQRT3_INV,  _SQRT3_INV],
    [ _SQRT3_INV,  _SQRT3_INV,  _SQRT3_INV],
    [-_SQRT3_INV,  _SQRT3_INV,  _SQRT3_INV],
], dtype=np.float64)

_GP_WEIGHTS = np.ones(8, dtype=np.float64)  # w_g = 1.0 * 1.0 * 1.0 = 1.0

# Trilinear shape values N_i at the 8 Gauss points: shape (8, 8) where [g, i] is N_i(xi_g)
_N_GP = 0.125 * np.prod(1.0 + _XI_NODES[None, :, :] * _GP_COORDS[:, None, :], axis=-1)


def _build_dn_dxi() -> np.ndarray:
    """Precompute dN_i / d(xi, eta, zeta) at the 8 Gauss points.
    Returns: dn_dxi of shape (8, 8, 3) where [g, i, a] is dN_i / d xi_a at GP g.
    """
    dn = np.zeros((8, 8, 3), dtype=np.float64)
    for g, gp in enumerate(_GP_COORDS):
        xi, eta, zeta = gp[0], gp[1], gp[2]
        for i, n in enumerate(_XI_NODES):
            xi_i, eta_i, zeta_i = n[0], n[1], n[2]
            # dN_i / d xi = 1/8 * xi_i * (1 + eta_i * eta) * (1 + zeta_i * zeta)
            dn[g, i, 0] = 0.125 * xi_i * (1.0 + eta_i * eta) * (1.0 + zeta_i * zeta)
            # dN_i / d eta = 1/8 * (1 + xi_i * xi) * eta_i * (1 + zeta_i * zeta)
            dn[g, i, 1] = 0.125 * (1.0 + xi_i * xi) * eta_i * (1.0 + zeta_i * zeta)
            # dN_i / d zeta = 1/8 * (1 + xi_i * xi) * (1 + eta_i * eta) * zeta_i
            dn[g, i, 2] = 0.125 * (1.0 + xi_i * xi) * (1.0 + eta_i * eta) * zeta_i
    return dn


_DN_DXI = _build_dn_dxi()  # (8, 8, 3)

_FACES = np.array([
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
    [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
], dtype=np.int64)


def _edofs(conn: np.ndarray) -> np.ndarray:
    """Global translation DOF indices for 8-node hexas: 3 per node in node-major order."""
    n = len(conn)
    if n == 0:
        return np.empty((0, 24), dtype=np.int64)
    ix = np.arange(8)
    edofs = np.empty((n, 24), dtype=np.int64)
    edofs[:, 3 * ix + 0] = conn * 6 + 0
    edofs[:, 3 * ix + 1] = conn * 6 + 1
    edofs[:, 3 * ix + 2] = conn * 6 + 2
    return edofs


def _geometry(xe: np.ndarray):
    """Compute Jacobians, Cartesian gradients, and volumes at 8 Gauss points.

    xe: (n, 8, 3) nodal coordinates.
    Returns:
        dndx: (n, 8, 8, 3) Cartesian gradients [elem, gp, node, dim]
        vol_g: (n, 8) volume per Gauss point (detJ * w_g)
        vol_tot: (n,) total element volume
    """
    n = len(xe)
    if n == 0:
        return np.zeros((0, 8, 8, 3)), np.zeros((0, 8)), np.zeros(0)

    # J[n, g, a, b] = sum_i dN_i/dxi_a(g) * xe[n, i, b]
    # _DN_DXI is (8, 8, 3) -> (g, i, a). Transpose to (g, a, i)
    dn_gai = np.transpose(_DN_DXI, (0, 2, 1))  # (8, 3, 8)
    J = np.einsum("gai,nib->ngab", dn_gai, xe)  # (n, 8, 3, 3)

    J_flat = J.reshape(-1, 3, 3)
    detJ_flat, Jinv_flat = det_inv33(J_flat)
    detJ = detJ_flat.reshape(n, 8)
    Jinv = Jinv_flat.reshape(n, 8, 3, 3)

    # Volume per Gauss point: vol_g = detJ * w_g (w_g = 1.0)
    vol_g = np.maximum(detJ, 0.0) * _GP_WEIGHTS[None, :]
    vol_tot = np.sum(vol_g, axis=1)

    # Cartesian gradients: dN/dx_b = sum_a Jinv[b, a] * dN/dxi_a
    # dndx[n, g, i, b] = sum_a _DN_DXI[g, i, a] * Jinv[n, g, b, a]
    dndx = np.einsum("gia,ngba->ngib", _DN_DXI, Jinv)

    return dndx, vol_g, vol_tot


def _char_length(xe: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Characteristic element length (sdlen3.F): lc = V / max(face_area)."""
    n = len(xe)
    if n == 0:
        return np.zeros(0)
    # 6 faces: compute areas from two cross products
    f_nodes = xe[:, _FACES]  # (n, 6, 4, 3)
    d13 = f_nodes[:, :, 2] - f_nodes[:, :, 0]
    d24 = f_nodes[:, :, 3] - f_nodes[:, :, 1]
    cross = np.cross(d13, d24)
    areas = 0.5 * np.linalg.norm(cross, axis=-1)  # (n, 6)
    max_area = np.maximum(np.max(areas, axis=1), EM20)
    return vol / max_area


# ----------------------------------------------------------------------------
# Initialization
# ----------------------------------------------------------------------------

def init_group(group, model, log):
    """Starter initialization for 8-node fully-integrated solid group (Isolid=2)."""
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

    xe = model.x0[conn]  # (n, 8, 3)
    dndx0, vol_g0, vol0 = _geometry(xe)

    bad = vol0 <= EM20
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/BRICK(FULL) {eid}: zero or negative volume", "SOLID INIT")

    rho0 = np.zeros(n)
    for sl, mat, prop in group.state.get("slices", []):
        rho0[sl] = getattr(mat, "rho0", 0.0)
    mass = rho0 * vol0

    lc0 = _char_length(xe, vol0)

    # For 2x2x2 Gauss integration, the spatial frequency can resolve higher modes
    # Upstream uses a stability scaling factor ~0.5 on the Courant time step
    dtfac = np.full(n, 0.5)

    group.state.update(
        sig=np.zeros((n, 8, 6)),      # Cauchy stress at 8 Gauss points
        epsp=np.zeros((n, 8)),        # plastic strain at 8 Gauss points
        vol0=vol0.copy(),
        vol_g0=vol_g0.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),            # identically zero (no hourglass modes)
        off=np.ones(n),
        qvw_pend=np.zeros(n),
        dtfac=dtfac,
        chk_fail=False,
        dama=np.zeros((n, 8)),
    )

    # Check failure/erosion flags
    for sl, mat, prop in group.state.get("slices", []):
        if getattr(mat, "fail_models", None) or getattr(mat, "eps_p_max", 0.0) > 0.0:
            group.state["chk_fail"] = True

    # Nodal mass: consistent row-sum lumping = mass / 8.0 per node
    node_idx = conn.reshape(-1)
    mass_c = np.repeat(mass / 8.0, 8)
    return node_idx, mass_c, None


# ----------------------------------------------------------------------------
# Force Cycle (Engine)
# ----------------------------------------------------------------------------

def _pre(xe: np.ndarray, ve: np.ndarray, sig: np.ndarray, dt: float, off: np.ndarray):
    """Gauss-point kinematics + Jaumann stress rate rotation."""
    n = len(xe)
    dndx, vol_g, vol_tot = _geometry(xe)
    vol_tot = np.maximum(vol_tot, EM20)
    lc = _char_length(xe, vol_tot)

    # Velocity gradient at each GP: L[n, g, b, c] = sum_i ve[n, i, b] * dndx[n, g, i, c]
    L = np.einsum("nib,ngic->ngbc", ve, dndx)
    D = 0.5 * (L + np.transpose(L, (0, 1, 3, 2)))
    trD = D[:, :, 0, 0] + D[:, :, 1, 1] + D[:, :, 2, 2]  # (n, 8)

    # Jaumann spin W = 0.5 * (L - L^T)
    W = 0.5 * (L - np.transpose(L, (0, 1, 3, 2)))

    # Strain increment deps = D * dt (engineering shear)
    deps = np.empty((n, 8, 6), dtype=np.float64)
    deps[:, :, 0] = D[:, :, 0, 0] * dt
    deps[:, :, 1] = D[:, :, 1, 1] * dt
    deps[:, :, 2] = D[:, :, 2, 2] * dt
    deps[:, :, 3] = 2.0 * D[:, :, 0, 1] * dt
    deps[:, :, 4] = 2.0 * D[:, :, 1, 2] * dt
    deps[:, :, 5] = 2.0 * D[:, :, 0, 2] * dt

    alive = off > 0.0
    if not np.all(alive):
        deps[~alive] = 0.0
        trD[~alive] = 0.0

    # Jaumann stress rotation: sigma <- sigma + (W . sigma - sigma . W) * dt
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

    return dndx, vol_g, vol_tot, lc, deps, trD


def _post(xe: np.ndarray, dndx: np.ndarray, vol_g: np.ndarray, vol_tot: np.ndarray,
          lc: np.ndarray, rho: np.ndarray, trD: np.ndarray, deps: np.ndarray,
          sig: np.ndarray, sig_old: np.ndarray, qa: np.ndarray, qb: np.ndarray,
          c: np.ndarray, alive: np.ndarray, qvw_pend: np.ndarray, dt: float, dtfac: np.ndarray):
    """Bulk viscosity, internal force assembly, energies, and Courant time step."""
    n = len(xe)

    # Bulk viscosity per Gauss point in compression (sbulk3.F)
    compressing = (trD < 0.0) & alive[:, None]
    qvisc = np.where(
        compressing,
        rho[:, None] * lc[:, None] * (qa[:, None]**2 * lc[:, None] * trD**2 - qb[:, None] * c[:, None] * trD),
        0.0)

    sig_tot = sig.copy()
    sig_tot[:, :, 0] -= qvisc
    sig_tot[:, :, 1] -= qvisc
    sig_tot[:, :, 2] -= qvisc

    # Internal force assembly: f_i = - sum_g vol_g * sigma_g . gradN_{g,i}
    # S has shape (n, 8, 3, 3)
    S = np.empty((n, 8, 3, 3), dtype=np.float64)
    S[:, :, 0, 0], S[:, :, 1, 1], S[:, :, 2, 2] = sig_tot[:, :, 0], sig_tot[:, :, 1], sig_tot[:, :, 2]
    S[:, :, 0, 1] = S[:, :, 1, 0] = sig_tot[:, :, 3]
    S[:, :, 1, 2] = S[:, :, 2, 1] = sig_tot[:, :, 4]
    S[:, :, 0, 2] = S[:, :, 2, 0] = sig_tot[:, :, 5]

    # fe[n, i, b] = - sum_g vol_g[n, g] * sum_c S[n, g, b, c] * dndx[n, g, i, c]
    fe = -np.einsum("ng,ngbc,ngic->nib", vol_g, S, dndx)

    # Energy bookkeeping: internal energy from midpoint stress * strain increment
    sig_mid = 0.5 * (sig_old + sig)
    w_visc = np.sum(vol_g * 0.5 * qvisc * (-trD * dt), axis=1) + qvw_pend * np.mean(-trD, axis=1)
    deint0 = np.sum(vol_g * np.einsum("nga,nga->ng", sig_mid, deps), axis=1) + w_visc
    qvw_new = np.sum(vol_g * 0.5 * qvisc * dt, axis=1)

    # Critical Courant time step (sdlen3.F)
    trD_min = np.min(trD, axis=1)
    Q = np.where(trD_min < 0.0, qb * c + qa * lc * np.abs(trD_min), 0.0)
    denom = Q + np.sqrt(Q * Q + c * c)
    safe_denom = np.where(denom > 0.0, denom, 1.0)
    dt_crit = np.where(denom > 0.0, dtfac * lc / safe_denom, EP30)
    dt_crit = np.where(alive, dt_crit, EP30)

    return fe, dt_crit, w_visc, qvw_new, deint0


def forces(group, x, v, vr, dt, fint, mint):
    """Engine explicit cycle force kernel for fully-integrated 8-node hexas."""
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

    dndx, vol_g, vol_tot, lc, deps, trD = _pre(xe, ve, sig, dt, st["off"])
    vol_tot_safe = np.maximum(vol_tot, EM20)
    rho = st["mass"] / vol_tot_safe

    # Constitutive update across the 8 Gauss points
    c_sound = np.zeros(n)
    qa = np.zeros(n)
    qb = np.zeros(n)

    for sl, mat, prop in st.get("slices", []):
        qa[sl] = getattr(prop, "qa", 1.1) if prop else 1.1
        qb[sl] = getattr(prop, "qb", 0.05) if prop else 0.05

        law = getattr(mat, "law", 1)
        if law == 0:  # VOID
            sig[sl] = 0.0
            continue

        K_sl = getattr(mat, "K", 0.0)
        G_sl = getattr(mat, "G", 0.0)
        c_sound[sl] = np.sqrt(np.maximum(K_sl + 4.0 * G_sl / 3.0, 0.0) / np.maximum(rho[sl], EM20))

        # Update each of the 8 Gauss points
        for g in range(8):
            sig_g = sig[sl, g]
            deps_g = deps[sl, g]
            epsp_g = st["epsp"][sl, g]
            materials.solid_update(mat, sig_g, deps_g, epsp_g, dt, None)

    # Internal force assembly and time step calculation
    fe, dt_crit, w_visc, qvw_new, deint0 = _post(
        xe, dndx, vol_g, vol_tot, lc, rho, trD, deps, sig, sig_old,
        qa, qb, c_sound, alive, st["qvw_pend"], dt, st["dtfac"]
    )

    st["qvw_pend"] = qvw_new
    st["eint"] += deint0

    # Scatter internal forces into global fint
    if fint is not None:
        scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3))

    return dt_crit


# ----------------------------------------------------------------------------
# Implicit solver tangent, geometric stiffness, and consistent mass
# ----------------------------------------------------------------------------

def tangent(group, x_geom=None, epsp_incr=None, x=None):
    """Element tangent stiffness for fully-integrated 8-node hex with B-bar dilution (s8forc3.F).

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

    xe = x_geom[conn]  # (n, 8, 3)
    dndx, vol_g, vol_tot = _geometry(xe)
    vol_tot_safe = np.maximum(vol_tot, EM20)

    # Construct compatible B at 8 Gauss points: (n, 8, 6, 24)
    B = np.zeros((n, 8, 6, 24), dtype=np.float64)
    ix = np.arange(8)
    for g in range(8):
        gx = dndx[:, g, :, 0]  # (n, 8)
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

    # Volumetric B operator at each GP: B_vol = B_xx + B_yy + B_zz: (n, 8, 24)
    B_vol = B[:, :, 0, :] + B[:, :, 1, :] + B[:, :, 2, :]

    # Dilatational part B_dil at each GP: (1/3) * m * B_vol: (n, 8, 6, 24)
    B_dil = np.zeros((n, 8, 6, 24), dtype=np.float64)
    B_dil[:, :, 0:3, :] = (1.0 / 3.0) * B_vol[:, :, None, :]

    # Deviatoric part B_dev: (n, 8, 6, 24)
    B_dev = B - B_dil

    # B-bar mean dilatation: B_bar_vol = (1 / V_tot) * sum_g vol_g * B_vol: (n, 24)
    B_bar_vol = np.sum(vol_g[:, :, None] * B_vol, axis=1) / vol_tot_safe[:, None]

    # Diluted dilatational operator B_bar_dil: (n, 6, 24)
    B_bar_dil = np.zeros((n, 6, 24), dtype=np.float64)
    B_bar_dil[:, 0:3, :] = (1.0 / 3.0) * B_bar_vol[:, None, :]

    # Modified B-bar operator at each Gauss point: (n, 8, 6, 24)
    B_bar = B_dev + B_bar_dil[:, None, :, :]

    # Integrate K = sum_g vol_g * B_bar_g^T D_g B_bar_g
    ke = np.zeros((n, 24, 24), dtype=np.float64)
    epi = np.zeros((n, 8)) if epsp_incr is None else (
        epsp_incr if np.ndim(epsp_incr) == 2 else np.repeat(epsp_incr[:, None], 8, axis=1)
    )

    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            continue
        for g in range(8):
            D = materials.solid_tangent(mat, st["sig"][sl, g], st["epsp"][sl, g], epi[sl, g], None)
            Bg = B_bar[sl, g]
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
    """Geometric (initial-stress) element stiffness for fully-integrated 8-node hex (s8forc3.F).

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

    # Reconstruct 3x3 Cauchy stress tensor per Gauss point: (n, 8, 3, 3)
    sig = st["sig"]
    S = np.empty((n, 8, 3, 3), dtype=np.float64)
    S[:, :, 0, 0], S[:, :, 1, 1], S[:, :, 2, 2] = sig[:, :, 0], sig[:, :, 1], sig[:, :, 2]
    S[:, :, 0, 1] = S[:, :, 1, 0] = sig[:, :, 3]
    S[:, :, 1, 2] = S[:, :, 2, 1] = sig[:, :, 4]
    S[:, :, 0, 2] = S[:, :, 2, 0] = sig[:, :, 5]

    # g_ab = sum_g vol_g * (gradN_a^T S_g gradN_b): (n, 8, 8)
    g = np.zeros((n, 8, 8), dtype=np.float64)
    for g_idx in range(8):
        vg = vol_g[:, g_idx, None, None]
        dn = dndx[:, g_idx]  # (n, 8, 3)
        Sg = S[:, g_idx]     # (n, 3, 3)
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
    """Consistent element mass matrix for 8-node hex by 2x2x2 Gauss integration (smass3b.F):
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

    rho = st["mass"] / np.maximum(st["vol0"], EM20)  # (n,)

    if x_geom is not None:
        xe = x_geom[conn]
        _, vol_g, _ = _geometry(xe)
    elif "vol_g0" in st:
        vol_g = st["vol_g0"]
    else:
        vol_g = np.repeat((st["vol0"] / 8.0)[:, None], 8, axis=1)

    # N at 8 Gauss points: _N_GP is (8, 8)
    N_outer = np.einsum("gi,gj->gij", _N_GP, _N_GP)  # (8, 8, 8)
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

"""
20-node quadratic hexahedral solid element (/BRIC20, /HEXA20, /PROP/TYPE23).

Fortran origin: ``engine/source/elements/solid/solide20/`` — the 8-point
Gauss-integrated 20-node serendipity brick element (s20forc3.F, s20ke3.F,
s20kgeo3.F, s20deri3.F, s20coor3.F, s20fint3.F, s20defo3.F, s20cumu3.F,
s20bilan.F, and starter s20mass3.F).
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import det_inv33, scatter_add3

# 20 node coordinates in reference space [-1, 1]^3:
# Corners 0..7 (Fortran 1..8)
_XI_CORNERS = np.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
], dtype=np.float64)

# Midsides 8..19 (Fortran 9..20):
# 8: 0-1, 9: 1-2, 10: 2-3, 11: 3-0  (bottom face edges)
# 12: 0-4, 13: 1-5, 14: 2-6, 15: 3-7 (vertical edges)
# 16: 4-5, 17: 5-6, 18: 6-7, 19: 7-4 (top face edges)
_XI_MIDSIDES = np.array([
    [ 0, -1, -1], [ 1,  0, -1], [ 0,  1, -1], [-1,  0, -1],  # bottom (0-1, 1-2, 2-3, 3-0)
    [-1, -1,  0], [ 1, -1,  0], [ 1,  1,  0], [-1,  1,  0],  # vertical (0-4, 1-5, 2-6, 3-7)
    [ 0, -1,  1], [ 1,  0,  1], [ 0,  1,  1], [-1,  0,  1],  # top (4-5, 5-6, 6-7, 7-4)
], dtype=np.float64)

_XI_NODES = np.vstack([_XI_CORNERS, _XI_MIDSIDES])  # (20, 3)

_BRIC20_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (0, 4), (1, 5), (2, 6), (3, 7),
    (4, 5), (5, 6), (6, 7), (7, 4),
]

# 2x2x2 Gauss integration points and weights for 8-point rule
_G = 1.0 / np.sqrt(3.0)
_GAUSS_PTS = np.array([
    [-_G, -_G, -_G], [ _G, -_G, -_G], [ _G,  _G, -_G], [-_G,  _G, -_G],
    [-_G, -_G,  _G], [ _G, -_G,  _G], [ _G,  _G,  _G], [-_G,  _G,  _G],
], dtype=np.float64)
_GAUSS_WTS = np.ones(8, dtype=np.float64)


def _eval_dndxi(xi_pts: np.ndarray) -> np.ndarray:
    """Evaluate dN_i / d(xi, eta, zeta) at given reference points (nip, 20, 3)."""
    nip = len(xi_pts)
    dndxi = np.zeros((nip, 20, 3), dtype=np.float64)

    xi = xi_pts[:, 0]
    eta = xi_pts[:, 1]
    zeta = xi_pts[:, 2]

    # Corner nodes 0..7
    for i in range(8):
        xi_i, eta_i, zeta_i = _XI_CORNERS[i]
        t_xi = 1.0 + xi * xi_i
        t_eta = 1.0 + eta * eta_i
        t_zeta = 1.0 + zeta * zeta_i
        sum_t = xi * xi_i + eta * eta_i + zeta * zeta_i - 2.0

        # dN/dxi
        dndxi[:, i, 0] = 0.125 * xi_i * t_eta * t_zeta * sum_t + 0.125 * t_xi * t_eta * t_zeta * xi_i
        # dN/deta
        dndxi[:, i, 1] = 0.125 * t_xi * eta_i * t_zeta * sum_t + 0.125 * t_xi * t_eta * t_zeta * eta_i
        # dN/dzeta
        dndxi[:, i, 2] = 0.125 * t_xi * t_eta * zeta_i * sum_t + 0.125 * t_xi * t_eta * t_zeta * zeta_i

    # Midside nodes 8..19
    for m in range(12):
        node_idx = 8 + m
        xi_i, eta_i, zeta_i = _XI_MIDSIDES[m]
        if xi_i == 0.0:
            # xi_i == 0: N = 0.25 * (1 - xi^2) * (1 + eta*eta_i) * (1 + zeta*zeta_i)
            t_eta = 1.0 + eta * eta_i
            t_zeta = 1.0 + zeta * zeta_i
            dndxi[:, node_idx, 0] = -0.5 * xi * t_eta * t_zeta
            dndxi[:, node_idx, 1] = 0.25 * (1.0 - xi ** 2) * eta_i * t_zeta
            dndxi[:, node_idx, 2] = 0.25 * (1.0 - xi ** 2) * t_eta * zeta_i
        elif eta_i == 0.0:
            # eta_i == 0: N = 0.25 * (1 + xi*xi_i) * (1 - eta^2) * (1 + zeta*zeta_i)
            t_xi = 1.0 + xi * xi_i
            t_zeta = 1.0 + zeta * zeta_i
            dndxi[:, node_idx, 0] = 0.25 * xi_i * (1.0 - eta ** 2) * t_zeta
            dndxi[:, node_idx, 1] = -0.5 * eta * t_xi * t_zeta
            dndxi[:, node_idx, 2] = 0.25 * t_xi * (1.0 - eta ** 2) * zeta_i
        else:
            # zeta_i == 0: N = 0.25 * (1 + xi*xi_i) * (1 + eta*eta_i) * (1 - zeta^2)
            t_xi = 1.0 + xi * xi_i
            t_eta = 1.0 + eta * eta_i
            dndxi[:, node_idx, 0] = 0.25 * xi_i * t_eta * (1.0 - zeta ** 2)
            dndxi[:, node_idx, 1] = 0.25 * t_xi * eta_i * (1.0 - zeta ** 2)
            dndxi[:, node_idx, 2] = -0.5 * zeta * t_xi * t_eta

    return dndxi


_DN_DXI_GAUSS = _eval_dndxi(_GAUSS_PTS)  # (8, 20, 3)

# Exact 27-point tensor product Gauss quadrature for consistent mass matrix
_PTS1D_3 = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)], dtype=np.float64)
_WTS1D_3 = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)
_PTS27 = np.array([[x, y, z] for z in _PTS1D_3 for y in _PTS1D_3 for x in _PTS1D_3], dtype=np.float64)
_WTS27 = np.array([wx * wy * wz for wz in _WTS1D_3 for wy in _WTS1D_3 for wx in _WTS1D_3], dtype=np.float64)


def _eval_n(xi_pts: np.ndarray) -> np.ndarray:
    """Evaluate N_i at given reference points (nip, 20)."""
    nip = len(xi_pts)
    N = np.zeros((nip, 20), dtype=np.float64)
    xi, eta, zeta = xi_pts[:, 0], xi_pts[:, 1], xi_pts[:, 2]
    for i in range(8):
        xi_i, eta_i, zeta_i = _XI_CORNERS[i]
        t_xi = 1.0 + xi * xi_i
        t_eta = 1.0 + eta * eta_i
        t_zeta = 1.0 + zeta * zeta_i
        sum_t = xi * xi_i + eta * eta_i + zeta * zeta_i - 2.0
        N[:, i] = 0.125 * t_xi * t_eta * t_zeta * sum_t
    for m in range(12):
        node_idx = 8 + m
        xi_i, eta_i, zeta_i = _XI_MIDSIDES[m]
        if xi_i == 0.0:
            N[:, node_idx] = 0.25 * (1.0 - xi ** 2) * (1.0 + eta * eta_i) * (1.0 + zeta * zeta_i)
        elif eta_i == 0.0:
            N[:, node_idx] = 0.25 * (1.0 + xi * xi_i) * (1.0 - eta ** 2) * (1.0 + zeta * zeta_i)
        else:
            N[:, node_idx] = 0.25 * (1.0 + xi * xi_i) * (1.0 + eta * eta_i) * (1.0 - zeta ** 2)
    return N


#: Normalized consistent element mass matrix integral (sum = 1.0)
_M_BRIC20 = np.einsum("k,ka,kb->ab", _WTS27, _eval_n(_PTS27), _eval_n(_PTS27)) / 8.0


def _geometry(xe: np.ndarray):
    """
    Compute Jacobian and spatial shape function derivatives for BRIC20 elements.
    xe: (n, 20, 3)
    Returns:
        dndx: (n, 8, 20, 3)
        vol_gp: (n, 8)
        vol_tot: (n,)
    """
    n = len(xe)
    if n == 0:
        return np.zeros((0, 8, 20, 3)), np.zeros((0, 8)), np.zeros(0)

    # J = dndxi^T @ xe -> J[n, k, a, b] = sum_i dndxi[k, i, a] * xe[n, i, b]
    J = np.einsum("kia,nib->nkab", _DN_DXI_GAUSS, xe)
    J_flat = J.reshape(-1, 3, 3)
    a, b, c = J_flat[:, 0, 0], J_flat[:, 0, 1], J_flat[:, 0, 2]
    d, e, f = J_flat[:, 1, 0], J_flat[:, 1, 1], J_flat[:, 1, 2]
    g, h, i = J_flat[:, 2, 0], J_flat[:, 2, 1], J_flat[:, 2, 2]
    A = e * i - f * h
    B = f * g - d * i
    C = d * h - e * g
    detJ_flat = a * A + b * B + c * C
    detJ = detJ_flat.reshape(n, 8)
    deg = np.abs(detJ_flat) < 1e-12
    safe_det = np.where(deg, 1e-12, detJ_flat)
    idet = 1.0 / safe_det
    Jinv = np.empty_like(J_flat)
    Jinv[:, 0, 0] = A * idet
    Jinv[:, 0, 1] = (c * h - b * i) * idet
    Jinv[:, 0, 2] = (b * f - c * e) * idet
    Jinv[:, 1, 0] = B * idet
    Jinv[:, 1, 1] = (a * i - c * g) * idet
    Jinv[:, 1, 2] = (c * d - a * f) * idet
    Jinv[:, 2, 0] = C * idet
    Jinv[:, 2, 1] = (b * g - a * h) * idet
    Jinv[:, 2, 2] = (a * e - b * d) * idet
    Jinv[deg] = 0.0
    Jinv = Jinv.reshape(n, 8, 3, 3)

    vol_gp = detJ * _GAUSS_WTS[None, :]
    vol_tot = np.sum(vol_gp, axis=1)

    # dndx[n, k, i, b] = sum_a dndxi[k, i, a] * Jinv[n, k, b, a]
    dndx = np.einsum("kia,nkba->nkib", _DN_DXI_GAUSS, Jinv)
    bad_elem = np.abs(vol_tot) < 1e-12
    if np.any(bad_elem):
        dndx[bad_elem] = 0.0
    return dndx, vol_gp, vol_tot


def _char_length(vol_tot: np.ndarray) -> np.ndarray:
    """Characteristic length scale for stable time step: cubic root of volume."""
    n = len(vol_tot)
    if n == 0:
        return np.zeros(0)
    return np.maximum(vol_tot, EM20) ** (1.0 / 3.0)


def init_group(group, model, log):
    """Starter initialization for 20-node quadratic hexahedral solids."""
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
            off=np.ones(0),
            qvw_pend=np.zeros(0),
            dtfac=np.ones(0),
            chk_fail=False,
            dama=np.zeros(0),
        )
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float), None

    # Reconstruct coordinates for virtual midside nodes (-1 or 0)
    xe = np.zeros((n, 20, 3), dtype=np.float64)
    xe[:, :8] = model.x0[conn[:, :8]]
    for m, (n1, n2) in enumerate(_BRIC20_EDGES):
        c_m = conn[:, 8 + m]
        real = c_m >= 0
        if real.any():
            xe[real, 8 + m] = model.x0[c_m[real]]
        virt = ~real
        if virt.any():
            xe[virt, 8 + m] = 0.5 * (xe[virt, n1] + xe[virt, n2])

    dndx, vol_gp, vol_tot = _geometry(xe)

    flip = vol_tot <= 0.0
    if np.any(flip):
        log.warning(f"{flip.sum()} BRIC20 elements have non-positive volume.", "BRIC20 INIT")

    slices = group.state.get("slices", [])
    rho0 = np.zeros(n)
    for sl, mat, prop in slices:
        rho0[sl] = getattr(mat, "rho0", 0.0)
    mass = rho0 * np.maximum(vol_tot, 0.0)
    lc0 = _char_length(vol_tot)

    group.state.update(
        sig=np.zeros((n, 8, 6)),
        epsp=np.zeros((n, 8)),
        vol0=vol_tot.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),
        off=np.ones(n),
        qvw_pend=np.zeros(n),
        dtfac=0.5 * np.ones(n),
    )

    from .solid_hexa8 import _init_material_state
    _init_material_state(group, dndx[:, 0])

    chk_fail = any(
        getattr(mat, "fail", None) is not None
        or (getattr(mat, "law", 1) == 2 and mat.params.get("eps_max", EP30) < 1e30)
        or (getattr(mat, "law", 1) == 2 and mat.params.get("eps_p_max", EP30) < 1e30)
        or (getattr(mat, "law", 1) == 36 and mat.params.get("eps_p_max", EP30) < 1e30)
        for _, mat, _ in slices
    )
    group.state["chk_fail"] = chk_fail

    # Fortran s20mass3.F lumped mass distribution:
    # Corner nodes receive AM = mass / 64
    # Midside nodes receive BM = 7 * mass / 96
    # If a midside node is virtual (c_m < 0), its BM is split 50%/50% to its two corner nodes
    mass_corner = mass / 64.0
    mass_mid = mass * (7.0 / 96.0)

    node_idx_list = []
    mass_c_list = []

    for i in range(n):
        node_masses = {}
        for c in range(8):
            nid = conn[i, c]
            node_masses[nid] = node_masses.get(nid, 0.0) + mass_corner[i]
        for m, (n1, n2) in enumerate(_BRIC20_EDGES):
            nid_m = conn[i, 8 + m]
            if nid_m >= 0:
                node_masses[nid_m] = node_masses.get(nid_m, 0.0) + mass_mid[i]
            else:
                nid1 = conn[i, n1]
                nid2 = conn[i, n2]
                node_masses[nid1] = node_masses.get(nid1, 0.0) + 0.5 * mass_mid[i]
                node_masses[nid2] = node_masses.get(nid2, 0.0) + 0.5 * mass_mid[i]
        for nid, m_val in node_masses.items():
            if nid >= 0:
                node_idx_list.append(nid)
                mass_c_list.append(m_val)

    node_idx = np.array(node_idx_list, dtype=np.int64)
    mass_c = np.array(mass_c_list, dtype=np.float64)
    return node_idx, mass_c, None


def forces(group, x, v, vr, dt, fint, mint):
    """Engine explicit cycle forces for BRIC20 elements."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)
    if dt < 0.0:
        return np.full(n, EP30)

    xe = np.zeros((n, 20, 3), dtype=np.float64)
    xe[:, :8] = x[conn[:, :8]]
    for m, (n1, n2) in enumerate(_BRIC20_EDGES):
        c_m = conn[:, 8 + m]
        real = c_m >= 0
        if real.any():
            xe[real, 8 + m] = x[c_m[real]]
        virt = ~real
        if virt.any():
            xe[virt, 8 + m] = 0.5 * (xe[virt, n1] + xe[virt, n2])

    dndx, vol_gp, vol_tot = _geometry(xe)
    alive = st["off"] > 0.0

    if v is None:
        ve = np.zeros((n, 20, 3), dtype=np.float64)
    else:
        ve = np.zeros((n, 20, 3), dtype=np.float64)
        ve[:, :8] = v[conn[:, :8]]
        for m, (n1, n2) in enumerate(_BRIC20_EDGES):
            c_m = conn[:, 8 + m]
            real = c_m >= 0
            if real.any():
                ve[real, 8 + m] = v[c_m[real]]
            virt = ~real
            if virt.any():
                ve[virt, 8 + m] = 0.5 * (ve[virt, n1] + ve[virt, n2])

    sig = st["sig"]
    epsp = st["epsp"]
    epsp_old = epsp.copy() if st.get("chk_fail") else None

    # Velocity gradient L_ab = sum_i ve[:, i, a] * dndx[:, k, i, b]
    L = np.einsum("nia,nkib->nkab", ve, dndx)
    D = 0.5 * (L + np.swapaxes(L, 2, 3))
    trD = D[:, :, 0, 0] + D[:, :, 1, 1] + D[:, :, 2, 2]

    deps = np.empty((n, 8, 6))
    deps[:, :, 0] = D[:, :, 0, 0] * dt
    deps[:, :, 1] = D[:, :, 1, 1] * dt
    deps[:, :, 2] = D[:, :, 2, 2] * dt
    deps[:, :, 3] = 2.0 * D[:, :, 0, 1] * dt
    deps[:, :, 4] = 2.0 * D[:, :, 1, 2] * dt
    deps[:, :, 5] = 2.0 * D[:, :, 0, 2] * dt

    if not alive.all():
        deps[~alive] = 0.0
        trD = np.where(alive[:, None], trD, 0.0)

    # Jaumann rotation of old stress (srota3.F)
    wxy = 0.5 * (L[:, :, 0, 1] - L[:, :, 1, 0]) * dt
    wyz = 0.5 * (L[:, :, 1, 2] - L[:, :, 2, 1]) * dt
    wxz = 0.5 * (L[:, :, 0, 2] - L[:, :, 2, 0]) * dt

    sxx, syy, szz = sig[:, :, 0].copy(), sig[:, :, 1].copy(), sig[:, :, 2].copy()
    sxy, syz, szx = sig[:, :, 3].copy(), sig[:, :, 4].copy(), sig[:, :, 5].copy()
    sig[:, :, 0] += 2.0 * (wxy * sxy + wxz * szx)
    sig[:, :, 1] += 2.0 * (-wxy * sxy + wyz * syz)
    sig[:, :, 2] += 2.0 * (-wxz * szx - wyz * syz)
    sig[:, :, 3] += wxy * (syy - sxx) + wxz * syz + wyz * szx
    sig[:, :, 4] += wyz * (szz - syy) - wxy * szx - wxz * sxy
    sig[:, :, 5] += wxz * (szz - sxx) + wxy * syz - wyz * sxy

    sig_old = sig.copy()
    c_sound = np.zeros(n)
    rho = np.zeros(n)
    qa = np.zeros(n)
    qb = np.zeros(n)

    for sl, mat, prop in st.get("slices", []):
        mask_sl = alive[sl]
        if not np.any(mask_sl):
            continue

        law = getattr(mat, "law", 1)
        if law == 0:
            sig[sl] = 0.0
            continue

        rho0 = getattr(mat, "rho0", 0.0)
        E = getattr(mat, "E", 0.0)
        rho[sl] = rho0
        c_sound[sl] = np.sqrt(max(E, 0.0) / max(rho0, EM20))
        qa[sl] = getattr(prop, "qa", 1.1)
        qb[sl] = getattr(prop, "qb", 0.05)

        extra = st.get("mat_extra", {})
        for k in range(8):
            sig_k = sig[sl, k]
            deps_k = deps[sl, k]
            epsp_k = epsp[sl, k]
            if law == 1:
                materials.law01_elastic.solid_update(mat, sig_k, deps_k)
            elif law == 2:
                materials.law02_johnson_cook.solid_update(mat, sig_k, deps_k, epsp_k, dt, extra)
            elif law == 36:
                materials.law36_tabulated.solid_update(mat, sig_k, deps_k, epsp_k, dt, extra)

    if st.get("chk_fail"):
        for sl, mat, prop in st.get("slices", []):
            if getattr(mat, "fail", None) is not None:
                fail_res = failure.evaluate_solid(mat.fail, sig[sl], epsp[sl], epsp_old[sl], dt)
                if fail_res is not None:
                    st["off"][sl] *= fail_res

    lc = _char_length(vol_tot)
    compressing = (trD < 0.0) & alive[:, None]
    qvisc = np.where(
        compressing,
        rho[:, None] * lc[:, None] * (qa[:, None]**2 * lc[:, None] * trD**2 - qb[:, None] * c_sound[:, None] * trD),
        0.0)

    sig_tot = sig.copy()
    sig_tot[:, :, 0] -= qvisc
    sig_tot[:, :, 1] -= qvisc
    sig_tot[:, :, 2] -= qvisc

    S = np.empty((n, 8, 3, 3))
    S[:, :, 0, 0], S[:, :, 1, 1], S[:, :, 2, 2] = sig_tot[:, :, 0], sig_tot[:, :, 1], sig_tot[:, :, 2]
    S[:, :, 0, 1] = S[:, :, 1, 0] = sig_tot[:, :, 3]
    S[:, :, 1, 2] = S[:, :, 2, 1] = sig_tot[:, :, 4]
    S[:, :, 0, 2] = S[:, :, 2, 0] = sig_tot[:, :, 5]

    # Internal force: fe[i, a] = -sum_k vol_gp_k S_k dndx_k (OpenRadioss s20fint3.F)
    fe = -np.einsum("nk,nkbc,nkic->nib", vol_gp, S, dndx)
    fe *= st["off"][:, None, None]

    # Energies: trapezoidal viscous booking and stress work
    sig_mid = 0.5 * (sig_old + sig)
    w_visc = np.sum(vol_gp * 0.5 * qvisc * (-trD * dt), axis=1) + st["qvw_pend"] * np.sum(-trD, axis=1) / 8.0
    deint0 = np.sum(vol_gp * np.einsum("nka,nka->nk", sig_mid, deps), axis=1) + w_visc
    st["eint"] += deint0 * alive
    st["qvw_pend"] = np.sum(vol_gp * 0.5 * qvisc * dt, axis=1)

    # Virtual midside node force redistribution (s20cumu3.F / s20fint3.F)
    for m, (n1, n2) in enumerate(_BRIC20_EDGES):
        c_m = conn[:, 8 + m]
        virt = c_m < 0
        if virt.any():
            fe[virt, n1] += 0.5 * fe[virt, 8 + m]
            fe[virt, n2] += 0.5 * fe[virt, 8 + m]
            fe[virt, 8 + m] = 0.0

    if fint is not None:
        conn_flat = conn.reshape(-1)
        fe_flat = fe.reshape(-1, 3)
        valid = conn_flat >= 0
        scatter_add3(fint, conn_flat[valid], fe_flat[valid], st.get("color_indices"), st.get("color_offsets"))

    # Critical time step
    trD_min = trD.min(axis=1)
    Q = np.where(trD_min < 0.0, qb * c_sound + qa * lc * np.abs(trD_min), 0.0)
    denom = Q + np.sqrt(Q * Q + c_sound * c_sound)
    safe_denom = np.where(denom > 0.0, denom, 1.0)
    dt_crit = np.where(denom > 0.0, st["dtfac"] * lc / safe_denom, EP30)
    dt_crit = np.where(alive & (c_sound > 0.0), dt_crit, EP30)
    return dt_crit


def _edofs(conn):
    """(n, 60) global scalar DOF slot ids, node-major [ux, uy, uz] * 20."""
    n = len(conn)
    if n == 0:
        return np.zeros((0, 60), dtype=np.int64)
    ix = np.arange(20)
    edofs = np.empty((n, 60), dtype=np.int64)
    safe_conn = np.maximum(conn, 0)
    for c in range(3):
        edofs[:, 3 * ix + c] = safe_conn * 6 + c
    return edofs


def tangent(group, x, epsp_incr=None):
    """Element tangent stiffness for the whole bric20 group.
    Returns (ke, edofs): ke (n, 60, 60), edofs (n, 60)."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 60, 60), dtype=float), np.zeros((0, 60), dtype=np.int64)

    xe = np.zeros((n, 20, 3), dtype=np.float64)
    xe[:, :8] = x[conn[:, :8]]
    for m, (n1, n2) in enumerate(_BRIC20_EDGES):
        c_m = conn[:, 8 + m]
        real = c_m >= 0
        if real.any():
            xe[real, 8 + m] = x[c_m[real]]
        virt = ~real
        if virt.any():
            xe[virt, 8 + m] = 0.5 * (xe[virt, n1] + xe[virt, n2])

    dndx, vol_gp, vol_tot = _geometry(xe)
    vol_gp = np.maximum(vol_gp, EM20)

    B = np.zeros((n, 8, 6, 60))
    for ix in range(20):
        gx = dndx[:, :, ix, 0]
        gy = dndx[:, :, ix, 1]
        gz = dndx[:, :, ix, 2]
        B[:, :, 0, 3 * ix + 0] = gx
        B[:, :, 1, 3 * ix + 1] = gy
        B[:, :, 2, 3 * ix + 2] = gz
        B[:, :, 3, 3 * ix + 0] = gy
        B[:, :, 3, 3 * ix + 1] = gx
        B[:, :, 4, 3 * ix + 1] = gz
        B[:, :, 4, 3 * ix + 2] = gy
        B[:, :, 5, 3 * ix + 0] = gz
        B[:, :, 5, 3 * ix + 2] = gx

    from .. import materials as _materials
    ke = np.zeros((n, 60, 60))
    epi = np.zeros((n, 8)) if epsp_incr is None else epsp_incr
    if epi.ndim == 1:
        epi = np.tile(epi[:, None], (1, 8))

    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            continue
        sig_sl = st["sig"][sl]    # (m, 8, 6)
        epsp_sl = st["epsp"][sl]  # (m, 8)
        epi_sl = epi[sl]          # (m, 8)
        for k in range(8):
            D = _materials.solid_tangent(
                mat, sig_sl[:, k], epsp_sl[:, k], epi_sl[:, k], None)  # (m, 6, 6)
            Bk = B[sl, k]  # (m, 6, 60)
            DB = np.einsum("mij,mjk->mik", D, Bk)
            ke[sl] += vol_gp[sl, k, None, None] * np.einsum("mji,mjk->mik", Bk, DB)

    return ke, _edofs(conn)


def kgeo(group, x):
    """Geometric (initial-stress) element stiffness for the bric20 group at
    geometry x: sum_k V_k gradN_a . sigma_k . gradN_b replicated over translations."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 60, 60), dtype=float), np.zeros((0, 60), dtype=np.int64)

    xe = np.zeros((n, 20, 3), dtype=np.float64)
    xe[:, :8] = x[conn[:, :8]]
    for m, (n1, n2) in enumerate(_BRIC20_EDGES):
        c_m = conn[:, 8 + m]
        real = c_m >= 0
        if real.any():
            xe[real, 8 + m] = x[c_m[real]]
        virt = ~real
        if virt.any():
            xe[virt, 8 + m] = 0.5 * (xe[virt, n1] + xe[virt, n2])

    dndx, vol_gp, vol_tot = _geometry(xe)
    vol_gp = np.maximum(vol_gp, EM20)
    sig = st["sig"]
    S = np.empty((n, 8, 3, 3))
    S[:, :, 0, 0], S[:, :, 1, 1], S[:, :, 2, 2] = sig[:, :, 0], sig[:, :, 1], sig[:, :, 2]
    S[:, :, 0, 1] = S[:, :, 1, 0] = sig[:, :, 3]
    S[:, :, 1, 2] = S[:, :, 2, 1] = sig[:, :, 4]
    S[:, :, 0, 2] = S[:, :, 2, 0] = sig[:, :, 5]

    ke = np.zeros((n, 60, 60))
    ix = np.arange(20)
    for k in range(8):
        gk = vol_gp[:, k, None, None] * np.einsum(
            "nac,ncd,nbd->nab", dndx[:, k], S[:, k], dndx[:, k])  # (n, 20, 20)
        for b in range(3):
            rows = (3 * ix + b)[:, None]
            cols = (3 * ix + b)[None, :]
            ke[:, rows, cols] += gk

    return ke, _edofs(conn)


def consistent_mass(group, x=None):
    """Consistent element mass integral rho N^T N dV of the 20-node quadratic brick:
    exact analytical reference moment matrix M_bric20 ⊗ I3, built on the stored
    element mass."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 60, 60), dtype=float), np.zeros((0, 60), dtype=np.int64)
    m = st["mass"]  # rho * V0, per element
    me = np.zeros((n, 60, 60))
    for a in range(20):
        for b in range(20):
            f = m * _M_BRIC20[a, b]
            for c in range(3):
                me[:, a * 3 + c, b * 3 + c] = f
    return me, _edofs(conn)


def static_internal_forces(group, x, u, ur, fint, mint):
    """Internal nodal force at configuration x from the CURRENT stress state:
    fe = -sum_k vol_k S_k gradN_k."""
    if group.n == 0 or len(group.conn) == 0 or fint is None:
        return
    st = group.state
    conn = group.conn
    n = group.n

    xe = np.zeros((n, 20, 3), dtype=np.float64)
    xe[:, :8] = x[conn[:, :8]]
    for m, (n1, n2) in enumerate(_BRIC20_EDGES):
        c_m = conn[:, 8 + m]
        real = c_m >= 0
        if real.any():
            xe[real, 8 + m] = x[c_m[real]]
        virt = ~real
        if virt.any():
            xe[virt, 8 + m] = 0.5 * (xe[virt, n1] + xe[virt, n2])

    dndx, vol_gp, vol_tot = _geometry(xe)
    vol_gp = np.maximum(vol_gp, EM20)
    sig = st["sig"]
    S = np.empty((n, 8, 3, 3))
    S[:, :, 0, 0], S[:, :, 1, 1], S[:, :, 2, 2] = sig[:, :, 0], sig[:, :, 1], sig[:, :, 2]
    S[:, :, 0, 1] = S[:, :, 1, 0] = sig[:, :, 3]
    S[:, :, 1, 2] = S[:, :, 2, 1] = sig[:, :, 4]
    S[:, :, 0, 2] = S[:, :, 2, 0] = sig[:, :, 5]

    fe = -np.einsum("nk,nkbc,nkic->nib", vol_gp, S, dndx)
    fe *= st["off"][:, None, None]

    # Virtual midside node force redistribution
    for m, (n1, n2) in enumerate(_BRIC20_EDGES):
        c_m = conn[:, 8 + m]
        virt = c_m < 0
        if virt.any():
            fe[virt, n1] += 0.5 * fe[virt, 8 + m]
            fe[virt, n2] += 0.5 * fe[virt, 8 + m]
            fe[virt, 8 + m] = 0.0

    conn_flat = conn.reshape(-1)
    fe_flat = fe.reshape(-1, 3)
    valid = conn_flat >= 0
    scatter_add3(fint, conn_flat[valid], fe_flat[valid], st.get("color_indices"), st.get("color_offsets"))


def implicit_internal_forces(group, x, u, ur, fint, mint, nlgeom=False):
    """Implicit solver internal force vector dispatcher.
    nlgeom=False: linear Ku from initial configuration.
    nlgeom=True: nonlinear updated Lagrangian static assembly."""
    if group.n == 0 or len(group.conn) == 0 or fint is None:
        return
    if not nlgeom:
        ke, edofs = tangent(group, x)
        ue = np.zeros((group.n, 60))
        for ix in range(20):
            nid = group.conn[:, ix]
            valid = nid >= 0
            for c in range(3):
                ue[valid, 3 * ix + c] = u[nid[valid], c]
        fe = -np.einsum("nij,nj->ni", ke, ue)
        fe = fe.reshape(-1, 20, 3)
        for m, (n1, n2) in enumerate(_BRIC20_EDGES):
            c_m = group.conn[:, 8 + m]
            virt = c_m < 0
            if virt.any():
                fe[virt, n1] += 0.5 * fe[virt, 8 + m]
                fe[virt, n2] += 0.5 * fe[virt, 8 + m]
                fe[virt, 8 + m] = 0.0
        conn_flat = group.conn.reshape(-1)
        fe_flat = fe.reshape(-1, 3)
        valid = conn_flat >= 0
        scatter_add3(fint, conn_flat[valid], fe_flat[valid],
                     group.state.get("color_indices"), group.state.get("color_offsets"))
    else:
        static_internal_forces(group, x, u, ur, fint, mint)

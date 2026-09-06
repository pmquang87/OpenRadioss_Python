"""
20-node quadratic hexahedral solid element (/BRIC20, /HEXA20, /PROP/TYPE23).

Serendipity 20-node brick with 8 Gauss points (2x2x2).
8 corner nodes + 12 midside nodes.
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import det_inv33, scatter_add3

# 20 node coordinates in reference space [-1, 1]^3:
# Corners 0..7
_XI_CORNERS = np.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
], dtype=np.float64)

# Midsides 8..19
# 8: 0-1, 9: 1-2, 10: 2-3, 11: 3-0
# 12: 4-5, 13: 5-6, 14: 6-7, 15: 7-4
# 16: 0-4, 17: 1-5, 18: 2-6, 19: 3-7
_XI_MIDSIDES = np.array([
    [ 0, -1, -1], [ 1,  0, -1], [ 0,  1, -1], [-1,  0, -1],
    [ 0, -1,  1], [ 1,  0,  1], [ 0,  1,  1], [-1,  0,  1],
    [-1, -1,  0], [ 1, -1,  0], [ 1,  1,  0], [-1,  1,  0],
], dtype=np.float64)

_XI_NODES = np.vstack([_XI_CORNERS, _XI_MIDSIDES])  # (20, 3)

# 2x2x2 Gauss integration points and weights
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
    # J = dndxi^T @ xe -> J[n, k, a, b] = sum_i dndxi[k, i, a] * xe[n, i, b]
    J = np.einsum("kia,nib->nkab", _DN_DXI_GAUSS, xe)
    detJ, Jinv = det_inv33(J.reshape(-1, 3, 3))
    detJ = detJ.reshape(n, 8)
    Jinv = Jinv.reshape(n, 8, 3, 3)

    vol_gp = detJ * _GAUSS_WTS[None, :]
    vol_tot = np.sum(vol_gp, axis=1)

    # dndx[n, k, i, b] = sum_a dndxi[k, i, a] * Jinv[n, k, b, a]
    dndx = np.einsum("kia,nkba->nkib", _DN_DXI_GAUSS, Jinv)
    return dndx, vol_gp, vol_tot


def _char_length(vol_tot: np.ndarray) -> np.ndarray:
    """Characteristic length scale for stable time step: cubic root of volume."""
    return np.maximum(vol_tot, EM20) ** (1.0 / 3.0)


def init_group(group, model, log):
    """Starter initialization for 20-node quadratic hexahedral solids."""
    conn = group.conn
    xe = model.x0[conn]
    dndx, vol_gp, vol_tot = _geometry(xe)

    flip = vol_tot <= 0.0
    if np.any(flip):
        log.warning(f"{flip.sum()} BRIC20 elements have non-positive volume.", "BRIC20 INIT")

    n = group.n
    rho0 = np.zeros(n)
    for sl, mat, prop in group.state["slices"]:
        rho0[sl] = mat.rho0
    mass = rho0 * np.maximum(vol_tot, EM20)
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
        mat.fail is not None
        or (mat.law == 2 and mat.params.get("eps_max", EP30) < 1e30)
        or (mat.law == 2 and mat.params.get("eps_p_max", EP30) < 1e30)
        or (mat.law == 36 and mat.params.get("eps_p_max", EP30) < 1e30)
        for _, mat, _ in group.state["slices"]
    )
    group.state["chk_fail"] = chk_fail

    node_idx = conn.reshape(-1)
    mass_c = np.repeat(mass / 20.0, 20)
    return node_idx, mass_c, None


def forces(group, x, v, vr, dt, fint, mint):
    """Engine explicit cycle forces for BRIC20 elements."""
    st = group.state
    conn = group.conn
    n = group.n
    xe = x[conn]
    ve = v[conn]

    dndx, vol_gp, vol_tot = _geometry(xe)
    alive = st["off"] > 0.0

    sig = st["sig"]
    epsp = st["epsp"]
    epsp_old = epsp.copy() if st.get("chk_fail") else None

    # Strain increment deps (n, 8, 6) = [exx, eyy, ezz, gxy, gyz, gzx]
    # L_ab = sum_i ve[:, i, a] * dndx[:, k, i, b]
    L = np.einsum("nia,nkib->nkab", ve, dndx)
    deps = np.zeros((n, 8, 6))
    deps[:, :, 0] = L[:, :, 0, 0] * dt
    deps[:, :, 1] = L[:, :, 1, 1] * dt
    deps[:, :, 2] = L[:, :, 2, 2] * dt
    deps[:, :, 3] = (L[:, :, 0, 1] + L[:, :, 1, 0]) * dt
    deps[:, :, 4] = (L[:, :, 1, 2] + L[:, :, 2, 1]) * dt
    deps[:, :, 5] = (L[:, :, 2, 0] + L[:, :, 0, 2]) * dt

    c_sound = np.zeros(n)

    for sl, mat, prop in st["slices"]:
        mask_sl = alive[sl]
        if not np.any(mask_sl):
            continue

        c_sound[sl] = np.sqrt(mat.E / max(mat.rho0, EM20))
        extra = st.get("mat_extra", {})

        for k in range(8):
            sig_k = sig[sl, k]
            deps_k = deps[sl, k]
            epsp_k = epsp[sl, k]

            if mat.law == 1:
                materials.law01_elastic.solid_update(mat, sig_k, deps_k)
            elif mat.law == 2:
                materials.law02_johnson_cook.solid_update(mat, sig_k, deps_k, epsp_k, dt, extra)
            elif mat.law == 36:
                materials.law36_tabulated.solid_update(mat, sig_k, deps_k, epsp_k, dt, extra)

            # Internal energy increment: dW = sum(sig : deps) * vol_gp
            dW = np.sum(sig_k * deps_k, axis=1) * vol_gp[sl, k]
            st["eint"][sl] += dW * mask_sl

    if st.get("chk_fail"):
        for sl, mat, prop in st["slices"]:
            if mat.fail is not None:
                # evaluate failure on integration points
                fail_res = failure.evaluate_solid(mat.fail, sig[sl], epsp[sl], epsp_old[sl], dt)
                if fail_res is not None:
                    st["off"][sl] *= fail_res

    # Internal force: f_ia = sum_k sum_b dndx_kib * sig_ab * vol_gp_k
    # Voigt matrix: [[s0, s3, s5], [s3, s1, s4], [s5, s4, s2]]
    # f_internal = (n, 20, 3)
    fe = np.zeros((n, 20, 3))
    for k in range(8):
        s = sig[:, k]
        w = vol_gp[:, k, None]
        # s_tensor (n, 3, 3)
        s_ten = np.zeros((n, 3, 3))
        s_ten[:, 0, 0] = s[:, 0]
        s_ten[:, 1, 1] = s[:, 1]
        s_ten[:, 2, 2] = s[:, 2]
        s_ten[:, 0, 1] = s_ten[:, 1, 0] = s[:, 3]
        s_ten[:, 1, 2] = s_ten[:, 2, 1] = s[:, 4]
        s_ten[:, 2, 0] = s_ten[:, 0, 2] = s[:, 5]

        # f_ia += sum_b dndx[:, k, i, b] * s_ten[:, a, b] * w
        fe += np.einsum("nib,nab,n->nia", dndx[:, k], s_ten, vol_gp[:, k])

    fe *= st["off"][:, None, None]

    # Scatter to global fint
    scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3), st.get("color_indices"), st.get("color_offsets"))

    lc = _char_length(vol_tot)
    dt_crit = np.where(c_sound > 0, st["dtfac"] * lc / np.maximum(c_sound, EM20), EP30)
    return dt_crit


def _edofs(conn):
    n = len(conn)
    ix = np.arange(20)
    edofs = np.empty((n, 60), dtype=np.int64)
    for c in range(3):
        edofs[:, 3 * ix + c] = conn * 6 + c
    return edofs


def tangent(group, x, epsp_incr=None):
    n = group.n
    return np.zeros((n, 60, 60)), _edofs(group.conn)


def kgeo(group, x):
    n = group.n
    return np.zeros((n, 60, 60)), _edofs(group.conn)


def consistent_mass(group, x=None):
    n = group.n
    return np.zeros((n, 60, 60)), _edofs(group.conn)


def static_internal_forces(group, x, u, ur, fint, mint):
    pass

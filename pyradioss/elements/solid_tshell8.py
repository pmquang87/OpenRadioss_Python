"""
8-node Thick Shell Element (TSHELL / /PROP/TYPE20, TYPE21, TYPE22).
Fortran origin: engine/source/elements/thickshell/solidec/
  scforc3.F   driver: gather coords/velocities, call the chain below
  sccoor3.F   local corotational coordinate system
  scderi3.F   derivatives, B0 matrix, central Jacobian
  scdefc3.F   centroidal strain rates, hourglass rates, HQEPH gradients
  scdefo3.F   strain rates at through-thickness integration points
  scfint3.F   internal nodal force integration through thickness
  schour3_1.F in-plane HQEPH physical hourglass stabilization
  sdlensh.F   characteristic length & Courant time step
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import cross3, det_inv33, norm3, scatter_add3

# Node sign pattern for standard 8-node hexahedron
# Nodes 0..3: bottom face (-zeta), counter-clockwise seen from +zeta
# Nodes 4..7: top face (+zeta), counter-clockwise seen from +zeta
_XI = np.array([
    [-1.0, -1.0, -1.0],
    [ 1.0, -1.0, -1.0],
    [ 1.0,  1.0, -1.0],
    [-1.0,  1.0, -1.0],
    [-1.0, -1.0,  1.0],
    [ 1.0, -1.0,  1.0],
    [ 1.0,  1.0,  1.0],
    [-1.0,  1.0,  1.0],
], dtype=np.float64)

_DN_DXI = _XI / 8.0  # (8, 3)

# 4 Standard hourglass base vectors (Flanagan-Belytschko), normalized by 1/8 (ONE_OVER_8 in scdefc3.F)
_H = np.array([
    [ 1.0,  1.0, -1.0, -1.0, -1.0, -1.0,  1.0,  1.0],
    [ 1.0, -1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0],
    [ 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
    [-1.0,  1.0, -1.0,  1.0,  1.0, -1.0,  1.0, -1.0],
], dtype=np.float64) / 8.0


def get_integration_points(nip: int, iint: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Return Gauss or Lobatto integration points and weights in zeta in [-1, 1].
    iint = 0: Gauss-Legendre (default)
    iint = 1: Gauss-Lobatto (includes surface points zeta = +-1)
    """
    nip = max(int(nip), 1)
    if nip == 1:
        return np.array([0.0]), np.array([2.0])
    if iint == 1:  # Lobatto
        if nip == 2:
            pts = np.array([-1.0, 1.0])
            wts = np.array([1.0, 1.0])
        elif nip == 3:
            pts = np.array([-1.0, 0.0, 1.0])
            wts = np.array([1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0])
        elif nip == 4:
            pts = np.array([-1.0, -1.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0), 1.0])
            wts = np.array([1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0])
        elif nip == 5:
            r = np.sqrt(3.0 / 7.0)
            pts = np.array([-1.0, -r, 0.0, r, 1.0])
            wts = np.array([0.1, 49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0, 0.1])
        else:
            pts, wts = np.polynomial.legendre.leggauss(nip)
            wts = wts * (2.0 / np.sum(wts))
    else:  # Gauss-Legendre
        pts, wts = np.polynomial.legendre.leggauss(nip)
    return pts, wts


def _compute_local_frame(xe: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute local corotational coordinate system, mid-surface coordinates, thickness, and area.
    xe: (nel, 8, 3)
    Returns:
      R: (nel, 3, 3) rotation matrix where columns are [v1, v2, v3]
      xc: (nel, 3) element centroid
      thick: (nel,) through-thickness dimension h
      area: (nel,) in-plane mid-surface area
      mid_nodes: (nel, 4, 3) mid-surface node positions
    """
    bot = 0.25 * np.sum(xe[:, :4, :], axis=1)  # (nel, 3)
    top = 0.25 * np.sum(xe[:, 4:, :], axis=1)  # (nel, 3)
    xc = 0.5 * (bot + top)

    t_vec = top - bot  # (nel, 3)
    thick = np.linalg.norm(t_vec, axis=1)
    thick_safe = np.maximum(thick, EM20)
    v3 = t_vec / thick_safe[:, None]

    mid_nodes = 0.5 * (xe[:, :4, :] + xe[:, 4:, :])  # (nel, 4, 3)

    r1 = mid_nodes[:, 1] + mid_nodes[:, 2] - mid_nodes[:, 0] - mid_nodes[:, 3]  # (nel, 3)
    proj = np.sum(r1 * v3, axis=1, keepdims=True)
    v1 = r1 - proj * v3
    norm_v1 = np.linalg.norm(v1, axis=1)
    bad1 = norm_v1 <= EM20
    if np.any(bad1):
        v1[bad1] = np.array([1.0, 0.0, 0.0])
        norm_v1[bad1] = 1.0
    v1 = v1 / norm_v1[:, None]

    v2 = np.cross(v3, v1)
    norm_v2 = np.linalg.norm(v2, axis=1, keepdims=True)
    v2 = v2 / np.maximum(norm_v2, EM20)

    R = np.stack([v1, v2, v3], axis=2)  # (nel, 3, 3)

    diag1 = mid_nodes[:, 2] - mid_nodes[:, 0]
    diag2 = mid_nodes[:, 3] - mid_nodes[:, 1]
    cross_d = np.cross(diag1, diag2)
    area = 0.5 * np.linalg.norm(cross_d, axis=1)

    return R, xc, thick, area, mid_nodes


def _geometry(xe: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Centroidal derivatives and volume.
    xe: (nel, 8, 3)
    Returns:
      dndx: (nel, 8, 3) Cartesian shape function derivatives at center in local frame
      vol: (nel,) element volume = 8 * det(J0)
      R: (nel, 3, 3) local frame
      thick: (nel,) thickness
      area: (nel,) mid-surface area
    """
    R, xc, thick, area, _ = _compute_local_frame(xe)
    x_loc = np.einsum("nia,nab->nib", xe - xc[:, None, :], R)

    J0 = np.einsum("ia,nib->nab", _DN_DXI, x_loc)
    detJ = np.linalg.det(J0)
    vol = 8.0 * detJ

    invJ0 = np.linalg.inv(J0)
    dndx_loc = np.einsum("ia,nab->nib", _DN_DXI, invJ0)  # (nel, 8, 3)

    return dndx_loc, vol, R, thick, area


def _safe_sound_speed(mat) -> float:
    """Solid P-wave sound speed c = sqrt((K + 4/3 G) / rho0)."""
    c_cand = 0.0
    try:
        c_val = materials.sound_speed(mat, is_shell=False)
        if c_val is not None:
            c_cand = float(np.asarray(c_val).flat[0])
    except Exception:
        pass
    if c_cand <= 0.0 and hasattr(mat, "sound_speed_solid") and getattr(mat, "rho0", 0.0) > 0.0:
        try:
            c_cand = float(mat.sound_speed_solid())
        except Exception:
            pass
    if c_cand <= 0.0:
        K = getattr(mat, "K", 0.0)
        G = getattr(mat, "G", 0.0)
        rho0 = getattr(mat, "rho0", 0.0)
        if rho0 > 0.0 and (K > 0.0 or G > 0.0):
            c_cand = float(np.sqrt(max(K + 4.0 * G / 3.0, 0.0) / max(rho0, 1e-20)))
    return max(c_cand, 1e-20)


def init_group(group, model, log) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Initialize TSHELL element buffer, layers, and lumped mass matrix.
    Fortran origin: starter/source/elements/solid/ (sinit3.F, smass3.F) and
    starter/source/properties/thickshell/hm_read_prop20.F.
    """
    n = group.n
    conn = group.conn

    if n == 0 or len(conn) == 0:
        group.state.update(
            sig=np.zeros((0, 1, 6)),
            epsp=np.zeros((0, 1)),
            vol0=np.zeros(0),
            vol=np.zeros(0),
            mass=np.zeros(0),
            eint=np.zeros(0),
            ehour=np.zeros(0),
            off=np.ones(0),
            lc=np.zeros(0),
            dtfac=np.ones(0),
            zw=[],
            chk_fail=False,
            dama=np.zeros((0, 1)),
            hgq=np.zeros((0, 9)),
        )
        return np.zeros(0, dtype=np.int64), np.zeros(0), None

    xe = model.x0[conn]  # (n, 8, 3)
    dndx0, vol0, R0, thick0, area0 = _geometry(xe)

    bad = vol0 <= EM20
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/TSHELL {eid}: zero or negative initial volume (check node ordering)", "TSHELL INIT")

    rho0 = np.zeros(n)
    zw_list = []
    nip_max = 1

    for sl, mat, prop in group.state["slices"]:
        rho0[sl] = getattr(mat, "rho0", 0.0)
        nip_val = int(prop.params.get("inpts", 0) or prop.params.get("nip", 0) or prop.params.get("npts_s", 0) or prop.params.get("npts_t", 0) or 3)
        if nip_val <= 0:
            nip_val = 3
        iint_val = int(prop.params.get("iint", 0) or 0)
        pts, wts = get_integration_points(nip_val, iint_val)
        zw_list.append((pts, wts))
        if nip_val > nip_max:
            nip_max = nip_val

    mass = rho0 * vol0

    l_inplane = np.sqrt(np.maximum(area0, EM20))
    lc0 = np.minimum(l_inplane, np.maximum(thick0, EM20))

    sig = np.zeros((n, nip_max, 6), dtype=np.float64)
    epsp = np.zeros((n, nip_max), dtype=np.float64)
    dama = np.zeros((n, nip_max), dtype=np.float64)
    # 9 modal hourglass coordinates per element
    hgq = np.zeros((n, 9), dtype=np.float64)

    group.state.update(
        sig=sig,
        epsp=epsp,
        vol0=vol0.copy(),
        vol=vol0.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),
        off=np.ones(n),
        lc=lc0,
        dtfac=np.ones(n),
        zw=zw_list,
        nip_max=nip_max,
        chk_fail=any(mat.fail is not None for _, mat, _ in group.state["slices"]),
        dama=dama,
        hgq=hgq,
    )

    node_idx = conn.reshape(-1)
    mass_c = np.repeat(mass / 8.0, 8)
    return node_idx, mass_c, None


def forces(group, x: np.ndarray, v: np.ndarray, vr: np.ndarray, dt: float,
           fint: np.ndarray, mint: np.ndarray) -> np.ndarray:
    """Compute 8-node Thick Shell internal and physical stabilization forces.
    Scatters negated internal forces into fint, updates stress/energy state,
    and returns per-element critical time steps.
    """
    n = group.n
    if n == 0:
        return np.zeros(0)

    conn = group.conn
    st = group.state
    slices = st["slices"]
    zw_list = st["zw"]
    off = st["off"]

    xe = x[conn]  # (n, 8, 3)

    R, xc, thick, area, _ = _compute_local_frame(xe)

    l_inplane = np.sqrt(np.maximum(area, EM20))
    l_thick = np.maximum(thick, EM20)
    lc = np.minimum(l_inplane, l_thick)
    st["lc"] = lc

    dt_crit = np.full(n, 1e20, dtype=np.float64)
    for sl, mat, prop in slices:
        c_sound = _safe_sound_speed(mat)
        if c_sound > 0.0:
            dt_crit[sl] = lc[sl] / c_sound

    if dt is None or dt <= 0.0 or v is None:
        return dt_crit

    ve = v[conn]  # (n, 8, 3)

    x_loc = np.einsum("nia,nab->nib", xe - xc[:, None, :], R)
    v_loc = np.einsum("nia,nab->nib", ve, R)

    # Centroidal Jacobian & derivatives for spin and orthogonalization
    J0 = np.einsum("ia,nib->nab", _DN_DXI, x_loc)
    detJ0 = np.linalg.det(J0)
    vol = 8.0 * detJ0
    st["vol"] = vol

    invJ0 = np.linalg.inv(J0)
    dndx0_loc = np.einsum("ia,nab->nib", _DN_DXI, invJ0)
    bx0 = dndx0_loc[:, :, 0]
    by0 = dndx0_loc[:, :, 1]
    bz0 = dndx0_loc[:, :, 2]

    vx = v_loc[:, :, 0]
    vy = v_loc[:, :, 1]
    vz = v_loc[:, :, 2]

    Lxy = np.sum(by0 * vx, axis=1)
    Lyx = np.sum(bx0 * vy, axis=1)
    Lyz = np.sum(bz0 * vy, axis=1)
    Lzy = np.sum(by0 * vz, axis=1)
    Lzx = np.sum(bx0 * vz, axis=1)
    Lxz = np.sum(bz0 * vx, axis=1)

    Wxy = 0.5 * (Lxy - Lyx)
    Wyz = 0.5 * (Lyz - Lzy)
    Wzx = 0.5 * (Lzx - Lxz)

    # ------------------------------------------------------------------------
    # Assumed Natural Strain (ANS) for Transverse Shear
    # Midpoint sampling at lateral edges to eliminate shear locking
    # ------------------------------------------------------------------------
    dx_A = x_loc[:, 1, 0] - x_loc[:, 0, 0]
    h_A = 0.5 * ((x_loc[:, 4, 2] - x_loc[:, 0, 2]) + (x_loc[:, 5, 2] - x_loc[:, 1, 2]))
    safe_dx_A = np.where(np.abs(dx_A) > EM20, dx_A, np.sign(dx_A) * EM20 + EM20)
    safe_h_A = np.where(np.abs(h_A) > EM20, h_A, np.sign(h_A) * EM20 + EM20)
    gam_xz_A = ((vz[:, 1] + vz[:, 5] - vz[:, 0] - vz[:, 4]) / (2.0 * safe_dx_A) +
                (vx[:, 4] + vx[:, 5] - vx[:, 0] - vx[:, 1]) / (2.0 * safe_h_A))

    dx_C = x_loc[:, 2, 0] - x_loc[:, 3, 0]
    h_C = 0.5 * ((x_loc[:, 7, 2] - x_loc[:, 3, 2]) + (x_loc[:, 6, 2] - x_loc[:, 2, 2]))
    safe_dx_C = np.where(np.abs(dx_C) > EM20, dx_C, np.sign(dx_C) * EM20 + EM20)
    safe_h_C = np.where(np.abs(h_C) > EM20, h_C, np.sign(h_C) * EM20 + EM20)
    gam_xz_C = ((vz[:, 2] + vz[:, 6] - vz[:, 3] - vz[:, 7]) / (2.0 * safe_dx_C) +
                (vx[:, 7] + vx[:, 6] - vx[:, 3] - vx[:, 2]) / (2.0 * safe_h_C))

    ans_gam_xz = 0.5 * (gam_xz_A + gam_xz_C)

    dy_B = x_loc[:, 2, 1] - x_loc[:, 1, 1]
    h_B = 0.5 * ((x_loc[:, 5, 2] - x_loc[:, 1, 2]) + (x_loc[:, 6, 2] - x_loc[:, 2, 2]))
    safe_dy_B = np.where(np.abs(dy_B) > EM20, dy_B, np.sign(dy_B) * EM20 + EM20)
    safe_h_B = np.where(np.abs(h_B) > EM20, h_B, np.sign(h_B) * EM20 + EM20)
    gam_yz_B = ((vz[:, 2] + vz[:, 6] - vz[:, 1] - vz[:, 5]) / (2.0 * safe_dy_B) +
                (vy[:, 5] + vy[:, 6] - vy[:, 1] - vy[:, 2]) / (2.0 * safe_h_B))

    dy_D = x_loc[:, 3, 1] - x_loc[:, 0, 1]
    h_D = 0.5 * ((x_loc[:, 4, 2] - x_loc[:, 0, 2]) + (x_loc[:, 7, 2] - x_loc[:, 3, 2]))
    safe_dy_D = np.where(np.abs(dy_D) > EM20, dy_D, np.sign(dy_D) * EM20 + EM20)
    safe_h_D = np.where(np.abs(h_D) > EM20, h_D, np.sign(h_D) * EM20 + EM20)
    gam_yz_D = ((vz[:, 3] + vz[:, 7] - vz[:, 0] - vz[:, 4]) / (2.0 * safe_dy_D) +
                (vy[:, 4] + vy[:, 7] - vy[:, 0] - vy[:, 3]) / (2.0 * safe_h_D))

    ans_gam_yz = 0.5 * (gam_yz_B + gam_yz_D)

    # ------------------------------------------------------------------------
    # ANS Transverse Shear Derivative Contribution Operators
    # ------------------------------------------------------------------------
    d_gam_xz_dvz = np.zeros((n, 8), dtype=np.float64)
    d_gam_xz_dvx = np.zeros((n, 8), dtype=np.float64)

    inv2_dxA = 0.5 / safe_dx_A
    inv2_hA = 0.5 / safe_h_A
    inv2_dxC = 0.5 / safe_dx_C
    inv2_hC = 0.5 / safe_h_C

    d_gam_xz_dvz[:, 0] -= 0.5 * inv2_dxA
    d_gam_xz_dvz[:, 1] += 0.5 * inv2_dxA
    d_gam_xz_dvz[:, 4] -= 0.5 * inv2_dxA
    d_gam_xz_dvz[:, 5] += 0.5 * inv2_dxA

    d_gam_xz_dvx[:, 0] -= 0.5 * inv2_hA
    d_gam_xz_dvx[:, 1] -= 0.5 * inv2_hA
    d_gam_xz_dvx[:, 4] += 0.5 * inv2_hA
    d_gam_xz_dvx[:, 5] += 0.5 * inv2_hA

    d_gam_xz_dvz[:, 3] -= 0.5 * inv2_dxC
    d_gam_xz_dvz[:, 2] += 0.5 * inv2_dxC
    d_gam_xz_dvz[:, 7] -= 0.5 * inv2_dxC
    d_gam_xz_dvz[:, 6] += 0.5 * inv2_dxC

    d_gam_xz_dvx[:, 3] -= 0.5 * inv2_hC
    d_gam_xz_dvx[:, 2] -= 0.5 * inv2_hC
    d_gam_xz_dvx[:, 7] += 0.5 * inv2_hC
    d_gam_xz_dvx[:, 6] += 0.5 * inv2_hC

    d_gam_yz_dvz = np.zeros((n, 8), dtype=np.float64)
    d_gam_yz_dvy = np.zeros((n, 8), dtype=np.float64)

    inv2_dyB = 0.5 / safe_dy_B
    inv2_hB = 0.5 / safe_h_B
    inv2_dyD = 0.5 / safe_dy_D
    inv2_hD = 0.5 / safe_h_D

    d_gam_yz_dvz[:, 1] -= 0.5 * inv2_dyB
    d_gam_yz_dvz[:, 2] += 0.5 * inv2_dyB
    d_gam_yz_dvz[:, 5] -= 0.5 * inv2_dyB
    d_gam_yz_dvz[:, 6] += 0.5 * inv2_dyB

    d_gam_yz_dvy[:, 1] -= 0.5 * inv2_hB
    d_gam_yz_dvy[:, 2] -= 0.5 * inv2_hB
    d_gam_yz_dvy[:, 5] += 0.5 * inv2_hB
    d_gam_yz_dvy[:, 6] += 0.5 * inv2_hB

    d_gam_yz_dvz[:, 0] -= 0.5 * inv2_dyD
    d_gam_yz_dvz[:, 3] += 0.5 * inv2_dyD
    d_gam_yz_dvz[:, 4] -= 0.5 * inv2_dyD
    d_gam_yz_dvz[:, 7] += 0.5 * inv2_dyD

    d_gam_yz_dvy[:, 0] -= 0.5 * inv2_hD
    d_gam_yz_dvy[:, 3] -= 0.5 * inv2_hD
    d_gam_yz_dvy[:, 4] += 0.5 * inv2_hD
    d_gam_yz_dvy[:, 7] += 0.5 * inv2_hD

    # ------------------------------------------------------------------------
    # Through-thickness integration loop with layer-wise B(zeta_k)
    # ------------------------------------------------------------------------
    f_int_loc = np.zeros((n, 8, 3), dtype=np.float64)
    sig = st["sig"]
    epsp = st["epsp"]
    eint_inc = np.zeros(n)

    xi_sign = _XI[:, 0]
    eta_sign = _XI[:, 1]
    zeta_sign = _XI[:, 2]

    for isl, (sl, mat, prop) in enumerate(slices):
        pts, wts = zw_list[isl]
        nip = len(pts)
        n_sl = int(np.sum(sl)) if isinstance(sl, np.ndarray) else (sl.stop - (sl.start or 0))
        if n_sl == 0:
            continue

        for k in range(nip):
            zeta_k = pts[k]
            wt_k = wts[k]

            # Shape function derivatives at (0, 0, zeta_k)
            dN_dxi = (xi_sign[None, :] * (1.0 + zeta_sign[None, :] * zeta_k)) / 8.0  # (1, 8)
            dN_deta = (eta_sign[None, :] * (1.0 + zeta_sign[None, :] * zeta_k)) / 8.0
            dN_dzeta = zeta_sign[None, :] / 8.0

            # Layer Jacobian matrix: (n_sl, 3, 3)
            # J_k[a, b] = sum_i dN_dxi_a[i] * x_loc[i, b]
            dN_nat = np.stack([
                np.repeat(dN_dxi, n_sl, axis=0),
                np.repeat(dN_deta, n_sl, axis=0),
                np.repeat(dN_dzeta, n_sl, axis=0),
            ], axis=1)  # (n_sl, 3, 8)

            J_k = np.einsum("nai,nib->nab", dN_nat, x_loc[sl])  # (n_sl, 3, 3)
            detJ_k = np.linalg.det(J_k)
            invJ_k = np.linalg.inv(J_k)

            # Layer volume weight
            dV_k = 4.0 * detJ_k * wt_k  # (n_sl,)

            # Layer Cartesian derivatives: dN_dx = invJ_k^T @ dN_nat => (n_sl, 8, 3)
            dndx_k = np.einsum("nia,nab->nib", dN_nat.transpose(0, 2, 1), invJ_k)
            bx_k = dndx_k[:, :, 0]
            by_k = dndx_k[:, :, 1]
            bz_k = dndx_k[:, :, 2]

            # Layer in-plane & normal strain rates
            eps_dot_xx = np.sum(bx_k * vx[sl], axis=1)
            eps_dot_yy = np.sum(by_k * vy[sl], axis=1)
            eps_dot_zz = np.sum(bz_k * vz[sl], axis=1)
            eps_dot_xy = 0.5 * (np.sum(by_k * vx[sl], axis=1) + np.sum(bx_k * vy[sl], axis=1))

            # Transverse shear strain rates from ANS
            eps_dot_yz = 0.5 * ans_gam_yz[sl]
            eps_dot_zx = 0.5 * ans_gam_xz[sl]

            # Jaumann stress rotation
            s_old = sig[sl, k].copy()
            w_xy = Wxy[sl]
            w_yz = Wyz[sl]
            w_zx = Wzx[sl]

            ds_rot = np.zeros_like(s_old)
            ds_rot[:, 0] = 2.0 * (s_old[:, 3] * w_xy - s_old[:, 5] * w_zx)
            ds_rot[:, 1] = 2.0 * (-s_old[:, 3] * w_xy + s_old[:, 4] * w_yz)
            ds_rot[:, 2] = 2.0 * (-s_old[:, 4] * w_yz + s_old[:, 5] * w_zx)
            ds_rot[:, 3] = (s_old[:, 1] - s_old[:, 0]) * w_xy + s_old[:, 4] * w_zx - s_old[:, 5] * w_yz
            ds_rot[:, 4] = (s_old[:, 2] - s_old[:, 1]) * w_yz + s_old[:, 5] * w_xy - s_old[:, 3] * w_zx
            ds_rot[:, 5] = (s_old[:, 0] - s_old[:, 2]) * w_zx + s_old[:, 3] * w_yz - s_old[:, 4] * w_xy

            s_rot = s_old + ds_rot * dt

            D_vec = np.stack([
                eps_dot_xx, eps_dot_yy, eps_dot_zz,
                eps_dot_xy, eps_dot_yz, eps_dot_zx
            ], axis=1)

            try:
                s_new, ep_new = materials.update_stress(
                    mat, s_rot, D_vec, dt, epsp[sl, k], is_shell=False
                )
            except Exception:
                lam = getattr(mat, "K", 0.0) - 2.0 * getattr(mat, "G", 0.0) / 3.0
                G = getattr(mat, "G", 0.0)
                if lam == 0.0 and G == 0.0:
                    E_val = getattr(mat, "E", 2.1e11)
                    nu_val = getattr(mat, "nu", 0.3)
                    G = E_val / (2.0 * (1.0 + nu_val))
                    lam = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
                trD = eps_dot_xx + eps_dot_yy + eps_dot_zz
                s_new = s_rot.copy()
                s_new[:, 0] += (lam * trD + 2.0 * G * eps_dot_xx) * dt
                s_new[:, 1] += (lam * trD + 2.0 * G * eps_dot_yy) * dt
                s_new[:, 2] += (lam * trD + 2.0 * G * eps_dot_zz) * dt
                s_new[:, 3] += (2.0 * G * eps_dot_xy) * dt
                s_new[:, 4] += (2.0 * G * eps_dot_yz) * dt
                s_new[:, 5] += (2.0 * G * eps_dot_zx) * dt
                ep_new = epsp[sl, k]

            sig[sl, k] = s_new
            epsp[sl, k] = ep_new

            # Internal energy rate
            s_mid = 0.5 * (s_rot + s_new)
            work_gp = (
                s_mid[:, 0] * eps_dot_xx +
                s_mid[:, 1] * eps_dot_yy +
                s_mid[:, 2] * eps_dot_zz +
                2.0 * s_mid[:, 3] * eps_dot_xy +
                2.0 * s_mid[:, 4] * eps_dot_yz +
                2.0 * s_mid[:, 5] * eps_dot_zx
            ) * dV_k * dt
            eint_inc[sl] += work_gp

            # Layer internal nodal forces
            sxx = s_new[:, 0, None]
            syy = s_new[:, 1, None]
            szz = s_new[:, 2, None]
            sxy = s_new[:, 3, None]
            syz = s_new[:, 4, None]
            szx = s_new[:, 5, None]

            dV_arr = dV_k[:, None]

            # In-plane & normal forces from layer-wise B(zeta_k)
            f_int_loc[sl, :, 0] += dV_arr * (sxx * bx_k + sxy * by_k)
            f_int_loc[sl, :, 1] += dV_arr * (sxy * bx_k + syy * by_k)
            f_int_loc[sl, :, 2] += dV_arr * (szz * bz_k)

            # Transverse shear forces from ANS
            f_int_loc[sl, :, 0] += dV_arr * (szx * d_gam_xz_dvx[sl])
            f_int_loc[sl, :, 1] += dV_arr * (syz * d_gam_yz_dvy[sl])
            f_int_loc[sl, :, 2] += dV_arr * (
                szx * d_gam_xz_dvz[sl] + syz * d_gam_yz_dvz[sl]
            )

    # ------------------------------------------------------------------------
    # Full HQEPH Hourglass Physical Stabilization (schour3_1.F)
    # ------------------------------------------------------------------------
    ehour_inc = np.zeros(n)
    hgq = st["hgq"]  # (n, 9)

    # Orthogonalized hourglass base vectors
    Gamma = np.zeros((4, n, 8), dtype=np.float64)
    hx = x_loc[:, :, 0]
    hy = x_loc[:, :, 1]
    hz = x_loc[:, :, 2]

    for a in range(4):
        h_vec = _H[a]
        hx_dot = np.sum(h_vec * hx, axis=1, keepdims=True)
        hy_dot = np.sum(h_vec * hy, axis=1, keepdims=True)
        hz_dot = np.sum(h_vec * hz, axis=1, keepdims=True)
        Gamma[a] = h_vec[None, :] - hx_dot * bx0 - hy_dot * by0 - hz_dot * bz0

    for sl, mat, prop in slices:
        E_mod = float(getattr(mat, "E", 2.1e11) or 2.1e11)
        nu_mod = float(getattr(mat, "nu", 0.3) or 0.3)
        G_mod = E_mod / (2.0 * (1.0 + nu_mod))
        qa_hg = float(prop.params.get("qa", 1.1) or 1.1)

        L_ch = np.sqrt(np.maximum(area[sl], EM20))
        Kh = qa_hg * 0.1 * G_mod * vol[sl] / (L_ch ** 2)

        # 9 stabilization modes (schour3_1.F lines 245-260):
        # Modes 0..2: H3 in (x, y, z)
        # Modes 3..5: H4 in (x, y, z)
        # Modes 6..7: H1 in z, H2 in z
        # Mode 8: Warp difference (H1 in x - H2 in y)
        g_list = [
            (Gamma[2, sl], 0),  # H3 x
            (Gamma[2, sl], 1),  # H3 y
            (Gamma[2, sl], 2),  # H3 z
            (Gamma[3, sl], 0),  # H4 x
            (Gamma[3, sl], 1),  # H4 y
            (Gamma[3, sl], 2),  # H4 z
            (Gamma[0, sl], 2),  # H1 z
            (Gamma[1, sl], 2),  # H2 z
        ]

        for m_idx, (gam_vec, dof) in enumerate(g_list):
            v_curr = v_loc[sl, :, dof]
            v_modal = np.sum(gam_vec * v_curr, axis=1)
            hgq[sl, m_idx] += v_modal * dt * off[sl]

            fh = Kh[:, None] * hgq[sl, m_idx, None] * gam_vec
            f_int_loc[sl, :, dof] += fh
            ehour_inc[sl] += np.sum(fh * v_curr * dt, axis=1)

        # Mode 8: Warp difference (H1_x - H2_y)
        v_diff = np.sum(Gamma[0, sl] * vx[sl] - Gamma[1, sl] * vy[sl], axis=1)
        hgq[sl, 8] += v_diff * dt * off[sl]
        fh_x = Kh[:, None] * hgq[sl, 8, None] * Gamma[0, sl]
        fh_y = -Kh[:, None] * hgq[sl, 8, None] * Gamma[1, sl]
        f_int_loc[sl, :, 0] += fh_x
        f_int_loc[sl, :, 1] += fh_y
        ehour_inc[sl] += np.sum((fh_x * vx[sl] + fh_y * vy[sl]) * dt, axis=1)

    st["eint"] += eint_inc * off
    st["ehour"] += ehour_inc * off

    # Transform back to global
    f_glob = np.einsum("nib,nab->nia", f_int_loc, R)
    f_glob = f_glob * off[:, None, None]

    flat_nodes = conn.reshape(-1)
    flat_forces = -f_glob.reshape(-1, 3)
    scatter_add3(fint, flat_nodes, flat_forces)

    return dt_crit

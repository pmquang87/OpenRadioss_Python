"""
6-node Wedge / Prism Solid Element (/PENTA6 + /PROP/SOLID, /PROP/TYPE14).

Fortran origin: ``engine/source/elements/solid/solide6z/`` — the dedicated
prismatic solid formulation:
    s6zforc3.F90  driver: gather coordinates/velocities, call the chain below
    s6zderi3.F90  geometry & shape function derivatives for 6-node wedge
    s6zdefo3.F90  velocity gradient -> rate of deformation D, spin W
    s6zrrota3.F90 local co-rotational coordinate frame & Jaumann rotation
    s6zfint3.F90  internal nodal forces: f_i = - sum_g (w_g * detJ_g * sigma_g . gradN_{g,i})
    s6zhourg3.F90 physical hourglass / assumed-strain stabilization
    checksvolume.F (CHECKVOLUME_6N) volume & orientation verification

Theory notes:
* Shape functions:
  In the natural triangular coordinates (xi, eta) with L1 = 1 - xi - eta, L2 = xi, L3 = eta,
  and thickness coordinate zeta in [-1, +1]:
      N_i(xi, eta, zeta) = L_i(xi, eta) * (1 - zeta) / 2  for bottom triangle i = 1, 2, 3
      N_i(xi, eta, zeta) = L_{i-3}(xi, eta) * (1 + zeta) / 2  for top triangle i = 4, 5, 6

* Integration:
  2-point Gauss-Legendre quadrature through the thickness (zeta = +- 1/sqrt(3))
  at the triangle area centroid (xi = 1/3, eta = 1/3).
  Quadrature weight per Gauss point in natural space is W_g = 0.5 (reference prism volume = 1.0).

* Volume & Mass:
  Exact prism volume V = A_tri * h_avg = sum_g (w_g * detJ_g).
  Lumped nodal mass distribution m_i = rho * V / 6 spread equally to all 6 nodes.

* Characteristic length:
  In-plane triangle altitude / equilateral side equivalent L_tri = sqrt(4 * A_tri / sqrt(3)),
  thickness h_avg = average of 3 vertical edge lengths.
  L_c = min(L_tri, h_avg).
  Courant critical time step: dt_crit = L_c / c (corrected by bulk viscosity and exact eigenfrequency factor).
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import cross3, det_inv33, norm3, scatter_add3

# ----------------------------------------------------------------------------
# Natural coordinates and shape function derivatives
# ----------------------------------------------------------------------------
# 2 Gauss points at triangle centroid (xi=1/3, eta=1/3), zeta = +- 1/sqrt(3)
_SQRT3_INV = 1.0 / np.sqrt(3.0)
_GP_ZETA = np.array([-_SQRT3_INV, _SQRT3_INV], dtype=np.float64)
_WIP = np.array([0.5, 0.5], dtype=np.float64)  # reference prism volume = 1.0

# Precomputed dN_i/d(xi, eta, zeta) at the 2 Gauss points: shape (2, 6, 3)
# GP 0: zeta = -1/sqrt(3)
#   (1 - zeta)/2 = (1 + 1/sqrt(3))/2
#   (1 + zeta)/2 = (1 - 1/sqrt(3))/2
# GP 1: zeta = +1/sqrt(3)
#   (1 - zeta)/2 = (1 - 1/sqrt(3))/2
#   (1 + zeta)/2 = (1 + 1/sqrt(3))/2
# At xi = 1/3, eta = 1/3:
#   dN_1..3/dzeta = -1/6
#   dN_4..6/dzeta = +1/6
def _build_dn_dxi():
    dn = np.zeros((2, 6, 3), dtype=np.float64)
    for g, z in enumerate(_GP_ZETA):
        cm = 0.5 * (1.0 - z)
        cp = 0.5 * (1.0 + z)
        # Node 1: (1 - xi - eta) * (1 - zeta)/2
        dn[g, 0, 0] = -cm
        dn[g, 0, 1] = -cm
        dn[g, 0, 2] = -1.0 / 6.0
        # Node 2: xi * (1 - zeta)/2
        dn[g, 1, 0] = cm
        dn[g, 1, 1] = 0.0
        dn[g, 1, 2] = -1.0 / 6.0
        # Node 3: eta * (1 - zeta)/2
        dn[g, 2, 0] = 0.0
        dn[g, 2, 1] = cm
        dn[g, 2, 2] = -1.0 / 6.0
        # Node 4: (1 - xi - eta) * (1 + zeta)/2
        dn[g, 3, 0] = -cp
        dn[g, 3, 1] = -cp
        dn[g, 3, 2] = 1.0 / 6.0
        # Node 5: xi * (1 + zeta)/2
        dn[g, 4, 0] = cp
        dn[g, 4, 1] = 0.0
        dn[g, 4, 2] = 1.0 / 6.0
        # Node 6: eta * (1 + zeta)/2
        dn[g, 5, 0] = 0.0
        dn[g, 5, 1] = cp
        dn[g, 5, 2] = 1.0 / 6.0
    return dn

_DN_DXI = _build_dn_dxi()

# Boundary faces: 2 triangular (bottom [0,1,2,2], top [3,4,5,5]) and 3 quad faces
_FACES = np.array([
    [0, 1, 2, 2],
    [3, 4, 5, 5],
    [0, 1, 4, 3],
    [1, 2, 5, 4],
    [2, 0, 3, 5],
], dtype=np.int64)


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def _geometry(xe: np.ndarray):
    """Jacobian, inverse, Cartesian gradients, and volumes for PENTA6 elements.

    xe : (n, 6, 3) nodal coordinates.
    Returns:
        dndx: (n, 2, 6, 3) Cartesian shape gradients at the 2 Gauss points.
        vol_g: (n, 2) volume contribution of each Gauss point.
        vol_tot: (n,) total element volume.
    """
    n = len(xe)
    if n == 0:
        return np.zeros((0, 2, 6, 3)), np.zeros((0, 2)), np.zeros(0)

    # J[n, g, a, b] = sum_i dN_i/dxi_a * x_{i, b}
    J = np.einsum("gia,nib->ngab", _DN_DXI, xe)  # (n, 2, 3, 3)
    J_flat = J.reshape(-1, 3, 3)

    # Determinant via cofactors
    a, b, c = J_flat[:, 0, 0], J_flat[:, 0, 1], J_flat[:, 0, 2]
    d, e, f = J_flat[:, 1, 0], J_flat[:, 1, 1], J_flat[:, 1, 2]
    g_x, h, i_coord = J_flat[:, 2, 0], J_flat[:, 2, 1], J_flat[:, 2, 2]

    A = e * i_coord - f * h
    B = f * g_x - d * i_coord
    C = d * h - e * g_x
    detJ_flat = a * A + b * B + c * C
    detJ = detJ_flat.reshape(n, 2)

    deg = np.abs(detJ_flat) < 1e-12
    safe_det = np.where(deg, np.where(detJ_flat >= 0.0, 1e-12, -1e-12), detJ_flat)
    idet = 1.0 / safe_det

    Jinv = np.empty_like(J_flat)
    Jinv[:, 0, 0] = A * idet
    Jinv[:, 0, 1] = (c * h - b * i_coord) * idet
    Jinv[:, 0, 2] = (b * f - c * e) * idet
    Jinv[:, 1, 0] = B * idet
    Jinv[:, 1, 1] = (a * i_coord - c * g_x) * idet
    Jinv[:, 1, 2] = (c * d - a * f) * idet
    Jinv[:, 2, 0] = C * idet
    Jinv[:, 2, 1] = (b * g_x - a * h) * idet
    Jinv[:, 2, 2] = (a * e - b * d) * idet
    Jinv[deg] = 0.0
    Jinv = Jinv.reshape(n, 2, 3, 3)

    # Gauss point volume = detJ * weight (w_g = 0.5)
    vol_g = detJ * _WIP[None, :]
    vol_tot = np.sum(vol_g, axis=1)

    # dN_i/dx_b = dN_i/dxi_a * Jinv[a, b]  (inv(J)[a, b] = dxi_a/dx_b)
    # einsum: gia, ngba -> ngib
    dndx = np.einsum("gia,ngba->ngib", _DN_DXI, Jinv)
    bad_elem = np.abs(vol_tot) < 1e-12
    if np.any(bad_elem):
        dndx[bad_elem] = 0.0

    return dndx, vol_g, vol_tot


def _char_length(xe: np.ndarray, vol_tot: np.ndarray) -> np.ndarray:
    """Courant characteristic length L_c = min(L_tri, h_avg).
    L_tri = sqrt(4 * A_tri / sqrt(3)), h_avg = average prism height.
    """
    n = len(xe)
    if n == 0:
        return np.zeros(0)

    # Bottom triangle area (nodes 0, 1, 2)
    e12_b = xe[:, 1] - xe[:, 0]
    e13_b = xe[:, 2] - xe[:, 0]
    a_bot = 0.5 * norm3(cross3(e12_b, e13_b))

    # Top triangle area (nodes 3, 4, 5)
    e12_t = xe[:, 4] - xe[:, 3]
    e13_t = xe[:, 5] - xe[:, 3]
    a_top = 0.5 * norm3(cross3(e12_t, e13_t))

    a_tri = np.maximum(0.5 * (a_bot + a_top), EM20)
    l_tri = np.sqrt(4.0 * a_tri / np.sqrt(3.0))

    # Average thickness / height (edges 0->3, 1->4, 2->5)
    h0 = norm3(xe[:, 3] - xe[:, 0])
    h1 = norm3(xe[:, 4] - xe[:, 1])
    h2 = norm3(xe[:, 5] - xe[:, 2])
    h_avg = np.maximum((h0 + h1 + h2) / 3.0, EM20)

    # Also bound by volume / max area to ensure strict safety on distorted wedges
    h_alt = 3.0 * np.maximum(vol_tot, EM20) / np.maximum(np.maximum(a_bot, a_top), EM20)
    h_eff = np.minimum(h_avg, h_alt)

    return np.minimum(l_tri, h_eff)


def _exact_dt_factor(dndx: np.ndarray, vol_g: np.ndarray, lc: np.ndarray,
                     slices, rho0: np.ndarray) -> np.ndarray:
    """Exact Courant time-step scaling factor from maximum element eigenfrequency."""
    n = len(lc)
    if n == 0:
        return np.zeros(0)
    fac = np.ones(n)

    for sl, mat, _ in slices:
        rho0_val = getattr(mat, "rho0", 0.0)
        E_val = getattr(mat, "E", 0.0)
        if rho0_val <= 0.0 or E_val <= 0.0:
            continue
        fac[sl] = 0.5
    return fac


def _edofs(conn: np.ndarray) -> np.ndarray:
    """(n, 18) global scalar translation DOF indices in node-major order."""
    n = len(conn)
    if n == 0:
        return np.zeros((0, 18), dtype=np.int64)
    ix = np.arange(6)
    edofs = np.full((n, 18), -1, dtype=np.int64)
    valid = conn >= 0
    for c in range(3):
        edofs[:, 3 * ix + c] = np.where(valid, conn * 6 + c, -1)
    return edofs


# ----------------------------------------------------------------------------
# Starter-side initialization
# ----------------------------------------------------------------------------

def init_group(group, model, log):
    """Element state initialization and lumped mass distribution for /PENTA6.

    Volume V = A_tri * h_avg = sum_g (w_g * detJ_g).
    Lumped nodal mass: m_i = rho0 * V / 6.
    """
    n = group.n
    conn = group.conn
    if n == 0 or len(conn) == 0:
        group.state.update(
            sig=np.zeros((0, 2, 6)),
            epsp=np.zeros((0, 2)),
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

    xe = model.x0[conn]  # (n, 6, 3)
    dndx0, vol_g0, vol_tot0 = _geometry(xe)

    # Canonicalize winding if negative volume (swap nodes 1<->2 and 4<->5 to reverse orientation)
    flip = vol_tot0 < 0.0
    if np.any(flip):
        group.conn[flip] = group.conn[flip][:, [0, 2, 1, 3, 5, 4]]
        xe = model.x0[group.conn]
        dndx0, vol_g0, vol_tot0 = _geometry(xe)

    # Degenerate element detection
    bad = np.abs(vol_tot0) <= EM20
    if np.any(bad):
        log.warning(f"{int(bad.sum())} /PENTA6 element(s) have zero or negative volume.",
                    "PENTA6 INIT")

    slices = group.state.get("slices", [])
    rho0 = np.zeros(n)
    for sl, mat, prop in slices:
        rho0[sl] = getattr(mat, "rho0", 0.0)

    mass = rho0 * np.maximum(vol_tot0, 0.0)
    lc0 = _char_length(xe, vol_tot0)

    chk_fail = any(
        getattr(mat, "fail", None) is not None or getattr(mat, "params", {}).get("eps_p_max", EP30) < 1e30
        for _, mat, _ in slices
    )

    dtfac = _exact_dt_factor(dndx0, vol_g0, lc0, slices, rho0)

    group.state.update(
        sig=np.zeros((n, 2, 6)),
        epsp=np.zeros((n, 2)),
        vol0=vol_tot0.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),
        off=np.ones(n),
        qvw_pend=np.zeros(n),
        dtfac=dtfac,
        chk_fail=chk_fail,
        dama=np.zeros(n),
    )

    # Lumped mass: m_i = mass / 6.0 equally to all 6 nodes
    elem_mass = np.repeat((mass / 6.0)[:, None], 6, axis=1)  # (n, 6)

    node_idx = conn.reshape(-1)
    mass_c = elem_mass.reshape(-1)
    valid = node_idx >= 0

    from .solid_hexa8 import _init_material_state
    _init_material_state(group, dndx0[:, 0])

    return node_idx[valid], mass_c[valid], None


# ----------------------------------------------------------------------------
# Engine-side force cycle (pre + post)
# ----------------------------------------------------------------------------

def _pre(xe, ve, sig, dt, off):
    """Geometry + kinematics + Jaumann rotation for PENTA6 elements."""
    n = len(xe)

    # Geometry at current configuration
    dndx, vol_g, vol_tot = _geometry(xe)
    vol_tot = np.maximum(vol_tot, EM20)
    lc = _char_length(xe, vol_tot)

    # Velocity gradient L[n, g, b, c] = sum_i ve[n, i, b] * dndx[n, g, i, c]
    L = np.einsum("nib,ngic->ngbc", ve, dndx)
    D = 0.5 * (L + np.transpose(L, (0, 1, 3, 2)))
    trD = D[:, :, 0, 0] + D[:, :, 1, 1] + D[:, :, 2, 2]

    # Flush round-off traces
    vgm = np.abs(ve).max(axis=(1, 2))[:, None] * np.abs(dndx).max(axis=(2, 3))
    trD = np.where(np.abs(trD) <= 1e-14 * vgm, 0.0, trD)

    # Strain increment in Voigt form [xx, yy, zz, xy, yz, zx]
    deps = np.empty((n, 2, 6))
    deps[:, :, 0] = D[:, :, 0, 0] * dt
    deps[:, :, 1] = D[:, :, 1, 1] * dt
    deps[:, :, 2] = D[:, :, 2, 2] * dt
    deps[:, :, 3] = 2.0 * D[:, :, 0, 1] * dt
    deps[:, :, 4] = 2.0 * D[:, :, 1, 2] * dt
    deps[:, :, 5] = 2.0 * D[:, :, 0, 2] * dt

    alive = off > 0.0
    if not alive.all():
        deps[~alive] = 0.0
        trD = np.where(alive[:, None], trD, 0.0)

    # Objective co-rotational Jaumann rotation of the old stress by spin increment W * dt
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

    return dndx, vol_g, vol_tot, lc, deps, trD


def _post(xe, dndx, vol_g, vol_tot, lc, rho, trD, deps, sig, sig_old,
          qa, qb, c, alive, qvw_pend, dt, dtfac):
    """Bulk viscosity, internal force assembly, energies, and Courant time step."""
    n = len(xe)

    # Bulk viscosity (shock damping in compression)
    compressing = (trD < 0.0) & alive[:, None]
    qvisc = np.where(
        compressing,
        rho[:, None] * lc[:, None] * (qa[:, None]**2 * lc[:, None] * trD**2 - qb[:, None] * c[:, None] * trD),
        0.0)

    # Total Cauchy stress tensor with viscous pressure
    sig_tot = sig.copy()
    sig_tot[:, :, 0] -= qvisc
    sig_tot[:, :, 1] -= qvisc
    sig_tot[:, :, 2] -= qvisc

    # Internal force assembly: F_int = sum_g B^T sigma W_g J_g
    # fe[i, b] = - sum_g vol_g * sum_c S[b, c] * dndx[g, i, c]
    S = np.empty((n, 2, 3, 3))
    S[:, :, 0, 0], S[:, :, 1, 1], S[:, :, 2, 2] = sig_tot[:, :, 0], sig_tot[:, :, 1], sig_tot[:, :, 2]
    S[:, :, 0, 1] = S[:, :, 1, 0] = sig_tot[:, :, 3]
    S[:, :, 1, 2] = S[:, :, 2, 1] = sig_tot[:, :, 4]
    S[:, :, 0, 2] = S[:, :, 2, 0] = sig_tot[:, :, 5]

    fe = -np.einsum("ng,ngbc,ngic->nib", vol_g, S, dndx)

    # Energy bookkeeping (midpoint work rule + trapezoidal bulk viscosity)
    sig_mid = 0.5 * (sig_old + sig)
    w_visc = np.sum(vol_g * 0.5 * qvisc * (-trD * dt), axis=1) + qvw_pend * np.sum(-trD, axis=1) / 2.0
    deint0 = np.sum(vol_g * np.einsum("nga,nga->ng", sig_mid, deps), axis=1) + w_visc
    qvw_new = np.sum(vol_g * 0.5 * qvisc * dt, axis=1)

    # Critical Courant time step
    trD_min = trD.min(axis=1)
    Q = np.where(trD_min < 0.0, qb * c + qa * lc * np.abs(trD_min), 0.0)
    denom = Q + np.sqrt(Q * Q + c * c)
    dt_crit = np.where(denom > 0.0, dtfac * lc / denom, EP30)
    dt_crit = np.where(alive, dt_crit, EP30)

    return fe, dt_crit, w_visc, qvw_new, deint0


def forces(group, x, v, vr, dt, fint, mint):
    """Engine explicit cycle force kernel for /PENTA6 solid elements."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)
    if dt is None or dt < 0.0:
        return np.full(n, EP30)

    xe = x[conn]  # (n, 6, 3)
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

    # Kinematics and pre-update
    dndx, vol_g, vol_tot, lc, deps, trD = _pre(xe, ve, sig, dt, st["off"])
    vol_tot_safe = np.maximum(vol_tot, EM20)
    rho = st["mass"] / vol_tot_safe

    # Constitutive law update across slices at all Gauss points
    c = np.zeros(n)
    c_from_law = np.zeros(n, dtype=bool)

    sig_flat = sig.reshape(-1, 6)
    deps_flat = deps.reshape(-1, 6)
    epsp_flat = st["epsp"].reshape(-1)
    epsp_old = epsp_flat.copy() if st.get("chk_fail", False) else None

    for sl, mat, prop in st.get("slices", []):
        law = getattr(mat, "law", 1)
        rho0_sl = getattr(mat, "rho0", 0.0)
        E_sl = getattr(mat, "E", 0.0)
        sl_flat = slice(sl.start * 2, sl.stop * 2)
        if law == 0 or rho0_sl <= 0.0 or E_sl <= 0.0:
            sig_flat[sl_flat] = 0.0
            c[sl] = 0.0
            c_from_law[sl] = True
            continue

        _, _, c_new = materials.solid_update(
            mat, sig_flat[sl_flat], deps_flat[sl_flat], epsp_flat[sl_flat], dt, None)
        if c_new is not None:
            c[sl] = c_new.reshape(-1, 2).max(axis=1)
            c_from_law[sl] = True
        else:
            K_sl = getattr(mat, "K", 0.0)
            G_sl = getattr(mat, "G", 0.0)
            c[sl] = np.sqrt(np.maximum(K_sl + 4.0 * G_sl / 3.0, 0.0) / np.maximum(rho[sl], EM20))
            c_from_law[sl] = True

    # Failure model evaluation
    if st.get("chk_fail", False):
        off = st["off"]
        for sl, mat, prop in st.get("slices", []):
            eps_max = getattr(mat, "params", {}).get("eps_p_max", EP30)
            fail_obj = getattr(mat, "fail", None)
            if fail_obj is None and eps_max >= 1e30:
                continue
            broken = np.zeros(sl.stop - sl.start, dtype=bool)
            if fail_obj is not None:
                sig_avg = sig[sl].mean(axis=1)
                deps_avg = deps[sl].mean(axis=1)
                depsp_avg = (st["epsp"][sl] - epsp_old.reshape(-1, 2)[sl]).mean(axis=1)
                broken |= failure.solid_step(
                    fail_obj, sig_avg, depsp_avg, deps_avg, dt, st["dama"][sl])
            if eps_max < 1e30:
                broken |= st["epsp"][sl].max(axis=1) > eps_max
            off[sl][broken] = 0.0
        alive = off > 0.0
        sig[~alive] = 0.0

    # Bulk viscosity parameters per slice
    qa = np.zeros(n)
    qb = np.zeros(n)
    for sl, mat, prop in st.get("slices", []):
        qa[sl] = getattr(prop, "params", {}).get("qa", 1.1) if hasattr(prop, "params") else getattr(prop, "qa", 1.1)
        qb[sl] = getattr(prop, "params", {}).get("qb", 0.05) if hasattr(prop, "params") else getattr(prop, "qb", 0.05)

    fe, dt_crit, w_visc, qvw_new, deint0 = _post(
        xe, dndx, vol_g, vol_tot, lc, rho, trD, deps, sig, sig_old,
        qa, qb, c, alive, st["qvw_pend"], dt, st["dtfac"])

    alive = st["off"] > 0.0
    fe[~alive] = 0.0
    st["eint"] += deint0
    st["qvw_pend"] = qvw_new

    # Scatter internal nodal forces to global fint array
    if fint is not None:
        conn_flat = conn.reshape(-1)
        fe_flat = fe.reshape(-1, 3)
        valid = conn_flat >= 0
        scatter_add3(fint, conn_flat[valid], fe_flat[valid],
                     st.get("color_indices"), st.get("color_offsets"))

    dt_crit = np.where((c > 0.0) & alive, dt_crit, EP30)
    return dt_crit


# ----------------------------------------------------------------------------
# Implicit solver tangent
# ----------------------------------------------------------------------------

def tangent(group, x, epsp_incr=None):
    """Element tangent stiffness for the whole PENTA6 group: (n, 18, 18)."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 18, 18), dtype=float), np.zeros((0, 18), dtype=np.int64)

    xe = x[conn]
    dndx, vol_g, vol_tot = _geometry(xe)

    ke = np.zeros((n, 18, 18), dtype=np.float64)
    epi = np.zeros((n, 2)) if epsp_incr is None else (
        epsp_incr if np.ndim(epsp_incr) == 2 else np.repeat(epsp_incr[:, None], 2, axis=1)
    )

    from .. import materials as _materials

    for g in range(2):
        # Construct B-matrix at Gauss point g: shape (n, 6, 18)
        B = np.zeros((n, 6, 18), dtype=np.float64)
        gx = dndx[:, g, :, 0]  # (n, 6)
        gy = dndx[:, g, :, 1]
        gz = dndx[:, g, :, 2]
        ix = np.arange(6)

        B[:, 0, 3 * ix + 0] = gx
        B[:, 1, 3 * ix + 1] = gy
        B[:, 2, 3 * ix + 2] = gz
        B[:, 3, 3 * ix + 0] = gy
        B[:, 3, 3 * ix + 1] = gx
        B[:, 4, 3 * ix + 1] = gz
        B[:, 4, 3 * ix + 2] = gy
        B[:, 5, 3 * ix + 0] = gz
        B[:, 5, 3 * ix + 2] = gx

        for sl, mat, prop in st.get("slices", []):
            if getattr(mat, "law", 1) == 0:
                continue
            D = _materials.solid_tangent(mat, st["sig"][sl, g], st["epsp"][sl, g], epi[sl, g], None)
            Bs = B[sl]
            DB = np.einsum("mij,mjk->mik", D, Bs)
            ke[sl] += vol_g[sl, g][:, None, None] * np.einsum("mji,mjk->mik", Bs, DB)

    return ke, _edofs(conn)

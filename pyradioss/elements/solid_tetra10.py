"""
10-node tetrahedral solid element, quadratic formulation (/TETRA10).

Fortran origin: ``engine/source/elements/solid/solide10/`` — the 4-point
Gauss-integrated 10-node tet. Like the 8-node brick (solid_hexa8) the cycle
path is geometry → kinematics → Jaumann rotation → material → bulk viscosity
→ internal forces → critical dt.

M7-style performance structure
------------------------------
The force path is split into two blocks, ``_pre`` (geometry + kinematics +
Jaumann rotation) and ``_post`` (bulk viscosity, internal + forces, energies,
critical dt).  Both have an optional numba mirror in
``pyradioss.accel.jit_kernels`` selected via ``accel.get`` (see the accel
package docstring for the backend architecture and the parity contract).
The NumPy code HERE is the reference implementation.
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..accel import get as accel_get
from ..common.constants import EM20, EP30
from ..common.fastmath import cross3, det_inv33, norm3, scatter_add3

# Shape function derivatives at 4 Gauss points (10 nodes × 3 dims each).
# Gauss points are the 4-point rule for the tetrahedron (4 sub-tets).
_DN_DXI = np.array([[[-0.44721360, 0.00000000, 0.00000000],
  [ 0.00000000,-0.44721360, 0.00000000],
  [ 0.00000000, 0.00000000,-0.44721360],
  [-1.34164079,-1.34164079,-1.34164079],
  [ 0.55278640, 0.55278640, 0.00000000],
  [ 0.00000000, 0.55278640, 0.55278640],
  [ 0.55278640, 0.00000000, 0.55278640],
  [ 1.78885438,-0.55278640,-0.55278640],
  [-0.55278640, 1.78885438,-0.55278640],
  [-0.55278640,-0.55278640, 1.78885438]],

 [[ 1.34164079, 0.00000000, 0.00000000],
  [ 0.00000000,-0.44721360, 0.00000000],
  [ 0.00000000, 0.00000000,-0.44721360],
  [ 0.44721360, 0.44721360, 0.44721360],
  [ 0.55278640, 2.34164079, 0.00000000],
  [ 0.00000000, 0.55278640, 0.55278640],
  [ 0.55278640, 0.00000000, 2.34164079],
  [-1.78885438,-2.34164079,-2.34164079],
  [-0.55278640, 0.00000000,-0.55278640],
  [-0.55278640,-0.55278640, 0.00000000]],

 [[-0.44721360, 0.00000000, 0.00000000],
  [ 0.00000000, 1.34164079, 0.00000000],
  [ 0.00000000, 0.00000000,-0.44721360],
  [ 0.44721360, 0.44721360, 0.44721360],
  [ 2.34164079, 0.55278640, 0.00000000],
  [ 0.00000000, 0.55278640, 2.34164079],
  [ 0.55278640, 0.00000000, 0.55278640],
  [ 0.00000000,-0.55278640,-0.55278640],
  [-2.34164079,-1.78885438,-2.34164079],
  [-0.55278640,-0.55278640, 0.00000000]],

 [[-0.44721360, 0.00000000, 0.00000000],
  [ 0.00000000,-0.44721360, 0.00000000],
  [ 0.00000000, 0.00000000, 1.34164079],
  [ 0.44721360, 0.44721360, 0.44721360],
  [ 0.55278640, 0.55278640, 0.00000000],
  [ 0.00000000, 2.34164079, 0.55278640],
  [ 2.34164079, 0.00000000, 0.55278640],
  [ 0.00000000,-0.55278640,-0.55278640],
  [-0.55278640, 0.00000000,-0.55278640],
  [-2.34164079,-2.34164079,-1.78885438]]])

_WIP = np.array([0.25, 0.25, 0.25, 0.25])

_FACES = np.array([
    [0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2],
])

def _geometry(xe: np.ndarray):
    n = len(xe)
    if n == 0:
        return np.zeros((0, 4, 10, 3)), np.zeros((0, 4)), np.zeros(0)
    J = np.einsum("kia,nib->nkab", _DN_DXI, xe)  # (n, 4, 3, 3)
    J_flat = J.reshape(-1, 3, 3)
    a, b, c = J_flat[:, 0, 0], J_flat[:, 0, 1], J_flat[:, 0, 2]
    d, e, f = J_flat[:, 1, 0], J_flat[:, 1, 1], J_flat[:, 1, 2]
    g, h, i = J_flat[:, 2, 0], J_flat[:, 2, 1], J_flat[:, 2, 2]
    A = e * i - f * h
    B = f * g - d * i
    C = d * h - e * g
    detJ_flat = a * A + b * B + c * C
    detJ = detJ_flat.reshape(n, 4)
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
    Jinv = Jinv.reshape(n, 4, 3, 3)
    vol = detJ / 6.0
    vol_tot = np.sum(vol * _WIP, axis=1)
    dndx = np.einsum("kia,nkba->nkib", _DN_DXI, Jinv)
    bad_elem = np.abs(vol_tot) < 1e-12
    if np.any(bad_elem):
        dndx[bad_elem] = 0.0
    return dndx, vol, vol_tot

def _char_length(xe: np.ndarray, vol_tot: np.ndarray) -> np.ndarray:
    n = len(xe)
    if n == 0:
        return np.zeros(0)
    e1 = xe[:, _FACES[:, 1]] - xe[:, _FACES[:, 0]]
    e2 = xe[:, _FACES[:, 2]] - xe[:, _FACES[:, 0]]
    a = 0.5 * norm3(cross3(e1, e2))
    return 3.0 * np.maximum(vol_tot, EM20) / np.maximum(a.max(axis=1), EM20)

def _exact_dt_factor(dndx: np.ndarray, vol: np.ndarray, lc: np.ndarray,
                     slices) -> np.ndarray:
    n = len(vol)
    if n == 0:
        return np.zeros(0)
    return 0.25 * np.ones(n)

_TETRA10_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]


def init_group(group, model, log):
    n = group.n
    conn = group.conn
    if n == 0 or len(conn) == 0:
        group.state.update(
            sig=np.zeros((0, 4, 6)),
            epsp=np.zeros((0, 4)),
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

    # Reconstruct coordinates for virtual midside nodes (-1)
    xe = np.zeros((n, 10, 3), dtype=np.float64)
    xe[:, :4] = model.x0[conn[:, :4]]
    for m, (n1, n2) in enumerate(_TETRA10_EDGES):
        c_m = conn[:, 4 + m]
        real = c_m >= 0
        if real.any():
            xe[real, 4 + m] = model.x0[c_m[real]]
        virt = ~real
        if virt.any():
            xe[virt, 4 + m] = 0.5 * (xe[virt, n1] + xe[virt, n2])

    dndx0, vol, vol_tot = _geometry(xe)

    flip = vol_tot < 0.0
    if np.any(flip):
        log.warning(f"{flip.sum()} TETRA10 elements have negative volume.", "TETRA10 INIT")

    slices = group.state.get("slices", [])
    rho0 = np.zeros(n)
    for sl, mat, prop in slices:
        rho0[sl] = getattr(mat, "rho0", 0.0)
    mass = rho0 * np.maximum(vol_tot, 0.0)
    lc0 = _char_length(xe, vol_tot)

    chk_fail = any(
        getattr(mat, "fail", None) is not None or getattr(mat, "params", {}).get("eps_p_max", EP30) < 1e30
        for _, mat, _ in slices
    )
    group.state.update(
        sig=np.zeros((n, 4, 6)),
        epsp=np.zeros((n, 4)),
        vol0=vol_tot.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),
        off=np.ones(n),
        qvw_pend=np.zeros(n),
        dtfac=_exact_dt_factor(dndx0, vol, lc0, slices),
        chk_fail=chk_fail,
        dama=np.zeros(n),
    )

    elem_mass = np.zeros((n, 10), dtype=np.float64)
    is_slaved = conn[:, 4] < 0
    elem_mass[~is_slaved, :] = mass[~is_slaved, None] / 10.0
    elem_mass[is_slaved, :4] = mass[is_slaved, None] / 4.0
    elem_mass[is_slaved, 4:] = 0.0

    node_idx = conn.reshape(-1)
    mass_c = elem_mass.reshape(-1)
    valid = node_idx >= 0

    from .solid_hexa8 import _init_material_state
    _init_material_state(group, dndx0[:, 0])

    return node_idx[valid], mass_c[valid], None


# Engine-side force computation (one cycle)
# ----------------------------------------------------------------------------

def _pre(xe, ve, sig, dt, off):
    """Geometry + kinematics + Jaumann rotation: everything BEFORE the
    material law. Rotates ``sig`` in place; returns
    (dndx, vol, vol_tot, lc, deps, trD).

    Mirrored by accel.jit_kernels.tetra10_pre (same formulas,
    element-serial — the M7 parity contract)."""
    n = len(xe)

    # ---- geometry at t_{n+1/2} ----------------------------------------
    dndx, vol, vol_tot = _geometry(xe)
    vol_tot = np.maximum(vol_tot, EM20)
    lc = _char_length(xe, vol_tot)

    # ---- velocity gradient L[k,b,c] = sum_i ve[i,b] dndx[k,i,c] ------
    L = np.einsum("nib,nkic->nkbc", ve, dndx)
    D = 0.5 * (L + np.transpose(L, (0, 1, 3, 2)))
    trD = D[:, :, 0, 0] + D[:, :, 1, 1] + D[:, :, 2, 2]

    # round-off trace flush
    vgm = np.abs(ve).max(axis=(1, 2))[:, None] * np.abs(dndx).max(axis=(2, 3))
    trD = np.where(np.abs(trD) <= 1e-14 * vgm, 0.0, trD)

    # strain increment in Voigt form
    deps = np.empty((n, 4, 6))
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

    # Jaumann rotation of the old stress by the spin increment
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

    return dndx, vol, vol_tot, lc, deps, trD


def _post(xe, dndx, vol, vol_tot, lc, rho, trD, deps, sig, sig_old,
          qa, qb, c, alive, qvw_pend, dt, dtfac):
    """Bulk viscosity, internal forces, energies, critical dt — everything
    AFTER the material law. Returns
    (fe, dt_crit, w_visc, qvw_new, deint0).

    Mirrored by accel.jit_kernels.tetra10_post (same formulas,
    element-serial — the M7 parity contract)."""
    n = len(xe)

    # ---- bulk viscosity ------------------------------------------------
    compressing = (trD < 0.0) & alive[:, None]
    qvisc = np.where(
        compressing,
        rho[:, None] * lc[:, None] * (qa[:, None]**2 * lc[:, None] * trD**2 - qb[:, None] * c[:, None] * trD),
        0.0)

    # total stress with viscous pressure
    sig_tot = sig.copy()
    sig_tot[:, :, 0] -= qvisc
    sig_tot[:, :, 1] -= qvisc
    sig_tot[:, :, 2] -= qvisc

    # ---- internal forces: fe[i,b] = -sum_k w_k vol_k S[k,b,c] dndx[k,i,c]
    S = np.empty((n, 4, 3, 3))
    S[:, :, 0, 0], S[:, :, 1, 1], S[:, :, 2, 2] = sig_tot[:, :, 0], sig_tot[:, :, 1], sig_tot[:, :, 2]
    S[:, :, 0, 1] = S[:, :, 1, 0] = sig_tot[:, :, 3]
    S[:, :, 1, 2] = S[:, :, 2, 1] = sig_tot[:, :, 4]
    S[:, :, 0, 2] = S[:, :, 2, 0] = sig_tot[:, :, 5]

    fe = -np.einsum("k,nk,nkbc,nkic->nib", _WIP, vol, S, dndx)

    # ---- energies (trapezoidal qb booking) -----------------------------
    sig_mid = 0.5 * (sig_old + sig)
    w_visc = np.sum(_WIP * vol * 0.5 * qvisc * (-trD * dt), axis=1) + qvw_pend * np.sum(-trD, axis=1) / 4.0
    sig_mid_4 = sig_mid.reshape(-1, 6).reshape(n, 4, 6)
    deint0 = np.sum(_WIP * vol * np.einsum("nka,nka->nk", sig_mid_4, deps), axis=1) + w_visc
    qvw_new = np.sum(_WIP * vol * 0.5 * qvisc * dt, axis=1)

    # ---- critical time step --------------------------------------------
    trD_min = trD.min(axis=1)
    Q = np.where(trD_min < 0.0, qb * c + qa * lc * np.abs(trD_min), 0.0)
    denom = Q + np.sqrt(Q * Q + c * c)
    dt_crit = np.where(denom > 0.0, dtfac * lc / denom, EP30)
    dt_crit = np.where(alive, dt_crit, EP30)

    return fe, dt_crit, w_visc, qvw_new, deint0


def forces(group, x, v, vr, dt, fint, mint):
    """Engine-cycle element force routine — dispatches to numba JIT mirrors
    when the numba backend is active, otherwise runs the NumPy reference
    (_pre + _post above)."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)
    if dt < 0.0:
        return np.full(n, EP30)

    # Reconstruct positions and velocities for virtual midside nodes (-1)
    xe = np.zeros((n, 10, 3), dtype=np.float64)
    ve = np.zeros((n, 10, 3), dtype=np.float64)
    xe[:, :4] = x[conn[:, :4]]
    if v is not None:
        ve[:, :4] = v[conn[:, :4]]
    for m, (n1, n2) in enumerate(_TETRA10_EDGES):
        c_m = conn[:, 4 + m]
        real = c_m >= 0
        if real.any():
            xe[real, 4 + m] = x[c_m[real]]
            if v is not None:
                ve[real, 4 + m] = v[c_m[real]]
        virt = ~real
        if virt.any():
            xe[virt, 4 + m] = 0.5 * (xe[virt, n1] + xe[virt, n2])
            if v is not None:
                ve[virt, 4 + m] = 0.5 * (ve[virt, n1] + ve[virt, n2])

    sig = st["sig"]
    sig_old = sig.copy()
    alive = st["off"] > 0.0

    # ---- pre block: geometry, D & W, Jaumann rotation ------------------
    jit = accel_get("tetra10_pre")
    if jit is not None:
        dndx, vol, vol_tot, lc, deps, trD = jit(xe, ve, sig, dt, st["off"])
    else:
        dndx, vol, vol_tot, lc, deps, trD = _pre(xe, ve, sig, dt, st["off"])
    vol_tot_safe = np.maximum(vol_tot, EM20)
    rho = st["mass"] / vol_tot_safe

    # ---- material law per part slice -----------------------------------
    c = np.zeros(group.n)
    c_from_law = np.zeros(group.n, dtype=bool)

    sig_flat = sig.reshape(-1, 6)
    deps_flat = deps.reshape(-1, 6)
    epsp_flat = st["epsp"].reshape(-1)
    epsp_old = epsp_flat.copy() if st.get("chk_fail", False) else None

    for sl, mat, prop in st.get("slices", []):
        law = getattr(mat, "law", 1)
        rho0_sl = getattr(mat, "rho0", 0.0)
        E_sl = getattr(mat, "E", 0.0)
        sl_flat = slice(sl.start * 4, sl.stop * 4)
        if law == 0 or rho0_sl <= 0.0 or E_sl <= 0.0:
            sig_flat[sl_flat] = 0.0
            c[sl] = 0.0
            c_from_law[sl] = True
            continue

        _, _, c_new = materials.solid_update(
            mat, sig_flat[sl_flat], deps_flat[sl_flat], epsp_flat[sl_flat], dt, None)
        if c_new is not None:
            c_new = c_new.reshape(-1, 4).max(axis=1)
            c[sl] = c_new
            c_from_law[sl] = True
        else:
            K_sl = getattr(mat, "K", 0.0)
            G_sl = getattr(mat, "G", 0.0)
            c[sl] = np.sqrt((K_sl + 4.0 * G_sl / 3.0) / np.maximum(rho[sl], EM20))
            c_from_law[sl] = True

    # ---- failure evaluation -------------------------------------------
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
                depsp_avg = (st["epsp"][sl] - epsp_old.reshape(-1, 4)[sl]).mean(axis=1)
                broken |= failure.solid_step(
                    fail_obj, sig_avg, depsp_avg, deps_avg, dt, st["dama"][sl])
            if eps_max < 1e30:
                broken |= st["epsp"][sl].max(axis=1) > eps_max
            off[sl][broken] = 0.0
        alive = off > 0.0
        sig[~alive] = 0.0

    # ---- bulk-viscosity coefficients per slice -------------------------
    qa = np.zeros(group.n)
    qb = np.zeros(group.n)
    for sl, mat, prop in st.get("slices", []):
        qa[sl] = getattr(prop, "params", {}).get("qa", 1.1) if hasattr(prop, "params") else getattr(prop, "qa", 1.1)
        qb[sl] = getattr(prop, "params", {}).get("qb", 0.05) if hasattr(prop, "params") else getattr(prop, "qb", 0.05)

    # ---- post block: viscosity, forces, energies, dt -------------------
    jit = accel_get("tetra10_post")
    if jit is not None:
        fe, dt_crit, w_visc, qvw_new, deint0 = jit(
            xe, dndx, vol, vol_tot, lc, rho, trD, deps, sig, sig_old,
            qa, qb, c, alive, st["qvw_pend"], dt, st["dtfac"])
    else:
        fe, dt_crit, w_visc, qvw_new, deint0 = _post(
            xe, dndx, vol, vol_tot, lc, rho, trD, deps, sig, sig_old,
            qa, qb, c, alive, st["qvw_pend"], dt, st["dtfac"])

    alive = st["off"] > 0.0
    fe[~alive] = 0.0
    st["eint"] += deint0
    st["qvw_pend"] = qvw_new

    # For any virtual midside node (conn < 0), redistribute 50/50 back to corner pairs
    for m, (n1, n2) in enumerate(_TETRA10_EDGES):
        virt = conn[:, 4 + m] < 0
        if virt.any():
            f_mid = fe[virt, 4 + m]
            fe[virt, n1] += 0.5 * f_mid
            fe[virt, n2] += 0.5 * f_mid
            fe[virt, 4 + m] = 0.0

    # ---- scatter to global arrays (only non-negative node indices) ----
    if fint is not None:
        conn_flat = conn.reshape(-1)
        fe_flat = fe.reshape(-1, 3)
        valid = conn_flat >= 0
        scatter_add3(fint, conn_flat[valid], fe_flat[valid], st.get("color_indices"), st.get("color_offsets"))

    # Guard dt_crit when sound speed c <= 0 (e.g. void / dead)
    dt_crit = np.where((c > 0.0) & alive, dt_crit, EP30)
    return dt_crit


def _edofs(conn):
    """(n, 30) global scalar DOF slot ids, node-major [ux, uy, uz] * 10."""
    n = len(conn)
    if n == 0:
        return np.zeros((0, 30), dtype=np.int64)
    ix = np.arange(10)
    edofs = np.empty((n, 30), dtype=np.int64)
    safe_conn = np.maximum(conn, 0)
    for c in range(3):
        edofs[:, 3 * ix + c] = safe_conn * 6 + c
    return edofs


def tangent(group, x, epsp_incr=None):
    """Element tangent stiffness for the whole tetra10 group.

    Returns ``(ke, edofs)``: ``ke`` (n, 30, 30) dense element tangents
    (translations only), ``edofs`` (n, 30) global scalar DOF slot ids."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 30, 30), dtype=float), np.zeros((0, 30), dtype=np.int64)

    xe = np.zeros((n, 10, 3), dtype=np.float64)
    xe[:, :4] = x[conn[:, :4]]
    for m, (n1, n2) in enumerate(_TETRA10_EDGES):
        c_m = conn[:, 4 + m]
        real = c_m >= 0
        if real.any():
            xe[real, 4 + m] = x[c_m[real]]
        virt = ~real
        if virt.any():
            xe[virt, 4 + m] = 0.5 * (xe[virt, n1] + xe[virt, n2])

    dndx, vol, vol_tot = _geometry(xe)
    vol = np.maximum(vol, EM20)

    # B-operator (n, 4, 6, 30) across 4 Gauss points
    B = np.zeros((n, 4, 6, 30))
    ix = np.arange(10)
    for k in range(4):
        gx = dndx[:, k, :, 0]
        gy = dndx[:, k, :, 1]
        gz = dndx[:, k, :, 2]
        B[:, k, 0, 3 * ix + 0] = gx
        B[:, k, 1, 3 * ix + 1] = gy
        B[:, k, 2, 3 * ix + 2] = gz
        B[:, k, 3, 3 * ix + 0] = gy
        B[:, k, 3, 3 * ix + 1] = gx
        B[:, k, 4, 3 * ix + 1] = gz
        B[:, k, 4, 3 * ix + 2] = gy
        B[:, k, 5, 3 * ix + 0] = gz
        B[:, k, 5, 3 * ix + 2] = gx

    from .. import materials as _materials
    ke = np.zeros((n, 30, 30))
    epi = np.zeros((n, 4)) if epsp_incr is None else epsp_incr
    if epi.ndim == 1:
        epi = np.tile(epi[:, None], (1, 4))

    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            continue
        sig_sl = st["sig"][sl]   # (m, 4, 6)
        epsp_sl = st["epsp"][sl] # (m, 4)
        epi_sl = epi[sl]         # (m, 4)
        for k in range(4):
            D = _materials.solid_tangent(
                mat, sig_sl[:, k], epsp_sl[:, k], epi_sl[:, k], None)  # (m, 6, 6)
            Bk = B[sl, k]  # (m, 6, 30)
            DB = np.einsum("mij,mjk->mik", D, Bk)
            ke[sl] += _WIP[k] * vol[sl, k, None, None] * np.einsum("mji,mjk->mik", Bk, DB)

    return ke, _edofs(conn)


def kgeo(group, x):
    """Geometric (initial-stress) element stiffness for the tetra10 group at
    geometry ``x`` — sum_k w_k V_k gradN_a . sigma_k . gradN_b replicated over
    the three translation directions."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 30, 30), dtype=float), np.zeros((0, 30), dtype=np.int64)

    xe = np.zeros((n, 10, 3), dtype=np.float64)
    xe[:, :4] = x[conn[:, :4]]
    for m, (n1, n2) in enumerate(_TETRA10_EDGES):
        c_m = conn[:, 4 + m]
        real = c_m >= 0
        if real.any():
            xe[real, 4 + m] = x[c_m[real]]
        virt = ~real
        if virt.any():
            xe[virt, 4 + m] = 0.5 * (xe[virt, n1] + xe[virt, n2])

    dndx, vol, vol_tot = _geometry(xe)
    vol = np.maximum(vol, EM20)
    sig = st["sig"]
    S = np.empty((n, 4, 3, 3))
    S[:, :, 0, 0], S[:, :, 1, 1], S[:, :, 2, 2] = sig[:, :, 0], sig[:, :, 1], sig[:, :, 2]
    S[:, :, 0, 1] = S[:, :, 1, 0] = sig[:, :, 3]
    S[:, :, 1, 2] = S[:, :, 2, 1] = sig[:, :, 4]
    S[:, :, 0, 2] = S[:, :, 2, 0] = sig[:, :, 5]

    ke = np.zeros((n, 30, 30))
    ix = np.arange(10)
    for k in range(4):
        gk = _WIP[k] * vol[:, k, None, None] * np.einsum(
            "nac,ncd,nbd->nab", dndx[:, k], S[:, k], dndx[:, k])  # (n, 10, 10)
        for b in range(3):
            rows = (3 * ix + b)[:, None]
            cols = (3 * ix + b)[None, :]
            ke[:, rows, cols] += gk

    return ke, _edofs(conn)


#: Analytical ∫ Nᵀ N dV over the reference tet (units of V), exact simplex moment matrix.
_M_TET10 = np.array([
    [ 6,  1,  1,  1, -4, -6, -4, -4, -6, -6],
    [ 1,  6,  1,  1, -4, -4, -6, -6, -4, -6],
    [ 1,  1,  6,  1, -6, -4, -4, -6, -6, -4],
    [ 1,  1,  1,  6, -6, -6, -6, -4, -4, -4],
    [-4, -4, -6, -6, 32, 16, 16, 16, 16,  8],
    [-6, -4, -4, -6, 16, 32, 16,  8, 16, 16],
    [-4, -6, -4, -6, 16, 16, 32, 16,  8, 16],
    [-4, -6, -6, -4, 16,  8, 16, 32, 16, 16],
    [-6, -4, -6, -4, 16, 16,  8, 16, 32, 16],
    [-6, -6, -4, -4,  8, 16, 16, 16, 16, 32],
], dtype=float) / 420.0


def consistent_mass(group, x=None):
    """Consistent element mass ∫ρ Nᵀ N dV of the 10-node quadratic tet:
    exact analytical simplex moment matrix M_tet10 ⊗ I3, built on the stored
    element mass.

    Returns ``(me (n, 30, 30), edofs (n, 30))`` — translations only, the same
    node-major addressing as ``tangent()``."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 30, 30), dtype=float), np.zeros((0, 30), dtype=np.int64)
    m = st["mass"]  # ρ V0, per element
    me = np.zeros((n, 30, 30))
    for a in range(10):
        for b in range(10):
            f = m * _M_TET10[a, b]
            for c in range(3):
                me[:, a * 3 + c, b * 3 + c] = f
    return me, _edofs(conn)


def static_internal_forces(group, x, u, ur, fint, mint):
    """Internal nodal force at configuration ``x`` from the CURRENT stress
    state — updated-Lagrangian end-configuration assembly of the implicit
    residual: fe = -sum_k w_k vol_k S_k gradN_k."""
    if group.n == 0 or len(group.conn) == 0 or fint is None:
        return
    st = group.state
    conn = group.conn
    n = group.n

    xe = np.zeros((n, 10, 3), dtype=np.float64)
    xe[:, :4] = x[conn[:, :4]]
    for m, (n1, n2) in enumerate(_TETRA10_EDGES):
        c_m = conn[:, 4 + m]
        real = c_m >= 0
        if real.any():
            xe[real, 4 + m] = x[c_m[real]]
        virt = ~real
        if virt.any():
            xe[virt, 4 + m] = 0.5 * (xe[virt, n1] + xe[virt, n2])

    dndx, vol, vol_tot = _geometry(xe)
    vol = np.maximum(vol, EM20)

    sig = st["sig"]
    S = np.empty((n, 4, 3, 3))
    S[:, :, 0, 0], S[:, :, 1, 1], S[:, :, 2, 2] = sig[:, :, 0], sig[:, :, 1], sig[:, :, 2]
    S[:, :, 0, 1] = S[:, :, 1, 0] = sig[:, :, 3]
    S[:, :, 1, 2] = S[:, :, 2, 1] = sig[:, :, 4]
    S[:, :, 0, 2] = S[:, :, 2, 0] = sig[:, :, 5]

    fe = -np.einsum("k,nk,nkbc,nkic->nib", _WIP, vol, S, dndx)

    # For any virtual midside node (conn < 0), redistribute 50/50 back to corner pairs
    for m, (n1, n2) in enumerate(_TETRA10_EDGES):
        virt = conn[:, 4 + m] < 0
        if virt.any():
            f_mid = fe[virt, 4 + m]
            fe[virt, n1] += 0.5 * f_mid
            fe[virt, n2] += 0.5 * f_mid
            fe[virt, 4 + m] = 0.0

    conn_flat = conn.reshape(-1)
    fe_flat = fe.reshape(-1, 3)
    valid = conn_flat >= 0
    scatter_add3(fint, conn_flat[valid], fe_flat[valid], st.get("color_indices"), st.get("color_offsets"))


def implicit_internal_forces(group, x_ref, u, ur, fint, mint, nlgeom):
    """Implicit residual internal forces dispatch for tetra10.

    Linear geometry (nlgeom=False): evaluates forces at x_ref with displacement u.
    Nonlinear geometry (nlgeom=True): advances state at midpoint configuration
    x_ref + 0.5*u, then assembles internal forces on end configuration x_ref + u."""
    if group.n == 0 or len(group.conn) == 0:
        return
    if not nlgeom:
        forces(group, x_ref, u, ur, 1.0, fint, mint)
    else:
        x_mid = x_ref + 0.5 * u
        x_end = x_ref + u
        junk_f = np.zeros_like(fint) if fint is not None else None
        forces(group, x_mid, u, ur, 1.0, junk_f, mint)
        static_internal_forces(group, x_end, u, ur, fint, mint)


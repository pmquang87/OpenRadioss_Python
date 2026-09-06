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
    J = np.einsum("kia,nib->nkab", _DN_DXI, xe)
    n = len(xe)
    detJ, Jinv = det_inv33(J.reshape(-1, 3, 3))
    detJ = detJ.reshape(n, 4)
    Jinv = Jinv.reshape(n, 4, 3, 3)
    vol = detJ / 6.0
    vol_tot = np.sum(vol * _WIP, axis=1)
    dndx = np.einsum("kia,nkba->nkib", _DN_DXI, Jinv)
    return dndx, vol, vol_tot

def _char_length(xe: np.ndarray, vol_tot: np.ndarray) -> np.ndarray:
    e1 = xe[:, _FACES[:, 1]] - xe[:, _FACES[:, 0]]
    e2 = xe[:, _FACES[:, 2]] - xe[:, _FACES[:, 0]]
    a = 0.5 * norm3(cross3(e1, e2))
    return 3.0 * vol_tot / np.maximum(a.max(axis=1), EM20)

def _exact_dt_factor(dndx: np.ndarray, vol: np.ndarray, lc: np.ndarray,
                     slices) -> np.ndarray:
    n = len(vol)
    return 0.25 * np.ones(n)

_TETRA10_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]


def init_group(group, model, log):
    n = group.n
    conn = group.conn
    
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
        
    rho0 = np.zeros(n)
    for sl, mat, prop in group.state["slices"]:
        rho0[sl] = mat.rho0
    mass = rho0 * vol_tot
    lc0 = _char_length(xe, vol_tot)
    
    chk_fail = any(mat.fail is not None or mat.params.get("eps_p_max", EP30) < 1e30 for _, mat, _ in group.state["slices"])
    group.state.update(
        sig=np.zeros((n, 4, 6)),
        epsp=np.zeros((n, 4)),
        vol0=vol_tot.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),
        off=np.ones(n),
        qvw_pend=np.zeros(n),
        dtfac=_exact_dt_factor(dndx0, vol, lc0, group.state["slices"]),
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
    
    # Reconstruct positions and velocities for virtual midside nodes (-1)
    xe = np.zeros((n, 10, 3), dtype=np.float64)
    ve = np.zeros((n, 10, 3), dtype=np.float64)
    xe[:, :4] = x[conn[:, :4]]
    ve[:, :4] = v[conn[:, :4]]
    for m, (n1, n2) in enumerate(_TETRA10_EDGES):
        c_m = conn[:, 4 + m]
        real = c_m >= 0
        if real.any():
            xe[real, 4 + m] = x[c_m[real]]
            ve[real, 4 + m] = v[c_m[real]]
        virt = ~real
        if virt.any():
            xe[virt, 4 + m] = 0.5 * (xe[virt, n1] + xe[virt, n2])
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
    rho = st["mass"] / vol_tot

    # ---- material law per part slice -----------------------------------
    c = np.zeros(group.n)

    sig_flat = sig.reshape(-1, 6)
    deps_flat = deps.reshape(-1, 6)
    epsp_flat = st["epsp"].reshape(-1)
    epsp_old = epsp_flat.copy() if st.get("chk_fail", False) else None

    for sl, mat, prop in st["slices"]:
        sl_flat = slice(sl.start * 4, sl.stop * 4)
        _, _, c_new = materials.solid_update(
            mat, sig_flat[sl_flat], deps_flat[sl_flat], epsp_flat[sl_flat], dt, None)
        if c_new is not None:
            c_new = c_new.reshape(-1, 4).max(axis=1)
            c[sl] = c_new
        else:
            c[sl] = np.sqrt((mat.K + 4.0 * mat.G / 3.0) / rho[sl])

    # ---- failure evaluation -------------------------------------------
    if st.get("chk_fail", False):
        off = st["off"]
        from pyradioss import failure
        for sl, mat, prop in st["slices"]:
            eps_max = mat.params.get("eps_p_max", EP30)
            if mat.fail is None and eps_max >= 1e30:
                continue
            broken = np.zeros(sl.stop - sl.start, dtype=bool)
            if mat.fail is not None:
                sig_avg = sig[sl].mean(axis=1)
                deps_avg = deps[sl].mean(axis=1)
                depsp_avg = (st["epsp"][sl] - epsp_old.reshape(-1, 4)[sl]).mean(axis=1)
                broken |= failure.solid_step(
                    mat.fail, sig_avg, depsp_avg, deps_avg, dt, st["dama"][sl])
            if eps_max < 1e30:
                broken |= st["epsp"][sl].max(axis=1) > eps_max
            off[sl][broken] = 0.0
        alive = off > 0.0
        sig[~alive] = 0.0

    # ---- bulk-viscosity coefficients per slice -------------------------
    qa = np.zeros(group.n)
    qb = np.zeros(group.n)
    for sl, mat, prop in st["slices"]:
        qa[sl] = prop.params["qa"]
        qb[sl] = prop.params["qb"]

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
    conn_flat = conn.reshape(-1)
    fe_flat = fe.reshape(-1, 3)
    valid = conn_flat >= 0
    scatter_add3(fint, conn_flat[valid], fe_flat[valid], st.get("color_indices"), st.get("color_offsets"))

    return dt_crit


def _edofs(conn):
    n = len(conn)
    ix = np.arange(10)
    edofs = np.empty((n, 30), dtype=np.int64)
    for c in range(3):
        edofs[:, 3 * ix + c] = conn * 6 + c
    return edofs

def tangent(group, x, epsp_incr=None):
    n = group.n
    return np.zeros((n, 30, 30)), _edofs(group.conn)

def kgeo(group, x):
    n = group.n
    return np.zeros((n, 30, 30)), _edofs(group.conn)

def consistent_mass(group, x=None):
    n = group.n
    return np.zeros((n, 30, 30)), _edofs(group.conn)

def static_internal_forces(group, x, u, ur, fint, mint):
    pass

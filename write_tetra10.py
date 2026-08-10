import numpy as np

dn_dxi = np.load('dn_dxi.npy')
dn_dxi_str = np.array2string(dn_dxi, separator=',', formatter={'float_kind':lambda x: f'{x: .8f}'})

content = '''"""
10-node tetrahedral solid element, quadratic formulation (/TETRA10).
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import cross3, det_inv33, norm3, scatter_add3

_DN_DXI = np.array(''' + dn_dxi_str + ''')

_WIP = np.array([0.25, 0.25, 0.25, 0.25])

_FACES = np.array([
    [0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2],
])

def _geometry(xe: np.ndarray):
    J = np.einsum("kia,nib->nkab", _DN_DXI, xe)
    detJ, Jinv = det_inv33(J)
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
    return np.ones(n)

def init_group(group, model, log):
    xe = model.x0[group.conn]
    dndx0, vol, vol_tot = _geometry(xe)
    
    flip = vol_tot < 0.0
    if np.any(flip):
        log.warning(f"{flip.sum()} TETRA10 elements have negative volume.", "TETRA10 INIT")
        
    n = group.n
    rho0 = np.zeros(n)
    for sl, mat, prop in group.state["slices"]:
        rho0[sl] = mat.rho0
    mass = rho0 * vol_tot
    lc0 = _char_length(xe, vol_tot)
    
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
    )
    mass_c = np.repeat(mass / 10.0, 10)
    node_idx = group.conn.reshape(-1)
    
    from .solid_hexa8 import _init_material_state
    _init_material_state(group, dndx0[:, 0]) 
    group.state["chk_fail"] = False
    
    return node_idx, mass_c, None

def forces(group, x, v, vr, dt, fint, mint):
    st = group.state
    conn = group.conn
    xe = x[conn]
    ve = v[conn]
    
    dndx, vol, vol_tot = _geometry(xe)
    vol_tot = np.maximum(vol_tot, EM20)
    rho = st["mass"] / vol_tot
    lc = _char_length(xe, vol_tot)
    
    L = np.einsum("nib,nkic->nkbc", ve, dndx)
    D = 0.5 * (L + np.transpose(L, (0, 1, 3, 2)))
    trD = D[:, :, 0, 0] + D[:, :, 1, 1] + D[:, :, 2, 2]
    
    vgm = np.abs(ve).max(axis=(1, 2))[:, None] * np.abs(dndx).max(axis=(2, 3))
    trD = np.where(np.abs(trD) <= 1e-14 * vgm, 0.0, trD)
    
    deps = np.empty((group.n, 4, 6))
    deps[:, :, 0] = D[:, :, 0, 0] * dt
    deps[:, :, 1] = D[:, :, 1, 1] * dt
    deps[:, :, 2] = D[:, :, 2, 2] * dt
    deps[:, :, 3] = 2.0 * D[:, :, 0, 1] * dt
    deps[:, :, 4] = 2.0 * D[:, :, 1, 2] * dt
    deps[:, :, 5] = 2.0 * D[:, :, 0, 2] * dt
    
    alive = st["off"] > 0.0
    if not alive.all():
        deps[~alive] = 0.0
        trD = np.where(alive[:, None], trD, 0.0)
        
    sig = st["sig"]
    sig_old = sig.copy()
    
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
    
    c = np.zeros(group.n)
    
    sig_flat = sig.reshape(-1, 6)
    deps_flat = deps.reshape(-1, 6)
    epsp_flat = st["epsp"].reshape(-1)
    
    for sl, mat, prop in st["slices"]:
        sl_flat = slice(sl.start * 4, sl.stop * 4)
        _, _, c_new = materials.solid_update(
            mat, sig_flat[sl_flat], deps_flat[sl_flat], epsp_flat[sl_flat], dt, None)
        if c_new is not None:
            c_new = c_new.reshape(-1, 4).max(axis=1)
            c[sl] = c_new
        else:
            c[sl] = np.sqrt((mat.K + 4.0 * mat.G / 3.0) / rho[sl])
            
    qa = np.zeros(group.n)
    qb = np.zeros(group.n)
    for sl, mat, prop in st["slices"]:
        qa[sl] = prop.params["qa"]
        qb[sl] = prop.params["qb"]
        
    compressing = (trD < 0.0) & alive[:, None]
    qvisc = np.where(
        compressing,
        rho[:, None] * lc[:, None] * (qa[:, None]**2 * lc[:, None] * trD**2 - qb[:, None] * c[:, None] * trD),
        0.0)
        
    sig_tot = sig.copy()
    sig_tot[:, :, 0] -= qvisc
    sig_tot[:, :, 1] -= qvisc
    sig_tot[:, :, 2] -= qvisc
    
    S = np.empty((group.n, 4, 3, 3))
    S[:, :, 0, 0], S[:, :, 1, 1], S[:, :, 2, 2] = sig_tot[:, :, 0], sig_tot[:, :, 1], sig_tot[:, :, 2]
    S[:, :, 0, 1] = S[:, :, 1, 0] = sig_tot[:, :, 3]
    S[:, :, 1, 2] = S[:, :, 2, 1] = sig_tot[:, :, 4]
    S[:, :, 0, 2] = S[:, :, 2, 0] = sig_tot[:, :, 5]
    
    fe = -np.einsum("k,nk,nkbc,nkic->nib", _WIP, vol, S, dndx)
    
    sig_mid = 0.5 * (sig_old + sig)
    w_visc = np.sum(_WIP * vol * 0.5 * qvisc * (-trD * dt), axis=1) + st["qvw_pend"] * np.sum(-trD, axis=1)/4.0 
    deint = np.sum(_WIP * vol * np.einsum("nk,nk->nk", sig_mid.reshape(-1,6).reshape(group.n, 4, 6), deps), axis=1) + w_visc
    st["eint"] += deint
    st["qvw_pend"] = np.sum(_WIP * vol * 0.5 * qvisc * dt, axis=1)
    
    scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3), st.get("color_indices"), st.get("color_offsets"))
    
    trD_min = trD.min(axis=1)
    Q = np.where(trD_min < 0.0, qb * c + qa * lc * np.abs(trD_min), 0.0)
    dt_crit = st["dtfac"] * lc / (Q + np.sqrt(Q * Q + c * c))
    return np.where(alive, dt_crit, EP30)

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
'''

with open('pyradioss/elements/solid_tetra10.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done writing solid_tetra10.py')

import logging
import numpy as np
from ..common.constants import EM20
from .shell_bt4 import _init_material_state
from .shell_tri3 import _local_geometry, _char_length, _exact_dt_factor

log = logging.getLogger(__name__)

def init_group(group, model, log):
    """Element buffer + lumped mass/inertia (starter c3init3/c3mass3)."""
    xe = model.x0[group.conn]
    E, xl, area, B1, B2 = _local_geometry(xe)
    bad = area <= 0.0
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/SH3N_DKT18 {eid}: zero area (coincident nodes?)",
                      "DKT18 INIT")

    n = group.n
    thick = np.zeros(n)
    rho0 = np.zeros(n)
    nip_max = 1
    for sl, mat, prop in group.state["slices"]:
        thick[sl] = prop.params["thick"]
        rho0[sl] = mat.rho0
        nip_max = max(nip_max, int(prop.params["nip"]))
    mass = rho0 * thick * area

    # Through-thickness Gauss stations per part slice (same as shell_bt4)
    zw = []
    for sl, mat, prop in group.state["slices"]:
        nip = int(prop.params["nip"])
        gp, gw = np.polynomial.legendre.leggauss(nip)
        zw.append((gp * 0.5, gw * 0.5))  # relative to thickness
    group.state.update(
        sig=np.zeros((n, nip_max, 3)),   # in-plane stress per layer
        qshear=np.zeros((n, 2)),         # transverse shear stress (elastic)
        epsp=np.zeros((n, nip_max)),
        thick=thick,
        area0=area.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),
        zw=zw,
        dtfac=_exact_dt_factor(B1, B2, area, _char_length(xl, area),
                               thick, group.state["slices"]),
    )
    _init_material_state(group, nip_max)
    # orthotropy fiber frame (/PROP/TYPE9 SH_ORTH, TYPE16) — see shell_bt4
    from . import shell_ortho
    group.state["ortho"] = shell_ortho.build_group_ortho(
        group.state["slices"], E, n, log, group.ids)
    node_idx = group.conn.reshape(-1)
    mass_c = np.repeat(mass / 3.0, 3)
    # generous lumped rotational inertia (Key's trick, see module docstring)
    group.state["dt_iner"] = mass / 3.0 * (area / 4.5 + thick ** 2 / 12.0)
    inertia_c = np.repeat(group.state["dt_iner"], 3)
    return node_idx, mass_c, inertia_c

from ..accel.jit_kernels.shells_dkt18 import (
    cdkcoor3, cdkderic3, cdkdefo3, cdkderi3, cdkcurv3, cdkfint3, cdkfcum3
)
from .shell_bt4 import _layer_extra, _layer_failure
from .. import materials

def forces(group, x, v, vr, dt, fint, mint):
    """Compute DKT18 internal forces and critical time step."""
    st = group.state
    conn = group.conn
    n = group.n
    xe = x[conn]
    ve = v[conn]
    re = vr[conn]
    
    # 1. Geometry and local frame
    (area2, xl2, yl2, xl3, yl3, vlx, vly, vlz, rlx, rly,
     e_frame) = cdkcoor3(xe, ve, re, dt)
    e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z = e_frame
    
    # 2. Derivatives and interpolation constants
    thick = st["thick"]
    nu = np.zeros(n)
    for sl, mat, prop in st["slices"]:
        nu[sl] = mat.nu
    
    # volume is area * thickness
    vol0 = area2 * thick
    vol00 = vol0.copy()
    
    px2, py2, px3, py3, px, py, pxy, pyy = cdkderic3(xl2, yl2, xl3, yl3, area2, vol00, nu, thick**2)
    
    # 3. Membrane rates
    exx = np.zeros(n)
    eyy = np.zeros(n)
    exy = np.zeros(n)
    exz = np.zeros(n)
    eyz = np.zeros(n)
    epsdot = np.zeros((3, n))
    gstr = np.zeros((n, 3))
    vdef = np.zeros((n, 3))
    
    cdkdefo3(vlx, vly, px2, py2, px3, py3, exx, eyy, exy, exz, eyz, dt, epsdot, 0, False, gstr, vdef, False)
    
    f11 = np.zeros(n)
    f12 = np.zeros(n)
    f13 = np.zeros(n)
    f21 = np.zeros(n)
    f22 = np.zeros(n)
    f23 = np.zeros(n)
    f31 = np.zeros(n)
    f32 = np.zeros(n)
    f33 = np.zeros(n)
    
    m11 = np.zeros(n)
    m12 = np.zeros(n)
    m13 = np.zeros(n)
    m21 = np.zeros(n)
    m22 = np.zeros(n)
    m23 = np.zeros(n)
    m31 = np.zeros(n)
    m32 = np.zeros(n)
    m33 = np.zeros(n)
    
    # DKT18 uses 3 Gauss points in-plane for bending
    A_HAMMER = [(0.166666666666667, 0.666666666666667),
                (0.666666666666667, 0.166666666666667),
                (0.166666666666667, 0.166666666666667)]
    
    for NG in range(3):
        eta, ksi = A_HAMMER[NG]
        
        bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3 = cdkderi3(
            px2, py2, px3, py3, px, py, pxy, pyy, ksi, eta)
            
        kxx = np.zeros(n)
        kyy = np.zeros(n)
        kxy = np.zeros(n)
        cdkcurv3(bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3, vlz, rlx, rly, kxx, kyy, kxy)
        
        dexx_bend = kxx * dt
        deyy_bend = kyy * dt
        dexy_bend = kxy * dt
        
        vol_gp = vol0 / 3.0
        force_pg = np.zeros((n, 3))
        mom_pg = np.zeros((n, 3))
        
        for k, (gp, gw) in enumerate(st["zw"]):
            z = gp * thick
            # Strain increment at layer k
            deps = np.stack([
                exx + z * dexx_bend,
                eyy + z * deyy_bend,
                exy + z * dexy_bend
            ], axis=1)
            
            st_sig_k = st["sig"][:, k, :].copy()
            st_epsp_k = st["epsp"][:, k].copy()
            
            for sl, mat, prop in st["slices"]:
                materials.shell_update(
                    mat, deps[sl], st_sig_k[sl], st_epsp_k[sl],
                    None, None, dt)
                    
            st["sig"][:, k, :] = st_sig_k
            st["epsp"][:, k] = st_epsp_k
            
            dz_vol = gw * vol_gp
            force_pg += st_sig_k * dz_vol[:, None]
            mom_pg += st_sig_k * (z * dz_vol)[:, None]
            
            st["eint"] += np.sum((st_sig_k - 0.5 * deps) * deps, axis=1) * dz_vol
            
        cdkfint3(vol_gp, thick, force_pg, mom_pg, px2, py2, px3, py3,
                 bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3,
                 f11, f12, f13, f21, f22, f23, f32, f33,
                 m11, m12, m13, m21, m22, m23)
                 
    cdkfcum3(px2, py2, px3, py3, e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z,
             f11, f12, f13, f21, f22, f23, f31, f32, f33,
             m11, m12, m13, m21, m22, m23, m31, m32, m33)
             
    np.add.at(fint[:, 0], conn[:, 0], f11)
    np.add.at(fint[:, 1], conn[:, 0], f21)
    np.add.at(fint[:, 2], conn[:, 0], f31)
    np.add.at(mint[:, 0], conn[:, 0], m11)
    np.add.at(mint[:, 1], conn[:, 0], m21)
    np.add.at(mint[:, 2], conn[:, 0], m31)
    
    np.add.at(fint[:, 0], conn[:, 1], f12)
    np.add.at(fint[:, 1], conn[:, 1], f22)
    np.add.at(fint[:, 2], conn[:, 1], f32)
    np.add.at(mint[:, 0], conn[:, 1], m12)
    np.add.at(mint[:, 1], conn[:, 1], m22)
    np.add.at(mint[:, 2], conn[:, 1], m32)

    np.add.at(fint[:, 0], conn[:, 2], f13)
    np.add.at(fint[:, 1], conn[:, 2], f23)
    np.add.at(fint[:, 2], conn[:, 2], f33)
    np.add.at(mint[:, 0], conn[:, 2], m13)
    np.add.at(mint[:, 1], conn[:, 2], m23)
    np.add.at(mint[:, 2], conn[:, 2], m33)
    
    return st["dtfac"]

def consistent_mass(group):
    """Consistent mass matrix for implicit assembly (NOT IMPLEMENTED for DKT18)."""
    raise NotImplementedError("DKT18 implicit mass not implemented.")

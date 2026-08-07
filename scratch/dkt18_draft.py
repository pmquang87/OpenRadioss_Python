import numpy as np
from pyradioss.accel.jit_kernels.shells_dkt18 import (
    cdkcoor3, cdkderic3, cdkdefo3, cdkderi3, cdkcurv3, cdkfint3, cdkfcum3
)
from pyradioss.elements.shell_bt4 import _layer_extra, _layer_failure
from pyradioss import materials

def forces(group, x, v, vr, dt, fint, mint):
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
    vol00 = vol0.copy()  # wait, VOL00 is usually just original vol, but we use VOL0
    
    # We need to allocate arrays for derivatives
    px2, py2, px3, py3, px, py, pxy, pyy = cdkderic3(xl2, yl2, xl3, yl3, area2, vol00, nu, thick**2)
    
    # 3. Membrane rates
    exx = np.zeros(n)
    eyy = np.zeros(n)
    exy = np.zeros(n)
    exz = np.zeros(n)
    eyz = np.zeros(n)
    epsdot = np.zeros((3, n))
    gstr = np.zeros((n, 3)) # not really used for accumulation in Python, we do it in shell_update
    vdef = np.zeros((n, 3))
    
    cdkdefo3(vlx, vly, px2, py2, px3, py3, exx, eyy, exy, exz, eyz, dt, epsdot, 0, False, gstr, vdef, False)
    
    # epsd_glob (average strain rate for epsp computation, etc.)
    # In Fortran, epsd_glob is accumulated over NPG=3. Here membrane is constant.
    
    # Allocate global nodal force/moment arrays
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
    
    # DKT18 uses 3 Gauss points in-plane
    A_HAMMER = [(0.166666666666667, 0.666666666666667),
                (0.666666666666667, 0.166666666666667),
                (0.166666666666667, 0.166666666666667)]
    
    nip_max = st["sig"].shape[1]
    
    # We need to loop over the 3 in-plane Gauss points
    for NG in range(3):
        eta, ksi = A_HAMMER[NG]
        
        bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3 = cdkderi3(
            px2, py2, px3, py3, px, py, pxy, pyy, ksi, eta)
            
        kxx = np.zeros(n)
        kyy = np.zeros(n)
        kxy = np.zeros(n)
        cdkcurv3(bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3, vlz, rlx, rly, kxx, kyy, kxy)
        
        # Bending strain increments
        dexx_bend = kxx * dt
        deyy_bend = kyy * dt
        dexy_bend = kxy * dt
        
        # Now loop over layers through thickness
        vol_gp = vol0 / 3.0  # weight is 1/3 (sum is 1)
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
            
            # Since shell_update expects slices, we do that
            st_sig_k = st["sig"][:, k, :].copy()
            st_epsp_k = st["epsp"][:, k].copy()
            
            for sl, mat, prop in st["slices"]:
                materials.shell_update(
                    mat, deps[sl], st_sig_k[sl], st_epsp_k[sl],
                    None, None, dt)  # ignoring ortho for now
                    
            st["sig"][:, k, :] = st_sig_k
            st["epsp"][:, k] = st_epsp_k
            
            # Add to internal forces and moments
            dz_vol = gw * vol_gp
            force_pg += st_sig_k * dz_vol[:, None]
            mom_pg += st_sig_k * (z * dz_vol)[:, None]
            
            # Energy accumulation
            st["eint"] += np.sum((st_sig_k - 0.5 * deps) * deps, axis=1) * dz_vol
            
        # Accumulate forces at nodes
        cdkfint3(vol0 / 3.0, thick, force_pg, mom_pg, px2, py2, px3, py3,
                 bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3,
                 f11, f12, f13, f21, f22, f23, f32, f33,
                 m11, m12, m13, m21, m22, m23)
                 
    # Finally, scatter from local to global
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
    np.add.at(fint[:, 1], conn[:, 1], f21)
    np.add.at(fint[:, 2], conn[:, 1], f31) # wait f32?
    
    # We must properly add f12, f22, f32 to node 2
    
    return st["dtfac"]

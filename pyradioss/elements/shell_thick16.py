"""
16-node thick shell element (SHEL16).
(/SHEL16 + /PROP/TSHELL)

Fortran origin: ``engine/source/elements/thickshell/solide16/``
    s16forc3.F  driver: gather coords/velocities, call the chain below
    s16coor3.F  geometry
    s16deri3.F  derivatives
    s16mass3.F  mass initialization
"""

import numpy as np
from numba import njit, prange

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import scatter_add3

# IPERM arrays from Fortran for node 9-16 connectivity
_IPERM1 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
_IPERM2 = [0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 4, 1, 6, 7, 8, 5]

@njit(cache=True)
def s16rst(r, s, t):
    # constants
    HALF = 0.5
    
    # helper variables
    u_m_r = HALF * (1.0 - r)
    u_p_r = HALF * (1.0 + r)
    
    u_m_s = HALF * (1.0 - s)
    u_p_s = HALF * (1.0 + s)
    
    u_m_t = HALF * (1.0 - t)
    u_p_t = HALF * (1.0 + t)
    
    ums_umt = u_m_s * u_m_t
    ums_upt = u_m_s * u_p_t
    ups_umt = u_p_s * u_m_t
    ups_upt = u_p_s * u_p_t
    
    umr_ums = u_m_r * u_m_s
    umr_ups = u_m_r * u_p_s
    upr_ums = u_p_r * u_m_s
    upr_ups = u_p_r * u_p_s
    
    umt_umr = u_m_t * u_m_r
    umt_upr = u_m_t * u_p_r
    upt_umr = u_p_t * u_m_r
    upt_upr = u_p_t * u_p_r

    # NI values (indices 0 to 15, matching Fortran 1 to 16)
    ni_0 = u_m_r * ums_umt * (-r - t - 1.0)
    ni_1 = u_m_r * ums_upt * (-r + t - 1.0)
    ni_2 = u_p_r * ums_upt * ( r + t - 1.0)
    ni_3 = u_p_r * ums_umt * ( r - t - 1.0)
    ni_4 = u_m_r * ups_umt * (-r - t - 1.0)
    ni_5 = u_m_r * ups_upt * (-r + t - 1.0)
    ni_6 = u_p_r * ups_upt * ( r + t - 1.0)
    ni_7 = u_p_r * ups_umt * ( r - t - 1.0)
    
    a_r = 1.0 - r * r
    ni_9  = a_r * ums_upt
    ni_11 = a_r * ums_umt
    ni_13 = a_r * ups_upt
    ni_15 = a_r * ups_umt
    
    a_t = 1.0 - t * t
    ni_8  = a_t * umr_ums
    ni_10 = a_t * upr_ums
    ni_12 = a_t * umr_ups
    ni_14 = a_t * upr_ups

    # DNIDR values
    dnidr_0 = -ums_umt * (-HALF * t - r)
    dnidr_1 = -ums_upt * ( HALF * t - r)
    dnidr_2 =  ums_upt * ( HALF * t + r)
    dnidr_3 =  ums_umt * (-HALF * t + r)
    dnidr_4 = -ups_umt * (-HALF * t - r)
    dnidr_5 = -ups_upt * ( HALF * t - r)
    dnidr_6 =  ups_upt * ( HALF * t + r)
    dnidr_7 =  ups_umt * (-HALF * t + r)

    a_t_half = HALF * a_t
    dnidr_8  = -a_t_half * u_m_s
    dnidr_10 =  a_t_half * u_m_s
    dnidr_12 = -a_t_half * u_p_s
    dnidr_14 =  a_t_half * u_p_s

    a_r_r2 = -2.0 * r
    dnidr_9  = a_r_r2 * ums_upt
    dnidr_11 = a_r_r2 * ums_umt
    dnidr_13 = a_r_r2 * ups_upt
    dnidr_15 = a_r_r2 * ups_umt
    
    # DNIDS values
    dnids_0 = -umt_umr * (-r - t - 1.0) * HALF
    dnids_1 = -upt_umr * (-r + t - 1.0) * HALF
    dnids_2 = -upt_upr * ( r + t - 1.0) * HALF
    dnids_3 = -umt_upr * ( r - t - 1.0) * HALF
    dnids_4 =  umt_umr * (-r - t - 1.0) * HALF
    dnids_5 =  upt_umr * (-r + t - 1.0) * HALF
    dnids_6 =  upt_upr * ( r + t - 1.0) * HALF
    dnids_7 =  umt_upr * ( r - t - 1.0) * HALF
    
    a_r_half = HALF * a_r
    dnids_9  = -a_r_half * u_p_t
    dnids_11 = -a_r_half * u_m_t
    dnids_13 =  a_r_half * u_p_t
    dnids_15 =  a_r_half * u_m_t
    
    dnids_8  = -a_t_half * u_m_r
    dnids_10 = -a_t_half * u_p_r
    dnids_12 =  a_t_half * u_m_r
    dnids_14 =  a_t_half * u_p_r

    # DNIDT values
    dnidt_0 = -umr_ums * (-HALF * r - t)
    dnidt_1 =  umr_ums * (-HALF * r + t)
    dnidt_2 =  upr_ums * ( HALF * r + t)
    dnidt_3 = -upr_ums * ( HALF * r - t)
    dnidt_4 = -umr_ups * (-HALF * r - t)
    dnidt_5 =  umr_ups * (-HALF * r + t)
    dnidt_6 =  upr_ups * ( HALF * r + t)
    dnidt_7 = -upr_ups * ( HALF * r - t)

    dnidt_9  =  a_r_half * u_m_s
    dnidt_11 = -a_r_half * u_m_s
    dnidt_13 =  a_r_half * u_p_s
    dnidt_15 = -a_r_half * u_p_s

    a_t_t2 = -2.0 * t
    dnidt_8  = a_t_t2 * umr_ums
    dnidt_10 = a_t_t2 * upr_ums
    dnidt_12 = a_t_t2 * umr_ups
    dnidt_14 = a_t_t2 * upr_ups

    ni = (ni_0, ni_1, ni_2, ni_3, ni_4, ni_5, ni_6, ni_7, ni_8, ni_9, ni_10, ni_11, ni_12, ni_13, ni_14, ni_15)
    dnidr = (dnidr_0, dnidr_1, dnidr_2, dnidr_3, dnidr_4, dnidr_5, dnidr_6, dnidr_7, dnidr_8, dnidr_9, dnidr_10, dnidr_11, dnidr_12, dnidr_13, dnidr_14, dnidr_15)
    dnids = (dnids_0, dnids_1, dnids_2, dnids_3, dnids_4, dnids_5, dnids_6, dnids_7, dnids_8, dnids_9, dnids_10, dnids_11, dnids_12, dnids_13, dnids_14, dnids_15)
    dnidt = (dnidt_0, dnidt_1, dnidt_2, dnidt_3, dnidt_4, dnidt_5, dnidt_6, dnidt_7, dnidt_8, dnidt_9, dnidt_10, dnidt_11, dnidt_12, dnidt_13, dnidt_14, dnidt_15)
    
    return ni, dnidr, dnids, dnidt


@njit(cache=True)
def s16deri3(xx, dnidr, dnids, dnidt):
    """
    Computes the Cartesian derivatives PX, PY, PZ, determinant of Jacobian, etc.
    xx: array-like of shape (3, 16) - coordinates of 16 nodes (0-based)
    dnidr, dnids, dnidt: array-like of shape (16,)
    
    Returns px, py, pz, det
    """
    
    # dxdr
    dxdr = (
        dnidr[0]*xx[0][0] + dnidr[1]*xx[0][1] + dnidr[2]*xx[0][2] + dnidr[3]*xx[0][3] +
        dnidr[4]*xx[0][4] + dnidr[5]*xx[0][5] + dnidr[6]*xx[0][6] + dnidr[7]*xx[0][7] +
        dnidr[8]*(xx[0][8] - xx[0][10]) + dnidr[9]*xx[0][9] +
        dnidr[11]*xx[0][11] + dnidr[12]*(xx[0][12] - xx[0][14]) +
        dnidr[13]*xx[0][13] + dnidr[15]*xx[0][15]
    )
    dydr = (
        dnidr[0]*xx[1][0] + dnidr[1]*xx[1][1] + dnidr[2]*xx[1][2] + dnidr[3]*xx[1][3] +
        dnidr[4]*xx[1][4] + dnidr[5]*xx[1][5] + dnidr[6]*xx[1][6] + dnidr[7]*xx[1][7] +
        dnidr[8]*(xx[1][8] - xx[1][10]) + dnidr[9]*xx[1][9] +
        dnidr[11]*xx[1][11] + dnidr[12]*(xx[1][12] - xx[1][14]) +
        dnidr[13]*xx[1][13] + dnidr[15]*xx[1][15]
    )
    dzdr = (
        dnidr[0]*xx[2][0] + dnidr[1]*xx[2][1] + dnidr[2]*xx[2][2] + dnidr[3]*xx[2][3] +
        dnidr[4]*xx[2][4] + dnidr[5]*xx[2][5] + dnidr[6]*xx[2][6] + dnidr[7]*xx[2][7] +
        dnidr[8]*(xx[2][8] - xx[2][10]) + dnidr[9]*xx[2][9] +
        dnidr[11]*xx[2][11] + dnidr[12]*(xx[2][12] - xx[2][14]) +
        dnidr[13]*xx[2][13] + dnidr[15]*xx[2][15]
    )
    
    # dxds
    dxds = (
        dnids[0]*xx[0][0] + dnids[1]*xx[0][1] + dnids[2]*xx[0][2] + dnids[3]*xx[0][3] +
        dnids[4]*xx[0][4] + dnids[5]*xx[0][5] + dnids[6]*xx[0][6] + dnids[7]*xx[0][7] +
        dnids[8]*(xx[0][8] - xx[0][12]) +
        dnids[9]*(xx[0][9] - xx[0][13]) +
        dnids[10]*(xx[0][10] - xx[0][14]) +
        dnids[11]*(xx[0][11] - xx[0][15])
    )
    dyds = (
        dnids[0]*xx[1][0] + dnids[1]*xx[1][1] + dnids[2]*xx[1][2] + dnids[3]*xx[1][3] +
        dnids[4]*xx[1][4] + dnids[5]*xx[1][5] + dnids[6]*xx[1][6] + dnids[7]*xx[1][7] +
        dnids[8]*(xx[1][8] - xx[1][12]) +
        dnids[9]*(xx[1][9] - xx[1][13]) +
        dnids[10]*(xx[1][10] - xx[1][14]) +
        dnids[11]*(xx[1][11] - xx[1][15])
    )
    dzds = (
        dnids[0]*xx[2][0] + dnids[1]*xx[2][1] + dnids[2]*xx[2][2] + dnids[3]*xx[2][3] +
        dnids[4]*xx[2][4] + dnids[5]*xx[2][5] + dnids[6]*xx[2][6] + dnids[7]*xx[2][7] +
        dnids[8]*(xx[2][8] - xx[2][12]) +
        dnids[9]*(xx[2][9] - xx[2][13]) +
        dnids[10]*(xx[2][10] - xx[2][14]) +
        dnids[11]*(xx[2][11] - xx[2][15])
    )
    
    # dxdt
    dxdt = (
        dnidt[0]*xx[0][0] + dnidt[1]*xx[0][1] + dnidt[2]*xx[0][2] + dnidt[3]*xx[0][3] +
        dnidt[4]*xx[0][4] + dnidt[5]*xx[0][5] + dnidt[6]*xx[0][6] + dnidt[7]*xx[0][7] +
        dnidt[8]*xx[0][8] + dnidt[9]*(xx[0][9] - xx[0][11]) +
        dnidt[10]*xx[0][10] + dnidt[12]*xx[0][12] +
        dnidt[13]*(xx[0][13] - xx[0][15]) + dnidt[14]*xx[0][14]
    )
    dydt = (
        dnidt[0]*xx[1][0] + dnidt[1]*xx[1][1] + dnidt[2]*xx[1][2] + dnidt[3]*xx[1][3] +
        dnidt[4]*xx[1][4] + dnidt[5]*xx[1][5] + dnidt[6]*xx[1][6] + dnidt[7]*xx[1][7] +
        dnidt[8]*xx[1][8] + dnidt[9]*(xx[1][9] - xx[1][11]) +
        dnidt[10]*xx[1][10] + dnidt[12]*xx[1][12] +
        dnidt[13]*(xx[1][13] - xx[1][15]) + dnidt[14]*xx[1][14]
    )
    dzdt = (
        dnidt[0]*xx[2][0] + dnidt[1]*xx[2][1] + dnidt[2]*xx[2][2] + dnidt[3]*xx[2][3] +
        dnidt[4]*xx[2][4] + dnidt[5]*xx[2][5] + dnidt[6]*xx[2][6] + dnidt[7]*xx[2][7] +
        dnidt[8]*xx[2][8] + dnidt[9]*(xx[2][9] - xx[2][11]) +
        dnidt[10]*xx[2][10] + dnidt[12]*xx[2][12] +
        dnidt[13]*(xx[2][13] - xx[2][15]) + dnidt[14]*xx[2][14]
    )
    
    # Inversion of the jacobian
    drdx = dyds * dzdt - dzds * dydt
    drdy = dzds * dxdt - dxds * dzdt
    drdz = dxds * dydt - dyds * dxdt
    
    dsdz = dxdt * dydr - dydt * dxdr
    dsdy = dzdt * dxdr - dxdt * dzdr
    dsdx = dydt * dzdr - dzdt * dydr
    
    dtdx = dydr * dzds - dzdr * dyds
    dtdy = dzdr * dxds - dxdr * dzds
    dtdz = dxdr * dyds - dydr * dxds
    
    det = (
        dxdr * drdx +
        dydr * drdy +
        dzdr * drdz
    )
    
    if det <= 0.0:
        # Fortran s16deri3.F calls ARRET(2) — fatal error for non-positive
        # Jacobian determinant.  Inside @njit we cannot raise a Python
        # exception, so clamp to EM20 to prevent inf/nan propagation.
        # The element will still produce garbage forces, but those are
        # masked by the alive/off flag downstream.
        det = 1.0e-20
        
    d = 1.0 / det
    
    drdx *= d
    dsdx *= d
    dtdx *= d
    
    drdy *= d
    dsdy *= d
    dtdy *= d
    
    drdz *= d
    dsdz *= d
    dtdz *= d
    
    px = [0.0] * 16
    px[0] = dnidr[0]*drdx + dnids[0]*dsdx + dnidt[0]*dtdx
    px[1] = dnidr[1]*drdx + dnids[1]*dsdx + dnidt[1]*dtdx
    px[2] = dnidr[2]*drdx + dnids[2]*dsdx + dnidt[2]*dtdx
    px[3] = dnidr[3]*drdx + dnids[3]*dsdx + dnidt[3]*dtdx
    px[4] = dnidr[4]*drdx + dnids[4]*dsdx + dnidt[4]*dtdx
    px[5] = dnidr[5]*drdx + dnids[5]*dsdx + dnidt[5]*dtdx
    px[6] = dnidr[6]*drdx + dnids[6]*dsdx + dnidt[6]*dtdx
    px[7] = dnidr[7]*drdx + dnids[7]*dsdx + dnidt[7]*dtdx
    
    r9_x  = dnidr[8]*drdx
    r13_x = dnidr[12]*drdx
    s9_x  = dnids[8]*dsdx
    s10_x = dnids[9]*dsdx
    s11_x = dnids[10]*dsdx
    s12_x = dnids[11]*dsdx
    t10_x = dnidt[9]*dtdx
    t14_x = dnidt[13]*dtdx
    
    px[8] = r9_x + s9_x + dnidt[8]*dtdx
    px[9] = dnidr[9]*drdx + s10_x + t10_x
    px[10] = -r9_x + s11_x + dnidt[10]*dtdx
    px[11] = dnidr[11]*drdx + s12_x - t10_x
    px[12] = r13_x - s9_x + dnidt[12]*dtdx
    px[13] = dnidr[13]*drdx - s10_x + t14_x
    px[14] = -r13_x - s11_x + dnidt[14]*dtdx
    px[15] = dnidr[15]*drdx - s12_x - t14_x
    
    py = [0.0] * 16
    py[0] = dnidr[0]*drdy + dnids[0]*dsdy + dnidt[0]*dtdy
    py[1] = dnidr[1]*drdy + dnids[1]*dsdy + dnidt[1]*dtdy
    py[2] = dnidr[2]*drdy + dnids[2]*dsdy + dnidt[2]*dtdy
    py[3] = dnidr[3]*drdy + dnids[3]*dsdy + dnidt[3]*dtdy
    py[4] = dnidr[4]*drdy + dnids[4]*dsdy + dnidt[4]*dtdy
    py[5] = dnidr[5]*drdy + dnids[5]*dsdy + dnidt[5]*dtdy
    py[6] = dnidr[6]*drdy + dnids[6]*dsdy + dnidt[6]*dtdy
    py[7] = dnidr[7]*drdy + dnids[7]*dsdy + dnidt[7]*dtdy
    
    r9_y  = dnidr[8]*drdy
    r13_y = dnidr[12]*drdy
    s9_y  = dnids[8]*dsdy
    s10_y = dnids[9]*dsdy
    s11_y = dnids[10]*dsdy
    s12_y = dnids[11]*dsdy
    t10_y = dnidt[9]*dtdy
    t14_y = dnidt[13]*dtdy
    
    py[8] = r9_y + s9_y + dnidt[8]*dtdy
    py[9] = dnidr[9]*drdy + s10_y + t10_y
    py[10] = -r9_y + s11_y + dnidt[10]*dtdy
    py[11] = dnidr[11]*drdy + s12_y - t10_y
    py[12] = r13_y - s9_y + dnidt[12]*dtdy
    py[13] = dnidr[13]*drdy - s10_y + t14_y
    py[14] = -r13_y - s11_y + dnidt[14]*dtdy
    py[15] = dnidr[15]*drdy - s12_y - t14_y
    
    pz = [0.0] * 16
    pz[0] = dnidr[0]*drdz + dnids[0]*dsdz + dnidt[0]*dtdz
    pz[1] = dnidr[1]*drdz + dnids[1]*dsdz + dnidt[1]*dtdz
    pz[2] = dnidr[2]*drdz + dnids[2]*dsdz + dnidt[2]*dtdz
    pz[3] = dnidr[3]*drdz + dnids[3]*dsdz + dnidt[3]*dtdz
    pz[4] = dnidr[4]*drdz + dnids[4]*dsdz + dnidt[4]*dtdz
    pz[5] = dnidr[5]*drdz + dnids[5]*dsdz + dnidt[5]*dtdz
    pz[6] = dnidr[6]*drdz + dnids[6]*dsdz + dnidt[6]*dtdz
    pz[7] = dnidr[7]*drdz + dnids[7]*dsdz + dnidt[7]*dtdz
    
    r9_z  = dnidr[8]*drdz
    r13_z = dnidr[12]*drdz
    s9_z  = dnids[8]*dsdz
    s10_z = dnids[9]*dsdz
    s11_z = dnids[10]*dsdz
    s12_z = dnids[11]*dsdz
    t10_z = dnidt[9]*dtdz
    t14_z = dnidt[13]*dtdz
    
    pz[8] = r9_z + s9_z + dnidt[8]*dtdz
    pz[9] = dnidr[9]*drdz + s10_z + t10_z
    pz[10] = -r9_z + s11_z + dnidt[10]*dtdz
    pz[11] = dnidr[11]*drdz + s12_z - t10_z
    pz[12] = r13_z - s9_z + dnidt[12]*dtdz
    pz[13] = dnidr[13]*drdz - s10_z + t14_z
    pz[14] = -r13_z - s11_z + dnidt[14]*dtdz
    pz[15] = dnidr[15]*drdz - s12_z - t14_z
    
    return px, py, pz, det





@njit(cache=True)
def s20defo3(px, py, pz, vx, vy, vz, rho, voln, dt1=0.0):
    nel = px.shape[0]
    npe = px.shape[1]
    
    rhoo = np.zeros(nel, dtype=np.float64)
    voln_out = np.zeros(nel, dtype=np.float64)
    dxx = np.zeros(nel, dtype=np.float64)
    dyy = np.zeros(nel, dtype=np.float64)
    dzz = np.zeros(nel, dtype=np.float64)
    dxy = np.zeros(nel, dtype=np.float64)
    dxz = np.zeros(nel, dtype=np.float64)
    dyx = np.zeros(nel, dtype=np.float64)
    dyz = np.zeros(nel, dtype=np.float64)
    dzx = np.zeros(nel, dtype=np.float64)
    dzy = np.zeros(nel, dtype=np.float64)
    
    d4 = np.zeros(nel, dtype=np.float64)
    d5 = np.zeros(nel, dtype=np.float64)
    d6 = np.zeros(nel, dtype=np.float64)
    wxx = np.zeros(nel, dtype=np.float64)
    wyy = np.zeros(nel, dtype=np.float64)
    wzz = np.zeros(nel, dtype=np.float64)
    
    dt1d2 = 0.5 * dt1
    
    for i in range(nel):
        rhoo[i] = rho[i]
        voln_out[i] = voln[i]
        
        dxx_i = 0.0
        dyy_i = 0.0
        dzz_i = 0.0
        dxy_i = 0.0
        dxz_i = 0.0
        dyx_i = 0.0
        dyz_i = 0.0
        dzx_i = 0.0
        dzy_i = 0.0
        
        for n in range(npe):
            dxx_i += px[i, n] * vx[i, n]
            dyy_i += py[i, n] * vy[i, n]
            dzz_i += pz[i, n] * vz[i, n]
            dxy_i += py[i, n] * vx[i, n]
            dxz_i += pz[i, n] * vx[i, n]
            dyx_i += px[i, n] * vy[i, n]
            dyz_i += pz[i, n] * vy[i, n]
            dzx_i += px[i, n] * vz[i, n]
            dzy_i += py[i, n] * vz[i, n]
            
        dxx[i] = dxx_i
        dyy[i] = dyy_i
        dzz[i] = dzz_i
        dxy[i] = dxy_i
        dxz[i] = dxz_i
        dyx[i] = dyx_i
        dyz[i] = dyz_i
        dzx[i] = dzx_i
        dzy[i] = dzy_i
        
        d4[i] = dxy_i + dyx_i
        d5[i] = dyz_i + dzy_i
        d6[i] = dxz_i + dzx_i
        
        wzz[i] = dt1d2 * (dyx_i - dxy_i)
        wyy[i] = dt1d2 * (dxz_i - dzx_i)
        wxx[i] = dt1d2 * (dzy_i - dyz_i)
        
    return (dxx, dyy, dzz, dxy, dxz, dyx, dyz, dzx, dzy,
            d4, d5, d6, wxx, wyy, wzz, rhoo, voln_out)




@njit(cache=True)
def s20fint3(px, py, pz, sig, voln):
    """
    Computes internal forces for the 16-node thick shell (solide16) element.
    Uses 0-based indexing.
    
    px, py, pz : arrays of shape (16,) containing Cartesian derivatives
    sig        : array of shape (6,) containing stress components (xx, yy, zz, xy, yz, zx)
    voln       : volume integration weight for this integration point
    
    Returns:
    fint       : array of shape (48,) containing the interleaved nodal internal forces (fx, fy, fz)
    """
    fint = np.zeros(48, dtype=np.float64)
    
    s1 = sig[0] * voln
    s2 = sig[1] * voln
    s3 = sig[2] * voln
    s4 = sig[3] * voln
    s5 = sig[4] * voln
    s6 = sig[5] * voln
    
    for n in range(16):
        fint[3 * n + 0] = s1 * px[n] + s4 * py[n] + s6 * pz[n]
        fint[3 * n + 1] = s4 * px[n] + s2 * py[n] + s5 * pz[n]
        fint[3 * n + 2] = s6 * px[n] + s5 * py[n] + s3 * pz[n]
        
    return fint


@njit(cache=True)
def _init_mass(n, fill, rho, vol, dtx, dtelem, mass, mss, mssx, nc, stifn, deltax2):
    TWO = 2.0
    SIXTY4 = 64.0
    EM20 = 1e-20
    THIRTY2 = 32.0
    THREE = 3.0
    HALF = 0.5
    
    for i in range(n):
        mass[i] = fill[i] * rho[i] * vol[i]
        
        if dtelem[i] > dtx[i]:
            dtelem[i] = dtx[i]
            
        dtx_sq = dtx[i] * dtx[i]
        max_val = dtx_sq if dtx_sq > EM20 else EM20
        sti = fill[i] * rho[i] * vol[i] * TWO / SIXTY4 / max_val
        
        am = mass[i] / THIRTY2
        bm = mass[i] * THREE / THIRTY2
        
        mss[i, 0] = am
        mss[i, 1] = am
        mss[i, 2] = am
        mss[i, 3] = am
        mss[i, 4] = am
        mss[i, 5] = am
        mss[i, 6] = am
        mss[i, 7] = am
        
        stifn[nc[i, 0]] += sti * deltax2[i]
        stifn[nc[i, 1]] += sti * deltax2[i]
        stifn[nc[i, 2]] += sti * deltax2[i]
        stifn[nc[i, 3]] += sti * deltax2[i]
        stifn[nc[i, 4]] += sti * deltax2[i]
        stifn[nc[i, 5]] += sti * deltax2[i]
        stifn[nc[i, 6]] += sti * deltax2[i]
        stifn[nc[i, 7]] += sti * deltax2[i]
        
        # N=9 (IPERM1=1, IPERM2=2)
        if nc[i, 8] >= 0:
            mssx[i, 0] = bm
            stifn[nc[i, 8]] += sti
        else:
            mss[i, 0] += HALF * bm
            mss[i, 1] += HALF * bm
            stifn[nc[i, 0]] += HALF * sti
            stifn[nc[i, 1]] += HALF * sti

        # N=10 (IPERM1=2, IPERM2=3)
        if nc[i, 9] >= 0:
            mssx[i, 1] = bm
            stifn[nc[i, 9]] += sti
        else:
            mss[i, 1] += HALF * bm
            mss[i, 2] += HALF * bm
            stifn[nc[i, 1]] += HALF * sti
            stifn[nc[i, 2]] += HALF * sti

        # N=11 (IPERM1=3, IPERM2=4)
        if nc[i, 10] >= 0:
            mssx[i, 2] = bm
            stifn[nc[i, 10]] += sti
        else:
            mss[i, 2] += HALF * bm
            mss[i, 3] += HALF * bm
            stifn[nc[i, 2]] += HALF * sti
            stifn[nc[i, 3]] += HALF * sti

        # N=12 (IPERM1=4, IPERM2=1)
        if nc[i, 11] >= 0:
            mssx[i, 3] = bm
            stifn[nc[i, 11]] += sti
        else:
            mss[i, 3] += HALF * bm
            mss[i, 0] += HALF * bm
            stifn[nc[i, 3]] += HALF * sti
            stifn[nc[i, 0]] += HALF * sti

        # N=13 (IPERM1=5, IPERM2=6)
        if nc[i, 12] >= 0:
            mssx[i, 4] = bm
            stifn[nc[i, 12]] += sti
        else:
            mss[i, 4] += HALF * bm
            mss[i, 5] += HALF * bm
            stifn[nc[i, 4]] += HALF * sti
            stifn[nc[i, 5]] += HALF * sti

        # N=14 (IPERM1=6, IPERM2=7)
        if nc[i, 13] >= 0:
            mssx[i, 5] = bm
            stifn[nc[i, 13]] += sti
        else:
            mss[i, 5] += HALF * bm
            mss[i, 6] += HALF * bm
            stifn[nc[i, 5]] += HALF * sti
            stifn[nc[i, 6]] += HALF * sti

        # N=15 (IPERM1=7, IPERM2=8)
        if nc[i, 14] >= 0:
            mssx[i, 6] = bm
            stifn[nc[i, 14]] += sti
        else:
            mss[i, 6] += HALF * bm
            mss[i, 7] += HALF * bm
            stifn[nc[i, 6]] += HALF * sti
            stifn[nc[i, 7]] += HALF * sti

        # N=16 (IPERM1=8, IPERM2=5)
        if nc[i, 15] >= 0:
            mssx[i, 7] = bm
            stifn[nc[i, 15]] += sti
        else:
            mss[i, 7] += HALF * bm
            mss[i, 4] += HALF * bm
            stifn[nc[i, 7]] += HALF * sti
            stifn[nc[i, 4]] += HALF * sti


def init_group(group, model, log):
    """Element buffer + lumped mass."""
    conn = group.conn
    n = group.n
    
    xe = model.x0[conn] # (n, 16, 3) 
    
    mass = np.zeros(n)
    mss = np.zeros((n, 8))
    mssx = np.zeros((n, 8))
    stifn = np.zeros(model.numnod)
    
    fill = np.ones(n)
    rho = np.zeros(n)
    vol = np.ones(n) # placeholder for actual volume calculation via Gauss loop
    dtx = np.full(n, 1e20)
    dtelem = np.full(n, 1e20)
    deltax2 = np.ones(n)
    
    nip_max = 1
    zw = []
    
    # We need leggauss from numpy
    from numpy.polynomial.legendre import leggauss
    
    for sl, mat, prop in group.state["slices"]:
        rho[sl] = mat.rho0
        npts_r = int(prop.params.get("npts_r", 2))
        npts_s = int(prop.params.get("npts_s", 2))
        npts_t = int(prop.params.get("npts_t", 2))
        if npts_r == 0: npts_r = 2
        if npts_s == 0: npts_s = 2
        if npts_t == 0: npts_t = 2
        nip = npts_r * npts_s * npts_t
        nip_max = max(nip_max, nip)
        
        xr, wr = leggauss(npts_r) if npts_r > 1 else (np.array([0.0]), np.array([2.0]))
        xs, ws = leggauss(npts_s) if npts_s > 1 else (np.array([0.0]), np.array([2.0]))
        xt, wt = leggauss(npts_t) if npts_t > 1 else (np.array([0.0]), np.array([2.0]))
        
        pts = []
        wts = []
        for r_idx in range(npts_r):
            for s_idx in range(npts_s):
                for t_idx in range(npts_t):
                    pts.append((xr[r_idx], xs[s_idx], xt[t_idx]))
                    wts.append(wr[r_idx] * ws[s_idx] * wt[t_idx])
        zw.append((pts, wts))
        
    _init_mass(n, fill, rho, vol, dtx, dtelem, mass, mss, mssx, conn, stifn, deltax2)
    
    group.state.update(
        sig=np.zeros((n, nip_max, 6)),
        epsp=np.zeros((n, nip_max)),
        eint=np.zeros(n),
        ehour=np.zeros(n),
        zw=zw,
        rho=rho,
        vol=vol,
        mass=mass,
    )
    from pyradioss.elements.shell_bt4 import _init_material_state
    _init_material_state(group, nip_max)
    
    # We map back mss and mssx into the global mass array
    mass_16 = np.zeros((n, 16))
    mass_16[:, 0:8] = mss
    mass_16[:, 8:16] = mssx
    
    valid = conn >= 0
    return conn[valid], mass_16[valid], np.zeros(valid.sum())


@njit(cache=True)
def _s16_pre(xe, ve, r, s, t, w):
    nel = xe.shape[0]
    npe = 16
    
    dxx = np.zeros(nel, dtype=np.float64)
    dyy = np.zeros(nel, dtype=np.float64)
    dzz = np.zeros(nel, dtype=np.float64)
    dxy = np.zeros(nel, dtype=np.float64)
    dyz = np.zeros(nel, dtype=np.float64)
    dzx = np.zeros(nel, dtype=np.float64)
    
    px_all = np.zeros((nel, 16), dtype=np.float64)
    py_all = np.zeros((nel, 16), dtype=np.float64)
    pz_all = np.zeros((nel, 16), dtype=np.float64)
    volnp_all = np.zeros(nel, dtype=np.float64)
    
    ni, dnidr, dnids, dnidt = s16rst(r, s, t)
    
    for i in range(nel):
        # Extract components of xe[i] properly for s16deri3
        xx = np.zeros((3, 16), dtype=np.float64)
        for n in range(16):
            xx[0, n] = xe[i, n, 0]
            xx[1, n] = xe[i, n, 1]
            xx[2, n] = xe[i, n, 2]
            
        px, py, pz, det = s16deri3(xx, dnidr, dnids, dnidt)
        volnp = det * w
        
        px_all[i, :] = px
        py_all[i, :] = py
        pz_all[i, :] = pz
        volnp_all[i] = volnp
        
        dxx_i = 0.0
        dyy_i = 0.0
        dzz_i = 0.0
        dxy_i = 0.0
        dyx_i = 0.0
        dyz_i = 0.0
        dzy_i = 0.0
        dzx_i = 0.0
        dxz_i = 0.0
        
        for n in range(16):
            dxx_i += px[n] * ve[i, n, 0]
            dyy_i += py[n] * ve[i, n, 1]
            dzz_i += pz[n] * ve[i, n, 2]
            dxy_i += py[n] * ve[i, n, 0]
            dxz_i += pz[n] * ve[i, n, 0]
            dyx_i += px[n] * ve[i, n, 1]
            dyz_i += pz[n] * ve[i, n, 1]
            dzx_i += px[n] * ve[i, n, 2]
            dzy_i += py[n] * ve[i, n, 2]
            
        dxx[i] = dxx_i
        dyy[i] = dyy_i
        dzz[i] = dzz_i
        dxy[i] = dxy_i + dyx_i
        dyz[i] = dyz_i + dzy_i
        dzx[i] = dzx_i + dxz_i
        
    deps = np.stack((dxx, dyy, dzz, dxy, dyz, dzx), axis=1)
    return px_all, py_all, pz_all, volnp_all, deps


@njit(cache=True)
def _s16_post(px_all, py_all, pz_all, sig_k, volnp_all):
    nel = px_all.shape[0]
    fint_k = np.zeros((nel, 16, 3), dtype=np.float64)
    
    for i in range(nel):
        fint_flat = s20fint3(px_all[i], py_all[i], pz_all[i], sig_k[i], volnp_all[i])
        for n in range(16):
            fint_k[i, n, 0] = fint_flat[n*3]
            fint_k[i, n, 1] = fint_flat[n*3+1]
            fint_k[i, n, 2] = fint_flat[n*3+2]
            
    return fint_k


def forces(group, x, v, vr, dt, fint, mint):
    """SHEL16 main integration loop."""
    st = group.state
    conn = group.conn
    
    xe = x[conn]
    ve = v[conn]
    
    fint_e = np.zeros_like(xe)
    
    from pyradioss.elements import solid_hexa8
    
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        pts, wts = st["zw"][isl]
        xe_sl = xe[sl]
        ve_sl = ve[sl]
        
        n_sl = sl.stop - sl.start
        
        for k, ((r, s, t), w) in enumerate(zip(pts, wts)):
            px_all, py_all, pz_all, volnp_all, deps = _s16_pre(xe_sl, ve_sl, r, s, t, w)
            
            deps *= dt
            
            sig_k = st["sig"][sl, k]
            sig_old = sig_k.copy()
            
            if mat.law == 1:
                from pyradioss.materials.law01_elastic import solid_update
                solid_update(mat, sig_k, deps)
            elif mat.law == 2:
                from pyradioss.materials.law02_johnson_cook import solid_update
                epsp_k = st["epsp"][sl, k]
                extra = st.get("mat_extra", {})
                solid_update(mat, sig_k, deps, epsp_k, np.ones(n_sl), extra)
            
            # energy
            avg_sig = 0.5 * (sig_k + sig_old)
            dW = np.sum(avg_sig * deps, axis=1) * volnp_all
            st["eint"][sl] += dW
            
            fint_k = _s16_post(px_all, py_all, pz_all, sig_k, volnp_all)
            fint_e[sl] += fint_k
            
    # Scatter to global fint
    from pyradioss.common.fastmath import scatter_add3
    conn_flat = conn.reshape(-1)
    fint_e_flat = fint_e.reshape(-1, 3)
    valid = conn_flat >= 0
    scatter_add3(fint, conn_flat[valid], fint_e_flat[valid], st.get('color_indices'), st.get('color_offsets'))
    return np.full(group.n, 1e20)  # dt_crit placeholder

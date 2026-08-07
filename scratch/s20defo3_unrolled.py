import numpy as np
from numba import njit

@njit
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

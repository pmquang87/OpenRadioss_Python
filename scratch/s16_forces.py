
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
    scatter_add3(fint, conn_flat[valid], fint_e_flat[valid])
    return np.full(group.n, 1e20)  # dt_crit placeholder

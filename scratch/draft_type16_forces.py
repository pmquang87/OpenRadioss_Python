import numpy as np

def _get_gauss(n):
    # Returns Gauss points and weights for 1D integration
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    elif n == 2:
        return np.array([-0.5773502691896257, 0.5773502691896257]), np.array([1.0, 1.0])
    elif n == 3:
        return np.array([-0.7745966692414834, 0.0, 0.7745966692414834]), np.array([0.5555555555555557, 0.8888888888888888, 0.5555555555555557])
    # fallback to leggauss for larger n
    return np.polynomial.legendre.leggauss(n)


def forces(group, x, v, vr, dt, fint, mint):
    """SHEL16 main integration loop."""
    st = group.state
    conn = group.conn
    
    xe = x[conn]
    ve = v[conn]
    
    # We will accumulate elemental forces here
    fe = np.zeros_like(xe)
    
    # We loop over slices because different slices might have different properties (like npts_r, npts_s, npts_t)
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        n_sl = sl.stop - sl.start
        
        # Get number of integration points
        npts_r = int(prop.params.get("npts_r", 2))
        npts_s = int(prop.params.get("npts_s", 2))
        npts_t = int(prop.params.get("npts_t", 2))
        if npts_r == 0: npts_r = 2
        if npts_s == 0: npts_s = 2
        if npts_t == 0: npts_t = 2
        
        xr, wr = _get_gauss(npts_r)
        xs, ws = _get_gauss(npts_s)
        xt, wt = _get_gauss(npts_t)
        
        xe_sl = xe[sl]
        ve_sl = ve[sl]
        
        fe_sl = np.zeros((n_sl, 16, 3))
        
        # TODO: Loop over integration points
        for r_idx in range(npts_r):
            for s_idx in range(npts_s):
                for t_idx in range(npts_t):
                    r_val = xr[r_idx]
                    s_val = xs[s_idx]
                    t_val = xt[t_idx]
                    w_val = wr[r_idx] * ws[s_idx] * wt[t_idx]
                    
                    # 1. s16rst
                    # 2. s16deri3
                    # 3. s20defo3
                    # 4. Material update
                    # 5. s20fint3
                    pass


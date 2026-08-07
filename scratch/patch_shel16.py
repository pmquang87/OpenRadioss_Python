import re

with open('pyradioss/elements/shell_thick16.py', 'r') as f:
    text = f.read()

init_group_new = '''def init_group(group, model, log):
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
'''

# Read s16_forces.py
with open('scratch/s16_forces.py', 'r') as f2:
    forces_new = f2.read()

# Replace init_group
text = re.sub(r'def init_group\(group, model, log\):.*?return conn\[valid\], mass_16\[valid\], np.zeros\(valid\.sum\(\)\)\n', init_group_new, text, flags=re.DOTALL)

# Replace forces
text = re.sub(r'def forces\(group, x, v, vr, dt, fint, mint\):\n\s+pass\s*', forces_new, text, flags=re.DOTALL)

with open('pyradioss/elements/shell_thick16.py', 'w') as f3:
    f3.write(text)


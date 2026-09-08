"""
Cohesive failure model for solid spotwelds (/FAIL/SNCONNECT).

Fortran origin: ``engine/source/materials/fail/snconnect/fail_snconnect.F``
"""

import numpy as np

_TINY = 1e-20

# State cache on fail instance to hold persistent state across cycles.
_fallback_cache = {}

def _get_state(fail, dama):
    """Retrieve or allocate persistent state arrays matching the base of `dama`.
    Bound to `fail._snconnect_state` so multiple instances/restarts don't conflict."""
    base = dama.base if dama.base is not None else dama
    base_id = id(base)

    if fail is not None:
        if not hasattr(fail, "_snconnect_state"):
            fail._snconnect_state = {}
        cache = fail._snconnect_state
    else:
        cache = _fallback_cache

    if base_id not in cache or cache[base_id][0].shape != base.shape:
        n = len(base)
        cache[base_id] = (
            np.zeros(n, dtype=np.float64),
            np.zeros(n, dtype=np.float64),
            np.zeros(n, dtype=np.float64),
            base,
        )

    epsp_all, pla1_all, pla2_all, _ = cache[base_id]

    if dama.base is not None:
        offset = (dama.__array_interface__['data'][0] - base.__array_interface__['data'][0]) // dama.itemsize
        sl = slice(offset, offset + len(dama))
    else:
        sl = slice(None)

    return epsp_all[sl], pla1_all[sl], pla2_all[sl]

def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """3-D damage step for SNCONNECT."""
    p = fail.params
    a2, b2, a3, b3 = p.get("a2", 0.0), p.get("b2", 1.0), p.get("a3", 0.0), p.get("b3", 1.0)
    isym = p.get("isym", 0)

    epsp, pla1, pla2 = _get_state(fail, dama)
    
    # accumulate local epsp
    epsp += np.maximum(d_epsp, 0.0)
    eps_dot = np.where(dt > 0.0, d_epsp / dt, 0.0)
    
    # evaluate rate functions (finter in Fortran)
    # The functions are stored in the model, but we just use 1.0 if not provided or 0
    # For M79, we will implement the formulas with fun=1.0 when missing.
    fun2n = np.ones_like(eps_dot)
    fun2t = np.ones_like(eps_dot)
    fun3n = np.ones_like(eps_dot)
    fun3t = np.ones_like(eps_dot)
    
    # normal and shear stresses
    signzz = sig[:, 2]
    signyz = sig[:, 4]
    signzx = sig[:, 5]
    
    ssym = 0.0 # sin(sym) - sym is from SYM array, but we don't have SYM in standard solids. We will use 0.0.
    svmn = np.abs(signzz)
    svmt = np.sqrt(signyz**2 + signzx**2)
    phi = np.arctan2(svmn, np.maximum(svmt, _TINY))
    sphi = np.sin(phi)
    cphi = np.cos(phi)
    
    # Phase 1: No damage yet (pla1 == 0)
    mask1 = (pla1 == 0.0)
    if np.any(mask1):
        t1 = np.where((isym == 1) & (signzz <= 0.0), 0.0, sphi / (1.0 - a2 * ssym) / fun2n)
        t2 = cphi / fun2t
        
        ttn = t1[mask1] * epsp[mask1]
        tts = t2[mask1] * epsp[mask1]
        fct = (ttn**b2 + tts**b2) ** (1.0 / b2)
        
        # check if damage starts
        start_dmg = fct > 1.0
        if np.any(start_dmg):
            # sub-mask of mask1 where damage starts
            idx = np.where(mask1)[0][start_dmg]
            
            # freeze PLA1
            pla1[idx] = (t1[idx]**b2 + t2[idx]**b2) ** (-1.0 / b2)
            
            # compute PLA2 for these elements
            t1_3 = np.where((isym == 1) & (signzz[idx] <= 0.0), 0.0, sphi[idx] / (1.0 - a3 * ssym) / fun3n[idx])
            t2_3 = cphi[idx] / fun3t[idx]
            
            ttn3 = t1_3 * epsp[idx]
            tts3 = t2_3 * epsp[idx]
            fct3 = (ttn3**b3 + tts3**b3) ** (1.0 / b3)
            
            pla2[idx] = (t1_3**b3 + t2_3**b3) ** (-1.0 / b3)
            
            d_val = (epsp[idx] - pla1[idx]) / np.maximum(_TINY, pla2[idx] - pla1[idx])
            dama[idx] = np.minimum(d_val, 1.0)
    
    # Phase 2: Damage is progressing (pla1 > 0)
    mask2 = (pla1 > 0.0)
    if np.any(mask2):
        t1_3 = np.where((isym == 1) & (signzz[mask2] <= 0.0), 0.0, sphi[mask2] / (1.0 - a3 * ssym) / fun3n[mask2])
        t2_3 = cphi[mask2] / fun3t[mask2]
        
        ttn3 = t1_3 * epsp[mask2]
        tts3 = t2_3 * epsp[mask2]
        fct3 = (ttn3**b3 + tts3**b3) ** (1.0 / b3)
        
        pla2[mask2] = (t1_3**b3 + t2_3**b3) ** (-1.0 / b3)
        
        d_val = (epsp[mask2] - pla1[mask2]) / np.maximum(_TINY, pla2[mask2] - pla1[mask2])
        dama[mask2] = np.minimum(d_val, 1.0)
        
        # check rupture
        rupture = fct3 > 1.0
        if np.any(rupture):
            idx = np.where(mask2)[0][rupture]
            dama[idx] = 1.0
            
    return dama >= 1.0

"""
LAW83 — solid spotweld material (/MAT/LAW83).

Fortran origin: ``engine/source/materials/mat/mat083/sigeps83.F``

This material is specifically designed for the CONNECT (TYPE43) solid element,
which only features normal and shear strains/stresses in its local frame (ZZ, YZ, ZX).
"""

from __future__ import annotations

import numpy as np

def build_law83(rec):
    p = rec.params
    from ..model.model import Material
    
    E = p.get("MAT_E", 0.0)
    G = p.get("MAT_G", 0.0)
    if G == 0.0:
        G = E / (2.0 * 1.3)
        
    params = {
        "E": E,
        "nu": 0.3,
        "alpha": p.get("MAT_ALPHA", 0.0),
        "beta": p.get("MAT_Beta", 2.0),
        "yfac": p.get("FScale11", 1.0),
        "xscale": p.get("FScale22", 1.0),
        "rn": p.get("MAT_R00", 0.0),
        "rs": p.get("MAT_R45", 0.0),
        "xfac": p.get("FScale33", 1.0),
        "rhoflag": int(p.get("MAT_REFRHO_Option", 0)),
        "iplas": int(p.get("MAT_IPLAS", 1)),
        "G": G,
        "icomp": int(p.get("COMP_OPT", 1)),
        "E_comp": p.get("MAT_ECOMP", E),
        "vp": int(p.get("VP", 0)),
        "ifun_n": int(p.get("FUN_A2", 0)),
        "ifun_t": int(p.get("FUN_A3", 0)),
        "id_yield": int(p.get("FUN_A1", 0))
    }
    return Material(id=rec.id, law=83, rho0=rec.density, title=rec.title, params=params)


def resolve(mat, model, log):
    """Resolve /FUNCT references into plain array views for the Engine."""
    def _resolve_one(fid, prefix):
        if fid == 0:
            return
        fct = model.functions.get(fid)
        if fct is None:
            log.error(f"/MAT/LAW83/{mat.id}: function {fid} not defined", "MAT CHECK")
            return
        if np.any(fct.x < 0.0):
            log.error(f"/MAT/LAW83/{mat.id}: curve {fid} has negative abscissae", "MAT CHECK")
        mat.params[f"{prefix}_x"] = fct.x.copy()
        mat.params[f"{prefix}_y"] = fct.y.copy()
        mat.params[f"{prefix}_s"] = fct.slope.copy()
        
    _resolve_one(mat.params["ifun_n"], "curve_n")
    _resolve_one(mat.params["ifun_t"], "curve_t")
    _resolve_one(mat.params["id_yield"], "curve_y")

def _curve_eval(cx: np.ndarray, cy: np.ndarray, cs: np.ndarray, e: np.ndarray):
    i = np.minimum(np.maximum(np.searchsorted(cx, e, side="right") - 1, 0), len(cx) - 2)
    return cy[i] + cs[i] * (e - cx[i]), cs[i]

def solid_update(mat, sig: np.ndarray, deps: np.ndarray, epsp: np.ndarray, dt: float, extra: dict):
    """Update solid spotweld stress (ZZ, YZ, ZX only)."""
    p = mat.params
    nel = len(sig)
    
    # 1. Update EPSP and EPSD
    if p["vp"] == 0:
        deps_eff = np.sqrt(deps[:, 2]**2 + deps[:, 4]**2 + deps[:, 5]**2)
        epsp_rate = deps_eff / dt if dt > 1e-20 else np.zeros_like(deps_eff)
        extra["asrate"][:] = p["xscale"] * epsp_rate + (1 - p["xscale"]) * extra["asrate"]
        epsd = extra["asrate"]
    else:
        epsd = extra["asrate"]
        
    epsp_val = extra["epsp"]
    
    # 2. Extract curves
    def _get_curve_val(prefix, x, fallback_val):
        if f"{prefix}_x" in p:
            cx, cy, cs = p[f"{prefix}_x"], p[f"{prefix}_y"], p[f"{prefix}_s"]
            val, slope = _curve_eval(cx, cy, cs, x)
            return val, slope
        return np.full(nel, fallback_val), np.zeros(nel)
        
    rn, _ = _get_curve_val("curve_n", epsp_val * p["xscale"], p["rn"])
    rs, _ = _get_curve_val("curve_t", epsp_val * p["xscale"], p["rs"])
    fyield, hyield = _get_curve_val("curve_y", epsp_val * p["xfac"], 0.0)
    
    dmg = np.zeros(nel)
    if "dmg" in extra:
        dmg = extra["dmg"]
    fyield = np.maximum(0.0, fyield) * (1.0 - dmg) * p["yfac"]
    hyield = hyield * p["yfac"]
    
    # 3. Compute Elastic Trial Stresses
    young = np.full(nel, p["E"])
    if p["icomp"] == 0:
        young = np.where(sig[:, 2] > 0.0, p["E"], p["E_comp"])
    elif p["icomp"] == 1:
        young = np.where(sig[:, 2] > 0.0, p["E"], p["E_comp"])
        
    sig_tr_zz = sig[:, 2] + young * deps[:, 2]
    sig_tr_yz = sig[:, 4] + p["G"] * deps[:, 4]
    sig_tr_zx = sig[:, 5] + p["G"] * deps[:, 5]
    
    # 4. Plasticity Return Mapping
    if p["beta"] == 2.0:
        mask1 = np.ones(nel, dtype=bool) if p["icomp"] == 0 else (sig_tr_zz > 0.0)
        mask2 = np.zeros(nel, dtype=bool) if p["icomp"] == 0 else (sig_tr_zz <= 0.0)
        
        # ---------------------------
        # Condition 1: icomp == 0 or (icomp == 1 and sig_tr_zz > 0)
        # ---------------------------
        if np.any(mask1):
            m1 = mask1
            aa = rn[m1] * (1.0 - p["alpha"] * 0.0)
            an = 1.0 / np.maximum(1e-20, aa)**2
            as_t = 1.0 / np.maximum(1e-20, rs[m1])**2
            
            szz, syz, szx = sig_tr_zz[m1], sig_tr_yz[m1], sig_tr_zx[m1]
            fy = fyield[m1]
            hy = hyield[m1]
            E = young[m1]
            
            sig_eff = an * szz**2 + as_t * (syz**2 + szx**2)
            phi = sig_eff - fy**2
            
            yield_mask = (sig_eff > 0.0) & (phi > 0.0)
            if np.any(yield_mask):
                ym = yield_mask
                an_y = an[ym]
                ast_y = as_t[ym]
                szz_y, syz_y, szx_y = szz[ym], syz[ym], szx[ym]
                fy_y, hy_y = fy[ym], hy[ym]
                E_y = E[ym]
                phi_y = phi[ym]
                
                normef = np.sqrt(szz_y**2 + syz_y**2 + szx_y**2)
                nzz = szz_y / normef
                nyz = syz_y / normef
                nzx = szx_y / normef
                
                facn = 2.0 * an_y * E_y
                fact = 2.0 * ast_y * p["G"]
                
                dep = np.zeros(np.count_nonzero(ym))
                sig_zz_n = szz_y
                sig_yz_n = syz_y
                sig_zx_n = szx_y
                fy_n = fy_y
                
                for it in range(3):
                    dszz = -facn * sig_zz_n * nzz
                    dsyz = -fact * sig_yz_n * nyz
                    dszx = -fact * sig_zx_n * nzx
                    dyld = -2.0 * fy_n * hy_y
                    fprim = dszz + dsyz + dszx + dyld
                    
                    valid = fprim != 0.0
                    if np.any(valid):
                        dep[valid] = dep[valid] - phi_y[valid] / fprim[valid]
                        sig_zz_n = szz_y - E_y * dep * nzz
                        sig_yz_n = syz_y - p["G"] * dep * nyz
                        sig_zx_n = szx_y - p["G"] * dep * nzx
                        fy_n = fy_y + hy_y * dep
                        sig_eff_n = an_y * sig_zz_n**2 + ast_y * (sig_yz_n**2 + sig_zx_n**2)
                        phi_y = sig_eff_n - fy_n**2
                        
                dep = np.maximum(0.0, dep)
                
                full_idx = np.nonzero(m1)[0][ym]
                epsp_val[full_idx] += dep
                sig_tr_zz[full_idx] = szz_y - E_y * dep * nzz
                sig_tr_yz[full_idx] = syz_y - p["G"] * dep * nyz
                sig_tr_zx[full_idx] = szx_y - p["G"] * dep * nzx

        # ---------------------------
        # Condition 2: icomp == 1 and sig_tr_zz <= 0
        # ---------------------------
        if np.any(mask2):
            m2 = mask2
            as_t = 1.0 / np.maximum(1e-20, rs[m2])**2
            
            syz, szx = sig_tr_yz[m2], sig_tr_zx[m2]
            fy = fyield[m2]
            hy = hyield[m2]
            
            sig_eff = as_t * (syz**2 + szx**2)
            phi = sig_eff - fy**2
            
            yield_mask = (sig_eff > 0.0) & (phi > 0.0)
            if np.any(yield_mask):
                ym = yield_mask
                ast_y = as_t[ym]
                syz_y, szx_y = syz[ym], szx[ym]
                fy_y, hy_y = fy[ym], hy[ym]
                phi_y = phi[ym]
                
                normef = np.sqrt(syz_y**2 + szx_y**2)
                nyz = syz_y / normef
                nzx = szx_y / normef
                
                fact = 2.0 * ast_y * p["G"]
                dep = np.zeros(np.count_nonzero(ym))
                sig_yz_n = syz_y
                sig_zx_n = szx_y
                fy_n = fy_y
                
                for it in range(3):
                    dsyz = -fact * sig_yz_n * nyz
                    dszx = -fact * sig_zx_n * nzx
                    dyld = -2.0 * fy_n * hy_y
                    fprim = dsyz + dszx + dyld
                    
                    valid = fprim != 0.0
                    if np.any(valid):
                        dep[valid] = dep[valid] - phi_y[valid] / fprim[valid]
                        sig_yz_n = syz_y - p["G"] * dep * nyz
                        sig_zx_n = szx_y - p["G"] * dep * nzx
                        fy_n = fy_y + hy_y * dep
                        sig_eff_n = ast_y * (sig_yz_n**2 + sig_zx_n**2)
                        phi_y = sig_eff_n - fy_n**2
                        
                dep = np.maximum(0.0, dep)
                
                full_idx = np.nonzero(m2)[0][ym]
                epsp_val[full_idx] += dep
                sig_tr_yz[full_idx] = syz_y - p["G"] * dep * nyz
                sig_tr_zx[full_idx] = szx_y - p["G"] * dep * nzx

    sig[:, 2] = sig_tr_zz
    sig[:, 4] = sig_tr_yz
    sig[:, 5] = sig_tr_zx
    return sig, epsp_val, None

def _register():
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["LAW83"] = build_law83

_register()

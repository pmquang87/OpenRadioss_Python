"""
LAW83 — solid spotweld material (/MAT/LAW83, /MAT/CONNECT, /MAT/SPR_JOU).

Fortran origin: ``engine/source/materials/mat/mat083/sigeps83.F``,
``starter/source/materials/mat/mat083/hm_read_mat83.F``.

This material is specifically designed for the CONNECT (TYPE43) solid element,
which only features normal and shear strains/stresses in its local frame (ZZ, YZ, ZX).
"""

from __future__ import annotations

import numpy as np

from ..model.entities import Material

_EM20 = 1e-20


def _ensure_params(mat: Material) -> dict:
    """Ensure material params contain both CFG and direct keys with robust defaults."""
    p = mat.params
    E = float(p.get("E") or p.get("MAT_E") or 0.0)
    G = float(p.get("G") or p.get("MAT_G") or 0.0)
    if G <= 0.0 and E > 0.0:
        G = E / (2.0 * 1.3)
    nu = float(p.get("nu") or 0.3)

    alpha = float(p.get("alpha") if p.get("alpha") is not None else p.get("MAT_ALPHA", 0.0))
    beta = float(p.get("beta") or p.get("MAT_Beta") or 2.0)
    rn = float(p.get("rn") or p.get("MAT_R00") or 1.0)
    rs = float(p.get("rs") or p.get("MAT_R45") or 1.0)
    yfac = float(p.get("yfac") or p.get("FScale11") or 1.0)
    xfac = float(p.get("xfac") or p.get("FScale22") or 1.0)
    xscale = float(p.get("xscale") or p.get("FScale33") or 1.0)
    vp = int(p.get("vp") or p.get("VP") or 0)
    icomp = int(p.get("icomp") if p.get("icomp") is not None else p.get("COMP_OPT", 1))
    e_comp = float(p.get("E_comp") or p.get("MAT_ECOMP") or (E if E > 0.0 else 1.0))
    sig_y = float(p.get("sig_y") or p.get("yield") or p.get("fyield") or p.get("MAT_SIGY") or 1e20)
    fsmooth = int(p.get("fsmooth") or p.get("Fsmooth") or 0)
    fcut = float(p.get("fcut") or p.get("Fcut") or 1e30)

    id_yield = int(p.get("id_yield") or p.get("FUN_A1") or p.get("fun_a1") or 0)
    ifun_n = int(p.get("ifun_n") or p.get("FUN_A2") or p.get("fun_a2") or 0)
    ifun_t = int(p.get("ifun_t") or p.get("FUN_A3") or p.get("fun_a3") or 0)

    p.setdefault("E", E)
    p.setdefault("G", G)
    p.setdefault("nu", nu)
    p.setdefault("alpha", alpha)
    p.setdefault("beta", beta)
    p.setdefault("rn", rn)
    p.setdefault("rs", rs)
    p.setdefault("yfac", yfac)
    p.setdefault("xfac", xfac)
    p.setdefault("xscale", xscale)
    p.setdefault("vp", vp)
    p.setdefault("icomp", icomp)
    p.setdefault("E_comp", e_comp)
    p.setdefault("sig_y", sig_y)
    p.setdefault("fsmooth", fsmooth)
    p.setdefault("fcut", fcut)
    p.setdefault("id_yield", id_yield)
    p.setdefault("ifun_n", ifun_n)
    p.setdefault("ifun_t", ifun_t)

    # CFG mirrored keys
    p.setdefault("MAT_E", E)
    p.setdefault("MAT_G", G)
    p.setdefault("MAT_ALPHA", alpha)
    p.setdefault("MAT_Beta", beta)
    p.setdefault("MAT_R00", rn)
    p.setdefault("MAT_R45", rs)
    p.setdefault("FScale11", yfac)
    p.setdefault("FScale22", xfac)
    p.setdefault("FScale33", xscale)
    p.setdefault("COMP_OPT", icomp)
    p.setdefault("MAT_ECOMP", e_comp)
    p.setdefault("FUN_A1", id_yield)
    p.setdefault("FUN_A2", ifun_n)
    p.setdefault("FUN_A3", ifun_t)
    p.setdefault("Fsmooth", fsmooth)
    p.setdefault("Fcut", fcut)
    return p


def build_law83(rec) -> Material:
    """hm_read_mat83.F: card parsing and validation -> Material."""
    if isinstance(rec, Material):
        p = rec.params
        mat_id = rec.id
        title = rec.title
        density = rec.rho0
    elif isinstance(rec, dict):
        p = rec.get("params", rec)
        mat_id = rec.get("id", 1)
        title = rec.get("title", "")
        val = None
        for k in ("density", "rho", "rho0", "MAT_RHO"):
            if k in rec and rec[k] is not None:
                val = rec[k]
                break
        if val is None and isinstance(p, dict):
            for k in ("density", "rho", "rho0", "MAT_RHO"):
                if k in p and p[k] is not None:
                    val = p[k]
                    break
        density = float(val) if val is not None else 1.0
    else:
        p = getattr(rec, "params", {})
        mat_id = getattr(rec, "id", 1)
        title = getattr(rec, "title", "")
        val = None
        for k in ("density", "rho", "rho0"):
            if hasattr(rec, k) and getattr(rec, k) is not None:
                val = getattr(rec, k)
                break
        density = float(val) if val is not None else 1.0

    E = float(p.get("MAT_E") or p.get("E") or 0.0)
    if E <= 0.0:
        raise ValueError("LAW83: Young modulus E must be positive (hm_read_mat83 error)")
    if density <= 0.0:
        raise ValueError("LAW83: Density rho0 must be positive (hm_read_mat83 error)")

    G = float(p.get("MAT_G") or p.get("G") or 0.0)
    if G <= 0.0:
        G = E / (2.0 * 1.3)

    params = {
        "E": E,
        "G": G,
        "nu": float(p.get("nu") or 0.3),
        "alpha": float(p.get("MAT_ALPHA") if p.get("MAT_ALPHA") is not None else p.get("alpha", 0.0)),
        "beta": float(p.get("MAT_Beta") or p.get("beta") or 2.0),
        "yfac": float(p.get("FScale11") or p.get("yfac") or 1.0),
        "xfac": float(p.get("FScale22") or p.get("xfac") or 1.0),
        "rn": float(p.get("MAT_R00") or p.get("rn") or 1.0),
        "rs": float(p.get("MAT_R45") or p.get("rs") or 1.0),
        "xscale": float(p.get("FScale33") or p.get("xscale") or 1.0),
        "rhoflag": int(p.get("MAT_REFRHO_Option", 0)),
        "iplas": int(p.get("MAT_IPLAS", 1)),
        "icomp": int(p.get("COMP_OPT") if p.get("COMP_OPT") is not None else p.get("icomp", 1)),
        "E_comp": float(p.get("MAT_ECOMP") or p.get("E_comp") or E),
        "vp": int(p.get("VP") or p.get("vp") or 0),
        "ifun_n": int(p.get("FUN_A2") or p.get("ifun_n") or 0),
        "ifun_t": int(p.get("FUN_A3") or p.get("ifun_t") or 0),
        "id_yield": int(p.get("FUN_A1") or p.get("id_yield") or 0),
        "sig_y": float(p.get("MAT_SIGY") or p.get("sig_y") or p.get("yield") or 1e20),
        "fsmooth": int(p.get("Fsmooth") or p.get("fsmooth") or 0),
        "fcut": float(p.get("Fcut") or p.get("fcut") or 1e30),
    }
    mat = Material(id=mat_id, law=83, rho0=density, title=title, params=params)
    _ensure_params(mat)
    return mat


def resolve(mat: Material, model, log):
    """Resolve /FUNCT references into plain array views for the Engine."""
    if model is None or not hasattr(model, "functions"):
        return

    def _resolve_one(fid, prefix):
        if fid == 0:
            return
        fct = model.functions.get(fid)
        if fct is None:
            if log is not None:
                log.error(f"/MAT/LAW83/{mat.id}: function {fid} not defined", "MAT CHECK")
            return
        if np.any(fct.x < 0.0) and log is not None:
            log.error(f"/MAT/LAW83/{mat.id}: curve {fid} has negative abscissae", "MAT CHECK")
        mat.params[f"{prefix}_x"] = fct.x.copy()
        mat.params[f"{prefix}_y"] = fct.y.copy()
        mat.params[f"{prefix}_s"] = getattr(fct, "slope", None)
        if mat.params[f"{prefix}_s"] is None:
            # Compute slopes on the fly if needed
            dx = np.diff(fct.x)
            dy = np.diff(fct.y)
            slopes = np.where(dx != 0.0, dy / np.maximum(dx, 1e-20), 0.0)
            mat.params[f"{prefix}_s"] = np.append(slopes, slopes[-1] if len(slopes) > 0 else 0.0)

    _resolve_one(mat.params.get("ifun_n", 0), "curve_n")
    _resolve_one(mat.params.get("ifun_t", 0), "curve_t")
    _resolve_one(mat.params.get("id_yield", 0), "curve_y")


def _curve_eval(cx: np.ndarray, cy: np.ndarray, cs: np.ndarray, e: np.ndarray):
    """Piecewise linear function evaluation with extrapolation."""
    if len(cx) == 0:
        return np.zeros_like(e), np.zeros_like(e)
    if len(cx) == 1:
        return np.full_like(e, cy[0]), np.zeros_like(e)
    i = np.minimum(np.maximum(np.searchsorted(cx, e, side="right") - 1, 0), len(cx) - 2)
    s = cs[i] if cs is not None and len(cs) > i.max() else (cy[i + 1] - cy[i]) / np.maximum(cx[i + 1] - cx[i], _EM20)
    return cy[i] + s * (e - cx[i]), s


def solid_update(mat, sig: np.ndarray, deps: np.ndarray, epsp=None, dt: float = 0.0, extra: dict = None):
    """Update solid spotweld stress (ZZ, YZ, ZX only in local element coordinates).
    Returns (sig, epsp, c).
    """
    nel = len(sig)
    if nel == 0:
        return sig, np.zeros(0, dtype=sig.dtype if hasattr(sig, "dtype") else float), np.zeros(0, dtype=float)

    p = _ensure_params(mat)
    E = p["E"]
    G = p["G"]
    E_comp = p["E_comp"]
    icomp = p["icomp"]
    beta = p["beta"]
    alpha = p["alpha"]
    yfac = p["yfac"]
    xfac = p["xfac"]
    xscale = p["xscale"]
    vp = p["vp"]
    rho0 = float(getattr(mat, "rho0", 1.0) or p.get("rho0") or 1.0)

    # Auto-initialize extra dictionary
    if extra is None:
        extra = {}
    if "epsp" not in extra or extra["epsp"] is None:
        extra["epsp"] = np.zeros(nel, dtype=float)
    elif len(extra["epsp"]) != nel:
        extra["epsp"] = np.zeros(nel, dtype=float)

    if "asrate" not in extra or extra["asrate"] is None:
        extra["asrate"] = np.zeros(nel, dtype=float)
    elif len(extra["asrate"]) != nel:
        extra["asrate"] = np.zeros(nel, dtype=float)

    epsp_val = extra["epsp"]

    # Extract strain increments: support 6-component [xx, yy, zz, xy, yz, zx] or 3-component [zz, yz, zx]
    if deps is None:
        d_zz = np.zeros(nel, dtype=float)
        d_yz = np.zeros(nel, dtype=float)
        d_zx = np.zeros(nel, dtype=float)
    else:
        if deps.ndim == 1:
            deps = deps.reshape(1, -1)
        if deps.shape[1] >= 6:
            d_zz = deps[:, 2]
            d_yz = deps[:, 4]
            d_zx = deps[:, 5]
        elif deps.shape[1] >= 3:
            d_zz = deps[:, 0]
            d_yz = deps[:, 1]
            d_zx = deps[:, 2]
        else:
            d_zz = deps[:, 0]
            d_yz = np.zeros(nel, dtype=float)
            d_zx = np.zeros(nel, dtype=float)

    # 1. Update strain rate filter (asrate)
    deps_eff = np.sqrt(d_zz**2 + d_yz**2 + d_zx**2)
    if vp == 0:
        epsp_rate = (deps_eff / dt) if dt > 1e-20 else np.zeros_like(deps_eff)
        extra["asrate"][:] = xscale * epsp_rate + (1.0 - xscale) * extra["asrate"]
    epsd = extra["asrate"]

    # 2. Extract curves / static properties
    def _get_curve_val(prefix, x, fallback_val):
        if f"{prefix}_x" in p:
            cx, cy, cs = p[f"{prefix}_x"], p[f"{prefix}_y"], p.get(f"{prefix}_s")
            val, slope = _curve_eval(cx, cy, cs, x)
            return val, slope
        return np.full(nel, fallback_val), np.zeros(nel)

    rn, _ = _get_curve_val("curve_n", epsd * xscale, p["rn"])
    rs, _ = _get_curve_val("curve_t", epsd * xscale, p["rs"])
    rn = np.maximum(rn, _EM20)
    rs = np.maximum(rs, _EM20)

    static_fy = float(p.get("sig_y") or p.get("yield") or p.get("fyield") or 1e20)
    fyield, hyield = _get_curve_val("curve_y", epsp_val * xfac, static_fy)

    dmg = extra.get("dmg") if "dmg" in extra and extra["dmg"] is not None else np.zeros(nel)
    fyield = np.maximum(0.0, fyield) * (1.0 - dmg) * yfac
    hyield = hyield * yfac

    # 3. Compute Elastic Trial Stresses
    if icomp == 0:
        young = np.full(nel, E)
    elif icomp == 1:
        sig_tr_test = sig[:, 2] + E * d_zz
        young = np.where(sig_tr_test >= 0.0, E, E_comp)
    else:
        young = np.full(nel, E)

    sig_tr_zz = sig[:, 2] + young * d_zz
    sig_tr_yz = sig[:, 4] + G * d_yz
    sig_tr_zx = sig[:, 5] + G * d_zx

    # 4. Plasticity Return Mapping
    if beta == 2.0:
        mask1 = np.ones(nel, dtype=bool) if icomp == 0 else (sig_tr_zz > 0.0)
        mask2 = np.zeros(nel, dtype=bool) if icomp == 0 else (sig_tr_zz <= 0.0)

        # ---------------------------
        # Branch 1: tension or icomp == 0
        # ---------------------------
        if np.any(mask1):
            m1 = mask1
            aa = rn[m1] * (1.0 - alpha * 0.0)
            an = 1.0 / np.maximum(_EM20, aa)**2
            as_t = 1.0 / np.maximum(_EM20, rs[m1])**2

            szz, syz, szx = sig_tr_zz[m1], sig_tr_yz[m1], sig_tr_zx[m1]
            fy = fyield[m1]
            hy = hyield[m1]
            E_val = young[m1]

            sig_eff = an * szz**2 + as_t * (syz**2 + szx**2)
            phi = sig_eff - fy**2

            yield_mask = (sig_eff > 0.0) & (phi > 0.0)
            if np.any(yield_mask):
                ym = yield_mask
                an_y = an[ym]
                ast_y = as_t[ym]
                szz_y, syz_y, szx_y = szz[ym], syz[ym], szx[ym]
                fy_y, hy_y = fy[ym], hy[ym]
                E_y = E_val[ym]
                phi_y = phi[ym]

                normef = np.maximum(np.sqrt(szz_y**2 + syz_y**2 + szx_y**2), _EM20)
                nzz = szz_y / normef
                nyz = syz_y / normef
                nzx = szx_y / normef

                facn = 2.0 * an_y * E_y
                fact = 2.0 * ast_y * G

                dep = np.zeros(np.count_nonzero(ym))
                sig_zz_n = szz_y.copy()
                sig_yz_n = syz_y.copy()
                sig_zx_n = szx_y.copy()
                fy_n = fy_y.copy()

                for it in range(10):
                    if np.all(np.abs(phi_y) <= 1e-8 * np.maximum(fy_n**2, 1.0)):
                        break
                    dszz = -facn * sig_zz_n * nzz
                    dsyz = -fact * sig_yz_n * nyz
                    dszx = -fact * sig_zx_n * nzx
                    dyld = -2.0 * fy_n * hy_y
                    fprim = dszz + dsyz + dszx + dyld

                    valid = np.abs(fprim) > _EM20
                    if np.any(valid):
                        dep[valid] = dep[valid] - phi_y[valid] / fprim[valid]
                        sig_zz_n = szz_y - E_y * dep * nzz
                        sig_yz_n = syz_y - G * dep * nyz
                        sig_zx_n = szx_y - G * dep * nzx
                        fy_n = fy_y + hy_y * dep
                        sig_eff_n = an_y * sig_zz_n**2 + ast_y * (sig_yz_n**2 + sig_zx_n**2)
                        phi_y = sig_eff_n - fy_n**2

                dep = np.maximum(0.0, dep)
                full_idx = np.nonzero(m1)[0][ym]
                epsp_val[full_idx] += dep
                sig_tr_zz[full_idx] = szz_y - E_y * dep * nzz
                sig_tr_yz[full_idx] = syz_y - G * dep * nyz
                sig_tr_zx[full_idx] = szx_y - G * dep * nzx

        # ---------------------------
        # Branch 2: icomp == 1 and compression (sig_tr_zz <= 0)
        # ---------------------------
        if np.any(mask2):
            m2 = mask2
            as_t = 1.0 / np.maximum(_EM20, rs[m2])**2
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

                normef = np.maximum(np.sqrt(syz_y**2 + szx_y**2), _EM20)
                nyz = syz_y / normef
                nzx = szx_y / normef

                fact = 2.0 * ast_y * G
                dep = np.zeros(np.count_nonzero(ym))
                sig_yz_n = syz_y.copy()
                sig_zx_n = szx_y.copy()
                fy_n = fy_y.copy()

                for it in range(10):
                    if np.all(np.abs(phi_y) <= 1e-8 * np.maximum(fy_n**2, 1.0)):
                        break
                    dsyz = -fact * sig_yz_n * nyz
                    dszx = -fact * sig_zx_n * nzx
                    dyld = -2.0 * fy_n * hy_y
                    fprim = dsyz + dszx + dyld

                    valid = np.abs(fprim) > _EM20
                    if np.any(valid):
                        dep[valid] = dep[valid] - phi_y[valid] / fprim[valid]
                        sig_yz_n = syz_y - G * dep * nyz
                        sig_zx_n = szx_y - G * dep * nzx
                        fy_n = fy_y + hy_y * dep
                        sig_eff_n = ast_y * (sig_yz_n**2 + sig_zx_n**2)
                        phi_y = sig_eff_n - fy_n**2

                dep = np.maximum(0.0, dep)
                full_idx = np.nonzero(m2)[0][ym]
                epsp_val[full_idx] += dep
                sig_tr_yz[full_idx] = syz_y - G * dep * nyz
                sig_tr_zx[full_idx] = szx_y - G * dep * nzx

    sig[:, 2] = sig_tr_zz
    sig[:, 4] = sig_tr_yz
    sig[:, 5] = sig_tr_zx

    c = np.full(nel, np.sqrt(max(E, _EM20) / max(rho0, _EM20)))
    return sig, epsp_val, c


def shell_update(mat, sig, deps, epsp=None, dt=0.0, extra=None):
    """LAW83 is defined strictly for solid spotweld (CONNECT) elements."""
    raise NotImplementedError("LAW83 (spotweld) is implemented for solid elements only.")


def consistent_solid_tangent(mat, sig=None, epsp=None, epsp_incr=None, extra=None):
    """(n, 6, 6) consistent algorithmic solid tangent matrix for LAW83 spotweld.
    Diagonal components:
    - (2, 2) ZZ: E (or E_comp in compression)
    - (4, 4) YZ: G
    - (5, 5) ZX: G
    With elastoplastic softening reduction on active yield surface.
    """
    p = _ensure_params(mat)
    E = p["E"]
    G = p["G"]
    E_comp = p["E_comp"]
    icomp = p["icomp"]

    if sig is not None and isinstance(sig, np.ndarray):
        n = sig.shape[0]
    elif extra is not None and "F" in extra:
        n = extra["F"].shape[0]
    else:
        n = 1

    if n == 0:
        return np.empty((0, 6, 6), dtype=float)

    D = np.zeros((n, 6, 6), dtype=float)

    # Set base elastic moduli
    for i in range(n):
        szz = sig[i, 2] if sig is not None and sig.ndim == 2 and sig.shape[1] > 2 else 1.0
        E_cur = E if (szz > 0.0 or icomp not in (0, 1)) else E_comp
        D[i, 2, 2] = E_cur
        D[i, 4, 4] = G
        D[i, 5, 5] = G

    # Plastic softening correction if actively yielding
    if sig is not None and epsp_incr is not None and np.any(epsp_incr > 0.0):
        yielding = np.asarray(epsp_incr > 0.0)
        rn = max(p["rn"], _EM20)
        rs = max(p["rs"], _EM20)
        an = 1.0 / (rn**2)
        ast = 1.0 / (rs**2)

        for i in np.nonzero(yielding)[0]:
            szz = sig[i, 2]
            syz = sig[i, 4]
            szx = sig[i, 5]
            E_cur = D[i, 2, 2]

            # Gradient of yield surface d_phi / d_sigma
            n_vec = np.zeros(6)
            if szz > 0.0 or icomp == 0:
                n_vec[2] = 2.0 * an * szz
            n_vec[4] = 2.0 * ast * syz
            n_vec[5] = 2.0 * ast * szx

            norm_grad = np.linalg.norm(n_vec)
            if norm_grad > _EM20:
                Ce_n = np.zeros(6)
                Ce_n[2] = E_cur * n_vec[2]
                Ce_n[4] = G * n_vec[4]
                Ce_n[5] = G * n_vec[5]
                denom = np.dot(n_vec, Ce_n)
                if denom > _EM20:
                    D[i] -= np.outer(Ce_n, Ce_n) / denom

    return D


def _register():
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["LAW83"] = build_law83
    MAT_PHYSICS_REGISTRY["CONNECT"] = build_law83
    MAT_PHYSICS_REGISTRY["SPR_JOU"] = build_law83


_register()

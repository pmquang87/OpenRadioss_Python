"""
LAW117, LAW120, LAW121, LAW124, LAW90 — Advanced constitutive material laws suite.

Fortran origin:
  - ``starter/source/materials/mat/mat117/hm_read_mat117.F`` (cohesive mixed-mode fracture)
  - ``starter/source/materials/mat/mat120/hm_read_mat120.F`` (tape orientation composite)
  - ``starter/source/materials/mat/mat121/hm_read_mat121.F`` (strain-rate plasticity)
  - ``starter/source/materials/mat/mat124/hm_read_mat124.F`` (concrete damage plasticity CDPM2)
  - ``starter/source/materials/mat/mat090/hm_read_mat90.F`` (tabulated foam with hysteresis)
"""
from __future__ import annotations

from ..model.entities import Material


def build_law117(rec) -> Material:
    """Constructor for /MAT/LAW117 cohesive interface material."""
    p = rec.params
    e_n = float(p.get("MAT_E_ELAS_N", 1.0))
    e_s = float(p.get("MAT_E_ELAS_S", 1.0))
    params = {
        "E": e_n if e_n > 0 else 1.0,
        "nu": 0.3,
        "e_elas_n": e_n,
        "e_elas_s": e_s,
        "fct_tn": int(p.get("MAT_Fct_TN", 0)),
        "fct_tt": int(p.get("MAT_Fct_TT", 0)),
        "tmax_n": float(p.get("MAT_TMAX_N", 0.0)),
        "tmax_s": float(p.get("MAT_TMAX_S", 0.0)),
        "fscale_x": float(p.get("MAT_Fscale_x", 1.0)),
        "irupt": int(p.get("MAT_IRUPT", 0)),
        "imass": int(p.get("MAT_IMASS", 0)),
        "idel": int(p.get("MAT_IDEL", 0)),
        "gic": float(p.get("MAT_GIC", 0.0)),
        "giic": float(p.get("MAT_GIIC", 0.0)),
        "exp_g": float(p.get("MAT_EXP_G", 1.0)),
        "exp_bk": float(p.get("MAT_EXP_BK", 1.0)),
        "gamma": float(p.get("MAT_GAMMA", 1.0)),
    }
    return Material(id=rec.id, law=117, rho0=rec.density, title=rec.title, params=params)


def build_law120(rec) -> Material:
    """Constructor for /MAT/LAW120 (/MAT/TAPO) tape orientation composite material."""
    p = rec.params
    e = float(p.get("MAT_E", 1.0))
    nu = float(p.get("MAT_NU", 0.3))
    params = {
        "E": e if e > 0 else 1.0,
        "nu": nu if (0.0 <= nu < 0.5) else 0.3,
        "thick": float(p.get("MAT_THICK", 0.0)),
        "tab_id": int(p.get("MAT_TAB_ID", 0)),
        "xscale": float(p.get("MAT_Xscale", 1.0)),
        "yscale": float(p.get("MAT_Yscale", 1.0)),
        "tau": float(p.get("MAT_TAU", 0.0)),
        "q": float(p.get("MAT_Q", 0.0)),
        "b": float(p.get("MAT_B", 0.0)),
        "h": float(p.get("MAT_H", 0.0)),
        "af1": float(p.get("MAT_AF1", 0.0)),
        "af2": float(p.get("MAT_AF2", 0.0)),
        "ah1": float(p.get("MAT_AH1", 0.0)),
        "ah2": float(p.get("MAT_AH2", 0.0)),
        "as": float(p.get("MAT_AS", 0.0)),
        "d1c": float(p.get("MAT_D1C", 0.0)),
        "d2c": float(p.get("MAT_D2C", 0.0)),
        "d1f": float(p.get("MAT_D1F", 0.0)),
        "d2f": float(p.get("MAT_D2F", 0.0)),
        "d_trx": float(p.get("D_TRX", 0.0)),
        "d_jc": float(p.get("D_JC", 0.0)),
        "exp": float(p.get("MAT_EXP", 0.0)),
        "cc": float(p.get("MAT_CC", 0.0)),
        "gam0": float(p.get("MAT_GAM0", 0.0)),
        "gamf": float(p.get("MAT_GAMF", 0.0)),
        "iform": int(p.get("MAT_IFORM", 0)),
        "itrx": int(p.get("MAT_ITRX", 0)),
        "idam": int(p.get("MAT_IDAM", 0)),
    }
    return Material(id=rec.id, law=120, rho0=rec.density, title=rec.title, params=params)


def build_law121(rec) -> Material:
    """Constructor for /MAT/LAW121 (/MAT/PLAS_RATE) strain-rate plasticity material."""
    p = rec.params
    e = float(p.get("MAT_E", 1.0))
    nu = float(p.get("MAT_NU", 0.3))
    params = {
        "E": e if e > 0 else 1.0,
        "nu": nu if (0.0 <= nu < 0.5) else 0.3,
        "ires": int(p.get("MAT_Ires", 0)),
        "ivisc": int(p.get("MAT_Ivisc", 0)),
        "fcut": float(p.get("Fcut", 0.0)),
        "fct_sig0": int(p.get("Fct_SIG0", 0)),
        "xscale_sig0": float(p.get("Xscale_SIG0", 1.0)),
        "yscale_sig0": float(p.get("Yscale_SIG0", 1.0)),
        "fct_youn": int(p.get("Fct_YOUN", 0)),
        "xscale_youn": float(p.get("Xscale_YOUN", 1.0)),
        "yscale_youn": float(p.get("Yscale_YOUN", 1.0)),
        "fct_tang": int(p.get("Fct_TANG", 0)),
        "xscale_tang": float(p.get("Xscale_TANG", 1.0)),
        "tang": float(p.get("MAT_TANG", 0.0)),
        "fct_fail": int(p.get("Fct_FAIL", 0)),
        "xscale_fail": float(p.get("Xscale_FAIL", 1.0)),
        "yscale_fail": float(p.get("Yscale_FAIL", 1.0)),
        "tdel": float(p.get("TDEL", 0.0)),
        "ifail": int(p.get("MAT_Ifail", 0)),
    }
    return Material(id=rec.id, law=121, rho0=rec.density, title=rec.title, params=params)


def build_law124(rec) -> Material:
    """Constructor for /MAT/LAW124 (/MAT/CDPM2) concrete damage plasticity model 2."""
    p = rec.params
    e = float(p.get("MAT_E", 1.0))
    nu = float(p.get("MAT_NU", 0.2))
    params = {
        "E": e if e > 0 else 1.0,
        "nu": nu if (0.0 <= nu < 0.5) else 0.2,
        "irate": int(p.get("IRATE", 0)),
        "fcut": float(p.get("FCUT", 0.0)),
        "ecc": float(p.get("MAT_ECC", 0.0)),
        "qh0": float(p.get("MAT_QH0", 0.0)),
        "ft": float(p.get("MAT_FT", 0.0)),
        "fc": float(p.get("MAT_FC", 0.0)),
        "hp": float(p.get("MAT_HP", 0.0)),
        "ah": float(p.get("MAT_AH", 0.0)),
        "bh": float(p.get("MAT_BH", 0.0)),
        "ch": float(p.get("MAT_CH", 0.0)),
        "dh": float(p.get("MAT_DH", 0.0)),
        "as": float(p.get("MAT_AS", 0.0)),
        "bs": float(p.get("MAT_BS", 0.0)),
        "df": float(p.get("MAT_DF", 0.0)),
        "dflag": int(p.get("DFLAG", 0)),
        "dtype": int(p.get("DTYPE", 0)),
        "ireg": int(p.get("IREG", 0)),
        "wf": float(p.get("MAT_WF", 0.0)),
        "wf1": float(p.get("MAT_WF1", 0.0)),
        "ft1": float(p.get("MAT_FT1", 0.0)),
        "efc": float(p.get("MAT_EFC", 0.0)),
    }
    return Material(id=rec.id, law=124, rho0=rec.density, title=rec.title, params=params)


def build_law90(rec) -> Material:
    """Constructor for /MAT/LAW90 tabulated foam with hysteresis."""
    p = rec.params
    e0 = float(p.get("MAT_E0", 1.0))
    nu = float(p.get("MAT_NU", 0.3))
    params = {
        "E": e0 if e0 > 0 else 1.0,
        "nu": nu if (0.0 <= nu < 0.5) else 0.3,
        "ismooth": int(p.get("Ismooth", 0)),
        "fcut": float(p.get("Fcut", 0.0)),
        "shape": float(p.get("MAT_SHAPE", 0.0)),
        "hys": float(p.get("Hys", 0.0)),
        "alpha": float(p.get("MAT_ALPHA", 0.0)),
        "nl": int(p.get("NL", 0)),
        "fct_idl": p.get("fct_IDL", []),
        "epsilondotl": p.get("EpsilondotL", []),
        "fscalel": p.get("FscaleL", []),
    }
    return Material(id=rec.id, law=90, rho0=rec.density, title=rec.title, params=params)


def _register() -> None:
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["LAW117"] = build_law117
    MAT_PHYSICS_REGISTRY["LAW120"] = build_law120
    MAT_PHYSICS_REGISTRY["TAPO"] = build_law120
    MAT_PHYSICS_REGISTRY["LAW121"] = build_law121
    MAT_PHYSICS_REGISTRY["PLAS_RATE"] = build_law121
    MAT_PHYSICS_REGISTRY["LAW124"] = build_law124
    MAT_PHYSICS_REGISTRY["CDPM2"] = build_law124
    MAT_PHYSICS_REGISTRY["LAW90"] = build_law90


_register()

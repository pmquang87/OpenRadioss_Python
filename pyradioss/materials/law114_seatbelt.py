"""
LAW114 & LAW119 — Seatbelt spring and shell materials (/MAT/LAW114, /MAT/LAW119).

Fortran origin:
  - ``starter/source/materials/mat/mat114/hm_read_mat114.F`` (1D spring seatbelt)
  - ``starter/source/materials/mat/mat119/hm_read_mat119.F`` (2D shell seatbelt)
"""
from __future__ import annotations

from ..model.entities import Material


def build_law114(rec) -> Material:
    """Constructor for /MAT/LAW114 (/MAT/SPR_SEATBELT) 1D seatbelt spring material."""
    p = rec.params
    stiff1 = float(p.get("STIFF1", 0.0))
    damp1 = float(p.get("DAMP1", 0.0))
    lmin = float(p.get("LMIN", 0.0))
    fun_l = int(p.get("FUN_L", 0))
    fun_ul = int(p.get("FUN_UL", 0))
    xscale = float(p.get("Xcoeft1", 1.0))
    fscale = float(p.get("Fcoeft1", 1.0))
    young = float(p.get("YOUNG", 0.0))
    e = young if young > 0 else (stiff1 if stiff1 > 0 else 1.0)
    nu = 0.3

    params = {
        "E": e,
        "nu": nu,
        "stiff1": stiff1,
        "damp1": damp1,
        "lmin": lmin,
        "fun_l": fun_l,
        "fun_ul": fun_ul,
        "xscale": xscale,
        "fscale": fscale,
        "young": young,
        "ibend": float(p.get("Ibend", 0.0)),
        "itors": float(p.get("Itors", 0.0)),
        "shear_area": float(p.get("SHEAR_AREA", 0.0)),
        "fmax": float(p.get("FMAX", 0.0)),
        "mmax": float(p.get("MMAX", 0.0)),
        "rfac": float(p.get("Rfac", 1.0)),
    }
    return Material(id=rec.id, law=114, rho0=rec.density, title=rec.title, params=params)


def build_law119(rec) -> Material:
    """Constructor for /MAT/LAW119 (/MAT/SH_SEATBELT) 2D shell seatbelt material."""
    p = rec.params
    stiff1 = float(p.get("STIFF1", 0.0))
    damp1 = float(p.get("DAMP1", 0.0))
    re = float(p.get("RE", 0.0))
    lmin = float(p.get("LMIN", 0.0))
    fun_l = int(p.get("FUN_L", 0))
    fun_ul = int(p.get("FUN_UL", 0))
    fcoeft1 = float(p.get("Fcoeft1", 1.0))
    fcoeft2 = float(p.get("Fcoeft2", 1.0))
    ireload = int(p.get("Ireload", 0))
    e22 = float(p.get("E22", 0.0))
    nu12 = float(p.get("NU12", 0.3))
    g12 = float(p.get("G12", 0.0))
    e = e22 if e22 > 0 else (stiff1 if stiff1 > 0 else 1.0)
    nu = nu12 if (0.0 <= nu12 < 0.5) else 0.3

    params = {
        "E": e,
        "nu": nu,
        "stiff1": stiff1,
        "damp1": damp1,
        "re": re,
        "lmin": lmin,
        "fun_l": fun_l,
        "fun_ul": fun_ul,
        "fcoeft1": fcoeft1,
        "fcoeft2": fcoeft2,
        "ireload": ireload,
        "e22": e22,
        "nu12": nu12,
        "g12": g12,
        "fcoeft22": float(p.get("Fcoeft22", 1.0)),
        "ecoat": float(p.get("ECOAT", 0.0)),
        "nucoat": float(p.get("NUCOAT", 0.0)),
        "tcoat": float(p.get("TCOAT", 0.0)),
    }
    return Material(id=rec.id, law=119, rho0=rec.density, title=rec.title, params=params)


def _register() -> None:
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["LAW114"] = build_law114
    MAT_PHYSICS_REGISTRY["SPR_SEATBELT"] = build_law114
    MAT_PHYSICS_REGISTRY["LAW119"] = build_law119
    MAT_PHYSICS_REGISTRY["SH_SEATBELT"] = build_law119


_register()

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


from .law120_tapo import build_law120
from .law121_plas_rate import build_law121


from .law124_cdpm2 import build_law124


from .law90_foam import build_law90


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

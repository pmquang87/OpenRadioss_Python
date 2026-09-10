"""
pyradioss.materials.law14_compso — /MAT/LAW14 (/MAT/COMPSO, /MAT/COMP_SOL).

3D Orthotropic Elastic-Plastic Composite Material for Solid Elements.

Fortran origin:
  - Starter reader: starter/source/materials/mat/mat014/hm_read_mat14.F
  - Engine kernel:  engine/source/materials/mat/mat014/m14law.F
  - Coordinate transformations:
      engine/source/materials/mat/mat014/m14ama.F
      engine/source/materials/mat/mat014/m14gtf.F
      engine/source/materials/mat/mat014/m14ftg.F
  - CFG specification:
      hm_cfg_files/config/CFG/radioss2020/MAT/matl14_compso.cfg
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material
from .law12_comp3d import _inv, _inv_prod, _tw_cross, m14ama, m14ftg, m14gtf, _update_one_element

_EM20 = 1.0e-20
_INF = 1.0e30


# ============================================================================
# Material Builder (hm_read_mat14.F)
# ============================================================================

def build_law14(rec: Any = None, **kwargs: Any) -> Material:
    """Construct /MAT/LAW14 (/MAT/COMPSO, /MAT/COMP_SOL) material with derived constants.

    Fortran source: starter/source/materials/mat/mat014/hm_read_mat14.F
    CFG: hm_cfg_files/config/CFG/radioss2020/MAT/matl14_compso.cfg
    """
    if rec is None:
        p_in: Dict[str, Any] = dict(kwargs)
        _id = int(kwargs.get("id", 1))
        _title = str(kwargs.get("title", "LAW14_COMPSO"))
    elif isinstance(rec, dict):
        base_params = rec.get("params", rec)
        p_in = {**base_params, **kwargs}
        _id = int(rec.get("id", kwargs.get("id", 1)))
        _title = str(rec.get("title", kwargs.get("title", "LAW14_COMPSO")))
    elif hasattr(rec, "params"):
        p_in = dict(rec.params) if isinstance(rec.params, dict) else {}
        if hasattr(rec, "density"):
            p_in.setdefault("density", getattr(rec, "density"))
            p_in.setdefault("rho0", getattr(rec, "density"))
        if hasattr(rec, "rho0"):
            p_in.setdefault("rho0", getattr(rec, "rho0"))
        p_in.update(kwargs)
        _id = int(getattr(rec, "id", kwargs.get("id", 1)))
        _title = str(getattr(rec, "title", kwargs.get("title", "LAW14_COMPSO")))
    elif hasattr(rec, "__dataclass_fields__"):
        base_dict = {
            k: getattr(rec, k) for k in rec.__dataclass_fields__ if hasattr(rec, k)
        }
        p_in = {**base_dict, **kwargs}
        _id = int(getattr(rec, "id", kwargs.get("id", 1)))
        _title = str(getattr(rec, "title", kwargs.get("title", "LAW14_COMPSO")))
    else:
        p_in = dict(kwargs)
        _id = int(kwargs.get("id", 1))
        _title = str(kwargs.get("title", "LAW14_COMPSO"))

    def _get(keys: Sequence[str], default: float = 0.0) -> float:
        for k in keys:
            if k in p_in and p_in[k] is not None:
                try:
                    return float(p_in[k])
                except (ValueError, TypeError):
                    pass
        return float(default)

    def _geti(keys: Sequence[str], default: int = 0) -> int:
        for k in keys:
            if k in p_in and p_in[k] is not None:
                try:
                    return int(p_in[k])
                except (ValueError, TypeError):
                    pass
        return int(default)

    # Card 1: rho0, refer_rho
    rho0 = _get(["MAT_RHO", "rho0", "density", "rho"], 0.0)
    refer_rho = _get(["Refer_Rho", "refer_rho", "RHO_O", "rhor", "rho_ref"], 0.0)
    if refer_rho <= 0.0:
        refer_rho = rho0

    # Card 2: E11, E22, E33
    e11 = _get(["MAT_EA", "E11", "e11", "ea", "MAT_E11"], 0.0)
    e22 = _get(["MAT_EB", "E22", "e22", "eb", "MAT_E22"], 0.0)
    e33 = _get(["MAT_EC", "E33", "e33", "ec", "MAT_E33"], 0.0)
    if e11 <= 0.0 or e22 <= 0.0 or e33 <= 0.0:
        raise ValueError(
            f"E11, E22, and E33 must be > 0 (got E11={e11}, E22={e22}, E33={e33})"
        )

    # Card 3: nu12, nu23, nu31
    nu12 = _get(["MAT_PRAB", "nu12", "NU12", "prab", "MAT_NU12"], 0.0)
    nu23 = _get(["MAT_PRBC", "nu23", "NU23", "prbc", "MAT_NU23"], 0.0)
    nu31 = _get(["MAT_PRCA", "nu31", "NU31", "prca", "MAT_NU31"], 0.0)

    # Card 4: G12, G23, G31
    g12 = _get(["MAT_GAB", "G12", "g12", "gab", "MAT_G12"], 0.0)
    g23 = _get(["MAT_GBC", "G23", "g23", "gbc", "MAT_G23"], 0.0)
    g31 = _get(["MAT_GCA", "G31", "g31", "gca", "MAT_G31"], 0.0)

    # Card 5: sigt1, sigt2, sigt3, delta
    sigt1 = _get(["MAT_SIGT1", "sigt1", "sig_t1", "sigma_t1", "SIGT1"], 0.0)
    sigt2 = _get(["MAT_SIGT2", "sigt2", "sig_t2", "sigma_t2", "SIGT2"], 0.0)
    sigt3 = _get(["MAT_SIGT3", "sigt3", "sig_t3", "sigma_t3", "SIGT3"], 0.0)
    delta = _get(["MAT_DAMAGE", "delta", "damage", "DELTA"], 0.05)

    if sigt1 <= 0.0:
        sigt1 = _INF
    if sigt2 <= 0.0:
        sigt2 = sigt1
    if sigt3 <= 0.0:
        sigt3 = sigt1
    if delta <= 0.0:
        delta = 0.05

    # Card 6: cb (b), cn (n), fmax, wplaref
    cb = _get(["MAT_BETA", "cb", "b", "B", "beta", "CB"], 0.0)
    cn = _get(["MAT_HARD", "cn", "n", "N", "hard", "CN"], 1.0)
    if cn <= 0.0:
        cn = 1.0
    fmax = _get(["MAT_SIG", "fmax", "FMAX", "sig", "sigmx", "sig_max"], 1.0e10)
    if fmax <= 0.0:
        fmax = 1.0e10
    fmax = max(1.0001, fmax)
    wplaref = _get(["WPREF", "wplaref", "wpref", "WPLAREF"], 1.0)
    if wplaref <= 0.0:
        wplaref = 1.0

    # Card 7: sigyt1, sigyt2, sigyc1, sigyc2
    sigyt1 = _get(["MAT_SIGYT1", "sigyt1", "sig_1yt", "sigma_1yt", "SIGYT1"], 0.0)
    sigyt2 = _get(["MAT_SIGYT2", "sigyt2", "sig_2yt", "sigma_2yt", "SIGYT2"], 0.0)
    sigyc1 = _get(["MAT_SIGYC1", "sigyc1", "sig_1yc", "sigma_1yc", "SIGYC1"], 0.0)
    sigyc2 = _get(["MAT_SIGYC2", "sigyc2", "sig_2yc", "sigma_2yc", "SIGYC2"], 0.0)

    if sigyt1 <= 0.0:
        sigyt1 = _INF
    if sigyc1 <= 0.0:
        sigyc1 = sigyt1
    if sigyt2 <= 0.0:
        sigyt2 = sigyt1
    if sigyc2 <= 0.0:
        sigyc2 = sigyc1

    # In plane 2-3, transverse isotropy:
    sigyt3 = sigyt2
    sigyc3 = sigyc2

    # Card 8: sigyt12, sigyc12, sigyt23, sigyc23
    sigyt12 = _get(["MAT_SIGT12", "sigyt12", "sigt12", "sig_12yt", "sigma_12yt", "SIGT12"], 0.0)
    sigyc12 = _get(["MAT_SIGC12", "sigyc12", "sigc12", "sig_12yc", "sigma_12yc", "SIGC12"], 0.0)
    sigyt23 = _get(["MAT_SIGT23", "sigyt23", "sigt23", "sig_23yt", "sigma_23yt", "SIGT23"], 0.0)
    sigyc23 = _get(["MAT_SIGC23", "sigyc23", "sigc23", "sig_23yc", "sigma_23yc", "SIGC23"], 0.0)

    if sigyt12 <= 0.0:
        sigyt12 = _INF
    if sigyc12 <= 0.0:
        sigyc12 = _INF
    if sigyt23 <= 0.0:
        sigyt23 = _INF
    if sigyc23 <= 0.0:
        sigyc23 = _INF

    # Transverse isotropy in shear: 13 matches 12
    sigyt13 = sigyt12
    sigyc13 = sigyc12

    # Card 9: alpha, efib, c, eps0, ICC
    alpha = _get(["MAT_ALPHA", "alpha", "ALPHA", "alpha_fib"], 0.0)
    if alpha >= 1.0:
        alpha = 0.999999999999
    efib = _get(["MAT_EFIB", "efib", "EFIB", "ef", "Ef", "e_fib"], 0.0)
    c_rate = _get(["MAT_SRC", "c", "C", "src", "cc", "CC"], 0.0)
    eps0 = _get(["MAT_SRP", "eps0", "srp", "eps_rate_0", "EPS_RATE_0", "EPS0"], 1.0)
    if eps0 <= 0.0 or c_rate == 0.0:
        eps0 = 1.0
    icc = _geti(["STRFLAG", "icc", "ICC", "strflag"], 1)
    if icc <= 0:
        icc = 1

    # Orthotropic compliance matrix inversion (hm_read_mat14.F:193-220)
    c11 = 1.0 / e11
    c22 = 1.0 / e22
    c33 = 1.0 / e33
    c12 = -nu12 / e11
    c13 = -nu31 / e33
    c23 = -nu23 / e22

    detc = (
        c11 * c22 * c33
        - c11 * (c23**2)
        - (c12**2) * c33
        + 2.0 * c12 * c13 * c23
        - (c13**2) * c22
    )
    if detc <= 0.0:
        raise ValueError(f"Compliance matrix determinant DETC must be > 0 (got {detc})")

    d11 = (c22 * c33 - c23**2) / detc
    d12 = -(c12 * c33 - c13 * c23) / detc
    d13 = (c12 * c23 - c13 * c22) / detc
    d22 = (c11 * c33 - c13**2) / detc
    d23 = -(c11 * c23 - c13 * c12) / detc
    d33 = (c11 * c22 - c12**2) / detc
    d21, d31, d32 = d12, d13, d23

    # Verification: A = C @ D = I (hm_read_mat14.F:222-227)
    a11 = c11 * d11 + c12 * d21 + c13 * d31
    a12 = c11 * d12 + c12 * d22 + c13 * d32
    a13 = c11 * d13 + c12 * d23 + c13 * d33
    a22 = c12 * d12 + c22 * d22 + c23 * d23
    a23 = c12 * d13 + c22 * d23 + c23 * d33
    a33 = c13 * d13 + c23 * d23 + c33 * d33

    # Timestep parameter PM105 check (hm_read_mat14.F:264)
    c1 = max(d11, d22, d33)
    dmin = min(d11 * d22 - d12**2, d22 * d33 - d23**2, d11 * d33 - d13**2)
    pm105 = dmin / (c1**2)

    # Sound speed (hm_read_mat14.F:238-239)
    r_ref = refer_rho if refer_rho > 0.0 else rho0
    ssp = math.sqrt(c1 / max(r_ref, _EM20))

    # Tsai-Wu yield constants (hm_read_mat14.F:278-290)
    # Transverse isotropy: F3 = F2, F33 = F22, F6 = F4, F66 = F44, F13 = F12, F23 = -0.5 / (sigyt2 * sigyc2)
    f1 = _inv(sigyt1) - _inv(sigyc1)
    f2 = _inv(sigyt2) - _inv(sigyc2)
    f3 = f2
    f4 = _inv(sigyt12) - _inv(sigyc12)
    f5 = _inv(sigyt23) - _inv(sigyc23)
    f6 = f4

    f11 = _inv_prod(sigyt1, sigyc1)
    f22 = _inv_prod(sigyt2, sigyc2)
    f33 = f22
    f44 = _inv_prod(sigyt12, sigyc12)
    f55 = _inv_prod(sigyt23, sigyc23)
    f66 = f44

    f12 = _tw_cross(sigyt1, sigyc1, sigyt2, sigyc2)
    f13 = f12
    if sigyt2 > 0.0 and sigyc2 > 0.0 and not math.isinf(sigyt2) and not math.isinf(sigyc2):
        f23 = -0.5 / (sigyt2 * sigyc2)
    else:
        f23 = _tw_cross(sigyt2, sigyc2, sigyt2, sigyc2)

    d_mat = np.array(
        [
            [d11, d12, d13, 0.0, 0.0, 0.0],
            [d12, d22, d23, 0.0, 0.0, 0.0],
            [d13, d23, d33, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, g12, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, g23, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, g31],
        ],
        dtype=float,
    )

    params: Dict[str, Any] = {
        # Card 1
        "rho": rho0,
        "MAT_RHO": rho0,
        "rho0": rho0,
        "Refer_Rho": refer_rho,
        "rhor": refer_rho,
        # Card 2
        "MAT_EA": e11,
        "E11": e11,
        "e11": e11,
        "ea": e11,
        "MAT_EB": e22,
        "E22": e22,
        "e22": e22,
        "eb": e22,
        "MAT_EC": e33,
        "E33": e33,
        "e33": e33,
        "ec": e33,
        # Card 3
        "MAT_PRAB": nu12,
        "NU12": nu12,
        "nu12": nu12,
        "prab": nu12,
        "MAT_PRBC": nu23,
        "NU23": nu23,
        "nu23": nu23,
        "prbc": nu23,
        "MAT_PRCA": nu31,
        "NU31": nu31,
        "nu31": nu31,
        "prca": nu31,
        # Card 4
        "MAT_GAB": g12,
        "G12": g12,
        "g12": g12,
        "gab": g12,
        "MAT_GBC": g23,
        "G23": g23,
        "g23": g23,
        "gbc": g23,
        "MAT_GCA": g31,
        "G31": g31,
        "g31": g31,
        "gca": g31,
        # Card 5
        "MAT_SIGT1": sigt1,
        "SIGT1": sigt1,
        "sigt1": sigt1,
        "sig_t1": sigt1,
        "MAT_SIGT2": sigt2,
        "SIGT2": sigt2,
        "sigt2": sigt2,
        "sig_t2": sigt2,
        "MAT_SIGT3": sigt3,
        "SIGT3": sigt3,
        "sigt3": sigt3,
        "sig_t3": sigt3,
        "MAT_DAMAGE": delta,
        "DELTA": delta,
        "delta": delta,
        "damage": delta,
        # Card 6
        "MAT_BETA": cb,
        "CB": cb,
        "cb": cb,
        "b": cb,
        "beta": cb,
        "MAT_HARD": cn,
        "CN": cn,
        "cn": cn,
        "n": cn,
        "hard": cn,
        "MAT_SIG": fmax,
        "FMAX": fmax,
        "fmax": fmax,
        "sigmx": fmax,
        "sig_max": fmax,
        "WPREF": wplaref,
        "WPLAREF": wplaref,
        "wplaref": wplaref,
        "wpref": wplaref,
        # Card 7
        "MAT_SIGYT1": sigyt1,
        "SIGYT1": sigyt1,
        "sigyt1": sigyt1,
        "sig_1yt": sigyt1,
        "MAT_SIGYT2": sigyt2,
        "SIGYT2": sigyt2,
        "sigyt2": sigyt2,
        "sig_2yt": sigyt2,
        "MAT_SIGYC1": sigyc1,
        "SIGYC1": sigyc1,
        "sigyc1": sigyc1,
        "sig_1yc": sigyc1,
        "MAT_SIGYC2": sigyc2,
        "SIGYC2": sigyc2,
        "sigyc2": sigyc2,
        "sig_2yc": sigyc2,
        # Card 8
        "MAT_SIGT12": sigyt12,
        "SIGYT12": sigyt12,
        "sigyt12": sigyt12,
        "sigt12": sigyt12,
        "sig_12yt": sigyt12,
        "MAT_SIGC12": sigyc12,
        "SIGYC12": sigyc12,
        "sigyc12": sigyc12,
        "sigc12": sigyc12,
        "sig_12yc": sigyc12,
        "MAT_SIGT23": sigyt23,
        "SIGYT23": sigyt23,
        "sigyt23": sigyt23,
        "sigt23": sigyt23,
        "sig_23yt": sigyt23,
        "MAT_SIGC23": sigyc23,
        "SIGYC23": sigyc23,
        "sigyc23": sigyc23,
        "sigc23": sigyc23,
        "sig_23yc": sigyc23,
        # Card 9
        "MAT_ALPHA": alpha,
        "ALPHA": alpha,
        "alpha": alpha,
        "alpha_fib": alpha,
        "MAT_EFIB": efib,
        "EFIB": efib,
        "efib": efib,
        "e_fib": efib,
        "MAT_SRC": c_rate,
        "src": c_rate,
        "c": c_rate,
        "cc": c_rate,
        "MAT_SRP": eps0,
        "srp": eps0,
        "eps0": eps0,
        "STRFLAG": icc,
        "ICC": icc,
        "icc": icc,
        "strflag": icc,
        # Compliance & Stiffness
        "c11": c11,
        "c22": c22,
        "c33": c33,
        "c12": c12,
        "c13": c13,
        "c23": c23,
        "DETC": detc,
        "detc": detc,
        "D11": d11,
        "d11": d11,
        "D12": d12,
        "d12": d12,
        "D13": d13,
        "d13": d13,
        "D22": d22,
        "d22": d22,
        "D23": d23,
        "d23": d23,
        "D33": d33,
        "d33": d33,
        "d_mat": d_mat,
        # Verification
        "A11": a11,
        "A12": a12,
        "A13": a13,
        "A22": a22,
        "A23": a23,
        "A33": a33,
        "C1": c1,
        "c1": c1,
        "DMIN": dmin,
        "dmin": dmin,
        "PM105": pm105,
        "pm105": pm105,
        "ssp": ssp,
        "SSP": ssp,
        # Tsai-Wu constants
        "F1": f1,
        "f1": f1,
        "F2": f2,
        "f2": f2,
        "F3": f3,
        "f3": f3,
        "F4": f4,
        "f4": f4,
        "F5": f5,
        "f5": f5,
        "F6": f6,
        "f6": f6,
        "F11": f11,
        "f11": f11,
        "F22": f22,
        "f22": f22,
        "F33": f33,
        "f33": f33,
        "F44": f44,
        "f44": f44,
        "F55": f55,
        "f55": f55,
        "F66": f66,
        "f66": f66,
        "F12": f12,
        "f12": f12,
        "F13": f13,
        "f13": f13,
        "F23": f23,
        "f23": f23,
        "E": e11,
        "nu": nu12,
    }

    mat = Material(
        id=_id,
        law=14,
        rho0=rho0,
        title=_title,
        law_name="LAW14",
        params=params,
    )
    mat.d_mat = d_mat
    mat.ssp = ssp
    mat.sound_speed_solid = lambda: ssp
    return mat


# ============================================================================
# Sound Speed & Shell Update
# ============================================================================

def sound_speed(mat: Any, rho: Optional[Any] = None, extra: Optional[Any] = None) -> Any:
    """Compute longitudinal sound speed for /MAT/LAW14 solids."""
    p = getattr(mat, "params", {}) or {}
    ssp = p.get("ssp", p.get("SSP"))
    if ssp is not None and rho is None:
        return float(ssp)
    d11 = float(p.get("D11", 0.0))
    d22 = float(p.get("D22", 0.0))
    d33 = float(p.get("D33", 0.0))
    c1 = max(d11, d22, d33)
    if c1 <= 0.0:
        c1 = float(p.get("C1", max(float(p.get("E11", 1.0)), 1.0)))
    r = (
        rho
        if (rho is not None and rho > 0.0)
        else float(getattr(mat, "rho0", 0.0) or p.get("rho0", 1.0))
    )
    if isinstance(r, np.ndarray):
        return np.sqrt(c1 / np.maximum(r, _EM20))
    return math.sqrt(c1 / max(float(r), _EM20))


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Any:
    """Raise error as LAW14 is solid-only."""
    raise NotImplementedError(
        "/MAT/LAW14 (COMPSO) is implemented for 3D solid elements only (ANCMSG 305)."
    )


# ============================================================================
# Extra Shapes (State Variables)
# ============================================================================

def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Per-element persistent state for /MAT/LAW14.

    State variables:
      - dam14: (5,) damage per direction (1, 2, 3, plastic work ratio, flags)
      - epe14: (3,) strain in crack directions
      - epc14: (3,) crack opening strain
      - wpla14: () plastic work
      - off14: () element status flag
      - epsf14: () fiber strain
      - sigf14: () fiber stress
      - tsaiwu14: () Tsai-Wu utilization factor
    """
    if nip is None or nip <= 1:
        return {
            "dam14": (5,),
            "epe14": (3,),
            "epc14": (3,),
            "wpla14": (),
            "off14": (),
            "epsf14": (),
            "sigf14": (),
            "tsaiwu14": (),
        }
    return {
        "dam14": (nip, 5),
        "epe14": (nip, 3),
        "epc14": (nip, 3),
        "wpla14": (nip,),
        "off14": (nip,),
        "epsf14": (nip,),
        "sigf14": (nip,),
        "tsaiwu14": (nip,),
    }


# ============================================================================
# Core Solid Constitutive Update (m14law.F)
# ============================================================================

def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, float | np.ndarray]:
    """3D solid constitutive update for LAW14 (/MAT/COMPSO).

    Fortran origin: engine/source/materials/mat/mat014/m14law.F
    """
    p = getattr(mat, "params", {}) or {}
    if "D11" not in p and "d_mat" not in p:
        mat_built = build_law14(mat)
        p = mat_built.params

    is_1d = (sig.ndim == 1)
    s_in = np.atleast_2d(sig).copy()
    d_in = np.atleast_2d(deps).copy()
    n = s_in.shape[0]

    if epsp is None:
        ep = np.zeros(n, dtype=float)
    else:
        ep = np.atleast_1d(epsp).astype(float).copy()
        if len(ep) == 1 and n > 1:
            ep = np.full(n, ep[0], dtype=float)

    if extra is None:
        extra = {}

    # State variables: support both 14-suffixed and generic keys
    dam = extra.get("dam14", extra.get("dam"))
    if dam is None:
        dam = np.zeros((n, 5), dtype=float)
    else:
        dam = np.atleast_2d(dam).astype(float).copy()
        if dam.shape[0] != n:
            dam = np.zeros((n, 5), dtype=float)

    epe = extra.get("epe14", extra.get("epe"))
    if epe is None:
        epe = np.zeros((n, 3), dtype=float)
    else:
        epe = np.atleast_2d(epe).astype(float).copy()
        if epe.shape[0] != n:
            epe = np.zeros((n, 3), dtype=float)

    epc = extra.get("epc14", extra.get("epc"))
    if epc is None:
        epc = np.zeros((n, 3), dtype=float)
    else:
        epc = np.atleast_2d(epc).astype(float).copy()
        if epc.shape[0] != n:
            epc = np.zeros((n, 3), dtype=float)

    wpla = extra.get("wpla14", extra.get("wpla"))
    if wpla is None:
        wpla = np.zeros(n, dtype=float)
    else:
        wpla = np.atleast_1d(wpla).astype(float).copy()
        if len(wpla) == 1 and n > 1:
            wpla = np.full(n, wpla[0], dtype=float)

    off = extra.get("off14", extra.get("off"))
    if off is None:
        off = np.ones(n, dtype=float)
    else:
        off = np.atleast_1d(off).astype(float).copy()
        if len(off) == 1 and n > 1:
            off = np.full(n, off[0], dtype=float)

    epsf = extra.get("epsf14", extra.get("epsf"))
    if epsf is None:
        epsf = np.zeros(n, dtype=float)
    else:
        epsf = np.atleast_1d(epsf).astype(float).copy()
        if len(epsf) == 1 and n > 1:
            epsf = np.full(n, epsf[0], dtype=float)

    sigf = extra.get("sigf14", extra.get("sigf"))
    if sigf is None:
        sigf = np.zeros(n, dtype=float)
    else:
        sigf = np.atleast_1d(sigf).astype(float).copy()
        if len(sigf) == 1 and n > 1:
            sigf = np.full(n, sigf[0], dtype=float)

    tsaiwu = extra.get("tsaiwu14", extra.get("tsaiwu"))
    if tsaiwu is None:
        tsaiwu = np.zeros(n, dtype=float)
    else:
        tsaiwu = np.atleast_1d(tsaiwu).astype(float).copy()
        if len(tsaiwu) == 1 and n > 1:
            tsaiwu = np.full(n, tsaiwu[0], dtype=float)

    # Coordinate triad vectors if supplied
    rx = extra.get("rx")
    ry = extra.get("ry")
    rz = extra.get("rz")
    sx = extra.get("sx")
    sy = extra.get("sy")
    sz = extra.get("sz")
    use_rot = rx is not None and ry is not None and rz is not None and sx is not None and sy is not None and sz is not None

    if use_rot:
        ax, ay, az, bx, by, bz, cx, cy, cz = m14ama(rx, ry, rz, sx, sy, sz)
        s_mat, d_mat = m14gtf(s_in, d_in, ax, ay, az, bx, by, bz, cx, cy, cz)
    else:
        s_mat = s_in
        d_mat = d_in

    s_out_mat = np.zeros_like(s_mat)
    ep_out = np.zeros(n, dtype=float)

    for i in range(n):
        s_i, ep_i, dam_i, epe_i, epc_i, wpla_i, off_i, epsf_i, sigf_i = _update_one_element(
            p,
            s_mat[i],
            d_mat[i],
            ep[i],
            dt,
            dam[i],
            epe[i],
            epc[i],
            wpla[i],
            off[i],
            epsf[i],
            sigf[i],
        )
        s_out_mat[i] = s_i
        ep_out[i] = ep_i
        dam[i] = dam_i
        epe[i] = epe_i
        epc[i] = epc_i
        wpla[i] = wpla_i
        off[i] = off_i
        epsf[i] = epsf_i
        sigf[i] = sigf_i

        # Compute Tsai-Wu utilization factor
        f1 = float(p["F1"])
        f2 = float(p["F2"])
        f3 = float(p["F3"])
        f4 = float(p["F4"])
        f5 = float(p["F5"])
        f6 = float(p["F6"])
        f11 = float(p["F11"])
        f22 = float(p["F22"])
        f33 = float(p["F33"])
        f44 = float(p["F44"])
        f55 = float(p["F55"])
        f66 = float(p["F66"])
        f12 = float(p["F12"])
        f23 = float(p["F23"])
        f13 = float(p["F13"])

        t1, t2, t3, t4, t5, t6 = s_i
        tw_val = (
            f1 * t1 + f2 * t2 + f3 * t3 + f4 * t4 + f5 * t5 + f6 * t6
            + f11 * (t1**2) + f22 * (t2**2) + f33 * (t3**2)
            + f44 * (t4**2) + f55 * (t5**2) + f66 * (t6**2)
            + 2.0 * f12 * t1 * t2 + 2.0 * f23 * t2 * t3 + 2.0 * f13 * t1 * t3
        )
        tsaiwu[i] = max(0.0, tw_val)

    if use_rot:
        s_out = m14ftg(s_out_mat, ax, ay, az, bx, by, bz, cx, cy, cz)
    else:
        s_out = s_out_mat

    # Store updated state back into extra dict
    for k_14, k_gen, val in (
        ("dam14", "dam", dam),
        ("epe14", "epe", epe),
        ("epc14", "epc", epc),
        ("wpla14", "wpla", wpla),
        ("off14", "off", off),
        ("epsf14", "epsf", epsf),
        ("sigf14", "sigf", sigf),
        ("tsaiwu14", "tsaiwu", tsaiwu),
    ):
        if k_14 in extra:
            extra[k_14] = val
        elif k_gen in extra:
            extra[k_gen] = val
        else:
            extra[k_14] = val

    c = sound_speed(mat)
    c_arr = np.full(n, c, dtype=float) if isinstance(c, (int, float)) else c

    if is_1d:
        return s_out[0], float(ep_out[0]), float(c_arr[0])
    return s_out, ep_out, c_arr


# ============================================================================
# Algorithmic Consistent Tangent Stiffness Matrix
# ============================================================================

def consistent_solid_tangent(
    mat: Any,
    sig: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    deps: Optional[np.ndarray] = None,
    symmetric: bool = True,
    h: float = 1.0e-7,
    epsp_incr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return 6x6 algorithmic consistent tangent stiffness tensor.

    In the elastic unyielding regime, returns the exact orthotropic D matrix.
    In the damaged or plastic regimes, computes the consistent tangent via
    central finite difference perturbation of _update_one_element.
    """
    p = getattr(mat, "params", {}) or {}
    if "D11" not in p and "d_mat" not in p:
        mat_built = build_law14(mat)
        p = mat_built.params

    is_1d = (sig.ndim == 1)
    s_in = np.atleast_2d(sig).copy()
    n = s_in.shape[0]

    if deps is None:
        deps_in = np.zeros((n, 6), dtype=float)
    else:
        deps_in = np.atleast_2d(deps).astype(float).copy()
        if deps_in.shape[0] != n:
            deps_in = np.zeros((n, 6), dtype=float)

    if epsp is None:
        ep = np.zeros(n, dtype=float)
    else:
        ep = np.atleast_1d(epsp).astype(float).copy()
        if len(ep) == 1 and n > 1:
            ep = np.full(n, ep[0], dtype=float)

    if extra is None:
        extra = {}

    dam = extra.get("dam14", extra.get("dam"))
    if dam is None:
        dam = np.zeros((n, 5), dtype=float)
    else:
        dam = np.atleast_2d(dam).astype(float).copy()
        if dam.shape[0] != n:
            dam = np.zeros((n, 5), dtype=float)

    epe = extra.get("epe14", extra.get("epe"))
    if epe is None:
        epe = np.zeros((n, 3), dtype=float)
    else:
        epe = np.atleast_2d(epe).astype(float).copy()
        if epe.shape[0] != n:
            epe = np.zeros((n, 3), dtype=float)

    epc = extra.get("epc14", extra.get("epc"))
    if epc is None:
        epc = np.zeros((n, 3), dtype=float)
    else:
        epc = np.atleast_2d(epc).astype(float).copy()
        if epc.shape[0] != n:
            epc = np.zeros((n, 3), dtype=float)

    wpla = extra.get("wpla14", extra.get("wpla"))
    if wpla is None:
        wpla = np.zeros(n, dtype=float)
    else:
        wpla = np.atleast_1d(wpla).astype(float).copy()
        if len(wpla) == 1 and n > 1:
            wpla = np.full(n, wpla[0], dtype=float)

    off = extra.get("off14", extra.get("off"))
    if off is None:
        off = np.ones(n, dtype=float)
    else:
        off = np.atleast_1d(off).astype(float).copy()
        if len(off) == 1 and n > 1:
            off = np.full(n, off[0], dtype=float)

    epsf = extra.get("epsf14", extra.get("epsf"))
    if epsf is None:
        epsf = np.zeros(n, dtype=float)
    else:
        epsf = np.atleast_1d(epsf).astype(float).copy()
        if len(epsf) == 1 and n > 1:
            epsf = np.full(n, epsf[0], dtype=float)

    sigf = extra.get("sigf14", extra.get("sigf"))
    if sigf is None:
        sigf = np.zeros(n, dtype=float)
    else:
        sigf = np.atleast_1d(sigf).astype(float).copy()
        if len(sigf) == 1 and n > 1:
            sigf = np.full(n, sigf[0], dtype=float)

    d11 = float(p["D11"])
    d12 = float(p["D12"])
    d13 = float(p["D13"])
    d22 = float(p["D22"])
    d23 = float(p["D23"])
    d33 = float(p["D33"])
    g12 = float(p["G12"])
    g23 = float(p["G23"])
    g31 = float(p["G31"])

    D_mat = np.array(
        [
            [d11, d12, d13, 0.0, 0.0, 0.0],
            [d12, d22, d23, 0.0, 0.0, 0.0],
            [d13, d23, d33, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, g12, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, g23, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, g31],
        ],
        dtype=float,
    )

    tangents = np.zeros((n, 6, 6), dtype=float)

    for i in range(n):
        s_i = s_in[i].copy()
        dam_i = dam[i].copy()
        off_i = off[i]

        # Element completely degraded/deleted
        if off_i == 0.0:
            tangents[i] = 0.0
            continue

        # Check if element is in pure elastic undamaged regime
        is_elastic = (
            dam_i[0] == 0.0
            and dam_i[1] == 0.0
            and dam_i[2] == 0.0
            and dam_i[3] == 0.0
            and wpla[i] == 0.0
            and ep[i] == 0.0
            and np.allclose(deps_in[i], 0.0)
            and np.allclose(s_i, 0.0)
        )

        if is_elastic:
            tangents[i] = D_mat.copy()
            continue

        # Numerical perturbation: d(sigma) / d(deps)
        base_d = deps_in[i].copy()
        for j in range(6):
            d_p = base_d.copy()
            d_m = base_d.copy()
            d_p[j] += h
            d_m[j] -= h

            s_p, _, _, _, _, _, _, _, _ = _update_one_element(
                p,
                s_i.copy(),
                d_p,
                ep[i],
                dt,
                dam_i.copy(),
                epe[i].copy(),
                epc[i].copy(),
                wpla[i],
                off_i,
                epsf[i],
                sigf[i],
            )
            s_m, _, _, _, _, _, _, _, _ = _update_one_element(
                p,
                s_i.copy(),
                d_m,
                ep[i],
                dt,
                dam_i.copy(),
                epe[i].copy(),
                epc[i].copy(),
                wpla[i],
                off_i,
                epsf[i],
                sigf[i],
            )

            tangents[i, :, j] = (s_p - s_m) / (2.0 * h)

        if symmetric:
            tangents[i] = 0.5 * (tangents[i] + tangents[i].T)

    if is_1d:
        return tangents[0]
    return tangents


# ============================================================================
# Registry Hook
# ============================================================================

def _register() -> None:
    """Register LAW14 in pyradioss MAT_PHYSICS_REGISTRY."""
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY

        for key in (
            14,
            "14",
            "LAW14",
            "COMPSO",
            "COMP_SOL",
            "MAT_LAW14",
            "MAT_COMPSO",
            "MAT_COMP_SOL",
        ):
            MAT_PHYSICS_REGISTRY[key] = build_law14
    except Exception:
        pass


_register()

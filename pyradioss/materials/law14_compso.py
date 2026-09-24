"""
pyradioss.materials.law14_compso — /MAT/LAW14 (/MAT/COMPSO, /MAT/COMP_SOL).

3D Orthotropic Elastic-Plastic Composite Material for Solid Elements and
Progressive Damage Multi-Layer Composite Shells.

Fortran origin:
  - Starter reader: starter/source/materials/mat/mat014/hm_read_mat14.F
  - Shell kernel:   engine/source/materials/mat/mat014/sigeps14c.F
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
    if sigyt2 > 0.0 and sigyc2 > 0.0 and not math.isinf(sigyt2) and not math.isinf(sigyc2) and sigyt2 < 1.0e29 and sigyc2 < 1.0e29:
        f23 = -0.5 / (sigyt2 * sigyc2)
    else:
        f23 = 0.0

    ft1 = f11 * f22 - 4.0 * (f12**2)
    ft2 = (f22**2) - 4.0 * (f23**2)

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
        "C11": c11,
        "c11": c11,
        "C22": c22,
        "c22": c22,
        "C33": c33,
        "c33": c33,
        "C12": c12,
        "c12": c12,
        "C13": c13,
        "c13": c13,
        "C23": c23,
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
        "FT1": ft1,
        "ft1": ft1,
        "FT2": ft2,
        "ft2": ft2,
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


def _rotate_shell_to_mat(
    sig: np.ndarray, deps: np.ndarray, angle_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate in-plane shell stress and strain increments to layer material frame.

    Fortran origin: engine/source/materials/mat/mat014/sigeps14c.F
    """
    if abs(angle_deg) < 1.0e-9:
        return sig.copy(), deps.copy()
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    c2 = c * c
    s2 = s * s
    cs = c * s

    # Strains: [de11, de22, dgamma12]
    d_mat = np.zeros_like(deps)
    d_mat[0] = c2 * deps[0] + s2 * deps[1] + cs * deps[2]
    d_mat[1] = s2 * deps[0] + c2 * deps[1] - cs * deps[2]
    d_mat[2] = -2.0 * cs * deps[0] + 2.0 * cs * deps[1] + (c2 - s2) * deps[2]

    # Stresses: [s11, s22, s12]
    s_mat = np.zeros_like(sig)
    s_mat[0] = c2 * sig[0] + s2 * sig[1] + 2.0 * cs * sig[2]
    s_mat[1] = s2 * sig[0] + c2 * sig[1] - 2.0 * cs * sig[2]
    s_mat[2] = -cs * sig[0] + cs * sig[1] + (c2 - s2) * sig[2]

    if len(sig) >= 5 and len(deps) >= 5:
        # Transverse shears: [..., dgamma23, dgamma31]
        d_mat[3] = c * deps[3] - s * deps[4]
        d_mat[4] = s * deps[3] + c * deps[4]
        s_mat[3] = c * sig[3] - s * sig[4]
        s_mat[4] = s * sig[3] + c * sig[4]

    return s_mat, d_mat


def _rotate_mat_to_shell(sig_mat: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate in-plane layer stress back to shell local coordinate system.

    Fortran origin: engine/source/materials/mat/mat014/sigeps14c.F
    """
    if abs(angle_deg) < 1.0e-9:
        return sig_mat.copy()
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    c2 = c * c
    s2 = s * s
    cs = c * s

    s_shell = np.zeros_like(sig_mat)
    s_shell[0] = c2 * sig_mat[0] + s2 * sig_mat[1] - 2.0 * cs * sig_mat[2]
    s_shell[1] = s2 * sig_mat[0] + c2 * sig_mat[1] + 2.0 * cs * sig_mat[2]
    s_shell[2] = cs * sig_mat[0] - cs * sig_mat[1] + (c2 - s2) * sig_mat[2]

    if len(sig_mat) >= 5:
        s_shell[3] = c * sig_mat[3] + s * sig_mat[4]
        s_shell[4] = -s * sig_mat[3] + c * sig_mat[4]

    return s_shell


def _update_point_law14_shell(
    p: Dict[str, Any],
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: float,
    dt: float,
    dam: np.ndarray,
    epe: np.ndarray,
    epc: np.ndarray,
    wpla: float,
    off: float,
    epsf: float,
    sigf: float,
    angle_deg: float = 0.0,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """Single integration point plane-stress constitutive update.

    Fortran origin:
      - engine/source/materials/mat/mat014/sigeps14c.F
      - engine/source/materials/mat/mat014/m14law.F
    """
    has_5 = (len(sig) >= 5)

    # 1. Rotate to fiber frame
    s_mat, d_mat = _rotate_shell_to_mat(sig, deps, angle_deg)

    # 2. Decode damage flags (dam[4] encodes kd1..kd4, offset 10000)
    dam5 = float(dam[4]) if len(dam) >= 5 else 0.0
    if dam5 >= 10000.0:
        idam = int(dam5) - 10000
        kd1 = idam // 1000
        kdx = idam - kd1 * 1000
        kd2 = kdx // 100
        kdx = kdx - kd2 * 100
        kd3 = kdx // 10
        kd4 = kdx - kd3 * 10
    else:
        kd1, kd2, kd3, kd4 = 0, 0, 0, 0

    dam1 = float(dam[0]) if len(dam) >= 1 else 0.0
    dam2 = float(dam[1]) if len(dam) >= 2 else 0.0
    epe1 = float(epe[0]) if len(epe) >= 1 else 0.0
    epe2 = float(epe[1]) if len(epe) >= 2 else 0.0
    epc1 = float(epc[0]) if len(epc) >= 1 else 0.0
    epc2 = float(epc[1]) if len(epc) >= 2 else 0.0

    # 3. In-plane elastic moduli
    e11 = float(p.get("E11", p.get("MAT_EA", 1.0)))
    e22 = float(p.get("E22", p.get("MAT_EB", 1.0)))
    nu12 = float(p.get("nu12", p.get("MAT_PRAB", 0.0)))
    g12 = float(p.get("G12", p.get("MAT_GAB", 0.0)))
    g23 = float(p.get("G23", p.get("MAT_GBC", 0.0)))
    g31 = float(p.get("G31", p.get("MAT_GCA", 0.0)))

    nu21 = nu12 * e22 / max(e11, _EM20) if e11 > 0.0 else 0.0
    detc = max(1.0e-15, 1.0 - nu12 * nu21)
    q11 = e11 / detc
    q22 = e22 / detc
    q12 = nu12 * e22 / detc

    # 4. Damage parameters & yield limits
    sigt1 = float(p.get("sigt1", p.get("SIGT1", _INF)))
    sigt2 = float(p.get("sigt2", p.get("SIGT2", _INF)))
    delta = float(p.get("delta", p.get("DELTA", 0.05)))
    cb = float(p.get("cb", p.get("MAT_BETA", 0.0)))
    cn = float(p.get("cn", p.get("MAT_HARD", 1.0)))
    fmax = float(p.get("fmax", p.get("MAT_SIG", _INF)))
    wplaref = float(p.get("wplaref", p.get("WPREF", 1.0)))

    # Strain rate scaling (m14law.F:176-195)
    c_rate = float(p.get("c", p.get("MAT_SRC", 0.0)))
    eps0 = float(p.get("eps0", p.get("MAT_SRP", 1.0)))
    icc = int(p.get("icc", p.get("STRFLAG", 1)))

    dt1 = max(dt, 1.0e-12) if dt > 0.0 else 1.0
    de1, de2, de4 = d_mat[0], d_mat[1], d_mat[2]
    rate_ep = max(abs(de1), abs(de2), abs(de4)) / dt1
    if rate_ep > eps0 and c_rate != 0.0:
        epsp_fac = 1.0 + c_rate * math.log(rate_ep / max(eps0, 1.0e-15))
    else:
        epsp_fac = 1.0

    ca = 1.0 * epsp_fac
    cb_eff = cb * epsp_fac
    if icc in (1, 3):
        sigmx = fmax * epsp_fac
    else:
        sigmx = fmax

    pw = wpla if wpla > 0.0 else 0.0
    sigmy = min(sigmx, ca + cb_eff * (pw ** cn))
    if sigmy >= sigmx and off == 1.0:
        off = 0.99
        kd4 = 2

    # 5. Strain increments & crack strain
    epe1 += de1
    epe2 += de2

    # Elastic trial stresses
    so1, so2, so4 = s_mat[0], s_mat[1], s_mat[2]
    t1 = so1 + q11 * de1 + q12 * de2
    t2 = so2 + q12 * de1 + q22 * de2
    t4 = so4 + g12 * de4
    if has_5:
        so5, so6 = s_mat[3], s_mat[4]
        t5 = so5 + g23 * d_mat[3]
        t6 = so6 + g31 * d_mat[4]
    else:
        so5, so6, t5, t6 = 0.0, 0.0, 0.0, 0.0

    # 6. Direction 1 tensile cracking (fiber)
    wvec1 = (1.0 - dam1) * sigt1
    if t1 > wvec1:
        if epc1 == 0.0:
            epc1 = max(epe1, 0.0)
        else:
            epc1 = max(epc1 + de1, 0.0)
        t1 = wvec1
        t2 -= q12 * de1 * dam1
        if kd1 == 0:
            kd1 = 1
        dam1 = min(dam1 + delta, 1.0)
        if dam1 >= 1.0 and kd1 != 2:
            kd1 = 2
    if de1 < 0.0 and dam1 > 0.0:
        epc1 = max(epc1 + de1, 0.0)

    # 7. Direction 2 tensile cracking (matrix)
    wvec2 = (1.0 - dam2) * sigt2
    if t2 > wvec2:
        if epc2 == 0.0:
            epc2 = max(epe2, 0.0)
        else:
            epc2 = max(epc2 + de2, 0.0)
        t1 -= q12 * de2 * dam2
        t2 = wvec2
        if kd2 == 0:
            kd2 = 1
        dam2 = min(dam2 + delta, 1.0)
        if dam2 >= 1.0 and kd2 != 2:
            kd2 = 2
    if de2 < 0.0 and dam2 > 0.0:
        epc2 = max(epc2 + de2, 0.0)

    # 8. Crack open -> no compression condition (m14law.F:433-449)
    if t1 < 0.0 and epc1 > 0.0:
        t1 = 0.0
        t2 -= q12 * de1 * dam1
    if t2 < 0.0 and epc2 > 0.0:
        t1 -= q12 * de2 * dam2
        t2 = 0.0

    # 9. Tsai-Wu yield evaluation (plane stress)
    f1 = float(p.get("F1", 0.0))
    f2 = float(p.get("F2", 0.0))
    f11 = float(p.get("F11", 0.0))
    f22 = float(p.get("F22", 0.0))
    f44 = float(p.get("F44", 0.0))
    f12 = float(p.get("F12", 0.0))

    wvec = f1 * t1 + f2 * t2 + f11 * (t1**2) + f22 * (t2**2) + 2.0 * f12 * t1 * t2 + f44 * (t4**2)
    if has_5:
        f55 = float(p.get("F55", 0.0))
        f66 = float(p.get("F66", f44))
        wvec += f55 * (t5**2) + f66 * (t6**2)

    # 10. Plasticity flow & return mapping (m14law.F:463-547)
    if wvec > sigmy and off == 1.0:
        if kd4 == 0:
            kd4 = 1
        dp1 = f1 + 2.0 * f11 * so1 + 2.0 * f12 * so2
        dp2 = f2 + 2.0 * f22 * so2 + 2.0 * f12 * so1
        dp4 = 2.0 * f44 * so4
        ds1 = t1 - so1
        ds2 = t2 - so2
        ds4 = t4 - so4

        num = dp1 * ds1 + dp2 * ds2 + dp4 * ds4
        if has_5:
            dp5 = 2.0 * f55 * so5
            dp6 = 2.0 * f66 * so6
            ds5 = t5 - so5
            ds6 = t6 - so6
            num += dp5 * ds5 + dp6 * ds6
        else:
            dp5, dp6 = 0.0, 0.0

        plas = 1.0 if wpla <= 0.0 else (wpla ** (cn - 1.0))
        denom = (
            dp1 * (q11 * dp1 + q12 * dp2)
            + dp2 * (q12 * dp1 + q22 * dp2)
            + 2.0 * dp4 * g12 * dp4
            + (so1 * dp1 + so2 * dp2 + 2.0 * so4 * dp4) * cn * cb_eff * plas
        )
        if has_5:
            denom += 2.0 * dp5 * g23 * dp5 + 2.0 * dp6 * g31 * dp6
            denom += (2.0 * so5 * dp5 + 2.0 * so6 * dp6) * cn * cb_eff * plas

        if denom > 1.0e-20 and num > 0.0:
            lam = num / denom
            dp1_lam = lam * dp1
            dp2_lam = lam * dp2
            dp4_lam = lam * dp4
            epe1 -= dp1_lam
            epe2 -= dp2_lam
            t1 -= q11 * dp1_lam + q12 * dp2_lam
            t2 -= q12 * dp1_lam + q22 * dp2_lam
            t4 -= 2.0 * g12 * dp4_lam

            dwpla = 0.5 * (
                dp1_lam * (t1 + so1)
                + dp2_lam * (t2 + so2)
                + 2.0 * dp4_lam * (t4 + so4)
            )
            if has_5:
                dp5_lam = lam * dp5
                dp6_lam = lam * dp6
                t5 -= 2.0 * g23 * dp5_lam
                t6 -= 2.0 * g31 * dp6_lam
                dwpla += 0.5 * (
                    2.0 * dp5_lam * (t5 + so5)
                    + 2.0 * dp6_lam * (t6 + so6)
                )
            wpla += max(0.0, dwpla) / max(wplaref, 1.0e-15)

    # 11. Fiber strain and stress
    efib = float(p.get("efib", p.get("MAT_EFIB", 0.0)))
    if efib > 0.0:
        epsf += de1
        sigf = efib * epsf

    # 12. Damage & failure degradation
    if off < 0.1:
        off = 0.0
    elif off < 1.0:
        off = off * 0.8

    if dam1 >= 1.0 or dam2 >= 1.0 or sigmy >= sigmx:
        if off == 1.0:
            off = 0.99

    t1 *= off
    t2 *= off
    t4 *= off
    if has_5:
        t5 *= off
        t6 *= off
        s_res = np.array([t1, t2, t4, t5, t6], dtype=float)
    else:
        s_res = np.array([t1, t2, t4], dtype=float)

    # 13. Rotate back to shell local axes
    s_out = _rotate_mat_to_shell(s_res, angle_deg)

    # 14. Pack state arrays
    dam_code = kd1 * 1000 + kd2 * 100 + kd3 * 10 + kd4 + 10000
    dam_out = np.array([dam1, dam2, 0.0, wvec, float(dam_code)], dtype=float)
    epe_out = np.array([epe1, epe2, 0.0], dtype=float)
    epc_out = np.array([epc1, epc2, 0.0], dtype=float)

    return s_out, wpla, dam_out, epe_out, epc_out, off, epsf, sigf, wvec


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, float | np.ndarray]:
    """Plane-stress (shell) constitutive update for LAW14 (/MAT/COMPSO).

    Fortran origin:
      - engine/source/materials/mat/mat014/sigeps14c.F
      - engine/source/materials/mat/mat014/m14law.F

    Parameters:
      mat: Material instance with LAW14 parameters
      sig: In-plane stress array (n, 3) = [s11, s22, s12] or (n, 5) or 1D
      deps: Strain increment array (n, 3) = [de11, de22, dgamma12] or (n, 5) or 1D
      epsp: Optional plastic strain array (n,) or scalar
      dt: Simulation time increment
      extra: Optional dict containing persistent state arrays:
             - 'dam14': (n, 5) damage per direction and flags
             - 'epe14': (n, 3) strain in crack directions
             - 'epc14': (n, 3) crack opening strain
             - 'wpla14': (n,) accumulated plastic work
             - 'off14': (n,) active status flag (1.0 active, 0.0 failed)
             - 'angle': layer orientation angle (degrees)
             - 'layers': list of layers for multi-layer composite shells

    Returns:
      (sig_new, epsp_new, sound_speed)
    """
    p = getattr(mat, "params", {}) or {}
    if "D11" not in p and "d_mat" not in p:
        mat_built = build_law14(mat)
        p = mat_built.params

    if extra is None:
        extra = {}

    # Check for multi-layer composite laminate integration
    layers = extra.get("layers", extra.get("plies"))
    if layers is not None and isinstance(layers, (list, tuple)) and len(layers) > 0:
        return multilayer_shell_update(
            mat=mat,
            sig=sig,
            deps=deps,
            epsp=epsp,
            dt=dt,
            extra=extra,
        )

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

    # State variables
    dam = extra.get("dam14", extra.get("dam"))
    if dam is None or np.asarray(dam).shape != (n, 5):
        dam = np.zeros((n, 5), dtype=float)
    else:
        dam = np.atleast_2d(dam).astype(float).copy()

    epe = extra.get("epe14", extra.get("epe"))
    if epe is None or np.asarray(epe).shape != (n, 3):
        epe = np.zeros((n, 3), dtype=float)
    else:
        epe = np.atleast_2d(epe).astype(float).copy()

    epc = extra.get("epc14", extra.get("epc"))
    if epc is None or np.asarray(epc).shape != (n, 3):
        epc = np.zeros((n, 3), dtype=float)
    else:
        epc = np.atleast_2d(epc).astype(float).copy()

    wpla = extra.get("wpla14", extra.get("wpla"))
    if wpla is None or len(np.atleast_1d(wpla)) != n:
        wpla = np.zeros(n, dtype=float)
    else:
        wpla = np.atleast_1d(wpla).astype(float).copy()

    off = extra.get("off14", extra.get("off"))
    if off is None or len(np.atleast_1d(off)) != n:
        off = np.ones(n, dtype=float)
    else:
        off = np.atleast_1d(off).astype(float).copy()

    epsf = extra.get("epsf14", extra.get("epsf"))
    if epsf is None or len(np.atleast_1d(epsf)) != n:
        epsf = np.zeros(n, dtype=float)
    else:
        epsf = np.atleast_1d(epsf).astype(float).copy()

    sigf = extra.get("sigf14", extra.get("sigf"))
    if sigf is None or len(np.atleast_1d(sigf)) != n:
        sigf = np.zeros(n, dtype=float)
    else:
        sigf = np.atleast_1d(sigf).astype(float).copy()

    tsaiwu = extra.get("tsaiwu14", extra.get("tsaiwu"))
    if tsaiwu is None or len(np.atleast_1d(tsaiwu)) != n:
        tsaiwu = np.zeros(n, dtype=float)
    else:
        tsaiwu = np.atleast_1d(tsaiwu).astype(float).copy()

    # Orientation angle
    angle_val = extra.get("angle", extra.get("theta", extra.get("phi", 0.0)))
    if isinstance(angle_val, (int, float)):
        angles = np.full(n, float(angle_val), dtype=float)
    else:
        angles = np.asarray(angle_val, dtype=float).flatten()
        if len(angles) != n:
            angles = np.zeros(n, dtype=float)

    s_out = np.zeros_like(s_in)
    ep_out = np.zeros(n, dtype=float)

    for i in range(n):
        s_i, wpla_i, dam_i, epe_i, epc_i, off_i, epsf_i, sigf_i, tw_i = _update_point_law14_shell(
            p=p,
            sig=s_in[i],
            deps=d_in[i],
            epsp=ep[i],
            dt=dt,
            dam=dam[i],
            epe=epe[i],
            epc=epc[i],
            wpla=wpla[i],
            off=off[i],
            epsf=epsf[i],
            sigf=sigf[i],
            angle_deg=angles[i],
        )
        s_out[i] = s_i
        wpla[i] = wpla_i
        ep_out[i] = wpla_i
        dam[i] = dam_i
        epe[i] = epe_i
        epc[i] = epc_i
        off[i] = off_i
        epsf[i] = epsf_i
        sigf[i] = sigf_i
        tsaiwu[i] = tw_i

    # Store back in extra
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
        extra[k_14] = val
        extra[k_gen] = val

    c = sound_speed(mat)
    c_arr = np.full(n, float(c), dtype=float)

    if epsp is not None and isinstance(epsp, np.ndarray):
        epsp[:] = ep_out.reshape(epsp.shape)

    if is_1d:
        return s_out[0], float(ep_out[0]), float(c_arr[0])
    return s_out, ep_out, c_arr


def multilayer_shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, float | np.ndarray]:
    """Through-thickness multi-layer composite shell integration for LAW14.

    Fortran origin: engine/source/materials/mat/mat014/sigeps14c.F

    Evaluates layer constitutive responses for composite laminate shells with:
      - arbitrary layer thicknesses (t_k)
      - arbitrary fiber orientation angles (theta_k)
      - mid-plane strain increments (deps) and optional curvature increments (dkappa)
      - calculates resultant membrane forces (N) and bending moments (M)
      - computes homogenized average membrane stresses (s_avg = N / H)
    """
    p = getattr(mat, "params", {}) or {}
    if extra is None:
        extra = {}

    layers = extra.get("layers", extra.get("plies", []))
    n_layers = len(layers)
    if n_layers == 0:
        return shell_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)

    # Thickness and angle extraction
    thicknesses = np.zeros(n_layers, dtype=float)
    angles = np.zeros(n_layers, dtype=float)
    for k, lay in enumerate(layers):
        if isinstance(lay, dict):
            thicknesses[k] = float(lay.get("thick", lay.get("thickness", lay.get("t", 1.0))))
            angles[k] = float(lay.get("angle", lay.get("theta", lay.get("phi", 0.0))))
        elif isinstance(lay, (list, tuple)):
            thicknesses[k] = float(lay[0])
            angles[k] = float(lay[1]) if len(lay) > 1 else 0.0
        else:
            thicknesses[k] = getattr(lay, "thick", getattr(lay, "thickness", 1.0))
            angles[k] = getattr(lay, "angle", 0.0)

    total_thickness = float(np.sum(thicknesses))
    if total_thickness <= 0.0:
        total_thickness = 1.0

    # Layer centroids z_k relative to mid-plane
    z_coords = np.zeros(n_layers, dtype=float)
    current_z = -0.5 * total_thickness
    for k in range(n_layers):
        z_coords[k] = current_z + 0.5 * thicknesses[k]
        current_z += thicknesses[k]

    # Strains
    dkappa = extra.get("dkappa", extra.get("curv", np.zeros_like(deps)))
    dkappa_arr = np.asarray(dkappa, dtype=float)

    # Layer state retrieval or initialization
    dam_layers = extra.get("dam14_layers", extra.get("layer_dam"))
    if dam_layers is None or np.asarray(dam_layers).shape != (n_layers, 5):
        dam_layers = np.zeros((n_layers, 5), dtype=float)
    else:
        dam_layers = np.copy(dam_layers)

    epe_layers = extra.get("epe14_layers", np.zeros((n_layers, 3), dtype=float))
    epc_layers = extra.get("epc14_layers", np.zeros((n_layers, 3), dtype=float))
    wpla_layers = extra.get("wpla14_layers", np.zeros(n_layers, dtype=float))
    off_layers = extra.get("off14_layers", np.ones(n_layers, dtype=float))
    epsf_layers = extra.get("epsf14_layers", np.zeros(n_layers, dtype=float))
    sigf_layers = extra.get("sigf14_layers", np.zeros(n_layers, dtype=float))

    sig_layers = extra.get("layer_stresses")
    if sig_layers is None or np.asarray(sig_layers).shape[0] != n_layers:
        n_comp = len(sig) if sig.ndim == 1 else sig.shape[-1]
        sig_layers = np.zeros((n_layers, n_comp), dtype=float)
        if sig.ndim == 1 and np.any(sig != 0.0):
            for k in range(n_layers):
                sig_layers[k] = sig.copy()
    else:
        sig_layers = np.copy(sig_layers)

    new_sig_layers = np.zeros_like(sig_layers)
    N_res = np.zeros(sig_layers.shape[1], dtype=float)
    M_res = np.zeros(sig_layers.shape[1], dtype=float)

    for k in range(n_layers):
        # Layer strain at centroid
        zk = z_coords[k]
        deps_k = np.asarray(deps, dtype=float) + zk * dkappa_arr

        s_k, wpla_k, dam_k, epe_k, epc_k, off_k, epsf_k, sigf_k, _ = _update_point_law14_shell(
            p=p,
            sig=sig_layers[k],
            deps=deps_k,
            epsp=wpla_layers[k],
            dt=dt,
            dam=dam_layers[k],
            epe=epe_layers[k],
            epc=epc_layers[k],
            wpla=wpla_layers[k],
            off=off_layers[k],
            epsf=epsf_layers[k],
            sigf=sigf_layers[k],
            angle_deg=angles[k],
        )
        new_sig_layers[k] = s_k
        wpla_layers[k] = wpla_k
        dam_layers[k] = dam_k
        epe_layers[k] = epe_k
        epc_layers[k] = epc_k
        off_layers[k] = off_k
        epsf_layers[k] = epsf_k
        sigf_layers[k] = sigf_k

        # Through-thickness integration of stress resultants
        tk = thicknesses[k]
        N_res += s_k * tk
        M_res += s_k * zk * tk

    # Homogenized average stress
    s_avg = N_res / total_thickness

    # Save layer state in extra
    extra["layer_stresses"] = new_sig_layers
    extra["dam14_layers"] = dam_layers
    extra["epe14_layers"] = epe_layers
    extra["epc14_layers"] = epc_layers
    extra["wpla14_layers"] = wpla_layers
    extra["off14_layers"] = off_layers
    extra["epsf14_layers"] = epsf_layers
    extra["sigf14_layers"] = sigf_layers
    extra["N"] = N_res
    extra["M"] = M_res
    extra["total_thickness"] = total_thickness

    c = sound_speed(mat)
    return s_avg, float(np.mean(wpla_layers)), float(c)


def shell_membrane_tangent(mat: Any, extra: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """(3, 3) orthotropic plane-stress elastic tangent matrix for shells.

    Accounts for layer orientation angle and damage states if provided.
    Fortran origin: engine/source/materials/mat/mat014/sigeps14c.F
    """
    p = getattr(mat, "params", {}) or {}
    e11 = float(p.get("E11", p.get("MAT_EA", 1.0)))
    e22 = float(p.get("E22", p.get("MAT_EB", 1.0)))
    nu12 = float(p.get("nu12", p.get("MAT_PRAB", 0.0)))
    g12 = float(p.get("G12", p.get("MAT_GAB", 0.0)))

    nu21 = nu12 * e22 / max(e11, _EM20) if e11 > 0.0 else 0.0
    detc = max(1.0e-15, 1.0 - nu12 * nu21)
    q11 = e11 / detc
    q22 = e22 / detc
    q12 = nu12 * e22 / detc

    off_val = 1.0
    dam1, dam2 = 0.0, 0.0
    angle_deg = 0.0

    if extra is not None and isinstance(extra, dict):
        for off_k in ("off14", "off"):
            if off_k in extra and extra[off_k] is not None:
                val = np.asarray(extra[off_k], dtype=float).flatten()
                if len(val) > 0:
                    off_val = float(val[0])
                break
        for dam_k in ("dam14", "dam"):
            if dam_k in extra and extra[dam_k] is not None:
                val = np.asarray(extra[dam_k], dtype=float)
                if val.ndim == 1 and len(val) >= 2:
                    dam1, dam2 = float(val[0]), float(val[1])
                elif val.ndim == 2 and val.shape[1] >= 2:
                    dam1, dam2 = float(val[0, 0]), float(val[0, 1])
                break
        angle_deg = float(extra.get("angle", extra.get("theta", 0.0)))

    if off_val <= 0.0:
        return np.zeros((3, 3), dtype=float)

    # Directional degradation
    scale1 = max(0.0, 1.0 - dam1) * off_val
    scale2 = max(0.0, 1.0 - dam2) * off_val

    Q_mat = np.array([
        [q11 * scale1, q12 * math.sqrt(scale1 * scale2), 0.0],
        [q12 * math.sqrt(scale1 * scale2), q22 * scale2, 0.0],
        [0.0, 0.0, g12 * math.sqrt(scale1 * scale2)],
    ], dtype=float)

    if abs(angle_deg) < 1.0e-9:
        return Q_mat

    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    c2 = c * c
    s2 = s * s
    cs = c * s

    T_sig_inv = np.array([
        [c2, s2, -2.0 * cs],
        [s2, c2, 2.0 * cs],
        [cs, -cs, c2 - s2],
    ], dtype=float)

    T_eps = np.array([
        [c2, s2, cs],
        [s2, c2, -cs],
        [-2.0 * cs, 2.0 * cs, c2 - s2],
    ], dtype=float)

    return T_sig_inv @ Q_mat @ T_eps


def consistent_shell_tangent(
    mat: Any,
    sig: np.ndarray,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    symmetric: bool = True,
) -> np.ndarray:
    """Numerical consistent shell tangent matrix by finite difference perturbation."""
    if deps is None:
        deps = np.zeros_like(sig)

    is_1d = (sig.ndim == 1)
    s = np.atleast_2d(sig).copy()
    d = np.atleast_2d(deps).copy()
    n = s.shape[0]
    n_comp = s.shape[1]

    h = 1.0e-7
    tangents = np.zeros((n, n_comp, n_comp), dtype=float)

    for i in range(n):
        for j in range(n_comp):
            d_p = d[i].copy()
            d_m = d[i].copy()
            d_p[j] += h
            d_m[j] -= h

            extra_p = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in (extra or {}).items()}
            extra_m = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in (extra or {}).items()}

            s_p, _, _ = shell_update(mat, s[i].copy(), d_p, epsp=epsp, dt=dt, extra=extra_p)
            s_m, _, _ = shell_update(mat, s[i].copy(), d_m, epsp=epsp, dt=dt, extra=extra_m)

            diff = s_p - s_m
            if np.all(np.isfinite(diff)):
                tangents[i, :, j] = diff / (2.0 * h)
            else:
                tangents[i, :, j] = 0.0

        np.nan_to_num(tangents[i], copy=False)
        if symmetric:
            tangents[i] = 0.5 * (tangents[i] + tangents[i].T)

    if is_1d:
        return tangents[0]
    return tangents


shell_tangent = consistent_shell_tangent


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

    # Coordinate rotation: support both axes rotation matrix and triad vectors
    axes = extra.get("axes")
    rx = extra.get("rx")
    ry = extra.get("ry")
    rz = extra.get("rz")
    sx = extra.get("sx")
    sy = extra.get("sy")
    sz = extra.get("sz")
    use_triad = rx is not None and ry is not None and rz is not None and sx is not None and sy is not None and sz is not None

    use_rot = False
    if axes is not None:
        axes_arr = np.asarray(axes, dtype=float)
        if axes_arr.ndim == 2 and axes_arr.shape == (n, 9):
            use_rot = True
            ax, ay, az = axes_arr[:, 0], axes_arr[:, 1], axes_arr[:, 2]
            bx, by, bz = axes_arr[:, 3], axes_arr[:, 4], axes_arr[:, 5]
            cx, cy, cz = axes_arr[:, 6], axes_arr[:, 7], axes_arr[:, 8]
        elif axes_arr.ndim == 3 and axes_arr.shape == (n, 3, 3):
            use_rot = True
            ax, ay, az = axes_arr[:, 0, 0], axes_arr[:, 0, 1], axes_arr[:, 0, 2]
            bx, by, bz = axes_arr[:, 1, 0], axes_arr[:, 1, 1], axes_arr[:, 1, 2]
            cx, cy, cz = axes_arr[:, 2, 0], axes_arr[:, 2, 1], axes_arr[:, 2, 2]
        elif axes_arr.ndim == 2 and axes_arr.shape == (3, 3):
            use_rot = True
            ax = np.full(n, axes_arr[0, 0])
            ay = np.full(n, axes_arr[0, 1])
            az = np.full(n, axes_arr[0, 2])
            bx = np.full(n, axes_arr[1, 0])
            by = np.full(n, axes_arr[1, 1])
            bz = np.full(n, axes_arr[1, 2])
            cx = np.full(n, axes_arr[2, 0])
            cy = np.full(n, axes_arr[2, 1])
            cz = np.full(n, axes_arr[2, 2])
    elif use_triad:
        use_rot = True
        ax, ay, az, bx, by, bz, cx, cy, cz = m14ama(rx, ry, rz, sx, sy, sz)

    if use_rot:
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
    epsp_incr: Any = None,
    deps: Optional[np.ndarray] = None,
    symmetric: bool = False,
    h: float = 1.0e-7,
    **kwargs: Any,
) -> np.ndarray:
    """Return (n, 6, 6) or (6, 6) algorithmic consistent tangent stiffness tensor for LAW14.

    In the elastic unyielding regime, returns the exact orthotropic D matrix.
    In the damaged or plastic regimes, computes the consistent tangent via
    central finite difference perturbation of _update_one_element.
    """
    if deps is None and "deps" in kwargs:
        deps = kwargs["deps"]
    if "symmetric" in kwargs and kwargs["symmetric"] is not None:
        symmetric = bool(kwargs["symmetric"])
    if "h" in kwargs and kwargs["h"] is not None:
        h = float(kwargs["h"])

    # If deps not provided as keyword, check if epsp_incr was passed as strain increment
    if deps is None and epsp_incr is not None:
        incr_arr = np.asarray(epsp_incr, dtype=float)
        if (incr_arr.ndim == 1 and incr_arr.shape[0] == 6) or (incr_arr.ndim == 2 and incr_arr.shape[1] == 6):
            deps = incr_arr

    p = getattr(mat, "params", {}) or {}
    if "D11" not in p and "d_mat" not in p:
        mat_built = build_law14(mat)
        p = mat_built.params

    sig_arr = np.asarray(sig, dtype=float)
    is_1d = (sig_arr.ndim == 1)
    sig_2d = sig_arr[None, :].copy() if is_1d else sig_arr.copy()
    n = sig_2d.shape[0]
    if n == 0:
        return np.empty((0, 6, 6), dtype=float)

    if deps is None:
        deps_2d = np.zeros((n, 6), dtype=float)
    else:
        deps_arr = np.asarray(deps, dtype=float)
        deps_2d = deps_arr[None, :].copy() if deps_arr.ndim == 1 else deps_arr.copy()
        if deps_2d.shape[0] != n:
            if deps_2d.shape[0] == 1 and n > 1:
                deps_2d = np.repeat(deps_2d, n, axis=0)
            else:
                deps_2d = np.zeros((n, 6), dtype=float)

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

    # Coordinate triad vectors if supplied
    rx = extra.get("rx")
    ry = extra.get("ry")
    rz = extra.get("rz")
    sx = extra.get("sx")
    sy = extra.get("sy")
    sz = extra.get("sz")
    use_triad = (
        rx is not None and ry is not None and rz is not None
        and sx is not None and sy is not None and sz is not None
    )

    axes = extra.get("axes", extra.get("frame", extra.get("A")))
    has_rot = False
    if axes is not None:
        axes_arr = np.asarray(axes, dtype=float)
        if axes_arr.ndim == 2 and axes_arr.shape == (n, 9):
            has_rot = True
            ax, ay, az = axes_arr[:, 0], axes_arr[:, 1], axes_arr[:, 2]
            bx, by, bz = axes_arr[:, 3], axes_arr[:, 4], axes_arr[:, 5]
            cx, cy, cz = axes_arr[:, 6], axes_arr[:, 7], axes_arr[:, 8]
        elif axes_arr.ndim == 3 and axes_arr.shape == (n, 3, 3):
            has_rot = True
            ax, ay, az = axes_arr[:, 0, 0], axes_arr[:, 0, 1], axes_arr[:, 0, 2]
            bx, by, bz = axes_arr[:, 1, 0], axes_arr[:, 1, 1], axes_arr[:, 1, 2]
            cx, cy, cz = axes_arr[:, 2, 0], axes_arr[:, 2, 1], axes_arr[:, 2, 2]
        elif axes_arr.ndim == 2 and axes_arr.shape == (3, 3):
            has_rot = True
            ax = np.full(n, axes_arr[0, 0])
            ay = np.full(n, axes_arr[0, 1])
            az = np.full(n, axes_arr[0, 2])
            bx = np.full(n, axes_arr[1, 0])
            by = np.full(n, axes_arr[1, 1])
            bz = np.full(n, axes_arr[1, 2])
            cx = np.full(n, axes_arr[2, 0])
            cy = np.full(n, axes_arr[2, 1])
            cz = np.full(n, axes_arr[2, 2])
    elif use_triad:
        has_rot = True
        ax, ay, az, bx, by, bz, cx, cy, cz = m14ama(rx, ry, rz, sx, sy, sz)

    if has_rot:
        s_mat, d_mat = m14gtf(
            sig_2d,
            deps_2d,
            ax, ay, az, bx, by, bz, cx, cy, cz,
        )
    else:
        s_mat = sig_2d
        d_mat = deps_2d

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
        off_i = off[i]

        # Element completely degraded or below cutoff
        if off_i <= 0.0 or off_i < 0.1:
            tangents[i] = 0.0
            continue

        base_d = d_mat[i]
        s_i = s_mat[i]
        dam_i = dam[i].copy()

        is_damaged = (
            dam_i[0] > 0.0
            or dam_i[1] > 0.0
            or dam_i[2] > 0.0
            or epc[i, 0] > 0.0
            or epc[i, 1] > 0.0
            or epc[i, 2] > 0.0
        )

        # Trial stresses under base_d
        t1 = s_i[0] + d11 * base_d[0] + d12 * base_d[1] + d13 * base_d[2]
        t2 = s_i[1] + d12 * base_d[0] + d22 * base_d[1] + d23 * base_d[2]
        t3 = s_i[2] + d13 * base_d[0] + d23 * base_d[1] + d33 * base_d[2]
        t4 = s_i[3] + g12 * base_d[3]
        t5 = s_i[4] + g23 * base_d[4]
        t6 = s_i[5] + g31 * base_d[5]

        sigt1 = float(p.get("sigt1", 0.0))
        sigt2 = float(p.get("sigt2", 0.0))
        sigt3 = float(p.get("sigt3", 0.0))
        will_crack = (
            (sigt1 > 0.0 and t1 > sigt1)
            or (sigt2 > 0.0 and t2 > sigt2)
            or (sigt3 > 0.0 and t3 > sigt3)
        )

        f1 = float(p.get("F1", 0.0))
        f2 = float(p.get("F2", 0.0))
        f3 = float(p.get("F3", 0.0))
        f4 = float(p.get("F4", 0.0))
        f5 = float(p.get("F5", 0.0))
        f6 = float(p.get("F6", 0.0))
        f11 = float(p.get("F11", 0.0))
        f22 = float(p.get("F22", 0.0))
        f33 = float(p.get("F33", 0.0))
        f44 = float(p.get("F44", 0.0))
        f55 = float(p.get("F55", 0.0))
        f66 = float(p.get("F66", 0.0))
        f12 = float(p.get("F12", 0.0))
        f23 = float(p.get("F23", 0.0))
        f13 = float(p.get("F13", 0.0))

        wvec_tw = (
            f1 * t1
            + f2 * t2
            + f3 * t3
            + f4 * t4
            + f5 * t5
            + f6 * t6
            + f11 * (t1**2)
            + f22 * (t2**2)
            + f33 * (t3**2)
            + f44 * (t4**2)
            + f55 * (t5**2)
            + f66 * (t6**2)
            + 2.0 * f12 * t1 * t2
            + 2.0 * f13 * t1 * t3
            + 2.0 * f23 * t2 * t3
        )

        cb_val = float(p.get("cb", 0.0))
        cn_val = float(p.get("cn", 1.0))
        fmax_val = float(p.get("fmax", 1.0e10))
        c_rate = float(p.get("c", 0.0))
        eps0 = float(p.get("eps0", 0.0))
        icc = int(p.get("ICC", 0))

        if dt > 0.0:
            epsp_rate = max(
                abs(base_d[0] / dt),
                abs(base_d[1] / dt),
                abs(base_d[2] / dt),
                0.5 * abs(base_d[3] / dt),
                0.5 * abs(base_d[4] / dt),
                0.5 * abs(base_d[5] / dt),
            )
        else:
            epsp_rate = 0.0

        if epsp_rate > eps0 and c_rate > 0.0 and eps0 > 0.0:
            rate_fac = 1.0 + c_rate * math.log(epsp_rate / eps0)
        else:
            rate_fac = 1.0

        sigmx = (fmax_val * rate_fac) if icc in (1, 3) else fmax_val
        cb_eff = cb_val * rate_fac
        ca_eff = 1.0 * rate_fac
        wpla_term = (wpla[i] ** cn_val) if wpla[i] > 0.0 else 0.0
        sigmy = min(sigmx, ca_eff + cb_eff * wpla_term)

        # Check if element is in pure elastic undamaged regime
        if not has_rot and not is_damaged and not will_crack and wvec_tw < sigmy and off_i == 1.0:
            tangents[i] = D_mat.copy()
            if symmetric:
                tangents[i] = 0.5 * (tangents[i] + tangents[i].T)
            continue

        # Numerical perturbation: d(sigma) / d(deps)
        for j in range(6):
            if has_rot:
                d_p_g = deps_2d[i].copy()
                d_m_g = deps_2d[i].copy()
                d_p_g[j] += h
                d_m_g[j] -= h

                _, d_p_m = m14gtf(
                    sig_2d[i : i + 1],
                    d_p_g[None, :],
                    ax[i : i + 1], ay[i : i + 1], az[i : i + 1],
                    bx[i : i + 1], by[i : i + 1], bz[i : i + 1],
                    cx[i : i + 1], cy[i : i + 1], cz[i : i + 1],
                )
                _, d_m_m = m14gtf(
                    sig_2d[i : i + 1],
                    d_m_g[None, :],
                    ax[i : i + 1], ay[i : i + 1], az[i : i + 1],
                    bx[i : i + 1], by[i : i + 1], bz[i : i + 1],
                    cx[i : i + 1], cy[i : i + 1], cz[i : i + 1],
                )

                s_p_m, _, _, _, _, _, _, _, _ = _update_one_element(
                    p,
                    s_i.copy(),
                    d_p_m[0],
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
                s_m_m, _, _, _, _, _, _, _, _ = _update_one_element(
                    p,
                    s_i.copy(),
                    d_m_m[0],
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

                s_p = m14ftg(
                    s_p_m[None, :],
                    ax[i : i + 1], ay[i : i + 1], az[i : i + 1],
                    bx[i : i + 1], by[i : i + 1], bz[i : i + 1],
                    cx[i : i + 1], cy[i : i + 1], cz[i : i + 1],
                )[0]
                s_m = m14ftg(
                    s_m_m[None, :],
                    ax[i : i + 1], ay[i : i + 1], az[i : i + 1],
                    bx[i : i + 1], by[i : i + 1], bz[i : i + 1],
                    cx[i : i + 1], cy[i : i + 1], cz[i : i + 1],
                )[0]
            else:
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

            diff = s_p - s_m
            if np.all(np.isfinite(diff)):
                tangents[i, :, j] = diff / (2.0 * h)
            else:
                tangents[i, :, j] = 0.0

        np.nan_to_num(tangents[i], copy=False)
        if symmetric:
            tangents[i] = 0.5 * (tangents[i] + tangents[i].T)

    if is_1d:
        return tangents[0]
    return tangents


solid_tangent = consistent_solid_tangent


def tangent(group: Any = None, sig: Optional[np.ndarray] = None, **kwargs: Any) -> Optional[np.ndarray]:
    """Elemental / group tangent interface compliance for LAW14.

    Dispatches to shell_membrane_tangent for shell elements or consistent_solid_tangent
    for 3D solid elements.
    """
    if group is None:
        return None
    mat = getattr(group, "mat", group)
    elem_type = getattr(group, "elem_type", getattr(group, "type", "solid"))
    if "shell" in str(elem_type).lower():
        if sig is None:
            sig = np.zeros(3, dtype=float)
        try:
            return shell_membrane_tangent(mat, **kwargs)
        except Exception:
            return None
    if sig is None:
        sig = np.zeros(6, dtype=float)
    try:
        return solid_tangent(mat, sig, **kwargs)
    except Exception:
        return None


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

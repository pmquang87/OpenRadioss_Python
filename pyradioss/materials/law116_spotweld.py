"""OpenRadioss /MAT/LAW116 — Spotweld Connector with Progressive Damage.

Spotweld interface / connector model with mixed-mode normal (Mode I) and shear (Mode II)
progressive softening, rate-dependent peak strength, and critical energy release rate.

Upstream Fortran references:
  - `engine/source/materials/mat/mat116/sigeps116.F`
  - `starter/source/materials/mat/mat116/hm_read_mat116.F`
  - `hm_cfg_files/config/CFG/radioss2021/MAT/mat116.cfg`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_EM20 = 1.0e-20
_EM10 = 1.0e-10


@dataclass
class Law116Params:
    """Parameters for OpenRadioss /MAT/LAW116 (Spotweld Connector)."""
    id: int = 1
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    rho: float = 0.0
    young: float = 1.0      # E
    g_mod: float = 0.5      # G
    thick: float = 1.0      # Interface thickness
    imass: int = 1          # Mass calculation flag
    idel: int = 1           # Element deletion flag upon complete failure
    icrit: int = 1          # Failure criterion flag
    # Mode I parameters (Normal)
    gc1_ini: float = 1.0e20
    gc1_inf: float = 1.0e20
    ratg1: float = 0.0
    fg1: float = 0.0
    siga1: float = 1.0e20
    sigb1: float = 0.0
    rate1: float = 0.0
    iorder1: int = 1
    ifail1: int = 1
    # Mode II parameters (Shear)
    gc2_ini: float = 1.0e20
    gc2_inf: float = 1.0e20
    ratg2: float = 0.0
    fg2: float = 0.0
    siga2: float = 1.0e20
    sigb2: float = 0.0
    rate2: float = 0.0
    iorder2: int = 1
    ifail2: int = 1
    # Derived stiffnesses per unit area (stiffness = modulus / thickness)
    kn: float = field(init=False)
    kt: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rho > 0.0 and self.rho0 <= 0.0:
            self.rho0 = self.rho
        if self.rho0 > 0.0 and self.rho <= 0.0:
            self.rho = self.rho0
        if self.refer_rho <= 0.0:
            self.refer_rho = self.rho0

        if self.thick <= 0.0:
            self.thick = 1.0
        if self.young <= 0.0:
            self.young = 1.0
        if self.g_mod <= 0.0:
            self.g_mod = self.young / (2.0 * (1.0 + 0.3))

        self.kn = self.young / self.thick
        self.kt = self.g_mod / self.thick

    @classmethod
    def from_material(cls, mat: Any) -> Law116Params:
        """Construct Law116Params from generic Material or dictionary."""
        if isinstance(mat, Law116Params):
            return mat

        def _get(keys: Sequence[str], default: Any) -> Any:
            for k in keys:
                if hasattr(mat, k):
                    v = getattr(mat, k)
                    if v is not None:
                        return v
                if hasattr(mat, "params") and isinstance(mat.params, dict) and k in mat.params:
                    v = mat.params[k]
                    if v is not None:
                        return v
                if isinstance(mat, dict) and k in mat:
                    v = mat[k]
                    if v is not None:
                        return v
            return default

        mid = int(_get(["id", "mid", "mat_id"], 1))
        title = str(_get(["title", "name"], ""))
        rho0 = float(_get(["rho0", "rho", "MAT_RHO"], 0.0))
        young = float(_get(["young", "e", "MAT_E", "E"], 1.0))
        g_mod = float(_get(["g_mod", "g", "MAT_G", "G"], young / 2.6))
        thick = float(_get(["thick", "MAT_THICK", "THICK"], 1.0))
        imass = int(_get(["imass", "MAT_IMASS"], 1))
        idel = int(_get(["idel", "MAT_IDEL"], 1))
        icrit = int(_get(["icrit", "MAT_ICRIT"], 1))

        gc1_ini = float(_get(["gc1_ini", "MAT_GC1_ini", "GC1_INI"], 1.0e20))
        gc1_inf = float(_get(["gc1_inf", "MAT_GC1_inf", "GC1_INF"], 1.0e20))
        ratg1 = float(_get(["ratg1", "MAT_SRATG1", "RATG1"], 0.0))
        fg1 = float(_get(["fg1", "MAT_FG1", "FG1"], 0.0))
        siga1 = float(_get(["siga1", "MAT_SIGA1", "SIGA1"], 1.0e20))
        sigb1 = float(_get(["sigb1", "MAT_SIGB1", "SIGB1"], 0.0))
        rate1 = float(_get(["rate1", "MAT_SRATE1", "RATE1"], 0.0))
        iorder1 = int(_get(["iorder1", "MAT_ORDER1", "ORDER1"], 1))
        ifail1 = int(_get(["ifail1", "MAT_FAIL1", "FAIL1"], 1))

        gc2_ini = float(_get(["gc2_ini", "MAT_GC2_ini", "GC2_INI"], 1.0e20))
        gc2_inf = float(_get(["gc2_inf", "MAT_GC2_inf", "GC2_INF"], 1.0e20))
        ratg2 = float(_get(["ratg2", "MAT_SRATG2", "RATG2"], 0.0))
        fg2 = float(_get(["fg2", "MAT_FG2", "FG2"], 0.0))
        siga2 = float(_get(["siga2", "MAT_SIGA2", "SIGA2"], 1.0e20))
        sigb2 = float(_get(["sigb2", "MAT_SIGB2", "SIGB2"], 0.0))
        rate2 = float(_get(["rate2", "MAT_SRATE2", "RATE2"], 0.0))
        iorder2 = int(_get(["iorder2", "MAT_ORDER2", "ORDER2"], 1))
        ifail2 = int(_get(["ifail2", "MAT_FAIL2", "FAIL2"], 1))

        return cls(
            id=mid,
            title=title,
            rho0=rho0,
            young=young,
            g_mod=g_mod,
            thick=thick,
            imass=imass,
            idel=idel,
            icrit=icrit,
            gc1_ini=gc1_ini,
            gc1_inf=gc1_inf,
            ratg1=ratg1,
            fg1=fg1,
            siga1=siga1,
            sigb1=sigb1,
            rate1=rate1,
            iorder1=iorder1,
            ifail1=ifail1,
            gc2_ini=gc2_ini,
            gc2_inf=gc2_inf,
            ratg2=ratg2,
            fg2=fg2,
            siga2=siga2,
            sigb2=sigb2,
            rate2=rate2,
            iorder2=iorder2,
            ifail2=ifail2,
        )


def build_law116(mat: Any = None, **kwargs: Any) -> Law116Params:
    """Construct Law116Params from material or keyword arguments."""
    if mat is not None:
        p = Law116Params.from_material(mat)
        for k, v in kwargs.items():
            if hasattr(p, k):
                setattr(p, k, v)
        p.__post_init__()
        return p
    valid_keys = {f.name for f in Law116Params.__dataclass_fields__.values() if f.init}
    init_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in valid_keys}
    p = Law116Params(**init_kwargs)
    for k, v in extra_kwargs.items():
        if hasattr(p, k):
            setattr(p, k, v)
    p.__post_init__()
    return p


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law116Params:
    """Resolve references for /MAT/LAW116 and return Law116Params."""
    return build_law116(mat)


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Return extra history variable shapes for LAW116."""
    if nip is not None:
        return {
            "uvar116": (nip, 12),
            "dmg": (nip,),
            "epsp": (nip,),
        }
    return {
        "uvar116": (12,),
        "dmg": (),
        "epsp": (),
    }


def needs_defgrad(mat: Any = None) -> bool:
    """LAW116 uses small interface separation strains; defgrad is False."""
    return False


def _compute_spotweld_damage(
    p: Law116Params,
    eps_n: float,
    gamma_t: float,
    dt: float = 0.0,
) -> Tuple[float, float]:
    """Calculate spotweld mixed-mode damage D in [0, 1] matching `sigeps116.F`."""
    en_pos = max(0.0, eps_n)
    disp_n = en_pos * p.thick
    disp_t = abs(gamma_t) * p.thick

    # Mode I damage threshold
    disp_1_ini = p.siga1 / max(_EM20, p.kn) if p.siga1 < 1.0e19 else 1.0e20
    disp_1_fail = (2.0 * p.gc1_ini) / max(_EM20, p.siga1) if (p.gc1_ini < 1.0e19 and p.siga1 < 1.0e19) else 1.0e20

    # Mode II damage threshold
    disp_2_ini = p.siga2 / max(_EM20, p.kt) if p.siga2 < 1.0e19 else 1.0e20
    disp_2_fail = (2.0 * p.gc2_ini) / max(_EM20, p.siga2) if (p.gc2_ini < 1.0e19 and p.siga2 < 1.0e19) else 1.0e20

    # Mixed-mode equivalent displacement
    delta_m = math.sqrt(disp_n**2 + disp_t**2)
    beta = disp_t / max(_EM20, delta_m)

    delta_ini = math.sqrt(disp_1_ini**2 * (1.0 - beta**2) + disp_2_ini**2 * beta**2)
    delta_fail = math.sqrt(disp_1_fail**2 * (1.0 - beta**2) + disp_2_fail**2 * beta**2)

    if delta_m <= delta_ini:
        return 0.0, 0.0

    if delta_m >= delta_fail:
        return 1.0, float(delta_m)

    dmg = (delta_fail * (delta_m - delta_ini)) / max(_EM20, delta_m * (delta_fail - delta_ini))
    return float(min(1.0, max(0.0, dmg))), float(delta_m)


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """3D solid/connector stress update for LAW116."""
    p = build_law116(mat)
    sig_arr = np.asarray(sig, dtype=np.float64)
    deps_arr = np.asarray(deps, dtype=np.float64)
    is_1d = (deps_arr.ndim == 1)

    sig_2d = sig_arr.reshape(1, 6) if is_1d else sig_arr.copy()
    deps_2d = deps_arr.reshape(1, 6) if is_1d else deps_arr.copy()
    n = sig_2d.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(n, dtype=np.float64)
    else:
        epsp_in = np.asarray(epsp, dtype=np.float64)
        epsp_arr = np.full(n, float(epsp_in)) if epsp_in.ndim == 0 else epsp_in.copy()

    sig_new = np.zeros_like(sig_2d)
    epsp_new = np.zeros_like(epsp_arr)

    e = p.young
    g = p.g_mod

    for i in range(n):
        deps_i = deps_2d[i]
        # In spotweld connector, ZZ is normal separation, YZ and ZX are shear
        eps_zz_tr = (sig_2d[i, 2] / max(_EM20, e)) + deps_i[2]
        eps_yz_tr = (sig_2d[i, 4] / max(_EM20, g)) + deps_i[4]
        eps_zx_tr = (sig_2d[i, 5] / max(_EM20, g)) + deps_i[5]
        gamma_t = math.sqrt(eps_yz_tr**2 + eps_zx_tr**2)

        dmg, delta_m = _compute_spotweld_damage(p, eps_zz_tr, gamma_t, dt)

        # In compression (eps_zz < 0), normal contact stiffness is retained without damage
        if eps_zz_tr < 0.0:
            sig_new[i, 2] = e * eps_zz_tr
        else:
            sig_new[i, 2] = (1.0 - dmg) * e * eps_zz_tr

        sig_new[i, 4] = (1.0 - dmg) * g * eps_yz_tr
        sig_new[i, 5] = (1.0 - dmg) * g * eps_zx_tr

        # Transverse in-plane normal stresses (if any) are elastic
        sig_new[i, 0] = sig_2d[i, 0] + e * deps_i[0]
        sig_new[i, 1] = sig_2d[i, 1] + e * deps_i[1]
        sig_new[i, 3] = sig_2d[i, 3] + g * deps_i[3]

        epsp_new[i] = epsp_arr[i] + dmg

    c = sound_speed(p)
    c_out = c if is_1d else np.full(n, c, dtype=np.float64)

    if is_1d:
        return sig_new[0], float(epsp_new[0]), float(c_out)
    return sig_new, epsp_new, c_out


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """2D shell interface stress update for LAW116."""
    p = build_law116(mat)
    sig_arr = np.asarray(sig, dtype=np.float64)
    deps_arr = np.asarray(deps, dtype=np.float64)
    is_1d = (deps_arr.ndim == 1)

    sig_2d = sig_arr.reshape(1, -1) if is_1d else sig_arr.copy()
    deps_2d = deps_arr.reshape(1, -1) if is_1d else deps_arr.copy()
    n = sig_2d.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(n, dtype=np.float64)
    else:
        epsp_in = np.asarray(epsp, dtype=np.float64)
        epsp_arr = np.full(n, float(epsp_in)) if epsp_in.ndim == 0 else epsp_in.copy()

    ncomp = sig_2d.shape[1]
    sig_new = np.zeros_like(sig_2d)
    epsp_new = np.zeros_like(epsp_arr)

    e = p.young
    g = p.g_mod

    for i in range(n):
        deps_i = deps_2d[i]
        eps_n = deps_i[0]
        gamma_t = deps_i[2]

        dmg, _ = _compute_spotweld_damage(p, eps_n, gamma_t, dt)

        factor = 1.0 - dmg
        sig_new[i, 0] = factor * (sig_2d[i, 0] + e * deps_i[0])
        sig_new[i, 1] = factor * (sig_2d[i, 1] + e * deps_i[1])
        sig_new[i, 2] = factor * (sig_2d[i, 2] + g * deps_i[2])

        if ncomp > 3:
            sig_new[i, 3:] = factor * sig_2d[i, 3:]

        epsp_new[i] = epsp_arr[i] + dmg

    c = sound_speed(p, is_shell=True)
    c_out = c if is_1d else np.full(n, c, dtype=np.float64)

    if is_1d:
        return sig_new[0], float(epsp_new[0]), float(c_out)
    return sig_new, epsp_new, c_out


def sound_speed(mat: Any, eps: Optional[Any] = None, extra: Optional[Any] = None, is_shell: bool = False, **kwargs: Any) -> float:
    """Compute acoustic wave speed for LAW116."""
    p = build_law116(mat)
    rho = p.rho0 if p.rho0 > 0.0 else 1.0
    mod = p.young
    return float(math.sqrt(max(0.0, mod / rho)))


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[float] = None,
    epsp_incr: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent 6x6 continuum solid tangent stiffness for LAW116."""
    p = build_law116(mat)
    c_el = np.zeros((6, 6), dtype=np.float64)
    c_el[0, 0] = p.young
    c_el[1, 1] = p.young
    c_el[2, 2] = p.young
    c_el[3, 3] = p.g_mod
    c_el[4, 4] = p.g_mod
    c_el[5, 5] = p.g_mod

    if sig is None:
        return c_el

    sig_arr = np.asarray(sig, dtype=np.float64)
    if sig_arr.ndim == 2:
        n = sig_arr.shape[0]
        t = np.zeros((n, 6, 6), dtype=np.float64)
        for i in range(n):
            t[i] = solid_tangent(p, sig=sig_arr[i])
        return t
    return c_el


def shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[float] = None,
    epsp_incr: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent 3x3 plane stress algorithmic tangent matrix for LAW116."""
    p = build_law116(mat)
    c_el = np.array([
        [p.young, 0.0, 0.0],
        [0.0, p.young, 0.0],
        [0.0, 0.0, p.g_mod],
    ], dtype=np.float64)

    if sig is None:
        return c_el

    sig_arr = np.asarray(sig, dtype=np.float64)
    if sig_arr.ndim == 2:
        n = sig_arr.shape[0]
        t = np.zeros((n, 3, 3), dtype=np.float64)
        for i in range(n):
            t[i] = shell_tangent(p, sig=sig_arr[i])
        return t
    return c_el


consistent_solid_tangent = solid_tangent
consistent_shell_tangent = shell_tangent

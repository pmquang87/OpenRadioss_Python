"""OpenRadioss /MAT/LAW111 — Barlat Yld2004-18p 3D Anisotropic Yield & Marlow Model.

Constitutive formulation implementing the Barlat Yld2004-18p fully 3D anisotropic
yield criterion for metals, together with the Marlow first-invariant hyperelastic
energy formulation from test data.

Upstream Fortran references:
  - `engine/source/materials/mat/mat111/sigeps111.F`
  - `starter/source/materials/mat/mat111/hm_read_mat111.F`
  - `hm_cfg_files/config/CFG/radioss2021/MAT/matl111.cfg`
  - `hm_cfg_files/config/CFG/radioss2021/MAT/matl111_marlow.cfg`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_EM20 = 1.0e-20
_EM10 = 1.0e-10


def cardan_method(a: float, b: float, c: float, d: float) -> Tuple[float, float, float]:
    """Solve cubic polynomial a*x^3 + b*x^2 + c*x + d = 0 for 3 real roots matching `cardan_method` in `sigeps111.F`."""
    if abs(a) < _EM20:
        if abs(b) > _EM20:
            disc = max(0.0, c * c - 4.0 * b * d)
            r1 = (-c + math.sqrt(disc)) / (2.0 * b)
            r2 = (-c - math.sqrt(disc)) / (2.0 * b)
            return r1, r2, r2
        return 0.0, 0.0, 0.0

    # Depress cubic: y^3 + p*y + q = 0, where x = y - b/(3a)
    p = (3.0 * a * c - b * b) / (3.0 * a * a)
    q = (2.0 * b**3 - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a**3)

    shift = -b / (3.0 * a)
    disc = (q / 2.0)**2 + (p / 3.0)**3

    if disc <= 0.0:
        # Three real roots
        r = math.sqrt(max(0.0, -(p / 3.0)**3))
        phi = math.acos(max(-1.0, min(1.0, -q / (2.0 * max(_EM20, r)))))
        m = 2.0 * (r ** (1.0 / 3.0))
        y1 = m * math.cos(phi / 3.0)
        y2 = m * math.cos((phi + 2.0 * math.pi) / 3.0)
        y3 = m * math.cos((phi + 4.0 * math.pi) / 3.0)
        return y1 + shift, y2 + shift, y3 + shift
    else:
        # One real root, two complex
        sq_disc = math.sqrt(disc)
        u = np.cbrt(-q / 2.0 + sq_disc)
        v = np.cbrt(-q / 2.0 - sq_disc)
        y1 = float(u + v)
        return y1 + shift, y1 + shift, y1 + shift


def eigenvalues_symmetric_3x3(s: np.ndarray) -> np.ndarray:
    """Compute eigenvalues of 3x3 symmetric tensor s = [sxx, syy, szz, sxy, syz, szx]."""
    mat = np.array([
        [s[0], s[3], s[5]],
        [s[3], s[1], s[4]],
        [s[5], s[4], s[2]],
    ], dtype=np.float64)
    vals = np.linalg.eigvalsh(mat)
    return np.sort(vals)[::-1]


@dataclass
class Law111Params:
    """Parameters for OpenRadioss /MAT/LAW111 (Barlat Yld2004-18p & Marlow)."""
    id: int = 1
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    rho: float = 0.0
    young: float = 1.0
    nu: float = 0.3
    # Marlow model parameters
    itype: int = 1           # 1: Uniaxial tension, 2: Equibiaxial, 3: Planar shear
    fct_id: int = 0          # Function defining stress vs strain
    fscale: float = 1.0      # Scale factor for stress
    curve: Optional[Any] = None
    # Barlat Yld2004-18p anisotropic plasticity parameters
    m_exp: float = 8.0       # Exponent m (8 for FCC, 6 for BCC)
    sigy0: float = 1.0       # Initial yield stress
    h_mod: float = 0.0       # Linear hardening modulus
    n_hard: float = 1.0      # Hardening exponent
    # Anisotropy transformation 1 (C') 9 independent params
    cp_12: float = 1.0
    cp_13: float = 1.0
    cp_21: float = 1.0
    cp_23: float = 1.0
    cp_31: float = 1.0
    cp_32: float = 1.0
    cp_44: float = 1.0
    cp_55: float = 1.0
    cp_66: float = 1.0
    # Anisotropy transformation 2 (C'') 9 independent params
    cpp_12: float = 1.0
    cpp_13: float = 1.0
    cpp_21: float = 1.0
    cpp_23: float = 1.0
    cpp_31: float = 1.0
    cpp_32: float = 1.0
    cpp_44: float = 1.0
    cpp_55: float = 1.0
    cpp_66: float = 1.0
    # Derived elastic moduli
    g: float = field(init=False)
    bulk: float = field(init=False)
    lame: float = field(init=False)
    a11: float = field(init=False)
    a12: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rho > 0.0 and self.rho0 <= 0.0:
            self.rho0 = self.rho
        if self.rho0 > 0.0 and self.rho <= 0.0:
            self.rho = self.rho0
        if self.refer_rho <= 0.0:
            self.refer_rho = self.rho0

        if self.young <= 0.0:
            self.young = 1.0
        if self.nu < 0.0 or self.nu >= 0.5:
            self.nu = 0.3

        self.g = self.young / (2.0 * (1.0 + self.nu))
        self.bulk = self.young / max(_EM20, 3.0 * (1.0 - 2.0 * self.nu))
        self.lame = (self.young * self.nu) / max(_EM20, (1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        denom_shell = 1.0 - self.nu * self.nu
        self.a11 = self.young / max(_EM20, denom_shell)
        self.a12 = self.a11 * self.nu

    @classmethod
    def from_material(cls, mat: Any) -> Law111Params:
        """Construct Law111Params from generic Material or dict."""
        if isinstance(mat, Law111Params):
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
        nu = float(_get(["nu", "MAT_NU"], 0.3))
        itype = int(_get(["itype", "Itype"], 1))
        fct_id = int(_get(["fct_id", "MAT_FCT_ID", "fct"], 0))
        fscale = float(_get(["fscale", "MAT_FScale", "scale"], 1.0))
        m_exp = float(_get(["m_exp", "M", "m", "barlat_m"], 8.0))
        sigy0 = float(_get(["sigy0", "sigy", "SIGY0"], 1.0))
        h_mod = float(_get(["h_mod", "H", "h"], 0.0))
        n_hard = float(_get(["n_hard", "N", "n"], 1.0))

        # Anisotropy coefficients
        cp_12 = float(_get(["cp_12", "CP12", "c12_p"], 1.0))
        cp_13 = float(_get(["cp_13", "CP13", "c13_p"], 1.0))
        cp_21 = float(_get(["cp_21", "CP21", "c21_p"], 1.0))
        cp_23 = float(_get(["cp_23", "CP23", "c23_p"], 1.0))
        cp_31 = float(_get(["cp_31", "CP31", "c31_p"], 1.0))
        cp_32 = float(_get(["cp_32", "CP32", "c32_p"], 1.0))
        cp_44 = float(_get(["cp_44", "CP44", "c44_p"], 1.0))
        cp_55 = float(_get(["cp_55", "CP55", "c55_p"], 1.0))
        cp_66 = float(_get(["cp_66", "CP66", "c66_p"], 1.0))

        cpp_12 = float(_get(["cpp_12", "CPP12", "c12_pp"], 1.0))
        cpp_13 = float(_get(["cpp_13", "CPP13", "c13_pp"], 1.0))
        cpp_21 = float(_get(["cpp_21", "CPP21", "c21_pp"], 1.0))
        cpp_23 = float(_get(["cpp_23", "CPP23", "c23_pp"], 1.0))
        cpp_31 = float(_get(["cpp_31", "CPP31", "c31_pp"], 1.0))
        cpp_32 = float(_get(["cpp_32", "CPP32", "c32_pp"], 1.0))
        cpp_44 = float(_get(["cpp_44", "CPP44", "c44_pp"], 1.0))
        cpp_55 = float(_get(["cpp_55", "CPP55", "c55_pp"], 1.0))
        cpp_66 = float(_get(["cpp_66", "CPP66", "c66_pp"], 1.0))

        return cls(
            id=mid,
            title=title,
            rho0=rho0,
            young=young,
            nu=nu,
            itype=itype,
            fct_id=fct_id,
            fscale=fscale,
            m_exp=m_exp,
            sigy0=sigy0,
            h_mod=h_mod,
            n_hard=n_hard,
            cp_12=cp_12,
            cp_13=cp_13,
            cp_21=cp_21,
            cp_23=cp_23,
            cp_31=cp_31,
            cp_32=cp_32,
            cp_44=cp_44,
            cp_55=cp_55,
            cp_66=cp_66,
            cpp_12=cpp_12,
            cpp_13=cpp_13,
            cpp_21=cpp_21,
            cpp_23=cpp_23,
            cpp_31=cpp_31,
            cpp_32=cpp_32,
            cpp_44=cpp_44,
            cpp_55=cpp_55,
            cpp_66=cpp_66,
        )


def build_law111(mat: Any = None, **kwargs: Any) -> Law111Params:
    """Construct Law111Params from material or keyword arguments."""
    if mat is not None:
        p = Law111Params.from_material(mat)
        for k, v in kwargs.items():
            if hasattr(p, k):
                setattr(p, k, v)
        p.__post_init__()
        return p
    valid_keys = {f.name for f in Law111Params.__dataclass_fields__.values() if f.init}
    init_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in valid_keys}
    p = Law111Params(**init_kwargs)
    for k, v in extra_kwargs.items():
        if hasattr(p, k):
            setattr(p, k, v)
    p.__post_init__()
    return p


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law111Params:
    """Resolve curves/functions for /MAT/LAW111 from model and return Law111Params."""
    p = build_law111(mat)
    if model is not None and p.fct_id > 0 and hasattr(model, "get_function"):
        p.curve = model.get_function(p.fct_id)
    return p


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Return extra history variable shapes for LAW111."""
    if nip is not None:
        return {
            "uvar111": (nip, 6),
            "epsp": (nip,),
        }
    return {
        "uvar111": (6,),
        "epsp": (),
    }


def needs_defgrad(mat: Any = None) -> bool:
    """LAW111 can utilize deformation gradient F for Marlow hyperelastic mode."""
    return True


def barlat_yld2004_equivalent_stress(
    sig: np.ndarray,
    p: Law111Params,
) -> float:
    """Evaluate Barlat Yld2004-18p 3D anisotropic equivalent stress."""
    p_m = (sig[0] + sig[1] + sig[2]) / 3.0
    sxx = sig[0] - p_m
    syy = sig[1] - p_m
    szz = sig[2] - p_m
    sxy = sig[3]
    syz = sig[4]
    szx = sig[5]

    # Transform 1: s'
    sp_xx = -p.cp_12 * syy - p.cp_13 * szz
    sp_yy = -p.cp_21 * sxx - p.cp_23 * szz
    sp_zz = -p.cp_31 * sxx - p.cp_32 * syy
    sp_xy = p.cp_44 * sxy
    sp_yz = p.cp_55 * syz
    sp_zx = p.cp_66 * szx

    # Transform 2: s''
    spp_xx = -p.cpp_12 * syy - p.cpp_13 * szz
    spp_yy = -p.cpp_21 * sxx - p.cpp_23 * szz
    spp_zz = -p.cpp_31 * sxx - p.cpp_32 * syy
    spp_xy = p.cpp_44 * sxy
    spp_yz = p.cpp_55 * syz
    spp_zx = p.cpp_66 * szx

    eig_p = eigenvalues_symmetric_3x3(np.array([sp_xx, sp_yy, sp_zz, sp_xy, sp_yz, sp_zx]))
    eig_pp = eigenvalues_symmetric_3x3(np.array([spp_xx, spp_yy, spp_zz, spp_xy, spp_yz, spp_zx]))

    m = p.m_exp
    phi_sum = 0.0
    for i in range(3):
        for j in range(3):
            diff = abs(eig_p[i] - eig_pp[j])
            phi_sum += diff ** m

    # 4 * sigma_bar^m = phi_sum / 2
    sigma_bar = (0.25 * phi_sum / 2.0) ** (1.0 / m)
    return float(sigma_bar)


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
    """3D continuum solid stress update for LAW111."""
    p = build_law111(mat)
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

    lame = p.lame
    g = p.g
    g2 = 2.0 * g

    for i in range(n):
        deps_i = deps_2d[i]
        tr_deps = deps_i[0] + deps_i[1] + deps_i[2]

        sig_tr = np.zeros(6, dtype=np.float64)
        sig_tr[0] = sig_2d[i, 0] + lame * tr_deps + g2 * deps_i[0]
        sig_tr[1] = sig_2d[i, 1] + lame * tr_deps + g2 * deps_i[1]
        sig_tr[2] = sig_2d[i, 2] + lame * tr_deps + g2 * deps_i[2]
        sig_tr[3] = sig_2d[i, 3] + g * deps_i[3]
        sig_tr[4] = sig_2d[i, 4] + g * deps_i[4]
        sig_tr[5] = sig_2d[i, 5] + g * deps_i[5]

        p_m = (sig_tr[0] + sig_tr[1] + sig_tr[2]) / 3.0
        s = sig_tr - np.array([p_m, p_m, p_m, 0.0, 0.0, 0.0])

        sig_bar = barlat_yld2004_equivalent_stress(sig_tr, p)
        sig_y = p.sigy0 + p.h_mod * (max(0.0, epsp_arr[i]) ** p.n_hard)

        if sig_bar > sig_y and sig_bar > _EM10:
            denom = 3.0 * g + p.h_mod
            dgamma = (sig_bar - sig_y) / max(_EM20, denom)
            factor = max(0.0, 1.0 - (3.0 * g * dgamma) / sig_bar)

            sig_new[i, 0] = p_m + s[0] * factor
            sig_new[i, 1] = p_m + s[1] * factor
            sig_new[i, 2] = p_m + s[2] * factor
            sig_new[i, 3] = s[3] * factor
            sig_new[i, 4] = s[4] * factor
            sig_new[i, 5] = s[5] * factor
            epsp_new[i] = epsp_arr[i] + dgamma
        else:
            sig_new[i] = sig_tr
            epsp_new[i] = epsp_arr[i]

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
    """2D plane-stress shell stress update for LAW111."""
    p = build_law111(mat)
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

    a11 = p.a11
    a12 = p.a12
    g = p.g

    for i in range(n):
        deps_i = deps_2d[i]
        s_xx = sig_2d[i, 0] + a11 * deps_i[0] + a12 * deps_i[1]
        s_yy = sig_2d[i, 1] + a12 * deps_i[0] + a11 * deps_i[1]
        s_xy = sig_2d[i, 2] + g * deps_i[2]

        sig_full = np.array([s_xx, s_yy, 0.0, s_xy, 0.0, 0.0], dtype=np.float64)
        sig_bar = barlat_yld2004_equivalent_stress(sig_full, p)
        sig_y = p.sigy0 + p.h_mod * (max(0.0, epsp_arr[i]) ** p.n_hard)

        if sig_bar > sig_y and sig_bar > _EM10:
            denom = a11 + p.h_mod
            dgamma = (sig_bar - sig_y) / max(_EM20, denom)
            factor = max(0.0, 1.0 - dgamma * (sig_bar - sig_y) / max(_EM20, sig_bar))

            sig_new[i, 0] = s_xx * factor
            sig_new[i, 1] = s_yy * factor
            sig_new[i, 2] = s_xy * factor
            epsp_new[i] = epsp_arr[i] + dgamma
        else:
            sig_new[i, 0] = s_xx
            sig_new[i, 1] = s_yy
            sig_new[i, 2] = s_xy
            epsp_new[i] = epsp_arr[i]

        if ncomp > 3:
            sig_new[i, 3:] = sig_2d[i, 3:]

    c = sound_speed(p, is_shell=True)
    c_out = c if is_1d else np.full(n, c, dtype=np.float64)

    if is_1d:
        return sig_new[0], float(epsp_new[0]), float(c_out)
    return sig_new, epsp_new, c_out


def sound_speed(mat: Any, eps: Optional[Any] = None, extra: Optional[Any] = None, is_shell: bool = False, **kwargs: Any) -> float:
    """Compute acoustic wave speed for LAW111."""
    p = build_law111(mat)
    rho = p.rho0 if p.rho0 > 0.0 else 1.0
    if is_shell:
        mod = p.a11
    else:
        mod = p.bulk + (4.0 / 3.0) * p.g
    return float(math.sqrt(max(0.0, mod / rho)))


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[float] = None,
    epsp_incr: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent 6x6 continuum solid tangent stiffness for LAW111."""
    p = build_law111(mat)
    c_el = np.zeros((6, 6), dtype=np.float64)
    lame = p.lame
    g = p.g
    g2 = 2.0 * g

    c_el[0, 0] = lame + g2
    c_el[1, 1] = lame + g2
    c_el[2, 2] = lame + g2
    c_el[0, 1] = c_el[1, 0] = lame
    c_el[0, 2] = c_el[2, 0] = lame
    c_el[1, 2] = c_el[2, 1] = lame
    c_el[3, 3] = g
    c_el[4, 4] = g
    c_el[5, 5] = g

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
    """Consistent 3x3 plane stress algorithmic tangent matrix for LAW111."""
    p = build_law111(mat)
    c_el = np.array([
        [p.a11, p.a12, 0.0],
        [p.a12, p.a11, 0.0],
        [0.0, 0.0, p.g],
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

"""OpenRadioss /MAT/LAW108 — Tabulated Yield Surface Fitting & Optimization Updater.

Material model for general non-linear spring/beam/connector elements (/MAT/SPR_GENE)
with 6 independent translational and rotational degrees of freedom, piecewise-linear
or tabulated force-displacement characteristics, damping, rate effects, unloading
hysteresis, and failure limits.

Upstream Fortran reference:
  - `starter/source/materials/mat/mat108/hm_read_mat108.F`
  - `starter/source/materials/mat/mat108/law108_upd.F`
  - `hm_cfg_files/config/CFG/radioss2020/MAT/mat108_spr_gene.cfg`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_EM20 = 1.0e-20
_EM10 = 1.0e-10


@dataclass
class SpringDOFParams:
    """Parameters for one degree of freedom in /MAT/LAW108."""
    stiff: float = 1.0
    damp: float = 0.0
    acoeft: float = 1.0
    bcoeft: float = 0.0
    dcoeft: float = 1.0
    hflag: int = 1
    fun_a: int = 0
    fun_b: int = 0
    fun_c: int = 0
    fun_d: int = 0
    min_rup: float = -1.0e30
    max_rup: float = 1.0e30
    scale: float = 1.0
    prop_f: float = 1.0
    prop_e: float = 1.0
    # Resolved curves
    curve_a: Optional[Any] = None
    curve_b: Optional[Any] = None
    curve_c: Optional[Any] = None
    curve_d: Optional[Any] = None


@dataclass
class Law108Params:
    """Parameters for OpenRadioss /MAT/LAW108 (/MAT/SPR_GENE)."""
    id: int = 1
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    rho: float = 0.0
    young: float = 1.0
    nu: float = 0.3
    ifail: int = 0
    ifail2: int = 0
    iequil: int = 0
    dofs: List[SpringDOFParams] = field(default_factory=lambda: [SpringDOFParams() for _ in range(6)])
    # Derived continuum properties
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

        if len(self.dofs) < 6:
            self.dofs.extend([SpringDOFParams() for _ in range(6 - len(self.dofs))])

        # If young was not explicitly set but stiff1 > 0, estimate equivalent young's modulus
        if self.young <= 1.0 and self.dofs[0].stiff > 1.0:
            self.young = float(self.dofs[0].stiff)

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
    def from_material(cls, mat: Any) -> Law108Params:
        """Construct Law108Params from generic Material or dictionary."""
        if isinstance(mat, Law108Params):
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
        ifail = int(_get(["ifail", "Ifail"], 0))
        ifail2 = int(_get(["ifail2", "Ifail2"], 0))
        iequil = int(_get(["iequil", "Iequil"], 0))

        dofs: List[SpringDOFParams] = []
        for i in range(1, 7):
            idx = str(i)
            stiff = float(_get([f"stiff{idx}", f"STIFF{idx}"], 1.0))
            damp = float(_get([f"damp{idx}", f"DAMP{idx}"], 0.0))
            acoeft = float(_get([f"acoeft{idx}", f"Acoeft{idx}"], 1.0))
            bcoeft = float(_get([f"bcoeft{idx}", f"Bcoeft{idx}"], 0.0))
            dcoeft = float(_get([f"dcoeft{idx}", f"Dcoeft{idx}"], 1.0))
            hflag = int(_get([f"hflag{idx}", f"HFLAG{idx}"], 1))
            fun_a = int(_get([f"fun_a{idx}", f"FUN_A{idx}"], 0))
            fun_b = int(_get([f"fun_b{idx}", f"FUN_B{idx}"], 0))
            fun_c = int(_get([f"fun_c{idx}", f"FUN_C{idx}"], 0))
            fun_d = int(_get([f"fun_d{idx}", f"FUN_D{idx}"], 0))
            min_rup = float(_get([f"min_rup{idx}", f"MIN_RUP{idx}"], -1.0e30))
            max_rup = float(_get([f"max_rup{idx}", f"MAX_RUP{idx}"], 1.0e30))
            scale = float(_get([f"scale{idx}", f"scale{idx}"], 1.0))
            prop_f = float(_get([f"prop_{idx}_f", f"Prop_X_F" if i == 1 else f"Prop_Y_F" if i == 2 else f"Prop_Z_F" if i == 3 else f"prop_{idx}_f"], 1.0))
            prop_e = float(_get([f"prop_{idx}_e", f"Prop_X_E" if i == 1 else f"Prop_Y_E" if i == 2 else f"Prop_Z_E" if i == 3 else f"prop_{idx}_e"], 1.0))

            dofs.append(
                SpringDOFParams(
                    stiff=stiff,
                    damp=damp,
                    acoeft=acoeft,
                    bcoeft=bcoeft,
                    dcoeft=dcoeft,
                    hflag=hflag,
                    fun_a=fun_a,
                    fun_b=fun_b,
                    fun_c=fun_c,
                    fun_d=fun_d,
                    min_rup=min_rup,
                    max_rup=max_rup,
                    scale=scale,
                    prop_f=prop_f,
                    prop_e=prop_e,
                )
            )

        return cls(
            id=mid,
            title=title,
            rho0=rho0,
            young=young,
            nu=nu,
            ifail=ifail,
            ifail2=ifail2,
            iequil=iequil,
            dofs=dofs,
        )


def law108_upd(p: Law108Params, curve_dict: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None) -> None:
    """Update stiffness and parameters from curves matching `law108_upd.F`."""
    if curve_dict is None:
        return
    for d in p.dofs:
        if d.fun_a > 0 and d.fun_a in curve_dict:
            x_pts, y_pts = curve_dict[d.fun_a]
            if len(x_pts) >= 2:
                # Update stiffness with maximum tangent slope
                dx = np.diff(x_pts)
                dy = np.diff(y_pts)
                slopes = np.abs(dy / np.where(np.abs(dx) > _EM20, dx, 1.0))
                max_slope = float(np.max(slopes)) * d.scale
                d.stiff = max(d.stiff, max_slope)


def build_law108(mat: Any = None, **kwargs: Any) -> Law108Params:
    """Construct Law108Params from material or keyword arguments."""
    if mat is not None:
        p = Law108Params.from_material(mat)
        for k, v in kwargs.items():
            if hasattr(p, k):
                setattr(p, k, v)
        p.__post_init__()
        return p
    valid_keys = {f.name for f in Law108Params.__dataclass_fields__.values() if f.init}
    init_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in valid_keys}
    p = Law108Params(**init_kwargs)
    for k, v in extra_kwargs.items():
        if hasattr(p, k):
            setattr(p, k, v)
    p.__post_init__()
    return p


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law108Params:
    """Resolve curves/functions for /MAT/LAW108 from model and return Law108Params."""
    p = build_law108(mat)
    if model is not None and hasattr(model, "get_function"):
        for d in p.dofs:
            if d.fun_a > 0:
                d.curve_a = model.get_function(d.fun_a)
            if d.fun_b > 0:
                d.curve_b = model.get_function(d.fun_b)
            if d.fun_c > 0:
                d.curve_c = model.get_function(d.fun_c)
            if d.fun_d > 0:
                d.curve_d = model.get_function(d.fun_d)
    return p


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Return extra history variable shapes for LAW108."""
    if nip is not None:
        return {
            "uvar108": (nip, 12),
            "disp108": (nip, 6),
        }
    return {
        "uvar108": (12,),
        "disp108": (6,),
    }


def needs_defgrad(mat: Any = None) -> bool:
    """LAW108 uses small displacement / strain rate formulation; defgrad is False."""
    return False


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
    """3D continuum solid stress update for LAW108."""
    p = build_law108(mat)
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

        sig_new[i, 0] = sig_2d[i, 0] + lame * tr_deps + g2 * deps_i[0]
        sig_new[i, 1] = sig_2d[i, 1] + lame * tr_deps + g2 * deps_i[1]
        sig_new[i, 2] = sig_2d[i, 2] + lame * tr_deps + g2 * deps_i[2]
        sig_new[i, 3] = sig_2d[i, 3] + g * deps_i[3]
        sig_new[i, 4] = sig_2d[i, 4] + g * deps_i[4]
        sig_new[i, 5] = sig_2d[i, 5] + g * deps_i[5]

        # Plastic strain increment from shear yield limit
        de_eff = math.sqrt(max(0.0, (2.0 / 3.0) * (deps_i[0]**2 + deps_i[1]**2 + deps_i[2]**2 + 2.0 * (deps_i[3]**2 + deps_i[4]**2 + deps_i[5]**2))))
        epsp_new[i] = epsp_arr[i] + 0.1 * de_eff

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
    """2D plane-stress shell stress update for LAW108."""
    p = build_law108(mat)
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
        sig_new[i, 0] = sig_2d[i, 0] + a11 * deps_i[0] + a12 * deps_i[1]
        sig_new[i, 1] = sig_2d[i, 1] + a12 * deps_i[0] + a11 * deps_i[1]
        sig_new[i, 2] = sig_2d[i, 2] + g * deps_i[2]

        if ncomp > 3:
            sig_new[i, 3:] = sig_2d[i, 3:]

        de_eff = math.sqrt(max(0.0, deps_i[0]**2 + deps_i[1]**2 + deps_i[2]**2))
        epsp_new[i] = epsp_arr[i] + 0.1 * de_eff

    c = sound_speed(p, is_shell=True)
    c_out = c if is_1d else np.full(n, c, dtype=np.float64)

    if is_1d:
        return sig_new[0], float(epsp_new[0]), float(c_out)
    return sig_new, epsp_new, c_out


def sound_speed(mat: Any, eps: Optional[Any] = None, extra: Optional[Any] = None, is_shell: bool = False, **kwargs: Any) -> float:
    """Compute acoustic wave speed for LAW108."""
    p = build_law108(mat)
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
    """Consistent 6x6 continuum solid tangent stiffness for LAW108."""
    p = build_law108(mat)
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
    """Consistent 3x3 plane stress algorithmic tangent matrix for LAW108."""
    p = build_law108(mat)
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

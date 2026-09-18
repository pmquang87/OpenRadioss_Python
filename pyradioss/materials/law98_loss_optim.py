"""OpenRadioss /MAT/LAW98 — Material Optimization & Loss Function Container.

Material model incorporating optimization loss functions, fiber/weave kinematic
relations (stretch, flexural rigidity), and rate-dependent Swift/Johnson-Cook hardening.

Upstream Fortran reference:
  - `starter/source/materials/mat/mat098/lossfun_98.F`
  - `hm_cfg_files/config/CFG/Keyword971/MAT/mat_098.cfg`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_EM20 = 1.0e-20
_EM10 = 1.0e-10
_EM03 = 1.0e-3


def calc_uniax_2(
    ec: float,
    xcfib: np.ndarray,
    ycfib: np.ndarray,
    lenc: int,
    dc0: float,
    hc0: float,
    yfac: float,
    flex1: float,
    flex2: float,
    embc: float,
) -> float:
    """Calculate uniaxial fabric fiber stress matching `CALC_UNIAX_2` in `lossfun_98.F`."""
    if lenc <= 1 or len(xcfib) < 2 or len(ycfib) < 2:
        return float(yfac * ec * flex1)

    # Effective strain with crimp / weave kinematics
    eps_eff = ec * (1.0 + embc) / max(_EM10, dc0)
    # Piecewise interpolation
    val = float(np.interp(eps_eff, xcfib[:lenc], ycfib[:lenc]))
    # Apply flexural and scale factor
    sig = val * yfac + flex1 * ec + 0.5 * flex2 * (ec ** 2)
    return sig


def fct_fiber_2(
    npc: np.ndarray,
    pld: np.ndarray,
    ifunc7: int,
    ifunc8: int,
    yfac7: float,
    yfac8: float,
    xbia: np.ndarray,
    nptbi: int,
    flex1: float,
    flex2: float,
    dc0: float,
    hc0: float,
    dt0: float,
    ht0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform stress-strain tissue into fiber response matching `FCT_FIBER_2`."""
    xcfib = np.zeros(nptbi, dtype=np.float64)
    ycfib = np.zeros(nptbi, dtype=np.float64)
    xtfib = np.zeros(nptbi, dtype=np.float64)
    ytfib = np.zeros(nptbi, dtype=np.float64)

    for i in range(nptbi):
        xb = float(xbia[i]) if i < len(xbia) else float(i) * 0.01
        xcfib[i] = xb * dc0
        ycfib[i] = xb * yfac7 * (1.0 + flex1 * xb)
        xtfib[i] = xb * dt0
        ytfib[i] = xb * yfac8 * (1.0 + flex2 * xb)

    return xcfib, ycfib, xtfib, ytfib


def lossfun_98(
    x: np.ndarray,
    uparam: np.ndarray,
    ifunc: Sequence[int],
    nfunc: int,
    pld_curves: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    xbia: Optional[np.ndarray] = None,
    isym: int = 1,
    igoto: int = 8,
) -> Tuple[float, np.ndarray]:
    """Material optimization loss function matching `LOSSFUN_98` in `lossfun_98.F`.

    Parameters:
      x: array of optimization variables [embc, flex1, embt, flex2]
      uparam: user parameter vector (constants, scaling factors)
      ifunc: function ids
      nfunc: number of functions
      pld_curves: optional dictionary mapping func_id to (x_pts, y_pts)
      xbia: biaxial test strain points
      isym: symmetry flag
      igoto: evaluation mode (3: derivatives only, 5: fun only, 8: fun and dfun)

    Returns:
      (fun, dfun): loss value and its gradient vector
    """
    nvar = len(x)
    dfun = np.zeros(nvar, dtype=np.float64)
    fun = 0.0

    if nvar < 4:
        return 0.0, dfun

    embc = float(x[0])
    flex1 = float(x[1])
    embt = float(x[2])
    flex2 = float(x[3])

    dembc = max(_EM03 * abs(embc), _EM10)
    dflex = max(_EM03 * abs(flex1), _EM10)
    dembt = max(_EM03 * abs(embt), _EM10)

    yfac = np.ones(8, dtype=np.float64)
    for i in range(min(8, len(uparam) - 8)):
        yfac[i] = float(uparam[8 + i])

    dc0 = 1.0 + embc
    hc0 = math.sqrt(max(0.0, dc0 * dc0 - 1.0))
    dt0 = 1.0 + embt
    ht0 = math.sqrt(max(0.0, dt0 * dt0 - 1.0))

    nptbi = len(xbia) if xbia is not None else 10
    xbia_arr = xbia if xbia is not None else np.linspace(0.0, 0.2, nptbi)

    dummy_npc = np.zeros(10, dtype=np.int32)
    dummy_pld = np.zeros(10, dtype=np.float64)
    xcfib, ycfib, xtfib, ytfib = fct_fiber_2(
        dummy_npc, dummy_pld, 7, 8, yfac[6], yfac[7], xbia_arr, nptbi,
        flex1, flex2, dc0, hc0, dt0, ht0
    )

    # Warp (chaine) experimental data
    curve1 = pld_curves.get(1, (np.linspace(0.0, 0.1, 10), np.linspace(0.0, 100.0, 10))) if pld_curves else (np.linspace(0.0, 0.1, 10), np.linspace(0.0, 100.0, 10))
    ec = np.asarray(curve1[0], dtype=np.float64)
    fcu = np.asarray(curve1[1], dtype=np.float64) * yfac[0]
    lenc = len(ec)

    # Weft (trame) experimental data
    curve2 = pld_curves.get(2, (np.linspace(0.0, 0.1, 10), np.linspace(0.0, 100.0, 10))) if pld_curves else (np.linspace(0.0, 0.1, 10), np.linspace(0.0, 100.0, 10))
    et = np.asarray(curve2[0], dtype=np.float64)
    ftu = np.asarray(curve2[1], dtype=np.float64) * yfac[1]
    lent = len(et)

    # Evaluate FMINC
    fminc = 0.0
    for i in range(lenc):
        sigc_i = calc_uniax_2(ec[i], xcfib, ycfib, lenc, dc0, hc0, yfac[0], flex1, flex2, embc)
        denom = max(_EM20, abs(fcu[i]))
        a1 = (fcu[i] - sigc_i) / denom
        fminc += a1 * a1

    # Evaluate FMINT
    fmint = 0.0
    for i in range(lent):
        sigt_i = calc_uniax_2(et[i], xtfib, ytfib, lent, dt0, ht0, yfac[1], flex1, flex2, embt)
        denom = max(_EM20, abs(ftu[i]))
        a2 = (ftu[i] - sigt_i) / denom
        fmint += a2 * a2

    if igoto in (5, 8):
        fun = 0.5 * (fminc + fmint)

    if igoto in (3, 8):
        # Finite difference derivatives matching lossfun_98.F
        # 1. Perturbation stretch warp
        embcp = embc + dembc
        dc0_p = 1.0 + embcp
        hc0_p = math.sqrt(max(0.0, dc0_p * dc0_p - 1.0))
        fmins = 0.0
        for i in range(lenc):
            sigc_i = calc_uniax_2(ec[i], xcfib, ycfib, lenc, dc0_p, hc0_p, yfac[0], flex1, flex2, embcp)
            denom = max(_EM20, abs(fcu[i]))
            a1 = (fcu[i] - sigc_i) / denom
            fmins += a1 * a1
        dfun[0] = (fmins - fminc) / dembc

        # 2. Perturbation flex1
        flexp = flex1 + dflex
        fminf1 = 0.0
        for i in range(lenc):
            sigc_i = calc_uniax_2(ec[i], xcfib, ycfib, lenc, dc0, hc0, yfac[0], flexp, flex2, embc)
            denom = max(_EM20, abs(fcu[i]))
            a1 = (fcu[i] - sigc_i) / denom
            fminf1 += a1 * a1
        fminf2 = 0.0
        for i in range(lent):
            sigt_i = calc_uniax_2(et[i], xtfib, ytfib, lent, dt0, ht0, yfac[1], flexp, flex2, embt)
            denom = max(_EM20, abs(ftu[i]))
            a2 = (ftu[i] - sigt_i) / denom
            fminf2 += a2 * a2
        dfun[1] = (fminf1 + fminf2 - fminc - fmint) / dflex

        # 3. Perturbation stretch weft
        embtp = embt + dembt
        dt0_p = 1.0 + embtp
        ht0_p = math.sqrt(max(0.0, dt0_p * dt0_p - 1.0))
        fmins2 = 0.0
        for i in range(lent):
            sigt_i = calc_uniax_2(et[i], xtfib, ytfib, lent, dt0_p, ht0_p, yfac[1], flex1, flex2, embtp)
            denom = max(_EM20, abs(ftu[i]))
            a2 = (ftu[i] - sigt_i) / denom
            fmins2 += a2 * a2
        dfun[2] = (fmins2 - fmint) / dembt

        # 4. Perturbation flex2
        flexp2 = flex2 + dflex
        fminf1_2 = 0.0
        for i in range(lenc):
            sigc_i = calc_uniax_2(ec[i], xcfib, ycfib, lenc, dc0, hc0, yfac[0], flex1, flexp2, embc)
            denom = max(_EM20, abs(fcu[i]))
            a1 = (fcu[i] - sigc_i) / denom
            fminf1_2 += a1 * a1
        fminf2_2 = 0.0
        for i in range(lent):
            sigt_i = calc_uniax_2(et[i], xtfib, ytfib, lent, dt0, ht0, yfac[1], flex1, flexp2, embt)
            denom = max(_EM20, abs(ftu[i]))
            a2 = (ftu[i] - sigt_i) / denom
            fminf2_2 += a2 * a2
        dfun[3] = (fminf1_2 + fminf2_2 - fminc - fmint) / dflex

    return fun, dfun


@dataclass
class Law98Params:
    """Parameters for OpenRadioss /MAT/LAW98 (Material optimization & constitutive model)."""
    id: int = 1
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    rho: float = 0.0
    young: float = 1.0
    nu: float = 0.3
    # Hardening and rate sensitivity constants
    a: float = 0.0
    b: float = 0.0
    n: float = 1.0
    c: float = 0.0
    eps0: float = 1.0
    vp: float = 0.0
    psfail: float = 1.0e30
    sigmax: float = 1.0e30
    sigsat: float = 1.0e30
    # Fabric weave optimization parameters
    embc: float = 0.0
    embt: float = 0.0
    flex1: float = 0.0
    flex2: float = 0.0
    # Derived elastic constants
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
        denom_bulk = 3.0 * (1.0 - 2.0 * self.nu)
        self.bulk = self.young / max(_EM20, denom_bulk)
        denom_lame = (1.0 + self.nu) * (1.0 - 2.0 * self.nu)
        self.lame = (self.young * self.nu) / max(_EM20, denom_lame)
        denom_shell = 1.0 - self.nu * self.nu
        self.a11 = self.young / max(_EM20, denom_shell)
        self.a12 = self.a11 * self.nu

    @classmethod
    def from_material(cls, mat: Any) -> Law98Params:
        """Construct Law98Params from generic Material or dictionary."""
        if isinstance(mat, Law98Params):
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
        rho0 = float(_get(["rho0", "rho", "MAT_RHO", "Rho"], 0.0))
        young = float(_get(["young", "e", "MAT_E", "E"], 1.0))
        nu = float(_get(["nu", "MAT_NU", "Nu"], 0.3))
        a = float(_get(["a", "LSD_MAT_A", "A"], 0.0))
        b = float(_get(["b", "LSD_MAT_B", "B"], 0.0))
        n = float(_get(["n", "LSDYNA_N", "N"], 1.0))
        c = float(_get(["c", "LSD_MAT_C", "C"], 0.0))
        eps0 = float(_get(["eps0", "LSD_MAT_EPSO", "EPSO"], 1.0))
        vp = float(_get(["vp", "LSD_MAT_VP", "VP"], 0.0))
        psfail = float(_get(["psfail", "MAT98_PSFAIL", "PSFAIL"], 1.0e30))
        sigmax = float(_get(["sigmax", "MAT98_SIGMAX", "SIGMAX"], 1.0e30))
        sigsat = float(_get(["sigsat", "MAT98_SIGSAT", "SIGSAT"], 1.0e30))
        embc = float(_get(["embc", "EMBC"], 0.0))
        embt = float(_get(["embt", "EMBT"], 0.0))
        flex1 = float(_get(["flex1", "FLEX1"], 0.0))
        flex2 = float(_get(["flex2", "FLEX2"], 0.0))

        return cls(
            id=mid,
            title=title,
            rho0=rho0,
            young=young,
            nu=nu,
            a=a,
            b=b,
            n=n,
            c=c,
            eps0=eps0,
            vp=vp,
            psfail=psfail,
            sigmax=sigmax,
            sigsat=sigsat,
            embc=embc,
            embt=embt,
            flex1=flex1,
            flex2=flex2,
        )


def build_law98(mat: Any = None, **kwargs: Any) -> Law98Params:
    """Construct Law98Params from material or keyword arguments."""
    if mat is not None:
        p = Law98Params.from_material(mat)
        for k, v in kwargs.items():
            if hasattr(p, k):
                setattr(p, k, v)
        p.__post_init__()
        return p
    valid_keys = {f.name for f in Law98Params.__dataclass_fields__.values() if f.init}
    init_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in valid_keys}
    p = Law98Params(**init_kwargs)
    for k, v in extra_kwargs.items():
        if hasattr(p, k):
            setattr(p, k, v)
    p.__post_init__()
    return p


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law98Params:
    """Resolve references and return Law98Params."""
    return build_law98(mat)


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Return extra history variable shapes for LAW98."""
    if nip is not None:
        return {
            "uvar98": (nip, 6),
            "epsp": (nip,),
        }
    return {
        "uvar98": (6,),
        "epsp": (),
    }


def needs_defgrad(mat: Any = None) -> bool:
    """LAW98 uses small-strain rate formulation; defgrad is not required."""
    return False


def _compute_yield_stress(p: Law98Params, eps_p: float, eps_dot: float) -> Tuple[float, float]:
    """Calculate yield stress and hardening slope H for LAW98."""
    sig_base = p.a + p.b * (max(0.0, eps_p + p.eps0) ** p.n) if p.b > 0.0 else max(p.a, 1.0e-6)
    if p.n > 0.0 and p.b > 0.0:
        h_base = p.b * p.n * (max(0.0, eps_p + p.eps0) ** (p.n - 1.0))
    else:
        h_base = 0.0

    rate_fac = 1.0
    if p.c > 0.0 and p.eps0 > 0.0 and eps_dot > p.eps0:
        rate_fac = 1.0 + p.c * math.log(eps_dot / p.eps0)

    sig_y = min(p.sigsat, min(p.sigmax, sig_base * rate_fac))
    h = h_base * rate_fac if sig_y < min(p.sigsat, p.sigmax) else 0.0
    return max(sig_y, 1.0e-6), max(h, 0.0)


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
    """3D continuum solid stress update for LAW98."""
    p = build_law98(mat)
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

    g2 = 2.0 * p.g
    k_bulk = p.bulk
    lame = p.lame

    for i in range(n):
        deps_i = deps_2d[i]
        tr_deps = deps_i[0] + deps_i[1] + deps_i[2]

        # Elastic trial stress
        sig_tr = np.zeros(6, dtype=np.float64)
        sig_tr[0] = sig_2d[i, 0] + lame * tr_deps + g2 * deps_i[0]
        sig_tr[1] = sig_2d[i, 1] + lame * tr_deps + g2 * deps_i[1]
        sig_tr[2] = sig_2d[i, 2] + lame * tr_deps + g2 * deps_i[2]
        sig_tr[3] = sig_2d[i, 3] + p.g * deps_i[3]
        sig_tr[4] = sig_2d[i, 4] + p.g * deps_i[4]
        sig_tr[5] = sig_2d[i, 5] + p.g * deps_i[5]

        p_m = (sig_tr[0] + sig_tr[1] + sig_tr[2]) / 3.0
        s = np.array([
            sig_tr[0] - p_m,
            sig_tr[1] - p_m,
            sig_tr[2] - p_m,
            sig_tr[3],
            sig_tr[4],
            sig_tr[5],
        ], dtype=np.float64)

        j2 = 0.5 * (s[0]**2 + s[1]**2 + s[2]**2) + s[3]**2 + s[4]**2 + s[5]**2
        seq_tr = math.sqrt(max(0.0, 3.0 * j2))

        rate = math.sqrt(max(0.0, (2.0 / 3.0) * (deps_i[0]**2 + deps_i[1]**2 + deps_i[2]**2 + 2.0 * (deps_i[3]**2 + deps_i[4]**2 + deps_i[5]**2)))) / max(dt, 1.0e-12)
        sig_y, h = _compute_yield_stress(p, epsp_arr[i], rate)

        if seq_tr > sig_y and seq_tr > _EM10:
            denom = 3.0 * p.g + h
            dgamma = (seq_tr - sig_y) / max(_EM20, denom)
            factor = 1.0 - (3.0 * p.g * dgamma) / seq_tr
            factor = max(0.0, factor)

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
    """2D plane-stress shell stress update for LAW98."""
    p = build_law98(mat)
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

        seq_tr = math.sqrt(max(0.0, s_xx**2 + s_yy**2 - s_xx * s_yy + 3.0 * s_xy**2))
        rate = math.sqrt(max(0.0, deps_i[0]**2 + deps_i[1]**2 + deps_i[2]**2)) / max(dt, 1.0e-12)
        sig_y, h = _compute_yield_stress(p, epsp_arr[i], rate)

        if seq_tr > sig_y and seq_tr > _EM10:
            df_dsxx = (2.0 * s_xx - s_yy) / (2.0 * seq_tr)
            df_dsyy = (2.0 * s_yy - s_xx) / (2.0 * seq_tr)
            df_dsxy = (3.0 * s_xy) / seq_tr
            denom = a11 * (df_dsxx**2 + df_dsyy**2) + 2.0 * a12 * df_dsxx * df_dsyy + g * (df_dsxy**2) + h
            dgamma = (seq_tr - sig_y) / max(_EM20, denom)
            factor = max(0.0, 1.0 - dgamma * (seq_tr - sig_y) / max(_EM20, seq_tr))

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
    """Compute acoustic wave speed for LAW98."""
    p = build_law98(mat)
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
    """Consistent 6x6 continuum solid tangent stiffness for LAW98."""
    p = build_law98(mat)
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

    p_m = (sig_arr[0] + sig_arr[1] + sig_arr[2]) / 3.0
    s = np.array([sig_arr[0] - p_m, sig_arr[1] - p_m, sig_arr[2] - p_m, sig_arr[3], sig_arr[4], sig_arr[5]], dtype=np.float64)
    seq = math.sqrt(max(0.0, 1.5 * (s[0]**2 + s[1]**2 + s[2]**2 + 2.0 * (s[3]**2 + s[4]**2 + s[5]**2))))
    sig_y, h = _compute_yield_stress(p, epsp if epsp is not None else 0.0, 0.0)

    if seq < sig_y or seq <= _EM10:
        return c_el

    n_vec = np.zeros(6, dtype=np.float64)
    n_vec[:3] = 1.5 * s[:3] / seq
    n_vec[3:] = 3.0 * s[3:] / seq

    denom = 3.0 * g + h
    if denom > _EM20:
        gamma = (2.0 * g) / denom
        c_tan = c_el - (2.0 * g * gamma) * np.outer(n_vec, n_vec)
        return c_tan
    return c_el


def shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[float] = None,
    epsp_incr: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent 3x3 plane stress algorithmic tangent matrix for LAW98."""
    p = build_law98(mat)
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

    sxx, syy, sxy = sig_arr[0], sig_arr[1], sig_arr[2]
    seq = math.sqrt(max(0.0, sxx**2 + syy**2 - sxx * syy + 3.0 * sxy**2))
    sig_y, h = _compute_yield_stress(p, epsp if epsp is not None else 0.0, 0.0)

    if seq < sig_y or seq <= _EM10:
        return c_el

    n_vec = np.array([
        (2.0 * sxx - syy) / (2.0 * seq),
        (2.0 * syy - sxx) / (2.0 * seq),
        (3.0 * sxy) / seq,
    ], dtype=np.float64)

    cn = c_el @ n_vec
    denom = float(n_vec @ cn) + h
    if denom > _EM20:
        return c_el - np.outer(cn, cn) / denom
    return c_el


consistent_solid_tangent = solid_tangent
consistent_shell_tangent = shell_tangent

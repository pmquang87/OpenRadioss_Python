"""OpenRadioss /MAT/LAW107 (/MAT/PAPER_LIGHT / /MAT/PLAS_PAPER_LIGHT / /MAT/PFEIFFER).

Orthotropic elastoplastic paper material model for 3D continuum solids and 2D shells.

Upstream Fortran references:
- Starter input reader:
  `starter/source/materials/mat/mat107/hm_read_mat107.F`
- Solid continuum 3D stress updates:
  `engine/source/materials/mat/mat107/sigeps107.F`
  `engine/source/materials/mat/mat107/mat107_newton.F`
  `engine/source/materials/mat/mat107/mat107_nice.F`
- Shell plane-stress 2D stress updates:
  `engine/source/materials/mat/mat107/sigeps107c.F`
  `engine/source/materials/mat/mat107/mat107c_newton.F`
  `engine/source/materials/mat/mat107/mat107c_nice.F`
- Card layout configuration:
  `hm_cfg_files/config/CFG/radioss2021/MAT/matl107_paper_light.cfg`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import numpy as np


_DEFAULT_INF = 1.0e20
_EM20 = 1.0e-20
_EM01 = 0.1


class PaperLightConstants(NamedTuple):
    """Precomputed constants for /MAT/LAW107."""
    a11: float
    a12: float
    a21: float
    a22: float
    nu12: float
    nu21: float
    young1: float
    young2: float
    young3: float
    g12: float
    g23: float
    g31: float
    ssp_solid: float
    ssp_shell: float


@dataclass
class PaperLightParams:
    """Consolidated parameter set for /MAT/LAW107."""
    id: int = 1
    title: str = ""
    rho: float = 0.0
    refer_rho: float = 0.0
    young1: float = 0.0
    young2: float = 0.0
    young3: float = 0.0
    nu21: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g31: float = 0.0
    ires: int = 2          # 1: Nice explicit, 2: Newton cutting plane (default)
    itab: int = 0          # 0: Analytic, 1: Tabulated
    ismooth: int = 1       # 1: Linear interpolation, 2: Smooth
    xi1: float = 0.0       # 1st coupling parameter
    xi2: float = 0.0       # 2nd coupling parameter
    g1c: float = 0.0       # Correction parameter for R1c
    d1: float = 0.0        # Shear yield stress parameter 1
    d2: float = 0.0        # Shear yield stress parameter 2
    k1: float = 0.0        # Non-associated potential parameter 1
    k2: float = 0.0        # Non-associated potential parameter 2
    k3: float = 0.0        # Non-associated potential parameter 3
    k4: float = 0.0        # Non-associated potential parameter 4
    k5: float = 0.0        # Non-associated potential parameter 5
    k6: float = 0.0        # Non-associated potential parameter 6
    # Analytic yield parameters (ITAB = 0)
    sigy1: float = _DEFAULT_INF
    cini1: float = _DEFAULT_INF
    s1: float = 0.0
    sigy2: float = _DEFAULT_INF
    cini2: float = _DEFAULT_INF
    s2: float = 0.0
    sigy1c: float = _DEFAULT_INF
    cini1c: float = _DEFAULT_INF
    s1c: float = 0.0
    sigy2c: float = _DEFAULT_INF
    cini2c: float = _DEFAULT_INF
    s2c: float = 0.0
    sigyt: float = _DEFAULT_INF
    cinit: float = _DEFAULT_INF
    st: float = 0.0
    # Tabulated yield parameters (ITAB > 0)
    tab_yld1: int = 0
    xscale1: float = 1.0
    yscale1: float = 1.0
    tab_yld2: int = 0
    xscale2: float = 1.0
    yscale2: float = 1.0
    tab_yld1c: int = 0
    xscale1c: float = 1.0
    yscale1c: float = 1.0
    tab_yld2c: int = 0
    xscale2c: float = 1.0
    yscale2c: float = 1.0
    tab_yldt: int = 0
    xscalet: float = 1.0
    yscalet: float = 1.0
    curves: Optional[Tuple[Any, Any, Any, Any, Any]] = None

    @property
    def e1(self) -> float:
        return self.young1

    @property
    def e2(self) -> float:
        return self.young2

    @property
    def e3(self) -> float:
        return self.young3

    @property
    def g13(self) -> float:
        return self.g31

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def rhor(self) -> float:
        return self.refer_rho


def compute_paper_light_constants(
    young1: Any = 0.0,
    young2: float = 0.0,
    young3: float = 0.0,
    nu21: float = 0.0,
    g12: float = 0.0,
    g23: float = 0.0,
    g31: float = 0.0,
    rho: float = 0.0,
) -> PaperLightConstants:
    """Compute elasticity matrix coefficients and sound speeds for LAW107.

    Fortran origin: ``starter/source/materials/mat/mat107/hm_read_mat107.F:308-314, 429-430``.
    """
    if isinstance(young1, PaperLightParams):
        p = young1
        young1 = p.young1
        young2 = p.young2
        young3 = p.young3
        nu21 = p.nu21
        g12 = p.g12
        g23 = p.g23
        g31 = p.g31
        rho = p.rho

    nu12 = (nu21 * young1 / young2) if abs(young2) > _EM20 else 0.0
    denom = 1.0 - nu12 * nu21
    if abs(denom) < _EM20:
        denom = 1.0e-6

    a11 = young1 / denom
    a12 = nu12 * young2 / denom
    a21 = nu21 * young1 / denom
    a22 = young2 / denom

    rho_pos = max(rho, _EM20)
    max_mod_solid = max(a11, a12, a21, a22, young3, g12, g23, g31)
    max_mod_shell = max(a11, a12, a21, a22, g12, g23, g31)

    ssp_solid = math.sqrt(max_mod_solid / rho_pos)
    ssp_shell = math.sqrt(max_mod_shell / rho_pos)

    return PaperLightConstants(
        a11=a11,
        a12=a12,
        a21=a21,
        a22=a22,
        nu12=nu12,
        nu21=nu21,
        young1=young1,
        young2=young2,
        young3=young3,
        g12=g12,
        g23=g23,
        g31=g31,
        ssp_solid=ssp_solid,
        ssp_shell=ssp_shell,
    )


def build_law107(mat: Any) -> PaperLightParams:
    """Extract PaperLightParams from a model entity or dictionary."""
    if isinstance(mat, PaperLightParams):
        return mat

    params: Dict[str, Any] = getattr(mat, "params", {}) if hasattr(mat, "params") else {}
    if isinstance(mat, dict):
        params = mat

    def _get(keys: Tuple[str, ...], default: Any) -> Any:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    return val
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    return val
            kl = k.lower()
            if hasattr(mat, kl):
                val = getattr(mat, kl)
                if val is not None:
                    return val
            if isinstance(params, dict) and kl in params:
                val = params[kl]
                if val is not None:
                    return val
        return default

    mid = int(_get(("id", "mid", "mat_id"), 1))
    title = str(_get(("title", "name"), ""))
    rho = float(_get(("rho", "rho0", "MAT_RHO"), 0.0))
    refer_rho = float(_get(("refer_rho", "rhor", "Refer_Rho"), rho))

    young1 = float(_get(("young1", "e1", "MAT_E1"), 0.0))
    young2 = float(_get(("young2", "e2", "MAT_E2"), young1))
    young3 = float(_get(("young3", "e3", "MAT_E3"), young1))
    nu21 = float(_get(("nu21", "MAT_NU21"), 0.0))
    g12 = float(_get(("g12", "MAT_G12"), 0.0))
    g23 = float(_get(("g23", "MAT_G23"), 0.0))
    g31 = float(_get(("g31", "g13", "MAT_G13"), 0.0))

    ires = int(_get(("ires", "MAT_IRES"), 2))
    if ires == 0:
        ires = 2
    itab = int(_get(("itab", "MAT_ITAB"), 0))
    ismooth = int(_get(("ismooth", "MAT_SMOOTH"), 1))
    if ismooth == 0:
        ismooth = 1

    xi1 = float(_get(("xi1", "MAT_XI1"), 0.0))
    xi2 = float(_get(("xi2", "MAT_XI2"), 0.0))
    g1c = float(_get(("g1c", "MAT_G1C"), 0.0))
    d1 = float(_get(("d1", "MAT_D1"), 0.0))
    d2 = float(_get(("d2", "MAT_D2"), 0.0))

    k1 = float(_get(("k1", "MAT_K1"), 0.0))
    k2 = float(_get(("k2", "MAT_K2"), 0.0))
    k3 = float(_get(("k3", "MAT_K3"), 0.0))
    k4 = float(_get(("k4", "MAT_K4"), 0.0))
    k5 = float(_get(("k5", "MAT_K5"), 0.0))
    k6 = float(_get(("k6", "MAT_K6"), 0.0))

    # Analytic
    def _nz(val: float) -> float:
        return val if val > 0.0 else _DEFAULT_INF

    sigy1 = _nz(float(_get(("sigy1", "MAT_SIGY1"), _DEFAULT_INF)))
    cini1 = _nz(float(_get(("cini1", "MAT_CINI1"), _DEFAULT_INF)))
    s1 = float(_get(("s1", "MAT_S1"), 0.0))

    sigy2 = _nz(float(_get(("sigy2", "MAT_SIGY2"), _DEFAULT_INF)))
    cini2 = _nz(float(_get(("cini2", "MAT_CINI2"), _DEFAULT_INF)))
    s2 = float(_get(("s2", "MAT_S2"), 0.0))

    sigy1c = _nz(float(_get(("sigy1c", "MAT_SIGY1C"), _DEFAULT_INF)))
    cini1c = _nz(float(_get(("cini1c", "MAT_CINI1C"), _DEFAULT_INF)))
    s1c = float(_get(("s1c", "MAT_S1C"), 0.0))

    sigy2c = _nz(float(_get(("sigy2c", "MAT_SIGY2C"), _DEFAULT_INF)))
    cini2c = _nz(float(_get(("cini2c", "MAT_CINI2C"), _DEFAULT_INF)))
    s2c = float(_get(("s2c", "MAT_S2C"), 0.0))

    sigyt = _nz(float(_get(("sigyt", "MAT_SIGYT"), _DEFAULT_INF)))
    cinit = _nz(float(_get(("cinit", "MAT_CINIT"), _DEFAULT_INF)))
    st = float(_get(("st", "MAT_ST"), 0.0))

    # Tabulated
    tab_yld1 = int(_get(("tab_yld1", "TAB_YLD1"), 0))
    xscale1 = float(_get(("xscale1", "MAT_Xscale1"), 1.0))
    yscale1 = float(_get(("yscale1", "MAT_Yscale1"), 1.0))

    tab_yld2 = int(_get(("tab_yld2", "TAB_YLD2"), 0))
    xscale2 = float(_get(("xscale2", "MAT_Xscale2"), 1.0))
    yscale2 = float(_get(("yscale2", "MAT_Yscale2"), 1.0))

    tab_yld1c = int(_get(("tab_yld1c", "TAB_YLD1C"), 0))
    xscale1c = float(_get(("xscale1c", "MAT_Xscale1C"), 1.0))
    yscale1c = float(_get(("yscale1c", "MAT_Yscale1C"), 1.0))

    tab_yld2c = int(_get(("tab_yld2c", "TAB_YLD2C"), 0))
    xscale2c = float(_get(("xscale2c", "MAT_Xscale2C"), 1.0))
    yscale2c = float(_get(("yscale2c", "MAT_Yscale2C"), 1.0))

    tab_yldt = int(_get(("tab_yldt", "tab_yld_t", "TAB_YLDT"), 0))
    xscalet = float(_get(("xscalet", "xscale_t", "MAT_XscaleT"), 1.0))
    yscalet = float(_get(("yscalet", "yscale_t", "MAT_YscaleT"), 1.0))

    curves = None
    c1 = _get(("curve_yld1", "c_yld1", "f_ct1"), None)
    c2 = _get(("curve_yld2", "c_yld2", "f_ct2"), None)
    c3 = _get(("curve_yld1c", "c_yld1c", "f_cc1"), None)
    c4 = _get(("curve_yld2c", "c_yld2c", "f_cc2"), None)
    ct = _get(("curve_yldt", "c_yldt", "f_s12"), None)
    if any(c is not None for c in (c1, c2, c3, c4, ct)):
        curves = (c1, c2, c3, c4, ct)

    return PaperLightParams(
        id=mid,
        title=title,
        rho=rho,
        refer_rho=refer_rho,
        young1=young1,
        young2=young2,
        young3=young3,
        nu21=nu21,
        g12=g12,
        g23=g23,
        g31=g31,
        ires=ires,
        itab=itab,
        ismooth=ismooth,
        xi1=xi1,
        xi2=xi2,
        g1c=g1c,
        d1=d1,
        d2=d2,
        k1=k1,
        k2=k2,
        k3=k3,
        k4=k4,
        k5=k5,
        k6=k6,
        sigy1=sigy1,
        cini1=cini1,
        s1=s1,
        sigy2=sigy2,
        cini2=cini2,
        s2=s2,
        sigy1c=sigy1c,
        cini1c=cini1c,
        s1c=s1c,
        sigy2c=sigy2c,
        cini2c=cini2c,
        s2c=s2c,
        sigyt=sigyt,
        cinit=cinit,
        st=st,
        tab_yld1=tab_yld1,
        xscale1=xscale1,
        yscale1=yscale1,
        tab_yld2=tab_yld2,
        xscale2=xscale2,
        yscale2=yscale2,
        tab_yld1c=tab_yld1c,
        xscale1c=xscale1c,
        yscale1c=yscale1c,
        tab_yld2c=tab_yld2c,
        xscale2c=xscale2c,
        yscale2c=yscale2c,
        tab_yldt=tab_yldt,
        xscalet=xscalet,
        yscalet=yscalet,
        curves=curves,
    )


def _eval_curve(curve: Any, x: float) -> float:
    """Evaluate a curve, function, callable, or tabulated lookup at abscissa x."""
    if curve is None:
        return 0.0
    if callable(curve):
        return float(curve(x))
    if hasattr(curve, "evaluate"):
        return float(curve.evaluate(x))
    if hasattr(curve, "interp"):
        return float(curve.interp(x))
    if hasattr(curve, "get_value"):
        return float(curve.get_value(x))
    if isinstance(curve, (tuple, list)) and len(curve) >= 2:
        xs, ys = curve[0], curve[1]
        return float(np.interp(x, xs, ys))
    if isinstance(curve, (int, float)):
        return float(curve)
    return 0.0


def resolve(mat: Any, model: Any, log: Any = None) -> None:
    """Resolve TAB_YLD1..T curve IDs against model functions/curves for LAW107."""
    p = getattr(mat, "params", {})
    if not isinstance(p, dict):
        p = {}

    curve_keys = [
        ("tab_yld1", "curve_yld1"),
        ("tab_yld2", "curve_yld2"),
        ("tab_yld1c", "curve_yld1c"),
        ("tab_yld2c", "curve_yld2c"),
        ("tab_yldt", "curve_yldt"),
    ]

    for fid_key, curve_key in curve_keys:
        fid = p.get(fid_key, getattr(mat, fid_key, 0))
        if fid and fid != 0:
            curve = None
            if hasattr(model, "curves") and fid in model.curves:
                curve = model.curves[fid]
            elif hasattr(model, "functions") and fid in model.functions:
                curve = model.functions[fid]
            elif hasattr(model, "tables") and fid in model.tables:
                curve = model.tables[fid]

            if curve is not None:
                if hasattr(mat, "params") and isinstance(mat.params, dict):
                    mat.params[curve_key] = curve
                if hasattr(mat, curve_key):
                    setattr(mat, curve_key, curve)
            elif log is not None and hasattr(log, "warning"):
                log.warning(
                    f"/MAT/LAW107/{getattr(mat, 'id', 0)}: function curve ID {fid} for {fid_key} not found in model",
                    "MAT CHECK",
                )


def compute_paper_light_yield_surface(
    sigxx: float,
    sigyy: float,
    sigxy: float,
    p: PaperLightParams,
    pla: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate indicator flags alpha_i, yield stresses R_i, and yield function phi.

    Fortran origin: ``mat107_newton.F:250-298``.
    """
    # 1. Indicator coefficients alpha_i
    alpha = np.zeros(5, dtype=np.float64)
    if (sigxx - p.xi1 * sigyy) > 0.0:
        alpha[0] = 1.0
    if (sigyy - p.xi2 * sigxx) > 0.0:
        alpha[1] = 1.0
    if (-sigxx) > 0.0:
        alpha[2] = 1.0
    if (-sigyy) > 0.0:
        alpha[3] = 1.0
    if abs(sigxy) > 0.0:
        alpha[4] = 1.0

    # 2. Hardening and yield stresses
    # pla indices: 0: global, 1: MD tension, 2: CD tension, 3: MD comp, 4: CD comp, 5: shear
    p1 = pla[1] if len(pla) > 1 else 0.0
    p2 = pla[2] if len(pla) > 2 else 0.0
    p3 = pla[3] if len(pla) > 3 else 0.0
    p4 = pla[4] if len(pla) > 4 else 0.0
    p5 = pla[5] if len(pla) > 5 else 0.0

    g1c_factor = math.sqrt(max(1.0 - p.g1c, 1.0e-6))

    curves = getattr(p, "curves", None)
    if p.itab > 0 and curves is not None:
        c1, c2, c3, c4, c5 = curves

        # Table 1: R1 (MD tension)
        if c1 is not None:
            x1 = p1 * p.xscale1
            r1 = _eval_curve(c1, x1) * p.yscale1
            dx = 1.0e-6
            dr1_dp = ((_eval_curve(c1, x1 + dx) - _eval_curve(c1, max(0.0, x1 - dx))) / (2.0 * dx)) * p.xscale1 * p.yscale1
        else:
            r1 = p.sigy1 + (1.0 / (p.cini1 + p.s1 * p1)) * p1
            denom1 = p.cini1 + p.s1 * p1
            dr1_dp = (1.0 / denom1) * (1.0 - (p.s1 * p1 / denom1)) if abs(denom1) > _EM20 else 0.0

        # Table 2: R2 (CD tension)
        if c2 is not None:
            x2 = p2 * p.xscale2
            r2 = _eval_curve(c2, x2) * p.yscale2
            dx = 1.0e-6
            dr2_dp = ((_eval_curve(c2, x2 + dx) - _eval_curve(c2, max(0.0, x2 - dx))) / (2.0 * dx)) * p.xscale2 * p.yscale2
        else:
            r2 = p.sigy2 + (1.0 / (p.cini2 + p.s2 * p2)) * p2
            denom2 = p.cini2 + p.s2 * p2
            dr2_dp = (1.0 / denom2) * (1.0 - (p.s2 * p2 / denom2)) if abs(denom2) > _EM20 else 0.0

        # Table 3: R1c (MD compression)
        if c3 is not None:
            x3 = p3 * p.xscale1c
            r1c = (_eval_curve(c3, x3) * p.yscale1c) / g1c_factor
            dx = 1.0e-6
            dr1c_dp = (((_eval_curve(c3, x3 + dx) - _eval_curve(c3, max(0.0, x3 - dx))) / (2.0 * dx)) * p.xscale1c * p.yscale1c) / g1c_factor
        else:
            r1c = (p.sigy1c + (1.0 / (p.cini1c + p.s1c * p3)) * p3) / g1c_factor
            denom3 = p.cini1c + p.s1c * p3
            dr1c_dp = ((1.0 / denom3) * (1.0 - (p.s1c * p3 / denom3)) / g1c_factor) if abs(denom3) > _EM20 else 0.0

        # Table 4: R2c (CD compression)
        if c4 is not None:
            x4 = p4 * p.xscale2c
            r2c = _eval_curve(c4, x4) * p.yscale2c
            dx = 1.0e-6
            dr2c_dp = ((_eval_curve(c4, x4 + dx) - _eval_curve(c4, max(0.0, x4 - dx))) / (2.0 * dx)) * p.xscale2c * p.yscale2c
        else:
            r2c = p.sigy2c + (1.0 / (p.cini2c + p.s2c * p4)) * p4
            denom4 = p.cini2c + p.s2c * p4
            dr2c_dp = (1.0 / denom4) * (1.0 - (p.s2c * p4 / denom4)) if abs(denom4) > _EM20 else 0.0

        # Table 5: Rt (shear)
        if c5 is not None:
            x5 = p5 * p.xscalet
            rt = _eval_curve(c5, x5) * p.yscalet
            dx = 1.0e-6
            drt_dp = ((_eval_curve(c5, x5 + dx) - _eval_curve(c5, max(0.0, x5 - dx))) / (2.0 * dx)) * p.xscalet * p.yscalet
        else:
            eta1 = 1.0 + alpha[2] * alpha[3] * p.d1
            eta2 = 1.0 + alpha[2] * alpha[3] * p.d2
            rt = eta1 * p.sigyt + eta2 * (1.0 / (p.cinit + p.st * p5)) * p5
            denom5 = p.cinit + p.st * p5
            drt_dp = eta2 * (1.0 / denom5) * (1.0 - (p.st * p5 / denom5)) if abs(denom5) > _EM20 else 0.0

        r_vec = np.array([r1, r2, r1c, r2c, rt], dtype=np.float64)
        dr_dp = np.array([dr1_dp, dr2_dp, dr1c_dp, dr2c_dp, drt_dp], dtype=np.float64)
    else:
        r1 = p.sigy1 + (1.0 / (p.cini1 + p.s1 * p1)) * p1
        r2 = p.sigy2 + (1.0 / (p.cini2 + p.s2 * p2)) * p2

        r1c = (p.sigy1c + (1.0 / (p.cini1c + p.s1c * p3)) * p3) / g1c_factor
        r2c = p.sigy2c + (1.0 / (p.cini2c + p.s2c * p4)) * p4

        eta1 = 1.0 + alpha[2] * alpha[3] * p.d1
        eta2 = 1.0 + alpha[2] * alpha[3] * p.d2
        rt = eta1 * p.sigyt + eta2 * (1.0 / (p.cinit + p.st * p5)) * p5

        r_vec = np.array([r1, r2, r1c, r2c, rt], dtype=np.float64)

        # 3. Derivatives of yield stresses with respect to plastic strains
        dr_dp = np.zeros(5, dtype=np.float64)
        denom1 = p.cini1 + p.s1 * p1
        dr_dp[0] = (1.0 / denom1) * (1.0 - (p.s1 * p1 / denom1)) if abs(denom1) > _EM20 else 0.0

        denom2 = p.cini2 + p.s2 * p2
        dr_dp[1] = (1.0 / denom2) * (1.0 - (p.s2 * p2 / denom2)) if abs(denom2) > _EM20 else 0.0

        denom3 = p.cini1c + p.s1c * p3
        dr_dp[2] = ((1.0 / denom3) * (1.0 - (p.s1c * p3 / denom3)) / g1c_factor) if abs(denom3) > _EM20 else 0.0

        denom4 = p.cini2c + p.s2c * p4
        dr_dp[3] = (1.0 / denom4) * (1.0 - (p.s2c * p4 / denom4)) if abs(denom4) > _EM20 else 0.0

        denom5 = p.cinit + p.st * p5
        dr_dp[4] = eta2 * (1.0 / denom5) * (1.0 - (p.st * p5 / denom5)) if abs(denom5) > _EM20 else 0.0

    # 4. Yield function phi
    phi = (
        alpha[0] * ((sigxx - p.xi1 * sigyy) / r1) ** 2
        + alpha[1] * ((sigyy - p.xi2 * sigxx) / r2) ** 2
        + alpha[2] * ((-sigxx) / r1c) ** 2
        + alpha[3] * ((-sigyy) / r2c) ** 2
        + alpha[4] * (abs(sigxy) / rt) ** 2
        - 1.0
    )

    return phi, alpha, r_vec, dr_dp


def compute_paper_light_normals(
    sigxx: float,
    sigyy: float,
    sigxy: float,
    p: PaperLightParams,
    alpha: np.ndarray,
    r_vec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute yield surface gradient n and non-associated plastic potential gradient np.

    Fortran origin: ``mat107_newton.F:339-359``.
    """
    r1, r2, r1c, r2c, rt = r_vec

    # 1. Normal to yield criterion n = dphi/dsig
    normxx = (
        2.0 * (alpha[0] / (r1 * r1)) * (sigxx - p.xi1 * sigyy)
        - 2.0 * (alpha[1] * p.xi2 / (r2 * r2)) * (sigyy - p.xi2 * sigxx)
        + 2.0 * (alpha[2] / (r1c * r1c)) * sigxx
    )
    normyy = (
        -2.0 * (alpha[0] * p.xi1 / (r1 * r1)) * (sigxx - p.xi1 * sigyy)
        + 2.0 * (alpha[1] / (r2 * r2)) * (sigyy - p.xi2 * sigxx)
        + 2.0 * (alpha[3] / (r2c * r2c)) * sigyy
    )
    sign_xy = 1.0 if sigxy >= 0.0 else -1.0
    normxy = (alpha[4] / (rt * rt)) * abs(sigxy) * sign_xy

    n_vec = np.array([normxx, normyy, normxy], dtype=np.float64)

    # 2. Beta coefficients for non-associated potential
    norm_sig = math.sqrt(sigxx * sigxx + sigyy * sigyy + 2.0 * sigxy * sigxy)
    norm_sig = max(norm_sig, _EM20)

    beta1 = 0.0
    beta2 = 0.0
    if (sigxx / norm_sig > _EM01) or (sigyy / norm_sig > _EM01):
        beta1 = 1.0
    if (sigxx / norm_sig < -_EM01) or (sigyy / norm_sig < -_EM01):
        beta2 = 1.0
    if (abs(sigxx) / norm_sig < _EM01) and (abs(sigyy) / norm_sig < _EM01):
        beta1 = 1.0
        beta2 = 1.0

    # 3. Derivatives of plastic potential
    normpxx = beta1 * (2.0 * sigxx + p.k2 * sigyy) + beta2 * (2.0 * sigxx + p.k4 * sigyy)
    normpyy = beta1 * (2.0 * p.k1 * sigyy + p.k2 * sigxx) + beta2 * (2.0 * p.k3 * sigyy + p.k4 * sigxx)
    normpxy = (beta1 * p.k5 + beta2 * p.k6) * sigxy

    norm_p = math.sqrt(normpxx * normpxx + normpyy * normpyy + 2.0 * normpxy * normpxy)
    norm_p = max(norm_p, _EM20)

    np_vec = np.array([normpxx / norm_p, normpyy / norm_p, normpxy / norm_p], dtype=np.float64)

    return n_vec, np_vec


def mat107_newton_update_single(
    sig_trial: np.ndarray,
    p: PaperLightParams,
    c: PaperLightConstants,
    pla: np.ndarray,
    niter: int = 3,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Execute Newton-Raphson cutting-plane return mapping for one element/IP (IRES=2).

    Fortran origin: ``mat107_newton.F:317-480``.
    Returns:
        sig_new (3,), pla_new (6,), dlam_tot, hardp, sigy_scalar
    """
    sigxx = float(sig_trial[0])
    sigyy = float(sig_trial[1])
    sigxy = float(sig_trial[2])

    pla_curr = pla.copy()
    dlam_tot = 0.0

    phi, alpha, r_vec, dr_dp = compute_paper_light_yield_surface(sigxx, sigyy, sigxy, p, pla_curr)

    if phi <= 0.0:
        # Elastic
        sigy_sc = math.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2 + r_vec[3]**2 + 2.0 * r_vec[4]**2)
        return np.array([sigxx, sigyy, sigxy]), pla_curr, 0.0, 0.0, sigy_sc

    hardp = 0.0
    for _ in range(niter):
        n_vec, np_vec = compute_paper_light_normals(sigxx, sigyy, sigxy, p, alpha, r_vec)
        normxx, normyy, normxy = n_vec
        normpxx, normpyy, normpxy = np_vec

        # dfdsig2 = n : C : np
        dfdsig2 = (
            normxx * (c.a11 * normpxx + c.a12 * normpyy)
            + normyy * (c.a21 * normpxx + c.a22 * normpyy)
            + 4.0 * normxy * normpxy * c.g12
        )

        r1, r2, r1c, r2c, rt = r_vec
        dphi_dr1 = -2.0 * alpha[0] * (((sigxx - p.xi1 * sigyy) ** 2) / (r1 ** 3))
        dphi_dr2 = -2.0 * alpha[1] * (((sigyy - p.xi2 * sigxx) ** 2) / (r2 ** 3))
        dphi_dr1c = -2.0 * alpha[2] * (sigxx ** 2) / (r1c ** 3)
        dphi_dr2c = -2.0 * alpha[3] * (sigyy ** 2) / (r2c ** 3)
        dphi_drt = -2.0 * alpha[4] * (sigxy ** 2) / (rt ** 3)

        hardp = math.sqrt(
            alpha[0] * (dr_dp[0] ** 2)
            + alpha[1] * (dr_dp[1] ** 2)
            + alpha[2] * (dr_dp[2] ** 2)
            + alpha[3] * (dr_dp[3] ** 2)
            + 2.0 * alpha[4] * (dr_dp[4] ** 2)
        )

        dphi_dpla = (
            dphi_dr1 * dr_dp[0] * alpha[0]
            + dphi_dr2 * dr_dp[1] * alpha[1]
            + dphi_dr1c * dr_dp[2] * alpha[2]
            + dphi_dr2c * dr_dp[3] * alpha[3]
            + dphi_drt * dr_dp[4] * alpha[4]
        )

        dphi_dlam = -dfdsig2 + dphi_dpla
        sign_dlam = 1.0 if dphi_dlam >= 0.0 else -1.0
        dphi_dlam = sign_dlam * max(abs(dphi_dlam), _EM20)

        # Multiplier
        dlam = -phi / dphi_dlam
        dlam_tot += dlam

        dpxx = dlam * normpxx
        dpyy = dlam * normpyy
        dpxy = 2.0 * dlam * normpxy

        # Stress update
        sigxx -= (c.a11 * dpxx + c.a12 * dpyy)
        sigyy -= (c.a21 * dpxx + c.a22 * dpyy)
        sigxy -= c.g12 * dpxy

        # Plastic strains update
        pla_curr[0] += dlam
        pla_curr[1] += alpha[0] * dlam
        pla_curr[2] += alpha[1] * dlam
        pla_curr[3] += alpha[2] * dlam
        pla_curr[4] += alpha[3] * dlam
        pla_curr[5] += alpha[4] * dlam

        # Re-evaluate yield surface
        phi, alpha, r_vec, dr_dp = compute_paper_light_yield_surface(sigxx, sigyy, sigxy, p, pla_curr)

    sigy_sc = math.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2 + r_vec[3]**2 + 2.0 * r_vec[4]**2)
    return np.array([sigxx, sigyy, sigxy]), pla_curr, dlam_tot, hardp, sigy_sc


def mat107_nice_update_single(
    sigo: np.ndarray,
    deps: np.ndarray,
    p: PaperLightParams,
    c: PaperLightConstants,
    pla: np.ndarray,
    phi_prev: float,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Execute Nice explicit return mapping for one element/IP (IRES=1).

    Fortran origin: ``mat107_nice.F:304-480``.
    Returns:
        sig_new (3,), pla_new (6,), dlam, hardp, sigy_scalar, phi_new
    """
    sigoxx = float(sigo[0])
    sigoyy = float(sigo[1])
    sigoxy = float(sigo[2])

    dsigxx = c.a11 * deps[0] + c.a12 * deps[1]
    dsigyy = c.a21 * deps[0] + c.a22 * deps[1]
    dsigxy = deps[2] * c.g12

    signxx = sigoxx + dsigxx
    signyy = sigoyy + dsigyy
    signxy = sigoxy + dsigxy

    pla_curr = pla.copy()
    phi_tr, alpha, r_vec, dr_dp = compute_paper_light_yield_surface(signxx, signyy, signxy, p, pla_curr)

    if phi_tr <= 0.0:
        sigy_sc = math.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2 + r_vec[3]**2 + 2.0 * r_vec[4]**2)
        return np.array([signxx, signyy, signxy]), pla_curr, 0.0, 0.0, sigy_sc, phi_tr

    # Use old stress state for normals in Nice method
    n_vec, np_vec = compute_paper_light_normals(sigoxx, sigoyy, sigoxy, p, alpha, r_vec)
    normxx, normyy, normxy = n_vec
    normpxx, normpyy, normpxy = np_vec

    dfdsig2 = (
        normxx * (c.a11 * normpxx + c.a12 * normpyy)
        + normyy * (c.a21 * normpxx + c.a22 * normpyy)
        + 4.0 * normxy * normpxy * c.g12
    )

    r1, r2, r1c, r2c, rt = r_vec
    dphi_dr1 = -2.0 * alpha[0] * (((sigoxx - p.xi1 * sigoyy) ** 2) / (r1 ** 3))
    dphi_dr2 = -2.0 * alpha[1] * (((sigoyy - p.xi2 * sigoxx) ** 2) / (r2 ** 3))
    dphi_dr1c = -2.0 * alpha[2] * (sigoxx ** 2) / (r1c ** 3)
    dphi_dr2c = -2.0 * alpha[3] * (sigoyy ** 2) / (r2c ** 3)
    dphi_drt = -2.0 * alpha[4] * (sigoxy ** 2) / (rt ** 3)

    hardp = math.sqrt(
        alpha[0] * (dr_dp[0] ** 2)
        + alpha[1] * (dr_dp[1] ** 2)
        + alpha[2] * (dr_dp[2] ** 2)
        + alpha[3] * (dr_dp[3] ** 2)
        + 2.0 * alpha[4] * (dr_dp[4] ** 2)
    )

    dphi_dpla = (
        dphi_dr1 * dr_dp[0] * alpha[0]
        + dphi_dr2 * dr_dp[1] * alpha[1]
        + dphi_dr1c * dr_dp[2] * alpha[2]
        + dphi_dr2c * dr_dp[3] * alpha[3]
        + dphi_drt * dr_dp[4] * alpha[4]
    )

    dphi_dlam = -dfdsig2 + dphi_dpla
    sign_dlam = 1.0 if dphi_dlam >= 0.0 else -1.0
    dphi_dlam = sign_dlam * max(abs(dphi_dlam), _EM20)

    # Increment of yield surface
    dphi = normxx * dsigxx + normyy * dsigyy + 2.0 * normxy * dsigxy

    dlam = -(phi_prev + dphi) / dphi_dlam
    dlam = max(dlam, 0.0)

    dpxx = dlam * normpxx
    dpyy = dlam * normpyy
    dpxy = 2.0 * dlam * normpxy

    signxx = sigoxx + dsigxx - (c.a11 * dpxx + c.a12 * dpyy)
    signyy = sigoyy + dsigyy - (c.a21 * dpxx + c.a22 * dpyy)
    signxy = sigoxy + dsigxy - c.g12 * dpxy

    pla_curr[0] += dlam
    pla_curr[1] += alpha[0] * dlam
    pla_curr[2] += alpha[1] * dlam
    pla_curr[3] += alpha[2] * dlam
    pla_curr[4] += alpha[3] * dlam
    pla_curr[5] += alpha[4] * dlam

    phi_new, _, r_vec, _ = compute_paper_light_yield_surface(signxx, signyy, signxy, p, pla_curr)
    sigy_sc = math.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2 + r_vec[3]**2 + 2.0 * r_vec[4]**2)

    return np.array([signxx, signyy, signxy]), pla_curr, dlam, hardp, sigy_sc, phi_new


def solid_update(
    mat_or_sig: Any,
    sig_or_deps: Any = None,
    deps_or_mat: Any = None,
    dt: float = 1.0e-6,
    epsp: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Any:
    """Engine 3D solid continuum constitutive update for /MAT/LAW107.

    Fortran origin: ``engine/source/materials/mat/mat107/sigeps107.F``.
    """
    if hasattr(mat_or_sig, "law") or hasattr(mat_or_sig, "id") or hasattr(mat_or_sig, "young1") or isinstance(mat_or_sig, (PaperLightParams, dict)):
        mat = mat_or_sig
        sig = sig_or_deps
        deps = deps_or_mat
    else:
        sig = mat_or_sig
        deps = sig_or_deps
        mat = deps_or_mat

    p = build_law107(mat)
    c = compute_paper_light_constants(
        p.young1, p.young2, p.young3, p.nu21, p.g12, p.g23, p.g31, p.rho
    )

    if sig is None:
        sig = np.zeros(6, dtype=np.float64)
    if deps is None:
        deps = np.zeros(6, dtype=np.float64)

    sig_arr = np.asarray(sig, dtype=np.float64)
    deps_arr = np.asarray(deps, dtype=np.float64)
    is_1d = sig_arr.ndim == 1
    if is_1d:
        sig_arr = sig_arr[np.newaxis, :]
        deps_arr = deps_arr[np.newaxis, :]

    n_elem = sig_arr.shape[0]
    sig_out = np.zeros_like(sig_arr)
    dpla_out = np.zeros(n_elem, dtype=np.float64)

    if extra is None:
        extra = {}

    pla_in = extra.get("pla")
    if pla_in is None:
        pla_arr = np.zeros((n_elem, 6), dtype=np.float64)
    else:
        pla_arr = np.array(pla_in, dtype=np.float64, copy=True)
        if pla_arr.ndim == 1:
            if n_elem == 1 and pla_arr.shape[0] == 6:
                pla_arr = pla_arr[np.newaxis, :]
            else:
                pla_arr = np.zeros((n_elem, 6), dtype=np.float64)
        elif pla_arr.ndim == 2 and pla_arr.shape == (n_elem, 6):
            pass
        else:
            pla_arr = np.zeros((n_elem, 6), dtype=np.float64)

    uvar_in = extra.get("uvar")
    if uvar_in is None:
        uvar_arr = np.zeros((n_elem, 1), dtype=np.float64)
    else:
        uvar_arr = np.array(uvar_in, dtype=np.float64, copy=True)
        if uvar_arr.ndim == 0:
            uvar_arr = uvar_arr.reshape((1, 1))
        elif uvar_arr.ndim == 1:
            if n_elem == 1:
                uvar_arr = uvar_arr.reshape((1, -1))
                if uvar_arr.shape[1] == 0:
                    uvar_arr = np.zeros((1, 1), dtype=np.float64)
            else:
                uvar_arr = uvar_arr[:, np.newaxis]
        elif uvar_arr.ndim == 2 and uvar_arr.shape[0] == n_elem:
            pass
        else:
            uvar_arr = np.zeros((n_elem, 1), dtype=np.float64)

    epsd_in = extra.get("epsd")
    if epsd_in is None:
        epsd_arr = np.zeros((n_elem, 6), dtype=np.float64)
    else:
        epsd_arr = np.array(epsd_in, dtype=np.float64, copy=True)
        if epsd_arr.ndim == 1:
            if n_elem == 1 and epsd_arr.shape[0] == 6:
                epsd_arr = epsd_arr[np.newaxis, :]
            else:
                epsd_arr = np.zeros((n_elem, 6), dtype=np.float64)
        elif epsd_arr.ndim == 2 and epsd_arr.shape == (n_elem, 6):
            pass
        else:
            epsd_arr = np.zeros((n_elem, 6), dtype=np.float64)

    soundsp = np.full(n_elem, c.ssp_solid, dtype=np.float64)
    sigy_arr = np.zeros(n_elem, dtype=np.float64)

    dt_pos = max(dt, _EM20)

    for i in range(n_elem):
        # 3D continuum trial stresses
        sigo = sig_arr[i]
        de = deps_arr[i]

        sigxx_tr = sigo[0] + c.a11 * de[0] + c.a12 * de[1]
        sigyy_tr = sigo[1] + c.a21 * de[0] + c.a22 * de[1]
        sigzz_tr = sigo[2] + c.young3 * de[2]
        sigxy_tr = sigo[3] + c.g12 * de[3]
        sigyz_tr = sigo[4] + c.g23 * de[4]
        sigzx_tr = sigo[5] + c.g31 * de[5]

        if p.ires == 1:
            # Nice explicit return
            sig_inplane_new, pla_new, dlam, hardp, sigy_sc, phi_new = mat107_nice_update_single(
                np.array([sigo[0], sigo[1], sigo[3]]),
                np.array([de[0], de[1], de[3]]),
                p,
                c,
                pla_arr[i],
                float(uvar_arr[i, 0]),
            )
            uvar_arr[i, 0] = phi_new
        else:
            # Newton-Raphson cutting plane (default)
            sig_inplane_new, pla_new, dlam, hardp, sigy_sc = mat107_newton_update_single(
                np.array([sigxx_tr, sigyy_tr, sigxy_tr]),
                p,
                c,
                pla_arr[i],
                niter=3,
            )

        sig_out[i, 0] = sig_inplane_new[0]
        sig_out[i, 1] = sig_inplane_new[1]
        sig_out[i, 2] = sigzz_tr
        sig_out[i, 3] = sig_inplane_new[2]
        sig_out[i, 4] = sigyz_tr
        sig_out[i, 5] = sigzx_tr

        dpla_out[i] = dlam
        dpla_vec = pla_new - pla_arr[i]
        pla_arr[i] = pla_new

        epsd_arr[i] = dpla_vec / dt_pos
        sigy_arr[i] = sigy_sc

    # Write back to mutable extra dictionary
    if isinstance(extra, dict):
        if "pla" in extra and isinstance(extra["pla"], np.ndarray):
            try:
                extra["pla"][:] = pla_arr[0] if is_1d and extra["pla"].ndim == 1 else pla_arr
            except Exception:
                extra["pla"] = pla_arr[0] if is_1d else pla_arr
        else:
            extra["pla"] = pla_arr[0] if is_1d else pla_arr

        if "uvar" in extra and isinstance(extra["uvar"], np.ndarray):
            try:
                extra["uvar"][:] = uvar_arr[0] if is_1d and extra["uvar"].ndim == 1 else uvar_arr
            except Exception:
                extra["uvar"] = uvar_arr[0] if is_1d else uvar_arr
        else:
            extra["uvar"] = uvar_arr[0] if is_1d else uvar_arr

        if "epsd" in extra and isinstance(extra["epsd"], np.ndarray):
            try:
                extra["epsd"][:] = epsd_arr[0] if is_1d and extra["epsd"].ndim == 1 else epsd_arr
            except Exception:
                extra["epsd"] = epsd_arr[0] if is_1d else epsd_arr
        else:
            extra["epsd"] = epsd_arr[0] if is_1d else epsd_arr

        extra["sigy"] = sigy_arr[0] if is_1d else sigy_arr
        extra["sound_speed"] = c.ssp_solid

    extra_out = {
        "pla": pla_arr[0] if is_1d else pla_arr,
        "uvar": uvar_arr[0] if is_1d else uvar_arr,
        "epsd": epsd_arr[0] if is_1d else epsd_arr,
        "sigy": sigy_arr[0] if is_1d else sigy_arr,
        "sound_speed": c.ssp_solid,
    }

    if return_tuple:
        if is_1d:
            return sig_out[0], float(pla_arr[0, 0]), float(soundsp[0])
        return sig_out, pla_arr[:, 0], soundsp

    if is_1d:
        return sig_out[0], float(dpla_out[0]), pla_arr[0, 0], extra_out
    return sig_out, np.mean(dpla_out), pla_arr[:, 0], extra_out


def shell_update(
    mat_or_sig: Any,
    sig_or_deps: Any = None,
    deps_or_mat: Any = None,
    dt: float = 1.0e-6,
    epsp: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Any:
    """Engine 2D shell plane-stress constitutive update for /MAT/LAW107.

    Fortran origin: ``engine/source/materials/mat/mat107/sigeps107c.F``.
    """
    if hasattr(mat_or_sig, "law") or hasattr(mat_or_sig, "id") or hasattr(mat_or_sig, "young1") or isinstance(mat_or_sig, (PaperLightParams, dict)):
        mat = mat_or_sig
        sig = sig_or_deps
        deps = deps_or_mat
    else:
        sig = mat_or_sig
        deps = sig_or_deps
        mat = deps_or_mat

    p = build_law107(mat)
    c = compute_paper_light_constants(
        p.young1, p.young2, p.young3, p.nu21, p.g12, p.g23, p.g31, p.rho
    )

    if sig is None:
        sig = np.zeros(3, dtype=np.float64)
    if deps is None:
        deps = np.zeros(3, dtype=np.float64)

    sig_arr = np.asarray(sig, dtype=np.float64)
    deps_arr = np.asarray(deps, dtype=np.float64)
    is_1d = sig_arr.ndim == 1
    if is_1d:
        sig_arr = sig_arr[np.newaxis, :]
        deps_arr = deps_arr[np.newaxis, :]

    n_elem = sig_arr.shape[0]
    n_stress = sig_arr.shape[1]
    sig_out = np.zeros_like(sig_arr)
    dpla_out = np.zeros(n_elem, dtype=np.float64)

    if extra is None:
        extra = {}

    shf = float(extra.get("shf", 5.0 / 6.0))

    pla_in = extra.get("pla")
    if pla_in is None:
        pla_arr = np.zeros((n_elem, 6), dtype=np.float64)
    else:
        pla_arr = np.array(pla_in, dtype=np.float64, copy=True)
        if pla_arr.ndim == 1:
            if n_elem == 1 and pla_arr.shape[0] == 6:
                pla_arr = pla_arr[np.newaxis, :]
            else:
                pla_arr = np.zeros((n_elem, 6), dtype=np.float64)
        elif pla_arr.ndim == 2 and pla_arr.shape == (n_elem, 6):
            pass
        else:
            pla_arr = np.zeros((n_elem, 6), dtype=np.float64)

    uvar_in = extra.get("uvar")
    if uvar_in is None:
        uvar_arr = np.zeros((n_elem, 1), dtype=np.float64)
    else:
        uvar_arr = np.array(uvar_in, dtype=np.float64, copy=True)
        if uvar_arr.ndim == 0:
            uvar_arr = uvar_arr.reshape((1, 1))
        elif uvar_arr.ndim == 1:
            if n_elem == 1:
                uvar_arr = uvar_arr.reshape((1, -1))
                if uvar_arr.shape[1] == 0:
                    uvar_arr = np.zeros((1, 1), dtype=np.float64)
            else:
                uvar_arr = uvar_arr[:, np.newaxis]
        elif uvar_arr.ndim == 2 and uvar_arr.shape[0] == n_elem:
            pass
        else:
            uvar_arr = np.zeros((n_elem, 1), dtype=np.float64)

    epsd_in = extra.get("epsd")
    if epsd_in is None:
        epsd_arr = np.zeros((n_elem, 6), dtype=np.float64)
    else:
        epsd_arr = np.array(epsd_in, dtype=np.float64, copy=True)
        if epsd_arr.ndim == 1:
            if n_elem == 1 and epsd_arr.shape[0] == 6:
                epsd_arr = epsd_arr[np.newaxis, :]
            else:
                epsd_arr = np.zeros((n_elem, 6), dtype=np.float64)
        elif epsd_arr.ndim == 2 and epsd_arr.shape == (n_elem, 6):
            pass
        else:
            epsd_arr = np.zeros((n_elem, 6), dtype=np.float64)

    soundsp = np.full(n_elem, c.ssp_shell, dtype=np.float64)
    sigy_arr = np.zeros(n_elem, dtype=np.float64)

    dt_pos = max(dt, _EM20)

    for i in range(n_elem):
        sigo = sig_arr[i]
        de = deps_arr[i]

        # In-plane stress update
        sigxx_tr = sigo[0] + c.a11 * de[0] + c.a12 * de[1]
        sigyy_tr = sigo[1] + c.a21 * de[0] + c.a22 * de[1]
        # In shell: deps_arr can be length 3 (xx, yy, xy) or 5 (xx, yy, xy, yz, zx) or 6 (xx, yy, zz, xy, yz, zx)
        if n_stress == 3:
            sigxy_tr = sigo[2] + c.g12 * de[2]
        elif n_stress == 5:
            sigxy_tr = sigo[2] + c.g12 * de[2]
            sigyz_tr = sigo[3] + c.g23 * shf * de[3]
            sigzx_tr = sigo[4] + c.g31 * shf * de[4]
        else:
            sigxy_tr = sigo[3] + c.g12 * de[3]
            sigyz_tr = sigo[4] + c.g23 * shf * de[4]
            sigzx_tr = sigo[5] + c.g31 * shf * de[5]

        if p.ires == 1:
            sig_inplane_new, pla_new, dlam, hardp, sigy_sc, phi_new = mat107_nice_update_single(
                np.array([sigo[0], sigo[1], sigo[2] if n_stress <= 5 else sigo[3]]),
                np.array([de[0], de[1], de[2] if n_stress <= 5 else de[3]]),
                p,
                c,
                pla_arr[i],
                float(uvar_arr[i, 0]),
            )
            uvar_arr[i, 0] = phi_new
        else:
            sig_inplane_new, pla_new, dlam, hardp, sigy_sc = mat107_newton_update_single(
                np.array([sigxx_tr, sigyy_tr, sigxy_tr]),
                p,
                c,
                pla_arr[i],
                niter=3,
            )

        if n_stress == 3:
            sig_out[i, 0] = sig_inplane_new[0]
            sig_out[i, 1] = sig_inplane_new[1]
            sig_out[i, 2] = sig_inplane_new[2]
        elif n_stress == 5:
            sig_out[i, 0] = sig_inplane_new[0]
            sig_out[i, 1] = sig_inplane_new[1]
            sig_out[i, 2] = sig_inplane_new[2]
            sig_out[i, 3] = sigyz_tr
            sig_out[i, 4] = sigzx_tr
        else:
            sig_out[i, 0] = sig_inplane_new[0]
            sig_out[i, 1] = sig_inplane_new[1]
            sig_out[i, 2] = 0.0  # Plane-stress sigma_zz = 0
            sig_out[i, 3] = sig_inplane_new[2]
            sig_out[i, 4] = sigyz_tr
            sig_out[i, 5] = sigzx_tr

        dpla_out[i] = dlam
        dpla_vec = pla_new - pla_arr[i]
        pla_arr[i] = pla_new

        epsd_arr[i] = dpla_vec / dt_pos
        sigy_arr[i] = sigy_sc

    # Write back to mutable extra dictionary
    if isinstance(extra, dict):
        if "pla" in extra and isinstance(extra["pla"], np.ndarray):
            try:
                extra["pla"][:] = pla_arr[0] if is_1d and extra["pla"].ndim == 1 else pla_arr
            except Exception:
                extra["pla"] = pla_arr[0] if is_1d else pla_arr
        else:
            extra["pla"] = pla_arr[0] if is_1d else pla_arr

        if "uvar" in extra and isinstance(extra["uvar"], np.ndarray):
            try:
                extra["uvar"][:] = uvar_arr[0] if is_1d and extra["uvar"].ndim == 1 else uvar_arr
            except Exception:
                extra["uvar"] = uvar_arr[0] if is_1d else uvar_arr
        else:
            extra["uvar"] = uvar_arr[0] if is_1d else uvar_arr

        if "epsd" in extra and isinstance(extra["epsd"], np.ndarray):
            try:
                extra["epsd"][:] = epsd_arr[0] if is_1d and extra["epsd"].ndim == 1 else epsd_arr
            except Exception:
                extra["epsd"] = epsd_arr[0] if is_1d else epsd_arr
        else:
            extra["epsd"] = epsd_arr[0] if is_1d else epsd_arr

        extra["sigy"] = sigy_arr[0] if is_1d else sigy_arr
        extra["sound_speed"] = c.ssp_shell

    extra_out = {
        "pla": pla_arr[0] if is_1d else pla_arr,
        "uvar": uvar_arr[0] if is_1d else uvar_arr,
        "epsd": epsd_arr[0] if is_1d else epsd_arr,
        "sigy": sigy_arr[0] if is_1d else sigy_arr,
        "sound_speed": c.ssp_shell,
    }

    if return_tuple:
        if is_1d:
            return sig_out[0], float(pla_arr[0, 0]), float(soundsp[0])
        return sig_out, pla_arr[:, 0], soundsp

    if is_1d:
        return sig_out[0], float(dpla_out[0]), pla_arr[0, 0], extra_out
    return sig_out, np.mean(dpla_out), pla_arr[:, 0], extra_out


def sound_speed(mat: Any, rho: Optional[float] = None) -> float:
    """Dilatational sound speed for 3D continuum solids."""
    p = build_law107(mat)
    dens = rho if rho is not None and rho > 0.0 else p.rho
    c = compute_paper_light_constants(
        p.young1, p.young2, p.young3, p.nu21, p.g12, p.g23, p.g31, dens
    )
    return c.ssp_solid


def sound_speed_solid(mat: Any, rho: Optional[float] = None) -> float:
    """Synonym for sound_speed."""
    return sound_speed(mat, rho)


def sound_speed_shell(mat: Any, rho: Optional[float] = None) -> float:
    """Plane-stress sound speed for 2D shells."""
    p = build_law107(mat)
    dens = rho if rho is not None and rho > 0.0 else p.rho
    c = compute_paper_light_constants(
        p.young1, p.young2, p.young3, p.nu21, p.g12, p.g23, p.g31, dens
    )
    return c.ssp_shell


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[float] = None,
    epsp_incr: Optional[float] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent algorithmic tangent for 3D continuum solids (6x6)."""
    p = build_law107(mat)
    c = compute_paper_light_constants(
        p.young1, p.young2, p.young3, p.nu21, p.g12, p.g23, p.g31, p.rho
    )

    c_mat = np.zeros((6, 6), dtype=np.float64)
    c_mat[0, 0] = c.a11
    c_mat[0, 1] = c.a12
    c_mat[1, 0] = c.a21
    c_mat[1, 1] = c.a22
    c_mat[2, 2] = c.young3
    c_mat[3, 3] = c.g12
    c_mat[4, 4] = c.g23
    c_mat[5, 5] = c.g31

    if sig is None or epsp_incr is None or epsp_incr <= 0.0:
        return c_mat

    pla = extra.get("pla", np.zeros(6, dtype=np.float64)) if extra else np.zeros(6, dtype=np.float64)
    phi, alpha, r_vec, dr_dp = compute_paper_light_yield_surface(sig[0], sig[1], sig[3], p, pla)
    if phi < -1.0e-6:
        return c_mat

    n_vec, np_vec = compute_paper_light_normals(sig[0], sig[1], sig[3], p, alpha, r_vec)
    n_full = np.array([n_vec[0], n_vec[1], 0.0, 2.0 * n_vec[2], 0.0, 0.0], dtype=np.float64)
    np_full = np.array([np_vec[0], np_vec[1], 0.0, 2.0 * np_vec[2], 0.0, 0.0], dtype=np.float64)

    c_np = c_mat @ np_full
    n_c = n_full @ c_mat
    denom = np.dot(n_full, c_np) + max(abs(dr_dp[0]), 1.0)
    if abs(denom) > _EM20:
        c_mat -= np.outer(c_np, n_c) / denom

    return c_mat


def consistent_shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[float] = None,
    epsp_incr: Optional[float] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent algorithmic membrane tangent for 2D shells (3x3)."""
    p = build_law107(mat)
    c = compute_paper_light_constants(
        p.young1, p.young2, p.young3, p.nu21, p.g12, p.g23, p.g31, p.rho
    )

    c_mat = np.zeros((3, 3), dtype=np.float64)
    c_mat[0, 0] = c.a11
    c_mat[0, 1] = c.a12
    c_mat[1, 0] = c.a21
    c_mat[1, 1] = c.a22
    c_mat[2, 2] = c.g12

    if sig is None or epsp_incr is None or epsp_incr <= 0.0:
        return c_mat

    pla = extra.get("pla", np.zeros(6, dtype=np.float64)) if extra else np.zeros(6, dtype=np.float64)
    sigxy = sig[2] if len(sig) <= 5 else sig[3]
    phi, alpha, r_vec, dr_dp = compute_paper_light_yield_surface(sig[0], sig[1], sigxy, p, pla)
    if phi < -1.0e-6:
        return c_mat

    n_vec, np_vec = compute_paper_light_normals(sig[0], sig[1], sigxy, p, alpha, r_vec)
    n_3 = np.array([n_vec[0], n_vec[1], 2.0 * n_vec[2]], dtype=np.float64)
    np_3 = np.array([np_vec[0], np_vec[1], 2.0 * np_vec[2]], dtype=np.float64)

    c_np = c_mat @ np_3
    n_c = n_3 @ c_mat
    denom = np.dot(n_3, c_np) + max(abs(dr_dp[0]), 1.0)
    if abs(denom) > _EM20:
        c_mat -= np.outer(c_np, n_c) / denom

    return c_mat


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Internal variables shape requirements for /MAT/LAW107."""
    if nip is not None:
        return {
            "uvar": (nip, 1),
            "pla": (nip, 6),
            "epsd": (nip, 6),
        }
    return {
        "uvar": (1,),
        "pla": (6,),
        "epsd": (6,),
    }

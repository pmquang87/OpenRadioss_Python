"""
LAW90 — Tabulated foam with hysteresis and strain rate dependency (/MAT/LAW90, /MAT/HYST_FOAM).

Fortran upstream sources:
- starter/source/materials/mat/mat090/hm_read_mat90.F
- starter/source/materials/mat/mat090/law90_upd.F
- engine/source/materials/mat/mat090/sigeps90.F

Theory & Formulation (sigeps90):
--------------------------------
LAW90 models crushable foam materials with tabulated stress-strain curves, strain rate
dependency, hysteretic energy dissipation during unloading, and tensile stress cutoff.
It is formulated for 3D continuum solid elements.

1. Kinematics & Principal Stretch Decomposition:
   Total strain tensor epsilon is formed from strain increments in the global frame.
   Spectral decomposition A = V * diag(w) * V^T computes the 3 principal logarithmic
   strains w_1, w_2, w_3 and orthonormal principal direction eigenvectors V.
   Principal stretches are lambda_k = exp(w_k).
   Engineering strains used for tabulated curve lookup are defined as:
       e_k = 1.0 - lambda_k
   where e_k > 0 in compression and e_k < 0 in tension.
   The total strain norm is EPST = sqrt(e_1^2 + e_2^2 + e_3^2).

2. Principal Strain Rates & Filtering:
   The strain rate tensor L = deps / dt is transformed onto principal directions:
       eps_dot_k = (V^T * L * V)_kk
   Engineering strain rate in each principal direction:
       e_dot_k = eps_dot_k * lambda_k
   Effective scalar strain rate:
       eps_dot_eff = sqrt(e_dot_1^2 + e_dot_2^2 + e_dot_3^2)
   Optional exponential smoothing (if Ismooth > 0 and Fcut > 0):
       alpha_rate = min(1.0, 2.0 * pi * Fcut * dt)
       eps_dot_filtered = alpha_rate * eps_dot_eff + (1.0 - alpha_rate) * eps_dot_prev

3. Loading vs. Unloading Detection:
   Quasi-static energy W_stat is computed from the static loading curve (curve 1):
       W_stat = 0.5 * sum(e_k * S_stat_k)
   Increment: Delta_W = W_stat - W_stat_prev; Delta_eps = EPST - EPST_prev.
   If Delta_W >= 0 or Delta_eps >= 0:
       Loading branch: dynamic rate interpolation is active.
   Else:
       Unloading branch: strain rate is clamped to previous rate, and unloading
       follows the quasi-static curve (upstream comment: "! unloading is quasi-static").

4. Hysteretic Energy Dissipation (Damage):
   Cumulative deformation energy:
       W_cum = max(0.0, W_cum_prev + 0.5 * (YLD + YLD_prev) * Delta_eps)
       W_max = max(W_max_prev, W_cum)
   During unloading (if W_max > 0 and Hys < 1.0):
       ratio = clip(W_cum / W_max, 0.0, 1.0)
       DAM = 1.0 - ratio^shape
       DAM = DAM^alpha
       DAM = 1.0 - (1.0 - Hys) * DAM
       S_k = DAM * S_k
   At the reversal point (W_cum = W_max), DAM = 1.0 (smooth, continuous transition).
   As the material unloads to zero strain, DAM -> Hys.

5. Tensile Cutoff & Cauchy Stress:
   Nominal tensile stresses are T_k = -S_k.
   Under tension, T_k is limited by the tensile cutoff TCUT (scaled by DAM on unloading).
   True Cauchy stress in principal directions accounts for lateral stretches:
       sigma_1 = T_1 / (lambda_2 * lambda_3)
       sigma_2 = T_2 / (lambda_1 * lambda_3)
       sigma_3 = T_3 / (lambda_1 * lambda_2)
   Global Cauchy stress tensor is reconstructed:
       sigma = V * diag(sigma_k) * V^T

6. Modulus Evolution & Sound Speed:
   Evolving modulus E_new relaxes from E0 towards E_max with residual strain memory
   EPSS = clip(EPST - YLD / E_old, 0.0, 1.0):
       E_new = min(E_max, E0 + (E_max - E0) * EPSS)
   Acoustic sound speed is based on the P-wave modulus:
       c = sqrt(C_11 / rho0) with C_11 = E_new * (1 - nu) / ((1 + nu) * (1 - 2*nu))
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EPS_MIN_STRETCH = 1e-6
_EM10 = 1e-10
_EM20 = 1e-20
_EP20 = 1e20


@dataclass
class Law90Params:
    """Parameters for LAW90 tabulated foam constitutive model."""

    rho0: float = 0.0
    refer_rho: float = 0.0
    E0: float = 0.0
    nu: float = 0.0
    shape: float = 1.0
    hys: float = 1.0
    gamma: float = 1.0  # shape factor / exponent factor (MAT_ALPHA)
    alpha: float = 1.0  # alias for gamma
    tcut: float = _EP20  # tension cutoff stress (LSD_MAT83_TC)
    tflag: int = 1  # 1: follow input curve in tension, 2: follow E0
    fail: int = 0  # 0: stress stays at tcut, 1: element failure
    econt: float = 0.0  # optional Young modulus
    ismooth: int = 0  # 0: no filtering, 1: filter strain rate
    fcut: float = 0.0  # cutoff frequency for filtering
    nl: int = 0  # number of loading functions
    fct_ids: List[int] = field(default_factory=list)
    eps_dots: List[float] = field(default_factory=list)
    fscales: List[float] = field(default_factory=list)
    curves: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    E_MAX: float = 0.0

    def __post_init__(self) -> None:
        if self.gamma != 1.0 and self.alpha == 1.0:
            self.alpha = self.gamma
        elif self.alpha != 1.0 and self.gamma == 1.0:
            self.gamma = self.alpha
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho0
        if self.econt == 0.0:
            self.econt = self.E0
        if self.tcut <= 0.0:
            self.tcut = _EP20
        if self.tflag == 0:
            self.tflag = 1
        if self.shape == 0.0:
            self.shape = 1.0
        if self.alpha == 0.0:
            self.alpha = 1.0
            self.gamma = 1.0
        if self.hys == 0.0:
            self.hys = 1.0
        if self.E_MAX == 0.0:
            self.E_MAX = max(self.E0, 100.0 * self.E0)


def _get_param_val(p: Dict[str, Any], keys: List[str], default: Any) -> Any:
    for k in keys:
        if k in p and p[k] is not None:
            return p[k]
    return default


def build_law90(rec: Any) -> Material:
    """Constructor for /MAT/LAW90 (/MAT/HYST_FOAM) tabulated foam.

    Supports dict, GenericMaterialRecord, MaterialLaw90 entity, or Law90Params.
    """
    if isinstance(rec, Law90Params):
        params_dict = {
            "E": rec.E0,
            "E0": rec.E0,
            "nu": rec.nu,
            "shape": rec.shape,
            "hys": rec.hys,
            "gamma": rec.gamma,
            "alpha": rec.alpha,
            "tcut": rec.tcut,
            "tflag": rec.tflag,
            "fail": rec.fail,
            "econt": rec.econt,
            "ismooth": rec.ismooth,
            "fcut": rec.fcut,
            "nl": rec.nl,
            "fct_ids": rec.fct_ids,
            "eps_dots": rec.eps_dots,
            "fscales": rec.fscales,
            "curves": rec.curves,
            "E_MAX": rec.E_MAX,
        }
        mat = Material(
            id=1,
            law=90,
            rho0=rec.rho0,
            title="LAW90",
            params=params_dict,
        )
        mat.law90_params = rec
        return mat

    if hasattr(rec, "params") and isinstance(rec.params, dict):
        p = rec.params
        rec_id = getattr(rec, "id", 1)
        density = getattr(rec, "density", getattr(rec, "rho0", 0.0))
        title = getattr(rec, "title", "")
    elif isinstance(rec, dict):
        p = rec
        rec_id = p.get("id", 1)
        density = p.get("rho", p.get("rho0", p.get("MAT_RHO", 0.0)))
        title = p.get("title", "")
    else:
        rec_id = getattr(rec, "id", 1)
        density = getattr(rec, "rho0", getattr(rec, "density", 0.0))
        title = getattr(rec, "title", "")
        p = {k: getattr(rec, k) for k in dir(rec) if not k.startswith("_")}

    e0 = float(_get_param_val(p, ["MAT_E0", "E0", "e0", "E", "e"], 1.0))
    nu = float(_get_param_val(p, ["MAT_NU", "nu", "NU"], 0.0))
    shape = float(_get_param_val(p, ["MAT_SHAPE", "shape", "Shape"], 1.0)) or 1.0
    hys = float(_get_param_val(p, ["Hys", "hys", "HYS", "MAT_HYST"], 1.0)) or 1.0
    alpha = float(_get_param_val(p, ["MAT_ALPHA", "alpha", "Alpha", "gamma", "MAT_GAMMA"], 1.0)) or 1.0
    tcut = float(_get_param_val(p, ["LSD_MAT83_TC", "tcut", "TCUT", "tcut0", "TCUT0"], _EP20))
    if tcut <= 0.0:
        tcut = _EP20
    tflag = int(_get_param_val(p, ["MAT_TFLAG", "tflag", "TFLAG"], 1)) or 1
    fail = int(_get_param_val(p, ["LSD_MAT83_FAIL", "fail", "FAIL"], 0))
    econt = float(_get_param_val(p, ["LSD_MAT83_ED", "econt", "ECONT"], e0)) or e0
    ismooth = int(_get_param_val(p, ["Ismooth", "ismooth", "ISMOOTH"], 0))
    fcut = float(_get_param_val(p, ["Fcut", "fcut", "FCUT"], 0.0))
    nl = int(_get_param_val(p, ["NL", "nl"], 0))

    fct_ids = list(_get_param_val(p, ["fct_IDL", "fct_ids", "FCT_IDL", "load_fids"], []))
    eps_dots = list(_get_param_val(p, ["EpsilondotL", "eps_dots", "EPSILONDOTL", "load_rates"], []))
    fscales = list(_get_param_val(p, ["FscaleL", "fscales", "FSCALEL", "load_scales"], []))
    curves = list(_get_param_val(p, ["curves"], []))

    if nl == 0 and fct_ids:
        nl = len(fct_ids)
    if nl > 0 and len(fscales) < nl:
        fscales = fscales + [1.0] * (nl - len(fscales))
    if nl > 0 and len(eps_dots) < nl:
        eps_dots = eps_dots + [0.0] * (nl - len(eps_dots))

    params = {
        "E": e0,
        "E0": e0,
        "nu": nu,
        "shape": shape,
        "hys": hys,
        "gamma": alpha,
        "alpha": alpha,
        "tcut": tcut,
        "tflag": tflag,
        "fail": fail,
        "econt": econt,
        "ismooth": ismooth,
        "fcut": fcut,
        "nl": nl,
        "fct_ids": fct_ids,
        "eps_dots": eps_dots,
        "fscales": fscales,
        "curves": curves,
        "E_MAX": float(_get_param_val(p, ["E_MAX", "emax"], max(e0, 100.0 * e0))),
    }

    mat = Material(id=rec_id, law=90, rho0=float(density), title=title, params=params)
    mat.law90_params = Law90Params(
        rho0=float(density),
        E0=e0,
        nu=nu,
        shape=shape,
        hys=hys,
        gamma=alpha,
        alpha=alpha,
        tcut=tcut,
        tflag=tflag,
        fail=fail,
        econt=econt,
        ismooth=ismooth,
        fcut=fcut,
        nl=nl,
        fct_ids=fct_ids,
        eps_dots=eps_dots,
        fscales=fscales,
        curves=curves,
        E_MAX=params["E_MAX"],
    )
    return mat


def resolve(mat: Material, model: Any, log: Any = None) -> None:
    """Resolve /FUNCT references and compute E_ini / E_max following law90_upd.F."""
    p = mat.params
    fids = p.get("fct_ids", [])
    rates = p.get("eps_dots", [])
    scales = p.get("fscales", [])
    nl = len(fids)

    curves: List[Tuple[np.ndarray, np.ndarray]] = []
    emax_curve = 0.0
    eini_curve = 0.0

    for j, fid in enumerate(fids):
        fct = model.functions.get(fid) if hasattr(model, "functions") else None
        if fct is None:
            if log is not None:
                log.error(f"/MAT/LAW90/{mat.id}: function {fid} not defined", "MAT CHECK")
            continue
        sc = scales[j] if j < len(scales) and scales[j] != 0.0 else 1.0
        x_data = np.asarray(fct.x, dtype=float).copy()
        y_data = np.asarray(fct.y, dtype=float).copy() * sc
        curves.append((x_data, y_data))

        # Slopes calculation (func_slope.F / law90_upd.F)
        if len(x_data) >= 2:
            dx = np.diff(x_data)
            dy = np.diff(y_data)
            nonzero_dx = dx != 0.0
            if np.any(nonzero_dx):
                slopes = dy[nonzero_dx] / dx[nonzero_dx]
                stiffmax = float(np.max(slopes))
                stiffini = float(slopes[0])
                emax_curve = max(emax_curve, stiffmax)
                eini_curve = max(eini_curve, stiffini)

    p["curves"] = curves
    if hasattr(mat, "law90_params") and mat.law90_params is not None:
        mat.law90_params.curves = curves

    e0 = p.get("E0", p.get("E", 1.0))
    if eini_curve > e0:
        e0 = eini_curve
        p["E0"] = e0
        p["E"] = e0
        if log is not None:
            log.warning(
                f"/MAT/LAW90/{mat.id}: initial Young modulus raised to initial curve slope {e0:g} (upstream msg 865)",
                "MAT CHECK",
            )

    if emax_curve <= e0:
        emax = e0
    else:
        emax = min(emax_curve, 100.0 * e0)

    p["E_MAX"] = emax
    if hasattr(mat, "law90_params") and mat.law90_params is not None:
        mat.law90_params.E0 = e0
        mat.law90_params.E = e0
        mat.law90_params.E_MAX = emax


def _eval_curve(x_data: np.ndarray, y_data: np.ndarray, strain: float) -> float:
    """Evaluate 1D piecewise linear tabulated curve with slope extrapolation."""
    n = len(x_data)
    if n == 0:
        return 0.0
    if n == 1:
        return float(y_data[0])

    if strain <= x_data[0]:
        s0 = (y_data[1] - y_data[0]) / (x_data[1] - x_data[0]) if (x_data[1] - x_data[0]) != 0.0 else 0.0
        return float(y_data[0] + s0 * (strain - x_data[0]))
    elif strain >= x_data[-1]:
        s1 = (y_data[-1] - y_data[-2]) / (x_data[-1] - x_data[-2]) if (x_data[-1] - x_data[-2]) != 0.0 else 0.0
        return float(y_data[-1] + s1 * (strain - x_data[-1]))
    else:
        return float(np.interp(strain, x_data, y_data))


def _eval_strain_rate_curves(
    curves: List[Tuple[np.ndarray, np.ndarray]],
    rates: List[float],
    strain: float,
    rate: float,
    tflag: int,
    e0: float,
) -> float:
    """Evaluate tabulated stress given compressive engineering strain and strain rate."""
    if tflag == 2 and strain < 0.0:
        return e0 * strain

    ncurves = len(curves)
    if ncurves == 0:
        return e0 * strain

    if ncurves == 1:
        return _eval_curve(curves[0][0], curves[0][1], strain)

    # Multiple strain rate curves: rate interpolation (sigeps90.F lines 447-497)
    # Curves are assumed indexed in ascending order of rate
    if rate <= rates[0]:
        return _eval_curve(curves[0][0], curves[0][1], strain)
    elif rate >= rates[-1]:
        # Extrapolate or clamp between last two curves
        s1 = _eval_curve(curves[-2][0], curves[-2][1], strain)
        s2 = _eval_curve(curves[-1][0], curves[-1][1], strain)
        dr = rates[-1] - rates[-2]
        fac = (rate - rates[-2]) / dr if dr > 0.0 else 1.0
        return s1 + fac * (s2 - s1)
    else:
        # Bracket search
        j = 0
        for idx in range(len(rates) - 1):
            if rate >= rates[idx]:
                j = idx
        r1, r2 = rates[j], rates[j + 1]
        s1 = _eval_curve(curves[j][0], curves[j][1], strain)
        s2 = _eval_curve(curves[j + 1][0], curves[j + 1][1], strain)
        fac = (rate - r1) / (r2 - r1) if (r2 - r1) > 0.0 else 0.0
        return s1 + fac * (s2 - s1)


def extra_shapes(mat: Any, nip: Optional[int] = None) -> Dict[str, Any]:
    """State variable buffer allocation for LAW90 solid elements."""
    return {
        "uv90": (10,),  # UVAR1..10 state array
        "eps90": (6,),  # Total strain tensor
        "epsd90": (),  # Filtered scalar strain rate
    }


def sound_speed(mat: Any, rho: Optional[float] = None, extra: Optional[Dict[str, Any]] = None) -> float:
    """Calculate acoustic sound speed c = sqrt(C11 / rho0) using current evolving modulus."""
    p = mat.params if hasattr(mat, "params") else {}
    e0 = float(p.get("E0", p.get("E", 1.0)))
    nu = float(p.get("nu", 0.0))
    rho0 = float(getattr(mat, "rho0", rho if rho is not None else 1.0))
    if rho0 <= 0.0:
        rho0 = 1.0

    e_cur = e0
    if extra is not None and "uv90" in extra:
        uv = extra["uv90"]
        if isinstance(uv, np.ndarray) and uv.size >= 8:
            if uv.ndim == 1 and uv[7] > 0.0:
                e_cur = float(uv[7])
            elif uv.ndim == 2 and uv.shape[0] > 0 and uv[0, 7] > 0.0:
                e_cur = float(np.mean(uv[:, 7]))

    denom = (1.0 + nu) * (1.0 - 2.0 * nu)
    c11 = e_cur * (1.0 - nu) / denom if abs(denom) > 1e-12 else e_cur
    return math.sqrt(max(0.0, c11 / rho0))


def solid_step(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Constitutive integration for LAW90 tabulated foam 3D continuum solids.

    Parameters
    ----------
    mat : Material or Law90Params
        Material object containing constitutive parameters.
    sig : np.ndarray
        Old stress tensor (n, 6) or (6,), Voigt order [xx, yy, zz, xy, yz, zx].
    deps : np.ndarray
        Strain increment tensor (n, 6) or (6,), engineering shear [xx, yy, zz, xy, yz, zx].
    epsp : np.ndarray, optional
        Plastic strain array (n,) or scalar.
    dt : float
        Time step increment.
    extra : dict, optional
        Element persistent history state views (uv90, eps90, epsd90, etc.).

    Returns
    -------
    sig_new : np.ndarray
        New Cauchy stress tensor (n, 6) or (6,).
    epsp : np.ndarray or None
        Updated plastic / equivalent strain.
    sound_speed : np.ndarray
        Acoustic sound speed per element.
    """
    p = mat.params if hasattr(mat, "params") else {}
    e0 = float(p.get("E0", p.get("E", 1.0)))
    nu = float(p.get("nu", 0.0))
    shape = float(p.get("shape", 1.0))
    hys = float(p.get("hys", 1.0))
    alpha = float(p.get("alpha", p.get("gamma", 1.0)))
    tcut0 = float(p.get("tcut", _EP20))
    tflag = int(p.get("tflag", 1))
    fail = int(p.get("fail", 0))
    ismooth = int(p.get("ismooth", 0))
    fcut = float(p.get("fcut", 0.0))
    emax = float(p.get("E_MAX", max(e0, 100.0 * e0)))
    rho0 = float(getattr(mat, "rho0", 1.0))
    if rho0 <= 0.0:
        rho0 = 1.0

    curves: List[Tuple[np.ndarray, np.ndarray]] = p.get("curves", [])
    rates: List[float] = [float(r) for r in p.get("eps_dots", [])]
    nfunc = len(curves)

    is_1d = sig.ndim == 1
    sig_arr = np.atleast_2d(sig).astype(float).copy()
    deps_arr = np.atleast_2d(deps).astype(float).copy()
    nel = sig_arr.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.atleast_1d(epsp).astype(float).copy()

    if extra is None:
        extra = {}

    # Total strain array
    eps_tot = extra.get("eps90")
    if eps_tot is None:
        eps_tot = np.zeros((nel, 6), dtype=float)
        extra["eps90"] = eps_tot
    else:
        eps_tot = np.atleast_2d(eps_tot)
        if eps_tot.shape != (nel, 6):
            eps_tot = np.zeros((nel, 6), dtype=float)
            extra["eps90"] = eps_tot

    # Update total strain in-place (mulaw.F90: eps = eps + deps)
    eps_tot += deps_arr

    # UVAR state array: 10 variables per element
    # UVAR1: YLD_prev
    # UVAR2: W_max
    # UVAR3: eps_dot_filtered
    # UVAR4: W_cum
    # UVAR5: scratch / unused
    # UVAR6: EPST_prev
    # UVAR7: DAM (hysteretic factor)
    # UVAR8: E_cur (evolving modulus)
    # UVAR9: W_stat_prev
    # UVAR10: EPSS_prev
    uv = extra.get("uv90")
    if uv is None:
        uv = np.zeros((nel, 10), dtype=float)
        uv[:, 6] = 1.0  # UVAR7: initial DAM = 1.0
        uv[:, 7] = e0  # UVAR8: initial E_cur = E0
        extra["uv90"] = uv
    else:
        uv = np.atleast_2d(uv)
        if uv.shape[0] != nel or uv.shape[1] < 10:
            new_uv = np.zeros((nel, 10), dtype=float)
            new_uv[:, 6] = 1.0
            new_uv[:, 7] = e0
            uv = new_uv
            extra["uv90"] = uv

    sound_speeds = np.zeros(nel, dtype=float)
    sig_new = np.zeros((nel, 6), dtype=float)

    # Strain rate smoothing factor
    asrate = min(1.0, 2.0 * math.pi * fcut * dt) if (ismooth > 0 and fcut > 0.0 and dt > 0.0) else 1.0

    # Element loop
    for i in range(nel):
        # 1. Symmetric strain tensor
        exx = eps_tot[i, 0]
        eyy = eps_tot[i, 1]
        ezz = eps_tot[i, 2]
        exy = 0.5 * eps_tot[i, 3]
        eyz = 0.5 * eps_tot[i, 4]
        ezx = 0.5 * eps_tot[i, 5]

        A = np.array([
            [exx, exy, ezx],
            [exy, eyy, eyz],
            [ezx, eyz, ezz],
        ], dtype=float)

        # Spectral decomposition (sigeps33.F / valpvec_v)
        evv, V = np.linalg.eigh(A)  # evv: eigenvalues, V[:, k]: eigenvector k

        # Principal stretches (logarithmic strain measure: lambda = exp(eps))
        ev = np.exp(evv)
        ev = np.maximum(ev, _EPS_MIN_STRETCH)

        # Compressive engineering strains e = 1 - lambda
        strain = 1.0 - ev
        epst = float(np.sqrt(np.sum(strain**2)))

        # 2. Strain rate tensor and principal transformation
        if dt > 0.0:
            ep_xx = deps_arr[i, 0] / dt
            ep_yy = deps_arr[i, 1] / dt
            ep_zz = deps_arr[i, 2] / dt
            ep_xy = 0.5 * deps_arr[i, 3] / dt
            ep_yz = 0.5 * deps_arr[i, 4] / dt
            ep_zx = 0.5 * deps_arr[i, 5] / dt

            L = np.array([
                [ep_xx, ep_xy, ep_zx],
                [ep_xy, ep_yy, ep_yz],
                [ep_zx, ep_yz, ep_zz],
            ], dtype=float)

            # Principal strain rates eps_dot_k = (V^T * L * V)_kk
            epsp_princ = np.diag(V.T @ L @ V)
            # Engineering strain rates e_dot_k = eps_dot_k * lambda_k
            strainrate_princ = epsp_princ * ev
            rateeps = float(np.sqrt(np.sum(strainrate_princ**2)))
        else:
            epsp_princ = np.zeros(3, dtype=float)
            rateeps = 0.0

        if ismooth > 0 and dt > 0.0:
            rateeps = asrate * rateeps + (1.0 - asrate) * uv[i, 2]
        uv[i, 2] = rateeps

        # 3. Quasi-static energy computation (sigeps90.F lines 324-348)
        sqstat = np.zeros(3, dtype=float)
        for k in range(3):
            sqstat[k] = _eval_curve(curves[0][0], curves[0][1], strain[k]) if nfunc > 0 else e0 * strain[k]
            if tflag == 2 and strain[k] < 0.0:
                sqstat[k] = e0 * strain[k]

        quasi_eint = float(0.5 * np.sum(strain * sqstat))
        deint = quasi_eint - uv[i, 8]
        uv[i, 8] = quasi_eint

        deps_norm = epst - uv[i, 5]

        # Loading vs. unloading detection (sigeps90.F lines 384-396)
        is_loading = (deint >= 0.0) or (deps_norm >= 0.0)
        if not is_loading:
            rateeps = min(rateeps, uv[i, 2])
            uv[i, 2] = rateeps

        # 4. Stress interpolation
        S = np.zeros(3, dtype=float)
        if nfunc == 0:
            S = e0 * strain
        elif nfunc == 1:
            S = sqstat.copy()
        else:
            if is_loading:
                for k in range(3):
                    S[k] = _eval_strain_rate_curves(curves, rates, strain[k], rateeps, tflag, e0)
            else:
                # Unloading is quasi-static
                S = sqstat.copy()

        # 5. Hysteretic energy damage factor (sigeps90.F lines 660-706)
        yld = float(np.sqrt(np.sum(S**2)))
        delta_epst = epst - uv[i, 5]
        w_cum = uv[i, 3] + 0.5 * (yld + uv[i, 0]) * delta_epst
        w_cum = max(0.0, w_cum)
        w_max = max(uv[i, 1], w_cum)

        uv[i, 3] = w_cum
        uv[i, 1] = w_max
        uv[i, 0] = yld
        uv[i, 5] = epst

        dam = 1.0
        if (not is_loading) and (w_max > 0.0) and (hys < 1.0 or abs(hys - 1.0) > 1e-12):
            frac = min(1.0, max(0.0, w_cum / w_max))
            dam_base = max(0.0, 1.0 - (frac**shape))
            dam_pow = dam_base**alpha
            dam = 1.0 - (1.0 - hys) * dam_pow
            dam = max(0.0, min(1.0, dam))
            S *= dam

        uv[i, 6] = dam

        # 6. Tensile stress cutoff (sigeps90.F lines 710-781)
        # S > 0: compression; S < 0: traction. Tensile stress is T = -S
        T = -S
        tcut = dam * tcut0 if (not is_loading) else tcut0

        if fail > 0:
            if np.any(T >= tcut0):
                T[:] = 0.0
        else:
            T = np.minimum(tcut, T)

        # Convert nominal stress to true Cauchy stress: sigma_k = T_k / (lambda_j * lambda_l)
        sigma_princ = np.zeros(3, dtype=float)
        sigma_princ[0] = T[0] / (ev[1] * ev[2])
        sigma_princ[1] = T[1] / (ev[0] * ev[2])
        sigma_princ[2] = T[2] / (ev[0] * ev[1])

        # Rotate Cauchy stress back to global frame: sigma = V * diag(sigma_princ) * V^T
        sigma_mat = V @ np.diag(sigma_princ) @ V.T
        sig_new[i, 0] = sigma_mat[0, 0]
        sig_new[i, 1] = sigma_mat[1, 1]
        sig_new[i, 2] = sigma_mat[2, 2]
        sig_new[i, 3] = sigma_mat[0, 1]
        sig_new[i, 4] = sigma_mat[1, 2]
        sig_new[i, 5] = sigma_mat[2, 0]

        # 7. Evolving modulus & Sound speed (sigeps90.F lines 750-762)
        e_old = uv[i, 7]
        if e_old <= 0.0:
            e_old = e0
        epss = max(0.0, min(1.0, epst - yld / max(e_old, _EM20)))
        e_new = min(emax, (emax - e0) * epss + e0)
        uv[i, 9] = epss
        uv[i, 7] = e_new

        denom = (1.0 + nu) * (1.0 - 2.0 * nu)
        c11 = e_new * (1.0 - nu) / denom if abs(denom) > 1e-12 else e_new
        sound_speeds[i] = math.sqrt(max(0.0, c11 / rho0))
        epsp_arr[i] = epst

    if is_1d:
        return sig_new[0], epsp_arr[0], sound_speeds[0]
    return sig_new, epsp_arr, sound_speeds


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Kernel adapter conforming to pyradioss constitutive dispatch."""
    return solid_step(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, **kwargs)


def shell_update(mat: Any, sig: np.ndarray, deps: np.ndarray, *args: Any, **kwargs: Any) -> Any:
    """LAW90 is strictly formulated for 3D solid continuum elements (hm_read_mat90.F)."""
    raise NotImplementedError("LAW90 (tabulated hysteretic foam) is formulated for 3D solid elements only.")


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent tangent stiffness tensor (6, 6) for LAW90 solid elements."""
    p = mat.params if hasattr(mat, "params") else {}
    e0 = float(p.get("E0", p.get("E", 1.0)))
    nu = float(p.get("nu", 0.0))

    e_cur = e0
    if extra is not None and "uv90" in extra:
        uv = extra["uv90"]
        if isinstance(uv, np.ndarray) and uv.size >= 8:
            if uv.ndim == 1 and uv[7] > 0.0:
                e_cur = float(uv[7])
            elif uv.ndim == 2 and uv.shape[0] > 0 and uv[0, 7] > 0.0:
                e_cur = float(np.mean(uv[:, 7]))

    # Analytical P-wave / shear elastic tangent
    denom = (1.0 + nu) * (1.0 - 2.0 * nu)
    if abs(denom) > 1e-12:
        c11 = e_cur * (1.0 - nu) / denom
        c12 = e_cur * nu / denom
        g = 0.5 * e_cur / (1.0 + nu)
    else:
        c11 = e_cur
        c12 = 0.0
        g = 0.5 * e_cur

    C = np.zeros((6, 6), dtype=float)
    C[0, 0] = c11
    C[1, 1] = c11
    C[2, 2] = c11
    C[0, 1] = c12
    C[1, 0] = c12
    C[0, 2] = c12
    C[2, 0] = c12
    C[1, 2] = c12
    C[2, 1] = c12
    C[3, 3] = g
    C[4, 4] = g
    C[5, 5] = g
    return C


def consistent_solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent solid tangent alias."""
    return solid_tangent(mat, sig=sig, deps=deps, dt=dt, extra=extra, **kwargs)


def _register() -> None:
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY

    MAT_PHYSICS_REGISTRY["LAW90"] = build_law90
    MAT_PHYSICS_REGISTRY["HYST_FOAM"] = build_law90
    MAT_PHYSICS_REGISTRY["TAB_FOAM"] = build_law90
    MAT_PHYSICS_REGISTRY["TABULAR_FOAM"] = build_law90


_register()

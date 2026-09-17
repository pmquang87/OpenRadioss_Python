"""OpenRadioss /MAT/LAW120 (/MAT/TAPO) — Pont-Pack Orthotropic Tape & Woven Fabric.

Fortran source references:
- Starter input reader:
  `starter/source/materials/mat/mat120/hm_read_mat120.F`
- 3D continuum and connection kernels:
  `engine/source/materials/mat/mat120/sigeps120.F`
  `engine/source/materials/mat/mat120/sigeps120_vm.F`
  `engine/source/materials/mat/mat120/sigeps120_dp.F`
  `engine/source/materials/mat/mat120/sigeps120_tab_vm.F`
  `engine/source/materials/mat/mat120/sigeps120_tab_dp.F`
- Card layout configuration:
  `hm_cfg_files/config/CFG/radioss2022/MAT/mat120_tapo.cfg`

Constitutive Formulation:
1. Orthotropic Pont-Pack tape / woven fabric model:
   - Longitudinal yarn direction (1 / warp) and transverse yarn direction (2 / weft).
   - Material orthotropy angle rotation into yarn frame.
2. Directional non-linear tensile response:
   - Yarn tension along longitudinal axis (1):
     sigma_11 = f_1(eps_11) in tension, with reduced compressive stiffness r_comp * E_1
     to capture yarn buckling / low compressive stiffness.
   - Yarn tension along transverse axis (2):
     sigma_22 = f_2(eps_22) in tension, with reduced compressive stiffness r_comp * E_2.
   - Tabulated non-linear stress-strain curves via /FUNCT or analytical hardening.
3. Scissor shear behavior with Trellis locking angle:
   - Engineering shear strain gamma_12 = 2 * eps_12.
   - Initial compliant shear modulus G_0 for |gamma_12| <= gamma_lock (scissor kinematics).
   - Trellis locking stiffening modulus G_lock for |gamma_12| > gamma_lock with C0 continuity:
     tau_12 = G_lock * gamma_12 + sign(gamma_12) * gamma_lock * (G_0 - G_lock).
4. Transverse shear and membrane uncoupling:
   - Out-of-plane transverse shears (sigma_23, sigma_31) are integrated elastically with
     transverse shear modulus G_trans, completely uncoupled from in-plane membrane response.
5. Exact acoustic sound speed:
   c_shell = sqrt(max(E_1, E_2, G_lock) / rho0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_DEFAULT_GAMMA_LOCK = 0.7  # ~40 degrees Trellis scissor locking angle


def _eval_curve_1d(
    curve: Any,
    x_in: Union[float, np.ndarray],
    xscale: float = 1.0,
    yscale: float = 1.0,
    default_val: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear 1D curve evaluation with slope derivation."""
    x_arr = np.atleast_1d(np.asarray(x_in, dtype=np.float64))
    x_scaled = x_arr / (xscale if abs(xscale) > _EM20 else 1.0)

    if curve is None:
        y = np.full_like(x_arr, default_val * yscale)
        s = np.zeros_like(x_arr)
        return y, s

    if isinstance(curve, (int, float, np.number)):
        y = np.full_like(x_arr, float(curve) * yscale)
        s = np.zeros_like(x_arr)
        return y, s

    if callable(curve):
        try:
            res = curve(x_scaled)
            if isinstance(res, tuple) and len(res) == 2:
                y = np.asarray(res[0], dtype=np.float64) * yscale
                s = np.asarray(res[1], dtype=np.float64) * (yscale / xscale)
                return y, s
            y = np.asarray(res, dtype=np.float64) * yscale
            dx = 1.0e-6
            y_p = np.asarray(curve(x_scaled + dx), dtype=np.float64) * yscale
            s = (y_p - y) / (dx * xscale)
            return y, s
        except Exception:
            pass

    cx, cy = None, None
    if hasattr(curve, "x") and hasattr(curve, "y"):
        cx = np.asarray(curve.x, dtype=np.float64)
        cy = np.asarray(curve.y, dtype=np.float64)
    elif hasattr(curve, "data"):
        data = np.asarray(curve.data, dtype=np.float64)
        if data.ndim == 2 and data.shape[1] >= 2:
            cx = data[:, 0]
            cy = data[:, 1]
    elif isinstance(curve, (list, tuple)):
        if len(curve) == 2 and isinstance(curve[0], (list, tuple, np.ndarray)) and isinstance(curve[1], (list, tuple, np.ndarray)):
            cx = np.asarray(curve[0], dtype=np.float64)
            cy = np.asarray(curve[1], dtype=np.float64)
        else:
            arr = np.asarray(curve, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                cx = arr[:, 0]
                cy = arr[:, 1]
    elif isinstance(curve, dict) and "x" in curve and "y" in curve:
        cx = np.asarray(curve["x"], dtype=np.float64)
        cy = np.asarray(curve["y"], dtype=np.float64)

    if cx is not None and cy is not None and len(cx) > 0:
        if len(cx) == 1:
            y = np.full_like(x_arr, cy[0] * yscale)
            s = np.zeros_like(x_arr)
            return y, s
        if np.any(np.diff(cx) < 0):
            order = np.argsort(cx)
            cx = cx[order]
            cy = cy[order]

        y_interp = np.interp(x_scaled, cx, cy) * yscale
        dx = np.diff(cx)
        dy = np.diff(cy)
        slopes_raw = np.where(dx > _EM20, dy / dx, 0.0)
        idx = np.searchsorted(cx, x_scaled, side="right") - 1
        idx = np.clip(idx, 0, len(slopes_raw) - 1)
        s_interp = slopes_raw[idx] * (yscale / xscale)
        return y_interp, s_interp

    y = np.full_like(x_arr, default_val * yscale)
    s = np.zeros_like(x_arr)
    return y, s


@dataclass
class Law120Params:
    """Parameters for OpenRadioss /MAT/LAW120 (/MAT/TAPO) Pont-Pack tape & fabric model."""
    id: int = 1
    title: str = ""
    law: int = 120
    law_name: str = "LAW120"
    rho0: float = 0.0
    refer_rho: float = 0.0
    rho: float = 0.0
    young: float = 1.0
    nu: float = 0.3

    # Orthotropic elastic constants
    e1: float = 0.0
    e2: float = 0.0
    e3: float = 0.0
    nu12: float = 0.0
    nu21: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g31: float = 0.0

    # Scissor shear & Trellis locking
    gamma_lock: float = _DEFAULT_GAMMA_LOCK
    g_lock: float = 0.0
    r_comp: float = 0.01  # Reduced compression stiffness ratio (yarn buckling)

    # TAPO card parameters (hm_read_mat120.F)
    iform: int = 1
    itrx: int = 2
    idam: int = 2
    thick: float = 0.0
    tab_id: int = 0
    xscale: float = 1.0
    yscale: float = 1.0
    tau0: float = 0.0
    q: float = 0.0
    beta: float = 0.0
    h: float = 0.0
    af1: float = 0.0
    af2: float = 0.0
    ah1: float = 0.0
    ah2: float = 0.0
    as_: float = 0.0
    cc: float = 0.0
    gam0: float = 0.0
    gamf: float = 0.0
    d1c: float = 0.0
    d2c: float = 0.0
    d1f: float = 0.0
    d2f: float = 0.0
    dtrx: float = 0.0
    djc: float = 0.0
    exp_n: float = 1.0

    # Curve / Table references
    curve_1: Any = None       # Directional warp / longitudinal curve
    curve_2: Any = None       # Directional weft / transverse curve
    curve_shear: Any = None   # Scissor shear curve (tab_id)
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rho > 0.0 and self.rho0 <= 0.0:
            self.rho0 = self.rho
        if self.rho0 > 0.0 and self.rho <= 0.0:
            self.rho = self.rho0
        if self.refer_rho > 0.0 and self.rho0 <= 0.0:
            self.rho0 = self.refer_rho
        if self.young <= 0.0:
            self.young = 1.0

        # Set directional moduli defaults
        if self.e1 <= 0.0:
            self.e1 = self.young
        if self.e2 <= 0.0:
            if self.af1 > 0.0 and self.af2 > 0.0:
                self.e2 = self.e1 * (self.af2 / self.af1)
            else:
                self.e2 = self.e1
        if self.e3 <= 0.0:
            self.e3 = min(self.e1, self.e2)

        if self.nu12 <= 0.0:
            self.nu12 = self.nu if (0.0 <= self.nu < 0.5) else 0.3
        self.nu21 = self.nu12 * self.e2 / (self.e1 if self.e1 > _EM20 else 1.0)

        # Initial scissor shear modulus G0
        if self.g12 <= 0.0:
            if self.tau0 > 0.0 and self.gam0 > 0.0:
                self.g12 = self.tau0 / self.gam0
            else:
                self.g12 = self.e1 / (2.0 * (1.0 + self.nu12))

        # Trellis locking modulus
        if self.g_lock <= 0.0:
            if self.h > self.g12:
                self.g_lock = self.h
            else:
                self.g_lock = 10.0 * self.g12

        # Trellis locking angle
        if self.gamma_lock <= 0.0 or self.gamma_lock == _DEFAULT_GAMMA_LOCK:
            if self.gam0 > 0.0:
                self.gamma_lock = self.gam0
            else:
                self.gamma_lock = _DEFAULT_GAMMA_LOCK

        if self.g23 <= 0.0:
            self.g23 = self.g12
        if self.g31 <= 0.0:
            self.g31 = self.g12

        if self.r_comp <= 0.0:
            self.r_comp = 0.01

    @property
    def e(self) -> float:
        return self.young

    @property
    def E(self) -> float:
        return self.young

    @property
    def Nu(self) -> float:
        return self.nu

    @classmethod
    def from_material(cls, mat: Any) -> Law120Params:
        """Construct Law120Params from generic Material or MatLaw120 entity."""
        if isinstance(mat, Law120Params):
            return mat

        def _get(keys: Sequence[str], default: Any) -> Any:
            for k in keys:
                if hasattr(mat, k):
                    val = getattr(mat, k)
                    if val is not None:
                        return val
                if hasattr(mat, "params") and isinstance(mat.params, dict) and k in mat.params:
                    val = mat.params[k]
                    if val is not None:
                        return val
                if isinstance(mat, dict) and k in mat:
                    val = mat[k]
                    if val is not None:
                        return val
            return default

        mid = int(_get(["id", "mid", "mat_id"], 1))
        title = str(_get(["title", "name"], ""))
        rho0 = float(_get(["rho0", "rho", "MAT_RHO"], 0.0))
        refer_rho = float(_get(["refer_rho", "Refer_Rho"], rho0))
        young = float(_get(["young", "e", "MAT_E", "E"], 1.0))
        nu = float(_get(["nu", "MAT_NU"], 0.3))

        e1 = float(_get(["e1", "E1", "E_1", "MAT_E1"], 0.0))
        e2 = float(_get(["e2", "E2", "E_2", "MAT_E2"], 0.0))
        e3 = float(_get(["e3", "E3", "E_3", "MAT_E3"], 0.0))
        nu12 = float(_get(["nu12", "NU12", "MAT_NU12"], 0.0))
        g12 = float(_get(["g12", "G12", "MAT_G12"], 0.0))
        g23 = float(_get(["g23", "G23", "MAT_G23"], 0.0))
        g31 = float(_get(["g31", "G31", "MAT_G31"], 0.0))
        g_lock = float(_get(["g_lock", "GLOCK", "G_LOCK", "MAT_GLOCK"], 0.0))
        gamma_lock = float(_get(["gamma_lock", "gam_lock", "GAM_LOCK", "gam0", "MAT_GAM0"], _DEFAULT_GAMMA_LOCK))
        r_comp = float(_get(["r_comp", "RCOMP", "rcomp"], 0.01))

        iform = int(_get(["iform", "MAT_IFORM"], 1))
        itrx = int(_get(["itrx", "MAT_ITRX"], 2))
        idam = int(_get(["idam", "MAT_IDAM"], 2))
        thick = float(_get(["thick", "MAT_THICK"], 0.0))
        tab_id = int(_get(["tab_id", "MAT_TAB_ID"], 0))
        xscale = float(_get(["xscale", "MAT_Xscale"], 1.0))
        yscale = float(_get(["yscale", "MAT_Yscale"], 1.0))
        tau0 = float(_get(["tau0", "tau", "MAT_TAU"], 0.0))
        q = float(_get(["q", "MAT_Q"], 0.0))
        beta = float(_get(["beta", "MAT_B"], 0.0))
        h = float(_get(["h", "MAT_H"], 0.0))
        af1 = float(_get(["af1", "MAT_AF1"], 0.0))
        af2 = float(_get(["af2", "MAT_AF2"], 0.0))
        ah1 = float(_get(["ah1", "MAT_AH1"], 0.0))
        ah2 = float(_get(["ah2", "MAT_AH2"], 0.0))
        as_ = float(_get(["as_", "as", "MAT_AS"], 0.0))
        cc = float(_get(["cc", "MAT_CC"], 0.0))
        gam0 = float(_get(["gam0", "MAT_GAM0"], 0.0))
        gamf = float(_get(["gamf", "MAT_GAMF"], 0.0))
        d1c = float(_get(["d1c", "MAT_D1C"], 0.0))
        d2c = float(_get(["d2c", "MAT_D2C"], 0.0))
        d1f = float(_get(["d1f", "MAT_D1F"], 0.0))
        d2f = float(_get(["d2f", "MAT_D2F"], 0.0))
        dtrx = float(_get(["dtrx", "D_TRX"], 0.0))
        djc = float(_get(["djc", "D_JC"], 0.0))
        exp_n = float(_get(["exp_n", "MAT_EXP"], 1.0))

        curve_1 = _get(["curve_1", "fct_1", "f_1"], None)
        curve_2 = _get(["curve_2", "fct_2", "f_2"], None)
        curve_shear = _get(["curve_shear", "fct_shear", "f_shear", "tab_shear"], None)

        p = cls(
            id=mid,
            title=title,
            rho0=rho0,
            refer_rho=refer_rho,
            rho=rho0,
            young=young,
            nu=nu,
            e1=e1,
            e2=e2,
            e3=e3,
            nu12=nu12,
            g12=g12,
            g23=g23,
            g31=g31,
            g_lock=g_lock,
            gamma_lock=gamma_lock,
            r_comp=r_comp,
            iform=iform,
            itrx=itrx,
            idam=idam,
            thick=thick,
            tab_id=tab_id,
            xscale=xscale,
            yscale=yscale,
            tau0=tau0,
            q=q,
            beta=beta,
            h=h,
            af1=af1,
            af2=af2,
            ah1=ah1,
            ah2=ah2,
            as_=as_,
            cc=cc,
            gam0=gam0,
            gamf=gamf,
            d1c=d1c,
            d2c=d2c,
            d1f=d1f,
            d2f=d2f,
            dtrx=dtrx,
            djc=djc,
            exp_n=exp_n,
            curve_1=curve_1,
            curve_2=curve_2,
            curve_shear=curve_shear,
            params=dict(mat.params) if hasattr(mat, "params") and isinstance(mat.params, dict) else {},
        )
        return p


def eval_yarn_stress_and_tangent(
    eps: np.ndarray,
    e_base: float,
    curve: Any,
    r_comp: float = 0.01,
    q: float = 0.0,
    beta: float = 0.0,
    h: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate directional yarn non-linear stress sigma(eps) and tangent d_sigma/d_eps.

    Tension (eps > 0): non-linear loading curve or exponential hardening.
    Compression (eps <= 0): reduced compressive stiffness (yarn buckling).
    """
    eps_arr = np.atleast_1d(np.asarray(eps, dtype=np.float64))
    n = len(eps_arr)

    sig = np.zeros(n, dtype=np.float64)
    tang = np.zeros(n, dtype=np.float64)

    tensile = eps_arr > 0.0
    compressive = ~tensile

    # 1. Compressive regime (buckling / reduced stiffness)
    if np.any(compressive):
        sig[compressive] = (r_comp * e_base) * eps_arr[compressive]
        tang[compressive] = r_comp * e_base

    # 2. Tensile regime
    if np.any(tensile):
        idx_t = np.where(tensile)[0]
        eps_t = eps_arr[idx_t]

        if curve is not None:
            val, slope = _eval_curve_1d(curve, eps_t, default_val=e_base)
            sig[idx_t] = val
            tang[idx_t] = slope
        else:
            # Analytical nonlinear curve: sigma = E * eps + Q * (1 - exp(-beta * eps)) + H * eps
            val_linear = e_base * eps_t
            if q > 0.0 and beta > 0.0:
                exp_term = np.exp(-np.clip(beta * eps_t, 0.0, 50.0))
                val_nl = q * (1.0 - exp_term) + h * eps_t
                sig[idx_t] = val_linear + val_nl
                tang[idx_t] = e_base + q * beta * exp_term + h
            else:
                sig[idx_t] = val_linear + h * eps_t
                tang[idx_t] = e_base + h

    return sig, tang


def eval_trellis_shear_and_tangent(
    gamma: np.ndarray,
    g0: float,
    g_lock: float,
    gamma_lock: float,
    curve: Any = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate scissor shear stress tau_12 and tangent G_t as a function of shear strain gamma_12.

    Pre-locking (|gamma| <= gamma_lock): compliant scissor shear tau = G_0 * gamma.
    Post-locking (|gamma| > gamma_lock): Trellis locking stiffening tau = G_lock * gamma + sign(gamma) * gamma_lock * (G0 - G_lock).
    """
    gam_arr = np.atleast_1d(np.asarray(gamma, dtype=np.float64))
    n = len(gam_arr)

    if curve is not None:
        val, slope = _eval_curve_1d(curve, np.abs(gam_arr), default_val=g0)
        tau = np.sign(gam_arr) * val
        tang = slope
        return tau, tang

    tau = np.zeros(n, dtype=np.float64)
    tang = np.zeros(n, dtype=np.float64)

    abs_gam = np.abs(gam_arr)
    pre_lock = abs_gam <= gamma_lock
    post_lock = ~pre_lock

    if np.any(pre_lock):
        tau[pre_lock] = g0 * gam_arr[pre_lock]
        tang[pre_lock] = g0

    if np.any(post_lock):
        idx_p = np.where(post_lock)[0]
        sign_p = np.sign(gam_arr[idx_p])
        # C0 continuous transition at gamma_lock
        tau[idx_p] = g_lock * gam_arr[idx_p] + sign_p * gamma_lock * (g0 - g_lock)
        tang[idx_p] = g_lock

    return tau, tang


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pont-Pack orthotropic tape / fabric plane-stress shell update.

    Parameters:
      mat: Law120Params or Material entity
      sig: in-plane stress array (3,) or (n, 3) [xx, yy, xy] (or (5,) with transverse shear)
      deps: in-plane strain increment (3,) or (n, 3) [xx, yy, xy]
      epsp: equivalent plastic/damage strain
      dt: time step
      extra: state dict with 'eps120', 'angle', etc.

    Returns:
      (sig_new, epsp_new)
    """
    p = mat if isinstance(mat, Law120Params) else Law120Params.from_material(mat)

    deps_arr = np.asarray(deps, dtype=np.float64)
    sig_arr = np.asarray(sig, dtype=np.float64)
    is_1d = (deps_arr.ndim == 1)

    if is_1d:
        deps_2d = deps_arr.reshape(1, -1)
        sig_2d = sig_arr.reshape(1, -1)
    else:
        deps_2d = deps_arr
        sig_2d = sig_arr

    n = deps_2d.shape[0]
    has_transverse = (deps_2d.shape[1] >= 5 and sig_2d.shape[1] >= 5)

    if extra is None:
        extra = {}

    # Total accumulated strain in material frame
    eps_tot = extra.get("eps120", None)
    if eps_tot is None:
        eps_tot = np.zeros((n, 3), dtype=np.float64)
    else:
        eps_tot = np.asarray(eps_tot, dtype=np.float64)
        if eps_tot.ndim == 1:
            eps_tot = eps_tot.reshape(1, -1)
        if eps_tot.shape[0] != n:
            eps_tot = np.resize(eps_tot, (n, 3))

    # Add increment to total strain
    eps_tot[:, 0] += deps_2d[:, 0]
    eps_tot[:, 1] += deps_2d[:, 1]
    eps_tot[:, 2] += deps_2d[:, 2]  # Engineering shear gamma_12

    # Material axes rotation (if angle is defined)
    ang = extra.get("angle", p.params.get("angle", 0.0))
    if isinstance(ang, (int, float, np.number)) and abs(ang) > 1.0e-5:
        cos_a = math.cos(float(ang))
        sin_a = math.sin(float(ang))
        c2 = cos_a * cos_a
        s2 = sin_a * sin_a
        cs = cos_a * sin_a

        # Strain in material axes
        eps_11 = eps_tot[:, 0] * c2 + eps_tot[:, 1] * s2 + eps_tot[:, 2] * cs
        eps_22 = eps_tot[:, 0] * s2 + eps_tot[:, 1] * c2 - eps_tot[:, 2] * cs
        gam_12 = -2.0 * (eps_tot[:, 0] - eps_tot[:, 1]) * cs + eps_tot[:, 2] * (c2 - s2)
    else:
        eps_11 = eps_tot[:, 0]
        eps_22 = eps_tot[:, 1]
        gam_12 = eps_tot[:, 2]
        cos_a, sin_a, c2, s2, cs = 1.0, 0.0, 1.0, 0.0, 0.0

    # 1. Directional yarn responses
    sig_11, _ = eval_yarn_stress_and_tangent(
        eps_11, p.e1, p.curve_1, r_comp=p.r_comp, q=p.q, beta=p.beta, h=p.h
    )
    sig_22, _ = eval_yarn_stress_and_tangent(
        eps_22, p.e2, p.curve_2, r_comp=p.r_comp, q=p.q, beta=p.beta, h=p.h
    )

    # Poisson coupling
    if p.nu12 > 0.0:
        det_c = 1.0 - p.nu12 * p.nu21
        det_c = det_c if abs(det_c) > _EM20 else 1.0
        sig_11_c = (sig_11 + p.nu21 * sig_22) / det_c
        sig_22_c = (sig_22 + p.nu12 * sig_11) / det_c
        sig_11 = sig_11_c
        sig_22 = sig_22_c

    # 2. Scissor shear behavior with Trellis locking
    tau_12, _ = eval_trellis_shear_and_tangent(
        gam_12, p.g12, p.g_lock, p.gamma_lock, curve=p.curve_shear
    )

    # Rotate stresses back to element frame if needed
    if abs(ang) > 1.0e-5:
        sig_xx = sig_11 * c2 + sig_22 * s2 - 2.0 * tau_12 * cs
        sig_yy = sig_11 * s2 + sig_22 * c2 + 2.0 * tau_12 * cs
        sig_xy = (sig_11 - sig_22) * cs + tau_12 * (c2 - s2)
    else:
        sig_xx = sig_11
        sig_yy = sig_22
        sig_xy = tau_12

    # 3. Transverse shear uncoupling (sigma_23, sigma_31)
    if has_transverse:
        sig_yz = sig_2d[:, 3] + p.g23 * deps_2d[:, 3]
        sig_zx = sig_2d[:, 4] + p.g31 * deps_2d[:, 4]
        sign_out = np.column_stack([sig_xx, sig_yy, sig_xy, sig_yz, sig_zx])
    else:
        sign_out = np.column_stack([sig_xx, sig_yy, sig_xy])

    # Save state
    extra["eps120"] = eps_tot
    if epsp is None:
        epsp_out = np.zeros(n, dtype=np.float64)
    else:
        epsp_out = np.asarray(epsp, dtype=np.float64)

    if is_1d:
        return sign_out[0], (epsp_out[0] if epsp_out.ndim > 0 else float(epsp_out))
    return sign_out, epsp_out


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D continuum solid update for LAW120 Pont-Pack tape / woven fabric model.

    Parameters:
      mat: Law120Params or Material entity
      sig: stress array (6,) or (n, 6) [xx, yy, zz, xy, yz, zx]
      deps: strain increment (6,) or (n, 6)
      epsp: equivalent plastic/damage strain
      dt: time step
      extra: state dict with 'eps120'

    Returns:
      (sig_new, epsp_new, soundsp)
    """
    p = mat if isinstance(mat, Law120Params) else Law120Params.from_material(mat)

    deps_arr = np.asarray(deps, dtype=np.float64)
    sig_arr = np.asarray(sig, dtype=np.float64)
    is_1d = (deps_arr.ndim == 1)

    if is_1d:
        deps_2d = deps_arr.reshape(1, 6)
        sig_2d = sig_arr.reshape(1, 6)
    else:
        deps_2d = deps_arr
        sig_2d = sig_arr

    n = deps_2d.shape[0]

    if extra is None:
        extra = {}

    eps_tot = extra.get("eps120", None)
    if eps_tot is None:
        eps_tot = np.zeros((n, 6), dtype=np.float64)
    else:
        eps_tot = np.asarray(eps_tot, dtype=np.float64)
        if eps_tot.ndim == 1:
            eps_tot = eps_tot.reshape(1, -1)
        if eps_tot.shape[0] != n:
            eps_tot = np.resize(eps_tot, (n, 6))

    # Accumulate total strains
    eps_tot += deps_2d

    # 1. Directional normal yarn stresses
    sig_11, _ = eval_yarn_stress_and_tangent(
        eps_tot[:, 0], p.e1, p.curve_1, r_comp=p.r_comp, q=p.q, beta=p.beta, h=p.h
    )
    sig_22, _ = eval_yarn_stress_and_tangent(
        eps_tot[:, 1], p.e2, p.curve_2, r_comp=p.r_comp, q=p.q, beta=p.beta, h=p.h
    )
    sig_33 = p.e3 * eps_tot[:, 2]

    # Poisson coupling
    if p.nu12 > 0.0:
        det_c = 1.0 - p.nu12 * p.nu21
        det_c = det_c if abs(det_c) > _EM20 else 1.0
        sig_11 = (sig_11 + p.nu21 * sig_22) / det_c
        sig_22 = (sig_22 + p.nu12 * sig_11) / det_c

    # 2. Scissor shear behavior with Trellis locking
    tau_12, _ = eval_trellis_shear_and_tangent(
        eps_tot[:, 3], p.g12, p.g_lock, p.gamma_lock, curve=p.curve_shear
    )

    # 3. Transverse shears (uncoupled)
    tau_23 = p.g23 * eps_tot[:, 4]
    tau_31 = p.g31 * eps_tot[:, 5]

    sign_out = np.column_stack([sig_11, sig_22, sig_33, tau_12, tau_23, tau_31])

    # Acoustic wave speed
    rho_eff = p.rho0 if p.rho0 > 0.0 else 1.0
    c_max = max(p.e1, p.e2, p.e3, p.g_lock)
    soundsp = np.full(n, math.sqrt(c_max / rho_eff), dtype=np.float64)

    extra["eps120"] = eps_tot

    if epsp is None:
        epsp_out = np.zeros(n, dtype=np.float64)
    else:
        epsp_out = np.asarray(epsp, dtype=np.float64)

    if is_1d:
        return sign_out[0], (epsp_out[0] if epsp_out.ndim > 0 else float(epsp_out)), soundsp[0]
    return sign_out, epsp_out, soundsp


def sound_speed(
    mat: Any,
    rho: Optional[Union[float, np.ndarray]] = None,
    extra: Optional[Dict[str, Any]] = None,
    is_shell: bool = False,
) -> float:
    """Acoustic wave speed estimate for LAW120."""
    p = mat if isinstance(mat, Law120Params) else Law120Params.from_material(mat)
    r = rho if (rho is not None and float(np.mean(rho)) > 0.0) else (p.rho0 if p.rho0 > 0.0 else 1.0)
    c_max = max(p.e1, p.e2, p.g_lock)
    return float(math.sqrt(c_max / r))


def sound_speed_solid(mat: Any, rho: Optional[Union[float, np.ndarray]] = None, extra: Optional[Dict[str, Any]] = None) -> float:
    return sound_speed(mat, rho=rho, extra=extra, is_shell=False)


def sound_speed_shell(mat: Any, rho: Optional[Union[float, np.ndarray]] = None, extra: Optional[Dict[str, Any]] = None) -> float:
    return sound_speed(mat, rho=rho, extra=extra, is_shell=True)


def consistent_shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Algorithmic plane-stress tangent tensor (3, 3) for LAW120 shells."""
    p = mat if isinstance(mat, Law120Params) else Law120Params.from_material(mat)

    eps_tot = extra.get("eps120", np.zeros(3)) if extra else np.zeros(3)
    eps_tot = np.atleast_1d(np.asarray(eps_tot, dtype=np.float64)).flatten()
    if len(eps_tot) < 3:
        eps_tot = np.zeros(3, dtype=np.float64)

    _, tang_1 = eval_yarn_stress_and_tangent(eps_tot[0], p.e1, p.curve_1, r_comp=p.r_comp, q=p.q, beta=p.beta, h=p.h)
    _, tang_2 = eval_yarn_stress_and_tangent(eps_tot[1], p.e2, p.curve_2, r_comp=p.r_comp, q=p.q, beta=p.beta, h=p.h)
    _, tang_12 = eval_trellis_shear_and_tangent(eps_tot[2], p.g12, p.g_lock, p.gamma_lock, curve=p.curve_shear)

    t1 = float(tang_1[0])
    t2 = float(tang_2[0])
    t12 = float(tang_12[0])

    det_c = 1.0 - p.nu12 * p.nu21
    det_c = det_c if abs(det_c) > _EM20 else 1.0

    c_mat = np.array([
        [t1 / det_c, p.nu21 * t1 / det_c, 0.0],
        [p.nu12 * t2 / det_c, t2 / det_c, 0.0],
        [0.0, 0.0, t12],
    ], dtype=np.float64)

    # Material angle rotation if applicable
    ang = extra.get("angle", p.params.get("angle", 0.0)) if extra else 0.0
    if isinstance(ang, (int, float, np.number)) and abs(ang) > 1.0e-5:
        cos_a = math.cos(float(ang))
        sin_a = math.sin(float(ang))
        c2 = cos_a * cos_a
        s2 = sin_a * sin_a
        cs = cos_a * sin_a

        t_trans = np.array([
            [c2, s2, 2.0 * cs],
            [s2, c2, -2.0 * cs],
            [-cs, cs, c2 - s2],
        ], dtype=np.float64)
        t_inv = np.linalg.inv(t_trans)
        return t_inv @ c_mat @ t_trans

    return c_mat


shell_tangent = consistent_shell_tangent
shell_membrane_tangent = consistent_shell_tangent


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Algorithmic 3D solid tangent tensor (6, 6) for LAW120."""
    p = mat if isinstance(mat, Law120Params) else Law120Params.from_material(mat)

    eps_tot = extra.get("eps120", np.zeros(6)) if extra else np.zeros(6)
    eps_tot = np.atleast_1d(np.asarray(eps_tot, dtype=np.float64)).flatten()
    if len(eps_tot) < 6:
        eps_tot = np.zeros(6, dtype=np.float64)

    _, tang_1 = eval_yarn_stress_and_tangent(eps_tot[0], p.e1, p.curve_1, r_comp=p.r_comp, q=p.q, beta=p.beta, h=p.h)
    _, tang_2 = eval_yarn_stress_and_tangent(eps_tot[1], p.e2, p.curve_2, r_comp=p.r_comp, q=p.q, beta=p.beta, h=p.h)
    _, tang_12 = eval_trellis_shear_and_tangent(eps_tot[3], p.g12, p.g_lock, p.gamma_lock, curve=p.curve_shear)

    t1 = float(tang_1[0])
    t2 = float(tang_2[0])
    t12 = float(tang_12[0])

    det_c = 1.0 - p.nu12 * p.nu21
    det_c = det_c if abs(det_c) > _EM20 else 1.0

    c_66 = np.zeros((6, 6), dtype=np.float64)
    c_66[0, 0] = t1 / det_c
    c_66[0, 1] = p.nu21 * t1 / det_c
    c_66[1, 0] = p.nu12 * t2 / det_c
    c_66[1, 1] = t2 / det_c
    c_66[2, 2] = p.e3
    c_66[3, 3] = t12
    c_66[4, 4] = p.g23
    c_66[5, 5] = p.g31

    return c_66


consistent_solid_tangent = solid_tangent


def resolve(mat: Any, model: Any, log: Any = None) -> None:
    """Resolve /FUNCT curve references for /MAT/LAW120 from model."""
    if hasattr(mat, "tab_id") and mat.tab_id > 0:
        c = model.get_function(mat.tab_id)
        if c is not None:
            mat.curve_shear = c
    if hasattr(mat, "fct_1") and mat.fct_1 > 0:
        c = model.get_function(mat.fct_1)
        if c is not None:
            mat.curve_1 = c
    if hasattr(mat, "fct_2") and mat.fct_2 > 0:
        c = model.get_function(mat.fct_2)
        if c is not None:
            mat.curve_2 = c


def extra_shapes(mat: Any, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Return persistent history array shapes for LAW120."""
    if nip is not None:
        return {
            "eps120": (nip, 3),
            "dam120": (nip,),
            "soundsp": (nip,),
        }
    return {
        "eps120": (6,),
        "dam120": (),
        "soundsp": (),
    }


def build_law120(rec: Any) -> Material:
    """Constructor for /MAT/LAW120 (/MAT/TAPO) tape orientation composite material."""
    p = rec.params
    e = float(p.get("MAT_E", 1.0))
    nu = float(p.get("MAT_NU", 0.3))
    params = {
        "E": e if e > 0 else 1.0,
        "nu": nu if (0.0 <= nu < 0.5) else 0.3,
        "thick": float(p.get("MAT_THICK", 0.0)),
        "tab_id": int(p.get("MAT_TAB_ID", 0)),
        "xscale": float(p.get("MAT_Xscale", 1.0)),
        "yscale": float(p.get("MAT_Yscale", 1.0)),
        "tau": float(p.get("MAT_TAU", 0.0)),
        "q": float(p.get("MAT_Q", 0.0)),
        "b": float(p.get("MAT_B", 0.0)),
        "h": float(p.get("MAT_H", 0.0)),
        "af1": float(p.get("MAT_AF1", 0.0)),
        "af2": float(p.get("MAT_AF2", 0.0)),
        "ah1": float(p.get("MAT_AH1", 0.0)),
        "ah2": float(p.get("MAT_AH2", 0.0)),
        "as": float(p.get("MAT_AS", 0.0)),
        "d1c": float(p.get("MAT_D1C", 0.0)),
        "d2c": float(p.get("MAT_D2C", 0.0)),
        "d1f": float(p.get("MAT_D1F", 0.0)),
        "d2f": float(p.get("MAT_D2F", 0.0)),
        "d_trx": float(p.get("D_TRX", 0.0)),
        "d_jc": float(p.get("D_JC", 0.0)),
        "exp": float(p.get("MAT_EXP", 0.0)),
        "cc": float(p.get("MAT_CC", 0.0)),
        "gam0": float(p.get("MAT_GAM0", 0.0)),
        "gamf": float(p.get("MAT_GAMF", 0.0)),
        "iform": int(p.get("MAT_IFORM", 1)),
        "itrx": int(p.get("MAT_ITRX", 2)),
        "idam": int(p.get("MAT_IDAM", 2)),
    }
    return Material(id=rec.id, law=120, rho0=rec.density, title=rec.title, params=params)

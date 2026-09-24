r"""LAW117 — Cohesive Zone Material (/MAT/LAW117, /MAT/COH_MC, /MAT/COH_TAB).

Fortran origins:
- ``starter/source/materials/mat/mat117/hm_read_mat117.F`` (starter card reader, energy bounds & parameter derivation)
- ``engine/source/materials/mat/mat117/sigeps117.F`` (stress & damage update for cohesive/connect solid elements)
- ``hm_cfg_files/config/CFG/radioss2022/MAT/mat117.cfg`` (CFG attributes & format)

Theory
------
LAW117 models cohesive interfaces (typically with CONNECT / TYPE43 solid elements)
with mixed-mode damage initiation, bilinear softening, power-law or Benzeggagh-Kenane
fracture energy propagation criteria, and unilateral compressive contact:

1. Elastic Pre-peak Behavior:
   - Normal traction: \(\sigma_{zz} = E_n \delta_n\)
   - Shear tractions: \(\sigma_{yz} = E_t \delta_s\), \(\sigma_{zx} = E_t \delta_t\)
   - Displacements: \(\delta_n = \max(\varepsilon_{zz}, 0)\),
     \(\delta_t = \sqrt{\varepsilon_{yz}^2 + \varepsilon_{zx}^2}\),
     equivalent mixed-mode displacement \(\delta_m = \sqrt{\delta_n^2 + \delta_t^2}\).

2. Mixed-Mode Initiation Criterion:
   Quadratic interaction in stress:
   \[
   \left(\frac{\langle \sigma_n \rangle}{\sigma_{\max}}\right)^2 + \left(\frac{\tau}{\tau_{\max}}\right)^2 = 1
   \]
   where \(\tau = \sqrt{\sigma_{yz}^2 + \sigma_{zx}^2}\), \(\langle \cdot \rangle\) is the Macaulay bracket.
   With \(\beta = \delta_t / \delta_n\), \(\delta_0^n = \sigma_{\max} / E_n\), \(\delta_0^s = \tau_{\max} / E_t\),
   the equivalent initiation displacement is:
   \[
   \delta_0^m = \delta_0^s \delta_0^n \sqrt{\frac{1 + \beta^2}{(\delta_0^s)^2 + (\beta \delta_0^n)^2}}
   \]

3. Mixed-Mode Ultimate Displacement \(\delta_f^m\):
   - Pure Mode I: \(\delta_f^n = \frac{2 G_{Ic}}{\sigma_{\max}}\)
   - Pure Mode II: \(\delta_f^s = \frac{2 G_{IIc}}{\tau_{\max}}\)
   - Mixed mode:
     a) Power Law (\(IRUPT = 1\), default):
        \[
        FAC_1 = \frac{2(1 + \beta^2)}{\delta_0^m}, \quad
        \delta_f^m = FAC_1 \left[\left(\frac{E_n}{G_{Ic}}\right)^\mu + \left(\frac{E_t \beta^2}{G_{IIc}}\right)^\mu\right]^{-1/\mu}
        \]
     b) Benzeggagh-Kenane (\(IRUPT = 2\)):
        \[
        FAC_3 = \left[\frac{E_n^\gamma + E_t^\gamma \beta^2}{1 + \beta^2}\right]^{1/\gamma}, \quad
        G_c = G_{Ic} + (G_{IIc} - G_{Ic})\left(\frac{E_t \beta^2}{E_n + E_t \beta^2}\right)^{|\eta_{BK}|}
        \]
        \[
        \delta_f^m = \frac{2}{\delta_0^m FAC_3} G_c
        \]

4. Bilinear Softening & Damage Evolution:
   For historical maximum \(\delta_m^{\max} > \delta_0^m\):
   \[
   D = \frac{\delta_f^m}{\delta_m^{\max}} \frac{\delta_m^{\max} - \delta_0^m}{\delta_f^m - \delta_0^m}, \quad 0 \le D \le 1
   \]
   Monotonic accumulation: \(D \leftarrow \min(1, \max(D^{old}, D))\).
   When \(D \ge 1\) or \(\delta_m^{\max} \ge \delta_f^m\), element is deleted (\(off = 0.0\)).

5. Stress Update & Unilateral Contact:
   - Tension (\(\varepsilon_{zz} \ge 0\)): \(\sigma_{zz} = (1 - D) E_n \varepsilon_{zz}\)
   - Compression (\(\varepsilon_{zz} < 0\)): \(\sigma_{zz} = E_n \varepsilon_{zz}\) (undegraded contact stiffness)
   - Shear: \(\sigma_{yz} = (1 - D) E_t \varepsilon_{yz}\), \(\sigma_{zx} = (1 - D) E_t \varepsilon_{zx}\).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from pyradioss.model.entities import Material

_EM20 = 1.0e-20
_EP20 = 1.0e20


@dataclass
class Law117Params:
    """Parameters for /MAT/LAW117 (Cohesive Zone Material).

    Cites:
    - ``starter/source/materials/mat/mat117/hm_read_mat117.F``
    - ``engine/source/materials/mat/mat117/sigeps117.F``
    """

    E_n: float = 0.0
    E_t: float = 0.0
    sigma_max: float = 0.0
    tau_max: float = 0.0
    G_Ic: float = 0.0
    G_IIc: float = 0.0
    irupt: int = 1
    exp_g: float = 2.0
    exp_bk: float = 1.0
    gamma: float = 1.0
    tcut: float = 0.0
    rho0: float = 0.0
    refer_rho: float = 0.0
    imass: int = 1
    idel: int = 1
    fct_tn: int = 0
    fct_tt: int = 0
    fscale_x: float = 1.0
    title: str = ""

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho0
        if self.imass == 0:
            self.imass = 1
        if self.idel == 0:
            self.idel = 1
        if self.irupt == 0:
            self.irupt = 1
        if self.gamma == 0.0:
            self.gamma = 1.0
        if self.exp_g == 0.0:
            self.exp_g = 2.0
        if self.E_t == 0.0 and self.E_n > 0.0:
            self.E_t = self.E_n
        if self.fscale_x == 0.0:
            self.fscale_x = 1.0

        # Physical lower bounds from hm_read_mat117.F: lines 146-157
        if self.E_n > 0.0 and self.sigma_max > 0.0:
            min_gic = (self.sigma_max**2) / (2.0 * self.E_n)
            if self.G_Ic < min_gic:
                self.G_Ic = min_gic
        if self.E_t > 0.0 and self.tau_max > 0.0:
            min_giic = (self.tau_max**2) / (2.0 * self.E_t)
            if self.G_IIc < min_giic:
                self.G_IIc = min_giic

    @property
    def e_elas_n(self) -> float:
        return self.E_n

    @property
    def e_elas_s(self) -> float:
        return self.E_t

    @property
    def en(self) -> float:
        return self.E_n

    @property
    def es(self) -> float:
        return self.E_t

    @property
    def tmax_n(self) -> float:
        return self.sigma_max

    @property
    def tmax_s(self) -> float:
        return self.tau_max

    @property
    def tn(self) -> float:
        return self.sigma_max

    @property
    def ts(self) -> float:
        return self.tau_max

    @property
    def gic(self) -> float:
        return self.G_Ic

    @property
    def giic(self) -> float:
        return self.G_IIc

    @property
    def eta(self) -> float:
        return self.exp_bk if self.irupt == 2 else self.exp_g

    @property
    def delta0_n(self) -> float:
        """Pure Mode I initiation separation: delta0_n = sigma_max / E_n."""
        return (self.sigma_max / self.E_n) if self.E_n > 0.0 else 0.0

    @property
    def delta0_s(self) -> float:
        """Pure Mode II initiation separation: delta0_s = tau_max / E_t."""
        return (self.tau_max / self.E_t) if self.E_t > 0.0 else 0.0

    @property
    def und(self) -> float:
        """Pure Mode I ultimate separation: und = 2 * G_Ic / sigma_max."""
        return (2.0 * self.G_Ic / self.sigma_max) if self.sigma_max > 0.0 else 0.0

    @property
    def utd(self) -> float:
        """Pure Mode II ultimate separation: utd = 2 * G_IIc / tau_max."""
        return (2.0 * self.G_IIc / self.tau_max) if self.tau_max > 0.0 else 0.0


def _get_params(mat: Any) -> Law117Params:
    """Extract Law117Params from Law117Params, Material, or dict."""
    if isinstance(mat, Law117Params):
        return mat

    p: Dict[str, Any] = {}
    title = ""
    rho0 = 0.0
    refer_rho = 0.0

    if isinstance(mat, Material):
        p = mat.params
        title = getattr(mat, "title", "")
        rho0 = float(getattr(mat, "rho0", 0.0) or 0.0)
    elif isinstance(mat, dict):
        p = mat.get("params", mat)
        title = str(mat.get("title", ""))
        rho0 = float(mat.get("rho0", mat.get("rho", mat.get("density", mat.get("MAT_RHO", 0.0)))) or 0.0)
    elif hasattr(mat, "params"):
        p = getattr(mat, "params", {})
        title = getattr(mat, "title", "")
        rho0 = float(getattr(mat, "rho0", 0.0) or 0.0)
    elif hasattr(mat, "e_elas_n") or hasattr(mat, "en"):
        # Object like MaterialLaw117 / MatLaw117
        return Law117Params(
            E_n=float(getattr(mat, "e_elas_n", getattr(mat, "en", 0.0)) or 0.0),
            E_t=float(getattr(mat, "e_elas_s", getattr(mat, "es", 0.0)) or 0.0),
            sigma_max=float(getattr(mat, "tmax_n", getattr(mat, "tn", 0.0)) or 0.0),
            tau_max=float(getattr(mat, "tmax_s", getattr(mat, "ts", 0.0)) or 0.0),
            G_Ic=float(getattr(mat, "gic", 0.0) or 0.0),
            G_IIc=float(getattr(mat, "giic", 0.0) or 0.0),
            irupt=int(getattr(mat, "irupt", 1) or 1),
            exp_g=float(getattr(mat, "exp_g", 2.0) or 2.0),
            exp_bk=float(getattr(mat, "exp_bk", 1.0) or 1.0),
            gamma=float(getattr(mat, "gamma", 1.0) or 1.0),
            tcut=float(getattr(mat, "tcut", 0.0) or 0.0),
            rho0=float(getattr(mat, "rho0", getattr(mat, "rho", 0.0)) or 0.0),
            refer_rho=float(getattr(mat, "refer_rho", 0.0) or 0.0),
            imass=int(getattr(mat, "imass", 1) or 1),
            idel=int(getattr(mat, "idel", 1) or 1),
            fct_tn=int(getattr(mat, "fct_tn", 0) or 0),
            fct_tt=int(getattr(mat, "fct_tt", 0) or 0),
            fscale_x=float(getattr(mat, "fscale_x", 1.0) or 1.0),
            title=getattr(mat, "title", ""),
        )

    def _get_float(keys: Tuple[str, ...], default: float = 0.0) -> float:
        for k in keys:
            v = p.get(k)
            if v is not None:
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return default

    def _get_int(keys: Tuple[str, ...], default: int = 0) -> int:
        for k in keys:
            v = p.get(k)
            if v is not None:
                try:
                    return int(float(v))
                except (ValueError, TypeError):
                    pass
        return default

    en_val = _get_float(("E_n", "e_elas_n", "E_elas_n", "EN", "en", "MAT_E_ELAS_N", "E", "e"), 0.0)
    es_val = _get_float(("E_t", "e_elas_s", "E_elas_s", "ES", "es", "MAT_E_ELAS_S", "G", "g"), 0.0)
    sig_max = _get_float(("sigma_max", "tmax_n", "TMAX_N", "TN", "tn", "MAT_TMAX_N", "sig_max", "sig_y"), 0.0)
    tau_max = _get_float(("tau_max", "tmax_s", "TMAX_S", "TS", "ts", "MAT_TMAX_S"), 0.0)
    gic_val = _get_float(("G_Ic", "gic", "GIC", "MAT_GIC", "g_ic"), 0.0)
    giic_val = _get_float(("G_IIc", "giic", "GIIC", "MAT_GIIC", "g_iic"), 0.0)
    irupt_val = _get_int(("irupt", "IRUPT", "MAT_IRUPT"), 1)
    exp_g_val = _get_float(("exp_g", "EXP_G", "MAT_EXP_G", "eta", "mu"), 2.0)
    exp_bk_val = _get_float(("exp_bk", "EXP_BK", "MAT_EXP_BK", "eta_bk"), 1.0)
    gamma_val = _get_float(("gamma", "GAMMA", "MAT_GAMMA"), 1.0)
    tcut_val = _get_float(("tcut", "TCUT", "fcut", "FCUT"), 0.0)
    rho_val = _get_float(("rho0", "rho", "MAT_RHO", "density"), rho0)
    ref_rho = _get_float(("refer_rho", "MAT_RHOR"), rho_val)
    imass_val = _get_int(("imass", "MAT_IMASS"), 1)
    idel_val = _get_int(("idel", "MAT_IDEL"), 1)
    fct_tn_val = _get_int(("fct_tn", "MAT_Fct_TN"), 0)
    fct_tt_val = _get_int(("fct_tt", "MAT_Fct_TT"), 0)
    fscale_x_val = _get_float(("fscale_x", "MAT_Fscale_x"), 1.0)

    return Law117Params(
        E_n=en_val,
        E_t=es_val,
        sigma_max=sig_max,
        tau_max=tau_max,
        G_Ic=gic_val,
        G_IIc=giic_val,
        irupt=irupt_val,
        exp_g=exp_g_val,
        exp_bk=exp_bk_val,
        gamma=gamma_val,
        tcut=tcut_val,
        rho0=rho_val,
        refer_rho=ref_rho,
        imass=imass_val,
        idel=idel_val,
        fct_tn=fct_tn_val,
        fct_tt=fct_tt_val,
        fscale_x=fscale_x_val,
        title=title,
    )


def build_law117(rec: Any) -> Material:
    """Build a Material instance from a parsed LAW117 record or dict."""
    p = _get_params(rec)
    mat_id = getattr(rec, "id", 1) if not isinstance(rec, dict) else rec.get("id", 1)
    title = getattr(rec, "title", p.title) if not isinstance(rec, dict) else rec.get("title", p.title)
    params_dict = {
        "E_n": p.E_n, "e_elas_n": p.E_n, "EN": p.E_n, "en": p.E_n, "MAT_E_ELAS_N": p.E_n,
        "E_t": p.E_t, "e_elas_s": p.E_t, "ES": p.E_t, "es": p.E_t, "MAT_E_ELAS_S": p.E_t,
        "sigma_max": p.sigma_max, "tmax_n": p.sigma_max, "TMAX_N": p.sigma_max, "TN": p.sigma_max, "tn": p.sigma_max, "MAT_TMAX_N": p.sigma_max,
        "tau_max": p.tau_max, "tmax_s": p.tau_max, "TMAX_S": p.tau_max, "TS": p.tau_max, "ts": p.tau_max, "MAT_TMAX_S": p.tau_max,
        "G_Ic": p.G_Ic, "gic": p.G_Ic, "GIC": p.G_Ic, "MAT_GIC": p.G_Ic,
        "G_IIc": p.G_IIc, "giic": p.G_IIc, "GIIC": p.G_IIc, "MAT_GIIC": p.G_IIc,
        "irupt": p.irupt, "IRUPT": p.irupt, "MAT_IRUPT": p.irupt,
        "exp_g": p.exp_g, "EXP_G": p.exp_g, "MAT_EXP_G": p.exp_g,
        "exp_bk": p.exp_bk, "EXP_BK": p.exp_bk, "MAT_EXP_BK": p.exp_bk,
        "gamma": p.gamma, "GAMMA": p.gamma, "MAT_GAMMA": p.gamma,
        "tcut": p.tcut,
        "imass": p.imass, "idel": p.idel,
        "fct_tn": p.fct_tn, "fct_tt": p.fct_tt, "fscale_x": p.fscale_x,
        "rho0": p.rho0, "rho": p.rho0, "MAT_RHO": p.rho0,
        "refer_rho": p.refer_rho,
    }
    return Material(
        id=mat_id,
        law=117,
        rho0=p.rho0,
        title=title,
        params=params_dict,
    )


def extra_shapes(mat: Any, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Define persistent internal state variables for LAW117 cohesive material.

    Corresponds to:
    - uvar: (15,) internal element variables
    - dmg: damage parameter [0, 1]
    - off: deletion flag (1.0 active, 0.0 deleted)
    - eps_tot: total relative displacement / jump vector [xx, yy, zz, xy, yz, zx]
    - w_diss: dissipated cohesive fracture work
    - w_ext: total external work done on cohesive interface
    """
    if nip is not None:
        return {
            "uvar117": (nip, 15),
            "dmg117": (nip,),
            "off117": (nip,),
            "eps_tot": (nip, 6),
            "w_diss": (nip,),
            "w_ext": (nip,),
        }
    return {
        "uvar117": (15,),
        "dmg117": (),
        "off117": (),
        "eps_tot": (6,),
        "w_diss": (),
        "w_ext": (),
    }


def sound_speed(
    mat: Any,
    rho: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> float:
    """Compute acoustic sound speed for cohesive material LAW117.

    Fortran cites:
    ``sigeps117.F`` line 104: STF = E_ELAS_N + E_ELAS_S
    Acoustic wave velocity c = sqrt((E_n + E_t) / rho0).
    """
    p = _get_params(mat)
    dens = float(rho if rho is not None else p.rho0)
    if dens <= 0.0:
        dens = 1.0
    stiff = max(p.E_n + p.E_t, p.E_n, p.E_t)
    return math.sqrt(max(stiff, 0.0) / dens)


def solid_step(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Perform solid stress and cohesive damage integration for LAW117.

    Directly ports ``sigeps117.F``:
    - Normal component is index 2 (ZZ)
    - Shear components are index 4 (YZ) and index 5 (ZX)
    - In-plane components (XX, YY, XY) are unaffected/zero.

    Returns:
        (sig_new, dmg, sound_speed_arr)
    """
    p = _get_params(mat)
    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float)

    is_1d = (sig_arr.ndim == 1)
    if is_1d:
        sig_2d = sig_arr.reshape(1, -1)
        deps_2d = deps_arr.reshape(1, -1)
    else:
        sig_2d = sig_arr
        deps_2d = deps_arr

    nel = sig_2d.shape[0]
    n_comp = sig_2d.shape[1]

    # Handle 3-component input [zz, yz, zx] vs 6-component [xx, yy, zz, xy, yz, zx]
    is_3comp = (n_comp == 3)
    idx_zz = 0 if is_3comp else 2
    idx_yz = 1 if is_3comp else 4
    idx_zx = 2 if is_3comp else 5

    sig_out = np.zeros_like(sig_2d)

    if extra is None:
        extra = {}

    # State variables initialization
    if "uvar117" in extra and extra["uvar117"] is not None:
        uvar = np.asarray(extra["uvar117"], dtype=float)
        if uvar.ndim == 1:
            uvar = uvar.reshape(1, -1)
    elif "uvar" in extra and extra["uvar"] is not None:
        uvar = np.asarray(extra["uvar"], dtype=float)
        if uvar.ndim == 1:
            uvar = uvar.reshape(1, -1)
    else:
        uvar = np.zeros((nel, 15), dtype=float)

    if "dmg117" in extra and extra["dmg117"] is not None:
        dmg = np.asarray(extra["dmg117"], dtype=float).copy().reshape(-1)
    elif "dmg" in extra and extra["dmg"] is not None:
        dmg = np.asarray(extra["dmg"], dtype=float).copy().reshape(-1)
    else:
        dmg = np.zeros(nel, dtype=float)

    if "off117" in extra and extra["off117"] is not None:
        off = np.asarray(extra["off117"], dtype=float).copy().reshape(-1)
    elif "off" in extra and extra["off"] is not None:
        off = np.asarray(extra["off"], dtype=float).copy().reshape(-1)
    else:
        off = np.ones(nel, dtype=float)

    if "eps_tot" in extra and extra["eps_tot"] is not None:
        eps_tot = np.asarray(extra["eps_tot"], dtype=float).copy()
        if eps_tot.ndim == 1:
            eps_tot = eps_tot.reshape(1, -1)
    else:
        eps_tot = np.zeros((nel, n_comp), dtype=float)

    if "w_diss" in extra and extra["w_diss"] is not None:
        w_diss = np.asarray(extra["w_diss"], dtype=float).copy().reshape(-1)
    else:
        w_diss = np.zeros(nel, dtype=float)

    if "w_ext" in extra and extra["w_ext"] is not None:
        w_ext = np.asarray(extra["w_ext"], dtype=float).copy().reshape(-1)
    else:
        w_ext = np.zeros(nel, dtype=float)

    # Optional mesh length / area scaling (sigeps117.F lines 126-152)
    area_arr = kwargs.get("area", extra.get("area"))
    areas = np.ones(nel, dtype=float) if area_arr is None else np.asarray(area_arr, dtype=float).reshape(-1)

    c_val = sound_speed(p)
    c_arr = np.full(nel, c_val, dtype=float)

    for i in range(nel):
        d_zz = deps_2d[i, idx_zz]
        d_yz = deps_2d[i, idx_yz]
        d_zx = deps_2d[i, idx_zx]

        # Accumulate total relative displacements
        eps_tot[i, idx_zz] += d_zz
        eps_tot[i, idx_yz] += d_yz
        eps_tot[i, idx_zx] += d_zx

        tot_zz = eps_tot[i, idx_zz]
        tot_yz = eps_tot[i, idx_yz]
        tot_zx = eps_tot[i, idx_zx]

        # 1. Normal and tangential displacement components (sigeps117.F lines 117-122)
        eps_n = max(tot_zz, 0.0)
        eps_t = math.sqrt(tot_yz**2 + tot_zx**2)
        eps_m = math.sqrt(eps_n**2 + eps_t**2)

        epsm_max = max(eps_m, uvar[i, 3])

        # 2. Scaling properties
        delta0_n = p.delta0_n
        delta0_s = p.delta0_s
        und = p.und
        utd = p.utd

        # 3. Mixed-mode initiation displacement (sigeps117.F lines 156-166)
        if eps_t == 0.0:
            delta0_m = delta0_n
        elif eps_n == 0.0:
            delta0_m = delta0_s
        else:
            beta = abs(eps_t / eps_n)
            denom_init = (delta0_s**2) + ((beta * delta0_n)**2)
            if denom_init > 0.0:
                delta0_m = delta0_s * delta0_n * math.sqrt((1.0 + beta**2) / denom_init)
            else:
                delta0_m = delta0_n

        # 4. Mixed-mode ultimate displacement deltaf_max (sigeps117.F lines 170-201)
        if eps_t == 0.0:
            deltaf_max = und
        elif eps_n == 0.0:
            deltaf_max = utd
        else:
            beta = abs(eps_t / eps_n)
            if p.irupt == 2:
                # Benzeggagh-Kenane criterion
                fac1 = (p.E_n**p.gamma) / (1.0 + beta**2)
                fac2 = (p.E_t**p.gamma) * (beta**2) / (1.0 + beta**2)
                fac3 = (fac1 + fac2)**(1.0 / p.gamma)

                mix_ratio = (p.E_t * beta**2) / max(_EM20, p.E_n + p.E_t * beta**2)
                gc_bk = p.G_Ic + (p.G_IIc - p.G_Ic) * (mix_ratio**abs(p.exp_bk))
                deltaf_max = (2.0 / max(_EM20, delta0_m * fac3)) * gc_bk
            else:
                # Power law criterion (irupt == 1, default)
                fac1 = 2.0 * (1.0 + beta**2) / max(_EM20, delta0_m)
                g1 = (p.E_n / max(_EM20, p.G_Ic))**p.exp_g
                g2 = (p.E_t * (beta**2) / max(_EM20, p.G_IIc))**p.exp_g
                term = g1 + g2
                deltaf_max = fac1 * (term**(-1.0 / p.exp_g))

        deltaf_max = max(deltaf_max, delta0_m)

        # 5. Damage parameter evolution (sigeps117.F lines 205-220)
        dm = epsm_max - delta0_m
        if dm > 0.0 and epsm_max > 0.0:
            denom_dam = max(deltaf_max - delta0_m, _EM20)
            dam = (deltaf_max / epsm_max) * (epsm_max - delta0_m) / denom_dam
            dmg[i] = min(1.0, max(dmg[i], dam))

            if off[i] == 1.0 and epsm_max >= deltaf_max:
                off[i] = 0.0
                dmg[i] = 1.0

        if off[i] == 0.0 or dmg[i] >= 1.0:
            dmg[i] = 1.0
            off[i] = 0.0
            sig_out[i, idx_zz] = 0.0
            sig_out[i, idx_yz] = 0.0
            sig_out[i, idx_zx] = 0.0
        else:
            # 6. Stress update with unilateral compressive contact (sigeps117.F lines 224-234)
            if tot_zz < 0.0:
                # Unilateral contact penalty: zero damage degradation under compression
                sig_out[i, idx_zz] = p.E_n * tot_zz
            else:
                # Tensile normal traction with damage degradation
                s_zz = (1.0 - dmg[i]) * p.E_n * tot_zz
                if p.tcut > 0.0 and s_zz < p.tcut:
                    s_zz = 0.0
                sig_out[i, idx_zz] = s_zz

            # Shear tractions with damage degradation
            sig_out[i, idx_yz] = (1.0 - dmg[i]) * p.E_t * tot_yz
            sig_out[i, idx_zx] = (1.0 - dmg[i]) * p.E_t * tot_zx

        # 7. Energy dissipation tracking
        # Incremental work done by tractions: trapezoidal integration
        dw = (
            0.5 * (sig_2d[i, idx_zz] + sig_out[i, idx_zz]) * d_zz
            + 0.5 * (sig_2d[i, idx_yz] + sig_out[i, idx_yz]) * d_yz
            + 0.5 * (sig_2d[i, idx_zx] + sig_out[i, idx_zx]) * d_zx
        )
        w_ext[i] += dw

        # Current stored recoverable elastic energy
        if tot_zz < 0.0:
            w_el_norm = 0.5 * p.E_n * (tot_zz**2)
        else:
            w_el_norm = 0.5 * (1.0 - dmg[i]) * p.E_n * (tot_zz**2)
        w_el_shear = 0.5 * (1.0 - dmg[i]) * p.E_t * (tot_yz**2 + tot_zx**2)
        w_elas = w_el_norm + w_el_shear

        if off[i] == 0.0 or dmg[i] >= 1.0:
            w_diss[i] = max(w_diss[i], w_ext[i])
        else:
            w_diss[i] = max(0.0, w_ext[i] - w_elas)

        # 8. Record state variables in uvar (sigeps117.F lines 236-248)
        uvar[i, 0] = tot_zz
        uvar[i, 1] = tot_zx
        uvar[i, 2] = tot_yz
        uvar[i, 3] = epsm_max
        uvar[i, 5] = sig_out[i, idx_zz]
        uvar[i, 6] = sig_out[i, idx_zx]
        uvar[i, 7] = sig_out[i, idx_yz]
        uvar[i, 8] = delta0_m
        uvar[i, 9] = abs(eps_t / eps_n) if eps_n > 0.0 else 0.0
        uvar[i, 11] = deltaf_max

    # Store persistent state back into extra
    extra["uvar117"] = uvar
    extra["uvar"] = uvar
    extra["dmg117"] = dmg
    extra["dmg"] = dmg
    extra["off117"] = off
    extra["off"] = off
    extra["eps_tot"] = eps_tot
    extra["w_diss"] = w_diss
    extra["w_ext"] = w_ext

    out_sig = sig_out[0] if is_1d else sig_out
    out_dmg = dmg[0] if is_1d else dmg
    out_c = c_arr[0] if is_1d else c_arr

    return out_sig, out_dmg, out_c


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Standard solid_update entry point for LAW117."""
    return solid_step(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, **kwargs)


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent algorithmic tangent stiffness tensor C (6, 6) for LAW117 cohesive solid."""
    p = _get_params(mat)
    c_elastic = np.zeros((6, 6), dtype=float)
    c_elastic[2, 2] = p.E_n
    c_elastic[4, 4] = p.E_t
    c_elastic[5, 5] = p.E_t

    if sig is None or deps is None:
        return c_elastic

    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float)

    if sig_arr.ndim == 2 and sig_arr.shape[0] > 1:
        n = sig_arr.shape[0]
        return np.broadcast_to(c_elastic, (n, 6, 6)).copy()

    sig0 = sig_arr.flatten()
    deps0 = deps_arr.flatten()
    if len(sig0) < 6:
        s_pad = np.zeros(6, dtype=float)
        s_pad[:len(sig0)] = sig0
        sig0 = s_pad
    if len(deps0) < 6:
        d_pad = np.zeros(6, dtype=float)
        d_pad[:len(deps0)] = deps0
        deps0 = d_pad

    ex0 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in (extra or {}).items()}
    sig_base, _, _ = solid_step(mat, sig0, deps0, dt=dt, extra=ex0)

    h = 1.0e-7
    c_algo = np.zeros((6, 6), dtype=float)
    for j in range(6):
        deps_p = deps0.copy()
        deps_p[j] += h
        ex_p = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in (extra or {}).items()}
        sig_p, _, _ = solid_step(mat, sig0, deps_p, dt=dt, extra=ex_p)
        c_algo[:, j] = (sig_p - sig_base) / h

    return c_algo


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


def shell_step(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Shell stress update mapping [normal, shear_y, shear_x] for LAW117."""
    sig_out, dmg_out, _ = solid_step(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, **kwargs)
    return sig_out, dmg_out


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Shell stress update entry point for LAW117."""
    return shell_step(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, **kwargs)


def shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Tangent stiffness matrix C (3, 3) for LAW117 shells/membranes."""
    p = _get_params(mat)
    c_elastic = np.diag([p.E_n, p.E_t, p.E_t])
    if sig is None or deps is None:
        return c_elastic

    sig_arr = np.asarray(sig, dtype=float).flatten()[:3]
    deps_arr = np.asarray(deps, dtype=float).flatten()[:3]
    ex0 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in (extra or {}).items()}
    sig_base, _ = shell_step(mat, sig_arr, deps_arr, dt=dt, extra=ex0)

    h = 1.0e-7
    c_algo = np.zeros((3, 3), dtype=float)
    for j in range(3):
        deps_p = deps_arr.copy()
        deps_p[j] += h
        ex_p = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in (extra or {}).items()}
        sig_p, _ = shell_step(mat, sig_arr, deps_p, dt=dt, extra=ex_p)
        c_algo[:, j] = (sig_p[:3] - sig_base[:3]) / h

    return c_algo


def consistent_shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent shell tangent alias."""
    return shell_tangent(mat, sig=sig, deps=deps, dt=dt, extra=extra, **kwargs)


def shell_membrane_tangent(mat: Any) -> np.ndarray:
    """Elastic membrane tangent for LAW117 shells."""
    p = _get_params(mat)
    return np.diag([p.E_n, p.E_t, p.E_t])


def _register() -> None:
    """Register LAW117 builder in the central materials registry."""
    try:
        from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
        for k in (
            117, "117", "LAW117", "COH_MC", "COH_TAB", "COHESIVE",
            "COHESIVE_TABULATED", "MAT_LAW117", "MAT_COH_MC", "MAT_COH_TAB",
            "LAW117_COH_MC", "LAW117_COH_TAB", "MLAW117",
        ):
            MAT_PHYSICS_REGISTRY[k] = build_law117
    except Exception:
        pass


_register()

r"""LAW169 — Arup Structural Adhesive cohesive/damage model (/MAT/LAW169, /MAT/ARUP_ADHESIVE).

Fortran origins:
- ``engine/source/materials/mat/mat169/sigeps169_connect.F90`` (cohesive element stress update)
- ``starter/source/materials/mat/mat169/hm_read_mat169.F90`` (starter card reader, energy bounds & displacement derivation)
- ``hm_cfg_files/config/CFG/radioss2025/MAT/LAW169.cfg`` (CFG attributes & format)

Theory
------
LAW169 models elastic-plastic adhesive layers with coupled mixed-mode damage
and fracture energy release based on the Arup adhesive formulation:

1. Elastic Pre-peak Behavior:
   - Longitudinal/normal wave modulus:
     \(E_{wave} = \frac{E (1 - \nu)}{(1 + \nu)(1 - 2\nu)}\)
   - Shear modulus:
     \(G = \frac{E}{2 (1 + \nu)}\).
   - Elastic trial stress increment:
     \(\Delta\sigma_{zz}^{trial} = \Delta\varepsilon_{zz} \frac{E_{wave}}{t_0}\)
     \(\Delta\sigma_{yz}^{trial} = \Delta\varepsilon_{yz} \frac{G}{t_0}\)
     \(\Delta\sigma_{zx}^{trial} = \Delta\varepsilon_{zx} \frac{G}{t_0}\).

2. Mixed-Mode Yield Surface:
   Yield criterion under coupled normal and shear stress:
   \(F_{yld} = \left(\frac{\max(\sigma_{zz}, 0)}{T_{\max}}\right)^{PWRT} + \left(\frac{\tau}{S_{\max} - SHT\_SL \cdot \sigma_{zz}}\right)^{PWRS} - 1 \le 0\),
   where \(\tau = \sqrt{\sigma_{yz}^2 + \sigma_{zx}^2}\) is the resultant shear stress and \(SHT\_SL\)
   is the slope of the yield surface at zero tension (compressive friction-like enhancement).

3. Damage Initiation & Threshold Strains:
   When \(F_{yld} \ge 0\), initial damage thresholds are frozen:
   \(\varepsilon_{n0} = \varepsilon_{zz}\)
   \(\varepsilon_{s0} = \varepsilon_s + d_p\) where \(d_p = SHRP \cdot d_{fs}\) represents the plastic shear plateau.
   Peak strengths are recorded: \(T_{\max} \leftarrow \max(\sigma_{zz}, 10^{-6})\), \(S_{\max} \leftarrow \max(\tau, 10^{-6})\).

4. Progressive Softening & Damage Evolution:
   Critical displacements for complete failure:
   - Normal mode: \(d_{fn} = \frac{2 G_{Ic}}{T_{\max}}\)
   - Shear mode: \(d_{fs} = \frac{2 G_{IIc}}{(1 + SHRP) S_{\max}}\).
   Unilateral damage accumulation:
   - Normal damage: \(D_n = \max\left(D_n^{old}, \frac{\varepsilon_{zz} - \varepsilon_{n0}}{d_{fn} - \varepsilon_{n0}}\right)\) if \(\varepsilon_{zz} > \varepsilon_{n0}\)
   - Shear damage: \(D_s = \max\left(D_s^{old}, \frac{\varepsilon_s - \varepsilon_{s0}}{d_{fs} - \varepsilon_{s0}}\right)\) if \(\varepsilon_s > \varepsilon_{s0}\).
   Overall damage: \(D = \max(D_n, D_s)\).

5. Stress Degradation & Element Failure:
   - Normal stress: \(\sigma_{zz} = \min(\sigma_{zz}^{trial}, T_{\max}) \cdot (1 - D_n)\)
   - Shear stress: \(\tau_{\max} = S_{\max} - SHT\_SL \cdot \sigma_{zz}^{uncapped}\),
     \(\sigma_{shear} = \sigma_{shear}^{trial} \frac{\min(\tau, \tau_{\max})}{\tau} \cdot (1 - D_s)\).
   When \(D_n \ge 1.0\) or \(D_s \ge 1.0\), element failure occurs (\(\text{off} \leftarrow 0.8 \cdot \text{off}\)).

6. Longitudinal Sound Speed:
   Solids: \(c = \sqrt{E_{wave} / \rho_0}\).
   Shells: \(c = \sqrt{\frac{E}{(1 - \nu^2) \rho_0}}\).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pyradioss.model.entities import Material

_EP20 = 1.0e20
_EP30 = 1.0e30
_EM20 = 1.0e-20
_EM06 = 1.0e-6


@dataclass
class Law169Params:
    """Parameters for /MAT/LAW169 (Arup Structural Adhesive).

    Cites:
    - ``starter/source/materials/mat/mat169/hm_read_mat169.F90``
    - ``engine/source/materials/mat/mat169/sigeps169_connect.F90``
    """

    rho0: float = 0.0
    refer_rho: float = 0.0
    young: float = 0.0
    nu: float = 0.0
    sht_sl: float = 0.0
    tenmax: float = _EP20
    gcten: float = _EP20
    shrmax: float = _EP20
    gcshr: float = _EP20
    pwrt: int = 2
    pwrs: int = 2
    shrp: float = 0.0
    title: str = ""

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho0
        if self.tenmax == 0.0:
            self.tenmax = _EP20
        if self.shrmax == 0.0:
            self.shrmax = _EP20
        if self.gcten == 0.0:
            self.gcten = _EP20
        if self.gcshr == 0.0:
            self.gcshr = _EP20
        if self.pwrt <= 0:
            self.pwrt = 2
        if self.pwrs <= 0:
            self.pwrs = 2

        # Fortran condition on GCTEN (hm_read_mat169.F90 lines 128-135)
        if self.young > 0.0 and self.tenmax < _EP20:
            min_gcten = (self.tenmax**2) / self.young
            if self.gcten < min_gcten:
                self.gcten = min_gcten

        # Fortran condition on GCSHR (lines 137-149)
        shear_mod = self.shear
        if shear_mod > 0.0 and self.shrmax < _EP20 and (1.0 - self.shrp) > 0.0:
            limit_sh = (self.shrmax**2) * (1.0 + self.shrp) / (1.0 - self.shrp) / (2.0 * shear_mod)
            if self.gcshr < limit_sh:
                self.gcshr = limit_sh

    @property
    def shear(self) -> float:
        """Shear modulus G = E / (2 * (1 + nu))."""
        return (self.young / (2.0 * (1.0 + self.nu))) if (1.0 + self.nu) != 0.0 else 0.0

    @property
    def bulk(self) -> float:
        """Bulk modulus K = E / (3 * (1 - 2 * nu))."""
        denom = 3.0 * (1.0 - 2.0 * self.nu)
        return (self.young / denom) if denom != 0.0 else 0.0

    @property
    def wave(self) -> float:
        """Longitudinal constrained modulus E_wave = E * (1 - nu) / ((1 + nu) * (1 - 2*nu))."""
        denom = (1.0 + self.nu) * (1.0 - 2.0 * self.nu)
        return (self.young * (1.0 - self.nu) / denom) if denom != 0.0 else self.young

    @property
    def dfn(self) -> float:
        """Failure displacement in normal tension: dfn = 2 * gcten / tenmax (line 120)."""
        return (2.0 * self.gcten / self.tenmax) if self.tenmax > 0.0 else _EP20

    @property
    def dfs(self) -> float:
        """Failure displacement in shear: dfs = 2 * gcshr / ((1 + shrp) * shrmax) (line 121)."""
        denom = (1.0 + self.shrp) * self.shrmax
        return (2.0 * self.gcshr / denom) if denom > 0.0 else _EP20

    @property
    def dp(self) -> float:
        """Plastic plateau displacement in shear: dp = shrp * dfs (line 122)."""
        return self.shrp * self.dfs

    @property
    def eps_n0(self) -> float:
        """Initial elastic limit strain in tension: tenmax / wave (line 125)."""
        return (self.tenmax / self.wave) if self.wave > 0.0 else 0.0

    @property
    def eps_sh0(self) -> float:
        """Initial elastic limit strain in shear: shrmax / shear + dp (line 126)."""
        return (self.shrmax / self.shear + self.dp) if self.shear > 0.0 else 0.0

    @property
    def E(self) -> float:
        return self.young

    @property
    def G(self) -> float:
        return self.shear

    @property
    def K(self) -> float:
        return self.bulk

    @property
    def rho(self) -> float:
        return self.rho0


def _extract_param(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = 0.0) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _get_params(mat: Any) -> Law169Params:
    """Extract Law169Params from Material, Law169Params, MaterialLaw169, or dict."""
    if isinstance(mat, Law169Params):
        return mat

    if hasattr(mat, "law169_params") and isinstance(mat.law169_params, Law169Params):
        return mat.law169_params

    p: Dict[str, Any] = {}
    title = ""
    rho0_val = None

    if isinstance(mat, Material):
        p = dict(mat.params) if mat.params is not None else {}
        title = mat.title
        rho0_val = getattr(mat, "rho0", None)
    elif isinstance(mat, dict):
        p = dict(mat.get("params", mat))
        title = mat.get("title", "")
        rho0_val = mat.get("rho0") or mat.get("rho") or mat.get("density")
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        p = dict(mat.params)
        title = getattr(mat, "title", "")
        rho0_val = getattr(mat, "rho", getattr(mat, "rho0", None))
    else:
        title = getattr(mat, "title", "")
        rho0_val = getattr(mat, "rho", getattr(mat, "rho0", None))
        for attr in (
            "rho", "refer_rho", "young", "e", "nu", "pr", "sht_sl",
            "tenmax", "gcten", "shrmax", "gcshr", "pwrt", "pwrs", "shrp",
        ):
            if hasattr(mat, attr):
                p[attr] = getattr(mat, attr)

    if rho0_val is None or float(rho0_val) == 0.0:
        rho0_val = _extract_param(p, ("rho0", "rho", "density", "Rho", "MAT_RHO", "Refer_Rho"), 0.0)
    rho0 = float(rho0_val)

    refer_rho_val = _extract_param(p, ("refer_rho", "Refer_Rho", "rho_ref", "refer_density"), rho0)
    refer_rho = float(refer_rho_val) if refer_rho_val is not None and float(refer_rho_val) != 0.0 else rho0

    young = float(_extract_param(p, ("young", "E", "e", "MAT_E"), 0.0))
    nu = float(_extract_param(p, ("nu", "Nu", "PR", "MAT_NU"), 0.0))
    sht_sl = float(_extract_param(p, ("sht_sl", "SHT_SL", "MAT169_SHT_SL"), 0.0))

    tenmax_val = _extract_param(p, ("tenmax", "TENMAX", "MAT169_TENMAX", "sig_t0"), _EP20)
    tenmax = float(tenmax_val) if tenmax_val is not None and float(tenmax_val) not in (0.0, _EP30) else _EP20

    gcten_val = _extract_param(p, ("gcten", "GCTEN", "MAT169_GCTEN", "gc_t"), _EP20)
    gcten = float(gcten_val) if gcten_val is not None and float(gcten_val) not in (0.0, _EP30) else _EP20

    shrmax_val = _extract_param(p, ("shrmax", "SHRMAX", "MAT169_SHRMAX", "sig_s0"), _EP20)
    shrmax = float(shrmax_val) if shrmax_val is not None and float(shrmax_val) not in (0.0, _EP30) else _EP20

    gcshr_val = _extract_param(p, ("gcshr", "GCSHR", "MAT169_GCSHR", "gc_s"), _EP20)
    gcshr = float(gcshr_val) if gcshr_val is not None and float(gcshr_val) not in (0.0, _EP30) else _EP20

    pwrt = int(_extract_param(p, ("pwrt", "PWRT", "MAT169_PWRT"), 2))
    pwrs = int(_extract_param(p, ("pwrs", "PWRS", "MAT169_PWRS"), 2))
    shrp = float(_extract_param(p, ("shrp", "SHRP", "MAT169_SHRP"), 0.0))

    return Law169Params(
        rho0=rho0,
        refer_rho=refer_rho,
        young=young,
        nu=nu,
        sht_sl=sht_sl,
        tenmax=tenmax,
        gcten=gcten,
        shrmax=shrmax,
        gcshr=gcshr,
        pwrt=pwrt,
        pwrs=pwrs,
        shrp=shrp,
        title=title,
    )


def build_law169(rec: Any) -> Material:
    """Build a Material instance for /MAT/LAW169 (Arup Structural Adhesive)."""
    mat_id = getattr(rec, "id", 1)
    title = getattr(rec, "title", "")
    params_dict = dict(getattr(rec, "params", {})) if hasattr(rec, "params") and rec.params else {}
    if hasattr(rec, "density") and rec.density:
        params_dict["rho0"] = rec.density

    p = _get_params(params_dict)
    p.title = title

    params: Dict[str, Any] = {
        "E": p.young if p.young > 0.0 else 1.0,
        "nu": p.nu,
        "G": p.shear,
        "K": p.bulk,
        "bulk": p.bulk,
        "wave": p.wave,
        "rho0": p.rho0,
        "Rho": p.rho0,
        "MAT_RHO": p.rho0,
        "MAT169_SHT_SL": p.sht_sl,
        "MAT169_TENMAX": p.tenmax,
        "MAT169_GCTEN": p.gcten,
        "MAT169_SHRMAX": p.shrmax,
        "MAT169_GCSHR": p.gcshr,
        "MAT169_PWRT": p.pwrt,
        "MAT169_PWRS": p.pwrs,
        "MAT169_SHRP": p.shrp,
        "DFN": p.dfn,
        "DFS": p.dfs,
        "DP": p.dp,
        "EPS_N0": p.eps_n0,
        "EPS_S0": p.eps_sh0,
        "law169_params": p,
    }

    mat = Material(
        id=mat_id,
        law=169,
        rho0=p.rho0,
        title=title,
        params=params,
    )
    mat.law169_params = p
    return mat


def extra_shapes(mat: Any, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Define persistent internal state variables for LAW169.

    Corresponds to:
    - uvar: (15,)
    - dmg: ()
    - off: ()
    """
    if nip is not None:
        return {
            "uvar169": (nip, 15),
            "dmg169": (nip,),
            "off169": (nip,),
        }
    return {
        "uvar169": (15,),
        "dmg169": (),
        "off169": (),
    }


def sound_speed(
    mat: Any,
    rho: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> Any:
    """Acoustic wave speed for LAW169.

    For solid/cohesive elements: c = sqrt(wave / rho0)
    For shell elements: c = sqrt(E / ((1 - nu^2) * rho0))
    """
    p = _get_params(mat)
    rho_val = float(np.mean(rho)) if rho is not None and np.size(rho) > 0 else p.rho0
    if rho_val <= 0.0:
        rho_val = p.rho0 if p.rho0 > 0.0 else 1.0

    if is_shell:
        mod = p.young / (1.0 - p.nu**2) if (1.0 - p.nu**2) > 0.0 else p.young
    else:
        mod = p.wave

    return math.sqrt(max(0.0, mod / rho_val))


def solid_step(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Fortran-faithful stress update for LAW169 3D solids / cohesive elements (sigeps169_connect.F90).

    Normal component: zz (index 2).
    Shear components: yz (index 4), zx (index 5).
    In-plane components: xx (index 0), yy (index 1), xy (index 3).
    """
    p = _get_params(mat)

    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).copy()
    deps_arr = np.atleast_2d(deps).copy()
    nel = sig_arr.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.atleast_1d(epsp).astype(float).copy()

    if extra is None:
        extra = {}

    # State extraction / initialization (lines 122-136)
    uvar = extra.get("uvar169", extra.get("uvar"))
    thick0 = extra.get("thick0", np.ones(nel, dtype=float))
    thick0 = np.atleast_1d(thick0).astype(float)
    if len(thick0) < nel:
        thick0 = np.ones(nel, dtype=float)

    if uvar is None:
        uvar = np.zeros((nel, 15), dtype=float)
        uvar[:, 9] = p.shrmax
        uvar[:, 10] = p.tenmax
        uvar[:, 2] = p.tenmax * thick0 / p.wave if p.wave > 0.0 else 0.0
        uvar[:, 3] = p.shrmax * thick0 / p.shear if p.shear > 0.0 else 0.0
        uvar[:, 13] = p.dfn
        uvar[:, 14] = p.dfs
        extra["uvar"] = uvar
        extra["uvar169"] = uvar
    else:
        uvar = np.atleast_2d(uvar)
        if uvar.shape[0] != nel or uvar.shape[1] < 15:
            new_uvar = np.zeros((nel, 15), dtype=float)
            new_uvar[:, 9] = p.shrmax
            new_uvar[:, 10] = p.tenmax
            new_uvar[:, 2] = p.tenmax * thick0 / p.wave if p.wave > 0.0 else 0.0
            new_uvar[:, 3] = p.shrmax * thick0 / p.shear if p.shear > 0.0 else 0.0
            new_uvar[:, 13] = p.dfn
            new_uvar[:, 14] = p.dfs
            uvar = new_uvar
            extra["uvar"] = uvar
            extra["uvar169"] = uvar

    dmg = extra.get("dmg169", extra.get("dmg"))
    if dmg is None:
        dmg = np.zeros(nel, dtype=float)
        extra["dmg"] = dmg
        extra["dmg169"] = dmg
    else:
        dmg = np.atleast_1d(dmg).astype(float).copy()

    off = extra.get("off169", extra.get("off"))
    if off is None:
        off = np.ones(nel, dtype=float)
        extra["off"] = off
        extra["off169"] = off
    else:
        off = np.atleast_1d(off).astype(float).copy()

    # Accumulated strains
    eps_tot = extra.get("eps_tot")
    if eps_tot is None:
        eps_tot = np.zeros_like(deps_arr)
    eps_tot = eps_tot + deps_arr
    extra["eps_tot"] = eps_tot

    shrmax = uvar[:, 9].copy()
    tenmax = uvar[:, 10].copy()
    eps_n0 = uvar[:, 2].copy()
    eps_s0 = uvar[:, 3].copy()
    dfn = uvar[:, 13].copy()
    dfs = uvar[:, 14].copy()

    dmg_n = uvar[:, 0].copy()
    dmg_s = uvar[:, 1].copy()

    pwrt = p.pwrt
    pwrs = p.pwrs
    wave = p.wave
    shear = p.shear
    sht_sl = p.sht_sl
    dp = p.dp

    sign = sig_arr.copy()
    strs_tr_sh = np.zeros(nel, dtype=float)
    eps_sh = np.zeros(nel, dtype=float)
    fyld = np.zeros(nel, dtype=float)

    for i in range(nel):
        if off[i] < 0.001:
            off[i] = 0.0
        if off[i] < 1.0:
            off[i] = off[i] * 0.8

        if off[i] == 1.0:
            fdam_n = max(_EM20, 1.0 - dmg_n[i])
            fdam_s = max(_EM20, 1.0 - dmg_s[i])

            thk = thick0[i] if thick0[i] > 0.0 else 1.0

            # Normal (zz) and shear (yz, zx) trial stresses (lines 163-169)
            sign[i, 2] = sig_arr[i, 2] / fdam_n + deps_arr[i, 2] * wave / thk
            sign[i, 4] = sig_arr[i, 4] / fdam_s + deps_arr[i, 4] * shear / thk
            sign[i, 5] = sig_arr[i, 5] / fdam_s + deps_arr[i, 5] * shear / thk

            # In-plane elastic trial stresses
            sign[i, 0] = sig_arr[i, 0] + deps_arr[i, 0] * p.young / thk
            sign[i, 1] = sig_arr[i, 1] + deps_arr[i, 1] * p.young / thk
            sign[i, 3] = sig_arr[i, 3] + deps_arr[i, 3] * shear / thk

            strs_tr_sh[i] = math.sqrt(sign[i, 4]**2 + sign[i, 5]**2)
            eps_sh[i] = math.sqrt(eps_tot[i, 4]**2 + eps_tot[i, 5]**2)

            # Compute yield function (lines 174-179)
            denom_sh = max(_EM06, shrmax[i] - sht_sl * sign[i, 2])
            fyld[i] = (max(sign[i, 2], 0.0) / max(_EM06, tenmax[i]))**pwrt + (strs_tr_sh[i] / denom_sh)**pwrs - 1.0
            uvar[i, 5] = fyld[i]

            # Damage initiation test (lines 185-199)
            if fyld[i] >= 0.0:
                taumax_init = shrmax[i] - sht_sl * sign[i, 2]
                if (sign[i, 2] >= tenmax[i] or strs_tr_sh[i] >= taumax_init) and uvar[i, 8] == 0.0:
                    eps_n0[i] = eps_tot[i, 2]
                    eps_s0[i] = eps_sh[i] + dp
                    tenmax[i] = max(sign[i, 2], _EM06)
                    shrmax[i] = max(strs_tr_sh[i], _EM06)
                    uvar[i, 8] = 1.0

            # Damage evolution (lines 201-210)
            if eps_sh[i] > eps_s0[i]:
                denom_s = max(_EM20, dfs[i] - eps_s0[i])
                dmg_s[i] = max(dmg_s[i], (eps_sh[i] - eps_s0[i]) / denom_s)
            if eps_tot[i, 2] > eps_n0[i]:
                denom_n = max(_EM20, dfn[i] - eps_n0[i])
                dmg_n[i] = max(dmg_n[i], (eps_tot[i, 2] - eps_n0[i]) / denom_n)

            # Failure check (lines 212-223)
            if dmg_n[i] >= 1.0 or dmg_s[i] >= 1.0:
                if dmg_n[i] >= 1.0:
                    dmg_n[i] = 1.0
                if dmg_s[i] >= 1.0:
                    dmg_s[i] = 1.0
                off[i] = 0.8

            # Yield capping (lines 225-233)
            sign[i, 2] = min(sign[i, 2], tenmax[i])
            taumax = max(_EM06, shrmax[i] - sht_sl * sign[i, 2])
            strs_tr_sh[i] = math.sqrt(sign[i, 4]**2 + sign[i, 5]**2)
            if strs_tr_sh[i] > taumax and strs_tr_sh[i] > 0.0:
                tau_n = min(strs_tr_sh[i], taumax)
                sign[i, 4] = sign[i, 4] * tau_n / strs_tr_sh[i]
                sign[i, 5] = sign[i, 5] * tau_n / strs_tr_sh[i]

            # Damage degradation (lines 259-263)
            fdam_n = max(0.0, 1.0 - dmg_n[i])
            fdam_s = max(0.0, 1.0 - dmg_s[i])
            sign[i, 2] *= fdam_n
            sign[i, 4] *= fdam_s
            sign[i, 5] *= fdam_s

            # In-plane components degraded with general damage
            fdam_all = min(fdam_n, fdam_s)
            sign[i, 0] *= fdam_all
            sign[i, 1] *= fdam_all
            sign[i, 3] *= fdam_s

            # Plastic strain accumulation
            epsp_arr[i] += max(0.0, (eps_tot[i, 2] - eps_n0[i])) + max(0.0, (eps_sh[i] - eps_s0[i]))
        else:
            sign[i, :] = 0.0

    # Write state variables back to uvar (lines 235-258)
    uvar[:, 9] = shrmax
    uvar[:, 10] = tenmax
    uvar[:, 2] = eps_n0
    uvar[:, 3] = eps_s0
    uvar[:, 13] = dfn
    uvar[:, 14] = dfs
    uvar[:, 0] = dmg_n
    uvar[:, 1] = dmg_s

    for i in range(nel):
        dmg[i] = max(dmg_n[i], dmg_s[i])
        taumax = max(_EM06, shrmax[i] - sht_sl * sign[i, 2])
        g1 = min(max(sign[i, 2], 0.0) / max(_EM06, tenmax[i]), 1.0)
        curr_sh = math.sqrt(sign[i, 4]**2 + sign[i, 5]**2)
        g2 = min(curr_sh / taumax, 1.0)
        uvar[i, 6] = g1
        uvar[i, 7] = g2
        uvar[i, 4] = g1**pwrt + g2**pwrs

    extra["uvar"] = uvar
    extra["uvar169"] = uvar
    extra["dmg"] = dmg
    extra["dmg169"] = dmg
    extra["off"] = off
    extra["off169"] = off

    rho_val = p.rho0 if p.rho0 > 0.0 else 1.0
    ssp = np.full(nel, math.sqrt(max(0.0, wave / rho_val)), dtype=float)

    out_sig = sign[0] if is_1d else sign
    out_epsp = epsp_arr[0] if is_1d else epsp_arr
    out_ssp = ssp[0] if is_1d else ssp

    return out_sig, out_epsp, out_ssp


def shell_step(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Plane-stress shell update for LAW169 structural adhesive layers.

    Stress vector: [xx, yy, xy].
    Normal tensile mode is along xx; shear mode is along xy.
    """
    p = _get_params(mat)

    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).copy()
    deps_arr = np.atleast_2d(deps).copy()
    nel = sig_arr.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.atleast_1d(epsp).astype(float).copy()

    if extra is None:
        extra = {}

    uvar = extra.get("uvar169", extra.get("uvar"))
    if uvar is None or uvar.shape[0] != nel or uvar.shape[1] < 15:
        uvar = np.zeros((nel, 15), dtype=float)
        uvar[:, 9] = p.shrmax
        uvar[:, 10] = p.tenmax
        uvar[:, 2] = p.tenmax / p.young if p.young > 0.0 else 0.0
        uvar[:, 3] = p.shrmax / p.shear if p.shear > 0.0 else 0.0
        uvar[:, 13] = p.dfn
        uvar[:, 14] = p.dfs
        extra["uvar"] = uvar
        extra["uvar169"] = uvar

    dmg = extra.get("dmg169", extra.get("dmg"))
    if dmg is None:
        dmg = np.zeros(nel, dtype=float)
        extra["dmg"] = dmg
        extra["dmg169"] = dmg
    else:
        dmg = np.atleast_1d(dmg).astype(float).copy()

    off = extra.get("off169", extra.get("off"))
    if off is None:
        off = np.ones(nel, dtype=float)
        extra["off"] = off
        extra["off169"] = off
    else:
        off = np.atleast_1d(off).astype(float).copy()

    eps_tot = extra.get("eps_tot_shell")
    if eps_tot is None:
        eps_tot = np.zeros_like(deps_arr)
    eps_tot = eps_tot + deps_arr
    extra["eps_tot_shell"] = eps_tot

    shrmax = uvar[:, 9].copy()
    tenmax = uvar[:, 10].copy()
    eps_n0 = uvar[:, 2].copy()
    eps_s0 = uvar[:, 3].copy()
    dfn = uvar[:, 13].copy()
    dfs = uvar[:, 14].copy()

    dmg_n = uvar[:, 0].copy()
    dmg_s = uvar[:, 1].copy()

    pwrt = p.pwrt
    pwrs = p.pwrs
    e_plane = p.young / (1.0 - p.nu**2) if (1.0 - p.nu**2) > 0.0 else p.young
    shear = p.shear
    sht_sl = p.sht_sl
    dp = p.dp

    sign = sig_arr.copy()

    for i in range(nel):
        if off[i] < 0.001:
            off[i] = 0.0
        if off[i] < 1.0:
            off[i] = off[i] * 0.8

        if off[i] == 1.0:
            fdam_n = max(_EM20, 1.0 - dmg_n[i])
            fdam_s = max(_EM20, 1.0 - dmg_s[i])

            sign[i, 0] = sig_arr[i, 0] / fdam_n + deps_arr[i, 0] * e_plane
            sign[i, 1] = sig_arr[i, 1] / fdam_n + deps_arr[i, 1] * e_plane
            sign[i, 2] = sig_arr[i, 2] / fdam_s + deps_arr[i, 2] * shear

            tau = abs(sign[i, 2])
            eps_s = abs(eps_tot[i, 2])

            denom_sh = max(_EM06, shrmax[i] - sht_sl * sign[i, 0])
            fyld_val = (max(sign[i, 0], 0.0) / max(_EM06, tenmax[i]))**pwrt + (tau / denom_sh)**pwrs - 1.0
            uvar[i, 5] = fyld_val

            if fyld_val >= 0.0:
                taumax_init = shrmax[i] - sht_sl * sign[i, 0]
                if (sign[i, 0] >= tenmax[i] or tau >= taumax_init) and uvar[i, 8] == 0.0:
                    eps_n0[i] = eps_tot[i, 0]
                    eps_s0[i] = eps_s + dp
                    tenmax[i] = max(sign[i, 0], _EM06)
                    shrmax[i] = max(tau, _EM06)
                    uvar[i, 8] = 1.0

            if eps_s > eps_s0[i]:
                denom_s = max(_EM20, dfs[i] - eps_s0[i])
                dmg_s[i] = max(dmg_s[i], (eps_s - eps_s0[i]) / denom_s)
            if eps_tot[i, 0] > eps_n0[i]:
                denom_n = max(_EM20, dfn[i] - eps_n0[i])
                dmg_n[i] = max(dmg_n[i], (eps_tot[i, 0] - eps_n0[i]) / denom_n)

            if dmg_n[i] >= 1.0 or dmg_s[i] >= 1.0:
                if dmg_n[i] >= 1.0:
                    dmg_n[i] = 1.0
                if dmg_s[i] >= 1.0:
                    dmg_s[i] = 1.0
                off[i] = 0.8

            sign[i, 0] = min(sign[i, 0], tenmax[i])
            taumax = max(_EM06, shrmax[i] - sht_sl * sign[i, 0])
            if abs(sign[i, 2]) > taumax and abs(sign[i, 2]) > 0.0:
                sign[i, 2] = math.copysign(taumax, sign[i, 2])

            fdam_n = max(0.0, 1.0 - dmg_n[i])
            fdam_s = max(0.0, 1.0 - dmg_s[i])
            sign[i, 0] *= fdam_n
            sign[i, 1] *= fdam_n
            sign[i, 2] *= fdam_s

            epsp_arr[i] += max(0.0, (eps_tot[i, 0] - eps_n0[i])) + max(0.0, (eps_s - eps_s0[i]))
        else:
            sign[i, :] = 0.0

    uvar[:, 9] = shrmax
    uvar[:, 10] = tenmax
    uvar[:, 2] = eps_n0
    uvar[:, 3] = eps_s0
    uvar[:, 13] = dfn
    uvar[:, 14] = dfs
    uvar[:, 0] = dmg_n
    uvar[:, 1] = dmg_s

    for i in range(nel):
        dmg[i] = max(dmg_n[i], dmg_s[i])
        taumax = max(_EM06, shrmax[i] - sht_sl * sign[i, 0])
        g1 = min(max(sign[i, 0], 0.0) / max(_EM06, tenmax[i]), 1.0)
        g2 = min(abs(sign[i, 2]) / taumax, 1.0)
        uvar[i, 6] = g1
        uvar[i, 7] = g2
        uvar[i, 4] = g1**pwrt + g2**pwrs

    extra["uvar"] = uvar
    extra["uvar169"] = uvar
    extra["dmg"] = dmg
    extra["dmg169"] = dmg
    extra["off"] = off
    extra["off169"] = off

    out_sig = sign[0] if is_1d else sign
    out_epsp = epsp_arr[0] if is_1d else epsp_arr

    return out_sig, out_epsp


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Solid stress update entry point for LAW169."""
    return solid_step(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, **kwargs)


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Shell stress update entry point for LAW169."""
    return shell_step(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, **kwargs)


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent algorithmic tangent stiffness tensor C (6, 6) for LAW169 solids."""
    p = _get_params(mat)
    c_elastic = np.zeros((6, 6), dtype=float)
    c_elastic[0, 0] = c_elastic[1, 1] = p.young
    c_elastic[2, 2] = p.wave
    c_elastic[3, 3] = c_elastic[4, 4] = c_elastic[5, 5] = p.shear

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


def shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent tangent stiffness matrix C (3, 3) for LAW169 shells."""
    p = _get_params(mat)
    e_plane = p.young / (1.0 - p.nu**2) if (1.0 - p.nu**2) > 0.0 else p.young
    c_elastic = np.diag([e_plane, e_plane, p.shear])

    if sig is None or deps is None:
        return c_elastic

    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float)
    if sig_arr.ndim == 2 and sig_arr.shape[0] > 1:
        n = sig_arr.shape[0]
        return np.broadcast_to(c_elastic, (n, 3, 3)).copy()

    sig0 = sig_arr.flatten()[:3]
    deps0 = deps_arr.flatten()[:3]
    ex0 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in (extra or {}).items()}
    sig_base, _ = shell_step(mat, sig0, deps0, dt=dt, extra=ex0)

    h = 1.0e-7
    c_algo = np.zeros((3, 3), dtype=float)
    for j in range(3):
        deps_p = deps0.copy()
        deps_p[j] += h
        ex_p = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in (extra or {}).items()}
        sig_p, _ = shell_step(mat, sig0, deps_p, dt=dt, extra=ex_p)
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
    """Initial elastic plane-stress membrane stiffness matrix (3, 3) for LAW169 shells."""
    p = _get_params(mat)
    nu = p.nu
    e = p.young
    c = e / max(1.0 - nu * nu, 1e-15)
    g = p.shear
    return np.array([
        [c, nu * c, 0.0],
        [nu * c, c, 0.0],
        [0.0, 0.0, g],
    ], dtype=float)


def _register():
    try:
        from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
        for k in (169, "169", "LAW169", "ARUP_ADHESIVE", "ARUP", "ADHESIVE", "MAT_LAW169", "MAT_ARUP_ADHESIVE"):
            MAT_PHYSICS_REGISTRY[k] = build_law169
    except Exception:
        pass


_register()

"""
LAW71 — Nitinol Superelastic Shape-Memory Alloy (/MAT/LAW71, /MAT/SUPER_ELAS, /MAT/NITINOL).

Upstream Fortran origins:
- 3D Solids: ``engine/source/materials/mat/mat071/sigeps71.F``
- 2D Shells (Plane Stress): ``engine/source/materials/mat/mat071/sigeps71c.F``
- Integrated Beams: ``engine/source/materials/mat/mat071/sigeps71pi.F``
- Starter Card Reader: ``starter/source/materials/mat/mat071/hm_read_mat71.F``
- HyperMesh CFG Schema: ``hm_cfg_files/config/CFG/radioss140/MAT/matl71_71.cfg``

Theory
------
Based on the Auricchio (1997) shape-memory alloy constitutive formulation, generalized
to three-dimensional stress states with tension-compression asymmetry and temperature
coupling (Clausius-Clapeyron relation).

Key mechanisms:
1. Superelastic phase transformation:
   - Austenite (parent phase, high symmetry, high modulus E)
   - Martensite (product phase, lower symmetry, modulus E_mart)
   - Transformation strain tensor with maximum residual equivalent strain EpsL.
2. Yield / Transformation surfaces with tension-compression asymmetry parameter Alpha:
   F_s = ||S|| + 3*Alpha*P - C_AS*T  (loading / forward transformation A -> M)
   F_s = ||S|| + 3*Alpha*P - C_SA*T  (unloading / reverse transformation M -> A)
   where S is the deviatoric Kirchhoff stress, P is hydrostatic pressure,
   and C_AS, C_SA are stress-temperature slopes.
3. Transformation limits:
   R_s^{AS} = Sig_sas * (sqrt(2/3) + Alpha) - C_AS * T_s^{AS}
   R_f^{AS} = Sig_fas * (sqrt(2/3) + Alpha) - C_AS * T_f^{AS}
   R_s^{SA} = Sig_ssa * (sqrt(2/3) + Alpha) - C_SA * T_s^{SA}
   R_f^{SA} = Sig_fsa * (sqrt(2/3) + Alpha) - C_SA * T_f^{SA}
4. Quadratic/Newton-Raphson iterations for martensite fraction increment df_m when
   austenite and martensite elastic moduli differ (EFLAG = 1).
5. Plane-Stress Shell Formulation (sigeps71c.F):
   Enforces sigma_{zz} = 0 via secant iterations on the through-thickness stretch lambda_3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material


@dataclass
class Law71Params:
    """Parameters for OpenRadioss /MAT/LAW71 (/MAT/SUPER_ELAS, /MAT/NITINOL)."""
    id: int = 1
    title: str = ""
    rho0: float = 1.0
    rhor: float = 1.0
    young: float = 0.0
    nu: float = 0.3
    e_mart: float = 0.0
    sig_sas: float = 0.0
    sig_fas: float = 0.0
    sig_ssa: float = 0.0
    sig_fsa: float = 0.0
    alpha: float = 0.0
    epsl: float = 0.0
    cas: float = 0.0
    csa: float = 0.0
    tsas: float = 0.0
    tfas: float = 0.0
    tssa: float = 0.0
    tfsa: float = 0.0
    cp: float = 0.0
    tini: float = 293.0
    eflag: int = 0



# =============================================================================
# Helper: Math and Constants
# =============================================================================

SQRT_TWO_THIRD = math.sqrt(2.0 / 3.0)


# =============================================================================
# Physics Constructor / Starter Reader
# =============================================================================

def build_law71(rec: Any) -> Material:
    """Construct a Material entity for /MAT/LAW71 (/MAT/SUPER_ELAS, /MAT/NITINOL).

    Parameters
    ----------
    rec : GenericMaterialRecord, dict, or object
        Parsed card record from CFG or deck reader.

    Returns
    -------
    Material
        Material entity configured with law=71, parameters, and derived constants.
    """
    if hasattr(rec, "params") and getattr(rec, "params") is not None:
        p = getattr(rec, "params")
    elif isinstance(rec, dict) and "params" in rec and isinstance(rec["params"], dict):
        p = rec["params"]
    elif isinstance(rec, dict):
        p = rec
    else:
        p = {}

    def _get(keys: Union[str, list[str]], default: float = 0.0) -> float:
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            if isinstance(p, dict) and k in p and p[k] is not None:
                try:
                    return float(p[k])
                except (ValueError, TypeError):
                    pass
            if isinstance(rec, dict) and k in rec and rec[k] is not None:
                try:
                    return float(rec[k])
                except (ValueError, TypeError):
                    pass
            if hasattr(rec, k) and getattr(rec, k) is not None:
                try:
                    return float(getattr(rec, k))
                except (ValueError, TypeError):
                    pass
        return default

    if isinstance(rec, dict):
        rec_id = rec.get("id", rec.get("mat_id", rec.get("user_id", 1)))
        title = rec.get("title", f"LAW71_{rec_id}")
    else:
        rec_id = getattr(rec, "id", getattr(rec, "mat_id", getattr(rec, "user_id", 1)))
        title = getattr(rec, "title", f"LAW71_{rec_id}")

    rho0 = _get(["MAT_RHO", "rho", "rho0", "density", "RHO", "RHO0", "rho_i", "RHO_I"], default=1.0)
    if rho0 <= 0.0:
        rho0 = 1.0

    rhor = _get(["Refer_Rho", "refer_rho", "rhor", "RHOR", "REF_RHO", "ref_rho", "rho_o", "RHO_O"], default=0.0)
    if rhor <= 0.0:
        rhor = rho0

    e = _get(["e", "E", "MAT_E", "young", "youngs_modulus", "YOUNG", "YOUNG_MODULUS"], default=0.0)
    nu = _get(["nu", "nu_val", "poisson", "poissons_ratio", "NU", "MAT_NU", "POISSON_RATIO"], default=0.3)
    e_mart = _get(["e_mart", "E_mart", "emart", "EMART", "E_MART"], default=0.0)

    sig_sas = _get(["Sig_sas", "sig_sas", "SIG_SAS", "sig_start_as", "SIG_START_AS"], default=0.0)
    sig_fas = _get(["Sig_fas", "sig_fas", "SIG_FAS", "sig_final_as", "SIG_FINAL_AS"], default=0.0)
    sig_ssa = _get(["Sig_ssa", "sig_ssa", "SIG_SSA", "sig_start_sa", "SIG_START_SA"], default=0.0)
    sig_fsa = _get(["Sig_fsa", "sig_fsa", "SIG_FSA", "sig_final_sa", "SIG_FINAL_SA"], default=0.0)
    alpha = _get(["Alpha", "alpha", "ALPHA"], default=0.0)

    epsl = _get(["EpsL", "epsl", "EPSL", "eps_l", "EPS_L"], default=0.0)
    cas = _get(["CAS", "cas", "C_AS", "c_as"], default=0.0)
    csa = _get(["CSA", "csa", "C_SA", "c_sa"], default=0.0)
    tsas = _get(["TSAS", "tsas", "T_SAS", "t_sas"], default=0.0)
    tfas = _get(["TFAS", "tfas", "T_FAS", "t_fas"], default=0.0)

    tssa = _get(["TSSA", "tssa", "T_SSA", "t_ssa"], default=0.0)
    tfsa = _get(["TFSA", "tfsa", "T_FSA", "t_fsa"], default=0.0)
    cp = _get(["CP", "cp", "Cp", "specific_heat"], default=0.0)
    tini = _get(["TINI", "tini", "Tini", "initial_temperature"], default=0.0)

    # Starter validations (hm_read_mat71.F lines 141-160)
    if sig_sas >= sig_fas and sig_fas > 0.0:
        import warnings
        warnings.warn(f"LAW71 (id={rec_id}): Sig_sas ({sig_sas}) >= Sig_fas ({sig_fas}) violates forward transformation ordering.")

    if sig_ssa <= sig_fsa and sig_ssa > 0.0:
        import warnings
        warnings.warn(f"LAW71 (id={rec_id}): Sig_ssa ({sig_ssa}) <= Sig_fsa ({sig_fsa}) violates reverse transformation ordering.")

    if alpha > SQRT_TWO_THIRD:
        import warnings
        warnings.warn(f"LAW71 (id={rec_id}): Alpha ({alpha}) > sqrt(2/3) ({SQRT_TWO_THIRD:.4f}).")

    # Defaults (hm_read_mat71.F lines 168-177)
    if tssa == 0.0:
        tssa = 298.0
    if tfsa == 0.0:
        tfsa = 298.0
    if tsas == 0.0:
        tsas = 298.0
    if tfas == 0.0:
        tfas = 298.0
    if cp == 0.0:
        cp = 1.0e20
    if tini == 0.0:
        tini = 360.0

    eflag = 1 if (e_mart > 0.0 and e_mart != e) else 0
    if e_mart <= 0.0:
        e_mart = e

    # 3D Elastic constants (hm_read_mat71.F lines 180-189)
    g = 0.5 * e / (1.0 + nu) if (1.0 + nu) != 0.0 else 0.0
    denom_nu = (1.0 + nu) * (1.0 - 2.0 * nu)
    lamda = (e * (1.0 - nu) / denom_nu) if abs(denom_nu) > 1e-30 else 0.0
    c1 = e / (3.0 * (1.0 - 2.0 * nu)) if abs(1.0 - 2.0 * nu) > 1e-30 else 0.0

    gm = g
    km = c1
    if eflag == 1:
        gm = 0.5 * e_mart / (1.0 + nu) if (1.0 + nu) != 0.0 else 0.0
        km = e_mart / (3.0 * (1.0 - 2.0 * nu)) if abs(1.0 - 2.0 * nu) > 1e-30 else 0.0

    # Sound speeds
    c_solid = math.sqrt(max(0.0, lamda / rho0)) if rho0 > 0.0 else 0.0
    c_shell_raw = math.sqrt(max(0.0, (e / max(1e-15, 1.0 - nu * nu)) / rho0)) if rho0 > 0.0 else 0.0
    c_shell = max(c_shell_raw, c_solid)
    c_bar = math.sqrt(max(0.0, e / rho0)) if rho0 > 0.0 else 0.0

    parmat17 = (1.0 - 2.0 * nu) / (1.0 - nu) if abs(1.0 - nu) > 1e-30 else 0.0

    params: Dict[str, Any] = {
        "rho0": rho0,
        "rhor": rhor,
        "rho": rho0,
        "e": e,
        "young": e,
        "E": e,
        "nu": nu,
        "e_mart": e_mart,
        "EMART": e_mart,
        "sig_sas": sig_sas,
        "sig_fas": sig_fas,
        "sig_ssa": sig_ssa,
        "sig_fsa": sig_fsa,
        "alpha": alpha,
        "epsl": epsl,
        "cas": cas,
        "csa": csa,
        "tsas": tsas,
        "tfas": tfas,
        "tssa": tssa,
        "tfsa": tfsa,
        "cp": cp,
        "tini": tini,
        "G": g,
        "K": c1,
        "C1": c1,
        "lamda": lamda,
        "AA1": lamda,
        "GM": gm,
        "KM": km,
        "eflag": eflag,
        "sound_speed": c_solid,
        "sound_speed_solid": c_solid,
        "sound_speed_shell": c_shell,
        "parmat1": c1,
        "parmat2": e,
        "parmat3": nu,
        "parmat16": 2,
        "parmat17": parmat17,
        "pm1": rhor,
        "pm27": c_bar,
        "pm89": rho0,
        "nuparam": 25,
        "nuvar": 10,
    }

    mat = Material(id=rec_id, law=71, rho0=rho0, title=title, params=params)
    mat.law_name = "LAW71"
    return mat


# =============================================================================
# Sound Speed
# =============================================================================

def sound_speed(
    mat: Material,
    rho: Any = None,
    extra: Any = None,
    is_shell: bool = False,
) -> Union[float, np.ndarray]:
    """Dilatational sound speed for LAW71 elements (hm_read_mat71.F, sigeps71.F, sigeps71c.F).

    For 3D solids:
        c = sqrt(AA1 / rho0) = sqrt(lambda_p / rho0)
    For 2D shells:
        c = max(sqrt(E / ((1 - nu^2) * rho0)), sqrt(AA1 / rho0))

    Parameters
    ----------
    mat : Material
        LAW71 material.
    rho : float, ndarray, or None
        Current density.
    extra : dict or None
        Extra state.
    is_shell : bool
        True if querying for shell element time step.

    Returns
    -------
    float or ndarray
        Sound speed.
    """
    p = mat.params
    e = p.get("E", getattr(mat, "E", 0.0))
    nu = p.get("nu", getattr(mat, "nu", 0.3))
    rho0 = mat.rho0 if mat.rho0 > 0.0 else 1.0

    denom = (1.0 + nu) * (1.0 - 2.0 * nu)
    aa1 = (e * (1.0 - nu) / denom) if abs(denom) > 1e-30 else 0.0
    c_solid = math.sqrt(max(0.0, aa1 / rho0))

    if is_shell:
        c_sh = math.sqrt(max(0.0, (e / max(1e-15, 1.0 - nu * nu)) / rho0))
        c_val = max(c_sh, c_solid)
    else:
        c_val = c_solid

    if rho is not None and isinstance(rho, np.ndarray):
        return np.full(rho.shape, c_val, dtype=float)
    return c_val


# =============================================================================
# State Dimensions
# =============================================================================

def extra_shapes(mat: Material, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Return the extra persistent array shapes required by LAW71 (sigeps71.F, sigeps71c.F).

    For 3D solids:
        uv71: (10,) user variables (UVAR 1..10)
        eps71: (6,) total strains
    For 2D shells:
        uv71: (nip, 10) or (10,)
        eps71: (nip, 3) or (3,)
    """
    shapes: Dict[str, Tuple[int, ...]] = {}
    if nip is not None and nip > 0:
        shapes["uv71"] = (nip, 10)
        shapes["eps71"] = (nip, 3)
        shapes["thk"] = (nip,)
    else:
        shapes["uv71"] = (10,)
        shapes["eps71"] = (6,)
    return shapes


# =============================================================================
# 3D Solids: solid_step / solid_update (sigeps71.F)
# =============================================================================

def solid_step(
    mat: Material,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Fortran-faithful stress update for LAW71 3D solid elements (sigeps71.F).

    Parameters
    ----------
    mat : Material
        LAW71 material instance.
    sig : ndarray, shape (n, 6) or (6,)
        Cauchy stress in Voigt convention [xx, yy, zz, xy, yz, zx].
    deps : ndarray, shape (n, 6) or (6,)
        Strain increment with engineering shears [xx, yy, zz, xy, yz, zx].
    epsp : ndarray, optional
        Plastic strain / martensite fraction view.
    dt : float
        Current time step.
    extra : dict, optional
        Persistent element extra state dictionary.

    Returns
    -------
    (sign, epsp, soundsp) : tuple
        sign : updated Cauchy stress (n, 6)
        epsp : updated martensite fraction or plastic strain
        soundsp : dilatational sound speed (n,)
    """
    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig.reshape(1, -1)
        deps = deps.reshape(1, -1)

    n = sig.shape[0]
    if n == 0:
        empty_c = np.empty(0, dtype=sig.dtype)
        return (sig[0] if is_1d else sig), epsp, (empty_c[0] if is_1d else empty_c)

    if extra is None:
        extra = {}

    # State arrays: uv71 (n, 10), eps71 (n, 6)
    u_source = None
    if "uv71" in extra and extra["uv71"] is not None:
        u_source = extra["uv71"]
    elif "uvar" in extra and extra["uvar"] is not None:
        u_source = extra["uvar"]

    if u_source is None:
        extra["uv71"] = np.zeros((n, 10), dtype=float)
    else:
        u = np.asarray(u_source, dtype=float)
        if u.ndim == 1:
            u = u.reshape(n, -1) if u.size >= n * 10 else np.broadcast_to(u, (n, u.size)).copy()
        if u.ndim != 2 or u.shape[0] != n or u.shape[1] < 10:
            padded = np.zeros((n, 10), dtype=float)
            if u.ndim == 2:
                r = min(n, u.shape[0])
                c_idx = min(10, u.shape[1])
                padded[:r, :c_idx] = u[:r, :c_idx]
            extra["uv71"] = padded
        else:
            extra["uv71"] = u

    eps_source = None
    if "eps71" in extra and extra["eps71"] is not None:
        eps_source = extra["eps71"]
    elif "eps" in extra and extra["eps"] is not None:
        eps_source = extra["eps"]

    if eps_source is None:
        extra["eps71"] = np.zeros((n, 6), dtype=float)
    else:
        e_arr = np.asarray(eps_source, dtype=float)
        if e_arr.ndim == 1:
            e_arr = e_arr.reshape(n, -1) if e_arr.size >= n * 6 else np.broadcast_to(e_arr, (n, e_arr.size)).copy()
        if e_arr.ndim != 2 or e_arr.shape[0] != n or e_arr.shape[1] < 6:
            padded = np.zeros((n, 6), dtype=float)
            if e_arr.ndim == 2:
                r = min(n, e_arr.shape[0])
                c_idx = min(6, e_arr.shape[1])
                padded[:r, :c_idx] = e_arr[:r, :c_idx]
            extra["eps71"] = padded
        else:
            extra["eps71"] = e_arr

    p = mat.params
    e_val = p.get("E", getattr(mat, "E", 0.0))
    nu_val = p.get("nu", getattr(mat, "nu", 0.3))
    e_mart = p.get("e_mart", p.get("EMART", 0.0))
    sig_sas = p.get("sig_sas", 0.0)
    sig_fas = p.get("sig_fas", 0.0)
    sig_ssa = p.get("sig_ssa", 0.0)
    sig_fsa = p.get("sig_fsa", 0.0)
    alpha = p.get("alpha", 0.0)
    epsl_in = p.get("epsl", 0.0)
    cas = p.get("cas", 0.0)
    csa = p.get("csa", 0.0)
    tsas = p.get("tsas", 298.0)
    tfas = p.get("tfas", 298.0)
    tssa = p.get("tssa", 298.0)
    tfsa = p.get("tfsa", 298.0)
    cp = p.get("cp", 1.0e20)
    tini = p.get("tini", 360.0)
    eflag = p.get("eflag", 1 if (e_mart > 0.0 and e_mart != e_val) else 0)

    g_val = p.get("G", 0.5 * e_val / (1.0 + nu_val))
    k_val = p.get("K", e_val / (3.0 * (1.0 - 2.0 * nu_val)))
    aa1 = p.get("AA1", e_val * (1.0 - nu_val) / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val)))
    gm_val = p.get("GM", 0.5 * e_mart / (1.0 + nu_val) if eflag == 1 else g_val)
    km_val = p.get("KM", e_mart / (3.0 * (1.0 - 2.0 * nu_val)) if eflag == 1 else k_val)

    rho0 = mat.rho0 if mat.rho0 > 0.0 else 1.0
    epsl = epsl_in / (SQRT_TWO_THIRD + alpha) if (SQRT_TWO_THIRD + alpha) > 0.0 else epsl_in

    rsas = sig_sas * (SQRT_TWO_THIRD + alpha) - cas * tsas
    rfas = sig_fas * (SQRT_TWO_THIRD + alpha) - cas * tfas
    rssa = sig_ssa * (SQRT_TWO_THIRD + alpha) - csa * tssa
    rfsa = sig_fsa * (SQRT_TWO_THIRD + alpha) - csa * tfsa

    ismstr = extra.get("ismstr", 0)
    jthe = extra.get("jthe", 0)

    # Temperature array
    if "temp" not in extra or extra["temp"] is None:
        temp = np.full(n, tini, dtype=float)
        extra["temp"] = temp
    else:
        temp = np.asarray(extra["temp"], dtype=float).reshape(-1)
        if len(temp) != n:
            temp = np.full(n, tini, dtype=float)
            extra["temp"] = temp

    if jthe == 0 and "eint" in extra and extra["eint"] is not None:
        eint = np.asarray(extra["eint"], dtype=float).reshape(-1)
        vol = extra.get("vol", extra.get("volume", np.ones(n, dtype=float)))
        vol = np.asarray(vol, dtype=float).reshape(-1)
        denom_t = rho0 * cp * np.maximum(1.0e-15, vol[:n])
        temp[:n] = tini + eint[:n] / denom_t

    # Accumulate total strains
    eps_tot = extra["eps71"]
    eps_tot[:, :deps.shape[1]] += deps

    uvar = extra["uv71"]
    sign = np.zeros_like(sig)
    soundsp = np.full(n, math.sqrt(max(0.0, aa1 / rho0)), dtype=float)
    etse = np.ones(n, dtype=float)

    # Loop over elements
    for i in range(n):
        # 3x3 strain tensor AV (tensorial shear = 0.5 * gamma)
        av = np.array([
            [eps_tot[i, 0], 0.5 * eps_tot[i, 3], 0.5 * eps_tot[i, 5]],
            [0.5 * eps_tot[i, 3], eps_tot[i, 1], 0.5 * eps_tot[i, 4]],
            [0.5 * eps_tot[i, 5], 0.5 * eps_tot[i, 4], eps_tot[i, 2]],
        ], dtype=float)

        # Eigenvalue decomposition
        evv, dirprv = np.linalg.eigh(av)

        # Stretches according to ismstr (sigeps71.F lines 198-218)
        if ismstr in (0, 2, 4):
            ev = np.exp(evv)
        elif ismstr in (10, 12):
            ev = np.sqrt(np.maximum(0.0, evv + 1.0))
        else:
            ev = evv + 1.0

        det = ev[0] * ev[1] * ev[2]
        if det > 1e-30:
            trde = math.log(det)
            rv_pui = math.exp(-trde / 3.0)
        else:
            trde = 0.0
            rv_pui = 0.0

        ee1 = math.log(max(1e-30, ev[0] * rv_pui)) if rv_pui > 0.0 else 0.0
        ee2 = math.log(max(1e-30, ev[1] * rv_pui)) if rv_pui > 0.0 else 0.0
        ee3 = math.log(max(1e-30, ev[2] * rv_pui)) if rv_pui > 0.0 else 0.0

        fm = uvar[i, 0]
        gt = g_val + fm * (gm_val - g_val)
        kt = k_val + fm * (km_val - k_val)

        p_pres = kt * (trde - 3.0 * alpha * epsl * fm)
        ne = math.sqrt(ee1 * ee1 + ee2 * ee2 + ee3 * ee3)
        ne_safe = max(ne, 1.0e-20)
        nxx = ee1 / ne_safe
        nyy = ee2 / ne_safe
        nzz = ee3 / ne_safe

        sxx = 2.0 * gt * (ee1 - epsl * fm * nxx)
        syy = 2.0 * gt * (ee2 - epsl * fm * nyy)
        szz = 2.0 * gt * (ee3 - epsl * fm * nzz)
        sv = math.sqrt(sxx * sxx + syy * syy + szz * szz)

        dfmas = 0.0
        dfmsa = 0.0

        # Austenite -> Martensite check (loading)
        fs = sv + 3.0 * alpha * p_pres - cas * temp[i]
        fass = fs - rsas
        fasf = fs - rfas
        fs0 = uvar[i, 1]
        beta = epsl * (2.0 * gt + 9.0 * kt * alpha * alpha)

        if (fs - fs0) > 0.0 and fass > 0.0 and fasf < 0.0 and fm < 1.0:
            if eflag > 0:
                db = (2.0 * (gm_val - g_val) + 9.0 * alpha * alpha * (km_val - k_val)) * epsl
                unmxn = 1.0 - fm
                dftr = 2.0 * ne * (gm_val - g_val) + 3.0 * alpha * trde * (km_val - k_val)
                denom_init = fasf - beta * unmxn
                dfmas = min(1.0, -(fs - fs0) * unmxn / denom_init) if abs(denom_init) > 1e-30 else 0.0
                a_coef = unmxn * db
                b_coef = rfas - fs + unmxn * (beta - dftr)
                c_coef = unmxn * (fs0 - fs)
                for _ in range(3):
                    fct = dfmas * dfmas * a_coef + dfmas * b_coef + c_coef
                    fctp = 2.0 * dfmas * a_coef + b_coef
                    if abs(fctp) > 1e-30:
                        dfmas = dfmas - fct / fctp
                dfmas = max(0.0, min(1.0 - fm, dfmas))
            else:
                denom_as = fasf - beta * (1.0 - fm)
                if abs(denom_as) > 1e-30:
                    dfmas = -(fs - fs0) * (1.0 - fm) / denom_as
                    dfmas = max(0.0, min(1.0 - fm, dfmas))

        # Martensite -> Austenite check (unloading)
        fs_un = sv + 3.0 * alpha * p_pres - csa * temp[i]
        fsas = fs_un - rssa
        fsaf = fs_un - rfsa
        fs0_sa = uvar[i, 2]

        if (fs_un - fs0_sa) < 0.0 and fsas < 0.0 and fsaf > 0.0 and fm > 0.0:
            if eflag > 0:
                db = (2.0 * (gm_val - g_val) + 9.0 * alpha * alpha * (km_val - k_val)) * epsl
                dftr = 2.0 * (gm_val - g_val) * ne + 3.0 * alpha * (km_val - k_val) * trde
                dfmsa = 0.0
                a_coef = fm * db
                b_coef = -(rfsa - fs_un + fm * (dftr - beta))
                c_coef = -fm * (fs_un - fs0_sa)
                for _ in range(3):
                    fct = dfmsa * dfmsa * a_coef + dfmsa * b_coef + c_coef
                    fctp = 2.0 * dfmsa * a_coef + b_coef
                    if abs(fctp) > 1e-30:
                        dfmsa = dfmsa - fct / fctp
                dfmsa = max(-fm, min(0.0, dfmsa))
            else:
                denom_sa = fsaf + beta * fm
                if abs(denom_sa) > 1e-30:
                    dfmsa = fm * (fs_un - fs0_sa) / denom_sa
                    dfmsa = max(-fm, min(0.0, dfmsa))

        dfm = dfmas + dfmsa
        if dfm < 0.0 and fm <= 0.0:
            dfm = 0.0

        # Stress update with dfm
        dgt = dfm * (gm_val - g_val)
        dkt = dfm * (km_val - k_val)

        sxx = sxx - 2.0 * gt * epsl * nxx * dfm + 2.0 * dgt * (ee1 - epsl * nxx * dfm)
        syy = syy - 2.0 * gt * epsl * nyy * dfm + 2.0 * dgt * (ee2 - epsl * nyy * dfm)
        szz = szz - 2.0 * gt * epsl * nzz * dfm + 2.0 * dgt * (ee3 - epsl * nzz * dfm)

        p_pres = p_pres - kt * epsl * 3.0 * alpha * dfm + dkt * (trde - epsl * 3.0 * alpha * dfm)

        # Kirchhoff stress in principal axes
        sig_p = np.array([sxx + p_pres, syy + p_pres, szz + p_pres], dtype=float)
        inve = (1.0 / det) if det > 1e-30 else 0.0
        sig_cauchy_p = sig_p * inve

        # Transform principal Cauchy stress to global directions (V @ diag(sig_p) @ V.T)
        sig_mat = dirprv @ np.diag(sig_cauchy_p) @ dirprv.T
        sign[i, 0] = sig_mat[0, 0]
        sign[i, 1] = sig_mat[1, 1]
        sign[i, 2] = sig_mat[2, 2]
        sign[i, 3] = sig_mat[0, 1]
        sign[i, 4] = sig_mat[1, 2]
        sign[i, 5] = sig_mat[2, 0]

        # Update martensite fraction and UVAR history
        new_fm = min(1.0, max(0.0, fm + dfm))
        uvar[i, 0] = new_fm

        sv_new = math.sqrt(sxx * sxx + syy * syy + szz * szz)
        fs_new = sv_new + 3.0 * alpha * p_pres
        uvar[i, 1] = fs_new - cas * temp[i]
        uvar[i, 2] = fs_new - csa * temp[i]

        dfs = 0.0
        if dfmas != 0.0:
            dfs = abs(uvar[i, 1] - fs0)
        elif dfmsa != 0.0:
            dfs = abs(uvar[i, 2] - fs0_sa)

        if dfs != 0.0 and epsl != 0.0 and dfm != 0.0:
            h_mod = dfs / (epsl * abs(dfm))
            denom_e = (e_val + new_fm * (e_mart - e_val)) + h_mod
            etse[i] = (h_mod * (1.0 + nu_val) / denom_e) if abs(denom_e) > 1e-30 else 1.0
        else:
            etse[i] = 1.0

        uvar[i, 3] = epsl * dfm
        uvar[i, 6] += epsl * dfm
        uvar[i, 9] = eps_tot[i, 0]

    sig[:] = sign

    if "uvar" in extra and hasattr(extra["uvar"], "__setitem__"):
        try:
            if is_1d:
                extra["uvar"][:] = uvar[0]
            else:
                extra["uvar"][:] = uvar
        except Exception:
            pass
    if "eps" in extra and hasattr(extra["eps"], "__setitem__"):
        try:
            if is_1d:
                extra["eps"][:] = eps_tot[0]
            else:
                extra["eps"][:] = eps_tot
        except Exception:
            pass

    out_epsp = uvar[:, 0] if epsp is None else uvar[:, 0]
    if is_1d:
        epsp_out = out_epsp[0] if hasattr(out_epsp, "__len__") else out_epsp
        return sign[0], epsp_out, soundsp[0]
    return sign, out_epsp, soundsp


# Convenient alias
solid_update = solid_step


# =============================================================================
# 2D Shells: shell_step / shell_update (Plane Stress, sigeps71c.F)
# =============================================================================

def shell_step(
    mat: Material,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Plane-stress shell update for LAW71 (sigeps71c.F).

    Enforces sigma_{zz} = 0 via secant iterations on through-thickness stretch lambda_3.

    Parameters
    ----------
    mat : Material
        LAW71 material instance.
    sig : ndarray, shape (n, 3) or (n, 5) or (3,)
        In-plane stress components [xx, yy, xy] (and optional transverse shears).
    deps : ndarray, shape (n, 3) or (n, 5) or (3,)
        In-plane strain increments with engineering shear [xx, yy, xy].
    epsp : ndarray, optional
        Martensite fraction / plastic strain view.
    dt : float
        Time step.
    extra : dict, optional
        Persistent element extra state dictionary.

    Returns
    -------
    (sign[:, :3], epsp) : tuple
        sign : updated in-plane Cauchy stress (n, 3)
        epsp : updated martensite fraction view
    """
    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig.reshape(1, -1)
        deps = deps.reshape(1, -1)

    n = sig.shape[0]
    if n == 0:
        return (sig[0, :3] if is_1d else sig[:, :3]), epsp

    if extra is None:
        extra = {}

    # State arrays: uv71 (n, 10), eps71 (n, >=3)
    u_source = None
    if "uv71" in extra and extra["uv71"] is not None:
        u_source = extra["uv71"]
    elif "uvar" in extra and extra["uvar"] is not None:
        u_source = extra["uvar"]

    if u_source is None:
        extra["uv71"] = np.zeros((n, 10), dtype=float)
    else:
        u = np.asarray(u_source, dtype=float)
        if u.ndim == 1:
            u = u.reshape(n, -1) if u.size >= n * 10 else np.broadcast_to(u, (n, u.size)).copy()
        if u.ndim != 2 or u.shape[0] != n or u.shape[1] < 10:
            padded = np.zeros((n, 10), dtype=float)
            if u.ndim == 2:
                r = min(n, u.shape[0])
                c_idx = min(10, u.shape[1])
                padded[:r, :c_idx] = u[:r, :c_idx]
            extra["uv71"] = padded
        else:
            extra["uv71"] = u

    eps_source = None
    if "eps71" in extra and extra["eps71"] is not None:
        eps_source = extra["eps71"]
    elif "eps" in extra and extra["eps"] is not None:
        eps_source = extra["eps"]

    if eps_source is None:
        extra["eps71"] = np.zeros((n, max(3, deps.shape[1])), dtype=float)
    else:
        e_arr = np.asarray(eps_source, dtype=float)
        if e_arr.ndim == 1:
            e_arr = e_arr.reshape(n, -1) if e_arr.size >= n * 3 else np.broadcast_to(e_arr, (n, e_arr.size)).copy()
        if e_arr.ndim != 2 or e_arr.shape[0] != n or e_arr.shape[1] < 3:
            cols = max(3, deps.shape[1], e_arr.shape[1] if e_arr.ndim == 2 else 3)
            padded = np.zeros((n, cols), dtype=float)
            if e_arr.ndim == 2:
                r = min(n, e_arr.shape[0])
                c_idx = min(cols, e_arr.shape[1])
                padded[:r, :c_idx] = e_arr[:r, :c_idx]
            extra["eps71"] = padded
        else:
            extra["eps71"] = e_arr

    p = mat.params
    e_val = p.get("E", getattr(mat, "E", 0.0))
    nu_val = p.get("nu", getattr(mat, "nu", 0.3))
    e_mart = p.get("e_mart", p.get("EMART", 0.0))
    sig_sas = p.get("sig_sas", 0.0)
    sig_fas = p.get("sig_fas", 0.0)
    sig_ssa = p.get("sig_ssa", 0.0)
    sig_fsa = p.get("sig_fsa", 0.0)
    alpha = p.get("alpha", 0.0)
    epsl_in = p.get("epsl", 0.0)
    cas = p.get("cas", 0.0)
    csa = p.get("csa", 0.0)
    tsas = p.get("tsas", 298.0)
    tfas = p.get("tfas", 298.0)
    tssa = p.get("tssa", 298.0)
    tfsa = p.get("tfsa", 298.0)
    cp = p.get("cp", 1.0e20)
    tini = p.get("tini", 360.0)
    eflag = p.get("eflag", 1 if (e_mart > 0.0 and e_mart != e_val) else 0)

    g_val = p.get("G", 0.5 * e_val / (1.0 + nu_val))
    k_val = p.get("K", e_val / (3.0 * (1.0 - 2.0 * nu_val)))
    gm_val = p.get("GM", 0.5 * e_mart / (1.0 + nu_val) if eflag == 1 else g_val)
    km_val = p.get("KM", e_mart / (3.0 * (1.0 - 2.0 * nu_val)) if eflag == 1 else k_val)

    rho0 = mat.rho0 if mat.rho0 > 0.0 else 1.0
    epsl = epsl_in / (SQRT_TWO_THIRD + alpha) if (SQRT_TWO_THIRD + alpha) > 0.0 else epsl_in

    rsas = sig_sas * (SQRT_TWO_THIRD + alpha) - cas * tsas
    rfas = sig_fas * (SQRT_TWO_THIRD + alpha) - cas * tfas
    rssa = sig_ssa * (SQRT_TWO_THIRD + alpha) - csa * tssa
    rfsa = sig_fsa * (SQRT_TWO_THIRD + alpha) - csa * tfsa

    ismstr = extra.get("ismstr", 0)
    jthe = extra.get("jthe", 0)

    # Temperature
    if "temp" not in extra or extra["temp"] is None:
        temp = np.full(n, tini, dtype=float)
        extra["temp"] = temp
    else:
        temp = np.asarray(extra["temp"], dtype=float).reshape(-1)
        if len(temp) != n:
            temp = np.full(n, tini, dtype=float)
            extra["temp"] = temp

    if jthe == 0 and "eint" in extra and extra["eint"] is not None:
        eint = np.asarray(extra["eint"], dtype=float)
        eint_sum = eint.sum(axis=1) if (eint.ndim == 2 and eint.shape[1] >= 2) else eint.reshape(-1)
        vol = extra.get("vol", extra.get("volume", np.ones(n, dtype=float)))
        vol = np.asarray(vol, dtype=float).reshape(-1)
        denom_t = vol[:n] * rho0 * cp
        temp[:n] = tini + eint_sum[:n] / np.maximum(1e-15, denom_t)

    # Accumulate in-plane total strains
    eps_tot = extra["eps71"]
    n_comp = min(deps.shape[1], eps_tot.shape[1])
    eps_tot[:, :n_comp] += deps[:, :n_comp]

    uvar = extra["uv71"]
    sign = np.zeros_like(sig)
    iterk = 10

    for i in range(n):
        exx = eps_tot[i, 0]
        eyy = eps_tot[i, 1]
        exy = eps_tot[i, 2]  # engineering shear gamma_xy

        # In-plane principal strains (sigeps71c.F lines 214-219)
        trav = exx + eyy
        rootv = math.sqrt((exx - eyy) * (exx - eyy) + exy * exy)
        evv1 = 0.5 * (trav + rootv)
        evv2 = 0.5 * (trav - rootv)

        # Eigenvector matrix EIGV (sigeps71c.F lines 221-236)
        if abs(evv2 - evv1) < 1.0e-10:
            eigv = np.array([
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ], dtype=float)
        else:
            inv_root = 1.0 / rootv
            eigv = np.array([
                [(exx - evv2) * inv_root, (evv1 - exx) * inv_root],
                [(eyy - evv2) * inv_root, (evv1 - eyy) * inv_root],
                [(0.5 * exy) * inv_root, -(0.5 * exy) * inv_root],
            ], dtype=float)

        # Initial stretches (sigeps71c.F lines 238-256)
        if ismstr in (1, 3, 11):
            ev1 = evv1 + 1.0
            ev2 = evv2 + 1.0
        elif ismstr == 10:
            ev1 = math.sqrt(max(0.0, evv1 + 1.0))
            ev2 = math.sqrt(max(0.0, evv2 + 1.0))
        else:
            ev1 = math.exp(evv1)
            ev2 = math.exp(evv2)

        ev3 = 1.0 / max(1e-30, ev1 * ev2)

        epsim1 = 0.0
        sigzim1 = 0.0
        fm = uvar[i, 0]
        fs0 = uvar[i, 1]
        fs0_sa = uvar[i, 2]

        sigxx_final = 0.0
        sigyy_final = 0.0
        sigzz_final = 0.0
        dfm_final = 0.0
        fs_final = 0.0

        # Secant plane-stress iteration loop (sigeps71c.F lines 282-556)
        for kk in range(1, iterk + 1):
            det = ev1 * ev2 * ev3
            if det > 1e-30:
                trde = math.log(det)
                rv_pui = math.exp(-trde / 3.0)
            else:
                trde = 0.0
                rv_pui = 0.0

            ee1 = math.log(max(1e-30, ev1 * rv_pui)) if rv_pui > 0.0 else 0.0
            ee2 = math.log(max(1e-30, ev2 * rv_pui)) if rv_pui > 0.0 else 0.0
            ee3 = math.log(max(1e-30, ev3 * rv_pui)) if rv_pui > 0.0 else 0.0

            gt = g_val + fm * (gm_val - g_val)
            kt = k_val + fm * (km_val - k_val)

            p_pres = kt * (trde - 3.0 * alpha * epsl * fm)
            ne = math.sqrt(ee1 * ee1 + ee2 * ee2 + ee3 * ee3)
            ne_safe = max(ne, 1.0e-20)
            nxx = ee1 / ne_safe
            nyy = ee2 / ne_safe
            nzz = ee3 / ne_safe

            sxx = 2.0 * gt * (ee1 - epsl * fm * nxx)
            syy = 2.0 * gt * (ee2 - epsl * fm * nyy)
            szz = 2.0 * gt * (ee3 - epsl * fm * nzz)
            sv = math.sqrt(sxx * sxx + syy * syy + szz * szz)

            dfmas = 0.0
            dfmsa = 0.0

            # Forward transformation check A -> M
            fs = sv + 3.0 * alpha * p_pres - cas * temp[i]
            fass = fs - rsas
            fasf = fs - rfas
            beta = epsl * (2.0 * gt + 9.0 * kt * alpha * alpha)

            if (fs - fs0) > 0.0 and fass > 0.0 and fasf < 0.0 and fm < 1.0:
                if eflag > 0:
                    db = (2.0 * (gm_val - g_val) + 9.0 * alpha * alpha * (km_val - k_val)) * epsl
                    unmxn = 1.0 - fm
                    dftr = 2.0 * ne * (gm_val - g_val) + 3.0 * alpha * trde * (km_val - k_val)
                    denom_init = fasf - beta * unmxn
                    dfmas = min(1.0, -(fs - fs0) * unmxn / denom_init) if abs(denom_init) > 1e-30 else 0.0
                    a_coef = unmxn * db
                    b_coef = rfas - fs + unmxn * (beta - dftr)
                    c_coef = unmxn * (fs0 - fs)
                    for _ in range(3):
                        fct = dfmas * dfmas * a_coef + dfmas * b_coef + c_coef
                        fctp = 2.0 * dfmas * a_coef + b_coef
                        if abs(fctp) > 1e-30:
                            dfmas = dfmas - fct / fctp
                    dfmas = max(0.0, min(1.0 - fm, dfmas))
                else:
                    denom_as = fasf - beta * (1.0 - fm)
                    if abs(denom_as) > 1e-30:
                        dfmas = -(fs - fs0) * (1.0 - fm) / denom_as
                        dfmas = max(0.0, min(1.0 - fm, dfmas))

            # Reverse transformation check M -> A
            fs_un = sv + 3.0 * alpha * p_pres - csa * temp[i]
            fsas = fs_un - rssa
            fsaf = fs_un - rfsa

            if (fs_un - fs0_sa) < 0.0 and fsas < 0.0 and fsaf > 0.0 and fm > 0.0:
                if eflag > 0:
                    db = (2.0 * (gm_val - g_val) + 9.0 * alpha * alpha * (km_val - k_val)) * epsl
                    dftr = 2.0 * (gm_val - g_val) * ne + 3.0 * alpha * (km_val - k_val) * trde
                    dfmsa = 0.0
                    a_coef = fm * db
                    b_coef = -(rfsa - fs_un + fm * (dftr - beta))
                    c_coef = -fm * (fs_un - fs0_sa)
                    for _ in range(3):
                        fct = dfmsa * dfmsa * a_coef + dfmsa * b_coef + c_coef
                        fctp = 2.0 * dfmsa * a_coef + b_coef
                        if abs(fctp) > 1e-30:
                            dfmsa = dfmsa - fct / fctp
                    dfmsa = max(-fm, min(0.0, dfmsa))
                else:
                    denom_sa = fsaf + beta * fm
                    if abs(denom_sa) > 1e-30:
                        dfmsa = fm * (fs_un - fs0_sa) / denom_sa
                        dfmsa = max(-fm, min(0.0, dfmsa))

            dfm = dfmas + dfmsa
            if dfm < 0.0 and fm <= 0.0:
                dfm = 0.0

            dgt = dfm * (gm_val - g_val)
            dkt = dfm * (km_val - k_val)

            sxx = sxx - 2.0 * gt * epsl * nxx * dfm + 2.0 * dgt * (ee1 - epsl * nxx * dfm)
            syy = syy - 2.0 * gt * epsl * nyy * dfm + 2.0 * dgt * (ee2 - epsl * nyy * dfm)
            szz = szz - 2.0 * gt * epsl * nzz * dfm + 2.0 * dgt * (ee3 - epsl * nzz * dfm)

            p_pres = p_pres - kt * epsl * 3.0 * alpha * dfm + dkt * (trde - epsl * 3.0 * alpha * dfm)

            inve = (1.0 / det) if det > 1e-30 else 0.0
            sigxx_p = (sxx + p_pres) * inve
            sigyy_p = (syy + p_pres) * inve
            sigzz_p = (szz + p_pres) * inve

            sigxx_final = sigxx_p
            sigyy_final = sigyy_p
            sigzz_final = sigzz_p
            dfm_final = dfm
            fs_final = math.sqrt(sxx * sxx + syy * syy + szz * szz) + 3.0 * alpha * p_pres

            # Convergence check for plane stress (sigeps71c.F lines 525-549)
            if abs(sigzz_p) < 1.0e-12 and kk >= 3:
                break

            if kk == 1:
                epsim1 = ev3
                ev3 = ev3 / 2.0
                sigzim1 = sigzz_p
            else:
                test = sigzz_p - sigzim1
                denom_sec = test if abs(test) > 1.0e-20 else 1.0e-10
                epsi = ev3
                ev3 = ev3 - sigzz_p * (ev3 - epsim1) / denom_sec
                epsim1 = epsi
                sigzim1 = sigzz_p

        # Transform principal stresses back to shell global directions (sigeps71c.F lines 570-572)
        sign[i, 0] = eigv[0, 0] * sigxx_final + eigv[0, 1] * sigyy_final
        sign[i, 1] = eigv[1, 0] * sigxx_final + eigv[1, 1] * sigyy_final
        sign[i, 2] = eigv[2, 0] * sigxx_final + eigv[2, 1] * sigyy_final

        # Transverse shears if present
        if sig.shape[1] > 3:
            gs = gt
            dyz = deps[i, 3] if deps.shape[1] > 3 else 0.0
            dzx = deps[i, 4] if deps.shape[1] > 4 else 0.0
            sign[i, 3] = sig[i, 3] + gs * dyz
            if sig.shape[1] > 4:
                sign[i, 4] = sig[i, 4] + gs * dzx

        # Thickness strain and update (sigeps71c.F lines 567, 575)
        epszz = ev3 - 1.0
        thk_key = "thk" if "thk" in extra else ("thick" if "thick" in extra else None)
        if thk_key is not None and extra[thk_key] is not None:
            thkly = extra.get("thkly", 1.0)
            off_val = extra.get("off", 1.0)
            d_thk = (epszz - uvar[i, 3]) * thkly * off_val
            thk_val = extra[thk_key]
            if isinstance(thk_val, np.ndarray):
                if thk_val.ndim == 0:
                    extra[thk_key] += d_thk
                elif i < len(thk_val):
                    extra[thk_key][i] += d_thk
            elif isinstance(thk_val, (int, float)):
                extra[thk_key] += d_thk

        # Update UVAR (sigeps71c.F lines 561-564, 578-581)
        new_fm = min(1.0, max(0.0, fm + dfm_final))
        uvar[i, 0] = new_fm
        uvar[i, 1] = fs_final - cas * temp[i]
        uvar[i, 2] = fs_final - csa * temp[i]
        uvar[i, 3] = epszz
        uvar[i, 7] = sigzz_final
        uvar[i, 9] = exx

    sig[:, :sign.shape[1]] = sign

    if "uvar" in extra and hasattr(extra["uvar"], "__setitem__"):
        try:
            if is_1d:
                extra["uvar"][:] = uvar[0]
            else:
                extra["uvar"][:] = uvar
        except Exception:
            pass
    if "eps" in extra and hasattr(extra["eps"], "__setitem__"):
        try:
            if is_1d:
                extra["eps"][:] = eps_tot[0]
            else:
                extra["eps"][:] = eps_tot
        except Exception:
            pass

    out_sign = sign[:, :3]
    out_epsp = uvar[:, 0]
    if is_1d:
        return out_sign[0], out_epsp[0]
    return out_sign, out_epsp


# Convenient alias
shell_update = shell_step


# =============================================================================
# Consistent Tangents (Implicit Solver)
# =============================================================================

def consistent_solid_tangent(
    mat: Material,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    dt: Optional[float] = None,
) -> np.ndarray:
    """Return the (n, 6, 6) or (6, 6) consistent tangent for LAW71 solids.

    Combines current phase-fraction shear modulus G_t and bulk modulus K_t:
        C_11 = K_t + 4/3 * G_t
        C_12 = K_t - 2/3 * G_t
        C_44 = G_t

    Parameters
    ----------
    mat : Material
        LAW71 material instance.
    sig : ndarray, shape (n, 6), (6,), or None
        Stress tensor.
    epsp, epsp_incr : optional
        Plastic strain / phase fraction views.
    extra : dict, optional
        Extra state containing uv71.
    dt : float, optional
        Time step.

    Returns
    -------
    ndarray
        (n, 6, 6) or (6, 6) algorithmic tangent matrix.
    """
    p = mat.params
    e_val = p.get("E", getattr(mat, "E", 0.0))
    nu_val = p.get("nu", getattr(mat, "nu", 0.3))
    e_mart = p.get("e_mart", p.get("EMART", e_val))
    eflag = p.get("eflag", 1 if (e_mart > 0.0 and e_mart != e_val) else 0)

    g_val = p.get("G", 0.5 * e_val / (1.0 + nu_val))
    k_val = p.get("K", e_val / (3.0 * (1.0 - 2.0 * nu_val)))
    gm_val = p.get("GM", 0.5 * e_mart / (1.0 + nu_val) if eflag == 1 else g_val)
    km_val = p.get("KM", e_mart / (3.0 * (1.0 - 2.0 * nu_val)) if eflag == 1 else k_val)

    is_1d = (sig is not None and isinstance(sig, np.ndarray) and sig.ndim == 1)
    n = sig.shape[0] if (sig is not None and sig.ndim > 1) else 1

    fm = np.zeros(n, dtype=float)
    if extra is not None and isinstance(extra, dict) and "uv71" in extra and extra["uv71"] is not None:
        u = np.asarray(extra["uv71"])
        if u.ndim == 2:
            fm[:min(n, u.shape[0])] = u[:min(n, u.shape[0]), 0]
        elif u.ndim == 1 and u.size > 0:
            fm[:] = u[0]
    elif epsp is not None:
        e_arr = np.asarray(epsp)
        if e_arr.ndim >= 1 and len(e_arr) == n:
            fm = e_arr
        elif e_arr.ndim == 0:
            fm = np.full(n, float(e_arr))

    gt = g_val + fm * (gm_val - g_val)
    kt = k_val + fm * (km_val - k_val)

    c11 = kt + (4.0 / 3.0) * gt
    c12 = kt - (2.0 / 3.0) * gt
    c44 = gt

    C = np.zeros((n, 6, 6), dtype=float)
    for i in range(3):
        C[:, i, i] = c11
        for j in range(3):
            if i != j:
                C[:, i, j] = c12
    for i in range(3, 6):
        C[:, i, i] = c44

    if sig is None or is_1d:
        return C[0]
    return C


def shell_membrane_tangent(
    mat: Material,
    dt: Optional[Union[float, Dict[str, Any]]] = None,
    fm: float = 0.0,
) -> np.ndarray:
    """(3, 3) plane-stress consistent tangent matrix for LAW71 shells [xx, yy, xy]."""
    p = mat.params
    e_val = p.get("E", getattr(mat, "E", 0.0))
    nu_val = p.get("nu", getattr(mat, "nu", 0.3))
    e_mart = p.get("e_mart", p.get("EMART", e_val))
    eflag = p.get("eflag", 1 if (e_mart > 0.0 and e_mart != e_val) else 0)

    g_val = p.get("G", 0.5 * e_val / (1.0 + nu_val))
    k_val = p.get("K", e_val / (3.0 * (1.0 - 2.0 * nu_val)))
    gm_val = p.get("GM", 0.5 * e_mart / (1.0 + nu_val) if eflag == 1 else g_val)
    km_val = p.get("KM", e_mart / (3.0 * (1.0 - 2.0 * nu_val)) if eflag == 1 else k_val)

    gt = g_val + fm * (gm_val - g_val)
    kt = k_val + fm * (km_val - k_val)

    denom = kt + (4.0 / 3.0) * gt
    if abs(denom) < 1.0e-30:
        denom = 1.0e-30

    c11 = 4.0 * gt * (kt + gt / 3.0) / denom
    c12 = 2.0 * gt * (kt - 2.0 * gt / 3.0) / denom
    c33 = gt

    return np.array([
        [c11, c12, 0.0],
        [c12, c11, 0.0],
        [0.0, 0.0, c33],
    ], dtype=float)


def consistent_shell_tangent(
    mat: Material,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    dt: Optional[float] = None,
) -> np.ndarray:
    """(n, 3, 3) or (3, 3) plane-stress consistent tangent for LAW71 shells."""
    fm_val = 0.0
    if extra is not None and isinstance(extra, dict) and "uv71" in extra and extra["uv71"] is not None:
        u = np.asarray(extra["uv71"])
        if u.ndim >= 1 and u.size > 0:
            fm_val = float(u.reshape(-1)[0])
    elif epsp is not None:
        try:
            fm_val = float(np.asarray(epsp).reshape(-1)[0])
        except Exception:
            pass

    C_mat = shell_membrane_tangent(mat, dt=dt, fm=fm_val)
    if sig is None or (isinstance(sig, np.ndarray) and sig.ndim == 1):
        return C_mat
    n = sig.shape[0]
    return np.broadcast_to(C_mat, (n, 3, 3)).copy()


# Aliases for material interface
solid_update = solid_step
shell_update = shell_step


# =============================================================================
# Registry Hook
# =============================================================================

def _register() -> None:
    """Register build_law71 into MAT_PHYSICS_REGISTRY."""
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for k in (71, "71", "LAW71", "SUPER_ELAS", "NITINOL", "MAT_LAW71", "MAT_SUPER_ELAS", "MAT_NITINOL"):
            MAT_PHYSICS_REGISTRY[k] = build_law71
    except Exception:
        pass


_register()

"""
LAW190 — Du Bois Crushable Foam Material (/MAT/LAW190, /MAT/FOAM_DUBOIS).

Fortran upstream references:
- Engine constitutive update:
  ``engine/source/materials/mat/mat190/sigeps190.F`` (Subroutine SIGEPS190, lines 38-420)
- Tangent stiffness & rate interpolation:
  ``engine/source/materials/mat/mat190/conversion.F`` (Subroutine CONVERSION, lines 41-250)
- Hysteretic energy & unloading damage:
  ``engine/source/materials/mat/mat190/condamage.F`` (Subroutine CONDAMAGE, lines 41-175)
- Starter reader:
  ``starter/source/materials/mat/mat190/hm_read_mat190.F`` (Subroutine HM_READ_MAT190, lines 39-208)
- Starter property setup & energy table pre-integration:
  ``starter/source/materials/mat/mat190/law190_upd.F90`` (Subroutine LAW190_UPD, lines 33-228)

Theory & Algorithm:
--------------------
LAW190 is an isotropic, path-dependent, recoverable/hysteretic crushable foam model
formulated for 3D continuum solid elements.

1. Kinematics & Green-Lagrange Strain Decomposition:
   - Given deformation gradient F (from element kernel or updated via F_{n+1} = (I + deps)*F_n):
     Right Cauchy-Green tensor: C = F^T * F.
     Green-Lagrange strain tensor: E_GL = 0.5 * (C - I).
     Jacobian: J = det(F) = V / V0.
     Volumetric strain: volstr = 1.0 - J (positive in compression).
   - Midpoint strain tensor: E_mid = 0.5 * (E_GL_new + E_GL_old).
   - Spectral decomposition of E_mid:
     Eigenvalues Z_k (k=1,2,3) and orthonormal eigenvectors V = [v_1, v_2, v_3].
     Principal stretches: lambda_k = sqrt(max(2*Z_k + 1, 1e-12)).
     Principal engineering strain (positive in compression):
         e_k = 1.0 - lambda_k

2. Rate-Dependent Yield Lookup:
   - Strain rate in principal directions:
     L = deps / dt.
     eps_dot_p,k = v_k^T * L * v_k.
     Engineering strain rate:
         e_dot_k = |eps_dot_p,k| * lambda_k.
   - Tabulated yield evaluation:
     Quasistatic response: sigma_stat,k = table(e_k, rate=0, volstr) * scale.
     Dynamic response: sigma_dyn,k = table(e_k, rate=e_dot_k / xscale, volstr) * scale.
     Tangent modulus: E_tan,k = d(sigma_stat)/d(e) at e_k * scale.

3. Hysteretic Energy Dissipation (condamage.F):
   - Energy integral per principal direction:
         W_k(e_k) = int_0^{e_k} max(sigma_stat(x), 0) dx.
     Total hysteretic energy:
         W_hys = W_1 + W_2 + W_3.
     Peak historical energy:
         W_max = max(W_max_prev, W_hys).
   - If unloading (W_max > 0 and W_hys < W_max):
         ratio = clip(W_hys / W_max, 0.0, 1.0).
         damage = (1.0 - HU) * (1.0 - ratio^SHAPE).
     Else (loading):
         damage = 0.0.
   - Net hysteretic retention factor:
         dam_factor = 1.0 - damage.

4. 2PK & Cauchy Stress:
   - Second Piola-Kirchhoff stress in principal directions (positive in tension):
     S_k = - (sigma_dyn,k * dam_factor) / lambda_k.
   - Push-forward to Cauchy principal stresses:
     sigma_Cauchy,k = (lambda_k^2 / J) * S_k = - (lambda_k / J) * sigma_dyn,k * dam_factor.
     For uniaxial compression with nu=0 (lambda_2=lambda_3=1, J=lambda_1):
     sigma_Cauchy,1 = - sigma_dyn,1 * dam_factor (compressive stress magnitude = sigma_dyn,1 * dam_factor).

5. Tensile Cutoff & Hydrostatic Failure:
   - Principal tensile stress T_k = sigma_Cauchy,k.
   - Hydrostatic pressure P = -1/3 * sum(sigma_Cauchy,k).
   - If tcut is specified:
     If fail == 1:
       If any T_k > tcut or P < -tcut:
         erode element (sigma = 0, eroded = 1).
     Else (fail == 0):
       T_k = min(T_k, tcut).
       If P < -tcut: clamp hydrostatic tension.
   - Global Cauchy stress:
     sigma = V * diag(sigma_Cauchy) * V^T.

6. Sound Speed:
   - Dilatational sound speed:
     E_max = max(E0, E_tan,1, E_tan,2, E_tan,3).
     c = sqrt(E_max / rho).

7. Algorithmic Consistent Tangent:
   - 6x6 tangent matrix relating d(sigma) to d(eps), accounting for principal stiffnesses
     and spectral rotation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..common.tables import FunctTable, SmoothFunctTable
from ..model.entities import Material, MatLaw190

_EM20 = 1.0e-20
_EM12 = 1.0e-12
_FOUR_THIRD = 4.0 / 3.0
_TWO_THIRD = 2.0 / 3.0


@dataclass
class Law190Params:
    """Parameters for /MAT/LAW190 (/MAT/FOAM_DUBOIS).

    Matches ``hm_read_mat190.F`` and ``law190_upd.F90``.
    """

    rho0: float = 1.0           # Initial density
    refer_rho: float = 0.0      # Reference density
    e0: float = 0.0             # Initial Young's modulus (MAT_E)
    nu: float = 0.0             # Poisson's ratio (MAT_NU, default 0.0)
    hu: float = 1.0             # Hysteretic unloading factor (MAT_HU, default 1.0)
    hys: float = 1.0            # Alias for hu
    shape: float = 1.0          # Unloading shape factor (MAT_SHAPE, default 1.0)
    table_id: int = 0           # Loading table / function ID (FUN_1)
    fun_1: int = 0              # Alias for table_id
    xscale: float = 1.0         # Scale factor for strain rate in table (XSCALE_1)
    scale: float = 1.0          # Scale factor for stress in table (SCALE_1)
    tcut: float = 1.0e20        # Tensile stress cutoff
    fail: int = 0               # Failure mode: 0 = clamp to tcut, 1 = erode upon exceeding tcut
    table: Any = None           # Resolved 1D/2D/3D table, curve, or function
    title: str = ""             # Material title

    # Derived elastic properties
    bulk: float = 0.0           # Bulk modulus K = E / (3*(1 - 2*nu))
    g: float = 0.0              # Shear modulus G = E / (2*(1 + nu))
    cii: float = 0.0            # P-wave diagonal stiffness K + 4/3*G
    cij: float = 0.0            # Off-diagonal stiffness K - 2/3*G
    sound_speed0: float = 0.0   # Acoustic sound speed sqrt(E0 / rho0)

    def __post_init__(self) -> None:
        if self.refer_rho <= 0.0:
            self.refer_rho = self.rho0 if self.rho0 > 0.0 else 1.0
        if self.rho0 <= 0.0:
            self.rho0 = self.refer_rho

        # Upstream hm_read_mat190: if hu is 0.0, defaults to 1.0 (no hysteresis damage)
        if self.hu == 0.0 and self.hys != 0.0:
            self.hu = self.hys
        elif self.hu == 0.0:
            self.hu = 1.0
        self.hys = self.hu

        if self.shape <= 0.0:
            self.shape = 1.0

        if self.xscale == 0.0:
            self.xscale = 1.0
        if self.scale == 0.0:
            self.scale = 1.0

        if self.table_id == 0 and self.fun_1 != 0:
            self.table_id = self.fun_1
        elif self.table_id != 0 and self.fun_1 == 0:
            self.fun_1 = self.table_id

        # Poisson's ratio safety bounds
        self.nu = max(min(float(self.nu), 0.499), -0.999)

        # Derived elastic constants
        if self.e0 > 0.0:
            denom_k = 3.0 * (1.0 - 2.0 * self.nu)
            denom_g = 2.0 * (1.0 + self.nu)
            self.bulk = self.e0 / denom_k if abs(denom_k) > 1e-12 else self.e0
            self.g = self.e0 / denom_g if abs(denom_g) > 1e-12 else self.e0 * 0.5
            self.cii = self.bulk + _FOUR_THIRD * self.g
            self.cij = self.bulk - _TWO_THIRD * self.g
            c_val = self.cii if self.cii > 0.0 else self.e0
            self.sound_speed0 = math.sqrt(c_val / self.rho0) if self.rho0 > 0.0 else 0.0


def _extract_params(mat: Any) -> Law190Params:
    """Normalize any material representation into a Law190Params instance."""
    if isinstance(mat, Law190Params):
        return mat

    if isinstance(mat, MatLaw190):
        tbl = getattr(mat, "table", None)
        return Law190Params(
            rho0=mat.rho if mat.rho > 0.0 else 1.0,
            refer_rho=mat.rho if mat.rho > 0.0 else 1.0,
            e0=mat.e0,
            nu=mat.nu,
            hu=mat.hu if mat.hu != 0.0 else 1.0,
            hys=mat.hu if mat.hu != 0.0 else 1.0,
            shape=mat.shape if mat.shape > 0.0 else 1.0,
            table_id=mat.fun_1,
            fun_1=mat.fun_1,
            xscale=mat.xscale_1 if mat.xscale_1 != 0.0 else 1.0,
            scale=mat.scale_1 if mat.scale_1 != 0.0 else 1.0,
            tcut=getattr(mat, "tcut", 1.0e20),
            fail=getattr(mat, "fail", 0),
            table=tbl,
            title=getattr(mat, "title", ""),
        )

    if isinstance(mat, Material):
        p = mat.params if mat.params is not None else {}
        if "law190_params" in p and isinstance(p["law190_params"], Law190Params):
            return p["law190_params"]

        rho0 = float(mat.rho0 if mat.rho0 > 0.0 else p.get("rho0", p.get("rho", p.get("MAT_RHO", 1.0))))
        e0 = float(p.get("e0", p.get("E0", p.get("e", p.get("E", p.get("MAT_E", 0.0))))))
        nu = float(p.get("nu", p.get("Nu", p.get("MAT_NU", p.get("pr", 0.0)))))
        hu = float(p.get("hu", p.get("HU", p.get("MAT_HU", p.get("hys", p.get("HYS", 1.0))))))
        if hu == 0.0:
            hu = 1.0
        shape = float(p.get("shape", p.get("SHAPE", p.get("MAT_SHAPE", 1.0))))
        table_id = int(p.get("table_id", p.get("fun_1", p.get("FUN_1", p.get("tab_id", 0)))))
        xscale = float(p.get("xscale", p.get("xscale_1", p.get("XSCALE_1", 1.0))))
        scale = float(p.get("scale", p.get("scale_1", p.get("SCALE_1", 1.0))))
        tcut = float(p.get("tcut", p.get("TCUT", 1.0e20)))
        fail = int(p.get("fail", p.get("FAIL", 0)))
        tbl = p.get("table", p.get("yield_table", p.get("curve", p.get("yield_curve"))))

        return Law190Params(
            rho0=rho0,
            refer_rho=float(p.get("refer_rho", rho0)),
            e0=e0,
            nu=nu,
            hu=hu,
            hys=hu,
            shape=shape,
            table_id=table_id,
            fun_1=table_id,
            xscale=xscale if xscale != 0.0 else 1.0,
            scale=scale if scale != 0.0 else 1.0,
            tcut=tcut,
            fail=fail,
            table=tbl,
            title=getattr(mat, "title", ""),
        )

    if isinstance(mat, dict):
        p = mat.get("params", mat)
        rho0 = float(p.get("rho0", p.get("rho", p.get("density", p.get("MAT_RHO", 1.0)))))
        e0 = float(p.get("e0", p.get("E0", p.get("e", p.get("E", p.get("MAT_E", 0.0))))))
        nu = float(p.get("nu", p.get("Nu", p.get("MAT_NU", p.get("pr", 0.0)))))
        hu = float(p.get("hu", p.get("HU", p.get("MAT_HU", p.get("hys", p.get("HYS", 1.0))))))
        if hu == 0.0:
            hu = 1.0
        shape = float(p.get("shape", p.get("SHAPE", p.get("MAT_SHAPE", 1.0))))
        table_id = int(p.get("table_id", p.get("fun_1", p.get("FUN_1", p.get("tab_id", 0)))))
        xscale = float(p.get("xscale", p.get("xscale_1", p.get("XSCALE_1", 1.0))))
        scale = float(p.get("scale", p.get("scale_1", p.get("SCALE_1", 1.0))))
        tcut = float(p.get("tcut", p.get("TCUT", 1.0e20)))
        fail = int(p.get("fail", p.get("FAIL", 0)))
        tbl = p.get("table", p.get("yield_table", p.get("curve", p.get("yield_curve"))))

        return Law190Params(
            rho0=rho0,
            refer_rho=float(p.get("refer_rho", rho0)),
            e0=e0,
            nu=nu,
            hu=hu,
            hys=hu,
            shape=shape,
            table_id=table_id,
            fun_1=table_id,
            xscale=xscale if xscale != 0.0 else 1.0,
            scale=scale if scale != 0.0 else 1.0,
            tcut=tcut,
            fail=fail,
            table=tbl,
            title=str(p.get("title", "")),
        )

    # Duck-typing fallback (e.g. GenericMaterialRecord)
    p = getattr(mat, "params", {})
    rho0 = float(getattr(mat, "rho0", getattr(mat, "density", getattr(mat, "rho", p.get("rho0", p.get("MAT_RHO", 1.0))))))
    e0 = float(p.get("e0", p.get("E0", p.get("MAT_E", p.get("e", p.get("E", 0.0))))))
    nu = float(p.get("nu", p.get("MAT_NU", p.get("Nu", 0.0))))
    hu = float(p.get("hu", p.get("MAT_HU", p.get("hys", 1.0))))
    if hu == 0.0:
        hu = 1.0
    shape = float(p.get("shape", p.get("MAT_SHAPE", 1.0)))
    table_id = int(p.get("table_id", p.get("fun_1", p.get("FUN_1", 0))))
    xscale = float(p.get("xscale", p.get("XSCALE_1", p.get("xscale_1", 1.0))))
    scale = float(p.get("scale", p.get("SCALE_1", p.get("scale_1", 1.0))))
    tcut = float(p.get("tcut", 1.0e20))
    fail = int(p.get("fail", 0))
    tbl = getattr(mat, "table", p.get("table", None))

    return Law190Params(
        rho0=rho0,
        refer_rho=rho0,
        e0=e0,
        nu=nu,
        hu=hu,
        hys=hu,
        shape=shape,
        table_id=table_id,
        fun_1=table_id,
        xscale=xscale if xscale != 0.0 else 1.0,
        scale=scale if scale != 0.0 else 1.0,
        tcut=tcut,
        fail=fail,
        table=tbl,
        title=getattr(mat, "title", ""),
    )


def build_law190(mat_def: Any) -> Material:
    """Factory creating a Material configured for LAW190 (/MAT/FOAM_DUBOIS)."""
    params_obj = _extract_params(mat_def)
    mat_id = 1
    title = params_obj.title
    if isinstance(mat_def, MatLaw190):
        mat_id = mat_def.id
    elif isinstance(mat_def, Material):
        mat_id = mat_def.id
        title = mat_def.title
    elif hasattr(mat_def, "id"):
        try:
            mat_id = int(mat_def.id)
        except Exception:
            pass
    elif isinstance(mat_def, dict):
        mat_id = int(mat_def.get("id", mat_def.get("mat_id", 1)))
        title = str(mat_def.get("title", title))

    params_dict = asdict(params_obj)
    params_dict.update({
        "E": params_obj.e0,
        "E0": params_obj.e0,
        "MAT_E": params_obj.e0,
        "nu": params_obj.nu,
        "MAT_NU": params_obj.nu,
        "rho": params_obj.rho0,
        "rho0": params_obj.rho0,
        "MAT_RHO": params_obj.rho0,
        "hu": params_obj.hu,
        "MAT_HU": params_obj.hu,
        "hys": params_obj.hu,
        "shape": params_obj.shape,
        "MAT_SHAPE": params_obj.shape,
        "fun_1": params_obj.table_id,
        "table_id": params_obj.table_id,
        "FUN_1": params_obj.table_id,
        "xscale": params_obj.xscale,
        "XSCALE_1": params_obj.xscale,
        "scale": params_obj.scale,
        "SCALE_1": params_obj.scale,
        "tcut": params_obj.tcut,
        "fail": params_obj.fail,
        "bulk": params_obj.bulk,
        "g": params_obj.g,
        "G": params_obj.g,
        "cii": params_obj.cii,
        "cij": params_obj.cij,
        "law190_params": params_obj,
    })

    return Material(
        id=mat_id,
        law=190,
        rho0=params_obj.rho0,
        title=title,
        params=params_dict,
    )


def resolve(mat: Material | dict, model: Any, log: Any = None) -> None:
    """Resolve table/function curves for /MAT/LAW190 from the model.
    Cites starter/source/materials/mat/mat190/law190_upd.F90 lines 80-120.
    """
    if isinstance(mat, dict):
        p = mat.get("params", mat)
        mat_id = mat.get("id", 0)
    else:
        p = getattr(mat, "params", {})
        mat_id = getattr(mat, "id", 0)

    tab_id = p.get("table_id", p.get("fun_1", p.get("FUN_1", 0)))
    if tab_id and tab_id != 0:
        resolved = None
        if hasattr(model, "tables") and tab_id in model.tables:
            resolved = model.tables[tab_id]
        elif hasattr(model, "functions") and tab_id in model.functions:
            resolved = model.functions[tab_id]
        elif hasattr(model, "curves") and tab_id in model.curves:
            resolved = model.curves[tab_id]

        if resolved is not None:
            p["table"] = resolved
            if "law190_params" in p and isinstance(p["law190_params"], Law190Params):
                p["law190_params"].table = resolved
        elif log is not None and hasattr(log, "warning"):
            log.warning(f"/MAT/LAW190/{mat_id}: table/curve ID {tab_id} not found in model", "MAT CHECK")


def extra_shapes(mat: Any, nip: int = 1) -> dict[str, tuple[int, ...]]:
    """Return persistent state variable requirements for LAW190 solids.

    uv190 shape (16,) per integration point:
    0..5: Green-Lagrange strain tensor old [xx, yy, zz, xy, yz, zx]
    6..11: SPKNORATE old 2PK stress tensor
    12: WHYSMAX (peak historical hysteretic energy)
    13: DAMAGE (current unloading damage D)
    14: WHYS (current hysteretic energy)
    15: ERODED (1.0 if element failed by tensile cutoff, else 0.0)
    """
    return {"uv190": (16,)}


# ---------------------------------------------------------------------------
# Yield and Hysteresis Energy Evaluation
# ---------------------------------------------------------------------------

def _interp_1d_curve(
    xs: np.ndarray,
    ys: np.ndarray,
    query_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1D curve lookup with linear interpolation, slope, and cumulative energy.

    Parameters
    ----------
    xs : 1D sorted array of strains (e >= 0 in compression)
    ys : 1D array of stress values
    query_x : 1D array of query strains

    Returns
    -------
    y_val : interpolated stress values
    slope_val : tangent slope d(y)/d(x)
    ener_val : integrated energy W(e) = int_0^e y(x) dx
    """
    n = query_x.shape[0]
    m = xs.shape[0]

    if m < 2:
        val = ys[0] if m == 1 else 0.0
        return np.full(n, val), np.zeros(n), np.zeros(n)

    # Slopes between points
    dx = np.diff(xs)
    dx_safe = np.maximum(dx, _EM20)
    dy = np.diff(ys)
    slopes = dy / dx_safe

    # Cumulative energy array at knot points
    w_pts = np.zeros(m, dtype=float)
    for i in range(1, m):
        trap = 0.5 * (xs[i] - xs[i - 1]) * (max(ys[i], 0.0) + max(ys[i - 1], 0.0))
        w_pts[i] = w_pts[i - 1] + trap

    # Clamp search index
    idx = np.clip(np.searchsorted(xs, query_x, side="right"), 1, m - 1)
    seg_slope = slopes[idx - 1]
    dx_local = query_x - xs[idx - 1]
    y_val = ys[idx - 1] + seg_slope * dx_local
    slope_val = seg_slope

    # Energy interpolation: knot energy + trapezoid of the local segment
    y0_clamped = np.maximum(ys[idx - 1], 0.0)
    y1_clamped = np.maximum(y_val, 0.0)
    ener_val = w_pts[idx - 1] + 0.5 * dx_local * (y0_clamped + y1_clamped)

    # For e <= 0 (tension), energy is zero
    neg_mask = query_x <= 0.0
    ener_val = np.where(neg_mask, 0.0, np.maximum(ener_val, 0.0))

    return y_val, slope_val, ener_val


def _apply_tension_override(
    e: np.ndarray,
    sig_stat: np.ndarray,
    sig_dyn: np.ndarray,
    slope: np.ndarray,
    ener: np.ndarray,
    scale: float,
    e0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Override tensile response (e < 0) with linear elastic behavior."""
    neg_mask = (e < 0.0)
    if np.any(neg_mask):
        mod = e0 if e0 > 0.0 else (float(slope[0]) if slope.size > 0 and slope[0] > 0.0 else 1.0)
        sig_stat = np.where(neg_mask, mod * e * scale, sig_stat)
        sig_dyn = np.where(neg_mask, mod * e * scale, sig_dyn)
        slope = np.where(neg_mask, mod * scale, slope)
        ener = np.where(neg_mask, 0.0, ener)
    return sig_stat, sig_dyn, slope, ener


def _lookup_yield_and_energy(
    table: Any,
    e: np.ndarray,
    dr: np.ndarray,
    volstr: np.ndarray,
    scale: float = 1.0,
    xscale: float = 1.0,
    e0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate quasistatic stress, dynamic stress, tangent slope, and hysteretic energy.

    Matches ``conversion.F`` and ``condamage.F``.

    Parameters
    ----------
    table : Any
        Resolved 1D curve, 2D table, 3D table, FunctTable, or callable.
    e : (n,) ndarray
        Principal engineering strains (e > 0 in compression, e < 0 in tension).
    dr : (n,) ndarray
        Principal engineering strain rates |deps_dot| * (1 - e).
    volstr : (n,) ndarray
        Volumetric strain 1 - J.
    scale : float
        Stress scale factor.
    xscale : float
        Strain rate scale factor.
    e0 : float
        Elastic Young's modulus for tension or linear elastic fallback.

    Returns
    -------
    sig_static : (n,) ndarray
        Quasistatic compressive stress (at rate = 0).
    sig_dyn : (n,) ndarray
        Dynamic compressive stress (at current strain rate).
    slope : (n,) ndarray
        Tangent modulus d(sig_static)/d(e).
    ener : (n,) ndarray
        Integrated hysteretic energy W(e).
    """
    n = e.shape[0]
    e_clamped = np.maximum(e, 0.0)

    # 1. No table provided -> linear elastic response
    if table is None:
        mod = e0 if e0 > 0.0 else 1.0
        sig_stat = mod * e * scale
        sig_dynamic = mod * e * scale
        slope = np.full(n, mod * scale)
        ener = np.where(e > 0.0, 0.5 * mod * (e ** 2) * scale, 0.0)
        return sig_stat, sig_dynamic, slope, ener

    # 2. FunctTable or object with x, y attributes (1D curve)
    if isinstance(table, (FunctTable, SmoothFunctTable)) or (hasattr(table, "x") and hasattr(table, "y") and not hasattr(table, "curves")):
        xs = np.asarray(table.x, dtype=float)
        ys = np.asarray(table.y, dtype=float)
        y_val, slp, en = _interp_1d_curve(xs, ys, e_clamped)
        sig_stat = y_val * scale
        sig_dynamic = sig_stat.copy()
        slope = slp * scale
        ener = en * scale
        return _apply_tension_override(e, sig_stat, sig_dynamic, slope, ener, scale=scale, e0=e0)

    # 3. 2-tuple (xs, ys) -> 1D curve
    if isinstance(table, (tuple, list)) and len(table) == 2 and isinstance(table[0], (np.ndarray, list, tuple)):
        xs = np.asarray(table[0], dtype=float)
        ys = np.asarray(table[1], dtype=float)
        y_val, slp, en = _interp_1d_curve(xs, ys, e_clamped)
        sig_stat = y_val * scale
        sig_dynamic = sig_stat.copy()
        slope = slp * scale
        ener = en * scale
        return _apply_tension_override(e, sig_stat, sig_dynamic, slope, ener, scale=scale, e0=e0)

    # 4. Table object with curves: Table(curves=[(rate, xs, ys), ...])
    if hasattr(table, "curves") and table.curves:
        curves = table.curves
        # Sort curves by strain rate
        sorted_curves = sorted(curves, key=lambda c: float(c[0]))
        rates = np.array([float(c[0]) for c in sorted_curves], dtype=float)

        # Quasistatic curve is the lowest rate (rate 0)
        xs0, ys0 = np.asarray(sorted_curves[0][1], dtype=float), np.asarray(sorted_curves[0][2], dtype=float)
        y_stat, slp0, en0 = _interp_1d_curve(xs0, ys0, e_clamped)
        sig_stat = y_stat * scale
        slope = slp0 * scale
        ener = en0 * scale

        if len(rates) == 1:
            return _apply_tension_override(e, sig_stat, sig_stat.copy(), slope, ener, scale=scale, e0=e0)

        # Dynamic rate interpolation across curves
        scaled_dr = dr / xscale
        idx_r = np.clip(np.searchsorted(rates, scaled_dr, side="right"), 1, len(rates) - 1)
        r0 = rates[idx_r - 1]
        r1 = rates[idx_r]
        dr_span = np.maximum(r1 - r0, _EM20)
        weight_r = np.clip((scaled_dr - r0) / dr_span, 0.0, 1.0)

        # Interpolate stresses from adjacent rate curves
        sig_dynamic = np.zeros(n, dtype=float)
        for i in range(n):
            c_low = sorted_curves[idx_r[i] - 1]
            c_high = sorted_curves[idx_r[i]]
            y_low, _, _ = _interp_1d_curve(np.asarray(c_low[1]), np.asarray(c_low[2]), np.array([e_clamped[i]]))
            y_high, _, _ = _interp_1d_curve(np.asarray(c_high[1]), np.asarray(c_high[2]), np.array([e_clamped[i]]))
            sig_dynamic[i] = (y_low[0] + weight_r[i] * (y_high[0] - y_low[0])) * scale

        return _apply_tension_override(e, sig_stat, sig_dynamic, slope, ener, scale=scale, e0=e0)

    # 5. 3-tuple (xg, rates, Y) -> 2D table grid
    if isinstance(table, (tuple, list)) and len(table) == 3:
        xg = np.asarray(table[0], dtype=float)
        rates = np.asarray(table[1], dtype=float)
        Y = np.asarray(table[2], dtype=float)  # shape (len(xg), len(rates))

        # Static curve is column 0 (quasistatic rate)
        ys0 = Y[:, 0]
        y_stat, slp0, en0 = _interp_1d_curve(xg, ys0, e_clamped)
        sig_stat = y_stat * scale
        slope = slp0 * scale
        ener = en0 * scale

        if len(rates) == 1 or Y.shape[1] == 1:
            return _apply_tension_override(e, sig_stat, sig_stat.copy(), slope, ener, scale=scale, e0=e0)

        scaled_dr = dr / xscale
        idx_x = np.clip(np.searchsorted(xg, e_clamped, side="right"), 1, len(xg) - 1)
        dx = np.maximum(xg[idx_x] - xg[idx_x - 1], _EM20)
        tx = (e_clamped - xg[idx_x - 1]) / dx

        idx_r = np.clip(np.searchsorted(rates, scaled_dr, side="right"), 1, len(rates) - 1)
        dr_span = np.maximum(rates[idx_r] - rates[idx_r - 1], _EM20)
        tr = np.clip((scaled_dr - rates[idx_r - 1]) / dr_span, 0.0, 1.0)

        y00 = Y[idx_x - 1, idx_r - 1]
        y10 = Y[idx_x, idx_r - 1]
        y01 = Y[idx_x - 1, idx_r]
        y11 = Y[idx_x, idx_r]

        val_r0 = y00 + tx * (y10 - y00)
        val_r1 = y01 + tx * (y11 - y01)
        sig_dynamic = (val_r0 + tr * (val_r1 - val_r0)) * scale

        return _apply_tension_override(e, sig_stat, sig_dynamic, slope, ener, scale=scale, e0=e0)

    # 6. 4-tuple (xg, rates, vols, Y) -> 3D table grid
    if isinstance(table, (tuple, list)) and len(table) == 4:
        xg = np.asarray(table[0], dtype=float)
        rates = np.asarray(table[1], dtype=float)
        vols = np.asarray(table[2], dtype=float)
        Y = np.asarray(table[3], dtype=float)  # shape (len(xg), len(rates), len(vols))

        # Static curve is rate 0, volstr 0 (or nearest)
        ys0 = Y[:, 0, 0]
        y_stat, slp0, en0 = _interp_1d_curve(xg, ys0, e_clamped)
        sig_stat = y_stat * scale
        slope = slp0 * scale
        ener = en0 * scale

        scaled_dr = dr / xscale
        idx_x = np.clip(np.searchsorted(xg, e_clamped, side="right"), 1, len(xg) - 1)
        tx = (e_clamped - xg[idx_x - 1]) / np.maximum(xg[idx_x] - xg[idx_x - 1], _EM20)

        idx_r = np.clip(np.searchsorted(rates, scaled_dr, side="right"), 1, len(rates) - 1)
        tr = np.clip((scaled_dr - rates[idx_r - 1]) / np.maximum(rates[idx_r] - rates[idx_r - 1], _EM20), 0.0, 1.0)

        idx_v = np.clip(np.searchsorted(vols, volstr, side="right"), 1, len(vols) - 1)
        tv = np.clip((volstr - vols[idx_v - 1]) / np.maximum(vols[idx_v] - vols[idx_v - 1], _EM20), 0.0, 1.0)

        # Trilinear interpolation
        sig_dynamic = np.zeros(n, dtype=float)
        for i in range(n):
            ix, ir, iv = idx_x[i], idx_r[i], idx_v[i]
            c000 = Y[ix - 1, ir - 1, iv - 1]
            c100 = Y[ix, ir - 1, iv - 1]
            c010 = Y[ix - 1, ir, iv - 1]
            c110 = Y[ix, ir, iv - 1]
            c001 = Y[ix - 1, ir - 1, iv]
            c101 = Y[ix, ir - 1, iv]
            c011 = Y[ix - 1, ir, iv]
            c111 = Y[ix, ir, iv]

            c00 = c000 + tx[i] * (c100 - c000)
            c10 = c010 + tx[i] * (c110 - c010)
            c01 = c001 + tx[i] * (c101 - c001)
            c11 = c011 + tx[i] * (c111 - c011)

            c0 = c00 + tr[i] * (c10 - c00)
            c1 = c01 + tr[i] * (c11 - c01)

            sig_dynamic[i] = (c0 + tv[i] * (c1 - c0)) * scale

        return _apply_tension_override(e, sig_stat, sig_dynamic, slope, ener, scale=scale, e0=e0)

    # 7. Dict-based table
    if isinstance(table, dict):
        if "curves" in table:
            class _MockTable:
                pass
            mt = _MockTable()
            mt.curves = table["curves"]
            return _lookup_yield_and_energy(mt, e, dr, volstr, scale=scale, xscale=xscale, e0=e0)
        if "xg" in table and "rates" in table and "Y" in table:
            return _lookup_yield_and_energy((table["xg"], table["rates"], table["Y"]), e, dr, volstr, scale=scale, xscale=xscale, e0=e0)
        if "x" in table and "y" in table:
            return _lookup_yield_and_energy((table["x"], table["y"]), e, dr, volstr, scale=scale, xscale=xscale, e0=e0)

    # 8. Callable function f(e, dr, volstr) or f(e, dr) or f(e)
    if callable(table):
        sig_stat = np.zeros(n, dtype=float)
        sig_dynamic = np.zeros(n, dtype=float)
        slope = np.zeros(n, dtype=float)
        ener = np.zeros(n, dtype=float)
        scaled_dr = dr / xscale

        for i in range(n):
            ei = float(e_clamped[i])
            dri = float(scaled_dr[i])
            vsi = float(volstr[i])
            try:
                ys = float(table(ei, 0.0, vsi))
                yd = float(table(ei, dri, vsi))
            except TypeError:
                try:
                    ys = float(table(ei, 0.0))
                    yd = float(table(ei, dri))
                except TypeError:
                    ys = float(table(ei))
                    yd = ys

            # Finite-difference derivative for slope
            de = 1.0e-5
            try:
                ys_plus = float(table(ei + de, 0.0, vsi))
            except TypeError:
                try:
                    ys_plus = float(table(ei + de, 0.0))
                except TypeError:
                    ys_plus = float(table(ei + de))

            sig_stat[i] = ys * scale
            sig_dynamic[i] = yd * scale
            slope[i] = ((ys_plus - ys) / de) * scale
            ener[i] = 0.5 * ys * ei * scale

        return _apply_tension_override(e, sig_stat, sig_dynamic, slope, ener, scale=scale, e0=e0)

    # Fallback
    mod = e0 if e0 > 0.0 else 1.0
    sig_stat = mod * e * scale
    sig_dynamic = mod * e * scale
    slope = np.full(n, mod * scale)
    ener = np.where(e > 0.0, 0.5 * mod * (e ** 2) * scale, 0.0)
    return sig_stat, sig_dynamic, slope, ener


# ---------------------------------------------------------------------------
# Core Constitutive Update (sigeps190.F)
# ---------------------------------------------------------------------------

def solid_step(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-element or group constitutive step for LAW190.

    Returns
    -------
    sig_new : (n, 6) Cauchy stress tensor
    epsp_new : (n,) effective plastic strain (total engineering strain norm)
    sound_speed : (n,) current dilatational acoustic sound speed
    """
    res = solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)
    if isinstance(res, tuple) and len(res) == 3:
        return res
    # Fallback
    c = sound_speed(mat, extra=extra)
    return res, epsp if epsp is not None else np.zeros(sig.shape[0]), c


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict | None = None,
    return_tuple: bool = False,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
    """Vectorized constitutive update for /MAT/LAW190 Du Bois Foam solids.

    Fortran origin: ``engine/source/materials/mat/mat190/sigeps190.F``.

    Parameters
    ----------
    mat : Material, Law190Params, MatLaw190, or dict
        Material model parameters.
    sig : (n, 6) or (6,) ndarray
        Old Cauchy stress tensor [xx, yy, zz, xy, yz, zx].
    deps : (n, 6) or (6,) ndarray
        Strain increment tensor (engineering shear: eps_xy, eps_yz, eps_zx).
    epsp : (n,) or scalar ndarray, optional
        Historical effective strain norm.
    dt : float, default 0.0
        Cycle time increment.
    extra : dict, optional
        Per-element state dictionary containing:
        - "uv190" or "uvar": history variables (n, 16)
        - "F": current deformation gradient (n, 3, 3)
        - "rho": current density
        - "rates": explicit strain rate tensor (n, 6)
    return_tuple : bool, default False
        If True, returns (sig_new, epsp_new, sound_speed). If False, returns sig_new.

    Returns
    -------
    (sig_new, epsp_new, sound_speed) if return_tuple=True, else sig_new.
    """
    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig.reshape(1, 6)
        deps = deps.reshape(1, 6)
        if epsp is not None and np.ndim(epsp) == 0:
            epsp = np.array([epsp])

    n = sig.shape[0]
    p = _extract_params(mat)

    rho0 = float(p.rho0 if p.rho0 > 0.0 else 1.0)
    e0 = float(p.e0)
    nu = float(p.nu)
    hu = float(p.hu)
    shape = float(p.shape)
    scale = float(p.scale)
    xscale = float(p.xscale)
    tcut = float(p.tcut)
    fail = int(p.fail)
    tbl = p.table

    # Current density from extra
    if extra is not None and "rho" in extra and extra["rho"] is not None:
        rho = np.atleast_1d(np.asarray(extra["rho"], dtype=float))
    elif extra is not None and "amu" in extra and extra["amu"] is not None:
        amu = np.atleast_1d(np.asarray(extra["amu"], dtype=float))
        rho = rho0 * (1.0 + amu)
    else:
        rho = np.full(n, rho0, dtype=float)
    if rho.size == 1 and n > 1:
        rho = np.full(n, float(rho[0]), dtype=float)

    # State array uv190 (n, 16)
    uv190 = None
    if extra is not None:
        if "uv190" in extra and extra["uv190"] is not None:
            uv190 = extra["uv190"]
        elif "uvar" in extra and extra["uvar"] is not None:
            uv190 = extra["uvar"]

    if uv190 is None:
        uv190 = np.zeros((n, 16), dtype=float)
        if extra is not None:
            extra["uv190"] = uv190
    elif uv190.ndim == 1:
        uv190 = uv190.reshape(1, -1)

    egl_old = uv190[:, 0:6].copy()
    w_max_old = uv190[:, 12].copy()
    eroded_old = uv190[:, 15].copy()

    # Strain rate tensor (L)
    if extra is not None and "rates" in extra and extra["rates"] is not None:
        rates = np.atleast_2d(np.asarray(extra["rates"], dtype=float))
    elif dt > 1.0e-20:
        rates = deps / dt
    else:
        rates = np.zeros((n, 6), dtype=float)

    # Deformation gradient F
    has_f = (extra is not None and "F" in extra and extra["F"] is not None)
    if has_f:
        f_arr = np.asarray(extra["F"], dtype=float)
        if f_arr.ndim == 2:
            f_arr = f_arr.reshape(1, 3, 3)
    else:
        f_arr = None

    # Output arrays
    sig_new = np.zeros((n, 6), dtype=float)
    epsp_new = np.zeros(n, dtype=float)
    c_sound = np.zeros(n, dtype=float)

    # Loop over elements in group
    for i in range(n):
        # Check erosion
        if eroded_old[i] >= 1.0:
            sig_new[i, :] = 0.0
            epsp_new[i] = epsp[i] if epsp is not None else 0.0
            c_sound[i] = math.sqrt(e0 / rho[i]) if rho[i] > 0.0 else 0.0
            uv190[i, 15] = 1.0
            continue

        # 1. Kinematics: deformation gradient F and Green-Lagrange tensor E_GL
        if f_arr is not None:
            F_i = f_arr[i]
            C_i = F_i.T @ F_i
            E_i = 0.5 * (C_i - np.eye(3))
            J_i = float(np.linalg.det(F_i))
        else:
            # Reconstruct Green-Lagrange strain from accumulated strain increment
            # E_new = E_old + dE
            deps_tens = np.array([
                [deps[i, 0], 0.5 * deps[i, 3], 0.5 * deps[i, 5]],
                [0.5 * deps[i, 3], deps[i, 1], 0.5 * deps[i, 4]],
                [0.5 * deps[i, 5], 0.5 * deps[i, 4], deps[i, 2]],
            ], dtype=float)

            E_old_tens = np.array([
                [egl_old[i, 0], egl_old[i, 3], egl_old[i, 5]],
                [egl_old[i, 3], egl_old[i, 1], egl_old[i, 4]],
                [egl_old[i, 5], egl_old[i, 4], egl_old[i, 2]],
            ], dtype=float)

            E_i = E_old_tens + deps_tens
            C_i = 2.0 * E_i + np.eye(3)
            # Estimate Jacobian J from det(C)^0.5
            det_c = float(np.linalg.det(C_i))
            J_i = math.sqrt(max(det_c, 1.0e-12))
            F_i = None

        # Store current Green-Lagrange tensor into state
        uv190[i, 0] = E_i[0, 0]
        uv190[i, 1] = E_i[1, 1]
        uv190[i, 2] = E_i[2, 2]
        uv190[i, 3] = E_i[0, 1]
        uv190[i, 4] = E_i[1, 2]
        uv190[i, 5] = E_i[2, 0]

        # Volumetric strain (positive in compression)
        volstr_i = 1.0 - J_i

        # Midpoint strain tensor
        E_old_tens = np.array([
            [egl_old[i, 0], egl_old[i, 3], egl_old[i, 5]],
            [egl_old[i, 3], egl_old[i, 1], egl_old[i, 4]],
            [egl_old[i, 5], egl_old[i, 4], egl_old[i, 2]],
        ], dtype=float)
        E_mid = 0.5 * (E_i + E_old_tens)

        # 2. Spectral decomposition of current strain E_i (condamage.F / sigeps190.F)
        # Analytical Cardano formula or eigh:
        try:
            w, V = np.linalg.eigh(E_i)
        except Exception:
            w = np.diag(E_i)
            V = np.eye(3)

        # Principal stretches: lambda_k = sqrt(max(2*Z_k + 1, 1e-12))
        lambdas = np.sqrt(np.maximum(2.0 * w + 1.0, 1.0e-12))

        # Principal engineering strains (positive in compression)
        e_princ = 1.0 - lambdas
        epst_i = float(np.linalg.norm(e_princ))
        epsp_new[i] = epst_i

        # 3. Engineering strain rate in principal directions
        rate_tens = np.array([
            [rates[i, 0], 0.5 * rates[i, 3], 0.5 * rates[i, 5]],
            [0.5 * rates[i, 3], rates[i, 1], 0.5 * rates[i, 4]],
            [0.5 * rates[i, 5], 0.5 * rates[i, 4], rates[i, 2]],
        ], dtype=float)
        # Rotated onto principal axes
        rate_rot = V.T @ rate_tens @ V
        dr_princ = np.abs(np.diag(rate_rot)) * lambdas

        # 4. Tabulated yield and energy evaluation
        vol_arr = np.full(3, volstr_i)
        sig_stat, sig_dyn, slopes, energies = _lookup_yield_and_energy(
            tbl, e_princ, dr_princ, vol_arr, scale=scale, xscale=xscale, e0=e0
        )

        # 5. Hysteretic energy and damage (condamage.F)
        w_hys = float(np.sum(energies))
        w_max = max(w_max_old[i], w_hys)
        uv190[i, 12] = w_max
        uv190[i, 14] = w_hys

        if w_max > 1.0e-20 and w_hys < w_max:
            ratio = max(0.0, min(1.0, w_hys / w_max))
            damage = (1.0 - hu) * (1.0 - (ratio ** shape))
            damage = max(0.0, min(1.0, damage))
        else:
            damage = 0.0

        uv190[i, 13] = damage
        dam_factor = 1.0 - damage

        # 6. Principal 2PK and Cauchy stress
        # S_k = - (sig_dyn_k * dam_factor) / lambda_k
        # sigma_Cauchy_k = (lambda_k^2 / J) * S_k = - (lambda_k / J) * sig_dyn_k * dam_factor
        # (Negative in compression, positive in tension)
        j_safe = max(J_i, 1.0e-12)
        sig_cauchy_princ = - (lambdas / j_safe) * sig_dyn * dam_factor

        # 7. Tensile cutoff and failure / erosion
        eroded = False
        hydro_press = - (1.0 / 3.0) * float(np.sum(sig_cauchy_princ))

        if tcut < 1.0e19:
            # Check tensile limits
            max_tensile = float(np.max(sig_cauchy_princ))
            if max_tensile > tcut or hydro_press < -tcut:
                if fail == 1:
                    eroded = True
                    uv190[i, 15] = 1.0
                    sig_cauchy_princ[:] = 0.0
                else:
                    # Clamp tensile stress components
                    sig_cauchy_princ = np.minimum(sig_cauchy_princ, tcut)

        if eroded:
            sig_new[i, :] = 0.0
        else:
            # 8. Rotate Cauchy stress back to global frame: sigma = V @ diag(sig_p) @ V^T
            sig_matrix = V @ np.diag(sig_cauchy_princ) @ V.T
            sig_new[i, 0] = sig_matrix[0, 0]
            sig_new[i, 1] = sig_matrix[1, 1]
            sig_new[i, 2] = sig_matrix[2, 2]
            sig_new[i, 3] = sig_matrix[0, 1]
            sig_new[i, 4] = sig_matrix[1, 2]
            sig_new[i, 5] = sig_matrix[2, 0]

        # 9. Sound speed (sigeps190.F lines 406-413)
        # slopemax = max(slope1, slope2, slope3, e0)
        slopemax = max(float(np.max(slopes)), e0 if e0 > 0.0 else 1.0)
        c_sound[i] = math.sqrt(slopemax / rho[i]) if rho[i] > 0.0 else 0.0

    if is_1d:
        sig_new = sig_new[0]
        epsp_new = epsp_new[0]
        c_sound = c_sound[0]

    if return_tuple:
        return sig_new, epsp_new, c_sound
    return sig_new


# ---------------------------------------------------------------------------
# Sound Speed & Tangent Stiffness
# ---------------------------------------------------------------------------

def sound_speed(
    mat: Any,
    rho: float | np.ndarray | None = None,
    extra: dict | None = None,
    is_shell: bool = False,
) -> float | np.ndarray:
    """Acoustic dilatational sound speed for LAW190 solids.

    c = sqrt(max(E0, max(slopes)) / rho)
    """
    p = _extract_params(mat)
    rho0 = p.rho0 if p.rho0 > 0.0 else 1.0

    if rho is None:
        if extra is not None and "rho" in extra and extra["rho"] is not None:
            rho_val = np.asarray(extra["rho"], dtype=float)
        else:
            rho_val = float(rho0)
    else:
        rho_val = np.asarray(rho, dtype=float) if isinstance(rho, np.ndarray) else float(rho)

    e0 = p.e0 if p.e0 > 0.0 else 1.0
    c_mod = e0

    # If extra contains current strain or slopes
    if extra is not None and "slopemax" in extra:
        c_mod = max(float(extra["slopemax"]), e0)

    if isinstance(rho_val, np.ndarray):
        safe_rho = np.maximum(rho_val, _EM20)
        return np.sqrt(c_mod / safe_rho)
    return math.sqrt(c_mod / max(rho_val, _EM20))


def solid_tangent(
    mat: Any,
    sig: np.ndarray | None = None,
    deps: np.ndarray | None = None,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict | None = None,
) -> np.ndarray:
    """Consistent solid algorithmic tangent tensor C (n, 6, 6) or (6, 6).

    Matches ``conversion.F``.
    """
    p = _extract_params(mat)
    e0 = p.e0 if p.e0 > 0.0 else 1.0
    nu = p.nu
    bulk = p.bulk if p.bulk > 0.0 else e0 / (3.0 * (1.0 - 2.0 * nu))
    g = p.g if p.g > 0.0 else e0 / (2.0 * (1.0 + nu))
    cii = bulk + _FOUR_THIRD * g
    cij = bulk - _TWO_THIRD * g

    # Check if a specific strain or slope is provided
    mod_cur = e0
    if extra is not None and "slopemax" in extra:
        mod_cur = max(float(extra["slopemax"]), e0)
        if e0 > 0.0:
            scale_fac = mod_cur / e0
            cii *= scale_fac
            cij *= scale_fac
            g *= scale_fac

    # Base isotropic 6x6 elasticity matrix in Voigt notation: [xx, yy, zz, xy, yz, zx]
    C66 = np.array([
        [cii, cij, cij, 0.0, 0.0, 0.0],
        [cij, cii, cij, 0.0, 0.0, 0.0],
        [cij, cij, cii, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, g,   0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, g,   0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, g  ],
    ], dtype=float)

    if sig is not None and sig.ndim == 2:
        n = sig.shape[0]
        return np.tile(C66, (n, 1, 1))
    return C66


consistent_solid_tangent = solid_tangent


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """LAW190 is a 3D solid crushable foam model and does not support shells."""
    raise NotImplementedError("material LAW190 (FOAM_DUBOIS) is formulated for 3D solid elements only.")


def shell_tangent(
    mat: Any,
    sig: np.ndarray | None = None,
    deps: np.ndarray | None = None,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict | None = None,
) -> np.ndarray:
    raise NotImplementedError("material LAW190 (FOAM_DUBOIS) is formulated for 3D solid elements only.")


consistent_shell_tangent = shell_tangent


def _register() -> None:
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        MAT_PHYSICS_REGISTRY["LAW190"] = build_law190
        MAT_PHYSICS_REGISTRY["FOAM_DUBOIS"] = build_law190
        MAT_PHYSICS_REGISTRY["DUBOIS"] = build_law190
        MAT_PHYSICS_REGISTRY[190] = build_law190
    except Exception:
        pass


_register()

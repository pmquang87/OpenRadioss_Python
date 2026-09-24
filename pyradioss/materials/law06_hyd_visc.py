"""
LAW06 — Hydrodynamic Viscous Fluid (/MAT/LAW6, /MAT/HYD_VISC, /MAT/HYDRO).

OpenRadioss /MAT/HYD_VISC (LAW06) Python implementation.

Upstream Fortran Reference:
  - Engine physics: engine/source/materials/mat/mat006/m6law.F (SUBROUTINE M6LAW, lines 30–146;
    also designated as sigeps06.F in standard law conventions)
  - Starter reader: starter/source/materials/mat/mat006/hm_read_mat06.F (SUBROUTINE HM_READ_MAT06, lines 38–264)
  - Parameter mapping: hm_cfg_files/config/CFG/Keyword971/MAT/mat_006.cfg

Theory & Formulation
--------------------
Hydrodynamic Newtonian fluid model with uncoupled volumetric and deviatoric responses:
1. Deviatoric Viscous Stress (Navier-Stokes Newtonian fluid):
       s_ij = 2 * eta * e_ij_dot
   where e_ij_dot = eps_ij_dot - (1/3) * tr(eps_dot) * delta_ij is the deviatoric strain rate,
   and dynamic viscosity eta = DAMP1 * rho (PM(24, MX) * RHO in m6law.F).
   In engineering Voigt notation [xx, yy, zz, xy, yz, zx]:
       s_xx = 2 * eta * (eps_xx_dot - (1/3) * tr(eps_dot))
       s_yy = 2 * eta * (eps_yy_dot - (1/3) * tr(eps_dot))
       s_zz = 2 * eta * (eps_zz_dot - (1/3) * tr(eps_dot))
       s_xy = eta * gamma_xy_dot
       s_yz = eta * gamma_yz_dot
       s_zx = eta * gamma_zx_dot

2. Volumetric Response (Equation of State):
   Hydrostatic pressure P is determined by an Equation of State (linear, polynomial,
   Mie-Gruneisen, Tait, etc.):
       sigma_ij = s_ij - P * delta_ij
   Embedded polynomial EOS computes:
       P = C0 + C1*mu + C2*mu^2 + C3*mu^3 + (C4 + C5*mu)*E   (compression mu >= 0)
       P = C0 + C1*mu + C3*mu^2 + C4*E                       (expansion mu < 0)
   with tensile pressure cutoff PMIN (preventing fluid cavitation/tensile instability).

3. Sound Speed:
       c = sqrt(K / rho0) = sqrt(bulk / rho0)
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from pyradioss.model.entities import EquationOfState, Material

_EM20 = 1e-20


# ============================================================================
# 1D & Analytical Helper Functions
# ============================================================================

def newtonian_shear_stress(
    rate: float | np.ndarray,
    visc: float,
    rho: Optional[float | np.ndarray] = None,
    kinematic: bool = False,
) -> float | np.ndarray:
    r"""Evaluate 1D Newtonian viscous shear stress:
        \tau = \eta \dot{\gamma} = 2 \eta \dot{\varepsilon}_{xy}
    where \eta is dynamic viscosity (or \eta = \nu \rho if kinematic=True).

    Cited from:
      - engine/source/materials/mat/mat006/m6law.F (lines 98, 132)
      - starter/source/materials/mat/mat006/hm_read_mat06.F (lines 110-111)

    Parameters:
        rate: Engineering shear strain rate \dot{\gamma} = d(gamma)/dt
        visc: Viscosity parameter DAMP1
        rho: Current density (optional, used if kinematic=True or in OpenRadioss scaling)
        kinematic: If True, visc is treated as kinematic viscosity \nu and scaled by rho
    """
    is_scalar = np.isscalar(rate)
    r_arr = np.asarray(rate, dtype=float)
    eta = float(visc)
    if kinematic and rho is not None:
        eta = eta * np.asarray(rho, dtype=float)
    tau = eta * r_arr
    return float(np.squeeze(tau)) if is_scalar else tau


def newtonian_deviatoric_stress(
    edot: Sequence[float] | np.ndarray,
    visc: float,
    rho: Optional[float | np.ndarray] = None,
    kinematic: bool = False,
) -> np.ndarray:
    r"""Evaluate 3D Newtonian viscous deviatoric stress tensor:
        s_{ij} = 2 \eta \dot{e}_{ij} = 2 \eta (\dot{\varepsilon}_{ij} - \frac{1}{3} \text{tr}(\dot{\varepsilon}) \delta_{ij})

    For Voigt vector [xx, yy, zz, xy, yz, zx] with engineering shear rates:
        s_xx = 2 \eta (\dot{\varepsilon}_{xx} - \frac{1}{3}\text{tr}(\dot{\varepsilon}))
        s_yy = 2 \eta (\dot{\varepsilon}_{yy} - \frac{1}{3}\text{tr}(\dot{\varepsilon}))
        s_zz = 2 \eta (\dot{\varepsilon}_{zz} - \frac{1}{3}\text{tr}(\dot{\varepsilon}))
        s_xy = \eta \dot{\gamma}_{xy}
        s_yz = \eta \dot{\gamma}_{yz}
        s_zx = \eta \dot{\gamma}_{zx}

    Cited from:
      - engine/source/materials/mat/mat006/m6law.F (lines 123-134)
    """
    edot_arr = np.asarray(edot, dtype=float)
    is_1d = (edot_arr.ndim == 1 and edot_arr.size == 6)
    if is_1d:
        edot_arr = edot_arr[None, :]

    eta = float(visc)
    if kinematic and rho is not None:
        rho_arr = np.atleast_1d(np.asarray(rho, dtype=float))
        eta = eta * rho_arr[:, None]

    tr3 = (edot_arr[:, 0] + edot_arr[:, 1] + edot_arr[:, 2]) / 3.0
    s = np.zeros_like(edot_arr)
    s[:, 0] = 2.0 * eta * (edot_arr[:, 0] - tr3)
    s[:, 1] = 2.0 * eta * (edot_arr[:, 1] - tr3)
    s[:, 2] = 2.0 * eta * (edot_arr[:, 2] - tr3)
    s[:, 3] = eta * edot_arr[:, 3]
    s[:, 4] = eta * edot_arr[:, 4]
    s[:, 5] = eta * edot_arr[:, 5]

    return s[0] if is_1d else s


def linear_eos_pressure(
    rho: float | np.ndarray,
    rho0: float,
    bulk: float,
    pmin: Optional[float] = None,
) -> float | np.ndarray:
    r"""Evaluate linear hydrodynamic equation of state pressure:
        P = K \mu = K (\rho / \rho_0 - 1)
    clamped at tensile cutoff P >= P_min.

    Cited from:
      - engine/source/materials/mat/mat006/m6law.F
      - starter/source/materials/mat/mat006/hm_read_mat06.F (line 99)
    """
    is_scalar = np.isscalar(rho)
    rho_arr = np.asarray(rho, dtype=float)
    mu = rho_arr / max(float(rho0), 1e-20) - 1.0
    p = float(bulk) * mu
    if pmin is not None:
        p = np.maximum(p, float(pmin))
    return float(np.squeeze(p)) if is_scalar else p


def polynomial_eos_pressure(
    mu: float | np.ndarray,
    c0: float = 0.0,
    c1: float = 0.0,
    c2: float = 0.0,
    c3: float = 0.0,
    c4: float = 0.0,
    c5: float = 0.0,
    e0: float = 0.0,
    pmin: Optional[float] = None,
) -> float | np.ndarray:
    r"""Evaluate polynomial hydrodynamic equation of state pressure:
        Compression (\mu >= 0): P = C_0 + C_1 \mu + C_2 \mu^2 + C_3 \mu^3 + (C_4 + C_5 \mu) E_0
        Expansion   (\mu < 0):  P = C_0 + C_1 \mu + C_3 \mu^2 + C_4 E_0
    clamped at tensile cutoff P >= P_min.

    Cited from:
      - engine/source/materials/mat/mat006/m6law.F
      - starter/source/materials/mat/mat006/hm_read_mat06.F (lines 116-141)
    """
    is_scalar = np.isscalar(mu)
    mu_arr = np.asarray(mu, dtype=float)
    c0_f, c1_f, c2_f = float(c0), float(c1), float(c2)
    c3_f, c4_f, c5_f = float(c3), float(c4), float(c5)
    e0_f = float(e0)

    p_comp = c0_f + c1_f * mu_arr + c2_f * (mu_arr ** 2) + c3_f * (mu_arr ** 3) + (c4_f + c5_f * mu_arr) * e0_f
    p_exp = c0_f + c1_f * mu_arr + c3_f * (mu_arr ** 2) + c4_f * e0_f
    p = np.where(mu_arr >= 0.0, p_comp, p_exp)
    if pmin is not None:
        p = np.maximum(p, float(pmin))
    return float(np.squeeze(p)) if is_scalar else p


def hydrodynamic_fluid_stress(
    deps: Sequence[float] | np.ndarray,
    dt: float,
    visc: float,
    rho: float,
    p_eos: float = 0.0,
) -> np.ndarray:
    r"""Evaluate total Cauchy stress tensor for hydrodynamic fluid:
        \sigma_{ij} = s_{ij} - P \delta_{ij}
    where s_{ij} is Newtonian deviatoric viscous stress and P is hydrostatic pressure.

    Cited from:
      - engine/source/materials/mat/mat006/m6law.F (lines 123-134)
    """
    deps_arr = np.asarray(deps, dtype=float)
    is_1d = (deps_arr.ndim == 1 and deps_arr.size == 6)
    if is_1d:
        deps_arr = deps_arr[None, :]

    rate = deps_arr / max(float(dt), 1e-20) if dt > 1e-20 else np.zeros_like(deps_arr)
    s = newtonian_deviatoric_stress(rate, visc=visc, rho=rho, kinematic=True)
    sig = s.copy()
    sig[:, 0] -= float(p_eos)
    sig[:, 1] -= float(p_eos)
    sig[:, 2] -= float(p_eos)
    return sig[0] if is_1d else sig


# ============================================================================
# Material Parameter Normalization & Construction
# ============================================================================

def _ensure_params(mat: Material) -> dict:
    """Ensure material params contain both CFG and direct keys with robust defaults."""
    p = mat.params if mat.params is not None else {}
    visc = float(p.get("visc") if p.get("visc") is not None else (p.get("DAMP1") or p.get("mu") or p.get("eta") or 0.0))
    pmin = float(p.get("pmin") if p.get("pmin") is not None else (p.get("MAT_PC") or 0.0))
    if pmin == 0.0 and ("MAT_PC" in p or "pmin" in p):
        # In OpenRadioss Fortran hm_read_mat06.F: IF (PMIN == ZERO) PMIN=-INFINITY
        pmin = -1e30

    psh = float(p.get("psh") if p.get("psh") is not None else (p.get("MAT_PSH") or 0.0))
    e0 = float(p.get("e0") if p.get("e0") is not None else (p.get("MAT_EA") or 0.0))
    c0 = float(p.get("c0") if p.get("c0") is not None else (p.get("MAT_C0") or 0.0))
    c1 = float(p.get("c1") if p.get("c1") is not None else (p.get("MAT_C1") or 0.0))
    c2 = float(p.get("c2") if p.get("c2") is not None else (p.get("MAT_C2") or 0.0))
    c3 = float(p.get("c3") if p.get("c3") is not None else (p.get("MAT_C3") or 0.0))
    c4 = float(p.get("c4") if p.get("c4") is not None else (p.get("MAT_C4") or 0.0))
    c5 = float(p.get("c5") if p.get("c5") is not None else (p.get("MAT_C5") or 0.0))
    law6_opt = int(p.get("law6_opt") if p.get("law6_opt") is not None else (p.get("Law6_opt") or 1))

    p.setdefault("visc", visc)
    p.setdefault("pmin", pmin)
    p.setdefault("psh", psh)
    p.setdefault("e0", e0)
    p.setdefault("c0", c0)
    p.setdefault("c1", c1)
    p.setdefault("c2", c2)
    p.setdefault("c3", c3)
    p.setdefault("c4", c4)
    p.setdefault("c5", c5)
    p.setdefault("law6_opt", law6_opt)
    p.setdefault("E", 0.0)
    p.setdefault("nu", 0.0)

    # CFG mirrored keys
    p.setdefault("DAMP1", visc)
    p.setdefault("MAT_PC", pmin)
    p.setdefault("MAT_PSH", psh)
    p.setdefault("MAT_EA", e0)
    p.setdefault("MAT_C0", c0)
    p.setdefault("MAT_C1", c1)
    p.setdefault("MAT_C2", c2)
    p.setdefault("MAT_C3", c3)
    p.setdefault("MAT_C4", c4)
    p.setdefault("MAT_C5", c5)
    p.setdefault("Law6_opt", law6_opt)
    return p


def build_law6(rec) -> Material:
    """hm_read_mat06.F: card parsing and validation -> Material."""
    if isinstance(rec, Material):
        p = rec.params
        mat_id = rec.id
        title = rec.title
        density = rec.rho0
        eos = rec.eos
    elif isinstance(rec, dict):
        p = rec.get("params", rec)
        mat_id = rec.get("id", 1)
        title = rec.get("title", "")
        val = None
        for k in ("density", "rho", "rho0", "MAT_RHO"):
            if k in rec and rec[k] is not None:
                val = rec[k]
                break
        if val is None and isinstance(p, dict):
            for k in ("density", "rho", "rho0", "MAT_RHO"):
                if k in p and p[k] is not None:
                    val = p[k]
                    break
        density = float(val) if val is not None else 1.0
        eos = rec.get("eos")
    else:
        p = getattr(rec, "params", {})
        mat_id = getattr(rec, "id", 1)
        title = getattr(rec, "title", "")
        val = None
        for k in ("density", "rho", "rho0", "MAT_RHO"):
            if hasattr(rec, k) and getattr(rec, k) is not None:
                val = getattr(rec, k)
                break
        if val is None and isinstance(p, dict):
            for k in ("density", "rho", "rho0", "MAT_RHO"):
                if k in p and p[k] is not None:
                    val = p[k]
                    break
        density = float(val) if val is not None else (getattr(rec, "density", None) or 1.0)
        eos = getattr(rec, "eos", None)

    if density <= 0.0:
        raise ValueError("LAW6: Density rho0 must be positive (hm_read_mat06 error)")

    visc = float(p.get("DAMP1") if p.get("DAMP1") is not None else (p.get("visc") if p.get("visc") is not None else (p.get("mu") or p.get("eta") or 0.0)))
    if visc < 0.0:
        raise ValueError("LAW6: Viscosity parameter DAMP1 must be non-negative")

    # Check for embedded polynomial EOS: MAT_C0..MAT_C5 or c0..c5
    c0 = p.get("MAT_C0") if p.get("MAT_C0") is not None else p.get("c0")
    c1 = p.get("MAT_C1") if p.get("MAT_C1") is not None else p.get("c1")
    if eos is None and (c0 is not None or c1 is not None):
        c0_val = float(c0 or 0.0)
        c1_val = float(c1 or 0.0)
        c2_val = float(p.get("MAT_C2") if p.get("MAT_C2") is not None else (p.get("c2") or 0.0))
        c3_val = float(p.get("MAT_C3") if p.get("MAT_C3") is not None else (p.get("c3") or 0.0))
        c4_val = float(p.get("MAT_C4") if p.get("MAT_C4") is not None else (p.get("c4") or 0.0))
        c5_val = float(p.get("MAT_C5") if p.get("MAT_C5") is not None else (p.get("c5") or 0.0))
        e0_val = float(p.get("MAT_EA") if p.get("MAT_EA") is not None else (p.get("e0") or 0.0))
        psh_val = float(p.get("MAT_PSH") if p.get("MAT_PSH") is not None else (p.get("psh") or 0.0))
        pmin_val = float(p.get("MAT_PC") if p.get("MAT_PC") is not None else (p.get("pmin") or 0.0))
        eos_params = {
            "c0": c0_val,
            "c1": c1_val,
            "c2": c2_val,
            "c3": c3_val,
            "c4": c4_val,
            "c5": c5_val,
            "e0": e0_val,
            "psh": psh_val,
            "pmin": pmin_val,
        }
        eos = EquationOfState(kind="POLYNOMIAL", params=eos_params, rho0=density)

    pmin = float(p.get("MAT_PC") if p.get("MAT_PC") is not None else (p.get("pmin") or 0.0))
    if pmin == 0.0 and ("MAT_PC" in p or "pmin" in p):
        pmin = -1e30

    params = {
        "visc": visc,
        "E": 0.0,
        "nu": 0.0,
        "pmin": pmin,
        "psh": float(p.get("MAT_PSH") if p.get("MAT_PSH") is not None else (p.get("psh") or 0.0)),
        "c0": float(c0 or 0.0),
        "c1": float(c1 or 0.0),
        "c2": float(p.get("MAT_C2") if p.get("MAT_C2") is not None else (p.get("c2") or 0.0)),
        "c3": float(p.get("MAT_C3") if p.get("MAT_C3") is not None else (p.get("c3") or 0.0)),
        "c4": float(p.get("MAT_C4") if p.get("MAT_C4") is not None else (p.get("c4") or 0.0)),
        "c5": float(p.get("MAT_C5") if p.get("MAT_C5") is not None else (p.get("c5") or 0.0)),
        "e0": float(p.get("MAT_EA") if p.get("MAT_EA") is not None else (p.get("e0") or 0.0)),
        "law6_opt": int(p.get("Law6_opt") if p.get("Law6_opt") is not None else (p.get("law6_opt") or 1)),
    }
    mat = Material(id=mat_id, law=6, rho0=density, title=title, params=params, eos=eos)
    _ensure_params(mat)
    return mat


# ============================================================================
# Constitutive Stress Updates
# ============================================================================

def solid_update(mat, sig: np.ndarray, deps: np.ndarray, epsp=None, dt: float = 0.0, extra: dict = None):
    """Update solid deviatoric stress for LAW6 (Newtonian fluid).

    Fortran origin: ``engine/source/materials/mat/mat006/m6law.F`` (lines 30-146).

    Args:
        mat: Material object containing visc parameter.
        sig: Deviatoric stress tensor [nel, 6]. Modified in place.
        deps: Strain increment tensor [nel, 6].
        epsp: Not used for LAW6.
        dt: Time step.
        extra: Dictionary containing "rho" (current density) among other things.

    Returns:
        (sig, epsp, c) where c is sound speed (or None if external EOS is used).
    """
    nel = sig.shape[0]
    p = _ensure_params(mat)
    rho0 = float(getattr(mat, "rho0", 1.0) or 1.0)

    if nel == 0:
        c_empty = None
        if getattr(mat, "eos", None) is not None:
            c1 = float(getattr(mat.eos, "params", {}).get("c1") or p.get("c1") or 0.0)
            e0 = float(getattr(mat.eos, "params", {}).get("e0") or p.get("e0") or 0.0)
            c4 = float(getattr(mat.eos, "params", {}).get("c4") or p.get("c4") or 0.0)
            bulk = max(c1 + c4 * abs(e0), 0.0)
            if bulk > 0.0 and rho0 > 0.0:
                c_empty = np.zeros(0, dtype=float)
        return sig, epsp, c_empty

    # Defensive extra dict handling
    if extra is None:
        extra = {}
    current_rho = extra.get("rho")
    if current_rho is None:
        current_rho = np.full(nel, rho0)
    elif np.isscalar(current_rho):
        current_rho = np.full(nel, float(current_rho))
    elif len(current_rho) != nel:
        current_rho = np.full(nel, rho0)

    if dt > 1e-20:
        deps_rate = deps / dt
    else:
        deps_rate = np.zeros_like(deps)

    # Volumetric strain rate (dav = -Dii / 3 in Fortran m6law.F)
    dav = -(deps_rate[:, 0] + deps_rate[:, 1] + deps_rate[:, 2]) / 3.0

    # Viscosity is scaled by current density: VIS(I) = PM(24,MX)*RHO(I)
    visc = p["visc"] * current_rho
    vis2 = 2.0 * visc

    # Compute viscous stress deviator
    sig[:, 0] = vis2 * (deps_rate[:, 0] + dav)
    sig[:, 1] = vis2 * (deps_rate[:, 1] + dav)
    sig[:, 2] = vis2 * (deps_rate[:, 2] + dav)

    # Shear components (engineering strain rate components)
    sig[:, 3] = visc * deps_rate[:, 3]
    sig[:, 4] = visc * deps_rate[:, 4]
    sig[:, 5] = visc * deps_rate[:, 5]

    # Sound speed: if embedded EOS is present, calculate c = sqrt(bulk / rho0)
    c = None
    if getattr(mat, "eos", None) is not None:
        c1 = float(getattr(mat.eos, "params", {}).get("c1") or p.get("c1") or 0.0)
        e0 = float(getattr(mat.eos, "params", {}).get("e0") or p.get("e0") or 0.0)
        c4 = float(getattr(mat.eos, "params", {}).get("c4") or p.get("c4") or 0.0)
        bulk = max(c1 + c4 * abs(e0), 0.0)
        if bulk > 0.0 and rho0 > 0.0:
            c = np.full(nel, np.sqrt(bulk / rho0))

    return sig, epsp, c


def shell_update(mat, sig, deps, epsp=None, dt=0.0, extra=None):
    """LAW6 is defined strictly for solid/SPH fluid elements."""
    raise NotImplementedError("LAW6 (hydrodynamic viscous fluid) is implemented for solid/SPH elements only.")


def consistent_solid_tangent(mat, sig: np.ndarray, epsp=None, epsp_incr=None, dt: float = 0.0, extra: dict = None) -> np.ndarray:
    """Consistent 6x6 solid algorithmic tangent for LAW6 fluid.

    Includes embedded EOS bulk modulus K and viscous deviatoric modulus:
    D = K (1 (x) 1) + 2 G_visc (I_dev)
    where G_visc = eta / dt for dt > 0.
    """
    nel = sig.shape[0]
    if nel == 0:
        return np.zeros((0, 6, 6), dtype=float)

    p = _ensure_params(mat)
    rho0 = float(getattr(mat, "rho0", 1.0) or 1.0)

    if extra is None:
        extra = {}
    current_rho = extra.get("rho")
    if current_rho is None:
        current_rho = np.full(nel, rho0)
    elif np.isscalar(current_rho):
        current_rho = np.full(nel, float(current_rho))
    elif len(current_rho) != nel:
        current_rho = np.full(nel, rho0)

    if dt <= 0.0 and "dt" in extra:
        dt = float(extra["dt"])

    # Bulk modulus from embedded EOS or curve if available
    bulk = 0.0
    if getattr(mat, "eos", None) is not None:
        c1 = float(getattr(mat.eos, "params", {}).get("c1") or p.get("c1") or 0.0)
        e0 = float(getattr(mat.eos, "params", {}).get("e0") or p.get("e0") or 0.0)
        c4 = float(getattr(mat.eos, "params", {}).get("c4") or p.get("c4") or 0.0)
        bulk = max(c1 + c4 * abs(e0), 0.0)
    elif p.get("c1", 0.0) > 0.0:
        bulk = float(p["c1"])

    g_visc = (p["visc"] * current_rho / dt) if dt > 1e-20 else np.zeros(nel, dtype=float)

    D = np.zeros((nel, 6, 6), dtype=float)
    # Volumetric block: K on (0..2, 0..2)
    if bulk > 0.0:
        for i in range(3):
            for j in range(3):
                D[:, i, j] += bulk

    # Deviatoric block: 2 G_visc (I - 1/3 (1 (x) 1))
    for i in range(3):
        for j in range(3):
            if i == j:
                D[:, i, j] += (4.0 / 3.0) * g_visc
            else:
                D[:, i, j] -= (2.0 / 3.0) * g_visc

    # Shear components: G_visc
    for s in (3, 4, 5):
        D[:, s, s] += g_visc

    return D


def _register():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    except (ImportError, ValueError):
        try:
            from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
        except ImportError:
            return
    MAT_PHYSICS_REGISTRY["LAW6"] = build_law6
    MAT_PHYSICS_REGISTRY["HYD_VISC"] = build_law6
    MAT_PHYSICS_REGISTRY["HYDRO"] = build_law6


_register()

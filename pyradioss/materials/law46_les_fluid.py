# Ported from OpenRadioss Fortran:
# Source: starter/source/materials/mat/mat046/hm_read_mat46.F (HM_READ_MAT46)
# Source: engine/source/materials/mat/mat046/m46law.F (M46LAW)
# Source: engine/source/materials/mat/mat046/sigeps46.F (SIGEPS46)
# Config: hm_cfg_files/config/CFG/radioss110/MAT/matl46_les_fluid.cfg
"""
LAW46 / LES_FLUID — Viscous Fluid / Foam Model with Smagorinsky SGS Turbulence.
(/MAT/LAW46, /MAT/LES_FLUID).

Upstream Fortran References:
- Starter reader: ``starter/source/materials/mat/mat046/hm_read_mat46.F``
- Engine material driver: ``engine/source/materials/mat/mat046/m46law.F``
- Stress update & SGS formulation: ``engine/source/materials/mat/mat046/sigeps46.F``
- Card layout: ``hm_cfg_files/config/CFG/radioss110/MAT/matl46_les_fluid.cfg``

Formulation:
1. Hydrodynamic pressure (Equation of State):
   P = - C1 * (1 - rho / rho0) = C1 * (rho / rho0 - 1)
   where C1 = rho_r * c^2 (bulk modulus K = rho_0 * c^2), c = speed of sound (MAT_C).
   In tension sign convention:
   sigma_hydro = C1 * (1 - rho / rho0)

2. Strain rate and deviator:
   eps_dot = deps / dt
   D_av = 1/3 * tr(eps_dot) = 1/3 * (Dxx + Dyy + Dzz)
   dev_D = eps_dot - D_av * I
   SS = sqrt(2 * (Dxx^2 + Dyy^2 + Dzz^2 + Dxy^2 + Dyz^2 + Dzx^2))
   where Dxy = 0.5 * gamma_xy, etc. (tensorial shear strain rates, sigeps46.F:158-161).

3. Sub-grid scale (SGS) Smagorinsky eddy viscosity:
   Filter scale:
     If Istf == 3:  Delta = deltax^2
     Else (3D):     Delta = Vol^(2/3)
     Else (2D):     Delta = Area
   Eddy viscosity:
     nu_1 = nu + Smag^2 * SS * Delta      (for Istf >= 1, Smag default = 0.1)
     nu_2 = CA * nu_1                     (for Istf >= 2, CA = (cps / Smag)^2)
     mu_1 = rho * nu_1
     mu_2 = 3 * rho * nu_2
     P_visc = mu_2 * D_av                 (acoustic pressure damping)

4. Viscous and total Cauchy stress:
   sigma_v = 2 * mu_1 * dev_D + P_visc * I
   sigma_total = sigma_hydro * I + sigma_v

5. Sound speed:
   c_sound = sqrt(C1 / rho)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material, MaterialLaw46

_EM20 = 1.0e-20
_EM30 = 1.0e-30


@dataclass
class Law46Params:
    """Parameters for /MAT/LAW46 (/MAT/LES_FLUID)."""

    id: int = 1
    title: str = ""
    rho0: float = 1.0
    ref_rho: float = 0.0
    c: float = 0.0          # speed of sound (MAT_C)
    nu: float = 0.0         # molecular kinematic viscosity (MAT_NU)
    istf: int = 1           # subgrid model: 0=none, 1=Smagorinsky, 2=acoustic SGS, 3=modified SGS
    smag: float = 0.1       # Smagorinsky constant (MAT_C5)
    cps: float = 0.0        # pressure damping coefficient (MAT_CO1)

    @property
    def c1(self) -> float:
        """Bulk modulus C1 = rho_r * c^2 (hm_read_mat46.F:137)."""
        r = self.ref_rho if self.ref_rho > 0.0 else self.rho0
        return r * (self.c ** 2)

    @property
    def smag2(self) -> float:
        """Smag squared (hm_read_mat46.F:122-130)."""
        if self.istf == 0:
            return 0.0
        s = self.smag if self.smag > 0.0 else 0.1
        return s ** 2

    @property
    def ca(self) -> float:
        """Acoustic damping factor CA = (cps / smag)^2 (hm_read_mat46.F:131-136)."""
        if self.istf >= 2:
            s = self.smag if self.smag > 0.0 else 0.1
            c = self.cps if self.cps > 0.0 else s
            return (c / s) ** 2
        return 0.0


def _ensure_params(mat: Any) -> Law46Params:
    """Extract or construct a Law46Params instance from material object or dict."""
    if isinstance(mat, Law46Params):
        return mat

    if isinstance(mat, MaterialLaw46):
        rho0 = float(mat.rho0 or 1.0)
        ref_rho = float(mat.ref_rho or rho0)
        c = float(mat.c or 0.0)
        nu = float(mat.nu or 0.0)
        istf = int(mat.istf if mat.istf is not None else 1)
        smag = float(mat.smag or 0.1)
        cps = float(mat.cps or 0.0)
        return Law46Params(
            id=int(mat.id or 1),
            title=str(mat.title or ""),
            rho0=rho0,
            ref_rho=ref_rho,
            c=c,
            nu=nu,
            istf=istf,
            smag=smag,
            cps=cps,
        )

    # Dictionary or generic Material entity
    p: Dict[str, Any] = {}
    mat_id = 1
    title = ""
    rho0 = 1.0

    if hasattr(mat, "params") and isinstance(mat.params, dict):
        p = mat.params
        mat_id = getattr(mat, "id", 1) or 1
        title = getattr(mat, "title", "") or ""
        rho0 = float(getattr(mat, "rho0", None) or p.get("MAT_RHO") or p.get("rho0") or 1.0)
    elif isinstance(mat, dict):
        p = mat
        mat_id = int(p.get("id", 1))
        title = str(p.get("title", ""))
        rho0 = float(p.get("rho0") or p.get("MAT_RHO") or 1.0)
    elif hasattr(mat, "record"):
        rec = mat.record
        p = getattr(rec, "params", {}) or {}
        mat_id = getattr(rec, "id", 1) or 1
        title = getattr(rec, "title", "") or ""
        rho0 = float(getattr(rec, "density", None) or p.get("MAT_RHO") or p.get("rho0") or 1.0)

    ref_rho = float(p.get("Refer_Rho") or p.get("ref_rho") or rho0)
    c = float(p.get("MAT_C") if p.get("MAT_C") is not None else (p.get("c") or 0.0))
    nu = float(p.get("MAT_NU") if p.get("MAT_NU") is not None else (p.get("nu") or p.get("visc") or 0.0))
    istf = int(p.get("Istf") if p.get("Istf") is not None else (p.get("istf") if p.get("istf") is not None else 1))
    smag = float(p.get("MAT_C5") if p.get("MAT_C5") is not None else (p.get("smag") or 0.1))
    cps = float(p.get("MAT_CO1") if p.get("MAT_CO1") is not None else (p.get("cps") or 0.0))

    if istf >= 1 and smag <= 0.0:
        smag = 0.1
    if istf >= 2 and cps <= 0.0:
        cps = smag

    return Law46Params(
        id=mat_id,
        title=title,
        rho0=rho0,
        ref_rho=ref_rho,
        c=c,
        nu=nu,
        istf=istf,
        smag=smag,
        cps=cps,
    )


# ============================================================================
# Sound Speed
# ============================================================================

def sound_speed(mat: Any, rho: Optional[Any] = None, extra: Optional[dict] = None, is_shell: bool = False) -> Any:
    """Compute sound speed c = sqrt(C1 / rho) (sigeps46.F:149)."""
    p = _ensure_params(mat)
    c1 = p.c1
    if rho is None:
        if extra and "rho" in extra:
            rho = extra["rho"]
        else:
            rho = p.rho0

    if isinstance(rho, np.ndarray):
        return np.sqrt(c1 / np.maximum(rho, _EM30))
    return math.sqrt(c1 / max(float(rho), _EM30))


# ============================================================================
# Constitutive Stress Updates
# ============================================================================

def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[dict] = None,
    return_sound_speed: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Update Cauchy stress for 3D solid elements with LAW46 (/MAT/LES_FLUID).

    Fortran origins:
      - ``engine/source/materials/mat/mat046/m46law.F`` (M46LAW)
      - ``engine/source/materials/mat/mat046/sigeps46.F`` (SIGEPS46)

    Args:
        mat: Material object or parameter structure.
        sig: In/out Cauchy stress array of shape [nel, 6] in Voigt order [xx, yy, zz, xy, yz, zx].
        deps: Strain increment tensor [nel, 6].
        epsp: In/out internal variable array [nel]. Unused or preserves plastic strain.
        dt: Current time step.
        extra: Environment dict with 'rho', 'vol', 'deltax', etc.
        return_sound_speed: Whether to return per-element sound speeds.

    Returns:
        (sig, epsp, c_sound)
    """
    nel = sig.shape[0]
    p = _ensure_params(mat)
    rho0 = p.rho0 if p.rho0 > 0.0 else 1.0
    c1 = p.c1

    if nel == 0:
        c_empty = np.zeros(0, dtype=float) if return_sound_speed else None
        return sig, epsp, c_empty

    if extra is None:
        extra = {}

    current_rho = extra.get("rho")
    if current_rho is None:
        current_rho = np.full(nel, rho0)
    elif np.isscalar(current_rho):
        current_rho = np.full(nel, float(current_rho))
    elif len(current_rho) != nel:
        current_rho = np.full(nel, rho0)

    # 1. Strain rates: d_rate = deps / dt (sigeps46.F)
    if dt > _EM20:
        d_rate = deps / dt
    else:
        d_rate = np.zeros_like(deps)

    # Tensorial components: engineering shear rates / 2 (sigeps46.F:158-160)
    dxx = d_rate[:, 0]
    dyy = d_rate[:, 1]
    dzz = d_rate[:, 2]
    dxy = 0.5 * d_rate[:, 3]
    dyz = 0.5 * d_rate[:, 4]
    dzx = 0.5 * d_rate[:, 5]

    # Mean strain rate: dav = 1/3 * (Dxx + Dyy + Dzz) (sigeps46.F:157)
    dav = (dxx + dyy + dzz) / 3.0

    # Deviatoric strain rates: D' = D - dav * I (sigeps46.F:162-164)
    dev_xx = dxx - dav
    dev_yy = dyy - dav
    dev_zz = dzz - dav

    # Strain rate invariant: SS = sqrt(2*(dxx^2 + dyy^2 + dzz^2 + dxy^2 + dyz^2 + dzx^2)) (sigeps46.F:161)
    ss = np.sqrt(2.0 * (dxx**2 + dyy**2 + dzz**2 + dxy**2 + dyz**2 + dzx**2))

    # Characteristic length / filter size delta (sigeps46.F:168-177)
    # N2D == 0 for 3D solid elements
    if p.istf == 3:
        deltax = extra.get("deltax")
        if deltax is not None:
            if np.isscalar(deltax):
                deltax = np.full(nel, float(deltax))
            delta = deltax ** 2
        else:
            delta = np.ones(nel)
    else:
        vol = extra.get("vol")
        if vol is not None:
            if np.isscalar(vol):
                vol = np.full(nel, float(vol))
            delta = np.maximum(vol, _EM30) ** (2.0 / 3.0)
        else:
            deltax = extra.get("deltax")
            if deltax is not None:
                if np.isscalar(deltax):
                    deltax = np.full(nel, float(deltax))
                delta = deltax ** 2
            else:
                delta = np.ones(nel)

    # Sub-grid scale turbulent viscosity (sigeps46.F:190-193)
    smag2 = p.smag2
    vis1 = p.nu + smag2 * ss * delta
    ca = p.ca
    vis2 = ca * vis1

    # Scaling by current density (sigeps46.F:220-224)
    vis1 = current_rho * vis1
    vis2 = 3.0 * current_rho * vis2
    viscpression = vis2 * dav

    # Dynamic shear viscosity factor 2*mu (sigeps46.F:225)
    vis1 = 2.0 * vis1

    # Viscous stress components (sigeps46.F:226-231)
    sigv_xx = vis1 * dev_xx + viscpression
    sigv_yy = vis1 * dev_yy + viscpression
    sigv_zz = vis1 * dev_zz + viscpression
    sigv_xy = vis1 * dxy
    sigv_yz = vis1 * dyz
    sigv_zx = vis1 * dzx

    # Hydrostatic pressure (sigeps46.F:142-150)
    # sign_xx = C1 * (1 - rho / rho0)
    sign_hydro = c1 * (1.0 - current_rho / rho0)

    # Total Cauchy stress (sigeps46.F:143 + m46law.F:155-160)
    sig[:, 0] = sign_hydro + sigv_xx
    sig[:, 1] = sign_hydro + sigv_yy
    sig[:, 2] = sign_hydro + sigv_zz
    sig[:, 3] = sigv_xy
    sig[:, 4] = sigv_yz
    sig[:, 5] = sigv_zx

    if return_sound_speed:
        c_sound = np.sqrt(c1 / np.maximum(current_rho, _EM30))
    else:
        c_sound = None

    return sig, epsp, c_sound


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[dict] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Update stress for shell / plane elements using LAW46 (sigeps46.F:179-187)."""
    nel = sig.shape[0]
    p = _ensure_params(mat)
    rho0 = p.rho0 if p.rho0 > 0.0 else 1.0
    c1 = p.c1

    if nel == 0:
        return sig, epsp, np.zeros(0, dtype=float)

    if extra is None:
        extra = {}

    current_rho = extra.get("rho")
    if current_rho is None:
        current_rho = np.full(nel, rho0)
    elif np.isscalar(current_rho):
        current_rho = np.full(nel, float(current_rho))

    if dt > _EM20:
        d_rate = deps / dt
    else:
        d_rate = np.zeros_like(deps)

    dxx = d_rate[:, 0]
    dyy = d_rate[:, 1]
    dzz = d_rate[:, 2] if d_rate.shape[1] > 2 else np.zeros(nel)
    dxy = 0.5 * d_rate[:, 3] if d_rate.shape[1] > 3 else np.zeros(nel)
    dyz = 0.5 * d_rate[:, 4] if d_rate.shape[1] > 4 else np.zeros(nel)
    dzx = 0.5 * d_rate[:, 5] if d_rate.shape[1] > 5 else np.zeros(nel)

    dav = (dxx + dyy + dzz) / 3.0
    dev_xx = dxx - dav
    dev_yy = dyy - dav
    dev_zz = dzz - dav
    ss = np.sqrt(2.0 * (dxx**2 + dyy**2 + dzz**2 + dxy**2 + dyz**2 + dzx**2))

    # In 2D/shell mode: DELTA = AIRE (sigeps46.F:185)
    area = extra.get("area") or extra.get("aire")
    if area is not None:
        if np.isscalar(area):
            area = np.full(nel, float(area))
        delta = area
    else:
        deltax = extra.get("deltax")
        delta = deltax**2 if deltax is not None else np.ones(nel)

    smag2 = p.smag2
    vis1 = p.nu + smag2 * ss * delta
    ca = p.ca
    vis2 = ca * vis1

    vis1 = current_rho * vis1
    vis2 = 3.0 * current_rho * vis2
    viscpression = vis2 * dav
    vis1 = 2.0 * vis1

    sigv_xx = vis1 * dev_xx + viscpression
    sigv_yy = vis1 * dev_yy + viscpression
    sigv_zz = vis1 * dev_zz + viscpression
    sigv_xy = vis1 * dxy

    sign_hydro = c1 * (1.0 - current_rho / rho0)

    sig[:, 0] = sign_hydro + sigv_xx
    sig[:, 1] = sign_hydro + sigv_yy
    if sig.shape[1] > 2:
        sig[:, 2] = sign_hydro + sigv_zz
    if sig.shape[1] > 3:
        sig[:, 3] = sigv_xy

    c_sound = np.sqrt(c1 / np.maximum(current_rho, _EM30))
    return sig, epsp, c_sound


# ============================================================================
# Tangent Operators (Implicit branch)
# ============================================================================

def solid_tangent(
    mat: Any,
    deps: Optional[np.ndarray] = None,
    sig: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[dict] = None,
) -> np.ndarray:
    """Return [6, 6] elastic/viscous continuum tangent for implicit solver."""
    p = _ensure_params(mat)
    rho0 = p.rho0 if p.rho0 > 0.0 else 1.0
    k_bulk = p.c1
    mu_eff = rho0 * p.nu
    if dt > _EM20:
        g_eff = mu_eff / dt
    else:
        g_eff = mu_eff

    c_mat = np.zeros((6, 6), dtype=float)
    c_mat[0, 0] = c_mat[1, 1] = c_mat[2, 2] = k_bulk + (4.0 / 3.0) * g_eff
    c_mat[0, 1] = c_mat[1, 0] = k_bulk - (2.0 / 3.0) * g_eff
    c_mat[0, 2] = c_mat[2, 0] = k_bulk - (2.0 / 3.0) * g_eff
    c_mat[1, 2] = c_mat[2, 1] = k_bulk - (2.0 / 3.0) * g_eff
    c_mat[3, 3] = g_eff
    c_mat[4, 4] = g_eff
    c_mat[5, 5] = g_eff
    return c_mat


consistent_solid_tangent = solid_tangent


# ============================================================================
# Registration Factory
# ============================================================================

def build_law46(record: Any) -> Material:
    """Factory to instantiate Material for /MAT/LAW46 or /MAT/LES_FLUID."""
    params = getattr(record, "params", {}) or {}
    mid = getattr(record, "id", 1) or 1
    title = getattr(record, "title", "") or ""
    rho0 = float(getattr(record, "density", None) or params.get("MAT_RHO") or params.get("rho0") or 1.0)
    lp = _ensure_params(record)

    mat = Material(
        id=mid,
        law=46,
        rho0=rho0,
        title=title,
        params={
            "MAT_RHO": lp.rho0,
            "Refer_Rho": lp.ref_rho,
            "MAT_C": lp.c,
            "MAT_NU": lp.nu,
            "Istf": lp.istf,
            "MAT_C5": lp.smag,
            "MAT_CO1": lp.cps,
            "rho0": lp.rho0,
            "ref_rho": lp.ref_rho,
            "c": lp.c,
            "nu": lp.nu,
            "istf": lp.istf,
            "smag": lp.smag,
            "cps": lp.cps,
            "c1": lp.c1,
        },
    )
    mat.law_name = "LAW46"
    return mat

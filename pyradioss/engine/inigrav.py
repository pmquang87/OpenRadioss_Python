# Ported from OpenRadioss Fortran:
# Source: starter/source/initial_conditions/inigrav/hm_read_inigrav.F
# Function: HM_READ_INIGRAV (lines 39-375)
# Source: starter/source/initial_conditions/inigrav/inigrav_load.F
# Function: INIGRAV_LOAD (lines 38-546)
# Source: starter/source/initial_conditions/inigrav/inigrav_m37.F
# Function: INIGRAV_M37 (lines 29-108)
# Source: starter/source/initial_conditions/inigrav/inigrav_m51.F
# Function: INIGRAV_M51 (lines 31-275)
# Source: starter/source/initial_conditions/inigrav/inigrav_eos.F
# Function: INIGRAV_EOS (lines 30-213)
"""
/INIGRAV — Initial Geostatic Gravity Stress & Pore Pressure Equilibrium.

Upstream OpenRadioss Fortran Reference:
----------------------------------------
- Starter card reader:
  ``starter/source/initial_conditions/inigrav/hm_read_inigrav.F`` (HM_READ_INIGRAV)
- Engine & Starter load initialization:
  ``starter/source/initial_conditions/inigrav/inigrav_load.F`` (INIGRAV_LOAD)
  ``starter/source/initial_conditions/inigrav/inigrav_m37.F`` (INIGRAV_M37 for LAW37 biphasic soil)
  ``starter/source/initial_conditions/inigrav/inigrav_m51.F`` (INIGRAV_M51 for LAW51 porous media)
  ``starter/source/initial_conditions/inigrav/inigrav_eos.F`` (INIGRAV_EOS for hydrodynamic / solid media)

Theory & Geomechanics Formulation:
----------------------------------
In geotechnical, soil, rock, and reservoir mechanics, initial self-weight creates a
depth-dependent pre-stress state prior to any external loading or excavation.
Without proper initialization, applying gravity dynamically causes artificial settlement waves,
plastic failure, and numerical oscillations.

1. Free Surface and Depth Computation:
   In `inigrav_load.F` (lines 322-331), given gravity acceleration vector g with unit vector
   n_g = g / |g|, a planar surface defined by reference point B and normal n_surf:
   For any cell centroid Z:
     alpha = ((B - Z) . n_surf) / (n_g . n_surf)
     h = -alpha = ((Z - B) . n_surf) / (n_g . n_surf)
   For vertical gravity g = (0, 0, -g) and horizontal ground plane B = (0, 0, z_surface) with
   upward normal n_surf = (0, 0, 1):
     n_g . n_surf = -1
     h = (z - z_surface) / (-1) = z_surface - z >= 0 (depth below surface).

2. Vertical Total Stress (Lithostatic Overburden):
   sigma_v(h) = P_ref + integral_0^h rho(zeta) * |g| dzeta
   For discrete layers k = 1..N:
     sigma_v(h) = P_ref + sum_k rho_k * |g| * delta_h_k

3. Hydrostatic Pore Water Pressure:
   Below the water table elevation z_water (depth h_w = max(0.0, z_water - z)):
     u_w = max(0.0, rho_w * |g| * (z_water - z))
   where rho_w is fluid / groundwater density (default 1000 kg/m^3 in SI).

4. Terzaghi Effective Vertical Stress:
     sigma'_v = sigma_v - u_w

5. Lateral Earth Pressure at Rest (K0):
   Horizontal effective stress:
     sigma'_h = K0 * sigma'_v
   Horizontal total stress:
     sigma_h = sigma'_h + u_w = K0 * (sigma_v - u_w) + u_w
   (or sigma_h = K0 * sigma_v if effective stress calculation is disabled).

   Coefficient K0 can be:
   - Specified directly (e.g. K0 = 0.5)
   - Computed from internal friction angle phi via Jaky's equation:
       K0 = 1 - sin(phi)
   - Computed from elastic Poisson's ratio nu under 1D lateral constraint:
       K0 = nu / (1 - nu)

6. Radioss Cauchy Stress Tensor Convention:
   In OpenRadioss, normal stresses follow continuum mechanics tension-positive /
   compression-negative convention:
     sigma_zz = -sigma_v
     sigma_xx = -sigma_hx
     sigma_yy = -sigma_hy
     sigma_xy = sigma_yz = sigma_zx = 0.0

7. Static Equilibrium:
   In static balance, the internal stress divergence exactly balances the gravity body force:
     div(sigma) + rho * g = 0
     d(sigma_zz)/dz + rho * g_z = 0  =>  d(-sigma_v)/dz + rho * (-|g|) = 0  =>  d(sigma_v)/dz = -rho * |g|.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class SoilLayer:
    """Stratigraphic soil or rock layer definition."""

    part_id: int = 0
    name: str = ""
    z_top: Optional[float] = None
    z_bottom: Optional[float] = None
    density: float = 2000.0  # Total / bulk soil density (kg/m^3)
    k0: Optional[float] = None  # Lateral earth pressure coefficient K0
    phi: Optional[float] = None  # Internal friction angle in degrees
    nu: Optional[float] = None  # Poisson's ratio for elastic K0 = nu / (1 - nu)


@dataclass
class IniGravParams:
    """Parameters for /INIGRAV geostatic stress initialization.

    Ported from:
      - ``starter/source/initial_conditions/inigrav/hm_read_inigrav.F``
      - ``starter/source/initial_conditions/inigrav/inigrav_load.F``
    """

    id: int = 1
    title: str = ""
    grpart_id: int = 0
    part_ids: List[int] = field(default_factory=list)
    surf_id: int = 0
    grav_id: int = 0

    # Gravity acceleration vector (default: standard Earth vertical gravity)
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

    # Atmospheric / surface reference pressure P_ref (Pref in hm_read_inigrav.F line 157)
    pref: float = 0.0

    # Planar ground surface definition: basis point B (Bx, By, Bz) and outward normal (Nx, Ny, Nz)
    surface_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    surface_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    z_surface: Optional[float] = None

    # Lateral earth pressure coefficients K0
    k0: float = 0.5
    k0_x: Optional[float] = None
    k0_y: Optional[float] = None
    phi: Optional[float] = None  # Friction angle in degrees
    nu: Optional[float] = None  # Poisson's ratio

    # Groundwater / pore pressure options
    z_water: Optional[float] = None  # Water table elevation
    rho_w: float = 1000.0  # Fluid / pore water density
    use_effective_stress_k0: bool = True  # sigma_h = K0 * sigma'_v + u_w

    # Soil stratigraphy
    layers: List[SoilLayer] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Resolve z_surface if not explicitly provided
        if self.z_surface is None:
            self.z_surface = float(self.surface_point[2])
        else:
            self.surface_point = (self.surface_point[0], self.surface_point[1], float(self.z_surface))

        # Resolve K0 from phi or nu if provided
        if self.phi is not None and self.phi > 0.0:
            self.k0 = k0_from_phi(self.phi)
        elif self.nu is not None and 0.0 < self.nu < 0.5:
            self.k0 = k0_from_nu(self.nu)

        if self.k0_x is None:
            self.k0_x = self.k0
        if self.k0_y is None:
            self.k0_y = self.k0

    @property
    def g_magnitude(self) -> float:
        gx, gy, gz = self.gravity
        return math.sqrt(gx * gx + gy * gy + gz * gz)

    @property
    def g_unit(self) -> np.ndarray:
        mag = self.g_magnitude
        if mag < 1e-15:
            return np.array([0.0, 0.0, -1.0])
        return np.array(self.gravity, dtype=float) / mag


@dataclass
class GeostaticStressResult:
    """Output results of geostatic gravity stress calculation."""

    sigma_total: np.ndarray  # Shape (N, 6): [sig_xx, sig_yy, sig_zz, sig_xy, sig_yz, sig_zx]
    sigma_effective: np.ndarray  # Shape (N, 6): [sig'_xx, sig'_yy, sig'_zz, sig'_xy, sig'_yz, sig'_zx]
    pore_pressure: np.ndarray  # Shape (N,): u_w
    depth: np.ndarray  # Shape (N,): depth h below free surface
    sigma_v: np.ndarray  # Shape (N,): total vertical overburden stress (positive)
    sigma_hx: np.ndarray  # Shape (N,): total horizontal stress X (positive)
    sigma_hy: np.ndarray  # Shape (N,): total horizontal stress Y (positive)
    sigma_v_eff: np.ndarray  # Shape (N,): effective vertical stress (positive)
    sigma_hx_eff: np.ndarray  # Shape (N,): effective horizontal stress X (positive)
    sigma_hy_eff: np.ndarray  # Shape (N,): effective horizontal stress Y (positive)
    centroids: np.ndarray  # Shape (N, 3): element centroids


def k0_from_phi(phi_deg: float) -> float:
    """Compute lateral earth pressure coefficient at rest using Jaky's formula: K0 = 1 - sin(phi)."""
    phi_rad = math.radians(phi_deg)
    return max(0.01, 1.0 - math.sin(phi_rad))


def k0_from_nu(nu: float) -> float:
    """Compute lateral earth pressure coefficient at rest from elastic Poisson's ratio: K0 = nu / (1 - nu)."""
    if nu >= 0.5:
        return 1.0
    denom = max(1.0 - nu, 1e-12)
    return max(0.01, nu / denom)


def compute_depth_along_gravity(
    centroids: np.ndarray,
    surface_point: Union[Sequence[float], np.ndarray],
    surface_normal: Union[Sequence[float], np.ndarray],
    gravity_unit: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    """Compute perpendicular or gravity-aligned depth from surface plane matching inigrav_load.F (lines 322-331).

    In Fortran:
      ALPHA = ((BX - Z(1))*NX + (BY - Z(2))*NY + (BZ - Z(3))*NZ) / (NGX*NX + NGY*NY + NGZ*NZ)
      DEPTH = -((INTERP - Z) . NG) = -ALPHA
    """
    pts = np.atleast_2d(centroids)
    b = np.asarray(surface_point, dtype=float)
    n = np.asarray(surface_normal, dtype=float)
    norm_n = np.linalg.norm(n)
    if norm_n > 1e-14:
        n = n / norm_n

    ng = np.asarray(gravity_unit, dtype=float)
    norm_ng = np.linalg.norm(ng)
    if norm_ng > 1e-14:
        ng = ng / norm_ng

    dot_ng_n = float(np.dot(ng, n))
    if abs(dot_ng_n) < 1e-12:
        # Gravity is parallel to surface plane: fall back to distance along normal
        diff = b[None, :] - pts
        return np.maximum(0.0, np.sum(diff * n[None, :], axis=1))

    diff = b[None, :] - pts  # B - Z
    alpha = np.sum(diff * n[None, :], axis=1) / dot_ng_n
    depth = -alpha
    return np.maximum(0.0, depth)


def compute_geostatic_stress(
    centroids: np.ndarray,
    densities: Union[float, np.ndarray, Sequence[float]],
    params: Optional[IniGravParams] = None,
    **kwargs: Any,
) -> GeostaticStressResult:
    """Compute geostatic vertical total stress, pore pressure, and lateral stresses.

    Parameters:
    -----------
    centroids : np.ndarray
        Coordinates of element centroids, shape (N, 3) or (3,).
    densities : float or np.ndarray
        Bulk soil/rock density for each element, or scalar for uniform media.
    params : IniGravParams, optional
        Initialization parameters. If not provided, constructed from kwargs.

    Returns:
    --------
    GeostaticStressResult
        Contains Cauchy stress tensors (total and effective) in Radioss convention
        (compression negative), pore pressure, depth, and directional components.
    """
    if params is None:
        params = build_inigrav(kwargs)

    pts = np.atleast_2d(centroids).copy()
    nel = pts.shape[0]

    if np.isscalar(densities):
        rho = np.full(nel, float(densities), dtype=float)
    else:
        rho = np.asarray(densities, dtype=float)
        if rho.ndim == 0:
            rho = np.full(nel, float(rho), dtype=float)
        elif len(rho) != nel:
            rho = np.resize(rho, nel)

    g_mag = params.g_magnitude
    ng = params.g_unit

    # 1. Depth calculation (inigrav_load.F lines 322-331)
    if params.surface_normal is not None and params.surface_point is not None:
        depth = compute_depth_along_gravity(pts, params.surface_point, params.surface_normal, ng)
    else:
        z_surf = params.z_surface if params.z_surface is not None else 0.0
        depth = np.maximum(0.0, z_surf - pts[:, 2])

    # 2. Vertical total stress integration: sigma_v = P_ref + rho * g * h
    # If layered stratigraphy is specified, sort and integrate layered overburden
    if params.layers:
        sigma_v = _integrate_layered_overburden(pts, params.layers, g_mag, params.pref)
    else:
        sigma_v = params.pref + rho * g_mag * depth

    # 3. Hydrostatic pore water pressure: u_w = max(0.0, rho_w * g * (z_water - z))
    pore_press = np.zeros(nel, dtype=float)
    if params.z_water is not None:
        h_water = np.maximum(0.0, params.z_water - pts[:, 2])
        pore_press = params.rho_w * g_mag * h_water

    # 4. Effective vertical stress: sigma'_v = sigma_v - u_w
    sigma_v_eff = np.maximum(0.0, sigma_v - pore_press)

    # 5. Horizontal stresses from K0
    k0_x = params.k0_x if params.k0_x is not None else params.k0
    k0_y = params.k0_y if params.k0_y is not None else params.k0

    if params.use_effective_stress_k0 and params.z_water is not None:
        # Terzaghi: sigma'_h = K0 * sigma'_v, sigma_h = sigma'_h + u_w
        sigma_hx_eff = k0_x * sigma_v_eff
        sigma_hy_eff = k0_y * sigma_v_eff
        sigma_hx = sigma_hx_eff + pore_press
        sigma_hy = sigma_hy_eff + pore_press
    else:
        # Total stress formulation: sigma_h = K0 * sigma_v
        sigma_hx = k0_x * sigma_v
        sigma_hy = k0_y * sigma_v
        sigma_hx_eff = np.maximum(0.0, sigma_hx - pore_press)
        sigma_hy_eff = np.maximum(0.0, sigma_hy - pore_press)

    # 6. Radioss Cauchy stress tensor assignment (compression is negative)
    # inigrav_m37.F line 103: GBUF%SIG = - (PGRAV - P0 - PSH)
    # inigrav_m51.F line 267: GBUF%SIG = - PGRAV
    # inigrav_eos.F line 195: GBUF%SIG = - PRES
    # Components: [sig_xx, sig_yy, sig_zz, sig_xy, sig_yz, sig_zx]
    sigma_total = np.zeros((nel, 6), dtype=float)
    sigma_total[:, 0] = -sigma_hx
    sigma_total[:, 1] = -sigma_hy
    sigma_total[:, 2] = -sigma_v

    sigma_effective = np.zeros((nel, 6), dtype=float)
    sigma_effective[:, 0] = -sigma_hx_eff
    sigma_effective[:, 1] = -sigma_hy_eff
    sigma_effective[:, 2] = -sigma_v_eff

    # If single element input, maintain clean indexing
    if centroids.ndim == 1:
        return GeostaticStressResult(
            sigma_total=sigma_total[0],
            sigma_effective=sigma_effective[0],
            pore_pressure=float(pore_press[0]),
            depth=float(depth[0]),
            sigma_v=float(sigma_v[0]),
            sigma_hx=float(sigma_hx[0]),
            sigma_hy=float(sigma_hy[0]),
            sigma_v_eff=float(sigma_v_eff[0]),
            sigma_hx_eff=float(sigma_hx_eff[0]),
            sigma_hy_eff=float(sigma_hy_eff[0]),
            centroids=pts[0],
        )

    return GeostaticStressResult(
        sigma_total=sigma_total,
        sigma_effective=sigma_effective,
        pore_pressure=pore_press,
        depth=depth,
        sigma_v=sigma_v,
        sigma_hx=sigma_hx,
        sigma_hy=sigma_hy,
        sigma_v_eff=sigma_v_eff,
        sigma_hx_eff=sigma_hx_eff,
        sigma_hy_eff=sigma_hy_eff,
        centroids=pts,
    )


def _integrate_layered_overburden(
    pts: np.ndarray,
    layers: Sequence[SoilLayer],
    g_mag: float,
    pref: float,
) -> np.ndarray:
    """Compute cumulative vertical stress through a multi-layered soil profile."""
    nel = pts.shape[0]
    sigma_v = np.full(nel, pref, dtype=float)

    # Sort layers by z_top descending (surface to deep)
    sorted_layers = sorted(
        layers,
        key=lambda lay: lay.z_top if lay.z_top is not None else float("inf"),
        reverse=True,
    )

    for i in range(nel):
        z_i = pts[i, 2]
        cum_stress = pref
        for lay in sorted_layers:
            z_top = lay.z_top if lay.z_top is not None else float("inf")
            z_bot = lay.z_bottom if lay.z_bottom is not None else float("-inf")
            if z_i >= z_top:
                # Point is above this layer, nothing from this layer
                continue
            if z_i <= z_bot:
                # Point is below this layer, full layer overburden contributes
                thick = z_top - z_bot
                cum_stress += lay.density * g_mag * thick
            else:
                # Point is inside this layer, partial depth contributes
                thick = z_top - z_i
                cum_stress += lay.density * g_mag * thick
        sigma_v[i] = cum_stress

    return sigma_v


def apply_inigrav(
    model: Any,
    params: Optional[IniGravParams] = None,
    elem_centroids: Optional[np.ndarray] = None,
    elem_densities: Optional[Union[float, np.ndarray]] = None,
    elem_parts: Optional[Sequence[int]] = None,
) -> GeostaticStressResult:
    """Apply /INIGRAV geostatic stress initialization to a model or element mesh.

    Parameters:
    -----------
    model : Any
        Radioss Model instance or element container.
    params : IniGravParams, optional
        Parameters defining surface, gravity, K0, and pore water.
    elem_centroids : np.ndarray, optional
        Precomputed centroids of solid elements.
    elem_densities : float or np.ndarray, optional
        Densities of elements.
    elem_parts : Sequence[int], optional
        Part ID for each element.

    Returns:
    --------
    GeostaticStressResult
        The computed initial stresses and pore pressures.
    """
    if params is None:
        params = IniGravParams()

    # Extract centroids from model if not provided
    if elem_centroids is None:
        if hasattr(model, "solid_elements") and model.solid_elements:
            pts_list = []
            rho_list = []
            for elem in model.solid_elements.values():
                # Compute centroid from node coordinates if available
                if hasattr(model, "nodes"):
                    coords = [model.nodes[nid].coords for nid in elem.node_ids if nid in model.nodes]
                    pts_list.append(np.mean(coords, axis=0) if coords else np.zeros(3))
                else:
                    pts_list.append(np.zeros(3))
                # Material density
                rho_val = 2000.0
                if hasattr(model, "materials") and hasattr(elem, "mat_id") and elem.mat_id in model.materials:
                    rho_val = getattr(model.materials[elem.mat_id], "rho0", 2000.0)
                rho_list.append(rho_val)
            elem_centroids = np.array(pts_list, dtype=float)
            elem_densities = np.array(rho_list, dtype=float)
        else:
            elem_centroids = np.zeros((1, 3))
            elem_densities = 2000.0

    res = compute_geostatic_stress(elem_centroids, elem_densities, params=params)

    # Store computed stresses on model if supported
    if hasattr(model, "solid_elements") and model.solid_elements:
        for idx, (eid, elem) in enumerate(model.solid_elements.items()):
            if idx < len(res.sigma_total):
                elem.initial_stress = res.sigma_total[idx].copy()

    return res


def check_geostatic_equilibrium(
    z_coords: np.ndarray,
    sigma_v: np.ndarray,
    densities: Union[float, np.ndarray],
    gravity: float = 9.81,
    tolerance: float = 1e-4,
) -> Tuple[bool, float]:
    """Verify internal vertical stress divergence d(sigma_v)/dz balances the body force rho * g.

    Continuous equilibrium along vertical z:
      d(sigma_v)/dz = -rho * g
    Using finite differences between vertical stations:
      Delta(sigma_v) / Delta(z) = -rho * g
      Residual = | Delta(sigma_v) + rho * g * Delta(z) |

    Returns:
    --------
    (is_balanced, max_relative_residual)
    """
    z_arr = np.asarray(z_coords, dtype=float)
    sig_arr = np.asarray(sigma_v, dtype=float)

    # Sort stations along ascending elevation z
    order = np.argsort(z_arr)
    z_sorted = z_arr[order]
    sig_sorted = sig_arr[order]

    if np.isscalar(densities):
        rho_sorted = np.full(len(z_sorted), float(densities), dtype=float)
    else:
        rho_sorted = np.asarray(densities, dtype=float)[order]

    max_rel_err = 0.0
    for i in range(len(z_sorted) - 1):
        dz = z_sorted[i + 1] - z_sorted[i]
        if abs(dz) < 1e-12:
            continue
        dsig = sig_sorted[i + 1] - sig_sorted[i]
        rho_mid = 0.5 * (rho_sorted[i] + rho_sorted[i + 1])

        # Overburden increases as depth increases (z decreases), so d(sigma_v)/dz = -rho * g
        expected_dsig = -rho_mid * gravity * dz
        rel_err = abs(dsig - expected_dsig) / max(abs(expected_dsig), 1e-6)
        max_rel_err = max(max_rel_err, rel_err)

    is_balanced = max_rel_err <= tolerance
    return is_balanced, max_rel_err


def build_inigrav(data: Union[Dict[str, Any], Any]) -> IniGravParams:
    """Build IniGravParams from dictionary, keyword record, or IniGrav entity."""
    if isinstance(data, IniGravParams):
        return data

    d: Dict[str, Any] = {}
    if hasattr(data, "__dict__"):
        d = dict(data.__dict__)
    elif isinstance(data, dict):
        d = dict(data)

    iid = int(d.get("id", d.get("user_id", 1)))
    title = str(d.get("title", ""))
    grpart_id = int(d.get("grpart_id", 0))
    surf_id = int(d.get("surf_id", 0))
    grav_id = int(d.get("grav_id", 0))
    pref = float(d.get("pref", d.get("psurf", d.get("p0", 0.0))))

    bx = float(d.get("bx", d.get("BX", 0.0)))
    by = float(d.get("by", d.get("BY", 0.0)))
    bz = float(d.get("bz", d.get("BZ", 0.0)))
    surface_point = (bx, by, bz)

    nx = float(d.get("nx", 0.0))
    ny = float(d.get("ny", 0.0))
    nz = float(d.get("nz", 1.0))
    surface_normal = (nx, ny, nz)

    gx = float(d.get("gx", 0.0))
    gy = float(d.get("gy", 0.0))
    gz = float(d.get("gz", -9.81))
    gravity = d.get("gravity", (gx, gy, gz))

    z_surf = d.get("z_surface", bz if "bz" in d or "BZ" in d else None)
    k0 = float(d.get("k0", 0.5))
    k0_x = float(d.get("k0_x")) if d.get("k0_x") is not None else None
    k0_y = float(d.get("k0_y")) if d.get("k0_y") is not None else None
    phi = float(d.get("phi")) if d.get("phi") is not None else None
    nu = float(d.get("nu")) if d.get("nu") is not None else None

    z_water = float(d.get("z_water")) if d.get("z_water") is not None else None
    rho_w = float(d.get("rho_w", 1000.0))
    use_eff = bool(d.get("use_effective_stress_k0", True))

    layers_in = d.get("layers", [])
    layers = [l if isinstance(l, SoilLayer) else SoilLayer(**l) for l in layers_in]

    return IniGravParams(
        id=iid,
        title=title,
        grpart_id=grpart_id,
        surf_id=surf_id,
        grav_id=grav_id,
        gravity=gravity,
        pref=pref,
        surface_point=surface_point,
        surface_normal=surface_normal,
        z_surface=z_surf,
        k0=k0,
        k0_x=k0_x,
        k0_y=k0_y,
        phi=phi,
        nu=nu,
        z_water=z_water,
        rho_w=rho_w,
        use_effective_stress_k0=use_eff,
        layers=layers,
    )

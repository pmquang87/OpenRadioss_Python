"""LAW6: Hydrodynamic viscous fluid.

Fortran origin: ``engine/source/materials/mat/mat006/m6law.F``,
``starter/source/materials/mat/mat006/hm_read_mat06.F``.
"""

from __future__ import annotations

import numpy as np

from pyradioss.model.entities import EquationOfState, Material

_EM20 = 1e-20


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


def solid_update(mat, sig: np.ndarray, deps: np.ndarray, epsp=None, dt: float = 0.0, extra: dict = None):
    """Update solid deviatoric stress for LAW6 (Newtonian fluid).

    Fortran origin: ``engine/source/materials/mat/mat006/m6law.F``.

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

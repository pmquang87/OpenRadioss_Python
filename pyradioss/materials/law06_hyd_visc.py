"""LAW6: Hydrodynamic viscous fluid."""

import numpy as np

from pyradioss.model.entities import EquationOfState, Material


def build_law6(rec):
    """Parse /MAT/LAW6 (HYD_VISC) parameters from CFG fields."""
    params = {}
    
    # Base density is already pulled into mat.rho0 by the generic reader,
    # but we can pull Refer_Rho if needed.
    # DAMP1 is the dynamic viscosity parameter.
    params["visc"] = float(rec.params.get("DAMP1") or 0.0)
    params["E"] = 0.0
    params["nu"] = 0.0
    
    # If the user supplied C0..C5, it has an embedded polynomial EOS.
    # In older Radioss formats, EOS was embedded in LAW6 directly.
    # We construct a synthetic /EOS/POLYNOMIAL object for it so that the engine's 
    # hexa8.py can just call eos_update(mat.eos, ...)
    
    eos = None
    c0 = rec.params.get("MAT_C0")
    if c0 is not None:
        c1 = float(rec.params.get("MAT_C1") or 0.0)
        c2 = float(rec.params.get("MAT_C2") or 0.0)
        c3 = float(rec.params.get("MAT_C3") or 0.0)
        c4 = float(rec.params.get("MAT_C4") or 0.0)
        c5 = float(rec.params.get("MAT_C5") or 0.0)
        e0 = float(rec.params.get("MAT_EA") or 0.0)
        psh = float(rec.params.get("MAT_PSH") or 0.0)
        pmin = float(rec.params.get("MAT_PC") or 0.0)
        
        eos_params = {
            "c0": float(c0),
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "c4": c4,
            "c5": c5,
            "e0": e0,
            "psh": psh,
            "pmin": pmin,
        }
        
        # We manually attach an EOS object.
        # This matches what starter_keywords.py does for explicit /EOS/POLYNOMIAL/
        eos = EquationOfState(kind="POLYNOMIAL", params=eos_params, rho0=rec.density)
        
    return Material(id=rec.id, law=6, rho0=rec.density, title=rec.title, params=params, eos=eos)


def solid_update(mat, sig, deps, epsp, dt, extra=None):
    """
    Update solid deviatoric stress for LAW6 (Newtonian fluid).
    
    Args:
        mat: The material object containing visc parameter.
        sig: Deviatoric stress tensor [nel, 6]. Modified in place.
        deps: Strain increment tensor [nel, 6].
        epsp: Not used for LAW6.
        dt: Time step.
        extra: Dictionary containing "rho" (current density) among other things.
        
    Returns:
        The updated deviatoric stress `sig` (in-place), `epsp`, and `c` (sound speed).
        Sound speed is returned as None so the element code falls back to the EOS sound speed.
    """
    nel = sig.shape[0]
    
    if dt > 0.0:
        deps_rate = deps / dt
    else:
        deps_rate = np.zeros_like(deps)
        
    # Volumetric strain rate (dav = Dii / 3)
    dav = -(deps_rate[:, 0] + deps_rate[:, 1] + deps_rate[:, 2]) / 3.0
    
    # Viscosity is scaled by current density in the Fortran code:
    # VIS(I) = PM(24,MX)*RHO(I)
    current_rho = extra["rho"]
    visc = mat.params["visc"] * current_rho
    vis2 = 2.0 * visc
    
    # Compute viscous stress deviator
    # SIG(I,1) = VIS2*(D1(I)+DAV)
    sig[:, 0] = vis2 * (deps_rate[:, 0] + dav)
    sig[:, 1] = vis2 * (deps_rate[:, 1] + dav)
    sig[:, 2] = vis2 * (deps_rate[:, 2] + dav)
    
    # Shear components
    # SIG(I,4) = VIS(I)*D4(I)
    sig[:, 3] = visc * deps_rate[:, 3]
    sig[:, 4] = visc * deps_rate[:, 4]
    sig[:, 5] = visc * deps_rate[:, 5]
    
    return sig, epsp, None

def _register():
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["LAW6"] = build_law6
    MAT_PHYSICS_REGISTRY["HYD_VISC"] = build_law6

_register()

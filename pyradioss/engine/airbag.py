import numpy as np
from pyradioss.model.model import Model

def update_airbag_thermodynamics(mv, model: Model, dt: float, current_time: float) -> None:
    if mv.iequil == 0:
        pass

    mat = model.materials[mv.matid]
    r_spec = mat.r_spec
    cpa, cpb, cpc = mat.params.get('CPA', 0.0), mat.params.get('CPB', 0.0), mat.params.get('CPC', 0.0)

    if mv.iequil == 1:
        v_eps = 0.0 
        gmi = mv.pext * (mv.volume + v_eps) / (r_spec * mv.t_initial)
        mv.mass = gmi
        mv.iequil = -1 
        mv.temperature = mv.t_initial
        mv.pressure = mv.pext
        mv.volume_old = mv.volume
        
    t_bag_old = getattr(mv, 'temperature', mv.t_initial)
    vol_old = getattr(mv, 'volume_old', mv.volume)
    
    vol = mv.volume
    dv = vol - vol_old
    
    gmi = getattr(mv, 'mass', 0.0)
    
    cvi = cpa + cpb * t_bag_old + cpc * t_bag_old**2 - r_spec
    
    left = gmi * cvi
    right = gmi * cvi * t_bag_old
    
    rnm_old = gmi * r_spec
    rnm = gmi * r_spec
    
    left += 0.5 * rnm * dv / vol
    right -= 0.5 * rnm_old * t_bag_old * dv / vol_old
    
    t_bag = right / left
    t_bag = max(t_bag, 0.0)
    
    p = rnm * t_bag / vol
    
    mv.temperature = t_bag
    mv.pressure = p
    mv.volume_old = vol

def update_airbag_volume(mv, model: Model, x: np.ndarray) -> None:
    """
    Recompute volume based on current nodal coordinates.
    Matches volpvga.F90 / get_volume_area.F90.
    """
    surf = model.surfaces[mv.surf_id]
    total_vol = 0.0
    
    for segment_nodes in surf.segments:
        n1 = segment_nodes[0]
        n2 = segment_nodes[1]
        n3 = segment_nodes[2]
        n4 = segment_nodes[3]
        
        x1 = x[n1]
        x2 = x[n2]
        x3 = x[n3]
        x4 = x[n4]
        
        v31 = x3 - x1
        if n4 != n3:
            v42 = x4 - x2
            xn = 0.5 * np.cross(v31, v42)
            vol_cont = np.dot(x1 + x2 + x3 + x4, xn) / 12.0
        else:
            v21 = x2 - x1
            xn = 0.5 * np.cross(v21, v31)
            vol_cont = np.dot(x1 + x2 + x3, xn) / 9.0
            
        total_vol += vol_cont
        
    mv.volume = total_vol

def apply_airbag_forces(mv, model: Model, x: np.ndarray, f: np.ndarray) -> None:
    """
    Apply pressure forces.
    Matches volpre.F.
    """
    surf = model.surfaces[mv.surf_id]
    p_eff = getattr(mv, 'pressure', 0.0) - mv.pext
    
    for segment_nodes in surf.segments:
        n1 = segment_nodes[0]
        n2 = segment_nodes[1]
        n3 = segment_nodes[2]
        n4 = segment_nodes[3]
        
        x1 = x[n1]
        x2 = x[n2]
        x3 = x[n3]
        x4 = x[n4]
        
        v31 = x3 - x1
        if n4 != n3:
            v42 = x4 - x2
            xn = 0.5 * np.cross(v31, v42)
            
            fx = p_eff * xn[0] / 4.0
            fy = p_eff * xn[1] / 4.0
            fz = p_eff * xn[2] / 4.0
            
            f[n1, 0] += fx; f[n1, 1] += fy; f[n1, 2] += fz
            f[n2, 0] += fx; f[n2, 1] += fy; f[n2, 2] += fz
            f[n3, 0] += fx; f[n3, 1] += fy; f[n3, 2] += fz
            f[n4, 0] += fx; f[n4, 1] += fy; f[n4, 2] += fz
            
        else:
            v21 = x2 - x1
            xn = 0.5 * np.cross(v21, v31)
            
            fx = p_eff * xn[0] / 3.0
            fy = p_eff * xn[1] / 3.0
            fz = p_eff * xn[2] / 3.0
            
            f[n1, 0] += fx; f[n1, 1] += fy; f[n1, 2] += fz
            f[n2, 0] += fx; f[n2, 1] += fy; f[n2, 2] += fz
            f[n3, 0] += fx; f[n3, 1] += fy; f[n3, 2] += fz

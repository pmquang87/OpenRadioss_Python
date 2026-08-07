import numpy as np
from pyradioss.model.model import Model

def initialize_monitored_volumes(model: Model) -> None:
    """
    Starter initialization for /MONVOL features.
    Computes initial volume and area for AIRBAG1 (Type 7).
    Matches the logic in fvmesh0.F / get_volume_area.F90.
    """
    for monvol in model.monitored_volumes.values():
        if monvol.vol_type != "AIRBAG1":
            continue
            
        if monvol.surf_id not in model.surfaces:
            # The starter checks this elsewhere usually, but just in case
            continue
            
        surf = model.surfaces[monvol.surf_id]
        if surf.segments is None or len(surf.segments) == 0:
            monvol.volume = 0.0
            monvol.area = 0.0
            continue
            
        x = model.x0
        
        n1 = surf.segments[:, 0]
        n2 = surf.segments[:, 1]
        n3 = surf.segments[:, 2]
        n4 = surf.segments[:, 3]
        
        x1 = x[n1]
        x2 = x[n2]
        x3 = x[n3]
        x4 = x[n4]
        
        xx = 0.5 * (x1 + x2)
        
        d13 = x3 - x1
        d24 = x4 - x2
        
        nx = 0.5 * (d13[:, 1] * d24[:, 2] - d13[:, 2] * d24[:, 1])
        ny = 0.5 * (d13[:, 2] * d24[:, 0] - d13[:, 0] * d24[:, 2])
        nz = 0.5 * (d13[:, 0] * d24[:, 1] - d13[:, 1] * d24[:, 0])
        
        area_seg = np.sqrt(nx**2 + ny**2 + nz**2)
        vol_seg = (1.0 / 3.0) * (nx * xx[:, 0] + ny * xx[:, 1] + nz * xx[:, 2])
        
        monvol.area = float(np.sum(area_seg))
        monvol.volume = float(np.sum(vol_seg))

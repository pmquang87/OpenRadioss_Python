import numpy as np
from pyradioss.model.model import Model

def initialize_monitored_volumes(model: Model) -> None:
    """
    Starter initialization for /MONVOL features.
    Computes initial volume and area for AIRBAG1 (Type 7).
    Matches the logic in fvmesh0.F / get_volume_area.F90.
    """
    for monvol in model.monitored_volumes.values():
        if monvol.vol_type not in ("AIRBAG1", "TYPE7", "GAS", "PRES", "TYPE2", "TYPE3"):
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
        
        segs = surf.segments
        if segs.shape[1] == 3:
            n1, n2, n3 = segs[:, 0], segs[:, 1], segs[:, 2]
            n4 = n3
        else:
            n1, n2, n3, n4 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
            mask_tri = (n4 < 0) | (n4 == n3)
            n4 = np.where(mask_tri, n3, n4)
        
        x1 = x[n1]
        x2 = x[n2]
        x3 = x[n3]
        x4 = x[n4]
        
        # Detect triangular facets (node 3 == node 4)
        mask_tri = (n4 == n3) if n4 is not None else np.zeros(len(x1), dtype=bool)
        xx = np.where(mask_tri[:, None],
                      (x1 + x2 + x3) / 3.0,
                      (x1 + x2 + x3 + x4) / 4.0)
        
        d13 = x3 - x1
        d24 = x4 - x2
        
        nx = 0.5 * (d13[:, 1] * d24[:, 2] - d13[:, 2] * d24[:, 1])
        ny = 0.5 * (d13[:, 2] * d24[:, 0] - d13[:, 0] * d24[:, 2])
        nz = 0.5 * (d13[:, 0] * d24[:, 1] - d13[:, 1] * d24[:, 0])
        
        area_seg = np.sqrt(nx**2 + ny**2 + nz**2)
        vol_seg = (1.0 / 3.0) * (nx * xx[:, 0] + ny * xx[:, 1] + nz * xx[:, 2])
        
        monvol.area = float(np.sum(area_seg))
        monvol.volume = float(np.sum(vol_seg))

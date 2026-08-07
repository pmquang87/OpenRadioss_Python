import numpy as np
from pyradioss.model.model import Model

def initialize_monitored_volumes(model: Model) -> None:
    """
    Starter initialization for /MONVOL features.
    Computes initial volume and area for AIRBAG1 (Type 7).
    """
    for monvol in model.monitored_volumes.values():
        if monvol.type != "AIRBAG1":
            continue
            
        if monvol.surf_id not in model.surfaces:
            # Starter check should catch this normally
            continue
            
        surf = model.surfaces[monvol.surf_id]
        
        # Array of node indices (0-indexed internally)
        # surf.nodes is (N, 4). 
        # But wait, does surf have a .nodes attribute?
        # Let's check pyradioss/model/entities.py Surface.

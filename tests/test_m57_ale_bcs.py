import pytest
import numpy as np
from pathlib import Path

from pyradioss.starter.starter import run_starter

def test_ale_bcs_parsing():
    """Verify /ALE/BCS is correctly parsed from a deck string."""
    deck = """
#RADIOSS STARTER
/ALE/BCS/101
ALE_Boundary_Condition_XYZ
#    WL_flags  skew_ID   grnod_ID
       110001        2        300
/SKEW/FIX/2
Skew_for_ALE
#              X_O               Y_O               Z_O
               0.0               0.0               0.0
#              X_Z               Y_Z               Z_Z
               0.0               0.0               1.0
#             X_ZX              Y_ZX              Z_ZX
               1.0               0.0               0.0
/GRNOD/NODE/300
My_Grnod
1
/NODE
    1 0.0 0.0 0.0
    2 1.0 0.0 0.0
    3 0.0 1.0 0.0
    4 0.0 0.0 1.0
/PART/1
Dummy_Part
1 1
/MAT/LAW1/1
Elastic
7.8e-9
210000 0.3
/PROP/SOLID/1
Solid_Prop
1
/TETRA4/1
1 1 2 3 4
#---1----|----2----|----3----|----4----|----5----|----6----|----7----|----8----|----9----|---10----|
"""
    tmp_path = Path("tests/data/m57_ale_tmp.rad")
    try:
        with open(tmp_path, "w") as f:
            f.write(deck)
            
        model = run_starter(str(tmp_path), log=None)
        
        assert len(model.ale_bcs) == 1
        
        bc = model.ale_bcs[0]
        assert bc.id == 101
        assert bc.grnod_id == 300
        assert bc.skew_id == 2
        assert bc.title == "ALE_Boundary_Condition_XYZ"
        
        # WX=1, WY=1, WZ=0 (110)
        assert np.array_equal(bc.fix_w, [True, True, False])
        # LX=0, LY=0, LZ=1 (001)
        assert np.array_equal(bc.fix_l, [False, False, True])

        # Check that skew_row is correctly assigned (0 is global, so it should be > 0 since we have a skew).
        assert bc.skew_row > 0  
    finally:
        tmp_path.unlink(missing_ok=True)

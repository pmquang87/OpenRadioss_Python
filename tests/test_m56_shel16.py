import pytest
import numpy as np

from pyradioss.model.model import Model
from pyradioss.starter import initialization

def test_shel16_starter():
    """Verify that SHEL16 creates element groups successfully."""
    deck = [
        "/BEGIN",
        "Test Model",
        "1",
        "1 1 1",
        "/NODE",
        "1  0.0 0.0 0.0",
        "2  1.0 0.0 0.0",
        "3  1.0 1.0 0.0",
        "4  0.0 1.0 0.0",
        "5  0.0 0.0 1.0",
        "6  1.0 0.0 1.0",
        "7  1.0 1.0 1.0",
        "8  0.0 1.0 1.0",
        # Mid-nodes 
        "9  0.5 0.0 0.0",
        "10 1.0 0.5 0.0",
        "11 0.5 1.0 0.0",
        "12 0.0 0.5 0.0",
        "13 0.5 0.0 1.0",
        "14 1.0 0.5 1.0",
        "15 0.5 1.0 1.0",
        "16 0.0 0.5 1.0",
        "/PART/1",
        "Test Part",
        "1 1",
        "/PROP/TSHELL/1",
        "Thick Shell Prop",
        "1 1 1",
        "0.1",
        "/MAT/LAW1/1",
        "Elastic",
        "7.8e-9",
        "210000 0.3",
        "/SHEL16/1",
        "1 1 2 3 4 5 6 7 8",
        "9 10 11 12",
        "13 14 15 16",
        "2 1 2 3 4 5 6 7 8",
        "0 0 0 0",
        "0 0 0 0",
    ]
    import pytest
    from pyradioss.starter.starter import run_starter
    from pyradioss.model.model import Model
    from pyradioss.common.messages import MessageLog
    from pathlib import Path

    # Write deck to file
    tmp_path = Path("tests/data/m56_tmp.rad")
    with open(tmp_path, "w") as f:
        f.write("\n".join(deck) + "\n")
        
    try:
        model = run_starter(str(tmp_path), log=None)
        
        assert model.shel16s is not None
        assert model.shel16s.n == 2
        assert model.shel16s.conn[0, 8] != -1
        assert model.shel16s.conn[1, 8] == -1
    finally:
        tmp_path.unlink(missing_ok=True)


def test_shel16_engine():
    """Verify that SHEL16 engine integration runs without error."""
    deck = [
        "/BEGIN",
        "Test Model",
        "1",
        "1 1 1",
        "/NODE",
        "1  0.0 0.0 0.0",
        "2  1.0 0.0 0.0",
        "3  1.0 1.0 0.0",
        "4  0.0 1.0 0.0",
        "5  0.0 0.0 1.0",
        "6  1.0 0.0 1.0",
        "7  1.0 1.0 1.0",
        "8  0.0 1.0 1.0",
        # Mid-nodes 
        "9  0.5 0.0 0.0",
        "10 1.0 0.5 0.0",
        "11 0.5 1.0 0.0",
        "12 0.0 0.5 0.0",
        "13 0.5 0.0 1.0",
        "14 1.0 0.5 1.0",
        "15 0.5 1.0 1.0",
        "16 0.0 0.5 1.0",
        "/PART/1",
        "Test Part",
        "1 1",
        "/PROP/TSHELL/1",
        "Thick Shell Prop",
        "1 1 1",
        "0.1",
        "/MAT/LAW1/1",
        "Elastic",
        "7.8e-9",
        "210000 0.3",
        "/SHEL16/1",
        "1 1 2 3 4 5 6 7 8",
        "9 10 11 12",
        "13 14 15 16",
        "2 1 2 3 4 5 6 7 8",
        "0 0 0 0",
        "0 0 0 0",
    ]
    
    deck_engine = [
        "/RUN/Test Model/1",
        "0.001",
        "/DT/NODA/CST",
        "0.9 1.0e-6",
        "/ANIM/ELEM/EPSP",
        "/ANIM/ELEM/VONM",
    ]
    
    import pytest
    from pyradioss.starter.starter import run_starter
    from pyradioss.engine.engine import run_engine
    from pyradioss.common.messages import MessageLog
    from pathlib import Path
    
    # Write deck to file
    tmp_path = Path("tests/data/m56_tmp2_0000.rad")
    tmp_path_eng = Path("tests/data/m56_tmp2_0001.rad")
    with open(tmp_path, "w") as f:
        f.write("\n".join(deck) + "\n")
    with open(tmp_path_eng, "w") as f:
        f.write("\n".join(deck_engine) + "\n")
        
    try:
        model = run_starter(str(tmp_path), log=None)
        assert model.shel16s is not None
        
        # Run engine 1 cycle
        run_engine(str(tmp_path_eng))
    finally:
        tmp_path.unlink(missing_ok=True)
        tmp_path_eng.unlink(missing_ok=True)

import pytest
from pathlib import Path
from pyradioss.engine import engine
from pyradioss.starter import starter
import numpy as np

# Use the LAW70 foam block test which uses Isolid=24 and compresses to densification
_DECK_DIR = Path(__file__).parent / "data" / "rd_decks" / "rd_v_material" / "RD-V-0220_Foam_LAW70" / "0220_foam_LAW70" / "0220_foam_LAW70_0"

@pytest.mark.slow
def test_heph_foam_compression(tmp_path):
    """M64: Verify HEPH (ISOLID=24) physically-stabilized brick runs stably through foam densification."""
    import shutil
    shutil.copy(_DECK_DIR / "BLOCK_H8_0000.rad", tmp_path / "BLOCK_H8_0000.rad")
    # write a short engine deck to run to 0.077s
    engine_deck = """/BEGIN
BLOCK_H8_0001
/RUN
0.077
/ANIM/DT
0.01
/TFILE
0.01
/PRINT/-10000
/END
"""
    (tmp_path / "BLOCK_H8_0001.rad").write_text(engine_deck)

    # Starter
    model = starter.run_starter(str(tmp_path / "BLOCK_H8_0000.rad"))
    # Engine reads the .rst file produced by Starter
    out = engine.run_engine(str(tmp_path / "BLOCK_H8_0001.rad"))
    
    assert out is not None
    # check that hourglass energy is small or managed
    final_ehour = float(out.bricks_heph.state["ehour"].sum())
    final_eint = float(out.bricks_heph.state["eint"].sum())
    assert final_ehour < 0.1 * final_eint, "Hourglass energy blew up"

import os
os.environ["PYRADIOSS_BACKEND"] = "numpy"
import pytest
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.deck_reader import read_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.input.starter_keywords import read_inter

def test_engine_type18(tmp_path):
    d = StarterDeck("test")
    
    # Add mock nodes and surf
    d.raw_block("GRNOD/NODE/100", ["fluid nodes", "       1       2"])
    d.raw_block("SURF/SEG/200", ["struct surf", "       1       2       3       4"])
    
    # grnod_id surf_id grbric_id ibag idel18
    # stfac gap
    # stiff_dc sort_fact
    d.raw_block("INTER/TYPE18/10", [
        "       100       200       300                              " + "         1         2",
        "                50.0" + "                    " + "                 1.5",
        "                                        " + "                 0.1" + "                    " + "                 0.5"
    ])

    deck_str = d.render()
    path = tmp_path / "deck.rad"
    path.write_text(deck_str)
    
    blocks = list(read_deck(str(path)))
    model = Model()
    log = MessageLog()
    
    from pyradioss.input.starter_keywords import read_grnod, read_surf, read_inter
    for b in blocks:
        if b.parts[0] == "GRNOD":
            read_grnod(b, model, log)
        elif b.parts[0] == "SURF":
            read_surf(b, model, log)
        elif b.parts[0] == "INTER":
            read_inter(b, model, log)
    
    from pyradioss.model.entities import NodeGroup, Surface
    import numpy as np
    
    # Mock node groups and surfaces
    ng = NodeGroup(id=100, title="fluid nodes")
    ng.node_idx = np.array([0, 1], dtype=np.int32)
    model.node_groups[100] = ng
    
    surf = Surface(id=200, title="struct surf")
    surf.segments = np.array([[2, 3, 4, 5]], dtype=np.int32)
    model.surfaces[200] = surf
    
    # Mock model dt
    model.dt = 1e-5
    
    # Init contacts directly
    from pyradioss.contact import build_contacts
    penalty, tied = build_contacts(model, log)
    
    assert len(penalty) == 1
    assert type(penalty[0]).__name__ == "ContactType18"
    assert penalty[0].stfac == 50.0
    assert penalty[0].gap == 1.5
    assert penalty[0].stiff_dc == 0.1
    
    # Test forces evaluation
    x = np.array([
        [0.0, 0.0, 1.0], # sec node 0
        [1.0, 1.0, 1.0], # sec node 1
        [0.0, 0.0, 0.0], # main node 0
        [1.0, 0.0, 0.0], # main node 1
        [1.0, 1.0, 0.0], # main node 2
        [0.0, 1.0, 0.0], # main node 3
    ], dtype=np.float64)
    v = np.zeros_like(x)
    v[0] = [0.0, 0.0, -10.0] # moving down towards quad
    m = np.ones(6)
    
    fcont = np.zeros((6, 3), dtype=np.float64)
    penalty[0].forces(x, v, m, model, fcont)
    
    # Check that fcont is populated
    assert np.any(fcont)

if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        test_engine_type18(Path(td))
        print("Success")

import pytest
import numpy as np
from pyradioss.model.model import Model
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.starter.starter import run_starter

def test_transform_tra_submodel(tmp_path):
    deck = """
/BEGIN
Test TRANSFORM
/SUBMODEL/1
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
/ENDSUB
/NODE
3 0.0 0.0 0.0
4 0.0 0.0 1.0
/PART/1
dummy part
1 1
/MAT/LAW1/1
dummy mat
1e-9
1e3 0.3
/PROP/SOLID/1
dummy prop
/TETRA4/1
1 1 2 3 4
/TRANSFORM/TRA/10
0 10.0 20.0 30.0 0 0 1
/END
"""
    f = tmp_path / "test_0000.rad"
    f.write_text(deck)
    
    # We test `run_starter` to ensure the transform loop applies properly
    model = run_starter(str(f))
    
    # Node 1 and 2 are in submodel 1, should be transformed
    assert np.allclose(model.x0[model.node_index(1)], [10.0, 20.0, 30.0])
    assert np.allclose(model.x0[model.node_index(2)], [11.0, 20.0, 30.0])
    # Node 3 is outside submodel 1, should not be transformed
    assert np.allclose(model.x0[model.node_index(3)], [0.0, 0.0, 0.0])

def test_transform_tra_nodepair(tmp_path):
    deck = """
/BEGIN
Test TRANSFORM node pair
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 0.0 0.0
4 5.0 5.0 5.0
5 0.0 0.0 1.0
/PART/1
dummy part
1 1
/MAT/LAW1/1
dummy mat
1e-9
1e3 0.3
/PROP/SOLID/1
dummy prop
/TETRA4/1
1 1 2 3 5
/GRNOD/NODE/50
2
3
/TRANSFORM/TRA/20
50 1.0 2.0 3.0 1 4 0
/END
"""
    f = tmp_path / "test2_0000.rad"
    f.write_text(deck)
    
    model = run_starter(str(f))
    
    # Per OpenRadioss Fortran lectrans.F:224-226: node pair replaces card TX, TY, TZ.
    # node pair is 1 -> 4, delta is (5, 5, 5).
    # Applied to node 2 (1, 0, 0) -> (6, 5, 5) and node 3 (0, 0, 0) -> (5, 5, 5).
    assert np.allclose(model.x0[model.node_index(2)], [6.0, 5.0, 5.0])
    assert np.allclose(model.x0[model.node_index(3)], [5.0, 5.0, 5.0])

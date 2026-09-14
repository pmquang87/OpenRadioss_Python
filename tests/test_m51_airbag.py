import pytest
import numpy as np
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.starter.initialization import resolve_surfaces, build_element_groups
from pyradioss.starter.airbag import initialize_monitored_volumes

def test_airbag1_volume_area(tmp_path):
    # We define a 1x1x1 closed box centered around 0.5, 0.5, 0.5
    # with 6 faces (quads), all normals pointing outwards.
    # Volume should be 1.0, Area should be 6.0
    rad = """/PART/1
Part 1
1 1 1
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 0.0 0.0 1.0
6 1.0 0.0 1.0
7 1.0 1.0 1.0
8 0.0 1.0 1.0
/SHELL/1
101 1 4 3 2
102 5 6 7 8
103 1 2 6 5
104 2 3 7 6
105 3 4 8 7
106 4 1 5 8
/SURF/PART/10
10
1
    /MONVOL/AIRBAG1/1
    My Airbag
            10       0.1
           1.0       1.0       1.0       1.0       1.0
             5       1.4    1.0e-4     300.0         0         1
    0
    0
"""
    p = tmp_path / "test.rad"
    p.write_text(rad)

    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    
    build_element_groups(model, log)
    resolve_surfaces(model, log)
    initialize_monitored_volumes(model)

    mv = model.monitored_volumes[1]
    
    np.testing.assert_allclose(mv.area, 6.0)
    np.testing.assert_allclose(mv.volume, 1.0)


def test_airbag1_triangles_volume_area(tmp_path):
    # Same 1x1x1 closed cube, but each quad split into 2 triangular shells
    # (n4 == n3) to test triangular facet centroid volume calculation.
    # Volume should still be 1.0, Area should still be 6.0
    rad = """/PART/1
Part 1
1 1 1
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 0.0 0.0 1.0
6 1.0 0.0 1.0
7 1.0 1.0 1.0
8 0.0 1.0 1.0
/SHELL/1
101 1 4 3 3
102 1 3 2 2
103 5 6 7 7
104 5 7 8 8
105 1 2 6 6
106 1 6 5 5
107 2 3 7 7
108 2 7 6 6
109 3 4 8 8
110 3 8 7 7
111 4 1 5 5
112 4 5 8 8
/SURF/PART/10
10
1
    /MONVOL/AIRBAG1/1
    My Airbag
            10       0.1
           1.0       1.0       1.0       1.0       1.0
             5       1.4    1.0e-4     300.0         0         1
    0
    0
"""
    p = tmp_path / "test_tri.rad"
    p.write_text(rad)

    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    
    build_element_groups(model, log)
    resolve_surfaces(model, log)
    initialize_monitored_volumes(model)

    mv = model.monitored_volumes[1]
    
    np.testing.assert_allclose(mv.area, 6.0)
    np.testing.assert_allclose(mv.volume, 1.0)


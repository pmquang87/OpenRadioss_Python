import pytest
import numpy as np
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
import os
import tempfile

def test_move_funct_basic():
    deck = [
        "/BEGIN",
        "TEST_MOVE_FUNCT",
        "/FUNCT/1",
        "Function 1",
        " 0.0  0.0",
        " 1.0  1.0",
        " 2.0  4.0",
        "/MOVE_FUNCT/1",
        "  2.0  3.0  0.5  0.1",
        "/END"
    ]
    
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test_0000.rad")
        with open(path, "w") as f:
            f.write("\n".join(deck))
        
        blocks = read_deck(path)
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        
        # Apply MOVE_FUNCT
        for funct_id, scx, scy, shx, shy in getattr(model, "move_functs", []):
            if funct_id in model.functions:
                model.functions[funct_id].transform(scx, scy, shx, shy)

        fct = model.functions[1]
        np.testing.assert_allclose(fct.x, [0.5, 2.5, 4.5])
        np.testing.assert_allclose(fct.y, [0.1, 3.1, 12.1])


def test_move_funct_negative_scale():
    deck = [
        "/BEGIN",
        "TEST_MOVE_FUNCT",
        "/FUNCT/1",
        "Function 1",
        " 0.0  0.0",
        " 1.0  1.0",
        " 2.0  4.0",
        "/MOVE_FUNCT/1",
        " -1.0  1.0  0.0  0.0",
        "/END"
    ]
    
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test_0000.rad")
        with open(path, "w") as f:
            f.write("\n".join(deck))
        
        blocks = read_deck(path)
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        
        # Apply MOVE_FUNCT
        for funct_id, scx, scy, shx, shy in getattr(model, "move_functs", []):
            if funct_id in model.functions:
                model.functions[funct_id].transform(scx, scy, shx, shy)

        fct = model.functions[1]
        np.testing.assert_allclose(fct.x, [-2.0, -1.0, 0.0])
        np.testing.assert_allclose(fct.y, [4.0, 1.0, 0.0])

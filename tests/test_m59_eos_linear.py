import pytest
import numpy as np
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_eos
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog


def test_eos_linear_parsing(tmp_path):
    """Verify /EOS/LINEAR is parsed correctly into the canonical polynomial form."""
    d = StarterDeck("test")
    d.raw_block("EOS/LINEAR/100", ["                  10                1000                   5                   0"])
    
    deck_str = d.render()
    path = tmp_path / "deck.rad"
    path.write_text(deck_str)
    
    blocks = list(read_deck(str(path)))
    assert len(blocks) == 3
    
    model = Model()
    log = MessageLog()
    read_eos(blocks[1], model, log)
    
    assert len(model.raw_eos) == 1
    mat_id, eos, source = model.raw_eos[0]
    assert mat_id == 100
    assert eos.kind == "LINEAR"
    
    # P0 = 10, Bulk = 1000, Psh = 5, Rho0 = 0
    # c0 = P0 - Psh = 5
    # c1 = Bulk = 1000
    assert eos.params["c0"] == 5.0
    assert eos.params["c1"] == 1000.0
    assert eos.params["c2"] == 0.0
    assert eos.params["c3"] == 0.0
    assert eos.params["c4"] == 0.0
    assert eos.params["c5"] == 0.0
    assert eos.params["psh"] == 5.0

    # also test the compact string version
    d2 = StarterDeck("test2")
    d2.eos_linear(200, 10, 1000, 5, 0)
    deck2_str = d2.render()
    path2 = tmp_path / "deck2.rad"
    path2.write_text(deck2_str)
    
    blocks2 = list(read_deck(str(path2)))
    
    model2 = Model()
    log2 = MessageLog()
    read_eos(blocks2[1], model2, log2)
    
    mat_id2, eos2, _ = model2.raw_eos[0]
    assert mat_id2 == 200
    assert eos2.params["c0"] == 5.0
    assert eos2.params["c1"] == 1000.0

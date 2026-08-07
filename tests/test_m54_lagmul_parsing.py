import pytest
from pyradioss.input.deck_reader import read_deck
from pyradioss.model.model import Model
from pyradioss.input.starter_keywords import read_inter
from pyradioss.common.messages import MessageLog

def test_read_inter_lagmul_type16(tmp_path):
    deck = """/FORMAT/1
/INTER/LAGMUL/TYPE16/1
Interface 1
         1         2
                   1
"""
    p = tmp_path / "deck.rad"
    p.write_text(deck)
    model = Model()
    log = MessageLog()
    blocks = list(read_deck(str(p)))
    read_inter(blocks[1], model, log)
    assert len(model.interfaces) == 1
    i16 = model.interfaces[0]
    assert i16.type == 16
    assert i16.grnod_id == 1
    assert i16.grbric_id1 == 2
    assert i16.itied == 1
    assert i16.lagmul is True

def test_read_inter_lagmul_type17(tmp_path):
    deck = """/FORMAT/1
/INTER/LAGMUL/TYPE17/2
Interface 2
         2         3
                   0
"""
    p = tmp_path / "deck.rad"
    p.write_text(deck)
    model = Model()
    log = MessageLog()
    blocks = list(read_deck(str(p)))
    read_inter(blocks[1], model, log)
    assert len(model.interfaces) == 1
    i17 = model.interfaces[0]
    assert i17.type == 17
    assert i17.grbric_id1 == 2
    assert i17.grbric_id2 == 3
    assert i17.itied == 0
    assert i17.lagmul is True

def test_read_inter_lagmul_type2(tmp_path):
    # TYPE2 requires Card 2: grnd_IDs surf_IDm Isearch dsearch
    # CFG: %10d%10d%30s%10d%20s%20lg
    deck = "/FORMAT/1\n/INTER/LAGMUL/TYPE2/3\nInterface 3\n"
    card1 = f"{4:>10}{5:>10}{'':>30}{0:>10}{'':>20}{1.5:>20}\n"
    deck += card1
    p = tmp_path / "deck.rad"
    p.write_text(deck)
    model = Model()
    log = MessageLog()
    blocks = list(read_deck(str(p)))
    read_inter(blocks[1], model, log)
    assert len(model.interfaces) == 1
    i2 = model.interfaces[0]
    assert i2.type == 2
    assert i2.grnod_id == 4
    assert i2.surf_id == 5
    assert i2.dsearch == 1.5
    assert i2.lagmul is True

def test_read_inter_lagmul_type7(tmp_path):
    # TYPE7 requires 4 cards:
    # Card 2: grnd_IDs  surf_IDm (20 chars)
    # Card 3: blank
    # Card 4: blank
    # Card 5: _BLANK_(40) Gapmin(20)
    deck = "/FORMAT/1\n/INTER/LAGMUL/TYPE7/4\nInterface 4\n"
    card2 = f"{6:>10}{7:>10}\n"
    card3 = "\n"
    card4 = "\n"
    card5 = f"{'':>40}{0.02:>20}\n"
    deck += card2 + card3 + card4 + card5
    p = tmp_path / "deck.rad"
    p.write_text(deck)
    model = Model()
    log = MessageLog()
    blocks = list(read_deck(str(p)))
    read_inter(blocks[1], model, log)
    assert len(model.interfaces) == 1
    i7 = model.interfaces[0]
    assert i7.type == 7
    assert i7.grnod_id == 6
    assert i7.surf_id == 7
    assert i7.gap == 0.02
    assert i7.lagmul is True

import pytest

from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.input.deck_reader import read_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog


def test_starter_engine_ignore():
    """M82: Engine output requests are silently bypassed by the starter."""
    # A mix of starter and engine keywords
    text = """/BEGIN
10 0
/ANIM/DT
0.0 0.1
/MON/ON
/PRINT
1000
/H3D/DT
0.0 0.1
/NODE
1 0.0 0.0 0.0
/END"""
    with open("m82_test.rad", "w") as f:
        f.write(text)

    log = MessageLog()
    model = Model()
    blocks = read_deck("m82_test.rad")
    parse_starter_deck(blocks, model, log)
    
    # The starter should not complain about /ANIM, /MON, /PRINT, /H3D
    assert len(log.warnings) == 0, f"Unexpected warnings: {log.warnings}"

    assert len(model.node_ids) == 1

def test_engine_print_cards():
    """M82: Engine parsing of /PRINT cards."""
    text = """/PRINT
1000
/TFILE
0.1
"""
    with open("m82_engine.rad", "w") as f:
        f.write(text)

    log = MessageLog()
    blocks = read_deck("m82_engine.rad")
    ec = parse_engine_deck(blocks, log)
    
    assert ec.print_cycles == 1000
    assert ec.th_dt == 0.1


import pytest

from pyradioss.input.deck_reader import read_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.common.messages import MessageLog

def test_impl_dt_fixp():
    # A simple engine deck with /IMPL/DT/FIXP and some time values spread across cards
    # Maximum 100 values allowed by Fortran, we'll just test a few on multiple lines.
    deck = """/RUN/TEST/1
100.0
/IMPL/DT/FIXP
0.01 0.05 0.12 0.3
0.45 0.6 1.2
3.5 10.0
"""
    with open("m83_test_engine.rad", "w") as f:
        f.write(deck)

    log = MessageLog()
    blocks = read_deck("m83_test_engine.rad")
    ec = parse_engine_deck(blocks, log)
    
    assert ec.impl_dt_fixp == [0.01, 0.05, 0.12, 0.3, 0.45, 0.6, 1.2, 3.5, 10.0]
    
    # We should also ensure that no "not ported" warning was generated
    assert not any("not ported" in rec.message for rec in log.warnings)

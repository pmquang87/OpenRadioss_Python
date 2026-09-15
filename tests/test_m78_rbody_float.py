import pytest
from pyradioss.input.deck_reader import KeywordBlock, _to_int

def test_deck_reader_float_id():
    # Simulate how deck_reader.py creates blocks with float IDs
    # Now _to_int handles '1001.0' -> 1001
    assert _to_int('1001.0') == 1001
    assert _to_int('500') == 500

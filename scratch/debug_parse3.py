from pyradioss.input.deck_reader import KeywordBlock
from pyradioss.input.prop_reader import _data_cards, _get, _row
lines = [
    "/PROP/SPR_PRE/2",
    "prop_spring",
    "                                                                      ",
    "           13744.468                                                                                ",
    "         1                                                                                          "
]
block = KeywordBlock(lines, "PROP_SPR_PRE", False)
title, cards, fixed = _data_cards(block)
print(f'fixed={fixed}')
print(f'cards={len(cards)}')
if len(cards) > 0:
    print(f'cards[0].raw = {cards[0].raw!r}')
    print(f'cards[0].is_blank = {cards[0].is_blank}')

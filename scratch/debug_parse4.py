from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_prop
from pyradioss.model.model import Model
from pyradioss.input.prop_reader import _data_cards

deck = "scratch/test_deck.rad"
body = (
    "/BEGIN\nm40-sprpre\n      2021         0\n"
    "                  kg                  mm                  ms\n"
    "/PROP/SPR_PRE/2\n"
    "prop_spring\n"
    "                                                                      \n"
    "           13744.468                                            "
    "                    \n"
    "                   1                                                   \n"
    "/END\n"
)
with open(deck, 'w') as f:
    f.write(body)

for b in read_deck(deck):
    if b.parts and b.parts[0] == "PROP":
        title, cards, fixed = _data_cards(b)
        print(f'title={title!r}')
        print(f'fixed={fixed}')
        print(f'len(cards)={len(cards)}')
        for i, c in enumerate(cards):
            print(f'card {i}: raw={c.raw!r}')

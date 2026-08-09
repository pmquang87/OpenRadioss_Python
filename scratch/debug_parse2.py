from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_prop
from pyradioss.model.model import Model

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

model, log = Model(), MessageLog()
for b in read_deck(deck):
    if b.parts and b.parts[0] == "PROP":
        read_prop(b, model, log)

print("PARAMS:", model.properties[2].params)

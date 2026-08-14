import sys
from pyradioss.input.deck_reader import read_deck
from pyradioss.common.messages import MessageLog
from pyradioss.input.starter_keywords import parse_starter_deck

log = MessageLog()
deck = read_deck(r"C:\Users\pmqua\.gemini\antigravity\brain\93ac0308-1509-4fd5-93df-987958311651\scratch\RD-E-4801\One_solid_element_validation_law83\run01_no_failure\KS2_model_v01_0000.rad")
from pyradioss.model.model import Model
model = Model()
parse_starter_deck(deck, model, log)

for m in model.materials.values():
    if getattr(m, "law_name", None) == "LAW83" or getattr(m, "law", None) == 83:
        print("LAW83 PARAMS:", m.params.keys())
        for k, v in m.params.items():
            print(k, ":", v)

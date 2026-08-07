from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.starter.initialization import resolve_surfaces, build_element_groups, resolve_entity_groups

blocks = read_deck('scratch/airbag_test_2/Model/Gravity/gravity_0000.rad')
model = Model()
log = MessageLog()
parse_starter_deck(blocks, model, log)
print(f"Number of SH3N elements: {model.sh3n.n if model.sh3n else 0}")
print(f"First 10 SH3N IDs: {model.sh3n.ids[:10] if model.sh3n else []}")

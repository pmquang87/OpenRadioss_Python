from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.starter.initialization import resolve_surfaces, build_element_groups
from pyradioss.starter.airbag import initialize_monitored_volumes

blocks = read_deck('scratch/airbag_test/27_Football_shoot/Bathenay_circular/BAT_CIR_0000.rad')
model = Model()
log = MessageLog()
parse_starter_deck(blocks, model, log)
build_element_groups(model, log)
resolve_surfaces(model, log)
initialize_monitored_volumes(model)

mv = model.monitored_volumes[1]
print(f"Computed Surface: {mv.area}")
print(f"Computed Volume:  {mv.volume}")

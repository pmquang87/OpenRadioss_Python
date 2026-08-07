from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.starter.initialization import resolve_surfaces, build_element_groups, resolve_entity_groups
from pyradioss.starter.airbag import initialize_monitored_volumes

blocks = read_deck('scratch/airbag_test_2/Model/Gravity/gravity_0000.rad')
model = Model()
log = MessageLog()
parse_starter_deck(blocks, model, log)
resolve_entity_groups(model, log)
build_element_groups(model, log)
resolve_surfaces(model, log)
initialize_monitored_volumes(model)

print(f"SH3N elements: {model.sh3n.n if model.sh3n else 0}")
print(f"SH3N group 1 size: {len(model.egroups['SH3N'][1].elem_ids)}")
print(f"SH3N group 1 resolved size: {len(model.egroups['SH3N'][1].members) if model.egroups['SH3N'][1].members else 0}")

mv = model.monitored_volumes[1]
print(f"Computed Surface: {mv.area}")
print(f"Computed Volume:  {mv.volume}")

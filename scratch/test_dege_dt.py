import numpy as np
from tests.test_m40_residuals import _wedge_starter
from pyradioss.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.starter.parser import parse_starter_deck
from pyradioss.starter.initialization import build_element_groups, initialize_elements_and_mass, resolve_materials, resolve_node_groups, resolve_surfaces

deck = _wedge_starter()
model = Model()
log = MessageLog()
parse_starter_deck(deck, model, log)
build_element_groups(model, log)
resolve_node_groups(model, log)
resolve_surfaces(model, log)
resolve_materials(model, log)
initialize_elements_and_mass(model, log)

for name, group in model.element_groups():
    if hasattr(group, 'state') and 'dtfac' in group.state:
        print(name, 'dege dtfac:', group.state['dtfac'])

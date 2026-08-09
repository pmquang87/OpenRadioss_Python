import sys
import numpy as np
from pathlib import Path
from pyradioss.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.input.deck_reader import read_deck
from tests.test_m40_residuals import _wedge_starter
from pyradioss.starter.initialization import build_element_groups, initialize_elements_and_mass, resolve_materials, resolve_node_groups, resolve_surfaces

try:
    deck_str = _wedge_starter()
    tmp = Path('scratch/tmp_wedge.rad')
    tmp.write_text(deck_str)
    
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(tmp)), model, log)
    build_element_groups(model, log)
    resolve_node_groups(model, log)
    resolve_surfaces(model, log)
    resolve_materials(model, log)
    initialize_elements_and_mass(model, log)

    for name, group in model.element_groups():
        if hasattr(group, 'state') and 'dtfac' in group.state:
            print(name, 'dtfac:', group.state['dtfac'])
except Exception as e:
    import traceback
    traceback.print_exc()

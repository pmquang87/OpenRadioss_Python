import sys
import numpy as np
from pathlib import Path
from pyradioss.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.input.deck_reader import read_deck
from pyradioss.starter.initialization import build_element_groups, initialize_elements_and_mass, resolve_materials, resolve_node_groups, resolve_surfaces

deck_str = '''/BEGIN
tetra
/NODE
1 0 0 0
2 10 0 0
3 0 10 0
4 0 0 10
/TETRA4/1
1 1 2 3 4
/PART/1
part
100001 1
/MAT/LAW1/1
mat
1.0
221.0 0.3
/PROP/SOLID/1
prop
/END'''

tmp = Path('scratch/tmp_tetra.rad')
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

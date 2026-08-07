from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.starter.initialization import resolve_surfaces, build_element_groups

rad = """/PART/1
Part 1
1 1 1
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 0.0 0.0 1.0
6 1.0 0.0 1.0
7 1.0 1.0 1.0
8 0.0 1.0 1.0
/SHELL/1
100 1 1 2 3 4
101 1 5 6 7 8
/SURF/PART/10
10
1
"""

with open("scratch/test_airbag_debug.rad", "w") as f:
    f.write(rad)

blocks = read_deck("scratch/test_airbag_debug.rad")
model = Model()
log = MessageLog()
parse_starter_deck(blocks, model, log)
build_element_groups(model, log)
resolve_surfaces(model, log)

surf = model.surfaces[10]
print("Segments node indices:")
print(surf.segments)
x = model.x0
print("Coords of segment 0:")
print(x[surf.segments[0]])

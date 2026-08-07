from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.starter.airbag import initialize_monitored_volumes

# We need a small surface
rad = """/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 0.0 0.0 1.0
6 1.0 0.0 1.0
7 1.0 1.0 1.0
8 0.0 1.0 1.0
/SHELL
100 1 1 2 3 4
101 1 5 6 7 8
/SURF/PART/10
10
1
/MONVOL/AIRBAG1/1
My Airbag
10 0.1
1.0 1.0 1.0 1.0 1.0
5 1.4 1.0e-4 300.0 0 1
0
0
"""

with open("scratch/test_airbag2.rad", "w") as f:
    f.write(rad)

blocks = read_deck("scratch/test_airbag2.rad")
model = Model()
log = MessageLog()
parse_starter_deck(blocks, model, log)

# But wait, to get surf.segments, we need to run resolve_surfaces!
from pyradioss.starter.initialization import resolve_surfaces
resolve_surfaces(model, log)
initialize_monitored_volumes(model)

mv = model.monitored_volumes[1]
print(f"Area: {mv.area}")
print(f"Volume: {mv.volume}")

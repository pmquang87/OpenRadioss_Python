from pyradioss.model.model import Model
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import KEYWORD_PARSERS
from pyradioss.common.messages import MessageLog

deck = """/MONVOL/AIRBAG1/1/My Airbag
# surf_IDex hconv
10 0.1
# scale_t scale_p scale_s scale_a scale_d
1.0 1.0 1.0 1.0 1.0
# matid mu pext t_initial iequil ittf
5 1.4 1.0e-4 300.0 0 1
# nb_jet
2
# inject_ID sensor ijet node1 node2 node3 (fct_pt fct_theta fct_delta fscale_pt fscale_ptheta fscale_pdelta)
1 0 1 100 101 102
10 11 12 1.0 1.0 1.0
2 0 0 200 201 202
# nb_vent nb_porous
1 0
# surf_IDv Iform Avent Bvent (vent_title)
20 0 0.1 0.2                     Venthole 1
# tstart tstop dpdef dtpdef idtpdef
0.0 0.1 1e5 0.0 0
"""

with open("scratch/test_airbag.rad", "w", encoding="utf-8") as f:
    f.write(deck)

log = MessageLog()
model = Model()
blocks = read_deck("scratch/test_airbag.rad")
for b in blocks:
    parser = KEYWORD_PARSERS.get(b.key0)
    if parser:
        parser(b, model, log)
    else:
        print(f"No parser for {b.key0}")

print("Parsed monitored volumes:")
for mvid, mv in model.monitored_volumes.items():
    print(f"ID: {mvid}, Title: {mv.title}")
    print(f"  Type: {mv.vol_type}, Surf: {mv.surf_id}")
    print(f"  Injectors: {mv.injectors}")
    print(f"  Vents: {mv.vents}")

for m in log.messages:
    print(m)

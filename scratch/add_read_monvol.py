import re

with open("pyradioss/input/starter_keywords.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Add MonitoredVolume to imports
text = text.replace(
    "Sensor, Surface, THRequest,",
    "MonitoredVolume, Sensor, Surface, THRequest,"
)

# 2. Add the read_monvol function before KEYWORD_PARSERS
read_monvol_func = """
def read_monvol(block: KeywordBlock, model: Model, log: MessageLog):
    \"\"\"
    /MONVOL/AIRBAG1
    card 1: surf_IDex hconv
    card 2: scale_t scale_p scale_s scale_a scale_d
    card 3: matid mu pext t_initial iequil ittf
    card 4: nb_jet
    card 5: inject_ID sensor ijet node1 node2 node3
    ... (nb_jet lines)
    card x: nb_vent nb_porous
    ... (vent lines)
    \"\"\"
    vol_type = block.parts[1].upper() if len(block.parts) > 1 else ""
    if vol_type != "AIRBAG1":
        log.warning(f"/MONVOL/{vol_type} not ported - skipped (supported: AIRBAG1)",
                    block.source)
        return

    title, cards = _title_and_data(block)
    if len(cards) < 4:
        log.error(f"/MONVOL/AIRBAG1 needs at least 4 cards, got {len(cards)}",
                  block.source)
        return

    # Card 1: surf_IDex hconv
    t1 = cards[0].tokens()
    surf_id = int(t1[0]) if len(t1) > 0 else 0
    hconv = float(t1[1]) if len(t1) > 1 else 0.0

    # Card 2: scale_t scale_p scale_s scale_a scale_d
    t2 = cards[1].tokens()
    scale_t = float(t2[0]) if len(t2) > 0 else 1.0
    scale_p = float(t2[1]) if len(t2) > 1 else 1.0
    scale_s = float(t2[2]) if len(t2) > 2 else 1.0
    scale_a = float(t2[3]) if len(t2) > 3 else 1.0
    scale_d = float(t2[4]) if len(t2) > 4 else 1.0

    # Card 3: matid mu pext t_initial iequil ittf
    t3 = cards[2].tokens()
    matid = int(t3[0]) if len(t3) > 0 else 0
    mu = float(t3[1]) if len(t3) > 1 else 0.0
    pext = float(t3[2]) if len(t3) > 2 else 0.0
    t_init = float(t3[3]) if len(t3) > 3 else 293.0
    iequil = int(t3[4]) if len(t3) > 4 else 0
    ittf = int(t3[5]) if len(t3) > 5 else 0

    mv = MonitoredVolume(
        id=block.user_id,
        title=title,
        vol_type=vol_type,
        surf_id=surf_id,
        hconv=hconv,
        matid=matid,
        mu=mu,
        pext=pext,
        t_initial=t_init,
        iequil=iequil,
        ittf=ittf,
        scale_t=scale_t,
        scale_p=scale_p,
        scale_s=scale_s,
        scale_a=scale_a,
        scale_d=scale_d
    )

    # Injectors
    t4 = cards[3].tokens()
    nb_jet = int(t4[0]) if len(t4) > 0 else 0
    card_idx = 4
    
    for _ in range(nb_jet):
        if card_idx >= len(cards):
            break
        tj = cards[card_idx].tokens()
        ijet = int(tj[2]) if len(tj) > 2 else 0
        inj = {
            "inject_ID": int(tj[0]) if len(tj) > 0 else 0,
            "sensor": int(tj[1]) if len(tj) > 1 else 0,
            "ijet": ijet,
            "node1": int(tj[3]) if len(tj) > 3 else 0,
            "node2": int(tj[4]) if len(tj) > 4 else 0,
            "node3": int(tj[5]) if len(tj) > 5 else 0,
        }
        if ijet > 0:
            if card_idx + 1 < len(cards):
                card_idx += 1
                tjp = cards[card_idx].tokens()
                inj.update({
                    "fct_pt": int(tjp[0]) if len(tjp) > 0 else 0,
                    "fct_theta": int(tjp[1]) if len(tjp) > 1 else 0,
                    "fct_delta": int(tjp[2]) if len(tjp) > 2 else 0,
                    "fscale_pt": float(tjp[3]) if len(tjp) > 3 else 1.0,
                    "fscale_ptheta": float(tjp[4]) if len(tjp) > 4 else 1.0,
                    "fscale_pdelta": float(tjp[5]) if len(tjp) > 5 else 1.0,
                })
        mv.injectors.append(inj)
        card_idx += 1

    # Ventholes & Porous surfaces
    if card_idx < len(cards):
        tv = cards[card_idx].tokens()
        nb_vent = int(tv[0]) if len(tv) > 0 else 0
        nb_porous = int(tv[1]) if len(tv) > 1 else 0
        card_idx += 1
        
        # We will skip fully parsing vent lines for now unless needed,
        # but we can parse the basic ID to avoid losing alignment
        # Wait, each venthole is 3 to 4 lines!
        # Ventholes:
        for _ in range(nb_vent):
            if card_idx >= len(cards): break
            t_v1 = cards[card_idx].tokens()
            iform = int(t_v1[1]) if len(t_v1) > 1 else 0
            vent = {
                "surf_IDv": int(t_v1[0]) if len(t_v1) > 0 else 0,
                "Iform": iform,
                "Avent": float(t_v1[2]) if len(t_v1) > 2 else 0.0,
                "Bvent": float(t_v1[3]) if len(t_v1) > 3 else 0.0,
                # title is cols 41-60
                "vent_title": cards[card_idx].raw[40:60].strip() if cards[card_idx].fixed else ""
            }
            card_idx += 1
            if card_idx < len(cards):
                t_v2 = cards[card_idx].tokens()
                vent.update({
                    "tstart": float(t_v2[0]) if len(t_v2) > 0 else 0.0,
                    "tstop": float(t_v2[1]) if len(t_v2) > 1 else 1e30,
                    "dpdef": float(t_v2[2]) if len(t_v2) > 2 else 0.0,
                    "dtpdef": float(t_v2[3]) if len(t_v2) > 3 else 0.0,
                    "idtpdef": int(t_v2[4]) if len(t_v2) > 4 else 0,
                })
                card_idx += 1
            if card_idx < len(cards):
                card_idx += 1 # Skip fct_IDt etc.
            if card_idx < len(cards):
                card_idx += 1 # Skip fct_IDt' etc.
            if iform == 2:
                if card_idx < len(cards):
                    card_idx += 1
            mv.vents.append(vent)
            
        for _ in range(nb_porous):
            if card_idx >= len(cards): break
            # Each porous surface is 3 lines
            card_idx += 3

    model.monitored_volumes[block.user_id] = mv

"""

if "def read_monvol" not in text:
    text = text.replace("KEYWORD_PARSERS: Dict[str, Callable] = {", read_monvol_func + "\nKEYWORD_PARSERS: Dict[str, Callable] = {\n    \"MONVOL\": read_monvol,")
    with open("pyradioss/input/starter_keywords.py", "w", encoding="utf-8") as f:
        f.write(text)
    print("Added read_monvol")
else:
    print("Already exists")

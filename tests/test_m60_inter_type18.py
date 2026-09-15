import pytest
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_inter
from pyradioss.model.model import Model


def test_inter_type18_parsing(tmp_path):
    """Verify /INTER/TYPE18 is parsed correctly."""
    d = StarterDeck("test")
    # Using the standard 3-card structure for TYPE18
    # grnod_id surf_id grbric_id ibag idel18
    # stfac gap
    # stiff_dc sort_fact
    d.raw_block("INTER/TYPE18/10", [
        "       100       200       300                              " + "         1         2",
        "                50.0" + "                    " + "                 1.5",
        "                                        " + "                 0.1" + "                    " + "                 0.5"
    ])
    
    deck_str = d.render()
    path = tmp_path / "deck.rad"
    path.write_text(deck_str)
    
    blocks = list(read_deck(str(path)))
    
    model = Model()
    from pyradioss.common.messages import MessageLog
    log = MessageLog()
    
    read_inter(blocks[1], model, log)
    
    assert len(model.interfaces) == 1
    inter = model.interfaces[0]
    
    assert inter.id == 10
    assert inter.type == 18
    assert inter.grnod_id == 100
    assert inter.surf_id == 200
    assert inter.grbric_id1 == 300
    assert inter.ibag == 1
    assert inter.idel18 == 2
    assert inter.stfac == 50.0
    assert inter.gap == 1.5
    assert inter.stiff_dc == 0.1
    assert inter.sort_fact == 0.5

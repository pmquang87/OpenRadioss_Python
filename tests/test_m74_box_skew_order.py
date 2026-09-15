"""M74: Fix resolve order for /BOX with /SKEW references in /GRNOD"""

import pytest
from pyradioss.input.deck_reader import KeywordBlock, Card
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.starter.starter import check_model, resolve_skews, resolve_node_groups
from pyradioss.starter.initialization import resolve_entity_groups

def test_box_with_skew_resolve_order():
    """Ensure that /SKEW is resolved before /GRNOD evaluates /BOX."""
    model = Model()
    log = MessageLog()

    # Minimal deck: Node 1, 2, 3 to define SKEW; /SKEW/MOV; /BOX; /GRNOD
    blocks = [
        KeywordBlock("NODE", ["NODE"], None, [
            Card("         1                   0                   0                   0", ""),
            Card("         2                   1                   0                   0", ""),
            Card("         3                   0                   1                   0", "")
        ]),
        KeywordBlock("SKEW/MOV", ["SKEW", "MOV"], 10, [
            Card("Global skew", ""),
            Card("         1         2         3         X", "")
        ], fixed=True),
        KeywordBlock("BOX/RECTA", ["BOX", "RECTA"], 20, [
            Card("Box", ""),
            Card("                                      10", ""),
            Card("                 -10                 -10                 -10", ""),
            Card("                  10                  10                  10", "")
        ], fixed=True),
        KeywordBlock("GRNOD/BOX", ["GRNOD", "BOX"], 30, [
            Card("Group", ""),
            Card("                  20", "") # Positive BOX id
        ], fixed=True)
    ]

    parse_starter_deck(blocks, model, log)
    
    # Run the exact sequence from starter.py
    resolve_entity_groups(model, log)
    resolve_skews(model, log)
    resolve_node_groups(model, log)

    # There should be no ERRORs about "unknown /SKEW"
    assert len(log.errors) == 0, f"Unexpected errors: {log.errors}"

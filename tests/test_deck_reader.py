"""Unit tests for the deck lexer (pyradioss/input/deck_reader.py)."""

import numpy as np

from pyradioss.input.deck_reader import read_deck


def test_blocks_ids_comments_and_include(tmp_path):
    mesh = tmp_path / "mesh.inc"
    mesh.write_text(
        "/NODE\n"
        "         1                 0.0                 0.0                 0.0\n"
        "         2                 1.0                 0.0                 0.0\n"
    )
    main = tmp_path / "main_0000.rad"
    main.write_text(
        "# a banner comment\n"
        "$ another comment style\n"
        "/BEGIN\n"
        "my title\n"
        "#include mesh.inc\n"
        "/MAT/LAW2/17\n"
        "steel\n"
        "7.8e-6\n"
        "210. 0.3\n"
        "0.4 0.5 0.5\n"
        "/END\n"
    )
    blocks = read_deck(str(main))
    kws = [b.keyword for b in blocks]
    assert kws == ["BEGIN", "NODE", "MAT/LAW2", "END"]
    mat = blocks[2]
    assert mat.user_id == 17
    assert mat.key0 == "MAT"
    assert mat.cards[0].raw.strip() == "steel"
    # include was spliced in place, its 2 node cards intact
    node = blocks[1]
    assert len(node.cards) == 2
    assert node.cards[1].floats() == [2.0, 1.0, 0.0, 0.0]


def test_fixed_fields_and_fortran_reals(tmp_path):
    f = tmp_path / "x_0000.rad"
    f.write_text("/TEST/5\n" + "A".ljust(10) + "B".ljust(10) + "\n"
                 "1.5D-3 2d0\n")
    b = read_deck(str(f))[0]
    assert b.cards[0].fields()[0] == "A"
    assert b.cards[0].fields()[1] == "B"
    assert b.cards[1].floats() == [1.5e-3, 2.0]

"""Tests for /PROP/INT_BEAM (/PROP/TYPE18) fixed-format card parsing with blank lines.

Upstream Fortran reference:
- ``starter/source/properties/beam/hm_read_prop18.F``
- ``starter/source/properties/beam/defbeam_sect_new.F90``

Verifies that blank cards in fixed-format decks (e.g. Card 2 for DM, DF damping)
are preserved as default/zero values and do NOT shift subsequent cards left,
which previously caused Card 4 (W_DOF) to be parsed as float in deck 3040.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import PropType18
from pyradioss.model.model import Model


def _parse_deck_str(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "DECK_0000.rad"
    p.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


class TestPropIntBeamBlankCards:
    """Verifies that fixed-format /PROP/INT_BEAM preserves blank cards."""

    def test_prop_int_beam_fixed_format_blank_card2_discrete_fiber(self, tmp_path: Path):
        """Card 2 (DM, DF damping) is blank; ensure Card 3 (NIP) and Card 4 (fibers)

        and Card 5 (W_DOF) are not shifted left.
        This reproduces and tests the fix for deck 3040.
        """
        deck = (
            "/BEGIN\n"
            "PROP_INT_BEAM_BLANK_CARD2_TEST\n"
            "      2022         0\n"
            "/PROP/INT_BEAM/1\n"
            "Fiber Beam Blank Card 2\n"
            "#  ISFLAG    ISMSTR\n"
            "         0         0\n"
            "# Card 2 (DM, DF) is BLANK:\n"
            "                                        \n"
            "#      NIP      IREF                  Y0                  Z0\n"
            "         1         0                 0.0                 0.0\n"
            "#                 Y                   Z                AREA\n"
            "                 0.0                 0.0                 2.5\n"
            "# W_DOF\n"
            "   000 000\n"
            "/END\n"
        )
        model, log = _parse_deck_str(tmp_path, deck)
        assert len(log.errors) == 0, f"Unexpected errors: {log.errors}"
        assert 1 in model.prop_int_beams
        prop = model.prop_int_beams[1]
        assert isinstance(prop, PropType18)

        # Blank Card 2 yields zero damping
        assert prop.dm == 0.0
        assert prop.df == 0.0

        # Card 3 NIP
        assert prop.nip == 1
        assert prop.iref == 0

        # Card 4 Fiber
        assert len(prop.ips) == 1
        assert prop.ips[0].y == 0.0
        assert prop.ips[0].z == 0.0
        assert np.isclose(prop.ips[0].area, 2.5)

        # Card 5 W_DOF
        assert prop.wx1 == 0
        assert prop.wy1 == 0
        assert prop.wz1 == 0
        assert prop.wx2 == 0
        assert prop.wy2 == 0
        assert prop.wz2 == 0

    def test_prop_int_beam_fixed_format_blank_card2_parametric(self, tmp_path: Path):
        """Parametric section (ISFLAG=1) with blank Card 2 (damping) and Card 5 (W_DOF)."""
        deck = (
            "/BEGIN\n"
            "PARAMETRIC_BEAM_BLANK_DAMP\n"
            "      2022         0\n"
            "/PROP/TYPE18/2\n"
            "Rectangular Fiber Beam\n"
            "#  ISFLAG    ISMSTR\n"
            "         1         0\n"
            "# Card 2 (DM, DF) is BLANK:\n"
            "                                        \n"
            "#      NIP      IREF                  Y0                  Z0\n"
            "         1         0                 0.0                 0.0\n"
            "#    NITRS      IREF                  L1                  L2                  L3                  L4\n"
            "         3         0                10.0                20.0                 0.0                 0.0\n"
            "#                 L5                  L6\n"
            "                 0.0                 0.0\n"
            "# W_DOF\n"
            "   000 000\n"
            "/END\n"
        )
        model, log = _parse_deck_str(tmp_path, deck)
        assert len(log.errors) == 0, f"Unexpected errors: {log.errors}"
        assert 2 in model.prop_int_beams
        prop = model.prop_int_beams[2]
        assert prop.dm == 0.0
        assert prop.df == 0.0
        assert prop.isflag == 1
        assert prop.nitrs == 3
        assert np.isclose(prop.l1, 10.0)
        assert np.isclose(prop.l2, 20.0)
        assert np.isclose(prop.area, 200.0)

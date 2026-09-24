"""Tests for /MAT/LAW25 fixed-format card parsing with blank lines (M543).

Upstream Fortran reference:
- ``starter/source/materials/mat/mat025/hm_read_mat25.F``
- ``starter/source/materials/mat/mat025/read_mat25_tsaiwu.F90``
- ``starter/source/materials/mat/mat025/read_mat25_crasurv.F90``

Verifies that blank cards in fixed-format decks (e.g. Card 5 for WPMAX, WPREF, IOFF)
are preserved as default/zero values and do NOT shift subsequent cards left.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_deck_str(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "DECK_0000.rad"
    p.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


class TestLaw25BlankCards:
    """Verifies that fixed format /MAT/LAW25 correctly preserves blank cards."""

    def test_law25_fixed_format_blank_card5_no_shift(self, tmp_path: Path):
        """Card 5 (WPMAX, WPREF, IOFF) is blank; ensure Card 6 (B, N, FMAX)

        and Card 7 (SIG_1YT, SIG_2YT, SIG_1YC, SIG_2YC, ALPHA) are not shifted left.
        This reproduces and tests the fix for deck 3583.
        """
        deck = (
            "/BEGIN\n"
            "LAW25_BLANK_CARD5_TEST\n"
            "      2022         0\n"
            "/MAT/LAW25/1\n"
            "TsaiWu_BlankCard5\n"
            "#              RHO_I               RHO_0\n"
            "              1.5e-9              1.5e-9\n"
            "#                E11                 E22                NU12     Iform                           E33\n"
            "            140000.0             10000.0                 0.3         0                      140000.0\n"
            "#                G12                 G23                 G31              EPS_f1              EPS_f2\n"
            "              5000.0              3000.0              5000.0                0.02                0.01\n"
            "#              EPST1               EPSM1               EPST2               EPSM2                DMAX\n"
            "               0.005                0.02               0.003                0.01                0.95\n"
            "# Card 5 (WPMAX, WPREF, IOFF) is BLANK:\n"
            "                                                                                \n"
            "#                  B                   N                FMAX\n"
            "                 0.5                 0.8                10.0\n"
            "#             SIGYT1              SIGYT2              SIGYC1              SIGYC2               ALPHA\n"
            "                20.0                50.0              1200.0               200.0                 1.0\n"
            "#            SIGYC12             SIGYT12                   C                EPDR                 ICC\n"
            "                80.0                70.0                 0.1                10.0                   1\n"
            "/END\n"
        )
        model, log = _parse_deck_str(tmp_path, deck)
        assert len(log.errors) == 0, f"Unexpected errors: {log.errors}"
        assert 1 in model.mat_law25s
        mat = model.mat_law25s[1]

        # Blank Card 5 yields default zeros
        assert mat.wpmax == 0.0
        assert mat.wpref == 0.0
        assert mat.ioff == 0

        # Card 6 must not be corrupted by Card 7
        assert mat.b == 0.5
        assert mat.n == 0.8
        assert mat.fmax == 10.0

        # Card 7 must have expected yield stresses (not shifted into Card 6)
        assert mat.sig_1yt == 20.0
        assert mat.sig_2yt == 50.0
        assert mat.sig_1yc == 1200.0
        assert mat.sig_2yc == 200.0
        assert mat.alpha == 1.0

        # Card 8
        assert mat.sig_12yc == 80.0
        assert mat.sig_12yt == 70.0

    def test_law25_fixed_format_with_dollar_comments(self, tmp_path: Path):
        """Verifies dollar comments ($) and hash comments (#) do not count as data cards."""
        deck = (
            "/BEGIN\n"
            "LAW25_COMMENTS_TEST\n"
            "      2022         0\n"
            "/MAT/LAW25/2\n"
            "TsaiWu_Comments\n"
            "$ Comment with dollar sign\n"
            "# Comment with hash\n"
            "              1.5e-9              1.5e-9\n"
            "            140000.0             10000.0                 0.3         0                      140000.0\n"
            "              5000.0              3000.0              5000.0                0.02                0.01\n"
            "               0.005                0.02               0.003                0.01                0.95\n"
            "                                                                                \n"
            "                 0.5                 0.8                10.0\n"
            "                20.0                50.0              1200.0               200.0                 1.0\n"
            "/END\n"
        )
        model, log = _parse_deck_str(tmp_path, deck)
        assert len(log.errors) == 0, f"Unexpected errors: {log.errors}"
        assert 2 in model.mat_law25s
        mat = model.mat_law25s[2]
        assert mat.rho0 == 1.5e-9
        assert mat.wpmax == 0.0
        assert mat.n == 0.8
        assert mat.sig_1yt == 20.0

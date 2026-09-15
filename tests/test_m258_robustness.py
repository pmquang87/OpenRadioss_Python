# -*- coding: utf-8 -*-
"""M258 regression tests: corpus robustness fixes.

Tests the three crash-fixes identified in the fresh coverage sweep:
1. LAW83 spotweld resolve — KeyError on missing ``ifun_n`` / ``nu``
2. LAW92/LAW69 fixed-format cut — NameError on undefined ``cut``
3. #include file-not-found — FileNotFoundError → clean error message
"""

import os
import tempfile
import textwrap

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _starter_run(deck_text: str) -> tuple:
    """Write *deck_text* to a temp file, lex + parse, return (model, log)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_0000.rad", delete=False, dir=tempfile.gettempdir()
    ) as fh:
        fh.write(textwrap.dedent(deck_text))
        path = fh.name
    try:
        blocks = read_deck(path)
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        return model, log
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 1. LAW83 spotweld resolve — missing ifun_n / nu
# ---------------------------------------------------------------------------

class TestLaw83Robustness:
    """LAW83 spotweld material should not crash if params lack ifun_n."""

    def test_resolve_missing_keys(self):
        """law83_spotweld.resolve() must tolerate missing ifun_n/ifun_t/id_yield."""
        from pyradioss.materials import law83_spotweld
        from pyradioss.model.entities import Material

        # Build a minimal Material without the ifun_* keys
        mat = Material(id=1, law=83, rho0=7800.0, title="test",
                       params={"E": 210000.0, "nu": 0.3})
        model = Model()
        log = MessageLog()
        # Should NOT raise KeyError
        law83_spotweld.resolve(mat, model, log)
        assert len(log.errors) == 0

    def test_law83_params_have_nu(self):
        """read_mat_law83 must set 'nu' in the Material.params dict."""
        deck = """\
        /BEGIN
        TENSILE_TEST
              2024         0
                                    kg  mm  ms
        /MAT/LAW83/1
        test_spotweld
              7850.               0.
           210000.         0
                 0       1.0       1.0       0.0       0.0
               0.0       0.0         0       0.0
                 0         0       1.0
        /END
        """
        model, log = _starter_run(deck)
        if 1 in model.materials:
            mat = model.materials[1]
            assert "nu" in mat.params, "LAW83 Material must have 'nu'"


# ---------------------------------------------------------------------------
# 2. LAW92/LAW69 fixed-format cut function
# ---------------------------------------------------------------------------

class TestHyperelasticCut:
    """The ``cut`` function must be available for hyperelastic readers."""

    def test_cut_function_importable(self):
        """cut, _f, _i must be importable from starter_keywords."""
        from pyradioss.input.starter_keywords import cut, _f, _i  # noqa: F401
        assert callable(cut)
        assert callable(_f)
        assert callable(_i)

    def test_cut_splits_correctly(self):
        """cut(raw, key) must split a raw line at the correct column widths."""
        from pyradioss.input.starter_keywords import cut
        # MAT_LAW92_1 layout = [20, 20]
        raw = "            7850.0                 0.0"
        fields = cut(raw, "MAT_LAW92_1")
        assert len(fields) == 2
        assert float(fields[0]) == pytest.approx(7850.0)

    def test_law92_layouts_exist(self):
        """All MAT_LAW92_* layout keys must exist in LAYOUTS."""
        from pyradioss.input.card_layouts import LAYOUTS
        for key in ("MAT_LAW92_1", "MAT_LAW92_2", "MAT_LAW92_3"):
            assert key in LAYOUTS, f"Missing layout: {key}"

    def test_law69_layouts_exist(self):
        """All MAT_LAW69_* layout keys must exist in LAYOUTS."""
        from pyradioss.input.card_layouts import LAYOUTS
        for key in ("MAT_LAW69_1", "MAT_LAW69_2", "MAT_LAW69_3"):
            assert key in LAYOUTS, f"Missing layout: {key}"


# ---------------------------------------------------------------------------
# 3. #include file-not-found → error, not crash
# ---------------------------------------------------------------------------

class TestIncludeFileNotFound:
    """Missing #include files should produce errors, not crashes."""

    def test_missing_include_produces_error(self):
        """read_deck must not crash on a missing #include file."""
        deck = """\
        /BEGIN
        test
              2024         0
                                    kg  mm  ms
        #include "this_file_does_not_exist_12345.inc"
        /END
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_0000.rad", delete=False, dir=tempfile.gettempdir()
        ) as fh:
            fh.write(textwrap.dedent(deck))
            path = fh.name
        try:
            blocks = read_deck(path)
            # Should NOT raise FileNotFoundError — should return blocks
            # including a synthetic __INCLUDE_ERROR__ block
            err_blocks = [b for b in blocks if b.keyword == "__INCLUDE_ERROR__"]
            assert len(err_blocks) == 1
            assert "this_file_does_not_exist_12345.inc" in err_blocks[0]._include_path
        finally:
            os.unlink(path)

    def test_missing_include_logged_as_error(self):
        """read_all_blocks must log the missing include as an error."""
        deck = """\
        /BEGIN
        test
              2024         0
                                    kg  mm  ms
        #include "nonexistent_99999.txt"
        /END
        """
        model, log = _starter_run(deck)
        # The error about the missing include should be logged
        assert any("include" in e.lower() or "not found" in e.lower()
                    for e in log.errors), \
            f"Expected include-not-found error, got: {log.errors}"


class TestMaterialElasticConstantsFallback:
    """Material E, nu, G, K should deduce elastic constants gracefully."""

    def test_deduce_from_shear_and_nu(self):
        from pyradioss.model.entities import Material
        mat = Material(id=1, law=82, rho0=1.0e-9, title="rubber",
                       params={"mu": 1.5, "nu": 0.49})
        assert mat.G == pytest.approx(1.5)
        assert mat.nu == pytest.approx(0.49)
        assert mat.E == pytest.approx(2.0 * 1.5 * 1.49)
        assert mat.K > 0.0
        assert mat.sound_speed_solid() > 0.0

    def test_default_fallback_without_params(self):
        from pyradioss.model.entities import Material
        mat = Material(id=2, law=0, rho0=1000.0, title="generic", params={})
        assert mat.nu == 0.3
        assert mat.E == 0.0
        assert mat.G == 0.0
        assert mat.K == 0.0


"""M437 -- /TH fixed-format parser robustness tests.

The ``_is_var_card`` heuristic in ``read_th`` (starter_keywords.py)
uses whitespace tokenisation to distinguish variable-name cards from
element-ID cards.  In fixed-format decks, this mis-classifies ID cards
whose element NAME text occupies the first whitespace token position
(the %10d ID field is blank/zero, and the %-80s name dominates).

The fix: in fixed-format mode, check the first 10 columns (the %10d
ID field) as an integer instead of tokenising by whitespace.

These tests exercise the exact card layouts from hm_cfg_files
``th_spring.cfg``, ``th_shel.cfg``, ``th_bric.cfg`` (radioss51 format):
  CARD(\"%10d          %-80s\", ids, NAME_ARRAY)
and the regression path for free-format blocks.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


# ---------------------------------------------------------------------------
# Helper: parse a deck string through the Starter pipeline
# ---------------------------------------------------------------------------

def _parse(tmp_path: Path, text: str):
    """Write *text* to a temp file, parse through the Starter, return
    (model, log)."""
    p = tmp_path / "TEST_TH_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def _count_th_errors(log: MessageLog) -> int:
    """Count only /TH-related errors (ignore unrelated prop/mat checks)."""
    return sum(1 for e in log.errors if "/TH" in e)


# ---------------------------------------------------------------------------
# Fixed-format deck template (spring variant -- no shell thickness needed)
# ---------------------------------------------------------------------------

_SPRING_DECK = """\
/BEGIN
{title}
      2019
                  kg                  mm                  ms
                  kg                  mm                  ms

/MAT/ELAST/1
elastic_mat
#              RHO_I
                7E-6
#                  E                  nu
             210000                 0.3
/PROP/TYPE13/1
spring_prop
/PART/1
part1
         1         1
/NODE
         1                0.0                0.0                0.0
         2               10.0                0.0                0.0
         3               10.0               10.0                0.0
         4                0.0               10.0                0.0
{elements}
{th_block}
/END
"""

_SHELL_DECK = """\
/BEGIN
{title}
      2019
                  kg                  mm                  ms
                  kg                  mm                  ms

/MAT/ELAST/1
elastic_mat
#              RHO_I
                7E-6
#                  E                  nu
             210000                 0.3
/PROP/SHELL/1
shell_prop
#   Ishell    Ismstr     Ish3n    Idrill                        Ithick
         1         0         0         0                             0
#                 hm                  hf                  hr                  dm                  dn
                 0.1                 0.0                 0.0                 0.0                 0.0
#        N    Istrain                              Thick
         5         1                              1.000
/PART/1
part1
         1         1
/NODE
         1                0.0                0.0                0.0
         2               10.0                0.0                0.0
         3               10.0               10.0                0.0
         4                0.0               10.0                0.0
{elements}
{th_block}
/END
"""

_SOLID_DECK = """\
/BEGIN
{title}
      2019
                  kg                  mm                  ms
                  kg                  mm                  ms

/MAT/ELAST/1
elastic_mat
#              RHO_I
                7E-6
#                  E                  nu
             210000                 0.3
/PROP/SOLID/1
solid_prop
#   Isolid    Ismstr                       Icpre   Inpts   Itetra4   Iframe                ICoSkew
        24         0                           0       0         0        0                       0
/PART/1
part1
         1         1
/NODE
         1                0.0                0.0                0.0
         2               10.0                0.0                0.0
         3               10.0               10.0                0.0
         4                0.0               10.0                0.0
         5                0.0                0.0               10.0
         6               10.0                0.0               10.0
         7               10.0               10.0               10.0
         8                0.0               10.0               10.0
{elements}
{th_block}
/END
"""


# ---------------------------------------------------------------------------
# Test 1: /TH/SPRING fixed-format with names in the ID cards
# ---------------------------------------------------------------------------

class TestTHSpringFixedFormat:
    """The /TH/SPRING ID card format (radioss51) is:
        CARD(\"%10d          %-80s\", ids, NAME_ARRAY)
    i.e. 10-char integer ID, 10 spaces, 80-char name.
    """

    def test_spring_th_with_names_parses_without_error(self, tmp_path):
        """A /TH/SPRING block with element names must parse cleanly."""
        deck = _SPRING_DECK.format(
            title="test_th_spring_names",
            elements="""\
/SPRING/1
         1         1         2""",
            th_block="""\
/TH/SPRING/1
spring_th
#  var_ID1   var_ID2   var_ID3   var_ID4   var_ID5   var_ID6   var_ID7   var_ID8   var_ID9  var_ID10
DEF       
#  elem_ID          elem_name
         1          Elastic                                                                         """,
        )
        model, log = _parse(tmp_path, deck)
        assert _count_th_errors(log) == 0, \
            f"TH errors: {[e for e in log.errors if '/TH' in e]}"
        th_req = [r for r in model.th_requests if r.kind == "SPRING"]
        assert len(th_req) == 1
        assert 1 in th_req[0].ids

    def test_spring_th_multiple_named_ids(self, tmp_path):
        """Multiple spring elements with text names."""
        deck = _SPRING_DECK.format(
            title="test_th_spring_multi",
            elements="""\
/SPRING/1
         1         1         2
/SPRING/2
         2         3         4""",
            th_block="""\
/TH/SPRING/2
spring_group
#  var_ID1   var_ID2   var_ID3   var_ID4   var_ID5   var_ID6   var_ID7   var_ID8   var_ID9  var_ID10
DEF       FX        
#  elem_ID          elem_name
         1          Spring_A_tension                                                                
         2          Spring_B_compression                                                            """,
        )
        model, log = _parse(tmp_path, deck)
        assert _count_th_errors(log) == 0, \
            f"TH errors: {[e for e in log.errors if '/TH' in e]}"
        th_req = [r for r in model.th_requests if r.kind == "SPRING"]
        assert len(th_req) == 1
        assert 1 in th_req[0].ids
        assert 2 in th_req[0].ids


# ---------------------------------------------------------------------------
# Test 2: /TH/SHEL fixed-format with skew=0 and element names
# ---------------------------------------------------------------------------

class TestTHShellFixedFormat:

    def test_shell_th_with_names_parses_without_error(self, tmp_path):
        """A /TH/SHEL block with element names must parse cleanly."""
        deck = _SHELL_DECK.format(
            title="test_th_shel_names",
            elements="""\
/SHELL/1
         1         1         1         2         3         4""",
            th_block="""\
/TH/SHEL/2
shell_history
#  var_ID1   var_ID2   var_ID3   var_ID4   var_ID5   var_ID6   var_ID7   var_ID8   var_ID9  var_ID10
DEF       EPSD      THIC      
#  elem_ID   skew_ID                                         elem_name
         1         0Master_group_1                                                                  """,
        )
        model, log = _parse(tmp_path, deck)
        assert _count_th_errors(log) == 0, \
            f"TH errors: {[e for e in log.errors if '/TH' in e]}"
        th_req = [r for r in model.th_requests if r.kind == "SHEL"]
        assert len(th_req) == 1
        assert 1 in th_req[0].ids


# ---------------------------------------------------------------------------
# Test 3: /TH/BRIC fixed-format with element names
# ---------------------------------------------------------------------------

class TestTHBrickFixedFormat:

    def test_brick_th_with_names_parses_without_error(self, tmp_path):
        """A /TH/BRIC block with element names must parse cleanly."""
        deck = _SOLID_DECK.format(
            title="test_th_bric_names",
            elements="""\
/BRICK/1
         1         1         1         2         3         4         5         6         7         8""",
            th_block="""\
/TH/BRIC/3
brick_history
#  var_ID1   var_ID2   var_ID3   var_ID4   var_ID5   var_ID6   var_ID7   var_ID8   var_ID9  var_ID10
DEF       
#  elem_ID   skew_ID                                         elem_name
         1         0Uniaxial_tension_0.333                                                          """,
        )
        model, log = _parse(tmp_path, deck)
        assert _count_th_errors(log) == 0, \
            f"TH errors: {[e for e in log.errors if '/TH' in e]}"
        th_req = [r for r in model.th_requests if r.kind == "BRIC"]
        assert len(th_req) == 1
        assert 1 in th_req[0].ids


# ---------------------------------------------------------------------------
# Test 4: Regression -- free-format /TH/SPRING still works
# ---------------------------------------------------------------------------

class TestTHFreeFormat:

    def test_free_format_spring_th_still_works(self, tmp_path):
        """Free-format /TH/SPRING must continue to work (regression)."""
        deck = """\
/BEGIN
test_th_free_spring
/NODE
1 0.0 0.0 0.0
2 10.0 0.0 0.0
/MAT/ELAST/1
mat
7E-6
210000 0.3
/PROP/TYPE13/1
spring_prop
/PART/1
part1
1 1
/SPRING/1
1 1 2
/TH/SPRING/1
spring_th
DEF
1
/END
"""
        model, log = _parse(tmp_path, deck)
        assert _count_th_errors(log) == 0, \
            f"TH errors: {[e for e in log.errors if '/TH' in e]}"
        th_req = [r for r in model.th_requests if r.kind == "SPRING"]
        assert len(th_req) == 1
        assert 1 in th_req[0].ids


# ---------------------------------------------------------------------------
# Test 5: Multi-line variable cards with extra variable names
# ---------------------------------------------------------------------------

class TestTHMultilineVars:

    def test_multiline_variables_then_named_ids(self, tmp_path):
        """Variable cards spanning two lines, followed by named ID cards."""
        deck = _SHELL_DECK.format(
            title="test_th_multivar",
            elements="""\
/SHELL/1
         1         1         1         2         3         4""",
            th_block="""\
/TH/SHEL/5
multivar_shell
#  var_ID1   var_ID2   var_ID3   var_ID4   var_ID5   var_ID6   var_ID7   var_ID8   var_ID9  var_ID10
DEF       EPSD      THIC      SIGXX     SIGYY     SIGXY     SIGYZ     SIGZX     IE        KE        
VONM      PLAS      
#  elem_ID   skew_ID                                         elem_name
         1         0Test_element_1                                                                  """,
        )
        model, log = _parse(tmp_path, deck)
        assert _count_th_errors(log) == 0, \
            f"TH errors: {[e for e in log.errors if '/TH' in e]}"
        th_req = [r for r in model.th_requests if r.kind == "SHEL"]
        assert len(th_req) == 1
        assert 1 in th_req[0].ids
        # Variable list should contain both default and extra vars
        assert "VONM" in th_req[0].variables
        assert "PLAS" in th_req[0].variables

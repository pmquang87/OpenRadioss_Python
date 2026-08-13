"""M68: parse /DEF_SHELL, /DEF_SOLID, /IOFLAG, /SPMD, and all /TH/* entity types.

Tests verify that:
1. /DEF_SHELL and /DEF_SOLID values land on ``model.def_shell`` / ``model.def_solid``.
2. /IOFLAG and /SPMD are accepted silently (no errors, no warnings).
3. /TH/RBODY, /TH/SHEL, /TH/SH3N, /TH/SPRING, /TH/BRIC, /TH/RWALL,
   /TH/SECTIO, /TH/INTER are parsed to THRequest objects.
"""
import io, contextlib, textwrap, pytest
from pathlib import Path

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse(tmp_path, tag, deck_text):
    """Write a deck, read it, and parse keywords into a Model.
    Returns (model, log) — parser level only, no element init."""
    path = tmp_path / f"{tag}_0000.rad"
    path.write_text(deck_text)
    blocks = read_deck(str(path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


# Free-format header (version 0 = legacy free)
_HEADER = """\
#RADIOSS STARTER
/BEGIN
global defaults test
         0         0
                  kg                   m                   s
                  kg                   m                   s
"""

_FOOTER = """\
/END
"""


# ---------------------------------------------------------------------------
# /DEF_SHELL
# ---------------------------------------------------------------------------

def test_def_shell_parsed(tmp_path):
    deck = _HEADER + """\
/DEF_SHELL
1 2 1 1 1 2 0
""" + _FOOTER
    m, log = _parse(tmp_path, "DS1", deck)
    assert not log.errors, log.errors
    assert m.def_shell['ishell'] == 1
    assert m.def_shell['ismstr'] == 2
    assert m.def_shell['ithick'] == 1
    assert m.def_shell['iplas'] == 1
    assert m.def_shell['istrain'] == 1
    assert m.def_shell['ish3n'] == 2
    assert m.def_shell['idrill'] == 0


def test_def_shell_defaults_when_absent(tmp_path):
    """Without /DEF_SHELL, model.def_shell is all zeros."""
    m, log = _parse(tmp_path, "DS0", _HEADER + _FOOTER)
    assert m.def_shell['ishell'] == 0
    assert m.def_shell['ish3n'] == 0


# ---------------------------------------------------------------------------
# /DEF_SOLID
# ---------------------------------------------------------------------------

def test_def_solid_parsed(tmp_path):
    deck = _HEADER + """\
/DEF_SOLID
14 4 1 1000 1000 0 1
""" + _FOOTER
    m, log = _parse(tmp_path, "DSO1", deck)
    assert not log.errors, log.errors
    assert m.def_solid['isolid'] == 14
    assert m.def_solid['ismstr'] == 4
    assert m.def_solid['icpre'] == 1
    assert m.def_solid['itetra4'] == 1000
    assert m.def_solid['itetra10'] == 1000
    assert m.def_solid['imas'] == 0
    assert m.def_solid['iframe'] == 1


# ---------------------------------------------------------------------------
# /IOFLAG — accepted silently
# ---------------------------------------------------------------------------

def test_ioflag_accepted(tmp_path):
    """Decks with /IOFLAG must not produce errors or 'not ported' warnings."""
    deck = _HEADER + """\
/IOFLAG
5 -1 3 0 0
""" + _FOOTER
    m, log = _parse(tmp_path, "IO1", deck)
    assert not log.errors, log.errors
    # No "not ported" warning for /IOFLAG
    ioflag_warns = [w for w in log.warnings if "IOFLAG" in w.upper()]
    assert not ioflag_warns, ioflag_warns


# ---------------------------------------------------------------------------
# /SPMD — accepted silently
# ---------------------------------------------------------------------------

def test_spmd_accepted(tmp_path):
    """Decks with /SPMD must not produce errors or 'not ported' warnings."""
    deck = _HEADER + """\
/SPMD
3 1 0 1
""" + _FOOTER
    m, log = _parse(tmp_path, "SP1", deck)
    assert not log.errors, log.errors
    spmd_warns = [w for w in log.warnings if "SPMD" in w.upper()]
    assert not spmd_warns, spmd_warns


# ---------------------------------------------------------------------------
# /TH/* — all entity types
# ---------------------------------------------------------------------------

def _th_deck(th_block):
    """Build a full deck wrapping an arbitrary /TH/... block."""
    return _HEADER + th_block + "\n" + _FOOTER


def test_th_rbody_parsed(tmp_path):
    deck = _th_deck("""\
/TH/RBODY/1
rbody_th
DEF
1
""")
    m, log = _parse(tmp_path, "THR1", deck)
    assert not log.errors, log.errors
    th = [r for r in m.th_requests if r.kind == "RBODY"]
    assert len(th) == 1
    assert 1 in th[0].ids
    assert "DX" in th[0].variables  # DEF expands to DX, DY, DZ, VX, VY, VZ


def test_th_shel_parsed(tmp_path):
    deck = _th_deck("""\
/TH/SHEL/2
shell_th
DEF
1
""")
    m, log = _parse(tmp_path, "THS1", deck)
    assert not log.errors, log.errors
    th = [r for r in m.th_requests if r.kind == "SHEL"]
    assert len(th) == 1
    assert 1 in th[0].ids
    assert "SIGXX" in th[0].variables


def test_th_spring_parsed(tmp_path):
    deck = _th_deck("""\
/TH/SPRING/3
spring_th
DEF
1
""")
    m, log = _parse(tmp_path, "THSP1", deck)
    assert not log.errors, log.errors
    th = [r for r in m.th_requests if r.kind == "SPRING"]
    assert len(th) == 1
    assert "FX" in th[0].variables


def test_th_bric_parsed(tmp_path):
    deck = _th_deck("""\
/TH/BRIC/4
brick_th
DEF
1
""")
    m, log = _parse(tmp_path, "THB1", deck)
    assert not log.errors, log.errors
    th = [r for r in m.th_requests if r.kind == "BRIC"]
    assert len(th) == 1
    assert "SIGZZ" in th[0].variables


def test_th_rwall_parsed(tmp_path):
    deck = _th_deck("""\
/TH/RWALL/5
rwall_th
DEF
1
""")
    m, log = _parse(tmp_path, "THRW1", deck)
    assert not log.errors, log.errors
    th = [r for r in m.th_requests if r.kind == "RWALL"]
    assert len(th) == 1
    assert "FN" in th[0].variables


def test_th_sectio_parsed(tmp_path):
    """TH/SECTIO normalises to kind='SECT'."""
    deck = _th_deck("""\
/TH/SECTIO/6
section_th
DEF
1
""")
    m, log = _parse(tmp_path, "THSE1", deck)
    assert not log.errors, log.errors
    th = [r for r in m.th_requests if r.kind == "SECT"]
    assert len(th) >= 1
    # SECTIO normalises to SECT, DEF -> FX, FY, FZ, MX, MY, MZ
    assert "FX" in th[-1].variables


def test_th_sh3n_parsed(tmp_path):
    deck = _th_deck("""\
/TH/SH3N/7
sh3n_th
DEF
1
""")
    m, log = _parse(tmp_path, "TH3N1", deck)
    assert not log.errors, log.errors
    th = [r for r in m.th_requests if r.kind == "SH3N"]
    assert len(th) == 1
    assert "SIGXX" in th[0].variables


def test_th_inter_parsed(tmp_path):
    deck = _th_deck("""\
/TH/INTER/8
inter_th
DEF
1
""")
    m, log = _parse(tmp_path, "THI1", deck)
    assert not log.errors, log.errors
    th = [r for r in m.th_requests if r.kind == "INTER"]
    assert len(th) == 1
    assert "FN" in th[0].variables

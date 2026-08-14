"""M89 – /GRNOD/NODENS subtype + /XREF reference geometry (Starter)."""
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.model.entities import Xref
from pyradioss.starter.starter import run_starter


# ──────────────────── reusable boilerplate ────────────────────────────
_ELEM = """\
/MAT/LAW1/1
Elastic
              7.8e-9
            210000.0                 0.3
/PROP/TYPE1/1
Shell_Prop
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/PART/1
Part_one
         1         1
/NODE
       101                 0.0                 0.0                 0.0
       102                10.0                 0.0                 0.0
       103                10.0                10.0                 0.0
       104                 0.0                10.0                 0.0
/SHELL/1
         1       101       102       103       104
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="utf-8")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


# ══════════════════════════════════════════════════════════════════════
#  /GRNOD/NODENS tests
# ══════════════════════════════════════════════════════════════════════

class TestGrnodNodens:
    """/GRNOD/NODENS — non-sorted node group subtype."""

    def test_nodens_free_format(self, tmp_path):
        """Free-format /GRNOD/NODENS stores node_ids preserving order."""
        deck = f"""\
/BEGIN
TEST_NODENS_FREE
{_ELEM}
/GRNOD/NODENS/10
My NODENS group
103 101 104 102
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 10 in model.node_groups
        g = model.node_groups[10]
        assert g.title == "My NODENS group"
        # Order must be preserved (NODENS = not sorted)
        assert g.node_ids == [103, 101, 104, 102]

    def test_nodens_fixed_format(self, tmp_path):
        """Fixed-format /GRNOD/NODENS with 10-col IDs."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_NODENS_FIXED
      2021         0
{_ELEM}
/GRNOD/NODENS/20
Fixed nodens
       104       103       102       101
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        g = model.node_groups[20]
        assert g.node_ids == [104, 103, 102, 101]


# ══════════════════════════════════════════════════════════════════════
#  /XREF tests
# ══════════════════════════════════════════════════════════════════════

class TestXref:
    """/XREF reference geometry keyword."""

    def test_xref_fixed_format(self, tmp_path):
        """Fixed-format /XREF with nitrs and node coordinate table."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_XREF_FIXED
      2021         0
{_ELEM}
/XREF/1
Reference state
       100
       101                 0.1                 0.2                 0.3
       102                10.1                 0.2                 0.3
       103                10.1                10.2                 0.3
       104                 0.1                10.2                 0.3
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 1 in model.xrefs
        xr = model.xrefs[1]
        assert isinstance(xr, Xref)
        assert xr.part_id == 1
        assert xr.title == "Reference state"
        assert xr.nitrs == 100
        assert len(xr.node_ids) == 4
        assert list(xr.node_ids) == [101, 102, 103, 104]
        assert abs(xr.coords[0, 0] - 0.1) < 1e-12
        assert abs(xr.coords[1, 0] - 10.1) < 1e-12
        assert abs(xr.coords[3, 2] - 0.3) < 1e-12

    def test_xref_free_format(self, tmp_path):
        """Free-format /XREF parsing."""
        deck = f"""\
/BEGIN
TEST_XREF_FREE
{_ELEM}
/XREF/1
Free ref state
50
101 0.5 0.6 0.7
102 10.5 0.6 0.7
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        xr = model.xrefs[1]
        assert xr.nitrs == 50
        assert len(xr.node_ids) == 2
        assert abs(xr.coords[0, 0] - 0.5) < 1e-12
        assert abs(xr.coords[1, 0] - 10.5) < 1e-12

    def test_xref_default_nitrs(self, tmp_path):
        """Default nitrs = 100 when card is blank or zero."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_XREF_DFLT
      2021         0
{_ELEM}
/XREF/1
Default nitrs
         0
       101                 0.0                 0.0                 0.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        xr = model.xrefs[1]
        # Fortran default: nitrs=100 when 0
        assert xr.nitrs == 100

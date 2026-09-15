"""M88 – /SUBDOMAIN domain-partition keyword (Starter)."""
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.model.entities import Subdomain
from pyradioss.starter.starter import run_starter


# ──────────────────── minimal element deck ───────────────────────────
# Reusable boilerplate: 1 mat, 1 prop, 1 part, 4 nodes, 1 shell.
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

# Same but with TWO parts (part 2 reuses the same mat/prop).
_ELEM2 = """\
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
/PART/2
Part_two
         1         1
/NODE
       101                 0.0                 0.0                 0.0
       102                10.0                 0.0                 0.0
       103                10.0                10.0                 0.0
       104                 0.0                10.0                 0.0
       201                20.0                 0.0                 0.0
       202                30.0                 0.0                 0.0
       203                30.0                10.0                 0.0
       204                20.0                10.0                 0.0
/SHELL/1
         1       101       102       103       104
/SHELL/2
         2       201       202       203       204
"""

# Three-part variant
_ELEM3 = """\
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
/PART/2
Part_two
         1         1
/PART/3
Part_three
         1         1
/NODE
       101                 0.0                 0.0                 0.0
       102                10.0                 0.0                 0.0
       103                10.0                10.0                 0.0
       104                 0.0                10.0                 0.0
       201                20.0                 0.0                 0.0
       202                30.0                 0.0                 0.0
       203                30.0                10.0                 0.0
       204                20.0                10.0                 0.0
       301                40.0                 0.0                 0.0
       302                50.0                 0.0                 0.0
       303                50.0                10.0                 0.0
       304                40.0                10.0                 0.0
/SHELL/1
         1       101       102       103       104
/SHELL/2
         2       201       202       203       204
/SHELL/3
         3       301       302       303       304
"""


# ──────────────────── helpers ────────────────────────────────────────
def _run(tmp_path, deck, name="TEST_0000.rad"):
    """Write deck, run starter, return (model, log)."""
    p = tmp_path / name
    p.write_text(deck, encoding="utf-8")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


# ──────────────────── tests ──────────────────────────────────────────

class TestSubdomainParsingFixed:
    """Fixed-format /SUBDOMAIN parsing with 10-col part ID lists."""

    def test_basic_fixed_format(self, tmp_path):
        """Two subdomains in fixed format, verify id/title/part_ids."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_SUBDOMAIN_FIXED
      2021         0
{_ELEM2}
/SUBDOMAIN/100
Domain A
         1
/SUBDOMAIN/200
Domain B
         2
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 100 in model.subdomains
        assert 200 in model.subdomains
        sd1 = model.subdomains[100]
        sd2 = model.subdomains[200]
        assert isinstance(sd1, Subdomain)
        assert sd1.id == 100
        assert sd1.title == "Domain A"
        assert sd1.part_ids == [1]
        assert sd2.id == 200
        assert sd2.title == "Domain B"
        assert sd2.part_ids == [2]

    def test_multi_part_ids(self, tmp_path):
        """Subdomain with multiple part IDs on a single data line."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_SUBDOMAIN_MULTI
      2021         0
{_ELEM3}
/SUBDOMAIN/10
Big domain
         1         2         3
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        sd = model.subdomains[10]
        assert sorted(sd.part_ids) == [1, 2, 3]
        assert sd.title == "Big domain"


class TestSubdomainParsingFree:
    """Free-format /SUBDOMAIN parsing."""

    def test_free_format(self, tmp_path):
        """Free-format deck with whitespace-separated part IDs."""
        deck = f"""\
/BEGIN
TEST_SUBDOMAIN_FREE
{_ELEM2}
/SUBDOMAIN/50
Free form domain
1 2
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        sd = model.subdomains[50]
        assert sd.id == 50
        assert sd.title == "Free form domain"
        assert sorted(sd.part_ids) == [1, 2]


class TestSubdomainNegativeIds:
    """Negative part IDs in /SUBDOMAIN (exclusion markers)."""

    def test_negative_ids_separated(self, tmp_path):
        """Negative IDs go to neg_part_ids, positive to part_ids."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_SUBDOMAIN_NEG
      2021         0
{_ELEM3}
/SUBDOMAIN/99
With exclusion
         1         2        -3
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        sd = model.subdomains[99]
        assert sorted(sd.part_ids) == [1, 2]
        assert sd.neg_part_ids == [3]


class TestSubdomainWarnings:
    """Warning on part ID not found in model."""

    def test_missing_part_warning(self, tmp_path):
        """Part ID 999 doesn't exist — should produce a warning."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_SUBDOMAIN_WARN
      2021         0
{_ELEM}
/SUBDOMAIN/77
Bad ref
         1       999
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        sd = model.subdomains[77]
        # Part 1 should be in the list; 999 too (stored as-is, warning logged)
        assert 1 in sd.part_ids
        assert 999 in sd.part_ids
        # Verify a warning was logged about part 999
        warn_msgs = [w for w in log.warnings if "999" in str(w)]
        assert len(warn_msgs) > 0, f"Expected warning about part 999, got: {log.warnings}"

"""M90 – /FAIL/CONNECT connector failure model (Starter parsing)."""
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
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
#  /FAIL/CONNECT tests
# ══════════════════════════════════════════════════════════════════════

class TestFailConnectFixed:
    """/FAIL/CONNECT in fixed-format decks."""

    def test_basic_connect_fixed(self, tmp_path):
        """Parse /FAIL/CONNECT with all 4 cards in fixed format."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_FAIL_CONNECT
      2021         0
{_ELEM}
/FAIL/CONNECT/1
#       EPSILON_MAXN          EXPONENT_N             ALPHA_N R_FCT_IDN     IFAIL  IFAIL_SO      ISYM
                 1.5                 2.0                 1.0         0         1         1         1
#       EPSILON_MAXT          EXPONENT_T             ALPHA_T  R_FCT_ID_T
                 2.5                 1.5                 1.0         0
#             EI_MAX              EN_MAX              ET_MAX                 N_N                 N_T
              1000.0               500.0               500.0                 1.0                 1.0
#              T_MAX              N_SOFT
               0.001                 2.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        # raw_fails should have one CONNECT entry for mat_id=1
        connect_fails = [(mid, fm) for mid, fm, _ in model.raw_fails
                         if fm.type == "CONNECT"]
        assert len(connect_fails) == 1
        mid, fm = connect_fails[0]
        assert mid == 1
        assert fm.type == "CONNECT"
        assert abs(fm.params["epsilon_maxN"] - 1.5) < 1e-12
        assert abs(fm.params["exponent_N"] - 2.0) < 1e-12
        assert abs(fm.params["alpha_N"] - 1.0) < 1e-12
        assert fm.params["ifail"] == 1
        assert fm.params["ifail_so"] == 1
        assert fm.params["isym"] == 1
        assert abs(fm.params["epsilon_maxT"] - 2.5) < 1e-12
        assert abs(fm.params["exponent_T"] - 1.5) < 1e-12
        assert abs(fm.params["EI_max"] - 1000.0) < 1e-12
        assert abs(fm.params["EN_max"] - 500.0) < 1e-12
        assert abs(fm.params["ET_max"] - 500.0) < 1e-12
        assert abs(fm.params["N_n"] - 1.0) < 1e-12
        assert abs(fm.params["N_t"] - 1.0) < 1e-12
        assert abs(fm.params["T_max"] - 0.001) < 1e-12
        assert abs(fm.params["N_soft"] - 2.0) < 1e-12


class TestFailConnectFree:
    """/FAIL/CONNECT in free-format decks."""

    def test_free_format(self, tmp_path):
        """Free-format /FAIL/CONNECT parsing."""
        deck = f"""\
/BEGIN
TEST_FAIL_CONNECT_FREE
{_ELEM}
/FAIL/CONNECT/1
1.5 2.0 1.0 0 1 1 1
2.5 1.5 1.0 0
1000.0 500.0 500.0 1.0 1.0
0.001 2.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        connect_fails = [(mid, fm) for mid, fm, _ in model.raw_fails
                         if fm.type == "CONNECT"]
        assert len(connect_fails) == 1
        mid, fm = connect_fails[0]
        assert mid == 1
        assert abs(fm.params["epsilon_maxN"] - 1.5) < 1e-12
        assert abs(fm.params["epsilon_maxT"] - 2.5) < 1e-12
        assert abs(fm.params["T_max"] - 0.001) < 1e-12


class TestFailConnectDefaults:
    """Default parameter handling for /FAIL/CONNECT."""

    def test_defaults_applied(self, tmp_path):
        """Missing optional fields get correct defaults."""
        deck = f"""\
/BEGIN
TEST_FAIL_CONNECT_DFLT
{_ELEM}
/FAIL/CONNECT/1
1.0 0 0 0 0 0 0
2.0 0 0 0
0 0 0 0 0
0 0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        connect_fails = [(mid, fm) for mid, fm, _ in model.raw_fails
                         if fm.type == "CONNECT"]
        assert len(connect_fails) == 1
        _, fm = connect_fails[0]
        assert abs(fm.params["epsilon_maxN"] - 1.0) < 1e-12
        assert abs(fm.params["epsilon_maxT"] - 2.0) < 1e-12
        # Exponents default to 1.0 when 0
        assert abs(fm.params["exponent_N"] - 1.0) < 1e-12
        assert abs(fm.params["exponent_T"] - 1.0) < 1e-12
        # Alpha defaults to 1.0 when 0
        assert abs(fm.params["alpha_N"] - 1.0) < 1e-12
        assert abs(fm.params["alpha_T"] - 1.0) < 1e-12


class TestFailConnectMinimalCards:
    """Minimal /FAIL/CONNECT with fewer cards (older formats)."""

    def test_two_cards_only(self, tmp_path):
        """Only 2 cards (normal + tangential), no energy/softening."""
        deck = f"""\
/BEGIN
TEST_FAIL_CONNECT_MIN
{_ELEM}
/FAIL/CONNECT/1
1.5 2.0 1.0 0 0 0 0
2.5 1.5 1.0 0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        connect_fails = [(mid, fm) for mid, fm, _ in model.raw_fails
                         if fm.type == "CONNECT"]
        assert len(connect_fails) == 1
        _, fm = connect_fails[0]
        assert abs(fm.params["epsilon_maxN"] - 1.5) < 1e-12
        assert abs(fm.params["epsilon_maxT"] - 2.5) < 1e-12
        # Energy and softening params should have defaults
        assert fm.params["EI_max"] == 0.0
        assert fm.params["T_max"] == 0.0
        assert fm.params["N_soft"] == 0.0

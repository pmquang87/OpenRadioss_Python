"""M94 – /CONVEC (thermal convection) and /INIVOL (initial volume fraction)."""
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.model.entities import ConvectionLoad, InitialVolume, InivolContainer
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
/SURF/SEG/1
OuterSurf
         1       101       102       103       104
/FUNCT/1
AmbientTempCurve
                 0.0               293.0
                 1.0               350.0
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="utf-8")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


# ══════════════════════════════════════════════════════════════════════
#  /CONVEC tests
# ══════════════════════════════════════════════════════════════════════

class TestConvecFixed:
    """/CONVEC in fixed-format decks."""

    def test_basic_convec_fixed(self, tmp_path):
        """Parse fixed-format /CONVEC."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_CONVEC_FIXED
      2021         0
{_ELEM}
/CONVEC/1
Convection_Load_1
#  SURF_ID  FUNCT_ID SENSOR_ID
         1         1         0
#             ASCALE              FSCALE              TSTART               TSTOP                   H
                 1.0                 1.0                 0.0                 0.5                25.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.convec_loads) == 1
        cl = model.convec_loads[0]
        assert isinstance(cl, ConvectionLoad)
        assert cl.id == 1
        assert cl.surf_id == 1
        assert cl.funct_id == 1
        assert cl.sens_id == 0
        assert abs(cl.xscale - 1.0) < 1e-12
        assert abs(cl.scale - 1.0) < 1e-12
        assert abs(cl.tstart - 0.0) < 1e-12
        assert abs(cl.tstop - 0.5) < 1e-12
        assert abs(cl.h - 25.0) < 1e-12
        assert cl.title == "Convection_Load_1"

    def test_convec_defaults(self, tmp_path):
        """Defaults for optional fields in /CONVEC."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_CONVEC_DFLT
      2021         0
{_ELEM}
/CONVEC/2
DefaultConvec
         1         1         0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.convec_loads) == 1
        cl = model.convec_loads[0]
        assert abs(cl.xscale - 1.0) < 1e-12
        assert abs(cl.scale - 1.0) < 1e-12
        assert abs(cl.tstop - 1.0e30) < 1e-12
        assert abs(cl.h - 0.0) < 1e-12


class TestConvecFree:
    """/CONVEC in free-format decks."""

    def test_basic_convec_free(self, tmp_path):
        """Parse free-format /CONVEC."""
        deck = f"""\
/BEGIN
TEST_CONVEC_FREE
{_ELEM}
/CONVEC/3
FreeConvec
1 1 0
1.0 1.0 0.1 0.8 50.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.convec_loads) == 1
        cl = model.convec_loads[0]
        assert cl.id == 3
        assert abs(cl.tstart - 0.1) < 1e-12
        assert abs(cl.tstop - 0.8) < 1e-12
        assert abs(cl.h - 50.0) < 1e-12


# ══════════════════════════════════════════════════════════════════════
#  /INIVOL tests
# ══════════════════════════════════════════════════════════════════════

class TestInivolFixed:
    """/INIVOL in fixed-format decks."""

    def test_basic_inivol_fixed(self, tmp_path):
        """Parse fixed-format /INIVOL with container card."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_INIVOL_FIXED
      2021         0
{_ELEM}
/INIVOL/1/10
Water_Fill
#  surf_ID ALE_PHASE  FILL_OPT     ICUMU          FILL_RATIO
         1         1         0         0                 0.8
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.inivol) == 1
        iv = model.inivol[0]
        assert isinstance(iv, InitialVolume)
        assert iv.id == 10
        assert iv.part_id == 1
        assert iv.title == "Water_Fill"
        assert len(iv.containers) == 1
        c = iv.containers[0]
        assert isinstance(c, InivolContainer)
        assert c.surf_id == 1
        assert c.ale_phase == 1
        assert c.fill_opt == 0
        assert c.icumu == 0
        assert abs(c.fill_ratio - 0.8) < 1e-12

    def test_multiple_containers_fixed(self, tmp_path):
        """Parse /INIVOL with multiple container cards."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_INIVOL_MULTI
      2021         0
{_ELEM}
/INIVOL/1/20
MultiContainer
         1         1         0         0                 0.5
         1         2         1         1                 1.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.inivol) == 1
        iv = model.inivol[0]
        assert len(iv.containers) == 2
        assert iv.containers[0].ale_phase == 1
        assert abs(iv.containers[0].fill_ratio - 0.5) < 1e-12
        assert iv.containers[1].ale_phase == 2
        assert iv.containers[1].fill_opt == 1
        assert iv.containers[1].icumu == 1
        assert abs(iv.containers[1].fill_ratio - 1.0) < 1e-12


class TestInivolFree:
    """/INIVOL in free-format decks."""

    def test_basic_inivol_free(self, tmp_path):
        """Parse free-format /INIVOL."""
        deck = f"""\
/BEGIN
TEST_INIVOL_FREE
{_ELEM}
/INIVOL/1/30
FreeInivol
1 1 0 0 0.95
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.inivol) == 1
        iv = model.inivol[0]
        assert iv.id == 30
        assert iv.part_id == 1
        assert len(iv.containers) == 1
        assert abs(iv.containers[0].fill_ratio - 0.95) < 1e-12

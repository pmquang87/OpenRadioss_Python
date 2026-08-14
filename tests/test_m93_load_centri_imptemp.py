"""M93 – /LOAD/CENTRI (centrifugal load) and /IMPTEMP (imposed temperature)."""
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.model.entities import CentrifugalLoad, ImposedTemperature
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
/GRNOD/NODE/1
AllNodes
       101       102       103       104
/FUNCT/1
CurveOne
                 0.0                 0.0
                 1.0               100.0
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="utf-8")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


# ══════════════════════════════════════════════════════════════════════
#  /LOAD/CENTRI tests
# ══════════════════════════════════════════════════════════════════════

class TestLoadCentriFixed:
    """/LOAD/CENTRI in fixed-format decks."""

    def test_basic_load_centri_fixed(self, tmp_path):
        """Parse fixed-format /LOAD/CENTRI."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_CENTRI_FIXED
      2021         0
{_ELEM}
/LOAD/CENTRI/1
Centrifugal_Load_1
#funct_IDT       Dir  frame_ID sensor_ID  grnod_ID      Ivar             Ascalex             Fscaley
         1        ZZ         0         0         1         2                 1.5                50.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.centri_loads) == 1
        cl = model.centri_loads[0]
        assert isinstance(cl, CentrifugalLoad)
        assert cl.id == 1
        assert cl.funct_id == 1
        assert cl.dir == "ZZ"
        assert cl.frame_id == 0
        assert cl.sens_id == 0
        assert cl.grnod_id == 1
        assert cl.ivar == 2
        assert abs(cl.scale_x - 1.5) < 1e-12
        assert abs(cl.scale_y - 50.0) < 1e-12
        assert cl.title == "Centrifugal_Load_1"

    def test_load_centri_defaults(self, tmp_path):
        """Defaults for optional fields in /LOAD/CENTRI."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_CENTRI_DFLT
      2021         0
{_ELEM}
/LOAD/CENTRI/2
DefaultCentri
         1                   0         0         1
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.centri_loads) == 1
        cl = model.centri_loads[0]
        assert cl.dir == "XX"  # default
        assert cl.ivar == 1   # default
        assert abs(cl.scale_x - 1.0) < 1e-12
        assert abs(cl.scale_y - 1.0) < 1e-12


class TestLoadCentriFree:
    """/LOAD/CENTRI in free-format decks."""

    def test_basic_load_centri_free(self, tmp_path):
        """Parse free-format /LOAD/CENTRI."""
        deck = f"""\
/BEGIN
TEST_CENTRI_FREE
{_ELEM}
/LOAD/CENTRI/3
FreeCentri
1 YY 0 0 1 1 2.0 75.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.centri_loads) == 1
        cl = model.centri_loads[0]
        assert cl.id == 3
        assert cl.dir == "YY"
        assert abs(cl.scale_x - 2.0) < 1e-12
        assert abs(cl.scale_y - 75.0) < 1e-12


# ══════════════════════════════════════════════════════════════════════
#  /IMPTEMP tests
# ══════════════════════════════════════════════════════════════════════

class TestImpTempFixed:
    """/IMPTEMP in fixed-format decks."""

    def test_basic_imptemp_fixed(self, tmp_path):
        """Parse fixed-format /IMPTEMP with control and window cards."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_IMPTEMP_FIXED
      2021         0
{_ELEM}
/IMPTEMP/1
Imposed_Temp_1
# func_IDT sensor_ID  grnod_ID
         1         0         1
#           Ascale_x            Fscale_y             T_start              T_stop
                 1.0               300.0                 0.0                 0.5
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.imptemp) == 1
        it = model.imptemp[0]
        assert isinstance(it, ImposedTemperature)
        assert it.id == 1
        assert it.funct_id == 1
        assert it.sens_id == 0
        assert it.grnod_id == 1
        assert abs(it.xscale - 1.0) < 1e-12
        assert abs(it.scale - 300.0) < 1e-12
        assert abs(it.tstart - 0.0) < 1e-12
        assert abs(it.tstop - 0.5) < 1e-12
        assert it.title == "Imposed_Temp_1"


class TestImpTempFree:
    """/IMPTEMP in free-format decks."""

    def test_basic_imptemp_free(self, tmp_path):
        """Parse free-format /IMPTEMP."""
        deck = f"""\
/BEGIN
TEST_IMPTEMP_FREE
{_ELEM}
/IMPTEMP/2
FreeTemp
1 0 1
1.0 450.0 0.1 1.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.imptemp) == 1
        it = model.imptemp[0]
        assert it.id == 2
        assert it.grnod_id == 1
        assert abs(it.scale - 450.0) < 1e-12
        assert abs(it.tstart - 0.1) < 1e-12
        assert abs(it.tstop - 1.0) < 1e-12

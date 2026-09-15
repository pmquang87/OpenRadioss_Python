"""M95 – Thermal Loads and Initial Conditions: /RADIATION, /IMPFLUX, /INITEMP."""
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.model.entities import RadiationLoad, ImposedFlux, InitialTemperature
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
/GRNOD/NODE/1
AllNodes
       101       102       103       104
/FUNCT/1
ThermalCurve
                 0.0               293.0
                 1.0               400.0
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="utf-8")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


# ══════════════════════════════════════════════════════════════════════
#  /RADIATION tests
# ══════════════════════════════════════════════════════════════════════

class TestRadiation:
    """/RADIATION in fixed and free format."""

    def test_basic_radiation_fixed(self, tmp_path):
        """Parse fixed-format /RADIATION."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_RAD_FIXED
      2021         0
{_ELEM}
/RADIATION/1
Radiation_Load_1
#  SURF_ID  FUNCT_ID SENSOR_ID
         1         1         0
#             ASCALE              FSCALE              TSTART               TSTOP                   E
                 1.0                 1.0                 0.0                 1.0                 0.8
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.radiation_loads) == 1
        rl = model.radiation_loads[0]
        assert isinstance(rl, RadiationLoad)
        assert rl.id == 1
        assert rl.surf_id == 1
        assert rl.funct_id == 1
        assert abs(rl.emissivity - 0.8) < 1e-12
        assert abs(rl.tstop - 1.0) < 1e-12
        assert rl.title == "Radiation_Load_1"

    def test_radiation_free(self, tmp_path):
        """Parse free-format /RADIATION."""
        deck = f"""\
/BEGIN
TEST_RAD_FREE
{_ELEM}
/RADIATION/2
FreeRad
1 1 0
1.0 1.0 0.0 0.5 0.95
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.radiation_loads) == 1
        rl = model.radiation_loads[0]
        assert rl.id == 2
        assert abs(rl.emissivity - 0.95) < 1e-12


# ══════════════════════════════════════════════════════════════════════
#  /IMPFLUX tests
# ══════════════════════════════════════════════════════════════════════

class TestImpflux:
    """/IMPFLUX in fixed and free format."""

    def test_basic_impflux_fixed(self, tmp_path):
        """Parse fixed-format surfacic /IMPFLUX."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_FLUX_FIXED
      2021         0
{_ELEM}
/IMPFLUX/1
Heat_Flux_1
#  SURF_ID  FUNCT_ID SENSOR_ID GRBRIC_ID
         1         1         0         0
#             ASCALE              FSCALE              TSTART               TSTOP
                 1.0               500.0                 0.0                 2.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.impflux_loads) == 1
        fl = model.impflux_loads[0]
        assert isinstance(fl, ImposedFlux)
        assert fl.id == 1
        assert fl.surf_id == 1
        assert fl.funct_id == 1
        assert fl.grbric_id == 0
        assert abs(fl.scale - 500.0) < 1e-12
        assert abs(fl.tstop - 2.0) < 1e-12

    def test_impflux_free(self, tmp_path):
        """Parse free-format /IMPFLUX."""
        deck = f"""\
/BEGIN
TEST_FLUX_FREE
{_ELEM}
/IMPFLUX/2
FreeFlux
1 1 0 0
1.0 1200.0 0.1 0.9
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.impflux_loads) == 1
        fl = model.impflux_loads[0]
        assert fl.id == 2
        assert abs(fl.scale - 1200.0) < 1e-12
        assert abs(fl.tstart - 0.1) < 1e-12


# ══════════════════════════════════════════════════════════════════════
#  /INITEMP tests
# ══════════════════════════════════════════════════════════════════════

class TestInitemp:
    """/INITEMP uniform and nodal table."""

    def test_uniform_initemp_fixed(self, tmp_path):
        """Parse fixed-format uniform group /INITEMP."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_INITEMP_UNIFORM
      2021         0
{_ELEM}
/INITEMP/1
Uniform_Temp
#                 T0   grnd_ID  fld_type
              298.15         1         0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.initemp) == 1
        it = model.initemp[0]
        assert isinstance(it, InitialTemperature)
        assert it.id == 1
        assert abs(it.t0 - 298.15) < 1e-12
        assert it.grnod_id == 1
        assert it.fld_type == 0

    def test_nodal_table_initemp(self, tmp_path):
        """Parse /INITEMP with individual nodal temperature overrides."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_INITEMP_NODAL
      2021         0
{_ELEM}
/INITEMP/2
Nodal_Temp_Map
#                 T0   grnd_ID  fld_type
               300.0         1         1
#                T0i  node_IDi
               310.0       101
               320.0       102
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert len(model.initemp) == 1
        it = model.initemp[0]
        assert it.id == 2
        assert it.fld_type == 1
        assert abs(it.t0 - 300.0) < 1e-12
        assert len(it.nodal_temps) == 2
        assert abs(it.nodal_temps[101] - 310.0) < 1e-12
        assert abs(it.nodal_temps[102] - 320.0) < 1e-12

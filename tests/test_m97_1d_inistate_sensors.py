"""
Tests for Milestone M97: 1D Element Initial State Suite (/INITRU, /INIBEA, /INISPR)
and Extended Sensor Suite (/SENSOR/DIST, /SENSOR/ENERGY, /SENSOR/INTER, /SENSOR/RBODY, /SENSOR/TEMP).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.model.entities import (
    InitialTrussState, InitialBeamState, InitialSpringState, Sensor,
)
from pyradioss.starter.starter import run_starter


_BOILERPLATE = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/LAW1/1
Steel
              7.8e-9
            210000.0                 0.3
/PROP/TYPE2/1
Truss_Prop
                 1.5
/PROP/TYPE3/2
Beam_Prop
         1
         0         0
                 1.0                 1.0                 1.0                 1.0
/PROP/TYPE4/3
Spring_Prop
                 1.0                 1.0                 0.0
/PART/1
Part_Truss
         1         1
/PART/2
Part_Beam
         2         1
/PART/3
Part_Spring
         3         1
/NODE
         1                 0.0                 0.0                 0.0
         2                10.0                 0.0                 0.0
         3                20.0                 0.0                 0.0
         4                30.0                 0.0                 0.0
         5                20.0                10.0                 0.0
/TRUSS/1
         1         1         2
/BEAM/2
         2         2         3         5
/SPRING/3
         3         3         4
/GRNOD/NODE/1
Group_1
         1         2
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


# ══════════════════════════════════════════════════════════════════════
#  /INITRU tests (Truss initial state)
# ══════════════════════════════════════════════════════════════════════

class TestInitru:
    """/INITRU initial state for truss elements."""

    def test_initru_scalar_fixed(self, tmp_path):
        """Parse fixed-format /INITRU/EPSP and /INITRU/FORCE."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_INITRU_SCALARS
      2021         0
{_BOILERPLATE}
/INITRU/EPSP
         1                0.05
/INITRU/FORCE
         1               150.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 1 in model.ini_trusses
        st = model.ini_trusses[1]
        assert isinstance(st, InitialTrussState)
        assert st.elem_id == 1
        assert abs(st.epsp - 0.05) < 1e-12
        assert abs(st.force - 150.0) < 1e-12

    def test_initru_full_free(self, tmp_path):
        """Parse free-format /INITRU/FULL."""
        deck = f"""\
/BEGIN
TEST_INITRU_FULL_FREE
{_BOILERPLATE}
/INITRU/FULL
1 2 25.0 300.0 1.75 0.08
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 1 in model.ini_trusses
        st = model.ini_trusses[1]
        assert st.prop_type == 2
        assert abs(st.eint - 25.0) < 1e-12
        assert abs(st.force - 300.0) < 1e-12
        assert abs(st.area - 1.75) < 1e-12
        assert abs(st.epsp - 0.08) < 1e-12


# ══════════════════════════════════════════════════════════════════════
#  /INIBEA tests (Beam initial state)
# ══════════════════════════════════════════════════════════════════════

class TestInibea:
    """/INIBEA initial state for beam elements."""

    def test_inibea_scalar_fixed(self, tmp_path):
        """Parse fixed-format /INIBEA/FORCE, MOMENT, EPSP."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_INIBEA_SCALARS
      2021         0
{_BOILERPLATE}
/INIBEA/FORCE
         2               250.0
/INIBEA/MOMENT
         2                45.0
/INIBEA/EPSP
         2                0.03
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 2 in model.ini_beams
        st = model.ini_beams[2]
        assert isinstance(st, InitialBeamState)
        assert st.elem_id == 2
        assert abs(st.force[0] - 250.0) < 1e-12
        assert abs(st.moment[0] - 45.0) < 1e-12
        assert abs(st.epsp - 0.03) < 1e-12

    def test_inibea_full_fixed(self, tmp_path):
        """Parse fixed-format /INIBEA/FULL."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_INIBEA_FULL_FIXED
      2021         0
{_BOILERPLATE}
/INIBEA/FULL
#  beam_ID nb_integr prop_type
         2         0         3
#               EImemb              EIbend                  F1                  F2                  F3
                12.5                 6.0               100.0                10.0                 5.0
#                   M1                  M2                  M3
                20.0                 8.0                 4.0
#             EpsilonP
                0.02
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 2 in model.ini_beams
        st = model.ini_beams[2]
        assert st.prop_type == 3
        assert abs(st.eint_memb - 12.5) < 1e-12
        assert abs(st.eint_bend - 6.0) < 1e-12
        assert np.allclose(st.force, [100.0, 10.0, 5.0])
        assert np.allclose(st.moment, [20.0, 8.0, 4.0])
        assert abs(st.epsp - 0.02) < 1e-12


# ══════════════════════════════════════════════════════════════════════
#  /INISPR tests (Spring initial state)
# ══════════════════════════════════════════════════════════════════════

class TestInispr:
    """/INISPR initial state for spring elements."""

    def test_inispr_scalar_free(self, tmp_path):
        """Parse free-format /INISPR/DISP and /INISPR/FORCE."""
        deck = f"""\
/BEGIN
TEST_INISPR_SCALARS_FREE
{_BOILERPLATE}
/INISPR/DISP
3 0.25
/INISPR/FORCE
3 80.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 3 in model.ini_springs
        st = model.ini_springs[3]
        assert isinstance(st, InitialSpringState)
        assert st.elem_id == 3
        assert abs(st.disp - 0.25) < 1e-12
        assert abs(st.force - 80.0) < 1e-12

    def test_inispr_full_fixed(self, tmp_path):
        """Parse fixed-format /INISPR/FULL."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_INISPR_FULL_FIXED
      2021         0
{_BOILERPLATE}
/INISPR/FULL
#spring_ID prop_type     nvars
         3         4         0
#                  F_X                 D_X               FEP_X              DPL_XP              DPL_XM
                90.0                0.45               120.0                0.05                 0.0
#                  L_X                  EI
                10.0                 4.5
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        assert 3 in model.ini_springs
        st = model.ini_springs[3]
        assert st.prop_type == 4
        assert abs(st.force - 90.0) < 1e-12
        assert abs(st.disp - 0.45) < 1e-12
        assert abs(st.fep - 120.0) < 1e-12
        assert abs(st.dpl_pos - 0.05) < 1e-12
        assert abs(st.length - 10.0) < 1e-12
        assert abs(st.eint - 4.5) < 1e-12


# ══════════════════════════════════════════════════════════════════════
#  Extended /SENSOR tests
# ══════════════════════════════════════════════════════════════════════

class TestSensorsExtended:
    """Extended /SENSOR types (DIST, ENERGY, INTER, RBODY, TEMP)."""

    def test_sensor_dist(self, tmp_path):
        """Parse /SENSOR/DIST."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_SENSOR_DIST
      2021         0
{_BOILERPLATE}
/SENSOR/DIST/10
Relative_Distance_Sensor
                 0.005
# node_ID1  node_ID2                Dmin                Dmax                Tmin
         1         2                 2.0                15.0               0.001
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        s = next(s for s in model.sensors if s.id == 10)
        assert s.kind == "DIST"
        assert abs(s.tdelay - 0.005) < 1e-12
        assert s.node_id1 == 1
        assert s.node_id2 == 2
        assert abs(s.dmin - 2.0) < 1e-12
        assert abs(s.dmax - 15.0) < 1e-12
        assert abs(s.tmin - 0.001) < 1e-12

    def test_sensor_energy(self, tmp_path):
        """Parse /SENSOR/ENERGY."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_SENSOR_ENERGY
      2021         0
{_BOILERPLATE}
/SENSOR/ENERGY/20
Energy_Threshold_Sensor
                   0.0
#  part_ID subset_ID   Iselect
         1         0         1
#              IEmin               IEmax               KEmin               KEmax                Tmin
                 0.0               500.0                 0.0               250.0               0.002
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        s = next(s for s in model.sensors if s.id == 20)
        assert s.kind == "ENERGY"
        assert s.part_id == 1
        assert s.iselect == 1
        assert abs(s.iemax - 500.0) < 1e-12
        assert abs(s.kemax - 250.0) < 1e-12
        assert abs(s.tmin - 0.002) < 1e-12

    def test_sensor_temp_and_inter_and_rbody(self, tmp_path):
        """Parse /SENSOR/TEMP, /SENSOR/INTER, /SENSOR/RBODY."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_SENSOR_MORE
      2021         0
{_BOILERPLATE}
/RBODY/100
Rigid_Body
         1         0         0         0                 0.0         1         0         1         0
/SENSOR/TEMP/30
Temperature_Sensor
                   0.0
# Grnod_Id             Tempmax             Tempmin            Tempmean                Tmin
         1               450.0               293.0               350.0               0.001
/SENSOR/RBODY/40
Rbody_Force_Sensor
                 0.002
# rbody_ID       DIR                Fmin                Fmax                Tmin
       100        TF                 0.0              1000.0               0.001
/SENSOR/INTER/50
Contact_Force_Sensor
                   0.0
#   int_ID       DIR                Fmin                Fmax                Tmin                Fcut
         1        FN                 0.0               500.0               0.001               100.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        s_temp = next(s for s in model.sensors if s.id == 30)
        assert s_temp.kind == "TEMP"
        assert s_temp.grnod_id == 1
        assert abs(s_temp.tempmax - 450.0) < 1e-12
        assert abs(s_temp.tempmin - 293.0) < 1e-12
        assert abs(s_temp.tempmean - 350.0) < 1e-12

        s_rb = next(s for s in model.sensors if s.id == 40)
        assert s_rb.kind == "RBODY"
        assert s_rb.rbody_id == 100
        assert s_rb.dir == "TF"
        assert abs(s_rb.fmax - 1000.0) < 1e-12

        s_int = next(s for s in model.sensors if s.id == 50)
        assert s_int.kind == "INTER"
        assert s_int.int_id == 1
        assert s_int.dir == "FN"
        assert abs(s_int.fmax - 500.0) < 1e-12
        assert abs(s_int.fcut - 100.0) < 1e-12

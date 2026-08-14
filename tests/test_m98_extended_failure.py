"""
Tests for Milestone M98: Extended Failure Models Suite
(/FAIL/TENSSTRAIN, /FAIL/ORTHSTRAIN, /FAIL/GURSON, /FAIL/ALTER, /FAIL/VISUAL, /FAIL/MULLINS_OR).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.model.entities import FailureModel
from pyradioss.starter.starter import run_starter


_BOILERPLATE = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/LAW2/1
Steel
              7.8e-9
            210000.0                 0.3
               500.0              1000.0                 0.5                 0.0
                 0.5
/PROP/TYPE1/1
Shell_Prop
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/PART/1
Part_1
         1         1
/NODE
         1                 0.0                 0.0                 0.0
         2                10.0                 0.0                 0.0
         3                10.0                10.0                 0.0
         4                 0.0                10.0                 0.0
/SHELL/1
         1         1         2         3         4
/FUNCT/10
Strain_Rate_Scaling
                 0.0                 1.0
              1000.0                 1.5
/FUNCT/20
Element_Size_Scaling
                 0.0                 1.0
                50.0                 0.8
/FUNCT/30
Temperature_Scaling
                 0.0                 1.0
               500.0                 0.5
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


# ══════════════════════════════════════════════════════════════════════
#  /FAIL/TENSSTRAIN tests
# ══════════════════════════════════════════════════════════════════════

class TestFailTensstrain:
    """/FAIL/TENSSTRAIN tensile strain failure model."""

    def test_tensstrain_basic_fixed(self, tmp_path):
        """Parse fixed-format /FAIL/TENSSTRAIN with S_Flag=1."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_TENSSTRAIN_BASIC
      2021         0
{_BOILERPLATE}
/FAIL/TENSSTRAIN/1
#         EPSILON_T1          EPSILON_T2    FCT_ID          EPSILON_F1          EPSILON_F2     S_Flag
                0.05                0.15        10                0.20                0.25         1
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        mat = model.materials[1]
        assert mat.fail is not None
        fm = mat.fail
        assert isinstance(fm, FailureModel)
        assert fm.type == "TENSSTRAIN"
        assert abs(fm.params["eps_t1"] - 0.05) < 1e-12
        assert abs(fm.params["eps_t2"] - 0.15) < 1e-12
        assert fm.params["fct_id"] == 10
        assert "function" in fm.params
        assert abs(fm.params["eps_f1"] - 0.20) < 1e-12
        assert abs(fm.params["eps_f2"] - 0.25) < 1e-12
        assert fm.params["s_flag"] == 1

    def test_tensstrain_with_scaling_free(self, tmp_path):
        """Parse free-format /FAIL/TENSSTRAIN with S_Flag=2 and scale functions."""
        deck = f"""\
/BEGIN
TEST_TENSSTRAIN_SCALING
{_BOILERPLATE}
/FAIL/TENSSTRAIN/1
0.04 0.12 10 0.18 0.22 2
20 1.2 5.0
30 0.9
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        mat = model.materials[1]
        fm = mat.fail
        assert fm.type == "TENSSTRAIN"
        assert fm.params["s_flag"] == 2
        assert fm.params["fct_idel"] == 20
        assert "function_el" in fm.params
        assert abs(fm.params["fscale_el"] - 1.2) < 1e-12
        assert abs(fm.params["ei_ref"] - 5.0) < 1e-12
        assert fm.params["fct_id_t"] == 30
        assert "function_t" in fm.params
        assert abs(fm.params["fscale_t"] - 0.9) < 1e-12


# ══════════════════════════════════════════════════════════════════════
#  /FAIL/ORTHSTRAIN tests
# ══════════════════════════════════════════════════════════════════════

class TestFailOrthstrain:
    """/FAIL/ORTHSTRAIN orthotropic strain failure model."""

    def test_orthstrain_fixed(self, tmp_path):
        """Parse fixed-format /FAIL/ORTHSTRAIN with directional cards."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_ORTHSTRAIN_FIXED
      2021         0
{_BOILERPLATE}
/FAIL/ORTHSTRAIN/1
#                                   PTHK
                                     0.5
#    EPSILON_DOT_REF                FCUT
                 1.0               100.0
# FCT_IDEL           FSCALE_EL              EI_REF    STRDEF
        20                 1.0                 5.0         1
#       EPSILON_11TF        EPSILON_11TM FCT_ID11T        EPSILON_11CF        EPSILON_11CM FCT_ID_11C
                0.02                0.05        10                0.03                0.06         0
#       EPSILON_22TF        EPSILON_22TM FCT_ID22T        EPSILON_22CF        EPSILON_22CM FCT_ID_22C
                0.01                0.03         0                0.02                0.04         0
#       EPSILON_33TF        EPSILON_33TM FCT_ID33T        EPSILON_33CF        EPSILON_33CM FCT_ID_33C
                0.01                0.03         0                0.02                0.04         0
#       EPSILON_12TF        EPSILON_12TM FCT_ID12T        EPSILON_12CF        EPSILON_12CM FCT_ID_12C
                0.04                0.08         0                0.04                0.08         0
#       EPSILON_23TF        EPSILON_23TM FCT_ID23T        EPSILON_23CF        EPSILON_23CM FCT_ID_23C
                0.04                0.08         0                0.04                0.08         0
#       EPSILON_31TF        EPSILON_31TM FCT_ID31T        EPSILON_31CF        EPSILON_31CM FCT_ID_31C
                0.04                0.08         0                0.04                0.08         0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        mat = model.materials[1]
        fm = mat.fail
        assert fm.type == "ORTHSTRAIN"
        assert abs(fm.params["pthk"] - 0.5) < 1e-12
        assert abs(fm.params["eps_dot_ref"] - 1.0) < 1e-12
        assert abs(fm.params["fcut"] - 100.0) < 1e-12
        assert fm.params["fct_idel"] == 20
        assert "function_el" in fm.params
        assert fm.params["strdef"] == 1
        assert abs(fm.params["eps_11_tf"] - 0.02) < 1e-12
        assert abs(fm.params["eps_11_tm"] - 0.05) < 1e-12
        assert fm.params["fct_id_11_t"] == 10
        assert "function_11_t" in fm.params
        assert abs(fm.params["eps_11_cf"] - 0.03) < 1e-12


# ══════════════════════════════════════════════════════════════════════
#  /FAIL/GURSON tests
# ══════════════════════════════════════════════════════════════════════

class TestFailGurson:
    """/FAIL/GURSON porous metal plasticity failure model."""

    def test_gurson_fixed(self, tmp_path):
        """Parse fixed-format /FAIL/GURSON."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_GURSON_FIXED
      2021         0
{_BOILERPLATE}
/FAIL/GURSON/1
#                 q1                  q2                                                        Iloc
                 1.5                 1.0                                                           1
#                 En                  As                  Kw
                0.30                0.04                 3.0
#                 fc                  fr                  f0
                0.15                0.25               0.005
#               Rlen                Hchi
                 2.5               100.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        mat = model.materials[1]
        fm = mat.fail
        assert fm.type == "GURSON"
        assert abs(fm.params["q1"] - 1.5) < 1e-12
        assert abs(fm.params["q2"] - 1.0) < 1e-12
        assert fm.params["iloc"] == 1
        assert abs(fm.params["eps_n"] - 0.30) < 1e-12
        assert abs(fm.params["a_s"] - 0.04) < 1e-12
        assert abs(fm.params["k_w"] - 3.0) < 1e-12
        assert abs(fm.params["f_c"] - 0.15) < 1e-12
        assert abs(fm.params["f_r"] - 0.25) < 1e-12
        assert abs(fm.params["f_0"] - 0.005) < 1e-12
        assert abs(fm.params["r_len"] - 2.5) < 1e-12
        assert abs(fm.params["h_chi"] - 100.0) < 1e-12


# ══════════════════════════════════════════════════════════════════════
#  /FAIL/ALTER tests
# ══════════════════════════════════════════════════════════════════════

class TestFailAlter:
    """/FAIL/ALTER glass subcritical crack growth failure model."""

    def test_alter_fixed(self, tmp_path):
        """Parse fixed-format /FAIL/ALTER."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_ALTER_FIXED
      2021         0
{_BOILERPLATE}
/FAIL/ALTER/1
#              EXP_N                  V0                  VC   NCYCLES     IRATE     ISIDE      MODE
                16.0               0.005              1500.0        50         0         1         0
#            CR_FOIL              CR_AIR             CR_CORE             CR_EDGE    GRSH4N    GRSH3N
               0.002               0.001              0.0005               0.003         0         0
#                KIC                 KTH                RLEN                TDEL
               0.750               0.250                 1.0              0.0001
#              KRES1               KRES2
                 1.0                 1.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        mat = model.materials[1]
        fm = mat.fail
        assert fm.type == "ALTER"
        assert abs(fm.params["exp_n"] - 16.0) < 1e-12
        assert abs(fm.params["v0"] - 0.005) < 1e-12
        assert abs(fm.params["vc"] - 1500.0) < 1e-12
        assert fm.params["ema"] == 50
        assert fm.params["irate"] == 0
        assert fm.params["iside"] == 1
        assert abs(fm.params["cr_foil"] - 0.002) < 1e-12
        assert abs(fm.params["kic"] - 0.750) < 1e-12
        assert abs(fm.params["kth"] - 0.250) < 1e-12
        assert abs(fm.params["tdel"] - 0.0001) < 1e-12


# ══════════════════════════════════════════════════════════════════════
#  /FAIL/VISUAL and /FAIL/MULLINS_OR tests
# ══════════════════════════════════════════════════════════════════════

class TestFailVisualAndMullins:
    """/FAIL/VISUAL and /FAIL/MULLINS_OR."""

    def test_visual_and_mullins(self, tmp_path):
        """Parse /FAIL/VISUAL and /FAIL/MULLINS_OR."""
        deck = f"""\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_VISUAL_AND_MULLINS
      2021         0
{_BOILERPLATE}
/MAT/LAW2/2
Rubber
              1.2e-9
               100.0                 0.4
                10.0                20.0                 0.5                 0.0
                 0.5
/FAIL/VISUAL/1
#     TYPE               C_MIN               C_MAX       F-COEFFICIENT    F-FLAG          STRDEF
         1                 0.0               600.0                 0.8         1               2
/FAIL/MULLINS_OR/2
#                  R                BETA                   m
                 1.0                 0.5                12.0
/END
"""
        model, log = _run(tmp_path, deck)
        assert len(log.errors) == 0
        mat1 = model.materials[1]
        fm1 = mat1.fail
        assert fm1.type == "VISUAL"
        assert fm1.params["type"] == 1
        assert abs(fm1.params["c_max"] - 600.0) < 1e-12
        assert abs(fm1.params["f_coeff"] - 0.8) < 1e-12
        assert fm1.params["f_flag"] == 1
        assert fm1.params["strdef"] == 2

        mat2 = model.materials[2]
        fm2 = mat2.fail
        assert fm2.type == "MULLINS_OR"
        assert abs(fm2.params["coefr"] - 1.0) < 1e-12
        assert abs(fm2.params["beta"] - 0.5) < 1e-12
        assert abs(fm2.params["coefm"] - 12.0) < 1e-12

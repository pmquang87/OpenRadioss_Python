"""Tests for Milestone M188:
- /MAT/LAW12 (/MAT/3PARBI, /MAT/3D_COMP, /MAT/RAGAB)
- /MAT/LAW13 (/MAT/HONEYCOMB, /MAT/RIGID)
- /MAT/LAW15 (/MAT/CHANG, /MAT/CHANG_CHANG)
- /MAT/LAW18 (/MAT/CONCR_DRA, /MAT/DRAGON, /MAT/THERM)
- /MAT/LAW22 (/MAT/TSAI_WU, /MAT/TSAIWU, /MAT/DAMA)
- /MAT/LAW25 (/MAT/COMP_PLAS, /MAT/COMPOSITE_PLAS, /MAT/COMPSH)
- /MAT/LAW28 (/MAT/HONEYCOMB_SOL)
- /PROP/TYPE9 (/PROP/SH_ORTH, /PROP/SHELL_ORTH)
- /PROP/TYPE10 (/PROP/SH_COMP, /PROP/SHELL_COMP)
- /PROP/TYPE51 (/PROP/SH_COH, /PROP/SHELL_COH, /PROP/COHESIVE)
- /PROP/TYPE5 (/PROP/RIVET)
- /PROP/TYPE6 (/PROP/SOL_ORTH, /PROP/SOLID_ORTH)
- /PROP/TYPE20 (/PROP/TSHELL, /PROP/THICK_SHELL)
"""
from __future__ import annotations
from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, deck_text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck_text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


class TestMatLaw12:
    def test_law12_free(self, tmp_path: Path):
        deck = """/BEGIN
/MAT/LAW12/101
3D composite
7.85e-6 0.0
150000.0 10000.0 10000.0
0.3 0.02 0.02
5000.0 3000.0 5000.0
2000.0 50.0 50.0 0.1
100.0 0.5 500.0
1500.0 40.0 1200.0 100.0
60.0 60.0 40.0 40.0
40.0 100.0 60.0 60.0
0.0 0.0 0.0 0.0 1
/END
"""
        model, log = _parse_starter(tmp_path, deck)
        assert 101 in model.mat_law12s
        mat = model.mat_law12s[101]
        assert mat.rho0 == pytest.approx(7.85e-6)
        assert mat.e11 == pytest.approx(150000.0)
        assert mat.e22 == pytest.approx(10000.0)
        assert mat.nu12 == pytest.approx(0.3)
        assert mat.g12 == pytest.approx(5000.0)
        assert mat.sig_t1 == pytest.approx(2000.0)
        assert mat.sig_1yt == pytest.approx(1500.0)
        assert mat.icc == 1
        assert 101 in model.materials

    def test_law12_fixed_alias_3parbi(self, tmp_path: Path):
        deck = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/3PARBI/102
Three param Drucker Prager
#              RHO_I               RHO_O
            7.850e-6                 0.0
#                E11                 E22                 E33
            150000.0             10000.0             10000.0
#               NU12                NU23                NU31
                 0.3                0.02                0.02
#                G12                 G23                 G31
              5000.0              3000.0              5000.0
#             SIG_T1              SIG_T2              SIG_T3               DELTA
              2000.0                50.0                50.0                 0.1
#                  B                   N                FMAX
               100.0                 0.5               500.0
#            SIG_1YT             SIG_2YT             SIG_1YC             SIG_2YC
              1500.0                40.0              1200.0               100.0
#           SIG_12YT            SIG_12YC            SIG_23YT            SIG_23YC
                60.0                60.0                40.0                40.0
#            SIG_3YT             SIG_3YC            SIG_13YT            SIG_13YC
                40.0               100.0                60.0                60.0
#              ALPHA                EFIB                   C                EPS0         ICC
                 0.0                 0.0                 0.0                 0.0           1
/END
"""
        model, log = _parse_starter(tmp_path, deck)
        assert 102 in model.mat_law12s
        mat = model.mat_law12s[102]
        assert mat.rho0 == pytest.approx(7.85e-6)
        assert mat.e11 == pytest.approx(150000.0)
        assert mat.b == pytest.approx(100.0)
        assert mat.n == pytest.approx(0.5)
        assert mat.fmax == pytest.approx(500.0)


class TestMatLaw13:
    def test_law13_free_and_fixed(self, tmp_path: Path):
        deck_free = """/BEGIN
/MAT/LAW13/201
Honeycomb crush
1.5e-7 0.0
500.0 0.25
/END
"""
        m_free, _ = _parse_starter(tmp_path, deck_free)
        assert 201 in m_free.mat_law13s
        mat = m_free.mat_law13s[201]
        assert mat.rho0 == pytest.approx(1.5e-7)
        assert mat.e == pytest.approx(500.0)
        assert mat.nu == pytest.approx(0.25)

        deck_fixed = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/RIGID/202
Rigid fixed
#              RHO_I               RHO_O
            1.500e-7                 0.0
#                  E                  NU
               500.0                0.25
/END
"""
        m_fixed, _ = _parse_starter(tmp_path, deck_fixed)
        assert 202 in m_fixed.mat_law13s
        assert m_fixed.mat_law13s[202].e == pytest.approx(500.0)


class TestMatLaw15:
    def test_law15_free_and_fixed(self, tmp_path: Path):
        deck_free = """/BEGIN
/MAT/LAW15/301
Chang-Chang Composite
1.6e-6 0.0
135000.0 10000.0 0.3
4500.0 3000.0 4500.0
120.0 0.45 600.0
50.0 1.0 1
1800.0 50.0 1400.0 180.0 0.0
80.0 80.0 0.0 0.0 0
0.1 0.05 10.0 20.0 30.0
1 0.02 0.0 0.0
/END
"""
        model, _ = _parse_starter(tmp_path, deck_free)
        assert 301 in model.mat_law15s
        mat = model.mat_law15s[301]
        assert mat.rho0 == pytest.approx(1.6e-6)
        assert mat.e11 == pytest.approx(135000.0)
        assert mat.e22 == pytest.approx(10000.0)
        assert mat.nu12 == pytest.approx(0.3)
        assert mat.g12 == pytest.approx(4500.0)
        assert mat.wpmax == pytest.approx(50.0)
        assert mat.sig_1yt == pytest.approx(1800.0)
        assert mat.beta == pytest.approx(0.1)
        assert mat.fsmooth == 1

        deck_fixed = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/CHANG/302
Chang-Chang Fixed
#              RHO_I               RHO_O
            1.600e-6                 0.0
#                E11                 E22                NU12
            135000.0             10000.0                 0.3
#                G12                 G23                 G31
              4500.0              3000.0              4500.0
#                  B                   N                FMAX
               120.0                0.45               600.0
#              WPMAX               WPREF                IOFF
                50.0                 1.0                   1
#            SIG_1YT             SIG_2YT             SIG_1YC             SIG_2YC               ALPHA
              1800.0                50.0              1400.0               180.0                 0.0
#           SIG_12YC            SIG_12YT                   C           EPS_DOT_0                 ICC
                80.0                80.0                 0.0                 0.0                   0
#               BETA                TMAX                  S1                  S2                 S12
                 0.1                0.05                10.0                20.0                30.0
#            FSMOOTH                FCUT                  C1                  C2
                   1                0.02                 0.0                 0.0
/END
"""
        m_fixed, _ = _parse_starter(tmp_path, deck_fixed)
        assert 302 in m_fixed.mat_law15s
        mat_f = m_fixed.mat_law15s[302]
        assert mat_f.e11 == pytest.approx(135000.0)
        assert mat_f.fsmooth == 1
        assert mat_f.fcut == pytest.approx(0.02)


class TestMatLaw18:
    def test_law18_free_and_fixed(self, tmp_path: Path):
        deck_free = """/BEGIN
/MAT/LAW18/401
Concrete thermal damage
2.4e-6 0.0
900.0 0.001 0.002
11 293.15 1.0
12 13 1.0 1.0 1.0
/END
"""
        model, _ = _parse_starter(tmp_path, deck_free)
        assert 401 in model.mat_law18s
        mat = model.mat_law18s[401]
        assert mat.rho0 == pytest.approx(2.4e-6)
        assert mat.spheat == pytest.approx(900.0)
        assert mat.a == pytest.approx(0.001)
        assert mat.fct_idt == 11
        assert mat.t0 == pytest.approx(293.15)
        assert mat.fct_idsph == 12

        deck_fixed = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/CONCR_DRA/402
Concrete Dragon
#              RHO_I               RHO_O
            2.400e-6                 0.0
#             SPHEAT                   A                   B
               900.0               0.001               0.002
#            FCT_IDT                  T0               SCALE
                  11              293.15                 1.0
#          FCT_IDSPH           FCT_IDAS            FSCALESPH             FSCALEE             FSCALEK
                  12                  13                 1.0                 1.0                 1.0
/END
"""
        m_fixed, _ = _parse_starter(tmp_path, deck_fixed)
        assert 402 in m_fixed.mat_law18s
        assert m_fixed.mat_law18s[402].fct_idt == 11


class TestMatLaw22:
    def test_law22_free_and_fixed(self, tmp_path: Path):
        deck_free = """/BEGIN
/MAT/LAW22/501
Tsai-Wu damage
1.8e-6 0.0
70000.0 0.33
300.0 500.0 0.3 0.15 600.0
0.0 0.0 0
0.05 1000.0
/END
"""
        model, _ = _parse_starter(tmp_path, deck_free)
        assert 501 in model.mat_law22s
        mat = model.mat_law22s[501]
        assert mat.rho0 == pytest.approx(1.8e-6)
        assert mat.e == pytest.approx(70000.0)
        assert mat.sigy == pytest.approx(300.0)
        assert mat.beta == pytest.approx(500.0)
        assert mat.n == pytest.approx(0.3)
        assert mat.eps_dam == pytest.approx(0.05)
        assert mat.e_t == pytest.approx(1000.0)

        deck_fixed = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/TSAI_WU/502
Tsai-Wu fixed
#              RHO_I               RHO_O
            1.800e-6                 0.0
#                  E                  NU
             70000.0                0.33
#               SIGY                BETA                   N             EPS_MAX             SIG_MAX
               300.0               500.0                 0.3                0.15               600.0
#                  C           EPS_DOT_0                 ICC
                 0.0                 0.0                   0
#            EPS_DAM                 E_T
                0.05              1000.0
/END
"""
        m_fixed, _ = _parse_starter(tmp_path, deck_fixed)
        assert 502 in m_fixed.mat_law22s
        assert m_fixed.mat_law22s[502].sigy == pytest.approx(300.0)


class TestMatLaw25:
    def test_law25_free_and_fixed(self, tmp_path: Path):
        deck_free = """/BEGIN
/MAT/LAW25/601
Composite Plasticity
1.55e-6 0.0
140000.0 9500.0 0.32 1 9500.0
5000.0 3200.0 5000.0 0.02 0.04
0.01 0.015 0.02 0.03 0.8
40.0 1.0 0
150.0 0.5 450.0
1600.0 45.0 1300.0 160.0 0.0
70.0 70.0 0.0 0.0 0
/END
"""
        model, _ = _parse_starter(tmp_path, deck_free)
        assert 601 in model.mat_law25s
        mat = model.mat_law25s[601]
        assert mat.rho0 == pytest.approx(1.55e-6)
        assert mat.e11 == pytest.approx(140000.0)
        assert mat.e22 == pytest.approx(9500.0)
        assert mat.nu12 == pytest.approx(0.32)
        assert mat.iform == 1
        assert mat.e33 == pytest.approx(9500.0)
        assert mat.g12 == pytest.approx(5000.0)
        assert mat.dmax == pytest.approx(0.8)
        assert mat.sig_1yt == pytest.approx(1600.0)

        deck_fixed = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/COMP_PLAS/602
Composite Plas Fixed
#              RHO_I               RHO_O
            1.550e-6                 0.0
#                E11                 E22                NU12               IFORM                                 E33
            140000.0              9500.0                0.32                   1                              9500.0
#                G12                 G23                 G31              EPS_F1              EPS_F2
              5000.0              3200.0              5000.0                0.02                0.04
#             EPS_T1              EPS_M1              EPS_T2              EPS_M2                DMAX
                0.01               0.015                0.02                0.03                 0.8
#              WPMAX               WPREF                IOFF
                40.0                 1.0                   0
#                  B                   N                FMAX
               150.0                 0.5               450.0
#            SIG_1YT             SIG_2YT             SIG_1YC             SIG_2YC               ALPHA
              1600.0                45.0              1300.0               160.0                 0.0
#           SIG_12YC            SIG_12YT                   C          EPS_RATE_0                 ICC
                70.0                70.0                 0.0                 0.0                   0
/END
"""
        m_fixed, _ = _parse_starter(tmp_path, deck_fixed)
        assert 602 in m_fixed.mat_law25s
        assert m_fixed.mat_law25s[602].e33 == pytest.approx(9500.0)


class TestMatLaw28:
    def test_law28_free_and_fixed(self, tmp_path: Path):
        deck_free = """/BEGIN
/MAT/LAW28/701
Solid Honeycomb
1.2e-7 0.0
800.0 600.0 2000.0
300.0 300.0 500.0
101 102 103 0 1.0 1.0 1.0
0.6 0.6 0.7
104 105 106 0 1.0 1.0 1.0
0.5 0.5 0.5
/END
"""
        model, _ = _parse_starter(tmp_path, deck_free)
        assert 701 in model.mat_law28s
        mat = model.mat_law28s[701]
        assert mat.rho0 == pytest.approx(1.2e-7)
        assert mat.e11 == pytest.approx(800.0)
        assert mat.e22 == pytest.approx(600.0)
        assert mat.e33 == pytest.approx(2000.0)
        assert mat.fun_id11 == 101
        assert mat.fun_id22 == 102
        assert mat.fun_id33 == 103
        assert mat.eps_max11 == pytest.approx(0.6)
        assert mat.fun_id12 == 104

        deck_fixed = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/HONEYCOMB_SOL/702
Solid Honeycomb Fixed
#              RHO_I               RHO_O
            1.200e-7                 0.0
#                E11                 E22                 E33
               800.0               600.0              2000.0
#                G12                 G23                 G31
               300.0               300.0               500.0
#           FUN_ID11            FUN_ID22            FUN_ID33              IFLAG1            FSCALE11            FSCALE22            FSCALE33
                 101                 102                 103                   0                 1.0                 1.0                 1.0
#          EPS_MAX11           EPS_MAX22           EPS_MAX33
                 0.6                 0.6                 0.7
#           FUN_ID12            FUN_ID23            FUN_ID31              IFLAG2            FSCALE12            FSCALE23            FSCALE31
                 104                 105                 106                   0                 1.0                 1.0                 1.0
#          EPS_MAX12           EPS_MAX23           EPS_MAX31
                 0.5                 0.5                 0.5
/END
"""
        m_fixed, _ = _parse_starter(tmp_path, deck_fixed)
        assert 702 in m_fixed.mat_law28s
        assert m_fixed.mat_law28s[702].fun_id33 == 103


class TestPropType9:
    def test_prop9_free_and_fixed(self, tmp_path: Path):
        deck_free = """/BEGIN
/PROP/TYPE9/901
Orthotropic Shell
1 1 1 0
0.1 0.1 0.1 0.0 0.0
5 1 2.5 0.833333 0 0
1.0 0.0 0.0 45.0
/END
"""
        model, _ = _parse_starter(tmp_path, deck_free)
        assert 901 in model.prop_type9s
        prop = model.prop_type9s[901]
        assert prop.ishell == 1
        assert prop.thick == pytest.approx(2.5)
        assert prop.nip == 5
        assert prop.phi == pytest.approx(45.0)
        assert 901 in model.properties

        deck_fixed = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/PROP/SH_ORTH/902
Orthotropic Shell Fixed
#             ISHELL              ISMSTR               ISH3N              IDRILL
                   1                   1                   1                   0
#                 HM                  HF                  HR                  DM                  DN
                0.10                0.10                0.10                0.00                0.00
#                NIP             ISTRAIN               THICK              ASHEAR                                  ITHICK               IPLAS
                   5                   1                 2.5            0.833333                                       0                   0
#                 VX                  VY                  VZ                 PHI
                 1.0                 0.0                 0.0                45.0
/END
"""
        m_fixed, _ = _parse_starter(tmp_path, deck_fixed)
        assert 902 in m_fixed.prop_type9s
        assert m_fixed.prop_type9s[902].phi == pytest.approx(45.0)


class TestPropType10:
    def test_prop10_free_and_fixed(self, tmp_path: Path):
        deck_free = """/BEGIN
/PROP/TYPE10/1001
Composite Shell
1 1 1 0
0.1 0.1 0.1 0.0 0.0
4 1 2.0 0.833333 0 0
1.0 0.0 0.0
0.0 45.0 -45.0 90.0
/END
"""
        model, _ = _parse_starter(tmp_path, deck_free)
        assert 1001 in model.prop_type10s
        prop = model.prop_type10s[1001]
        assert prop.nip == 4
        assert prop.thick == pytest.approx(2.0)
        assert len(prop.phis) == 4
        assert prop.phis == [pytest.approx(0.0), pytest.approx(45.0), pytest.approx(-45.0), pytest.approx(90.0)]

        deck_fixed = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/PROP/SH_COMP/1002
Composite Shell Fixed
#             ISHELL              ISMSTR               ISH3N              IDRILL
                   1                   1                   1                   0
#                 HM                  HF                  HR                  DM                  DN
                0.10                0.10                0.10                0.00                0.00
#                NIP             ISTRAIN               THICK              ASHEAR                                  ITHICK               IPLAS
                   4                   1                 2.0            0.833333                                       0                   0
#                 VX                  VY                  VZ
                 1.0                 0.0                 0.0
#                PHI
                 0.0                45.0               -45.0                90.0
/END
"""
        m_fixed, _ = _parse_starter(tmp_path, deck_fixed)
        assert 1002 in m_fixed.prop_type10s
        assert m_fixed.prop_type10s[1002].phis == [pytest.approx(0.0), pytest.approx(45.0), pytest.approx(-45.0), pytest.approx(90.0)]


class TestPropType51:
    def test_prop51_free_and_fixed(self, tmp_path: Path):
        deck_free = """/BEGIN
/PROP/TYPE51/5101
Cohesive Shell
1 1 0 0 0.2 0.1
0.05 0.05 0.05 0.0 0.0
0.833333 0 0 1.5
1.0 0.0 0.0 1 1 0 0
/END
"""
        model, _ = _parse_starter(tmp_path, deck_free)
        assert 5101 in model.prop_type51s
        prop = model.prop_type51s[5101]
        assert prop.pthk == pytest.approx(0.2)
        assert prop.zshift == pytest.approx(0.1)
        assert prop.failexp == pytest.approx(1.5)
        assert prop.idsk == 1
        assert prop.iorth == 1

        deck_fixed = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/PROP/SH_COH/5102
Cohesive Shell Fixed
#             ISHELL              ISMSTR               ISH3N              IDRILL                PTHK              ZSHIFT
                   1                   1                   0                   0                 0.2                 0.1
#                 HM                  HF                  HR                  DM                  DN
                0.05                0.05                0.05                0.00                0.00
#             ASHEAR                IINT              ITHICK             FAILEXP
            0.833333                   0                   0                 1.5
#                 VX                  VY                  VZ                IDSK               IORTH                IPOS                 IRP
                 1.0                 0.0                 0.0                   1                   1                   0                   0
/END
"""
        m_fixed, _ = _parse_starter(tmp_path, deck_fixed)
        assert 5102 in m_fixed.prop_type51s
        assert m_fixed.prop_type51s[5102].pthk == pytest.approx(0.2)
        assert m_fixed.prop_type51s[5102].failexp == pytest.approx(1.5)


class TestPropType5:
    def test_prop5_free_and_fixed(self, tmp_path: Path):
        deck_free = """/BEGIN
/PROP/TYPE5/501
Rivet connection
5000.0 3000.0 10.0 1 2
/END
"""
        model, _ = _parse_starter(tmp_path, deck_free)
        assert 501 in model.prop_type5s
        prop = model.prop_type5s[501]
        assert prop.nforce == pytest.approx(5000.0)
        assert prop.tforce == pytest.approx(3000.0)
        assert prop.length == pytest.approx(10.0)
        assert prop.wflag == 1
        assert prop.imod == 2

        deck_fixed = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/PROP/RIVET/502
Rivet Fixed
#             NFORCE              TFORCE              LENGTH               WFLAG                IMOD
              5000.0              3000.0                10.0                   1                   2
/END
"""
        m_fixed, _ = _parse_starter(tmp_path, deck_fixed)
        assert 502 in m_fixed.prop_type5s
        assert m_fixed.prop_type5s[502].nforce == pytest.approx(5000.0)


class TestPropType6:
    def test_prop6_free_and_fixed(self, tmp_path: Path):
        deck_free = """/BEGIN
/PROP/TYPE6/601
Solid Orthotropic
1 1 0 0 0 0.05
1.2 0.06 0.12
1.0 0.0 0.0 10 1 1 0.0 1e-7
/END
"""
        model, _ = _parse_starter(tmp_path, deck_free)
        assert 601 in model.prop_type6s
        prop = model.prop_type6s[601]
        assert prop.isolid == 1
        assert prop.dn == pytest.approx(0.05)
        assert prop.qa == pytest.approx(1.2)
        assert prop.skew_csid == 10
        assert prop.refplane == 1
        assert prop.orthtrop == 1

        deck_fixed = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/PROP/SOL_ORTH/602
Solid Orthotropic Fixed
#             ISOLID              ISMSTR               ICPRE                 NBP              IFRAME                  DN
                   1                   1                   0                   0                   0                0.05
#                 QA                  QB                   H
                 1.2                0.06                0.12
#                 VX                  VY                  VZ           SKEW_CSID            REFPLANE            ORTHTROP            MAT_BETA          DELTAT_MIN
                 1.0                 0.0                 0.0                  10                   1                   1                 0.0                1e-7
/END
"""
        m_fixed, _ = _parse_starter(tmp_path, deck_fixed)
        assert 602 in m_fixed.prop_type6s
        assert m_fixed.prop_type6s[602].skew_csid == 10
        assert m_fixed.prop_type6s[602].refplane == 1


class TestPropType20:
    def test_prop20_free_and_fixed(self, tmp_path: Path):
        deck_free = """/BEGIN
/PROP/TYPE20/2001
Thick Shell
15 1 0 0 0 2 0.1
1.1 0.05 0.1
1000.0
/END
"""
        model, _ = _parse_starter(tmp_path, deck_free)
        assert 2001 in model.prop_type20s
        prop = model.prop_type20s[2001]
        assert prop.isolid == 15
        assert prop.ismstr == 1
        assert prop.iint == 2
        assert prop.dn == pytest.approx(0.1)
        assert prop.qa == pytest.approx(1.1)
        assert prop.deltat_min == pytest.approx(1000.0)

        deck_fixed = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/PROP/TSHELL/2002
Thick Shell Fixed
#             ISOLID              ISMSTR                                   ICPRE               ICSTR                 NBP                IINT                  DN
                  15                   1                                       0                   0                   0                   2                0.10
#                 QA                  QB                   H
                 1.1                0.05                 0.1
#         DELTAT_MIN
              1000.0
/END
"""
        m_fixed, _ = _parse_starter(tmp_path, deck_fixed)
        assert 2002 in m_fixed.prop_type20s
        assert m_fixed.prop_type20s[2002].iint == 2
        assert m_fixed.prop_type20s[2002].deltat_min == pytest.approx(1000.0)

"""
Milestone M184 Test Suite:
Steinberg-Guinan Plasticity, SAMP Plasticity, Sandwich Shell,
Fabric Shell, Composite Stack & Crushing Spring Suite:
- /MAT/LAW49, /MAT/STEINB, /MAT/STEINBERG, /MAT/STEINBERG_GUINAN
- /MAT/LAW76, /MAT/SAMP, /MAT/PLAS_SAMP, /MAT/SAMP_PLAS
- /PROP/TYPE11, /PROP/SH_SANDW, /PROP/SANDWICH
- /PROP/TYPE16, /PROP/SH_FABR, /PROP/FABRIC_SHELL, /PROP/FABRIC
- /PROP/TYPE17, /PROP/STACK, /PROP/COMP_STACK
- /PROP/TYPE44, /PROP/SPR_CRUS, /PROP/CRUSH_SPRING, /PROP/SPRING_CRUSH
"""

from pathlib import Path
import pytest
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog


def _parse_deck(tmp_path: Path, text: str, name: str = "TEST_0000.rad") -> tuple[Model, MessageLog]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    deck_path = tmp_path / name
    deck_path.write_text(text, encoding="ascii")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(deck_path))
    parse_starter_deck(blocks, model, log)
    return model, log


def test_mat_law49_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW49 & /MAT/STEINB in fixed and free formats."""
    deck_fixed = """\
#RADIOSS STARTER
/BEGIN
test_law49_fixed
                  10                   1
/MAT/LAW49/1
Steinberg Guinan Material
#              RHO_I               RHO_O
              8.9E-6              8.9E-6
#                 E0                  nu
             12000.0                0.34
#            sigma_0                beta                   n             EPS_max           SIGMA_max
               120.0                36.0                0.45                 0.8               500.0
#                T_0               Tmelt              rhoC_p                Pmin
               300.0              1790.0              3.8E-4               -50.0
#                 b1                  b2                   h                   f
                0.04                0.01              3.8E-4                0.02
/END
"""
    model, log = _parse_deck(tmp_path, deck_fixed, "fixed.rad")

    assert 1 in model.mat_law49s
    m1 = model.mat_law49s[1]
    assert m1.rho == pytest.approx(8.9e-6)
    assert m1.rho0 == pytest.approx(8.9e-6)
    assert m1.refer_rho == pytest.approx(8.9e-6)
    assert m1.e0 == pytest.approx(12000.0)
    assert m1.e == pytest.approx(12000.0)
    assert m1.nu == pytest.approx(0.34)
    assert m1.sigy == pytest.approx(120.0)
    assert m1.sigma_0 == pytest.approx(120.0)
    assert m1.beta == pytest.approx(36.0)
    assert m1.n == pytest.approx(0.45)
    assert m1.hard == pytest.approx(0.45)
    assert m1.eps_max == pytest.approx(0.8)
    assert m1.sigma_max == pytest.approx(500.0)
    assert m1.t0 == pytest.approx(300.0)
    assert m1.tmelt == pytest.approx(1790.0)
    assert m1.rhoc_p == pytest.approx(3.8e-4)
    assert m1.pmin == pytest.approx(-50.0)
    assert m1.b1 == pytest.approx(0.04)
    assert m1.b2 == pytest.approx(0.01)
    assert m1.h == pytest.approx(3.8e-4)
    assert m1.f == pytest.approx(0.02)
    assert 1 in model.materials

    deck_free = """\
#RADIOSS STARTER
/BEGIN
test_law49_free
10 1
/MAT/STEINB/2
Steinberg Free Format
8.9e-6 8.9e-6
12000.0 0.34
120.0 36.0 0.45 0.8 500.0
300.0 1790.0 3.8e-4 -50.0
0.04 0.01 3.8e-4 0.02
/END
"""
    model2, log2 = _parse_deck(tmp_path, deck_free, "free.rad")
    assert 2 in model2.mat_law49s
    m2 = model2.mat_law49s[2]
    assert m2.rho == pytest.approx(8.9e-6)
    assert m2.sigy == pytest.approx(120.0)
    assert m2.tmelt == pytest.approx(1790.0)
    assert m2.b1 == pytest.approx(0.04)


def test_mat_law76_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW76 & /MAT/SAMP in fixed and free formats."""
    deck_fixed = """\
#RADIOSS STARTER
/BEGIN
test_law76_fixed
                  10                   1
/MAT/LAW76/1
SAMP Plastic Material
#              RHO_I               RHO_O
              1.1E-6              1.1E-6
#                  E                  nu
              2500.0                0.38
#     FUN_D1    FUN_D2    FUN_D3    FUN_D4            FScale11            FScale22            FScale33
         101       102       103       104                 1.0                 1.0                 1.0
#           FScale12                FACX              MAT_NU             FUN_B5          MAT_PScale    ISRATE            asrate
                 1.0                 1.0                0.35                 105                 1.0         1              50.0
#       MAT_Eps_Fail               Eps_0              MAT_Dc              FUN_A1              FUN_A2              FUN_A3               SCALE
                 0.6                 0.7                 0.8                 106                 107                 108                 1.0
#      IFORM     MAT_Iflg      Gflag
           1            0          1
/END
"""
    model, log = _parse_deck(tmp_path, deck_fixed, "fixed.rad")

    assert 1 in model.mat_law76s
    m1 = model.mat_law76s[1]
    assert m1.rho == pytest.approx(1.1e-6)
    assert m1.rho0 == pytest.approx(1.1e-6)
    assert m1.e == pytest.approx(2500.0)
    assert m1.nu == pytest.approx(0.38)
    assert m1.fun_d1 == 101
    assert m1.fun_d2 == 102
    assert m1.fun_d3 == 103
    assert m1.fun_d4 == 104
    assert m1.fscale11 == pytest.approx(1.0)
    assert m1.fscale12 == pytest.approx(1.0)
    assert m1.facx == pytest.approx(1.0)
    assert m1.mat_nut == pytest.approx(0.35)
    assert m1.nu_p == pytest.approx(0.35)
    assert m1.fun_b5 == 105
    assert m1.israte == 1
    assert m1.asrate == pytest.approx(50.0)
    assert m1.epsilon_f == pytest.approx(0.6)
    assert m1.epsilon_0 == pytest.approx(0.7)
    assert m1.dc == pytest.approx(0.8)
    assert m1.fun_a1 == 106
    assert m1.fun_a2 == 107
    assert m1.fun_a3 == 108
    assert m1.scale == pytest.approx(1.0)
    assert m1.iform == 1
    assert m1.iflag == 0
    assert m1.gflag == 1
    assert 1 in model.materials

    deck_free = """\
#RADIOSS STARTER
/BEGIN
test_law76_free
10 1
/MAT/SAMP/2
SAMP Free Format
1.1e-6 1.1e-6
2500.0 0.38
101 102 103 104 1.0 1.0 1.0
1.0 1.0 0.35 105 1.0 1 50.0
0.6 0.7 0.8 106 107 108 1.0
1 0 1
/END
"""
    model2, log2 = _parse_deck(tmp_path, deck_free, "free.rad")
    assert 2 in model2.mat_law76s
    m2 = model2.mat_law76s[2]
    assert m2.rho == pytest.approx(1.1e-6)
    assert m2.fun_d1 == 101
    assert m2.epsilon_f == pytest.approx(0.6)
    assert m2.iform == 1


def test_prop_type11_fixed_and_free(tmp_path: Path):
    """Test /PROP/TYPE11 & /PROP/SH_SANDW in fixed and free formats."""
    deck_fixed = """\
#RADIOSS STARTER
/BEGIN
test_type11_fixed
                  10                   1
/PROP/TYPE11/1
Sandwich Shell Property
#   Ishell    Ismstr     Ish3n    Idrill                            P_Thick_Fail
        24         2         1         1                                     0.5
#                 Hm                  Hf                  Hr                  Dm                  Dn
                0.01                0.01                0.01                 0.0                0.01
#        N   Istrain               Thick              Ashear              Ithick     Iplas
         3         1                 3.0                0.83                   1         1
#                 Vx                  Vy                  Vz     Iskew     Iorth      Ipos        Ip
                 1.0                 0.0                 0.0         0         1         0         0
#                Phi               Thick                   Z         m                      F_weight
                 0.0                 1.0                -1.0         1                           1.0
                45.0                 1.0                 0.0         2                           1.0
                90.0                 1.0                 1.0         1                           1.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_fixed, "fixed.rad")

    assert 1 in model.prop_type11s
    p1 = model.prop_type11s[1]
    assert p1.ishell == 24
    assert p1.ismstr == 2
    assert p1.ish3n == 1
    assert p1.idrill == 1
    assert p1.p_thick_fail == pytest.approx(0.5)
    assert p1.hm == pytest.approx(0.01)
    assert p1.hf == pytest.approx(0.01)
    assert p1.hr == pytest.approx(0.01)
    assert p1.dn == pytest.approx(0.01)
    assert p1.nip == 3
    assert p1.istrain == 1
    assert p1.thick == pytest.approx(3.0)
    assert p1.ashear == pytest.approx(0.83)
    assert p1.ithick == 1
    assert p1.iplas == 1
    assert p1.vx == pytest.approx(1.0)
    assert p1.iorth == 1
    assert len(p1.layers) == 3
    assert p1.layers[0].phi == pytest.approx(0.0)
    assert p1.layers[0].thick == pytest.approx(1.0)
    assert p1.layers[0].z == pytest.approx(-1.0)
    assert p1.layers[0].mat_id == 1
    assert p1.layers[1].phi == pytest.approx(45.0)
    assert p1.layers[1].mat_id == 2
    assert 1 in model.properties

    deck_free = """\
#RADIOSS STARTER
/BEGIN
test_type11_free
10 1
/PROP/SH_SANDW/2
Sandwich Free Format
24 2 1 1 0.5
0.01 0.01 0.01 0.0 0.01
3 1 3.0 0.83 1 1
1.0 0.0 0.0 0 1 0 0
0.0 1.0 -1.0 1 1.0
45.0 1.0 0.0 2 1.0
90.0 1.0 1.0 1 1.0
/END
"""
    model2, log2 = _parse_deck(tmp_path, deck_free, "free.rad")
    assert 2 in model2.prop_type11s
    p2 = model2.prop_type11s[2]
    assert p2.ishell == 24
    assert p2.thick == pytest.approx(3.0)
    assert len(p2.layers) == 3
    assert p2.layers[2].phi == pytest.approx(90.0)


def test_prop_type16_fixed_and_free(tmp_path: Path):
    """Test /PROP/TYPE16 & /PROP/SH_FABR in fixed and free formats."""
    deck_fixed = """\
#RADIOSS STARTER
/BEGIN
test_type16_fixed
                  10                   1
/PROP/TYPE16/1
Fabric Shell Property
#   Ishell    Ismstr     Ish3n                                      P_Thick_Fail
        24         2         1                                               0.4
#                 Hm                  Hf                  Hr                  Dm                  Dn
                0.01                0.01                0.01                 0.0                0.01
#        N   Istrain               Thick              Ashear              Ithick
         2         1                 2.0                0.83                   1
#                 Vx                  Vy                  Vz   Skew_ID      Ipos                  Ip
                 1.0                 0.0                 0.0         0         0                   0
#              Phi_i             Alpha_i                 T_i                 Z_i   mat_IDi
                 0.0                45.0                 1.0                -0.5         1
                90.0                45.0                 1.0                 0.5         1
/END
"""
    model, log = _parse_deck(tmp_path, deck_fixed, "fixed.rad")

    assert 1 in model.prop_type16s
    p1 = model.prop_type16s[1]
    assert p1.ishell == 24
    assert p1.ismstr == 2
    assert p1.ish3n == 1
    assert p1.p_thick_fail == pytest.approx(0.4)
    assert p1.hm == pytest.approx(0.01)
    assert p1.dn == pytest.approx(0.01)
    assert p1.nip == 2
    assert p1.istrain == 1
    assert p1.thick == pytest.approx(2.0)
    assert p1.ashear == pytest.approx(0.83)
    assert p1.ithick == 1
    assert p1.vx == pytest.approx(1.0)
    assert len(p1.layers) == 2
    assert p1.layers[0].phi == pytest.approx(0.0)
    assert p1.layers[0].alpha == pytest.approx(45.0)
    assert p1.layers[0].thick == pytest.approx(1.0)
    assert p1.layers[0].z == pytest.approx(-0.5)
    assert p1.layers[0].mat_id == 1
    assert p1.layers[1].phi == pytest.approx(90.0)
    assert 1 in model.properties

    deck_free = """\
#RADIOSS STARTER
/BEGIN
test_type16_free
10 1
/PROP/SH_FABR/2
Fabric Free Format
24 2 1 0.4
0.01 0.01 0.01 0.0 0.01
2 1 2.0 0.83 1
1.0 0.0 0.0 0 0 0
0.0 45.0 1.0 -0.5 1
90.0 45.0 1.0 0.5 1
/END
"""
    model2, log2 = _parse_deck(tmp_path, deck_free, "free.rad")
    assert 2 in model2.prop_type16s
    p2 = model2.prop_type16s[2]
    assert p2.ishell == 24
    assert p2.thick == pytest.approx(2.0)
    assert len(p2.layers) == 2
    assert p2.layers[1].phi == pytest.approx(90.0)
    assert p2.layers[1].alpha == pytest.approx(45.0)


def test_prop_type17_fixed_and_free(tmp_path: Path):
    """Test /PROP/TYPE17 & /PROP/STACK in fixed and free formats."""
    deck_fixed = """\
#RADIOSS STARTER
/BEGIN
test_type17_fixed
                  10                   1
/PROP/TYPE17/1
Composite Stack Property
#   Ishell    Ismstr     Ish3n    Idrill   plyxfem                            Z0           Vinterply
        24         2         1         1         0                           0.0                 0.1
#                 Hm                  Hf                  Hr                  Dm                  Dn
                0.01                0.01                0.01                 0.0                0.01
#                                  Thick              Ashear              Ithick     Iplas
                                     2.5                0.83                   1         1
#                 Vx                  Vy                  Vz   skew_ID     Iorth      Ipos        Ip
                 1.0                 0.0                 0.0         0         1         0         0
/END
"""
    model, log = _parse_deck(tmp_path, deck_fixed, "fixed.rad")

    assert 1 in model.prop_type17s
    p1 = model.prop_type17s[1]
    assert p1.ishell == 24
    assert p1.ismstr == 2
    assert p1.ish3n == 1
    assert p1.idrill == 1
    assert p1.plyxfem == 0
    assert p1.z0 == pytest.approx(0.0)
    assert p1.vinterply == pytest.approx(0.1)
    assert p1.hm == pytest.approx(0.01)
    assert p1.dn == pytest.approx(0.01)
    assert p1.thick == pytest.approx(2.5)
    assert p1.ashear == pytest.approx(0.83)
    assert p1.ithick == 1
    assert p1.iplas == 1
    assert p1.vx == pytest.approx(1.0)
    assert p1.iorth == 1
    assert 1 in model.properties

    deck_free = """\
#RADIOSS STARTER
/BEGIN
test_type17_free
10 1
/PROP/STACK/2
Stack Free Format
24 2 1 1 0 0.0 0.1
0.01 0.01 0.01 0.0 0.01
2.5 0.83 1 1
1.0 0.0 0.0 0 1 0 0
/END
"""
    model2, log2 = _parse_deck(tmp_path, deck_free, "free.rad")
    assert 2 in model2.prop_type17s
    p2 = model2.prop_type17s[2]
    assert p2.ishell == 24
    assert p2.thick == pytest.approx(2.5)
    assert p2.vinterply == pytest.approx(0.1)


def test_prop_type44_fixed_and_free(tmp_path: Path):
    """Test /PROP/TYPE44 & /PROP/SPR_CRUS in fixed and free formats."""
    deck_fixed = """\
#RADIOSS STARTER
/BEGIN
test_type44_fixed
                  10                   1
/PROP/TYPE44/1
Crushing Spring Property
#             MASS/L           INERTIA/L              Kinter   Skew_Id Icoupling    Ifiltr
                 0.1               0.001              1000.0         0         1         0
#               K11L                K44L                K55L                K66L     Idamp
              2000.0               500.0               400.0               400.0         1
#               K5bL                K6cL
               300.0               300.0
#  Fct_X+i   Fct_X-i   Fct_X-r                     F_scaleXY
       101       102       103                           1.0
# fct_XX+i  fct_XX-i  fct_XX+r  fct_XX-r          Fscale_XXY
       201       202       203       204                 1.0
#fct_YY+i fct_YY1-i fct_YY1+r fct_YY1-r         Fscale_YY1Y
       301       302       303       304                 1.0
#fct_ZZ1+i fct_ZZ1-i fct_ZZ1+r fct_ZZ1-r         Fscale_ZZ1Y
       401       402       403       404                 1.0
#fct_YY2+i fct_YY2-i fct_YY2+r fct_YY2-r         Fscale_YY2Y
       501       502       503       504                 1.0
#fct_ZZ2+i fct_ZZ2-i fct_ZZ2+r fct_ZZ2-r         Fscale_ZZ2Y
       601       602       603       604                 1.0
#            X_lim_g               X_lim              XX_lim
                0.25                 0.2                0.15
#            YY1_lim             ZZ1_lim             YY2_lim             ZZ2_lim
                 0.1                 0.1                 0.1                 0.1
#  fct_D_x            Dscale_x                 F_x
       701                 1.5                 0.5
#  fct_D_y            Dscale_y                 F_y
       702                 1.5                 0.5
#  fct_D_z            Dscale_z                 F_z
       703                 1.5                 0.5
# fct_D_xx           Dscale_xx                F_xx
       704                 1.5                 0.5
# fct_D_yy           Dscale_yy                F_yy
       705                 1.5                 0.5
# fct_D_zz           Dscale_zz                F_zz
       706                 1.5                 0.5
/END
"""
    model, log = _parse_deck(tmp_path, deck_fixed, "fixed.rad")

    assert 1 in model.prop_type44s
    p1 = model.prop_type44s[1]
    assert p1.mass == pytest.approx(0.1)
    assert p1.inertia == pytest.approx(0.001)
    assert p1.stiff1 == pytest.approx(1000.0)
    assert p1.icoupling == 1
    assert p1.k11 == pytest.approx(2000.0)
    assert p1.k44 == pytest.approx(500.0)
    assert p1.k55 == pytest.approx(400.0)
    assert p1.k66 == pytest.approx(400.0)
    assert p1.idamp == 1
    assert p1.k5b == pytest.approx(300.0)
    assert p1.k6c == pytest.approx(300.0)
    assert p1.fun_a1 == 101
    assert p1.fun_b1 == 102
    assert p1.fun_a2 == 103
    assert p1.fscale11 == pytest.approx(1.0)
    assert p1.fun_b2 == 201
    assert p1.fun_b4 == 301
    assert p1.fun_b6 == 401
    assert p1.fun_c4 == 501
    assert p1.fun_d2 == 601
    assert p1.strain1 == pytest.approx(0.25)
    assert p1.strain2 == pytest.approx(0.2)
    assert p1.strain3 == pytest.approx(0.15)
    assert p1.strain4 == pytest.approx(0.1)
    assert p1.fct_d_x == 701
    assert p1.dscale_x == pytest.approx(1.5)
    assert p1.f_x == pytest.approx(0.5)
    assert p1.fct_d_zz == 706
    assert 1 in model.properties

    deck_free = """\
#RADIOSS STARTER
/BEGIN
test_type44_free
10 1
/PROP/SPR_CRUS/2
Crushing Spring Free Format
0.1 0.001 1000.0 0 1 0
2000.0 500.0 400.0 400.0 1
300.0 300.0
101 102 103 1.0
201 202 203 204 1.0
301 302 303 304 1.0
401 402 403 404 1.0
501 502 503 504 1.0
601 602 603 604 1.0
0.25 0.2 0.15
0.1 0.1 0.1 0.1
701 1.5 0.5
702 1.5 0.5
703 1.5 0.5
704 1.5 0.5
705 1.5 0.5
706 1.5 0.5
/END
"""
    model2, log2 = _parse_deck(tmp_path, deck_free, "free.rad")
    assert 2 in model2.prop_type44s
    p2 = model2.prop_type44s[2]
    assert p2.mass == pytest.approx(0.1)
    assert p2.k11 == pytest.approx(2000.0)
    assert p2.fun_a1 == 101
    assert p2.fct_d_x == 701


def test_m184_error_guards(tmp_path: Path):
    """Test error guards for missing cards."""
    deck_empty = """\
#RADIOSS STARTER
/BEGIN
empty
10 1
/MAT/LAW49/1
/MAT/LAW76/2
/PROP/TYPE11/3
/PROP/TYPE16/4
/PROP/TYPE17/5
/PROP/TYPE44/6
/END
"""
    model, log = _parse_deck(tmp_path, deck_empty, "empty.rad")
    assert 1 not in model.mat_law49s
    assert 2 not in model.mat_law76s
    assert 3 not in model.prop_type11s
    assert 4 not in model.prop_type16s
    assert 5 not in model.prop_type17s
    assert 6 not in model.prop_type44s
    assert len(log.errors) >= 6
